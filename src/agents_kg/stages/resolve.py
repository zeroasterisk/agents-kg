"""Stage 5b: Entity resolution — merge duplicates and link cross-type relatives.

Runs after extract, before review. Three passes:
1. Exact match: merge entities with identical entity_id (already handled by DB UNIQUE)
2. Fuzzy match: merge entities with similar names + same type
3. Cross-type linking: connect related entities across types (e.g., protocol:mcp ↔ project:mcp-sdk)

Based on Graphiti research: entropy-gated fuzzy matching, two-pass dedup.
"""

import logging
import re
import struct
from collections import defaultdict
from difflib import SequenceMatcher
from ..db import Database
from ..seed import get_seed_entities

try:
    from google import genai
except ImportError:
    genai = None

EMBEDDING_MODEL = "gemini-embedding-2"

def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f'{len(floats)}f', *floats)

def _bytes_to_floats(b: bytes) -> list[float]:
    return list(struct.unpack(f'{len(b)//4}f', b))

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)

try:
    from prefect.logging import get_run_logger as _get_logger
except ImportError:
    _get_logger = None


def _log():
    if _get_logger:
        try:
            return _get_logger()
        except Exception:
            pass
    return logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def _similarity(a: str, b: str) -> float:
    """String similarity ratio (0-1)."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _build_alias_index(entities: list[dict]) -> dict[str, list[dict]]:
    """Build a lookup: normalized name/alias → list of entities."""
    index = defaultdict(list)
    for e in entities:
        index[_normalize(e["name"])].append(e)
        aliases = e.get("aliases", "[]")
        if isinstance(aliases, str):
            import json
            try:
                aliases = json.loads(aliases)
            except (json.JSONDecodeError, TypeError):
                aliases = []
        for alias in aliases:
            index[_normalize(alias)].append(e)
    return dict(index)


def _compute_entity_embeddings(db: Database, entities: list[dict], log):
    """Compute embeddings for entities that don't have them."""
    if not genai:
        log.warning("google-genai not installed, skipping entity embeddings")
        return

    import os
    kwargs = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["enterprise"] = True
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    client = genai.Client(**kwargs)

    to_embed = [e for e in entities if not e.get("embedding")]
    if not to_embed:
        return

    log.info("Computing embeddings for %d entities", len(to_embed))
    try:
        for ent in to_embed:
            text = ent["description"] or ent["name"]
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            emb_bytes = _floats_to_bytes(result.embeddings[0].values)
            db.conn.execute(
                "UPDATE entities SET embedding = ? WHERE id = ?",
                (emb_bytes, ent["id"])
            )
            ent["embedding"] = emb_bytes
        db.conn.commit()
    except Exception as e:
        log.error("Failed to compute entity embeddings: %s", e)


def run(db: Database, source: dict) -> bool:
    """Run entity resolution on a source's extracted entities.

    Returns True if any merges were performed.
    """
    source_id = source["id"]
    log = _log()

    # Get all entities for this source
    entities = db.conn.execute(
        "SELECT * FROM entities WHERE source_id = ?", (source_id,)
    ).fetchall()
    entities = [dict(e) for e in entities]
    
    if not entities:
        log.info("No entities to resolve for source %d", source_id)
        db.update_source(source_id, stage="review", status="pending_review")
        return True

    # Compute embeddings for entities
    _compute_entity_embeddings(db, entities, log)

    # Build seed index for canonical matching
    seed = get_seed_entities()
    seed_index = _build_alias_index(seed)

    merges = 0
    skips = 0

    # --- Pass 1: Match against seed entities ---
    for entity in entities:
        norm_name = _normalize(entity["name"])

        # Direct match in seed index
        canonical = seed_index.get(norm_name)
        if canonical and len(canonical) == 1:
            seed_ent = canonical[0]
            if entity["entity_id"] != seed_ent["entity_id"] and entity["type"] == seed_ent["type"]:
                log.info("Merging %s → %s (seed match: %s)",
                        entity["entity_id"], seed_ent["entity_id"], entity["name"])
                _merge_entity(db, entity, seed_ent["entity_id"])
                merges += 1
                continue

        # Fuzzy match against seed
        best_match = None
        best_score = 0.0
        for seed_ent in seed:
            if seed_ent["type"] != entity["type"]:
                continue
            score = _similarity(entity["name"], seed_ent["name"])
            if score > best_score:
                best_score = score
                best_match = seed_ent

            # Also check aliases
            for alias in seed_ent.get("aliases", []):
                score = _similarity(entity["name"], alias)
                if score > best_score:
                    best_score = score
                    best_match = seed_ent

        if best_match and best_score >= 0.85 and entity["entity_id"] != best_match["entity_id"]:
            log.info("Merging %s → %s (fuzzy %.2f: %s ≈ %s)",
                    entity["entity_id"], best_match["entity_id"],
                    best_score, entity["name"], best_match["name"])
            _merge_entity(db, entity, best_match["entity_id"])
            merges += 1

    # --- Pass 2: Same-source dedup (same type, very similar names) ---
    # Re-fetch after seed merges
    entities = db.conn.execute(
        "SELECT * FROM entities WHERE source_id = ? AND merged_into IS NULL", (source_id,)
    ).fetchall()
    entities = [dict(e) for e in entities]

    by_type = defaultdict(list)
    for e in entities:
        by_type[e["type"]].append(e)

    for etype, group in by_type.items():
        seen = {}  # normalized name → canonical entity_id
        for entity in sorted(group, key=lambda e: e["id"]):
            norm = _normalize(entity["name"])
            if norm in seen and seen[norm] != entity["entity_id"]:
                log.info("Dedup %s → %s (exact name match within source)",
                        entity["entity_id"], seen[norm])
                _merge_entity(db, entity, seen[norm])
                merges += 1
            else:
                seen[norm] = entity["entity_id"]

    # --- Pass 3: Filter out noise entities ---
    noise_kinds = {"headphones", "attack", "discipline", "book", "whitepaper", "benchmark"}
    noise_entities = db.conn.execute(
        "SELECT * FROM entities WHERE source_id = ? AND merged_into IS NULL AND kind IN ({})".format(
            ",".join("?" * len(noise_kinds))
        ), (source_id, *noise_kinds)
    ).fetchall()

    for ent in noise_entities:
        ent = dict(ent)
        log.info("Flagging noise entity: %s (%s/%s)", ent["entity_id"], ent["type"], ent["kind"])
        db.update_entity(ent["id"], status="rejected", merged_into="noise")
        merges += 1

    # --- Pass 4: Vector similarity resolution ---
    all_entities = db.conn.execute(
        "SELECT * FROM entities WHERE status = 'approved' AND merged_into IS NULL"
    ).fetchall()
    all_entities = [dict(e) for e in all_entities]
    
    for entity in entities:
        if entity.get("merged_into") or entity["status"] == "rejected":
            continue
        if not entity.get("embedding"):
            continue
            
        ent_emb = _bytes_to_floats(entity["embedding"])
        
        for other in all_entities:
            if other["entity_id"] == entity["entity_id"]:
                continue
            if other["type"] != entity["type"]:
                continue
            if not other.get("embedding"):
                continue
                
            other_emb = _bytes_to_floats(other["embedding"])
            sim = _cosine_similarity(ent_emb, other_emb)
            
            if sim > 0.92:
                log.info("Vector merge %s → %s (sim: %.3f, %s ≈ %s)",
                         entity["entity_id"], other["entity_id"], sim, entity["name"], other["name"])
                _merge_entity(db, entity, other["entity_id"])
                merges += 1
                break

    log.info("Resolution complete for source %d: %d merges/rejections", source_id, merges)
    db.update_source(source_id, stage="review", status="pending_review")
    return True


def _merge_entity(db: Database, entity: dict, canonical_id: str):
    """Mark an entity as merged into a canonical entity, and repoint its edges."""
    entity_id = entity["entity_id"]
    db_id = entity["id"]

    # Mark as merged
    db.update_entity(db_id, status="merged", merged_into=canonical_id)

    # Ensure the canonical entity exists in DB (it might be a seed entity not yet stored)
    existing = db.conn.execute(
        "SELECT id FROM entities WHERE entity_id = ?", (canonical_id,)
    ).fetchone()

    if not existing:
        # Import the seed entity
        from ..seed import get_seed_entities
        for seed_ent in get_seed_entities():
            if seed_ent["entity_id"] == canonical_id:
                db.add_entity(
                    entity_id=canonical_id,
                    name=seed_ent["name"],
                    entity_type=seed_ent["type"],
                    kind=seed_ent.get("kind"),
                    description=seed_ent.get("description", entity.get("description")),
                    aliases=seed_ent.get("aliases"),
                    source_id=entity.get("source_id"),
                )
                break

    # Repoint edges from old entity_id to canonical
    db.conn.execute(
        "UPDATE edges SET source_entity_id = ? WHERE source_entity_id = ?",
        (canonical_id, entity_id)
    )
    db.conn.execute(
        "UPDATE edges SET target_entity_id = ? WHERE target_entity_id = ?",
        (canonical_id, entity_id)
    )
    db.conn.commit()

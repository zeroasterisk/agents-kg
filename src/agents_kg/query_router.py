"""Three-tier query router: Text-to-Cypher -> Neo4j -> optional Gemini synthesis.

Tier 1: Translate natural language to Cypher, execute against Neo4j.
Tier 2: If structured results are insufficient, synthesize an answer
        from KG subgraph + source chunks via Gemini, cache with 7-day TTL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("agents_kg.query_router")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY_MODEL = "gemini-3.5-flash"

CACHE_DIR = Path(
    os.environ.get(
        "KG_SYNTHESIS_CACHE_DIR",
        "/scion-volumes/scratchpad/agents-kg-synthesis-cache",
    )
)

DEFAULT_TTL_DAYS = 7

CYPHER_SYSTEM_PROMPT = """\
You are a Neo4j Cypher expert. The graph has the following schema:

NODE LABELS (each node also carries the :Entity label):
  Protocol, Organization, Project, Capability, Group, Person, Concept

NODE PROPERTIES (all labels share these):
  entity_id (string, unique), name, type, kind, description, aliases (list), source_id

Additional node types:
  Source — properties: uri, title, source_type, submitter_email, created_at, updated_at, content_hash
  Chunk  — properties: chunk_id, text, position, source_id

RELATIONSHIPS:
  IMPLEMENTS, CONTRIBUTES_TO, AUTHORED, MEMBER_OF, DEVELOPS,
  COMPLEMENTS, GOVERNS, SPONSORS, CHAIRS, PART_OF, USES, ADDRESSES,
  FROM_SOURCE, EXTRACTED_FROM, COMPETES_WITH, SUPERSEDES, DEFINES

Relationship properties: edge_id, confidence, source_type, valid_from, valid_to, chunk_id

RULES:
- Return ONLY the Cypher query, no explanation, no markdown fences.
- READ-ONLY queries only (MATCH/RETURN/WHERE/ORDER BY/LIMIT/WITH/OPTIONAL MATCH).
- Always RETURN entity_id, name, description for entity nodes when possible.
- Use parameterized label matching (e.g. MATCH (n:Protocol) not MATCH (n)).
- LIMIT results to 25 unless the question asks for a specific count.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a knowledge assistant for the agentic technology ecosystem.
Answer based on the provided graph data and source material.
Express uncertainty where the evidence is incomplete. Cite sources.

Return your answer as JSON with these fields:
  answer: string (your full answer text)
  confidence: "high" | "medium" | "low"
  sources: list of source references (URIs or titles)
  entity_ids: list of entity_id strings you referenced

Return ONLY valid JSON, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Gemini client (lazy singleton)
# ---------------------------------------------------------------------------

_client = None


def _get_genai_client():
    """Return a cached google.genai client, initialised on first call."""
    global _client
    if _client is not None:
        return _client

    from google import genai

    kwargs: dict = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["vertexai"] = True
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    _client = genai.Client(**kwargs)
    return _client


# ---------------------------------------------------------------------------
# Tier 1 — Text-to-Cypher
# ---------------------------------------------------------------------------


def generate_cypher(question: str) -> str:
    """Translate a natural-language question into a Cypher READ query."""
    client = _get_genai_client()
    response = client.models.generate_content(
        model=QUERY_MODEL,
        contents=question,
        config={
            "system_instruction": CYPHER_SYSTEM_PROMPT,
            "temperature": 0.0,
        },
    )
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text


def execute_cypher(cypher: str, driver) -> list[dict]:
    """Execute a Cypher query against Neo4j with a 10s timeout.

    Returns [] on any error.
    """
    try:
        records, _, _ = driver.execute_query(
            cypher,
            routing_="r",
            timeout_=10.0,
        )
        results = [dict(record) for record in records]
        return _serialize_neo4j(results)
    except Exception as exc:
        log.warning("Cypher execution failed: %s — query: %s", exc, cypher)
        return []


def _serialize_neo4j(results: list[dict]) -> list[dict]:
    """Convert Neo4j types (Node, Relationship) to JSON-serializable dicts."""
    out = []
    for row in results:
        clean: dict = {}
        for key, val in row.items():
            if hasattr(val, "items"):
                clean[key] = dict(val)
            elif hasattr(val, "_properties"):
                clean[key] = dict(val._properties)
            else:
                clean[key] = val
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Sufficiency check
# ---------------------------------------------------------------------------


def is_sufficient(results: list[dict], question: str) -> bool:
    """Heuristic: sufficient if >=1 result and all have name + description."""
    if len(results) < 1:
        return False
    for r in results:
        # Results may be nested (e.g. {n: {name: ..., description: ...}})
        target = r
        if len(r) == 1:
            val = next(iter(r.values()))
            if isinstance(val, dict):
                target = val
        if not target.get("name") or not target.get("description"):
            return False
    return True


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def get_cache_path(cache_key: str) -> Path:
    """Deterministic path for a cache key (SHA-256 prefix)."""
    h = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def is_cache_valid(cache_path: Path, ttl_days: int = DEFAULT_TTL_DAYS) -> bool:
    """Check if a cache file exists and is within the TTL."""
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < (ttl_days * 86400)


def _read_cache(cache_path: Path) -> Optional[dict]:
    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_path: Path, data: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# Source chunks from SQLite
# ---------------------------------------------------------------------------


def get_source_chunks(
    entity_ids: list[str], db_path: str, limit: int = 10
) -> list[str]:
    """Retrieve source text chunks associated with the given entity IDs."""
    if not entity_ids:
        return []
    from .db import Database

    db = Database(db_path)
    try:
        placeholders = ",".join("?" for _ in entity_ids)
        rows = db.conn.execute(
            f"""SELECT DISTINCT c.text
                FROM chunks c
                JOIN entities e ON c.source_id = e.source_id
                WHERE e.entity_id IN ({placeholders})
                ORDER BY c.id
                LIMIT ?""",
            [*entity_ids, limit],
        ).fetchall()
        return [row["text"] for row in rows]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tier 2 — Gemini synthesis
# ---------------------------------------------------------------------------


def synthesize(
    question: str, kg_results: list[dict], chunks: list[str]
) -> dict:
    """Synthesize an answer from KG results and source chunks via Gemini."""
    client = _get_genai_client()
    context = json.dumps(kg_results, indent=2, default=str)
    chunk_text = "\n---\n".join(chunks) if chunks else "(no source text available)"

    response = client.models.generate_content(
        model=QUERY_MODEL,
        contents=(
            f"Question: {question}\n\n"
            f"Knowledge Graph Results:\n{context}\n\n"
            f"Source Text:\n{chunk_text}"
        ),
        config={
            "system_instruction": SYNTHESIS_SYSTEM_PROMPT,
            "temperature": 0.2,
        },
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer": response.text.strip(),
            "confidence": "medium",
            "sources": [],
            "entity_ids": [],
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _extract_entity_ids(results: list[dict]) -> list[str]:
    """Pull entity_id values from Cypher results (handles nested nodes)."""
    ids: list[str] = []
    for row in results:
        for val in row.values():
            if isinstance(val, dict) and val.get("entity_id"):
                ids.append(val["entity_id"])
            elif isinstance(val, str) and ":" in val:
                # Might be an entity_id returned directly
                pass
        if row.get("entity_id"):
            ids.append(row["entity_id"])
    return list(dict.fromkeys(ids))  # dedupe, preserve order


def answer(
    question: str,
    driver,
    db_path: str,
    force_refresh: bool = False,
) -> dict:
    """Full query pipeline: cache -> Cypher -> sufficiency -> synthesize -> cache.

    Returns a dict with: answer, source, entity_ids, cypher_used, sources,
    confidence, cached, cache_age_hours.
    """
    cache_key = question.strip().lower()
    cache_path = get_cache_path(cache_key)

    # ---- Check cache ----
    if not force_refresh and is_cache_valid(cache_path):
        cached = _read_cache(cache_path)
        if cached:
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            cached["source"] = "cached"
            cached["cached"] = True
            cached["cache_age_hours"] = round(age_hours, 1)
            return cached

    # ---- Tier 1: Text-to-Cypher ----
    cypher = generate_cypher(question)
    results = execute_cypher(cypher, driver)

    if is_sufficient(results, question):
        entity_ids = _extract_entity_ids(results)
        return {
            "answer": _format_kg_answer(results),
            "source": "kg",
            "entity_ids": entity_ids,
            "cypher_used": cypher,
            "sources": [],
            "confidence": "high",
            "cached": False,
            "cache_age_hours": None,
        }

    # ---- Tier 2: Synthesis ----
    entity_ids = _extract_entity_ids(results)
    chunks = get_source_chunks(entity_ids, db_path) if entity_ids else []
    synthesis = synthesize(question, results, chunks)

    result = {
        "answer": synthesis.get("answer", ""),
        "source": "synthesis",
        "entity_ids": synthesis.get("entity_ids", entity_ids),
        "cypher_used": cypher,
        "sources": synthesis.get("sources", []),
        "confidence": synthesis.get("confidence", "medium"),
        "cached": False,
        "cache_age_hours": None,
    }

    # Write to cache
    cache_data = {
        "question": question,
        "cache_key": cache_key,
        **result,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_cache(cache_path, cache_data)

    return result


def _format_kg_answer(results: list[dict]) -> str:
    """Format raw KG results into a readable text answer."""
    lines: list[str] = []
    for row in results:
        parts: list[str] = []
        for key, val in row.items():
            if isinstance(val, dict):
                name = val.get("name", "")
                desc = val.get("description", "")
                if name:
                    parts.append(f"{name}: {desc}" if desc else name)
            elif isinstance(val, str):
                parts.append(val)
        if parts:
            lines.append(" — ".join(parts))
    return "\n".join(lines) if lines else json.dumps(results, default=str)


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def invalidate_entity_cache(entity_id: str) -> int:
    """Delete cached answers referencing the given entity_id.

    Returns the number of cache files deleted.
    """
    if not CACHE_DIR.exists():
        return 0
    deleted = 0
    for cache_file in CACHE_DIR.rglob("*.json"):
        try:
            data = json.loads(cache_file.read_text())
            if entity_id in data.get("entity_ids", []):
                cache_file.unlink()
                log.info(
                    "Invalidated cache %s (entity: %s)", cache_file.name, entity_id
                )
                deleted += 1
        except (json.JSONDecodeError, OSError):
            continue
    return deleted

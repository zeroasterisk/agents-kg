"""Stage 5: Extract entities and edges using Gemini Flash structured output."""

import json
import logging
import hashlib
from ..db import Database

log = logging.getLogger(__name__)

EXTRACT_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are a knowledge graph extraction engine for the agentic web ecosystem.

Given a text chunk, extract entities and relationships according to this ontology:

NODE TYPES:
- Organization (kind: company, standards_body, foundation, consortium)
- Group (kind: tsc, wg, sig, task_force, team)
- Person
- Project (kind: framework, sdk, library, tool, platform)
- Protocol (kind: spec, standard, rfc, draft)
- Capability (recursive via PART_OF)

EDGE TYPES:
MEMBER_OF, GOVERNS, DEVELOPS, IMPLEMENTS, COMPETES_WITH, ADDRESSES, AUTHORED, CHAIRS, SPONSORS, PART_OF, SUPERSEDES, FROM_SOURCE, CONTRIBUTES_TO, DEFINES, COMPLEMENTS

Respond with valid JSON matching this schema:
{
  "entities": [
    {
      "entity_id": "type:kebab-case-name",
      "name": "Display Name",
      "type": "Organization|Group|Person|Project|Protocol|Capability",
      "kind": "specific kind or null",
      "description": "Brief description",
      "aliases": ["alt name 1"]
    }
  ],
  "edges": [
    {
      "source_entity_id": "type:name",
      "target_entity_id": "type:name",
      "edge_type": "DEVELOPS",
      "confidence": 0.9,
      "properties": {}
    }
  ]
}

Rules:
- Use kebab-case for entity_id, prefixed with type (e.g., "organization:google", "project:a2a")
- Only extract what's explicitly stated or strongly implied
- Set confidence 0.5-1.0 based on how explicit the relationship is
- If nothing relevant found, return {"entities": [], "edges": []}
"""


def _make_edge_id(src: str, tgt: str, edge_type: str) -> str:
    raw = f"{src}|{edge_type}|{tgt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run(db: Database, source: dict) -> bool:
    source_id = source["id"]
    chunks = db.get_chunks(source_id)
    if not chunks:
        raise RuntimeError("No chunks to extract from")

    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed")

    import os
    kwargs = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs["vertexai"] = True
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    client = genai.Client(**kwargs)

    total_entities = 0
    total_edges = 0

    for chunk in chunks:
        log.info("Extracting from chunk %d (source %d)", chunk["id"], source_id)

        try:
            response = client.models.generate_content(
                model=EXTRACT_MODEL,
                contents=f"Extract entities and relationships from this text:\n\n{chunk['text']}",
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )

            data = json.loads(response.text)
        except (json.JSONDecodeError, Exception) as e:
            log.warning("Extraction failed for chunk %d: %s", chunk["id"], e)
            continue

        for ent in data.get("entities", []):
            db.add_entity(
                entity_id=ent["entity_id"],
                name=ent["name"],
                entity_type=ent["type"],
                kind=ent.get("kind"),
                description=ent.get("description"),
                aliases=ent.get("aliases"),
                source_id=source_id,
                chunk_id=chunk["id"],
            )
            total_entities += 1

        for edge in data.get("edges", []):
            edge_id = _make_edge_id(edge["source_entity_id"], edge["target_entity_id"], edge["edge_type"])
            db.add_edge(
                edge_id=edge_id,
                source_entity_id=edge["source_entity_id"],
                target_entity_id=edge["target_entity_id"],
                edge_type=edge["edge_type"],
                properties=edge.get("properties"),
                confidence=edge.get("confidence", 0.5),
                chunk_id=chunk["id"],
                source_id=source_id,
            )
            total_edges += 1

    log.info("Extracted %d entities, %d edges from source %d", total_entities, total_edges, source_id)
    db.update_source(source_id, stage="review", status="pending_review")
    return True

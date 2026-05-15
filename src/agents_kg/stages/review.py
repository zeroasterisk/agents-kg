"""Stage 6: Review -- auto-approve high-confidence entities/edges for Neo4j load.

Three-tier design (tier 1 implemented here):
1. Auto-approve: entities always approved; edges with confidence >= threshold
2. Agentic research: (future) ambiguous entities get LLM research pass
3. Human review: (future) unresolvable conflicts queued for human decision
"""

import logging
from ..db import Database

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


EDGE_CONFIDENCE_THRESHOLD = 0.7


def run(db: Database, source: dict) -> bool:
    source_id = source["id"]
    log = _log()

    entities = db.conn.execute(
        "SELECT id, entity_id, name, type FROM entities "
        "WHERE source_id = ? AND status = 'pending_review'",
        (source_id,),
    ).fetchall()

    edges = db.conn.execute(
        "SELECT id, edge_id, source_entity_id, target_entity_id, edge_type, confidence "
        "FROM edges WHERE source_id = ? AND status = 'pending_review'",
        (source_id,),
    ).fetchall()

    approved_entities = 0
    approved_edges = 0
    deferred_edges = 0

    for ent in entities:
        db.approve_entity(ent["id"])
        approved_entities += 1

    for edge in edges:
        if edge["confidence"] >= EDGE_CONFIDENCE_THRESHOLD:
            db.approve_edge(edge["id"])
            approved_edges += 1
        else:
            deferred_edges += 1
            log.info(
                "Deferred edge %s (%s -[%s]-> %s, confidence=%.2f)",
                edge["edge_id"], edge["source_entity_id"],
                edge["edge_type"], edge["target_entity_id"],
                edge["confidence"],
            )

    log.info(
        "Review source %d: approved %d entities, %d edges; deferred %d edges",
        source_id, approved_entities, approved_edges, deferred_edges,
    )

    db.update_source(source_id, stage="load", status="processing")
    return True

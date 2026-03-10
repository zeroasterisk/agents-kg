"""Pipeline runner: process sources through stages with error handling."""

import logging
import time
from .db import Database
from .stages import fetch, parse, chunk, embed, extract, load

log = logging.getLogger(__name__)

STAGES = {
    "fetch": fetch,
    "parse": parse,
    "chunk": chunk,
    "embed": embed,
    "extract": extract,
    "load": load,
}

STAGE_ORDER = ["fetch", "parse", "chunk", "embed", "extract", "review", "load"]


def process_source(db: Database, source: dict, neo4j_driver=None) -> bool:
    """Process a single source through its next stage. Returns True if progress made."""
    stage = source["stage"]
    source_id = source["id"]

    if stage == "review":
        # Review stage is manual — skip
        return False
    if stage == "done":
        return False

    stage_mod = STAGES.get(stage)
    if not stage_mod:
        log.error("Unknown stage %s for source %d", stage, source_id)
        return False

    try:
        if stage == "load":
            return stage_mod.run(db, source, neo4j_driver=neo4j_driver)
        else:
            return stage_mod.run(db, source)
    except Exception as e:
        log.error("Stage %s failed for source %d: %s", stage, source_id, e)
        db.fail_source(source_id, f"[{stage}] {e}")
        return False


def process_all(db: Database, neo4j_driver=None) -> dict:
    """Process all pending sources. Returns summary stats."""
    sources = db.get_pending_sources()
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    for source in sources:
        # Keep processing through stages until blocked
        while True:
            source = db.get_source(source["id"])
            if not source or source["status"] in ("complete", "failed", "dead_letter", "pending_review"):
                break

            try:
                made_progress = process_source(db, source, neo4j_driver=neo4j_driver)
                if made_progress:
                    stats["processed"] += 1
                else:
                    stats["skipped"] += 1
                    break
            except Exception as e:
                log.error("Unexpected error processing source %d: %s", source["id"], e)
                db.fail_source(source["id"], str(e))
                stats["failed"] += 1
                break

    return stats

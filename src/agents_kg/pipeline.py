"""Pipeline runner: process sources through stages with Prefect orchestration."""

import logging
from prefect import task, flow
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


@task(
    name="fetch",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["network"],
)
def run_fetch(db: Database, source: dict) -> bool:
    return fetch.run(db, source)


@task(name="parse")
def run_parse(db: Database, source: dict) -> bool:
    return parse.run(db, source)


@task(name="chunk")
def run_chunk(db: Database, source: dict) -> bool:
    return chunk.run(db, source)


@task(
    name="embed",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["gemini", "network"],
)
def run_embed(db: Database, source: dict) -> bool:
    return embed.run(db, source)


@task(
    name="extract",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["gemini", "network"],
)
def run_extract(db: Database, source: dict) -> bool:
    return extract.run(db, source)


@task(name="load")
def run_load(db: Database, source: dict, neo4j_driver=None) -> bool:
    return load.run(db, source, neo4j_driver=neo4j_driver)


TASK_MAP = {
    "fetch": run_fetch,
    "parse": run_parse,
    "chunk": run_chunk,
    "embed": run_embed,
    "extract": run_extract,
    "load": run_load,
}


def process_source(db: Database, source: dict, neo4j_driver=None) -> bool:
    """Process a single source through its next stage. Returns True if progress made."""
    stage = source["stage"]
    source_id = source["id"]

    if stage == "review":
        return False
    if stage == "done":
        return False

    task_fn = TASK_MAP.get(stage)
    if not task_fn:
        log.error("Unknown stage %s for source %d", stage, source_id)
        return False

    try:
        if stage == "load":
            return task_fn(db, source, neo4j_driver=neo4j_driver)
        else:
            return task_fn(db, source)
    except Exception as e:
        log.error("Stage %s failed for source %d: %s", stage, source_id, e)
        db.fail_source(source_id, f"[{stage}] {e}")
        return False


@flow(name="process-all-sources", log_prints=True)
def process_all(db: Database, neo4j_driver=None) -> dict:
    """Process all pending sources. Returns summary stats."""
    sources = db.get_pending_sources()
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    for source in sources:
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

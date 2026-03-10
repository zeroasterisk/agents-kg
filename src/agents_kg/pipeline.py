"""Pipeline runner: idiomatic Prefect 3 orchestration for the agents-kg pipeline.

Architecture:
- process_all: top-level @flow that iterates sources and spawns sub-flows
- process_source: @flow per source (visible as separate flow runs in UI)
- Each stage is a @task with appropriate retries, tags, and caching
- Domain logic stays in stages/; this module is pure orchestration
- SQLite stays for domain state; Prefect handles orchestration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import INPUTS

from .db import Database
from .stages import fetch, parse, chunk, embed, extract, load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_content_cache_key(context, parameters):
    """Cache key based on source content_hash — skip if content unchanged."""
    source = parameters.get("source", {})
    content_hash = source.get("content_hash") or "none"
    return f"{parameters.get('stage', 'unknown')}:{source.get('id', 0)}:{content_hash}"


def _on_stage_failure(task, task_run, state):
    """Hook: log failures through Prefect's structured logger."""
    logger = get_run_logger()
    logger.error("Task %s failed: %s", task_run.name, state.message)


# ---------------------------------------------------------------------------
# Stage tasks
# ---------------------------------------------------------------------------

@task(
    name="fetch",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["network"],
    on_failure=[_on_stage_failure],
    persist_result=True,
)
def run_fetch(db: Database, source: dict) -> bool:
    log = get_run_logger()
    log.info("Fetching source %d: %s", source["id"], source["uri"])
    return fetch.run(db, source)


@task(
    name="parse",
    on_failure=[_on_stage_failure],
)
def run_parse(db: Database, source: dict) -> bool:
    log = get_run_logger()
    log.info("Parsing source %d", source["id"])
    return parse.run(db, source)


@task(
    name="chunk",
    on_failure=[_on_stage_failure],
)
def run_chunk(db: Database, source: dict) -> bool:
    log = get_run_logger()
    log.info("Chunking source %d", source["id"])
    return chunk.run(db, source)


@task(
    name="embed",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["gemini", "network"],
    on_failure=[_on_stage_failure],
    persist_result=True,
)
def run_embed(db: Database, source: dict) -> bool:
    log = get_run_logger()
    log.info("Embedding source %d", source["id"])
    return embed.run(db, source)


@task(
    name="extract",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["gemini", "network"],
    on_failure=[_on_stage_failure],
    persist_result=True,
)
def run_extract(db: Database, source: dict) -> bool:
    log = get_run_logger()
    log.info("Extracting entities from source %d", source["id"])
    return extract.run(db, source)


@task(
    name="load",
    on_failure=[_on_stage_failure],
    persist_result=True,
)
def run_load(db: Database, source: dict, neo4j_driver=None) -> bool:
    log = get_run_logger()
    log.info("Loading source %d to graph", source["id"])
    return load.run(db, source, neo4j_driver=neo4j_driver)


TASK_MAP = {
    "fetch": run_fetch,
    "parse": run_parse,
    "chunk": run_chunk,
    "embed": run_embed,
    "extract": run_extract,
    "load": run_load,
}

STAGE_ORDER = ["fetch", "parse", "chunk", "embed", "extract", "review", "load"]


# ---------------------------------------------------------------------------
# Source extraction summary (artifact helper)
# ---------------------------------------------------------------------------

def _source_artifact(db: Database, source_id: int, source_uri: str):
    """Create a markdown artifact summarizing what was extracted from a source."""
    entities = db.conn.execute(
        "SELECT entity_id, name, type, kind FROM entities WHERE source_id = ?", (source_id,)
    ).fetchall()
    edges = db.conn.execute(
        "SELECT source_entity_id, target_entity_id, edge_type FROM edges WHERE source_id = ?", (source_id,)
    ).fetchall()

    lines = [f"## Source #{source_id}", f"**URI:** {source_uri}", ""]

    if entities:
        lines.append(f"### Entities ({len(entities)})")
        for e in entities:
            kind = f"/{e['kind']}" if e["kind"] else ""
            lines.append(f"- `{e['entity_id']}` — {e['name']} ({e['type']}{kind})")
        lines.append("")

    if edges:
        lines.append(f"### Edges ({len(edges)})")
        for e in edges:
            lines.append(f"- `{e['source_entity_id']}` —{e['edge_type']}→ `{e['target_entity_id']}`")
        lines.append("")

    if not entities and not edges:
        lines.append("*No entities or edges extracted.*")

    try:
        create_markdown_artifact(
            key=f"source-{source_id}-extraction",
            markdown="\n".join(lines),
            description=f"Extraction results for source #{source_id}",
        )
    except Exception:
        pass  # Artifacts require a Prefect server; graceful degradation


# ---------------------------------------------------------------------------
# Sub-flow: process a single source through all stages
# ---------------------------------------------------------------------------

@flow(name="process-source", log_prints=True)
def process_source(
    db: Database,
    source_id: int,
    source_uri: str,
    neo4j_driver=None,
) -> dict:
    """Process a single source through its remaining stages.

    Returns a dict with counts: {"stages_completed": N, "failed": bool}
    """
    log = get_run_logger()
    result = {"stages_completed": 0, "failed": False}

    while True:
        source = db.get_source(source_id)
        if not source:
            log.warning("Source %d not found", source_id)
            result["failed"] = True
            break

        stage = source["stage"]
        status = source["status"]

        if status in ("complete", "failed", "dead_letter", "pending_review"):
            break
        if stage in ("review", "done"):
            break

        task_fn = TASK_MAP.get(stage)
        if not task_fn:
            log.error("Unknown stage %s for source %d", stage, source_id)
            result["failed"] = True
            break

        try:
            if stage == "load":
                made_progress = task_fn(db, source, neo4j_driver=neo4j_driver)
            else:
                made_progress = task_fn(db, source)

            if made_progress:
                result["stages_completed"] += 1
            else:
                break
        except Exception as e:
            log.error("Stage %s failed for source %d: %s", stage, source_id, e)
            db.fail_source(source_id, f"[{stage}] {e}")
            result["failed"] = True
            break

    # Create extraction artifact
    _source_artifact(db, source_id, source_uri)

    return result


# ---------------------------------------------------------------------------
# Top-level flow: process all pending sources
# ---------------------------------------------------------------------------

@flow(name="process-all-sources", log_prints=True)
def process_all(
    db: Database,
    neo4j_driver=None,
) -> dict:
    """Process all pending sources. Returns summary stats."""
    log = get_run_logger()
    sources = db.get_pending_sources()

    stats = {"processed": 0, "failed": 0, "skipped": 0, "sources": len(sources)}
    log.info("Found %d pending sources", len(sources))

    for source in sources:
        log.info("Starting source %d: %s", source["id"], source["uri"][:80])
        result = process_source(
            db=db,
            source_id=source["id"],
            source_uri=source["uri"],
            neo4j_driver=neo4j_driver,
        )

        if result["failed"]:
            stats["failed"] += 1
        elif result["stages_completed"] > 0:
            stats["processed"] += 1
        else:
            stats["skipped"] += 1

    # Flow-level summary artifact
    _create_summary_artifact(stats)

    log.info(
        "Pipeline complete: %d processed, %d failed, %d skipped (of %d sources)",
        stats["processed"], stats["failed"], stats["skipped"], stats["sources"],
    )
    return stats


def _create_summary_artifact(stats: dict):
    """Create a flow-level summary artifact."""
    lines = [
        "## Pipeline Run Summary",
        "",
        f"- **Total sources:** {stats['sources']}",
        f"- **Processed:** {stats['processed']}",
        f"- **Failed:** {stats['failed']}",
        f"- **Skipped:** {stats['skipped']}",
    ]
    try:
        create_markdown_artifact(
            key="pipeline-run-summary",
            markdown="\n".join(lines),
            description="Pipeline processing summary",
        )
    except Exception:
        pass  # Graceful degradation without Prefect server

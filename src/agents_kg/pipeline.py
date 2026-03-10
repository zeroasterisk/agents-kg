"""Pipeline orchestration with Prefect 3.

SQLite owns domain state (sources, chunks, entities, edges).
Prefect owns orchestration: retry, caching, concurrency, observability, artifacts.
"""

from __future__ import annotations

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.concurrency.sync import rate_limit
from prefect.tasks import task_input_hash

from .db import Database, content_hash
from .stages import fetch, parse, chunk, embed, extract, load


# ---------------------------------------------------------------------------
# Cache key: hash source_id + content_hash so unchanged content skips work
# ---------------------------------------------------------------------------

def _source_cache_key(context, parameters):
    """Cache key from source id + content_hash (skips if content unchanged)."""
    source = parameters.get("source") or {}
    return f"{source.get('id', '')}:{source.get('content_hash', '')}"


def _chunk_cache_key(context, parameters):
    """Cache key from source id + parsed_text hash."""
    source = parameters.get("source") or {}
    text = source.get("parsed_text") or source.get("raw_text") or ""
    return f"{source.get('id', '')}:{content_hash(text)}"


# ---------------------------------------------------------------------------
# Tasks — each returns a typed result dict, logs via Prefect, creates artifacts
# ---------------------------------------------------------------------------

@task(
    name="fetch-source",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["network"],
    description="Fetch URL content and store raw HTML/text",
)
def task_fetch(db_path: str, source: dict) -> dict:
    """Fetch URL → raw_text. Returns {fetched, content_hash, content_type}."""
    logger = get_run_logger()
    db = Database(db_path)

    try:
        result = fetch.run(db, source)
        refreshed = db.get_source(source["id"])
        return {
            "fetched": result,
            "source_id": source["id"],
            "uri": source["uri"],
            "content_hash": refreshed.get("content_hash") if refreshed else None,
            "stage": refreshed.get("stage") if refreshed else None,
        }
    finally:
        db.close()


@task(
    name="parse-content",
    tags=["cpu"],
    cache_key_fn=_source_cache_key,
    description="Parse raw HTML/text to clean markdown",
)
def task_parse(db_path: str, source: dict) -> dict:
    """Parse raw_text → parsed_text. Returns {title, text_length}."""
    logger = get_run_logger()
    db = Database(db_path)

    try:
        parse.run(db, source)
        refreshed = db.get_source(source["id"])
        title = refreshed.get("title") if refreshed else None
        parsed_len = len(refreshed.get("parsed_text") or "") if refreshed else 0
        logger.info("Parsed source %d → %d chars, title=%s", source["id"], parsed_len, title)
        return {
            "source_id": source["id"],
            "title": title,
            "text_length": parsed_len,
        }
    finally:
        db.close()


@task(
    name="chunk-text",
    tags=["cpu"],
    cache_key_fn=_chunk_cache_key,
    description="Split parsed text into ~500-token chunks",
)
def task_chunk(db_path: str, source: dict) -> dict:
    """Chunk parsed_text → chunks table. Returns {chunk_count}."""
    logger = get_run_logger()
    db = Database(db_path)

    try:
        chunk.run(db, source)
        chunks = db.get_chunks(source["id"])
        logger.info("Chunked source %d → %d chunks", source["id"], len(chunks))
        return {
            "source_id": source["id"],
            "chunk_count": len(chunks),
        }
    finally:
        db.close()


@task(
    name="embed-chunks",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["gemini", "network"],
    description="Embed chunks via Gemini embedding API",
)
def task_embed(db_path: str, source: dict) -> dict:
    """Embed all chunks for a source. Returns {embedded_count}."""
    logger = get_run_logger()
    rate_limit("gemini-api", occupy=1)
    db = Database(db_path)

    try:
        unembedded_before = len(db.get_unembedded_chunks(source["id"]))
        embed.run(db, source)
        unembedded_after = len(db.get_unembedded_chunks(source["id"]))
        embedded = unembedded_before - unembedded_after
        logger.info("Embedded %d chunks for source %d", embedded, source["id"])
        return {
            "source_id": source["id"],
            "embedded_count": embedded,
        }
    finally:
        db.close()


@task(
    name="extract-entities",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    tags=["gemini", "network"],
    description="Extract entities and edges via Gemini Flash",
)
def task_extract(db_path: str, source: dict) -> dict:
    """Extract entities/edges from chunks. Returns {entities, edges}."""
    logger = get_run_logger()
    rate_limit("gemini-api", occupy=1)
    db = Database(db_path)

    try:
        entities_before = len(db.get_entities_by_status("pending_review"))
        edges_before = len(db.get_edges_by_status("pending_review"))

        extract.run(db, source)

        entities_after = len(db.get_entities_by_status("pending_review"))
        edges_after = len(db.get_edges_by_status("pending_review"))

        new_entities = entities_after - entities_before
        new_edges = edges_after - edges_before

        logger.info(
            "Extracted %d entities, %d edges from source %d",
            new_entities, new_edges, source["id"],
        )

        # Create artifact with extraction summary
        create_markdown_artifact(
            key=f"extraction-source-{source['id']}",
            markdown=(
                f"## Extraction: source {source['id']}\n"
                f"**URI:** {source.get('uri', '?')}\n\n"
                f"- **Entities found:** {new_entities}\n"
                f"- **Edges found:** {new_edges}\n"
            ),
            description=f"KG extraction results for source {source['id']}",
        )

        return {
            "source_id": source["id"],
            "entities": new_entities,
            "edges": new_edges,
        }
    finally:
        db.close()


@task(
    name="load-to-graph",
    retries=2,
    retry_delay_seconds=[5, 15],
    tags=["neo4j"],
    description="Load approved entities/edges to Neo4j + YAML export",
)
def task_load(db_path: str, source: dict, neo4j_uri: str | None = None,
              neo4j_auth: tuple[str, str] | None = None) -> dict:
    """Load approved items to Neo4j and export YAML. Returns {entities_loaded, edges_loaded}."""
    logger = get_run_logger()
    db = Database(db_path)

    neo4j_driver = None
    if neo4j_uri:
        try:
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            neo4j_driver.verify_connectivity()
        except Exception as e:
            logger.warning("Neo4j unavailable (%s), will export YAML only", e)
            neo4j_driver = None

    try:
        # Count before
        entities = db.conn.execute(
            "SELECT COUNT(*) FROM entities WHERE source_id = ? AND status = 'approved'",
            (source["id"],),
        ).fetchone()[0]
        edges = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id = ? AND status = 'approved'",
            (source["id"],),
        ).fetchone()[0]

        load.run(db, source, neo4j_driver=neo4j_driver)

        logger.info(
            "Loaded %d entities, %d edges for source %d (neo4j=%s)",
            entities, edges, source["id"], neo4j_driver is not None,
        )
        return {
            "source_id": source["id"],
            "entities_loaded": entities,
            "edges_loaded": edges,
            "neo4j": neo4j_driver is not None,
        }
    finally:
        if neo4j_driver:
            neo4j_driver.close()
        db.close()


# ---------------------------------------------------------------------------
# Sub-flow: process a single source through all stages
# ---------------------------------------------------------------------------

@flow(name="process-source", log_prints=True)
def process_source(
    db_path: str,
    source_id: int,
    neo4j_uri: str | None = None,
    neo4j_auth: tuple[str, str] | None = None,
) -> dict:
    """Drive a single source through its remaining stages. Returns stage results."""
    logger = get_run_logger()
    db = Database(db_path)
    source = db.get_source(source_id)
    db.close()

    if not source:
        logger.warning("Source %d not found", source_id)
        return {"source_id": source_id, "status": "not_found"}

    results = {"source_id": source_id, "uri": source["uri"], "stages": {}}

    # Stage pipeline — each stage checks current state, runs if applicable
    stage_sequence = [
        ("fetch", task_fetch),
        ("parse", task_parse),
        ("chunk", task_chunk),
        ("embed", task_embed),
        ("extract", task_extract),
    ]

    for stage_name, stage_task in stage_sequence:
        # Re-read source to get current stage
        db = Database(db_path)
        source = db.get_source(source_id)
        db.close()

        if not source or source["status"] in ("complete", "failed", "dead_letter", "pending_review"):
            break

        if source["stage"] != stage_name:
            continue

        try:
            result = stage_task(db_path, source)
            results["stages"][stage_name] = result
        except Exception as e:
            logger.error("Stage %s failed for source %d: %s", stage_name, source_id, e)
            db = Database(db_path)
            db.fail_source(source_id, f"[{stage_name}] {e}")
            db.close()
            results["stages"][stage_name] = {"error": str(e)}
            break

    # Load stage (needs neo4j params)
    db = Database(db_path)
    source = db.get_source(source_id)
    db.close()

    if source and source["stage"] == "load" and source["status"] not in ("complete", "failed", "dead_letter"):
        try:
            result = task_load(db_path, source, neo4j_uri=neo4j_uri, neo4j_auth=neo4j_auth)
            results["stages"]["load"] = result
        except Exception as e:
            logger.error("Load failed for source %d: %s", source_id, e)
            db = Database(db_path)
            db.fail_source(source_id, f"[load] {e}")
            db.close()
            results["stages"]["load"] = {"error": str(e)}

    # Final status
    db = Database(db_path)
    source = db.get_source(source_id)
    db.close()
    results["final_status"] = source["status"] if source else "unknown"
    results["final_stage"] = source["stage"] if source else "unknown"

    return results


# ---------------------------------------------------------------------------
# Top-level flow: process all pending sources
# ---------------------------------------------------------------------------

@flow(name="process-all-sources", log_prints=True)
def process_all(
    db_path: str = "pipeline.db",
    neo4j_uri: str | None = None,
    neo4j_auth: tuple[str, str] | None = None,
) -> dict:
    """Process all pending sources. Returns summary with artifacts."""
    logger = get_run_logger()
    db = Database(db_path)
    sources = db.get_pending_sources()
    db.close()

    if not sources:
        logger.info("No pending sources")
        return {"processed": 0, "failed": 0, "skipped": 0, "results": []}

    logger.info("Processing %d pending sources", len(sources))

    all_results = []
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    for source in sources:
        result = process_source(
            db_path=db_path,
            source_id=source["id"],
            neo4j_uri=neo4j_uri,
            neo4j_auth=neo4j_auth,
        )
        all_results.append(result)

        if result.get("final_status") == "complete":
            stats["processed"] += 1
        elif result.get("final_status") in ("failed", "dead_letter"):
            stats["failed"] += 1
        else:
            stats["skipped"] += 1

    # Create summary artifact
    if all_results:
        table_rows = []
        for r in all_results:
            stages_done = list(r.get("stages", {}).keys())
            entities = sum(
                s.get("entities", 0) for s in r.get("stages", {}).values() if isinstance(s, dict)
            )
            edges = sum(
                s.get("edges", 0) for s in r.get("stages", {}).values() if isinstance(s, dict)
            )
            table_rows.append({
                "source_id": r["source_id"],
                "uri": (r.get("uri") or "")[:60],
                "status": r.get("final_status", "?"),
                "stages": ", ".join(stages_done),
                "entities": entities,
                "edges": edges,
            })

        create_table_artifact(
            key="pipeline-run-summary",
            table=table_rows,
            description=f"Pipeline run: {stats['processed']} processed, {stats['failed']} failed",
        )

    logger.info(
        "Pipeline complete: %d processed, %d failed, %d skipped",
        stats["processed"], stats["failed"], stats["skipped"],
    )

    stats["results"] = all_results
    return stats

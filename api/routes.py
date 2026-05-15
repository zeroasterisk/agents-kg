"""API routes for the agents-kg REST API."""

from __future__ import annotations

import logging
import os
import re
import sys

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents_kg.db import Database
from agents_kg.pipeline import process_source

from .auth import User, get_current_user
from .models import (
    HealthResponse,
    HistoryItem,
    IngestRequest,
    IngestResponse,
    JobStatus,
    QueryRequest,
    QueryResponse,
)

log = logging.getLogger("agents_kg.api")

router = APIRouter()

WRITE_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+\{)\b",
    re.IGNORECASE,
)


def _status_label(source: dict) -> str:
    """Map DB status/stage to the API lifecycle labels."""
    db_status = source["status"]
    stage = source.get("stage", "")

    if db_status == "pending":
        return "queued"
    if db_status == "processing":
        stage_map = {
            "fetch": "fetching",
            "parse": "parsing",
            "chunk": "chunking",
            "embed": "embedding",
            "extract": "extracting",
            "resolve": "resolving",
            "load": "loading",
        }
        return stage_map.get(stage, stage)
    if db_status == "pending_review":
        return "review_needed"
    if db_status == "complete":
        return "ingested"
    if db_status in ("failed", "dead_letter"):
        return "failed"
    return db_status


def _get_db(request: Request) -> Database:
    return request.app.state.db


def _run_pipeline(db_path: str, source_id: int, source_uri: str, neo4j_driver):
    """Background task: run the pipeline for a single source."""
    try:
        process_source(
            db_path=db_path,
            source_id=source_id,
            source_uri=source_uri,
            neo4j_driver=neo4j_driver,
        )
    except Exception:
        log.exception("Pipeline failed for source %d", source_id)


# ---- POST /ingest ----

@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    user: User = Depends(get_current_user),
):
    db = _get_db(request)
    neo4j_driver = getattr(request.app.state, "neo4j_driver", None)
    job_ids: list[int] = []

    for url in body.urls:
        source_id = db.add_source(
            uri=url,
            source_type="url",
            submitter_email=user.user_id,
        )
        if source_id is None:
            existing = db.get_source_by_uri(url)
            if existing:
                job_ids.append(existing["id"])
                if existing["status"] in ("failed", "dead_letter", "pending"):
                    db.reset_source(existing["id"])
                    _tag_source_category(db, existing["id"], body.source_type)
                    background_tasks.add_task(
                        _run_pipeline,
                        db_path=db.path,
                        source_id=existing["id"],
                        source_uri=url,
                        neo4j_driver=neo4j_driver,
                    )
            continue

        _tag_source_category(db, source_id, body.source_type)
        job_ids.append(source_id)

        background_tasks.add_task(
            _run_pipeline,
            db_path=db.path,
            source_id=source_id,
            source_uri=url,
            neo4j_driver=neo4j_driver,
        )

    return IngestResponse(job_ids=job_ids, submitted_by=user.user_id)


def _tag_source_category(db: Database, source_id: int, category: str):
    """Store source_category via a lightweight column addition."""
    try:
        db.conn.execute("SELECT source_category FROM sources LIMIT 0")
    except Exception:
        db.conn.execute("ALTER TABLE sources ADD COLUMN source_category TEXT")
        db.conn.commit()
    db.conn.execute(
        "UPDATE sources SET source_category = ? WHERE id = ?",
        (category, source_id),
    )
    db.conn.commit()


# ---- GET /ingest/status/{job_id} ----

@router.get("/ingest/status/{job_id}", response_model=JobStatus)
async def ingest_status(
    job_id: int,
    request: Request,
    user: User = Depends(get_current_user),
):
    db = _get_db(request)
    source = db.get_source(job_id)
    if not source:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=source["id"],
        uri=source["uri"],
        status=_status_label(source),
        stage=source.get("stage"),
        error=source.get("error"),
        created_at=source["created_at"],
        updated_at=source["updated_at"],
    )


# ---- GET /ingest/history ----

@router.get("/ingest/history", response_model=list[HistoryItem])
async def ingest_history(
    request: Request,
    user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=500),
    status: str | None = Query(None),
):
    db = _get_db(request)

    query = "SELECT * FROM sources WHERE 1=1"
    params: list = []

    if not user.is_agent:
        query += " AND submitter_email = ?"
        params.append(user.user_id)

    if status:
        db_statuses = _api_status_to_db(status)
        placeholders = ",".join("?" for _ in db_statuses)
        query += f" AND status IN ({placeholders})"
        params.extend(db_statuses)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = db.conn.execute(query, params).fetchall()

    items = []
    for row in rows:
        source = dict(row)
        cat = None
        try:
            cat = source.get("source_category")
        except Exception:
            pass

        items.append(HistoryItem(
            job_id=source["id"],
            uri=source["uri"],
            status=_status_label(source),
            stage=source.get("stage"),
            submitted_by=source.get("submitter_email"),
            source_type=cat,
            created_at=source["created_at"],
            updated_at=source["updated_at"],
        ))
    return items


def _api_status_to_db(api_status: str) -> list[str]:
    """Map API status labels back to DB status values."""
    mapping = {
        "queued": ["pending"],
        "review_needed": ["pending_review"],
        "ingested": ["complete"],
        "failed": ["failed", "dead_letter"],
    }
    return mapping.get(api_status, [api_status])


# ---- POST /query ----

@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    if WRITE_KEYWORDS.search(body.cypher):
        raise HTTPException(
            status_code=400,
            detail="Write operations are not allowed via the query endpoint",
        )

    neo4j_driver = getattr(request.app.state, "neo4j_driver", None)
    if not neo4j_driver:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        records, _, _ = neo4j_driver.execute_query(body.cypher)
        results = [dict(record) for record in records]
        serialized = _serialize_neo4j(results)
        return QueryResponse(results=serialized)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _serialize_neo4j(results: list[dict]) -> list[dict]:
    """Convert Neo4j types to JSON-serializable dicts."""
    out = []
    for row in results:
        clean = {}
        for key, val in row.items():
            if hasattr(val, "items"):
                clean[key] = dict(val)
            elif hasattr(val, "_properties"):
                clean[key] = dict(val._properties)
            else:
                clean[key] = val
        out.append(clean)
    return out


# ---- GET /health ----

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    neo4j_driver = getattr(request.app.state, "neo4j_driver", None)
    neo4j_status = "not_configured"

    if neo4j_driver:
        try:
            neo4j_driver.verify_connectivity()
            neo4j_status = "connected"
        except Exception:
            neo4j_status = "disconnected"

    return HealthResponse(
        status="ok" if neo4j_status == "connected" else "degraded",
        neo4j=neo4j_status,
    )

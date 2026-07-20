"""Natural-language query endpoint for the agents-kg knowledge graph."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from fastapi import APIRouter, Depends, HTTPException, Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents_kg.query_router import answer as query_answer

from .auth import User, get_current_user
from .models import AskRequest, AskResponse

log = logging.getLogger("agents_kg.api.ask")

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Answer a natural-language question about the agentic ecosystem.

    Uses a three-tier pipeline:
      1. Text-to-Cypher against Neo4j (fast, structured)
      2. On-demand Gemini synthesis from KG subgraph + source chunks
      3. Disk-cached synthesis results (7-day TTL)
    """
    neo4j_driver = getattr(request.app.state, "neo4j_driver", None)
    if not neo4j_driver:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    db_path = request.app.state.db.path

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                query_answer,
                question=body.question,
                driver=neo4j_driver,
                db_path=db_path,
                force_refresh=body.force_refresh,
            ),
            timeout=30.0,
        )
        return AskResponse(**result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out (30s limit)")
    except Exception as exc:
        log.exception("Query failed: %s", body.question)
        raise HTTPException(status_code=500, detail=str(exc))

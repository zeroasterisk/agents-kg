"""FastAPI application for the agents-kg REST API."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents_kg.db import Database

log = logging.getLogger("agents_kg.api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _get_db_path() -> str:
    return os.environ.get("KG_DB_PATH", "pipeline.db")


def _get_neo4j_config():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")
    return uri, (user, password)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = _get_db_path()
    app.state.db = Database(db_path)
    log.info("SQLite database: %s", db_path)

    recovered = app.state.db.reset_stalled_jobs()
    if recovered:
        log.info("Recovered %d stalled jobs (stuck in processing >30min)", recovered)

    neo4j_uri, neo4j_auth = _get_neo4j_config()
    app.state.neo4j_driver = None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        driver.verify_connectivity()
        app.state.neo4j_driver = driver
        log.info("Neo4j connected: %s", neo4j_uri)
    except Exception as e:
        log.warning("Neo4j not available (%s), query endpoint will be disabled", e)

    yield

    app.state.db.close()
    if app.state.neo4j_driver:
        app.state.neo4j_driver.close()


app = FastAPI(
    title="agents-kg API",
    description="REST API for the agents-kg knowledge graph ingestion pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .auth_routes import router as auth_router  # noqa: E402
from .routes import router  # noqa: E402

app.include_router(auth_router)
app.include_router(router)

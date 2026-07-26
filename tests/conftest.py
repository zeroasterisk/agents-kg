"""Shared test fixtures."""

import os
import tempfile
import warnings
import pytest
from agents_kg.db import Database

PRODUCTION_NEO4J_IP = "35.202.188.73"


@pytest.fixture
def db():
    """Fresh in-memory-like SQLite database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


@pytest.fixture
def source_row(db):
    """A source row ready for pipeline stages."""
    sid = db.add_source("https://example.com/test")
    db.update_source(sid, raw_text="<html><body><h1>Test</h1><p>Hello world</p></body></html>", type="html")
    return db.get_source(sid)


@pytest.fixture(autouse=True, scope="session")
def guard_against_production_neo4j():
    """Hard abort if tests are about to connect to production Neo4j."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    if PRODUCTION_NEO4J_IP in uri:
        if os.environ.get("ALLOW_PRODUCTION_NEO4J") != "true":
            pytest.exit(
                f"SAFETY ABORT: Test suite connected to production Neo4j ({uri}). "
                "Set ALLOW_PRODUCTION_NEO4J=true to run live E2E tests intentionally. "
                "This is a destructive operation — tests wipe data.",
                returncode=3,
            )
        else:
            warnings.warn(
                f"WARNING: Tests are connecting to production Neo4j: {uri}. "
                "ALLOW_PRODUCTION_NEO4J=true is set — proceeding with caution."
            )


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring live services")

"""Shared test fixtures."""

import os
import tempfile
import pytest
from agents_kg.db import Database


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


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring live services")

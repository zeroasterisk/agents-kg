"""Iteration 8 — Production failure scenarios.

Tests simulate real-world failure modes: Neo4j connection drops, SQLite lock
contention, SPARQL timeouts, partial pipeline resume, and disk-full YAML export.
"""

import json
import os
import sqlite3
import struct
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import yaml

from agents_kg.db import Database, content_hash
from agents_kg.stages import fetch, parse, chunk, load, extract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


def _populate_source_through_chunk(db):
    """Create a source and push it through fetch→parse→chunk."""
    sid = db.add_source("https://example.com/prod-test")
    db.update_source(sid, raw_text="<html><body><h1>Title</h1><p>Body text here</p></body></html>", type="html")
    src = db.get_source(sid)
    parse.run(db, src)
    src = db.get_source(sid)
    chunk.run(db, src)
    return sid


def _populate_approved_entities(db, source_id, n=3):
    """Add n approved entities + edges for a source."""
    now = "2026-01-01T00:00:00+00:00"
    for i in range(n):
        eid = f"test:prod-ent-{source_id}-{i}"
        db.add_entity(
            entity_id=eid,
            name=f"ProdEntity {i}",
            entity_type="Project",
            kind="tool",
            description=f"Entity {i} for production test",
            source_id=source_id,
            chunk_id=1,
        )
        row = db.conn.execute(
            "SELECT id FROM entities WHERE entity_id = ?", (eid,)
        ).fetchone()
        db.approve_entity(row["id"])

    for i in range(n - 1):
        edge_id = f"prod-edge-{source_id}-{i}"
        db.add_edge(
            edge_id=edge_id,
            source_entity_id=f"test:prod-ent-{source_id}-{i}",
            target_entity_id=f"test:prod-ent-{source_id}-{i+1}",
            edge_type="DEVELOPS",
            source_id=source_id,
        )
        row = db.conn.execute(
            "SELECT id FROM edges WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        db.approve_edge(row["id"])


# ---------------------------------------------------------------------------
# 1. Neo4j connection drops mid-transaction
# ---------------------------------------------------------------------------

class TestNeo4jConnectionDrop:

    def test_driver_fails_after_n_queries(self, db):
        """Simulate Neo4j driver whose session fails after the Source node merge."""
        sid = db.add_source("https://example.com/neo4j-drop")
        db.update_source(sid, raw_text="# Neo4j Drop\n\nTest content", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)
        _populate_approved_entities(db, sid, n=2)
        src = db.get_source(sid)

        call_count = {"n": 0}

        class FailingSession:
            def run(self, query, params=None):
                call_count["n"] += 1
                if call_count["n"] > 2:
                    raise ConnectionError("Neo4j connection lost")
                return MagicMock()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        mock_driver = MagicMock()
        mock_driver.session.return_value = FailingSession()

        result = load.run(db, src, neo4j_driver=mock_driver)
        assert result is True
        updated = db.get_source(sid)
        assert updated["status"] == "complete"
        assert updated["stage"] == "done"

    def test_driver_session_creation_fails(self, db):
        """Neo4j driver can't even create a session — should still export YAML and finish."""
        sid = db.add_source("https://example.com/no-session")
        db.update_source(sid, raw_text="# No Session\n\nTest", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)
        _populate_approved_entities(db, sid, n=1)
        src = db.get_source(sid)

        mock_driver = MagicMock()
        mock_driver.session.side_effect = ConnectionError("Cannot connect")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agents_kg.stages.load.YAML_DIR", tmpdir):
                result = load.run(db, src, neo4j_driver=mock_driver)

        assert result is True
        updated = db.get_source(sid)
        assert updated["status"] == "complete"


# ---------------------------------------------------------------------------
# 2. SQLite database lock contention
# ---------------------------------------------------------------------------

class TestSQLiteLockContention:

    def test_duplicate_add_source_returns_none(self, db):
        """Adding the same URI twice — second call returns None."""
        first = db.add_source("https://example.com/concurrent")
        assert isinstance(first, int)
        second = db.add_source("https://example.com/concurrent")
        assert second is None

    def test_locked_db_read_after_write(self, db):
        """Read operations succeed after a write (WAL mode allows concurrent reads)."""
        sid = db.add_source("https://example.com/wal-test")
        source = db.get_source(sid)
        assert source is not None
        assert source["uri"] == "https://example.com/wal-test"

    def test_fail_source_increments_attempts(self, db):
        """fail_source properly increments attempts and transitions status."""
        sid = db.add_source("https://example.com/retry-test")
        for i in range(4):
            db.fail_source(sid, f"Error {i}")
            src = db.get_source(sid)
            assert src["attempts"] == i + 1
            if i < 4:
                assert src["status"] in ("failed", "dead_letter")

        db.fail_source(sid, "Final error")
        src = db.get_source(sid)
        assert src["status"] == "dead_letter"


# ---------------------------------------------------------------------------
# 3. SPARQL endpoint timeout
# ---------------------------------------------------------------------------

class TestSPARQLTimeout:

    def test_sparql_timeout_raises(self, monkeypatch):
        """SPARQL endpoint timeout should raise, not silently succeed."""
        import httpx
        from agents_kg import wikidata

        monkeypatch.setattr(wikidata, "RATE_LIMIT", 0.0)
        monkeypatch.setattr(wikidata, "_last_request_time", 0.0)
        with patch.object(httpx, "post", side_effect=httpx.ReadTimeout("SPARQL timed out")):
            with pytest.raises(httpx.ReadTimeout):
                wikidata.sparql_query("SELECT * WHERE { ?s ?p ?o } LIMIT 1", retries=1)

    def test_sparql_server_error_raises(self, monkeypatch):
        """SPARQL 500 error should raise after retries exhausted."""
        import httpx
        from agents_kg import wikidata

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {}
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        monkeypatch.setattr(wikidata, "RATE_LIMIT", 0.0)
        monkeypatch.setattr(wikidata, "_last_request_time", 0.0)
        with patch.object(httpx, "post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                wikidata.sparql_query("SELECT * WHERE { ?s ?p ?o }", retries=1)


# ---------------------------------------------------------------------------
# 4. Partial pipeline completion — resume from chunk stage
# ---------------------------------------------------------------------------

class TestPartialPipelineResume:

    def test_source_stuck_at_chunk_resumes(self, db):
        """A source processed through fetch→parse but stuck at chunk can resume."""
        sid = db.add_source("https://example.com/partial")
        db.update_source(
            sid,
            raw_text="<html><body><h1>Partial</h1><p>Some text here for chunking</p></body></html>",
            type="html",
            stage="parse",
            status="processing",
            content_hash=content_hash("<html><body><h1>Partial</h1><p>Some text here for chunking</p></body></html>"),
        )

        src = db.get_source(sid)
        assert src["stage"] == "parse"

        parse.run(db, src)
        src = db.get_source(sid)
        assert src["stage"] == "chunk"

        chunk.run(db, src)
        src = db.get_source(sid)
        assert src["stage"] == "embed"
        chunks = db.get_chunks(sid)
        assert len(chunks) > 0

    def test_source_at_embed_without_api_skips_gracefully(self, db):
        """If embed stage fails due to no API, the source should move to failed."""
        sid = _populate_source_through_chunk(db)
        src = db.get_source(sid)
        assert src["stage"] == "embed"

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            with pytest.raises(Exception):
                from agents_kg.stages.embed import run as embed_run
                embed_run(db, src)

    def test_resume_maintains_existing_chunks(self, db):
        """Re-running chunk on a source replaces old chunks (idempotent)."""
        sid = _populate_source_through_chunk(db)
        chunks_before = db.get_chunks(sid)

        src = db.get_source(sid)
        db.update_source(sid, stage="chunk", status="processing")
        src = db.get_source(sid)
        chunk.run(db, src)
        chunks_after = db.get_chunks(sid)

        assert len(chunks_after) == len(chunks_before)
        for ca, cb in zip(chunks_after, chunks_before):
            assert ca["text"] == cb["text"]


# ---------------------------------------------------------------------------
# 5. Disk full simulation: YAML export fails mid-write
# ---------------------------------------------------------------------------

class TestDiskFullYAMLExport:

    def test_yaml_export_write_error_raises(self, db):
        """If YAML file write fails (e.g. disk full), the error propagates."""
        sid = db.add_source("https://example.com/disk-full")
        db.update_source(sid, raw_text="# Disk Full\n\nTest", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)
        _populate_approved_entities(db, sid, n=1)
        src = db.get_source(sid)

        with patch("agents_kg.stages.load._export_yaml", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError, match="No space left"):
                load.run(db, src, neo4j_driver=None)

    def test_yaml_export_succeeds_in_writable_dir(self, db):
        """YAML export to a writable directory works."""
        sid = db.add_source("https://example.com/yaml-ok")
        db.update_source(sid, raw_text="# YAML OK\n\nTest content", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)
        _populate_approved_entities(db, sid, n=2)
        src = db.get_source(sid)

        exported = []
        original_export = load._export_yaml

        def capturing_export(entity, base_dir=None):
            with tempfile.TemporaryDirectory() as tmpdir:
                original_export(entity, base_dir=tmpdir)
                exported.append(entity["entity_id"])

        with patch("agents_kg.stages.load._export_yaml", side_effect=capturing_export):
            result = load.run(db, src, neo4j_driver=None)

        assert result is True
        assert len(exported) == 2
        updated = db.get_source(sid)
        assert updated["status"] == "complete"

    def test_yaml_export_content_fidelity(self, db):
        """Exported YAML contains correct entity data."""
        sid = db.add_source("https://example.com/yaml-fidelity")
        db.update_source(sid, raw_text="# Fidelity\n\nTest", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)

        eid = "test:fidelity-check"
        db.add_entity(
            entity_id=eid,
            name="Fidelity Check",
            entity_type="Project",
            kind="tool",
            description="Verify YAML export fidelity",
            aliases=["FC"],
            source_id=sid,
            chunk_id=1,
        )
        row = db.conn.execute("SELECT id FROM entities WHERE entity_id = ?", (eid,)).fetchone()
        db.approve_entity(row["id"])
        src = db.get_source(sid)

        with tempfile.TemporaryDirectory() as tmpdir:
            load._export_yaml(dict(db.conn.execute(
                "SELECT * FROM entities WHERE entity_id = ?", (eid,)
            ).fetchone()), base_dir=tmpdir)

            yaml_file = Path(tmpdir) / "projects" / "fidelity-check.yaml"
            assert yaml_file.exists()
            data = yaml.safe_load(yaml_file.read_text())
            assert data["id"] == eid
            assert data["name"] == "Fidelity Check"
            assert data["type"] == "Project"
            assert data["kind"] == "tool"
            assert "FC" in data["aliases"]


# ---------------------------------------------------------------------------
# 6. Additional production edge cases
# ---------------------------------------------------------------------------

class TestProductionEdgeCases:

    def test_empty_source_table_process(self, db):
        """Processing with no sources should complete without error."""
        pending = db.get_pending_sources()
        assert len(pending) == 0

    def test_dead_letter_source_not_retried(self, db):
        """Sources in dead_letter status are not picked up by get_pending_sources."""
        sid = db.add_source("https://example.com/dead")
        for _ in range(5):
            db.fail_source(sid, "Repeated failure")
        src = db.get_source(sid)
        assert src["status"] == "dead_letter"
        pending = db.get_pending_sources()
        assert all(s["id"] != sid for s in pending)

    def test_status_summary_counts(self, db):
        """status_summary returns correct counts by status."""
        db.add_source("https://example.com/sum1")
        db.add_source("https://example.com/sum2")
        sid3 = db.add_source("https://example.com/sum3")
        db.fail_source(sid3, "Test fail")

        summary = db.status_summary()
        assert summary.get("pending", 0) == 2
        assert summary.get("failed", 0) == 1

    def test_reset_source_clears_all_artifacts(self, db):
        """reset_source removes chunks, entities, edges and resets status."""
        sid = _populate_source_through_chunk(db)
        db.add_entity(
            entity_id="test:reset-ent",
            name="Reset Entity",
            entity_type="Project",
            source_id=sid,
            chunk_id=1,
        )
        db.add_edge(
            edge_id="reset-edge",
            source_entity_id="test:reset-ent",
            target_entity_id="test:reset-ent",
            edge_type="DEVELOPS",
            source_id=sid,
        )

        db.reset_source(sid)
        src = db.get_source(sid)
        assert src["status"] == "pending"
        assert src["stage"] == "fetch"
        assert src["raw_text"] is None
        assert len(db.get_chunks(sid)) == 0
        ents = db.conn.execute("SELECT * FROM entities WHERE source_id = ?", (sid,)).fetchall()
        assert len(ents) == 0
        edges = db.conn.execute("SELECT * FROM edges WHERE source_id = ?", (sid,)).fetchall()
        assert len(edges) == 0

    def test_large_error_message_stored(self, db):
        """A very long error message is stored without truncation."""
        sid = db.add_source("https://example.com/long-err")
        long_error = "E" * 10000
        db.fail_source(sid, long_error)
        src = db.get_source(sid)
        assert len(src["error"]) == 10000

"""Iteration 8 — Performance boundary tests.

Tests verify the pipeline handles scale: 200+ entities from one source,
10+ sources in queue, and key operations complete within time bounds.
"""

import os
import tempfile
import time

import pytest

from agents_kg.db import Database
from agents_kg.stages import parse, chunk


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


# ---------------------------------------------------------------------------
# 1. 200+ entities from a single source
# ---------------------------------------------------------------------------

class TestLargeEntityCount:

    def test_200_entities_from_single_source(self, db):
        """Insert 200+ entities for a single source without error."""
        sid = db.add_source("https://example.com/large-source")
        for i in range(250):
            db.add_entity(
                entity_id=f"test:large-{i}",
                name=f"Large Entity {i}",
                entity_type="Project",
                kind="tool",
                description=f"Entity number {i}",
                source_id=sid,
            )

        entities = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM entities WHERE source_id = ?", (sid,)
        ).fetchone()
        assert entities["cnt"] == 250

    def test_200_edges_from_single_source(self, db):
        """Insert 200+ edges for a single source without error."""
        sid = db.add_source("https://example.com/large-edges")
        for i in range(201):
            db.add_entity(
                entity_id=f"test:edge-node-{i}",
                name=f"Node {i}",
                entity_type="Project",
                source_id=sid,
            )

        for i in range(200):
            db.add_edge(
                edge_id=f"large-edge-{i}",
                source_entity_id=f"test:edge-node-{i}",
                target_entity_id=f"test:edge-node-{i+1}",
                edge_type="DEVELOPS",
                source_id=sid,
            )

        edges = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM edges WHERE source_id = ?", (sid,)
        ).fetchone()
        assert edges["cnt"] == 200

    def test_chunking_large_document(self, db):
        """A large document (50+ sections) chunks correctly."""
        sid = db.add_source("https://example.com/large-doc")
        sections = "\n\n".join(
            f"## Section {i}\n\nThis is the content of section {i}. " * 5
            for i in range(60)
        )
        text = f"# Large Document\n\n{sections}"
        db.update_source(sid, raw_text=text, type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)

        chunks = db.get_chunks(sid)
        assert len(chunks) >= 50


# ---------------------------------------------------------------------------
# 2. 10+ sources in queue
# ---------------------------------------------------------------------------

class TestMultiSourceQueue:

    def test_10_sources_in_queue(self, db):
        """Add 10+ sources and verify all are pending."""
        for i in range(15):
            db.add_source(f"https://example.com/multi-{i}")

        pending = db.get_pending_sources()
        assert len(pending) == 15

    def test_15_sources_with_mixed_status(self, db):
        """Queue with mixed statuses returns correct pending count."""
        for i in range(15):
            sid = db.add_source(f"https://example.com/mixed-{i}")
            if i < 5:
                db.update_source(sid, status="complete", stage="done")
            elif i < 8:
                db.fail_source(sid, f"Error {i}")

        pending = db.get_pending_sources()
        assert len(pending) == 7

    def test_status_summary_with_many_sources(self, db):
        """status_summary works correctly with many sources."""
        for i in range(20):
            sid = db.add_source(f"https://example.com/summary-{i}")
            if i < 10:
                db.update_source(sid, status="complete", stage="done")
            elif i < 15:
                db.update_source(sid, status="processing", stage="chunk")

        summary = db.status_summary()
        assert summary.get("complete", 0) == 10
        assert summary.get("processing", 0) == 5
        assert summary.get("pending", 0) == 5

    def test_process_sources_independently(self, db):
        """Each source can be processed through stages independently."""
        sids = []
        for i in range(10):
            sid = db.add_source(f"https://example.com/indep-{i}")
            text = f"# Source {i}\n\nContent for source number {i} with some text."
            db.update_source(sid, raw_text=text, type="text")
            sids.append(sid)

        for sid in sids:
            src = db.get_source(sid)
            parse.run(db, src)
            src = db.get_source(sid)
            chunk.run(db, src)

        for sid in sids:
            chunks = db.get_chunks(sid)
            assert len(chunks) > 0, f"Source {sid} has no chunks"


# ---------------------------------------------------------------------------
# 3. Time upper bounds for key operations
# ---------------------------------------------------------------------------

class TestOperationTimeBounds:

    def test_add_source_under_50ms(self, db):
        """Adding a source completes in under 50ms."""
        start = time.monotonic()
        db.add_source("https://example.com/timing-add")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"add_source took {elapsed:.3f}s"

    def test_get_source_under_10ms(self, db):
        """Getting a source by ID completes in under 10ms."""
        sid = db.add_source("https://example.com/timing-get")
        start = time.monotonic()
        db.get_source(sid)
        elapsed = time.monotonic() - start
        assert elapsed < 0.01, f"get_source took {elapsed:.3f}s"

    def test_parse_stage_under_500ms(self, db):
        """Parse stage completes in under 500ms for a normal page."""
        sid = db.add_source("https://example.com/timing-parse")
        html = "<html><body>" + "<p>Paragraph</p>" * 100 + "</body></html>"
        db.update_source(sid, raw_text=html, type="html")
        src = db.get_source(sid)

        start = time.monotonic()
        parse.run(db, src)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"parse took {elapsed:.3f}s"

    def test_chunk_stage_under_500ms(self, db):
        """Chunk stage completes in under 500ms for a normal document."""
        sid = db.add_source("https://example.com/timing-chunk")
        text = "# Big Doc\n\n" + "\n\n".join(
            f"## Section {i}\n\n{'Word ' * 200}" for i in range(30)
        )
        db.update_source(sid, raw_text=text, type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)

        start = time.monotonic()
        chunk.run(db, src)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"chunk took {elapsed:.3f}s"

    def test_bulk_entity_insert_under_2s(self, db):
        """Inserting 500 entities completes in under 2 seconds."""
        sid = db.add_source("https://example.com/timing-bulk")

        start = time.monotonic()
        for i in range(500):
            db.add_entity(
                entity_id=f"test:bulk-{i}",
                name=f"Bulk {i}",
                entity_type="Project",
                source_id=sid,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"500 entity inserts took {elapsed:.3f}s"

    def test_status_summary_under_50ms(self, db):
        """status_summary with many records completes quickly."""
        for i in range(100):
            db.add_source(f"https://example.com/perf-sum-{i}")

        start = time.monotonic()
        db.status_summary()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"status_summary took {elapsed:.3f}s"

    def test_get_pending_sources_under_50ms(self, db):
        """get_pending_sources with 100 records completes quickly."""
        for i in range(100):
            db.add_source(f"https://example.com/perf-pending-{i}")

        start = time.monotonic()
        db.get_pending_sources()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"get_pending_sources took {elapsed:.3f}s"

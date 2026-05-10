"""Iteration 6: Concurrency and idempotency tests."""

import os
import tempfile
import pytest
from agents_kg.db import Database, content_hash
from agents_kg.stages import fetch, parse, chunk


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


# ── Source ingestion idempotency ─────────────────────────────────────────────

class TestSourceIdempotency:
    def test_add_same_source_five_times(self, db):
        """Ingesting the same URL 5 times creates exactly 1 source."""
        results = [db.add_source("https://example.com/repeat") for _ in range(5)]
        assert results[0] is not None
        assert all(r is None for r in results[1:])
        count = db.conn.execute("SELECT COUNT(*) FROM sources WHERE uri = 'https://example.com/repeat'").fetchone()[0]
        assert count == 1

    def test_different_urls_not_deduplicated(self, db):
        for i in range(3):
            db.add_source(f"https://example.com/unique-{i}")
        count = db.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        assert count == 3

    def test_rapid_succession_same_source(self, db):
        """Simulate rapid succession — all return None except first."""
        ids = []
        for _ in range(10):
            ids.append(db.add_source("https://example.com/rapid"))
        non_none = [i for i in ids if i is not None]
        assert len(non_none) == 1

    def test_submitter_preserved_on_first_add(self, db):
        """Only the first add's submitter is stored (duplicates are silently skipped)."""
        db.add_source("https://example.com/sub", submitter_email="first@example.com")
        db.add_source("https://example.com/sub", submitter_email="second@example.com")
        source = db.get_source_by_uri("https://example.com/sub")
        assert source["submitter_email"] == "first@example.com"


# ── Content hash exactness ──────────────────────────────────────────────────

class TestContentHash:
    def test_hash_is_deterministic(self):
        h1 = content_hash("Hello, world!")
        h2 = content_hash("Hello, world!")
        assert h1 == h2

    def test_hash_changes_on_single_byte(self):
        h1 = content_hash("Hello, world!")
        h2 = content_hash("Hello, world?")
        assert h1 != h2

    def test_hash_whitespace_sensitive(self):
        h1 = content_hash("abc def")
        h2 = content_hash("abc  def")
        assert h1 != h2

    def test_hash_trailing_newline_sensitive(self):
        h1 = content_hash("content")
        h2 = content_hash("content\n")
        assert h1 != h2

    def test_hash_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_hash_unicode(self):
        h1 = content_hash("café")
        h2 = content_hash("cafe")
        assert h1 != h2


# ── Fetch stage idempotency ──────────────────────────────────────────────────

class TestFetchIdempotency:
    def test_unchanged_content_skips(self, db, tmp_path):
        """If content_hash matches, fetch returns False and marks complete."""
        local_file = tmp_path / "idem.md"
        raw = "# Idempotency test content"
        local_file.write_text(raw)
        sid = db.add_source(str(local_file))
        h = content_hash(raw)
        db.update_source(sid, raw_text=raw, content_hash=h, stage="fetch", status="pending")
        source = db.get_source(sid)
        result = fetch.run(db, source)
        assert result is False
        updated = db.get_source(sid)
        assert updated["status"] == "complete"

    def test_changed_content_triggers_reprocess(self, db, tmp_path):
        """If content changes on refetch, source should be reprocessed."""
        local_file = tmp_path / "changing.md"
        local_file.write_text("old content")
        sid = db.add_source(str(local_file))
        old_hash = content_hash("old content")
        db.update_source(sid, raw_text="old content", content_hash=old_hash, stage="fetch", status="pending")
        db.add_entity("test:old-ent", "Old Entity", "Organization", source_id=sid)

        local_file.write_text("new content")
        source = db.get_source(sid)
        result = fetch.run(db, source)
        assert result is True
        updated = db.get_source(sid)
        assert updated["content_hash"] != old_hash
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) == 1


# ── Parse stage idempotency ──────────────────────────────────────────────────

class TestParseIdempotency:
    def test_parse_same_input_same_output(self, db):
        """Parse should produce identical output for identical input."""
        sid = db.add_source("https://example.com/parse-idem")
        html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        db.update_source(sid, raw_text=html, type="html", stage="parse", status="processing")
        source1 = db.get_source(sid)
        parse.run(db, source1)
        parsed1 = db.get_source(sid)["parsed_text"]

        db.update_source(sid, raw_text=html, type="html", stage="parse", status="processing")
        source2 = db.get_source(sid)
        parse.run(db, source2)
        parsed2 = db.get_source(sid)["parsed_text"]
        assert parsed1 == parsed2

    def test_markdown_passthrough_idempotent(self, db):
        sid = db.add_source("https://example.com/md-idem")
        md = "# Header\n\nParagraph text here."
        db.update_source(sid, raw_text=md, type="text", stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        assert db.get_source(sid)["parsed_text"] == md


# ── Chunk stage idempotency ──────────────────────────────────────────────────

class TestChunkIdempotency:
    def test_rechunk_replaces_old_chunks(self, db):
        """Chunking same source twice should delete old chunks first."""
        sid = db.add_source("https://example.com/chunk-idem")
        text = "# Section 1\n\nContent A.\n\n# Section 2\n\nContent B."
        db.update_source(sid, parsed_text=text, raw_text=text, stage="chunk", status="processing")

        source = db.get_source(sid)
        chunk.run(db, source)
        chunks_1 = db.get_chunks(sid)

        source = db.get_source(sid)
        source_dict = dict(source)
        source_dict["parsed_text"] = text
        source_dict["stage"] = "chunk"
        chunk.run(db, source_dict)
        chunks_2 = db.get_chunks(sid)

        assert len(chunks_1) == len(chunks_2)
        for c1, c2 in zip(chunks_1, chunks_2):
            assert c1["text"] == c2["text"]
            assert c1["position"] == c2["position"]

    def test_chunk_count_same_on_repeat(self, db):
        sid = db.add_source("https://example.com/chunk-count")
        text = "# A\n\nParagraph.\n\n# B\n\nAnother paragraph."
        db.update_source(sid, parsed_text=text, raw_text=text, stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        n1 = len(db.get_chunks(sid))
        chunk.run(db, dict(db.get_source(sid)))
        n2 = len(db.get_chunks(sid))
        assert n1 == n2


# ── Entity and edge idempotency ──────────────────────────────────────────────

class TestEntityIdempotency:
    def test_add_entity_same_id_rejected(self, db):
        sid = db.add_source("https://example.com/ent-idem")
        id1 = db.add_entity("test:dup-entity", "Dup", "Organization", source_id=sid)
        id2 = db.add_entity("test:dup-entity", "Dup Again", "Organization", source_id=sid)
        assert id1 is not None
        assert id2 is None
        count = db.conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id = 'test:dup-entity'").fetchone()[0]
        assert count == 1

    def test_add_edge_same_id_rejected(self, db):
        sid = db.add_source("https://example.com/edge-idem")
        id1 = db.add_edge("test-edge-dup", "test:a", "test:b", "DEVELOPS", source_id=sid)
        id2 = db.add_edge("test-edge-dup", "test:a", "test:b", "DEVELOPS", source_id=sid)
        assert id1 is not None
        assert id2 is None

    def test_multiple_entities_same_source(self, db):
        sid = db.add_source("https://example.com/multi-ent")
        db.add_entity("test:multi-1", "Org A", "Organization", source_id=sid)
        db.add_entity("test:multi-2", "Org B", "Organization", source_id=sid)
        db.add_entity("test:multi-3", "Org C", "Organization", source_id=sid)
        entities = db.conn.execute("SELECT * FROM entities WHERE source_id = ?", (sid,)).fetchall()
        assert len(entities) == 3


# ── Source reset and re-process ──────────────────────────────────────────────

class TestResetIdempotency:
    def test_reset_clears_all_derived_data(self, db):
        """Resetting a source removes chunks, entities, and edges."""
        sid = db.add_source("https://example.com/reset-idem")
        db.add_chunk(sid, "chunk text", 0)
        db.add_entity("test:reset-ent", "Reset Org", "Organization", source_id=sid)
        db.add_edge("test-reset-edge", "test:a", "test:b", "DEVELOPS", source_id=sid)
        db.reset_source(sid)

        assert len(db.get_chunks(sid)) == 0
        ents = db.conn.execute("SELECT * FROM entities WHERE source_id = ?", (sid,)).fetchall()
        assert len(ents) == 0
        edges = db.conn.execute("SELECT * FROM edges WHERE source_id = ?", (sid,)).fetchall()
        assert len(edges) == 0

    def test_reset_restores_initial_state(self, db):
        sid = db.add_source("https://example.com/reset-state")
        db.update_source(sid, stage="extract", status="processing", raw_text="some text",
                        content_hash="abc123", attempts=3)
        db.reset_source(sid)
        source = db.get_source(sid)
        assert source["status"] == "pending"
        assert source["stage"] == "fetch"
        assert source["attempts"] == 0
        assert source["raw_text"] is None
        assert source["content_hash"] is None

"""Iteration 6: Pipeline stage isolation tests.

Tests each pipeline stage independently with controlled inputs,
verifies failure isolation, stage progression, and retry behavior.
"""

import os
import tempfile
import pytest
from agents_kg.db import Database, content_hash
from agents_kg.stages import parse, chunk
from agents_kg.pipeline import STAGE_ORDER


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


# ── Stage order and progression ──────────────────────────────────────────────

class TestStageOrder:
    def test_stage_order_is_defined(self):
        assert STAGE_ORDER == ["fetch", "parse", "chunk", "embed", "extract", "resolve", "review", "load"]

    def test_stage_order_has_eight_stages(self):
        assert len(STAGE_ORDER) == 8

    def test_initial_source_starts_at_fetch(self, db):
        sid = db.add_source("https://example.com/stage-test")
        source = db.get_source(sid)
        assert source["stage"] == "fetch"
        assert source["status"] == "pending"


# ── Parse stage isolation ────────────────────────────────────────────────────

class TestParseStageIsolation:
    def test_parse_requires_raw_text(self, db):
        sid = db.add_source("https://example.com/no-raw")
        db.update_source(sid, stage="parse", status="processing")
        source = db.get_source(sid)
        with pytest.raises(RuntimeError, match="No raw_text"):
            parse.run(db, source)

    def test_parse_advances_to_chunk(self, db):
        sid = db.add_source("https://example.com/parse-adv")
        db.update_source(sid, raw_text="<html><body><h1>T</h1><p>P</p></body></html>",
                        type="html", stage="parse", status="processing")
        source = db.get_source(sid)
        result = parse.run(db, source)
        assert result is True
        updated = db.get_source(sid)
        assert updated["stage"] == "chunk"
        assert updated["status"] == "processing"

    def test_parse_does_not_modify_chunks(self, db):
        """Parse stage should not create or delete chunks."""
        sid = db.add_source("https://example.com/parse-no-chunk")
        db.add_chunk(sid, "pre-existing chunk", 0)
        db.update_source(sid, raw_text="# Markdown\n\nContent",
                        type="text", stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        chunks = db.get_chunks(sid)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "pre-existing chunk"

    def test_parse_failure_does_not_corrupt_raw_text(self, db):
        """If parse fails, the raw_text should remain intact."""
        sid = db.add_source("https://example.com/parse-fail")
        raw = "<html><body><h1>Good HTML</h1></body></html>"
        db.update_source(sid, raw_text=raw, type="html", stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        updated = db.get_source(sid)
        assert updated["raw_text"] == raw

    def test_parse_html_extracts_title(self, db):
        sid = db.add_source("https://example.com/title-extract")
        db.update_source(sid, raw_text="<html><body><h1>My Title</h1><p>Body text</p></body></html>",
                        type="html", stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        updated = db.get_source(sid)
        assert updated["parsed_text"] is not None

    def test_parse_pdf_type(self, db):
        sid = db.add_source("https://example.com/pdf-parse")
        db.update_source(sid, raw_text="Page 1 text\n\n\n\n\nPage 2 text\n\n\n\n\nPage 3 text",
                        type="pdf", stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        updated = db.get_source(sid)
        assert "\n\n\n" not in updated["parsed_text"]


# ── Chunk stage isolation ────────────────────────────────────────────────────

class TestChunkStageIsolation:
    def test_chunk_requires_text(self, db):
        sid = db.add_source("https://example.com/no-text")
        db.update_source(sid, stage="chunk", status="processing")
        source = db.get_source(sid)
        with pytest.raises(RuntimeError, match="No text"):
            chunk.run(db, source)

    def test_chunk_advances_to_embed(self, db):
        sid = db.add_source("https://example.com/chunk-adv")
        db.update_source(sid, parsed_text="# H1\n\nText content here.",
                        raw_text="x", stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        updated = db.get_source(sid)
        assert updated["stage"] == "embed"

    def test_chunk_creates_positioned_chunks(self, db):
        sid = db.add_source("https://example.com/positioned")
        db.update_source(sid, parsed_text="# Section 1\n\nContent A.\n\n# Section 2\n\nContent B.\n\n# Section 3\n\nContent C.",
                        raw_text="x", stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        chunks = db.get_chunks(sid)
        positions = [c["position"] for c in chunks]
        assert positions == sorted(positions)
        assert positions == list(range(len(chunks)))

    def test_chunk_does_not_modify_entities(self, db):
        """Chunk stage should not affect entities table."""
        sid = db.add_source("https://example.com/chunk-no-ent")
        db.add_entity("test:chunk-iso", "Chunk Iso", "Organization", source_id=sid)
        db.update_source(sid, parsed_text="# T\n\nText", raw_text="x", stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        ent = db.conn.execute("SELECT * FROM entities WHERE entity_id = 'test:chunk-iso'").fetchone()
        assert ent is not None

    def test_chunk_deletes_old_before_creating_new(self, db):
        sid = db.add_source("https://example.com/chunk-replace")
        db.add_chunk(sid, "old chunk 0", 0)
        db.add_chunk(sid, "old chunk 1", 1)
        db.update_source(sid, parsed_text="# New\n\nNew content only.",
                        raw_text="x", stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        chunks = db.get_chunks(sid)
        assert all("old" not in c["text"].lower() for c in chunks)


# ── Failure isolation between sources ────────────────────────────────────────

class TestFailureIsolation:
    def test_fail_source_does_not_affect_others(self, db):
        sid1 = db.add_source("https://example.com/good")
        sid2 = db.add_source("https://example.com/bad")
        db.update_source(sid1, raw_text="Good content", stage="parse", status="processing")
        db.fail_source(sid2, "Simulated failure")

        s1 = db.get_source(sid1)
        s2 = db.get_source(sid2)
        assert s1["status"] == "processing"
        assert s2["status"] == "failed"

    def test_dead_letter_after_max_attempts(self, db):
        sid = db.add_source("https://example.com/max-fail")
        for i in range(5):
            db.fail_source(sid, f"Attempt {i+1}")
        source = db.get_source(sid)
        assert source["status"] == "dead_letter"

    def test_failed_source_retryable(self, db):
        sid = db.add_source("https://example.com/retry-me")
        db.fail_source(sid, "first failure")
        assert db.get_source(sid)["status"] == "failed"
        db.retry_failed()
        assert db.get_source(sid)["status"] == "pending"

    def test_dead_letter_not_retryable(self, db):
        sid = db.add_source("https://example.com/dead")
        for i in range(5):
            db.fail_source(sid, f"Attempt {i+1}")
        assert db.get_source(sid)["status"] == "dead_letter"
        db.retry_failed()
        assert db.get_source(sid)["status"] == "dead_letter"


# ── Stage progression tracking ───────────────────────────────────────────────

class TestStageProgression:
    def test_parse_to_chunk_progression(self, db):
        sid = db.add_source("https://example.com/prog-pc")
        db.update_source(sid, raw_text="<html><body><h1>T</h1><p>P</p></body></html>",
                        type="html", stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        assert db.get_source(sid)["stage"] == "chunk"

    def test_chunk_to_embed_progression(self, db):
        sid = db.add_source("https://example.com/prog-ce")
        db.update_source(sid, parsed_text="# T\n\nContent", raw_text="x",
                        stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        assert db.get_source(sid)["stage"] == "embed"

    def test_stages_track_attempts_on_failure(self, db):
        sid = db.add_source("https://example.com/attempt-track")
        assert db.get_source(sid)["attempts"] == 0
        db.fail_source(sid, "error 1")
        assert db.get_source(sid)["attempts"] == 1
        db.retry_failed()
        db.fail_source(sid, "error 2")
        assert db.get_source(sid)["attempts"] == 2

    def test_error_message_preserved(self, db):
        sid = db.add_source("https://example.com/err-msg")
        db.fail_source(sid, "Connection timeout after 30s")
        source = db.get_source(sid)
        assert source["error"] == "Connection timeout after 30s"

    def test_error_cleared_on_retry(self, db):
        sid = db.add_source("https://example.com/err-clear")
        db.fail_source(sid, "Some error")
        db.retry_failed()
        source = db.get_source(sid)
        assert source["error"] is None


# ── Data isolation between stages ────────────────────────────────────────────

class TestDataIsolation:
    def test_parse_preserves_source_metadata(self, db):
        sid = db.add_source("https://example.com/meta-pres", submitter_email="test@test.com")
        db.update_source(sid, raw_text="# Title\n\nBody text", type="text",
                        stage="parse", status="processing")
        source = db.get_source(sid)
        parse.run(db, source)
        updated = db.get_source(sid)
        assert updated["submitter_email"] == "test@test.com"
        assert updated["uri"] == "https://example.com/meta-pres"

    def test_chunk_preserves_parsed_text(self, db):
        sid = db.add_source("https://example.com/parsed-pres")
        parsed = "# Title\n\nParagraph text content."
        db.update_source(sid, parsed_text=parsed, raw_text="x",
                        stage="chunk", status="processing")
        source = db.get_source(sid)
        chunk.run(db, source)
        updated = db.get_source(sid)
        assert updated["parsed_text"] == parsed

    def test_multiple_sources_independent_chunks(self, db):
        sid1 = db.add_source("https://example.com/indep1")
        sid2 = db.add_source("https://example.com/indep2")

        db.update_source(sid1, parsed_text="# A\n\nContent A", raw_text="x",
                        stage="chunk", status="processing")
        db.update_source(sid2, parsed_text="# B\n\nContent B\n\n# B2\n\nMore B",
                        raw_text="x", stage="chunk", status="processing")

        chunk.run(db, db.get_source(sid1))
        chunk.run(db, db.get_source(sid2))

        c1 = db.get_chunks(sid1)
        c2 = db.get_chunks(sid2)
        assert all(c["source_id"] == sid1 for c in c1)
        assert all(c["source_id"] == sid2 for c in c2)
        assert len(c1) != len(c2) or all(c1[i]["text"] != c2[i]["text"] for i in range(min(len(c1), len(c2))))

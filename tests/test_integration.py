"""Integration tests: full pipeline with mocked external services."""

import json
import struct
import pytest
from unittest.mock import MagicMock, patch
from agents_kg.db import Database


class TestFullPipeline:
    """Source → fetch → parse → chunk → embed → extract with mocks."""

    def _mock_fetch(self, db, source):
        """Simulate fetch stage."""
        from agents_kg.stages.fetch import run
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><h1>A2A Protocol</h1><p>Google develops A2A, an agent-to-agent protocol. It competes with MCP by Anthropic.</p><h2>Details</h2><p>A2A enables discovery and communication between agents.</p></body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("agents_kg.stages.fetch.httpx.Client", return_value=mock_client):
            return run(db, source)

    def _mock_embed(self, db, source):
        """Simulate embed by writing fake embeddings."""
        source_id = source["id"]
        chunks = db.get_unembedded_chunks(source_id)
        for c in chunks:
            emb = struct.pack("3f", 0.1, 0.2, 0.3)
            db.update_chunk_embedding(c["id"], emb, "text-embedding-004")
        db.update_source(source_id, stage="extract", status="processing")
        return True

    def _mock_extract(self, db, source):
        """Simulate extract with canned response."""
        from agents_kg.stages.extract import _make_edge_id
        source_id = source["id"]
        chunks = db.get_chunks(source_id)

        entities = [
            {"entity_id": "organization:google", "name": "Google", "type": "Organization", "kind": "company"},
            {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol", "kind": "spec"},
            {"entity_id": "organization:anthropic", "name": "Anthropic", "type": "Organization", "kind": "company"},
        ]
        edges = [
            {"src": "organization:google", "tgt": "protocol:a2a", "type": "DEVELOPS", "conf": 0.9},
        ]

        for ent in entities:
            db.add_entity(
                entity_id=ent["entity_id"], name=ent["name"],
                entity_type=ent["type"], kind=ent.get("kind"),
                source_id=source_id, chunk_id=chunks[0]["id"] if chunks else None,
            )
        for e in edges:
            eid = _make_edge_id(e["src"], e["tgt"], e["type"])
            db.add_edge(eid, e["src"], e["tgt"], e["type"], confidence=e["conf"],
                       source_id=source_id, chunk_id=chunks[0]["id"] if chunks else None)

        db.update_source(source_id, stage="review", status="pending_review")
        return True

    def test_full_pipeline(self, db):
        """Run all stages with mocks, verify end state."""
        from agents_kg.stages.parse import run as parse_run
        from agents_kg.stages.chunk import run as chunk_run

        # Add source
        sid = db.add_source("https://example.com/a2a")
        source = db.get_source(sid)
        assert source["stage"] == "fetch"

        # Fetch
        self._mock_fetch(db, source)
        source = db.get_source(sid)
        assert source["stage"] == "parse"

        # Parse
        parse_run(db, source)
        source = db.get_source(sid)
        assert source["stage"] == "chunk"
        assert source["parsed_text"] is not None

        # Chunk
        chunk_run(db, source)
        source = db.get_source(sid)
        assert source["stage"] == "embed"
        chunks = db.get_chunks(sid)
        assert len(chunks) >= 1

        # Embed (mocked)
        self._mock_embed(db, source)
        source = db.get_source(sid)
        assert source["stage"] == "extract"

        # Extract (mocked)
        self._mock_extract(db, source)
        source = db.get_source(sid)
        assert source["stage"] == "review"
        assert source["status"] == "pending_review"

        # Verify entities and edges
        entities = db.get_entities_by_status("pending_review")
        assert len(entities) == 3
        edges = db.get_edges_by_status("pending_review")
        assert len(edges) == 1

    def test_stage_progression_in_db(self, db):
        """Verify stage field transitions correctly."""
        from agents_kg.stages.parse import run as parse_run
        from agents_kg.stages.chunk import run as chunk_run

        sid = db.add_source("https://example.com")
        stages_seen = [db.get_source(sid)["stage"]]

        source = db.get_source(sid)
        self._mock_fetch(db, source)
        stages_seen.append(db.get_source(sid)["stage"])

        source = db.get_source(sid)
        parse_run(db, source)
        stages_seen.append(db.get_source(sid)["stage"])

        source = db.get_source(sid)
        chunk_run(db, source)
        stages_seen.append(db.get_source(sid)["stage"])

        assert stages_seen == ["fetch", "parse", "chunk", "embed"]

    def test_failure_and_retry(self, db):
        """Simulate stage failure, verify retry behavior."""
        sid = db.add_source("https://example.com")

        # Fail it
        db.fail_source(sid, "network error")
        source = db.get_source(sid)
        assert source["status"] == "failed"
        assert source["attempts"] == 1

        # Retry
        count = db.retry_failed()
        assert count == 1
        source = db.get_source(sid)
        assert source["status"] == "pending"

        # Fail again multiple times -> dead letter
        for i in range(5):
            db.fail_source(sid, f"error {i}")
        source = db.get_source(sid)
        assert source["status"] == "dead_letter"

    def test_idempotency_no_duplicates(self, db):
        """Re-process same source, verify no duplicate entities."""
        from agents_kg.stages.parse import run as parse_run
        from agents_kg.stages.chunk import run as chunk_run

        sid = db.add_source("https://example.com/dup")
        source = db.get_source(sid)

        # First run
        self._mock_fetch(db, source)
        source = db.get_source(sid)
        parse_run(db, source)
        source = db.get_source(sid)
        chunk_run(db, source)
        source = db.get_source(sid)
        self._mock_embed(db, source)
        source = db.get_source(sid)
        self._mock_extract(db, source)

        entities_1 = db.get_entities_by_status("pending_review")
        count_1 = len(entities_1)

        # Reset and reprocess
        db.reset_source(sid)
        source = db.get_source(sid)
        self._mock_fetch(db, source)
        source = db.get_source(sid)
        parse_run(db, source)
        source = db.get_source(sid)
        chunk_run(db, source)
        source = db.get_source(sid)
        self._mock_embed(db, source)
        source = db.get_source(sid)
        self._mock_extract(db, source)

        entities_2 = db.get_entities_by_status("pending_review")
        # Should be same count (add_entity uses UNIQUE constraint on entity_id)
        assert len(entities_2) == count_1

    def test_duplicate_source_rejected(self, db):
        """Adding same URL twice returns None."""
        sid1 = db.add_source("https://example.com/dup")
        sid2 = db.add_source("https://example.com/dup")
        assert sid1 is not None
        assert sid2 is None

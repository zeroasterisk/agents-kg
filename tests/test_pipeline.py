"""Tests for the agents-kg pipeline."""

import pytest
from agents_kg.db import Database, content_hash


class TestDatabase:
    def test_add_source(self, db):
        sid = db.add_source("https://example.com")
        assert sid is not None
        source = db.get_source(sid)
        assert source["uri"] == "https://example.com"
        assert source["status"] == "pending"
        assert source["stage"] == "fetch"

    def test_duplicate_source(self, db):
        db.add_source("https://example.com")
        assert db.add_source("https://example.com") is None

    def test_fail_source(self, db):
        sid = db.add_source("https://example.com")
        db.fail_source(sid, "test error")
        s = db.get_source(sid)
        assert s["status"] == "failed"
        assert s["attempts"] == 1

    def test_dead_letter(self, db):
        sid = db.add_source("https://example.com")
        for i in range(5):
            db.fail_source(sid, f"error {i}")
        s = db.get_source(sid)
        assert s["status"] == "dead_letter"

    def test_retry_failed(self, db):
        sid = db.add_source("https://example.com")
        db.fail_source(sid, "err")
        count = db.retry_failed()
        assert count == 1
        s = db.get_source(sid)
        assert s["status"] == "pending"

    def test_reset_source(self, db):
        sid = db.add_source("https://example.com")
        db.update_source(sid, status="processing", stage="chunk", raw_text="hello")
        db.add_chunk(sid, "chunk text", 0)
        db.reset_source(sid)
        s = db.get_source(sid)
        assert s["status"] == "pending"
        assert s["stage"] == "fetch"
        assert db.get_chunks(sid) == []

    def test_status_summary(self, db):
        db.add_source("https://a.com")
        db.add_source("https://b.com")
        sid3 = db.add_source("https://c.com")
        db.fail_source(sid3, "err")
        summary = db.status_summary()
        assert summary["pending"] == 2
        assert summary["failed"] == 1

    def test_chunks(self, db):
        sid = db.add_source("https://example.com")
        cid = db.add_chunk(sid, "hello world", 0, section_heading="# Intro", token_count=3)
        chunks = db.get_chunks(sid)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello world"

    def test_entities(self, db):
        eid = db.add_entity("org:test", "Test Org", "Organization", kind="company")
        assert eid is not None
        entities = db.get_entities_by_status("pending_review")
        assert len(entities) == 1
        db.approve_entity(eid)
        assert db.get_entities_by_status("approved")[0]["name"] == "Test Org"

    def test_edges(self, db):
        eid = db.add_edge("e1", "org:a", "org:b", "MEMBER_OF", confidence=0.8)
        assert eid is not None
        edges = db.get_edges_by_status("pending_review")
        assert len(edges) == 1
        db.approve_edge(eid)
        assert db.get_edges_by_status("approved")[0]["edge_type"] == "MEMBER_OF"

    def test_content_hash(self):
        h1 = content_hash("hello")
        h2 = content_hash("hello")
        h3 = content_hash("world")
        assert h1 == h2
        assert h1 != h3


class TestChunking:
    def test_section_split(self):
        from agents_kg.stages.chunk import _split_sections
        text = "# Title\nSome text\n## Section 2\nMore text"
        sections = _split_sections(text)
        assert len(sections) == 2

    def test_estimate_tokens(self):
        from agents_kg.stages.chunk import _estimate_tokens
        assert _estimate_tokens("a" * 400) == 100


class TestParsing:
    def test_markdown_passthrough(self):
        from agents_kg.stages.parse import _is_markdown
        assert _is_markdown("# Hello\nWorld")
        assert not _is_markdown("Just plain text")

    def test_html_to_text(self):
        from agents_kg.stages.parse import _html_to_text
        result = _html_to_text("<html><body><h1>Title</h1><p>Hello</p></body></html>")
        assert "Hello" in result

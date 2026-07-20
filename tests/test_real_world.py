"""Real-world scenario tests — exercise the pipeline with diverse, realistic inputs.

These tests run fetch(mock)->parse->chunk->embed(mock)->extract(mock)->load(no Neo4j)
to validate the pipeline handles varied source types correctly.
"""

import os
import struct
import pytest
from unittest.mock import MagicMock, patch

from agents_kg.db import Database, content_hash
from agents_kg.stages.extract import _make_edge_id

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_cujs.py)
# ---------------------------------------------------------------------------


def _mock_fetch(db, source, content):
    from agents_kg.stages.fetch import run as run_fetch
    mock_resp = MagicMock()
    mock_resp.text = content
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    with patch("agents_kg.stages.fetch.httpx.Client", return_value=mock_client):
        return run_fetch(db, source)


def _run_through_chunk(db, source_id, content):
    from agents_kg.stages.parse import run as run_parse
    from agents_kg.stages.chunk import run as run_chunk
    source = db.get_source(source_id)
    _mock_fetch(db, source, content)
    source = db.get_source(source_id)
    run_parse(db, source)
    source = db.get_source(source_id)
    run_chunk(db, source)
    return db.get_source(source_id)


def _mock_embed(db, source):
    source_id = source["id"]
    chunks = db.get_unembedded_chunks(source_id)
    for c in chunks:
        emb = struct.pack("3f", 0.1, 0.2, 0.3)
        db.update_chunk_embedding(c["id"], emb, "gemini-embedding-2")
    db.update_source(source_id, stage="extract", status="processing")


def _mock_extract(db, source, entities, edges):
    source_id = source["id"]
    chunks = db.get_chunks(source_id)
    chunk_id = chunks[0]["id"] if chunks else None
    for ent in entities:
        db.add_entity(
            entity_id=ent["entity_id"],
            name=ent["name"],
            entity_type=ent["type"],
            kind=ent.get("kind"),
            description=ent.get("description"),
            source_id=source_id,
            chunk_id=chunk_id,
        )
    for e in edges:
        eid = _make_edge_id(e["src"], e["tgt"], e["type"])
        db.add_edge(eid, e["src"], e["tgt"], e["type"],
                    confidence=e.get("conf", 0.9),
                    source_id=source_id, chunk_id=chunk_id)
    db.update_source(source_id, stage="resolve", status="processing")


def _approve_all(db):
    for ent in db.get_entities_by_status("pending_review"):
        db.approve_entity(ent["id"])
    for edge in db.get_edges_by_status("pending_review"):
        db.approve_edge(edge["id"])


def _full_pipeline_no_neo4j(db, uri, content, entities, edges,
                            submitter_email="tester@example.com"):
    """Run the full pipeline through load (no Neo4j) and return source_id."""
    from agents_kg.stages.load import run as run_load
    sid = db.add_source(uri, submitter_email=submitter_email)
    assert sid is not None, f"Failed to add source {uri}"
    source = _run_through_chunk(db, sid, content)
    _mock_embed(db, source)
    source = db.get_source(sid)
    _mock_extract(db, source, entities, edges)
    db.update_source(sid, stage="review", status="pending_review")
    _approve_all(db)
    db.update_source(sid, status="processing", stage="load")
    source = db.get_source(sid)
    run_load(db, source, neo4j_driver=None)
    return sid


# ---------------------------------------------------------------------------
# 1. AUTHORITATIVE SOURCE — RFC / Protocol Specification
# ---------------------------------------------------------------------------

RFC_CONTENT = """\
# Model Context Protocol (MCP) Specification v1.0

## Abstract

The Model Context Protocol (MCP) is an open protocol that standardizes how
applications provide context to Large Language Models (LLMs). Developed by
Anthropic, MCP follows a client-server architecture where a host application
can connect to multiple MCP servers.

## 1. Introduction

MCP aims to solve the "M×N" integration problem between AI applications and
data sources. Rather than building custom integrations for each combination,
MCP provides a universal standard.

## 2. Architecture

MCP defines three roles:
- **Host**: The AI application (e.g., Claude Desktop, an IDE plugin)
- **Client**: A protocol connector maintained by the host
- **Server**: A lightweight process exposing resources and tools

Communication uses JSON-RPC 2.0 over stdio or HTTP with Server-Sent Events (SSE).

## 3. Capabilities

Servers can expose:
- **Resources**: Read-only data (files, database records, API responses)
- **Tools**: Executable functions the LLM can invoke
- **Prompts**: Templated prompts for common interactions

## 4. Security Considerations

All tool invocations require explicit user approval. Resource access is scoped
to what the server exposes. Transport-level security (TLS) is recommended for
HTTP deployments.
"""

RFC_ENTITIES = [
    {"entity_id": "rw:protocol-mcp", "name": "Model Context Protocol",
     "type": "Protocol", "kind": "spec",
     "description": "Open protocol for standardizing LLM context provision"},
    {"entity_id": "rw:org-anthropic", "name": "Anthropic",
     "type": "Organization", "kind": "company",
     "description": "AI company that developed MCP"},
    {"entity_id": "rw:cap-resources", "name": "MCP Resources",
     "type": "Capability", "kind": "feature",
     "description": "Read-only data exposure capability of MCP"},
    {"entity_id": "rw:cap-tools", "name": "MCP Tools",
     "type": "Capability", "kind": "feature",
     "description": "Executable function invocation capability of MCP"},
]

RFC_EDGES = [
    {"src": "rw:org-anthropic", "tgt": "rw:protocol-mcp",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "rw:protocol-mcp", "tgt": "rw:cap-resources",
     "type": "PROVIDES", "conf": 0.95},
    {"src": "rw:protocol-mcp", "tgt": "rw:cap-tools",
     "type": "PROVIDES", "conf": 0.95},
]


class TestAuthoritativeSource:
    """Scenario 1: A formal RFC/spec — well-structured, dense with entities."""

    def test_rfc_ingestion_produces_entities_and_edges(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://specs/mcp-v1.0", RFC_CONTENT,
            RFC_ENTITIES, RFC_EDGES,
            submitter_email="specs-bot@anthropic.com",
        )
        source = db.get_source(sid)
        assert source["status"] == "complete"
        assert source["stage"] == "done"

        entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        assert len(entities) == len(RFC_ENTITIES)

        edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        assert len(edges) == len(RFC_EDGES)

    def test_rfc_chunks_capture_sections(self, db):
        sid = db.add_source("test://specs/mcp-chunks")
        _run_through_chunk(db, sid, RFC_CONTENT)
        chunks = db.get_chunks(sid)
        assert len(chunks) >= 3
        headings = [c["section_heading"] for c in chunks if c["section_heading"]]
        heading_text = " ".join(h for h in headings if h)
        assert "Architecture" in heading_text or "Introduction" in heading_text

    def test_rfc_provenance_tracked(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://specs/mcp-provenance", RFC_CONTENT,
            RFC_ENTITIES[:1], [],
            submitter_email="specs-bot@anthropic.com",
        )
        source = db.get_source(sid)
        assert source["submitter_email"] == "specs-bot@anthropic.com"
        assert source["content_hash"] == content_hash(RFC_CONTENT)
        assert source["created_at"] is not None


# ---------------------------------------------------------------------------
# 2. INFORMAL SOURCE — Slack/Chat Discussion
# ---------------------------------------------------------------------------

SLACK_CONTENT = """\
# Slack Discussion: #ai-protocols — Protocol Comparison Thread

**alice@acme.com** (10:32 AM):
Hey team, we need to decide between MCP and A2A for our agent integration layer.
Anyone have thoughts?

**bob@acme.com** (10:35 AM):
I've been looking at both. MCP is more about giving an LLM access to tools and
data — like a "context window extension." A2A is about agents talking to each
other directly.

**carol@acme.com** (10:38 AM):
Right, they're complementary, not competing. We'd use MCP for our Claude
integration and A2A when our agents need to coordinate with external ones.

**dave@acme.com** (10:42 AM):
Google just released ADK (Agent Development Kit) which supports both protocols.
Might be worth evaluating. Microsoft's AutoGen also has some protocol support.

**alice@acme.com** (10:45 AM):
OK let's do a spike. Bob, can you prototype MCP server? Carol, take a look at
A2A discovery? Dave, evaluate ADK for our use case.

**bob@acme.com** (10:47 AM):
On it. I'll set up a basic MCP server with Anthropic's Python SDK. Should have
something by Thursday.
"""

SLACK_ENTITIES = [
    {"entity_id": "rw:protocol-mcp-2", "name": "MCP",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "rw:protocol-a2a", "name": "A2A",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "rw:project-adk", "name": "Agent Development Kit",
     "type": "Project", "kind": "framework",
     "description": "Google's toolkit supporting MCP and A2A"},
    {"entity_id": "rw:org-google", "name": "Google",
     "type": "Organization", "kind": "company"},
    {"entity_id": "rw:org-microsoft", "name": "Microsoft",
     "type": "Organization", "kind": "company"},
    {"entity_id": "rw:project-autogen", "name": "AutoGen",
     "type": "Project", "kind": "framework"},
]

SLACK_EDGES = [
    {"src": "rw:org-google", "tgt": "rw:project-adk",
     "type": "DEVELOPS", "conf": 0.90},
    {"src": "rw:project-adk", "tgt": "rw:protocol-mcp-2",
     "type": "SUPPORTS", "conf": 0.85},
    {"src": "rw:project-adk", "tgt": "rw:protocol-a2a",
     "type": "SUPPORTS", "conf": 0.85},
    {"src": "rw:org-microsoft", "tgt": "rw:project-autogen",
     "type": "DEVELOPS", "conf": 0.80},
]


class TestInformalSource:
    """Scenario 2: Messy Slack transcript with opinions and action items."""

    def test_slack_ingestion_completes(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://chat/protocol-comparison", SLACK_CONTENT,
            SLACK_ENTITIES, SLACK_EDGES,
            submitter_email="alice@acme.com",
        )
        source = db.get_source(sid)
        assert source["status"] == "complete"

    def test_slack_produces_multiple_entity_types(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://chat/slack-types", SLACK_CONTENT,
            SLACK_ENTITIES, SLACK_EDGES,
        )
        types_found = set()
        entities = db.conn.execute(
            "SELECT DISTINCT type FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        for e in entities:
            types_found.add(dict(e)["type"])
        assert "Protocol" in types_found
        assert "Project" in types_found
        assert "Organization" in types_found

    def test_slack_edges_capture_relationships(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://chat/slack-edges", SLACK_CONTENT,
            SLACK_ENTITIES, SLACK_EDGES,
        )
        edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        edge_types = {dict(e)["edge_type"] for e in edges}
        assert "DEVELOPS" in edge_types
        assert "SUPPORTS" in edge_types


# ---------------------------------------------------------------------------
# 3. MIXED QUALITY — Enrichment + Contradiction + Ambiguity
# ---------------------------------------------------------------------------

ENRICHMENT_CONTENT = """\
# Anthropic Company Update — Q1 2026

Anthropic, the AI safety company founded in 2021, announced several updates:

1. Claude 4 launched in March 2026 with improved reasoning capabilities
2. MCP 1.1 specification released with streaming support
3. Partnership with Amazon Web Services expanded for enterprise deployment
4. New safety research paper on constitutional AI published

The company, headquartered in San Francisco, now has over 1,500 employees.
"""

ENRICHMENT_ENTITIES = [
    {"entity_id": "rw:org-anthropic", "name": "Anthropic",
     "type": "Organization", "kind": "company",
     "description": "AI safety company, founded 2021, HQ San Francisco, 1500+ employees"},
    {"entity_id": "rw:project-claude-4", "name": "Claude 4",
     "type": "Project", "kind": "model",
     "description": "LLM launched March 2026 with improved reasoning"},
    {"entity_id": "rw:protocol-mcp-1.1", "name": "MCP 1.1",
     "type": "Protocol", "kind": "spec",
     "description": "Updated MCP spec with streaming support"},
    {"entity_id": "rw:org-aws", "name": "Amazon Web Services",
     "type": "Organization", "kind": "company"},
]

ENRICHMENT_EDGES = [
    {"src": "rw:org-anthropic", "tgt": "rw:project-claude-4",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "rw:org-anthropic", "tgt": "rw:protocol-mcp-1.1",
     "type": "DEVELOPS", "conf": 0.95},
    {"src": "rw:org-anthropic", "tgt": "rw:org-aws",
     "type": "PARTNERS_WITH", "conf": 0.90},
]


class TestMixedQualitySource:
    """Scenario 3: Source that enriches existing entities and adds new ones."""

    def test_enrichment_source_after_rfc(self, db):
        """Ingest RFC first, then enrichment doc — entities should coexist."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://specs/mcp-for-enrichment", RFC_CONTENT,
            RFC_ENTITIES, RFC_EDGES,
            submitter_email="specs-bot@anthropic.com",
        )

        sid2 = _full_pipeline_no_neo4j(
            db, "test://news/anthropic-q1-2026", ENRICHMENT_CONTENT,
            ENRICHMENT_ENTITIES, ENRICHMENT_EDGES,
            submitter_email="news-bot@acme.com",
        )

        s1 = db.get_source(sid1)
        s2 = db.get_source(sid2)
        assert s1["status"] == "complete"
        assert s2["status"] == "complete"

        all_entities = db.conn.execute(
            "SELECT * FROM entities WHERE status = 'approved'"
        ).fetchall()
        entity_ids = {dict(e)["entity_id"] for e in all_entities}

        assert "rw:protocol-mcp" in entity_ids
        assert "rw:project-claude-4" in entity_ids
        assert "rw:org-aws" in entity_ids

    def test_different_submitters_tracked(self, db):
        """Two sources from different submitters — provenance preserved."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://multi-sub/source-a", RFC_CONTENT,
            RFC_ENTITIES[:1], [],
            submitter_email="alice@company.com",
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://multi-sub/source-b", ENRICHMENT_CONTENT,
            ENRICHMENT_ENTITIES[:1], [],
            submitter_email="bob@other.org",
        )

        s1 = db.get_source(sid1)
        s2 = db.get_source(sid2)
        assert s1["submitter_email"] == "alice@company.com"
        assert s2["submitter_email"] == "bob@other.org"

    def test_source_update_enriches_not_corrupts(self, db):
        """After updating a source, entities from other sources survive."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://enrich/stable", RFC_CONTENT,
            RFC_ENTITIES, RFC_EDGES,
        )

        sid2 = _full_pipeline_no_neo4j(
            db, "test://enrich/volatile", ENRICHMENT_CONTENT,
            ENRICHMENT_ENTITIES, ENRICHMENT_EDGES,
        )

        # Now "update" sid2 with new content — deprecates its entities
        db.update_source(sid2, status="pending", stage="fetch")
        source2 = db.get_source(sid2)
        _mock_fetch(db, source2, ENRICHMENT_CONTENT + "\n\nAddendum: More info.")

        # Entities from sid1 should NOT be deprecated
        all_entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND deprecated_at IS NULL",
            (sid1,)
        ).fetchall()
        assert len(all_entities) == len(RFC_ENTITIES)


# ---------------------------------------------------------------------------
# 4. EDGE CASE — Very Short Source
# ---------------------------------------------------------------------------

SHORT_CONTENT = """\
Anthropic released Claude Code, a CLI tool for developers.
"""

SHORT_ENTITIES = [
    {"entity_id": "rw:org-anthropic-short", "name": "Anthropic",
     "type": "Organization", "kind": "company"},
    {"entity_id": "rw:project-claude-code", "name": "Claude Code",
     "type": "Project", "kind": "tool",
     "description": "CLI tool for developers by Anthropic"},
]

SHORT_EDGES = [
    {"src": "rw:org-anthropic-short", "tgt": "rw:project-claude-code",
     "type": "DEVELOPS", "conf": 0.95},
]


class TestEdgeCases:
    """Scenario 4: Edge cases — short sources, many entities few edges, re-ingestion."""

    def test_very_short_source(self, db):
        """A one-sentence source should still produce chunks and entities."""
        sid = _full_pipeline_no_neo4j(
            db, "test://edge/short", SHORT_CONTENT,
            SHORT_ENTITIES, SHORT_EDGES,
        )
        source = db.get_source(sid)
        assert source["status"] == "complete"

        chunks = db.get_chunks(sid)
        assert len(chunks) >= 1

        entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        assert len(entities) == 2

    def test_many_entities_few_edges(self, db):
        """Source with many entities but only one relationship."""
        many_entities = [
            {"entity_id": f"rw:standalone-{i}", "name": f"Entity {i}",
             "type": "Organization", "kind": "company"}
            for i in range(8)
        ]
        one_edge = [
            {"src": "rw:standalone-0", "tgt": "rw:standalone-1",
             "type": "PARTNERS_WITH", "conf": 0.7},
        ]

        content = "Organizations: " + ", ".join(
            f"Entity {i}" for i in range(8)
        ) + ". Entity 0 partners with Entity 1."

        sid = _full_pipeline_no_neo4j(
            db, "test://edge/many-ents", content,
            many_entities, one_edge,
        )

        entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        assert len(entities) == 8
        assert len(edges) == 1

    def test_reingest_after_deprecation(self, db):
        """Deprecate a source's entities, then re-ingest with updated content."""
        from agents_kg.stages.load import run as run_load
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        sid = _full_pipeline_no_neo4j(
            db, "test://edge/reingest", RFC_CONTENT,
            RFC_ENTITIES[:2], RFC_EDGES[:1],
        )

        # Deprecate
        db.deprecate_entities_for_source(sid)
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) >= 2

        # Re-ingest with updated content
        updated_content = RFC_CONTENT.replace("v1.0", "v2.0")
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        _mock_fetch(db, source, updated_content)

        source = db.get_source(sid)
        run_parse(db, source)
        source = db.get_source(sid)
        run_chunk(db, source)
        source = db.get_source(sid)
        _mock_embed(db, source)

        new_entities = [
            {"entity_id": "rw:protocol-mcp-v2", "name": "MCP v2.0",
             "type": "Protocol", "kind": "spec"},
        ]
        source = db.get_source(sid)
        _mock_extract(db, source, new_entities, [])
        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=None)

        source = db.get_source(sid)
        assert source["status"] == "complete"
        assert source["content_hash"] == content_hash(updated_content)

    def test_person_as_org_founder(self, db):
        """Entity that spans roles: a person who founded an organization."""
        content = """\
# Dario Amodei — CEO and Co-founder of Anthropic

Dario Amodei co-founded Anthropic in 2021 after leaving OpenAI. He serves as
CEO and leads the company's research direction on AI safety.
"""
        entities = [
            {"entity_id": "rw:person-dario", "name": "Dario Amodei",
             "type": "Person", "kind": "executive",
             "description": "CEO and co-founder of Anthropic"},
            {"entity_id": "rw:org-anthropic-founder", "name": "Anthropic",
             "type": "Organization", "kind": "company"},
            {"entity_id": "rw:org-openai-prev", "name": "OpenAI",
             "type": "Organization", "kind": "company"},
        ]
        edges = [
            {"src": "rw:person-dario", "tgt": "rw:org-anthropic-founder",
             "type": "FOUNDED", "conf": 0.98},
            {"src": "rw:person-dario", "tgt": "rw:org-anthropic-founder",
             "type": "LEADS", "conf": 0.95},
            {"src": "rw:person-dario", "tgt": "rw:org-openai-prev",
             "type": "PREVIOUSLY_AT", "conf": 0.90},
        ]

        sid = _full_pipeline_no_neo4j(
            db, "test://edge/person-founder", content,
            entities, edges,
        )

        stored_entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        entity_types = {dict(e)["type"] for e in stored_entities}
        assert "Person" in entity_types
        assert "Organization" in entity_types

        stored_edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        edge_types = {dict(e)["edge_type"] for e in stored_edges}
        assert "FOUNDED" in edge_types
        assert "LEADS" in edge_types

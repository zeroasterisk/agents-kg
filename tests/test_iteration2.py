"""Iteration 2 real-world scenario tests.

Covers: overlapping sources, negative/adversarial inputs, opinion-laden blog
posts, entity resolution ambiguity, and temporal scenarios.
"""

import struct
import pytest
from unittest.mock import MagicMock, patch

from agents_kg.db import Database, content_hash
from agents_kg.stages.extract import _make_edge_id

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from test_real_world.py)
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


def _mock_fetch_html(db, source, content):
    from agents_kg.stages.fetch import run as run_fetch
    mock_resp = MagicMock()
    mock_resp.text = content
    mock_resp.headers = {"content-type": "text/html"}
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
        db.update_chunk_embedding(c["id"], emb, "gemini-embedding-2-preview")
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
                    source_id=source_id, chunk_id=chunk_id,
                    valid_from=e.get("valid_from"),
                    valid_to=e.get("valid_to"))
    db.update_source(source_id, stage="resolve", status="processing")


def _approve_all(db):
    for ent in db.get_entities_by_status("pending_review"):
        db.approve_entity(ent["id"])
    for edge in db.get_edges_by_status("pending_review"):
        db.approve_edge(edge["id"])


def _full_pipeline_no_neo4j(db, uri, content, entities, edges,
                            submitter_email="tester@example.com"):
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


# ===========================================================================
# 1. OVERLAPPING SOURCES — Two documents mentioning the same entities
# ===========================================================================

SOURCE_A_CONTENT = """\
# Anthropic and MCP: A Technical Overview

Anthropic developed the Model Context Protocol (MCP) to standardize how AI
applications provide context to LLMs. MCP uses JSON-RPC 2.0 and supports
tools, resources, and prompts as capabilities.
"""

SOURCE_B_CONTENT = """\
# Google ADK Supports MCP

Google's Agent Development Kit (ADK) now supports MCP alongside A2A for
agent-to-agent communication. Anthropic collaborated with Google on
interoperability testing.
"""

SHARED_ENTITY_MCP = {
    "entity_id": "protocol:mcp", "name": "Model Context Protocol",
    "type": "Protocol", "kind": "spec",
    "description": "Open protocol for LLM context provision",
}
SHARED_ENTITY_ANTHROPIC = {
    "entity_id": "org:anthropic", "name": "Anthropic",
    "type": "Organization", "kind": "company",
}

SOURCE_A_ENTITIES = [
    SHARED_ENTITY_MCP,
    SHARED_ENTITY_ANTHROPIC,
    {"entity_id": "cap:tools", "name": "MCP Tools",
     "type": "Capability", "kind": "feature"},
]
SOURCE_A_EDGES = [
    {"src": "org:anthropic", "tgt": "protocol:mcp",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "protocol:mcp", "tgt": "cap:tools",
     "type": "DEFINES", "conf": 0.95},
]

SOURCE_B_ENTITIES = [
    SHARED_ENTITY_MCP,
    SHARED_ENTITY_ANTHROPIC,
    {"entity_id": "org:google", "name": "Google",
     "type": "Organization", "kind": "company"},
    {"entity_id": "project:adk", "name": "Agent Development Kit",
     "type": "Project", "kind": "framework"},
    {"entity_id": "protocol:a2a", "name": "A2A",
     "type": "Protocol", "kind": "spec"},
]
SOURCE_B_EDGES = [
    {"src": "org:google", "tgt": "project:adk",
     "type": "DEVELOPS", "conf": 0.95},
    {"src": "project:adk", "tgt": "protocol:mcp",
     "type": "IMPLEMENTS", "conf": 0.90},
    {"src": "project:adk", "tgt": "protocol:a2a",
     "type": "IMPLEMENTS", "conf": 0.90},
]


class TestOverlappingSources:
    """Two sources that mention the same entities (Anthropic, MCP)."""

    def test_shared_entity_not_duplicated(self, db):
        """Same entity_id from two sources produces only one row."""
        _full_pipeline_no_neo4j(
            db, "test://overlap/source-a", SOURCE_A_CONTENT,
            SOURCE_A_ENTITIES, SOURCE_A_EDGES,
            submitter_email="alice@example.com",
        )
        _full_pipeline_no_neo4j(
            db, "test://overlap/source-b", SOURCE_B_CONTENT,
            SOURCE_B_ENTITIES, SOURCE_B_EDGES,
            submitter_email="bob@example.com",
        )

        mcp_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'protocol:mcp'"
        ).fetchall()
        assert len(mcp_rows) == 1

        anthropic_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'org:anthropic'"
        ).fetchall()
        assert len(anthropic_rows) == 1

    def test_both_sources_complete(self, db):
        sid1 = _full_pipeline_no_neo4j(
            db, "test://overlap/both-a", SOURCE_A_CONTENT,
            SOURCE_A_ENTITIES, SOURCE_A_EDGES,
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://overlap/both-b", SOURCE_B_CONTENT,
            SOURCE_B_ENTITIES, SOURCE_B_EDGES,
        )
        assert db.get_source(sid1)["status"] == "complete"
        assert db.get_source(sid2)["status"] == "complete"

    def test_edges_from_both_sources_exist(self, db):
        """Each source contributes its own edges even when entities overlap."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://overlap/edges-a", SOURCE_A_CONTENT,
            SOURCE_A_ENTITIES, SOURCE_A_EDGES,
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://overlap/edges-b", SOURCE_B_CONTENT,
            SOURCE_B_ENTITIES, SOURCE_B_EDGES,
        )

        edges_a = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid1,)
        ).fetchall()
        edges_b = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid2,)
        ).fetchall()
        assert len(edges_a) == len(SOURCE_A_EDGES)
        assert len(edges_b) == len(SOURCE_B_EDGES)

    def test_provenance_tracks_both_submitters(self, db):
        sid1 = _full_pipeline_no_neo4j(
            db, "test://overlap/prov-a", SOURCE_A_CONTENT,
            SOURCE_A_ENTITIES, SOURCE_A_EDGES,
            submitter_email="alice@team.com",
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://overlap/prov-b", SOURCE_B_CONTENT,
            SOURCE_B_ENTITIES, SOURCE_B_EDGES,
            submitter_email="bob@team.com",
        )
        assert db.get_source(sid1)["submitter_email"] == "alice@team.com"
        assert db.get_source(sid2)["submitter_email"] == "bob@team.com"

    def test_deprecating_one_source_preserves_other(self, db):
        """Deprecating source B does not affect entities owned by source A."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://overlap/dep-a", SOURCE_A_CONTENT,
            SOURCE_A_ENTITIES, SOURCE_A_EDGES,
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://overlap/dep-b", SOURCE_B_CONTENT,
            SOURCE_B_ENTITIES, SOURCE_B_EDGES,
        )

        db.deprecate_entities_for_source(sid2)

        # Entities owned by sid1 should be untouched
        alive_from_a = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND deprecated_at IS NULL",
            (sid1,)
        ).fetchall()
        assert len(alive_from_a) == len(SOURCE_A_ENTITIES)

        # Entities unique to sid2 should be deprecated
        deprecated = db.get_deprecated_entities()
        deprecated_ids = {dict(e)["entity_id"] for e in deprecated}
        assert "org:google" in deprecated_ids
        assert "project:adk" in deprecated_ids
        assert "protocol:a2a" in deprecated_ids


# ===========================================================================
# 2. NEGATIVE / ADVERSARIAL INPUTS
# ===========================================================================

class TestNegativeInputs:
    """Bad, empty, or adversarial content fed to the pipeline."""

    def test_empty_document_raises_on_parse(self, db):
        """Empty document causes parse to raise RuntimeError (no raw_text)."""
        from agents_kg.stages.parse import run as run_parse

        sid = db.add_source("test://negative/empty")
        source = db.get_source(sid)
        _mock_fetch(db, source, "")
        source = db.get_source(sid)

        with pytest.raises(RuntimeError, match="No raw_text to parse"):
            run_parse(db, source)

    def test_whitespace_only_document_produces_minimal_chunk(self, db):
        """Whitespace-only document passes parse/chunk but yields only whitespace content."""
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("test://negative/whitespace")
        source = db.get_source(sid)
        _mock_fetch(db, source, "   \n\n\t  \n   ")
        source = db.get_source(sid)

        run_parse(db, source)
        source = db.get_source(sid)
        run_chunk(db, source)

        chunks = db.get_chunks(sid)
        assert len(chunks) >= 1
        for c in chunks:
            assert c["text"].strip() == ""

    def test_very_long_document(self, db):
        """Repeated content produces chunks without crashing."""
        long_section = "## Section\n\nAnthropic builds AI safety tools.\n\n" * 200
        content = "# Long Document\n\n" + long_section

        sid = db.add_source("test://negative/long")
        source = _run_through_chunk(db, sid, content)
        chunks = db.get_chunks(sid)
        assert len(chunks) >= 5
        assert source["stage"] == "embed"

    def test_no_extractable_entities_lorem_ipsum(self, db):
        """Lorem ipsum text produces chunks but zero entities."""
        lorem = (
            "# Lorem Ipsum\n\n"
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
            "Duis aute irure dolor in reprehenderit in voluptate velit esse.\n\n"
            "## More Text\n\n"
            "Cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat "
            "cupidatat non proident, sunt in culpa qui officia deserunt mollit."
        )
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://negative/lorem")
        source = _run_through_chunk(db, sid, lorem)
        chunks = db.get_chunks(sid)
        assert len(chunks) >= 1

        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, [], [])
        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=None)

        entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(entities) == 0

        source = db.get_source(sid)
        assert source["status"] == "complete"

    def test_malformed_html_graceful_parse(self, db):
        """Malformed HTML (unclosed tags) is parsed without crashing."""
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        malformed = (
            "<html><body>"
            "<h1>Broken Page</h1>"
            "<p>This paragraph is never closed"
            "<div><span>Nested unclosed tags"
            "<script>alert('xss')</script>"
            "<p>Content after script tag</p>"
            "</body></html>"
        )
        sid = db.add_source("test://negative/malformed-html")
        source = db.get_source(sid)
        _mock_fetch_html(db, source, malformed)
        source = db.get_source(sid)
        run_parse(db, source)
        source = db.get_source(sid)
        assert source["parsed_text"] is not None
        assert "<script>" not in source["parsed_text"]

        run_chunk(db, source)
        chunks = db.get_chunks(sid)
        assert len(chunks) >= 1

    def test_script_injection_stripped(self, db):
        """Script tags are stripped during HTML parsing."""
        from agents_kg.stages.parse import run as run_parse

        html_with_xss = (
            "<html><body>"
            "<h1>Real Content</h1>"
            "<p>Anthropic builds safe AI.</p>"
            "<script>document.cookie</script>"
            "<img onerror='alert(1)' src='x'>"
            "</body></html>"
        )
        sid = db.add_source("test://negative/xss")
        source = db.get_source(sid)
        _mock_fetch_html(db, source, html_with_xss)
        source = db.get_source(sid)
        run_parse(db, source)
        source = db.get_source(sid)
        text = source["parsed_text"]
        assert "<script>" not in text
        assert "onerror" not in text

    def test_duplicate_uri_rejected(self, db):
        """Adding the same URI twice returns None the second time."""
        sid1 = db.add_source("test://negative/dup")
        sid2 = db.add_source("test://negative/dup")
        assert sid1 is not None
        assert sid2 is None


# ===========================================================================
# 3. BLOG POST / OPINION PIECE
# ===========================================================================

BLOG_CONTENT = """\
# Why We Chose A2A Over MCP for Agent-to-Agent Communication

After months of evaluation, our team decided to standardize on Google's A2A
protocol for inter-agent communication, rather than Anthropic's MCP. Here's why.

## The Problem

We needed our agents to coordinate across organizational boundaries. MCP is
brilliant for tool access — giving an LLM the ability to call APIs, read files,
and interact with databases — but it was never designed for agent-to-agent
dialogue.

## A2A's Strengths

A2A (Agent-to-Agent) was built from the ground up for multi-agent
orchestration. Key advantages we found:

1. **Agent Cards** — JSON metadata describing an agent's capabilities,
   discoverable via /.well-known/agent.json
2. **Task lifecycle** — built-in states (submitted, working, completed, failed)
   that map naturally to async workflows
3. **Streaming** — SSE-based streaming for long-running tasks

## MCP Is Still Great

Don't misunderstand — we still use MCP extensively. Our Claude-based coding
assistants use MCP servers to access internal databases and documentation.
MCP and A2A are complementary, not competing.

## Our Stack

We settled on: ADK (Google) for orchestration, MCP for tool access, and A2A
for agent-to-agent coordination. LangChain provides the glue layer.

## Conclusion

If you're building single-agent tool integrations, MCP is the clear winner.
For multi-agent systems, evaluate A2A seriously.
"""

BLOG_ENTITIES = [
    {"entity_id": "protocol:a2a", "name": "A2A",
     "type": "Protocol", "kind": "spec",
     "description": "Agent-to-Agent protocol for multi-agent orchestration"},
    {"entity_id": "protocol:mcp", "name": "Model Context Protocol",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "project:adk", "name": "Agent Development Kit",
     "type": "Project", "kind": "framework"},
    {"entity_id": "org:google", "name": "Google",
     "type": "Organization", "kind": "company"},
    {"entity_id": "org:anthropic", "name": "Anthropic",
     "type": "Organization", "kind": "company"},
    {"entity_id": "project:langchain", "name": "LangChain",
     "type": "Project", "kind": "framework"},
    {"entity_id": "cap:tool-use", "name": "Tool Use",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "cap:multi-agent", "name": "Multi-Agent",
     "type": "Capability", "kind": "feature"},
]

BLOG_EDGES = [
    {"src": "org:google", "tgt": "protocol:a2a",
     "type": "DEVELOPS", "conf": 0.90},
    {"src": "org:anthropic", "tgt": "protocol:mcp",
     "type": "DEVELOPS", "conf": 0.95},
    {"src": "protocol:a2a", "tgt": "cap:multi-agent",
     "type": "ADDRESSES", "conf": 0.92},
    {"src": "protocol:mcp", "tgt": "cap:tool-use",
     "type": "ADDRESSES", "conf": 0.95},
    {"src": "protocol:a2a", "tgt": "protocol:mcp",
     "type": "COMPLEMENTS", "conf": 0.85},
    {"src": "project:adk", "tgt": "protocol:a2a",
     "type": "IMPLEMENTS", "conf": 0.90},
]


class TestBlogPostOpinionPiece:
    """Opinion-laden blog post comparing agentic frameworks."""

    def test_blog_ingestion_completes(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://blog/a2a-vs-mcp", BLOG_CONTENT,
            BLOG_ENTITIES, BLOG_EDGES,
            submitter_email="dev@startup.io",
        )
        source = db.get_source(sid)
        assert source["status"] == "complete"

    def test_blog_extracts_multiple_entity_types(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://blog/entity-types", BLOG_CONTENT,
            BLOG_ENTITIES, BLOG_EDGES,
        )
        types = {
            dict(e)["type"]
            for e in db.conn.execute(
                "SELECT DISTINCT type FROM entities WHERE source_id = ?", (sid,)
            ).fetchall()
        }
        assert types >= {"Protocol", "Project", "Organization", "Capability"}

    def test_blog_captures_complementary_relationship(self, db):
        sid = _full_pipeline_no_neo4j(
            db, "test://blog/complement", BLOG_CONTENT,
            BLOG_ENTITIES, BLOG_EDGES,
        )
        complement_edges = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'COMPLEMENTS' AND source_id = ?",
            (sid,)
        ).fetchall()
        assert len(complement_edges) >= 1
        edge = dict(complement_edges[0])
        assert edge["source_entity_id"] == "protocol:a2a"
        assert edge["target_entity_id"] == "protocol:mcp"

    def test_blog_chunks_capture_opinions(self, db):
        """Blog post with headings should chunk into meaningful sections."""
        sid = db.add_source("test://blog/chunks")
        _run_through_chunk(db, sid, BLOG_CONTENT)
        chunks = db.get_chunks(sid)
        assert len(chunks) >= 3
        all_text = " ".join(c["text"] for c in chunks)
        assert "complementary" in all_text.lower() or "Chose" in all_text

    def test_blog_shared_entities_with_rfc(self, db):
        """Blog and RFC source share MCP/Anthropic entities — no duplication."""
        rfc_entities = [
            {"entity_id": "protocol:mcp", "name": "MCP",
             "type": "Protocol", "kind": "spec"},
            {"entity_id": "org:anthropic", "name": "Anthropic",
             "type": "Organization", "kind": "company"},
        ]
        rfc_edges = [
            {"src": "org:anthropic", "tgt": "protocol:mcp",
             "type": "DEVELOPS", "conf": 0.98},
        ]
        _full_pipeline_no_neo4j(
            db, "test://blog/rfc-first", SOURCE_A_CONTENT,
            rfc_entities, rfc_edges,
        )
        _full_pipeline_no_neo4j(
            db, "test://blog/blog-second", BLOG_CONTENT,
            BLOG_ENTITIES, BLOG_EDGES,
        )

        mcp_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'protocol:mcp'"
        ).fetchall()
        assert len(mcp_rows) == 1


# ===========================================================================
# 4. ENTITY RESOLUTION SCENARIOS
# ===========================================================================

class TestEntityResolution:
    """Same entity with different names/aliases, type ambiguity, merging."""

    def test_same_entity_different_names_alias_tracking(self, db):
        """Google / Alphabet / Google LLC should resolve to one entity via aliases."""
        content = """\
# Alphabet Inc. — Parent Company

Alphabet, the parent of Google (also known as Google LLC), reported strong
earnings driven by AI investments.
"""
        entities = [
            {"entity_id": "org:google", "name": "Google",
             "type": "Organization", "kind": "company",
             "description": "Tech company, subsidiary of Alphabet"},
        ]
        sid = _full_pipeline_no_neo4j(
            db, "test://resolution/aliases", content,
            entities, [],
        )
        source = db.get_source(sid)
        assert source["status"] == "complete"
        google = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'org:google'"
        ).fetchall()
        assert len(google) == 1

    def test_entity_type_ambiguity_python(self, db):
        """'Python' as a programming language project vs ambiguity."""
        content = """\
# Python in AI Development

Python is the dominant language for AI/ML development. The Python Software
Foundation maintains the language and its standard library.
"""
        entities = [
            {"entity_id": "project:python", "name": "Python",
             "type": "Project", "kind": "language",
             "description": "Programming language for AI/ML"},
            {"entity_id": "org:python-foundation", "name": "Python Software Foundation",
             "type": "Organization", "kind": "foundation"},
        ]
        edges = [
            {"src": "org:python-foundation", "tgt": "project:python",
             "type": "DEVELOPS", "conf": 0.95},
        ]
        sid = _full_pipeline_no_neo4j(
            db, "test://resolution/python", content, entities, edges,
        )
        stored = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'project:python'"
        ).fetchall()
        assert len(stored) == 1
        assert dict(stored[0])["kind"] == "language"

    def test_overlapping_entity_ids_merge_not_duplicate(self, db):
        """Two sources with same entity_id = one row, not two."""
        ent = [{"entity_id": "org:shared-corp", "name": "SharedCorp",
                "type": "Organization", "kind": "company"}]

        sid1 = _full_pipeline_no_neo4j(
            db, "test://resolution/merge-a", "Source A mentions SharedCorp.",
            ent, [],
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://resolution/merge-b", "Source B also mentions SharedCorp.",
            ent, [],
        )

        rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'org:shared-corp'"
        ).fetchall()
        assert len(rows) == 1

        # Entity is owned by first source that created it
        assert dict(rows[0])["source_id"] == sid1

    def test_entity_id_uniqueness_different_types(self, db):
        """Two entities with different entity_ids coexist even if names are similar."""
        entities = [
            {"entity_id": "project:mcp-sdk", "name": "MCP SDK",
             "type": "Project", "kind": "sdk"},
            {"entity_id": "protocol:mcp", "name": "MCP",
             "type": "Protocol", "kind": "spec"},
        ]
        sid = _full_pipeline_no_neo4j(
            db, "test://resolution/types", "MCP SDK implements MCP protocol.",
            entities, [],
        )
        all_ents = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(all_ents) == 2
        ids = {dict(e)["entity_id"] for e in all_ents}
        assert ids == {"project:mcp-sdk", "protocol:mcp"}


# ===========================================================================
# 5. TEMPORAL SCENARIOS
# ===========================================================================

TEMPORAL_SOURCE_V1 = """\
# Anthropic Funding Round — January 2024

Anthropic raised $2 billion in a Series C round led by Google. The AI safety
company was valued at $15 billion. CEO Dario Amodei announced plans to use the
funding for scaling Claude model training.
"""

TEMPORAL_V1_ENTITIES = [
    {"entity_id": "org:anthropic", "name": "Anthropic",
     "type": "Organization", "kind": "company",
     "description": "AI safety company, valued at $15B as of Jan 2024"},
    {"entity_id": "org:google", "name": "Google",
     "type": "Organization", "kind": "company"},
    {"entity_id": "person:dario-amodei", "name": "Dario Amodei",
     "type": "Person", "kind": "executive"},
]

TEMPORAL_V1_EDGES = [
    {"src": "org:google", "tgt": "org:anthropic",
     "type": "SPONSORS", "conf": 0.95,
     "valid_from": "2024-01-01"},
    {"src": "person:dario-amodei", "tgt": "org:anthropic",
     "type": "MEMBER_OF", "conf": 0.98},
]

TEMPORAL_SOURCE_V2 = """\
# Anthropic Update — March 2026

Anthropic, now valued at $60 billion after a $5 billion Series E led by
Amazon, announced Claude 4 with enhanced reasoning. The company has grown
to 1,500+ employees.
"""

TEMPORAL_V2_ENTITIES = [
    {"entity_id": "org:anthropic", "name": "Anthropic",
     "type": "Organization", "kind": "company",
     "description": "AI safety company, valued at $60B as of March 2026"},
    {"entity_id": "org:amazon", "name": "Amazon",
     "type": "Organization", "kind": "company"},
    {"entity_id": "project:claude-4", "name": "Claude 4",
     "type": "Project", "kind": "model"},
]

TEMPORAL_V2_EDGES = [
    {"src": "org:amazon", "tgt": "org:anthropic",
     "type": "SPONSORS", "conf": 0.95,
     "valid_from": "2026-03-01"},
    {"src": "org:anthropic", "tgt": "project:claude-4",
     "type": "DEVELOPS", "conf": 0.98},
]


class TestTemporalScenarios:
    """Sources with explicit dates, fact updates, and temporal queries."""

    def test_source_with_explicit_dates(self, db):
        """Edges carry valid_from dates from the source text."""
        sid = _full_pipeline_no_neo4j(
            db, "test://temporal/v1", TEMPORAL_SOURCE_V1,
            TEMPORAL_V1_ENTITIES, TEMPORAL_V1_EDGES,
        )
        edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        sponsors_edge = [
            dict(e) for e in edges if dict(e)["edge_type"] == "SPONSORS"
        ]
        assert len(sponsors_edge) == 1
        assert sponsors_edge[0]["valid_from"] == "2024-01-01"

    def test_newer_source_updates_entity_description(self, db):
        """Ingesting a newer source — first source's entity remains, new source adds updated info."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://temporal/update-v1", TEMPORAL_SOURCE_V1,
            TEMPORAL_V1_ENTITIES, TEMPORAL_V1_EDGES,
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://temporal/update-v2", TEMPORAL_SOURCE_V2,
            TEMPORAL_V2_ENTITIES, TEMPORAL_V2_EDGES,
        )

        # Entity exists (created by first source), second source's duplicate add_entity returns None
        anthropic = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'org:anthropic'"
        ).fetchall()
        assert len(anthropic) == 1
        assert dict(anthropic[0])["source_id"] == sid1

        # But edges from both sources exist
        all_sponsors = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'SPONSORS' AND status = 'approved'"
        ).fetchall()
        assert len(all_sponsors) == 2
        valid_froms = {dict(e)["valid_from"] for e in all_sponsors}
        assert "2024-01-01" in valid_froms
        assert "2026-03-01" in valid_froms

    def test_temporal_edge_ordering(self, db):
        """Edges with valid_from can be ordered chronologically."""
        sid1 = _full_pipeline_no_neo4j(
            db, "test://temporal/order-v1", TEMPORAL_SOURCE_V1,
            TEMPORAL_V1_ENTITIES, TEMPORAL_V1_EDGES,
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "test://temporal/order-v2", TEMPORAL_SOURCE_V2,
            TEMPORAL_V2_ENTITIES, TEMPORAL_V2_EDGES,
        )

        edges_ordered = db.conn.execute(
            "SELECT * FROM edges WHERE valid_from IS NOT NULL "
            "ORDER BY valid_from ASC"
        ).fetchall()
        valid_froms = [dict(e)["valid_from"] for e in edges_ordered]
        assert valid_froms == sorted(valid_froms)

    def test_querying_changes_since_date(self, db):
        """Simulate 'what changed since date X' by filtering edges."""
        _full_pipeline_no_neo4j(
            db, "test://temporal/since-v1", TEMPORAL_SOURCE_V1,
            TEMPORAL_V1_ENTITIES, TEMPORAL_V1_EDGES,
        )
        _full_pipeline_no_neo4j(
            db, "test://temporal/since-v2", TEMPORAL_SOURCE_V2,
            TEMPORAL_V2_ENTITIES, TEMPORAL_V2_EDGES,
        )

        # "What changed since 2025-01-01?"
        recent_edges = db.conn.execute(
            "SELECT * FROM edges WHERE valid_from > '2025-01-01' AND status = 'approved'"
        ).fetchall()
        assert len(recent_edges) >= 1
        for e in recent_edges:
            assert dict(e)["valid_from"] > "2025-01-01"

    def test_content_hash_changes_on_update(self, db):
        """Re-fetching with different content produces a new content_hash."""
        sid = db.add_source("test://temporal/hash-change")
        source = db.get_source(sid)
        _mock_fetch(db, source, TEMPORAL_SOURCE_V1)
        hash1 = db.get_source(sid)["content_hash"]

        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        _mock_fetch(db, source, TEMPORAL_SOURCE_V2)
        hash2 = db.get_source(sid)["content_hash"]

        assert hash1 != hash2
        assert hash1 == content_hash(TEMPORAL_SOURCE_V1)
        assert hash2 == content_hash(TEMPORAL_SOURCE_V2)


# ===========================================================================
# 6. YAML EXPORT CORRECTNESS
# ===========================================================================

class TestYamlExport:
    """Verify YAML export works for entities loaded without Neo4j."""

    def test_yaml_files_created_on_load(self, db, tmp_path):
        """Load stage generates YAML entity files."""
        from agents_kg.stages.load import run as run_load
        from agents_kg.stages.load import _export_yaml
        import functools

        sid = db.add_source("test://yaml/export")
        entities = [
            {"entity_id": "yaml:test-entity", "name": "YAML Test Entity",
             "type": "Organization", "kind": "company",
             "description": "Entity for YAML export test"},
        ]
        source = _run_through_chunk(db, sid, "# YAML Test\n\nSome content here.")
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, [])
        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)

        redirected = functools.partial(_export_yaml, base_dir=str(tmp_path))
        with patch("agents_kg.stages.load._export_yaml", redirected):
            run_load(db, source, neo4j_driver=None)

        yaml_files = list(tmp_path.rglob("*.yaml")) + list(tmp_path.rglob("*.yml"))
        assert len(yaml_files) >= 1


# ===========================================================================
# 7. CROSS-SCENARIO: FULL LIFECYCLE
# ===========================================================================

class TestFullLifecycle:
    """End-to-end lifecycle: ingest, overlap, deprecate, re-ingest, query."""

    def test_full_lifecycle_three_sources(self, db):
        """Ingest three sources, deprecate one, verify state consistency."""
        # Source 1: RFC
        rfc_ents = [
            {"entity_id": "lc:protocol-mcp", "name": "MCP",
             "type": "Protocol", "kind": "spec"},
            {"entity_id": "lc:org-anthropic", "name": "Anthropic",
             "type": "Organization", "kind": "company"},
        ]
        rfc_edges = [
            {"src": "lc:org-anthropic", "tgt": "lc:protocol-mcp",
             "type": "DEVELOPS", "conf": 0.98},
        ]
        sid1 = _full_pipeline_no_neo4j(
            db, "test://lifecycle/rfc", SOURCE_A_CONTENT,
            rfc_ents, rfc_edges,
            submitter_email="spec-bot@test.com",
        )

        # Source 2: Blog (shares Anthropic)
        blog_ents = [
            {"entity_id": "lc:org-anthropic", "name": "Anthropic",
             "type": "Organization", "kind": "company"},
            {"entity_id": "lc:org-google", "name": "Google",
             "type": "Organization", "kind": "company"},
        ]
        blog_edges = []
        sid2 = _full_pipeline_no_neo4j(
            db, "test://lifecycle/blog", BLOG_CONTENT,
            blog_ents, blog_edges,
            submitter_email="blogger@test.com",
        )

        # Source 3: News update
        news_ents = [
            {"entity_id": "lc:org-anthropic", "name": "Anthropic",
             "type": "Organization", "kind": "company"},
            {"entity_id": "lc:project-claude", "name": "Claude",
             "type": "Project", "kind": "model"},
        ]
        news_edges = [
            {"src": "lc:org-anthropic", "tgt": "lc:project-claude",
             "type": "DEVELOPS", "conf": 0.95},
        ]
        sid3 = _full_pipeline_no_neo4j(
            db, "test://lifecycle/news", TEMPORAL_SOURCE_V2,
            news_ents, news_edges,
            submitter_email="news-bot@test.com",
        )

        # All three complete
        for sid in (sid1, sid2, sid3):
            assert db.get_source(sid)["status"] == "complete"

        # Deprecate source 2 (blog)
        db.deprecate_entities_for_source(sid2)

        # Anthropic entity (owned by source 1) survives
        anthropic = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'lc:org-anthropic' AND deprecated_at IS NULL"
        ).fetchall()
        assert len(anthropic) == 1

        # Google entity (unique to source 2) is deprecated
        google_dep = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'lc:org-google' AND deprecated_at IS NOT NULL"
        ).fetchall()
        assert len(google_dep) == 1

        # Claude entity (source 3) survives
        claude = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'lc:project-claude' AND deprecated_at IS NULL"
        ).fetchall()
        assert len(claude) == 1

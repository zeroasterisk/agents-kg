"""Iteration 4 tests.

Covers: multi-session knowledge building, KG querying patterns,
source lifecycle (full cycle), review/audit patterns, and real
agentic-web domain content.
"""

import json
import os
import struct
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agents_kg.db import Database, _now, content_hash
from agents_kg.stages.extract import _make_edge_id, VALID_EDGE_TYPES, VALID_ENTITY_TYPES
from agents_kg.stages.resolve import (
    _normalize,
    _similarity,
    run as run_resolve,
)
from agents_kg.stages.load import _entity_to_cypher, _edge_to_cypher

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers (shared with iteration 3 style)
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
            aliases=ent.get("aliases"),
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


def _full_ingest(db, uri, content, entities, edges):
    sid = db.add_source(uri)
    source = _run_through_chunk(db, sid, content)
    _mock_embed(db, source)
    source = db.get_source(sid)
    _mock_extract(db, source, entities, edges)
    source = db.get_source(sid)
    with patch("agents_kg.stages.resolve.genai", None):
        run_resolve(db, source)
    _approve_all(db)
    return sid


def _get_active_entity_ids(db):
    rows = db.conn.execute(
        "SELECT entity_id FROM entities WHERE merged_into IS NULL "
        "AND status != 'rejected' AND deprecated_at IS NULL"
    ).fetchall()
    return {r["entity_id"] for r in rows}


def _get_edge_triples(db):
    rows = db.conn.execute(
        "SELECT source_entity_id, target_entity_id, edge_type FROM edges"
    ).fetchall()
    return {(r["source_entity_id"], r["target_entity_id"], r["edge_type"]) for r in rows}


# ============================================================
# 1. MULTI-SESSION KNOWLEDGE BUILDING
# ============================================================


class TestMultiSessionKnowledgeBuilding:
    """Simulate a user building knowledge over multiple sessions."""

    # --- Day 1: foundational sources ---
    DAY1_SOURCES = [
        {
            "uri": "https://example.com/day1/a2a-spec",
            "content": """# A2A Protocol Specification v1.0

The Agent-to-Agent (A2A) protocol was developed by Google to enable
interoperability between AI agents from different vendors. It defines
a standard JSON-RPC interface for agent communication.

## Core Capabilities

A2A defines task management, streaming, and push notifications as
core capabilities that compliant agents must support.
""",
            "entities": [
                {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
                 "kind": "spec", "description": "Agent-to-Agent interoperability protocol"},
                {"entity_id": "organization:google", "name": "Google", "type": "Organization",
                 "kind": "company", "description": "Technology company"},
                {"entity_id": "capability:task-management", "name": "Task Management",
                 "type": "Capability", "description": "Manage agent tasks"},
                {"entity_id": "capability:streaming", "name": "Streaming",
                 "type": "Capability", "description": "Real-time streaming support"},
            ],
            "edges": [
                {"src": "organization:google", "tgt": "protocol:a2a", "type": "DEVELOPS"},
                {"src": "protocol:a2a", "tgt": "capability:task-management", "type": "DEFINES"},
                {"src": "protocol:a2a", "tgt": "capability:streaming", "type": "DEFINES"},
            ],
        },
        {
            "uri": "https://example.com/day1/mcp-spec",
            "content": """# Model Context Protocol (MCP)

MCP was created by Anthropic as a standard for connecting AI models to
external tools and data sources. It uses a client-server architecture
with JSON-RPC transport.

## Key Features

MCP supports tool invocation, resource access, and prompt templates.
""",
            "entities": [
                {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
                 "kind": "spec", "description": "Model Context Protocol"},
                {"entity_id": "organization:anthropic", "name": "Anthropic",
                 "type": "Organization", "kind": "company", "description": "AI safety company"},
                {"entity_id": "capability:tool-use", "name": "Tool Use",
                 "type": "Capability", "description": "Invoke external tools"},
                {"entity_id": "capability:resource-access", "name": "Resource Access",
                 "type": "Capability", "description": "Access external data sources"},
            ],
            "edges": [
                {"src": "organization:anthropic", "tgt": "protocol:mcp", "type": "DEVELOPS"},
                {"src": "protocol:mcp", "tgt": "capability:tool-use", "type": "DEFINES"},
                {"src": "protocol:mcp", "tgt": "capability:resource-access", "type": "DEFINES"},
            ],
        },
        {
            "uri": "https://example.com/day1/agentic-overview",
            "content": """# Agentic AI Infrastructure

The agentic web relies on multiple complementary protocols. A2A handles
agent-to-agent communication while MCP handles model-to-tool integration.
Together they form the foundation of the agentic web ecosystem.
""",
            "entities": [
                {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
                 "kind": "spec", "description": "Agent-to-Agent protocol"},
                {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
                 "kind": "spec", "description": "Model Context Protocol"},
            ],
            "edges": [
                {"src": "protocol:a2a", "tgt": "protocol:mcp", "type": "COMPLEMENTS"},
            ],
        },
    ]

    # --- Day 2: news articles referencing day 1 entities ---
    DAY2_SOURCES = [
        {
            "uri": "https://example.com/day2/google-a2a-update",
            "content": """# Google Expands A2A Protocol Adoption

Google announced that its Vertex AI platform now fully supports the
A2A protocol. Microsoft and Salesforce have also committed to implementing
A2A in their agent platforms.

## Industry Impact

The Linux Foundation has formed a working group to oversee A2A governance.
""",
            "entities": [
                {"entity_id": "organization:google", "name": "Google", "type": "Organization",
                 "kind": "company", "description": "Technology company"},
                {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
                 "kind": "spec", "description": "Agent-to-Agent protocol"},
                {"entity_id": "project:vertex-ai", "name": "Vertex AI", "type": "Project",
                 "kind": "platform", "description": "Google cloud AI platform"},
                {"entity_id": "organization:microsoft", "name": "Microsoft",
                 "type": "Organization", "kind": "company", "description": "Tech company"},
                {"entity_id": "organization:salesforce", "name": "Salesforce",
                 "type": "Organization", "kind": "company", "description": "CRM company"},
                {"entity_id": "organization:linux-foundation", "name": "Linux Foundation",
                 "type": "Organization", "kind": "foundation", "description": "Open source foundation"},
            ],
            "edges": [
                {"src": "project:vertex-ai", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
                {"src": "organization:google", "tgt": "project:vertex-ai", "type": "DEVELOPS"},
                {"src": "organization:linux-foundation", "tgt": "protocol:a2a", "type": "GOVERNS"},
            ],
        },
        {
            "uri": "https://example.com/day2/anthropic-mcp-news",
            "content": """# Anthropic MCP SDK Reaches 1.0

Anthropic released MCP SDK 1.0 for both Python and TypeScript.
The SDK makes it easy for developers to build MCP-compatible tools
and connect them to any MCP-capable AI model.
""",
            "entities": [
                {"entity_id": "organization:anthropic", "name": "Anthropic",
                 "type": "Organization", "kind": "company", "description": "AI safety company"},
                {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
                 "kind": "spec", "description": "Model Context Protocol"},
                {"entity_id": "project:mcp-sdk-python", "name": "MCP SDK Python",
                 "type": "Project", "kind": "sdk", "description": "Python SDK for MCP"},
                {"entity_id": "project:mcp-sdk-typescript", "name": "MCP SDK TypeScript",
                 "type": "Project", "kind": "sdk", "description": "TypeScript SDK for MCP"},
            ],
            "edges": [
                {"src": "organization:anthropic", "tgt": "project:mcp-sdk-python", "type": "DEVELOPS"},
                {"src": "organization:anthropic", "tgt": "project:mcp-sdk-typescript", "type": "DEVELOPS"},
                {"src": "project:mcp-sdk-python", "tgt": "protocol:mcp", "type": "IMPLEMENTS"},
                {"src": "project:mcp-sdk-typescript", "tgt": "protocol:mcp", "type": "IMPLEMENTS"},
            ],
        },
    ]

    # --- Day 3: chat transcript referencing entities from days 1-2 ---
    DAY3_SOURCES = [
        {
            "uri": "https://example.com/day3/team-chat-transcript",
            "content": """# Engineering Chat Transcript - 2026-05-08

Alice: Has anyone tested Vertex AI with A2A yet? Google says it's GA.
Bob: Yes, I tried the MCP SDK Python to connect our tools. Works great.
Carol: Anthropic's MCP and Google's A2A complement each other well.
Alice: Microsoft and Salesforce are also joining the A2A ecosystem.
Bob: The Linux Foundation governance should help standardization.
""",
            "entities": [
                {"entity_id": "project:vertex-ai", "name": "Vertex AI", "type": "Project",
                 "kind": "platform", "description": "Google cloud AI platform"},
                {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
                 "kind": "spec", "description": "A2A protocol"},
                {"entity_id": "project:mcp-sdk-python", "name": "MCP SDK Python",
                 "type": "Project", "kind": "sdk", "description": "Python SDK for MCP"},
                {"entity_id": "organization:anthropic", "name": "Anthropic",
                 "type": "Organization", "kind": "company", "description": "AI safety company"},
                {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
                 "kind": "spec", "description": "Model Context Protocol"},
                {"entity_id": "organization:google", "name": "Google", "type": "Organization",
                 "kind": "company", "description": "Technology company"},
                {"entity_id": "organization:microsoft", "name": "Microsoft",
                 "type": "Organization", "kind": "company", "description": "Tech company"},
                {"entity_id": "organization:salesforce", "name": "Salesforce",
                 "type": "Organization", "kind": "company", "description": "CRM company"},
                {"entity_id": "organization:linux-foundation", "name": "Linux Foundation",
                 "type": "Organization", "kind": "foundation", "description": "Open source foundation"},
            ],
            "edges": [
                {"src": "project:vertex-ai", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
                {"src": "protocol:a2a", "tgt": "protocol:mcp", "type": "COMPLEMENTS"},
            ],
        },
    ]

    def _ingest_day(self, db, sources):
        sids = []
        for s in sources:
            sid = _full_ingest(db, s["uri"], s["content"], s["entities"], s["edges"])
            sids.append(sid)
        return sids

    def test_day1_foundation_ingestion(self, db):
        """Day 1: Ingest foundational specs, verify core entities."""
        self._ingest_day(db, self.DAY1_SOURCES)
        active = _get_active_entity_ids(db)

        assert "protocol:a2a" in active
        assert "protocol:mcp" in active
        assert "organization:google" in active
        assert "organization:anthropic" in active
        assert "capability:tool-use" in active

    def test_day2_enriches_graph(self, db):
        """Day 2: News articles add new entities, don't duplicate existing ones."""
        self._ingest_day(db, self.DAY1_SOURCES)
        day1_entities = _get_active_entity_ids(db)

        self._ingest_day(db, self.DAY2_SOURCES)
        day2_entities = _get_active_entity_ids(db)

        assert day1_entities.issubset(day2_entities)
        assert "project:vertex-ai" in day2_entities
        assert "project:mcp-sdk-python" in day2_entities
        assert "organization:salesforce" in day2_entities

    def test_day3_references_no_duplication(self, db):
        """Day 3: Chat transcript references prior entities, no duplication."""
        self._ingest_day(db, self.DAY1_SOURCES)
        self._ingest_day(db, self.DAY2_SOURCES)
        self._ingest_day(db, self.DAY3_SOURCES)

        google_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'organization:google' AND merged_into IS NULL"
        ).fetchall()
        assert len(google_rows) == 1

        a2a_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'protocol:a2a' AND merged_into IS NULL"
        ).fetchall()
        assert len(a2a_rows) == 1

    def test_graph_grows_coherently(self, db):
        """Graph grows monotonically: each day adds entities, none are lost."""
        self._ingest_day(db, self.DAY1_SOURCES)
        count1 = len(_get_active_entity_ids(db))

        self._ingest_day(db, self.DAY2_SOURCES)
        count2 = len(_get_active_entity_ids(db))

        self._ingest_day(db, self.DAY3_SOURCES)
        count3 = len(_get_active_entity_ids(db))

        assert count2 >= count1
        assert count3 >= count2

    def test_edges_accumulate_across_sessions(self, db):
        """Edges from all sessions are preserved."""
        self._ingest_day(db, self.DAY1_SOURCES)
        edges1 = _get_edge_triples(db)

        self._ingest_day(db, self.DAY2_SOURCES)
        edges2 = _get_edge_triples(db)

        assert edges1.issubset(edges2)
        assert ("project:vertex-ai", "protocol:a2a", "IMPLEMENTS") in edges2
        assert ("organization:anthropic", "project:mcp-sdk-python", "DEVELOPS") in edges2

    def test_provenance_chains_clear(self, db):
        """Each entity tracks which source it came from."""
        self._ingest_day(db, self.DAY1_SOURCES)
        self._ingest_day(db, self.DAY2_SOURCES)

        vertex = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'project:vertex-ai' AND merged_into IS NULL"
        ).fetchone()
        assert vertex is not None
        assert vertex["source_id"] is not None

        source = db.get_source(vertex["source_id"])
        assert "day2" in source["uri"]


# ============================================================
# 2. KNOWLEDGE GRAPH QUERYING PATTERNS
# ============================================================


class TestKGQueryingPatterns:
    """Test real query patterns against the SQLite graph model."""

    @pytest.fixture(autouse=True)
    def setup_graph(self, db):
        """Populate the graph with a realistic dataset for querying."""
        self.db = db

        entities = [
            {"entity_id": "organization:google", "name": "Google", "type": "Organization",
             "kind": "company", "description": "Technology company"},
            {"entity_id": "organization:anthropic", "name": "Anthropic", "type": "Organization",
             "kind": "company", "description": "AI safety company"},
            {"entity_id": "organization:microsoft", "name": "Microsoft", "type": "Organization",
             "kind": "company", "description": "Technology company"},
            {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
             "kind": "spec", "description": "Agent-to-Agent protocol"},
            {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
             "kind": "spec", "description": "Model Context Protocol"},
            {"entity_id": "protocol:openapi", "name": "OpenAPI", "type": "Protocol",
             "kind": "spec", "description": "API specification standard"},
            {"entity_id": "person:david-li", "name": "David Li", "type": "Person",
             "description": "A2A spec lead at Google"},
            {"entity_id": "person:amanda-askell", "name": "Amanda Askell", "type": "Person",
             "description": "Researcher at Anthropic"},
            {"entity_id": "capability:agent-interop", "name": "Agent Interoperability",
             "type": "Capability", "description": "Enable agents to communicate"},
            {"entity_id": "capability:tool-use", "name": "Tool Use",
             "type": "Capability", "description": "Invoke external tools"},
            {"entity_id": "project:vertex-ai", "name": "Vertex AI", "type": "Project",
             "kind": "platform", "description": "Google AI platform"},
        ]

        edges_data = [
            {"src": "organization:google", "tgt": "protocol:a2a", "type": "DEVELOPS"},
            {"src": "organization:anthropic", "tgt": "protocol:mcp", "type": "DEVELOPS"},
            {"src": "protocol:a2a", "tgt": "capability:agent-interop", "type": "ADDRESSES"},
            {"src": "protocol:mcp", "tgt": "capability:tool-use", "type": "ADDRESSES"},
            {"src": "protocol:openapi", "tgt": "capability:tool-use", "type": "ADDRESSES"},
            {"src": "person:david-li", "tgt": "protocol:a2a", "type": "AUTHORED"},
            {"src": "person:amanda-askell", "tgt": "organization:anthropic", "type": "MEMBER_OF"},
            {"src": "protocol:a2a", "tgt": "protocol:mcp", "type": "COMPLEMENTS"},
            {"src": "organization:google", "tgt": "project:vertex-ai", "type": "DEVELOPS"},
            {"src": "project:vertex-ai", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
        ]

        src1 = db.add_source("https://example.com/query-data-1")
        src2 = db.add_source("https://example.com/query-data-2")

        for ent in entities:
            db.add_entity(
                entity_id=ent["entity_id"], name=ent["name"],
                entity_type=ent["type"], kind=ent.get("kind"),
                description=ent.get("description"),
                source_id=src1 if "google" in ent["entity_id"] or "a2a" in ent["entity_id"] else src2,
            )

        for e in edges_data:
            eid = _make_edge_id(e["src"], e["tgt"], e["type"])
            db.add_edge(eid, e["src"], e["tgt"], e["type"],
                        confidence=0.9,
                        source_id=src1 if "google" in e["src"] or "a2a" in e["src"] else src2)

        _approve_all(db)

    def test_org_to_protocol_traversal(self):
        """'What protocols does Google support?' — org → DEVELOPS → protocol."""
        rows = self.db.conn.execute(
            """SELECT e.entity_id, e.name FROM entities e
               JOIN edges ed ON ed.target_entity_id = e.entity_id
               WHERE ed.source_entity_id = 'organization:google'
               AND ed.edge_type = 'DEVELOPS'
               AND e.type = 'Protocol'""",
        ).fetchall()
        proto_ids = {r["entity_id"] for r in rows}
        assert "protocol:a2a" in proto_ids

    def test_protocol_to_person_traversal(self):
        """'Who are key people in A2A?' — protocol ← AUTHORED ← person."""
        rows = self.db.conn.execute(
            """SELECT e.entity_id, e.name FROM entities e
               JOIN edges ed ON ed.source_entity_id = e.entity_id
               WHERE ed.target_entity_id = 'protocol:a2a'
               AND ed.edge_type = 'AUTHORED'
               AND e.type = 'Person'""",
        ).fetchall()
        person_ids = {r["entity_id"] for r in rows}
        assert "person:david-li" in person_ids

    def test_competing_approaches_query(self):
        """'What protocols address agent interop?' — capability ← ADDRESSES ← protocol."""
        rows = self.db.conn.execute(
            """SELECT e.entity_id, e.name FROM entities e
               JOIN edges ed ON ed.source_entity_id = e.entity_id
               WHERE ed.target_entity_id = 'capability:agent-interop'
               AND ed.edge_type = 'ADDRESSES'
               AND e.type = 'Protocol'""",
        ).fetchall()
        proto_ids = {r["entity_id"] for r in rows}
        assert "protocol:a2a" in proto_ids

    def test_provenance_scoped_query(self):
        """'Show me everything from source X' — entities filtered by source."""
        rows = self.db.conn.execute(
            """SELECT entity_id FROM entities
               WHERE source_id = (SELECT id FROM sources WHERE uri = 'https://example.com/query-data-1')""",
        ).fetchall()
        entity_ids = {r["entity_id"] for r in rows}
        assert "organization:google" in entity_ids
        assert "protocol:a2a" in entity_ids

    def test_temporal_query(self):
        """'What was added recently?' — entities ordered by created_at."""
        rows = self.db.conn.execute(
            """SELECT entity_id, created_at FROM entities
               ORDER BY created_at DESC LIMIT 5""",
        ).fetchall()
        assert len(rows) > 0
        for r in rows:
            assert r["created_at"] is not None

    def test_multi_hop_query(self):
        """'What implements protocols developed by Google?' — 2-hop traversal."""
        rows = self.db.conn.execute(
            """SELECT impl.entity_id, impl.name FROM entities impl
               JOIN edges e1 ON e1.source_entity_id = impl.entity_id AND e1.edge_type = 'IMPLEMENTS'
               JOIN edges e2 ON e2.target_entity_id = e1.target_entity_id AND e2.edge_type = 'DEVELOPS'
               WHERE e2.source_entity_id = 'organization:google'""",
        ).fetchall()
        project_ids = {r["entity_id"] for r in rows}
        assert "project:vertex-ai" in project_ids

    def test_complementary_protocols_query(self):
        """'What protocols complement A2A?'"""
        rows = self.db.conn.execute(
            """SELECT e.entity_id FROM entities e
               JOIN edges ed ON (
                   (ed.source_entity_id = 'protocol:a2a' AND ed.target_entity_id = e.entity_id)
                   OR (ed.target_entity_id = 'protocol:a2a' AND ed.source_entity_id = e.entity_id)
               )
               WHERE ed.edge_type = 'COMPLEMENTS'""",
        ).fetchall()
        ids = {r["entity_id"] for r in rows}
        assert "protocol:mcp" in ids

    def test_entity_neighborhood(self):
        """Get all edges touching a specific entity (both directions)."""
        entity_id = "protocol:a2a"
        rows = self.db.conn.execute(
            """SELECT source_entity_id, target_entity_id, edge_type FROM edges
               WHERE source_entity_id = ? OR target_entity_id = ?""",
            (entity_id, entity_id),
        ).fetchall()
        assert len(rows) >= 3


# ============================================================
# 3. SOURCE LIFECYCLE: FULL CYCLE
# ============================================================


class TestSourceLifecycleFullCycle:
    """Full lifecycle: ingest → update → deprecate → re-ingest."""

    INITIAL_CONTENT = """# MCP Implementation Guide v1

MCP is developed by Anthropic. It supports tool invocation and
resource access. The SDK is available in Python and TypeScript.
"""
    INITIAL_ENTITIES = [
        {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
         "kind": "spec", "description": "Model Context Protocol v1"},
        {"entity_id": "organization:anthropic", "name": "Anthropic",
         "type": "Organization", "kind": "company", "description": "AI safety company"},
        {"entity_id": "capability:tool-use", "name": "Tool Use",
         "type": "Capability", "description": "Invoke external tools"},
    ]
    INITIAL_EDGES = [
        {"src": "organization:anthropic", "tgt": "protocol:mcp", "type": "DEVELOPS"},
        {"src": "protocol:mcp", "tgt": "capability:tool-use", "type": "DEFINES"},
    ]

    UPDATED_CONTENT = """# MCP Implementation Guide v2

MCP v2 is developed by Anthropic with contributions from Microsoft.
It supports tool invocation, resource access, prompt templates, and sampling.
"""
    UPDATED_ENTITIES = [
        {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
         "kind": "spec", "description": "Model Context Protocol v2"},
        {"entity_id": "organization:anthropic", "name": "Anthropic",
         "type": "Organization", "kind": "company", "description": "AI safety company"},
        {"entity_id": "organization:microsoft", "name": "Microsoft",
         "type": "Organization", "kind": "company", "description": "Tech company"},
        {"entity_id": "capability:tool-use", "name": "Tool Use",
         "type": "Capability", "description": "Invoke external tools"},
        {"entity_id": "capability:sampling", "name": "Sampling",
         "type": "Capability", "description": "Request LLM completions from client"},
    ]
    UPDATED_EDGES = [
        {"src": "organization:anthropic", "tgt": "protocol:mcp", "type": "DEVELOPS"},
        {"src": "organization:microsoft", "tgt": "protocol:mcp", "type": "CONTRIBUTES_TO"},
        {"src": "protocol:mcp", "tgt": "capability:tool-use", "type": "DEFINES"},
        {"src": "protocol:mcp", "tgt": "capability:sampling", "type": "DEFINES"},
    ]

    def test_initial_ingest_creates_entities(self, db):
        sid = _full_ingest(db, "https://example.com/mcp-guide",
                           self.INITIAL_CONTENT, self.INITIAL_ENTITIES, self.INITIAL_EDGES)
        active = _get_active_entity_ids(db)
        assert "protocol:mcp" in active
        assert "organization:anthropic" in active
        assert "capability:tool-use" in active

    def test_update_source_deprecates_old(self, db):
        """After reset + re-ingest, old chunks are gone and new entities appear."""
        sid = _full_ingest(db, "https://example.com/mcp-guide-upd",
                           self.INITIAL_CONTENT, self.INITIAL_ENTITIES, self.INITIAL_EDGES)

        old_chunks = db.get_chunks(sid)
        assert len(old_chunks) > 0

        db.reset_source(sid)
        source = db.get_source(sid)
        assert source["status"] == "pending"

        reset_chunks = db.get_chunks(sid)
        assert len(reset_chunks) == 0

        _run_through_chunk(db, sid, self.UPDATED_CONTENT)
        source = db.get_source(sid)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, self.UPDATED_ENTITIES, self.UPDATED_EDGES)
        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)
        _approve_all(db)

        active = _get_active_entity_ids(db)
        assert "capability:sampling" in active
        assert "organization:microsoft" in active

    def test_deprecate_source_marks_entities(self, db):
        """Deprecating a source marks its entities as deprecated."""
        sid = _full_ingest(db, "https://example.com/mcp-guide-dep",
                           self.INITIAL_CONTENT, self.INITIAL_ENTITIES, self.INITIAL_EDGES)

        db.deprecate_entities_for_source(sid)

        deprecated = db.get_deprecated_entities()
        dep_ids = {e["entity_id"] for e in deprecated}
        assert len(dep_ids) > 0

    def test_unique_entities_survive_other_source_deprecation(self, db):
        """Entities unique to source 2 survive deprecation of source 1."""
        sid1 = _full_ingest(db, "https://example.com/lifecycle-src1",
                            self.INITIAL_CONTENT, self.INITIAL_ENTITIES, self.INITIAL_EDGES)

        other_entities = [
            {"entity_id": "project:claude", "name": "Claude", "type": "Project",
             "kind": "platform", "description": "AI assistant"},
            {"entity_id": "capability:code-generation", "name": "Code Generation",
             "type": "Capability", "description": "Generate code from prompts"},
        ]
        other_edges = [
            {"src": "project:claude", "tgt": "capability:code-generation", "type": "ADDRESSES"},
        ]
        sid2 = _full_ingest(db, "https://example.com/lifecycle-src2",
                            "# Claude\nClaude generates code.", other_entities, other_edges)

        db.deprecate_entities_for_source(sid1)

        claude_entity = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'project:claude' "
            "AND source_id = ? AND deprecated_at IS NULL", (sid2,)
        ).fetchone()
        assert claude_entity is not None

        src1_deprecated = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND deprecated_at IS NOT NULL", (sid1,)
        ).fetchall()
        assert len(src1_deprecated) > 0

    def test_reingest_after_reset_recovers_graph(self, db):
        """After reset (not deprecation), re-ingesting the same source recovers the graph."""
        sid = _full_ingest(db, "https://example.com/mcp-guide-recover",
                           self.INITIAL_CONTENT, self.INITIAL_ENTITIES, self.INITIAL_EDGES)

        active_before = _get_active_entity_ids(db)
        assert "protocol:mcp" in active_before

        db.reset_source(sid)
        source = db.get_source(sid)
        assert source["status"] == "pending"

        entities_after_reset = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(entities_after_reset) == 0

        _run_through_chunk(db, sid, self.UPDATED_CONTENT)
        source = db.get_source(sid)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, self.UPDATED_ENTITIES, self.UPDATED_EDGES)
        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)
        _approve_all(db)

        active = _get_active_entity_ids(db)
        assert "capability:sampling" in active

    def test_full_lifecycle_reset_update_deprecate(self, db):
        """Ingest → reset → update → deprecate sequence."""
        sid = _full_ingest(db, "https://example.com/full-lifecycle",
                           self.INITIAL_CONTENT, self.INITIAL_ENTITIES, self.INITIAL_EDGES)
        source = db.get_source(sid)
        assert source["stage"] == "review"

        active_v1 = _get_active_entity_ids(db)
        assert "protocol:mcp" in active_v1
        assert "capability:sampling" not in active_v1

        db.reset_source(sid)
        _run_through_chunk(db, sid, self.UPDATED_CONTENT)
        source = db.get_source(sid)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, self.UPDATED_ENTITIES, self.UPDATED_EDGES)
        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)
        _approve_all(db)

        active_v2 = _get_active_entity_ids(db)
        assert "capability:sampling" in active_v2

        db.deprecate_entities_for_source(sid)
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) > 0

        still_active = _get_active_entity_ids(db)
        assert "capability:sampling" not in still_active


# ============================================================
# 4. REVIEW AND AUDIT PATTERNS
# ============================================================


class TestReviewAndAuditPatterns:
    """Audit queries: stale entities, orphans, single-source, clusters."""

    def _populate_with_ages(self, db):
        """Create entities with various ages and staleness levels."""
        now = datetime.now(timezone.utc)
        old_date = (now - timedelta(days=120)).isoformat()
        recent_date = (now - timedelta(days=10)).isoformat()

        src = db.add_source("https://example.com/audit-data")

        old_entities = [
            ("protocol:legacy-proto", "Legacy Protocol", "Protocol", old_date),
            ("organization:defunct-corp", "Defunct Corp", "Organization", old_date),
            ("capability:old-feature", "Old Feature", "Capability", old_date),
        ]
        for eid, name, etype, ts in old_entities:
            db.conn.execute(
                "INSERT INTO entities (entity_id, name, type, source_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'approved', ?, ?)",
                (eid, name, etype, src, ts, ts),
            )

        recent_entities = [
            ("protocol:new-proto", "New Protocol", "Protocol", recent_date),
            ("organization:active-corp", "Active Corp", "Organization", recent_date),
        ]
        for eid, name, etype, ts in recent_entities:
            db.conn.execute(
                "INSERT INTO entities (entity_id, name, type, source_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'approved', ?, ?)",
                (eid, name, etype, src, ts, ts),
            )

        db.conn.commit()
        return src

    def test_find_stale_entities(self, db):
        """Find entities not updated in 90 days."""
        self._populate_with_ages(db)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

        rows = db.conn.execute(
            "SELECT entity_id, name, updated_at FROM entities "
            "WHERE updated_at < ? AND status = 'approved'",
            (cutoff,),
        ).fetchall()

        stale_ids = {r["entity_id"] for r in rows}
        assert "protocol:legacy-proto" in stale_ids
        assert "organization:defunct-corp" in stale_ids
        assert "protocol:new-proto" not in stale_ids

    def test_find_single_source_entities(self, db):
        """Find entities backed by only one source."""
        src1 = db.add_source("https://example.com/audit-single-1")
        src2 = db.add_source("https://example.com/audit-single-2")

        db.add_entity("protocol:well-sourced", "Well Sourced", "Protocol", source_id=src1)
        db.add_entity("protocol:well-sourced-2", "Well Sourced", "Protocol", source_id=src2)
        db.add_entity("protocol:lone-entity", "Lone Entity", "Protocol", source_id=src1)

        rows = db.conn.execute(
            """SELECT entity_id, COUNT(DISTINCT source_id) as source_count
               FROM entities
               WHERE merged_into IS NULL AND status != 'rejected'
               GROUP BY entity_id
               HAVING source_count = 1"""
        ).fetchall()

        single_source = {r["entity_id"] for r in rows}
        assert "protocol:lone-entity" in single_source

    def test_find_orphan_entities_no_edges(self, db):
        """Find entities with no relationships (orphans)."""
        sid = _full_ingest(db, "https://example.com/audit-orphan",
                           "# Orphan test", [
                               {"entity_id": "project:connected", "name": "Connected",
                                "type": "Project", "description": "Has edges"},
                               {"entity_id": "project:orphan", "name": "Orphan",
                                "type": "Project", "description": "No edges"},
                               {"entity_id": "organization:owner", "name": "Owner",
                                "type": "Organization", "description": "Org"},
                           ], [
                               {"src": "organization:owner", "tgt": "project:connected",
                                "type": "DEVELOPS"},
                           ])

        rows = db.conn.execute(
            """SELECT e.entity_id FROM entities e
               WHERE e.merged_into IS NULL AND e.status = 'approved'
               AND e.entity_id NOT IN (
                   SELECT source_entity_id FROM edges
                   UNION
                   SELECT target_entity_id FROM edges
               )"""
        ).fetchall()
        orphan_ids = {r["entity_id"] for r in rows}
        assert "project:orphan" in orphan_ids
        assert "project:connected" not in orphan_ids

    def test_find_entity_clusters(self, db):
        """Identify connected components via edge grouping."""
        _full_ingest(db, "https://example.com/audit-cluster", "# Clusters", [
            {"entity_id": "organization:cluster-a1", "name": "ClusterA1",
             "type": "Organization", "description": "Part of cluster A"},
            {"entity_id": "project:cluster-a2", "name": "ClusterA2",
             "type": "Project", "description": "Part of cluster A"},
            {"entity_id": "organization:cluster-b1", "name": "ClusterB1",
             "type": "Organization", "description": "Part of cluster B"},
            {"entity_id": "project:cluster-b2", "name": "ClusterB2",
             "type": "Project", "description": "Part of cluster B"},
            {"entity_id": "project:isolated", "name": "Isolated",
             "type": "Project", "description": "No connections"},
        ], [
            {"src": "organization:cluster-a1", "tgt": "project:cluster-a2", "type": "DEVELOPS"},
            {"src": "organization:cluster-b1", "tgt": "project:cluster-b2", "type": "DEVELOPS"},
        ])

        edges = db.conn.execute("SELECT source_entity_id, target_entity_id FROM edges").fetchall()
        adjacency = {}
        for e in edges:
            s, t = e["source_entity_id"], e["target_entity_id"]
            adjacency.setdefault(s, set()).add(t)
            adjacency.setdefault(t, set()).add(s)

        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in adjacency:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        assert len(components) >= 2
        component_sizes = sorted([len(c) for c in components], reverse=True)
        assert component_sizes[0] == 2
        assert component_sizes[1] == 2

    def test_status_summary(self, db):
        """Status summary returns correct counts."""
        db.add_source("https://example.com/audit-status-1")
        sid2 = db.add_source("https://example.com/audit-status-2")
        db.update_source(sid2, status="complete")

        summary = db.status_summary()
        assert summary.get("pending", 0) >= 1
        assert summary.get("complete", 0) >= 1

    def test_entity_status_breakdown(self, db):
        """Count entities by status for audit dashboard."""
        sid = db.add_source("https://example.com/audit-status-ent")
        db.add_entity("project:pending-ent", "Pending", "Project", source_id=sid)
        id2 = db.add_entity("project:approved-ent", "Approved", "Project", source_id=sid)
        db.approve_entity(id2)

        rows = db.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM entities GROUP BY status"
        ).fetchall()
        breakdown = {r["status"]: r["cnt"] for r in rows}
        assert breakdown.get("pending_review", 0) >= 1
        assert breakdown.get("approved", 0) >= 1


# ============================================================
# 5. REAL DOMAIN CONTENT
# ============================================================


class TestRealDomainContent:
    """Test with realistic agentic-web domain content."""

    def test_a2a_protocol_summary(self, db):
        """A2A protocol summary produces domain-appropriate entities."""
        content = """# Agent-to-Agent (A2A) Protocol

Google developed the A2A protocol to enable AI agents from different
vendors to communicate and collaborate. A2A uses JSON-RPC over HTTP
and defines an AgentCard discovery mechanism.

## Supported Capabilities
- Task management with lifecycle tracking
- Streaming via Server-Sent Events (SSE)
- Push notifications for async workflows
- Multi-turn conversations between agents

## Implementations
Vertex AI Agent Builder fully supports A2A. Third-party implementations
include LangGraph and CrewAI adapters.
"""
        entities = [
            {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
             "kind": "spec", "description": "Agent-to-Agent interoperability protocol"},
            {"entity_id": "organization:google", "name": "Google", "type": "Organization",
             "kind": "company", "description": "Technology company"},
            {"entity_id": "capability:task-management", "name": "Task Management",
             "type": "Capability", "description": "Manage agent tasks with lifecycle tracking"},
            {"entity_id": "capability:streaming", "name": "Streaming",
             "type": "Capability", "description": "Real-time SSE streaming"},
            {"entity_id": "capability:push-notifications", "name": "Push Notifications",
             "type": "Capability", "description": "Async workflow notifications"},
            {"entity_id": "project:vertex-ai", "name": "Vertex AI", "type": "Project",
             "kind": "platform", "description": "Google AI platform"},
            {"entity_id": "project:langgraph", "name": "LangGraph", "type": "Project",
             "kind": "framework", "description": "Agent orchestration framework"},
            {"entity_id": "project:crewai", "name": "CrewAI", "type": "Project",
             "kind": "framework", "description": "Multi-agent framework"},
        ]
        edges = [
            {"src": "organization:google", "tgt": "protocol:a2a", "type": "DEVELOPS"},
            {"src": "protocol:a2a", "tgt": "capability:task-management", "type": "DEFINES"},
            {"src": "protocol:a2a", "tgt": "capability:streaming", "type": "DEFINES"},
            {"src": "protocol:a2a", "tgt": "capability:push-notifications", "type": "DEFINES"},
            {"src": "project:vertex-ai", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
            {"src": "project:langgraph", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
            {"src": "project:crewai", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
            {"src": "organization:google", "tgt": "project:vertex-ai", "type": "DEVELOPS"},
        ]

        sid = _full_ingest(db, "https://example.com/domain/a2a-summary",
                           content, entities, edges)
        active = _get_active_entity_ids(db)
        assert "protocol:a2a" in active
        assert "project:vertex-ai" in active
        assert "capability:task-management" in active

        impl_edges = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'IMPLEMENTS' AND target_entity_id = 'protocol:a2a'"
        ).fetchall()
        assert len(impl_edges) >= 3

    def test_mcp_server_implementation_guide(self, db):
        """MCP server guide produces correct entity types."""
        content = """# Building an MCP Server

Anthropic's Model Context Protocol (MCP) allows you to expose tools,
resources, and prompts to AI models. The MCP Python SDK provides
decorators for easy server implementation.

## Architecture
An MCP server exposes:
- Tools: executable functions the model can call
- Resources: data the model can read
- Prompts: reusable prompt templates

## Transport Options
MCP supports stdio transport for local development and SSE transport
for remote deployment.
"""
        entities = [
            {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
             "kind": "spec", "description": "Model Context Protocol"},
            {"entity_id": "organization:anthropic", "name": "Anthropic",
             "type": "Organization", "kind": "company", "description": "AI safety company"},
            {"entity_id": "project:mcp-sdk-python", "name": "MCP Python SDK",
             "type": "Project", "kind": "sdk", "description": "Python SDK for MCP"},
            {"entity_id": "capability:tool-use", "name": "Tool Use",
             "type": "Capability", "description": "Executable functions for AI models"},
            {"entity_id": "capability:resource-access", "name": "Resource Access",
             "type": "Capability", "description": "Read external data"},
            {"entity_id": "capability:prompt-templates", "name": "Prompt Templates",
             "type": "Capability", "description": "Reusable prompt definitions"},
        ]
        edges = [
            {"src": "organization:anthropic", "tgt": "protocol:mcp", "type": "DEVELOPS"},
            {"src": "organization:anthropic", "tgt": "project:mcp-sdk-python", "type": "DEVELOPS"},
            {"src": "project:mcp-sdk-python", "tgt": "protocol:mcp", "type": "IMPLEMENTS"},
            {"src": "protocol:mcp", "tgt": "capability:tool-use", "type": "DEFINES"},
            {"src": "protocol:mcp", "tgt": "capability:resource-access", "type": "DEFINES"},
            {"src": "protocol:mcp", "tgt": "capability:prompt-templates", "type": "DEFINES"},
        ]

        sid = _full_ingest(db, "https://example.com/domain/mcp-guide",
                           content, entities, edges)
        active = _get_active_entity_ids(db)
        assert "protocol:mcp" in active
        assert "project:mcp-sdk-python" in active
        assert "capability:tool-use" in active
        assert "capability:prompt-templates" in active

    def test_agent_framework_comparison(self, db):
        """Comparison of agent frameworks produces correct competition edges."""
        content = """# Agent Framework Comparison 2026

## CrewAI
CrewAI by Joao Moura enables role-based multi-agent collaboration.
It supports sequential and hierarchical agent workflows.

## LangGraph
LangGraph by LangChain Inc provides graph-based agent orchestration.
It models agent workflows as state machines with cycles.

## AutoGen
AutoGen by Microsoft Research enables multi-agent conversations.
It focuses on conversational patterns between agents.

## Comparison
All three frameworks address agent orchestration but with different
paradigms: CrewAI uses roles, LangGraph uses graphs, AutoGen uses
conversations.
"""
        entities = [
            {"entity_id": "project:crewai", "name": "CrewAI", "type": "Project",
             "kind": "framework", "description": "Role-based multi-agent framework"},
            {"entity_id": "person:joao-moura", "name": "Joao Moura", "type": "Person",
             "description": "Creator of CrewAI"},
            {"entity_id": "project:langgraph", "name": "LangGraph", "type": "Project",
             "kind": "framework", "description": "Graph-based agent orchestration"},
            {"entity_id": "organization:langchain-inc", "name": "LangChain Inc",
             "type": "Organization", "kind": "company", "description": "LLM tooling company"},
            {"entity_id": "project:autogen", "name": "AutoGen", "type": "Project",
             "kind": "framework", "description": "Multi-agent conversation framework"},
            {"entity_id": "organization:microsoft-research", "name": "Microsoft Research",
             "type": "Organization", "kind": "company", "description": "Research division"},
            {"entity_id": "capability:agent-orchestration", "name": "Agent Orchestration",
             "type": "Capability", "description": "Coordinate multiple AI agents"},
        ]
        edges = [
            {"src": "person:joao-moura", "tgt": "project:crewai", "type": "AUTHORED"},
            {"src": "organization:langchain-inc", "tgt": "project:langgraph", "type": "DEVELOPS"},
            {"src": "organization:microsoft-research", "tgt": "project:autogen", "type": "DEVELOPS"},
            {"src": "project:crewai", "tgt": "capability:agent-orchestration", "type": "ADDRESSES"},
            {"src": "project:langgraph", "tgt": "capability:agent-orchestration", "type": "ADDRESSES"},
            {"src": "project:autogen", "tgt": "capability:agent-orchestration", "type": "ADDRESSES"},
            {"src": "project:crewai", "tgt": "project:langgraph", "type": "COMPETES_WITH"},
            {"src": "project:crewai", "tgt": "project:autogen", "type": "COMPETES_WITH"},
            {"src": "project:langgraph", "tgt": "project:autogen", "type": "COMPETES_WITH"},
        ]

        sid = _full_ingest(db, "https://example.com/domain/framework-comparison",
                           content, entities, edges)

        active = _get_active_entity_ids(db)
        assert "project:crewai" in active
        # LangGraph merges into project:langchain via seed alias resolution
        assert "project:langchain" in active or "project:langgraph" in active
        assert "project:autogen" in active
        assert "capability:agent-orchestration" in active

        compete_edges = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'COMPETES_WITH'"
        ).fetchall()
        assert len(compete_edges) >= 3

        addresses_edges = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'ADDRESSES'"
        ).fetchall()
        orchestration_targets = [e for e in addresses_edges
                                  if e["target_entity_id"] == "capability:agent-orchestration"]
        assert len(orchestration_targets) >= 2

    def test_agentic_tool_news(self, db):
        """News article about a new agentic tool produces correct relationships."""
        content = """# Acme Corp Launches AgentForge

Acme Corp today announced AgentForge, an enterprise platform for
deploying AI agents at scale. AgentForge implements both the A2A
protocol for agent-to-agent communication and MCP for tool integration.

CEO Jane Smith said: "AgentForge bridges the gap between A2A and MCP,
giving enterprises a unified agent platform."

AgentForge is built on top of Kubernetes and integrates with existing
CI/CD pipelines.
"""
        entities = [
            {"entity_id": "organization:acme-corp", "name": "Acme Corp",
             "type": "Organization", "kind": "company", "description": "Enterprise AI company"},
            {"entity_id": "project:agentforge", "name": "AgentForge", "type": "Project",
             "kind": "platform", "description": "Enterprise agent platform"},
            {"entity_id": "person:jane-smith", "name": "Jane Smith", "type": "Person",
             "description": "CEO of Acme Corp"},
            {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
             "kind": "spec", "description": "Agent-to-Agent protocol"},
            {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
             "kind": "spec", "description": "Model Context Protocol"},
        ]
        edges = [
            {"src": "organization:acme-corp", "tgt": "project:agentforge", "type": "DEVELOPS"},
            {"src": "person:jane-smith", "tgt": "organization:acme-corp", "type": "MEMBER_OF"},
            {"src": "project:agentforge", "tgt": "protocol:a2a", "type": "IMPLEMENTS"},
            {"src": "project:agentforge", "tgt": "protocol:mcp", "type": "IMPLEMENTS"},
        ]

        sid = _full_ingest(db, "https://example.com/domain/agentforge-launch",
                           content, entities, edges)

        active = _get_active_entity_ids(db)
        assert "organization:acme-corp" in active
        assert "project:agentforge" in active
        assert "person:jane-smith" in active

        impl_edges = db.conn.execute(
            "SELECT target_entity_id FROM edges "
            "WHERE source_entity_id = 'project:agentforge' AND edge_type = 'IMPLEMENTS'"
        ).fetchall()
        targets = {r["target_entity_id"] for r in impl_edges}
        assert "protocol:a2a" in targets
        assert "protocol:mcp" in targets

    def test_edge_types_are_valid(self, db):
        """All edges in domain content tests use valid edge types."""
        _full_ingest(db, "https://example.com/domain/edge-validation",
                     "# Validation content", [
                         {"entity_id": "organization:test-org", "name": "Test",
                          "type": "Organization", "description": "Test org"},
                         {"entity_id": "project:test-proj", "name": "TestProj",
                          "type": "Project", "description": "Test project"},
                     ], [
                         {"src": "organization:test-org", "tgt": "project:test-proj",
                          "type": "DEVELOPS"},
                     ])

        edges = db.conn.execute("SELECT edge_type FROM edges").fetchall()
        for e in edges:
            assert e["edge_type"] in VALID_EDGE_TYPES, f"Invalid edge type: {e['edge_type']}"

    def test_entity_types_are_valid(self, db):
        """All entities use valid entity types."""
        _full_ingest(db, "https://example.com/domain/ent-validation",
                     "# Entity validation", [
                         {"entity_id": "organization:val-org", "name": "ValOrg",
                          "type": "Organization", "description": "Validation org"},
                         {"entity_id": "protocol:val-proto", "name": "ValProto",
                          "type": "Protocol", "description": "Validation proto"},
                         {"entity_id": "capability:val-cap", "name": "ValCap",
                          "type": "Capability", "description": "Validation cap"},
                         {"entity_id": "person:val-person", "name": "ValPerson",
                          "type": "Person", "description": "Validation person"},
                         {"entity_id": "project:val-project", "name": "ValProject",
                          "type": "Project", "description": "Validation project"},
                     ], [])

        entities = db.conn.execute(
            "SELECT type FROM entities WHERE merged_into IS NULL"
        ).fetchall()
        for e in entities:
            assert e["type"] in VALID_ENTITY_TYPES, f"Invalid entity type: {e['type']}"

    def test_cypher_generation_for_domain_entities(self, db):
        """Cypher queries generated for domain entities are well-formed."""
        entity = {
            "entity_id": "project:agentforge",
            "name": "AgentForge",
            "type": "Project",
            "kind": "platform",
            "description": "Enterprise agent platform",
            "aliases": json.dumps(["AF", "Agent Forge"]),
            "source_id": 1,
        }
        query, params = _entity_to_cypher(entity)
        assert "MERGE" in query
        assert "$entity_id" in query
        assert params["entity_id"] == "project:agentforge"
        assert params["name"] == "AgentForge"
        assert "Project" in query

    def test_cypher_generation_for_domain_edges(self, db):
        """Cypher queries generated for domain edges are well-formed."""
        edge = {
            "edge_id": "test-edge-123",
            "source_entity_id": "project:agentforge",
            "target_entity_id": "protocol:a2a",
            "edge_type": "IMPLEMENTS",
            "properties": json.dumps({}),
            "confidence": 0.95,
            "source_type": "automated",
            "valid_from": "2026-01-01",
            "valid_to": None,
            "chunk_id": None,
        }
        query, params = _edge_to_cypher(edge)
        assert "MERGE" in query
        assert "IMPLEMENTS" in query
        assert params["src"] == "project:agentforge"
        assert params["tgt"] == "protocol:a2a"
        assert params["confidence"] == 0.95

    def test_domain_content_chunking_quality(self, db):
        """Domain content chunks preserve semantic boundaries."""
        content = """# MCP Architecture Overview

## Transport Layer

MCP supports multiple transport mechanisms: stdio for local processes,
HTTP with SSE for remote servers, and WebSocket for bidirectional streaming.

## Tool Definition

Tools are defined with JSON Schema input schemas. The server declares
available tools via the tools/list endpoint.

## Resource System

Resources provide read-only access to data. Each resource has a URI
and a MIME type. Resources can be static or dynamic.

## Prompt Templates

Prompts allow servers to define reusable prompt templates with
parameter substitution and role-based message construction.
"""
        sid = db.add_source("https://example.com/domain/mcp-arch")
        _run_through_chunk(db, sid, content)

        chunks = db.get_chunks(sid)
        assert len(chunks) >= 4

        headings = [c["section_heading"] for c in chunks if c["section_heading"]]
        assert any("Transport" in h for h in headings)
        assert any("Tool" in h for h in headings)

        for chunk in chunks:
            assert chunk["text"].strip() != ""
            assert chunk["token_count"] is not None
            assert chunk["token_count"] > 0

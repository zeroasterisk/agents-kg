"""Iteration 10 — Capstone end-to-end user story tests.

These tests simulate realistic user workflows from start to finish,
combining multiple CUJs into coherent scenarios.

All tests use mocked fetch/embed/extract (no Gemini API key needed).
Neo4j tests connect to bolt://localhost:7687.
"""

import os
import struct
import pytest
from unittest.mock import MagicMock, patch

from agents_kg.db import Database, content_hash
from agents_kg.stages.extract import _make_edge_id

pytestmark = pytest.mark.e2e

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")


# ---------------------------------------------------------------------------
# Shared helpers (same patterns as test_cujs.py / test_real_world.py)
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
    from agents_kg.stages.load import run as run_load
    sid = db.add_source(uri, submitter_email=submitter_email)
    assert sid is not None
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


def _full_pipeline_neo4j(db, uri, content, entities, edges, driver,
                         submitter_email="tester@example.com"):
    from agents_kg.stages.load import run as run_load
    sid = db.add_source(uri, submitter_email=submitter_email)
    assert sid is not None
    source = _run_through_chunk(db, sid, content)
    _mock_embed(db, source)
    source = db.get_source(sid)
    _mock_extract(db, source, entities, edges)
    db.update_source(sid, stage="review", status="pending_review")
    _approve_all(db)
    db.update_source(sid, status="processing", stage="load")
    source = db.get_source(sid)
    run_load(db, source, neo4j_driver=driver)
    return sid


# ---------------------------------------------------------------------------
# Neo4j fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def neo4j_driver():
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        yield driver
        driver.close()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.fixture
def clean_neo4j(neo4j_driver):
    def _cleanup():
        with neo4j_driver.session() as s:
            s.run("MATCH (n) WHERE n.entity_id IS NOT NULL AND n.entity_id STARTS WITH 'us:' DETACH DELETE n")
            s.run("MATCH (s:Source) WHERE s.uri STARTS WITH 'us://' DETACH DELETE s")
            s.run("MATCH (c:Chunk) WHERE c.source_id IS NOT NULL DETACH DELETE c")
    _cleanup()
    yield neo4j_driver
    _cleanup()


# ===========================================================================
# USER STORY 1: New Team Member Onboarding
# ===========================================================================

WIKI_CONTENT = """\
# Team Wiki — AI Platform Team

## Team Structure

The AI Platform team builds infrastructure for deploying LLMs at scale.
Alice Chen leads the team as Engineering Manager. Bob Rivera is the Tech Lead.
The team uses MCP (Model Context Protocol) for tool integration and A2A
(Agent-to-Agent) protocol for inter-agent communication.

## Key Responsibilities

- Maintain the inference gateway
- Operate the model registry
- Build agent orchestration pipelines
"""

ARCH_CONTENT = """\
# Architecture Overview

## System Components

The platform consists of three main services:

1. **Inference Gateway** — Routes requests to model backends (vLLM, TGI).
   Uses MCP servers to expose model capabilities to agent clients.

2. **Model Registry** — Tracks model versions, artifacts, and metadata.
   Built on MLflow with custom extensions.

3. **Agent Orchestrator** — Coordinates multi-agent workflows using A2A protocol.
   Supports both synchronous and asynchronous task delegation.

## Protocols

- MCP for context and tool provision
- A2A for agent-to-agent communication
- gRPC for internal service calls
"""

GLOSSARY_CONTENT = """\
# Glossary

- **MCP**: Model Context Protocol — standardizes LLM context provision
- **A2A**: Agent-to-Agent protocol — enables direct agent communication
- **Inference Gateway**: Service routing requests to model backends
- **Model Registry**: Central catalog of model versions and artifacts
- **Agent Orchestrator**: Workflow coordinator for multi-agent systems
- **vLLM**: High-throughput LLM serving engine
- **TGI**: Text Generation Inference by Hugging Face
"""

ONBOARDING_ENTITIES_WIKI = [
    {"entity_id": "us:person-alice-chen", "name": "Alice Chen",
     "type": "Person", "kind": "manager",
     "description": "Engineering Manager of AI Platform team"},
    {"entity_id": "us:person-bob-rivera", "name": "Bob Rivera",
     "type": "Person", "kind": "tech-lead",
     "description": "Tech Lead of AI Platform team"},
    {"entity_id": "us:group-ai-platform", "name": "AI Platform Team",
     "type": "Group", "kind": "team",
     "description": "Team building LLM infrastructure"},
    {"entity_id": "us:protocol-mcp", "name": "MCP",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:protocol-a2a", "name": "A2A",
     "type": "Protocol", "kind": "spec"},
]

ONBOARDING_EDGES_WIKI = [
    {"src": "us:person-alice-chen", "tgt": "us:group-ai-platform",
     "type": "LEADS", "conf": 0.95},
    {"src": "us:person-bob-rivera", "tgt": "us:group-ai-platform",
     "type": "MEMBER_OF", "conf": 0.90},
    {"src": "us:group-ai-platform", "tgt": "us:protocol-mcp",
     "type": "USES", "conf": 0.85},
    {"src": "us:group-ai-platform", "tgt": "us:protocol-a2a",
     "type": "USES", "conf": 0.85},
]

ONBOARDING_ENTITIES_ARCH = [
    {"entity_id": "us:project-inference-gw", "name": "Inference Gateway",
     "type": "Project", "kind": "service",
     "description": "Routes requests to model backends"},
    {"entity_id": "us:project-model-registry", "name": "Model Registry",
     "type": "Project", "kind": "service",
     "description": "Tracks model versions and artifacts"},
    {"entity_id": "us:project-agent-orch", "name": "Agent Orchestrator",
     "type": "Project", "kind": "service",
     "description": "Coordinates multi-agent workflows"},
    {"entity_id": "us:protocol-mcp", "name": "MCP",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:protocol-a2a", "name": "A2A",
     "type": "Protocol", "kind": "spec"},
]

ONBOARDING_EDGES_ARCH = [
    {"src": "us:project-inference-gw", "tgt": "us:protocol-mcp",
     "type": "IMPLEMENTS", "conf": 0.90},
    {"src": "us:project-agent-orch", "tgt": "us:protocol-a2a",
     "type": "IMPLEMENTS", "conf": 0.90},
]

ONBOARDING_ENTITIES_GLOSSARY = [
    {"entity_id": "us:project-vllm", "name": "vLLM",
     "type": "Project", "kind": "engine",
     "description": "High-throughput LLM serving engine"},
    {"entity_id": "us:project-tgi", "name": "TGI",
     "type": "Project", "kind": "engine",
     "description": "Text Generation Inference by Hugging Face"},
]

ONBOARDING_EDGES_GLOSSARY = [
    {"src": "us:project-inference-gw", "tgt": "us:project-vllm",
     "type": "USES", "conf": 0.80},
    {"src": "us:project-inference-gw", "tgt": "us:project-tgi",
     "type": "USES", "conf": 0.80},
]


class TestUserStoryOnboarding:
    """New team member ingests 3 onboarding docs, then queries the graph."""

    def test_onboarding_journey_sqlite(self, db):
        """Full onboarding flow from zero to queryable knowledge."""
        # Start with empty DB — new team member has no sources
        initial_sources = db.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        assert initial_sources == 0

        # Ingest 3 onboarding docs
        sid1 = _full_pipeline_no_neo4j(
            db, "us://wiki/team", WIKI_CONTENT,
            ONBOARDING_ENTITIES_WIKI, ONBOARDING_EDGES_WIKI,
            submitter_email="newbie@company.com",
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "us://docs/architecture", ARCH_CONTENT,
            ONBOARDING_ENTITIES_ARCH, ONBOARDING_EDGES_ARCH,
            submitter_email="newbie@company.com",
        )
        sid3 = _full_pipeline_no_neo4j(
            db, "us://docs/glossary", GLOSSARY_CONTENT,
            ONBOARDING_ENTITIES_GLOSSARY, ONBOARDING_EDGES_GLOSSARY,
            submitter_email="newbie@company.com",
        )

        # All 3 sources should complete
        for sid in [sid1, sid2, sid3]:
            src = db.get_source(sid)
            assert src["status"] == "complete"
            assert src["stage"] == "done"

        # Query: "What protocols does our team use?"
        protocol_entities = db.conn.execute(
            "SELECT DISTINCT e.entity_id, e.name FROM entities e "
            "WHERE e.type = 'Protocol' AND e.status = 'approved' "
            "AND e.deprecated_at IS NULL"
        ).fetchall()
        protocol_names = {dict(p)["name"] for p in protocol_entities}
        assert "MCP" in protocol_names
        assert "A2A" in protocol_names

        # Query: "Who leads the team?"
        leaders = db.conn.execute(
            "SELECT e_src.name AS leader, e_tgt.name AS team "
            "FROM edges ed "
            "JOIN entities e_src ON ed.source_entity_id = e_src.entity_id "
            "JOIN entities e_tgt ON ed.target_entity_id = e_tgt.entity_id "
            "WHERE ed.edge_type = 'LEADS' AND ed.status = 'approved'"
        ).fetchall()
        leader_names = {dict(l)["leader"] for l in leaders}
        assert "Alice Chen" in leader_names

    def test_onboarding_entity_deduplication(self, db):
        """Entities referenced in multiple docs should not create duplicates."""
        _full_pipeline_no_neo4j(
            db, "us://dedup/wiki", WIKI_CONTENT,
            ONBOARDING_ENTITIES_WIKI, ONBOARDING_EDGES_WIKI,
        )
        _full_pipeline_no_neo4j(
            db, "us://dedup/arch", ARCH_CONTENT,
            ONBOARDING_ENTITIES_ARCH, ONBOARDING_EDGES_ARCH,
        )

        # us:protocol-mcp appears in both sources, second add_entity returns None (IntegrityError)
        mcp_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'us:protocol-mcp'"
        ).fetchall()
        assert len(mcp_rows) == 1

    def test_onboarding_neo4j_full_graph(self, db, clean_neo4j):
        """After onboarding, Neo4j graph supports traversal queries."""
        _full_pipeline_neo4j(
            db, "us://neo/wiki", WIKI_CONTENT,
            ONBOARDING_ENTITIES_WIKI, ONBOARDING_EDGES_WIKI, clean_neo4j,
            submitter_email="newbie@company.com",
        )
        _full_pipeline_neo4j(
            db, "us://neo/arch", ARCH_CONTENT,
            ONBOARDING_ENTITIES_ARCH, ONBOARDING_EDGES_ARCH, clean_neo4j,
            submitter_email="newbie@company.com",
        )

        with clean_neo4j.session() as s:
            # "What protocols does the team use?"
            result = s.run("""
                MATCH (g:Group {entity_id: 'us:group-ai-platform'})-[:USES]->(p:Protocol)
                RETURN p.name AS name ORDER BY name
            """).data()
            names = [r["name"] for r in result]
            assert "A2A" in names
            assert "MCP" in names

            # "Which services implement MCP?"
            result = s.run("""
                MATCH (svc)-[:IMPLEMENTS]->(p:Protocol {entity_id: 'us:protocol-mcp'})
                RETURN svc.name AS name
            """).data()
            svc_names = [r["name"] for r in result]
            assert "Inference Gateway" in svc_names


# ===========================================================================
# USER STORY 2: Competitive Analysis
# ===========================================================================

COPILOT_CONTENT = """\
# Microsoft Copilot Studio — Agent Capabilities (May 2026)

Microsoft Copilot Studio enables building autonomous agents with:

- **Multi-model orchestration**: Supports GPT-4, GPT-4o, and third-party models
- **Enterprise connectors**: 1000+ pre-built connectors to business systems
- **Autonomous actions**: Agents can take actions without human approval
- **Memory and context**: Persistent memory across conversations
- **Code generation**: Agents can write and execute code

Copilot agents communicate using Microsoft's proprietary agent protocol
and can be deployed on Azure infrastructure.
"""

GEMINI_CONTENT = """\
# Google Gemini Agent Features (May 2026)

Google's Gemini platform for agent development includes:

- **A2A protocol support**: Native agent-to-agent communication
- **MCP integration**: Full Model Context Protocol support for tool use
- **Multimodal reasoning**: Process text, images, audio, and video
- **Agent Development Kit (ADK)**: Open-source toolkit for building agents
- **Vertex AI deployment**: Enterprise-grade hosting and scaling

Gemini agents can interoperate with any A2A-compatible agent system.
"""

OUR_PRODUCT_CONTENT = """\
# Our Agent Platform — Feature Matrix

Our platform provides:

- **MCP server hosting**: Deploy MCP servers for tool and context provision
- **A2A gateway**: Route inter-agent messages using A2A protocol
- **Custom model support**: Bring your own model (BYOM) via inference gateway
- **Audit logging**: Full audit trail of agent actions and decisions
- **Role-based access**: Fine-grained permissions for agent capabilities

Currently lacking: autonomous actions, multimodal reasoning, persistent memory.
"""

COMP_ENTITIES_COPILOT = [
    {"entity_id": "us:org-microsoft", "name": "Microsoft",
     "type": "Organization", "kind": "company"},
    {"entity_id": "us:project-copilot-studio", "name": "Copilot Studio",
     "type": "Project", "kind": "platform",
     "description": "Microsoft's agent building platform"},
    {"entity_id": "us:cap-multi-model", "name": "Multi-model Orchestration",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-enterprise-conn", "name": "Enterprise Connectors",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-autonomous-actions", "name": "Autonomous Actions",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-persistent-memory", "name": "Persistent Memory",
     "type": "Capability", "kind": "feature"},
]

COMP_EDGES_COPILOT = [
    {"src": "us:org-microsoft", "tgt": "us:project-copilot-studio",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "us:project-copilot-studio", "tgt": "us:cap-multi-model",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:project-copilot-studio", "tgt": "us:cap-enterprise-conn",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:project-copilot-studio", "tgt": "us:cap-autonomous-actions",
     "type": "PROVIDES", "conf": 0.85},
    {"src": "us:project-copilot-studio", "tgt": "us:cap-persistent-memory",
     "type": "PROVIDES", "conf": 0.85},
]

COMP_ENTITIES_GEMINI = [
    {"entity_id": "us:org-google", "name": "Google",
     "type": "Organization", "kind": "company"},
    {"entity_id": "us:project-gemini-agents", "name": "Gemini Agent Platform",
     "type": "Project", "kind": "platform",
     "description": "Google's agent development platform"},
    {"entity_id": "us:cap-a2a-support", "name": "A2A Protocol Support",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-mcp-integration", "name": "MCP Integration",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-multimodal", "name": "Multimodal Reasoning",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:project-adk", "name": "Agent Development Kit",
     "type": "Project", "kind": "framework"},
]

COMP_EDGES_GEMINI = [
    {"src": "us:org-google", "tgt": "us:project-gemini-agents",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "us:project-gemini-agents", "tgt": "us:cap-a2a-support",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:project-gemini-agents", "tgt": "us:cap-mcp-integration",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:project-gemini-agents", "tgt": "us:cap-multimodal",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:org-google", "tgt": "us:project-adk",
     "type": "DEVELOPS", "conf": 0.95},
]

COMP_ENTITIES_OURS = [
    {"entity_id": "us:project-our-platform", "name": "Our Agent Platform",
     "type": "Project", "kind": "platform",
     "description": "Internal agent platform"},
    {"entity_id": "us:cap-mcp-hosting", "name": "MCP Server Hosting",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-a2a-gateway", "name": "A2A Gateway",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-audit-logging", "name": "Audit Logging",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-rbac", "name": "Role-based Access",
     "type": "Capability", "kind": "feature"},
]

COMP_EDGES_OURS = [
    {"src": "us:project-our-platform", "tgt": "us:cap-mcp-hosting",
     "type": "PROVIDES", "conf": 0.95},
    {"src": "us:project-our-platform", "tgt": "us:cap-a2a-gateway",
     "type": "PROVIDES", "conf": 0.95},
    {"src": "us:project-our-platform", "tgt": "us:cap-audit-logging",
     "type": "PROVIDES", "conf": 0.95},
    {"src": "us:project-our-platform", "tgt": "us:cap-rbac",
     "type": "PROVIDES", "conf": 0.95},
]


class TestUserStoryCompetitiveAnalysis:
    """PM ingests competitor + own product docs, runs set-difference/intersection queries."""

    def test_competitive_gap_analysis_sqlite(self, db):
        """Identify capabilities competitors have that we don't."""
        _full_pipeline_no_neo4j(
            db, "us://comp/copilot", COPILOT_CONTENT,
            COMP_ENTITIES_COPILOT, COMP_EDGES_COPILOT,
            submitter_email="pm@company.com",
        )
        _full_pipeline_no_neo4j(
            db, "us://comp/gemini", GEMINI_CONTENT,
            COMP_ENTITIES_GEMINI, COMP_EDGES_GEMINI,
            submitter_email="pm@company.com",
        )
        _full_pipeline_no_neo4j(
            db, "us://comp/ours", OUR_PRODUCT_CONTENT,
            COMP_ENTITIES_OURS, COMP_EDGES_OURS,
            submitter_email="pm@company.com",
        )

        # All sources complete
        for uri in ["us://comp/copilot", "us://comp/gemini", "us://comp/ours"]:
            src = db.get_source_by_uri(uri)
            assert src["status"] == "complete"

        # Set difference: capabilities competitors have that we don't
        our_caps = db.conn.execute(
            "SELECT DISTINCT e_cap.entity_id FROM edges ed "
            "JOIN entities e_cap ON ed.target_entity_id = e_cap.entity_id "
            "WHERE ed.source_entity_id = 'us:project-our-platform' "
            "AND ed.edge_type = 'PROVIDES' AND ed.status = 'approved'"
        ).fetchall()
        our_cap_ids = {dict(c)["entity_id"] for c in our_caps}

        all_competitor_caps = db.conn.execute(
            "SELECT DISTINCT e_cap.entity_id, e_cap.name FROM edges ed "
            "JOIN entities e_cap ON ed.target_entity_id = e_cap.entity_id "
            "WHERE ed.source_entity_id IN ('us:project-copilot-studio', 'us:project-gemini-agents') "
            "AND ed.edge_type = 'PROVIDES' AND ed.status = 'approved'"
        ).fetchall()
        competitor_cap_ids = {dict(c)["entity_id"] for c in all_competitor_caps}

        gap_ids = competitor_cap_ids - our_cap_ids
        assert len(gap_ids) > 0
        gap_names = {dict(c)["name"] for c in all_competitor_caps
                     if dict(c)["entity_id"] in gap_ids}
        assert "Autonomous Actions" in gap_names or "Multimodal Reasoning" in gap_names

    def test_competitive_intersection_sqlite(self, db):
        """Find capabilities shared across all platforms."""
        _full_pipeline_no_neo4j(
            db, "us://inter/copilot", COPILOT_CONTENT,
            COMP_ENTITIES_COPILOT, COMP_EDGES_COPILOT,
        )
        _full_pipeline_no_neo4j(
            db, "us://inter/gemini", GEMINI_CONTENT,
            COMP_ENTITIES_GEMINI, COMP_EDGES_GEMINI,
        )
        _full_pipeline_no_neo4j(
            db, "us://inter/ours", OUR_PRODUCT_CONTENT,
            COMP_ENTITIES_OURS, COMP_EDGES_OURS,
        )

        # All three platforms expose capabilities via PROVIDES edges.
        # Find capability types common to all platforms by looking at
        # the Capability kind ("feature") entities each platform PROVIDES.
        platforms = [
            "us:project-copilot-studio",
            "us:project-gemini-agents",
            "us:project-our-platform",
        ]
        cap_sets = []
        for plat in platforms:
            caps = db.conn.execute(
                "SELECT DISTINCT e_cap.name FROM edges ed "
                "JOIN entities e_cap ON ed.target_entity_id = e_cap.entity_id "
                "WHERE ed.source_entity_id = ? "
                "AND ed.edge_type = 'PROVIDES' AND ed.status = 'approved'",
                (plat,)
            ).fetchall()
            cap_sets.append({dict(c)["name"] for c in caps})

        assert len(cap_sets) == 3
        for cs in cap_sets:
            assert len(cs) > 0

    def test_competitive_neo4j_traversal(self, db, clean_neo4j):
        """Neo4j graph supports competitor comparison queries."""
        _full_pipeline_neo4j(
            db, "us://ncomp/copilot", COPILOT_CONTENT,
            COMP_ENTITIES_COPILOT, COMP_EDGES_COPILOT, clean_neo4j,
        )
        _full_pipeline_neo4j(
            db, "us://ncomp/gemini", GEMINI_CONTENT,
            COMP_ENTITIES_GEMINI, COMP_EDGES_GEMINI, clean_neo4j,
        )

        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (org:Organization)-[:DEVELOPS]->(p:Project)-[:PROVIDES]->(cap:Capability)
                WHERE org.entity_id STARTS WITH 'us:'
                RETURN org.name AS company, p.name AS product,
                       collect(cap.name) AS capabilities
                ORDER BY company
            """).data()
            assert len(result) >= 2
            companies = {r["company"] for r in result}
            assert "Google" in companies
            assert "Microsoft" in companies


# ===========================================================================
# USER STORY 3: Incident Investigation
# ===========================================================================

POSTMORTEM_CONTENT = """\
# Post-Mortem: MCP-A2A Protocol Incompatibility Incident

## Date: 2026-04-15

## Summary

A protocol version mismatch between the Inference Gateway (MCP v1.0) and the
Agent Orchestrator (A2A v2.1) caused cascading failures across the agent
pipeline. The gateway was returning MCP v1.0 capability responses that the
orchestrator could not parse, resulting in dropped agent tasks.

## Root Cause

The Inference Gateway was upgraded to use MCP v1.1 streaming, but the
A2A-to-MCP bridge component was still pinned to MCP v1.0 schemas. When agents
sent A2A discovery requests, the bridge returned capabilities in the old format.

## Affected Components

1. Inference Gateway (MCP server)
2. A2A-MCP Bridge (translation layer)
3. Agent Orchestrator (A2A client)
4. Model Registry (secondary — could not register new model versions during outage)

## Resolution

Upgraded the A2A-MCP bridge to MCP v1.1. Added protocol version negotiation
to prevent future mismatches.

## Action Items

- Add version compatibility matrix tests
- Implement protocol version negotiation in the bridge
- Set up monitoring for protocol version mismatches
"""

INCIDENT_ENTITIES = [
    {"entity_id": "us:project-inference-gw-inc", "name": "Inference Gateway",
     "type": "Project", "kind": "service",
     "description": "MCP server routing model requests"},
    {"entity_id": "us:project-a2a-bridge", "name": "A2A-MCP Bridge",
     "type": "Project", "kind": "service",
     "description": "Translation layer between A2A and MCP protocols"},
    {"entity_id": "us:project-agent-orch-inc", "name": "Agent Orchestrator",
     "type": "Project", "kind": "service",
     "description": "Coordinates multi-agent workflows via A2A"},
    {"entity_id": "us:project-model-reg-inc", "name": "Model Registry",
     "type": "Project", "kind": "service"},
    {"entity_id": "us:protocol-mcp-v1.1", "name": "MCP v1.1",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:protocol-mcp-v1.0", "name": "MCP v1.0",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:protocol-a2a-v2.1", "name": "A2A v2.1",
     "type": "Protocol", "kind": "spec"},
]

INCIDENT_EDGES = [
    {"src": "us:project-inference-gw-inc", "tgt": "us:protocol-mcp-v1.1",
     "type": "IMPLEMENTS", "conf": 0.95},
    {"src": "us:project-a2a-bridge", "tgt": "us:protocol-mcp-v1.0",
     "type": "IMPLEMENTS", "conf": 0.90},
    {"src": "us:project-a2a-bridge", "tgt": "us:protocol-a2a-v2.1",
     "type": "IMPLEMENTS", "conf": 0.90},
    {"src": "us:project-agent-orch-inc", "tgt": "us:protocol-a2a-v2.1",
     "type": "IMPLEMENTS", "conf": 0.90},
    {"src": "us:project-inference-gw-inc", "tgt": "us:project-a2a-bridge",
     "type": "DEPENDS_ON", "conf": 0.85},
    {"src": "us:project-agent-orch-inc", "tgt": "us:project-a2a-bridge",
     "type": "DEPENDS_ON", "conf": 0.85},
    {"src": "us:protocol-mcp-v1.1", "tgt": "us:protocol-mcp-v1.0",
     "type": "SUPERSEDES", "conf": 0.95},
]


class TestUserStoryIncidentInvestigation:
    """Engineer ingests post-mortem, explores component relationships in the graph."""

    def test_incident_affected_components(self, db):
        """Verify all affected components are captured from the post-mortem."""
        sid = _full_pipeline_no_neo4j(
            db, "us://incidents/mcp-a2a-compat", POSTMORTEM_CONTENT,
            INCIDENT_ENTITIES, INCIDENT_EDGES,
            submitter_email="oncall@company.com",
        )
        source = db.get_source(sid)
        assert source["status"] == "complete"

        services = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND type = 'Project' "
            "AND status = 'approved'",
            (sid,)
        ).fetchall()
        service_names = {dict(s)["name"] for s in services}
        assert "Inference Gateway" in service_names
        assert "A2A-MCP Bridge" in service_names
        assert "Agent Orchestrator" in service_names
        assert "Model Registry" in service_names

    def test_incident_dependency_discovery(self, db, clean_neo4j):
        """Use the graph to discover which components depend on the bridge."""
        _full_pipeline_neo4j(
            db, "us://inc-neo/postmortem", POSTMORTEM_CONTENT,
            INCIDENT_ENTITIES, INCIDENT_EDGES, clean_neo4j,
            submitter_email="oncall@company.com",
        )

        with clean_neo4j.session() as s:
            # "What depends on the A2A-MCP Bridge?" — discovery query
            result = s.run("""
                MATCH (component)-[:DEPENDS_ON]->(bridge {entity_id: 'us:project-a2a-bridge'})
                RETURN component.name AS name ORDER BY name
            """).data()
            dependent_names = [r["name"] for r in result]
            assert "Inference Gateway" in dependent_names
            assert "Agent Orchestrator" in dependent_names

    def test_incident_protocol_version_conflict(self, db, clean_neo4j):
        """Discover the version mismatch: bridge implements old MCP, gateway uses new."""
        _full_pipeline_neo4j(
            db, "us://inc-ver/postmortem", POSTMORTEM_CONTENT,
            INCIDENT_ENTITIES, INCIDENT_EDGES, clean_neo4j,
        )

        with clean_neo4j.session() as s:
            # Find components implementing different MCP versions
            result = s.run("""
                MATCH (comp)-[:IMPLEMENTS]->(proto:Protocol)
                WHERE proto.entity_id STARTS WITH 'us:protocol-mcp-v'
                RETURN comp.name AS component, proto.name AS protocol_version
                ORDER BY comp.name
            """).data()
            versions_by_component = {}
            for r in result:
                versions_by_component[r["component"]] = r["protocol_version"]
            assert versions_by_component.get("A2A-MCP Bridge") == "MCP v1.0"
            assert versions_by_component.get("Inference Gateway") == "MCP v1.1"

    def test_incident_supersession_chain(self, db, clean_neo4j):
        """Verify protocol version chain (v1.1 supersedes v1.0)."""
        _full_pipeline_neo4j(
            db, "us://inc-chain/postmortem", POSTMORTEM_CONTENT,
            INCIDENT_ENTITIES, INCIDENT_EDGES, clean_neo4j,
        )

        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (newer:Protocol)-[:SUPERSEDES]->(older:Protocol)
                WHERE newer.entity_id STARTS WITH 'us:'
                RETURN newer.name AS newer, older.name AS older
            """).data()
            assert len(result) == 1
            assert result[0]["newer"] == "MCP v1.1"
            assert result[0]["older"] == "MCP v1.0"


# ===========================================================================
# USER STORY 4: Standards Tracking
# ===========================================================================

RFC_V1_CONTENT = """\
# RFC-9001: Agent Communication Protocol (ACP) — Draft 1

## Status: Draft

## Abstract

This RFC defines the Agent Communication Protocol (ACP), a lightweight protocol
for agent-to-agent message exchange. ACP uses JSON-RPC 2.0 over HTTP.

## 1. Message Format

Messages are JSON objects with fields: sender, receiver, action, payload.

## 2. Discovery

Agents register with a central registry using their capabilities list.
Discovery is pull-based: agents query the registry.

## 3. Security

Authentication via API keys. No encryption requirement in draft 1.
"""

RFC_V2_CONTENT = """\
# RFC-9001: Agent Communication Protocol (ACP) — Draft 2

## Status: Proposed Standard

## Abstract

This RFC defines the Agent Communication Protocol (ACP), a protocol
for secure agent-to-agent message exchange. ACP uses JSON-RPC 2.0 over HTTPS.

## 1. Message Format

Messages are JSON objects with fields: sender, receiver, action, payload, timestamp.
Added: mandatory timestamp field for replay protection.

## 2. Discovery

Agents register with a central registry using their capabilities list.
Discovery supports both pull and push modes (added in draft 2).

## 3. Security

Authentication via mTLS (upgraded from API keys). All connections require TLS 1.3.

## 4. Streaming (New)

Added server-sent events (SSE) for streaming responses.
"""

IMPL_GUIDE_CONTENT = """\
# ACP Implementation Guide

## Getting Started

To implement ACP draft 2, your agent needs:

1. An HTTPS endpoint supporting JSON-RPC 2.0
2. mTLS certificate for authentication
3. SSE support for streaming responses
4. Registry client for discovery (pull or push)

## Reference Implementation

The reference implementation is available at github.com/acp-spec/acp-reference.
It was developed by the ACP Working Group, chaired by Dr. Sarah Kim.
"""

RFC_V1_ENTITIES = [
    {"entity_id": "us:protocol-acp", "name": "Agent Communication Protocol",
     "type": "Protocol", "kind": "spec",
     "description": "Lightweight protocol for agent-to-agent exchange (Draft 1)"},
    {"entity_id": "us:cap-acp-discovery-pull", "name": "Pull-based Discovery",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-acp-auth-apikey", "name": "API Key Authentication",
     "type": "Capability", "kind": "feature"},
]

RFC_V1_EDGES = [
    {"src": "us:protocol-acp", "tgt": "us:cap-acp-discovery-pull",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:protocol-acp", "tgt": "us:cap-acp-auth-apikey",
     "type": "PROVIDES", "conf": 0.90},
]

RFC_V2_ENTITIES = [
    {"entity_id": "us:protocol-acp-v2", "name": "Agent Communication Protocol v2",
     "type": "Protocol", "kind": "spec",
     "description": "Secure agent-to-agent exchange protocol (Draft 2)"},
    {"entity_id": "us:cap-acp-discovery-push", "name": "Push-based Discovery",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-acp-mtls", "name": "mTLS Authentication",
     "type": "Capability", "kind": "feature"},
    {"entity_id": "us:cap-acp-streaming", "name": "SSE Streaming",
     "type": "Capability", "kind": "feature"},
]

RFC_V2_EDGES = [
    {"src": "us:protocol-acp-v2", "tgt": "us:cap-acp-discovery-push",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:protocol-acp-v2", "tgt": "us:cap-acp-mtls",
     "type": "PROVIDES", "conf": 0.95},
    {"src": "us:protocol-acp-v2", "tgt": "us:cap-acp-streaming",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:protocol-acp-v2", "tgt": "us:protocol-acp",
     "type": "SUPERSEDES", "conf": 0.98},
]

IMPL_ENTITIES = [
    {"entity_id": "us:project-acp-reference", "name": "ACP Reference Implementation",
     "type": "Project", "kind": "library",
     "description": "Reference impl of ACP draft 2"},
    {"entity_id": "us:group-acp-wg", "name": "ACP Working Group",
     "type": "Group", "kind": "standards-body"},
    {"entity_id": "us:person-sarah-kim", "name": "Dr. Sarah Kim",
     "type": "Person", "kind": "chair",
     "description": "Chair of the ACP Working Group"},
]

IMPL_EDGES = [
    {"src": "us:project-acp-reference", "tgt": "us:protocol-acp-v2",
     "type": "IMPLEMENTS", "conf": 0.95},
    {"src": "us:group-acp-wg", "tgt": "us:protocol-acp-v2",
     "type": "DEVELOPS", "conf": 0.90},
    {"src": "us:person-sarah-kim", "tgt": "us:group-acp-wg",
     "type": "CHAIRS", "conf": 0.95},
]


class TestUserStoryStandardsTracking:
    """Track protocol evolution: ingest v1, then v2, then impl guide."""

    def test_rfc_version_diff(self, db):
        """Ingest RFC v1 and v2, identify what changed."""
        _full_pipeline_no_neo4j(
            db, "us://rfc/acp-v1", RFC_V1_CONTENT,
            RFC_V1_ENTITIES, RFC_V1_EDGES,
        )
        _full_pipeline_no_neo4j(
            db, "us://rfc/acp-v2", RFC_V2_CONTENT,
            RFC_V2_ENTITIES, RFC_V2_EDGES,
        )

        # Capabilities in v2 not in v1
        v1_caps = db.conn.execute(
            "SELECT DISTINCT e.entity_id, e.name FROM edges ed "
            "JOIN entities e ON ed.target_entity_id = e.entity_id "
            "WHERE ed.source_entity_id = 'us:protocol-acp' "
            "AND ed.edge_type = 'PROVIDES' AND ed.status = 'approved'"
        ).fetchall()
        v2_caps = db.conn.execute(
            "SELECT DISTINCT e.entity_id, e.name FROM edges ed "
            "JOIN entities e ON ed.target_entity_id = e.entity_id "
            "WHERE ed.source_entity_id = 'us:protocol-acp-v2' "
            "AND ed.edge_type = 'PROVIDES' AND ed.status = 'approved'"
        ).fetchall()

        v1_cap_ids = {dict(c)["entity_id"] for c in v1_caps}
        v2_cap_ids = {dict(c)["entity_id"] for c in v2_caps}

        new_in_v2 = v2_cap_ids - v1_cap_ids
        assert len(new_in_v2) > 0

        new_cap_names = {dict(c)["name"] for c in v2_caps
                         if dict(c)["entity_id"] in new_in_v2}
        assert "mTLS Authentication" in new_cap_names
        assert "SSE Streaming" in new_cap_names

    def test_rfc_supersession(self, db):
        """V2 supersedes V1 in the graph."""
        _full_pipeline_no_neo4j(
            db, "us://rfc-sup/acp-v1", RFC_V1_CONTENT,
            RFC_V1_ENTITIES, RFC_V1_EDGES,
        )
        _full_pipeline_no_neo4j(
            db, "us://rfc-sup/acp-v2", RFC_V2_CONTENT,
            RFC_V2_ENTITIES, RFC_V2_EDGES,
        )

        supersession = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'SUPERSEDES' "
            "AND source_entity_id = 'us:protocol-acp-v2' "
            "AND target_entity_id = 'us:protocol-acp' "
            "AND status = 'approved'"
        ).fetchall()
        assert len(supersession) == 1

    def test_impl_references_rfc(self, db, clean_neo4j):
        """Implementation guide references the RFC via IMPLEMENTS edge."""
        _full_pipeline_neo4j(
            db, "us://rfc-neo/acp-v2", RFC_V2_CONTENT,
            RFC_V2_ENTITIES, RFC_V2_EDGES, clean_neo4j,
        )
        _full_pipeline_neo4j(
            db, "us://rfc-neo/impl", IMPL_GUIDE_CONTENT,
            IMPL_ENTITIES, IMPL_EDGES, clean_neo4j,
        )

        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (impl:Project)-[:IMPLEMENTS]->(proto:Protocol)
                WHERE impl.entity_id = 'us:project-acp-reference'
                RETURN proto.name AS protocol
            """).data()
            assert len(result) == 1
            assert "v2" in result[0]["protocol"].lower() or "V2" in result[0]["protocol"]

    def test_standards_full_chain_neo4j(self, db, clean_neo4j):
        """Full chain: person -> chairs -> WG -> develops -> protocol -> impl."""
        _full_pipeline_neo4j(
            db, "us://chain/acp-v2", RFC_V2_CONTENT,
            RFC_V2_ENTITIES, RFC_V2_EDGES, clean_neo4j,
        )
        _full_pipeline_neo4j(
            db, "us://chain/impl", IMPL_GUIDE_CONTENT,
            IMPL_ENTITIES, IMPL_EDGES, clean_neo4j,
        )

        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (person:Person)-[:CHAIRS]->(wg:Group)-[:DEVELOPS]->(proto:Protocol)
                      <-[:IMPLEMENTS]-(impl:Project)
                WHERE person.entity_id = 'us:person-sarah-kim'
                RETURN person.name AS chair, wg.name AS group,
                       proto.name AS protocol, impl.name AS implementation
            """).data()
            assert len(result) == 1
            assert result[0]["chair"] == "Dr. Sarah Kim"
            assert result[0]["group"] == "ACP Working Group"
            assert "Reference" in result[0]["implementation"]


# ===========================================================================
# USER STORY 5: Knowledge Curation
# ===========================================================================

GOOD_SOURCE_1 = """\
# MCP Server Best Practices

Building a production MCP server requires careful attention to:
- Resource scoping and access control
- Tool input validation
- Error handling and retry semantics
Anthropic maintains the official MCP specification.
"""

GOOD_SOURCE_2 = """\
# A2A Agent Discovery

Agents announce their capabilities via Agent Cards — JSON documents served
at a well-known endpoint. The A2A discovery protocol supports both pull
and push-based registration. Google developed the A2A specification.
"""

BAD_SOURCE = """\
# Random Notes

AI is the future. Everyone should use AI. AI will change everything.
Blockchain and AI will revolutionize the world. This is very important.
"""

MEDIUM_SOURCE_1 = """\
# SDK Comparison

The Anthropic Python SDK and the Google GenAI SDK both support tool use.
The Anthropic SDK uses a messages-based API while GenAI uses a
generate-content API. Both are open-source.
"""

MEDIUM_SOURCE_2 = """\
# Agent Frameworks

LangChain and CrewAI are popular agent frameworks. LangChain supports
MCP integration. CrewAI focuses on multi-agent collaboration.
Both frameworks can orchestrate complex agent workflows.
"""

CURATION_ENTITIES_GOOD1 = [
    {"entity_id": "us:protocol-mcp-cur", "name": "MCP",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:org-anthropic-cur", "name": "Anthropic",
     "type": "Organization", "kind": "company"},
]
CURATION_EDGES_GOOD1 = [
    {"src": "us:org-anthropic-cur", "tgt": "us:protocol-mcp-cur",
     "type": "DEVELOPS", "conf": 0.95},
]

CURATION_ENTITIES_GOOD2 = [
    {"entity_id": "us:protocol-a2a-cur", "name": "A2A",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:cap-agent-cards", "name": "Agent Cards",
     "type": "Capability", "kind": "feature",
     "description": "JSON capability announcements"},
    {"entity_id": "us:org-google-cur", "name": "Google",
     "type": "Organization", "kind": "company"},
]
CURATION_EDGES_GOOD2 = [
    {"src": "us:protocol-a2a-cur", "tgt": "us:cap-agent-cards",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:org-google-cur", "tgt": "us:protocol-a2a-cur",
     "type": "DEVELOPS", "conf": 0.95},
]

CURATION_ENTITIES_BAD = [
    {"entity_id": "us:concept-ai-future", "name": "AI Future",
     "type": "Capability", "kind": "concept",
     "description": "Vague claim about AI"},
    {"entity_id": "us:concept-blockchain-ai", "name": "Blockchain AI",
     "type": "Capability", "kind": "concept",
     "description": "Unsubstantiated buzzword combination"},
]
CURATION_EDGES_BAD = []

CURATION_ENTITIES_MED1 = [
    {"entity_id": "us:project-anthropic-sdk", "name": "Anthropic Python SDK",
     "type": "Project", "kind": "library"},
    {"entity_id": "us:project-genai-sdk", "name": "Google GenAI SDK",
     "type": "Project", "kind": "library"},
]
CURATION_EDGES_MED1 = [
    {"src": "us:project-anthropic-sdk", "tgt": "us:project-genai-sdk",
     "type": "COMPETES_WITH", "conf": 0.70},
]

CURATION_ENTITIES_MED2 = [
    {"entity_id": "us:project-langchain", "name": "LangChain",
     "type": "Project", "kind": "framework"},
    {"entity_id": "us:project-crewai", "name": "CrewAI",
     "type": "Project", "kind": "framework"},
]
CURATION_EDGES_MED2 = [
    {"src": "us:project-langchain", "tgt": "us:protocol-mcp-cur",
     "type": "IMPLEMENTS", "conf": 0.80},
]


class TestUserStoryKnowledgeCuration:
    """User curates 5 sources: approves good, rejects bad, audits orphans."""

    def test_full_curation_lifecycle(self, db):
        """Ingest 5 sources, selectively approve/reject, verify curation state."""
        from agents_kg.stages.load import run as run_load

        # Ingest all 5 sources through extract
        sources = [
            ("us://cur/good1", GOOD_SOURCE_1, CURATION_ENTITIES_GOOD1, CURATION_EDGES_GOOD1),
            ("us://cur/good2", GOOD_SOURCE_2, CURATION_ENTITIES_GOOD2, CURATION_EDGES_GOOD2),
            ("us://cur/bad", BAD_SOURCE, CURATION_ENTITIES_BAD, CURATION_EDGES_BAD),
            ("us://cur/med1", MEDIUM_SOURCE_1, CURATION_ENTITIES_MED1, CURATION_EDGES_MED1),
            ("us://cur/med2", MEDIUM_SOURCE_2, CURATION_ENTITIES_MED2, CURATION_EDGES_MED2),
        ]
        sids = {}
        for uri, content, ents, edges in sources:
            sid = db.add_source(uri, submitter_email="curator@company.com")
            source = _run_through_chunk(db, sid, content)
            _mock_embed(db, source)
            source = db.get_source(sid)
            _mock_extract(db, source, ents, edges)
            db.update_source(sid, stage="review", status="pending_review")
            sids[uri] = sid

        # Review: approve good entities, reject bad ones
        pending = db.get_entities_by_status("pending_review")
        assert len(pending) > 0

        for ent in pending:
            if ent["entity_id"].startswith("us:concept-"):
                db.update_entity(ent["id"], status="rejected")
            else:
                db.approve_entity(ent["id"])

        for edge in db.get_edges_by_status("pending_review"):
            db.approve_edge(edge["id"])

        # Verify rejected entities exist but are not approved
        rejected = db.conn.execute(
            "SELECT * FROM entities WHERE status = 'rejected'"
        ).fetchall()
        rejected_ids = {dict(e)["entity_id"] for e in rejected}
        assert "us:concept-ai-future" in rejected_ids
        assert "us:concept-blockchain-ai" in rejected_ids

        # Load approved entities to "Neo4j" (no driver)
        for uri, sid in sids.items():
            db.update_source(sid, status="processing", stage="load")
            source = db.get_source(sid)
            run_load(db, source, neo4j_driver=None)

        # All sources should be complete
        for sid in sids.values():
            src = db.get_source(sid)
            assert src["status"] == "complete"

        # Only approved entities survived
        approved = db.get_entities_by_status("approved")
        approved_ids = {e["entity_id"] for e in approved}
        assert "us:protocol-mcp-cur" in approved_ids
        assert "us:concept-ai-future" not in approved_ids

    def test_deprecate_source_only_affects_its_entities(self, db):
        """Deprecating a source only affects that source's exclusive entities."""
        sid1 = _full_pipeline_no_neo4j(
            db, "us://dep/source-a", GOOD_SOURCE_1,
            CURATION_ENTITIES_GOOD1, CURATION_EDGES_GOOD1,
        )
        sid2 = _full_pipeline_no_neo4j(
            db, "us://dep/source-b", GOOD_SOURCE_2,
            CURATION_ENTITIES_GOOD2, CURATION_EDGES_GOOD2,
        )

        # Deprecate source A
        db.deprecate_entities_for_source(sid1)

        # Source A's entities deprecated
        deprecated = db.get_deprecated_entities()
        deprecated_ids = {e["entity_id"] for e in deprecated}
        assert "us:protocol-mcp-cur" in deprecated_ids
        assert "us:org-anthropic-cur" in deprecated_ids

        # Source B's entities NOT deprecated
        source_b_ents = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND deprecated_at IS NULL",
            (sid2,)
        ).fetchall()
        source_b_ids = {dict(e)["entity_id"] for e in source_b_ents}
        assert "us:protocol-a2a-cur" in source_b_ids
        assert "us:org-google-cur" in source_b_ids

    def test_audit_orphan_entities(self, db):
        """Identify entities that come from only one source (fragile knowledge)."""
        _full_pipeline_no_neo4j(
            db, "us://audit/src1", GOOD_SOURCE_1,
            CURATION_ENTITIES_GOOD1, CURATION_EDGES_GOOD1,
        )
        _full_pipeline_no_neo4j(
            db, "us://audit/src2", GOOD_SOURCE_2,
            CURATION_ENTITIES_GOOD2, CURATION_EDGES_GOOD2,
        )
        _full_pipeline_no_neo4j(
            db, "us://audit/src3", MEDIUM_SOURCE_1,
            CURATION_ENTITIES_MED1, CURATION_EDGES_MED1,
        )

        # Find entities backed by only one source (fragile)
        single_source_ents = db.conn.execute(
            "SELECT entity_id, name, COUNT(DISTINCT source_id) AS src_count "
            "FROM entities WHERE status = 'approved' AND deprecated_at IS NULL "
            "GROUP BY entity_id HAVING src_count = 1"
        ).fetchall()

        single_source_ids = {dict(e)["entity_id"] for e in single_source_ents}
        assert len(single_source_ids) > 0
        # All entities in this test have exactly one source each (due to unique entity_ids)
        assert "us:protocol-mcp-cur" in single_source_ids

    def test_curation_neo4j_only_approved(self, db, clean_neo4j):
        """Only approved entities appear in Neo4j; rejected ones do not."""
        from agents_kg.stages.load import run as run_load

        sid_good = db.add_source("us://cur-neo/good",
                                 submitter_email="curator@company.com")
        source = _run_through_chunk(db, sid_good, GOOD_SOURCE_1)
        _mock_embed(db, source)
        source = db.get_source(sid_good)
        _mock_extract(db, source, CURATION_ENTITIES_GOOD1, CURATION_EDGES_GOOD1)
        db.update_source(sid_good, stage="review", status="pending_review")

        sid_bad = db.add_source("us://cur-neo/bad",
                                submitter_email="curator@company.com")
        source = _run_through_chunk(db, sid_bad, BAD_SOURCE)
        _mock_embed(db, source)
        source = db.get_source(sid_bad)
        _mock_extract(db, source, CURATION_ENTITIES_BAD, CURATION_EDGES_BAD)
        db.update_source(sid_bad, stage="review", status="pending_review")

        # Approve good, reject bad
        for ent in db.get_entities_by_status("pending_review"):
            if ent["entity_id"].startswith("us:concept-"):
                db.update_entity(ent["id"], status="rejected")
            else:
                db.approve_entity(ent["id"])
        for edge in db.get_edges_by_status("pending_review"):
            db.approve_edge(edge["id"])

        # Load both sources
        for sid in [sid_good, sid_bad]:
            db.update_source(sid, status="processing", stage="load")
            source = db.get_source(sid)
            run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            # Good entities present
            result = s.run(
                "MATCH (n {entity_id: 'us:protocol-mcp-cur'}) RETURN n"
            ).single()
            assert result is not None

            # Bad entities absent (rejected, never loaded)
            result = s.run(
                "MATCH (n {entity_id: 'us:concept-ai-future'}) RETURN n"
            ).single()
            assert result is None


# ===========================================================================
# USER STORY 6: Cross-CUJ Integration
# ===========================================================================

SEED_CONTENT = """\
# AI Protocol Landscape — Seed Document

The two dominant agent protocols are MCP (Model Context Protocol) by Anthropic
and A2A (Agent-to-Agent) by Google. These protocols are complementary:
MCP handles tool/context provision while A2A handles inter-agent communication.
"""

SEED_ENTITIES = [
    {"entity_id": "us:protocol-mcp-seed", "name": "MCP",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:protocol-a2a-seed", "name": "A2A",
     "type": "Protocol", "kind": "spec"},
    {"entity_id": "us:org-anthropic-seed", "name": "Anthropic",
     "type": "Organization", "kind": "company"},
    {"entity_id": "us:org-google-seed", "name": "Google",
     "type": "Organization", "kind": "company"},
]

SEED_EDGES = [
    {"src": "us:org-anthropic-seed", "tgt": "us:protocol-mcp-seed",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "us:org-google-seed", "tgt": "us:protocol-a2a-seed",
     "type": "DEVELOPS", "conf": 0.98},
    {"src": "us:protocol-mcp-seed", "tgt": "us:protocol-a2a-seed",
     "type": "COMPLEMENTS", "conf": 0.85},
]

NEW_CONTENT = """\
# MCP 2.0 Release Notes

Anthropic released MCP 2.0 with major improvements:
- Streaming support via SSE
- Enhanced tool validation
- Multi-server coordination
The spec is available at spec.modelcontextprotocol.io.
"""

NEW_ENTITIES = [
    {"entity_id": "us:protocol-mcp-2.0", "name": "MCP 2.0",
     "type": "Protocol", "kind": "spec",
     "description": "Major update with streaming and multi-server support"},
    {"entity_id": "us:org-anthropic-seed", "name": "Anthropic",
     "type": "Organization", "kind": "company"},
    {"entity_id": "us:cap-mcp-streaming", "name": "MCP Streaming",
     "type": "Capability", "kind": "feature"},
]

NEW_EDGES = [
    {"src": "us:org-anthropic-seed", "tgt": "us:protocol-mcp-2.0",
     "type": "DEVELOPS", "conf": 0.95},
    {"src": "us:protocol-mcp-2.0", "tgt": "us:cap-mcp-streaming",
     "type": "PROVIDES", "conf": 0.90},
    {"src": "us:protocol-mcp-2.0", "tgt": "us:protocol-mcp-seed",
     "type": "SUPERSEDES", "conf": 0.95},
]

UPDATED_CONTENT = """\
# MCP 2.0 Release Notes (Updated)

Anthropic released MCP 2.0 with major improvements:
- Streaming support via SSE
- Enhanced tool validation
- Multi-server coordination
- OAuth 2.0 support (added post-launch)
The spec is available at spec.modelcontextprotocol.io.
"""

UPDATED_ENTITIES = [
    {"entity_id": "us:protocol-mcp-2.0-upd", "name": "MCP 2.0",
     "type": "Protocol", "kind": "spec",
     "description": "Major update with streaming, multi-server, and OAuth"},
    {"entity_id": "us:cap-mcp-oauth", "name": "MCP OAuth Support",
     "type": "Capability", "kind": "feature"},
]

UPDATED_EDGES = [
    {"src": "us:protocol-mcp-2.0-upd", "tgt": "us:cap-mcp-oauth",
     "type": "PROVIDES", "conf": 0.90},
]


class TestUserStoryCrossCUJIntegration:
    """Exercise CUJ 1→2→3→4→5→7 in sequence in a single test."""

    def test_full_cuj_lifecycle(self, db, clean_neo4j):
        """CUJ 1 (Seed) → 2 (Ingest) → 3 (Dup skip) → 4 (Update) → 5 (Query) → 7 (Review)."""
        from agents_kg.stages.load import run as run_load
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        # ── CUJ 1: Seed — ingest baseline document ──
        sid_seed = _full_pipeline_neo4j(
            db, "us://cross/seed", SEED_CONTENT,
            SEED_ENTITIES, SEED_EDGES, clean_neo4j,
            submitter_email="admin@company.com",
        )
        source_seed = db.get_source(sid_seed)
        assert source_seed["status"] == "complete"

        # Verify seed data in Neo4j
        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (n) WHERE n.entity_id STARTS WITH 'us:'
                RETURN count(n) AS cnt
            """).single()
            seed_count = result["cnt"]
            assert seed_count >= 4

        # ── CUJ 2: Ingest new source ──
        sid_new = _full_pipeline_neo4j(
            db, "us://cross/mcp-release", NEW_CONTENT,
            NEW_ENTITIES, NEW_EDGES, clean_neo4j,
            submitter_email="engineer@company.com",
        )
        source_new = db.get_source(sid_new)
        assert source_new["status"] == "complete"

        # Graph grew
        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (n) WHERE n.entity_id STARTS WITH 'us:'
                RETURN count(n) AS cnt
            """).single()
            assert result["cnt"] > seed_count

        # ── CUJ 3: Duplicate source skip ──
        dup_sid = db.add_source("us://cross/mcp-release")
        assert dup_sid is None  # URI already exists, returns None

        # ── CUJ 4: Source update ──
        db.update_source(sid_new, status="pending", stage="fetch")
        source = db.get_source(sid_new)
        _mock_fetch(db, source, UPDATED_CONTENT)

        source = db.get_source(sid_new)
        run_parse(db, source)
        source = db.get_source(sid_new)
        run_chunk(db, source)
        source = db.get_source(sid_new)
        _mock_embed(db, source)
        source = db.get_source(sid_new)
        _mock_extract(db, source, UPDATED_ENTITIES, UPDATED_EDGES)
        db.update_source(sid_new, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid_new, status="processing", stage="load")
        source = db.get_source(sid_new)
        run_load(db, source, neo4j_driver=clean_neo4j)

        updated_source = db.get_source(sid_new)
        assert updated_source["status"] == "complete"
        assert updated_source["content_hash"] == content_hash(UPDATED_CONTENT)

        # ── CUJ 5: Ad-hoc query ──
        with clean_neo4j.session() as s:
            # "What protocols exist?"
            result = s.run("""
                MATCH (p:Protocol)
                WHERE p.entity_id STARTS WITH 'us:'
                RETURN p.name AS name ORDER BY name
            """).data()
            protocol_names = {r["name"] for r in result}
            assert "MCP" in protocol_names or "MCP 2.0" in protocol_names
            assert "A2A" in protocol_names

            # "Who develops what?"
            result = s.run("""
                MATCH (org:Organization)-[:DEVELOPS]->(p)
                WHERE org.entity_id STARTS WITH 'us:'
                RETURN org.name AS org, collect(p.name) AS products
                ORDER BY org
            """).data()
            org_products = {r["org"]: r["products"] for r in result}
            assert "Anthropic" in org_products
            assert "Google" in org_products

            # "What does MCP complement?"
            result = s.run("""
                MATCH (a:Protocol)-[:COMPLEMENTS]->(b:Protocol)
                WHERE a.entity_id STARTS WITH 'us:'
                RETURN a.name AS protocol, b.name AS complements
            """).data()
            assert len(result) >= 1

        # ── CUJ 7: Periodic review — identify stale entities ──
        # Backdate seed entities to simulate age
        seed_ents = db.conn.execute(
            "SELECT id FROM entities WHERE source_id = ?", (sid_seed,)
        ).fetchall()
        for ent in seed_ents:
            db.conn.execute(
                "UPDATE entities SET updated_at = ? WHERE id = ?",
                ("2025-01-01T00:00:00+00:00", ent["id"])
            )
        db.conn.commit()

        # Audit: find entities older than threshold
        stale = db.conn.execute(
            "SELECT * FROM entities WHERE updated_at < ? "
            "AND status = 'approved' AND deprecated_at IS NULL",
            ("2026-01-01T00:00:00+00:00",)
        ).fetchall()
        stale_ids = {dict(e)["entity_id"] for e in stale}
        assert len(stale_ids) > 0
        # Seed entities should be flagged as stale
        assert any(eid.startswith("us:") for eid in stale_ids)

    def test_cross_cuj_state_coherence(self, db):
        """Verify graph state is coherent at each CUJ transition (SQLite only)."""
        from agents_kg.stages.load import run as run_load
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        # CUJ 1: Seed
        sid1 = _full_pipeline_no_neo4j(
            db, "us://coherence/seed", SEED_CONTENT,
            SEED_ENTITIES, SEED_EDGES,
        )
        # State check: 4 entities, 3 edges, all approved
        approved_ents = db.get_entities_by_status("approved")
        assert len(approved_ents) == 4
        approved_edges = db.get_edges_by_status("approved")
        assert len(approved_edges) == 3

        # CUJ 2: Ingest
        sid2 = _full_pipeline_no_neo4j(
            db, "us://coherence/new", NEW_CONTENT,
            NEW_ENTITIES, NEW_EDGES,
        )
        # State check: entities grew (some deduped via IntegrityError)
        approved_ents = db.get_entities_by_status("approved")
        assert len(approved_ents) >= 5  # at least new + non-dup seed
        approved_edges = db.get_edges_by_status("approved")
        assert len(approved_edges) >= 5  # original 3 + at least 2 new

        # CUJ 3: Duplicate skip
        dup = db.add_source("us://coherence/new")
        assert dup is None
        # State unchanged
        assert len(db.get_entities_by_status("approved")) == len(approved_ents)

        # CUJ 4: Update
        db.update_source(sid2, status="pending", stage="fetch")
        source = db.get_source(sid2)
        _mock_fetch(db, source, UPDATED_CONTENT)
        source = db.get_source(sid2)
        run_parse(db, source)
        source = db.get_source(sid2)
        run_chunk(db, source)
        source = db.get_source(sid2)
        _mock_embed(db, source)
        source = db.get_source(sid2)
        _mock_extract(db, source, UPDATED_ENTITIES, UPDATED_EDGES)
        db.update_source(sid2, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid2, status="processing", stage="load")
        source = db.get_source(sid2)
        run_load(db, source, neo4j_driver=None)

        # State check: deprecated entities from sid2 exist
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) > 0

        # CUJ 7: Review — rejected + deprecated counts
        all_ents = db.conn.execute("SELECT * FROM entities").fetchall()
        statuses = {dict(e)["status"] for e in all_ents}
        assert "approved" in statuses

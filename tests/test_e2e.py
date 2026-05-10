"""E2E tests — require live Neo4j and optionally Gemini API.

Run with: uv run pytest tests/test_e2e.py -m e2e -v
"""

import os
import pytest

pytestmark = pytest.mark.e2e

VALID_NODE_TYPES = {"Organization", "Group", "Person", "Project", "Protocol", "Capability", "Source", "Chunk"}
VALID_EDGE_TYPES = {"MEMBER_OF", "GOVERNS", "DEVELOPS", "IMPLEMENTS", "COMPETES_WITH", "ADDRESSES",
                    "AUTHORED", "CHAIRS", "SPONSORS", "PART_OF", "SUPERSEDES", "FROM_SOURCE",
                    "CONTRIBUTES_TO", "DEFINES", "COMPLEMENTS"}


@pytest.fixture(scope="module")
def neo4j_driver():
    """Connect to live Neo4j."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "agents-kg-2026"),
            ),
        )
        driver.verify_connectivity()
        yield driver
        driver.close()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.fixture
def clean_neo4j(neo4j_driver):
    """Clean test data before/after each test."""
    def cleanup():
        with neo4j_driver.session() as s:
            s.run("MATCH (n) WHERE n.entity_id STARTS WITH 'test:' DETACH DELETE n")
    cleanup()
    yield neo4j_driver
    cleanup()


class TestCUJ1EntityExtraction:
    """CUJ 1: Ingest URL → extract entities → verify types match ontology."""

    def test_extracted_entities_match_ontology(self, db):
        """Using mocked extraction, verify entity types are valid."""
        from agents_kg.stages.chunk import run as chunk_run
        from agents_kg.stages.parse import run as parse_run

        sid = db.add_source("https://example.com/a2a-spec")
        db.update_source(sid, raw_text="# A2A Protocol\n\nGoogle developed A2A for agent communication.\n\n## Overview\n\nA2A is an open protocol.", type="text")
        source = db.get_source(sid)
        parse_run(db, source)
        source = db.get_source(sid)
        chunk_run(db, source)

        # Simulate extraction
        chunks = db.get_chunks(sid)
        db.add_entity("organization:google", "Google", "Organization", kind="company", source_id=sid, chunk_id=chunks[0]["id"])
        db.add_entity("protocol:a2a", "A2A", "Protocol", kind="spec", source_id=sid, chunk_id=chunks[0]["id"])
        db.add_edge("e1", "organization:google", "protocol:a2a", "DEVELOPS", confidence=0.9, source_id=sid, chunk_id=chunks[0]["id"])

        entities = db.get_entities_by_status("pending_review")
        for e in entities:
            assert e["type"] in VALID_NODE_TYPES, f"Invalid type: {e['type']}"

        edges = db.get_edges_by_status("pending_review")
        for e in edges:
            assert e["edge_type"] in VALID_EDGE_TYPES, f"Invalid edge type: {e['edge_type']}"


class TestCUJ2Neo4jLoad:
    """CUJ 2: Load to Neo4j → query back → verify graph structure."""

    def test_load_and_query(self, db, clean_neo4j):
        from agents_kg.stages.load import run

        sid = db.add_source("https://example.com/test-load")

        # Add and approve entities
        db.add_entity("test:org-alpha", "Alpha Corp", "Organization", kind="company", source_id=sid)
        db.add_entity("test:project-beta", "Beta SDK", "Project", kind="sdk", source_id=sid)
        for ent in db.get_entities_by_status("pending_review"):
            db.approve_entity(ent["id"])

        db.add_edge("test-e1", "test:org-alpha", "test:project-beta", "DEVELOPS", confidence=0.95, source_id=sid)
        for edge in db.get_edges_by_status("pending_review"):
            db.approve_edge(edge["id"])

        source = db.get_source(sid)
        result = run(db, source, neo4j_driver=clean_neo4j)
        assert result is True

        # Query back
        with clean_neo4j.session() as session:
            result = session.run("MATCH (n {entity_id: 'test:org-alpha'}) RETURN n").single()
            assert result is not None
            node = result["n"]
            assert node["name"] == "Alpha Corp"
            assert node["type"] == "Organization"

            # Verify edge
            result = session.run(
                "MATCH (a {entity_id: 'test:org-alpha'})-[r:DEVELOPS]->(b {entity_id: 'test:project-beta'}) RETURN r"
            ).single()
            assert result is not None
            assert result["r"]["confidence"] == 0.95


class TestCUJ3Idempotency:
    """CUJ 3: Re-ingest same URL → verify no duplicates in Neo4j."""

    def test_no_duplicate_nodes(self, db, clean_neo4j):
        from agents_kg.stages.load import run

        sid = db.add_source("https://example.com/test-idem")
        db.add_entity("test:idem-org", "Idem Corp", "Organization", kind="company", source_id=sid)
        for ent in db.get_entities_by_status("pending_review"):
            db.approve_entity(ent["id"])

        source = db.get_source(sid)

        # Load twice
        run(db, source, neo4j_driver=clean_neo4j)
        # Reset and load again
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run(db, source, neo4j_driver=clean_neo4j)

        # Count nodes — should be exactly 1
        with clean_neo4j.session() as session:
            result = session.run(
                "MATCH (n {entity_id: 'test:idem-org'}) RETURN count(n) AS cnt"
            ).single()
            assert result["cnt"] == 1

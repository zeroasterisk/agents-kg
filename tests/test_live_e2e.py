"""Live end-to-end tests — NO MOCKS, real Neo4j + Gemini + Wikidata.

Each test class is a Critical User Journey (CUJ). Tests run against:
- Neo4j: bolt://35.202.188.73:7687
- Gemini: Vertex AI in data-ingest-demo
- Wikidata: https://query.wikidata.org/sparql

Run:
    NEO4J_URI=bolt://35.202.188.73:7687 \
    GOOGLE_CLOUD_PROJECT=data-ingest-demo \
    GOOGLE_GENAI_USE_VERTEXAI=true \
    .venv/bin/python -m pytest tests/test_live_e2e.py -v --tb=long
"""

import os
import tempfile
import time

import pytest
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://35.202.188.73:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")

MCP_OVERVIEW_DOC = """\
# Model Context Protocol (MCP) Overview

The Model Context Protocol (MCP) is an open protocol developed by Anthropic that \
standardizes how AI applications connect to external data sources and tools. MCP \
follows a client-server architecture where host applications (like Claude Desktop \
or IDEs) connect to MCP servers that expose resources, tools, and prompts.

## Key Features

MCP uses JSON-RPC 2.0 as its transport layer. Servers can expose three main \
primitives: Resources (data the AI can read), Tools (functions the AI can call), \
and Prompts (templated interactions). The protocol supports both stdio and \
HTTP+SSE transports.

## Ecosystem

Google has announced MCP support in Vertex AI and the Agent Development Kit (ADK). \
Anthropic's Claude Desktop was the first client to implement MCP. The MCP \
TypeScript SDK and MCP Python SDK are the official reference implementations, \
maintained by the modelcontextprotocol GitHub organization.

## Security Considerations

MCP addresses authentication and authorization through OAuth 2.1 integration. \
Server identity verification uses the SPIFFE framework for workload attestation. \
The protocol provides human-in-the-loop capabilities for sensitive operations.
"""


@pytest.fixture(scope="session")
def neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest.fixture()
def clean_neo4j(neo4j_driver):
    """Wipe all nodes and relationships before each test."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        # Drop all constraints/indexes so schema tests start fresh
        for record in session.run("SHOW CONSTRAINTS"):
            session.run(f"DROP CONSTRAINT {record['name']} IF EXISTS")
        for record in session.run("SHOW INDEXES"):
            if record["type"] != "LOOKUP":
                session.run(f"DROP INDEX {record['name']} IF EXISTS")


@pytest.fixture()
def tmp_db():
    """Provide a fresh SQLite pipeline DB in a temp dir."""
    from agents_kg.db import Database
    with tempfile.TemporaryDirectory() as d:
        db = Database(os.path.join(d, "test.db"))
        yield db
        db.close()


# ---------------------------------------------------------------------------
# CUJ 1 — Schema and Seed
# ---------------------------------------------------------------------------

class TestSchemaAndSeed:
    """Alan applies the schema, loads seed entities, and verifies constraints."""

    def test_apply_schema_creates_constraints_and_indexes(
        self, neo4j_driver, clean_neo4j
    ):
        from agents_kg.schema import apply_schema

        result = apply_schema(neo4j_driver)
        assert result["constraints"] >= 4
        assert result["indexes"] >= 4
        assert result["errors"] == []

        with neo4j_driver.session() as session:
            constraints = session.run("SHOW CONSTRAINTS").data()
            names = {c["name"] for c in constraints}
            assert "entity_id_unique" in names
            assert "source_uri_unique" in names

    def test_seed_entities_load_with_correct_labels(
        self, neo4j_driver, clean_neo4j
    ):
        from agents_kg.schema import apply_schema
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities

        apply_schema(neo4j_driver)
        seeds = get_seed_entities()
        load_wikidata_entities(neo4j_driver, seeds)

        with neo4j_driver.session() as session:
            count = session.run(
                "MATCH (n:Entity) RETURN count(n) AS c"
            ).single()["c"]
            assert count >= len(seeds) - 5  # some may dedup by entity_id

            google = session.run(
                "MATCH (n:Organization {entity_id: 'organization:google'}) "
                "RETURN n.name AS name"
            ).single()
            assert google is not None
            assert google["name"] == "Google"

            mcp = session.run(
                "MATCH (n:Protocol {entity_id: 'protocol:mcp'}) "
                "RETURN n.name AS name"
            ).single()
            assert mcp is not None
            assert mcp["name"] == "Model Context Protocol"

    def test_unique_constraint_prevents_duplicate_entity_id(
        self, neo4j_driver, clean_neo4j
    ):
        from agents_kg.schema import apply_schema
        from neo4j.exceptions import ClientError

        apply_schema(neo4j_driver)

        with neo4j_driver.session() as session:
            session.run(
                "CREATE (n:Entity {entity_id: 'test:dup', name: 'First'})"
            )
            with pytest.raises(ClientError):
                session.run(
                    "CREATE (n:Entity {entity_id: 'test:dup', name: 'Second'})"
                )


# ---------------------------------------------------------------------------
# CUJ 2 — Real Source Ingestion with Live Gemini
# ---------------------------------------------------------------------------

class TestRealSourceIngestion:
    """Alan ingests a short MCP overview document through the full pipeline
    with real Gemini embedding + extraction, then verifies entities in Neo4j.
    """

    def test_full_pipeline_ingestion(self, neo4j_driver, clean_neo4j, tmp_db):
        from agents_kg.schema import apply_schema
        from agents_kg.stages import fetch, parse, chunk, embed, extract, resolve, load

        apply_schema(neo4j_driver)

        # Write the test document to a temp file so fetch can read it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(MCP_OVERVIEW_DOC)
            doc_path = f.name

        try:
            source_id = tmp_db.add_source(
                doc_path, title="MCP Overview", source_type="text",
                submitter_email="alan@test.com"
            )
            assert source_id is not None

            source = tmp_db.get_source(source_id)

            # Stage 1: Fetch
            assert fetch.run(tmp_db, source)
            source = tmp_db.get_source(source_id)
            assert source["content_hash"] is not None
            assert source["stage"] == "parse"

            # Stage 2: Parse
            assert parse.run(tmp_db, source)
            source = tmp_db.get_source(source_id)
            assert source["parsed_text"] is not None
            assert source["stage"] == "chunk"

            # Stage 3: Chunk
            assert chunk.run(tmp_db, source)
            source = tmp_db.get_source(source_id)
            chunks = tmp_db.get_chunks(source_id)
            assert len(chunks) >= 1
            assert source["stage"] == "embed"

            # Stage 4: Embed — REAL GEMINI CALL
            assert embed.run(tmp_db, source)
            source = tmp_db.get_source(source_id)
            chunks = tmp_db.get_chunks(source_id)
            assert all(c["embedding"] is not None for c in chunks)
            assert source["stage"] == "extract"

            # Stage 5: Extract — REAL GEMINI CALL
            assert extract.run(tmp_db, source)
            source = tmp_db.get_source(source_id)
            entities = tmp_db.conn.execute(
                "SELECT * FROM entities WHERE source_id = ?", (source_id,)
            ).fetchall()
            assert len(entities) >= 2, f"Expected >=2 entities, got {len(entities)}"

            # Stage 6: Resolve
            assert resolve.run(tmp_db, source)
            source = tmp_db.get_source(source_id)

            # Auto-approve all entities and edges for testing
            tmp_db.conn.execute(
                "UPDATE entities SET status = 'approved' WHERE source_id = ? AND status = 'pending_review'",
                (source_id,),
            )
            tmp_db.conn.execute(
                "UPDATE edges SET status = 'approved' WHERE source_id = ? AND status = 'pending_review'",
                (source_id,),
            )
            tmp_db.conn.commit()
            tmp_db.update_source(source_id, status="processing", stage="load")
            source = tmp_db.get_source(source_id)

            # Stage 7: Load to Neo4j — REAL NEO4J WRITE
            assert load.run(tmp_db, source, neo4j_driver=neo4j_driver)

            # Verify entities in Neo4j
            with neo4j_driver.session() as session:
                entity_count = session.run(
                    "MATCH (n:Entity) RETURN count(n) AS c"
                ).single()["c"]
                assert entity_count >= 2

                # The source node should exist with provenance
                src_node = session.run(
                    "MATCH (s:Source) WHERE s.uri = $uri RETURN s.submitter_email AS email",
                    {"uri": doc_path},
                ).single()
                assert src_node is not None
                assert src_node["email"] == "alan@test.com"

                # There should be FROM_SOURCE edges
                from_source = session.run(
                    "MATCH ()-[r:FROM_SOURCE]->() RETURN count(r) AS c"
                ).single()["c"]
                assert from_source >= 1

        finally:
            os.unlink(doc_path)


# ---------------------------------------------------------------------------
# CUJ 3 — Content-Hash Dedup on Re-Ingestion
# ---------------------------------------------------------------------------

class TestContentHashDedup:
    """Alan submits the same source twice; the content hash prevents
    reprocessing on the second submission.
    """

    def test_same_content_skips_reprocessing(self, tmp_db):
        from agents_kg.stages import fetch

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Dedup Test\n\nSome content about protocols.")
            doc_path = f.name

        try:
            # First ingest
            s1 = tmp_db.add_source(doc_path, title="Dedup test")
            source = tmp_db.get_source(s1)
            assert fetch.run(tmp_db, source)
            source = tmp_db.get_source(s1)
            first_hash = source["content_hash"]
            assert first_hash is not None

            # Reset to fetch stage to re-ingest same content
            tmp_db.update_source(s1, stage="fetch", status="pending")
            source = tmp_db.get_source(s1)

            # Second fetch — same content, should return False (no progress)
            result = fetch.run(tmp_db, source)
            assert result is False, "Expected fetch to skip unchanged content"

            source = tmp_db.get_source(s1)
            assert source["status"] == "complete"
            assert source["content_hash"] == first_hash
        finally:
            os.unlink(doc_path)

    def test_duplicate_uri_rejected(self, tmp_db):
        s1 = tmp_db.add_source("https://example.com/test-doc")
        assert s1 is not None
        s2 = tmp_db.add_source("https://example.com/test-doc")
        assert s2 is None, "Duplicate URI should return None"


# ---------------------------------------------------------------------------
# CUJ 4 — Real Wikidata Pull
# ---------------------------------------------------------------------------

class TestWikidataPull:
    """Alan pulls protocol entities from Wikidata and verifies they appear
    in Neo4j with wikidata_id properties.
    """

    def test_pull_protocols_to_neo4j(self, neo4j_driver, clean_neo4j):
        from agents_kg.schema import apply_schema
        from agents_kg.wikidata import pull_and_load

        apply_schema(neo4j_driver)

        result = pull_and_load(neo4j_driver, entity_type="protocols")
        assert result["entities"] > 0, "Expected Wikidata to return protocol entities"

        with neo4j_driver.session() as session:
            # Check that at least some entities have wikidata_id set
            wd_count = session.run(
                "MATCH (n:Entity) WHERE n.wikidata_id IS NOT NULL "
                "RETURN count(n) AS c"
            ).single()["c"]
            assert wd_count > 0, f"Expected wikidata entities, got {wd_count}"

            # Check that entities were created with source_type=wikidata
            wikidata_src = session.run(
                "MATCH (n:Entity {source_type: 'wikidata'}) RETURN count(n) AS c"
            ).single()["c"]
            assert wikidata_src > 0


# ---------------------------------------------------------------------------
# CUJ 5 — Cross-Domain Query (Seed + Wikidata)
# ---------------------------------------------------------------------------

class TestCrossDomainQuery:
    """With seed and Wikidata data loaded together, Alan runs a Cypher query
    to find organizations that develop protocols.
    """

    def test_seed_plus_wikidata_cross_query(self, neo4j_driver, clean_neo4j):
        from agents_kg.schema import apply_schema
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities, pull_and_load

        apply_schema(neo4j_driver)

        # Load seed entities
        seeds = get_seed_entities()
        load_wikidata_entities(neo4j_driver, seeds)

        # Pull Wikidata protocols (real SPARQL)
        pull_and_load(neo4j_driver, entity_type="protocols")

        with neo4j_driver.session() as session:
            # Verify seed entities are present (check known seed entity_ids)
            seed_present = session.run(
                "MATCH (n:Entity) WHERE n.entity_id IN "
                "['organization:google', 'protocol:mcp', 'project:gemini'] "
                "RETURN count(n) AS c"
            ).single()["c"]
            # Wikidata entities have wikidata_id property set
            wd_count = session.run(
                "MATCH (n:Entity) WHERE n.wikidata_id IS NOT NULL "
                "RETURN count(n) AS c"
            ).single()["c"]
            assert seed_present >= 3, f"Seed entities missing, found {seed_present}"
            assert wd_count > 0, "No Wikidata entities found"

            # Cross-domain: find all Protocol entities regardless of source
            all_protocols = session.run(
                "MATCH (n:Protocol) RETURN n.entity_id AS eid, "
                "n.name AS name, n.source_type AS src ORDER BY n.name"
            ).data()
            assert len(all_protocols) >= 5, (
                f"Expected >=5 protocols from seed+wikidata, got {len(all_protocols)}"
            )

            # Verify at least one seed protocol is present
            seed_eids = {r["eid"] for r in all_protocols}
            assert "protocol:mcp" in seed_eids, "Seed protocol:mcp not found"

            # Verify organizations exist to query against
            org_count = session.run(
                "MATCH (n:Organization) RETURN count(n) AS c"
            ).single()["c"]
            assert org_count > 0, "No organizations found"

            # Query: find DEVELOPS relationships (from Wikidata edges)
            develops = session.run(
                "MATCH (o)-[r:DEVELOPS]->(p:Protocol) "
                "RETURN o.name AS org, p.name AS proto LIMIT 20"
            ).data()
            # DEVELOPS edges come from Wikidata edge extraction — may or may not exist
            # depending on which protocols had developer metadata, so just log
            print(f"  Found {len(develops)} DEVELOPS->Protocol relationships")
            for d in develops[:5]:
                print(f"    {d['org']} -> {d['proto']}")

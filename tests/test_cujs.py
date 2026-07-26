"""CUJ integration tests — require live Neo4j.

Run with: .venv/bin/python -m pytest tests/test_cujs.py -v
"""

import os
import struct
import pytest
from unittest.mock import MagicMock, patch

from agents_kg.db import Database, content_hash

pytestmark = pytest.mark.e2e

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")

# ---------------------------------------------------------------------------
# Test content fixtures
# ---------------------------------------------------------------------------

TEST_SOURCE_CONTENT_V1 = """\
# Agent-to-Agent Protocol

Google developed the A2A protocol for inter-agent communication.
Anthropic's MCP provides a complementary approach.

## Architecture

A2A uses JSON-RPC 2.0 over HTTP for transport.
"""

TEST_SOURCE_CONTENT_V2 = """\
# Agent-to-Agent Protocol v2

Google updated the A2A protocol with new discovery features.
OpenAI also joined the A2A ecosystem.

## Architecture

A2A v2 adds streaming support over SSE.
"""

TEST_ENTITIES = [
    {"entity_id": "test:org-google", "name": "Google", "type": "Organization", "kind": "company"},
    {"entity_id": "test:protocol-a2a", "name": "A2A", "type": "Protocol", "kind": "spec"},
    {"entity_id": "test:org-anthropic", "name": "Anthropic", "type": "Organization", "kind": "company"},
]

TEST_EDGES = [
    {"src": "test:org-google", "tgt": "test:protocol-a2a", "type": "DEVELOPS", "conf": 0.95},
]

TEST_ENTITIES_V2 = [
    {"entity_id": "test:org-google", "name": "Google", "type": "Organization", "kind": "company"},
    {"entity_id": "test:protocol-a2a", "name": "A2A v2", "type": "Protocol", "kind": "spec"},
    {"entity_id": "test:org-openai", "name": "OpenAI", "type": "Organization", "kind": "company"},
]

TEST_EDGES_V2 = [
    {"src": "test:org-google", "tgt": "test:protocol-a2a", "type": "DEVELOPS", "conf": 0.95},
    {"src": "test:org-openai", "tgt": "test:protocol-a2a", "type": "CONTRIBUTES_TO", "conf": 0.8},
]

# ---------------------------------------------------------------------------
# Neo4j fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neo4j_driver():
    """Connect to live Neo4j, skip if unavailable."""
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
    """Delete test-prefixed data before and after each test."""
    def _cleanup():
        with neo4j_driver.session() as s:
            s.run("MATCH (n) WHERE n.entity_id IS NOT NULL AND n.entity_id STARTS WITH 'test:' DETACH DELETE n")
            s.run("MATCH (s:Source) WHERE s.uri STARTS WITH 'test://' DETACH DELETE s")
            s.run("MATCH (e:Event) WHERE e.event_id STARTS WITH 'test-' DETACH DELETE e")
    _cleanup()
    yield neo4j_driver
    _cleanup()


@pytest.fixture
def full_clean_neo4j(neo4j_driver):
    """Clean test-scoped nodes for CUJ 1 seed reset.

    ⚠️  NEVER run ``MATCH (n) DETACH DELETE n`` — that wipes production data.
    """
    with neo4j_driver.session() as s:
        s.run("MATCH (n) WHERE n.entity_id STARTS WITH 'test:' DETACH DELETE n")
        s.run("MATCH (n:_Test) DETACH DELETE n")
    yield neo4j_driver

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def mock_embed(db, source):
    """Write fake embeddings directly to DB, skip Gemini API."""
    source_id = source["id"]
    chunks = db.get_unembedded_chunks(source_id)
    for c in chunks:
        emb = struct.pack("3f", 0.1, 0.2, 0.3)
        db.update_chunk_embedding(c["id"], emb, "gemini-embedding-2")
    db.update_source(source_id, stage="extract", status="processing")


def mock_extract_known_entities(db, source, entities, edges):
    """Write known entities and edges to DB, skip Gemini extraction."""
    from agents_kg.stages.extract import _make_edge_id
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


def _mock_fetch_with_content(db, source, content):
    """Run fetch stage with mocked HTTP returning the given content."""
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
    """Run fetch(mock)->parse->chunk and return refreshed source."""
    from agents_kg.stages.parse import run as run_parse
    from agents_kg.stages.chunk import run as run_chunk

    source = db.get_source(source_id)
    _mock_fetch_with_content(db, source, content)

    source = db.get_source(source_id)
    run_parse(db, source)

    source = db.get_source(source_id)
    run_chunk(db, source)

    return db.get_source(source_id)


def _approve_all(db):
    """Approve all pending entities and edges."""
    for ent in db.get_entities_by_status("pending_review"):
        db.approve_entity(ent["id"])
    for edge in db.get_edges_by_status("pending_review"):
        db.approve_edge(edge["id"])


# ---------------------------------------------------------------------------
# CUJ 1: Seed Reset
# ---------------------------------------------------------------------------


class TestCUJ1SeedReset:
    """CUJ 1: Clear graph, apply schema, load seeds, verify baseline."""

    def test_full_reset_and_seed(self, full_clean_neo4j):
        driver = full_clean_neo4j

        # Verify empty after wipe
        with driver.session() as s:
            count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            assert count == 0

        # Apply schema
        from agents_kg.schema import apply_schema, CONSTRAINTS, INDEXES
        results = apply_schema(driver)
        assert results["constraints"] == len(CONSTRAINTS)
        assert results["indexes"] == len(INDEXES)
        assert results["errors"] == []

        # Load seed entities
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities
        seeds = get_seed_entities()
        load_wikidata_entities(driver, seeds)

        # Verify nodes exist
        with driver.session() as s:
            node_count = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            assert node_count >= len(seeds)

        # Verify constraint exists
        with driver.session() as s:
            constraints = s.run("SHOW CONSTRAINTS").data()
            names = [c.get("name", "") for c in constraints]
            assert any("entity_id" in n for n in names)

        # Verify entity types present
        with driver.session() as s:
            types = s.run(
                "MATCH (n:Entity) RETURN DISTINCT n.type AS t ORDER BY t"
            ).data()
            type_set = {t["t"] for t in types}
            assert "Organization" in type_set
            assert "Protocol" in type_set
            assert "Project" in type_set
            assert "Capability" in type_set

    def test_schema_idempotent(self, full_clean_neo4j):
        from agents_kg.schema import apply_schema
        r1 = apply_schema(full_clean_neo4j)
        r2 = apply_schema(full_clean_neo4j)
        assert r2["errors"] == []

    def test_seed_entities_idempotent(self, full_clean_neo4j):
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities
        seeds = get_seed_entities()
        load_wikidata_entities(full_clean_neo4j, seeds)
        load_wikidata_entities(full_clean_neo4j, seeds)

        with full_clean_neo4j.session() as s:
            count = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            # MERGE prevents duplicates — count should equal seed count
            assert count == len(seeds)


# ---------------------------------------------------------------------------
# CUJ 2: New Source Ingestion
# ---------------------------------------------------------------------------


class TestCUJ2NewSourceIngestion:
    """CUJ 2: Add source, process pipeline, verify entities in Neo4j with provenance."""

    def test_full_ingestion_to_neo4j(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://example.com/a2a-doc",
                            submitter_email="tester@example.com")
        assert sid is not None

        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES, TEST_EDGES)

        # Skip resolve — advance to review
        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")

        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        # Verify entities in Neo4j
        with clean_neo4j.session() as s:
            for ent in TEST_ENTITIES:
                result = s.run(
                    "MATCH (n {entity_id: $eid}) RETURN n",
                    {"eid": ent["entity_id"]}
                ).single()
                assert result is not None, f"Entity {ent['entity_id']} not found in Neo4j"
                node = result["n"]
                assert node["name"] == ent["name"]
                assert node["type"] == ent["type"]

        # Verify edges in Neo4j
        with clean_neo4j.session() as s:
            for edge in TEST_EDGES:
                result = s.run(
                    f"MATCH (a {{entity_id: $src}})-[r:{edge['type']}]->(b {{entity_id: $tgt}}) RETURN r",
                    {"src": edge["src"], "tgt": edge["tgt"]}
                ).single()
                assert result is not None, f"Edge {edge['src']}->{edge['tgt']} not found"

    def test_provenance_metadata(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://example.com/provenance-test",
                            submitter_email="alice@example.com")
        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES[:1], [])

        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")

        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        # Verify provenance in SQLite
        stored = db.get_source(sid)
        assert stored["submitter_email"] == "alice@example.com"
        assert stored["content_hash"] is not None
        assert stored["created_at"] is not None

        # Verify Source node in Neo4j
        with clean_neo4j.session() as s:
            result = s.run(
                "MATCH (s:Source {uri: $uri}) RETURN s",
                {"uri": "test://example.com/provenance-test"}
            ).single()
            assert result is not None, "Source node not found in Neo4j"
            source_node = result["s"]
            assert source_node["submitter_email"] == "alice@example.com"
            assert source_node["content_hash"] is not None

    def test_from_source_edge(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://example.com/from-source-test",
                            submitter_email="bob@example.com")
        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES[:1], [])

        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")

        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        # Verify FROM_SOURCE edge
        with clean_neo4j.session() as s:
            result = s.run(
                "MATCH (n {entity_id: $eid})-[:FROM_SOURCE]->(s:Source {uri: $uri}) RETURN s",
                {"eid": TEST_ENTITIES[0]["entity_id"], "uri": "test://example.com/from-source-test"}
            ).single()
            assert result is not None, "FROM_SOURCE edge not found"

    def test_stage_progression(self, db):
        sid = db.add_source("test://example.com/stages")
        assert db.get_source(sid)["stage"] == "fetch"

        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        assert db.get_source(sid)["stage"] == "embed"

        mock_embed(db, source)
        assert db.get_source(sid)["stage"] == "extract"

        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES[:1], [])
        assert db.get_source(sid)["stage"] == "resolve"


# ---------------------------------------------------------------------------
# CUJ 3: Duplicate Source Skip
# ---------------------------------------------------------------------------


class TestCUJ3DuplicateSourceSkip:
    """CUJ 3: Ingest same source twice, verify skip on second pass."""

    def test_duplicate_uri_rejected_at_add(self, db):
        sid1 = db.add_source("test://example.com/dup")
        sid2 = db.add_source("test://example.com/dup")
        assert sid1 is not None
        assert sid2 is None

    def test_content_hash_skip(self, db):
        content = TEST_SOURCE_CONTENT_V1
        sid = db.add_source("test://example.com/hash-dup")

        # First fetch
        source = db.get_source(sid)
        result1 = _mock_fetch_with_content(db, source, content)
        assert result1 is True

        # Simulate re-fetch: reset stage but keep content_hash
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        assert source["content_hash"] == content_hash(content)

        result2 = _mock_fetch_with_content(db, source, content)
        assert result2 is False  # Skipped — content unchanged

        updated = db.get_source(sid)
        assert updated["status"] == "complete"
        assert updated["stage"] == "done"

    def test_no_new_chunks_on_skip(self, db):
        content = TEST_SOURCE_CONTENT_V1
        sid = db.add_source("test://example.com/chunk-dup")

        # First pass — full pipeline through chunk
        _run_through_chunk(db, sid, content)
        chunks_after_first = db.get_chunks(sid)
        assert len(chunks_after_first) >= 1

        # Re-fetch same content — should skip entirely
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        result = _mock_fetch_with_content(db, source, content)
        assert result is False

    def test_no_duplicate_entities_in_neo4j(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://example.com/neo4j-dup")
        db.add_entity("test:dup-org", "Dup Corp", "Organization",
                       kind="company", source_id=sid)
        _approve_all(db)

        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        # Load again
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            count = s.run(
                "MATCH (n {entity_id: 'test:dup-org'}) RETURN count(n) AS c"
            ).single()["c"]
            assert count == 1


# ---------------------------------------------------------------------------
# CUJ 4: Source Update (Changed Content)
# ---------------------------------------------------------------------------


class TestCUJ4SourceUpdate:
    """CUJ 4: Update source content, verify stale marking and replacement."""

    def test_content_change_marks_old_entities_stale(self, db):
        sid = db.add_source("test://example.com/update-test")

        # First ingestion
        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES, TEST_EDGES)
        _approve_all(db)

        original_hash = db.get_source(sid)["content_hash"]
        assert original_hash == content_hash(TEST_SOURCE_CONTENT_V1)

        # Re-fetch with changed content
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        _mock_fetch_with_content(db, source, TEST_SOURCE_CONTENT_V2)

        new_hash = db.get_source(sid)["content_hash"]
        assert new_hash != original_hash

        # Verify old entities are deprecated
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) == len(TEST_ENTITIES)
        deprecated_ids = {e["entity_id"] for e in deprecated}
        for ent in TEST_ENTITIES:
            assert ent["entity_id"] in deprecated_ids

    def test_new_content_produces_new_entities(self, db):
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("test://example.com/update-new")

        # V1
        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES, TEST_EDGES)
        _approve_all(db)

        # V2 — re-fetch triggers deprecation + clears chunk refs
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        _mock_fetch_with_content(db, source, TEST_SOURCE_CONTENT_V2)

        # Continue V2 pipeline: parse -> chunk -> embed -> extract
        source = db.get_source(sid)
        run_parse(db, source)
        source = db.get_source(sid)
        run_chunk(db, source)
        source = db.get_source(sid)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES_V2, TEST_EDGES_V2)

        # test:org-openai is new (from V2 only)
        all_entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        entity_ids = {dict(e)["entity_id"] for e in all_entities}
        assert "test:org-openai" in entity_ids

    def test_deprecated_entities_preserved(self, db):
        sid = db.add_source("test://example.com/preserve-test")

        # V1
        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES, [])
        _approve_all(db)

        # V2 — Anthropic not in V2, so it stays deprecated
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        _mock_fetch_with_content(db, source, TEST_SOURCE_CONTENT_V2)

        deprecated = db.get_deprecated_entities()
        deprecated_ids = {e["entity_id"] for e in deprecated}
        assert "test:org-anthropic" in deprecated_ids

    def test_stale_entities_in_neo4j(self, db, clean_neo4j):
        """After update, loading V2 to Neo4j includes new entities."""
        from agents_kg.stages.load import run as run_load
        from agents_kg.stages.parse import run as run_parse
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("test://example.com/neo4j-update")

        # V1 full pipeline + load
        source = _run_through_chunk(db, sid, TEST_SOURCE_CONTENT_V1)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES, TEST_EDGES)
        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        # V2 — re-fetch triggers deprecation, then continue pipeline
        db.update_source(sid, status="pending", stage="fetch")
        source = db.get_source(sid)
        _mock_fetch_with_content(db, source, TEST_SOURCE_CONTENT_V2)

        source = db.get_source(sid)
        run_parse(db, source)
        source = db.get_source(sid)
        run_chunk(db, source)
        source = db.get_source(sid)
        mock_embed(db, source)
        source = db.get_source(sid)
        mock_extract_known_entities(db, source, TEST_ENTITIES_V2, TEST_EDGES_V2)
        db.update_source(sid, stage="review", status="pending_review")
        _approve_all(db)
        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        # test:org-openai should now be in Neo4j
        with clean_neo4j.session() as s:
            result = s.run(
                "MATCH (n {entity_id: 'test:org-openai'}) RETURN n"
            ).single()
            assert result is not None, "New entity test:org-openai not found in Neo4j"


# ---------------------------------------------------------------------------
# CUJ 5: Ad-hoc Query
# ---------------------------------------------------------------------------


class TestCUJ5AdhocQuery:
    """CUJ 5: Load known data, run Cypher queries, verify results."""

    @pytest.fixture(autouse=True)
    def _setup_test_data(self, clean_neo4j):
        self.driver = clean_neo4j
        with self.driver.session() as s:
            s.run("""
                CREATE (g:Entity:Organization {entity_id: 'test:org-google',
                    name: 'Google', type: 'Organization', kind: 'company'})
                CREATE (a2a:Entity:Protocol {entity_id: 'test:protocol-a2a',
                    name: 'A2A', type: 'Protocol', kind: 'spec'})
                CREATE (mcp:Entity:Protocol {entity_id: 'test:protocol-mcp',
                    name: 'MCP', type: 'Protocol', kind: 'spec', wikidata_id: 'Q12345'})
                CREATE (adk:Entity:Project {entity_id: 'test:project-adk',
                    name: 'ADK', type: 'Project', kind: 'framework'})
                CREATE (g)-[:DEVELOPS]->(a2a)
                CREATE (g)-[:DEVELOPS]->(adk)
                CREATE (a2a)-[:COMPLEMENTS]->(mcp)
            """)
            s.run("""
                CREATE (e:Event {event_id: 'test-launch-2025-06-01',
                    title: 'Test Launch', event_type: 'launch',
                    date: date('2025-06-01')})
                WITH e
                MATCH (g:Entity {entity_id: 'test:org-google'})
                CREATE (g)-[:PARTICIPATED_IN {role: 'organizer'}]->(e)
            """)

    def test_query_org_develops(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (org:Organization {entity_id: 'test:org-google'})
                      -[:DEVELOPS]->(p)
                RETURN p.name AS name ORDER BY name
            """).data()
            names = [r["name"] for r in result]
            assert "A2A" in names
            assert "ADK" in names

    def test_query_entities_with_wikidata(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.entity_id STARTS WITH 'test:'
                  AND n.wikidata_id IS NOT NULL
                RETURN n.name AS name, n.wikidata_id AS qid
            """).data()
            assert len(result) >= 1
            assert any(r["qid"] == "Q12345" for r in result)

    def test_query_graph_stats(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.entity_id STARTS WITH 'test:'
                RETURN n.type AS type, count(*) AS count
                ORDER BY count DESC
            """).data()
            type_counts = {r["type"]: r["count"] for r in result}
            assert "Organization" in type_counts
            assert "Protocol" in type_counts
            assert "Project" in type_counts

    def test_query_timeline_events(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (e:Event)
                WHERE e.event_id STARTS WITH 'test-'
                  AND e.date >= date('2025-01-01')
                  AND e.date < date('2026-01-01')
                OPTIONAL MATCH (entity)-[:PARTICIPATED_IN]->(e)
                RETURN e.title AS title, e.date AS date,
                       collect(entity.name) AS participants
            """).data()
            assert len(result) >= 1
            assert result[0]["title"] == "Test Launch"
            assert "Google" in result[0]["participants"]

    def test_query_relationship_traversal(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (org:Organization {entity_id: 'test:org-google'})
                      -[:DEVELOPS]->(p:Protocol)-[:COMPLEMENTS]->(other:Protocol)
                RETURN org.name AS org, p.name AS protocol, other.name AS complement
            """).data()
            assert len(result) == 1
            assert result[0]["complement"] == "MCP"

    def test_query_entity_by_kind(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.entity_id STARTS WITH 'test:' AND n.kind = 'spec'
                RETURN n.name AS name ORDER BY name
            """).data()
            names = [r["name"] for r in result]
            assert "A2A" in names
            assert "MCP" in names


# ---------------------------------------------------------------------------
# CUJ 6: Temporary Source Analysis (DEFERRED)
# ---------------------------------------------------------------------------


class TestCUJ6TemporarySource:
    """CUJ 6: Temporary source with soft-delete. DEFERRED."""

    @pytest.mark.skip(reason="CUJ 6 deferred — temporary source analysis not yet designed")
    def test_temporary_source_placeholder(self):
        # TODO: Implement temporary source analysis
        # Intent: User submits a source, analyzes it, then removes it.
        # Removal = soft delete: mark source and its exclusive entities as inactive/hidden.
        # Entities shared with other sources remain active.
        # Suggested approach:
        #   - Add is_active/hidden flag to sources table
        #   - On soft-delete, mark source inactive
        #   - Mark entities exclusive to that source as hidden
        #   - Entities shared with other active sources stay visible
        pass


# ---------------------------------------------------------------------------
# CUJ 7: Periodic Review and Cleanup
# ---------------------------------------------------------------------------


class TestCUJ7PeriodicReview:
    """CUJ 7: Identify stale entities via deprecated_at or old timestamps."""

    def test_deprecated_entities_queryable(self, db):
        sid = db.add_source("test://example.com/review-test")
        db.add_entity("test:old-org", "Old Corp", "Organization",
                       kind="company", source_id=sid)

        ent = db.get_entities_by_status("pending_review")[0]
        db.approve_entity(ent["id"])

        db.update_entity(ent["id"], deprecated_at="2025-01-01T00:00:00+00:00")

        deprecated = db.get_deprecated_entities()
        assert len(deprecated) >= 1
        assert any(e["entity_id"] == "test:old-org" for e in deprecated)

    def test_old_entities_identifiable_by_timestamp(self, db):
        sid = db.add_source("test://example.com/old-data")
        db.add_entity("test:stale-project", "Stale Project", "Project",
                       kind="framework", source_id=sid)

        ent = db.get_entities_by_status("pending_review")[0]
        db.approve_entity(ent["id"])

        # Backdate the entity
        db.conn.execute(
            "UPDATE entities SET created_at = ?, updated_at = ? WHERE id = ?",
            ("2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00", ent["id"])
        )
        db.conn.commit()

        old_entities = db.conn.execute(
            "SELECT * FROM entities WHERE updated_at < ? AND deprecated_at IS NULL AND status = 'approved'",
            ("2025-06-01T00:00:00+00:00",)
        ).fetchall()

        assert len(old_entities) >= 1
        assert any(dict(e)["entity_id"] == "test:stale-project" for e in old_entities)

    def test_bulk_deprecation_by_source(self, db):
        sid = db.add_source("test://example.com/bulk-dep")
        db.add_entity("test:bulk-1", "Entity 1", "Organization", source_id=sid)
        db.add_entity("test:bulk-2", "Entity 2", "Project", source_id=sid)

        for ent in db.get_entities_by_status("pending_review"):
            db.approve_entity(ent["id"])

        db.deprecate_entities_for_source(sid)

        deprecated = db.get_deprecated_entities()
        deprecated_ids = {e["entity_id"] for e in deprecated}
        assert "test:bulk-1" in deprecated_ids
        assert "test:bulk-2" in deprecated_ids

    def test_deprecation_skips_already_merged(self, db):
        sid = db.add_source("test://example.com/merged-dep")
        db.add_entity("test:merged-ent", "Merged Entity", "Organization", source_id=sid)

        ent = db.get_entities_by_status("pending_review")[0]
        db.update_entity(ent["id"], status="merged", merged_into="test:canonical")

        db.deprecate_entities_for_source(sid)

        # Merged entity should NOT be deprecated
        deprecated = db.get_deprecated_entities()
        deprecated_ids = {e["entity_id"] for e in deprecated}
        assert "test:merged-ent" not in deprecated_ids

    def test_stale_review_neo4j_query(self, db, clean_neo4j):
        """Verify we can identify entities needing review via Neo4j queries."""
        with clean_neo4j.session() as s:
            # Create entities with varying ages
            s.run("""
                CREATE (old:Entity:Organization {entity_id: 'test:old-corp',
                    name: 'Old Corp', type: 'Organization',
                    updated_at: '2024-01-01'})
                CREATE (new:Entity:Organization {entity_id: 'test:new-corp',
                    name: 'New Corp', type: 'Organization',
                    updated_at: '2026-04-01'})
            """)

        with clean_neo4j.session() as s:
            # Query for entities with old updated_at
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.entity_id STARTS WITH 'test:'
                  AND n.updated_at < '2025-01-01'
                RETURN n.entity_id AS id, n.name AS name, n.updated_at AS updated
            """).data()
            assert len(result) == 1
            assert result[0]["id"] == "test:old-corp"

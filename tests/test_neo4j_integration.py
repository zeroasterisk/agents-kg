"""Neo4j integration tests — exercises real Neo4j for end-to-end pipeline,
graph traversal, schema enforcement, temporal model, and wikidata cross-ref.

All test data uses 'test:' prefixed entity_ids and 'test://' URIs for cleanup.
"""

import os
import struct
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import yaml

from agents_kg.db import Database, content_hash

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "agents-kg-2026")


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
            s.run("MATCH (n) WHERE n.entity_id IS NOT NULL AND n.entity_id STARTS WITH 'test:' DETACH DELETE n")
            s.run("MATCH (s:Source) WHERE s.uri STARTS WITH 'test://' DETACH DELETE s")
            s.run("MATCH (e:Event) WHERE e.event_id STARTS WITH 'test-' DETACH DELETE e")
            s.run("MATCH (c:Chunk) WHERE c.chunk_id IS NOT NULL AND c.source_id < 0 DETACH DELETE c")
    _cleanup()
    yield neo4j_driver
    _cleanup()


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


# ---------------------------------------------------------------------------
# Helpers
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


def _run_pipeline_to_load(db, sid, content, entities, edges):
    from agents_kg.stages.parse import run as run_parse
    from agents_kg.stages.chunk import run as run_chunk
    from agents_kg.stages.extract import _make_edge_id

    source = db.get_source(sid)
    _mock_fetch(db, source, content)

    source = db.get_source(sid)
    run_parse(db, source)

    source = db.get_source(sid)
    run_chunk(db, source)

    chunks = db.get_chunks(sid)
    for c in chunks:
        emb = struct.pack("3f", 0.1, 0.2, 0.3)
        db.update_chunk_embedding(c["id"], emb, "mock-model")
    db.update_source(sid, stage="extract", status="processing")

    source = db.get_source(sid)
    chunk_id = chunks[0]["id"] if chunks else None
    for ent in entities:
        db.add_entity(
            entity_id=ent["entity_id"], name=ent["name"],
            entity_type=ent["type"], kind=ent.get("kind"),
            description=ent.get("description"),
            source_id=sid, chunk_id=chunk_id,
        )
    for e in edges:
        eid = _make_edge_id(e["src"], e["tgt"], e["type"])
        db.add_edge(eid, e["src"], e["tgt"], e["type"],
                    confidence=e.get("conf", 0.9),
                    source_id=sid, chunk_id=chunk_id,
                    valid_from=e.get("valid_from"),
                    valid_to=e.get("valid_to"))

    for ent in db.get_entities_by_status("pending_review"):
        db.approve_entity(ent["id"])
    for edge in db.get_edges_by_status("pending_review"):
        db.approve_edge(edge["id"])

    db.update_source(sid, stage="load", status="processing")
    return db.get_source(sid)


# ---------------------------------------------------------------------------
# 1. NEO4J END-TO-END TESTS
# ---------------------------------------------------------------------------


class TestNeo4jEndToEnd:
    """Full pipeline -> Neo4j path: ingest, process, load, query back."""

    CONTENT = """\
# Ethereum Protocol

Ethereum Foundation develops the Ethereum blockchain.
Vitalik Buterin co-founded the project in 2015.

## Architecture

Ethereum uses a proof-of-stake consensus mechanism.
"""

    ENTITIES = [
        {"entity_id": "test:org-eth-foundation", "name": "Ethereum Foundation",
         "type": "Organization", "kind": "foundation"},
        {"entity_id": "test:protocol-ethereum", "name": "Ethereum",
         "type": "Protocol", "kind": "blockchain"},
        {"entity_id": "test:person-vitalik", "name": "Vitalik Buterin",
         "type": "Person", "kind": "founder"},
    ]

    EDGES = [
        {"src": "test:org-eth-foundation", "tgt": "test:protocol-ethereum",
         "type": "DEVELOPS", "conf": 0.95},
        {"src": "test:person-vitalik", "tgt": "test:org-eth-foundation",
         "type": "MEMBER_OF", "conf": 0.85},
    ]

    def test_full_pipeline_to_neo4j(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://e2e/ethereum-doc", submitter_email="e2e@test.com")
        source = _run_pipeline_to_load(db, sid, self.CONTENT, self.ENTITIES, self.EDGES)
        result = run_load(db, source, neo4j_driver=clean_neo4j)
        assert result is True

        with clean_neo4j.session() as s:
            for ent in self.ENTITIES:
                node = s.run(
                    "MATCH (n {entity_id: $eid}) RETURN n",
                    {"eid": ent["entity_id"]}
                ).single()
                assert node is not None, f"Missing entity: {ent['entity_id']}"
                assert node["n"]["name"] == ent["name"]
                assert node["n"]["type"] == ent["type"]

    def test_entity_labels_match_type(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://e2e/label-check")
        source = _run_pipeline_to_load(db, sid, self.CONTENT, self.ENTITIES, self.EDGES)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            org = s.run(
                "MATCH (n:Organization {entity_id: 'test:org-eth-foundation'}) RETURN n"
            ).single()
            assert org is not None, "Organization label not set"

            protocol = s.run(
                "MATCH (n:Protocol {entity_id: 'test:protocol-ethereum'}) RETURN n"
            ).single()
            assert protocol is not None, "Protocol label not set"

            person = s.run(
                "MATCH (n:Person {entity_id: 'test:person-vitalik'}) RETURN n"
            ).single()
            assert person is not None, "Person label not set"

    def test_relationship_types_match_edge_type(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://e2e/edge-types")
        source = _run_pipeline_to_load(db, sid, self.CONTENT, self.ENTITIES, self.EDGES)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            develops = s.run(
                "MATCH (a {entity_id: 'test:org-eth-foundation'})"
                "-[r:DEVELOPS]->"
                "(b {entity_id: 'test:protocol-ethereum'}) RETURN r"
            ).single()
            assert develops is not None, "DEVELOPS edge not found"
            assert develops["r"]["confidence"] == 0.95

            member_of = s.run(
                "MATCH (a {entity_id: 'test:person-vitalik'})"
                "-[r:MEMBER_OF]->"
                "(b {entity_id: 'test:org-eth-foundation'}) RETURN r"
            ).single()
            assert member_of is not None, "MEMBER_OF edge not found"

    def test_source_node_created(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://e2e/source-node", submitter_email="alice@test.com")
        source = _run_pipeline_to_load(db, sid, self.CONTENT, self.ENTITIES[:1], [])
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            src = s.run(
                "MATCH (s:Source {uri: 'test://e2e/source-node'}) RETURN s"
            ).single()
            assert src is not None, "Source node not created"
            assert src["s"]["submitter_email"] == "alice@test.com"

    def test_from_source_edges(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://e2e/from-source-edges")
        source = _run_pipeline_to_load(db, sid, self.CONTENT, self.ENTITIES, self.EDGES)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            for ent in self.ENTITIES:
                result = s.run(
                    "MATCH (n {entity_id: $eid})-[:FROM_SOURCE]->(s:Source {uri: $uri}) RETURN s",
                    {"eid": ent["entity_id"], "uri": "test://e2e/from-source-edges"}
                ).single()
                assert result is not None, f"FROM_SOURCE edge missing for {ent['entity_id']}"

    def test_chunk_nodes_created(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://e2e/chunks")
        source = _run_pipeline_to_load(db, sid, self.CONTENT, self.ENTITIES[:1], [])
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            result = s.run(
                "MATCH (n {entity_id: 'test:org-eth-foundation'})"
                "-[:EXTRACTED_FROM]->(c:Chunk) RETURN c"
            ).single()
            assert result is not None, "Chunk node or EXTRACTED_FROM edge not found"


# ---------------------------------------------------------------------------
# 2. GRAPH TRAVERSAL TESTS
# ---------------------------------------------------------------------------


class TestGraphTraversal:
    """Multi-hop queries, path finding, aggregation in Neo4j."""

    @pytest.fixture(autouse=True)
    def _setup_graph(self, clean_neo4j):
        self.driver = clean_neo4j
        with self.driver.session() as s:
            s.run("""
                CREATE (google:Entity:Organization {entity_id: 'test:org-google',
                    name: 'Google', type: 'Organization', kind: 'company'})
                CREATE (anthropic:Entity:Organization {entity_id: 'test:org-anthropic',
                    name: 'Anthropic', type: 'Organization', kind: 'company'})
                CREATE (a2a:Entity:Protocol {entity_id: 'test:protocol-a2a',
                    name: 'A2A', type: 'Protocol', kind: 'spec'})
                CREATE (mcp:Entity:Protocol {entity_id: 'test:protocol-mcp',
                    name: 'MCP', type: 'Protocol', kind: 'spec', wikidata_id: 'Q99999'})
                CREATE (adk:Entity:Project {entity_id: 'test:project-adk',
                    name: 'ADK', type: 'Project', kind: 'framework'})
                CREATE (claude:Entity:Project {entity_id: 'test:project-claude',
                    name: 'Claude', type: 'Project', kind: 'platform'})
                CREATE (tooluse:Entity:Capability {entity_id: 'test:cap-tool-use',
                    name: 'Tool Use', type: 'Capability'})

                CREATE (google)-[:DEVELOPS]->(a2a)
                CREATE (google)-[:DEVELOPS]->(adk)
                CREATE (anthropic)-[:DEVELOPS]->(mcp)
                CREATE (anthropic)-[:DEVELOPS]->(claude)
                CREATE (adk)-[:IMPLEMENTS]->(a2a)
                CREATE (a2a)-[:COMPLEMENTS]->(mcp)
                CREATE (claude)-[:ADDRESSES]->(tooluse)
            """)

    def test_multi_hop_org_to_protocol(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (org:Organization {entity_id: 'test:org-google'})
                      -[:DEVELOPS]->(proj:Project)
                      -[:IMPLEMENTS]->(proto:Protocol)
                RETURN org.name AS org, proj.name AS project, proto.name AS protocol
            """).data()
            assert len(result) == 1
            assert result[0]["org"] == "Google"
            assert result[0]["project"] == "ADK"
            assert result[0]["protocol"] == "A2A"

    def test_multi_hop_via_complements(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (org:Organization {entity_id: 'test:org-google'})
                      -[:DEVELOPS]->(p1:Protocol)
                      -[:COMPLEMENTS]->(p2:Protocol)
                      <-[:DEVELOPS]-(org2:Organization)
                RETURN org.name AS org1, p1.name AS proto1,
                       p2.name AS proto2, org2.name AS org2
            """).data()
            assert len(result) == 1
            assert result[0]["org1"] == "Google"
            assert result[0]["proto1"] == "A2A"
            assert result[0]["proto2"] == "MCP"
            assert result[0]["org2"] == "Anthropic"

    def test_shortest_path(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH p = shortestPath(
                    (a {entity_id: 'test:org-google'})-[*]-(b {entity_id: 'test:project-claude'})
                )
                RETURN length(p) AS hops, [n IN nodes(p) | n.name] AS names
            """).single()
            assert result is not None
            assert result["hops"] >= 2

    def test_aggregation_count_by_type(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.entity_id STARTS WITH 'test:'
                RETURN n.type AS type, count(*) AS count
                ORDER BY count DESC
            """).data()
            counts = {r["type"]: r["count"] for r in result}
            assert counts.get("Organization", 0) == 2
            assert counts.get("Protocol", 0) == 2
            assert counts.get("Project", 0) == 2
            assert counts.get("Capability", 0) == 1

    def test_most_connected_entity(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n:Entity)-[r]-()
                WHERE n.entity_id STARTS WITH 'test:'
                RETURN n.entity_id AS id, n.name AS name, count(r) AS connections
                ORDER BY connections DESC
                LIMIT 1
            """).single()
            assert result is not None
            assert result["connections"] >= 2

    def test_variable_length_path(self):
        with self.driver.session() as s:
            result = s.run("""
                MATCH (start {entity_id: 'test:org-google'})
                      -[*1..3]->(end)
                WHERE end.entity_id STARTS WITH 'test:'
                RETURN DISTINCT end.name AS name
                ORDER BY name
            """).data()
            names = [r["name"] for r in result]
            assert "A2A" in names
            assert "ADK" in names
            assert "MCP" in names


# ---------------------------------------------------------------------------
# 3. SCHEMA VERIFICATION
# ---------------------------------------------------------------------------


class TestSchemaVerification:
    """Test constraint enforcement, indexes, and schema idempotency in Neo4j."""

    def test_schema_apply(self, clean_neo4j):
        from agents_kg.schema import apply_schema, CONSTRAINTS, INDEXES
        results = apply_schema(clean_neo4j)
        assert results["constraints"] == len(CONSTRAINTS)
        assert results["indexes"] == len(INDEXES)
        assert results["errors"] == []

    def test_schema_idempotent(self, clean_neo4j):
        from agents_kg.schema import apply_schema
        r1 = apply_schema(clean_neo4j)
        r2 = apply_schema(clean_neo4j)
        assert r1["errors"] == []
        assert r2["errors"] == []

    def test_entity_id_uniqueness_constraint(self, clean_neo4j):
        from agents_kg.schema import apply_schema
        apply_schema(clean_neo4j)

        with clean_neo4j.session() as s:
            s.run("CREATE (n:Entity {entity_id: 'test:unique-check', name: 'First'})")

            from neo4j.exceptions import ClientError
            with pytest.raises(ClientError):
                s.run("CREATE (n:Entity {entity_id: 'test:unique-check', name: 'Duplicate'})")

    def test_indexes_exist(self, clean_neo4j):
        from agents_kg.schema import apply_schema
        apply_schema(clean_neo4j)

        with clean_neo4j.session() as s:
            indexes = s.run("SHOW INDEXES").data()
            index_props = set()
            for idx in indexes:
                props = idx.get("properties") or []
                for prop in props:
                    index_props.add(prop)

            assert "entity_id" in index_props
            assert "type" in index_props
            assert "wikidata_id" in index_props

    def test_constraint_on_source_uri(self, clean_neo4j):
        from agents_kg.schema import apply_schema
        apply_schema(clean_neo4j)

        with clean_neo4j.session() as s:
            s.run("CREATE (s:Source {uri: 'test://schema/unique-uri'})")
            from neo4j.exceptions import ClientError
            with pytest.raises(ClientError):
                s.run("CREATE (s:Source {uri: 'test://schema/unique-uri'})")

    def test_explain_uses_index(self, clean_neo4j):
        from agents_kg.schema import apply_schema
        apply_schema(clean_neo4j)

        with clean_neo4j.session() as s:
            s.run("CREATE (n:Entity {entity_id: 'test:explain-idx', name: 'Test', type: 'Project'})")
            result = s.run(
                "EXPLAIN MATCH (n:Entity {entity_id: 'test:explain-idx'}) RETURN n"
            )
            plan = result.consume().plan
            plan_str = str(plan)
            assert plan is not None


# ---------------------------------------------------------------------------
# 4. TEMPORAL MODEL
# ---------------------------------------------------------------------------


class TestTemporalModel:
    """Event nodes, PARTICIPATED_IN edges, and temporal queries."""

    @pytest.fixture(autouse=True)
    def _setup_temporal_data(self, clean_neo4j):
        self.driver = clean_neo4j
        with self.driver.session() as s:
            s.run("""
                CREATE (google:Entity:Organization {entity_id: 'test:org-google',
                    name: 'Google', type: 'Organization'})
                CREATE (anthropic:Entity:Organization {entity_id: 'test:org-anthropic',
                    name: 'Anthropic', type: 'Organization'})
                CREATE (openai:Entity:Organization {entity_id: 'test:org-openai',
                    name: 'OpenAI', type: 'Organization'})
            """)

    def test_load_events_from_yaml(self, tmp_path):
        from agents_kg.temporal import load_events_from_yaml

        event_file = tmp_path / "launch.yaml"
        event_file.write_text(yaml.dump({
            "event_id": "test-launch-2025-04-01",
            "title": "Test Product Launch",
            "event_type": "launch",
            "date": "2025-04-01",
            "description": "A test launch event",
            "participants": [
                {"entity_id": "test:org-google", "role": "organizer"},
                {"entity_id": "test:org-anthropic", "role": "attendee"},
            ]
        }))

        result = load_events_from_yaml(self.driver, str(tmp_path))
        assert result["events"] == 1
        assert result["participations"] == 2

        with self.driver.session() as s:
            event = s.run(
                "MATCH (e:Event {event_id: 'test-launch-2025-04-01'}) RETURN e"
            ).single()
            assert event is not None
            assert event["e"]["title"] == "Test Product Launch"

    def test_participation_edges(self, tmp_path):
        from agents_kg.temporal import load_events_from_yaml

        event_file = tmp_path / "conf.yaml"
        event_file.write_text(yaml.dump({
            "event_id": "test-conf-2025-06-15",
            "title": "Test Conference",
            "event_type": "conference",
            "date": "2025-06-15",
            "participants": [
                {"entity_id": "test:org-google", "role": "sponsor"},
                {"entity_id": "test:org-anthropic", "role": "speaker"},
                {"entity_id": "test:org-openai", "role": "attendee"},
            ]
        }))

        load_events_from_yaml(self.driver, str(tmp_path))

        with self.driver.session() as s:
            result = s.run("""
                MATCH (entity:Entity)-[r:PARTICIPATED_IN]->(e:Event {event_id: 'test-conf-2025-06-15'})
                RETURN entity.name AS name, r.role AS role
                ORDER BY name
            """).data()
            roles = {r["name"]: r["role"] for r in result}
            assert roles.get("Anthropic") == "speaker"
            assert roles.get("Google") == "sponsor"
            assert roles.get("OpenAI") == "attendee"

    def test_temporal_date_range_query(self, tmp_path):
        from agents_kg.temporal import load_events_from_yaml

        for month in ["03", "06", "09"]:
            ef = tmp_path / f"event-{month}.yaml"
            ef.write_text(yaml.dump({
                "event_id": f"test-event-2025-{month}-01",
                "title": f"Event {month}",
                "event_type": "release",
                "date": f"2025-{month}-01",
            }))

        load_events_from_yaml(self.driver, str(tmp_path))

        with self.driver.session() as s:
            result = s.run("""
                MATCH (e:Event)
                WHERE e.event_id STARTS WITH 'test-'
                  AND e.date >= date('2025-04-01')
                  AND e.date <= date('2025-08-31')
                RETURN e.title AS title, e.date AS date
                ORDER BY e.date
            """).data()
            assert len(result) == 1
            assert result[0]["title"] == "Event 06"

    def test_entity_timeline(self, tmp_path):
        from agents_kg.temporal import load_events_from_yaml

        for i, (title, date) in enumerate([
            ("Product Alpha Launch", "2025-01-15"),
            ("Series B Funding", "2025-03-20"),
            ("Partnership Announced", "2025-07-10"),
        ]):
            ef = tmp_path / f"timeline-{i}.yaml"
            ef.write_text(yaml.dump({
                "event_id": f"test-tl-{i}-{date}",
                "title": title,
                "event_type": "milestone",
                "date": date,
                "participants": [{"entity_id": "test:org-google", "role": "subject"}],
            }))

        load_events_from_yaml(self.driver, str(tmp_path))

        with self.driver.session() as s:
            result = s.run("""
                MATCH (org:Entity {entity_id: 'test:org-google'})
                      -[:PARTICIPATED_IN]->(e:Event)
                WHERE e.event_id STARTS WITH 'test-tl-'
                RETURN e.title AS title, e.date AS date
                ORDER BY e.date
            """).data()
            assert len(result) == 3
            assert result[0]["title"] == "Product Alpha Launch"
            assert result[2]["title"] == "Partnership Announced"

    def test_temporal_constraints(self, clean_neo4j):
        from agents_kg.temporal import create_temporal_constraints
        create_temporal_constraints(clean_neo4j)
        create_temporal_constraints(clean_neo4j)

        with clean_neo4j.session() as s:
            constraints = s.run("SHOW CONSTRAINTS").data()
            names = [c.get("name", "") for c in constraints]
            assert any("event_id" in n for n in names)

    def test_valid_from_valid_to_on_edges(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://temporal/valid-range")
        entities = [
            {"entity_id": "test:org-alpha", "name": "Alpha", "type": "Organization", "kind": "company"},
            {"entity_id": "test:org-beta", "name": "Beta", "type": "Organization", "kind": "company"},
        ]
        edges = [
            {"src": "test:org-alpha", "tgt": "test:org-beta",
             "type": "PART_OF", "conf": 0.9,
             "valid_from": "2023-01-01", "valid_to": "2025-12-31"},
        ]
        source = _run_pipeline_to_load(
            db, sid,
            "# Alpha Beta\n\nAlpha is part of Beta.\n",
            entities, edges,
        )
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (a {entity_id: 'test:org-alpha'})-[r:PART_OF]->(b {entity_id: 'test:org-beta'})
                RETURN r.valid_from AS vf, r.valid_to AS vt
            """).single()
            assert result is not None
            assert result["vf"] == "2023-01-01"
            assert result["vt"] == "2025-12-31"


# ---------------------------------------------------------------------------
# 5. WIKIDATA INTEGRATION WITH NEO4J
# ---------------------------------------------------------------------------


class TestWikidataIntegration:
    """Test wikidata entity loading and cross-referencing in Neo4j."""

    def test_load_seed_entities(self, clean_neo4j):
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities

        seeds = get_seed_entities()
        load_wikidata_entities(clean_neo4j, seeds)

        with clean_neo4j.session() as s:
            count = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            assert count >= len(seeds)

            types = s.run(
                "MATCH (n:Entity) RETURN DISTINCT n.type AS t"
            ).data()
            type_set = {r["t"] for r in types}
            assert "Organization" in type_set
            assert "Protocol" in type_set
            assert "Project" in type_set
            assert "Capability" in type_set
            assert "Person" in type_set

    def test_seed_entities_have_correct_labels(self, clean_neo4j):
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities

        load_wikidata_entities(clean_neo4j, get_seed_entities())

        with clean_neo4j.session() as s:
            orgs = s.run("MATCH (n:Organization) RETURN count(n) AS c").single()["c"]
            assert orgs > 0
            protocols = s.run("MATCH (n:Protocol) RETURN count(n) AS c").single()["c"]
            assert protocols > 0
            projects = s.run("MATCH (n:Project) RETURN count(n) AS c").single()["c"]
            assert projects > 0

    def test_crossref_applies_wikidata_ids(self, clean_neo4j):
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities
        from agents_kg.wikidata_crossref import apply_crossref

        load_wikidata_entities(clean_neo4j, get_seed_entities())
        result = apply_crossref(neo4j_driver=clean_neo4j)
        assert result["applied"] > 0

        with clean_neo4j.session() as s:
            entities_with_wikidata = s.run("""
                MATCH (n:Entity)
                WHERE n.wikidata_id IS NOT NULL
                RETURN n.entity_id AS eid, n.wikidata_id AS qid
            """).data()
            assert len(entities_with_wikidata) > 0
            for r in entities_with_wikidata:
                assert r["qid"].startswith("Q")

    def test_yaml_and_wikidata_entities_coexist(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities

        load_wikidata_entities(clean_neo4j, get_seed_entities()[:5])

        sid = db.add_source("test://wikidata/coexist")
        yaml_entities = [
            {"entity_id": "test:custom-proj", "name": "Custom Project",
             "type": "Project", "kind": "custom"},
        ]
        source = _run_pipeline_to_load(
            db, sid,
            "# Custom\n\nA custom project.\n",
            yaml_entities, [],
        )
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            custom = s.run(
                "MATCH (n {entity_id: 'test:custom-proj'}) RETURN n"
            ).single()
            assert custom is not None

            seeds = s.run(
                "MATCH (n:Entity) WHERE n.source_type = 'wikidata' RETURN count(n) AS c"
            ).single()["c"]
            assert seeds > 0

    def test_seed_idempotent(self, clean_neo4j):
        from agents_kg.seed import get_seed_entities
        from agents_kg.wikidata import load_wikidata_entities

        seeds = get_seed_entities()
        load_wikidata_entities(clean_neo4j, seeds)

        with clean_neo4j.session() as s:
            count1 = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]

        load_wikidata_entities(clean_neo4j, seeds)

        with clean_neo4j.session() as s:
            count2 = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]

        assert count1 == count2


# ---------------------------------------------------------------------------
# 6. CLEANUP AND ISOLATION
# ---------------------------------------------------------------------------


class TestCleanupAndIsolation:
    """Verify test data isolation using test: prefix and cleanup fixture."""

    def test_cleanup_removes_test_data(self, clean_neo4j):
        with clean_neo4j.session() as s:
            s.run("CREATE (n:Entity {entity_id: 'test:cleanup-check', name: 'Temp'})")
            count = s.run(
                "MATCH (n {entity_id: 'test:cleanup-check'}) RETURN count(n) AS c"
            ).single()["c"]
            assert count == 1

    def test_test_data_isolated(self, clean_neo4j):
        with clean_neo4j.session() as s:
            leftover = s.run(
                "MATCH (n) WHERE n.entity_id IS NOT NULL AND n.entity_id STARTS WITH 'test:' "
                "RETURN count(n) AS c"
            ).single()["c"]
            assert leftover == 0

    def test_concurrent_test_sources_isolated(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        for i in range(3):
            sid = db.add_source(f"test://isolation/source-{i}")
            entities = [
                {"entity_id": f"test:iso-ent-{i}", "name": f"Entity {i}",
                 "type": "Project", "kind": "test"},
            ]
            source = _run_pipeline_to_load(
                db, sid, f"# Source {i}\n\nContent for source {i}.\n", entities, [],
            )
            run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.entity_id STARTS WITH 'test:iso-ent-'
                RETURN count(n) AS c
            """).single()["c"]
            assert result == 3

            for i in range(3):
                src = s.run(
                    "MATCH (s:Source {uri: $uri}) RETURN s",
                    {"uri": f"test://isolation/source-{i}"}
                ).single()
                assert src is not None


# ---------------------------------------------------------------------------
# 7. IDEMPOTENCY
# ---------------------------------------------------------------------------


class TestNeo4jIdempotency:
    """Verify MERGE semantics prevent duplicates."""

    def test_entity_merge_no_duplicates(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://idem/entity")
        entities = [
            {"entity_id": "test:idem-org", "name": "Idem Corp",
             "type": "Organization", "kind": "company"},
        ]
        source = _run_pipeline_to_load(
            db, sid, "# Idem\n\nIdem Corp content.\n", entities, [],
        )
        run_load(db, source, neo4j_driver=clean_neo4j)

        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            count = s.run(
                "MATCH (n {entity_id: 'test:idem-org'}) RETURN count(n) AS c"
            ).single()["c"]
            assert count == 1

    def test_edge_merge_no_duplicates(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://idem/edge")
        entities = [
            {"entity_id": "test:idem-a", "name": "A", "type": "Organization", "kind": "company"},
            {"entity_id": "test:idem-b", "name": "B", "type": "Project", "kind": "framework"},
        ]
        edges = [{"src": "test:idem-a", "tgt": "test:idem-b", "type": "DEVELOPS", "conf": 0.9}]
        source = _run_pipeline_to_load(
            db, sid, "# A and B\n\nA develops B.\n", entities, edges,
        )
        run_load(db, source, neo4j_driver=clean_neo4j)

        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            count = s.run("""
                MATCH (a {entity_id: 'test:idem-a'})-[r:DEVELOPS]->(b {entity_id: 'test:idem-b'})
                RETURN count(r) AS c
            """).single()["c"]
            assert count == 1

    def test_source_node_merge_no_duplicates(self, db, clean_neo4j):
        from agents_kg.stages.load import run as run_load

        sid = db.add_source("test://idem/source-node")
        entities = [{"entity_id": "test:idem-src", "name": "S", "type": "Project", "kind": "test"}]
        source = _run_pipeline_to_load(
            db, sid, "# Source\n\nContent.\n", entities, [],
        )
        run_load(db, source, neo4j_driver=clean_neo4j)

        db.update_source(sid, status="processing", stage="load")
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=clean_neo4j)

        with clean_neo4j.session() as s:
            count = s.run(
                "MATCH (s:Source {uri: 'test://idem/source-node'}) RETURN count(s) AS c"
            ).single()["c"]
            assert count == 1

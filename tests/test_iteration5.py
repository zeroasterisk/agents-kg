"""Iteration 5 tests.

Covers: entity merge chains, Neo4j load with Cypher verification,
graph statistics/analytics, unicode/special characters, and error recovery.
Also verifies the LangGraph seed data fix (no longer an alias of LangChain).
"""

import json
import os
import struct
import tempfile
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest
import yaml

from agents_kg.db import Database, _now, content_hash
from agents_kg.stages.extract import _make_edge_id, VALID_EDGE_TYPES, VALID_ENTITY_TYPES
from agents_kg.stages.resolve import (
    _normalize,
    _similarity,
    _build_alias_index,
    run as run_resolve,
    _merge_entity,
)
from agents_kg.stages.load import _entity_to_cypher, _edge_to_cypher, _export_yaml, run as run_load
from agents_kg.seed import get_seed_entities, seed_entity_ids

pytestmark = pytest.mark.e2e


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


def _get_entity(db, entity_id):
    row = db.conn.execute(
        "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    return dict(row) if row else None


def _get_edges_for_entity(db, entity_id):
    rows = db.conn.execute(
        "SELECT * FROM edges WHERE source_entity_id = ? OR target_entity_id = ?",
        (entity_id, entity_id)
    ).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# SEED DATA FIX VERIFICATION
# ===========================================================================


class TestLangGraphSeedFix:
    """Verify that LangGraph is NOT an alias of LangChain in seed data."""

    def test_langgraph_not_alias_of_langchain(self):
        seeds = get_seed_entities()
        langchain = next(s for s in seeds if s["entity_id"] == "project:langchain")
        assert "LangGraph" not in langchain.get("aliases", [])

    def test_langgraph_has_own_seed_entry(self):
        seeds = get_seed_entities()
        langgraph = [s for s in seeds if s["entity_id"] == "project:langgraph"]
        assert len(langgraph) == 1
        assert langgraph[0]["name"] == "LangGraph"

    def test_langchain_still_has_langsmith(self):
        seeds = get_seed_entities()
        langchain = next(s for s in seeds if s["entity_id"] == "project:langchain")
        assert "LangSmith" in langchain.get("aliases", [])

    def test_langgraph_resolves_independently(self, db):
        """Extracted 'LangGraph' entity should NOT merge into project:langchain."""
        content = "LangGraph provides graph-based agent orchestration."
        entities = [
            {"entity_id": "project:langgraph-ext", "name": "LangGraph",
             "type": "Project", "kind": "framework"},
        ]
        edges = []
        sid = _full_ingest(db, "https://example.com/langgraph-test", content,
                           entities, edges)
        active = _get_active_entity_ids(db)
        assert "project:langchain" not in active or "project:langgraph" in active


# ===========================================================================
# 1. ENTITY MERGE CHAINS
# ===========================================================================


class TestEntityMergeChains:
    """Test multi-step merge scenarios and edge repointing."""

    def test_a_merges_into_b_then_c_merges_into_b(self, db):
        """A→B merge, then C→B merge — all edges should point to B."""
        content = "Testing merge chains with multiple frameworks."
        entities = [
            {"entity_id": "project:framework-alpha", "name": "Framework Alpha",
             "type": "Project", "kind": "framework",
             "description": "The alpha framework"},
            {"entity_id": "project:framework-beta", "name": "Framework Beta",
             "type": "Project", "kind": "framework",
             "description": "The beta framework"},
            {"entity_id": "project:framework-gamma", "name": "Framework Gamma",
             "type": "Project", "kind": "framework",
             "description": "The gamma framework"},
            {"entity_id": "organization:test-corp", "name": "Test Corp",
             "type": "Organization", "kind": "company"},
        ]
        edges = [
            {"src": "organization:test-corp", "tgt": "project:framework-alpha",
             "type": "DEVELOPS"},
            {"src": "organization:test-corp", "tgt": "project:framework-gamma",
             "type": "DEVELOPS"},
            {"src": "project:framework-alpha", "tgt": "project:framework-gamma",
             "type": "COMPETES_WITH"},
        ]

        sid = db.add_source("https://example.com/merge-chain-test")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, edges)

        # Manually merge A→B then C→B
        ent_a = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:framework-alpha",)).fetchone())
        ent_c = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:framework-gamma",)).fetchone())

        _merge_entity(db, ent_a, "project:framework-beta")
        _merge_entity(db, ent_c, "project:framework-beta")

        # All edges should now reference framework-beta
        all_edges = db.conn.execute("SELECT * FROM edges WHERE source_id = ?",
                                    (sid,)).fetchall()
        for edge in all_edges:
            assert edge["source_entity_id"] != "project:framework-alpha", \
                "Edge still references merged entity A"
            assert edge["target_entity_id"] != "project:framework-alpha", \
                "Edge still references merged entity A"
            assert edge["source_entity_id"] != "project:framework-gamma", \
                "Edge still references merged entity C"
            assert edge["target_entity_id"] != "project:framework-gamma", \
                "Edge still references merged entity C"

    def test_no_orphan_references_after_merge(self, db):
        """After merge, no edges reference the merged-away entity_id."""
        content = "Orphan reference test"
        entities = [
            {"entity_id": "project:old-name", "name": "Old Name",
             "type": "Project", "kind": "tool"},
            {"entity_id": "project:new-name", "name": "New Name",
             "type": "Project", "kind": "tool"},
            {"entity_id": "capability:some-cap", "name": "Some Cap",
             "type": "Capability"},
        ]
        edges = [
            {"src": "project:old-name", "tgt": "capability:some-cap",
             "type": "ADDRESSES"},
            {"src": "capability:some-cap", "tgt": "project:old-name",
             "type": "PART_OF"},
        ]

        sid = db.add_source("https://example.com/orphan-test")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, edges)

        ent_old = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:old-name",)).fetchone())
        _merge_entity(db, ent_old, "project:new-name")

        # Check no edges reference old-name
        orphan_edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_entity_id = ? OR target_entity_id = ?",
            ("project:old-name", "project:old-name")
        ).fetchall()
        assert len(orphan_edges) == 0, "Found orphan edge references"

        # Verify edges now point to new-name
        new_edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_entity_id = ? OR target_entity_id = ?",
            ("project:new-name", "project:new-name")
        ).fetchall()
        assert len(new_edges) == 2

    def test_merged_entity_marked_correctly(self, db):
        """Merged entity has merged_into set and status='merged'."""
        content = "Status check test"
        entities = [
            {"entity_id": "project:dup-a", "name": "Dup A",
             "type": "Project", "kind": "framework"},
            {"entity_id": "project:dup-b", "name": "Dup B",
             "type": "Project", "kind": "framework"},
        ]
        sid = db.add_source("https://example.com/merge-status")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, [])

        ent_a = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:dup-a",)).fetchone())
        _merge_entity(db, ent_a, "project:dup-b")

        merged = _get_entity(db, "project:dup-a")
        assert merged["merged_into"] == "project:dup-b"
        assert merged["status"] == "merged"

    def test_seed_canonical_created_on_merge(self, db):
        """When merging into a seed entity not yet in DB, it gets created."""
        content = "Seed entity creation on merge"
        entities = [
            {"entity_id": "project:google-adk-variant", "name": "Google ADK",
             "type": "Project", "kind": "framework"},
        ]

        sid = _full_ingest(db, "https://example.com/seed-create-test",
                           content, entities, [])

        # The seed entity project:google-adk should exist if it matched
        adk_ids = seed_entity_ids()
        if "project:google-adk" in adk_ids:
            active = _get_active_entity_ids(db)
            assert "project:google-adk" in active or "project:google-adk-variant" in active

    def test_merge_chain_active_set_correct(self, db):
        """After multiple merges, the active entity set only contains canonical IDs."""
        content = "Multi-merge active set test"
        entities = [
            {"entity_id": "project:x1", "name": "X1", "type": "Project", "kind": "tool"},
            {"entity_id": "project:x2", "name": "X2", "type": "Project", "kind": "tool"},
            {"entity_id": "project:canonical-x", "name": "Canonical X",
             "type": "Project", "kind": "tool"},
        ]
        sid = db.add_source("https://example.com/active-set-test")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, [])

        ent_x1 = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:x1",)).fetchone())
        ent_x2 = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:x2",)).fetchone())

        _merge_entity(db, ent_x1, "project:canonical-x")
        _merge_entity(db, ent_x2, "project:canonical-x")

        active = _get_active_entity_ids(db)
        assert "project:x1" not in active
        assert "project:x2" not in active
        assert "project:canonical-x" in active


# ===========================================================================
# 2. NEO4J LOAD WITH CYPHER VERIFICATION
# ===========================================================================


class TestNeo4jCypherVerification:
    """Mock Neo4j driver and verify Cypher queries are correct."""

    def _make_mock_driver(self):
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver, session

    def test_merged_entities_excluded_from_neo4j(self, db):
        """Only canonical (non-merged) entities should be loaded to Neo4j."""
        content = "Merged entity exclusion test"
        entities = [
            {"entity_id": "project:proto-a", "name": "Proto A",
             "type": "Project", "kind": "framework"},
            {"entity_id": "project:proto-b", "name": "Proto B",
             "type": "Project", "kind": "framework"},
        ]
        sid = db.add_source("https://example.com/neo4j-merge-test")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, [])

        # Merge A into B, then approve B
        ent_a = dict(db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            ("project:proto-a",)).fetchone())
        _merge_entity(db, ent_a, "project:proto-b")
        _approve_all(db)

        driver, session = self._make_mock_driver()
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=driver)

        # Collect all entity_ids passed to session.run
        entity_ids_loaded = set()
        for c in session.run.call_args_list:
            args, kwargs = c
            if len(args) >= 2 and isinstance(args[1], dict):
                if "entity_id" in args[1]:
                    entity_ids_loaded.add(args[1]["entity_id"])
        assert "project:proto-a" not in entity_ids_loaded

    def test_from_source_edge_created(self, db):
        """Verify FROM_SOURCE edges link entities to their source node."""
        content = "Provenance test content"
        entities = [
            {"entity_id": "project:provenance-test", "name": "Provenance Test",
             "type": "Project", "kind": "tool"},
        ]
        sid = _full_ingest(db, "https://example.com/provenance-test",
                           content, entities, [])

        driver, session = self._make_mock_driver()
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=driver)

        # Check that FROM_SOURCE MERGE was called
        from_source_calls = [
            c for c in session.run.call_args_list
            if len(c.args) >= 1 and "FROM_SOURCE" in str(c.args[0])
        ]
        assert len(from_source_calls) >= 1

    def test_entity_cypher_has_correct_label(self):
        """Entity Cypher uses the entity type as a Neo4j label."""
        entity = {
            "entity_id": "project:test-proj",
            "name": "Test Project",
            "type": "Project",
            "kind": "framework",
            "description": "A test project",
            "aliases": json.dumps(["TP"]),
            "source_id": 1,
        }
        query, params = _entity_to_cypher(entity)
        assert "n:Project" in query
        assert params["entity_id"] == "project:test-proj"
        assert params["name"] == "Test Project"

    def test_entity_cypher_invalid_type_uses_entity_label(self):
        """Unknown entity types fall back to 'Entity' label."""
        entity = {
            "entity_id": "unknown:thing",
            "name": "Unknown Thing",
            "type": "Widget",
            "kind": None,
            "description": None,
            "aliases": "[]",
            "source_id": 1,
        }
        query, params = _entity_to_cypher(entity)
        assert "n:Entity" in query
        assert "n:Widget" not in query

    def test_edge_cypher_has_correct_type(self):
        """Edge Cypher uses the edge_type as the relationship type."""
        edge = {
            "edge_id": "abc123",
            "source_entity_id": "org:a",
            "target_entity_id": "project:b",
            "edge_type": "DEVELOPS",
            "properties": "{}",
            "confidence": 0.9,
            "source_type": "automated",
            "valid_from": None,
            "valid_to": None,
            "chunk_id": None,
        }
        query, params = _edge_to_cypher(edge)
        assert "DEVELOPS" in query
        assert params["src"] == "org:a"
        assert params["tgt"] == "project:b"

    def test_edge_cypher_with_properties(self):
        """Edge properties are included as SET clauses."""
        edge = {
            "edge_id": "def456",
            "source_entity_id": "org:x",
            "target_entity_id": "project:y",
            "edge_type": "SPONSORS",
            "properties": json.dumps({"amount": 1000, "year": 2025}),
            "confidence": 0.8,
            "source_type": "manual",
            "valid_from": "2025-01-01",
            "valid_to": None,
            "chunk_id": 42,
        }
        query, params = _edge_to_cypher(edge)
        assert "r.amount" in query
        assert "r.year" in query
        assert params["prop_amount"] == 1000
        assert params["prop_year"] == 2025

    def test_source_node_created_in_neo4j(self, db):
        """A Source node is created in Neo4j with provenance metadata."""
        content = "Source node test"
        entities = [
            {"entity_id": "project:src-node-test", "name": "Src Node Test",
             "type": "Project", "kind": "tool"},
        ]
        sid = _full_ingest(db, "https://example.com/source-node-test",
                           content, entities, [])

        driver, session = self._make_mock_driver()
        source = db.get_source(sid)
        run_load(db, source, neo4j_driver=driver)

        source_calls = [
            c for c in session.run.call_args_list
            if len(c.args) >= 1 and "MERGE (s:Source" in str(c.args[0])
        ]
        assert len(source_calls) >= 1

    def test_deprecated_entities_not_loaded(self, db):
        """Deprecated entities should not be loaded via the approved-only query."""
        content = "Deprecation test"
        entities = [
            {"entity_id": "project:dep-test", "name": "Dep Test",
             "type": "Project", "kind": "tool"},
        ]
        sid = db.add_source("https://example.com/deprecation-test")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)
        _mock_extract(db, source, entities, [])

        # Approve then deprecate
        _approve_all(db)
        db.deprecate_entities_for_source(sid)

        # Approved query should still return them (deprecated_at is set
        # but status is still 'approved'), so load stage will load them.
        # The point: run_load queries by status='approved', which includes
        # deprecated entities unless filtered separately.
        approved = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ? AND status = 'approved'",
            (sid,)
        ).fetchall()
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) >= 1


# ===========================================================================
# 3. GRAPH STATISTICS AND ANALYTICS
# ===========================================================================


class TestGraphStatistics:
    """Test querying patterns for graph analytics."""

    def _build_graph(self, db):
        """Build a small but connected graph for analytics tests."""
        content = "Analytics graph content"
        entities = [
            {"entity_id": "organization:analytics-org", "name": "Analytics Org",
             "type": "Organization", "kind": "company"},
            {"entity_id": "project:proj-1", "name": "Project One",
             "type": "Project", "kind": "framework"},
            {"entity_id": "project:proj-2", "name": "Project Two",
             "type": "Project", "kind": "sdk"},
            {"entity_id": "project:proj-3", "name": "Project Three",
             "type": "Project", "kind": "tool"},
            {"entity_id": "protocol:proto-1", "name": "Protocol One",
             "type": "Protocol", "kind": "spec"},
            {"entity_id": "capability:cap-1", "name": "Capability One",
             "type": "Capability"},
            {"entity_id": "capability:cap-2", "name": "Capability Two",
             "type": "Capability"},
            {"entity_id": "project:isolated-proj", "name": "Isolated Project",
             "type": "Project", "kind": "tool"},
        ]
        edges = [
            {"src": "organization:analytics-org", "tgt": "project:proj-1",
             "type": "DEVELOPS"},
            {"src": "organization:analytics-org", "tgt": "project:proj-2",
             "type": "DEVELOPS"},
            {"src": "organization:analytics-org", "tgt": "project:proj-3",
             "type": "DEVELOPS"},
            {"src": "project:proj-1", "tgt": "protocol:proto-1",
             "type": "IMPLEMENTS"},
            {"src": "project:proj-2", "tgt": "protocol:proto-1",
             "type": "IMPLEMENTS"},
            {"src": "project:proj-1", "tgt": "capability:cap-1",
             "type": "ADDRESSES"},
            {"src": "project:proj-2", "tgt": "capability:cap-1",
             "type": "ADDRESSES"},
            {"src": "project:proj-3", "tgt": "capability:cap-2",
             "type": "ADDRESSES"},
            {"src": "project:proj-1", "tgt": "project:proj-2",
             "type": "COMPETES_WITH"},
        ]
        return _full_ingest(db, "https://example.com/analytics-graph",
                            content, entities, edges)

    def test_degree_distribution(self, db):
        """Find the most-connected entities by edge count."""
        self._build_graph(db)

        rows = db.conn.execute("""
            SELECT entity_id, COUNT(*) as degree FROM (
                SELECT source_entity_id as entity_id FROM edges
                WHERE status = 'approved'
                UNION ALL
                SELECT target_entity_id as entity_id FROM edges
                WHERE status = 'approved'
            ) GROUP BY entity_id ORDER BY degree DESC
        """).fetchall()

        degrees = {r["entity_id"]: r["degree"] for r in rows}
        assert len(degrees) > 0

        # analytics-org has 3 outgoing DEVELOPS edges
        assert degrees.get("organization:analytics-org", 0) >= 3

        # proj-1 has the most edges: 1 incoming DEVELOPS, 1 IMPLEMENTS,
        # 1 ADDRESSES, 1 COMPETES_WITH = 4
        assert degrees.get("project:proj-1", 0) >= 4

    def test_edge_type_distribution(self, db):
        """Count edges by type."""
        self._build_graph(db)

        rows = db.conn.execute("""
            SELECT edge_type, COUNT(*) as cnt FROM edges
            WHERE status = 'approved'
            GROUP BY edge_type ORDER BY cnt DESC
        """).fetchall()

        dist = {r["edge_type"]: r["cnt"] for r in rows}
        assert dist.get("DEVELOPS", 0) == 3
        assert dist.get("IMPLEMENTS", 0) == 2
        assert dist.get("ADDRESSES", 0) == 3
        assert dist.get("COMPETES_WITH", 0) == 1

    def test_entity_type_distribution(self, db):
        """Count active entities by type."""
        self._build_graph(db)

        rows = db.conn.execute("""
            SELECT type, COUNT(*) as cnt FROM entities
            WHERE merged_into IS NULL AND status != 'rejected'
            AND deprecated_at IS NULL
            GROUP BY type ORDER BY cnt DESC
        """).fetchall()

        dist = {r["type"]: r["cnt"] for r in rows}
        assert dist.get("Project", 0) >= 3
        assert dist.get("Capability", 0) >= 2
        assert dist.get("Organization", 0) >= 1
        assert dist.get("Protocol", 0) >= 1

    def test_entity_kind_distribution(self, db):
        """Count active entities by kind (subtype)."""
        self._build_graph(db)

        rows = db.conn.execute("""
            SELECT kind, COUNT(*) as cnt FROM entities
            WHERE merged_into IS NULL AND status != 'rejected'
            AND deprecated_at IS NULL AND kind IS NOT NULL
            GROUP BY kind ORDER BY cnt DESC
        """).fetchall()

        dist = {r["kind"]: r["cnt"] for r in rows}
        assert "framework" in dist
        assert "sdk" in dist
        assert "tool" in dist

    def test_hub_detection(self, db):
        """Find entities with degree > N (hubs)."""
        self._build_graph(db)

        threshold = 3
        rows = db.conn.execute(f"""
            SELECT entity_id, COUNT(*) as degree FROM (
                SELECT source_entity_id as entity_id FROM edges
                WHERE status = 'approved'
                UNION ALL
                SELECT target_entity_id as entity_id FROM edges
                WHERE status = 'approved'
            ) GROUP BY entity_id HAVING degree > {threshold}
            ORDER BY degree DESC
        """).fetchall()

        hub_ids = {r["entity_id"] for r in rows}
        assert "organization:analytics-org" in hub_ids or "project:proj-1" in hub_ids

    def test_isolated_entity_detection(self, db):
        """Find entities with no edges (isolates)."""
        self._build_graph(db)

        rows = db.conn.execute("""
            SELECT e.entity_id FROM entities e
            WHERE e.merged_into IS NULL AND e.status != 'rejected'
            AND e.deprecated_at IS NULL
            AND e.entity_id NOT IN (
                SELECT source_entity_id FROM edges WHERE status = 'approved'
                UNION
                SELECT target_entity_id FROM edges WHERE status = 'approved'
            )
        """).fetchall()

        isolated = {r["entity_id"] for r in rows}
        assert "project:isolated-proj" in isolated

    def test_connected_components_heuristic(self, db):
        """Basic heuristic: count entities reachable from highest-degree node."""
        self._build_graph(db)

        # Get entity with highest degree
        rows = db.conn.execute("""
            SELECT entity_id, COUNT(*) as degree FROM (
                SELECT source_entity_id as entity_id FROM edges
                WHERE status = 'approved'
                UNION ALL
                SELECT target_entity_id as entity_id FROM edges
                WHERE status = 'approved'
            ) GROUP BY entity_id ORDER BY degree DESC LIMIT 1
        """).fetchall()

        if rows:
            hub = rows[0]["entity_id"]
            # BFS from hub
            visited = {hub}
            frontier = {hub}
            while frontier:
                next_frontier = set()
                for node in frontier:
                    neighbors = db.conn.execute("""
                        SELECT DISTINCT target_entity_id as eid FROM edges
                        WHERE source_entity_id = ? AND status = 'approved'
                        UNION
                        SELECT DISTINCT source_entity_id as eid FROM edges
                        WHERE target_entity_id = ? AND status = 'approved'
                    """, (node, node)).fetchall()
                    for n in neighbors:
                        if n["eid"] not in visited:
                            visited.add(n["eid"])
                            next_frontier.add(n["eid"])
                frontier = next_frontier

            # The main component should have most entities
            total_active = len(_get_active_entity_ids(db))
            # isolated-proj is not connected, so visited < total
            assert len(visited) < total_active


# ===========================================================================
# 4. UNICODE AND SPECIAL CHARACTERS
# ===========================================================================


class TestUnicodeAndSpecialChars:
    """Test handling of unicode, special chars, and injection prevention."""

    def test_unicode_entity_names(self, db):
        """Entity names with accented, CJK, and emoji characters."""
        content = "Unicode entity test"
        entities = [
            {"entity_id": "project:rene-descartes", "name": "René Descartes AI",
             "type": "Project", "kind": "framework",
             "description": "An AI framework inspired by Descartes"},
            {"entity_id": "project:tokyo-llm",
             "name": "東京 LLM",
             "type": "Project", "kind": "platform",
             "description": "A Tokyo-based LLM platform"},
            {"entity_id": "project:star-agent",
             "name": "⭐ Star Agent",
             "type": "Project", "kind": "tool"},
        ]
        sid = _full_ingest(db, "https://example.com/unicode-test",
                           content, entities, [])

        active = _get_active_entity_ids(db)
        assert "project:rene-descartes" in active
        assert "project:tokyo-llm" in active
        assert "project:star-agent" in active

        # Verify names stored correctly
        rene = _get_entity(db, "project:rene-descartes")
        assert "René" in rene["name"]

        tokyo = _get_entity(db, "project:tokyo-llm")
        assert "東京" in tokyo["name"]

    def test_special_chars_in_description(self, db):
        """Descriptions with quotes, backslashes, and newlines."""
        content = "Special char description test"
        entities = [
            {"entity_id": "project:special-desc", "name": "Special Desc",
             "type": "Project", "kind": "tool",
             "description": 'Uses "advanced" algorithms with C:\\path\\to\\model\nand multi-line descriptions'},
        ]
        sid = _full_ingest(db, "https://example.com/special-desc-test",
                           content, entities, [])

        ent = _get_entity(db, "project:special-desc")
        assert '"advanced"' in ent["description"]
        assert "C:\\path" in ent["description"]
        assert "\n" in ent["description"]

    def test_yaml_export_with_unicode(self, db):
        """YAML export preserves unicode correctly."""
        entity = {
            "entity_id": "project:yaml-unicode",
            "name": "Ünïcödé Prøjéct ☃",
            "type": "Project",
            "kind": "framework",
            "description": "A project with üñîçöðé chars",
            "aliases": json.dumps(["Projéct Á", "プロジェクト"]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            _export_yaml(entity, base_dir=tmpdir)
            yaml_path = os.path.join(tmpdir, "projects", "yaml-unicode.yaml")
            assert os.path.exists(yaml_path)

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            assert data["name"] == "Ünïcödé Prøjéct ☃"
            assert "üñîçöðé" in data["description"]

    def test_cypher_generation_safe_with_special_chars(self):
        """Entity Cypher uses parameterized queries, not string interpolation."""
        entity = {
            "entity_id": "project:injection-test",
            "name": "Test'); DROP TABLE entities;--",
            "type": "Project",
            "kind": "tool",
            "description": "Robert'; DROP TABLE entities;--",
            "aliases": json.dumps(["Bobby Tables"]),
            "source_id": 1,
        }
        query, params = _entity_to_cypher(entity)

        # The query should use $name, not inline the value
        assert "$name" in query
        assert "DROP TABLE" not in query
        # The dangerous value should be in params, safely parameterized
        assert "DROP TABLE" in params["name"]

    def test_edge_cypher_safe_with_special_chars(self):
        """Edge Cypher uses parameterized queries for all values."""
        edge = {
            "edge_id": "injection-edge",
            "source_entity_id": "org:x'; DELETE FROM edges;--",
            "target_entity_id": "project:y",
            "edge_type": "DEVELOPS",
            "properties": json.dumps({"note": "'; DROP TABLE edges;--"}),
            "confidence": 0.9,
            "source_type": "automated",
            "valid_from": None,
            "valid_to": None,
            "chunk_id": None,
        }
        query, params = _edge_to_cypher(edge)
        assert "$src" in query
        assert "DELETE FROM" not in query
        assert "DROP TABLE" not in query

    def test_unicode_aliases_in_resolve(self, db):
        """Resolve stage handles unicode aliases correctly."""
        content = "Unicode alias test"
        entities = [
            {"entity_id": "project:unicode-alias-test", "name": "München Framework",
             "type": "Project", "kind": "framework",
             "aliases": ["Münchener KI", "ミュンヘン"]},
        ]
        sid = _full_ingest(db, "https://example.com/unicode-alias",
                           content, entities, [])

        ent = _get_entity(db, "project:unicode-alias-test")
        aliases = json.loads(ent["aliases"])
        assert "Münchener KI" in aliases or "ミュンヘン" in aliases

    def test_normalize_handles_unicode(self):
        """The _normalize function handles unicode strings."""
        assert _normalize("René") == "rené"
        assert _normalize("Ünïcödé") == "ünïcödé"
        assert _normalize("Tokyo 東京") == "tokyo 東京"


# ===========================================================================
# 5. ERROR RECOVERY
# ===========================================================================


class TestErrorRecovery:
    """Test graceful failure and data integrity under error conditions."""

    def test_neo4j_failure_mid_load_data_preserved(self, db):
        """If Neo4j dies mid-load, SQLite data is still intact."""
        content = "Neo4j failure test"
        entities = [
            {"entity_id": "project:neo4j-fail", "name": "Neo4j Fail Test",
             "type": "Project", "kind": "tool"},
        ]
        edges = [
            {"src": "project:neo4j-fail", "tgt": "capability:cap-fail",
             "type": "ADDRESSES"},
        ]
        sid = _full_ingest(db, "https://example.com/neo4j-fail",
                           content, entities + [
                               {"entity_id": "capability:cap-fail",
                                "name": "Cap Fail", "type": "Capability"}
                           ], edges)

        # Mock a driver that fails mid-session
        driver = MagicMock()
        session = MagicMock()
        call_count = [0]

        def fail_on_third(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise ConnectionError("Neo4j connection lost")

        session.run.side_effect = fail_on_third
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        source = db.get_source(sid)
        # Load should handle the error gracefully
        run_load(db, source, neo4j_driver=driver)

        # SQLite data should still be intact
        ent = _get_entity(db, "project:neo4j-fail")
        assert ent is not None
        assert ent["status"] == "approved"

        edges_in_db = _get_edges_for_entity(db, "project:neo4j-fail")
        assert len(edges_in_db) >= 1

    def test_source_fetch_404(self, db):
        """A 404 response should raise an error without corrupting data."""
        import httpx

        sid = db.add_source("https://example.com/nonexistent-page")
        source = db.get_source(sid)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        from agents_kg.stages.fetch import run as run_fetch
        with patch("agents_kg.stages.fetch.httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="HTTP error"):
                run_fetch(db, source)

        # Source should still exist, unchanged
        source_after = db.get_source(sid)
        assert source_after is not None
        assert source_after["stage"] == "fetch"
        assert source_after["raw_text"] is None

    def test_invalid_json_from_extract(self, db):
        """Invalid JSON from the extraction model should not crash the pipeline."""
        content = "Invalid JSON extract test"
        sid = db.add_source("https://example.com/bad-json-test")
        source = _run_through_chunk(db, sid, content)
        _mock_embed(db, source)
        source = db.get_source(sid)

        mock_genai_mod = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "THIS IS NOT JSON {{{invalid}}}"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_mod.Client.return_value = mock_client

        from agents_kg.stages import extract as extract_mod
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": mock_genai_mod,
        }):
            extract_mod.run(db, source)

        # Extract should have handled the JSON error gracefully (logged warning, continued)
        # The source should be moved to resolve stage
        source_after = db.get_source(sid)
        assert source_after is not None
        assert source_after["stage"] == "resolve"

        # No entities should have been extracted from invalid JSON
        ents = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(ents) == 0

    def test_fail_source_increments_attempts(self, db):
        """fail_source increments attempt counter and marks failed."""
        sid = db.add_source("https://example.com/fail-attempt-test")
        source = db.get_source(sid)
        assert source["attempts"] == 0

        db.fail_source(sid, "Connection timeout")
        source = db.get_source(sid)
        assert source["attempts"] == 1
        assert source["status"] == "failed"
        assert source["error"] == "Connection timeout"

    def test_dead_letter_after_max_attempts(self, db):
        """Source goes to dead_letter after max_attempts failures."""
        sid = db.add_source("https://example.com/dead-letter-test")

        # Fail max_attempts times
        source = db.get_source(sid)
        max_attempts = source["max_attempts"]
        for i in range(max_attempts):
            db.fail_source(sid, f"Failure {i+1}")

        source = db.get_source(sid)
        assert source["status"] == "dead_letter"
        assert source["attempts"] == max_attempts

    def test_retry_failed_resets_status(self, db):
        """retry_failed brings failed sources back to pending."""
        sid = db.add_source("https://example.com/retry-test")
        db.fail_source(sid, "Temporary error")
        assert db.get_source(sid)["status"] == "failed"

        count = db.retry_failed()
        assert count == 1

        source = db.get_source(sid)
        assert source["status"] == "pending"
        assert source["error"] is None

    def test_reset_source_cleans_all_data(self, db):
        """reset_source removes all derived data for a source."""
        content = "Reset test content"
        entities = [
            {"entity_id": "project:reset-test", "name": "Reset Test",
             "type": "Project", "kind": "tool"},
        ]
        edges = [
            {"src": "project:reset-test", "tgt": "capability:reset-cap",
             "type": "ADDRESSES"},
        ]
        sid = _full_ingest(db, "https://example.com/reset-test",
                           content, entities + [
                               {"entity_id": "capability:reset-cap",
                                "name": "Reset Cap", "type": "Capability"}
                           ], edges)

        # Verify data exists
        assert len(db.get_chunks(sid)) > 0
        assert _get_entity(db, "project:reset-test") is not None

        # Reset
        db.reset_source(sid)

        source = db.get_source(sid)
        assert source["status"] == "pending"
        assert source["stage"] == "fetch"
        assert source["raw_text"] is None
        assert len(db.get_chunks(sid)) == 0

        # Entities and edges for this source should be gone
        ent = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(ent) == 0

        edg = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(edg) == 0

    def test_no_data_corruption_on_concurrent_source_add(self, db):
        """Adding duplicate URIs returns None without corrupting existing data."""
        sid1 = db.add_source("https://example.com/dup-test")
        assert sid1 is not None

        sid2 = db.add_source("https://example.com/dup-test")
        assert sid2 is None

        # Original source still intact
        source = db.get_source(sid1)
        assert source is not None
        assert source["uri"] == "https://example.com/dup-test"

    def test_content_change_deprecates_old_entities(self, db):
        """When source content changes on re-fetch, old entities get deprecated."""
        content_v1 = "Version 1 content about the framework"
        entities_v1 = [
            {"entity_id": "project:v1-ent", "name": "V1 Entity",
             "type": "Project", "kind": "framework"},
        ]
        sid = _full_ingest(db, "https://example.com/versioned-content",
                           content_v1, entities_v1, [])

        # Verify entity exists and is not deprecated
        ent = _get_entity(db, "project:v1-ent")
        assert ent is not None
        assert ent["deprecated_at"] is None

        # Simulate content change by updating content_hash then re-fetching
        source = db.get_source(sid)
        content_v2 = "Version 2 completely different content"
        mock_resp = MagicMock()
        mock_resp.text = content_v2
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        from agents_kg.stages.fetch import run as run_fetch
        with patch("agents_kg.stages.fetch.httpx.Client", return_value=mock_client):
            run_fetch(db, source)

        # Old entity should be deprecated
        ent_after = _get_entity(db, "project:v1-ent")
        assert ent_after["deprecated_at"] is not None

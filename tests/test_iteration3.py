"""Iteration 3 tests.

Covers: resolve stage vector similarity, Wikidata cross-reference enrichment,
ingestion order independence, stress testing, conference proceedings,
and whitespace handling fix.
"""

import math
import struct
import pytest
from unittest.mock import MagicMock, patch

from agents_kg.db import Database, content_hash
from agents_kg.stages.extract import _make_edge_id
from agents_kg.stages.resolve import (
    _cosine_similarity,
    _floats_to_bytes,
    _bytes_to_floats,
    _normalize,
    _similarity,
    run as run_resolve,
)

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


# ============================================================
# 1. RESOLVE STAGE WITH MOCK VECTOR SIMILARITY
# ============================================================


class TestResolveVectorSimilarity:
    """Exercise embedding-based entity deduplication in the resolve stage."""

    def test_identical_embeddings_merge(self, db):
        """Entities with identical embeddings (cosine=1.0) should merge."""
        sid = db.add_source("https://example.com/vec1")
        emb = _floats_to_bytes([0.5, 0.5, 0.5, 0.5])

        id1 = db.add_entity("protocol:http-v1", "HTTP Version 1", "Protocol",
                            description="HTTP", source_id=sid, embedding=emb)
        id2 = db.add_entity("protocol:http-version-one", "HTTP Version One", "Protocol",
                            description="HTTP v1", source_id=sid, embedding=emb)
        db.update_entity(id1, status="approved")

        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)

        ent2 = dict(db.conn.execute("SELECT * FROM entities WHERE id = ?", (id2,)).fetchone())
        assert ent2["status"] == "merged"
        assert ent2["merged_into"] == "protocol:http-v1"

    def test_below_threshold_stay_separate(self, db):
        """Entities with cosine < 0.92 should NOT merge."""
        sid = db.add_source("https://example.com/vec2")
        emb_a = _floats_to_bytes([1.0, 0.0, 0.0, 0.0])
        emb_b = _floats_to_bytes([0.0, 1.0, 0.0, 0.0])

        sim = _cosine_similarity([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        assert sim < 0.92

        id1 = db.add_entity("protocol:rest-api", "REST API", "Protocol",
                            description="RESTful", source_id=sid, embedding=emb_a)
        id2 = db.add_entity("protocol:graphql-api", "GraphQL API", "Protocol",
                            description="Graph query", source_id=sid, embedding=emb_b)
        db.update_entity(id1, status="approved")

        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)

        ent2 = dict(db.conn.execute("SELECT * FROM entities WHERE id = ?", (id2,)).fetchone())
        assert ent2["merged_into"] is None or ent2["merged_into"] == "noise"
        if ent2["merged_into"] is None:
            assert ent2["status"] != "merged"

    def test_just_below_threshold_stays_separate(self, db):
        """Entities with cosine ~0.91 should NOT merge (> 0.92 required)."""
        cos_target = 0.91
        a = [1.0, 0.0]
        b_x = cos_target
        b_y = math.sqrt(1 - cos_target ** 2)
        b = [b_x, b_y]

        actual_sim = _cosine_similarity(a, b)
        assert actual_sim < 0.92

        sid = db.add_source("https://example.com/vec3")
        emb_a = _floats_to_bytes(a)
        emb_b = _floats_to_bytes(b)

        id1 = db.add_entity("project:tool-alpha", "Tool Alpha", "Project",
                            description="Alpha", source_id=sid, embedding=emb_a)
        id2 = db.add_entity("project:tool-beta", "Tool Beta", "Project",
                            description="Beta", source_id=sid, embedding=emb_b)
        db.update_entity(id1, status="approved")

        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)

        ent2 = dict(db.conn.execute("SELECT * FROM entities WHERE id = ?", (id2,)).fetchone())
        assert ent2["status"] != "merged" or ent2["merged_into"] == "noise"

    def test_cross_type_no_merge(self, db):
        """Entities of different types should not merge even with identical embeddings."""
        sid = db.add_source("https://example.com/vec4")
        emb = _floats_to_bytes([0.5, 0.5, 0.5])

        id1 = db.add_entity("organization:acme", "Acme", "Organization",
                            description="Company", source_id=sid, embedding=emb)
        id2 = db.add_entity("project:acme", "Acme", "Project",
                            description="Software", source_id=sid, embedding=emb)
        db.update_entity(id1, status="approved")

        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)

        ent2 = dict(db.conn.execute("SELECT * FROM entities WHERE id = ?", (id2,)).fetchone())
        assert ent2["merged_into"] is None or ent2["merged_into"] == "noise"

    def test_edges_repointed_on_merge(self, db):
        """When entity B merges into entity A, B's edges should point to A."""
        sid = db.add_source("https://example.com/vec5")
        emb = _floats_to_bytes([0.9, 0.1, 0.1])

        id1 = db.add_entity("organization:bigcorp", "BigCorp", "Organization",
                            description="Corp", source_id=sid, embedding=emb)
        id2 = db.add_entity("organization:big-corp-inc", "Big Corp Inc", "Organization",
                            description="Corp Inc", source_id=sid, embedding=emb)
        db.update_entity(id1, status="approved")

        eid = _make_edge_id("organization:big-corp-inc", "project:widget", "DEVELOPS")
        db.add_edge(eid, "organization:big-corp-inc", "project:widget", "DEVELOPS",
                    source_id=sid)

        source = db.get_source(sid)
        with patch("agents_kg.stages.resolve.genai", None):
            run_resolve(db, source)

        edge = dict(db.conn.execute("SELECT * FROM edges WHERE edge_id = ?", (eid,)).fetchone())
        assert edge["source_entity_id"] == "organization:bigcorp"

    def test_cosine_utility_functions(self):
        """Unit tests for the cosine similarity helpers."""
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0

        vec = [0.1, 0.2, 0.3, 0.4]
        roundtripped = _bytes_to_floats(_floats_to_bytes(vec))
        for a, b in zip(vec, roundtripped):
            assert a == pytest.approx(b, abs=1e-6)

    def test_normalize_and_similarity(self):
        assert _normalize("  Model Context Protocol  ") == "model context protocol"
        assert _normalize("A2A-Protocol!") == "a2a-protocol"
        assert _similarity("Google", "google") == pytest.approx(1.0)
        assert _similarity("Google LLC", "Google") > 0.7


# ============================================================
# 2. WIKIDATA CROSS-REFERENCE ENRICHMENT
# ============================================================


class TestWikidataCrossRefEnrichment:
    """Mock SPARQL responses and test Wikidata cross-referencing."""

    def test_yaml_entities_matched_to_wikidata_ids(self, db, tmp_path):
        from agents_kg.wikidata_crossref import load_mappings, apply_crossref
        import yaml

        mappings = {
            "mappings": {
                "organization:google": "Q95",
                "protocol:http": "Q8777",
                "project:linux": "Q388",
            }
        }
        f = tmp_path / "mappings.yaml"
        f.write_text(yaml.dump(mappings))

        loaded = load_mappings(str(f))
        assert loaded["organization:google"] == "Q95"
        assert loaded["protocol:http"] == "Q8777"
        assert loaded["project:linux"] == "Q388"

    def test_no_match_entities_handled_gracefully(self, db, tmp_path):
        from agents_kg.wikidata_crossref import apply_crossref
        import yaml

        mappings = {"mappings": {"organization:nonexistent": None}}
        f = tmp_path / "mappings.yaml"
        f.write_text(yaml.dump(mappings))

        result = apply_crossref(neo4j_driver=None, mappings_path=str(f))
        assert result["skipped"] == 1
        assert result["applied"] == 0

    def test_apply_crossref_without_neo4j(self, db, tmp_path):
        from agents_kg.wikidata_crossref import apply_crossref
        import yaml

        mappings = {
            "mappings": {
                "organization:google": "Q95",
                "project:kubernetes": "Q22661317",
            }
        }
        f = tmp_path / "mappings.yaml"
        f.write_text(yaml.dump(mappings))

        result = apply_crossref(neo4j_driver=None, mappings_path=str(f))
        assert result["applied"] == 2
        assert result["skipped"] == 0

    def test_qid_normalization(self, db, tmp_path):
        """Q-IDs without 'Q' prefix get normalized."""
        from agents_kg.wikidata_crossref import apply_crossref
        import yaml

        mappings = {"mappings": {"organization:test": 12345}}
        f = tmp_path / "mappings.yaml"
        f.write_text(yaml.dump(mappings))

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"updated": 1}
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = lambda s: mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = apply_crossref(neo4j_driver=mock_driver, mappings_path=str(f))
        assert result["applied"] == 1
        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["wikidata_id"] == "Q12345"

    def test_mock_sparql_transform(self):
        """Mock SPARQL bindings → entity transform."""
        from agents_kg.wikidata import transform_to_entities

        bindings = [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q95"},
                "itemLabel": {"value": "Google"},
                "itemDescription": {"value": "American technology company"},
                "inception": {"value": "1998-09-04T00:00:00Z"},
            },
            {
                "item": {"value": "http://www.wikidata.org/entity/Q312"},
                "itemLabel": {"value": "Apple Inc."},
                "itemDescription": {"value": "American technology company"},
            },
        ]

        entities = transform_to_entities(bindings, "Organization", "company")
        assert len(entities) == 2
        google = [e for e in entities if e["wikidata_id"] == "Q95"][0]
        assert google["name"] == "Google"
        assert google["type"] == "Organization"
        assert google["created_at"] == "1998-09-04"

    def test_sparql_ambiguous_match(self):
        """Multiple bindings with same label but different Q-IDs — dedup by QID."""
        from agents_kg.wikidata import transform_to_entities

        bindings = [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q95"},
                "itemLabel": {"value": "Google"},
                "itemDescription": {"value": "Tech company"},
            },
            {
                "item": {"value": "http://www.wikidata.org/entity/Q95"},
                "itemLabel": {"value": "Google"},
                "itemDescription": {"value": "Different description"},
            },
            {
                "item": {"value": "http://www.wikidata.org/entity/Q99999"},
                "itemLabel": {"value": "Google"},
                "itemDescription": {"value": "A different Google"},
            },
        ]

        entities = transform_to_entities(bindings, "Organization", "company")
        assert len(entities) == 2
        qids = {e["wikidata_id"] for e in entities}
        assert qids == {"Q95", "Q99999"}


# ============================================================
# 3. INGESTION ORDER INDEPENDENCE
# ============================================================


class TestIngestionOrderIndependence:
    """Verify that entity/edge sets are equivalent regardless of ingestion order."""

    SOURCE_A_CONTENT = """# Kubernetes Architecture

Kubernetes was originally designed by Google and is now maintained by
the Cloud Native Computing Foundation (CNCF). It automates deployment,
scaling, and management of containerized applications.
"""

    SOURCE_A_ENTITIES = [
        {"entity_id": "project:kubernetes", "name": "Kubernetes", "type": "Project",
         "kind": "platform", "description": "Container orchestration platform"},
        {"entity_id": "organization:google", "name": "Google", "type": "Organization",
         "kind": "company", "description": "Tech company"},
        {"entity_id": "organization:cncf", "name": "CNCF", "type": "Organization",
         "kind": "foundation", "description": "Cloud Native Computing Foundation"},
    ]
    SOURCE_A_EDGES = [
        {"src": "organization:google", "tgt": "project:kubernetes", "type": "DEVELOPS"},
        {"src": "organization:cncf", "tgt": "project:kubernetes", "type": "DEVELOPS"},
    ]

    SOURCE_B_CONTENT = """# Cloud Native Ecosystem

Docker provides containerization technology. Google Cloud Platform
offers managed Kubernetes. The CNCF governs many cloud native projects
including Envoy and Prometheus.
"""

    SOURCE_B_ENTITIES = [
        {"entity_id": "project:docker", "name": "Docker", "type": "Project",
         "kind": "platform", "description": "Containerization platform"},
        {"entity_id": "organization:google", "name": "Google", "type": "Organization",
         "kind": "company", "description": "Tech company"},
        {"entity_id": "organization:cncf", "name": "CNCF", "type": "Organization",
         "kind": "foundation", "description": "Cloud Native Computing Foundation"},
        {"entity_id": "project:envoy", "name": "Envoy", "type": "Project",
         "kind": "tool", "description": "Service proxy"},
    ]
    SOURCE_B_EDGES = [
        {"src": "organization:cncf", "tgt": "project:envoy", "type": "DEVELOPS"},
    ]

    def _get_entity_ids(self, db):
        rows = db.conn.execute(
            "SELECT entity_id FROM entities WHERE merged_into IS NULL AND status != 'rejected'"
        ).fetchall()
        return {r["entity_id"] for r in rows}

    def _get_edge_set(self, db):
        rows = db.conn.execute(
            "SELECT source_entity_id, target_entity_id, edge_type FROM edges"
        ).fetchall()
        return {(r["source_entity_id"], r["target_entity_id"], r["edge_type"]) for r in rows}

    def test_order_ab_equals_order_ba(self, db, tmp_path):
        import os, tempfile

        # Order A→B
        _full_ingest(db, "https://example.com/k8s-arch",
                     self.SOURCE_A_CONTENT, self.SOURCE_A_ENTITIES, self.SOURCE_A_EDGES)
        _full_ingest(db, "https://example.com/cloud-native",
                     self.SOURCE_B_CONTENT, self.SOURCE_B_ENTITIES, self.SOURCE_B_EDGES)
        entities_ab = self._get_entity_ids(db)
        edges_ab = self._get_edge_set(db)

        # Order B→A in a fresh DB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path2 = f.name
        db2 = Database(path2)
        try:
            _full_ingest(db2, "https://example.com/cloud-native",
                         self.SOURCE_B_CONTENT, self.SOURCE_B_ENTITIES, self.SOURCE_B_EDGES)
            _full_ingest(db2, "https://example.com/k8s-arch",
                         self.SOURCE_A_CONTENT, self.SOURCE_A_ENTITIES, self.SOURCE_A_EDGES)
            entities_ba = self._get_entity_ids(db2)
            edges_ba = self._get_edge_set(db2)
        finally:
            db2.close()
            os.unlink(path2)

        assert entities_ab == entities_ba
        assert edges_ab == edges_ba

    def test_shared_entity_not_duplicated(self, db):
        _full_ingest(db, "https://example.com/k8s-arch",
                     self.SOURCE_A_CONTENT, self.SOURCE_A_ENTITIES, self.SOURCE_A_EDGES)
        _full_ingest(db, "https://example.com/cloud-native",
                     self.SOURCE_B_CONTENT, self.SOURCE_B_ENTITIES, self.SOURCE_B_EDGES)

        google_rows = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'organization:google' AND merged_into IS NULL"
        ).fetchall()
        assert len(google_rows) == 1

    def test_all_edges_preserved(self, db):
        _full_ingest(db, "https://example.com/k8s-arch",
                     self.SOURCE_A_CONTENT, self.SOURCE_A_ENTITIES, self.SOURCE_A_EDGES)
        _full_ingest(db, "https://example.com/cloud-native",
                     self.SOURCE_B_CONTENT, self.SOURCE_B_ENTITIES, self.SOURCE_B_EDGES)

        edges = self._get_edge_set(db)
        assert ("organization:google", "project:kubernetes", "DEVELOPS") in edges
        assert ("organization:cncf", "project:kubernetes", "DEVELOPS") in edges
        assert ("organization:cncf", "project:envoy", "DEVELOPS") in edges


# ============================================================
# 4. STRESS TEST (50+ entities)
# ============================================================


class TestStressTest:
    """Source with 50+ entity mentions — verify integrity at scale."""

    def _generate_large_source(self):
        lines = ["# AI Ecosystem Survey 2026\n"]
        entities = []
        edges = []

        for i in range(55):
            org_name = f"AICorp{i}"
            proj_name = f"AITool{i}"
            org_id = f"organization:aicorp{i}"
            proj_id = f"project:aitool{i}"

            lines.append(f"\n## {org_name}\n{org_name} develops {proj_name}, "
                         f"an AI tool for enterprise automation.\n")
            entities.append({"entity_id": org_id, "name": org_name,
                             "type": "Organization", "kind": "company",
                             "description": f"{org_name} is an AI company"})
            entities.append({"entity_id": proj_id, "name": proj_name,
                             "type": "Project", "kind": "tool",
                             "description": f"{proj_name} is an AI tool"})
            edges.append({"src": org_id, "tgt": proj_id, "type": "DEVELOPS"})

        return "\n".join(lines), entities, edges

    def test_50_plus_entities_integrity(self, db):
        content, entities, edges = self._generate_large_source()
        assert len(entities) == 110
        assert len(edges) == 55

        sid = _full_ingest(db, "https://example.com/ai-survey",
                           content, entities, edges)

        all_entities = db.conn.execute(
            "SELECT * FROM entities WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(all_entities) == 110

        all_edges = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (sid,)
        ).fetchall()
        assert len(all_edges) == 55

        chunks = db.get_chunks(sid)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["text"].strip() != ""

    def test_large_source_entity_edge_linking(self, db):
        content, entities, edges = self._generate_large_source()
        sid = _full_ingest(db, "https://example.com/ai-survey-link",
                           content, entities, edges)

        for i in range(55):
            org_id = f"organization:aicorp{i}"
            proj_id = f"project:aitool{i}"
            org = db.conn.execute(
                "SELECT * FROM entities WHERE entity_id = ? AND source_id = ?",
                (org_id, sid)
            ).fetchone()
            proj = db.conn.execute(
                "SELECT * FROM entities WHERE entity_id = ? AND source_id = ?",
                (proj_id, sid)
            ).fetchone()
            assert org is not None, f"Missing entity {org_id}"
            assert proj is not None, f"Missing entity {proj_id}"

        edge_rows = db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (sid,)
        ).fetchall()
        edge_set = {(dict(e)["source_entity_id"], dict(e)["target_entity_id"])
                    for e in edge_rows}
        for i in range(55):
            assert (f"organization:aicorp{i}", f"project:aitool{i}") in edge_set

    def test_entity_types_distribution(self, db):
        content, entities, edges = self._generate_large_source()
        _full_ingest(db, "https://example.com/ai-survey-dist",
                     content, entities, edges)

        orgs = db.conn.execute(
            "SELECT COUNT(*) as c FROM entities WHERE type = 'Organization'"
        ).fetchone()["c"]
        projs = db.conn.execute(
            "SELECT COUNT(*) as c FROM entities WHERE type = 'Project'"
        ).fetchone()["c"]
        assert orgs == 55
        assert projs == 55


# ============================================================
# 5. CONFERENCE PROCEEDINGS SOURCE
# ============================================================


class TestConferenceProceedings:
    """Structured document: speakers, talks, organizations, schedules."""

    CONF_CONTENT = """# AgentConf 2026 Proceedings

## Keynote: The Future of Autonomous Agents
Speaker: Dr. Alice Chen, VP of AI Research at DeepMind

The keynote explored how autonomous agents will reshape enterprise workflows.

## Talk: Building Multi-Agent Systems with A2A
Speaker: Bob Martinez, Staff Engineer at Google

Bob demonstrated how the A2A protocol enables inter-agent communication
across organizational boundaries.

## Talk: Securing Agent Pipelines
Speaker: Carol Davis, Security Lead at Microsoft

Carol presented threat models for agent-to-agent interactions and proposed
defense mechanisms using the MCP protocol.

## Workshop: Hands-on Agent Development
Facilitator: David Kim, CTO at AgentStack

Participants built agents using the CrewAI framework integrated with
LangChain for orchestration.

## Panel: Standards for Agentic AI
Panelists: Dr. Alice Chen (DeepMind), Eve Wong (Anthropic), Frank Liu (OpenAI)

The panel discussed the need for standardization in the agentic AI space.
"""

    CONF_ENTITIES = [
        {"entity_id": "person:alice-chen", "name": "Dr. Alice Chen", "type": "Person",
         "description": "VP of AI Research at DeepMind"},
        {"entity_id": "organization:deepmind", "name": "DeepMind", "type": "Organization",
         "kind": "company", "description": "AI research lab"},
        {"entity_id": "person:bob-martinez", "name": "Bob Martinez", "type": "Person",
         "description": "Staff Engineer at Google"},
        {"entity_id": "organization:google", "name": "Google", "type": "Organization",
         "kind": "company", "description": "Tech company"},
        {"entity_id": "protocol:a2a", "name": "A2A", "type": "Protocol",
         "kind": "spec", "description": "Agent-to-Agent protocol"},
        {"entity_id": "person:carol-davis", "name": "Carol Davis", "type": "Person",
         "description": "Security Lead at Microsoft"},
        {"entity_id": "organization:microsoft", "name": "Microsoft", "type": "Organization",
         "kind": "company", "description": "Tech company"},
        {"entity_id": "protocol:mcp", "name": "MCP", "type": "Protocol",
         "kind": "spec", "description": "Model Context Protocol"},
        {"entity_id": "person:david-kim", "name": "David Kim", "type": "Person",
         "description": "CTO at AgentStack"},
        {"entity_id": "organization:agentstack", "name": "AgentStack", "type": "Organization",
         "kind": "company", "description": "Agent development company"},
        {"entity_id": "project:crewai", "name": "CrewAI", "type": "Project",
         "kind": "framework", "description": "Multi-agent framework"},
        {"entity_id": "project:langchain", "name": "LangChain", "type": "Project",
         "kind": "framework", "description": "LLM orchestration framework"},
        {"entity_id": "person:eve-wong", "name": "Eve Wong", "type": "Person",
         "description": "Anthropic representative"},
        {"entity_id": "organization:anthropic", "name": "Anthropic", "type": "Organization",
         "kind": "company", "description": "AI safety company"},
        {"entity_id": "person:frank-liu", "name": "Frank Liu", "type": "Person",
         "description": "OpenAI representative"},
        {"entity_id": "organization:openai", "name": "OpenAI", "type": "Organization",
         "kind": "company", "description": "AI research lab"},
    ]

    CONF_EDGES = [
        {"src": "person:alice-chen", "tgt": "organization:deepmind", "type": "MEMBER_OF"},
        {"src": "person:bob-martinez", "tgt": "organization:google", "type": "MEMBER_OF"},
        {"src": "organization:google", "tgt": "protocol:a2a", "type": "DEVELOPS"},
        {"src": "person:carol-davis", "tgt": "organization:microsoft", "type": "MEMBER_OF"},
        {"src": "person:david-kim", "tgt": "organization:agentstack", "type": "MEMBER_OF"},
        {"src": "person:eve-wong", "tgt": "organization:anthropic", "type": "MEMBER_OF"},
        {"src": "person:frank-liu", "tgt": "organization:openai", "type": "MEMBER_OF"},
    ]

    def test_conference_ingestion_completes(self, db):
        sid = _full_ingest(db, "https://example.com/agentconf2026",
                           self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)
        source = db.get_source(sid)
        assert source["stage"] == "review"

    def test_person_entities_extracted(self, db):
        _full_ingest(db, "https://example.com/agentconf2026-p",
                     self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)

        people = db.conn.execute(
            "SELECT * FROM entities WHERE type = 'Person' AND merged_into IS NULL"
        ).fetchall()
        names = {dict(p)["name"] for p in people}
        assert "Dr. Alice Chen" in names
        assert "Bob Martinez" in names
        assert "Carol Davis" in names
        assert "David Kim" in names

    def test_org_entities_extracted(self, db):
        _full_ingest(db, "https://example.com/agentconf2026-o",
                     self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)

        orgs = db.conn.execute(
            "SELECT * FROM entities WHERE type = 'Organization' AND merged_into IS NULL"
        ).fetchall()
        org_ids = {dict(o)["entity_id"] for o in orgs}
        assert "organization:deepmind" in org_ids
        assert "organization:google" in org_ids
        assert "organization:microsoft" in org_ids
        assert "organization:anthropic" in org_ids

    def test_person_affiliated_with_org(self, db):
        _full_ingest(db, "https://example.com/agentconf2026-a",
                     self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)

        edges = db.conn.execute(
            "SELECT * FROM edges WHERE edge_type = 'MEMBER_OF'"
        ).fetchall()
        affiliations = {(dict(e)["source_entity_id"], dict(e)["target_entity_id"])
                        for e in edges}
        assert ("person:alice-chen", "organization:deepmind") in affiliations
        assert ("person:bob-martinez", "organization:google") in affiliations
        assert ("person:carol-davis", "organization:microsoft") in affiliations

    def test_protocol_entities_present(self, db):
        _full_ingest(db, "https://example.com/agentconf2026-pr",
                     self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)

        protocols = db.conn.execute(
            "SELECT * FROM entities WHERE type = 'Protocol' AND merged_into IS NULL"
        ).fetchall()
        proto_ids = {dict(p)["entity_id"] for p in protocols}
        assert "protocol:a2a" in proto_ids
        assert "protocol:mcp" in proto_ids

    def test_project_entities_present(self, db):
        _full_ingest(db, "https://example.com/agentconf2026-proj",
                     self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)

        projects = db.conn.execute(
            "SELECT * FROM entities WHERE type = 'Project' AND merged_into IS NULL"
        ).fetchall()
        proj_ids = {dict(p)["entity_id"] for p in projects}
        assert "project:crewai" in proj_ids
        assert "project:langchain" in proj_ids

    def test_chunks_capture_structured_sections(self, db):
        sid = _full_ingest(db, "https://example.com/agentconf2026-ch",
                           self.CONF_CONTENT, self.CONF_ENTITIES, self.CONF_EDGES)

        chunks = db.get_chunks(sid)
        assert len(chunks) >= 5
        all_text = " ".join(c["text"] for c in chunks)
        assert "Keynote" in all_text
        assert "Workshop" in all_text
        assert "Panel" in all_text


# ============================================================
# 6. WHITESPACE HANDLING FIX
# ============================================================


class TestWhitespaceHandling:
    """Validate the whitespace chunk fix: empty/whitespace-only chunks are skipped."""

    def test_whitespace_only_content_produces_no_empty_chunks(self, db):
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("https://example.com/ws1")
        db.update_source(sid, parsed_text="# Header\n\nReal content here.\n\n   \n\n\t\n\n## Another\n\nMore text.")
        source = db.get_source(sid)
        run_chunk(db, source)

        chunks = db.get_chunks(sid)
        for chunk in chunks:
            assert chunk["text"].strip() != "", f"Empty chunk at position {chunk['position']}"

    def test_all_whitespace_document_produces_zero_chunks(self, db):
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("https://example.com/ws2")
        db.update_source(sid, parsed_text="   \n\n\t\n   \n\n  ")
        source = db.get_source(sid)
        run_chunk(db, source)

        chunks = db.get_chunks(sid)
        assert len(chunks) == 0

    def test_tabs_and_spaces_only_skipped(self, db):
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("https://example.com/ws3")
        db.update_source(sid, parsed_text="# Real Section\n\nGood content\n\n\t   \t\n\n# Another\n\nAlso good")
        source = db.get_source(sid)
        run_chunk(db, source)

        chunks = db.get_chunks(sid)
        for chunk in chunks:
            assert len(chunk["text"].strip()) > 0

    def test_newlines_only_section_skipped(self, db):
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("https://example.com/ws4")
        db.update_source(sid, parsed_text="# Title\n\nContent\n\n\n\n\n\n# End\n\nFinal")
        source = db.get_source(sid)
        run_chunk(db, source)

        chunks = db.get_chunks(sid)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk["text"].strip() != ""

    def test_mixed_valid_and_whitespace_preserves_positions(self, db):
        from agents_kg.stages.chunk import run as run_chunk

        sid = db.add_source("https://example.com/ws5")
        db.update_source(sid, parsed_text="# A\n\nText A\n\n# B\n\nText B\n\n# C\n\nText C")
        source = db.get_source(sid)
        run_chunk(db, source)

        chunks = db.get_chunks(sid)
        positions = [c["position"] for c in chunks]
        assert positions == list(range(len(chunks)))

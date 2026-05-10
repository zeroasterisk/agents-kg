"""Iteration 8 — Data quality and consistency tests.

Validates cross-database consistency (SQLite ↔ Neo4j), referential integrity
of edges, source/chunk traceability, and wikidata ID format correctness.
"""

import json
import os
import re
import tempfile

import pytest
import yaml
from pathlib import Path

from agents_kg.db import Database
from agents_kg.seed import get_seed_entities, SEED_ENTITIES
from agents_kg.stages import parse, chunk, load
from agents_kg.stages.extract import VALID_ENTITY_TYPES, VALID_EDGE_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


def _build_populated_db(db, n_sources=2, n_entities_per=3):
    """Populate a DB with sources, chunks, entities, and edges."""
    for s in range(n_sources):
        sid = db.add_source(f"https://example.com/quality-{s}")
        text = f"# Source {s}\n\n" + "\n\n".join(
            f"## Section {i}\n\nContent about entity {i} in source {s}" for i in range(n_entities_per)
        )
        db.update_source(sid, raw_text=text, type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)

        chunks = db.get_chunks(sid)

        for i in range(n_entities_per):
            eid = f"test:quality-s{s}-e{i}"
            chunk_id = chunks[min(i, len(chunks) - 1)]["id"]
            db.add_entity(
                entity_id=eid,
                name=f"QualityEntity S{s} E{i}",
                entity_type="Project",
                kind="tool",
                description=f"Entity {i} from source {s}",
                source_id=sid,
                chunk_id=chunk_id,
            )

        for i in range(n_entities_per - 1):
            db.add_edge(
                edge_id=f"quality-edge-s{s}-{i}",
                source_entity_id=f"test:quality-s{s}-e{i}",
                target_entity_id=f"test:quality-s{s}-e{i+1}",
                edge_type="DEVELOPS",
                source_id=sid,
                chunk_id=chunks[0]["id"],
            )


# ---------------------------------------------------------------------------
# 1. Cross-validate SQLite entities and edges
# ---------------------------------------------------------------------------

class TestSQLiteConsistency:

    def test_every_edge_references_existing_entities(self, db):
        """Every edge's source_entity_id and target_entity_id must exist in entities table."""
        _build_populated_db(db)
        edges = db.conn.execute("SELECT * FROM edges").fetchall()
        entity_ids = {
            r["entity_id"]
            for r in db.conn.execute("SELECT entity_id FROM entities").fetchall()
        }

        for edge in edges:
            assert edge["source_entity_id"] in entity_ids, \
                f"Dangling source ref: {edge['source_entity_id']} in edge {edge['edge_id']}"
            assert edge["target_entity_id"] in entity_ids, \
                f"Dangling target ref: {edge['target_entity_id']} in edge {edge['edge_id']}"

    def test_every_entity_has_a_source(self, db):
        """Every entity traces back to a source."""
        _build_populated_db(db)
        entities = db.conn.execute("SELECT * FROM entities").fetchall()
        source_ids = {
            r["id"] for r in db.conn.execute("SELECT id FROM sources").fetchall()
        }

        for ent in entities:
            assert ent["source_id"] in source_ids, \
                f"Entity {ent['entity_id']} has invalid source_id={ent['source_id']}"

    def test_every_entity_traces_to_chunk_that_traces_to_source(self, db):
        """Full traceability: entity → chunk → source."""
        _build_populated_db(db)
        entities = db.conn.execute("SELECT * FROM entities WHERE chunk_id IS NOT NULL").fetchall()

        for ent in entities:
            chunk_row = db.conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (ent["chunk_id"],)
            ).fetchone()
            assert chunk_row is not None, \
                f"Entity {ent['entity_id']} references non-existent chunk {ent['chunk_id']}"
            assert chunk_row["source_id"] == ent["source_id"], \
                f"Entity {ent['entity_id']} chunk source mismatch: chunk.source_id={chunk_row['source_id']} != entity.source_id={ent['source_id']}"

    def test_no_orphan_chunks(self, db):
        """Every chunk belongs to an existing source."""
        _build_populated_db(db)
        chunks = db.conn.execute("SELECT * FROM chunks").fetchall()
        source_ids = {
            r["id"] for r in db.conn.execute("SELECT id FROM sources").fetchall()
        }

        for c in chunks:
            assert c["source_id"] in source_ids, \
                f"Orphan chunk {c['id']} references non-existent source {c['source_id']}"

    def test_no_duplicate_entity_ids(self, db):
        """Entity IDs are unique (enforced by schema)."""
        _build_populated_db(db)
        rows = db.conn.execute("SELECT entity_id, COUNT(*) as cnt FROM entities GROUP BY entity_id HAVING cnt > 1").fetchall()
        assert len(rows) == 0, f"Duplicate entity_ids: {[r['entity_id'] for r in rows]}"

    def test_no_duplicate_edge_ids(self, db):
        """Edge IDs are unique (enforced by schema)."""
        _build_populated_db(db)
        rows = db.conn.execute("SELECT edge_id, COUNT(*) as cnt FROM edges GROUP BY edge_id HAVING cnt > 1").fetchall()
        assert len(rows) == 0, f"Duplicate edge_ids: {[r['edge_id'] for r in rows]}"

    def test_edge_types_are_valid(self, db):
        """All edges use valid edge types from the ontology."""
        _build_populated_db(db)
        edges = db.conn.execute("SELECT DISTINCT edge_type FROM edges").fetchall()
        for edge in edges:
            assert edge["edge_type"] in VALID_EDGE_TYPES, \
                f"Invalid edge type: {edge['edge_type']}"

    def test_entity_types_are_valid(self, db):
        """All entities use valid types from the ontology."""
        _build_populated_db(db)
        entities = db.conn.execute("SELECT DISTINCT type FROM entities").fetchall()
        for ent in entities:
            assert ent["type"] in VALID_ENTITY_TYPES, \
                f"Invalid entity type: {ent['type']}"


# ---------------------------------------------------------------------------
# 2. Source consistency after full pipeline
# ---------------------------------------------------------------------------

class TestSourceConsistency:

    def test_completed_source_has_content_hash(self, db):
        """A source that completed processing has a content hash."""
        sid = db.add_source("https://example.com/hash-check")
        db.update_source(sid, raw_text="# Test\n\nContent", type="text",
                         content_hash="abc123", status="complete", stage="done")
        src = db.get_source(sid)
        assert src["content_hash"] is not None
        assert len(src["content_hash"]) > 0

    def test_pending_source_has_no_chunks(self, db):
        """A pending source (stage=fetch) has no chunks yet."""
        sid = db.add_source("https://example.com/no-chunks")
        chunks = db.get_chunks(sid)
        assert len(chunks) == 0

    def test_source_stages_are_ordered(self, db):
        """Processing sources follow the expected stage order."""
        from agents_kg.pipeline import STAGE_ORDER
        sid = db.add_source("https://example.com/stage-order")
        db.update_source(sid, raw_text="# Test\n\nContent", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        assert STAGE_ORDER.index(src["stage"]) > STAGE_ORDER.index("parse")

    def test_uri_uniqueness_enforced(self, db):
        """Duplicate URIs return None from add_source."""
        sid1 = db.add_source("https://example.com/unique")
        sid2 = db.add_source("https://example.com/unique")
        assert sid1 is not None
        assert sid2 is None


# ---------------------------------------------------------------------------
# 3. Wikidata ID format validation
# ---------------------------------------------------------------------------

class TestWikidataIDFormat:

    def test_qid_format_regex(self):
        """Valid Q-format IDs match pattern Q[0-9]+."""
        valid = ["Q42", "Q12345", "Q1"]
        invalid = ["P42", "Q", "42", "Qabc", ""]
        for qid in valid:
            assert re.match(r'^Q\d+$', qid), f"Should be valid: {qid}"
        for qid in invalid:
            assert not re.match(r'^Q\d+$', qid), f"Should be invalid: {qid}"

    def test_wikidata_mappings_have_valid_qids(self):
        """All non-null QIDs in wikidata_mappings.yaml are well-formed Q-IDs."""
        mappings_path = Path("/scion-volumes/scratchpad/agents-kg/kg/wikidata_mappings.yaml")
        if not mappings_path.exists():
            pytest.skip("No wikidata_mappings.yaml found")

        data = yaml.safe_load(mappings_path.read_text())
        if not data:
            pytest.skip("Empty mappings file")

        mappings = data.get("mappings", data)
        if not isinstance(mappings, dict):
            pytest.skip("Unexpected mappings format")

        for entity_id, qid in mappings.items():
            if qid is None:
                continue
            assert re.match(r'^Q\d+$', str(qid)), \
                f"Entity {entity_id} has malformed QID: {qid}"

    def test_wikidata_crossref_module_handles_empty(self):
        """wikidata_crossref handles missing mapping file gracefully."""
        from agents_kg.wikidata_crossref import load_mappings
        result = load_mappings("/nonexistent/path.yaml")
        assert result == {}


# ---------------------------------------------------------------------------
# 4. Entity field completeness
# ---------------------------------------------------------------------------

class TestEntityFieldCompleteness:

    def test_entities_have_required_fields(self, db):
        """All entities have entity_id, name, and type."""
        _build_populated_db(db)
        entities = db.conn.execute("SELECT * FROM entities").fetchall()
        for ent in entities:
            assert ent["entity_id"], f"Missing entity_id on row {ent['id']}"
            assert ent["name"], f"Missing name on entity {ent['entity_id']}"
            assert ent["type"], f"Missing type on entity {ent['entity_id']}"

    def test_entity_id_format(self, db):
        """Entity IDs follow type:kebab-case pattern."""
        _build_populated_db(db)
        entities = db.conn.execute("SELECT entity_id FROM entities").fetchall()
        for ent in entities:
            eid = ent["entity_id"]
            assert ":" in eid, f"Entity ID missing colon separator: {eid}"

    def test_edges_have_required_fields(self, db):
        """All edges have edge_id, source_entity_id, target_entity_id, edge_type."""
        _build_populated_db(db)
        edges = db.conn.execute("SELECT * FROM edges").fetchall()
        for edge in edges:
            assert edge["edge_id"], f"Missing edge_id on row {edge['id']}"
            assert edge["source_entity_id"], f"Missing source_entity_id on edge {edge['edge_id']}"
            assert edge["target_entity_id"], f"Missing target_entity_id on edge {edge['edge_id']}"
            assert edge["edge_type"], f"Missing edge_type on edge {edge['edge_id']}"

    def test_entity_aliases_are_json_arrays(self, db):
        """Entity aliases field is valid JSON array."""
        _build_populated_db(db)
        entities = db.conn.execute("SELECT entity_id, aliases FROM entities").fetchall()
        for ent in entities:
            aliases = ent["aliases"]
            if aliases:
                parsed = json.loads(aliases)
                assert isinstance(parsed, list), \
                    f"Entity {ent['entity_id']} aliases is not a list: {aliases}"

    def test_edge_properties_are_json_objects(self, db):
        """Edge properties field is valid JSON object."""
        _build_populated_db(db)
        edges = db.conn.execute("SELECT edge_id, properties FROM edges").fetchall()
        for edge in edges:
            props = edge["properties"]
            if props:
                parsed = json.loads(props)
                assert isinstance(parsed, dict), \
                    f"Edge {edge['edge_id']} properties is not a dict: {props}"


# ---------------------------------------------------------------------------
# 5. YAML entity file consistency
# ---------------------------------------------------------------------------

class TestYAMLEntityFiles:

    ENTITIES_DIR = Path("/scion-volumes/scratchpad/agents-kg/kg/entities")

    def _load_all_yaml_entities(self):
        """Load all entity YAML files."""
        entities = []
        for yf in sorted(self.ENTITIES_DIR.rglob("*.yaml")):
            data = yaml.safe_load(yf.read_text())
            if data and "id" in data:
                data["_file"] = str(yf)
                entities.append(data)
        return entities

    def test_yaml_entities_have_required_fields(self):
        """All YAML entity files have id and name."""
        entities = self._load_all_yaml_entities()
        assert len(entities) > 0, "No YAML entity files found"
        for ent in entities:
            assert "id" in ent, f"Missing 'id' in {ent['_file']}"
            assert "name" in ent, f"Missing 'name' in {ent['_file']}"

    def test_yaml_entities_have_valid_types(self):
        """All YAML entities have valid types."""
        entities = self._load_all_yaml_entities()
        for ent in entities:
            if "type" in ent:
                etype = ent["type"]
                etype_title = etype.capitalize() if etype == etype.lower() else etype
                assert etype_title in VALID_ENTITY_TYPES or etype_title == "Concept", \
                    f"Invalid type '{etype}' in {ent['_file']}"

    def test_no_duplicate_yaml_entity_ids(self):
        """No two YAML files share the same entity ID."""
        entities = self._load_all_yaml_entities()
        seen = {}
        for ent in entities:
            eid = ent["id"]
            if eid in seen:
                pytest.fail(f"Duplicate entity ID '{eid}' in {ent['_file']} and {seen[eid]}")
            seen[eid] = ent["_file"]

    def test_yaml_aliases_are_lists(self):
        """Aliases in YAML files are lists, not strings."""
        entities = self._load_all_yaml_entities()
        for ent in entities:
            if "aliases" in ent:
                assert isinstance(ent["aliases"], list), \
                    f"Aliases should be a list in {ent['_file']}, got {type(ent['aliases'])}"

"""Iteration 6: Data export, YAML well-formedness, and round-trip tests."""

import json
import os
import tempfile
import pytest
import yaml
from pathlib import Path
from agents_kg.db import Database
from agents_kg.stages.load import _export_yaml, _entity_to_cypher, _edge_to_cypher


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


@pytest.fixture
def export_dir(tmp_path):
    return str(tmp_path / "kg_export")


# ── YAML export ──────────────────────────────────────────────────────────────

class TestYamlExport:
    def test_export_creates_file(self, export_dir):
        entity = {
            "entity_id": "organization:test-org",
            "name": "Test Organization",
            "type": "Organization",
            "kind": "company",
            "description": "A test organization",
            "aliases": json.dumps(["TestOrg", "TO"]),
        }
        _export_yaml(entity, base_dir=export_dir)
        expected = Path(export_dir) / "organizations" / "test-org.yaml"
        assert expected.exists()

    def test_export_yaml_well_formed(self, export_dir):
        entity = {
            "entity_id": "protocol:test-proto",
            "name": "Test Protocol",
            "type": "Protocol",
            "kind": "spec",
            "description": "A test protocol specification",
            "aliases": json.dumps(["TP", "TestProto"]),
        }
        _export_yaml(entity, base_dir=export_dir)
        path = Path(export_dir) / "protocols" / "test-proto.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["id"] == "protocol:test-proto"
        assert data["name"] == "Test Protocol"
        assert data["type"] == "Protocol"
        assert data["kind"] == "spec"
        assert "TP" in data["aliases"]

    def test_export_removes_none_values(self, export_dir):
        entity = {
            "entity_id": "project:no-desc",
            "name": "No Description",
            "type": "Project",
            "kind": None,
            "description": None,
            "aliases": "[]",
        }
        _export_yaml(entity, base_dir=export_dir)
        path = Path(export_dir) / "projects" / "no-desc.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "kind" not in data
        assert "description" not in data

    def test_export_different_types_to_different_dirs(self, export_dir):
        for etype, eid in [("Organization", "org"), ("Protocol", "proto"), ("Project", "proj")]:
            entity = {
                "entity_id": f"{etype.lower()}:{eid}",
                "name": f"Test {etype}",
                "type": etype,
                "kind": None,
                "description": None,
                "aliases": "[]",
            }
            _export_yaml(entity, base_dir=export_dir)

        assert (Path(export_dir) / "organizations" / "org.yaml").exists()
        assert (Path(export_dir) / "protocols" / "proto.yaml").exists()
        assert (Path(export_dir) / "projects" / "proj.yaml").exists()

    def test_export_entity_with_colon_in_id(self, export_dir):
        entity = {
            "entity_id": "organization:multi-word-name",
            "name": "Multi Word",
            "type": "Organization",
            "kind": "company",
            "description": None,
            "aliases": "[]",
        }
        _export_yaml(entity, base_dir=export_dir)
        path = Path(export_dir) / "organizations" / "multi-word-name.yaml"
        assert path.exists()

    def test_export_unicode_content(self, export_dir):
        entity = {
            "entity_id": "organization:café-corp",
            "name": "Café Corporation",
            "type": "Organization",
            "kind": "company",
            "description": "A company with unicode: résumé, naïve, über",
            "aliases": json.dumps(["Café Corp"]),
        }
        _export_yaml(entity, base_dir=export_dir)
        path = Path(export_dir) / "organizations" / "café-corp.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "Café Corporation"
        assert "résumé" in data["description"]


# ── YAML round-trip ──────────────────────────────────────────────────────────

class TestYamlRoundTrip:
    def test_export_and_reimport(self, export_dir):
        """Export an entity to YAML, read it back, verify all fields match."""
        original = {
            "entity_id": "organization:roundtrip",
            "name": "RoundTrip Corp",
            "type": "Organization",
            "kind": "company",
            "description": "Tests the round-trip export/import cycle",
            "aliases": json.dumps(["RT Corp", "RTC"]),
        }
        _export_yaml(original, base_dir=export_dir)

        path = Path(export_dir) / "organizations" / "roundtrip.yaml"
        with open(path) as f:
            reimported = yaml.safe_load(f)

        assert reimported["id"] == original["entity_id"]
        assert reimported["name"] == original["name"]
        assert reimported["type"] == original["type"]
        assert reimported["kind"] == original["kind"]
        assert reimported["description"] == original["description"]
        assert reimported["aliases"] == json.loads(original["aliases"])

    def test_multiple_export_reimport(self, export_dir):
        entities = [
            {"entity_id": "protocol:rt-a", "name": "Proto A", "type": "Protocol",
             "kind": "spec", "description": "First", "aliases": json.dumps(["A"])},
            {"entity_id": "protocol:rt-b", "name": "Proto B", "type": "Protocol",
             "kind": "standard", "description": "Second", "aliases": json.dumps(["B"])},
        ]
        for ent in entities:
            _export_yaml(ent, base_dir=export_dir)

        proto_dir = Path(export_dir) / "protocols"
        reimported = []
        for yf in sorted(proto_dir.glob("*.yaml")):
            with open(yf) as f:
                reimported.append(yaml.safe_load(f))

        assert len(reimported) == 2
        names = {r["name"] for r in reimported}
        assert names == {"Proto A", "Proto B"}

    def test_overwrite_preserves_latest(self, export_dir):
        """Exporting the same entity_id twice should overwrite with latest data."""
        v1 = {
            "entity_id": "organization:overwrite",
            "name": "Version 1",
            "type": "Organization",
            "kind": "company",
            "description": "Original",
            "aliases": "[]",
        }
        _export_yaml(v1, base_dir=export_dir)

        v2 = {
            "entity_id": "organization:overwrite",
            "name": "Version 2",
            "type": "Organization",
            "kind": "company",
            "description": "Updated",
            "aliases": json.dumps(["V2"]),
        }
        _export_yaml(v2, base_dir=export_dir)

        path = Path(export_dir) / "organizations" / "overwrite.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "Version 2"
        assert data["description"] == "Updated"


# ── Cypher generation ────────────────────────────────────────────────────────

class TestCypherGeneration:
    def test_entity_cypher_valid_label(self):
        entity = {
            "entity_id": "organization:cypher-test",
            "name": "Cypher Test",
            "type": "Organization",
            "kind": "company",
            "description": "Test",
            "aliases": "[]",
            "source_id": 1,
        }
        query, params = _entity_to_cypher(entity)
        assert "MERGE" in query
        assert "Organization" in query
        assert params["entity_id"] == "organization:cypher-test"

    def test_entity_cypher_unknown_type_uses_entity(self):
        entity = {
            "entity_id": "custom:thing",
            "name": "Custom Thing",
            "type": "CustomType",
            "kind": None,
            "description": None,
            "aliases": "[]",
            "source_id": 1,
        }
        query, params = _entity_to_cypher(entity)
        assert ":Entity" in query
        assert "CustomType" not in query.split("SET")[0].split("REMOVE")[0]

    def test_edge_cypher_has_merge(self):
        edge = {
            "source_entity_id": "test:a",
            "target_entity_id": "test:b",
            "edge_id": "test-edge",
            "edge_type": "DEVELOPS",
            "confidence": 0.9,
            "source_type": "automated",
            "properties": "{}",
            "valid_from": None,
            "valid_to": None,
            "chunk_id": None,
        }
        query, params = _edge_to_cypher(edge)
        assert "MERGE" in query
        assert "DEVELOPS" in query
        assert params["src"] == "test:a"
        assert params["tgt"] == "test:b"

    def test_edge_cypher_with_properties(self):
        edge = {
            "source_entity_id": "test:c",
            "target_entity_id": "test:d",
            "edge_id": "prop-edge",
            "edge_type": "IMPLEMENTS",
            "confidence": 0.8,
            "source_type": "manual",
            "properties": json.dumps({"version": "2.0", "verified": True}),
            "valid_from": "2024-01-01",
            "valid_to": None,
            "chunk_id": 5,
        }
        query, params = _edge_to_cypher(edge)
        assert "prop_version" in params
        assert params["prop_version"] == "2.0"
        assert "IMPLEMENTS" in query

    def test_edge_cypher_temporal_fields(self):
        edge = {
            "source_entity_id": "test:x",
            "target_entity_id": "test:y",
            "edge_id": "temporal-edge",
            "edge_type": "MEMBER_OF",
            "confidence": 1.0,
            "source_type": "manual",
            "properties": "{}",
            "valid_from": "2023-06-01",
            "valid_to": "2024-12-31",
            "chunk_id": None,
        }
        query, params = _edge_to_cypher(edge)
        assert params["valid_from"] == "2023-06-01"
        assert params["valid_to"] == "2024-12-31"


# ── Existing YAML entity file validation ────────────────────────────────────

class TestExistingYamlFiles:
    @pytest.fixture
    def entity_dir(self):
        d = Path("/scion-volumes/scratchpad/agents-kg/kg/entities")
        if not d.exists():
            pytest.skip("kg/entities directory not found")
        return d

    def test_yaml_files_parseable(self, entity_dir):
        yaml_files = list(entity_dir.rglob("*.yaml"))
        if not yaml_files:
            pytest.skip("No YAML entity files found")
        for yf in yaml_files:
            with open(yf) as f:
                data = yaml.safe_load(f)
            assert data is not None, f"Empty or invalid YAML: {yf}"

    def test_yaml_files_have_required_fields(self, entity_dir):
        yaml_files = list(entity_dir.rglob("*.yaml"))
        if not yaml_files:
            pytest.skip("No YAML entity files found")
        for yf in yaml_files:
            with open(yf) as f:
                data = yaml.safe_load(f)
            if data is None:
                continue
            assert "id" in data, f"Missing 'id' in {yf}"
            assert "name" in data, f"Missing 'name' in {yf}"

    def test_yaml_entity_ids_unique(self, entity_dir):
        yaml_files = list(entity_dir.rglob("*.yaml"))
        if not yaml_files:
            pytest.skip("No YAML entity files found")
        ids = []
        for yf in yaml_files:
            with open(yf) as f:
                data = yaml.safe_load(f)
            if data and "id" in data:
                ids.append(data["id"])
        assert len(ids) == len(set(ids)), f"Duplicate entity IDs found: {[x for x in ids if ids.count(x) > 1]}"


# ── Database status summary ──────────────────────────────────────────────────

class TestStatusSummary:
    def test_empty_summary(self, db):
        assert db.status_summary() == {}

    def test_summary_counts(self, db):
        db.add_source("https://example.com/s1")
        db.add_source("https://example.com/s2")
        sid3 = db.add_source("https://example.com/s3")
        db.fail_source(sid3, "error")
        summary = db.status_summary()
        assert summary["pending"] == 2
        assert summary["failed"] == 1

    def test_summary_with_all_statuses(self, db):
        sid1 = db.add_source("https://example.com/sum1")
        sid2 = db.add_source("https://example.com/sum2")
        sid3 = db.add_source("https://example.com/sum3")
        sid4 = db.add_source("https://example.com/sum4")
        db.update_source(sid1, status="complete")
        db.fail_source(sid2, "err")
        db.update_source(sid3, status="processing")
        summary = db.status_summary()
        total = sum(summary.values())
        assert total == 4

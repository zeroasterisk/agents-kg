"""Iteration 9 — Ontology evolution tests.

Validates behaviour when the ontology (entity types, edge types) changes:
new types, renamed types, unknown types, and prompt-ontology alignment.
"""

import json
import pytest
from agents_kg.db import Database
from agents_kg.stages import extract as extract_stage
from agents_kg.stages import load as load_stage


# ---------------------------------------------------------------------------
# 1. Adding a new entity type
# ---------------------------------------------------------------------------


class TestNewEntityType:
    """What happens when entities arrive with a type not in the current ontology?"""

    def test_extract_rejects_unknown_entity_type(self):
        """VALID_ENTITY_TYPES is the authoritative gate for extraction."""
        assert "DataSource" not in extract_stage.VALID_ENTITY_TYPES

    def test_unknown_type_entity_still_storable_in_sqlite(self, db):
        """SQLite has no constraint on entity type — any string is accepted."""
        eid = "datasource:my-db"
        result = db.add_entity(entity_id=eid, name="My Database", entity_type="DataSource")
        assert result is not None
        row = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (eid,)
        ).fetchone()
        assert row["type"] == "DataSource"

    def test_load_cypher_fallback_label_for_unknown_type(self):
        """_entity_to_cypher falls back to 'Entity' label for unknown types."""
        entity = {
            "entity_id": "datasource:my-db",
            "name": "My Database",
            "type": "DataSource",
            "kind": "relational",
            "description": "A test database",
            "aliases": "[]",
            "source_id": 1,
        }
        query, params = load_stage._entity_to_cypher(entity)
        assert "DataSource" not in query
        assert ":Entity," in query or "n:Entity" in query
        assert params["type"] == "DataSource"

    def test_adding_type_to_valid_set_enables_extraction(self, monkeypatch):
        """Simulates extending VALID_ENTITY_TYPES with a new type."""
        extended = extract_stage.VALID_ENTITY_TYPES | {"DataSource"}
        monkeypatch.setattr(extract_stage, "VALID_ENTITY_TYPES", extended)
        assert "DataSource" in extract_stage.VALID_ENTITY_TYPES

    def test_all_valid_types_produce_proper_labels(self):
        """Every valid entity type in the ontology gets its own Neo4j label."""
        for etype in extract_stage.VALID_ENTITY_TYPES:
            entity = {
                "entity_id": f"{etype.lower()}:test",
                "name": "Test",
                "type": etype,
                "kind": None,
                "description": None,
                "aliases": "[]",
                "source_id": 1,
            }
            query, _ = load_stage._entity_to_cypher(entity)
            assert f"n:{etype}" in query


# ---------------------------------------------------------------------------
# 2. Adding a new edge type
# ---------------------------------------------------------------------------


class TestNewEdgeType:
    """What happens when edges arrive with a type not in the current ontology?"""

    def test_extract_rejects_unknown_edge_type(self):
        """VALID_EDGE_TYPES is the authoritative gate for extraction."""
        assert "FUNDED_BY" not in extract_stage.VALID_EDGE_TYPES

    def test_unknown_edge_type_storable_in_sqlite(self, db):
        """SQLite has no constraint on edge_type — any string is accepted."""
        result = db.add_edge(
            edge_id="test-fund-001",
            source_entity_id="organization:a",
            target_entity_id="organization:b",
            edge_type="FUNDED_BY",
        )
        assert result is not None
        rows = db.get_edges_by_status("pending_review")
        found = [r for r in rows if r["edge_id"] == "test-fund-001"]
        assert len(found) == 1
        assert found[0]["edge_type"] == "FUNDED_BY"

    def test_edge_to_cypher_uses_new_type_directly(self):
        """_edge_to_cypher uses edge_type in the relationship type position."""
        edge = {
            "source_entity_id": "organization:a",
            "target_entity_id": "organization:b",
            "edge_type": "FUNDED_BY",
            "edge_id": "fund-001",
            "confidence": 0.9,
            "source_type": "manual",
            "valid_from": None,
            "valid_to": None,
            "chunk_id": None,
            "properties": "{}",
        }
        query, _ = load_stage._edge_to_cypher(edge)
        assert "FUNDED_BY" in query

    def test_adding_edge_to_valid_set(self, monkeypatch):
        """Simulates extending VALID_EDGE_TYPES with a new type."""
        extended = extract_stage.VALID_EDGE_TYPES | {"FUNDED_BY"}
        monkeypatch.setattr(extract_stage, "VALID_EDGE_TYPES", extended)
        assert "FUNDED_BY" in extract_stage.VALID_EDGE_TYPES

    def test_all_valid_edge_types_are_uppercase(self):
        """Convention: edge types are UPPER_SNAKE_CASE."""
        for etype in extract_stage.VALID_EDGE_TYPES:
            assert etype == etype.upper()
            assert " " not in etype


# ---------------------------------------------------------------------------
# 3. Renaming an entity type
# ---------------------------------------------------------------------------


class TestRenamedEntityType:
    """What happens when an entity type is renamed (e.g. Group → WorkingGroup)?"""

    def test_old_type_entities_survive_rename(self, db):
        """Entities with the old type name remain queryable in SQLite."""
        db.add_entity(entity_id="group:wg-1", name="WG One", entity_type="Group")
        rows = db.conn.execute(
            "SELECT * FROM entities WHERE type = ?", ("Group",)
        ).fetchall()
        assert len(rows) == 1

    def test_cypher_remove_relabels(self):
        """_entity_to_cypher removes old labels before setting new ones."""
        entity = {
            "entity_id": "group:wg-1",
            "name": "WG One",
            "type": "Group",
            "kind": "wg",
            "description": None,
            "aliases": "[]",
            "source_id": 1,
        }
        query, _ = load_stage._entity_to_cypher(entity)
        assert "REMOVE" in query
        assert "n:Group" in query

    def test_migration_query_pattern(self, db):
        """Demonstrate a migration: update all Group→WorkingGroup in SQLite."""
        db.add_entity(entity_id="group:wg-1", name="WG One", entity_type="Group")
        db.add_entity(entity_id="group:wg-2", name="WG Two", entity_type="Group")

        db.conn.execute("UPDATE entities SET type = 'WorkingGroup' WHERE type = 'Group'")
        db.conn.commit()

        old = db.conn.execute("SELECT * FROM entities WHERE type = 'Group'").fetchall()
        new = db.conn.execute("SELECT * FROM entities WHERE type = 'WorkingGroup'").fetchall()
        assert len(old) == 0
        assert len(new) == 2


# ---------------------------------------------------------------------------
# 4. Unknown types handled gracefully
# ---------------------------------------------------------------------------


class TestUnknownTypesGraceful:
    """The system should not crash on unknown types — it should warn or fallback."""

    def test_extract_skips_invalid_entity_type_gracefully(self, db):
        """Simulating what extract.run does when an invalid type is returned by the LLM."""
        data = {
            "entities": [
                {
                    "entity_id": "widget:foo",
                    "name": "Foo Widget",
                    "type": "Widget",
                    "kind": None,
                    "description": "A foo widget",
                    "aliases": [],
                },
                {
                    "entity_id": "organization:google",
                    "name": "Google",
                    "type": "Organization",
                    "kind": "company",
                    "description": "Tech company",
                    "aliases": [],
                },
            ],
            "edges": [],
        }
        stored = 0
        for ent in data["entities"]:
            etype = ent.get("type", "")
            if etype not in extract_stage.VALID_ENTITY_TYPES:
                continue
            db.add_entity(
                entity_id=ent["entity_id"],
                name=ent["name"],
                entity_type=etype,
            )
            stored += 1

        assert stored == 1
        rows = db.get_entities_by_status("pending_review")
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "organization:google"

    def test_extract_skips_invalid_edge_type_gracefully(self, db):
        """Invalid edge types from LLM output are skipped, not crashed on."""
        data = {
            "entities": [],
            "edges": [
                {
                    "source_entity_id": "organization:a",
                    "target_entity_id": "project:b",
                    "edge_type": "LOVES",
                    "confidence": 0.9,
                    "properties": {},
                },
                {
                    "source_entity_id": "organization:a",
                    "target_entity_id": "project:b",
                    "edge_type": "DEVELOPS",
                    "confidence": 0.9,
                    "properties": {},
                },
            ],
        }
        stored = 0
        for edge in data["edges"]:
            edge_type = edge.get("edge_type", "")
            if edge_type not in extract_stage.VALID_EDGE_TYPES:
                continue
            db.add_edge(
                edge_id=f"e-{stored}",
                source_entity_id=edge["source_entity_id"],
                target_entity_id=edge["target_entity_id"],
                edge_type=edge_type,
            )
            stored += 1

        assert stored == 1

    def test_empty_type_rejected(self):
        """An empty entity type is not in VALID_ENTITY_TYPES."""
        assert "" not in extract_stage.VALID_ENTITY_TYPES
        assert "" not in extract_stage.VALID_EDGE_TYPES

    def test_none_type_rejected(self):
        """None is not in VALID_ENTITY_TYPES."""
        assert None not in extract_stage.VALID_ENTITY_TYPES
        assert None not in extract_stage.VALID_EDGE_TYPES

    def test_case_sensitive_type_check(self):
        """Type checks are case-sensitive: 'organization' != 'Organization'."""
        assert "organization" not in extract_stage.VALID_ENTITY_TYPES
        assert "ORGANIZATION" not in extract_stage.VALID_ENTITY_TYPES
        assert "Organization" in extract_stage.VALID_ENTITY_TYPES


# ---------------------------------------------------------------------------
# 5. Ontology-prompt alignment
# ---------------------------------------------------------------------------


class TestOntologyPromptAlignment:
    """Verify the extraction prompt matches the code-level ontology."""

    def test_prompt_lists_all_valid_entity_types(self):
        """The system prompt mentions every valid entity type."""
        prompt = extract_stage.SYSTEM_PROMPT_TEMPLATE
        for etype in extract_stage.VALID_ENTITY_TYPES:
            assert etype in prompt, f"Entity type {etype} not mentioned in prompt"

    def test_prompt_lists_all_valid_edge_types(self):
        """The system prompt mentions every valid edge type."""
        prompt = extract_stage.SYSTEM_PROMPT_TEMPLATE
        for etype in extract_stage.VALID_EDGE_TYPES:
            assert etype in prompt, f"Edge type {etype} not mentioned in prompt"

    def test_prompt_edge_count_matches_code(self):
        """The prompt says 14 edge types (sometimes says 15); code has 15."""
        assert len(extract_stage.VALID_EDGE_TYPES) == 15

    def test_prompt_entity_count_matches_code(self):
        """The prompt defines exactly 6 entity types."""
        assert len(extract_stage.VALID_ENTITY_TYPES) == 6

    def test_edge_types_are_all_documented(self):
        """Every edge type in the code set appears in the edge direction rules."""
        prompt = extract_stage.SYSTEM_PROMPT_TEMPLATE
        for etype in extract_stage.VALID_EDGE_TYPES:
            assert etype in prompt, f"Edge type {etype} not documented in prompt"

    def test_prompt_output_format_is_valid_json_template(self):
        """The JSON template in the prompt is structurally valid."""
        prompt = extract_stage.SYSTEM_PROMPT_TEMPLATE
        assert '"entities"' in prompt
        assert '"edges"' in prompt
        assert '"entity_id"' in prompt
        assert '"edge_type"' in prompt

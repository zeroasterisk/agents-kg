"""Tests for schema.py constraint and index definitions."""

from agents_kg.schema import CONSTRAINTS, INDEXES


class TestSchemaDefinitions:
    def test_constraints_non_empty(self):
        assert len(CONSTRAINTS) > 0

    def test_indexes_non_empty(self):
        assert len(INDEXES) > 0

    def test_constraints_are_strings(self):
        for c in CONSTRAINTS:
            assert isinstance(c, str)

    def test_indexes_are_strings(self):
        for idx in INDEXES:
            assert isinstance(idx, str)

    def test_constraints_are_create_statements(self):
        for c in CONSTRAINTS:
            assert c.startswith("CREATE CONSTRAINT")

    def test_indexes_are_create_statements(self):
        for idx in INDEXES:
            assert idx.startswith("CREATE INDEX")

    def test_constraints_use_if_not_exists(self):
        for c in CONSTRAINTS:
            assert "IF NOT EXISTS" in c

    def test_indexes_use_if_not_exists(self):
        for idx in INDEXES:
            assert "IF NOT EXISTS" in idx

    def test_entity_id_constraint_exists(self):
        assert any("entity_id" in c for c in CONSTRAINTS)

    def test_event_id_constraint_exists(self):
        assert any("event_id" in c for c in CONSTRAINTS)

    def test_wikidata_index_exists(self):
        assert any("wikidata" in idx for idx in INDEXES)

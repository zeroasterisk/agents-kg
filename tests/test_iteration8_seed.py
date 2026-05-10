"""Iteration 8 — Seed data completeness tests.

Validates that all seed entities are well-formed, have required fields,
use valid ontology types/kinds, have unique entity_ids, and unique aliases.
"""

import re
from collections import Counter

import pytest

from agents_kg.seed import SEED_ENTITIES, get_seed_entities, seed_entity_ids, format_seed_for_prompt
from agents_kg.stages.extract import VALID_ENTITY_TYPES


VALID_KINDS = {
    "Organization": {"company", "standards_body", "foundation", "consortium"},
    "Group": {"tsc", "wg", "sig", "task_force", "team"},
    "Person": set(),
    "Project": {"framework", "sdk", "library", "tool", "platform"},
    "Protocol": {"spec", "standard", "rfc", "draft"},
    "Capability": set(),
}


# ---------------------------------------------------------------------------
# 1. Required fields
# ---------------------------------------------------------------------------

class TestSeedRequiredFields:

    def test_every_seed_has_entity_id(self):
        for ent in SEED_ENTITIES:
            assert "entity_id" in ent and ent["entity_id"], \
                f"Seed missing entity_id: {ent}"

    def test_every_seed_has_name(self):
        for ent in SEED_ENTITIES:
            assert "name" in ent and ent["name"], \
                f"Seed {ent.get('entity_id')} missing name"

    def test_every_seed_has_type(self):
        for ent in SEED_ENTITIES:
            assert "type" in ent and ent["type"], \
                f"Seed {ent.get('entity_id')} missing type"

    def test_every_seed_has_aliases(self):
        for ent in SEED_ENTITIES:
            assert "aliases" in ent, \
                f"Seed {ent.get('entity_id')} missing aliases field"
            assert isinstance(ent["aliases"], list), \
                f"Seed {ent.get('entity_id')} aliases should be a list"


# ---------------------------------------------------------------------------
# 2. No duplicate entity_ids
# ---------------------------------------------------------------------------

class TestSeedUniqueness:

    def test_no_duplicate_entity_ids(self):
        ids = [ent["entity_id"] for ent in SEED_ENTITIES]
        counts = Counter(ids)
        dupes = {eid: cnt for eid, cnt in counts.items() if cnt > 1}
        assert not dupes, f"Duplicate entity_ids in seed data: {dupes}"

    def test_seed_entity_ids_set_matches(self):
        """seed_entity_ids() returns the right count."""
        ids_set = seed_entity_ids()
        assert len(ids_set) == len(SEED_ENTITIES)

    def test_get_seed_entities_returns_all(self):
        """get_seed_entities() returns same list."""
        result = get_seed_entities()
        assert len(result) == len(SEED_ENTITIES)


# ---------------------------------------------------------------------------
# 3. Valid ontology types
# ---------------------------------------------------------------------------

class TestSeedOntologyTypes:

    def test_seed_types_are_valid(self):
        for ent in SEED_ENTITIES:
            assert ent["type"] in VALID_ENTITY_TYPES, \
                f"Seed {ent['entity_id']} has invalid type: {ent['type']}"

    def test_seed_entity_id_prefix_matches_type(self):
        """entity_id prefix should correspond to type."""
        type_to_prefix = {
            "Organization": "organization",
            "Group": "group",
            "Person": "person",
            "Project": "project",
            "Protocol": "protocol",
            "Capability": "capability",
        }
        for ent in SEED_ENTITIES:
            prefix = ent["entity_id"].split(":")[0]
            expected = type_to_prefix.get(ent["type"])
            assert prefix == expected, \
                f"Seed {ent['entity_id']} prefix '{prefix}' doesn't match type '{ent['type']}' (expected '{expected}')"


# ---------------------------------------------------------------------------
# 4. Valid kind values for their type
# ---------------------------------------------------------------------------

class TestSeedKindValues:

    def test_seed_kinds_are_valid_for_type(self):
        for ent in SEED_ENTITIES:
            etype = ent["type"]
            kind = ent.get("kind")
            valid = VALID_KINDS.get(etype, set())
            if not valid:
                continue
            if kind is not None:
                assert kind in valid, \
                    f"Seed {ent['entity_id']} has invalid kind '{kind}' for type '{etype}'. Valid: {valid}"

    def test_organizations_have_kind(self):
        for ent in SEED_ENTITIES:
            if ent["type"] == "Organization":
                assert ent.get("kind"), \
                    f"Organization {ent['entity_id']} missing kind"

    def test_groups_have_kind(self):
        for ent in SEED_ENTITIES:
            if ent["type"] == "Group":
                assert ent.get("kind"), \
                    f"Group {ent['entity_id']} missing kind"

    def test_protocols_have_kind(self):
        for ent in SEED_ENTITIES:
            if ent["type"] == "Protocol":
                assert ent.get("kind"), \
                    f"Protocol {ent['entity_id']} missing kind"

    def test_projects_have_kind(self):
        for ent in SEED_ENTITIES:
            if ent["type"] == "Project":
                assert ent.get("kind"), \
                    f"Project {ent['entity_id']} missing kind"


# ---------------------------------------------------------------------------
# 5. Unique aliases
# ---------------------------------------------------------------------------

class TestSeedAliases:

    def test_no_alias_appears_in_multiple_entities(self):
        """Each alias should map to at most one entity."""
        alias_to_entities = {}
        for ent in SEED_ENTITIES:
            for alias in ent.get("aliases", []):
                normalized = alias.lower().strip()
                if not normalized:
                    continue
                alias_to_entities.setdefault(normalized, []).append(ent["entity_id"])

        conflicts = {
            alias: eids for alias, eids in alias_to_entities.items()
            if len(eids) > 1
        }
        assert not conflicts, f"Aliases shared by multiple entities: {conflicts}"

    def test_no_alias_duplicates_within_entity(self):
        """No entity has duplicate aliases."""
        for ent in SEED_ENTITIES:
            aliases = ent.get("aliases", [])
            normalized = [a.lower().strip() for a in aliases]
            counts = Counter(normalized)
            dupes = {a: c for a, c in counts.items() if c > 1}
            assert not dupes, \
                f"Seed {ent['entity_id']} has duplicate aliases: {dupes}"

    def test_name_not_in_own_aliases(self):
        """An entity's name should not appear in its own aliases."""
        for ent in SEED_ENTITIES:
            name_lower = ent["name"].lower().strip()
            aliases_lower = [a.lower().strip() for a in ent.get("aliases", [])]
            assert name_lower not in aliases_lower, \
                f"Seed {ent['entity_id']} has its own name '{ent['name']}' in aliases"


# ---------------------------------------------------------------------------
# 6. Entity ID format
# ---------------------------------------------------------------------------

class TestSeedIDFormat:

    def test_entity_id_has_colon(self):
        for ent in SEED_ENTITIES:
            assert ":" in ent["entity_id"], \
                f"Entity ID missing colon: {ent['entity_id']}"

    def test_entity_id_is_kebab_case(self):
        """The part after the colon should be kebab-case."""
        for ent in SEED_ENTITIES:
            suffix = ent["entity_id"].split(":", 1)[1]
            assert re.match(r'^[a-z0-9][a-z0-9\-\.]*$', suffix), \
                f"Entity ID suffix not kebab-case: {ent['entity_id']}"

    def test_entity_id_no_trailing_hyphen(self):
        for ent in SEED_ENTITIES:
            assert not ent["entity_id"].endswith("-"), \
                f"Entity ID has trailing hyphen: {ent['entity_id']}"


# ---------------------------------------------------------------------------
# 7. Prompt formatting
# ---------------------------------------------------------------------------

class TestSeedPromptFormatting:

    def test_format_seed_for_prompt_not_empty(self):
        result = format_seed_for_prompt()
        assert len(result) > 100

    def test_format_seed_includes_all_types(self):
        result = format_seed_for_prompt()
        for etype in ["Organization", "Protocol", "Project", "Capability", "Person"]:
            assert etype in result, f"Prompt missing type: {etype}"

    def test_format_seed_includes_entity_ids(self):
        """Prompt contains actual entity_ids."""
        result = format_seed_for_prompt()
        assert "organization:google" in result
        assert "protocol:mcp" in result
        assert "project:langchain" in result

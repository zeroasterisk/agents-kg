"""Tests for wikidata_crossref.py mapping file parsing (no Neo4j required)."""

import re
from pathlib import Path

from agents_kg.wikidata_crossref import load_mappings

MAPPINGS_PATH = Path(__file__).resolve().parent.parent / "kg" / "wikidata_mappings.yaml"


class TestLoadMappings:
    def test_loads_mappings_file(self):
        mappings = load_mappings(str(MAPPINGS_PATH))
        assert len(mappings) > 0

    def test_keys_are_entity_ids(self):
        mappings = load_mappings(str(MAPPINGS_PATH))
        for key in mappings:
            assert ":" in key, f"entity_id should have type:name format, got: {key}"

    def test_qids_are_well_formed(self):
        """Non-null Q-IDs must match Q followed by digits."""
        mappings = load_mappings(str(MAPPINGS_PATH))
        for entity_id, qid in mappings.items():
            if qid is None:
                continue
            qid_str = str(qid)
            if not qid_str.startswith("Q"):
                qid_str = f"Q{qid_str}"
            assert re.match(r"^Q\d+$", qid_str), \
                f"Bad Q-ID '{qid}' for {entity_id}"

    def test_has_known_entities(self):
        mappings = load_mappings(str(MAPPINGS_PATH))
        assert "organization:google" in mappings
        assert "organization:openai" in mappings

    def test_nonexistent_file_returns_empty(self, tmp_path):
        mappings = load_mappings(str(tmp_path / "nope.yaml"))
        assert mappings == {}

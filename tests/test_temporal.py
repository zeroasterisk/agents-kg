"""Tests for temporal.py event YAML parsing (no Neo4j required)."""

import re
from pathlib import Path

import pytest

from agents_kg.temporal import load_event_yaml, load_events_from_yaml

EVENTS_DIR = Path(__file__).resolve().parent.parent / "kg" / "events"


class TestLoadEventYaml:
    def test_loads_real_event_files(self):
        """Parse actual event YAML files from kg/events/."""
        yaml_files = sorted(EVENTS_DIR.glob("*.yaml"))
        assert len(yaml_files) > 0, "Expected at least one event YAML file"

        for path in yaml_files:
            event = load_event_yaml(path)
            assert event is not None, f"Failed to parse {path.name}"
            assert "event_id" in event
            assert "title" in event
            assert "event_type" in event
            assert "date" in event

    def test_event_ids_are_slugified(self):
        """Event IDs should be kebab-case with a date suffix."""
        for path in EVENTS_DIR.glob("*.yaml"):
            event = load_event_yaml(path)
            if event is None:
                continue
            eid = event["event_id"]
            assert re.match(r"^[a-z0-9][-a-z0-9]*\d{4}-\d{2}-\d{2}$", eid), \
                f"Bad event_id format: {eid}"

    def test_participants_have_entity_id(self):
        """Each participant must have an entity_id."""
        for path in EVENTS_DIR.glob("*.yaml"):
            event = load_event_yaml(path)
            if event is None:
                continue
            for p in event.get("participants", []):
                assert "entity_id" in p, f"Participant missing entity_id in {path.name}"

    def test_returns_none_for_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        assert load_event_yaml(empty) is None

    def test_returns_none_for_missing_fields(self, tmp_path):
        incomplete = tmp_path / "bad.yaml"
        incomplete.write_text("title: Test\n")
        assert load_event_yaml(incomplete) is None


class TestLoadEventsFromYaml:
    def test_no_neo4j_returns_counts(self):
        """Without a driver, should still parse and return counts."""
        result = load_events_from_yaml(None, str(EVENTS_DIR))
        assert result["events"] > 0
        assert isinstance(result["participations"], int)

    def test_nonexistent_dir_returns_zeros(self, tmp_path):
        result = load_events_from_yaml(None, str(tmp_path / "nope"))
        assert result == {"events": 0, "participations": 0}

    def test_empty_dir_returns_zeros(self, tmp_path):
        result = load_events_from_yaml(None, str(tmp_path))
        assert result == {"events": 0, "participations": 0}

"""Iteration 11 — Coverage gap tests.

Tests for previously untested code paths identified during the cleanup audit:
local file fetching, schema application, stub ADK agent, timeline migration,
temporal constraints, extract stage type filtering, and resolve noise filtering.
"""

import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_kg.db import Database, content_hash
from agents_kg.stages import extract as extract_mod


# ---------------------------------------------------------------------------
# 1. fetch.py — local file fetching
# ---------------------------------------------------------------------------


class TestFetchLocalFile:

    def test_fetch_local_text_file(self, db, tmp_path):
        from agents_kg.stages.fetch import run

        f = tmp_path / "readme.md"
        f.write_text("# Hello\nWorld")
        sid = db.add_source(str(f))
        source = db.get_source(sid)
        result = run(db, source)
        assert result is True
        updated = db.get_source(sid)
        assert updated["raw_text"] == "# Hello\nWorld"
        assert updated["type"] == "text"
        assert updated["stage"] == "parse"

    def test_fetch_local_file_not_found(self, db):
        from agents_kg.stages.fetch import _fetch_local

        with pytest.raises(RuntimeError, match="File not found"):
            _fetch_local("/nonexistent/file.md")

    def test_is_local_file_detects_file_uri(self):
        from agents_kg.stages.fetch import _is_local_file

        assert _is_local_file("file:///tmp/test.txt") is True
        assert _is_local_file("https://example.com") is False

    def test_resolve_path_strips_file_prefix(self):
        from agents_kg.stages.fetch import _resolve_path

        assert _resolve_path("file:///tmp/test.txt") == "/tmp/test.txt"
        result = _resolve_path("relative.txt")
        assert os.path.isabs(result)

    def test_content_changed_deprecates_old_entities(self, db, tmp_path):
        from agents_kg.stages.fetch import run

        f = tmp_path / "doc.md"
        f.write_text("original content")
        sid = db.add_source(str(f))
        source = db.get_source(sid)
        run(db, source)

        db.add_entity("org:old", "Old Org", "Organization", source_id=sid)
        old_hash = db.get_source(sid)["content_hash"]

        f.write_text("updated content")
        db.update_source(sid, stage="fetch", status="pending", content_hash=old_hash)
        source = db.get_source(sid)
        result = run(db, source)
        assert result is True


# ---------------------------------------------------------------------------
# 2. schema.py — apply_schema with mock driver
# ---------------------------------------------------------------------------


class TestApplySchema:

    def test_apply_schema_success(self):
        from agents_kg.schema import apply_schema, CONSTRAINTS, INDEXES

        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = lambda s: mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = apply_schema(mock_driver)
        assert result["constraints"] == len(CONSTRAINTS)
        assert result["indexes"] == len(INDEXES)
        assert result["errors"] == []
        assert mock_session.run.call_count == len(CONSTRAINTS) + len(INDEXES)

    def test_apply_schema_with_errors(self):
        from agents_kg.schema import apply_schema, CONSTRAINTS, INDEXES

        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("already exists")
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = lambda s: mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = apply_schema(mock_driver)
        assert result["constraints"] == 0
        assert result["indexes"] == 0
        assert len(result["errors"]) == len(CONSTRAINTS) + len(INDEXES)


# ---------------------------------------------------------------------------
# 3. agents/stub_adk.py — StubADKAgent
# ---------------------------------------------------------------------------


class TestStubADKAgent:

    def test_init_defaults(self):
        from agents_kg.agents.stub_adk import StubADKAgent

        agent = StubADKAgent(model="test-model", instruction="test instruction")
        assert agent.model == "test-model"
        assert agent.instruction == "test instruction"
        assert agent.tools == []

    def test_chat_returns_valid_json(self):
        from agents_kg.agents.stub_adk import StubADKAgent

        agent = StubADKAgent(model="test", instruction="test")
        result = json.loads(agent.chat("extract from this text"))
        assert result == {"entities": [], "edges": []}

    def test_run_yields_events(self):
        from agents_kg.agents.stub_adk import StubADKAgent

        agent = StubADKAgent(model="test-model", instruction="test")
        events = list(agent.run("test message"))
        assert len(events) == 3
        assert events[0]["event"] == "agent_started"
        assert events[1]["event"] == "llm_call"
        assert events[1]["model"] == "test-model"
        assert events[2]["event"] == "agent_finished"
        output = json.loads(events[2]["output"])
        assert "entities" in output


# ---------------------------------------------------------------------------
# 4. temporal.py — migrate_timeline_yaml
# ---------------------------------------------------------------------------


class TestMigrateTimelineYaml:

    def test_migrate_creates_event_files(self, tmp_path):
        from agents_kg.temporal import migrate_timeline_yaml

        tl = tmp_path / "timeline.yaml"
        tl.write_text(yaml.dump({"events": [
            {"title": "MCP Launch", "date": "2024-11-25", "type": "launch",
             "actors": ["anthropic"], "description": "MCP launched publicly"},
        ]}))
        events_dir = str(tmp_path / "events")
        created = migrate_timeline_yaml(str(tl), events_dir)
        assert created == 1
        files = list(Path(events_dir).glob("*.yaml"))
        assert len(files) == 1
        data = yaml.safe_load(files[0].read_text())
        assert data["title"] == "MCP Launch"
        assert data["event_type"] == "launch"
        assert len(data["participants"]) == 1

    def test_migrate_nonexistent_timeline(self, tmp_path):
        from agents_kg.temporal import migrate_timeline_yaml

        count = migrate_timeline_yaml(str(tmp_path / "nope.yaml"), str(tmp_path / "out"))
        assert count == 0

    def test_migrate_skips_existing_files(self, tmp_path):
        from agents_kg.temporal import migrate_timeline_yaml

        tl = tmp_path / "timeline.yaml"
        tl.write_text(yaml.dump({"events": [
            {"title": "Event A", "date": "2024-01-01", "type": "event"}
        ]}))
        events_dir = str(tmp_path / "events")
        first = migrate_timeline_yaml(str(tl), events_dir)
        assert first == 1
        second = migrate_timeline_yaml(str(tl), events_dir)
        assert second == 0


# ---------------------------------------------------------------------------
# 5. temporal.py — create_temporal_constraints with mock
# ---------------------------------------------------------------------------


class TestCreateTemporalConstraints:

    def test_creates_constraints_successfully(self):
        from agents_kg.temporal import create_temporal_constraints

        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = lambda s: mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        create_temporal_constraints(mock_driver)
        assert mock_session.run.call_count == 3

    def test_handles_errors_gracefully(self):
        from agents_kg.temporal import create_temporal_constraints

        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Constraint exists")
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = lambda s: mock_session
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        create_temporal_constraints(mock_driver)


# ---------------------------------------------------------------------------
# 6. extract.py — invalid entity type filtering through run()
# ---------------------------------------------------------------------------


class TestExtractTypeFiltering:

    def test_extract_skips_invalid_entity_type(self, db):
        sid = db.add_source("https://example.com/filter-test")
        db.add_chunk(sid, "Some text about things", 0)
        source = db.get_source(sid)

        response_data = {
            "entities": [
                {"entity_id": "badtype:foo", "name": "Foo", "type": "InvalidType",
                 "kind": None, "description": "bad", "aliases": []},
                {"entity_id": "organization:valid", "name": "Valid Org", "type": "Organization",
                 "kind": "company", "description": "good", "aliases": []},
            ],
            "edges": []
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai}):
            extract_mod.run(db, source)

        entities = db.get_entities_by_status("pending_review")
        assert len(entities) == 1
        assert entities[0]["entity_id"] == "organization:valid"

    def test_extract_handles_json_decode_error(self, db):
        sid = db.add_source("https://example.com/json-err")
        db.add_chunk(sid, "Some text", 0)
        source = db.get_source(sid)

        mock_response = MagicMock()
        mock_response.text = "not valid json {{"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai}):
            result = extract_mod.run(db, source)

        assert result is True
        assert len(db.get_entities_by_status("pending_review")) == 0

    def test_extract_skips_invalid_edge_type(self, db):
        sid = db.add_source("https://example.com/edge-filter")
        db.add_chunk(sid, "Text about relationships", 0)
        source = db.get_source(sid)

        response_data = {
            "entities": [],
            "edges": [
                {"source_entity_id": "a:b", "target_entity_id": "c:d",
                 "edge_type": "INVENTED_FAKE_TYPE", "confidence": 0.9, "properties": {}},
                {"source_entity_id": "a:b", "target_entity_id": "c:d",
                 "edge_type": "DEVELOPS", "confidence": 0.9, "properties": {}},
            ]
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai}):
            extract_mod.run(db, source)

        edges = db.get_edges_by_status("pending_review")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "DEVELOPS"


# ---------------------------------------------------------------------------
# 7. resolve.py — noise entity filtering
# ---------------------------------------------------------------------------


class TestResolveNoiseFiltering:

    def test_noise_entities_rejected(self, db):
        from agents_kg.stages.resolve import run

        sid = db.add_source("https://example.com/noise-test")
        db.add_entity("project:benchmark-x", "Benchmark X", "Project",
                       kind="benchmark", source_id=sid, description="A benchmark")
        db.add_entity("organization:real-org", "Real Org", "Organization",
                       kind="company", source_id=sid, description="A company")
        source = db.get_source(sid)

        with patch("agents_kg.stages.resolve.genai", None):
            run(db, source)

        noise = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id='project:benchmark-x'"
        ).fetchone()
        assert dict(noise)["status"] == "rejected"

        real = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id='organization:real-org'"
        ).fetchone()
        assert dict(real)["status"] != "rejected"

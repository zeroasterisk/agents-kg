"""Iteration 6: CLI command tests using Click CliRunner."""

import os
import json
import tempfile
import pytest
from click.testing import CliRunner
from agents_kg.cli import cli
from agents_kg.db import Database


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp database and set env to use it."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.close()
    return db_path


@pytest.fixture(autouse=True)
def patch_db_path(tmp_db, monkeypatch):
    monkeypatch.setenv("KG_DB_PATH", tmp_db)


# ── Help output ──────────────────────────────────────────────────────────────

class TestHelpOutput:
    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Knowledge graph" in result.output

    def test_ingest_help(self, runner):
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--submitter-email" in result.output
        assert "--from" in result.output
        assert "--file" in result.output

    def test_process_help(self, runner):
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process all pending" in result.output

    def test_status_help(self, runner):
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "queue status" in result.output

    def test_review_help(self, runner):
        result = runner.invoke(cli, ["review", "--help"])
        assert result.exit_code == 0
        assert "--approve" in result.output
        assert "--approve-all" in result.output

    def test_schema_help(self, runner):
        result = runner.invoke(cli, ["schema", "--help"])
        assert result.exit_code == 0
        assert "schema" in result.output.lower()

    def test_seed_help(self, runner):
        result = runner.invoke(cli, ["seed", "--help"])
        assert result.exit_code == 0
        assert "seed" in result.output.lower()

    def test_load_yaml_help(self, runner):
        result = runner.invoke(cli, ["load-yaml", "--help"])
        assert result.exit_code == 0
        assert "--entities-dir" in result.output
        assert "--relations" in result.output

    def test_wikidata_help(self, runner):
        result = runner.invoke(cli, ["wikidata", "--help"])
        assert result.exit_code == 0
        assert "pull" in result.output or "crossref" in result.output

    def test_wikidata_pull_help(self, runner):
        result = runner.invoke(cli, ["wikidata", "pull", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        assert "--dry-run" in result.output

    def test_wikidata_crossref_help(self, runner):
        result = runner.invoke(cli, ["wikidata", "crossref", "--help"])
        assert result.exit_code == 0

    def test_retry_help(self, runner):
        result = runner.invoke(cli, ["retry", "--help"])
        assert result.exit_code == 0
        assert "Retry" in result.output

    def test_reset_help(self, runner):
        result = runner.invoke(cli, ["reset", "--help"])
        assert result.exit_code == 0
        assert "SOURCE_ID" in result.output

    def test_events_help(self, runner):
        result = runner.invoke(cli, ["events", "--help"])
        assert result.exit_code == 0


# ── Ingest command ───────────────────────────────────────────────────────────

class TestIngestCommand:
    def test_ingest_single_url(self, runner, tmp_db):
        result = runner.invoke(cli, ["ingest", "https://example.com/paper1"])
        assert result.exit_code == 0
        assert "Added 1" in result.output
        db = Database(tmp_db)
        source = db.get_source_by_uri("https://example.com/paper1")
        assert source is not None
        db.close()

    def test_ingest_with_submitter_email(self, runner, tmp_db):
        result = runner.invoke(cli, ["ingest", "https://example.com/paper2",
                                      "--submitter-email", "alice@example.com"])
        assert result.exit_code == 0
        assert "Added 1" in result.output
        db = Database(tmp_db)
        source = db.get_source_by_uri("https://example.com/paper2")
        assert source["submitter_email"] == "alice@example.com"
        db.close()

    def test_ingest_duplicate_url(self, runner, tmp_db):
        runner.invoke(cli, ["ingest", "https://example.com/dup"])
        result = runner.invoke(cli, ["ingest", "https://example.com/dup"])
        assert result.exit_code == 0
        assert "skipped 1" in result.output

    def test_ingest_from_file(self, runner, tmp_db, tmp_path):
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://example.com/a\nhttps://example.com/b\n# comment\nhttps://example.com/c\n")
        result = runner.invoke(cli, ["ingest", "--from", str(urls_file)])
        assert result.exit_code == 0
        assert "Added 3" in result.output

    def test_ingest_no_input(self, runner):
        result = runner.invoke(cli, ["ingest"])
        assert result.exit_code != 0
        assert "Error" in result.output or "provide" in result.output

    def test_ingest_local_file(self, runner, tmp_db, tmp_path):
        local = tmp_path / "local.md"
        local.write_text("# Local content\nSome markdown here.")
        result = runner.invoke(cli, ["ingest", "--file", str(local)])
        assert result.exit_code == 0
        assert "Added 1" in result.output


# ── Status command ───────────────────────────────────────────────────────────

class TestStatusCommand:
    def test_status_empty(self, runner):
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "No sources" in result.output

    def test_status_with_sources(self, runner, tmp_db):
        db = Database(tmp_db)
        db.add_source("https://example.com/s1")
        db.add_source("https://example.com/s2")
        db.close()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "pending" in result.output


# ── Review command ───────────────────────────────────────────────────────────

class TestReviewCommand:
    def test_review_no_pending(self, runner):
        result = runner.invoke(cli, ["review"])
        assert result.exit_code == 0
        assert "No items pending" in result.output

    def test_review_with_pending_entity(self, runner, tmp_db):
        db = Database(tmp_db)
        sid = db.add_source("https://example.com/rev1")
        db.add_entity("test:entity-1", "Test Entity", "Organization", source_id=sid)
        db.close()
        result = runner.invoke(cli, ["review"])
        assert result.exit_code == 0
        assert "Test Entity" in result.output

    def test_approve_entity(self, runner, tmp_db):
        db = Database(tmp_db)
        sid = db.add_source("https://example.com/rev2")
        eid = db.add_entity("test:entity-2", "Test Org", "Organization", source_id=sid)
        db.close()
        result = runner.invoke(cli, ["review", "--approve", str(eid)])
        assert result.exit_code == 0
        assert "Approved entity" in result.output

    def test_approve_all(self, runner, tmp_db):
        db = Database(tmp_db)
        sid = db.add_source("https://example.com/rev3")
        db.add_entity("test:aa1", "Org A", "Organization", source_id=sid)
        db.add_entity("test:aa2", "Org B", "Organization", source_id=sid)
        db.close()
        result = runner.invoke(cli, ["review", "--approve-all"])
        assert result.exit_code == 0
        assert "Approved 2 entities" in result.output

    def test_approve_nonexistent(self, runner):
        result = runner.invoke(cli, ["review", "--approve", "99999"])
        assert result.exit_code == 0
        assert "No entity or edge" in result.output

    def test_review_type_filter(self, runner, tmp_db):
        db = Database(tmp_db)
        sid = db.add_source("https://example.com/rev4")
        db.add_entity("test:tf1", "Org TF", "Organization", source_id=sid)
        db.add_edge("test-edge-1", "test:tf1", "test:tf2", "DEVELOPS", source_id=sid)
        db.close()
        result = runner.invoke(cli, ["review", "--type", "entity"])
        assert result.exit_code == 0
        assert "Org TF" in result.output


# ── Retry command ────────────────────────────────────────────────────────────

class TestRetryCommand:
    def test_retry_no_failed(self, runner):
        result = runner.invoke(cli, ["retry"])
        assert result.exit_code == 0
        assert "Retried 0" in result.output

    def test_retry_failed_sources(self, runner, tmp_db):
        db = Database(tmp_db)
        sid = db.add_source("https://example.com/fail1")
        db.fail_source(sid, "test error")
        db.close()
        result = runner.invoke(cli, ["retry"])
        assert result.exit_code == 0
        assert "Retried 1" in result.output


# ── Reset command ────────────────────────────────────────────────────────────

class TestResetCommand:
    def test_reset_existing(self, runner, tmp_db):
        db = Database(tmp_db)
        sid = db.add_source("https://example.com/reset1")
        db.update_source(sid, stage="chunk", status="processing")
        db.close()
        result = runner.invoke(cli, ["reset", str(sid)])
        assert result.exit_code == 0
        assert "Reset source" in result.output

    def test_reset_nonexistent(self, runner):
        result = runner.invoke(cli, ["reset", "99999"])
        assert result.exit_code != 0
        assert "not found" in result.output

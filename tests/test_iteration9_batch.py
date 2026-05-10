"""Iteration 9 — Batch operations tests.

Validates bulk workflows: ingesting from a file, processing all pending,
batch-loading to Neo4j, approving all, and retrying all failed.
"""

import json
import os
import tempfile
import pytest
from click.testing import CliRunner
from agents_kg.cli import cli
from agents_kg.db import Database


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


@pytest.fixture
def cli_env(tmp_path):
    """Create a temp DB and return env dict for CLI runner."""
    db_path = str(tmp_path / "test.db")
    return {"KG_DB_PATH": db_path}


# ---------------------------------------------------------------------------
# 1. Ingest from file (--from flag)
# ---------------------------------------------------------------------------


class TestBulkIngest:
    """Ingest multiple sources from a URL list file."""

    def test_ingest_from_file(self, tmp_path, cli_env):
        """--from flag reads one URL per line and ingests all."""
        url_file = tmp_path / "urls.txt"
        urls = [f"https://example.com/doc{i}" for i in range(10)]
        url_file.write_text("\n".join(urls))

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--from", str(url_file)], env=cli_env)
        assert result.exit_code == 0
        assert "Added 10" in result.output

    def test_ingest_from_file_skips_comments_and_blank_lines(self, tmp_path, cli_env):
        """Lines starting with # and blank lines are skipped."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "# This is a comment\n"
            "https://example.com/doc1\n"
            "\n"
            "# Another comment\n"
            "https://example.com/doc2\n"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--from", str(url_file)], env=cli_env)
        assert result.exit_code == 0
        assert "Added 2" in result.output

    def test_ingest_from_file_deduplicates(self, tmp_path, cli_env):
        """Duplicate URLs in the file are skipped."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com/doc1\n"
            "https://example.com/doc1\n"
            "https://example.com/doc2\n"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--from", str(url_file)], env=cli_env)
        assert result.exit_code == 0
        assert "Added 2" in result.output
        assert "skipped 1" in result.output

    def test_ingest_from_file_large_batch(self, tmp_path, cli_env):
        """Can ingest a larger batch (50 URLs)."""
        url_file = tmp_path / "urls.txt"
        urls = [f"https://example.com/article/{i}" for i in range(50)]
        url_file.write_text("\n".join(urls))

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--from", str(url_file)], env=cli_env)
        assert result.exit_code == 0
        assert "Added 50" in result.output

    def test_ingest_no_args_errors(self, cli_env):
        """Calling ingest with no URL, --file, or --from errors."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest"], env=cli_env)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 2. Process all pending sources
# ---------------------------------------------------------------------------


class TestProcessAllPending:
    """Process all pending sources in one go."""

    def test_process_all_pending_db_level(self, db):
        """Database correctly returns all pending sources."""
        for i in range(5):
            db.add_source(f"https://example.com/pending-{i}")

        pending = db.get_pending_sources()
        assert len(pending) == 5

    def test_pending_includes_processing_status(self, db):
        """get_pending_sources includes both 'pending' and 'processing' statuses."""
        s1 = db.add_source("https://example.com/p1")
        s2 = db.add_source("https://example.com/p2")
        db.update_source(s2, status="processing")

        pending = db.get_pending_sources()
        assert len(pending) == 2

    def test_completed_not_in_pending(self, db):
        """Completed sources are not returned by get_pending_sources."""
        s1 = db.add_source("https://example.com/done")
        db.update_source(s1, status="complete")

        pending = db.get_pending_sources()
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# 3. Batch load approved entities to Neo4j
# ---------------------------------------------------------------------------


class TestBatchLoadApproved:
    """Load all approved entities in one batch."""

    def test_batch_approve_and_query(self, db):
        """After approving entities, they appear in the approved status."""
        source_id = db.add_source("https://example.com/batch-load")
        for i in range(10):
            db.add_entity(
                entity_id=f"organization:batch-{i}",
                name=f"Batch Org {i}",
                entity_type="Organization",
                source_id=source_id,
            )

        pending = db.get_entities_by_status("pending_review")
        assert len(pending) == 10

        for ent in pending:
            db.approve_entity(ent["id"])

        approved = db.get_entities_by_status("approved")
        assert len(approved) == 10

    def test_batch_load_generates_correct_cypher(self, db):
        """Loading N approved entities generates N Cypher statements."""
        source_id = db.add_source("https://example.com/batch-cypher")
        for i in range(5):
            eid = db.add_entity(
                entity_id=f"project:batch-p-{i}",
                name=f"Project {i}",
                entity_type="Project",
                kind="tool",
                source_id=source_id,
            )
            db.approve_entity(eid)

        from agents_kg.stages.load import _entity_to_cypher
        approved = db.get_entities_by_status("approved")
        queries = [_entity_to_cypher(dict(e)) for e in approved]
        assert len(queries) == 5
        for query, params in queries:
            assert "$entity_id" in query
            assert params["type"] == "Project"


# ---------------------------------------------------------------------------
# 4. Approve all pending items
# ---------------------------------------------------------------------------


class TestApproveAll:
    """Approve all pending entities and edges in one command."""

    def test_approve_all_entities_cli(self, tmp_path, cli_env):
        """CLI --approve-all approves all pending entities."""
        db_path = cli_env["KG_DB_PATH"]
        db = Database(db_path)
        source_id = db.add_source("https://example.com/approve-all")
        for i in range(5):
            db.add_entity(
                entity_id=f"organization:approve-{i}",
                name=f"Org {i}",
                entity_type="Organization",
                source_id=source_id,
            )
        db.close()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["review", "--approve-all", "--type", "entity"], env=cli_env
        )
        assert result.exit_code == 0
        assert "Approved 5 entities" in result.output

        db2 = Database(db_path)
        approved = db2.get_entities_by_status("approved")
        assert len(approved) == 5
        db2.close()

    def test_approve_all_edges_cli(self, tmp_path, cli_env):
        """CLI --approve-all approves all pending edges."""
        db_path = cli_env["KG_DB_PATH"]
        db = Database(db_path)
        source_id = db.add_source("https://example.com/approve-edges")
        for i in range(3):
            db.add_edge(
                edge_id=f"approve-edge-{i}",
                source_entity_id="organization:a",
                target_entity_id=f"project:p{i}",
                edge_type="DEVELOPS",
                source_id=source_id,
            )
        db.close()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["review", "--approve-all", "--type", "edge"], env=cli_env
        )
        assert result.exit_code == 0
        assert "Approved 3 edges" in result.output

    def test_approve_all_both_types(self, tmp_path, cli_env):
        """CLI --approve-all with --type all approves both entities and edges."""
        db_path = cli_env["KG_DB_PATH"]
        db = Database(db_path)
        source_id = db.add_source("https://example.com/approve-both")
        db.add_entity(
            entity_id="organization:both-ent",
            name="Both Ent",
            entity_type="Organization",
            source_id=source_id,
        )
        db.add_edge(
            edge_id="both-edge",
            source_entity_id="organization:both-ent",
            target_entity_id="project:x",
            edge_type="DEVELOPS",
            source_id=source_id,
        )
        db.close()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["review", "--approve-all", "--type", "all"], env=cli_env
        )
        assert result.exit_code == 0
        assert "Approved 1 entities" in result.output
        assert "Approved 1 edges" in result.output

    def test_approve_all_advances_sources(self, tmp_path, cli_env):
        """Approving all advances sources in pending_review to load stage."""
        db_path = cli_env["KG_DB_PATH"]
        db = Database(db_path)
        source_id = db.add_source("https://example.com/advance")
        db.update_source(source_id, status="pending_review", stage="review")
        db.add_entity(
            entity_id="organization:advance-ent",
            name="Advance",
            entity_type="Organization",
            source_id=source_id,
        )
        db.close()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["review", "--approve-all"], env=cli_env
        )
        assert result.exit_code == 0
        assert "Advanced 1 sources to load stage" in result.output


# ---------------------------------------------------------------------------
# 5. Retry all failed sources
# ---------------------------------------------------------------------------


class TestRetryAllFailed:
    """Retry all failed sources in one command."""

    def test_retry_failed_db_level(self, db):
        """retry_failed resets all 'failed' sources to 'pending'."""
        for i in range(5):
            sid = db.add_source(f"https://example.com/fail-{i}")
            db.fail_source(sid, "network error")

        failed = db.get_sources_by_status("failed")
        assert len(failed) == 5

        count = db.retry_failed()
        assert count == 5

        pending = db.get_sources_by_status("pending")
        assert len(pending) == 5
        failed = db.get_sources_by_status("failed")
        assert len(failed) == 0

    def test_retry_does_not_affect_dead_letter(self, db):
        """retry_failed does not retry dead-letter sources."""
        sid = db.add_source("https://example.com/dead")
        for _ in range(5):
            db.fail_source(sid, "repeated failure")

        source = db.get_source(sid)
        assert source["status"] == "dead_letter"

        count = db.retry_failed()
        assert count == 0

    def test_retry_cli(self, tmp_path, cli_env):
        """CLI retry command calls retry_failed."""
        db_path = cli_env["KG_DB_PATH"]
        db = Database(db_path)
        for i in range(3):
            sid = db.add_source(f"https://example.com/retry-cli-{i}")
            db.fail_source(sid, "error")
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["retry"], env=cli_env)
        assert result.exit_code == 0
        assert "Retried 3" in result.output

    def test_retry_preserves_attempt_count(self, db):
        """retry_failed resets status but preserves attempt count."""
        sid = db.add_source("https://example.com/attempts")
        db.fail_source(sid, "error 1")
        db.fail_source(sid, "error 2")

        db.retry_failed()
        source = db.get_source(sid)
        assert source["status"] == "pending"
        assert source["attempts"] == 2

    def test_retry_clears_error_message(self, db):
        """retry_failed clears the error field."""
        sid = db.add_source("https://example.com/clear-error")
        db.fail_source(sid, "some error")
        db.retry_failed()

        source = db.get_source(sid)
        assert source["error"] is None


# ---------------------------------------------------------------------------
# 6. Status summary across batch operations
# ---------------------------------------------------------------------------


class TestStatusSummary:
    """Status summary reflects batch operations correctly."""

    def test_status_summary_counts(self, db):
        """status_summary groups sources by status."""
        for i in range(3):
            db.add_source(f"https://example.com/pending-{i}")
        for i in range(2):
            sid = db.add_source(f"https://example.com/fail-{i}")
            db.fail_source(sid, "error")
        sid = db.add_source("https://example.com/complete")
        db.update_source(sid, status="complete")

        summary = db.status_summary()
        assert summary.get("pending", 0) == 3
        assert summary.get("failed", 0) == 2
        assert summary.get("complete", 0) == 1

    def test_status_cli_output(self, tmp_path, cli_env):
        """CLI status command shows correct counts."""
        db_path = cli_env["KG_DB_PATH"]
        db = Database(db_path)
        for i in range(4):
            db.add_source(f"https://example.com/status-{i}")
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["status"], env=cli_env)
        assert result.exit_code == 0
        assert "pending: 4" in result.output

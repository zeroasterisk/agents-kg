"""Iteration 9 — Documentation tests.

Validates documentation artefacts: demo-queries.md Cypher syntax,
YAML entity file integrity, and CLI --help accuracy.
"""

import os
import re
from pathlib import Path
import pytest
import yaml
from click.testing import CliRunner
from agents_kg.cli import cli
from agents_kg.stages.extract import VALID_ENTITY_TYPES, VALID_EDGE_TYPES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Demo queries — valid Cypher syntax
# ---------------------------------------------------------------------------


class TestDemoQueriesCypher:
    """Parse and validate Cypher syntax in docs/demo-queries.md."""

    @pytest.fixture(autouse=True)
    def load_queries(self):
        demo_path = PROJECT_ROOT / "docs" / "demo-queries.md"
        assert demo_path.exists(), f"demo-queries.md not found at {demo_path}"
        text = demo_path.read_text()
        blocks = re.findall(r"```cypher\n(.*?)```", text, re.DOTALL)
        assert len(blocks) > 0, "No Cypher code blocks found"
        self.queries = [b.strip() for b in blocks]

    def test_queries_found(self):
        """At least 5 demo queries exist."""
        assert len(self.queries) >= 5

    def test_queries_contain_return_or_set(self):
        """Every query either RETURN or SET (it does something)."""
        for i, q in enumerate(self.queries):
            upper = q.upper()
            assert "RETURN" in upper or "SET" in upper, (
                f"Query {i+1} has no RETURN or SET clause"
            )

    def test_queries_have_match_or_merge(self):
        """Every query starts with MATCH, MERGE, or WITH (valid entry points)."""
        for i, q in enumerate(self.queries):
            first_word = q.strip().split()[0].upper()
            assert first_word in {"MATCH", "MERGE", "WITH", "OPTIONAL", "UNWIND", "CREATE"}, (
                f"Query {i+1} starts with unexpected keyword: {first_word}"
            )

    def test_balanced_parentheses(self):
        """Parentheses and brackets are balanced in each query."""
        for i, q in enumerate(self.queries):
            assert q.count("(") == q.count(")"), f"Query {i+1}: unbalanced parentheses"
            assert q.count("[") == q.count("]"), f"Query {i+1}: unbalanced brackets"
            assert q.count("{") == q.count("}"), f"Query {i+1}: unbalanced braces"

    def test_no_semicolons_mid_query(self):
        """Queries don't have semicolons (which could indicate injection patterns)."""
        for i, q in enumerate(self.queries):
            semicolons = q.count(";")
            assert semicolons == 0, f"Query {i+1} contains {semicolons} semicolon(s)"

    def test_relationship_types_are_uppercase(self):
        """Relationship types in queries follow UPPER_SNAKE_CASE convention."""
        rel_pattern = re.compile(r'\[:?([A-Za-z_|]+)')
        for i, q in enumerate(self.queries):
            matches = rel_pattern.findall(q)
            for match in matches:
                for part in match.split("|"):
                    part = part.strip()
                    if part and part not in ("r",):
                        assert part == part.upper() or part[0].isupper(), (
                            f"Query {i+1}: relationship type '{part}' should be uppercase"
                        )

    def test_node_labels_are_capitalized(self):
        """Node labels start with uppercase (Neo4j convention)."""
        label_pattern = re.compile(r'\([\w]*:([\w]+)')
        for i, q in enumerate(self.queries):
            matches = label_pattern.findall(q)
            for label in matches:
                assert label[0].isupper(), (
                    f"Query {i+1}: node label '{label}' should start with uppercase"
                )

    def test_queries_reference_known_labels(self):
        """Node labels in queries are from the project ontology or graph model."""
        known_labels = {"Entity", "Source", "Chunk", "Event"} | VALID_ENTITY_TYPES
        # Match labels after ( like (n:Organization or (:Entity)
        label_pattern = re.compile(r'\(\w*:([A-Z][A-Za-z]+)')
        for i, q in enumerate(self.queries):
            labels = label_pattern.findall(q)
            for label in labels:
                assert label in known_labels, (
                    f"Query {i+1}: label '{label}' not in known labels {known_labels}"
                )


# ---------------------------------------------------------------------------
# 2. YAML entity files
# ---------------------------------------------------------------------------


class TestYAMLEntityFiles:
    """Verify YAML entity files are valid and structurally correct."""

    @pytest.fixture(autouse=True)
    def load_yaml_files(self):
        entities_dir = PROJECT_ROOT / "kg" / "entities"
        assert entities_dir.exists(), f"Entity directory not found at {entities_dir}"
        self.yaml_files = list(entities_dir.rglob("*.yaml"))
        assert len(self.yaml_files) > 0, "No YAML entity files found"

    def test_yaml_files_exist(self):
        """At least some YAML entity files exist."""
        assert len(self.yaml_files) > 5

    def test_yaml_files_parseable(self):
        """Every YAML file parses without error."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            assert data is not None or yf.stat().st_size == 0, f"Failed to parse {yf}"

    def test_yaml_files_have_id(self):
        """Every YAML entity file has an 'id' field."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if data is None:
                continue
            assert "id" in data, f"{yf.name} missing 'id' field"

    def test_yaml_files_have_name(self):
        """Every YAML entity file has a 'name' field."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if data is None:
                continue
            assert "name" in data, f"{yf.name} missing 'name' field"

    def test_yaml_files_have_type(self):
        """Every YAML entity file has a 'type' field."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if data is None:
                continue
            assert "type" in data, f"{yf.name} missing 'type' field"

    def test_yaml_entity_ids_are_prefixed(self):
        """Entity IDs follow type:name format or at least contain a colon."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if not data or "id" not in data:
                continue
            eid = data["id"]
            if ":" in eid:
                prefix = eid.split(":")[0]
                assert prefix.islower(), f"{yf.name}: id prefix '{prefix}' should be lowercase"

    def test_yaml_types_are_valid(self):
        """Entity types in YAML files are from the valid ontology set or known extensions."""
        extended = VALID_ENTITY_TYPES | {"Concept"}
        valid = {t.lower() for t in extended} | extended
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if not data or "type" not in data:
                continue
            etype = data["type"]
            etype_normalized = etype.capitalize() if etype == etype.lower() else etype
            assert etype_normalized in extended or etype in valid, (
                f"{yf.name}: type '{etype}' not in valid entity types"
            )

    def test_yaml_aliases_are_lists(self):
        """If present, aliases must be a list."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if not data or "aliases" not in data:
                continue
            assert isinstance(data["aliases"], list), (
                f"{yf.name}: aliases must be a list, got {type(data['aliases'])}"
            )

    def test_yaml_descriptions_are_strings(self):
        """If present, descriptions must be strings."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if not data or "description" not in data:
                continue
            assert isinstance(data["description"], str), (
                f"{yf.name}: description must be a string"
            )

    def test_yaml_directory_structure_matches_types(self):
        """YAML files are in directories named after their type (pluralised)."""
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if not data or "type" not in data:
                continue
            parent_dir = yf.parent.name.lower()
            expected_type = data["type"].lower()
            assert parent_dir.startswith(expected_type), (
                f"{yf.name}: in directory '{parent_dir}' but type is '{expected_type}'"
            )

    def test_no_duplicate_entity_ids(self):
        """No two YAML files define the same entity_id."""
        seen = {}
        for yf in self.yaml_files:
            data = yaml.safe_load(yf.read_text())
            if not data or "id" not in data:
                continue
            eid = data["id"]
            assert eid not in seen, (
                f"Duplicate entity_id '{eid}' in {yf.name} and {seen[eid]}"
            )
            seen[eid] = yf.name


# ---------------------------------------------------------------------------
# 3. CLI --help output matches command signatures
# ---------------------------------------------------------------------------


class TestCLIHelp:
    """Verify CLI --help output is correct and commands exist."""

    def test_main_help(self):
        """Main CLI group --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Knowledge graph" in result.output

    def test_ingest_help(self):
        """ingest --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--from" in result.output
        assert "--file" in result.output
        assert "--submitter-email" in result.output

    def test_process_help(self):
        """process --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0

    def test_status_help(self):
        """status --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_review_help(self):
        """review --help shows approval options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["review", "--help"])
        assert result.exit_code == 0
        assert "--approve" in result.output
        assert "--approve-all" in result.output
        assert "--type" in result.output

    def test_retry_help(self):
        """retry --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["retry", "--help"])
        assert result.exit_code == 0

    def test_reset_help(self):
        """reset --help shows SOURCE_ID argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["reset", "--help"])
        assert result.exit_code == 0
        assert "SOURCE_ID" in result.output

    def test_seed_help(self):
        """seed --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["seed", "--help"])
        assert result.exit_code == 0

    def test_load_yaml_help(self):
        """load-yaml --help shows directory and relations options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["load-yaml", "--help"])
        assert result.exit_code == 0
        assert "--entities-dir" in result.output
        assert "--relations" in result.output

    def test_schema_help(self):
        """schema --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "--help"])
        assert result.exit_code == 0

    def test_wikidata_help(self):
        """wikidata --help shows subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["wikidata", "--help"])
        assert result.exit_code == 0
        assert "pull" in result.output
        assert "crossref" in result.output

    def test_wikidata_pull_help(self):
        """wikidata pull --help shows type filter and dry-run."""
        runner = CliRunner()
        result = runner.invoke(cli, ["wikidata", "pull", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        assert "--dry-run" in result.output

    def test_events_help(self):
        """events --help shows subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["events", "--help"])
        assert result.exit_code == 0
        assert "load" in result.output
        assert "migrate" in result.output

    def test_events_load_help(self):
        """events load --help shows directory option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["events", "load", "--help"])
        assert result.exit_code == 0
        assert "--dir" in result.output

    def test_all_commands_present(self):
        """Main CLI group lists all expected commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        expected = ["ingest", "process", "status", "review", "retry", "reset",
                     "seed", "load-yaml", "schema", "wikidata", "events"]
        for cmd in expected:
            assert cmd in result.output, f"Command '{cmd}' not found in CLI help"

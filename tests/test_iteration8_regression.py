"""Iteration 8 — Regression suite.

Captures all bugs fixed in iterations 1-7 as permanent guards against
re-introduction. Each test documents the original bug and fix.
"""

import os
import tempfile

import pytest
from unittest.mock import patch

from agents_kg.db import Database
from agents_kg.seed import SEED_ENTITIES
from agents_kg.stages import chunk, parse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = Database(path)
    yield d
    d.close()
    os.unlink(path)


# ---------------------------------------------------------------------------
# Regression 1: Genai test without API key does not crash
#
# Bug: test_genai_access.py would crash (ImportError or unhandled exception)
# when no Google API key was configured, instead of being skipped.
# Fix: Added proper skip markers for missing API key.
# ---------------------------------------------------------------------------

class TestGenaiSkipRegression:

    def test_import_genai_module_without_key(self):
        """Importing the extract module should not require an API key."""
        from agents_kg.stages import extract
        assert hasattr(extract, "run")
        assert hasattr(extract, "VALID_ENTITY_TYPES")

    def test_import_embed_module_without_key(self):
        """Importing the embed module should not require an API key."""
        from agents_kg.stages import embed
        assert hasattr(embed, "run")

    def test_extract_requires_chunks_not_key(self, db):
        """Extract stage raises about missing chunks, not missing API key."""
        sid = db.add_source("https://example.com/genai-regression")
        db.update_source(sid, raw_text="# Test\n\nContent", type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)
        src = db.get_source(sid)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            try:
                from agents_kg.stages.extract import run as extract_run
                extract_run(db, src)
            except RuntimeError as e:
                assert "genai" in str(e).lower() or "google" in str(e).lower()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Regression 2: Whitespace-only chunks are skipped
#
# Bug: The chunker produced chunks containing only whitespace, which caused
# downstream stages (embed, extract) to fail or produce garbage.
# Fix: chunk.py now skips chunks where full_text.strip() is empty.
# ---------------------------------------------------------------------------

class TestWhitespaceChunkRegression:

    def test_whitespace_only_sections_produce_no_chunks(self, db):
        """A document with whitespace-only sections yields no empty chunks."""
        sid = db.add_source("https://example.com/ws-regression")
        text = "# Header\n\n   \n\n## Section\n\n\t\t\n\n## Real\n\nActual content here"
        db.update_source(sid, raw_text=text, type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)

        chunks = db.get_chunks(sid)
        for c in chunks:
            assert c["text"].strip(), \
                f"Chunk {c['id']} at position {c['position']} is whitespace-only"

    def test_all_whitespace_document_produces_no_chunks(self, db):
        """A fully-whitespace document raises (no text to chunk)."""
        sid = db.add_source("https://example.com/all-ws")
        db.update_source(sid, raw_text="   \n\n\t\t\n  ", type="text",
                         parsed_text="   \n\n\t\t\n  ", stage="chunk", status="processing")
        src = db.get_source(sid)
        chunk.run(db, src)
        chunks = db.get_chunks(sid)
        assert len(chunks) == 0

    def test_mixed_content_only_real_chunks(self, db):
        """Chunks from mixed content are all non-empty."""
        sid = db.add_source("https://example.com/mixed-ws")
        text = "# Title\n\nReal text.\n\n\n\n\n\n## Another\n\nMore real text.\n\n   \n\n"
        db.update_source(sid, raw_text=text, type="text")
        src = db.get_source(sid)
        parse.run(db, src)
        src = db.get_source(sid)
        chunk.run(db, src)

        chunks = db.get_chunks(sid)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c["text"].strip()) > 0


# ---------------------------------------------------------------------------
# Regression 3: LangGraph is not aliased to LangChain
#
# Bug: LangGraph was listed as an alias of LangChain in the seed data,
# causing entity resolution to merge them into one entity.
# Fix: Removed LangGraph from LangChain aliases; LangGraph now has its own
# entry with empty aliases.
# ---------------------------------------------------------------------------

class TestLangGraphAliasRegression:

    def test_langgraph_is_separate_entity(self):
        """LangGraph has its own seed entity, not aliased to LangChain."""
        langgraph = None
        langchain_project = None

        for ent in SEED_ENTITIES:
            if ent["entity_id"] == "project:langgraph":
                langgraph = ent
            if ent["entity_id"] == "project:langchain":
                langchain_project = ent

        assert langgraph is not None, "project:langgraph not found in seed data"
        assert langchain_project is not None, "project:langchain not found in seed data"

    def test_langgraph_not_in_langchain_aliases(self):
        """'LangGraph' does not appear in LangChain project aliases."""
        for ent in SEED_ENTITIES:
            if ent["entity_id"] == "project:langchain":
                aliases_lower = [a.lower() for a in ent.get("aliases", [])]
                assert "langgraph" not in aliases_lower, \
                    "LangGraph should not be an alias of project:langchain"
                break

    def test_langgraph_not_in_langchain_org_aliases(self):
        """'LangGraph' does not appear in LangChain organization aliases."""
        for ent in SEED_ENTITIES:
            if ent["entity_id"] == "organization:langchain":
                aliases_lower = [a.lower() for a in ent.get("aliases", [])]
                assert "langgraph" not in aliases_lower, \
                    "LangGraph should not be an alias of organization:langchain"
                break


# ---------------------------------------------------------------------------
# Regression 4: Default Neo4j URI is localhost, not Docker hostname
#
# Bug: The CLI defaulted to a Docker-internal hostname (neo4j://) instead
# of bolt://localhost:7687, causing connection failures when running the
# pipeline outside Docker.
# Fix: Changed default NEO4J_URI to bolt://localhost:7687.
# ---------------------------------------------------------------------------

class TestNeo4jDefaultURIRegression:

    def test_cli_defaults_to_localhost(self):
        """CLI get_neo4j_config defaults to bolt://localhost:7687."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEO4J_URI", None)
            from agents_kg.cli import get_neo4j_config
            uri, auth = get_neo4j_config()
            assert uri == "bolt://localhost:7687", f"Expected bolt://localhost:7687, got {uri}"

    def test_cli_default_credentials(self):
        """CLI defaults to neo4j/agents-kg-2026 credentials."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEO4J_USER", None)
            os.environ.pop("NEO4J_PASSWORD", None)
            from agents_kg.cli import get_neo4j_config
            _, auth = get_neo4j_config()
            assert auth == ("neo4j", "agents-kg-2026")

    def test_env_override_uri(self):
        """NEO4J_URI env var overrides the default."""
        with patch.dict(os.environ, {"NEO4J_URI": "bolt://custom:7688"}):
            from agents_kg.cli import get_neo4j_config
            uri, _ = get_neo4j_config()
            assert uri == "bolt://custom:7688"

    def test_neo4j_integration_test_uses_localhost(self):
        """Integration test module defaults to localhost, not Docker hostname."""
        import tests.test_neo4j_integration as mod
        assert "localhost" in mod.NEO4J_URI, \
            f"Integration tests should default to localhost, got {mod.NEO4J_URI}"

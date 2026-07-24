"""Unit tests for the query router — all external calls mocked."""

from __future__ import annotations

import json
import os
import time

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents_kg.query_router import (
    answer,
    generate_cypher,
    get_cache_path,
    is_cache_valid,
    is_sufficient,
)


# ---------------------------------------------------------------------------
# test_generate_cypher_returns_string
# ---------------------------------------------------------------------------


class TestGenerateCypher:
    def test_generate_cypher_returns_string(self):
        mock_response = MagicMock()
        mock_response.text = "MATCH (n:Protocol) RETURN n.entity_id, n.name, n.description"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("agents_kg.query_router._get_genai_client", return_value=mock_client):
            result = generate_cypher("What protocols exist?")

        assert isinstance(result, str)
        assert result.startswith("MATCH") or result.startswith("WITH")


# ---------------------------------------------------------------------------
# test_is_sufficient_empty / test_is_sufficient_good
# ---------------------------------------------------------------------------


class TestIsSufficient:
    def test_is_sufficient_empty(self):
        assert is_sufficient([], "any question") is False

    def test_is_sufficient_good(self):
        results = [{"name": "MCP", "description": "Model Context Protocol"}]
        assert is_sufficient(results, "what is MCP") is True

    def test_missing_description(self):
        results = [{"name": "MCP", "description": ""}]
        assert is_sufficient(results, "what is MCP") is False

    def test_missing_name(self):
        results = [{"name": "", "description": "some description"}]
        assert is_sufficient(results, "question") is False


# ---------------------------------------------------------------------------
# test_cache_valid / test_cache_expired
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_valid(self, tmp_path: Path):
        cache_file = tmp_path / "test.json"
        cache_file.write_text("{}")
        assert is_cache_valid(cache_file, ttl_days=7) is True

    def test_cache_expired(self, tmp_path: Path):
        cache_file = tmp_path / "expired.json"
        cache_file.write_text("{}")
        # Set mtime to 8 days ago
        old_time = time.time() - (8 * 86400)
        os.utime(cache_file, (old_time, old_time))
        assert is_cache_valid(cache_file, ttl_days=7) is False

    def test_cache_missing(self, tmp_path: Path):
        cache_file = tmp_path / "nonexistent.json"
        assert is_cache_valid(cache_file) is False

    def test_cache_path_deterministic(self):
        p1 = get_cache_path("hello world")
        p2 = get_cache_path("hello world")
        assert p1 == p2

    def test_cache_path_different_keys(self):
        p1 = get_cache_path("hello")
        p2 = get_cache_path("world")
        assert p1 != p2


# ---------------------------------------------------------------------------
# test_answer_uses_cache
# ---------------------------------------------------------------------------


class TestAnswerPipeline:
    def test_answer_uses_cache(self, tmp_path: Path):
        """Pre-write a cache file and verify answer() returns it."""
        cached_data = {
            "question": "what is MCP",
            "cache_key": "what is mcp",
            "answer": "MCP is the Model Context Protocol.",
            "source": "synthesis",
            "entity_ids": ["protocol:mcp"],
            "cypher_used": "MATCH (n:Protocol) WHERE n.name = 'MCP' RETURN n",
            "sources": ["https://spec.modelcontextprotocol.io"],
            "confidence": "high",
            "cached": False,
            "cache_age_hours": None,
            "generated_at": "2026-07-19T12:00:00Z",
        }

        # Write the cache file at the expected path
        cache_key = "what is mcp"
        with patch("agents_kg.query_router.CACHE_DIR", tmp_path):
            cache_path = get_cache_path(cache_key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached_data))

            # answer() should return the cached result without calling Gemini/Neo4j
            mock_driver = MagicMock()
            result = answer(
                question="What is MCP",
                driver=mock_driver,
                db_path=str(tmp_path / "test.db"),
            )

        assert result["cached"] is True
        assert result["source"] == "cached"
        assert result["answer"] == "MCP is the Model Context Protocol."
        # Gemini and Neo4j should NOT have been called
        mock_driver.execute_query.assert_not_called()

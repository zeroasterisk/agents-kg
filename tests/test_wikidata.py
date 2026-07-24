"""Tests for wikidata.py transform/mapping logic (no Neo4j required)."""

import pytest

from agents_kg.wikidata import (
    _qid,
    _to_entity_id,
    _val,
    _is_real_label,
    transform_to_entities,
    extract_edges,
    _merge_implicit,
)


class TestHelpers:
    def test_qid_from_uri(self):
        assert _qid("http://www.wikidata.org/entity/Q42") == "Q42"

    def test_qid_bare(self):
        assert _qid("Q42") == "Q42"

    def test_qid_empty(self):
        assert _qid("") == ""

    def test_val_present(self):
        binding = {"name": {"value": "Python"}}
        assert _val(binding, "name") == "Python"

    def test_val_missing(self):
        assert _val({}, "name") is None

    def test_is_real_label(self):
        assert _is_real_label("Python", "Q42")
        assert not _is_real_label("Q42", "Q42")
        assert not _is_real_label("Q999999", "Q999999")

    def test_entity_id_kebab_case(self):
        assert _to_entity_id("Hello World", "Project") == "project:hello-world"

    def test_entity_id_special_chars(self):
        assert _to_entity_id("C++ (lang)", "Project") == "project:c-lang"

    def test_entity_id_truncates_long_names(self):
        long_name = "a" * 200
        result = _to_entity_id(long_name, "Project")
        slug = result.split(":", 1)[1]
        assert len(slug) <= 80

    def test_entity_id_type_prefix(self):
        result = _to_entity_id("Linux", "Organization")
        assert result.startswith("organization:")

    def test_entity_id_strips_trailing_hyphens(self):
        result = _to_entity_id("test---", "Project")
        assert not result.endswith("-")


def _make_binding(qid, label, desc=None, inception=None, **extras):
    """Helper to build a SPARQL binding dict."""
    b = {
        "item": {"value": f"http://www.wikidata.org/entity/{qid}"},
        "itemLabel": {"value": label},
    }
    if desc:
        b["itemDescription"] = {"value": desc}
    if inception:
        b["inception"] = {"value": inception}
    for k, v in extras.items():
        b[k] = {"value": v}
    return b


class TestTransformToEntities:
    def test_basic_transform(self):
        bindings = [_make_binding("Q42", "Python", desc="A language", inception="1991-02-20T00:00:00Z")]
        entities = transform_to_entities(bindings, "Project", "programming_language")
        assert len(entities) == 1
        e = entities[0]
        assert e["entity_id"] == "project:python"
        assert e["name"] == "Python"
        assert e["type"] == "Project"
        assert e["kind"] == "programming_language"
        assert e["wikidata_id"] == "Q42"
        assert e["created_at"] == "1991-02-20"
        assert e["source_type"] == "wikidata"

    def test_deduplication_by_qid(self):
        bindings = [
            _make_binding("Q42", "Python"),
            _make_binding("Q42", "Python"),
            _make_binding("Q42", "Python (different row)"),
        ]
        entities = transform_to_entities(bindings, "Project", "language")
        assert len(entities) == 1

    def test_skips_fake_labels(self):
        bindings = [_make_binding("Q999", "Q999")]
        entities = transform_to_entities(bindings, "Project", "language")
        assert len(entities) == 0

    def test_truncates_long_descriptions(self):
        bindings = [_make_binding("Q1", "Test", desc="x" * 1000)]
        entities = transform_to_entities(bindings, "Project", "tool")
        assert len(entities[0]["description"]) <= 500

    def test_no_inception_when_malformed(self):
        bindings = [_make_binding("Q1", "Test", inception="unknown")]
        entities = transform_to_entities(bindings, "Project", "tool")
        assert entities[0]["created_at"] is None

    def test_empty_bindings(self):
        assert transform_to_entities([], "Project", "tool") == []


class TestExtractEdges:
    def _lang_bindings(self):
        return [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q42"},
                "itemLabel": {"value": "Python"},
                "developerLabel": {"value": "Python Foundation"},
                "developer": {"value": "http://www.wikidata.org/entity/Q123"},
                "inception": {"value": "1991-02-20T00:00:00Z"},
            }
        ]

    def _edge_configs(self):
        return [
            {
                "label_key": "developerLabel",
                "qid_key": "developer",
                "edge_type": "DEVELOPS",
                "target_type": "Organization",
                "reverse": True,
            }
        ]

    def test_extracts_edges(self):
        edges, implicit = extract_edges(self._lang_bindings(), "Project", self._edge_configs())
        assert len(edges) == 1
        edge = edges[0]
        assert edge["edge_type"] == "DEVELOPS"
        assert edge["source_type"] == "wikidata"
        assert edge["confidence"] == 0.9

    def test_reverse_edge_direction(self):
        edges, _ = extract_edges(self._lang_bindings(), "Project", self._edge_configs())
        edge = edges[0]
        assert edge["source_entity_id"] == "organization:python-foundation"
        assert edge["target_entity_id"] == "project:python"

    def test_forward_edge_direction(self):
        configs = [
            {
                "label_key": "developerLabel",
                "qid_key": "developer",
                "edge_type": "DEVELOPED_BY",
                "target_type": "Organization",
            }
        ]
        edges, _ = extract_edges(self._lang_bindings(), "Project", configs)
        edge = edges[0]
        assert edge["source_entity_id"] == "project:python"
        assert edge["target_entity_id"] == "organization:python-foundation"

    def test_implicit_entities(self):
        _, implicit = extract_edges(self._lang_bindings(), "Project", self._edge_configs())
        assert len(implicit) == 1
        imp = implicit[0]
        assert imp["name"] == "Python Foundation"
        assert imp["type"] == "Organization"
        assert imp["wikidata_id"] == "Q123"

    def test_edge_deduplication(self):
        bindings = self._lang_bindings() + self._lang_bindings()
        edges, _ = extract_edges(bindings, "Project", self._edge_configs())
        assert len(edges) == 1

    def test_valid_from_extracted(self):
        edges, _ = extract_edges(self._lang_bindings(), "Project", self._edge_configs())
        assert edges[0]["valid_from"] == "1991-02-20"

    def test_empty_bindings(self):
        edges, implicit = extract_edges([], "Project", self._edge_configs())
        assert edges == []
        assert implicit == []


class TestMergeImplicit:
    def test_merges_new(self):
        entities = [{"entity_id": "project:a"}]
        implicit = [{"entity_id": "organization:b"}]
        _merge_implicit(entities, implicit)
        assert len(entities) == 2

    def test_skips_duplicates(self):
        entities = [{"entity_id": "project:a"}]
        implicit = [{"entity_id": "project:a"}]
        _merge_implicit(entities, implicit)
        assert len(entities) == 1

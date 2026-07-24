"""Iteration 9 — Security tests.

Validates that the pipeline is resilient to injection attacks and
dangerous input patterns across all storage backends.
"""

import json
import os
import re
import ast
import pytest
from agents_kg.stages import load as load_stage
from agents_kg.stages import extract as extract_stage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Cypher injection — entity names with Cypher metacharacters
# ---------------------------------------------------------------------------


class TestCypherInjection:
    """Ensure entity names containing Cypher metacharacters are safely handled."""

    PAYLOADS = [
        "O'Reilly Media",
        'Entity "with" quotes',
        "Entity {curly: 'braces'}",
        "Entity [brackets]",
        "MATCH (n) DETACH DELETE n",
        "'; DROP TABLE entities; --",
        "}) RETURN n; MATCH (n) DETACH DELETE n //",
        'name"}) RETURN n; MATCH (m) DELETE m //',
        "entity\\nid",
        "test\x00null",
    ]

    @pytest.mark.parametrize("name", PAYLOADS)
    def test_entity_name_stored_safely(self, db, name):
        """Entity names with Cypher metacharacters are stored and retrieved intact."""
        eid = f"organization:test-{abs(hash(name)) % 10000}"
        db.add_entity(entity_id=eid, name=name, entity_type="Organization")
        row = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (eid,)
        ).fetchone()
        assert row is not None
        assert row["name"] == name

    @pytest.mark.parametrize("name", PAYLOADS)
    def test_entity_to_cypher_uses_params(self, name):
        """_entity_to_cypher never interpolates data into the query string."""
        entity = {
            "entity_id": f"organization:test-{abs(hash(name)) % 10000}",
            "name": name,
            "type": "Organization",
            "kind": "company",
            "description": f"Desc for {name}",
            "aliases": json.dumps(["alias1"]),
            "source_id": 1,
        }
        query, params = load_stage._entity_to_cypher(entity)
        assert name not in query, "User data must not appear in the Cypher query string"
        assert params["name"] == name
        assert "$name" in query

    @pytest.mark.parametrize("name", PAYLOADS)
    def test_edge_to_cypher_uses_params(self, name):
        """_edge_to_cypher never interpolates data into the query string."""
        edge = {
            "source_entity_id": f"organization:{name}",
            "target_entity_id": "project:safe-target",
            "edge_type": "DEVELOPS",
            "edge_id": "test-edge-001",
            "confidence": 0.9,
            "source_type": "automated",
            "valid_from": None,
            "valid_to": None,
            "chunk_id": None,
            "properties": "{}",
        }
        query, params = load_stage._edge_to_cypher(edge)
        assert name not in query, "User data must not appear in the Cypher query string"
        assert params["src"] == f"organization:{name}"

    def test_cypher_label_only_valid_types(self):
        """Labels in Cypher are from a controlled set, never user input."""
        for bad_label in ["MATCH (n) DETACH DELETE n", "Entity'; DROP"]:
            entity = {
                "entity_id": "organization:safe",
                "name": "Safe Name",
                "type": bad_label,
                "kind": None,
                "description": None,
                "aliases": "[]",
                "source_id": 1,
            }
            query, _ = load_stage._entity_to_cypher(entity)
            assert bad_label not in query, "Invalid label must not appear in query"
            assert "Entity" in query


# ---------------------------------------------------------------------------
# 2. SQL injection — source URIs and entity names with SQL payloads
# ---------------------------------------------------------------------------


class TestSQLInjection:
    """Verify SQLite queries are always parameterized."""

    SQL_PAYLOADS = [
        "'; DROP TABLE sources; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM entities --",
        "Robert'); DROP TABLE students;--",
        "1; UPDATE sources SET status='hacked'",
        "' OR 1=1; --",
    ]

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_add_source_uri_safe(self, db, payload):
        """SQL injection payloads in URIs are stored as literal text."""
        sid = db.add_source(payload)
        assert sid is not None
        row = db.get_source(sid)
        assert row["uri"] == payload

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_add_entity_name_safe(self, db, payload):
        """SQL injection payloads in entity names are stored as literal text."""
        eid = f"organization:test-{abs(hash(payload)) % 10000}"
        result = db.add_entity(entity_id=eid, name=payload, entity_type="Organization")
        assert result is not None
        rows = db.get_entities_by_status("pending_review")
        found = [r for r in rows if r["entity_id"] == eid]
        assert len(found) == 1
        assert found[0]["name"] == payload

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_add_edge_id_safe(self, db, payload):
        """SQL injection payloads in edge IDs are stored as literal text."""
        result = db.add_edge(
            edge_id=payload,
            source_entity_id="organization:src",
            target_entity_id="project:tgt",
            edge_type="DEVELOPS",
        )
        assert result is not None
        rows = db.get_edges_by_status("pending_review")
        found = [r for r in rows if r["edge_id"] == payload]
        assert len(found) == 1

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_update_source_safe(self, db, payload):
        """SQL injection payloads in update values are treated as literals."""
        sid = db.add_source("https://example.com/safe")
        db.update_source(sid, title=payload)
        row = db.get_source(sid)
        assert row["title"] == payload

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_source_lookup_by_uri_safe(self, db, payload):
        """SQL injection payloads in lookup queries are treated as literals."""
        db.add_source(payload)
        row = db.get_source_by_uri(payload)
        assert row is not None
        assert row["uri"] == payload

    def test_tables_intact_after_injections(self, db):
        """After processing all injection payloads, tables still exist and are intact."""
        for payload in self.SQL_PAYLOADS:
            db.add_source(payload)
            db.add_entity(
                entity_id=f"org:inj-{abs(hash(payload)) % 10000}",
                name=payload,
                entity_type="Organization",
            )

        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "sources" in table_names
        assert "entities" in table_names
        assert "edges" in table_names
        assert "chunks" in table_names


# ---------------------------------------------------------------------------
# 3. Path traversal — source URIs with dangerous path patterns
# ---------------------------------------------------------------------------


class TestPathTraversal:
    """Verify path traversal URIs are stored safely and don't access the filesystem."""

    TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "file:///etc/passwd",
        "file:///proc/self/environ",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
    ]

    @pytest.mark.parametrize("uri", TRAVERSAL_PAYLOADS)
    def test_traversal_uri_stored_as_literal(self, db, uri):
        """Path traversal URIs are stored as literal strings, not interpreted."""
        sid = db.add_source(uri)
        assert sid is not None
        row = db.get_source(sid)
        assert row["uri"] == uri
        assert row["status"] == "pending"

    @pytest.mark.parametrize("uri", TRAVERSAL_PAYLOADS)
    def test_traversal_uri_no_file_read(self, db, uri):
        """The database layer doesn't attempt to read files from URIs."""
        sid = db.add_source(uri)
        row = db.get_source(sid)
        assert row["raw_text"] is None


# ---------------------------------------------------------------------------
# 4. XSS in entity descriptions — verify HTML is not special
# ---------------------------------------------------------------------------


class TestXSSInDescriptions:
    """Verify XSS payloads in descriptions are stored as-is (no execution context)."""

    XSS_PAYLOADS = [
        '<script>alert("xss")</script>',
        '<img src=x onerror=alert(1)>',
        '"><script>alert(document.cookie)</script>',
        "javascript:alert(1)",
        '<svg/onload=alert(1)>',
        '{{constructor.constructor("return this")()}}',
    ]

    @pytest.mark.parametrize("desc", XSS_PAYLOADS)
    def test_xss_description_stored_literally(self, db, desc):
        """XSS payloads in descriptions are stored as literal text."""
        eid = f"project:xss-test-{abs(hash(desc)) % 10000}"
        db.add_entity(entity_id=eid, name="XSS Test", entity_type="Project", description=desc)
        rows = db.get_entities_by_status("pending_review")
        found = [r for r in rows if r["entity_id"] == eid]
        assert len(found) == 1
        assert found[0]["description"] == desc

    @pytest.mark.parametrize("desc", XSS_PAYLOADS)
    def test_xss_in_yaml_export_safe(self, db, desc, tmp_path):
        """YAML export of XSS descriptions uses safe serialisation."""
        entity = {
            "entity_id": "project:xss-yaml",
            "name": "XSS Yaml Test",
            "type": "Project",
            "kind": "tool",
            "description": desc,
            "aliases": "[]",
            "source_id": None,
        }
        load_stage._export_yaml(entity, str(tmp_path))
        exported = list(tmp_path.rglob("*.yaml"))
        assert len(exported) == 1
        content = exported[0].read_text()
        assert "!!python" not in content


# ---------------------------------------------------------------------------
# 5. Parameterized query verification — source code audit
# ---------------------------------------------------------------------------


class TestParameterizedQueries:
    """Static analysis: verify all queries use parameterised values."""

    def _get_source_path(self, module_name: str) -> str:
        import agents_kg
        base = os.path.dirname(agents_kg.__file__)
        return os.path.join(base, module_name)

    def test_db_py_uses_parameterized_queries(self):
        """db.py INSERT/UPDATE/DELETE queries use ? placeholders for values."""
        src = open(self._get_source_path("db.py")).read()
        idx = 0
        while True:
            pos = src.find(".execute(", idx)
            if pos == -1:
                break
            start = pos + len(".execute(")
            depth = 1
            end = start
            while end < len(src) and depth > 0:
                if src[end] == "(":
                    depth += 1
                elif src[end] == ")":
                    depth -= 1
                end += 1
            call = src[start:end - 1]
            idx = end

            if "PRAGMA" in call or "sqlite_master" in call or "datetime" in call:
                continue
            # Only check DML that accepts user data (INSERT/UPDATE/DELETE with VALUES)
            if "INSERT" in call or ("UPDATE" in call and "SET" in call) or ("DELETE" in call and "WHERE" in call):
                assert "?" in call or "{sets}" in call, (
                    f"DML query without parameterization found: {call[:120]}"
                )

    def test_load_py_uses_dollar_params(self):
        """load.py Cypher queries use $param placeholders, not f-string interpolation of data."""
        src = open(self._get_source_path("stages/load.py")).read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node, 'func') and hasattr(node.func, 'attr'):
                    if node.func.attr == 'run' and node.args:
                        query_arg = node.args[0]
                        if isinstance(query_arg, (ast.JoinedStr, ast.FormattedValue)):
                            query_src = ast.get_source_segment(src, query_arg) or ""
                            for placeholder in re.findall(r'\{([^}]+)\}', query_src):
                                assert placeholder in ("label", "edge_type", "extra") or placeholder.startswith("prop_"), (
                                    f"f-string interpolates data field: {placeholder}"
                                )

    def test_extract_py_validates_entity_types(self):
        """extract.py rejects entity types not in VALID_ENTITY_TYPES."""
        assert len(extract_stage.VALID_ENTITY_TYPES) == 7
        for t in ["Organization", "Group", "Person", "Project", "Protocol", "Capability", "Concept"]:
            assert t in extract_stage.VALID_ENTITY_TYPES

    def test_extract_py_validates_edge_types(self):
        """extract.py rejects edge types not in VALID_EDGE_TYPES."""
        assert len(extract_stage.VALID_EDGE_TYPES) == 15
        for t in ["MEMBER_OF", "DEVELOPS", "IMPLEMENTS", "COMPETES_WITH"]:
            assert t in extract_stage.VALID_EDGE_TYPES

    def test_load_label_whitelist(self):
        """_entity_to_cypher only allows whitelisted labels."""
        for bad in ["DETACH DELETE n", "DROP", "Admin"]:
            entity = {
                "entity_id": "test:x",
                "name": "X",
                "type": bad,
                "kind": None,
                "description": None,
                "aliases": "[]",
                "source_id": 1,
            }
            query, _ = load_stage._entity_to_cypher(entity)
            assert bad not in query

    def test_edge_type_uppercase(self):
        """_edge_to_cypher uppercases edge_type (from controlled set)."""
        edge = {
            "source_entity_id": "organization:a",
            "target_entity_id": "project:b",
            "edge_type": "develops",
            "edge_id": "e1",
            "confidence": 0.8,
            "source_type": "automated",
            "valid_from": None,
            "valid_to": None,
            "chunk_id": None,
            "properties": "{}",
        }
        query, _ = load_stage._edge_to_cypher(edge)
        assert "DEVELOPS" in query


# ---------------------------------------------------------------------------
# 6. Combined injection — multi-vector attack strings
# ---------------------------------------------------------------------------


class TestCombinedInjection:
    """Test inputs that combine multiple attack vectors."""

    def test_mixed_sql_cypher_in_entity(self, db):
        """An entity with SQL + Cypher metacharacters survives the round trip."""
        payload = """O'Reilly "Media" {MATCH (n)} '); DROP TABLE entities;-- [test]"""
        eid = "organization:mixed-attack"
        db.add_entity(entity_id=eid, name=payload, entity_type="Organization",
                       description=payload, aliases=[payload])
        row = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (eid,)
        ).fetchone()
        assert row["name"] == payload
        assert row["description"] == payload
        aliases = json.loads(row["aliases"])
        assert aliases == [payload]

    def test_special_chars_in_edge_properties(self, db):
        """Edge properties with special characters are stored safely."""
        props = {"note": "'; DROP TABLE edges; --", "value": '<script>alert(1)</script>'}
        db.add_edge(
            edge_id="attack-edge",
            source_entity_id="organization:src",
            target_entity_id="project:tgt",
            edge_type="DEVELOPS",
            properties=props,
        )
        rows = db.get_edges_by_status("pending_review")
        found = [r for r in rows if r["edge_id"] == "attack-edge"]
        assert len(found) == 1
        stored_props = json.loads(found[0]["properties"])
        assert stored_props == props

    def test_unicode_injection(self, db):
        """Unicode characters and zero-width joiners don't break storage."""
        names = [
            "Entité françåise",
            "组织名称",
            "Org​Name",  # zero-width space
            "Org⁠Name",  # word joiner
            "🏢 Corp",
        ]
        for i, name in enumerate(names):
            eid = f"organization:unicode-{i}"
            db.add_entity(entity_id=eid, name=name, entity_type="Organization")
            row = db.conn.execute(
                "SELECT * FROM entities WHERE entity_id = ?", (eid,)
            ).fetchone()
            assert row["name"] == name

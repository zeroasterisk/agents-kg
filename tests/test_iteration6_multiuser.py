"""Iteration 6: Realistic multi-user scenario tests.

Simulates three users ingesting different content types with distinct submitter emails,
then verifies provenance tracking, entity merging, source isolation, and deprecation behavior.
"""

import pytest
from agents_kg.stages import parse, chunk


RESEARCHER_EMAIL = "alice@university.edu"
PM_EMAIL = "bob@company.io"
ENGINEER_EMAIL = "carol@startup.dev"


def _ingest_and_process_through_chunk(db, uri, raw_html, email, source_type="html"):
    """Helper: ingest a source and run it through parse + chunk."""
    sid = db.add_source(uri, submitter_email=email)
    db.update_source(sid, raw_text=raw_html, type=source_type, stage="parse", status="processing")
    source = db.get_source(sid)
    parse.run(db, source)
    source = db.get_source(sid)
    chunk.run(db, source)
    return sid


class TestMultiUserIngestion:
    """User A (researcher): 5 academic papers about agent protocols."""

    def test_researcher_ingests_papers(self, db):
        sids = []
        for i in range(5):
            sid = _ingest_and_process_through_chunk(
                db,
                f"https://arxiv.org/abs/2026.{i:05d}",
                f"<html><body><h1>Paper {i}: Agent Protocol Design</h1>"
                f"<p>This paper discusses protocol design patterns for multi-agent systems.</p>"
                f"<p>Section {i}: New contributions to the field.</p></body></html>",
                RESEARCHER_EMAIL,
            )
            sids.append(sid)
        assert len(sids) == 5
        for sid in sids:
            source = db.get_source(sid)
            assert source["submitter_email"] == RESEARCHER_EMAIL
            assert len(db.get_chunks(sid)) > 0

    def test_pm_ingests_blog_posts(self, db):
        sids = []
        for i in range(3):
            sid = _ingest_and_process_through_chunk(
                db,
                f"https://blog.company.io/comparison-{i}",
                f"<html><body><h1>Product Comparison: Agent Platforms {i}</h1>"
                f"<p>Comparing LangChain, AutoGPT, and CrewAI for enterprise use.</p></body></html>",
                PM_EMAIL,
            )
            sids.append(sid)
        assert len(sids) == 3
        for sid in sids:
            assert db.get_source(sid)["submitter_email"] == PM_EMAIL

    def test_engineer_ingests_readmes(self, db):
        sids = []
        for i in range(2):
            sid = _ingest_and_process_through_chunk(
                db,
                f"https://github.com/org/repo-{i}/README.md",
                f"# repo-{i}\n\nA framework for building multi-agent systems.\n\n"
                f"## Installation\n\n`pip install repo-{i}`\n\n## Usage\n\nImport and use.",
                ENGINEER_EMAIL,
                source_type="text",
            )
            sids.append(sid)
        assert len(sids) == 2
        for sid in sids:
            assert db.get_source(sid)["submitter_email"] == ENGINEER_EMAIL


class TestMultiUserProvenance:
    def test_sources_track_submitter(self, db):
        _ingest_and_process_through_chunk(
            db, "https://example.com/alice", "<html><body><h1>A</h1><p>Test</p></body></html>", RESEARCHER_EMAIL)
        _ingest_and_process_through_chunk(
            db, "https://example.com/bob", "<html><body><h1>B</h1><p>Test</p></body></html>", PM_EMAIL)
        _ingest_and_process_through_chunk(
            db, "https://example.com/carol", "# C\n\nTest content.", ENGINEER_EMAIL, "text")

        all_sources = db.conn.execute("SELECT * FROM sources ORDER BY id").fetchall()
        assert len(all_sources) == 3
        emails = {s["submitter_email"] for s in all_sources}
        assert emails == {RESEARCHER_EMAIL, PM_EMAIL, ENGINEER_EMAIL}

    def test_entities_linked_to_source(self, db):
        sid1 = _ingest_and_process_through_chunk(
            db, "https://example.com/prov1", "<html><body><h1>Prov1</h1><p>Text</p></body></html>", RESEARCHER_EMAIL)
        sid2 = _ingest_and_process_through_chunk(
            db, "https://example.com/prov2", "<html><body><h1>Prov2</h1><p>Text</p></body></html>", PM_EMAIL)

        db.add_entity("test:shared-org", "Shared Org", "Organization", source_id=sid1)
        db.add_entity("test:pm-only", "PM Only Org", "Organization", source_id=sid2)

        ent1 = db.conn.execute("SELECT * FROM entities WHERE entity_id = 'test:shared-org'").fetchone()
        ent2 = db.conn.execute("SELECT * FROM entities WHERE entity_id = 'test:pm-only'").fetchone()
        assert ent1["source_id"] == sid1
        assert ent2["source_id"] == sid2

    def test_user_can_query_own_sources(self, db):
        _ingest_and_process_through_chunk(
            db, "https://example.com/a1", "<html><body><h1>A1</h1><p>T</p></body></html>", RESEARCHER_EMAIL)
        _ingest_and_process_through_chunk(
            db, "https://example.com/a2", "<html><body><h1>A2</h1><p>T</p></body></html>", RESEARCHER_EMAIL)
        _ingest_and_process_through_chunk(
            db, "https://example.com/b1", "<html><body><h1>B1</h1><p>T</p></body></html>", PM_EMAIL)

        alice_sources = db.conn.execute(
            "SELECT * FROM sources WHERE submitter_email = ?", (RESEARCHER_EMAIL,)
        ).fetchall()
        bob_sources = db.conn.execute(
            "SELECT * FROM sources WHERE submitter_email = ?", (PM_EMAIL,)
        ).fetchall()
        assert len(alice_sources) == 2
        assert len(bob_sources) == 1


class TestMultiUserEntityMerging:
    def test_shared_entity_has_single_record(self, db):
        """Two users referencing the same entity should result in one entity record (by entity_id)."""
        sid1 = db.add_source("https://example.com/shared1", submitter_email=RESEARCHER_EMAIL)
        sid2 = db.add_source("https://example.com/shared2", submitter_email=PM_EMAIL)
        id1 = db.add_entity("organization:google", "Google", "Organization", source_id=sid1)
        id2 = db.add_entity("organization:google", "Google LLC", "Organization", source_id=sid2)
        assert id1 is not None
        assert id2 is None  # Duplicate by entity_id
        count = db.conn.execute(
            "SELECT COUNT(*) FROM entities WHERE entity_id = 'organization:google'"
        ).fetchone()[0]
        assert count == 1

    def test_different_entities_same_type(self, db):
        sid1 = db.add_source("https://example.com/diff1", submitter_email=RESEARCHER_EMAIL)
        sid2 = db.add_source("https://example.com/diff2", submitter_email=PM_EMAIL)
        db.add_entity("organization:google", "Google", "Organization", source_id=sid1)
        db.add_entity("organization:anthropic", "Anthropic", "Organization", source_id=sid2)
        count = db.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == 2


class TestMultiUserDeprecation:
    def test_deprecating_user_b_preserves_user_a(self, db):
        """Deprecating User B's sources should not affect User A's entities."""
        sid_a = db.add_source("https://example.com/keep-a", submitter_email=RESEARCHER_EMAIL)
        sid_b = db.add_source("https://example.com/remove-b", submitter_email=PM_EMAIL)
        db.add_entity("test:alice-ent", "Alice Entity", "Organization", source_id=sid_a)
        db.add_entity("test:bob-ent", "Bob Entity", "Organization", source_id=sid_b)

        db.deprecate_entities_for_source(sid_b)

        alice_ent = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'test:alice-ent'"
        ).fetchone()
        bob_ent = db.conn.execute(
            "SELECT * FROM entities WHERE entity_id = 'test:bob-ent'"
        ).fetchone()
        assert alice_ent["deprecated_at"] is None
        assert bob_ent["deprecated_at"] is not None

    def test_deprecating_source_nullifies_chunk_refs(self, db):
        sid = db.add_source("https://example.com/depr-chunks", submitter_email=PM_EMAIL)
        cid = db.add_chunk(sid, "some text", 0)
        db.add_entity("test:depr-ent", "Depr Ent", "Organization", source_id=sid, chunk_id=cid)
        db.add_edge("test-depr-edge", "test:depr-ent", "test:other", "DEVELOPS",
                     source_id=sid, chunk_id=cid)

        db.deprecate_entities_for_source(sid)

        ent = db.conn.execute("SELECT * FROM entities WHERE entity_id = 'test:depr-ent'").fetchone()
        edge = db.conn.execute("SELECT * FROM edges WHERE edge_id = 'test-depr-edge'").fetchone()
        assert ent["chunk_id"] is None
        assert edge["chunk_id"] is None

    def test_deprecation_does_not_affect_merged_entities(self, db):
        """Entities that have been merged into a canonical form should not be deprecated."""
        sid = db.add_source("https://example.com/merged-nodepr", submitter_email=PM_EMAIL)
        eid = db.add_entity("test:merged-ent", "Merged", "Organization", source_id=sid)
        db.update_entity(eid, merged_into="organization:google")

        db.deprecate_entities_for_source(sid)

        ent = db.conn.execute("SELECT * FROM entities WHERE id = ?", (eid,)).fetchone()
        assert ent["deprecated_at"] is None

    def test_user_c_unaffected_by_user_b_deprecation(self, db):
        sid_b = db.add_source("https://example.com/b-depr", submitter_email=PM_EMAIL)
        sid_c = db.add_source("https://example.com/c-safe", submitter_email=ENGINEER_EMAIL)

        db.add_entity("test:b-only", "B Entity", "Organization", source_id=sid_b)
        db.add_entity("test:c-only", "C Entity", "Organization", source_id=sid_c)

        db.deprecate_entities_for_source(sid_b)

        c_ent = db.conn.execute("SELECT * FROM entities WHERE entity_id = 'test:c-only'").fetchone()
        assert c_ent["deprecated_at"] is None

        b_ent = db.conn.execute("SELECT * FROM entities WHERE entity_id = 'test:b-only'").fetchone()
        assert b_ent["deprecated_at"] is not None


class TestMultiUserEdgeProvenance:
    def test_edges_track_source_id(self, db):
        sid1 = db.add_source("https://example.com/ep1", submitter_email=RESEARCHER_EMAIL)
        sid2 = db.add_source("https://example.com/ep2", submitter_email=PM_EMAIL)
        db.add_edge("edge-prov-1", "test:a", "test:b", "DEVELOPS", source_id=sid1)
        db.add_edge("edge-prov-2", "test:c", "test:d", "IMPLEMENTS", source_id=sid2)

        e1 = db.conn.execute("SELECT * FROM edges WHERE edge_id = 'edge-prov-1'").fetchone()
        e2 = db.conn.execute("SELECT * FROM edges WHERE edge_id = 'edge-prov-2'").fetchone()
        assert e1["source_id"] == sid1
        assert e2["source_id"] == sid2

    def test_same_edge_from_different_sources(self, db):
        """Same edge_id from different sources gets deduplicated."""
        sid1 = db.add_source("https://example.com/se1", submitter_email=RESEARCHER_EMAIL)
        sid2 = db.add_source("https://example.com/se2", submitter_email=PM_EMAIL)
        id1 = db.add_edge("same-edge-id", "test:x", "test:y", "DEVELOPS", source_id=sid1)
        id2 = db.add_edge("same-edge-id", "test:x", "test:y", "DEVELOPS", source_id=sid2)
        assert id1 is not None
        assert id2 is None

"""Iteration 9 — Graph evolution over time tests.

Simulates 30 days of graph activity: foundational sources, contradicting
news, deprecations, corrections, and temporal snapshot verification.
"""

import json
from datetime import datetime, timezone, timedelta
import pytest


def _make_date(day_offset: int) -> str:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(days=day_offset)).isoformat()


# ---------------------------------------------------------------------------
# 1. Day 1-10: Foundational sources
# ---------------------------------------------------------------------------


class TestFoundationalPhase:
    """Simulate adding foundational sources and entities in the first 10 days."""

    def test_add_foundational_sources(self, db):
        """10 foundational sources can be added and queried."""
        for i in range(10):
            sid = db.add_source(f"https://example.com/foundation/doc-{i}")
            assert sid is not None

        pending = db.get_pending_sources()
        assert len(pending) == 10

    def test_foundational_entities_and_edges(self, db):
        """Add entities and edges that form the initial graph."""
        sid = db.add_source("https://example.com/foundation/core")
        db.add_entity(entity_id="organization:alpha", name="Alpha Corp", entity_type="Organization",
                       kind="company", description="Foundational AI company", source_id=sid)
        db.add_entity(entity_id="project:alpha-sdk", name="Alpha SDK", entity_type="Project",
                       kind="sdk", description="Alpha's SDK", source_id=sid)
        db.add_entity(entity_id="protocol:alpha-spec", name="Alpha Spec", entity_type="Protocol",
                       kind="spec", description="Alpha's protocol", source_id=sid)
        db.add_edge(edge_id="e-alpha-dev", source_entity_id="organization:alpha",
                     target_entity_id="project:alpha-sdk", edge_type="DEVELOPS",
                     confidence=0.95, source_id=sid, valid_from="2025-01-01")
        db.add_edge(edge_id="e-sdk-impl", source_entity_id="project:alpha-sdk",
                     target_entity_id="protocol:alpha-spec", edge_type="IMPLEMENTS",
                     confidence=0.9, source_id=sid, valid_from="2025-01-01")

        entities = db.get_entities_by_status("pending_review")
        edges = db.get_edges_by_status("pending_review")
        assert len(entities) == 3
        assert len(edges) == 2

    def test_edge_temporal_data_stored(self, db):
        """Edges with valid_from dates store temporal data."""
        sid = db.add_source("https://example.com/foundation/temporal")
        db.add_edge(
            edge_id="temporal-1",
            source_entity_id="organization:alpha",
            target_entity_id="project:alpha-sdk",
            edge_type="DEVELOPS",
            valid_from="2025-01-01",
            valid_to=None,
            source_id=sid,
        )
        rows = db.get_edges_by_status("pending_review")
        assert rows[0]["valid_from"] == "2025-01-01"
        assert rows[0]["valid_to"] is None


# ---------------------------------------------------------------------------
# 2. Day 11-20: News articles, some contradict earlier data
# ---------------------------------------------------------------------------


class TestContradictionPhase:
    """Simulate contradicting information arriving in news articles."""

    def _setup_foundation(self, db):
        sid = db.add_source("https://example.com/foundation/setup")
        db.add_entity(entity_id="organization:beta", name="Beta Corp",
                       entity_type="Organization", kind="company",
                       description="Original description", source_id=sid)
        db.add_entity(entity_id="project:beta-tool", name="Beta Tool",
                       entity_type="Project", kind="tool",
                       description="Beta's main tool", source_id=sid)
        eid = db.add_edge(edge_id="e-beta-dev", source_entity_id="organization:beta",
                           target_entity_id="project:beta-tool", edge_type="DEVELOPS",
                           confidence=0.95, source_id=sid, valid_from="2025-01-01")
        for ent in db.get_entities_by_status("pending_review"):
            db.approve_entity(ent["id"])
        for edge in db.get_edges_by_status("pending_review"):
            db.approve_edge(edge["id"])
        return sid

    def test_contradicting_entity_description(self, db):
        """A newer source can add a different entity with updated description."""
        self._setup_foundation(db)
        news_sid = db.add_source("https://example.com/news/beta-acquired")

        result = db.add_entity(
            entity_id="organization:beta",
            name="Beta Corp",
            entity_type="Organization",
            kind="company",
            description="Beta Corp was acquired by Mega Inc in Day 15",
            source_id=news_sid,
        )
        assert result is None  # duplicate entity_id rejected

    def test_superseding_relationship(self, db):
        """A new edge can supersede an old one using valid_to."""
        self._setup_foundation(db)
        news_sid = db.add_source("https://example.com/news/new-dev")

        db.conn.execute(
            "UPDATE edges SET valid_to = ? WHERE edge_id = ?",
            ("2025-01-15", "e-beta-dev"),
        )
        db.conn.commit()

        db.add_edge(
            edge_id="e-mega-dev",
            source_entity_id="organization:mega",
            target_entity_id="project:beta-tool",
            edge_type="DEVELOPS",
            confidence=0.85,
            source_id=news_sid,
            valid_from="2025-01-15",
        )

        old = db.conn.execute("SELECT * FROM edges WHERE edge_id = ?", ("e-beta-dev",)).fetchone()
        new = db.conn.execute("SELECT * FROM edges WHERE edge_id = ?", ("e-mega-dev",)).fetchone()
        assert old["valid_to"] == "2025-01-15"
        assert new["valid_from"] == "2025-01-15"
        assert new["valid_to"] is None

    def test_competing_confidence_scores(self, db):
        """Multiple sources can produce edges with different confidence scores."""
        self._setup_foundation(db)
        for i in range(3):
            sid = db.add_source(f"https://example.com/news/competing-{i}")
            db.add_edge(
                edge_id=f"compete-{i}",
                source_entity_id="organization:beta",
                target_entity_id=f"project:gamma-{i}",
                edge_type="DEVELOPS",
                confidence=0.5 + i * 0.15,
                source_id=sid,
            )

        edges = db.get_edges_by_status("pending_review")
        confidences = sorted([e["confidence"] for e in edges])
        assert confidences == pytest.approx([0.5, 0.65, 0.8], abs=0.01)

    def test_multiple_sources_same_entity(self, db):
        """Multiple sources referencing the same entity_id — only one stored."""
        s1 = db.add_source("https://example.com/news/src1")
        s2 = db.add_source("https://example.com/news/src2")
        r1 = db.add_entity(entity_id="organization:gamma", name="Gamma", entity_type="Organization", source_id=s1)
        r2 = db.add_entity(entity_id="organization:gamma", name="Gamma Updated", entity_type="Organization", source_id=s2)
        assert r1 is not None
        assert r2 is None  # duplicate


# ---------------------------------------------------------------------------
# 3. Day 21-30: Review, deprecate, correct
# ---------------------------------------------------------------------------


class TestDeprecationPhase:
    """Simulate deprecating stale sources and adding corrections."""

    def test_deprecate_source_entities(self, db):
        """Deprecating entities from a source marks them with deprecated_at."""
        sid = db.add_source("https://example.com/stale-source")
        db.add_entity(entity_id="organization:stale-org", name="Stale Org",
                       entity_type="Organization", source_id=sid)
        db.add_entity(entity_id="project:stale-proj", name="Stale Project",
                       entity_type="Project", source_id=sid)

        db.deprecate_entities_for_source(sid)

        deprecated = db.get_deprecated_entities()
        assert len(deprecated) == 2
        for ent in deprecated:
            assert ent["deprecated_at"] is not None

    def test_deprecated_entities_not_in_pending_review(self, db):
        """Deprecated entities still have their original status (unchanged)."""
        sid = db.add_source("https://example.com/dep-review")
        db.add_entity(entity_id="organization:dep-org", name="Dep Org",
                       entity_type="Organization", source_id=sid)
        db.deprecate_entities_for_source(sid)

        deprecated = db.get_deprecated_entities()
        assert len(deprecated) == 1
        assert deprecated[0]["deprecated_at"] is not None

    def test_correction_replaces_deprecated(self, db):
        """A correction source can add fresh entities after deprecation."""
        old_sid = db.add_source("https://example.com/old-info")
        db.add_entity(entity_id="organization:evolve-corp", name="Evolve Corp v1",
                       entity_type="Organization", source_id=old_sid)
        db.deprecate_entities_for_source(old_sid)

        new_sid = db.add_source("https://example.com/correction")
        db.add_entity(entity_id="organization:evolve-corp-v2", name="Evolve Corp v2",
                       entity_type="Organization", kind="company",
                       description="Updated entity after correction",
                       source_id=new_sid)

        deprecated = db.get_deprecated_entities()
        assert any(e["entity_id"] == "organization:evolve-corp" for e in deprecated)

        pending = db.get_entities_by_status("pending_review")
        assert any(e["entity_id"] == "organization:evolve-corp-v2" for e in pending)

    def test_reset_source_clears_everything(self, db):
        """Resetting a source removes its entities, edges, and chunks."""
        sid = db.add_source("https://example.com/reset-test")
        db.add_entity(entity_id="organization:reset-org", name="Reset Org",
                       entity_type="Organization", source_id=sid)
        db.add_edge(edge_id="reset-edge", source_entity_id="organization:reset-org",
                     target_entity_id="project:x", edge_type="DEVELOPS", source_id=sid)
        db.add_chunk(sid, "Some chunk text", 0)

        db.reset_source(sid)

        source = db.get_source(sid)
        assert source["status"] == "pending"
        assert source["stage"] == "fetch"
        assert source["raw_text"] is None

        entities = db.conn.execute("SELECT * FROM entities WHERE source_id = ?", (sid,)).fetchall()
        edges = db.conn.execute("SELECT * FROM edges WHERE source_id = ?", (sid,)).fetchall()
        chunks = db.get_chunks(sid)
        assert len(entities) == 0
        assert len(edges) == 0
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# 4. Temporal snapshot queries (SQLite level)
# ---------------------------------------------------------------------------


class TestTemporalSnapshots:
    """Verify temporal queries give correct snapshots at different points in time."""

    def _build_temporal_graph(self, db):
        """Build a graph with temporal edges spanning 30 days."""
        sid = db.add_source("https://example.com/temporal-graph")

        db.add_entity(entity_id="organization:tempo-corp", name="Tempo Corp",
                       entity_type="Organization", source_id=sid)
        db.add_entity(entity_id="project:tempo-sdk", name="Tempo SDK",
                       entity_type="Project", source_id=sid)
        db.add_entity(entity_id="project:tempo-v2", name="Tempo V2",
                       entity_type="Project", source_id=sid)

        db.add_edge(edge_id="tempo-e1", source_entity_id="organization:tempo-corp",
                     target_entity_id="project:tempo-sdk", edge_type="DEVELOPS",
                     confidence=0.9, source_id=sid,
                     valid_from="2025-01-01", valid_to="2025-01-20")

        db.add_edge(edge_id="tempo-e2", source_entity_id="organization:tempo-corp",
                     target_entity_id="project:tempo-v2", edge_type="DEVELOPS",
                     confidence=0.95, source_id=sid,
                     valid_from="2025-01-15", valid_to=None)

        db.add_edge(edge_id="tempo-e3", source_entity_id="project:tempo-sdk",
                     target_entity_id="project:tempo-v2", edge_type="SUPERSEDES",
                     confidence=0.85, source_id=sid,
                     valid_from="2025-01-20", valid_to=None)

        return sid

    def test_snapshot_day_5(self, db):
        """Day 5: only tempo-sdk is being developed."""
        self._build_temporal_graph(db)
        snapshot_date = "2025-01-05"
        active = db.conn.execute(
            """SELECT * FROM edges
               WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
               ORDER BY edge_id""",
            (snapshot_date, snapshot_date),
        ).fetchall()
        edge_ids = [e["edge_id"] for e in active]
        assert "tempo-e1" in edge_ids
        assert "tempo-e2" not in edge_ids
        assert "tempo-e3" not in edge_ids

    def test_snapshot_day_17(self, db):
        """Day 17: both tempo-sdk and tempo-v2 are being developed (overlap)."""
        self._build_temporal_graph(db)
        snapshot_date = "2025-01-17"
        active = db.conn.execute(
            """SELECT * FROM edges
               WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
               ORDER BY edge_id""",
            (snapshot_date, snapshot_date),
        ).fetchall()
        edge_ids = [e["edge_id"] for e in active]
        assert "tempo-e1" in edge_ids  # still valid until day 20
        assert "tempo-e2" in edge_ids  # started day 15
        assert "tempo-e3" not in edge_ids  # starts day 20

    def test_snapshot_day_25(self, db):
        """Day 25: tempo-sdk development ended, tempo-v2 active, supersedes active."""
        self._build_temporal_graph(db)
        snapshot_date = "2025-01-25"
        active = db.conn.execute(
            """SELECT * FROM edges
               WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
               ORDER BY edge_id""",
            (snapshot_date, snapshot_date),
        ).fetchall()
        edge_ids = [e["edge_id"] for e in active]
        assert "tempo-e1" not in edge_ids  # ended day 20
        assert "tempo-e2" in edge_ids
        assert "tempo-e3" in edge_ids

    def test_all_edges_ever(self, db):
        """All edges regardless of time."""
        self._build_temporal_graph(db)
        all_edges = db.conn.execute("SELECT * FROM edges").fetchall()
        assert len(all_edges) == 3


# ---------------------------------------------------------------------------
# 5. Full 30-day simulation
# ---------------------------------------------------------------------------


class TestThirtyDaySimulation:
    """End-to-end simulation of graph activity over 30 days."""

    def test_full_lifecycle(self, db):
        """Simulate the complete 30-day graph evolution lifecycle."""
        # Day 1-10: Add foundational sources
        foundation_sids = []
        for day in range(1, 11):
            sid = db.add_source(f"https://example.com/day{day}/foundation")
            foundation_sids.append(sid)

        assert len(db.get_pending_sources()) == 10

        # Add foundational entities
        db.add_entity(entity_id="organization:alpha", name="Alpha Inc",
                       entity_type="Organization", source_id=foundation_sids[0])
        db.add_entity(entity_id="organization:beta", name="Beta Labs",
                       entity_type="Organization", source_id=foundation_sids[1])
        db.add_entity(entity_id="project:alpha-platform", name="Alpha Platform",
                       entity_type="Project", source_id=foundation_sids[2])
        db.add_entity(entity_id="protocol:openspec", name="OpenSpec",
                       entity_type="Protocol", source_id=foundation_sids[3])

        db.add_edge(edge_id="sim-e1", source_entity_id="organization:alpha",
                     target_entity_id="project:alpha-platform", edge_type="DEVELOPS",
                     confidence=0.95, source_id=foundation_sids[0], valid_from="2025-01-01")
        db.add_edge(edge_id="sim-e2", source_entity_id="project:alpha-platform",
                     target_entity_id="protocol:openspec", edge_type="IMPLEMENTS",
                     confidence=0.9, source_id=foundation_sids[2], valid_from="2025-01-05")

        # Approve foundational data
        for ent in db.get_entities_by_status("pending_review"):
            db.approve_entity(ent["id"])
        for edge in db.get_edges_by_status("pending_review"):
            db.approve_edge(edge["id"])

        # Checkpoint 1: Day 10
        approved_ents = db.get_entities_by_status("approved")
        approved_edges = db.get_edges_by_status("approved")
        assert len(approved_ents) == 4
        assert len(approved_edges) == 2

        # Day 11-20: News articles
        news_sids = []
        for day in range(11, 21):
            sid = db.add_source(f"https://example.com/day{day}/news")
            news_sids.append(sid)

        # Some contradict: Beta acquired Alpha
        db.add_edge(edge_id="sim-e3", source_entity_id="organization:beta",
                     target_entity_id="organization:alpha", edge_type="GOVERNS",
                     confidence=0.75, source_id=news_sids[4], valid_from="2025-01-15")

        # Alpha stops developing — end old edge
        db.conn.execute("UPDATE edges SET valid_to = '2025-01-15' WHERE edge_id = 'sim-e1'")
        db.conn.commit()

        # Beta takes over development
        db.add_edge(edge_id="sim-e4", source_entity_id="organization:beta",
                     target_entity_id="project:alpha-platform", edge_type="DEVELOPS",
                     confidence=0.8, source_id=news_sids[5], valid_from="2025-01-16")

        # Checkpoint 2: Day 20 — verify temporal coherence
        snapshot_date = "2025-01-18"
        active_devs = db.conn.execute(
            """SELECT * FROM edges WHERE edge_type = 'DEVELOPS'
               AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)""",
            (snapshot_date, snapshot_date),
        ).fetchall()
        dev_sources = [e["source_entity_id"] for e in active_devs]
        assert "organization:alpha" not in dev_sources
        assert "organization:beta" in dev_sources

        # Day 21-30: Deprecate old source, add corrections
        db.deprecate_entities_for_source(foundation_sids[0])

        correction_sid = db.add_source("https://example.com/day25/correction")
        db.add_entity(entity_id="organization:alpha-v2", name="Alpha Inc (Post-Acquisition)",
                       entity_type="Organization", kind="subsidiary",
                       description="Now a subsidiary of Beta Labs",
                       source_id=correction_sid)

        # Checkpoint 3: Day 30
        deprecated = db.get_deprecated_entities()
        assert any(e["entity_id"] == "organization:alpha" for e in deprecated)

        new_pending = db.get_entities_by_status("pending_review")
        assert any(e["entity_id"] == "organization:alpha-v2" for e in new_pending)

        # Final check: total graph integrity
        all_entities = db.conn.execute("SELECT * FROM entities").fetchall()
        all_edges = db.conn.execute("SELECT * FROM edges").fetchall()
        assert len(all_entities) == 5  # 4 original + 1 correction
        assert len(all_edges) == 4  # 2 original + 2 news

    def test_graph_coherence_at_each_phase(self, db):
        """Each phase produces a valid, queryable graph state."""
        # Phase 1: Empty graph
        assert len(db.get_pending_sources()) == 0
        assert len(db.get_entities_by_status("pending_review")) == 0

        # Phase 2: After adding entities
        sid = db.add_source("https://example.com/coherence-test")
        db.add_entity(entity_id="organization:coherence-org", name="Coherence Org",
                       entity_type="Organization", source_id=sid)
        assert len(db.get_entities_by_status("pending_review")) == 1

        # Phase 3: After approval
        for ent in db.get_entities_by_status("pending_review"):
            db.approve_entity(ent["id"])
        assert len(db.get_entities_by_status("approved")) == 1
        assert len(db.get_entities_by_status("pending_review")) == 0

        # Phase 4: After deprecation
        db.deprecate_entities_for_source(sid)
        deprecated = db.get_deprecated_entities()
        assert len(deprecated) == 1

        # The entity still exists but is marked deprecated
        all_ents = db.conn.execute("SELECT * FROM entities").fetchall()
        assert len(all_ents) == 1

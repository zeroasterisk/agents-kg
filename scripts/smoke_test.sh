#!/bin/bash
# End-to-end smoke test for the agents-kg Wikidata integration.
# Runs the full pipeline: schema → seed → wikidata pull → load-yaml → crossref → events → demo queries.
#
# Usage:
#   ./scripts/smoke_test.sh             # Full test (requires Neo4j)
#   ./scripts/smoke_test.sh --dry-run   # SPARQL + YAML validation only (no Neo4j needed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

KG="${PROJECT_DIR}/.venv/bin/kg"

if [[ ! -x "$KG" ]]; then
    KG="${PROJECT_DIR}/.venv/bin/python3 -m agents_kg.cli"
fi

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== SMOKE TEST (dry-run mode, no Neo4j required) ==="
else
    echo "=== SMOKE TEST ==="
fi

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  ✓ $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  ✗ $1"; }

echo ""

# Step 1: Apply schema constraints (skip in dry-run)
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 1: Applying Neo4j schema ---"
    $KG schema && pass "schema applied" || fail "schema apply"
    echo ""
fi

# Step 2: Load seed entities (skip in dry-run)
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 2: Loading seed entities ---"
    $KG seed && pass "seed loaded" || fail "seed load"
    echo ""
fi

# Step 3: Pull programming languages from Wikidata
echo "--- Step 3: Pulling programming languages from Wikidata ---"
$KG wikidata pull --type languages $DRY_RUN && pass "wikidata pull" || fail "wikidata pull"
echo ""

# Step 4: Load YAML entities and relations (skip in dry-run)
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 4: Loading YAML entities and relations ---"
    $KG load-yaml && pass "yaml entities loaded" || fail "yaml entity load"
    echo ""
fi

# Step 5: Apply Wikidata cross-references
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 5: Applying Wikidata cross-references ---"
    $KG wikidata crossref && pass "crossref applied" || fail "crossref"
    echo ""
fi

# Step 6: Load events
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 6: Loading events ---"
    $KG events load && pass "events loaded" || fail "events load"
    echo ""
fi

# Step 7: Run demo queries and stats (only with Neo4j)
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 7: Running demo queries ---"

    .venv/bin/python3 - <<'PYEOF'
import sys
from neo4j import GraphDatabase

d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agents-kg-2026"))
failures = 0

with d.session() as s:
    # Graph stats
    print("Graph stats:")
    rows = s.run("""
        MATCH (n:Entity)
        WITH n.type AS type, COUNT(*) AS count,
             SUM(CASE WHEN n.wikidata_id IS NOT NULL THEN 1 ELSE 0 END) AS with_wikidata
        RETURN type, count, with_wikidata ORDER BY count DESC
    """).data()
    for r in rows:
        print(f"  {r['type']}: {r['count']} ({r['with_wikidata']} with wikidata)")
    if not rows:
        print("  ERROR: no entities found")
        failures += 1

    # Event count
    evt = s.run("MATCH (e:Event) RETURN COUNT(e) AS cnt").single()
    print(f"\nEvents: {evt['cnt']}")
    if evt["cnt"] == 0:
        print("  WARNING: no events loaded")

    # Edge stats
    print("\nEdge types:")
    edges = s.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS cnt ORDER BY cnt DESC").data()
    for r in edges:
        print(f"  {r['type']}: {r['cnt']}")

    # Cross-domain query: Cisco->AGNTCY->Linux Foundation
    print("\nCross-domain query (Cisco->AGNTCY->LF):")
    chain = s.run("""
        MATCH (c:Entity)-[:CREATED]->(a)-[:DONATED_TO]->(lf)
        WHERE c.entity_id ENDS WITH 'cisco'
        RETURN c.name, a.name, lf.name, lf.wikidata_id
    """).data()
    if chain:
        for r in chain:
            print(f"  {r['c.name']} -> {r['a.name']} -> {r['lf.name']} (wd:{r['lf.wikidata_id']})")
    else:
        print("  WARNING: cross-domain chain not found")

    # Verify no duplicate entity_ids
    dupes = s.run("""
        MATCH (n:Entity)
        WITH n.entity_id AS eid, count(*) AS cnt
        WHERE cnt > 1
        RETURN eid, cnt LIMIT 5
    """).data()
    if dupes:
        print(f"\nERROR: duplicate entity_ids found: {dupes}")
        failures += 1
    else:
        print("\nNo duplicate entity_ids (integrity check passed)")

d.close()
sys.exit(1 if failures > 0 else 0)
PYEOF

    if [[ $? -eq 0 ]]; then
        pass "demo queries"
    else
        fail "demo queries"
    fi
    echo ""
fi

echo "=== SMOKE TEST COMPLETE ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

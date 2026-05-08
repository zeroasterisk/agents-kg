#!/bin/bash
# End-to-end smoke test for the agents-kg Wikidata integration.
# Runs the full pipeline: schema → wikidata pull → crossref → events → demo queries.
#
# Usage:
#   ./scripts/smoke_test.sh             # Full test (requires Neo4j)
#   ./scripts/smoke_test.sh --dry-run   # SPARQL + YAML validation only (no Neo4j needed)
set -euo pipefail

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== SMOKE TEST (dry-run mode, no Neo4j required) ==="
else
    echo "=== SMOKE TEST ==="
fi

echo ""

# Step 1: Apply schema constraints (skip in dry-run)
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 1: Applying Neo4j schema ---"
    kg schema
    echo ""
fi

# Step 2: Pull programming languages from Wikidata
echo "--- Step 2: Pulling programming languages from Wikidata ---"
kg wikidata pull --type languages $DRY_RUN
echo ""

# Step 3: Apply Wikidata cross-references
echo "--- Step 3: Applying Wikidata cross-references ---"
kg wikidata crossref
echo ""

# Step 4: Load events
echo "--- Step 4: Loading events ---"
kg events load
echo ""

# Step 5: Run demo queries (only with Neo4j)
if [[ -z "$DRY_RUN" ]]; then
    echo "--- Step 5: Running demo queries ---"

    CYPHER_QUERY='MATCH (n:Entity) WITH n.type AS type, COUNT(*) AS count, SUM(CASE WHEN n.wikidata_id IS NOT NULL THEN 1 ELSE 0 END) AS with_wikidata RETURN type, count, with_wikidata ORDER BY count DESC'
    echo "Graph stats:"
    cypher-shell -u neo4j -p agents-kg-2026 "$CYPHER_QUERY" 2>/dev/null || echo "  (cypher-shell not available, use Neo4j Browser to run demo queries from docs/demo-queries.md)"

    echo ""
    echo "Event count:"
    cypher-shell -u neo4j -p agents-kg-2026 "MATCH (e:Event) RETURN COUNT(e) AS event_count" 2>/dev/null || echo "  (skipped)"
fi

echo ""
echo "=== SMOKE TEST COMPLETE ==="

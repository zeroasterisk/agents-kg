# Entity Resolution — Redesigned Pipeline

## The Problem I Missed

The v0.1 pipeline does blind inserts: extract entities → dump into SQLite. No lookup, no dedup, no resolution. This produces garbage at scale — duplicate entities with slightly different names, no edge updates, no lifecycle respect.

Entity resolution during ingestion is **the hard problem** of KG construction. Graphiti spent significant engineering on this. I need to steal their architecture properly.

## Graphiti's Resolution Architecture (from source code review)

### `resolve_extracted_nodes()` — the core function

```
For each newly extracted entity:

1. COLLECT CANDIDATES
   - Hybrid search (text overlap + embedding similarity) against existing graph
   - Returns candidate nodes that might be the same entity

2. BUILD INDEXES (once per batch)
   - Exact name index (normalized lowercase + whitespace collapse)
   - MinHash signatures + LSH buckets for fuzzy matching
   - Shingle cache (LRU) for repeated comparisons

3. DETERMINISTIC PASS (no LLM, no cost)
   For each extracted entity:
   a. Compute Shannon entropy of name
      - Low entropy (short/repetitive like "AI") → skip to LLM
      - High entropy → proceed with heuristics
   b. Exact match on normalized name → resolved ✅
   c. MinHash + LSH → candidate set
   d. Jaccard similarity on 3-gram shingles
      - Score ≥ 0.9 → resolved ✅
      - Score < 0.9 → unresolved, escalate

4. LLM PASS (only for unresolved entities)
   - Send unresolved entities + candidate list to LLM
   - LLM returns: "entity X is duplicate of existing Y" or "entity X is new"
   - Guardrails: validate LLM response IDs, skip malformed/duplicate responses
```

### For edges: `resolve_extracted_edges()`
- Text overlap + embedding similarity to find existing edges
- Hybrid search with RRF (Reciprocal Rank Fusion)
- Contradiction detection between new and existing edges

### Key insight: the graph IS the resolution index
Graphiti uses the graph database itself as the lookup layer. `_collect_candidate_nodes()` does a **hybrid search per extracted entity name** against the graph. This is where Neo4j earns its keep — it's not just storage, it's the resolution engine.

## Data Fan-Out Analysis

For a single source document:
```
1 source
  → ~10-20 chunks (after parsing)
    → ~5-15 entities extracted per chunk (with overlap)
      → ~50-200 raw entity mentions per source
        → Each needs resolution against ALL existing entities
```

At 100 entities in the graph:
- 200 mentions × 100 existing = 20,000 comparisons (deterministic, fast)
- ~10-30 unresolved → LLM calls (expensive, slow)

At 1,000 entities:
- 200 × 1,000 = 200,000 comparisons (still fast with indexes)
- ~10-30 LLM calls (scales with ambiguity, not graph size)

At 10,000 entities:
- Exact match + LSH keeps it O(n) per entity, not O(n²)
- LLM calls stay ~10-30 per source (only truly ambiguous cases)

### Cost per source (estimated)
| Step | Gemini Flash calls | Embedding calls | Cost |
|------|-------------------|-----------------|------|
| Extract entities | 1 per chunk (~15) | 0 | ~$0.01 |
| Extract edges | 1 per chunk (~15) | 0 | ~$0.01 |
| Entity resolution (deterministic) | 0 | 0 | $0 |
| Entity resolution (LLM fallback) | 1 batch call | 0 | ~$0.005 |
| Embed chunks | 0 | 1 batch | ~$0.001 |
| **Total per source** | **~31** | **1** | **~$0.025** |

At 500 sources: ~$12.50. Manageable.

### Where it gets expensive
- **Summarization**: Graphiti also summarizes entities (merging descriptions across sources). Each summary update = 1 LLM call. This adds up.
- **Edge contradiction detection**: Checking if new edge contradicts existing edges = LLM calls.
- **Re-resolution on updates**: When a source changes, re-extract + re-resolve.

## What SQLite Can Do (honestly)

**Works fine:**
- Exact name matching (indexed column lookup)
- Normalized name index
- Alias matching (`aliases` JSON column with LIKE or JSON_EACH)
- Content hash for idempotency
- Job queue, pipeline state, retry logic

**Works but clunky:**
- Fuzzy matching (MinHash/LSH can be done in Python, indexed in SQLite)
- The candidate search step — in Graphiti this is a hybrid search against the graph. In SQLite it's a full-table scan + Python filtering.

**Doesn't work well:**
- Graph neighborhood checks ("what's connected to this entity?")
- Multi-hop traversal queries for analysis
- Vector similarity search for embeddings
- The visualization and exploration you want

## Revised Pipeline Design

```
fetch → parse → chunk → embed → extract → RESOLVE → review → load
                                              ↑
                                    NEW: entity resolution
```

### Resolution step (new)

```python
async def resolve(extracted_entities, extracted_edges, db):
    # 1. Build candidate index from existing entities
    existing = db.get_all_entities()  # SQLite for now, Neo4j later
    indexes = build_candidate_indexes(existing)
    
    # 2. Deterministic pass
    resolved, unresolved = resolve_with_similarity(extracted_entities, indexes)
    
    # 3. LLM pass (only unresolved)
    if unresolved:
        llm_resolved = await resolve_with_llm(unresolved, indexes)
        resolved.extend(llm_resolved)
    
    # 4. For each resolved entity:
    #    - If matched existing: UPDATE (merge descriptions, add aliases)
    #    - If new: INSERT with status=pending_review
    #    - If ambiguous: INSERT with status=ambiguous
    
    # 5. Edge resolution (same pattern)
    resolve_edges(extracted_edges, resolved_entities, db)
```

### Efficiency gains from writing things down

**Alias accumulation**: Every time we resolve "Agent-to-Agent Protocol" to existing entity "A2A", add "Agent-to-Agent Protocol" to its aliases. Next time, exact match hits.

**Confidence accumulation**: Same relationship asserted by multiple sources → confidence increases. Multiple chunk_ids as provenance.

**Embedding cache**: Store entity name embeddings in SQLite. Use for candidate search without hitting Gemini API.

**Intra-batch dedup**: Before resolving against the graph, dedup within the current batch (Graphiti does this as a second pass).

## SQLite vs Neo4j — Honest Assessment for Resolution

| Capability | SQLite | Neo4j | Verdict |
|-----------|--------|-------|---------|
| Exact name lookup | ✅ Fast (index) | ✅ Fast | Tie |
| Alias search | ⚠️ JSON_EACH, slower | ✅ Array index | Neo4j slightly better |
| Fuzzy matching | ⚠️ Python-side | ⚠️ Python-side | Tie (both use Python) |
| MinHash/LSH | ✅ Python + SQLite cache | ✅ Python + Neo4j cache | Tie |
| Candidate search (hybrid) | ❌ Full scan + filter | ✅ Native hybrid search | Neo4j wins |
| Graph neighborhood | ❌ Multiple JOINs | ✅ Single traversal | Neo4j wins |
| Vector similarity | ❌ No native support | ✅ Vector index | Neo4j wins |
| Embedding storage | ⚠️ BLOB column | ✅ Vector property | Neo4j better |

**Bottom line**: SQLite handles the deterministic fast path fine. But candidate collection (the search step that finds potential matches) is where Neo4j genuinely helps — especially as the graph grows past ~500 entities.

## Recommendation

**Phase 1 (now, SQLite-only):**
- Implement the resolution step with deterministic matching (exact + MinHash/LSH)
- LLM fallback for ambiguous cases
- Alias accumulation
- This works for our current scale (~100s of entities)

**Phase 2 (when Alan spins up Neo4j):**
- Move candidate search to Neo4j hybrid search
- Add vector similarity for entity matching
- Enable graph neighborhood checks
- Visualization

**The resolution module should be backend-agnostic** — a `CandidateSearcher` interface that can be backed by SQLite queries or Neo4j searches. Same resolution logic either way.

## What I'm Building

1. `src/agents_kg/resolve.py` — Entity resolution module
   - `build_candidate_indexes()` — MinHash/LSH index builder
   - `resolve_with_similarity()` — deterministic pass
   - `resolve_with_llm()` — LLM fallback
   - `resolve_edges()` — edge resolution
2. Update pipeline to insert `resolve` between `extract` and `review`
3. Add alias accumulation to entity updates
4. Add confidence accumulation for edges
5. Tests for exact match, fuzzy match, ambiguous cases

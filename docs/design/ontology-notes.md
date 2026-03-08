# Ontology Design Notes

## Core Principles (from Alan)

### 1. Temporal edges (Graphiti/Zep-inspired)
- Relationships change over time — projects adopt protocols, people change orgs
- Every edge should have `valid_from` / `valid_to` timestamps
- Append-only history: never delete an edge, just set `valid_to`
- Enables "state at time T" queries and trend analysis

### 2. Source chunk provenance (non-negotiable)
- Every edge MUST reference the source chunk that asserted it
- Source = original URL, document, post, conversation
- Chunks = the specific excerpt/paragraph that contains the assertion
- This is for:
  - **Verification**: "why do you think Project X implements Protocol Y?"
  - **Confidence**: weak source → low confidence edge
  - **Updates**: when source is updated, re-extract and compare
  - **Dedup**: multiple sources asserting same relationship = higher confidence

### 3. Comparison & dedup is the killer use case
- Many projects solve the same problems differently
- KG should enable: "show me all projects that solve [concept] and how they differ"
- Edge properties should capture *how* something is solved, not just *that* it's solved
- e.g. "Project X implements identity via OAuth2 tokens" vs "Project Y uses DID-based identity"

## ETL Design Constraints
- Graphiti was flaky in practice — reliability matters more than features
- Simpler is better for now
- Alan guides intake — no autonomous scraping yet
- Need to plan congestion management for batch ingestion
- Options still open: ADK, adk-elixir, Beam, Temporal/Restate
- **Priority: get out of the way** — tooling should not slow down the human curator

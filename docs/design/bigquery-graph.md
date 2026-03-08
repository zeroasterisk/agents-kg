# BigQuery Graph for Knowledge Graphs

## Source

3-part blog series by Rachael Deacon-smith (Google Cloud), Feb 2026:
- [Part 1: From "Dark Data" to Knowledge Graphs](https://medium.com/google-cloud/bigquery-graph-series-part-1-from-dark-data-to-knowledge-graphs-5a37f052d043)
- [Part 2: Tutorial — Build a Graph from unstructured data](https://medium.com/@rachaelds/bigquery-graph-series-6a768ccb351b)
- [Part 3: Query and Visualize your Graph](https://medium.com/@rachaelds/bigquery-graph-series-2e35bb203aac)
- [Companion notebook](https://github.com/GoogleCloudPlatform/devrel-demos/blob/main/data-analytics/knowledge_graph_demo/kg_demo_template.ipynb)

**Status:** BigQuery Graph is in **private preview** ([sign up](http://tinyurl.com/bq-graph)).

## Architecture

Hybrid AI pipeline, entirely within BigQuery:

```
Unstructured docs (GCS)
  → Object Table (BigQuery)
    → ML.PROCESS_DOCUMENT (Document AI Layout Parser)
      → Clean chunks with structural context
        → AI.GENERATE (Gemini) with output_schema
          → Node tables + Edge tables
            → CREATE PROPERTY GRAPH
```

### Key Components

1. **Document AI Layout Parser** — Not just OCR. Preserves headers, tables, lists, multi-column layouts. Produces context-aware chunks (e.g. "Section 4.1") rather than arbitrary token splits. Gotcha: 130 page limit, 120s timeout per doc.

2. **AI.GENERATE with output_schema** — Calls Gemini to extract entities + relationships. The `output_schema` parameter enforces strict JSON structure (no parsing LLM outputs). You define valid entity types and relationship types in the prompt.

3. **CREATE PROPERTY GRAPH** — DDL that maps existing BigQuery tables into a logical graph:
   - NODE TABLES with KEY and LABEL
   - EDGE TABLES with SOURCE KEY, DESTINATION KEY, and LABEL
   - Can mix extracted graph data with existing structured tables (customers, products, etc.)

4. **GQL (Graph Query Language)** — ISO-standard, Cypher-like syntax:
   - `(nodes)` in parentheses, `[edges]` in brackets
   - `->` for direction, `{}` for inline filters
   - Multi-hop traversals in a single readable line vs. complex JOINs
   - `%%bigquery --graph` magic command for notebook visualization

## Relevance to agents-kg

### What fits well
- **Scale**: BQ handles billions of nodes / tens of billions of edges natively
- **Unified warehouse**: Graph lives alongside structured data — can JOIN graph traversals with regular tables
- **SQL-native**: No separate graph DB to manage, familiar tooling
- **MCP integration**: BQ already has [MCP tools](https://docs.cloud.google.com/bigquery/docs/pre-built-tools-with-mcp-toolbox) for agent queries
- **Vector search**: Can add embeddings as node properties for hybrid search (semantic + structural)
- **Visualization**: Built-in graph viz in BQ notebooks

### What doesn't fit (yet)
- **Private preview** — need allowlist access
- **Temporal edges** — BQ Graph has no native temporal/versioning model like Graphiti. We'd need to model this ourselves (valid_from/valid_to on edge tables, or append-only edge history)
- **Source chunk references** — achievable by storing chunk_id/doc_uri as edge properties, but requires deliberate schema design
- **GQL + LLMs** — LLMs are undertrained on GQL vs SQL. Need few-shot examples in prompts for NL2GQL
- **Cost** — AI.GENERATE + Document AI have per-call costs. Need to estimate for our corpus size

## Design Considerations for agents-kg

### Temporal edges (Graphiti-inspired)
Alan wants edges that reference source chunks and have temporal validity. In BQ Graph:

```sql
-- Edge table with provenance and temporality
CREATE TABLE edges_project_uses_protocol (
  project_id STRING,
  protocol_id STRING,
  source_chunk_id STRING,      -- links back to the chunk that asserted this
  source_uri STRING,            -- original document/URL
  confidence FLOAT64,           -- extraction confidence
  valid_from TIMESTAMP,         -- when this relationship was first observed
  valid_to TIMESTAMP,           -- NULL = still valid
  extracted_at TIMESTAMP        -- when we extracted it
);
```

This gives us:
- **Provenance**: Every edge traces back to its source material
- **Temporality**: Can query "what was the state at time T" by filtering valid_from/valid_to
- **Confidence**: Can filter low-confidence extractions
- **Audit trail**: extracted_at for pipeline debugging

### Node types for agentic web KG
- Organization (Google, Cisco, Microsoft, AAIF, AGNTCY...)
- Person (named individuals — private repo)
- Project (ADK, LangGraph, CrewAI...)
- Protocol (A2A, MCP, ACP...)
- Concept (identity, tool use, streaming, discovery...)
- Document (specs, blog posts, presentations...)

### Edge types
- DEVELOPS (Org/Person → Project)
- IMPLEMENTS (Project → Protocol)
- COMPETES_WITH (Project → Project)
- SOLVES (Project → Concept)
- MEMBER_OF (Person → Org)
- AUTHORED (Person → Document)
- REFERENCES (Document → Protocol/Project/Concept)
- SUCCEEDS / SUPERSEDES (Protocol → Protocol)

## ETL Options (for discussion)

Alan's notes: Graphiti was flaky in practice. Options range from simple to complex:

| Option | Complexity | Fit |
|--------|-----------|-----|
| **Manual + Gemini** | Low | Good for now — I curate sources, Gemini extracts, BQ stores |
| **ADK pipeline** | Medium | ADK agent with tools for fetch → extract → load |
| **adk-elixir** | Medium | GenStage/Broadway for backpressure, OTP for reliability |
| **Apache Beam** | High | Overkill unless we need massive parallel ingestion |
| **Temporal/Restate** | High | Good for complex workflows, but adds infra |

**Recommendation for MVP**: Start with manual curation + a simple ADK extract-and-load agent. Graduate to something with backpressure (Beam or Elixir Broadway) only when volume demands it.

## Next Steps
- [ ] Sign up for BQ Graph private preview
- [ ] Sprint on ontology design (Alan has ideas)
- [ ] Prototype: manual ingest of 5-10 sources → extract → BQ Graph
- [ ] Test GQL queries for comparison use cases ("who solves identity?")
- [ ] Evaluate MCP interface for agent-queryable KG

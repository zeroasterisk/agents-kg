# Graph Backend Comparison: Neo4j vs BigQuery Graph

## Overview

| | **Neo4j** | **BigQuery Graph** |
|---|---|---|
| **Type** | Native graph database | Graph layer on columnar warehouse |
| **Query lang** | Cypher (openCypher) | GQL (ISO standard) |
| **Maturity** | 15+ years, massive ecosystem | Private preview (Feb 2026) |
| **Hosting** | Docker, AuraDB (managed), GCP/AWS/Azure | GCP only |
| **Cost** | Free (Community), ~$65/mo (AuraDB), or self-hosted Docker | BQ pricing (storage + query bytes) |
| **Graphiti/Zep** | Native target — Graphiti is built on Neo4j | Would need custom implementation |

## Neo4j Strengths

### Native graph storage & traversal
- Purpose-built for graph operations — index-free adjacency means traversals don't slow down with scale
- Multi-hop queries are first-class, not bolted onto a relational engine

### Ecosystem & tooling
- **Graphiti/Zep**: Temporal KG framework built directly on Neo4j — exactly what Alan wants (temporal edges, source provenance, episodic + semantic memory)
- **GraphRAG**: Neo4j is the default backend for most GraphRAG implementations
- **APOC**: Massive utility library (import/export, AI extraction, text processing)
- **Graph Data Science (GDS)**: PageRank, community detection, node embeddings, link prediction — built-in
- **Bloom**: Visual graph exploration (like BQ notebook viz but more mature)
- **Neo4j Browser**: Interactive Cypher shell with visualization
- **LangChain/LlamaIndex**: Direct Neo4j integrations for KG-backed RAG

### Temporal modeling
Graphiti on Neo4j gives us temporal edges natively:
- Episodic nodes (raw source chunks) linked to entity nodes
- Edges have `created_at`, `valid_at`, `invalid_at` timestamps
- Entity nodes deduplicated with temporal history
- This is exactly the "edges referencing source chunks" requirement

### Self-hosting
- `docker run neo4j:community` — done
- Runs great on the NAS alongside everything else
- Full control, no preview waitlist, no per-query billing

### MCP / Agent integration
- Neo4j has official MCP server (`@neo4j/mcp-neo4j`)
- Cypher is well-understood by LLMs (vs GQL which is too new)

## BigQuery Graph Strengths

### Scale & unified warehouse
- Billions of nodes / tens of billions of edges
- Graph lives alongside structured tables — can JOIN graph traversals with business data
- No ETL between graph DB and analytics warehouse

### SQL-native pipeline
- `ML.PROCESS_DOCUMENT` → `AI.GENERATE` → `CREATE PROPERTY GRAPH` — all in SQL
- Document AI for structured parsing, Gemini for extraction
- No external pipeline code needed for simple cases

### GQL (ISO standard)
- ISO/IEC 39075 — the "official" graph query language
- Forward-looking: other databases will adopt GQL
- Cypher-like syntax but standardized

### Cost model
- Pay per query (bytes scanned) — good for bursty/low-frequency use
- No idle server cost
- But: AI.GENERATE calls add up for extraction

### MCP tools
- BQ already has MCP toolbox for agent queries
- Vector search built-in for hybrid (semantic + structural) queries

## Neo4j Weaknesses

- **Memory-bound**: Large graphs need RAM. Community edition has no clustering
- **No built-in vector search** (need plugin or separate index — though Neo4j 5.x added vector indexes)
- **Cypher ≠ SQL**: Can't easily JOIN with structured/tabular data
- **Enterprise features gated**: Clustering, role-based security, etc. require Enterprise license
- **Graphiti flakiness**: Alan experienced reliability issues — extraction pipeline was brittle

## BigQuery Graph Weaknesses

- **Private preview** — can't use yet without allowlist
- **No temporal model** — must build valid_from/valid_to ourselves
- **No Graphiti equivalent** — would need to build the episodic memory layer from scratch
- **GQL is new** — LLMs can't generate it reliably, small community, few examples
- **Not a native graph engine** — traversals are compiled to columnar operations, may not match Neo4j for deep multi-hop queries
- **GCP lock-in** — only runs on GCP

## Portability Strategy

### The key insight: separate the data model from the storage engine

```
┌─────────────────────────────────┐
│  Canonical Data Model (YAML/JSON) │  ← Source of truth
│  Nodes + Edges + Provenance      │
└──────────┬──────────┬────────────┘
           │          │
     ┌─────▼────┐ ┌───▼──────────┐
     │  Neo4j   │ │ BigQuery Graph│
     │ (Cypher) │ │ (GQL/SQL)    │
     └──────────┘ └──────────────┘
```

### Portable primitives
If we define our nodes and edges as simple typed records:

```yaml
# Node
- id: "google"
  type: Organization
  properties:
    name: "Google"
    url: "https://google.com"

# Edge
- source: "google"
  target: "a2a"
  type: DEVELOPS
  properties:
    source_chunk_id: "chunk_abc123"
    source_uri: "https://..."
    valid_from: "2024-03-01"
    valid_to: null
    confidence: 0.95
```

Then we can write importers for either backend:
- **Neo4j**: `MERGE (n:Organization {id: $id}) SET n += $props` + `MERGE (a)-[r:DEVELOPS]->(b) SET r += $edge_props`
- **BigQuery**: INSERT into node/edge tables, `CREATE PROPERTY GRAPH` over them

### Import/export paths
- **Neo4j → BQ**: APOC export to CSV/JSON → load into BQ tables
- **BQ → Neo4j**: BQ export to GCS (JSON/CSV) → `neo4j-admin import` or APOC load
- **Both ← canonical**: Load from YAML/JSON flat files (git-tracked)

### What changes between backends
| Concern | Neo4j | BigQuery Graph |
|---------|-------|----------------|
| Temporal queries | Graphiti handles it | Custom SQL/GQL with valid_from/valid_to |
| Source provenance | Episodic nodes (Graphiti) | Edge properties + chunk table |
| Vector search | Neo4j vector index (5.x) | BQ vector search (native) |
| Graph algorithms | GDS library | Would need external compute |
| Multi-hop traversal | Native, fast | Compiled to columnar ops |
| Analytics/aggregation | Weak (not its strength) | Strong (it's a warehouse) |

## Recommendation

**Start with Neo4j (Docker on NAS) + canonical flat files.**

Rationale:
- We can start TODAY (no waitlist)
- Graphiti gives us temporal edges + source provenance out of the box
- Massive ecosystem expects Neo4j (GraphRAG, LangChain, ADK tools)
- Self-hosted Docker = free, full control
- Canonical YAML/JSON files in git = portability insurance
- When BQ Graph goes GA, we can load the same data there for analytics/scale

**Mitigate Graphiti flakiness** by:
- Using it only for the KG storage layer (not the full extraction pipeline)
- Building our own extraction/ETL that produces clean node/edge records
- Feeding those records into Neo4j via simple Cypher MERGE (bypassing Graphiti's extraction)
- Using Graphiti's temporal model but our own ingestion

### Docker quick-start
```bash
docker run -d \
  --name agents-kg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v $HOME/.openclaw/data/neo4j:/data \
  -e NEO4J_AUTH=neo4j/agents-kg-2026 \
  neo4j:community
```

## Next Steps
- [ ] Alan signs up for BQ Graph preview
- [ ] Spin up Neo4j Docker on NAS (when ready)
- [ ] Define canonical node/edge schema in YAML
- [ ] Build import scripts for Neo4j (Cypher MERGE)
- [ ] Prototype: load 5-10 entities manually, test traversals
- [ ] Evaluate Graphiti as temporal layer vs rolling our own
- [ ] Plan BQ Graph loader for when preview access arrives

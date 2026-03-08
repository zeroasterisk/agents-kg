# Data Architecture

## Core Principle

**Sources are durable. Embeddings are ephemeral.**

Sources and chunks are the permanent record — stored so we can always regenerate derived data (embeddings, extracted entities) when models improve or strategies change.

## Three-Layer Model

```
┌─────────────────────────────────────────────┐
│  Layer 1: Sources (durable)                 │
│  URL, raw text, fetch date, type            │
├─────────────────────────────────────────────┤
│  Layer 2: Chunks (durable)                  │
│  text, position, chunk strategy, source_id  │
│  + embedding vector (ephemeral property)    │
├─────────────────────────────────────────────┤
│  Layer 3: Knowledge Graph (durable)         │
│  Entities + Edges, temporal, with           │
│  provenance links back to chunks            │
└─────────────────────────────────────────────┘
```

## Neo4j Schema

### Source nodes
```cypher
CREATE CONSTRAINT source_uri IF NOT EXISTS
FOR (s:Source) REQUIRE s.uri IS UNIQUE;

// Properties:
// uri: STRING (canonical URL or file path)
// type: STRING (blog, spec, conversation, repo, paper, post)
// title: STRING
// author: STRING (if known)
// fetched_at: DATETIME
// published_at: DATETIME (if known)
// raw_text: STRING
// content_hash: STRING (SHA256 of raw_text, for change detection)
```

### Chunk nodes
```cypher
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// Properties:
// id: STRING (source_uri + position hash)
// text: STRING
// position: INTEGER (order within source)
// chunk_strategy: STRING (e.g. "semantic_sections_v1")
// token_count: INTEGER
// embedding: LIST<FLOAT> (ephemeral — regenerated on model change)
// embedding_model: STRING (e.g. "text-embedding-005")
// embedded_at: DATETIME

// Relationship:
// (chunk)-[:FROM_SOURCE]->(source)
```

### Vector index on chunks
```cypher
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};
```

### Entity nodes (per ontology)
```cypher
// :Organization, :Person, :Project, :Protocol, :Concept, :Document
// Each has:
// id: STRING (slug, e.g. "google", "a2a", "mcp")
// name: STRING (display name)
// description: STRING
// aliases: LIST<STRING> (for dedup matching)
// created_at: DATETIME
// updated_at: DATETIME
```

### Edges (temporal, with provenance)
```cypher
// (org)-[:DEVELOPS {since, until, chunk_id, confidence}]->(project)
// (project)-[:IMPLEMENTS {since, until, chunk_id, how}]->(protocol)
// (project)-[:SOLVES {since, until, chunk_id, approach}]->(concept)
// etc.

// All edges have:
// chunk_id: STRING (provenance — which chunk asserted this)
// confidence: FLOAT (extraction confidence)
// valid_from: DATETIME (when relationship started)
// valid_to: DATETIME (null = still valid)
// extracted_at: DATETIME (when we extracted it)
```

## Embedding Strategy

### Current: Gemini text-embedding-005
- Dimensions: 768
- Use for: chunk similarity, semantic search, clustering
- API: `vertexai.language_models.TextEmbeddingModel`

### Migration plan
When new Gemini embedding model releases:
1. Batch-iterate all Chunk nodes
2. Re-embed with new model
3. Update `embedding`, `embedding_model`, `embedded_at` properties
4. Recreate vector index if dimensions change
5. Sources and chunks unchanged — only the ephemeral vector property updates

### Cost awareness
- Embedding is cheap (batch API, ~$0.0001/1K tokens)
- But re-embedding entire corpus = plan for it
- Track `embedding_model` on each chunk so we can do incremental migration

## RAG Strategies

### 1. Vector RAG (semantic similarity)
```cypher
// Find chunks similar to a query
CALL db.index.vector.queryNodes('chunk_embedding', 10, $query_embedding)
YIELD node AS chunk, score
RETURN chunk.text, score
```

### 2. Graph RAG (structured traversal)
```cypher
// Multi-hop: what protocols does Google develop projects that implement?
MATCH (o:Organization {id: 'google'})-[:DEVELOPS]->(p:Project)-[:IMPLEMENTS]->(pr:Protocol)
RETURN p.name, pr.name
```

### 3. Hybrid RAG (vector → graph)
```cypher
// Find chunks about "identity", then traverse to related entities
CALL db.index.vector.queryNodes('chunk_embedding', 5, $identity_embedding)
YIELD node AS chunk
MATCH (chunk)<-[:EXTRACTED_FROM]-(e)
MATCH (e)-[r]->(related)
RETURN e.name, type(r), related.name
```

### 4. Clustering (community detection)
```cypher
// Project graph into GDS, run community detection
CALL gds.graph.project('entities', ['Project', 'Protocol'], 'IMPLEMENTS');
CALL gds.louvain.write('entities', {writeProperty: 'community'});
// → groups of projects that implement similar protocols
```

### 5. Dedup / Disambiguation
- **Vector similarity**: chunks mentioning same entity will cluster
- **Alias matching**: entity.aliases for known synonyms
- **Graph neighborhood**: entities with similar edge patterns are likely duplicates
- **Candidate pairs** → human review → merge

## Canonical Files (Git)

Alongside Neo4j, we keep canonical YAML files as the source of truth:

```
kg/
  sources/           # fetched source metadata (not raw text — too large)
  entities/
    organizations/
    people/
    projects/
    protocols/
    concepts/
  edges/             # relationship records with provenance
```

These serve as:
- Backup / portability (can reload Neo4j from scratch)
- Git history = audit trail
- Human-editable for corrections
- Import into BigQuery Graph later

Raw source text stored in Neo4j (or GCS if too large for git).

## Pipeline (simple-first)

```python
# 1. Fetch
source = fetch(url)  # → {uri, raw_text, title, type, ...}

# 2. Chunk
chunks = chunk(source)  # → [{text, position}, ...]

# 3. Embed
for chunk in chunks:
    chunk.embedding = embed(chunk.text)  # Gemini embedding

# 4. Extract
entities, edges = extract(chunks, ontology)  # Gemini Flash structured output

# 5. Review
# Alan approves/edits in YAML or via chat

# 6. Load
load_to_neo4j(source, chunks, entities, edges)  # Cypher MERGE
save_to_yaml(entities, edges)  # Git canonical files
```

## Next Steps
- [ ] Define ontology v1 (node types, edge types, properties)
- [ ] Set up Neo4j Docker on NAS
- [ ] Build fetch + chunk + embed pipeline
- [ ] Build extract pipeline (Gemini Flash structured output)
- [ ] Build Neo4j loader (Cypher MERGE)
- [ ] Test with 5 sources end-to-end
- [ ] Build vector search + graph traversal queries

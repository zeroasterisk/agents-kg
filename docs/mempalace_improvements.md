# Improvements Informed by MemPalace Project

After reviewing the `MemPalace` project, here are several key improvements we can make to our Knowledge Graph extraction and retrieval pipeline.

While `MemPalace` uses SQLite for its graph and ChromaDB for verbatim storage, our project uses Neo4j for the graph and SQLite for stage management. We can adopt several of their powerful patterns to enhance our system.

## 1. Temporal Validity for Facts (Edges)
*   **Concept**: MemPalace triples have `valid_from` and `valid_to` properties, allowing the system to know *when* a fact was true.
*   **Application**: We should add `valid_from` and `valid_to` properties to relationships in Neo4j.
*   **Benefit**: This enables time-aware queries (e.g., "What was true about project X in January?").
*   **Implementation**:
    *   Update the extraction prompt to ask the LLM to identify temporal markers for facts.
    *   Update `db.py` schema for `edges` to store these dates.
    *   Update `load.py` to pass them to Neo4j.

## 2. Verbatim Chunk Linking in Graph
*   **Concept**: MemPalace scores highly on benchmarks by keeping raw verbatim text findable, rather than relying solely on summaries or extractions.
*   **Application**: We should load the text chunks into Neo4j as `:Chunk` nodes and link entities to them (e.g., `(Entity)-[:EXTRACTED_FROM]->(Chunk)`).
*   **Benefit**: This enables powerful hybrid retrieval. An agent can find an entity in the graph and then immediately retrieve the raw source text it was extracted from, preserving full context.
*   **Implementation**:
    *   Update `load.py` to create `:Chunk` nodes and create relationships from entities/edges to their source chunks.

## 3. Source Traceability
*   **Concept**: MemPalace stores `source_file` in its triples.
*   **Application**: Add `source_id` and `source_url` as properties to Neo4j nodes and relationships.
*   **Benefit**: Auditing and verification. Users can trace any node or edge back to the specific source document it came from.
*   **Implementation**:
    *   We already have `source_id` in SQLite tables; we just need to include it in the Cypher queries in `load.py`.

## 4. Palace Structure (Wings/Rooms) as Graph Taxonomy
*   **Concept**: MemPalace organizes data into "Wings" (e.g., Person or Project) and "Rooms" (specific topics like `auth` or `billing`), which yielded a 34% improvement in retrieval precision.
*   **Application**: We can use Neo4j labels or properties to categorize nodes into similar domains.
*   **Benefit**: Better scoping for queries. An agent can search specifically within the "MCP Project" domain rather than scanning the whole graph.

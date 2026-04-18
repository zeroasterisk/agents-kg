# Ontology v1 — Agentic Web Interoperability Knowledge Graph

> **Status:** Baseline
> **Date:** 2026-04-18
> **Author:** Zaf

---

## 1. Implementation Status

The ontology is implemented via a SQLite-backed pipeline with the following stages:
- **Fetch:** Supports HTTP(S) and local files (including PDF text extraction).
- **Parse:** Markdown passthrough and HTML-to-text conversion.
- **Chunk:** Section-based chunking with token estimation.
- **Embed:** Using `models/gemini-embedding-001`.
- **Extract:** Using `models/gemini-2.5-flash` with structured output.
- **Resolve:** Seed-based entity resolution and fuzzy matching.
- **Load:** YAML export and Neo4j graph population.

## 2. Node Types

| Type | Key Properties | Description |
|------|---------------|-------------|
| **Organization** | `id`, `name`, `type` (standards_body \| consortium \| foundation), `url`, `founded`, `status` | Standards bodies and industry groups: AAIF, AGNTCY, IETF, W3C, LF AI & Data |
| **Company** | `id`, `name`, `url`, `sector`, `status` | Companies participating in the ecosystem: Google, Microsoft, Anthropic, Cisco, Salesforce, IBM |
| **Person** | `id`, `name`, `email`, `github`, `status` | Named individuals (chairs, contributors, authors). Repo is private. |
| **Committee** | `id`, `name`, `type` (tsc \| wg \| sig \| task_force), `charter_url`, `status` | Sub-structures of organizations: TSCs, Working Groups, SIGs |
| **Project** | `id`, `name`, `repo_url`, `docs_url`, `license`, `status` | Software projects and frameworks: A2A, MCP, ACP, ANP, ADK, LangGraph, CrewAI, AutoGen |
| **Protocol** | `id`, `name`, `version`, `spec_url`, `rfc`, `status` | The specification/standard itself, versioned independently of implementing projects |
| **Capability** | `id`, `name`, `description`, `status` | Concepts and capabilities: identity, discovery, tool use, streaming, orchestration, delegation |
| **Source** | `id`, `url`, `title`, `type` (webpage \| paper \| spec \| repo \| announcement), `published_at`, `status` | A retrievable document or URL used as evidence |
| **Chunk** | `id`, `source_id`, `text`, `offset_start`, `offset_end`, `hash`, `extracted_at` | A text segment extracted from a Source; the atomic unit of provenance |

### Common Node Properties

Every node carries these base properties:

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | Unique identifier (UUID or namespaced slug) |
| `name` | STRING | Human-readable label |
| `status` | ENUM | `active` \| `deprecated` \| `merged` \| `ambiguous` |
| `merged_into` | STRING? | Target entity ID if status = merged |
| `deprecation_reason` | STRING? | Why this entity was deprecated or merged |
| `created_at` | DATETIME | When the node was created in the KG |
| `updated_at` | DATETIME | Last modification timestamp |
| `aliases` | STRING[] | Alternative names for disambiguation |

---

## 2. Edge Types

| Edge | From → To | Description |
|------|-----------|-------------|
| **MEMBER_OF** | Person → Company | Employment or affiliation |
| **MEMBER_OF** | Person → Committee | Participation in a committee |
| **MEMBER_OF** | Company → Organization | Organizational membership |
| **CHAIRS** | Person → Committee | Leadership role |
| **GOVERNS** | Organization → Committee | Org owns/charters a committee |
| **GOVERNS** | Committee → Project | Committee has governance over a project |
| **DEVELOPS** | Company → Project | Company actively develops/maintains a project |
| **CONTRIBUTES_TO** | Person → Project | Individual contribution |
| **IMPLEMENTS** | Project → Protocol | Project implements a spec/standard |
| **DEFINES** | Committee → Protocol | Committee authors/owns a protocol |
| **COMPETES_WITH** | Project ↔ Project | Competing or overlapping projects (bidirectional) |
| **COMPLEMENTS** | Project ↔ Project | Projects designed to work together |
| **ADDRESSES** | Project → Capability | Project solves or provides a capability |
| **REQUIRES** | Protocol → Capability | Protocol depends on a concept |
| **SPONSORS** | Company → Organization | Financial or resource sponsorship |
| **AUTHORED** | Person → Source | Person wrote or published a source |
| **FROM_SOURCE** | Chunk → Source | Chunk was extracted from this source |
| **EXTRACTED_FROM** | *any edge* → Chunk | Provenance: this relationship was asserted by this chunk |

### Common Edge Properties

Every edge carries these temporal and provenance properties:

| Property | Type | Description |
|----------|------|-------------|
| `valid_from` | DATETIME | When this relationship became true |
| `valid_to` | DATETIME? | When it ceased to be true (`null` = still current) |
| `confidence` | FLOAT | 0.0–1.0, how confident we are in this assertion |
| `chunk_id` | STRING? | The Chunk that asserted this relationship |
| `extracted_at` | DATETIME | When this edge was added to the KG |
| `source_type` | ENUM | `manual` \| `automated` \| `inferred` |

---

## 3. Temporal Model

The KG uses a **bitemporal** approach:

1. **Validity time** (`valid_from` / `valid_to`) — when the fact is true in the real world.
2. **Transaction time** (`extracted_at`) — when we recorded it.

### Rules

- **Current facts:** `valid_to IS NULL` means the relationship is still active.
- **Historical facts:** Both `valid_from` and `valid_to` are set. The edge is never deleted, only closed.
- **Corrections:** If a fact was recorded incorrectly, close the old edge (`valid_to = now`) and create a new edge with corrected data. Both edges retain their `extracted_at` for auditability.
- **Point-in-time queries:** "What was true as of 2025-06-01?" → filter where `valid_from <= date AND (valid_to IS NULL OR valid_to > date)`.

### Example

Google's relationship with A2A:

```
(Google)-[DEVELOPS {valid_from: 2024-04-01, valid_to: null, confidence: 1.0}]->(A2A)
```

If governance transfers:

```
(Google)-[GOVERNS {valid_from: 2024-04-01, valid_to: 2025-03-01}]->(A2A)
(AAIF_TSC)-[GOVERNS {valid_from: 2025-03-01, valid_to: null}]->(A2A)
```

---

## 4. Entity Lifecycle & Deprecation

### Status Values

| Status | Meaning |
|--------|---------|
| `active` | Current, valid entity |
| `deprecated` | No longer relevant; kept for history. Set `deprecation_reason`. |
| `merged` | Duplicate resolved. Set `merged_into` to the canonical entity ID. |
| `ambiguous` | Entity needs human review for disambiguation. |

### Deprecation Flow

1. Set `status = deprecated` and `deprecation_reason`.
2. Close all active edges (`valid_to = now`).
3. Entity remains queryable for historical queries.

### Merge Flow

1. Identify canonical entity (the one to keep).
2. On the duplicate: set `status = merged`, `merged_into = canonical_id`.
3. Re-point all edges from the duplicate to the canonical entity (creating new edges with fresh `extracted_at`, preserving original `valid_from`).
4. Close original edges on the duplicate.

### Supersession

When a Protocol or Project is replaced by a newer version:

```
(A2A_v1)-[SUPERSEDED_BY {valid_from: 2025-09-01}]->(A2A_v2)
```

The old entity gets `status = deprecated`, `deprecation_reason = "Superseded by A2A v2"`.

---

## 5. Disambiguation Strategy

### Problem

The same name can refer to different things (e.g., "A2A" the protocol vs "A2A" the project), or the same entity appears under multiple names.

### Approach

1. **Aliases:** Every node has an `aliases[]` array. Search and ingestion match against all aliases.
2. **Namespaced IDs:** IDs use the pattern `{type}:{slug}` — e.g., `project:a2a`, `protocol:a2a-spec`, `org:aaif`.
3. **Ambiguity flag:** Ingestion can create a node with `status = ambiguous` when it can't resolve. These surface in a review queue.
4. **Merge workflow:** When duplicates are found, use the merge flow (§4) to consolidate.
5. **Canonical name:** The `name` field is the canonical/preferred form. `aliases` holds variants (acronyms, former names, common misspellings).

---

## 6. Seed Data Examples

### Organizations

| ID | Name | Type |
|----|------|------|
| `org:aaif` | Agent Interoperability Forum (AAIF) | standards_body |
| `org:agntcy` | AGNTCY | consortium |
| `org:ietf` | Internet Engineering Task Force | standards_body |
| `org:w3c` | World Wide Web Consortium | standards_body |
| `org:lf-ai` | Linux Foundation AI & Data | foundation |

### Companies

| ID | Name |
|----|------|
| `company:google` | Google |
| `company:anthropic` | Anthropic |
| `company:microsoft` | Microsoft |
| `company:cisco` | Cisco |
| `company:salesforce` | Salesforce |
| `company:ibm` | IBM |

### Committees

| ID | Name | Type |
|----|------|------|
| `committee:aaif-tsc` | AAIF Technical Steering Committee | tsc |
| `committee:aaif-discovery-wg` | AAIF Discovery Working Group | wg |

### Projects

| ID | Name |
|----|------|
| `project:a2a` | Agent-to-Agent Protocol (A2A) |
| `project:mcp` | Model Context Protocol (MCP) |
| `project:acp` | Agent Communication Protocol (ACP) |
| `project:anp` | Agent Network Protocol (ANP) |
| `project:adk` | Agent Development Kit (ADK) |
| `project:langgraph` | LangGraph |
| `project:crewai` | CrewAI |
| `project:autogen` | AutoGen |

### Capabilities

| ID | Name |
|----|------|
| `cap:discovery` | Agent Discovery |
| `cap:identity` | Agent Identity |
| `cap:tool-use` | Tool Use |
| `cap:streaming` | Streaming |
| `cap:orchestration` | Multi-Agent Orchestration |
| `cap:delegation` | Task Delegation |

### Example Edges

```
# Google develops A2A
(company:google)-[DEVELOPS {valid_from: 2024-04-01, confidence: 1.0}]->(project:a2a)

# AAIF TSC governs A2A
(committee:aaif-tsc)-[GOVERNS {valid_from: 2025-03-01, confidence: 1.0}]->(project:a2a)

# Anthropic develops MCP
(company:anthropic)-[DEVELOPS {valid_from: 2024-11-01, confidence: 1.0}]->(project:mcp)

# A2A addresses discovery
(project:a2a)-[ADDRESSES {valid_from: 2024-04-01, confidence: 0.9}]->(cap:discovery)

# A2A and MCP are complementary
(project:a2a)-[COMPLEMENTS {valid_from: 2025-01-01, confidence: 0.8}]->(project:mcp)

# Google sponsors AAIF
(company:google)-[SPONSORS {valid_from: 2025-01-01, confidence: 1.0}]->(org:aaif)

# AAIF governs the TSC
(org:aaif)-[GOVERNS {valid_from: 2025-01-01, confidence: 1.0}]->(committee:aaif-tsc)

# Google is a member of AAIF
(company:google)-[MEMBER_OF {valid_from: 2025-01-01, confidence: 1.0}]->(org:aaif)
```

---

## 7. Open Questions

- **Graph database choice:** Neo4j, Memgraph, or property graph over PostgreSQL (Apache AGE)?
- **Chunk granularity:** How large should chunks be? Paragraph-level? Sentence-level?
- **Confidence thresholds:** What's the minimum confidence for an edge to be considered "reliable"?
- **Automated ingestion:** Which sources to crawl first? (GitHub repos, AAIF docs, IETF drafts)
- **Access control:** Any edges/nodes that need finer-grained permissions beyond repo-level privacy?

---

*This document is a starting point. Review, poke holes, and iterate.*

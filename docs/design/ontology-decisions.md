# Ontology Decisions — Confirmed

## Node Types (7 confirmed)

| Type | `kind` values | Description |
|------|--------------|-------------|
| **Organization** | company, standards_body, foundation, consortium | Unified type for all orgs. Google, AAIF, LF AI are all Organization nodes with different `kind`. |
| **Group** | tsc, wg, sig, task_force, team, division | Internal structures. Recursive via `PART_OF` edges. A WG can be inside a TSC inside an Org. |
| **Person** | — | Named individuals. Repo is private. |
| **Project** | framework, sdk, library, tool, platform | Software/code. ADK, LangGraph, CrewAI, AutoGen. |
| **Protocol** | spec, standard, rfc, draft | The specification document, versioned. A2A spec, MCP spec, ACP spec. |
| **Capability** | — | Concepts and capabilities: identity, discovery, tool use, streaming, orchestration. |
| **Source** | webpage, paper, spec, repo, announcement, conversation | A retrievable document used as evidence. |
| **Chunk** | — | Text segment from a Source. Atomic unit of provenance. |

### Design principles applied
- **Merge when boundary is blurry** → Organization absorbs Company (Anthropic is both)
- **Separate when boundary is meaningful** → Protocol stays separate from Project (enables "which projects implement this spec?")
- **Recursive over rigid** → Group with `PART_OF` edges, not fixed hierarchy
- **`kind` property over multiple node types** → portable, works in YAML/BQ/Neo4j
- **Ontology grows and prunes** → start tight, add types only when `kind` isn't enough

### Common properties on all nodes
- `id`: STRING (namespaced slug, e.g. `org:google`, `project:a2a`, `protocol:a2a-spec`)
- `name`: STRING
- `kind`: STRING (type-specific, see table above)
- `description`: STRING
- `aliases`: STRING[] (for disambiguation)
- `status`: active | deprecated | merged | ambiguous
- `merged_into`: STRING (if merged)
- `deprecation_reason`: STRING
- `url`: STRING (canonical URL)
- `created_at`, `updated_at`: DATETIME

## Edge Properties (all edges)

Every relationship carries:
- `valid_from`: DATETIME — when the fact became true
- `valid_to`: DATETIME (null = still current)
- `confidence`: FLOAT (0.0–1.0)
- `chunk_id`: STRING — provenance link to the Chunk that asserted this
- `extracted_at`: DATETIME — when we recorded it
- `source_type`: manual | automated | inferred

## Temporal Model
- Bitemporal: validity time + transaction time
- Never delete edges — close them (`valid_to = now`)
- Point-in-time queries: `valid_from <= date AND (valid_to IS NULL OR valid_to > date)`

## Entity Lifecycle
- `active` → `deprecated` (with reason) or `merged` (with target)
- `ambiguous` for entities needing human review
- Merge flow: re-point edges to canonical entity, close originals

## Disambiguation
- `aliases[]` on every node
- Namespaced IDs (`org:google`, `project:a2a`)
- `ambiguous` status → surfaces in review queue
- Merge workflow when duplicates confirmed

## Infrastructure
- **Neo4j** (Docker) for graph + vectors + source/chunk storage
- **SQLite** for pipeline job queue
- **Python CLI** for ingestion (`kg ingest`, `kg status`, `kg review`)
- **Gemini Flash** for extraction + embedding
- **YAML files in git** for canonical backup / portability

# KG Onboarding Plan for Alan

> **Date:** 2026-05-12
> **Author:** PM Agent
> **Status:** Updated with Alan's clarifications

---

## 1. Pre-Ingestion Checklist

### 1a. Document Submission Model

Alan submits **source documents** (URLs or file paths). The pipeline handles all entity and edge extraction automatically. Alan never needs to manually specify node types, edge types, or graph structure.

Supported document formats, ordered by preference:

| Format | Why | Notes |
|--------|-----|-------|
| **Markdown (.md)** | Native to the existing pipeline (parse stage handles it directly) | Best for structured docs with clear headings |
| **PDF** | Supported via pymupdf in the fetch stage | Good for specs, whitepapers, published reports |
| **HTML / web URL** | Supported via httpx + readability-lxml | Link to the canonical URL; pipeline will fetch and parse |

**Recommendation:** Submit URLs when a canonical web version exists. For internal or unpublished docs, submit markdown or PDF files. The pipeline will extract all entities and relationships automatically.

### 1b. Authoritative vs. Theoretical Tagging

The existing schema supports this distinction through two mechanisms:

| Document Type | How to Tag | Where It Appears |
|---------------|-----------|-----------------|
| **Authoritative** (ground-truth facts) | source_type: authoritative on the Source node; extracted entities get confidence: 1.0 on edges | Entities auto-approved in review step; edges treated as high-confidence |
| **Theoretical/speculative** | source_type: theoretical on the Source node; extracted entities get confidence: 0.3-0.7 range | Entities routed through 3-tier review system (see Section 1e) |

Alan should indicate at submission time whether a doc is authoritative or theoretical. A simple convention:
- Place authoritative docs in a sources/authoritative/ directory
- Place theoretical docs in a sources/theoretical/ directory
- The pipeline reads the directory to set source_type automatically

### 1c. Required Metadata Per Document

Each submitted document needs:

| Field | Required? | Example |
|-------|-----------|---------|
| **Title** | Yes | "Agent Identity Framework Design Doc" |
| **URL or file path** | Yes | https://docs.google.com/... or sources/authoritative/agent-identity.md |
| **Date** (publication or last update) | Yes | 2026-05-10 |
| **Author** | Recommended | "Alan Blount" |
| **Source type** | Yes | authoritative or theoretical |
| **Domain tags** | Recommended | identity, security, agent-protocols |
| **Brief description** | Recommended | 1-2 sentences on what this doc covers |

This metadata is submitted via the Google Sheet (see Section 2b). The pipeline handles all entity extraction, edge creation, and ontology mapping from the document content.

### 1d. Ontology Reference

The pipeline extracts entities and edges that map to the existing ontology. Alan does not need to specify these -- the pipeline determines them automatically from document content.

**Entity types the pipeline extracts:**
- Organization (kinds: company, standards_body, foundation, consortium)
- Group (kinds: tsc, wg, sig, task_force, team)
- Person
- Project (kinds: framework, sdk, library, tool, platform, programming_language)
- Protocol (kinds: spec, standard, rfc, draft)
- Capability (abstract abilities: tool-use, reasoning, planning, security)

**Edge types the pipeline extracts (15 validated types):**
MEMBER_OF, GOVERNS, DEVELOPS, IMPLEMENTS, COMPETES_WITH, ADDRESSES, AUTHORED, CHAIRS, SPONSORS, PART_OF, SUPERSEDES, CONTRIBUTES_TO, DEFINES, COMPLEMENTS, USES

**If documents introduce concepts outside this ontology:**
- New entity kind values can be added freely (the kind field is a string, not an enum)
- New entity type values require a seed.py update and extraction prompt change -- flag these for the engineering team
- New edge types require an extraction prompt update -- flag these too

### 1e. Theoretical Document Handling

For theoretical/speculative documents, the pipeline uses a multi-strategy approach to find real-world analogues:

1. **Extract entities** as normal through the pipeline
2. **Tag extracted entities** with status: theoretical (new status value alongside active/deprecated/merged)
3. **Run all similarity approaches** to find matches against the 800+ entities in the graph:
   - **Vector similarity** -- embedding-based search via the resolve stage
   - **Graph traversal** -- follow edges to find structurally related entities
   - **Tag matching** -- match on domain tags and capability overlaps
4. **Trigger agentic research** if confidence is low at ingestion time -- an LLM agent investigates ambiguous entities, searches for additional context, and proposes matches before human review
5. The resolver surfaces "similar entities" with similarity scores -- these are the real-world analogues Alan is looking for

### 1f. Review System (3-Tier)

All extracted entities and edges pass through a 3-tier review system:

| Tier | Trigger | Action |
|------|---------|--------|
| **Tier 1: Auto-approve** | High-confidence extractions from authoritative sources (confidence >= 0.9) | Load directly into Neo4j, no manual review |
| **Tier 2: Agentic research** | Ambiguous or conflicting entities (e.g., name collisions, uncertain mappings) | LLM agent researches the entity, gathers context, proposes resolution |
| **Tier 3: Human review** | Entities that remain unresolved after agentic research, or low-confidence theoretical extractions | Queued for human review via the KG CLI review tools |

This 3-tier system is already part of the existing pipeline design.

---

## 2. UI/UX Recommendation

### 2a. Evaluation of Options

| Option | Effort | Value for Alan | Scalability | Verdict |
|--------|--------|---------------|-------------|---------|
| **Pub/Sub** (event-driven ingestion) | M | Low -- Alan still needs a UI to publish | High for automation | Not a UI; plumbing. Useful later for automated source watching. |
| **Spreadsheet** (Google Sheets) | S | High -- familiar, immediate for document submission | Medium (scales to thousands of docs) | Good v0 for document submission and status tracking. |
| **Chat interface** | M | High -- natural language, low barrier | Medium (requires prompt engineering, LLM costs) | Good v1. Alan describes docs in plain language, pipeline extracts. |
| **Custom webapp** | L-XL | Highest -- purpose-built for the KG | High | Good v2+. Premature now -- build after interaction patterns proven. |

### 2b. Recommended Starting Point: Document Submission Sheet + Pipeline (v0)

**Why:** Alan submits source documents, not entities. A Google Sheet serves as a document submission interface where Alan adds URLs or file paths. The pipeline polls the sheet, fetches documents, and handles all extraction automatically.

**v0 Workflow:**

Alan adds doc URL to Sheet --> Pipeline polls sheet --> Fetch --> Parse --> Extract --> 3-tier Review --> Neo4j

**Sheet schema (single Documents tab -- see sheet-design.md for full spec):**

| uri | title | source_type | submitter | status | entity_count | edge_count | notes |
|-----|-------|------------|-----------|--------|-------------|------------|-------|
| https://spec.modelcontextprotocol.io/... | MCP Specification | authoritative | alan@example.com | submitted | | | Core MCP spec |
| https://arxiv.org/abs/2402.05120 | More Agents Is All You Need | speculative | alan@example.com | submitted | | | Agent scaling research |

The sheet uses the schema defined in sheet-design.md, with status lifecycle tracking (submitted --> queued --> fetching --> parsing --> extracting --> reviewing --> ingested) and automatic pipeline writeback for entity/edge counts and status updates.

**Hosting:** All pipeline services, Neo4j, and UI components run on a shared VM (already decided based on Neo4j instance size).

### 2c. v1: Chat Interface + Pipeline

**What it looks like:** A simple chat UI (Slack bot, CLI prompt, or lightweight web form) where Alan types natural language:

> "AGNTCY has a sub-project called AGNTCY Identity that defines the Agent Identity Badge spec. It is similar to DID (Decentralized Identifiers) from W3C."

The system:
1. Sends this through the existing LLM extraction pipeline (chunk --> extract --> resolve)
2. Runs through the 3-tier review system
3. On approval, loads to Neo4j

**Why v1 not v0:** Requires a thin API layer (FastAPI endpoint wrapping the pipeline), a simple frontend, and prompt tuning for conversational input. Estimated 2-3 weeks of engineering. Deployed on the shared VM.

### 2d. v2+: Custom Webapp (Later)

Purpose-built UI with:
- Graph visualization (Neo4j Bloom or custom D3.js)
- Inline entity editing
- Relationship drawing (click node A, drag to node B, select edge type)
- Temporal timeline view
- Query builder for non-Cypher users

This is a 1-2 month project. Deployed on the shared VM. Should only start after v0 and v1 have validated the interaction patterns.

---

## 3. Burn Down List

### Immediate (This Week)

| # | Task | Description | Effort | Who |
|---|------|-------------|--------|-----|
| 1 | **Set up document submission sheet** | Create Google Sheet with the document submission schema from sheet-design.md. Add data validation for source_type and status. Include the CSV template rows as starter examples. | S | Agent |
| 2 | **Build sheet poller service** | Module that polls the Google Sheet (or watches CSV export), detects new submitted rows, validates URIs, checks for duplicates, and queues documents into pipeline.db. Updates sheet with status writeback. | M | Agent |
| 3 | **Add source_type field to pipeline** | Update the fetch/extract stages to propagate source_type (authoritative/theoretical) from source metadata to extracted entities and edges. | S | Agent |
| 4 | **Add status: theoretical support** | Extend the entity lifecycle states (currently: active/deprecated/merged/ambiguous) to include theoretical. Update seed.py and the extraction prompt. | S | Agent |
| 5 | **Document the submission workflow** | Write a short how-to for Alan: how to add docs to the sheet, what metadata to provide, and how to monitor ingestion progress. | S | Agent |
| 6 | **Alan: submit first batch of authoritative docs** | Alan adds 3-5 authoritative document URLs or file paths to the submission sheet. | S | Alan |

### Short-Term (2-4 Weeks)

| # | Task | Description | Effort | Who |
|---|------|-------------|--------|-----|
| 7 | **Build multi-strategy similarity search** | After theoretical entities are extracted, run all three approaches: vector similarity, graph traversal, and tag matching against the full entity corpus. Return top-5 matches with similarity scores. Trigger agentic research when confidence is low. | L | Agent |
| 8 | **Chat ingestion endpoint** | FastAPI endpoint that accepts natural-language text, runs it through the extraction pipeline, and returns structured entities/edges for review. Deployed on shared VM. | M | Agent |
| 9 | **Simple chat UI** | Minimal web form or CLI chat mode that posts to the chat endpoint and displays extracted results for approval. Hosted on shared VM. | M | Agent |
| 10 | **Implement 3-tier review system** | Wire up the auto-approve / agentic research / human review pipeline: (1) auto-approve high-confidence authoritative extractions, (2) trigger agentic research for ambiguous entities, (3) queue remainder for human review. | M | Agent |
| 11 | **Alan: submit theoretical/speculative docs** | Alan submits theoretical document URLs. Pipeline extracts entities, multi-strategy similarity search surfaces analogues, agentic research resolves ambiguities. Alan reviews via human review queue. | M | Alan |
| 12 | **Multi-user query access** | Set up access for additional team members to query the KG. Pre-built Cypher query templates, Neo4j Browser access on the shared VM, and basic role separation (submitters vs. reviewers). | M | Agent |

### Medium-Term (1-2 Months)

| # | Task | Description | Effort | Who |
|---|------|-------------|--------|-----|
| 13 | **Graph visualization UI** | Web-based graph explorer using Neo4j Bloom or a custom D3.js/vis.js frontend on the shared VM. Shows entity neighborhoods, edge types, temporal sliders. | L | Agent |
| 14 | **Inline entity editor** | Click an entity in the graph UI to edit its properties, add edges, or flag for review. Changes write back to YAML and Neo4j. | L | Agent |
| 15 | **Automated source watching** | Pub/sub or cron-based system that monitors a list of URLs for changes, re-runs the pipeline on updated sources, and flags new/changed entities for review. | L | Agent |
| 16 | **Confidence decay model** | Entities extracted from theoretical docs that are confirmed by authoritative sources get promoted to status: active with higher confidence. Unconfirmed theoretical entities decay in confidence over time. | M | Agent |
| 17 | **Multi-user access controls** | Per-user permissions: who can submit, who can approve, who can edit directly. Required as more team members join. | M | Agent |

---

## 4. Open Questions for Alan

1. **Document inventory:** How many authoritative docs do you have ready now? And how many theoretical/speculative docs? Rough count helps us estimate pipeline load and review time.

2. **Access preferences:** Do you want to interact with the KG through Neo4j Browser (Cypher queries), or do you need a non-technical UI from day one? This determines whether we prioritize query templates or the chat UI.

---

## 5. Resolved Questions

The following questions from the original draft have been resolved based on Alan's input:

- **Entity/edge specification:** RESOLVED -- Alan submits source documents (URLs or file paths). The pipeline extracts all entities and edges automatically. Alan does not manually specify node types, edge types, or graph structure.

- **Theoretical-to-real mapping approach:** RESOLVED -- Use all available approaches: vector similarity, graph traversal, and tag matching. Additionally, trigger agentic research when confidence is low at ingestion time.

- **Collaboration scope:** RESOLVED -- Multiple team members will query the KG within the next few weeks. Multi-user access is prioritized in the short-term burn-down.

- **Review tolerance:** RESOLVED -- 3-tier review system: (1) auto-approve high-confidence extractions, (2) agentic research for ambiguous or conflicting entities, (3) human review queue as final fallback. This design is already part of the existing pipeline.

- **Hosting:** RESOLVED -- Shared VM, already decided due to Neo4j instance size. All UI/UX tasks target VM-hosted deployment.

---

## Appendix: Current Graph State

For reference, the KG currently contains:

| Entity Type | Count (YAML files) |
|------------|-------------------|
| Protocols | 169 |
| Organizations | 88 |
| Projects | 203 |
| Capabilities | 325 |
| Groups | 26 |
| Persons | 19 |
| Concepts | 2 |
| **Total** | **832** |

Plus ~8,000+ Wikidata-sourced entities (programming languages, protocols, standards, software companies, open-source projects) loaded via the SPARQL ingestion module.

The pipeline has processed 27 sources so far, with 15 validated edge types and a 4-pass entity resolution system (exact, fuzzy, vector, noise filter).

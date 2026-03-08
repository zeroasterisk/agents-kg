# DeepLearning.AI: Agentic Knowledge Graph Construction

**Course:** [Agentic Knowledge Graph Construction](https://learn.deeplearning.ai/courses/agentic-knowledge-graph-construction/)
**Instructor:** Andreas Kollegger (Developer Evangelist for Generative AI, Neo4j)
**Stack:** Google ADK (Python) + Neo4j + Gemini
**Published:** Aug 2025

## Architecture: Multi-Agent KG Construction

The course builds a hierarchical multi-agent system using ADK:

```
Knowledge Graph Agent (top-level, conversational)
├── User Intent Agent (conversational) → approved_user_goal
├── Structured Data Agent (workflow)
│   ├── File Suggestion Agent (tool-use) → approved_files
│   ├── Schema Proposal Agent (critic pattern, iterative) → approved_schema
│   └── Graph Construction Plan → domain graph via Cypher
├── Unstructured Data Agent (workflow)
│   ├── Entity & Fact Type Proposal Agent (tool-use) → entity/fact types
│   └── Knowledge Extraction Plan → chunks → entities → facts
└── GraphRAG Agent (tool-use) → retrieval strategy for Q&A
```

### Agent Types Used
- **Conversational**: Back-and-forth with user (intent, clarification)
- **Workflow**: Orchestrates sub-agents sequentially
- **Tool-use**: Calls tools, analyzes results, decides next action
- **Critic pattern**: Pair of agents that iteratively refine output (schema proposal)

## Key Patterns Worth Adopting

### 1. Human-in-the-loop at checkpoints
Every major decision has a `set_perceived_X` → present to user → `approve_X` pattern:
- `set_perceived_user_goal` → user confirms → `approve_perceived_user_goal`
- `set_suggested_files` → user confirms → `approve_suggested_files`
- Schema proposal → user reviews → approve

This is critical for us — Alan wants to guide intake, not have autonomous extraction.

### 2. Structured state via tool calls
Instead of free-form LLM prose, decisions are captured as structured data via tool calls:
```python
def set_perceived_user_goal(kind_of_graph: str, graph_description: str, tool_context: ToolContext):
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    tool_context.state[PERCEIVED_USER_GOAL] = user_goal_data
```
State keys: `perceived_user_goal`, `approved_user_goal`, `suggested_files`, `approved_files`, etc.

### 3. Critic pattern for schema refinement
Two agents iterate: one proposes a schema, the other critiques it. This produces better ontologies than single-shot extraction.

### 4. Separate structured vs unstructured pipelines
- **Structured (CSV)**: File suggestion → schema proposal → construction plan → Cypher import
- **Unstructured (Markdown)**: Entity/fact type proposal → chunking → extraction → entity linking

### 5. Neo4j import directory pattern
CSV files placed in Neo4j's `import/` directory, then loaded via Cypher `LOAD CSV`. The agent samples files to understand structure before proposing schema.

### 6. Domain graph + Knowledge graph separation
- **Domain graph**: Structured data import (products, suppliers, parts) — deterministic, rule-based
- **Knowledge graph**: Entities and facts extracted from unstructured text — LLM-powered
- Connected via entity linking (extracted entities matched to domain entities)

## Relevance to agents-kg

### Direct applicability
- We have both structured (YAML entity files) and unstructured (blog posts, specs, conversations) data
- The human-in-the-loop pattern matches Alan's "guided by me for now" requirement
- ADK is our stack — we can reuse these patterns directly
- Neo4j is our target backend

### What we'd adapt
- **User Intent Agent** → Not needed (we know our goal: agentic web ecosystem KG)
- **File Suggestion Agent** → Becomes "Source Suggestion Agent" — given a URL/doc, suggest what entities to extract
- **Schema Proposal Agent** → We pre-define our ontology, but could use this for schema evolution
- **Graph Construction Plan** → Our ETL pipeline
- **Entity & Fact Type Proposal** → Core of our extraction — given a blog post, what entities and relationships exist?
- **GraphRAG Agent** → Our query interface (MCP server wrapping Neo4j)

### What to build
1. **Extraction Agent**: Given a URL/document + our ontology, extract entities and relationships as structured records
2. **Review Agent** (or just Alan): Confirm/edit extracted entities before import
3. **Import Tool**: Cypher MERGE into Neo4j from approved records
4. **Query Agent**: GraphRAG-style retrieval for comparison queries

## Code References
- [Course notebooks](https://learn.deeplearning.ai/courses/agentic-knowledge-graph-construction/) (requires enrollment, free)
- [Student walkthrough with code](https://shilpathota.medium.com/agentic-knowledge-graph-construction-with-neo4j-aadda43b71d9)
- Helper: `neo4j_for_adk.py` (course utility, wraps Neo4j driver for ADK tool context)

## Next Steps
- [ ] Enroll and walk through the notebooks hands-on
- [ ] Extract the `neo4j_for_adk.py` helper patterns for our own tooling
- [ ] Design our extraction agent using ADK + our ontology
- [ ] Prototype the critic pattern for ontology evolution

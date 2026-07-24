# Embedding Model Migration Design

**Date:** 2026-05-09
**Author:** Scion Agent (on behalf of Alan Blount)
**Status:** Superseded — gemini-embedding-2 (GA, Apr 2026) chosen instead of gemini-embedding-001. See commit on feature/rest-api.
**Branch:** feature/wikidata-integration

---

## 1. Current State

### 1.1 Model

The repo currently uses **`gemini-embedding-2-preview`**, a preview-tier Gemini embedding model from Google.

| Property | Value |
|----------|-------|
| Model ID | `gemini-embedding-2-preview` |
| SDK | `google-genai>=1.0` (Python) |
| Auth | Vertex AI (service account) or Gemini API key |
| Default region | `us-central1` |

The model ID is hardcoded as a module-level constant in two files:

- `src/agents_kg/stages/embed.py:21` — `EMBEDDING_MODEL = "gemini-embedding-2-preview"`
- `src/agents_kg/stages/resolve.py:24` — `EMBEDDING_MODEL = "gemini-embedding-2-preview"`

### 1.2 How Embeddings Are Generated

Two stages produce embeddings:

1. **Stage 4 (`embed.py`)** — embeds text chunks in batches of 100 via `client.models.embed_content()`. Each chunk's embedding is stored alongside its `embedding_model` and `embedded_at` timestamp.

2. **Stage 5b (`resolve.py`)** — embeds entity descriptions/names for vector-based deduplication. Entities with cosine similarity > 0.92 (same type) are merged.

Both stages duplicate the same client initialization pattern (Vertex AI vs API key detection), model constant, and float-to-bytes conversion.

### 1.3 How Embeddings Are Stored

Embeddings are stored in **SQLite** as `BLOB` columns (binary-packed float32 arrays via `struct.pack`):

| Table | Columns | Notes |
|-------|---------|-------|
| `chunks` | `embedding BLOB`, `embedding_model TEXT`, `embedded_at TEXT` | Tracks which model produced each embedding |
| `entities` | `embedding BLOB` | No model tracking column |

Dimensions are **not hardcoded** — they're implicit from whatever the model returns and stored as variable-length BLOBs. This is good for migration.

### 1.4 How Embeddings Are Used

- **Entity deduplication** (resolve stage): in-memory cosine similarity between entity embedding vectors. Threshold: 0.92.
- Embeddings are **not transferred to Neo4j** — the graph has no vector indexes. Neo4j uses only property indexes (name, type, kind, wikidata_id).
- There is no runtime vector search endpoint; embeddings are purely for offline entity resolution.

### 1.5 Abstraction Layer

**None.** The Google GenAI SDK is called directly in both `embed.py` and `resolve.py` with duplicated initialization logic. There is no embedding service interface, provider abstraction, or centralized configuration.

---

## 2. Target Models: Gemini Embedding Family

Google offers two GA embedding models that supersede preview models. Both use the same `google-genai` SDK the repo already depends on.

### 2.1 Model Comparison

| Property | gemini-embedding-2-preview (current) | gemini-embedding-001 | gemini-embedding-2 |
|----------|--------------------------------------|----------------------|---------------------|
| Status | Preview (may be deprecated) | GA (production) | GA (production, April 2026) |
| Modality | Text only | Text only | **Multimodal** (text, image, video, audio, PDF) |
| Default dimensions | ~768 (varies) | 3072 | 3072 |
| Configurable dimensions | No | Yes (MRL, 128-3072) | Yes (MRL, 128-3072, auto-normalized) |
| Max input tokens | ~2048 | 8192 | 8192 |
| Task types | Generic | 9 via `task_type` param | Prompt-based instructions |
| Multilingual | Partial | 100+ languages | 100+ languages |
| Code understanding | Basic | Dedicated task type | Dedicated task type |

### 2.2 Recommendation: `gemini-embedding-001`

For this project's use case (text-only entity resolution and chunk embedding), **`gemini-embedding-001`** is the right choice:

1. **Text-only workload**: The repo embeds text chunks and entity descriptions. The multimodal capabilities of `gemini-embedding-2` add cost ($0.20/1M tokens vs $0.15/1M tokens) without benefit.
2. **`task_type` parameter**: The structured `task_type` enum (`SEMANTIC_SIMILARITY`, `RETRIEVAL_DOCUMENT`) is cleaner for programmatic use than `gemini-embedding-2`'s prompt-based instructions.
3. **Batch API available**: `gemini-embedding-001` supports batch pricing at $0.075/1M tokens (50% discount). Batch is not yet available for `gemini-embedding-2`.
4. **Production stability**: GA model with SLA, replacing a preview model.

If the project later needs to embed images/PDFs (e.g., ingesting slide decks or diagrams), `gemini-embedding-2` would become relevant. The migration design below supports switching between any Gemini embedding model.

### 2.3 Key Advantages Over Current Model

1. **Production stability**: GA model with SLA, replacing a preview model that could be deprecated at any time.

2. **Matryoshka Representation Learning (MRL)**: Generate high-dimensional embeddings (3072) and truncate to lower dimensions (e.g., 768, 256) while preserving quality. This lets us tune the storage/quality tradeoff without re-embedding. MTEB scores: 3072-dim = 68.4, 768-dim = 68.0, 256-dim = 66.2.

3. **Task-specific optimization**: Specifying `task_type` (e.g., `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLUSTERING`) improves quality for that use case. Entity resolution would benefit from `SEMANTIC_SIMILARITY`; future vector search would use `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT`.

4. **4x input context**: 8192 tokens vs ~2048 allows embedding larger chunks without truncation.

5. **Same SDK**: Uses the same `google-genai` package — migration is an API parameter change, not a dependency change.

### 2.4 API Usage

```python
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config={
        "task_type": "SEMANTIC_SIMILARITY",
        "output_dimensionality": 768,  # optional MRL truncation
    },
)
```

### 2.5 Pricing

| Model | Standard (per 1M tokens) | Batch (per 1M tokens) |
|-------|--------------------------|----------------------|
| gemini-embedding-001 | $0.15 | $0.075 |
| gemini-embedding-2 | $0.20 | Not yet available |
| Free tier (AI Studio) | Free with rate limits | N/A |

Vertex AI pricing is comparable (~$0.15/1M input tokens for gemini-embedding-001).

---

## 3. Migration Design

### 3.1 Coupling Assessment

Migration is **low-risk** because:

- Embedding dimensions are not hardcoded anywhere — BLOBs are variable-length.
- No Neo4j vector indexes need rebuilding.
- The `chunks` table already tracks `embedding_model` per row, so mixed-model data is structurally supported.
- The same SDK and API shape (`client.models.embed_content()`) are used.
- Embeddings are consumed only in the resolve stage (in-memory cosine similarity), which is dimension-agnostic as long as all vectors in a comparison have the same length.

**The main risk**: comparing embeddings from different models (different vector spaces) produces meaningless similarity scores. The resolve stage compares new-source entity embeddings against *all* approved entities — if those were embedded with the old model, cross-model comparisons will silently degrade entity resolution quality.

### 3.2 What Needs to Change

#### Phase 1: Centralize Configuration (Low effort, do first)

Create a single embedding configuration module to eliminate duplication:

```
src/agents_kg/embedding.py  (new)
```

This module should:
- Define `EMBEDDING_MODEL` once (from env var with default)
- Provide a `get_embedding_client()` function encapsulating Vertex AI / API key logic
- Provide `embed_texts(texts, task_type=None)` as the single entry point
- Expose `floats_to_bytes()` and `bytes_to_floats()` utilities

Update `embed.py` and `resolve.py` to import from this module instead of duplicating logic.

#### Phase 2: Switch to gemini-embedding-001 (Low effort)

1. Change the default model to `gemini-embedding-001`.
2. Add `EMBEDDING_TASK_TYPE` config for per-stage task type specification.
3. Add `EMBEDDING_DIMENSIONS` config (optional, for MRL truncation).

#### Phase 3: Re-embed Existing Data (Medium effort)

Add a CLI command to re-embed all data with the new model:

```
kg re-embed [--model gemini-embedding-001] [--batch-size 100] [--source-id N]
```

This command should:

1. **Clear existing embeddings** for the target scope (all chunks, or a specific source).
2. **Re-run the embed stage** for all chunks with cleared embeddings.
3. **Re-run entity embedding** for all entities with cleared embeddings.
4. **Track progress** — log batch progress, handle rate limits, support resume on failure.

Implementation:
- Add `Database.clear_embeddings(source_id=None)` method that NULLs out `embedding`, `embedding_model`, and `embedded_at` on chunks, and `embedding` on entities.
- Add `Database.clear_entity_embeddings(source_id=None)` for entities only.
- Wire into CLI as a click command.

#### Phase 4: Add Model Tracking to Entities (Low effort)

The `entities` table lacks `embedding_model` tracking (unlike `chunks`). Add:

```sql
ALTER TABLE entities ADD COLUMN embedding_model TEXT;
ALTER TABLE entities ADD COLUMN embedded_at TEXT;
```

This lets the system detect which entities need re-embedding after a model change.

### 3.3 Dimension Change Handling

If switching from the preview model's dimensions to a different dimension:

- **SQLite BLOBs**: No schema change needed — BLOBs are variable-length.
- **Cosine similarity** (`resolve.py`): Works with any dimension — no code change.
- **Critical constraint**: All vectors compared in a single similarity operation must have the same dimensionality. The re-embed workflow must be run to completion before the resolve stage can produce valid results.

**Recommendation**: Use MRL to set `output_dimensionality=768` initially (matching the preview model's output) to minimize disruption. Once all data is re-embedded, optionally increase to 3072 for better quality if storage allows.

A 3072-dim float32 vector = 12 KB per embedding. For a corpus with 100K chunks and 10K entities, that's ~1.3 GB. For this project's scale, 3072 is fine. If storage becomes a concern, MRL truncation to 768 (3 KB/vector) is always available.

### 3.4 Rollback Strategy

1. **Before re-embedding**: The old model still works. Rollback = revert the model constant.
2. **During re-embedding**: The `embedding_model` column on chunks lets you identify which rows have been re-embedded. Rollback = clear embeddings for rows with the new model name and re-embed with the old model (or restore from DB backup).
3. **After re-embedding**: Take a SQLite backup (`cp pipeline.db pipeline.db.bak`) before starting. Rollback = restore the backup.

**Recommended workflow**:
```bash
# 1. Backup
cp pipeline.db pipeline.db.bak

# 2. Re-embed
kg re-embed --model gemini-embedding-001

# 3. Validate (run resolve on a test source, check merge quality)
kg process --source-id 1 --stage resolve

# 4. If bad, rollback
cp pipeline.db.bak pipeline.db
```

### 3.5 Future Considerations

- **`gemini-embedding-2` upgrade**: If the project needs to embed non-text content (images, PDFs, video), switch to `gemini-embedding-2`. The centralized embedding module makes this a config change. Note: `gemini-embedding-2` uses prompt-based task instructions instead of the `task_type` parameter, so the `embed_texts()` function would need to format prompts differently.
- **Neo4j vector indexes**: If vector search is later added to Neo4j, the index `CREATE VECTOR INDEX` statement would need the dimension specified. Centralizing the dimension config now makes this easy.
- **Provider abstraction**: If the project may switch away from Google embeddings (e.g., to OpenAI, Cohere), introduce an `EmbeddingProvider` protocol. For now, staying within the Google ecosystem, a centralized module is sufficient.
- **Batch API**: For large re-embedding jobs, the Gemini Batch API offers 50% cost savings and higher throughput. Worth adding for corpora > 10K chunks.

---

## 4. Recommendations

### Immediate (this sprint)

1. **Switch to `gemini-embedding-001`** — the preview model has no stability guarantees and could be deprecated. This is a one-line change per file (or one-line if centralized first).

2. **Centralize embedding logic** — extract the duplicated model constant, client init, and byte conversion from `embed.py` and `resolve.py` into `src/agents_kg/embedding.py`.

3. **Add `embedding_model` column to entities table** — parity with chunks table, needed for migration tracking.

### Short-term (next 1-2 sprints)

4. **Add `kg re-embed` CLI command** — required for any model migration. Without this, changing models silently degrades entity resolution.

5. **Use task types** — pass `SEMANTIC_SIMILARITY` for entity resolution embeddings and `RETRIEVAL_DOCUMENT` for chunk embeddings.

6. **Set `output_dimensionality=768`** initially via MRL to keep vectors compatible during the transition, then evaluate whether 3072 improves resolve quality.

### Deferred

7. **Batch API support** for large-scale re-embedding.
8. **Neo4j vector index** if runtime similarity search is needed.
9. **Provider abstraction** only if multi-provider support becomes a requirement.

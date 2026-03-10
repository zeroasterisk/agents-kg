# Pipeline Orchestrator Comparison

## Context

Our pipeline (`fetch → parse → chunk → embed → extract → resolve → review → load`) currently runs on a hand-rolled SQLite job queue. This works for sequential processing of hundreds of sources, but as scope grows we should evaluate proper orchestrators.

## Current: SQLite Job Queue

**What we have:** Custom `pipeline.db` with sources/chunks/entities/edges tables, per-stage status tracking, retry with backoff, content-hash idempotency.

| ✅ Strengths | ❌ Limitations |
|---|---|
| Zero dependencies | No DAG visualization |
| Simple, fully understood | No parallelism primitives |
| Portable (single file) | No scheduling/cron built-in |
| Works offline | No observability UI |
| ~200 lines of code | Manual retry logic |

**Verdict:** Sufficient for Phase 1 (hundreds of sources, sequential). Gets painful at scale.

---

## Option A: Prefect 3

**What:** Python-native workflow orchestration. Decorators turn functions into observable, retryable tasks. Self-hosted or cloud.

**GitHub:** 20K+ stars, very active. v3 released 2024, latest 3.6+.

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=[10, 30, 60])
def fetch_source(url: str) -> Source:
    ...

@task
def parse(source: Source) -> str:
    ...

@task
def chunk(text: str) -> list[Chunk]:
    ...

@task(tags=["gemini"])  # rate-limit by tag
def extract(chunks: list[Chunk]) -> tuple[list[Entity], list[Edge]]:
    ...

@flow(log_prints=True)
def ingest_source(url: str):
    source = fetch_source(url)
    text = parse(source)
    chunks = chunk(text)
    entities, edges = extract(chunks)
    # resolve, review, load...
```

| ✅ Strengths | ❌ Limitations |
|---|---|
| Minimal code changes (decorators) | Prefect server = another container |
| Built-in retry, backoff, caching | Heavy dependency (~100MB) |
| DAG visualization UI | Overkill for <500 sources? |
| Scheduling, concurrency limits | Learning curve for advanced features |
| Self-hosted free tier | SQLite still needed for entity state |
| Background tasks, Redis-backed durable execution | |
| Native async support | |
| Artifact tracking (results, logs per task) | |

**Fit for us:** High. Our pipeline is already structured as discrete stages. Wrapping in `@task` decorators is ~1 day of work. The UI alone (seeing which sources failed at which stage) would save debugging time.

**Migration path:**
1. Add `@task` / `@flow` decorators to existing functions
2. Run `prefect server start` (Docker or bare)
3. Keep SQLite for entity/edge state (Prefect handles orchestration state)
4. Add scheduling for periodic re-crawls

---

## Option B: Luigi

**What:** Spotify's pipeline framework. Task classes with explicit dependencies. Battle-tested, mature, minimal.

**GitHub:** 18K+ stars, maintained but slower cadence than Prefect.

```python
import luigi

class FetchSource(luigi.Task):
    url = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"data/fetched/{hash(self.url)}.json")

    def run(self):
        source = fetch(self.url)
        with self.output().open('w') as f:
            json.dump(source, f)

class ParseSource(luigi.Task):
    url = luigi.Parameter()

    def requires(self):
        return FetchSource(url=self.url)

    def output(self):
        return luigi.LocalTarget(f"data/parsed/{hash(self.url)}.txt")

    def run(self):
        with self.input().open() as f:
            source = json.load(f)
        text = parse(source)
        with self.output().open('w') as f:
            f.write(text)
```

| ✅ Strengths | ❌ Limitations |
|---|---|
| Minimal, well-understood | Verbose (class per task) |
| File-based targets (natural checkpointing) | No built-in retry/backoff |
| Dependency graph visualization | Dated API (pre-async, pre-type-hints) |
| No server required (runs as script) | No caching/memoization |
| Lightweight (~5MB) | No concurrency control |
| Stable, battle-tested | Community momentum declining |

**Fit for us:** Medium. Luigi's file-target model maps well to our YAML canonical files (each stage produces a file → next stage reads it). But the verbose class-based API is friction compared to Prefect's decorators, and no retry/backoff means we keep our custom logic.

**Migration path:**
1. Wrap each pipeline stage as a `luigi.Task`
2. YAML files become Luigi targets (natural!)
3. Run `luigi --module pipeline IngestSource --url=...`
4. Still need custom retry logic

---

## Option C: Stay with SQLite (enhanced)

Add the missing pieces to our current system without an external orchestrator:

- **DAG viz:** Generate Mermaid diagrams from pipeline state
- **Scheduling:** Use system cron or OpenClaw cron jobs
- **Parallelism:** `concurrent.futures.ThreadPoolExecutor` for embed/extract
- **Observability:** CLI `kg status --detailed` already exists, add richer output

| ✅ Strengths | ❌ Limitations |
|---|---|
| No new dependencies | Reinventing wheels |
| Full control | No ecosystem/community |
| Already working | DAG viz is DIY |
| Minimal resource usage | |

---

## Recommendation

| Criteria | SQLite (current) | Prefect 3 | Luigi |
|---|---|---|---|
| Migration effort | — | Low (decorators) | Medium (class rewrites) |
| Retry/backoff | Custom ✅ | Built-in ✅ | Manual ❌ |
| Observability UI | CLI only | Web dashboard ✅ | Basic web ✅ |
| Dependencies | None | Heavy (~100MB) | Light (~5MB) |
| Scheduling | External | Built-in ✅ | External |
| Concurrency | Manual | Built-in ✅ | Basic |
| Community momentum | — | High 📈 | Declining 📉 |
| Async support | Manual | Native ✅ | No ❌ |
| Fit for our scale | ✅ now | ✅ grows with us | ✅ adequate |

**Phase 1 (now, <500 sources):** SQLite is fine. Don't add complexity yet.

**Phase 2 (when we hit pain):** Prefect 3 is the clear winner. Minimal migration (decorators), great UI, built-in retry/scheduling/concurrency, active community. Luigi is mature but dated — Prefect does everything Luigi does and more, with less boilerplate.

**Trigger to migrate:** When any of these hurt:
- Debugging failed sources without a UI
- Needing parallel extraction across many sources
- Wanting scheduled re-crawls
- Pipeline grows beyond 8 stages

---

## Adding Prefect (when ready)

```bash
# Add to project
uv add prefect

# Start server (Docker, same network as Neo4j)
# Add to Portainer stack:
#   prefect-server:
#     image: prefecthq/prefect:3-latest
#     ports:
#       - "4200:4200"
#     command: prefect server start --host 0.0.0.0

# Or run serverless (no UI, just local execution)
prefect config set PREFECT_SERVER_API_HOST=0.0.0.0
```

Prefect can also connect to our Neo4j — task results could write directly to the graph, with Prefect handling retry/backoff/observability and our code handling entity resolution and domain logic.

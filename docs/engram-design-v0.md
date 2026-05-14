# Engram — Local Agent Memory Service

**Status:** Draft v0.1 · for review
**Working name:** Engram (placeholder; trivial to rename)
**Author:** [you]
**Reviewers:** [TBD]
**Last updated:** 2026-05-13

---

## 1. Summary

Engram is a self-hosted, MCP-native memory service for AI agents. It runs alongside vLLM (or equivalent) on a personal headless inference server, reachable from the operator's devices over Tailscale wherever they happen to be. It provides a persistent, model-agnostic backing store that any MCP-capable client (Claude Code, Claude Desktop, opencode, ChatGPT, Cursor, Gemini CLI, custom Rust agents) can read from and write to.

It is OB1's architectural shape — Postgres + pgvector + a thin MCP gateway — implemented as a single Rust binary, with the local vLLM endpoint serving as the embedding and extraction backend, designed so that swapping the underlying embedding or extraction model is a routine operation rather than a migration.

The deployment target is single-user, single-active-session. Concurrent multi-user serving is explicitly not in scope.

The system is built incrementally across five milestones (§3.5). The remainder of this document describes the *terminal* state — all five milestones complete. Inline milestone callouts (e.g. `[M1]`, `[M2+]`) flag features that arrive at a specific milestone. §3.5 is the source of truth for what ships when, and supersedes anything elsewhere in the document that reads as if a feature is "v0."

## 2. Goals

- **Single source of memory** across every agent and model the operator uses.
- **Model-independence** at the storage layer: changing embedding or extraction model must not invalidate captured content.
- **Local-first**: defaults run with no cloud dependency. Cloud is a configurable opt-in per provider.
- **Provenance-preserving**: every derived fact links to the immutable raw thought that produced it. Extraction drift must be detectable and correctable.
- **Tiered exposure**: localhost / mesh / public, configurable, with auth that scales accordingly.
- **Operationally simple**: single Postgres, single Rust binary, runs under systemd.

## 3. Non-goals

- Not an agent runtime (cf. Letta). Engram stores and retrieves; agents live elsewhere.
- Not a temporal knowledge graph (cf. Graphiti). Facts are timestamped and supersedable, but we do not model validity windows as first-class entities.
- Not a vector database product. We use pgvector and we are happy.
- Not multi-tenant SaaS. Single operator, optional shared with trusted humans.
- No ML training. We use existing embedding / instruct models as black boxes.

## 3.5 Milestone roadmap

The system is built in five capability milestones, preceded by a small environment-setup milestone (M0). Each capability milestone is independently shippable: at the end of M1 the operator has a usable memory service; subsequent milestones add capability without invalidating prior ones.

**M0 — Development environment.** *The floor under the floor.*
- Postgres 16 running in Docker via `docker-compose.yml` at the repo root, using the `pgvector/pgvector:pg16` image (bundles `vector`, `pg_trgm`, `pgcrypto`).
- Ollama (already installed on the operator's box) serves as the dev-mode embedder via its OpenAI-compatible endpoint (`http://localhost:11434/v1/embeddings`, model `bge-m3`). Production retains the TEI sidecar.
- `DEVELOPMENT.md` runbook for first-time setup. No code is written; M0 only ensures M1's code has somewhere to run.

**M1 — Capture and search.** *The floor.*
- Schema ships in full (`thoughts`, `embeddings`, `facts`, `artifacts`, `artifact_chunks`) but only `thoughts` and `embeddings` are populated. Future-milestone tables exist now so later migrations don't touch live data.
- Sync embedding on `capture` via TEI sidecar (BGE-M3, 1024-dim).
- Hybrid retrieval: vector kNN ∪ trigram lexical search, fused via reciprocal rank fusion (RRF). No reranker.
- Four MCP tools: `capture`, `search_thoughts`, `recent_thoughts`, `get_thought`.
- Single binary; subcommands `serve` and `migrate`. No worker process.
- Tier 0 auth (localhost-only). Tier 1 (Tailnet) is a config change, not a code change.

**M2 — Facts pipeline.**
- `engram-extract` crate becomes real with a vLLM client; `Extractor` trait gains its first implementation.
- Worker process appears (`engram worker` subcommand). Reflector cron job runs.
- `facts` table populated; new MCP tools `search_facts`, `correct_fact`.
- The async-embedding seam designed at M1 is exercised: `capture` posts a job; the worker computes the embedding.

**M3 — Search quality.**
- BGE-reranker (also via TEI) plugged in after RRF fusion. Retrieve top-50, rerank to top-N.
- MCP search interface unchanged; quality goes up.

**M4 — Artifacts.**
- Long-form ingestion: `artifacts` and `artifact_chunks` populated. Chunking strategy lands here.
- New MCP tool: `ingest_artifact`.
- Search results unify thoughts and chunks under one ranking.

**M5 — Operational maturity.**
- Prometheus `/metrics` endpoint.
- Tier 2 bearer-token auth + audit log.
- Backup tooling (scripts, retention policy).
- Eval suite (capture-recall, cross-model retrieval consistency, LongMemEval-style).
- The `stats` MCP tool.

**Order rationale.** M1 is the floor: nothing else makes sense without capture and retrieval. M2 (facts) before M3 (rerank) because facts add capability and rerank improves quality, and quality without capability is unmotivated. M4 (artifacts) before M5 (operational) because ingesting existing notes/transcripts earns its keep faster than auth/eval ceremony for a single-operator tool.

## 4. High-level architecture

```
                   ┌──────────────────────────────────────────┐
                   │             Engram (single binary)       │
                   │                                          │
  MCP clients      │   ┌──────────┐    ┌────────────────┐     │
  (Claude Code, ──→│──→│ MCP/HTTP │───→│   Core service │     │──┐
   Desktop, etc.)  │   │  surface │    │  (capture,     │     │  │
   over Tailscale  │   └──────────┘    │   retrieval,   │     │  │
                   │                   │   reflection)  │     │  │
                   │   ┌──────────┐    └────────────────┘     │  │
                   │   │  Worker  │            │              │  │
                   │   │  (cron)  │────────────┘              │  │
                   │   └──────────┘   [M2+]                   │  │
                   │         │                                │  │
                   │         ▼                                │  │
                   │   ┌──────────────────────────────────┐   │  │
                   │   │  Embedder + Extractor (traits)   │   │  │
                   │   │  default: OpenAI-compatible      │   │  │
                   │   └──────────────────────────────────┘   │  │
                   │                  │                       │  │
                   └──────────────────┼───────────────────────┘  │
                                      ▼                          ▼
                            ┌──────────────────┐         ┌────────────┐
                            │  vLLM endpoint   │         │ Postgres   │
                            │  (instruct +     │         │ + pgvector │
                            │   embedding,     │         │            │
                            │   localhost:8000)│         └────────────┘
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  RTX 3090(s)     │
                            └──────────────────┘
```

Engram is a *client* of the local vLLM endpoint, not the operator of it. vLLM is presumed to be serving primary inference traffic to other Tailscale-connected devices anyway; Engram piggybacks on that infrastructure. Three logical components, one binary:

- **MCP/HTTP surface.** Streamable HTTP transport speaking MCP. Same binary also exposes an admin HTTP API.
- **Core service.** Capture, search, fact retrieval, scope management.
- **Worker.** [M2+] Periodic session reflection, deferred re-embedding, fact compaction. Runs in-process with a Tokio scheduler when the binary is launched in `worker` mode. **The worker process does not exist in M1**; capture-side embedding is synchronous in the server process.

## 5. Data model

The model is deliberately small. Three primary entities — thoughts, embeddings, facts — plus an artifacts table for long-form content. Embeddings are intentionally a separate first-class table so model swaps are routine rather than migrations.

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Raw, immutable captures. Single source of truth.
CREATE TABLE thoughts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope           TEXT NOT NULL DEFAULT 'global',
    content         TEXT NOT NULL,
    source          TEXT NOT NULL,           -- 'manual', 'agent:claude-code', 'reflector', etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX thoughts_scope_recent_idx
    ON thoughts (scope, created_at DESC);
CREATE INDEX thoughts_content_trgm_idx
    ON thoughts USING gin (content gin_trgm_ops);

-- Long-form content. Reserved for M4.
CREATE TABLE artifacts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope           TEXT NOT NULL DEFAULT 'global',
    kind            TEXT NOT NULL,           -- 'document'|'transcript'|'code'|'web'|...
    title           TEXT,
    content_uri     TEXT,                    -- file:// or s3:// for blobs
    content_text    TEXT,                    -- inline if small
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE artifact_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id     UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    UNIQUE (artifact_id, chunk_index)
);

-- Embeddings are first-class. Multiple per target during model migration.
CREATE TABLE embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind     TEXT NOT NULL CHECK (target_kind IN ('thought','artifact_chunk','fact')),
    target_id       UUID NOT NULL,
    model_id        TEXT NOT NULL,           -- e.g. 'bge-m3:1024'
    model_version   INT NOT NULL DEFAULT 1,
    vector          vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (target_kind, target_id, model_id, model_version)
);

-- One HNSW partial index per active embedding model. M1 ships this one.
-- Adding a new model = a future migration adds a new partial index over
-- the same table; old rows stay; the active-model concept lives in config
-- (see §9), not in a Postgres GUC.
CREATE INDEX embeddings_bge_m3_hnsw
    ON embeddings USING hnsw (vector vector_cosine_ops)
    WHERE model_id = 'bge-m3:1024';

-- Structured facts. Populated from M2 onward by the reflector (and by
-- `correct_fact` for manual overrides). `source_run_id` joins back to
-- `reflector_runs` so a whole bad run can be jointly retracted later.
CREATE TABLE facts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope               TEXT NOT NULL,
    statement           TEXT NOT NULL,       -- natural-language fact
    subject             TEXT,                -- optional structured triple
    predicate           TEXT,
    object              TEXT,
    source_thought_id   UUID REFERENCES thoughts(id) ON DELETE CASCADE,
    source_chunk_id     UUID REFERENCES artifact_chunks(id) ON DELETE CASCADE,
    extractor_model     TEXT NOT NULL,       -- 'vllm/qwen2.5-7b-instruct' | 'manual' | ...
    extractor_version   INT NOT NULL,
    confidence          REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    superseded_by       UUID REFERENCES facts(id),
    superseded_at       TIMESTAMPTZ,
    source_run_id       UUID REFERENCES reflector_runs(id),  -- added at M2; NULL for manual rows
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (source_thought_id IS NOT NULL OR source_chunk_id IS NOT NULL)
);

CREATE INDEX facts_active_idx
    ON facts (scope, created_at DESC)
    WHERE superseded_at IS NULL;

-- M2 additions follow. These three tables ship in migration 0002.

-- Durable FIFO queue backing the async embedding seam. Capture inserts a
-- row; `engram worker`'s drainer task pulls batches via
-- `UPDATE ... FROM (SELECT ... FOR UPDATE SKIP LOCKED LIMIT $1)`.
-- The UNIQUE constraint makes enqueue idempotent.
CREATE TABLE pending_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind     TEXT NOT NULL CHECK (target_kind IN ('thought','artifact_chunk','fact')),
    target_id       UUID NOT NULL,
    model_id        TEXT NOT NULL,                    -- pairs the job with the right embedder
    enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    attempts        INT NOT NULL DEFAULT 0,
    last_attempt_at TIMESTAMPTZ,
    last_error      TEXT,
    UNIQUE (target_kind, target_id, model_id)
);
CREATE INDEX pending_embeddings_dequeue_idx ON pending_embeddings (enqueued_at ASC);

-- One row per reflector pass. Backs facts.source_run_id so an entire bad
-- run can be retracted by joining on this id. `error` is only populated
-- when the run itself fails at the orchestrator level — per-thought
-- extractor failures are soft and counted via the n_* fields.
CREATE TABLE reflector_runs (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at             TIMESTAMPTZ,
    extractor_model         TEXT NOT NULL,
    extractor_version       INT NOT NULL,
    scope_filter            TEXT,
    n_thoughts_processed    INT NOT NULL DEFAULT 0,
    n_facts_committed       INT NOT NULL DEFAULT 0,
    n_review_queue          INT NOT NULL DEFAULT 0,
    error                   TEXT
);

-- Landing zone for low-confidence extractions. `decision` is operator-flipped
-- (Phase D ships pending → accept/reject; the review-queue UX lands at M5).
CREATE TABLE facts_review_queue (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement           TEXT NOT NULL,
    subject             TEXT,
    predicate           TEXT,
    object              TEXT,
    confidence          REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    source_thought_id   UUID REFERENCES thoughts(id) ON DELETE CASCADE,
    source_chunk_id     UUID REFERENCES artifact_chunks(id) ON DELETE CASCADE,
    extractor_model     TEXT NOT NULL,
    extractor_version   INT NOT NULL,
    source_run_id       UUID REFERENCES reflector_runs(id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at         TIMESTAMPTZ,
    decision            TEXT NOT NULL DEFAULT 'pending'
                          CHECK (decision IN ('pending','accept','reject')),
    CHECK (source_thought_id IS NOT NULL OR source_chunk_id IS NOT NULL)
);
```

**Why embeddings are a separate table.** A model swap is a re-index, not a re-write. With this layout we insert a new row in `embeddings` per `(target, new model)`, build a new HNSW partial index for the new model, and once the operator is satisfied with retrieval quality, optionally drop the old rows and old index. No data is lost during the swap.

**One HNSW index per model.** Each active embedding model gets its own partial index, predicated on a literal `model_id` string. This is required for correctness: Postgres demands partial-index predicates be `IMMUTABLE`, and `current_setting()` is `STABLE`. The "active embedder" is therefore a config concern (see §9), not a database GUC. Operationally, swapping the active model means: ship a migration that adds the new partial index, update config to point at the new `model_id`, restart.

**Scoping.** Free-form string, default `global`. Convention rather than enforcement: `work.tcgplayer`, `personal`, `project.engram`, etc. A `scopes` registry table can come later if introspection is wanted.

## 6. Ingest path

There are two write paths. Both terminate in the same `thoughts` row plus an embedding.

1. **Direct capture.** [M1] Agent calls `capture(content, scope?, source?, metadata?)`. The handler inserts the thought, computes its embedding via TEI, writes the embedding row, returns the thought ID. **In M1 this is fully synchronous** — capture returns when the embedding is durable. At low-hundreds-of-captures-per-day with TEI sidecar latency under 200 ms, the wait is invisible.

2. **Artifact ingestion.** [M4] Agent calls `ingest_artifact(uri, kind, scope?)`. The handler inserts the artifact row and hands off to the worker, which fetches, chunks, embeds, and writes `artifact_chunks` plus their embeddings.

**Designed-in seam for async embedding.** [M2+] In M1 the capture handler calls `Embedder::embed(...)` directly. In M2 the worker process appears, and the same capture handler is changed in *one place*: instead of calling `embed` inline, it enqueues a job; the worker drains the queue. The MCP tool contract stays identical; capture continues to return a thought ID immediately; the embedding row becomes available shortly after (with a brief window during which `search_thoughts` may not surface the brand-new thought via vector — trigram still finds it).

The worker also runs the **session reflector** [M2+], opt-in via `[reflector] enabled = true`. On its cron schedule (`tokio-cron-scheduler` 6-field cron; default `0 0 3 * * *` — 03:00 daily) it walks **unfacted thoughts** (`LEFT JOIN facts ON source_thought_id WHERE facts.id IS NULL`, ASC by `created_at`), asks the extractor to derive structured facts, and writes them with full provenance (`source_run_id` → `reflector_runs`). The reflector pipeline is the subject of §6.5; §10 covers drift defense.

## 6.5 Fact extraction pipeline

This is the M2 capability that distinguishes Engram from "MCP server + Postgres + a vector index." On a schedule the operator controls, the reflector reads each unfacted thought and derives **facts** — structured rows that capture what the thought claims, queryable as data.

**Why facts matter.** Thoughts are raw and free-form; you search them lexically (trigram) or semantically (vector kNN). Facts are a second, structured layer over the same captures: each row is a self-contained natural-language statement, optionally decomposed into an `(S, P, O)` triple, with a `confidence`, a pointer back to the source thought, and provenance (`extractor_model`, `extractor_version`, `source_run_id`). The same captures become two independently-queryable surfaces — natural language for "what did I write?" and structured data for "what claims exist about X?" — without either surface being the source of truth. The thought is the source of truth; facts are derived data that can be re-derived from a different model.

**The pipeline.** Six discrete steps, all in `engram worker`'s reflector task:

1. **Open a run.** `INSERT INTO reflector_runs (extractor_model, extractor_version, scope_filter) RETURNING id`. Every fact produced by this pass will carry this `run_id` in `facts.source_run_id` — so an entire bad run can later be jointly retracted with `UPDATE facts SET superseded_at = NOW() WHERE source_run_id = ...`.

2. **Walk unfacted thoughts.** `SELECT thoughts.* FROM thoughts LEFT JOIN facts ON facts.source_thought_id = thoughts.id WHERE facts.id IS NULL [AND thoughts.scope = $1] ORDER BY thoughts.created_at ASC LIMIT $2`. The LEFT-JOIN-IS-NULL guarantees idempotency: re-running the reflector on a stable corpus produces no new rows. Thoughts whose only facts are *superseded* still have rows in `facts` and so are excluded — re-extracting a thought the operator already corrected would defeat the correction.

3. **Extract.** For each thought, call `Extractor::extract(thought, ctx)`. The default impl (`OpenAICompatibleExtractor`) POSTs to `/v1/chat/completions` with a system prompt that defines the output schema and `response_format: { type: "json_schema", json_schema: { strict: true, schema: { ... } } }`. JSON-Schema-guided decoding (vLLM's `xgrammar`/`outlines`, OpenRouter's structured outputs) makes the response shape guaranteed-parseable. Schema:

    ```json
    {
      "type": "object", "additionalProperties": false,
      "properties": {
        "facts": {
          "type": "array",
          "items": {
            "type": "object", "additionalProperties": false,
            "properties": {
              "statement":  { "type": "string" },
              "subject":    { "type": ["string", "null"] },
              "predicate":  { "type": ["string", "null"] },
              "object":     { "type": ["string", "null"] },
              "confidence": { "type": "number" }
            },
            "required": ["statement", "subject", "predicate", "object", "confidence"]
          }
        }
      },
      "required": ["facts"]
    }
    ```

    On per-thought extractor failure (`Timeout`, `Unreachable`, `Backend 5xx`, malformed response, model-id mismatch), the reflector **soft-fails**: logs a warning with `transient = err.is_transient()`, increments `n_extractor_failures`, and continues with the next thought. The unfacted thought stays in the LEFT-JOIN-IS-NULL set and the next tick retries it. No special "extractor attempted but failed" marker is needed.

4. **Route by confidence.** Facts with `confidence ≥ review_queue_below` (default 0.7) commit to `facts` via `insert_fact(NewFact { ... })`. Below threshold, they land in `facts_review_queue` for operator decision — `decision` defaults to `'pending'`; the review-queue UX lands at M5. M2 ships single-band routing; the three-band design (with a "stored but flagged" middle band requiring a `flagged` column on `facts`) is deferred — see §10.

5. **Close the run.** `UPDATE reflector_runs SET finished_at = NOW(), n_thoughts_processed = $1, n_facts_committed = $2, n_review_queue = $3 WHERE id = $4`. The run-row is the audit anchor: an operator can later see exactly when each extractor pass happened, how much it produced, and which model produced it.

6. **(Optional) Operator review.** The CLI surface is `engram reflect [--scope <s>] [--limit <n>]` for an on-demand pass instead of waiting for the cron, and `engram reflect --rerun [--since <RFC3339>]` to re-evaluate already-facted thoughts when the extractor model is upgraded. Rerun is **additive only**: existing active facts the new extractor doesn't reproduce stay active. Rationale in §10 §5.

**Concrete example.** A thought: *"Talked to Sarah today about the PR backlog. She wants migration #0042 fast-tracked because the mobile freeze starts Thursday."* A reasonable extraction:

```json
{ "facts": [
    { "statement": "Sarah wants migration #0042 fast-tracked",
      "subject": "Sarah", "predicate": "wants fast-tracked", "object": "migration #0042",
      "confidence": 0.9 },
    { "statement": "Mobile freeze starts Thursday",
      "subject": "mobile freeze", "predicate": "starts", "object": "Thursday",
      "confidence": 0.85 }
] }
```

Both facts commit (0.9 and 0.85 are above the default 0.7). They're now retrievable via `search_facts("mobile freeze")` independently of vector kNN over the thought, and `get_thought(source_id)` returns both in `linked_facts`.

**What the operator gets.** Facts are first-class data with full provenance. Concrete queries that become trivial:

```sql
-- All active facts the current extractor version produced this week
SELECT statement, confidence FROM facts
WHERE superseded_at IS NULL
  AND extractor_model = 'vllm/qwen2.5-7b-instruct'
  AND extractor_version = 1
  AND created_at >= NOW() - INTERVAL '7 days';

-- Facts produced by a specific run (useful when reviewing a run before/after a model swap)
SELECT statement, confidence FROM facts WHERE source_run_id = $1;

-- Which thoughts are still unfacted in a scope?
SELECT t.id, t.content FROM thoughts t
LEFT JOIN facts f ON f.source_thought_id = t.id
WHERE f.id IS NULL AND t.scope = 'work';

-- Operator-authored corrections vs. machine-authored facts
SELECT statement FROM facts WHERE extractor_model = 'manual';
```

These are the queries that make `engram audit` (M5) trivial. They also make Engram's "memory I can reason about" pitch real: a year from now, the operator can ask the agent "what did I learn about X?" and get back structured rows with confidence and provenance — not a vector-search blob.

## 7. Retrieval path

Three retrieval primitives, composable:

- **Semantic** — vector kNN over the active embedding model.
- **Lexical** — `pg_trgm` similarity over `content`. Cheap; complements vector search for proper nouns, acronyms, and code identifiers — exactly the queries pure embeddings are notoriously bad at.
- **Recency** — `ORDER BY created_at DESC` with a scope filter.

**Default `search_thoughts` from M1 is a hybrid.** Concretely:

1. Run two SQL queries in parallel: a vector kNN limited to top-K against the active model's HNSW index; a trigram similarity query limited to top-K against `thoughts_content_trgm_idx`.
2. Fuse the result sets with reciprocal rank fusion (RRF): `score(d) = Σᵢ 1 / (k + rankᵢ(d))` over the two rankings, with `k` typically 60.
3. Apply scope filtering and a recency boost (multiplicative `exp(-age/τ)` with `τ` = 30 days, configurable per call). Return the top N.

Why RRF over a weighted-score blend like `α·cos_sim + β·bm25 + γ·exp(-age/τ)`: RRF is parameter-light, robust to score-distribution differences between heterogeneous rankers, and is the de-facto choice for vector + lexical hybrids in current information-retrieval literature. It also generalizes cleanly to a third ranker when the M3 reranker is added.

```rust
pub struct SearchRequest {                       // [M1]
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,                    // defaults to 10; max 100
    pub recency_half_life_days: Option<f32>,     // default 30; 0 disables
}

pub struct SearchFactsRequest {                  // [M2 Phase D]
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,
    pub recency_half_life_days: Option<f32>,     // keyed on source thought's created_at
}
```

The two share a shape and an error type (`ReadError`). `search_facts` is trigram-only in M2 — the vector leg lands in M3 alongside the reranker. Active (non-superseded) facts derived from a thought come back via `get_thought`'s `linked_facts` field rather than a flag on `SearchRequest`.

**Reranker.** [M3] M3 adds a cross-encoder rerank pass after RRF fusion: retrieve a wider candidate set (typically top-50), rerank with BGE-reranker via TEI to get the final top-N. The MCP search interface is unchanged.

## 8. MCP surface

Tools and the milestone in which each ships. Names and signatures are part of the contract once shipped.

| Tool | Milestone | Purpose |
|---|---|---|
| `capture` | M1 | Store a thought. Returns `thought_id` + `embedding_status: "pending"`; the worker drains the embed queue on its tick. From M2, `embedding_status` is *always* `pending` (the synchronous-embed path is gone). |
| `search_thoughts` | M1 | Hybrid retrieval over thoughts. RRF-fused vector + trigram + recency boost; gracefully degrades to trigram-only when the embedder is unreachable. |
| `recent_thoughts` | M1 | Browse by recency in a scope. |
| `get_thought` | M1 (M2 adds `linked_facts`) | Full thought + provenance (`embedding_status`, `embedded_at`) plus active (non-superseded) facts derived from it. |
| `search_facts` | M2 | Trigram retrieval over `facts.statement`, filtered to active rows. Each result includes the source thought's content/scope/created_at — no follow-up `get_thought` call needed. M3 adds the vector leg. |
| `correct_fact` | M2 | Operator-driven correction. With a replacement: inserts a manual-author row (`extractor_model="manual"`, `extractor_version=0`, `source_run_id=NULL`, `confidence=1.0`) and supersedes the old one. Without: retracts via supersede with no successor. Audit trail (`superseded_by`, `superseded_at`) is preserved either way. |
| `ingest_artifact` | M4 | Async ingest of a longer document. |
| `stats` | M5 | Per-scope counts, last activity, embedding model version. |

`correct_fact` is the explicit operator knob for "the extractor got it wrong." The manual-sentinel provenance shape (`extractor_model = "manual"`, `extractor_version = 0`) lets a single query separate machine-authored facts from human-authored ones: `WHERE extractor_model = 'manual'` finds operator corrections; `WHERE extractor_model <> 'manual' AND extractor_version < N` finds stale-extractor rows that need re-evaluation.

## 9. Embedding & extraction abstraction

Two traits, one config struct, no other architectural concession to model choice.

```rust
#[async_trait]                      // [M1]
pub trait Embedder: Send + Sync {
    fn model(&self) -> &EmbeddingModel;          // { id: String, dimensions: usize }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError>;
}

#[async_trait]                      // [M2]
pub trait Extractor: Send + Sync {
    fn model_id(&self) -> &str;                  // e.g. "vllm/qwen2.5-7b-instruct"
    fn version(&self) -> i32;                    // bumped when prompt/schema changes
    async fn extract(
        &self,
        thought: &Thought,                       // takes the full row, not just &str
        ctx: &ExtractionContext,
    ) -> Result<Vec<ExtractedFact>, ExtractorError>;
}

pub struct ExtractedFact {          // [M2] — write shape (extractor output)
    pub statement: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub confidence: f32,
}

pub struct Fact {                   // [M2 Phase D] — read shape (DB row)
    pub id: Uuid,
    pub scope: Scope,
    pub statement: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub source_thought_id: ThoughtId,
    pub extractor_model: String,
    pub extractor_version: i32,
    pub source_run_id: Option<Uuid>,
    pub confidence: f32,
    pub created_at: OffsetDateTime,
}
```

Both `EmbedderError` and `ExtractorError` carry an `is_transient(&self) -> bool` so capture/drain/reflector orchestrators can decide retry vs. mark-failed on a per-call basis. `Timeout`, `Unreachable`, and `Backend { status: 500..=599, .. }` are transient; everything else is not.

**Default implementations:**

- **`OpenAICompatibleEmbedder`** [M1] — one type covering every OpenAI-`/v1/embeddings`-shaped backend by config alone: Ollama (dev default, `http://localhost:11434/v1` + `bge-m3`), Hugging Face TEI (production sidecar), OpenAI, Voyage. Dimension validation against the declared `model.dimensions` is built in.
- **`FakeEmbedder`** [M1] — deterministic in-memory embedder for tests. Configurable via `FakeBehavior` to always fail (`Timeout` / `Unreachable`) so soft-fail paths can be exercised without standing up TEI/Ollama.
- **`OpenAICompatibleExtractor`** [M2] — one type covering every OpenAI-`/v1/chat/completions`-shaped backend that supports `response_format: { type: "json_schema" }`. Named-constructor presets `OpenAICompatibleConfig::vllm_local()` (default `http://localhost:8000/v1`, no key) and `::open_router(api_key, model)` (`https://openrouter.ai/api/v1` + Bearer auth) cover the production-sidecar and cloud-fallback paths. Operators picking any other backend (LM Studio, SGLang, etc.) use the `openai-compatible` provider with custom `endpoint` + `model_name`.
- **`FakeExtractor`** [M2] — deterministic in-memory extractor mirroring `FakeEmbedder`. `with_confidence(f32)` drives review-queue-routing tests; `with_facts(Vec<ExtractedFact>)` pins explicit outputs; `always_failing(FakeBehavior)` exercises the soft-fail path.

The trait boundary is the buffer-from-model-changes guarantee. Swapping vLLM's served model, swapping to SGLang, swapping to a cloud provider — all happen behind the same interface. The only operation that propagates beyond the trait is a re-embed when the *embedder* changes (which is why §5 makes embeddings a separate table).

**Active-embedder selection.** From M1 onward the active embedder is identified by `model_id` (e.g. `bge-m3:1024`) and is a config field — the engram TOML declares which model is active; that string must match the predicate of an existing HNSW partial index. There is intentionally no Postgres-side GUC.

Configuration is a TOML file:

```toml
[database]
# Postgres connection. Overridden by DATABASE_URL env var if set (sqlx convention).
url = "postgres://engram:engram@localhost:5432/engram"
max_connections = 10

[server]                                        # [M1]
bind = "127.0.0.1:8080"                         # Tier 0 default — see §12

[embedder]                                      # [M1]
provider     = "openai-compatible"
endpoint     = "http://localhost:11434/v1"      # Ollama in dev; TEI in production
model        = "bge-m3"                         # backend-side model name
model_id     = "bge-m3:1024"                    # Engram-side identity; must match an HNSW index
dimensions   = 1024
timeout_seconds = 5

[worker]                                        # [M2]
tick_interval_seconds = 5                       # embed-drainer wakeup cadence
batch_size            = 16                      # max jobs per tick

[extractor]                                     # [M2+]
provider              = "openai-compatible"     # alternative: "openrouter"
endpoint              = "http://localhost:8000/v1"   # vLLM default
model_name            = "qwen2.5-7b-instruct"   # backend-side model name
model_id              = "vllm/qwen2.5-7b-instruct"   # provenance label → facts.extractor_model
model_version         = 1                       # bump when prompt/schema changes
timeout_seconds       = 60
temperature           = 0.2
max_facts_per_thought = 8

[reflector]                                     # [M2+]
enabled               = false                   # opt-in: flip to true once vLLM is up
schedule              = "0 0 3 * * *"           # 6-field cron (sec min hour dom month dow). 03:00 daily.
scope_filter          = ""                      # blank = all scopes
max_thoughts_per_run  = 1000
max_facts_per_thought = 8
review_queue_below    = 0.7                     # confidence < this → facts_review_queue; ≥ → facts
```

Single-threshold routing (`review_queue_below`) is the M2 Phase D simplification — the three-band design originally sketched in §10 (with a "stored but flagged" middle band) is deferred until `facts` grows a `flagged` column. See §10 and `docs/milestones/m2-progress.md`.

**Hardware sizing — concrete on the Phase 1 / Phase 2 BOM, single-user.**

The box is a personal inference server: one operator, one active session at a time, accessed over Tailscale from wherever the operator is. There is no concurrent multi-user load to budget for. The binding constraint is fitting the served instruct model + embedder + a single session's KV cache in available VRAM.

**Phase 1 (single RTX 3090, 24 GB VRAM):**

The default optimizes for tool-use quality, since the operator's stated use case is opencode / Claude Code against the local endpoint:

| Component | Choice | VRAM |
|---|---|---|
| vLLM-served instruct | Qwen2.5-Coder-32B-Instruct AWQ-int4 | ~19 GB |
| Embedder | BGE-M3 in TEI, **CPU build** | 0 GB (system RAM) |
| **KV cache headroom** | | **~5 GB → ~32K tokens single-session** |

CPU embeddings via TEI on the 9800X3D run at ~50–150 ms per call. Engram's actual call rate is a few embeddings per minute at peak personal use, not thousands, so the latency is invisible. The trade is real: capture latency goes from ~10 ms (GPU TEI) to ~100 ms (CPU TEI), and ~5 GB of KV cache headroom comes back to vLLM. For single-user code-agent work that almost always stays under 32K tokens, this is the right deal.

**Why Coder-32B over a smaller model.** For strong tool use against opencode / Claude Code, model quality at the tool-call schema and multi-step planning level matters more than peak throughput. Qwen2.5-Coder-32B is one of the few open models where tool calling holds up under real agent loops — error recovery, multi-step planning, long tool-result reasoning. A 14B class model is sufficient for Engram's *own* extraction needs but underperforms on the operator's primary use case.

**Reflection cost** [M2+]. A reflector pass over 50 thoughts is ~4k input tokens → ~1k structured output. At Coder-32B's vLLM throughput on a 3090 (≈75 tok/s per stream per the BOM), that's roughly 60 seconds. Default schedule is `0 0 3 * * *` — 6-field `tokio-cron-scheduler` cron meaning 03:00 daily; contention with active agent work is non-existent at that hour. For dev/test, tighten to something like `*/30 * * * * *` (every 30 seconds) or drive a one-shot pass with `engram reflect [--scope <s>] [--limit <n>]`.

**Embedder placement is a deployment-time choice, not a code change.** TEI ships CPU and CUDA builds with identical HTTP APIs (`ghcr.io/huggingface/text-embeddings-inference:cpu-1.x` vs `:1.x`). Switching is a systemd unit edit; the Engram TOML doesn't change. CPU is the v0 default; GPU is appropriate later if capture rate grows or the operator wants sub-100ms capture latency for some interaction pattern.

**Phase 2 (dual RTX 3090, 48 GB VRAM):**

Phase 2 is a quality upgrade rather than a necessity-driven one — Phase 1 single-user is genuinely a credible primary daily driver. The upgrade unlocks:

- Qwen2.5-Coder-32B at Q6/Q8 (better quality than AWQ-int4) with full KV cache via tensor-parallel
- 70B-class general models at Q4 (Llama 3.3 70B, Qwen 2.5 72B) for harder reasoning tasks, ~32 tok/s per the BOM
- DeepSeek-V2.5/V3 (235B MoE, ~21B active) at Q4 — explicitly strong at agentic work, ~25 tok/s per the BOM

vLLM's `--tensor-parallel-size 2` is the obvious deployment shape. The embedder either stays on CPU or moves to a single card via TEI's CUDA build; both are easy.

**System RAM and storage.** Postgres + pgvector will be MB-to-low-GB scale even with 100k+ thoughts; the 64 GB system RAM is overprovisioned for Engram's purposes (and is there for vLLM's CPU offload / weights loading anyway). With CPU embedding the embedder also runs out of system RAM — BGE-M3 is ~2 GB resident — well within budget. On the 2 TB NVMe, Engram's footprint is dominated by the database (single-digit GB at realistic scale); vLLM model weights are the actual storage hog.

## 10. Provenance, extraction drift, and reconciliation

§6.5 describes the fact-extraction pipeline as a capability — what it produces and how. This section is the defensive counterweight: how Engram keeps extraction honest, given that LLMs hallucinate, model versions drift, and the operator may not notice for months. Everything below is M2+ behavior — M1 has no extractor and therefore no facts to drift.

This section addresses the operator's specific concern: *"I don't want there to be drift from truth when the session reflector extracts facts."*

**Five mechanisms, in order of importance:**

1. **Raw thoughts are immutable.** Extractors never modify `thoughts`. They only write to `facts` with a foreign key back. If extraction is wrong, the truth is still recoverable.

2. **Every fact carries its extractor identity.** `extractor_model` + `extractor_version` on every row. When the local extractor is upgraded, `WHERE extractor_version < N` finds every fact that needs re-evaluation.

3. **Confidence-gated commit.** The extractor returns a self-rated confidence. Below `review_queue_below` (default 0.7 as shipped in M2 Phase C/D), facts go to `facts_review_queue` for operator review; at-or-above, they commit directly to `facts`. M2 ships single-threshold routing. The original three-band design — adding a middle "stored but flagged" band between `review_queue_below` and a separate `min_confidence_to_store` — is deferred until `facts` gains a `flagged` column; tracked in `docs/milestones/m2-progress.md`.

4. **Dual-extractor reconciliation (optional).** When `extractor.dual_run = true`, every reflection pass runs two distinct models (e.g. local Qwen3 and cloud Claude Haiku) and only commits facts both produced. Disagreements surface as review-queue items. Roughly doubles cost; recommended for high-stakes scopes only.

5. **Re-extraction is a first-class operation.** `engram reflect --rerun --scope work --since 2026-01-01` re-runs the current extractor against historical thoughts and reconciles. New facts whose `(subject, predicate, object)` exactly matches an existing active row and whose `statement` is identical are no-ops (idempotency keystone); same `(S, P, O)` with a different `statement` inserts a new row and supersedes the old via `superseded_by`; new `(S, P, O)` triples insert as additional facts. **Phase D ships this additively only** — existing active facts that the new extractor doesn't reproduce stay active. Rationale: a single rerun reflects model drift in *how* facts are stated, not *what* the thought says, and subtractive logic risks losing real facts to sampling variance. Operators retract obsolete rows manually via `correct_fact`.

**What this does not protect against:** a confident-and-wrong extractor producing a high-confidence wrong fact that no other extractor disagrees with. The mitigation is human review via `correct_fact` and periodic `engram audit` reports that surface low-traffic facts (potentially stale) and high-confidence facts that contradict source thoughts on lexical inspection.

## 11. Deployment & ops

**Target hardware:** Phase 1 of the BOM — RTX 3090 (24 GB), Ryzen 7 9800X3D, 64 GB DDR5-6000, 2 TB PCIe 5.0 NVMe, Ubuntu 24.04 LTS, NVIDIA driver 560+, CUDA 12.6+. Postgres 16+ with `pgvector` ≥ 0.7 (HNSW required), `pg_trgm`, `pgcrypto`. Phase 2 (dual 3090) is fully supported by the same software stack with one config change (`CUDA_VISIBLE_DEVICES`).

**Components:**

- `engram` — the single Rust binary. M1 supports `serve` and `migrate` subcommands; `worker` joins at M2.
- Postgres 16 with `pgvector` ≥ 0.7, `pg_trgm`, `pgcrypto`. Connection is configured by URL (TOML or `DATABASE_URL` env). Local Unix socket is the simplest deployment; remote TCP — same Tailnet, separate NAS or DB host, or anywhere reachable — is fully supported. **Extensions must be installed on the Postgres server**, not the Engram host. At personal-scale data with HNSW indexes, network round-trip on a LAN adds negligible latency to queries.
- `text-embeddings-inference` HTTP server for BGE-M3, sidecar pattern. **CPU build by default** for v0; swap to CUDA build by changing the systemd unit's container image (no Engram code or config change needed). Required from M1.
- vLLM serving an instruct model — required from M2 onward (no extractor in M1). **Operated independently of Engram.** Engram is a client; the operator manages vLLM's lifecycle, model choice, and serving config. Engram only requires the OpenAI-compatible endpoint to be reachable.

**Process model:** systemd units. `engram-server.service` exists from M1. `engram-tei.service` (the embeddings sidecar; CPU build by default — see §9) is required from M1. `engram-worker.service` joins at M2 and runs *two* Tokio tasks inside one process: the embed-drainer (always on, pulls jobs off `pending_embeddings`) and the reflector (cron-scheduled via `tokio-cron-scheduler`, default `0 0 3 * * *` — 6-field cron, 03:00 daily). vLLM and Postgres run as their own units, managed independently. **The reflector task is opt-in**: `engram worker` defaults to `[reflector] enabled = false` so the binary works without vLLM running; the operator flips the flag once the extractor backend is up.

**Why a cron schedule rather than continuous** [M2+]. Single-user means the reflector competes only with the operator's own active agent sessions for vLLM throughput. Scheduling it for off-hours (overnight, default) eliminates that contention entirely. If the operator wants more aggressive extraction for a specific scope, the schedule is per-scope tunable via the admin API.

**Backups:** `pg_dump --format=custom` nightly to a separate disk; weekly to a remote (Backblaze B2 or rsync.net). Embeddings are derived data and don't strictly need backing up — `engram reflect --rebuild-embeddings` regenerates them — but including them speeds disaster recovery.

**Migrations:** `sqlx migrate`. Schema changes ship with the binary.

**Observability** [M5]. Structured `tracing` logs to journald are present from M1. The Prometheus `/metrics` endpoint exposing capture-rate, search-latency P50/P95/P99, embedding-queue depth, extractor failures, and fact-review-queue size lands at M5.

## 12. Auth & network exposure

Three relevant tiers. They map to milestones, not to deployment options offered all at once.

| Tier | Network | Auth | Milestone | Use case |
|---|---|---|---|---|
| **0 — Localhost** | `127.0.0.1` only | None | M1 | First-run validation; the development default. |
| **1 — Mesh** | Tailscale / WireGuard | None (mesh = auth) | M1 (config change) | Personal devices already on the Tailnet. The ops-recommended endpoint for single-user deployment. |
| **2 — Tunnel** | Cloudflare Tunnel / Caddy + LE | Bearer token | M5 | Non-Tailnet clients (Claude Desktop, ChatGPT) that need a public HTTPS MCP URL. |

A "Tier 3 — public + multi-user" option exists in principle but is **explicitly out of scope** for the current roadmap. It would require OAuth2, per-client tokens, and audit log; implementable later if the system is genuinely shared with another person, which is not a current requirement.

**Tier 1 is the recommended endpoint for single-user deployment.** Engram binds to the Tailnet interface and is reachable as `engram.tailXXXX.ts.net` from every personal device, using the same MagicDNS pattern as vLLM. No code change vs. Tier 0; only the bind address.

**Auth at Tier 2** [M5]. Bearer token validated against a hashed allowlist in `engram_tokens`. Tokens carry a scope-list — a token can be locked to `work.*` and not see `personal.*`. Audit log records `(token_id, tool, args_hash, ts)` for every call.

## 13. Evaluation

[M5] — eval suite ships at the operational-maturity milestone. We don't ship without it because "did the model swap regress retrieval" is the kind of question we'll ask ourselves often.

**Three suites, all reproducible from a fixture corpus:**

1. **Capture-recall.** Synthetic conversations seeded with target facts; check that subsequent semantically-relevant queries surface the right thoughts and facts.
2. **Cross-model retrieval consistency.** Re-embed the same fixture with a new embedder; measure overlap of top-10 results vs. baseline. Drop > 30% triggers a manual review before the swap is committed in production scopes.
3. **LongMemEval-style.** Subset of the public benchmark adapted to our schema. Apples-to-apples comparison against published Mem0 / Zep / Letta numbers.

Eval runs end-to-end in `engram eval --suite <name>` and dumps a JSON report.

## 14. Open questions

Resolved during the milestone-roadmap planning conversation (see Revision history):

1. ~~**Inference box specs.**~~ Resolved: Phase 1 RTX 3090 / 9800X3D / 64 GB; Phase 2 adds a second 3090.
2. ~~**v0 scope.**~~ Resolved: see §3.5 milestone roadmap. M1 = capture + hybrid search + MCP; facts/extractor/worker deferred to M2.
3. ~~**Search architecture.**~~ Resolved: hybrid (vector ∪ trigram, RRF) at M1; reranker at M3.
4. ~~**Active-embedder mechanism.**~~ Resolved: config-driven `model_id`, one HNSW partial index per model.

Carrying forward:

5. **Naming.** Engram is a placeholder. (Hippocampus, Cortex, Lattice, Mneme are all in the drawer.)
6. **Sync.** Do we ever want multi-machine replication? Logical replication on Postgres is straightforward, but only worth doing if you'll actually use it. Defer.
7. **Capture UX.** OB1's Slack capture is clever. Equivalents: a Telegram bot, a CLI `engram capture`, a Raycast/Alfred extension, a browser extension. Out of scope until at least M5.
8. **Embedding model default.** v0 commits to BGE-M3 (well-established, multilingual, runs in ~1.5 GB, supports rerank). A future milestone should bake off Qwen3-Embedding-4B and Qwen3-Embedding-8B against our own eval fixture before any production-scope re-embed. The embeddings table design (§5) makes this a routine swap rather than a migration.
9. **Are we storing agent transcripts?** Currently artifacts can hold them (M4+); we haven't decided whether agents auto-capture session transcripts on close or whether that's an explicit flush.
10. **Extractor model: dense vs. MoE.** Phase 2 unlocks Qwen3-30B-A3B (MoE, 3B active) as an alternative to Qwen2.5-32B (dense). The MoE option likely wins on throughput; quality on our specific extraction prompts is unmeasured. Decide via the eval suite (M5).

## 15. Out of scope (for the foreseeable future)

- Knowledge-graph reasoning (Cognee/Graphiti territory).
- Memory forgetting / TTL policies (everything is forever; pruning is a post-M5 conversation).
- Multi-modal memory (images, audio).
- Federated query across multiple Engram instances.
- A web UI. Postgres + `psql` is the admin interface.
- Public + multi-user deployment ("Tier 3" in §12).

## Revision history

- **2026-05-09** — Initial v0 draft by Claude Desktop in a "technical PM" capacity.
- **2026-05-09** — Revised by engineer + architect after the milestone-roadmap brainstorm. Added §3.5 milestone roadmap. Corrected schema in §5: added `CREATE EXTENSION` lines for `pgcrypto`/`vector`/`pg_trgm`; removed trailing comma in `thoughts`; replaced the `current_setting`-based partial HNSW index (which the Postgres planner rejects, since `current_setting` is `STABLE` not `IMMUTABLE`) with a literal-model partial index (`embeddings_bge_m3_hnsw`); added `thoughts_scope_recent_idx` and `thoughts_content_trgm_idx`; added `target_kind` CHECK on `embeddings`. Reframed §6 (M1 sync embedding via TEI; M2+ async seam), §7 (RRF hybrid; reranker M3), §8 (per-tool milestone column), §9 (Embedder M1, Extractor M2; `CloudEmbedder` added; active-embedder via config). Reframed §12 auth tiers as a milestone progression and dropped Tier 3 from the table. Pruned resolved open questions in §14. Doc now describes the M5-complete terminal state with milestone callouts inline.
- **2026-05-13** — **M2 complete.** Shipped in four phases A–D (see `docs/milestones/m2-progress.md`). Facts pipeline live: async embedding seam (capture enqueues; `engram worker` drains), reflector cron via `tokio-cron-scheduler` 0.15 (default off — opt-in via `[reflector] enabled = true`), `OpenAICompatibleExtractor` covering vLLM and OpenRouter via named-constructor presets, two new MCP tools (`search_facts`, `correct_fact`), `get_thought` now carries active `linked_facts`, and a new `engram reflect` subcommand with `--rerun [--since <RFC3339>]` for re-extracting historical thoughts (idempotent; supersedes on (S,P,O)-match-but-statement-differs; additive only). **Phase D simplification:** `search_facts` ships trigram-only inside an RRF-shaped pipeline — fact embeddings are wired through migration 0001's `target_kind = 'fact'` enum but the worker doesn't yet enqueue facts; the vector leg lands in M3 (search quality) alongside the cross-encoder reranker. **`correct_fact` provenance:** manual rows use the sentinel `extractor_model = "manual"`, `extractor_version = 0`, `source_run_id = NULL`, `confidence = 1.0`. Three-band confidence routing (the "flagged but committed" middle band from §10) is deferred — needs a `flagged` column on `facts` that doesn't exist yet. M2 success criteria #1–#5 met by code; #6 (operator dogfood ≥ 1 week) is the only remaining open item.
- **2026-05-13** — Reconciled doc against shipped M1 + M2 code. §5 schema block extended with migration 0002's three tables (`pending_embeddings`, `reflector_runs`, `facts_review_queue`) and `facts.source_run_id`. §7 `SearchRequest` snippet matches the shipped struct (no `mode: SearchMode` or `include_facts: bool` — neither was implemented); added a `SearchFactsRequest` peer. §8 tool descriptions tightened: `capture` documents the always-pending return; `get_thought` calls out `linked_facts`; `search_facts` notes the trigram-only / M3-vector-leg state; `correct_fact` documents the manual-sentinel provenance and retract-via-no-replacement variant. §9 trait signatures match the code (`Embedder::model() -> &EmbeddingModel`, `Extractor` takes `&Thought`, both return typed errors with `is_transient()`); added the `Fact` read-shape struct; replaced the fictional `TeiEmbedder` / `CloudEmbedder` / `OpenRouterExtractor` defaults with the actual `OpenAICompatibleEmbedder` / `FakeEmbedder` / `OpenAICompatibleExtractor` / `FakeExtractor` set. §9 default-config TOML and §11 process-model paragraph match shipped fields and values (6-field cron, opt-in reflector, single-threshold routing). §10 mechanism #3 reframed as single-band with the three-band band noted as deferred; mechanism #5 documents `--rerun` as additive-only.
- **2026-05-13** — Added §6.5 "Fact extraction pipeline" as the affirmative companion to §10. §6.5 leads with *why facts matter* (the structured-second-layer story: same captures, two queryable surfaces, thought stays source-of-truth), walks the six-step pipeline (open run → walk unfacted thoughts → extract via JSON-Schema-guided decoding → route by confidence → close run → optional operator review/rerun), shows the exact `response_format` JSON Schema, gives a worked example (a casual conversation capture becoming two facts), and ends with operator-facing SQL ("here are the queries that become trivial once you have a facts table"). §10 reframed as the drift-defense counterweight — same content, but explicitly positioned as the defensive complement to §6.5 rather than the only place facts are discussed.
- **2026-05-13** — M3 starter shipped early in response to M2 dogfood: first-class thought retraction. Migration `0003_thoughts_retraction.sql` adds `thoughts.retracted_at` + `thoughts.retracted_reason` and an `(scope, created_at DESC) WHERE retracted_at IS NULL` partial index. New `retract_thought(thought_id, reason?)` MCP tool and `engram-storage::retract_thought` fn atomically (a) sets the trust-state column and (b) auto-supersedes every active fact derived from the thought. All retrieval paths (`recent_thoughts`, `search_trigram`, `search_vector_knn`, `search_facts_trigram`) and reflector paths (`find_unfacted_thoughts`, `find_facted_thoughts`, `enqueue_unembedded_thoughts`) now filter `retracted_at IS NULL`; `get_thought` is the audit path and still returns the row with `retracted_at` / `retracted_reason` exposed on the response. Motivation: M2 dogfood (see `docs/milestones/m2-progress.md` 2026-05-13 history) showed that the previous workaround — retract every derived fact one at a time via `correct_fact` — fails as soon as the operator misses any fact, because the unretracted-thought-with-one-active-fact stays in the reflector's `find_facted_thoughts` set and gets re-extracted on the next `engram reflect --rerun`. The atomic supersede + DB-invariant filter closes that gap. Note: this expands the M3 scope (M3 was originally search-quality only) but the work was pulled in early because it gates honest dogfood — operators iterating on captures will inevitably need a way to mark wrong claims as untrusted. The reranker + fact embeddings remain the rest of M3.

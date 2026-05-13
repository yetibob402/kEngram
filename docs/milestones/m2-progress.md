# M2 — Progress

Living checklist tracking M2 implementation. Each phase ends in a runnable, reviewable checkpoint. Items are checked off as they land; the **History** section at the bottom captures dated notes — decisions made in passing, surprises, things deferred. The companion design doc is `m2-facts-pipeline.md` in this directory; the operator's 12 inline-answered open questions there are the binding decisions this plan is built on.

## Operator decisions captured (from m2-facts-pipeline.md)

| # | Question | Decision |
|---|---|---|
| 1 | Async embedding mechanism | `pending_embeddings` queue table, `SELECT FOR UPDATE SKIP LOCKED` drain |
| 2 | Reflector batching | "Thoughts without facts" (LEFT JOIN IS NULL), ASC by `created_at` |
| 3 | Extractor prompt design | OpenAI `response_format` JSON Schema |
| 4 | `source_run_id` | **Yes**, with a `reflector_runs` table backing it |
| 5 | Dual-extractor disagreement | Defer entirely to M5 |
| 6 | Facts search strategy | Same RRF hybrid as `search_thoughts` |
| 7 | `Extractor` trait location | `engram-core` |
| 8 | Worker process structure | One `engram worker` process, two Tokio tasks |
| 9 | vLLM unreachable | Per-thought soft-fail, log, continue |
| 10 | `correct_fact` provenance | `extractor_model = "manual"`, version 0 |
| 11 | Cron scheduler crate | `tokio-cron-scheduler 0.15` |
| 12 | `search_facts` response shape | Include source thought content + scope + created_at |

**Other settled sub-decisions:**

- `engram embed-backfill` (M1's CLI) **survives** as a manual one-shot drain escape hatch — semantics unchanged.
- `capture`'s `embedding_status` becomes `"pending"` as the *normal* return (was the exception case in M1). MCP wire shape unchanged.
- `reqwest 0.13.3` upgrade landed in its own commit before M2 Phase A starts (`ddd3aad`).

## Phase A — Foundation

End state: migration applied; new crate compiles; types and traits exist; nothing wired up yet.

- [ ] Migration `0002_facts_pipeline.sql`:
  - [ ] `pending_embeddings` queue: `(id UUID PK, target_kind TEXT, target_id UUID, model_id TEXT, enqueued_at TIMESTAMPTZ, attempts INT, last_attempt_at TIMESTAMPTZ, last_error TEXT)`
  - [ ] `facts_review_queue`: `(id UUID PK, statement, subject, predicate, object, confidence, source_thought_id, extractor_model, extractor_version, source_run_id, created_at, reviewed_at, decision TEXT)` — `decision` ∈ `pending|accept|reject`
  - [ ] `reflector_runs` table: `(id UUID PK, started_at, finished_at, extractor_model, extractor_version, scope_filter TEXT, n_thoughts_processed INT, n_facts_committed INT, n_review_queue INT, error TEXT)`
  - [ ] Add `source_run_id UUID REFERENCES reflector_runs(id)` to `facts` (nullable for `manual` rows)
  - [ ] Index `pending_embeddings_dequeue_idx (enqueued_at ASC)` for FIFO drain
  - [ ] Index `facts_review_queue_pending_idx (created_at ASC) WHERE decision = 'pending'`
- [ ] New crate `engram-extract`:
  - [ ] Workspace member added in root `Cargo.toml`
  - [ ] Compiles empty
- [ ] `engram-core` additions:
  - [ ] `ExtractedFact { statement, subject?, predicate?, object?, confidence }` type
  - [ ] `ExtractionContext { scope, source_thought_id, max_facts }` type
  - [ ] `Extractor` trait with `model_id()`, `version()`, `async extract(thought, ctx) -> Result<Vec<ExtractedFact>, ExtractorError>`
  - [ ] `ExtractorError` enum with `is_transient()` classification (Timeout, Unreachable, 5xx are transient — mirror `EmbedderError`)
- [ ] `cargo test --workspace`: still 106 passing
- [ ] `cargo clippy --all-targets -- -D warnings`: clean

## Phase B — Async embedding seam

End state: capture handler no longer calls `Embedder::embed` inline; it enqueues. Worker process drains. New thought visible by trigram immediately; vector kNN within one worker tick.

- [ ] `engram-storage` repository functions:
  - [ ] `enqueue_embedding(pool, target_kind, target_id, model_id)`
  - [ ] `claim_pending(pool, batch_size) -> Vec<PendingJob>` (SKIP LOCKED inside a tx)
  - [ ] `mark_embedded(pool, pending_id)` (delete the row)
  - [ ] `mark_failed(pool, pending_id, error_msg)` (bump attempts, leave row for next tick)
  - [ ] `sqlx::test`: enqueue + claim + mark_embedded round trip; SKIP LOCKED across two simulated workers
- [ ] `engram-mcp::capture` rewritten:
  - [ ] No `Embedder` dependency in capture path
  - [ ] Insert thought, enqueue, return `embedding_status: "pending"`
  - [ ] `sqlx::test`: capture writes thought + pending_embeddings row; no embedding row yet
- [ ] `engram-cli` `worker` subcommand:
  - [ ] Embed-drainer task: loop, claim up to N, embed via `Embedder::embed`, persist, mark embedded; on transient failure log + mark_failed; on permanent failure log + leave (operator inspects)
  - [ ] Configurable tick interval (default 5s) and batch size (default 16)
  - [ ] Graceful shutdown on ctrl-c
- [ ] `engram-mcp::backfill` (M1's `embed-backfill`) updated to also drain `pending_embeddings` in addition to LEFT-JOIN-IS-NULL set
- [ ] End-to-end test: capture while no worker running → thought exists, no embedding → start worker → embedding row appears within one tick
- [ ] DEVELOPMENT.md: section on running `engram worker` alongside `engram serve`

## Phase C — Extractor + reflector

End state: vLLM-driven extractor produces facts; reflector cron walks unfacted thoughts and writes facts with run provenance; review queue receives low-confidence rows.

- [ ] `engram-extract` impls:
  - [ ] `OpenAICompatibleExtractor` — `/v1/chat/completions` with `response_format: { type: "json_schema", json_schema: {...} }`; default `endpoint = http://localhost:8000/v1`; default model name from config
  - [ ] `OpenRouterExtractor` — same shape, with `Authorization: Bearer <key>` and OpenRouter's endpoint
  - [ ] Tests with `wiremock`: valid response → facts; malformed JSON → MalformedResponse error; 5xx → transient; missing API key → fatal
  - [ ] **`integration` feature**: live test against running vLLM (skipped by default, run with `--features integration`)
- [ ] `engram-storage` repository functions for facts:
  - [ ] `start_run(pool, extractor_model, extractor_version, scope_filter) -> RunId`
  - [ ] `finish_run(pool, run_id, n_processed, n_committed, n_review, error?)`
  - [ ] `find_unfacted_thoughts(pool, scope?, limit) -> Vec<Thought>` (LEFT JOIN facts IS NULL, ASC by created_at)
  - [ ] `insert_fact(pool, NewFact)` with `extractor_model`, `extractor_version`, `source_run_id`, `confidence`
  - [ ] `insert_review_queue_row(pool, NewReviewRow)`
- [ ] Reflector task (in `engram-cli` worker):
  - [ ] `tokio-cron-scheduler` set up with default schedule from config (`0 3 * * *`)
  - [ ] On tick: `start_run`, walk unfacted thoughts in scope-order, call extractor per thought, soft-fail on extractor unreachable, route facts to `facts` or `facts_review_queue` per confidence thresholds, `finish_run`
  - [ ] `sqlx::test` with `FakeExtractor` (analogue of `FakeEmbedder`): produces N facts, all committed; thresholds route low-confidence to review queue; failed extractor calls bump nothing
- [ ] Config:
  - [ ] `[extractor]` section in `engram.toml`: provider, endpoint, model, temperature, max_facts_per_thought, response_format
  - [ ] `[reflector]` section: schedule cron string, `review_queue_below`, `min_confidence_to_store`
  - [ ] Validation: extractor present only if M2+ features needed; `engram serve` doesn't require it

## Phase D — MCP tools + manual reflect + dogfood

End state: M2 success criteria from m2-facts-pipeline.md met. Operator-driven dogfood ticked off after a week of use.

- [ ] `engram-mcp` tools:
  - [ ] `search_facts(query, scope?, limit?) -> { results: [{ fact_id, statement, subject?, predicate?, object?, confidence, source_thought_id, source_thought_content, source_thought_scope, source_thought_created_at, score }] }` — same RRF hybrid as thoughts, filters `WHERE superseded_at IS NULL`
  - [ ] `correct_fact(fact_id, replacement?) -> { new_fact_id?, superseded: bool }` — special `extractor_model = "manual"`, `extractor_version = 0`; sets `superseded_by`, `superseded_at` on old row; inserts new row pointing at same `source_thought_id` if `replacement` provided
  - [ ] `get_thought` updated to include `linked_facts: [...]` (rows where `source_thought_id = ?` and `superseded_at IS NULL`)
  - [ ] `sqlx::test`s for each: round trip, supersession audit, `search_facts` filters superseded
- [ ] `engram reflect` subcommand:
  - [ ] `engram reflect [--scope <s>] [--limit <n>]` — one-shot reflector run, exits when done
  - [ ] `engram reflect --rerun --scope <s> [--since <date>]` — re-extract historical thoughts; for each, if `(subject, predicate, object)` matches an existing non-superseded fact, **merge** (no new row); if it conflicts, supersede via `superseded_by`. Audit trail preserved.
  - [ ] `sqlx::test`: rerun twice produces identical fact set (idempotency criterion)
- [ ] Documentation:
  - [ ] README.md status table: M2 ✅ with brief sentence
  - [ ] DEVELOPMENT.md: vLLM prerequisites, `engram worker` runbook, `engram reflect` examples
  - [ ] Design doc revision-history entry
- [ ] **Operator-driven**: MCP smoke test from Claude Code / `mcp-inspector` invoking `search_facts`, `correct_fact`, and the updated `get_thought` against `engram serve` (with `engram worker` running in parallel)
- [ ] **Operator-driven**: real dogfood — run engram with extractor for ≥1 week, confirm fact rate and false-positive/-negative balance is acceptable, do at least one `correct_fact` round trip

## History

Dated notes appended as items land. Format: `YYYY-MM-DD — <one-line summary>`. Multi-line entries fine for decisions that need explanation.

<!-- Most recent entry first. -->

- **2026-05-12** — M2 design conversation closed. All 12 open questions in m2-facts-pipeline.md answered inline by RJF; only #4 diverged from the engineer's lean (operator opted **For** adding `source_run_id`, and during synthesis we agreed to back it with a small `reflector_runs` table so the data is actually queryable). Three additional sub-decisions settled: `engram embed-backfill` survives as an escape hatch; capture's `embedding_status` becomes `"pending"` as the normal return (semantic shift only — MCP wire shape unchanged); `reqwest 0.13.3` upgrade landed as its own commit (`ddd3aad`) before Phase A. Plan above is the next-conversation artifact; Phase A is the first concrete unit of work.

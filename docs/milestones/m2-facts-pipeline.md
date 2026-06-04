# M2 — Facts pipeline

## Goal

kEngram derives structured facts from captured thoughts on a scheduled basis. The operator can search facts, correct wrong ones, and trust that the thoughts/facts split preserves provenance and supports re-extraction.

This is the milestone that takes kengram beyond "search engine for thoughts" and toward "memory with structure." It also exercises the async-embedding seam designed (but not used) in M1.

## In scope

- The `Extractor` trait (in `kengram-core`) plus two implementations: `OpenAICompatibleExtractor` (vLLM `/v1/chat/completions` with structured-output) and `OpenRouterExtractor` (cloud fallback).
- Worker process: a new `kengram worker` subcommand. Long-running; runs the reflector cron + drains async-embedding queue.
- Async embedding: capture posts a job (in `pending_embeddings` or via NOTIFY/LISTEN; mechanism TBD); worker drains and calls `Embedder::embed`. Capture returns immediately with the thought ID. Brief window where the thought is searchable by trigram only.
- Reflector: scheduled task that walks recent thoughts in a scope, calls `Extractor::extract`, writes facts with `extractor_model`, `extractor_version`, `confidence`.
- Confidence-gated commit: facts below `review_queue_below` go to a review queue; between that and `min_confidence_to_store` are flagged but committed; above are committed normally.
- Two new MCP tools: `search_facts`, `correct_fact`. `get_thought` now joins linked facts.
- New CLI subcommands: `kengram worker`, `kengram reflect [--rerun] [--scope <s>] [--since <date>]`.
- Optional dual-extractor reconciliation (`extractor.dual_run = true`): commit only facts that two distinct extractors both produce.

## Out of scope (deferred to which milestone)

- Cross-encoder reranker → **M3**
- Artifact ingestion → **M4**
- `kengram audit` reports, human review UI, eval suite, Prometheus metrics, Tier 2 auth → **M5**
- Knowledge-graph reasoning → out of scope indefinitely

## Schema impact

Migration `0002_facts_pipeline.sql` adds:

- A `pending_embeddings` queue table (or equivalent NOTIFY/LISTEN setup; design TBD in M2 planning).
- A `facts_review_queue` table for low-confidence facts.

The existing `facts` table is now populated by code. No structural change to `facts` itself.

## MCP surface delta

- `search_facts(query: string, scope?: string, limit?: int) -> { results: [{ fact_id, statement, subject?, predicate?, object?, confidence, source_thought_id, score }] }`
- `correct_fact(fact_id: uuid, replacement?: { statement, subject?, predicate?, object? }) -> { new_fact_id?: uuid, superseded: bool }` — if `replacement` is provided, writes a new fact pointing at the same source; marks the old one superseded. If omitted, just supersedes (effectively delete-by-supersede).
- `get_thought(thought_id)` response now includes `linked_facts: [...]` populated from `facts WHERE source_thought_id = ?`.

## Crate structure delta

- **New crate: `kengram-extract`.** Defines the `Extractor` trait (moved from `kengram-core` — or kept in `kengram-core` and re-exported, TBD) and concrete impls `OpenAICompatibleExtractor`, `OpenRouterExtractor`. JSON-Schema response handling lives here.
- **`kengram-cli`** gains the `worker` and `reflect` subcommands; the `serve` subcommand learns to refuse async-embedding work (it goes to the worker).
- **`kengram-storage`** gains repository functions for facts: insert with provenance, search facts (vector + trigram fused, similar shape to thoughts), supersede a fact, query the review queue.
- **`kengram-mcp`** gains the two new tool handlers.

## Dependencies

- **Prior milestones:** M1 (capture, search, embedder, MCP scaffold).
- **External services:** vLLM serving an instruct model on `:8000/v1` (or an OpenRouter API key for cloud fallback). vLLM was not required in M1; it's required from M2.

## Success criteria

1. Reflector runs on schedule; produces facts with confidence; respects review-queue thresholds.
2. `search_facts` returns relevant facts for a query that the underlying thoughts cover.
3. `correct_fact` correctly supersedes a prior fact (`superseded_by`, `superseded_at` set on the old row; new row inserted) and `search_facts` no longer surfaces the old one (assuming it filters `WHERE superseded_at IS NULL`).
4. **Async embedding correctness:** capture a thought while TEI is down. Capture succeeds and returns a thought ID. Bring TEI back up. Within one worker tick, the embedding row appears and `search_thoughts` finds the thought via vector.
5. **Re-extraction idempotency:** `kengram reflect --rerun --scope work --since 2026-01-01` run twice produces the same `facts` table (same rows; same supersession history; no duplicate facts).
6. **Operator dogfood:** the operator runs M2 for at least a week, has at least one `correct_fact` round-trip, and is satisfied with the rate of false-positive vs. false-negative facts.

## Open questions

Each item is the engineer's lean + reasoning, followed by an `RJF:` placeholder for the operator's answer. Once all are filled in, this milestone gets a Phase-A planning conversation that produces a plan file.

### 1. Async embedding mechanism

Capture must return immediately with a thought ID; the actual `Embedder::embed` call moves to the worker. How does the capture side hand the work off?

- **(a)** A dedicated `pending_embeddings` queue table. Capture inserts a row; worker drains with `SELECT FOR UPDATE SKIP LOCKED`. Durable, observable (`SELECT COUNT(*) FROM pending_embeddings` = backlog), no new ops dependency.
- **(b)** Postgres `LISTEN/NOTIFY` from capture. Worker holds a long-lived subscription. Racy: notifications can be dropped on connection death — you'd want (a) as a backstop anyway.
- **(c)** External queue (`pgmq` extension, `apalis` Redis-backed, etc.). New service to operate.

**Lean: (a).** Well-trodden pattern, observable, zero new dependencies. If we ever want push-style wakeup we can layer NOTIFY on top of (a) later.

**RJF:** (a)

### 2. Reflector batching strategy

What does "recent thoughts" mean when the reflector wakes up?

- **(a)** Strict cron-window — thoughts created since last tick. Fragile: worker downtime = lost extractions.
- **(b)** "All thoughts that don't yet have any facts" — LEFT JOIN on `facts`, IS NULL. Same shape as M1's embed-backfill. Survives downtime cleanly.
- **(c)** Per-thought as soon as it's "old enough" (e.g. 5 minutes). Real-time-ish; competes with active vLLM use.

Within whichever set, ordering matters: ASC by `created_at` so oldest unfacted thoughts get processed first. Per-scope round-robin is over-engineered for single-user (one scope will dominate).

**Lean: (b)** with ASC ordering. No per-scope round-robin.

**RJF:** (b)

### 3. Extractor prompt design

How do we get structured output from vLLM?

- **(a)** `response_format = {"type": "json_schema", "json_schema": {...}}`. Supported by vLLM and OpenRouter. One JSON Schema defines the output shape; the model is constrained to it.
- **(b)** Grammar-constrained (gbnf-style). More powerful, more backend-specific.
- **(c)** Free-text response + parse / fix-it loop. Fragile.

**Lean: (a).**

**RJF:** (a)

### 4. Facts `source_run_id`

A nullable `UUID` column populated per reflection run. Lets a bad run's facts be jointly retracted later.

- **For:** cheap to add now; the consumer UX (`kengram audit`) lands in M5 but the data is there waiting.
- **Against:** M2 has no consumer for it; we usually push back on "add column for hypothetical future."

**Lean: genuinely undecided. Operator call.** (My instinct says yes — small cost, real provenance value — but it's exactly the kind of speculative addition the project pattern rejects.)

**RJF:** For

### 5. Dual-extractor disagreement handling

`extractor.dual_run = true`: run two distinct extractors, do what when they disagree?

- **(a)** Ship the mechanism in M2. Disagreements go to the review queue; agreements get committed.
- **(b)** Defer dual-run entirely to M5. M2 ships single-extractor + confidence gating only.

**Lean: (b), defer to M5.** Dual-run is a quality-evaluation mechanism, and M5 ships the eval suite. M2's confidence gating is already a complete useful slice.

**RJF:** (b)

### 6. Facts search strategy

- **(a)** Same RRF hybrid (vector kNN ∪ trigram, fused) as `search_thoughts`. Code reuse; trigram works fine on short text.
- **(b)** Weighted toward exact-statement match, since facts are short and more structured than thoughts.

**Lean: (a).** Revise after dogfood if retrieval quality disappoints.

**RJF:** (a)

### 7. Trait location for `Extractor`

- **(a)** In `kengram-core`, alongside `Embedder`. Trait lives at the abstraction boundary; impls live in `kengram-extract`. Symmetry with M1.
- **(b)** In `kengram-extract` (where it lives). Smaller `kengram-core`.

**Lean: (a).** Avoids any circular-dep concern; lets `kengram-mcp` depend only on `kengram-core` for the trait.

**RJF:** (a)

### 8. Worker process structure

- **(a)** One `kengram worker` process running two Tokio tasks: embed-queue drainer (every few seconds) + reflector cron (daily by default). One systemd unit.
- **(b)** Two subcommands (`kengram worker --drain`, `kengram worker --reflect`). Two systemd units.

**Lean: (a).** Reflector idle 23h/day is fine. Simpler to operate.

**RJF:** definitely (a)

### 9. Reflector behavior when vLLM is unreachable

- **(a)** Per-thought soft-fail: log error, mark the thought as "extractor-attempted but failed" somehow, continue with next. Mirrors M1's embedder soft-fail. Next tick retries.
- **(b)** Bail the whole run on first failure.

**Lean: (a).** Soft-fail per thought; no special "attempted but failed" mark needed if we just retry from the LEFT-JOIN-IS-NULL set next tick.

**RJF:** (a)

### 10. `correct_fact` provenance

When the operator manually corrects a fact, what goes in `extractor_model` / `extractor_version` on the new row?

- **(a)** Special sentinel: `extractor_model = "manual"`, `extractor_version = 0`. Provenance stays uniform; queries like "facts not produced by the current extractor" work cleanly across machine- and human-authored facts.
- **(b)** Some other column (`source = "manual"`) instead of squashing them through `extractor_model`.

**Lean: (a).**

**RJF:** (a)

### 11. Cron scheduler crate

- **(a)** `tokio-cron-scheduler` — mature, in-process, supports standard cron syntax.
- **(b)** `cron` crate + a small Tokio loop. More DIY but fewer dependencies.

**Lean: (a).** Flag for revision in Phase-A planning if a closer look turns up issues.

**RJF:** (a)

### 12. `search_facts` response shape

Does the response include a snippet of the source thought, or just the fact + `source_thought_id`?

- **(a)** Include `source_thought_content` (and `source_thought_scope`, `source_thought_created_at`) in each result. One extra JOIN. Agent UX: no follow-up `get_thought` call needed to make sense of the fact.
- **(b)** Just the fact rows + `source_thought_id`. Agent calls `get_thought` if it wants context.

**Lean: (a).**

**RJF:** (a)

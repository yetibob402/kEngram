# M3 — Search and extraction quality

## Goal

The original framing was narrow: improve retrieval with a cross-encoder rerank pass; "almost-right" results become "the right one is in the top three." That stays the headline. But M3's scope expanded mid-M2-dogfood (2026-05-13) as the facts pipeline produced enough real data to expose three classes of quality issue that aren't reranker-shaped: (a) the extractor produces noise on mixed-content thoughts; (b) `--rerun` is additive-only so paraphrase-level duplicates accumulate; (c) observability gaps make it hard to tell run-time symptoms apart from data-quality symptoms.

M3 absorbs those alongside the reranker. The milestone is now "everything that makes search and extraction trustworthy in dogfood," not just "rerank thoughts." The reranker is still the largest and most-impactful single change.

## In scope

### Retrieval quality (the original M3)

- **Cross-encoder reranker** (default: BGE-reranker-v2-m3 or comparable) running in TEI's rerank-task mode — either as a second TEI instance or as a multi-task TEI deployment (TBD).
- **Rerank stage in the search pipeline:** retrieve top-K (default 50) via RRF fusion, rerank with the cross-encoder to top-N (default 10).
- **Configurable per-call:** `rerank: bool` (default `true`), `candidate_pool: int` (default 50).
- **Both `search_thoughts` and `search_facts` (from M2) gain rerank support.**
- **Fact embeddings.** Extend the async-embedding seam to enqueue `target_kind = 'fact'` rows in `pending_embeddings`. The embed-drainer learns to vectorize fact statements alongside thought content (no schema change — `embeddings.target_kind` already lists `'fact'`). The vector leg in `search_facts_trigram` becomes a real `search_facts_vector_knn` call; `search_facts` graduates from trigram-only-inside-RRF-shape to actual hybrid retrieval. This was the M2 Phase D simplification; M3 closes it.
- **Eval-suite-style A/B comparison harness** — small, ad-hoc; the full eval suite lands at M5 — used to validate that rerank actually helps on a fixture corpus.

### Pulled forward from M2 (already landed)

- **✅ Thought retraction** (`retract_thought` MCP tool + `thoughts.retracted_at` + auto-supersede of derived facts). Landed 2026-05-13 in commit `636c910`. Was originally framed as M5 (`engram audit`); promoted to M3 starter when M2 dogfood demonstrated that retract-per-fact-via-correct_fact was a structural footgun. See design-doc revision history for the architectural rationale; m2-progress.md 2026-05-13 history for the dogfood data that motivated it.

### Pipeline quality (added 2026-05-13 from M2 dogfood)

- **Extractor prompt v3 with anti-examples.** v2 added an episodic-skip clause but didn't prevent test-narrative noise leaking through when it was embedded in an otherwise-durable thought. Real offending phrasings from dogfood (carry these as negative examples in v3):
  - "A search was conducted with scope='X'"
  - "A probe thought was captured at scope X"
  - "The probe thought was NOT returned when searching with scope='X'"
  - "The behavior was re-verified on 2026-05-13"
  Plus a positive heuristic: *"would this claim still be useful to a reader six months from now, independent of when it was captured? If no, skip."* And explicit mixed-content guidance: *"if a thought contains both a finding (durable) and a test report (transient), extract only the finding."* Bump `model_version` 2 → 3. Decision deferred to *after* the M2 dogfood week so v2's signal stays clean; revisit then.

- **Capture-time `extract` metadata flag.** Some thoughts mix durable findings with test narrative; the operator can't `retract_thought` them without losing the signal. A first-class `metadata.extract: "durable-only" | "none" | "all"` flag the reflector honors at extraction time. Pairs naturally with the metadata-schema-promotion conversation we punted on when adding `thoughts.retracted_at` — there's an emerging case for a small structured metadata vocabulary rather than fully free-form JSONB.

- **Subtractive logic + paraphrase-aware merge on `--rerun`.** M2 Phase D shipped additive-only rerun: facts the new extractor doesn't reproduce stay active. Visible in M2 dogfood: thought `3cc013b3` carries 6 active near-duplicates because `(subject, predicate, object)` match doesn't catch paraphrase-level overlaps ("the scope convention is dotted notation" vs. "engram thoughts use dotted scope strings"). Two paths to evaluate, decide during M3 planning:
  - **Embedding-space similarity at merge time** — load each new fact's embedding, compare against existing-active-fact embeddings on the same thought, supersede on cosine >some-threshold even when (S, P, O) doesn't match. Most correct, more code, depends on fact embeddings shipping (see above) first.
  - **`engram reflect --rerun --mode replace`** — flag that supersedes *all* existing-active-facts-on-this-thought before inserting v3 outputs. Simpler, more aggressive; risks losing real facts to sampling variance.

- **Three-band confidence routing with `facts.flagged` column.** M2 Phase D deferral. Migration adds `facts.flagged BOOLEAN NOT NULL DEFAULT FALSE`. Routing changes from current two-band to design-doc-§10's original three-band:
  - confidence < `review_queue_below` → `facts_review_queue` (current)
  - `review_queue_below` ≤ confidence < `min_confidence_to_store` → `facts` with `flagged = true` (NEW)
  - confidence ≥ `min_confidence_to_store` → `facts` with `flagged = false` (current normal commit)
  `search_facts` and `get_thought` surface `flagged` so agents can downweight or filter.

- **Persist `n_extractor_failures` on `reflector_runs`.** Observability gap noted during M2 dogfood: `ReflectorReport` carries the per-thought-failure count but the persisted `reflector_runs` row only stores n_thoughts_processed / n_facts_committed / n_review_queue / error. Result: operator can't tell from the runs table alone whether "0 facts" means "no facts to find" or "extractor unreachable for every call." One-column migration; trivial reflector.rs update.

## Out of scope (deferred to which milestone)

- Artifact-chunk search → **M4**
- Personalization / learned-rank → indefinitely (post-M5; probably never for single-user)
- Auth / observability formalization (Prometheus, audit log) → **M5**
- Eval-suite formalization (capture-recall, cross-model, LongMemEval-style fixtures) → **M5**; the M3 A/B harness is intentionally small and ad-hoc
- Subtractive-via-deletion on rerun (vs. supersession) → indefinitely; supersession preserves the audit trail, deletion doesn't
- Reranker model bake-off (Qwen3-Reranker, Cohere, etc.) → **M5** eval-suite; M3 picks the obvious default (BGE-reranker-v2-m3)

## Schema impact

- ✅ Migration 0003 (already landed): `thoughts.retracted_at` + `retracted_reason` + active-thoughts partial index.
- New migrations expected during M3:
  - `0004_facts_flagged.sql` — adds `facts.flagged BOOLEAN NOT NULL DEFAULT FALSE` for three-band routing.
  - `0005_reflector_runs_failures.sql` — adds `reflector_runs.n_extractor_failures INT NOT NULL DEFAULT 0` for the observability gap.
- **No** migration for fact embeddings — `embeddings.target_kind` already includes `'fact'` (migration 0001, M1 schema).

## MCP surface delta

- ✅ `retract_thought(thought_id, reason?)` — already shipped 2026-05-13.
- `search_thoughts(..., rerank?: bool, candidate_pool?: int)` — both fields optional with defaults; existing M1/M2 callers continue to work unchanged.
- `search_facts(..., rerank?: bool, candidate_pool?: int)` — same shape; gains a *real* vector leg in addition to rerank.
- Results from `search_facts` and `get_thought.linked_facts` gain a `flagged: bool` field per fact (default false for v1/v2 facts produced before 0004).

## Crate structure delta

- **`engram-embed`** (most likely) gains a `Reranker` trait and a `TeiReranker` implementation. Alternative: a separate `engram-rerank` crate. To be decided in M3 planning based on whether reranker shares HTTP-client infrastructure with the embedder.
- **`engram-extract`** bumps `OpenAICompatibleConfig::model_version` 2 → 3 when v3 prompt ships.
- **`engram-mcp`** updates the two search tool handlers to call the reranker after RRF fusion; reflector's routing logic gains the three-band case.
- **`engram-storage`** gains `search_facts_vector_knn` (the dual of `search_vector_knn` for thoughts) and either a paraphrase-similarity helper or a flag-driven full-replace path on rerun.

## Dependencies

- **Prior milestones:** M1 (search), M2 (`search_facts`, reflector, async-embed seam).
- **External services:** TEI configured with a rerank-task model loaded. May be the same TEI instance as the embedder if running in multi-task mode, or a second instance.
- **Dogfood data:** the v3 prompt-revision negative examples are gifts from the M2 dogfood week. The decision to ship v3 should happen *after* the dogfood week produces enough signal to know whether v2's existing guidance was directionally right or off-target.

## Success criteria

1. **A/B quality:** on a fixture set of ~50 query/expected-result pairs (drawn from the operator's actual captured thoughts), reranked nDCG@10 is materially higher than RRF-only nDCG@10. "Materially" = a difference the operator can feel in daily use; we'll define a numerical threshold during M3 planning.
2. **Latency:** rerank stage adds < 200 ms P95 to search latency on the operator's hardware (CPU TEI on the 9800X3D or GPU TEI on the 3090 — depends on deployment choice).
3. **Backward-compatible default:** clients calling `search_thoughts` with no rerank parameter get reranked results by default; existing M1/M2 client code continues to work.
4. **Fact embeddings + vector leg:** `search_facts` returns results from vector similarity *in addition to* trigram, fused via RRF, with `vector_search_available: bool` on the response (mirroring `search_thoughts`).
5. **Three-band routing:** dogfood evidence shows facts in the "flagged but committed" middle band are a meaningfully different population from both the review queue and the auto-commit set (i.e., the band carries information). If not, three-band is over-engineered and we collapse back to two.
6. **`--rerun` deduplication:** a single rerun on the same source thoughts under the same extractor model_version produces zero new active fact rows (currently produces near-duplicates because of paraphrase blindness).
7. **Operator dogfood:** the operator runs M3 for at least a week and reports whether rerank "feels worth the latency." If no, the default flips to `rerank: false` and we re-evaluate. Same dogfood pass evaluates whether v3 prompt + `extract` flag actually reduced the trivia-fact rate.

## Open questions

### Retrieval

- **TEI deployment shape.** One TEI instance running both embed and rerank tasks, or two instances (one per task). Hinges on TEI's multi-task support maturity at the time of implementation.
- **Default candidate pool.** 50 is a reasonable default; should it be 100 for quality, 25 for latency? Per-tool defaults?
- **Rerank cutoff.** If reranker confidence is below some threshold across all candidates, should we report "low confidence overall"? Or just always return top-N?
- **Default-on vs. default-off.** The operator's preference matters here; default-on is the recommendation, but default-off may be safer initially.
- **Reranker model choice.** BGE-reranker-v2-m3 is the obvious default; bake-off against alternatives is an M5 eval-suite job, not M3.

### Pipeline quality

- **v3 prompt revision timing.** Ship now or wait for the rest of the M2 dogfood week? Current call (2026-05-13): wait. Revisit after the dogfood log accumulates enough additional examples — v2 may catch noise patterns we haven't seen yet, or fail in ways the listed phrasings don't cover.
- **Capture-time `extract` flag vocabulary.** `"durable-only" | "none" | "all"` is one shape; an `extract_section: "finding"` reference into a structured-thought-content convention is another. UX decision worth a small consult with how the operator actually wants to mark mixed-content captures.
- **Paraphrase-aware merge strategy.** Embedding-similarity at merge time (more correct, more code, depends on fact embeddings landing first) vs. a `--mode replace` rerun flag (simpler, more aggressive, risks losing real facts to sampling variance). The dogfood near-dup case argues for the former.
- **Three-band thresholds.** Defaults are `review_queue_below = 0.7` and `min_confidence_to_store = 0.85` per the original design-doc §10 framing. Dogfood may move these.

### Scope semantics (carried forward from M1)

- **Scope prefix filtering.** Today `scope` is exact-match only (confirmed empirically on 2026-05-12 during the M1 smoke test). The dotted-scope convention in the design doc is purely human-readable — `work.tcgplayer.platform.pricing` is not findable when filtering by `work.tcgplayer`. The operator has adopted a "flat-and-few" scope convention to live with the constraint. Open: after a week+ of M1+M2 dogfood, is the lack of prefix filtering actually painful? If yes, add `WHERE scope = $1 OR scope LIKE $1 || '.%'` to the storage layer (one-line change, no schema impact). If no — i.e. the discipline of flat scopes is doing useful work — leave it alone. Decide with dogfood evidence, not speculation.

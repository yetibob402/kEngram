# kEngram Architecture & Design Review

**Date:** 2026-06-09
**Reviewed at:** commit `b4185bd` (main, clean tree)
**Reviewer:** Claude Code (Fable 5), commissioned by the operator
**Scope:** Full-system architecture/design review against four operator-stated criteria:

- **(A) Fidelity & integrity** — does the system protect the *accuracy and fidelity* of stored information? (Operational protections — backup, network hardening — are explicitly out of scope; the operator has those covered. Security items appear only where they threaten data integrity.)
- **(B) Retrieval** — accurate and consistent retrieval?
- **(C) Metadata** — the right metadata model?
- **(D) Small-LLM utilization** — given the 100%-local constraint, is the system extracting the best possible accuracy and utility from small local models?

**Method.** Three parallel code-exploration passes over all eight crates (~21.6k lines of Rust), DESIGN.md, the milestone docs, and `docs/tagger-improvements.md`; every High/Medium finding below was then hand-verified against the exact code path before being asserted. `cargo test --workspace` run against the local Postgres: **473 passed, 0 failed, 2 ignored** (the ignored pair are the `--features integration` kind-stability diagnostics). Findings are labeled **Confirmed** (code read, file:line cited) or **Likely** (consistent evidence, not exhaustively traced).

---

## 1. Verdict summary

| Criterion | Verdict |
|---|---|
| (A) Fidelity & integrity | **Strong.** Immutability is real and enforced by construction (only two UPDATE paths exist on `thoughts`, neither touches content). The notable semantic edge is content-only dedup silently discarding the duplicate capture's scope/source/metadata (A1). |
| (B) Retrieval | **Sound design, unmeasured quality.** The hybrid RRF + rerank pipeline is correctly built and degrades gracefully, but every quality-bearing constant (RRF k, per-leg K, trigram threshold, recency half-life, candidate pool) is unvalidated — the M7 eval suite is the single highest-leverage piece of unbuilt work in the system (B1). Several degradation modes are silent-but-flagged (B2–B4). |
| (C) Metadata | **Right shape, thin query surface.** The M4 collapse to tags-as-advisory-sidecar was the correct architecture call and the corpus evidence backs it. The gaps are missing search dimensions (time-range, source), not schema (C1). |
| (D) Small-LLM utilization | **Exceptionally disciplined; two cheap wins left on the table.** Sixteen documented prompt iterations, deterministic post-filters, provenance binding, and an honest record of structural ceilings. The cheapest unrealized gains: a deterministic entity surface-presence post-check (D1) and ratifying the default tagger model from the project's own sweep data (D2). |

The overall architecture — immutable raw thoughts, recomputable derived signals, closed vocabularies where small models are weak, deterministic filters where prompts hit ceilings — is coherent and the codebase matches the design doc unusually closely. Nothing found rises to "data is being lost or corrupted today." The findings below are ranked edges, asymmetries, and unbuilt verification.

---

## 2. (A) Fidelity & integrity of stored information

### A1 — Dedup silently discards the duplicate capture's scope, source, and metadata — **Medium, Confirmed**

The content fingerprint is SHA-256 of `content` alone (`crates/kengram-mcp/src/capture.rs:58-62,95`). On conflict, `insert_thought` returns the pre-existing row's id (`crates/kengram-storage/src/lib.rs:105-155`) and the new capture's `scope`, `source`, and `metadata` are **dropped without trace**. The caller sees `is_duplicate: true` but is not told the existing thought lives in a *different scope* than the one they asked for.

Concrete failure: an agent captures "Prefer RRF over weighted blends" into `project.kengram`; months later another agent captures the identical sentence into `rjf.tech`. The second capture no-ops, the response looks successful, and the thought remains invisible to `scope_prefix: "rjf."` searches. The agent believes it persisted something it didn't (in the scope sense). The same applies to metadata — a second capture's `session_id` is silently lost.

This may well be intended ("same bytes = same thought, scopes are labels not containers"), but it is currently neither documented in SERVER_INSTRUCTIONS nor detectable by the caller.

**Recommendation (design question for the operator, then a small change):**
1. Decide the semantics: content-global dedup (status quo) vs. scope-qualified fingerprint (`sha256(scope ‖ content)`). The latter is a migration + backfill; the former is fine *if disclosed*.
2. Either way, cheap fix: include the existing thought's `scope` (and optionally `created_at`) in the duplicate-path `capture` response so the agent can detect the mismatch and react (e.g., re-capture with a scope-distinguishing prefix).

### A2 — Content immutability holds by construction — **Positive, Confirmed**

Exhaustive audit of mutations on `thoughts`: exactly two UPDATE statements exist in the storage crate — `update_thought_tags` (tags + provenance triplet only, `lib.rs:697-722`) and `retract_thought` (trust-state columns only, guarded `WHERE retracted_at IS NULL` so a retraction can't be overwritten or reversed, `lib.rs:1452-1472`). The only `DELETE FROM thoughts` in the tree is inside a cascade test (`lib.rs:3116-3132`). Migrations 0006/0011 touched only the `tags` JSONB and are recorded in `migration_audit`. Content round-trips byte-exact: no trimming or normalization anywhere on the capture path; the 1 MiB cap **rejects** rather than truncates (`capture.rs:83-91`), pinned by tests.

### A3 — Permanently failed tag jobs are dropped with no automatic recovery — **Medium, Confirmed**

After `MAX_TAG_ATTEMPTS = 5` (or any non-transient tagger error), the tag drainer deletes the queue row and moves on (`crates/kengram-mcp/src/drain.rs:154-190`). The thought keeps `tags = '{}'` forever unless the operator notices. Mitigations that already exist: `kengram stats` surfaces the untagged count (`kengram-cli/src/main.rs:1108`), and `kengram tag` (no flags) re-walks `tags_extractor_version IS NULL` rows (`lib.rs:877-900`) — so the state is *visible* and *recoverable*, but only by manual operator action. The worker never self-heals these, and a transient-but-long outage (vLLM down for an hour while a batch of captures lands) converts to permanent abandonment after 5 ticks ≈ 25 seconds at default settings.

**Recommendation:** add exponential backoff or a much higher transient cap (5 attempts × 5-second ticks is a ~25-second tolerance for an external service the design assumes is independently operated), and/or have the worker periodically re-enqueue untagged-but-live thoughts. An M7 metric (`untagged_thoughts` gauge) makes the drift observable.

Asymmetry note: the embed queue has the *opposite* policy — no attempts cap, retried forever (`drain.rs` embed path; row keeps `last_error`). That's the safer default for fidelity; the tag queue's give-up-fast policy is the outlier. Neither has backoff, so a poison head-of-queue job is re-tried every tick (`ORDER BY enqueued_at ASC`, `lib.rs:577`) — harmless at current scale, noisy in logs.

### A4 — Re-tag is a total, history-free overwrite; the snapshot safety net is opt-in — **Low, Confirmed**

By design (§10 of DESIGN.md) tags are advisory and overwritten. Since M4.1, however, `tag_filter` recall *does* depend on tags, so a regressive tagger pass (model swap, prompt regression) degrades retrieval for filtered queries corpus-wide with no rollback short of a DB restore. The `kengram tag --snapshot` pre-retag snapshot (`kengram-cli/src/main.rs:780-806`) is exactly the right tool and costs almost nothing.

**Recommendation:** make the snapshot the default on `--rerun`/`--force` (write to a configured directory, keep last N), with `--no-snapshot` as the opt-out. One-line behavioral change, closes the only practical "derived signal destroys queryability" path.

### A5 — Unauthenticated write surface is an integrity exposure, not just an ops one — **Low (today), Confirmed**

Noted briefly per the review's scoping: there is no auth of any kind on the MCP surface today (router wiring `kengram-cli/src/main.rs:423-438`; Tier 2 is unshipped M7). The integrity-relevant part: `capture`, `retract_thought`, and `link_thoughts` are **writes** — anything that can reach the port can pollute the corpus or retract true thoughts, and retraction, while soft, silently removes content from every retrieval path. On Tier 0/1 the network perimeter is the control and that's acceptable; this becomes the gating item before any Tier 2 exposure (already planned). Related hygiene item while in this file: the DB URL (with password) is interpolated into connection-failure error contexts at `main.rs:382, 487, 503, 691` — `backup.rs` already has `redact_db_url()`; reuse it.

### A6 — `metadata` is unbounded at the MCP boundary — **Low, Confirmed**

`content` is capped at 1 MiB but `metadata` (JSONB) has no size or depth limit (`kengram-mcp/src/server.rs` capture args). A misbehaving agent can bloat rows arbitrarily. Cheap cap (e.g., 64 KiB serialized) for symmetry.

---

## 3. (B) Retrieval accuracy & consistency

### B1 — Retrieval quality is architecturally sound and empirically unmeasured — **High (as a gap), Confirmed**

The pipeline (verified end-to-end in `kengram-mcp/src/search.rs:166-291` and `kengram-core/src/search.rs`):

1. Vector kNN, top-50, HNSW partial index, scope + retraction filters in-query (`kengram-storage/src/lib.rs:1474-1530`)
2. Trigram leg, top-50, `similarity > 0.1` (`lib.rs:435-482`)
3. RRF fusion, k = 60 (`kengram-core/src/search.rs:26,101-141`)
4. Recency boost, multiplicative `0.5^(age/half_life)`, default 30 d (`core/search.rs:143-165`)
5. `tag_filter` (Rust-side `@>`-equivalent containment, deliberately pre-rerank, `mcp/search.rs:237-254`)
6. Cross-encoder rerank over top-`candidate_pool` (default 32), then `take(limit)`

Every constant in that chain — 50/leg, k=60, 0.1 trigram floor, 30-day half-life, pool 32 — is a reasonable literature default and **none has been validated against this corpus**. The project's own tagger story shows what happens when assumptions meet measurement (the entire M4 collapse came from dogfood). Retrieval has had no equivalent. The M7 eval suite (capture-recall, cross-model consistency, LongMemEval-style) is already planned; this review's strongest recommendation is to **treat it as the next milestone item, before further tagger iteration** — tagger quality is now well-characterized; retrieval quality is not. Also worth folding in: the missing edge-case tests noted during review (rerank-failure fallback, `limit`/`candidate_pool` interaction, scope_prefix vs scope, retracted-exclusion across legs — only some of these are pinned today).

### B2 — Degraded search looks like normal search — **Medium, Confirmed**

When the embedder is down, search soft-fails to trigram-only and sets `vector_search_available: false` (`mcp/search.rs:188-218`). Correct design — but on natural-language queries the trigram leg's 0.1 similarity floor gates out nearly everything, so the operator's agents receive a near-empty (or surface-match-only) result set whose only distress signal is one boolean most clients won't surface. Same shape for rerank failure (`rerank_used: false`, RRF order preserved, `search.rs:361-370`).

**Recommendation:** the M7 `/metrics` endpoint should count degraded searches; consider having SERVER_INSTRUCTIONS tell agents explicitly to check `vector_search_available` and re-query later rather than trusting a thin result.

### B3 — `limit` > `candidate_pool` silently shrinks results when rerank is on — **Medium, Confirmed**

`apply_rerank_to_thought_hits` truncates to the candidate pool (`mcp/search.rs:377`), so `limit: 100, candidate_pool: 32` returns at most 32 hits with no indication that 68 fused candidates were discarded. Defensible semantics ("only return what the reranker vetted") but undocumented and untested. Cheap fixes: clamp `candidate_pool = max(candidate_pool, limit)` in the orchestrator, or document the truncation in the tool schema. Related, worth a deliberate decision: because recency boost runs **before** rerank (`search.rs:230-235`), the cross-encoder's order fully overrides recency within the pool — recency currently only influences *pool membership*, not final order. That may be exactly what you want (rerank = final discriminator), but DESIGN.md §7 doesn't say so.

### B4 — Embedder/model_id misconfiguration degrades silently — **Low, Confirmed**

The HNSW partial index predicate is a literal model_id; a config/index mismatch makes the vector leg return empty with no error (only `vector_search_available` stays `true` in this case, since the query itself succeeds — arguably the most silent failure in the system). The drainer side is better: jobs with mismatched model_id error and stay queued with `last_error` (`drain.rs:107-113`). **Recommendation:** a startup invariant check in `serve`/`worker` — "configured `embedder.model_id` has a matching partial index and ≥1 embedding row (or empty corpus)" — would convert this whole class to a loud boot-time failure. ~20 lines.

### B5 — Long thoughts are embedded as silent prefixes — **Low today, Confirmed**

Capture accepts up to 1 MiB; BGE-M3 reads ~8k tokens; the drainer sends full content with no length check (`drain.rs:132-137`), so the backend truncates silently and only the prefix is semantically searchable. At the current corpus shape (~1.5 KB/thought, by design a high-signal-density store) this is theoretical — but it will bite the first time a transcript or document gets pasted into `capture`. **Recommendation:** log a warning (and/or set a metadata flag) when content exceeds the embedder's token window; it also acts as a tripwire for the §14 "are we storing transcripts?" question drifting in unannounced.

### B6 — Retraction is enforced in every retrieval path — **Positive, Confirmed**

Verified in all four read paths: trigram (`lib.rs:452`), vector kNN (`lib.rs:1498`), recent (`lib.rs:349`), list_scopes (`lib.rs:410`), plus the backfill/re-tag walkers; `get_thought` deliberately remains the audit path. Dedup correctness also verified: the `ON CONFLICT DO NOTHING` + SELECT-by-fingerprint fallback handles concurrent identical captures cleanly, and crash-replay in the embed drainer is idempotent (`ON CONFLICT` on the embeddings insert).

---

## 4. (C) Metadata model

### C1 — Missing search dimensions: time-range and source — **Medium, Confirmed**

`search_thoughts` accepts query/scope/scope_prefix/limit/recency/rerank/candidate_pool/tag_filter — there is **no `created_before`/`created_after` and no `source` filter** (`mcp/search.rs:36-63`), and `recent_thoughts` has no `tag_filter` (`RecentRequest`, `search.rs:97-103`). The recency *boost* cannot express "what did we decide in the last two weeks" or "only thoughts captured by claude-code" — both are natural queries for a store of record fed by many clients, and both are plain indexed columns already (`thoughts_scope_recent_idx`). `dates_mentioned` in tags partially covers prose-mentioned dates but not capture-time windows.

**Recommendation:** additive, low-risk: `created_after`/`created_before` (RFC3339) and `source` on `search_thoughts`; `tag_filter` on `recent_thoughts`. All compose via AND with existing filters. This is the highest-value metadata change available and requires no schema work.

### C2 — `metadata` opacity is fine until agents need it back — **Low / design question**

`metadata` is documented as opaque agent bookkeeping, and keeping it out of retrieval is a good tag-hygiene decision. But the recommended keys (`session_id`, `client_name`) imply queries the system can't answer ("everything from session X") except via psql. If that's intentional (psql is the admin interface), say so in SERVER_INSTRUCTIONS; if not, a `metadata_filter` (same JSONB containment machinery, GIN index on `metadata`) is mechanical. Surface as a design question rather than building speculatively.

### C3 — The tags schema itself is right — **Positive**

The six-field shape + closed 7-value `kind` + closed 7-relation link vocabulary is well matched to what the dogfood record shows the corpus actually contains. Tags-without-confidence is the correct call *given* tags are advisory and the server instructions tell consumers to treat entity filters as positive signal only — the documentation and the architecture agree. The provenance triplet (`tags_extractor_model/version/extracted_at`) plus provenance binding (D6) makes the metadata auditable. The `thought_links` design (polymorphic targets, soft-delete, `link_source` discriminator separating agent intent from tagger advice) is the strongest part of the metadata layer.

One forward-looking note: free-text `to_entity`/`to_person` link targets have no normalization ("Postgres" vs "postgres" are distinct nodes). Known and explicitly deferred (entity resolution, §6.6); it will surface as graph fragmentation when link counts grow — fine to defer, worth watching in `kengram stats`.

---

## 5. (D) Small-LLM utilization

What's already in place is genuinely strong, and worth ratifying explicitly: 16 versioned prompt iterations with failure-mode-driven changes; the v3→v4→v7 negative-example lesson (don't list forbidden phrases) applied twice and recorded as methodology; strict JSON-schema decoding; the v11 emission/normalization split that broke the vocab feedback loop; scope vocab as tie-breaker rather than gate; five deterministic post-filters in a single `finalize_tags` seam shared by worker and CLI (`kengram-mcp/src/finalize.rs:26-42`); provenance binding so version stamps can't drift from prompts (`kengram-extract/src/openai_compatible.rs:269-295`); a fixture/sweep harness (`tagger-sweep.sh`, `tagger_eval.rs`); and a documented decision to *stop* prompt-iterating at a structural ceiling (v8) rather than thrash. This is the discipline most teams claim and few practice.

The remaining gains, in value order:

### D1 — No deterministic surface-presence check on entities — **Medium, Confirmed**

The v16 prompt's load-bearing rule — every entity must appear in the prose; "do NOT infer entities from world knowledge" — is enforced only by the model re-reading its own output. The finalize seam (the natural home) has no such check. A deterministic post-filter — strip any entity whose text doesn't appear (case-insensitive substring) in the content — would have caught the documented `pg_trgm` hallucination outright, costs microseconds, and is exactly the "next lever is architectural" move the v8 entry prescribes. It will *not* fix the adjectival-phrase ceiling (`embedding-based` does appear in the prose) — that one stays an accepted ceiling or a model-swap question — but it closes the world-knowledge-hallucination class completely. Care needed for minor inflection (plural/possessive) — a token-overlap fallback handles it.

### D2 — Default tagger model contradicts the project's own sweep evidence — **Medium, Confirmed**

The shipped default is `qwen3-coder:30b` (`kengram-cli/src/config.rs:315-316`), while the sweep harness defaults to comparing `gemma3:12b qwen3.6:27b qwen3-coder:30b` and the iteration log's measured results (e.g., gemma3:12b at 24/25 on the fixture corpus, with the coder model implicated in earlier regressions) favor general-purpose instruct models for this NER/classification-shaped task. A code-specialized model is a surprising default for a discourse task. **Recommendation:** run the sweep across the current fixture set one more time and ratify whichever wins as the documented default — this is config-only and the harness already exists. (DESIGN.md §14 Q10's dense-vs-MoE question folds into the same sweep.)

### D3 — The deterministic sidecar is built but not in the eval loop — **Low, Confirmed**

`kengram-tagger-deterministic` (GLiNER + regex + bge-m3 prototypes; 20/25 vs the LLM's 24/25, with disjoint failure modes — all 5 LLM-only wins are discourse pragmatics) isn't exercised by `tagger-sweep.sh`. Two cheap uses: (1) add it as a sweep arm so regressions in either path are visible; (2) longer-term, its people/date output is a natural *cross-check* signal for the LLM's (disagreement → flag), which is the cheapest form of two-pass verification available — no second LLM call needed.

### D4 — Fixtures are hand-written and small; no corpus-derived golden set — **Medium, Confirmed**

Three fixture files (~25 cases) drive all tagger evaluation. They encode known failure modes well, but they can't detect *new* regression classes, and there is no retrieval-side eval at all (B1). **Recommendation:** sample 50–100 real thoughts from the live corpus, label ground truth once (a frontier model + operator spot-check is fine for a single pass — this is eval data, not runtime dependency, so it doesn't violate local-first), and make that the regression gate for every future `BUNDLED_TAGGER_VERSION` bump and model swap. This converts tagger iteration from probe-driven to measured.

### D5 — Minor sampling note — **Low**

Temperature 0.2 for a pure extraction/classification task is slightly loose; the kind-stability diagnostics showed within-model determinism at 0.2, but 0.0 is the conventional choice and removes one variable from cross-model comparisons. Worth one sweep arm to confirm no quality cost, then pin.

---

## 6. Prioritized recommendations

**Design questions to settle first (operator decisions, per the project's ambiguity convention):**
1. **A1** — Dedup semantics: is content-global dedup across scopes intended? Either scope-qualify the fingerprint or disclose + return the existing scope in the duplicate response.
2. **B3/recency** — Is "rerank fully overrides recency within the pool" the intended final-order semantics? Document in §7 either way.
3. **C2** — Is `metadata` permanently opaque (psql is the query path) or does it need a containment filter?

**Cheap, high-value code changes (each ≤ a day, most ≤ an hour):**
4. **D1** — Entity surface-presence post-filter in `finalize_tags`.
5. **B4** — Startup invariant check: configured `model_id` ↔ HNSW partial index.
6. **A4** — Make pre-retag snapshot default-on for destructive re-tags.
7. **B3** — Clamp or document `candidate_pool` vs `limit`; add the missing edge-case tests (rerank fallback, pool/limit, retracted-across-legs).
8. **A3** — Backoff / higher transient tolerance on the tag queue; symmetric attempts policy with the embed queue.
9. **A5** — Reuse `redact_db_url()` in the four connection-error contexts.
10. **C1** — `created_after`/`created_before` + `source` filters on `search_thoughts`; `tag_filter` on `recent_thoughts`.
11. **B5** — Warn when content exceeds the embedder token window.
12. **A6** — Size cap on `metadata`.

**The big one (already planned — raise its priority within M7):**
13. **B1/D4** — Build the eval suite next, and extend its planned scope: retrieval eval (capture-recall on a corpus-derived golden set, constant-sweep for k/threshold/half-life) **plus** the corpus-derived tagger golden set, **plus** D2's model-default ratification sweep. Every other quality question in this review (trigram floor, recency interaction, tagger model, temperature, dense-vs-MoE) becomes answerable the week this exists. Prometheus metrics (degraded-search counter, untagged gauge, queue ages) are the runtime complement.

**Explicitly fine as-is (reviewed, no action):** content immutability and retraction enforcement (A2, B6); the M4 tags-as-sidecar architecture and closed vocabularies (C3); the thought_links design; SQL injection posture (fully parameterized; the "dynamic" hybrid query is bound-parameter SQL, and `tag_filter` never reaches SQL); the v8 decision to accept the adjectival-entity ceiling rather than continue prompt-thrashing; no-confidence-on-tags; embeddings-as-separate-table model-swap design.

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

  **Also for v3: describe the JSON envelope shape in the prose.** Today the contract is split — prose carries the semantic shape of a fact (fields, confidence rubric, what to skip), while the `response_format: json_schema` channel carries the syntactic wrapper (`{"facts": [...]}` + per-item types + strict mode). Guided decoding enforces the schema at token-generation time, so the model physically can't return malformed JSON. But the prose-alone is incomplete: a reader inspecting the prompt sees the *semantic* contract, not the *syntactic* one, and if `response_format` ever silently no-ops on some future backend, the model would have to invent the envelope. v3 should restate the envelope shape in the prose so the prompt is self-describing and survives loss of schema enforcement.

- **Capture-time `extract` metadata flag.** Some thoughts mix durable findings with test narrative; the operator can't `retract_thought` them without losing the signal. A first-class `metadata.extract: "durable-only" | "none" | "all"` flag the reflector honors at extraction time. Pairs naturally with the metadata-schema-promotion conversation we punted on when adding `thoughts.retracted_at` — there's an emerging case for a small structured metadata vocabulary rather than fully free-form JSONB.

- **`--rerun` default-behavior change: from additive to merge.** M2 Phase D shipped additive-only rerun on the (since-disproved) theory that strict `(subject, predicate, object)` triple match would collapse paraphrases of existing claims into existing rows. By dogfood day 2 (2026-05-14, 16-hour observation window), every manual `engram reflect --rerun` was adding 3–9 paraphrase duplicates per non-retracted thought — `"work.tcgplayer as a catch-all for work"` vs. `"work.tcgplayer as a catch-all for work-related content"` etc. Triple match catches almost nothing because LLMs decompose the same claim into different (S, P, O) each run. The current default is **regarded as a bug**, not a kept invariant. Growth is per-rerun, not per-day (the cron path uses `run_reflector_once` / `find_unfacted_thoughts`, which can't produce duplicates by construction).

  **Concrete failure mode**: thought `a7b63f3b` carries v1 fact `c5799e68` (subject="current API surface", predicate="is", object="append-only", statement="The current API surface does not include update_thought or delete_thought functions; thoughts are append-only.") *and* v2 fact `1c4a53c1` (subject="thoughts in current API surface", predicate="are", object="append-only", **same statement verbatim**). Both active, no `superseded_at` on either. The existing dedup predicate keys on `(S, P, O)` via `find_matching_active_fact`, sees a mismatch, falls into the "no existing match → insert new" branch. The statement field — which IS identical — is never consulted. **No supersession, no audit linkage.** Two rows in the active set claiming the same thing in different syntactic clothes.

  The fix is small: the dedup predicate becomes **"statement matches verbatim OR (S, P, O) match"**, not just `(S, P, O)`. Statement-match is the catch-rate-dominant check in practice because LLM decomposition is more variable than its natural-language output. Even without paraphrase-similarity, exact-statement dedup would have caught every dogfood duplicate we've seen.

  M3 changes the default. Three implementation tiers, in order of completeness (operator's framing 2026-05-14):
  - **Crude (bleed-stopper, available as explicit `--mode replace` flag):** on rerun against a thought, supersede all existing active facts from the same `(source_thought_id, extractor_model)` and insert the fresh batch. Trivially correct; loses any operator-curated `correct_fact` state on that thought; gets duplicates to zero. Used when the operator explicitly wants to recreate from scratch (e.g. after a model upgrade where they trust the new extractor more than the old).
  - **Better (new M3 default for bare `--rerun`):** exact-`statement`-keyed dedup within `source_thought_id`. New extraction emits a statement already present → existing row stays (optionally bump `extractor_version` / `confidence`); emits a new statement → insert. Linear in fact count per thought; cheap; no embedding dependency. Preserves operator-curated retractions because the existing row (including its retracted-by-`correct_fact` state) is what gets kept.
  - **Best (M3 upgrade once fact embeddings land):** vector-similarity dedup on `statement` within `source_thought_id`, threshold-tunable. Collapses paraphrases across model versions and prompt revisions. Depends on fact embeddings landing first (M3 in-scope; see "Fact embeddings" above).

  **Default-flag taxonomy after M3:**
  - `engram reflect` — process unfacted thoughts only (unchanged).
  - `engram reflect --rerun` — re-evaluate facted thoughts with **merge** semantics (Better tier; later upgraded to Best when embeddings land).
  - `engram reflect --rerun --mode replace` — destructive recreate (Crude tier); opt-in.
  - `engram reflect --rerun --dry-run` — show what would change without committing. Useful with `--mode replace`.

- **Three-band confidence routing with `facts.flagged` column.** M2 Phase D deferral. Migration adds `facts.flagged BOOLEAN NOT NULL DEFAULT FALSE`. Routing changes from current two-band to design-doc-§10's original three-band:
  - confidence < `review_queue_below` → `facts_review_queue` (current)
  - `review_queue_below` ≤ confidence < `min_confidence_to_store` → `facts` with `flagged = true` (NEW)
  - confidence ≥ `min_confidence_to_store` → `facts` with `flagged = false` (current normal commit)
  `search_facts` and `get_thought` surface `flagged` so agents can downweight or filter.

- **Persist `n_extractor_failures` on `reflector_runs`.** Observability gap noted during M2 dogfood: `ReflectorReport` carries the per-thought-failure count but the persisted `reflector_runs` row only stores n_thoughts_processed / n_facts_committed / n_review_queue / error. Result: operator can't tell from the runs table alone whether "0 facts" means "no facts to find" or "extractor unreachable for every call." One-column migration; trivial reflector.rs update.

- **Subsumption-aware dedup.** Distinct from the paraphrase-aware merge above. A second M2 dogfood pass (2026-05-13, two thoughts about Ron's language preferences) produced 4 facts on a single thought where 2 were strict specializations of the other 2: "Ron does not like Python" and "Ron does not like Python for enterprise software" both got committed as separate active facts, ditto for JavaScript. Two atomic claims represented as four rows. Pattern: same `(subject, predicate)`, where one `object` is a substring or refinement of the other. The dedup logic needs to either prefer the more-specific row (drop the general) or the more-general row (drop the specific) — operator-policy call, probably exposed as a knob. Different from paraphrase-merge: this one's structural (subset relation), the other's semantic (similarity).

- **Structured relations in the output schema.** Same dogfood pass: a source thought stating "Rust > Go for me; Go is the next choice when Rust isn't available" produced two separate facts ("Ron prefers Rust for software projects" and "Go is the next choice when Rust is unavailable") with no explicit ranking between them. The current schema (`statement, subject?, predicate?, object?, confidence`) has no way to express "Ron prefers Rust over Go" as one structured claim, so the ranking only exists by inference. Two paths to evaluate during M3 planning: (a) extend the response schema with an optional `compared_to` field for relative claims, or (b) prompt-level guidance that requires the model to flatten "A > B" into a single fact with `subject=Ron`, `predicate="prefers over"`, `object="Rust over Go"` (or similar). The former is cleaner structurally; the latter avoids touching the schema.

- **Trigram search should index the (S, P, O) fields, not just `statement`.** Surfaced in the same pass: a fact with `subject=Ron, predicate=is the next choice when Rust is not available or appropriate, object=Go` but `statement="When Rust is not available or appropriate, Go is the next choice."` — the subject "Ron" appears only in the triple, not the statement. So `search_facts("Ron Go")` doesn't match this fact via the trigram leg even though semantically it's about exactly that. Two paths: (a) update `search_facts_trigram` to compute similarity across `statement || ' ' || COALESCE(subject,'') || ' ' || COALESCE(predicate,'') || ' ' || COALESCE(object,'')` (small storage-layer change, ~5 lines), or (b) ensure the prompt always restates the subject in the statement (extractor-side, fragile). Probably (a) is the right move — defensive, no LLM dependency, no schema change.

- **Confidence over-anchors at the v2 default of 0.85.** Subsequent dogfood pass produced 6 facts at uniformly 0.85 — the v2 prompt's "default to 0.85 unless the rubric says otherwise" instruction is being interpreted as "always 0.85." Hedged conditional claims ("when Rust is unavailable, Go is the next choice") got the same confidence as direct declarative ones ("Ron prefers Rust") despite the structural difference. v3 prompt-revision territory; possibly also a "require justification for non-default confidence" approach to force the model to think about it. Goes in the same v3 ship as the episodic anti-examples.

- **Per-claim retraction durability across re-extraction** *(added 2026-05-14 from 16-hour drift observation)*. `correct_fact` currently retracts a *row* (fact_id), not a *claim*. A subsequent `engram reflect --rerun` produces new fact_ids carrying paraphrases of the retracted claim, and the retraction silently doesn't propagate — operator-side curation work is undone on the next extraction cycle. Concrete example: four trivia facts manually retracted via `correct_fact` on thought `a7b63f3b` ("A search was conducted with scope=X", etc.) re-appeared as new fact_ids `4da87370` and `b1e2ddf2` carrying functionally the same claims after a subsequent `--rerun`. This is the worst failure mode for a trust system — `correct_fact` looks correct in the moment, fails silently later. Two architectural fixes to evaluate during M3 planning:
  - **Negative-claim list keyed by `(source_thought_id, statement-or-embedding)`.** On `correct_fact`-without-replacement, record the retracted statement in a new `retracted_claims` table (or a column on a new shape); the extractor consults this list and skips matching outputs at insert time.
  - **Inherit retraction state at insert time.** On every `insert_fact`, look up whether a row exists for `(source_thought_id, statement)` with `superseded_at` set — if yes, the new row inherits the supersession (effectively, retractions are sticky-by-claim, not sticky-by-id). Composes naturally with the exact-statement-keyed dedup option above.
  Either path closes the trust loop. The second composes with Finding 1 above into a single fix — claim-keyed identity becomes the dedup invariant *and* the retraction-durability mechanism, one piece of work.

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
7. **Subsumption dedup:** for the M2-dogfood "Python / Python for enterprise software" case, only one of the two facts is active after extraction (the operator-policy-chosen general or specific form).
8. **(S, P, O) lexical reach:** `search_facts("Ron Go")` returns the fact whose subject is Ron and object is Go even when the statement text doesn't mention Ron — the trigram leg consults all four fields.
9. **Per-claim retraction is durable across re-extraction:** a fact retracted via `correct_fact` does not reappear as a new fact_id carrying the same claim on the next `engram reflect --rerun`. Operator-curated state survives extractor cycles.
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
- **Subsumption dedup policy.** When two facts on the same thought share `(subject, predicate)` but one's `object` is a substring/refinement of the other, prefer the general or the specific? A reasonable default is "keep the more specific, drop the general" (the specific implies the general), but the operator may want it inverted depending on how they're querying. Probably a knob: `[extractor] subsumption_keep: "specific" | "general"`. Decision deferred to M3 planning.
- **Relational-claim representation.** Extend the response schema with an optional `compared_to: string` field for relative claims, or flatten via prompt guidance into a single fact like `subject=Ron, predicate="prefers over", object="Rust over Go"`. Schema extension is more honest; prompt-flattening avoids touching the wire format. Trade-off worth a small consult.
- **Three-band thresholds.** Defaults are `review_queue_below = 0.7` and `min_confidence_to_store = 0.85` per the original design-doc §10 framing. Dogfood may move these.

### Scope semantics (carried forward from M1)

- **Scope prefix filtering.** Today `scope` is exact-match only (confirmed empirically on 2026-05-12 during the M1 smoke test). The dotted-scope convention in the design doc is purely human-readable — `work.tcgplayer.platform.pricing` is not findable when filtering by `work.tcgplayer`. The operator has adopted a "flat-and-few" scope convention to live with the constraint. Open: after a week+ of M1+M2 dogfood, is the lack of prefix filtering actually painful? If yes, add `WHERE scope = $1 OR scope LIKE $1 || '.%'` to the storage layer (one-line change, no schema impact). If no — i.e. the discipline of flat scopes is doing useful work — leave it alone. Decide with dogfood evidence, not speculation.

# M3 — Search and extraction quality

## Goal

The original framing was narrow: improve retrieval with a cross-encoder rerank pass; "almost-right" results become "the right one is in the top three." That stays the headline. But M3's scope expanded mid-M2-dogfood (2026-05-13) as the facts pipeline produced enough real data to expose three classes of quality issue that aren't reranker-shaped: (a) the extractor produces noise on mixed-content thoughts; (b) `--rerun` is additive-only so paraphrase-level duplicates accumulate; (c) observability gaps make it hard to tell run-time symptoms apart from data-quality symptoms.

M3 absorbs those alongside the reranker. The milestone is now "everything that makes search and extraction trustworthy in dogfood," not just "rerank thoughts." The reranker is still the largest and most-impactful single change.

## In scope

### Retrieval quality (the original M3)

- **[L] Cross-encoder reranker** (default: BGE-reranker-v2-m3 or comparable) running in TEI's rerank-task mode — either as a second TEI instance or as a multi-task TEI deployment (TBD). The next three bullets fold into this same work increment.
- **Rerank stage in the search pipeline:** retrieve top-K (default 50) via RRF fusion, rerank with the cross-encoder to top-N (default 10).
- **Configurable per-call:** `rerank: bool` (default `true`), `candidate_pool: int` (default 50).
- **Both `search_thoughts` and `search_facts` (from M2) gain rerank support.**
- **[M] Fact embeddings.** Extend the async-embedding seam to enqueue `target_kind = 'fact'` rows in `pending_embeddings`. The embed-drainer learns to vectorize fact statements alongside thought content (no schema change — `embeddings.target_kind` already lists `'fact'`). The vector leg in `search_facts_trigram` becomes a real `search_facts_vector_knn` call; `search_facts` graduates from trigram-only-inside-RRF-shape to actual hybrid retrieval. This was the M2 Phase D simplification; M3 closes it.
- **[S] Eval-suite-style A/B comparison harness** — small, ad-hoc; the full eval suite lands at M5 — used to validate that rerank actually helps on a fixture corpus.

### Pulled forward from M2 (already landed)

- **✅ Thought retraction** (`retract_thought` MCP tool + `thoughts.retracted_at` + auto-supersede of derived facts). Landed 2026-05-13 in commit `636c910`. Was originally framed as M5 (`engram audit`); promoted to M3 starter when M2 dogfood demonstrated that retract-per-fact-via-correct_fact was a structural footgun. See design-doc revision history for the architectural rationale; m2-progress.md 2026-05-13 history for the dogfood data that motivated it.

### Pipeline quality (added 2026-05-13 from M2 dogfood)

- **[M] Extractor prompt v3 with anti-examples.** v2 added an episodic-skip clause but didn't prevent test-narrative noise leaking through when it was embedded in an otherwise-durable thought. Real offending phrasings from dogfood (carry these as negative examples in v3):
  - "A search was conducted with scope='X'"
  - "A probe thought was captured at scope X"
  - "The probe thought was NOT returned when searching with scope='X'"
  - "The behavior was re-verified on 2026-05-13"
  - "simd-json was 3.2x faster than serde_json for parsing 100MB of test JSON" *(2026-05-14 dogfood, fact_id `e69eff9b`)* — reads as durable, is actually a single benchmark result on one machine; the durable claim is "SIMD parsing tends to outperform scalar above ~1MB," not the specific 3.2x measurement.
  - "The benchmark was conducted on this hardware (M2 Pro, 16GB)" *(2026-05-14 dogfood, fact_id `ec465660`)* — session metadata, not a claim about the world.
  Plus a positive heuristic: *"would this claim still be useful to a reader six months from now, independent of when it was captured? If no, skip."* And explicit mixed-content guidance: *"if a thought contains both a finding (durable) and a test report (transient), extract only the finding."* Bump `model_version` 2 → 3. Decision deferred to *after* the M2 dogfood week so v2's signal stays clean; revisit then.

  **Also for v3: SPO-decomposition rules (added 2026-05-14 from the dogfood test-corpus pass).** The v2 prompt leaves the (subject, predicate, object) mapping under-specified for several common constructions, and `qwen3-coder:30b` consistently mishandles them. The `statement` field stays correct in all cases — the bug is silent at the prose level, visible only in the triple. Three failure families to cover with explicit rules + few-shot examples:
  - *S and O swapped on comparatives (HIGH — silent failure for any triple-keying consumer).* For comparative constructions ("A is more X than B" / "A is simpler than B" / "A outperforms B"), v2 systematically swaps subject and object: the triple comes back as S=B, P=…, O=A while the `statement` field is correct. The correct mapping is S=A, P="is more X than" / "outperforms", O=B. Affected dogfood fact_ids: `8da1fa45` ("Bazel … Make"), `64e26652` ("Bazel … Make"), `fb38bf42` ("Nix … Bazel"), `51744197` ("Nix … Make"), `e0238c2f` ("SSE … WebSockets"). Add a positive few-shot for the canonical form; ideally test the other comparative phrasings ("is faster than", "outperforms", "is bigger than") to confirm the rule generalizes beyond "more X than."
  - *Self-referential triples (subject == object).* The triple becomes semantically empty. Rule: **subject MUST NOT equal object**; if two distinct entities aren't recoverable, leave subject/object null and rely on the statement. Affected dogfood fact_ids: `39016e00` (S=O="SIMD-accelerated JSON parsing" — correct object would have been "scalar implementations"), `eeced4b3` (S=O="smaller documents"), `582b76e1` (S=O="SIMD setup overhead").
  - *Conditional-as-subject.* "If <condition>, A is the right choice" → S=A, P=…; the conditional belongs in the statement context, not the subject. v2 promotes the conditional clause to S, leaving A in the object slot. Affected dogfood fact_ids: `103f44c9` ("If you need sub-millisecond reads…" → Redis was demoted to O), `bea3629d` (PostgreSQL demoted), `e9032602` (Cassandra demoted, plus "in 2026" pulled into the subject as bonus drift).

  **Also for v3: describe the JSON envelope shape in the prose.** Today the contract is split — prose carries the semantic shape of a fact (fields, confidence rubric, what to skip), while the `response_format: json_schema` channel carries the syntactic wrapper (`{"facts": [...]}` + per-item types + strict mode). Guided decoding enforces the schema at token-generation time, so the model physically can't return malformed JSON. But the prose-alone is incomplete: a reader inspecting the prompt sees the *semantic* contract, not the *syntactic* one, and if `response_format` ever silently no-ops on some future backend, the model would have to invent the envelope. v3 should restate the envelope shape in the prose so the prompt is self-describing and survives loss of schema enforcement.

  **Done means**: re-extracting the 2026-05-14 dogfood corpus produces correct SPO triples on the 11 listed fact_ids (`8da1fa45`, `64e26652`, `fb38bf42`, `51744197`, `e0238c2f`, `39016e00`, `eeced4b3`, `582b76e1`, `103f44c9`, `bea3629d`, `e9032602`); the two new episodic-skip negatives (`e69eff9b`, `ec465660`) are no longer emitted; confidence varies across hedged vs declarative claims rather than uniformly anchoring at 0.85.

- **[S] Capture-time `extract` metadata flag.** Some thoughts mix durable findings with test narrative; the operator can't `retract_thought` them without losing the signal. A first-class `metadata.extract: "durable-only" | "none" | "all"` flag the reflector honors at extraction time. Pairs naturally with the metadata-schema-promotion conversation we punted on when adding `thoughts.retracted_at` — there's an emerging case for a small structured metadata vocabulary rather than fully free-form JSONB. **Done means**: a thought with `metadata.extract: "none"` produces zero facts via the reflector; `"durable-only"` applies the v3 mixed-content rule even when surrounding content is transient; `"all"` (or no flag — back-compat) extracts as today.

- **✅ `--rerun` default-behavior change: from additive to dedup-via-supersession.** *Shipped 2026-05-14.* M2 Phase D shipped additive-only rerun on the (since-disproved) theory that strict `(subject, predicate, object)` triple match would collapse paraphrases of existing claims into existing rows. By dogfood day 2 (2026-05-14, 16-hour observation window), every manual `engram reflect --rerun` was adding 3–9 paraphrase duplicates per non-retracted thought — `"work.tcgplayer as a catch-all for work"` vs. `"work.tcgplayer as a catch-all for work-related content"` etc. Triple match catches almost nothing because LLMs decompose the same claim into different (S, P, O) each run.

  **Concrete failure mode (now a regression test)**: thought `a7b63f3b` carries v1 fact `c5799e68` (subject="current API surface", predicate="is", object="append-only", statement="The current API surface does not include update_thought or delete_thought functions; thoughts are append-only.") *and* v2 fact `1c4a53c1` (subject="thoughts in current API surface", predicate="are", object="append-only", **same statement verbatim**). Both active, no `superseded_at` on either. The pre-M3 dedup predicate keyed on `(S, P, O)` via `find_matching_active_fact`, saw a mismatch, fell into the "no existing match → insert new" branch. The statement field — which IS identical — was never consulted. **No supersession, no audit linkage.** This scenario is now `rerun_supersedes_when_statement_matches_but_triple_differs` in `engram-mcp/src/reflect.rs`.

  **What shipped (design principle locked 2026-05-14 with operator):** facts table is append-only audit; supersession is the deprecation mechanism; claim transitions produce a new active row + `superseded_at`/`superseded_by` on the old one. The dedup predicate widens to **statement match OR (S, P, O) match** via `find_matching_active_facts` (note plural — multiple drift rows can match, all get folded). Three-case decision: 0 matches → insert; ≥1 match with a byte-identical row → no-op floor, drift rows fold into the byte-identical canonical; ≥1 match, none byte-identical → insert new as canonical, drift rows fold into it. `superseded_by` always points at the canonical row so audit consumers walk one chain. All writes per matched group are in a single transaction so a crash can't orphan rows.

  **Out of scope from this item (deferred):**
  - **Paraphrase-similarity dedup (different statements meaning the same thing).** Requires fact embeddings + a similarity threshold. The "Best tier" below; deferred until fact embeddings land.
  - **`--mode replace` and `--dry-run` flags.** The supersede-on-match path is the design's answer for re-extraction; the operator's "bleed-stopper" framing argued for an aggressive recreate mode, but the principle picked supersede-on-match instead. If a use case emerges (e.g. "I rewrote the prompt drastically and want a clean slate per thought") it gets its own plan.

  **Best (still M3, blocked on fact embeddings):** vector-similarity dedup on `statement` within `source_thought_id`, threshold-tunable. Collapses paraphrases that aren't byte-identical and don't share (S, P, O). Depends on fact embeddings landing first (M3 in-scope; see "Fact embeddings" above).

- **[M] Extend dedup-via-supersession to `run_reflector_once` (within-call).** *Added 2026-05-14 from dogfood test-corpus pass.* The 2026-05-14 dedup work above only fires on `engram reflect --rerun`. First-time extraction via the worker cron / `engram reflect` (no `--rerun`) still inserts every emitted fact without consulting existing rows, so the same-statement-different-triple duplicate class is reproduced **inside a single LLM extraction call**.

  **Concrete failure case (regression-test target):** thought `86c3392f` (test 4 in the 2026-05-14 dogfood corpus), facts `39016e00` and `bce8ac05` — byte-identical statements, different (and both broken) triples, both active, both share `source_run_id = 1200a19e-1e98-4d17-af74-8edc7247a752` (same extraction call). One run, one thought, one LLM response → two parallel-active rows for the same claim.

  **Fix shape:** apply the four-case decision tree (`0 matches` / `≥1 with byte-identical → no-op floor + fold` / `≥1 no byte-identical → insert new canonical + fold`) already shipped in `engram-mcp/src/reflect.rs::run_reflector_rerun` to `run_reflector_once` as well. The natural refactor is a private helper `commit_or_supersede(pool, run_id, options, thought, fact, extractor)` that both functions call per emitted fact. The `find_matching_active_facts` predicate is unchanged — on first-time extraction it returns empty for the thought's first fact and then sees subsequent inserts within the same call, exactly the within-call dedup we want. Closes the "rerun is a required cleanup pass after every first extraction" UX wart: rerun becomes a recovery tool, not a hygiene chore.

  **Regression test to port:** `engram-mcp/src/reflect.rs::tests::rerun_supersedes_when_statement_matches_but_triple_differs` becomes the template for an analogous `once_supersedes_when_statement_matches_but_triple_differs_within_call` covering the `run_reflector_once` path with a `FakeExtractor` configured to emit two same-statement-different-triple facts in a single `extract()` call.

  **Done means**: an initial-extraction call that emits two facts with byte-identical statements and different triples produces exactly one active row plus the other(s) superseded with `superseded_by` linkage to the canonical. The ported regression test passes. The dogfood pair `39016e00` / `bce8ac05` is no longer reproducible by repeating the test-4 capture.

- **[M] Three-band confidence routing with `facts.flagged` column.** M2 Phase D deferral. Migration adds `facts.flagged BOOLEAN NOT NULL DEFAULT FALSE`. Routing changes from current two-band to design-doc-§10's original three-band:
  - confidence < `review_queue_below` → `facts_review_queue` (current)
  - `review_queue_below` ≤ confidence < `min_confidence_to_store` → `facts` with `flagged = true` (NEW)
  - confidence ≥ `min_confidence_to_store` → `facts` with `flagged = false` (current normal commit)
  `search_facts` and `get_thought` surface `flagged` so agents can downweight or filter. **Done means**: dogfood evidence shows the middle band (`flagged = true`) is a meaningfully different population from both the review queue and the auto-commit set. If not, collapse back to two-band; the migration stays but `flagged` defaults false forever (kill-switch built in from the start).

- **[S] Persist `n_extractor_failures` on `reflector_runs`.** Observability gap noted during M2 dogfood: `ReflectorReport` carries the per-thought-failure count but the persisted `reflector_runs` row only stores n_thoughts_processed / n_facts_committed / n_review_queue / error. Result: operator can't tell from the runs table alone whether "0 facts" means "no facts to find" or "extractor unreachable for every call." One-column migration; trivial reflector.rs update. **Done means**: after a run with mixed successes and failures, `SELECT n_thoughts_processed, n_facts_committed, n_extractor_failures FROM reflector_runs ORDER BY started_at DESC LIMIT 1;` shows the split, and the operator can distinguish "no facts found" (0 failures) from "extractor was unreachable" (N failures, N processed) without reading the log.

- **[M] Subsumption-aware dedup.** Distinct from the paraphrase-aware merge above. A second M2 dogfood pass (2026-05-13, two thoughts about Ron's language preferences) produced 4 facts on a single thought where 2 were strict specializations of the other 2: "Ron does not like Python" and "Ron does not like Python for enterprise software" both got committed as separate active facts, ditto for JavaScript. Two atomic claims represented as four rows. Pattern: same `(subject, predicate)`, where one `object` is a substring or refinement of the other. The dedup logic needs to either prefer the more-specific row (drop the general) or the more-general row (drop the specific) — operator-policy call, probably exposed as a knob. Different from paraphrase-merge: this one's structural (subset relation), the other's semantic (similarity).

- **[M] Structured relations in the output schema.** Same dogfood pass: a source thought stating "Rust > Go for me; Go is the next choice when Rust isn't available" produced two separate facts ("Ron prefers Rust for software projects" and "Go is the next choice when Rust is unavailable") with no explicit ranking between them. The current schema (`statement, subject?, predicate?, object?, confidence`) has no way to express "Ron prefers Rust over Go" as one structured claim, so the ranking only exists by inference. Two paths to evaluate during M3 planning: (a) extend the response schema with an optional `compared_to` field for relative claims, or (b) prompt-level guidance that requires the model to flatten "A > B" into a single fact with `subject=Ron`, `predicate="prefers over"`, `object="Rust over Go"` (or similar). The former is cleaner structurally; the latter avoids touching the schema.

- **[S] Trigram search should index the (S, P, O) fields, not just `statement`.** Surfaced in the same pass: a fact with `subject=Ron, predicate=is the next choice when Rust is not available or appropriate, object=Go` but `statement="When Rust is not available or appropriate, Go is the next choice."` — the subject "Ron" appears only in the triple, not the statement. So `search_facts("Ron Go")` doesn't match this fact via the trigram leg even though semantically it's about exactly that. Two paths: (a) update `search_facts_trigram` to compute similarity across `statement || ' ' || COALESCE(subject,'') || ' ' || COALESCE(predicate,'') || ' ' || COALESCE(object,'')` (small storage-layer change, ~5 lines), or (b) ensure the prompt always restates the subject in the statement (extractor-side, fragile). Probably (a) is the right move — defensive, no LLM dependency, no schema change.

- **[rolls into v3 prompt above — no separate effort tag] Confidence over-anchors at the v2 default of 0.85.** Subsequent dogfood pass produced 6 facts at uniformly 0.85 — the v2 prompt's "default to 0.85 unless the rubric says otherwise" instruction is being interpreted as "always 0.85." Hedged conditional claims ("when Rust is unavailable, Go is the next choice") got the same confidence as direct declarative ones ("Ron prefers Rust") despite the structural difference. v3 prompt-revision territory; possibly also a "require justification for non-default confidence" approach to force the model to think about it. Goes in the same v3 ship as the episodic anti-examples. *Re-confirmed 2026-05-14 in the test-corpus pass: tests 2 and 3 produced uniform 0.85 across explicitly hedged claims ("usually the right choice", "most workloads", "up to a few thousand QPS", "remains competitive even in 2026"), no rubric-driven variation.*

- **[M] Per-claim retraction durability across re-extraction** *(added 2026-05-14 from 16-hour drift observation)*. `correct_fact` currently retracts a *row* (fact_id), not a *claim*. A subsequent `engram reflect --rerun` produces new fact_ids carrying paraphrases of the retracted claim, and the retraction silently doesn't propagate — operator-side curation work is undone on the next extraction cycle. Concrete example: four trivia facts manually retracted via `correct_fact` on thought `a7b63f3b` ("A search was conducted with scope=X", etc.) re-appeared as new fact_ids `4da87370` and `b1e2ddf2` carrying functionally the same claims after a subsequent `--rerun`. This is the worst failure mode for a trust system — `correct_fact` looks correct in the moment, fails silently later.

  **Chosen approach (2026-05-14): inherit retraction state at insert time.** On every `insert_fact`, consult `find_matching_active_facts` (the predicate widened in the dedup-via-supersession work shipped 2026-05-14) against rows where `superseded_at IS NOT NULL` as well — if any match, the new row inherits the supersession (`superseded_at = NOW()`, `superseded_by = <the previously-superseding canonical>` or NULL if the retraction had no replacement). Retractions become sticky-by-claim, not sticky-by-id. The same claim-keyed identity that powers `--rerun` dedup powers retraction durability — one storage-layer extension serves both. Composes naturally with the `commit_or_supersede` helper proposed for `run_reflector_once` (both paths route through the same predicate).

  **Rejected alternative (2026-05-14): negative-claim registry table.** Was: a new `retracted_claims` table keyed by `(source_thought_id, statement-or-embedding)`, populated on `correct_fact`-without-replacement, consulted by the extractor at insert time. Rejected because it adds a new table and a new query path for state that can be expressed via the `superseded_at` / `superseded_by` columns already in `facts`. The chosen path leverages existing schema.

## Phase plan

Decided 2026-05-14 in the pre-M3 design pass. M3 ships in four phases following the M1/M2 cadence; each phase is its own focused planning conversation that produces a phase-specific plan, lands code, and writes a History entry in `m3-progress.md` (created when Phase A starts). Effort tags (S/M/L) on each item above; this section assigns items to phases.

### Phase A — pipeline-quality fixes (M2 dogfood remediation)

In: `[M]` v3 prompt revision, `[M]` `commit_or_supersede` on `run_reflector_once`, `[S]` `extract` metadata flag, `[S]` `n_extractor_failures` persistence, `[S]` `(S, P, O)` trigram.

Rationale: each item is independently shippable; each addresses a dogfood pain point the operator hits daily; together they clean up corpus noise so the reranker (Phase B) is evaluating on cleaner data. Phase A leans heavily on the dogfood fact_ids enumerated above as regression-test targets — Phase A is "done" when those cases stop reproducing.

Estimated effort: ~1 week.

### Phase B — retrieval quality (the original M3 headline)

In: `[L]` cross-encoder reranker + rerank stage in `search_thoughts` / `search_facts` + per-call parameters, `[M]` fact embeddings, `[S]` A/B harness.

Rationale: largest single quality lever in the milestone. Depends on the Phase A items only in the soft sense that cleaner data makes A/B comparison more honest. New external service config (TEI rerank mode); first-touch HTTP integration on the rerank task. A/B harness is small but load-bearing — it's how we know rerank earns its latency.

Estimated effort: ~1.5 weeks (TEI deployment is the long pole).

### Phase C — deeper pipeline quality

In: `[M]` subsumption-aware dedup, `[M]` structured relations in output schema, `[M]` per-claim retraction durability (inherit-at-insert), `[M]` three-band confidence routing (with kill-switch).

Rationale: each needs more dogfood signal than Phase A's items to settle implementation details (subsumption-keep policy, schema-vs-prompt for relations, three-band threshold tuning). Several compose: per-claim retraction durability rides on the same predicate as `--rerun` dedup; subsumption + structured relations both involve the response schema. Phase C is the last code-shipping phase.

Estimated effort: ~1 week.

### Phase D — operator dogfood + close-out

No new code. Run M3 for ~1 week of real use. Evaluate against the milestone-level Success criteria below. Decide rerank-on-by-default (vs off) based on operator's daily-use feel. Write the closing `m3-progress.md` History entry. Mark M3 ✅ in `README.md`. Surface anything for M4/M5 that emerges from the dogfood pass.

Estimated effort: ~1 week of mostly-passive operator use; ~half a day of close-out edits.

### Sequencing notes

- Phases A and B can in principle run in parallel — they touch different code paths — but the operator's stated preference is focused conversations, so they ship sequentially.
- Phase C's per-claim retraction durability deliberately follows Phase A's `commit_or_supersede` so both can share the helper.
- Phase D is gated by Phase C finishing; the dogfood signal is only meaningful once all the code is shipped.

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
10. **Operator dogfood:** the operator runs M3 for at least a week and reports whether rerank "feels worth the latency." If no, the default flips to `rerank: false` and we re-evaluate. Same dogfood pass evaluates whether v3 prompt + `extract` flag actually reduced the trivia-fact rate.

## Open questions

### Planning meta — decisions captured 2026-05-14

The pre-M3 design pass settled the milestone-shape hinges:

- **Phasing**: M2-style A/B/C/D — see `## Phase plan` above.
- **MVP slice / sequencing**: Phase A (pipeline-quality fixes) first; Phase B (reranker + fact embeddings) second; Phase C (deeper pipeline quality) third; Phase D (dogfood close-out) last.
- **Per-claim retraction durability — architectural choice**: inherit retraction state at insert time (composes with the dedup-via-supersession work shipped 2026-05-14). Negative-claim registry table rejected. See the item body for the chosen-vs-rejected note.
- **Effort tags + item-level success criteria**: every in-scope item is now S/M/L tagged inline, and the 5 previously-uncovered items have "Done means…" lines.

The per-phase implementation questions below stay open and are settled when each phase is planned.

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

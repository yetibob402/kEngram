# M3 — Search and extraction quality

**Status: ✅ retrieval portion shipped (Phase A + Phase B steps 1-3 + Phase C). Extraction-side findings carried forward into M4 — see [`m4-collapse-to-thoughts.md`](./m4-collapse-to-thoughts.md).**

## Goal

The original framing was narrow: improve retrieval with a cross-encoder rerank pass; "almost-right" results become "the right one is in the top three." That stays the headline. But M3's scope expanded mid-M2-dogfood (2026-05-13) as the facts pipeline produced enough real data to expose three classes of quality issue that aren't reranker-shaped: (a) the extractor produces noise on mixed-content thoughts; (b) `--rerun` is additive-only so paraphrase-level duplicates accumulate; (c) observability gaps make it hard to tell run-time symptoms apart from data-quality symptoms.

M3 absorbs those alongside the reranker. The milestone is now "everything that makes search and extraction trustworthy in dogfood," not just "rerank thoughts." The reranker is still the largest and most-impactful single change.

## In scope

### Retrieval quality (the original M3)

- **✅ [L] Cross-encoder reranker** (default: BGE-reranker-v2-m3 or comparable) running in TEI's rerank-task mode — either as a second TEI instance or as a multi-task TEI deployment (TBD). The next three bullets fold into this same work increment. *Shipped 2026-05-15 as M3 Phase B step 2; TEI Docker container in `docker-compose.yml`, rerank-only mode (embeddings stay on Ollama).*

  **Regression target (from Phase B step 1 dogfood, 2026-05-15):** query `tooling for compiling codebases reproducibly` against the operator's live fact corpus currently ranks Redis above Bazel and misses the Nix-reproducibility facts entirely. After step 2 ships, this query should rank the Nix-reproducibility facts ("Nix is more reproducible than Bazel" / "Nix is more reproducible than Make") above Redis. The cross-encoder produces calibrated absolute relevance scores against the candidate pool — re-evaluating "compiling reproducibly" → "Nix is more reproducible…" semantically is exactly what it's designed for.
- **Rerank stage in the search pipeline:** retrieve top-K (default 50) via RRF fusion, rerank with the cross-encoder to top-N (default 10).
- **Configurable per-call:** `rerank: bool` (default `true`), `candidate_pool: int` (default 50).
- **Both `search_thoughts` and `search_facts` (from M2) gain rerank support.**
- **[M] Fact embeddings.** Extend the async-embedding seam to enqueue `target_kind = 'fact'` rows in `pending_embeddings`. The embed-drainer learns to vectorize fact statements alongside thought content (no schema change — `embeddings.target_kind` already lists `'fact'`). The vector leg in `search_facts_trigram` becomes a real `search_facts_vector_knn` call; `search_facts` graduates from trigram-only-inside-RRF-shape to actual hybrid retrieval. This was the M2 Phase D simplification; M3 closes it.
- **✅ [S] Eval-suite-style A/B comparison harness** *(shipped 2026-05-15, M3 Phase B step 3)* — small, ad-hoc; the full eval suite lands at M5. `kengram bench rerank --corpus <path>` reports nDCG@10 + MRR for RRF-only vs reranked on an operator-curated fixture; closes success criterion 1 with a concrete number rather than feel.

### Pulled forward from M2 (already landed)

- **✅ Thought retraction** (`retract_thought` MCP tool + `thoughts.retracted_at` + auto-supersede of derived facts). Landed 2026-05-13 in commit `636c910`. Was originally framed as M5 (`kengram audit`); promoted to M3 starter when M2 dogfood demonstrated that retract-per-fact-via-correct_fact was a structural footgun. See design-doc revision history for the architectural rationale; m2-progress.md 2026-05-13 history for the dogfood data that motivated it.

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

- **✅ `--rerun` default-behavior change: from additive to dedup-via-supersession.** *Shipped 2026-05-14.* M2 Phase D shipped additive-only rerun on the (since-disproved) theory that strict `(subject, predicate, object)` triple match would collapse paraphrases of existing claims into existing rows. By dogfood day 2 (2026-05-14, 16-hour observation window), every manual `kengram reflect --rerun` was adding 3–9 paraphrase duplicates per non-retracted thought — `"work.tcgplayer as a catch-all for work"` vs. `"work.tcgplayer as a catch-all for work-related content"` etc. Triple match catches almost nothing because LLMs decompose the same claim into different (S, P, O) each run.

  **Concrete failure mode (now a regression test)**: thought `a7b63f3b` carries v1 fact `c5799e68` (subject="current API surface", predicate="is", object="append-only", statement="The current API surface does not include update_thought or delete_thought functions; thoughts are append-only.") *and* v2 fact `1c4a53c1` (subject="thoughts in current API surface", predicate="are", object="append-only", **same statement verbatim**). Both active, no `superseded_at` on either. The pre-M3 dedup predicate keyed on `(S, P, O)` via `find_matching_active_fact`, saw a mismatch, fell into the "no existing match → insert new" branch. The statement field — which IS identical — was never consulted. **No supersession, no audit linkage.** This scenario is now `rerun_supersedes_when_statement_matches_but_triple_differs` in `kengram-mcp/src/reflect.rs`.

  **What shipped (design principle locked 2026-05-14 with operator):** facts table is append-only audit; supersession is the deprecation mechanism; claim transitions produce a new active row + `superseded_at`/`superseded_by` on the old one. The dedup predicate widens to **statement match OR (S, P, O) match** via `find_matching_active_facts` (note plural — multiple drift rows can match, all get folded). Three-case decision: 0 matches → insert; ≥1 match with a byte-identical row → no-op floor, drift rows fold into the byte-identical canonical; ≥1 match, none byte-identical → insert new as canonical, drift rows fold into it. `superseded_by` always points at the canonical row so audit consumers walk one chain. All writes per matched group are in a single transaction so a crash can't orphan rows.

  **Out of scope from this item (deferred):**
  - **Paraphrase-similarity dedup (different statements meaning the same thing).** Requires fact embeddings + a similarity threshold. The "Best tier" below; deferred until fact embeddings land.
  - **`--mode replace` and `--dry-run` flags.** The supersede-on-match path is the design's answer for re-extraction; the operator's "bleed-stopper" framing argued for an aggressive recreate mode, but the principle picked supersede-on-match instead. If a use case emerges (e.g. "I rewrote the prompt drastically and want a clean slate per thought") it gets its own plan.

  **Best (still M3, blocked on fact embeddings):** vector-similarity dedup on `statement` within `source_thought_id`, threshold-tunable. Collapses paraphrases that aren't byte-identical and don't share (S, P, O). Depends on fact embeddings landing first (M3 in-scope; see "Fact embeddings" above).

- **[M] Extend dedup-via-supersession to `run_reflector_once` (within-call).** *Added 2026-05-14 from dogfood test-corpus pass.* The 2026-05-14 dedup work above only fires on `kengram reflect --rerun`. First-time extraction via the worker cron / `kengram reflect` (no `--rerun`) still inserts every emitted fact without consulting existing rows, so the same-statement-different-triple duplicate class is reproduced **inside a single LLM extraction call**.

  **Concrete failure case (regression-test target):** thought `86c3392f` (test 4 in the 2026-05-14 dogfood corpus), facts `39016e00` and `bce8ac05` — byte-identical statements, different (and both broken) triples, both active, both share `source_run_id = 1200a19e-1e98-4d17-af74-8edc7247a752` (same extraction call). One run, one thought, one LLM response → two parallel-active rows for the same claim.

  **Fix shape:** apply the four-case decision tree (`0 matches` / `≥1 with byte-identical → no-op floor + fold` / `≥1 no byte-identical → insert new canonical + fold`) already shipped in `kengram-mcp/src/reflect.rs::run_reflector_rerun` to `run_reflector_once` as well. The natural refactor is a private helper `commit_or_supersede(pool, run_id, options, thought, fact, extractor)` that both functions call per emitted fact. The `find_matching_active_facts` predicate is unchanged — on first-time extraction it returns empty for the thought's first fact and then sees subsequent inserts within the same call, exactly the within-call dedup we want. Closes the "rerun is a required cleanup pass after every first extraction" UX wart: rerun becomes a recovery tool, not a hygiene chore.

  **Regression test to port:** `kengram-mcp/src/reflect.rs::tests::rerun_supersedes_when_statement_matches_but_triple_differs` becomes the template for an analogous `once_supersedes_when_statement_matches_but_triple_differs_within_call` covering the `run_reflector_once` path with a `FakeExtractor` configured to emit two same-statement-different-triple facts in a single `extract()` call.

  **Done means**: an initial-extraction call that emits two facts with byte-identical statements and different triples produces exactly one active row plus the other(s) superseded with `superseded_by` linkage to the canonical. The ported regression test passes. The dogfood pair `39016e00` / `bce8ac05` is no longer reproducible by repeating the test-4 capture.

- **✅ [M] Three-band confidence routing with `facts.flagged` column.** *Shipped 2026-05-15, M3 Phase C.* Migration adds `facts.flagged BOOLEAN NOT NULL DEFAULT FALSE`. Routing changes from current two-band to design-doc-§10's original three-band:
  - confidence < `review_queue_below` → `facts_review_queue` (current)
  - `review_queue_below` ≤ confidence < `min_confidence_to_store` → `facts` with `flagged = true` (NEW)
  - confidence ≥ `min_confidence_to_store` → `facts` with `flagged = false` (current normal commit)
  `search_facts` and `get_thought` surface `flagged` so agents can downweight or filter. **Done means**: dogfood evidence shows the middle band (`flagged = true`) is a meaningfully different population from both the review queue and the auto-commit set. If not, collapse back to two-band; the migration stays but `flagged` defaults false forever (kill-switch built in from the start).

- **[S] Persist `n_extractor_failures` on `reflector_runs`.** Observability gap noted during M2 dogfood: `ReflectorReport` carries the per-thought-failure count but the persisted `reflector_runs` row only stores n_thoughts_processed / n_facts_committed / n_review_queue / error. Result: operator can't tell from the runs table alone whether "0 facts" means "no facts to find" or "extractor unreachable for every call." One-column migration; trivial reflector.rs update. **Done means**: after a run with mixed successes and failures, `SELECT n_thoughts_processed, n_facts_committed, n_extractor_failures FROM reflector_runs ORDER BY started_at DESC LIMIT 1;` shows the split, and the operator can distinguish "no facts found" (0 failures) from "extractor was unreachable" (N failures, N processed) without reading the log.

- **✅ [S] Surface per-leg scores in search response shape** *(surfaced 2026-05-15 during Phase B step 1 dogfood; shipped 2026-05-15 alongside Phase B step 2)*. The current `score` field on `SearchHit` / `SearchFactHit` is the RRF-fused score, which is rank-based by construction (`Σ 1/(60 + rank_i)`) — top-of-both-legs caps at `2/61 ≈ 0.033`. Consumers building thresholding logic ("only show hits above similarity X") find this score uninformative because it's calibrated against rank order, not semantic distance. Raw per-leg scores would be more actionable.

  **Proposed shape** (additive; existing `score` stays for back-compat):
  ```json
  {
    "score": 0.0164,           // RRF fused (unchanged)
    "vector_score": 0.78,      // raw cosine similarity, or null when vector leg unavailable
    "trigram_score": 0.42      // raw word_similarity, or null when not matched
  }
  ```
  Implementation surface: thread per-leg scores through `Hit` / `FactHit` (kengram-storage) → `SearchHit` / `SearchFactHit` (kengram-mcp/search) → JSON serializer (kengram-mcp/server). The data already exists pre-fusion in the storage layer's per-leg `score` field; we just stop discarding it during RRF. Phase B step 2 will later add `rerank_score: Option<f32>` (calibrated absolute, from the cross-encoder) alongside these; consumers get the full picture across all three signals.

- **✅ [M] Quality-aware dedup for within-call duplicates** *(surfaced 2026-05-14 during Phase A close-out; shipped 2026-05-15, M3 Phase C).* The Phase A `commit_or_supersede` work correctly folds same-statement-different-triple emissions via supersession, but does so by **emission order** — the last-emitted fact wins as canonical regardless of correctness. Concrete failure case: thought `fe07320f` (Bazel/Make/Nix), the LLM emitted **two** facts with byte-identical statements "For build systems, Bazel is more powerful than Make but has a steeper learning curve":
  - fact `2472dc0c` — S=Bazel, P="is more powerful than", O=Make (**CORRECT comparative SPO**)
  - fact `876cdb9c` — S=Make, P="has a steeper learning curve than", O=Bazel (inverted, captures the *second* clause)

  Dedup folded them; `876cdb9c` won because it came second in the extractor output. The "correct" fact got superseded. Same pattern reproduces across multiple compound statements in the dogfood corpus. Root cause is twofold: (1) the v3 prompt's "one fact per claim" rule isn't preventing the LLM from emitting one statement-string with multiple SPO decompositions; (2) `commit_or_supersede` has no quality-aware preference between matches. Two architectural fixes worth evaluating:
  - **Within-call atomic-claim enforcement** (prompt + post-extraction validator): reject within-batch duplicates by statement before any dedup pass — keep the first emission per statement, drop the rest. Cheap. Doesn't solve "the LLM made the wrong SPO call on the first emission," but caps the impact.
  - **Quality-aware canonical selection in `commit_or_supersede`**: when multiple matches exist, prefer S != O facts over S == O facts; prefer triples where statement contains both subject and object as tokens. Composes naturally with the subsumption-aware dedup item below.

  Goes in the same Phase that addresses subsumption — both are quality-aware dedup work.

- **✅ [M] Subsumption-aware dedup.** *Shipped 2026-05-15, M3 Phase C.* Distinct from the paraphrase-aware merge above. A second M2 dogfood pass (2026-05-13, two thoughts about Ron's language preferences) produced 4 facts on a single thought where 2 were strict specializations of the other 2: "Ron does not like Python" and "Ron does not like Python for enterprise software" both got committed as separate active facts, ditto for JavaScript. Two atomic claims represented as four rows. Pattern: same `(subject, predicate)`, where one `object` is a substring or refinement of the other. The dedup logic needs to either prefer the more-specific row (drop the general) or the more-general row (drop the specific) — operator-policy call, probably exposed as a knob. Different from paraphrase-merge: this one's structural (subset relation), the other's semantic (similarity).

- **✅ [M] Structured relations in the output schema.** *Shipped 2026-05-15, M3 Phase C — prompt-only flattening; no JSON schema change. v4 prompt teaches the LLM to emit a single fact whose object names both sides of a relative claim. Schema-extension path (e.g. a `compared_to` column) remains deferred unless dogfood shows the prompt approach doesn't generalize.* Same dogfood pass: a source thought stating "Rust > Go for me; Go is the next choice when Rust isn't available" produced two separate facts ("Ron prefers Rust for software projects" and "Go is the next choice when Rust is unavailable") with no explicit ranking between them. The current schema (`statement, subject?, predicate?, object?, confidence`) has no way to express "Ron prefers Rust over Go" as one structured claim, so the ranking only exists by inference. Two paths to evaluate during M3 planning: (a) extend the response schema with an optional `compared_to` field for relative claims, or (b) prompt-level guidance that requires the model to flatten "A > B" into a single fact with `subject=Ron`, `predicate="prefers over"`, `object="Rust over Go"` (or similar). The former is cleaner structurally; the latter avoids touching the schema.

- **[S] Trigram search should index the (S, P, O) fields, not just `statement`.** Surfaced in the same pass: a fact with `subject=Ron, predicate=is the next choice when Rust is not available or appropriate, object=Go` but `statement="When Rust is not available or appropriate, Go is the next choice."` — the subject "Ron" appears only in the triple, not the statement. So `search_facts("Ron Go")` doesn't match this fact via the trigram leg even though semantically it's about exactly that. Two paths: (a) update `search_facts_trigram` to compute similarity across `statement || ' ' || COALESCE(subject,'') || ' ' || COALESCE(predicate,'') || ' ' || COALESCE(object,'')` (small storage-layer change, ~5 lines), or (b) ensure the prompt always restates the subject in the statement (extractor-side, fragile). Probably (a) is the right move — defensive, no LLM dependency, no schema change.

- **[rolls into v3 prompt above — no separate effort tag] Confidence over-anchors at the v2 default of 0.85.** Subsequent dogfood pass produced 6 facts at uniformly 0.85 — the v2 prompt's "default to 0.85 unless the rubric says otherwise" instruction is being interpreted as "always 0.85." Hedged conditional claims ("when Rust is unavailable, Go is the next choice") got the same confidence as direct declarative ones ("Ron prefers Rust") despite the structural difference. v3 prompt-revision territory; possibly also a "require justification for non-default confidence" approach to force the model to think about it. Goes in the same v3 ship as the episodic anti-examples. *Re-confirmed 2026-05-14 in the test-corpus pass: tests 2 and 3 produced uniform 0.85 across explicitly hedged claims ("usually the right choice", "most workloads", "up to a few thousand QPS", "remains competitive even in 2026"), no rubric-driven variation.*

- **✅ [M] Per-claim retraction durability across re-extraction** *(added 2026-05-14 from 16-hour drift observation; shipped 2026-05-15, M3 Phase C via inherit-at-insert).* `correct_fact` currently retracts a *row* (fact_id), not a *claim*. A subsequent `kengram reflect --rerun` produces new fact_ids carrying paraphrases of the retracted claim, and the retraction silently doesn't propagate — operator-side curation work is undone on the next extraction cycle. Concrete example: four trivia facts manually retracted via `correct_fact` on thought `a7b63f3b` ("A search was conducted with scope=X", etc.) re-appeared as new fact_ids `4da87370` and `b1e2ddf2` carrying functionally the same claims after a subsequent `--rerun`. This is the worst failure mode for a trust system — `correct_fact` looks correct in the moment, fails silently later.

  **Chosen approach (2026-05-14): inherit retraction state at insert time.** On every `insert_fact`, consult `find_matching_active_facts` (the predicate widened in the dedup-via-supersession work shipped 2026-05-14) against rows where `superseded_at IS NOT NULL` as well — if any match, the new row inherits the supersession (`superseded_at = NOW()`, `superseded_by = <the previously-superseding canonical>` or NULL if the retraction had no replacement). Retractions become sticky-by-claim, not sticky-by-id. The same claim-keyed identity that powers `--rerun` dedup powers retraction durability — one storage-layer extension serves both. Composes naturally with the `commit_or_supersede` helper proposed for `run_reflector_once` (both paths route through the same predicate).

  **Rejected alternative (2026-05-14): negative-claim registry table.** Was: a new `retracted_claims` table keyed by `(source_thought_id, statement-or-embedding)`, populated on `correct_fact`-without-replacement, consulted by the extractor at insert time. Rejected because it adds a new table and a new query path for state that can be expressed via the `superseded_at` / `superseded_by` columns already in `facts`. The chosen path leverages existing schema.

- **[S] Predicate slot doing double duty as relation + qualifier** *(surfaced 2026-05-15 during Phase C dogfood)*. The v4 prompt instructs the LLM to preserve qualifiers — and `qwen3-coder:30b` interprets that by stuffing them into the predicate slot rather than the statement. Real dogfood phrasings:
  - `(Cap'n Proto, "can outperform by an order of magnitude for very high-throughput RPC workloads compared to", Protobuf)` — 14-word predicate.
  - `(<format>, "offers zero-copy reads but at the cost of more rigid schema evolution compared to", <other>)` — 12-word predicate.

  The natural-language `statement` carries the full claim correctly, so nothing is lost — but the (S, P, O) triple loses its utility as a structured join/filter key. A consumer asking "what does Cap'n Proto outperform?" with `predicate ILIKE '%outperform%'` works; with `predicate = 'outperforms'` it doesn't. The trailing `compared to` suffix is the same artifact in another form — it signals to the consumer that the object is the comparand, but only if the consumer parses the predicate as natural language.

  Three paths, in order of cost:
  - **(a) Prompt-side brevity rule (cheapest)**: v4.1 adds an explicit instruction that predicates be short verb-phrase relation tags ('outperforms', 'is more reproducible than') — 1–3 words. Qualifiers ('by an order of magnitude', 'for high-throughput workloads') belong in `statement`, not `predicate`. Also explicitly forbid trailing `compared to`. `model_version` 4 → 5.
  - **(b) Schema extension**: add `predicate_modifier: Option<String>` (or `comparand: Option<String>` for the comparative-specific case). Migration + struct + serializer touchpoints; one more optional field for consumers to learn. Cleanest for structured consumers.
  - **(c) Multi-fact decomposition**: extractor emits the core triple plus a separate qualifier fact — `(Cap'n Proto, outperforms, Protobuf)` + `(Cap'n Proto, outperforms by, "order of magnitude for high-throughput RPC")`. Risks the "two facts for one claim" pattern Phase C's quality-aware pick was designed to fold; net negative unless tied to subsumption rules.

  Probably (a) first; revisit (b) if more dogfood examples cluster around genuinely-structured comparands rather than verbose-predicate stylistic choices. **Done means**: re-extracting the dogfood corpus produces predicates ≤ 3 words on at least 80% of comparative facts; no predicate ends in `compared to`; consumers can match comparative facts on `predicate = 'outperforms'` (or similar canonical relation tag) without ILIKE.

- **[M] Same-source-thought polarity-contradiction detection** *(surfaced 2026-05-16 during Phase D dogfood)*. M3 Phase C's claim-keyed dedup-via-supersession (statement-or-triple match) collapses *similarity* but not *contradiction*. Concrete dogfood case on thought `1802ff2c`: the v4 rerun emitted a correct fact (`c5fcfb8b`: "Long-polling is dated but still works behind restrictive corporate proxies where neither WebSockets nor SSE can establish a connection") alongside two pre-existing polarity-flipped hallucinations (`1e016df7`, `cf1cfe8b`) saying WebSockets/SSE *can* establish a connection in those environments. All three stayed active until manually retracted via `correct_fact` — because their statements don't match and their (S, P, O) triples don't match, the dedup predicate sees them as three distinct claims.

  Architecturally, `correct_fact` is row-local; even Phase C's retraction durability is claim-local. Neither catches "two active facts on the same thought disagree." Without contradiction detection, cleanup is Sisyphean against systematic prompt-level extractor bugs: every rerun can manufacture new fact_ids carrying the same wrong claim alongside a correct one, and the operator has to keep retracting them.

  **Implementation sketch** (cheap now that fact embeddings shipped in Phase B step 1):
  - On `commit_or_supersede` (and as a one-shot audit pass over the existing corpus), for each active fact on a given `source_thought_id`, find its embedding-neighbours among the *other* active facts on the same thought. Pairs that are within a tunable cosine-distance threshold (~0.85+ similarity) are candidate contradictions.
  - Polarity signal: cheapest first pass is lexical — detect asymmetric negation markers across the two statements ("not", "no", "cannot", "can't", "neither", antonym pairs). Lossy but catches the dogfood case (`c5fcfb8b` says "neither WebSockets nor SSE can"; the others say WebSockets/SSE *can*).
  - More accurate alternative: a small NLI (natural-language-inference) model classifies each candidate pair as entail/contradict/neutral. M5-eval-suite territory if the lexical pass turns out too lossy; M3 doesn't need it.
  - Surface via the existing `flagged: bool` column on `facts` (or, if `flagged` is already loaded with three-band-routing semantics, a new `contradicts_fact_id: Option<Uuid>` column on `facts` pointing at the contradicting peer). Decision deferred to planning.

  **Done means**: re-running the audit pass on the live corpus surfaces the `1802ff2c` triple (`c5fcfb8b` vs `1e016df7`/`cf1cfe8b`) as a contradiction pair, exposes it via `search_facts` / `get_thought` for operator review, and a subsequent rerun against the same prompt doesn't re-introduce the contradictions silently. Highest-impact safety feature on the backlog per the operator's 2026-05-16 read.

- **[S] Confidence policy on supersede with identical (statement, triple)** *(surfaced 2026-05-16 during Phase D dogfood)*. Concrete case: v3 fact `654816ed` ("Rust is more memory-safe than C", confidence 0.95) was superseded by v4 fact `39e3ce15` carrying the identical statement + triple but confidence 0.90. The supersede silently used `next` confidence; the historical 0.95 high-water mark was lost without explicit signal. Worth making the decision explicit.

  Three reasonable policies:
  - **(a) Status quo: take `next`** — current behavior. Confidence reflects the *current* model's assessment under the *current* prompt. Defensible when the v4 rubric narrows what "0.95" means relative to v3.
  - **(b) Take `max(prev, next)`** — preserve the highest confidence observed across reruns. Lossless; treats confidence as a high-water mark. Risk: a confident-but-wrong v3 row stays confident-but-wrong forever.
  - **(c) Take `next` but log on drop > 0.05** — keep behavior; add observability so operators can see "this rerun is downgrading historical claims" in the logs. Cheapest middle ground.

  Recommended path: **(c)** — change nothing in storage, add a `tracing::info!` in `commit_or_supersede`'s same-statement-same-triple branch when the new confidence is materially lower than the old. Composes with the existing run-level observability work (n_extractor_failures, etc.). If dogfood shows (b) is the right call, the policy switch is a one-line storage change later. **Done means**: a v3→v4 rerun that drops confidence on a same-statement-same-triple supersede emits an INFO log naming both facts and both confidences; the policy is documented in `m3-search-quality.md` and `DEVELOPMENT.md`.

- **[S] Reflect coverage observability — which thought_ids were actually re-extracted** *(surfaced 2026-05-16 during Phase D dogfood)*. The v4 rerun touched some thoughts (`c5617a9e`, `1802ff2c`) but not others (`22bccb3a` still at `extractor_version: 3`). The selection depends on `--scope`, `--since`, `--limit`, ordering inside `find_facted_thoughts`, and whether intervening runs supersede facts in ways that move thoughts in/out of the facted set. Today the `reflector_runs` row records counts (n_thoughts_processed / n_facts_committed / n_review_queue / n_extractor_failures) but not the *identities* of processed thoughts.

  Implication for any pre/post comparison (including the Phase B step 3 A/B harness): deltas conflate "prompt change effect" with "selection sample." If we want to A/B prompts honestly, we need to know which thought_ids appeared in the run's sample.

  **Implementation sketch**: new `reflector_run_thoughts (run_id UUID, thought_id UUID, PRIMARY KEY (run_id, thought_id))` table, written inside the per-thought loop in `run_reflector_once` / `run_reflector_rerun`. Or simpler: a `thought_ids: UUID[]` array column on `reflector_runs`. Array is denormalized but cheap to query (`unnest(thought_ids)`) and keeps observability in one table. **Done means**: after a `kengram reflect --rerun --since X`, a single SQL query against `reflector_runs` returns the list of thought_ids the run actually processed; the bench harness (or future eval-suite work) can correlate pre/post deltas to exactly that set.

- **[M] Triple coherence beyond lexical anchoring** *(surfaced 2026-05-16 during Phase D dogfood round 2)*. The v4 prompt's anchor rule ("subject AND object should appear in the statement") catches lexical-token mismatches but not semantic-relation mismatches. The dogfood case: source thought `8a533e15` (a ~700-char mission statement with 6 propositions) produced 5 faithful statements but 4-of-5 broken triples. Statements were paraphrase-correct (all confidence ≥ 0.90); triples were systematically inverted, partial, or drifted from what the statement says. Concrete examples (scope `kengram.m3.dogfood`):
  - `fact_id 66855ac2` — statement "The agent will consult the facts and thoughts in that scope, giving more weight to facts than thoughts" → triple `(agent, are given less weight than, thoughts)`. Subject wrong (should be `facts` or `thoughts`, not `agent`); verb agreement broken (singular `agent` + plural `are`); directionality *inverted from the source policy* (source says `facts > thoughts`; triple expresses `agent < thoughts`, which the source doesn't claim). The anchor check passes (`agent` and `thoughts` both appear in the statement) but the triple expresses a relationship the statement doesn't.
  - `fact_id 4e7a324f` — triple `(X, is the scope for, kengram.m3.dogfood)`. `kengram.m3.dogfood` IS the scope; the predicate is inverted.
  - `fact_id 4635bd9d` — triple `(scope, is used as a parameter for, kengram.m3.dogfood)`. Same shape: object is the value, not the parameterised entity.
  - `fact_id 1104fb5b` — triple `(agent, will be added by the agent, interesting thought)`. Passive-voice predicate makes subject and object semantically reversed (true subject is `interesting thought`).
  - `fact_id 4e4b08af` — partial: source statement names 3 test criteria, triple's object slot carries only one.

  Three reasonable fix paths:
  - **(a) Prompt-side coherence self-check (cheapest)**: v4 → v5 adds an explicit rule — "Before emitting a fact, read the (S, P, O) triple aloud as a sentence (`<subject> <predicate> <object>`). If the resulting sentence doesn't convey the same relationship as your statement, set subject/predicate/object to null and rely on the statement alone. Examples: 'agent are given less weight than thoughts' is broken (verb agreement + drifted relation) → emit null SPO." Lossy but cheap; relies on the model's self-correction.
  - **(b) Post-extraction validator with NLI**: a small natural-language-inference model classifies each `<S> <P> <O>` rendered sentence against the statement as entail / contradict / neutral. Triples in `contradict` or `neutral` get routed to `flagged = true` (or set to NULL) before the fact lands. More accurate, costs another model on the inference path. M5-eval-suite territory.
  - **(c) Two-stage extraction**: emit statements first as standalone string facts, then ask the extractor "for this statement, what's the cleanest (S, P, O)?" Slower (extra LLM call per fact) but separates statement-faithfulness from triple-coherence in the cost function.

  **Done means**: re-extracting thought `8a533e15` against a fixed prompt produces either correct triples on the 4 listed fact_ids, or null SPO on the ones the model can't construct coherently — but not "correct statement + 0.90 confidence + wrong triple" silently. Composes with the polarity-contradiction item from round 1: both want the model to self-check triple validity before emission, and both want NLI as the longer-term answer.

- **[S] Meta-policy / conditional clauses as first-class facts** *(surfaced 2026-05-16 during Phase D dogfood round 2)*. The source thought `8a533e15` contained the clause *"If unsure, the agent will ask the operator whether it should be stored"* — a meta-policy / agent-directive. The v4 extraction produced **no fact** for it. Likely cause: the v4 prompt's conditional-as-subject rule (added Phase A) and episodic-skip negatives push the model away from emitting conditional clauses, treating them like session-narrative ("if X happens, do Y").

  But: when an operator captures a thought that includes policy directives, those directives are durable signals the agent should remember. They're the opposite of session-narrative — they're rules that apply across sessions. v4 currently can't distinguish "if a search was conducted, return X" (transient, skip) from "if unsure, ask the operator" (durable agent-directive, extract).

  **Implementation sketch**: v5 prompt adds an explicit "agent directives are extractable" rule with positive examples. Possible shape: `subject = "the agent"`, `predicate = "should ask the operator"`, `object = "when unsure whether to store a captured thought"`. Or: the directive becomes a fact whose statement is the rule, and SPO is null (relying on the statement carries the meaning). **Done means**: re-extracting `8a533e15` emits a fact for the meta-policy clause; the prompt distinguishes meta-policy from session-narrative with at least one positive and one negative few-shot example.

- **[S] Confidence rubric should reflect triple coherence, not just statement faithfulness** *(surfaced 2026-05-16 during Phase D dogfood round 2)*. v4's confidence rubric bands by statement faithfulness ("0.95-1.00 = direct quotation; 0.90-0.95 = clean paraphrase; …"). It doesn't penalize triple incoherence. Dogfood case: `66855ac2` scored 0.90 (clean-paraphrase band) despite an inverted/drifted triple. Result: an agent reading `confidence ≥ 0.85` as "trustworthy" gets a wrong triple at full confidence.

  Three reasonable paths:
  - **(a) Add a triple-coherence sub-rule to the rubric**: "If the triple doesn't faithfully decompose the statement, drop confidence by one band (e.g., 0.90 → 0.80) AND/OR set SPO to null." Composes with the triple-coherence item above — same self-check, different consequence.
  - **(b) Two-axis confidence**: split into `statement_confidence` and `triple_confidence`. Cleaner; requires schema change (one new column).
  - **(c) Triple coherence enforced via fallback to null SPO**: when the model can't construct a coherent triple, it MUST emit SPO=null rather than guessing. Confidence stays single-axis; consumers reading null SPO know to fall back to the statement. This is the cheapest combine with item 1 (just one prompt-side rule does double duty).

  Recommended: **(c)** — pairs naturally with the triple-coherence self-check above. If dogfood after the v5 prompt still shows broken triples at high confidence, escalate to (a) or (b).

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
- ✅ Migration 0004 (M3 Phase A, 2026-05-14): `reflector_runs.n_extractor_failures INT NOT NULL DEFAULT 0` for the observability gap.
- ✅ Migration 0005 (M3 Phase C, 2026-05-15): `facts.flagged BOOLEAN NOT NULL DEFAULT FALSE` for three-band routing.
- **No** migration for fact embeddings — `embeddings.target_kind` already includes `'fact'` (migration 0001, M1 schema).
- **No** migration for structured relations — Phase C ships prompt-only flattening (no `compared_to` column).

## MCP surface delta

- ✅ `retract_thought(thought_id, reason?)` — already shipped 2026-05-13.
- `search_thoughts(..., rerank?: bool, candidate_pool?: int)` — both fields optional with defaults; existing M1/M2 callers continue to work unchanged.
- `search_facts(..., rerank?: bool, candidate_pool?: int)` — same shape; gains a *real* vector leg in addition to rerank.
- Results from `search_facts` and `get_thought.linked_facts` gain a `flagged: bool` field per fact (default false for v1/v2 facts produced before 0004).

## Crate structure delta

- **`kengram-embed`** (most likely) gains a `Reranker` trait and a `TeiReranker` implementation. Alternative: a separate `kengram-rerank` crate. To be decided in M3 planning based on whether reranker shares HTTP-client infrastructure with the embedder.
- **`kengram-extract`** bumps `OpenAICompatibleConfig::model_version` 2 → 3 when v3 prompt ships.
- **`kengram-mcp`** updates the two search tool handlers to call the reranker after RRF fusion; reflector's routing logic gains the three-band case.
- **`kengram-storage`** gains `search_facts_vector_knn` (the dual of `search_vector_knn` for thoughts) and either a paraphrase-similarity helper or a flag-driven full-replace path on rerun.

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
9. **Per-claim retraction is durable across re-extraction:** a fact retracted via `correct_fact` does not reappear as a new fact_id carrying the same claim on the next `kengram reflect --rerun`. Operator-curated state survives extractor cycles.
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

## Close-out (2026-05-16)

M3 shipped its retrieval improvements in full: hybrid (vector ∪ trigram) RRF retrieval over thoughts and facts, fact embeddings on the same async-embedding seam as thoughts, the cross-encoder reranker via TEI with per-leg `vector_score` / `trigram_score` / `rrf_score` / `rerank_score` surfaced on every hit, and the `kengram bench rerank` A/B harness reporting nDCG@10 + MRR for RRF-only vs reranked rankings.

The pipeline-quality items pulled in mid-M3 (Phase A v3 prompt + within-call dedup + `extract` flag + observability + SPO trigram; Phase C three-band routing with `flagged` + subsumption-aware dedup + quality-aware canonical selection + per-claim retraction durability + v4 extractor prompt) also shipped. But Phase D dogfood — 7 rounds against the v4 prompt on the operator's local 30B coder model — produced a consistent finding: **statements were faithful, triples were broken.** Comparative S/O inversion, self-referential subjects, conditional-as-subject, predicate verbosity, polarity contradictions, triple-semantic drift. Each v4-prompt patch traded one failure mode for another. The substrate generating the failure modes was the wrong abstraction for the operator's use case (LLM agents reading prose, not querying by `(S, P, O)`).

The architectural pivot — collapse the facts pipeline, replace it with a JSONB metadata sidecar (people / action_items / topics / dates_mentioned / kind) on the `thoughts` table — lives at [`docs/milestones/m4-collapse-to-thoughts.md`](./m4-collapse-to-thoughts.md). The retrieval improvements M3 shipped (hybrid + reranker + A/B harness) carry forward unchanged onto the simpler thoughts-only schema.

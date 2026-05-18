# M6 — Corpus stats CLI + tagger-extracted relations (v1, non-thought targets)

**Status:** shipped 2026-05-17.

**One-line:** an operator-facing `engram stats` CLI subcommand for corpus + storage telemetry, plus the tagger's first auto-emission of relational edges (URLs, entities, persons mentioned in prose) to land directly in `thought_links` with `source = 'tagger'`.

## Design pivot

The original M6 milestone was **artifacts** — long-form document ingestion with `artifact_chunks` populated, chunking strategy, `ingest_artifact` MCP tool, and unified search across thoughts + chunks. Three signals shifted the plan:

1. The M5.2 `to_url` link target already covers the "this thought references that external doc" case without ingesting the document. Most operator needs were satisfied.
2. A 2026-05-17 live-corpus measurement (12 MB across 41 thoughts; ~1.5 KB user-data-per-thought) showed engram occupies a high-signal-density "sweet spot" between transcripts (byte-heavy) and tags (information-lean). Storing arbitrary long-form documents would dilute that.
3. Two more pressing needs surfaced: operators couldn't ask the corpus "how big are you?" without psql, and the **tagger-extracted relations** capability (LLM emits edges from prose) had a low-cost v1 shape thanks to M5.2's polymorphic targets.

The artifacts plan is preserved in `m6-artifacts.md` for historical reference. M6 reshaped to the present scope; M7 (operational maturity) unchanged.

## Scope

### M6.0 — `engram stats` CLI subcommand

- New top-level CLI subcommand: `engram stats [--scope-prefix X] [--top-scopes N]`.
- New storage helper `corpus_stats(pool, scope_prefix) -> CorpusStats` aggregating thought counts (live/retracted/untagged), content/tags/metadata byte totals, embeddings by model, link counts (by relation / by_kind / by_source), queue depths (pending_embeddings + new `pending_tags`), per-scope summary (reuses `list_scopes`), per-table heap/index/total sizes (via runtime-checked query against `pg_class`/`pg_relation_size`), and database total size.
- Plain-println rendering matching the `engram audit migrations` style; no new table-printing dependency. Sizes via `humanize_bytes` helper (1 KB = 1024 B; matches `pg_size_pretty` framing).
- No MCP surface in v1 — Ron's stated preference is operator-only ("more for me to track operational constraints without accessing the DB directly").

### M6.1 — Tagger-extracted relations v1

- `Tags` struct gains `relations: Vec<ExtractedRelation>` field (serde-default empty for backward compat with v1-v4 tags).
- New `ExtractedRelation { relation: RelationKind, target: ExtractedTarget, note: Option<String> }` and `ExtractedTarget` enum (`Entity | Person | Url` — no `Thought` variant in v1; thought-target tagger relations are deferred until entity resolution lands).
- v5 tagger prompt + JSON schema: Relations section explains the closed 7-relation vocabulary, the three target kinds, selectivity rules ("default to []", "require an explicit relational claim"), anti-patterns ("mere mention is not a relation"). Schema enforces `maxItems: 5`, closed enums on `relation` and `to_kind`, validates `to_value` length.
- `BUNDLED_TAGGER_VERSION: 4 → 5`. `engram tag --rerun --since 1970-01-01T00:00:00Z` re-tags v4 thoughts under v5.
- Drainer-side wiring (`engram_mcp::apply_tagger_relations`): after `update_thought_tags`, soft-delete prior `source='tagger'` edges from the thought (preserves audit trail; preserves `source='agent'` edges), then `insert_link` each emission with `source = 'tagger'`. Validates each target via `link::validate_target` at the same gate the agent-side `link_thoughts` uses. Bypass-on-error: a single malformed emission (e.g., non-`http(s)://` URL) is logged and skipped, never fails the whole tag job.
- New storage helper `soft_delete_tagger_edges_for_thought(pool, thought_id) -> i64`.
- `link::validate_target` visibility: `fn` → `pub(crate) fn` so the drainer reuses the same validation.
- `run_tag` CLI mirrors the drainer's relation-emission loop for synchronous re-tag runs.

## Decision log

- **CLI-only stats v1.** Operator preference; MCP `stats` tool deferred. The storage helper (`corpus_stats`) is reusable, so a future MCP wrapper is ~50 LOC. Re-evaluate if dogfood reveals agents wanting the data in-conversation.
- **`engram stats` is a top-level subcommand, not under `audit`.** The `audit` namespace is for log-table queries (migration_audit); stats is a live operator query. Different shape, different name.
- **Non-thought targets only for v1 tagger relations.** Thought-target extraction requires entity resolution ("which thought is the earlier finding?"), substantial design surface. Shipping non-thought targets first validates whether tagger-emitted edges feel right in dogfood before paying the resolution cost.
- **Soft-delete-then-insert on re-tag.** Re-tagging a thought soft-deletes its prior tagger-emitted edges and inserts fresh ones. Preserves audit trail (operator can see what v4 emitted via `deleted_at`); no accumulation if prompt drifts; mirrors M5.2's pattern. Agent-supplied edges (`source='agent'`) are unaffected.
- **Bypass-on-error in the drainer.** A malformed individual emission (failed validation, FK miss, etc.) is logged and skipped; the rest of the relations and the tag job itself proceed. Operators see warns; the corpus isn't blocked.
- **System-catalog query via `sqlx::query()` (runtime-checked).** `pg_class` / `pg_relation_size` can't be macro-checked. Matches the `insert_embedding` precedent for pgvector binds. Postgres-specific; called out in `corpus_stats`'s doc comment.
- **`maxItems: 5` on relations.** Caps per-thought tagger emission to keep responses small and force selectivity. Iterable in v6 if dogfood shows the cap is biting useful cases.

## Schema impact

No migrations. M5.2 already shipped:
- Polymorphic `thought_links` targets (entity / person / url) — used directly by tagger emissions.
- `LinkSource::Tagger` enum value (`source` column already allows it).
- Soft-delete (`deleted_at` + partial unique index) — used by `soft_delete_tagger_edges_for_thought`.

## MCP surface

- No new MCP tools.
- `link_source` field in `get_related_thoughts` responses now reliably returns `"tagger"` for tagger-emitted edges. (Operators / agents can distinguish them from agent-supplied edges via this discriminator.)

## CLI surface

- New `engram stats [--scope-prefix X] [--top-scopes N]`.
- New `engram audit migrations` (was M5.2; mentioned here for completeness alongside the new stats subcommand).
- `engram tag` and `engram embed-backfill` were extended in M5.2 with `--scope-prefix` flags.

## Tests added

- engram-storage: `corpus_stats_returns_aggregate_counts`, `corpus_stats_scope_prefix_filters_scopes_section_only`, `corpus_stats_table_sizes_include_thoughts_and_embeddings`, `corpus_stats_empty_corpus_returns_zeros`, `soft_delete_tagger_edges_for_thought_only_touches_tagger_source`, `soft_delete_tagger_edges_for_thought_idempotent_on_already_deleted`.
- engram-core: `tags_relations_serde_round_trip` (`extracted_relation_serde_round_trip`), `extracted_relation_note_optional`, `extracted_target_into_link_target_preserves_kind_and_value`, `v4_shape_without_relations_deserializes_with_empty_relations`.
- engram-extract: `valid_response_with_relations_parses_to_tags`, `tags_response_format_includes_relations_array`, plus v4→v5 prompt regression rename with new assertions on the Relations section.
- engram-mcp drain: `drain_tags_inserts_emitted_relations_with_source_tagger`, `drain_tags_re_run_soft_deletes_prior_tagger_edges_then_inserts_fresh`, `drain_tags_preserves_agent_edges_during_retag`, `drain_tags_skips_invalid_target_continues_others`.
- engram-cli: `humanize_bytes_renders_unit_scale`.

334 total tests passing post-M6.

## Out of scope (deferred)

- **MCP `stats` tool.** Operator can revisit if dogfood reveals agents wanting in-conversation corpus telemetry.
- **Thought-target tagger relations.** v2 work. Requires entity resolution (heuristic + LLM disambiguation against recent same-scope thoughts).
- **First-class entity / person tables.** Entities/persons remain free-text strings on `thought_links.to_entity` / `to_person`.
- **Tagger relation confidence scoring.** v1 emits-or-doesn't; threshold-filtering can land later if dogfood shows noisy emissions.
- **`engram stats --json`.** Plain-text only for v1.
- **Original M6 (artifacts).** Permanently dropped. The M5.2 `to_url` link target covers the common "reference external doc" use case; storing arbitrary documents was the wrong shape for engram's signal-density corpus.
- **Hard-purge of soft-deleted tagger edges.** Backlogged. Pair with a retention-policy CLI subcommand when growth becomes interesting.

## Risks

- **Tagger prompt v5 quality is empirical.** The wiring is straightforward; whether the prompt produces useful relations vs. noisy ones is a dogfood question. Same pattern as M4.1's v2→v3→v4 prompt iteration — ship a deliberately selective starting point, iterate.
- **Tagger latency increase.** Adding a `relations` field to the LLM response is a minor extension of the same JSON call. Should be small; the schema-constrained mode keeps inference bounded.
- **Re-tag churn rows.** Each `--rerun` soft-deletes the prior tagger edges and inserts fresh ones. At single-operator scale this is trivial; flagged for M7 if storage growth becomes operationally interesting.

---

# v6 + v7 — Post-M6.1 dogfood iterations

**Status:** shipped 2026-05-17 as a single bundled commit (v6 prompt + v7 entities-section fix + JSON schema concrete-type fix on `tag_filter` / `metadata`). v6 was the initial post-M6.1 dogfood iteration; v7 dropped a phrase-list backfire surfaced by a second dogfood pass on the WIP v6 prompt before commit.

## What surfaced from the v5 dogfood

A 2026-05-17 dogfood pass on the v5 tagger (post-M6.1 re-tag across `engram.m3.dogfood`, 17 thoughts) surfaced three regressions:

1. **Kind classification collapsed to `observation`** — 17/17 thoughts came back as `observation`, including mission/charter statements that should be `task`, definitional thoughts that should be `reference`, and finding-shaped thoughts that should be `idea`. The closed 6-value enum was empirically reduced to 1 in practice.
2. **Entity field regressed on world-knowledge hallucination** — thought `63ad01e0` (Probe 2A) extracted `pg_trgm` from prose containing only "trigram retrieval"; the model inferred the underlying Postgres extension from world-knowledge.
3. **Entity field regressed on adjectival miscategorization** — thought `047d0ce8` (Probe 2B) extracted `embedding-based` and `lexical signals` as entities. Same class as the v3 regression v4 was supposed to close.
4. **URL emissions failed 2/2** with "URL target must start with http:// or https://" — the model emitted partial URLs / bare domains in the new v5 `relations` field; app-side validation rejected at the gate.

A separate finding in the dogfood report — `tag_filter` silently ignored — was investigated and proven a false positive (orchestrator filters correctly against the live corpus; tracing instrumentation in commit `8b8dc9a` makes future false claims diagnosable from server logs).

## v6 prompt changes

1. **Kind rebalanced as a 5-step decision tree** with `observation` as the explicit catchall (not the default). The model walks the tree in order; only the catchall step lands on `observation`. Anti-default framing inverts v5's degenerate bias. Each step has 1-2 worked examples in the prompt body.
2. **Entity surface-only rule** with explicit "do NOT infer from world knowledge" and `pg_trgm` cited as the failure case. Final-pass "re-read and verify" instruction added.
3. **Adjectival re-tightening** via pattern-based negative examples (adjectives, descriptive noun phrases, descriptively-used phrases) — the v3→v4 lesson preserved: structural patterns, not literal phrase lists.
4. **URL emission tightening** — explicit "FULL `http://` or `https://` URL only" with the `arxiv.org/abs/...` partial-URL case as the failure example.
5. **Structural tweaks**: kind reordered to sit next to entities (the two highest-signal classification fields adjacent); relations block shortened to free attention budget; closing "Before you emit" final-pass review section.

`BUNDLED_TAGGER_VERSION: 5 → 6`. `TaggerConfig::default().model_version` likewise. Operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill the corpus under v6.

Schema unchanged.

## Diagnostic updates

`kind_stability_diagnostic` and `kind_stability_diagnostic_with_vocab` (`crates/engram-extract/src/openai_compatible.rs`) gain:
- A 7th fixture (`63ad01e0`) pinning the pg_trgm hallucination case.
- Updated descriptors on `8a533e15` (kind=task target) and `047d0ce8` (no-adjectival-entity target).
- Per-run entity capture + printed entity emissions section so the operator can visually verify surface-only behavior alongside kind stability.

Per-fixture v6 pass criterion: dogfood failure cases emit the expected behavior in ≥7/10 runs. Imperfect stability is acceptable; the goal is bias-shift, not deterministic output.

## Verification (operator-driven)

1. `engram tag --rerun --since 1970-01-01T00:00:00Z` re-tags the corpus under v6.
2. `engram stats` confirms kind diversity restored (not all `observation`).
3. Sample 5-10 thoughts in dogfood scope: verify entity field is surface-only (no `pg_trgm`-class hallucination, no adjectival phrases).
4. Sample relations: verify URLs that land start with `http(s)://`; verify the model isn't refusing to emit (zero relations everywhere would signal over-conservatism).
5. Optional, Ollama-gated: `cargo test -p engram-extract --features integration --release -- kind_stability_diagnostic --nocapture --ignored` runs the extended diagnostic against the 7 fixtures.

## Risk notes

- **Long-prompt attention budget.** v6 grows the prompt (decision tree + worked examples + final-pass review). The relations block was shortened to compensate. If v7 dogfood shows attention-budget degradation, the relations section can shrink further (the anti-pattern documentation lives more naturally in the MCP `link_thoughts` tool description anyway).
- **Over-conservative emission.** Tightening entity rules + narrowing observation may cause the model to under-emit. Verification step catches it; v6.1 can re-soften.
- **The v3→v4 negative-example backfire** (acknowledged risk in the v6 plan; reproduced empirically; addressed in v7).

## v7 amendment — drop the entities NOT-list

A second dogfood pass on the WIP v6 prompt (against `engram.m3.dogfood`) confirmed the entities backfire reproduced: thought `047d0ce8`'s entities were `["agent memory protocol", "embedding-based", "lexical signals"]` — the same v5-era output. The v6 entities section's "Patterns that are NOT entities" block listed `embedding-based` and `lexical signals` explicitly as examples of adjectival / descriptive failures. Same v3→v4 lesson: listing the phrases (or even their structural suffix patterns) in the prompt reinforces them. Verified by Ron with `search_thoughts(tag_filter={"entities": ["embedding-based"]})` returning the offending thought.

v7 drops the entire "Patterns that are NOT entities" block. The entities section now contains only:
- The lead-with-empty framing (kept from v4).
- A surface-only rule citing the `pg_trgm` hallucination case as a positive example of what NOT to do (the only acceptable negative example is one not in the actual corpus).
- The NAME-vs-DESCRIBE structural test (kept from v4).
- The final-pass re-read verification (new in v6, kept in v7).

The positive examples list (`engram`, `pgvector`, `PostgreSQL`, ...) is retained — those reinforce desired behavior.

## v7 also documents topics-as-concept-mapping explicitly

The same dogfood surfaced the v4→v5 topics shift (phrase-driven → concept-mapping) had stayed undocumented. Five existing corpus findings claim "topics are phrase-driven" with empirical support — those claims are now empirically false on their own cited evidence (Probe 2 disjoint-vocab pair has 2/3 topic overlap). v7 adds an explicit statement to the topics section: "Topics map prose to canonical subject categories — they may be inferred when the subject is clear, even if the exact topic word doesn't appear. Two thoughts about the same subject may share topics even with disjoint surface vocabulary. This is concept-mapping behavior, not surface-lexeme lifting."

This makes the long-standing v4 behavior explicit. The stale claims in `6d2ef58e`, `74eb781c`, `137dba1d`, `ce83b7ba` remain operator-action items (retract-and-replace).

## JSON schema concrete-type fix on `tag_filter` and `metadata`

Bundled into the v7 commit because it surfaced from the same dogfood: claude.ai's MCP client silently strips fields whose schema declarations lack a concrete `type`. Engram's `SearchThoughtsArgs.tag_filter: Option<serde_json::Value>` and `CaptureArgs.metadata: Option<serde_json::Value>` produced schemas with only `description` (no `type`). Wire-tested with raw curl: the orchestrator filters correctly when the field arrives. Audited with the claude.ai client: the field never arrives. Fix: change both Rust types to `Option<serde_json::Map<String, serde_json::Value>>` so schemars renders `type: ["object", "null"]`. New regression test `tool_args_object_fields_have_concrete_schema_type` pins the shape so a regression to `Option<Value>` fails CI before ship.

The diagnostic that surfaced this is `tag_filter-strip-diagnostic.md` in the repo root (operator-supplied; not committed).

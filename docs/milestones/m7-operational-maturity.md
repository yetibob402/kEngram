# M7 — Operational maturity

## Goal

kEngram becomes production-shaped: observable, securable for non-Tailnet access, backed up, and continuously evaluated. After M6 the operator can confidently run kengram for years and — should they choose — share access with non-Tailnet clients (Claude Desktop, ChatGPT) without giving up on auth or audit.

This milestone is omnibus by design. It bundles the operational concerns that earned their keep in M1–M5 but were deferred so the core loop could ship faster.

## In scope

- **Prometheus `/metrics` endpoint.** Exposed on the same axum server. Metrics: capture-rate, search-latency P50/P95/P99 (per tool), embedding-queue depth, embedder failure count, tagger-queue depth, tagger failure count.
- **Tier 2 auth.** Bearer tokens validated against a hashed allowlist in a new `kengram_tokens` table. Per-token scope-list — a token can be locked to `work.*` and not see `personal.*`. Audit log records `(token_id, tool, args_hash, ts)` for every call. Allow/deny is enforced at the MCP handler layer.
- **Backup tooling.** **`kengram backup` / `kengram restore` subcommands shipped at M7.0** — wrap `pg_dump`/`pg_restore` with a `manifest.json` sidecar (kengram version, schema head version, embedder model, tagger version, corpus counts). Restore validates the manifest against the target before proceeding: numeric schema-head compare (refuses on mismatch), embedder/tagger drift surfaced as warnings, `--force` required only when the target has existing thoughts. Single-file `.tar.gz` archive containing the dump + manifest. The nightly/weekly retention story (systemd timers + off-site copy to Backblaze B2 / rsync.net) is the remaining piece of this scope item.
- **Eval suite.** Four suites per design doc §13 — **tagger quality (shipped M7.1)**, capture-recall, cross-model retrieval consistency, LongMemEval-style. Runs via `kengram eval <suite>` subcommands; emits a JSON report. Retrieval-suite fixture corpus is a small, version-controlled set of seeded conversations and target queries; the tagger suite's golden corpus is operator-personal (drafted by read-only export, hand-reviewed, gitignored) with a committed synthetic example.
- **`stats` MCP tool.** Per-scope counts, last activity timestamp, embedding model id and version, tagger model id and version, queue depths.
- **Tier 1 → Tier 2 deployment guide.** A short ops doc covering the steps to expose kengram outside the Tailnet (Cloudflare Tunnel or Caddy + Let's Encrypt; token issuance; revocation).

## Out of scope (deferred to which milestone)

- Tier 3 (public + multi-user) — out of scope indefinitely. Would require OAuth2, per-user data partitioning, much more audit infrastructure. Implementable later if there's a reason, which is currently not foreseen.
- Web UI — out of scope indefinitely.
- Cross-instance replication — Postgres logical replication is straightforward but only worth doing if actually used.
- Memory forgetting / TTL policies — possibly post-M6 if the operator finds kengram retains things they want pruned.
- Capture UX surfaces (Telegram bot, Raycast extension, browser extension) — possibly post-M6 as standalone projects that talk to kengram via MCP.

## Schema impact

A new migration (next available number after the M4 `0006_collapse_to_thoughts.sql`) adds:

- `kengram_tokens (id, hash, scopes, label, created_at, revoked_at)` — bearer token registry.
- `kengram_audit (id, token_id, tool, args_hash, ts)` — call audit log. Append-only; rotated externally.

The eval suite is filesystem-resident (fixture YAML / JSON; report JSON output). No tables.

## MCP surface delta

- `stats(scope?: string) -> { thoughts: int, last_capture_at: timestamptz, embedding_model_id: string, embedding_count: int, tagger_model_id: string, tagged_count: int, queue_depth: int, ... }` — the omnibus introspection tool.

No other tool's signature changes; auth is enforced at the MCP-handler level and is invisible to compliant callers.

## Crate structure delta

- **`kengram-cli`** gains subcommands: `eval`, `audit` (lightweight report over `kengram_audit`), `restore`. The auth-config plumbing also lives here.
- **New module** in `kengram-core` (or a small new `kengram-eval` crate) for the eval-suite logic — fixture loading, query execution, scoring, report emission.
- **`kengram-mcp`** gains the `stats` handler and an auth middleware that enforces token validation + scope filtering.
- **`kengram-storage`** gains repository functions for tokens and audit log.

## Dependencies

- **Prior milestones:** all (M1–M5). Many of M6's metrics are only meaningful once the worker (M2), reranker (M3), tagger (M4), and artifacts (M5) exist.
- **External services:** a Prometheus scraper for `/metrics` to be useful (operator-managed; out of kengram's scope). Cloudflare Tunnel / Caddy if Tier 2 is used.

## Success criteria

1. **Prometheus integrated.** A scraper targeting `/metrics` produces a usable dashboard. Operator can answer "what's my capture-rate this week?" and "is my embedding queue growing?" without `psql`.
2. **Tier 2 auth enforced.** A request with a missing or invalid token gets a clean 401. A token with `scope: ["work.*"]` cannot read `personal.*` thoughts. The audit log shows one row per request.
3. **Backup + restore round-trip.** A `pg_dump` taken yesterday, restored to a fresh Postgres, plus the kengram binary booted against that DB → all M1–M5 functionality works end-to-end. The operator has done this at least once and trusts the runbook.
4. **Eval suite reproducible.** `kengram eval --suite capture-recall` produces the same JSON report on a clean checkout. Cross-model eval (re-embed with a different model, measure top-10 overlap) detects an obvious quality regression intentionally introduced for the test.
5. **Operator dogfood.** kEngram has been running for a quarter; backups exist and have been restored at least once for verification; the eval suite has caught at least one real regression during development; the operator is comfortable enough to consider sharing it.

## Open questions

- **Token scope-list grammar.** Glob (`work.*`), prefix-match exactly, regex? Glob is simplest and matches the dotted-scope convention.
- **Eval fixture corpus.** Synthetic-only (entirely seeded conversations) or include some anonymized real captures? Synthetic-only is reproducible and shareable; real captures cover odd cases the operator actually hits.
- **Audit log retention.** Keep forever, rotate after N days, compact into daily summaries? Affects storage and the privacy story for any future Tier 2 / Tier 3 deployment.
- **Metrics cardinality.** Per-scope metrics are useful but explode cardinality if scopes proliferate. Cap, sample, or aggregate?
- **Reranker eval integration.** Should the eval suite run cross-model both with and without reranker, to track that as a separate axis?
- **Tunnel vs. reverse proxy.** Cloudflare Tunnel is the lower-config option; Caddy + Let's Encrypt is the more-portable one. Both, with the tunnel as default in the docs?

## History

- **2026-06-11 — M7.1.1: eval-harness hardening for unattended runs.** Prompted by the first real overnight eval (4 arms × 116 golden items × 3 repeats; three arms CPU/GPU-split at 26 GB, plus the documented gemma constrained-decoding loop risk). Three additions to `crates/kengram-cli/src/eval/`: (1) per-arm stderr progress (`[arm 2/4 …] 120/348 done (3 failed), avg 13.8s/call, ETA 52m`); (2) incremental per-arm report flushing to the `--out` path with a `run.complete` flag (a crash/Ctrl-C keeps finished arms; table shows a PARTIAL banner); (3) `--max-consecutive-failures` guard (default 5) — abandons a looping/dead arm via `JoinSet::abort_all`, records `aborted` reason + `skipped` count on the arm (its resolved calls still score), joins the exit-1 path. `CallStats.attempted` now means resolved calls. Report struct additions: `RunInfo.complete`, `CallStats.skipped`, `ArmReport.aborted`. 4 new tests (70 total in kengram-cli) incl. a slow-failing fake tagger (instant fakes finish before an abort can land — the test had to model real latency). Smoke-verified: live progress + flush against Ollama, dead-endpoint arm aborts after 3 failures in seconds with exit 1 while the healthy arm completes.

- **2026-06-09 — M7.1 ship: tagger-quality eval suite (`kengram eval tagger` + `kengram eval export-corpus`).** First eval-suite surface to land, prioritized by the 2026-06-09 architecture review (findings B1/D2/D4: retrieval/tagger quality unmeasured; default tagger model contradicts sweep evidence; fixtures too small). New `crates/kengram-cli/src/eval/` module (seven files; `backup.rs`-precedent layout): `corpus.rs` (golden-corpus JSON schema v1 — items with full golden `Tags` labels plus corpus-level `known_scopes` + per-scope vocab snapshots so the production pipeline context travels with the file), `arms.rs` (models TOML; arms constructed through the production `OpenAICompatibleTagger`/`HttpTagger` builders, so the GLiNER sidecar is just another arm and prompt-A/B inherits the provenance binding), `score.rs` (pure math: normalized set P/R/F1 with micro+macro aggregation, greedy token-F1 action-item matching at 0.6 with substring fallback, 8×8 kind confusion matrix, modal-kind agreement + per-field pairwise Jaccard for stability), `run.rs` (bounded-concurrency orchestration; every call scored on raw AND finalized output — `kengram_mcp::finalize_tags` runs in-eval, so headline numbers measure what production would persist, with per-field `finalize_delta`), `report.rs` (JSON report with full per-call audit trail + terminal table), `export.rs` (read-only corpus drafting via `find_untagged_or_stale_thoughts(force=true)` + `list_scopes` + `fetch_scope_vocab`; draft labels = current prod tags, `reviewed: false` until hand-checked). **The no-production-DB invariant is structural**: the `eval tagger` dispatch arm never receives the config, the DB-free modules are pinned by a tripwire test, and `export-corpus` was behaviorally verified read-only (sqlx::test row-identity check + live `kengram stats` before/after diff). Committed assets: `eval/corpora/example.json` (9 synthetic items covering all 7 kinds + null, the metadata decision_type override, and a scope-identifier-bait case), `eval/models.example.toml`; `.gitignore` blocks personal corpora/reports/models.toml (repo is planned OSS). 26 new tests (66 total in kengram-cli); smoke-verified end-to-end against live Ollama (qwen3-coder:30b, 2-item run: report + table + audit trail correct). Deferred to v1.1: relations scoring, stratified corpus sampling, retiring `tagger-sweep.sh`. Exit codes: 0 ok / 1 dead arm (report still written) / 2 usage error.

- **2026-05-18 — M7.0 ship: backup/restore subcommands.** First M7 surface to land. New `crates/engram-cli/src/backup.rs` (~580 LOC including tests); two new `Command` variants (`Backup` / `Restore`); manifest sidecar (`BackupManifest` struct, version 1) with engram version, RFC3339 created_at, schema head (numeric `head_version` + display `head_name`), all migrations list, full `migration_audit` journal, embedder model_id+dimensions, tagger model_id+version, corpus counts. Restore compatibility model: refuses on schema-head mismatch (numeric compare to avoid "11" vs "9" lex-sort bugs), warns on embedder/tagger drift, requires `--force` only when target has existing thoughts. Smoke-tested end-to-end: backup of live corpus (42 live + 10 retracted thoughts, 52 embeddings, 96 links, 5 scopes) into a fresh `engram_test` database round-tripped with exact-match counts. Postgres client tools (`pg_dump` / `pg_restore`) added as a runtime dependency, documented in DEVELOPMENT.md. Remaining M7 scope: Prometheus /metrics, Tier 2 auth, eval suite, retention cron, tunnel deployment guide.


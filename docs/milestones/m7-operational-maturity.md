# M6 — Operational maturity

## Goal

Engram becomes production-shaped: observable, securable for non-Tailnet access, backed up, and continuously evaluated. After M6 the operator can confidently run engram for years and — should they choose — share access with non-Tailnet clients (Claude Desktop, ChatGPT) without giving up on auth or audit.

This milestone is omnibus by design. It bundles the operational concerns that earned their keep in M1–M5 but were deferred so the core loop could ship faster.

## In scope

- **Prometheus `/metrics` endpoint.** Exposed on the same axum server. Metrics: capture-rate, search-latency P50/P95/P99 (per tool), embedding-queue depth, embedder failure count, tagger-queue depth, tagger failure count.
- **Tier 2 auth.** Bearer tokens validated against a hashed allowlist in a new `engram_tokens` table. Per-token scope-list — a token can be locked to `work.*` and not see `personal.*`. Audit log records `(token_id, tool, args_hash, ts)` for every call. Allow/deny is enforced at the MCP handler layer.
- **Backup tooling.** Scripts (likely systemd timers + small shell wrappers; the operator's preference) for nightly `pg_dump --format=custom` to a separate disk, weekly off-site copy (Backblaze B2 / rsync.net). A new CLI subcommand `engram restore --from <dump>` for disaster-recovery validation.
- **Eval suite.** Three suites per design doc §13 — capture-recall, cross-model retrieval consistency, LongMemEval-style. Runs via `engram eval --suite <name>`; emits a JSON report. Fixture corpus is a small, version-controlled set of seeded conversations and target queries.
- **`stats` MCP tool.** Per-scope counts, last activity timestamp, embedding model id and version, tagger model id and version, queue depths.
- **Tier 1 → Tier 2 deployment guide.** A short ops doc covering the steps to expose engram outside the Tailnet (Cloudflare Tunnel or Caddy + Let's Encrypt; token issuance; revocation).

## Out of scope (deferred to which milestone)

- Tier 3 (public + multi-user) — out of scope indefinitely. Would require OAuth2, per-user data partitioning, much more audit infrastructure. Implementable later if there's a reason, which is currently not foreseen.
- Web UI — out of scope indefinitely.
- Cross-instance replication — Postgres logical replication is straightforward but only worth doing if actually used.
- Memory forgetting / TTL policies — possibly post-M6 if the operator finds engram retains things they want pruned.
- Capture UX surfaces (Telegram bot, Raycast extension, browser extension) — possibly post-M6 as standalone projects that talk to engram via MCP.

## Schema impact

A new migration (next available number after the M4 `0006_collapse_to_thoughts.sql`) adds:

- `engram_tokens (id, hash, scopes, label, created_at, revoked_at)` — bearer token registry.
- `engram_audit (id, token_id, tool, args_hash, ts)` — call audit log. Append-only; rotated externally.

The eval suite is filesystem-resident (fixture YAML / JSON; report JSON output). No tables.

## MCP surface delta

- `stats(scope?: string) -> { thoughts: int, last_capture_at: timestamptz, embedding_model_id: string, embedding_count: int, tagger_model_id: string, tagged_count: int, queue_depth: int, ... }` — the omnibus introspection tool.

No other tool's signature changes; auth is enforced at the MCP-handler level and is invisible to compliant callers.

## Crate structure delta

- **`engram-cli`** gains subcommands: `eval`, `audit` (lightweight report over `engram_audit`), `restore`. The auth-config plumbing also lives here.
- **New module** in `engram-core` (or a small new `engram-eval` crate) for the eval-suite logic — fixture loading, query execution, scoring, report emission.
- **`engram-mcp`** gains the `stats` handler and an auth middleware that enforces token validation + scope filtering.
- **`engram-storage`** gains repository functions for tokens and audit log.

## Dependencies

- **Prior milestones:** all (M1–M5). Many of M6's metrics are only meaningful once the worker (M2), reranker (M3), tagger (M4), and artifacts (M5) exist.
- **External services:** a Prometheus scraper for `/metrics` to be useful (operator-managed; out of engram's scope). Cloudflare Tunnel / Caddy if Tier 2 is used.

## Success criteria

1. **Prometheus integrated.** A scraper targeting `/metrics` produces a usable dashboard. Operator can answer "what's my capture-rate this week?" and "is my embedding queue growing?" without `psql`.
2. **Tier 2 auth enforced.** A request with a missing or invalid token gets a clean 401. A token with `scope: ["work.*"]` cannot read `personal.*` thoughts. The audit log shows one row per request.
3. **Backup + restore round-trip.** A `pg_dump` taken yesterday, restored to a fresh Postgres, plus the engram binary booted against that DB → all M1–M5 functionality works end-to-end. The operator has done this at least once and trusts the runbook.
4. **Eval suite reproducible.** `engram eval --suite capture-recall` produces the same JSON report on a clean checkout. Cross-model eval (re-embed with a different model, measure top-10 overlap) detects an obvious quality regression intentionally introduced for the test.
5. **Operator dogfood.** Engram has been running for a quarter; backups exist and have been restored at least once for verification; the eval suite has caught at least one real regression during development; the operator is comfortable enough to consider sharing it.

## Open questions

- **Token scope-list grammar.** Glob (`work.*`), prefix-match exactly, regex? Glob is simplest and matches the dotted-scope convention.
- **Eval fixture corpus.** Synthetic-only (entirely seeded conversations) or include some anonymized real captures? Synthetic-only is reproducible and shareable; real captures cover odd cases the operator actually hits.
- **Audit log retention.** Keep forever, rotate after N days, compact into daily summaries? Affects storage and the privacy story for any future Tier 2 / Tier 3 deployment.
- **Metrics cardinality.** Per-scope metrics are useful but explode cardinality if scopes proliferate. Cap, sample, or aggregate?
- **Reranker eval integration.** Should the eval suite run cross-model both with and without reranker, to track that as a separate axis?
- **Tunnel vs. reverse proxy.** Cloudflare Tunnel is the lower-config option; Caddy + Let's Encrypt is the more-portable one. Both, with the tunnel as default in the docs?

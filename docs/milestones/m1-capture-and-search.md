# M1 — Capture and search

## Goal

Deliver a usable single-user memory service. The operator can capture thoughts via MCP and retrieve them via hybrid search. State is durable, persistent across sessions, and accessible from any MCP-capable client (Claude Code, Claude Desktop, opencode, etc.) configured to point at the same kengram instance.

This is the floor: nothing else in the roadmap is useful without it. M1 must run end-to-end on the operator's hardware and be pleasant enough that they would actually use it for a week before reaching for the next milestone.

## In scope

- Cargo workspace with five crates (`kengram-core`, `kengram-storage`, `kengram-embed`, `kengram-mcp`, `kengram-cli`).
- Migration `0001_initial.sql` shipping the full schema described in design doc §5 (thoughts, embeddings, facts, artifacts, artifact_chunks). Future-milestone tables ship empty.
- The `Embedder` trait (in `kengram-core`) plus two implementations: `TeiEmbedder` (production: HTTP client for the TEI sidecar, BGE-M3, 1024-dim, `model_id = "bge-m3:1024"`) and `CloudEmbedder` (dev/test: Voyage AI or OpenAI; off by default).
- Sync embedding on capture: the handler inserts the thought, calls `Embedder::embed`, writes the embedding row, returns the thought ID.
- Hybrid retrieval: vector kNN ∪ trigram lexical, fused via reciprocal rank fusion (`k = 60`), with a recency boost (`exp(-age/τ)`, default `τ = 30 days`).
- Four MCP tools: `capture`, `search_thoughts`, `recent_thoughts`, `get_thought`.
- Single binary `kengram` with subcommands `serve` and `migrate`. No worker process.
- Tier 0 auth (bind `127.0.0.1`). Tier 1 (Tailnet bind) is a config change, not a code change.
- Structured `tracing` logs to stderr / journald.
- `figment`-based layered TOML + env config.
- Integration tests using `sqlx::test` against a real Postgres, plus a TEI-backed integration test gated by a feature flag.

## Out of scope (deferred to which milestone)

- Facts extraction, `Extractor` trait, vLLM client → **M2**
- Worker process and async embedding seam → **M2**
- `kengram-extract` crate → **M2**
- `search_facts`, `correct_fact` MCP tools → **M2**
- Cross-encoder reranker → **M3**
- `ingest_artifact` MCP tool, chunking, artifact_chunks population → **M4**
- Prometheus `/metrics`, bearer-token auth, audit log, eval suite, `stats` MCP tool → **M5**
- CLI subcommands beyond `serve` and `migrate` (e.g. `capture`, `search` via shell) → later milestones; not required for M1
- Web UI → out of scope indefinitely

## Schema impact

Migration `0001_initial.sql` is the only migration in M1. It ships:

- Extensions: `pgcrypto`, `vector` (pgvector ≥ 0.7), `pg_trgm`.
- Tables: `thoughts`, `artifacts`, `artifact_chunks`, `embeddings`, `facts`. Only `thoughts` and `embeddings` are populated by M1 code.
- Indexes: `thoughts_scope_recent_idx (scope, created_at DESC)`, `thoughts_content_trgm_idx (gin/trgm)`, `embeddings_bge_m3_hnsw` (HNSW partial, predicated on `model_id = 'bge-m3:1024'`), `facts_active_idx (scope, created_at DESC) WHERE superseded_at IS NULL`.

The exact SQL is the source of truth for design doc §5 and will be written in the M1 detailed design conversation. Do not improvise; the migration file should match §5 byte-for-byte once authored.

## MCP surface delta

Four tools are added. Argument and response shapes resolved; the precise rmcp/serde-derived JSON Schema is produced in the M1 detailed design conversation.

### Conventions

- JSON keys: `snake_case` (serde default).
- Timestamps: RFC 3339 strings.
- Error model: **hard failures** (validation, not-found, DB unreachable, content over limit) return an MCP tool error. **Soft failures** (embedding timed out, TEI unavailable during search) return success with an in-band field flagging the degraded state.

### `capture`

**Args:**
- `content` — required, string, max 1 MiB.
- `source` — required, string.
- `scope` — optional, string, default `"global"`.
- `metadata` — optional, object, default `{}`. Free-form; standard keys (`client_name`, `session_id`, `tool_name`, `agent_role`) are recommended via convention but not enforced.

**Response:** `{ thought_id: uuid, embedding_status: "indexed" | "pending" }`

**Errors:** content over limit; database unreachable.

### `search_thoughts`

**Args:**
- `query` — required, string, non-empty.
- `scope` — optional, string. **When omitted, searches across *all* scopes.**
- `limit` — optional, int, default `10`, max `100`.
- `recency_half_life_days` — optional, float, default `30`.

**Response:** `{ results: [{ thought_id, content, scope, source, created_at, metadata, score }], vector_search_available: bool }`

When TEI is unreachable the call **still succeeds** with `vector_search_available: false` and trigram-only results. Full thought content is included in each result (acceptable at M1 volumes; revisit if response sizes become a problem).

**Errors:** empty query; limit out of bounds.

### `recent_thoughts`

**Args:**
- `scope` — optional, string. When omitted, returns across all scopes.
- `limit` — optional, int, default `10`, max `100`.

**Response:** `{ results: [{ thought_id, content, scope, source, created_at, metadata }] }` ordered by `created_at DESC`.

### `get_thought`

**Args:**
- `thought_id` — required, uuid.

**Response:** `{ thought: { thought_id, content, scope, source, created_at, metadata }, provenance: { embedding_status: "indexed" | "pending", embedded_at?: timestamp, linked_facts: [] /* populated at M2 */ } }`

**Errors:** not found.

## Crate structure delta

Five new crates introduced (the workspace doesn't exist yet).

- **`kengram-core`** — domain types (`Thought`, `ThoughtId`, `Scope`, `Embedding`, `EmbeddingModel`, `Query`, `SearchHit`), the `Embedder` trait, RRF fusion logic. No I/O. No sqlx, no HTTP, no rmcp.
- **`kengram-storage`** — sqlx-typed queries, migration runner, repository functions for `thoughts` and `embeddings` (insert thought, insert embedding, fetch thought by id, vector kNN, trigram search, recent-by-scope). Knows nothing about MCP.
- **`kengram-embed`** — `TeiEmbedder` (HTTP client for the TEI sidecar) and `CloudEmbedder` (dev/test). Both implement the `Embedder` trait from `kengram-core`.
- **`kengram-mcp`** — `rmcp` tool descriptors and handler implementations for the four M1 tools. Depends on `kengram-core`, `kengram-storage`, `kengram-embed`.
- **`kengram-cli`** — the only binary. `main.rs`, axum bootstrap, MCP transport wiring, `figment` config loader, `tracing` init, subcommand dispatch.

## Dependencies

- **Prior milestone: M0** (dev environment — Docker Postgres, Ollama with `bge-m3` pulled). M1 assumes a working `DATABASE_URL` and a reachable embedding endpoint per `DEVELOPMENT.md`.
- **External services required at runtime in production:** Postgres 16+ with `pgvector` ≥ 0.7, `pg_trgm`, `pgcrypto`; TEI sidecar reachable on `embedder.endpoint` (CPU build is fine; latency is invisible at single-user volumes).
- **External services required at runtime in dev:** Same Postgres (via the M0 compose file); Ollama with `bge-m3` pulled, in place of TEI.
- **External services NOT required:** vLLM. (M1 has no extractor; vLLM is M2 onward.)

## Success criteria

M1 is complete when:

1. `cargo build --workspace` and `cargo clippy --all-targets -- -D warnings` pass clean.
2. `sqlx migrate run` against a fresh Postgres 16 with the three required extensions completes cleanly. The schema matches design doc §5.
3. **Integration test (no TEI dependency):** `sqlx::test` test, using a fake `Embedder` that returns deterministic vectors, exercises the full path: capture writes a thought + embedding row in one transaction; `search_thoughts` returns the thought when queried for its exact content; trigram fallback finds it via a partial-string query that vector misses; `recent_thoughts` orders by `created_at DESC`; `get_thought` returns the row.
4. **Integration test (TEI):** gated by a `--features integration` flag; runs against a live TEI sidecar; round-trips capture → embed → search end-to-end.
5. **MCP smoke test:** running `kengram serve` and pointing Claude Code (or `mcp-inspector`) at it, the operator can call all four tools and observe correct behavior.
6. **Operator dogfood:** the operator runs M1 against their personal Postgres + TEI for at least a week with no crashes and no manual database surgery required.

## Open questions

To be resolved in the M1 detailed design conversation:

- ~~**MCP argument/response shapes.**~~ **Resolved** — see the "MCP surface delta" section above. Notable choices: `search_thoughts` and `recent_thoughts` default to all-scopes when `scope` is omitted; soft-fail on TEI unavailability (search returns trigram-only with a flag, capture marks `embedding_status: "pending"`); full thought content returned in search results at M1 volumes.
- **Capture flow on embedding failure.** If TEI is down, do we (a) refuse the capture, (b) commit the thought with no embedding row and queue for retry, (c) commit and silently skip indexing? Each has implications for recovery and search consistency. RJF: I like (b) in that it would seem to safeguard the captures with the ability to run an embedding catchup later, but let me know if not.
- **Capture timeout.** Sync embedding implies a timeout budget. What is it (e.g. 2s)? What's the response on timeout? RJF: Hard to know until I understand the embedding cost on the hardware I run it on. Configurable I guess? As far as response, just report a timeout. 
- **Config file location.** `~/.config/kengram/kengram.toml`, `./config/kengram.toml`, or a `--config` flag? What's the figment layer order? RJF: The first one.
- **sqlx query strategy.** `sqlx::query!` for everything statically known; the hybrid search query may need string SQL because the WHERE clause varies with optional scope filtering. Where is the seam? RJF: Yeah, use query! for as much as possible, and for the rest just be sure we protect against injection attacks. 
- **RRF parameters.** `k = 60` is the literature default; do we deviate? Top-K per leg (50? 100? 200?)? Does the recency boost apply pre-fusion (per leg) or post-fusion (to the merged ranking)? RJF: Configurable, at least to start, until we find the sweet spot? Can we test our way into the sweet spot?
- **Dev-mode embedder.** Resolved by M0: dev points `CloudEmbedder` (or, more accurately, an OpenAI-compatible embedder impl) at Ollama's `http://localhost:11434/v1/embeddings` with `bge-m3`. A deterministic fake `Embedder` impl still belongs alongside it for sqlx-test unit tests where Ollama need not be running. Whether the OpenAI-compatible path is one struct (`OpenAICompatibleEmbedder`) covering Ollama / Voyage / OpenAI / TEI, or three separate structs sharing an HTTP client, is the M1 implementation question.
- **`metadata` JSONB shape.** Free-form, or do we recommend (not enforce) a few standard keys (`client_name`, `session_id`, `tool_name`)? RJF: standard keys would seem to increase predicability, but push back if that's not the point.
- **Connection-pool sizing default.** `max_connections = 10` for single-user — or smaller? RJF: configurable. 
- **Transaction granularity in capture.** Single transaction wrapping thought + embedding insert (atomic; failure of either rolls back), or two transactions (thought commits first; embedding can be retried independently)? Hinges on the failure-handling answer above. RJF: assuming embedding can be retried any time, then two transactions make sense. 
- **Default `source` value.** If the agent doesn't pass `source`, do we infer from the MCP client's identity, default to `"manual"`, or require it? RJF: require it
- **Scope hierarchy semantics.** `work.tcgplayer` is convention. Does `search_thoughts(scope="work")` match `work.tcgplayer` rows by prefix, or only exact? RJF: I would say only exact matches. Claude: agreed for M1; if we later find ourselves writing `OR scope = ...` ad-hoc, that's the signal to add a `scope_match: "exact" | "prefix"` parameter additively. Don't pre-build it.

### Surfaced by the answers above (new for M1)

- **Backfill mechanism for unembedded thoughts.** Answer (b) on capture-failure means a TEI/Ollama outage produces thoughts with no `embeddings` row. M1 has no worker. Simplest path: a CLI subcommand `kengram embed-backfill [--scope <s>] [--limit <n>]` that finds thoughts missing an embedding (LEFT JOIN, IS NULL) and embeds them inline. M2's worker takes this over and the subcommand stays for ad-hoc use. To be confirmed in the M1 detailed design conversation; alternative is no backfill in M1 (operator runs SQL by hand or restarts capture). RJF: This sounds like a reasonable M1 decision until we have a worker framework.
- **Visibility of capture-without-embedding.** Without surfacing this, a silent TEI outage degrades vector search invisibly. At minimum, emit a `tracing` WARN with the thought_id for every capture that didn't index. Possibly also a small admin endpoint or log-aggregator-friendly metric counting "thoughts pending embedding." RJF: for M1, I think trace is appropriate. We can revisit as the design evolves. 
- **Capture response shape — embedding status.** Should `capture` return `{ thought_id, embedding_status: "indexed" | "pending" }` so an agent that just captured knows whether `search_thoughts` will immediately surface their thought via vector? Tiny addition; large clarity win for agents acting on freshly captured content. RJF: I like the idea of at least letting the agent know the real state with pending.
- **`metadata` enforcement vs. convention.** RJF preferred standard keys for predictability. Claude pushes back: keep `metadata` JSONB free-form, document a *recommended* set (`client_name`, `session_id`, `tool_name`, `agent_role`) in CLAUDE.md and query against them when present, but don't validate. Validation is CPU + code + flexibility cost; convention is free. RJF: approved.

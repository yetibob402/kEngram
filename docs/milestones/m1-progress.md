# M1 — Progress

Living checklist tracking M1 implementation. Each phase ends in a runnable, reviewable checkpoint. Items are checked off as they land; the **History** section at the bottom captures dated notes — decisions made in passing, surprises, things deferred. The companion design doc is `m1-capture-and-search.md` in this directory.

## Phase A — Foundation ✅

End state: workspace compiles clean; database schema is loaded.

- [x] Root `Cargo.toml`: `[workspace]` members + `[workspace.dependencies]` block listing every crate from the CLAUDE.md Stack table, pinned to current stable versions
- [x] Library crates: `engram-core`, `engram-storage`, `engram-embed`, `engram-mcp` (all empty, all compile)
- [x] Binary crate: `engram-cli` declaring `[[bin]] name = "engram"`
- [x] `.gitignore` (Rust `target/`, IDE files, `.env`, `.DS_Store`)
- [x] `migrations/0001_initial.sql` matching design doc §5
- [x] `sqlx migrate run` succeeds against the M0 Docker Postgres
- [x] `cargo build --workspace` clean
- [x] `cargo clippy --all-targets -- -D warnings` clean

## Phase B — Capture vertical slice

End state: an agent can call `capture` over MCP; thought row + embedding row land in the database; soft-fail returns `embedding_status: "pending"` cleanly.

- [ ] `engram-core` domain types: `Thought`, `ThoughtId`, `Scope`, `Source`, `Embedding`, `EmbeddingModel`, `Metadata`
- [ ] `engram-core` `Embedder` trait
- [ ] `engram-embed` `OpenAICompatibleEmbedder` (covers Ollama / TEI / OpenAI / Voyage by config)
- [ ] `engram-embed` `FakeEmbedder` (deterministic; for sqlx-tests with no Ollama dependency)
- [ ] `engram-storage` repository functions: insert thought, insert embedding, fetch thought by id
- [ ] `engram-mcp` `capture` tool descriptor + handler (using rmcp)
- [ ] `engram-cli` `serve` subcommand: axum + rmcp transport on `127.0.0.1:<port>`
- [ ] `figment` config loader: `~/.config/engram/engram.toml` + `ENGRAM_*` env overrides + `--config <path>` override
- [ ] `tracing` initialization: structured output to stderr
- [ ] `sqlx::test`: `capture` with `FakeEmbedder` writes both rows, returns `embedding_status: "indexed"`
- [ ] `sqlx::test`: `capture` with a failing `FakeEmbedder` returns `embedding_status: "pending"`; thought row exists; embedding row absent; WARN logged

## Phase C — Search vertical slice

End state: capture → search end-to-end via MCP. Hybrid retrieval (vector ∪ trigram, RRF) returns ranked results. Trigram-only fallback works when the embedder is down.

- [ ] `engram-storage` vector kNN query against `embeddings_bge_m3_hnsw`
- [ ] `engram-storage` trigram similarity query against `thoughts_content_trgm_idx`
- [ ] `engram-storage` recent-by-scope query against `thoughts_scope_recent_idx`
- [ ] `engram-core` RRF fusion (`k = 60` default; configurable) + post-fusion recency boost
- [ ] `engram-mcp` tools: `search_thoughts`, `recent_thoughts`, `get_thought`
- [ ] Soft-fail on embedder unavailable: `search_thoughts` returns `vector_search_available: false` with trigram-only results
- [ ] `sqlx::test`: full hybrid search round-trip with `FakeEmbedder`
- [ ] `sqlx::test`: search with embedder unavailable returns degraded results plus the flag
- [ ] `sqlx::test`: `recent_thoughts` orders by `created_at DESC`
- [ ] `sqlx::test`: `get_thought` returns full row with `embedding_status` in provenance

## Phase D — Hardening

End state: M1 success criteria from `m1-capture-and-search.md` met.

- [ ] `engram embed-backfill [--scope <s>] [--limit <n>]` subcommand: finds thoughts missing an embedding (LEFT JOIN, IS NULL), embeds them inline
- [ ] `sqlx::test`: backfill finds and embeds previously-pending thoughts
- [ ] `engram migrate` subcommand (wraps sqlx migration runner)
- [ ] `cargo test --workspace --features integration` against live Ollama: real capture → embed → search round-trip
- [ ] MCP smoke test: Claude Code (or `mcp-inspector`) calls all four tools against `engram serve` successfully
- [ ] README quick-start for the operator (or fold into `DEVELOPMENT.md`)
- [ ] Operator dogfood begins (informal; reported in History)

## History

Dated notes appended as items land. Format: `YYYY-MM-DD — <one-line summary>`. Multi-line entries fine for decisions that need explanation.

<!-- Most recent entry first. -->

- **2026-05-09** — Phase A complete. Workspace skeleton (`Cargo.toml` + 5 crates) compiles cleanly with edition 2024 on rustc 1.95. Resolved versions: `tokio 1`, `axum 0.8.9`, `sqlx 0.8.6`, `pgvector 0.4.1`, `reqwest 0.12.28`, `figment 0.10.19`, `clap 4.6.1`, `rmcp 0.1.5`, `tracing-subscriber 0.3.23`. Migration `0001_initial.sql` applied in 39 ms; all five tables, three required extensions (`pgcrypto`, `vector 0.8.2`, `pg_trgm 1.6`), and the four named indexes (including `embeddings_bge_m3_hnsw` HNSW partial) confirmed via `\dt`/`\dx`/`\di`. Note: `chrono` resolved transitively (figment → uncased → chrono); we use `time` directly per workspace deps, so this is a transitive duplicate, not a workspace-level inconsistency.

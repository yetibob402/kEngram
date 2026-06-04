# kEngram — Claude Code Project Context

## What this is

kEngram is a self-hosted, MCP-native AI memory service. It stores agent "thoughts" and extracted facts in Postgres + pgvector, and exposes a small set of MCP tools (capture, search, etc.) so any MCP-capable client — Claude Code, opencode, Claude Desktop, ChatGPT — reads and writes the same persistent backing store.

The system is being built across seven capability milestones (M1 → M7), preceded by an environment milestone (M0). All milestones through M6.1 are shipped as of 2026-05-18, plus M7.0 (the `kengram backup` / `restore` surface); the rest of M7 (operational maturity) is the next focus. The terminal-state design lives in `DESIGN.md`; per-milestone scope and success criteria live in `docs/milestones/`.

## Documents

- **`DESIGN.md`** — design doc. Describes the M5-complete terminal system. Inline milestone callouts (`[M1]`, `[M2+]`, etc.) flag features by milestone. **Read it before starting any non-trivial work.**
- **`docs/milestones/m{N}-*.md`** — one document per milestone (M0–M7). Scope, success criteria, schema/MCP/crate deltas, dependencies, open questions. The current-focus milestone document is the unit of work for the next planning conversation.
- **`docs/milestones/m{N}-progress.md`** — paired progress doc for the current-focus milestone. Phased checklist + dated History notes. Updated as work lands.
- **`DEVELOPMENT.md`** — first-time setup runbook (Docker Postgres, Ollama, `DATABASE_URL`).

If you find yourself reasoning about a design choice that the design doc already addresses, re-read it. If you find yourself working on something that doesn't appear in the current-focus milestone document, stop and ask.

## Stack (fixed)

| Layer | Choice |
|---|---|
| Language | Rust (stable, latest) |
| Async runtime | Tokio |
| HTTP server | axum |
| Database | Postgres 16+ with `pgvector` ≥ 0.7 and `pg_trgm` |
| DB access | `sqlx` (compile-time checked; no ORM) |
| Vector type | `pgvector` crate with sqlx integration |
| MCP protocol | `rmcp` crate |
| Config | `figment` for layered TOML + env |
| Logging | `tracing` + `tracing-subscriber` (JSON output for journald) |
| Errors | `thiserror` (libraries), `anyhow` (binary only) |
| Embedding sidecar | Hugging Face `text-embeddings-inference` (HTTP API, external process) |
| Extraction backend | vLLM (OpenAI-compatible API at `/v1`, external process) |

**Do not introduce Python, Node, or TypeScript dependencies.** The deployment is the kEngram binary plus pre-existing TEI and vLLM services. If you find yourself wanting a Python tool for something, check whether a Rust crate or `sqlx-cli` covers it first.

## Repo layout

```
.
├── Cargo.toml              # workspace manifest, [workspace.dependencies] block
├── CLAUDE.md               # this file
├── DESIGN.md               # the design doc (terminal state)
├── README.md
├── DEVELOPMENT.md          # operator setup + configuration reference
├── docs/
│   └── milestones/         # one doc per milestone
├── crates/
│   ├── kengram-core/        # domain types, Embedder/Tagger/Reranker traits, retrieval fusion (pure)
│   ├── kengram-storage/     # sqlx + migrations + repository functions
│   ├── kengram-embed/       # Embedder + Reranker impls: TEI, cloud (dev/test)
│   ├── kengram-extract/     # Tagger impls: OpenAI-compatible (vLLM/OpenRouter/Ollama) + HTTP-sidecar client
│   ├── kengram-tagger-protocol/      # wire types for the HTTP tagger-sidecar contract
│   ├── kengram-tagger-deterministic/ # reference non-LLM tagger sidecar (opt-in)
│   ├── kengram-mcp/         # rmcp tools + the embed/tag drainers and finalize pipeline
│   └── kengram-cli/         # the only binary; axum + transport, config, subcommands
├── migrations/             # sqlx migrations, numbered (0001_initial.sql, ...)
└── config/
    └── kengram.example.toml
```

Eight crates. `kengram-cli` is the only binary; the rest are libraries with no `main`. Each library crate exposes a small, well-named API surface; cross-crate calls go through trait abstractions where the design doc indicates (`Embedder`, `Tagger`, `Reranker`). (The workspace started at five crates in M1; `kengram-extract` landed with the facts pipeline, and the two `kengram-tagger-*` crates with the pluggable-tagger work.)

## Conventions

- **Async by default.** Any I/O is async. Sync code is reserved for pure computation.
- **No `unwrap()` outside tests.** Use `expect()` with a descriptive message if you genuinely cannot recover at the call site.
- **Errors:** `thiserror` enums in library crates with one variant per genuine failure mode. `anyhow::Error` is fine in `kengram-cli` only. Do not propagate `anyhow::Error` across library boundaries.
- **Compile-time SQL.** Use `sqlx::query!` / `sqlx::query_as!`. Reserve string SQL for genuinely dynamic shapes (e.g., constructing the hybrid search query at runtime).
- **No clever macros** unless they replace at least 50 lines of boilerplate they're not just hiding.
- **No `Box<dyn Error>` in public APIs.** It eats type information.
- **Trait objects for swappable backends** (`Embedder`, `Tagger`, `Reranker`); concrete types everywhere else.
- **Tests live next to the code** they test. Integration tests use `sqlx::test` against a real Postgres.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` pass before any commit.

## What NOT to do

- Do not make kEngram operate vLLM or TEI. Those are external services with their own systemd units. kEngram only *consumes* their HTTP APIs.
- Do not add a "hosted mode" or multi-tenant abstractions. Single user, single active session is a design assumption (§1 of the design doc).
- Do not introduce an ORM (Diesel, SeaORM). `sqlx` is the choice and the reasons (compile-time checking, async-native, no macro magic) are intentional.
- Do not implement a web UI in v0. Postgres + `psql` is the admin interface.
- Do not implement any reflector or re-embedding flow that cannot be re-run. Idempotency is required.
- Do not invent an MCP tool that's not in the current milestone's MCP surface (see the milestone doc) without raising it as a design question first. The full tool set lives in §8 of the design doc, but tools ship by milestone.
- Do not edit `thoughts` rows once written. They are immutable. (§10 of the design doc explains why.)

## Build / run / test

```bash
# First time
cp config/kengram.example.toml config/kengram.toml  # then edit
cargo build --workspace

# Database (assumes local Postgres with the kengram db created)
sqlx database create
sqlx migrate run

# Run the server
cargo run --bin kengram -- serve

# Run the worker (drains pending_embeddings + pending_tags on each tick)
cargo run --bin kengram -- worker

# Tests
cargo test --workspace
cargo test --workspace --features integration  # requires running Postgres + TEI
```

## Current state

- ✅ **M0–M6.1 shipped** (as of 2026-05-18). Eight-crate workspace; migrations `0001`–`0011`. Live: the capture/search MCP surface, hybrid retrieval (vector kNN + trigram, RRF fusion, cross-encoder rerank), thought retraction, the relational link graph (`thought_links`, M5/M6.1), and the LLM tagging sidecar (M4) with a worker that drains `pending_embeddings` + `pending_tags`.
- ✅ **Tagger at prompt v16** (`BUNDLED_TAGGER_VERSION`). Post-M6.1 dogfood iteration added: the `decision_record` kind, forward-looking `action_items`, deterministic scope-identifier / relationship-noun filters + a `metadata.decision_type` override in the shared `kengram_mcp::finalize` seam (run by both the worker drainer and `kengram tag`), provenance binding (the version stamp can't drift from the prompt), and an entities cap of 15. An opt-in deterministic NER backend lives in `kengram-tagger-deterministic`. Iteration log: `docs/tagger-improvements.md`; cross-model eval harness: `./tagger-sweep.sh`.
- 🚧 **M7 (operational maturity)** is in progress — M7.0 backup/restore (`kengram backup` / `restore`) shipped 2026-05-18; remaining: Prometheus `/metrics`, Tier 2 auth, eval suite. Day-to-day work right now is dogfooding the live corpus and refining the tagger.

## Next concrete step

The shipped milestones (M0–M6.1, plus M7.0 backup/restore) are done; the remaining **M7 (operational maturity)** work — Prometheus metrics, Tier 2 auth, eval suite — is next; read `docs/milestones/m7-operational-maturity.md` for its scope and success criteria. New milestone work follows the doc-driven flow: read the milestone doc, produce a plan file, get approval, then code (same discipline as *How to handle ambiguity* below). Between milestones, the active work is dogfooding the live corpus and refining the tagger (`docs/tagger-improvements.md` is the iteration log).

## How to handle ambiguity

If you encounter a design ambiguity that the doc doesn't resolve, surface it explicitly rather than guessing. The format that works:

> The design doc says A in §X. Issue Y is unaddressed. Two options are (1) ..., (2) .... Which does the operator want?

Wait for a real answer before committing a direction. Guessing creates technical debt that's expensive to undo because the wrong guess often shapes downstream code. The operator prefers a question now over a rewrite later.

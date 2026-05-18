# Engram — Claude Code Project Context

## What this is

Engram is a self-hosted, MCP-native AI memory service. It stores agent "thoughts" and extracted facts in Postgres + pgvector, and exposes a small set of MCP tools (capture, search, etc.) so any MCP-capable client — Claude Code, opencode, Claude Desktop, ChatGPT — reads and writes the same persistent backing store.

The system is being built across seven capability milestones (M1 → M7), preceded by an environment milestone (M0). All milestones through M6.1 are shipped as of 2026-05-18; M7 (operational maturity) is the next focus. The terminal-state design lives in `DESIGN.md`; per-milestone scope and success criteria live in `docs/milestones/`.

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

**Do not introduce Python, Node, or TypeScript dependencies.** The deployment is the Engram binary plus pre-existing TEI and vLLM services. If you find yourself wanting a Python tool for something, check whether a Rust crate or `sqlx-cli` covers it first.

## Repo layout

```
.
├── Cargo.toml              # workspace manifest, [workspace.dependencies] block
├── CLAUDE.md               # this file
├── DESIGN.md               # the design doc (terminal state)
├── README.md
├── DEVELOPMENT.md          # operator setup + configuration reference
├── AGENTS.md               # agent-facing usage preferences
├── docs/
│   └── milestones/         # one doc per milestone
├── crates/
│   ├── engram-core/        # domain types, Embedder trait, retrieval fusion (pure)
│   ├── engram-storage/     # sqlx + migrations + repository functions
│   ├── engram-embed/       # Embedder impls: TEI, cloud (dev/test)
│   ├── engram-mcp/         # rmcp tool definitions and handlers
│   └── engram-cli/         # the only binary; axum + transport, config, subcommands
├── migrations/             # sqlx migrations, numbered (0001_initial.sql, ...)
└── config/
    └── engram.example.toml
```

Five crates in M1. `engram-extract` joins at M2 when the facts pipeline lands. Library crates have no `main`. `engram-cli` is the only binary. Each library crate exposes a small, well-named API surface; cross-crate calls go through trait abstractions where the design doc indicates (the `Embedder` trait being the primary one in M1).

## Conventions

- **Async by default.** Any I/O is async. Sync code is reserved for pure computation.
- **No `unwrap()` outside tests.** Use `expect()` with a descriptive message if you genuinely cannot recover at the call site.
- **Errors:** `thiserror` enums in library crates with one variant per genuine failure mode. `anyhow::Error` is fine in `engram-cli` only. Do not propagate `anyhow::Error` across library boundaries.
- **Compile-time SQL.** Use `sqlx::query!` / `sqlx::query_as!`. Reserve string SQL for genuinely dynamic shapes (e.g., constructing the hybrid search query at runtime).
- **No clever macros** unless they replace at least 50 lines of boilerplate they're not just hiding.
- **No `Box<dyn Error>` in public APIs.** It eats type information.
- **Trait objects for swappable backends** (`Embedder`, `Extractor`); concrete types everywhere else.
- **Tests live next to the code** they test. Integration tests use `sqlx::test` against a real Postgres.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` pass before any commit.

## What NOT to do

- Do not make Engram operate vLLM or TEI. Those are external services with their own systemd units. Engram only *consumes* their HTTP APIs.
- Do not add a "hosted mode" or multi-tenant abstractions. Single user, single active session is a design assumption (§1 of the design doc).
- Do not introduce an ORM (Diesel, SeaORM). `sqlx` is the choice and the reasons (compile-time checking, async-native, no macro magic) are intentional.
- Do not implement a web UI in v0. Postgres + `psql` is the admin interface.
- Do not implement any reflector or re-embedding flow that cannot be re-run. Idempotency is required.
- Do not invent an MCP tool that's not in the current milestone's MCP surface (see the milestone doc) without raising it as a design question first. The full tool set lives in §8 of the design doc, but tools ship by milestone.
- Do not edit `thoughts` rows once written. They are immutable. (§10 of the design doc explains why.)

## Build / run / test

```bash
# First time
cp config/engram.example.toml config/engram.toml  # then edit
cargo build --workspace

# Database (assumes local Postgres with the engram db created)
sqlx database create
sqlx migrate run

# Run the server
cargo run --bin engram -- serve

# (M2+) Run the worker (reflector + re-embed jobs)
# cargo run --bin engram -- worker

# Tests
cargo test --workspace
cargo test --workspace --features integration  # requires running Postgres + TEI
```

## Current state

- ✅ Design doc revised (v0.1) after milestone-roadmap brainstorm. See `DESIGN.md`.
- ✅ Per-milestone documents drafted at `docs/milestones/m{0..5}-*.md`.
- ✅ **M0 dev environment shipped.** `docker-compose.yml` at repo root for Postgres 16 + pgvector + pg_trgm + pgcrypto. `DEVELOPMENT.md` documents first-time setup. Dev-mode embedder is Ollama at `http://localhost:11434/v1/embeddings` (model `bge-m3`).
- ⏳ **Current focus: M1 design.** Workspace, migration, capture/search, and MCP surface are all yet to be designed in detail.
- ⏳ No Rust code written yet.

## Next concrete step

Plan M1 in detail. Read `docs/milestones/m1-capture-and-search.md` and the M1-tagged sections of the design doc, then start a fresh planning conversation that produces:

1. The exact migration SQL (a file, not a description).
2. The Cargo workspace skeleton — five crates per the layout above, plus `[workspace.dependencies]` with pinned versions for everything in the Stack table.
3. The capture and search request/response types and handler shapes.
4. The four M1 MCP tool signatures.

Before starting that conversation, **bring up M0**: `docker compose up -d postgres` and `ollama pull bge-m3`. See `DEVELOPMENT.md`.

That conversation produces another plan file and, after approval, becomes code. Subsequent milestones get their own planning conversations driven by their milestone documents.

## How to handle ambiguity

If you encounter a design ambiguity that the doc doesn't resolve, surface it explicitly rather than guessing. The format that works:

> The design doc says A in §X. Issue Y is unaddressed. Two options are (1) ..., (2) .... Which does the operator want?

Wait for a real answer before committing a direction. Guessing creates technical debt that's expensive to undo because the wrong guess often shapes downstream code. The operator prefers a question now over a rewrite later.

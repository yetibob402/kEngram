# M0 — Development environment

## Goal

A reproducible dev environment on the operator's Mac. Postgres with required extensions runs in Docker; an embedder is reachable; the operator can clone the repo, run two or three commands, and have a working environment that M1 will then build code against.

This is the floor under the floor. M0 doesn't deliver any user-facing capability; it ensures M1's code has somewhere to run.

## In scope

- A `docker-compose.yml` at the repo root that brings up Postgres 16 with `pgvector` extension support (the `pgvector/pgvector:pg16` image bundles `vector`, `pg_trgm`, and `pgcrypto`, which is everything our schema needs). Persistent named volume; port `5432` exposed on localhost; healthcheck.
- A short `DEVELOPMENT.md` (or equivalent section in the README) with first-time-setup steps: clone, `docker compose up -d postgres`, set `DATABASE_URL`, `ollama pull bge-m3`, you're ready.
- Defaults: database name `kengram`, role `kengram`, password `kengram` (dev-only — fine because the container only accepts connections from `localhost`). Connection string `postgres://kengram:kengram@localhost:5432/kengram`.
- Decision on the M1 dev-mode embedder: **Ollama**, since the operator already has it installed. The `CloudEmbedder` impl in `kengram-embed` is pointed at Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1/embeddings`) with the `bge-m3` model. No additional sidecar process needed in dev. Production retains the TEI sidecar via systemd — this is a dev-only convenience.
- (Optional) a `Makefile` or `justfile` with `db-up`, `db-down`, `db-reset`, `db-shell` recipes. Optional because `docker compose up -d`, `docker compose down`, and `docker exec -it kengram-postgres psql -U kengram` are all short enough to type.

## Out of scope (deferred to which milestone)

- **TEI sidecar in dev** — the production embedder. Available behind a config switch from M1 onward. Not the dev default because Ollama is already on the operator's box. Included in the design doc, not in the compose file.
- **vLLM in dev** → **M2**. Once M2 lands, the same Ollama trick covers extraction (`http://localhost:11434/v1/chat/completions` against any pulled instruct model).
- CI/CD configuration → not a current milestone; revisit when there's something worth running in CI.
- Production systemd units → described in design doc §11; not part of dev setup.

## Schema impact

None. M0 doesn't write migrations. The `pgvector/pgvector:pg16` image ships the extensions; the M1 migration's `CREATE EXTENSION IF NOT EXISTS …` lines actually install them in the kengram database when the migration runs.

## MCP surface delta

None. M0 has no code.

## Crate structure delta

None. M0 has no code.

## Dependencies

- Operator has Docker installed (confirmed).
- Operator has Rust toolchain (`rustc 1.95`, `cargo 1.95`) installed (confirmed).
- Operator has `sqlx-cli` installed at `~/.cargo/bin/sqlx` (confirmed).
- Operator has Ollama installed (confirmed).

`psql` is *not* required (M1 uses `sqlx`, the worker has its own DB connection); but it's useful for poking at the database, and the simplest way to get it on macOS is `brew install libpq && brew link --force libpq`.

## Success criteria

M0 is complete when, on a fresh checkout:

1. `docker compose up -d postgres` starts a healthy Postgres container within ~10 seconds.
2. `docker exec -it kengram-postgres psql -U kengram -d kengram -c '\dx'` lists the available extensions; `pgvector`, `pg_trgm`, and `pgcrypto` are all installable (will be `CREATE EXTENSION`-ed by the M1 migration; M0 just needs them available).
3. `ollama pull bge-m3` succeeds; `curl http://localhost:11434/v1/embeddings -d '{"model":"bge-m3","input":"hello"}'` returns a 1024-element vector.
4. The operator can stop and start the database (`docker compose down` / `up -d`) without losing data, thanks to the named volume.
5. `DEVELOPMENT.md` walks a new contributor (which is to say: future-you) through these steps in order without surprises.

## Open questions

- **Host port conflict.** If the operator already runs Postgres on `5432`, the bind needs to move (e.g. `5433:5432`). Default to `5432`; document the override in `DEVELOPMENT.md`. Resolve as needed when the conflict is real.
- **Ollama embedding output dim.** Need to verify `ollama pull bge-m3` produces 1024-dim vectors (it should — `bge-m3` is 1024 by definition — but worth confirming on first run).
- **Volume strategy.** Named volume (`kengram-pg-data`) is the default — survives `compose down`, lost on `compose down -v`. Bind mount to `./.data/postgres` is an alternative (visible in the working tree). Picked named volume for cleanliness; revisit if the operator wants to inspect the raw DB files.
- **Should `DEVELOPMENT.md` also document the production TEI path?** Probably yes, briefly, for the operator's future-self. Not a blocker.

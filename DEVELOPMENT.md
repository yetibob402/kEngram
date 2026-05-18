# Development setup

Quick start for working on Engram on macOS. Assumes Docker, Rust (`rustc` 1.95+), `sqlx-cli`, and Ollama are already installed.

## First-time setup

### 1. Start Postgres in Docker

```bash
docker compose up -d postgres
```

This launches `pgvector/pgvector:pg16` with the `vector`, `pg_trgm`, and `pgcrypto` extensions available. The container is named `engram-postgres`. Data lives in a named Docker volume (`engram-pg-data`) and survives `docker compose down`.

Wait for it to be healthy:

```bash
docker compose ps postgres
# STATUS should show "healthy"
```

### 2. Set the database URL

```bash
export DATABASE_URL="postgres://engram:engram@localhost:5432/engram"
```

`sqlx` and the engram binary both read this. Add it to your shell rc if you don't want to set it every session.

### 3. Pull the embedding model in Ollama

```bash
ollama pull bge-m3
```

Engram's dev-mode embedder talks to Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1/embeddings`) and uses `bge-m3` for 1024-dim embeddings.

Verify:

```bash
curl http://localhost:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"hello"}' | jq '.data[0].embedding | length'
# expect: 1024
```

### 3b. (Optional) Start TEI for the rerank stage

The M3 Phase B step 2 cross-encoder reranker runs in a TEI Docker container alongside Postgres. It's optional — `engram serve` works without it (the search pipeline silently skips the rerank stage when no `[reranker]` section is configured).

```bash
docker compose up -d tei
# First boot downloads ~85 MB (cross-encoder/ms-marco-MiniLM-L-6-v2) and
# warms up in a few seconds. The healthcheck has a 120s start_period
# (mostly headroom for first-time downloads on slow networks).

docker compose ps tei
# STATUS should reach "healthy"
```

Smoke:

```bash
curl -s http://localhost:8080/health
# expect: 200 (empty body is OK)

curl -s http://localhost:8080/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"reproducibility","texts":["Nix is reproducible","Redis is fast","Bazel is powerful"]}' \
  | jq .
# expect: array of {"index": i, "score": s} sorted by score desc
```

Then add a `[reranker]` section to your `engram.toml` (see Configuration below) and `engram serve`'s startup log will show `reranker: resolved config`.

**Model choice.** `docker-compose.yml` pins `cross-encoder/ms-marco-MiniLM-L-6-v2` — the small (~22M parameter) dev reranker that has ONNX exports on HF (TEI takes the fast ORT path; sub-100ms per call on Apple Silicon CPU). For production with a GPU host, override via `[reranker].model_id` to `BAAI/bge-reranker-v2-m3` or another full-size model.

The Apple Silicon variant of the image (`cpu-arm64-latest`) is what's pinned. Production deployments use TEI as a systemd-managed sidecar, not Docker — same HTTP interface either way.

### 4. Run migrations

```bash
cargo run --bin engram -- migrate
# or, equivalently
sqlx migrate run
```

The migration creates the three required extensions in the `engram` database and ships the schema described in `docs/engram-design-v0.md` §5.

**Migration audit (M5.2+).** The `migration_audit` table (introduced in 0010) records what each migration did — `migration`, `ran_at`, `rows_touched`, optional `notes`. Convention going forward: any row-touching migration ends with an `INSERT INTO migration_audit (...)` statement so the operator can verify per-migration impact via `engram audit migrations` rather than psql. Schema-only migrations should still insert an audit row with `rows_touched = 0` and a one-line `notes` summary.

### 5. Build, test, run

```bash
cargo build --workspace
cargo test --workspace                       # unit + sqlx::test
cargo test --workspace --features integration   # adds a live-Ollama round-trip test

cargo run --bin engram -- serve              # starts the MCP server on 127.0.0.1:8080
cargo run --bin engram -- worker             # in a second shell — drains pending_embeddings
```

Point an MCP-capable client (Claude Code, Claude Desktop, `mcp-inspector`) at `http://127.0.0.1:8080/mcp` (streamable-HTTP transport, per the current MCP spec). Five tools are exposed:

- `capture` — write a thought; returns `thought_id`, `embedding_status: "pending"`, and `is_duplicate`. Same content captured twice (SHA-256 fingerprint match) returns the existing `thought_id`.
- `search_thoughts` — RRF-fused vector + trigram retrieval over thoughts; recency-boosted; optional cross-encoder rerank; optional `tag_filter` JSONB-containment filter (e.g. `{"kind": "task"}`). Each hit carries its `tags` object.
- `recent_thoughts` — chronological browse.
- `get_thought` — full thought + provenance + tags + tagger provenance.
- `retract_thought` — mark a thought as untrusted (excluded from retrieval; still visible via `get_thought` for audit).

`engram serve` and `engram worker` are paired: `serve` writes thoughts and enqueues embedding + tag jobs; `worker` drains both queues (`pending_embeddings` and `pending_tags`). Running `serve` without `worker` is fine — thoughts are still durable and trigram-searchable — but vector kNN won't surface them and tags stay empty until the worker runs. When `[tagger].provider` is empty, the tag-job enqueue at capture is a no-op and the tag drainer doesn't spawn.

`sqlx::query!` macros and the `sqlx::test` attribute both require `DATABASE_URL` to be set at *build time*, not just at runtime. The `.env` file at the workspace root is read by `sqlx-cli` but NOT by `cargo build` — set `DATABASE_URL` in your shell or pass it inline: `DATABASE_URL=... cargo build`.

## Common operations

```bash
# Stop the database (data persists)
docker compose down

# Stop and wipe the database
docker compose down -v

# Open a psql session in the container
docker exec -it engram-postgres psql -U engram -d engram

# Tail Postgres logs
docker compose logs -f postgres

# Heal-then-drain backfill: enqueue any unembedded thoughts that lack a
# queue row (pre-M2 captures, or captures whose enqueue lost a crash race),
# then drain the queue inline. Use this if you've been running `serve`
# without `worker` and want to catch up without spinning up the worker.
cargo run --bin engram -- embed-backfill --limit 1000
# or restricted to one scope:
cargo run --bin engram -- embed-backfill --scope work --limit 100

# One-shot tagger run: like a single tick of the worker's tag drainer.
# Tags thoughts where `tags_extractor_version IS NULL`. Requires vLLM
# (or another OpenAI-compatible chat endpoint) reachable on the
# `[tagger]` config's endpoint. Useful for catching up after capturing
# a batch of thoughts before configuring `[tagger]`.
cargo run --bin engram -- tag --limit 50
cargo run --bin engram -- tag --scope work --limit 100

# Re-tag: re-run the tagger over thoughts whose stored
# `tags_extractor_version` is below the configured current version.
# Use this after bumping `[tagger].model_version` (typically after a
# prompt or schema change). Tags are overwritten in place — no
# supersede semantics, no audit chain. Pair with --since to bound the
# rerun to recent thoughts. Use `--since 1970-01-01T00:00:00Z` to
# re-tag the entire corpus (e.g. after enabling the tagger for the
# first time on a previously-captured backlog, or after M4.1's v1→v2
# bump for the entities split + scope vocabulary prompt).
cargo run --bin engram -- tag --rerun --scope work
cargo run --bin engram -- tag --rerun --since 2026-04-01T00:00:00Z

# M4.1 backfill recipe: after deploying a new prompt version, refresh the
# entire corpus under the new schema. The same command applies to any
# bump — the drainer walks rows where tags_extractor_version < the
# configured target.
#
# IMPORTANT: if you've pinned `model_version` in your local
# `~/.config/engram/engram.toml`, bump it manually — the new bundled
# default (4) only applies when the field is absent from your TOML. The
# log line at startup reports the resolved value: look for
# `target_version=4`. If it says `target_version=3` (or lower), your
# config still overrides; either update the line or delete it.
cargo run --bin engram -- tag --rerun --since 1970-01-01T00:00:00Z

# Caveat: agents that hardcoded tag_filter queries against an earlier
# version's shape may need updating. The v3 iteration tightened entities to
# canonical proper names; v4 restructured the rule with a NAME-vs-DESCRIBE
# test and lowered entities maxItems 5→3. Descriptive phrases like "agent
# memory protocol" or "cross-encoder" that v2 sometimes landed in entities
# are now consistently excluded; expect those queries to miss thoughts.
# Migrate `tag_filter` to use topics for descriptive-phrase searches.
# v4 also softened scope-vocabulary from "vocab dominates topic choice"
# to "vocab tie-breaks" — expect modestly more topic diversity at the
# cost of slightly weaker corpus-level coherence on topics.

# A/B-benchmark the reranker against RRF-only on an operator-curated
# fixture corpus. Prints a markdown table to stdout with per-query
# nDCG@10 and MRR for both rankings, plus an AVERAGE row. Requires a
# configured [reranker] section in engram.toml and the corpus's
# relevant_ids to point at real thought_id rows in your DB. See
# tests/fixtures/bench-rerank.example.json for the schema.
cargo run --bin engram -- bench rerank --corpus ~/.engram/my-bench.json
```

## Configuration

Defaults live in code. Override via `~/.config/engram/engram.toml`, a `--config <path>` argument, or `ENGRAM_*` env vars (nested via `__`, e.g. `ENGRAM_DATABASE__URL`).

Example `engram.toml`:

```toml
[server]
bind = "127.0.0.1:8080"
# Optional `allowed_hosts` extends the MCP DNS-rebinding allowlist beyond
# rmcp's safe default (localhost / 127.0.0.1 / ::1). When binding non-
# loopback (Tailnet, LAN, 0.0.0.0), include both the bare hostname AND
# `hostname:port` forms the client uses, plus IP and `ip:port` forms.
# Leaving this list empty when bind is non-loopback rejects every non-
# localhost request (look for "rejected request with disallowed Host
# header" warnings in the serve log). Bypass-all is intentionally not
# exposed — Tailnet ACLs plus an explicit allowlist is Tier 1 auth.
# Example for a Tailnet bind:
#   bind = "0.0.0.0:8081"
#   allowed_hosts = [
#     "localhost", "127.0.0.1", "::1",
#     "repromax", "repromax:8081",
#     "100.110.75.74", "100.110.75.74:8081",
#   ]

[database]
url = "postgres://engram:engram@localhost:5432/engram"
max_connections = 10

[embedder]
provider = "openai-compatible"
endpoint = "http://localhost:11434/v1"   # Ollama for dev; TEI in production
model = "bge-m3"
model_id = "bge-m3:1024"                 # must match an HNSW partial index
dimensions = 1024
timeout_seconds = 5

[worker]
tick_interval_seconds = 5                # how often `engram worker` drains pending_embeddings and pending_tags
batch_size = 16                          # max jobs per tick (per queue)

[tagger]                                 # M4. Empty provider = silent disable: no tag jobs enqueued, no tag drainer.
provider = "openai-compatible"           # also "openrouter"; "" = disabled
endpoint = "http://localhost:8000/v1"    # vLLM default; OpenRouter is https://openrouter.ai/api/v1
model_name = "qwen2.5-7b-instruct"       # the model the backend serves
model_id = "vllm/qwen2.5-7b-instruct"    # provenance written into thoughts.tags_extractor_model
model_version = 4                        # tagger prompt/schema version (M4.1 shipped at 2 for the entities split + vocab v2 prompt; v3 added entities anti-padding + kind-isolation; v4 restructured entities to lead-with-empty + softened scope vocab from "dominate" to "tie-break"). Bump on change and `engram tag --rerun` to re-tag rows whose stored version is older.
api_key = ""                             # bearer token for hosted endpoints (OpenRouter, etc.)
timeout_seconds = 60                     # vLLM JSON-Schema responses can run long
temperature = 0.2
scope_vocab_enabled = true               # M4.1: inject the top topic + entity terms from the thought's scope into the tagger prompt as a controlled-vocabulary hint. Encourages consistent term reuse across captures.
scope_vocab_size = 50                    # M4.1: top-N established terms (each for topics and entities) fed to the tagger. Larger = more vocabulary stability; smaller = faster emergence of new terms.
# system_prompt_file = "~/.config/engram/tagger-prompt.txt"
# When set, the file's contents replace the bundled v4 tagger prompt.
# Operator is responsible for bumping model_version when the prompt changes.

[reranker]                                              # M3 Phase B step 2; opt-in
provider = "tei"                                        # "" = disabled (default); "tei" = TEI sidecar
endpoint = "http://localhost:8080"                      # no /v1 suffix; reranker appends /rerank
model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # small/fast dev default; bigger models for prod
timeout_seconds = 30
```

Env override examples: `ENGRAM_WORKER__TICK_INTERVAL_SECONDS=2 cargo run --bin engram -- worker` (snappier ticks for development), `ENGRAM_TAGGER__API_KEY=sk-...` (OpenRouter key without checking it into config), `ENGRAM_TAGGER__PROVIDER=""` (silent-disable the tagger for a run).

## Port conflicts

If something else already binds `5432`, edit `docker-compose.yml` to map a different host port (e.g. `"5433:5432"`) and update `DATABASE_URL` accordingly.

## Production note

In production, Postgres runs as a systemd-managed service (not Docker), and the embedder is a TEI sidecar (also systemd-managed) rather than Ollama. Both deployment shapes are described in `docs/engram-design-v0.md` §11. The dev setup here exists for ergonomics — the production setup is operator-managed and out of scope for this file.

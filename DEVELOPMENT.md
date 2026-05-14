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

### 4. Run migrations

```bash
cargo run --bin engram -- migrate
# or, equivalently
sqlx migrate run
```

The migration creates the three required extensions in the `engram` database and ships the schema described in `docs/engram-design-v0.md` §5.

### 5. Build, test, run

```bash
cargo build --workspace
cargo test --workspace                       # unit + sqlx::test (~140 from M2 Phase B)
cargo test --workspace --features integration   # adds a live-Ollama round-trip test

cargo run --bin engram -- serve              # starts the MCP server on 127.0.0.1:8080
cargo run --bin engram -- worker             # in a second shell — drains pending_embeddings
```

Point an MCP-capable client (Claude Code, Claude Desktop, `mcp-inspector`) at `http://127.0.0.1:8080/mcp` (streamable-HTTP transport, per the current MCP spec). Six tools are exposed:

- `capture` — write a thought; returns `thought_id` + `embedding_status: "pending"`.
- `search_thoughts` — RRF-fused vector + trigram retrieval over thoughts; recency-boosted.
- `recent_thoughts` — chronological browse.
- `get_thought` — full thought + provenance + active `linked_facts`.
- `search_facts` — trigram search over facts with source-thought enrichment (M3 adds the vector leg).
- `correct_fact` — operator-driven supersede with optional manual-author replacement.

`engram serve` and `engram worker` are paired: `serve` writes thoughts and enqueues embedding jobs; `worker` drains the queue and (with `reflector.enabled = true`) runs the reflector cron that produces facts. Running `serve` without `worker` is fine — thoughts are still durable and trigram-searchable — but vector kNN won't surface them and no facts will be extracted until the worker runs.

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

# One-shot reflector run: like a single tick of the worker's cron task.
# Requires vLLM (or another OpenAI-compatible extractor endpoint) up on the
# `[extractor]` config's endpoint. Useful for catching up after the operator
# captured a batch of thoughts with `reflector.enabled = false`.
cargo run --bin engram -- reflect --limit 50
cargo run --bin engram -- reflect --scope work --limit 100

# Rerun: re-evaluate already-facted thoughts. Use this after upgrading the
# extractor model — if the new extractor's (S, P, O) matches an existing
# fact but rephrases the statement, the old row is superseded with
# `superseded_by` pointing at the new row (audit trail preserved). Existing
# facts the new extractor *doesn't* produce stay active (no subtractive
# logic). Pair with --since to bound the rerun to recent thoughts.
cargo run --bin engram -- reflect --rerun --scope work
cargo run --bin engram -- reflect --rerun --since 2026-04-01T00:00:00Z
```

## Configuration

Defaults live in code. Override via `~/.config/engram/engram.toml`, a `--config <path>` argument, or `ENGRAM_*` env vars (nested via `__`, e.g. `ENGRAM_DATABASE__URL`).

Example `engram.toml`:

```toml
[server]
bind = "127.0.0.1:8080"

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
tick_interval_seconds = 5                # how often `engram worker` drains the queue
batch_size = 16                          # max jobs per tick

[extractor]
provider = "openai-compatible"           # also "openrouter"
endpoint = "http://localhost:8000/v1"    # vLLM default; OpenRouter is https://openrouter.ai/api/v1
model_name = "qwen2.5-7b-instruct"       # the model the backend serves
model_id = "vllm/qwen2.5-7b-instruct"    # provenance written into facts.extractor_model
model_version = 2                        # bump when prompt/schema changes; v2 = rubric-anchored prompt (2026-05-13)
timeout_seconds = 60                     # vLLM JSON-Schema responses can run long
temperature = 0.2
max_facts_per_thought = 8
# system_prompt_file = "~/.config/engram/extractor-prompt.txt"
# When set, the file's contents replace the bundled prompt. Must contain
# the {MAX_FACTS} placeholder. Operator is responsible for bumping
# model_version when the prompt changes. The extractor emits a WARN at
# startup when a custom prompt is in use.

[reflector]
enabled = false                          # opt-in: flip to true when vLLM is running
schedule = "0 0 3 * * *"                 # 6-field cron: sec min hour dom month dow (03:00 daily)
scope_filter = ""                        # leave blank for all scopes
max_thoughts_per_run = 1000
max_facts_per_thought = 8
review_queue_below = 0.7                 # confidence below → facts_review_queue; ≥ → facts
```

Env override examples: `ENGRAM_WORKER__TICK_INTERVAL_SECONDS=2 cargo run --bin engram -- worker` (snappier ticks for development), `ENGRAM_REFLECTOR__ENABLED=true ENGRAM_REFLECTOR__SCHEDULE="*/30 * * * * *"` (every 30 seconds — useful for live dogfood), `ENGRAM_EXTRACTOR__API_KEY=sk-...` (OpenRouter key without checking it into config).

## Port conflicts

If something else already binds `5432`, edit `docker-compose.yml` to map a different host port (e.g. `"5433:5432"`) and update `DATABASE_URL` accordingly.

## Production note

In production, Postgres runs as a systemd-managed service (not Docker), and the embedder is a TEI sidecar (also systemd-managed) rather than Ollama. Both deployment shapes are described in `docs/engram-design-v0.md` §11. The dev setup here exists for ergonomics — the production setup is operator-managed and out of scope for this file.

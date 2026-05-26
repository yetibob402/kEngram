# Development setup

The operator reference for Kengram: first-time setup, common operations, the full configuration knob list, tagger version history, the relational link graph, day-to-day workflows, and troubleshooting. README is the front-door pitch; everything operator-facing lives here.

Quick start assumes macOS with Docker, Rust (`rustc` 1.95+), `sqlx-cli`, and Ollama already installed.

## Install prerequisites

If you don't already have these, here's the canonical install for each:

- **Docker** — install [Docker Desktop](https://www.docker.com/products/docker-desktop/). GUI installer; no command needed.

- **Rust toolchain** — bootstrap via [rustup](https://rustup.rs/):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

- **`sqlx-cli`** — matches the workspace's `sqlx 0.8` + Postgres + rustls stack:

  ```bash
  cargo install sqlx-cli --no-default-features --features rustls,postgres
  ```

  `--no-default-features` keeps MySQL/SQLite codepaths out; `rustls` matches the workspace's TLS choice; `postgres` is the only DB Kengram uses. This gives you the `sqlx migrate run` command used in step 4 below, plus `cargo sqlx prepare` for regenerating the committed `.sqlx/` offline-query metadata after changing any `sqlx::query!` macro (see step 5).

- **Ollama** — `brew install ollama` on macOS (or download from [ollama.com](https://ollama.com/download)). The desktop app starts the daemon automatically; otherwise run `ollama serve`.

## Quick launch (recommended)

Four checked-in scripts at the repo root drive the dev stack with minimal typing. This is the recommended path; the step-by-step [Manual setup (advanced)](#manual-setup-advanced) below is the fallback when you need to run or customize an individual step.

**One-time prerequisites** (see [Install prerequisites](#install-prerequisites)): Docker, the Rust toolchain, `sqlx-cli`, and Ollama installed and running. Pull the models the scripts use:

```bash
ollama pull bge-m3                 # embeddings
ollama pull qwen2.5:7b-instruct    # tagging (worker, on by default)
```

| Script | What it does |
|---|---|
| `./start_stack.sh` | Brings up the backing containers (`postgres` + the `tei` reranker) and blocks until Postgres is ready. Pass `--tagger` to also start the opt-in deterministic tagger sidecar. Sets no env vars — service config lives in `docker-compose.yml`. |
| `./start_server.sh` | Runs `kengram serve` in the foreground (MCP server on `127.0.0.1:8080`, endpoint `/mcp`). |
| `./start_worker.sh` | Runs `kengram worker` in the foreground — drains `pending_embeddings` and (by default) `pending_tags`, tagging via local Ollama. |
| `./stop_stack.sh` | Stops the backing containers. Default keeps the containers and the Postgres data volume for a fast resume; `--down` removes the containers and network (the data volume is still preserved). |

**Two-terminal flow:**

```bash
./start_stack.sh                 # add --tagger only if using the sidecar tagger

# Terminal 1 — MCP server (foreground; Ctrl-C to stop):
./start_server.sh

# Terminal 2 — worker (foreground; Ctrl-C to stop):
./start_worker.sh

# When done — server/worker stop with Ctrl-C; halt the containers with:
./stop_stack.sh                  # add --down to also remove the containers
```

`start_stack.sh` exits once Postgres is ready (TEI keeps warming in the background — only reranked search waits on it). The server and worker both run in the foreground, so each wants its own terminal.

**Tagging is on by default** via local Ollama (`qwen2.5:7b-instruct`). For an embed-only worker, pass `off`:

```bash
./start_worker.sh off            # drains embeddings only; no tagging
```

Every tagger value is overridable from the environment without editing the script, e.g.:

```bash
KENGRAM_TAGGER__MODEL_NAME=qwen2.5:14b-instruct \
KENGRAM_TAGGER__MODEL_ID=ollama/qwen2.5:14b-instruct \
  ./start_worker.sh
```

> **Backfill note.** The worker only tags *newly enqueued* thoughts. If you captured thoughts before enabling the tagger, catch up once with:
> ```bash
> cargo run --bin kengram -- tag --rerun --since 1970-01-01T00:00:00Z
> ```

To use vLLM, OpenRouter, or the HTTP sidecar instead of Ollama, put the relevant `[tagger]` block in `~/.config/kengram/kengram.toml` (see [Configuration reference](#configuration-reference)); config-file settings and the script env defaults layer cleanly, with env winning.

---

## Manual setup (advanced)

You don't need these steps if you used [Quick launch (recommended)](#quick-launch-recommended) above — they're here for when you want to run, understand, or customize an individual step. The numbered steps below are exactly what the scripts automate.

### 1. Start Postgres in Docker

```bash
docker compose up -d postgres
```

This launches `pgvector/pgvector:pg16` with the `vector`, `pg_trgm`, and `pgcrypto` extensions available. The container is named `kengram-postgres`. Data lives in a named Docker volume (`kengram-pg-data`) and survives `docker compose down`.

Wait for it to be healthy:

```bash
docker compose ps postgres
# STATUS should show "healthy"
```

### 2. Set the database URL

```bash
export DATABASE_URL="postgres://kengram:kengram@localhost:5432/kengram"
```

Plain `DATABASE_URL` is read by `sqlx-cli` (for `sqlx migrate run` / `cargo sqlx prepare`) and by the build-time `sqlx::query!` macros — it is NOT read by the running kengram binary. At runtime the binary reads its database URL from config via figment: the `KENGRAM_` env prefix with `__` nesting (`KENGRAM_DATABASE__URL`), the `~/.config/kengram/kengram.toml` `[database].url` key, or the hardcoded default. The two carry the same value by default, so exporting `DATABASE_URL` to the connection string above keeps the toolchain and the binary in agreement. Add it to your shell rc if you don't want to set it every session.

`sqlx::query!` macros and the `sqlx::test` attribute both require `DATABASE_URL` to be set at *build time*, not just at runtime. The `.env` file at the workspace root is read by `sqlx-cli` but NOT by `cargo build` — set `DATABASE_URL` in your shell or pass it inline: `DATABASE_URL=... cargo build`.

### 3. Pull the embedding model in Ollama

Make sure the Ollama daemon is running (`ollama serve` if it isn't already; the macOS desktop app launches it automatically), then:

```bash
ollama pull bge-m3
```

Kengram's dev-mode embedder talks to Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1/embeddings`) and uses `bge-m3` for 1024-dim embeddings.

Verify:

```bash
curl http://localhost:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"hello"}' | jq '.data[0].embedding | length'
# expect: 1024
```

The configured `embedder.model_id = "bge-m3:1024"` carries the dimension as a suffix. That suffix is load-bearing: the HNSW vector index in Postgres is a partial index keyed on `(embedding_dim, model_id)`, and the embedder writes the `:NNNN` dim into the model_id field so the query planner can route lookups to the matching partial. If you change the embedding model, change the suffix in lockstep with the migration that adds the new partial index. See the troubleshooting section for the symptom when these drift.

### 3b. (Optional) Start TEI for the rerank stage

The cross-encoder reranker runs in a TEI Docker container alongside Postgres. It's optional — `kengram serve` works without it. The search pipeline silently skips the rerank stage when no `[reranker]` section is configured and the results come back in RRF + recency order with `rerank_used: false`.

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

Then add a `[reranker]` section to your `kengram.toml` (see Configuration below) and `kengram serve`'s startup log will show `reranker: resolved config`.

**Model choice.** `docker-compose.yml` pins `cross-encoder/ms-marco-MiniLM-L-6-v2` — the small (~22M parameter) dev reranker that has ONNX exports on HF (TEI takes the fast ORT path; sub-100ms per call on Apple Silicon CPU). For production with a GPU host, override via `[reranker].model_id` to `BAAI/bge-reranker-v2-m3` or another full-size model.

The Apple Silicon variant of the image (`cpu-arm64-latest`) is what's pinned. Production deployments use TEI as a systemd-managed sidecar, not Docker — same HTTP interface either way.

### 3c. (Optional) Start the deterministic tagger sidecar

If you want non-LLM tagging (the kengram-native HTTP-tagger pattern), the reference sidecar runs in docker-compose under the `tagger` profile. It's opt-in — `docker compose up -d` does NOT start it by default.

**Prerequisites.** Before first `docker compose --profile tagger up`:

1. Download the GLiNER ONNX model to `~/models/gliner_small-v2.1/`. See [`crates/kengram-tagger-deterministic/README.md`](crates/kengram-tagger-deterministic/README.md#1-download-the-gliner-onnx-model) for the curl invocations (~580MB).
2. Make sure Ollama is running with `bge-m3` available (the sidecar's default `EMBEDDER_ENDPOINT` points at the host's Ollama via `host.docker.internal:11434`).

```bash
docker compose --profile tagger up -d tagger-deterministic
# First build is slow (~5-10 min on Apple Silicon) — cargo compiles
# gline-rs + ort native deps. Subsequent builds are fast (cached layers).
docker compose --profile tagger ps tagger-deterministic
# STATUS should reach "healthy" within ~30s of starting.
```

Smoke:

```bash
curl -fsS http://localhost:8082/health
# expect: {"status":"ok"}

curl -sS -X POST http://localhost:8082/tag \
  -H 'Content-Type: application/json' \
  -d '{"protocol_version":"1","content":"Sarah pushed the bge-m3 reranker config."}' \
  | jq .
# expect: {"protocol_version":"1","tags":{"people":["Sarah"],...},"relations":[]}
```

Then flip `[tagger]` in your `~/.config/kengram/kengram.toml` to the http provider. Complete block (replace your existing `[tagger]` section with this):

```toml
[tagger]
provider = "http"
model_id = "deterministic/gliner-small-v2.1+regex+bge-m3"   # stamped onto thoughts.tags_extractor_model
model_version = 1                                            # the sidecar's schema version

[tagger.http]
endpoint = "http://localhost:8082"
timeout_seconds = 30
# api_key = "..."   # optional bearer; sidecars on a private network typically omit
```

The flat `[tagger]` fields openai-compatible uses (`endpoint`, `model_name`, `api_key`, `temperature`, `system_prompt_file`, `scope_vocab_*`) are ignored when `provider = "http"`. Leave them as-is — they'll quietly do nothing — or delete them entirely.

Restart the worker (and `kengram serve` if you want clean logs):

```bash
cargo run --bin kengram -- worker
# expect: "tagger: resolved config ... provider=http ..." in the startup logs
```

**Port note.** The sidecar's default host port is `8082` (the Tier 1 `kengram serve` convention is `:8081`, so the defaults coexist on one machine). If you've customized either to overlap, change one of them — `docker-compose.yml`'s `ports` line for the sidecar or `[server].bind` for kengram serve.

**To bring the sidecar down** without stopping the rest of the stack: `docker compose --profile tagger stop tagger-deterministic`. **To recreate after editing `topic-taxonomy.toml`**: `docker compose --profile tagger restart tagger-deterministic` (taxonomy is embedded once at startup, so a restart is required for new vectors to take effect). **To stop the whole stack and bring back only specific services**: `docker compose down` tears down everything regardless of profile; bring back with `docker compose --profile tagger up -d` (default-profile services + tagger) or omit the profile to leave the sidecar off.

**Switching back to the LLM tagger** is the same shape: change `provider = "openai-compatible"`, restore the LLM `model_id` / `model_version` / `endpoint` / `model_name` fields, restart the worker. The `[tagger.http]` block can stay in the file; it's ignored when `provider != "http"`.

### 4. Run migrations

On a fresh checkout, run migrations with `sqlx-cli` directly — no compilation needed:

```bash
sqlx migrate run
```

Once the workspace is built (step 5), `cargo run --bin kengram -- migrate` is the equivalent idempotent form.

The migration set (currently 11 numbered files in `migrations/`) ships the schema described in `DESIGN.md` §5, plus subsequent additions: thought retraction, the thought_links graph layer, polymorphic link targets, soft-delete + migration_audit, and the JSONB cleanup that removed the redundant `tags.relations` copy.

**Migration audit.** The `migration_audit` table (introduced in 0010) records what each migration did — `migration`, `ran_at`, `rows_touched`, optional `notes`. Convention going forward: any row-touching migration ends with an `INSERT INTO migration_audit (...)` statement so the operator can verify per-migration impact via `kengram audit migrations` rather than psql. Schema-only migrations should still insert an audit row with `rows_touched = 0` and a one-line `notes` summary. See [Operator workflows](#operator-workflows) for the `kengram audit migrations` walkthrough.

### 5. Build, test, run

```bash
cargo build --workspace
cargo test --workspace                       # unit + sqlx::test
cargo test --workspace --features integration   # adds a live-Ollama round-trip test

cargo run --bin kengram -- serve              # starts the MCP server on 127.0.0.1:8080
cargo run --bin kengram -- worker             # in a second shell — drains pending_embeddings + pending_tags
cargo run --bin kengram -- stats              # corpus + storage telemetry; operator-facing snapshot
cargo run --bin kengram -- audit migrations   # per-migration audit log
```

**Offline `sqlx::query!` validation.** The `.sqlx/` directory at the workspace root is committed; it holds per-query JSON metadata generated by `cargo sqlx prepare --workspace` and lets `cargo build` succeed without a live database when `DATABASE_URL` is unset. If you've set `DATABASE_URL` and want to skip the live-DB round-trip anyway (e.g. you've changed branches and don't want to migrate first), set `SQLX_OFFLINE=true` and the macros will read `.sqlx/` instead.

When you change a `sqlx::query!` / `sqlx::query_as!` macro — adding a column, changing a JOIN, or introducing a new query — regenerate the metadata against your live DB and commit the updated `.sqlx/` files:

```bash
cargo sqlx prepare --workspace
git add .sqlx/
```

CI (when it lands at M7) will enforce that `.sqlx/` is up to date.

Point an MCP-capable client (Claude Code, Claude Desktop, `mcp-inspector`) at `http://127.0.0.1:8080/mcp` (streamable-HTTP transport, per the current MCP spec). Nine tools are exposed:

- `capture` — write a thought; returns `thought_id`, `embedding_status: "pending"`, and `is_duplicate`. Same content captured twice (SHA-256 fingerprint match) returns the existing `thought_id`.
- `search_thoughts` — RRF-fused vector + trigram retrieval over thoughts; recency-boosted; optional cross-encoder rerank; optional `tag_filter` JSONB-containment filter (e.g. `{"kind": "task"}`); `scope` (exact) or `scope_prefix` (namespace) for scope filtering. Each hit carries its `tags` object.
- `recent_thoughts` — chronological browse; supports `scope` or `scope_prefix`.
- `list_scopes` — discover what scopes are in use (optionally `prefix`-filtered). Pair with `scope_prefix` on the search/recent tools for a discover-then-query workflow.
- `get_thought` — full thought + provenance + tags + tagger provenance.
- `retract_thought` — mark a thought as untrusted (excluded from retrieval; still visible via `get_thought` for audit).
- `link_thoughts`, `unlink_thoughts`, `get_related_thoughts` — the graph layer. See [Relational data and link graph](#relational-data-and-link-graph).

`kengram serve` and `kengram worker` are paired: `serve` writes thoughts and enqueues embedding + tag jobs; `worker` drains both queues (`pending_embeddings` and `pending_tags`). Running `serve` without `worker` is fine — thoughts are still durable and trigram-searchable — but vector kNN won't surface them and tags stay empty until the worker runs. When `[tagger].provider` is empty, the tag-job enqueue at capture is a no-op and the tag drainer doesn't spawn.

## Common operations

```bash
# Stop the database (data persists)
docker compose down

# Stop and wipe the database
docker compose down -v

# Open a psql session in the container
docker exec -it kengram-postgres psql -U kengram -d kengram

# Tail Postgres logs
docker compose logs -f postgres
```

### Embed-backfill

Heal-then-drain: enqueue any unembedded thoughts that lack a queue row (pre-M2 captures, or captures whose enqueue lost a crash race), then drain the queue inline. Use this if you've been running `serve` without `worker` and want to catch up without spinning up the worker, or after embedder downtime to drain the backlog.

```bash
# Whole corpus, up to 1000 rows.
cargo run --bin kengram -- embed-backfill --limit 1000

# One scope only (exact match).
cargo run --bin kengram -- embed-backfill --scope work --limit 100

# A namespace of scopes (prefix match). Mutually exclusive with --scope.
cargo run --bin kengram -- embed-backfill --scope-prefix kengram. --limit 500
```

`--scope` and `--scope-prefix` are mutually exclusive. Empty strings on either flag are normalised to "no filter."

### One-shot tagger run

Like a single tick of the worker's tag drainer. Tags thoughts where `tags_extractor_version IS NULL`. Requires a configured `[tagger]` section. Useful for catching up after capturing a batch of thoughts before enabling the tagger.

```bash
cargo run --bin kengram -- tag --limit 50
cargo run --bin kengram -- tag --scope work --limit 100
cargo run --bin kengram -- tag --scope-prefix kengram. --limit 200
```

### Re-tag after tagger version bump

Re-run the tagger over thoughts whose stored `tags_extractor_version` is below the configured current version. Use this after bumping `[tagger].model_version` (typically after a prompt or schema change). Tags are overwritten in place — no supersede semantics, no audit chain. Pair with `--since` to bound the rerun to recent thoughts; use `--since 1970-01-01T00:00:00Z` to re-tag the entire corpus.

If you **switched the tagger model** without bumping the prompt version, the stored version isn't actually lower, so `--rerun` skips those rows. Use `--force` to re-tag every matching thought regardless of version — it re-stamps the configured `model_version` and records the new `model_id`. Bound it with `--scope` / `--scope-prefix` / `--since` / `--limit`.

```bash
cargo run --bin kengram -- tag --rerun --scope work
cargo run --bin kengram -- tag --rerun --scope-prefix kengram. --since 2026-04-01T00:00:00Z
cargo run --bin kengram -- tag --rerun --since 1970-01-01T00:00:00Z   # whole corpus
cargo run --bin kengram -- tag --force --scope work                  # re-tag regardless of version (e.g. after a model swap)
```

If you've pinned `model_version` in your local `~/.config/kengram/kengram.toml`, bump it manually. The new bundled default (currently 13) only applies when the field is absent from your TOML. The log line at startup reports the resolved value: look for `target_version=13`. If it says `target_version=N` with `N < 13`, your config still overrides; either update the line or delete it.

For the procedural detail and the full v1→v13 changelog, see [Tagger version history and safe re-tag procedure](#tagger-version-history-and-safe-re-tag-procedure).

### Reranker A/B benchmark

A/B-benchmark the reranker against RRF-only on an operator-curated fixture corpus. Prints a markdown table to stdout with per-query nDCG@10 and MRR for both rankings, plus an AVERAGE row. Requires a configured `[reranker]` section in `kengram.toml` and the corpus's `relevant_ids` to point at real `thought_id` rows in your DB. See `tests/fixtures/bench-rerank.example.json` for the schema.

```bash
cargo run --bin kengram -- bench rerank --corpus ~/.kengram/my-bench.json
```

## Configuration reference

Defaults live in code. Override via `~/.config/kengram/kengram.toml`, a `--config <path>` argument, or `KENGRAM_*` env vars (nested via `__`, e.g. `KENGRAM_DATABASE__URL`). Layering order: defaults → user TOML → `--config` TOML → env. Later wins.

Example `kengram.toml` (every knob spelled out — most can be omitted to take the default):

```toml
[server]
bind = "127.0.0.1:8080"
allowed_hosts = []                                      # see below

[database]
url = "postgres://kengram:kengram@localhost:5432/kengram"
max_connections = 10

[embedder]
provider = "openai-compatible"
endpoint = "http://localhost:11434/v1"                  # Ollama for dev; TEI in production
model = "bge-m3"
model_id = "bge-m3:1024"                                # must match an HNSW partial index
dimensions = 1024
# api_key = "sk-..."
timeout_seconds = 5

[reranker]                                              # opt-in
provider = "tei"                                        # "" = silent-disable
endpoint = "http://localhost:8080"                      # no /v1 suffix; reranker appends /rerank
model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
timeout_seconds = 30

[tagger]                                                # opt-in
provider = "openai-compatible"                          # "" = silent-disable; "openai-compatible" / "openrouter" / "http"
endpoint = "http://localhost:8000/v1"                   # vLLM default; ignored when provider = "http"
model_name = "qwen2.5-7b-instruct"                      # the model the backend serves
model_id = "vllm/qwen2.5-7b-instruct"                   # provenance written into thoughts.tags_extractor_model
model_version = 13                                      # tracks BUNDLED_TAGGER_VERSION; see Tagger version history
# api_key = ""
timeout_seconds = 60
temperature = 0.2
scope_vocab_enabled = true
scope_vocab_size = 50
# system_prompt_file = "~/.config/kengram/tagger-prompt.txt"

# HTTP-sidecar backend (provider = "http"). Points kengram at any tagger
# sidecar speaking the kengram-tagger-protocol wire shape. The reference
# implementation is crates/kengram-tagger-deterministic/ (Rust-native, no
# LLM); operators can also ship sidecars in Python, Go, etc. See
# docs/tagger-backends.md + docs/tagger-sidecar-protocol.md.
# [tagger.http]
# endpoint = "http://localhost:8082"
# timeout_seconds = 30
# api_key = ""                                          # optional bearer

[worker]
tick_interval_seconds = 5
batch_size = 16
```

Env override examples: `KENGRAM_WORKER__TICK_INTERVAL_SECONDS=2 cargo run --bin kengram -- worker` (snappier ticks for development), `KENGRAM_TAGGER__API_KEY=sk-...` (OpenRouter key without checking it into config), `KENGRAM_TAGGER__PROVIDER="" cargo run --bin kengram -- serve` (silent-disable the tagger for a run).

### `[server]`

| knob | default | what it does |
|---|---|---|
| `bind` | `"127.0.0.1:8080"` | Listen address. Tier 0 (localhost) is the default. Tier 1 (Tailnet) is a non-loopback bind — set this to the Tailscale interface IP or `0.0.0.0:<port>`. No code change required. |
| `allowed_hosts` | `[]` (use rmcp's safe default) | Host names / IPs the MCP server's DNS-rebinding protection accepts on the `Host` header. Empty = rmcp default (`localhost` / `127.0.0.1` / `::1`). A non-empty list REPLACES the default. |

When binding non-loopback, include both the bare hostname AND `hostname:port` forms the client uses, plus IP and `ip:port` forms — the rmcp matcher checks both. Leaving this list empty when bind is non-loopback effectively rejects every non-localhost request; the symptom is "rejected request with disallowed Host header" warnings in the serve log. Bypass-all is intentionally not exposed — Tailnet ACLs plus an explicit allowlist is Tier 1 auth.

```toml
[server]
bind = "0.0.0.0:8081"
allowed_hosts = [
    "localhost", "127.0.0.1", "::1",
    "repromax", "repromax:8081",
    "100.110.75.74", "100.110.75.74:8081",
]
```

### `[database]`

| knob | default | what it does |
|---|---|---|
| `url` | `"postgres://kengram:kengram@localhost:5432/kengram"` | Postgres connection string. Single-tenant; one database per Kengram deployment. |
| `max_connections` | `10` | Size of the sqlx connection pool. Bump for high-tag-volume worker hosts; single-user dogfood is fine at 10. |

### `[embedder]`

| knob | default | what it does |
|---|---|---|
| `provider` | `"openai-compatible"` | Only provider in current builds. Covers Ollama, TEI, OpenAI, Voyage by varying `endpoint`/`model`. |
| `endpoint` | `"http://localhost:11434/v1"` | `/v1` base URL. Ollama in dev; TEI in production. |
| `model` | `"bge-m3"` | Backend model name as the server understands it. |
| `model_id` | `"bge-m3:1024"` | Kengram-side stable identity. The `:NNNN` suffix is the embedding dimension — must match an HNSW partial index in Postgres. Change this only in lockstep with a migration. |
| `dimensions` | `1024` | Embedding vector dimension. Must match the model's actual output dim AND the `:NNNN` suffix above. |
| `api_key` | `None` | Bearer token for hosted endpoints. Omit for Ollama/TEI. |
| `timeout_seconds` | `5` | Per-request timeout. Local Ollama is sub-100ms typical; bump for slower hosted endpoints. |

### `[reranker]`

The reranker is the optional cross-encoder stage that re-scores the top `candidate_pool` post-RRF hits. Empty `provider` silent-disables — the search pipeline falls through to RRF + recency.

| knob | default | what it does |
|---|---|---|
| `provider` | `""` | `""` = disabled; `"tei"` = TEI sidecar (currently the only supported provider). |
| `endpoint` | `"http://localhost:8080"` | Service root, no `/v1` suffix. The reranker client appends `/rerank`. |
| `model_id` | `"BAAI/bge-reranker-v2-m3"` | Kengram-side stable identity. Dev override: `"cross-encoder/ms-marco-MiniLM-L-6-v2"` (matches the docker-compose pin). |
| `timeout_seconds` | `30` | Per-request timeout. MiniLM is sub-100ms on Apple Silicon; BGE-v2-m3 on GPU is similar; ARM CPU runs of BGE-v2-m3 are minutes per call (don't). |

When the reranker times out or errors, the pipeline silently degrades to RRF + recency order and the response has `rerank_used: false`. Search still returns results.

### `[tagger]`

The tagger is the per-thought metadata sidecar. Empty `provider` is the silent-disable sentinel: capture proceeds, no tag jobs enqueue, the worker doesn't spawn a tag drainer. Flip `provider = "openai-compatible"` (LLM via vLLM / Ollama / OpenRouter) or `"http"` (kengram-native HTTP sidecar, any language) to enable. See [`docs/tagger-backends.md`](docs/tagger-backends.md) for the pluggability contract.

| knob | default | what it does |
|---|---|---|
| `provider` | `""` | `""` = disabled; `"openai-compatible"` (vLLM, etc.), `"openrouter"`, or `"http"` (kengram-native sidecar — requires `[tagger.http]` below). |
| `endpoint` | `"http://localhost:8000/v1"` | `/v1` base URL. vLLM default port. OpenRouter is `"https://openrouter.ai/api/v1"`. Ignored when `provider = "http"`. |
| `model_name` | `"qwen2.5-7b-instruct"` | Model name as the backend understands it. For OpenRouter: a model slug like `"anthropic/claude-haiku-4.5"`. Ignored when `provider = "http"`. |
| `model_id` | `"vllm/qwen2.5-7b-instruct"` | Kengram-side stable identity written into `thoughts.tags_extractor_model`. Conventionally `<vendor>/<model>`. Used by both LLM and HTTP-sidecar providers. |
| `model_version` | `13` | Tracks `kengram_extract::BUNDLED_TAGGER_VERSION`. Written into `thoughts.tags_extractor_version`. Bump when the prompt or schema changes such that prior tags shouldn't be considered comparable; then `kengram tag --rerun`. See [Tagger version history](#tagger-version-history-and-safe-re-tag-procedure). |
| `api_key` | `None` | Bearer token for hosted LLM endpoints. The HTTP sidecar provider has its own `[tagger.http].api_key`. |
| `timeout_seconds` | `60` | Per-request timeout for the LLM provider. The HTTP sidecar provider has its own `[tagger.http].timeout_seconds`. |
| `temperature` | `0.2` | Generation temperature. Lower = more deterministic. 0 makes some backends loop. LLM provider only. |
| `system_prompt_file` | `None` | Path to a file containing a custom system prompt. `None` = use `BUNDLED_TAGGER_PROMPT`. Operators who supply a custom prompt are responsible for also bumping `model_version` so provenance stays meaningful; a WARN log is emitted at startup. LLM provider only. |
| `scope_vocab_enabled` | `true` | Inject the top topic + entity terms from the thought's scope into the tagger prompt as a controlled-vocabulary hint. Encourages consistent term reuse across captures. LLM provider only. |
| `scope_vocab_size` | `50` | Top-N established terms (each for topics and entities) fed to the tagger. Larger = more vocabulary stability; smaller = faster emergence of new terms. LLM provider only. |

### `[tagger.http]`

Active only when `[tagger].provider = "http"`. Kengram POSTs `/tag` against the sidecar's `endpoint` using the [`kengram-tagger-protocol`](crates/kengram-tagger-protocol/) wire shape. Sidecars can be in any language; the reference implementation is [`kengram-tagger-deterministic`](crates/kengram-tagger-deterministic/) (a Rust-native zero-LLM tagger). See [`docs/tagger-sidecar-protocol.md`](docs/tagger-sidecar-protocol.md) for the wire contract.

| knob | default | what it does |
|---|---|---|
| `endpoint` | `"http://localhost:8082"` | Base URL of the sidecar. The client appends `/tag` to this. Default-coexists with the Tier 1 `kengram serve` convention at `:8081`. |
| `api_key` | `None` | Optional bearer token sent as `Authorization: Bearer <token>` to the sidecar. |
| `timeout_seconds` | `60` | Per-request timeout. Sidecars doing CPU inference can run long on first call. |

### `[worker]`

| knob | default | what it does |
|---|---|---|
| `tick_interval_seconds` | `5` | How often the embed and tag drainers wake up and claim a batch off their respective queues. 5s is fine for single-user dogfood; tune lower for snappier vector-search readiness, higher to be gentler on the backends. |
| `batch_size` | `16` | Max jobs claimed per tick (per queue). Bigger batches are kinder to the backend; smaller batches mean shorter critical sections and faster failover when a job hangs. |

## Tagger version history and safe re-tag procedure

The tagger's prompt + JSON schema is versioned by `kengram_extract::BUNDLED_TAGGER_VERSION` (currently **13** for the openai-compatible LLM backend). Each thought row carries a `tags_extractor_version` recording the version it was tagged under, so the drainer can identify stale rows when the version is bumped.

The deterministic HTTP-sidecar backend has its own independent version line (currently **1**) stamped via the sidecar's `MODEL_VERSION` env var. Re-tagging across backends works the same way (`kengram tag --rerun`) but the version comparison is per-`tags_extractor_model` — a row stamped by the LLM backend isn't "stale" relative to the deterministic backend's version 1.

### Version changelog (LLM backend)

The full prompt-iteration history (v1 through v13, plus the v14 deterministic-backend transition) lives at [`docs/tagger-improvements.md`](docs/tagger-improvements.md) — the canonical source of decisions, dogfood evaluations, and rationale. Brief summary:

- **v1–v4** (M4, 2026-05-16/17). Initial tagger + entities/topics split + iterative prompt cleanup.
- **v5** (M6.1, 2026-05-17). Added tagger-extracted relations into `thought_links`.
- **v6–v9** (post-M6.1 dogfood, 2026-05-18). Kind classification rebalance, NOT-entities-list iteration, topics-as-concept-mapping, `tags.relations` dropped from persisted JSONB (migration 0011).
- **v10–v13** (2026-05-22/23). Scope-vocab experiment, topic canonical-form normalization moved to post-process, people↔entities disjointness validator, USE-vs-MENTION discipline added to the prompt. v13 is the current bundled default.
- **v14** (2026-05-24). Not a prompt bump — the pluggability framework + reference HTTP-sidecar tagger (`kengram-tagger-deterministic`) shipped. LLM backend default unchanged; deterministic backend is opt-in via `provider = "http"` per the `[tagger.http]` config recipe in Section 3c above.

See [`docs/tagger-backends.md`](docs/tagger-backends.md) for the pluggability contract, [`docs/tagger-sidecar-protocol.md`](docs/tagger-sidecar-protocol.md) for the HTTP-sidecar wire spec, and [`docs/tagger-improvements.md`](docs/tagger-improvements.md) for the v14 head-to-head measurement and rollout rationale.

### Safe re-tag procedure

After bumping the tagger version (or the bundled default rolls forward and you want the corpus on the new schema):

1. **Verify the resolved target version.** Start `kengram serve` (or `kengram worker`). The startup log line is:
   ```
   tagger: resolved config ... model_version=13 ...
   ```
   And on the re-tag side, `kengram tag` prints:
   ```
   kengram tag starting ... target_version=13 ...
   ```
   If `target_version` is lower than expected, your `~/.config/kengram/kengram.toml` is overriding the bundled default. Bump it manually or delete the `model_version` line so the bundled default takes over.

2. **Re-tag the corpus.** Whole corpus:
   ```bash
   cargo run --bin kengram -- tag --rerun --since 1970-01-01T00:00:00Z
   ```
   The drainer walks rows where `tags_extractor_version < target_version`. Bound it tighter if you only want a recent window — `--since 2026-04-01T00:00:00Z` or `--scope-prefix kengram.`.

   **Cross-backend / model-swap retag note.** If you're switching providers or models (e.g. flipping from `openai-compatible` to `http`, or to a stronger model) without bumping the prompt version, the version comparison no longer marks the old rows as "stale" — a row stamped `tags_extractor_version = 13` isn't lower than the new backend's `model_version`, so `--rerun` skips it. Use `--force` to re-tag every matching thought regardless of version, bounded by `--scope` / `--scope-prefix` / `--since` / `--limit`: `kengram tag --force --since 1970-01-01T00:00:00Z` for the whole corpus. The refreshed `tags_extractor_model` records which backend did the re-tag.

3. **Monitor the worker logs for failures.** The `kengram tag complete` line reports `n_candidates`, `tagged`, `failed`. Non-zero `failed` exits non-zero so cron / scripts can detect partial failures; per-row errors are logged at WARN with the `thought_id`.

4. **Spot-check.** psql is fine for this; there's no dedicated CLI:
   ```sql
   SELECT id, tags_extractor_version, tags_extractor_model
   FROM thoughts
   WHERE tags_extractor_version IS NOT NULL
   ORDER BY tags_extracted_at DESC
   LIMIT 10;
   ```
   If the change was a migration (e.g. v9's 0011), use `kengram audit migrations` instead — see [Operator workflows](#operator-workflows).

5. **Note for `tag_filter` consumers.** Agents that hardcoded `tag_filter` queries against an earlier prompt shape may need updating. For example, descriptive phrases like `agent memory protocol` or `cross-encoder` that v2 sometimes landed in `entities` are now consistently routed to `topics` (v3+); queries on `entities` will miss those thoughts. Migrate `tag_filter` to use `topics` for descriptive-phrase searches.

## Relational data and link graph

The graph layer is anchored on the `thought_links` table. Every edge has a `from_thought_id` (always a thought), a `relation` from a closed vocabulary, and a polymorphic target that's either another thought or a typed non-thought (entity, person, URL). Edges are agent-supplied (via `link_thoughts`) or tagger-emitted; both kinds live in the same table, distinguished by a `link_source` column (`agent` vs `tagger`).

### Closed relation vocabulary

Seven relations. One-sentence semantics:

- **`replaces`** — this thought replaces an earlier one; the older one is no longer the current thinking. Most recent supersedes.
- **`requires`** — this thought depends on another. (Prerequisite, precondition, blocking dependency.)
- **`references`** — this thought points at another for context, like a passive citation. Mention without endorsement.
- **`supports`** — this thought makes a claim that actively confirms a claim made in another. Direction is `FROM=confirmer, TO=claim-maker`. (Added M5.1 to split active corroboration from passive cite.)
- **`belongs_to`** — this thought is a member or sub-element of another (a finding under a parent thread, a probe under an experiment).
- **`decided_by`** — this thought is a decision attributed to another (a person-note or session-anchor).
- **`refines`** — this thought is a refinement / iteration of an earlier one. Both still stand; the newer one represents updated thinking on the same proposition.

The vocabulary is enforced as a Postgres `CHECK` constraint (TEXT, not `ENUM`, to keep extensions cheap — `supports` was added by migration 0008 with a constraint-relax).

### Polymorphic targets

The `to_kind` column discriminates between four target shapes. Exactly one per-kind column is set per row, enforced by a `target_valid` `CHECK`:

- **`thought`** — `to_thought_id` is a UUID into `thoughts(id)`. Use when the target was captured as its own thought.
- **`entity`** — `to_entity` is a free-text name. Use for experiments, projects, sessions, abstract concepts that aren't worth capturing as a thought of their own.
- **`person`** — `to_person` is a free-text name. Use for attribution (`decided_by` Ron, etc.).
- **`url`** — `to_url` is `http://...` or `https://...` (lightweight format `CHECK`). Use for external resources.

The generated `to_value` column (`COALESCE(to_thought_id::text, to_entity, to_person, to_url)`) anchors the unique-edge index across all four kinds.

### MCP tools

Three tools on the graph layer. Full request/response schemas are documented in the MCP server instructions (the `SERVER_INSTRUCTIONS` constant in `crates/kengram-mcp/src/server.rs`); one-paragraph overview each:

- **`link_thoughts(from_thought_id, relation, {to_thought_id | to_entity | to_person | to_url}, note?)`** — Assert an edge from a thought to a polymorphic target. Supply exactly one of the four target fields. Returns `is_new` + `link_id` + the `to_kind`/`to_value` discriminator. Idempotent on the `(from, relation, to_kind, to_value)` quadruple: re-asserting a live edge returns `is_new=false`. If the edge was previously soft-deleted, a fresh live row is inserted and `is_new=true`. Validates target existence + the no-self-reference rule for thought targets.
- **`unlink_thoughts(from_thought_id, relation, {one-of-four-targets})`** — Soft-delete a link by its `(from, relation, target)` triple. Returns a three-way `status`: `deleted_now`, `already_deleted`, or `never_existed`. Soft-deleted edges sit inert in the table; re-creating the same edge via `link_thoughts` succeeds (fresh row).
- **`get_related_thoughts(thought_id, relations?, target_kinds?, direction?)`** — Walk the graph from a single thought. Returns grouped `outbound` (edges where this thought is `from`) and `inbound` arrays. Each entry carries the edge's `link_id`, `relation`, `to_kind`, `to_value`, `link_created_at`, `link_source` (`agent` or `tagger`), `note`, plus — when `to_kind = thought` — the target thought's full metadata (preview, scope, retracted flag). For non-thought targets those thought-specific fields are null. Retracted thoughts surface with `retracted: true` so the caller can decide whether to show, dim, or hide them. Soft-deleted edges are excluded.

The `search → get_related_thoughts` pattern is the canonical discovery walk: after a search hit, fetch its graph neighbours to see the relational context (what refines it, what it supports, what cited it).

### Idempotency

The `(from_thought_id, relation, to_kind, to_value)` quadruple is unique among live (non-soft-deleted) rows, enforced by a partial unique index (`thought_links_unique_edge`). Re-asserting the same edge is a no-op that returns the existing row. This matches `capture`'s SHA-256 fingerprint dedup: the operator can replay a write without checking for prior state.

### Soft-delete semantics

Migration 0010 added `deleted_at TIMESTAMPTZ NULL` to `thought_links`. `unlink_thoughts` sets `deleted_at = NOW()` rather than `DELETE`-ing. This buys two things:

1. **Three-way status discriminator.** `unlink_thoughts` can distinguish "I just removed it" (`deleted_now`) from "it was already gone" (`already_deleted`) from "no edge with this triple ever existed" (`never_existed`). Pre-0010, those last two were indistinguishable.
2. **Reversible removals.** Re-linking after a soft-delete inserts a fresh live row — the partial unique index ignores soft-deleted rows. The old row stays in place for audit.

The `thought_links_deleted_at_idx` partial index (`WHERE deleted_at IS NOT NULL`) keeps "find soft-deleted edges" diagnostic queries cheap.

### Tagger-emitted edges

When the tagger (v5+) extracts relational claims from prose, the worker drainer writes them to `thought_links` with `source = 'tagger'`. They appear in `get_related_thoughts` alongside agent-supplied edges; the caller distinguishes via `link_source`.

On re-tag, the drainer first soft-deletes the prior tagger edges from that thought before applying fresh emissions. Agent edges are unaffected — only `source = 'tagger'` rows are touched. This makes re-tagging idempotent at the edge level: prompt iteration can't accumulate stale tagger-emitted edges.

Tagger-emitted edges target non-thought endpoints only (`entity` / `person` / `url`); the LLM doesn't synthesise UUIDs. To express a thought-to-thought relation, the agent calls `link_thoughts` directly.

## Operator workflows

Day-to-day patterns. Each one assumes `kengram serve` is running and reachable.

### Discover-then-query (scopes)

Before capturing into a new scope, see what's already in use. Before searching across a namespace, see what scopes are under that namespace.

```text
list_scopes()                                  # all scopes, most-recently-used first
list_scopes(prefix="kengram.")                  # namespace discovery
search_thoughts(query="...", scope_prefix="kengram.")    # query across the namespace
recent_thoughts(scope_prefix="kengram.", limit=20)       # browse a namespace
```

The CLI equivalent for the search side: `kengram embed-backfill` and `kengram tag` both accept `--scope-prefix`. The shell-side flag for `kengram stats` is `--scope-prefix` (filters the scopes summary section).

### Capacity audit

```bash
cargo run --bin kengram -- stats
cargo run --bin kengram -- stats --scope-prefix kengram. --top-scopes 50
```

Output is a sectional plain-text report:

- **Corpus.** Live and retracted thought counts (and retracted %), untagged count, content bytes total + average per thought, embeddings broken out by `(count × model_id, dim, version)`, links live + soft-deleted with breakdowns by relation / target kind / source.
- **Queues.** `pending_embeddings` and `pending_tags` row counts. These are the *current backlog* — what hasn't drained yet. Diverges from the on-disk-tables section: queues show pending rows, on-disk tables show allocated bytes.
- **Scopes.** Per-scope thought count + last-activity date, sorted by recency. Truncated to `--top-scopes` (default 20); summary line says how many are hidden.
- **On-disk tables.** Per-table heap / index / total bytes, plus a corpus-wide total. This is Postgres's view of how much space each table is using (heap + all indexes). It will diverge from "row count × content size" because of TOAST, dead tuples awaiting vacuum, index overhead, and the `embeddings` table's pgvector storage.

A non-zero `pending_embeddings` count with on-disk-tables that match expectations means the worker is behind. A growing on-disk-tables size with no corresponding row growth means dead-tuple accumulation — `VACUUM` is the answer.

### Re-tag after tagger version bump

See [Tagger version history and safe re-tag procedure](#tagger-version-history-and-safe-re-tag-procedure) for the procedure. Short version: verify `target_version` in the startup log, `kengram tag --rerun --since 1970-01-01T00:00:00Z`, watch the WARN logs for per-row failures, spot-check.

### Migration audit

```bash
cargo run --bin kengram -- audit migrations
cargo run --bin kengram -- audit migrations --since 2026-05-01T00:00:00Z --limit 20
```

Prints one entry per migration (most recent first), with `ran_at` timestamp, `rows_touched`, the migration filename, and the free-text `notes` column on a second line. Use this:

- After running `kengram migrate` — confirm each pending migration ran and touched the expected row count.
- When a migration's effect is in question — `rows_touched > 0` means data changed; `notes` describes why.
- For audit / forensics — the table is append-only (one row per `sqlx migrate run` per migration), so it's a complete record of schema evolution since 0010.

### Embedder unreachable

`search_thoughts` falls through to trigram-only when the embedder won't answer. The response has `vector_search_available: false`; results still come back, just from the lexical leg. This is not an error.

Meanwhile, the worker keeps `pending_embeddings` rows pinned. When the embedder recovers, the drainer picks them up on the next tick. To force a heal-then-drain without waiting: `kengram embed-backfill`.

If thoughts captured during the outage didn't get a queue row at all (a crash race between capture and enqueue), `embed-backfill` finds them too — it walks `thoughts` left-joined against the embeddings table and re-enqueues anything missing for the active model.

### Tagger silent-disable

`[tagger].provider = ""` is the silent-disable sentinel. Capture proceeds normally; the tag-job enqueue is a no-op and the worker doesn't spawn a tag drainer. Thoughts go in with `tags_extractor_version = NULL`.

To enable later: set `provider = "openai-compatible"` (or `"openrouter"`), bring up the backend, restart `serve` + `worker`, and run `kengram tag --rerun --since 1970-01-01T00:00:00Z` to catch the backlog. The serve startup log line reports which state it resolved to:

```text
kengram serve started ... tagger=disabled
# or
kengram serve started ... tagger=enabled (vllm/qwen2.5-7b-instruct)
```

### Embed-backfill after embedder downtime

```bash
cargo run --bin kengram -- embed-backfill --limit 1000
cargo run --bin kengram -- embed-backfill --scope work --limit 100
cargo run --bin kengram -- embed-backfill --scope-prefix kengram. --limit 500
```

Heal-then-drain: finds thoughts missing an embedding row for the active model, enqueues them if they aren't already, then drains the queue inline. `--limit` caps the run so a huge backlog doesn't pin the embedder for an hour; iterate until the queue clears.

Exit code is non-zero on partial failure (`failed > 0`), suitable for cron-style retry.

### Migrating between machines

`kengram backup` and `kengram restore` wrap `pg_dump` / `pg_restore` with a `manifest.json` sidecar (kengram version, schema head version, embedder model, tagger version, corpus counts). Restore validates the manifest against the target before touching anything destructive.

**Prereq:** Postgres client tools on PATH on both source and target (`brew install postgresql@16` on macOS; `apt install postgresql-client-16` on Debian/Ubuntu). `pg_dump` and `pg_restore` are the same binaries the Postgres server ships; the client-only package suffices.

**Source machine.** Back up the corpus:

```bash
cargo run --bin kengram -- backup
# → ./kengram-backup-2026-05-19T01-02-46-Z.tar.gz (308.3 KiB)
#   schema:    11_drop_tags_relations
#   thoughts:  42 live, 10 retracted
#   embeddings: 52
#   links:     96 live
#   scopes:    5
#   embedder:  bge-m3:1024 (1024d)
#   tagger:    vllm/qwen2.5-7b-instruct v13
```

Defaults to `./kengram-backup-<timestamp>.tar.gz`; override with `--to <path>`. Use `--skip-embeddings` to drop embedding rows from the archive (smaller backup; restore requires `kengram embed-backfill` to repopulate vectors; HNSW index survives an empty table).

**Transfer.** Plain file move. Over Tailnet:

```bash
rsync -avP ./kengram-backup-*.tar.gz ron@target.tailnet.ts.net:/tmp/
```

**Target machine.** Prereqs (per the [Install prerequisites](#install-prerequisites) section). On a fresh box: install Docker / Rust / sqlx-cli / Ollama, clone the repo, `docker compose up -d postgres`, then bring the schema to head:

```bash
sqlx migrate run
```

Then restore. On an empty target the `--force` flag is unnecessary; on a target with existing thoughts, `--force` is required and the command first prints a dry-run summary:

```bash
cargo run --bin kengram -- restore --from /tmp/kengram-backup-*.tar.gz
# Empty target → proceeds; prints "restored from ... Run `kengram stats` to verify."

cargo run --bin kengram -- restore --from /tmp/kengram-backup-*.tar.gz --force
# Non-empty target → required; replaces existing data.
```

**Compatibility checks** run before any destructive operation (skip with `--skip-version-check` only when you understand the implications):

| Mismatch | Outcome |
|---|---|
| Target schema head < source | Refuses. Run `sqlx migrate run` on target first. |
| Target schema head > source | Refuses. Restore on a matching kengram version, or use `--skip-version-check`. |
| Embedder `model_id` / dimensions differ | Warns only — embeddings restore as-is; run `kengram embed-backfill` after if you want to recompute under the new model. |
| Tagger `model_id` or `version` differs | Warns only — tags restore as-is; `kengram tag --rerun --since 1970-01-01T00:00:00Z` to refresh. |

**Verify after restore:**

```bash
cargo run --bin kengram -- stats
# Counts should match the manifest summary printed by `kengram backup`.
```

**Docker-Postgres vs systemd-Postgres** on the target — no practical difference for backup/restore. Both speak the same network Postgres protocol that `pg_dump` and `pg_restore` use; the only thing that has to match is the `DATABASE_URL` (or `KENGRAM_DATABASE__URL` env override).

## Configuration presets and troubleshooting

### Preset: vLLM-local tagger (dev)

Local vLLM serving qwen2.5-7b-instruct on port 8000:

```toml
[tagger]
provider = "openai-compatible"
endpoint = "http://localhost:8000/v1"
model_name = "qwen2.5-7b-instruct"
model_id = "vllm/qwen2.5-7b-instruct"
model_version = 13
timeout_seconds = 60
temperature = 0.2
scope_vocab_enabled = true
scope_vocab_size = 50
```

vLLM's JSON-Schema-constrained generation occasionally takes 5–10s on first-token latency for a cold model; the 60s timeout has headroom.

### Preset: Ollama embedder (dev)

Local Ollama serving bge-m3 on port 11434:

```toml
[embedder]
provider = "openai-compatible"
endpoint = "http://localhost:11434/v1"
model = "bge-m3"
model_id = "bge-m3:1024"
dimensions = 1024
timeout_seconds = 5
```

Ensure `ollama serve` (or the macOS desktop app) is running and `ollama pull bge-m3` has completed. The `:1024` suffix on `model_id` must match the HNSW partial index dimension.

### Preset: OpenRouter tagger (cloud fallback)

OpenRouter as a fallback for when local vLLM isn't reachable (e.g. on the road):

```toml
[tagger]
provider = "openrouter"
endpoint = "https://openrouter.ai/api/v1"
model_name = "anthropic/claude-haiku-4.5"
model_id = "openrouter/anthropic/claude-haiku-4.5"
model_version = 13
timeout_seconds = 30
temperature = 0.2
```

Set the API key out-of-band via env to keep it out of TOML:

```bash
export KENGRAM_TAGGER__API_KEY="sk-or-v1-..."
```

The `model_id` prefix is the convention — `openrouter/<slug>` — so provenance reads cleanly when looking at `thoughts.tags_extractor_model` later.

### Preset: TEI reranker

Dev (MiniLM, ARM CPU):

```toml
[reranker]
provider = "tei"
endpoint = "http://localhost:8080"
model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
timeout_seconds = 30
```

Production (BGE-reranker-v2-m3 on GPU host):

```toml
[reranker]
provider = "tei"
endpoint = "http://tei-internal:8080"
model_id = "BAAI/bge-reranker-v2-m3"
timeout_seconds = 30
```

The reranker model is set in TEI itself (the `--model-id` arg to `text-embeddings-router`); `[reranker].model_id` is just the Kengram-side stable identity.

### Troubleshooting

**Embedder dimension mismatch.** The HNSW vector index is a *partial* index keyed on `(embedding_dim, model_id)` — the dimension is literal in the partial-index predicate. If `[embedder].dimensions` doesn't match the dim in `[embedder].model_id`'s `:NNNN` suffix, or if neither matches a partial index that exists in Postgres, the search planner won't use HNSW and vector search degrades to a sequential scan. The fix is to keep all three in lockstep — `dimensions = 1024`, `model_id = "...:1024"`, and a migration that adds the matching partial index — and run a migration when the embedding model changes.

**Tagger silent-disable: no tags landing.** Symptom: `thoughts.tags_extractor_version` stays `NULL` on new captures. Verify via the serve startup log:

```text
kengram serve started ... tagger=disabled
```

This means `[tagger].provider` resolved to `""`. Set it to `"openai-compatible"` (or `"openrouter"`), restart, and run `kengram tag --rerun --since 1970-01-01T00:00:00Z` to backfill.

**Reranker timeout.** When the reranker is unreachable or slow, the pipeline silently degrades to RRF + recency. The response has `rerank_used: false`; results still come back. No error is raised. If reranks are systematically falling through, check the TEI container health and `[reranker].timeout_seconds`.

**Port collisions on `:8080`.** docker-compose maps TEI to host `:8080`, and the kengram serve default is also `:8080`. When both run on the same machine, set `[server].bind` away from `:8080` (`127.0.0.1:8081` for local Tier 0, or `0.0.0.0:8081` + `allowed_hosts` for Tailnet Tier 1 — the M5.2 history). The other direction works too — remap TEI to a different host port in `docker-compose.yml` — but moving kengram is usually less disruptive.

**Tagger schema field stripped by claude.ai's MCP client.** The hosted claude.ai web client strips optional MCP tool fields whose JSON schema lacks a concrete `type`. The fix (M5.2) is to declare `tag_filter` and `metadata` on the relevant tools as `Option<Map<String, Value>>` rather than `Option<Value>`, so schemars renders `type: ["object", "null"]` (concrete) rather than letting `type` go missing. If a new tool field goes missing in the claude.ai client but works in `mcp-inspector`, check the field's Rust type — `serde_json::Value` is too lax, use a concrete container.

**Migration didn't run / unexpected schema.** Run `kengram audit migrations` first. If a migration is missing from the audit log, it didn't run — re-run `kengram migrate`. If it's there but `rows_touched` is unexpected, read the `notes` column for the migration's rationale and compare against the migration file in `migrations/`.

## Port conflicts

If something else already binds `5432`, edit `docker-compose.yml` to map a different host port (e.g. `"5433:5432"`) and update `DATABASE_URL` accordingly. Same for `8080` — see the troubleshooting note above on the kengram-vs-TEI collision.

## Production note

In production, Postgres runs as a systemd-managed service (not Docker), and the embedder is a TEI sidecar (also systemd-managed) rather than Ollama. Both deployment shapes are described in `DESIGN.md` §11. The dev setup here exists for ergonomics — the production setup is operator-managed and out of scope for this file.

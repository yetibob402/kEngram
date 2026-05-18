# Development setup

The operator reference for Engram: first-time setup, common operations, the full configuration knob list, tagger version history, the relational link graph, day-to-day workflows, and troubleshooting. README is the front-door pitch; everything operator-facing lives here.

Quick start assumes macOS with Docker, Rust (`rustc` 1.95+), `sqlx-cli`, and Ollama already installed.

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

`sqlx::query!` macros and the `sqlx::test` attribute both require `DATABASE_URL` to be set at *build time*, not just at runtime. The `.env` file at the workspace root is read by `sqlx-cli` but NOT by `cargo build` — set `DATABASE_URL` in your shell or pass it inline: `DATABASE_URL=... cargo build`.

### 3. Pull the embedding model in Ollama

Make sure the Ollama daemon is running (`ollama serve` if it isn't already; the macOS desktop app launches it automatically), then:

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

The configured `embedder.model_id = "bge-m3:1024"` carries the dimension as a suffix. That suffix is load-bearing: the HNSW vector index in Postgres is a partial index keyed on `(embedding_dim, model_id)`, and the embedder writes the `:NNNN` dim into the model_id field so the query planner can route lookups to the matching partial. If you change the embedding model, change the suffix in lockstep with the migration that adds the new partial index. See the troubleshooting section for the symptom when these drift.

### 3b. (Optional) Start TEI for the rerank stage

The cross-encoder reranker runs in a TEI Docker container alongside Postgres. It's optional — `engram serve` works without it. The search pipeline silently skips the rerank stage when no `[reranker]` section is configured and the results come back in RRF + recency order with `rerank_used: false`.

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

The migration set (currently 11 numbered files in `migrations/`) ships the schema described in `DESIGN.md` §5, plus subsequent additions: thought retraction, the thought_links graph layer, polymorphic link targets, soft-delete + migration_audit, and the JSONB cleanup that removed the redundant `tags.relations` copy.

**Migration audit.** The `migration_audit` table (introduced in 0010) records what each migration did — `migration`, `ran_at`, `rows_touched`, optional `notes`. Convention going forward: any row-touching migration ends with an `INSERT INTO migration_audit (...)` statement so the operator can verify per-migration impact via `engram audit migrations` rather than psql. Schema-only migrations should still insert an audit row with `rows_touched = 0` and a one-line `notes` summary. See [Operator workflows](#operator-workflows) for the `engram audit migrations` walkthrough.

### 5. Build, test, run

```bash
cargo build --workspace
cargo test --workspace                       # unit + sqlx::test
cargo test --workspace --features integration   # adds a live-Ollama round-trip test

cargo run --bin engram -- serve              # starts the MCP server on 127.0.0.1:8080
cargo run --bin engram -- worker             # in a second shell — drains pending_embeddings + pending_tags
cargo run --bin engram -- stats              # corpus + storage telemetry; operator-facing snapshot
cargo run --bin engram -- audit migrations   # per-migration audit log
```

Point an MCP-capable client (Claude Code, Claude Desktop, `mcp-inspector`) at `http://127.0.0.1:8080/mcp` (streamable-HTTP transport, per the current MCP spec). Nine tools are exposed:

- `capture` — write a thought; returns `thought_id`, `embedding_status: "pending"`, and `is_duplicate`. Same content captured twice (SHA-256 fingerprint match) returns the existing `thought_id`.
- `search_thoughts` — RRF-fused vector + trigram retrieval over thoughts; recency-boosted; optional cross-encoder rerank; optional `tag_filter` JSONB-containment filter (e.g. `{"kind": "task"}`); `scope` (exact) or `scope_prefix` (namespace) for scope filtering. Each hit carries its `tags` object.
- `recent_thoughts` — chronological browse; supports `scope` or `scope_prefix`.
- `list_scopes` — discover what scopes are in use (optionally `prefix`-filtered). Pair with `scope_prefix` on the search/recent tools for a discover-then-query workflow.
- `get_thought` — full thought + provenance + tags + tagger provenance.
- `retract_thought` — mark a thought as untrusted (excluded from retrieval; still visible via `get_thought` for audit).
- `link_thoughts`, `unlink_thoughts`, `get_related_thoughts` — the graph layer. See [Relational data and link graph](#relational-data-and-link-graph).

`engram serve` and `engram worker` are paired: `serve` writes thoughts and enqueues embedding + tag jobs; `worker` drains both queues (`pending_embeddings` and `pending_tags`). Running `serve` without `worker` is fine — thoughts are still durable and trigram-searchable — but vector kNN won't surface them and tags stay empty until the worker runs. When `[tagger].provider` is empty, the tag-job enqueue at capture is a no-op and the tag drainer doesn't spawn.

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
```

### Embed-backfill

Heal-then-drain: enqueue any unembedded thoughts that lack a queue row (pre-M2 captures, or captures whose enqueue lost a crash race), then drain the queue inline. Use this if you've been running `serve` without `worker` and want to catch up without spinning up the worker, or after embedder downtime to drain the backlog.

```bash
# Whole corpus, up to 1000 rows.
cargo run --bin engram -- embed-backfill --limit 1000

# One scope only (exact match).
cargo run --bin engram -- embed-backfill --scope work --limit 100

# A namespace of scopes (prefix match). Mutually exclusive with --scope.
cargo run --bin engram -- embed-backfill --scope-prefix engram. --limit 500
```

`--scope` and `--scope-prefix` are mutually exclusive. Empty strings on either flag are normalised to "no filter."

### One-shot tagger run

Like a single tick of the worker's tag drainer. Tags thoughts where `tags_extractor_version IS NULL`. Requires a configured `[tagger]` section. Useful for catching up after capturing a batch of thoughts before enabling the tagger.

```bash
cargo run --bin engram -- tag --limit 50
cargo run --bin engram -- tag --scope work --limit 100
cargo run --bin engram -- tag --scope-prefix engram. --limit 200
```

### Re-tag after tagger version bump

Re-run the tagger over thoughts whose stored `tags_extractor_version` is below the configured current version. Use this after bumping `[tagger].model_version` (typically after a prompt or schema change). Tags are overwritten in place — no supersede semantics, no audit chain. Pair with `--since` to bound the rerun to recent thoughts; use `--since 1970-01-01T00:00:00Z` to re-tag the entire corpus.

```bash
cargo run --bin engram -- tag --rerun --scope work
cargo run --bin engram -- tag --rerun --scope-prefix engram. --since 2026-04-01T00:00:00Z
cargo run --bin engram -- tag --rerun --since 1970-01-01T00:00:00Z   # whole corpus
```

If you've pinned `model_version` in your local `~/.config/engram/engram.toml`, bump it manually. The new bundled default (currently 7) only applies when the field is absent from your TOML. The log line at startup reports the resolved value: look for `target_version=7`. If it says `target_version=N` with `N < 7`, your config still overrides; either update the line or delete it.

For the procedural detail and the full v1→v7 changelog, see [Tagger version history and safe re-tag procedure](#tagger-version-history-and-safe-re-tag-procedure).

### Reranker A/B benchmark

A/B-benchmark the reranker against RRF-only on an operator-curated fixture corpus. Prints a markdown table to stdout with per-query nDCG@10 and MRR for both rankings, plus an AVERAGE row. Requires a configured `[reranker]` section in `engram.toml` and the corpus's `relevant_ids` to point at real `thought_id` rows in your DB. See `tests/fixtures/bench-rerank.example.json` for the schema.

```bash
cargo run --bin engram -- bench rerank --corpus ~/.engram/my-bench.json
```

## Configuration reference

Defaults live in code. Override via `~/.config/engram/engram.toml`, a `--config <path>` argument, or `ENGRAM_*` env vars (nested via `__`, e.g. `ENGRAM_DATABASE__URL`). Layering order: defaults → user TOML → `--config` TOML → env. Later wins.

Example `engram.toml` (every knob spelled out — most can be omitted to take the default):

```toml
[server]
bind = "127.0.0.1:8080"
allowed_hosts = []                                      # see below

[database]
url = "postgres://engram:engram@localhost:5432/engram"
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
provider = "openai-compatible"                          # "" = silent-disable; "openai-compatible" or "openrouter"
endpoint = "http://localhost:8000/v1"                   # vLLM default
model_name = "qwen2.5-7b-instruct"                      # the model the backend serves
model_id = "vllm/qwen2.5-7b-instruct"                   # provenance written into thoughts.tags_extractor_model
model_version = 7                                       # tracks BUNDLED_TAGGER_VERSION; see Tagger version history
# api_key = ""
timeout_seconds = 60
temperature = 0.2
scope_vocab_enabled = true
scope_vocab_size = 50
# system_prompt_file = "~/.config/engram/tagger-prompt.txt"

[worker]
tick_interval_seconds = 5
batch_size = 16
```

Env override examples: `ENGRAM_WORKER__TICK_INTERVAL_SECONDS=2 cargo run --bin engram -- worker` (snappier ticks for development), `ENGRAM_TAGGER__API_KEY=sk-...` (OpenRouter key without checking it into config), `ENGRAM_TAGGER__PROVIDER="" cargo run --bin engram -- serve` (silent-disable the tagger for a run).

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
| `url` | `"postgres://engram:engram@localhost:5432/engram"` | Postgres connection string. Single-tenant; one database per Engram deployment. |
| `max_connections` | `10` | Size of the sqlx connection pool. Bump for high-tag-volume worker hosts; single-user dogfood is fine at 10. |

### `[embedder]`

| knob | default | what it does |
|---|---|---|
| `provider` | `"openai-compatible"` | Only provider in current builds. Covers Ollama, TEI, OpenAI, Voyage by varying `endpoint`/`model`. |
| `endpoint` | `"http://localhost:11434/v1"` | `/v1` base URL. Ollama in dev; TEI in production. |
| `model` | `"bge-m3"` | Backend model name as the server understands it. |
| `model_id` | `"bge-m3:1024"` | Engram-side stable identity. The `:NNNN` suffix is the embedding dimension — must match an HNSW partial index in Postgres. Change this only in lockstep with a migration. |
| `dimensions` | `1024` | Embedding vector dimension. Must match the model's actual output dim AND the `:NNNN` suffix above. |
| `api_key` | `None` | Bearer token for hosted endpoints. Omit for Ollama/TEI. |
| `timeout_seconds` | `5` | Per-request timeout. Local Ollama is sub-100ms typical; bump for slower hosted endpoints. |

### `[reranker]`

The reranker is the optional cross-encoder stage that re-scores the top `candidate_pool` post-RRF hits. Empty `provider` silent-disables — the search pipeline falls through to RRF + recency.

| knob | default | what it does |
|---|---|---|
| `provider` | `""` | `""` = disabled; `"tei"` = TEI sidecar (currently the only supported provider). |
| `endpoint` | `"http://localhost:8080"` | Service root, no `/v1` suffix. The reranker client appends `/rerank`. |
| `model_id` | `"BAAI/bge-reranker-v2-m3"` | Engram-side stable identity. Dev override: `"cross-encoder/ms-marco-MiniLM-L-6-v2"` (matches the docker-compose pin). |
| `timeout_seconds` | `30` | Per-request timeout. MiniLM is sub-100ms on Apple Silicon; BGE-v2-m3 on GPU is similar; ARM CPU runs of BGE-v2-m3 are minutes per call (don't). |

When the reranker times out or errors, the pipeline silently degrades to RRF + recency order and the response has `rerank_used: false`. Search still returns results.

### `[tagger]`

The tagger is the per-thought metadata sidecar. Empty `provider` is the silent-disable sentinel: capture proceeds, no tag jobs enqueue, the worker doesn't spawn a tag drainer. Flip `provider = "openai-compatible"` (or `"openrouter"`) to enable.

| knob | default | what it does |
|---|---|---|
| `provider` | `""` | `""` = disabled; `"openai-compatible"` (vLLM, etc.) or `"openrouter"`. |
| `endpoint` | `"http://localhost:8000/v1"` | `/v1` base URL. vLLM default port. OpenRouter is `"https://openrouter.ai/api/v1"`. |
| `model_name` | `"qwen2.5-7b-instruct"` | Model name as the backend understands it. For OpenRouter: a model slug like `"anthropic/claude-haiku-4.5"`. |
| `model_id` | `"vllm/qwen2.5-7b-instruct"` | Engram-side stable identity written into `thoughts.tags_extractor_model`. Conventionally `<vendor>/<model>`. |
| `model_version` | `7` | Tracks `engram_extract::BUNDLED_TAGGER_VERSION`. Written into `thoughts.tags_extractor_version`. Bump when the prompt or schema changes such that prior tags shouldn't be considered comparable; then `engram tag --rerun`. See [Tagger version history](#tagger-version-history-and-safe-re-tag-procedure). |
| `api_key` | `None` | Bearer token for hosted endpoints. |
| `timeout_seconds` | `60` | Per-request timeout. vLLM JSON-Schema responses can run long. |
| `temperature` | `0.2` | Generation temperature. Lower = more deterministic. 0 makes some backends loop. |
| `system_prompt_file` | `None` | Path to a file containing a custom system prompt. `None` = use `BUNDLED_TAGGER_PROMPT`. Operators who supply a custom prompt are responsible for also bumping `model_version` so provenance stays meaningful; a WARN log is emitted at startup. |
| `scope_vocab_enabled` | `true` | Inject the top topic + entity terms from the thought's scope into the tagger prompt as a controlled-vocabulary hint. Encourages consistent term reuse across captures. |
| `scope_vocab_size` | `50` | Top-N established terms (each for topics and entities) fed to the tagger. Larger = more vocabulary stability; smaller = faster emergence of new terms. |

### `[worker]`

| knob | default | what it does |
|---|---|---|
| `tick_interval_seconds` | `5` | How often the embed and tag drainers wake up and claim a batch off their respective queues. 5s is fine for single-user dogfood; tune lower for snappier vector-search readiness, higher to be gentler on the backends. |
| `batch_size` | `16` | Max jobs claimed per tick (per queue). Bigger batches are kinder to the backend; smaller batches mean shorter critical sections and faster failover when a job hangs. |

## Tagger version history and safe re-tag procedure

The tagger's prompt + JSON schema is versioned by `engram_extract::BUNDLED_TAGGER_VERSION` (currently **7**). Each thought row carries a `tags_extractor_version` recording the version it was tagged under, so the drainer can identify stale rows when the version is bumped.

### Version changelog

- **v1** (M4 launch, 2026-05-16). Initial thoughts-only tagger. Single `topics` field for both proper-noun identifiers and subject categories.
- **v2** (M4.1, 2026-05-16). Split `topics` into `entities` (proper-noun-style identifiers) + `topics` (subject categories). Added the optional scope-vocabulary controlled-vocab section: the drainer pre-fetches the top-N established terms from the thought's scope and feeds them as a hint so new captures prefer existing vocabulary.
- **v3** (M4.1 iteration, 2026-05-17). Tightened `entities` to canonical proper names only, with an explicit anti-padding rule. Added a kind-isolation clause forbidding the controlled vocabulary from influencing `kind` classification.
- **v4** (M4.1 iteration, 2026-05-17). Restructured the `entities` description to lead with the empty case and a structural NAME-vs-DESCRIBE test. (The v3 negative-example list backfired — on thought `047d0ce8` the model emitted those exact phrases verbatim.) Dropped `entities` `maxItems` from 5 to 3. Softened the scope-vocabulary section from "vocab dominates" to "vocab tie-breaks" — precision over consistency.
- **v5** (M6.1, 2026-05-17). Added tagger-extracted relations: the LLM emits closed-vocabulary `(relation, to_kind, to_value)` edges for explicit relational claims in prose. Non-thought targets only (`entity` / `person` / `url`); the LLM does not synthesise UUIDs.
- **v6** (post-M6.1 dogfood pass 1, 2026-05-18). Rebalanced `kind` classification as a 5-step decision tree. Added an entity surface-only rule. Tightened URL emission criteria. Listed `embedding-based` and `lexical signals` as literal negative examples — which repeated the v3→v4 backfire pattern (the model emitted those phrases verbatim on `047d0ce8` again).
- **v7** (post-v6 dogfood pass 2, 2026-05-18). Dropped the literal-phrase NOT-entities list and any suffix hints (e.g. `-based`), relying on the structural NAME-vs-DESCRIBE test, the surface-only rule, and the re-read verification alone. Mirrors v4's clean-pattern fix and documents the v6 lesson explicitly so a v8 doesn't reintroduce phrase hints. Adds an explicit topics-as-concept-mapping intent statement (topics may be inferred when the subject is clear; surface lexemes are not required), which had been de-facto behavior since v4 vocab-softening but wasn't stated.

There's an outstanding doc-only acknowledgement that came with v8 (2026-05-18): the "entities-adjectival regression" — corner-case adjectival phrases sometimes still land as entities — is accepted as structural. v8 is not a tagger version bump (no prompt or schema change), just the documented decision not to chase it further with another phrase-list iteration.

There's also a v9 (2026-05-18) code-side change that dropped `tags.relations` from the persisted JSONB. `thought_links` is now the sole canonical store for tagger-emitted relations; the LLM emission shape is unchanged and `BUNDLED_TAGGER_VERSION` did NOT bump (still 7). Migration 0011 removed the key from existing rows.

### Safe re-tag procedure

After bumping the tagger version (or the bundled default rolls forward and you want the corpus on the new schema):

1. **Verify the resolved target version.** Start `engram serve` (or `engram worker`). The startup log line is:
   ```
   tagger: resolved config ... model_version=7 ...
   ```
   And on the re-tag side, `engram tag` prints:
   ```
   engram tag starting ... target_version=7 ...
   ```
   If `target_version` is lower than expected, your `~/.config/engram/engram.toml` is overriding the bundled default. Bump it manually or delete the line.

2. **Re-tag the corpus.** Whole corpus:
   ```bash
   cargo run --bin engram -- tag --rerun --since 1970-01-01T00:00:00Z
   ```
   The drainer walks rows where `tags_extractor_version < target_version`. Bound it tighter if you only want a recent window — `--since 2026-04-01T00:00:00Z` or `--scope-prefix engram.`.

3. **Monitor the worker logs for failures.** The `engram tag complete` line reports `n_candidates`, `tagged`, `failed`. Non-zero `failed` exits non-zero so cron / scripts can detect partial failures; per-row errors are logged at WARN with the `thought_id`.

4. **Spot-check.** psql is fine for this; there's no dedicated CLI:
   ```sql
   SELECT id, tags_extractor_version, tags_extractor_model
   FROM thoughts
   WHERE tags_extractor_version IS NOT NULL
   ORDER BY tags_extractor_updated_at DESC
   LIMIT 10;
   ```
   If the change was a migration (e.g. v9's 0011), use `engram audit migrations` instead — see [Operator workflows](#operator-workflows).

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

Three tools on the graph layer. Full request/response schemas are documented in the MCP server instructions (the `SERVER_INSTRUCTIONS` constant in `crates/engram-mcp/src/server.rs`); one-paragraph overview each:

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

Day-to-day patterns. Each one assumes `engram serve` is running and reachable.

### Discover-then-query (scopes)

Before capturing into a new scope, see what's already in use. Before searching across a namespace, see what scopes are under that namespace.

```text
list_scopes()                                  # all scopes, most-recently-used first
list_scopes(prefix="engram.")                  # namespace discovery
search_thoughts(query="...", scope_prefix="engram.")    # query across the namespace
recent_thoughts(scope_prefix="engram.", limit=20)       # browse a namespace
```

The CLI equivalent for the search side: `engram embed-backfill` and `engram tag` both accept `--scope-prefix`. The shell-side flag for `engram stats` is `--scope-prefix` (filters the scopes summary section).

### Capacity audit

```bash
cargo run --bin engram -- stats
cargo run --bin engram -- stats --scope-prefix engram. --top-scopes 50
```

Output is a sectional plain-text report:

- **Corpus.** Live and retracted thought counts (and retracted %), untagged count, content bytes total + average per thought, embeddings broken out by `(count × model_id, dim, version)`, links live + soft-deleted with breakdowns by relation / target kind / source.
- **Queues.** `pending_embeddings` and `pending_tags` row counts. These are the *current backlog* — what hasn't drained yet. Diverges from the on-disk-tables section: queues show pending rows, on-disk tables show allocated bytes.
- **Scopes.** Per-scope thought count + last-activity date, sorted by recency. Truncated to `--top-scopes` (default 20); summary line says how many are hidden.
- **On-disk tables.** Per-table heap / index / total bytes, plus a corpus-wide total. This is Postgres's view of how much space each table is using (heap + all indexes). It will diverge from "row count × content size" because of TOAST, dead tuples awaiting vacuum, index overhead, and the `embeddings` table's pgvector storage.

A non-zero `pending_embeddings` count with on-disk-tables that match expectations means the worker is behind. A growing on-disk-tables size with no corresponding row growth means dead-tuple accumulation — `VACUUM` is the answer.

### Re-tag after tagger version bump

See [Tagger version history and safe re-tag procedure](#tagger-version-history-and-safe-re-tag-procedure) for the procedure. Short version: verify `target_version` in the startup log, `engram tag --rerun --since 1970-01-01T00:00:00Z`, watch the WARN logs for per-row failures, spot-check.

### Migration audit

```bash
cargo run --bin engram -- audit migrations
cargo run --bin engram -- audit migrations --since 2026-05-01T00:00:00Z --limit 20
```

Prints one entry per migration (most recent first), with `ran_at` timestamp, `rows_touched`, the migration filename, and the free-text `notes` column on a second line. Use this:

- After running `engram migrate` — confirm each pending migration ran and touched the expected row count.
- When a migration's effect is in question — `rows_touched > 0` means data changed; `notes` describes why.
- For audit / forensics — the table is append-only (one row per `sqlx migrate run` per migration), so it's a complete record of schema evolution since 0010.

### Embedder unreachable

`search_thoughts` falls through to trigram-only when the embedder won't answer. The response has `vector_search_available: false`; results still come back, just from the lexical leg. This is not an error.

Meanwhile, the worker keeps `pending_embeddings` rows pinned. When the embedder recovers, the drainer picks them up on the next tick. To force a heal-then-drain without waiting: `engram embed-backfill`.

If thoughts captured during the outage didn't get a queue row at all (a crash race between capture and enqueue), `embed-backfill` finds them too — it walks `thoughts` left-joined against the embeddings table and re-enqueues anything missing for the active model.

### Tagger silent-disable

`[tagger].provider = ""` is the silent-disable sentinel. Capture proceeds normally; the tag-job enqueue is a no-op and the worker doesn't spawn a tag drainer. Thoughts go in with `tags_extractor_version = NULL`.

To enable later: set `provider = "openai-compatible"` (or `"openrouter"`), bring up the backend, restart `serve` + `worker`, and run `engram tag --rerun --since 1970-01-01T00:00:00Z` to catch the backlog. The serve startup log line reports which state it resolved to:

```text
engram serve started ... tagger=disabled
# or
engram serve started ... tagger=enabled (vllm/qwen2.5-7b-instruct)
```

### Embed-backfill after embedder downtime

```bash
cargo run --bin engram -- embed-backfill --limit 1000
cargo run --bin engram -- embed-backfill --scope work --limit 100
cargo run --bin engram -- embed-backfill --scope-prefix engram. --limit 500
```

Heal-then-drain: finds thoughts missing an embedding row for the active model, enqueues them if they aren't already, then drains the queue inline. `--limit` caps the run so a huge backlog doesn't pin the embedder for an hour; iterate until the queue clears.

Exit code is non-zero on partial failure (`failed > 0`), suitable for cron-style retry.

## Configuration presets and troubleshooting

### Preset: vLLM-local tagger (dev)

Local vLLM serving qwen2.5-7b-instruct on port 8000:

```toml
[tagger]
provider = "openai-compatible"
endpoint = "http://localhost:8000/v1"
model_name = "qwen2.5-7b-instruct"
model_id = "vllm/qwen2.5-7b-instruct"
model_version = 7
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
model_version = 7
timeout_seconds = 30
temperature = 0.2
```

Set the API key out-of-band via env to keep it out of TOML:

```bash
export ENGRAM_TAGGER__API_KEY="sk-or-v1-..."
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

The reranker model is set in TEI itself (the `--model-id` arg to `text-embeddings-router`); `[reranker].model_id` is just the Engram-side stable identity.

### Troubleshooting

**Embedder dimension mismatch.** The HNSW vector index is a *partial* index keyed on `(embedding_dim, model_id)` — the dimension is literal in the partial-index predicate. If `[embedder].dimensions` doesn't match the dim in `[embedder].model_id`'s `:NNNN` suffix, or if neither matches a partial index that exists in Postgres, the search planner won't use HNSW and vector search degrades to a sequential scan. The fix is to keep all three in lockstep — `dimensions = 1024`, `model_id = "...:1024"`, and a migration that adds the matching partial index — and run a migration when the embedding model changes.

**Tagger silent-disable: no tags landing.** Symptom: `thoughts.tags_extractor_version` stays `NULL` on new captures. Verify via the serve startup log:

```text
engram serve started ... tagger=disabled
```

This means `[tagger].provider` resolved to `""`. Set it to `"openai-compatible"` (or `"openrouter"`), restart, and run `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill.

**Reranker timeout.** When the reranker is unreachable or slow, the pipeline silently degrades to RRF + recency. The response has `rerank_used: false`; results still come back. No error is raised. If reranks are systematically falling through, check the TEI container health and `[reranker].timeout_seconds`.

**Port collisions on `:8080`.** docker-compose maps TEI to host `:8080`, and the engram serve default is also `:8080`. When both run on the same machine, set `[server].bind` away from `:8080` (`127.0.0.1:8081` for local Tier 0, or `0.0.0.0:8081` + `allowed_hosts` for Tailnet Tier 1 — the M5.2 history). The other direction works too — remap TEI to a different host port in `docker-compose.yml` — but moving engram is usually less disruptive.

**Tagger schema field stripped by claude.ai's MCP client.** The hosted claude.ai web client strips optional MCP tool fields whose JSON schema lacks a concrete `type`. The fix (M5.2) is to declare `tag_filter` and `metadata` on the relevant tools as `Option<Map<String, Value>>` rather than `Option<Value>`, so schemars renders `type: ["object", "null"]` (concrete) rather than letting `type` go missing. If a new tool field goes missing in the claude.ai client but works in `mcp-inspector`, check the field's Rust type — `serde_json::Value` is too lax, use a concrete container.

**Migration didn't run / unexpected schema.** Run `engram audit migrations` first. If a migration is missing from the audit log, it didn't run — re-run `engram migrate`. If it's there but `rows_touched` is unexpected, read the `notes` column for the migration's rationale and compare against the migration file in `migrations/`.

## Port conflicts

If something else already binds `5432`, edit `docker-compose.yml` to map a different host port (e.g. `"5433:5432"`) and update `DATABASE_URL` accordingly. Same for `8080` — see the troubleshooting note above on the engram-vs-TEI collision.

## Production note

In production, Postgres runs as a systemd-managed service (not Docker), and the embedder is a TEI sidecar (also systemd-managed) rather than Ollama. Both deployment shapes are described in `DESIGN.md` §11. The dev setup here exists for ergonomics — the production setup is operator-managed and out of scope for this file.

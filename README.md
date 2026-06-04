# kEngram

Self-hosted, MCP-native memory service for AI agents. Single Rust binary; Postgres + pgvector backing store; vendor-neutral via an OpenAI-compatible embedding endpoint.

> *kEngram* — **ken** (to know) + **-gram** (a recorded mark): a recorded unit of knowing. The name nods to the *engram*, the hypothetical trace a memory leaves in the brain.

## Why kEngram

kEngram gives any MCP-capable agent (Claude Code, Claude Desktop, opencode, Cline, …) a **shared, persistent memory** backed by your own Postgres. A thought captured from one client is searchable from any other — across sessions, models, and machines.

Thoughts are the only thing kEngram stores (with optional tags). Retrieval is **hybrid** — vector kNN over BGE-M3 embeddings, lexical trigram search, fused via reciprocal rank fusion and (optionally) reranked by a cross-encoder. A small **tagging sidecar** reads each new thought and writes a JSONB metadata blob (people, entities, topics, action_items, dates_mentioned, kind) onto the same row, plus auto-emits relational edges into a `thought_links` graph. Tags are advisory — they don't gate retrieval — and duplicate captures (same content) collapse to the same `thought_id` via SHA-256 fingerprinting.

The deployment is the `kengram` binary plus Postgres plus any OpenAI-compatible embedding endpoint (Ollama is the zero-config dev path). No SaaS, no per-seat fees — change LLM provider whenever you like; your memory comes with you.

For design rationale see [`DESIGN.md`](DESIGN.md); per-milestone scope and progress live in [`docs/milestones/`](docs/milestones/); operator-facing setup and runbooks are in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Quick start

### Prereqs

Install once if you don't already have them. See [DEVELOPMENT.md](DEVELOPMENT.md#install-prerequisites) for notes on each.

```bash
# Docker Desktop — https://www.docker.com/products/docker-desktop/

# Rust toolchain (latest stable):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# sqlx-cli matching the workspace's sqlx 0.8 + Postgres + rustls stack:
cargo install sqlx-cli --no-default-features --features rustls,postgres

# Ollama (macOS):
brew install ollama   # or: download from https://ollama.com/
```

### Run

The fastest path uses the launch scripts at the repo root. (Manual steps and customization: [DEVELOPMENT.md](DEVELOPMENT.md#manual-setup-advanced).)

```bash
# 0. Pull the tagger model once into the host Ollama. The embedder runs
#    in Docker — start_stack.sh pulls bge-m3 into the ollama-embed
#    container on first run, so you don't pull it on the host.
ollama pull qwen3-coder:30b        # tagging (worker, on by default)

# 1. Backing containers (Postgres + TEI reranker + ollama-embed for
#    embeddings); waits for Postgres + ollama-embed to be ready.
./start_stack.sh

# 2. Apply migrations (idempotent; safe to re-run)
export DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram'
sqlx migrate run

# 3. Terminal 1 — MCP server (binds 127.0.0.1:8081, MCP endpoint /mcp)
./start_server.sh

# 4. Terminal 2 — worker (drains embeddings + tags; pass `off` for embed-only)
./start_worker.sh

# When done — server/worker stop with Ctrl-C; halt the containers with:
./stop_stack.sh                    # add --down to also remove the containers
```

A freshly-captured thought returns `embedding_status: "pending"` — that's normal; the worker picks it up on the next tick (default 5 seconds) and the vector becomes searchable. Lexical (trigram) search finds it immediately, so retrieval works during the gap.

Once the server is up, point an MCP client at `http://127.0.0.1:8081/mcp` — see [Connecting MCP clients](#connecting-mcp-clients).

## Connecting MCP clients

### Claude Code

Speaks streamable-HTTP natively, so no bridge is needed.

```bash
# Project-scoped (writes a checked-in .mcp.json):
claude mcp add --transport http kengram --scope project http://127.0.0.1:8081/mcp

# User-scoped (writes to ~/.claude.json for the current project):
claude mcp add --transport http kengram http://127.0.0.1:8081/mcp
```

Equivalent JSON for `.mcp.json`:

```json
{
  "mcpServers": {
    "kengram": {
      "type": "http",
      "url": "http://127.0.0.1:8081/mcp"
    }
  }
}
```

All nine tools (see below) are available with full schemas.

### Claude Desktop

stdio-only, so a bridge process is required. The community-standard `mcp-remote` (Node, runs via `npx`) relays stdio ↔ HTTP:

```jsonc
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
// Windows: %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "kengram": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8081/mcp"]
    }
  }
}
```

**Pointing at a Tailnet (or any non-localhost) host:** add `--allow-http` to the args. `mcp-remote` rejects non-HTTPS URLs by default for anything other than `localhost`; on a Tailnet the mesh is the security boundary, so plain HTTP is the intended Tier 1 deployment. Field-tested working config:

```jsonc
{
  "mcpServers": {
    "kengram": {
      "command": "/opt/homebrew/bin/npx",
      "args": [
        "-y",
        "mcp-remote",
        "http://<tailnet-host>:8081/mcp",
        "--allow-http"
      ]
    }
  }
}
```

Two gotchas worth knowing:

- **Absolute path to `npx`.** Claude Desktop on macOS doesn't inherit your shell's PATH, so bare `npx` resolves to nothing and the bridge silently fails to start. Use the absolute path (`/opt/homebrew/bin/npx` for Homebrew on Apple Silicon, `/usr/local/bin/npx` on Intel Macs, `which npx` to confirm yours).
- **Port collisions.** If you're running TEI on `:8080` you've probably moved kengram to `:8081` — match that in the URL.

Server-side, the host running kengram needs `[server].bind` set to a non-loopback address (`0.0.0.0:<port>` or the Tailscale interface IP) AND `[server].allowed_hosts` populated with the hostnames / IPs the client uses — see [DEVELOPMENT.md Configuration reference](DEVELOPMENT.md#configuration-reference) for the `[server]` table and a worked allowlist example.

Restart Claude Desktop after editing the config.

### claude.ai (web)

claude.ai supports remote MCP servers as custom connectors. Kengram's tool schemas declare concrete JSON-Schema types on every object field, because claude.ai's MCP client silently strips fields without an explicit `type` annotation from outbound tool calls — so every tool argument (including `metadata` and `tag_filter`) round-trips correctly through the web client.

### opencode (Ollama-backed)

Kengram doesn't host the chat — it just publishes the tool surface. To drive Kengram from a *local Ollama model*, [opencode](https://opencode.ai) is the most direct fit: a TUI coding agent with native streamable-HTTP MCP support and a built-in Ollama provider.

Config lives at `opencode.json` (project root) or `~/.config/opencode/opencode.json`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "kengram": {
      "type": "remote",
      "url": "http://127.0.0.1:8081/mcp",
      "enabled": true
    }
  },
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": {
        "qwen3:14b": { "name": "Qwen3 14B" }
      }
    }
  }
}
```

In opencode, pick the model via `/models` (it appears as `ollama/qwen3:14b`); Kengram's nine tools become available alongside opencode's built-ins. **The model must be tool-capable** — `qwen3` family, `llama3.1+`, `gpt-oss` work; many smaller Llama variants silently no-op on tool calls. Kengram's default deployment is single-user localhost (Tier 0); LAN/Tailnet exposure is Tier 1. Tier 2 bearer auth is M7 ahead — see [Roadmap](#roadmap).

### Other MCP clients

Any client speaking streamable-HTTP can point at `http://127.0.0.1:8081/mcp` directly. Known-good options for Ollama-driven chat include [Cline](https://github.com/cline/cline) and [Roo Code](https://github.com/RooCodeInc/Roo-Code) (VS Code extensions) and [OpenWebUI](https://openwebui.com) via the [MCPO](https://github.com/open-webui/mcpo) bridge. For a quick smoke test without a chat UI, `npx @modelcontextprotocol/inspector` opens an interactive tool browser.

## What you get (MCP surface)

**Status:** everything through M6 is shipped — capture & search, hybrid retrieval with cross-encoder rerank, the tagging sidecar, and the relational link graph. M7 (operational maturity — metrics, Tier 2 auth, eval suite, backups) is ahead. See the [Roadmap](#roadmap) for the per-milestone breakdown.

| Tool | What it does |
|---|---|
| `capture` | Record a thought. Returns `thought_id`, `embedding_status: "pending"`, and `is_duplicate: bool`. Same content captured twice (SHA-256 fingerprint match) returns the existing `thought_id` with `is_duplicate: true` and no new embedding/tag jobs enqueued. New captures enqueue both the embed job and the tag job; the `kengram worker` drains both. |
| `search_thoughts` | Hybrid retrieval (vector kNN ∪ trigram, fused by RRF, recency-boosted, optionally reranked by a cross-encoder). Gracefully degrades to trigram-only when the embedder is unreachable; result includes `vector_search_available: bool`. Scope filtering: pass `scope` for exact match, or `scope_prefix` to query across a namespace (e.g. `scope_prefix: "rjf."`) — mutually exclusive. Accepts an optional `tag_filter` (JSONB containment, e.g. `{"kind": "task"}`, `{"people": ["Sarah"]}`) and surfaces each hit's full `tags` object. Excludes retracted thoughts. |
| `recent_thoughts` | Browse by recency in a (optional) scope. Accepts `scope` (exact match) or `scope_prefix` (namespace match) — mutually exclusive. Excludes retracted thoughts. |
| `get_thought` | Full thought + provenance (embedding status, embedded-at, tags, tagger provenance — `tags_extractor_model` / `tags_extractor_version` / `tags_extracted_at` — retraction state). Direct lookup returns the row even if retracted — the audit path. |
| `retract_thought` | Mark a thought as untrusted. Sets `thoughts.retracted_at` so the row is excluded from retrieval. The row stays in the DB; `get_thought` still returns it with retraction state surfaced. |
| `link_thoughts` | Create a link from a thought to a polymorphic target (another thought, an entity name, a person name, or a URL) with one of seven closed-vocabulary relations. Idempotent on `(from, relation, to_kind, to_value)`. |
| `unlink_thoughts` | Soft-delete a link by its `(from, relation, target)` triple. Returns a three-way `status`: `deleted_now`, `already_deleted`, or `never_existed`. Re-creating a soft-deleted edge via `link_thoughts` succeeds (fresh row). |
| `get_related_thoughts` | Walk the link graph from a thought. Returns `outbound` + `inbound` groups; each hit carries a `to_kind`/`to_value` discriminator and a `link_source` (`agent` vs `tagger`). For thought-target hits, also surfaces the target's content-preview, scope, retraction state, and timestamps. Optional filters by relation, target kind, and direction. |
| `list_scopes` | Enumerate scopes currently in use (each with `thought_count`, `first_activity_at`, `last_activity_at`), sorted most-recently-used first. Optional `prefix` filter. Pair with `scope_prefix` on `search_thoughts` / `recent_thoughts` to query across a namespace. |

CLI subcommands: `kengram serve`, `kengram worker`, `kengram migrate`, `kengram embed-backfill`, `kengram tag`, `kengram stats`, `kengram bench`, `kengram audit migrations`, `kengram backup`, `kengram restore`. Operational details in [Operator workflows](DEVELOPMENT.md#operator-workflows) in DEVELOPMENT.md.

## How tagging works

Thoughts are the unit of storage. Each thought is a free-form blob of text plus a stable `thought_id`, an embedding (computed async by the worker), and a JSONB `tags` sidecar populated by the tagger.

**1. Capture.** `capture(content, scope?, source?, metadata?)` computes `content_fingerprint = sha256(content)` and `INSERT ... ON CONFLICT (content_fingerprint) DO NOTHING`. If the row is new, two jobs land in queue tables: one in `pending_embeddings`, one in `pending_tags`. Otherwise the response returns the pre-existing `thought_id` with `is_duplicate: true` and no jobs are enqueued.

**2. Drain.** The `kengram worker` process runs two drainers in parallel on the `[worker] tick_interval_seconds` cadence. The embed drainer calls the configured `[embedder]` and inserts vectors; the tag drainer calls the configured `[tagger]` and writes `tags` + provenance (`tags_extractor_model`, `tags_extractor_version`, `tags_extracted_at`).

**3. Tag shape.** The tagger speaks OpenAI's `/v1/chat/completions` with `response_format: { type: "json_schema", strict: true, ... }`. Guided decoding (vLLM's `xgrammar`, OpenRouter's structured outputs) makes the response guaranteed-parseable. The prompt and schema live in `crates/kengram-extract/src/openai_compatible.rs`; the current version is **v16** (`BUNDLED_TAGGER_VERSION = 16`). Output shape:

```json
{
  "people": ["Sarah"],
  "entities": ["kengram", "pgvector"],
  "action_items": ["fast-track migration #0042"],
  "topics": ["memory-systems", "release-process"],
  "dates_mentioned": ["Thursday"],
  "kind": "task"
}
```

Tagger-emitted relations land in `thought_links` (with `source='tagger'`), not in this JSONB — `thought_links` is the single canonical store for the link graph; the `tags` JSONB is metadata only. Migration `0011_drop_tags_relations` removed the legacy `tags.relations` key from existing rows.

`entities` are canonical proper names the prose mentions (projects, products, libraries, tools, organizations); descriptive phrases belong in `topics`. `topics` are broader subject categories. Keeping them separate lets `tag_filter` distinguish "thoughts that mention kengram by name" from "thoughts categorized under memory-systems." `kind` is one of `observation | task | idea | reference | person_note | session | decision_record` (or `null` if the model is unsure). Tags are **advisory** — a wrong tag is low-impact because retrieval still works against the raw content via vector + trigram. See [Tagger version history](DEVELOPMENT.md#tagger-version-history-and-safe-re-tag-procedure) in DEVELOPMENT.md for the full v1→v16 changelog and re-tag procedure.

**Filter at search time.** `search_thoughts(query, tag_filter?)` runs `WHERE tags @> $tag_filter` (JSONB containment, GIN-indexed). Examples: `{"kind": "task"}`, `{"entities": ["kengram"]}`, `{"topics": ["rust"], "kind": "idea"}`.

**Re-tag.** `kengram tag [--rerun] [--force] [--snapshot[=PATH]] [--since <RFC3339>] [--scope X | --scope-prefix Y]` runs the tagger on demand. Plain, it tags only thoughts where `tags_extractor_version IS NULL`; `--rerun` also re-tags thoughts whose stored version is below the current tagger version; `--force` re-tags every matching thought regardless of version (for a model swap that didn't bump the prompt version). Tags are overwritten in place — no supersede semantics, no audit chain, because the raw text is what's queryable — so `--snapshot` first writes the current tags to a JSON file before a destructive pass.

## How relations work

On top of the tagging sidecar, Kengram has a graph layer of edges in a closed vocabulary of seven relations:

| relation | meaning |
|---|---|
| `replaces` | newer thought supersedes an earlier one (both stay, retrieval prefers newer) |
| `requires` | this thought depends on another (decision presupposes a constraint) |
| `references` | this thought points at another for context (citation, follow-up) |
| `supports` | this thought confirms a claim made in another (evidence, corroboration). Distinct from `references`: a citation that *endorses*, not just *cites*. |
| `belongs_to` | membership / containment (a finding under a parent thread) |
| `decided_by` | provenance: a decision attributable to a person or session anchor |
| `refines` | newer thought refines an earlier one (both stand; newer is updated thinking) |

Edges are either agent-supplied via `link_thoughts` or auto-emitted by the tagger. Each row in `thought_links` carries a `source` discriminator (`agent` vs `tagger`), surfaced in `get_related_thoughts` responses as `link_source`. The closed vocab is intentionally small: it captures the relational structure that shows up in conversation memory without trying to be a general knowledge graph.

**Polymorphic targets.** The `from` side is always a thought; the `to` side can be a thought (UUID), a free-text entity name, a free-text person name, or a URL (must start with `http://` or `https://`). Pass exactly one of `to_thought_id` / `to_entity` / `to_person` / `to_url`. Tagger-emitted edges currently target non-thoughts only (`url` / `entity` / `person`); thought-to-thought tagger extraction requires entity resolution and is deferred.

**Idempotency.** `link_thoughts` is idempotent on `(from, relation, to_kind, to_value)` — re-asserting the same live edge returns the existing `link_id` with `is_new: false`.

**Soft-delete.** `unlink_thoughts` soft-deletes via `thought_links.deleted_at`. The three-way `status` distinguishes `deleted_now` (was live, just removed), `already_deleted` (was previously removed), and `never_existed` (no such edge). Re-creating a previously-removed edge via `link_thoughts` succeeds — the partial unique index ignores soft-deleted rows. On re-tag (`kengram tag --rerun`), prior tagger-emitted edges are soft-deleted before fresh emissions land; agent-supplied edges are untouched.

**Retraction interaction.** Edges survive thought retraction — retracted thoughts on either side surface with `retracted: true` rather than being filtered out. To fully sever the link, use `unlink_thoughts`.

Deeper workflows (link audit, restore, bulk operations) live in [Relational data and link graph](DEVELOPMENT.md#relational-data-and-link-graph) in DEVELOPMENT.md.

## Configuration

Kengram defaults wire up a working dev environment against `localhost` Postgres + Ollama. The minimum to override anything is a TOML file at `~/.config/kengram/kengram.toml`:

```toml
[database]
url = "postgres://kengram:kengram@localhost:5432/kengram"

# Override embedder, tagger, or reranker here; everything is optional.
# See DEVELOPMENT.md for the annotated config reference.
```

- **Embedder.** Recommended dev setup runs `bge-m3` in the Dockerized `ollama-embed` container on `http://localhost:11435/v1` (CPU-only, isolated from the host Ollama that serves the tagger); the built-in code default is `http://localhost:11434/v1` for operators running everything through one host Ollama. To change providers (TEI in production, OpenAI, OpenRouter), see [Configuration presets](DEVELOPMENT.md#configuration-presets-and-troubleshooting) in DEVELOPMENT.md.
- **Tagger.** Silent-disable when `[tagger].provider` is empty (the default) — capture and search work normally; thoughts stay with `tags = '{}'`. Local vLLM and OpenRouter presets ship. See [Configuration presets](DEVELOPMENT.md#configuration-presets-and-troubleshooting).
- **Reranker.** Optional cross-encoder re-scores the top RRF candidates: BGE-reranker-v2-m3 in production, MiniLM in dev. Disabled when `[reranker].provider` is empty (the default). See [Configuration presets](DEVELOPMENT.md#configuration-presets-and-troubleshooting).

Env vars override the file (e.g. `KENGRAM_DATABASE__URL=...`, nested via `__`). The full annotated reference for every section and every field lives in DEVELOPMENT.md.

## Repo layout

```
crates/
├── kengram-core/                 # domain types, Embedder + Reranker + Tagger traits, RRF + recency_boost (pure)
├── kengram-storage/              # sqlx queries, migrations, repository functions
├── kengram-embed/                # Embedder + Reranker impls: OpenAICompatibleEmbedder, TeiReranker, fakes
├── kengram-extract/              # Tagger impls: OpenAICompatibleTagger (vLLM/OpenRouter/Ollama), HttpTagger (sidecar client), FakeTagger
├── kengram-tagger-protocol/      # wire types for the HTTP tagger-sidecar contract
├── kengram-tagger-deterministic/ # reference non-LLM tagger sidecar (opt-in)
├── kengram-mcp/                  # capture/search/get/recent/retract/link/unlink/related/scopes orchestrators + rmcp wiring
└── kengram-cli/                  # binary; serve/migrate/worker/embed-backfill/tag/stats/bench/audit/backup/restore subcommands
migrations/                       # sqlx migrations (numbered)
docs/                             # design doc + per-milestone scope/progress
scripts/                          # operator-driven runbooks (bench-rerank.md, smoke.md)
```

## Roadmap

Built across seven capability milestones (M1 → M7), preceded by an environment milestone (M0):

| Milestone | Status | What it adds |
|---|---|---|
| [M0 — dev environment](docs/milestones/m0-dev-environment.md) | ✅ | Docker Postgres + Ollama dev path |
| [M1 — capture & search](docs/milestones/m1-capture-and-search.md) | ✅ | `capture`, `search_thoughts`, `recent_thoughts`, `get_thought` over MCP |
| [M2 — facts pipeline](docs/milestones/m2-facts-pipeline.md) | ✅ | Async embedding seam, reflector cron, `search_facts`, `correct_fact`. *(Superseded by M4; the facts pipeline was retired and replaced by a tagging sidecar.)* |
| [M3 — search & extraction quality](docs/milestones/m3-search-quality.md) | ✅ | Cross-encoder reranker; A/B benchmarking harness. Retrieval portion shipped; extraction-side dogfood motivated M4. |
| [M4 — collapse to thoughts-only](docs/milestones/m4-collapse-to-thoughts.md) | ✅ | Drop the facts pipeline; thoughts-only with content-fingerprint dedup and a JSONB tagging sidecar. `search_thoughts` gains `tag_filter`. |
| [M4.1 — v2 tagging](docs/milestones/m4.1-tagging-v2.md) | ✅ | Split `topics` into `entities` + `topics`; scope-aware controlled-vocabulary injection. |
| [M5 — selective relations](docs/milestones/m5-selective-relations.md) | ✅ | Thought-to-thought graph layer with seven closed-vocabulary relations. M5.1 added `supports`. M5.2 added polymorphic targets, soft-delete with three-way unlink status, and operator audit (`migration_audit` + `kengram audit migrations`). |
| [M6 — stats CLI + tagger-extracted relations](docs/milestones/m6-stats-and-tagger-relations.md) | ✅ | `kengram stats` CLI for corpus + storage telemetry. v5+ tagger auto-emits non-thought relations (URLs / entities / persons) from prose with `source='tagger'`. The original M6 (artifacts) was dropped after a live-corpus measurement showed kengram occupies a high-signal-density sweet spot. |
| [M7 — operational maturity](docs/milestones/m7-operational-maturity.md) | ⏳ | Metrics, Tier 2 auth, eval suite, backups |

Per-milestone progress is tracked in `docs/milestones/m{N}-progress.md`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

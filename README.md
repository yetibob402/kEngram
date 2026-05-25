# Kengram

Self-hosted, MCP-native memory service for AI agents. Single Rust binary; Postgres + pgvector backing store; vendor-neutral via an OpenAI-compatible embedding endpoint.

## Why Kengram

Kengram gives any MCP-capable agent (Claude Code, Claude Desktop, opencode, Cline, …) a **shared, persistent memory** backed by your own Postgres. A thought captured from one client is searchable from any other — across sessions, models, and machines.

Thoughts are the only thing Kengram stores. Retrieval is **hybrid** — vector kNN over BGE-M3 embeddings, lexical trigram search, fused via reciprocal rank fusion and (optionally) reranked by a cross-encoder. A small **tagging sidecar** reads each new thought and writes a JSONB metadata blob (people, entities, topics, action_items, dates_mentioned, kind) onto the same row, plus auto-emits relational edges into a `thought_links` graph. Tags are advisory — they don't gate retrieval — and duplicate captures (same content) collapse to the same `thought_id` via SHA-256 fingerprinting.

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

```bash
# 1. Bring up Postgres
docker compose up -d postgres

# 2. Bring up the embedding backend. Ollama runs as a background daemon on
#    macOS via `brew services start ollama`; otherwise leave `ollama serve`
#    running in a terminal. Then pull the 1024-dim BGE-M3 model:
ollama pull bge-m3

# 3. Apply migrations (uses sqlx-cli directly; no compilation needed)
export DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram'
sqlx migrate run

# 4. Terminal 1: the MCP server (binds 127.0.0.1:8080, MCP endpoint at /mcp)
cargo run --bin kengram -- serve

# 5. Terminal 2: the worker (drains pending_embeddings and pending_tags)
cargo run --bin kengram -- worker
```

A freshly-captured thought returns `embedding_status: "pending"`; that's normal — the worker picks it up on the next tick (default 5 seconds) and the vector becomes searchable. Trigram (lexical) search picks the thought up immediately, so retrieval still works during the gap.

Once the server is up, point an MCP client at `http://127.0.0.1:8080/mcp` — see [Connecting MCP clients](#connecting-mcp-clients) below.

## Connecting MCP clients

### Claude Code

Speaks streamable-HTTP natively, so no bridge is needed.

```bash
# Project-scoped (writes a checked-in .mcp.json):
claude mcp add --transport http kengram --scope project http://127.0.0.1:8080/mcp

# User-scoped (writes to ~/.claude.json for the current project):
claude mcp add --transport http kengram http://127.0.0.1:8080/mcp
```

Equivalent JSON for `.mcp.json`:

```json
{
  "mcpServers": {
    "kengram": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp"
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
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8080/mcp"]
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

claude.ai supports remote MCP servers as custom connectors. Kengram's tool schemas declare concrete JSON-Schema types on every object field (a v9-cycle fix) because claude.ai's MCP client silently strips fields without explicit `type` annotations from outbound tool calls. With v9+ Kengram every tool argument round-trips correctly through the web client; older revisions saw silent drops on `metadata` and `tag_filter`.

### opencode (Ollama-backed)

Kengram doesn't host the chat — it just publishes the tool surface. To drive Kengram from a *local Ollama model*, [opencode](https://opencode.ai) is the most direct fit: a TUI coding agent with native streamable-HTTP MCP support and a built-in Ollama provider.

Config lives at `opencode.json` (project root) or `~/.config/opencode/opencode.json`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "kengram": {
      "type": "remote",
      "url": "http://127.0.0.1:8080/mcp",
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

Any client speaking streamable-HTTP can point at `http://127.0.0.1:8080/mcp` directly. Known-good options for Ollama-driven chat include [Cline](https://github.com/cline/cline) and [Roo Code](https://github.com/RooCodeInc/Roo-Code) (VS Code extensions) and [OpenWebUI](https://openwebui.com) via the [MCPO](https://github.com/open-webui/mcpo) bridge. For a quick smoke test without a chat UI, `npx @modelcontextprotocol/inspector` opens an interactive tool browser.

## What you get (MCP surface)

**Status:** M1 (capture & search), M2 (async-embed seam, since superseded by M4), M3 (hybrid search + cross-encoder rerank + A/B harness), M4 (collapse to thoughts-only with tagging sidecar), M4.1 (entities split + scope-aware vocabulary), M5 (selective relations — closed-vocab thought graph), and M6 (`kengram stats` + tagger-extracted relations) are shipped. M7 (operational maturity — metrics, Tier 2 auth, eval suite, backups) is ahead. See the [Roadmap](#roadmap).

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

CLI subcommands: `kengram serve`, `kengram worker`, `kengram migrate`, `kengram embed-backfill`, `kengram tag`, `kengram stats`, `kengram bench`, `kengram audit migrations`. Operational details in [Operator workflows](DEVELOPMENT.md#operator-workflows) in DEVELOPMENT.md.

## How tagging works

Thoughts are the unit of storage. Each thought is a free-form blob of text plus a stable `thought_id`, an embedding (computed async by the worker), and a JSONB `tags` sidecar populated by the tagger.

**1. Capture.** `capture(content, scope?, source?, metadata?)` computes `content_fingerprint = sha256(content)` and `INSERT ... ON CONFLICT (content_fingerprint) DO NOTHING`. If the row is new, two jobs land in queue tables: one in `pending_embeddings`, one in `pending_tags`. Otherwise the response returns the pre-existing `thought_id` with `is_duplicate: true` and no jobs are enqueued.

**2. Drain.** The `kengram worker` process runs two drainers in parallel on the `[worker] tick_interval_seconds` cadence. The embed drainer calls the configured `[embedder]` and inserts vectors; the tag drainer calls the configured `[tagger]` and writes `tags` + provenance (`tags_extractor_model`, `tags_extractor_version`, `tags_extracted_at`).

**3. Tag shape.** The tagger speaks OpenAI's `/v1/chat/completions` with `response_format: { type: "json_schema", strict: true, ... }`. Guided decoding (vLLM's `xgrammar`, OpenRouter's structured outputs) makes the response guaranteed-parseable. The prompt and schema live in `crates/kengram-extract/src/openai_compatible.rs`; the current version is **v7** (`BUNDLED_TAGGER_VERSION = 7`). Output shape:

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

`entities` are canonical proper names the prose mentions (projects, products, libraries, tools, organizations); descriptive phrases belong in `topics`. `topics` are broader subject categories. Keeping them separate lets `tag_filter` distinguish "thoughts that mention kengram by name" from "thoughts categorized under memory-systems." `kind` is one of `observation | task | idea | reference | person_note | session` (or `null` if the model is unsure). Tags are **advisory** — a wrong tag is low-impact because retrieval still works against the raw content via vector + trigram. See [Tagger version history](DEVELOPMENT.md#tagger-version-history-and-safe-re-tag-procedure) in DEVELOPMENT.md for the full v1→v7 changelog and re-tag procedure.

**Scope-aware vocabulary.** Before tagging, the drainer fetches the top-N most-frequent topic and entity terms used in the same scope and injects them as a tie-break vocabulary hint. The v4+ prompt frames vocab as suggestions — the model uses a vocab term when it accurately fits and coins a new one when nothing does (precision over consistency). Configurable via `[tagger].scope_vocab_enabled` (default `true`) and `[tagger].scope_vocab_size` (default `50`). Tuning details in [Configuration presets and troubleshooting](DEVELOPMENT.md#configuration-presets-and-troubleshooting) in DEVELOPMENT.md.

**Filter at search time.** `search_thoughts(query, tag_filter?)` runs `WHERE tags @> $tag_filter` (JSONB containment, GIN-indexed). Examples: `{"kind": "task"}`, `{"entities": ["kengram"]}`, `{"topics": ["rust"], "kind": "idea"}`.

**Re-tag.** `kengram tag [--rerun --since <RFC3339>] [--scope X | --scope-prefix Y]` runs the tagger on demand. Without `--rerun`, tags thoughts where `tags_extractor_version IS NULL`. With `--rerun`, re-tags thoughts whose stored version is below the current tagger version. Tags are overwritten — no supersede semantics, no audit chain, because the raw text is what's queryable.

## How relations work

On top of the tagging sidecar, Kengram has a graph layer of edges in a closed vocabulary of seven relations (M5 shipped six; M5.1 added `supports` after day-one dogfood):

| relation | meaning |
|---|---|
| `replaces` | newer thought supersedes an earlier one (both stay, retrieval prefers newer) |
| `requires` | this thought depends on another (decision presupposes a constraint) |
| `references` | this thought points at another for context (citation, follow-up) |
| `supports` | this thought confirms a claim made in another (evidence, corroboration). Distinct from `references`: a citation that *endorses*, not just *cites*. |
| `belongs_to` | membership / containment (a finding under a parent thread) |
| `decided_by` | provenance: a decision attributable to a person or session anchor |
| `refines` | newer thought refines an earlier one (both stand; newer is updated thinking) |

Edges are either agent-supplied via `link_thoughts` or auto-emitted by the tagger (M6.1). Each row in `thought_links` carries a `source` discriminator (`agent` vs `tagger`), surfaced in `get_related_thoughts` responses as `link_source`. The closed vocab is intentionally small: it captures the relational structure that shows up in conversation memory without trying to be a general knowledge graph.

**Polymorphic targets.** The `from` side is always a thought; the `to` side can be a thought (UUID), a free-text entity name, a free-text person name, or a URL (must start with `http://` or `https://`). Pass exactly one of `to_thought_id` / `to_entity` / `to_person` / `to_url`. Tagger-emitted edges ship v1 as non-thought targets only (`url` / `entity` / `person`); thought-to-thought tagger extraction requires entity resolution and is deferred.

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

- **Embedder.** Defaults to Ollama at `http://localhost:11434/v1` with `bge-m3`. To change providers (TEI in production, OpenAI, OpenRouter), see [Configuration presets](DEVELOPMENT.md#configuration-presets-and-troubleshooting) in DEVELOPMENT.md.
- **Tagger.** Silent-disable when `[tagger].provider` is empty (the default) — capture and search work normally; thoughts stay with `tags = '{}'`. Local vLLM and OpenRouter presets ship. See [Configuration presets](DEVELOPMENT.md#configuration-presets-and-troubleshooting).
- **Reranker.** Optional cross-encoder re-scores the top RRF candidates: BGE-reranker-v2-m3 in production, MiniLM in dev. Disabled when `[reranker].provider` is empty (the default). See [Configuration presets](DEVELOPMENT.md#configuration-presets-and-troubleshooting).

Env vars override the file (e.g. `KENGRAM_DATABASE__URL=...`, nested via `__`). The full annotated reference for every section and every field lives in DEVELOPMENT.md.

## Repo layout

```
crates/
├── kengram-core/      # domain types, Embedder + Reranker + Tagger traits, RRF + recency_boost (pure)
├── kengram-storage/   # sqlx queries, migrations, repository functions
├── kengram-embed/     # Embedder + Reranker impls: OpenAICompatibleEmbedder, TeiReranker, fakes
├── kengram-extract/   # Tagger impls: OpenAICompatibleTagger (vLLM/OpenRouter), FakeTagger
├── kengram-mcp/       # capture/search/get/recent/retract/link/unlink/related/scopes orchestrators + rmcp wiring
└── kengram-cli/       # binary; serve/migrate/worker/embed-backfill/tag/stats/bench/audit subcommands
migrations/           # sqlx migrations (numbered)
docs/                 # design doc + per-milestone scope/progress
scripts/              # operator-driven runbooks (smoke.md, bench-rerank.md)
```

## Roadmap

Built in six capability milestones (M1 → M6), preceded by an environment milestone (M0):

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

TBD — not currently published.

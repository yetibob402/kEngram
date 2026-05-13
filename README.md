# Engram

Self-hosted, MCP-native memory service for AI agents. Single Rust binary; Postgres + pgvector backing store; vendor-neutral via an OpenAI-compatible embedding endpoint.

## Why Engram

Engram gives any MCP-capable agent (Claude Code, Claude Desktop, opencode, Cline, …) a **shared, persistent memory** backed by your own Postgres. A thought captured from one client is searchable from any other — across sessions, models, and machines.

What sets Engram apart from "an MCP server that talks to a database" is the **facts pipeline**. On a schedule you control, a local LLM (vLLM, or OpenRouter for cloud fallback) reads each new thought and extracts structured `(subject, predicate, object, confidence)` triples. So a casual capture like *"Talked to Sarah — she wants migration #0042 fast-tracked, mobile freeze starts Thursday"* becomes two queryable rows: *Sarah → wants fast-tracked → migration #0042* and *mobile freeze → starts → Thursday*. Agents can search both the raw thoughts (natural language) and the derived facts (structured statements with source-thought provenance), correct extractions that got it wrong (with the audit trail preserved), and re-run extraction when you upgrade the model. The raw thought stays immutable underneath; facts are derived data that can be re-derived.

The deployment is the `engram` binary plus Postgres plus any OpenAI-compatible embedding endpoint (Ollama is the zero-config dev path). No SaaS, no per-seat fees, no vendor lock-in — change LLM provider whenever you like; your memory comes with you.

For design rationale see [`docs/engram-design-v0.md`](docs/engram-design-v0.md); per-milestone scope and progress live in [`docs/milestones/`](docs/milestones/); first-time setup details are in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## What you get (MCP surface)

**Status:** M1 and M2 are shipped (capture, hybrid search, facts pipeline, six MCP tools). M3–M5 are planned — see the [Roadmap](#roadmap) at the end of this doc.

| Tool | What it does |
|---|---|
| `capture` | Record a thought. Returns `thought_id` + `embedding_status: "pending"`; the `engram worker` drains the embed queue on its tick. |
| `search_thoughts` | Hybrid retrieval (vector kNN ∪ trigram, fused by RRF, recency-boosted). Gracefully degrades to trigram-only when the embedder is unreachable; result includes `vector_search_available: bool`. |
| `recent_thoughts` | Browse by recency in a (optional) scope. |
| `get_thought` | Full thought + provenance (embedding status, embedded-at, and active `linked_facts`). |
| `search_facts` | Trigram search over `facts.statement`, filtered to active (non-superseded) rows. Each result includes the fact's S/P/O triple plus the source thought's content/scope/created_at (no follow-up `get_thought` needed). M3 adds the vector leg. |
| `correct_fact` | Operator-driven correction. With a replacement, inserts a manual-author fact (`extractor_model="manual"`, `extractor_version=0`, `confidence=1.0`) and supersedes the old row, preserving the audit trail. Without a replacement, retracts via supersede. |

CLI subcommands: `engram serve`, `engram worker`, `engram migrate`, `engram embed-backfill`, `engram reflect [--rerun --since <RFC3339>]`. Operational details in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## How fact extraction works

Thoughts are raw, immutable, free-form text — what an agent or operator typed. Facts are structured rows derived from thoughts on a schedule: each one is a self-contained natural-language statement, optionally decomposed into an (S, P, O) triple, with a confidence score and a pointer back to the source thought. Facts are queryable independently of thoughts; you can ask `search_facts("migration #0042")` and get back the fact rows with their source-thought context attached.

The lifecycle, end to end:

**1. Capture.** A client (Claude Code, Claude Desktop, opencode, …) calls `capture(content, scope?, source?, metadata?)`. Engram writes the thought row and returns the `thought_id` immediately. Embedding is async — the `engram worker` drainer task vectorizes the thought within a few seconds so vector search picks it up. Trigram search works against the raw content immediately.

**2. Reflect.** On a cron you control (default `0 0 3 * * *` — 03:00 daily; or run `engram reflect --limit 50` on demand), the same worker process wakes up its reflector task. It walks **unfacted thoughts** — `LEFT JOIN facts WHERE facts.id IS NULL`, oldest first — and asks your configured extractor LLM to extract structured facts.

**3. Extract.** The extractor speaks OpenAI's `/v1/chat/completions` with `response_format: { type: "json_schema", ... }`. The model gets the thought content plus a system prompt that defines the output schema: an array of `{ statement, subject?, predicate?, object?, confidence }`. JSON Schema-guided decoding (vLLM's `xgrammar`/`outlines` backends, OpenRouter's structured-outputs) makes the response guaranteed-parseable. So given:

> *"Talked to Sarah today about the PR backlog. She wants migration #0042 fast-tracked because the mobile freeze starts Thursday."*

…a reasonable extractor returns:

```json
{ "facts": [
    { "statement": "Sarah wants migration #0042 fast-tracked",
      "subject": "Sarah", "predicate": "wants fast-tracked", "object": "migration #0042",
      "confidence": 0.9 },
    { "statement": "Mobile freeze starts Thursday",
      "subject": "mobile freeze", "predicate": "starts", "object": "Thursday",
      "confidence": 0.85 }
] }
```

**4. Route.** Facts with `confidence ≥ review_queue_below` (default 0.7) land in `facts` and become immediately searchable. Lower-confidence rows go to `facts_review_queue` for operator decision. Either path records `extractor_model`, `extractor_version`, and `source_run_id` — so a whole bad run can be jointly retracted later (`UPDATE facts SET superseded_at = NOW() WHERE source_run_id = ...`) and re-evaluating after a model upgrade is `WHERE extractor_version < N`.

**5. Search.** `search_facts(query, scope?)` does trigram retrieval over `facts.statement`, filtered to active (non-superseded) rows, joined to the source thought. Each result is self-contained: the fact, the (S, P, O) triple, the confidence, *and* the source thought's content/scope/created_at. No follow-up `get_thought` call needed. `get_thought(id)` carries `linked_facts` for the reverse direction.

**6. Correct.** When the extractor gets it wrong, `correct_fact(fact_id, replacement?)` supersedes the row. With a replacement, a new fact is inserted with sentinel provenance (`extractor_model="manual"`, `extractor_version=0`, `confidence=1.0`) — the operator is the authority. Without a replacement, the row is retracted. The old row stays in the database with `superseded_at` set; the audit trail is complete.

**7. Rerun.** `engram reflect --rerun [--since <RFC3339>]` re-extracts already-facted thoughts. Exact `(S, P, O, statement)` matches are no-ops (idempotency keystone); same triple with a different statement supersedes the old row (preserving the audit trail); new triples insert as additional facts. It's additive only — existing facts the new extractor *doesn't* reproduce stay active, because rerun reflects model drift in how facts are stated, not in what the thought says.

**What this gives you.** Raw thoughts you can search lexically and semantically *and* structured facts you can query as data — without the database becoming the source of truth for either. The raw capture is immutable; the facts are reproducible from it; bad extractions are correctable in place; model upgrades are routine re-runs rather than migrations. Every fact carries enough provenance (`extractor_model`, `extractor_version`, `source_run_id`, `superseded_by`) that "where did this come from and is it still current?" is always a single `SELECT`.

## Quick start

```bash
# 1. Bring up the dev environment (M0 — see DEVELOPMENT.md)
docker compose up -d postgres
ollama pull bge-m3                           # 1024-dim BGE-M3 — Engram's default model

# 2. Apply migrations
DATABASE_URL='postgres://engram:engram@localhost:5432/engram' \
  cargo run --bin engram -- migrate

# 3. Run the MCP server (and the worker, in another terminal, to drain embeddings)
DATABASE_URL='postgres://engram:engram@localhost:5432/engram' \
  cargo run --bin engram -- serve

DATABASE_URL='postgres://engram:engram@localhost:5432/engram' \
  cargo run --bin engram -- worker
```

The server binds `127.0.0.1:8080` by default and exposes a streamable-HTTP MCP endpoint at `/mcp` (per the current MCP spec, via rmcp's `StreamableHttpService`). With it running, point a chat client at the endpoint — see [Connecting MCP clients](#connecting-mcp-clients) below.

## Connecting MCP clients

### Claude Code

The official Claude Code CLI speaks streamable-HTTP natively, so no bridge is needed.

```bash
# Project-scoped (writes to a checked-in .mcp.json):
claude mcp add --transport http engram --scope project http://127.0.0.1:8080/mcp

# User-scoped (writes to ~/.claude.json for the current project):
claude mcp add --transport http engram http://127.0.0.1:8080/mcp
```

Equivalent JSON for `.mcp.json`:

```json
{
  "mcpServers": {
    "engram": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

### Claude Desktop

Claude Desktop's MCP support is stdio-only, so a bridge process is required. The community-standard `mcp-remote` (Node, runs via `npx`) relays stdio ↔ HTTP:

```jsonc
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8080/mcp"]
    }
  }
}
```

Restart Claude Desktop after editing the config. Equivalent paths on Windows: `%APPDATA%\Claude\claude_desktop_config.json`.

### opencode (Ollama-backed)

Engram doesn't host the chat — it just publishes the tool surface. To drive Engram from a *local Ollama model* you need an MCP-capable agent that supports both. [opencode](https://opencode.ai) is the most direct fit: a TUI coding agent with native streamable-HTTP MCP support and a built-in Ollama provider.

Config lives at `opencode.json` (project root) or `~/.config/opencode/opencode.json` (user). One file, two blocks — the `mcp` entry points at Engram; the `provider` entry wires a tool-capable Ollama model:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "engram": {
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

In opencode, pick the model via `/models` (it appears as `ollama/qwen3:14b`); Engram's six tools become available alongside opencode's built-ins. **The model must be tool-capable** — `qwen3` family, `llama3.1+`, `gpt-oss` work; many smaller Llama variants silently no-op on tool calls. No `opencode auth` step is needed (Ollama has no API key; Engram has no auth in M1).

### Other MCP clients

Any client that speaks streamable-HTTP (per the current MCP spec) can point at `http://127.0.0.1:8080/mcp` directly. Known-good options for Ollama-driven chat include [Cline](https://github.com/cline/cline) and [Roo Code](https://github.com/RooCodeInc/Roo-Code) (VS Code extensions) and [OpenWebUI](https://openwebui.com) via the [MCPO](https://github.com/open-webui/mcpo) bridge. For a quick smoke test without a chat UI, `npx @modelcontextprotocol/inspector` opens an interactive tool browser.

## Configuring the embedding backend

Engram talks to an OpenAI-compatible `/v1/embeddings` endpoint — the same shape served by Ollama, Hugging Face TEI, OpenAI, and Voyage. The default targets local Ollama; the dev path is zero-config.

### Ollama (default)

```bash
# Install (macOS via Homebrew; Linux: curl -fsSL https://ollama.com/install.sh | sh)
brew install ollama

# Start the daemon — leave running in a terminal, or `brew services start ollama`
ollama serve

# Pull the 1024-dim BGE-M3 model Engram is pre-configured for
ollama pull bge-m3

# Verify the OpenAI-compatible endpoint responds with a 1024-element vector
curl http://localhost:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"hello"}' | jq '.data[0].embedding | length'
# expected output: 1024
```

That's it. Engram's built-in defaults already point at `http://localhost:11434/v1` with model `bge-m3` and `model_id = "bge-m3:1024"` — no config file required.

### Overriding the defaults

If you want a different endpoint (TEI in production, OpenAI/Voyage for cloud), provide a config file or env vars. Env-var form (nested via `__`):

```bash
ENGRAM_EMBEDDER__ENDPOINT='http://tei.internal:8080/v1' \
ENGRAM_EMBEDDER__MODEL='bge-m3' \
ENGRAM_EMBEDDER__MODEL_ID='bge-m3:1024' \
ENGRAM_EMBEDDER__API_KEY='...' \
  cargo run --bin engram -- serve
```

TOML form, in `~/.config/engram/engram.toml` (or `--config <path>`):

```toml
[embedder]
provider     = "openai-compatible"
endpoint     = "http://localhost:11434/v1"   # vary as needed
model        = "bge-m3"                       # what the backend expects
model_id     = "bge-m3:1024"                  # Engram's stable identity; must match an HNSW index
dimensions   = 1024
timeout_seconds = 5
```

**Heads-up on dimensions:** the M1 migration ships exactly one HNSW partial index (`embeddings_bge_m3_hnsw` over `WHERE model_id = 'bge-m3:1024'`). Switching to a model with different dimensions or a different `model_id` requires a new migration that adds the matching partial index — see [`docs/engram-design-v0.md`](docs/engram-design-v0.md) §5 and §9. Sticking with `bge-m3:1024` via Ollama/TEI/HF requires no schema change.

## Configuring the extractor backend (M2+)

The **reflector** — the `engram worker` task that turns captured thoughts into structured facts — talks to an OpenAI-compatible `/v1/chat/completions` endpoint with `response_format: { type: "json_schema", ... }`. Two presets ship: local vLLM (production sidecar) and OpenRouter (cloud fallback). Any other OpenAI-compatible chat endpoint that supports JSON-Schema response_format also works via the `openai-compatible` provider.

The reflector is **opt-in**. `engram worker` always runs the embed-drainer; the reflector task only spawns when `[reflector] enabled = true`. So nothing in this section matters until you flip that flag.

### vLLM (default preset)

Engram's `[extractor]` defaults point at `http://localhost:8000/v1` serving `qwen2.5-7b-instruct`. To bring vLLM up alongside Engram:

```bash
# Install (CUDA/ROCm prereqs apply — see https://docs.vllm.ai/en/latest/getting_started/installation.html)
pip install vllm                   # or `uv pip install vllm`

# Serve a tool-capable model with JSON-Schema guided decoding
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --guided-decoding-backend xgrammar

# Verify the chat-completions endpoint
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }' | jq '.choices[0].message.content'
```

If your vLLM model name differs from Engram's default, update `[extractor].model_name` to match what vLLM serves. `model_id` is Engram's stable provenance label (written to `facts.extractor_model`) — bump `model_version` whenever the prompt or schema changes such that prior facts shouldn't be considered comparable.

### OpenRouter (cloud fallback)

No local GPU? Use OpenRouter — the `openrouter` provider preset sets the base URL and bearer auth automatically:

```bash
ENGRAM_REFLECTOR__ENABLED=true \
ENGRAM_EXTRACTOR__PROVIDER=openrouter \
ENGRAM_EXTRACTOR__ENDPOINT='https://openrouter.ai/api/v1' \
ENGRAM_EXTRACTOR__MODEL_NAME='anthropic/claude-haiku-4.5' \
ENGRAM_EXTRACTOR__MODEL_ID='openrouter/anthropic/claude-haiku-4.5' \
ENGRAM_EXTRACTOR__API_KEY='sk-or-...' \
  cargo run --bin engram -- worker
```

Equivalent TOML in `~/.config/engram/engram.toml`:

```toml
[extractor]
provider              = "openrouter"
endpoint              = "https://openrouter.ai/api/v1"
model_name            = "anthropic/claude-haiku-4.5"
model_id              = "openrouter/anthropic/claude-haiku-4.5"
api_key               = "sk-or-..."
timeout_seconds       = 60
temperature           = 0.2
max_facts_per_thought = 8
```

### Turning the reflector on

```toml
[reflector]
enabled               = true
schedule              = "0 0 3 * * *"   # 6-field cron (sec min hour dom month dow). 03:00 daily.
review_queue_below    = 0.7             # confidence < this → facts_review_queue; ≥ → facts
scope_filter          = ""               # leave blank for all scopes
max_thoughts_per_run  = 1000
max_facts_per_thought = 8
```

For development, tighten the schedule to something like `"*/30 * * * * *"` (every 30 seconds) and watch the worker logs. Alternatively, drive a one-shot pass without waiting for cron:

```bash
cargo run --bin engram -- reflect --scope smoke-test --limit 10
```

`engram reflect --rerun [--since <RFC3339>]` re-extracts already-facted thoughts and supersedes obsolete rows (preserving the audit trail) — useful after upgrading the extractor model.

### Model selection notes

The reflector uses structured outputs, so the model + serving stack must:

- **Support `response_format: { type: "json_schema" }`** — vLLM's `xgrammar` and `outlines` guided-decoding backends do; most OpenRouter chat models do.
- **Be instruction-following enough to populate the (S, P, O) triple cleanly** — Qwen 2.5 7B/14B Instruct, Llama 3.1 8B+, Claude Haiku / Sonnet via OpenRouter all work well. Smaller / non-instruct models often return malformed payloads (logged as `ExtractorError::MalformedResponse` and soft-failed per Q9).

## Repo layout

```
crates/
├── engram-core/      # domain types, Embedder + Extractor traits, RRF + recency_boost (pure)
├── engram-storage/   # sqlx queries, migrations, repository functions
├── engram-embed/     # Embedder impls: OpenAICompatibleEmbedder, FakeEmbedder
├── engram-extract/   # Extractor impls: OpenAICompatibleExtractor (vLLM/OpenRouter), FakeExtractor
├── engram-mcp/       # capture/search/get/recent/correct/reflect/drain orchestrators + rmcp tool wiring
└── engram-cli/       # binary; serve/migrate/worker/embed-backfill/reflect subcommands
migrations/           # sqlx migrations (numbered)
docs/                 # design doc + per-milestone scope/progress
scripts/              # operator-driven runbooks (smoke.md)
```

## Roadmap

Built in five capability milestones (M1 → M5), preceded by an environment milestone (M0):

| Milestone | Status | What it adds |
|---|---|---|
| [M0 — dev environment](docs/milestones/m0-dev-environment.md) | ✅ | Docker Postgres + Ollama dev path |
| [M1 — capture & search](docs/milestones/m1-capture-and-search.md) | ✅ | `capture`, `search_thoughts`, `recent_thoughts`, `get_thought` over MCP |
| [M2 — facts pipeline](docs/milestones/m2-facts-pipeline.md) | ✅ | Async embedding seam, reflector cron, `search_facts`, `correct_fact`, `engram reflect` |
| [M3 — search quality](docs/milestones/m3-search-quality.md) | ⏳ | Cross-encoder reranker; fact embeddings (vector leg in `search_facts`) |
| [M4 — artifacts](docs/milestones/m4-artifacts.md) | ⏳ | Long-form document ingestion |
| [M5 — operational maturity](docs/milestones/m5-operational-maturity.md) | ⏳ | Metrics, Tier 2 auth, eval suite, backups |

Per-milestone progress is tracked in `docs/milestones/m{N}-progress.md`.

## License

TBD — not currently published.

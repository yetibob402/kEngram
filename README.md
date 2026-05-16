# Engram

Self-hosted, MCP-native memory service for AI agents. Single Rust binary; Postgres + pgvector backing store; vendor-neutral via an OpenAI-compatible embedding endpoint.

## Why Engram

Engram gives any MCP-capable agent (Claude Code, Claude Desktop, opencode, Cline, …) a **shared, persistent memory** backed by your own Postgres. A thought captured from one client is searchable from any other — across sessions, models, and machines.

What sets Engram apart from "an MCP server that talks to a database" is the **facts pipeline**. On a schedule you control, a local LLM (vLLM, or OpenRouter for cloud fallback) reads each new thought and extracts structured `(subject, predicate, object, confidence)` triples. So a casual capture like *"Talked to Sarah — she wants migration #0042 fast-tracked, mobile freeze starts Thursday"* becomes two queryable rows: *Sarah → wants fast-tracked → migration #0042* and *mobile freeze → starts → Thursday*. Agents can search both the raw thoughts (natural language) and the derived facts (structured statements with source-thought provenance), correct extractions that got it wrong (with the audit trail preserved), and re-run extraction when you upgrade the model. The raw thought stays immutable underneath; facts are derived data that can be re-derived.

The deployment is the `engram` binary plus Postgres plus any OpenAI-compatible embedding endpoint (Ollama is the zero-config dev path). No SaaS, no per-seat fees, no vendor lock-in — change LLM provider whenever you like; your memory comes with you.

For design rationale see [`docs/engram-design-v0.md`](docs/engram-design-v0.md); per-milestone scope and progress live in [`docs/milestones/`](docs/milestones/); first-time setup details are in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## What you get (MCP surface)

**Status:** M1 and M2 are shipped (capture, hybrid search, facts pipeline, seven MCP tools — `retract_thought` was added in 2026-05-13 from dogfood feedback). M3 (search & extraction quality) is mostly shipped: fact embeddings, the cross-encoder reranker (see [Reranking search results](#reranking-search-results) below), the v4 extractor prompt, the per-thought `extract` flag, three-band confidence routing (`flagged`), per-claim retraction durability, subsumption-aware dedup, and quality-aware canonical selection are all live as of 2026-05-15. Phase B step 3 (A/B benchmarking harness) and Phase D (operator dogfood + close-out) remain. M4–M5 are planned — see the [Roadmap](#roadmap) at the end of this doc.

| Tool | What it does |
|---|---|
| `capture` | Record a thought. Returns `thought_id` + `embedding_status: "pending"`; the `engram worker` drains the embed queue on its tick. |
| `search_thoughts` | Hybrid retrieval (vector kNN ∪ trigram, fused by RRF, recency-boosted, optionally reranked by a cross-encoder — see [Reranking](#reranking-search-results)). Gracefully degrades to trigram-only when the embedder is unreachable; result includes `vector_search_available: bool`. Excludes retracted thoughts. |
| `recent_thoughts` | Browse by recency in a (optional) scope. Excludes retracted thoughts. |
| `get_thought` | Full thought + provenance (embedding status, embedded-at, active `linked_facts`, retraction state). Direct lookup by ID returns the row even if retracted — this is the audit path. |
| `search_facts` | Hybrid retrieval (vector kNN ∪ trigram over `facts.statement`, fused by RRF, optionally reranked) filtered to active (non-superseded) rows whose source thought is also not retracted. Each result includes the fact's S/P/O triple plus the source thought's content/scope/created_at. |
| `correct_fact` | Operator-driven correction. With a replacement, inserts a manual-author fact (`extractor_model="manual"`, `extractor_version=0`, `confidence=1.0`) and supersedes the old row, preserving the audit trail. Without a replacement, retracts via supersede. Operates on a single fact. |
| `retract_thought` | Mark a thought as untrusted (e.g. you captured a wrong claim). Atomically sets `thoughts.retracted_at` *and* auto-supersedes every active fact derived from it — so a subsequent reflector run can't re-extract from the untrusted source. The row stays in the DB; `get_thought` still returns it with retraction state. Use this rather than `correct_fact`-ing each derived fact one at a time. |

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

**6a. Retract a whole thought.** When the *source thought* was wrong (not just one of its derived facts), `retract_thought(thought_id, reason?)` is the right tool. Atomic: sets `thoughts.retracted_at` *and* supersedes every active fact derived from the thought in one transaction. The thought is now invisible to retrieval and to the reflector — a subsequent `engram reflect --rerun` can't re-extract from the untrusted source, which is the failure mode that motivated this tool's existence. `get_thought` still returns the row with retraction state surfaced; the row never leaves the database.

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

### Inspecting or overriding the system prompt

The extractor's system prompt is bundled in the binary as
`engram_extract::openai_compatible::BUNDLED_SYSTEM_PROMPT`. To inspect it
without reading source, point `[extractor].system_prompt_file` at a copy
on disk — the file's contents replace the bundled prompt for that run.
The file must contain the `{MAX_FACTS}` placeholder (the extractor
substitutes it per request); the extractor refuses to construct
otherwise.

```toml
[extractor]
# … other extractor fields …
system_prompt_file = "/Users/you/.config/engram/extractor-prompt.txt"
```

**The prompt and `model_version` are paired.** If you override the
prompt, you are responsible for also bumping `model_version` so the
provenance partition on `facts.extractor_version` stays meaningful —
otherwise rows tagged v2 may have been produced under two different
prompts. The extractor emits a `WARN`-level log line at startup whenever
a custom prompt is in use, naming the active `model_version`.

### Turning the reflector on

```toml
[reflector]
enabled                  = true
schedule                 = "0 0 3 * * *"   # 6-field cron (sec min hour dom month dow). 03:00 daily.
review_queue_below       = 0.7             # confidence < this → facts_review_queue; ≥ → facts
min_confidence_to_store  = 0.85            # confidence in [review_queue_below, this) → facts with flagged=true; ≥ → flagged=false. Set equal to review_queue_below to collapse to two-band (M3 Phase C).
subsumption_keep         = "specific"      # when (subject, predicate) matches and one object refines another, keep "specific" (default) or "general" (M3 Phase C).
scope_filter             = ""              # leave blank for all scopes
max_thoughts_per_run     = 1000
max_facts_per_thought    = 8
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

## Reranking search results

Hybrid retrieval (vector ∪ trigram fused by RRF) gives Engram strong **recall** — both meaning-similar and lexically-similar candidates surface together. The catch: RRF scores are rank-based (`1 / (60 + rank)`) and *uncalibrated* — two adjacent results differ by ~0.005 whether they're equally relevant or one is clearly better. Recency tilts the order toward fresh thoughts, but recency is not relevance.

The **reranker** stage closes that gap. After RRF + recency, the top *K* candidates (default 32) are sent to a **cross-encoder** that scores each `(query, candidate)` pair *jointly* and returns calibrated absolute relevance scores. Cross-encoders are too slow to use as the initial retriever (they re-run the model per pair), but they're the highest-quality re-scorer available — and re-ranking a small, RRF-shortlisted pool keeps the cost bounded.

Concrete example from dogfood: a query like *"tooling for compiling codebases reproducibly"* surfaces Redis-related thoughts via trigram (shared tokens like "tooling") and via vector similarity (some semantic overlap with infrastructure topics). RRF can't tell which lexical hit is closer in *meaning*; the reranker can — and reorders the Nix-reproducibility fact ahead of Redis where it belongs.

### Where it sits in the pipeline

```
search_thoughts / search_facts
  │
  ├── vector kNN  (cosine similarity, top 100)   ─┐
  │                                                ├── RRF fuse ──── recency boost ──── rerank (top K) ──── return top N
  ├── trigram     (word_similarity, top 100)     ─┘                                       ↑
  │                                                                                       │
  └── (each leg populates vector_score / trigram_score on its hits)             (optional; skipped if disabled / unavailable)
```

Each hit exposes every per-stage signal independently — `vector_score`, `trigram_score`, `rrf_score`, and `rerank_score` — rather than a single post-pipeline scalar. Results come back sorted in post-pipeline order; consumers building threshold logic pick the appropriate signal (typically `rerank_score ?? rrf_score`) and A/B comparisons between RRF-only and reranked orderings don't need a second request.

### Setup

The reranker is **opt-in**. It runs in [Hugging Face TEI](https://github.com/huggingface/text-embeddings-inference) — a Rust HTTP server purpose-built for embedding and rerank workloads. `docker-compose.yml` ships a `tei` service that wraps the dev default model:

```bash
docker compose up -d tei
# First boot downloads ~85 MB (cross-encoder/ms-marco-MiniLM-L-6-v2);
# warms up in seconds on Apple Silicon CPU.

# Smoke:
curl -s http://localhost:8080/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"reproducibility","texts":["Nix is reproducible","Redis is fast"]}' | jq .
# expect: [{ "index": 0, "score": ~0.9x }, { "index": 1, "score": ~0.0x }]
```

Then add a `[reranker]` section to `~/.config/engram/engram.toml`:

```toml
[reranker]
provider        = "tei"                                    # "" = disabled (default); "tei" = TEI sidecar
endpoint        = "http://localhost:8080"                  # no /v1 suffix; the impl appends /rerank
model_id        = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~22M params, ONNX-exported; fast on Mac CPU
timeout_seconds = 30
```

Restart `engram serve` — the startup log shows `reranker resolved provider=tei model_id=…` when the section is picked up.

### Model choice

| Model | Params | TEI backend | Notes |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22M | ORT (ONNX) | Default for dev. Sub-100ms per call on Apple Silicon CPU. Trained on MS MARCO; quality is fine for single-user dogfood. |
| `BAAI/bge-reranker-v2-m3` | ~568M | Candle | Production-grade multilingual reranker. Too slow on CPU; use only with a GPU host. |

In production with a GPU sidecar, override `[reranker].model_id` (and `endpoint` if the service is remote). The HTTP shape is identical, so swapping models is config-only.

### Per-call controls

`search_thoughts` and `search_facts` accept two optional parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `rerank` | `true` when a reranker is configured | Set `false` to return the pre-rerank (RRF + recency) order — useful for A/B comparison and debugging. |
| `candidate_pool` | `32` | How many post-RRF candidates to send to the reranker. Matches TEI's default `--max-client-batch-size`; raise it for GPU deployments that can handle larger batches. |

The response shape gains two additions (additive — existing consumers ignore unknown fields):

- Top-level **`rerank_used: bool`** — disambiguates a `null` `rerank_score` ("rerank was off" vs "this hit fell outside the candidate pool").
- Per-hit **`rerank_score: Option<f32>`** and **`rrf_score: Option<f32>`** — the calibrated rerank score and the pre-rerank RRF aggregate. The pre-rerank RRF score is preserved so consumers can compare rankings without a second request.

### Soft-fail semantics

- **`[reranker]` section omitted** → reranker silently disabled. `rerank_used: false`. No error, no warning. The pipeline degrades to RRF + recency, which is the M1/M2 behavior.
- **TEI unreachable or 5xx mid-request** → the rerank stage is skipped for that request; a `debug`-level log records the failure. `rerank_used: false`. The RRF + recency results return normally — search must never fail because the rerank sidecar is down. `RerankerError::is_transient()` distinguishes infra blips from misconfiguration.
- **`rerank: false` per-request override** → skip rerank for that call only. `rerank_used: false`.

This matches the embedder's degradation pattern: a missing or down sidecar narrows quality, never blocks search.

## Configuration reference

Engram's configuration lives in a single TOML file. Defaults are baked into the binary; the file overrides them; environment variables override the file. The task-oriented sections above ([embedder](#configuring-the-embedding-backend), [extractor](#configuring-the-extractor-backend-m2), [reranker](#reranking-search-results)) show minimal blocks for setting up each backend; this section is the canonical reference for every section and every field.

### File location and overrides

| Precedence | Source | Notes |
|---|---|---|
| 1 (highest) | `ENGRAM_*` environment variables | Nested via `__` (e.g. `ENGRAM_DATABASE__URL`, `ENGRAM_REFLECTOR__SCHEDULE`). One-off overrides without editing the file. |
| 2 | `--config <path>` CLI argument | `engram --config ./my-engram.toml serve` |
| 3 | `~/.config/engram/engram.toml` | Default lookup path on macOS / Linux. |
| 4 (lowest) | Built-in defaults | Encoded in `crates/engram-cli/src/config.rs`. Everything is optional in the file. |

### Annotated example (every field)

This is the complete config surface as of M3 Phase C. Every field is optional — omit any line to take the built-in default. Sections themselves are optional too (an empty `engram.toml` boots a working dev server against local Postgres + Ollama).

```toml
[server]
bind = "127.0.0.1:8080"                   # SocketAddr to bind. 0.0.0.0 to listen on all interfaces.

[database]
url             = "postgres://engram:engram@localhost:5432/engram"
max_connections = 10                      # sqlx connection pool ceiling.

[embedder]
provider        = "openai-compatible"     # only valid value today; covers Ollama, TEI, OpenAI, Voyage, …
endpoint        = "http://localhost:11434/v1"   # OpenAI-compatible /v1/embeddings root.
model           = "bge-m3"                # The name the backend serves (passed in the request payload).
model_id        = "bge-m3:1024"           # Engram's stable provenance label. Must match an HNSW partial index (the M1 migration ships one for `bge-m3:1024`).
dimensions      = 1024                    # Output vector dimensionality. Must match the model.
api_key         = ""                      # Optional bearer token for hosted endpoints. Leave blank for Ollama / TEI.
timeout_seconds = 5

[reranker]                                # M3 Phase B step 2; opt-in. Omit the entire section to disable.
provider        = "tei"                   # "" = disabled (default if the section is present but provider is empty); "tei" = TEI sidecar.
endpoint        = "http://localhost:8080" # No /v1 suffix; the impl appends /rerank.
model_id        = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # MiniLM for Mac dev (~22M params, ONNX-fast); BAAI/bge-reranker-v2-m3 for GPU prod.
timeout_seconds = 30

[worker]
tick_interval_seconds = 5                 # How often `engram worker` drains the pending_embeddings queue.
batch_size            = 16                # Max jobs per tick.

[extractor]                               # M2+; only consulted when [reflector].enabled = true.
provider              = "openai-compatible"   # also "openrouter" for the cloud preset.
endpoint              = "http://localhost:8000/v1"   # vLLM default; "https://openrouter.ai/api/v1" for OpenRouter.
model_name            = "qwen2.5-7b-instruct"   # The model the backend serves.
model_id              = "vllm/qwen2.5-7b-instruct"   # Provenance label written into facts.extractor_model.
model_version         = 4                 # v4 = M3 Phase C (relations rule + reinforced SPO few-shots + flagged-band framing). Bump whenever the prompt or schema changes such that prior facts shouldn't be considered comparable.
api_key               = ""                # Bearer token for hosted endpoints (OpenRouter, etc.).
timeout_seconds       = 60                # vLLM JSON-Schema responses can run long.
temperature           = 0.2
max_facts_per_thought = 8
# system_prompt_file = "~/.config/engram/extractor-prompt.txt"
# Optional: replace the bundled v4 prompt with the file's contents. Must
# contain the {MAX_FACTS} placeholder. You are responsible for also bumping
# model_version when overriding. The extractor logs a WARN at startup when
# a custom prompt is in use.

[reflector]                               # M2+; opt-in (defaults to disabled).
enabled                  = false          # Flip to true when the extractor endpoint is reachable.
schedule                 = "0 0 3 * * *"  # 6-field cron (sec min hour dom month dow). Default = 03:00 daily.
review_queue_below       = 0.7            # Confidence < this routes to facts_review_queue.
min_confidence_to_store  = 0.85           # Confidence in [review_queue_below, this) commits to facts with flagged=true (M3 Phase C three-band routing). Set equal to review_queue_below for two-band (kill-switch).
subsumption_keep         = "specific"     # "specific" (default — drop the more general) or "general" (drop the more specific) when (subject, predicate) matches and one object refines another. M3 Phase C.
scope_filter             = ""             # Restrict the reflector to a single scope. Blank = all scopes.
max_thoughts_per_run     = 1000
max_facts_per_thought    = 8
```

### Environment-variable overrides (worked examples)

The `ENGRAM_` prefix + `__` nesting maps to the TOML hierarchy. Examples:

```bash
# Snappier worker ticks for development:
ENGRAM_WORKER__TICK_INTERVAL_SECONDS=2 cargo run --bin engram -- worker

# Reflect every 30 seconds (live dogfood) and turn the reflector on
# without flipping the file:
ENGRAM_REFLECTOR__ENABLED=true \
ENGRAM_REFLECTOR__SCHEDULE="*/30 * * * * *" \
  cargo run --bin engram -- worker

# OpenRouter API key without checking it into config:
ENGRAM_EXTRACTOR__API_KEY=sk-or-... \
ENGRAM_EXTRACTOR__PROVIDER=openrouter \
ENGRAM_EXTRACTOR__ENDPOINT='https://openrouter.ai/api/v1' \
ENGRAM_EXTRACTOR__MODEL_NAME='anthropic/claude-haiku-4.5' \
  cargo run --bin engram -- worker

# Listen on all interfaces (LAN-accessible dev):
ENGRAM_SERVER__BIND=0.0.0.0:8080 cargo run --bin engram -- serve

# Different database (e.g. a second pool for testing):
ENGRAM_DATABASE__URL='postgres://engram:engram@localhost:5432/engram_test' \
  cargo run --bin engram -- migrate
```

### What's required vs optional

Strictly required for `engram serve` to boot:

- A reachable Postgres at `[database].url` with the migrations applied.

Required when the corresponding feature is in use:

- `[embedder]` reachable, for `search_thoughts` / `search_facts` to populate the vector leg (soft-fails to trigram-only if unreachable; `vector_search_available: false` on responses).
- `[reranker]` configured + reachable, for the rerank stage to fire (soft-fails to RRF + recency if absent or down; `rerank_used: false` on responses).
- `[extractor]` reachable + `[reflector] enabled = true`, for fact extraction to run.

The defaults wire up a working dev environment against `localhost` Postgres + Ollama; you only need an `engram.toml` when you're overriding something.

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
| [M3 — search & extraction quality](docs/milestones/m3-search-quality.md) | ⏳ | Cross-encoder reranker; fact embeddings (vector leg in `search_facts`); v4 extractor prompt; capture-time `extract` flag; dedup-via-supersession on rerun; three-band confidence routing with `flagged`; per-claim retraction durability; subsumption-aware dedup; quality-aware canonical selection. Phase B step 3 (A/B harness) + Phase D (operator dogfood) remain. |
| [M4 — artifacts](docs/milestones/m4-artifacts.md) | ⏳ | Long-form document ingestion |
| [M5 — operational maturity](docs/milestones/m5-operational-maturity.md) | ⏳ | Metrics, Tier 2 auth, eval suite, backups |

Per-milestone progress is tracked in `docs/milestones/m{N}-progress.md`.

## License

TBD — not currently published.

# Engram

Self-hosted, MCP-native memory service for AI agents. Single Rust binary; Postgres + pgvector backing store; vendor-neutral via an OpenAI-compatible embedding endpoint.

## Why Engram

Engram gives any MCP-capable agent (Claude Code, Claude Desktop, opencode, Cline, …) a **shared, persistent memory** backed by your own Postgres. A thought captured from one client is searchable from any other — across sessions, models, and machines.

Thoughts are the only thing Engram stores. Retrieval is **hybrid** — vector kNN over BGE-M3 embeddings, lexical trigram search, fused via reciprocal rank fusion and (optionally) reranked by a cross-encoder. On top of the raw text Engram runs a small **tagging sidecar**: a local LLM reads each new thought and writes a JSONB metadata blob (people, action_items, topics, dates_mentioned, kind) onto the same row. Tags are advisory — they don't gate storage or retrieval, but `search_thoughts` accepts a JSONB containment filter so you can scope queries to e.g. `{"kind": "task"}` or `{"people": ["Sarah"]}`. Duplicate captures (same content) collapse to the same thought_id via SHA-256 fingerprinting, so agents that re-capture aren't fighting the store.

The deployment is the `engram` binary plus Postgres plus any OpenAI-compatible embedding endpoint (Ollama is the zero-config dev path). No SaaS, no per-seat fees, no vendor lock-in — change LLM provider whenever you like; your memory comes with you.

For design rationale see [`docs/engram-design-v0.md`](docs/engram-design-v0.md); per-milestone scope and progress live in [`docs/milestones/`](docs/milestones/); first-time setup details are in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## What you get (MCP surface)

**Status:** M1 (capture & search), M2 (async-embed seam), M3 (search quality: hybrid + cross-encoder rerank + A/B harness), and M4 (collapse to thoughts-only with tagging sidecar) are shipped. The five MCP tools below are the operator-facing surface. M5 (artifacts) and M6 (operational maturity) are planned — see the [Roadmap](#roadmap) at the end of this doc.

| Tool | What it does |
|---|---|
| `capture` | Record a thought. Returns `thought_id`, `embedding_status: "pending"`, and `is_duplicate: bool`. Same content captured twice (SHA-256 fingerprint match) returns the existing `thought_id` with `is_duplicate: true` and no new embedding/tag jobs enqueued. New captures enqueue both the embed job and the tag job; the `engram worker` drains both. |
| `search_thoughts` | Hybrid retrieval (vector kNN ∪ trigram, fused by RRF, recency-boosted, optionally reranked by a cross-encoder — see [Reranking](#reranking-search-results)). Gracefully degrades to trigram-only when the embedder is unreachable; result includes `vector_search_available: bool`. Accepts an optional `tag_filter` (JSONB containment against `thoughts.tags` — e.g. `{"kind": "task"}`, `{"entities": ["engram"]}`, or `{"people": ["Sarah"]}`) and surfaces each hit's full `tags` object alongside the content. Excludes retracted thoughts. |
| `recent_thoughts` | Browse by recency in a (optional) scope. Excludes retracted thoughts. |
| `get_thought` | Full thought + provenance (embedding status, embedded-at, tags, tagger provenance — `tags_extractor_model` / `tags_extractor_version` / `tags_extracted_at` —, retraction state). Direct lookup by ID returns the row even if retracted — this is the audit path. |
| `retract_thought` | Mark a thought as untrusted (e.g. you captured a wrong claim). Sets `thoughts.retracted_at` so the row is excluded from retrieval. The row stays in the DB; `get_thought` still returns it with retraction state surfaced. |
| `link_thoughts` | Create a thought-to-thought edge in the M5 graph layer with one of seven closed-vocabulary relations (`replaces`, `requires`, `references`, `supports`, `belongs_to`, `decided_by`, `refines`). Idempotent on the `(from, relation, to)` triple. |
| `unlink_thoughts` | Delete a thought-to-thought edge by `(from, relation, to)`. Idempotent on already-deleted. |
| `get_related_thoughts` | Walk the thought graph from a thought. Returns `outbound` + `inbound` groups, each with the related thought's content-preview, retraction state, and edge metadata. Optional filters by relation type and direction. |

CLI subcommands: `engram serve`, `engram worker`, `engram migrate`, `engram embed-backfill`, `engram tag [--rerun --since <RFC3339>]`. Operational details in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## How tagging works

Thoughts are the unit of storage. Each thought is a free-form blob of text plus a stable `thought_id`, an embedding (computed async by the worker), and a JSONB `tags` sidecar populated by the tagger.

The lifecycle, end to end:

**1. Capture.** A client calls `capture(content, scope?, source?, metadata?)`. Engram computes `content_fingerprint = sha256(content)` and `INSERT ... ON CONFLICT (content_fingerprint) DO NOTHING`. If the row is new, two jobs land in queue tables: one in `pending_embeddings`, one in `pending_tags`. The response carries `thought_id`, `embedding_status: "pending"`, and `is_duplicate: false`. If the fingerprint already existed, the response returns the pre-existing `thought_id` with `is_duplicate: true` and no jobs are enqueued.

**2. Drain.** The `engram worker` process runs two drainers in parallel, both on the `[worker] tick_interval_seconds` cadence. The embed drainer calls the configured `[embedder]` and inserts vectors; the tag drainer calls the configured `[tagger]` and writes the JSONB `tags` column plus provenance (`tags_extractor_model`, `tags_extractor_version`, `tags_extracted_at`).

**3. Tag shape.** The tagger speaks OpenAI's `/v1/chat/completions` with `response_format: { type: "json_schema", strict: true, ... }`. Guided decoding (vLLM's `xgrammar`, OpenRouter's structured-outputs) makes the response guaranteed-parseable. The schema and current prompt live in `crates/engram-extract/src/openai_compatible.rs` (constants `BUNDLED_TAGGER_PROMPT` and `BUNDLED_TAGGER_VERSION`; currently at v4 after two M4.1 dogfood iterations). The output shape:

```json
{
  "people": ["Sarah"],
  "entities": ["engram", "pgvector"],
  "action_items": ["fast-track migration #0042"],
  "topics": ["memory-systems", "release-process"],
  "dates_mentioned": ["Thursday"],
  "kind": "task"
}
```

`entities` are canonical proper names the prose mentions by name — projects, products, libraries, tools, organizations. The v3/v4 prompts narrowed this to recognizable-from-outside-the-conversation named entities only; the v4 prompt defaults to `[]` and applies a structural NAME-vs-DESCRIBE test (does this phrase NAME a specific thing, or does the thought DESCRIBE an action using a noun phrase?), with `maxItems: 3` to force selectivity. Descriptive phrases like "agent memory protocol" or "cross-encoder" belong in topics if anywhere. `topics` are broader subject categories the thought falls under, capped at 3. Keeping entities and topics separate (added in M4.1) lets `tag_filter` distinguish "thoughts that mention engram by name" from "thoughts categorized under memory-systems." `kind` is one of `observation | task | idea | reference | person_note | session` (or `null` if the model is unsure); it's classified from the thought's intrinsic shape, deliberately isolated from the scope's typical content. Tags are **advisory metadata** — they don't gate storage, don't supersede each other, and a wrong tag is low-impact because retrieval still works against the raw content via vector + trigram. Expect modest kind drift across re-tag cycles for ambiguous-shape content; the closed enum gives a stable shape but the choice within it can be sensitive to model + corpus state.

**3a. Scope-aware vocabulary.** Before tagging, the drainer fetches the top-N most-frequent topic and entity terms used in the same scope and injects them into the prompt as a "controlled vocabulary" hint. The v4 prompt frames these as tie-break suggestions — the model uses a vocab term when it accurately describes the thought's subject, and chooses a more specific term when no vocab term is a close fit (precision over consistency). v2 and v3 framed vocab as a hard preference; v4 softened it after dogfood showed the dominate-framing sacrificed topic precision for cross-thought consistency. The corpus-coherence benefit on broad subjects is preserved; thoughts with distinctive content can still surface new terms. Configurable via `[tagger].scope_vocab_enabled` (default `true`) and `[tagger].scope_vocab_size` (default `50`).

**4. Filter at search time.** `search_thoughts(query, tag_filter?)` can scope retrieval to thoughts whose `tags` JSONB contains a given fragment. Implementation is `WHERE tags @> $tag_filter` (JSONB containment, GIN-indexed). Examples: `{"kind": "task"}`, `{"people": ["Sarah"]}`, `{"entities": ["engram"]}`, `{"topics": ["rust"], "kind": "idea"}`. When `tag_filter` is omitted, no filter applies.

**5. Re-tag.** `engram tag [--rerun --since <RFC3339>] [--scope X] [--limit N]` runs the tagger on demand. Without `--rerun`, tags thoughts where `tags_extractor_version IS NULL`. With `--rerun`, re-tags thoughts whose `tags_extractor_version` is below the current tagger version (i.e., the prompt or schema has changed). Tags are simply overwritten — no supersede semantics, no audit chain. The raw thought stays untouched.

**6. Dedup.** Same content captured twice — by the same agent within a session, or by two different agents at different times — collapses to the same `thought_id` via the SHA-256 unique constraint. This is the OB1 invariant: raw data is permanent and uniquely-keyed, derived signals (embeddings, tags) are recomputable on top.

**What this gives you.** A single source of truth (the thought) with two derived layers that can be re-computed independently: the embedding for hybrid retrieval, and the JSONB tags for structured filtering. No drift-defense ceremony, no supersession chains on extracted claims — the tagger's output is overwritten freely on every run because the raw text is what's queryable. The simplification was the M4 outcome of M3 dogfood (see `docs/milestones/m4-collapse-to-thoughts.md` for the rationale).

## How relations work

[M5+] Thoughts can be linked into a graph via a closed vocabulary of seven relations (M5 shipped six; M5.1 added `supports` after day-one dogfood revealed `references` was over-firing on evidential cases):

| relation | meaning |
|---|---|
| `replaces` | newer thought supersedes an earlier one (decision changed; both stay, retrieval prefers newer) |
| `requires` | this thought depends on another (decision presupposes a constraint; refinement presupposes an earlier finding) |
| `references` | this thought points at another for context (citation, follow-up, related observation) |
| `supports` | this thought confirms a claim made in another (experimental evidence, corroborating data, logical support). Distinct from `references`: a citation that *endorses*, not just *cites*. |
| `belongs_to` | membership / containment (a finding under a parent thread; a decision under a session) |
| `decided_by` | provenance: this thought is a decision attributable to a person or session anchor |
| `refines` | newer thought refines an earlier one (both stand; the newer one represents updated thinking) |

Edges are agent-supplied via MCP (`link_thoughts(from, relation, to, note?)`) — there's no LLM extraction at M5. The closed vocab is intentionally small: it captures the relational structure that actually shows up in conversation memory without trying to be a general knowledge graph. Tractable for downstream tooling because every query is "give me thoughts where (this_id, R, ?) or (?, R, this_id) holds" with R drawn from a six-element set.

**Idempotency.** `link_thoughts` is idempotent on the `(from, relation, to)` triple — re-asserting the same edge returns the existing `link_id` with `is_new: false`. Mirrors `capture`'s content-fingerprint dedup. To remove an edge, use `unlink_thoughts(from, relation, to)`.

**Traversal.** `get_related_thoughts(thought_id, relations?, direction?)` returns grouped `outbound` (edges where this thought is `from`) and `inbound` (edges where it's `to`) arrays, each carrying the related thought's `content_preview` (first 400 chars), `scope`, retraction state, edge `note`/`source`, and timestamps. Optional filters by relation type and direction (`outbound` / `inbound` / `both`, default `both`).

**Retraction interaction.** Edges survive thought retraction — retracted thoughts on either side surface with `retracted: true` in `get_related_thoughts` responses rather than being filtered out. To fully sever the link, use `unlink_thoughts`. Hard-delete of a thought (not currently exposed by the system; soft-retraction is the operator path) CASCADE-deletes its edges via the foreign-key constraint.

**Out of scope for M5.** Tagger-extracted relations (the LLM finding "this refines the earlier finding" in prose and emitting an edge automatically) — that's M5.x; requires entity resolution. Heterogeneous targets (`to-entity`, `to-person`, `to-URL`) — deferred; M5 is thought-to-thought only.

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

In opencode, pick the model via `/models` (it appears as `ollama/qwen3:14b`); Engram's five tools become available alongside opencode's built-ins. **The model must be tool-capable** — `qwen3` family, `llama3.1+`, `gpt-oss` work; many smaller Llama variants silently no-op on tool calls. No `opencode auth` step is needed (Ollama has no API key; Engram has no auth in M1).

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

## Configuring the tagger backend

The **tagger** — the `engram worker` task that writes JSONB metadata onto each captured thought — talks to an OpenAI-compatible `/v1/chat/completions` endpoint with `response_format: { type: "json_schema", strict: true, ... }`. Two provider presets ship: local vLLM (production sidecar) and OpenRouter (cloud fallback). Any other OpenAI-compatible chat endpoint that supports strict-mode `json_schema` response_format also works via the `openai-compatible` provider.

The tagger is **silent-disable**: if `[tagger].provider` is empty (or the section is missing), `capture` does not enqueue tag jobs, the tag drainer task in `engram worker` does not spawn, and thoughts stay with `tags = '{}'` forever — embedding and retrieval continue to work normally. This matches the `[reranker]` opt-in pattern. Flip `[tagger].provider = "openai-compatible"` to turn it on; then `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill.

The v1 tagger prompt and the JSON schema sent to the LLM are documented in [`docs/milestones/m4-spec.md`](docs/milestones/m4-spec.md). The schema is short — five fields (`people`, `action_items`, `topics`, `dates_mentioned`, `kind`) — and the prompt is roughly an order of magnitude smaller than the M3 fact-extraction prompt that preceded it.

### vLLM (default preset)

Engram's `[tagger]` defaults point at `http://localhost:8000/v1` serving `qwen2.5-7b-instruct`. To bring vLLM up alongside Engram:

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

If your vLLM model name differs from Engram's default, update `[tagger].model_name` to match. `model_id` is Engram's stable provenance label (written to `thoughts.tags_extractor_model`) — bump `model_version` whenever the prompt or schema changes such that prior tags shouldn't be considered comparable.

### OpenRouter (cloud fallback)

No local GPU? Use OpenRouter — the `openrouter` provider preset sets the base URL and bearer auth automatically:

```bash
ENGRAM_TAGGER__PROVIDER=openrouter \
ENGRAM_TAGGER__ENDPOINT='https://openrouter.ai/api/v1' \
ENGRAM_TAGGER__MODEL_NAME='anthropic/claude-haiku-4.5' \
ENGRAM_TAGGER__MODEL_ID='openrouter/anthropic/claude-haiku-4.5' \
ENGRAM_TAGGER__API_KEY='sk-or-...' \
  cargo run --bin engram -- worker
```

Equivalent TOML in `~/.config/engram/engram.toml`:

```toml
[tagger]
provider        = "openrouter"
endpoint        = "https://openrouter.ai/api/v1"
model_name      = "anthropic/claude-haiku-4.5"
model_id        = "openrouter/anthropic/claude-haiku-4.5"
api_key         = "sk-or-..."
timeout_seconds = 60
temperature     = 0.2
```

### Inspecting or overriding the system prompt

The tagger's system prompt is bundled in the binary. To inspect or override it without reading source, point `[tagger].system_prompt_file` at a copy on disk — the file's contents replace the bundled prompt for that run.

```toml
[tagger]
# … other tagger fields …
system_prompt_file = "/Users/you/.config/engram/tagger-prompt.txt"
```

**The prompt and `model_version` are paired.** If you override the prompt, bump `model_version` so the provenance partition on `thoughts.tags_extractor_version` stays meaningful. Then `engram tag --rerun` re-tags everything whose stored version is below the new value.

### Model selection notes

The tagger uses structured outputs, so the model + serving stack must:

- **Support `response_format: { type: "json_schema", strict: true }`** — vLLM's `xgrammar` and `outlines` guided-decoding backends do; most OpenRouter chat models do.
- **Be instruction-following enough to populate the five tag fields** — Qwen 2.5 7B/14B Instruct, Llama 3.1 8B+, Claude Haiku / Sonnet via OpenRouter all work well. The tagger schema is simpler than the M3 fact-extraction one, so smaller models often suffice.

Tagger output is advisory — a wrong tag on a single thought is low-impact because retrieval still works on the content. Soft-failures (`Timeout`, `Unreachable`, `MalformedResponse`) leave the job in `pending_tags` and are retried on the next worker tick.

## Reranking search results

Hybrid retrieval (vector ∪ trigram fused by RRF) gives Engram strong **recall** — both meaning-similar and lexically-similar candidates surface together. The catch: RRF scores are rank-based (`1 / (60 + rank)`) and *uncalibrated* — two adjacent results differ by ~0.005 whether they're equally relevant or one is clearly better. Recency tilts the order toward fresh thoughts, but recency is not relevance.

The **reranker** stage closes that gap. After RRF + recency, the top *K* candidates (default 32) are sent to a **cross-encoder** that scores each `(query, candidate)` pair *jointly* and returns calibrated absolute relevance scores. Cross-encoders are too slow to use as the initial retriever (they re-run the model per pair), but they're the highest-quality re-scorer available — and re-ranking a small, RRF-shortlisted pool keeps the cost bounded.

Concrete example from dogfood: a query like *"tooling for compiling codebases reproducibly"* surfaces Redis-related thoughts via trigram (shared tokens like "tooling") and via vector similarity (some semantic overlap with infrastructure topics). RRF can't tell which lexical hit is closer in *meaning*; the reranker can — and reorders the Nix-reproducibility thought ahead of Redis where it belongs.

### Where it sits in the pipeline

```
search_thoughts
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

`search_thoughts` accepts two optional reranker-related parameters:

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

Engram's configuration lives in a single TOML file. Defaults are baked into the binary; the file overrides them; environment variables override the file. The task-oriented sections above ([embedder](#configuring-the-embedding-backend), [tagger](#configuring-the-tagger-backend), [reranker](#reranking-search-results)) show minimal blocks for setting up each backend; this section is the canonical reference for every section and every field.

### File location and overrides

| Precedence | Source | Notes |
|---|---|---|
| 1 (highest) | `ENGRAM_*` environment variables | Nested via `__` (e.g. `ENGRAM_DATABASE__URL`, `ENGRAM_TAGGER__ENDPOINT`). One-off overrides without editing the file. |
| 2 | `--config <path>` CLI argument | `engram --config ./my-engram.toml serve` |
| 3 | `~/.config/engram/engram.toml` | Default lookup path on macOS / Linux. |
| 4 (lowest) | Built-in defaults | Encoded in `crates/engram-cli/src/config.rs`. Everything is optional in the file. |

### Annotated example (every field)

This is the complete config surface as of M4. Every field is optional — omit any line to take the built-in default. Sections themselves are optional too (an empty `engram.toml` boots a working dev server against local Postgres + Ollama).

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
tick_interval_seconds = 5                 # How often `engram worker` drains the pending_embeddings and pending_tags queues.
batch_size            = 16                # Max jobs per tick (per queue).

[tagger]                                  # M4; opt-in. Empty provider = silent-disable (tagger doesn't run; thoughts stay tags = '{}').
provider        = "openai-compatible"     # also "openrouter" for the cloud preset; "" to disable.
endpoint        = "http://localhost:8000/v1"   # vLLM default; "https://openrouter.ai/api/v1" for OpenRouter.
model_name      = "qwen2.5-7b-instruct"   # The model the backend serves.
model_id        = "vllm/qwen2.5-7b-instruct"   # Provenance label written into thoughts.tags_extractor_model.
model_version   = 1                       # Tagger prompt/schema version. Bump on any prompt/schema change; `engram tag --rerun` re-tags rows whose stored version is below this.
api_key         = ""                      # Bearer token for hosted endpoints (OpenRouter, etc.).
timeout_seconds = 60                      # vLLM JSON-Schema responses can run long.
temperature     = 0.2
# system_prompt_file = "~/.config/engram/tagger-prompt.txt"
# Optional: replace the bundled v1 tagger prompt with the file's contents.
# You are responsible for also bumping model_version when overriding.
```

### Environment-variable overrides (worked examples)

The `ENGRAM_` prefix + `__` nesting maps to the TOML hierarchy. Examples:

```bash
# Snappier worker ticks for development (applies to both embed and tag drainers):
ENGRAM_WORKER__TICK_INTERVAL_SECONDS=2 cargo run --bin engram -- worker

# OpenRouter API key without checking it into config:
ENGRAM_TAGGER__API_KEY=sk-or-... \
ENGRAM_TAGGER__PROVIDER=openrouter \
ENGRAM_TAGGER__ENDPOINT='https://openrouter.ai/api/v1' \
ENGRAM_TAGGER__MODEL_NAME='anthropic/claude-haiku-4.5' \
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

- `[embedder]` reachable, for `search_thoughts` to populate the vector leg (soft-fails to trigram-only if unreachable; `vector_search_available: false` on responses).
- `[reranker]` configured + reachable, for the rerank stage to fire (soft-fails to RRF + recency if absent or down; `rerank_used: false` on responses).
- `[tagger]` configured (non-empty `provider`) + reachable, for the tag drainer to populate `thoughts.tags`. When omitted or empty, thoughts capture and search normally; `tags` stays `{}`.

The defaults wire up a working dev environment against `localhost` Postgres + Ollama; you only need an `engram.toml` when you're overriding something.

## Repo layout

```
crates/
├── engram-core/      # domain types, Embedder + Reranker + Tagger traits, RRF + recency_boost (pure)
├── engram-storage/   # sqlx queries, migrations, repository functions
├── engram-embed/     # Embedder + Reranker impls: OpenAICompatibleEmbedder, TeiReranker, fakes
├── engram-extract/   # Tagger impls: OpenAICompatibleTagger (vLLM/OpenRouter), FakeTagger
├── engram-mcp/       # capture/search/get/recent/retract/drain orchestrators + rmcp tool wiring
└── engram-cli/       # binary; serve/migrate/worker/embed-backfill/tag/bench subcommands
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
| [M2 — facts pipeline](docs/milestones/m2-facts-pipeline.md) | ✅ | Async embedding seam, reflector cron, `search_facts`, `correct_fact`, `engram reflect`. *(Superseded by M4; the facts pipeline was retired and replaced by a tagging sidecar.)* |
| [M3 — search & extraction quality](docs/milestones/m3-search-quality.md) | ✅ | Cross-encoder reranker; fact embeddings (M4-retired); v4 extractor prompt (M4-retired); A/B benchmarking harness. Retrieval portion shipped; extraction-side dogfood produced negative knowledge that motivated M4. |
| [M4 — collapse to thoughts-only](docs/milestones/m4-collapse-to-thoughts.md) | ✅ | Drop the facts pipeline; thoughts-only with content-fingerprint dedup and a JSONB tagging sidecar (people / action_items / topics / dates_mentioned / kind). `search_thoughts` gains `tag_filter`. Tagger is silent-disable. |
| [M4.1 — v2 tagging](docs/milestones/m4.1-tagging-v2.md) | ✅ | Split `topics` into `entities` (named identifiers) + `topics` (subject categories); add scope-aware controlled-vocabulary injection so the tagger prefers established terms over coining new ones. Initial v2 prompt shipped at tagger version 2; the v3 iteration added entities anti-padding + kind isolation (version 3); the v4 iteration restructured entities to lead-with-empty + softened vocab to tie-break after v3 dogfood revealed the negative-example list backfired (version 4). Operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill. |
| [M5 — selective relations](docs/milestones/m5-selective-relations.md) | ✅ | Thought-to-thought graph layer: closed relation vocabulary (initially `replaces`, `requires`, `references`, `belongs_to`, `decided_by`, `refines`; M5.1 added `supports` after dogfood), `thought_links` table, three new MCP tools (`link_thoughts`, `unlink_thoughts`, `get_related_thoughts`). Agent-supplied edges only; heterogeneous targets + tagger-extracted relations deferred. |
| [M6 — artifacts](docs/milestones/m6-artifacts.md) | ⏳ | Long-form document ingestion |
| [M7 — operational maturity](docs/milestones/m7-operational-maturity.md) | ⏳ | Metrics, Tier 2 auth, eval suite, backups |

Per-milestone progress is tracked in `docs/milestones/m{N}-progress.md`.

## License

TBD — not currently published.

# kengram-tagger-deterministic

Reference sidecar implementation for the kengram HTTP-tagger wire
contract. Bundles a Rust-native deterministic tagging pipeline
(`gline-rs` zero-shot NER + regex date extraction + `bge-m3` taxonomy
+ kind classification) behind an `axum` HTTP server that speaks
[`kengram-tagger-protocol`](../kengram-tagger-protocol/).

## What this is for

You have two reasons to care about this crate:

1. **Direct use.** Run the sidecar as-is, point kengram at it
   (`provider = "http"` in `[tagger]`), and tag thoughts without any
   LLM cost or hallucination risk. Trade-off: ~16-point quality gap
   on discourse pragmatics vs the LLM tagger (see Empirical findings
   below).

2. **Reference for writing your own.** Want a non-LLM tagger in
   Python, Go, Node, or anything else? This is a working example
   that speaks kengram's wire contract from end to end — read
   `src/main.rs` for the HTTP shape, read
   `../kengram-tagger-protocol/src/lib.rs` for the JSON spec, then
   reimplement in your language of choice.

## What this is NOT

A drop-in replacement for kengram's default LLM tagger. The 2026-05-24
head-to-head against `gemma3:12b` on 25 fixtures gave 20/25 (80%) vs
24/25 (96%). The 5 LLM-only wins were all discourse pragmatics
(use-vs-mention, role-vs-person, implicit action items, list
discrimination, nested quotation) — `gline-rs` is a token-level NER
model, not a discourse reasoner.

If your priority is the highest-quality tags, run the LLM tagger
(`provider = "openai-compatible"`). If your priority is sub-second
latency / zero LLM cost / deterministic behavior, run this sidecar.
The pluggability infrastructure means switching is one config line.

## Wire contract

Per [`kengram-tagger-protocol`](../kengram-tagger-protocol/) (the serde
derives ARE the spec). Quick reference:

```
POST {endpoint}/tag
Content-Type: application/json

{
  "protocol_version": "1",
  "content": "<thought text>",
  "vocab": { "topics": [...], "entities": [...] }   // optional
}

HTTP 200 OK
{
  "protocol_version": "1",
  "tags": {
    "people": [...],
    "entities": [],                  // always [] for this sidecar
    "action_items": [...],
    "topics": [...],
    "dates_mentioned": [...],
    "kind": "task"                   // or null
  },
  "relations": []                    // always [] for this sidecar
}
```

Plus `GET /health` for liveness probes — returns `{"status":"ok"}`.

Error responses:
- `400 Bad Request` — protocol_version mismatch
- `422 Unprocessable Entity` — non-transient pipeline error (drainer skips)
- `503 Service Unavailable` — transient error, drainer retries next tick

## Environment variables

All config via env. Twelve-factor.

| Var | Default | What |
|---|---|---|
| `BIND_ADDR` | `0.0.0.0:8082` | Listen address |
| `EMBEDDER_ENDPOINT` | `http://localhost:11434/v1` | OpenAI-compatible embedder (Ollama, TEI, OpenAI, ...) |
| `EMBEDDER_MODEL` | `bge-m3` | Model name as the embedder backend understands it |
| `EMBEDDER_API_KEY` | unset | Bearer token for the embedder, if needed |
| `GLINER_MODEL_DIR` | `$HOME/models/gliner_small-v2.1` | Directory with `tokenizer.json` + `onnx/model.onnx` |
| `TOPIC_TAXONOMY_PATH` | `topic-taxonomy.toml` (relative to CWD) | Operator-curated taxonomy TOML |
| `KIND_THRESHOLD` | `0.55` | Minimum cosine for kind classification argmax |
| `TOPIC_THRESHOLD` | `0.45` | Default topic-taxonomy threshold (per-label overrides in the TOML) |
| `MODEL_ID` | `deterministic/gliner-small-v2.1+regex+bge-m3` | Identity advertised + stamped on tag rows |
| `MODEL_VERSION` | `1` | Schema-version on tag rows |
| `RUST_LOG` | `info` | Standard tracing env-filter |

## Asset setup

### 1. Download the GLiNER ONNX model

~580MB. Not bundled into the Docker image; the operator supplies it.

```sh
mkdir -p ~/models/gliner_small-v2.1/onnx
curl -L -o ~/models/gliner_small-v2.1/tokenizer.json \
  https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/tokenizer.json
curl -L -o ~/models/gliner_small-v2.1/onnx/model.onnx \
  https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main/onnx/model.onnx
```

Final layout must be exactly:
```
$GLINER_MODEL_DIR/tokenizer.json
$GLINER_MODEL_DIR/onnx/model.onnx
```

### 2. Place the topic taxonomy

The seed taxonomy at `topic-taxonomy.toml` covers ~30 starter labels
for kengram-style content. Copy it somewhere your sidecar can read:

```sh
mkdir -p ~/.config/kengram
cp topic-taxonomy.toml ~/.config/kengram/topic-taxonomy.toml
```

Operators are expected to grow this against their corpus over time.
The TOML schema is `[topics.<label>] description = "..."` with optional
per-label `threshold` overrides.

## Running

### Cargo (development)

```sh
cargo run --release -p kengram-tagger-deterministic
# Or with custom env:
EMBEDDER_ENDPOINT=http://localhost:11434/v1 \
  GLINER_MODEL_DIR=$HOME/models/gliner_small-v2.1 \
  TOPIC_TAXONOMY_PATH=$PWD/crates/kengram-tagger-deterministic/topic-taxonomy.toml \
  cargo run --release -p kengram-tagger-deterministic
```

First request after startup loads the GLiNER ONNX model (~5–15s).
Subsequent requests are ~200ms CPU per thought (per Phase 0 spike on
Apple Silicon).

### Docker

```sh
docker build -t kengram-tagger-deterministic:latest \
  -f crates/kengram-tagger-deterministic/Dockerfile .

docker run --rm \
  -e EMBEDDER_ENDPOINT=http://host.docker.internal:11434/v1 \
  -e GLINER_MODEL_DIR=/models/gliner \
  -e TOPIC_TAXONOMY_PATH=/etc/kengram/topic-taxonomy.toml \
  -v $HOME/models/gliner_small-v2.1:/models/gliner:ro \
  -v $PWD/crates/kengram-tagger-deterministic/topic-taxonomy.toml:/etc/kengram/topic-taxonomy.toml:ro \
  -p 8082:8082 \
  kengram-tagger-deterministic:latest
```

### docker-compose snippet

Drop this alongside your existing kengram stack:

```yaml
services:
  tagger-deterministic:
    build:
      context: .
      dockerfile: crates/kengram-tagger-deterministic/Dockerfile
    environment:
      EMBEDDER_ENDPOINT: http://tei:80  # or your bge-m3 endpoint
      GLINER_MODEL_DIR: /models/gliner
      TOPIC_TAXONOMY_PATH: /etc/kengram/topic-taxonomy.toml
    volumes:
      - $HOME/models/gliner_small-v2.1:/models/gliner:ro
      - ./crates/kengram-tagger-deterministic/topic-taxonomy.toml:/etc/kengram/topic-taxonomy.toml:ro
    ports:
      - "8082:8082"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8082/health"]
      interval: 30s
      timeout: 5s
      start_period: 30s
      retries: 3
```

Then in your kengram.toml:

```toml
[tagger]
provider = "http"
model_id = "deterministic/gliner-small-v2.1+regex+bge-m3"
model_version = 1

[tagger.http]
endpoint = "http://localhost:8082"
timeout_seconds = 30
```

## Pipeline overview

Per thought, the sidecar runs:

1. **preprocess** — strips quoted spans, parenthetical `e.g.` / `such as`
   / `like` enumerations, and normalizes ALL-CAPS section headings.
   Defangs use-mention discourse failures before NER sees the text.
2. **NER** (gline-rs) — single zero-shot call with labels
   `[person, product, organization, title, action item, task to do]`.
   Person spans filtered against product/org/title overlap.
3. **dates** — regex-only surface-form extraction (ISO / year / decade
   / quarter / month-day / weekday / relative-period / n-ago / in-n).
   Deterministic by construction: the LLM's "1904 → 2004" transposition
   class of failure is impossible.
4. **kind** — bge-m3 embedding of the content vs 6 pre-embedded
   prototype paragraphs (one per `TagKind` variant). Cosine argmax
   above `KIND_THRESHOLD` or `null`.
5. **topics** — bge-m3 cosine of the content vs the topic taxonomy.
   Returns up to 3 labels above per-label threshold.

The taxonomy + kind prototypes are embedded **once** at startup;
per-thought work is one content embed plus pure cosine math plus one
gline-rs inference.

## 5-field schema

This sidecar always returns `entities: []` and `relations: []`. Rationale:
- `entities` drove the worst LLM-tagger failure modes (entities-adjectival
  overreach, people↔entities collision). Token-level NER doesn't have
  the discourse reasoning to do better.
- `relations` is LLM-specific extraction of `(from, relation, to)`
  edges from prose; not something a token-level NER pipeline produces
  meaningfully.

See [docs/tagger-improvements.md](../../docs/tagger-improvements.md) v14
section for the rationale and the head-to-head methodology.

## Empirical findings (2026-05-24)

Calibrated against local Ollama (`bge-m3`) + 25 fixtures
(`fixtures.json` here + `tests/fixtures/use_mention.json` in the
kengram repo):

| | Deterministic (this sidecar) | LLM (gemma3:12b v13) |
|---|---|---|
| Pass rate | 20/25 (80%) | 24/25 (96%) |
| Per-fixture latency | ~300ms | ~5–30s |

The 5 LLM-only wins all involve discourse pragmatics. The
1 deterministic-only win was a GitHub handle the v13 prompt extracted
as a person.

## Calibrating thresholds

`KIND_THRESHOLD = 0.55` and `TOPIC_THRESHOLD = 0.45` are the
2026-05-24 calibration values, validated against kengram's dogfood
content. Different corpus distributions may want different values:

- If `kind` is over-confident (assigning the wrong variant), raise
  `KIND_THRESHOLD` to 0.60-0.65.
- If `topics` is empty too often, lower `TOPIC_THRESHOLD` to 0.40 or
  add tighter per-label `threshold` entries in the taxonomy TOML.
- If a specific topic over-fires, set its per-label threshold to 0.55+
  in the taxonomy without affecting the default.

No code changes needed — just env vars + the TOML.

## Related

- [`../kengram-tagger-protocol/`](../kengram-tagger-protocol/) — wire spec
- [`../../docs/tagger-backends.md`](../../docs/tagger-backends.md) — the pluggability contract
- [`../../docs/tagger-sidecar-protocol.md`](../../docs/tagger-sidecar-protocol.md) — human-readable wire contract for non-Rust implementers
- [`../../docs/tagger-improvements.md`](../../docs/tagger-improvements.md) — v14 section: methodology + empirical findings

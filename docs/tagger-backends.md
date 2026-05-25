# Tagger backends — pluggability contract

Engram's tagger is pluggable. Anyone who implements the `Tagger` trait
in `engram-core` can swap in their own implementation; the operator
selects which one runs at config time. This document explains the
contract and the recipe to register a new backend.

## The contract

```rust
// crates/engram-core/src/tagger.rs

#[async_trait]
pub trait Tagger: Send + Sync {
    fn model_id(&self) -> &str;
    fn version(&self) -> i32;
    async fn tag(
        &self,
        thought_content: &str,
        vocab: Option<&ScopeVocab>,
    ) -> Result<TagOutput, TaggerError>;
}
```

That's the entire surface. A backend takes a thought's content (+
optional scope vocabulary hint) and returns a `TagOutput` bundling:

- `tags: Tags` — persisted metadata (people, entities, action_items,
  topics, dates_mentioned, kind). See `engram-core/src/tags.rs`.
- `relations: Vec<ExtractedRelation>` — thought→non-thought edges
  routed into `thought_links` by the drainer.

An empty `TagOutput` is a valid "nothing extractable here" answer; it
is not a failure. `TaggerError::is_transient()` distinguishes "the
drainer should retry next tick" from "give up and drop the job."

The trait is the contract. The `provider` config string + match arm
in `engram-cli/src/main.rs::build_tagger` is the *registry of known
implementations*, not the contract itself.

## Two wires, both pluggable

Engram ships with two HTTP-tagger clients built in. Both implement the
`Tagger` trait above and live in `engram-extract`; they differ in the
JSON shape they put on the wire:

| Provider | Wire shape | When to use |
|---|---|---|
| `openai-compatible` (default) | `POST /v1/chat/completions` with `response_format: json_schema`. Industry-standard LLM API. | LLM-backed tagging. Talks to vLLM, Ollama, OpenRouter, OpenAI, or anything that implements the OpenAI chat-completions API. |
| `openrouter` | Same as above. | Convenience alias documenting OpenRouter-hosted models. |
| `http` | `POST /tag` with the `engram-tagger-protocol` JSON shape. | Non-LLM taggers (or LLM taggers that don't fit the OpenAI shape). Run any HTTP service that speaks engram's wire contract. |
| `` (empty) | — | Silent disable. `pending_tags` jobs never enqueue; the worker doesn't spawn a tag drainer. |

The `openai-compatible` client exists because the OpenAI chat-completions
shape is a community standard — engram doesn't define it. The `http`
client exists for taggers that aren't LLMs (and therefore have no
standard wire). The `engram-tagger-protocol` crate documents that shape
in serde-derived Rust types; non-Rust sidecars implement the same JSON.

## How to add a backend

You have two paths depending on whether your tagger is in Rust:

### Path A — In-tree Rust backend

If your tagger is a Rust crate that depends on `engram-core`, implement
the trait directly:

1. Write a struct that implements `engram_core::Tagger`. It can live in
   any crate — inside the engram workspace as a new feature-gated module,
   or in your own crate that depends on `engram-core`.

   ```rust
   use engram_core::{Tagger, TagOutput, ScopeVocab};

   pub struct MyTagger { /* ... your state ... */ }

   #[async_trait::async_trait]
   impl Tagger for MyTagger {
       fn model_id(&self) -> &str { "myorg/my-tagger-v1" }
       fn version(&self) -> i32 { 1 }
       async fn tag(
           &self,
           content: &str,
           _vocab: Option<&ScopeVocab>,
       ) -> Result<TagOutput, engram_core::TaggerError> {
           // ... your extraction logic ...
           Ok(TagOutput::default())
       }
   }
   ```

2. Add a config sub-section in `crates/engram-cli/src/config.rs` if you
   need knobs beyond what `openai-compatible` already uses (e.g.
   `Option<MyTaggerConfig>` field on `TaggerConfig`).

3. Add a match arm in `crates/engram-cli/src/main.rs::build_tagger`
   mapping a new provider string to your struct's constructor.

### Path B — Out-of-tree sidecar (any language)

If your tagger is in Python, Go, Node, or anything that isn't Rust, run
it as an HTTP sidecar and point engram at it via `provider = "http"`:

1. Implement an HTTP server that accepts `POST /tag` per the
   `engram-tagger-protocol` JSON wire contract. The reference sidecar at
   `crates/engram-tagger-deterministic/` is a working example you can
   study or fork.

2. Set the operator config:

   ```toml
   [tagger]
   provider = "http"
   model_id = "myorg/my-sidecar-v1"
   model_version = 1

   [tagger.http]
   endpoint = "http://localhost:8081"
   timeout_seconds = 30
   ```

3. Run your sidecar. Engram's worker tags thoughts by calling your
   endpoint; failures with 5xx + connection-level errors are treated as
   transient (retried next tick), 4xx + malformed responses are
   non-transient (logged + skipped).

The reference sidecar ships as a binary you can `docker run` or just
`cargo run`. See its README for the asset + env-var setup.

## Why this design

The pluggability isn't bolted on — it's the natural shape of the
abstraction. Two reasons it matters:

1. **Single-user, on-prem.** Engram doesn't run a SaaS tagger; the
   operator's deployment IS the deployment. Different deployments may
   have different cost/latency/quality trade-offs (a 12B LLM on a
   workstation, a small NER pipeline in a Docker container, an external
   service you trust). The trait lets each deployment pick.
2. **Empirical iteration.** The `examples/tagger_eval.rs` harness can
   compare any two backends against the same fixture set via
   `TAGGER_BACKEND={openai-compatible,http}`. This is the path used to
   ship v13 (LLM prompt iteration) and the path used to validate the
   reference sidecar against the LLM backend before cutover.

## Related references

- `crates/engram-core/src/tagger.rs` — trait definition + doc-comments
- `crates/engram-cli/src/main.rs::build_tagger` — provider registry
- `crates/engram-cli/src/config.rs::TaggerConfig` — config plumbing
- `crates/engram-tagger-protocol/` — wire-shape spec for sidecars
- `crates/engram-tagger-deterministic/` — reference sidecar implementation
- `docs/tagger-sidecar-protocol.md` — human-readable wire contract for non-Rust implementers
- `docs/tagger-improvements.md` — historical record of tagger work
  (v6 through v13 LLM prompt iteration + the v14 deterministic-backend
  exploration that produced the reference sidecar)

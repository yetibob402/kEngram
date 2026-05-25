//! engram-extract: `Tagger` implementations.
//!
//! - [`OpenAICompatibleTagger`] talks to anything that speaks the OpenAI
//!   `/v1/chat/completions` API with `response_format: json_schema` — vLLM
//!   (production sidecar), OpenRouter (cloud fallback), OpenAI itself.
//!   Distinguished only by config.
//! - [`HttpTagger`] talks to a sidecar speaking engram's own JSON wire
//!   contract (see `engram-tagger-protocol`). Use for non-LLM taggers
//!   that don't fit OpenAI semantics. Reference sidecar lives at
//!   `crates/engram-tagger-deterministic/`.
//! - [`FakeTagger`] is a deterministic in-memory tagger for tests;
//!   mirrors `engram-embed::FakeEmbedder` in shape.
//!
//! The `Tagger` trait itself lives in `engram-core` (`engram_core::Tagger`)
//! so the tag drainer loop in `engram-mcp` / `engram-cli` can depend on the
//! abstraction without pulling in this crate's HTTP machinery.

pub mod fake_tagger;
pub mod http_tagger;
pub mod openai_compatible;

pub use fake_tagger::{FakeBehavior, FakeTagger, FakeTaggerOutput, RecordedTag};
pub use http_tagger::{HttpTagger, HttpTaggerConfig};
pub use openai_compatible::{
    BUNDLED_TAGGER_PROMPT, BUNDLED_TAGGER_VERSION, OpenAICompatibleConfig, OpenAICompatibleTagger,
};

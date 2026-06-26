//! kengram-embed: `Embedder` and `Reranker` implementations.
//!
//! Embedders:
//! - [`OpenAICompatibleEmbedder`] talks to anything speaking the OpenAI
//!   `/v1/embeddings` API — Ollama (dev default), TEI sidecar (production),
//!   OpenAI, Voyage. Endpoint and model name come from config.
//! - [`FakeEmbedder`] is a deterministic in-memory embedder for tests; it
//!   does not require Ollama / TEI to be running.
//!
//! Rerankers (M3 Phase B step 2):
//! - [`TeiReranker`] talks to Hugging Face's text-embeddings-inference
//!   sidecar in rerank-task mode. Default deployment is a Docker container
//!   serving BGE-reranker-v2-m3 on port 8080.
//! - [`FakeReranker`] is a deterministic in-memory reranker for tests with
//!   pluggable scoring (position-descending / substring-boost / etc.).

pub mod fake;
pub mod fake_reranker;
pub mod openai_compatible;
pub mod openai_compatible_sparse;
pub mod reranker;
pub mod tei_reranker;

pub use fake::{FakeBehavior, FakeEmbedder};
pub use fake_reranker::{FakeReranker, FakeRerankerBehavior, FakeRerankerScoring, RecordedRerank};
pub use openai_compatible::{OpenAICompatibleConfig, OpenAICompatibleEmbedder};
pub use openai_compatible_sparse::{OpenAICompatibleSparseConfig, OpenAICompatibleSparseEmbedder};
pub use reranker::{RerankScore, Reranker, RerankerError};
pub use tei_reranker::{TeiReranker, TeiRerankerConfig};

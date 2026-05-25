//! kengram-core: domain types, the `Embedder` trait, and retrieval fusion logic.
//!
//! Pure logic, no I/O. Implementations live in `kengram-storage` (Postgres),
//! `kengram-embed` (TEI/Ollama/OpenAI), and `kengram-mcp` (rmcp tool handlers).

pub mod embedder;
pub mod embedding;
pub mod metadata;
pub mod metrics;
pub mod relation;
pub mod scope;
pub mod search;
pub mod source;
pub mod tagger;
pub mod tags;
pub mod thought;

pub use embedder::{Embedder, EmbedderError};
pub use embedding::{Embedding, EmbeddingError, EmbeddingModel, EmbeddingStatus};
pub use metadata::Metadata;
pub use metrics::{ndcg_at_k, reciprocal_rank};
pub use relation::{
    LinkDirection, LinkId, LinkSource, LinkTarget, RelationKind, ThoughtLink, UnknownLinkDirection,
    UnknownLinkSource, UnknownRelationKind,
};
pub use scope::{Scope, ScopeError};
pub use search::{DEFAULT_RECENCY_HALF_LIFE_DAYS, DEFAULT_RRF_K, Hit, recency_boost, rrf_fuse};
pub use source::{Source, SourceError};
pub use tagger::{ExtractedRelation, ExtractedTarget, TagOutput, Tagger, TaggerError};
pub use tags::{ScopeVocab, TagKind, Tags};
pub use thought::{Thought, ThoughtId};

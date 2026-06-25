//! The `Embedder` trait — the seam between kengram and whatever backend
//! produces embedding vectors. Implementations live in `kengram-embed`.

use async_trait::async_trait;

use crate::{EmbeddingModel, SparseEmbeddingModel, SparseLexicalVector};

#[async_trait]
pub trait Embedder: Send + Sync {
    /// The model identity. The returned `EmbeddingModel.id` must match the
    /// HNSW partial index in Postgres (e.g. `"bge-m3:1024"`).
    fn model(&self) -> &EmbeddingModel;

    /// Embed a batch of texts. Returns one vector per input, each of length
    /// `self.model().dimensions`. The order of outputs matches the order of
    /// inputs.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError>;
}

#[async_trait]
pub trait SparseEmbedder: Send + Sync {
    /// Sparse model identity. BGE-M3 lexical weights use the tokenizer
    /// vocabulary size rather than the dense vector dimensionality.
    fn sparse_model(&self) -> &SparseEmbeddingModel;

    /// Encode a batch of texts into sparse lexical weights. This is separate
    /// from [`Embedder::embed`] because OpenAI-compatible dense endpoints may
    /// not expose sparse BGE-M3 weights.
    async fn encode_sparse(
        &self,
        texts: &[String],
    ) -> Result<Vec<SparseLexicalVector>, EmbedderError>;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("embedder timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("embedder backend unreachable: {0}")]
    Unreachable(String),

    #[error("embedder returned malformed response: {0}")]
    MalformedResponse(String),

    #[error("embedder returned wrong dimensions: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("embedder backend reported error (status {status}): {message}")]
    Backend { status: u16, message: String },

    #[error("input batch was empty")]
    EmptyBatch,
}

impl EmbedderError {
    /// True when the failure is something a retry might fix (network blip,
    /// timeout, transient backend error). Used by capture to decide whether
    /// to mark `embedding_status: "pending"` vs. surface a hard error.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. }
                | Self::Unreachable(_)
                | Self::Backend {
                    status: 500..=599,
                    ..
                }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_is_transient() {
        assert!(EmbedderError::Timeout { seconds: 5 }.is_transient());
    }

    #[test]
    fn unreachable_is_transient() {
        assert!(EmbedderError::Unreachable("connection refused".into()).is_transient());
    }

    #[test]
    fn server_5xx_is_transient() {
        assert!(
            EmbedderError::Backend {
                status: 503,
                message: "unavailable".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn client_4xx_is_not_transient() {
        assert!(
            !EmbedderError::Backend {
                status: 400,
                message: "bad request".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn malformed_is_not_transient() {
        assert!(!EmbedderError::MalformedResponse("nope".into()).is_transient());
    }

    #[test]
    fn dimension_mismatch_is_not_transient() {
        assert!(
            !EmbedderError::DimensionMismatch {
                expected: 1024,
                got: 512,
            }
            .is_transient()
        );
    }
}

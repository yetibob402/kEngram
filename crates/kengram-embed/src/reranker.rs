//! The `Reranker` trait — the seam between Kengram's search pipeline and a
//! cross-encoder reranker backend. Implementations live in this crate
//! (`TeiReranker` for the TEI sidecar; `FakeReranker` for tests).
//!
//! Mirrors the `Embedder` trait in shape and `is_transient()` discipline:
//! the search orchestrator (in `kengram-mcp/src/search.rs`) uses
//! `is_transient()` to decide whether a per-request rerank failure should
//! soft-fail back to the RRF + recency pipeline (response carries
//! `rerank_used: false`) or surface as an error.

use async_trait::async_trait;

/// A single rerank score for one candidate. `index` is the position in the
/// `candidates` slice that was passed to `Reranker::rerank`; `score` is the
/// model's calibrated relevance score, typically (but not strictly) in
/// `[0.0, 1.0]`. Higher is more relevant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankScore {
    pub index: usize,
    pub score: f32,
}

#[async_trait]
pub trait Reranker: Send + Sync {
    /// Stable model identifier — conventionally `<vendor>/<model>`, e.g.
    /// `"BAAI/bge-reranker-v2-m3"`. Written into startup logs and (later)
    /// observable per response for provenance.
    fn model_id(&self) -> &str;

    /// Score each candidate against the query. Returns one `RerankScore`
    /// per input candidate; ordering of the returned vec is NOT guaranteed
    /// to be sorted — callers consult `RerankScore.index` to map back.
    ///
    /// `candidates` is borrowed as `&[&str]` rather than `&[String]` so the
    /// orchestrator can pass slices over the existing post-RRF hit list
    /// without re-cloning every candidate's text.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<RerankScore>, RerankerError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RerankerError {
    #[error("reranker timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("reranker backend unreachable: {0}")]
    Unreachable(String),

    #[error("reranker returned malformed response: {0}")]
    MalformedResponse(String),

    #[error("reranker backend reported error (status {status}): {message}")]
    Backend { status: u16, message: String },

    #[error("reranker is misconfigured: {0}")]
    Misconfigured(String),
}

impl RerankerError {
    /// True when the failure is something the next request might resolve
    /// on its own (network blip, timeout, transient 5xx). The search
    /// orchestrator can choose to soft-fail (return RRF-only) on transient
    /// errors and log + propagate non-transient ones.
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
        assert!(RerankerError::Timeout { seconds: 5 }.is_transient());
    }

    #[test]
    fn unreachable_is_transient() {
        assert!(RerankerError::Unreachable("connection refused".into()).is_transient());
    }

    #[test]
    fn server_5xx_is_transient() {
        assert!(
            RerankerError::Backend {
                status: 503,
                message: "unavailable".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn client_4xx_is_not_transient() {
        assert!(
            !RerankerError::Backend {
                status: 400,
                message: "bad request".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn malformed_is_not_transient() {
        assert!(!RerankerError::MalformedResponse("nope".into()).is_transient());
    }

    #[test]
    fn misconfigured_is_not_transient() {
        assert!(!RerankerError::Misconfigured("missing endpoint".into()).is_transient());
    }
}

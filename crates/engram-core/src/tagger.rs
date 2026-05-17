//! The `Tagger` trait — the seam between engram and whatever backend
//! turns a thought's content into a bag of metadata tags. Implementations
//! live in `engram-extract`.
//!
//! Mirrors the `Embedder` / `Reranker` trait shape and `is_transient()`
//! discipline. The tag drainer loop uses `is_transient()` to decide whether
//! to soft-fail (leave the job in `pending_tags` with attempts++) and retry
//! on the next tick, or to log and skip without retry.

use async_trait::async_trait;

use crate::{ScopeVocab, Tags};

#[async_trait]
pub trait Tagger: Send + Sync {
    /// Stable model identifier — written into `thoughts.tags_extractor_model`
    /// for provenance. Conventionally `"<vendor>/<model>"`, e.g.
    /// `"openai/gpt-4o-mini"` or `"vllm/qwen2.5-7b-instruct"`.
    fn model_id(&self) -> &str;

    /// Schema-version of *this* tagger's prompt/response contract. Bump
    /// when the JSON Schema or system prompt changes in a way that makes
    /// prior tags no longer comparable. The drainer writes this into
    /// `thoughts.tags_extractor_version`.
    fn version(&self) -> i32;

    /// Tag a single thought's content. `vocab`, when supplied, lists the
    /// established topic and entity terms most frequently used in the
    /// thought's scope; implementations should encourage the model to prefer
    /// those terms when they fit, and coin new ones only for genuinely
    /// unseen concepts. Passing `None` runs the tagger without any
    /// vocabulary guidance.
    ///
    /// Returning a `Tags::default()` is a valid "no extractable tags here"
    /// answer and is not a failure.
    async fn tag(
        &self,
        thought_content: &str,
        vocab: Option<&ScopeVocab>,
    ) -> Result<Tags, TaggerError>;
}

#[derive(Debug, thiserror::Error)]
pub enum TaggerError {
    #[error("tagger endpoint unreachable: {0}")]
    Unreachable(String),

    #[error("tagger timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("tagger returned malformed response: {0}")]
    MalformedResponse(String),

    #[error("tagger misconfigured: {0}")]
    Misconfigured(String),

    #[error("tagger backend error (status {status}): {body}")]
    Backend { status: u16, body: String },
}

impl TaggerError {
    /// True when the failure is something the next drainer tick might
    /// resolve on its own (network blip, timeout, transient 5xx). The
    /// drainer soft-fails per thought on transient errors and continues;
    /// on non-transient errors it logs and removes the job from the queue.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Unreachable(_)
                | Self::Timeout { .. }
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
    fn unreachable_is_transient() {
        assert!(TaggerError::Unreachable("connection refused".into()).is_transient());
    }

    #[test]
    fn timeout_is_transient() {
        assert!(TaggerError::Timeout { seconds: 5 }.is_transient());
    }

    #[test]
    fn server_5xx_is_transient() {
        assert!(
            TaggerError::Backend {
                status: 503,
                body: "unavailable".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn client_4xx_is_not_transient() {
        assert!(
            !TaggerError::Backend {
                status: 400,
                body: "bad request".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn malformed_is_not_transient() {
        assert!(!TaggerError::MalformedResponse("nope".into()).is_transient());
    }

    #[test]
    fn misconfigured_is_not_transient() {
        assert!(!TaggerError::Misconfigured("missing api key".into()).is_transient());
    }
}

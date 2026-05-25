//! `FakeReranker` — deterministic in-memory `Reranker` for tests.
//!
//! Mirrors `FakeEmbedder` / `FakeExtractor` in shape: configurable
//! behavior (deterministic / always-fail-with-X), records the last call
//! for assertions, and a small set of canned scoring strategies that
//! let reflect / search tests assert "this fact ranked first" without
//! a live TEI sidecar.

use async_trait::async_trait;
use std::sync::{Arc, Mutex};

use crate::reranker::{RerankScore, Reranker, RerankerError};

#[derive(Debug, Clone)]
pub struct FakeReranker {
    model_id: String,
    behavior: FakeRerankerBehavior,
    scoring: FakeRerankerScoring,
    /// Records the most recent successful `rerank` call so tests can
    /// assert what the search orchestrator asked for. `Arc<Mutex<_>>`
    /// because `FakeReranker` is passed by shared reference through
    /// the search pipeline.
    last_call: Arc<Mutex<Option<RecordedRerank>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeRerankerBehavior {
    Deterministic,
    Timeout,
    Unreachable,
    Misconfigured,
}

#[derive(Debug, Clone)]
pub enum FakeRerankerScoring {
    /// Score = 1.0 - position * 0.05 (position 0 → 1.0, position 1 → 0.95,
    /// …). Useful for "rerank produced *some* score for every input" tests
    /// where the actual ranking doesn't matter.
    PositionDescending,
    /// Score = 1.0 if the candidate text contains the substring; else 0.0.
    /// Useful for tests that need to *reorder* a candidate set
    /// deterministically (e.g. "after rerank, the candidate containing
    /// 'Nix' should rank first").
    SubstringBoost { needle: String },
    /// Reverse: 1.0 - PositionDescending. Useful for "did the orchestrator
    /// re-sort by rerank_score" tests where the trigram leg ordered the
    /// candidates one way and we need rerank to clearly invert it.
    PositionAscending,
}

#[derive(Debug, Clone)]
pub struct RecordedRerank {
    pub query: String,
    pub candidates: Vec<String>,
}

impl FakeReranker {
    /// Default: deterministic + position-descending scoring + the
    /// canonical `BAAI/bge-reranker-v2-m3` model id (so test assertions
    /// against `model_id` work without spelling it out everywhere).
    pub fn new() -> Self {
        Self {
            model_id: "BAAI/bge-reranker-v2-m3".to_string(),
            behavior: FakeRerankerBehavior::Deterministic,
            scoring: FakeRerankerScoring::PositionDescending,
            last_call: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_scoring(scoring: FakeRerankerScoring) -> Self {
        Self {
            scoring,
            ..Self::new()
        }
    }

    pub fn always_failing(behavior: FakeRerankerBehavior) -> Self {
        Self {
            behavior,
            ..Self::new()
        }
    }

    pub fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Self::new()
        }
    }

    /// Returns a clone of the most recent successful `rerank` call's
    /// arguments. Tests assert against this to confirm the search
    /// orchestrator passed the right query + candidate list.
    pub fn last_call(&self) -> Option<RecordedRerank> {
        self.last_call
            .lock()
            .expect("last_call mutex poisoned")
            .clone()
    }
}

impl Default for FakeReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reranker for FakeReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn rerank(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        match self.behavior {
            FakeRerankerBehavior::Timeout => return Err(RerankerError::Timeout { seconds: 5 }),
            FakeRerankerBehavior::Unreachable => {
                return Err(RerankerError::Unreachable(
                    "fake reranker configured to fail".into(),
                ));
            }
            FakeRerankerBehavior::Misconfigured => {
                return Err(RerankerError::Misconfigured(
                    "fake reranker configured to fail".into(),
                ));
            }
            FakeRerankerBehavior::Deterministic => {}
        }

        *self.last_call.lock().expect("last_call mutex poisoned") = Some(RecordedRerank {
            query: query.to_string(),
            candidates: candidates.iter().map(|s| s.to_string()).collect(),
        });

        let scores: Vec<RerankScore> = candidates
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let score = match &self.scoring {
                    FakeRerankerScoring::PositionDescending => 1.0 - (i as f32) * 0.05,
                    FakeRerankerScoring::PositionAscending => (i as f32) * 0.05,
                    FakeRerankerScoring::SubstringBoost { needle } => {
                        if text.contains(needle.as_str()) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
                RerankScore { index: i, score }
            })
            .collect();
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn position_descending_scoring() {
        let r = FakeReranker::new();
        let scores = r.rerank("q", &["a", "b", "c"]).await.unwrap();
        assert_eq!(scores.len(), 3);
        assert!((scores[0].score - 1.0).abs() < 1e-5);
        assert!((scores[1].score - 0.95).abs() < 1e-5);
        assert!((scores[2].score - 0.90).abs() < 1e-5);
    }

    #[tokio::test]
    async fn substring_boost_picks_matching_candidate() {
        let r = FakeReranker::with_scoring(FakeRerankerScoring::SubstringBoost {
            needle: "Nix".into(),
        });
        let scores = r
            .rerank(
                "reproducibility",
                &["Bazel is powerful", "Nix is reproducible", "Redis is fast"],
            )
            .await
            .unwrap();
        // The "Nix is reproducible" candidate (index 1) should score 1.0.
        assert!((scores[1].score - 1.0).abs() < 1e-5);
        // Others score 0.0.
        assert!(scores[0].score.abs() < 1e-5);
        assert!(scores[2].score.abs() < 1e-5);
    }

    #[tokio::test]
    async fn records_last_call() {
        let r = FakeReranker::new();
        r.rerank("query text", &["a", "b"]).await.unwrap();
        let call = r.last_call().expect("rerank was called");
        assert_eq!(call.query, "query text");
        assert_eq!(call.candidates, vec!["a".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn always_failing_timeout_is_transient() {
        let r = FakeReranker::always_failing(FakeRerankerBehavior::Timeout);
        let err = r.rerank("q", &["a"]).await.unwrap_err();
        assert!(matches!(err, RerankerError::Timeout { .. }));
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn always_failing_unreachable_is_transient() {
        let r = FakeReranker::always_failing(FakeRerankerBehavior::Unreachable);
        let err = r.rerank("q", &["a"]).await.unwrap_err();
        assert!(matches!(err, RerankerError::Unreachable(_)));
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn empty_candidates_records_nothing() {
        let r = FakeReranker::new();
        let scores = r.rerank("q", &[]).await.unwrap();
        assert!(scores.is_empty());
        // FakeReranker still records the call (the impl always records on
        // Deterministic, including the empty case). The TeiReranker impl
        // bails before HTTP; FakeReranker doesn't need that optimization.
        let call = r.last_call().expect("call recorded");
        assert!(call.candidates.is_empty());
    }
}

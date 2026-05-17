//! `FakeEmbedder` — a deterministic, in-memory `Embedder` for tests.
//!
//! Given the same input, returns the same vector — useful for asserting
//! exact equality of inserted embeddings in sqlx-tests. Configurable to
//! always fail in specific ways for testing soft-fail paths.

use async_trait::async_trait;
use engram_core::{Embedder, EmbedderError, EmbeddingModel};

#[derive(Debug, Clone)]
pub struct FakeEmbedder {
    model: EmbeddingModel,
    behavior: FakeBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeBehavior {
    /// Return a deterministic vector derived from each input string.
    Deterministic,
    /// Always fail with `EmbedderError::Timeout`.
    Timeout,
    /// Always fail with `EmbedderError::Unreachable`.
    Unreachable,
}

impl FakeEmbedder {
    /// New deterministic embedder with the BGE-M3 model (1024 dims).
    pub fn new() -> Self {
        Self::with_model(EmbeddingModel::bge_m3())
    }

    pub fn with_model(model: EmbeddingModel) -> Self {
        Self {
            model,
            behavior: FakeBehavior::Deterministic,
        }
    }

    /// Build a copy of this embedder that always fails with the given behavior.
    pub fn always_failing(model: EmbeddingModel, behavior: FakeBehavior) -> Self {
        Self { model, behavior }
    }
}

impl Default for FakeEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for FakeEmbedder {
    fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Err(EmbedderError::EmptyBatch);
        }
        match self.behavior {
            FakeBehavior::Timeout => Err(EmbedderError::Timeout { seconds: 5 }),
            FakeBehavior::Unreachable => Err(EmbedderError::Unreachable(
                "fake embedder configured to fail".into(),
            )),
            FakeBehavior::Deterministic => Ok(texts
                .iter()
                .map(|t| deterministic_vector(t, self.model.dimensions))
                .collect()),
        }
    }
}

/// Produce a deterministic vector from a string. Not cryptographic; just
/// stable across calls within a process. Output range: roughly [-1, 1].
fn deterministic_vector(input: &str, dim: usize) -> Vec<f32> {
    let seed: u64 = input.bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(u64::from(b))
    });
    (0..dim)
        .map(|i| {
            let mut x = seed.wrapping_add(i as u64);
            // splitmix64 step — fast, deterministic, decent distribution
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^= x >> 31;
            // Map u64 to roughly [-1, 1]
            ((x as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_vectors_of_correct_dimensions() {
        let e = FakeEmbedder::new();
        let out = e.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1024);
    }

    #[tokio::test]
    async fn deterministic_for_same_input() {
        let e = FakeEmbedder::new();
        let a = e.embed(&["hello".to_string()]).await.unwrap();
        let b = e.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn different_inputs_produce_different_vectors() {
        let e = FakeEmbedder::new();
        let a = e.embed(&["hello".to_string()]).await.unwrap();
        let b = e.embed(&["world".to_string()]).await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn batches_preserve_order() {
        let e = FakeEmbedder::new();
        let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let out = e.embed(&inputs).await.unwrap();
        assert_eq!(out.len(), 3);
        // Each output should match the singleton embedding of the corresponding input.
        for (i, text) in inputs.iter().enumerate() {
            let solo = e.embed(std::slice::from_ref(text)).await.unwrap();
            assert_eq!(out[i], solo[0]);
        }
    }

    #[tokio::test]
    async fn empty_batch_errors() {
        let e = FakeEmbedder::new();
        assert!(matches!(e.embed(&[]).await, Err(EmbedderError::EmptyBatch)));
    }

    #[tokio::test]
    async fn timeout_behavior_returns_timeout_error() {
        let e = FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Timeout);
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        assert!(matches!(err, EmbedderError::Timeout { .. }));
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn unreachable_behavior_returns_unreachable_error() {
        let e = FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Unreachable);
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        assert!(matches!(err, EmbedderError::Unreachable(_)));
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn model_dimensions_can_be_customised() {
        let m = EmbeddingModel::new("tiny:8", 8);
        let e = FakeEmbedder::with_model(m);
        let out = e.embed(&["x".to_string()]).await.unwrap();
        assert_eq!(out[0].len(), 8);
    }
}

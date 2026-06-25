//! Sparse lexical embedding contract for BGE-M3-style lexical weights.
//!
//! The dense [`crate::Embedder`] contract intentionally stays dense-only.
//! Sparse lexical weights are a separate producer path because not every
//! OpenAI-compatible embedding backend exposes BGE-M3 `lexical_weights`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SparseEmbeddingModel {
    pub id: String,
    pub version: i32,
    pub vocab_size: usize,
}

impl SparseEmbeddingModel {
    pub fn new(id: impl Into<String>, version: i32, vocab_size: usize) -> Self {
        Self {
            id: id.into(),
            version,
            vocab_size,
        }
    }

    /// BGE-M3 sparse lexical weights emitted by FlagEmbedding.
    pub fn bge_m3_sparse() -> Self {
        Self::new("bge-m3:sparse", 1, 250_002)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseWeight {
    /// Zero-based tokenizer token id from the sparse producer.
    pub token_id: u32,
    pub weight: f32,
}

impl SparseWeight {
    pub fn new(token_id: u32, weight: f32) -> Self {
        Self { token_id, weight }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseLexicalVector {
    pub model: SparseEmbeddingModel,
    weights: Vec<SparseWeight>,
}

#[derive(Debug, thiserror::Error)]
pub enum SparseVectorError {
    #[error("sparse model vocab size must be positive")]
    EmptyVocab,

    #[error("sparse vector has no nonzero weights")]
    EmptyWeights,

    #[error("sparse token id {token_id} outside vocab size {vocab_size}")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },

    #[error("sparse weight for token {token_id} is not finite")]
    NonFiniteWeight { token_id: u32 },
}

impl SparseLexicalVector {
    pub fn new(
        model: SparseEmbeddingModel,
        weights: Vec<SparseWeight>,
    ) -> Result<Self, SparseVectorError> {
        if model.vocab_size == 0 {
            return Err(SparseVectorError::EmptyVocab);
        }

        let mut merged = BTreeMap::<u32, f32>::new();
        for SparseWeight { token_id, weight } in weights {
            if token_id as usize >= model.vocab_size {
                return Err(SparseVectorError::TokenOutOfRange {
                    token_id,
                    vocab_size: model.vocab_size,
                });
            }
            if !weight.is_finite() {
                return Err(SparseVectorError::NonFiniteWeight { token_id });
            }
            if weight != 0.0 {
                *merged.entry(token_id).or_insert(0.0) += weight;
            }
        }

        let weights = merged
            .into_iter()
            .filter_map(|(token_id, weight)| {
                if weight == 0.0 {
                    None
                } else {
                    Some(SparseWeight { token_id, weight })
                }
            })
            .collect::<Vec<_>>();

        if weights.is_empty() {
            return Err(SparseVectorError::EmptyWeights);
        }

        Ok(Self { model, weights })
    }

    pub fn weights(&self) -> &[SparseWeight] {
        &self.weights
    }

    pub fn nonzero_count(&self) -> usize {
        self.weights.len()
    }

    /// Convert zero-based tokenizer ids to pgvector's one-based `sparsevec`
    /// literal format, e.g. `{1:0.5,3:1.25}/250002`.
    pub fn sparsevec_literal(&self) -> String {
        let entries = self
            .weights
            .iter()
            .map(|w| format!("{}:{}", w.token_id + 1, w.weight))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{entries}}}/{}", self.model.vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bge_m3_sparse_model_contract_is_stable() {
        let model = SparseEmbeddingModel::bge_m3_sparse();
        assert_eq!(model.id, "bge-m3:sparse");
        assert_eq!(model.version, 1);
        assert_eq!(model.vocab_size, 250_002);
    }

    #[test]
    fn sparsevec_literal_sorts_merges_and_converts_to_one_based_indices() {
        let model = SparseEmbeddingModel::new("test:sparse", 1, 6);
        let v = SparseLexicalVector::new(
            model,
            vec![
                SparseWeight::new(2, 1.25),
                SparseWeight::new(0, 0.5),
                SparseWeight::new(2, 0.75),
            ],
        )
        .unwrap();

        assert_eq!(v.nonzero_count(), 2);
        assert_eq!(v.sparsevec_literal(), "{1:0.5,3:2}/6");
    }

    #[test]
    fn sparsevec_literal_allows_last_vocab_token() {
        let model = SparseEmbeddingModel::new("test:sparse", 1, 250_002);
        let v = SparseLexicalVector::new(model, vec![SparseWeight::new(250_001, 1.0)]).unwrap();
        assert_eq!(v.sparsevec_literal(), "{250002:1}/250002");
    }

    #[test]
    fn sparse_vector_rejects_out_of_range_token() {
        let model = SparseEmbeddingModel::new("test:sparse", 1, 3);
        let err = SparseLexicalVector::new(model, vec![SparseWeight::new(3, 1.0)]).unwrap_err();
        assert!(matches!(
            err,
            SparseVectorError::TokenOutOfRange {
                token_id: 3,
                vocab_size: 3
            }
        ));
    }

    #[test]
    fn sparse_vector_rejects_nonfinite_weight() {
        let model = SparseEmbeddingModel::new("test:sparse", 1, 3);
        let err =
            SparseLexicalVector::new(model, vec![SparseWeight::new(1, f32::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            SparseVectorError::NonFiniteWeight { token_id: 1 }
        ));
    }
}

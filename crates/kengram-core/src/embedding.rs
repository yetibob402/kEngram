//! Embedding-related types. `EmbeddingModel` identifies the active embedder
//! by `model_id` (the same string that predicates the HNSW partial index in
//! Postgres). `Embedding` carries the vector itself.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// `"bge-m3:1024"`, `"voyage-3:1024"`, etc. Conventionally `"<name>:<dim>"`.
    pub id: String,
    pub dimensions: usize,
}

impl EmbeddingModel {
    pub fn new(id: impl Into<String>, dimensions: usize) -> Self {
        Self {
            id: id.into(),
            dimensions,
        }
    }

    /// BGE-M3 1024-dim — the default for M1 (the active model on the HNSW
    /// partial index `embeddings_bge_m3_hnsw`).
    pub fn bge_m3() -> Self {
        Self::new("bge-m3:1024", 1024)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub model: EmbeddingModel,
    pub vector: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("vector length {got} does not match model dimensions {expected}")]
    DimensionMismatch { expected: usize, got: usize },
}

impl Embedding {
    pub fn new(model: EmbeddingModel, vector: Vec<f32>) -> Result<Self, EmbeddingError> {
        if vector.len() != model.dimensions {
            return Err(EmbeddingError::DimensionMismatch {
                expected: model.dimensions,
                got: vector.len(),
            });
        }
        Ok(Self { model, vector })
    }
}

/// Indicator returned in capture responses so callers know whether a search
/// will immediately surface the new thought via vector retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStatus {
    Indexed,
    Pending,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bge_m3_constants() {
        let m = EmbeddingModel::bge_m3();
        assert_eq!(m.id, "bge-m3:1024");
        assert_eq!(m.dimensions, 1024);
    }

    #[test]
    fn embedding_constructs_with_correct_dims() {
        let m = EmbeddingModel::new("test", 3);
        let e = Embedding::new(m.clone(), vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(e.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(e.model, m);
    }

    #[test]
    fn embedding_rejects_dimension_mismatch() {
        let m = EmbeddingModel::new("test", 3);
        let err = Embedding::new(m, vec![0.1, 0.2]).unwrap_err();
        assert!(matches!(
            err,
            EmbeddingError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn embedding_status_serialises_snake_case() {
        assert_eq!(
            serde_json::to_string(&EmbeddingStatus::Indexed).unwrap(),
            "\"indexed\""
        );
        assert_eq!(
            serde_json::to_string(&EmbeddingStatus::Pending).unwrap(),
            "\"pending\""
        );
    }
}

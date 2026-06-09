//! `retract_thought` — operator-driven path that marks a thought
//! untrusted. Sets `thoughts.retracted_at = NOW()` (preserves the row;
//! preserves the operator's reason).
//!
//! Post-M4: no more fact-cascade. The facts table is gone; retracting a
//! thought is just a single UPDATE on `thoughts`. Once retracted, the
//! thought is invisible to retrieval (`search_thoughts`, `recent_thoughts`,
//! `search_trigram`, `search_vector_knn` skip its rows) and to the tag
//! drainer's `find_untagged_or_stale_thoughts` walk. `get_thought` still
//! returns the row with the retraction state surfaced — direct lookup by
//! ID is the audit path.

use kengram_core::ThoughtId;
use sqlx::PgPool;

#[derive(Debug, Clone)]
pub struct RetractThoughtRequest {
    pub thought_id: ThoughtId,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetractThoughtResponse {
    pub retracted: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum RetractError {
    #[error("thought not found or already retracted: {0}")]
    NotFoundOrAlreadyRetracted(ThoughtId),

    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

/// Retract a thought.
///
/// Returns `Err(NotFoundOrAlreadyRetracted)` when the row doesn't exist
/// or has already been retracted — distinguishes "no-op" from "did the
/// work" so the MCP tool can surface a clean error to the client.
pub async fn retract_thought(
    pool: &PgPool,
    request: RetractThoughtRequest,
) -> Result<RetractThoughtResponse, RetractError> {
    let outcome =
        kengram_storage::retract_thought(pool, request.thought_id, request.reason.as_deref())
            .await?;

    if !outcome.retracted {
        return Err(RetractError::NotFoundOrAlreadyRetracted(request.thought_id));
    }

    Ok(RetractThoughtResponse { retracted: true })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use kengram_core::{Scope, Source};

    const TEST_EMBEDDER_MODEL_ID: &str = "bge-m3:1024";

    async fn cap(pool: &PgPool, content: &str) -> ThoughtId {
        capture(
            pool,
            TEST_EMBEDDER_MODEL_ID,
            None,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new("global").unwrap()),
                metadata: None,
                argus_source_event: None,
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_orchestrator_returns_response(pool: PgPool) {
        let id = cap(&pool, "wrong claim").await;
        let resp = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: id,
                reason: Some("test".into()),
            },
        )
        .await
        .unwrap();
        assert!(resp.retracted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_orchestrator_errors_on_already_retracted(pool: PgPool) {
        let id = cap(&pool, "wrong claim").await;
        retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: id,
                reason: None,
            },
        )
        .await
        .unwrap();
        let err = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: id,
                reason: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RetractError::NotFoundOrAlreadyRetracted(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_orchestrator_errors_on_unknown_id(pool: PgPool) {
        let err = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: ThoughtId::new(),
                reason: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RetractError::NotFoundOrAlreadyRetracted(_)));
    }
}

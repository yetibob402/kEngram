//! `retract_thought` — operator-driven path that marks a thought
//! untrusted. Atomically:
//!   1. Sets `thoughts.retracted_at = NOW()` (preserves the row; preserves
//!      the operator's reason).
//!   2. Supersedes every active fact derived from the thought.
//!
//! Once retracted, the thought is invisible to the reflector
//! (`find_unfacted_thoughts` / `find_facted_thoughts` skip it) and to
//! retrieval (`search_thoughts`, `recent_thoughts`, `search_facts` skip
//! its rows). `get_thought` still returns the row with the retraction
//! state surfaced — direct lookup by ID is the audit path.
//!
//! Motivated by the M2 dogfood: the previous workaround
//! ("`correct_fact` each derived fact one at a time") fails as soon as
//! the operator misses any fact, because the unretracted-thought-with-
//! one-active-fact stays in the reflector's set and gets re-extracted
//! on the next `engram reflect --rerun`. See
//! `docs/milestones/m2-progress.md` 2026-05-13 history entry.

use engram_core::ThoughtId;
use sqlx::PgPool;

#[derive(Debug, Clone)]
pub struct RetractThoughtRequest {
    pub thought_id: ThoughtId,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetractThoughtResponse {
    pub retracted: bool,
    pub facts_superseded: i64,
}

#[derive(Debug, thiserror::Error)]
pub enum RetractError {
    #[error("thought not found or already retracted: {0}")]
    NotFoundOrAlreadyRetracted(ThoughtId),

    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Retract a thought + auto-supersede its derived facts.
///
/// Returns `Err(NotFoundOrAlreadyRetracted)` when the row doesn't exist
/// or has already been retracted — distinguishes "no-op" from "did the
/// work" so the MCP tool can surface a clean error to the client.
pub async fn retract_thought(
    pool: &PgPool,
    request: RetractThoughtRequest,
) -> Result<RetractThoughtResponse, RetractError> {
    let outcome = engram_storage::retract_thought(
        pool,
        request.thought_id,
        request.reason.as_deref(),
    )
    .await?;

    if !outcome.retracted {
        return Err(RetractError::NotFoundOrAlreadyRetracted(request.thought_id));
    }

    Ok(RetractThoughtResponse {
        retracted: true,
        facts_superseded: outcome.facts_superseded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{capture, CaptureRequest};
    use engram_core::{Scope, Source};

    const TEST_MODEL_ID: &str = "bge-m3:1024";

    async fn cap(pool: &PgPool, content: &str) -> ThoughtId {
        capture(
            pool,
            TEST_MODEL_ID,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new("global").unwrap()),
                metadata: None,
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
            RetractThoughtRequest { thought_id: id, reason: Some("test".into()) },
        )
        .await
        .unwrap();
        assert!(resp.retracted);
        assert_eq!(resp.facts_superseded, 0); // no extraction has run yet
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_orchestrator_errors_on_already_retracted(pool: PgPool) {
        let id = cap(&pool, "wrong claim").await;
        retract_thought(&pool, RetractThoughtRequest { thought_id: id, reason: None })
            .await
            .unwrap();
        let err = retract_thought(
            &pool,
            RetractThoughtRequest { thought_id: id, reason: None },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RetractError::NotFoundOrAlreadyRetracted(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_orchestrator_errors_on_unknown_id(pool: PgPool) {
        let err = retract_thought(
            &pool,
            RetractThoughtRequest { thought_id: ThoughtId::new(), reason: None },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RetractError::NotFoundOrAlreadyRetracted(_)));
    }
}

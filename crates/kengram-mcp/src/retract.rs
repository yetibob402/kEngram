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
//!
//! Gate statuses (migration 0034): `retracted`, `not_found`, `already_retracted`,
//! `thought_chain_from_requires_unlink`. MCP must not collapse non-not_found
//! failures into a false not-found (board 547350).

use kengram_core::ThoughtId;
use sqlx::PgPool;

#[derive(Debug, Clone)]
pub struct RetractThoughtRequest {
    pub thought_id: ThoughtId,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetractThoughtResponse {
    pub retracted: bool,
    /// Gate status string for audit (`retracted` on success).
    pub status: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RetractError {
    #[error("thought not found: {0}")]
    NotFound(ThoughtId),

    #[error("thought already retracted: {0}")]
    AlreadyRetracted(ThoughtId),

    /// Thought is the FROM side of a live replaces/refines edge. Unlink or
    /// repoint that supersession edge before retracting the successor.
    #[error(
        "thought {0} is from-side of a live replaces/refines edge; unlink or repoint before retract (status=thought_chain_from_requires_unlink)"
    )]
    ChainFromRequiresUnlink(ThoughtId),

    #[error("retract refused for thought {thought_id}: status={status}")]
    Refused {
        thought_id: ThoughtId,
        status: String,
    },

    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

/// Retract a thought.
///
/// Maps gate `status` to distinct errors so callers never see a false
/// not-found when the row is live (board 547350).
pub async fn retract_thought(
    pool: &PgPool,
    request: RetractThoughtRequest,
) -> Result<RetractThoughtResponse, RetractError> {
    let outcome =
        kengram_storage::retract_thought(pool, request.thought_id, request.reason.as_deref())
            .await?;

    if outcome.retracted {
        return Ok(RetractThoughtResponse {
            retracted: true,
            status: outcome.status,
        });
    }

    match outcome.status.as_str() {
        "not_found" => Err(RetractError::NotFound(request.thought_id)),
        "already_retracted" => Err(RetractError::AlreadyRetracted(request.thought_id)),
        "thought_chain_from_requires_unlink" | "thought_chain_participant_requires_repoint" => {
            // Legacy status string still mapped if old gate body is live.
            Err(RetractError::ChainFromRequiresUnlink(request.thought_id))
        }
        other => Err(RetractError::Refused {
            thought_id: request.thought_id,
            status: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use crate::link::{LinkThoughtsRequest, RelationSourceEventRequest, link_thoughts};
    use kengram_core::{LinkTarget, RelationKind, Scope, Source};

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

    fn relation_event() -> RelationSourceEventRequest {
        let id = uuid::Uuid::new_v4().to_string();
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();
        let payload_hash: String = a
            .as_bytes()
            .iter()
            .chain(b.as_bytes().iter())
            .map(|byte| format!("{byte:02x}"))
            .collect();
        RelationSourceEventRequest {
            namespace: "tests/retract".to_string(),
            source_ref: id,
            payload_hash,
            metadata: serde_json::json!({}),
        }
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
        assert_eq!(resp.status, "retracted");
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
        assert!(matches!(err, RetractError::AlreadyRetracted(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_orchestrator_errors_on_unknown_id(pool: PgPool) {
        let missing = ThoughtId::new();
        let err = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: missing,
                reason: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RetractError::NotFound(_)));
    }

    /// Board 547350 class: thought is TO of replaces (superseded) — must
    /// still retract successfully; must not surface false not-found.
    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_succeeds_when_only_to_of_replaces(pool: PgPool) {
        let old = cap(&pool, "wrong memory to supersede").await;
        let corrected = cap(&pool, "correcting successor").await;
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: corrected,
                relation: RelationKind::Replaces,
                target: LinkTarget::Thought(old),
                note: Some("supersession before retract".into()),
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .expect("replaces to live TO");

        let resp = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: old,
                reason: Some("superseded wrong memory".into()),
            },
        )
        .await
        .expect("TO of replaces must retract; contract must not lie not-found");
        assert!(resp.retracted);
    }

    /// FROM of replaces still blocked, with honest status (not not-found).
    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_from_of_replaces_refuses_with_chain_status(pool: PgPool) {
        let old = cap(&pool, "predecessor").await;
        let corrected = cap(&pool, "successor from-side").await;
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: corrected,
                relation: RelationKind::Replaces,
                target: LinkTarget::Thought(old),
                note: None,
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .unwrap();

        let err = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: corrected,
                reason: Some("try retract successor".into()),
            },
        )
        .await
        .unwrap_err();
        match err {
            RetractError::ChainFromRequiresUnlink(id) => assert_eq!(id, corrected),
            other => panic!("expected ChainFromRequiresUnlink, got {other:?}"),
        }
        // Must not look like not-found
        let msg = err.to_string();
        assert!(!msg.contains("not found"), "{msg}");
        assert!(msg.contains("from-side") || msg.contains("unlink"), "{msg}");
    }

    /// Jones exact historical discriminator (board 547350 a2a cb357a4f):
    /// Unlinking inbound replaces (710a47d8→3424acda) is NOT enough when the
    /// thought is also FROM of its own replaces edge to a predecessor.
    /// Retract must refuse with chain-from status, never false not-found.
    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_after_unlink_inbound_still_blocked_when_from(pool: PgPool) {
        use crate::link::unlink_thoughts;

        let predecessor = cap(&pool, "predecessor of wrong memory").await;
        let wrong = cap(&pool, "wrong memory also supersedes predecessor").await;
        let corrector = cap(&pool, "correcting thought inbound supersession").await;

        // wrong is FROM of replaces → predecessor
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: wrong,
                relation: RelationKind::Replaces,
                target: LinkTarget::Thought(predecessor),
                note: Some("wrong also claims supersession".into()),
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .unwrap();

        // corrector replaces wrong (inbound supersession on wrong)
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: corrector,
                relation: RelationKind::Replaces,
                target: LinkTarget::Thought(wrong),
                note: Some("inbound supersession".into()),
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .unwrap();

        // Unlink only inbound pair (corrector→wrong) — jones step (2)
        let un = unlink_thoughts(
            &pool,
            corrector,
            RelationKind::Replaces,
            &LinkTarget::Thought(wrong),
            relation_event(),
            None,
        )
        .await
        .expect("unlink inbound");
        assert_eq!(un.status.as_str(), "deleted_now");

        // Retract wrong — still FROM of replaces to predecessor
        let err = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: wrong,
                reason: Some("jones discriminator after inbound unlink".into()),
            },
        )
        .await
        .unwrap_err();
        match err {
            RetractError::ChainFromRequiresUnlink(id) => assert_eq!(id, wrong),
            other => panic!("expected ChainFromRequiresUnlink not not-found: {other:?}"),
        }
        let msg = err.to_string();
        assert!(!msg.contains("not found"), "must not lie not-found: {msg}");

        // Unlink outbound FROM edge, then retract succeeds
        unlink_thoughts(
            &pool,
            wrong,
            RelationKind::Replaces,
            &LinkTarget::Thought(predecessor),
            relation_event(),
            None,
        )
        .await
        .unwrap();
        let resp = retract_thought(
            &pool,
            RetractThoughtRequest {
                thought_id: wrong,
                reason: Some("after both edges cleared".into()),
            },
        )
        .await
        .expect("retract after no FROM supersession edges");
        assert!(resp.retracted);
    }
}

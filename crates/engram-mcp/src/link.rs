//! `link_thoughts` and `unlink_thoughts` — agent-supplied thought-to-thought
//! edges in the closed M5 relation vocabulary. Edges live in `thought_links`
//! (migration 0007) and are queryable via [`crate::relate::get_related_thoughts`].
//!
//! Pre-validates the (from, to, note) triple before hitting storage so the
//! operator-facing error is actionable rather than a generic FK/CHECK
//! violation from Postgres.

use engram_core::{LinkId, LinkSource, RelationKind, ThoughtId};
use sqlx::PgPool;

/// Note column max length — bounded so a single bogus note can't OOM a
/// response. Same shape as `capture`'s `MAX_CONTENT_LEN`, but smaller since
/// notes are short rationales, not full prose.
pub const MAX_LINK_NOTE_LEN: usize = 1_000;

#[derive(Debug, Clone)]
pub struct LinkThoughtsRequest {
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub to_thought_id: ThoughtId,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LinkThoughtsResponse {
    pub link_id: LinkId,
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub to_thought_id: ThoughtId,
    /// `false` when the (from, relation, to) triple already existed — the
    /// returned `link_id` belongs to the pre-existing row and no new row
    /// was inserted. Mirrors `capture`'s `is_duplicate` semantics but with
    /// inverted polarity (positive: "was a fresh insert").
    pub is_new: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnlinkThoughtsResponse {
    /// `true` when an edge matched and was deleted; `false` when no edge
    /// matched (idempotent on already-deleted).
    pub existed: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum LinkError {
    #[error("from_thought_id and to_thought_id must differ — self-links are not supported")]
    SelfLink,

    #[error("from_thought_id {0} not found")]
    FromThoughtMissing(ThoughtId),

    #[error("to_thought_id {0} not found")]
    ToThoughtMissing(ThoughtId),

    #[error("note too long: {got} bytes, max {max}")]
    NoteTooLong { got: usize, max: usize },

    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Create a thought-to-thought edge. Idempotent on `(from, relation, to)`:
/// re-asserting the same triple returns the existing `LinkId` with
/// `is_new = false`.
///
/// Validation order: self-link check → note length → endpoint existence
/// (from then to). Each rejection produces a distinct `LinkError` variant
/// so the MCP handler can format an actionable message.
pub async fn link_thoughts(
    pool: &PgPool,
    request: LinkThoughtsRequest,
) -> Result<LinkThoughtsResponse, LinkError> {
    if request.from_thought_id == request.to_thought_id {
        return Err(LinkError::SelfLink);
    }
    if let Some(note) = &request.note
        && note.len() > MAX_LINK_NOTE_LEN
    {
        return Err(LinkError::NoteTooLong {
            got: note.len(),
            max: MAX_LINK_NOTE_LEN,
        });
    }

    // Pre-validate endpoints. Cheaper than a FK violation round-trip and
    // gives the caller a clear "which side was missing" diagnosis.
    if engram_storage::fetch_thought(pool, request.from_thought_id)
        .await?
        .is_none()
    {
        return Err(LinkError::FromThoughtMissing(request.from_thought_id));
    }
    if engram_storage::fetch_thought(pool, request.to_thought_id)
        .await?
        .is_none()
    {
        return Err(LinkError::ToThoughtMissing(request.to_thought_id));
    }

    let (link_id, is_new) = engram_storage::insert_link(
        pool,
        request.from_thought_id,
        request.relation,
        request.to_thought_id,
        LinkSource::Agent,
        request.note.as_deref(),
    )
    .await?;

    Ok(LinkThoughtsResponse {
        link_id,
        from_thought_id: request.from_thought_id,
        relation: request.relation,
        to_thought_id: request.to_thought_id,
        is_new,
    })
}

/// Delete a thought-to-thought edge by its `(from, relation, to)` triple.
/// Idempotent on already-deleted (returns `existed: false`). No endpoint
/// validation — deleting an edge to a missing thought is a no-op the
/// caller can verify via the boolean.
pub async fn unlink_thoughts(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    to: ThoughtId,
) -> Result<UnlinkThoughtsResponse, LinkError> {
    let existed = engram_storage::delete_link(pool, from, relation, to).await?;
    Ok(UnlinkThoughtsResponse { existed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use engram_core::{Scope, Source};

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
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_happy_path_returns_is_new(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let resp = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::Refines,
                to_thought_id: b,
                note: Some("first link".into()),
            },
        )
        .await
        .unwrap();
        assert!(resp.is_new);
        assert_eq!(resp.from_thought_id, a);
        assert_eq!(resp.to_thought_id, b);
        assert_eq!(resp.relation, RelationKind::Refines);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_duplicate_triple_is_idempotent(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let req = || LinkThoughtsRequest {
            from_thought_id: a,
            relation: RelationKind::Refines,
            to_thought_id: b,
            note: None,
        };
        let first = link_thoughts(&pool, req()).await.unwrap();
        let second = link_thoughts(&pool, req()).await.unwrap();
        assert!(first.is_new);
        assert!(!second.is_new);
        assert_eq!(first.link_id, second.link_id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_rejects_self_link(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let err = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::Refines,
                to_thought_id: a,
                note: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::SelfLink));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_rejects_missing_from(pool: PgPool) {
        let b = cap(&pool, "B").await;
        let phantom = ThoughtId::new();
        let err = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: phantom,
                relation: RelationKind::Refines,
                to_thought_id: b,
                note: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::FromThoughtMissing(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_rejects_missing_to(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let phantom = ThoughtId::new();
        let err = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::References,
                to_thought_id: phantom,
                note: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::ToThoughtMissing(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_rejects_overlong_note(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let too_long = "x".repeat(MAX_LINK_NOTE_LEN + 1);
        let err = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::Refines,
                to_thought_id: b,
                note: Some(too_long),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::NoteTooLong { .. }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn unlink_thoughts_returns_existed_then_false(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::Refines,
                to_thought_id: b,
                note: None,
            },
        )
        .await
        .unwrap();

        let first = unlink_thoughts(&pool, a, RelationKind::Refines, b)
            .await
            .unwrap();
        assert!(first.existed);

        let second = unlink_thoughts(&pool, a, RelationKind::Refines, b)
            .await
            .unwrap();
        assert!(!second.existed);
    }
}

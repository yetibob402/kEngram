//! `get_related_thoughts` — traverse the M5 thought-to-thought graph from
//! a given thought. Wraps `engram_storage::fetch_related_thoughts` and
//! groups the response into `outbound` (edges where the queried thought is
//! `from`) and `inbound` (edges where it's `to`) for ergonomics. Content is
//! preview-truncated so single-thought traversal responses stay bounded.

use engram_core::{LinkDirection, LinkId, LinkSource, RelationKind, Scope, ThoughtId};
use sqlx::PgPool;
use time::OffsetDateTime;

/// Content preview cap on each related thought's body. Single-thought
/// traversal rarely exceeds ~10 edges, so the full response stays
/// well-bounded even with this generous limit. Callers wanting the full
/// content can follow up with `get_thought` on a specific id.
pub const RELATED_CONTENT_PREVIEW_LEN: usize = 400;

#[derive(Debug, Clone)]
pub struct GetRelatedThoughtsRequest {
    pub thought_id: ThoughtId,
    /// Optional filter to a subset of the closed relation vocabulary.
    /// `None` returns edges of every type.
    pub relations: Option<Vec<RelationKind>>,
    /// Traversal direction. Defaults to `Both` when unspecified.
    pub direction: LinkDirection,
}

#[derive(Debug, Clone)]
pub struct GetRelatedThoughtsResponse {
    pub thought_id: ThoughtId,
    pub outbound: Vec<RelatedThoughtHit>,
    pub inbound: Vec<RelatedThoughtHit>,
}

#[derive(Debug, Clone)]
pub struct RelatedThoughtHit {
    pub link_id: LinkId,
    pub relation: RelationKind,
    pub thought_id: ThoughtId,
    pub scope: Scope,
    pub content_preview: String,
    pub content_truncated: bool,
    pub thought_created_at: OffsetDateTime,
    pub link_created_at: OffsetDateTime,
    pub link_source: LinkSource,
    pub note: Option<String>,
    pub retracted: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum RelateError {
    #[error("thought not found: {0}")]
    ThoughtNotFound(ThoughtId),

    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Fetch the related-thoughts response for a single thought, grouped by
/// edge direction. Errors `ThoughtNotFound` if the queried thought doesn't
/// exist (rather than silently returning empty groups). Retracted thoughts
/// on either side are surfaced with `retracted: true` and not filtered out.
pub async fn get_related_thoughts(
    pool: &PgPool,
    request: GetRelatedThoughtsRequest,
) -> Result<GetRelatedThoughtsResponse, RelateError> {
    if engram_storage::fetch_thought(pool, request.thought_id)
        .await?
        .is_none()
    {
        return Err(RelateError::ThoughtNotFound(request.thought_id));
    }

    let relations_slice = request.relations.as_deref();
    let rows = engram_storage::fetch_related_thoughts(
        pool,
        request.thought_id,
        relations_slice,
        request.direction,
    )
    .await?;

    let mut outbound = Vec::new();
    let mut inbound = Vec::new();
    for r in rows {
        let truncated = r.content.len() > RELATED_CONTENT_PREVIEW_LEN;
        let content_preview = if truncated {
            let mut end = RELATED_CONTENT_PREVIEW_LEN;
            // Don't slice mid-codepoint.
            while !r.content.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            r.content[..end].to_string()
        } else {
            r.content.clone()
        };

        let hit = RelatedThoughtHit {
            link_id: r.link_id,
            relation: r.relation,
            thought_id: r.thought_id,
            scope: r.scope,
            content_preview,
            content_truncated: truncated,
            thought_created_at: r.thought_created_at,
            link_created_at: r.link_created_at,
            link_source: r.link_source,
            note: r.note,
            retracted: r.retracted,
        };
        match r.direction {
            LinkDirection::Outbound => outbound.push(hit),
            LinkDirection::Inbound => inbound.push(hit),
            LinkDirection::Both => {
                // Storage layer never emits Both for individual rows — it's
                // only valid as a query-time directive — but match
                // exhaustively for safety.
                outbound.push(hit);
            }
        }
    }

    Ok(GetRelatedThoughtsResponse {
        thought_id: request.thought_id,
        outbound,
        inbound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use crate::link::{LinkThoughtsRequest, link_thoughts};
    use engram_core::Source;

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

    async fn link(pool: &PgPool, from: ThoughtId, rel: RelationKind, to: ThoughtId) {
        link_thoughts(
            pool,
            LinkThoughtsRequest {
                from_thought_id: from,
                relation: rel,
                to_thought_id: to,
                note: None,
            },
        )
        .await
        .unwrap();
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_groups_outbound_and_inbound(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let c = cap(&pool, "C").await;
        // A refines B; C refines A.
        link(&pool, a, RelationKind::Refines, b).await;
        link(&pool, c, RelationKind::Refines, a).await;

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                direction: LinkDirection::Both,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.thought_id, a);
        assert_eq!(resp.outbound.len(), 1);
        assert_eq!(resp.outbound[0].thought_id, b);
        assert_eq!(resp.inbound.len(), 1);
        assert_eq!(resp.inbound[0].thought_id, c);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_respects_direction_filter(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let c = cap(&pool, "C").await;
        link(&pool, a, RelationKind::Refines, b).await;
        link(&pool, c, RelationKind::Refines, a).await;

        let outbound_only = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(outbound_only.outbound.len(), 1);
        assert!(outbound_only.inbound.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_respects_relation_filter(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let c = cap(&pool, "C").await;
        link(&pool, a, RelationKind::Refines, b).await;
        link(&pool, a, RelationKind::Replaces, c).await;

        let only_replaces = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: Some(vec![RelationKind::Replaces]),
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(only_replaces.outbound.len(), 1);
        assert_eq!(only_replaces.outbound[0].relation, RelationKind::Replaces);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_truncates_long_content(pool: PgPool) {
        let big = "x".repeat(RELATED_CONTENT_PREVIEW_LEN + 200);
        let a = cap(&pool, "A").await;
        let b = cap(&pool, &big).await;
        link(&pool, a, RelationKind::References, b).await;

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.outbound.len(), 1);
        assert!(resp.outbound[0].content_truncated);
        assert!(resp.outbound[0].content_preview.len() <= RELATED_CONTENT_PREVIEW_LEN);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_surfaces_retracted_state(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        link(&pool, a, RelationKind::Refines, b).await;
        engram_storage::retract_thought(&pool, b, Some("test"))
            .await
            .unwrap();

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.outbound.len(), 1);
        assert!(resp.outbound[0].retracted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_errors_on_missing_thought(pool: PgPool) {
        let phantom = ThoughtId::new();
        let err = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: phantom,
                relations: None,
                direction: LinkDirection::Both,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RelateError::ThoughtNotFound(_)));
    }
}

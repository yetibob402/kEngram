//! `get_related_thoughts` — traverse the M5 link graph from a given
//! thought. Wraps `kengram_storage::fetch_related_thoughts` and groups the
//! response into `outbound` (edges where the queried thought is `from`)
//! and `inbound` (edges where it's `to`) for ergonomics. Content is
//! preview-truncated so single-thought traversal responses stay bounded.
//!
//! M5.2: outbound edges may target a thought, an entity, a person, or a
//! URL. Each hit carries a `to_kind` discriminator; thought-shaped fields
//! (scope, content_preview, retracted) are populated only when the target
//! is a thought. Inbound edges are always thought-shaped by schema.

use kengram_core::{LinkDirection, LinkId, LinkSource, LinkTarget, RelationKind, Scope, ThoughtId};
use sqlx::PgPool;
use time::OffsetDateTime;

/// Content preview cap on each thought-target hit's body. Non-thought hits
/// have no content_preview. Single-thought traversal rarely exceeds ~10
/// edges, so the full response stays well-bounded even with this generous
/// limit. Callers wanting the full content can follow up with
/// `get_thought` on a specific id.
pub const RELATED_CONTENT_PREVIEW_LEN: usize = 400;

#[derive(Debug, Clone)]
pub struct GetRelatedThoughtsRequest {
    pub thought_id: ThoughtId,
    /// Optional filter to a subset of the closed relation vocabulary.
    /// `None` returns edges of every type.
    pub relations: Option<Vec<RelationKind>>,
    /// Optional filter to a subset of target kinds (`thought`, `entity`,
    /// `person`, `url`). Applies to outbound edges only — inbound edges
    /// are always thought→thought by schema. `None` returns every kind.
    pub target_kinds: Option<Vec<String>>,
    /// Traversal direction. Defaults to `Both` when unspecified.
    pub direction: LinkDirection,
}

#[derive(Debug, Clone)]
pub struct GetRelatedThoughtsResponse {
    pub thought_id: ThoughtId,
    pub outbound: Vec<RelatedTargetHit>,
    pub inbound: Vec<RelatedTargetHit>,
}

/// One side of a link returned by `get_related_thoughts`. M5.2 generalized
/// from "always a thought" to a polymorphic target. The thought-shaped
/// fields (`scope`, `content_preview`, `content_truncated`,
/// `thought_created_at`, `retracted`) are populated only when
/// `target = LinkTarget::Thought(_)` — `None` otherwise.
#[derive(Debug, Clone)]
pub struct RelatedTargetHit {
    pub link_id: LinkId,
    pub relation: RelationKind,
    pub target: LinkTarget,
    pub scope: Option<Scope>,
    pub content_preview: Option<String>,
    pub content_truncated: Option<bool>,
    pub thought_created_at: Option<OffsetDateTime>,
    pub link_created_at: OffsetDateTime,
    pub link_source: LinkSource,
    pub note: Option<String>,
    pub retracted: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum RelateError {
    #[error("thought not found: {0}")]
    ThoughtNotFound(ThoughtId),

    #[error("unknown target kind: {0:?} (expected one of thought, entity, person, url)")]
    UnknownTargetKind(String),

    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

/// Fetch the related-targets response for a single thought, grouped by
/// edge direction. Errors `ThoughtNotFound` if the queried thought doesn't
/// exist (rather than silently returning empty groups). Retracted thoughts
/// on the far end are surfaced with `retracted: Some(true)` and not
/// filtered out. Soft-deleted edges are excluded.
pub async fn get_related_thoughts(
    pool: &PgPool,
    request: GetRelatedThoughtsRequest,
) -> Result<GetRelatedThoughtsResponse, RelateError> {
    if kengram_storage::fetch_thought(pool, request.thought_id)
        .await?
        .is_none()
    {
        return Err(RelateError::ThoughtNotFound(request.thought_id));
    }

    if let Some(kinds) = &request.target_kinds {
        for k in kinds {
            if !matches!(k.as_str(), "thought" | "entity" | "person" | "url") {
                return Err(RelateError::UnknownTargetKind(k.clone()));
            }
        }
    }

    let relations_slice = request.relations.as_deref();
    let kinds_owned: Option<Vec<&str>> = request
        .target_kinds
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let kinds_slice = kinds_owned.as_deref();
    let rows = kengram_storage::fetch_related_thoughts(
        pool,
        request.thought_id,
        relations_slice,
        kinds_slice,
        request.direction,
    )
    .await?;

    let mut outbound = Vec::new();
    let mut inbound = Vec::new();
    for r in rows {
        let (content_preview, content_truncated) = match &r.thought_content {
            Some(content) => {
                let truncated = content.len() > RELATED_CONTENT_PREVIEW_LEN;
                let preview = if truncated {
                    let mut end = RELATED_CONTENT_PREVIEW_LEN;
                    while !content.is_char_boundary(end) && end > 0 {
                        end -= 1;
                    }
                    content[..end].to_string()
                } else {
                    content.clone()
                };
                (Some(preview), Some(truncated))
            }
            None => (None, None),
        };

        let hit = RelatedTargetHit {
            link_id: r.link_id,
            relation: r.relation,
            target: r.target,
            scope: r.thought_scope,
            content_preview,
            content_truncated,
            thought_created_at: r.thought_created_at,
            link_created_at: r.link_created_at,
            link_source: r.link_source,
            note: r.note,
            retracted: r.thought_retracted,
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
    use kengram_core::Source;

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

    async fn link(pool: &PgPool, from: ThoughtId, rel: RelationKind, target: LinkTarget) {
        link_thoughts(
            pool,
            LinkThoughtsRequest {
                from_thought_id: from,
                relation: rel,
                target,
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
        link(&pool, a, RelationKind::Refines, LinkTarget::Thought(b)).await;
        link(&pool, c, RelationKind::Refines, LinkTarget::Thought(a)).await;

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: None,
                direction: LinkDirection::Both,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.thought_id, a);
        assert_eq!(resp.outbound.len(), 1);
        assert_eq!(resp.outbound[0].target, LinkTarget::Thought(b));
        assert_eq!(resp.inbound.len(), 1);
        assert_eq!(resp.inbound[0].target, LinkTarget::Thought(c));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_respects_direction_filter(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let c = cap(&pool, "C").await;
        link(&pool, a, RelationKind::Refines, LinkTarget::Thought(b)).await;
        link(&pool, c, RelationKind::Refines, LinkTarget::Thought(a)).await;

        let outbound_only = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: None,
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
        link(&pool, a, RelationKind::Refines, LinkTarget::Thought(b)).await;
        link(&pool, a, RelationKind::Replaces, LinkTarget::Thought(c)).await;

        let only_replaces = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: Some(vec![RelationKind::Replaces]),
                target_kinds: None,
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
        link(&pool, a, RelationKind::References, LinkTarget::Thought(b)).await;

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: None,
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.outbound.len(), 1);
        assert_eq!(resp.outbound[0].content_truncated, Some(true));
        assert!(
            resp.outbound[0].content_preview.as_ref().unwrap().len() <= RELATED_CONTENT_PREVIEW_LEN
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_surfaces_retracted_state(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        link(&pool, a, RelationKind::Refines, LinkTarget::Thought(b)).await;
        kengram_storage::retract_thought(&pool, b, Some("test"))
            .await
            .unwrap();

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: None,
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.outbound.len(), 1);
        assert_eq!(resp.outbound[0].retracted, Some(true));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_returns_heterogeneous_outbound(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        link(&pool, a, RelationKind::Refines, LinkTarget::Thought(b)).await;
        link(
            &pool,
            a,
            RelationKind::BelongsTo,
            LinkTarget::Entity("Probe 2".into()),
        )
        .await;
        link(
            &pool,
            a,
            RelationKind::References,
            LinkTarget::Url("https://anthropic.com".into()),
        )
        .await;

        let resp = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: None,
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.outbound.len(), 3);
        let kinds: Vec<&'static str> = resp.outbound.iter().map(|h| h.target.kind_str()).collect();
        assert!(kinds.contains(&"thought"));
        assert!(kinds.contains(&"entity"));
        assert!(kinds.contains(&"url"));
        // Thought target retains content_preview; non-thought targets are None.
        for hit in &resp.outbound {
            if matches!(hit.target, LinkTarget::Thought(_)) {
                assert!(hit.content_preview.is_some());
                assert!(hit.scope.is_some());
            } else {
                assert!(hit.content_preview.is_none());
                assert!(hit.scope.is_none());
                assert!(hit.retracted.is_none());
            }
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_filters_by_target_kinds(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        link(&pool, a, RelationKind::Refines, LinkTarget::Thought(b)).await;
        link(
            &pool,
            a,
            RelationKind::References,
            LinkTarget::Url("https://anthropic.com".into()),
        )
        .await;

        let only_urls = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: Some(vec!["url".into()]),
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap();
        assert_eq!(only_urls.outbound.len(), 1);
        assert_eq!(only_urls.outbound[0].target.kind_str(), "url");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_errors_on_missing_thought(pool: PgPool) {
        let phantom = ThoughtId::new();
        let err = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: phantom,
                relations: None,
                target_kinds: None,
                direction: LinkDirection::Both,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RelateError::ThoughtNotFound(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_related_thoughts_rejects_unknown_target_kind(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let err = get_related_thoughts(
            &pool,
            GetRelatedThoughtsRequest {
                thought_id: a,
                relations: None,
                target_kinds: Some(vec!["movie".into()]),
                direction: LinkDirection::Outbound,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RelateError::UnknownTargetKind(_)));
    }
}

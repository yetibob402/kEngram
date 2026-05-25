//! `link_thoughts` and `unlink_thoughts` — agent-supplied links from a
//! thought to a polymorphic target (thought, entity, person, or URL).
//! Edges live in `thought_links` (migrations 0007, 0009, 0010) and are
//! queryable via [`crate::relate::get_related_thoughts`].
//!
//! Pre-validates the request before hitting storage so the operator-facing
//! error is actionable rather than a generic FK/CHECK violation from
//! Postgres.

use kengram_core::{LinkId, LinkSource, LinkTarget, RelationKind, ThoughtId};
use kengram_storage::LinkStatus;
use sqlx::PgPool;

/// Note column max length — bounded so a single bogus note can't OOM a
/// response. Same shape as `capture`'s `MAX_CONTENT_LEN`, but smaller since
/// notes are short rationales, not full prose.
pub const MAX_LINK_NOTE_LEN: usize = 1_000;

/// Free-text entity/person target max length. Generous; kengram has no
/// first-class entity/person tables so these strings are user-supplied
/// labels.
pub const MAX_TARGET_NAME_LEN: usize = 200;

/// URL target max length. Generous; covers any reasonable web URL.
pub const MAX_TARGET_URL_LEN: usize = 2_048;

#[derive(Debug, Clone)]
pub struct LinkThoughtsRequest {
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub target: LinkTarget,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LinkThoughtsResponse {
    pub link_id: LinkId,
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub target: LinkTarget,
    /// `false` when the (from, relation, target) edge already existed live —
    /// the returned `link_id` belongs to the pre-existing row and no new
    /// row was inserted. If the edge was previously soft-deleted, a fresh
    /// row is inserted and `is_new = true`.
    pub is_new: bool,
}

/// Three-way status returned by `unlink_thoughts`.
///
/// - `DeletedNow`: the edge was live and was just soft-deleted by this call.
/// - `AlreadyDeleted`: the edge previously existed but had already been
///   soft-deleted (no DB write occurred this call).
/// - `NeverExisted`: no edge with the given triple ever existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnlinkStatus {
    DeletedNow,
    AlreadyDeleted,
    NeverExisted,
}

impl UnlinkStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DeletedNow => "deleted_now",
            Self::AlreadyDeleted => "already_deleted",
            Self::NeverExisted => "never_existed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnlinkThoughtsResponse {
    pub status: UnlinkStatus,
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

    #[error("entity/person target must not be empty")]
    EmptyTargetName,

    #[error("entity/person target too long: {got} bytes, max {max}")]
    TargetNameTooLong { got: usize, max: usize },

    #[error("URL target too long: {got} bytes, max {max}")]
    TargetUrlTooLong { got: usize, max: usize },

    #[error("URL target must start with http:// or https://")]
    InvalidUrl,

    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

/// Create a link from a thought to a polymorphic target. Idempotent on the
/// `(from, relation, to_kind, to_value)` quadruple: re-asserting the same
/// live edge returns the existing `LinkId` with `is_new = false`. If the
/// edge was previously soft-deleted, a fresh live row is inserted and
/// `is_new = true`.
///
/// Validation order: target shape (non-empty / well-formed URL / length) →
/// self-link check (thought targets only) → note length → endpoint
/// existence (thought targets only; from then to). Each rejection produces
/// a distinct `LinkError` variant so the MCP handler can format an
/// actionable message.
pub async fn link_thoughts(
    pool: &PgPool,
    request: LinkThoughtsRequest,
) -> Result<LinkThoughtsResponse, LinkError> {
    validate_target(&request.target)?;

    if let LinkTarget::Thought(to) = &request.target
        && request.from_thought_id == *to
    {
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

    // Endpoint existence pre-check. Cheaper than a FK violation round-trip
    // and gives the caller a clear "which side was missing" diagnosis.
    if kengram_storage::fetch_thought(pool, request.from_thought_id)
        .await?
        .is_none()
    {
        return Err(LinkError::FromThoughtMissing(request.from_thought_id));
    }
    if let LinkTarget::Thought(to) = &request.target
        && kengram_storage::fetch_thought(pool, *to).await?.is_none()
    {
        return Err(LinkError::ToThoughtMissing(*to));
    }

    let (link_id, is_new) = kengram_storage::insert_link(
        pool,
        request.from_thought_id,
        request.relation,
        &request.target,
        LinkSource::Agent,
        request.note.as_deref(),
    )
    .await?;

    Ok(LinkThoughtsResponse {
        link_id,
        from_thought_id: request.from_thought_id,
        relation: request.relation,
        target: request.target,
        is_new,
    })
}

/// Soft-delete a link by its `(from, relation, target)` triple. Returns a
/// three-way status: `DeletedNow` (a live edge was just soft-deleted),
/// `AlreadyDeleted` (the edge existed but was already soft-deleted), or
/// `NeverExisted`.
pub async fn unlink_thoughts(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    target: &LinkTarget,
) -> Result<UnlinkThoughtsResponse, LinkError> {
    match kengram_storage::lookup_link_status(pool, from, relation, target).await? {
        LinkStatus::Live => {
            // The lookup → delete pair is racy in theory but kengram is
            // single-user single-active-session. If the row was deleted
            // between lookup and update we'd return None from delete_link;
            // map that to AlreadyDeleted (the operator-facing meaning is
            // accurate either way).
            match kengram_storage::delete_link(pool, from, relation, target).await? {
                Some(_) => Ok(UnlinkThoughtsResponse {
                    status: UnlinkStatus::DeletedNow,
                }),
                None => Ok(UnlinkThoughtsResponse {
                    status: UnlinkStatus::AlreadyDeleted,
                }),
            }
        }
        LinkStatus::SoftDeleted => Ok(UnlinkThoughtsResponse {
            status: UnlinkStatus::AlreadyDeleted,
        }),
        LinkStatus::NeverExisted => Ok(UnlinkThoughtsResponse {
            status: UnlinkStatus::NeverExisted,
        }),
    }
}

pub(crate) fn validate_target(target: &LinkTarget) -> Result<(), LinkError> {
    match target {
        LinkTarget::Thought(_) => Ok(()),
        LinkTarget::Entity(name) | LinkTarget::Person(name) => {
            if name.trim().is_empty() {
                Err(LinkError::EmptyTargetName)
            } else if name.len() > MAX_TARGET_NAME_LEN {
                Err(LinkError::TargetNameTooLong {
                    got: name.len(),
                    max: MAX_TARGET_NAME_LEN,
                })
            } else {
                Ok(())
            }
        }
        LinkTarget::Url(url) => {
            if url.len() > MAX_TARGET_URL_LEN {
                Err(LinkError::TargetUrlTooLong {
                    got: url.len(),
                    max: MAX_TARGET_URL_LEN,
                })
            } else if !url.starts_with("http://") && !url.starts_with("https://") {
                Err(LinkError::InvalidUrl)
            } else {
                Ok(())
            }
        }
    }
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
                target: LinkTarget::Thought(b),
                note: Some("first link".into()),
            },
        )
        .await
        .unwrap();
        assert!(resp.is_new);
        assert_eq!(resp.from_thought_id, a);
        assert_eq!(resp.target, LinkTarget::Thought(b));
        assert_eq!(resp.relation, RelationKind::Refines);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_duplicate_triple_is_idempotent(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let req = || LinkThoughtsRequest {
            from_thought_id: a,
            relation: RelationKind::Refines,
            target: LinkTarget::Thought(b),
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
                target: LinkTarget::Thought(a),
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
                target: LinkTarget::Thought(b),
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
                target: LinkTarget::Thought(phantom),
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
                target: LinkTarget::Thought(b),
                note: Some(too_long),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::NoteTooLong { .. }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_writes_entity_target(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let resp = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::BelongsTo,
                target: LinkTarget::Entity("Probe 2 experiment".into()),
                note: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.is_new);
        assert_eq!(resp.target, LinkTarget::Entity("Probe 2 experiment".into()));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_writes_url_target(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let resp = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::References,
                target: LinkTarget::Url("https://anthropic.com".into()),
                note: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.is_new);
        assert_eq!(resp.target, LinkTarget::Url("https://anthropic.com".into()));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_rejects_non_http_url(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let err = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::References,
                target: LinkTarget::Url("ftp://example.com".into()),
                note: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::InvalidUrl));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_thoughts_rejects_empty_entity_name(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let err = link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::BelongsTo,
                target: LinkTarget::Entity("   ".into()),
                note: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, LinkError::EmptyTargetName));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn unlink_thoughts_three_way_status(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let target = LinkTarget::Thought(b);

        // Never existed.
        let resp = unlink_thoughts(&pool, a, RelationKind::Refines, &target)
            .await
            .unwrap();
        assert_eq!(resp.status, UnlinkStatus::NeverExisted);

        // Live → DeletedNow.
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: a,
                relation: RelationKind::Refines,
                target: target.clone(),
                note: None,
            },
        )
        .await
        .unwrap();
        let resp = unlink_thoughts(&pool, a, RelationKind::Refines, &target)
            .await
            .unwrap();
        assert_eq!(resp.status, UnlinkStatus::DeletedNow);

        // Already soft-deleted → AlreadyDeleted.
        let resp = unlink_thoughts(&pool, a, RelationKind::Refines, &target)
            .await
            .unwrap();
        assert_eq!(resp.status, UnlinkStatus::AlreadyDeleted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn link_after_unlink_creates_fresh_row(pool: PgPool) {
        let a = cap(&pool, "A").await;
        let b = cap(&pool, "B").await;
        let target = LinkTarget::Thought(b);
        let req = || LinkThoughtsRequest {
            from_thought_id: a,
            relation: RelationKind::Refines,
            target: target.clone(),
            note: None,
        };
        let first = link_thoughts(&pool, req()).await.unwrap();
        unlink_thoughts(&pool, a, RelationKind::Refines, &target)
            .await
            .unwrap();
        let second = link_thoughts(&pool, req()).await.unwrap();
        assert!(second.is_new);
        assert_ne!(first.link_id, second.link_id);
    }
}

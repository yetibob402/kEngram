//! `link_thoughts` and `unlink_thoughts` — agent-supplied links from a
//! thought to a polymorphic target (thought, entity, person, or URL).
//! Edges live in `thought_links` (migrations 0007, 0009, 0010) and are
//! queryable via [`crate::relate::get_related_thoughts`].
//!
//! Pre-validates the request before hitting storage so the operator-facing
//! error is actionable rather than a generic FK/CHECK violation from
//! Postgres.

use kengram_core::{LinkId, LinkTarget, RelationKind, ThoughtId};
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
    pub source_event: RelationSourceEventRequest,
    pub claimed_producer_class: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RelationSourceEventRequest {
    pub namespace: String,
    pub source_ref: String,
    pub payload_hash: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct LinkThoughtsResponse {
    pub link_id: LinkId,
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub target: LinkTarget,
    /// The durable result recorded for this request identity. `false` when
    /// the edge already existed live; `true` when this request created it.
    /// Exact retries return the originally recorded value verbatim.
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

    #[error("invalid relation source event: {0}")]
    InvalidSourceEvent(&'static str),

    #[error("relation source-event replay conflicts with its prior payload or intent")]
    SourceEventConflict,
}

/// Create a link from a thought to a polymorphic target. Idempotent on the
/// `(from, relation, to_kind, to_value)` quadruple: re-asserting the same
/// live edge under a fresh request identity returns the existing `LinkId`
/// with `is_new = false`. If the edge was previously soft-deleted, a fresh
/// live row is inserted and `is_new = true`. Exact request retries return
/// their recorded `LinkId` and `is_new` outcome verbatim.
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
    validate_source_event(&request.source_event)?;
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

    let operations = serde_json::json!([{
        "action": "create",
        "from_thought_id": request.from_thought_id.to_string(),
        "relation": request.relation.as_str(),
        "to_kind": request.target.kind_str(),
        "to_value": request.target.value_str(),
        "source": "agent",
        "note": request.note,
    }]);
    let result = kengram_storage::corpus_hygiene::mutate_thought_relations_serialized(
        pool,
        kengram_storage::corpus_hygiene::RelationMutationRequest {
            operations: &operations,
            source_event_namespace: &request.source_event.namespace,
            source_event_ref: &request.source_event.source_ref,
            source_event_payload_hash: &request.source_event.payload_hash,
            request_metadata: &request.source_event.metadata,
            claimed_producer_class: request.claimed_producer_class.as_deref(),
        },
    )
    .await?;
    if result.get("status").and_then(|v| v.as_str()) == Some("source_event_conflict") {
        return Err(LinkError::SourceEventConflict);
    }
    let link_id = result
        .get("link_ids")
        .and_then(|ids| ids.as_array())
        .and_then(|ids| ids.first())
        .and_then(|id| id.as_str())
        .and_then(|id| uuid::Uuid::parse_str(id).ok())
        .map(LinkId::from)
        .ok_or_else(|| {
            LinkError::Storage(kengram_storage::StorageError::Database(
                sqlx::Error::Protocol(format!("serialized relation returned no link id: {result}")),
            ))
        })?;
    let is_new = match result
        .pointer("/operation_results/0/outcome")
        .and_then(|outcome| outcome.as_str())
    {
        Some("created") => true,
        Some("already_live") => false,
        _ => {
            return Err(LinkError::Storage(kengram_storage::StorageError::Database(
                sqlx::Error::Protocol(format!(
                    "serialized create returned no durable outcome: {result}"
                )),
            )));
        }
    };

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
    source_event: RelationSourceEventRequest,
    claimed_producer_class: Option<&str>,
) -> Result<UnlinkThoughtsResponse, LinkError> {
    validate_source_event(&source_event)?;
    let operations = serde_json::json!([{
        "action": "delete",
        "from_thought_id": from.to_string(),
        "relation": relation.as_str(),
        "to_kind": target.kind_str(),
        "to_value": target.value_str(),
        "source": "agent",
    }]);
    let result = kengram_storage::corpus_hygiene::mutate_thought_relations_serialized(
        pool,
        kengram_storage::corpus_hygiene::RelationMutationRequest {
            operations: &operations,
            source_event_namespace: &source_event.namespace,
            source_event_ref: &source_event.source_ref,
            source_event_payload_hash: &source_event.payload_hash,
            request_metadata: &source_event.metadata,
            claimed_producer_class,
        },
    )
    .await?;
    if result.get("status").and_then(|v| v.as_str()) == Some("source_event_conflict") {
        return Err(LinkError::SourceEventConflict);
    }
    let outcome = result
        .get("operation_results")
        .and_then(|results| results.as_array())
        .and_then(|results| results.first())
        .and_then(|result| result.get("outcome"))
        .and_then(|outcome| outcome.as_str());
    let status = match outcome {
        Some("deleted_now") => UnlinkStatus::DeletedNow,
        Some("already_deleted") => UnlinkStatus::AlreadyDeleted,
        Some("never_existed") => UnlinkStatus::NeverExisted,
        _ => {
            return Err(LinkError::Storage(kengram_storage::StorageError::Database(
                sqlx::Error::Protocol(format!(
                    "serialized unlink returned no durable outcome: {result}"
                )),
            )));
        }
    };
    Ok(UnlinkThoughtsResponse { status })
}

fn validate_source_event(event: &RelationSourceEventRequest) -> Result<(), LinkError> {
    if event.namespace.trim().is_empty() {
        return Err(LinkError::InvalidSourceEvent("namespace is required"));
    }
    if event.source_ref.trim().is_empty() {
        return Err(LinkError::InvalidSourceEvent("source_ref is required"));
    }
    if event.payload_hash.trim().is_empty() {
        return Err(LinkError::InvalidSourceEvent("payload_hash is required"));
    }
    Ok(())
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
    use std::time::Duration;

    const TEST_EMBEDDER_MODEL_ID: &str = "bge-m3:1024";

    fn relation_event() -> RelationSourceEventRequest {
        let id = uuid::Uuid::new_v4().to_string();
        RelationSourceEventRequest {
            namespace: "tests/relation".to_string(),
            source_ref: id.clone(),
            payload_hash: id,
            metadata: serde_json::json!({}),
        }
    }

    const RELATION_CLAIM_TEST_LOCK: i64 = 7_301_002;

    async fn install_relation_claim_barrier(pool: &PgPool) {
        sqlx::query(
            r#"
            CREATE FUNCTION public.test_block_relation_claim()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $test$
            BEGIN
                PERFORM pg_catalog.pg_advisory_xact_lock(7301002);
                RETURN NEW;
            END
            $test$
            "#,
        )
        .execute(pool)
        .await
        .unwrap();
        sqlx::query(
            r#"
            CREATE TRIGGER test_block_relation_claim
            BEFORE INSERT ON public.thought_relation_request_events
            FOR EACH ROW EXECUTE FUNCTION public.test_block_relation_claim()
            "#,
        )
        .execute(pool)
        .await
        .unwrap();
    }

    async fn wait_for_active_relation_clients(pool: &PgPool, prefix: &str) {
        let pattern = format!("{prefix}%");
        for _ in 0..200 {
            let active: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM pg_catalog.pg_stat_activity WHERE application_name LIKE $1 AND state = 'active'",
            )
            .bind(&pattern)
            .fetch_one(pool)
            .await
            .unwrap();
            if active >= 2 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for two concurrent {prefix} clients");
    }

    async fn wait_for_relation_claim_waiters(pool: &PgPool) {
        for _ in 0..500 {
            let waiting: i64 = sqlx::query_scalar(
                r#"
                SELECT COUNT(*)::bigint
                FROM pg_catalog.pg_locks AS locks
                JOIN pg_catalog.pg_stat_activity AS activity ON activity.pid = locks.pid
                WHERE locks.locktype = 'advisory'
                  AND NOT locks.granted
                  AND locks.classid::bigint = 0
                  AND locks.objid::bigint = $1
                  AND activity.datname = current_database()
                "#,
            )
            .bind(RELATION_CLAIM_TEST_LOCK)
            .fetch_one(pool)
            .await
            .unwrap();
            if waiting >= 2 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for two public create calls at the relation claim barrier");
    }

    fn spawn_relation_mutation(
        pool: PgPool,
        application_name: String,
        operations: serde_json::Value,
        source_ref: String,
    ) -> tokio::task::JoinHandle<serde_json::Value> {
        tokio::spawn(async move {
            let mut connection = pool.acquire().await.unwrap();
            sqlx::query("SET SESSION AUTHORIZATION kengram_rt_native_mcp")
                .execute(&mut *connection)
                .await
                .unwrap();
            sqlx::query("SELECT pg_catalog.set_config('application_name', $1, false)")
                .bind(application_name)
                .execute(&mut *connection)
                .await
                .unwrap();
            let result = sqlx::query_scalar::<_, serde_json::Value>(
                r#"
                SELECT public.mutate_thought_relations_serialized(
                    $1, 'tests/concurrent-relation-claim', $2,
                    'same-payload', '{}'::jsonb, NULL::text
                )
                "#,
            )
            .bind(operations)
            .bind(source_ref)
            .fetch_one(&mut *connection)
            .await
            .unwrap();
            sqlx::query("RESET SESSION AUTHORIZATION")
                .execute(&mut *connection)
                .await
                .unwrap();
            result
        })
    }

    async fn run_relation_claim_race(
        pool: &PgPool,
        application_prefix: &str,
        operations: serde_json::Value,
        source_ref: &str,
    ) -> (serde_json::Value, serde_json::Value) {
        let mut blocker = pool.acquire().await.unwrap();
        sqlx::query("SELECT pg_catalog.pg_advisory_lock($1)")
            .bind(RELATION_CLAIM_TEST_LOCK)
            .execute(&mut *blocker)
            .await
            .unwrap();
        let first = spawn_relation_mutation(
            pool.clone(),
            format!("{application_prefix}first"),
            operations.clone(),
            source_ref.to_string(),
        );
        let second = spawn_relation_mutation(
            pool.clone(),
            format!("{application_prefix}second"),
            operations,
            source_ref.to_string(),
        );
        wait_for_active_relation_clients(pool, application_prefix).await;
        sqlx::query("SELECT pg_catalog.pg_advisory_unlock($1)")
            .bind(RELATION_CLAIM_TEST_LOCK)
            .execute(&mut *blocker)
            .await
            .unwrap();
        (first.await.unwrap(), second.await.unwrap())
    }

    fn assert_completed_replay_pair(first: &serde_json::Value, second: &serde_json::Value) {
        assert_eq!(
            first.get("status").and_then(|v| v.as_str()),
            Some("completed")
        );
        assert_eq!(
            second.get("status").and_then(|v| v.as_str()),
            Some("completed")
        );
        assert_eq!(first.get("link_ids"), second.get("link_ids"));
        assert_eq!(
            first.get("operation_results"),
            second.get("operation_results")
        );
        assert_ne!(first.get("replayed"), second.get("replayed"));
    }

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
                source_event: relation_event(),
                claimed_producer_class: None,
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
            source_event: relation_event(),
            claimed_producer_class: None,
        };
        let first = link_thoughts(&pool, req()).await.unwrap();
        let second = link_thoughts(&pool, req()).await.unwrap();
        assert!(first.is_new);
        assert!(!second.is_new);
        assert_eq!(first.link_id, second.link_id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn exact_create_replays_recorded_outcome_after_independent_relink(pool: PgPool) {
        let from = cap(&pool, "stable create source").await;
        let target = LinkTarget::Thought(cap(&pool, "stable create target").await);
        let original_event = relation_event();
        let request = |source_event| LinkThoughtsRequest {
            from_thought_id: from,
            relation: RelationKind::References,
            target: target.clone(),
            note: None,
            source_event,
            claimed_producer_class: None,
        };

        let first = link_thoughts(&pool, request(original_event.clone()))
            .await
            .unwrap();
        assert!(first.is_new);
        unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &target,
            relation_event(),
            None,
        )
        .await
        .unwrap();
        let replacement = link_thoughts(&pool, request(relation_event()))
            .await
            .unwrap();
        assert!(replacement.is_new);
        assert_ne!(replacement.link_id, first.link_id);

        let replay = link_thoughts(&pool, request(original_event)).await.unwrap();
        assert_eq!(replay.link_id, first.link_id);
        assert_eq!(replay.is_new, first.is_new);
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
                source_event: relation_event(),
                claimed_producer_class: None,
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
        let resp = unlink_thoughts(
            &pool,
            a,
            RelationKind::Refines,
            &target,
            relation_event(),
            None,
        )
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
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .unwrap();
        let resp = unlink_thoughts(
            &pool,
            a,
            RelationKind::Refines,
            &target,
            relation_event(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(resp.status, UnlinkStatus::DeletedNow);

        // Already soft-deleted → AlreadyDeleted.
        let resp = unlink_thoughts(
            &pool,
            a,
            RelationKind::Refines,
            &target,
            relation_event(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(resp.status, UnlinkStatus::AlreadyDeleted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn unlink_conflict_is_checked_for_already_deleted_and_never_existed(pool: PgPool) {
        let from = cap(&pool, "unlink conflict source").await;
        let first_target = LinkTarget::Thought(cap(&pool, "first unlink target").await);
        let deleted_target = LinkTarget::Thought(cap(&pool, "already deleted target").await);
        let never_target = LinkTarget::Thought(cap(&pool, "never existed target").await);

        for target in [&first_target, &deleted_target] {
            link_thoughts(
                &pool,
                LinkThoughtsRequest {
                    from_thought_id: from,
                    relation: RelationKind::References,
                    target: target.clone(),
                    note: None,
                    source_event: relation_event(),
                    claimed_producer_class: None,
                },
            )
            .await
            .unwrap();
        }

        let accepted_event = relation_event();
        let first = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &first_target,
            accepted_event.clone(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(first.status, UnlinkStatus::DeletedNow);
        unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &deleted_target,
            relation_event(),
            None,
        )
        .await
        .unwrap();

        let already_deleted_conflict = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &deleted_target,
            accepted_event.clone(),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            already_deleted_conflict,
            LinkError::SourceEventConflict
        ));

        let never_existed_conflict = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &never_target,
            accepted_event,
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            never_existed_conflict,
            LinkError::SourceEventConflict
        ));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn concurrent_identical_relation_and_unlink_first_deliveries_replay_without_error(
        pool: PgPool,
    ) {
        let from = cap(&pool, "concurrent relation source").await;
        let target = cap(&pool, "concurrent relation target").await;
        install_relation_claim_barrier(&pool).await;

        let create = serde_json::json!([{
            "action": "create",
            "from_thought_id": from.to_string(),
            "relation": "references",
            "to_kind": "thought",
            "to_value": target.to_string(),
            "source": "agent",
        }]);
        let (first, second) =
            run_relation_claim_race(&pool, "relation_claim_race_create_", create, "same-create")
                .await;
        assert_completed_replay_pair(&first, &second);

        let delete = serde_json::json!([{
            "action": "delete",
            "from_thought_id": from.to_string(),
            "relation": "references",
            "to_kind": "thought",
            "to_value": target.to_string(),
            "source": "agent",
        }]);
        let (first, second) =
            run_relation_claim_race(&pool, "relation_claim_race_delete_", delete, "same-delete")
                .await;
        assert_completed_replay_pair(&first, &second);
        assert_eq!(
            first
                .pointer("/operation_results/0/outcome")
                .and_then(|v| v.as_str()),
            Some("deleted_now")
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn concurrent_distinct_creates_report_one_recorded_creation(pool: PgPool) {
        let from = cap(&pool, "concurrent public create source").await;
        let target = LinkTarget::Thought(cap(&pool, "concurrent public create target").await);
        let first_event = relation_event();
        let second_event = relation_event();
        install_relation_claim_barrier(&pool).await;

        let mut blocker = pool.acquire().await.unwrap();
        sqlx::query("SELECT pg_catalog.pg_advisory_lock($1)")
            .bind(RELATION_CLAIM_TEST_LOCK)
            .execute(&mut *blocker)
            .await
            .unwrap();

        let first_pool = pool.clone();
        let first_target = target.clone();
        let first_source_ref = first_event.source_ref.clone();
        let first = tokio::spawn(async move {
            link_thoughts(
                &first_pool,
                LinkThoughtsRequest {
                    from_thought_id: from,
                    relation: RelationKind::References,
                    target: first_target,
                    note: None,
                    source_event: first_event,
                    claimed_producer_class: None,
                },
            )
            .await
            .unwrap()
        });
        let second_pool = pool.clone();
        let second_target = target.clone();
        let second_source_ref = second_event.source_ref.clone();
        let second = tokio::spawn(async move {
            link_thoughts(
                &second_pool,
                LinkThoughtsRequest {
                    from_thought_id: from,
                    relation: RelationKind::References,
                    target: second_target,
                    note: None,
                    source_event: second_event,
                    claimed_producer_class: None,
                },
            )
            .await
            .unwrap()
        });

        wait_for_relation_claim_waiters(&pool).await;
        sqlx::query("SELECT pg_catalog.pg_advisory_unlock($1)")
            .bind(RELATION_CLAIM_TEST_LOCK)
            .execute(&mut *blocker)
            .await
            .unwrap();
        let first = first.await.unwrap();
        let second = second.await.unwrap();

        assert_eq!(first.link_id, second.link_id);
        assert_eq!(usize::from(first.is_new) + usize::from(second.is_new), 1);
        let (created, already_live): (i64, i64) = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE operation_results->0->>'outcome' = 'created')::bigint,
                COUNT(*) FILTER (WHERE operation_results->0->>'outcome' = 'already_live')::bigint
            FROM thought_relation_request_events
            WHERE source_event_namespace = 'tests/relation'
              AND source_event_ref IN ($1, $2)
            "#,
        )
        .bind(first_source_ref)
        .bind(second_source_ref)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!((created, already_live), (1, 1));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn exact_noop_unlink_replays_recorded_outcome_after_independent_relink(pool: PgPool) {
        let from = cap(&pool, "stable no-op unlink source").await;
        let target = LinkTarget::Thought(cap(&pool, "stable no-op unlink target").await);

        let never_existed_event = relation_event();
        let first = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &target,
            never_existed_event.clone(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(first.status, UnlinkStatus::NeverExisted);
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: from,
                relation: RelationKind::References,
                target: target.clone(),
                note: None,
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .unwrap();
        let replay = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &target,
            never_existed_event,
            None,
        )
        .await
        .unwrap();
        assert_eq!(replay.status, UnlinkStatus::NeverExisted);

        unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &target,
            relation_event(),
            None,
        )
        .await
        .unwrap();
        let already_deleted_event = relation_event();
        let first = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &target,
            already_deleted_event.clone(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(first.status, UnlinkStatus::AlreadyDeleted);
        link_thoughts(
            &pool,
            LinkThoughtsRequest {
                from_thought_id: from,
                relation: RelationKind::References,
                target: target.clone(),
                note: None,
                source_event: relation_event(),
                claimed_producer_class: None,
            },
        )
        .await
        .unwrap();
        let replay = unlink_thoughts(
            &pool,
            from,
            RelationKind::References,
            &target,
            already_deleted_event,
            None,
        )
        .await
        .unwrap();
        assert_eq!(replay.status, UnlinkStatus::AlreadyDeleted);
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
            source_event: relation_event(),
            claimed_producer_class: None,
        };
        let first = link_thoughts(&pool, req()).await.unwrap();
        unlink_thoughts(
            &pool,
            a,
            RelationKind::Refines,
            &target,
            relation_event(),
            None,
        )
        .await
        .unwrap();
        let second = link_thoughts(&pool, req()).await.unwrap();
        assert!(second.is_new);
        assert_ne!(first.link_id, second.link_id);
    }
}

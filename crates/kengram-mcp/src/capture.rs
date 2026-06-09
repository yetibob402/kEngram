//! The `capture` operation: write a thought row and enqueue embedding +
//! tag jobs.
//!
//! Capture does **not** call the embedder or tagger. It records the thought
//! (durable), enqueues a job in `pending_embeddings` (and, when a tagger is
//! configured, in `pending_tags`), and returns immediately with
//! `embedding_status: "pending"`. The `kengram worker` process drains both
//! queues and writes results on its next tick.
//!
//! M4: content_fingerprint dedup. We SHA-256 the content client-side before
//! the DB write so we can carry the fingerprint into `insert_thought`, which
//! does an `INSERT ... ON CONFLICT (content_fingerprint) DO NOTHING`. On
//! conflict the storage layer returns the pre-existing thought_id; we surface
//! that as `is_duplicate: true` and skip the enqueue calls so we don't
//! re-embed/re-tag content that already exists.

use kengram_core::{EmbeddingStatus, Metadata, Scope, Source, ThoughtId};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use uuid::Uuid;

/// Hard upper bound on a single thought's content. Enforced before the DB
/// write so callers get a clean rejection.
pub const MAX_CONTENT_LEN: usize = 1_048_576; // 1 MiB

#[derive(Debug, Clone)]
pub struct CaptureRequest {
    pub content: String,
    pub source: Source,
    pub scope: Option<Scope>,
    pub metadata: Option<Metadata>,
    pub argus_source_event: Option<ArgusSourceEventRequest>,
}

#[derive(Debug, Clone)]
pub struct ArgusSourceEventRequest {
    pub namespace: String,
    pub source_ref: String,
    pub payload_hash: String,
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Clone)]
pub struct ArgusSourceEventResponse {
    pub action: String,
    pub namespace: String,
    pub source_ref: String,
    pub payload_hash: String,
    pub status: String,
    pub thought_id: Option<ThoughtId>,
}

#[derive(Debug, Clone)]
pub struct CaptureResponse {
    pub thought_id: ThoughtId,
    pub embedding_status: EmbeddingStatus,
    /// `true` when the inserted fingerprint conflicted with an existing
    /// row — the returned `thought_id` belongs to the pre-existing row and
    /// no new embedding/tag jobs were enqueued. `false` when a fresh row
    /// was inserted.
    pub is_duplicate: bool,
    pub argus_source_event: Option<ArgusSourceEventResponse>,
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("content must be non-empty")]
    EmptyContent,

    #[error("content is too long: {got} bytes (max {max})")]
    ContentTooLong { got: usize, max: usize },

    #[error("invalid argus_source_event: {0}")]
    InvalidArgusSourceEvent(&'static str),

    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

/// Compute the SHA-256 of `content` as a 32-byte array. Mirrors what the
/// storage tests do when constructing `NewThought` directly.
fn fingerprint_of(content: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.finalize().into()
}

/// Capture a thought. Inserts the `thoughts` row (deduping by content
/// fingerprint), and — only on a fresh insert — enqueues an embedding job
/// and (when `tagger_model_id` is `Some`) a tag job. Always returns
/// `EmbeddingStatus::Pending`.
///
/// `embedder_model_id` is the active embedder's identity (e.g.
/// `"bge-m3:1024"`). The worker uses it to pair the row with the right
/// embedder on drain.
///
/// `tagger_model_id` is the active tagger's identity (e.g.
/// `"vllm/qwen3-coder:30b"`). `None` silent-disables the tag-job
/// enqueue — captures still work, the thought just stays with `tags = '{}'`
/// until a tagger is configured and the operator runs `kengram tag --rerun`.
pub async fn capture(
    pool: &PgPool,
    embedder_model_id: &str,
    tagger_model_id: Option<&str>,
    request: CaptureRequest,
) -> Result<CaptureResponse, CaptureError> {
    if request.content.is_empty() {
        return Err(CaptureError::EmptyContent);
    }
    if request.content.len() > MAX_CONTENT_LEN {
        return Err(CaptureError::ContentTooLong {
            got: request.content.len(),
            max: MAX_CONTENT_LEN,
        });
    }

    if request.argus_source_event.is_some() {
        return capture_with_argus_source_event(pool, embedder_model_id, tagger_model_id, request)
            .await;
    }

    let scope = request.scope.unwrap_or_default();
    let metadata = request.metadata.unwrap_or_default();
    let fingerprint = fingerprint_of(&request.content);

    let (inserted, is_new) = kengram_storage::insert_thought(
        pool,
        kengram_storage::NewThought {
            scope: &scope,
            content: &request.content,
            source: &request.source,
            metadata: &metadata,
            content_fingerprint: fingerprint,
        },
    )
    .await?;

    if is_new {
        kengram_storage::enqueue_embedding(
            pool,
            kengram_storage::target::THOUGHT,
            inserted.id.into_uuid(),
            embedder_model_id,
        )
        .await?;
        if let Some(tagger_id) = tagger_model_id {
            kengram_storage::enqueue_tag_job(pool, inserted.id, tagger_id).await?;
        }
    }

    Ok(CaptureResponse {
        thought_id: inserted.id,
        embedding_status: EmbeddingStatus::Pending,
        is_duplicate: !is_new,
        argus_source_event: None,
    })
}

async fn capture_with_argus_source_event(
    pool: &PgPool,
    embedder_model_id: &str,
    tagger_model_id: Option<&str>,
    request: CaptureRequest,
) -> Result<CaptureResponse, CaptureError> {
    let source_event = request
        .argus_source_event
        .ok_or(CaptureError::InvalidArgusSourceEvent(
            "missing source event",
        ))?;
    if source_event.namespace.trim().is_empty() {
        return Err(CaptureError::InvalidArgusSourceEvent(
            "namespace is required",
        ));
    }
    if source_event.source_ref.trim().is_empty() {
        return Err(CaptureError::InvalidArgusSourceEvent(
            "source_ref is required",
        ));
    }
    if source_event.payload_hash.trim().is_empty() {
        return Err(CaptureError::InvalidArgusSourceEvent(
            "payload_hash is required",
        ));
    }

    let scope = request.scope.unwrap_or_default();
    let metadata = request.metadata.unwrap_or_default();
    let source_event_metadata = source_event.metadata.unwrap_or_default();
    let fingerprint = fingerprint_of(&request.content);
    let fingerprint_slice: &[u8] = &fingerprint;

    let mut tx = pool
        .begin()
        .await
        .map_err(kengram_storage::StorageError::from)?;

    let existing = sqlx::query(
        r#"
        SELECT payload_hash, status, thought_id
        FROM argus_source_events
        WHERE namespace = $1 AND source_ref = $2
        FOR UPDATE
        "#,
    )
    .bind(&source_event.namespace)
    .bind(&source_event.source_ref)
    .fetch_optional(&mut *tx)
    .await
    .map_err(kengram_storage::StorageError::from)?;

    if let Some(row) = existing {
        let existing_hash: String = row
            .try_get("payload_hash")
            .map_err(kengram_storage::StorageError::from)?;
        let existing_status: String = row
            .try_get("status")
            .map_err(kengram_storage::StorageError::from)?;
        let existing_thought_id: Option<Uuid> = row
            .try_get("thought_id")
            .map_err(kengram_storage::StorageError::from)?;

        if existing_hash == source_event.payload_hash {
            sqlx::query(
                r#"
                UPDATE argus_source_events
                SET last_seen_at = NOW()
                WHERE namespace = $1 AND source_ref = $2
                "#,
            )
            .bind(&source_event.namespace)
            .bind(&source_event.source_ref)
            .execute(&mut *tx)
            .await
            .map_err(kengram_storage::StorageError::from)?;
            tx.commit()
                .await
                .map_err(kengram_storage::StorageError::from)?;

            let thought_id = existing_thought_id.map(ThoughtId::from);
            return Ok(CaptureResponse {
                thought_id: thought_id.unwrap_or_default(),
                embedding_status: EmbeddingStatus::Pending,
                is_duplicate: true,
                argus_source_event: Some(ArgusSourceEventResponse {
                    action: "duplicate_skip".to_string(),
                    namespace: source_event.namespace,
                    source_ref: source_event.source_ref,
                    payload_hash: source_event.payload_hash,
                    status: existing_status,
                    thought_id,
                }),
            });
        }

        let conflict_metadata = serde_json::json!({
            "conflict": {
                "incoming_payload_sha256": source_event.payload_hash,
                "prior_payload_sha256": existing_hash
            }
        });
        sqlx::query(
            r#"
            UPDATE argus_source_events
            SET status = 'conflict',
                error = 'payload_hash_conflict',
                last_seen_at = NOW(),
                metadata = metadata || $3::jsonb
            WHERE namespace = $1 AND source_ref = $2
            "#,
        )
        .bind(&source_event.namespace)
        .bind(&source_event.source_ref)
        .bind(conflict_metadata)
        .execute(&mut *tx)
        .await
        .map_err(kengram_storage::StorageError::from)?;
        tx.commit()
            .await
            .map_err(kengram_storage::StorageError::from)?;

        let thought_id = existing_thought_id.map(ThoughtId::from);
        return Ok(CaptureResponse {
            thought_id: thought_id.unwrap_or_default(),
            embedding_status: EmbeddingStatus::Pending,
            is_duplicate: true,
            argus_source_event: Some(ArgusSourceEventResponse {
                action: "conflict".to_string(),
                namespace: source_event.namespace,
                source_ref: source_event.source_ref,
                payload_hash: source_event.payload_hash,
                status: "conflict".to_string(),
                thought_id,
            }),
        });
    }

    sqlx::query(
        r#"
        INSERT INTO argus_source_events (namespace, source_ref, payload_hash, status, metadata)
        VALUES ($1, $2, $3, 'pending', $4::jsonb)
        "#,
    )
    .bind(&source_event.namespace)
    .bind(&source_event.source_ref)
    .bind(&source_event.payload_hash)
    .bind(source_event_metadata.as_value())
    .execute(&mut *tx)
    .await
    .map_err(kengram_storage::StorageError::from)?;

    let inserted = sqlx::query(
        r#"
        INSERT INTO thoughts (scope, content, source, metadata, content_fingerprint)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        ON CONFLICT (content_fingerprint) DO NOTHING
        RETURNING id
        "#,
    )
    .bind(scope.as_str())
    .bind(&request.content)
    .bind(request.source.as_str())
    .bind(metadata.as_value())
    .bind(fingerprint_slice)
    .fetch_optional(&mut *tx)
    .await
    .map_err(kengram_storage::StorageError::from)?;

    let (thought_id, is_new) = if let Some(row) = inserted {
        let id: Uuid = row
            .try_get("id")
            .map_err(kengram_storage::StorageError::from)?;
        (ThoughtId::from(id), true)
    } else {
        let row = sqlx::query(
            r#"
            SELECT id
            FROM thoughts
            WHERE content_fingerprint = $1
            "#,
        )
        .bind(fingerprint_slice)
        .fetch_one(&mut *tx)
        .await
        .map_err(kengram_storage::StorageError::from)?;
        let id: Uuid = row
            .try_get("id")
            .map_err(kengram_storage::StorageError::from)?;
        (ThoughtId::from(id), false)
    };

    if is_new {
        sqlx::query(
            r#"
            INSERT INTO pending_embeddings (target_kind, target_id, model_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
            "#,
        )
        .bind(kengram_storage::target::THOUGHT)
        .bind(thought_id.into_uuid())
        .bind(embedder_model_id)
        .execute(&mut *tx)
        .await
        .map_err(kengram_storage::StorageError::from)?;

        if let Some(tagger_id) = tagger_model_id {
            sqlx::query(
                r#"
                INSERT INTO pending_tags (thought_id, tagger_model_id)
                VALUES ($1, $2)
                ON CONFLICT (thought_id) DO NOTHING
                "#,
            )
            .bind(thought_id.into_uuid())
            .bind(tagger_id)
            .execute(&mut *tx)
            .await
            .map_err(kengram_storage::StorageError::from)?;
        }
    }

    sqlx::query(
        r#"
        UPDATE argus_source_events
        SET thought_id = $3, status = 'stored', last_seen_at = NOW()
        WHERE namespace = $1 AND source_ref = $2
        "#,
    )
    .bind(&source_event.namespace)
    .bind(&source_event.source_ref)
    .bind(thought_id.into_uuid())
    .execute(&mut *tx)
    .await
    .map_err(kengram_storage::StorageError::from)?;

    tx.commit()
        .await
        .map_err(kengram_storage::StorageError::from)?;

    Ok(CaptureResponse {
        thought_id,
        embedding_status: EmbeddingStatus::Pending,
        is_duplicate: !is_new,
        argus_source_event: Some(ArgusSourceEventResponse {
            action: "stored".to_string(),
            namespace: source_event.namespace,
            source_ref: source_event.source_ref,
            payload_hash: source_event.payload_hash,
            status: "stored".to_string(),
            thought_id: Some(thought_id),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::EmbeddingModel;
    use serde_json::json;

    const TEST_EMBEDDER_MODEL_ID: &str = "bge-m3:1024";
    const TEST_TAGGER_MODEL_ID: &str = "fake/tagger";

    fn req(content: &str, source: &str) -> CaptureRequest {
        CaptureRequest {
            content: content.to_string(),
            source: Source::new(source).unwrap(),
            scope: None,
            metadata: None,
            argus_source_event: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn writes_thought_and_enqueues_returns_pending(pool: PgPool) {
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("first thought", "manual"),
        )
        .await
        .unwrap();

        assert_eq!(resp.embedding_status, EmbeddingStatus::Pending);
        assert!(!resp.is_duplicate);

        let fetched = kengram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.content, "first thought");

        // Queue rows exist; no embedding row yet.
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
        let tag_jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(tag_jobs.len(), 1);
        assert!(
            !kengram_storage::thought_has_embedding(
                &pool,
                resp.thought_id,
                &EmbeddingModel::bge_m3(),
            )
            .await
            .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn empty_content_returns_error(pool: PgPool) {
        let err = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("", "manual"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CaptureError::EmptyContent));
        // Errored before the insert; queues stay empty.
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);
        let tag_jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(tag_jobs.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn overlong_content_returns_error(pool: PgPool) {
        let big = "x".repeat(MAX_CONTENT_LEN + 1);
        let err = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req(&big, "manual"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CaptureError::ContentTooLong { got, max } if got > max));
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn defaults_scope_to_global_when_missing(pool: PgPool) {
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("hello", "manual"),
        )
        .await
        .unwrap();
        let fetched = kengram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.scope, Scope::global());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn defaults_metadata_to_empty_when_missing(pool: PgPool) {
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("hello", "manual"),
        )
        .await
        .unwrap();
        let fetched = kengram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.metadata, Metadata::empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn preserves_scope_source_metadata(pool: PgPool) {
        let request = CaptureRequest {
            content: "remember this".to_string(),
            source: Source::new("agent:claude-code").unwrap(),
            scope: Some(Scope::new("work.tcgplayer").unwrap()),
            metadata: Some(Metadata::from(
                json!({"session_id": "abc", "tool_name": "TodoWrite"}),
            )),
            argus_source_event: None,
        };
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            request.clone(),
        )
        .await
        .unwrap();

        let fetched = kengram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.scope, request.scope.unwrap());
        assert_eq!(fetched.source, request.source);
        assert_eq!(fetched.metadata, request.metadata.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn argus_source_event_gates_store_duplicate_and_conflict(pool: PgPool) {
        let source_event = ArgusSourceEventRequest {
            namespace: "agents/trinity".to_string(),
            source_ref: "mem_save:agents/trinity:source-event-test".to_string(),
            payload_hash: "payload-a".to_string(),
            metadata: Some(Metadata::from(json!({"legacy_tool": "mem_save"}))),
        };

        let request = CaptureRequest {
            content: "Argus source-event capture test v1".to_string(),
            source: Source::new("agent:trinity").unwrap(),
            scope: Some(Scope::new("agents/trinity").unwrap()),
            metadata: Some(Metadata::from(json!({"title": "Argus source-event test"}))),
            argus_source_event: Some(source_event.clone()),
        };

        let first = capture(&pool, TEST_EMBEDDER_MODEL_ID, None, request.clone())
            .await
            .unwrap();
        assert!(!first.is_duplicate);
        let first_event = first.argus_source_event.as_ref().unwrap();
        assert_eq!(first_event.action, "stored");
        assert_eq!(first_event.status, "stored");
        assert_eq!(first_event.thought_id, Some(first.thought_id));

        let dup = capture(&pool, TEST_EMBEDDER_MODEL_ID, None, request.clone())
            .await
            .unwrap();
        assert!(dup.is_duplicate);
        let dup_event = dup.argus_source_event.as_ref().unwrap();
        assert_eq!(dup_event.action, "duplicate_skip");
        assert_eq!(dup_event.thought_id, Some(first.thought_id));

        let thoughts_before_conflict: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM thoughts")
            .fetch_one(&pool)
            .await
            .unwrap();

        let mut conflict_request = request;
        conflict_request.content = "Argus source-event capture test v2".to_string();
        conflict_request.argus_source_event = Some(ArgusSourceEventRequest {
            payload_hash: "payload-b".to_string(),
            ..source_event
        });
        let conflict = capture(&pool, TEST_EMBEDDER_MODEL_ID, None, conflict_request)
            .await
            .unwrap();
        let conflict_event = conflict.argus_source_event.as_ref().unwrap();
        assert_eq!(conflict_event.action, "conflict");
        assert_eq!(conflict_event.status, "conflict");
        assert_eq!(conflict_event.thought_id, Some(first.thought_id));

        let thoughts_after_conflict: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM thoughts")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(thoughts_after_conflict, thoughts_before_conflict);

        let row = sqlx::query(
            "SELECT status, error, payload_hash FROM argus_source_events WHERE namespace = $1 AND source_ref = $2",
        )
        .bind("agents/trinity")
        .bind("mem_save:agents/trinity:source-event-test")
        .fetch_one(&pool)
        .await
        .unwrap();
        let status: String = row.try_get("status").unwrap();
        let error: Option<String> = row.try_get("error").unwrap();
        let payload_hash: String = row.try_get("payload_hash").unwrap();
        assert_eq!(status, "conflict");
        assert_eq!(error.as_deref(), Some("payload_hash_conflict"));
        assert_eq!(payload_hash, "payload-a");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_targets_thought_kind_with_active_model(pool: PgPool) {
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("queue me", "manual"),
        )
        .await
        .unwrap();

        // Inspect the queue row directly.
        let row =
            sqlx::query!(r#"SELECT target_kind, target_id, model_id FROM pending_embeddings"#,)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row.target_kind, "thought");
        assert_eq!(row.target_id, resp.thought_id.into_uuid());
        assert_eq!(row.model_id, TEST_EMBEDDER_MODEL_ID);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn returns_existing_id_on_duplicate_content(pool: PgPool) {
        let first = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("same content", "manual"),
        )
        .await
        .unwrap();
        assert!(!first.is_duplicate);
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
        let tag_jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(tag_jobs.len(), 1);

        // Second capture with same content returns the existing id + duplicate flag.
        let second = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("same content", "manual"),
        )
        .await
        .unwrap();
        assert!(second.is_duplicate);
        assert_eq!(first.thought_id, second.thought_id);

        // No new jobs were enqueued — queues unchanged.
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
        let tag_jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(tag_jobs.len(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueues_embedding_and_tag_jobs_on_new_insert(pool: PgPool) {
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            Some(TEST_TAGGER_MODEL_ID),
            req("dual-enqueue", "manual"),
        )
        .await
        .unwrap();

        let pending_embeds = kengram_storage::count_pending(&pool).await.unwrap();
        assert_eq!(pending_embeds, 1);

        let pending_tag_jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(pending_tag_jobs.len(), 1);
        assert_eq!(pending_tag_jobs[0].thought_id, resp.thought_id);
        assert_eq!(pending_tag_jobs[0].tagger_model_id, TEST_TAGGER_MODEL_ID);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn skips_tag_enqueue_when_tagger_disabled(pool: PgPool) {
        // tagger_model_id = None silent-disables the tag enqueue. Embedding
        // job still goes through.
        let resp = capture(
            &pool,
            TEST_EMBEDDER_MODEL_ID,
            None,
            req("no-tagger", "manual"),
        )
        .await
        .unwrap();

        assert!(!resp.is_duplicate);
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
        let tag_jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(tag_jobs.is_empty());
    }
}

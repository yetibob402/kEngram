//! Capture orchestration for migration 0030's database gate.
//!
//! The database derives producer policy from `session_user`, computes the
//! fingerprint from stored content, performs exact/semantic decisions, and
//! atomically records source-event, queue, relation, and gate evidence.

use kengram_core::{EmbeddingStatus, Metadata, Scope, Source, ThoughtId};
use sqlx::PgPool;
use time::OffsetDateTime;

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

/// Gate-only inputs used by callers that can supply source time, a
/// synchronous semantic vector, or atomic relation intents. Keeping these
/// separate preserves the established basic capture request for internal
/// queue/search fixtures.
#[derive(Debug, Clone, Default)]
pub struct CaptureGateOptions {
    pub source_created_at: Option<OffsetDateTime>,
    pub candidate_embedding: Option<Vec<f32>>,
    pub bypass_reason: Option<serde_json::Value>,
    pub relation_intents: Vec<serde_json::Value>,
    pub claimed_producer_class: Option<String>,
    pub correlation_id: Option<String>,
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
    pub dedup_kind: Option<String>,
    pub matched_thought_id: Option<ThoughtId>,
    pub similarity: Option<f64>,
    pub relation_results: serde_json::Value,
    pub gate_event_id: Option<uuid::Uuid>,
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

/// Capture through the one database chokepoint.  A caller without a
/// synchronous candidate embedding must provide (or receives) a structured
/// bypass reason; shadow/enforce failures then keep and queue the thought.
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
    capture_with_gate_options(
        pool,
        embedder_model_id,
        tagger_model_id,
        request,
        CaptureGateOptions::default(),
    )
    .await
}

pub async fn capture_with_gate_options(
    pool: &PgPool,
    embedder_model_id: &str,
    tagger_model_id: Option<&str>,
    request: CaptureRequest,
    options: CaptureGateOptions,
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

    let scope = request.scope.unwrap_or_default();
    let metadata = request.metadata.unwrap_or_default();
    let source_event = request.argus_source_event;
    if let Some(event) = &source_event {
        if event.namespace.trim().is_empty() {
            return Err(CaptureError::InvalidArgusSourceEvent(
                "namespace is required",
            ));
        }
        if event.source_ref.trim().is_empty() {
            return Err(CaptureError::InvalidArgusSourceEvent(
                "source_ref is required",
            ));
        }
        if event.payload_hash.trim().is_empty() {
            return Err(CaptureError::InvalidArgusSourceEvent(
                "payload_hash is required",
            ));
        }
    }

    let source_event_metadata = source_event
        .as_ref()
        .and_then(|event| event.metadata.as_ref())
        .map(Metadata::as_value);
    let relation_intents = serde_json::Value::Array(options.relation_intents);
    let default_bypass = serde_json::json!({
        "code": "candidate_embedding_unavailable",
        "detail": "capture caller did not provide a synchronous bge-m3 vector"
    });
    let bypass_reason = if options.candidate_embedding.is_none() {
        Some(options.bypass_reason.as_ref().unwrap_or(&default_bypass))
    } else {
        options.bypass_reason.as_ref()
    };
    let source_created_at = options
        .source_created_at
        .or_else(|| Some(OffsetDateTime::now_utc()));

    let result = kengram_storage::corpus_hygiene::capture_thought_gated(
        pool,
        kengram_storage::corpus_hygiene::GatedCaptureRequest {
            scope: scope.as_str(),
            content: &request.content,
            source: request.source.as_str(),
            metadata: metadata.as_value(),
            source_created_at,
            candidate_embedding: options.candidate_embedding.as_deref(),
            embedding_model_id: Some(embedder_model_id),
            embedding_model_version: Some(1),
            bypass_reason,
            source_event_namespace: source_event.as_ref().map(|event| event.namespace.as_str()),
            source_event_ref: source_event.as_ref().map(|event| event.source_ref.as_str()),
            source_event_payload_hash: source_event
                .as_ref()
                .map(|event| event.payload_hash.as_str()),
            source_event_metadata,
            relation_intents: &relation_intents,
            tagger_model_id,
            claimed_producer_class: options.claimed_producer_class.as_deref(),
            correlation_id: options.correlation_id.as_deref(),
            force_keep_token: None,
        },
    )
    .await?;
    // A conflicting replay does not select a new corpus row, but the source
    // event ledger still identifies the original thought for the established
    // MCP conflict response.
    let thought_id = result
        .thought_id
        .or(result.matched_thought_id)
        .ok_or_else(|| {
            CaptureError::Storage(kengram_storage::StorageError::Database(
                sqlx::Error::Protocol(format!(
                    "capture gate returned action={} without thought_id",
                    result.action
                )),
            ))
        })?;
    let thought_id = ThoughtId::from(thought_id);
    let is_duplicate = matches!(
        result.action.as_str(),
        "exact_duplicate" | "semantic_duplicate"
    );
    let argus_source_event = source_event.map(|event| {
        let action = match result.source_event_action.as_deref() {
            // Preserve the established MCP response contract while the gate
            // records the more precise replay disposition in its ledger.
            Some("replay") => "duplicate_skip".to_string(),
            Some(action) => action.to_string(),
            None => result.action.clone(),
        };
        ArgusSourceEventResponse {
            action,
            namespace: event.namespace,
            source_ref: event.source_ref,
            payload_hash: event.payload_hash,
            status: result
                .source_event_status
                .clone()
                .unwrap_or_else(|| "stored".to_string()),
            thought_id: Some(thought_id),
        }
    });

    Ok(CaptureResponse {
        thought_id,
        embedding_status: EmbeddingStatus::Pending,
        is_duplicate,
        argus_source_event,
        dedup_kind: is_duplicate.then(|| result.action.clone()),
        matched_thought_id: result.matched_thought_id.map(ThoughtId::from),
        similarity: result.similarity,
        relation_results: result.relation_results,
        gate_event_id: result.gate_event_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::EmbeddingModel;
    use serde_json::json;
    use sqlx::{PgConnection, Row, postgres::PgRow};
    use std::time::Duration;

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

    fn unit_vector_literal() -> String {
        format!("[{}]", vec!["0.03125"; 1024].join(","))
    }

    const SOURCE_EVENT_CLAIM_TEST_LOCK: i64 = 7_301_001;

    async fn install_source_event_claim_barrier(pool: &PgPool) {
        sqlx::query(
            r#"
            CREATE FUNCTION public.test_block_source_event_claim()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $test$
            BEGIN
                PERFORM pg_catalog.pg_advisory_xact_lock(7301001);
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
            CREATE TRIGGER test_block_source_event_claim
            BEFORE INSERT ON public.argus_source_events
            FOR EACH ROW EXECUTE FUNCTION public.test_block_source_event_claim()
            "#,
        )
        .execute(pool)
        .await
        .unwrap();
    }

    async fn wait_for_active_test_clients(pool: &PgPool, prefix: &str, expected: i64) {
        let pattern = format!("{prefix}%");
        for _ in 0..200 {
            let active: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM pg_catalog.pg_stat_activity WHERE application_name LIKE $1 AND state = 'active'",
            )
            .bind(&pattern)
            .fetch_one(pool)
            .await
            .unwrap();
            if active >= expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {expected} concurrent {prefix} clients");
    }

    async fn gated_capture_on_connection(
        connection: &mut PgConnection,
        scope: &str,
        content: &str,
        source_event: Option<(&str, &str, &str)>,
        relation_intents: &serde_json::Value,
        tagger_model_id: Option<&str>,
    ) -> PgRow {
        let (namespace, source_ref, payload_hash) = source_event
            .map(|(namespace, source_ref, payload_hash)| {
                (Some(namespace), Some(source_ref), Some(payload_hash))
            })
            .unwrap_or((None, None, None));
        sqlx::query(
            r#"
            SELECT *
            FROM public.capture_thought_gated(
                p_scope => $1,
                p_content => $2,
                p_source => 'test',
                p_metadata => '{}'::jsonb,
                p_source_created_at => NULL::timestamptz,
                p_candidate_embedding => $3::vector(1024),
                p_embedding_model_id => 'bge-m3:1024',
                p_embedding_model_version => 1,
                p_bypass_reason => NULL::jsonb,
                p_source_event_namespace => $4,
                p_source_event_ref => $5,
                p_source_event_payload_hash => $6,
                p_source_event_metadata => '{}'::jsonb,
                p_relation_intents => $7,
                p_tagger_model_id => $8,
                p_claimed_producer_class => NULL::text,
                p_correlation_id => NULL::text,
                p_force_keep_token => NULL::text
            )
            "#,
        )
        .bind(scope)
        .bind(content)
        .bind(unit_vector_literal())
        .bind(namespace)
        .bind(source_ref)
        .bind(payload_hash)
        .bind(relation_intents)
        .bind(tagger_model_id)
        .fetch_one(&mut *connection)
        .await
        .unwrap()
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
    async fn concurrent_identical_source_event_first_delivery_replays_without_error(pool: PgPool) {
        install_source_event_claim_barrier(&pool).await;

        let mut blocker = pool.acquire().await.unwrap();
        sqlx::query("SELECT pg_catalog.pg_advisory_lock($1)")
            .bind(SOURCE_EVENT_CLAIM_TEST_LOCK)
            .execute(&mut *blocker)
            .await
            .unwrap();

        let spawn_capture = |pool: PgPool, application_name: &'static str| {
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
                let row = gated_capture_on_connection(
                    &mut connection,
                    "agents/concurrent-source-claim",
                    "Concurrent byte-identical first deliveries share one durable source claim.",
                    Some((
                        "tests/concurrent-source-claim",
                        "same-first-delivery",
                        "same-payload",
                    )),
                    &json!([]),
                    None,
                )
                .await;
                let result = (
                    row.try_get::<uuid::Uuid, _>("thought_id").unwrap(),
                    row.try_get::<String, _>("action").unwrap(),
                    row.try_get::<String, _>("source_event_status").unwrap(),
                    row.try_get::<String, _>("source_event_action").unwrap(),
                );
                sqlx::query("RESET SESSION AUTHORIZATION")
                    .execute(&mut *connection)
                    .await
                    .unwrap();
                result
            })
        };

        let first = spawn_capture(pool.clone(), "source_claim_race_first");
        let second = spawn_capture(pool.clone(), "source_claim_race_second");
        wait_for_active_test_clients(&pool, "source_claim_race_", 2).await;
        sqlx::query("SELECT pg_catalog.pg_advisory_unlock($1)")
            .bind(SOURCE_EVENT_CLAIM_TEST_LOCK)
            .execute(&mut *blocker)
            .await
            .unwrap();

        let first = first.await.unwrap();
        let second = second.await.unwrap();
        assert_eq!(first.0, second.0);
        assert_eq!(first.2, "stored");
        assert_eq!(second.2, "stored");
        let mut dispositions = [(first.1, first.3), (second.1, second.3)];
        dispositions.sort();
        assert_eq!(
            dispositions,
            [
                ("exact_duplicate".to_string(), "replay".to_string()),
                ("inserted".to_string(), "stored".to_string()),
            ]
        );
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

    #[sqlx::test(migrations = "../../migrations")]
    async fn invalid_candidate_model_queues_bge_and_audits_derived_bypass(pool: PgPool) {
        let metadata = json!({});
        let intents = json!([]);
        let candidate = vec![0.25_f32; 1024];
        let result = kengram_storage::corpus_hygiene::capture_thought_gated(
            &pool,
            kengram_storage::corpus_hygiene::GatedCaptureRequest {
                scope: "agents/model-probe",
                content: "A candidate with an unauthenticated model assertion must queue the reviewed BGE model.",
                source: "test",
                metadata: &metadata,
                source_created_at: Some(OffsetDateTime::now_utc()),
                candidate_embedding: Some(&candidate),
                embedding_model_id: Some("wrong-model"),
                embedding_model_version: Some(7),
                bypass_reason: None,
                source_event_namespace: None,
                source_event_ref: None,
                source_event_payload_hash: None,
                source_event_metadata: None,
                relation_intents: &intents,
                tagger_model_id: None,
                claimed_producer_class: None,
                correlation_id: Some("wrong-model-probe"),
                force_keep_token: None,
            },
        )
        .await
        .unwrap();
        let thought_id = result.thought_id.unwrap();
        let queued_model: String = sqlx::query_scalar(
            "SELECT model_id FROM pending_embeddings WHERE target_kind='thought' AND target_id=$1",
        )
        .bind(thought_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(queued_model, "bge-m3:1024");
        let bypass_code: String = sqlx::query_scalar(
            "SELECT bypass_reason->>'code' FROM thought_ingest_gate_events WHERE id=$1",
        )
        .bind(result.gate_event_id.unwrap())
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(bypass_code, "candidate_embedding_contract_mismatch");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn capture_relation_intent_cannot_override_from_endpoint(pool: PgPool) {
        let metadata = json!({});
        let bypass = json!({"code": "fixture"});
        let event_metadata = json!({});
        let intents = json!([{
            "action": "create",
            "from_thought_id": uuid::Uuid::new_v4(),
            "relation": "supports",
            "to_kind": "thought",
            "to_value": uuid::Uuid::new_v4(),
            "source": "agent"
        }]);
        let error = kengram_storage::corpus_hygiene::capture_thought_gated(
            &pool,
            kengram_storage::corpus_hygiene::GatedCaptureRequest {
                scope: "agents/relation-probe",
                content: "A capture caller cannot redirect its atomic relation away from the gated thought.",
                source: "test",
                metadata: &metadata,
                source_created_at: Some(OffsetDateTime::now_utc()),
                candidate_embedding: None,
                embedding_model_id: Some("bge-m3:1024"),
                embedding_model_version: None,
                bypass_reason: Some(&bypass),
                source_event_namespace: Some("tests/capture-relations"),
                source_event_ref: Some("forged-from-endpoint"),
                source_event_payload_hash: Some("forged-from-endpoint"),
                source_event_metadata: Some(&event_metadata),
                relation_intents: &intents,
                tagger_model_id: None,
                claimed_producer_class: None,
                correlation_id: None,
                force_keep_token: None,
            },
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid_capture_relation_intent")
        );
        let counts: (i64, i64, i64) = sqlx::query_as(
            "SELECT (SELECT COUNT(*) FROM thoughts), (SELECT COUNT(*) FROM argus_source_events), (SELECT COUNT(*) FROM thought_ingest_gate_events)",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(counts, (0, 0, 0));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn protected_atoms_use_exact_canonical_membership(pool: PgPool) {
        sqlx::query(
            "UPDATE corpus_hygiene_gate_settings SET mode='enforce' WHERE principal_name='kengram_rt_session'",
        )
        .execute(&pool)
        .await
        .unwrap();

        let mut connection = pool.acquire().await.unwrap();
        sqlx::query("SET SESSION AUTHORIZATION kengram_rt_session")
            .execute(&mut *connection)
            .await
            .unwrap();

        let mut outcomes = Vec::new();
        for (case_name, keeper_atom, candidate_atom) in [
            ("bare-count", "12", "1"),
            ("issue-number", "#12", "#1"),
            ("currency-amount", "$12", "$1"),
        ] {
            let scope = format!("agents/protected-{case_name}");
            let keeper = format!(
                "The operational queue reports {keeper_atom} pending items and this deliberately long shared description repeats the same stable words so semantic similarity stays exact while the protected fact differs."
            );
            let candidate = format!(
                "The operational queue reports {candidate_atom} pending items and this deliberately long shared description repeats the same stable words so semantic similarity stays exact while the protected fact differs."
            );
            let empty_intents = json!([]);
            let first = gated_capture_on_connection(
                &mut connection,
                &scope,
                &keeper,
                None,
                &empty_intents,
                None,
            )
            .await;
            let second = gated_capture_on_connection(
                &mut connection,
                &scope,
                &candidate,
                None,
                &empty_intents,
                None,
            )
            .await;
            outcomes.push((
                case_name,
                first.try_get::<String, _>("action").unwrap(),
                second.try_get::<String, _>("action").unwrap(),
                second.try_get::<uuid::Uuid, _>("gate_event_id").unwrap(),
                candidate_atom.to_string(),
            ));
        }

        sqlx::query("RESET SESSION AUTHORIZATION")
            .execute(&mut *connection)
            .await
            .unwrap();
        drop(connection);

        for (case_name, first_action, second_action, gate_event_id, candidate_atom) in outcomes {
            assert_eq!(first_action, "inserted", "{case_name} keeper");
            assert_eq!(second_action, "inserted", "{case_name} candidate");
            let missing: Vec<String> = sqlx::query_scalar(
                "SELECT unnest(missing_protected_atoms) FROM thought_ingest_gate_events WHERE id=$1",
            )
            .bind(gate_event_id)
            .fetch_all(&pool)
            .await
            .unwrap();
            assert_eq!(missing, vec![candidate_atom], "{case_name}");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn capture_replay_rejects_changed_canonical_relation_intent(pool: PgPool) {
        let mut connection = pool.acquire().await.unwrap();
        sqlx::query("SET SESSION AUTHORIZATION kengram_rt_native_mcp")
            .execute(&mut *connection)
            .await
            .unwrap();

        let empty_intents = json!([]);
        let target = gated_capture_on_connection(
            &mut connection,
            "agents/relation-replay",
            "Target thought for changed relation intent replay validation.",
            None,
            &empty_intents,
            None,
        )
        .await
        .try_get::<uuid::Uuid, _>("thought_id")
        .unwrap();
        let supports = json!([{
            "action": "create",
            "relation": "supports",
            "to_kind": "thought",
            "to_value": target,
            "source": "agent",
        }]);
        let references = json!([{
            "action": "create",
            "relation": "references",
            "to_kind": "thought",
            "to_value": target,
            "source": "agent",
        }]);
        let source_identity = Some(("tests/capture-replay", "changed-intent", "payload-a"));
        let first = gated_capture_on_connection(
            &mut connection,
            "agents/relation-replay",
            "Source thought whose durable replay intent must remain supports.",
            source_identity,
            &supports,
            None,
        )
        .await;
        let source_thought_id = first.try_get::<uuid::Uuid, _>("thought_id").unwrap();
        let replay = gated_capture_on_connection(
            &mut connection,
            "agents/relation-replay",
            "Source thought whose durable replay intent must remain supports.",
            source_identity,
            &references,
            None,
        )
        .await;
        let replay_action = replay.try_get::<String, _>("action").unwrap();

        sqlx::query("RESET SESSION AUTHORIZATION")
            .execute(&mut *connection)
            .await
            .unwrap();
        drop(connection);

        assert_eq!(replay_action, "source_event_conflict");
        let relations: Vec<String> = sqlx::query_scalar(
            "SELECT relation FROM thought_links WHERE from_thought_id=$1 AND deleted_at IS NULL ORDER BY relation",
        )
        .bind(source_thought_id)
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(relations, vec!["supports"]);
        let recorded_relation: String = sqlx::query_scalar(
            "SELECT operations->0->>'relation' FROM thought_relation_request_events WHERE source_event_namespace='tests/capture-replay' AND source_event_ref='changed-intent'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(recorded_relation, "supports");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn runtime_role_duplicate_is_classified_by_gate_without_table_update(pool: PgPool) {
        let runtime_can_update: bool = sqlx::query_scalar(
            "SELECT has_table_privilege('kengram_rt_session', 'argus_source_events', 'UPDATE')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(!runtime_can_update);

        let mut connection = pool.acquire().await.unwrap();
        sqlx::query("SET SESSION AUTHORIZATION kengram_rt_session")
            .execute(&mut *connection)
            .await
            .unwrap();
        let identity = Some(("tests/runtime-adapter", "duplicate-1", "payload-1"));
        let content = "An adapter-equivalent duplicate must be accepted and classified through the security-definer gate without direct source-event UPDATE privilege.";
        let first = gated_capture_on_connection(
            &mut connection,
            "agents/runtime-adapter",
            content,
            identity,
            &json!([]),
            None,
        )
        .await;
        let replay = gated_capture_on_connection(
            &mut connection,
            "agents/runtime-adapter",
            content,
            identity,
            &json!([]),
            None,
        )
        .await;
        let dispositions = (
            first.try_get::<String, _>("action").unwrap(),
            first.try_get::<String, _>("source_event_action").unwrap(),
            replay.try_get::<String, _>("action").unwrap(),
            replay.try_get::<String, _>("source_event_action").unwrap(),
        );
        sqlx::query("RESET SESSION AUTHORIZATION")
            .execute(&mut *connection)
            .await
            .unwrap();

        assert_eq!(
            dispositions,
            (
                "inserted".into(),
                "stored".into(),
                "exact_duplicate".into(),
                "replay".into()
            )
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn contended_family_lock_fails_open_and_queues_with_signal(pool: PgPool) {
        sqlx::query(
            "UPDATE corpus_hygiene_gate_settings SET mode='enforce' WHERE principal_name='kengram_rt_session'",
        )
        .execute(&pool)
        .await
        .unwrap();

        let agent_key = "blocked-lock-agent";
        let mut lock_connection = pool.acquire().await.unwrap();
        sqlx::query("BEGIN")
            .execute(&mut *lock_connection)
            .await
            .unwrap();
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 727061667331::bigint))")
            .bind(agent_key)
            .execute(&mut *lock_connection)
            .await
            .unwrap();

        let mut capture_connection = pool.acquire().await.unwrap();
        sqlx::query("SET SESSION AUTHORIZATION kengram_rt_session")
            .execute(&mut *capture_connection)
            .await
            .unwrap();
        let started = std::time::Instant::now();
        let row = gated_capture_on_connection(
            &mut capture_connection,
            &format!("agents/{agent_key}"),
            "A contended family lock must keep this capture and queue ordinary asynchronous embedding and tag work instead of waiting.",
            None,
            &json!([]),
            Some(TEST_TAGGER_MODEL_ID),
        )
        .await;
        let elapsed = started.elapsed();
        let thought_id = row.try_get::<uuid::Uuid, _>("thought_id").unwrap();
        let action = row.try_get::<String, _>("action").unwrap();
        let gate_event_id = row.try_get::<uuid::Uuid, _>("gate_event_id").unwrap();
        sqlx::query("RESET SESSION AUTHORIZATION")
            .execute(&mut *capture_connection)
            .await
            .unwrap();
        sqlx::query("ROLLBACK")
            .execute(&mut *lock_connection)
            .await
            .unwrap();
        drop(capture_connection);
        drop(lock_connection);

        assert_eq!(action, "fail_open_insert");
        assert!(elapsed < std::time::Duration::from_secs(1));
        let pending_embedding: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pending_embeddings WHERE target_kind='thought' AND target_id=$1 AND model_id='bge-m3:1024'",
        )
        .bind(thought_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        let pending_tag: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM pending_tags WHERE thought_id=$1")
                .bind(thought_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        let signal: (String, String) = sqlx::query_as(
            "SELECT action, bypass_reason->>'code' FROM thought_ingest_gate_events WHERE id=$1",
        )
        .bind(gate_event_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(pending_embedding, 1);
        assert_eq!(pending_tag, 1);
        assert_eq!(
            signal,
            (
                "fail_open_insert".into(),
                "similarity_lock_contended".into()
            )
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn comparison_statement_timeout_fails_open_and_queues(pool: PgPool) {
        let mut lock_connection = pool.acquire().await.unwrap();
        sqlx::query("BEGIN")
            .execute(&mut *lock_connection)
            .await
            .unwrap();
        sqlx::query("LOCK TABLE thought_embeddings_bge_m3 IN ACCESS EXCLUSIVE MODE")
            .execute(&mut *lock_connection)
            .await
            .unwrap();

        let metadata = json!({});
        let intents = json!([]);
        let candidate = vec![0.03125_f32; 1024];
        let started = std::time::Instant::now();
        let result = kengram_storage::corpus_hygiene::capture_thought_gated(
            &pool,
            kengram_storage::corpus_hygiene::GatedCaptureRequest {
                scope: "agents/comparison-timeout",
                content: "A blocked similarity table read must become a durable queued fail-open capture before the total caller deadline.",
                source: "test",
                metadata: &metadata,
                source_created_at: Some(OffsetDateTime::now_utc()),
                candidate_embedding: Some(&candidate),
                embedding_model_id: Some("bge-m3:1024"),
                embedding_model_version: Some(1),
                bypass_reason: None,
                source_event_namespace: None,
                source_event_ref: None,
                source_event_payload_hash: None,
                source_event_metadata: None,
                relation_intents: &intents,
                tagger_model_id: Some(TEST_TAGGER_MODEL_ID),
                claimed_producer_class: None,
                correlation_id: Some("comparison-timeout"),
                force_keep_token: None,
            },
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();
        sqlx::query("ROLLBACK")
            .execute(&mut *lock_connection)
            .await
            .unwrap();
        drop(lock_connection);

        assert_eq!(result.action, "fail_open_insert");
        assert!(elapsed < std::time::Duration::from_secs(1));
        let thought_id = result.thought_id.unwrap();
        let pending_embedding: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pending_embeddings WHERE target_kind='thought' AND target_id=$1",
        )
        .bind(thought_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        let pending_tag: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM pending_tags WHERE thought_id=$1")
                .bind(thought_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        let bypass: (String, String) = sqlx::query_as(
            "SELECT bypass_reason->>'code', bypass_reason->>'sqlstate' FROM thought_ingest_gate_events WHERE id=$1",
        )
        .bind(result.gate_event_id.unwrap())
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(pending_embedding, 1);
        assert_eq!(pending_tag, 1);
        assert_eq!(
            bypass,
            ("similarity_comparison_unavailable".into(), "57014".into())
        );
    }
}

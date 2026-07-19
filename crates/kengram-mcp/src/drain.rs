//! Drainers for the two background queues — `pending_embeddings` (embed
//! drainer) and `pending_tags` (tag drainer).
//!
//! Both are pulled on every `kengram worker` tick. Each tick processes a
//! bounded batch (`batch_size`) and reports per-job outcome. On transient
//! failures the job stays queued (attempts++) for the next tick; on
//! permanent failures we either log and drop the job (tag drainer, after
//! `MAX_TAG_ATTEMPTS`) or set `last_error` (embed drainer, keeping the row
//! for operator inspection).
//!
//! Crash-replay safety for the embed drainer: `insert_thought_embedding` is
//! idempotent via `ON CONFLICT DO NOTHING`. So if the worker dies between
//! `insert_thought_embedding` and `mark_embedded`, the next tick re-claims
//! the row, re-embeds, re-inserts (no-op), and marks embedded — clean.

use crate::finalize;
use kengram_core::{
    Embedder, EmbedderError, Embedding, EmbeddingError, ExtractedRelation, Tagger, ThoughtId,
};
use sha2::{Digest, Sha256};
use sqlx::PgPool;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrainReport {
    /// Number of jobs claimed this drain call.
    pub found: usize,
    /// Number that successfully embedded + persisted + marked.
    pub embedded: usize,
    /// Number permanently skipped and dequeued by the ingest hygiene fence.
    pub skipped: usize,
    /// Number that failed embed/persist.
    pub failed: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DrainError {
    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

/// Drain up to `batch_size` jobs from `pending_embeddings`. Errors only on
/// claim-level storage failures; per-job failures stay in the queue and
/// are reflected in `report.failed`.
pub async fn drain_pending_embeddings(
    pool: &PgPool,
    embedder: &dyn Embedder,
    batch_size: i64,
) -> Result<DrainReport, DrainError> {
    drain_pending_embeddings_with_hygiene(pool, embedder, batch_size, false).await
}

/// Drain pending embeddings with optional Stage-7 ingest hygiene enforcement.
/// When enabled, retrieval-denied targets are dequeued without calling the
/// embedder, so they cannot become retrieval-bearing sidecars.
pub async fn drain_pending_embeddings_with_hygiene(
    pool: &PgPool,
    embedder: &dyn Embedder,
    batch_size: i64,
    hygiene_enabled: bool,
) -> Result<DrainReport, DrainError> {
    let jobs = kengram_storage::claim_pending(pool, batch_size).await?;
    let mut report = DrainReport {
        found: jobs.len(),
        ..Default::default()
    };

    for job in jobs {
        match process_embed_job(pool, embedder, &job, hygiene_enabled).await {
            Ok(EmbedJobOutcome::Embedded) => report.embedded += 1,
            Ok(EmbedJobOutcome::Skipped { reason }) => {
                tracing::info!(
                    pending_id = %job.id,
                    target_kind = %job.target_kind,
                    target_id = %job.target_id,
                    reason,
                    "embed-drain: skipped retrieval-denied target",
                );
                report.skipped += 1;
            }
            Err(err) => {
                tracing::warn!(
                    pending_id = %job.id,
                    target_kind = %job.target_kind,
                    target_id = %job.target_id,
                    attempts = job.attempts,
                    reason = %err,
                    "embed-drain: job failed; row stays queued",
                );
                let _ = kengram_storage::mark_failed(pool, job.id, &err.to_string()).await;
                report.failed += 1;
            }
        }
    }

    Ok(report)
}

#[derive(Debug)]
enum EmbedJobOutcome {
    Embedded,
    Skipped { reason: String },
}

#[derive(Debug)]
enum EmbedJobError {
    ModelMismatch { expected: String, got: String },
    UnsupportedTargetKind(String),
    SourceMissing,
    Embedder(EmbedderError),
    Embedding(EmbeddingError),
    Storage(kengram_storage::StorageError),
    EmptyEmbedderOutput,
}

impl std::fmt::Display for EmbedJobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelMismatch { expected, got } => write!(
                f,
                "job model_id={got} does not match active embedder model={expected}"
            ),
            Self::UnsupportedTargetKind(k) => write!(f, "unsupported target_kind: {k}"),
            Self::SourceMissing => f.write_str("embedding source no longer exists"),
            Self::Embedder(e) => write!(f, "embedder: {e}"),
            Self::Embedding(e) => write!(f, "embedding: {e}"),
            Self::Storage(e) => write!(f, "storage: {e}"),
            Self::EmptyEmbedderOutput => f.write_str("embedder returned no vectors"),
        }
    }
}

async fn process_embed_job(
    pool: &PgPool,
    embedder: &dyn Embedder,
    job: &kengram_storage::PendingJob,
    hygiene_enabled: bool,
) -> Result<EmbedJobOutcome, EmbedJobError> {
    let active_model = &embedder.model().id;
    if &job.model_id != active_model {
        return Err(EmbedJobError::ModelMismatch {
            expected: active_model.clone(),
            got: job.model_id.clone(),
        });
    }

    // Facts are gone; artifact chunks are now first-class embedding targets
    // for the recall chunk-serving path.
    let text = if hygiene_enabled {
        match kengram_storage::fetch_embedding_index_source(pool, &job.target_kind, job.target_id)
            .await
            .map_err(EmbedJobError::Storage)?
        {
            kengram_storage::EmbeddingIndexSourceStatus::Ready(source) => source.content,
            kengram_storage::EmbeddingIndexSourceStatus::Denied { reason } => {
                kengram_storage::mark_embedded(pool, job.id)
                    .await
                    .map_err(EmbedJobError::Storage)?;
                return Ok(EmbedJobOutcome::Skipped { reason });
            }
            kengram_storage::EmbeddingIndexSourceStatus::Missing => {
                kengram_storage::mark_embedded(pool, job.id)
                    .await
                    .map_err(EmbedJobError::Storage)?;
                return Ok(EmbedJobOutcome::Skipped {
                    reason: "source_missing".to_string(),
                });
            }
        }
    } else {
        match job.target_kind.as_str() {
            kengram_storage::target::THOUGHT => {
                let thought_id = ThoughtId::from(job.target_id);
                let thought = kengram_storage::fetch_thought(pool, thought_id)
                    .await
                    .map_err(EmbedJobError::Storage)?
                    .ok_or(EmbedJobError::SourceMissing)?;
                thought.content
            }
            kengram_storage::target::ARTIFACT_CHUNK => {
                kengram_storage::fetch_artifact_chunk_content(pool, job.target_id)
                    .await
                    .map_err(EmbedJobError::Storage)?
                    .ok_or(EmbedJobError::SourceMissing)?
            }
            _ => {
                return Err(EmbedJobError::UnsupportedTargetKind(
                    job.target_kind.clone(),
                ));
            }
        }
    };

    let texts = vec![text];
    let mut vectors = embedder
        .embed(&texts)
        .await
        .map_err(EmbedJobError::Embedder)?;
    let vector = vectors.pop().ok_or(EmbedJobError::EmptyEmbedderOutput)?;
    let embedding =
        Embedding::new(embedder.model().clone(), vector).map_err(EmbedJobError::Embedding)?;

    match job.target_kind.as_str() {
        kengram_storage::target::THOUGHT => {
            kengram_storage::insert_thought_embedding(
                pool,
                ThoughtId::from(job.target_id),
                &embedding,
            )
            .await
            .map_err(EmbedJobError::Storage)?;
        }
        kengram_storage::target::ARTIFACT_CHUNK => {
            kengram_storage::insert_artifact_chunk_embedding(pool, job.target_id, &embedding)
                .await
                .map_err(EmbedJobError::Storage)?;
        }
        _ => {
            return Err(EmbedJobError::UnsupportedTargetKind(
                job.target_kind.clone(),
            ));
        }
    }

    kengram_storage::mark_embedded(pool, job.id)
        .await
        .map_err(EmbedJobError::Storage)?;

    Ok(EmbedJobOutcome::Embedded)
}

// -- tag drainer ------------------------------------------------------------

/// Permanent-failure cap. After this many attempts on a single thought
/// (counting the initial enqueue as 0, so 5 attempts = 5 tagger calls
/// that all failed), the tag drainer logs and removes the job rather
/// than leaving it pinned in the queue forever.
pub const MAX_TAG_ATTEMPTS: i32 = 5;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrainTagsReport {
    /// Number of jobs fetched this drain call.
    pub processed: usize,
    /// Number that successfully tagged + persisted + completed.
    pub completed: usize,
    /// Number that hit a transient TaggerError; job stays in queue with
    /// attempts++.
    pub failed_transient: usize,
    /// Number that hit a non-transient TaggerError or hit
    /// `MAX_TAG_ATTEMPTS`; job is removed from queue.
    pub failed_permanent: usize,
}

/// Drain up to `batch_size` jobs from `pending_tags`.
///
/// `scope_vocab_limit`, when `Some(n)`, instructs the drainer to pre-fetch the
/// top-`n` established topic and entity terms for each thought's scope and
/// pass them to the tagger as controlled-vocabulary hints. `None` runs the
/// tagger without any vocab guidance (legacy behavior).
///
/// For each job:
/// 1. Fetch the thought (skip-with-permanent-fail if the thought no longer exists).
/// 2. When `scope_vocab_limit` is `Some`, fetch the scope's vocabulary. Vocab
///    fetch failure folds into transient-failure semantics so the next tick
///    retries.
/// 3. Call `tagger.tag(content, vocab)`.
/// 4. On Ok: `update_thought_tags` + `complete_tag_job`.
/// 5. On Err(transient): `increment_tag_job_attempts` (job stays).
/// 6. On Err(non-transient): log, `complete_tag_job` (job dropped).
/// 7. After `MAX_TAG_ATTEMPTS` regardless of transience, `complete_tag_job`.
pub async fn drain_pending_tags(
    pool: &PgPool,
    tagger: &dyn Tagger,
    batch_size: i64,
    scope_vocab_limit: Option<i64>,
) -> Result<DrainTagsReport, DrainError> {
    let jobs = kengram_storage::fetch_pending_tag_jobs(pool, batch_size).await?;
    let mut report = DrainTagsReport {
        processed: jobs.len(),
        ..Default::default()
    };

    // Corpus scope set, fetched once per batch (low cardinality, cheap) so the
    // deterministic scope-identifier filter runs per job without a per-job
    // query. A failure degrades to empty — the filter then strips only each
    // thought's own scope.
    let known_scopes = match kengram_storage::list_scopes(pool, None).await {
        Ok(s) => s
            .into_iter()
            .map(|x| x.scope.as_str().to_string())
            .collect::<Vec<_>>(),
        Err(e) => {
            tracing::debug!(error = %e, "tag-drain: list_scopes failed; scope-id filter uses own-scope only");
            Vec::new()
        }
    };

    for job in jobs {
        match process_tag_job(pool, tagger, scope_vocab_limit, &job, &known_scopes).await {
            TagJobOutcome::Completed => report.completed += 1,
            TagJobOutcome::Transient => report.failed_transient += 1,
            TagJobOutcome::Permanent => report.failed_permanent += 1,
        }
    }

    Ok(report)
}

enum TagJobOutcome {
    Completed,
    Transient,
    Permanent,
}

async fn process_tag_job(
    pool: &PgPool,
    tagger: &dyn Tagger,
    scope_vocab_limit: Option<i64>,
    job: &kengram_storage::PendingTagJob,
    known_scopes: &[String],
) -> TagJobOutcome {
    // Fetch the thought's content.
    let thought = match kengram_storage::fetch_thought(pool, job.thought_id).await {
        Ok(Some(t)) => t,
        Ok(None) => {
            tracing::warn!(
                thought_id = %job.thought_id,
                "tag-drain: thought no longer exists; dropping job",
            );
            let _ =
                kengram_storage::complete_tag_job(pool, job.thought_id, job.tag_job_generation_id)
                    .await;
            return TagJobOutcome::Permanent;
        }
        Err(e) => {
            tracing::warn!(
                thought_id = %job.thought_id,
                error = %e,
                "tag-drain: storage error fetching thought; leaving job for retry",
            );
            let _ = kengram_storage::increment_tag_job_attempts(
                pool,
                job.thought_id,
                job.tag_job_generation_id,
            )
            .await;
            return TagJobOutcome::Transient;
        }
    };

    // Optionally fetch controlled-vocabulary hints for the thought's scope.
    // A storage failure here is transient — leaves the job queued for retry.
    let vocab = match scope_vocab_limit {
        Some(limit) if limit > 0 => {
            match kengram_storage::fetch_scope_vocab(pool, thought.scope.as_str(), limit).await {
                Ok(v) if v.is_empty() => None,
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!(
                        thought_id = %job.thought_id,
                        scope = %thought.scope.as_str(),
                        error = %e,
                        "tag-drain: scope vocab fetch failed; leaving job for retry",
                    );
                    let _ = kengram_storage::increment_tag_job_attempts(
                        pool,
                        job.thought_id,
                        job.tag_job_generation_id,
                    )
                    .await;
                    return TagJobOutcome::Transient;
                }
            }
        }
        _ => None,
    };

    match tagger.tag(&thought.content, vocab.as_ref()).await {
        Ok(mut output) => {
            // `output` is a TagOutput { tags, relations }. The Tags portion
            // is persisted to thoughts.tags JSONB; the relations portion is
            // routed to thought_links via apply_tagger_relations. They are
            // NOT mirrored — thought_links is the canonical store for the
            // link graph; tags.relations is no longer persisted (migration
            // 0011 dropped the field from existing rows).

            // Deterministic post-tag pipeline: v11 topic normalization
            // (vocab-gated) + v12 people/entities disjointness. Extracted into
            // `finalize` so the one-shot `kengram tag` path runs the identical
            // steps instead of skipping them.
            finalize::finalize_tags(
                &mut output.tags,
                &thought.metadata,
                &thought.scope,
                vocab.as_ref(),
                known_scopes,
            );
            if let Err(e) = kengram_storage::update_thought_tags(
                pool,
                job.thought_id,
                &output.tags,
                tagger.model_id(),
                tagger.version(),
            )
            .await
            {
                tracing::warn!(
                    thought_id = %job.thought_id,
                    error = %e,
                    "tag-drain: failed to persist tags; leaving job for retry",
                );
                let _ = kengram_storage::increment_tag_job_attempts(
                    pool,
                    job.thought_id,
                    job.tag_job_generation_id,
                )
                .await;
                return TagJobOutcome::Transient;
            }
            // Emit tagger-extracted relations (M6.1). Soft-delete prior
            // tagger edges first so re-tag cycles produce a clean replacement;
            // agent-supplied edges are untouched.
            if let Err(e) = apply_tagger_relations(
                pool,
                job.thought_id,
                job.tag_job_generation_id,
                tagger.model_id(),
                tagger.version(),
                &output.relations,
            )
            .await
            {
                tracing::warn!(
                    thought_id = %job.thought_id,
                    generation_id = %job.tag_job_generation_id,
                    error = %e,
                    "tag-drain: atomic relation replacement failed; leaving job for retry",
                );
                let _ = kengram_storage::increment_tag_job_attempts(
                    pool,
                    job.thought_id,
                    job.tag_job_generation_id,
                )
                .await;
                return TagJobOutcome::Transient;
            }
            if let Err(e) =
                kengram_storage::complete_tag_job(pool, job.thought_id, job.tag_job_generation_id)
                    .await
            {
                tracing::warn!(
                    thought_id = %job.thought_id,
                    error = %e,
                    "tag-drain: tags persisted but failed to dequeue; next tick re-runs idempotently",
                );
                // Not a failure-of-tagging — tags were written. The next
                // tick will re-tag (idempotent overwrite). Report as
                // completed.
            }
            TagJobOutcome::Completed
        }
        Err(err) => {
            let transient = err.is_transient();
            let attempts_after = job.attempts.saturating_add(1);
            tracing::warn!(
                thought_id = %job.thought_id,
                error = %err,
                attempts = attempts_after,
                transient,
                "tag-drain: tagger error",
            );

            if !transient || attempts_after >= MAX_TAG_ATTEMPTS {
                // Permanent failure or exhausted attempts — drop the job.
                let _ = kengram_storage::complete_tag_job(
                    pool,
                    job.thought_id,
                    job.tag_job_generation_id,
                )
                .await;
                TagJobOutcome::Permanent
            } else {
                let _ = kengram_storage::increment_tag_job_attempts(
                    pool,
                    job.thought_id,
                    job.tag_job_generation_id,
                )
                .await;
                TagJobOutcome::Transient
            }
        }
    }
}

/// Apply tagger-extracted relations to `thought_links` for a single
/// thought. Soft-deletes prior tagger-emitted edges from this thought
/// first (so a re-tag cycle replaces them cleanly without accumulation),
/// then inserts each emitted relation with `source = 'tagger'`.
///
/// The emitted set is validated before the serialized replacement call.
/// Any malformed emission fails the whole set and leaves the job queued so
/// a retry can never publish a partial generation.
pub async fn apply_tagger_relations(
    pool: &PgPool,
    from_thought_id: ThoughtId,
    generation_id: uuid::Uuid,
    tagger_model_id: &str,
    tagger_version: i32,
    relations: &[ExtractedRelation],
) -> Result<serde_json::Value, kengram_storage::StorageError> {
    let mut emitted = Vec::with_capacity(relations.len());
    for rel in relations {
        let target = rel.target.clone().into_link_target();
        crate::link::validate_target(&target).map_err(|error| {
            kengram_storage::StorageError::Database(sqlx::Error::Protocol(format!(
                "invalid tagger relation target: {error}"
            )))
        })?;
        emitted.push(serde_json::json!({
            "relation": rel.relation.as_str(),
            "to_kind": target.kind_str(),
            "to_value": target.value_str(),
            "note": rel.note,
        }));
    }
    emitted.sort_by_key(|value| value.to_string());
    let canonical = serde_json::to_vec(&emitted).map_err(|error| {
        kengram_storage::StorageError::Database(sqlx::Error::Protocol(error.to_string()))
    })?;
    let emitted_hash = Sha256::digest(&canonical)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let operations = serde_json::json!([{
        "action": "replace_tagger_set",
        "from_thought_id": from_thought_id.to_string(),
        "relations": emitted,
    }]);
    let metadata = serde_json::json!({
        "tagger_model_id": tagger_model_id,
        "tagger_version": tagger_version,
        "emitted_set_sha256": emitted_hash,
    });
    let source_ref = format!("{}:{}", from_thought_id, generation_id);
    let result = kengram_storage::corpus_hygiene::mutate_thought_relations_serialized(
        pool,
        kengram_storage::corpus_hygiene::RelationMutationRequest {
            operations: &operations,
            source_event_namespace: "kengram/tagger-relations",
            source_event_ref: &source_ref,
            source_event_payload_hash: &emitted_hash,
            request_metadata: &metadata,
            claimed_producer_class: None,
        },
    )
    .await?;
    if result.get("status").and_then(|v| v.as_str()) != Some("completed") {
        return Err(kengram_storage::StorageError::Database(
            sqlx::Error::Protocol(format!("tagger relation request failed: {result}")),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use kengram_core::{
        EmbeddingModel, ExtractedTarget, LinkSource, LinkTarget, RelationKind, Scope, Source,
        TagKind, TagOutput, Tags,
    };
    use kengram_embed::{FakeBehavior, FakeEmbedder};
    use kengram_extract::{FakeBehavior as TaggerFakeBehavior, FakeTagger};

    const TEST_EMBEDDER_MODEL_ID: &str = "bge-m3:1024";

    fn test_embedding_model() -> EmbeddingModel {
        EmbeddingModel::new(TEST_EMBEDDER_MODEL_ID, 1024)
    }

    fn test_embedder() -> FakeEmbedder {
        FakeEmbedder::with_model(test_embedding_model())
    }

    async fn capture_one(pool: &PgPool, content: &str) -> ThoughtId {
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

    // -- embed drainer (preserved from M3 with fact branch deleted) ----------

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_processes_pending_to_embedding(pool: PgPool) {
        let id = capture_one(&pool, "drain me").await;
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);

        let good = test_embedder();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 1);
        assert_eq!(report.failed, 0);

        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);
        assert!(
            kengram_storage::thought_has_embedding(&pool, id, &test_embedding_model())
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_marks_failed_and_leaves_row_on_embedder_error(pool: PgPool) {
        let _id = capture_one(&pool, "stays queued").await;

        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let report = drain_pending_embeddings(&pool, &bad, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 1);

        let row = sqlx::query!(r#"SELECT attempts, last_error FROM pending_embeddings"#,)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(row.attempts, 1);
        assert!(row.last_error.is_some());
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_idempotent_on_crash_replay(pool: PgPool) {
        let id = capture_one(&pool, "replay").await;
        let model = test_embedding_model();

        let job = kengram_storage::claim_pending(&pool, 10)
            .await
            .unwrap()
            .pop()
            .unwrap();
        let emb = Embedding::new(model.clone(), vec![0.5_f32; 1024]).unwrap();
        kengram_storage::insert_thought_embedding(&pool, id, &emb)
            .await
            .unwrap();

        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
        let _ = job;

        let good = test_embedder();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 1, "replay tick must mark embedded cleanly");
        assert_eq!(report.failed, 0);
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_marks_failed_when_model_id_mismatch(pool: PgPool) {
        let _id = capture_one(&pool, "mismatched model").await;

        let other = FakeEmbedder::with_model(EmbeddingModel::new("other:1024", 1024));
        let report = drain_pending_embeddings(&pool, &other, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 1);

        let row = sqlx::query!(r#"SELECT last_error FROM pending_embeddings"#,)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(
            row.last_error
                .as_deref()
                .map(|e| e.contains("model_id") || e.contains("model="))
                .unwrap_or(false),
            "expected model-mismatch message, got {:?}",
            row.last_error
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_empty_queue_is_a_noop(pool: PgPool) {
        let good = test_embedder();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 0);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.failed, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_hygiene_skips_kgr_target_before_embedding(pool: PgPool) {
        let id = capture_one(&pool, "KGR024 should never be indexed").await;
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);

        let good = test_embedder();
        let report = drain_pending_embeddings_with_hygiene(&pool, &good, 10, true)
            .await
            .unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.failed, 0);
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);
        assert!(
            !kengram_storage::thought_has_embedding(&pool, id, &test_embedding_model())
                .await
                .unwrap(),
            "hygiene skip must not write an embedding sidecar"
        );
    }

    // -- tag drainer ----------------------------------------------------------

    /// Capture and enqueue a tag job for the captured thought.
    async fn capture_and_enqueue_tag(pool: &PgPool, content: &str) -> ThoughtId {
        capture(
            pool,
            TEST_EMBEDDER_MODEL_ID,
            Some("fake/tagger"),
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
    async fn drain_tags_updates_thought_tags_on_success(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "captured thought needing tags").await;

        let tags = Tags {
            people: vec!["Sarah".into()],
            kind: Some(TagKind::Task),
            ..Tags::default()
        };
        let tagger = FakeTagger::with_canned(tags.clone());

        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.processed, 1);
        assert_eq!(report.completed, 1);
        assert_eq!(report.failed_transient, 0);
        assert_eq!(report.failed_permanent, 0);

        // Tags persisted; queue drained.
        let read = kengram_storage::fetch_thought_tags(&pool, id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read.tags, tags);
        assert_eq!(read.tagger_model_id.as_deref(), Some("fake/tagger"));
        assert_eq!(read.tagger_version, Some(1));
        let remaining = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(remaining.is_empty());

        // Tagger was called with the thought's content.
        let recorded = tagger.last_call().unwrap();
        assert_eq!(recorded.content, "captured thought needing tags");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_increments_attempts_on_transient_failure(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "transient-fail content").await;

        let tagger = FakeTagger::always_failing(TaggerFakeBehavior::Timeout);
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.processed, 1);
        assert_eq!(report.completed, 0);
        assert_eq!(report.failed_transient, 1);
        assert_eq!(report.failed_permanent, 0);

        // Job stays in queue with attempts bumped.
        let jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].thought_id, id);
        assert_eq!(jobs[0].attempts, 1);

        // Thought still has no tags.
        let read = kengram_storage::fetch_thought_tags(&pool, id)
            .await
            .unwrap()
            .unwrap();
        assert!(read.tagger_model_id.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_drops_job_after_max_attempts(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "exhaust attempts").await;
        let generation_id = kengram_storage::fetch_pending_tag_jobs(&pool, 1)
            .await
            .unwrap()[0]
            .tag_job_generation_id;

        // Hand-poke attempts to MAX-1 so a single transient failure trips
        // the drop threshold.
        for _ in 0..(MAX_TAG_ATTEMPTS - 1) {
            kengram_storage::increment_tag_job_attempts(&pool, id, generation_id)
                .await
                .unwrap();
        }

        let tagger = FakeTagger::always_failing(TaggerFakeBehavior::Timeout);
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.processed, 1);
        assert_eq!(report.failed_permanent, 1);

        // Job dropped from queue.
        let jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(jobs.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_drops_job_on_permanent_failure(pool: PgPool) {
        let _id = capture_and_enqueue_tag(&pool, "misconfigured tagger").await;

        let tagger = FakeTagger::always_failing(TaggerFakeBehavior::Misconfigured);
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.processed, 1);
        assert_eq!(report.failed_permanent, 1);

        // Misconfigured is non-transient → drop on first failure.
        let jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(jobs.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_empty_queue_is_a_noop(pool: PgPool) {
        let tagger = FakeTagger::new();
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.processed, 0);
        assert_eq!(report.completed, 0);
    }

    // -- M4.1: scope-vocab injection -------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_passes_scope_vocab_when_limit_some(pool: PgPool) {
        // Seed an earlier tagged thought in the same scope so vocab has terms.
        let prior = capture_and_enqueue_tag(&pool, "prior context").await;
        kengram_storage::update_thought_tags(
            &pool,
            prior,
            &Tags {
                topics: vec!["memory-systems".into()],
                entities: vec!["kengram".into()],
                ..Tags::default()
            },
            "fake/tagger",
            2,
        )
        .await
        .unwrap();
        let generation_id = kengram_storage::fetch_pending_tag_jobs(&pool, 1)
            .await
            .unwrap()
            .into_iter()
            .find(|job| job.thought_id == prior)
            .unwrap()
            .tag_job_generation_id;
        kengram_storage::complete_tag_job(&pool, prior, generation_id)
            .await
            .unwrap();

        // Enqueue a fresh thought to be tagged.
        let _id = capture_and_enqueue_tag(&pool, "fresh thought needing vocab").await;

        let tagger = FakeTagger::new();
        let report = drain_pending_tags(&pool, &tagger, 10, Some(50))
            .await
            .unwrap();
        assert_eq!(report.completed, 1);

        let rec = tagger.last_call().expect("tag call recorded");
        let vocab = rec.vocab.expect("vocab must be supplied when limit > 0");
        assert!(vocab.topics.contains(&"memory-systems".to_string()));
        assert!(vocab.entities.contains(&"kengram".to_string()));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_omits_vocab_when_limit_none(pool: PgPool) {
        // Same setup as above but pass `None` as the vocab limit — vocab
        // must NOT be supplied to the tagger.
        let prior = capture_and_enqueue_tag(&pool, "prior context").await;
        kengram_storage::update_thought_tags(
            &pool,
            prior,
            &Tags {
                topics: vec!["memory-systems".into()],
                ..Tags::default()
            },
            "fake/tagger",
            2,
        )
        .await
        .unwrap();
        let generation_id = kengram_storage::fetch_pending_tag_jobs(&pool, 1)
            .await
            .unwrap()
            .into_iter()
            .find(|job| job.thought_id == prior)
            .unwrap()
            .tag_job_generation_id;
        kengram_storage::complete_tag_job(&pool, prior, generation_id)
            .await
            .unwrap();

        let _id = capture_and_enqueue_tag(&pool, "fresh thought").await;

        let tagger = FakeTagger::new();
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.completed, 1);

        let rec = tagger.last_call().expect("tag call recorded");
        assert!(rec.vocab.is_none(), "vocab must be None when limit is None");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_omits_vocab_when_scope_has_no_history(pool: PgPool) {
        // Limit is Some, but no prior tagged thoughts in the scope — vocab
        // resolves to empty and the drainer should pass None to the tagger
        // (avoids sending an empty controlled-vocabulary section).
        let _id = capture_and_enqueue_tag(&pool, "first-ever thought in this scope").await;

        let tagger = FakeTagger::new();
        let report = drain_pending_tags(&pool, &tagger, 10, Some(50))
            .await
            .unwrap();
        assert_eq!(report.completed, 1);

        let rec = tagger.last_call().expect("tag call recorded");
        assert!(
            rec.vocab.is_none(),
            "empty vocab should be normalized to None"
        );
    }

    // -- M6.1: tagger-extracted relations ---------------------------------

    fn tag_output_with_relations(rels: Vec<ExtractedRelation>) -> TagOutput {
        TagOutput {
            tags: Tags {
                kind: Some(TagKind::Reference),
                ..Tags::default()
            },
            relations: rels,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_inserts_emitted_relations_with_source_tagger(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "thought citing https://example.com").await;
        let canned = tag_output_with_relations(vec![
            ExtractedRelation {
                relation: RelationKind::References,
                target: ExtractedTarget::Url("https://example.com".into()),
                note: Some("explicit citation".into()),
            },
            ExtractedRelation {
                relation: RelationKind::BelongsTo,
                target: ExtractedTarget::Entity("Probe 2".into()),
                note: None,
            },
        ]);
        let tagger = FakeTagger::with_canned_output(canned);
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.completed, 1);

        let related = kengram_storage::fetch_related_thoughts(
            &pool,
            id,
            None,
            None,
            kengram_core::LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert_eq!(related.len(), 2);
        // All inserted edges have source = Tagger.
        for r in &related {
            assert_eq!(r.link_source, LinkSource::Tagger);
        }
        let kinds: Vec<&'static str> = related.iter().map(|r| r.target.kind_str()).collect();
        assert!(kinds.contains(&"url"));
        assert!(kinds.contains(&"entity"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_re_run_soft_deletes_prior_tagger_edges_then_inserts_fresh(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "first pass").await;

        // First drain: emit one URL relation.
        let first = tag_output_with_relations(vec![ExtractedRelation {
            relation: RelationKind::References,
            target: ExtractedTarget::Url("https://old.example".into()),
            note: None,
        }]);
        let tagger = FakeTagger::with_canned_output(first);
        drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();

        let after_first = kengram_storage::fetch_related_thoughts(
            &pool,
            id,
            None,
            None,
            kengram_core::LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert_eq!(after_first.len(), 1);
        assert_eq!(
            after_first[0].target,
            LinkTarget::Url("https://old.example".into())
        );

        // Re-enqueue and re-drain with a different emission.
        kengram_storage::enqueue_tag_job(&pool, id, "fake/tagger")
            .await
            .unwrap();
        let second = tag_output_with_relations(vec![ExtractedRelation {
            relation: RelationKind::References,
            target: ExtractedTarget::Url("https://new.example".into()),
            note: None,
        }]);
        let tagger = FakeTagger::with_canned_output(second);
        drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();

        let after_second = kengram_storage::fetch_related_thoughts(
            &pool,
            id,
            None,
            None,
            kengram_core::LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert_eq!(
            after_second.len(),
            1,
            "old edge soft-deleted, new edge inserted; fetch excludes soft-deleted"
        );
        assert_eq!(
            after_second[0].target,
            LinkTarget::Url("https://new.example".into())
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_preserves_agent_edges_during_retag(pool: PgPool) {
        let a = capture_and_enqueue_tag(&pool, "thought A").await;
        let b = capture_and_enqueue_tag(&pool, "thought B").await;

        // Agent-supplied edge from a to b.
        kengram_storage::insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &LinkTarget::Thought(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        // Tagger-supplied URL edge from a, then drain re-runs with different relations.
        let canned = tag_output_with_relations(vec![ExtractedRelation {
            relation: RelationKind::References,
            target: ExtractedTarget::Url("https://new.example".into()),
            note: None,
        }]);
        let tagger = FakeTagger::with_canned_output(canned);
        drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();

        let related = kengram_storage::fetch_related_thoughts(
            &pool,
            a,
            None,
            None,
            kengram_core::LinkDirection::Outbound,
        )
        .await
        .unwrap();
        // The agent edge survives the tag drain.
        assert!(related.iter().any(|r| r.link_source == LinkSource::Agent));
        // And the tagger edge is present.
        assert!(related.iter().any(|r| r.link_source == LinkSource::Tagger));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_skips_invalid_target_continues_others(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "thought with mixed-validity relations").await;
        // First emission has a non-http URL. The generation must remain
        // atomic: the valid second emission cannot land by itself.
        let canned = tag_output_with_relations(vec![
            ExtractedRelation {
                relation: RelationKind::References,
                target: ExtractedTarget::Url("ftp://bad.example".into()),
                note: None,
            },
            ExtractedRelation {
                relation: RelationKind::References,
                target: ExtractedTarget::Url("https://good.example".into()),
                note: None,
            },
        ]);
        let tagger = FakeTagger::with_canned_output(canned);
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.completed, 0);
        assert_eq!(report.failed_transient, 1);
        assert_eq!(report.failed_permanent, 0);

        let related = kengram_storage::fetch_related_thoughts(
            &pool,
            id,
            None,
            None,
            kengram_core::LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert!(related.is_empty());

        let jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].thought_id, id);
        assert_eq!(jobs[0].attempts, 1);
    }
}

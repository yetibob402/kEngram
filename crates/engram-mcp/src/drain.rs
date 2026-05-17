//! Drainers for the two background queues — `pending_embeddings` (embed
//! drainer) and `pending_tags` (tag drainer).
//!
//! Both are pulled on every `engram worker` tick. Each tick processes a
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

use engram_core::{Embedder, EmbedderError, Embedding, EmbeddingError, Tagger, ThoughtId};
use sqlx::PgPool;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrainReport {
    /// Number of jobs claimed this drain call.
    pub found: usize,
    /// Number that successfully embedded + persisted + marked.
    pub embedded: usize,
    /// Number that failed embed/persist.
    pub failed: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DrainError {
    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Drain up to `batch_size` jobs from `pending_embeddings`. Errors only on
/// claim-level storage failures; per-job failures stay in the queue and
/// are reflected in `report.failed`.
pub async fn drain_pending_embeddings(
    pool: &PgPool,
    embedder: &dyn Embedder,
    batch_size: i64,
) -> Result<DrainReport, DrainError> {
    let jobs = engram_storage::claim_pending(pool, batch_size).await?;
    let mut report = DrainReport {
        found: jobs.len(),
        ..Default::default()
    };

    for job in jobs {
        match process_embed_job(pool, embedder, &job).await {
            Ok(()) => report.embedded += 1,
            Err(err) => {
                tracing::warn!(
                    pending_id = %job.id,
                    target_kind = %job.target_kind,
                    target_id = %job.target_id,
                    attempts = job.attempts,
                    reason = %err,
                    "embed-drain: job failed; row stays queued",
                );
                let _ = engram_storage::mark_failed(pool, job.id, &err.to_string()).await;
                report.failed += 1;
            }
        }
    }

    Ok(report)
}

#[derive(Debug)]
enum EmbedJobError {
    ModelMismatch { expected: String, got: String },
    UnsupportedTargetKind(String),
    SourceMissing,
    Embedder(EmbedderError),
    Embedding(EmbeddingError),
    Storage(engram_storage::StorageError),
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
            Self::SourceMissing => f.write_str("source thought no longer exists"),
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
    job: &engram_storage::PendingJob,
) -> Result<(), EmbedJobError> {
    let active_model = &embedder.model().id;
    if &job.model_id != active_model {
        return Err(EmbedJobError::ModelMismatch {
            expected: active_model.clone(),
            got: job.model_id.clone(),
        });
    }

    // M4: only thoughts are embeddable; facts are gone.
    let text = match job.target_kind.as_str() {
        engram_storage::target::THOUGHT => {
            let thought_id = ThoughtId::from(job.target_id);
            let thought = engram_storage::fetch_thought(pool, thought_id)
                .await
                .map_err(EmbedJobError::Storage)?
                .ok_or(EmbedJobError::SourceMissing)?;
            thought.content
        }
        _ => {
            return Err(EmbedJobError::UnsupportedTargetKind(
                job.target_kind.clone(),
            ));
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

    engram_storage::insert_thought_embedding(pool, ThoughtId::from(job.target_id), &embedding)
        .await
        .map_err(EmbedJobError::Storage)?;

    engram_storage::mark_embedded(pool, job.id)
        .await
        .map_err(EmbedJobError::Storage)?;

    Ok(())
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
    let jobs = engram_storage::fetch_pending_tag_jobs(pool, batch_size).await?;
    let mut report = DrainTagsReport {
        processed: jobs.len(),
        ..Default::default()
    };

    for job in jobs {
        match process_tag_job(pool, tagger, scope_vocab_limit, &job).await {
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
    job: &engram_storage::PendingTagJob,
) -> TagJobOutcome {
    // Fetch the thought's content.
    let thought = match engram_storage::fetch_thought(pool, job.thought_id).await {
        Ok(Some(t)) => t,
        Ok(None) => {
            tracing::warn!(
                thought_id = %job.thought_id,
                "tag-drain: thought no longer exists; dropping job",
            );
            let _ = engram_storage::complete_tag_job(pool, job.thought_id).await;
            return TagJobOutcome::Permanent;
        }
        Err(e) => {
            tracing::warn!(
                thought_id = %job.thought_id,
                error = %e,
                "tag-drain: storage error fetching thought; leaving job for retry",
            );
            let _ = engram_storage::increment_tag_job_attempts(pool, job.thought_id).await;
            return TagJobOutcome::Transient;
        }
    };

    // Optionally fetch controlled-vocabulary hints for the thought's scope.
    // A storage failure here is transient — leaves the job queued for retry.
    let vocab = match scope_vocab_limit {
        Some(limit) if limit > 0 => {
            match engram_storage::fetch_scope_vocab(pool, thought.scope.as_str(), limit).await {
                Ok(v) if v.is_empty() => None,
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!(
                        thought_id = %job.thought_id,
                        scope = %thought.scope.as_str(),
                        error = %e,
                        "tag-drain: scope vocab fetch failed; leaving job for retry",
                    );
                    let _ = engram_storage::increment_tag_job_attempts(pool, job.thought_id).await;
                    return TagJobOutcome::Transient;
                }
            }
        }
        _ => None,
    };

    match tagger.tag(&thought.content, vocab.as_ref()).await {
        Ok(tags) => {
            if let Err(e) = engram_storage::update_thought_tags(
                pool,
                job.thought_id,
                &tags,
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
                let _ = engram_storage::increment_tag_job_attempts(pool, job.thought_id).await;
                return TagJobOutcome::Transient;
            }
            if let Err(e) = engram_storage::complete_tag_job(pool, job.thought_id).await {
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
                let _ = engram_storage::complete_tag_job(pool, job.thought_id).await;
                TagJobOutcome::Permanent
            } else {
                let _ = engram_storage::increment_tag_job_attempts(pool, job.thought_id).await;
                TagJobOutcome::Transient
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use engram_core::{EmbeddingModel, Scope, Source, TagKind, Tags};
    use engram_embed::{FakeBehavior, FakeEmbedder};
    use engram_extract::{FakeBehavior as TaggerFakeBehavior, FakeTagger};

    const TEST_EMBEDDER_MODEL_ID: &str = "bge-m3:1024";

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
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);

        let good = FakeEmbedder::new();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 1);
        assert_eq!(report.failed, 0);

        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);
        assert!(
            engram_storage::thought_has_embedding(&pool, id, &EmbeddingModel::bge_m3())
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_marks_failed_and_leaves_row_on_embedder_error(pool: PgPool) {
        let _id = capture_one(&pool, "stays queued").await;

        let bad = FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Unreachable);
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
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_idempotent_on_crash_replay(pool: PgPool) {
        let id = capture_one(&pool, "replay").await;
        let model = EmbeddingModel::bge_m3();

        let job = engram_storage::claim_pending(&pool, 10)
            .await
            .unwrap()
            .pop()
            .unwrap();
        let emb = Embedding::new(model.clone(), vec![0.5_f32; 1024]).unwrap();
        engram_storage::insert_thought_embedding(&pool, id, &emb)
            .await
            .unwrap();

        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);
        let _ = job;

        let good = FakeEmbedder::new();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 1, "replay tick must mark embedded cleanly");
        assert_eq!(report.failed, 0);
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);
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
        let good = FakeEmbedder::new();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 0);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 0);
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
        let read = engram_storage::fetch_thought_tags(&pool, id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read.tags, tags);
        assert_eq!(read.tagger_model_id.as_deref(), Some("fake/tagger"));
        assert_eq!(read.tagger_version, Some(1));
        let remaining = engram_storage::fetch_pending_tag_jobs(&pool, 10)
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
        let jobs = engram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].thought_id, id);
        assert_eq!(jobs[0].attempts, 1);

        // Thought still has no tags.
        let read = engram_storage::fetch_thought_tags(&pool, id)
            .await
            .unwrap()
            .unwrap();
        assert!(read.tagger_model_id.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_drops_job_after_max_attempts(pool: PgPool) {
        let id = capture_and_enqueue_tag(&pool, "exhaust attempts").await;

        // Hand-poke attempts to MAX-1 so a single transient failure trips
        // the drop threshold.
        for _ in 0..(MAX_TAG_ATTEMPTS - 1) {
            engram_storage::increment_tag_job_attempts(&pool, id)
                .await
                .unwrap();
        }

        let tagger = FakeTagger::always_failing(TaggerFakeBehavior::Timeout);
        let report = drain_pending_tags(&pool, &tagger, 10, None).await.unwrap();
        assert_eq!(report.processed, 1);
        assert_eq!(report.failed_permanent, 1);

        // Job dropped from queue.
        let jobs = engram_storage::fetch_pending_tag_jobs(&pool, 10)
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
        let jobs = engram_storage::fetch_pending_tag_jobs(&pool, 10)
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
        engram_storage::update_thought_tags(
            &pool,
            prior,
            &Tags {
                topics: vec!["memory-systems".into()],
                entities: vec!["engram".into()],
                ..Tags::default()
            },
            "fake/tagger",
            2,
        )
        .await
        .unwrap();
        engram_storage::complete_tag_job(&pool, prior)
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
        assert!(vocab.entities.contains(&"engram".to_string()));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_tags_omits_vocab_when_limit_none(pool: PgPool) {
        // Same setup as above but pass `None` as the vocab limit — vocab
        // must NOT be supplied to the tagger.
        let prior = capture_and_enqueue_tag(&pool, "prior context").await;
        engram_storage::update_thought_tags(
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
        engram_storage::complete_tag_job(&pool, prior)
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
}

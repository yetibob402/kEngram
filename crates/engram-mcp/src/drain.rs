//! `drain_pending_embeddings` — pull a batch of claimed jobs off the
//! `pending_embeddings` queue, embed each one, and persist the result. The
//! worker process (`engram worker`) calls this on every tick; the
//! `engram embed-backfill` CLI calls it in a loop after healing any gaps.
//!
//! Crash-replay safety: `insert_thought_embedding` is idempotent via
//! `ON CONFLICT DO NOTHING` (migration 0001 has the UNIQUE constraint;
//! `engram-storage` was updated to use it during M2 Phase B). So if the
//! worker dies between `insert_thought_embedding` and `mark_embedded`, the
//! next tick re-claims the row, re-embeds, re-inserts (no-op), and marks
//! embedded — clean.

use engram_core::{Embedder, Embedding, EmbedderError, EmbeddingError, ThoughtId};
use sqlx::PgPool;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrainReport {
    /// Number of jobs claimed this drain call.
    pub found: usize,
    /// Number that successfully embedded + persisted + marked.
    pub embedded: usize,
    /// Number that failed embed/persist. Each is logged with thought_id +
    /// reason and left in the queue (with `last_error` set) for the next tick.
    pub failed: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DrainError {
    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Drain up to `batch_size` jobs. Returns a `DrainReport`. Errors only on
/// claim-level storage failures (the queue itself is unreachable); per-job
/// failures stay in the queue and are reflected in `report.failed`.
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
        match process_job(pool, embedder, &job).await {
            Ok(()) => report.embedded += 1,
            Err(err) => {
                tracing::warn!(
                    pending_id = %job.id,
                    target_kind = %job.target_kind,
                    target_id = %job.target_id,
                    attempts = job.attempts,
                    reason = %err,
                    "drain: job failed; row stays queued",
                );
                let _ = engram_storage::mark_failed(pool, job.id, &err.to_string()).await;
                report.failed += 1;
            }
        }
    }

    Ok(report)
}

#[derive(Debug)]
enum JobError {
    /// The job's `model_id` doesn't match the active embedder. Multi-model
    /// support isn't in Phase B; this is a guardrail for the future.
    ModelMismatch { expected: String, got: String },
    /// The job targets something other than `thought`. Phase B only handles
    /// thoughts; M4 will extend to artifact chunks.
    UnsupportedTargetKind(String),
    /// The source thought was deleted (or never existed) between enqueue
    /// and drain. The job is unprocessable.
    SourceMissing,
    Embedder(EmbedderError),
    Embedding(EmbeddingError),
    Storage(engram_storage::StorageError),
    EmptyEmbedderOutput,
}

impl std::fmt::Display for JobError {
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

async fn process_job(
    pool: &PgPool,
    embedder: &dyn Embedder,
    job: &engram_storage::PendingJob,
) -> Result<(), JobError> {
    let active_model = &embedder.model().id;
    if &job.model_id != active_model {
        return Err(JobError::ModelMismatch {
            expected: active_model.clone(),
            got: job.model_id.clone(),
        });
    }

    // Resolve the text to embed and the insert path by `target_kind`. Both
    // branches share the embed → insert → mark_embedded shape; the only
    // difference is which DB row provides the input text and which
    // `insert_*_embedding` wrapper receives the vector.
    let text = match job.target_kind.as_str() {
        engram_storage::target::THOUGHT => {
            let thought_id = ThoughtId::from(job.target_id);
            let thought = engram_storage::fetch_thought(pool, thought_id)
                .await
                .map_err(JobError::Storage)?
                .ok_or(JobError::SourceMissing)?;
            thought.content
        }
        engram_storage::target::FACT => {
            let fact = engram_storage::fetch_fact(pool, job.target_id)
                .await
                .map_err(JobError::Storage)?
                .ok_or(JobError::SourceMissing)?;
            fact.statement
        }
        _ => return Err(JobError::UnsupportedTargetKind(job.target_kind.clone())),
    };

    let texts = vec![text];
    let mut vectors = embedder.embed(&texts).await.map_err(JobError::Embedder)?;
    let vector = vectors.pop().ok_or(JobError::EmptyEmbedderOutput)?;
    let embedding = Embedding::new(embedder.model().clone(), vector).map_err(JobError::Embedding)?;

    match job.target_kind.as_str() {
        engram_storage::target::THOUGHT => {
            engram_storage::insert_thought_embedding(
                pool,
                ThoughtId::from(job.target_id),
                &embedding,
            )
            .await
            .map_err(JobError::Storage)?;
        }
        engram_storage::target::FACT => {
            engram_storage::insert_fact_embedding(pool, job.target_id, &embedding)
                .await
                .map_err(JobError::Storage)?;
        }
        _ => unreachable!("target_kind validated above"),
    }

    engram_storage::mark_embedded(pool, job.id)
        .await
        .map_err(JobError::Storage)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{capture, CaptureRequest};
    use engram_core::{EmbeddingModel, Scope, Source};
    use engram_embed::{FakeBehavior, FakeEmbedder};

    const TEST_MODEL_ID: &str = "bge-m3:1024";

    async fn capture_one(pool: &PgPool, content: &str) -> ThoughtId {
        capture(
            pool,
            TEST_MODEL_ID,
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

        // Row stays in queue; last_error is set; attempts bumped to 1.
        let row = sqlx::query!(
            r#"SELECT attempts, last_error FROM pending_embeddings"#,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.attempts, 1);
        assert!(row.last_error.is_some());
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_idempotent_on_crash_replay(pool: PgPool) {
        // Simulate the worker crashing between insert_thought_embedding and
        // mark_embedded: do the first two steps by hand (leaving the queue
        // row in place with attempts=1), then run drain — the second
        // re-insert must be a no-op rather than a UNIQUE-violation error.
        let id = capture_one(&pool, "replay").await;
        let model = EmbeddingModel::bge_m3();

        // Claim (bumps attempts→1, leaves row).
        let job = engram_storage::claim_pending(&pool, 10)
            .await
            .unwrap()
            .pop()
            .unwrap();
        // Insert the embedding directly (worker did this) but skip mark_embedded.
        let emb = Embedding::new(model.clone(), vec![0.5_f32; 1024]).unwrap();
        engram_storage::insert_thought_embedding(&pool, id, &emb)
            .await
            .unwrap();

        // Queue row still present (operator's "crash" happened here).
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);
        let _ = job;

        // Recovery drain: must not panic on duplicate insert.
        let good = FakeEmbedder::new();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 1, "replay tick must mark embedded cleanly");
        assert_eq!(report.failed, 0);
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_marks_failed_when_model_id_mismatch(pool: PgPool) {
        // Capture under the active model, then run drain with a *different*
        // embedder model. The job should be marked failed (not silently
        // embedded under the wrong model).
        let _id = capture_one(&pool, "mismatched model").await;

        let other = FakeEmbedder::with_model(EmbeddingModel::new("other:1024", 1024));
        let report = drain_pending_embeddings(&pool, &other, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 1);

        let row = sqlx::query!(
            r#"SELECT last_error FROM pending_embeddings"#,
        )
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

    // -- M3 Phase B step 1: fact dispatch in process_job ----------------------

    /// Helper: insert a fact and enqueue an embedding for it, bypassing the
    /// reflector. Returns the new fact's id.
    async fn enqueue_fact(pool: &PgPool, statement: &str) -> uuid::Uuid {
        let scope = Scope::new("global").unwrap();
        let thought_id = capture_one(pool, "anchor thought").await;
        let run_id = engram_storage::start_run(pool, "fake/extractor", 1, None)
            .await
            .unwrap();
        let fact_id = engram_storage::insert_fact(
            pool,
            engram_storage::NewFact {
                scope: &scope,
                statement,
                subject: None,
                predicate: None,
                object: None,
                source_thought_id: thought_id,
                extractor_model: "fake/extractor",
                extractor_version: 1,
                source_run_id: Some(run_id),
                confidence: 0.9,
                flagged: false,
            },
        )
        .await
        .unwrap();
        engram_storage::enqueue_embedding(
            pool,
            engram_storage::target::FACT,
            fact_id,
            TEST_MODEL_ID,
        )
        .await
        .unwrap();
        fact_id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_processes_fact_pending_to_embedding(pool: PgPool) {
        let fact_id = enqueue_fact(&pool, "Engram uses pgvector").await;

        let good = FakeEmbedder::new();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        // capture_one also enqueued the source thought → 2 jobs found, 2 embedded.
        assert!(report.found >= 1);
        assert!(report.embedded >= 1);
        assert_eq!(report.failed, 0);

        // Fact row landed under target_kind='fact'.
        let n = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM embeddings
               WHERE target_kind = 'fact' AND target_id = $1 AND model_id = $2"#,
            fact_id,
            TEST_MODEL_ID,
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(n, 1, "fact embedding should be present after drain");

        // pending row removed.
        let pending = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM pending_embeddings
               WHERE target_kind = 'fact' AND target_id = $1"#,
            fact_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(pending, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drain_marks_failed_when_fact_target_missing(pool: PgPool) {
        // Enqueue a pending row for a fact_id that doesn't exist.
        let bogus = uuid::Uuid::new_v4();
        engram_storage::enqueue_embedding(
            &pool,
            engram_storage::target::FACT,
            bogus,
            TEST_MODEL_ID,
        )
        .await
        .unwrap();

        let good = FakeEmbedder::new();
        let report = drain_pending_embeddings(&pool, &good, 10).await.unwrap();
        assert_eq!(report.found, 1);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 1);

        // Row stays in the queue with last_error set (SourceMissing).
        let row = sqlx::query!(
            r#"SELECT last_error FROM pending_embeddings
               WHERE target_kind = 'fact' AND target_id = $1"#,
            bogus,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            row.last_error
                .as_deref()
                .is_some_and(|e| e.contains("source thought no longer exists")),
            "expected SourceMissing-derived error, got {:?}",
            row.last_error
        );
    }
}

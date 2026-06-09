//! `embed_backfill` — operator escape hatch for healing the embedding state.
//!
//! Two failure modes this is meant to recover from:
//! 1. Thoughts captured *before* the queue existed: they were written
//!    without a `pending_embeddings` row, and never embedded if the embedder
//!    was down at capture time.
//! 2. Thoughts captured *after* the queue existed but whose enqueue lost
//!    the race against a server crash (insert_thought succeeded;
//!    enqueue_embedding didn't).
//!
//! The flow is "heal then drain": enqueue every unembedded thought (skipping
//! ones already queued — idempotent), then drain the queue. Bounded by
//! `--limit` so the operator can run it as a controlled one-shot.
//!
//! M4 simplification: facts are gone, so the heal step only touches
//! thoughts. The `--target` flag at the CLI is dropped; this function no
//! longer takes a target enum.

use kengram_core::Embedder;
use sqlx::PgPool;

use crate::drain::{DrainError, drain_pending_embeddings};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackfillReport {
    /// Number of unembedded thoughts found and freshly enqueued.
    pub healed: usize,
    /// Number processed off the queue and successfully embedded.
    pub embedded: usize,
    /// Number that failed during embed/persist. Each failure is logged
    /// with pending_id + reason; the row stays in the queue for the worker.
    pub failed: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum BackfillError {
    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),

    #[error("drain error: {0}")]
    Drain(#[from] DrainError),
}

/// Heal-then-drain backfill for thoughts. `limit` caps the total number of
/// embeddings produced this call.
pub async fn embed_backfill(
    pool: &PgPool,
    embedder: &dyn Embedder,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<BackfillReport, BackfillError> {
    let model_id = &embedder.model().id;
    let healed =
        kengram_storage::enqueue_unembedded_thoughts(pool, model_id, scope, scope_prefix, limit)
            .await?;

    const BATCH: i64 = 16;
    let mut report = BackfillReport {
        healed,
        ..Default::default()
    };
    let mut budget = limit;

    while budget > 0 {
        let take = BATCH.min(budget);
        let drain = drain_pending_embeddings(pool, embedder, take).await?;
        if drain.found == 0 {
            break;
        }
        report.embedded += drain.embedded;
        report.failed += drain.failed;
        budget -= drain.found as i64;
        if drain.embedded == 0 {
            break;
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use kengram_core::{EmbeddingModel, Metadata, Scope, Source, ThoughtId};
    use kengram_embed::{FakeBehavior, FakeEmbedder};
    use sha2::{Digest, Sha256};

    const TEST_EMBEDDER_MODEL_ID: &str = "bge-m3:1024";

    async fn cap(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        capture(
            pool,
            TEST_EMBEDDER_MODEL_ID,
            None,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new(scope).unwrap()),
                metadata: None,
                argus_source_event: None,
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    fn fingerprint_of(content: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    }

    /// Insert a thought *directly* via the storage layer (no enqueue). Used
    /// to simulate a captured-but-lost-the-enqueue thought.
    async fn raw_insert(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        let scope = Scope::new(scope).unwrap();
        let source = Source::new("test").unwrap();
        let metadata = Metadata::empty();
        let (inserted, _) = kengram_storage::insert_thought(
            pool,
            kengram_storage::NewThought {
                scope: &scope,
                content,
                source: &source,
                metadata: &metadata,
                content_fingerprint: fingerprint_of(content),
            },
        )
        .await
        .unwrap();
        inserted.id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drains_already_queued_thoughts(pool: PgPool) {
        let id_a = cap(&pool, "alpha", "global").await;
        let id_b = cap(&pool, "beta", "global").await;
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 2);

        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, None, None, 100).await.unwrap();
        assert_eq!(report.healed, 0, "both thoughts were already queued");
        assert_eq!(report.embedded, 2);
        assert_eq!(report.failed, 0);

        assert!(
            kengram_storage::thought_has_embedding(&pool, id_a, good.model())
                .await
                .unwrap()
        );
        assert!(
            kengram_storage::thought_has_embedding(&pool, id_b, good.model())
                .await
                .unwrap()
        );
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn heals_unembedded_thoughts_then_drains(pool: PgPool) {
        // A thought inserted directly bypasses the queue. Backfill must
        // heal then drain it.
        let id = raw_insert(&pool, "stranded", "global").await;
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 0);

        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, None, None, 100).await.unwrap();
        assert_eq!(report.healed, 1);
        assert_eq!(report.embedded, 1);
        assert!(
            kengram_storage::thought_has_embedding(&pool, id, good.model())
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn skips_already_embedded(pool: PgPool) {
        let id = cap(&pool, "already done", "global").await;
        let good = FakeEmbedder::new();
        let first = embed_backfill(&pool, &good, None, None, 100).await.unwrap();
        assert_eq!(first.embedded, 1);

        let second = embed_backfill(&pool, &good, None, None, 100).await.unwrap();
        assert_eq!(second.healed, 0);
        assert_eq!(second.embedded, 0);
        assert_eq!(second.failed, 0);
        assert!(
            kengram_storage::thought_has_embedding(&pool, id, good.model())
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn respects_scope_filter(pool: PgPool) {
        let w_a = cap(&pool, "work-a", "work").await;
        let w_b = cap(&pool, "work-b", "work").await;
        let p = cap(&pool, "personal", "personal").await;

        sqlx::query!("DELETE FROM pending_embeddings")
            .execute(&pool)
            .await
            .unwrap();

        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, Some("work"), None, 100)
            .await
            .unwrap();
        assert_eq!(report.healed, 2);
        assert_eq!(report.embedded, 2);

        assert!(
            kengram_storage::thought_has_embedding(&pool, w_a, good.model())
                .await
                .unwrap()
        );
        assert!(
            kengram_storage::thought_has_embedding(&pool, w_b, good.model())
                .await
                .unwrap()
        );
        assert!(
            !kengram_storage::thought_has_embedding(&pool, p, good.model())
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn respects_limit(pool: PgPool) {
        for i in 0..5 {
            cap(&pool, &format!("t-{i}"), "global").await;
        }
        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, None, None, 2).await.unwrap();
        assert!(
            report.embedded <= 2,
            "must not exceed limit; got {}",
            report.embedded
        );
        assert!(report.embedded >= 1);
        assert!(kengram_storage::count_pending(&pool).await.unwrap() > 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn handles_embedder_failure_for_individual_thoughts(pool: PgPool) {
        cap(&pool, "stays pending", "global").await;

        let still_bad =
            FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Timeout);
        let report = embed_backfill(&pool, &still_bad, None, None, 100)
            .await
            .unwrap();
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 1);
        assert_eq!(kengram_storage::count_pending(&pool).await.unwrap(), 1);
    }
}

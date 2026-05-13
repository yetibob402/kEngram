//! `embed_backfill` — operator escape hatch for healing the embedding state.
//!
//! Two failure modes this is meant to recover from:
//! 1. Thoughts captured *before* M2 Phase B existed: they were written
//!    without a `pending_embeddings` row, and never embedded if the embedder
//!    was down at capture time.
//! 2. Thoughts captured *after* Phase B but whose enqueue lost the race
//!    against a server crash (insert_thought succeeded; enqueue_embedding
//!    didn't).
//!
//! The flow is "heal then drain": enqueue every unembedded thought (skipping
//! ones already queued — idempotent), then drain the queue. Bounded by
//! `--limit` so the operator can run it as a controlled one-shot.
//!
//! The `engram worker` process is the continuous version; this is the
//! manual button. See `docs/milestones/m2-progress.md`.

use engram_core::Embedder;
use sqlx::PgPool;

use crate::drain::{drain_pending_embeddings, DrainError};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackfillReport {
    /// Number of unembedded thoughts found and freshly enqueued (excludes
    /// rows already in the queue from a prior server-side enqueue).
    pub healed: usize,
    /// Number processed off the queue and successfully embedded.
    pub embedded: usize,
    /// Number that failed during embed/persist. Each failure is logged with
    /// pending_id + reason; the row stays in the queue for the worker.
    pub failed: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum BackfillError {
    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),

    #[error("drain error: {0}")]
    Drain(#[from] DrainError),
}

/// Heal-then-drain backfill. `limit` caps the total number of embeddings
/// produced this call.
pub async fn embed_backfill(
    pool: &PgPool,
    embedder: &dyn Embedder,
    scope: Option<&str>,
    limit: i64,
) -> Result<BackfillReport, BackfillError> {
    // (1) Enqueue any unembedded thoughts that don't already have a queue
    //     row. Bounded by `limit`. `ON CONFLICT DO NOTHING` makes this
    //     idempotent over rows that are already pending.
    let healed = engram_storage::enqueue_unembedded_thoughts(
        pool,
        &embedder.model().id,
        scope,
        limit,
    )
    .await?;

    // (2) Drain up to `limit` total — pull batches of 16 (a reasonable
    //     default tick batch) until either we hit the budget or the queue
    //     dries up. This mirrors what the worker would do across multiple
    //     ticks, condensed into one foreground run.
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
        // If everything in this batch failed, stop — otherwise we'd loop
        // indefinitely re-claiming the same dead rows.
        if drain.embedded == 0 {
            break;
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{capture, CaptureRequest};
    use engram_core::{EmbeddingModel, Scope, Source, ThoughtId};
    use engram_embed::{FakeBehavior, FakeEmbedder};

    const TEST_MODEL_ID: &str = "bge-m3:1024";

    async fn cap(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        capture(
            pool,
            TEST_MODEL_ID,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new(scope).unwrap()),
                metadata: None,
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    /// Insert a thought *directly* via the storage layer (no enqueue). Used
    /// to simulate a pre-M2 thought or a captured-but-lost-the-enqueue thought.
    async fn raw_insert(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        let scope = Scope::new(scope).unwrap();
        let source = Source::new("test").unwrap();
        let metadata = engram_core::Metadata::empty();
        engram_storage::insert_thought(
            pool,
            engram_storage::NewThought {
                scope: &scope,
                content,
                source: &source,
                metadata: &metadata,
            },
        )
        .await
        .unwrap()
        .id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn drains_already_queued_thoughts(pool: PgPool) {
        // Two captures → two queued rows; embedder is up → backfill embeds them.
        let id_a = cap(&pool, "alpha", "global").await;
        let id_b = cap(&pool, "beta", "global").await;
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 2);

        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, None, 100).await.unwrap();
        assert_eq!(report.healed, 0, "both thoughts were already queued");
        assert_eq!(report.embedded, 2);
        assert_eq!(report.failed, 0);

        assert!(engram_storage::thought_has_embedding(&pool, id_a, good.model()).await.unwrap());
        assert!(engram_storage::thought_has_embedding(&pool, id_b, good.model()).await.unwrap());
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn heals_pre_m2_thoughts_then_drains(pool: PgPool) {
        // A thought inserted directly bypasses the queue (simulating a
        // pre-M2 row or a crashed enqueue). Backfill must heal then drain it.
        let id = raw_insert(&pool, "stranded", "global").await;
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);

        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, None, 100).await.unwrap();
        assert_eq!(report.healed, 1);
        assert_eq!(report.embedded, 1);
        assert!(engram_storage::thought_has_embedding(&pool, id, good.model()).await.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn skips_already_embedded(pool: PgPool) {
        let id = cap(&pool, "already done", "global").await;
        let good = FakeEmbedder::new();
        // First backfill embeds it.
        let first = embed_backfill(&pool, &good, None, 100).await.unwrap();
        assert_eq!(first.embedded, 1);

        // Second backfill: no work to do.
        let second = embed_backfill(&pool, &good, None, 100).await.unwrap();
        assert_eq!(second.healed, 0);
        assert_eq!(second.embedded, 0);
        assert_eq!(second.failed, 0);
        assert!(engram_storage::thought_has_embedding(&pool, id, good.model()).await.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn respects_scope_filter(pool: PgPool) {
        // Capture (so each is queued) under two scopes; only "work" should drain.
        let w_a = cap(&pool, "work-a", "work").await;
        let w_b = cap(&pool, "work-b", "work").await;
        let p = cap(&pool, "personal", "personal").await;

        // Heal step is scope-bounded; queue rows from capture exist for *all*
        // three thoughts. The drain step, however, isn't scope-bounded — it
        // processes whatever's in the queue. To keep the test's intent
        // honest, first clear the queue and re-drive via heal with scope.
        sqlx::query!("DELETE FROM pending_embeddings").execute(&pool).await.unwrap();

        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, Some("work"), 100).await.unwrap();
        assert_eq!(report.healed, 2);
        assert_eq!(report.embedded, 2);

        assert!(engram_storage::thought_has_embedding(&pool, w_a, good.model()).await.unwrap());
        assert!(engram_storage::thought_has_embedding(&pool, w_b, good.model()).await.unwrap());
        // Personal stays unembedded.
        assert!(!engram_storage::thought_has_embedding(&pool, p, good.model()).await.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn respects_limit(pool: PgPool) {
        // Capture 5 thoughts, ask backfill to only process 2.
        for i in 0..5 {
            cap(&pool, &format!("t-{i}"), "global").await;
        }
        let good = FakeEmbedder::new();
        let report = embed_backfill(&pool, &good, None, 2).await.unwrap();
        assert!(report.embedded <= 2, "must not exceed limit; got {}", report.embedded);
        // At least one batch must complete to be useful.
        assert!(report.embedded >= 1);
        // Some rows are left in the queue for the worker.
        assert!(engram_storage::count_pending(&pool).await.unwrap() > 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn handles_embedder_failure_for_individual_thoughts(pool: PgPool) {
        cap(&pool, "stays pending", "global").await;

        let still_bad =
            FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Timeout);
        let report = embed_backfill(&pool, &still_bad, None, 100).await.unwrap();
        assert_eq!(report.embedded, 0);
        assert_eq!(report.failed, 1);
        // Row stayed in the queue with last_error set.
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);
    }
}

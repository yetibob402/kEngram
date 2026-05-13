//! The `capture` operation: write a thought row and enqueue an embedding job.
//!
//! Capture does **not** call the embedder. It records the thought (durable),
//! enqueues a job in `pending_embeddings`, and returns immediately with
//! `embedding_status: "pending"`. The `engram worker` process drains the
//! queue and writes the embedding row on its next tick — the brief window
//! between capture and embed is exactly the "thought is searchable by
//! trigram only" window described in `docs/milestones/m2-facts-pipeline.md`.
//!
//! Phase B success criterion: capture latency is decoupled from embedder
//! responsiveness, and a flaky embedder can no longer block new captures.

use engram_core::{EmbeddingStatus, Metadata, Scope, Source, ThoughtId};
use sqlx::PgPool;

/// Hard upper bound on a single thought's content. Enforced before the DB
/// write so callers get a clean rejection.
pub const MAX_CONTENT_LEN: usize = 1_048_576; // 1 MiB

#[derive(Debug, Clone)]
pub struct CaptureRequest {
    pub content: String,
    pub source: Source,
    pub scope: Option<Scope>,
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Clone)]
pub struct CaptureResponse {
    pub thought_id: ThoughtId,
    pub embedding_status: EmbeddingStatus,
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("content must be non-empty")]
    EmptyContent,

    #[error("content is too long: {got} bytes (max {max})")]
    ContentTooLong { got: usize, max: usize },

    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Capture a thought. Inserts the `thoughts` row, then inserts a job into
/// `pending_embeddings` for the worker to pick up. Always returns
/// `EmbeddingStatus::Pending`.
///
/// `model_id` is the active embedder's identity (e.g. `"bge-m3:1024"`). The
/// worker uses it to pair the row with the right embedder on drain.
pub async fn capture(
    pool: &PgPool,
    model_id: &str,
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

    let scope = request.scope.unwrap_or_default();
    let metadata = request.metadata.unwrap_or_default();

    let inserted = engram_storage::insert_thought(
        pool,
        engram_storage::NewThought {
            scope: &scope,
            content: &request.content,
            source: &request.source,
            metadata: &metadata,
        },
    )
    .await?;

    engram_storage::enqueue_embedding(
        pool,
        engram_storage::target::THOUGHT,
        inserted.id.into_uuid(),
        model_id,
    )
    .await?;

    Ok(CaptureResponse {
        thought_id: inserted.id,
        embedding_status: EmbeddingStatus::Pending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::EmbeddingModel;
    use serde_json::json;

    const TEST_MODEL_ID: &str = "bge-m3:1024";

    fn req(content: &str, source: &str) -> CaptureRequest {
        CaptureRequest {
            content: content.to_string(),
            source: Source::new(source).unwrap(),
            scope: None,
            metadata: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn writes_thought_and_enqueues_returns_pending(pool: PgPool) {
        let resp = capture(&pool, TEST_MODEL_ID, req("first thought", "manual"))
            .await
            .unwrap();

        assert_eq!(resp.embedding_status, EmbeddingStatus::Pending);

        let fetched = engram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.content, "first thought");

        // Queue row exists; no embedding row yet.
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 1);
        assert!(
            !engram_storage::thought_has_embedding(
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
        let err = capture(&pool, TEST_MODEL_ID, req("", "manual"))
            .await
            .unwrap_err();
        assert!(matches!(err, CaptureError::EmptyContent));
        // Errored before the insert; queue stays empty.
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn overlong_content_returns_error(pool: PgPool) {
        let big = "x".repeat(MAX_CONTENT_LEN + 1);
        let err = capture(&pool, TEST_MODEL_ID, req(&big, "manual"))
            .await
            .unwrap_err();
        assert!(matches!(err, CaptureError::ContentTooLong { got, max } if got > max));
        assert_eq!(engram_storage::count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn defaults_scope_to_global_when_missing(pool: PgPool) {
        let resp = capture(&pool, TEST_MODEL_ID, req("hello", "manual"))
            .await
            .unwrap();
        let fetched = engram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.scope, Scope::global());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn defaults_metadata_to_empty_when_missing(pool: PgPool) {
        let resp = capture(&pool, TEST_MODEL_ID, req("hello", "manual"))
            .await
            .unwrap();
        let fetched = engram_storage::fetch_thought(&pool, resp.thought_id)
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
            metadata: Some(Metadata::from(json!({"session_id": "abc", "tool_name": "TodoWrite"}))),
        };
        let resp = capture(&pool, TEST_MODEL_ID, request.clone()).await.unwrap();

        let fetched = engram_storage::fetch_thought(&pool, resp.thought_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.scope, request.scope.unwrap());
        assert_eq!(fetched.source, request.source);
        assert_eq!(fetched.metadata, request.metadata.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_targets_thought_kind_with_active_model(pool: PgPool) {
        let resp = capture(&pool, TEST_MODEL_ID, req("queue me", "manual"))
            .await
            .unwrap();

        // Inspect the queue row directly.
        let row = sqlx::query!(
            r#"SELECT target_kind, target_id, model_id FROM pending_embeddings"#,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.target_kind, "thought");
        assert_eq!(row.target_id, resp.thought_id.into_uuid());
        assert_eq!(row.model_id, TEST_MODEL_ID);
    }
}

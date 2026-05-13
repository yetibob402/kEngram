//! engram-storage: sqlx-backed repository functions.
//!
//! The `Embedder` trait is the only place we hide a backend choice behind a
//! trait — storage is concrete sqlx + Postgres. CLAUDE.md rule: compile-time
//! `sqlx::query!` everywhere except where pgvector's vector binding gets in
//! the way of the macro (currently: only `insert_embedding`).

use engram_core::{
    Embedding, EmbeddingModel, EmbeddingStatus, Hit, Metadata, Scope, ScopeError, Source,
    SourceError, Thought, ThoughtId,
};
use sqlx::PgPool;
use time::OffsetDateTime;
use uuid::Uuid;

pub mod target {
    //! `embeddings.target_kind` enum-as-string. Matches the CHECK constraint
    //! on the column.
    pub const THOUGHT: &str = "thought";
    pub const ARTIFACT_CHUNK: &str = "artifact_chunk";
    pub const FACT: &str = "fact";
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("invalid scope decoded from database: {0}")]
    InvalidScope(#[from] ScopeError),

    #[error("invalid source decoded from database: {0}")]
    InvalidSource(#[from] SourceError),
}

/// Inputs for inserting a new thought. Borrowing keeps the call cheap.
#[derive(Debug, Clone, Copy)]
pub struct NewThought<'a> {
    pub scope: &'a Scope,
    pub content: &'a str,
    pub source: &'a Source,
    pub metadata: &'a Metadata,
}

/// What the DB tells us after a thought is inserted.
#[derive(Debug, Clone)]
pub struct InsertedThought {
    pub id: ThoughtId,
    pub created_at: OffsetDateTime,
}

/// Insert a thought. The database generates `id` and `created_at`.
pub async fn insert_thought(
    pool: &PgPool,
    t: NewThought<'_>,
) -> Result<InsertedThought, StorageError> {
    let row = sqlx::query!(
        r#"
        INSERT INTO thoughts (scope, content, source, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING id, created_at
        "#,
        t.scope.as_str(),
        t.content,
        t.source.as_str(),
        t.metadata.as_value(),
    )
    .fetch_one(pool)
    .await?;

    Ok(InsertedThought {
        id: ThoughtId::from(row.id),
        created_at: row.created_at,
    })
}

/// Insert an embedding row tied to some target (thought / artifact_chunk / fact).
///
/// Uses `sqlx::query` (runtime-checked) rather than the macro because pgvector's
/// `Vector` type is awkward to bind through `query!` — the macro can't infer
/// the column type from the schema alone. The query is still parameterised, so
/// no injection risk.
pub async fn insert_embedding(
    pool: &PgPool,
    target_kind: &'static str,
    target_id: Uuid,
    model: &EmbeddingModel,
    vector: Vec<f32>,
) -> Result<(), StorageError> {
    let pgv = pgvector::Vector::from(vector);
    // ON CONFLICT DO NOTHING: makes the insert idempotent under M2-style worker
    // replay (worker crashes after this insert but before `mark_embedded` →
    // next tick re-claims, re-embeds, and re-inserts; the UNIQUE constraint
    // would otherwise reject the duplicate).
    sqlx::query(
        r#"
        INSERT INTO embeddings (target_kind, target_id, model_id, model_version, vector)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (target_kind, target_id, model_id, model_version) DO NOTHING
        "#,
    )
    .bind(target_kind)
    .bind(target_id)
    .bind(&model.id)
    .bind(1_i32) // model_version: bumped only when the same model_id changes its meaning
    .bind(pgv)
    .execute(pool)
    .await?;
    Ok(())
}

/// Convenience: insert an embedding tied to a thought, taking the engram-core
/// `Embedding` wrapper.
pub async fn insert_thought_embedding(
    pool: &PgPool,
    thought_id: ThoughtId,
    embedding: &Embedding,
) -> Result<(), StorageError> {
    insert_embedding(
        pool,
        target::THOUGHT,
        thought_id.into_uuid(),
        &embedding.model,
        embedding.vector.clone(),
    )
    .await
}

/// Look up a thought by id. Returns `None` if not found.
pub async fn fetch_thought(
    pool: &PgPool,
    id: ThoughtId,
) -> Result<Option<Thought>, StorageError> {
    let row = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata
        FROM thoughts
        WHERE id = $1
        "#,
        id.into_uuid(),
    )
    .fetch_optional(pool)
    .await?;

    let Some(r) = row else {
        return Ok(None);
    };

    Ok(Some(Thought {
        id: ThoughtId::from(r.id),
        scope: Scope::new(r.scope)?,
        content: r.content,
        source: Source::new(r.source)?,
        created_at: r.created_at,
        metadata: Metadata::from(r.metadata),
    }))
}

/// True if an embedding exists for the given thought under the given model.
pub async fn thought_has_embedding(
    pool: &PgPool,
    id: ThoughtId,
    model: &EmbeddingModel,
) -> Result<bool, StorageError> {
    let row = sqlx::query!(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM embeddings
            WHERE target_kind = 'thought' AND target_id = $1 AND model_id = $2
        ) AS "exists!"
        "#,
        id.into_uuid(),
        model.id,
    )
    .fetch_one(pool)
    .await?;
    Ok(row.exists)
}

/// Result of `fetch_thought_with_provenance`. `embedded_at` is `None` when
/// no embedding row exists for the active model.
#[derive(Debug, Clone)]
pub struct ThoughtWithProvenance {
    pub thought: Thought,
    pub embedding_status: EmbeddingStatus,
    pub embedded_at: Option<OffsetDateTime>,
}

/// Fetch a thought along with its embedding provenance for the given model.
pub async fn fetch_thought_with_provenance(
    pool: &PgPool,
    id: ThoughtId,
    model: &EmbeddingModel,
) -> Result<Option<ThoughtWithProvenance>, StorageError> {
    let row = sqlx::query!(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               e.created_at AS "embedded_at?"
        FROM thoughts t
        LEFT JOIN embeddings e
            ON e.target_kind = 'thought'
           AND e.target_id = t.id
           AND e.model_id = $2
        WHERE t.id = $1
        "#,
        id.into_uuid(),
        model.id,
    )
    .fetch_optional(pool)
    .await?;

    let Some(r) = row else {
        return Ok(None);
    };

    let thought = Thought {
        id: ThoughtId::from(r.id),
        scope: Scope::new(r.scope)?,
        content: r.content,
        source: Source::new(r.source)?,
        created_at: r.created_at,
        metadata: Metadata::from(r.metadata),
    };

    let embedding_status = if r.embedded_at.is_some() {
        EmbeddingStatus::Indexed
    } else {
        EmbeddingStatus::Pending
    };

    Ok(Some(ThoughtWithProvenance {
        thought,
        embedding_status,
        embedded_at: r.embedded_at,
    }))
}

/// Recent thoughts in (optional) scope, ordered newest-first.
pub async fn recent_thoughts(
    pool: &PgPool,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata
        FROM thoughts
        WHERE ($1::text IS NULL OR scope = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        scope,
        limit,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            Ok(Thought {
                id: ThoughtId::from(r.id),
                scope: Scope::new(r.scope)?,
                content: r.content,
                source: Source::new(r.source)?,
                created_at: r.created_at,
                metadata: Metadata::from(r.metadata),
            })
        })
        .collect()
}

/// Trigram-similarity search over `thoughts.content`. Hits are returned in
/// descending order of `similarity(content, query)` and filtered to a
/// minimum similarity of 0.1 — much lower than the default `pg_trgm.%`
/// threshold of 0.3, which is too strict for "user typed a short word
/// hoping to find it inside a long thought." At M1 volumes (low hundreds
/// of thoughts) the sequential scan is fast; once data grows we can switch
/// to an index-friendly `ORDER BY content <-> $1 LIMIT N` shape.
pub async fn search_trigram(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata,
               similarity(content, $1) AS "sim!: f32"
        FROM thoughts
        WHERE similarity(content, $1) > 0.1
          AND ($2::text IS NULL OR scope = $2)
        ORDER BY similarity(content, $1) DESC
        LIMIT $3
        "#,
        query,
        scope,
        limit,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            Ok(Hit {
                thought: Thought {
                    id: ThoughtId::from(r.id),
                    scope: Scope::new(r.scope)?,
                    content: r.content,
                    source: Source::new(r.source)?,
                    created_at: r.created_at,
                    metadata: Metadata::from(r.metadata),
                },
                score: r.sim,
            })
        })
        .collect()
}

/// Find thoughts that don't yet have an embedding row for the given model.
/// Oldest first — backfill should clear the backlog FIFO.
pub async fn find_unembedded_thoughts(
    pool: &PgPool,
    model: &EmbeddingModel,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata
        FROM thoughts t
        LEFT JOIN embeddings e
            ON e.target_kind = 'thought'
           AND e.target_id = t.id
           AND e.model_id = $1
        WHERE e.id IS NULL
          AND ($2::text IS NULL OR t.scope = $2)
        ORDER BY t.created_at ASC
        LIMIT $3
        "#,
        model.id,
        scope,
        limit,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            Ok(Thought {
                id: ThoughtId::from(r.id),
                scope: Scope::new(r.scope)?,
                content: r.content,
                source: Source::new(r.source)?,
                created_at: r.created_at,
                metadata: Metadata::from(r.metadata),
            })
        })
        .collect()
}

/// A row pulled off the `pending_embeddings` queue by `claim_pending`.
/// `attempts` is the *new* attempt count (post-bump): a job freshly claimed
/// for its first attempt returns `attempts = 1`.
#[derive(Debug, Clone)]
pub struct PendingJob {
    pub id: Uuid,
    pub target_kind: String,
    pub target_id: Uuid,
    pub model_id: String,
    pub attempts: i32,
}

/// Enqueue a target for embedding by the worker.
///
/// Idempotent: the UNIQUE `(target_kind, target_id, model_id)` constraint on
/// `pending_embeddings` (migration 0002) means a duplicate enqueue is a no-op.
/// Returns `true` if a new row was inserted, `false` if the row already existed.
pub async fn enqueue_embedding(
    pool: &PgPool,
    target_kind: &str,
    target_id: Uuid,
    model_id: &str,
) -> Result<bool, StorageError> {
    let result = sqlx::query!(
        r#"
        INSERT INTO pending_embeddings (target_kind, target_id, model_id)
        VALUES ($1, $2, $3)
        ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
        "#,
        target_kind,
        target_id,
        model_id,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Atomically claim up to `batch_size` pending jobs, oldest first, bumping
/// `attempts` and `last_attempt_at` on each.
///
/// The inner `SELECT ... FOR UPDATE SKIP LOCKED` is the canonical Postgres
/// pattern for a competing-consumers queue: rows already locked by another
/// transaction are skipped, so concurrent workers see disjoint claims. Locks
/// release at statement commit; no long-held transaction is required.
pub async fn claim_pending(
    pool: &PgPool,
    batch_size: i64,
) -> Result<Vec<PendingJob>, StorageError> {
    let rows = sqlx::query!(
        r#"
        UPDATE pending_embeddings p
        SET attempts = p.attempts + 1, last_attempt_at = NOW()
        FROM (
            SELECT id FROM pending_embeddings
            ORDER BY enqueued_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
        ) AS sub
        WHERE p.id = sub.id
        RETURNING p.id, p.target_kind, p.target_id, p.model_id, p.attempts
        "#,
        batch_size,
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| PendingJob {
            id: r.id,
            target_kind: r.target_kind,
            target_id: r.target_id,
            model_id: r.model_id,
            attempts: r.attempts,
        })
        .collect())
}

/// Mark a claimed job as successfully embedded — removes it from the queue.
pub async fn mark_embedded(pool: &PgPool, pending_id: Uuid) -> Result<(), StorageError> {
    sqlx::query!(
        r#"DELETE FROM pending_embeddings WHERE id = $1"#,
        pending_id
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Record a failure for a claimed job. The row stays in the queue (so the
/// next tick re-claims it); `last_error` captures why this attempt failed.
/// `attempts` is *not* bumped here — `claim_pending` already bumped it.
pub async fn mark_failed(
    pool: &PgPool,
    pending_id: Uuid,
    error_msg: &str,
) -> Result<(), StorageError> {
    sqlx::query!(
        r#"UPDATE pending_embeddings SET last_error = $2 WHERE id = $1"#,
        pending_id,
        error_msg,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Heal-step companion to the worker: enqueue every unembedded thought (for
/// the given model and optional scope) that doesn't already have a queue row.
/// Used by `engram embed-backfill` to catch pre-M2 thoughts (captured before
/// the queue existed) and any thought that slipped through a server crash
/// between `insert_thought` and `enqueue_embedding`.
///
/// `ON CONFLICT DO NOTHING` keeps it idempotent even when the queue already
/// has entries for some of the thoughts in the LEFT-JOIN set.
pub async fn enqueue_unembedded_thoughts(
    pool: &PgPool,
    model_id: &str,
    scope: Option<&str>,
    limit: i64,
) -> Result<usize, StorageError> {
    let result = sqlx::query!(
        r#"
        INSERT INTO pending_embeddings (target_kind, target_id, model_id)
        SELECT 'thought', t.id, $1
        FROM thoughts t
        LEFT JOIN embeddings e
            ON e.target_kind = 'thought'
           AND e.target_id = t.id
           AND e.model_id = $1
        WHERE e.id IS NULL
          AND ($2::text IS NULL OR t.scope = $2)
        ORDER BY t.created_at ASC
        LIMIT $3
        ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
        "#,
        model_id,
        scope,
        limit,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() as usize)
}

/// Total rows currently in `pending_embeddings`. Cheap (no index scan
/// required for the small queue sizes this is meant for); intended for
/// tests and operator-driven observability.
pub async fn count_pending(pool: &PgPool) -> Result<i64, StorageError> {
    let row = sqlx::query!(
        r#"SELECT COUNT(*) AS "count!" FROM pending_embeddings"#
    )
    .fetch_one(pool)
    .await?;
    Ok(row.count)
}

/// Vector-similarity kNN over `embeddings` for the given model. Hits are
/// returned in descending order of cosine similarity (`1 - cosine_distance`).
/// Uses the per-model HNSW partial index (`embeddings_<model>_hnsw`).
///
/// Uses `sqlx::query_as` rather than `sqlx::query!` because pgvector's
/// `Vector` binding is awkward through the macro. The query is still fully
/// parameterised.
pub async fn search_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    model: &EmbeddingModel,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    let pgv = pgvector::Vector::from(query_vector);

    let rows: Vec<VectorSearchRow> = sqlx::query_as(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               (e.vector <=> $1) AS distance
        FROM thoughts t
        JOIN embeddings e ON e.target_kind = 'thought' AND e.target_id = t.id
        WHERE e.model_id = $2
          AND ($3::text IS NULL OR t.scope = $3)
        ORDER BY e.vector <=> $1
        LIMIT $4
        "#,
    )
    .bind(pgv)
    .bind(&model.id)
    .bind(scope)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            // cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1] (typically [0, 1]).
            let score = (1.0 - r.distance) as f32;
            Ok(Hit {
                thought: Thought {
                    id: ThoughtId::from(r.id),
                    scope: Scope::new(r.scope)?,
                    content: r.content,
                    source: Source::new(r.source)?,
                    created_at: r.created_at,
                    metadata: Metadata::from(r.metadata),
                },
                score,
            })
        })
        .collect()
}

#[derive(sqlx::FromRow)]
struct VectorSearchRow {
    id: Uuid,
    scope: String,
    content: String,
    source: String,
    created_at: OffsetDateTime,
    metadata: serde_json::Value,
    distance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{EmbeddingModel, Metadata, Scope, Source};
    use serde_json::json;

    fn new_thought<'a>(scope: &'a Scope, source: &'a Source, metadata: &'a Metadata) -> NewThought<'a> {
        NewThought {
            scope,
            content: "remember this",
            source,
            metadata,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_returns_id_and_timestamp(pool: PgPool) {
        let scope = Scope::new("work").unwrap();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::from(json!({"client_name": "test"}));

        let inserted = insert_thought(&pool, new_thought(&scope, &source, &metadata))
            .await
            .unwrap();

        // ID is non-nil, created_at is recent
        assert_ne!(*inserted.id.as_uuid(), Uuid::nil());
        let now = OffsetDateTime::now_utc();
        let drift = (now - inserted.created_at).whole_seconds().abs();
        assert!(drift < 10, "created_at not within 10s of now: drift={drift}s");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_returns_inserted_row(pool: PgPool) {
        let scope = Scope::new("personal").unwrap();
        let source = Source::new("agent:claude-code").unwrap();
        let metadata = Metadata::from(json!({"session_id": "abc"}));

        let inserted = insert_thought(&pool, new_thought(&scope, &source, &metadata))
            .await
            .unwrap();

        let fetched = fetch_thought(&pool, inserted.id).await.unwrap().unwrap();

        assert_eq!(fetched.id, inserted.id);
        assert_eq!(fetched.scope, scope);
        assert_eq!(fetched.content, "remember this");
        assert_eq!(fetched.source, source);
        assert_eq!(fetched.metadata, metadata);
        assert_eq!(fetched.created_at, inserted.created_at);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_returns_none_when_missing(pool: PgPool) {
        let id = ThoughtId::new();
        let result = fetch_thought(&pool, id).await.unwrap();
        assert!(result.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_embedding_persists_row(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();
        let inserted = insert_thought(&pool, new_thought(&scope, &source, &metadata))
            .await
            .unwrap();

        let model = EmbeddingModel::bge_m3();
        let vector = vec![0.0_f32; 1024];
        insert_embedding(
            &pool,
            target::THOUGHT,
            inserted.id.into_uuid(),
            &model,
            vector,
        )
        .await
        .unwrap();

        assert!(thought_has_embedding(&pool, inserted.id, &model).await.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn thought_has_embedding_is_false_for_unembedded(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();
        let inserted = insert_thought(&pool, new_thought(&scope, &source, &metadata))
            .await
            .unwrap();

        let model = EmbeddingModel::bge_m3();
        assert!(!thought_has_embedding(&pool, inserted.id, &model).await.unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_embedding_convenience_works(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();
        let inserted = insert_thought(&pool, new_thought(&scope, &source, &metadata))
            .await
            .unwrap();

        let model = EmbeddingModel::bge_m3();
        let embedding = Embedding::new(model.clone(), vec![0.5_f32; 1024]).unwrap();
        insert_thought_embedding(&pool, inserted.id, &embedding)
            .await
            .unwrap();

        assert!(thought_has_embedding(&pool, inserted.id, &model).await.unwrap());
    }

    async fn insert_test_thought(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        let scope = Scope::new(scope).unwrap();
        let source = Source::new("test").unwrap();
        let metadata = Metadata::empty();
        let inserted = insert_thought(
            pool,
            NewThought {
                scope: &scope,
                content,
                source: &source,
                metadata: &metadata,
            },
        )
        .await
        .unwrap();
        inserted.id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_newest_first(pool: PgPool) {
        let _a = insert_test_thought(&pool, "first", "global").await;
        // Tiny sleep so the timestamps differ.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let _b = insert_test_thought(&pool, "second", "global").await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let _c = insert_test_thought(&pool, "third", "global").await;

        let results = recent_thoughts(&pool, None, 10).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].content, "third");
        assert_eq!(results[1].content, "second");
        assert_eq!(results[2].content, "first");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_respects_scope_filter(pool: PgPool) {
        insert_test_thought(&pool, "work-1", "work").await;
        insert_test_thought(&pool, "personal-1", "personal").await;
        insert_test_thought(&pool, "work-2", "work").await;

        let work = recent_thoughts(&pool, Some("work"), 10).await.unwrap();
        assert_eq!(work.len(), 2);
        assert!(work.iter().all(|t| t.scope.as_str() == "work"));

        let personal = recent_thoughts(&pool, Some("personal"), 10).await.unwrap();
        assert_eq!(personal.len(), 1);
        assert_eq!(personal[0].content, "personal-1");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_respects_limit(pool: PgPool) {
        for i in 0..5 {
            insert_test_thought(&pool, &format!("t{i}"), "global").await;
        }
        let r = recent_thoughts(&pool, None, 2).await.unwrap();
        assert_eq!(r.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_finds_exact_match(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;
        insert_test_thought(&pool, "weather is nice today", "personal").await;

        let hits = search_trigram(&pool, "tcgplayer", None, 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].thought.content.contains("tcgplayer"));
        assert!(hits[0].score > 0.0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_respects_scope(pool: PgPool) {
        insert_test_thought(&pool, "tcgplayer info", "work").await;
        insert_test_thought(&pool, "tcgplayer info", "personal").await;

        let hits = search_trigram(&pool, "tcgplayer", Some("work"), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].thought.scope.as_str(), "work");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_returns_empty_for_no_match(pool: PgPool) {
        insert_test_thought(&pool, "completely unrelated text", "global").await;
        let hits = search_trigram(&pool, "xyzzyqwerty", None, 10).await.unwrap();
        assert!(hits.is_empty());
    }

    /// Helper: returns a 1024-dim unit vector with `1.0` at the given index.
    /// The `embeddings.vector` column is `vector(1024)` (matches BGE-M3 dims),
    /// so all test vectors must be that size.
    fn unit_vector_1024(pos: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; 1024];
        v[pos] = 1.0;
        v
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_finds_inserted_vector(pool: PgPool) {
        let model = EmbeddingModel::new("test:1024", 1024);

        let id_a = insert_test_thought(&pool, "a", "global").await;
        let id_b = insert_test_thought(&pool, "b", "global").await;

        let va = unit_vector_1024(0);
        let vb = unit_vector_1024(1);

        insert_thought_embedding(&pool, id_a, &Embedding::new(model.clone(), va.clone()).unwrap())
            .await
            .unwrap();
        insert_thought_embedding(&pool, id_b, &Embedding::new(model.clone(), vb).unwrap())
            .await
            .unwrap();

        // Query with the exact vector for 'a' → 'a' should rank first.
        let hits = search_vector_knn(&pool, va, &model, None, 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].thought.id, id_a);
        // Cosine similarity with itself = 1, so score ≈ 1.
        assert!((hits[0].score - 1.0).abs() < 1e-4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_filters_by_model(pool: PgPool) {
        let model_a = EmbeddingModel::new("test-a:1024", 1024);
        let model_b = EmbeddingModel::new("test-b:1024", 1024);

        let id_a = insert_test_thought(&pool, "a", "global").await;
        let id_b = insert_test_thought(&pool, "b", "global").await;

        let v = unit_vector_1024(0);
        insert_thought_embedding(&pool, id_a, &Embedding::new(model_a.clone(), v.clone()).unwrap())
            .await
            .unwrap();
        insert_thought_embedding(&pool, id_b, &Embedding::new(model_b.clone(), v.clone()).unwrap())
            .await
            .unwrap();

        // Query with model_a should only return id_a (not id_b, embedded under model_b).
        let hits = search_vector_knn(&pool, v, &model_a, None, 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].thought.id, id_a);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_with_provenance_indexed_when_embedded(pool: PgPool) {
        let id = insert_test_thought(&pool, "hello", "global").await;
        let model = EmbeddingModel::bge_m3();
        insert_thought_embedding(&pool, id, &Embedding::new(model.clone(), vec![0.0_f32; 1024]).unwrap())
            .await
            .unwrap();

        let prov = fetch_thought_with_provenance(&pool, id, &model)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(prov.embedding_status, EmbeddingStatus::Indexed);
        assert!(prov.embedded_at.is_some());
        assert_eq!(prov.thought.id, id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_with_provenance_pending_when_unembedded(pool: PgPool) {
        let id = insert_test_thought(&pool, "hello", "global").await;
        let model = EmbeddingModel::bge_m3();
        let prov = fetch_thought_with_provenance(&pool, id, &model)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(prov.embedding_status, EmbeddingStatus::Pending);
        assert!(prov.embedded_at.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_with_provenance_returns_none_for_missing(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let id = ThoughtId::new();
        let prov = fetch_thought_with_provenance(&pool, id, &model).await.unwrap();
        assert!(prov.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unembedded_thoughts_returns_thoughts_without_embedding(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let embedded = insert_test_thought(&pool, "embedded", "global").await;
        let unembedded = insert_test_thought(&pool, "unembedded", "global").await;

        insert_thought_embedding(
            &pool,
            embedded,
            &Embedding::new(model.clone(), vec![0.0_f32; 1024]).unwrap(),
        )
        .await
        .unwrap();

        let pending = find_unembedded_thoughts(&pool, &model, None, 100).await.unwrap();
        let ids: Vec<ThoughtId> = pending.iter().map(|t| t.id).collect();
        assert!(ids.contains(&unembedded));
        assert!(!ids.contains(&embedded));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unembedded_thoughts_respects_scope_and_limit(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        for i in 0..5 {
            insert_test_thought(&pool, &format!("work-{i}"), "work").await;
        }
        for i in 0..3 {
            insert_test_thought(&pool, &format!("personal-{i}"), "personal").await;
        }

        let work = find_unembedded_thoughts(&pool, &model, Some("work"), 100).await.unwrap();
        assert_eq!(work.len(), 5);
        assert!(work.iter().all(|t| t.scope.as_str() == "work"));

        let limited = find_unembedded_thoughts(&pool, &model, None, 4).await.unwrap();
        assert_eq!(limited.len(), 4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unembedded_thoughts_is_per_model(pool: PgPool) {
        let model_a = EmbeddingModel::new("a:1024", 1024);
        let model_b = EmbeddingModel::new("b:1024", 1024);

        let t = insert_test_thought(&pool, "hi", "global").await;
        insert_thought_embedding(
            &pool,
            t,
            &Embedding::new(model_a.clone(), vec![0.0_f32; 1024]).unwrap(),
        )
        .await
        .unwrap();

        // Under model_a it's embedded; under model_b it's still pending.
        let pending_a = find_unembedded_thoughts(&pool, &model_a, None, 10).await.unwrap();
        let pending_b = find_unembedded_thoughts(&pool, &model_b, None, 10).await.unwrap();
        assert!(pending_a.iter().all(|x| x.id != t));
        assert!(pending_b.iter().any(|x| x.id == t));
    }

    // -- M2 Phase B: pending_embeddings queue --------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_embedding_inserts_row(pool: PgPool) {
        let id = insert_test_thought(&pool, "queue me", "global").await;
        let inserted = enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        assert!(inserted);
        assert_eq!(count_pending(&pool).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_embedding_is_idempotent(pool: PgPool) {
        let id = insert_test_thought(&pool, "queue me", "global").await;
        let first = enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        let second = enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        assert!(first);
        assert!(!second, "second enqueue must be a no-op");
        assert_eq!(count_pending(&pool).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn claim_pending_returns_oldest_first_and_bumps_attempts(pool: PgPool) {
        let id_a = insert_test_thought(&pool, "a", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id_a.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        // Sleep so `enqueued_at` is comfortably different across the two
        // auto-commit transactions; 50ms is well above any timer-resolution
        // surprises on the dev machine.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let id_b = insert_test_thought(&pool, "b", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id_b.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();

        let first_batch = claim_pending(&pool, 1).await.unwrap();
        assert_eq!(first_batch.len(), 1);
        assert_eq!(first_batch[0].target_id, id_a.into_uuid());
        assert_eq!(first_batch[0].attempts, 1, "first claim bumps attempts 0→1");

        // Worker finishes the first job before claiming the next batch.
        // Otherwise the still-present row a (oldest) would be re-claimed.
        mark_embedded(&pool, first_batch[0].id).await.unwrap();

        let second_batch = claim_pending(&pool, 1).await.unwrap();
        assert_eq!(second_batch.len(), 1);
        assert_eq!(second_batch[0].target_id, id_b.into_uuid());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn claim_pending_skips_locked_rows(pool: PgPool) {
        let id_a = insert_test_thought(&pool, "a", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id_a.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        let id_b = insert_test_thought(&pool, "b", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id_b.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();

        // Hold a row-level lock on whichever row enqueued first (a) from a
        // separate transaction. From another connection, claim_pending must
        // skip past it and return b.
        let mut tx = pool.begin().await.unwrap();
        let _ = sqlx::query!(
            r#"
            SELECT id FROM pending_embeddings
            WHERE target_id = $1
            FOR UPDATE
            "#,
            id_a.into_uuid(),
        )
        .fetch_one(&mut *tx)
        .await
        .unwrap();

        let claimed = claim_pending(&pool, 10).await.unwrap();
        assert_eq!(claimed.len(), 1, "must skip the locked row");
        assert_eq!(claimed[0].target_id, id_b.into_uuid());

        // Releasing the tx lets future claims see row a again.
        drop(tx);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn mark_embedded_removes_row(pool: PgPool) {
        let id = insert_test_thought(&pool, "x", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        let job = claim_pending(&pool, 10).await.unwrap().pop().unwrap();
        mark_embedded(&pool, job.id).await.unwrap();
        assert_eq!(count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn mark_failed_records_error_but_keeps_row(pool: PgPool) {
        let id = insert_test_thought(&pool, "x", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();
        let job = claim_pending(&pool, 10).await.unwrap().pop().unwrap();
        mark_failed(&pool, job.id, "embedder unreachable").await.unwrap();

        assert_eq!(count_pending(&pool).await.unwrap(), 1);
        let row = sqlx::query!(
            r#"SELECT attempts, last_error FROM pending_embeddings WHERE id = $1"#,
            job.id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.attempts, 1, "claim already bumped to 1; mark_failed does not bump");
        assert_eq!(row.last_error.as_deref(), Some("embedder unreachable"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn count_pending_returns_queue_depth(pool: PgPool) {
        assert_eq!(count_pending(&pool).await.unwrap(), 0);
        for i in 0..3 {
            let id = insert_test_thought(&pool, &format!("t{i}"), "global").await;
            enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
                .await
                .unwrap();
        }
        assert_eq!(count_pending(&pool).await.unwrap(), 3);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_unembedded_thoughts_heals_gaps_and_skips_duplicates(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let already_embedded = insert_test_thought(&pool, "done", "global").await;
        insert_thought_embedding(
            &pool,
            already_embedded,
            &Embedding::new(model.clone(), vec![0.0_f32; 1024]).unwrap(),
        )
        .await
        .unwrap();

        let already_queued = insert_test_thought(&pool, "queued", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, already_queued.into_uuid(), &model.id)
            .await
            .unwrap();

        let orphan = insert_test_thought(&pool, "orphan", "global").await;

        // Heal: should enqueue only the orphan (skips embedded + already-queued).
        let inserted = enqueue_unembedded_thoughts(&pool, &model.id, None, 100)
            .await
            .unwrap();
        assert_eq!(inserted, 1);
        assert_eq!(count_pending(&pool).await.unwrap(), 2);

        // Verify it's the orphan that landed in the queue alongside `already_queued`.
        let claimed = claim_pending(&pool, 10).await.unwrap();
        let target_ids: Vec<Uuid> = claimed.iter().map(|j| j.target_id).collect();
        assert!(target_ids.contains(&orphan.into_uuid()));
        assert!(target_ids.contains(&already_queued.into_uuid()));
        assert!(!target_ids.contains(&already_embedded.into_uuid()));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_embedding_is_idempotent_under_replay(pool: PgPool) {
        // Regression test: simulate the worker crashing between
        // insert_thought_embedding and mark_embedded, then re-inserting on
        // the next claim. The ON CONFLICT DO NOTHING means the duplicate is
        // a no-op rather than a UNIQUE-violation error.
        let id = insert_test_thought(&pool, "replay me", "global").await;
        let model = EmbeddingModel::bge_m3();
        let emb = Embedding::new(model.clone(), vec![0.5_f32; 1024]).unwrap();

        insert_thought_embedding(&pool, id, &emb).await.unwrap();
        insert_thought_embedding(&pool, id, &emb)
            .await
            .expect("second insert must be a no-op, not a UNIQUE violation");

        assert!(thought_has_embedding(&pool, id, &model).await.unwrap());
    }
}

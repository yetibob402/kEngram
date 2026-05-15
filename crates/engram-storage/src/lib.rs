//! engram-storage: sqlx-backed repository functions.
//!
//! The `Embedder` trait is the only place we hide a backend choice behind a
//! trait — storage is concrete sqlx + Postgres. CLAUDE.md rule: compile-time
//! `sqlx::query!` everywhere except where pgvector's vector binding gets in
//! the way of the macro (currently: only `insert_embedding`).

use engram_core::{
    Embedding, EmbeddingModel, EmbeddingStatus, Fact, Hit, Metadata, Scope, ScopeError, Source,
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

/// Convenience: insert an embedding tied to a fact. Facts have no newtype
/// id (unlike thoughts), so this takes a raw `Uuid`. Same ON CONFLICT DO
/// NOTHING idempotency under M2-style worker replay.
pub async fn insert_fact_embedding(
    pool: &PgPool,
    fact_id: Uuid,
    embedding: &Embedding,
) -> Result<(), StorageError> {
    insert_embedding(
        pool,
        target::FACT,
        fact_id,
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
    /// `Some(_)` when the operator has marked this thought as untrusted via
    /// `retract_thought`. Retracted thoughts are excluded from retrieval
    /// (`search_thoughts`, `recent_thoughts`, `search_facts`) and from the
    /// reflector's extraction set (`find_unfacted_thoughts`,
    /// `find_facted_thoughts`); their derived facts are auto-superseded as
    /// part of the retraction tx.
    pub retracted_at: Option<OffsetDateTime>,
    pub retracted_reason: Option<String>,
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
               t.retracted_at, t.retracted_reason,
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
        retracted_at: r.retracted_at,
        retracted_reason: r.retracted_reason,
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
          AND retracted_at IS NULL
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
          AND retracted_at IS NULL
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
          AND t.retracted_at IS NULL
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
          AND t.retracted_at IS NULL
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

/// Heal-side companion to `enqueue_unembedded_thoughts`: enqueues any
/// `facts` row that lacks an `embeddings` entry for `model_id`. Skips
/// superseded facts and facts whose source thought has been retracted —
/// the same active-row invariants that govern retrieval.
///
/// Used by `engram embed-backfill --target facts` to catch pre-M3-Phase-B
/// facts (committed before the fact-embedding seam existed) and any fact
/// that slipped through a crash between `insert_fact` and `enqueue_embedding`.
pub async fn enqueue_unembedded_facts(
    pool: &PgPool,
    model_id: &str,
    scope: Option<&str>,
    limit: i64,
) -> Result<usize, StorageError> {
    let result = sqlx::query!(
        r#"
        INSERT INTO pending_embeddings (target_kind, target_id, model_id)
        SELECT 'fact', f.id, $1
        FROM facts f
        JOIN thoughts t ON t.id = f.source_thought_id
        LEFT JOIN embeddings e
            ON e.target_kind = 'fact'
           AND e.target_id = f.id
           AND e.model_id = $1
        WHERE e.id IS NULL
          AND f.superseded_at IS NULL
          AND t.retracted_at IS NULL
          AND ($2::text IS NULL OR f.scope = $2)
        ORDER BY f.created_at ASC
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

// -- M2 Phase C: facts pipeline (reflector_runs, facts, facts_review_queue) -

/// Strongly-typed wrapper around `reflector_runs.id`. Returned by
/// `start_run`, consumed by `finish_run`, embedded in `NewFact.source_run_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub Uuid);

impl RunId {
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    pub fn into_uuid(self) -> Uuid {
        self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Inputs for `insert_fact`. Borrowing keeps the call cheap; the reflector
/// loops over many thoughts and produces many facts per run.
#[derive(Debug, Clone, Copy)]
pub struct NewFact<'a> {
    pub scope: &'a Scope,
    pub statement: &'a str,
    pub subject: Option<&'a str>,
    pub predicate: Option<&'a str>,
    pub object: Option<&'a str>,
    pub source_thought_id: ThoughtId,
    pub extractor_model: &'a str,
    pub extractor_version: i32,
    pub source_run_id: Option<RunId>,
    pub confidence: f32,
}

/// Inputs for `insert_review_queue_row`. Note: review-queue rows don't carry
/// scope — they reference the source thought, which has the scope. The
/// `decision` column defaults to `'pending'` via the schema; this struct
/// doesn't expose it (callers always insert pending rows; reviewers update
/// them later via a separate path that lands in Phase D).
#[derive(Debug, Clone, Copy)]
pub struct NewReviewRow<'a> {
    pub statement: &'a str,
    pub subject: Option<&'a str>,
    pub predicate: Option<&'a str>,
    pub object: Option<&'a str>,
    pub source_thought_id: ThoughtId,
    pub extractor_model: &'a str,
    pub extractor_version: i32,
    pub source_run_id: Option<RunId>,
    pub confidence: f32,
}

/// Open a reflector run. Returns the new `RunId`. `started_at` defaults to
/// NOW(); the counts default to 0 and are bumped by `finish_run`.
pub async fn start_run(
    pool: &PgPool,
    extractor_model: &str,
    extractor_version: i32,
    scope_filter: Option<&str>,
) -> Result<RunId, StorageError> {
    let row = sqlx::query!(
        r#"
        INSERT INTO reflector_runs (extractor_model, extractor_version, scope_filter)
        VALUES ($1, $2, $3)
        RETURNING id
        "#,
        extractor_model,
        extractor_version,
        scope_filter,
    )
    .fetch_one(pool)
    .await?;
    Ok(RunId(row.id))
}

/// Close out a reflector run with final counts. `error` is `Some(_)` only
/// when the run itself errored at the orchestrator level (per-thought
/// extractor failures are counted via `n_thoughts_processed` minus committed
/// + review, and don't populate `error`).
pub async fn finish_run(
    pool: &PgPool,
    run_id: RunId,
    n_processed: i32,
    n_committed: i32,
    n_review: i32,
    n_failures: i32,
    error: Option<&str>,
) -> Result<(), StorageError> {
    sqlx::query!(
        r#"
        UPDATE reflector_runs
        SET finished_at = NOW(),
            n_thoughts_processed = $2,
            n_facts_committed = $3,
            n_review_queue = $4,
            n_extractor_failures = $5,
            error = $6
        WHERE id = $1
        "#,
        run_id.0,
        n_processed,
        n_committed,
        n_review,
        n_failures,
        error,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Find thoughts that don't yet have any row in `facts` pointing at them.
/// Oldest first — reflector should drain the backlog FIFO.
///
/// A thought whose facts have all been *superseded* still has rows in
/// `facts` (with `superseded_at` set), so it's correctly excluded by this
/// query — re-extracting a corrected thought would defeat the operator's
/// correction. Phase D's `engram reflect --rerun` will use a different
/// query for the explicit-rerun case.
pub async fn find_unfacted_thoughts(
    pool: &PgPool,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata
        FROM thoughts t
        LEFT JOIN facts f
            ON f.source_thought_id = t.id
        WHERE f.id IS NULL
          AND ($1::text IS NULL OR t.scope = $1)
          AND t.retracted_at IS NULL
        ORDER BY t.created_at ASC
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

/// Insert a committed fact. Returns the new row id.
pub async fn insert_fact(pool: &PgPool, f: NewFact<'_>) -> Result<Uuid, StorageError> {
    let row = sqlx::query!(
        r#"
        INSERT INTO facts (
            scope, statement, subject, predicate, object,
            source_thought_id, extractor_model, extractor_version,
            source_run_id, confidence
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id
        "#,
        f.scope.as_str(),
        f.statement,
        f.subject,
        f.predicate,
        f.object,
        f.source_thought_id.into_uuid(),
        f.extractor_model,
        f.extractor_version,
        f.source_run_id.map(|r| r.0),
        f.confidence,
    )
    .fetch_one(pool)
    .await?;
    Ok(row.id)
}

/// Insert a low-confidence extraction into `facts_review_queue` for operator
/// review. The `decision` column defaults to `'pending'` via the schema.
pub async fn insert_review_queue_row(
    pool: &PgPool,
    r: NewReviewRow<'_>,
) -> Result<Uuid, StorageError> {
    let row = sqlx::query!(
        r#"
        INSERT INTO facts_review_queue (
            statement, subject, predicate, object,
            source_thought_id, extractor_model, extractor_version,
            source_run_id, confidence
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        "#,
        r.statement,
        r.subject,
        r.predicate,
        r.object,
        r.source_thought_id.into_uuid(),
        r.extractor_model,
        r.extractor_version,
        r.source_run_id.map(|x| x.0),
        r.confidence,
    )
    .fetch_one(pool)
    .await?;
    Ok(row.id)
}

// -- M2 Phase D: facts read surface (search, fetch, supersede, rerun) -------

/// A trigram-search hit on `facts`, enriched with source-thought fields per
/// m2-facts-pipeline.md Q12 (the agent should be able to make sense of a
/// fact without a follow-up `get_thought` call).
#[derive(Debug, Clone)]
pub struct FactHit {
    pub fact: Fact,
    pub source_thought_content: String,
    pub source_thought_scope: Scope,
    pub source_thought_created_at: OffsetDateTime,
    pub score: f32,
}

// Column unpacking helper shared across the read-side facts queries. The
// argument list mirrors the SELECT order; clippy's "too_many_arguments"
// lint fires here, but adding a row struct would be net-negative ceremony.
#[allow(clippy::too_many_arguments)]
fn fact_from_columns(
    id: Uuid,
    scope: String,
    statement: String,
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
    source_thought_id: Uuid,
    extractor_model: String,
    extractor_version: i32,
    source_run_id: Option<Uuid>,
    confidence: f32,
    created_at: OffsetDateTime,
) -> Result<Fact, StorageError> {
    Ok(Fact {
        id,
        scope: Scope::new(scope)?,
        statement,
        subject,
        predicate,
        object,
        source_thought_id: ThoughtId::from(source_thought_id),
        extractor_model,
        extractor_version,
        source_run_id,
        confidence,
        created_at,
    })
}

/// Trigram-similarity search over `facts.statement`, joined to `thoughts`
/// for source-thought enrichment. Filters `superseded_at IS NULL` (matches
/// the `facts_active_idx` partial index). Same min-similarity threshold
/// (0.1) as `search_trigram` — facts are short, recall matters more than
/// precision at the leg level (RRF fusion handles precision in the
/// orchestrator).
pub async fn search_facts_trigram(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<FactHit>, StorageError> {
    // Lexical scoring concatenates statement + (subject, predicate, object)
    // so a query whose terms only appear in the triple (e.g. `subject="Ron"`
    // on a fact whose statement starts with "When Rust is unavailable…")
    // still matches via the trigram leg. Empty/null triple components fall
    // back to an empty string and contribute nothing to the score.
    //
    // Uses `word_similarity(query, target)` rather than the symmetric
    // `similarity(...)` because `word_similarity` finds the best matching
    // window of trigrams within the target — so a short query like
    // "Ron Go" against a long concatenated text scores by the best window,
    // not by the global trigram-set ratio (which dilutes with target length).
    let rows = sqlx::query!(
        r#"
        WITH searchable AS (
            SELECT f.*,
                   f.statement
                   || ' ' || COALESCE(f.subject, '')
                   || ' ' || COALESCE(f.predicate, '')
                   || ' ' || COALESCE(f.object, '') AS searchable_text
            FROM facts f
            WHERE f.superseded_at IS NULL
              AND ($2::text IS NULL OR f.scope = $2)
        )
        SELECT s.id, s.scope, s.statement, s.subject, s.predicate, s.object,
               s.source_thought_id AS "source_thought_id!",
               s.extractor_model, s.extractor_version, s.source_run_id,
               s.confidence, s.created_at,
               t.content        AS source_thought_content,
               t.scope          AS source_thought_scope,
               t.created_at     AS source_thought_created_at,
               word_similarity($1, s.searchable_text) AS "sim!: f32"
        FROM searchable s
        JOIN thoughts t ON t.id = s.source_thought_id
        WHERE word_similarity($1, s.searchable_text) > 0.3
          AND t.retracted_at IS NULL
        ORDER BY word_similarity($1, s.searchable_text) DESC
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
            let fact = fact_from_columns(
                r.id,
                r.scope,
                r.statement,
                r.subject,
                r.predicate,
                r.object,
                r.source_thought_id,
                r.extractor_model,
                r.extractor_version,
                r.source_run_id,
                r.confidence,
                r.created_at,
            )?;
            Ok(FactHit {
                fact,
                source_thought_content: r.source_thought_content,
                source_thought_scope: Scope::new(r.source_thought_scope)?,
                source_thought_created_at: r.source_thought_created_at,
                score: r.sim,
            })
        })
        .collect()
}

/// All active (non-superseded) facts for a thought, oldest first. Powers
/// `get_thought`'s `linked_facts` field.
pub async fn list_active_facts_for_thought(
    pool: &PgPool,
    thought_id: ThoughtId,
) -> Result<Vec<Fact>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, statement, subject, predicate, object,
               source_thought_id AS "source_thought_id!",
               extractor_model, extractor_version, source_run_id,
               confidence, created_at
        FROM facts
        WHERE source_thought_id = $1
          AND superseded_at IS NULL
        ORDER BY created_at ASC
        "#,
        thought_id.into_uuid(),
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            fact_from_columns(
                r.id,
                r.scope,
                r.statement,
                r.subject,
                r.predicate,
                r.object,
                r.source_thought_id,
                r.extractor_model,
                r.extractor_version,
                r.source_run_id,
                r.confidence,
                r.created_at,
            )
        })
        .collect()
}

/// Look up a single fact by id. Returns the full row regardless of
/// supersession state — callers (`correct_fact`) decide what to do with a
/// superseded row.
pub async fn fetch_fact(
    pool: &PgPool,
    fact_id: Uuid,
) -> Result<Option<Fact>, StorageError> {
    let row = sqlx::query!(
        r#"
        SELECT id, scope, statement, subject, predicate, object,
               source_thought_id AS "source_thought_id!",
               extractor_model, extractor_version, source_run_id,
               confidence, created_at, superseded_at
        FROM facts
        WHERE id = $1
        "#,
        fact_id,
    )
    .fetch_optional(pool)
    .await?;

    let Some(r) = row else {
        return Ok(None);
    };
    // Note: we return the row even if superseded; callers inspect their own
    // copy of the fact for state. The `superseded_at` column itself isn't on
    // `Fact` (Phase D doesn't surface it through the read shape) — it's an
    // internal supersession marker.
    let _ = r.superseded_at;
    let fact = fact_from_columns(
        r.id,
        r.scope,
        r.statement,
        r.subject,
        r.predicate,
        r.object,
        r.source_thought_id,
        r.extractor_model,
        r.extractor_version,
        r.source_run_id,
        r.confidence,
        r.created_at,
    )?;
    Ok(Some(fact))
}

/// Mark a fact as superseded. Atomic: the UPDATE only fires when the row
/// is still active (`superseded_at IS NULL`), so a concurrent supersede
/// loses cleanly. Returns `true` if the row was actually superseded;
/// `false` if it was already superseded or doesn't exist.
///
/// `new_fact_id = None` is the "delete-by-supersede" path from
/// m2-facts-pipeline.md — the row stays in `facts` with `superseded_at`
/// set but no replacement pointer.
pub async fn supersede_fact(
    pool: &PgPool,
    old_fact_id: Uuid,
    new_fact_id: Option<Uuid>,
) -> Result<bool, StorageError> {
    let result = sqlx::query!(
        r#"
        UPDATE facts
        SET superseded_by = $2, superseded_at = NOW()
        WHERE id = $1 AND superseded_at IS NULL
        "#,
        old_fact_id,
        new_fact_id,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() == 1)
}

/// Result of `retract_thought`. Distinguishes "actually retracted this row"
/// from "row didn't exist or was already retracted." The `facts_superseded`
/// count is for operator-facing observability ("you retracted N facts as a
/// side effect").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetractThoughtOutcome {
    pub retracted: bool,
    pub facts_superseded: i64,
}

/// Mark a thought as retracted and auto-supersede every active fact derived
/// from it. Atomic: both operations run in a single transaction so a crash
/// between the UPDATEs can't leave the thought retracted with facts still
/// live (or vice versa).
///
/// Idempotent on a row that's already retracted (`retracted: false,
/// facts_superseded: 0`); idempotent on a missing row (same shape). The
/// caller maps that to an operator-facing error string if it wants — the
/// storage layer just reports what it did.
///
/// The auto-supersede side-effect is the dogfood-driven decision: without
/// it, the operator has to retract every derived fact one at a time, and a
/// single missed fact keeps the source thought in the reflector's
/// `find_facted_thoughts` set, which re-extracts under the next rerun.
/// Tying the two together at the storage tx level closes that gap.
pub async fn retract_thought(
    pool: &PgPool,
    thought_id: ThoughtId,
    reason: Option<&str>,
) -> Result<RetractThoughtOutcome, StorageError> {
    let mut tx = pool.begin().await?;

    let updated = sqlx::query!(
        r#"
        UPDATE thoughts
        SET retracted_at = NOW(), retracted_reason = $2
        WHERE id = $1 AND retracted_at IS NULL
        "#,
        thought_id.into_uuid(),
        reason,
    )
    .execute(&mut *tx)
    .await?;

    if updated.rows_affected() == 0 {
        // Either missing or already retracted; either way nothing to do.
        tx.rollback().await?;
        return Ok(RetractThoughtOutcome {
            retracted: false,
            facts_superseded: 0,
        });
    }

    let facts = sqlx::query!(
        r#"
        UPDATE facts
        SET superseded_at = NOW(), superseded_by = NULL
        WHERE source_thought_id = $1 AND superseded_at IS NULL
        "#,
        thought_id.into_uuid(),
    )
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(RetractThoughtOutcome {
        retracted: true,
        facts_superseded: facts.rows_affected() as i64,
    })
}

/// Thoughts that have at least one active (non-superseded) fact. Inverse of
/// `find_unfacted_thoughts`. Used by `engram reflect --rerun` to re-evaluate
/// already-facted thoughts. `since` filters by `thoughts.created_at` if
/// provided.
pub async fn find_facted_thoughts(
    pool: &PgPool,
    scope: Option<&str>,
    since: Option<OffsetDateTime>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT DISTINCT t.id, t.scope, t.content, t.source, t.created_at, t.metadata
        FROM thoughts t
        INNER JOIN facts f
            ON f.source_thought_id = t.id
           AND f.superseded_at IS NULL
        WHERE ($1::text IS NULL OR t.scope = $1)
          AND ($2::timestamptz IS NULL OR t.created_at >= $2)
          AND t.retracted_at IS NULL
        ORDER BY t.created_at ASC
        LIMIT $3
        "#,
        scope,
        since,
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

/// Find the active fact (if any) on this thought whose (subject, predicate,
/// object) triple matches the given values. Uses `IS NOT DISTINCT FROM` so
/// `NULL` vs `NULL` counts as a match (Postgres `=` returns NULL for that
/// case). Used by `--rerun` to decide merge-vs-supersede.
/// Returns the active facts on `thought_id` that match the proposed new fact
/// on either of two predicates:
///
///   1. Exact statement match (`facts.statement = $2`), or
///   2. Triple match via `IS NOT DISTINCT FROM` (NULL-aware) on
///      `(subject, predicate, object)`.
///
/// The reflector rerun loop uses this to fold drift duplicates into a single
/// canonical row: an LLM may produce the same statement with a different
/// (S, P, O) decomposition on a different sampling, and either signal is
/// enough to recognize "the same claim." Multiple rows may match if the
/// audit table is already in a pre-existing duplicated state — callers
/// supersede all of them.
///
/// Ordered by `created_at ASC` so audit consumers see the oldest match first.
pub async fn find_matching_active_facts(
    pool: &PgPool,
    thought_id: ThoughtId,
    statement: &str,
    subject: Option<&str>,
    predicate: Option<&str>,
    object: Option<&str>,
) -> Result<Vec<Fact>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, statement, subject, predicate, object,
               source_thought_id AS "source_thought_id!",
               extractor_model, extractor_version, source_run_id,
               confidence, created_at
        FROM facts
        WHERE source_thought_id = $1
          AND superseded_at IS NULL
          AND (
              statement = $2
              OR (subject   IS NOT DISTINCT FROM $3
              AND predicate IS NOT DISTINCT FROM $4
              AND object    IS NOT DISTINCT FROM $5)
          )
        ORDER BY created_at ASC
        "#,
        thought_id.into_uuid(),
        statement,
        subject,
        predicate,
        object,
    )
    .fetch_all(pool)
    .await?;

    let mut facts = Vec::with_capacity(rows.len());
    for r in rows {
        facts.push(fact_from_columns(
            r.id,
            r.scope,
            r.statement,
            r.subject,
            r.predicate,
            r.object,
            r.source_thought_id,
            r.extractor_model,
            r.extractor_version,
            r.source_run_id,
            r.confidence,
            r.created_at,
        )?);
    }
    Ok(facts)
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
          AND t.retracted_at IS NULL
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

/// kNN over `embeddings` joined with `facts` (and `thoughts` for source-thought
/// enrichment matching `search_facts_trigram`'s response shape). Mirrors
/// `search_vector_knn` for thoughts: per-model HNSW partial index, cosine
/// distance ordering, scope filter, active-only via `superseded_at IS NULL`
/// and `retracted_at IS NULL`.
pub async fn search_facts_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    model: &EmbeddingModel,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<FactHit>, StorageError> {
    let pgv = pgvector::Vector::from(query_vector);

    let rows: Vec<FactVectorSearchRow> = sqlx::query_as(
        r#"
        SELECT f.id            AS fact_id,
               f.scope         AS fact_scope,
               f.statement,
               f.subject,
               f.predicate,
               f.object,
               f.source_thought_id,
               f.extractor_model,
               f.extractor_version,
               f.source_run_id,
               f.confidence,
               f.created_at    AS fact_created_at,
               t.content       AS source_thought_content,
               t.scope         AS source_thought_scope,
               t.created_at    AS source_thought_created_at,
               (e.vector <=> $1) AS distance
        FROM facts f
        JOIN embeddings e ON e.target_kind = 'fact' AND e.target_id = f.id
        JOIN thoughts t ON t.id = f.source_thought_id
        WHERE e.model_id = $2
          AND f.superseded_at IS NULL
          AND t.retracted_at IS NULL
          AND ($3::text IS NULL OR f.scope = $3)
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
            // cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            // (typically [0, 1]). Matches `search_vector_knn`'s convention.
            let score = (1.0 - r.distance) as f32;
            let fact = fact_from_columns(
                r.fact_id,
                r.fact_scope,
                r.statement,
                r.subject,
                r.predicate,
                r.object,
                r.source_thought_id,
                r.extractor_model,
                r.extractor_version,
                r.source_run_id,
                r.confidence,
                r.fact_created_at,
            )?;
            Ok(FactHit {
                fact,
                source_thought_content: r.source_thought_content,
                source_thought_scope: Scope::new(r.source_thought_scope)?,
                source_thought_created_at: r.source_thought_created_at,
                score,
            })
        })
        .collect()
}

#[derive(sqlx::FromRow)]
struct FactVectorSearchRow {
    fact_id: Uuid,
    fact_scope: String,
    statement: String,
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
    source_thought_id: Uuid,
    extractor_model: String,
    extractor_version: i32,
    source_run_id: Option<Uuid>,
    confidence: f32,
    fact_created_at: OffsetDateTime,
    source_thought_content: String,
    source_thought_scope: String,
    source_thought_created_at: OffsetDateTime,
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

    // -- M2 Phase C: reflector_runs + facts + facts_review_queue ------------

    fn new_fact<'a>(
        scope: &'a Scope,
        statement: &'a str,
        thought_id: ThoughtId,
        run_id: RunId,
        confidence: f32,
    ) -> NewFact<'a> {
        NewFact {
            scope,
            statement,
            subject: None,
            predicate: None,
            object: None,
            source_thought_id: thought_id,
            extractor_model: "fake/extractor",
            extractor_version: 1,
            source_run_id: Some(run_id),
            confidence,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn start_run_inserts_row_with_started_at(pool: PgPool) {
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let row = sqlx::query!(
            r#"SELECT started_at, finished_at, extractor_model, extractor_version, scope_filter
               FROM reflector_runs WHERE id = $1"#,
            run_id.0,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        // started_at must be recent; finished_at must still be NULL.
        let drift = (OffsetDateTime::now_utc() - row.started_at).whole_seconds().abs();
        assert!(drift < 10, "started_at drift {drift}s");
        assert!(row.finished_at.is_none());
        assert_eq!(row.extractor_model, "fake/extractor");
        assert_eq!(row.extractor_version, 1);
        assert!(row.scope_filter.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn finish_run_sets_finished_at_and_counts_and_error(pool: PgPool) {
        let run_id = start_run(&pool, "fake/extractor", 1, Some("work")).await.unwrap();
        finish_run(&pool, run_id, 5, 3, 2, 1, Some("partial failure")).await.unwrap();

        let row = sqlx::query!(
            r#"SELECT finished_at, n_thoughts_processed, n_facts_committed,
                      n_review_queue, n_extractor_failures, error
               FROM reflector_runs WHERE id = $1"#,
            run_id.0,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.finished_at.is_some());
        assert_eq!(row.n_thoughts_processed, 5);
        assert_eq!(row.n_facts_committed, 3);
        assert_eq!(row.n_review_queue, 2);
        assert_eq!(row.n_extractor_failures, 1);
        assert_eq!(row.error.as_deref(), Some("partial failure"));
    }

    /// `n_extractor_failures` defaults to 0 (the migration's column default)
    /// so existing reflector_runs rows from pre-0004 schema don't crash any
    /// reader. New rows written by post-M3-Phase-A finish_run propagate the
    /// reflector's observed failure count.
    #[sqlx::test(migrations = "../../migrations")]
    async fn finish_run_persists_n_extractor_failures(pool: PgPool) {
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        finish_run(&pool, run_id, 10, 4, 1, 5, None).await.unwrap();

        let n_failures = sqlx::query_scalar!(
            r#"SELECT n_extractor_failures FROM reflector_runs WHERE id = $1"#,
            run_id.0,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(n_failures, 5);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unfacted_thoughts_returns_thought_without_facts(pool: PgPool) {
        let unfacted = insert_test_thought(&pool, "no facts yet", "global").await;
        let unfacted_too = insert_test_thought(&pool, "also fresh", "global").await;

        let pending = find_unfacted_thoughts(&pool, None, 100).await.unwrap();
        let ids: Vec<ThoughtId> = pending.iter().map(|t| t.id).collect();
        assert!(ids.contains(&unfacted));
        assert!(ids.contains(&unfacted_too));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unfacted_thoughts_skips_thought_with_existing_fact(pool: PgPool) {
        let facted = insert_test_thought(&pool, "already extracted", "global").await;
        let scope = Scope::global();
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_fact(&pool, new_fact(&scope, "a fact", facted, run_id, 0.9))
            .await
            .unwrap();
        let unfacted = insert_test_thought(&pool, "still fresh", "global").await;

        let pending = find_unfacted_thoughts(&pool, None, 100).await.unwrap();
        let ids: Vec<ThoughtId> = pending.iter().map(|t| t.id).collect();
        assert!(ids.contains(&unfacted));
        assert!(!ids.contains(&facted));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unfacted_thoughts_orders_ascending_by_created_at(pool: PgPool) {
        insert_test_thought(&pool, "first", "global").await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        insert_test_thought(&pool, "second", "global").await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        insert_test_thought(&pool, "third", "global").await;

        let pending = find_unfacted_thoughts(&pool, None, 10).await.unwrap();
        assert_eq!(pending.len(), 3);
        assert_eq!(pending[0].content, "first");
        assert_eq!(pending[1].content, "second");
        assert_eq!(pending[2].content, "third");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unfacted_thoughts_respects_scope_and_limit(pool: PgPool) {
        for i in 0..5 {
            insert_test_thought(&pool, &format!("work-{i}"), "work").await;
        }
        for i in 0..3 {
            insert_test_thought(&pool, &format!("personal-{i}"), "personal").await;
        }

        let work = find_unfacted_thoughts(&pool, Some("work"), 100).await.unwrap();
        assert_eq!(work.len(), 5);
        assert!(work.iter().all(|t| t.scope.as_str() == "work"));

        let limited = find_unfacted_thoughts(&pool, None, 4).await.unwrap();
        assert_eq!(limited.len(), 4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_fact_persists_with_provenance(pool: PgPool) {
        let thought_id = insert_test_thought(&pool, "source", "global").await;
        let scope = Scope::global();
        let run_id = start_run(&pool, "fake/extractor", 7, None).await.unwrap();

        let fact_id = insert_fact(
            &pool,
            NewFact {
                scope: &scope,
                statement: "Engram uses pgvector",
                subject: Some("Engram"),
                predicate: Some("uses"),
                object: Some("pgvector"),
                source_thought_id: thought_id,
                extractor_model: "fake/extractor",
                extractor_version: 7,
                source_run_id: Some(run_id),
                confidence: 0.91,
            },
        )
        .await
        .unwrap();

        let row = sqlx::query!(
            r#"SELECT statement, subject, predicate, object, source_thought_id,
                      extractor_model, extractor_version, source_run_id,
                      confidence, scope, superseded_at
               FROM facts WHERE id = $1"#,
            fact_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.statement, "Engram uses pgvector");
        assert_eq!(row.subject.as_deref(), Some("Engram"));
        assert_eq!(row.predicate.as_deref(), Some("uses"));
        assert_eq!(row.object.as_deref(), Some("pgvector"));
        assert_eq!(row.source_thought_id, Some(thought_id.into_uuid()));
        assert_eq!(row.extractor_model, "fake/extractor");
        assert_eq!(row.extractor_version, 7);
        assert_eq!(row.source_run_id, Some(run_id.0));
        assert!((row.confidence - 0.91).abs() < 1e-5);
        assert_eq!(row.scope, "global");
        assert!(row.superseded_at.is_none());
    }

    // -- M2 Phase D: facts read surface --

    async fn insert_active_fact(
        pool: &PgPool,
        thought_id: ThoughtId,
        scope: &Scope,
        statement: &str,
        triple: (Option<&str>, Option<&str>, Option<&str>),
        run_id: RunId,
        confidence: f32,
    ) -> Uuid {
        insert_fact(
            pool,
            NewFact {
                scope,
                statement,
                subject: triple.0,
                predicate: triple.1,
                object: triple.2,
                source_thought_id: thought_id,
                extractor_model: "fake/extractor",
                extractor_version: 1,
                source_run_id: Some(run_id),
                confidence,
            },
        )
        .await
        .unwrap()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_trigram_finds_match_and_returns_source_thought_content(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "Engram uses pgvector for storage", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "Engram uses pgvector",
            (Some("Engram"), Some("uses"), Some("pgvector")),
            run_id,
            0.9,
        )
        .await;

        let hits = search_facts_trigram(&pool, "pgvector", None, 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].fact.statement, "Engram uses pgvector");
        assert_eq!(hits[0].source_thought_content, "Engram uses pgvector for storage");
        assert_eq!(hits[0].source_thought_scope, scope);
        assert!(hits[0].score > 0.0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_trigram_filters_superseded(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "facts about widgets", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_id = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "widgets are useful",
            (None, None, None),
            run_id,
            0.9,
        )
        .await;

        // Visible before supersede.
        let before = search_facts_trigram(&pool, "widgets", None, 10).await.unwrap();
        assert_eq!(before.len(), 1);

        // Supersede with no replacement.
        let did = supersede_fact(&pool, fact_id, None).await.unwrap();
        assert!(did);

        // Filtered out after.
        let after = search_facts_trigram(&pool, "widgets", None, 10).await.unwrap();
        assert!(after.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_trigram_respects_scope_and_limit(pool: PgPool) {
        let work = Scope::new("work").unwrap();
        let personal = Scope::new("personal").unwrap();
        let t1 = insert_test_thought(&pool, "work thought one", "work").await;
        let t2 = insert_test_thought(&pool, "work thought two", "work").await;
        let t3 = insert_test_thought(&pool, "personal thought", "personal").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(&pool, t1, &work, "widget alpha", (None, None, None), run_id, 0.9).await;
        insert_active_fact(&pool, t2, &work, "widget beta", (None, None, None), run_id, 0.9).await;
        insert_active_fact(&pool, t3, &personal, "widget gamma", (None, None, None), run_id, 0.9).await;

        let work_only = search_facts_trigram(&pool, "widget", Some("work"), 10).await.unwrap();
        assert_eq!(work_only.len(), 2);
        assert!(work_only.iter().all(|h| h.fact.scope.as_str() == "work"));

        let limited = search_facts_trigram(&pool, "widget", None, 2).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    /// M3 Phase A: lexical scoring now consults `subject || predicate || object`
    /// in addition to `statement`. A fact whose subject is "Ron" but whose
    /// statement starts "When Rust is unavailable…" should match a search
    /// for "Ron Go" via the trigram leg.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_trigram_matches_via_triple_when_statement_does_not_mention_query(
        pool: PgPool,
    ) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "language preferences", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "When Rust is not available or appropriate, Go is the next choice.",
            (Some("Ron"), Some("prefers as fallback"), Some("Go")),
            run_id,
            0.9,
        )
        .await;
        // A decoy fact with no Ron / Go content.
        insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "JavaScript is widely deployed in browsers.",
            (Some("JavaScript"), Some("is deployed in"), Some("browsers")),
            run_id,
            0.9,
        )
        .await;

        let hits = search_facts_trigram(&pool, "Ron Go", None, 10).await.unwrap();
        assert!(!hits.is_empty(), "trigram leg should match via triple");
        assert!(
            hits[0]
                .fact
                .statement
                .contains("Rust is not available"),
            "expected Ron/Go fact to be the top hit, got: {}",
            hits[0].fact.statement
        );
    }

    // -- M3 Phase B step 1: fact embeddings -----------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_fact_embedding_persists_vector_for_model(pool: PgPool) {
        let model = EmbeddingModel::new("test:1024", 1024);
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_id = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "fact statement",
            (Some("S"), Some("P"), Some("O")),
            run_id,
            0.9,
        )
        .await;

        let v = unit_vector_1024(7);
        insert_fact_embedding(&pool, fact_id, &Embedding::new(model.clone(), v).unwrap())
            .await
            .unwrap();

        // Verify the row landed under target_kind='fact'.
        let n = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM embeddings
               WHERE target_kind = 'fact' AND target_id = $1 AND model_id = $2"#,
            fact_id,
            model.id,
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(n, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_vector_knn_finds_inserted_vector(pool: PgPool) {
        let model = EmbeddingModel::new("test:1024", 1024);
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();

        let fact_a = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "alpha fact",
            (Some("A"), None, None),
            run_id,
            0.9,
        )
        .await;
        let fact_b = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "beta fact",
            (Some("B"), None, None),
            run_id,
            0.9,
        )
        .await;

        let va = unit_vector_1024(0);
        let vb = unit_vector_1024(1);
        insert_fact_embedding(&pool, fact_a, &Embedding::new(model.clone(), va.clone()).unwrap())
            .await
            .unwrap();
        insert_fact_embedding(&pool, fact_b, &Embedding::new(model.clone(), vb).unwrap())
            .await
            .unwrap();

        // Query with the exact 'a' vector → fact_a ranks first with score ≈ 1.
        let hits = search_facts_vector_knn(&pool, va, &model, None, 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].fact.id, fact_a);
        assert!((hits[0].score - 1.0).abs() < 1e-4);
        // Source-thought enrichment fields populated.
        assert_eq!(hits[0].source_thought_content, "anchor");
        assert_eq!(hits[0].source_thought_scope.as_str(), "global");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_unembedded_facts_skips_already_embedded(pool: PgPool) {
        let model = EmbeddingModel::new("bge-m3:1024", 1024);
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_a = insert_active_fact(
            &pool, thought_id, &scope, "fact-a", (None, None, None), run_id, 0.9,
        )
        .await;
        let _fact_b = insert_active_fact(
            &pool, thought_id, &scope, "fact-b", (None, None, None), run_id, 0.9,
        )
        .await;

        // Pre-embed fact_a only; it should be skipped by the heal.
        insert_fact_embedding(
            &pool,
            fact_a,
            &Embedding::new(model.clone(), unit_vector_1024(0)).unwrap(),
        )
        .await
        .unwrap();

        let healed = enqueue_unembedded_facts(&pool, &model.id, None, 100)
            .await
            .unwrap();
        assert_eq!(healed, 1, "fact_a was already embedded; only fact_b enqueues");

        let pending = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM pending_embeddings WHERE target_kind = 'fact'"#
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(pending, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_active_facts_for_thought_returns_only_active(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let active = insert_active_fact(
            &pool, thought_id, &scope, "still alive", (None, None, None), run_id, 0.9,
        )
        .await;
        let _other = insert_active_fact(
            &pool, thought_id, &scope, "also alive", (None, None, None), run_id, 0.9,
        )
        .await;
        let gone = insert_active_fact(
            &pool, thought_id, &scope, "to be superseded", (None, None, None), run_id, 0.9,
        )
        .await;
        supersede_fact(&pool, gone, None).await.unwrap();

        let facts = list_active_facts_for_thought(&pool, thought_id).await.unwrap();
        assert_eq!(facts.len(), 2);
        let ids: Vec<Uuid> = facts.iter().map(|f| f.id).collect();
        assert!(ids.contains(&active));
        assert!(!ids.contains(&gone));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_fact_returns_none_for_missing(pool: PgPool) {
        let id = Uuid::new_v4();
        let result = fetch_fact(&pool, id).await.unwrap();
        assert!(result.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_fact_returns_row_with_provenance(pool: PgPool) {
        let scope = Scope::new("work").unwrap();
        let thought_id = insert_test_thought(&pool, "anchor", "work").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_id = insert_active_fact(
            &pool, thought_id, &scope, "fetched", (Some("S"), Some("P"), Some("O")), run_id, 0.91,
        )
        .await;

        let fact = fetch_fact(&pool, fact_id).await.unwrap().expect("must exist");
        assert_eq!(fact.id, fact_id);
        assert_eq!(fact.statement, "fetched");
        assert_eq!(fact.subject.as_deref(), Some("S"));
        assert_eq!(fact.source_thought_id, thought_id);
        assert_eq!(fact.source_run_id, Some(run_id.0));
        assert!((fact.confidence - 0.91).abs() < 1e-5);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_fact_with_replacement_sets_columns(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let old = insert_active_fact(
            &pool, thought_id, &scope, "old", (None, None, None), run_id, 0.9,
        )
        .await;
        let new = insert_active_fact(
            &pool, thought_id, &scope, "new", (None, None, None), run_id, 0.9,
        )
        .await;

        let did = supersede_fact(&pool, old, Some(new)).await.unwrap();
        assert!(did);

        let row = sqlx::query!(
            r#"SELECT superseded_by, superseded_at FROM facts WHERE id = $1"#,
            old,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.superseded_by, Some(new));
        assert!(row.superseded_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_fact_without_replacement_sets_only_timestamp(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_id = insert_active_fact(
            &pool, thought_id, &scope, "doomed", (None, None, None), run_id, 0.9,
        )
        .await;

        let did = supersede_fact(&pool, fact_id, None).await.unwrap();
        assert!(did);

        let row = sqlx::query!(
            r#"SELECT superseded_by, superseded_at FROM facts WHERE id = $1"#,
            fact_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.superseded_by.is_none());
        assert!(row.superseded_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_fact_on_already_superseded_returns_false(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_id = insert_active_fact(
            &pool, thought_id, &scope, "loser", (None, None, None), run_id, 0.9,
        )
        .await;

        assert!(supersede_fact(&pool, fact_id, None).await.unwrap());
        assert!(!supersede_fact(&pool, fact_id, None).await.unwrap(),
                "second supersede must report no-change");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_facted_thoughts_returns_only_thoughts_with_active_facts(pool: PgPool) {
        let scope = Scope::global();
        let with_fact = insert_test_thought(&pool, "with fact", "global").await;
        let _no_fact = insert_test_thought(&pool, "no fact", "global").await;
        let only_superseded = insert_test_thought(&pool, "only superseded", "global").await;

        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(&pool, with_fact, &scope, "active", (None, None, None), run_id, 0.9).await;
        let gone = insert_active_fact(
            &pool, only_superseded, &scope, "gone", (None, None, None), run_id, 0.9,
        )
        .await;
        supersede_fact(&pool, gone, None).await.unwrap();

        let facted = find_facted_thoughts(&pool, None, None, 100).await.unwrap();
        let ids: Vec<ThoughtId> = facted.iter().map(|t| t.id).collect();
        assert!(ids.contains(&with_fact));
        assert!(!ids.contains(&only_superseded), "thought with only superseded facts must not appear");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_facted_thoughts_respects_since_scope_limit(pool: PgPool) {
        let work = Scope::new("work").unwrap();
        let personal = Scope::new("personal").unwrap();
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();

        let t_work = insert_test_thought(&pool, "work one", "work").await;
        let t_personal = insert_test_thought(&pool, "personal one", "personal").await;
        insert_active_fact(&pool, t_work, &work, "wf", (None, None, None), run_id, 0.9).await;
        insert_active_fact(&pool, t_personal, &personal, "pf", (None, None, None), run_id, 0.9).await;

        // Scope filter.
        let work_only = find_facted_thoughts(&pool, Some("work"), None, 10).await.unwrap();
        assert_eq!(work_only.len(), 1);
        assert_eq!(work_only[0].id, t_work);

        // Since-in-future filter eliminates everything.
        let future = OffsetDateTime::now_utc() + time::Duration::days(1);
        let empty = find_facted_thoughts(&pool, None, Some(future), 10).await.unwrap();
        assert!(empty.is_empty());

        // Since-in-past keeps both.
        let past = OffsetDateTime::now_utc() - time::Duration::days(1);
        let both = find_facted_thoughts(&pool, None, Some(past), 10).await.unwrap();
        assert_eq!(both.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_matching_active_facts_handles_null_via_is_not_distinct_from(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        // Insert a fact with all-null triple.
        let null_fact = insert_active_fact(
            &pool, thought_id, &scope, "no triple", (None, None, None), run_id, 0.9,
        )
        .await;
        // And one with a triple.
        insert_active_fact(
            &pool, thought_id, &scope, "with triple", (Some("S"), Some("P"), Some("O")), run_id, 0.9,
        )
        .await;

        // Statement-or-triple OR predicate with a null-triple probe matches
        // the null-triple row by the triple branch (Postgres `=` would return
        // NULL on (NULL, NULL, NULL); `IS NOT DISTINCT FROM` is what makes
        // it work).
        let matches =
            find_matching_active_facts(&pool, thought_id, "unrelated statement", None, None, None)
                .await
                .unwrap();
        assert_eq!(matches.len(), 1, "null triple should match exactly one row");
        assert_eq!(matches[0].id, null_fact);

        // Exact triple match.
        let triple_match = find_matching_active_facts(
            &pool, thought_id, "unrelated statement", Some("S"), Some("P"), Some("O"),
        )
        .await
        .unwrap();
        assert_eq!(triple_match.len(), 1);
        assert_eq!(triple_match[0].statement, "with triple");

        // No match for an unrelated triple AND an unrelated statement.
        let none = find_matching_active_facts(
            &pool, thought_id, "still unrelated", Some("X"), Some("Y"), Some("Z"),
        )
        .await
        .unwrap();
        assert!(none.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_matching_active_facts_skips_superseded(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let fact_id = insert_active_fact(
            &pool, thought_id, &scope, "gone", (Some("S"), Some("P"), Some("O")), run_id, 0.9,
        )
        .await;
        supersede_fact(&pool, fact_id, None).await.unwrap();

        let result = find_matching_active_facts(
            &pool, thought_id, "gone", Some("S"), Some("P"), Some("O"),
        )
        .await
        .unwrap();
        assert!(
            result.is_empty(),
            "superseded match must not be returned"
        );
    }

    /// The M2 dogfood failure case: v1 fact has statement S, triple (a, b, c);
    /// v2 extraction produces the same statement S but triple (x, y, z).
    /// (S, P, O) match is empty; statement match catches it.
    #[sqlx::test(migrations = "../../migrations")]
    async fn find_matching_active_facts_matches_on_statement_alone_when_triple_differs(
        pool: PgPool,
    ) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let v1 = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "current API surface is append-only",
            (Some("current API surface"), Some("is"), Some("append-only")),
            run_id,
            0.9,
        )
        .await;

        // v2 probe: same statement, different decomposition.
        let matches = find_matching_active_facts(
            &pool,
            thought_id,
            "current API surface is append-only",
            Some("thoughts in current API surface"),
            Some("are"),
            Some("append-only"),
        )
        .await
        .unwrap();
        assert_eq!(matches.len(), 1, "statement match must catch the v1 row");
        assert_eq!(matches[0].id, v1);
    }

    /// The prior intended case: v1 has triple T and statement S; v2 produces
    /// the same triple T but a different statement S'. Triple match catches it.
    #[sqlx::test(migrations = "../../migrations")]
    async fn find_matching_active_facts_matches_on_triple_alone_when_statement_differs(
        pool: PgPool,
    ) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        let v1 = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "old wording",
            (Some("S"), Some("P"), Some("O")),
            run_id,
            0.9,
        )
        .await;

        let matches = find_matching_active_facts(
            &pool,
            thought_id,
            "rephrased wording",
            Some("S"),
            Some("P"),
            Some("O"),
        )
        .await
        .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, v1);
    }

    /// Pre-existing audit-corrupt state: two parallel-active rows that both
    /// match the new probe. Both should come back so the caller can fold
    /// them into one canonical row.
    #[sqlx::test(migrations = "../../migrations")]
    async fn find_matching_active_facts_returns_multiple_when_audit_already_corrupt(
        pool: PgPool,
    ) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        // Two pre-existing actives: one matches by statement, the other by triple.
        let a = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "shared statement",
            (Some("a"), Some("b"), Some("c")),
            run_id,
            0.9,
        )
        .await;
        let b = insert_active_fact(
            &pool,
            thought_id,
            &scope,
            "different wording",
            (Some("S"), Some("P"), Some("O")),
            run_id,
            0.9,
        )
        .await;

        // Probe matches `a` by statement and `b` by triple.
        let matches = find_matching_active_facts(
            &pool,
            thought_id,
            "shared statement",
            Some("S"),
            Some("P"),
            Some("O"),
        )
        .await
        .unwrap();
        let ids: Vec<_> = matches.iter().map(|f| f.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_review_queue_row_defaults_decision_to_pending(pool: PgPool) {
        let thought_id = insert_test_thought(&pool, "questionable claim", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();

        let row_id = insert_review_queue_row(
            &pool,
            NewReviewRow {
                statement: "weak fact",
                subject: None,
                predicate: None,
                object: None,
                source_thought_id: thought_id,
                extractor_model: "fake/extractor",
                extractor_version: 1,
                source_run_id: Some(run_id),
                confidence: 0.3,
            },
        )
        .await
        .unwrap();

        let row = sqlx::query!(
            r#"SELECT statement, decision, confidence, reviewed_at, source_run_id
               FROM facts_review_queue WHERE id = $1"#,
            row_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.statement, "weak fact");
        assert_eq!(row.decision, "pending");
        assert!((row.confidence - 0.3).abs() < 1e-5);
        assert!(row.reviewed_at.is_none());
        assert_eq!(row.source_run_id, Some(run_id.0));
    }

    // -- M3 starter: thought retraction ----------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_sets_retracted_at_and_supersedes_facts(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "wrong claim", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(&pool, thought_id, &scope, "f1", (None, None, None), run_id, 0.9).await;
        insert_active_fact(&pool, thought_id, &scope, "f2", (None, None, None), run_id, 0.9).await;
        insert_active_fact(&pool, thought_id, &scope, "f3", (None, None, None), run_id, 0.9).await;

        let outcome = retract_thought(&pool, thought_id, Some("operator error"))
            .await
            .unwrap();
        assert!(outcome.retracted);
        assert_eq!(outcome.facts_superseded, 3);

        let row = sqlx::query!(
            r#"SELECT retracted_at, retracted_reason FROM thoughts WHERE id = $1"#,
            thought_id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.retracted_at.is_some());
        assert_eq!(row.retracted_reason.as_deref(), Some("operator error"));

        let active_facts = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM facts
               WHERE source_thought_id = $1 AND superseded_at IS NULL"#,
            thought_id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(active_facts, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_is_idempotent_on_already_retracted(pool: PgPool) {
        let thought_id = insert_test_thought(&pool, "wrong", "global").await;
        let first = retract_thought(&pool, thought_id, None).await.unwrap();
        assert!(first.retracted);

        let second = retract_thought(&pool, thought_id, Some("ignored")).await.unwrap();
        assert!(!second.retracted);
        assert_eq!(second.facts_superseded, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_on_missing_id_reports_no_op(pool: PgPool) {
        let outcome = retract_thought(&pool, ThoughtId::new(), None).await.unwrap();
        assert!(!outcome.retracted);
        assert_eq!(outcome.facts_superseded, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_recent_thoughts(pool: PgPool) {
        let kept = insert_test_thought(&pool, "kept", "global").await;
        let gone = insert_test_thought(&pool, "gone", "global").await;
        retract_thought(&pool, gone, None).await.unwrap();

        let results = recent_thoughts(&pool, None, 10).await.unwrap();
        let ids: Vec<ThoughtId> = results.iter().map(|t| t.id).collect();
        assert!(ids.contains(&kept));
        assert!(!ids.contains(&gone), "retracted thought must not appear");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_search_trigram(pool: PgPool) {
        let kept = insert_test_thought(&pool, "kept widget alpha", "global").await;
        let gone = insert_test_thought(&pool, "gone widget alpha", "global").await;
        retract_thought(&pool, gone, None).await.unwrap();

        let hits = search_trigram(&pool, "widget", None, 10).await.unwrap();
        let ids: Vec<ThoughtId> = hits.iter().map(|h| h.thought.id).collect();
        assert!(ids.contains(&kept));
        assert!(!ids.contains(&gone));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_find_unfacted_thoughts(pool: PgPool) {
        let kept = insert_test_thought(&pool, "kept", "global").await;
        let gone = insert_test_thought(&pool, "gone", "global").await;
        retract_thought(&pool, gone, None).await.unwrap();

        let pending = find_unfacted_thoughts(&pool, None, 100).await.unwrap();
        let ids: Vec<ThoughtId> = pending.iter().map(|t| t.id).collect();
        assert!(ids.contains(&kept));
        assert!(!ids.contains(&gone), "reflector must not see retracted thoughts");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_find_facted_thoughts(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(&pool, thought_id, &scope, "f1", (None, None, None), run_id, 0.9).await;

        // Before retraction: visible in find_facted_thoughts.
        let before = find_facted_thoughts(&pool, None, None, 100).await.unwrap();
        assert!(before.iter().any(|t| t.id == thought_id));

        retract_thought(&pool, thought_id, None).await.unwrap();

        // After retraction: gone — because the auto-supersede dropped its
        // active facts, *and* the new retraction filter would have caught
        // it anyway. Belt-and-braces.
        let after = find_facted_thoughts(&pool, None, None, 100).await.unwrap();
        assert!(after.iter().all(|t| t.id != thought_id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excludes_its_facts_from_search_facts_trigram(pool: PgPool) {
        let scope = Scope::global();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;
        let run_id = start_run(&pool, "fake/extractor", 1, None).await.unwrap();
        insert_active_fact(
            &pool, thought_id, &scope, "search me later", (None, None, None), run_id, 0.9,
        )
        .await;

        // Before retraction: fact is searchable.
        let before = search_facts_trigram(&pool, "search me", None, 10).await.unwrap();
        assert_eq!(before.len(), 1);

        retract_thought(&pool, thought_id, None).await.unwrap();

        // After: fact is gone from search_facts (because auto-superseded *and*
        // the JOIN on thoughts now filters retracted).
        let after = search_facts_trigram(&pool, "search me", None, 10).await.unwrap();
        assert!(after.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_with_provenance_surfaces_retracted_at(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let thought_id = insert_test_thought(&pool, "anchor", "global").await;

        let before = fetch_thought_with_provenance(&pool, thought_id, &model)
            .await
            .unwrap()
            .unwrap();
        assert!(before.retracted_at.is_none());
        assert!(before.retracted_reason.is_none());

        retract_thought(&pool, thought_id, Some("test reason"))
            .await
            .unwrap();

        let after = fetch_thought_with_provenance(&pool, thought_id, &model)
            .await
            .unwrap()
            .unwrap();
        assert!(after.retracted_at.is_some());
        assert_eq!(after.retracted_reason.as_deref(), Some("test reason"));
    }
}

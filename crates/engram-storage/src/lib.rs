//! engram-storage: sqlx-backed repository functions.
//!
//! The `Embedder` trait is the only place we hide a backend choice behind a
//! trait — storage is concrete sqlx + Postgres. CLAUDE.md rule: compile-time
//! `sqlx::query!` everywhere except where pgvector's vector binding gets in
//! the way of the macro (currently: only `insert_embedding`).

use engram_core::{
    Embedding, EmbeddingModel, EmbeddingStatus, Hit, LinkDirection, LinkId, LinkSource, Metadata,
    RelationKind, Scope, ScopeError, ScopeVocab, Source, SourceError, Tags, Thought, ThoughtId,
    UnknownLinkSource, UnknownRelationKind,
};
use sqlx::PgPool;
use time::OffsetDateTime;
use uuid::Uuid;

pub mod target {
    //! `embeddings.target_kind` enum-as-string. Matches the CHECK constraint
    //! on the column. The `FACT` value is preserved for migration
    //! reversibility (Path B-OB1 dropped the facts table but left the enum
    //! value in place so we could re-add facts without another schema
    //! migration).
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

    #[error("content_fingerprint length mismatch: expected 32 bytes, got {0}")]
    InvalidFingerprintLength(usize),

    #[error("invalid tags JSON decoded from database: {0}")]
    InvalidTags(#[from] serde_json::Error),

    #[error("invalid relation kind decoded from database: {0}")]
    InvalidRelationKind(#[from] UnknownRelationKind),

    #[error("invalid link source decoded from database: {0}")]
    InvalidLinkSource(#[from] UnknownLinkSource),
}

/// Convert a BYTEA `content_fingerprint` blob from the database into the
/// 32-byte SHA-256 array on `Thought`. Returns `StorageError::InvalidFingerprintLength`
/// if the column somehow held something other than 32 bytes (the migration
/// backfills via `digest(content, 'sha256')` which always produces 32, but
/// the column itself is just BYTEA NOT NULL — no DB-level length check).
fn fingerprint_from_bytes(bytes: Vec<u8>) -> Result<[u8; 32], StorageError> {
    let len = bytes.len();
    bytes
        .try_into()
        .map_err(|_| StorageError::InvalidFingerprintLength(len))
}

/// Decode the `tags` JSONB column into the typed `Tags` struct.
fn tags_from_value(value: serde_json::Value) -> Result<Tags, StorageError> {
    Ok(serde_json::from_value(value)?)
}

/// Inputs for inserting a new thought. Borrowing keeps the call cheap.
/// `content_fingerprint` is the SHA-256 of `content`; callers compute it
/// (the MCP capture layer does this so it can also dedup before round-tripping
/// to the DB).
#[derive(Debug, Clone, Copy)]
pub struct NewThought<'a> {
    pub scope: &'a Scope,
    pub content: &'a str,
    pub source: &'a Source,
    pub metadata: &'a Metadata,
    pub content_fingerprint: [u8; 32],
}

/// What the DB tells us after a thought is inserted.
#[derive(Debug, Clone)]
pub struct InsertedThought {
    pub id: ThoughtId,
    pub created_at: OffsetDateTime,
}

/// Insert a thought. The database generates `id` and `created_at`. Returns
/// `(InsertedThought, is_new)`:
/// - `is_new = true`: a fresh row was inserted; caller should enqueue
///   embedding + tag jobs.
/// - `is_new = false`: a row with the same `content_fingerprint` already
///   exists; the returned `id` + `created_at` belong to that existing row;
///   no new jobs should be enqueued.
///
/// Implementation: `INSERT ... ON CONFLICT (content_fingerprint) DO NOTHING
/// RETURNING ...`. On conflict no row is returned, so we fall through to a
/// SELECT by fingerprint to fetch the existing row.
pub async fn insert_thought(
    pool: &PgPool,
    t: NewThought<'_>,
) -> Result<(InsertedThought, bool), StorageError> {
    let fingerprint: &[u8] = &t.content_fingerprint;
    let inserted = sqlx::query!(
        r#"
        INSERT INTO thoughts (scope, content, source, metadata, content_fingerprint)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (content_fingerprint) DO NOTHING
        RETURNING id, created_at
        "#,
        t.scope.as_str(),
        t.content,
        t.source.as_str(),
        t.metadata.as_value(),
        fingerprint,
    )
    .fetch_optional(pool)
    .await?;

    if let Some(row) = inserted {
        return Ok((
            InsertedThought {
                id: ThoughtId::from(row.id),
                created_at: row.created_at,
            },
            true,
        ));
    }

    // Fingerprint collision: fetch the existing row.
    let existing = sqlx::query!(
        r#"
        SELECT id, created_at
        FROM thoughts
        WHERE content_fingerprint = $1
        "#,
        fingerprint,
    )
    .fetch_one(pool)
    .await?;

    Ok((
        InsertedThought {
            id: ThoughtId::from(existing.id),
            created_at: existing.created_at,
        },
        false,
    ))
}

/// Insert an embedding row tied to some target (thought / artifact_chunk).
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
    .bind(1_i32)
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
pub async fn fetch_thought(pool: &PgPool, id: ThoughtId) -> Result<Option<Thought>, StorageError> {
    let row = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata,
               content_fingerprint, tags,
               tags_extractor_model, tags_extractor_version, tags_extracted_at
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
        content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
        tags: tags_from_value(r.tags)?,
        tags_extractor_model: r.tags_extractor_model,
        tags_extractor_version: r.tags_extractor_version,
        tags_extracted_at: r.tags_extracted_at,
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
    /// (`search_thoughts`, `recent_thoughts`); `get_thought` is the audit
    /// path and continues to return the row regardless of retraction state.
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
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
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
        content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
        tags: tags_from_value(r.tags)?,
        tags_extractor_model: r.tags_extractor_model,
        tags_extractor_version: r.tags_extractor_version,
        tags_extracted_at: r.tags_extracted_at,
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
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata,
               content_fingerprint, tags,
               tags_extractor_model, tags_extractor_version, tags_extracted_at
        FROM thoughts
        WHERE ($1::text IS NULL OR scope = $1)
          AND ($2::text IS NULL OR scope LIKE $2 || '%')
          AND retracted_at IS NULL
        ORDER BY created_at DESC
        LIMIT $3
        "#,
        scope,
        scope_prefix,
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
                content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
                tags: tags_from_value(r.tags)?,
                tags_extractor_model: r.tags_extractor_model,
                tags_extractor_version: r.tags_extractor_version,
                tags_extracted_at: r.tags_extracted_at,
            })
        })
        .collect()
}

/// Per-scope rollup row returned by [`list_scopes`]. Aggregates active
/// (non-retracted) thoughts by scope value and surfaces a count plus the
/// first / last activity timestamps so agents can discover what scopes
/// exist and operators can see scope sprawl at a glance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeSummary {
    pub scope: Scope,
    pub thought_count: i64,
    pub first_activity_at: OffsetDateTime,
    pub last_activity_at: OffsetDateTime,
}

/// Enumerate scopes currently in use, with per-scope counts and activity
/// timestamps. Optional `prefix` matches scopes starting with the given
/// string (e.g., `prefix = Some("rjf.")` matches `rjf.professional.cto`,
/// `rjf.personal.health`, etc.). Retracted thoughts are excluded from
/// counts and from the visible scope set; if every thought in a scope is
/// retracted the scope doesn't appear. Sorted by `last_activity_at`
/// descending (most recently used first).
pub async fn list_scopes(
    pool: &PgPool,
    prefix: Option<&str>,
) -> Result<Vec<ScopeSummary>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT
            scope AS "scope!",
            COUNT(*) AS "thought_count!",
            MIN(created_at) AS "first_activity_at!",
            MAX(created_at) AS "last_activity_at!"
        FROM thoughts
        WHERE retracted_at IS NULL
          AND ($1::text IS NULL OR scope LIKE $1 || '%')
        GROUP BY scope
        ORDER BY MAX(created_at) DESC
        "#,
        prefix,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            Ok(ScopeSummary {
                scope: Scope::new(r.scope)?,
                thought_count: r.thought_count,
                first_activity_at: r.first_activity_at,
                last_activity_at: r.last_activity_at,
            })
        })
        .collect()
}

/// Trigram-similarity search over `thoughts.content`. Hits are returned in
/// descending order of `similarity(content, query)` and filtered to a minimum
/// similarity of 0.1.
pub async fn search_trigram(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata,
               content_fingerprint, tags,
               tags_extractor_model, tags_extractor_version, tags_extracted_at,
               similarity(content, $1) AS "sim!: f32"
        FROM thoughts
        WHERE similarity(content, $1) > 0.1
          AND ($2::text IS NULL OR scope = $2)
          AND ($3::text IS NULL OR scope LIKE $3 || '%')
          AND retracted_at IS NULL
        ORDER BY similarity(content, $1) DESC
        LIMIT $4
        "#,
        query,
        scope,
        scope_prefix,
        limit,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            let thought = Thought {
                id: ThoughtId::from(r.id),
                scope: Scope::new(r.scope)?,
                content: r.content,
                source: Source::new(r.source)?,
                created_at: r.created_at,
                metadata: Metadata::from(r.metadata),
                content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
                tags: tags_from_value(r.tags)?,
                tags_extractor_model: r.tags_extractor_model,
                tags_extractor_version: r.tags_extractor_version,
                tags_extracted_at: r.tags_extracted_at,
            };
            Ok(Hit::from_trigram_leg(thought, r.sim))
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
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at
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
                content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
                tags: tags_from_value(r.tags)?,
                tags_extractor_model: r.tags_extractor_model,
                tags_extractor_version: r.tags_extractor_version,
                tags_extracted_at: r.tags_extracted_at,
            })
        })
        .collect()
}

/// A row pulled off the `pending_embeddings` queue by `claim_pending`.
#[derive(Debug, Clone)]
pub struct PendingJob {
    pub id: Uuid,
    pub target_kind: String,
    pub target_id: Uuid,
    pub model_id: String,
    pub attempts: i32,
}

/// Enqueue a target for embedding by the worker.
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

/// Atomically claim up to `batch_size` pending embedding jobs.
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

/// Record a failure for a claimed job.
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

/// Heal-step companion to the worker: enqueue every unembedded thought.
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

/// Total rows currently in `pending_embeddings`. Cheap; intended for tests
/// and operator-driven observability.
pub async fn count_pending(pool: &PgPool) -> Result<i64, StorageError> {
    let row = sqlx::query!(r#"SELECT COUNT(*) AS "count!" FROM pending_embeddings"#)
        .fetch_one(pool)
        .await?;
    Ok(row.count)
}

// -- M4 Path B-OB1: thought tagging sidecar --------------------------------

/// Tag-side read shape for `get_thought` — pairs the JSONB `tags` blob with
/// its provenance columns. `tagger_model_id`/`version`/`tagged_at` are all
/// `None` until the tag drainer has run on the thought.
#[derive(Debug, Clone)]
pub struct ThoughtTags {
    pub tags: Tags,
    pub tagger_model_id: Option<String>,
    pub tagger_version: Option<i32>,
    pub tagged_at: Option<OffsetDateTime>,
}

/// A row claimed off the `pending_tags` queue. `attempts` is post-bump
/// (a freshly claimed job returns `attempts = 1`).
#[derive(Debug, Clone)]
pub struct PendingTagJob {
    pub thought_id: ThoughtId,
    pub tagger_model_id: String,
    pub attempts: i32,
}

/// Overwrite a thought's tags + tag provenance. Called by the tag drainer
/// after a successful `tagger.tag()` call. Updates `tags_extracted_at` to
/// NOW(); no supersede semantics — tags are advisory and re-derivable.
pub async fn update_thought_tags(
    pool: &PgPool,
    thought_id: ThoughtId,
    tags: &Tags,
    tagger_model_id: &str,
    tagger_version: i32,
) -> Result<(), StorageError> {
    let tags_value = serde_json::to_value(tags)?;
    sqlx::query!(
        r#"
        UPDATE thoughts
        SET tags = $2,
            tags_extractor_model = $3,
            tags_extractor_version = $4,
            tags_extracted_at = NOW()
        WHERE id = $1
        "#,
        thought_id.into_uuid(),
        tags_value,
        tagger_model_id,
        tagger_version,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Enqueue a thought for the tag drainer. Idempotent on `thought_id`
/// conflict — re-enqueuing the same thought is a no-op.
pub async fn enqueue_tag_job(
    pool: &PgPool,
    thought_id: ThoughtId,
    tagger_model_id: &str,
) -> Result<bool, StorageError> {
    let result = sqlx::query!(
        r#"
        INSERT INTO pending_tags (thought_id, tagger_model_id)
        VALUES ($1, $2)
        ON CONFLICT (thought_id) DO NOTHING
        "#,
        thought_id.into_uuid(),
        tagger_model_id,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Read just the tag block for a thought. Returns `None` if the thought
/// doesn't exist. Used by `get_thought` to enrich its provenance section.
pub async fn fetch_thought_tags(
    pool: &PgPool,
    thought_id: ThoughtId,
) -> Result<Option<ThoughtTags>, StorageError> {
    let row = sqlx::query!(
        r#"
        SELECT tags, tags_extractor_model, tags_extractor_version, tags_extracted_at
        FROM thoughts
        WHERE id = $1
        "#,
        thought_id.into_uuid(),
    )
    .fetch_optional(pool)
    .await?;
    let Some(r) = row else {
        return Ok(None);
    };
    Ok(Some(ThoughtTags {
        tags: tags_from_value(r.tags)?,
        tagger_model_id: r.tags_extractor_model,
        tagger_version: r.tags_extractor_version,
        tagged_at: r.tags_extracted_at,
    }))
}

/// Fetch up to `batch_size` pending tag jobs, oldest first. Does NOT
/// claim/lock — the drainer is single-process at v1 and pops one batch at
/// a time, calling `complete_tag_job` or `increment_tag_job_attempts` per
/// job. If/when we want competing-consumers semantics for tags, replicate
/// `claim_pending`'s `FOR UPDATE SKIP LOCKED` shape here.
pub async fn fetch_pending_tag_jobs(
    pool: &PgPool,
    batch_size: i64,
) -> Result<Vec<PendingTagJob>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT thought_id, tagger_model_id, attempts
        FROM pending_tags
        ORDER BY enqueued_at ASC
        LIMIT $1
        "#,
        batch_size,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| PendingTagJob {
            thought_id: ThoughtId::from(r.thought_id),
            tagger_model_id: r.tagger_model_id,
            attempts: r.attempts,
        })
        .collect())
}

/// Remove a tag job from the queue after a successful tagger.tag() call.
pub async fn complete_tag_job(pool: &PgPool, thought_id: ThoughtId) -> Result<(), StorageError> {
    sqlx::query!(
        r#"DELETE FROM pending_tags WHERE thought_id = $1"#,
        thought_id.into_uuid(),
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Bump the `attempts` counter on a pending tag job after a soft failure.
/// The job stays in the queue; the next drainer tick re-attempts.
pub async fn increment_tag_job_attempts(
    pool: &PgPool,
    thought_id: ThoughtId,
) -> Result<(), StorageError> {
    sqlx::query!(
        r#"UPDATE pending_tags SET attempts = attempts + 1 WHERE thought_id = $1"#,
        thought_id.into_uuid(),
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Walk thoughts that need tagging — either never-tagged (`tags_extractor_version
/// IS NULL`) or stale (`tags_extractor_version < target_tagger_version`, only
/// when `rerun = true`). Oldest first. Used by `engram tag [--rerun]`.
pub async fn find_untagged_or_stale_thoughts(
    pool: &PgPool,
    target_tagger_version: i32,
    rerun: bool,
    scope: Option<&str>,
    since: Option<OffsetDateTime>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, scope, content, source, created_at, metadata,
               content_fingerprint, tags,
               tags_extractor_model, tags_extractor_version, tags_extracted_at
        FROM thoughts
        WHERE retracted_at IS NULL
          AND ($1::text IS NULL OR scope = $1)
          AND ($2::timestamptz IS NULL OR created_at >= $2)
          AND (
              tags_extractor_version IS NULL
              OR ($3::boolean AND tags_extractor_version < $4)
          )
        ORDER BY created_at ASC
        LIMIT $5
        "#,
        scope,
        since,
        rerun,
        target_tagger_version,
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
                content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
                tags: tags_from_value(r.tags)?,
                tags_extractor_model: r.tags_extractor_model,
                tags_extractor_version: r.tags_extractor_version,
                tags_extracted_at: r.tags_extracted_at,
            })
        })
        .collect()
}

/// Compute the established topic + entity vocabulary for a given scope. Used
/// by the tag drainer to supply the tagger with a controlled-vocabulary hint
/// section so it prefers established terms over coining new ones — addresses
/// the v1 corpus-coherence finding (same author's different prose produced
/// divergent topics).
///
/// Returns the top-`limit` most-frequent terms in each of `topics` and
/// `entities`, ranked by occurrence count desc then term asc (stable tie-break).
/// Empty results are valid — they signal "no established vocabulary yet" and
/// the tagger falls back to free-form term coinage.
///
/// Retracted thoughts are excluded so retracted-vocab doesn't bleed into new
/// captures' tags.
pub async fn fetch_scope_vocab(
    pool: &PgPool,
    scope: &str,
    limit: i64,
) -> Result<ScopeVocab, StorageError> {
    let topics = sqlx::query!(
        r#"
        SELECT term AS "term!"
        FROM thoughts,
             LATERAL jsonb_array_elements_text(tags->'topics') AS term
        WHERE scope = $1 AND retracted_at IS NULL
        GROUP BY term
        ORDER BY COUNT(*) DESC, term ASC
        LIMIT $2
        "#,
        scope,
        limit,
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|r| r.term)
    .collect();

    let entities = sqlx::query!(
        r#"
        SELECT term AS "term!"
        FROM thoughts,
             LATERAL jsonb_array_elements_text(tags->'entities') AS term
        WHERE scope = $1 AND retracted_at IS NULL
        GROUP BY term
        ORDER BY COUNT(*) DESC, term ASC
        LIMIT $2
        "#,
        scope,
        limit,
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|r| r.term)
    .collect();

    Ok(ScopeVocab { topics, entities })
}

// -- M5: selective relations (thought-to-thought edges) -----------------

/// One related thought returned by `fetch_related_thoughts`. Carries the
/// edge metadata plus enough enrichment from the joined `thoughts` row that
/// callers can render results without a follow-up `get_thought`.
///
/// `direction` is `Outbound` when the queried thought sits on the edge's
/// `from` side (so `thought_id` is the `to` side here) and `Inbound`
/// otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedThought {
    pub link_id: LinkId,
    pub relation: RelationKind,
    pub direction: LinkDirection,
    pub thought_id: ThoughtId,
    pub scope: Scope,
    pub content: String,
    pub thought_created_at: OffsetDateTime,
    pub link_created_at: OffsetDateTime,
    pub link_source: LinkSource,
    pub note: Option<String>,
    pub retracted: bool,
}

/// Insert a thought-to-thought edge. Idempotent on the `(from, relation, to)`
/// triple via the `thought_links_unique_edge` constraint: re-asserting the
/// same edge returns the existing row's `LinkId` with `is_new = false` and
/// is a no-op otherwise.
///
/// Foreign-key violations (either endpoint missing in `thoughts`) are
/// surfaced as `StorageError::Database`. The MCP layer should pre-validate
/// endpoint existence so the operator-facing error is actionable; this
/// layer is the last line of defense.
pub async fn insert_link(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    to: ThoughtId,
    source: LinkSource,
    note: Option<&str>,
) -> Result<(LinkId, bool), StorageError> {
    let row = sqlx::query!(
        r#"
        INSERT INTO thought_links (from_thought_id, relation, to_thought_id, source, note)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT ON CONSTRAINT thought_links_unique_edge DO NOTHING
        RETURNING id
        "#,
        from.into_uuid(),
        relation.as_str(),
        to.into_uuid(),
        source.as_str(),
        note,
    )
    .fetch_optional(pool)
    .await?;

    if let Some(r) = row {
        return Ok((LinkId::from(r.id), true));
    }

    // ON CONFLICT path: fetch the existing row's id so the caller has a
    // stable identifier regardless of whether the insert was a no-op.
    let existing = sqlx::query!(
        r#"
        SELECT id
        FROM thought_links
        WHERE from_thought_id = $1 AND relation = $2 AND to_thought_id = $3
        "#,
        from.into_uuid(),
        relation.as_str(),
        to.into_uuid(),
    )
    .fetch_one(pool)
    .await?;

    Ok((LinkId::from(existing.id), false))
}

/// Delete a thought-to-thought edge identified by `(from, relation, to)`.
/// Returns `true` when a row was deleted, `false` when no edge matched
/// (idempotent on already-deleted).
pub async fn delete_link(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    to: ThoughtId,
) -> Result<bool, StorageError> {
    let result = sqlx::query!(
        r#"
        DELETE FROM thought_links
        WHERE from_thought_id = $1 AND relation = $2 AND to_thought_id = $3
        "#,
        from.into_uuid(),
        relation.as_str(),
        to.into_uuid(),
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Walk thought-to-thought edges from a given thought. `direction` selects
/// whether to traverse outbound (where `thought_id` is `from`), inbound
/// (where `thought_id` is `to`), or both. `relations`, when supplied,
/// restricts to that subset of the closed vocabulary.
///
/// The returned `RelatedThought` rows carry the *other* end of each edge
/// (so callers can render them directly) along with the edge's metadata.
/// Retracted thoughts on the far end aren't filtered out — the `retracted`
/// flag is surfaced so consumers can decide whether to show/dim/hide.
pub async fn fetch_related_thoughts(
    pool: &PgPool,
    thought_id: ThoughtId,
    relations: Option<&[RelationKind]>,
    direction: LinkDirection,
) -> Result<Vec<RelatedThought>, StorageError> {
    // sqlx::query!'s compile-time check doesn't play well with conditional
    // ANY($N::text[]) when the bound type is Option<Vec<String>>, so we
    // pre-flatten the relation filter into a Vec<String> and use the empty
    // case as the "no filter" sentinel via cardinality(...) = 0 OR ANY(...).
    let relation_filter: Vec<String> = relations
        .map(|rs| rs.iter().map(|r| r.as_str().to_string()).collect())
        .unwrap_or_default();

    let mut rows = Vec::new();

    if matches!(direction, LinkDirection::Outbound | LinkDirection::Both) {
        let out = sqlx::query!(
            r#"
            SELECT
                tl.id AS link_id,
                tl.relation,
                tl.created_at AS link_created_at,
                tl.source AS link_source,
                tl.note,
                t.id AS thought_id,
                t.scope,
                t.content,
                t.created_at AS thought_created_at,
                (t.retracted_at IS NOT NULL) AS "retracted!"
            FROM thought_links tl
            JOIN thoughts t ON t.id = tl.to_thought_id
            WHERE tl.from_thought_id = $1
              AND (cardinality($2::text[]) = 0 OR tl.relation = ANY($2::text[]))
            ORDER BY tl.created_at DESC
            "#,
            thought_id.into_uuid(),
            &relation_filter,
        )
        .fetch_all(pool)
        .await?;

        for r in out {
            rows.push(RelatedThought {
                link_id: LinkId::from(r.link_id),
                relation: r.relation.parse()?,
                direction: LinkDirection::Outbound,
                thought_id: ThoughtId::from(r.thought_id),
                scope: Scope::new(r.scope)?,
                content: r.content,
                thought_created_at: r.thought_created_at,
                link_created_at: r.link_created_at,
                link_source: r.link_source.parse()?,
                note: r.note,
                retracted: r.retracted,
            });
        }
    }

    if matches!(direction, LinkDirection::Inbound | LinkDirection::Both) {
        let inb = sqlx::query!(
            r#"
            SELECT
                tl.id AS link_id,
                tl.relation,
                tl.created_at AS link_created_at,
                tl.source AS link_source,
                tl.note,
                t.id AS thought_id,
                t.scope,
                t.content,
                t.created_at AS thought_created_at,
                (t.retracted_at IS NOT NULL) AS "retracted!"
            FROM thought_links tl
            JOIN thoughts t ON t.id = tl.from_thought_id
            WHERE tl.to_thought_id = $1
              AND (cardinality($2::text[]) = 0 OR tl.relation = ANY($2::text[]))
            ORDER BY tl.created_at DESC
            "#,
            thought_id.into_uuid(),
            &relation_filter,
        )
        .fetch_all(pool)
        .await?;

        for r in inb {
            rows.push(RelatedThought {
                link_id: LinkId::from(r.link_id),
                relation: r.relation.parse()?,
                direction: LinkDirection::Inbound,
                thought_id: ThoughtId::from(r.thought_id),
                scope: Scope::new(r.scope)?,
                content: r.content,
                thought_created_at: r.thought_created_at,
                link_created_at: r.link_created_at,
                link_source: r.link_source.parse()?,
                note: r.note,
                retracted: r.retracted,
            });
        }
    }

    // Both-direction queries are stable-sorted by link_created_at DESC across
    // the union. Outbound rows are already in order from their fetch; inbound
    // rows likewise; merge by re-sorting the combined Vec.
    if matches!(direction, LinkDirection::Both) {
        rows.sort_by_key(|r| std::cmp::Reverse(r.link_created_at));
    }

    Ok(rows)
}

// -- thought retraction (simplified post-M4; no fact cascade) --------------

/// Result of `retract_thought`. Distinguishes "actually retracted this row"
/// from "row didn't exist or was already retracted." Post-M4: no more
/// `facts_superseded` field since the facts table is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetractThoughtOutcome {
    pub retracted: bool,
}

/// Mark a thought as retracted. Retracted thoughts are excluded from
/// retrieval (`recent_thoughts`, `search_trigram`, `search_vector_knn`);
/// `get_thought` is the audit path and continues to return the row.
///
/// Idempotent on a row that's already retracted (`retracted: false`);
/// idempotent on a missing row (same shape). The caller maps that to an
/// operator-facing error string if it wants.
pub async fn retract_thought(
    pool: &PgPool,
    thought_id: ThoughtId,
    reason: Option<&str>,
) -> Result<RetractThoughtOutcome, StorageError> {
    let updated = sqlx::query!(
        r#"
        UPDATE thoughts
        SET retracted_at = NOW(), retracted_reason = $2
        WHERE id = $1 AND retracted_at IS NULL
        "#,
        thought_id.into_uuid(),
        reason,
    )
    .execute(pool)
    .await?;

    Ok(RetractThoughtOutcome {
        retracted: updated.rows_affected() == 1,
    })
}

/// Vector-similarity kNN over `embeddings` for the given model. Hits are
/// returned in descending order of cosine similarity (`1 - cosine_distance`).
/// Uses the per-model HNSW partial index (`embeddings_<model>_hnsw`).
pub async fn search_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    model: &EmbeddingModel,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    let pgv = pgvector::Vector::from(query_vector);

    let rows: Vec<VectorSearchRow> = sqlx::query_as(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
               (e.vector <=> $1) AS distance
        FROM thoughts t
        JOIN embeddings e ON e.target_kind = 'thought' AND e.target_id = t.id
        WHERE e.model_id = $2
          AND ($3::text IS NULL OR t.scope = $3)
          AND ($4::text IS NULL OR t.scope LIKE $4 || '%')
          AND t.retracted_at IS NULL
        ORDER BY e.vector <=> $1
        LIMIT $5
        "#,
    )
    .bind(pgv)
    .bind(&model.id)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            let score = (1.0 - r.distance) as f32;
            let thought = Thought {
                id: ThoughtId::from(r.id),
                scope: Scope::new(r.scope)?,
                content: r.content,
                source: Source::new(r.source)?,
                created_at: r.created_at,
                metadata: Metadata::from(r.metadata),
                content_fingerprint: fingerprint_from_bytes(r.content_fingerprint)?,
                tags: tags_from_value(r.tags)?,
                tags_extractor_model: r.tags_extractor_model,
                tags_extractor_version: r.tags_extractor_version,
                tags_extracted_at: r.tags_extracted_at,
            };
            Ok(Hit::from_vector_leg(thought, score))
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
    content_fingerprint: Vec<u8>,
    tags: serde_json::Value,
    tags_extractor_model: Option<String>,
    tags_extractor_version: Option<i32>,
    tags_extracted_at: Option<OffsetDateTime>,
    distance: f64,
}

// -- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{EmbeddingModel, Metadata, Scope, Source, TagKind};
    use serde_json::json;
    use sha2::{Digest, Sha256};

    /// Compute SHA-256 of `content` and return the 32-byte array. Mirrors
    /// what the MCP capture layer will do before calling insert_thought.
    fn sha256_of(content: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    }

    fn new_thought<'a>(
        scope: &'a Scope,
        source: &'a Source,
        metadata: &'a Metadata,
        content: &'a str,
    ) -> NewThought<'a> {
        NewThought {
            scope,
            content,
            source,
            metadata,
            content_fingerprint: sha256_of(content),
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_returns_id_and_timestamp(pool: PgPool) {
        let scope = Scope::new("work").unwrap();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::from(json!({"client_name": "test"}));

        let (inserted, is_new) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "remember this"),
        )
        .await
        .unwrap();

        assert!(is_new);
        assert_ne!(*inserted.id.as_uuid(), Uuid::nil());
        let now = OffsetDateTime::now_utc();
        let drift = (now - inserted.created_at).whole_seconds().abs();
        assert!(
            drift < 10,
            "created_at not within 10s of now: drift={drift}s"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_returns_existing_id_on_duplicate_content_fingerprint(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();

        let (first, first_is_new) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "same content"),
        )
        .await
        .unwrap();
        assert!(first_is_new);

        // Different metadata is fine — fingerprint is over content only.
        let other_metadata = Metadata::from(json!({"client_name": "different"}));
        let (second, second_is_new) = insert_thought(
            &pool,
            new_thought(&scope, &source, &other_metadata, "same content"),
        )
        .await
        .unwrap();

        assert!(
            !second_is_new,
            "duplicate fingerprint must return is_new=false"
        );
        assert_eq!(first.id, second.id, "duplicate must return the existing id");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_with_distinct_content_returns_distinct_ids(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();

        let (a, a_is_new) = insert_thought(&pool, new_thought(&scope, &source, &metadata, "alpha"))
            .await
            .unwrap();
        let (b, b_is_new) = insert_thought(&pool, new_thought(&scope, &source, &metadata, "beta"))
            .await
            .unwrap();

        assert!(a_is_new);
        assert!(b_is_new);
        assert_ne!(a.id, b.id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_returns_inserted_row(pool: PgPool) {
        let scope = Scope::new("personal").unwrap();
        let source = Source::new("agent:claude-code").unwrap();
        let metadata = Metadata::from(json!({"session_id": "abc"}));

        let (inserted, _) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "remember this"),
        )
        .await
        .unwrap();

        let fetched = fetch_thought(&pool, inserted.id).await.unwrap().unwrap();

        assert_eq!(fetched.id, inserted.id);
        assert_eq!(fetched.scope, scope);
        assert_eq!(fetched.content, "remember this");
        assert_eq!(fetched.source, source);
        assert_eq!(fetched.metadata, metadata);
        assert_eq!(fetched.created_at, inserted.created_at);
        // M4 defaults: empty tags + no provenance until the tag drainer runs.
        assert_eq!(fetched.tags, Tags::default());
        assert!(fetched.tags_extractor_model.is_none());
        assert!(fetched.tags_extractor_version.is_none());
        assert!(fetched.tags_extracted_at.is_none());
        assert_eq!(fetched.content_fingerprint, sha256_of("remember this"));
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
        let (inserted, _) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "remember this"),
        )
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

        assert!(
            thought_has_embedding(&pool, inserted.id, &model)
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn thought_has_embedding_is_false_for_unembedded(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();
        let (inserted, _) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "unembedded thought"),
        )
        .await
        .unwrap();
        let model = EmbeddingModel::bge_m3();
        assert!(
            !thought_has_embedding(&pool, inserted.id, &model)
                .await
                .unwrap()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_embedding_convenience_works(pool: PgPool) {
        let scope = Scope::default();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();
        let (inserted, _) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "convenience test"),
        )
        .await
        .unwrap();

        let model = EmbeddingModel::bge_m3();
        let embedding = Embedding::new(model.clone(), vec![0.5_f32; 1024]).unwrap();
        insert_thought_embedding(&pool, inserted.id, &embedding)
            .await
            .unwrap();
        assert!(
            thought_has_embedding(&pool, inserted.id, &model)
                .await
                .unwrap()
        );
    }

    /// Helper: insert a thought with the given content + scope, return its id.
    async fn insert_test_thought(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        let scope = Scope::new(scope).unwrap();
        let source = Source::new("test").unwrap();
        let metadata = Metadata::empty();
        let (inserted, _) = insert_thought(pool, new_thought(&scope, &source, &metadata, content))
            .await
            .unwrap();
        inserted.id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_newest_first(pool: PgPool) {
        let _a = insert_test_thought(&pool, "first", "global").await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let _b = insert_test_thought(&pool, "second", "global").await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let _c = insert_test_thought(&pool, "third", "global").await;

        let results = recent_thoughts(&pool, None, None, 10).await.unwrap();
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

        let work = recent_thoughts(&pool, Some("work"), None, 10)
            .await
            .unwrap();
        assert_eq!(work.len(), 2);
        assert!(work.iter().all(|t| t.scope.as_str() == "work"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_respects_limit(pool: PgPool) {
        for i in 0..5 {
            insert_test_thought(&pool, &format!("t{i}"), "global").await;
        }
        let r = recent_thoughts(&pool, None, None, 2).await.unwrap();
        assert_eq!(r.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_finds_exact_match(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;
        insert_test_thought(&pool, "weather is nice today", "personal").await;

        let hits = search_trigram(&pool, "tcgplayer", None, None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].thought.content.contains("tcgplayer"));
        assert!(hits[0].trigram_score.unwrap() > 0.0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_respects_scope(pool: PgPool) {
        insert_test_thought(&pool, "tcgplayer info", "work").await;
        insert_test_thought(&pool, "tcgplayer info two", "personal").await;

        let hits = search_trigram(&pool, "tcgplayer", Some("work"), None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].thought.scope.as_str(), "work");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_returns_empty_for_no_match(pool: PgPool) {
        insert_test_thought(&pool, "completely unrelated text", "global").await;
        let hits = search_trigram(&pool, "xyzzyqwerty", None, None, 10)
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

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

        insert_thought_embedding(
            &pool,
            id_a,
            &Embedding::new(model.clone(), va.clone()).unwrap(),
        )
        .await
        .unwrap();
        insert_thought_embedding(&pool, id_b, &Embedding::new(model.clone(), vb).unwrap())
            .await
            .unwrap();

        let hits = search_vector_knn(&pool, va, &model, None, None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].thought.id, id_a);
        assert!((hits[0].vector_score.unwrap() - 1.0).abs() < 1e-4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_filters_by_model(pool: PgPool) {
        let model_a = EmbeddingModel::new("test-a:1024", 1024);
        let model_b = EmbeddingModel::new("test-b:1024", 1024);

        let id = insert_test_thought(&pool, "thought", "global").await;
        let va = unit_vector_1024(0);
        insert_thought_embedding(
            &pool,
            id,
            &Embedding::new(model_a.clone(), va.clone()).unwrap(),
        )
        .await
        .unwrap();

        // Query with model_b — no embeddings → no hits.
        let hits = search_vector_knn(&pool, va, &model_b, None, None, 10)
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_unembedded_thoughts_returns_thoughts_without_embedding(pool: PgPool) {
        let model = EmbeddingModel::new("test:1024", 1024);

        let id_a = insert_test_thought(&pool, "a", "global").await;
        let _id_b = insert_test_thought(&pool, "b", "global").await;

        // Embed only `a`.
        let va = unit_vector_1024(0);
        insert_thought_embedding(&pool, id_a, &Embedding::new(model.clone(), va).unwrap())
            .await
            .unwrap();

        let unembedded = find_unembedded_thoughts(&pool, &model, None, 100)
            .await
            .unwrap();
        assert_eq!(unembedded.len(), 1);
        assert_eq!(unembedded[0].content, "b");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_embedding_is_idempotent(pool: PgPool) {
        let id = insert_test_thought(&pool, "to embed", "global").await;
        let model_id = "bge-m3:1024";

        let first = enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), model_id)
            .await
            .unwrap();
        assert!(first);

        let second = enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), model_id)
            .await
            .unwrap();
        assert!(!second, "duplicate enqueue should be a no-op");

        assert_eq!(count_pending(&pool).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn claim_pending_bumps_attempts_and_returns_jobs(pool: PgPool) {
        let id_a = insert_test_thought(&pool, "a", "global").await;
        let id_b = insert_test_thought(&pool, "b", "global").await;
        let model_id = "bge-m3:1024";

        enqueue_embedding(&pool, target::THOUGHT, id_a.into_uuid(), model_id)
            .await
            .unwrap();
        enqueue_embedding(&pool, target::THOUGHT, id_b.into_uuid(), model_id)
            .await
            .unwrap();

        let claimed = claim_pending(&pool, 10).await.unwrap();
        assert_eq!(claimed.len(), 2);
        assert!(claimed.iter().all(|j| j.attempts == 1));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn mark_embedded_removes_from_queue(pool: PgPool) {
        let id = insert_test_thought(&pool, "to embed", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();

        let claimed = claim_pending(&pool, 1).await.unwrap();
        assert_eq!(claimed.len(), 1);

        mark_embedded(&pool, claimed[0].id).await.unwrap();
        assert_eq!(count_pending(&pool).await.unwrap(), 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn mark_failed_keeps_in_queue_and_sets_error(pool: PgPool) {
        let id = insert_test_thought(&pool, "to embed", "global").await;
        enqueue_embedding(&pool, target::THOUGHT, id.into_uuid(), "bge-m3:1024")
            .await
            .unwrap();

        let claimed = claim_pending(&pool, 1).await.unwrap();
        assert_eq!(claimed.len(), 1);

        mark_failed(&pool, claimed[0].id, "timeout").await.unwrap();
        assert_eq!(count_pending(&pool).await.unwrap(), 1);

        let row = sqlx::query!(
            r#"SELECT last_error FROM pending_embeddings WHERE id = $1"#,
            claimed[0].id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.last_error.as_deref(), Some("timeout"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_unembedded_thoughts_skips_already_embedded(pool: PgPool) {
        let model = EmbeddingModel::new("test:1024", 1024);
        let id_a = insert_test_thought(&pool, "a", "global").await;
        let _id_b = insert_test_thought(&pool, "b", "global").await;

        // Embed only `a`.
        let va = unit_vector_1024(0);
        insert_thought_embedding(&pool, id_a, &Embedding::new(model.clone(), va).unwrap())
            .await
            .unwrap();

        let enqueued = enqueue_unembedded_thoughts(&pool, &model.id, None, 100)
            .await
            .unwrap();
        assert_eq!(enqueued, 1, "only `b` should be enqueued");
    }

    // -- M4: tag-sidecar tests ------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn update_thought_tags_persists_jsonb_and_provenance(pool: PgPool) {
        let id = insert_test_thought(&pool, "tagged thought", "global").await;

        let tags = Tags {
            people: vec!["Sarah".into()],
            entities: vec!["engram".into()],
            action_items: vec!["follow up".into()],
            topics: vec!["meetings".into()],
            dates_mentioned: vec!["Thursday".into()],
            kind: Some(TagKind::Task),
        };
        update_thought_tags(&pool, id, &tags, "vllm/qwen2.5-7b-instruct", 1)
            .await
            .unwrap();

        let read = fetch_thought_tags(&pool, id).await.unwrap().unwrap();
        assert_eq!(read.tags, tags);
        assert_eq!(
            read.tagger_model_id.as_deref(),
            Some("vllm/qwen2.5-7b-instruct")
        );
        assert_eq!(read.tagger_version, Some(1));
        assert!(read.tagged_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_tag_job_inserts_into_pending_tags(pool: PgPool) {
        let id = insert_test_thought(&pool, "to tag", "global").await;
        let inserted = enqueue_tag_job(&pool, id, "vllm/qwen2.5-7b-instruct")
            .await
            .unwrap();
        assert!(inserted);

        let jobs = fetch_pending_tag_jobs(&pool, 10).await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].thought_id, id);
        assert_eq!(jobs[0].tagger_model_id, "vllm/qwen2.5-7b-instruct");
        assert_eq!(jobs[0].attempts, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_tag_job_idempotent_on_conflict(pool: PgPool) {
        let id = insert_test_thought(&pool, "to tag", "global").await;
        let first = enqueue_tag_job(&pool, id, "v1").await.unwrap();
        let second = enqueue_tag_job(&pool, id, "v1").await.unwrap();
        assert!(first);
        assert!(!second, "duplicate enqueue should be a no-op");

        let jobs = fetch_pending_tag_jobs(&pool, 10).await.unwrap();
        assert_eq!(jobs.len(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn complete_tag_job_removes_from_queue(pool: PgPool) {
        let id = insert_test_thought(&pool, "to tag", "global").await;
        enqueue_tag_job(&pool, id, "v1").await.unwrap();

        complete_tag_job(&pool, id).await.unwrap();

        let jobs = fetch_pending_tag_jobs(&pool, 10).await.unwrap();
        assert!(jobs.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn increment_tag_job_attempts_bumps_counter(pool: PgPool) {
        let id = insert_test_thought(&pool, "to tag", "global").await;
        enqueue_tag_job(&pool, id, "v1").await.unwrap();

        increment_tag_job_attempts(&pool, id).await.unwrap();
        increment_tag_job_attempts(&pool, id).await.unwrap();

        let jobs = fetch_pending_tag_jobs(&pool, 10).await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].attempts, 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_untagged_or_stale_thoughts_returns_only_null_when_rerun_false(pool: PgPool) {
        let untagged = insert_test_thought(&pool, "untagged", "global").await;
        let already_tagged = insert_test_thought(&pool, "already tagged", "global").await;
        update_thought_tags(&pool, already_tagged, &Tags::default(), "v1-model", 1)
            .await
            .unwrap();

        let walk = find_untagged_or_stale_thoughts(
            &pool, /*target_version*/ 1, /*rerun*/ false, None, None, 100,
        )
        .await
        .unwrap();
        let ids: Vec<ThoughtId> = walk.iter().map(|t| t.id).collect();
        assert!(ids.contains(&untagged));
        assert!(!ids.contains(&already_tagged));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_untagged_or_stale_thoughts_returns_stale_when_rerun_true(pool: PgPool) {
        let untagged = insert_test_thought(&pool, "untagged", "global").await;
        let stale_v1 = insert_test_thought(&pool, "stale at v1", "global").await;
        update_thought_tags(&pool, stale_v1, &Tags::default(), "v1-model", 1)
            .await
            .unwrap();
        let fresh_v2 = insert_test_thought(&pool, "fresh at v2", "global").await;
        update_thought_tags(&pool, fresh_v2, &Tags::default(), "v2-model", 2)
            .await
            .unwrap();

        // target_version=2, rerun=true → walks NULL and version<2.
        let walk = find_untagged_or_stale_thoughts(&pool, 2, true, None, None, 100)
            .await
            .unwrap();
        let ids: Vec<ThoughtId> = walk.iter().map(|t| t.id).collect();
        assert!(ids.contains(&untagged));
        assert!(ids.contains(&stale_v1));
        assert!(!ids.contains(&fresh_v2));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_tags_returns_none_for_missing_thought(pool: PgPool) {
        let id = ThoughtId::new();
        assert!(fetch_thought_tags(&pool, id).await.unwrap().is_none());
    }

    // -- M4.1: scope vocabulary -------------------------------------------

    /// Helper for fetch_scope_vocab tests — insert a thought and immediately
    /// attach the given tags. Keeps each test body terse and focused on the
    /// vocabulary aggregation behavior.
    async fn seed_tagged(pool: &PgPool, scope: &str, content: &str, tags: Tags) -> ThoughtId {
        let id = insert_test_thought(pool, content, scope).await;
        update_thought_tags(pool, id, &tags, "test-model", 1)
            .await
            .unwrap();
        id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_scope_vocab_ranks_by_count_desc_then_term_asc(pool: PgPool) {
        // Three thoughts in the same scope sharing "rust" (3x), with "build-systems"
        // appearing twice and "team-management" once. Ties on count fall back to
        // term-ascending for stable ranking.
        seed_tagged(
            &pool,
            "work",
            "a",
            Tags {
                topics: vec!["rust".into(), "build-systems".into()],
                ..Tags::default()
            },
        )
        .await;
        seed_tagged(
            &pool,
            "work",
            "b",
            Tags {
                topics: vec![
                    "rust".into(),
                    "build-systems".into(),
                    "team-management".into(),
                ],
                ..Tags::default()
            },
        )
        .await;
        seed_tagged(
            &pool,
            "work",
            "c",
            Tags {
                topics: vec!["rust".into()],
                ..Tags::default()
            },
        )
        .await;

        let v = fetch_scope_vocab(&pool, "work", 10).await.unwrap();
        assert_eq!(
            v.topics,
            vec![
                "rust".to_string(),
                "build-systems".to_string(),
                "team-management".to_string(),
            ]
        );
        assert!(v.entities.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_scope_vocab_isolates_by_scope(pool: PgPool) {
        seed_tagged(
            &pool,
            "work",
            "a",
            Tags {
                topics: vec!["work-only".into()],
                entities: vec!["engram".into()],
                ..Tags::default()
            },
        )
        .await;
        seed_tagged(
            &pool,
            "personal",
            "b",
            Tags {
                topics: vec!["personal-only".into()],
                entities: vec!["garmin".into()],
                ..Tags::default()
            },
        )
        .await;

        let work_v = fetch_scope_vocab(&pool, "work", 10).await.unwrap();
        assert_eq!(work_v.topics, vec!["work-only".to_string()]);
        assert_eq!(work_v.entities, vec!["engram".to_string()]);

        let personal_v = fetch_scope_vocab(&pool, "personal", 10).await.unwrap();
        assert_eq!(personal_v.topics, vec!["personal-only".to_string()]);
        assert_eq!(personal_v.entities, vec!["garmin".to_string()]);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_scope_vocab_honors_limit(pool: PgPool) {
        seed_tagged(
            &pool,
            "global",
            "a",
            Tags {
                topics: vec!["t1".into(), "t2".into(), "t3".into()],
                ..Tags::default()
            },
        )
        .await;

        let v = fetch_scope_vocab(&pool, "global", 2).await.unwrap();
        assert_eq!(v.topics.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_scope_vocab_excludes_retracted_thoughts(pool: PgPool) {
        let retracted = seed_tagged(
            &pool,
            "global",
            "retracted",
            Tags {
                topics: vec!["dropped".into()],
                entities: vec!["ghost".into()],
                ..Tags::default()
            },
        )
        .await;
        seed_tagged(
            &pool,
            "global",
            "active",
            Tags {
                topics: vec!["kept".into()],
                entities: vec!["real".into()],
                ..Tags::default()
            },
        )
        .await;
        retract_thought(&pool, retracted, None).await.unwrap();

        let v = fetch_scope_vocab(&pool, "global", 10).await.unwrap();
        assert_eq!(v.topics, vec!["kept".to_string()]);
        assert_eq!(v.entities, vec!["real".to_string()]);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_scope_vocab_empty_scope_returns_empty_vocab(pool: PgPool) {
        seed_tagged(
            &pool,
            "elsewhere",
            "a",
            Tags {
                topics: vec!["foo".into()],
                ..Tags::default()
            },
        )
        .await;

        let v = fetch_scope_vocab(&pool, "nonexistent", 10).await.unwrap();
        assert!(v.is_empty());
    }

    // -- M5: selective relations (thought-to-thought edges) -----------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_returns_id_and_is_new(pool: PgPool) {
        let a = insert_test_thought(&pool, "thought A", "global").await;
        let b = insert_test_thought(&pool, "thought B", "global").await;

        let (link_id, is_new) =
            insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
                .await
                .unwrap();
        assert!(is_new);
        assert_ne!(*link_id.as_uuid(), Uuid::nil());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_duplicate_triple_is_idempotent(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;

        let (first_id, first_is_new) =
            insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
                .await
                .unwrap();
        let (second_id, second_is_new) =
            insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
                .await
                .unwrap();

        assert!(first_is_new);
        assert!(!second_is_new, "second insert of same triple must be no-op");
        assert_eq!(first_id, second_id, "must return same link id on conflict");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_self_reference_rejected_by_db(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let err = insert_link(&pool, a, RelationKind::Refines, a, LinkSource::Agent, None)
            .await
            .unwrap_err();
        // CHECK constraint surfaces as a Database error; the MCP layer
        // should pre-validate so callers never hit this path.
        assert!(matches!(err, StorageError::Database(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_unknown_thought_rejected_by_fk(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let phantom = ThoughtId::new();
        let err = insert_link(
            &pool,
            a,
            RelationKind::References,
            phantom,
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap_err();
        // Foreign-key violation surfaces as a Database error.
        assert!(matches!(err, StorageError::Database(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_persists_note_and_source(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;

        let (_id, _is_new) = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            b,
            LinkSource::Agent,
            Some("first refinement during dogfood"),
        )
        .await
        .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(
            related[0].note.as_deref(),
            Some("first refinement during dogfood")
        );
        assert_eq!(related[0].link_source, LinkSource::Agent);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_link_returns_existed_then_false(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();

        let existed = delete_link(&pool, a, RelationKind::Refines, b)
            .await
            .unwrap();
        assert!(existed);

        let existed_again = delete_link(&pool, a, RelationKind::Refines, b)
            .await
            .unwrap();
        assert!(
            !existed_again,
            "second delete on missing edge must be no-op"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_outbound_returns_to_side_only(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();

        let from_a = fetch_related_thoughts(&pool, a, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].thought_id, b);
        assert_eq!(from_a[0].direction, LinkDirection::Outbound);

        let from_b = fetch_related_thoughts(&pool, b, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert!(from_b.is_empty(), "B has no outbound edges");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_inbound_returns_from_side_only(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();

        let into_b = fetch_related_thoughts(&pool, b, None, LinkDirection::Inbound)
            .await
            .unwrap();
        assert_eq!(into_b.len(), 1);
        assert_eq!(into_b[0].thought_id, a);
        assert_eq!(into_b[0].direction, LinkDirection::Inbound);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_both_returns_outbound_plus_inbound(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        let c = insert_test_thought(&pool, "C", "global").await;
        // A refines B; C refines A. So A has 1 outbound + 1 inbound edge.
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();
        insert_link(&pool, c, RelationKind::Refines, a, LinkSource::Agent, None)
            .await
            .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, LinkDirection::Both)
            .await
            .unwrap();
        assert_eq!(related.len(), 2);
        let directions: Vec<LinkDirection> = related.iter().map(|r| r.direction).collect();
        assert!(directions.contains(&LinkDirection::Outbound));
        assert!(directions.contains(&LinkDirection::Inbound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_filtered_by_relation(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        let c = insert_test_thought(&pool, "C", "global").await;
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();
        insert_link(&pool, a, RelationKind::Replaces, c, LinkSource::Agent, None)
            .await
            .unwrap();

        let only_refines = fetch_related_thoughts(
            &pool,
            a,
            Some(&[RelationKind::Refines]),
            LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert_eq!(only_refines.len(), 1);
        assert_eq!(only_refines[0].relation, RelationKind::Refines);

        let multi = fetch_related_thoughts(
            &pool,
            a,
            Some(&[RelationKind::Refines, RelationKind::Replaces]),
            LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert_eq!(multi.len(), 2);

        let only_requires = fetch_related_thoughts(
            &pool,
            a,
            Some(&[RelationKind::Requires]),
            LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert!(
            only_requires.is_empty(),
            "filter must exclude non-matching relations"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_surfaces_retracted_state(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();
        retract_thought(&pool, b, Some("dogfood retraction"))
            .await
            .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, LinkDirection::Outbound)
            .await
            .unwrap();
        // Soft retraction preserves the edge — it just surfaces the flag.
        assert_eq!(related.len(), 1);
        assert!(
            related[0].retracted,
            "retracted state must propagate to the response"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn cascade_on_thought_hard_delete_removes_edges(pool: PgPool) {
        // Direct DELETE FROM thoughts triggers the ON DELETE CASCADE on
        // thought_links. Engram itself uses soft-retraction, but the DB
        // invariant should still hold for any future hard-delete pathway.
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(&pool, a, RelationKind::Refines, b, LinkSource::Agent, None)
            .await
            .unwrap();

        sqlx::query!("DELETE FROM thoughts WHERE id = $1", b.into_uuid())
            .execute(&pool)
            .await
            .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert!(
            related.is_empty(),
            "edge must be CASCADE-deleted with the thought"
        );
    }

    // -- M5.x: scope discoverability (list_scopes + scope_prefix) -----------

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_returns_summary_with_counts_and_timestamps(pool: PgPool) {
        insert_test_thought(&pool, "a1", "work.tcgplayer").await;
        insert_test_thought(&pool, "a2", "work.tcgplayer").await;
        insert_test_thought(&pool, "b1", "project.engram").await;

        let scopes = list_scopes(&pool, None).await.unwrap();
        assert_eq!(scopes.len(), 2);
        let by_scope: std::collections::HashMap<&str, &ScopeSummary> =
            scopes.iter().map(|s| (s.scope.as_str(), s)).collect();
        assert_eq!(by_scope.get("work.tcgplayer").unwrap().thought_count, 2);
        assert_eq!(by_scope.get("project.engram").unwrap().thought_count, 1);
        // first_activity_at <= last_activity_at always.
        for s in &scopes {
            assert!(s.first_activity_at <= s.last_activity_at);
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_prefix_filter_matches_namespace(pool: PgPool) {
        insert_test_thought(&pool, "x", "rjf.a").await;
        insert_test_thought(&pool, "y", "rjf.b").await;
        insert_test_thought(&pool, "z", "other").await;

        let rjf_scopes = list_scopes(&pool, Some("rjf.")).await.unwrap();
        let names: Vec<&str> = rjf_scopes.iter().map(|s| s.scope.as_str()).collect();
        assert_eq!(rjf_scopes.len(), 2);
        assert!(names.contains(&"rjf.a"));
        assert!(names.contains(&"rjf.b"));
        assert!(!names.contains(&"other"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_excludes_retracted_thoughts(pool: PgPool) {
        let only = insert_test_thought(&pool, "doomed", "ephemeral").await;
        insert_test_thought(&pool, "kept", "kept").await;
        retract_thought(&pool, only, None).await.unwrap();

        let scopes = list_scopes(&pool, None).await.unwrap();
        let names: Vec<&str> = scopes.iter().map(|s| s.scope.as_str()).collect();
        assert!(!names.contains(&"ephemeral"));
        assert!(names.contains(&"kept"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_empty_corpus_returns_empty_vec(pool: PgPool) {
        let scopes = list_scopes(&pool, None).await.unwrap();
        assert!(scopes.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_orders_by_last_activity_desc(pool: PgPool) {
        // First insert lives in scope A; later inserts in scope B and then C.
        // Expectation: order is C, B, A (most recent last_activity_at first).
        insert_test_thought(&pool, "early", "scope.a").await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        insert_test_thought(&pool, "middle", "scope.b").await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        insert_test_thought(&pool, "late", "scope.c").await;

        let scopes = list_scopes(&pool, None).await.unwrap();
        let order: Vec<&str> = scopes.iter().map(|s| s.scope.as_str()).collect();
        assert_eq!(order, vec!["scope.c", "scope.b", "scope.a"]);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_scope_prefix_matches_multiple_scopes(pool: PgPool) {
        insert_test_thought(&pool, "alpha", "rjf.a").await;
        insert_test_thought(&pool, "beta", "rjf.b").await;
        insert_test_thought(&pool, "gamma", "other").await;

        let hits = recent_thoughts(&pool, None, Some("rjf."), 10)
            .await
            .unwrap();
        let scopes: Vec<&str> = hits.iter().map(|t| t.scope.as_str()).collect();
        assert_eq!(hits.len(), 2);
        assert!(scopes.iter().all(|s| s.starts_with("rjf.")));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_trigram_scope_prefix_matches_multiple_scopes(pool: PgPool) {
        insert_test_thought(&pool, "unique_keyword in rjf.a", "rjf.a").await;
        insert_test_thought(&pool, "unique_keyword in rjf.b", "rjf.b").await;
        insert_test_thought(&pool, "unique_keyword in other", "other").await;

        let hits = search_trigram(&pool, "unique_keyword", None, Some("rjf."), 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        let scopes: Vec<&str> = hits.iter().map(|h| h.thought.scope.as_str()).collect();
        assert!(scopes.iter().all(|s| s.starts_with("rjf.")));
    }

    // -- M4: retraction (simplified — no fact cascade) ----------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_sets_retracted_at(pool: PgPool) {
        let id = insert_test_thought(&pool, "to retract", "global").await;
        let outcome = retract_thought(&pool, id, Some("test reason"))
            .await
            .unwrap();
        assert!(outcome.retracted);

        let row = sqlx::query!(
            r#"SELECT retracted_at, retracted_reason FROM thoughts WHERE id = $1"#,
            id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.retracted_at.is_some());
        assert_eq!(row.retracted_reason.as_deref(), Some("test reason"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_is_idempotent_on_already_retracted(pool: PgPool) {
        let id = insert_test_thought(&pool, "to retract", "global").await;
        let first = retract_thought(&pool, id, None).await.unwrap();
        let second = retract_thought(&pool, id, None).await.unwrap();
        assert!(first.retracted);
        assert!(!second.retracted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_on_missing_id_reports_no_op(pool: PgPool) {
        let outcome = retract_thought(&pool, ThoughtId::new(), None)
            .await
            .unwrap();
        assert!(!outcome.retracted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_recent_thoughts(pool: PgPool) {
        let active = insert_test_thought(&pool, "active", "global").await;
        let retracted = insert_test_thought(&pool, "retracted", "global").await;
        retract_thought(&pool, retracted, None).await.unwrap();

        let recent = recent_thoughts(&pool, None, None, 10).await.unwrap();
        let ids: Vec<ThoughtId> = recent.iter().map(|t| t.id).collect();
        assert!(ids.contains(&active));
        assert!(!ids.contains(&retracted));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_search_trigram(pool: PgPool) {
        let _active = insert_test_thought(&pool, "unique_keyword active", "global").await;
        let retracted = insert_test_thought(&pool, "unique_keyword retracted", "global").await;
        retract_thought(&pool, retracted, None).await.unwrap();

        let hits = search_trigram(&pool, "unique_keyword", None, None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_ne!(hits[0].thought.id, retracted);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retracted_thought_excluded_from_find_untagged_or_stale(pool: PgPool) {
        let active = insert_test_thought(&pool, "active", "global").await;
        let retracted = insert_test_thought(&pool, "retracted", "global").await;
        retract_thought(&pool, retracted, None).await.unwrap();

        let walk = find_untagged_or_stale_thoughts(&pool, 1, false, None, None, 100)
            .await
            .unwrap();
        let ids: Vec<ThoughtId> = walk.iter().map(|t| t.id).collect();
        assert!(ids.contains(&active));
        assert!(!ids.contains(&retracted));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_thought_with_provenance_surfaces_retracted_at(pool: PgPool) {
        let id = insert_test_thought(&pool, "to retract", "global").await;
        retract_thought(&pool, id, Some("operator decision"))
            .await
            .unwrap();

        let model = EmbeddingModel::bge_m3();
        let prov = fetch_thought_with_provenance(&pool, id, &model)
            .await
            .unwrap()
            .unwrap();
        assert!(prov.retracted_at.is_some());
        assert_eq!(prov.retracted_reason.as_deref(), Some("operator decision"));
    }
}

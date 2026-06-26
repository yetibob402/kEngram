//! kengram-storage: sqlx-backed repository functions.
//!
//! The `Embedder` trait is the only place we hide a backend choice behind a
//! trait — storage is concrete sqlx + Postgres. CLAUDE.md rule: compile-time
//! `sqlx::query!` everywhere except where pgvector's vector binding gets in
//! the way of the macro (currently: only `insert_embedding`).

use kengram_core::{
    ChunkProvenance, Embedding, EmbeddingModel, EmbeddingStatus, Hit, LinkDirection, LinkId,
    LinkSource, LinkTarget, Metadata, RelationKind, Scope, ScopeError, ScopeVocab, Source,
    SourceError, SparseEmbeddingModel, SparseLexicalVector, Tags, Thought, ThoughtId,
    UnknownLinkSource, UnknownRelationKind,
};
use sqlx::{PgPool, PgTransaction};
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

mod ann {
    pub const HALF_3072_DIMS: usize = 3072;
    pub const HALF_3072_DIMS_I32: i32 = 3072;
    pub const HALF_3072_HNSW_EF_SEARCH: i32 = 1000;
    pub const PROJECTION_SUFFIX: &str = "halfvec:3072";
}

mod bge {
    pub const MODEL_ID: &str = "bge-m3:1024";
    pub const DIMS: usize = 1024;
    pub const DIMS_I32: i32 = 1024;
    pub const MODEL_VERSION: i32 = 1;
    pub const HNSW_EF_SEARCH: i32 = 1000;
    pub const THOUGHT_TABLE: &str = "thought_embeddings_bge_m3";
    pub const THOUGHT_HNSW_INDEX: &str = "thought_embeddings_bge_m3_hnsw";
    pub const CHUNK_TABLE: &str = "artifact_chunk_embeddings_bge_m3";
    pub const CHUNK_HNSW_INDEX: &str = "artifact_chunk_embeddings_bge_m3_hnsw";
    pub const CONTEXT_TABLE: &str = "artifact_chunk_context_embeddings_bge_m3";
    pub const CONTEXT_HNSW_INDEX: &str = "artifact_chunk_context_embeddings_bge_m3_hnsw";
}

mod bge_sparse {
    pub const MODEL_ID: &str = "bge-m3:sparse";
    pub const SOURCE_MODEL: &str = "BAAI/bge-m3";
    pub const VOCAB_SIZE: usize = 250_002;
    pub const VOCAB_SIZE_I32: i32 = 250_002;
    pub const MODEL_VERSION: i32 = 1;
}

#[derive(Debug, Clone)]
struct AnnProjection {
    projection_id: String,
    dimensions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnProjectionCoverage {
    pub projection_id: String,
    pub model_id: String,
    pub model_version: i32,
    pub embedding_count: i64,
    pub projection_count: i64,
    pub missing_count: i64,
    pub inserted_missing: i64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SparseEmbeddingProvenance {
    pub source_model: String,
    pub generator: String,
    pub generator_version: String,
    pub pipeline_run_id: Option<Uuid>,
    pub producer_metadata: serde_json::Value,
}

impl SparseEmbeddingProvenance {
    pub fn bge_m3_flag_embedding(generator_version: impl Into<String>) -> Self {
        Self {
            source_model: bge_sparse::SOURCE_MODEL.to_string(),
            generator: "FlagEmbedding.BGEM3FlagModel".to_string(),
            generator_version: generator_version.into(),
            pipeline_run_id: None,
            producer_metadata: serde_json::json!({}),
        }
    }
}

pub const CONTEXTUAL_CONTAMINATION_FILTER_VERSION: &str = "eval-contamination-v1";

#[derive(Debug, Clone)]
pub struct ContextGenerationSource {
    pub chunk_id: Uuid,
    pub source_thought_id: ThoughtId,
    pub scope: Scope,
    pub parent_source: Source,
    pub parent_created_at: OffsetDateTime,
    pub parent_metadata: Metadata,
    pub parent_content: String,
    pub chunk_index: i32,
    pub chunk_content: String,
    pub chunk_metadata: serde_json::Value,
    pub raw_chunk_fingerprint: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct ArtifactChunkContextInsert {
    pub chunk_id: Uuid,
    pub context_text: String,
    pub generator_id: String,
    pub generator_version: i32,
    pub prompt_version: String,
    pub prompt_hash: String,
    pub model_id: String,
    pub model_version: String,
    pub pipeline_run_id: Option<Uuid>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactChunkContextOutcome {
    pub context_id: Uuid,
    pub status: String,
    pub rejection_reason: Option<String>,
}

fn ann_projection_for_model(model: &EmbeddingModel) -> Option<AnnProjection> {
    if is_bge_m3_1024(model) {
        return None;
    }
    if model.dimensions < ann::HALF_3072_DIMS {
        return None;
    }

    Some(AnnProjection {
        projection_id: format!("{}:{}", model.id, ann::PROJECTION_SUFFIX),
        dimensions: ann::HALF_3072_DIMS,
    })
}

fn is_bge_m3_1024(model: &EmbeddingModel) -> bool {
    model.id == bge::MODEL_ID && model.dimensions == bge::DIMS
}

fn is_bge_m3_sparse(model: &SparseEmbeddingModel) -> bool {
    model.id == bge_sparse::MODEL_ID
        && model.version == bge_sparse::MODEL_VERSION
        && model.vocab_size == bge_sparse::VOCAB_SIZE
}

fn validate_bge_m3_sparse(vector: &SparseLexicalVector) -> Result<(), StorageError> {
    if is_bge_m3_sparse(&vector.model) {
        Ok(())
    } else {
        Err(StorageError::InvalidSparseModel {
            model_id: vector.model.id.clone(),
            model_version: vector.model.version,
            vocab_size: vector.model.vocab_size,
        })
    }
}

fn project_halfvec_3072(
    vector: &[f32],
    dimensions: usize,
) -> Result<pgvector::HalfVector, StorageError> {
    if vector.len() < dimensions {
        return Err(StorageError::InvalidAnnProjectionDimensions {
            expected: dimensions,
            got: vector.len(),
        });
    }

    let prefix = &vector[..dimensions];
    let norm = prefix
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let projected = if norm > 0.0 {
        prefix
            .iter()
            .map(|v| (*v as f64 / norm) as f32)
            .collect::<Vec<_>>()
    } else {
        prefix.to_vec()
    };

    Ok(pgvector::HalfVector::from_f32_slice(&projected))
}

fn sql_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn sql_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn normalize_fts_query(query: &str) -> Option<String> {
    let normalized = query
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>();
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed
        .chars()
        .all(|c| c.is_ascii_punctuation() || c.is_whitespace())
    {
        return None;
    }

    Some(trimmed.to_string())
}

// Phase0 eval-contamination fence: keep gold/eval/adjudication artifacts out
// of retrieval candidate pools before RRF/rerank. This is intentionally
// conservative and duplicated into SQL so every leg filters at source.
const EVAL_CONTAMINATION_SOURCE_FILE_REGEX: &str = "(kengram-recall-97|kengram-gold|gold100|gold-100|miss-analysis|label-repair|adjudication|answer-key|retrieval-baseline)";
const EVAL_CONTAMINATION_CONTENT_REGEX: &str = "KGR[0-9]{3}";
const EVAL_CONTAMINATION_KNOWN_IDS: &[Uuid] = &[
    uuid::uuid!("43ec4976-d33b-4292-bbf6-ce141f6418dd"),
    uuid::uuid!("5853f4c5-afca-433b-9506-40c015646c23"),
    uuid::uuid!("a58e47fa-e933-4f75-9af8-3f7873ab9f58"),
];
const MAX_CONTEXTUAL_CONTEXT_CHARS: usize = 1_200;

fn text_trips_eval_contamination(text: &str) -> bool {
    let upper = text.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    if bytes.len() < 6 {
        return false;
    }
    bytes.windows(6).any(|window| {
        window[0] == b'K'
            && window[1] == b'G'
            && window[2] == b'R'
            && window[3].is_ascii_digit()
            && window[4].is_ascii_digit()
            && window[5].is_ascii_digit()
    })
}

fn source_file_trips_eval_contamination(metadata: &serde_json::Value) -> bool {
    let Some(source_file) = metadata.get("source_file").and_then(|v| v.as_str()) else {
        return false;
    };
    let lower = source_file.to_ascii_lowercase();
    [
        "kengram-recall-97",
        "kengram-gold",
        "gold100",
        "gold-100",
        "miss-analysis",
        "label-repair",
        "adjudication",
        "answer-key",
        "retrieval-baseline",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn known_eval_thought_id(id: ThoughtId) -> bool {
    EVAL_CONTAMINATION_KNOWN_IDS.contains(&id.into_uuid())
}

fn contamination_rejection_reason(
    source_thought_id: ThoughtId,
    parent_metadata: &serde_json::Value,
    parent_content: &str,
    chunk_content: &str,
) -> Option<String> {
    if known_eval_thought_id(source_thought_id) {
        return Some("known_eval_thought_id".to_string());
    }
    if source_file_trips_eval_contamination(parent_metadata) {
        return Some("eval_source_file".to_string());
    }
    if text_trips_eval_contamination(parent_content) {
        return Some("parent_eval_marker".to_string());
    }
    if text_trips_eval_contamination(chunk_content) {
        return Some("raw_chunk_eval_marker".to_string());
    }
    None
}

fn ann_projection_index_name(projection_id: &str) -> String {
    let mut sanitized = projection_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();

    while sanitized.contains("__") {
        sanitized = sanitized.replace("__", "_");
    }
    sanitized = sanitized.trim_matches('_').to_string();

    let base = format!("embedding_ann_projection_{sanitized}_hnsw");
    if base.len() <= 63 {
        base
    } else {
        base[..63].trim_end_matches('_').to_string()
    }
}

async fn set_ann_projection_ef_search(tx: &mut PgTransaction<'_>) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT set_config('hnsw.ef_search', $1, true)")
        .bind(ann::HALF_3072_HNSW_EF_SEARCH.to_string())
        .execute(&mut **tx)
        .await?;

    Ok(())
}

async fn set_bge_hnsw_ef_search(tx: &mut PgTransaction<'_>) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT set_config('hnsw.ef_search', $1, true)")
        .bind(bge::HNSW_EF_SEARCH.to_string())
        .execute(&mut **tx)
        .await?;

    Ok(())
}

async fn set_statement_timeout(
    tx: &mut PgTransaction<'_>,
    timeout_ms: u64,
) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT set_config('statement_timeout', $1, true)")
        .bind(format!("{timeout_ms}ms"))
        .execute(&mut **tx)
        .await?;

    Ok(())
}

async fn ann_projection_index_ready_on_conn(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Postgres>,
    index_name: &str,
) -> Result<(bool,), sqlx::Error> {
    sqlx::query_as(
        r#"
        SELECT EXISTS (
            SELECT 1
            FROM pg_class c
            JOIN pg_index i ON i.indexrelid = c.oid
            WHERE c.relname = $1
              AND i.indisready
              AND i.indisvalid
        )
        "#,
    )
    .bind(index_name)
    .fetch_one(&mut **conn)
    .await
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

    #[error(
        "invalid link target shape decoded from database: to_kind={0:?} but per-kind columns don't match"
    )]
    InvalidLinkTargetShape(String),

    #[error(
        "embedding vector too short for ANN projection: expected at least {expected} dims, got {got}"
    )]
    InvalidAnnProjectionDimensions { expected: usize, got: usize },

    #[error("embedding vector dimensions mismatch for {model_id}: expected {expected}, got {got}")]
    InvalidEmbeddingDimensions {
        model_id: String,
        expected: usize,
        got: usize,
    },

    #[error(
        "bge-m3 sidecar only supports thought and artifact_chunk embeddings, got target_kind={0}"
    )]
    UnsupportedBgeTargetKind(String),

    #[error(
        "bge-m3 sparse sidecar only supports thought and artifact_chunk embeddings, got target_kind={0}"
    )]
    UnsupportedBgeSparseTargetKind(String),

    #[error(
        "invalid sparse model: id={model_id}, version={model_version}, vocab_size={vocab_size}"
    )]
    InvalidSparseModel {
        model_id: String,
        model_version: i32,
        vocab_size: usize,
    },

    #[error("sparse source content length must be non-negative, got {0}")]
    InvalidSparseSourceContentChars(i32),

    #[error(
        "ANN projection coverage mismatch for {projection_id}: embeddings={embedding_count}, projections={projection_count}, missing={missing_count}"
    )]
    AnnProjectionCoverageMismatch {
        projection_id: String,
        embedding_count: i64,
        projection_count: i64,
        missing_count: i64,
    },

    #[error("ANN projection index {0} exists but is not ready/valid")]
    AnnProjectionIndexNotReady(String),

    #[error("bge-m3 sidecar index {0} is missing or not ready/valid")]
    BgeSidecarIndexNotReady(String),

    #[error("bge-m3 sidecar table {0} is missing")]
    BgeSidecarTableMissing(String),
}

impl StorageError {
    pub fn is_query_canceled(&self) -> bool {
        matches!(
            self,
            StorageError::Database(sqlx::Error::Database(db))
                if db.code().is_some_and(|code| code == "57014")
        )
    }
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
    if is_bge_m3_1024(model) {
        return insert_bge_embedding(pool, target_kind, target_id, model, vector).await;
    }

    let mut tx = pool.begin().await?;
    let pgv = pgvector::Vector::from(vector.clone());
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
    .execute(&mut *tx)
    .await?;

    insert_ann_projection(&mut tx, target_kind, target_id, model, &vector).await?;
    tx.commit().await?;
    Ok(())
}

async fn insert_bge_embedding(
    pool: &PgPool,
    target_kind: &'static str,
    target_id: Uuid,
    model: &EmbeddingModel,
    vector: Vec<f32>,
) -> Result<(), StorageError> {
    if target_kind != target::THOUGHT && target_kind != target::ARTIFACT_CHUNK {
        return Err(StorageError::UnsupportedBgeTargetKind(
            target_kind.to_string(),
        ));
    }
    if vector.len() != bge::DIMS {
        return Err(StorageError::InvalidEmbeddingDimensions {
            model_id: model.id.clone(),
            expected: bge::DIMS,
            got: vector.len(),
        });
    }

    let pgv = pgvector::Vector::from(vector);
    let mut tx = pool.begin().await?;
    if target_kind == target::THOUGHT {
        sqlx::query(
            r#"
            INSERT INTO thought_embeddings_bge_m3 (
                thought_id,
                model_id,
                model_version,
                dimensions,
                embedding
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (thought_id, model_id, model_version)
            DO UPDATE SET
                dimensions = EXCLUDED.dimensions,
                embedding = EXCLUDED.embedding,
                created_at = NOW()
            "#,
        )
        .bind(target_id)
        .bind(bge::MODEL_ID)
        .bind(bge::MODEL_VERSION)
        .bind(bge::DIMS_I32)
        .bind(pgv)
        .execute(&mut *tx)
        .await?;
    } else {
        sqlx::query(
            r#"
            INSERT INTO artifact_chunk_embeddings_bge_m3 (
                chunk_id,
                model_id,
                model_version,
                dimensions,
                embedding
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (chunk_id, model_id, model_version)
            DO UPDATE SET
                dimensions = EXCLUDED.dimensions,
                embedding = EXCLUDED.embedding,
                created_at = NOW()
            "#,
        )
        .bind(target_id)
        .bind(bge::MODEL_ID)
        .bind(bge::MODEL_VERSION)
        .bind(bge::DIMS_I32)
        .bind(pgv)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

async fn insert_ann_projection(
    tx: &mut PgTransaction<'_>,
    target_kind: &'static str,
    target_id: Uuid,
    model: &EmbeddingModel,
    vector: &[f32],
) -> Result<(), StorageError> {
    let Some(projection) = ann_projection_for_model(model) else {
        return Ok(());
    };

    let halfvec = project_halfvec_3072(vector, projection.dimensions)?;
    sqlx::query(
        r#"
        INSERT INTO embedding_ann_projections (
            source_embedding_id,
            target_kind,
            target_id,
            model_id,
            model_version,
            projection_id,
            dimensions,
            embedding
        )
        SELECT
            e.id,
            e.target_kind,
            e.target_id,
            e.model_id,
            e.model_version,
            $5,
            $6,
            $7
        FROM embeddings e
        WHERE e.target_kind = $1
          AND e.target_id = $2
          AND e.model_id = $3
          AND e.model_version = $4
        ON CONFLICT (target_kind, target_id, model_id, model_version, projection_id)
        DO UPDATE SET
            source_embedding_id = EXCLUDED.source_embedding_id,
            dimensions = EXCLUDED.dimensions,
            embedding = EXCLUDED.embedding
        "#,
    )
    .bind(target_kind)
    .bind(target_id)
    .bind(&model.id)
    .bind(1_i32)
    .bind(&projection.projection_id)
    .bind(ann::HALF_3072_DIMS_I32)
    .bind(halfvec)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Reconcile the ANN projection sidecar for the active embedding model and
/// assert full coverage. This is the deploy-window heal step: migration 0013
/// backfills existing rows, atomic writes cover new rows after this PR, and
/// this function catches any raw embeddings inserted between migrate and
/// deploy or by future drift.
pub async fn reconcile_ann_projections(
    pool: &PgPool,
    model: &EmbeddingModel,
) -> Result<Option<AnnProjectionCoverage>, StorageError> {
    let Some(projection) = ann_projection_for_model(model) else {
        return Ok(None);
    };

    let result = sqlx::query(
        r#"
        INSERT INTO embedding_ann_projections (
            source_embedding_id,
            target_kind,
            target_id,
            model_id,
            model_version,
            projection_id,
            dimensions,
            embedding
        )
        SELECT
            e.id,
            e.target_kind,
            e.target_id,
            e.model_id,
            e.model_version,
            $2,
            $3,
            (l2_normalize(subvector(e.vector, 1, 3072)::vector(3072)))::halfvec(3072)
        FROM embeddings e
        WHERE e.model_id = $1
          AND vector_dims(e.vector) >= 3072
          AND NOT EXISTS (
              SELECT 1
              FROM embedding_ann_projections p
              WHERE p.source_embedding_id = e.id
                AND p.projection_id = $2
          )
        ON CONFLICT (target_kind, target_id, model_id, model_version, projection_id)
        DO UPDATE SET
            source_embedding_id = EXCLUDED.source_embedding_id,
            dimensions = EXCLUDED.dimensions,
            embedding = EXCLUDED.embedding
        "#,
    )
    .bind(&model.id)
    .bind(&projection.projection_id)
    .bind(ann::HALF_3072_DIMS_I32)
    .execute(pool)
    .await?;

    let inserted_missing = result.rows_affected() as i64;
    let coverage =
        record_ann_projection_coverage(pool, model, &projection, inserted_missing).await?;
    assert_ann_projection_coverage_ok(&coverage)?;
    Ok(Some(coverage))
}

/// Re-count ANN projection coverage without inserting missing rows. Worker
/// ticks use this as the periodic SLO assertion; tests use it to prove raw-only
/// drift is detected instead of being silently hidden by projection search.
pub async fn assert_ann_projection_coverage(
    pool: &PgPool,
    model: &EmbeddingModel,
) -> Result<Option<AnnProjectionCoverage>, StorageError> {
    let Some(projection) = ann_projection_for_model(model) else {
        return Ok(None);
    };

    let coverage = record_ann_projection_coverage(pool, model, &projection, 0).await?;
    assert_ann_projection_coverage_ok(&coverage)?;
    Ok(Some(coverage))
}

async fn record_ann_projection_coverage(
    pool: &PgPool,
    model: &EmbeddingModel,
    projection: &AnnProjection,
    inserted_missing: i64,
) -> Result<AnnProjectionCoverage, StorageError> {
    let (embedding_count, projection_count, missing_count): (i64, i64, i64) = sqlx::query_as(
        r#"
        SELECT
            (
                SELECT COUNT(*)
                FROM embeddings e
                WHERE e.model_id = $1
                  AND vector_dims(e.vector) >= 3072
            )::bigint AS embedding_count,
            (
                SELECT COUNT(*)
                FROM embedding_ann_projections p
                WHERE p.projection_id = $2
                  AND p.model_id = $1
            )::bigint AS projection_count,
            (
                SELECT COUNT(*)
                FROM embeddings e
                WHERE e.model_id = $1
                  AND vector_dims(e.vector) >= 3072
                  AND NOT EXISTS (
                      SELECT 1
                      FROM embedding_ann_projections p
                      WHERE p.source_embedding_id = e.id
                        AND p.projection_id = $2
                  )
            )::bigint AS missing_count
        "#,
    )
    .bind(&model.id)
    .bind(&projection.projection_id)
    .fetch_one(pool)
    .await?;

    let status = if missing_count == 0 && projection_count == embedding_count {
        "ok".to_string()
    } else {
        "diverged".to_string()
    };

    sqlx::query(
        r#"
        INSERT INTO embedding_ann_projection_coverage (
            projection_id,
            model_id,
            model_version,
            embedding_count,
            projection_count,
            missing_count,
            status,
            last_reconciled_at,
            last_checked_at
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            $5,
            $6,
            $7,
            CASE WHEN $8::bigint > 0 THEN NOW() ELSE NULL END,
            NOW()
        )
        ON CONFLICT (projection_id) DO UPDATE SET
            model_id = EXCLUDED.model_id,
            model_version = EXCLUDED.model_version,
            embedding_count = EXCLUDED.embedding_count,
            projection_count = EXCLUDED.projection_count,
            missing_count = EXCLUDED.missing_count,
            status = EXCLUDED.status,
            last_reconciled_at = CASE
                WHEN $8::bigint > 0 THEN NOW()
                ELSE embedding_ann_projection_coverage.last_reconciled_at
            END,
            last_checked_at = NOW()
        "#,
    )
    .bind(&projection.projection_id)
    .bind(&model.id)
    .bind(1_i32)
    .bind(embedding_count)
    .bind(projection_count)
    .bind(missing_count)
    .bind(&status)
    .bind(inserted_missing)
    .execute(pool)
    .await?;

    let coverage = AnnProjectionCoverage {
        projection_id: projection.projection_id.clone(),
        model_id: model.id.clone(),
        model_version: 1,
        embedding_count,
        projection_count,
        missing_count,
        inserted_missing,
        status,
    };

    if coverage.status == "ok" {
        tracing::info!(
            projection_id = %coverage.projection_id,
            model_id = %coverage.model_id,
            embedding_count = coverage.embedding_count,
            projection_count = coverage.projection_count,
            inserted_missing = coverage.inserted_missing,
            "ANN projection coverage ok",
        );
    } else {
        tracing::error!(
            projection_id = %coverage.projection_id,
            model_id = %coverage.model_id,
            embedding_count = coverage.embedding_count,
            projection_count = coverage.projection_count,
            missing_count = coverage.missing_count,
            inserted_missing = coverage.inserted_missing,
            "ANN projection coverage diverged",
        );
    }

    Ok(coverage)
}

fn assert_ann_projection_coverage_ok(coverage: &AnnProjectionCoverage) -> Result<(), StorageError> {
    if coverage.missing_count == 0 && coverage.projection_count == coverage.embedding_count {
        Ok(())
    } else {
        Err(StorageError::AnnProjectionCoverageMismatch {
            projection_id: coverage.projection_id.clone(),
            embedding_count: coverage.embedding_count,
            projection_count: coverage.projection_count,
            missing_count: coverage.missing_count,
        })
    }
}

/// Convenience: insert an embedding tied to a thought, taking the kengram-core
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

/// Convenience: insert an embedding tied to an artifact chunk.
pub async fn insert_artifact_chunk_embedding(
    pool: &PgPool,
    chunk_id: Uuid,
    embedding: &Embedding,
) -> Result<(), StorageError> {
    insert_embedding(
        pool,
        target::ARTIFACT_CHUNK,
        chunk_id,
        &embedding.model,
        embedding.vector.clone(),
    )
    .await
}

pub async fn select_artifact_chunk_context_generation_sources(
    pool: &PgPool,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    generator_id: &str,
    generator_version: i32,
    prompt_hash: &str,
    limit: i64,
) -> Result<Vec<ContextGenerationSource>, StorageError> {
    if limit <= 0 {
        return Ok(Vec::new());
    }

    let rows: Vec<ContextGenerationSourceRow> = sqlx::query_as(
        r#"
        SELECT ac.id AS chunk_id,
               ac.source_thought_id,
               t.scope,
               t.source AS parent_source,
               t.created_at AS parent_created_at,
               t.metadata AS parent_metadata,
               t.content AS parent_content,
               ac.chunk_index,
               ac.content AS chunk_content,
               ac.metadata AS chunk_metadata,
               ac.content_fingerprint AS raw_chunk_fingerprint
        FROM artifact_chunks ac
        JOIN thoughts t ON t.id = ac.source_thought_id
        WHERE ac.retracted_at IS NULL
          AND ac.source_thought_id IS NOT NULL
          AND t.retracted_at IS NULL
          AND ($1::text IS NULL OR t.scope = $1)
          AND ($2::text IS NULL OR t.scope LIKE $2 || '%')
          AND t.id <> ALL($6::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $7
          AND t.content !~ $8
          AND ac.content !~ $8
          AND NOT EXISTS (
              SELECT 1
              FROM artifact_chunk_contexts c
              WHERE c.chunk_id = ac.id
                AND c.generator_id = $3
                AND c.generator_version = $4
                AND c.prompt_hash = $5
                AND c.raw_chunk_fingerprint = ac.content_fingerprint
                AND c.retracted_at IS NULL
          )
        ORDER BY t.created_at ASC, ac.chunk_index ASC
        LIMIT $9
        "#,
    )
    .bind(scope)
    .bind(scope_prefix)
    .bind(generator_id)
    .bind(generator_version)
    .bind(prompt_hash)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(context_generation_source_from_row)
        .collect()
}

pub async fn insert_artifact_chunk_context(
    pool: &PgPool,
    input: ArtifactChunkContextInsert,
) -> Result<ArtifactChunkContextOutcome, StorageError> {
    let row: ContextGenerationSourceRow = sqlx::query_as(
        r#"
        SELECT ac.id AS chunk_id,
               ac.source_thought_id,
               t.scope,
               t.source AS parent_source,
               t.created_at AS parent_created_at,
               t.metadata AS parent_metadata,
               t.content AS parent_content,
               ac.chunk_index,
               ac.content AS chunk_content,
               ac.metadata AS chunk_metadata,
               ac.content_fingerprint AS raw_chunk_fingerprint
        FROM artifact_chunks ac
        JOIN thoughts t ON t.id = ac.source_thought_id
        WHERE ac.id = $1
          AND ac.retracted_at IS NULL
          AND ac.source_thought_id IS NOT NULL
          AND t.retracted_at IS NULL
        "#,
    )
    .bind(input.chunk_id)
    .fetch_one(pool)
    .await?;
    let source = context_generation_source_from_row(row)?;

    let mut rejection_reason = contamination_rejection_reason(
        source.source_thought_id,
        source.parent_metadata.as_value(),
        &source.parent_content,
        &source.chunk_content,
    );
    if rejection_reason.is_none() && text_trips_eval_contamination(&input.context_text) {
        rejection_reason = Some("generated_context_eval_marker".to_string());
    }
    let trimmed_context = input.context_text.trim();
    if rejection_reason.is_none() && trimmed_context.is_empty() {
        rejection_reason = Some("empty_generated_context".to_string());
    }
    if rejection_reason.is_none() && trimmed_context.chars().count() > MAX_CONTEXTUAL_CONTEXT_CHARS
    {
        rejection_reason = Some("overlong_generated_context".to_string());
    }

    let (status, context_text, contextual_content, rejection_reason) =
        if let Some(reason) = rejection_reason {
            ("rejected", String::new(), String::new(), Some(reason))
        } else {
            let context_text = trimmed_context.to_string();
            let contextual_content = format!("{context_text}\n\n{}", source.chunk_content);
            ("ready", context_text, contextual_content, None)
        };

    let row: (Uuid, String, Option<String>) = sqlx::query_as(
        r#"
        INSERT INTO artifact_chunk_contexts (
            chunk_id,
            source_thought_id,
            context_text,
            contextual_content,
            raw_chunk_fingerprint,
            contextual_content_fingerprint,
            generator_id,
            generator_version,
            prompt_version,
            prompt_hash,
            model_id,
            model_version,
            contamination_filter_version,
            pipeline_run_id,
            status,
            rejection_reason,
            metadata
        )
        VALUES (
            $1, $2, $3, $4, $5, digest($4, 'sha256'), $6, $7, $8, $9, $10, $11,
            $12, $13, $14, $15, $16
        )
        ON CONFLICT (
            chunk_id,
            generator_id,
            generator_version,
            prompt_hash,
            raw_chunk_fingerprint
        )
        WHERE retracted_at IS NULL
        DO UPDATE SET
            context_text = EXCLUDED.context_text,
            contextual_content = EXCLUDED.contextual_content,
            contextual_content_fingerprint = EXCLUDED.contextual_content_fingerprint,
            prompt_version = EXCLUDED.prompt_version,
            model_id = EXCLUDED.model_id,
            model_version = EXCLUDED.model_version,
            contamination_filter_version = EXCLUDED.contamination_filter_version,
            pipeline_run_id = EXCLUDED.pipeline_run_id,
            status = EXCLUDED.status,
            rejection_reason = EXCLUDED.rejection_reason,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING id, status, rejection_reason
        "#,
    )
    .bind(source.chunk_id)
    .bind(source.source_thought_id.into_uuid())
    .bind(&context_text)
    .bind(&contextual_content)
    .bind(source.raw_chunk_fingerprint.to_vec())
    .bind(&input.generator_id)
    .bind(input.generator_version)
    .bind(&input.prompt_version)
    .bind(&input.prompt_hash)
    .bind(&input.model_id)
    .bind(&input.model_version)
    .bind(CONTEXTUAL_CONTAMINATION_FILTER_VERSION)
    .bind(input.pipeline_run_id)
    .bind(status)
    .bind(&rejection_reason)
    .bind(&input.metadata)
    .fetch_one(pool)
    .await?;

    Ok(ArtifactChunkContextOutcome {
        context_id: row.0,
        status: row.1,
        rejection_reason: row.2,
    })
}

pub async fn insert_artifact_chunk_context_embedding(
    pool: &PgPool,
    context_id: Uuid,
    embedding: &Embedding,
) -> Result<bool, StorageError> {
    if !is_bge_m3_1024(&embedding.model) {
        return Err(StorageError::InvalidEmbeddingDimensions {
            model_id: embedding.model.id.clone(),
            expected: bge::DIMS,
            got: embedding.model.dimensions,
        });
    }
    if embedding.vector.len() != bge::DIMS {
        return Err(StorageError::InvalidEmbeddingDimensions {
            model_id: embedding.model.id.clone(),
            expected: bge::DIMS,
            got: embedding.vector.len(),
        });
    }

    let pgv = pgvector::Vector::from(embedding.vector.clone());
    let result = sqlx::query(
        r#"
        INSERT INTO artifact_chunk_context_embeddings_bge_m3 (
            context_id,
            model_id,
            model_version,
            dimensions,
            embedding
        )
        SELECT c.id, $2, $3, $4, $5
        FROM artifact_chunk_contexts c
        JOIN artifact_chunks ac ON ac.id = c.chunk_id
        JOIN thoughts t ON t.id = c.source_thought_id
        WHERE c.id = $1
          AND c.status = 'ready'
          AND c.retracted_at IS NULL
          AND ac.retracted_at IS NULL
          AND t.retracted_at IS NULL
          AND t.id <> ALL($6::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $7
          AND t.content !~ $8
          AND ac.content !~ $8
          AND c.context_text !~ $8
          AND c.contextual_content !~ $8
        ON CONFLICT (context_id, model_id, model_version)
        DO UPDATE SET
            dimensions = EXCLUDED.dimensions,
            embedding = EXCLUDED.embedding,
            updated_at = NOW()
        "#,
    )
    .bind(context_id)
    .bind(bge::MODEL_ID)
    .bind(bge::MODEL_VERSION)
    .bind(bge::DIMS_I32)
    .bind(pgv)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

async fn insert_bge_sparse_embedding(
    pool: &PgPool,
    target_kind: &'static str,
    target_id: Uuid,
    content_fingerprint: [u8; 32],
    source_content_chars: i32,
    vector: &SparseLexicalVector,
    provenance: &SparseEmbeddingProvenance,
) -> Result<(), StorageError> {
    if target_kind != target::THOUGHT && target_kind != target::ARTIFACT_CHUNK {
        return Err(StorageError::UnsupportedBgeSparseTargetKind(
            target_kind.to_string(),
        ));
    }
    if source_content_chars < 0 {
        return Err(StorageError::InvalidSparseSourceContentChars(
            source_content_chars,
        ));
    }
    validate_bge_m3_sparse(vector)?;

    let sparsevec_literal = vector.sparsevec_literal();
    let content_fingerprint = content_fingerprint.to_vec();
    let nonzero_count = vector.nonzero_count() as i32;
    if target_kind == target::THOUGHT {
        sqlx::query(
            r#"
            INSERT INTO thought_sparse_embeddings_bge_m3 (
                thought_id,
                model_id,
                model_version,
                source_model,
                vocab_size,
                nonzero_count,
                content_fingerprint,
                source_content_chars,
                generator,
                generator_version,
                pipeline_run_id,
                producer_metadata,
                embedding
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13::text::sparsevec
            )
            ON CONFLICT (thought_id, model_id, model_version)
            DO UPDATE SET
                source_model = EXCLUDED.source_model,
                vocab_size = EXCLUDED.vocab_size,
                nonzero_count = EXCLUDED.nonzero_count,
                content_fingerprint = EXCLUDED.content_fingerprint,
                source_content_chars = EXCLUDED.source_content_chars,
                generator = EXCLUDED.generator,
                generator_version = EXCLUDED.generator_version,
                pipeline_run_id = EXCLUDED.pipeline_run_id,
                producer_metadata = EXCLUDED.producer_metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            "#,
        )
        .bind(target_id)
        .bind(bge_sparse::MODEL_ID)
        .bind(bge_sparse::MODEL_VERSION)
        .bind(&provenance.source_model)
        .bind(bge_sparse::VOCAB_SIZE_I32)
        .bind(nonzero_count)
        .bind(&content_fingerprint)
        .bind(source_content_chars)
        .bind(&provenance.generator)
        .bind(&provenance.generator_version)
        .bind(provenance.pipeline_run_id)
        .bind(&provenance.producer_metadata)
        .bind(&sparsevec_literal)
        .execute(pool)
        .await?;
    } else {
        sqlx::query(
            r#"
            INSERT INTO artifact_chunk_sparse_embeddings_bge_m3 (
                chunk_id,
                model_id,
                model_version,
                source_model,
                vocab_size,
                nonzero_count,
                content_fingerprint,
                source_content_chars,
                generator,
                generator_version,
                pipeline_run_id,
                producer_metadata,
                embedding
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13::text::sparsevec
            )
            ON CONFLICT (chunk_id, model_id, model_version)
            DO UPDATE SET
                source_model = EXCLUDED.source_model,
                vocab_size = EXCLUDED.vocab_size,
                nonzero_count = EXCLUDED.nonzero_count,
                content_fingerprint = EXCLUDED.content_fingerprint,
                source_content_chars = EXCLUDED.source_content_chars,
                generator = EXCLUDED.generator,
                generator_version = EXCLUDED.generator_version,
                pipeline_run_id = EXCLUDED.pipeline_run_id,
                producer_metadata = EXCLUDED.producer_metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            "#,
        )
        .bind(target_id)
        .bind(bge_sparse::MODEL_ID)
        .bind(bge_sparse::MODEL_VERSION)
        .bind(&provenance.source_model)
        .bind(bge_sparse::VOCAB_SIZE_I32)
        .bind(nonzero_count)
        .bind(&content_fingerprint)
        .bind(source_content_chars)
        .bind(&provenance.generator)
        .bind(&provenance.generator_version)
        .bind(provenance.pipeline_run_id)
        .bind(&provenance.producer_metadata)
        .bind(&sparsevec_literal)
        .execute(pool)
        .await?;
    }

    Ok(())
}

pub async fn insert_thought_sparse_embedding(
    pool: &PgPool,
    thought_id: ThoughtId,
    content_fingerprint: [u8; 32],
    source_content_chars: i32,
    vector: &SparseLexicalVector,
    provenance: &SparseEmbeddingProvenance,
) -> Result<(), StorageError> {
    insert_bge_sparse_embedding(
        pool,
        target::THOUGHT,
        thought_id.into_uuid(),
        content_fingerprint,
        source_content_chars,
        vector,
        provenance,
    )
    .await
}

pub async fn insert_artifact_chunk_sparse_embedding(
    pool: &PgPool,
    chunk_id: Uuid,
    content_fingerprint: [u8; 32],
    source_content_chars: i32,
    vector: &SparseLexicalVector,
    provenance: &SparseEmbeddingProvenance,
) -> Result<(), StorageError> {
    insert_bge_sparse_embedding(
        pool,
        target::ARTIFACT_CHUNK,
        chunk_id,
        content_fingerprint,
        source_content_chars,
        vector,
        provenance,
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

/// Look up the text body for an artifact chunk. Returns `None` when the chunk
/// is missing or retracted.
pub async fn fetch_artifact_chunk_content(
    pool: &PgPool,
    id: Uuid,
) -> Result<Option<String>, StorageError> {
    let row = sqlx::query_as::<_, (String,)>(
        r#"
        SELECT content
        FROM artifact_chunks
        WHERE id = $1
          AND retracted_at IS NULL
        "#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.0))
}

/// True if an embedding exists for the given thought under the given model.
pub async fn thought_has_embedding(
    pool: &PgPool,
    id: ThoughtId,
    model: &EmbeddingModel,
) -> Result<bool, StorageError> {
    if is_bge_m3_1024(model) {
        let (exists,): (bool,) = sqlx::query_as(
            r#"
            SELECT EXISTS (
                SELECT 1
                FROM thought_embeddings_bge_m3
                WHERE thought_id = $1
                  AND model_id = $2
                  AND model_version = $3
            )
            "#,
        )
        .bind(id.into_uuid())
        .bind(bge::MODEL_ID)
        .bind(bge::MODEL_VERSION)
        .fetch_one(pool)
        .await?;
        return Ok(exists);
    }

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
    if is_bge_m3_1024(model) {
        let row: Option<ThoughtProvenanceRow> = sqlx::query_as(
            r#"
            SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
                   t.content_fingerprint, t.tags,
                   t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
                   t.retracted_at, t.retracted_reason,
                   b.created_at AS embedded_at
            FROM thoughts t
            LEFT JOIN thought_embeddings_bge_m3 b
              ON b.thought_id = t.id
             AND b.model_id = $2
             AND b.model_version = $3
            WHERE t.id = $1
            "#,
        )
        .bind(id.into_uuid())
        .bind(bge::MODEL_ID)
        .bind(bge::MODEL_VERSION)
        .fetch_optional(pool)
        .await?;

        return row.map(thought_provenance_row_to_result).transpose();
    }

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

    row.map(|r| {
        thought_provenance_row_to_result(ThoughtProvenanceRow {
            id: r.id,
            scope: r.scope,
            content: r.content,
            source: r.source,
            created_at: r.created_at,
            metadata: r.metadata,
            content_fingerprint: r.content_fingerprint,
            tags: r.tags,
            tags_extractor_model: r.tags_extractor_model,
            tags_extractor_version: r.tags_extractor_version,
            tags_extracted_at: r.tags_extracted_at,
            retracted_at: r.retracted_at,
            retracted_reason: r.retracted_reason,
            embedded_at: r.embedded_at,
        })
    })
    .transpose()
}

#[derive(sqlx::FromRow)]
struct ThoughtProvenanceRow {
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
    retracted_at: Option<OffsetDateTime>,
    retracted_reason: Option<String>,
    embedded_at: Option<OffsetDateTime>,
}

fn thought_provenance_row_to_result(
    r: ThoughtProvenanceRow,
) -> Result<ThoughtWithProvenance, StorageError> {
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

    Ok(ThoughtWithProvenance {
        thought,
        embedding_status,
        embedded_at: r.embedded_at,
        retracted_at: r.retracted_at,
        retracted_reason: r.retracted_reason,
    })
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
          AND id <> ALL($5::uuid[])
          AND lower(coalesce(metadata->>'source_file', '')) !~ $6
          AND content !~ $7
        ORDER BY similarity(content, $1) DESC
        LIMIT $4
        "#,
        query,
        scope,
        scope_prefix,
        limit,
        EVAL_CONTAMINATION_KNOWN_IDS,
        EVAL_CONTAMINATION_SOURCE_FILE_REGEX,
        EVAL_CONTAMINATION_CONTENT_REGEX,
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

/// Full-text lexical search over `thoughts.content`, ranked by
/// `ts_rank_cd`. This is the production lexical leg for hybrid search; it
/// should use the `thoughts_content_fts_idx` GIN expression index from
/// migration 0014.
pub async fn search_fts(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    let Some(query) = normalize_fts_query(query) else {
        return Ok(Vec::new());
    };

    let rows = sqlx::query!(
        r#"
        WITH fts AS (
            SELECT websearch_to_tsquery('english', $1) AS tsq
        )
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
               ts_rank_cd(to_tsvector('english', t.content), fts.tsq) AS "rank!: f32"
        FROM thoughts t
        CROSS JOIN fts
        WHERE to_tsvector('english', t.content) @@ fts.tsq
          AND ($2::text IS NULL OR t.scope = $2)
          AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
          AND t.retracted_at IS NULL
          AND t.id <> ALL($5::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $6
          AND t.content !~ $7
        ORDER BY ts_rank_cd(to_tsvector('english', t.content), fts.tsq) DESC,
                 t.created_at DESC
        LIMIT $4
        "#,
        query,
        scope,
        scope_prefix,
        limit,
        EVAL_CONTAMINATION_KNOWN_IDS,
        EVAL_CONTAMINATION_SOURCE_FILE_REGEX,
        EVAL_CONTAMINATION_CONTENT_REGEX,
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
            Ok(Hit::from_lexical_leg(thought, r.rank))
        })
        .collect()
}

/// FTS lexical search bounded by a transaction-local statement timeout. This
/// should be a belt over the indexed FTS path, not the main performance
/// mechanism. Callers should soft-fail timeout/budget errors when a faster
/// leg has usable hits.
pub async fn search_fts_bounded(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
    timeout_ms: u64,
) -> Result<Vec<Hit>, StorageError> {
    let Some(query) = normalize_fts_query(query) else {
        return Ok(Vec::new());
    };

    let mut tx = pool.begin().await?;
    if let Err(e) = set_statement_timeout(&mut tx, timeout_ms).await {
        if let Err(rollback_err) = tx.rollback().await {
            tracing::warn!(
                error = %rollback_err,
                "failed to roll back bounded FTS transaction after statement_timeout setup error",
            );
        }
        return Err(e.into());
    }

    let rows = match sqlx::query!(
        r#"
        WITH fts AS (
            SELECT websearch_to_tsquery('english', $1) AS tsq
        )
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
               ts_rank_cd(to_tsvector('english', t.content), fts.tsq) AS "rank!: f32"
        FROM thoughts t
        CROSS JOIN fts
        WHERE to_tsvector('english', t.content) @@ fts.tsq
          AND ($2::text IS NULL OR t.scope = $2)
          AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
          AND t.retracted_at IS NULL
          AND t.id <> ALL($5::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $6
          AND t.content !~ $7
        ORDER BY ts_rank_cd(to_tsvector('english', t.content), fts.tsq) DESC,
                 t.created_at DESC
        LIMIT $4
        "#,
        query,
        scope,
        scope_prefix,
        limit,
        EVAL_CONTAMINATION_KNOWN_IDS,
        EVAL_CONTAMINATION_SOURCE_FILE_REGEX,
        EVAL_CONTAMINATION_CONTENT_REGEX,
    )
    .fetch_all(&mut *tx)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(rollback_err) = tx.rollback().await {
                tracing::warn!(
                    error = %rollback_err,
                    "failed to roll back bounded FTS transaction after query error",
                );
            }
            return Err(e.into());
        }
    };
    tx.commit().await?;

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
            Ok(Hit::from_lexical_leg(thought, r.rank))
        })
        .collect()
}

/// Soft domain routing leg for the full-pipeline canary path. This never
/// filters baseline candidates; callers fuse it with dense/FTS legs via RRF.
pub async fn search_domain_scope_aliases_bounded(
    pool: &PgPool,
    domains: &[String],
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
    timeout_ms: u64,
) -> Result<Vec<Hit>, StorageError> {
    if domains.is_empty() {
        return Ok(Vec::new());
    }

    let mut tx = pool.begin().await?;
    if let Err(e) = set_statement_timeout(&mut tx, timeout_ms).await {
        if let Err(rollback_err) = tx.rollback().await {
            tracing::warn!(
                error = %rollback_err,
                "failed to roll back bounded domain-scope transaction after statement_timeout setup error",
            );
        }
        return Err(e.into());
    }

    let rows = match sqlx::query_as::<_, LexicalSearchRow>(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
               GREATEST(COALESCE(MAX(tsa.confidence), 0.0), 0.5)::real AS rank
        FROM thoughts t
        LEFT JOIN thought_scope_aliases tsa
          ON tsa.thought_id = t.id
         AND tsa.axis = 'domain'
         AND tsa.retracted_at IS NULL
         AND tsa.scope = ANY($1::text[])
        WHERE t.retracted_at IS NULL
          AND ($2::text IS NULL OR t.scope = $2)
          AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
          AND (
              tsa.thought_id IS NOT NULL
              OR (
                  jsonb_typeof(t.tags->'domain_scope') = 'string'
                  AND t.tags->>'domain_scope' = ANY($1::text[])
              )
          )
          AND t.id <> ALL($5::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $6
          AND t.content !~ $7
        GROUP BY t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
                 t.content_fingerprint, t.tags,
                 t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at
        ORDER BY rank DESC, t.created_at DESC
        LIMIT $4
        "#,
    )
    .bind(domains)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(rollback_err) = tx.rollback().await {
                tracing::warn!(
                    error = %rollback_err,
                    "failed to roll back bounded domain-scope transaction after query error",
                );
            }
            return Err(e.into());
        }
    };
    tx.commit().await?;
    lexical_rows_to_hits(rows)
}

/// Soft tag/retrieval-alias leg for the full-pipeline canary path.
pub async fn search_tag_facets_bounded(
    pool: &PgPool,
    terms: &[String],
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
    timeout_ms: u64,
) -> Result<Vec<Hit>, StorageError> {
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut tx = pool.begin().await?;
    if let Err(e) = set_statement_timeout(&mut tx, timeout_ms).await {
        if let Err(rollback_err) = tx.rollback().await {
            tracing::warn!(
                error = %rollback_err,
                "failed to roll back bounded tag-facet transaction after statement_timeout setup error",
            );
        }
        return Err(e.into());
    }

    let rows = match sqlx::query_as::<_, LexicalSearchRow>(
        r#"
        WITH wanted AS (
            SELECT lower(unnest($1::text[])) AS term
        ),
        matched AS (
            SELECT t.id, count(*)::real AS rank
            FROM thoughts t
            JOIN wanted w ON
                EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(
                        CASE WHEN jsonb_typeof(t.tags->'retrieval_aliases') = 'array'
                             THEN t.tags->'retrieval_aliases'
                             ELSE '[]'::jsonb END
                    ) AS alias(value)
                    WHERE lower(alias.value) = w.term
                )
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(
                        CASE WHEN jsonb_typeof(t.tags->'topics') = 'array'
                             THEN t.tags->'topics'
                             ELSE '[]'::jsonb END
                    ) AS topic(value)
                    WHERE lower(topic.value) = w.term
                )
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(
                        CASE WHEN jsonb_typeof(t.tags->'entities') = 'array'
                             THEN t.tags->'entities'
                             ELSE '[]'::jsonb END
                    ) AS entity(value)
                    WHERE lower(entity.value) = w.term
                )
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(
                        CASE WHEN jsonb_typeof(t.tags->'people') = 'array'
                             THEN t.tags->'people'
                             ELSE '[]'::jsonb END
                    ) AS person(value)
                    WHERE lower(person.value) = w.term
                )
                OR lower(t.tags->>'kind') = w.term
            WHERE t.retracted_at IS NULL
              AND ($2::text IS NULL OR t.scope = $2)
              AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
              AND t.id <> ALL($5::uuid[])
              AND lower(coalesce(t.metadata->>'source_file', '')) !~ $6
              AND t.content !~ $7
            GROUP BY t.id
        )
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
               matched.rank AS rank
        FROM matched
        JOIN thoughts t ON t.id = matched.id
        ORDER BY matched.rank DESC, t.created_at DESC
        LIMIT $4
        "#,
    )
    .bind(terms)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(rollback_err) = tx.rollback().await {
                tracing::warn!(
                    error = %rollback_err,
                    "failed to roll back bounded tag-facet transaction after query error",
                );
            }
            return Err(e.into());
        }
    };
    tx.commit().await?;
    lexical_rows_to_hits(rows)
}

/// FTS lexical search over artifact chunks. Each result resolves to the
/// source parent thought and carries the best matching chunk as provenance.
pub async fn search_artifact_chunks_fts_bounded(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
    timeout_ms: u64,
) -> Result<Vec<Hit>, StorageError> {
    let Some(query) = normalize_fts_query(query) else {
        return Ok(Vec::new());
    };
    if !index_ready(pool, "artifact_chunks_content_fts_idx").await? {
        return Err(StorageError::BgeSidecarIndexNotReady(
            "artifact_chunks_content_fts_idx".to_string(),
        ));
    }

    let mut tx = pool.begin().await?;
    if let Err(e) = set_statement_timeout(&mut tx, timeout_ms).await {
        if let Err(rollback_err) = tx.rollback().await {
            tracing::warn!(
                error = %rollback_err,
                "failed to roll back bounded artifact-chunk FTS transaction after statement_timeout setup error",
            );
        }
        return Err(e.into());
    }

    let rows: Vec<ChunkLexicalSearchRow> = match sqlx::query_as(
        r#"
        WITH fts AS (
            SELECT websearch_to_tsquery('english', $1) AS tsq
        ),
        candidates AS (
            SELECT t.id,
                   t.scope,
                   t.content AS parent_content,
                   t.source,
                   t.created_at,
                   t.metadata AS parent_metadata,
                   t.content_fingerprint,
                   t.tags,
                   t.tags_extractor_model,
                   t.tags_extractor_version,
                   t.tags_extracted_at,
                   ac.id AS chunk_id,
                   ac.artifact_id,
                   ac.source_thought_id,
                   ac.chunk_index,
                   ac.content AS chunk_content,
                   ac.chunker_id,
                   ac.chunker_version,
                   ac.token_estimate,
                   ac.start_char,
                   ac.end_char,
                   ac.metadata AS chunk_metadata,
                   ts_rank_cd(to_tsvector('english', ac.content), fts.tsq) AS rank
            FROM artifact_chunks ac
            JOIN thoughts t ON t.id = ac.source_thought_id
            CROSS JOIN fts
            WHERE to_tsvector('english', ac.content) @@ fts.tsq
              AND ac.retracted_at IS NULL
              AND ac.source_thought_id IS NOT NULL
              AND t.retracted_at IS NULL
              AND ($2::text IS NULL OR t.scope = $2)
              AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
              AND t.id <> ALL($5::uuid[])
              AND lower(coalesce(t.metadata->>'source_file', '')) !~ $6
              AND t.content !~ $7
              AND ac.content !~ $7
            ORDER BY ts_rank_cd(to_tsvector('english', ac.content), fts.tsq) DESC,
                     t.created_at DESC,
                     ac.chunk_index ASC
            LIMIT GREATEST($4, $4 * 8)
        ),
        best_per_parent AS (
            SELECT DISTINCT ON (id) *
            FROM candidates
            ORDER BY id, rank DESC, chunk_index ASC
        )
        SELECT *
        FROM best_per_parent
        ORDER BY rank DESC, created_at DESC, chunk_index ASC
        LIMIT $4
        "#,
    )
    .bind(&query)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(rollback_err) = tx.rollback().await {
                tracing::warn!(
                    error = %rollback_err,
                    "failed to roll back bounded artifact-chunk FTS transaction after query error",
                );
            }
            return Err(e.into());
        }
    };
    tx.commit().await?;

    chunk_lexical_rows_to_hits(rows)
}

/// FTS lexical search over generated contextual chunk documents. Resolves to
/// parent thoughts like raw chunk search, but uses the immutable contextual
/// sidecar as ranking evidence and never exposes rejected context rows.
pub async fn search_artifact_chunk_contexts_fts_bounded(
    pool: &PgPool,
    query: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
    timeout_ms: u64,
) -> Result<Vec<Hit>, StorageError> {
    let Some(query) = normalize_fts_query(query) else {
        return Ok(Vec::new());
    };
    if !index_ready(pool, "artifact_chunk_contexts_ready_fts_idx").await? {
        return Err(StorageError::BgeSidecarIndexNotReady(
            "artifact_chunk_contexts_ready_fts_idx".to_string(),
        ));
    }

    let mut tx = pool.begin().await?;
    if let Err(e) = set_statement_timeout(&mut tx, timeout_ms).await {
        if let Err(rollback_err) = tx.rollback().await {
            tracing::warn!(
                error = %rollback_err,
                "failed to roll back bounded contextual-chunk FTS transaction after statement_timeout setup error",
            );
        }
        return Err(e.into());
    }

    let rows: Vec<ChunkLexicalSearchRow> = match sqlx::query_as(
        r#"
        WITH fts AS (
            SELECT websearch_to_tsquery('english', $1) AS tsq
        ),
        candidates AS (
            SELECT t.id,
                   t.scope,
                   t.content AS parent_content,
                   t.source,
                   t.created_at,
                   t.metadata AS parent_metadata,
                   t.content_fingerprint,
                   t.tags,
                   t.tags_extractor_model,
                   t.tags_extractor_version,
                   t.tags_extracted_at,
                   ac.id AS chunk_id,
                   ac.artifact_id,
                   ac.source_thought_id,
                   ac.chunk_index,
                   ac.content AS chunk_content,
                   ac.chunker_id,
                   ac.chunker_version,
                   ac.token_estimate,
                   ac.start_char,
                   ac.end_char,
                   jsonb_set(
                       ac.metadata,
                       '{contextual_retrieval}',
                       jsonb_build_object(
                           'context_id', c.id::text,
                           'generator_id', c.generator_id,
                           'generator_version', c.generator_version,
                           'prompt_version', c.prompt_version,
                           'contextual', true
                       ),
                       true
                   ) AS chunk_metadata,
                   ts_rank_cd(to_tsvector('english', c.contextual_content), fts.tsq) AS rank
            FROM artifact_chunk_contexts c
            JOIN artifact_chunks ac ON ac.id = c.chunk_id
            JOIN thoughts t ON t.id = c.source_thought_id
            CROSS JOIN fts
            WHERE to_tsvector('english', c.contextual_content) @@ fts.tsq
              AND c.status = 'ready'
              AND c.retracted_at IS NULL
              AND ac.retracted_at IS NULL
              AND ac.source_thought_id IS NOT NULL
              AND t.retracted_at IS NULL
              AND ($2::text IS NULL OR t.scope = $2)
              AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
              AND t.id <> ALL($5::uuid[])
              AND lower(coalesce(t.metadata->>'source_file', '')) !~ $6
              AND t.content !~ $7
              AND ac.content !~ $7
              AND c.context_text !~ $7
              AND c.contextual_content !~ $7
            ORDER BY ts_rank_cd(to_tsvector('english', c.contextual_content), fts.tsq) DESC,
                     t.created_at DESC,
                     ac.chunk_index ASC
            LIMIT GREATEST($4, $4 * 8)
        ),
        best_per_parent AS (
            SELECT DISTINCT ON (id) *
            FROM candidates
            ORDER BY id, rank DESC, chunk_index ASC
        )
        SELECT *
        FROM best_per_parent
        ORDER BY rank DESC, created_at DESC, chunk_index ASC
        LIMIT $4
        "#,
    )
    .bind(&query)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            if let Err(rollback_err) = tx.rollback().await {
                tracing::warn!(
                    error = %rollback_err,
                    "failed to roll back bounded contextual-chunk FTS transaction after query error",
                );
            }
            return Err(e.into());
        }
    };
    tx.commit().await?;

    chunk_lexical_rows_to_hits(rows)
}

/// Find thoughts that don't yet have an embedding row for the given model.
/// Oldest first — backfill should clear the backlog FIFO.
pub async fn find_unembedded_thoughts(
    pool: &PgPool,
    model: &EmbeddingModel,
    scope: Option<&str>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError> {
    if is_bge_m3_1024(model) {
        let rows: Vec<ThoughtRow> = sqlx::query_as(
            r#"
            SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
                   t.content_fingerprint, t.tags,
                   t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at
            FROM thoughts t
            LEFT JOIN thought_embeddings_bge_m3 b
              ON b.thought_id = t.id
             AND b.model_id = $1
             AND b.model_version = $2
            WHERE b.thought_id IS NULL
              AND ($3::text IS NULL OR t.scope = $3)
              AND t.retracted_at IS NULL
            ORDER BY t.created_at ASC
            LIMIT $4
            "#,
        )
        .bind(bge::MODEL_ID)
        .bind(bge::MODEL_VERSION)
        .bind(scope)
        .bind(limit)
        .fetch_all(pool)
        .await?;

        return rows.into_iter().map(thought_row_to_thought).collect();
    }

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
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<usize, StorageError> {
    if model_id == bge::MODEL_ID {
        let result = sqlx::query(
            r#"
            INSERT INTO pending_embeddings (target_kind, target_id, model_id)
            SELECT 'thought', t.id, $1
            FROM thoughts t
            LEFT JOIN thought_embeddings_bge_m3 b
              ON b.thought_id = t.id
             AND b.model_id = $1
             AND b.model_version = $2
            WHERE b.thought_id IS NULL
              AND ($3::text IS NULL OR t.scope = $3)
              AND ($4::text IS NULL OR t.scope LIKE $4 || '%')
              AND t.retracted_at IS NULL
            ORDER BY t.created_at ASC
            LIMIT $5
            ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
            "#,
        )
        .bind(model_id)
        .bind(bge::MODEL_VERSION)
        .bind(scope)
        .bind(scope_prefix)
        .bind(limit)
        .execute(pool)
        .await?;
        return Ok(result.rows_affected() as usize);
    }

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
          AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
          AND t.retracted_at IS NULL
        ORDER BY t.created_at ASC
        LIMIT $4
        ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
        "#,
        model_id,
        scope,
        scope_prefix,
        limit,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() as usize)
}

/// Heal-step companion to the worker: enqueue every unembedded artifact chunk.
/// Scope filters apply through the source parent thought. Chunks without
/// source_thought_id are skipped because they cannot resolve to a search hit.
pub async fn enqueue_unembedded_artifact_chunks(
    pool: &PgPool,
    model_id: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<usize, StorageError> {
    if model_id == bge::MODEL_ID {
        let result = sqlx::query(
            r#"
            INSERT INTO pending_embeddings (target_kind, target_id, model_id)
            SELECT 'artifact_chunk', ac.id, $1
            FROM artifact_chunks ac
            JOIN thoughts t ON t.id = ac.source_thought_id
            LEFT JOIN artifact_chunk_embeddings_bge_m3 b
              ON b.chunk_id = ac.id
             AND b.model_id = $1
             AND b.model_version = $2
            WHERE b.chunk_id IS NULL
              AND ac.source_thought_id IS NOT NULL
              AND ac.retracted_at IS NULL
              AND t.retracted_at IS NULL
              AND ($3::text IS NULL OR t.scope = $3)
              AND ($4::text IS NULL OR t.scope LIKE $4 || '%')
            ORDER BY ac.created_at ASC
            LIMIT $5
            ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
            "#,
        )
        .bind(model_id)
        .bind(bge::MODEL_VERSION)
        .bind(scope)
        .bind(scope_prefix)
        .bind(limit)
        .execute(pool)
        .await?;
        return Ok(result.rows_affected() as usize);
    }

    let result = sqlx::query(
        r#"
        INSERT INTO pending_embeddings (target_kind, target_id, model_id)
        SELECT 'artifact_chunk', ac.id, $1
        FROM artifact_chunks ac
        JOIN thoughts t ON t.id = ac.source_thought_id
        LEFT JOIN embeddings e
            ON e.target_kind = 'artifact_chunk'
           AND e.target_id = ac.id
           AND e.model_id = $1
        WHERE e.id IS NULL
          AND ac.source_thought_id IS NOT NULL
          AND ac.retracted_at IS NULL
          AND t.retracted_at IS NULL
          AND ($2::text IS NULL OR t.scope = $2)
          AND ($3::text IS NULL OR t.scope LIKE $3 || '%')
        ORDER BY ac.created_at ASC
        LIMIT $4
        ON CONFLICT (target_kind, target_id, model_id) DO NOTHING
        "#,
    )
    .bind(model_id)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
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

/// One row of a pre-retag tag snapshot — the tag provenance of a single
/// non-retracted thought, captured before a destructive retag overwrites it.
/// Serialization is the caller's concern (the CLI writes the JSON file);
/// `kengram-storage` deliberately depends only on `serde_json`, not `serde`.
#[derive(Debug, Clone)]
pub struct TagSnapshotRow {
    pub thought_id: ThoughtId,
    pub tags: serde_json::Value,
    pub tags_extractor_model: Option<String>,
    pub tags_extractor_version: Option<i32>,
}

/// Snapshot the current tags + provenance for every non-retracted thought.
/// Retag passes overwrite `tags` in place (there is no tag-history table), so
/// the operator captures this before a `--rerun`/`--force` retag to keep a
/// recoverable copy. Retracted thoughts are excluded: retag passes skip them,
/// so their tags never change. Ordered by `created_at` for a stable dump.
pub async fn snapshot_nonretracted_tags(
    pool: &PgPool,
) -> Result<Vec<TagSnapshotRow>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, tags, tags_extractor_model, tags_extractor_version
        FROM thoughts
        WHERE retracted_at IS NULL
        ORDER BY created_at
        "#,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| TagSnapshotRow {
            thought_id: ThoughtId(r.id),
            tags: r.tags,
            tags_extractor_model: r.tags_extractor_model,
            tags_extractor_version: r.tags_extractor_version,
        })
        .collect())
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
/// when `rerun = true`). When `force = true`, every matching thought is walked
/// regardless of version (the scope / scope_prefix / since / limit filters still
/// apply). Oldest first. Used by `kengram tag [--rerun] [--force]`.
// Selection filters are independent knobs the CLI passes straight through;
// bundling them into a struct would just relocate the same arguments.
#[allow(clippy::too_many_arguments)]
pub async fn find_untagged_or_stale_thoughts(
    pool: &PgPool,
    target_tagger_version: i32,
    rerun: bool,
    force: bool,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
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
          AND ($2::text IS NULL OR scope LIKE $2 || '%')
          AND ($3::timestamptz IS NULL OR created_at >= $3)
          AND (
              $7::boolean
              OR tags_extractor_version IS NULL
              OR ($4::boolean AND tags_extractor_version < $5)
          )
        ORDER BY created_at ASC
        LIMIT $6
        "#,
        scope,
        scope_prefix,
        since,
        rerun,
        target_tagger_version,
        limit,
        force,
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

// -- M5: selective relations (thought-to-* edges) -----------------------
//
// M5 shipped thought-to-thought only. M5.2 generalized targets to
// (thought | entity | person | url) via the polymorphic columns added in
// migration 0009 and added soft-delete via `deleted_at` (migration 0010).

/// One related target returned by `fetch_related_thoughts`. Carries the
/// edge metadata plus, when the target is a thought, enough enrichment
/// from the joined `thoughts` row that callers can render results without
/// a follow-up `get_thought`.
///
/// `direction` is `Outbound` when the queried thought sits on the edge's
/// `from` side (so `target` is the `to` side here) and `Inbound`
/// otherwise. Inbound edges are always thought→thought by schema (the
/// `from` side of any link is always a thought), so `target` for inbound
/// rows is always `LinkTarget::Thought(_)` and the `thought_*` fields are
/// always populated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedTarget {
    pub link_id: LinkId,
    pub relation: RelationKind,
    pub direction: LinkDirection,
    pub target: LinkTarget,
    /// Populated only when `target = LinkTarget::Thought(_)`. None for
    /// entity/person/URL targets on outbound edges.
    pub thought_scope: Option<Scope>,
    pub thought_content: Option<String>,
    pub thought_created_at: Option<OffsetDateTime>,
    pub thought_retracted: Option<bool>,
    pub link_created_at: OffsetDateTime,
    pub link_source: LinkSource,
    pub note: Option<String>,
}

/// One thought-target graph neighbor safe to feed into search-time graph
/// augmentation. Unlike `fetch_related_thoughts`, this shape is already
/// filtered for search semantics: live links only, thought targets only,
/// non-retracted neighbor thoughts, scope/scope_prefix respected, and eval
/// contamination excluded at source.
#[derive(Debug, Clone)]
pub struct GraphNeighborHit {
    pub seed_thought_id: ThoughtId,
    pub link_id: LinkId,
    pub relation: RelationKind,
    pub direction: LinkDirection,
    pub link_source: LinkSource,
    pub note: Option<String>,
    pub thought: Thought,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphNonThoughtTargetCounts {
    pub entity: i64,
    pub person: i64,
    pub url: i64,
}

/// Three-way live/soft-deleted/never-existed status of an edge identified
/// by `(from, relation, target)`. Used by the MCP `unlink_thoughts`
/// orchestrator to distinguish "we just removed this edge" from "this
/// edge was previously removed" from "this edge never existed."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkStatus {
    Live,
    SoftDeleted,
    NeverExisted,
}

/// Insert a link with a polymorphic target. Idempotent on the
/// `(from, relation, to_kind, to_value)` quadruple via the partial unique
/// index `thought_links_unique_edge` (which only covers rows with
/// `deleted_at IS NULL`): re-asserting a live edge returns the existing
/// row's `LinkId` with `is_new = false`. If the edge previously existed
/// but was soft-deleted, this inserts a fresh live row (the partial unique
/// index ignores the soft-deleted predecessor).
///
/// Foreign-key violations (thought target missing in `thoughts`) and
/// CHECK violations (e.g., `to_url` not matching `^https?://`) are surfaced
/// as `StorageError::Database`. The MCP layer should pre-validate where
/// it can so the operator-facing error is actionable; this layer is the
/// last line of defense.
pub async fn insert_link(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    target: &LinkTarget,
    source: LinkSource,
    note: Option<&str>,
) -> Result<(LinkId, bool), StorageError> {
    let to_thought_id = target.as_thought_id().map(|id| *id.as_uuid());
    let to_entity = target.as_entity();
    let to_person = target.as_person();
    let to_url = target.as_url();

    let row = sqlx::query!(
        r#"
        INSERT INTO thought_links
            (from_thought_id, relation, to_kind, to_thought_id, to_entity, to_person, to_url, source, note)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (from_thought_id, relation, to_kind, to_value) WHERE deleted_at IS NULL
        DO NOTHING
        RETURNING id
        "#,
        from.into_uuid(),
        relation.as_str(),
        target.kind_str(),
        to_thought_id,
        to_entity,
        to_person,
        to_url,
        source.as_str(),
        note,
    )
    .fetch_optional(pool)
    .await?;

    if let Some(r) = row {
        return Ok((LinkId::from(r.id), true));
    }

    // ON CONFLICT path: fetch the existing live row's id.
    let value = target.value_str();
    let existing = sqlx::query!(
        r#"
        SELECT id
        FROM thought_links
        WHERE from_thought_id = $1
          AND relation = $2
          AND to_kind = $3
          AND to_value = $4
          AND deleted_at IS NULL
        "#,
        from.into_uuid(),
        relation.as_str(),
        target.kind_str(),
        value,
    )
    .fetch_one(pool)
    .await?;

    Ok((LinkId::from(existing.id), false))
}

/// Determine the live/soft-deleted/never-existed status of an edge
/// identified by `(from, relation, target)`. The MCP layer uses this to
/// drive the three-way `UnlinkStatus` discriminator returned from
/// `unlink_thoughts`.
///
/// "Live" if any row matches with `deleted_at IS NULL`. "SoftDeleted" if
/// no live row matches but at least one soft-deleted row exists.
/// "NeverExisted" otherwise.
pub async fn lookup_link_status(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    target: &LinkTarget,
) -> Result<LinkStatus, StorageError> {
    let value = target.value_str();
    let counts = sqlx::query!(
        r#"
        SELECT
            COUNT(*) FILTER (WHERE deleted_at IS NULL)     AS "live_count!",
            COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) AS "deleted_count!"
        FROM thought_links
        WHERE from_thought_id = $1
          AND relation = $2
          AND to_kind = $3
          AND to_value = $4
        "#,
        from.into_uuid(),
        relation.as_str(),
        target.kind_str(),
        value,
    )
    .fetch_one(pool)
    .await?;

    if counts.live_count > 0 {
        Ok(LinkStatus::Live)
    } else if counts.deleted_count > 0 {
        Ok(LinkStatus::SoftDeleted)
    } else {
        Ok(LinkStatus::NeverExisted)
    }
}

/// Soft-delete the live edge identified by `(from, relation, target)`.
/// Returns `Some(link_id)` if a live row was just soft-deleted; `None`
/// otherwise (the edge was already soft-deleted or never existed —
/// callers should pair with `lookup_link_status` to disambiguate).
pub async fn delete_link(
    pool: &PgPool,
    from: ThoughtId,
    relation: RelationKind,
    target: &LinkTarget,
) -> Result<Option<LinkId>, StorageError> {
    let value = target.value_str();
    let row = sqlx::query!(
        r#"
        UPDATE thought_links
        SET deleted_at = NOW()
        WHERE from_thought_id = $1
          AND relation = $2
          AND to_kind = $3
          AND to_value = $4
          AND deleted_at IS NULL
        RETURNING id
        "#,
        from.into_uuid(),
        relation.as_str(),
        target.kind_str(),
        value,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| LinkId::from(r.id)))
}

/// Soft-delete all live (`deleted_at IS NULL`) edges where this thought is
/// the `from` side AND `source = 'tagger'`. Used by the tag drainer (M6.1)
/// to invalidate prior tagger-emitted edges before inserting fresh ones on
/// re-tag. Returns the count of soft-deleted rows for observability.
///
/// Agent-supplied edges (`source = 'agent'`) are unaffected — the operator
/// has explicit authority over those, and a tagger-prompt iteration must
/// not silently erase them.
pub async fn soft_delete_tagger_edges_for_thought(
    pool: &PgPool,
    from_thought_id: ThoughtId,
) -> Result<i64, StorageError> {
    let result = sqlx::query!(
        r#"
        UPDATE thought_links
        SET deleted_at = NOW()
        WHERE from_thought_id = $1
          AND source = 'tagger'
          AND deleted_at IS NULL
        "#,
        from_thought_id.into_uuid(),
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() as i64)
}

/// Walk edges from a given thought. `direction` selects whether to
/// traverse outbound (where `thought_id` is `from`), inbound (where
/// `thought_id` is `to`), or both. `relations`, when supplied, restricts
/// to that subset of the closed vocabulary. `target_kinds`, when supplied,
/// restricts outbound edges to those target kinds (no effect on inbound,
/// which is always thought→thought by schema).
///
/// Soft-deleted edges are excluded (`deleted_at IS NULL`).
///
/// The returned `RelatedTarget` rows carry the *other* end of each edge
/// (so callers can render them directly) along with the edge's metadata.
/// For thought targets, the joined `thoughts` row's scope/content/etc. is
/// surfaced via the `thought_*` fields. Retracted thoughts on the far end
/// aren't filtered out — the `thought_retracted` flag is surfaced so
/// consumers can decide whether to show/dim/hide.
pub async fn fetch_related_thoughts(
    pool: &PgPool,
    thought_id: ThoughtId,
    relations: Option<&[RelationKind]>,
    target_kinds: Option<&[&str]>,
    direction: LinkDirection,
) -> Result<Vec<RelatedTarget>, StorageError> {
    // Empty-as-no-filter sentinel via cardinality(...) = 0 OR ANY(...) —
    // same trick the M5 implementation used for `relations`, generalized
    // to a second filter for `target_kinds`.
    let relation_filter: Vec<String> = relations
        .map(|rs| rs.iter().map(|r| r.as_str().to_string()).collect())
        .unwrap_or_default();
    let kind_filter: Vec<String> = target_kinds
        .map(|ks| ks.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let inbound_includes_thought_kind =
        kind_filter.is_empty() || kind_filter.iter().any(|s| s == "thought");

    let mut rows = Vec::new();

    if matches!(direction, LinkDirection::Outbound | LinkDirection::Both) {
        let out = sqlx::query!(
            r#"
            SELECT
                tl.id AS link_id,
                tl.relation,
                tl.to_kind,
                tl.to_thought_id,
                tl.to_entity,
                tl.to_person,
                tl.to_url,
                tl.created_at AS link_created_at,
                tl.source AS link_source,
                tl.note,
                t.scope                 AS "t_scope?",
                t.content               AS "t_content?",
                t.created_at            AS "t_created_at?",
                CASE WHEN t.id IS NOT NULL THEN (t.retracted_at IS NOT NULL) END
                    AS "t_retracted?"
            FROM thought_links tl
            LEFT JOIN thoughts t ON t.id = tl.to_thought_id
            WHERE tl.from_thought_id = $1
              AND tl.deleted_at IS NULL
              AND (cardinality($2::text[]) = 0 OR tl.relation = ANY($2::text[]))
              AND (cardinality($3::text[]) = 0 OR tl.to_kind = ANY($3::text[]))
            ORDER BY tl.created_at DESC
            "#,
            thought_id.into_uuid(),
            &relation_filter,
            &kind_filter,
        )
        .fetch_all(pool)
        .await?;

        for r in out {
            let target = link_target_from_row(
                &r.to_kind,
                r.to_thought_id,
                r.to_entity.as_deref(),
                r.to_person.as_deref(),
                r.to_url.as_deref(),
            )?;
            let thought_scope = r.t_scope.map(Scope::new).transpose()?;
            rows.push(RelatedTarget {
                link_id: LinkId::from(r.link_id),
                relation: r.relation.parse()?,
                direction: LinkDirection::Outbound,
                target,
                thought_scope,
                thought_content: r.t_content,
                thought_created_at: r.t_created_at,
                thought_retracted: r.t_retracted,
                link_created_at: r.link_created_at,
                link_source: r.link_source.parse()?,
                note: r.note,
            });
        }
    }

    if matches!(direction, LinkDirection::Inbound | LinkDirection::Both)
        && inbound_includes_thought_kind
    {
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
              AND tl.deleted_at IS NULL
              AND (cardinality($2::text[]) = 0 OR tl.relation = ANY($2::text[]))
            ORDER BY tl.created_at DESC
            "#,
            thought_id.into_uuid(),
            &relation_filter,
        )
        .fetch_all(pool)
        .await?;

        for r in inb {
            rows.push(RelatedTarget {
                link_id: LinkId::from(r.link_id),
                relation: r.relation.parse()?,
                direction: LinkDirection::Inbound,
                target: LinkTarget::Thought(ThoughtId::from(r.thought_id)),
                thought_scope: Some(Scope::new(r.scope)?),
                thought_content: Some(r.content),
                thought_created_at: Some(r.thought_created_at),
                thought_retracted: Some(r.retracted),
                link_created_at: r.link_created_at,
                link_source: r.link_source.parse()?,
                note: r.note,
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

/// Fetch bounded graph neighbors for search-time augmentation. This is
/// stricter than `fetch_related_thoughts`: only thought targets are returned
/// as candidates, soft-deleted links are excluded, retracted neighbor thoughts
/// are excluded, explicit scope filters are enforced, and eval/adjudication
/// contamination is filtered before the rows leave storage.
pub async fn search_graph_neighbors(
    pool: &PgPool,
    seed_ids: &[ThoughtId],
    relations: &[RelationKind],
    direction: LinkDirection,
    per_seed_cap: usize,
    total_cap: usize,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
) -> Result<Vec<GraphNeighborHit>, StorageError> {
    if seed_ids.is_empty() || per_seed_cap == 0 || total_cap == 0 {
        return Ok(Vec::new());
    }

    let seed_uuids: Vec<Uuid> = seed_ids.iter().map(|id| id.into_uuid()).collect();
    let seed_ranks: Vec<i32> = (0..seed_uuids.len()).map(|idx| idx as i32).collect();
    let relation_filter: Vec<String> = relations
        .iter()
        .map(|relation| relation.as_str().to_string())
        .collect();
    let include_outbound = matches!(direction, LinkDirection::Outbound | LinkDirection::Both);
    let include_inbound = matches!(direction, LinkDirection::Inbound | LinkDirection::Both);
    if !include_outbound && !include_inbound {
        return Ok(Vec::new());
    }

    let rows = sqlx::query!(
        r#"
        WITH seeds(seed_id, seed_rank) AS (
            SELECT * FROM unnest($1::uuid[], $2::int[])
        ),
        candidate_edges AS (
            SELECT
                s.seed_id,
                s.seed_rank,
                tl.id AS link_id,
                tl.relation,
                'outbound'::text AS direction,
                tl.source AS link_source,
                tl.note,
                tl.created_at AS link_created_at,
                tl.to_thought_id AS neighbor_id
            FROM seeds s
            JOIN thought_links tl ON tl.from_thought_id = s.seed_id
            WHERE $4::bool
              AND tl.deleted_at IS NULL
              AND tl.to_kind = 'thought'
              AND (cardinality($3::text[]) = 0 OR tl.relation = ANY($3::text[]))

            UNION ALL

            SELECT
                s.seed_id,
                s.seed_rank,
                tl.id AS link_id,
                tl.relation,
                'inbound'::text AS direction,
                tl.source AS link_source,
                tl.note,
                tl.created_at AS link_created_at,
                tl.from_thought_id AS neighbor_id
            FROM seeds s
            JOIN thought_links tl ON tl.to_thought_id = s.seed_id
            WHERE $5::bool
              AND tl.deleted_at IS NULL
              AND (cardinality($3::text[]) = 0 OR tl.relation = ANY($3::text[]))
        ),
        filtered_edges AS (
            SELECT
                e.seed_id,
                e.seed_rank,
                e.link_id,
                e.relation,
                e.direction,
                e.link_source,
                e.note,
                e.link_created_at,
                t.id,
                t.scope,
                t.content,
                t.source,
                t.created_at,
                t.metadata,
                t.content_fingerprint,
                t.tags,
                t.tags_extractor_model,
                t.tags_extractor_version,
                t.tags_extracted_at
            FROM candidate_edges e
            JOIN thoughts t ON t.id = e.neighbor_id
            WHERE ($6::text IS NULL OR t.scope = $6)
              AND ($7::text IS NULL OR t.scope LIKE $7 || '%')
              AND t.retracted_at IS NULL
              AND t.id <> ALL($10::uuid[])
              AND lower(coalesce(t.metadata->>'source_file', '')) !~ $11
              AND t.content !~ $12
        ),
        ranked_edges AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY seed_id
                    ORDER BY link_created_at DESC, link_id ASC
                ) AS per_seed_rank
            FROM filtered_edges
        )
        SELECT
            seed_id AS "seed_id!",
            link_id AS "link_id!",
            relation AS "relation!",
            direction AS "direction!",
            link_source AS "link_source!",
            note,
            id,
            scope,
            content,
            source,
            created_at,
            metadata,
            content_fingerprint,
            tags,
            tags_extractor_model,
            tags_extractor_version,
            tags_extracted_at
        FROM ranked_edges
        WHERE per_seed_rank <= $8
        ORDER BY seed_rank ASC, per_seed_rank ASC, link_created_at DESC, link_id ASC
        LIMIT $9
        "#,
        &seed_uuids,
        &seed_ranks,
        &relation_filter,
        include_outbound,
        include_inbound,
        scope,
        scope_prefix,
        per_seed_cap as i64,
        total_cap as i64,
        EVAL_CONTAMINATION_KNOWN_IDS,
        EVAL_CONTAMINATION_SOURCE_FILE_REGEX,
        EVAL_CONTAMINATION_CONTENT_REGEX,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|r| {
            let direction = match r.direction.as_str() {
                "outbound" => LinkDirection::Outbound,
                _ => LinkDirection::Inbound,
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
            Ok(GraphNeighborHit {
                seed_thought_id: ThoughtId::from(r.seed_id),
                link_id: LinkId::from(r.link_id),
                relation: r.relation.parse()?,
                direction,
                link_source: r.link_source.parse()?,
                note: r.note,
                thought,
            })
        })
        .collect()
}

/// Count outbound non-thought targets adjacent to the graph seeds. Search
/// graph augmentation never follows these into invented hits; the counts are
/// only profile/provenance evidence for Smith and operators.
pub async fn count_graph_non_thought_targets(
    pool: &PgPool,
    seed_ids: &[ThoughtId],
    relations: &[RelationKind],
    direction: LinkDirection,
) -> Result<GraphNonThoughtTargetCounts, StorageError> {
    if seed_ids.is_empty() || matches!(direction, LinkDirection::Inbound) {
        return Ok(GraphNonThoughtTargetCounts::default());
    }

    let seed_uuids: Vec<Uuid> = seed_ids.iter().map(|id| id.into_uuid()).collect();
    let relation_filter: Vec<String> = relations
        .iter()
        .map(|relation| relation.as_str().to_string())
        .collect();
    let rows = sqlx::query!(
        r#"
        SELECT tl.to_kind, COUNT(*) AS "count!"
        FROM thought_links tl
        WHERE tl.from_thought_id = ANY($1::uuid[])
          AND tl.deleted_at IS NULL
          AND tl.to_kind <> 'thought'
          AND (cardinality($2::text[]) = 0 OR tl.relation = ANY($2::text[]))
        GROUP BY tl.to_kind
        "#,
        &seed_uuids,
        &relation_filter,
    )
    .fetch_all(pool)
    .await?;

    let mut counts = GraphNonThoughtTargetCounts::default();
    for row in rows {
        match row.to_kind.as_str() {
            "entity" => counts.entity = row.count,
            "person" => counts.person = row.count,
            "url" => counts.url = row.count,
            _ => {}
        }
    }
    Ok(counts)
}

fn link_target_from_row(
    to_kind: &str,
    to_thought_id: Option<Uuid>,
    to_entity: Option<&str>,
    to_person: Option<&str>,
    to_url: Option<&str>,
) -> Result<LinkTarget, StorageError> {
    match to_kind {
        "thought" => to_thought_id
            .map(|id| LinkTarget::Thought(ThoughtId::from(id)))
            .ok_or_else(|| StorageError::InvalidLinkTargetShape(to_kind.to_string())),
        "entity" => to_entity
            .map(|s| LinkTarget::Entity(s.to_string()))
            .ok_or_else(|| StorageError::InvalidLinkTargetShape(to_kind.to_string())),
        "person" => to_person
            .map(|s| LinkTarget::Person(s.to_string()))
            .ok_or_else(|| StorageError::InvalidLinkTargetShape(to_kind.to_string())),
        "url" => to_url
            .map(|s| LinkTarget::Url(s.to_string()))
            .ok_or_else(|| StorageError::InvalidLinkTargetShape(to_kind.to_string())),
        other => Err(StorageError::InvalidLinkTargetShape(other.to_string())),
    }
}

/// One row from the `migration_audit` table (created in migration 0010).
/// Surfaced to operators by `kengram audit migrations`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationAuditRow {
    pub id: Uuid,
    pub migration: String,
    pub ran_at: OffsetDateTime,
    pub rows_touched: i64,
    pub notes: Option<String>,
}

/// Read the migration_audit log, most-recent first. `since`, when set,
/// filters to entries with `ran_at >= since`.
pub async fn query_migration_audit(
    pool: &PgPool,
    since: Option<OffsetDateTime>,
    limit: i64,
) -> Result<Vec<MigrationAuditRow>, StorageError> {
    let rows = sqlx::query!(
        r#"
        SELECT id, migration, ran_at, rows_touched, notes
        FROM migration_audit
        WHERE ($1::timestamptz IS NULL OR ran_at >= $1)
        ORDER BY ran_at DESC
        LIMIT $2
        "#,
        since,
        limit,
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| MigrationAuditRow {
            id: r.id,
            migration: r.migration,
            ran_at: r.ran_at,
            rows_touched: r.rows_touched,
            notes: r.notes,
        })
        .collect())
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
/// retrieval (`recent_thoughts`, `search_fts`, `search_vector_knn`);
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

/// Vector-similarity kNN for the given model. Large raw vectors use the ANN
/// projection sidecar when configured; smaller/non-projected models fall back
/// to exact search over `embeddings.vector`.
pub async fn search_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    model: &EmbeddingModel,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    if is_bge_m3_1024(model) {
        return search_bge_vector_knn(pool, query_vector, scope, scope_prefix, limit).await;
    }

    if let Some(projection) = ann_projection_for_model(model) {
        if ann_projection_search_ready(pool, &projection.projection_id).await? {
            if ann_projection_filter_has_missing_rows(
                pool,
                &projection.projection_id,
                &model.id,
                scope,
                scope_prefix,
            )
            .await?
            {
                tracing::error!(
                    model_id = %model.id,
                    projection_id = %projection.projection_id,
                    scope = ?scope,
                    scope_prefix = ?scope_prefix,
                    "ANN projection coverage gap overlaps requested filter; falling back to exact raw vector search"
                );
            } else {
                let halfvec = project_halfvec_3072(&query_vector, projection.dimensions)?;
                let mut tx = pool.begin().await?;
                set_ann_projection_ef_search(&mut tx).await?;
                let rows: Vec<VectorSearchRow> = sqlx::query_as(
                    r#"
                SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
                       t.content_fingerprint, t.tags,
                       t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
                       (p.embedding <=> $1) AS distance
                FROM embedding_ann_projections p
                JOIN thoughts t ON t.id = p.target_id
                WHERE p.projection_id = $2
                  AND p.model_id = $3
                  AND p.target_kind = 'thought'
                  AND ($4::text IS NULL OR t.scope = $4)
                  AND ($5::text IS NULL OR t.scope LIKE $5 || '%')
                  AND t.retracted_at IS NULL
                  AND t.id <> ALL($7::uuid[])
                  AND lower(coalesce(t.metadata->>'source_file', '')) !~ $8
                  AND t.content !~ $9
                ORDER BY p.embedding <=> $1
                LIMIT $6
                "#,
                )
                .bind(halfvec)
                .bind(&projection.projection_id)
                .bind(&model.id)
                .bind(scope)
                .bind(scope_prefix)
                .bind(limit)
                .bind(EVAL_CONTAMINATION_KNOWN_IDS)
                .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
                .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
                .fetch_all(&mut *tx)
                .await?;
                tx.commit().await?;
                return vector_rows_to_hits(rows);
            }
        } else {
            tracing::warn!(
                model_id = %model.id,
                projection_id = %projection.projection_id,
                "ANN projection coverage is not marked complete; falling back to exact raw vector search"
            );
        }
    }

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
          AND t.id <> ALL($6::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $7
          AND t.content !~ $8
        ORDER BY e.vector <=> $1
        LIMIT $5
        "#,
    )
    .bind(pgv)
    .bind(&model.id)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(pool)
    .await?;

    vector_rows_to_hits(rows)
}

async fn search_bge_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    if query_vector.len() != bge::DIMS {
        return Err(StorageError::InvalidEmbeddingDimensions {
            model_id: bge::MODEL_ID.to_string(),
            expected: bge::DIMS,
            got: query_vector.len(),
        });
    }

    assert_bge_vector_search_ready(pool).await?;

    let pgv = pgvector::Vector::from(query_vector);
    let mut tx = pool.begin().await?;
    set_bge_hnsw_ef_search(&mut tx).await?;
    let rows: Vec<VectorSearchRow> = sqlx::query_as(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata,
               t.content_fingerprint, t.tags,
               t.tags_extractor_model, t.tags_extractor_version, t.tags_extracted_at,
               (b.embedding <=> $1::vector(1024)) AS distance
        FROM thought_embeddings_bge_m3 b
        JOIN thoughts t ON t.id = b.thought_id
        WHERE b.model_id = $2
          AND b.model_version = $3
          AND ($4::text IS NULL OR t.scope = $4)
          AND ($5::text IS NULL OR t.scope LIKE $5 || '%')
          AND t.retracted_at IS NULL
          AND t.id <> ALL($7::uuid[])
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ $8
          AND t.content !~ $9
        ORDER BY b.embedding <=> $1::vector(1024)
        LIMIT $6
        "#,
    )
    .bind(pgv)
    .bind(bge::MODEL_ID)
    .bind(bge::MODEL_VERSION)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;

    vector_rows_to_hits(rows)
}

/// Vector kNN over BGE-M3 artifact chunk embeddings. Each hit resolves to its
/// source parent thought and carries the best matching chunk as provenance.
pub async fn search_artifact_chunks_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    model: &EmbeddingModel,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    if !is_bge_m3_1024(model) {
        return Ok(Vec::new());
    }
    if query_vector.len() != bge::DIMS {
        return Err(StorageError::InvalidEmbeddingDimensions {
            model_id: model.id.clone(),
            expected: bge::DIMS,
            got: query_vector.len(),
        });
    }

    assert_bge_chunk_vector_search_ready(pool).await?;

    let pgv = pgvector::Vector::from(query_vector);
    let mut tx = pool.begin().await?;
    set_bge_hnsw_ef_search(&mut tx).await?;
    let rows: Vec<ChunkVectorSearchRow> = sqlx::query_as(
        r#"
        WITH candidates AS (
            SELECT t.id,
                   t.scope,
                   t.content AS parent_content,
                   t.source,
                   t.created_at,
                   t.metadata AS parent_metadata,
                   t.content_fingerprint,
                   t.tags,
                   t.tags_extractor_model,
                   t.tags_extractor_version,
                   t.tags_extracted_at,
                   ac.id AS chunk_id,
                   ac.artifact_id,
                   ac.source_thought_id,
                   ac.chunk_index,
                   ac.content AS chunk_content,
                   ac.chunker_id,
                   ac.chunker_version,
                   ac.token_estimate,
                   ac.start_char,
                   ac.end_char,
                   ac.metadata AS chunk_metadata,
                   (b.embedding <=> $1::vector(1024)) AS distance
            FROM artifact_chunk_embeddings_bge_m3 b
            JOIN artifact_chunks ac ON ac.id = b.chunk_id
            JOIN thoughts t ON t.id = ac.source_thought_id
            WHERE b.model_id = $2
              AND b.model_version = $3
              AND ac.retracted_at IS NULL
              AND ac.source_thought_id IS NOT NULL
              AND t.retracted_at IS NULL
              AND ($4::text IS NULL OR t.scope = $4)
              AND ($5::text IS NULL OR t.scope LIKE $5 || '%')
              AND t.id <> ALL($7::uuid[])
              AND lower(coalesce(t.metadata->>'source_file', '')) !~ $8
              AND t.content !~ $9
              AND ac.content !~ $9
            ORDER BY b.embedding <=> $1::vector(1024)
            LIMIT GREATEST($6, $6 * 8)
        ),
        best_per_parent AS (
            SELECT DISTINCT ON (id) *
            FROM candidates
            ORDER BY id, distance ASC, chunk_index ASC
        )
        SELECT *
        FROM best_per_parent
        ORDER BY distance ASC, created_at DESC, chunk_index ASC
        LIMIT $6
        "#,
    )
    .bind(pgv)
    .bind(bge::MODEL_ID)
    .bind(bge::MODEL_VERSION)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;

    chunk_vector_rows_to_hits(rows)
}

/// Vector kNN over generated contextual chunk documents. This reads only the
/// contextual sidecar embedding table; raw chunk embeddings remain untouched.
pub async fn search_artifact_chunk_contexts_vector_knn(
    pool: &PgPool,
    query_vector: Vec<f32>,
    model: &EmbeddingModel,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    limit: i64,
) -> Result<Vec<Hit>, StorageError> {
    if !is_bge_m3_1024(model) {
        return Ok(Vec::new());
    }
    if query_vector.len() != bge::DIMS {
        return Err(StorageError::InvalidEmbeddingDimensions {
            model_id: model.id.clone(),
            expected: bge::DIMS,
            got: query_vector.len(),
        });
    }

    assert_bge_context_vector_search_ready(pool).await?;

    let pgv = pgvector::Vector::from(query_vector);
    let mut tx = pool.begin().await?;
    set_bge_hnsw_ef_search(&mut tx).await?;
    let rows: Vec<ChunkVectorSearchRow> = sqlx::query_as(
        r#"
        WITH candidates AS (
            SELECT t.id,
                   t.scope,
                   t.content AS parent_content,
                   t.source,
                   t.created_at,
                   t.metadata AS parent_metadata,
                   t.content_fingerprint,
                   t.tags,
                   t.tags_extractor_model,
                   t.tags_extractor_version,
                   t.tags_extracted_at,
                   ac.id AS chunk_id,
                   ac.artifact_id,
                   ac.source_thought_id,
                   ac.chunk_index,
                   ac.content AS chunk_content,
                   ac.chunker_id,
                   ac.chunker_version,
                   ac.token_estimate,
                   ac.start_char,
                   ac.end_char,
                   jsonb_set(
                       ac.metadata,
                       '{contextual_retrieval}',
                       jsonb_build_object(
                           'context_id', c.id::text,
                           'generator_id', c.generator_id,
                           'generator_version', c.generator_version,
                           'prompt_version', c.prompt_version,
                           'contextual', true
                       ),
                       true
                   ) AS chunk_metadata,
                   (b.embedding <=> $1::vector(1024)) AS distance
            FROM artifact_chunk_context_embeddings_bge_m3 b
            JOIN artifact_chunk_contexts c ON c.id = b.context_id
            JOIN artifact_chunks ac ON ac.id = c.chunk_id
            JOIN thoughts t ON t.id = c.source_thought_id
            WHERE b.model_id = $2
              AND b.model_version = $3
              AND c.status = 'ready'
              AND c.retracted_at IS NULL
              AND ac.retracted_at IS NULL
              AND ac.source_thought_id IS NOT NULL
              AND t.retracted_at IS NULL
              AND ($4::text IS NULL OR t.scope = $4)
              AND ($5::text IS NULL OR t.scope LIKE $5 || '%')
              AND t.id <> ALL($7::uuid[])
              AND lower(coalesce(t.metadata->>'source_file', '')) !~ $8
              AND t.content !~ $9
              AND ac.content !~ $9
              AND c.context_text !~ $9
              AND c.contextual_content !~ $9
            ORDER BY b.embedding <=> $1::vector(1024)
            LIMIT GREATEST($6, $6 * 8)
        ),
        best_per_parent AS (
            SELECT DISTINCT ON (id) *
            FROM candidates
            ORDER BY id, distance ASC, chunk_index ASC
        )
        SELECT *
        FROM best_per_parent
        ORDER BY distance ASC, created_at DESC, chunk_index ASC
        LIMIT $6
        "#,
    )
    .bind(pgv)
    .bind(bge::MODEL_ID)
    .bind(bge::MODEL_VERSION)
    .bind(scope)
    .bind(scope_prefix)
    .bind(limit)
    .bind(EVAL_CONTAMINATION_KNOWN_IDS)
    .bind(EVAL_CONTAMINATION_SOURCE_FILE_REGEX)
    .bind(EVAL_CONTAMINATION_CONTENT_REGEX)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;

    chunk_vector_rows_to_hits(rows)
}

async fn ann_projection_search_ready(
    pool: &PgPool,
    projection_id: &str,
) -> Result<bool, StorageError> {
    let (ready,): (bool,) = sqlx::query_as(
        r#"
        SELECT COALESCE((
            SELECT true
            FROM embedding_ann_projection_coverage
            WHERE projection_id = $1
              AND status = 'ok'
              AND missing_count = 0
            LIMIT 1
        ), false)
        "#,
    )
    .bind(projection_id)
    .fetch_one(pool)
    .await?;

    Ok(ready)
}

async fn ann_projection_filter_has_missing_rows(
    pool: &PgPool,
    projection_id: &str,
    model_id: &str,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
) -> Result<bool, StorageError> {
    let (exists,): (bool,) = sqlx::query_as(
        r#"
        SELECT EXISTS (
            SELECT 1
            FROM embeddings e
            JOIN thoughts t
              ON t.id = e.target_id
             AND t.retracted_at IS NULL
            WHERE e.model_id = $1
              AND e.target_kind = 'thought'
              AND vector_dims(e.vector) >= 3072
              AND ($3::text IS NULL OR t.scope = $3)
              AND ($4::text IS NULL OR t.scope LIKE $4 || '%')
              AND NOT EXISTS (
                  SELECT 1
                  FROM embedding_ann_projections p
                  WHERE p.source_embedding_id = e.id
                    AND p.projection_id = $2
              )
            LIMIT 1
        )
        "#,
    )
    .bind(model_id)
    .bind(projection_id)
    .bind(scope)
    .bind(scope_prefix)
    .fetch_one(pool)
    .await?;

    Ok(exists)
}

fn vector_rows_to_hits(rows: Vec<VectorSearchRow>) -> Result<Vec<Hit>, StorageError> {
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

fn chunk_vector_rows_to_hits(rows: Vec<ChunkVectorSearchRow>) -> Result<Vec<Hit>, StorageError> {
    rows.into_iter()
        .map(|r| {
            let score = (1.0 - r.distance) as f32;
            let (thought, chunk) = chunk_row_parts_to_hit_parts(
                r.id,
                r.scope,
                r.parent_content,
                r.source,
                r.created_at,
                r.parent_metadata,
                r.content_fingerprint,
                r.tags,
                r.tags_extractor_model,
                r.tags_extractor_version,
                r.tags_extracted_at,
                r.chunk_id,
                r.artifact_id,
                r.source_thought_id,
                r.chunk_index,
                r.chunk_content,
                r.chunker_id,
                r.chunker_version,
                r.token_estimate,
                r.start_char,
                r.end_char,
                r.chunk_metadata,
            )?;
            Ok(Hit::from_vector_leg(thought, score).with_chunk_provenance(chunk))
        })
        .collect()
}

fn chunk_lexical_rows_to_hits(rows: Vec<ChunkLexicalSearchRow>) -> Result<Vec<Hit>, StorageError> {
    rows.into_iter()
        .map(|r| {
            let (thought, chunk) = chunk_row_parts_to_hit_parts(
                r.id,
                r.scope,
                r.parent_content,
                r.source,
                r.created_at,
                r.parent_metadata,
                r.content_fingerprint,
                r.tags,
                r.tags_extractor_model,
                r.tags_extractor_version,
                r.tags_extracted_at,
                r.chunk_id,
                r.artifact_id,
                r.source_thought_id,
                r.chunk_index,
                r.chunk_content,
                r.chunker_id,
                r.chunker_version,
                r.token_estimate,
                r.start_char,
                r.end_char,
                r.chunk_metadata,
            )?;
            Ok(Hit::from_lexical_leg(thought, r.rank).with_chunk_provenance(chunk))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn chunk_row_parts_to_hit_parts(
    id: Uuid,
    scope: String,
    parent_content: String,
    source: String,
    created_at: OffsetDateTime,
    parent_metadata: serde_json::Value,
    content_fingerprint: Vec<u8>,
    tags: serde_json::Value,
    tags_extractor_model: Option<String>,
    tags_extractor_version: Option<i32>,
    tags_extracted_at: Option<OffsetDateTime>,
    chunk_id: Uuid,
    artifact_id: Uuid,
    source_thought_id: Uuid,
    chunk_index: i32,
    chunk_content: String,
    chunker_id: String,
    chunker_version: i32,
    token_estimate: Option<i32>,
    start_char: Option<i32>,
    end_char: Option<i32>,
    chunk_metadata: serde_json::Value,
) -> Result<(Thought, ChunkProvenance), StorageError> {
    let thought = Thought {
        id: ThoughtId::from(id),
        scope: Scope::new(scope)?,
        content: parent_content,
        source: Source::new(source)?,
        created_at,
        metadata: Metadata::from(parent_metadata),
        content_fingerprint: fingerprint_from_bytes(content_fingerprint)?,
        tags: tags_from_value(tags)?,
        tags_extractor_model,
        tags_extractor_version,
        tags_extracted_at,
    };
    let chunk = ChunkProvenance {
        chunk_id,
        artifact_id,
        source_thought_id: ThoughtId::from(source_thought_id),
        chunk_index,
        content: chunk_content,
        chunker_id,
        chunker_version,
        token_estimate,
        start_char,
        end_char,
        metadata: chunk_metadata,
    };
    Ok((thought, chunk))
}

/// Runtime guard against the exact regression that produced the 4096-dim
/// seqscan path: when a model has a configured ANN projection, startup ensures
/// a matching per-projection HNSW index exists. Migrations still create the
/// expected production index; this is the drift fuse for future model ids.
pub async fn ensure_ann_projection_index(
    pool: &PgPool,
    model: &EmbeddingModel,
) -> Result<(), StorageError> {
    let Some(projection) = ann_projection_for_model(model) else {
        tracing::warn!(
            model_id = %model.id,
            dimensions = model.dimensions,
            "active embedding model has no ANN projection; vector search may use exact scan"
        );
        return Ok(());
    };

    let index_name = ann_projection_index_name(&projection.projection_id);
    let ddl = format!(
        r#"
        CREATE INDEX CONCURRENTLY IF NOT EXISTS {}
        ON embedding_ann_projections
        USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 100)
        WHERE projection_id = {}
          AND target_kind = 'thought'
        "#,
        sql_identifier(&index_name),
        sql_literal(&projection.projection_id)
    );

    let mut conn = pool.acquire().await?;
    let lock_key = format!("kengram-ann-index:{}", projection.projection_id);
    sqlx::query("SET lock_timeout = '5s'")
        .execute(&mut *conn)
        .await?;
    sqlx::query("SELECT pg_advisory_lock(hashtext($1)::bigint)")
        .bind(&lock_key)
        .execute(&mut *conn)
        .await?;

    let ready_before = ann_projection_index_ready_on_conn(&mut conn, &index_name).await;

    let mut create_error: Option<sqlx::Error> = None;
    if matches!(ready_before, Ok((false,)))
        && let Err(err) = sqlx::query(&ddl).execute(&mut *conn).await
    {
        create_error = Some(err);
    }

    let ready_after: Result<(bool,), sqlx::Error> = if create_error.is_none() {
        ann_projection_index_ready_on_conn(&mut conn, &index_name).await
    } else {
        Ok((false,))
    };

    let analyze_result = if create_error.is_none() && matches!(ready_after, Ok((true,))) {
        sqlx::query("ANALYZE embedding_ann_projections")
            .execute(&mut *conn)
            .await
            .map(|_| ())
    } else {
        Ok(())
    };
    let unlock_result = sqlx::query("SELECT pg_advisory_unlock(hashtext($1)::bigint)")
        .bind(&lock_key)
        .execute(&mut *conn)
        .await;

    let ready_before = ready_before?;
    if let Some(err) = create_error {
        return Err(err.into());
    }
    let ready_after = ready_after?;
    analyze_result?;
    unlock_result?;

    if !ready_after.0 {
        return Err(StorageError::AnnProjectionIndexNotReady(index_name));
    }

    tracing::info!(
        model_id = %model.id,
        projection_id = %projection.projection_id,
        index_name = %index_name,
        existed_before = ready_before.0,
        "ANN projection index ensured"
    );

    Ok(())
}

/// Full startup gate for ANN search. Reconciles any deploy-window gaps,
/// asserts coverage, then ensures the per-projection HNSW index is present
/// and valid. Serve/worker call this before accepting traffic.
pub async fn ensure_ann_projection_ready(
    pool: &PgPool,
    model: &EmbeddingModel,
) -> Result<Option<AnnProjectionCoverage>, StorageError> {
    let coverage = reconcile_ann_projections(pool, model).await?;
    ensure_ann_projection_index(pool, model).await?;
    Ok(coverage)
}

/// Generic startup gate for the active vector-search path. Qwen-sized models
/// use the halfvec projection sidecar; bge-m3:1024 uses its typed vector(1024)
/// thought sidecar and fails closed if that table or HNSW index is not valid.
pub async fn ensure_vector_search_ready(
    pool: &PgPool,
    model: &EmbeddingModel,
) -> Result<Option<AnnProjectionCoverage>, StorageError> {
    if is_bge_m3_1024(model) {
        assert_bge_vector_search_ready(pool).await?;
        return Ok(None);
    }

    ensure_ann_projection_ready(pool, model).await
}

async fn assert_bge_vector_search_ready(pool: &PgPool) -> Result<(), StorageError> {
    if !table_exists(pool, bge::THOUGHT_TABLE).await? {
        return Err(StorageError::BgeSidecarTableMissing(
            bge::THOUGHT_TABLE.to_string(),
        ));
    }

    if !index_ready(pool, bge::THOUGHT_HNSW_INDEX).await? {
        return Err(StorageError::BgeSidecarIndexNotReady(
            bge::THOUGHT_HNSW_INDEX.to_string(),
        ));
    }

    Ok(())
}

async fn assert_bge_chunk_vector_search_ready(pool: &PgPool) -> Result<(), StorageError> {
    if !table_exists(pool, bge::CHUNK_TABLE).await? {
        return Err(StorageError::BgeSidecarTableMissing(
            bge::CHUNK_TABLE.to_string(),
        ));
    }

    if !index_ready(pool, bge::CHUNK_HNSW_INDEX).await? {
        return Err(StorageError::BgeSidecarIndexNotReady(
            bge::CHUNK_HNSW_INDEX.to_string(),
        ));
    }

    Ok(())
}

async fn assert_bge_context_vector_search_ready(pool: &PgPool) -> Result<(), StorageError> {
    if !table_exists(pool, bge::CONTEXT_TABLE).await? {
        return Err(StorageError::BgeSidecarTableMissing(
            bge::CONTEXT_TABLE.to_string(),
        ));
    }

    if !index_ready(pool, bge::CONTEXT_HNSW_INDEX).await? {
        return Err(StorageError::BgeSidecarIndexNotReady(
            bge::CONTEXT_HNSW_INDEX.to_string(),
        ));
    }

    Ok(())
}

async fn table_exists(pool: &PgPool, table_name: &str) -> Result<bool, StorageError> {
    let (exists,): (bool,) = sqlx::query_as("SELECT to_regclass($1) IS NOT NULL")
        .bind(format!("public.{table_name}"))
        .fetch_one(pool)
        .await?;
    Ok(exists)
}

async fn index_ready(pool: &PgPool, index_name: &str) -> Result<bool, StorageError> {
    let (ready,): (bool,) = sqlx::query_as(
        r#"
        SELECT EXISTS (
            SELECT 1
            FROM pg_class c
            JOIN pg_index i ON i.indexrelid = c.oid
            WHERE c.relname = $1
              AND i.indisready
              AND i.indisvalid
        )
        "#,
    )
    .bind(index_name)
    .fetch_one(pool)
    .await?;
    Ok(ready)
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

#[derive(sqlx::FromRow)]
struct ChunkVectorSearchRow {
    id: Uuid,
    scope: String,
    parent_content: String,
    source: String,
    created_at: OffsetDateTime,
    parent_metadata: serde_json::Value,
    content_fingerprint: Vec<u8>,
    tags: serde_json::Value,
    tags_extractor_model: Option<String>,
    tags_extractor_version: Option<i32>,
    tags_extracted_at: Option<OffsetDateTime>,
    chunk_id: Uuid,
    artifact_id: Uuid,
    source_thought_id: Uuid,
    chunk_index: i32,
    chunk_content: String,
    chunker_id: String,
    chunker_version: i32,
    token_estimate: Option<i32>,
    start_char: Option<i32>,
    end_char: Option<i32>,
    chunk_metadata: serde_json::Value,
    distance: f64,
}

#[derive(sqlx::FromRow)]
struct ChunkLexicalSearchRow {
    id: Uuid,
    scope: String,
    parent_content: String,
    source: String,
    created_at: OffsetDateTime,
    parent_metadata: serde_json::Value,
    content_fingerprint: Vec<u8>,
    tags: serde_json::Value,
    tags_extractor_model: Option<String>,
    tags_extractor_version: Option<i32>,
    tags_extracted_at: Option<OffsetDateTime>,
    chunk_id: Uuid,
    artifact_id: Uuid,
    source_thought_id: Uuid,
    chunk_index: i32,
    chunk_content: String,
    chunker_id: String,
    chunker_version: i32,
    token_estimate: Option<i32>,
    start_char: Option<i32>,
    end_char: Option<i32>,
    chunk_metadata: serde_json::Value,
    rank: f32,
}

#[derive(sqlx::FromRow)]
struct ContextGenerationSourceRow {
    chunk_id: Uuid,
    source_thought_id: Uuid,
    scope: String,
    parent_source: String,
    parent_created_at: OffsetDateTime,
    parent_metadata: serde_json::Value,
    parent_content: String,
    chunk_index: i32,
    chunk_content: String,
    chunk_metadata: serde_json::Value,
    raw_chunk_fingerprint: Vec<u8>,
}

#[derive(sqlx::FromRow)]
struct ThoughtRow {
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
}

#[derive(sqlx::FromRow)]
struct LexicalSearchRow {
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
    rank: f32,
}

fn context_generation_source_from_row(
    r: ContextGenerationSourceRow,
) -> Result<ContextGenerationSource, StorageError> {
    Ok(ContextGenerationSource {
        chunk_id: r.chunk_id,
        source_thought_id: ThoughtId::from(r.source_thought_id),
        scope: Scope::new(r.scope)?,
        parent_source: Source::new(r.parent_source)?,
        parent_created_at: r.parent_created_at,
        parent_metadata: Metadata::from(r.parent_metadata),
        parent_content: r.parent_content,
        chunk_index: r.chunk_index,
        chunk_content: r.chunk_content,
        chunk_metadata: r.chunk_metadata,
        raw_chunk_fingerprint: fingerprint_from_bytes(r.raw_chunk_fingerprint)?,
    })
}

fn thought_row_to_thought(r: ThoughtRow) -> Result<Thought, StorageError> {
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
}

fn lexical_rows_to_hits(rows: Vec<LexicalSearchRow>) -> Result<Vec<Hit>, StorageError> {
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
            Ok(Hit::from_lexical_leg(thought, r.rank))
        })
        .collect()
}

// -- M6.0: corpus stats (operator-facing telemetry) ---------------------
//
// `corpus_stats` aggregates counts, byte totals, and per-table storage
// sizes into a single CorpusStats struct for the `kengram stats` CLI
// subcommand. Postgres-specific: uses pg_class / pg_relation_size /
// pg_database_size, which means corpus_stats can't move out of the
// Postgres-only storage layer.

#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub thoughts: ThoughtStats,
    pub embeddings: Vec<EmbeddingModelCount>,
    pub ann_projections: Vec<AnnProjectionCoverage>,
    pub links: LinkStats,
    pub queues: QueueStats,
    pub scopes: Vec<ScopeSummary>,
    pub storage: Vec<TableSize>,
    pub database_total_bytes: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct ThoughtStats {
    pub live: i64,
    pub retracted: i64,
    pub untagged: i64,
    pub content_bytes_total: i64,
    pub content_bytes_avg: i64,
    pub tags_bytes_total: i64,
    pub metadata_bytes_total: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingModelCount {
    pub model_id: String,
    pub model_version: i32,
    pub dimensions: i32,
    pub count: i64,
}

#[derive(Debug, Clone)]
pub struct LinkStats {
    pub live: i64,
    pub soft_deleted: i64,
    pub by_relation: Vec<(String, i64)>,
    pub by_kind: Vec<(String, i64)>,
    pub by_source: Vec<(String, i64)>,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueStats {
    pub pending_embeddings: i64,
    pub pending_tags: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSize {
    pub table: String,
    pub heap_bytes: i64,
    pub indexes_bytes: i64,
    pub total_bytes: i64,
}

/// Count of rows currently in `pending_tags`. Sibling of `count_pending`
/// (which covers `pending_embeddings`).
pub async fn count_pending_tags(pool: &PgPool) -> Result<i64, StorageError> {
    let row = sqlx::query!(r#"SELECT COUNT(*) AS "count!" FROM pending_tags"#)
        .fetch_one(pool)
        .await?;
    Ok(row.count)
}

/// Aggregate corpus + storage telemetry. `scope_prefix` only filters the
/// `scopes` summary section (passed through to `list_scopes(prefix)`);
/// all other counts and byte totals are corpus-global.
///
/// Postgres-specific: uses `pg_class`, `pg_relation_size`,
/// `pg_indexes_size`, `pg_total_relation_size`, and `pg_database_size`
/// from the Postgres system catalogs. These can't be checked via
/// `sqlx::query!` (the macro doesn't introspect system catalogs), so the
/// table-size query uses the runtime-checked `sqlx::query()` form —
/// matches the `insert_embedding` precedent for pgvector binds.
pub async fn corpus_stats(
    pool: &PgPool,
    scope_prefix: Option<&str>,
) -> Result<CorpusStats, StorageError> {
    // 1. Thoughts aggregates (one query, FILTER aggregates).
    let t_row = sqlx::query!(
        r#"
        SELECT
            COUNT(*) FILTER (WHERE retracted_at IS NULL)             AS "live!",
            COUNT(*) FILTER (WHERE retracted_at IS NOT NULL)         AS "retracted!",
            COUNT(*) FILTER (WHERE tags_extractor_version IS NULL
                              AND retracted_at IS NULL)              AS "untagged!",
            COALESCE(SUM(LENGTH(content))         FILTER (WHERE retracted_at IS NULL), 0)::bigint
                AS "content_bytes_total!",
            COALESCE(AVG(LENGTH(content))         FILTER (WHERE retracted_at IS NULL), 0)::bigint
                AS "content_bytes_avg!",
            COALESCE(SUM(LENGTH(tags::text))      FILTER (WHERE retracted_at IS NULL), 0)::bigint
                AS "tags_bytes_total!",
            COALESCE(SUM(LENGTH(metadata::text))  FILTER (WHERE retracted_at IS NULL), 0)::bigint
                AS "metadata_bytes_total!"
        FROM thoughts
        "#,
    )
    .fetch_one(pool)
    .await?;
    let thoughts = ThoughtStats {
        live: t_row.live,
        retracted: t_row.retracted,
        untagged: t_row.untagged,
        content_bytes_total: t_row.content_bytes_total,
        content_bytes_avg: t_row.content_bytes_avg,
        tags_bytes_total: t_row.tags_bytes_total,
        metadata_bytes_total: t_row.metadata_bytes_total,
    };

    // 2. Embeddings by model. Vector dims are constant within a model id;
    // sample one row per group to recover the dimension.
    let e_rows = sqlx::query!(
        r#"
        SELECT
            model_id,
            model_version,
            COUNT(*) AS "count!",
            (vector_dims((SELECT vector FROM embeddings e2
                          WHERE e2.model_id = e.model_id
                            AND e2.model_version = e.model_version
                          LIMIT 1))) AS "dimensions!"
        FROM embeddings e
        GROUP BY model_id, model_version
        ORDER BY model_id, model_version
        "#,
    )
    .fetch_all(pool)
    .await?;
    let embeddings: Vec<EmbeddingModelCount> = e_rows
        .into_iter()
        .map(|r| EmbeddingModelCount {
            model_id: r.model_id,
            model_version: r.model_version,
            dimensions: r.dimensions,
            count: r.count,
        })
        .collect();

    // 3. ANN projection coverage metrics. Populated by migration/startup and
    // periodically refreshed by the worker; this is the operator-visible
    // metric/SLO for projection drift.
    let ann_rows: Vec<(String, String, i32, i64, i64, i64, String)> = sqlx::query_as(
        r#"
        SELECT
            projection_id,
            model_id,
            model_version,
            embedding_count,
            projection_count,
            missing_count,
            status
        FROM embedding_ann_projection_coverage
        ORDER BY projection_id
        "#,
    )
    .fetch_all(pool)
    .await?;
    let ann_projections = ann_rows
        .into_iter()
        .map(
            |(
                projection_id,
                model_id,
                model_version,
                embedding_count,
                projection_count,
                missing_count,
                status,
            )| AnnProjectionCoverage {
                projection_id,
                model_id,
                model_version,
                embedding_count,
                projection_count,
                missing_count,
                inserted_missing: 0,
                status,
            },
        )
        .collect();

    // 4. Link stats — live/soft-deleted counts + group-by aggregates.
    let l_row = sqlx::query!(
        r#"
        SELECT
            COUNT(*) FILTER (WHERE deleted_at IS NULL)     AS "live!",
            COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) AS "soft_deleted!"
        FROM thought_links
        "#,
    )
    .fetch_one(pool)
    .await?;
    let by_relation = sqlx::query!(
        r#"
        SELECT relation, COUNT(*) AS "count!"
        FROM thought_links
        WHERE deleted_at IS NULL
        GROUP BY relation
        ORDER BY COUNT(*) DESC, relation ASC
        "#,
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|r| (r.relation, r.count))
    .collect();
    let by_kind = sqlx::query!(
        r#"
        SELECT to_kind, COUNT(*) AS "count!"
        FROM thought_links
        WHERE deleted_at IS NULL
        GROUP BY to_kind
        ORDER BY COUNT(*) DESC, to_kind ASC
        "#,
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|r| (r.to_kind, r.count))
    .collect();
    let by_source = sqlx::query!(
        r#"
        SELECT source, COUNT(*) AS "count!"
        FROM thought_links
        WHERE deleted_at IS NULL
        GROUP BY source
        ORDER BY COUNT(*) DESC, source ASC
        "#,
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|r| (r.source, r.count))
    .collect();
    let links = LinkStats {
        live: l_row.live,
        soft_deleted: l_row.soft_deleted,
        by_relation,
        by_kind,
        by_source,
    };

    // 5. Queue depths.
    let queues = QueueStats {
        pending_embeddings: count_pending(pool).await?,
        pending_tags: count_pending_tags(pool).await?,
    };

    // 6. Scopes summary (reuses list_scopes; scope_prefix only applies here).
    let scopes = list_scopes(pool, scope_prefix).await?;

    // 7. Per-table sizes via pg_class system catalog. Restricted to public
    // schema regular tables (`relkind='r'`) so we don't surface pg_catalog
    // or sqlx's _sqlx_migrations table cruft. Runtime-checked query (the
    // macro doesn't introspect system catalogs).
    let storage_rows: Vec<(String, i64, i64, i64)> = sqlx::query_as(
        r#"
        SELECT
            c.relname::text AS table_name,
            pg_relation_size(c.oid)::bigint AS heap_bytes,
            pg_indexes_size(c.oid)::bigint AS indexes_bytes,
            pg_total_relation_size(c.oid)::bigint AS total_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        ORDER BY pg_total_relation_size(c.oid) DESC
        "#,
    )
    .fetch_all(pool)
    .await?;
    let storage = storage_rows
        .into_iter()
        .map(
            |(table, heap_bytes, indexes_bytes, total_bytes)| TableSize {
                table,
                heap_bytes,
                indexes_bytes,
                total_bytes,
            },
        )
        .collect();

    let db_row: (i64,) = sqlx::query_as("SELECT pg_database_size(current_database())::bigint")
        .fetch_one(pool)
        .await?;

    Ok(CorpusStats {
        thoughts,
        embeddings,
        ann_projections,
        links,
        queues,
        scopes,
        storage,
        database_total_bytes: db_row.0,
    })
}

// -- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::{
        EmbeddingModel, LinkTarget, Metadata, Scope, Source, SparseEmbeddingModel,
        SparseLexicalVector, SparseWeight, TagKind,
    };
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
    async fn snapshot_excludes_retracted_and_captures_tag_provenance(pool: PgPool) {
        let scope = Scope::new("work").unwrap();
        let source = Source::new("manual").unwrap();
        let metadata = Metadata::empty();

        // A tagged, non-retracted thought.
        let (tagged, _) =
            insert_thought(&pool, new_thought(&scope, &source, &metadata, "tagged one"))
                .await
                .unwrap();
        let tags = Tags {
            people: vec!["Ron".to_string()],
            kind: Some(TagKind::Observation),
            ..Default::default()
        };
        update_thought_tags(&pool, tagged.id, &tags, "ollama/test", 13)
            .await
            .unwrap();

        // An untagged, non-retracted thought (default tags, NULL provenance).
        let (untagged, _) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "untagged one"),
        )
        .await
        .unwrap();

        // A retracted thought — must be excluded.
        let (gone, _) = insert_thought(
            &pool,
            new_thought(&scope, &source, &metadata, "retracted one"),
        )
        .await
        .unwrap();
        retract_thought(&pool, gone.id, Some("test")).await.unwrap();

        let snap = snapshot_nonretracted_tags(&pool).await.unwrap();

        let ids: Vec<ThoughtId> = snap.iter().map(|r| r.thought_id).collect();
        assert!(
            ids.contains(&tagged.id),
            "tagged thought must be in snapshot"
        );
        assert!(
            ids.contains(&untagged.id),
            "untagged thought must be in snapshot"
        );
        assert!(
            !ids.contains(&gone.id),
            "retracted thought must be excluded"
        );
        assert_eq!(snap.len(), 2);

        // Provenance is captured for the tagged row.
        let tagged_row = snap.iter().find(|r| r.thought_id == tagged.id).unwrap();
        assert_eq!(
            tagged_row.tags_extractor_model.as_deref(),
            Some("ollama/test")
        );
        assert_eq!(tagged_row.tags_extractor_version, Some(13));
        assert_eq!(tagged_row.tags["people"][0], "Ron");

        // Untagged row carries NULL provenance.
        let untagged_row = snap.iter().find(|r| r.thought_id == untagged.id).unwrap();
        assert!(untagged_row.tags_extractor_model.is_none());
        assert!(untagged_row.tags_extractor_version.is_none());
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

        let model = EmbeddingModel::new("qwen3-embedding", 4096);
        let vector = vec![0.0_f32; 4096];
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

        let model = EmbeddingModel::new("qwen3-embedding", 4096);
        let embedding = Embedding::new(model.clone(), vec![0.5_f32; 4096]).unwrap();
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

    async fn insert_test_thought_with_metadata(
        pool: &PgPool,
        content: &str,
        scope: &str,
        metadata: Metadata,
    ) -> ThoughtId {
        let scope = Scope::new(scope).unwrap();
        let source = Source::new("test").unwrap();
        let (inserted, _) = insert_thought(pool, new_thought(&scope, &source, &metadata, content))
            .await
            .unwrap();
        inserted.id
    }

    async fn insert_test_chunk(pool: &PgPool, parent_id: ThoughtId, content: &str) -> Uuid {
        let artifact_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        sqlx::query(
            r#"
            INSERT INTO artifacts (id, scope, kind, title, metadata)
            VALUES ($1, 'global', 'thought_chunks', 'test artifact', '{}')
            "#,
        )
        .bind(artifact_id)
        .execute(pool)
        .await
        .unwrap();
        sqlx::query(
            r#"
            INSERT INTO artifact_chunks (
                id,
                artifact_id,
                source_thought_id,
                chunk_index,
                content,
                content_fingerprint,
                chunker_id,
                chunker_version,
                token_estimate,
                start_char,
                end_char,
                metadata
            )
            VALUES ($1,$2,$3,0,$4,$5,'test-chunker',1,6,0,$6,'{"fixture":true}')
            "#,
        )
        .bind(chunk_id)
        .bind(artifact_id)
        .bind(parent_id.into_uuid())
        .bind(content)
        .bind(sha256_of(content).to_vec())
        .bind(content.len() as i32)
        .execute(pool)
        .await
        .unwrap();

        chunk_id
    }

    fn test_context_insert(chunk_id: Uuid, context_text: &str) -> ArtifactChunkContextInsert {
        ArtifactChunkContextInsert {
            chunk_id,
            context_text: context_text.to_string(),
            generator_id: "test-context-generator".to_string(),
            generator_version: 1,
            prompt_version: "test-prompt-v1".to_string(),
            prompt_hash: "test-prompt-hash".to_string(),
            model_id: "test-context-model".to_string(),
            model_version: "1".to_string(),
            pipeline_run_id: None,
            metadata: json!({"fixture": true}),
        }
    }

    fn test_bge_embedding(seed: f32) -> Embedding {
        Embedding::new(EmbeddingModel::bge_m3(), vec![seed; bge::DIMS]).unwrap()
    }

    async fn insert_ready_context_direct(
        pool: &PgPool,
        parent_id: ThoughtId,
        chunk_id: Uuid,
        context_text: &str,
        contextual_content: &str,
    ) -> Uuid {
        let (raw_fingerprint,): (Vec<u8>,) =
            sqlx::query_as("SELECT content_fingerprint FROM artifact_chunks WHERE id = $1")
                .bind(chunk_id)
                .fetch_one(pool)
                .await
                .unwrap();
        let context_id = Uuid::new_v4();
        sqlx::query(
            r#"
            INSERT INTO artifact_chunk_contexts (
                id,
                chunk_id,
                source_thought_id,
                context_text,
                contextual_content,
                raw_chunk_fingerprint,
                contextual_content_fingerprint,
                generator_id,
                generator_version,
                prompt_version,
                prompt_hash,
                model_id,
                model_version,
                contamination_filter_version,
                status,
                metadata
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, digest($5, 'sha256'),
                'direct-test-generator', 1, 'direct-test-prompt', $7,
                'direct-test-model', '1', $8, 'ready', '{"direct":true}'
            )
            "#,
        )
        .bind(context_id)
        .bind(chunk_id)
        .bind(parent_id.into_uuid())
        .bind(context_text)
        .bind(contextual_content)
        .bind(raw_fingerprint)
        .bind(format!("direct-{}", Uuid::new_v4()))
        .bind(CONTEXTUAL_CONTAMINATION_FILTER_VERSION)
        .execute(pool)
        .await
        .unwrap();
        context_id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn contextual_generation_sources_skip_gold_source_file_before_generation(pool: PgPool) {
        let clean_parent =
            insert_test_thought(&pool, "clean parent for contextual generation", "global").await;
        let clean_chunk = insert_test_chunk(&pool, clean_parent, "clean raw chunk").await;
        let gold_parent = insert_test_thought_with_metadata(
            &pool,
            "gold parent should never be prompted",
            "global",
            Metadata::from(json!({"source_file": "eval/gold/kengram-gold-100-answer-key.md"})),
        )
        .await;
        let gold_chunk = insert_test_chunk(&pool, gold_parent, "gold raw chunk").await;

        let sources = select_artifact_chunk_context_generation_sources(
            &pool,
            None,
            None,
            "test-context-generator",
            1,
            "test-prompt-hash",
            10,
        )
        .await
        .unwrap();
        let ids = sources
            .iter()
            .map(|source| source.chunk_id)
            .collect::<Vec<_>>();

        assert!(ids.contains(&clean_chunk));
        assert!(!ids.contains(&gold_chunk));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn generated_kgr_context_rejected_before_embedding_and_pool(pool: PgPool) {
        let parent =
            insert_test_thought(&pool, "clean parent contextual rejection", "global").await;
        let chunk = insert_test_chunk(&pool, parent, "clean raw chunk for context").await;

        let outcome = insert_artifact_chunk_context(
            &pool,
            test_context_insert(
                chunk,
                "Generated context mentions KGR024 answer-key material.",
            ),
        )
        .await
        .unwrap();

        assert_eq!(outcome.status, "rejected");
        assert_eq!(
            outcome.rejection_reason.as_deref(),
            Some("generated_context_eval_marker")
        );
        let stored: (String, String, String, Option<String>) = sqlx::query_as(
            r#"
            SELECT status, context_text, contextual_content, rejection_reason
            FROM artifact_chunk_contexts
            WHERE id = $1
            "#,
        )
        .bind(outcome.context_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(stored.0, "rejected");
        assert!(stored.1.is_empty());
        assert!(stored.2.is_empty());
        assert_eq!(stored.3.as_deref(), Some("generated_context_eval_marker"));

        let embedded = insert_artifact_chunk_context_embedding(
            &pool,
            outcome.context_id,
            &test_bge_embedding(0.2),
        )
        .await
        .unwrap();
        assert!(!embedded);
        let embedding_count: (i64,) = sqlx::query_as(
            "SELECT count(*)::bigint FROM artifact_chunk_context_embeddings_bge_m3 WHERE context_id = $1",
        )
        .bind(outcome.context_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(embedding_count.0, 0);

        let hits =
            search_artifact_chunk_contexts_fts_bounded(&pool, "KGR024", None, None, 10, 1_000)
                .await
                .unwrap();
        assert!(hits.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn contextual_insert_does_not_mutate_raw_chunk_content(pool: PgPool) {
        let parent = insert_test_thought(&pool, "parent for raw immutability", "global").await;
        let raw = "original raw artifact chunk";
        let chunk = insert_test_chunk(&pool, parent, raw).await;

        let outcome = insert_artifact_chunk_context(
            &pool,
            test_context_insert(chunk, "Helpful local generated context."),
        )
        .await
        .unwrap();
        assert_eq!(outcome.status, "ready");

        let stored_raw: (String,) =
            sqlx::query_as("SELECT content FROM artifact_chunks WHERE id = $1")
                .bind(chunk)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(stored_raw.0, raw);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn contextual_fts_and_vector_filter_forced_bad_ready_context(pool: PgPool) {
        let bad_parent =
            insert_test_thought(&pool, "clean parent for forced bad context", "global").await;
        let bad_chunk = insert_test_chunk(&pool, bad_parent, "clean raw forced bad chunk").await;
        let bad_context = insert_ready_context_direct(
            &pool,
            bad_parent,
            bad_chunk,
            "KGR024 forced bad generated context",
            "KGR024 forced bad generated context\n\nclean raw forced bad chunk",
        )
        .await;
        sqlx::query(
            r#"
            INSERT INTO artifact_chunk_context_embeddings_bge_m3 (
                context_id,
                model_id,
                model_version,
                dimensions,
                embedding
            )
            VALUES ($1, $2, $3, $4, $5)
            "#,
        )
        .bind(bad_context)
        .bind(bge::MODEL_ID)
        .bind(bge::MODEL_VERSION)
        .bind(bge::DIMS_I32)
        .bind(pgvector::Vector::from(vec![0.3_f32; bge::DIMS]))
        .execute(&pool)
        .await
        .unwrap();

        let fts_hits = search_artifact_chunk_contexts_fts_bounded(
            &pool,
            "KGR024 forced bad",
            None,
            None,
            10,
            1_000,
        )
        .await
        .unwrap();
        assert!(fts_hits.iter().all(|hit| hit.thought.id != bad_parent));

        let vector_hits = search_artifact_chunk_contexts_vector_knn(
            &pool,
            vec![0.3_f32; bge::DIMS],
            &EmbeddingModel::bge_m3(),
            None,
            None,
            10,
        )
        .await
        .unwrap();
        assert!(vector_hits.iter().all(|hit| hit.thought.id != bad_parent));
    }

    fn test_sparse_vector(weights: Vec<(u32, f32)>) -> SparseLexicalVector {
        SparseLexicalVector::new(
            SparseEmbeddingModel::bge_m3_sparse(),
            weights
                .into_iter()
                .map(|(token_id, weight)| SparseWeight::new(token_id, weight))
                .collect(),
        )
        .unwrap()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_thought_sparse_embedding_persists_sparsevec_row(pool: PgPool) {
        let content = "stage3 sparse thought";
        let thought_id = insert_test_thought(&pool, content, "global").await;
        let vector = test_sparse_vector(vec![(0, 0.5), (2, 1.25)]);
        let mut provenance =
            SparseEmbeddingProvenance::bge_m3_flag_embedding("FlagEmbedding 1.0 test");
        provenance.producer_metadata = json!({"device": "mps", "fixture": true});

        insert_thought_sparse_embedding(
            &pool,
            thought_id,
            sha256_of(content),
            content.len() as i32,
            &vector,
            &provenance,
        )
        .await
        .unwrap();

        let row: (String, i32, String, serde_json::Value) = sqlx::query_as(
            r#"
            SELECT embedding::text, nonzero_count, generator, producer_metadata
            FROM thought_sparse_embeddings_bge_m3
            WHERE thought_id = $1
            "#,
        )
        .bind(thought_id.into_uuid())
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(row.0, "{1:0.5,3:1.25}/250002");
        assert_eq!(row.1, 2);
        assert_eq!(row.2, "FlagEmbedding.BGEM3FlagModel");
        assert_eq!(row.3["device"], "mps");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_artifact_chunk_sparse_embedding_updates_on_conflict(pool: PgPool) {
        let parent = insert_test_thought(&pool, "sparse parent", "global").await;
        let chunk_content = "stage3 sparse chunk";
        let chunk_id = insert_test_chunk(&pool, parent, chunk_content).await;
        let provenance = SparseEmbeddingProvenance::bge_m3_flag_embedding("FlagEmbedding 1.0 test");
        let first = test_sparse_vector(vec![(0, 0.5)]);
        let second = test_sparse_vector(vec![(1, 2.0), (250_001, 1.0)]);

        insert_artifact_chunk_sparse_embedding(
            &pool,
            chunk_id,
            sha256_of(chunk_content),
            chunk_content.len() as i32,
            &first,
            &provenance,
        )
        .await
        .unwrap();
        insert_artifact_chunk_sparse_embedding(
            &pool,
            chunk_id,
            sha256_of(chunk_content),
            chunk_content.len() as i32,
            &second,
            &provenance,
        )
        .await
        .unwrap();

        let row: (i64, String, i32) = sqlx::query_as(
            r#"
            SELECT count(*)::bigint, max(embedding::text), max(nonzero_count)
            FROM artifact_chunk_sparse_embeddings_bge_m3
            WHERE chunk_id = $1
            "#,
        )
        .bind(chunk_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(row.0, 1);
        assert_eq!(row.1, "{2:2,250002:1}/250002");
        assert_eq!(row.2, 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_sparse_embedding_rejects_wrong_sparse_model(pool: PgPool) {
        let thought_id = insert_test_thought(&pool, "wrong sparse model", "global").await;
        let vector = SparseLexicalVector::new(
            SparseEmbeddingModel::new("other:sparse", 1, 250_002),
            vec![SparseWeight::new(0, 1.0)],
        )
        .unwrap();
        let provenance = SparseEmbeddingProvenance::bge_m3_flag_embedding("FlagEmbedding 1.0 test");

        let err = insert_thought_sparse_embedding(
            &pool,
            thought_id,
            sha256_of("wrong sparse model"),
            "wrong sparse model".len() as i32,
            &vector,
            &provenance,
        )
        .await
        .unwrap_err();

        assert!(matches!(err, StorageError::InvalidSparseModel { .. }));
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

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_finds_exact_match(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;
        insert_test_thought(&pool, "weather is nice today", "personal").await;

        let hits = search_fts(&pool, "tcgplayer", None, None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].thought.content.contains("tcgplayer"));
        assert!(hits[0].lexical_score.unwrap() > 0.0);
        assert_eq!(hits[0].trigram_score, hits[0].lexical_score);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_excludes_eval_contamination_candidates(pool: PgPool) {
        let clean = insert_test_thought(
            &pool,
            "clean answer marker tcgplayer canonical baseline",
            "global",
        )
        .await;
        let denied = insert_test_thought(
            &pool,
            "KGR024 answer marker tcgplayer canonical baseline",
            "global",
        )
        .await;

        let hits = search_fts(&pool, "tcgplayer canonical", None, None, 10)
            .await
            .unwrap();
        let hit_ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();

        assert!(hit_ids.contains(&clean));
        assert!(
            !hit_ids.contains(&denied),
            "KGR-labeled eval rows must be excluded before FTS candidate pooling"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_respects_scope(pool: PgPool) {
        insert_test_thought(&pool, "tcgplayer info", "work").await;
        insert_test_thought(&pool, "tcgplayer info two", "personal").await;

        let hits = search_fts(&pool, "tcgplayer", Some("work"), None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].thought.scope.as_str(), "work");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_returns_empty_for_no_match(pool: PgPool) {
        insert_test_thought(&pool, "completely unrelated text", "global").await;
        let hits = search_fts(&pool, "xyzzyqwerty", None, None, 10)
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_returns_empty_for_empty_or_operator_only_queries(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;

        for query in ["", "   ", "&", "|", "!", "&&", "||", " ! | & "] {
            let hits = search_fts(&pool, query, None, None, 10).await.unwrap();
            assert!(hits.is_empty(), "query {query:?} should not hit FTS");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_normalizes_control_chars_before_querying(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;

        let hits = search_fts(&pool, "\u{0000}\u{0007}tcgplayer\u{001f}", None, None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].thought.content.contains("tcgplayer"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_bounded_sets_statement_timeout(pool: PgPool) {
        let mut tx = pool.begin().await.unwrap();

        set_statement_timeout(&mut tx, 300).await.unwrap();

        let (value,): (String,) = sqlx::query_as("SELECT current_setting('statement_timeout')")
            .fetch_one(&mut *tx)
            .await
            .unwrap();

        assert_eq!(value, "300ms");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_bounded_finds_exact_match(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;

        let hits = search_fts_bounded(&pool, "tcgplayer", None, None, 10, 300)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].thought.content.contains("tcgplayer"));
        assert!(hits[0].lexical_score.unwrap() > 0.0);
        assert_eq!(hits[0].trigram_score, hits[0].lexical_score);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_bounded_excludes_eval_source_file_candidates(pool: PgPool) {
        let clean = insert_test_thought(
            &pool,
            "clean retrieval baseline answer marker tcgplayer canonical",
            "global",
        )
        .await;
        let denied = insert_test_thought_with_metadata(
            &pool,
            "retrieval baseline answer marker tcgplayer canonical",
            "global",
            Metadata::from(json!({"source_file": "reports/kengram-gold-100-answer-key.md"})),
        )
        .await;

        let hits = search_fts_bounded(&pool, "tcgplayer canonical", None, None, 10, 300)
            .await
            .unwrap();
        let hit_ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();

        assert!(hit_ids.contains(&clean));
        assert!(
            !hit_ids.contains(&denied),
            "eval/gold source_file rows must be excluded before bounded FTS candidate pooling"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn artifact_chunk_fts_excludes_eval_contamination_in_chunk_content(pool: PgPool) {
        let clean = insert_test_thought(&pool, "clean parent row", "global").await;
        let denied = insert_test_thought(&pool, "also clean parent row", "global").await;
        insert_test_chunk(
            &pool,
            clean,
            "clean answer marker tcgplayer canonical chunk body",
        )
        .await;
        insert_test_chunk(
            &pool,
            denied,
            "KGR024 answer marker tcgplayer canonical chunk body",
        )
        .await;

        let hits =
            search_artifact_chunks_fts_bounded(&pool, "tcgplayer canonical", None, None, 10, 300)
                .await
                .unwrap();
        let hit_ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();

        assert!(hit_ids.contains(&clean));
        assert!(
            !hit_ids.contains(&denied),
            "KGR-labeled chunk content must be excluded before chunk FTS candidate pooling"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_bounded_rolls_back_after_statement_timeout(pool: PgPool) {
        insert_test_thought(&pool, "remembering tcgplayer integration", "work").await;

        let mut blocker = pool.begin().await.unwrap();
        sqlx::query("LOCK TABLE thoughts IN ACCESS EXCLUSIVE MODE")
            .execute(&mut *blocker)
            .await
            .unwrap();

        let started = std::time::Instant::now();
        let err = search_fts_bounded(&pool, "tcgplayer", None, None, 10, 50)
            .await
            .unwrap_err();
        assert!(
            started.elapsed() < std::time::Duration::from_millis(800),
            "statement_timeout should cancel the blocked FTS query promptly"
        );
        assert!(
            err.is_query_canceled(),
            "expected Postgres query-canceled error, got {err:?}"
        );

        blocker.rollback().await.unwrap();

        let hits = search_fts_bounded(&pool, "tcgplayer", None, None, 10, 300)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].thought.content.contains("tcgplayer"));
    }

    fn unit_vector_4096(pos: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; 4096];
        v[pos] = 1.0;
        v
    }

    fn unit_vector_1024(pos: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; 1024];
        v[pos] = 1.0;
        v
    }

    async fn insert_raw_embedding_without_projection(
        pool: &PgPool,
        target_id: Uuid,
        model: &EmbeddingModel,
        vector: Vec<f32>,
    ) {
        sqlx::query(
            r#"
            INSERT INTO embeddings (target_kind, target_id, model_id, model_version, vector)
            VALUES ($1, $2, $3, $4, $5)
            "#,
        )
        .bind(target::THOUGHT)
        .bind(target_id)
        .bind(&model.id)
        .bind(1_i32)
        .bind(pgvector::Vector::from(vector))
        .execute(pool)
        .await
        .unwrap();
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_finds_inserted_vector(pool: PgPool) {
        let model = EmbeddingModel::new("test:4096", 4096);

        let id_a = insert_test_thought(&pool, "a", "global").await;
        let id_b = insert_test_thought(&pool, "b", "global").await;

        let va = unit_vector_4096(0);
        let vb = unit_vector_4096(1);

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

        reconcile_ann_projections(&pool, &model).await.unwrap();

        let hits = search_vector_knn(&pool, va, &model, None, None, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].thought.id, id_a);
        assert!((hits[0].vector_score.unwrap() - 1.0).abs() < 1e-4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_excludes_eval_contamination_candidates(pool: PgPool) {
        let model = EmbeddingModel::new("test:4096", 4096);

        let clean = insert_test_thought(&pool, "clean vector candidate", "global").await;
        let denied = insert_test_thought(&pool, "KGR024 vector candidate", "global").await;

        let query = unit_vector_4096(0);
        let clean_vector = unit_vector_4096(1);

        insert_thought_embedding(
            &pool,
            clean,
            &Embedding::new(model.clone(), clean_vector).unwrap(),
        )
        .await
        .unwrap();
        insert_thought_embedding(
            &pool,
            denied,
            &Embedding::new(model.clone(), query.clone()).unwrap(),
        )
        .await
        .unwrap();

        reconcile_ann_projections(&pool, &model).await.unwrap();

        let hits = search_vector_knn(&pool, query, &model, None, None, 10)
            .await
            .unwrap();
        let hit_ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();

        assert!(hit_ids.contains(&clean));
        assert!(
            !hit_ids.contains(&denied),
            "KGR-labeled eval rows must be excluded before vector candidate pooling"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn artifact_chunk_vector_excludes_eval_contamination_in_chunk_content(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let clean = insert_test_thought(&pool, "clean parent vector row", "global").await;
        let denied = insert_test_thought(&pool, "also clean parent vector row", "global").await;
        let clean_chunk = insert_test_chunk(&pool, clean, "clean chunk vector candidate").await;
        let denied_chunk = insert_test_chunk(&pool, denied, "KGR024 chunk vector candidate").await;

        let query = unit_vector_1024(0);
        insert_artifact_chunk_embedding(
            &pool,
            denied_chunk,
            &Embedding::new(model.clone(), query.clone()).unwrap(),
        )
        .await
        .unwrap();
        insert_artifact_chunk_embedding(
            &pool,
            clean_chunk,
            &Embedding::new(model.clone(), unit_vector_1024(1)).unwrap(),
        )
        .await
        .unwrap();

        let hits = search_artifact_chunks_vector_knn(&pool, query, &model, None, None, 10)
            .await
            .unwrap();
        let hit_ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();

        assert!(hit_ids.contains(&clean));
        assert!(
            !hit_ids.contains(&denied),
            "KGR-labeled chunk content must be excluded before chunk vector candidate pooling"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_raw_fallback_covers_scoped_projection_gap(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);

        let id_a = insert_test_thought(&pool, "projected", "scope-a").await;
        let id_b = insert_test_thought(&pool, "raw-only", "scope-b").await;

        let va = unit_vector_4096(0);
        let vb = unit_vector_4096(1);

        insert_thought_embedding(&pool, id_a, &Embedding::new(model.clone(), va).unwrap())
            .await
            .unwrap();
        reconcile_ann_projections(&pool, &model).await.unwrap();

        insert_raw_embedding_without_projection(&pool, id_b.into_uuid(), &model, vb.clone()).await;

        let hits = search_vector_knn(&pool, vb, &model, Some("scope-b"), None, 10)
            .await
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].thought.id, id_b);
        assert_eq!(hits[0].thought.scope.as_str(), "scope-b");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_raw_fallback_covers_scope_prefix_projection_gap(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);

        let id_projected = insert_test_thought(&pool, "projected lake alpha", "lake.alpha").await;
        let id_raw_only = insert_test_thought(&pool, "raw-only lake beta", "lake.beta").await;
        let id_outside = insert_test_thought(&pool, "raw-only outside", "outside").await;

        let v_projected = unit_vector_4096(0);
        let v_raw_only = unit_vector_4096(1);
        let v_outside = unit_vector_4096(2);

        insert_thought_embedding(
            &pool,
            id_projected,
            &Embedding::new(model.clone(), v_projected).unwrap(),
        )
        .await
        .unwrap();
        reconcile_ann_projections(&pool, &model).await.unwrap();

        insert_raw_embedding_without_projection(
            &pool,
            id_raw_only.into_uuid(),
            &model,
            v_raw_only.clone(),
        )
        .await;
        insert_raw_embedding_without_projection(&pool, id_outside.into_uuid(), &model, v_outside)
            .await;

        let hits = search_vector_knn(&pool, v_raw_only, &model, None, Some("lake."), 10)
            .await
            .unwrap();

        let hit_ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();
        assert!(
            hit_ids.contains(&id_raw_only),
            "prefix fallback must include raw-only rows under the requested scope_prefix"
        );
        assert!(
            hit_ids.contains(&id_projected),
            "prefix fallback should still include projected rows under the requested scope_prefix"
        );
        assert!(
            !hit_ids.contains(&id_outside),
            "prefix fallback must preserve the requested scope_prefix filter"
        );
        assert_eq!(hits[0].thought.id, id_raw_only);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ensure_ann_projection_index_reuses_migration_index_name(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);

        ensure_ann_projection_index(&pool, &model).await.unwrap();

        let names: Vec<(String,)> = sqlx::query_as(
            r#"
            SELECT c.relname::text
            FROM pg_class c
            JOIN pg_index i ON i.indexrelid = c.oid
            WHERE c.relname LIKE 'embedding_ann_projection_qwen3%hnsw'
              AND i.indisvalid
              AND i.indisready
            ORDER BY c.relname
            "#,
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        let names = names.into_iter().map(|(name,)| name).collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["embedding_ann_projection_qwen3_embedding_halfvec_3072_hnsw"]
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_embedding_creates_ann_projection_for_large_model(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);
        let id = insert_test_thought(&pool, "projection source", "global").await;
        let target_id = id.into_uuid();

        insert_embedding(
            &pool,
            target::THOUGHT,
            target_id,
            &model,
            unit_vector_4096(0),
        )
        .await
        .unwrap();

        let (count,): (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM embedding_ann_projections
            WHERE target_id = $1
              AND projection_id = 'qwen3-embedding:halfvec:3072'
            "#,
        )
        .bind(target_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_embedding_routes_bge_m3_to_typed_sidecar_only(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let id = insert_test_thought(&pool, "bge sidecar source", "global").await;
        let target_id = id.into_uuid();

        insert_thought_embedding(
            &pool,
            id,
            &Embedding::new(model.clone(), unit_vector_1024(0)).unwrap(),
        )
        .await
        .unwrap();

        assert!(thought_has_embedding(&pool, id, &model).await.unwrap());

        let (sidecar_count,): (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM thought_embeddings_bge_m3
            WHERE thought_id = $1
              AND model_id = 'bge-m3:1024'
              AND model_version = 1
            "#,
        )
        .bind(target_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        let (raw_count,): (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM embeddings
            WHERE target_kind = 'thought'
              AND target_id = $1
              AND model_id = 'bge-m3:1024'
            "#,
        )
        .bind(target_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(sidecar_count, 1);
        assert_eq!(raw_count, 0, "bge-m3 must not hit vector(4096)");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn reconcile_ann_projections_backfills_missing_projection_and_marks_ok(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);
        let id = insert_test_thought(&pool, "raw-only projection source", "scope-a").await;
        let target_id = id.into_uuid();
        insert_raw_embedding_without_projection(&pool, target_id, &model, unit_vector_4096(0))
            .await;

        let coverage = reconcile_ann_projections(&pool, &model)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(coverage.inserted_missing, 1);
        assert_eq!(coverage.missing_count, 0);
        assert_eq!(coverage.status, "ok");

        let (count,): (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM embedding_ann_projections
            WHERE target_id = $1
              AND projection_id = 'qwen3-embedding:halfvec:3072'
            "#,
        )
        .bind(target_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn assert_ann_projection_coverage_detects_raw_only_drift(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);
        let id = insert_test_thought(&pool, "raw-only drift", "scope-b").await;
        insert_raw_embedding_without_projection(&pool, id.into_uuid(), &model, unit_vector_4096(1))
            .await;

        let err = assert_ann_projection_coverage(&pool, &model)
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            StorageError::AnnProjectionCoverageMismatch {
                missing_count: 1,
                ..
            }
        ));

        let (status, missing_count): (String, i64) = sqlx::query_as(
            r#"
            SELECT status, missing_count
            FROM embedding_ann_projection_coverage
            WHERE projection_id = 'qwen3-embedding:halfvec:3072'
            "#,
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(status, "diverged");
        assert_eq!(missing_count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_embedding_rolls_back_raw_row_when_projection_insert_fails(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);
        let id = insert_test_thought(&pool, "projection rollback", "global").await;
        let target_id = id.into_uuid();

        sqlx::query(
            r#"
            ALTER TABLE embedding_ann_projections
            ADD CONSTRAINT force_projection_insert_failure
            CHECK (projection_id <> 'qwen3-embedding:halfvec:3072')
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        let result = insert_embedding(
            &pool,
            target::THOUGHT,
            target_id,
            &model,
            unit_vector_4096(0),
        )
        .await;

        assert!(result.is_err());

        let (embedding_count,): (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM embeddings
            WHERE target_id = $1
              AND model_id = 'qwen3-embedding'
            "#,
        )
        .bind(target_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(embedding_count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_uses_ann_projection_for_large_model(pool: PgPool) {
        let model = EmbeddingModel::new("qwen3-embedding", 4096);

        let id_a = insert_test_thought(&pool, "a", "global").await;
        let id_b = insert_test_thought(&pool, "b", "global").await;

        let va = unit_vector_4096(0);
        let vb = unit_vector_4096(1);

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
    async fn search_vector_knn_uses_bge_m3_typed_sidecar(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();

        let id_a = insert_test_thought(&pool, "bge a", "global").await;
        let id_b = insert_test_thought(&pool, "bge b", "global").await;

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

        ensure_vector_search_ready(&pool, &model).await.unwrap();
        let hits = search_vector_knn(&pool, va, &model, None, None, 10)
            .await
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].thought.id, id_a);
        assert!((hits[0].vector_score.unwrap() - 1.0).abs() < 1e-4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ensure_vector_search_ready_fails_closed_for_missing_bge_index(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();

        sqlx::query("DROP INDEX thought_embeddings_bge_m3_hnsw")
            .execute(&pool)
            .await
            .unwrap();

        let err = ensure_vector_search_ready(&pool, &model).await.unwrap_err();

        assert!(matches!(
            err,
            StorageError::BgeSidecarIndexNotReady(ref name)
                if name == "thought_embeddings_bge_m3_hnsw"
        ));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ann_projection_search_sets_measured_hnsw_ef_search(pool: PgPool) {
        let mut tx = pool.begin().await.unwrap();

        set_ann_projection_ef_search(&mut tx).await.unwrap();

        let (value,): (String,) = sqlx::query_as("SELECT current_setting('hnsw.ef_search')")
            .fetch_one(&mut *tx)
            .await
            .unwrap();

        assert_eq!(value, ann::HALF_3072_HNSW_EF_SEARCH.to_string());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_vector_knn_filters_by_model(pool: PgPool) {
        let model_a = EmbeddingModel::new("test-a:4096", 4096);
        let model_b = EmbeddingModel::new("test-b:4096", 4096);

        let id = insert_test_thought(&pool, "thought", "global").await;
        let va = unit_vector_4096(0);
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
        let model = EmbeddingModel::new("test:4096", 4096);

        let id_a = insert_test_thought(&pool, "a", "global").await;
        let _id_b = insert_test_thought(&pool, "b", "global").await;

        // Embed only `a`.
        let va = unit_vector_4096(0);
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
        let model = EmbeddingModel::new("test:4096", 4096);
        let id_a = insert_test_thought(&pool, "a", "global").await;
        let _id_b = insert_test_thought(&pool, "b", "global").await;

        // Embed only `a`.
        let va = unit_vector_4096(0);
        insert_thought_embedding(&pool, id_a, &Embedding::new(model.clone(), va).unwrap())
            .await
            .unwrap();

        let enqueued = enqueue_unembedded_thoughts(&pool, &model.id, None, None, 100)
            .await
            .unwrap();
        assert_eq!(enqueued, 1, "only `b` should be enqueued");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_unembedded_thoughts_uses_bge_m3_sidecar(pool: PgPool) {
        let model = EmbeddingModel::bge_m3();
        let id_a = insert_test_thought(&pool, "bge embedded", "global").await;
        let _id_b = insert_test_thought(&pool, "bge unembedded", "global").await;

        insert_thought_embedding(
            &pool,
            id_a,
            &Embedding::new(model.clone(), unit_vector_1024(0)).unwrap(),
        )
        .await
        .unwrap();

        let enqueued = enqueue_unembedded_thoughts(&pool, &model.id, None, None, 100)
            .await
            .unwrap();
        assert_eq!(
            enqueued, 1,
            "only the sidecar-missing thought should enqueue"
        );

        let rows: Vec<(Uuid,)> = sqlx::query_as(
            r#"
            SELECT target_id
            FROM pending_embeddings
            WHERE model_id = 'bge-m3:1024'
            "#,
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_ne!(rows[0].0, id_a.into_uuid());
    }

    // -- M4: tag-sidecar tests ------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn update_thought_tags_persists_jsonb_and_provenance(pool: PgPool) {
        let id = insert_test_thought(&pool, "tagged thought", "global").await;

        let tags = Tags {
            people: vec!["Sarah".into()],
            entities: vec!["kengram".into()],
            action_items: vec!["follow up".into()],
            topics: vec!["meetings".into()],
            dates_mentioned: vec!["Thursday".into()],
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        update_thought_tags(&pool, id, &tags, "vllm/qwen3-coder:30b", 1)
            .await
            .unwrap();

        let read = fetch_thought_tags(&pool, id).await.unwrap().unwrap();
        assert_eq!(read.tags, tags);
        assert_eq!(
            read.tagger_model_id.as_deref(),
            Some("vllm/qwen3-coder:30b")
        );
        assert_eq!(read.tagger_version, Some(1));
        assert!(read.tagged_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn enqueue_tag_job_inserts_into_pending_tags(pool: PgPool) {
        let id = insert_test_thought(&pool, "to tag", "global").await;
        let inserted = enqueue_tag_job(&pool, id, "vllm/qwen3-coder:30b")
            .await
            .unwrap();
        assert!(inserted);

        let jobs = fetch_pending_tag_jobs(&pool, 10).await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].thought_id, id);
        assert_eq!(jobs[0].tagger_model_id, "vllm/qwen3-coder:30b");
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
            &pool, /*target_version*/ 1, /*rerun*/ false, /*force*/ false, None,
            None, None, 100,
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
        let walk = find_untagged_or_stale_thoughts(&pool, 2, true, false, None, None, None, 100)
            .await
            .unwrap();
        let ids: Vec<ThoughtId> = walk.iter().map(|t| t.id).collect();
        assert!(ids.contains(&untagged));
        assert!(ids.contains(&stale_v1));
        assert!(!ids.contains(&fresh_v2));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn find_untagged_or_stale_thoughts_returns_all_when_force_true(pool: PgPool) {
        let untagged = insert_test_thought(&pool, "untagged", "global").await;
        let stale_v1 = insert_test_thought(&pool, "stale at v1", "global").await;
        update_thought_tags(&pool, stale_v1, &Tags::default(), "v1-model", 1)
            .await
            .unwrap();
        let current_v2 = insert_test_thought(&pool, "current at v2", "global").await;
        update_thought_tags(&pool, current_v2, &Tags::default(), "v2-model", 2)
            .await
            .unwrap();

        // force=true (rerun=false) → every thought is walked regardless of
        // version, including the one already at the target version.
        let walk = find_untagged_or_stale_thoughts(&pool, 2, false, true, None, None, None, 100)
            .await
            .unwrap();
        let ids: Vec<ThoughtId> = walk.iter().map(|t| t.id).collect();
        assert!(ids.contains(&untagged));
        assert!(ids.contains(&stale_v1));
        assert!(ids.contains(&current_v2));

        // force still honours the scope filter (it narrows the forced set).
        let other = insert_test_thought(&pool, "elsewhere", "other").await;
        update_thought_tags(&pool, other, &Tags::default(), "v2-model", 2)
            .await
            .unwrap();
        let scoped =
            find_untagged_or_stale_thoughts(&pool, 2, false, true, Some("global"), None, None, 100)
                .await
                .unwrap();
        let scoped_ids: Vec<ThoughtId> = scoped.iter().map(|t| t.id).collect();
        assert!(scoped_ids.contains(&current_v2));
        assert!(!scoped_ids.contains(&other));
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
                entities: vec!["kengram".into()],
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
        assert_eq!(work_v.entities, vec!["kengram".to_string()]);

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

    // -- M5: selective relations (links from a thought to a polymorphic
    //        target — thought, entity, person, or URL since M5.2). --------

    fn t(id: ThoughtId) -> LinkTarget {
        LinkTarget::Thought(id)
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_returns_id_and_is_new(pool: PgPool) {
        let a = insert_test_thought(&pool, "thought A", "global").await;
        let b = insert_test_thought(&pool, "thought B", "global").await;

        let (link_id, is_new) = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        assert!(is_new);
        assert_ne!(*link_id.as_uuid(), Uuid::nil());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_duplicate_triple_is_idempotent(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;

        let (first_id, first_is_new) = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        let (second_id, second_is_new) = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        assert!(first_is_new);
        assert!(!second_is_new, "second insert of same triple must be no-op");
        assert_eq!(first_id, second_id, "must return same link id on conflict");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_self_reference_rejected_by_db(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let err = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(a),
            LinkSource::Agent,
            None,
        )
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
            &t(phantom),
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
            &t(b),
            LinkSource::Agent,
            Some("first refinement during dogfood"),
        )
        .await
        .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
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
    async fn delete_link_soft_deletes_and_status_three_ways(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        // Live → DeletedNow (soft-deletes, returns Some(id)).
        let soft = delete_link(&pool, a, RelationKind::Refines, &t(b))
            .await
            .unwrap();
        assert!(soft.is_some(), "live edge must soft-delete");
        assert_eq!(
            lookup_link_status(&pool, a, RelationKind::Refines, &t(b))
                .await
                .unwrap(),
            LinkStatus::SoftDeleted
        );

        // Already soft-deleted → returns None.
        let again = delete_link(&pool, a, RelationKind::Refines, &t(b))
            .await
            .unwrap();
        assert!(
            again.is_none(),
            "second delete on soft-deleted edge is no-op"
        );

        // Edge sits inert in the table — not hard-deleted.
        let row = sqlx::query!(
            "SELECT deleted_at FROM thought_links WHERE from_thought_id = $1",
            a.into_uuid()
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.deleted_at.is_some(), "deleted_at must be populated");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn lookup_link_status_distinguishes_three_states(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        // NeverExisted.
        assert_eq!(
            lookup_link_status(&pool, a, RelationKind::Refines, &t(b))
                .await
                .unwrap(),
            LinkStatus::NeverExisted
        );
        // Live.
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            lookup_link_status(&pool, a, RelationKind::Refines, &t(b))
                .await
                .unwrap(),
            LinkStatus::Live
        );
        // SoftDeleted.
        delete_link(&pool, a, RelationKind::Refines, &t(b))
            .await
            .unwrap();
        assert_eq!(
            lookup_link_status(&pool, a, RelationKind::Refines, &t(b))
                .await
                .unwrap(),
            LinkStatus::SoftDeleted
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_excludes_soft_deleted_edges(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        delete_link(&pool, a, RelationKind::Refines, &t(b))
            .await
            .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert!(
            related.is_empty(),
            "soft-deleted edges must not appear in fetch_related_thoughts"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_after_soft_delete_creates_fresh_live_row(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        let (first_id, _) = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        delete_link(&pool, a, RelationKind::Refines, &t(b))
            .await
            .unwrap();
        let (second_id, is_new) = insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        // The partial unique index ignores soft-deleted rows, so re-asserting
        // succeeds with a fresh link id.
        assert!(is_new);
        assert_ne!(first_id, second_id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_outbound_returns_to_side_only(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        let from_a = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].target, LinkTarget::Thought(b));
        assert_eq!(from_a[0].direction, LinkDirection::Outbound);

        let from_b = fetch_related_thoughts(&pool, b, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert!(from_b.is_empty(), "B has no outbound edges");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_inbound_returns_from_side_only(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        let into_b = fetch_related_thoughts(&pool, b, None, None, LinkDirection::Inbound)
            .await
            .unwrap();
        assert_eq!(into_b.len(), 1);
        assert_eq!(into_b[0].target, LinkTarget::Thought(a));
        assert_eq!(into_b[0].direction, LinkDirection::Inbound);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_both_returns_outbound_plus_inbound(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        let c = insert_test_thought(&pool, "C", "global").await;
        // A refines B; C refines A. So A has 1 outbound + 1 inbound edge.
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        insert_link(
            &pool,
            c,
            RelationKind::Refines,
            &t(a),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Both)
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
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        insert_link(
            &pool,
            a,
            RelationKind::Replaces,
            &t(c),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        let only_refines = fetch_related_thoughts(
            &pool,
            a,
            Some(&[RelationKind::Refines]),
            None,
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
            None,
            LinkDirection::Outbound,
        )
        .await
        .unwrap();
        assert_eq!(multi.len(), 2);

        let only_requires = fetch_related_thoughts(
            &pool,
            a,
            Some(&[RelationKind::Requires]),
            None,
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
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        retract_thought(&pool, b, Some("dogfood retraction"))
            .await
            .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        // Soft retraction preserves the edge — it just surfaces the flag.
        assert_eq!(related.len(), 1);
        assert_eq!(
            related[0].thought_retracted,
            Some(true),
            "retracted state must propagate to the response"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn cascade_on_thought_hard_delete_removes_edges(pool: PgPool) {
        // Direct DELETE FROM thoughts triggers the ON DELETE CASCADE on
        // thought_links. Kengram itself uses soft-retraction, but the DB
        // invariant should still hold for any future hard-delete pathway.
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();

        sqlx::query!("DELETE FROM thoughts WHERE id = $1", b.into_uuid())
            .execute(&pool)
            .await
            .unwrap();

        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert!(
            related.is_empty(),
            "edge must be CASCADE-deleted with the thought"
        );
    }

    // -- M5.2: heterogeneous targets + migration audit ----------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_writes_entity_target(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let target = LinkTarget::Entity("Probe 2".into());
        let (_id, is_new) = insert_link(
            &pool,
            a,
            RelationKind::BelongsTo,
            &target,
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        assert!(is_new);
        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].target, target);
        assert!(related[0].thought_scope.is_none());
        assert!(related[0].thought_content.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn insert_link_writes_url_target(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let target = LinkTarget::Url("https://anthropic.com".into());
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &target,
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].target, target);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn url_target_check_rejects_non_http_scheme(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let err = insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Url("ftp://example.com".into()),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap_err();
        // DB CHECK violation. (kengram-mcp also pre-validates; this test pins
        // the last-line-of-defense at the schema level.)
        assert!(matches!(err, StorageError::Database(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn unique_edge_constraint_includes_to_kind(pool: PgPool) {
        // Same (from, relation, value) across different to_kind is allowed.
        let a = insert_test_thought(&pool, "A", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Entity("foo".into()),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        // Different to_kind (person) with same value — allowed.
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Person("foo".into()),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        let related = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(related.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_related_filters_by_target_kinds(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Url("https://x.io".into()),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        let url_only =
            fetch_related_thoughts(&pool, a, None, Some(&["url"]), LinkDirection::Outbound)
                .await
                .unwrap();
        assert_eq!(url_only.len(), 1);
        assert_eq!(url_only[0].target.kind_str(), "url");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn soft_delete_tagger_edges_for_thought_only_touches_tagger_source(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        let b = insert_test_thought(&pool, "B", "global").await;
        // One agent-supplied edge and one tagger-supplied edge from the same thought.
        insert_link(
            &pool,
            a,
            RelationKind::Refines,
            &t(b),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Url("https://example.com".into()),
            LinkSource::Tagger,
            None,
        )
        .await
        .unwrap();

        let n = soft_delete_tagger_edges_for_thought(&pool, a)
            .await
            .unwrap();
        assert_eq!(n, 1, "only the tagger edge should be soft-deleted");

        let live = fetch_related_thoughts(&pool, a, None, None, LinkDirection::Outbound)
            .await
            .unwrap();
        assert_eq!(live.len(), 1, "agent edge survives; tagger edge gone");
        assert_eq!(live[0].link_source, LinkSource::Agent);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn soft_delete_tagger_edges_for_thought_idempotent_on_already_deleted(pool: PgPool) {
        let a = insert_test_thought(&pool, "A", "global").await;
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Url("https://example.com".into()),
            LinkSource::Tagger,
            None,
        )
        .await
        .unwrap();

        let first = soft_delete_tagger_edges_for_thought(&pool, a)
            .await
            .unwrap();
        assert_eq!(first, 1);
        let second = soft_delete_tagger_edges_for_thought(&pool, a)
            .await
            .unwrap();
        assert_eq!(second, 0, "second call finds no live tagger edges");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn migration_audit_rows_present_for_0009_and_0010(pool: PgPool) {
        let rows = query_migration_audit(&pool, None, 100).await.unwrap();
        let names: Vec<&str> = rows.iter().map(|r| r.migration.as_str()).collect();
        assert!(
            names.iter().any(|n| n.starts_with("0009_")),
            "0009 audit row should be seeded"
        );
        assert!(
            names.iter().any(|n| n.starts_with("0010_")),
            "0010 audit row should be seeded"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn migration_audit_orders_by_ran_at_desc(pool: PgPool) {
        let rows = query_migration_audit(&pool, None, 100).await.unwrap();
        assert!(rows.len() >= 2);
        for pair in rows.windows(2) {
            assert!(pair[0].ran_at >= pair[1].ran_at, "expect descending order");
        }
    }

    // -- M6.0: corpus stats -----------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn corpus_stats_returns_aggregate_counts(pool: PgPool) {
        let a = insert_test_thought(&pool, "alpha bravo", "global").await;
        let b = insert_test_thought(&pool, "charlie delta echo", "global").await;
        // Retract one so the live/retracted split has both branches.
        retract_thought(&pool, b, Some("test")).await.unwrap();
        // Insert a tagger edge + an agent edge to test by_source split.
        insert_link(
            &pool,
            a,
            RelationKind::References,
            &LinkTarget::Url("https://example.com".into()),
            LinkSource::Tagger,
            None,
        )
        .await
        .unwrap();

        let stats = corpus_stats(&pool, None).await.unwrap();
        assert_eq!(stats.thoughts.live, 1);
        assert_eq!(stats.thoughts.retracted, 1);
        // Live thought's content is "alpha bravo" (11 bytes).
        assert!(stats.thoughts.content_bytes_total >= 11);
        assert_eq!(stats.links.live, 1);
        // One link from a tagger source.
        assert!(
            stats
                .links
                .by_source
                .iter()
                .any(|(s, n)| s == "tagger" && *n == 1)
        );
        assert!(
            stats
                .links
                .by_kind
                .iter()
                .any(|(k, n)| k == "url" && *n == 1)
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn corpus_stats_scope_prefix_filters_scopes_section_only(pool: PgPool) {
        insert_test_thought(&pool, "a", "rjf.work").await;
        insert_test_thought(&pool, "b", "rjf.personal").await;
        insert_test_thought(&pool, "c", "other.scope").await;

        let stats = corpus_stats(&pool, Some("rjf.")).await.unwrap();
        // Aggregate counts are corpus-global — all 3 thoughts.
        assert_eq!(stats.thoughts.live, 3);
        // But the scopes section is prefix-filtered.
        assert_eq!(stats.scopes.len(), 2);
        for s in &stats.scopes {
            assert!(s.scope.as_str().starts_with("rjf."));
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn corpus_stats_table_sizes_include_thoughts_and_embeddings(pool: PgPool) {
        let stats = corpus_stats(&pool, None).await.unwrap();
        let names: Vec<&str> = stats.storage.iter().map(|t| t.table.as_str()).collect();
        // These two tables always exist and always have non-zero index sizes
        // (btree pkey at minimum) even on an empty corpus.
        assert!(names.contains(&"thoughts"));
        assert!(names.contains(&"embeddings"));
        assert!(names.contains(&"thought_links"));
        // Database total is positive.
        assert!(stats.database_total_bytes > 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn corpus_stats_empty_corpus_returns_zeros(pool: PgPool) {
        let stats = corpus_stats(&pool, None).await.unwrap();
        assert_eq!(stats.thoughts.live, 0);
        assert_eq!(stats.thoughts.retracted, 0);
        assert_eq!(stats.thoughts.content_bytes_total, 0);
        assert_eq!(stats.thoughts.content_bytes_avg, 0);
        assert!(stats.embeddings.is_empty());
        assert_eq!(stats.links.live, 0);
        assert_eq!(stats.queues.pending_embeddings, 0);
        assert_eq!(stats.queues.pending_tags, 0);
        assert!(stats.scopes.is_empty());
        // pg_class still returns the table list even on an empty corpus.
        assert!(!stats.storage.is_empty());
    }

    // -- M5.x: scope discoverability (list_scopes + scope_prefix) -----------

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_returns_summary_with_counts_and_timestamps(pool: PgPool) {
        insert_test_thought(&pool, "a1", "work.tcgplayer").await;
        insert_test_thought(&pool, "a2", "work.tcgplayer").await;
        insert_test_thought(&pool, "b1", "project.kengram").await;

        let scopes = list_scopes(&pool, None).await.unwrap();
        assert_eq!(scopes.len(), 2);
        let by_scope: std::collections::HashMap<&str, &ScopeSummary> =
            scopes.iter().map(|s| (s.scope.as_str(), s)).collect();
        assert_eq!(by_scope.get("work.tcgplayer").unwrap().thought_count, 2);
        assert_eq!(by_scope.get("project.kengram").unwrap().thought_count, 1);
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

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_fts_scope_prefix_matches_multiple_scopes(pool: PgPool) {
        insert_test_thought(&pool, "uniquekeyword in rjf.a", "rjf.a").await;
        insert_test_thought(&pool, "uniquekeyword in rjf.b", "rjf.b").await;
        insert_test_thought(&pool, "uniquekeyword in other", "other").await;

        let hits = search_fts(&pool, "uniquekeyword", None, Some("rjf."), 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        let scopes: Vec<&str> = hits.iter().map(|h| h.thought.scope.as_str()).collect();
        assert!(scopes.iter().all(|s| s.starts_with("rjf.")));
        assert!(hits.iter().all(|h| h.lexical_score.is_some()));
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
    async fn retracted_thought_excluded_from_search_fts(pool: PgPool) {
        let _active = insert_test_thought(&pool, "uniquekeyword active", "global").await;
        let retracted = insert_test_thought(&pool, "uniquekeyword retracted", "global").await;
        retract_thought(&pool, retracted, None).await.unwrap();

        let hits = search_fts(&pool, "uniquekeyword", None, None, 10)
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

        let walk = find_untagged_or_stale_thoughts(&pool, 1, false, false, None, None, None, 100)
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

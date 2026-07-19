//! Typed callers for the migration-0030 database chokepoints.
//!
//! These queries intentionally use `sqlx::query` rather than `query!`: the
//! gate accepts pgvector's `vector(1024)` type and is introduced in the same
//! change, so there is no checked-in offline description until migration
//! integration runs.  Every value remains parameter-bound.

use crate::StorageError;
use pgvector::Vector;
use sqlx::{PgPool, Row};
use time::OffsetDateTime;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct GatedCaptureRequest<'a> {
    pub scope: &'a str,
    pub content: &'a str,
    pub source: &'a str,
    pub metadata: &'a serde_json::Value,
    pub source_created_at: Option<OffsetDateTime>,
    pub candidate_embedding: Option<&'a [f32]>,
    pub embedding_model_id: Option<&'a str>,
    pub embedding_model_version: Option<i32>,
    pub bypass_reason: Option<&'a serde_json::Value>,
    pub source_event_namespace: Option<&'a str>,
    pub source_event_ref: Option<&'a str>,
    pub source_event_payload_hash: Option<&'a str>,
    pub source_event_metadata: Option<&'a serde_json::Value>,
    pub relation_intents: &'a serde_json::Value,
    pub tagger_model_id: Option<&'a str>,
    pub claimed_producer_class: Option<&'a str>,
    pub correlation_id: Option<&'a str>,
    pub force_keep_token: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct GatedCaptureResult {
    pub thought_id: Option<Uuid>,
    pub action: String,
    pub matched_thought_id: Option<Uuid>,
    pub similarity: Option<f64>,
    pub threshold: f64,
    pub effective_created_at: OffsetDateTime,
    pub observed_at: OffsetDateTime,
    pub source_event_status: Option<String>,
    pub source_event_action: Option<String>,
    pub relation_results: serde_json::Value,
    pub gate_event_id: Option<Uuid>,
}

pub async fn capture_thought_gated(
    pool: &PgPool,
    request: GatedCaptureRequest<'_>,
) -> Result<GatedCaptureResult, StorageError> {
    let vector = request
        .candidate_embedding
        .map(|values| Vector::from(values.to_vec()));
    let row = sqlx::query(
        r#"
        SELECT *
        FROM public.capture_thought_gated(
            $1, $2, $3, $4, $5, $6::vector, $7, $8, $9,
            $10, $11, $12, $13, $14, $15, $16, $17, $18
        )
        "#,
    )
    .bind(request.scope)
    .bind(request.content)
    .bind(request.source)
    .bind(request.metadata)
    .bind(request.source_created_at)
    .bind(vector)
    .bind(request.embedding_model_id)
    .bind(request.embedding_model_version)
    .bind(request.bypass_reason)
    .bind(request.source_event_namespace)
    .bind(request.source_event_ref)
    .bind(request.source_event_payload_hash)
    .bind(request.source_event_metadata)
    .bind(request.relation_intents)
    .bind(request.tagger_model_id)
    .bind(request.claimed_producer_class)
    .bind(request.correlation_id)
    .bind(request.force_keep_token)
    .fetch_one(pool)
    .await?;

    Ok(GatedCaptureResult {
        thought_id: row.try_get("thought_id")?,
        action: row.try_get("action")?,
        matched_thought_id: row.try_get("matched_thought_id")?,
        similarity: row.try_get("similarity")?,
        threshold: row.try_get("threshold")?,
        effective_created_at: row.try_get("effective_created_at")?,
        observed_at: row.try_get("observed_at")?,
        source_event_status: row.try_get("source_event_status")?,
        source_event_action: row.try_get("source_event_action")?,
        relation_results: row.try_get("relation_results")?,
        gate_event_id: row.try_get("gate_event_id")?,
    })
}

#[derive(Debug, Clone)]
pub struct RelationMutationRequest<'a> {
    pub operations: &'a serde_json::Value,
    pub source_event_namespace: &'a str,
    pub source_event_ref: &'a str,
    pub source_event_payload_hash: &'a str,
    pub request_metadata: &'a serde_json::Value,
    pub claimed_producer_class: Option<&'a str>,
}

pub async fn mutate_thought_relations_serialized(
    pool: &PgPool,
    request: RelationMutationRequest<'_>,
) -> Result<serde_json::Value, StorageError> {
    let row = sqlx::query_scalar::<_, serde_json::Value>(
        r#"
        SELECT public.mutate_thought_relations_serialized($1, $2, $3, $4, $5, $6)
        "#,
    )
    .bind(request.operations)
    .bind(request.source_event_namespace)
    .bind(request.source_event_ref)
    .bind(request.source_event_payload_hash)
    .bind(request.request_metadata)
    .bind(request.claimed_producer_class)
    .fetch_one(pool)
    .await?;
    Ok(row)
}

pub async fn retract_thought_serialized(
    pool: &PgPool,
    thought_id: Uuid,
    reason: Option<&str>,
    claimed_producer_class: Option<&str>,
) -> Result<serde_json::Value, StorageError> {
    let result = sqlx::query_scalar::<_, serde_json::Value>(
        "SELECT public.retract_thought_serialized($1, $2, $3)",
    )
    .bind(thought_id)
    .bind(reason)
    .bind(claimed_producer_class)
    .fetch_one(pool)
    .await?;
    Ok(result)
}

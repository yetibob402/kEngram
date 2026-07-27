//! Typed callers for the migration-0030 database chokepoints.
//!
//! These queries intentionally use `sqlx::query` rather than `query!`: the
//! gate accepts pgvector's `vector(1024)` type and is introduced in the same
//! change, so there is no checked-in offline description until migration
//! integration runs.  Every value remains parameter-bound.
//!
//! Capture timeout contract (ambiguous-outcome round 2): statement work runs
//! under a 400ms PostgreSQL statement_timeout; a protected 500ms
//! commit-and-response reserve is owned by the MCP layer. Test-only injectors
//! enable deterministic pre-commit holds and commit-ack failures without
//! migrations or `capture.rs` edits.

use crate::StorageError;
use pgvector::Vector;
use sqlx::{PgPool, Row};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;
use time::OffsetDateTime;
use uuid::Uuid;

/// Serializes tests that mutate process-global capture timeout injectors.
static CAPTURE_TIMEOUT_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquire exclusive access for injector-backed capture-timeout tests.
pub fn test_lock_capture_timeout_injectors() -> MutexGuard<'static, ()> {
    CAPTURE_TIMEOUT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Leaves the MCP caller enough of its one-second end-to-end budget to commit
/// and serialize a structured fail-open result after a comparison timeout.
pub const CAPTURE_GATE_STATEMENT_TIMEOUT: &str = "400ms";

/// Statement-phase budget mirrored for MCP independent phase accounting.
pub const CAPTURE_STATEMENT_BUDGET: Duration = Duration::from_millis(400);

/// Protected commit-and-response reserve owned by the MCP total deadline.
pub const CAPTURE_PROTECTED_RESERVE: Duration = Duration::from_millis(500);

/// Test-only: sleep this many milliseconds after the gate query, before commit.
static TEST_HOLD_BEFORE_COMMIT_MS: AtomicU64 = AtomicU64::new(0);
/// Test-only: force commit acknowledgement failure after a successful gate body.
static TEST_FAIL_COMMIT: AtomicBool = AtomicBool::new(false);
/// Test-only mutant: collapse statement-cancel into a plain final-looking path
/// marker so RED tests can prove classification must stay on the unknown path.
static TEST_MUTANT_CANCEL_AS_FINAL: AtomicBool = AtomicBool::new(false);
/// Test-only mutant: treat non-key Argus fields as identity (server-layer uses
/// this flag when proving RED; storage remains identity-correct).
static TEST_MUTANT_NONKEY_AS_IDENTITY: AtomicBool = AtomicBool::new(false);
/// Test-only: when set, raw correlation canary is echoed into an error string
/// (must stay off in production paths; RED tests enable briefly).
static TEST_MUTANT_LEAK_CORRELATION: AtomicBool = AtomicBool::new(false);

/// Clear all capture-timeout test injectors/mutants.
pub fn test_clear_capture_timeout_injectors() {
    TEST_HOLD_BEFORE_COMMIT_MS.store(0, Ordering::SeqCst);
    TEST_FAIL_COMMIT.store(false, Ordering::SeqCst);
    TEST_MUTANT_CANCEL_AS_FINAL.store(false, Ordering::SeqCst);
    TEST_MUTANT_NONKEY_AS_IDENTITY.store(false, Ordering::SeqCst);
    TEST_MUTANT_LEAK_CORRELATION.store(false, Ordering::SeqCst);
}

pub fn test_set_hold_before_commit_ms(ms: u64) {
    TEST_HOLD_BEFORE_COMMIT_MS.store(ms, Ordering::SeqCst);
}

pub fn test_set_fail_commit(fail: bool) {
    TEST_FAIL_COMMIT.store(fail, Ordering::SeqCst);
}

pub fn test_set_mutant_cancel_as_final(on: bool) {
    TEST_MUTANT_CANCEL_AS_FINAL.store(on, Ordering::SeqCst);
}

pub fn test_set_mutant_nonkey_as_identity(on: bool) {
    TEST_MUTANT_NONKEY_AS_IDENTITY.store(on, Ordering::SeqCst);
}

pub fn test_set_mutant_leak_correlation(on: bool) {
    TEST_MUTANT_LEAK_CORRELATION.store(on, Ordering::SeqCst);
}

pub fn test_mutant_nonkey_as_identity() -> bool {
    TEST_MUTANT_NONKEY_AS_IDENTITY.load(Ordering::SeqCst)
}

pub fn test_mutant_leak_correlation() -> bool {
    TEST_MUTANT_LEAK_CORRELATION.load(Ordering::SeqCst)
}

/// True when a sqlx/database error is PostgreSQL statement cancellation
/// (SQLSTATE 57014) or an equivalent statement-timeout surface.
pub fn is_statement_canceled(err: &sqlx::Error) -> bool {
    match err {
        sqlx::Error::Database(db) => {
            if db.code().as_deref() == Some("57014") {
                return true;
            }
            let msg = db.message().to_ascii_lowercase();
            msg.contains("canceling statement")
                || msg.contains("statement timeout")
                || msg.contains("query_canceled")
        }
        other => {
            let msg = other.to_string().to_ascii_lowercase();
            msg.contains("57014")
                || msg.contains("canceling statement")
                || msg.contains("statement timeout")
                || msg.contains("query_canceled")
        }
    }
}

/// True when an error indicates commit acknowledgement / transaction end
/// uncertainty rather than a clean pre-write validation failure.
pub fn is_commit_uncertainty(err: &sqlx::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("commit")
        || msg.contains("connection closed")
        || msg.contains("server closed the connection")
        || msg.contains("error during commit")
        || msg.contains("capture_commit_ack_failed")
}

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
    // pool checkout + begin (phase: pool) — must complete before statement config.
    let mut tx = pool.begin().await?;
    // Statement-timeout configuration is local to the transaction.
    sqlx::query("SELECT set_config('statement_timeout', $1, true)")
        .bind(CAPTURE_GATE_STATEMENT_TIMEOUT)
        .execute(&mut *tx)
        .await?;

    let row_result = sqlx::query(
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
    .fetch_one(&mut *tx)
    .await;

    let row = match row_result {
        Ok(row) => row,
        Err(err) if is_statement_canceled(&err) => {
            // Drop the transaction (rollback) — commit outcome is not confirmed
            // as a write success for this attempt.
            drop(tx);
            if TEST_MUTANT_CANCEL_AS_FINAL.load(Ordering::SeqCst) {
                // Mutant surface: look like the old final storage string path.
                return Err(StorageError::Database(sqlx::Error::Protocol(
                    "internal database error during capture".into(),
                )));
            }
            return Err(StorageError::Database(sqlx::Error::Protocol(format!(
                "capture_statement_canceled: {err}"
            ))));
        }
        Err(err) => return Err(StorageError::Database(err)),
    };

    let result = GatedCaptureResult {
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
    };

    let hold_ms = TEST_HOLD_BEFORE_COMMIT_MS.load(Ordering::SeqCst);
    if hold_ms > 0 {
        // Non-fail-open critical section before commit (gate body already ran).
        tokio::time::sleep(Duration::from_millis(hold_ms)).await;
    }

    if TEST_FAIL_COMMIT.load(Ordering::SeqCst) {
        // Prove commit-ack failure path without relying on a real I/O fault.
        drop(tx);
        return Err(StorageError::Database(sqlx::Error::Protocol(
            "capture_commit_ack_failed: injected commit acknowledgement failure".into(),
        )));
    }

    if let Err(err) = tx.commit().await {
        if is_commit_uncertainty(&err) || is_statement_canceled(&err) {
            return Err(StorageError::Database(sqlx::Error::Protocol(format!(
                "capture_commit_ack_failed: {err}"
            ))));
        }
        return Err(StorageError::Database(err));
    }
    Ok(result)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_timeout_budget_constants_protect_reserve() {
        assert_eq!(CAPTURE_STATEMENT_BUDGET, Duration::from_millis(400));
        assert_eq!(CAPTURE_PROTECTED_RESERVE, Duration::from_millis(500));
        assert_eq!(CAPTURE_GATE_STATEMENT_TIMEOUT, "400ms");
    }

    #[test]
    fn is_statement_canceled_detects_57014_protocol_text() {
        let err = sqlx::Error::Protocol(
            "ERROR: canceling statement due to statement timeout (SQLSTATE 57014)".into(),
        );
        assert!(is_statement_canceled(&err));
        let plain = sqlx::Error::Protocol("unique violation".into());
        assert!(!is_statement_canceled(&plain));
    }

    #[test]
    fn is_commit_uncertainty_detects_injected_marker() {
        let err = sqlx::Error::Protocol("capture_commit_ack_failed: boom".into());
        assert!(is_commit_uncertainty(&err));
    }

    #[test]
    fn injectors_default_off_and_clear() {
        test_clear_capture_timeout_injectors();
        assert_eq!(TEST_HOLD_BEFORE_COMMIT_MS.load(Ordering::SeqCst), 0);
        assert!(!TEST_FAIL_COMMIT.load(Ordering::SeqCst));
        assert!(!test_mutant_nonkey_as_identity());
        assert!(!test_mutant_leak_correlation());
        test_set_hold_before_commit_ms(10);
        test_set_fail_commit(true);
        test_set_mutant_nonkey_as_identity(true);
        test_set_mutant_leak_correlation(true);
        assert_eq!(TEST_HOLD_BEFORE_COMMIT_MS.load(Ordering::SeqCst), 10);
        test_clear_capture_timeout_injectors();
        assert_eq!(TEST_HOLD_BEFORE_COMMIT_MS.load(Ordering::SeqCst), 0);
        assert!(!TEST_FAIL_COMMIT.load(Ordering::SeqCst));
    }
}

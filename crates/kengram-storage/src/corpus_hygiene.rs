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
/// Test-only: inject statement cancel via short timeout + pg_sleep before gate body.
static TEST_INJECT_STATEMENT_CANCEL: AtomicBool = AtomicBool::new(false);
/// Test-only: delay pool/begin phase by this many ms (for reserve-spend mutants).
static TEST_POOL_DELAY_MS: AtomicU64 = AtomicU64::new(0);
/// Test-only: delay commit phase by this many ms (fits inside reserve for positive path).
static TEST_COMMIT_DELAY_MS: AtomicU64 = AtomicU64::new(0);
/// Active production phase budgets installed by MCP for this call (ms). 0 = use defaults.
static ACTIVE_POOL_BUDGET_MS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMMIT_BUDGET_MS: AtomicU64 = AtomicU64::new(0);
/// Mutant: allow pool phase to use a budget that invades the protected reserve.
static TEST_MUTANT_POOL_SPENDS_RESERVE: AtomicBool = AtomicBool::new(false);
/// When false, all test injectors are ignored (prevents parallel-test leakage).
static TEST_INJECTORS_ARMED: AtomicBool = AtomicBool::new(false);

/// Install per-call phase budgets (milliseconds). Pass 0 to clear to defaults.
pub fn install_phase_budgets(pool_ms: u64, commit_ms: u64) {
    ACTIVE_POOL_BUDGET_MS.store(pool_ms, Ordering::SeqCst);
    ACTIVE_COMMIT_BUDGET_MS.store(commit_ms, Ordering::SeqCst);
}

pub fn clear_phase_budgets() {
    ACTIVE_POOL_BUDGET_MS.store(0, Ordering::SeqCst);
    ACTIVE_COMMIT_BUDGET_MS.store(0, Ordering::SeqCst);
}

/// Clear all capture-timeout test injectors/mutants.
pub fn test_arm_capture_timeout_injectors() {
    TEST_INJECTORS_ARMED.store(true, Ordering::SeqCst);
}

fn injectors_armed() -> bool {
    TEST_INJECTORS_ARMED.load(Ordering::SeqCst)
}

pub fn test_clear_capture_timeout_injectors() {
    TEST_INJECTORS_ARMED.store(false, Ordering::SeqCst);
    TEST_HOLD_BEFORE_COMMIT_MS.store(0, Ordering::SeqCst);
    TEST_FAIL_COMMIT.store(false, Ordering::SeqCst);
    TEST_MUTANT_CANCEL_AS_FINAL.store(false, Ordering::SeqCst);
    TEST_MUTANT_NONKEY_AS_IDENTITY.store(false, Ordering::SeqCst);
    TEST_MUTANT_LEAK_CORRELATION.store(false, Ordering::SeqCst);
    TEST_INJECT_STATEMENT_CANCEL.store(false, Ordering::SeqCst);
    TEST_POOL_DELAY_MS.store(0, Ordering::SeqCst);
    TEST_COMMIT_DELAY_MS.store(0, Ordering::SeqCst);
    TEST_MUTANT_POOL_SPENDS_RESERVE.store(false, Ordering::SeqCst);
    clear_phase_budgets();
}

pub fn test_set_hold_before_commit_ms(ms: u64) {
    test_arm_capture_timeout_injectors();
    TEST_HOLD_BEFORE_COMMIT_MS.store(ms, Ordering::SeqCst);
}

pub fn test_set_fail_commit(fail: bool) {
    test_arm_capture_timeout_injectors();
    TEST_FAIL_COMMIT.store(fail, Ordering::SeqCst);
}

pub fn test_set_mutant_cancel_as_final(on: bool) {
    test_arm_capture_timeout_injectors();
    TEST_MUTANT_CANCEL_AS_FINAL.store(on, Ordering::SeqCst);
}

pub fn test_set_mutant_nonkey_as_identity(on: bool) {
    test_arm_capture_timeout_injectors();
    TEST_MUTANT_NONKEY_AS_IDENTITY.store(on, Ordering::SeqCst);
}

pub fn test_set_mutant_leak_correlation(on: bool) {
    test_arm_capture_timeout_injectors();
    TEST_MUTANT_LEAK_CORRELATION.store(on, Ordering::SeqCst);
}

pub fn test_set_inject_statement_cancel(on: bool) {
    test_arm_capture_timeout_injectors();
    TEST_INJECT_STATEMENT_CANCEL.store(on, Ordering::SeqCst);
}

pub fn test_set_pool_delay_ms(ms: u64) {
    test_arm_capture_timeout_injectors();
    TEST_POOL_DELAY_MS.store(ms, Ordering::SeqCst);
}

pub fn test_set_commit_delay_ms(ms: u64) {
    test_arm_capture_timeout_injectors();
    TEST_COMMIT_DELAY_MS.store(ms, Ordering::SeqCst);
}

pub fn test_set_mutant_pool_spends_reserve(on: bool) {
    test_arm_capture_timeout_injectors();
    TEST_MUTANT_POOL_SPENDS_RESERVE.store(on, Ordering::SeqCst);
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

    // Phase budgets: MCP installs per-call values; defaults preserve statement=400ms
    // and commit inside the 500ms protected reserve.
    let mut pool_budget = {
        let ms = ACTIVE_POOL_BUDGET_MS.load(Ordering::SeqCst);
        if ms == 0 {
            Duration::from_millis(100)
        } else {
            Duration::from_millis(ms)
        }
    };
    if injectors_armed() && TEST_MUTANT_POOL_SPENDS_RESERVE.load(Ordering::SeqCst) {
        // Mutant: inflate pool budget so it can spend into the protected reserve.
        pool_budget = CAPTURE_PROTECTED_RESERVE + Duration::from_millis(200);
    }
    let commit_budget = {
        let ms = ACTIVE_COMMIT_BUDGET_MS.load(Ordering::SeqCst);
        if ms == 0 {
            Duration::from_millis(400)
        } else {
            Duration::from_millis(ms)
        }
    };

    // --- pool / begin / config phase (must not use protected reserve in production) ---
    let pool_phase = async {
        let delay = if injectors_armed() {
            TEST_POOL_DELAY_MS.load(Ordering::SeqCst)
        } else {
            0
        };
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        let mut tx = pool.begin().await.map_err(StorageError::Database)?;
        sqlx::query("SELECT set_config('statement_timeout', $1, true)")
            .bind(CAPTURE_GATE_STATEMENT_TIMEOUT)
            .execute(&mut *tx)
            .await
            .map_err(StorageError::Database)?;
        Ok::<_, StorageError>(tx)
    };

    let mut tx = match tokio::time::timeout(pool_budget, pool_phase).await {
        Ok(Ok(tx)) => tx,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(StorageError::Database(sqlx::Error::Protocol(
                "capture_pool_timeout: pool/begin/config exceeded owned budget".into(),
            )));
        }
    };

    // Optional statement-cancel injector (outside similarity fail-open).
    if injectors_armed() && TEST_INJECT_STATEMENT_CANCEL.load(Ordering::SeqCst) {
        // Force a cancelable wait under the transaction-local statement_timeout.
        let cancel_result = sqlx::query("SELECT pg_sleep(2)").execute(&mut *tx).await;
        if let Err(err) = cancel_result {
            drop(tx);
            if is_statement_canceled(&err) || err.to_string().contains("cancel") {
                if injectors_armed() && TEST_MUTANT_CANCEL_AS_FINAL.load(Ordering::SeqCst) {
                    return Err(StorageError::Database(sqlx::Error::Protocol(
                        "internal database error during capture".into(),
                    )));
                }
                return Err(StorageError::Database(sqlx::Error::Protocol(format!(
                    "capture_statement_canceled: {err}"
                ))));
            }
            return Err(StorageError::Database(err));
        }
    }

    // --- statement / gate body phase (independently bounded by statement_timeout) ---
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
            drop(tx);
            if injectors_armed() && TEST_MUTANT_CANCEL_AS_FINAL.load(Ordering::SeqCst) {
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

    let hold_ms = if injectors_armed() {
        TEST_HOLD_BEFORE_COMMIT_MS.load(Ordering::SeqCst)
    } else {
        0
    };
    if hold_ms > 0 {
        tokio::time::sleep(Duration::from_millis(hold_ms)).await;
    }

    // --- commit phase (owned budget inside protected reserve) ---
    let commit_phase = async {
        let delay = if injectors_armed() {
            TEST_COMMIT_DELAY_MS.load(Ordering::SeqCst)
        } else {
            0
        };
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        if injectors_armed() && TEST_FAIL_COMMIT.load(Ordering::SeqCst) {
            drop(tx);
            return Err(StorageError::Database(sqlx::Error::Protocol(
                "capture_commit_ack_failed: injected commit acknowledgement failure".into(),
            )));
        }
        tx.commit().await.map_err(|err| {
            if is_commit_uncertainty(&err) || is_statement_canceled(&err) {
                StorageError::Database(sqlx::Error::Protocol(format!(
                    "capture_commit_ack_failed: {err}"
                )))
            } else {
                StorageError::Database(err)
            }
        })
    };

    match tokio::time::timeout(commit_budget, commit_phase).await {
        Ok(Ok(())) => Ok(result),
        Ok(Err(e)) => Err(e),
        Err(_) => {
            // Commit outcome unconfirmed after budget; transaction dropped with future.
            Err(StorageError::Database(sqlx::Error::Protocol(
                "capture_commit_ack_failed: commit exceeded owned budget".into(),
            )))
        }
    }
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
        // Arithmetic: embedding 500 + pool 100 + statement 400 + reserve 500 = 1500.
        let embed = Duration::from_millis(500);
        let pool = Duration::from_millis(100);
        let total_min = embed + pool + CAPTURE_STATEMENT_BUDGET + CAPTURE_PROTECTED_RESERVE;
        assert_eq!(total_min, Duration::from_millis(1500));
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

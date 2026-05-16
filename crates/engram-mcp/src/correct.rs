//! `correct_fact` — the operator-driven path that turns a wrong extraction
//! into a corrected one (or just retracts it). Per m2-facts-pipeline.md Q10,
//! manual rows carry the sentinel provenance `extractor_model = "manual"`,
//! `extractor_version = 0`, so a single query like "facts not produced by
//! the current extractor" works across machine- and human-authored facts.
//!
//! The replacement-and-supersede pair runs inside a transaction so a crash
//! between the two writes can't leave the old fact superseded but pointing
//! at no new row.

use engram_core::Scope;
use engram_storage::{NewFact, supersede_fact};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct CorrectFactRequest {
    pub fact_id: Uuid,
    pub replacement: Option<FactReplacement>,
}

#[derive(Debug, Clone)]
pub struct FactReplacement {
    pub statement: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorrectFactResponse {
    pub superseded: bool,
    pub new_fact_id: Option<Uuid>,
}

#[derive(Debug, thiserror::Error)]
pub enum CorrectError {
    #[error("fact not found or already superseded: {0}")]
    AlreadySupersededOrMissing(Uuid),

    #[error("replacement statement must not be empty")]
    EmptyReplacementStatement,

    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// Sentinel provenance values for manually-authored facts.
pub const MANUAL_EXTRACTOR_MODEL: &str = "manual";
pub const MANUAL_EXTRACTOR_VERSION: i32 = 0;

/// Supersede an existing fact, optionally inserting a replacement row.
///
/// - With `replacement: Some(_)`: insert a new fact (with manual-sentinel
///   provenance) at the same `source_thought_id` and `scope` as the old
///   one, then mark the old row superseded with `superseded_by` pointing at
///   the new row. Both writes happen in a tx.
/// - With `replacement: None`: mark the old row superseded with no
///   replacement. This is the "delete-by-supersede" path.
pub async fn correct_fact(
    pool: &PgPool,
    request: CorrectFactRequest,
) -> Result<CorrectFactResponse, CorrectError> {
    let old = engram_storage::fetch_fact(pool, request.fact_id)
        .await?
        .ok_or(CorrectError::AlreadySupersededOrMissing(request.fact_id))?;

    if let Some(replacement) = request.replacement {
        if replacement.statement.trim().is_empty() {
            return Err(CorrectError::EmptyReplacementStatement);
        }

        // Insert + supersede in one transaction. The supersede uses the
        // partial-WHERE shape that no-ops on already-superseded rows, so a
        // concurrent supersede loses cleanly and we surface the right error.
        let mut tx = pool.begin().await.map_err(engram_storage::StorageError::from)?;

        // Reuse the engram_storage helpers but run them against the tx
        // executor. The fns take &PgPool; we need to issue the SQL against
        // &mut *tx instead. Inline the two queries here to keep the tx
        // boundary honest.
        let new_id: Uuid = sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version,
                source_run_id, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULL, 1.0)
            RETURNING id
            "#,
            old.scope.as_str(),
            replacement.statement,
            replacement.subject,
            replacement.predicate,
            replacement.object,
            old.source_thought_id.into_uuid(),
            MANUAL_EXTRACTOR_MODEL,
            MANUAL_EXTRACTOR_VERSION,
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(engram_storage::StorageError::from)?;

        let result = sqlx::query!(
            r#"
            UPDATE facts
            SET superseded_by = $2, superseded_at = NOW()
            WHERE id = $1 AND superseded_at IS NULL
            "#,
            request.fact_id,
            new_id,
        )
        .execute(&mut *tx)
        .await
        .map_err(engram_storage::StorageError::from)?;

        if result.rows_affected() != 1 {
            // Lost a race to a concurrent supersede; roll back the new fact.
            tx.rollback()
                .await
                .map_err(engram_storage::StorageError::from)?;
            return Err(CorrectError::AlreadySupersededOrMissing(request.fact_id));
        }

        tx.commit().await.map_err(engram_storage::StorageError::from)?;
        // Silence unused-import warnings (`NewFact`, `Scope` are kept for
        // doc clarity — the manual-sentinel pattern *would* use NewFact in a
        // non-tx path).
        let _: Option<NewFact> = None;
        let _: Option<&Scope> = None;
        Ok(CorrectFactResponse {
            superseded: true,
            new_fact_id: Some(new_id),
        })
    } else {
        let did = supersede_fact(pool, request.fact_id, None).await?;
        if !did {
            return Err(CorrectError::AlreadySupersededOrMissing(request.fact_id));
        }
        Ok(CorrectFactResponse {
            superseded: true,
            new_fact_id: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{capture, CaptureRequest};
    use engram_core::{Source, ThoughtId};

    const TEST_MODEL_ID: &str = "bge-m3:1024";

    async fn cap(pool: &PgPool, content: &str) -> ThoughtId {
        capture(
            pool,
            TEST_MODEL_ID,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new("global").unwrap()),
                metadata: None,
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    async fn insert_one_fact(pool: &PgPool, thought_id: ThoughtId, statement: &str) -> Uuid {
        let scope = Scope::global();
        let run_id = engram_storage::start_run(pool, "fake/extractor", 1, None)
            .await
            .unwrap();
        engram_storage::insert_fact(
            pool,
            NewFact {
                scope: &scope,
                statement,
                subject: None,
                predicate: None,
                object: None,
                source_thought_id: thought_id,
                extractor_model: "fake/extractor",
                extractor_version: 1,
                source_run_id: Some(run_id),
                confidence: 0.9,
                flagged: false,
            },
        )
        .await
        .unwrap()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_with_replacement_inserts_new_and_supersedes_old(pool: PgPool) {
        let thought_id = cap(&pool, "anchor").await;
        let old_id = insert_one_fact(&pool, thought_id, "old wording").await;

        let resp = correct_fact(
            &pool,
            CorrectFactRequest {
                fact_id: old_id,
                replacement: Some(FactReplacement {
                    statement: "new wording".to_string(),
                    subject: Some("S".to_string()),
                    predicate: Some("P".to_string()),
                    object: Some("O".to_string()),
                }),
            },
        )
        .await
        .unwrap();
        assert!(resp.superseded);
        let new_id = resp.new_fact_id.expect("must have new fact id");

        // Old row is superseded; superseded_by points at new_id.
        let old_row = sqlx::query!(
            r#"SELECT superseded_by, superseded_at FROM facts WHERE id = $1"#,
            old_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(old_row.superseded_by, Some(new_id));
        assert!(old_row.superseded_at.is_some());

        // New row exists with correct shape.
        let new_row = sqlx::query!(
            r#"SELECT statement, subject, predicate, object, source_thought_id,
                      extractor_model, extractor_version, source_run_id, confidence
               FROM facts WHERE id = $1"#,
            new_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(new_row.statement, "new wording");
        assert_eq!(new_row.subject.as_deref(), Some("S"));
        assert_eq!(new_row.source_thought_id, Some(thought_id.into_uuid()));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_uses_manual_sentinel_provenance(pool: PgPool) {
        let thought_id = cap(&pool, "anchor").await;
        let old_id = insert_one_fact(&pool, thought_id, "old").await;

        let resp = correct_fact(
            &pool,
            CorrectFactRequest {
                fact_id: old_id,
                replacement: Some(FactReplacement {
                    statement: "corrected".to_string(),
                    subject: None,
                    predicate: None,
                    object: None,
                }),
            },
        )
        .await
        .unwrap();
        let new_id = resp.new_fact_id.unwrap();

        let row = sqlx::query!(
            r#"SELECT extractor_model, extractor_version, source_run_id, confidence
               FROM facts WHERE id = $1"#,
            new_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.extractor_model, MANUAL_EXTRACTOR_MODEL);
        assert_eq!(row.extractor_version, MANUAL_EXTRACTOR_VERSION);
        assert!(row.source_run_id.is_none(), "manual rows have no source run");
        assert!((row.confidence - 1.0).abs() < 1e-5, "operator authority = 1.0");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_without_replacement_supersedes_only(pool: PgPool) {
        let thought_id = cap(&pool, "anchor").await;
        let old_id = insert_one_fact(&pool, thought_id, "doomed").await;

        let resp = correct_fact(
            &pool,
            CorrectFactRequest {
                fact_id: old_id,
                replacement: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.superseded);
        assert!(resp.new_fact_id.is_none());

        let row = sqlx::query!(
            r#"SELECT superseded_by, superseded_at FROM facts WHERE id = $1"#,
            old_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.superseded_by.is_none());
        assert!(row.superseded_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_errors_on_unknown_fact_id(pool: PgPool) {
        let unknown = Uuid::new_v4();
        let err = correct_fact(
            &pool,
            CorrectFactRequest {
                fact_id: unknown,
                replacement: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CorrectError::AlreadySupersededOrMissing(id) if id == unknown));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_errors_on_already_superseded_fact(pool: PgPool) {
        let thought_id = cap(&pool, "anchor").await;
        let old_id = insert_one_fact(&pool, thought_id, "old").await;
        engram_storage::supersede_fact(&pool, old_id, None).await.unwrap();

        let err = correct_fact(
            &pool,
            CorrectFactRequest {
                fact_id: old_id,
                replacement: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CorrectError::AlreadySupersededOrMissing(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_rejects_empty_replacement_statement(pool: PgPool) {
        let thought_id = cap(&pool, "anchor").await;
        let old_id = insert_one_fact(&pool, thought_id, "old").await;

        let err = correct_fact(
            &pool,
            CorrectFactRequest {
                fact_id: old_id,
                replacement: Some(FactReplacement {
                    statement: "   ".to_string(),
                    subject: None,
                    predicate: None,
                    object: None,
                }),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CorrectError::EmptyReplacementStatement));
    }
}

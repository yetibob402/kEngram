//! `run_reflector_once` — single reflector pass driven by `tokio-cron-scheduler`
//! in the worker. Walks the LEFT-JOIN-IS-NULL unfacted-thoughts set, calls
//! the extractor per thought, routes each resulting fact to either `facts`
//! (committed) or `facts_review_queue` based on a configurable confidence
//! threshold. Per-thought extractor failures are soft (logged + counted +
//! continue); the thought re-appears in the next tick's unfacted set.
//!
//! Mirrors the `drain.rs` shape: a pure function over `&PgPool` + `&dyn
//! Extractor` + options, returning a `ReflectorReport`. The cron loop in
//! `engram-cli` wraps this call.

use engram_core::{
    ExtractMode, ExtractedFact, ExtractionContext, Extractor, Fact, Metadata, Thought,
};
use engram_storage::{NewFact, NewReviewRow, RunId};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use time::OffsetDateTime;

/// Operator-tunable knobs for a reflector run. Deserialized straight from
/// `[reflector]` in `engram.toml`.
///
/// `enabled` is `false` by default: starting `engram worker` without an
/// `[extractor]` config or a running vLLM should be a no-op for the
/// reflector. The operator flips this to `true` once vLLM is up.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReflectorOptions {
    pub enabled: bool,
    /// Cron expression. `tokio-cron-scheduler` accepts 6-field cron
    /// (sec min hour dom month dow). Default is 03:00 every day.
    pub schedule: String,
    pub scope_filter: Option<String>,
    pub max_thoughts_per_run: i64,
    pub max_facts_per_thought: usize,
    /// Confidence below this threshold routes the fact to
    /// `facts_review_queue` for operator review. At-or-above commits to
    /// `facts`. Single-band routing in Phase C; m2-facts-pipeline.md's
    /// three-band design (with a `flagged` column on `facts`) is deferred.
    pub review_queue_below: f32,
}

impl Default for ReflectorOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            schedule: "0 0 3 * * *".to_string(),
            scope_filter: None,
            max_thoughts_per_run: 1000,
            max_facts_per_thought: 8,
            review_queue_below: 0.7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReflectorReport {
    pub run_id: RunId,
    pub n_thoughts_processed: i32,
    pub n_facts_committed: i32,
    pub n_review_queue: i32,
    pub n_extractor_failures: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum ReflectorError {
    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

/// One reflector pass. Opens a `reflector_runs` row, walks unfacted
/// thoughts, extracts + routes, closes the run with final counts.
pub async fn run_reflector_once(
    pool: &PgPool,
    extractor: &dyn Extractor,
    embedder_model_id: &str,
    options: &ReflectorOptions,
) -> Result<ReflectorReport, ReflectorError> {
    let run_id = engram_storage::start_run(
        pool,
        extractor.model_id(),
        extractor.version(),
        options.scope_filter.as_deref(),
    )
    .await?;

    let thoughts = engram_storage::find_unfacted_thoughts(
        pool,
        options.scope_filter.as_deref(),
        options.max_thoughts_per_run,
    )
    .await?;

    let mut n_processed: i32 = 0;
    let mut n_committed: i32 = 0;
    let mut n_review: i32 = 0;
    let mut n_failures: i32 = 0;

    for thought in &thoughts {
        let extract_mode = match extract_directive(&thought.metadata) {
            ExtractDirective::Skip => {
                // `metadata.extract: "none"` — operator-flagged skip. Still
                // counted as processed so reflector_runs accounting reflects
                // that we considered (and skipped) the thought.
                n_processed += 1;
                continue;
            }
            ExtractDirective::Run(mode) => mode,
        };
        let ctx = ExtractionContext::new(thought.scope.clone(), options.max_facts_per_thought)
            .with_extract_mode(extract_mode);
        let facts = match extractor.extract(thought, &ctx).await {
            Ok(facts) => facts,
            Err(err) => {
                // Per Q9: per-thought soft-fail. The unfacted thought
                // remains in the next tick's LEFT-JOIN-IS-NULL set.
                tracing::warn!(
                    run_id = %run_id,
                    thought_id = %thought.id,
                    error = %err,
                    transient = err.is_transient(),
                    "reflector: extractor failed; thought skipped this run",
                );
                n_failures += 1;
                n_processed += 1;
                continue;
            }
        };

        for fact in facts {
            match commit_or_supersede(
                pool,
                run_id,
                options,
                thought,
                &fact,
                extractor,
                embedder_model_id,
            )
            .await
            {
                Ok(CommitOutcome::Committed) => n_committed += 1,
                Ok(CommitOutcome::Review) => n_review += 1,
                Ok(CommitOutcome::Skipped) | Ok(CommitOutcome::NoOp) => {}
                Err(err) => {
                    tracing::error!(
                        run_id = %run_id,
                        thought_id = %thought.id,
                        statement = %fact.statement,
                        error = %err,
                        "reflector: failed to persist extracted fact",
                    );
                }
            }
        }
        n_processed += 1;
    }

    engram_storage::finish_run(
        pool,
        run_id,
        n_processed,
        n_committed,
        n_review,
        n_failures,
        None,
    )
    .await?;

    Ok(ReflectorReport {
        run_id,
        n_thoughts_processed: n_processed,
        n_facts_committed: n_committed,
        n_review_queue: n_review,
        n_extractor_failures: n_failures,
    })
}

/// `true` iff a newly-extracted fact matches an existing active row on every
/// content field: statement, the (S, P, O) triple, extractor version, and
/// confidence (with float epsilon). `extractor_model` is deliberately not
/// in the comparison (a rerun against the same extractor instance is a
/// precondition of the call); `source_run_id` is provenance, not content.
fn is_byte_identical(new: &ExtractedFact, new_version: i32, existing: &Fact) -> bool {
    existing.statement == new.statement
        && existing.subject == new.subject
        && existing.predicate == new.predicate
        && existing.object == new.object
        && existing.extractor_version == new_version
        && (existing.confidence - new.confidence).abs() < 1e-6
}

/// Outcome of `commit_or_supersede` for a single extracted fact. Callers use
/// this to keep their counters (`n_committed` / `n_review` / etc.) accurate
/// without needing to know the internal decision tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitOutcome {
    /// Empty / whitespace-only statement; skipped without DB write.
    Skipped,
    /// New row inserted into `facts` (with or without drift folding via
    /// supersession). Includes the brand-new-claim and the new-canonical-with-fold cases.
    Committed,
    /// Low-confidence; routed to `facts_review_queue` instead of `facts`.
    Review,
    /// Byte-identical match already active on the thought; no new row
    /// inserted (the no-op floor). Drift rows may still have been folded
    /// into the existing canonical via supersession.
    NoOp,
}

/// What `metadata.extract` directs the reflector to do for a given thought.
/// Plumbed from `thoughts.metadata` JSONB; see [`ExtractMode`] in
/// `engram-core` for the semantics that ride on `Run(_)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractDirective {
    /// `metadata.extract: "none"` — skip extraction entirely; the thought is
    /// intentionally not subject to fact derivation. Still counted as
    /// `n_processed` so reflector_runs accounting stays honest.
    Skip,
    /// Call the extractor with this mode. Covers absent flag (back-compat),
    /// `"all"`, `"durable-only"`, and unknown-value fall-through.
    Run(ExtractMode),
}

/// Parse `metadata.extract` (free-form JSONB) into a typed directive.
/// Unknown / non-string values fall through to `Run(ExtractMode::All)` —
/// back-compat for any thought captured before this flag existed.
fn extract_directive(metadata: &Metadata) -> ExtractDirective {
    match metadata.as_value().get("extract").and_then(|v| v.as_str()) {
        Some("none") => ExtractDirective::Skip,
        Some("durable-only") => ExtractDirective::Run(ExtractMode::DurableOnly),
        _ => ExtractDirective::Run(ExtractMode::All),
    }
}

/// Per-fact decision logic shared by `run_reflector_once` (first-time
/// extraction) and `run_reflector_rerun` (re-evaluating facted thoughts).
/// Implements the four-case decision tree under the dedup-via-supersession
/// design principle locked 2026-05-14: facts table is append-only audit;
/// `superseded_at` / `superseded_by` are the deprecation mechanism; claim
/// transitions produce a new active row + supersession on the old one.
///
/// Cases:
/// - empty/whitespace `statement` → [`CommitOutcome::Skipped`].
/// - `fact.confidence < options.review_queue_below` → row to
///   `facts_review_queue`; existing active rows unchanged → [`CommitOutcome::Review`].
/// - 0 matches via `find_matching_active_facts` → insert as brand-new claim
///   → [`CommitOutcome::Committed`].
/// - ≥1 match, byte-identical row already active → no-op floor; non-identical
///   drift rows fold into the canonical via `superseded_by` → [`CommitOutcome::NoOp`].
/// - ≥1 match, none byte-identical → insert new as canonical; fold all
///   matches via `superseded_by` → [`CommitOutcome::Committed`].
///
/// All writes for a single matched group happen in one transaction so a
/// crash between writes cannot orphan rows.
async fn commit_or_supersede(
    pool: &PgPool,
    run_id: RunId,
    options: &ReflectorOptions,
    thought: &Thought,
    fact: &ExtractedFact,
    extractor: &dyn Extractor,
    embedder_model_id: &str,
) -> Result<CommitOutcome, ReflectorError> {
    if fact.statement.trim().is_empty() {
        return Ok(CommitOutcome::Skipped);
    }

    // Review-routed facts bypass supersession: a low-confidence extraction
    // does not deprecate an existing high-confidence active fact. The row
    // lands in the queue and the operator adjudicates.
    if fact.confidence < options.review_queue_below {
        engram_storage::insert_review_queue_row(
            pool,
            NewReviewRow {
                statement: &fact.statement,
                subject: fact.subject.as_deref(),
                predicate: fact.predicate.as_deref(),
                object: fact.object.as_deref(),
                source_thought_id: thought.id,
                extractor_model: extractor.model_id(),
                extractor_version: extractor.version(),
                source_run_id: Some(run_id),
                confidence: fact.confidence,
            },
        )
        .await?;
        return Ok(CommitOutcome::Review);
    }

    // Commit path: "same claim" predicate is `statement` match OR (S, P, O)
    // match via `find_matching_active_facts`. Multiple drift rows can match
    // (pre-existing audit-corrupt state); they all get folded.
    let matches = engram_storage::find_matching_active_facts(
        pool,
        thought.id,
        &fact.statement,
        fact.subject.as_deref(),
        fact.predicate.as_deref(),
        fact.object.as_deref(),
    )
    .await?;

    if matches.is_empty() {
        let new_fact_id = engram_storage::insert_fact(
            pool,
            NewFact {
                scope: &thought.scope,
                statement: &fact.statement,
                subject: fact.subject.as_deref(),
                predicate: fact.predicate.as_deref(),
                object: fact.object.as_deref(),
                source_thought_id: thought.id,
                extractor_model: extractor.model_id(),
                extractor_version: extractor.version(),
                source_run_id: Some(run_id),
                confidence: fact.confidence,
            },
        )
        .await?;
        engram_storage::enqueue_embedding(
            pool,
            engram_storage::target::FACT,
            new_fact_id,
            embedder_model_id,
        )
        .await?;
        return Ok(CommitOutcome::Committed);
    }

    let canonical = matches
        .iter()
        .find(|m| is_byte_identical(fact, extractor.version(), m))
        .map(|m| m.id);

    let mut tx = pool
        .begin()
        .await
        .map_err(engram_storage::StorageError::from)?;

    let target_id = if let Some(existing_id) = canonical {
        existing_id
    } else {
        sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version,
                source_run_id, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            "#,
            thought.scope.as_str(),
            fact.statement,
            fact.subject,
            fact.predicate,
            fact.object,
            thought.id.into_uuid(),
            extractor.model_id(),
            extractor.version(),
            run_id.0,
            fact.confidence,
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(engram_storage::StorageError::from)?
    };

    for m in &matches {
        if m.id == target_id {
            continue;
        }
        sqlx::query!(
            r#"
            UPDATE facts
            SET superseded_by = $2, superseded_at = NOW()
            WHERE id = $1 AND superseded_at IS NULL
            "#,
            m.id,
            target_id,
        )
        .execute(&mut *tx)
        .await
        .map_err(engram_storage::StorageError::from)?;
    }

    tx.commit()
        .await
        .map_err(engram_storage::StorageError::from)?;

    if canonical.is_some() {
        Ok(CommitOutcome::NoOp)
    } else {
        // `target_id` is the row we just inserted; enqueue it for embedding.
        // The no-op-floor path above (canonical = Some) skips this because
        // the byte-identical existing row was already enqueued when it was
        // first inserted.
        engram_storage::enqueue_embedding(
            pool,
            engram_storage::target::FACT,
            target_id,
            embedder_model_id,
        )
        .await?;
        Ok(CommitOutcome::Committed)
    }
}

/// Re-evaluate already-facted thoughts and reconcile against the current
/// extractor. For each thought, walk the new extractions and treat the
/// `facts` table as append-only audit: claim transitions produce a new
/// active row plus `superseded_at`/`superseded_by` on the old one.
///
/// "Same claim" predicate: `statement` match OR `(subject, predicate, object)`
/// match via `IS NOT DISTINCT FROM` (either signal counts). For each new
/// extraction:
///
/// - **0 matches** → insert as a brand-new claim.
/// - **\>= 1 matches, one is byte-identical** → no insert (no-op floor);
///   fold the non-identical drift rows into the byte-identical one via
///   `superseded_by`.
/// - **\>= 1 matches, none byte-identical** → insert new row as the
///   canonical; fold all drift rows into it via `superseded_by`.
///
/// All writes for a single matched group happen in one transaction so a
/// crash between writes cannot orphan rows.
///
/// Low-confidence rerun extractions bypass supersession entirely — they
/// route to `facts_review_queue` and the existing active row stays put.
///
/// **No subtractive logic:** existing active facts that the new extractor
/// *doesn't* reproduce stay active. A single rerun reflects model drift in
/// *how* facts are stated, not in *what* the thought says — subtractive
/// logic risks losing real facts to sampling variance. Operators can
/// `correct_fact` such rows manually.
///
/// Per-thought extractor failures are soft (logged + counted + continue),
/// matching `run_reflector_once`'s Q9 behavior.
pub async fn run_reflector_rerun(
    pool: &PgPool,
    extractor: &dyn Extractor,
    embedder_model_id: &str,
    options: &ReflectorOptions,
    since: Option<OffsetDateTime>,
) -> Result<ReflectorReport, ReflectorError> {
    let run_id = engram_storage::start_run(
        pool,
        extractor.model_id(),
        extractor.version(),
        options.scope_filter.as_deref(),
    )
    .await?;

    let thoughts = engram_storage::find_facted_thoughts(
        pool,
        options.scope_filter.as_deref(),
        since,
        options.max_thoughts_per_run,
    )
    .await?;

    let mut n_processed: i32 = 0;
    let mut n_committed: i32 = 0;
    let mut n_review: i32 = 0;
    let mut n_failures: i32 = 0;

    for thought in &thoughts {
        let extract_mode = match extract_directive(&thought.metadata) {
            ExtractDirective::Skip => {
                n_processed += 1;
                continue;
            }
            ExtractDirective::Run(mode) => mode,
        };
        let ctx = ExtractionContext::new(thought.scope.clone(), options.max_facts_per_thought)
            .with_extract_mode(extract_mode);
        let facts = match extractor.extract(thought, &ctx).await {
            Ok(facts) => facts,
            Err(err) => {
                tracing::warn!(
                    run_id = %run_id,
                    thought_id = %thought.id,
                    error = %err,
                    transient = err.is_transient(),
                    "reflector rerun: extractor failed; thought skipped this run",
                );
                n_failures += 1;
                n_processed += 1;
                continue;
            }
        };

        for fact in facts {
            match commit_or_supersede(
                pool,
                run_id,
                options,
                thought,
                &fact,
                extractor,
                embedder_model_id,
            )
            .await
            {
                Ok(CommitOutcome::Committed) => n_committed += 1,
                Ok(CommitOutcome::Review) => n_review += 1,
                Ok(CommitOutcome::Skipped) | Ok(CommitOutcome::NoOp) => {}
                Err(err) => {
                    tracing::error!(
                        run_id = %run_id,
                        thought_id = %thought.id,
                        statement = %fact.statement,
                        error = %err,
                        "reflector rerun: failed to persist extracted fact",
                    );
                }
            }
        }
        n_processed += 1;
    }

    engram_storage::finish_run(
        pool,
        run_id,
        n_processed,
        n_committed,
        n_review,
        n_failures,
        None,
    )
    .await?;

    Ok(ReflectorReport {
        run_id,
        n_thoughts_processed: n_processed,
        n_facts_committed: n_committed,
        n_review_queue: n_review,
        n_extractor_failures: n_failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{capture, CaptureRequest};
    use engram_core::{ExtractedFact, Scope, Source, ThoughtId};
    use engram_extract::{FakeBehavior, FakeExtractor};

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

    fn options(review_below: f32) -> ReflectorOptions {
        ReflectorOptions {
            enabled: true,
            schedule: "0 0 3 * * *".to_string(),
            scope_filter: None,
            max_thoughts_per_run: 100,
            max_facts_per_thought: 8,
            review_queue_below: review_below,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn commits_high_confidence_facts(pool: PgPool) {
        let id = cap(&pool, "Engram uses pgvector", "global").await;
        let extractor = FakeExtractor::with_confidence(0.9);
        let report = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        assert_eq!(report.n_thoughts_processed, 1);
        assert_eq!(report.n_facts_committed, 1);
        assert_eq!(report.n_review_queue, 0);
        assert_eq!(report.n_extractor_failures, 0);

        let fact_count = sqlx::query!(
            r#"SELECT COUNT(*) AS "count!" FROM facts WHERE source_thought_id = $1"#,
            id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .count;
        assert_eq!(fact_count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn routes_low_confidence_to_review_queue(pool: PgPool) {
        let id = cap(&pool, "vague claim", "global").await;
        let extractor = FakeExtractor::with_confidence(0.3);
        let report = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        assert_eq!(report.n_facts_committed, 0);
        assert_eq!(report.n_review_queue, 1);

        let fact_count = sqlx::query!(
            r#"SELECT COUNT(*) AS "count!" FROM facts WHERE source_thought_id = $1"#,
            id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .count;
        let review_count = sqlx::query!(
            r#"SELECT COUNT(*) AS "count!" FROM facts_review_queue WHERE source_thought_id = $1"#,
            id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .count;
        assert_eq!(fact_count, 0);
        assert_eq!(review_count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn writes_source_run_id_on_committed_facts(pool: PgPool) {
        cap(&pool, "stamp me", "global").await;
        let extractor = FakeExtractor::with_confidence(0.85);
        let report = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();

        let row = sqlx::query!(
            r#"SELECT source_run_id FROM facts LIMIT 1"#
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.source_run_id, Some(report.run_id.0));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn skips_thought_when_extractor_fails(pool: PgPool) {
        cap(&pool, "unreachable extractor", "global").await;
        let extractor = FakeExtractor::always_failing(FakeBehavior::Unreachable);
        let report = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        assert_eq!(report.n_thoughts_processed, 1);
        assert_eq!(report.n_extractor_failures, 1);
        assert_eq!(report.n_facts_committed, 0);
        assert_eq!(report.n_review_queue, 0);

        let fact_count = sqlx::query!(r#"SELECT COUNT(*) AS "count!" FROM facts"#)
            .fetch_one(&pool)
            .await
            .unwrap()
            .count;
        assert_eq!(fact_count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn is_idempotent_on_rerun(pool: PgPool) {
        cap(&pool, "extract me once", "global").await;
        let extractor = FakeExtractor::with_confidence(0.9);

        let first = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        assert_eq!(first.n_facts_committed, 1);

        // Second run: the thought now has a fact, so it's excluded from
        // find_unfacted_thoughts and produces no new rows.
        let second = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        assert_eq!(second.n_thoughts_processed, 0);
        assert_eq!(second.n_facts_committed, 0);

        let fact_count = sqlx::query!(r#"SELECT COUNT(*) AS "count!" FROM facts"#)
            .fetch_one(&pool)
            .await
            .unwrap()
            .count;
        assert_eq!(fact_count, 1, "rerun must not duplicate facts");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn updates_run_counts(pool: PgPool) {
        for i in 0..3 {
            cap(&pool, &format!("t-{i}"), "global").await;
        }
        let extractor = FakeExtractor::with_confidence(0.9);
        let report = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();

        let row = sqlx::query!(
            r#"SELECT n_thoughts_processed, n_facts_committed, n_review_queue, finished_at, error
               FROM reflector_runs WHERE id = $1"#,
            report.run_id.0,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.n_thoughts_processed, 3);
        assert_eq!(row.n_facts_committed, 3);
        assert_eq!(row.n_review_queue, 0);
        assert!(row.finished_at.is_some());
        assert!(row.error.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn scope_filter_only_processes_in_scope(pool: PgPool) {
        cap(&pool, "in scope", "work").await;
        cap(&pool, "out of scope", "personal").await;

        let extractor = FakeExtractor::with_confidence(0.9);
        let mut opts = options(0.7);
        opts.scope_filter = Some("work".to_string());
        let report = run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &opts).await.unwrap();
        assert_eq!(report.n_thoughts_processed, 1);
        assert_eq!(report.n_facts_committed, 1);

        // The personal thought is still unfacted.
        let remaining =
            engram_storage::find_unfacted_thoughts(&pool, None, 10).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].scope.as_str(), "personal");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn explicit_facts_override_persists_subject_predicate_object(pool: PgPool) {
        cap(&pool, "anchor", "global").await;
        let extractor = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "Engram uses pgvector".into(),
            subject: Some("Engram".into()),
            predicate: Some("uses".into()),
            object: Some("pgvector".into()),
            confidence: 0.95,
        }]);
        run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();

        let row = sqlx::query!(
            r#"SELECT statement, subject, predicate, object FROM facts LIMIT 1"#,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.statement, "Engram uses pgvector");
        assert_eq!(row.subject.as_deref(), Some("Engram"));
        assert_eq!(row.predicate.as_deref(), Some("uses"));
        assert_eq!(row.object.as_deref(), Some("pgvector"));
    }

    // -- M2 Phase D / M3 dedup-via-supersession: run_reflector_rerun ------

    /// No-op floor: a rerun whose new extraction is byte-identical (statement,
    /// triple, confidence, extractor_version) to an existing active row must
    /// not write anything new — neither an insert nor a supersession.
    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_no_ops_on_byte_identical_match(pool: PgPool) {
        cap(&pool, "stable thought", "global").await;
        let extractor = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "stable fact".into(),
            subject: Some("S".into()),
            predicate: Some("P".into()),
            object: Some("O".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7)).await.unwrap();
        let before = sqlx::query!(r#"SELECT COUNT(*) AS "n!" FROM facts"#).fetch_one(&pool).await.unwrap().n;

        let report = run_reflector_rerun(&pool, &extractor, TEST_MODEL_ID, &options(0.7), None).await.unwrap();
        let after = sqlx::query!(r#"SELECT COUNT(*) AS "n!" FROM facts"#).fetch_one(&pool).await.unwrap().n;
        assert_eq!(before, after, "byte-identical match must produce zero new rows");
        assert_eq!(report.n_facts_committed, 0);

        // And no supersession either.
        let superseded = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM facts WHERE superseded_at IS NOT NULL"#
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(superseded, 0, "byte-identical match must not trigger supersession");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_supersedes_when_triple_matches_but_statement_differs(pool: PgPool) {
        cap(&pool, "drifting thought", "global").await;
        // First pass: produce the old statement.
        let v1 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "old wording".into(),
            subject: Some("S".into()),
            predicate: Some("P".into()),
            object: Some("O".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &v1, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        // Rerun with an extractor that gives the same (S,P,O) but new statement.
        let v2 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "new wording".into(),
            subject: Some("S".into()),
            predicate: Some("P".into()),
            object: Some("O".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &v2, TEST_MODEL_ID, &options(0.7), None).await.unwrap();

        // Active facts now: only the new one.
        let active = sqlx::query!(
            r#"SELECT statement FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].statement, "new wording");

        // Audit: old row is still in `facts`, marked superseded, with superseded_by → new.
        let old = sqlx::query!(
            r#"SELECT id, superseded_by, superseded_at
               FROM facts WHERE statement = 'old wording'"#
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(old.superseded_at.is_some());
        assert!(old.superseded_by.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_inserts_when_no_match(pool: PgPool) {
        cap(&pool, "growing thought", "global").await;
        let v1 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "old fact".into(),
            subject: Some("A".into()),
            predicate: Some("rel".into()),
            object: Some("B".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &v1, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        // Rerun with an extractor that produces a *different* (S,P,O).
        let v2 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "additional insight".into(),
            subject: Some("X".into()),
            predicate: Some("rel".into()),
            object: Some("Y".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &v2, TEST_MODEL_ID, &options(0.7), None).await.unwrap();

        // Both should be active — the old one is not subtracted.
        let active = sqlx::query!(
            r#"SELECT statement FROM facts WHERE superseded_at IS NULL ORDER BY statement"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 2);
    }

    /// The M2 dogfood regression: v1 produced fact `c5799e68` with statement
    /// "current API surface is append-only" + triple ("current API surface",
    /// "is", "append-only"); v2 produced fact `1c4a53c1` with the **same**
    /// statement but a different triple ("thoughts in current API surface",
    /// "are", "append-only"). The pre-M3 dedup keyed on (S, P, O) only and
    /// missed the match, leaving both rows parallel-active. The widened
    /// predicate (statement OR triple) catches the statement match; the
    /// v2 row becomes canonical and the v1 row is superseded.
    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_supersedes_when_statement_matches_but_triple_differs(pool: PgPool) {
        cap(&pool, "API doc thought", "global").await;
        let v1 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "current API surface is append-only".into(),
            subject: Some("current API surface".into()),
            predicate: Some("is".into()),
            object: Some("append-only".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &v1, TEST_MODEL_ID, &options(0.7)).await.unwrap();
        let v1_id = sqlx::query!(
            r#"SELECT id FROM facts WHERE statement = 'current API surface is append-only'"#
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .id;

        let v2 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "current API surface is append-only".into(),
            subject: Some("thoughts in current API surface".into()),
            predicate: Some("are".into()),
            object: Some("append-only".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &v2, TEST_MODEL_ID, &options(0.7), None).await.unwrap();

        // Exactly one active row remains.
        let active = sqlx::query!(
            r#"SELECT id, subject FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 1, "drift duplicate must fold into canonical");
        // The active row is the v2-shape (different subject than v1).
        assert_eq!(
            active[0].subject.as_deref(),
            Some("thoughts in current API surface")
        );

        // v1 is superseded with superseded_by pointing at the v2 row.
        let v1_row = sqlx::query!(
            r#"SELECT superseded_at, superseded_by FROM facts WHERE id = $1"#,
            v1_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(v1_row.superseded_at.is_some());
        assert_eq!(v1_row.superseded_by, Some(active[0].id));
    }

    /// Pre-seed two parallel-active drift duplicates (audit-corrupt state),
    /// then rerun. Both should fold into a single canonical row via
    /// `superseded_by`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_folds_multiple_matches_into_one_supersede_by(pool: PgPool) {
        let thought_id = cap(&pool, "drifty thought", "global").await;

        // Seed two drift duplicates by hand: matching statement, different
        // triples, both active. This simulates the pre-existing corrupt
        // audit state that a rerun should clean up.
        let drift_a = sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'fake/extractor', 1, 0.9)
            RETURNING id
            "#,
            "global",
            "shared statement",
            "subj-a",
            "pred",
            "obj",
            thought_id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let drift_b = sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'fake/extractor', 1, 0.9)
            RETURNING id
            "#,
            "global",
            "shared statement",
            "subj-b",
            "pred",
            "obj",
            thought_id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        // Now a rerun that produces a v3-decomposition: same statement,
        // yet another triple.
        let v3 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "shared statement".into(),
            subject: Some("subj-c".into()),
            predicate: Some("pred".into()),
            object: Some("obj".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &v3, TEST_MODEL_ID, &options(0.7), None).await.unwrap();

        // Both drifts are superseded; new row is canonical.
        let active = sqlx::query!(
            r#"SELECT id, subject FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].subject.as_deref(), Some("subj-c"));

        let new_id = active[0].id;
        let drift_a_row = sqlx::query!(
            r#"SELECT superseded_at, superseded_by FROM facts WHERE id = $1"#,
            drift_a,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let drift_b_row = sqlx::query!(
            r#"SELECT superseded_at, superseded_by FROM facts WHERE id = $1"#,
            drift_b,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(drift_a_row.superseded_at.is_some());
        assert!(drift_b_row.superseded_at.is_some());
        assert_eq!(drift_a_row.superseded_by, Some(new_id));
        assert_eq!(drift_b_row.superseded_by, Some(new_id));
    }

    /// When an exact byte-identical match coexists with a drift row, the
    /// no-op floor wins: the exact match stays canonical (no new insert),
    /// and the drift row folds into it.
    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_keeps_exact_match_and_folds_drift(pool: PgPool) {
        let thought_id = cap(&pool, "two-active thought", "global").await;

        // Seed an exact-match row and a drift row, both active.
        let exact = sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'fake/extractor', 1, 0.9)
            RETURNING id
            "#,
            "global",
            "canonical statement",
            "S",
            "P",
            "O",
            thought_id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let drift = sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'fake/extractor', 1, 0.9)
            RETURNING id
            "#,
            "global",
            "canonical statement",
            "S-drift",
            "P",
            "O",
            thought_id.into_uuid(),
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        let before_total =
            sqlx::query!(r#"SELECT COUNT(*) AS "n!" FROM facts"#).fetch_one(&pool).await.unwrap().n;

        // Rerun emits a fact byte-identical to `exact`.
        let v = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "canonical statement".into(),
            subject: Some("S".into()),
            predicate: Some("P".into()),
            object: Some("O".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &v, TEST_MODEL_ID, &options(0.7), None).await.unwrap();

        // No new row was inserted — the exact match served as canonical.
        let after_total =
            sqlx::query!(r#"SELECT COUNT(*) AS "n!" FROM facts"#).fetch_one(&pool).await.unwrap().n;
        assert_eq!(before_total, after_total, "no-op floor must skip insert when exact match exists");

        // `exact` stays active; `drift` folds into it.
        let exact_row = sqlx::query!(
            r#"SELECT superseded_at FROM facts WHERE id = $1"#,
            exact,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let drift_row = sqlx::query!(
            r#"SELECT superseded_at, superseded_by FROM facts WHERE id = $1"#,
            drift,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exact_row.superseded_at.is_none(), "exact match must stay active");
        assert!(drift_row.superseded_at.is_some(), "drift must be superseded");
        assert_eq!(drift_row.superseded_by, Some(exact));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_run_twice_produces_identical_fact_set(pool: PgPool) {
        // m2-facts-pipeline.md success criterion #5: idempotency.
        cap(&pool, "anchor", "global").await;
        let extractor = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "stable".into(),
            subject: Some("S".into()),
            predicate: Some("P".into()),
            object: Some("O".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &extractor, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        // First rerun.
        run_reflector_rerun(&pool, &extractor, TEST_MODEL_ID, &options(0.7), None).await.unwrap();
        let snap1 = sqlx::query!(
            r#"SELECT id, statement, superseded_at FROM facts ORDER BY id"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        // Second rerun.
        run_reflector_rerun(&pool, &extractor, TEST_MODEL_ID, &options(0.7), None).await.unwrap();
        let snap2 = sqlx::query!(
            r#"SELECT id, statement, superseded_at FROM facts ORDER BY id"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        assert_eq!(snap1.len(), snap2.len(), "rerun count must not change after second rerun");
        for (a, b) in snap1.iter().zip(snap2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.statement, b.statement);
            assert_eq!(
                a.superseded_at.is_some(),
                b.superseded_at.is_some(),
                "supersession state must be stable across reruns"
            );
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rerun_respects_scope_and_since(pool: PgPool) {
        cap(&pool, "work thought", "work").await;
        cap(&pool, "personal thought", "personal").await;

        let v1 = FakeExtractor::with_confidence(0.9);
        run_reflector_once(&pool, &v1, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        // Rerun scoped to "work" with a since cutoff in the past — should
        // process only the work thought.
        let v2 = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "rerun result".into(),
            subject: None, predicate: None, object: None,
            confidence: 0.9,
        }]);
        let mut opts = options(0.7);
        opts.scope_filter = Some("work".to_string());
        let since = OffsetDateTime::now_utc() - time::Duration::days(1);
        let report = run_reflector_rerun(&pool, &v2, TEST_MODEL_ID, &opts, Some(since))
            .await
            .unwrap();
        assert_eq!(report.n_thoughts_processed, 1);
    }

    // -- M3 Phase A: extract metadata flag --------------------------------

    /// Helper: capture a thought with `metadata.extract = <flag>`.
    async fn cap_with_extract(pool: &PgPool, content: &str, scope: &str, flag: &str) -> ThoughtId {
        capture(
            pool,
            TEST_MODEL_ID,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new(scope).unwrap()),
                metadata: Some(serde_json::json!({ "extract": flag }).into()),
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    /// `metadata.extract = "none"` short-circuits the reflector before the
    /// extractor is called — zero facts written, but the thought is still
    /// counted as processed for accurate run accounting.
    #[sqlx::test(migrations = "../../migrations")]
    async fn reflector_skips_thought_with_extract_none(pool: PgPool) {
        cap_with_extract(&pool, "skip-this thought", "global", "none").await;
        let e = FakeExtractor::with_confidence(0.9);
        let report = run_reflector_once(&pool, &e, TEST_MODEL_ID, &options(0.7)).await.unwrap();
        assert_eq!(report.n_thoughts_processed, 1);
        assert_eq!(report.n_facts_committed, 0);

        let n_facts =
            sqlx::query!(r#"SELECT COUNT(*) AS "n!" FROM facts"#).fetch_one(&pool).await.unwrap().n;
        assert_eq!(n_facts, 0, "no facts should be written for extract=none");

        // Extractor was never invoked.
        assert!(
            e.last_ctx().is_none(),
            "extractor should not have been called for extract=none"
        );
    }

    /// `metadata.extract = "durable-only"` propagates through
    /// `ExtractionContext::extract_mode` to the extractor (where the
    /// openai-compatible impl injects an additional system message).
    #[sqlx::test(migrations = "../../migrations")]
    async fn reflector_propagates_durable_only_via_context(pool: PgPool) {
        cap_with_extract(&pool, "mixed content thought", "global", "durable-only").await;
        let e = FakeExtractor::with_confidence(0.9);
        run_reflector_once(&pool, &e, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        let ctx = e
            .last_ctx()
            .expect("extractor should have been called once");
        assert_eq!(ctx.extract_mode, ExtractMode::DurableOnly);
    }

    /// Absent / unknown `metadata.extract` falls through to
    /// `ExtractMode::All` (back-compat with thoughts captured before the
    /// flag existed).
    #[sqlx::test(migrations = "../../migrations")]
    async fn reflector_treats_absent_extract_as_all(pool: PgPool) {
        cap(&pool, "plain thought, no metadata flag", "global").await;
        let e = FakeExtractor::with_confidence(0.9);
        run_reflector_once(&pool, &e, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        let ctx = e
            .last_ctx()
            .expect("extractor should have been called once");
        assert_eq!(ctx.extract_mode, ExtractMode::All);
    }

    // -- M3 Phase A: commit_or_supersede on run_reflector_once ------------

    /// The within-call analogue of `rerun_supersedes_when_statement_matches_but_triple_differs`:
    /// an initial-extraction call (not a rerun) that emits two facts on the
    /// same thought with byte-identical statements and different triples
    /// must produce exactly one active row, with the second row folded into
    /// the first via `superseded_by`.
    ///
    /// Concrete dogfood case: thought 86c3392f produced `39016e00` and
    /// `bce8ac05` (same statement, different broken triples) in a single
    /// extraction call before M3 Phase A. After this fix, that pattern is
    /// no longer reproducible.
    #[sqlx::test(migrations = "../../migrations")]
    async fn once_supersedes_when_statement_matches_but_triple_differs_within_call(pool: PgPool) {
        cap(&pool, "thought with paraphrase drift in same call", "global").await;
        // Two facts with byte-identical statements, different triples.
        let e = FakeExtractor::with_facts(vec![
            ExtractedFact {
                statement: "SIMD parsing outperforms scalar on large documents".into(),
                subject: Some("SIMD parsing".into()),
                predicate: Some("outperforms".into()),
                object: Some("SIMD parsing".into()), // self-referential drift
                confidence: 0.9,
            },
            ExtractedFact {
                statement: "SIMD parsing outperforms scalar on large documents".into(),
                subject: Some("SIMD parsing".into()),
                predicate: Some("outperforms".into()),
                object: Some("scalar".into()),
                confidence: 0.9,
            },
        ]);
        run_reflector_once(&pool, &e, TEST_MODEL_ID, &options(0.7)).await.unwrap();

        // Exactly one active row remains; the other is superseded.
        let active = sqlx::query!(
            r#"SELECT id, subject, predicate, object FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(
            active.len(),
            1,
            "within-call same-statement-different-triple emit should fold to one active row"
        );

        let superseded = sqlx::query!(
            r#"SELECT id, superseded_by FROM facts WHERE superseded_at IS NOT NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(superseded.len(), 1);
        assert_eq!(
            superseded[0].superseded_by,
            Some(active[0].id),
            "superseded row should link to the canonical (active) row"
        );
    }

    // -- M3 Phase B step 1: reflector enqueues fact embeddings ---------------

    /// After every fact insert inside `commit_or_supersede`, the reflector
    /// should enqueue a `pending_embeddings` row for the new fact under the
    /// active embedder model_id. The no-op-floor branch (byte-identical
    /// match exists) must NOT enqueue (no new fact written).
    #[sqlx::test(migrations = "../../migrations")]
    async fn reflector_enqueues_fact_embedding_after_commit_or_supersede(pool: PgPool) {
        cap(&pool, "anchor thought", "global").await;
        let e = FakeExtractor::with_facts(vec![
            ExtractedFact {
                statement: "first claim".into(),
                subject: Some("a".into()),
                predicate: Some("rel".into()),
                object: Some("b".into()),
                confidence: 0.9,
            },
            ExtractedFact {
                statement: "second claim".into(),
                subject: None,
                predicate: None,
                object: None,
                confidence: 0.9,
            },
        ]);
        run_reflector_once(&pool, &e, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();

        // Two fact rows committed → two pending_embeddings rows under
        // target_kind='fact' with the right model_id.
        let n_fact_pending = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM pending_embeddings
               WHERE target_kind = 'fact' AND model_id = $1"#,
            TEST_MODEL_ID,
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(n_fact_pending, 2);

        // Second rerun against the same extractor → byte-identical match,
        // no-op floor fires, no new pending row.
        run_reflector_rerun(&pool, &e, TEST_MODEL_ID, &options(0.7), None)
            .await
            .unwrap();
        let after = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM pending_embeddings
               WHERE target_kind = 'fact' AND model_id = $1"#,
            TEST_MODEL_ID,
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(after, n_fact_pending, "no-op floor must not enqueue");
    }
}

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
use engram_storage::{NewReviewRow, RunId};
use uuid::Uuid;
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
    /// `facts`. Lower bound of the three-band routing (M3 Phase C).
    pub review_queue_below: f32,
    /// Confidence below this threshold (but ≥ `review_queue_below`) commits
    /// the fact to `facts` with `flagged = true` — the "stored but
    /// flagged" middle band. At-or-above commits with `flagged = false`.
    /// Kill-switch: set this equal to `review_queue_below` to collapse
    /// back to two-band routing (every committed row gets `flagged =
    /// false`). M3 Phase C.
    pub min_confidence_to_store: f32,
    /// Policy for subsumption-aware dedup: when two facts on the same
    /// thought share `(subject, predicate)` and one's `object` is a
    /// substring of the other's, which do we keep? Default `Specific`
    /// drops the more general; `General` does the inverse. M3 Phase C.
    pub subsumption_keep: SubsumptionKeep,
}

/// Operator policy for [`commit_or_supersede`]'s subsumption-aware dedup
/// pass. Configured via `[reflector] subsumption_keep` in `engram.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SubsumptionKeep {
    /// Keep the more specific fact; supersede the more general. Default —
    /// matches the dogfood expectation that the specific claim implies
    /// the general one.
    #[default]
    Specific,
    /// Keep the more general fact; supersede the more specific.
    General,
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
            min_confidence_to_store: 0.85,
            subsumption_keep: SubsumptionKeep::Specific,
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
                Ok(CommitOutcome::Skipped)
                | Ok(CommitOutcome::NoOp)
                | Ok(CommitOutcome::Inherited)
                | Ok(CommitOutcome::DriftFolded)
                | Ok(CommitOutcome::SubsumedByActive) => {}
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
    /// New active row inserted into `facts` (with or without drift folding
    /// via supersession). Includes the brand-new-claim and the
    /// new-canonical-with-fold cases.
    Committed,
    /// Low-confidence; routed to `facts_review_queue` instead of `facts`.
    Review,
    /// Byte-identical match already active on the thought; no new row
    /// inserted (the no-op floor). Drift rows may still have been folded
    /// into the existing canonical via supersession. Counter-equivalent
    /// to `Inherited` / `DriftFolded`.
    NoOp,
    /// M3 Phase C: the same claim was previously retracted (matched in
    /// `find_matching_superseded_facts`). The new emission is written for
    /// audit and immediately superseded — `superseded_by` inherits the
    /// existing canonical's pointer (or `NULL` for retraction-without-
    /// replacement). No active state change; the active fact set is
    /// unchanged. Treated like `NoOp` by counters; the distinct variant
    /// exists for test introspection.
    Inherited,
    /// M3 Phase C: the quality-aware pick chose an existing active fact
    /// over the new emission (statements matched but full rows weren't
    /// byte-identical, and the existing row scored higher on the quality
    /// tiebreakers). The new emission was written for audit and
    /// immediately superseded into the chosen canonical. No active state
    /// change. Treated like `NoOp` by counters.
    DriftFolded,
    /// M3 Phase C: subsumption-aware dedup decided the new emission was
    /// less informative (more general / more specific per
    /// `subsumption_keep`) than an already-active fact sharing
    /// `(subject, predicate)`. The new emission was not inserted; the
    /// existing fact stays canonical. Treated like `NoOp` by counters.
    SubsumedByActive,
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
/// Under the dedup-via-supersession design principle locked 2026-05-14
/// and the M3 Phase C extensions: facts table is append-only audit;
/// `superseded_at` / `superseded_by` are the deprecation mechanism; claim
/// transitions produce a new active row + supersession on the old one.
///
/// Decision tree (top to bottom, first match wins):
/// 1. Empty/whitespace `statement` → [`CommitOutcome::Skipped`].
/// 2. `confidence < options.review_queue_below` → row to
///    `facts_review_queue`; existing active rows unchanged →
///    [`CommitOutcome::Review`].
/// 3. Compute `flagged = confidence < options.min_confidence_to_store`
///    for the three-band routing (Phase C). Kill-switch:
///    `min_confidence_to_store == review_queue_below` collapses to
///    two-band (always `flagged = false`).
/// 4. Find active matches (statement OR (S, P, O)). Branches:
/// - **0 active matches**: run the subsumption pass (Phase C) — if an
///   active fact on this thought shares `(subject, predicate)` with the
///   new emission's `object` in a substring relation, apply
///   `options.subsumption_keep` to either short-circuit to
///   [`CommitOutcome::SubsumedByActive`] or mark existing rows for
///   folding into the upcoming canonical. Then run the retraction
///   durability check (Phase C): if the claim has a previously-superseded
///   match, insert the new row and immediately supersede it inheriting
///   `superseded_by` from the existing canonical (or `NULL` for
///   retraction-without-replacement) → [`CommitOutcome::Inherited`].
///   Otherwise, fresh insert (plus subsumption folds) →
///   [`CommitOutcome::Committed`].
/// - **≥1 active match, byte-identical**: no-op floor; drift rows fold
///   into the byte-identical canonical → [`CommitOutcome::NoOp`].
/// - **≥1 active match, none byte-identical**: quality-aware pick
///   across `{new emission} ∪ active_matches` using a lex tuple
///   (subject != object, both-tokens-in-statement, subject-before-
///   object). If the new emission wins → insert as canonical, fold the
///   rest → [`CommitOutcome::Committed`]. If an existing fact wins →
///   insert new as already-superseded into the chosen canonical, fold
///   non-canonical matches → [`CommitOutcome::DriftFolded`]. The
///   triple-match rerun-reword partition (same SPO + different
///   statement on any active match) short-circuits to "new wins".
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

    // Three-band routing flag: facts in `[review_queue_below,
    // min_confidence_to_store)` land in `facts` with `flagged = true`.
    let flagged = fact.confidence < options.min_confidence_to_store;

    // Commit path: "same claim" predicate is `statement` match OR (S, P, O)
    // match. Multiple drift rows can match (pre-existing audit-corrupt
    // state); they all get folded.
    let active_matches = engram_storage::find_matching_active_facts(
        pool,
        thought.id,
        &fact.statement,
        fact.subject.as_deref(),
        fact.predicate.as_deref(),
        fact.object.as_deref(),
    )
    .await?;

    if active_matches.is_empty() {
        // Subsumption pre-pass (Phase C). Only fires for fully-populated
        // (subject, predicate, object) — substring-relation isn't
        // well-defined otherwise.
        let mut subsumption_folds: Vec<Uuid> = Vec::new();
        if let (Some(s), Some(p), Some(new_obj)) =
            (fact.subject.as_deref(), fact.predicate.as_deref(), fact.object.as_deref())
        {
            let subsuming =
                engram_storage::find_subsuming_active_facts(pool, thought.id, s, p).await?;
            for existing in &subsuming {
                let Some(existing_obj) = existing.object.as_deref() else {
                    continue;
                };
                match subsumption_relation(existing_obj, new_obj) {
                    SubsumptionRelation::Equal => {
                        // Identical object; the main dedup pass would have
                        // caught this on statement-or-triple match. Skip.
                    }
                    SubsumptionRelation::NewIsMoreSpecific => {
                        // New emission refines existing.
                        match options.subsumption_keep {
                            SubsumptionKeep::Specific => subsumption_folds.push(existing.id),
                            SubsumptionKeep::General => {
                                return Ok(CommitOutcome::SubsumedByActive);
                            }
                        }
                    }
                    SubsumptionRelation::NewIsMoreGeneral => {
                        // New emission is a generalisation of existing.
                        match options.subsumption_keep {
                            SubsumptionKeep::Specific => {
                                return Ok(CommitOutcome::SubsumedByActive);
                            }
                            SubsumptionKeep::General => subsumption_folds.push(existing.id),
                        }
                    }
                    SubsumptionRelation::Unrelated => {}
                }
            }
        }

        // Retraction durability (Phase C). If the same claim has been
        // retracted before, inherit its supersession state on the new
        // emission so retracted claims don't reappear on rerun.
        let superseded_matches = engram_storage::find_matching_superseded_facts(
            pool,
            thought.id,
            &fact.statement,
            fact.subject.as_deref(),
            fact.predicate.as_deref(),
            fact.object.as_deref(),
        )
        .await?;
        if !superseded_matches.is_empty() {
            // Most recent retraction shape wins (find_matching_superseded_facts
            // orders DESC). Inherit its canonical pointer; this is `None` for
            // retraction-without-replacement.
            let inherited_canonical = superseded_matches[0].1;

            let mut tx = pool.begin().await.map_err(engram_storage::StorageError::from)?;
            let new_id = sqlx::query_scalar!(
                r#"
                INSERT INTO facts (
                    scope, statement, subject, predicate, object,
                    source_thought_id, extractor_model, extractor_version,
                    source_run_id, confidence, flagged,
                    superseded_at, superseded_by
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), $12)
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
                flagged,
                inherited_canonical,
            )
            .fetch_one(&mut *tx)
            .await
            .map_err(engram_storage::StorageError::from)?;
            tx.commit().await.map_err(engram_storage::StorageError::from)?;

            // Skip embedding enqueue — the row is soft-deleted on insert,
            // so vectorizing it is wasted work.
            let _ = new_id;
            return Ok(CommitOutcome::Inherited);
        }

        // Fresh insert (plus folds for any subsumption-marked rows).
        let mut tx = pool.begin().await.map_err(engram_storage::StorageError::from)?;
        let new_fact_id = sqlx::query_scalar!(
            r#"
            INSERT INTO facts (
                scope, statement, subject, predicate, object,
                source_thought_id, extractor_model, extractor_version,
                source_run_id, confidence, flagged
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
            flagged,
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(engram_storage::StorageError::from)?;

        for old_id in &subsumption_folds {
            sqlx::query!(
                r#"
                UPDATE facts
                SET superseded_by = $2, superseded_at = NOW()
                WHERE id = $1 AND superseded_at IS NULL
                "#,
                old_id,
                new_fact_id,
            )
            .execute(&mut *tx)
            .await
            .map_err(engram_storage::StorageError::from)?;
        }

        tx.commit().await.map_err(engram_storage::StorageError::from)?;

        engram_storage::enqueue_embedding(
            pool,
            engram_storage::target::FACT,
            new_fact_id,
            embedder_model_id,
        )
        .await?;
        return Ok(CommitOutcome::Committed);
    }

    // ≥1 active match path.
    let byte_identical = active_matches
        .iter()
        .find(|m| is_byte_identical(fact, extractor.version(), m))
        .map(|m| m.id);

    if let Some(canonical_id) = byte_identical {
        // No-op floor: byte-identical canonical already active; fold any
        // non-identical drift rows.
        let mut tx = pool.begin().await.map_err(engram_storage::StorageError::from)?;
        for m in &active_matches {
            if m.id == canonical_id {
                continue;
            }
            sqlx::query!(
                r#"
                UPDATE facts
                SET superseded_by = $2, superseded_at = NOW()
                WHERE id = $1 AND superseded_at IS NULL
                "#,
                m.id,
                canonical_id,
            )
            .execute(&mut *tx)
            .await
            .map_err(engram_storage::StorageError::from)?;
        }
        tx.commit().await.map_err(engram_storage::StorageError::from)?;
        return Ok(CommitOutcome::NoOp);
    }

    // ≥1 active match, none byte-identical. Two sub-cases:
    //   - Triple-match rerun-reword: any active match has the same
    //     (S, P, O) as the new emission, but a different statement. The
    //     extractor is re-stating the same claim with new wording; the
    //     new statement is what the operator wants going forward, so the
    //     new emission wins.
    //   - SPO drift (statement match with different SPO): the extractor
    //     produced a different (S, P, O) decomposition of the same
    //     statement. This is the Phase A dogfood class — pick the best
    //     decomposition via quality_aware_pick rather than letting
    //     emission order decide.
    let triple_match = active_matches.iter().any(|m| {
        m.subject.as_deref() == fact.subject.as_deref()
            && m.predicate.as_deref() == fact.predicate.as_deref()
            && m.object.as_deref() == fact.object.as_deref()
    });
    let canonical = if triple_match {
        QualityWinner::New
    } else {
        quality_aware_pick(fact, &active_matches)
    };

    let mut tx = pool.begin().await.map_err(engram_storage::StorageError::from)?;

    match canonical {
        QualityWinner::New => {
            // New emission wins — insert as canonical, fold all existing
            // matches.
            let new_id = sqlx::query_scalar!(
                r#"
                INSERT INTO facts (
                    scope, statement, subject, predicate, object,
                    source_thought_id, extractor_model, extractor_version,
                    source_run_id, confidence, flagged
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                flagged,
            )
            .fetch_one(&mut *tx)
            .await
            .map_err(engram_storage::StorageError::from)?;

            for m in &active_matches {
                sqlx::query!(
                    r#"
                    UPDATE facts
                    SET superseded_by = $2, superseded_at = NOW()
                    WHERE id = $1 AND superseded_at IS NULL
                    "#,
                    m.id,
                    new_id,
                )
                .execute(&mut *tx)
                .await
                .map_err(engram_storage::StorageError::from)?;
            }
            tx.commit().await.map_err(engram_storage::StorageError::from)?;

            engram_storage::enqueue_embedding(
                pool,
                engram_storage::target::FACT,
                new_id,
                embedder_model_id,
            )
            .await?;
            Ok(CommitOutcome::Committed)
        }
        QualityWinner::Existing(canonical_id) => {
            // An existing fact wins — insert new emission as already-
            // superseded into the canonical (for audit); fold other
            // non-canonical matches into the canonical too.
            let _new_id = sqlx::query_scalar!(
                r#"
                INSERT INTO facts (
                    scope, statement, subject, predicate, object,
                    source_thought_id, extractor_model, extractor_version,
                    source_run_id, confidence, flagged,
                    superseded_at, superseded_by
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), $12)
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
                flagged,
                canonical_id,
            )
            .fetch_one(&mut *tx)
            .await
            .map_err(engram_storage::StorageError::from)?;

            for m in &active_matches {
                if m.id == canonical_id {
                    continue;
                }
                sqlx::query!(
                    r#"
                    UPDATE facts
                    SET superseded_by = $2, superseded_at = NOW()
                    WHERE id = $1 AND superseded_at IS NULL
                    "#,
                    m.id,
                    canonical_id,
                )
                .execute(&mut *tx)
                .await
                .map_err(engram_storage::StorageError::from)?;
            }
            tx.commit().await.map_err(engram_storage::StorageError::from)?;
            // No embedding enqueue — the new row landed superseded.
            Ok(CommitOutcome::DriftFolded)
        }
    }
}

/// Result of comparing two `object` values under the subsumption-aware
/// dedup pass. Case-insensitive substring check; whitespace-trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubsumptionRelation {
    /// Both objects are equal after normalisation. Caller defers to the
    /// statement-or-triple dedup pass.
    Equal,
    /// Existing object is a substring of the new emission's object — the
    /// new emission is the more-specific refinement.
    NewIsMoreSpecific,
    /// New emission's object is a substring of the existing — the new
    /// emission is the more-general claim.
    NewIsMoreGeneral,
    /// Neither object is a substring of the other.
    Unrelated,
}

fn subsumption_relation(existing: &str, new: &str) -> SubsumptionRelation {
    let e = existing.trim().to_lowercase();
    let n = new.trim().to_lowercase();
    if e == n {
        return SubsumptionRelation::Equal;
    }
    let e_in_n = n.contains(&e);
    let n_in_e = e.contains(&n);
    match (e_in_n, n_in_e) {
        (true, false) => SubsumptionRelation::NewIsMoreSpecific,
        (false, true) => SubsumptionRelation::NewIsMoreGeneral,
        _ => SubsumptionRelation::Unrelated,
    }
}

/// Quality-aware canonical pick for the "≥1 active match, none
/// byte-identical" branch. Compares the new emission against each
/// candidate via a lex tuple:
///   1. `subject != object` (1 if true) — rejects self-referential triples.
///   2. Both `subject` and `object` appear as case-insensitive tokens in
///      `statement` (1 if true) — rejects inverted-SPO decompositions
///      where the statement is correct but the triple swaps S and O.
///   3. `created_at` (oldest first for existing candidates; new emission
///      ranks last) — preserves the existing-first default when quality
///      bits all tie.
///
/// Returns whether the new emission wins or, if an existing candidate
/// wins, which one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityWinner {
    New,
    Existing(Uuid),
}

fn quality_aware_pick(new: &ExtractedFact, existing: &[Fact]) -> QualityWinner {
    let new_score = quality_score(
        &new.statement,
        new.subject.as_deref(),
        new.object.as_deref(),
    );

    let mut best_existing: Option<(&Fact, (i32, i32, i32))> = None;
    for f in existing {
        let s = quality_score(&f.statement, f.subject.as_deref(), f.object.as_deref());
        match best_existing {
            None => best_existing = Some((f, s)),
            Some((_, prev)) if s > prev => best_existing = Some((f, s)),
            _ => {}
        }
    }

    match best_existing {
        // No existing candidates (shouldn't happen in this branch, but
        // defensively): new wins.
        None => QualityWinner::New,
        // Existing strictly outscores new: existing wins. This is the
        // Phase A dogfood class — e.g. existing `(S=Bazel, O=Make)`
        // wins over new emission `(S=Make, O=Bazel)` because the existing
        // SPO matches the statement's subject-before-object order.
        Some((f, prev)) if prev > new_score => QualityWinner::Existing(f.id),
        // Tied or new strictly better: new wins. Ties favor the new
        // emission so a rerun's intentional reword (or a same-quality
        // SPO drift) propagates rather than getting frozen.
        _ => QualityWinner::New,
    }
}

/// Score triple in lex order, higher is better:
///   1. `subject != object` — rejects self-referential triples (1/0).
///   2. Both `subject` and `object` appear as case-insensitive substrings
///      of `statement` — rejects triples whose subject doesn't anchor in
///      the prose (1/0).
///   3. `subject`'s first occurrence in `statement` precedes `object`'s —
///      catches comparative-inversion ("A is more X than B" wants S=A,
///      O=B; an inverted triple S=B/O=A loses this bit). 1/0.
fn quality_score(
    statement: &str,
    subject: Option<&str>,
    object: Option<&str>,
) -> (i32, i32, i32) {
    let stmt_lower = statement.to_lowercase();
    let (Some(s), Some(o)) = (subject, object) else {
        return (0, 0, 0);
    };
    let s_lower = s.trim().to_lowercase();
    let o_lower = o.trim().to_lowercase();

    let s_ne_o = if s_lower != o_lower { 1 } else { 0 };

    let s_pos = stmt_lower.find(&s_lower);
    let o_pos = stmt_lower.find(&o_lower);
    let both_in_stmt = if s_pos.is_some() && o_pos.is_some() { 1 } else { 0 };
    let subj_before_obj = match (s_pos, o_pos) {
        (Some(sp), Some(op)) if sp < op => 1,
        _ => 0,
    };

    (s_ne_o, both_in_stmt, subj_before_obj)
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
                Ok(CommitOutcome::Skipped)
                | Ok(CommitOutcome::NoOp)
                | Ok(CommitOutcome::Inherited)
                | Ok(CommitOutcome::DriftFolded)
                | Ok(CommitOutcome::SubsumedByActive) => {}
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
            min_confidence_to_store: 0.85,
            subsumption_keep: SubsumptionKeep::Specific,
        }
    }

    fn options_with(
        review_below: f32,
        min_to_store: f32,
        keep: SubsumptionKeep,
    ) -> ReflectorOptions {
        ReflectorOptions {
            enabled: true,
            schedule: "0 0 3 * * *".to_string(),
            scope_filter: None,
            max_thoughts_per_run: 100,
            max_facts_per_thought: 8,
            review_queue_below: review_below,
            min_confidence_to_store: min_to_store,
            subsumption_keep: keep,
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
    /// Same statement, different (S, P, O). Pre-M3 dedup keyed on triple
    /// alone and missed this, leaving both rows parallel-active. Phase A
    /// widened the predicate to "statement OR triple"; Phase C added
    /// quality-aware canonical selection — when the statements tie but
    /// the SPOs differ, the better-scoring SPO wins (subject literally
    /// appears in the statement; subject precedes object in the prose).
    ///
    /// Concrete case (from m3-search-quality.md dogfood, thought `a7b63f3b`):
    /// v1 SPO `(current API surface, is, append-only)` has the subject
    /// literally in the statement; v2 SPO `(thoughts in current API
    /// surface, are, append-only)` has a subject that does NOT appear in
    /// the statement. Phase C keeps v1 active; v2 lands superseded.
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
            r#"SELECT id FROM facts WHERE statement = 'current API surface is append-only'
                                          AND subject = 'current API surface'"#
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

        // Exactly one active row remains (Phase A dedup invariant).
        let active = sqlx::query!(
            r#"SELECT id, subject FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 1, "drift duplicate must fold into canonical");
        // Phase C: v1 is canonical because its subject literally appears
        // in the statement; v2's subject does not.
        assert_eq!(active[0].id, v1_id);
        assert_eq!(active[0].subject.as_deref(), Some("current API surface"));

        // v2 was inserted for audit and immediately superseded into v1.
        let v2_row = sqlx::query!(
            r#"SELECT id, superseded_at, superseded_by FROM facts
               WHERE subject = 'thoughts in current API surface'"#,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(v2_row.superseded_at.is_some());
        assert_eq!(v2_row.superseded_by, Some(v1_id));
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

    // -- M3 Phase C: three-band routing, retraction durability,
    //                subsumption, quality-aware pick -----------------

    /// Hedged claims (0.70–0.85 confidence) commit with `flagged = true`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn three_band_routes_to_facts_flagged_true_when_below_min_confidence(pool: PgPool) {
        cap(&pool, "Hedged claim source", "global").await;
        let e = FakeExtractor::with_confidence(0.75);
        run_reflector_once(
            &pool,
            &e,
            TEST_MODEL_ID,
            &options_with(0.7, 0.85, SubsumptionKeep::Specific),
        )
        .await
        .unwrap();
        let rows = sqlx::query!(r#"SELECT flagged, confidence FROM facts WHERE superseded_at IS NULL"#)
            .fetch_all(&pool)
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].flagged, "middle-band confidence must land flagged=true");
    }

    /// Declarative claims (≥ 0.85) commit with `flagged = false`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn three_band_routes_to_facts_flagged_false_when_above(pool: PgPool) {
        cap(&pool, "Declarative claim source", "global").await;
        let e = FakeExtractor::with_confidence(0.92);
        run_reflector_once(
            &pool,
            &e,
            TEST_MODEL_ID,
            &options_with(0.7, 0.85, SubsumptionKeep::Specific),
        )
        .await
        .unwrap();
        let rows = sqlx::query!(r#"SELECT flagged FROM facts WHERE superseded_at IS NULL"#)
            .fetch_all(&pool)
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].flagged, "high-confidence must land flagged=false");
    }

    /// Kill-switch: setting `min_confidence_to_store == review_queue_below`
    /// collapses to two-band routing — every committed row gets
    /// `flagged = false` regardless of confidence.
    #[sqlx::test(migrations = "../../migrations")]
    async fn three_band_collapses_when_min_equals_review_below(pool: PgPool) {
        cap(&pool, "Source 1", "global").await;
        cap(&pool, "Source 2", "global").await;
        let opts = options_with(0.7, 0.7, SubsumptionKeep::Specific);
        // Hedged confidence (0.75) — would normally flag but the kill-switch
        // collapses the middle band.
        let e1 = FakeExtractor::with_confidence(0.75);
        run_reflector_once(&pool, &e1, TEST_MODEL_ID, &opts).await.unwrap();
        let rows = sqlx::query!(r#"SELECT flagged FROM facts WHERE superseded_at IS NULL"#)
            .fetch_all(&pool)
            .await
            .unwrap();
        assert!(
            rows.iter().all(|r| !r.flagged),
            "kill-switch must make every committed row flagged=false",
        );
    }

    /// Retraction-without-replacement: the operator `correct_fact`-retracts
    /// the trivia fact (no replacement), then a rerun re-emits both the
    /// retracted claim and a separate keeper claim. Phase C inheritance:
    /// the retracted claim's new emission lands superseded with
    /// `superseded_by = NULL`, matching the original retraction shape.
    /// The keeper claim stays active and keeps the thought visible to
    /// `find_facted_thoughts` so the rerun loop processes it.
    #[sqlx::test(migrations = "../../migrations")]
    async fn retraction_durability_inherits_no_replacement_on_rerun(pool: PgPool) {
        cap(&pool, "Trivia source", "global").await;
        let trivia = ExtractedFact {
            statement: "Trivia claim".into(),
            subject: Some("trivia".into()),
            predicate: Some("is".into()),
            object: Some("retracted".into()),
            confidence: 0.9,
        };
        let keeper = ExtractedFact {
            statement: "Keeper claim".into(),
            subject: Some("keeper".into()),
            predicate: Some("is".into()),
            object: Some("active".into()),
            confidence: 0.9,
        };
        let e = FakeExtractor::with_facts(vec![trivia.clone(), keeper.clone()]);
        run_reflector_once(&pool, &e, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        let trivia_id = sqlx::query!(
            r#"SELECT id FROM facts WHERE statement = 'Trivia claim'"#
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .id;
        // Operator retracts the trivia fact (no replacement).
        engram_storage::supersede_fact(&pool, trivia_id, None)
            .await
            .unwrap();

        // Rerun re-emits both claims (keeper keeps the thought in the
        // facted set).
        run_reflector_rerun(&pool, &e, TEST_MODEL_ID, &options(0.7), None)
            .await
            .unwrap();

        // Trivia claim must NOT reappear active.
        let trivia_active = sqlx::query!(
            r#"SELECT COUNT(*) AS "n!" FROM facts
               WHERE statement = 'Trivia claim' AND superseded_at IS NULL"#
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .n;
        assert_eq!(trivia_active, 0, "retracted claim must not reappear active");

        // A second trivia row exists (the rerun's emission) with
        // superseded_at set and superseded_by = NULL.
        let inherited = sqlx::query!(
            r#"SELECT superseded_at, superseded_by FROM facts
               WHERE statement = 'Trivia claim' AND id != $1"#,
            trivia_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(inherited.superseded_at.is_some());
        assert!(inherited.superseded_by.is_none());
    }

    /// Retraction-with-replacement: an old row was superseded into a
    /// `correct_fact` replacement carrying *different* SPO (operator
    /// reshaped the triple, not just the prose). A rerun re-extracts the
    /// old claim → the new emission inherits `superseded_by = canonical`
    /// so the operator's replacement stays the authoritative active row.
    #[sqlx::test(migrations = "../../migrations")]
    async fn retraction_durability_inherits_with_canonical_on_rerun(pool: PgPool) {
        let thought_id = cap(&pool, "anchor source", "global").await;
        let e_old = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "Original wording".into(),
            subject: Some("subj-old".into()),
            predicate: Some("rel-old".into()),
            object: Some("obj-old".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &e_old, TEST_MODEL_ID, &options(0.7))
            .await
            .unwrap();
        let old_id = sqlx::query!(
            r#"SELECT id FROM facts WHERE statement = 'Original wording'"#
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .id;

        // Insert a manual replacement with DIFFERENT SPO (so the rerun
        // emission doesn't trigger the triple-match rerun-reword path —
        // we want to exercise retraction-inherit specifically).
        let canonical_id = engram_storage::insert_fact(
            &pool,
            engram_storage::NewFact {
                scope: &Scope::new("global").unwrap(),
                statement: "Operator-corrected wording",
                subject: Some("subj-new"),
                predicate: Some("rel-new"),
                object: Some("obj-new"),
                source_thought_id: thought_id,
                extractor_model: "manual",
                extractor_version: 0,
                source_run_id: None,
                confidence: 1.0,
                flagged: false,
            },
        )
        .await
        .unwrap();
        engram_storage::supersede_fact(&pool, old_id, Some(canonical_id)).await.unwrap();

        // Rerun re-emits the old wording (same statement + same old SPO).
        // Since the only active row has different statement AND different
        // SPO, `find_matching_active_facts` is empty for this claim, and
        // `find_matching_superseded_facts` finds the old row with
        // `superseded_by = canonical_id`. Inheritance lands the new
        // emission as superseded into `canonical_id`.
        run_reflector_rerun(&pool, &e_old, TEST_MODEL_ID, &options(0.7), None)
            .await
            .unwrap();

        let active = sqlx::query!(r#"SELECT id FROM facts WHERE superseded_at IS NULL"#)
            .fetch_all(&pool)
            .await
            .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, canonical_id);

        // Verify the rerun's emission specifically inherited the canonical.
        let inherited = sqlx::query!(
            r#"SELECT superseded_at, superseded_by FROM facts
               WHERE statement = 'Original wording' AND id != $1"#,
            old_id,
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(inherited.len(), 1);
        assert!(inherited[0].superseded_at.is_some());
        assert_eq!(inherited[0].superseded_by, Some(canonical_id));
    }

    /// Default policy (`Specific`): when an existing active fact shares
    /// `(subject, predicate)` and one object refines the other, drop the
    /// more general. Regression: "Ron does not like Python" + "Ron does
    /// not like Python for enterprise software" — only the specific
    /// stays active.
    #[sqlx::test(migrations = "../../migrations")]
    async fn subsumption_keep_specific_drops_general(pool: PgPool) {
        cap(&pool, "Source on languages", "global").await;
        let opts = options_with(0.7, 0.85, SubsumptionKeep::Specific);

        // Seed the general fact first.
        let general = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "Ron does not like Python".into(),
            subject: Some("Ron".into()),
            predicate: Some("does not like".into()),
            object: Some("Python".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &general, TEST_MODEL_ID, &opts).await.unwrap();

        // Now emit the more-specific form via rerun.
        let specific = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "Ron does not like Python for enterprise software".into(),
            subject: Some("Ron".into()),
            predicate: Some("does not like".into()),
            object: Some("Python for enterprise software".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &specific, TEST_MODEL_ID, &opts, None)
            .await
            .unwrap();

        let active = sqlx::query!(
            r#"SELECT object FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object.as_deref(), Some("Python for enterprise software"));
    }

    /// `General` knob: invert the policy — drop the more specific, keep
    /// the general.
    #[sqlx::test(migrations = "../../migrations")]
    async fn subsumption_keep_general_drops_specific(pool: PgPool) {
        cap(&pool, "Source on languages", "global").await;
        let opts = options_with(0.7, 0.85, SubsumptionKeep::General);

        // Seed the general first.
        let general = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "Ron does not like Python".into(),
            subject: Some("Ron".into()),
            predicate: Some("does not like".into()),
            object: Some("Python".into()),
            confidence: 0.9,
        }]);
        run_reflector_once(&pool, &general, TEST_MODEL_ID, &opts).await.unwrap();

        // Specific emission lands → SubsumedByActive (general kept).
        let specific = FakeExtractor::with_facts(vec![ExtractedFact {
            statement: "Ron does not like Python for enterprise software".into(),
            subject: Some("Ron".into()),
            predicate: Some("does not like".into()),
            object: Some("Python for enterprise software".into()),
            confidence: 0.9,
        }]);
        run_reflector_rerun(&pool, &specific, TEST_MODEL_ID, &opts, None)
            .await
            .unwrap();

        let active = sqlx::query!(
            r#"SELECT object FROM facts WHERE superseded_at IS NULL"#
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object.as_deref(), Some("Python"));
    }

    /// Pure-function tests for `quality_aware_pick`'s tiebreakers. No DB
    /// needed; exercises the lex tuple directly.
    #[test]
    fn quality_aware_pick_prefers_s_not_o_over_self_referential() {
        let new = ExtractedFact {
            statement: "x".into(),
            subject: Some("a".into()),
            predicate: Some("p".into()),
            object: Some("a".into()), // self-referential
            confidence: 0.9,
        };
        let existing = Fact {
            id: uuid::Uuid::from_u128(1),
            scope: Scope::new("global").unwrap(),
            statement: "x".into(),
            subject: Some("a".into()),
            predicate: Some("p".into()),
            object: Some("b".into()), // not self-referential
            source_thought_id: ThoughtId::from(uuid::Uuid::nil()),
            extractor_model: "fake".into(),
            extractor_version: 1,
            source_run_id: None,
            confidence: 0.9,
            flagged: false,
            created_at: time::OffsetDateTime::now_utc(),
        };
        let existing_id = existing.id;
        match quality_aware_pick(&new, std::slice::from_ref(&existing)) {
            QualityWinner::Existing(id) => assert_eq!(id, existing_id),
            QualityWinner::New => panic!("expected existing to win (s != o); new is self-referential"),
        }
    }

    /// Regression: Bazel/Make/Nix dogfood. Existing fact with correct
    /// comparative SPO (subject before object in the statement) must beat
    /// a new emission with inverted SPO.
    #[test]
    fn quality_aware_pick_prefers_subject_and_object_tokens_in_statement() {
        let stmt = "Bazel is more powerful than Make";
        let new_inverted = ExtractedFact {
            statement: stmt.into(),
            subject: Some("Make".into()),
            predicate: Some("has a steeper learning curve than".into()),
            object: Some("Bazel".into()),
            confidence: 0.9,
        };
        let existing_correct = Fact {
            id: uuid::Uuid::from_u128(1),
            scope: Scope::new("global").unwrap(),
            statement: stmt.into(),
            subject: Some("Bazel".into()),
            predicate: Some("is more powerful than".into()),
            object: Some("Make".into()),
            source_thought_id: ThoughtId::from(uuid::Uuid::nil()),
            extractor_model: "fake".into(),
            extractor_version: 1,
            source_run_id: None,
            confidence: 0.9,
            flagged: false,
            created_at: time::OffsetDateTime::now_utc(),
        };
        let existing_id = existing_correct.id;
        match quality_aware_pick(&new_inverted, std::slice::from_ref(&existing_correct)) {
            QualityWinner::Existing(id) => assert_eq!(id, existing_id),
            QualityWinner::New => {
                panic!("expected existing (correct comparative SPO) to win over inverted new emission")
            }
        }
    }
}

//! Orchestration for `kengram eval tagger`: run every arm over every corpus
//! item (× repeats), score raw and finalized output against the golden
//! labels, and aggregate into a [`Report`].
//!
//! This module is DATABASE-FREE by design (see `eval/mod.rs`). Everything
//! the production pipeline would fetch from Postgres comes from the corpus
//! file: per-scope vocab snapshots feed `Tagger::tag`, and the corpus's
//! `known_scopes` feeds `finalize_tags` — the same deterministic
//! post-filter seam the worker drainer and `kengram tag` run, so the
//! headline scores measure what production would actually persist.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use kengram_core::{Metadata, Scope, TagKind, Tagger, Tags};
use kengram_mcp::finalize::finalize_tags;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use super::corpus::{Corpus, CorpusItem};
use super::report::{
    self, ArmReport, CallRecord, CallStats, FIELD_NAMES, FieldReport, KindReport, LatencyStats,
    MicroScores, PairwiseDelta, REPORT_FORMAT_VERSION, Report, RunInfo, StabilityReport, WorstItem,
};
use super::score::{
    self, ConfusionMatrix, Counts, KIND_LABELS, mean_pairwise_jaccard, modal_kind_agreement,
    percentile_nearest_rank, score_tags,
};

/// Number of worst-scoring items surfaced per arm as failure exemplars.
const WORST_ITEMS_PER_ARM: usize = 5;

/// One arm as actually run — a built arm plus the vocab on/off decision
/// (after `--vocab both` expansion, which duplicates arms with
/// `+vocab`/`-vocab` name suffixes).
pub(crate) struct RunArm {
    pub name: String,
    pub provider: String,
    pub endpoint: String,
    pub model_name: String,
    pub model_version: i32,
    pub prompt: String,
    pub vocab_on: bool,
    pub tagger: Arc<dyn Tagger>,
}

pub(crate) struct RunOpts {
    pub repeats: u32,
    pub concurrency: usize,
    pub corpus_path: String,
    pub vocab_mode: String,
    pub unreviewed_items_included: bool,
    /// Total items in the corpus file (pre-filter), for the report header.
    pub corpus_item_count: usize,
    /// When set, the report-so-far is flushed here after every completed
    /// arm (`complete: false`), so a crash or Ctrl-C keeps finished arms.
    pub flush_path: Option<std::path::PathBuf>,
    /// Abandon an arm after this many consecutive call failures (looping
    /// or dead endpoint guard). 0 disables. "Consecutive" is in completion
    /// order — exact at concurrency 1, approximate above it.
    pub max_consecutive_failures: u32,
}

/// Emit a progress line every this many resolved calls per arm.
const PROGRESS_EVERY: usize = 10;

fn fmt_eta(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m", s.div_ceil(60))
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

/// Run the full eval. `items` is the post-filter (reviewed / `--limit`)
/// item list.
pub(crate) async fn run_tagger_eval(
    corpus: &Corpus,
    items: &[CorpusItem],
    arms: &[RunArm],
    opts: &RunOpts,
) -> anyhow::Result<Report> {
    let started_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("formatting run timestamp")?;

    // Parse per-item scope + metadata once; a failure here is a corpus
    // usage error, better surfaced before any tagger call.
    let mut item_ctx: Vec<(Scope, Metadata)> = Vec::with_capacity(items.len());
    for item in items {
        let scope = Scope::new(item.scope.clone()).with_context(|| {
            format!(
                "corpus item {:?} has invalid scope {:?}",
                item.id, item.scope
            )
        })?;
        let metadata = Metadata::from(item.metadata.clone());
        item_ctx.push((scope, metadata));
    }

    let mut all_calls: Vec<CallRecord> = Vec::new();
    let mut arm_reports: Vec<ArmReport> = Vec::new();

    for (arm_idx, arm) in arms.iter().enumerate() {
        let label = format!("arm {}/{} {}", arm_idx + 1, arms.len(), arm.name);
        eprintln!(
            "[{label}] starting ({} call(s))",
            items.len() * opts.repeats as usize
        );
        let results = run_arm_calls(arm, items, corpus, opts, &label).await?;
        let (report, calls) = aggregate_arm(arm, items, &item_ctx, corpus, opts, results);
        arm_reports.push(report);
        all_calls.extend(calls);

        // Flush the report-so-far so an overnight crash keeps every
        // finished arm. The final (complete: true) write below overwrites.
        if let Some(path) = &opts.flush_path {
            let partial = build_report(
                &started_at,
                opts,
                items.len(),
                arm_reports.clone(),
                all_calls.clone(),
                false,
            );
            report::write_report(&partial, path).context("flushing partial report")?;
            eprintln!(
                "[{label}] arm complete — report-so-far flushed to {}",
                path.display()
            );
        }
    }

    Ok(build_report(
        &started_at,
        opts,
        items.len(),
        arm_reports,
        all_calls,
        true,
    ))
}

/// Assemble a report from the arms completed so far. `complete: false`
/// marks an incremental flush; `true` the final report.
fn build_report(
    started_at: &str,
    opts: &RunOpts,
    scored_item_count: usize,
    arms: Vec<ArmReport>,
    calls: Vec<CallRecord>,
    complete: bool,
) -> Report {
    let pairwise = pairwise_deltas(&arms);
    Report {
        format_version: REPORT_FORMAT_VERSION,
        run: RunInfo {
            started_at: started_at.to_string(),
            corpus_path: opts.corpus_path.clone(),
            corpus_item_count: opts.corpus_item_count,
            scored_item_count,
            unreviewed_items_included: opts.unreviewed_items_included,
            repeats: opts.repeats,
            vocab_mode: opts.vocab_mode.clone(),
            kengram_version: env!("CARGO_PKG_VERSION").to_string(),
            complete,
        },
        arms,
        pairwise,
        calls,
    }
}

/// Raw outcome of one tagger call, before scoring.
struct CallOutcome {
    item_idx: usize,
    repeat: u32,
    latency_ms: f64,
    /// Raw tags on success; stringified `TaggerError` on failure.
    result: Result<Tags, String>,
}

/// Everything one arm's call phase produced: resolved outcomes plus
/// whether (and why) the consecutive-failure guard abandoned the arm.
struct ArmCallResults {
    outcomes: Vec<CallOutcome>,
    aborted: Option<String>,
    skipped: usize,
}

/// Fire item × repeat calls for one arm with bounded concurrency,
/// emitting progress to stderr and enforcing the consecutive-failure
/// guard. Results are sorted back into (item, repeat) order so reports
/// are deterministic regardless of completion order.
async fn run_arm_calls(
    arm: &RunArm,
    items: &[CorpusItem],
    corpus: &Corpus,
    opts: &RunOpts,
    label: &str,
) -> anyhow::Result<ArmCallResults> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(opts.concurrency.max(1)));
    let mut join = tokio::task::JoinSet::new();

    for (item_idx, item) in items.iter().enumerate() {
        let vocab = if arm.vocab_on {
            corpus.vocab_for(&item.scope)
        } else {
            None
        };
        for repeat in 0..opts.repeats {
            let semaphore = semaphore.clone();
            let tagger = arm.tagger.clone();
            let content = item.content.clone();
            let vocab = vocab.clone();
            join.spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .expect("eval semaphore closed unexpectedly");
                let t0 = Instant::now();
                let result = tagger.tag(&content, vocab.as_ref()).await;
                CallOutcome {
                    item_idx,
                    repeat,
                    latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    result: result.map(|o| o.tags).map_err(|e| e.to_string()),
                }
            });
        }
    }

    let planned = items.len() * opts.repeats as usize;
    let mut outcomes = Vec::with_capacity(planned);
    let mut failed_total = 0usize;
    let mut consecutive_failures = 0u32;
    let mut latency_sum_ms = 0.0_f64;
    let mut aborted: Option<String> = None;

    while let Some(joined) = join.join_next().await {
        let outcome = match joined {
            Ok(o) => o,
            // abort_all() cancellations are expected during fail-fast.
            Err(e) if e.is_cancelled() => continue,
            Err(e) => return Err(e).context("eval tagger task panicked"),
        };
        latency_sum_ms += outcome.latency_ms;
        if outcome.result.is_err() {
            failed_total += 1;
            consecutive_failures += 1;
        } else {
            consecutive_failures = 0;
        }
        outcomes.push(outcome);

        let done = outcomes.len();
        if done % PROGRESS_EVERY == 0 || done == planned {
            let avg_s = latency_sum_ms / done as f64 / 1000.0;
            let eta_s = (planned - done) as f64 * avg_s / opts.concurrency.max(1) as f64;
            let fail_pct = if done > 0 {
                100.0 * failed_total as f64 / done as f64
            } else {
                0.0
            };
            eprintln!(
                "[{label}] {done}/{planned} done · {failed_total} failed total ({fail_pct:.1}%) · avg {avg_s:.1}s/call · ETA {}",
                fmt_eta(eta_s)
            );
        }

        if opts.max_consecutive_failures > 0
            && consecutive_failures >= opts.max_consecutive_failures
        {
            aborted = Some(format!(
                "{consecutive_failures} consecutive call failures ({done}/{planned} calls resolved)"
            ));
            eprintln!(
                "[{label}] ABORTING ARM: {} — skipping its remaining calls",
                aborted.as_deref().unwrap_or_default()
            );
            join.abort_all();
            // Drain: keep outcomes that finished before the abort landed,
            // swallow the cancellations.
            while let Some(joined) = join.join_next().await {
                match joined {
                    Ok(o) => outcomes.push(o),
                    Err(e) if e.is_cancelled() => {}
                    Err(e) => return Err(e).context("eval tagger task panicked"),
                }
            }
            break;
        }
    }

    let skipped = planned - outcomes.len();
    outcomes.sort_by_key(|o| (o.item_idx, o.repeat));
    Ok(ArmCallResults {
        outcomes,
        aborted,
        skipped,
    })
}

fn field_of<'a>(tags: &'a Tags, name: &str) -> &'a [String] {
    match name {
        "people" => &tags.people,
        "entities" => &tags.entities,
        "action_items" => &tags.action_items,
        "topics" => &tags.topics,
        "dates_mentioned" => &tags.dates_mentioned,
        other => unreachable!("unknown tag field {other}"),
    }
}

fn score_field(name: &str, predicted: &Tags, golden: &Tags) -> Counts {
    if name == "action_items" {
        score::score_action_items(&predicted.action_items, &golden.action_items)
    } else {
        score::score_set_field(field_of(predicted, name), field_of(golden, name))
    }
}

#[derive(Default)]
struct FieldAgg {
    finalized: Counts,
    raw: Counts,
    per_call_f1: Vec<f64>,
}

/// Score one arm's outcomes and build its report section plus the per-call
/// audit records.
fn aggregate_arm(
    arm: &RunArm,
    items: &[CorpusItem],
    item_ctx: &[(Scope, Metadata)],
    corpus: &Corpus,
    opts: &RunOpts,
    results: ArmCallResults,
) -> (ArmReport, Vec<CallRecord>) {
    let ArmCallResults {
        outcomes,
        aborted,
        skipped,
    } = results;
    let resolved = outcomes.len();
    let mut fields: BTreeMap<&str, FieldAgg> = FIELD_NAMES
        .iter()
        .map(|n| (*n, FieldAgg::default()))
        .collect();
    let mut confusion = ConfusionMatrix::default();
    let mut latencies: Vec<f64> = Vec::new();
    let mut failed = 0usize;
    let mut calls: Vec<CallRecord> = Vec::new();

    // Per-item accumulators for stability + worst-item ranking.
    let mut per_item_kinds: Vec<Vec<Option<TagKind>>> = vec![Vec::new(); items.len()];
    let mut per_item_field_sets: Vec<BTreeMap<&str, Vec<std::collections::BTreeSet<String>>>> =
        vec![FIELD_NAMES.iter().map(|n| (*n, Vec::new())).collect(); items.len()];
    let mut per_item_mean_f1s: Vec<Vec<f64>> = vec![Vec::new(); items.len()];
    let mut per_item_first_prediction: Vec<Option<Tags>> = vec![None; items.len()];

    for outcome in outcomes {
        let item = &items[outcome.item_idx];
        let (scope, metadata) = &item_ctx[outcome.item_idx];
        latencies.push(outcome.latency_ms);

        match outcome.result {
            Err(error) => {
                failed += 1;
                calls.push(CallRecord {
                    arm: arm.name.clone(),
                    item_id: item.id.clone(),
                    repeat: outcome.repeat,
                    vocab_used: arm.vocab_on,
                    latency_ms: outcome.latency_ms,
                    raw_tags: None,
                    finalized_tags: None,
                    error: Some(error),
                });
            }
            Ok(raw) => {
                // The exact production persistence shape: run the shared
                // deterministic post-filter seam on a copy of the raw output.
                let mut finalized = raw.clone();
                let vocab = if arm.vocab_on {
                    corpus.vocab_for(&item.scope)
                } else {
                    None
                };
                finalize_tags(
                    &mut finalized,
                    metadata,
                    scope,
                    vocab.as_ref(),
                    &corpus.known_scopes,
                );

                let fin_scores = score_tags(&finalized, &item.golden);
                for name in FIELD_NAMES {
                    let agg = fields.get_mut(name).expect("all fields pre-seeded");
                    let fin = score_field(name, &finalized, &item.golden);
                    agg.finalized.add(fin);
                    agg.per_call_f1.push(fin.f1());
                    agg.raw.add(score_field(name, &raw, &item.golden));
                }
                confusion.add(item.golden.kind, finalized.kind);

                per_item_kinds[outcome.item_idx].push(finalized.kind);
                for name in FIELD_NAMES {
                    per_item_field_sets[outcome.item_idx]
                        .get_mut(name)
                        .expect("pre-seeded")
                        .push(score::norm_set(field_of(&finalized, name)));
                }
                per_item_mean_f1s[outcome.item_idx].push(fin_scores.mean_f1());
                per_item_first_prediction[outcome.item_idx].get_or_insert(finalized.clone());

                calls.push(CallRecord {
                    arm: arm.name.clone(),
                    item_id: item.id.clone(),
                    repeat: outcome.repeat,
                    vocab_used: arm.vocab_on,
                    latency_ms: outcome.latency_ms,
                    raw_tags: Some(raw),
                    finalized_tags: Some(finalized),
                    error: None,
                });
            }
        }
    }

    let succeeded = resolved - failed;

    let field_reports: BTreeMap<String, FieldReport> = fields
        .iter()
        .map(|(name, agg)| {
            let micro = MicroScores {
                precision: agg.finalized.precision(),
                recall: agg.finalized.recall(),
                f1: agg.finalized.f1(),
                tp: agg.finalized.tp,
                fp: agg.finalized.fp,
                fn_count: agg.finalized.fn_count,
            };
            let macro_f1 = if agg.per_call_f1.is_empty() {
                0.0
            } else {
                agg.per_call_f1.iter().sum::<f64>() / agg.per_call_f1.len() as f64
            };
            (
                name.to_string(),
                FieldReport {
                    micro,
                    macro_f1,
                    raw_micro_f1: agg.raw.f1(),
                    finalize_delta: micro.f1 - agg.raw.f1(),
                },
            )
        })
        .collect();

    let stability = (opts.repeats >= 2).then(|| {
        let mut agreements = Vec::new();
        let mut unanimous = 0usize;
        let mut unstable_items = Vec::new();
        let mut jaccard_sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for (idx, kinds) in per_item_kinds.iter().enumerate() {
            if kinds.is_empty() {
                continue; // every repeat failed for this item
            }
            let (_, agreement) = modal_kind_agreement(kinds);
            agreements.push(agreement);
            if agreement >= 1.0 {
                unanimous += 1;
            } else {
                unstable_items.push(items[idx].id.clone());
            }
            for name in FIELD_NAMES {
                let sets = &per_item_field_sets[idx][name];
                let entry = jaccard_sums.entry(name.to_string()).or_insert((0.0, 0));
                entry.0 += mean_pairwise_jaccard(sets);
                entry.1 += 1;
            }
        }
        let n = agreements.len().max(1) as f64;
        StabilityReport {
            mean_kind_agreement: agreements.iter().sum::<f64>() / n,
            unanimous_kind_pct: unanimous as f64 / n,
            mean_pairwise_jaccard: jaccard_sums
                .into_iter()
                .map(|(k, (sum, count))| (k, sum / count.max(1) as f64))
                .collect(),
            unstable_items,
        }
    });

    // Worst items by mean (over repeats) of per-call mean field F1.
    let mut ranked: Vec<(f64, usize)> = per_item_mean_f1s
        .iter()
        .enumerate()
        .filter(|(_, f1s)| !f1s.is_empty())
        .map(|(idx, f1s)| (f1s.iter().sum::<f64>() / f1s.len() as f64, idx))
        .collect();
    ranked.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| items[a.1].id.cmp(&items[b.1].id))
    });
    let worst_items: Vec<WorstItem> = ranked
        .into_iter()
        .take(WORST_ITEMS_PER_ARM)
        .map(|(mean_f1, idx)| WorstItem {
            id: items[idx].id.clone(),
            mean_f1,
            golden: items[idx].golden.clone(),
            predicted: per_item_first_prediction[idx].clone().unwrap_or_default(),
        })
        .collect();

    let mean_latency = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };

    let report = ArmReport {
        name: arm.name.clone(),
        provider: arm.provider.clone(),
        endpoint: arm.endpoint.clone(),
        model_name: arm.model_name.clone(),
        model_version: arm.model_version,
        prompt: arm.prompt.clone(),
        vocab_used: arm.vocab_on,
        aborted,
        calls: CallStats {
            attempted: resolved,
            succeeded,
            failed,
            skipped,
            failure_rate: if resolved == 0 {
                0.0
            } else {
                failed as f64 / resolved as f64
            },
        },
        latency_ms: LatencyStats {
            p50: percentile_nearest_rank(&latencies, 50.0),
            p95: percentile_nearest_rank(&latencies, 95.0),
            mean: mean_latency,
        },
        kind: KindReport {
            accuracy: confusion.accuracy(),
            labels: KIND_LABELS.iter().map(|s| s.to_string()).collect(),
            matrix: confusion.to_rows(),
        },
        fields: field_reports,
        stability,
        worst_items,
    };
    (report, calls)
}

fn pairwise_deltas(arms: &[ArmReport]) -> Vec<PairwiseDelta> {
    let mut deltas = Vec::new();
    for i in 0..arms.len() {
        for j in (i + 1)..arms.len() {
            let (a, b) = (&arms[i], &arms[j]);
            deltas.push(PairwiseDelta {
                a: a.name.clone(),
                b: b.name.clone(),
                kind_accuracy_delta: a.kind.accuracy - b.kind.accuracy,
                micro_f1_delta: FIELD_NAMES
                    .iter()
                    .map(|name| {
                        let fa = a.fields.get(*name).map(|f| f.micro.f1).unwrap_or(0.0);
                        let fb = b.fields.get(*name).map(|f| f.micro.f1).unwrap_or(0.0);
                        (name.to_string(), fa - fb)
                    })
                    .collect(),
            });
        }
    }
    deltas
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::TagOutput;
    use kengram_extract::{FakeBehavior, FakeTagger};

    fn corpus_4_items() -> Corpus {
        serde_json::from_str(
            r#"{
              "format_version": 1,
              "kind": "tagger-golden-corpus",
              "known_scopes": ["demo.eval"],
              "scope_vocab": { "demo.eval": { "topics": ["tagging"], "entities": ["kengram"] } },
              "items": [
                { "id": "i1", "content": "Sarah will ship kengram tagging tomorrow.",
                  "scope": "demo.eval", "reviewed": true,
                  "golden": { "people": ["Sarah"], "entities": ["kengram"],
                              "action_items": ["ship kengram tagging"],
                              "topics": ["tagging"], "dates_mentioned": ["tomorrow"],
                              "kind": "task" } },
                { "id": "i2", "content": "Observation about nothing in particular.",
                  "scope": "demo.eval", "reviewed": true,
                  "golden": { "people": [], "entities": [], "action_items": [],
                              "topics": [], "dates_mentioned": [], "kind": "observation" } },
                { "id": "i3", "content": "We decided to keep RRF.",
                  "scope": "demo.eval", "reviewed": true,
                  "metadata": { "decision_type": "architecture" },
                  "golden": { "people": [], "entities": [], "action_items": [],
                              "topics": [], "dates_mentioned": [], "kind": "decision_record" } },
                { "id": "i4", "content": "Maria likes async reviews.",
                  "scope": "demo.eval", "reviewed": true,
                  "golden": { "people": ["Maria"], "entities": [], "action_items": [],
                              "topics": [], "dates_mentioned": [], "kind": "person_note" } }
              ]
            }"#,
        )
        .unwrap()
    }

    fn opts(repeats: u32) -> RunOpts {
        RunOpts {
            repeats,
            concurrency: 2,
            corpus_path: "test".to_string(),
            vocab_mode: "on".to_string(),
            unreviewed_items_included: false,
            corpus_item_count: 4,
            flush_path: None,
            max_consecutive_failures: 0, // guard off: legacy-behavior tests
        }
    }

    fn arm(name: &str, tagger: FakeTagger, vocab_on: bool) -> RunArm {
        RunArm {
            name: name.to_string(),
            provider: "fake".to_string(),
            endpoint: "fake://".to_string(),
            model_name: "fake".to_string(),
            model_version: 0,
            prompt: "bundled".to_string(),
            vocab_on,
            tagger: Arc::new(tagger),
        }
    }

    fn tags(people: &[&str], kind: Option<TagKind>) -> Tags {
        Tags {
            people: people.iter().map(|s| s.to_string()).collect(),
            kind,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn deterministic_fake_arm_scores_and_is_fully_stable() {
        // Substring-keyed fake: i1 gets the right person+kind; everything
        // else gets empty observation tags.
        let exact_i1 = Tags {
            people: vec!["Sarah".to_string()],
            entities: vec!["kengram".to_string()],
            action_items: vec!["ship kengram tagging".to_string()],
            topics: vec!["tagging".to_string()],
            dates_mentioned: vec!["tomorrow".to_string()],
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        let fake = FakeTagger::with_substring(vec![
            (
                "Sarah".to_string(),
                TagOutput {
                    tags: exact_i1,
                    relations: vec![],
                },
            ),
            (
                "Maria".to_string(),
                TagOutput {
                    tags: tags(&["Maria"], Some(TagKind::PersonNote)),
                    relations: vec![],
                },
            ),
        ]);
        let corpus = corpus_4_items();
        let arms = vec![arm("det", fake, true)];
        let report = run_tagger_eval(&corpus, &corpus.items, &arms, &opts(3))
            .await
            .unwrap();

        let a = &report.arms[0];
        assert_eq!(a.calls.attempted, 12);
        assert_eq!(a.calls.failed, 0);
        // people: i1 TP=1 (x3 repeats), i4 TP=1 (x3); i2/i3 both-empty.
        let people = &a.fields["people"];
        assert_eq!(people.micro.tp, 6);
        assert_eq!(people.micro.fp, 0);
        assert_eq!(people.micro.f1, 1.0);
        // Deterministic tagger => perfect stability across 3 repeats.
        let stab = a.stability.as_ref().expect("repeats=3 => stability");
        assert_eq!(stab.mean_kind_agreement, 1.0);
        assert_eq!(stab.unanimous_kind_pct, 1.0);
        assert!(stab.unstable_items.is_empty());
        // i3's metadata.decision_type override fires in finalize_tags even
        // though the fake returned no kind: substring rules don't match i2
        // and i3 content -> default Empty tags -> kind None -> override sets
        // decision_record. Kind accuracy: i1 task ok, i2 observation vs None
        // miss, i3 decision_record ok (via override), i4 person_note ok.
        assert!((a.kind.accuracy - 0.75).abs() < 1e-9);
        // Per-call audit recorded for every call.
        assert_eq!(report.calls.len(), 12);
        assert!(report.calls.iter().all(|c| c.error.is_none()));
    }

    #[tokio::test]
    async fn finalize_delta_visible_when_person_duplicated_into_entities() {
        // Fake emits Maria in BOTH people and entities. finalize_tags's
        // disjointness filter strips the entity copy, so finalized entities
        // is empty (golden) while raw has 1 FP -> positive finalize_delta.
        let mut t = tags(&["Maria"], Some(TagKind::PersonNote));
        t.entities = vec!["Maria".to_string()];
        let fake = FakeTagger::with_canned(t);
        let corpus = corpus_4_items();
        let items = vec![corpus.items[3].clone()]; // i4 only
        let arms = vec![arm("dup", fake, true)];
        let report = run_tagger_eval(&corpus, &items, &arms, &opts(1))
            .await
            .unwrap();

        let entities = &report.arms[0].fields["entities"];
        assert_eq!(entities.micro.f1, 1.0, "finalized output should be clean");
        assert!(entities.raw_micro_f1 < 1.0, "raw output had the duplicate");
        assert!(entities.finalize_delta > 0.0);
    }

    #[tokio::test]
    async fn failing_arm_reports_full_failure_rate_and_run_survives() {
        let dead = FakeTagger::always_failing(FakeBehavior::Unreachable);
        let ok = FakeTagger::with_canned(tags(&[], Some(TagKind::Observation)));
        let corpus = corpus_4_items();
        let arms = vec![arm("dead", dead, true), arm("ok", ok, true)];
        let report = run_tagger_eval(&corpus, &corpus.items, &arms, &opts(1))
            .await
            .unwrap();

        let dead_arm = &report.arms[0];
        assert_eq!(dead_arm.calls.failure_rate, 1.0);
        assert_eq!(dead_arm.calls.failed, 4);
        // Guard disabled in opts(): no abort, nothing skipped.
        assert!(dead_arm.aborted.is_none());
        assert_eq!(dead_arm.calls.skipped, 0);
        assert!(
            report
                .calls
                .iter()
                .filter(|c| c.arm == "dead")
                .all(|c| c.error.is_some())
        );
        // The healthy arm still scored.
        assert_eq!(report.arms[1].calls.failed, 0);
        // Pairwise section covers the pair.
        assert_eq!(report.pairwise.len(), 1);
    }

    #[tokio::test]
    async fn worst_items_rank_lowest_scoring_first() {
        // Canned tags match only i4 well; i1 (rich golden labels) scores worst.
        let fake = FakeTagger::with_canned(tags(&["Maria"], Some(TagKind::PersonNote)));
        let corpus = corpus_4_items();
        let arms = vec![arm("canned", fake, true)];
        let report = run_tagger_eval(&corpus, &corpus.items, &arms, &opts(1))
            .await
            .unwrap();
        let worst = &report.arms[0].worst_items;
        assert!(!worst.is_empty());
        assert_eq!(
            worst[0].id, "i1",
            "the rich-labeled item should score worst"
        );
        assert!(worst[0].mean_f1 < worst.last().unwrap().mean_f1);
    }

    /// A tagger that fails slowly — unlike `FakeTagger`, which resolves
    /// instantly and therefore finishes every spawned task before the
    /// consecutive-failure guard can cancel anything. The sleep keeps
    /// not-yet-started calls pending at abort time, like a real looping
    /// HTTP backend.
    struct SlowFailTagger;

    #[async_trait::async_trait]
    impl Tagger for SlowFailTagger {
        fn model_id(&self) -> &str {
            "test/slow-fail"
        }
        fn version(&self) -> i32 {
            0
        }
        async fn tag(
            &self,
            _content: &str,
            _vocab: Option<&kengram_core::ScopeVocab>,
        ) -> Result<kengram_core::TagOutput, kengram_core::TaggerError> {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            Err(kengram_core::TaggerError::Unreachable(
                "slow fake failure".into(),
            ))
        }
    }

    #[tokio::test]
    async fn consecutive_failure_guard_aborts_arm_and_spares_the_rest() {
        // 4 items x 3 repeats = 12 planned calls; guard fires at 3
        // consecutive failures. concurrency 1 makes "consecutive" exact and
        // keeps calls 4..12 pending (semaphore) when the abort lands.
        let ok = FakeTagger::with_canned(tags(&[], Some(TagKind::Observation)));
        let corpus = corpus_4_items();
        let arms = vec![
            RunArm {
                name: "looping".to_string(),
                provider: "fake".to_string(),
                endpoint: "fake://".to_string(),
                model_name: "fake".to_string(),
                model_version: 0,
                prompt: "bundled".to_string(),
                vocab_on: true,
                tagger: Arc::new(SlowFailTagger),
            },
            arm("ok", ok, true),
        ];
        let mut o = opts(3);
        o.concurrency = 1;
        o.max_consecutive_failures = 3;
        let report = run_tagger_eval(&corpus, &corpus.items, &arms, &o)
            .await
            .unwrap();

        let aborted_arm = &report.arms[0];
        assert!(
            aborted_arm
                .aborted
                .as_deref()
                .unwrap_or_default()
                .contains("3 consecutive"),
            "abort reason should name the threshold: {:?}",
            aborted_arm.aborted
        );
        assert_eq!(aborted_arm.calls.attempted, 3, "exactly N resolved calls");
        assert_eq!(aborted_arm.calls.failed, 3);
        assert_eq!(aborted_arm.calls.skipped, 9);
        assert_eq!(aborted_arm.calls.failure_rate, 1.0);
        // Audit trail covers only resolved calls.
        assert_eq!(
            report.calls.iter().filter(|c| c.arm == "looping").count(),
            3
        );
        // The healthy arm ran in full and is untouched by the abort.
        let ok_arm = &report.arms[1];
        assert!(ok_arm.aborted.is_none());
        assert_eq!(ok_arm.calls.attempted, 12);
        assert_eq!(ok_arm.calls.skipped, 0);
        assert_eq!(ok_arm.calls.failed, 0);
    }

    #[tokio::test]
    async fn per_arm_flush_writes_partial_reports_and_final_is_complete() {
        let a = FakeTagger::with_canned(tags(&["Maria"], Some(TagKind::PersonNote)));
        let b = FakeTagger::with_canned(tags(&[], Some(TagKind::Observation)));
        let corpus = corpus_4_items();
        let arms = vec![arm("a", a, true), arm("b", b, true)];
        let dir = std::env::temp_dir().join("kengram-eval-flush-tests");
        let _ = std::fs::remove_dir_all(&dir);
        let flush_path = dir.join("report.json");
        let mut o = opts(1);
        o.flush_path = Some(flush_path.clone());

        let report = run_tagger_eval(&corpus, &corpus.items, &arms, &o)
            .await
            .unwrap();

        // The returned (final) report is complete.
        assert!(report.run.complete);
        assert_eq!(report.arms.len(), 2);
        // The on-disk file is the last per-arm flush: both arms present,
        // marked incomplete (the CLI's final write_report overwrites it).
        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&flush_path).unwrap()).unwrap();
        assert_eq!(on_disk["run"]["complete"], serde_json::json!(false));
        assert_eq!(on_disk["arms"].as_array().unwrap().len(), 2);
        assert_eq!(on_disk["format_version"], serde_json::json!(1));
    }

    #[test]
    fn partial_report_builder_marks_incomplete_and_scopes_pairwise() {
        let o = opts(1);
        let partial = build_report("2026-06-11T00:00:00Z", &o, 4, vec![], vec![], false);
        assert!(!partial.run.complete);
        assert!(partial.arms.is_empty());
        assert!(partial.pairwise.is_empty());
        let fin = build_report("2026-06-11T00:00:00Z", &o, 4, vec![], vec![], true);
        assert!(fin.run.complete);
    }

    #[test]
    fn fmt_eta_buckets() {
        assert_eq!(fmt_eta(42.0), "42s");
        assert_eq!(fmt_eta(90.0), "2m");
        assert_eq!(fmt_eta(3700.0), "1h01m");
        assert_eq!(fmt_eta(-5.0), "0s");
    }
}

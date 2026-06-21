//! Report shapes + JSON writer + terminal renderer for the tagger eval
//! harness.
//!
//! The JSON report is the artifact of record: per-arm aggregates AND the
//! full per-call audit trail (raw + finalized tags per call), so every
//! aggregate number is re-derivable from the report alone. Field order is
//! stable (serde struct order; `BTreeMap` for keyed maps; explicit sorts in
//! the builder) so diffs between runs are meaningful.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use kengram_core::Tags;
use serde::Serialize;

pub const REPORT_FORMAT_VERSION: u32 = 1;

/// The five scored tag fields, in fixed report order.
pub const FIELD_NAMES: [&str; 5] = [
    "people",
    "entities",
    "action_items",
    "topics",
    "dates_mentioned",
];

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub format_version: u32,
    pub run: RunInfo,
    pub arms: Vec<ArmReport>,
    pub pairwise: Vec<PairwiseDelta>,
    pub calls: Vec<CallRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunInfo {
    pub started_at: String,
    pub corpus_path: String,
    pub corpus_item_count: usize,
    /// Items actually scored (after the reviewed filter and `--limit`).
    pub scored_item_count: usize,
    /// True when `--include-unreviewed` pulled draft-labeled items into the
    /// run — numbers from such a run are not golden.
    pub unreviewed_items_included: bool,
    pub repeats: u32,
    pub vocab_mode: String,
    pub kengram_version: String,
    /// False in the incremental per-arm flushes written while the run is
    /// still in progress; true only in the final report. A `complete:
    /// false` report on disk means the run crashed or was aborted — the
    /// arms it contains are still fully valid.
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArmReport {
    pub name: String,
    pub provider: String,
    pub endpoint: String,
    pub model_name: String,
    pub model_version: i32,
    /// `"bundled"` or `"file:<path>"` for a prompt override.
    pub prompt: String,
    pub vocab_used: bool,
    /// `Some(reason)` when the consecutive-failure guard abandoned this
    /// arm mid-run (looping or dead endpoint). Quality numbers cover only
    /// the resolved calls.
    pub aborted: Option<String>,
    pub calls: CallStats,
    pub latency_ms: LatencyStats,
    pub kind: KindReport,
    /// Keyed by field name; BTreeMap for stable key order.
    pub fields: BTreeMap<String, FieldReport>,
    /// Present only when `--repeats N` with N >= 2.
    pub stability: Option<StabilityReport>,
    /// Bottom items by mean field F1 — failure exemplars for eyeballing.
    pub worst_items: Vec<WorstItem>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CallStats {
    /// Calls actually dispatched and resolved (success or failure). Equals
    /// the planned call count unless the arm was aborted by the
    /// consecutive-failure guard.
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    /// Planned calls never dispatched because the arm was aborted.
    pub skipped: usize,
    /// failed / attempted (resolved calls only; skipped calls excluded).
    pub failure_rate: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct LatencyStats {
    pub p50: f64,
    pub p95: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KindReport {
    pub accuracy: f64,
    pub labels: Vec<String>,
    /// rows = golden, cols = predicted, order = `labels`.
    pub matrix: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MicroScores {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_count: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FieldReport {
    /// Headline numbers — computed on FINALIZED output (what production
    /// would persist).
    pub micro: MicroScores,
    pub macro_f1: f64,
    /// Micro F1 of the raw (pre-finalize) output, for the delta story.
    pub raw_micro_f1: f64,
    /// `micro.f1 - raw_micro_f1`: what the deterministic post-filters
    /// bought (or cost) on this field for this arm.
    pub finalize_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StabilityReport {
    pub mean_kind_agreement: f64,
    pub unanimous_kind_pct: f64,
    /// Mean pairwise Jaccard per field, keyed by field name.
    pub mean_pairwise_jaccard: BTreeMap<String, f64>,
    /// Item ids whose kind was not unanimous across repeats.
    pub unstable_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorstItem {
    pub id: String,
    pub mean_f1: f64,
    pub golden: Tags,
    pub predicted: Tags,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairwiseDelta {
    pub a: String,
    pub b: String,
    /// `accuracy(a) - accuracy(b)`.
    pub kind_accuracy_delta: f64,
    /// `micro_f1(a) - micro_f1(b)` per field (finalized).
    pub micro_f1_delta: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CallRecord {
    pub arm: String,
    pub item_id: String,
    pub repeat: u32,
    pub vocab_used: bool,
    pub latency_ms: f64,
    /// Raw tagger output. `None` when the call errored.
    pub raw_tags: Option<Tags>,
    /// Output after `finalize_tags` (the production persistence shape).
    pub finalized_tags: Option<Tags>,
    pub error: Option<String>,
}

/// Write the report as pretty JSON, creating parent directories as needed.
pub fn write_report(report: &Report, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating report directory {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(report).context("serializing report")?;
    std::fs::write(path, json).with_context(|| format!("writing report to {}", path.display()))?;
    Ok(())
}

/// Human-readable per-arm summary table + worst-exemplar section.
pub fn render_table(report: &Report) -> String {
    let mut out = String::new();
    if !report.run.complete {
        out.push_str("[PARTIAL — run did not finish; arms below are complete and valid]\n");
    }
    out.push_str(&format!(
        "Tagger eval — {} item(s), repeats={}, vocab={}{}\n\n",
        report.run.scored_item_count,
        report.run.repeats,
        report.run.vocab_mode,
        if report.run.unreviewed_items_included {
            "  [INCLUDES UNREVIEWED DRAFT LABELS]"
        } else {
            ""
        },
    ));

    out.push_str(&format!(
        "{:<24} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12} {:>6}\n",
        "arm",
        "kind",
        "people",
        "entity",
        "action",
        "topics",
        "dates",
        "stab",
        "p50/p95 ms",
        "fail%"
    ));
    for arm in &report.arms {
        let f = |name: &str| -> String {
            arm.fields
                .get(name)
                .map(|fr| format!("{:.3}", fr.micro.f1))
                .unwrap_or_else(|| "-".to_string())
        };
        let stab = arm
            .stability
            .as_ref()
            .map(|s| format!("{:.2}", s.mean_kind_agreement))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "{:<24} {:>6.3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5.0}/{:<6.0} {:>6.1}{}\n",
            arm.name,
            arm.kind.accuracy,
            f("people"),
            f("entities"),
            f("action_items"),
            f("topics"),
            f("dates_mentioned"),
            stab,
            arm.latency_ms.p50,
            arm.latency_ms.p95,
            arm.calls.failure_rate * 100.0,
            if arm.aborted.is_some() {
                "  ABORTED"
            } else {
                ""
            },
        ));
    }

    for arm in &report.arms {
        if let Some(reason) = &arm.aborted {
            out.push_str(&format!(
                "\nABORTED {} — {} ({} call(s) skipped)\n",
                arm.name, reason, arm.calls.skipped
            ));
        }
    }

    for arm in &report.arms {
        if arm.worst_items.is_empty() {
            continue;
        }
        out.push_str(&format!("\nworst items — {}:\n", arm.name));
        for w in &arm.worst_items {
            out.push_str(&format!("  {} (mean F1 {:.3})\n", w.id, w.mean_f1));
            out.push_str(&format!(
                "    golden:    {}\n    predicted: {}\n",
                compact_tags(&w.golden),
                compact_tags(&w.predicted)
            ));
        }
    }

    out
}

fn compact_tags(tags: &Tags) -> String {
    serde_json::to_string(tags).unwrap_or_else(|_| "<unserializable>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::TagKind;

    fn fixed_report() -> Report {
        let micro = MicroScores {
            precision: 0.9,
            recall: 0.8,
            f1: 0.847,
            tp: 8,
            fp: 1,
            fn_count: 2,
        };
        let field = FieldReport {
            micro,
            macro_f1: 0.81,
            raw_micro_f1: 0.8,
            finalize_delta: 0.047,
        };
        let mut fields = BTreeMap::new();
        for name in FIELD_NAMES {
            fields.insert(name.to_string(), field);
        }
        Report {
            format_version: REPORT_FORMAT_VERSION,
            run: RunInfo {
                started_at: "2026-06-10T00:00:00Z".to_string(),
                corpus_path: "eval/corpora/example.json".to_string(),
                corpus_item_count: 9,
                scored_item_count: 9,
                unreviewed_items_included: false,
                repeats: 1,
                vocab_mode: "on".to_string(),
                kengram_version: "test".to_string(),
                complete: true,
            },
            arms: vec![ArmReport {
                name: "fake-arm".to_string(),
                provider: "openai-compatible".to_string(),
                endpoint: "http://localhost:11434/v1".to_string(),
                model_name: "fake:1b".to_string(),
                model_version: 16,
                prompt: "bundled".to_string(),
                vocab_used: true,
                aborted: None,
                calls: CallStats {
                    attempted: 9,
                    succeeded: 9,
                    failed: 0,
                    skipped: 0,
                    failure_rate: 0.0,
                },
                latency_ms: LatencyStats {
                    p50: 100.0,
                    p95: 200.0,
                    mean: 120.0,
                },
                kind: KindReport {
                    accuracy: 0.889,
                    labels: crate::eval::score::KIND_LABELS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    matrix: vec![vec![0; 8]; 8],
                },
                fields,
                stability: None,
                worst_items: vec![WorstItem {
                    id: "item-1".to_string(),
                    mean_f1: 0.4,
                    golden: Tags {
                        people: vec!["Sarah".to_string()],
                        kind: Some(TagKind::Task),
                        ..Default::default()
                    },
                    predicted: Tags::default(),
                }],
            }],
            pairwise: vec![],
            calls: vec![CallRecord {
                arm: "fake-arm".to_string(),
                item_id: "item-1".to_string(),
                repeat: 0,
                vocab_used: true,
                latency_ms: 100.0,
                raw_tags: Some(Tags::default()),
                finalized_tags: Some(Tags::default()),
                error: None,
            }],
        }
    }

    #[test]
    fn report_serialization_is_byte_stable() {
        // Pins serde field order and the fn-rename. If this test changes
        // shape unexpectedly, the report format_version needs a bump.
        let a = serde_json::to_string_pretty(&fixed_report()).unwrap();
        let b = serde_json::to_string_pretty(&fixed_report()).unwrap();
        assert_eq!(a, b);
        assert!(a.contains("\"format_version\": 1"));
        assert!(a.contains("\"fn\": 2"), "fn_count must serialize as \"fn\"");
        assert!(a.contains("\"complete\": true"));
        assert!(a.contains("\"skipped\": 0"));
        assert!(a.contains("\"aborted\": null"));
        // Stable key order in fields map (BTreeMap): action_items first.
        let fields_pos = a.find("\"fields\"").unwrap();
        let action_pos = a[fields_pos..].find("\"action_items\"").unwrap();
        let people_pos = a[fields_pos..].find("\"people\"").unwrap();
        assert!(action_pos < people_pos);
    }

    #[test]
    fn write_report_creates_directories() {
        let dir = std::env::temp_dir().join("kengram-eval-report-tests/nested");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("report.json");
        write_report(&fixed_report(), &path).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"fake-arm\""));
    }

    #[test]
    fn render_table_smoke() {
        let table = render_table(&fixed_report());
        assert!(table.contains("fake-arm"));
        assert!(table.contains("kind"));
        assert!(table.contains("worst items — fake-arm"));
        assert!(table.contains("item-1"));
        // No unreviewed warning for a golden run.
        assert!(!table.contains("UNREVIEWED"));
    }
}

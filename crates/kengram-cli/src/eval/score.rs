//! Pure scoring math for the tagger eval harness. No I/O, no tokio, no sqlx.
//!
//! Everything here is deterministic given its inputs; ties in fuzzy matching
//! and modal-kind selection break by fixed, documented orderings so a report
//! is re-derivable from its recorded calls alone.

use std::collections::BTreeSet;

use kengram_core::{TagKind, Tags};

/// Token-F1 threshold above which a predicted/golden action-item pair is
/// eligible for matching. Substring containment floors a pair to exactly
/// this score.
pub const ACTION_ITEM_MATCH_THRESHOLD: f64 = 0.6;

/// Fixed label order for the kind confusion matrix: the seven `TagKind`
/// variants in enum order, then `null` (no classification) last. The order
/// is also the tie-break order for modal-kind selection.
pub const KIND_LABELS: [&str; 8] = [
    "observation",
    "task",
    "idea",
    "reference",
    "person_note",
    "session",
    "decision_record",
    "null",
];

/// Index of a kind value in [`KIND_LABELS`].
pub fn kind_index(kind: Option<TagKind>) -> usize {
    match kind {
        Some(TagKind::Observation) => 0,
        Some(TagKind::Task) => 1,
        Some(TagKind::Idea) => 2,
        Some(TagKind::Reference) => 3,
        Some(TagKind::PersonNote) => 4,
        Some(TagKind::Session) => 5,
        Some(TagKind::DecisionRecord) => 6,
        None => 7,
    }
}

/// Normalize a tag string for comparison: Unicode lowercase, trim, collapse
/// internal whitespace runs to a single space.
pub fn norm(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Normalize a list into a deduped set. Empty-after-normalization strings
/// are dropped (an empty tag carries no information either way).
pub fn norm_set(items: &[String]) -> BTreeSet<String> {
    items
        .iter()
        .map(|s| norm(s))
        .filter(|s| !s.is_empty())
        .collect()
}

/// TP/FP/FN counts for one field on one call. `fn` is a keyword, hence
/// `fn_count`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub fn_count: usize,
}

impl Counts {
    /// Precision; defined as 1.0 when nothing was predicted (TP+FP == 0).
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            1.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        }
    }

    /// Recall; defined as 1.0 when nothing was expected (TP+FN == 0).
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_count == 0 {
            1.0
        } else {
            self.tp as f64 / (self.tp + self.fn_count) as f64
        }
    }

    /// F1; defined as 0.0 when P+R == 0. Both-empty prediction and golden
    /// therefore yields P = R = F1 = 1.0 — correctly predicting "nothing"
    /// is correct.
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    pub fn add(&mut self, other: Counts) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.fn_count += other.fn_count;
    }
}

/// Exact set-based scoring over normalized, deduped values. Used for
/// `people`, `entities`, `topics`, and `dates_mentioned`.
pub fn score_set_field(predicted: &[String], golden: &[String]) -> Counts {
    let pred = norm_set(predicted);
    let gold = norm_set(golden);
    let tp = pred.intersection(&gold).count();
    Counts {
        tp,
        fp: pred.len() - tp,
        fn_count: gold.len() - tp,
    }
}

/// Tokenize for fuzzy matching: normalize, replace non-alphanumeric chars
/// with spaces, split, dedupe into a token set.
fn token_set(s: &str) -> BTreeSet<String> {
    norm(s)
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// Set-based token F1 between two strings. Both-empty token sets score 1.0.
pub fn token_f1(a: &str, b: &str) -> f64 {
    let (ta, tb) = (token_set(a), token_set(b));
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    let inter = ta.intersection(&tb).count();
    2.0 * inter as f64 / (ta.len() + tb.len()) as f64
}

/// Fuzzy scoring for `action_items` via greedy bipartite matching.
///
/// A predicted/golden pair is eligible when its token-F1 is >=
/// [`ACTION_ITEM_MATCH_THRESHOLD`], or when one normalized string contains
/// the other as a substring (which floors the pair score to the threshold so
/// it sorts deterministically). Eligible pairs are sorted by
/// `(score desc, pred index asc, gold index asc)` and accepted greedily when
/// both sides are still unmatched. Matched pairs are TP; leftover
/// predictions FP; leftover goldens FN.
pub fn score_action_items(predicted: &[String], golden: &[String]) -> Counts {
    let pred: Vec<String> = predicted
        .iter()
        .map(|s| norm(s))
        .filter(|s| !s.is_empty())
        .collect();
    let gold: Vec<String> = golden
        .iter()
        .map(|s| norm(s))
        .filter(|s| !s.is_empty())
        .collect();

    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, p) in pred.iter().enumerate() {
        for (j, g) in gold.iter().enumerate() {
            let mut score = token_f1(p, g);
            if score < ACTION_ITEM_MATCH_THRESHOLD
                && (p.contains(g.as_str()) || g.contains(p.as_str()))
            {
                score = ACTION_ITEM_MATCH_THRESHOLD;
            }
            if score >= ACTION_ITEM_MATCH_THRESHOLD {
                pairs.push((score, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut pred_used = vec![false; pred.len()];
    let mut gold_used = vec![false; gold.len()];
    let mut tp = 0usize;
    for (_, i, j) in pairs {
        if !pred_used[i] && !gold_used[j] {
            pred_used[i] = true;
            gold_used[j] = true;
            tp += 1;
        }
    }
    Counts {
        tp,
        fp: pred.len() - tp,
        fn_count: gold.len() - tp,
    }
}

/// Per-call scores for a full `Tags` prediction against a golden `Tags`.
/// Kind correctness is tracked separately via [`ConfusionMatrix`].
#[derive(Debug, Clone, Copy)]
pub struct TagScores {
    pub people: Counts,
    pub entities: Counts,
    pub action_items: Counts,
    pub topics: Counts,
    pub dates_mentioned: Counts,
}

impl TagScores {
    /// Unweighted mean F1 across the five fields — used to rank worst items.
    pub fn mean_f1(&self) -> f64 {
        (self.people.f1()
            + self.entities.f1()
            + self.action_items.f1()
            + self.topics.f1()
            + self.dates_mentioned.f1())
            / 5.0
    }
}

/// Score one predicted `Tags` against the golden labels.
pub fn score_tags(predicted: &Tags, golden: &Tags) -> TagScores {
    TagScores {
        people: score_set_field(&predicted.people, &golden.people),
        entities: score_set_field(&predicted.entities, &golden.entities),
        action_items: score_action_items(&predicted.action_items, &golden.action_items),
        topics: score_set_field(&predicted.topics, &golden.topics),
        dates_mentioned: score_set_field(&predicted.dates_mentioned, &golden.dates_mentioned),
    }
}

/// 8x8 kind confusion matrix; rows = golden, cols = predicted, label order
/// per [`KIND_LABELS`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfusionMatrix {
    pub matrix: [[u32; 8]; 8],
}

impl ConfusionMatrix {
    pub fn add(&mut self, golden: Option<TagKind>, predicted: Option<TagKind>) {
        self.matrix[kind_index(golden)][kind_index(predicted)] += 1;
    }

    pub fn total(&self) -> u32 {
        self.matrix.iter().flatten().sum()
    }

    /// Exact-match rate over all recorded calls; 0.0 when empty.
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let diag: u32 = (0..8).map(|i| self.matrix[i][i]).sum();
        diag as f64 / total as f64
    }

    pub fn to_rows(&self) -> Vec<Vec<u32>> {
        self.matrix.iter().map(|r| r.to_vec()).collect()
    }
}

/// Modal-kind agreement across N repeated classifications of one item:
/// `count(modal kind) / N`. Ties between kinds with equal counts break by
/// [`KIND_LABELS`] order (the returned modal kind is the earliest label).
/// Returns `(modal kind index, agreement)`; agreement is 0.0 for empty input.
pub fn modal_kind_agreement(kinds: &[Option<TagKind>]) -> (usize, f64) {
    if kinds.is_empty() {
        return (kind_index(None), 0.0);
    }
    let mut counts = [0usize; 8];
    for k in kinds {
        counts[kind_index(*k)] += 1;
    }
    let modal = (0..8)
        .max_by_key(|&i| (counts[i], usize::MAX - i))
        .unwrap_or(7);
    (modal, counts[modal] as f64 / kinds.len() as f64)
}

/// Mean pairwise Jaccard similarity across N normalized sets.
/// `Jaccard(empty, empty) := 1.0`. Returns 1.0 for fewer than two sets
/// (a single observation is trivially stable).
pub fn mean_pairwise_jaccard(sets: &[BTreeSet<String>]) -> f64 {
    if sets.len() < 2 {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut pairs = 0usize;
    for i in 0..sets.len() {
        for j in (i + 1)..sets.len() {
            sum += jaccard(&sets[i], &sets[j]);
            pairs += 1;
        }
    }
    sum / pairs as f64
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    inter as f64 / union as f64
}

/// Nearest-rank percentile over an unsorted sample (sorted internally).
/// `p` in [0, 100]. Returns 0.0 for an empty sample.
pub fn percentile_nearest_rank(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn norm_lowercases_trims_and_collapses_whitespace() {
        assert_eq!(norm("  Foo   BAR\tbaz  "), "foo bar baz");
        assert_eq!(norm("PgVector"), "pgvector");
        assert_eq!(norm(""), "");
    }

    #[test]
    fn norm_set_dedupes_post_normalization_and_drops_empties() {
        let s = norm_set(&v(&["Sarah", "sarah", "  SARAH ", "", "  "]));
        assert_eq!(s.len(), 1);
        assert!(s.contains("sarah"));
    }

    #[test]
    fn set_field_both_empty_is_perfect() {
        let c = score_set_field(&[], &[]);
        assert_eq!((c.tp, c.fp, c.fn_count), (0, 0, 0));
        assert_eq!(c.precision(), 1.0);
        assert_eq!(c.recall(), 1.0);
        assert_eq!(c.f1(), 1.0);
    }

    #[test]
    fn set_field_pred_empty_gold_nonempty() {
        let c = score_set_field(&[], &v(&["sarah"]));
        assert_eq!(c.precision(), 1.0); // predicted nothing wrong
        assert_eq!(c.recall(), 0.0);
        assert_eq!(c.f1(), 0.0);
    }

    #[test]
    fn set_field_gold_empty_pred_nonempty() {
        let c = score_set_field(&v(&["noise"]), &[]);
        assert_eq!(c.precision(), 0.0);
        assert_eq!(c.recall(), 1.0);
        assert_eq!(c.f1(), 0.0);
    }

    #[test]
    fn set_field_duplicates_collapse_before_counting() {
        // Two spellings of the same normalized value count once.
        let c = score_set_field(&v(&["Kengram", "kengram"]), &v(&["kengram"]));
        assert_eq!((c.tp, c.fp, c.fn_count), (1, 0, 0));
    }

    #[test]
    fn set_field_partial_overlap_counts() {
        let c = score_set_field(&v(&["a", "b", "c"]), &v(&["b", "c", "d"]));
        assert_eq!((c.tp, c.fp, c.fn_count), (2, 1, 1));
        assert!((c.f1() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn token_f1_boundary_cases() {
        assert_eq!(token_f1("fix the login bug", "fix the login bug"), 1.0);
        assert_eq!(token_f1("", ""), 1.0);
        assert_eq!(token_f1("alpha", "beta"), 0.0);
        // Punctuation is stripped before tokenizing.
        assert_eq!(token_f1("fix login-bug!", "fix login bug"), 1.0);
    }

    #[test]
    fn action_items_match_at_exactly_threshold_is_inclusive() {
        // tokens: {a, b, c} vs {a, b, d}: inter 2, sizes 3+3 -> F1 = 4/6 = 0.666...
        // tokens: {a, b, c, d, e} vs {a, b, c}: inter 3 -> F1 = 6/8 = 0.75
        // Construct exactly 0.6: inter 3, sizes 4+6 -> 6/10 = 0.6.
        let pred = v(&["a b c d"]);
        let gold = v(&["a b c x y z"]);
        assert!((token_f1(&pred[0], &gold[0]) - 0.6).abs() < 1e-9);
        let c = score_action_items(&pred, &gold);
        assert_eq!((c.tp, c.fp, c.fn_count), (1, 0, 0));
    }

    #[test]
    fn action_items_below_threshold_no_match() {
        // inter 2, sizes 4+6 -> 4/10 = 0.4 < 0.6, and no substring relation.
        let c = score_action_items(&v(&["a b q r"]), &v(&["a b x y z w"]));
        assert_eq!((c.tp, c.fp, c.fn_count), (0, 1, 1));
    }

    #[test]
    fn action_items_substring_fallback_matches() {
        // Token F1 = 2*2/(2+8) = 0.4 < 0.6, but pred is a substring of gold.
        let pred = v(&["renew domain"]);
        let gold = v(&["renew domain before august first deadline passes us"]);
        assert!(token_f1(&pred[0], &gold[0]) < 0.6);
        let c = score_action_items(&pred, &gold);
        assert_eq!((c.tp, c.fp, c.fn_count), (1, 0, 0));
    }

    #[test]
    fn action_items_greedy_matching_is_one_to_one_and_deterministic() {
        // Both predictions are eligible against the single golden item; only
        // one may match (the higher-scoring; tie broken by pred index).
        let pred = v(&["fix the login bug", "fix the login bug now"]);
        let gold = v(&["fix the login bug"]);
        let c = score_action_items(&pred, &gold);
        assert_eq!((c.tp, c.fp, c.fn_count), (1, 1, 0));
    }

    #[test]
    fn action_items_unequal_lengths_account_fp_fn() {
        let c = score_action_items(&v(&["alpha beta gamma"]), &v(&[]));
        assert_eq!((c.tp, c.fp, c.fn_count), (0, 1, 0));
        let c = score_action_items(&[], &v(&["alpha beta gamma"]));
        assert_eq!((c.tp, c.fp, c.fn_count), (0, 0, 1));
    }

    #[test]
    fn confusion_matrix_placement_including_null() {
        let mut m = ConfusionMatrix::default();
        m.add(Some(TagKind::Task), Some(TagKind::Observation));
        m.add(Some(TagKind::Task), Some(TagKind::Task));
        m.add(None, None);
        m.add(None, Some(TagKind::Idea));
        assert_eq!(m.matrix[1][0], 1); // gold task, pred observation
        assert_eq!(m.matrix[1][1], 1); // gold task, pred task
        assert_eq!(m.matrix[7][7], 1); // gold null, pred null
        assert_eq!(m.matrix[7][2], 1); // gold null, pred idea
        assert_eq!(m.total(), 4);
        assert_eq!(m.accuracy(), 0.5);
    }

    #[test]
    fn micro_vs_macro_divergence() {
        // Call A: 10 TP, 0 FP/FN -> F1 1.0. Call B: 0 TP, 1 FP, 1 FN -> F1 0.0.
        // Macro = 0.5. Micro = F1 of (10 TP, 1 FP, 1 FN) = 20/22 ≈ 0.909.
        let a = Counts {
            tp: 10,
            fp: 0,
            fn_count: 0,
        };
        let b = Counts {
            tp: 0,
            fp: 1,
            fn_count: 1,
        };
        let macro_f1 = (a.f1() + b.f1()) / 2.0;
        let mut micro = a;
        micro.add(b);
        assert_eq!(macro_f1, 0.5);
        assert!((micro.f1() - 20.0 / 22.0).abs() < 1e-9);
    }

    #[test]
    fn modal_kind_tie_breaks_by_label_order() {
        // 1x task, 1x idea: tie -> task (earlier in KIND_LABELS).
        let (modal, agreement) = modal_kind_agreement(&[Some(TagKind::Idea), Some(TagKind::Task)]);
        assert_eq!(KIND_LABELS[modal], "task");
        assert_eq!(agreement, 0.5);
    }

    #[test]
    fn modal_kind_unanimous() {
        let (modal, agreement) = modal_kind_agreement(&[
            Some(TagKind::Reference),
            Some(TagKind::Reference),
            Some(TagKind::Reference),
        ]);
        assert_eq!(KIND_LABELS[modal], "reference");
        assert_eq!(agreement, 1.0);
    }

    #[test]
    fn jaccard_of_two_empty_sets_is_one() {
        let sets = vec![BTreeSet::new(), BTreeSet::new()];
        assert_eq!(mean_pairwise_jaccard(&sets), 1.0);
    }

    #[test]
    fn mean_pairwise_jaccard_mixed() {
        let a: BTreeSet<String> = ["x".to_string(), "y".to_string()].into();
        let b: BTreeSet<String> = ["x".to_string()].into();
        let c: BTreeSet<String> = BTreeSet::new();
        // pairs: (a,b)=1/2, (a,c)=0, (b,c)=0 -> mean = 1/6
        let m = mean_pairwise_jaccard(&[a, b, c]);
        assert!((m - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_nearest_rank_basics() {
        let samples = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile_nearest_rank(&samples, 50.0), 20.0);
        assert_eq!(percentile_nearest_rank(&samples, 95.0), 40.0);
        assert_eq!(percentile_nearest_rank(&[], 50.0), 0.0);
        assert_eq!(percentile_nearest_rank(&[7.0], 50.0), 7.0);
    }

    #[test]
    fn score_tags_full_call() {
        let golden = Tags {
            people: v(&["Sarah"]),
            entities: v(&["kengram"]),
            action_items: v(&["fast-track migration #0042"]),
            topics: v(&["release-process"]),
            dates_mentioned: v(&["Thursday"]),
            kind: Some(TagKind::Task),
        };
        let predicted = Tags {
            people: v(&["sarah"]),                           // norm match
            entities: v(&["kengram", "noise"]),              // 1 TP 1 FP
            action_items: v(&["fast track migration 0042"]), // fuzzy match
            topics: v(&[]),                                  // 1 FN
            dates_mentioned: v(&["thursday"]),               // match
            kind: Some(TagKind::Task),
        };
        let s = score_tags(&predicted, &golden);
        assert_eq!((s.people.tp, s.people.fp, s.people.fn_count), (1, 0, 0));
        assert_eq!((s.entities.tp, s.entities.fp), (1, 1));
        assert_eq!(s.action_items.tp, 1);
        assert_eq!(s.topics.fn_count, 1);
        assert!(s.mean_f1() > 0.5 && s.mean_f1() < 1.0);
    }
}

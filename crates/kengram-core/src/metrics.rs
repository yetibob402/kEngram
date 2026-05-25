//! Pure ranking-metric helpers. Used by the M3 Phase B step 3 A/B
//! benchmarking harness (`kengram bench rerank`) to compare RRF-only vs
//! reranked orderings against an operator-curated fixture corpus.
//!
//! No I/O; no kengram-specific types. The functions are keyed on `Uuid` so
//! they apply equally to thought_ids and fact_ids.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Discounted Cumulative Gain at K, normalized by the ideal DCG.
///
/// Graded relevance: each id in `graded` carries a weight in [0, 1].
/// Items not present in the map count as 0. The DCG of a ranking is
/// `Σ rel_i / log2(i + 1)` for positions `i ∈ [1, k]` (1-indexed); the
/// nDCG is `DCG / IDCG`, where IDCG is the DCG of the ideal ranking
/// (the items in `graded` sorted by descending weight, truncated to K).
///
/// Returns 0.0 when the ideal DCG is zero (no relevant items at all)
/// or when the ranking contains no items with positive graded weight.
/// Range is [0, 1]; 1.0 means the top-K of the ranking matches the
/// ideal top-K exactly.
pub fn ndcg_at_k(ranking: &[Uuid], graded: &HashMap<Uuid, f32>, k: usize) -> f32 {
    if k == 0 || ranking.is_empty() || graded.is_empty() {
        return 0.0;
    }

    let dcg = ranking
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = graded.get(id).copied().unwrap_or(0.0);
            // log2(i + 2) because i is 0-indexed but DCG positions are 1-indexed.
            rel / ((i as f32 + 2.0).log2())
        })
        .sum::<f32>();

    // Ideal DCG: same K positions filled with the top-K weights from
    // `graded`, descending.
    let mut weights: Vec<f32> = graded.values().copied().collect();
    weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = weights
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, w)| w / ((i as f32 + 2.0).log2()))
        .sum::<f32>();

    if idcg <= 0.0 { 0.0 } else { dcg / idcg }
}

/// Reciprocal rank of the first relevant item in `ranking`. Returns
/// `1 / (rank of first relevant)`, where rank is 1-indexed, or 0.0 if
/// no relevant item appears in the ranking. The bench harness averages
/// this across queries to produce MRR (Mean Reciprocal Rank).
pub fn reciprocal_rank(ranking: &[Uuid], relevant: &HashSet<Uuid>) -> f32 {
    if relevant.is_empty() {
        return 0.0;
    }
    for (i, id) in ranking.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(seed: u128) -> Uuid {
        Uuid::from_u128(seed)
    }

    #[test]
    fn ndcg_at_k_perfect_ranking_is_one() {
        let a = id(1);
        let b = id(2);
        let c = id(3);
        let ranking = vec![a, b, c];
        let graded = HashMap::from([(a, 1.0), (b, 0.7), (c, 0.3)]);
        assert!((ndcg_at_k(&ranking, &graded, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ndcg_at_k_zero_when_no_relevant_in_ranking() {
        let a = id(1);
        let b = id(2);
        let c = id(3);
        let ranking = vec![a, b];
        // None of `a` / `b` are in the graded map.
        let graded = HashMap::from([(c, 1.0)]);
        assert_eq!(ndcg_at_k(&ranking, &graded, 10), 0.0);
    }

    #[test]
    fn ndcg_at_k_binary_case_matches_graded_with_unit_weights() {
        // Same ranking + same relevant set; one expressed as graded with
        // weight 1.0, the other as "graded" with weight 1.0 — should be
        // identical scores. (Confirms binary→unit-weight promotion in the
        // bench harness is metric-faithful.)
        let a = id(1);
        let b = id(2);
        let c = id(3);
        let ranking = vec![c, a, b];
        let graded_unit = HashMap::from([(a, 1.0), (b, 1.0)]);
        let n_unit = ndcg_at_k(&ranking, &graded_unit, 10);
        // Compare with an explicit graded form: relevant set { a, b } with
        // arbitrary equal weights 0.5 — different weights but same
        // ideal-vs-actual ordering should yield the same nDCG.
        let graded_half = HashMap::from([(a, 0.5), (b, 0.5)]);
        let n_half = ndcg_at_k(&ranking, &graded_half, 10);
        assert!((n_unit - n_half).abs() < 1e-6);
    }

    #[test]
    fn ndcg_at_k_truncates_to_k() {
        // The relevant item is at position 11 (1-indexed). nDCG@10 should
        // be 0 (it's outside the top-K window).
        let mut ranking = Vec::new();
        for i in 1..=10 {
            ranking.push(id(i));
        }
        let target = id(100);
        ranking.push(target);
        let graded = HashMap::from([(target, 1.0)]);
        assert_eq!(ndcg_at_k(&ranking, &graded, 10), 0.0);
        // But at K=11 it's the only relevant item — DCG = 1 / log2(12) ≈ 0.279;
        // IDCG = 1 / log2(2) = 1.0; nDCG ≈ 0.279.
        let n11 = ndcg_at_k(&ranking, &graded, 11);
        assert!((n11 - 1.0 / 12.0_f32.log2()).abs() < 1e-5);
    }

    #[test]
    fn ndcg_at_k_reversed_ranking_is_known_sub_one() {
        // Ranking puts the highest-weight item last → nDCG < 1.0.
        let a = id(1); // weight 1.0
        let b = id(2); // weight 0.5
        let ranking = vec![b, a];
        let graded = HashMap::from([(a, 1.0), (b, 0.5)]);
        let n = ndcg_at_k(&ranking, &graded, 10);
        // DCG  = 0.5/log2(2) + 1.0/log2(3) = 0.5 + 0.6309
        // IDCG = 1.0/log2(2) + 0.5/log2(3) = 1.0 + 0.3155
        // nDCG ≈ 1.1309 / 1.3155 ≈ 0.8597
        assert!(n > 0.0 && n < 1.0);
        assert!((n - 0.8597).abs() < 1e-3);
    }

    #[test]
    fn reciprocal_rank_returns_inverse_of_first_relevant_position() {
        let a = id(1);
        let b = id(2);
        let c = id(3);
        let ranking = vec![a, b, c];
        let relevant = HashSet::from([c]);
        // c is at position 3 (1-indexed) → 1/3.
        let rr = reciprocal_rank(&ranking, &relevant);
        assert!((rr - 1.0 / 3.0).abs() < 1e-6);

        let relevant_first = HashSet::from([a]);
        assert!((reciprocal_rank(&ranking, &relevant_first) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reciprocal_rank_zero_when_no_relevant_found() {
        let a = id(1);
        let b = id(2);
        let ranking = vec![a, b];
        let relevant: HashSet<Uuid> = HashSet::from([id(99)]);
        assert_eq!(reciprocal_rank(&ranking, &relevant), 0.0);

        // Empty relevant set is also 0.
        let empty: HashSet<Uuid> = HashSet::new();
        assert_eq!(reciprocal_rank(&ranking, &empty), 0.0);
    }
}

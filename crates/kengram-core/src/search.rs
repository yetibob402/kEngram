//! Search composition primitives. Pure logic; storage and embedder I/O
//! live elsewhere.
//!
//! The hybrid retrieval pipeline is:
//!   1. Each retrieval leg (vector kNN, lexical FTS) returns a
//!      ranked `Vec<Hit>` of length ≤ top_k_per_leg.
//!   2. [`rrf_fuse`] combines the rankings into a single ordering by
//!      reciprocal rank fusion: `rrf_score(d) = Σ 1 / (k + rank_i(d))`.
//!   3. [`recency_boost`] multiplies each fused score by a half-life
//!      decay factor (`0.5^(age_days / half_life_days)`) and re-sorts.
//!
//! Default RRF `k = 60` matches the IR-literature standard. Default
//! recency half-life is 30 days.
//!
//! As of M3 Phase C, `Hit` exposes only per-leg scores (`vector_score` /
//! `lexical_score` / `trigram_score` / `rrf_score` / `rerank_score`) rather than a unified
//! `score`. Result ordering reflects the post-pipeline stage; consumers
//! who want a threshold-able scalar pick the load-bearing per-leg field
//! (typically `rerank_score ?? rrf_score`).

use std::collections::HashMap;
use time::OffsetDateTime;

use crate::Thought;

pub const DEFAULT_RRF_K: f32 = 60.0;
pub const DEFAULT_RECENCY_HALF_LIFE_DAYS: f32 = 30.0;

/// A single retrieval hit. Storage layers return these from each leg; the
/// fusion layer accumulates and re-orders them.
///
/// Per-leg fields capture each pipeline stage's signal:
///   - `vector_score`: raw cosine similarity from the vector kNN leg.
///   - `lexical_score`: raw rank from the current lexical leg.
///   - `trigram_score`: legacy raw `similarity` from the old trigram leg.
///   - `rrf_score`: RRF aggregate `Σ 1/(k + rank_i)` after fusion; updated
///     in-place by [`recency_boost`] (multiplied by the decay factor).
///   - `rerank_score`: calibrated absolute relevance from the cross-encoder,
///     set by the search orchestrator when a reranker is configured.
///
/// Returned hits are sorted by the most-load-bearing signal: `rerank_score`
/// when reranked, otherwise `rrf_score`. There's no unified `score` field —
/// consumers reach the raw per-leg signals directly when they need to
/// threshold or compare across pipeline configurations.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub thought: Thought,
    /// Raw cosine similarity from the vector kNN leg. `None` when the hit
    /// did not appear in the vector leg (lexical-only match or vector leg
    /// unavailable).
    pub vector_score: Option<f32>,
    /// Raw rank from the current lexical leg. Today this is Postgres FTS
    /// `ts_rank_cd`; older trigram-only callers leave it unset.
    pub lexical_score: Option<f32>,
    /// Raw `similarity` (pg_trgm symmetric n-gram Jaccard) from the trigram
    /// leg. Kept for wire compatibility with older clients; the production
    /// hybrid lexical leg now uses `lexical_score`.
    pub trigram_score: Option<f32>,
    /// Reciprocal Rank Fusion aggregate, optionally adjusted by
    /// [`recency_boost`]. Set by [`rrf_fuse`]; `None` on raw leg hits
    /// before fusion runs. After fusion this is the pre-rerank signal —
    /// preserved across the rerank stage so consumers can compare RRF-only
    /// ordering against rerank ordering without re-running search.
    pub rrf_score: Option<f32>,
    /// Calibrated absolute relevance score from the cross-encoder
    /// reranker. `None` when rerank was off, no reranker was configured,
    /// or the hit fell outside the reranked candidate pool.
    pub rerank_score: Option<f32>,
}

impl Hit {
    /// Construct a hit produced by the vector kNN leg. The raw cosine
    /// similarity lands in `vector_score`; lexical + rerank + RRF fields
    /// default to `None` (fusion fills `rrf_score` later).
    pub fn from_vector_leg(thought: Thought, cosine_similarity: f32) -> Self {
        Self {
            thought,
            vector_score: Some(cosine_similarity),
            lexical_score: None,
            trigram_score: None,
            rrf_score: None,
            rerank_score: None,
        }
    }

    /// Construct a hit produced by the lexical FTS leg. The raw Postgres FTS
    /// rank lands in `lexical_score`; vector + trigram + rerank + RRF fields
    /// default to `None`.
    pub fn from_lexical_leg(thought: Thought, rank: f32) -> Self {
        Self {
            thought,
            vector_score: None,
            lexical_score: Some(rank),
            trigram_score: None,
            rrf_score: None,
            rerank_score: None,
        }
    }

    /// Construct a hit produced by the trigram leg. The raw `similarity`
    /// (pg_trgm symmetric n-gram Jaccard) lands in `trigram_score`; vector +
    /// rerank + RRF fields default to `None`.
    pub fn from_trigram_leg(thought: Thought, similarity: f32) -> Self {
        Self {
            thought,
            vector_score: None,
            lexical_score: None,
            trigram_score: Some(similarity),
            rrf_score: None,
            rerank_score: None,
        }
    }
}

/// Reciprocal Rank Fusion. Each ranking is taken in the order given. The
/// aggregate `Σ 1/(k + rank_i)` lands in each hit's `rrf_score`. Per-leg
/// score fields (`vector_score`, `lexical_score`, `trigram_score`) are preserved across
/// the fusion: when both rankings carry the same thought, the leg-specific
/// scores from each input merge into the accumulator (an input's `Some(_)`
/// always wins over an existing `None`). Output is sorted by descending
/// `rrf_score`.
pub fn rrf_fuse(rankings: Vec<Vec<Hit>>, k: f32) -> Vec<Hit> {
    let mut acc: HashMap<crate::ThoughtId, Hit> = HashMap::new();

    for ranking in rankings {
        for (i, hit) in ranking.into_iter().enumerate() {
            let rank = (i + 1) as f32;
            let contribution = 1.0 / (k + rank);
            match acc.get_mut(&hit.thought.id) {
                Some(existing) => {
                    let current = existing.rrf_score.unwrap_or(0.0);
                    existing.rrf_score = Some(current + contribution);
                    if existing.vector_score.is_none() {
                        existing.vector_score = hit.vector_score;
                    }
                    if existing.trigram_score.is_none() {
                        existing.trigram_score = hit.trigram_score;
                    }
                    if existing.lexical_score.is_none() {
                        existing.lexical_score = hit.lexical_score;
                    }
                }
                None => {
                    let id = hit.thought.id;
                    let merged = Hit {
                        thought: hit.thought,
                        vector_score: hit.vector_score,
                        lexical_score: hit.lexical_score,
                        trigram_score: hit.trigram_score,
                        rrf_score: Some(contribution),
                        rerank_score: None,
                    };
                    acc.insert(id, merged);
                }
            }
        }
    }

    let mut fused: Vec<Hit> = acc.into_values().collect();
    fused.sort_by(|a, b| {
        let av = a.rrf_score.unwrap_or(0.0);
        let bv = b.rrf_score.unwrap_or(0.0);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

/// Post-fusion recency boost. Multiplies each hit's `rrf_score` by
/// `0.5^(age_days / half_life_days)` and re-sorts by the boosted RRF score.
/// A hit captured exactly `half_life_days` ago has its score halved; one
/// captured now is unchanged. Hits without an `rrf_score` (raw leg hits
/// fed in pre-fusion) are left untouched.
pub fn recency_boost(hits: &mut [Hit], half_life_days: f32, now: OffsetDateTime) {
    if half_life_days <= 0.0 {
        return; // disabled
    }
    for h in hits.iter_mut() {
        let age_secs = (now - h.thought.created_at).whole_seconds() as f32;
        let age_days = age_secs / 86_400.0;
        let factor = 0.5_f32.powf(age_days / half_life_days);
        if let Some(s) = h.rrf_score {
            h.rrf_score = Some(s * factor);
        }
    }
    hits.sort_by(|a, b| {
        let av = a.rrf_score.unwrap_or(0.0);
        let bv = b.rrf_score.unwrap_or(0.0);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Metadata, Scope, Source, Tags, ThoughtId};

    fn thought(id_seed: u128, content: &str, age_seconds: i64) -> Thought {
        let id = ThoughtId::from(uuid::Uuid::from_u128(id_seed));
        Thought {
            id,
            scope: Scope::default(),
            content: content.to_string(),
            source: Source::new("test").unwrap(),
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000 - age_seconds).unwrap(),
            metadata: Metadata::empty(),
            content_fingerprint: [0u8; 32],
            tags: Tags::default(),
            tags_extractor_model: None,
            tags_extractor_version: None,
            tags_extracted_at: None,
        }
    }

    /// Build a pre-fusion vector-leg hit at a given raw similarity. Mirrors
    /// what the storage layer hands to `rrf_fuse`.
    fn vec_hit(t: Thought, vector_score: f32) -> Hit {
        Hit::from_vector_leg(t, vector_score)
    }

    #[test]
    fn rrf_empty_rankings_returns_empty() {
        let out = rrf_fuse(vec![], DEFAULT_RRF_K);
        assert!(out.is_empty());
    }

    #[test]
    fn rrf_single_ranking_preserves_order_with_decreasing_scores() {
        let r = vec![
            vec_hit(thought(1, "a", 0), 0.9),
            vec_hit(thought(2, "b", 0), 0.5),
            vec_hit(thought(3, "c", 0), 0.1),
        ];
        let out = rrf_fuse(vec![r], 60.0);
        assert_eq!(out.len(), 3);
        // First item gets 1/(60+1) = 0.0164; second 1/62 = 0.0161; third 1/63 = 0.0159.
        assert!(out[0].rrf_score > out[1].rrf_score);
        assert!(out[1].rrf_score > out[2].rrf_score);
        assert_eq!(out[0].thought.content, "a");
        assert_eq!(out[1].thought.content, "b");
        assert_eq!(out[2].thought.content, "c");
    }

    #[test]
    fn rrf_overlapping_hit_accumulates_score() {
        // 'a' appears at rank 1 in both rankings; 'b' only in the first.
        // rrf_score(a) = 2 * 1/61 ≈ 0.0328; rrf_score(b) = 1/62 ≈ 0.0161.
        let r1 = vec![
            vec_hit(thought(1, "a", 0), 0.9),
            vec_hit(thought(2, "b", 0), 0.5),
        ];
        let r2 = vec![vec_hit(thought(1, "a", 0), 0.8)];
        let out = rrf_fuse(vec![r1, r2], 60.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].thought.content, "a");
        assert!((out[0].rrf_score.unwrap() - 2.0 / 61.0).abs() < 1e-6);
        assert_eq!(out[1].thought.content, "b");
        assert!((out[1].rrf_score.unwrap() - 1.0 / 62.0).abs() < 1e-6);
    }

    #[test]
    fn rrf_preserves_per_leg_scores_when_present() {
        // Build a vector-leg hit and a lexical-leg hit for two different
        // thoughts. After fusion, each hit should carry the per-leg score
        // from its origin (and None for the other leg).
        let v_hit = Hit::from_vector_leg(thought(1, "vec only", 0), 0.85);
        let l_hit = Hit::from_lexical_leg(thought(2, "lex only", 0), 0.42);
        let out = rrf_fuse(vec![vec![v_hit], vec![l_hit]], 60.0);
        let by_content: std::collections::HashMap<String, Hit> = out
            .into_iter()
            .map(|h| (h.thought.content.clone(), h))
            .collect();
        let v = &by_content["vec only"];
        assert_eq!(v.vector_score, Some(0.85));
        assert_eq!(v.lexical_score, None);
        assert_eq!(v.trigram_score, None);
        let l = &by_content["lex only"];
        assert_eq!(l.vector_score, None);
        assert_eq!(l.lexical_score, Some(0.42));
        assert_eq!(l.trigram_score, None);
    }

    #[test]
    fn rrf_merges_per_leg_scores_when_both_legs_match() {
        // Same thought appears in both legs — vector_score AND lexical_score
        // should survive the fusion.
        let v_hit = Hit::from_vector_leg(thought(1, "both", 0), 0.91);
        let l_hit = Hit::from_lexical_leg(thought(1, "both", 0), 0.33);
        let out = rrf_fuse(vec![vec![v_hit], vec![l_hit]], 60.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vector_score, Some(0.91));
        assert_eq!(out[0].lexical_score, Some(0.33));
        assert_eq!(out[0].trigram_score, None);
    }

    #[test]
    fn rrf_disjoint_rankings_union_both_with_correct_scores() {
        let r1 = vec![vec_hit(thought(1, "a", 0), 0.9)];
        let r2 = vec![vec_hit(thought(2, "b", 0), 0.5)];
        let out = rrf_fuse(vec![r1, r2], 60.0);
        assert_eq!(out.len(), 2);
        // Each should have rrf_score 1/61.
        assert!((out[0].rrf_score.unwrap() - 1.0 / 61.0).abs() < 1e-6);
        assert!((out[1].rrf_score.unwrap() - 1.0 / 61.0).abs() < 1e-6);
    }

    /// Regression: M3 Phase C removed the unified `score` field; the RRF
    /// aggregate now lives ONLY in `rrf_score`. This test pins the
    /// `Hit` shape so a future refactor can't reintroduce a redundant
    /// post-pipeline scalar without explicit operator review.
    #[test]
    fn rrf_fuse_updates_rrf_score_only_after_score_drop() {
        let r = vec![vec_hit(thought(1, "a", 0), 0.9)];
        let out = rrf_fuse(vec![r], 60.0);
        assert_eq!(out.len(), 1);
        assert!(out[0].rrf_score.is_some());
        // Vector signal preserved at construction time.
        assert_eq!(out[0].vector_score, Some(0.9));
        // No rerank stage was run.
        assert!(out[0].rerank_score.is_none());
    }

    #[test]
    fn recency_boost_halves_score_at_half_life() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        // Created exactly 30 days ago; pre-set rrf_score = 1.0.
        let mut hits = vec![Hit {
            thought: thought(1, "old", 30 * 86_400),
            vector_score: None,
            lexical_score: None,
            trigram_score: None,
            rrf_score: Some(1.0),
            rerank_score: None,
        }];
        recency_boost(&mut hits, 30.0, now);
        assert!((hits[0].rrf_score.unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn recency_boost_leaves_fresh_hits_unchanged() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let mut hits = vec![Hit {
            thought: thought(1, "fresh", 0),
            vector_score: None,
            lexical_score: None,
            trigram_score: None,
            rrf_score: Some(1.0),
            rerank_score: None,
        }];
        recency_boost(&mut hits, 30.0, now);
        assert!((hits[0].rrf_score.unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn recency_boost_resorts_when_older_hit_had_higher_raw_score() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        // Old hit with high score, fresh hit with lower score.
        // After boost the old one halves and the fresh one wins.
        let mut hits = vec![
            Hit {
                thought: thought(1, "old", 30 * 86_400),
                vector_score: None,
                lexical_score: None,
                trigram_score: None,
                rrf_score: Some(0.8),
                rerank_score: None,
            },
            Hit {
                thought: thought(2, "fresh", 0),
                vector_score: None,
                lexical_score: None,
                trigram_score: None,
                rrf_score: Some(0.5),
                rerank_score: None,
            },
        ];
        recency_boost(&mut hits, 30.0, now);
        assert_eq!(hits[0].thought.content, "fresh");
        assert_eq!(hits[1].thought.content, "old");
    }

    #[test]
    fn recency_boost_disabled_when_half_life_zero() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let mut hits = vec![Hit {
            thought: thought(1, "old", 30 * 86_400),
            vector_score: None,
            lexical_score: None,
            trigram_score: None,
            rrf_score: Some(1.0),
            rerank_score: None,
        }];
        recency_boost(&mut hits, 0.0, now);
        assert_eq!(hits[0].rrf_score.unwrap(), 1.0);
    }
}

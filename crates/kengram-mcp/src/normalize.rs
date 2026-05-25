//! Post-process topic normalization for the tag drainer (v11 onward).
//!
//! Replaces the in-prompt topic vocabulary feed that was driving the
//! `databases`/`rust` overreach (see `docs/tagger-improvements.md`). The
//! LLM emits topics fresh from prose; this module maps each emitted topic
//! to the scope's canonical form when the emission is similar enough to an
//! established term. Novel emissions pass through unchanged.
//!
//! Similarity is computed with three lightweight signals (no embedder
//! dependency at this layer — we operate on the bare topic strings):
//!
//! - **Exact match** (case-insensitive) — definitionally canonical.
//! - **Levenshtein ratio** — catches morphological variants
//!   (`memory-system` ↔ `memory-systems`, `database` ↔ `databases`).
//! - **Token-subset** — catches prefix/suffix expansions
//!   (`rust` ↔ `rust-programming`, `search` ↔ `search-engines`).
//!
//! Threshold is conservative: only normalize when one of the signals is
//! strong. False normalizations are worse than missed ones because they
//! erase real distinctions; false misses are recoverable by future
//! retag passes once the vocab grows the missing canonical form.

use std::collections::HashSet;

/// Threshold above which two terms are considered close enough to
/// normalize. Two signals contribute (whichever scores higher wins):
///
/// - Levenshtein ratio — `database`/`databases` is 0.89, `memory-system`
///   /`memory-systems` is 0.93. Captures morphological pairs (singular/
///   plural, hyphenation differences) cleanly above 0.8.
/// - Token-subset score — when one term's `-`-tokens are a strict subset
///   of the other's (`rust` ⊂ `rust-programming`), assign a flat 0.8.
///
/// The unified threshold means both pathways trigger normalization;
/// semantically-distant pairs (`branding` vs `databases`, `agent-memory`
/// vs `databases`) score well under 0.8 and pass through.
const NORMALIZATION_THRESHOLD: f64 = 0.8;

/// Score assigned when one term's `-`-tokens are a strict subset of the
/// other's. Sits exactly at the threshold so subset matches normalize but
/// don't dominate over a stronger Levenshtein signal.
const TOKEN_SUBSET_SIMILARITY: f64 = 0.8;

/// Normalize each emitted topic against the scope's canonical topic vocab.
/// Returns a fresh vector; preserves the order of `emitted` but de-duplicates
/// in case two emissions normalize to the same canonical form.
pub fn normalize_topics(emitted: &[String], vocab: &[String]) -> Vec<String> {
    if vocab.is_empty() {
        return emitted.to_vec();
    }
    let mut out = Vec::with_capacity(emitted.len());
    let mut seen: HashSet<String> = HashSet::new();
    for topic in emitted {
        let normalized = normalize_one(topic, vocab);
        if seen.insert(normalized.clone()) {
            out.push(normalized);
        }
    }
    out
}

fn normalize_one(emitted: &str, vocab: &[String]) -> String {
    let emitted_lower = emitted.to_lowercase();
    // Find the vocab term with the highest similarity; if it clears the
    // threshold and isn't the emitted term itself, normalize.
    let best = vocab
        .iter()
        .map(|v| (v, similarity(&emitted_lower, &v.to_lowercase())))
        .filter(|(_, score)| *score >= NORMALIZATION_THRESHOLD)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match best {
        Some((canonical, _)) => canonical.clone(),
        None => emitted.to_string(),
    }
}

/// Combined similarity score in `[0.0, 1.0]`. Takes the max of two signals:
/// Levenshtein ratio (morphological), and a `TOKEN_SUBSET_SIMILARITY`
/// constant when one term's tokens (split on `-`) are a strict subset of
/// the other's.
fn similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let lev = levenshtein_ratio(a, b);
    let subset = if token_subset(a, b) {
        TOKEN_SUBSET_SIMILARITY
    } else {
        0.0
    };
    lev.max(subset)
}

fn levenshtein_ratio(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Standard Wagner-Fischer Levenshtein. Operates on `char`s, so multi-byte
/// codepoints count as a single unit.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// True when one term's `-`-separated tokens are a strict subset of the
/// other's. Catches `rust` ↔ `rust-programming`, `search` ↔ `search-engines`.
/// Same-size token sets fall through to Levenshtein.
fn token_subset(a: &str, b: &str) -> bool {
    let a_tokens: HashSet<&str> = a.split('-').collect();
    let b_tokens: HashSet<&str> = b.split('-').collect();
    if a_tokens.len() == b_tokens.len() {
        return false;
    }
    a_tokens.is_subset(&b_tokens) || b_tokens.is_subset(&a_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vocab_passes_through() {
        let emitted = vec!["rust".to_string(), "agile".to_string()];
        assert_eq!(normalize_topics(&emitted, &[]), emitted);
    }

    #[test]
    fn exact_match_preserved() {
        let vocab = vec!["rust".to_string()];
        assert_eq!(
            normalize_topics(&["rust".to_string()], &vocab),
            vec!["rust".to_string()]
        );
    }

    #[test]
    fn morphological_variants_normalize() {
        let vocab = vec!["databases".to_string(), "memory-systems".to_string()];
        // singular → plural
        assert_eq!(
            normalize_topics(&["database".to_string()], &vocab),
            vec!["databases".to_string()]
        );
        // hyphenated singular → hyphenated plural
        assert_eq!(
            normalize_topics(&["memory-system".to_string()], &vocab),
            vec!["memory-systems".to_string()]
        );
    }

    #[test]
    fn token_subset_normalizes() {
        let vocab = vec!["rust".to_string(), "search".to_string()];
        assert_eq!(
            normalize_topics(&["rust-programming".to_string()], &vocab),
            vec!["rust".to_string()]
        );
        assert_eq!(
            normalize_topics(&["search-engines".to_string()], &vocab),
            vec!["search".to_string()]
        );
    }

    #[test]
    fn semantically_distant_pairs_do_not_normalize() {
        let vocab = vec!["databases".to_string(), "rust".to_string()];
        assert_eq!(
            normalize_topics(&["branding".to_string()], &vocab),
            vec!["branding".to_string()]
        );
        // "agent-memory" vs "databases" or "rust" — Levenshtein < 0.85 and
        // no token subset. Should pass through unchanged.
        assert_eq!(
            normalize_topics(&["agent-memory".to_string()], &vocab),
            vec!["agent-memory".to_string()]
        );
    }

    #[test]
    fn dedup_after_normalization() {
        // Two distinct emissions that both normalize to the same canonical
        // form should collapse to a single output entry.
        let vocab = vec!["rust".to_string()];
        let emitted = vec!["rust-programming".to_string(), "rust-lang".to_string()];
        let out = normalize_topics(&emitted, &vocab);
        assert_eq!(out, vec!["rust".to_string()]);
    }

    #[test]
    fn preserves_emission_order_for_novel_topics() {
        let vocab = vec!["databases".to_string()];
        let emitted = vec![
            "agile-development".to_string(),
            "leadership".to_string(),
            "databases".to_string(),
        ];
        let out = normalize_topics(&emitted, &vocab);
        assert_eq!(out, emitted);
    }

    #[test]
    fn case_insensitive_match() {
        let vocab = vec!["Rust".to_string()];
        // Emitted lowercase should still match vocab regardless of case;
        // the canonical form (from vocab) is what wins.
        assert_eq!(
            normalize_topics(&["rust".to_string()], &vocab),
            vec!["Rust".to_string()]
        );
    }

    #[test]
    fn case_insensitive_token_subset() {
        let vocab = vec!["search".to_string()];
        assert_eq!(
            normalize_topics(&["Search-Engines".to_string()], &vocab),
            vec!["search".to_string()]
        );
    }

    #[test]
    fn empty_emissions_passes_through() {
        assert_eq!(
            normalize_topics(&[], &["rust".to_string()]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn token_subset_requires_distinct_sizes() {
        // Two single-token forms shouldn't subset-match (would over-normalize).
        let vocab = vec!["rust".to_string()];
        assert_eq!(
            normalize_topics(&["go".to_string()], &vocab),
            vec!["go".to_string()]
        );
    }

    #[test]
    fn levenshtein_basic_distances() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("databases", "database"), 1);
    }

    #[test]
    fn token_subset_helper() {
        assert!(token_subset("rust", "rust-programming"));
        assert!(token_subset("rust-programming", "rust"));
        assert!(!token_subset("rust", "go"));
        assert!(!token_subset("rust", "rust")); // same size → false (handled by exact match elsewhere)
    }
}

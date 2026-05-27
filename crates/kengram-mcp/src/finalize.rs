//! Shared deterministic post-tag pipeline for the worker drainer AND the
//! one-shot `kengram tag` CLI path.
//!
//! Both paths must apply the same deterministic post-process steps before
//! persisting tags. Historically the worker drainer ran topic-normalize +
//! people/entities disjointness while `kengram tag` skipped both, so the same
//! thought tagged via the two paths could land different tags. This module is
//! the single seam both call, so the two paths can never drift again.

use kengram_core::{ScopeVocab, Tags};

/// Apply the deterministic post-tag pipeline in one place.
///
/// Order is load-bearing:
/// 1. topic normalization against the scope vocab (v11; vocab-gated — a no-op
///    when `vocab` is `None` or carries no topics)
/// 2. people/entities disjointness (v12)
pub fn finalize_tags(tags: &mut Tags, vocab: Option<&ScopeVocab>) {
    if let Some(v) = vocab
        && !v.topics.is_empty()
    {
        tags.topics = crate::normalize::normalize_topics(&tags.topics, &v.topics);
    }
    crate::validate::enforce_people_entities_disjoint(tags);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags_with(people: &[&str], entities: &[&str], topics: &[&str]) -> Tags {
        Tags {
            people: people.iter().map(|s| s.to_string()).collect(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            topics: topics.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn applies_people_entities_disjointness_even_without_vocab() {
        // The `kengram tag` path used to skip this; finalize is the seam that
        // guarantees both paths enforce it.
        let mut t = tags_with(&["Sarah"], &["Sarah", "kengram"], &[]);
        finalize_tags(&mut t, None);
        assert_eq!(t.entities, vec!["kengram".to_string()]);
        assert_eq!(t.people, vec!["Sarah".to_string()]);
    }

    #[test]
    fn normalizes_topics_against_scope_vocab() {
        let vocab = ScopeVocab {
            topics: vec!["databases".to_string()],
            entities: vec![],
        };
        let mut t = tags_with(&[], &[], &["database"]);
        finalize_tags(&mut t, Some(&vocab));
        assert_eq!(t.topics, vec!["databases".to_string()]);
    }

    #[test]
    fn vocab_with_empty_topics_leaves_topics_unchanged() {
        let vocab = ScopeVocab {
            topics: vec![],
            entities: vec!["kengram".to_string()],
        };
        let mut t = tags_with(&[], &[], &["whatever"]);
        finalize_tags(&mut t, Some(&vocab));
        assert_eq!(t.topics, vec!["whatever".to_string()]);
    }
}

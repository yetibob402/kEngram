//! Shared deterministic post-tag pipeline for the worker drainer AND the
//! one-shot `kengram tag` CLI path.
//!
//! Both paths must apply the same deterministic post-process steps before
//! persisting tags. Historically the worker drainer ran topic-normalize +
//! people/entities disjointness while `kengram tag` skipped both, so the same
//! thought tagged via the two paths could land different tags. This module is
//! the single seam both call, so the two paths can never drift again — and it
//! is also where the v14 model-agnostic filters live, so they apply to every
//! tagger backend rather than only the LLM prompt.

use kengram_core::{Metadata, Scope, ScopeVocab, Tags};

/// Apply the deterministic post-tag pipeline in one place.
///
/// Order is load-bearing:
/// 1. topic normalization against the scope vocab (v11; vocab-gated — a no-op
///    when `vocab` is `None` or carries no topics)
/// 2. people/entities disjointness (v12)
/// 3. scope-identifier stripping from entities/people (v14)
/// 4. relationship-noun stripping from people (v14)
/// 5. `metadata.decision_type` → `decision_record` kind override (v14)
///
/// `known_scopes` is the corpus's set of scope strings (from
/// `kengram_storage::list_scopes`), fetched once per batch by the caller.
pub fn finalize_tags(
    tags: &mut Tags,
    metadata: &Metadata,
    own_scope: &Scope,
    vocab: Option<&ScopeVocab>,
    known_scopes: &[String],
) {
    if let Some(v) = vocab
        && !v.topics.is_empty()
    {
        tags.topics = crate::normalize::normalize_topics(&tags.topics, &v.topics);
    }
    crate::validate::enforce_people_entities_disjoint(tags);
    crate::filters::strip_scope_identifiers(tags, own_scope, known_scopes);
    crate::filters::strip_relationship_nouns(tags);
    crate::filters::apply_decision_type_override(tags, metadata);
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::TagKind;
    use serde_json::json;

    fn tags_with(people: &[&str], entities: &[&str], topics: &[&str]) -> Tags {
        Tags {
            people: people.iter().map(|s| s.to_string()).collect(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            topics: topics.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn scope(s: &str) -> Scope {
        Scope::new(s).unwrap()
    }

    #[test]
    fn applies_people_entities_disjointness_even_without_vocab() {
        // The `kengram tag` path used to skip this; finalize is the seam that
        // guarantees both paths enforce it.
        let mut t = tags_with(&["Sarah"], &["Sarah", "kengram"], &[]);
        finalize_tags(&mut t, &Metadata::empty(), &scope("work"), None, &[]);
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
        finalize_tags(
            &mut t,
            &Metadata::empty(),
            &scope("work"),
            Some(&vocab),
            &[],
        );
        assert_eq!(t.topics, vec!["databases".to_string()]);
    }

    #[test]
    fn vocab_with_empty_topics_leaves_topics_unchanged() {
        let vocab = ScopeVocab {
            topics: vec![],
            entities: vec!["kengram".to_string()],
        };
        let mut t = tags_with(&[], &[], &["whatever"]);
        finalize_tags(
            &mut t,
            &Metadata::empty(),
            &scope("work"),
            Some(&vocab),
            &[],
        );
        assert_eq!(t.topics, vec!["whatever".to_string()]);
    }

    #[test]
    fn applies_v14_filters_and_decision_override_together() {
        // A scope-id entity + a relationship-noun person + decision_type
        // metadata: finalize strips both and forces the kind in one pass.
        let mut t = Tags {
            people: vec!["Ron".to_string(), "buddy".to_string()],
            entities: vec!["rjf.personal".to_string(), "pgvector".to_string()],
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        finalize_tags(
            &mut t,
            &Metadata::from(json!({"decision_type": "build-spec"})),
            &scope("rjf.tech"),
            None,
            &["rjf.personal".to_string()],
        );
        assert_eq!(t.people, vec!["Ron".to_string()]);
        assert_eq!(t.entities, vec!["pgvector".to_string()]);
        assert_eq!(t.kind, Some(TagKind::DecisionRecord));
    }
}

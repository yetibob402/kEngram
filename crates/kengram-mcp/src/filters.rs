//! Deterministic, model-agnostic tag filters (v14 onward).
//!
//! These run in the shared [`crate::finalize`] seam, so they apply to every
//! tagger backend (LLM or sidecar) and to both the worker drainer and the
//! one-shot `kengram tag` path. Where the prompt teaches the model what to do,
//! these enforce structural invariants the model can't be relied on to honour
//! across the 12B→397B range — a regex/denylist gives byte-identical output on
//! every model, so corpus quality doesn't sag on the small ones.
//!
//! Each function is a pure transform over `&mut Tags`, mirroring
//! [`crate::validate::enforce_people_entities_disjoint`].

use kengram_core::{Metadata, Scope, TagKind, Tags};
use std::collections::HashSet;

/// Bare relationship / role nouns the LLM sometimes routes into `people`
/// (e.g. "Ron's buddy who plays Wordle" → `people: ["Ron", "buddy"]`).
/// Deliberately limited to unambiguous common nouns — NO first names. "Casey",
/// "Ron", "Mark", "Will", "Bob" are absent: they are real names that must
/// survive. This only catches the generic-noun failure mode.
const RELATIONSHIP_NOUNS: &[&str] = &[
    "buddy",
    "friend",
    "colleague",
    "coworker",
    "co-worker",
    "partner",
    "boss",
    "kid",
    "kids",
    "parent",
    "parents",
    "roommate",
    "teammate",
    "neighbor",
    "neighbour",
    "spouse",
    "sibling",
    "manager",
    "mentor",
    "mentee",
];

/// Strip strings that are Kengram scope identifiers from `entities` and
/// `people`. Data-driven: a string is only removed when it case-insensitively
/// equals the thought's own scope or one of the corpus's known scopes — never
/// a pattern guess, so legitimate dotted entities (`example.com`, `v1.2.3`)
/// survive. Addresses scope names like `rjf.personal` being emitted as
/// entities.
pub fn strip_scope_identifiers(tags: &mut Tags, own_scope: &Scope, known_scopes: &[String]) {
    let mut scope_set: HashSet<String> = known_scopes.iter().map(|s| s.to_lowercase()).collect();
    scope_set.insert(own_scope.as_str().to_lowercase());
    let is_scope = |s: &String| scope_set.contains(&s.to_lowercase());
    tags.entities.retain(|e| !is_scope(e));
    tags.people.retain(|p| !is_scope(p));
}

/// Strip bare relationship/role nouns (see [`RELATIONSHIP_NOUNS`]) from
/// `people`. Case-insensitive; `people` order is otherwise preserved.
pub fn strip_relationship_nouns(tags: &mut Tags) {
    let deny: HashSet<&str> = RELATIONSHIP_NOUNS.iter().copied().collect();
    tags.people
        .retain(|p| !deny.contains(p.to_lowercase().as_str()));
}

/// Force `kind = decision_record` when the thought's `metadata.decision_type`
/// is set to a non-empty string. Authoritative regardless of model — the
/// tagger never sees metadata, so this is the only place the signal can be
/// applied. Sets only `kind`; `action_items` hygiene is the prompt's job.
pub fn apply_decision_type_override(tags: &mut Tags, metadata: &Metadata) {
    let has_decision_type = metadata
        .as_value()
        .get("decision_type")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if has_decision_type {
        tags.kind = Some(TagKind::DecisionRecord);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tags_with(people: &[&str], entities: &[&str]) -> Tags {
        Tags {
            people: people.iter().map(|s| s.to_string()).collect(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn scope(s: &str) -> Scope {
        Scope::new(s).unwrap()
    }

    // --- strip_scope_identifiers ---

    #[test]
    fn scope_id_stripped_from_entities() {
        let mut t = tags_with(&[], &["rjf.personal", "pgvector"]);
        strip_scope_identifiers(
            &mut t,
            &scope("engram.m3.dogfood"),
            &["rjf.personal".to_string(), "engram.m3.dogfood".to_string()],
        );
        assert_eq!(t.entities, vec!["pgvector".to_string()]);
    }

    #[test]
    fn scope_id_stripped_from_people() {
        let mut t = tags_with(&["Ron", "engram.m3.dogfood"], &[]);
        strip_scope_identifiers(
            &mut t,
            &scope("rjf.personal"),
            &["engram.m3.dogfood".to_string()],
        );
        assert_eq!(t.people, vec!["Ron".to_string()]);
    }

    #[test]
    fn own_scope_stripped_even_when_absent_from_known() {
        let mut t = tags_with(&[], &["work.foo", "kengram"]);
        strip_scope_identifiers(&mut t, &scope("work.foo"), &[]);
        assert_eq!(t.entities, vec!["kengram".to_string()]);
    }

    #[test]
    fn dotted_non_scope_entities_survive() {
        // example.com / v1.2.3 are dotted but not known scopes — a regex would
        // wrongly strip them; the data-driven match must not.
        let mut t = tags_with(&[], &["example.com", "v1.2.3", "pgvector"]);
        strip_scope_identifiers(&mut t, &scope("rjf.tech"), &["rjf.tech".to_string()]);
        assert_eq!(
            t.entities,
            vec![
                "example.com".to_string(),
                "v1.2.3".to_string(),
                "pgvector".to_string()
            ]
        );
    }

    #[test]
    fn scope_match_is_case_insensitive_and_order_preserving() {
        let mut t = tags_with(&[], &["alpha", "RJF.Personal", "beta"]);
        strip_scope_identifiers(&mut t, &scope("x.y"), &["rjf.personal".to_string()]);
        assert_eq!(t.entities, vec!["alpha".to_string(), "beta".to_string()]);
    }

    // --- strip_relationship_nouns ---

    #[test]
    fn relationship_noun_stripped() {
        let mut t = tags_with(&["Ron", "buddy"], &[]);
        strip_relationship_nouns(&mut t);
        assert_eq!(t.people, vec!["Ron".to_string()]);
    }

    #[test]
    fn casey_trap_real_names_survive_alongside_noun() {
        // The use_mention fixture proves "buddy" co-occurs with real names.
        let mut t = tags_with(&["Casey", "Ron", "buddy"], &[]);
        strip_relationship_nouns(&mut t);
        assert_eq!(t.people, vec!["Casey".to_string(), "Ron".to_string()]);
    }

    #[test]
    fn relationship_noun_match_is_case_insensitive() {
        let mut t = tags_with(&["Buddy", "COLLEAGUE"], &[]);
        strip_relationship_nouns(&mut t);
        assert!(t.people.is_empty());
    }

    #[test]
    fn relationship_strip_empty_people_is_noop() {
        let mut t = tags_with(&[], &["kengram"]);
        strip_relationship_nouns(&mut t);
        assert!(t.people.is_empty());
        assert_eq!(t.entities, vec!["kengram".to_string()]);
    }

    // --- apply_decision_type_override ---

    #[test]
    fn decision_type_forces_kind() {
        let mut t = Tags {
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        apply_decision_type_override(
            &mut t,
            &Metadata::from(json!({"decision_type": "build-spec"})),
        );
        assert_eq!(t.kind, Some(TagKind::DecisionRecord));
    }

    #[test]
    fn absent_decision_type_leaves_kind() {
        let mut t = Tags {
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        apply_decision_type_override(&mut t, &Metadata::from(json!({"client_name": "x"})));
        assert_eq!(t.kind, Some(TagKind::Task));
    }

    #[test]
    fn empty_or_non_string_decision_type_leaves_kind() {
        let mut t = Tags {
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        apply_decision_type_override(&mut t, &Metadata::from(json!({"decision_type": ""})));
        assert_eq!(t.kind, Some(TagKind::Task));
        apply_decision_type_override(&mut t, &Metadata::from(json!({"decision_type": true})));
        assert_eq!(t.kind, Some(TagKind::Task));
    }
}

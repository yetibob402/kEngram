//! Post-extraction structural validators for the tag drainer (v12 onward).
//!
//! Where `normalize.rs` converges variants of the same concept to a
//! canonical form, this module enforces structural invariants that a
//! valid `Tags` struct must satisfy regardless of which model or prompt
//! emitted it. The first such invariant: `people` and `entities` must be
//! disjoint — a single string cannot legitimately appear in both arrays.
//!
//! Small LLMs occasionally route the same name into both arrays (probe E
//! in the v11 retag put `"Sarah"` in both). No prompt instruction
//! reliably prevents this, so the fix lives here as a deterministic
//! post-process step. Person wins on tie: the `people` classification is
//! more semantically constrained than the open-vocabulary `entities`
//! field, so dropping the entity duplicate preserves the more specific
//! commitment.

use engram_core::Tags;
use std::collections::HashSet;

/// Strip any `entities` entry whose lowercased form matches a `people`
/// entry. Mutates in place. The `people` array is untouched.
pub fn enforce_people_entities_disjoint(tags: &mut Tags) {
    if tags.people.is_empty() || tags.entities.is_empty() {
        return;
    }
    let people_lower: HashSet<String> = tags.people.iter().map(|p| p.to_lowercase()).collect();
    tags.entities
        .retain(|e| !people_lower.contains(&e.to_lowercase()));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags_with(people: &[&str], entities: &[&str]) -> Tags {
        Tags {
            people: people.iter().map(|s| s.to_string()).collect(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn disjoint_is_noop() {
        let mut t = tags_with(&["Sarah"], &["engram", "pgvector"]);
        let before_people = t.people.clone();
        let before_entities = t.entities.clone();
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(t.people, before_people);
        assert_eq!(t.entities, before_entities);
    }

    #[test]
    fn exact_duplicate_removed_from_entities() {
        let mut t = tags_with(&["Sarah"], &["Sarah", "engram"]);
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(t.people, vec!["Sarah".to_string()]);
        assert_eq!(t.entities, vec!["engram".to_string()]);
    }

    #[test]
    fn case_insensitive_match_removed() {
        // "Sarah" in people, "sarah" in entities — lowercased comparison
        // catches the duplicate even when casing differs.
        let mut t = tags_with(&["Sarah"], &["sarah", "engram"]);
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(t.entities, vec!["engram".to_string()]);
    }

    #[test]
    fn multiple_duplicates_all_removed() {
        let mut t = tags_with(&["Sarah", "Ron"], &["engram", "Sarah", "Ron", "pgvector"]);
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(
            t.entities,
            vec!["engram".to_string(), "pgvector".to_string()]
        );
    }

    #[test]
    fn empty_entities_is_noop() {
        let mut t = tags_with(&["Sarah", "Ron"], &[]);
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(t.people, vec!["Sarah".to_string(), "Ron".to_string()]);
        assert!(t.entities.is_empty());
    }

    #[test]
    fn empty_people_is_noop() {
        let mut t = tags_with(&[], &["engram", "pgvector"]);
        enforce_people_entities_disjoint(&mut t);
        assert!(t.people.is_empty());
        assert_eq!(
            t.entities,
            vec!["engram".to_string(), "pgvector".to_string()]
        );
    }

    #[test]
    fn order_preserved_on_partial_strip() {
        // After removing the duplicate, the surviving entities should be
        // in the same relative order as the input.
        let mut t = tags_with(
            &["Sarah"],
            &["alpha", "Sarah", "beta", "gamma", "sarah", "delta"],
        );
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(
            t.entities,
            vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string(),
                "delta".to_string(),
            ]
        );
    }

    #[test]
    fn people_array_untouched() {
        // The validator only mutates entities; people stays exactly as the
        // LLM emitted it, even when duplicates of itself appear.
        let mut t = tags_with(&["Sarah", "Sarah"], &["Sarah"]);
        enforce_people_entities_disjoint(&mut t);
        assert_eq!(t.people, vec!["Sarah".to_string(), "Sarah".to_string()]);
        assert!(t.entities.is_empty());
    }
}

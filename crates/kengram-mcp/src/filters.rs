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

fn metadata_str<'a>(metadata: &'a Metadata, key: &str) -> Option<&'a str> {
    metadata.as_value().get(key).and_then(|v| v.as_str())
}

fn slugify_domain_part(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in s.trim().chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if (ch == '-' || ch == '_' || ch == ' ' || ch == '.')
            && !last_dash
            && !out.is_empty()
        {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() { None } else { Some(out) }
}

fn is_imported_archive(metadata: &Metadata) -> bool {
    [
        metadata_str(metadata, "import"),
        metadata_str(metadata, "source_file"),
    ]
    .into_iter()
    .flatten()
    .any(|s| {
        let s = s.to_lowercase();
        s.contains("archive") || s.contains("macbook_air_archive")
    })
}

/// Imported historical archive digests often contain sections like
/// "Open Followups" from months-old subagent reports. They are useful
/// reference material, but they are not live commitments or newly-authored
/// decisions for the current operator. Strip action_items and avoid live
/// task/decision kind unless source metadata explicitly says it is a
/// decision artifact.
pub fn apply_imported_archive_guard(tags: &mut Tags, metadata: &Metadata) {
    if !is_imported_archive(metadata) {
        return;
    }
    tags.action_items.clear();
    tags.people.retain(|p| !looks_like_lowercase_handle(p));
    let explicit_decision =
        metadata_str(metadata, "decision_type").is_some_and(|s| !s.trim().is_empty());
    if !explicit_decision && matches!(tags.kind, Some(TagKind::Task | TagKind::DecisionRecord)) {
        tags.kind = Some(TagKind::Reference);
    }
}

fn looks_like_lowercase_handle(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.len() >= 3
        && !trimmed.contains(char::is_whitespace)
        && trimmed.chars().any(|ch| ch.is_ascii_alphabetic())
        && trimmed.chars().all(|ch| {
            ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-' || ch == '.'
        })
}

fn looks_like_path_or_route(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.contains("/api/")
        || trimmed.contains("/src/")
        || trimmed.contains("/crates/")
        || trimmed.ends_with(".rs")
        || trimmed.ends_with(".ts")
        || trimmed.ends_with(".tsx")
        || trimmed.ends_with(".js")
        || trimmed.ends_with(".jsx")
        || trimmed.ends_with(".py")
        || trimmed.ends_with(".sql")
        || trimmed.ends_with(".toml")
        || trimmed.ends_with(".json")
        || trimmed.ends_with(".md")
}

/// Strip filesystem paths and API route paths from entities. They are usually
/// evidence inside an operational report, not named objects with durable
/// identity. Retrieval aliases may still carry explicit identifiers, but the
/// alias cleaner below applies a tighter cap.
pub fn strip_path_like_entities(tags: &mut Tags) {
    tags.entities.retain(|e| !looks_like_path_or_route(e));
}

/// Clean retrieval aliases into short grounded query terms. The prompt already
/// asks for this; this guard keeps noisy model output from bloating JSONB and
/// future retrieval filters.
pub fn clean_retrieval_aliases(tags: &mut Tags) {
    let mut seen = HashSet::new();
    tags.retrieval_aliases.retain(|alias| {
        let trimmed = alias.trim();
        if trimmed.is_empty() || trimmed.len() > 80 || looks_like_path_or_route(trimmed) {
            return false;
        }
        seen.insert(trimmed.to_lowercase())
    });
    for alias in &mut tags.retrieval_aliases {
        *alias = alias.trim().to_string();
    }
    tags.retrieval_aliases.truncate(6);
}

/// Normalize the second routing axis. This is deliberately conservative:
/// unknown free-form domains are nulled rather than persisted as drift.
pub fn normalize_domain_scope(tags: &mut Tags) {
    let Some(raw) = tags.domain_scope.as_deref() else {
        return;
    };
    let normalized = raw.trim().to_lowercase().replace('_', "-");
    let normalized = normalized.trim_matches('/').to_string();
    let mapped = if normalized.is_empty()
        || normalized == "knox"
        || normalized == "agents/knox"
        || normalized == "sessions/knox"
        || normalized.starts_with("agents/")
        || normalized.starts_with("sessions/")
    {
        None
    } else if normalized == "argus"
        || normalized == "kengram"
        || normalized == "memory"
        || normalized == "fleet"
        || normalized == "ops"
        || normalized == "platform"
        || normalized.starts_with("infra/")
    {
        Some("infra".to_string())
    } else if normalized == "decision" || normalized == "decision-records" {
        Some("decisions".to_string())
    } else if normalized == "decisions"
        || normalized == "infra"
        || normalized.starts_with("apps/")
        || normalized.starts_with("customers/")
    {
        Some(normalized)
    } else {
        None
    };
    tags.domain_scope = mapped;
}

/// Authoritative metadata beats the model for domain routing. Imported
/// archive digests commonly carry `metadata.project`; customer ingests may
/// carry `customer`, `client`, or `customer_slug`.
pub fn apply_metadata_domain_override(tags: &mut Tags, metadata: &Metadata) {
    if let Some(customer) = ["customer_slug", "customer", "client"]
        .into_iter()
        .filter_map(|k| metadata_str(metadata, k))
        .find_map(slugify_domain_part)
    {
        tags.domain_scope = Some(format!("customers/{customer}"));
        return;
    }
    if let Some(project) = metadata_str(metadata, "project").and_then(slugify_domain_part) {
        tags.domain_scope = Some(format!("apps/{project}"));
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
    fn imported_archive_guard_strips_historical_action_items() {
        let mut t = Tags {
            action_items: vec!["implement distributed rate limiting".to_string()],
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        apply_imported_archive_guard(
            &mut t,
            &Metadata::from(json!({"import": "kengram_macbook_air_archive_2026_06"})),
        );
        assert!(t.action_items.is_empty());
        assert_eq!(t.kind, Some(TagKind::Reference));
    }

    #[test]
    fn imported_archive_guard_demotes_unreviewed_archive_decision_records() {
        let mut t = Tags {
            action_items: vec!["follow up".to_string()],
            kind: Some(TagKind::DecisionRecord),
            ..Default::default()
        };
        apply_imported_archive_guard(
            &mut t,
            &Metadata::from(json!({"source_file": "old/archive/session.jsonl"})),
        );
        assert!(t.action_items.is_empty());
        assert_eq!(t.kind, Some(TagKind::Reference));
    }

    #[test]
    fn imported_archive_guard_preserves_explicit_decision_metadata() {
        let mut t = Tags {
            action_items: vec!["follow up".to_string()],
            kind: Some(TagKind::DecisionRecord),
            ..Default::default()
        };
        apply_imported_archive_guard(
            &mut t,
            &Metadata::from(
                json!({"source_file": "old/archive/session.jsonl", "decision_type": "build-spec"}),
            ),
        );
        assert!(t.action_items.is_empty());
        assert_eq!(t.kind, Some(TagKind::DecisionRecord));
    }

    #[test]
    fn imported_archive_guard_strips_lowercase_operator_handles() {
        let mut t = Tags {
            people: vec![
                "hrbekr".to_string(),
                "Bob".to_string(),
                "Sarah Jane".to_string(),
            ],
            ..Default::default()
        };
        apply_imported_archive_guard(
            &mut t,
            &Metadata::from(json!({"import": "kengram_macbook_air_archive_2026_06"})),
        );
        assert_eq!(t.people, vec!["Bob".to_string(), "Sarah Jane".to_string()]);
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

    #[test]
    fn path_like_entities_are_stripped() {
        let mut t = Tags {
            entities: vec![
                "/api/auth/delete-user.js".to_string(),
                "kengram".to_string(),
                "crates/kengram-core/src/tags.rs".to_string(),
            ],
            ..Default::default()
        };
        strip_path_like_entities(&mut t);
        assert_eq!(t.entities, vec!["kengram".to_string()]);
    }

    #[test]
    fn retrieval_aliases_are_deduped_capped_and_path_filtered() {
        let mut t = Tags {
            retrieval_aliases: vec![
                " semantic memory ".to_string(),
                "semantic memory".to_string(),
                "/api/auth/delete-user.js".to_string(),
                "x".repeat(81),
                "operator recall".to_string(),
            ],
            ..Default::default()
        };
        clean_retrieval_aliases(&mut t);
        assert_eq!(
            t.retrieval_aliases,
            vec!["semantic memory".to_string(), "operator recall".to_string()]
        );
    }

    #[test]
    fn domain_scope_is_normalized_conservatively() {
        let mut t = Tags {
            domain_scope: Some("kEngram".to_string()),
            ..Default::default()
        };
        normalize_domain_scope(&mut t);
        assert_eq!(t.domain_scope.as_deref(), Some("infra"));

        t.domain_scope = Some("agents/knox".to_string());
        normalize_domain_scope(&mut t);
        assert_eq!(t.domain_scope, None);

        t.domain_scope = Some("apps/Kengram".to_string());
        normalize_domain_scope(&mut t);
        assert_eq!(t.domain_scope.as_deref(), Some("apps/kengram"));
    }

    #[test]
    fn metadata_project_overrides_domain_scope() {
        let mut t = Tags {
            domain_scope: Some("infra".to_string()),
            ..Default::default()
        };
        apply_metadata_domain_override(&mut t, &Metadata::from(json!({"project": "MyLakeAccess"})));
        assert_eq!(t.domain_scope.as_deref(), Some("apps/mylakeaccess"));
    }

    #[test]
    fn metadata_customer_overrides_project_domain_scope() {
        let mut t = Tags {
            domain_scope: Some("apps/foo".to_string()),
            ..Default::default()
        };
        apply_metadata_domain_override(
            &mut t,
            &Metadata::from(json!({"project": "MLA", "customer": "Blue Water"})),
        );
        assert_eq!(t.domain_scope.as_deref(), Some("customers/blue-water"));
    }
}

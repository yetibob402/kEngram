//! `Tags` — the LLM-extracted metadata sidecar attached to each thought.
//!
//! Replaces the M3 facts pipeline. Where facts were full sentences with
//! provenance and embeddings of their own, tags are bare metadata fields
//! attached to the thought row itself: who is mentioned, what tasks the
//! thought commits to, what topics it's about, and a single
//! kind-classification.
//!
//! Schema lives on the wire as a flat JSON object. Default for every field
//! is the empty value (empty vec / `None`), so deserializing `{}` yields a
//! valid `Tags::default()`. New tagger versions can add fields without
//! breaking older readers.

use crate::Metadata;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const MAX_RETRIEVAL_ALIASES: usize = 6;
pub const MAX_RETRIEVAL_ALIAS_LEN: usize = 80;

/// LLM-extracted metadata attached to a single thought. See the
/// `BUNDLED_TAGGER_PROMPT` for the field-by-field semantics.
///
/// `relations` does NOT live on this struct (despite the LLM emitting
/// them in the same response). Tagger-emitted relations are routed
/// directly to `thought_links` via the drainer's
/// `kengram_mcp::apply_tagger_relations` helper; persisting them again in
/// the `tags` JSONB was duplication. The tagger's full output (Tags +
/// relations) is represented as [`crate::TagOutput`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tags {
    #[serde(default)]
    pub people: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub retrieval_aliases: Vec<String>,
    #[serde(default)]
    pub domain_scope: Option<String>,
    #[serde(default)]
    pub action_items: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub dates_mentioned: Vec<String>,
    #[serde(default)]
    pub kind: Option<TagKind>,
}

/// Top-N established tag terms from a given scope, supplied to the tagger as
/// controlled-vocabulary hints. Helps the tagger emit consistent terms when
/// it sees similar concepts in different prose, addressing v1's phrase-driven
/// divergence at corpus level.
///
/// Empty vectors are valid — they signal "no established vocabulary yet" and
/// the tagger falls back to free-form term coinage.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScopeVocab {
    pub topics: Vec<String>,
    pub entities: Vec<String>,
}

impl ScopeVocab {
    pub fn is_empty(&self) -> bool {
        self.topics.is_empty() && self.entities.is_empty()
    }
}

/// Single high-level classification a thought belongs to. `PersonNote`
/// serializes as `"person_note"` and `DecisionRecord` as `"decision_record"`
/// per the snake_case rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagKind {
    Observation,
    Task,
    Idea,
    Reference,
    PersonNote,
    Session,
    /// A choice already made and recorded — past-tense indicative ("we
    /// decided", "chose X over Y"). Distinct from `Task` (forward-looking
    /// work) and `Idea` (a proposal not yet decided). Added in tagger v14.
    DecisionRecord,
}

/// Clean a model-emitted retrieval alias into a short grounded query term.
pub fn normalize_retrieval_alias(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_RETRIEVAL_ALIAS_LEN
        || looks_like_path_or_route(trimmed)
    {
        return None;
    }
    Some(trimmed.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Normalize, dedupe, and cap retrieval aliases while preserving first-seen order.
pub fn normalize_retrieval_aliases<I, S>(aliases: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut seen = HashSet::new();
    let mut cleaned = Vec::new();
    for alias in aliases {
        let Some(normalized) = normalize_retrieval_alias(alias.as_ref()) else {
            continue;
        };
        if seen.insert(normalized.to_lowercase()) {
            cleaned.push(normalized);
        }
        if cleaned.len() >= MAX_RETRIEVAL_ALIASES {
            break;
        }
    }
    cleaned
}

/// Normalize the soft domain-routing axis. Scope/session/agent labels are not domains.
pub fn normalize_domain_scope(raw: &str) -> Option<String> {
    let normalized = raw
        .trim()
        .to_lowercase()
        .replace('_', "-")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-");
    let normalized = normalized.trim_matches('/').to_string();
    if normalized.is_empty()
        || normalized == "knox"
        || normalized.starts_with("agents/")
        || normalized.starts_with("sessions/")
    {
        return None;
    }
    if matches!(
        normalized.as_str(),
        "argus" | "kengram" | "memory" | "fleet" | "ops" | "platform"
    ) || normalized.starts_with("infra/")
    {
        return Some("infra".to_string());
    }
    if matches!(normalized.as_str(), "decision" | "decision-records") {
        return Some("decisions".to_string());
    }
    if normalized == "decisions"
        || normalized == "infra"
        || normalized.starts_with("apps/")
        || normalized.starts_with("customers/")
    {
        return Some(normalized);
    }
    None
}

pub fn normalize_routing_fields(tags: &mut Tags) {
    tags.retrieval_aliases = normalize_retrieval_aliases(tags.retrieval_aliases.iter());
    tags.domain_scope = tags
        .domain_scope
        .as_deref()
        .and_then(normalize_domain_scope);
}

/// Authoritative customer/project metadata overrides model-emitted domain scope.
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

fn metadata_str<'a>(metadata: &'a Metadata, key: &str) -> Option<&'a str> {
    metadata
        .as_value()
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn slugify_domain_part(raw: &str) -> Option<String> {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in raw.trim().to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_dash = false;
        } else if matches!(ch, '-' | '_' | ' ' | '.' | '/') && !slug.is_empty() && !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    (!slug.is_empty()).then_some(slug)
}

fn looks_like_path_or_route(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.contains("/api/")
        || trimmed.contains("/src/")
        || trimmed.contains("/crates/")
        || [
            ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".sql", ".toml", ".json", ".md", ".yaml",
            ".yml", ".sh",
        ]
        .iter()
        .any(|suffix| trimmed.to_ascii_lowercase().ends_with(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_default_round_trips_as_empty_object_shape() {
        let t = Tags::default();
        let json = serde_json::to_value(&t).unwrap();
        // Default emits every field as empty/null, not as a literal `{}`,
        // but the inverse direction below confirms `{}` is accepted as default.
        assert_eq!(json["people"], serde_json::json!([]));
        assert_eq!(json["entities"], serde_json::json!([]));
        assert_eq!(json["retrieval_aliases"], serde_json::json!([]));
        assert_eq!(json["domain_scope"], serde_json::Value::Null);
        assert_eq!(json["action_items"], serde_json::json!([]));
        assert_eq!(json["topics"], serde_json::json!([]));
        assert_eq!(json["dates_mentioned"], serde_json::json!([]));
        assert_eq!(json["kind"], serde_json::Value::Null);
    }

    #[test]
    fn v1_shape_without_entities_deserializes_with_empty_entities() {
        // Backward-compat: rows tagged under v1 (no `entities` key) must still
        // parse, with `entities` defaulting to `vec![]`.
        let v1_json = r#"{
            "people":["Sarah"],
            "action_items":[],
            "topics":["rust"],
            "dates_mentioned":[],
            "kind":"observation"
        }"#;
        let t: Tags = serde_json::from_str(v1_json).unwrap();
        assert_eq!(t.entities, Vec::<String>::new());
        assert_eq!(t.retrieval_aliases, Vec::<String>::new());
        assert_eq!(t.domain_scope, None);
        assert_eq!(t.topics, vec!["rust".to_string()]);
        assert_eq!(t.kind, Some(TagKind::Observation));
    }

    #[test]
    fn empty_object_deserializes_into_default_tags() {
        let t: Tags = serde_json::from_str("{}").unwrap();
        assert_eq!(t, Tags::default());
    }

    #[test]
    fn full_field_serde_roundtrip() {
        let t = Tags {
            people: vec!["Sarah".to_string(), "Ron".to_string()],
            entities: vec!["kengram".to_string(), "pgvector".to_string()],
            retrieval_aliases: vec!["memory search".to_string()],
            domain_scope: Some("infra".to_string()),
            action_items: vec!["fix the login bug".to_string()],
            topics: vec!["rust".to_string(), "build-systems".to_string()],
            dates_mentioned: vec!["next Thursday".to_string(), "Q3".to_string()],
            kind: Some(TagKind::Task),
        };
        let json = serde_json::to_string(&t).unwrap();
        let parsed: Tags = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }

    #[test]
    fn historical_shape_with_relations_key_still_parses() {
        // Migration 0011 dropped `tags.relations` from the persisted JSONB,
        // but rows that pre-date the migration may transiently still carry
        // the key. Tags deserializes without `deny_unknown_fields`, so the
        // stray key is silently ignored. This pins that behavior so a future
        // serde tightening doesn't accidentally break old-shape rows.
        let pre_migration_json = r#"{
            "people":["Sarah"],
            "entities":["kengram"],
            "action_items":[],
            "topics":["rust"],
            "dates_mentioned":[],
            "kind":"observation",
            "relations":[{"relation":"references","to_kind":"url","to_value":"https://x.io"}]
        }"#;
        let t: Tags = serde_json::from_str(pre_migration_json).unwrap();
        assert_eq!(t.entities, vec!["kengram".to_string()]);
        assert_eq!(t.kind, Some(TagKind::Observation));
    }

    #[test]
    fn scope_vocab_is_empty_helper() {
        assert!(ScopeVocab::default().is_empty());
        let v = ScopeVocab {
            topics: vec!["rust".to_string()],
            entities: vec![],
        };
        assert!(!v.is_empty());
    }

    #[test]
    fn tag_kind_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&TagKind::Observation).unwrap(),
            "\"observation\""
        );
        assert_eq!(serde_json::to_string(&TagKind::Task).unwrap(), "\"task\"");
        assert_eq!(serde_json::to_string(&TagKind::Idea).unwrap(), "\"idea\"");
        assert_eq!(
            serde_json::to_string(&TagKind::Reference).unwrap(),
            "\"reference\""
        );
        assert_eq!(
            serde_json::to_string(&TagKind::PersonNote).unwrap(),
            "\"person_note\""
        );
        assert_eq!(
            serde_json::to_string(&TagKind::Session).unwrap(),
            "\"session\""
        );
        assert_eq!(
            serde_json::to_string(&TagKind::DecisionRecord).unwrap(),
            "\"decision_record\""
        );
    }

    #[test]
    fn tag_kind_deserializes_snake_case() {
        let k: TagKind = serde_json::from_str("\"person_note\"").unwrap();
        assert_eq!(k, TagKind::PersonNote);
        let k: TagKind = serde_json::from_str("\"observation\"").unwrap();
        assert_eq!(k, TagKind::Observation);
        let k: TagKind = serde_json::from_str("\"decision_record\"").unwrap();
        assert_eq!(k, TagKind::DecisionRecord);
    }

    #[test]
    fn kind_null_deserializes_to_none() {
        let json = r#"{"people":[],"entities":[],"action_items":[],"topics":[],"dates_mentioned":[],"kind":null}"#;
        let t: Tags = serde_json::from_str(json).unwrap();
        assert!(t.kind.is_none());
    }

    #[test]
    fn domain_scope_json_null_deserializes_to_none() {
        let json = r#"{"domain_scope":null}"#;
        let t: Tags = serde_json::from_str(json).unwrap();
        assert_eq!(t.domain_scope, None);
    }

    #[test]
    fn retrieval_alias_normalization_trims_dedupes_caps_and_strips_paths() {
        let aliases = normalize_retrieval_aliases([
            "  Memory Search  ",
            "memory search",
            "/api/search",
            "valid-one",
            "valid-two",
            "valid-three",
            "valid-four",
            "valid-five",
            "valid-six",
            "valid-seven",
        ]);
        assert_eq!(
            aliases,
            vec![
                "Memory Search",
                "valid-one",
                "valid-two",
                "valid-three",
                "valid-four",
                "valid-five"
            ]
        );
    }

    #[test]
    fn domain_scope_normalization_is_conservative() {
        assert_eq!(normalize_domain_scope("Kengram"), Some("infra".to_string()));
        assert_eq!(
            normalize_domain_scope("Customers/Bluewater"),
            Some("customers/bluewater".to_string())
        );
        assert_eq!(normalize_domain_scope("agents/knox"), None);
        assert_eq!(normalize_domain_scope(" sessions/neo "), None);
        assert_eq!(normalize_domain_scope("random free form"), None);
    }

    #[test]
    fn metadata_domain_override_prefers_customer_then_project() {
        let mut tags = Tags {
            domain_scope: Some("infra".to_string()),
            ..Tags::default()
        };
        apply_metadata_domain_override(
            &mut tags,
            &Metadata::from(serde_json::json!({
                "customer": "Bluewater North",
                "project": "Should Lose"
            })),
        );
        assert_eq!(
            tags.domain_scope.as_deref(),
            Some("customers/bluewater-north")
        );

        let mut tags = Tags::default();
        apply_metadata_domain_override(
            &mut tags,
            &Metadata::from(serde_json::json!({"project": "MLA A360"})),
        );
        assert_eq!(tags.domain_scope.as_deref(), Some("apps/mla-a360"));
    }
}

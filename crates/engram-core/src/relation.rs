//! Relation types — the M5 thought-to-thought graph layer.
//!
//! Edges live in `thought_links` (migration 0007). The closed relation
//! vocabulary is `replaces`, `requires`, `references`, `supports`,
//! `belongs_to`, `decided_by`, `refines` (M5 shipped six; M5.1 added
//! `supports`).
//!
//! M5.2 added heterogeneous targets — a link's `to` side can be a thought,
//! an entity, a person, or a URL (see [`LinkTarget`]) — and soft-delete on
//! the edge row (`deleted_at`). Tagger-extracted relations remain M5.x.

use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::ThoughtId;

/// Closed relation vocabulary. Serializes to/from snake_case strings on the
/// wire and matches the CHECK constraint values on `thought_links.relation`.
///
/// v1 shipped six relations; M5.1 added `supports` after day-one dogfood
/// revealed `references` was over-firing on what was actually evidence /
/// corroboration ("experimental result confirming a claim"). The split
/// separates "I cite for context" (`references`) from "I confirm a claim"
/// (`supports`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Replaces,
    Requires,
    References,
    Supports,
    BelongsTo,
    DecidedBy,
    Refines,
}

impl RelationKind {
    /// Stable string form matching the DB CHECK constraint values. Mirrors
    /// the snake_case serde rename so callers building SQL directly (e.g.
    /// the storage layer's relation-filter `ANY($N::text[])`) don't need
    /// to round-trip through serde_json.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Replaces => "replaces",
            Self::Requires => "requires",
            Self::References => "references",
            Self::Supports => "supports",
            Self::BelongsTo => "belongs_to",
            Self::DecidedBy => "decided_by",
            Self::Refines => "refines",
        }
    }

    /// All variants, useful for documentation and exhaustive iteration.
    pub const ALL: [RelationKind; 7] = [
        Self::Replaces,
        Self::Requires,
        Self::References,
        Self::Supports,
        Self::BelongsTo,
        Self::DecidedBy,
        Self::Refines,
    ];
}

impl fmt::Display for RelationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RelationKind {
    type Err = UnknownRelationKind;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "replaces" => Ok(Self::Replaces),
            "requires" => Ok(Self::Requires),
            "references" => Ok(Self::References),
            "supports" => Ok(Self::Supports),
            "belongs_to" => Ok(Self::BelongsTo),
            "decided_by" => Ok(Self::DecidedBy),
            "refines" => Ok(Self::Refines),
            other => Err(UnknownRelationKind(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "unknown relation kind: {0:?} (expected one of replaces, requires, references, supports, belongs_to, decided_by, refines)"
)]
pub struct UnknownRelationKind(pub String);

/// Provenance of a link: who put the edge in `thought_links`. M5 inserts
/// only `Agent`; `Tagger` is reserved for the M5.x tagger-extraction work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkSource {
    Agent,
    Tagger,
}

impl LinkSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Tagger => "tagger",
        }
    }
}

impl FromStr for LinkSource {
    type Err = UnknownLinkSource;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "agent" => Ok(Self::Agent),
            "tagger" => Ok(Self::Tagger),
            other => Err(UnknownLinkSource(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown link source: {0:?} (expected 'agent' or 'tagger')")]
pub struct UnknownLinkSource(pub String);

/// Traversal direction for `fetch_related_thoughts`. `Outbound` walks edges
/// where the query thought is on the `from` side; `Inbound` walks edges
/// where it's on the `to` side; `Both` returns the union with a stable order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkDirection {
    Outbound,
    Inbound,
    #[default]
    Both,
}

impl LinkDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Outbound => "outbound",
            Self::Inbound => "inbound",
            Self::Both => "both",
        }
    }
}

impl FromStr for LinkDirection {
    type Err = UnknownLinkDirection;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "outbound" => Ok(Self::Outbound),
            "inbound" => Ok(Self::Inbound),
            "both" => Ok(Self::Both),
            other => Err(UnknownLinkDirection(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown link direction: {0:?} (expected 'outbound', 'inbound', or 'both')")]
pub struct UnknownLinkDirection(pub String);

/// Polymorphic target of a link (M5.2). The `from` side of a link is
/// always a thought; the `to` side can be a thought, a free-text entity
/// name, a free-text person name, or a URL.
///
/// Wire shape uses `{"kind": "<variant>", "value": "<string>"}` —
/// matches the `to_kind` / `to_value` columns on `thought_links` and the
/// flattened shape used by the MCP layer (`to_thought_id` / `to_entity` /
/// `to_person` / `to_url` args on `link_thoughts`).
///
/// Entity and person targets are free-text strings; engram has no
/// first-class entity or person table in v1. URL targets must start with
/// `http://` or `https://` (DB-side CHECK).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum LinkTarget {
    Thought(ThoughtId),
    Entity(String),
    Person(String),
    Url(String),
}

impl LinkTarget {
    /// Stable discriminator string matching the `thought_links.to_kind`
    /// CHECK constraint values.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Thought(_) => "thought",
            Self::Entity(_) => "entity",
            Self::Person(_) => "person",
            Self::Url(_) => "url",
        }
    }

    /// Stable string form of the target value matching the
    /// `thought_links.to_value` generated column. Thought targets
    /// stringify the UUID; other kinds return the underlying string.
    pub fn value_str(&self) -> String {
        match self {
            Self::Thought(id) => id.as_uuid().to_string(),
            Self::Entity(name) | Self::Person(name) => name.clone(),
            Self::Url(url) => url.clone(),
        }
    }

    /// Extracts the thought id when the target is a thought; `None` otherwise.
    /// Convenient for storage-layer code writing the four typed columns.
    pub fn as_thought_id(&self) -> Option<&ThoughtId> {
        if let Self::Thought(id) = self {
            Some(id)
        } else {
            None
        }
    }

    /// Extracts the entity name when the target is an entity; `None` otherwise.
    pub fn as_entity(&self) -> Option<&str> {
        if let Self::Entity(name) = self {
            Some(name.as_str())
        } else {
            None
        }
    }

    /// Extracts the person name when the target is a person; `None` otherwise.
    pub fn as_person(&self) -> Option<&str> {
        if let Self::Person(name) = self {
            Some(name.as_str())
        } else {
            None
        }
    }

    /// Extracts the URL when the target is a URL; `None` otherwise.
    pub fn as_url(&self) -> Option<&str> {
        if let Self::Url(url) = self {
            Some(url.as_str())
        } else {
            None
        }
    }
}

/// Stable identity for a row in `thought_links`. Wraps a UUID so the type
/// system can distinguish a link id from a thought id (mirrors `ThoughtId`'s
/// shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LinkId(pub Uuid);

impl LinkId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    pub fn into_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for LinkId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for LinkId {
    fn from(u: Uuid) -> Self {
        Self(u)
    }
}

impl fmt::Display for LinkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for LinkId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::from_str(s).map(Self)
    }
}

/// The full row-shape of a `thought_links` edge.
///
/// M5.2 generalized the target from `to_thought_id: ThoughtId` to a
/// polymorphic `target: LinkTarget` and added the `deleted_at` soft-delete
/// marker. Live (non-soft-deleted) rows have `deleted_at = None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThoughtLink {
    pub id: LinkId,
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub target: LinkTarget,
    pub source: LinkSource,
    pub note: Option<String>,
    pub created_at: OffsetDateTime,
    pub deleted_at: Option<OffsetDateTime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_kind_serializes_to_snake_case() {
        assert_eq!(
            serde_json::to_string(&RelationKind::Replaces).unwrap(),
            "\"replaces\""
        );
        assert_eq!(
            serde_json::to_string(&RelationKind::BelongsTo).unwrap(),
            "\"belongs_to\""
        );
        assert_eq!(
            serde_json::to_string(&RelationKind::DecidedBy).unwrap(),
            "\"decided_by\""
        );
        assert_eq!(
            serde_json::to_string(&RelationKind::Refines).unwrap(),
            "\"refines\""
        );
    }

    #[test]
    fn relation_kind_deserializes_from_snake_case() {
        let k: RelationKind = serde_json::from_str("\"belongs_to\"").unwrap();
        assert_eq!(k, RelationKind::BelongsTo);
        let k: RelationKind = serde_json::from_str("\"refines\"").unwrap();
        assert_eq!(k, RelationKind::Refines);
    }

    #[test]
    fn relation_kind_as_str_matches_serde() {
        for k in RelationKind::ALL {
            let serde = serde_json::to_value(k).unwrap();
            assert_eq!(serde.as_str().unwrap(), k.as_str());
        }
    }

    #[test]
    fn relation_kind_from_str_round_trips() {
        for k in RelationKind::ALL {
            let parsed: RelationKind = k.as_str().parse().unwrap();
            assert_eq!(parsed, k);
        }
    }

    #[test]
    fn relation_kind_from_str_rejects_unknown() {
        let err = "contradicts".parse::<RelationKind>().unwrap_err();
        assert!(err.to_string().contains("contradicts"));
    }

    #[test]
    fn link_source_round_trips() {
        for src in [LinkSource::Agent, LinkSource::Tagger] {
            let s = serde_json::to_string(&src).unwrap();
            let parsed: LinkSource = serde_json::from_str(&s).unwrap();
            assert_eq!(parsed, src);
            assert_eq!(src.as_str().parse::<LinkSource>().unwrap(), src);
        }
    }

    #[test]
    fn link_direction_default_is_both() {
        assert_eq!(LinkDirection::default(), LinkDirection::Both);
    }

    #[test]
    fn link_direction_round_trips() {
        for dir in [
            LinkDirection::Outbound,
            LinkDirection::Inbound,
            LinkDirection::Both,
        ] {
            let s = serde_json::to_string(&dir).unwrap();
            let parsed: LinkDirection = serde_json::from_str(&s).unwrap();
            assert_eq!(parsed, dir);
            assert_eq!(dir.as_str().parse::<LinkDirection>().unwrap(), dir);
        }
    }

    #[test]
    fn link_id_round_trips_via_uuid() {
        let id = LinkId::new();
        let s = id.to_string();
        let parsed: LinkId = s.parse().unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn link_target_kind_str_matches_variant() {
        let tid = ThoughtId::new();
        assert_eq!(LinkTarget::Thought(tid).kind_str(), "thought");
        assert_eq!(LinkTarget::Entity("foo".into()).kind_str(), "entity");
        assert_eq!(LinkTarget::Person("alice".into()).kind_str(), "person");
        assert_eq!(
            LinkTarget::Url("https://example.com".into()).kind_str(),
            "url"
        );
    }

    #[test]
    fn link_target_value_str_round_trips() {
        let tid = ThoughtId::new();
        assert_eq!(
            LinkTarget::Thought(tid).value_str(),
            tid.as_uuid().to_string()
        );
        assert_eq!(
            LinkTarget::Entity("acme corp".into()).value_str(),
            "acme corp"
        );
        assert_eq!(LinkTarget::Person("Ron".into()).value_str(), "Ron");
        assert_eq!(
            LinkTarget::Url("https://anthropic.com".into()).value_str(),
            "https://anthropic.com"
        );
    }

    #[test]
    fn link_target_accessors_return_some_only_for_matching_variant() {
        let tid = ThoughtId::new();
        let t = LinkTarget::Thought(tid);
        assert_eq!(t.as_thought_id(), Some(&tid));
        assert_eq!(t.as_entity(), None);
        assert_eq!(t.as_person(), None);
        assert_eq!(t.as_url(), None);

        let e = LinkTarget::Entity("foo".into());
        assert_eq!(e.as_thought_id(), None);
        assert_eq!(e.as_entity(), Some("foo"));
        assert_eq!(e.as_person(), None);
        assert_eq!(e.as_url(), None);
    }

    #[test]
    fn link_target_serializes_as_kind_value_tagged() {
        let tid = ThoughtId::new();
        let json = serde_json::to_value(LinkTarget::Thought(tid)).unwrap();
        assert_eq!(json["kind"], "thought");
        assert_eq!(json["value"], tid.as_uuid().to_string());

        let json = serde_json::to_value(LinkTarget::Url("https://x.io".into())).unwrap();
        assert_eq!(json["kind"], "url");
        assert_eq!(json["value"], "https://x.io");
    }

    #[test]
    fn link_target_deserializes_from_kind_value_tagged() {
        let tid = ThoughtId::new();
        let s = format!(r#"{{"kind": "thought", "value": "{}"}}"#, tid.as_uuid());
        let parsed: LinkTarget = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, LinkTarget::Thought(tid));

        let parsed: LinkTarget =
            serde_json::from_str(r#"{"kind": "entity", "value": "acme"}"#).unwrap();
        assert_eq!(parsed, LinkTarget::Entity("acme".into()));
    }
}

//! Relation types — the M5 thought-to-thought graph layer.
//!
//! Edges live in `thought_links` (migration 0007). M5 commits to a small
//! closed relation vocabulary (`replaces`, `requires`, `references`,
//! `belongs_to`, `decided_by`, `refines`) and to thought-to-thought edges
//! only — heterogeneous targets (to-entity, to-person, to-URL) and
//! tagger-extracted relations are M5.x.

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThoughtLink {
    pub id: LinkId,
    pub from_thought_id: ThoughtId,
    pub relation: RelationKind,
    pub to_thought_id: ThoughtId,
    pub source: LinkSource,
    pub note: Option<String>,
    pub created_at: OffsetDateTime,
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
}

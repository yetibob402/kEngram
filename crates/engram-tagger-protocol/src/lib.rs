//! Wire contract for the engram tagger sidecar pattern.
//!
//! Engram's HTTP-tagger client (`engram_extract::HttpTagger`) and any
//! sidecar implementing the contract (Rust or otherwise) agree on the
//! JSON shapes defined here. The serde derives ARE the spec; the
//! human-readable companion lives at `docs/tagger-sidecar-protocol.md`
//! for non-Rust implementers.
//!
//! # Wire shape
//!
//! ```text
//! POST {endpoint}/tag
//! Content-Type: application/json
//!
//! {
//!   "protocol_version": "1",
//!   "content": "<thought text>",
//!   "vocab": {                        // optional
//!     "topics":   ["...", "..."],
//!     "entities": ["...", "..."]
//!   }
//! }
//!
//! HTTP 200 OK
//! Content-Type: application/json
//!
//! {
//!   "tags": {
//!     "people":          ["..."],
//!     "entities":        ["..."],    // may be empty (5-field schema)
//!     "action_items":    ["..."],
//!     "topics":          ["..."],
//!     "dates_mentioned": ["..."],
//!     "kind":            "task"      // or null
//!   },
//!   "relations": [
//!     {
//!       "relation":  "supports",
//!       "to_kind":   "entity",
//!       "to_value":  "engram",
//!       "note":      "..."           // optional
//!     }
//!   ]
//! }
//! ```
//!
//! # Error semantics
//!
//! Sidecars signal failure via HTTP status:
//! - **5xx** + connection-level failures (timeout, refused) are treated
//!   as **transient** by engram's drainer — the job stays in the queue
//!   and is retried on the next tick.
//! - **4xx** + malformed response bodies are **non-transient** —
//!   engram logs and skips the thought without retry.
//!
//! See `engram_core::TaggerError::is_transient`.
//!
//! # Versioning
//!
//! Requests carry an explicit `protocol_version` (currently `"1"`).
//! Breaking changes bump the version; sidecars may advertise which
//! versions they speak via the response's optional `protocol_version`
//! field. Engram's client refuses to use a sidecar whose declared
//! version doesn't match what the client sends.

use serde::{Deserialize, Serialize};

pub use engram_core::{ExtractedRelation, ScopeVocab, Tags};

/// The wire protocol version. Bumped when the JSON shapes change in
/// a way that older sidecars can't honor. Sent on every request and
/// (optionally) echoed by the sidecar on every response so a mismatch
/// surfaces at the first call rather than as silent misbehavior.
pub const PROTOCOL_VERSION: &str = "1";

/// Request body for `POST {endpoint}/tag`. Sent by engram's HTTP
/// tagger client; received by the sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TagRequest {
    /// Protocol version this request is encoded against. Currently
    /// always [`PROTOCOL_VERSION`] (`"1"`). Sidecars should reject
    /// requests whose `protocol_version` they don't recognize with
    /// HTTP 400 + a clear error body.
    #[serde(default = "default_protocol_version")]
    pub protocol_version: String,

    /// The thought's content as captured. Sidecars run their tagger
    /// against this text and return [`TagResponse`].
    pub content: String,

    /// Optional controlled-vocabulary hint: the most-used topic and
    /// entity terms in the thought's scope. Sidecars may use this to
    /// nudge toward consistent term reuse, but should not treat it
    /// as a closed vocabulary — new terms are expected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab: Option<ScopeVocabHint>,
}

/// Serializable mirror of [`ScopeVocab`]. The original lives in
/// `engram-core` and is `PartialEq + Clone + Default` but not `Serialize`,
/// so we mirror it here for wire use. Convert via `From` either direction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScopeVocabHint {
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
}

impl From<&ScopeVocab> for ScopeVocabHint {
    fn from(v: &ScopeVocab) -> Self {
        Self {
            topics: v.topics.clone(),
            entities: v.entities.clone(),
        }
    }
}

impl From<ScopeVocabHint> for ScopeVocab {
    fn from(h: ScopeVocabHint) -> Self {
        Self {
            topics: h.topics,
            entities: h.entities,
        }
    }
}

/// Response body for `POST {endpoint}/tag`. Returned by the sidecar
/// on a successful tag; parsed by engram's HTTP tagger client into
/// `engram_core::TagOutput`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TagResponse {
    /// The protocol version the sidecar speaks. Optional on the wire
    /// (sidecars may omit it for backward-compat with version-1
    /// clients that don't check). When present, engram's client
    /// verifies it matches the request's version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<String>,

    /// Persisted metadata for the thought. An empty `Tags` (all fields
    /// empty/null) is a valid "nothing extractable" response, not a
    /// failure.
    #[serde(default)]
    pub tags: Tags,

    /// Tagger-extracted relations routed into `thought_links` by the
    /// drainer (source = "tagger"). Empty vec is the common case for
    /// non-LLM sidecars; LLM-shaped sidecars may populate this.
    #[serde(default)]
    pub relations: Vec<ExtractedRelation>,
}

fn default_protocol_version() -> String {
    PROTOCOL_VERSION.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::TagKind;

    #[test]
    fn protocol_version_constant_pins_v1() {
        // The literal "1" is part of the wire spec. If you bump this
        // you must also update docs/tagger-sidecar-protocol.md and
        // confirm all known sidecars advertise the new version.
        assert_eq!(PROTOCOL_VERSION, "1");
    }

    #[test]
    fn tag_request_serializes_with_protocol_version() {
        let req = TagRequest {
            protocol_version: PROTOCOL_VERSION.to_string(),
            content: "hello world".to_string(),
            vocab: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["protocol_version"], "1");
        assert_eq!(json["content"], "hello world");
        // vocab is None → omitted by skip_serializing_if
        assert!(json.get("vocab").is_none());
    }

    #[test]
    fn tag_request_default_protocol_version_on_deserialize() {
        // Sidecars in some languages may forget the protocol_version
        // field; the serde default fills it in. Pins back-compat.
        let raw = r#"{"content":"hi"}"#;
        let req: TagRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.protocol_version, "1");
        assert!(req.vocab.is_none());
    }

    #[test]
    fn tag_request_with_vocab_round_trips() {
        let req = TagRequest {
            protocol_version: PROTOCOL_VERSION.to_string(),
            content: "thought about Sarah".to_string(),
            vocab: Some(ScopeVocabHint {
                topics: vec!["memory-systems".to_string()],
                entities: vec!["engram".to_string()],
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: TagRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn tag_response_empty_round_trips_as_default() {
        // A sidecar that found nothing extractable can return an
        // empty {} body — Tags default fills the rest.
        let raw = r#"{}"#;
        let resp: TagResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp, TagResponse::default());
    }

    #[test]
    fn tag_response_full_shape_round_trips() {
        let resp = TagResponse {
            protocol_version: Some(PROTOCOL_VERSION.to_string()),
            tags: Tags {
                people: vec!["Sarah".to_string()],
                entities: vec![],
                action_items: vec!["ship the sidecar".to_string()],
                topics: vec!["memory-systems".to_string()],
                dates_mentioned: vec!["next Friday".to_string()],
                kind: Some(TagKind::Task),
            },
            relations: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: TagResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, resp);
    }

    #[test]
    fn scope_vocab_hint_converts_both_directions() {
        let original = ScopeVocab {
            topics: vec!["rust".to_string()],
            entities: vec!["pgvector".to_string()],
        };
        let hint: ScopeVocabHint = (&original).into();
        let round_tripped: ScopeVocab = hint.into();
        assert_eq!(round_tripped, original);
    }
}

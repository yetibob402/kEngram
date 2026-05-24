//! The `Tagger` trait — the seam between engram and whatever backend
//! turns a thought's content into a bag of metadata tags. Implementations
//! live in `engram-extract`.
//!
//! Mirrors the `Embedder` / `Reranker` trait shape and `is_transient()`
//! discipline. The tag drainer loop uses `is_transient()` to decide whether
//! to soft-fail (leave the job in `pending_tags` with attempts++) and retry
//! on the next tick, or to log and skip without retry.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{LinkTarget, RelationKind, ScopeVocab, Tags};

/// Result of a single `Tagger::tag()` call. Bundles two distinct outputs
/// that the same LLM call produces:
///
/// - `tags` — metadata classifying the thought (people, entities, topics,
///   kind, etc.). Persisted as JSONB on `thoughts.tags`.
/// - `relations` — LLM-extracted edges to non-thought targets (entity /
///   person / URL). Persisted into `thought_links` via the drainer's
///   `apply_tagger_relations` helper with `source = 'tagger'`. NOT
///   persisted back into the `tags` JSONB; `thought_links` is the
///   canonical store for the link graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TagOutput {
    pub tags: Tags,
    pub relations: Vec<ExtractedRelation>,
}

/// One LLM-extracted edge attached to a thought. The drainer inserts these
/// into `thought_links` with `source = 'tagger'`. v1 (M6.1) targets
/// non-thoughts only — thought-target tagger relations are deferred until
/// entity resolution lands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub relation: RelationKind,
    #[serde(flatten)]
    pub target: ExtractedTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Non-thought target shape emitted by the tagger. Mirrors `LinkTarget`'s
/// non-thought variants and matches the JSON shape `{to_kind, to_value}` on
/// the wire (flattened into `ExtractedRelation`). Converts losslessly into
/// the full `LinkTarget` for insertion via `engram_storage::insert_link`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "to_kind", content = "to_value", rename_all = "snake_case")]
pub enum ExtractedTarget {
    Entity(String),
    Person(String),
    Url(String),
}

impl ExtractedTarget {
    /// Convert into the polymorphic `LinkTarget` used by the storage layer.
    /// Lossless — `ExtractedTarget` is a strict subset of `LinkTarget`
    /// (omits the `Thought` variant).
    pub fn into_link_target(self) -> LinkTarget {
        match self {
            Self::Entity(name) => LinkTarget::Entity(name),
            Self::Person(name) => LinkTarget::Person(name),
            Self::Url(url) => LinkTarget::Url(url),
        }
    }

    /// Stable discriminator string (mirrors `LinkTarget::kind_str`).
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Entity(_) => "entity",
            Self::Person(_) => "person",
            Self::Url(_) => "url",
        }
    }
}

/// The public contract for tagger backends. Anyone implementing this
/// trait is pluggable — the engram-cli `build_tagger` match arm is the
/// registry of *known* implementations, not the contract itself. See
/// `docs/tagger-backends.md` for the recipe to add a new backend.
///
/// Stock implementations live in `engram-extract`:
/// - `OpenAICompatibleTagger` — calls vLLM / OpenRouter / Ollama via the
///   `/v1/chat/completions` endpoint with structured-output enforcement.
/// - `HttpTagger` — speaks engram's own JSON wire contract (see
///   `engram-tagger-protocol`) for sidecar implementations in any
///   language. Reference sidecar lives at
///   `crates/engram-tagger-deterministic/`.
/// - `FakeTagger` — deterministic in-memory tagger for tests.
///
/// External implementations are welcome and don't require forking engram;
/// they can either implement this trait directly (in-process, Rust) or
/// run as a sidecar speaking the `engram-tagger-protocol` wire shape (any
/// language, over HTTP).
#[async_trait]
pub trait Tagger: Send + Sync {
    /// Stable model identifier — written into `thoughts.tags_extractor_model`
    /// for provenance. Conventionally `"<vendor>/<model>"`, e.g.
    /// `"openai/gpt-4o-mini"` or `"vllm/qwen2.5-7b-instruct"`.
    fn model_id(&self) -> &str;

    /// Schema-version of *this* tagger's prompt/response contract. Bump
    /// when the JSON Schema or system prompt changes in a way that makes
    /// prior tags no longer comparable. The drainer writes this into
    /// `thoughts.tags_extractor_version`.
    fn version(&self) -> i32;

    /// Tag a single thought's content. `vocab`, when supplied, lists the
    /// established topic and entity terms most frequently used in the
    /// thought's scope; implementations should encourage the model to prefer
    /// those terms when they fit, and coin new ones only for genuinely
    /// unseen concepts. Passing `None` runs the tagger without any
    /// vocabulary guidance.
    ///
    /// Returns a [`TagOutput`] bundling the persisted `Tags` and the
    /// transient `Vec<ExtractedRelation>` that the drainer routes into
    /// `thought_links`. An empty TagOutput is a valid "no extractable
    /// metadata here" answer and is not a failure.
    async fn tag(
        &self,
        thought_content: &str,
        vocab: Option<&ScopeVocab>,
    ) -> Result<TagOutput, TaggerError>;
}

#[derive(Debug, thiserror::Error)]
pub enum TaggerError {
    #[error("tagger endpoint unreachable: {0}")]
    Unreachable(String),

    #[error("tagger timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("tagger returned malformed response: {0}")]
    MalformedResponse(String),

    #[error("tagger misconfigured: {0}")]
    Misconfigured(String),

    #[error("tagger backend error (status {status}): {body}")]
    Backend { status: u16, body: String },
}

impl TaggerError {
    /// True when the failure is something the next drainer tick might
    /// resolve on its own (network blip, timeout, transient 5xx). The
    /// drainer soft-fails per thought on transient errors and continues;
    /// on non-transient errors it logs and removes the job from the queue.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Unreachable(_)
                | Self::Timeout { .. }
                | Self::Backend {
                    status: 500..=599,
                    ..
                }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_relation_serde_round_trip() {
        let r = ExtractedRelation {
            relation: RelationKind::References,
            target: ExtractedTarget::Url("https://anthropic.com".into()),
            note: Some("explicit citation".into()),
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["relation"], "references");
        assert_eq!(json["to_kind"], "url");
        assert_eq!(json["to_value"], "https://anthropic.com");
        assert_eq!(json["note"], "explicit citation");

        let parsed: ExtractedRelation = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, r);
    }

    #[test]
    fn extracted_relation_note_optional() {
        let r = ExtractedRelation {
            relation: RelationKind::BelongsTo,
            target: ExtractedTarget::Entity("Probe 2".into()),
            note: None,
        };
        let json = serde_json::to_value(&r).unwrap();
        assert!(
            json.get("note").is_none(),
            "note should be omitted when None"
        );

        let parsed: ExtractedRelation = serde_json::from_str(
            r#"{"relation":"belongs_to","to_kind":"entity","to_value":"Probe 2"}"#,
        )
        .unwrap();
        assert_eq!(parsed.note, None);
    }

    #[test]
    fn extracted_target_into_link_target_preserves_kind_and_value() {
        assert_eq!(
            ExtractedTarget::Entity("foo".into()).into_link_target(),
            LinkTarget::Entity("foo".into())
        );
        assert_eq!(
            ExtractedTarget::Person("Ron".into()).into_link_target(),
            LinkTarget::Person("Ron".into())
        );
        assert_eq!(
            ExtractedTarget::Url("https://x.io".into()).into_link_target(),
            LinkTarget::Url("https://x.io".into())
        );
    }

    #[test]
    fn tag_output_default_is_empty() {
        let o = TagOutput::default();
        assert_eq!(o.tags, Tags::default());
        assert!(o.relations.is_empty());
    }

    #[test]
    fn unreachable_is_transient() {
        assert!(TaggerError::Unreachable("connection refused".into()).is_transient());
    }

    #[test]
    fn timeout_is_transient() {
        assert!(TaggerError::Timeout { seconds: 5 }.is_transient());
    }

    #[test]
    fn server_5xx_is_transient() {
        assert!(
            TaggerError::Backend {
                status: 503,
                body: "unavailable".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn client_4xx_is_not_transient() {
        assert!(
            !TaggerError::Backend {
                status: 400,
                body: "bad request".into(),
            }
            .is_transient()
        );
    }

    #[test]
    fn malformed_is_not_transient() {
        assert!(!TaggerError::MalformedResponse("nope".into()).is_transient());
    }

    #[test]
    fn misconfigured_is_not_transient() {
        assert!(!TaggerError::Misconfigured("missing api key".into()).is_transient());
    }
}

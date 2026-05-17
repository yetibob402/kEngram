//! `FakeTagger` — deterministic, in-memory `Tagger` for tests.
//!
//! Configurable via [`FakeBehavior`] (always-succeed vs always-fail-with-X)
//! and [`FakeTaggerOutput`] (what `Tags` to return when succeeding). Records
//! the most recent call's content as a [`RecordedTag`] so tests can assert
//! the drainer passed the correct thought content.
//!
//! Mirrors `engram-embed::FakeEmbedder` in shape.

use async_trait::async_trait;
use engram_core::{ScopeVocab, Tagger, TaggerError, Tags};
use std::sync::{Arc, Mutex};

/// Failure-mode selector. `Deterministic` succeeds and returns whatever
/// `FakeTaggerOutput` dictates; everything else always fails with the
/// corresponding `TaggerError` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeBehavior {
    /// Return `Tags` per the configured `FakeTaggerOutput`.
    Deterministic,
    /// Always fail with `TaggerError::Timeout`.
    Timeout,
    /// Always fail with `TaggerError::Unreachable`.
    Unreachable,
    /// Always fail with `TaggerError::Misconfigured`.
    Misconfigured,
}

/// Output strategy when the fake tagger is in [`FakeBehavior::Deterministic`]
/// mode. Mirrors `FakeScoring` in engram-embed: composable, predictable
/// responses for test fixtures.
#[derive(Debug, Clone)]
pub enum FakeTaggerOutput {
    /// Always return `Tags::default()` (every field empty / `None`).
    Empty,
    /// Always return the given `Tags`, regardless of input content.
    Canned(Tags),
    /// If the input content contains any of the substrings (case-sensitive),
    /// return the corresponding `Tags`. First match wins. Falls back to
    /// `Tags::default()` when no substring matches.
    Substring(Vec<(String, Tags)>),
}

/// One observed call to `tag()` — content and optional scope vocabulary the
/// drainer passed in. Tests inspect this to confirm the drainer wired both
/// the thought content and (when enabled) the controlled vocabulary through.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedTag {
    pub content: String,
    pub vocab: Option<ScopeVocab>,
}

#[derive(Debug, Clone)]
pub struct FakeTagger {
    model_id: String,
    version: i32,
    behavior: FakeBehavior,
    output: FakeTaggerOutput,
    /// Records the content of the most recent `tag()` call (regardless of
    /// success/failure). `Arc<Mutex<_>>` because `FakeTagger` is passed by
    /// shared reference through the drainer loop and tests need to inspect
    /// post-run state.
    last_call: Arc<Mutex<Option<RecordedTag>>>,
}

impl FakeTagger {
    /// New deterministic tagger with sensible defaults (model_id
    /// `"fake/tagger"`, version 1, output `FakeTaggerOutput::Empty`).
    pub fn new() -> Self {
        Self {
            model_id: "fake/tagger".to_string(),
            version: 1,
            behavior: FakeBehavior::Deterministic,
            output: FakeTaggerOutput::Empty,
            last_call: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_model(model_id: impl Into<String>, version: i32) -> Self {
        Self {
            model_id: model_id.into(),
            version,
            ..Self::new()
        }
    }

    /// Build a deterministic tagger that always returns the given `Tags`.
    pub fn with_canned(tags: Tags) -> Self {
        Self {
            output: FakeTaggerOutput::Canned(tags),
            ..Self::new()
        }
    }

    /// Build a deterministic tagger driven by substring matching against the
    /// thought content.
    pub fn with_substring(rules: Vec<(String, Tags)>) -> Self {
        Self {
            output: FakeTaggerOutput::Substring(rules),
            ..Self::new()
        }
    }

    /// Build a copy of this tagger that always fails with the given behavior.
    pub fn always_failing(behavior: FakeBehavior) -> Self {
        Self {
            behavior,
            ..Self::new()
        }
    }

    /// Returns the most recently captured `tag()` call, if any.
    pub fn last_call(&self) -> Option<RecordedTag> {
        self.last_call
            .lock()
            .expect("last_call mutex poisoned")
            .clone()
    }
}

impl Default for FakeTagger {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tagger for FakeTagger {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn version(&self) -> i32 {
        self.version
    }

    async fn tag(
        &self,
        thought_content: &str,
        vocab: Option<&ScopeVocab>,
    ) -> Result<Tags, TaggerError> {
        *self.last_call.lock().expect("last_call mutex poisoned") = Some(RecordedTag {
            content: thought_content.to_string(),
            vocab: vocab.cloned(),
        });
        match self.behavior {
            FakeBehavior::Timeout => Err(TaggerError::Timeout { seconds: 5 }),
            FakeBehavior::Unreachable => Err(TaggerError::Unreachable(
                "fake tagger configured to fail".into(),
            )),
            FakeBehavior::Misconfigured => Err(TaggerError::Misconfigured(
                "fake tagger configured to fail".into(),
            )),
            FakeBehavior::Deterministic => Ok(match &self.output {
                FakeTaggerOutput::Empty => Tags::default(),
                FakeTaggerOutput::Canned(t) => t.clone(),
                FakeTaggerOutput::Substring(rules) => rules
                    .iter()
                    .find(|(key, _)| thought_content.contains(key))
                    .map(|(_, tags)| tags.clone())
                    .unwrap_or_default(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::TagKind;

    fn sample_tags() -> Tags {
        Tags {
            people: vec!["Sarah".to_string()],
            entities: vec!["engram".to_string()],
            action_items: vec!["review the migration".to_string()],
            topics: vec!["rust".to_string()],
            dates_mentioned: vec!["next Thursday".to_string()],
            kind: Some(TagKind::Task),
        }
    }

    #[tokio::test]
    async fn fake_tagger_empty_returns_default_tags() {
        let t = FakeTagger::new();
        let tags = t.tag("anything", None).await.unwrap();
        assert_eq!(tags, Tags::default());
    }

    #[tokio::test]
    async fn fake_tagger_canned_returns_given_tags() {
        let t = FakeTagger::with_canned(sample_tags());
        let tags = t.tag("any content goes here", None).await.unwrap();
        assert_eq!(tags, sample_tags());
    }

    #[tokio::test]
    async fn fake_tagger_substring_matches_content() {
        let rules = vec![
            (
                "Sarah".to_string(),
                Tags {
                    people: vec!["Sarah".to_string()],
                    ..Tags::default()
                },
            ),
            (
                "rust".to_string(),
                Tags {
                    topics: vec!["rust".to_string()],
                    ..Tags::default()
                },
            ),
        ];
        let t = FakeTagger::with_substring(rules);

        let hit_sarah = t.tag("Met with Sarah yesterday", None).await.unwrap();
        assert_eq!(hit_sarah.people, vec!["Sarah".to_string()]);
        assert!(hit_sarah.topics.is_empty());

        let hit_rust = t.tag("learning rust ownership", None).await.unwrap();
        assert_eq!(hit_rust.topics, vec!["rust".to_string()]);
        assert!(hit_rust.people.is_empty());

        let no_match = t.tag("nothing here", None).await.unwrap();
        assert_eq!(no_match, Tags::default());
    }

    #[tokio::test]
    async fn fake_tagger_records_last_call() {
        let t = FakeTagger::new();
        assert!(t.last_call().is_none());
        let _ = t.tag("first call content", None).await.unwrap();
        assert_eq!(
            t.last_call(),
            Some(RecordedTag {
                content: "first call content".to_string(),
                vocab: None,
            })
        );
        let _ = t.tag("second call content", None).await.unwrap();
        assert_eq!(
            t.last_call(),
            Some(RecordedTag {
                content: "second call content".to_string(),
                vocab: None,
            })
        );
    }

    #[tokio::test]
    async fn fake_tagger_records_vocab_when_supplied() {
        let t = FakeTagger::new();
        let vocab = ScopeVocab {
            topics: vec!["rust".to_string()],
            entities: vec!["engram".to_string()],
        };
        let _ = t.tag("content", Some(&vocab)).await.unwrap();
        let rec = t.last_call().expect("call recorded");
        assert_eq!(rec.vocab, Some(vocab));
    }

    #[tokio::test]
    async fn fake_tagger_timeout_behavior_returns_timeout_error() {
        let t = FakeTagger::always_failing(FakeBehavior::Timeout);
        let err = t.tag("x", None).await.unwrap_err();
        assert!(matches!(err, TaggerError::Timeout { .. }));
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn fake_tagger_unreachable_behavior_returns_unreachable_error() {
        let t = FakeTagger::always_failing(FakeBehavior::Unreachable);
        let err = t.tag("x", None).await.unwrap_err();
        assert!(matches!(err, TaggerError::Unreachable(_)));
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn fake_tagger_misconfigured_behavior_returns_misconfigured_error() {
        let t = FakeTagger::always_failing(FakeBehavior::Misconfigured);
        let err = t.tag("x", None).await.unwrap_err();
        assert!(matches!(err, TaggerError::Misconfigured(_)));
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn fake_tagger_records_call_even_on_failure() {
        let t = FakeTagger::always_failing(FakeBehavior::Timeout);
        let _ = t.tag("captured on failure", None).await;
        assert_eq!(
            t.last_call(),
            Some(RecordedTag {
                content: "captured on failure".to_string(),
                vocab: None,
            })
        );
    }

    #[test]
    fn fake_tagger_model_id_and_version_are_stable() {
        let t = FakeTagger::with_model("custom/m", 7);
        assert_eq!(t.model_id(), "custom/m");
        assert_eq!(t.version(), 7);
    }
}

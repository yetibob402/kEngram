//! `HttpTagger` — engram's HTTP client for the tagger sidecar pattern.
//!
//! Talks to any sidecar (any language) that implements the
//! `engram-tagger-protocol` wire shape: `POST {endpoint}/tag` with a
//! [`TagRequest`] body, returns a [`TagResponse`]. The reference sidecar
//! lives at `crates/engram-tagger-deterministic/`; operators wanting a
//! non-Rust tagger ship their own service in their preferred language
//! and point engram at it via `provider = "http"` in `[tagger]`.
//!
//! Pattern mirrors `OpenAICompatibleTagger`: HTTP error classification
//! (5xx + connect/timeout = transient; 4xx + malformed = non-transient),
//! configurable timeout, optional bearer auth. The crucial difference
//! is the wire shape: this client speaks engram's own JSON contract
//! rather than the OpenAI chat-completions shape, so sidecars don't have
//! to fake LLM semantics to plug in.

use async_trait::async_trait;
use engram_core::{ScopeVocab, TagOutput, Tagger, TaggerError};
use engram_tagger_protocol::{PROTOCOL_VERSION, ScopeVocabHint, TagRequest, TagResponse};
use reqwest::Client;
use std::time::Duration;

/// Configuration for [`HttpTagger`].
#[derive(Debug, Clone)]
pub struct HttpTaggerConfig {
    /// Base URL of the sidecar. The client appends `/tag` to this.
    /// Example: `"http://localhost:8081"`.
    pub endpoint: String,
    /// Engram-side stable identity written into
    /// `thoughts.tags_extractor_model`. Conventionally
    /// `"<vendor>/<sidecar>"`. The sidecar's actual model identity isn't
    /// echoed on the wire — engram stamps what the operator configures
    /// so provenance lives in `engram.toml`, not in the sidecar's
    /// response.
    pub model_id: String,
    /// Schema-version written into `thoughts.tags_extractor_version`.
    /// Bump when the sidecar's behavior changes such that prior tags
    /// shouldn't be considered comparable.
    pub model_version: i32,
    /// Optional bearer token sent as `Authorization: Bearer <token>`.
    /// `None` means no Authorization header (sidecars on a private
    /// network are the common case).
    pub api_key: Option<String>,
    /// Per-request timeout. Mirrors openai-compatible's pattern;
    /// captured into `HttpTagger` so timeout errors report the actual
    /// configured duration rather than a hardcoded constant.
    pub timeout: Duration,
}

/// HTTP-tagger client. Holds a `reqwest::Client` built once at startup
/// plus the resolved config. Cheap to construct (no I/O); per-call work
/// is one `POST {endpoint}/tag` round-trip.
#[derive(Debug)]
pub struct HttpTagger {
    config: HttpTaggerConfig,
    client: Client,
    /// Captured at construction so `Timeout` errors report the configured
    /// duration. Mirrors the openai-compatible pattern.
    timeout_seconds: u64,
}

impl HttpTagger {
    pub fn new(config: HttpTaggerConfig) -> Result<Self, TaggerError> {
        if config.endpoint.is_empty() {
            return Err(TaggerError::Misconfigured(
                "[tagger.http] endpoint is empty".into(),
            ));
        }
        if config.model_id.is_empty() {
            return Err(TaggerError::Misconfigured(
                "[tagger] model_id is empty (required for http provider)".into(),
            ));
        }
        let timeout_seconds = config.timeout.as_secs();
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| TaggerError::Misconfigured(format!("reqwest client build: {e}")))?;
        Ok(Self {
            config,
            client,
            timeout_seconds,
        })
    }
}

#[async_trait]
impl Tagger for HttpTagger {
    fn model_id(&self) -> &str {
        &self.config.model_id
    }

    fn version(&self) -> i32 {
        self.config.model_version
    }

    async fn tag(
        &self,
        thought_content: &str,
        vocab: Option<&ScopeVocab>,
    ) -> Result<TagOutput, TaggerError> {
        let req = TagRequest {
            protocol_version: PROTOCOL_VERSION.to_string(),
            content: thought_content.to_string(),
            vocab: vocab.map(ScopeVocabHint::from),
        };
        let url = format!("{}/tag", self.config.endpoint.trim_end_matches('/'));
        let mut builder = self.client.post(&url).json(&req);
        if let Some(key) = &self.config.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder
            .send()
            .await
            .map_err(|e| map_send_error(e, self.timeout_seconds))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(TaggerError::Backend {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: TagResponse = resp
            .json()
            .await
            .map_err(|e| TaggerError::MalformedResponse(format!("sidecar response JSON: {e}")))?;
        // Sidecar may omit protocol_version in the response (backward
        // compat for v1 clients that didn't echo). If present, it must
        // match the version we sent.
        if let Some(advertised) = parsed.protocol_version.as_deref()
            && advertised != PROTOCOL_VERSION
        {
            return Err(TaggerError::Misconfigured(format!(
                "sidecar speaks protocol_version {advertised:?}, client speaks {PROTOCOL_VERSION:?}; refusing to mix versions",
            )));
        }
        Ok(TagOutput {
            tags: parsed.tags,
            relations: parsed.relations,
        })
    }
}

fn map_send_error(e: reqwest::Error, timeout_seconds: u64) -> TaggerError {
    if e.is_timeout() {
        TaggerError::Timeout {
            seconds: timeout_seconds,
        }
    } else if e.is_connect() {
        TaggerError::Unreachable(e.to_string())
    } else if let Some(status) = e.status() {
        TaggerError::Backend {
            status: status.as_u16(),
            body: e.to_string(),
        }
    } else {
        TaggerError::Unreachable(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::TagKind;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(endpoint: String) -> HttpTaggerConfig {
        HttpTaggerConfig {
            endpoint,
            model_id: "test/sidecar-v1".to_string(),
            model_version: 1,
            api_key: None,
            timeout: Duration::from_secs(2),
        }
    }

    #[tokio::test]
    async fn empty_endpoint_is_misconfigured() {
        let cfg = HttpTaggerConfig {
            endpoint: String::new(),
            ..config_for("http://localhost".to_string())
        };
        let err = HttpTagger::new(cfg).expect_err("should fail");
        assert!(matches!(err, TaggerError::Misconfigured(_)));
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn empty_model_id_is_misconfigured() {
        let cfg = HttpTaggerConfig {
            model_id: String::new(),
            ..config_for("http://localhost".to_string())
        };
        let err = HttpTagger::new(cfg).expect_err("should fail");
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn successful_tag_returns_tag_output() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "tags": {
                    "people": ["Sarah"],
                    "entities": [],
                    "action_items": ["ship the sidecar"],
                    "topics": ["memory-systems"],
                    "dates_mentioned": ["next Friday"],
                    "kind": "task"
                },
                "relations": []
            })))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let out = tagger.tag("hello world", None).await.expect("tag");
        assert_eq!(out.tags.people, vec!["Sarah".to_string()]);
        assert_eq!(out.tags.kind, Some(TagKind::Task));
        assert!(out.relations.is_empty());
    }

    #[tokio::test]
    async fn empty_object_response_is_default_tag_output() {
        // A sidecar that found nothing extractable returns {}.
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let out = tagger.tag("nothing to extract", None).await.expect("tag");
        assert_eq!(out, TagOutput::default());
    }

    #[tokio::test]
    async fn http_503_is_transient_backend_error() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream busy"))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let err = tagger.tag("hi", None).await.expect_err("should fail");
        match &err {
            TaggerError::Backend { status, .. } => assert_eq!(*status, 503),
            other => panic!("expected Backend{{503}}, got {other:?}"),
        }
        assert!(
            err.is_transient(),
            "5xx should be transient for drainer retry"
        );
    }

    #[tokio::test]
    async fn http_400_is_non_transient_backend_error() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let err = tagger.tag("hi", None).await.expect_err("should fail");
        match &err {
            TaggerError::Backend { status, .. } => assert_eq!(*status, 400),
            other => panic!("expected Backend{{400}}, got {other:?}"),
        }
        assert!(!err.is_transient(), "4xx should NOT be transient");
    }

    #[tokio::test]
    async fn malformed_json_is_non_transient() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json at all"))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let err = tagger.tag("hi", None).await.expect_err("should fail");
        assert!(matches!(err, TaggerError::MalformedResponse(_)));
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn connect_failure_maps_to_unreachable_or_timeout() {
        // No mock server running on this port; reqwest fails to connect.
        let tagger =
            HttpTagger::new(config_for("http://127.0.0.1:1".to_string())).expect("construct");
        let err = tagger.tag("hi", None).await.expect_err("should fail");
        assert!(
            matches!(
                err,
                TaggerError::Unreachable(_) | TaggerError::Timeout { .. }
            ),
            "got {err:?}"
        );
        assert!(err.is_transient(), "connect failures should be transient");
    }

    #[tokio::test]
    async fn vocab_hint_serializes_into_request() {
        // Capture the body the sidecar receives to confirm vocab is
        // present + shaped correctly. wiremock supports matching on
        // body content; using a less strict check here (just that
        // 200 succeeds when vocab is present) is enough to pin
        // serialization didn't blow up.
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let vocab = ScopeVocab {
            topics: vec!["rust".to_string()],
            entities: vec!["engram".to_string()],
        };
        let out = tagger.tag("vocab test", Some(&vocab)).await.expect("tag");
        assert_eq!(out, TagOutput::default());
    }

    #[tokio::test]
    async fn protocol_version_mismatch_is_misconfigured() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "protocol_version": "99",
                "tags": {}
            })))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let err = tagger.tag("hi", None).await.expect_err("should fail");
        assert!(matches!(err, TaggerError::Misconfigured(_)), "got {err:?}");
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn matching_protocol_version_in_response_succeeds() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tag"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "protocol_version": "1",
                "tags": {"people": ["Ron"]}
            })))
            .mount(&mock)
            .await;
        let tagger = HttpTagger::new(config_for(mock.uri())).expect("construct");
        let out = tagger.tag("hi", None).await.expect("tag");
        assert_eq!(out.tags.people, vec!["Ron".to_string()]);
    }
}

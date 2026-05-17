//! `OpenAICompatibleTagger` — talks to any backend that implements the
//! OpenAI `/v1/chat/completions` API with `response_format: json_schema`.
//! That covers vLLM (production), OpenRouter (cloud fallback), and OpenAI
//! itself, distinguished only by config.
//!
//! Endpoint convention: the configured `endpoint` is the `/v1` base, and
//! the tagger appends `/chat/completions`. For local vLLM that's
//! `http://localhost:8000/v1`.

use async_trait::async_trait;
use engram_core::{ScopeVocab, Tagger, TaggerError, Tags};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OpenAICompatibleConfig {
    /// Base URL ending in `/v1`.
    pub endpoint: String,
    /// Model name as the backend understands it. For vLLM: the deployed
    /// model (`"qwen2.5-7b-instruct"`). For OpenRouter: a model slug
    /// (`"anthropic/claude-haiku-4.5"`).
    pub model_name: String,
    /// Engram-side stable identity written into `thoughts.tags_extractor_model`.
    /// Conventionally `<vendor>/<model>` — `"vllm/qwen2.5-7b-instruct"`,
    /// `"openrouter/anthropic/claude-haiku-4.5"`.
    pub model_id: String,
    /// Schema-version of this tagger's prompt/response contract. Bump
    /// when the JSON Schema or system prompt changes such that prior tags
    /// are no longer comparable. Written into
    /// `thoughts.tags_extractor_version`.
    pub model_version: i32,
    pub api_key: Option<String>,
    pub timeout: Duration,
    /// Generation temperature. Lower = more deterministic tagging. 0.2 is
    /// a reasonable default; 0 makes some backends loop.
    pub temperature: f32,
    /// Override the bundled system prompt (`BUNDLED_TAGGER_PROMPT`). `None`
    /// means use the bundled default. `Some(_)` means the operator supplied
    /// a custom prompt — the operator is responsible for also bumping
    /// `model_version` so `thoughts.tags_extractor_version` remains
    /// meaningful provenance. A WARN is emitted at construction when this
    /// is `Some(_)`.
    pub system_prompt: Option<String>,
}

impl OpenAICompatibleConfig {
    /// Defaults for a local vLLM dev path on port 8000 with the qwen-7b
    /// instruct model. No API key.
    pub fn vllm_local() -> Self {
        Self {
            endpoint: "http://localhost:8000/v1".to_string(),
            model_name: "qwen2.5-7b-instruct".to_string(),
            model_id: "vllm/qwen2.5-7b-instruct".to_string(),
            model_version: BUNDLED_TAGGER_VERSION,
            api_key: None,
            timeout: Duration::from_secs(60),
            temperature: 0.2,
            system_prompt: None,
        }
    }

    /// Preset for OpenRouter cloud fallback. `model_name` is an OpenRouter
    /// model slug (e.g. `"anthropic/claude-haiku-4.5"`); the model_id is
    /// derived by prefixing with `"openrouter/"` so tags retain a clean
    /// provenance string.
    pub fn open_router(api_key: String, model_name: String) -> Self {
        Self {
            endpoint: "https://openrouter.ai/api/v1".to_string(),
            model_id: format!("openrouter/{model_name}"),
            model_name,
            model_version: BUNDLED_TAGGER_VERSION,
            api_key: Some(api_key),
            timeout: Duration::from_secs(60),
            temperature: 0.2,
            system_prompt: None,
        }
    }
}

/// Version of the bundled tagger prompt + response schema. Paired with the
/// model_version field on each thought row's tag provenance. Bump when the
/// prompt or schema changes such that prior tags shouldn't be considered
/// comparable. Operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z`
/// to backfill after a bump.
///
/// History: v1 was the initial M4 thoughts-only tagger; v2 (M4.1) split
/// `topics` into `entities` (proper-noun-style identifiers) + `topics`
/// (subject categories) and added the optional scope-vocabulary
/// controlled-vocabulary section.
pub const BUNDLED_TAGGER_VERSION: i32 = 2;

#[derive(Debug, Clone)]
pub struct OpenAICompatibleTagger {
    endpoint: String,
    model_name: String,
    model_id: String,
    model_version: i32,
    api_key: Option<String>,
    temperature: f32,
    /// Resolved system prompt — either the bundled default or the operator's
    /// override. Stored at construction so `tag()` doesn't re-resolve on
    /// every request.
    system_prompt: String,
    /// Stored alongside the client so the timeout-error path reports the
    /// actual configured value (the reqwest client owns the same duration
    /// internally but doesn't expose it).
    timeout_seconds: u64,
    client: Client,
}

impl OpenAICompatibleTagger {
    pub fn new(config: OpenAICompatibleConfig) -> Result<Self, TaggerError> {
        if config.endpoint.is_empty() {
            return Err(TaggerError::Misconfigured(
                "tagger endpoint must not be empty".into(),
            ));
        }
        if config.model_name.is_empty() {
            return Err(TaggerError::Misconfigured(
                "tagger model_name must not be empty".into(),
            ));
        }

        // Resolve the system prompt: operator override wins; otherwise the
        // bundled default.
        let (system_prompt, is_override) = match config.system_prompt {
            Some(custom) => (custom, true),
            None => (BUNDLED_TAGGER_PROMPT.to_string(), false),
        };
        if is_override {
            tracing::warn!(
                model_id = %config.model_id,
                model_version = config.model_version,
                "tagger: custom system_prompt in use; ensure model_version reflects this prompt's identity. \
                 Past tags with the same tagger_version were produced under the bundled prompt; \
                 tags produced under a custom prompt should bump model_version so provenance partitions cleanly."
            );
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| TaggerError::Unreachable(format!("client build: {e}")))?;
        Ok(Self {
            endpoint: config.endpoint,
            model_name: config.model_name,
            model_id: config.model_id,
            model_version: config.model_version,
            api_key: config.api_key,
            temperature: config.temperature,
            system_prompt,
            timeout_seconds: config.timeout.as_secs(),
            client,
        })
    }
}

/// The bundled tagger system prompt. Exposed `pub const` so operators can
/// inspect it (`engram-cli` can print it; configuration can compare against
/// it) and so a custom prompt loaded from `system_prompt_file` can be diffed
/// against the bundled one at startup.
///
/// The prompt is **paired** with `OpenAICompatibleConfig::model_version`
/// (default 1 when the bundled prompt is in use). Bump the version whenever
/// this prompt or the response schema changes such that prior tags
/// shouldn't be considered comparable; `engram tag --rerun` then re-tags
/// under the new version. If you override this via
/// `OpenAICompatibleConfig::system_prompt`, you are responsible for also
/// bumping the version — see `docs/engram-design-v0.md` §6 / §10.
pub const BUNDLED_TAGGER_PROMPT: &str = "\
You are a tagging assistant. Given a single thought from a memory service, return its metadata tags as JSON.

# Output shape
{ \"people\": [...], \"entities\": [...], \"action_items\": [...], \"topics\": [...], \"dates_mentioned\": [...], \"kind\": \"...\" }

# Field semantics
- people: bare names of people mentioned. Empty array if none.
- entities: named, proper-noun-style identifiers explicitly mentioned in the thought — projects, products, libraries, tools, technologies, named concepts (e.g., \"engram\", \"pgvector\", \"vLLM\", \"PostgreSQL\", \"MCP\"). Preserve the casing the thought uses, or the canonical casing if the thought is inconsistent. Empty array if none.
- action_items: short imperative phrases describing tasks the thought commits to or implies (e.g., \"fix the login bug\", \"review the migration plan\"). Empty array if none.
- topics: 1-3 short tag-like subject categories, lowercase, hyphen-separated, no punctuation. What broad SUBJECT AREA is this thought about? Examples: \"rust\", \"build-systems\", \"team-management\", \"memory-systems\". Distinct from entities: a topic is a category the thought falls under; an entity is a specific named thing the thought mentions. A thought naming \"engram\" and \"pgvector\" might have entities [\"engram\", \"pgvector\"] and topics [\"memory-systems\", \"databases\"].
- dates_mentioned: any dates or temporal references appearing in the prose (\"next Thursday\", \"Q3\", \"2026-05-15\", \"before the release\"). Free-form strings, copied roughly as they appear. Empty array if none.
- kind: a single classification. Use null if uncertain. Categories:
  - observation: a factual claim about the world (\"Rust has stronger memory safety than C\").
  - task: a thing the writer or someone else needs to do (\"fix the login bug\").
  - idea: a proposal or hypothesis (\"we could use Bloom filters here\").
  - reference: a pointer to an external resource (a URL, a paper, a tool).
  - person_note: a fact about a specific person (\"Sarah prefers async meetings\").
  - session: transient session/test narrative (\"the search returned 3 results\", \"I just ran the migration\"). These should also have otherwise-empty arrays.

# Rules
- Entities require explicit mention by name in the thought. Do not invent entities.
- Topics may be inferred from prose context when the subject is clear, even if the exact topic word doesn't appear.
- Empty arrays are correct for any field that has no content.
- One classification only; pick the most-load-bearing category. If genuinely ambiguous, return null.
- This is a tagging pass, not a paraphrase or rewrite. Do not rephrase the thought's content; only emit metadata.";

/// Render the optional controlled-vocabulary section appended to the system
/// prompt when scope vocabulary is available. Returns an empty string when
/// the vocab is `None` or completely empty, so callers can unconditionally
/// concatenate the result.
fn render_vocab_section(vocab: Option<&ScopeVocab>) -> String {
    let Some(v) = vocab else {
        return String::new();
    };
    if v.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\n# Controlled vocabulary (this scope's established terms)\n");
    if !v.topics.is_empty() {
        out.push_str("Topics already used in this scope: ");
        out.push_str(&v.topics.join(", "));
        out.push_str(".\n");
    }
    if !v.entities.is_empty() {
        out.push_str("Entities already used in this scope: ");
        out.push_str(&v.entities.join(", "));
        out.push_str(".\n");
    }
    out.push_str(
        "When a concept in the thought matches one of these established terms, prefer the established form. Coin a new term only when the prose introduces something genuinely unseen.",
    );
    out
}

#[derive(Serialize)]
struct ChatRequestBody<'a> {
    model: &'a str,
    temperature: f32,
    messages: Vec<ChatMessage<'a>>,
    response_format: serde_json::Value,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponseBody {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[async_trait]
impl Tagger for OpenAICompatibleTagger {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn version(&self) -> i32 {
        self.model_version
    }

    async fn tag(
        &self,
        thought_content: &str,
        vocab: Option<&ScopeVocab>,
    ) -> Result<Tags, TaggerError> {
        let url = format!("{}/chat/completions", self.endpoint.trim_end_matches('/'));

        let system_content = {
            let vocab_section = render_vocab_section(vocab);
            if vocab_section.is_empty() {
                self.system_prompt.clone()
            } else {
                let mut s = self.system_prompt.clone();
                s.push_str(&vocab_section);
                s
            }
        };
        let messages: Vec<ChatMessage<'_>> = vec![
            ChatMessage {
                role: "system",
                content: system_content,
            },
            ChatMessage {
                role: "user",
                content: thought_content.to_string(),
            },
        ];
        let body = ChatRequestBody {
            model: &self.model_name,
            temperature: self.temperature,
            messages,
            response_format: tags_response_format(),
        };

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
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

        let parsed: ChatResponseBody = resp.json().await.map_err(|e| {
            TaggerError::MalformedResponse(format!("decoding chat completions response: {e}"))
        })?;

        let content = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| TaggerError::MalformedResponse("response had zero choices".into()))?
            .message
            .content;

        let tags: Tags = serde_json::from_str(&content).map_err(|e| {
            TaggerError::MalformedResponse(format!(
                "decoding tags payload (content={content:?}): {e}"
            ))
        })?;

        Ok(tags)
    }
}

/// The `response_format` JSON object sent to the chat completions API. The
/// schema constrains the model to the `Tags` wire shape with six required
/// fields; `topics` is capped at 3 items, `entities` at 5, and `kind` is
/// nullable with an enum of `TagKind` snake_case variants.
fn tags_response_format() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "engram_tags",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "required": ["people", "entities", "action_items", "topics", "dates_mentioned", "kind"],
                "properties": {
                    "people": { "type": "array", "items": { "type": "string" } },
                    "entities": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
                    "action_items": { "type": "array", "items": { "type": "string" } },
                    "topics": { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
                    "dates_mentioned": { "type": "array", "items": { "type": "string" } },
                    "kind": {
                        "type": ["string", "null"],
                        "enum": ["observation", "task", "idea", "reference", "person_note", "session", null]
                    }
                }
            }
        }
    })
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
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(endpoint: String, api_key: Option<String>) -> OpenAICompatibleConfig {
        OpenAICompatibleConfig {
            endpoint,
            model_name: "test-model".to_string(),
            model_id: "test/test-model".to_string(),
            model_version: 1,
            api_key,
            timeout: Duration::from_secs(2),
            temperature: 0.0,
            system_prompt: None,
        }
    }

    fn chat_response_with_tags(tags: serde_json::Value) -> serde_json::Value {
        let content = serde_json::to_string(&tags).unwrap();
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        })
    }

    #[tokio::test]
    async fn valid_response_parses_to_tags() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_with_tags(json!({
                    "people": ["Sarah", "Ron"],
                    "entities": ["engram", "pgvector"],
                    "action_items": ["fix the login bug"],
                    "topics": ["rust", "build-systems"],
                    "dates_mentioned": ["next Thursday"],
                    "kind": "task"
                }))),
            )
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let tags = t.tag("anything", None).await.unwrap();
        assert_eq!(tags.people, vec!["Sarah".to_string(), "Ron".to_string()]);
        assert_eq!(
            tags.entities,
            vec!["engram".to_string(), "pgvector".to_string()]
        );
        assert_eq!(tags.action_items, vec!["fix the login bug".to_string()]);
        assert_eq!(
            tags.topics,
            vec!["rust".to_string(), "build-systems".to_string()]
        );
        assert_eq!(tags.dates_mentioned, vec!["next Thursday".to_string()]);
        assert_eq!(tags.kind, Some(TagKind::Task));
    }

    #[tokio::test]
    async fn malformed_response_returns_malformed_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"role": "assistant", "content": "not json"}}]
            })))
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let err = t.tag("x", None).await.unwrap_err();
        assert!(matches!(err, TaggerError::MalformedResponse(_)));
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn timeout_returns_transient_error() {
        let server = MockServer::start().await;
        // Delay > configured timeout (2s) — reqwest will time out first.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response_with_tags(json!({
                        "people": [], "action_items": [], "topics": [],
                        "dates_mentioned": [], "kind": null
                    })))
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let err = t.tag("x", None).await.unwrap_err();
        assert!(
            matches!(err, TaggerError::Timeout { .. }),
            "expected Timeout, got {err:?}"
        );
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn http_500_returns_backend_transient() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream gone"))
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let err = t.tag("x", None).await.unwrap_err();
        match err {
            TaggerError::Backend { status, .. } => assert_eq!(status, 503),
            other => panic!("expected Backend error, got {other:?}"),
        }
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn http_400_returns_backend_non_transient() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let err = t.tag("x", None).await.unwrap_err();
        match &err {
            TaggerError::Backend { status, .. } => assert_eq!(*status, 400),
            other => panic!("expected Backend error, got {other:?}"),
        }
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn connect_failure_maps_to_unreachable_or_timeout() {
        // Port 1 is reliably refused on macOS/Linux.
        let t = OpenAICompatibleTagger::new(config_for("http://127.0.0.1:1/v1".to_string(), None))
            .unwrap();
        let err = t.tag("x", None).await.unwrap_err();
        assert!(
            matches!(
                err,
                TaggerError::Unreachable(_) | TaggerError::Timeout { .. }
            ),
            "expected Unreachable or Timeout, got {err:?}"
        );
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn request_uses_bearer_auth_when_api_key_present() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_with_tags(json!({
                    "people": [], "action_items": [], "topics": [],
                    "dates_mentioned": [], "kind": null
                }))),
            )
            .mount(&server)
            .await;

        let t = OpenAICompatibleTagger::new(config_for(
            format!("{}/v1", server.uri()),
            Some("sk-test".into()),
        ))
        .unwrap();
        // If the auth header is wrong, wiremock returns 404 and the parse fails.
        t.tag("x", None).await.expect("auth header must match");
    }

    #[tokio::test]
    async fn empty_endpoint_is_misconfigured() {
        let mut cfg = config_for("".to_string(), None);
        cfg.endpoint = "".into();
        let err = OpenAICompatibleTagger::new(cfg).unwrap_err();
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn empty_model_name_is_misconfigured() {
        let mut cfg = config_for("http://127.0.0.1:1/v1".to_string(), None);
        cfg.model_name = "".into();
        let err = OpenAICompatibleTagger::new(cfg).unwrap_err();
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn custom_system_prompt_flows_into_request_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_with_tags(json!({
                    "people": [], "action_items": [], "topics": [],
                    "dates_mentioned": [], "kind": null
                }))),
            )
            .mount(&server)
            .await;

        let mut cfg = config_for(format!("{}/v1", server.uri()), None);
        cfg.system_prompt =
            Some("Custom prompt for the dogfood week. Return tags only.".to_string());
        let t = OpenAICompatibleTagger::new(cfg).unwrap();
        let _ = t.tag("x", None).await;

        let received = server.received_requests().await.unwrap();
        let last = received.last().expect("at least one request");
        let body: serde_json::Value = serde_json::from_slice(&last.body).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.contains("Custom prompt for the dogfood week"));
        // Bundled-prompt language must NOT leak in.
        assert!(!sys.contains("Field semantics"));
    }

    /// v2 prompt content pin: the tagger prompt must mention field semantics
    /// and list each of the six fields, including the new `entities` field
    /// added in M4.1. Catches accidental deletions during downstream edits.
    #[test]
    fn tagger_v2_prompt_contains_field_semantics_and_entities() {
        let p = BUNDLED_TAGGER_PROMPT;
        assert!(
            p.contains("Field semantics"),
            "v2 prompt must contain a 'Field semantics' section"
        );
        for field in [
            "people",
            "entities",
            "action_items",
            "topics",
            "dates_mentioned",
            "kind",
        ] {
            assert!(p.contains(field), "v2 prompt must mention field {field}");
        }
        // The entities/topics distinction must be explicit in the prompt so
        // the model can disambiguate the two open-vocabulary slots.
        assert!(
            p.contains("Distinct from entities"),
            "v2 prompt must explicitly distinguish entities from topics"
        );
        // Presets pinned to the bundled version (2 as of M4.1).
        assert_eq!(BUNDLED_TAGGER_VERSION, 2);
        let cfg = OpenAICompatibleConfig::vllm_local();
        assert_eq!(cfg.model_version, BUNDLED_TAGGER_VERSION);
        let cfg = OpenAICompatibleConfig::open_router("k".into(), "m".into());
        assert_eq!(cfg.model_version, BUNDLED_TAGGER_VERSION);
    }

    #[test]
    fn tags_response_format_pins_v2_shape() {
        let v = tags_response_format();
        let schema = &v["json_schema"]["schema"];
        let required = schema["required"].as_array().unwrap();
        let required: Vec<&str> = required.iter().map(|x| x.as_str().unwrap()).collect();
        assert_eq!(
            required,
            vec![
                "people",
                "entities",
                "action_items",
                "topics",
                "dates_mentioned",
                "kind"
            ]
        );
        assert_eq!(schema["properties"]["topics"]["maxItems"], 3);
        assert_eq!(schema["properties"]["entities"]["maxItems"], 5);
        // `kind` must allow null on the wire.
        let kind_type = &schema["properties"]["kind"]["type"];
        assert!(
            kind_type.as_array().unwrap().iter().any(|x| x == "null"),
            "kind must be nullable: {kind_type:?}"
        );
    }

    #[test]
    fn render_vocab_section_handles_none_and_empty() {
        assert_eq!(render_vocab_section(None), "");
        assert_eq!(render_vocab_section(Some(&ScopeVocab::default())), "");
    }

    #[test]
    fn render_vocab_section_lists_topics_and_entities() {
        let v = ScopeVocab {
            topics: vec!["rust".into(), "memory-systems".into()],
            entities: vec!["engram".into(), "pgvector".into()],
        };
        let rendered = render_vocab_section(Some(&v));
        assert!(rendered.contains("Controlled vocabulary"));
        assert!(rendered.contains("rust, memory-systems"));
        assert!(rendered.contains("engram, pgvector"));
        assert!(rendered.contains("prefer the established form"));
    }

    #[test]
    fn render_vocab_section_omits_empty_arm() {
        let topics_only = ScopeVocab {
            topics: vec!["rust".into()],
            entities: vec![],
        };
        let rendered = render_vocab_section(Some(&topics_only));
        assert!(rendered.contains("Topics already used"));
        assert!(!rendered.contains("Entities already used"));
    }

    #[tokio::test]
    async fn vocab_section_flows_into_request_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_with_tags(json!({
                    "people": [], "entities": [], "action_items": [], "topics": [],
                    "dates_mentioned": [], "kind": null
                }))),
            )
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let vocab = ScopeVocab {
            topics: vec!["memory-systems".into()],
            entities: vec!["engram".into()],
        };
        let _ = t.tag("any thought", Some(&vocab)).await;

        let received = server.received_requests().await.unwrap();
        let last = received.last().expect("at least one request");
        let body: serde_json::Value = serde_json::from_slice(&last.body).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(
            sys.contains("Controlled vocabulary"),
            "vocab section must be present in system message"
        );
        assert!(sys.contains("memory-systems"));
        assert!(sys.contains("engram"));
    }

    #[tokio::test]
    async fn no_vocab_omits_section_from_request_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_with_tags(json!({
                    "people": [], "entities": [], "action_items": [], "topics": [],
                    "dates_mentioned": [], "kind": null
                }))),
            )
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let _ = t.tag("any thought", None).await;

        let received = server.received_requests().await.unwrap();
        let last = received.last().expect("at least one request");
        let body: serde_json::Value = serde_json::from_slice(&last.body).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(
            !sys.contains("Controlled vocabulary"),
            "vocab section must be absent when vocab is None"
        );
    }

    /// Live test against a real OpenAI-compatible endpoint (vLLM by default).
    /// Gated on the `integration` feature; off in CI. Run with
    /// `cargo test -p engram-extract --features integration -- live_vllm`.
    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn live_vllm_round_trip() {
        let cfg = OpenAICompatibleConfig::vllm_local();
        let t = OpenAICompatibleTagger::new(cfg).unwrap();
        let tags = t
            .tag(
                "Engram uses pgvector for vector storage. Sarah will review the migration plan.",
                None,
            )
            .await
            .expect("vLLM unreachable — is it running on :8000?");
        // We can't assert specific tags (model output varies) but the call
        // must succeed and parse.
        let _ = tags;
    }
}

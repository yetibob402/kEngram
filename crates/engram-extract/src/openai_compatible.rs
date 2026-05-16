//! `OpenAICompatibleExtractor` — talks to any backend that implements the
//! OpenAI `/v1/chat/completions` API with `response_format: json_schema`.
//! That covers vLLM (production), OpenRouter (cloud fallback), and OpenAI
//! itself, distinguished only by config.
//!
//! Endpoint convention: the configured `endpoint` is the `/v1` base, and
//! the extractor appends `/chat/completions`. For local vLLM that's
//! `http://localhost:8000/v1`.

use async_trait::async_trait;
use engram_core::{ExtractMode, ExtractedFact, ExtractionContext, Extractor, ExtractorError, Thought};
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
    /// Engram-side stable identity written into `facts.extractor_model`.
    /// Conventionally `<vendor>/<model>` — `"vllm/qwen2.5-7b-instruct"`,
    /// `"openrouter/anthropic/claude-haiku-4.5"`.
    pub model_id: String,
    /// Schema-version of this extractor's prompt/response contract. Bump
    /// when the JSON Schema or system prompt changes such that prior facts
    /// are no longer comparable. Written into `facts.extractor_version`.
    pub model_version: i32,
    pub api_key: Option<String>,
    pub timeout: Duration,
    /// Generation temperature. Lower = more deterministic extraction. 0.2
    /// is a reasonable default; 0 makes some backends loop.
    pub temperature: f32,
    /// Soft cap on facts per thought. The reflector context's `max_facts`
    /// wins if it's smaller (so per-run policy can throttle independently).
    pub max_facts_per_thought: usize,
    /// Override the bundled system prompt (`BUNDLED_SYSTEM_PROMPT`). `None`
    /// means use the bundled default. `Some(_)` means the operator supplied
    /// a custom prompt — must contain the `{MAX_FACTS}` placeholder, and
    /// the operator is responsible for also bumping `model_version` so
    /// `facts.extractor_version` remains meaningful provenance. A WARN is
    /// emitted at construction when this is `Some(_)`.
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
            // v3 = SPO decomposition rules (comparative S/O mapping,
            // self-referential rejection, conditional-as-subject), tighter
            // confidence rubric, two new episodic-skip negatives, JSON
            // envelope restated in prose (2026-05-14, M3 Phase A). v4
            // (2026-05-15, M3 Phase C) folds in: structured-relations
            // rule for "A prefers X over Y"; reinforced few-shots for
            // the Phase A + Phase B step 2 leak set (Bazel/Make, Nix/Make
            // comparative, Redis conditional, SIMD self-reference);
            // 0.70–0.85 framed as the "stored but flagged" middle band;
            // explicit "no same-statement-different-SPO" rule. Earlier
            // facts carry version=1, 2, or 3 and can be re-extracted via
            // `engram reflect --rerun`.
            model_version: 4,
            api_key: None,
            timeout: Duration::from_secs(60),
            temperature: 0.2,
            max_facts_per_thought: 8,
            system_prompt: None,
        }
    }

    /// Preset for OpenRouter cloud fallback. `model_name` is an OpenRouter
    /// model slug (e.g. `"anthropic/claude-haiku-4.5"`); the model_id is
    /// derived by prefixing with `"openrouter/"` so facts retain a clean
    /// provenance string.
    pub fn open_router(api_key: String, model_name: String) -> Self {
        Self {
            endpoint: "https://openrouter.ai/api/v1".to_string(),
            model_id: format!("openrouter/{model_name}"),
            model_name,
            model_version: 4, // see `vllm_local()` for rationale on the bump
            api_key: Some(api_key),
            timeout: Duration::from_secs(60),
            temperature: 0.2,
            max_facts_per_thought: 8,
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAICompatibleExtractor {
    endpoint: String,
    model_name: String,
    model_id: String,
    model_version: i32,
    api_key: Option<String>,
    temperature: f32,
    max_facts_per_thought: usize,
    /// Resolved system prompt — either the bundled default or the
    /// operator's override. Stored at construction so `extract()` doesn't
    /// re-resolve on every request. The `{MAX_FACTS}` placeholder inside
    /// is substituted per-call.
    system_prompt: String,
    /// Stored alongside the client so the timeout-error path reports the
    /// actual configured value (the reqwest client owns the same duration
    /// internally but doesn't expose it).
    timeout_seconds: u64,
    client: Client,
}

impl OpenAICompatibleExtractor {
    pub fn new(config: OpenAICompatibleConfig) -> Result<Self, ExtractorError> {
        if config.endpoint.is_empty() {
            return Err(ExtractorError::Misconfigured(
                "extractor endpoint must not be empty".into(),
            ));
        }
        if config.model_name.is_empty() {
            return Err(ExtractorError::Misconfigured(
                "extractor model_name must not be empty".into(),
            ));
        }
        if config.max_facts_per_thought == 0 {
            return Err(ExtractorError::Misconfigured(
                "max_facts_per_thought must be > 0".into(),
            ));
        }

        // Resolve the system prompt: operator override wins; otherwise the
        // bundled default. Any override must keep the {MAX_FACTS} placeholder
        // or per-call substitution silently no-ops, leaving the model with
        // an unanchored prompt.
        let (system_prompt, is_override) = match config.system_prompt {
            Some(custom) => {
                if !custom.contains("{MAX_FACTS}") {
                    return Err(ExtractorError::Misconfigured(
                        "custom system_prompt must contain the {MAX_FACTS} placeholder".into(),
                    ));
                }
                (custom, true)
            }
            None => (BUNDLED_SYSTEM_PROMPT.to_string(), false),
        };
        if is_override {
            tracing::warn!(
                model_id = %config.model_id,
                model_version = config.model_version,
                "extractor: custom system_prompt in use; ensure model_version reflects this prompt's identity. \
                 Past facts with the same extractor_version were produced under the bundled prompt; \
                 facts produced under a custom prompt should bump model_version so provenance partitions cleanly."
            );
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| ExtractorError::Unreachable(format!("client build: {e}")))?;
        Ok(Self {
            endpoint: config.endpoint,
            model_name: config.model_name,
            model_id: config.model_id,
            model_version: config.model_version,
            api_key: config.api_key,
            temperature: config.temperature,
            max_facts_per_thought: config.max_facts_per_thought,
            system_prompt,
            timeout_seconds: config.timeout.as_secs(),
            client,
        })
    }
}

/// The bundled extractor system prompt. Exposed `pub const` so operators
/// can inspect it (`engram-cli` can print it; configuration can compare
/// against it) and so a custom prompt loaded from `system_prompt_file`
/// can be diffed against the bundled one at startup.
///
/// The prompt is **paired** with `OpenAICompatibleConfig::model_version`
/// (default 2 when the bundled prompt is in use). Bump the version
/// whenever this prompt or the response schema changes such that prior
/// facts shouldn't be considered comparable; `engram reflect --rerun`
/// then re-extracts under the new version. If you override this via
/// `OpenAICompatibleConfig::system_prompt`, you are responsible for
/// also bumping the version — see `docs/engram-design-v0.md` §6.5 / §10.
///
/// The `{MAX_FACTS}` placeholder is required; the extractor substitutes
/// it per request from `ctx.max_facts.min(config.max_facts_per_thought)`.
/// Custom prompts that omit the placeholder are rejected at construction
/// time with `ExtractorError::Misconfigured`.
pub const BUNDLED_SYSTEM_PROMPT: &str = "\
You are an information-extraction assistant. Given a single thought from a memory service, identify discrete factual claims and return them as structured JSON.

# Output shape

You MUST return a JSON object of the form:
  { \"facts\": [ { \"statement\": …, \"subject\": …, \"predicate\": …, \"object\": …, \"confidence\": … }, … ] }
The wrapper key is always \"facts\" and the value is always an array (possibly empty). Each item describes one fact.

Each fact has:
- statement: a self-contained natural-language sentence a reader could understand on its own.
- subject, predicate, object: optional (S, P, O) triple if the fact maps cleanly to one. Use null when no clean triple exists.
- confidence: your self-reported [0.0, 1.0] calibrated score (see rubric below).

# (S, P, O) decomposition rules

Subject and object must be *distinct* entities. If you cannot identify two distinct entities, set subject and/or object to null and rely on the statement to carry the claim.

Both subject and object SHOULD appear (as substrings, case-insensitive) inside the statement. If your decomposition leaves either out, you have the SPO wrong — revisit it.

For comparatives (\"A is more X than B\" / \"A is simpler than B\" / \"A outperforms B\"): the subject is A and the object is B. Subject must appear before object in the statement.
  DO emit: \"Bazel is more powerful than Make\" → subject=\"Bazel\", predicate=\"is more powerful than\", object=\"Make\".
  DO NOT emit: subject=\"Make\", object=\"Bazel\". The second clause of a compound sentence (\"…but has a steeper learning curve\") does NOT make Make the subject — it is still a comparison of Bazel against Make.
  DO emit: \"Nix is more reproducible than Make\" → subject=\"Nix\", object=\"Make\".
  DO NOT emit: subject=\"Make\", object=\"Nix\".
  DO emit: \"SIMD-accelerated JSON parsing outperforms scalar implementations\" → subject=\"SIMD-accelerated JSON parsing\", object=\"scalar implementations\".
  DO NOT emit: subject=object=\"SIMD-accelerated JSON parsing\" (self-referential).

For relative / preference claims naming two entities (\"A prefers X over Y\" / \"A favors X to Y\" / \"A picks X over Y\"): emit ONE fact whose object names both sides of the comparison. The comparison is one claim, not two.
  DO emit: \"Ron prefers Rust over Go for software projects\" → subject=\"Ron\", predicate=\"prefers over\", object=\"Rust to Go\".
  DO NOT emit two facts (\"Ron prefers Rust\" + \"Go is the next choice when Rust isn't available\") — flatten into a single ranking claim.

For conditionals (\"If <condition>, A is the right choice for X\"): the subject is A; the condition belongs in the statement context, not in the subject slot.
  DO emit: \"If you need sub-millisecond reads at high throughput, Redis is usually the right choice\" → subject=\"Redis\", predicate=\"is the right choice when\".
  DO NOT emit: subject=\"sub-millisecond reads at high throughput\" (the conditional clause).

# Confidence calibration

Pick the band the source actually supports — DO NOT default to a single value. Vary across hedged vs declarative claims.
- 0.95–1.00: direct quotation or near-verbatim restatement. The source unambiguously asserts this exact claim, in roughly these words.
- 0.90–0.95: clean paraphrase of a declarative statement, no hedging in the source, no added inference.
- 0.70–0.90: declarative claim that required moderate paraphrase, or a claim qualified by mild hedges ('usually', 'most', 'tends to'). This is the \"stored but flagged\" middle band — useful claims that are not as load-bearing as direct quotations. Use it intentionally rather than as a default.
- 0.50–0.70: claim is strongly hedged in the source ('likely', 'might', 'I suspect', 'often but not always'), or requires inference from surrounding context.
- below 0.50: speculative; a human should review.

Declarative paraphrase-of-source belongs at 0.90+, not in the middle band.

# Rules

- Do not invent facts that aren't supported by the input. If the source is uncertain, the fact's confidence must reflect that uncertainty.
- Skip purely conversational, social, or temporal-greeting content — return an empty facts array.
- Skip episodic / transient content: descriptions of one-off operations ('a search was conducted', 'the test returned X', 'today I ran Y'), individual test runs, or snapshots of system state at a particular moment. Extract durable claims about how the system or domain works, not what happened during a session.
- Skip single-benchmark measurements as if they were durable properties. \"simd-json was 3.2x faster than serde_json for parsing 100MB of test JSON\" is a one-run observation; the durable claim is \"SIMD parsing tends to outperform scalar JSON parsing on documents over ~1MB.\" Extract the latter, not the specific 3.2x measurement.
- Skip hardware-spec / session-metadata phrasings ('The benchmark was conducted on this hardware (M2 Pro, 16GB)'). These are observational context, not durable claims.
- Useful sanity check: would this claim still be useful to a reader six months from now, independent of when it was captured? If no, skip.
- For mixed-content thoughts (a thought containing both a durable finding and a transient session-narrative): extract only the finding.
- One fact per claim. Don't bundle multiple distinct claims into a single statement, and don't emit two facts with the same statement but different SPO decompositions — pick the one that follows the rules above.
- Return at most {MAX_FACTS} facts.";

/// Additional system message appended after [`BUNDLED_SYSTEM_PROMPT`] when
/// `ctx.extract_mode == ExtractMode::DurableOnly`. Plumbed from the source
/// thought's `metadata.extract` field by the reflector. The bundled prompt
/// already includes a mixed-content rule; this hint reinforces it per-thought
/// when the operator has explicitly flagged the capture as mixed-content.
pub const DURABLE_ONLY_HINT: &str = "\
The captured thought is known to mix durable claims with transient session narrative — the operator has flagged it as mixed-content. \
Extract only the durable claims; skip the session-narrative parts entirely, even if they're well-formed sentences.";

#[derive(Serialize)]
struct ChatRequestBody<'a> {
    model: &'a str,
    temperature: f32,
    /// Length-varying because `ExtractMode::DurableOnly` appends a second
    /// system message after the bundled prompt. JSON wire shape is unchanged
    /// (an array of role/content objects).
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

#[derive(Deserialize)]
struct ExtractionPayload {
    facts: Vec<ExtractedFactDto>,
}

#[derive(Deserialize)]
struct ExtractedFactDto {
    statement: String,
    #[serde(default)]
    subject: Option<String>,
    #[serde(default)]
    predicate: Option<String>,
    #[serde(default)]
    object: Option<String>,
    confidence: f32,
}

impl From<ExtractedFactDto> for ExtractedFact {
    fn from(d: ExtractedFactDto) -> Self {
        Self {
            statement: d.statement,
            subject: d.subject.filter(|s| !s.is_empty()),
            predicate: d.predicate.filter(|s| !s.is_empty()),
            object: d.object.filter(|s| !s.is_empty()),
            confidence: d.confidence.clamp(0.0, 1.0),
        }
    }
}

#[async_trait]
impl Extractor for OpenAICompatibleExtractor {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn version(&self) -> i32 {
        self.model_version
    }

    async fn extract(
        &self,
        thought: &Thought,
        ctx: &ExtractionContext,
    ) -> Result<Vec<ExtractedFact>, ExtractorError> {
        let max_facts = ctx.max_facts.min(self.max_facts_per_thought).max(1);
        let url = format!("{}/chat/completions", self.endpoint.trim_end_matches('/'));

        let system_prompt = self.system_prompt.replace("{MAX_FACTS}", &max_facts.to_string());
        let mut messages: Vec<ChatMessage<'_>> = Vec::with_capacity(3);
        messages.push(ChatMessage {
            role: "system",
            content: system_prompt,
        });
        // Per-thought `metadata.extract = "durable-only"` propagates through
        // `ctx.extract_mode` and adds a second system message instructing the
        // model to bias toward durable claims. The bundled prompt's
        // mixed-content rule does most of the work; this hint reinforces it
        // when the operator has explicitly flagged the thought.
        if ctx.extract_mode == ExtractMode::DurableOnly {
            messages.push(ChatMessage {
                role: "system",
                content: DURABLE_ONLY_HINT.to_string(),
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: thought.content.clone(),
        });
        let body = ChatRequestBody {
            model: &self.model_name,
            temperature: self.temperature,
            messages,
            response_format: facts_response_format(),
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
            let message = resp.text().await.unwrap_or_default();
            return Err(ExtractorError::Backend {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: ChatResponseBody = resp.json().await.map_err(|e| {
            ExtractorError::MalformedResponse(format!("decoding chat completions response: {e}"))
        })?;

        let content = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ExtractorError::MalformedResponse("response had zero choices".into()))?
            .message
            .content;

        let payload: ExtractionPayload = serde_json::from_str(&content).map_err(|e| {
            ExtractorError::MalformedResponse(format!(
                "decoding facts payload (content={content:?}): {e}"
            ))
        })?;

        Ok(payload
            .facts
            .into_iter()
            .take(max_facts)
            .map(ExtractedFact::from)
            .collect())
    }
}

/// The `response_format` JSON object sent to the chat completions API. The
/// schema constrains the model to a `{facts: [...]}` shape with the
/// statement/subject/predicate/object/confidence fields per item.
fn facts_response_format() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "engram_facts",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "statement": {"type": "string"},
                                "subject": {"type": ["string", "null"]},
                                "predicate": {"type": ["string", "null"]},
                                "object": {"type": ["string", "null"]},
                                "confidence": {"type": "number"}
                            },
                            "required": ["statement", "subject", "predicate", "object", "confidence"]
                        }
                    }
                },
                "required": ["facts"]
            }
        }
    })
}

fn map_send_error(e: reqwest::Error, timeout_seconds: u64) -> ExtractorError {
    if e.is_timeout() {
        ExtractorError::Timeout {
            seconds: timeout_seconds,
        }
    } else if e.is_connect() {
        ExtractorError::Unreachable(e.to_string())
    } else if let Some(status) = e.status() {
        ExtractorError::Backend {
            status: status.as_u16(),
            message: e.to_string(),
        }
    } else {
        ExtractorError::Unreachable(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{Metadata, Scope, Source, ThoughtId};
    use serde_json::json;
    use time::OffsetDateTime;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_thought(content: &str) -> Thought {
        Thought {
            id: ThoughtId::new(),
            scope: Scope::global(),
            content: content.to_string(),
            source: Source::new("test").unwrap(),
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
            metadata: Metadata::empty(),
        }
    }

    fn ctx(max: usize) -> ExtractionContext {
        ExtractionContext::new(Scope::global(), max)
    }

    /// Phase C v4 prompt content pin: relations rule, reinforced few-shots
    /// for the leak set, and the flagged-band framing must all be present.
    /// Catches accidental deletions during downstream edits.
    #[test]
    fn bundled_prompt_v4_contains_relations_rule_and_flagged_band_note() {
        let p = BUNDLED_SYSTEM_PROMPT;
        // Relations rule + canonical example.
        assert!(
            p.contains("relative / preference claims") || p.contains("Ron prefers Rust over Go"),
            "v4 relations rule missing",
        );
        // Reinforced few-shots for the leak set.
        assert!(p.contains("Nix is more reproducible than Make"), "Nix/Make few-shot missing");
        assert!(p.contains("subject=object"), "self-referential rejection example missing");
        // Confidence rubric calls out the 0.70–0.85 middle band as
        // "stored but flagged".
        assert!(
            p.contains("stored but flagged"),
            "flagged-band confidence framing missing",
        );
        // Vllm + open_router presets bumped to v4.
        let cfg = OpenAICompatibleConfig::vllm_local();
        assert_eq!(cfg.model_version, 4);
        let cfg = OpenAICompatibleConfig::open_router("k".into(), "m".into());
        assert_eq!(cfg.model_version, 4);
    }

    fn config_for(endpoint: String, api_key: Option<String>) -> OpenAICompatibleConfig {
        OpenAICompatibleConfig {
            endpoint,
            model_name: "test-model".to_string(),
            model_id: "test/test-model".to_string(),
            model_version: 1,
            api_key,
            timeout: Duration::from_secs(2),
            temperature: 0.0,
            max_facts_per_thought: 8,
            system_prompt: None,
        }
    }

    fn chat_response_with_facts(facts: serde_json::Value) -> serde_json::Value {
        let content = serde_json::to_string(&json!({"facts": facts})).unwrap();
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
    async fn valid_response_parses_to_facts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_with_facts(json!([
                {
                    "statement": "Engram uses pgvector",
                    "subject": "Engram",
                    "predicate": "uses",
                    "object": "pgvector",
                    "confidence": 0.92
                },
                {
                    "statement": "Single-user assumption holds in v0",
                    "subject": null,
                    "predicate": null,
                    "object": null,
                    "confidence": 0.75
                }
            ]))))
            .mount(&server)
            .await;

        let e = OpenAICompatibleExtractor::new(config_for(format!("{}/v1", server.uri()), None))
            .unwrap();
        let facts = e.extract(&make_thought("..."), &ctx(8)).await.unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].statement, "Engram uses pgvector");
        assert_eq!(facts[0].subject.as_deref(), Some("Engram"));
        assert!((facts[0].confidence - 0.92).abs() < 1e-4);
        assert!(facts[1].subject.is_none());
    }

    #[tokio::test]
    async fn malformed_json_in_message_content_returns_malformed_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"role": "assistant", "content": "not json"}}]
            })))
            .mount(&server)
            .await;

        let e = OpenAICompatibleExtractor::new(config_for(format!("{}/v1", server.uri()), None))
            .unwrap();
        let err = e.extract(&make_thought("x"), &ctx(8)).await.unwrap_err();
        assert!(matches!(err, ExtractorError::MalformedResponse(_)));
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn http_500_returns_backend_transient() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream gone"))
            .mount(&server)
            .await;

        let e = OpenAICompatibleExtractor::new(config_for(format!("{}/v1", server.uri()), None))
            .unwrap();
        let err = e.extract(&make_thought("x"), &ctx(8)).await.unwrap_err();
        match err {
            ExtractorError::Backend { status, .. } => assert_eq!(status, 503),
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

        let e = OpenAICompatibleExtractor::new(config_for(format!("{}/v1", server.uri()), None))
            .unwrap();
        let err = e.extract(&make_thought("x"), &ctx(8)).await.unwrap_err();
        match &err {
            ExtractorError::Backend { status, .. } => assert_eq!(*status, 400),
            other => panic!("expected Backend error, got {other:?}"),
        }
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn connect_failure_maps_to_unreachable_or_timeout() {
        // Port 1 is reliably refused on macOS/Linux.
        let e =
            OpenAICompatibleExtractor::new(config_for("http://127.0.0.1:1/v1".to_string(), None))
                .unwrap();
        let err = e.extract(&make_thought("x"), &ctx(8)).await.unwrap_err();
        assert!(
            matches!(err, ExtractorError::Unreachable(_) | ExtractorError::Timeout { .. }),
            "expected Unreachable or Timeout, got {err:?}"
        );
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn system_prompt_includes_max_facts_limit() {
        let server = MockServer::start().await;
        // Match only when the system message text mentions "at most 4 facts."
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "messages": [
                    {"role": "system", "content": serde_json::Value::String("__placeholder__".to_string())},
                    {"role": "user", "content": "x"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_with_facts(json!([]))))
            .mount(&server)
            .await;

        let e = OpenAICompatibleExtractor::new(config_for(format!("{}/v1", server.uri()), None))
            .unwrap();
        // Lower max — used to substitute {MAX_FACTS} in the system prompt.
        let _ = e.extract(&make_thought("x"), &ctx(4)).await;
        // The mock accepts any system content, but we also verify by
        // inspecting all requests it received and asserting the substitution
        // happened.
        let received = server.received_requests().await.unwrap();
        let last = received.last().expect("at least one request");
        let body: serde_json::Value = serde_json::from_slice(&last.body).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(
            sys.contains("at most 4 facts"),
            "system prompt did not substitute max_facts: {sys}"
        );
    }

    #[tokio::test]
    async fn request_uses_bearer_auth_when_api_key_present() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_with_facts(json!([]))))
            .mount(&server)
            .await;

        let e = OpenAICompatibleExtractor::new(config_for(
            format!("{}/v1", server.uri()),
            Some("sk-test".into()),
        ))
        .unwrap();
        // If the auth header is wrong, wiremock returns 404 and the parse fails.
        e.extract(&make_thought("x"), &ctx(8))
            .await
            .expect("auth header must match");
    }

    #[tokio::test]
    async fn empty_endpoint_is_misconfigured() {
        let mut cfg = config_for("".to_string(), None);
        cfg.endpoint = "".into();
        let err = OpenAICompatibleExtractor::new(cfg).unwrap_err();
        assert!(matches!(err, ExtractorError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn custom_system_prompt_flows_into_request_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_with_facts(json!([]))))
            .mount(&server)
            .await;

        let mut cfg = config_for(format!("{}/v1", server.uri()), None);
        cfg.system_prompt = Some(
            "Custom prompt for the dogfood week. Return at most {MAX_FACTS} facts.".to_string(),
        );
        let e = OpenAICompatibleExtractor::new(cfg).unwrap();
        let _ = e.extract(&make_thought("x"), &ctx(7)).await;

        let received = server.received_requests().await.unwrap();
        let last = received.last().expect("at least one request");
        let body: serde_json::Value = serde_json::from_slice(&last.body).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.contains("Custom prompt for the dogfood week"));
        // Per-call substitution still works for custom prompts.
        assert!(sys.contains("at most 7 facts"));
        // Bundled-prompt language must NOT leak in.
        assert!(!sys.contains("episodic"));
    }

    #[tokio::test]
    async fn custom_system_prompt_missing_max_facts_placeholder_is_misconfigured() {
        let mut cfg = config_for("http://127.0.0.1:1/v1".to_string(), None);
        cfg.system_prompt = Some("a prompt that forgot to include the placeholder".to_string());
        let err = OpenAICompatibleExtractor::new(cfg).unwrap_err();
        match err {
            ExtractorError::Misconfigured(msg) => {
                assert!(msg.contains("MAX_FACTS"), "msg should name the placeholder: {msg}");
            }
            other => panic!("expected Misconfigured, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bundled_system_prompt_constant_contains_max_facts_placeholder() {
        // Regression guard: if anyone ever edits the bundled prompt and
        // drops the placeholder, this test will catch it before facts
        // start landing under an unanchored prompt.
        assert!(
            BUNDLED_SYSTEM_PROMPT.contains("{MAX_FACTS}"),
            "BUNDLED_SYSTEM_PROMPT must contain the {{MAX_FACTS}} placeholder",
        );
    }

    #[tokio::test]
    async fn caps_facts_at_min_of_ctx_and_config() {
        let server = MockServer::start().await;
        // Server returns 10 facts; ctx max is 3; config max is 8 — result is 3.
        let many = (0..10).map(|i| json!({
            "statement": format!("f{i}"),
            "subject": null, "predicate": null, "object": null,
            "confidence": 0.9
        })).collect::<Vec<_>>();
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_with_facts(json!(many))))
            .mount(&server)
            .await;

        let e = OpenAICompatibleExtractor::new(config_for(format!("{}/v1", server.uri()), None))
            .unwrap();
        let facts = e.extract(&make_thought("x"), &ctx(3)).await.unwrap();
        assert_eq!(facts.len(), 3);
    }

    /// Live test against a real OpenAI-compatible endpoint (vLLM by default).
    /// Gated on the `integration` feature; off in CI. Run with
    /// `cargo test -p engram-extract --features integration -- live_vllm`.
    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn live_vllm_round_trip() {
        let cfg = OpenAICompatibleConfig::vllm_local();
        let e = OpenAICompatibleExtractor::new(cfg).unwrap();
        let t = make_thought(
            "Engram uses pgvector for vector storage and pg_trgm for trigram search.",
        );
        let facts = e
            .extract(&t, &ctx(4))
            .await
            .expect("vLLM unreachable — is it running on :8000?");
        assert!(!facts.is_empty(), "live extractor produced zero facts");
        assert!(facts.iter().all(|f| (0.0..=1.0).contains(&f.confidence)));
    }
}

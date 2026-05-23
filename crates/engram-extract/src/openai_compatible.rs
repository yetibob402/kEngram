//! `OpenAICompatibleTagger` — talks to any backend that implements the
//! OpenAI `/v1/chat/completions` API with `response_format: json_schema`.
//! That covers vLLM (production), OpenRouter (cloud fallback), and OpenAI
//! itself, distinguished only by config.
//!
//! Endpoint convention: the configured `endpoint` is the `/v1` base, and
//! the tagger appends `/chat/completions`. For local vLLM that's
//! `http://localhost:8000/v1`.

use async_trait::async_trait;
use engram_core::{ExtractedRelation, ScopeVocab, TagOutput, Tagger, TaggerError, Tags};
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
/// controlled-vocabulary section; v3 (M4.1 prompt iteration) tightened
/// entities to canonical proper names only with an explicit anti-padding
/// rule, and added a kind-isolation clause forbidding the controlled
/// vocabulary from influencing kind classification; v4 (M4.1 prompt
/// iteration, second pass) restructured the entities description to lead
/// with the empty case and a structural NAME-vs-DESCRIBE test (the v3
/// negative-example list backfired — the model emitted those exact phrases
/// from `047d0ce8`), dropped entities maxItems 5→3, and softened the
/// scope-vocabulary section from "vocab dominates" to "vocab tie-breaks"
/// (precision over consistency); v5 (M6.1) added tagger-extracted
/// relations — the LLM emits closed-vocabulary `(relation, to_kind,
/// to_value)` edges for explicit relational claims in prose, non-thought
/// targets only (entity / person / url); v6 (post-M6.1 dogfood pass 1)
/// rebalanced kind classification + added an entity surface-only rule +
/// tightened URL emission, but the entities section listed `embedding-based`
/// and `lexical signals` as literal negative examples, which repeated the
/// v3→v4 backfire pattern (the model emitted the listed phrases verbatim
/// on `047d0ce8` again); **v7 (post-v6 dogfood pass 2)** drops the
/// literal-phrase NOT-entities list and any suffix hints (e.g. `-based`),
/// relying on the structural NAME-vs-DESCRIBE test, the surface-only
/// rule, and the re-read verification alone — mirrors v4's clean-pattern
/// fix and documents the v6 lesson explicitly so a v8 doesn't
/// reintroduce phrase hints. v7 also adds an explicit topics
/// concept-mapping intent statement (topics may be inferred when the
/// subject is clear; surface lexemes are not required), which had been
/// de-facto behavior since v4 vocab-softening but wasn't stated.
/// **v8 (post-v7 dogfood pass 3)** removes Rust from the `topics`
/// examples list and from the kind=observation exemplar pair. Root
/// cause: first-item example-list priming caused gemma3:12b to over-
/// emit `"rust"` as a topic on tech-adjacent thoughts that weren't
/// about Rust (probes C/D/G of the 2026-05-22 tagger-test scope).
/// Entities examples still include "Rust" — the surface-only rule
/// prevents spurious emission there, and it's a legitimate canonical
/// entity in this corpus.
/// **v9 (post-v8 dogfood pass 4)** drops the standalone topics
/// `Examples: ...` clause entirely. v8's swap (Rust out, databases
/// in) just rotated the priming target — across the post-v8 retag
/// 13 of 18 thoughts emitting `"databases"` as a topic weren't
/// about databases (engram API design, branding, capture discipline,
/// serialization, etc.). The structural issue was the example list,
/// not which items were in it. v9 leans on the prose example at the
/// end of the topics paragraph ("a thought naming engram and
/// pgvector might have topics [memory-systems, databases]") and
/// the concept-mapping intent statement to teach the field, without
/// a free-floating priming list.
/// **v10** was an ephemeral version used as a toml override during
/// the 2026-05-22 scope_vocab experiment (toggle vocab off, measure).
/// Not shipped to source — the const skips from 9 to 11.
/// **v11 (post-v9 dogfood pass 5)** moves topic canonical-form
/// convergence from in-prompt vocab hints to a post-process
/// normalization step in `engram-mcp::drain`. The post-v9 retag
/// revealed a second overreach mechanism: the worker drainer feeds
/// the scope's established topic vocab into the prompt as canonical
/// hints, and the LLM treats them as menu items — overreaching on
/// `"databases"` exactly the way it had previously overreached on
/// `"rust"`. v11 separates emission (LLM's job — what is this
/// thought about?) from normalization (post-process — what's the
/// canonical form?). Topics are no longer fed into the LLM prompt;
/// after the LLM emits topics, the drainer normalizes each emitted
/// topic against the scope vocab via string similarity (lowercased
/// Levenshtein + token-subset detection). Entities continue to
/// flow into the prompt vocab section — the surface-only rule
/// prevents entity overreach, and the in-prompt hint preserves
/// canonical casing/spelling.
/// **v12 (post-v11 dogfood pass 6)** adds two related structural
/// fixes for residual issues the v8-v11 work didn't address. (a)
/// Positive syntactic-disambiguation rule in the `people` field
/// instruction: "name + article-at-sentence-start = verb usage."
/// Framed positively to avoid the v3→v4/v6→v7 backfire pattern of
/// negative-example lists. Targets the "Bob the worker batch limit"
/// failure mode where probes B and E consistently mis-route a
/// sentence-start verb into `people`. (b) Post-process disjointness
/// validator in `engram-mcp::validate` — strips any `entities` entry
/// whose lowercased form duplicates a `people` entry. Person wins on
/// tie; the validator is unconditional and runs after the existing
/// topic-normalize step. Catches field contamination regardless of
/// which model or prompt version emitted the tags.
/// **v13 (post-v12 dogfood pass 7)** adds use-mention discipline to
/// the prompt — a `# First discipline: USE vs MENTION` section at the
/// top, a matching `# Rules` bullet, and a `# Examples` section with
/// 6 worked input→output pairs covering parenthetical mentions,
/// demonstrative lists, quoted-directive citations, real-reference
/// controls, and meta-discussion of other thoughts. Iterated locally
/// against gemma3:12b using the new `examples/tagger_eval.rs` harness
/// against `crates/engram-extract/tests/fixtures/use_mention.json`
/// (12 fixtures) until ≥11/12 stably passed (6/6 control plus 5/6
/// use_mention; only `meta-discussion-of-contamination` resists —
/// Sarah lands in `entities`, a residual that needs either a
/// structural pre-process or a larger model). Fixes the corpus-wide
/// pollution of meta-content thoughts (engram.m3.dogfood scope,
/// recommendation thoughts) that mention names as linguistic
/// examples.
pub const BUNDLED_TAGGER_VERSION: i32 = 13;

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
/// bumping the version — see `DESIGN.md` §6 / §10.
pub const BUNDLED_TAGGER_PROMPT: &str = "\
You are a tagging assistant. Given a single thought from a memory service, return its metadata tags as JSON.

# First discipline: USE vs MENTION

Before extracting anything, decide: is each token in the prose being USED, or being MENTIONED?

USED — a real reference, an actor, a directive THIS thought commits to, a named thing this thought is genuinely about. Extract.

MENTIONED — a token cited as a linguistic example, a candidate being weighed in a brainstorm, a quoted directive THIS thought describes but does not itself issue, a name referenced because another thought tagged or discussed it, an item in an \"e.g.\" or \"such as\" or \"like\" list. Do NOT extract.

When a thought is itself ABOUT names, tags, or other thoughts (meta-discussion), every name listed as a discussion target is a MENTION. Empty `people`, `entities`, and `action_items` arrays are the correct output when the entire thought consists of mentions.

Apply this check FIRST. Then proceed to the field rules below.

# Output shape
{ \"people\": [...], \"entities\": [...], \"kind\": \"...\", \"action_items\": [...], \"topics\": [...], \"dates_mentioned\": [...], \"relations\": [...] }

# Field semantics

- people: bare names of people mentioned. Empty array if none. Syntactic disambiguation: when a capitalized word appears at the START of a sentence followed by an article (\"the\", \"a\", \"an\") or a possessive (\"your\", \"my\", \"this\"), it is being used as an imperative verb, not as a person's name — emit it as part of the action_items phrase instead, and leave it out of `people`. Example: \"Mark the migration as complete\" — \"Mark\" is a verb; if no other names are mentioned, `people` is empty.

- entities: default to []. Only emit a name that the thought MENTIONS BY ITS SURFACE NAME — a specific named thing (project, product, library, tool, technology, organization) with its own canonical identity independent of this thought. Examples of valid entities: \"engram\", \"pgvector\", \"PostgreSQL\", \"MCP\", \"TCGplayer\", \"Cap'n Proto\", \"Rust\", \"Hummingbird\". Preserve the thought's casing, or use canonical casing if the thought is inconsistent.

  Surface-only rule (load-bearing): entities must appear in the thought's prose. Do NOT infer entities from world knowledge. Example failure: if the thought says \"trigram retrieval\", do NOT emit `pg_trgm` even though pg_trgm is the Postgres extension that implements trigram retrieval — that is a world-knowledge inference, not surface evidence. The name (or a clearly-recognizable abbreviation that maps to it) must appear in the prose.

  NAME-vs-DESCRIBE test (apply this to every candidate before including it): ask \"does this phrase NAME a specific thing that has its own canonical identity outside this thought, or does the thought DESCRIBE an action / concept / style using a noun phrase?\" Only names belong in entities. When in doubt, omit. If the same phrase could be a name elsewhere but is used descriptively here, omit.

  Before you emit: re-read the thought. Verify each entity in your output appears (by name or close paraphrase) in the prose. Remove any that don't.

- kind: a single closed-enum classification of what the thought DOES. Pick exactly one of: observation | task | idea | reference | person_note | session | null. Walk this decision tree in order and pick the FIRST kind that fits — do NOT default to observation:

  1. Does the thought DEFINE or POINT AT a specific named thing (a project, paper, tool, term, organization)?
     - Yes, and the named thing is a person → person_note. (\"Sarah prefers async meetings.\")
     - Yes, otherwise → reference. (\"Hummingbird is our internal rollout coordinator.\" \"Cap'n Proto offers zero-copy reads.\")

  2. Does the thought COMMIT TO or DESCRIBE an action to take?
     - Yes → task. (\"Mission: test the engram MCP toolset for accuracy.\" \"Fix the login bug.\")

  3. Does the thought PROPOSE, HYPOTHESIZE, or REPORT a finding/conclusion?
     - Yes → idea. (\"We could use Bloom filters here.\" \"Probe 2 confirms that topics are phrase-driven.\" \"The reranker beat RRF-only by 7% on this fixture.\")

  4. Is the thought NARRATING current-session activity (\"I just ran X\", \"the search returned Y\", \"this test passed\")?
     - Yes → session. Session-shaped thoughts typically have otherwise-empty arrays.

  5. Otherwise — a pure factual claim about the world with no commitment, no proposal, no definition, no narrative — observation. (\"Postgres autovacuum thresholds are tunable per-table.\" \"JSON parsing benefits from SIMD on documents over 1 MB.\")

  Anti-default: observation is the CATCHALL, not the default. When the thought arguably fits a more specific kind, prefer the more specific kind. A degenerate tagger that classifies every thought as observation is a failure mode v6 is designed to inverse.

  Kind is classified from the thought's intrinsic shape only — never from the scope's typical content, never from controlled-vocabulary hints below. The vocabulary section informs topic and entity term choice; it does NOT influence kind.

- action_items: short imperative phrases describing tasks the thought commits to or implies (e.g., \"fix the login bug\", \"review the migration plan\"). Empty array if none. Distinct from kind=task: action_items is the per-thought list of items; kind=task is the thought's overall classification.

- topics: 1-3 short tag-like subject categories, lowercase, hyphen-separated, no punctuation. What broad SUBJECT AREA is this thought about? Topics map prose to canonical subject categories — they may be inferred from context when the subject is clear, even if the exact topic word doesn't appear in the thought. Two thoughts about the same subject (e.g. one mentioning \"trigram retrieval\", another mentioning \"vector similarity\") may share topics (\"information-retrieval\") even with disjoint surface vocabulary. This is concept-mapping behavior, not surface-lexeme lifting. Distinct from entities: a topic is a category the thought falls under; an entity is a specific named thing the thought mentions. A thought naming \"engram\" and \"pgvector\" might have entities [\"engram\", \"pgvector\"] and topics [\"memory-systems\", \"databases\"].

- dates_mentioned: any dates or temporal references appearing in the prose (\"next Thursday\", \"Q3\", \"2026-05-15\", \"before the release\"). Free-form strings, copied roughly as they appear. Empty array if none.

- relations: default to []. Closed-vocabulary edges from this thought to non-thought targets. Emit ONLY when the prose makes an EXPLICIT relational claim — not on adjacency, mere mention, or vague allusion. Each entry: `relation` (closed vocab), `to_kind` ∈ {entity, person, url}, `to_value` (canonical name or full URL), optional `note`.

  - Relation vocabulary:
    - references: prose cites or points at the target for context (passive mention).
    - supports: prose contains a claim that CONFIRMS the target. Distinct from references (which is a passive citation).
    - decided_by: thought attributes a decision to the target. Requires explicit attribution (\"the team decided X\" → decided_by team).
    - belongs_to: thought is a sub-element/member of the target (\"a finding under Probe 2\" → belongs_to entity \"Probe 2\").
    - requires: thought depends on the target (explicit prerequisite).
    - refines, replaces: skip — these target other thoughts; v1 cannot resolve thought targets.

  - to_kind:
    - url: a FULL `http://` or `https://` URL appearing verbatim in the prose. If the thought has only a bare domain (\"example.com\"), partial path (\"/docs/foo\"), or paper citation without scheme (\"arxiv.org/abs/2004.04906\" — no http(s):// prefix), do NOT emit a url relation; the schema validation will reject it. Prefer an entity target instead, or omit.
    - entity: a named non-person thing.
    - person: a named individual.

  - Selectivity: maxItems 5. Require an explicit relational verb or construction in the prose; mere mention is not a relation.

# Rules

- Entities require explicit surface mention. Topics may be inferred from context. Kind is intrinsic-shape only.
- Empty arrays are correct when there's no content. Empty arrays are NOT a tagger-failure signal; over-emission is.
- One kind only; if genuinely ambiguous, return null.
- This is a tagging pass, not a paraphrase. Do not rephrase content; only emit metadata.
- A name belongs in EITHER `people` OR `entities`, never both. Persons go in `people`; non-person named things go in `entities`. If a single string would legitimately fit both categories (a tool named after a person), pick one based on the thought's dominant framing.
- USE vs MENTION discipline: tokens that appear inside quotation marks, inside parenthetical \"e.g.\" or \"such as\" or \"like\" constructs, in demonstrative lists of candidates, as enumerated brainstorm items, or as illustrative linguistic examples are MENTIONED, not USED. They describe candidates being weighed, examples being cited, or words being discussed — not actors, references, or directives. Do NOT extract their contents into `people`, `entities`, or `action_items`. Apply this check before each emission. When a thought is itself ABOUT names or tags (meta-discussion), names listed as examples are mentions, not people.

# Examples

These show how to apply the rules above. Pay attention to use-mention discipline: tokens that appear inside quotation marks, parenthetical \"e.g.\" or \"such as\" constructs, demonstrative lists of candidates, or brainstorm enumerations are MENTIONED, not USED. Do not extract their contents into people, entities, or action_items.

Example 1 — parenthetical mention does not extract:
Thought: 'The Bob-as-verb pattern (e.g., \"Bob the index rebuild\") needs investigation in the next sprint.'
Output: {\"people\": [], \"entities\": [], \"action_items\": [\"investigate the Bob-as-verb pattern\"], \"topics\": [\"linguistics\"], \"dates_mentioned\": [\"next sprint\"], \"kind\": \"task\", \"relations\": []}

Example 2 — demonstrative list of names does not extract:
Thought: 'Common verb-as-name first names include: Bob, Mark, Rob, Frank.'
Output: {\"people\": [], \"entities\": [], \"action_items\": [], \"topics\": [\"linguistics\"], \"dates_mentioned\": [], \"kind\": \"observation\", \"relations\": []}

Example 3 — real references DO extract (contrast with Example 2):
Thought: 'Sarah and Bob agreed on the migration plan after this morning's standup.'
Output: {\"people\": [\"Sarah\", \"Bob\"], \"entities\": [], \"action_items\": [], \"topics\": [\"project-management\"], \"dates_mentioned\": [\"this morning\"], \"kind\": \"observation\", \"relations\": []}

Example 4 — quoted directives are citations of language, not directives THIS thought makes:
Thought: 'The probe should use prompts like \"evaluate options A through F\" and \"pick one\" to test how the tagger handles list-shaped instructions.'
Output: {\"people\": [], \"entities\": [], \"action_items\": [\"test how the tagger handles list-shaped instructions\"], \"topics\": [\"tagging-systems\"], \"dates_mentioned\": [], \"kind\": \"task\", \"relations\": []}

Example 5 — meta-discussion: when a thought is ABOUT other thoughts and lists names as illustrative cases, those names are mentions, not people:
Thought: 'Thought abc123 mentions Ron, Sarah, and Bob only as examples of contamination cases — none of them are referenced as actors in this thought.'
Output: {\"people\": [], \"entities\": [], \"action_items\": [], \"topics\": [\"tagging-systems\"], \"dates_mentioned\": [], \"kind\": \"observation\", \"relations\": []}

Example 6 — meta-discussion of tagger behavior: when prose describes a previous tagger's behavior on a different thought, the names in that description are mentions of what was tagged, not extractions for THIS thought:
Thought: 'Probe E at v11 emitted Sarah in both people and entities. The disjointness validator catches this contamination pattern.'
Output: {\"people\": [], \"entities\": [], \"action_items\": [], \"topics\": [\"tagging-systems\"], \"dates_mentioned\": [], \"kind\": \"observation\", \"relations\": []}

Key signal: if the prose explicitly says \"as examples,\" \"only as mentions,\" \"not referenced as actors,\" or describes what a name was tagged-as in another context, the name is a mention. Do not extract.

# Before you emit — final pass

1. Kind: did you walk the 5-step decision tree, or did you default to observation? Walk it now if not.
2. Entities: re-read the thought. Does each entity in your output appear (by name or close paraphrase) in the prose? Remove any that don't.
3. Relations: does each emission correspond to an explicit relational claim in the prose? Remove speculative ones. For url-kind relations: does `to_value` start with `http://` or `https://`? If not, drop or convert to entity.";

/// Render the optional controlled-vocabulary section appended to the system
/// prompt when scope vocabulary is available. Returns an empty string when
/// the vocab is `None` or has no entities, so callers can unconditionally
/// concatenate the result.
///
/// v11: only entities render into the prompt. Topics are intentionally
/// excluded — they get post-process normalization in `engram-mcp::drain`
/// instead. Entities are surface-bound (the prose must mention them) so
/// the in-prompt hint helps with canonical casing/spelling without
/// risking the overreach feedback loop that topics exhibited at v8-v10.
fn render_vocab_section(vocab: Option<&ScopeVocab>) -> String {
    let Some(v) = vocab else {
        return String::new();
    };
    if v.entities.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\n# Controlled vocabulary (this scope's established terms)\n");
    out.push_str("Entities already used in this scope: ");
    out.push_str(&v.entities.join(", "));
    out.push_str(".\n");
    out.push_str(
        "These are entity names other thoughts in this scope have used. When a thought genuinely mentions one of them, prefer the established casing/spelling. Do not emit an entity that doesn't appear in the prose just because it's listed here — the surface-only rule still binds.",
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
    ) -> Result<TagOutput, TaggerError> {
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

        // Parse the LLM response into a transient document with both
        // Tags fields AND relations. Split into TagOutput; the Tags
        // portion gets persisted to thoughts.tags JSONB, the relations
        // portion gets routed to thought_links by the drainer.
        let doc: TaggerResponseDoc = serde_json::from_str(&content).map_err(|e| {
            TaggerError::MalformedResponse(format!(
                "decoding tags payload (content={content:?}): {e}"
            ))
        })?;

        Ok(TagOutput {
            tags: doc.tags,
            relations: doc.relations,
        })
    }
}

/// Wire shape of the LLM response: the same `tags_response_format()` JSON
/// schema the prompt enforces. Parsed transiently before splitting into
/// the persisted `Tags` (JSONB column) and the transient
/// `Vec<ExtractedRelation>` (routed to `thought_links` by the drainer).
#[derive(Debug, Deserialize)]
struct TaggerResponseDoc {
    #[serde(flatten)]
    tags: Tags,
    #[serde(default)]
    relations: Vec<ExtractedRelation>,
}

/// The `response_format` JSON object sent to the chat completions API. The
/// schema constrains the model to the `Tags` wire shape with six required
/// fields; `topics` is capped at 3 items, `entities` at 3 (lowered from 5
/// in the v4 prompt iteration to force selectivity), and `kind` is nullable
/// with an enum of `TagKind` snake_case variants.
fn tags_response_format() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "engram_tags",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "required": [
                    "people", "entities", "action_items", "topics", "dates_mentioned",
                    "kind", "relations"
                ],
                "properties": {
                    "people": { "type": "array", "items": { "type": "string" } },
                    "entities": { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
                    "action_items": { "type": "array", "items": { "type": "string" } },
                    "topics": { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
                    "dates_mentioned": { "type": "array", "items": { "type": "string" } },
                    "kind": {
                        "type": ["string", "null"],
                        "enum": ["observation", "task", "idea", "reference", "person_note", "session", null]
                    },
                    "relations": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["relation", "to_kind", "to_value", "note"],
                            "properties": {
                                "relation": {
                                    "type": "string",
                                    "enum": [
                                        "replaces", "requires", "references", "supports",
                                        "belongs_to", "decided_by", "refines"
                                    ]
                                },
                                "to_kind": {
                                    "type": "string",
                                    "enum": ["entity", "person", "url"]
                                },
                                // No maxLength — Ollama/llama.cpp's GBNF
                                // grammar generation chokes on string-length
                                // caps for some models (e.g.
                                // qwen3-coder:30b → "failed to load model
                                // vocabulary required for format"). The
                                // app-side `link::validate_target` enforces
                                // 2048-char URL / 200-char name caps anyway,
                                // so the schema constraint was redundant
                                // defense-in-depth.
                                "to_value": { "type": "string" },
                                "note": { "type": ["string", "null"] }
                            }
                        }
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
                    "kind": "task",
                    "relations": []
                }))),
            )
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let output = t.tag("anything", None).await.unwrap();
        assert_eq!(
            output.tags.people,
            vec!["Sarah".to_string(), "Ron".to_string()]
        );
        assert_eq!(
            output.tags.entities,
            vec!["engram".to_string(), "pgvector".to_string()]
        );
        assert_eq!(
            output.tags.action_items,
            vec!["fix the login bug".to_string()]
        );
        assert_eq!(
            output.tags.topics,
            vec!["rust".to_string(), "build-systems".to_string()]
        );
        assert_eq!(
            output.tags.dates_mentioned,
            vec!["next Thursday".to_string()]
        );
        assert_eq!(output.tags.kind, Some(TagKind::Task));
        assert!(output.relations.is_empty());
    }

    #[tokio::test]
    async fn valid_response_with_relations_parses_to_tag_output() {
        use engram_core::{ExtractedTarget, RelationKind};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response_with_tags(json!({
                    "people": [],
                    "entities": ["engram"],
                    "action_items": [],
                    "topics": ["memory-systems"],
                    "dates_mentioned": [],
                    "kind": "reference",
                    "relations": [
                        {
                            "relation": "references",
                            "to_kind": "url",
                            "to_value": "https://arxiv.org/abs/2004.04906",
                            "note": "explicit citation"
                        },
                        {
                            "relation": "belongs_to",
                            "to_kind": "entity",
                            "to_value": "Probe 2 experiment",
                            "note": null
                        }
                    ]
                }))),
            )
            .mount(&server)
            .await;

        let t =
            OpenAICompatibleTagger::new(config_for(format!("{}/v1", server.uri()), None)).unwrap();
        let output = t.tag("anything", None).await.unwrap();
        // Relations land in the transient TagOutput.relations vec, NOT in
        // the persisted Tags struct (the tags JSONB column never sees them
        // post-M6.x).
        assert_eq!(output.relations.len(), 2);
        assert_eq!(output.relations[0].relation, RelationKind::References);
        assert_eq!(
            output.relations[0].target,
            ExtractedTarget::Url("https://arxiv.org/abs/2004.04906".into())
        );
        assert_eq!(
            output.relations[0].note.as_deref(),
            Some("explicit citation")
        );
        assert_eq!(output.relations[1].relation, RelationKind::BelongsTo);
        assert_eq!(
            output.relations[1].target,
            ExtractedTarget::Entity("Probe 2 experiment".into())
        );
        assert_eq!(output.relations[1].note, None);
        // Tags structure still has its 6 named fields; relations is not one of them.
        assert_eq!(output.tags.entities, vec!["engram".to_string()]);
    }

    #[test]
    fn tags_response_format_includes_relations_array() {
        let schema = tags_response_format();
        let props = &schema["json_schema"]["schema"]["properties"];
        assert!(props.get("relations").is_some(), "relations missing");
        assert_eq!(props["relations"]["type"], "array");
        assert_eq!(props["relations"]["maxItems"], 5);
        let item_props = &props["relations"]["items"]["properties"];
        assert!(item_props.get("relation").is_some());
        assert!(item_props.get("to_kind").is_some());
        assert!(item_props.get("to_value").is_some());
        // Closed enums on relation + to_kind — pin them so a careless edit
        // doesn't open the vocabulary.
        let relation_enum = item_props["relation"]["enum"].as_array().unwrap();
        assert_eq!(relation_enum.len(), 7);
        let kind_enum = item_props["to_kind"]["enum"].as_array().unwrap();
        assert_eq!(kind_enum.len(), 3);
        for k in ["entity", "person", "url"] {
            assert!(kind_enum.iter().any(|v| v == k));
        }
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

    /// v4 prompt content pin: the tagger prompt must mention field semantics,
    /// list each of the six fields, preserve the entities/topics distinction
    /// and kind-isolation clauses from earlier versions, and surface the v4
    /// entities restructuring (lead-with-empty + NAME-vs-DESCRIBE structural
    /// test) without regressing to the v3 negative-example list (which
    /// backfired in dogfood — the model emitted the listed phrases verbatim).
    #[test]
    fn tagger_v7_prompt_kind_decision_tree_entity_surface_only_and_url_tightening() {
        let p = BUNDLED_TAGGER_PROMPT;
        assert!(
            p.contains("Field semantics"),
            "v7 prompt must contain a 'Field semantics' section"
        );
        for field in [
            "people",
            "entities",
            "action_items",
            "topics",
            "dates_mentioned",
            "kind",
            "relations",
        ] {
            assert!(p.contains(field), "v7 prompt must mention field {field}");
        }
        // The entities/topics distinction must be explicit (kept from v2).
        assert!(
            p.contains("Distinct from entities"),
            "v7 prompt must explicitly distinguish entities from topics"
        );
        // v4 entities lead-with-empty framing preserved.
        assert!(
            p.contains("entities: default to []"),
            "v7 prompt must lead the entities description with the empty case"
        );
        // v4 NAME-vs-DESCRIBE test preserved.
        assert!(
            p.contains("NAME a specific thing"),
            "v7 prompt must include the structural NAME-vs-DESCRIBE entities test"
        );
        // v3 negative-example list still must NOT be present (its backfire
        // was the v3→v4 lesson; v6 uses *patterns* not literal phrases).
        assert!(
            !p.contains("The following are NOT entities"),
            "v7 prompt must NOT contain the v3 negative-example list lead-in"
        );
        // v6 surface-only rule (load-bearing addition addressing the
        // `pg_trgm` knowledge-based hallucination from v5 dogfood).
        assert!(
            p.contains("Surface-only rule"),
            "v7 prompt must contain the surface-only rule heading"
        );
        assert!(
            p.contains("pg_trgm"),
            "v7 prompt must cite the pg_trgm hallucination case as the surface-only example"
        );
        assert!(
            p.contains("world-knowledge"),
            "v7 prompt must explicitly forbid world-knowledge inference"
        );
        // v6 final-pass verification on entities.
        assert!(
            p.contains("re-read the thought"),
            "v7 prompt must include a re-read verification on entities"
        );

        // v6 kind rebalance: 5-step decision tree + anti-default framing.
        assert!(
            p.contains("decision tree"),
            "v7 prompt must frame kind as a decision tree"
        );
        assert!(
            p.contains("Anti-default"),
            "v7 prompt must include the Anti-default framing inverting v5's observation collapse"
        );
        assert!(
            p.contains("CATCHALL"),
            "v7 prompt must label observation as the CATCHALL (not the default)"
        );
        // v3 kind-isolation clause preserved.
        assert!(
            p.contains("intrinsic-shape only") || p.contains("intrinsic shape"),
            "v7 prompt must keep the intrinsic-shape framing on kind"
        );
        assert!(
            p.contains("not from the scope's typical content")
                || p.contains("never from the scope's typical content"),
            "v7 prompt must keep kind isolated from scope-typical content"
        );
        // All 6 non-null kind enum values + null mentioned.
        for k in [
            "observation",
            "task",
            "idea",
            "reference",
            "person_note",
            "session",
        ] {
            assert!(p.contains(k), "v7 prompt must enumerate kind {k:?}");
        }

        // v5 Relations section preserved.
        assert!(
            p.contains("relations: default to []"),
            "v7 prompt must lead the relations description with the empty case"
        );
        assert!(
            p.contains("EXPLICIT relational claim"),
            "v7 prompt must require explicit relational claims (no over-emission)"
        );
        for kind in ["url", "entity", "person"] {
            assert!(
                p.contains(kind),
                "v7 prompt must enumerate target kind {kind:?}"
            );
        }
        // Closed relations vocabulary — all 7 must appear.
        for r in [
            "replaces",
            "requires",
            "references",
            "supports",
            "belongs_to",
            "decided_by",
            "refines",
        ] {
            assert!(p.contains(r), "v7 prompt must enumerate relation `{r}`");
        }
        // v6 URL tightening (addresses the 2/2 dogfood URL rejections).
        assert!(
            p.contains("FULL `http://` or `https://` URL"),
            "v7 prompt must require full http(s):// URLs only"
        );
        assert!(
            p.contains("bare domain"),
            "v7 prompt must explicitly call out bare-domain rejection"
        );

        // v6 final-pass review section (retained in v7).
        assert!(
            p.contains("Before you emit"),
            "v7 prompt must include a 'Before you emit' final-pass review section"
        );

        // v7-specific: the structural NAME-vs-DESCRIBE test must be present
        // AND there must be no "Patterns that are NOT entities" block. The
        // v6 attempt to list adjectival patterns triggered the v3→v4
        // backfire again — the model emitted the listed phrases verbatim
        // from `047d0ce8`. v7 drops the NOT-entities block; the structural
        // test + surface-only rule + final-pass verify alone handle it.
        assert!(
            !p.contains("Patterns that are NOT entities"),
            "v7 prompt must NOT contain a 'Patterns that are NOT entities' block (v6 backfire pattern)"
        );
        // v7 topics-as-concept-mapping intent (was de-facto since v4 vocab
        // softening; now stated explicitly).
        assert!(
            p.contains("concept-mapping"),
            "v7 prompt must explicitly document topics as concept-mapping behavior"
        );

        // Presets track BUNDLED_TAGGER_VERSION.
        assert_eq!(BUNDLED_TAGGER_VERSION, 13);
        let cfg = OpenAICompatibleConfig::vllm_local();
        assert_eq!(cfg.model_version, BUNDLED_TAGGER_VERSION);
        let cfg = OpenAICompatibleConfig::open_router("k".into(), "m".into());
        assert_eq!(cfg.model_version, BUNDLED_TAGGER_VERSION);
    }

    #[test]
    fn tags_response_format_pins_v7_shape() {
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
                "kind",
                "relations"
            ]
        );
        assert_eq!(schema["properties"]["topics"]["maxItems"], 3);
        assert_eq!(schema["properties"]["entities"]["maxItems"], 3);
        assert_eq!(schema["properties"]["relations"]["maxItems"], 5);
        // `kind` must allow null on the wire.
        let kind_type = &schema["properties"]["kind"]["type"];
        assert!(
            kind_type.as_array().unwrap().iter().any(|x| x == "null"),
            "kind must be nullable: {kind_type:?}"
        );
        // relations items pin to the closed vocabularies.
        let item_props = &schema["properties"]["relations"]["items"]["properties"];
        assert_eq!(item_props["relation"]["enum"].as_array().unwrap().len(), 7);
        let to_kind_enum: Vec<&str> = item_props["to_kind"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        assert_eq!(to_kind_enum, vec!["entity", "person", "url"]);
    }

    #[test]
    fn render_vocab_section_handles_none_and_empty() {
        assert_eq!(render_vocab_section(None), "");
        assert_eq!(render_vocab_section(Some(&ScopeVocab::default())), "");
    }

    #[test]
    fn render_vocab_section_renders_entities_only_not_topics() {
        // v11: only entities flow into the prompt vocab section. Topics
        // are intentionally excluded and get post-process normalization
        // in engram-mcp::drain instead.
        let v = ScopeVocab {
            topics: vec!["rust".into(), "memory-systems".into()],
            entities: vec!["engram".into(), "pgvector".into()],
        };
        let rendered = render_vocab_section(Some(&v));
        assert!(rendered.contains("Controlled vocabulary"));
        assert!(rendered.contains("engram, pgvector"));
        assert!(
            !rendered.contains("Topics already used"),
            "v11 must not render topics into the prompt vocab section"
        );
        assert!(
            !rendered.contains("rust, memory-systems"),
            "v11 must not list topic vocab in the prompt — topics are post-normalized"
        );
    }

    #[test]
    fn render_vocab_section_with_topics_only_is_empty() {
        // v11: a vocab with topics but no entities renders to nothing, since
        // topics no longer go through the prompt.
        let topics_only = ScopeVocab {
            topics: vec!["rust".into()],
            entities: vec![],
        };
        assert_eq!(render_vocab_section(Some(&topics_only)), "");
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
            "vocab section must be present in system message when entities are non-empty"
        );
        assert!(
            sys.contains("Entities already used in this scope:"),
            "vocab section must list entities under the established marker"
        );
        assert!(sys.contains("engram"), "entities flow into prompt vocab");
        // v11: topics no longer render into the prompt — they get post-process
        // normalization in engram-mcp::drain instead. The bundled prompt's
        // topics paragraph still mentions "memory-systems" as a prose example
        // (legitimately), so we assert on the vocab-section marker that the
        // old render_vocab_section used for topics.
        assert!(
            !sys.contains("Topics already used in this scope:"),
            "v11 must not render a topics sub-section in the vocab block"
        );
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

    /// Common fixtures for both kind-stability diagnostics (vocab-off and
    /// vocab-on). Seven fixture thoughts pulled from the post-M4.1 corpus,
    /// with `63ad01e0` added in v6 to pin the pg_trgm world-knowledge
    /// hallucination regression. Format: (short_id, scope, current_stored_kind,
    /// descriptor, content).
    #[cfg(feature = "integration")]
    fn diagnostic_fixtures() -> Vec<(
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    )> {
        vec![
            (
                "8a533e15",
                "engram.m3.dogfood",
                "observation",
                "Mission/setup (drift candidate: was task in v1, observation in v2-v5; v6 target: task)",
                "Mission for the engram.m3.dogfood scope: We are testing the Engram agent memory system via its MCP toolset. We are testing for accuracy, making sure that facts don't drift negatively, and that searches return expected information.\n\nWhen scope is a parameter, we will use \"engram.m3.dogfood\".\n\nFor any of our conversations, the agent will always consult the facts and thoughts in that scope, giving more weight to facts than thoughts. The agent will add any interesting thought that it or the operator comes up with during the conversation. If unsure, the agent will ask the operator whether it should be stored.",
            ),
            (
                "047d0ce8",
                "engram.m3.dogfood",
                "observation",
                "Probe 2B definitional (v5 dogfood: emitted adjectival entities `embedding-based`, `lexical signals`; v6 target: NEITHER in entities)",
                "The agent memory protocol provides five operations: writing notes, querying them by similarity or recency, fetching by id, and marking notes untrusted. Querying combines embedding-based and lexical signals, optionally re-scored by a cross-encoder.",
            ),
            (
                "63ad01e0",
                "engram.m3.dogfood",
                "observation",
                "Probe 2A surface-check (v5 dogfood: hallucinated `pg_trgm` from \"trigram retrieval\"; v6 target: `pg_trgm` NOT in entities)",
                "engram's MCP tool surface exposes capture, search_thoughts, get_thought, recent_thoughts, and retract_thought. Search supports hybrid vector + trigram retrieval with optional cross-encoder rerank.",
            ),
            (
                "22bccb3a",
                "engram.test",
                "reference",
                "Clean definitional/reference control (Cap'n Proto)",
                "Cap'n Proto is a serialization format that uses the same memory layout in-memory and on-the-wire, eliminating parse and encode steps. Compared to Protocol Buffers, it offers zero-copy reads but at the cost of more rigid schema evolution. For very high-throughput RPC workloads, it can outperform Protobuf by an order of magnitude.",
            ),
            (
                "5aacd2d8",
                "engram.m3.dogfood",
                "reference",
                "Short definitional/reference control (Hummingbird)",
                "Hummingbird is our internal rollout coordinator. It exposes a percentage knob per cohort and a kill-switch endpoint. Currently used by the privacy-sensitive code paths only.",
            ),
            (
                "b67db532",
                "work.tcgplayer",
                "person_note",
                "Closed-enum person_note control (Ron / Python)",
                "Ron (CTO of TCGplayer) does not like Python or JavaScript, particularly for enterprise software.",
            ),
            (
                "86c3392f",
                "engram.test",
                "observation",
                "Session-shaped narrative control (benchmark run)",
                "I ran a benchmark this morning comparing serde_json and simd-json for parsing 100MB of test JSON. simd-json was 3.2x faster on this hardware (M2 Pro, 16GB). The general finding holds across many tests in the community: SIMD-accelerated JSON parsing significantly outperforms scalar implementations for documents over roughly 1MB. For smaller documents the SIMD setup overhead can negate the benefit.",
            ),
        ]
    }

    /// Top-50 scope vocab for each fixture scope, frozen from the live DB at
    /// 2026-05-17. Lets the vocab-on diagnostic faithfully reproduce what the
    /// worker tick / `engram tag --rerun` actually passes to the tagger,
    /// without taking a sqlx dependency in this crate.
    #[cfg(feature = "integration")]
    fn diagnostic_scope_vocab(scope: &str) -> ScopeVocab {
        match scope {
            "engram.m3.dogfood" => ScopeVocab {
                topics: vec![
                    "tagging-systems".into(),
                    "information-retrieval".into(),
                    "memory-systems".into(),
                    "agent-memory".into(),
                    "concept-mapping".into(),
                    "embedding-models".into(),
                    "search".into(),
                    "fact-management".into(),
                    "internal-tools".into(),
                    "metadata".into(),
                    "privacy".into(),
                    "rollout".into(),
                    "topic-extraction".into(),
                ],
                entities: vec![
                    "Engram".into(),
                    "engram.m3.dogfood".into(),
                    "MCP".into(),
                    "agent memory protocol".into(),
                    "cross-encoder".into(),
                    "embedding-based".into(),
                    "Hummingbird".into(),
                    "lexical signals".into(),
                    "ollama/qwen3-coder:30b v1".into(),
                ],
            },
            "engram.test" => ScopeVocab {
                topics: vec![
                    "performance".into(),
                    "database".into(),
                    "serialization".into(),
                    "storage".into(),
                    "benchmarking".into(),
                    "build-systems".into(),
                    "development-environment".into(),
                    "go".into(),
                    "real-time-updates".into(),
                    "rpc".into(),
                    "rust".into(),
                    "server-sent-events".into(),
                    "tool-comparison".into(),
                    "websockets".into(),
                    "zig".into(),
                ],
                entities: vec![
                    "PostgreSQL".into(),
                    "100MB".into(),
                    "1MB".into(),
                    "Bazel".into(),
                    "C".into(),
                    "Cap'n Proto".into(),
                    "Cassandra".into(),
                    "Go".into(),
                    "long-polling".into(),
                    "M2 Pro".into(),
                    "Make".into(),
                    "MVCC".into(),
                    "Nix".into(),
                    "Protobuf".into(),
                    "Protocol Buffers".into(),
                    "Redis".into(),
                    "Rust".into(),
                    "serde_json".into(),
                    "Server-Sent Events".into(),
                    "simd-json".into(),
                    "SSE".into(),
                    "VACUUM".into(),
                    "WebSockets".into(),
                    "Zig".into(),
                ],
            },
            "work.tcgplayer" => ScopeVocab {
                topics: vec![
                    "programming-languages".into(),
                    "software-development".into(),
                    "technology-preferences".into(),
                    "engram".into(),
                    "enterprise-software".into(),
                    "scope-convention".into(),
                    "scope-design".into(),
                    "search".into(),
                    "thought-management".into(),
                ],
                entities: vec![
                    "TCGplayer".into(),
                    "engram".into(),
                    "Go".into(),
                    "Rust".into(),
                ],
            },
            other => panic!("no frozen scope vocab for scope {other:?}"),
        }
    }

    /// Build the OpenAI-compatible config matching Ron's runtime tagger:
    /// Ollama on :11434 with qwen3-coder:30b, bundled v2 prompt, temperature
    /// 0.2, model_version = BUNDLED_TAGGER_VERSION.
    #[cfg(feature = "integration")]
    fn diagnostic_tagger() -> OpenAICompatibleTagger {
        let cfg = OpenAICompatibleConfig {
            endpoint: "http://localhost:11434/v1".to_string(),
            model_name: "qwen3-coder:30b".to_string(),
            model_id: "ollama/qwen3-coder:30b".to_string(),
            model_version: BUNDLED_TAGGER_VERSION,
            api_key: None,
            timeout: Duration::from_secs(180),
            temperature: 0.2,
            system_prompt: None,
        };
        OpenAICompatibleTagger::new(cfg).expect("OpenAICompatibleTagger::new should succeed")
    }

    /// M4.1 dogfood diagnostic: measure within-tagger `kind` stability by
    /// running N=10 tag passes on each of six fixture thoughts pulled from the
    /// operator's local corpus (two drift-candidates Ron cited plus four
    /// controls). Prints a markdown distribution table; does not assert.
    ///
    /// Configured for Ron's current setup: Ollama on `http://localhost:11434/v1`
    /// with `qwen3-coder:30b`, bundled v2 prompt, temperature 0.2, vocab=None
    /// to isolate from scope-vocab effects.
    ///
    /// Run with:
    /// `cargo test -p engram-extract --features integration --release -- kind_stability_diagnostic --nocapture --ignored`
    ///
    /// `--ignored` because each call is ~5-20s on a 30B Ollama model; 6×10=60
    /// calls means 5-20 minutes wallclock. Not appropriate for the default
    /// integration suite.
    #[cfg(feature = "integration")]
    #[tokio::test]
    #[ignore]
    async fn kind_stability_diagnostic() {
        // Seven fixture thoughts pulled from the post-M4.1 corpus. Format:
        // (short_id, current_stored_kind, descriptor, content). `63ad01e0`
        // added in v6 to pin the pg_trgm world-knowledge hallucination
        // regression observed in v5 dogfood.
        let fixtures: Vec<(&str, &str, &str, &str)> = vec![
            (
                "8a533e15",
                "observation",
                "Mission/setup (drift candidate: was task in v1, observation in v2-v5; v6 target: task)",
                "Mission for the engram.m3.dogfood scope: We are testing the Engram agent memory system via its MCP toolset. We are testing for accuracy, making sure that facts don't drift negatively, and that searches return expected information.\n\nWhen scope is a parameter, we will use \"engram.m3.dogfood\".\n\nFor any of our conversations, the agent will always consult the facts and thoughts in that scope, giving more weight to facts than thoughts. The agent will add any interesting thought that it or the operator comes up with during the conversation. If unsure, the agent will ask the operator whether it should be stored.",
            ),
            (
                "047d0ce8",
                "observation",
                "Probe 2B definitional (v5 dogfood: emitted adjectival entities `embedding-based`, `lexical signals`; v6 target: NEITHER in entities)",
                "The agent memory protocol provides five operations: writing notes, querying them by similarity or recency, fetching by id, and marking notes untrusted. Querying combines embedding-based and lexical signals, optionally re-scored by a cross-encoder.",
            ),
            (
                "63ad01e0",
                "observation",
                "Probe 2A surface-check (v5 dogfood: hallucinated `pg_trgm` from \"trigram retrieval\"; v6 target: `pg_trgm` NOT in entities)",
                "engram's MCP tool surface exposes capture, search_thoughts, get_thought, recent_thoughts, and retract_thought. Search supports hybrid vector + trigram retrieval with optional cross-encoder rerank.",
            ),
            (
                "22bccb3a",
                "reference",
                "Clean definitional/reference control (Cap'n Proto)",
                "Cap'n Proto is a serialization format that uses the same memory layout in-memory and on-the-wire, eliminating parse and encode steps. Compared to Protocol Buffers, it offers zero-copy reads but at the cost of more rigid schema evolution. For very high-throughput RPC workloads, it can outperform Protobuf by an order of magnitude.",
            ),
            (
                "5aacd2d8",
                "reference",
                "Short definitional/reference control (Hummingbird)",
                "Hummingbird is our internal rollout coordinator. It exposes a percentage knob per cohort and a kill-switch endpoint. Currently used by the privacy-sensitive code paths only.",
            ),
            (
                "b67db532",
                "person_note",
                "Closed-enum person_note control (Ron / Python)",
                "Ron (CTO of TCGplayer) does not like Python or JavaScript, particularly for enterprise software.",
            ),
            (
                "86c3392f",
                "observation",
                "Session-shaped narrative control (benchmark run)",
                "I ran a benchmark this morning comparing serde_json and simd-json for parsing 100MB of test JSON. simd-json was 3.2x faster on this hardware (M2 Pro, 16GB). The general finding holds across many tests in the community: SIMD-accelerated JSON parsing significantly outperforms scalar implementations for documents over roughly 1MB. For smaller documents the SIMD setup overhead can negate the benefit.",
            ),
        ];
        const N_RUNS: usize = 10;

        // Match the operator's actual runtime config: Ollama on :11434, the
        // qwen3-coder:30b model, bundled v2 prompt, temperature 0.2, version 2.
        let cfg = OpenAICompatibleConfig {
            endpoint: "http://localhost:11434/v1".to_string(),
            model_name: "qwen3-coder:30b".to_string(),
            model_id: "ollama/qwen3-coder:30b".to_string(),
            model_version: BUNDLED_TAGGER_VERSION,
            api_key: None,
            timeout: Duration::from_secs(180),
            temperature: 0.2,
            system_prompt: None,
        };
        let t =
            OpenAICompatibleTagger::new(cfg).expect("OpenAICompatibleTagger::new should succeed");

        // Per-fixture results: short_id -> (descriptor, current_kind,
        // [observed_kinds; N], [observed_entities; N]). v6 added the
        // entity-set capture so the operator can eyeball the pg_trgm
        // hallucination and adjectival-regression cases in the same run.
        type Observed = (String, String, Vec<String>, Vec<Vec<String>>);
        let mut results: Vec<(String, Observed)> = Vec::new();

        for (short_id, current_kind, descriptor, content) in &fixtures {
            let mut observed_kinds: Vec<String> = Vec::with_capacity(N_RUNS);
            let mut observed_entities: Vec<Vec<String>> = Vec::with_capacity(N_RUNS);
            for run in 0..N_RUNS {
                eprintln!("[diagnostic] {short_id} run {}/{} ...", run + 1, N_RUNS);
                match t.tag(content, None).await {
                    Ok(tags) => {
                        let k = tags
                            .kind
                            .map(|k| format!("{k:?}").to_lowercase())
                            .unwrap_or_else(|| "null".to_string());
                        observed_kinds.push(k);
                        observed_entities.push(tags.entities);
                    }
                    Err(e) => {
                        eprintln!("[diagnostic] {short_id} run {} ERR: {e}", run + 1);
                        observed_kinds.push(format!("ERR({e})"));
                        observed_entities.push(vec![]);
                    }
                }
            }
            results.push((
                short_id.to_string(),
                (
                    descriptor.to_string(),
                    current_kind.to_string(),
                    observed_kinds,
                    observed_entities,
                ),
            ));
        }

        // Render results as a markdown table on stderr.
        eprintln!();
        eprintln!(
            "## v{ver} kind-stability diagnostic results (N={N_RUNS} per thought)",
            ver = BUNDLED_TAGGER_VERSION
        );
        eprintln!();
        eprintln!(
            "Tagger: ollama/qwen3-coder:30b @ http://localhost:11434/v1, bundled v{ver} prompt, temperature 0.2, vocab=None.",
            ver = BUNDLED_TAGGER_VERSION
        );
        eprintln!();
        eprintln!("| short_id | current kind (stored) | descriptor | kind distribution (N=10) |");
        eprintln!("|---|---|---|---|");
        for (short_id, (descriptor, current_kind, kinds, _)) in &results {
            let mut counts: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for k in kinds {
                *counts.entry(k.as_str()).or_insert(0) += 1;
            }
            let dist = counts
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("| `{short_id}` | `{current_kind}` | {descriptor} | {dist} |");
        }
        eprintln!();
        eprintln!("Raw kind observations (one row per fixture):");
        for (short_id, (_, _, kinds, _)) in &results {
            eprintln!("- {short_id}: {kinds:?}");
        }
        eprintln!();
        eprintln!("Entity emissions per fixture (one row per run):");
        for (short_id, (_, _, _, entities)) in &results {
            eprintln!("- {short_id}:");
            for (i, e) in entities.iter().enumerate() {
                eprintln!("    run {}: {e:?}", i + 1);
            }
        }
    }

    /// M4.1 dogfood diagnostic, vocab-ON variant. Same six fixtures, same
    /// tagger config, N=10 — but each call is made with `vocab=Some(<frozen
    /// scope vocab>)` matching what the worker tick / `engram tag --rerun`
    /// would pass at runtime. Tests the hypothesis that scope-vocab injection
    /// is the lever causing the stored-vs-vocab-off kind divergence.
    ///
    /// Hypothesis: under vocab-on, each fixture's diagnostic kind matches its
    /// stored kind (i.e. vocab is the differentiator, not some unknown drift).
    /// Confirmation supports v3 adding an explicit kind-isolation clause to
    /// the prompt; refutation means a third mechanism is at play and v3 needs
    /// more investigation.
    ///
    /// Run with:
    /// `cargo test -p engram-extract --features integration --release -- kind_stability_diagnostic_with_vocab --nocapture --ignored`
    #[cfg(feature = "integration")]
    #[tokio::test]
    #[ignore]
    async fn kind_stability_diagnostic_with_vocab() {
        let fixtures = diagnostic_fixtures();
        const N_RUNS: usize = 10;

        let t = diagnostic_tagger();

        // (scope, descriptor, current_kind, observed_kinds, observed_entities).
        // v6 captures entities alongside kind so the operator can verify the
        // surface-only / adjectival regression cases in the same run.
        type Observed = (String, String, String, Vec<String>, Vec<Vec<String>>);
        let mut results: Vec<(String, Observed)> = Vec::new();

        for (short_id, scope, current_kind, descriptor, content) in &fixtures {
            let vocab = diagnostic_scope_vocab(scope);
            let mut observed_kinds: Vec<String> = Vec::with_capacity(N_RUNS);
            let mut observed_entities: Vec<Vec<String>> = Vec::with_capacity(N_RUNS);
            for run in 0..N_RUNS {
                eprintln!(
                    "[diagnostic-vocab] {short_id} ({scope}) run {}/{} ...",
                    run + 1,
                    N_RUNS
                );
                match t.tag(content, Some(&vocab)).await {
                    Ok(tags) => {
                        let k = tags
                            .kind
                            .map(|k| format!("{k:?}").to_lowercase())
                            .unwrap_or_else(|| "null".to_string());
                        observed_kinds.push(k);
                        observed_entities.push(tags.entities);
                    }
                    Err(e) => {
                        eprintln!("[diagnostic-vocab] {short_id} run {} ERR: {e}", run + 1);
                        observed_kinds.push(format!("ERR({e})"));
                        observed_entities.push(vec![]);
                    }
                }
            }
            results.push((
                short_id.to_string(),
                (
                    scope.to_string(),
                    descriptor.to_string(),
                    current_kind.to_string(),
                    observed_kinds,
                    observed_entities,
                ),
            ));
        }

        eprintln!();
        eprintln!(
            "## v{ver} kind-stability diagnostic results — VOCAB-ON (N={N_RUNS} per thought)",
            ver = BUNDLED_TAGGER_VERSION
        );
        eprintln!();
        eprintln!(
            "Tagger: ollama/qwen3-coder:30b @ http://localhost:11434/v1, bundled v{ver} prompt, temperature 0.2, vocab=Some(<frozen scope vocab>).",
            ver = BUNDLED_TAGGER_VERSION
        );
        eprintln!();
        eprintln!("| short_id | scope | stored kind | descriptor | kind distribution (N=10) |");
        eprintln!("|---|---|---|---|---|");
        for (short_id, (scope, descriptor, current_kind, kinds, _)) in &results {
            let mut counts: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for k in kinds {
                *counts.entry(k.as_str()).or_insert(0) += 1;
            }
            let dist = counts
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("| `{short_id}` | `{scope}` | `{current_kind}` | {descriptor} | {dist} |");
        }
        eprintln!();
        eprintln!("Raw kind observations (one row per fixture):");
        for (short_id, (_, _, _, kinds, _)) in &results {
            eprintln!("- {short_id}: {kinds:?}");
        }
        eprintln!();
        eprintln!("Entity emissions per fixture (one row per run):");
        for (short_id, (_, _, _, _, entities)) in &results {
            eprintln!("- {short_id}:");
            for (i, e) in entities.iter().enumerate() {
                eprintln!("    run {}: {e:?}", i + 1);
            }
        }
    }
}

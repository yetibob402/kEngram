//! Layered TOML + env config. Order (later wins):
//! 1. Hard-coded defaults
//! 2. `~/.config/kengram/kengram.toml` (if present)
//! 3. `--config <path>` (if passed)
//! 4. `KENGRAM_*` environment variables (nested via `__`, e.g. `KENGRAM_DATABASE__URL`)

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use kengram_extract::BUNDLED_TAGGER_VERSION;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// PathBuf is referenced from TaggerConfig::system_prompt_file below.

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub embedder: EmbedderConfig,
    pub worker: WorkerConfig,
    /// M4: renamed from `[extractor]`. The tagger is a metadata-tagging
    /// sidecar on the thoughts pipeline (people, action_items, topics,
    /// dates_mentioned, kind). Empty `provider` silent-disables — capture
    /// proceeds, no tag jobs enqueue, the worker doesn't spawn a tag
    /// drainer task. Flip `provider = "openai-compatible"` to enable.
    pub tagger: TaggerConfig,
    pub reranker: RerankerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind address. Tier 0 (localhost) is the M1 default. Tier 1 (Tailnet)
    /// is achieved by changing this to the Tailscale interface IP (or
    /// `0.0.0.0:<port>`) — no code change.
    pub bind: String,

    /// Host names / IPs the MCP server's DNS-rebinding protection accepts on
    /// the `Host` header. Empty = use rmcp's safe default
    /// (`localhost` / `127.0.0.1` / `::1`); a non-empty list REPLACES the
    /// default and is the operator-managed allowlist (the rmcp transport
    /// rejects any request whose `Host` header isn't in this list with a
    /// "rejected request with disallowed Host header" warning).
    ///
    /// When binding to a non-loopback interface (Tailnet, LAN), add the
    /// hostname AND `hostname:port` forms the client uses, plus the IP and
    /// `ip:port` forms. The rmcp matcher checks both:
    ///
    /// ```toml
    /// [server]
    /// bind = "0.0.0.0:8081"
    /// allowed_hosts = [
    ///     "localhost", "127.0.0.1", "::1",
    ///     "repromax", "repromax:8081",
    ///     "100.110.75.74", "100.110.75.74:8081",
    /// ]
    /// ```
    ///
    /// Leaving this list empty when bind is non-loopback effectively rejects
    /// every non-localhost request — the symptom is "rejected request"
    /// warnings in the serve log and connection failures from remote
    /// clients. Bypass-all (clearing the rmcp default entirely) is
    /// intentionally not exposed; if you need it, edit the source.
    pub allowed_hosts: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".to_string(),
            allowed_hosts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgres://kengram:kengram@localhost:5432/kengram".to_string(),
            max_connections: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedderConfig {
    /// `"openai-compatible"` is the only provider in M1; covers Ollama,
    /// TEI, OpenAI, and Voyage by varying `endpoint` and `model`.
    pub provider: String,
    /// `/v1` base URL. For Ollama: `"http://localhost:11434/v1"`.
    pub endpoint: String,
    /// Backend model name. For Ollama: `"bge-m3"`.
    pub model: String,
    /// Kengram-side model identity (`"bge-m3:1024"`). Must match the HNSW
    /// partial index in Postgres.
    pub model_id: String,
    pub dimensions: usize,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: "openai-compatible".to_string(),
            endpoint: "http://localhost:11434/v1".to_string(),
            model: "bge-m3".to_string(),
            model_id: "bge-m3:1024".to_string(),
            dimensions: 1024,
            api_key: None,
            timeout_seconds: 5,
        }
    }
}

/// Reranker configuration (M3 Phase B step 2). Empty `provider` disables
/// the rerank stage silently — the search pipeline falls through to the
/// Phase B step 1 RRF + recency pipeline. Currently the only supported
/// provider is `"tei"` (Hugging Face text-embeddings-inference in rerank
/// mode, default deployment is the `tei` service in docker-compose.yml).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// `""` (default) = no reranker, `"tei"` = TEI sidecar.
    pub provider: String,
    /// Service root (no `/v1` suffix). Default: `"http://localhost:8080"`.
    pub endpoint: String,
    /// Kengram-side stable identity. Conventionally `<vendor>/<model>`.
    pub model_id: String,
    pub timeout_seconds: u64,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            // Empty by default — opt-in. Phase B step 2 ships rerank-on-by-
            // default behavior *when configured*; not having a `[reranker]`
            // section in the TOML keeps the M1/M2/Phase-B-step-1 behavior.
            provider: String::new(),
            endpoint: "http://localhost:8080".to_string(),
            model_id: "BAAI/bge-reranker-v2-m3".to_string(),
            timeout_seconds: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerConfig {
    /// How often the embed-drainer wakes up and claims a batch off the
    /// `pending_embeddings` queue. 5s is fine for single-user dogfood; tune
    /// lower for snappier vector-search readiness, higher to be gentler on
    /// the embedder backend.
    pub tick_interval_seconds: u64,
    /// Max jobs claimed per tick. Bigger batches are kinder to the embedder
    /// (one HTTP call per batch instead of per row); smaller batches mean
    /// shorter critical sections and faster failover when a job hangs.
    pub batch_size: i64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            tick_interval_seconds: 5,
            batch_size: 16,
        }
    }
}

/// `[tagger]` — M4-replacement of the M3 `[extractor]` section. Same
/// provider/endpoint/model_name/model_id/temperature/timeout/api_key shape,
/// minus `max_facts_per_thought` (the tagger emits one tags object per
/// thought, no per-thought cap). `model_version` resets to `1` (the tagger
/// prompt is v1; separate version line from the M3 extractor's v4).
///
/// Empty `provider` is the silent-disable sentinel — matches `[reranker]`'s
/// pattern. `kengram serve`'s capture path checks the resolved tagger model
/// id; when `None`, no `pending_tags` row is enqueued and the worker's
/// tag-drainer task doesn't spawn. The operator can flip
/// `provider = "openai-compatible"` later and run
/// `kengram tag --rerun --since 1970-01-01T00:00:00Z` to catch the backlog.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaggerConfig {
    /// `""` (default) = disabled; `"openai-compatible"` (vLLM, etc.),
    /// `"openrouter"`, or `"http"` (kengram-native sidecar — see
    /// `[tagger.http]` below + `docs/tagger-backends.md`). Other providers
    /// can be added later by extending the `build_tagger` match.
    pub provider: String,
    /// `/v1` base URL. For vLLM: `"http://localhost:8000/v1"`. For
    /// OpenRouter: `"https://openrouter.ai/api/v1"`.
    pub endpoint: String,
    /// Backend model name. For vLLM: the deployed model (`"qwen2.5-7b-instruct"`).
    /// For OpenRouter: a model slug (`"anthropic/claude-haiku-4.5"`).
    pub model_name: String,
    /// Kengram-side stable identity written into `thoughts.tags_extractor_model`.
    /// Conventionally `<vendor>/<model>`. Defaults to `"vllm/qwen2.5-7b-instruct"`.
    pub model_id: String,
    /// Schema-version for `thoughts.tags_extractor_version`. Default tracks
    /// `kengram_extract::BUNDLED_TAGGER_VERSION` (currently 13 — seven
    /// post-M6.1 dogfood iterations: v6 rebalanced kind classification +
    /// added entity surface-only rule + tightened URL emission but
    /// repeated the v3→v4 backfire by listing adjectival phrases as
    /// negative examples; v7 drops the literal-phrase NOT-entities list,
    /// relying on the structural NAME-vs-DESCRIBE test alone, and
    /// documents topics-as-concept-mapping intent explicitly; v8 removes
    /// Rust from the topics examples list and from the kind=observation
    /// exemplar after first-item example-list priming caused topic-
    /// overreach to `"rust"` on tech-adjacent thoughts; v9 drops the
    /// topics `Examples: ...` clause entirely after v8's swap just
    /// rotated the priming target from `"rust"` to `"databases"`; v10
    /// was an ephemeral toml-override version used during the 2026-05-22
    /// scope_vocab experiment, never shipped to source; v11 moves topic
    /// canonical-form convergence out of the prompt entirely — topic
    /// vocab now feeds a post-process normalization step in
    /// `kengram-mcp::drain`, breaking the prompt-vocab feedback loop that
    /// caused the v8/v9 corpus-wide topic overreach; v12 adds a positive
    /// syntactic-disambiguation rule for verb-vs-name ambiguity at
    /// sentence-start, plus a post-process disjointness validator in
    /// `kengram-mcp::drain` that strips entities-array entries that
    /// duplicate a people-array entry; v13 adds use-mention discipline
    /// to the prompt — a top-level USE-vs-MENTION section plus 6 worked
    /// examples — fixing the corpus-wide pollution of meta-content
    /// thoughts that mention names as linguistic examples). Bump when
    /// the prompt or schema changes such that prior tags shouldn't be
    /// considered comparable; `kengram tag --rerun --since
    /// 1970-01-01T00:00:00Z` then backfills.
    pub model_version: i32,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub temperature: f32,
    /// Path to a file containing the tagger system prompt. `None` means
    /// use `kengram_extract::BUNDLED_TAGGER_PROMPT` (recommended). Operators
    /// who supply a custom prompt are responsible for also bumping
    /// `model_version` so `thoughts.tags_extractor_version` remains
    /// meaningful provenance.
    pub system_prompt_file: Option<PathBuf>,
    /// When true, the tagger receives a controlled-vocabulary hint section
    /// listing the top topic and entity terms already used in the thought's
    /// scope. Encourages the model to prefer established terms over coining
    /// new ones — addresses the v1 corpus-coherence finding. Default true.
    pub scope_vocab_enabled: bool,
    /// Maximum number of established topic terms and entity terms (each)
    /// fed into the tagger's controlled-vocabulary section. Larger values
    /// give the model more context but cost prompt tokens; smaller values
    /// let new terms emerge faster. Default 50.
    pub scope_vocab_size: u32,
    /// Sub-section for the HTTP-sidecar backend (`provider = "http"`).
    /// `None` when that backend isn't selected; the `[tagger.http]` toml
    /// block deserializes into `Some(...)`. The flat fields above remain
    /// the openai-compatible backend's home — no operator's existing
    /// `kengram.toml` needs to change to keep their LLM tagging working.
    pub http: Option<HttpTaggerConfig>,
}

/// `[tagger.http]` — configuration for the kengram-native HTTP-tagger
/// sidecar backend (`provider = "http"`). The sidecar speaks the
/// `kengram-tagger-protocol` wire shape; kengram POSTs `/tag` with the
/// thought content and parses the response as `Tags + relations`. See
/// `docs/tagger-backends.md` and `docs/tagger-sidecar-protocol.md` for
/// the wire contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpTaggerConfig {
    /// Base URL of the sidecar. The client appends `/tag` to this.
    /// Example: `"http://localhost:8082"`.
    pub endpoint: String,
    /// Optional bearer token sent as `Authorization: Bearer <token>`.
    /// `None` means no Authorization header (sidecars on a private
    /// network are the common case).
    pub api_key: Option<String>,
    /// Per-request timeout in seconds. Defaults to 60s to match the
    /// openai-compatible backend's default (sidecars doing CPU
    /// inference can run long on first call).
    pub timeout_seconds: u64,
}

impl Default for HttpTaggerConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8082".to_string(),
            api_key: None,
            timeout_seconds: 60,
        }
    }
}

impl Default for TaggerConfig {
    fn default() -> Self {
        Self {
            // Empty default → silent-disable. The operator must opt in by
            // setting provider = "openai-compatible" (or "openrouter").
            provider: String::new(),
            endpoint: "http://localhost:8000/v1".to_string(),
            model_name: "qwen2.5-7b-instruct".to_string(),
            model_id: "vllm/qwen2.5-7b-instruct".to_string(),
            // Track the bundled const so a prompt-version bump in
            // kengram-extract propagates without a hand-edit here. The
            // version history lives on `BUNDLED_TAGGER_VERSION` itself.
            model_version: BUNDLED_TAGGER_VERSION,
            api_key: None,
            timeout_seconds: 60,
            temperature: 0.2,
            system_prompt_file: None,
            scope_vocab_enabled: true,
            scope_vocab_size: 50,
            http: None,
        }
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/kengram/kengram.toml"))
}

pub fn load(cli_config: Option<&Path>) -> anyhow::Result<Config> {
    let mut figment = Figment::from(Serialized::defaults(Config::default()));

    if let Some(path) = default_config_path()
        && path.exists()
    {
        figment = figment.merge(Toml::file(path));
    }

    if let Some(path) = cli_config {
        figment = figment.merge(Toml::file(path));
    }

    figment = figment.merge(Env::prefixed("KENGRAM_").split("__"));

    Ok(figment.extract()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_localhost_bind() {
        let c = Config::default();
        assert_eq!(c.server.bind, "127.0.0.1:8080");
        // Empty allowed_hosts = use rmcp's safe default (localhost-only).
        // Operator must extend this list when binding non-loopback.
        assert!(c.server.allowed_hosts.is_empty());
    }

    /// Operator-provided allowed_hosts round-trips through figment. Common
    /// usage when binding to a Tailnet interface or 0.0.0.0 — include both
    /// bare-hostname and hostname:port forms since the rmcp matcher
    /// distinguishes them.
    #[test]
    fn server_allowed_hosts_round_trips_from_toml() {
        let toml = r#"
            [server]
            bind = "0.0.0.0:8081"
            allowed_hosts = ["localhost", "127.0.0.1", "::1", "repromax", "repromax:8081"]
        "#;
        let c: Config = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::string(toml))
            .extract()
            .unwrap();
        assert_eq!(c.server.bind, "0.0.0.0:8081");
        assert_eq!(c.server.allowed_hosts.len(), 5);
        assert!(c.server.allowed_hosts.contains(&"repromax".to_string()));
        assert!(
            c.server
                .allowed_hosts
                .contains(&"repromax:8081".to_string())
        );
    }

    #[test]
    fn default_config_uses_dev_postgres_url() {
        let c = Config::default();
        assert!(c.database.url.contains("localhost:5432"));
        assert_eq!(c.database.max_connections, 10);
    }

    #[test]
    fn default_embedder_targets_ollama_bge_m3() {
        let c = Config::default();
        assert_eq!(c.embedder.provider, "openai-compatible");
        assert_eq!(c.embedder.endpoint, "http://localhost:11434/v1");
        assert_eq!(c.embedder.model, "bge-m3");
        assert_eq!(c.embedder.model_id, "bge-m3:1024");
        assert_eq!(c.embedder.dimensions, 1024);
        assert_eq!(c.embedder.timeout_seconds, 5);
    }

    #[test]
    fn default_worker_uses_5s_tick_and_batch_16() {
        let c = Config::default();
        assert_eq!(c.worker.tick_interval_seconds, 5);
        assert_eq!(c.worker.batch_size, 16);
    }

    /// M4: tagger defaults to silent-disabled (empty `provider`). The
    /// operator opts in by setting `provider = "openai-compatible"` (or
    /// `"openrouter"`) in `[tagger]`. Other fields default to a local-vLLM
    /// shape so flipping `provider` is the only required change on a stock
    /// dev box with vLLM running on 8000.
    #[test]
    fn default_tagger_is_silent_disabled() {
        let c = Config::default();
        assert_eq!(c.tagger.provider, "");
        assert_eq!(c.tagger.endpoint, "http://localhost:8000/v1");
        assert_eq!(c.tagger.model_name, "qwen2.5-7b-instruct");
        assert_eq!(c.tagger.model_id, "vllm/qwen2.5-7b-instruct");
        assert_eq!(c.tagger.model_version, BUNDLED_TAGGER_VERSION);
        assert!(c.tagger.api_key.is_none());
        // Default is the bundled prompt — no file override.
        assert!(c.tagger.system_prompt_file.is_none());
        // Scope vocabulary injection is enabled by default at 50 terms each.
        assert!(c.tagger.scope_vocab_enabled);
        assert_eq!(c.tagger.scope_vocab_size, 50);
        // The `http` sub-section is opt-in. Absent `[tagger.http]` block
        // → None; back-compat for every existing kengram.toml.
        assert!(c.tagger.http.is_none());
    }

    /// Operator opt-in: setting `[tagger.http]` round-trips through
    /// figment and the new HttpTaggerConfig values surface as `Some(...)`
    /// on `TaggerConfig.http`.
    #[test]
    fn tagger_http_subsection_round_trips_from_toml() {
        let toml = r#"
            [tagger]
            provider = "http"
            model_id = "myorg/my-sidecar-v1"
            model_version = 1

            [tagger.http]
            endpoint = "http://localhost:8082"
            timeout_seconds = 30
        "#;
        let c: Config = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::string(toml))
            .extract()
            .unwrap();
        assert_eq!(c.tagger.provider, "http");
        assert_eq!(c.tagger.model_id, "myorg/my-sidecar-v1");
        let http = c
            .tagger
            .http
            .as_ref()
            .expect("http sub-section should be Some after toml override");
        assert_eq!(http.endpoint, "http://localhost:8082");
        assert_eq!(http.timeout_seconds, 30);
        assert!(http.api_key.is_none());
    }

    /// Operator opt-in: setting `[tagger].provider = "openai-compatible"`
    /// round-trips through figment without disturbing the other defaults.
    #[test]
    fn tagger_provider_round_trips_from_toml() {
        let toml = r#"
            [tagger]
            provider = "openai-compatible"
        "#;
        let c: Config = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::string(toml))
            .extract()
            .unwrap();
        assert_eq!(c.tagger.provider, "openai-compatible");
        assert_eq!(c.tagger.endpoint, "http://localhost:8000/v1");
        assert_eq!(c.tagger.model_version, BUNDLED_TAGGER_VERSION);
    }

    /// Operator can disable scope-vocabulary injection or tune its size via
    /// TOML — both knobs round-trip cleanly through figment.
    #[test]
    fn tagger_scope_vocab_overrides_round_trip_from_toml() {
        let toml = r#"
            [tagger]
            provider = "openai-compatible"
            scope_vocab_enabled = false
            scope_vocab_size = 20
        "#;
        let c: Config = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::string(toml))
            .extract()
            .unwrap();
        assert!(!c.tagger.scope_vocab_enabled);
        assert_eq!(c.tagger.scope_vocab_size, 20);
    }
}

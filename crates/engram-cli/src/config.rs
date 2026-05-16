//! Layered TOML + env config. Order (later wins):
//! 1. Hard-coded defaults
//! 2. `~/.config/engram/engram.toml` (if present)
//! 3. `--config <path>` (if passed)
//! 4. `ENGRAM_*` environment variables (nested via `__`, e.g. `ENGRAM_DATABASE__URL`)

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// PathBuf is referenced from ExtractorConfig::system_prompt_file below.

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub embedder: EmbedderConfig,
    pub worker: WorkerConfig,
    pub extractor: ExtractorConfig,
    pub reflector: engram_mcp::ReflectorOptions,
    pub reranker: RerankerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind address. Tier 0 (localhost) is the M1 default. Tier 1 (Tailnet)
    /// is achieved by changing this to the Tailscale interface IP — no code
    /// change.
    pub bind: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".to_string(),
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
            url: "postgres://engram:engram@localhost:5432/engram".to_string(),
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
    /// Engram-side model identity (`"bge-m3:1024"`). Must match the HNSW
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
    /// Engram-side stable identity. Conventionally `<vendor>/<model>`.
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractorConfig {
    /// `"openai-compatible"` (vLLM, etc.) or `"openrouter"`. Other providers
    /// can be added later by extending the `build_extractor` match.
    pub provider: String,
    /// `/v1` base URL. For vLLM: `"http://localhost:8000/v1"`. For
    /// OpenRouter: `"https://openrouter.ai/api/v1"`.
    pub endpoint: String,
    /// Backend model name. For vLLM: the deployed model (`"qwen2.5-7b-instruct"`).
    /// For OpenRouter: a model slug (`"anthropic/claude-haiku-4.5"`).
    pub model_name: String,
    /// Engram-side stable identity written into `facts.extractor_model`.
    /// Conventionally `<vendor>/<model>`. Defaults to `"vllm/qwen2.5-7b-instruct"`.
    pub model_id: String,
    /// Schema-version for `facts.extractor_version`. Bump when the prompt
    /// or schema changes such that prior facts are no longer comparable.
    pub model_version: i32,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub temperature: f32,
    pub max_facts_per_thought: usize,
    /// Path to a file containing the extractor system prompt. `None` means
    /// use `engram_extract::openai_compatible::BUNDLED_SYSTEM_PROMPT`
    /// (recommended). Operators who supply a custom prompt are responsible
    /// for also bumping `model_version` so `facts.extractor_version`
    /// remains meaningful provenance. The file's contents must contain the
    /// `{MAX_FACTS}` placeholder; the extractor refuses to construct
    /// otherwise.
    pub system_prompt_file: Option<PathBuf>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            provider: "openai-compatible".to_string(),
            endpoint: "http://localhost:8000/v1".to_string(),
            model_name: "qwen2.5-7b-instruct".to_string(),
            model_id: "vllm/qwen2.5-7b-instruct".to_string(),
            // v4 = relations rule + reinforced SPO few-shots for leak set
            // (Bazel/Make, Nix/Make, Redis conditional, SIMD self-reference)
            // + flagged-band confidence framing + "no same-statement-different-
            // SPO" rule (2026-05-15, M3 Phase C). See
            // crates/engram-extract/src/openai_compatible.rs.
            model_version: 4,
            api_key: None,
            timeout_seconds: 60,
            temperature: 0.2,
            max_facts_per_thought: 8,
            system_prompt_file: None,
        }
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/engram/engram.toml"))
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

    figment = figment.merge(Env::prefixed("ENGRAM_").split("__"));

    Ok(figment.extract()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_localhost_bind() {
        let c = Config::default();
        assert_eq!(c.server.bind, "127.0.0.1:8080");
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

    #[test]
    fn default_extractor_targets_vllm_localhost() {
        let c = Config::default();
        assert_eq!(c.extractor.provider, "openai-compatible");
        assert_eq!(c.extractor.endpoint, "http://localhost:8000/v1");
        assert_eq!(c.extractor.model_name, "qwen2.5-7b-instruct");
        assert_eq!(c.extractor.model_id, "vllm/qwen2.5-7b-instruct");
        // Bumped to 4 on 2026-05-15 (M3 Phase C) when the system prompt
        // gained the relations rule, reinforced SPO few-shots for the
        // Phase A + Phase B step 2 leak set, and flagged-band confidence
        // framing. (v3 = 2026-05-14, M3 Phase A: SPO decomposition rules
        // + tighter confidence rubric + episodic-skip negatives.)
        assert_eq!(c.extractor.model_version, 4);
        assert!(c.extractor.api_key.is_none());
        assert_eq!(c.extractor.max_facts_per_thought, 8);
        // Default is the bundled prompt — no file override.
        assert!(c.extractor.system_prompt_file.is_none());
    }

    #[test]
    fn default_reflector_is_disabled() {
        let c = Config::default();
        assert!(!c.reflector.enabled, "reflector must default to off — opt-in");
    }

    #[test]
    fn default_reflector_schedule_is_3am_daily() {
        let c = Config::default();
        assert_eq!(c.reflector.schedule, "0 0 3 * * *");
    }

    #[test]
    fn default_reflector_review_queue_below_is_0_7() {
        let c = Config::default();
        assert!((c.reflector.review_queue_below - 0.7).abs() < f32::EPSILON);
    }

    /// M3 Phase C: middle confidence band defaults to 0.85, matching the
    /// design-doc §10 framing. Subsumption-keep defaults to `Specific`.
    #[test]
    fn reflector_config_loads_min_confidence_and_subsumption_keep() {
        // Defaults.
        let c = Config::default();
        assert!((c.reflector.min_confidence_to_store - 0.85).abs() < f32::EPSILON);
        assert_eq!(
            c.reflector.subsumption_keep,
            engram_mcp::SubsumptionKeep::Specific,
        );

        // TOML override round-trips both fields.
        let toml = r#"
            [reflector]
            min_confidence_to_store = 0.9
            subsumption_keep = "general"
        "#;
        let c: Config = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::string(toml))
            .extract()
            .unwrap();
        assert!((c.reflector.min_confidence_to_store - 0.9).abs() < f32::EPSILON);
        assert_eq!(
            c.reflector.subsumption_keep,
            engram_mcp::SubsumptionKeep::General,
        );
    }
}

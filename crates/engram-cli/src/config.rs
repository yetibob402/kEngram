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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub embedder: EmbedderConfig,
    pub worker: WorkerConfig,
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
}

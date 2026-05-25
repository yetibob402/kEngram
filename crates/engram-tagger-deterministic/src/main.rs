//! Reference sidecar binary — exposes the deterministic tagger pipeline
//! over the `engram-tagger-protocol` HTTP wire contract.
//!
//! Operators run this binary either standalone or inside a Docker
//! container, then point engram at it via `provider = "http"` in
//! `[tagger]`. The binary's only inputs are env vars (see below); no
//! TOML config file. Twelve-factor.
//!
//! Env vars:
//!   BIND_ADDR             — default 0.0.0.0:8082 — listen address
//!   EMBEDDER_ENDPOINT     — default http://localhost:11434/v1
//!   EMBEDDER_MODEL        — default bge-m3
//!   EMBEDDER_API_KEY      — optional, sent as Bearer to the embedder
//!   GLINER_MODEL_DIR      — default $HOME/models/gliner_small-v2.1
//!                           (must contain tokenizer.json + onnx/model.onnx)
//!   TOPIC_TAXONOMY_PATH   — default topic-taxonomy.toml (relative to CWD)
//!   KIND_THRESHOLD        — default 0.55 — minimum cosine for kind argmax
//!   TOPIC_THRESHOLD       — default 0.45 — default threshold for topics
//!   MODEL_ID              — default deterministic/gliner-small-v2.1+regex+bge-m3
//!                           (advertised back to engram; stamped on tag rows)
//!   MODEL_VERSION         — default 1 — schema-version line on tag rows
//!   RUST_LOG              — standard tracing env-filter; default "info"
//!
//! On startup the binary:
//! 1. Builds the embedder HTTP client.
//! 2. Loads the gline-rs ONNX model (~5-15s first time).
//! 3. Embeds the topic taxonomy + 6 kind prototypes.
//! 4. Binds + serves `POST /tag` and `GET /health`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use engram_core::{Embedder, EmbeddingModel, Tagger};
use engram_embed::{OpenAICompatibleConfig, OpenAICompatibleEmbedder};
use engram_tagger_deterministic::{DeterministicTagger, DeterministicTaggerConfig};
use engram_tagger_protocol::{PROTOCOL_VERSION, TagRequest, TagResponse};
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct AppState {
    tagger: Arc<DeterministicTagger>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let bind_addr: SocketAddr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8082".to_string())
        .parse()
        .context("parsing BIND_ADDR")?;

    let embedder = build_embedder()?;
    let tagger_config = tagger_config_from_env();
    tracing::info!(
        bind_addr = %bind_addr,
        embedder_endpoint = %std::env::var("EMBEDDER_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
        gliner_model_dir = %tagger_config.gliner_model_dir.display(),
        topic_taxonomy_path = %tagger_config.topic_taxonomy_path.display(),
        kind_threshold = tagger_config.kind_threshold,
        topic_threshold_default = tagger_config.topic_threshold_default,
        model_id = %tagger_config.model_id,
        model_version = tagger_config.model_version,
        "engram-tagger-deterministic: resolved config",
    );

    tracing::info!(
        "loading gline-rs model + embedding taxonomy + kind prototypes (one-time startup cost)"
    );
    let tagger = DeterministicTagger::new(tagger_config, embedder)
        .await
        .context("constructing DeterministicTagger")?;
    let state = AppState {
        tagger: Arc::new(tagger),
    };
    tracing::info!("startup complete; serving");

    let app = Router::new()
        .route("/tag", post(tag_handler))
        .route("/health", get(health_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("binding {bind_addr}"))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve")?;
    tracing::info!("graceful shutdown complete");
    Ok(())
}

async fn tag_handler(
    State(state): State<AppState>,
    Json(req): Json<TagRequest>,
) -> Result<Json<TagResponse>, ApiError> {
    if req.protocol_version != PROTOCOL_VERSION {
        return Err(ApiError::BadRequest(format!(
            "unsupported protocol_version {:?}; this sidecar speaks {:?}",
            req.protocol_version, PROTOCOL_VERSION,
        )));
    }
    let vocab = req.vocab.map(engram_core::ScopeVocab::from);
    let out = state
        .tagger
        .tag(&req.content, vocab.as_ref())
        .await
        .map_err(ApiError::from)?;
    Ok(Json(TagResponse {
        protocol_version: Some(PROTOCOL_VERSION.to_string()),
        tags: out.tags,
        relations: out.relations,
    }))
}

async fn health_handler() -> impl IntoResponse {
    // Liveness only — the model + taxonomy already loaded successfully
    // during startup, so by the time we're serving the binary is ready.
    // For finer-grained readiness an operator can probe /tag with a
    // minimal payload.
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

fn build_embedder() -> anyhow::Result<Arc<dyn Embedder>> {
    let endpoint = std::env::var("EMBEDDER_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model_name = std::env::var("EMBEDDER_MODEL").unwrap_or_else(|_| "bge-m3".to_string());
    let api_key = std::env::var("EMBEDDER_API_KEY").ok();
    let cfg = OpenAICompatibleConfig {
        endpoint,
        model_name,
        model: EmbeddingModel::bge_m3(),
        api_key,
        timeout: std::time::Duration::from_secs(60),
    };
    let embedder =
        OpenAICompatibleEmbedder::new(cfg).context("constructing OpenAICompatibleEmbedder")?;
    Ok(Arc::new(embedder))
}

fn tagger_config_from_env() -> DeterministicTaggerConfig {
    let mut cfg = DeterministicTaggerConfig::default();
    if let Ok(p) = std::env::var("GLINER_MODEL_DIR") {
        cfg.gliner_model_dir = PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("TOPIC_TAXONOMY_PATH") {
        cfg.topic_taxonomy_path = PathBuf::from(p);
    }
    if let Some(v) = std::env::var("KIND_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        cfg.kind_threshold = v;
    }
    if let Some(v) = std::env::var("TOPIC_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        cfg.topic_threshold_default = v;
    }
    if let Ok(v) = std::env::var("MODEL_ID") {
        cfg.model_id = v;
    }
    if let Some(v) = std::env::var("MODEL_VERSION")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
    {
        cfg.model_version = v;
    }
    cfg
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install ctrl-c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received; draining");
}

/// HTTP error envelope. Maps TaggerError to status codes that match the
/// transient-vs-non-transient classification engram's drainer relies on.
#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    /// Maps to 5xx — engram's drainer will treat as transient and retry.
    Transient(String),
    /// Maps to 4xx (non-500 non-2xx) — engram will treat as non-transient.
    NonTransient(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, body) = match self {
            Self::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            Self::NonTransient(m) => (StatusCode::UNPROCESSABLE_ENTITY, m),
            Self::Transient(m) => (StatusCode::SERVICE_UNAVAILABLE, m),
        };
        (status, Json(serde_json::json!({"error": body}))).into_response()
    }
}

impl From<engram_core::TaggerError> for ApiError {
    fn from(e: engram_core::TaggerError) -> Self {
        if e.is_transient() {
            Self::Transient(e.to_string())
        } else {
            Self::NonTransient(e.to_string())
        }
    }
}

//! engram — the only binary. Phase B wires up the `serve` and `migrate`
//! subcommands; `embed-backfill` lands in Phase D.

mod config;

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::{Parser, Subcommand};
use engram_core::{Embedder, EmbeddingModel};
use engram_embed::{OpenAICompatibleConfig, OpenAICompatibleEmbedder};
use engram_mcp::EngramServer;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use sqlx::postgres::PgPoolOptions;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, EmbedderConfig, WorkerConfig};

#[derive(Parser, Debug)]
#[command(name = "engram", version, about = "Self-hosted MCP-native memory service")]
struct Cli {
    /// Path to an `engram.toml` config file. Overrides `~/.config/engram/engram.toml`.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the MCP/HTTP server.
    Serve,
    /// Apply pending database migrations.
    Migrate,
    /// Long-running worker process: drains the `pending_embeddings` queue.
    /// (M2 Phase C adds the reflector cron as a second task in the same
    /// process.) Knobs live in `[worker]` config — no CLI flags needed.
    Worker,
    /// Embed thoughts that don't yet have an embedding row for the active model.
    EmbedBackfill {
        /// Restrict to a single scope.
        #[arg(long)]
        scope: Option<String>,
        /// Maximum number of thoughts to embed in this run. Defaults to 1000.
        #[arg(long, default_value_t = 1000)]
        limit: i64,
    },
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,engram=debug"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn build_embedder(c: &EmbedderConfig) -> anyhow::Result<Arc<dyn Embedder>> {
    match c.provider.as_str() {
        "openai-compatible" => {
            let embedder = OpenAICompatibleEmbedder::new(OpenAICompatibleConfig {
                endpoint: c.endpoint.clone(),
                model_name: c.model.clone(),
                model: EmbeddingModel::new(c.model_id.clone(), c.dimensions),
                api_key: c.api_key.clone(),
                timeout: Duration::from_secs(c.timeout_seconds),
            })
            .with_context(|| format!("constructing embedder for endpoint {}", c.endpoint))?;
            Ok(Arc::new(embedder))
        }
        other => anyhow::bail!(
            "unknown embedder provider: {other:?} (valid: 'openai-compatible')"
        ),
    }
}

async fn run_serve(config: Config) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    let embedder = build_embedder(&config.embedder)?;

    let bind: SocketAddr = config
        .server
        .bind
        .parse()
        .with_context(|| format!("parsing server.bind = {:?}", config.server.bind))?;

    let pool_for_factory = pool.clone();
    let embedder_for_factory = embedder.clone();
    let factory = move || Ok(EngramServer::new(pool_for_factory.clone(), embedder_for_factory.clone()));

    let cancel = CancellationToken::new();
    let mcp_service = StreamableHttpService::new(
        factory,
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let app = axum::Router::new().nest_service("/mcp", mcp_service);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding HTTP server to {bind}"))?;

    tracing::info!(
        bind = %bind,
        embedder_endpoint = %config.embedder.endpoint,
        model_id = %config.embedder.model_id,
        "engram serve started"
    );

    let shutdown = {
        let cancel = cancel.clone();
        async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("shutdown signal received");
                }
                _ = cancel.cancelled() => {
                    tracing::info!("cancellation token tripped");
                }
            }
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("axum::serve")?;

    Ok(())
}

async fn run_migrate(config: Config) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    sqlx::migrate!("../../migrations")
        .run(&pool)
        .await
        .context("running migrations")?;

    tracing::info!("migrations applied");
    Ok(())
}

async fn run_worker(config: Config) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;
    let embedder = build_embedder(&config.embedder)?;

    let cancel = CancellationToken::new();
    let mut set = tokio::task::JoinSet::new();

    let drain_pool = pool.clone();
    let drain_embedder = embedder.clone();
    let drain_cancel = cancel.clone();
    let WorkerConfig {
        tick_interval_seconds,
        batch_size,
    } = config.worker;
    let interval = Duration::from_secs(tick_interval_seconds);
    set.spawn(async move {
        embed_drainer_loop(drain_pool, drain_embedder, interval, batch_size, drain_cancel).await;
    });

    tracing::info!(
        tick_interval_seconds,
        batch_size,
        embedder_endpoint = %config.embedder.endpoint,
        model_id = %config.embedder.model_id,
        "engram worker started"
    );

    // Wait for ctrl-c, then signal the loop(s) to wind down and join with a
    // 30s ceiling so a hung embed call can't block shutdown forever.
    tokio::signal::ctrl_c()
        .await
        .context("waiting for shutdown signal")?;
    tracing::info!("shutdown signal received");
    cancel.cancel();

    let shutdown_deadline = tokio::time::sleep(Duration::from_secs(30));
    tokio::select! {
        _ = async { while set.join_next().await.is_some() {} } => {
            tracing::info!("worker tasks exited cleanly");
        }
        _ = shutdown_deadline => {
            tracing::warn!("worker tasks did not exit within 30s; forcing exit");
            set.abort_all();
        }
    }
    Ok(())
}

async fn embed_drainer_loop(
    pool: sqlx::PgPool,
    embedder: std::sync::Arc<dyn Embedder>,
    interval: Duration,
    batch_size: i64,
    cancel: CancellationToken,
) {
    let mut ticker = tokio::time::interval(interval);
    // Skip the eager first tick that `interval` fires immediately so we wait
    // a full interval before the first drain.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    ticker.tick().await;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::info!("embed drainer shutting down");
                return;
            }
            _ = ticker.tick() => {
                match engram_mcp::drain_pending_embeddings(&pool, embedder.as_ref(), batch_size).await {
                    Ok(report) if report.found > 0 => tracing::info!(
                        found = report.found,
                        embedded = report.embedded,
                        failed = report.failed,
                        "embed drain tick",
                    ),
                    Ok(_) => {} // idle tick; stay quiet
                    Err(err) => tracing::error!(error = ?err, "embed drain tick failed"),
                }
            }
        }
    }
}

async fn run_embed_backfill(
    config: Config,
    scope: Option<String>,
    limit: i64,
) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    let embedder = build_embedder(&config.embedder)?;

    let report =
        engram_mcp::embed_backfill(&pool, embedder.as_ref(), scope.as_deref(), limit).await?;

    tracing::info!(
        healed = report.healed,
        embedded = report.embedded,
        failed = report.failed,
        "backfill complete"
    );

    if report.failed > 0 {
        // Non-zero exit so scripts/cron can detect partial failures.
        anyhow::bail!(
            "{} thoughts failed to embed (see logs); {} succeeded",
            report.failed,
            report.embedded
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let config = config::load(cli.config.as_deref()).context("loading config")?;

    match cli.command {
        Command::Serve => run_serve(config).await,
        Command::Migrate => run_migrate(config).await,
        Command::Worker => run_worker(config).await,
        Command::EmbedBackfill { scope, limit } => run_embed_backfill(config, scope, limit).await,
    }
}

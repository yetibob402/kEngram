//! engram — the only binary. Phase B wires up the `serve` and `migrate`
//! subcommands; `embed-backfill` lands in Phase D.

mod config;

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::{Parser, Subcommand};
use engram_core::{Embedder, EmbeddingModel, Extractor};
use engram_embed::{OpenAICompatibleConfig, OpenAICompatibleEmbedder};
use engram_extract::{
    OpenAICompatibleConfig as ExtractorConfigBuilder, OpenAICompatibleExtractor,
};
use engram_mcp::{EngramServer, ReflectorOptions};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, StreamableHttpServerConfig, session::local::LocalSessionManager,
};
use sqlx::postgres::PgPoolOptions;
use tokio_cron_scheduler::{Job, JobScheduler};
use tokio_util::sync::CancellationToken;

use crate::config::{Config, EmbedderConfig, ExtractorConfig, WorkerConfig};

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
    /// One-shot reflector run. By default acts on unfacted thoughts (same as
    /// the worker's cron tick). With --rerun, re-evaluates already-facted
    /// thoughts and supersedes obsolete extractions (preserving the audit
    /// trail) — useful after upgrading the extractor model or schema.
    Reflect {
        /// Restrict to a single scope. Overrides `[reflector] scope_filter`.
        #[arg(long)]
        scope: Option<String>,
        /// Max thoughts to process. Overrides `[reflector] max_thoughts_per_run`.
        #[arg(long)]
        limit: Option<i64>,
        /// Re-evaluate already-facted thoughts. Pairs naturally with --since.
        #[arg(long)]
        rerun: bool,
        /// With --rerun, only re-evaluate thoughts created at or after this
        /// RFC-3339 timestamp. Rejected without --rerun.
        #[arg(long)]
        since: Option<String>,
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

fn build_extractor(c: &ExtractorConfig) -> anyhow::Result<Arc<dyn Extractor>> {
    // Resolve the system prompt: bundled by default; load from a file when
    // `system_prompt_file` is set. The path is anyhow-context'd so errors
    // surface with the path the operator typed.
    let system_prompt = match c.system_prompt_file.as_ref() {
        Some(path) => Some(std::fs::read_to_string(path).with_context(|| {
            format!("reading extractor system_prompt_file at {}", path.display())
        })?),
        None => None,
    };
    tracing::info!(
        system_prompt = %match c.system_prompt_file.as_ref() {
            Some(p) => format!("file:{}", p.display()),
            None => "bundled".to_string(),
        },
        "extractor: resolved system prompt",
    );

    match c.provider.as_str() {
        "openai-compatible" | "openrouter" => {
            let extractor = OpenAICompatibleExtractor::new(ExtractorConfigBuilder {
                endpoint: c.endpoint.clone(),
                model_name: c.model_name.clone(),
                model_id: c.model_id.clone(),
                model_version: c.model_version,
                api_key: c.api_key.clone(),
                timeout: Duration::from_secs(c.timeout_seconds),
                temperature: c.temperature,
                max_facts_per_thought: c.max_facts_per_thought,
                system_prompt,
            })
            .with_context(|| format!("constructing extractor for endpoint {}", c.endpoint))?;
            Ok(Arc::new(extractor))
        }
        other => anyhow::bail!(
            "unknown extractor provider: {other:?} (valid: 'openai-compatible', 'openrouter')"
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
    // Stateless mode (allowed by MCP Streamable HTTP spec 2025-06-18 for
    // simple request-response tools). Engram has no per-session state — every
    // tool is `Result<Response, Error>`, returns synchronously, doesn't push
    // events. Disabling stateful mode means rmcp never issues a session id,
    // never runs the idle-session reaper, and therefore can't return
    // `Session not found` 404s when a long-lived MCP client (Claude Desktop,
    // mcp-remote bridge) comes back after idling past the 5-minute default.
    // `json_response: true` pairs naturally — replies are plain JSON, no SSE
    // framing overhead.
    let mcp_service = StreamableHttpService::new(
        factory,
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default()
            .with_stateful_mode(false)
            .with_json_response(true),
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

    // Reflector task is opt-in. Build the extractor only when enabled so
    // `engram worker` with `reflector.enabled = false` doesn't require vLLM
    // (or even an `[extractor]` block validating cleanly).
    let reflector_enabled = config.reflector.enabled;
    let reflector_summary = if reflector_enabled {
        let extractor = build_extractor(&config.extractor)?;
        let reflector_pool = pool.clone();
        let reflector_cancel = cancel.clone();
        let reflector_options = config.reflector.clone();
        let schedule = reflector_options.schedule.clone();
        let model_id = extractor.model_id().to_string();
        set.spawn(async move {
            if let Err(err) =
                reflector_loop(reflector_pool, extractor, reflector_options, reflector_cancel)
                    .await
            {
                tracing::error!(error = ?err, "reflector loop exited with error");
            }
        });
        format!("reflector enabled (schedule={schedule:?}, model={model_id})")
    } else {
        "reflector disabled".to_string()
    };

    tracing::info!(
        tick_interval_seconds,
        batch_size,
        embedder_endpoint = %config.embedder.endpoint,
        model_id = %config.embedder.model_id,
        reflector = %reflector_summary,
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

async fn reflector_loop(
    pool: sqlx::PgPool,
    extractor: Arc<dyn Extractor>,
    options: ReflectorOptions,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let mut sched = JobScheduler::new()
        .await
        .context("constructing JobScheduler")?;

    let pool_for_job = pool.clone();
    let extractor_for_job = extractor.clone();
    let options_for_job = options.clone();
    let job = Job::new_async(options.schedule.as_str(), move |_uuid, _l| {
        let pool = pool_for_job.clone();
        let extractor = extractor_for_job.clone();
        let options = options_for_job.clone();
        Box::pin(async move {
            match engram_mcp::run_reflector_once(&pool, extractor.as_ref(), &options).await {
                Ok(r) => tracing::info!(
                    run_id = %r.run_id,
                    processed = r.n_thoughts_processed,
                    committed = r.n_facts_committed,
                    review = r.n_review_queue,
                    failures = r.n_extractor_failures,
                    "reflector run complete",
                ),
                Err(err) => tracing::error!(error = ?err, "reflector run failed"),
            }
        })
    })
    .with_context(|| format!("parsing cron schedule {:?}", options.schedule))?;

    sched.add(job).await.context("registering reflector job")?;
    sched.start().await.context("starting JobScheduler")?;
    tracing::info!(schedule = %options.schedule, "reflector scheduler started");

    cancel.cancelled().await;
    tracing::info!("reflector shutting down");
    let _ = sched.shutdown().await;
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

async fn run_reflect(
    config: Config,
    scope: Option<String>,
    limit: Option<i64>,
    rerun: bool,
    since: Option<String>,
) -> anyhow::Result<()> {
    if since.is_some() && !rerun {
        anyhow::bail!("--since is only meaningful with --rerun");
    }

    let parsed_since = match since {
        Some(s) => Some(
            time::OffsetDateTime::parse(&s, &time::format_description::well_known::Rfc3339)
                .with_context(|| format!("parsing --since={s:?} as RFC-3339"))?,
        ),
        None => None,
    };

    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;
    let extractor = build_extractor(&config.extractor)?;

    // CLI flags override config defaults.
    let mut options = config.reflector.clone();
    if let Some(s) = scope {
        options.scope_filter = Some(s);
    }
    if let Some(l) = limit {
        options.max_thoughts_per_run = l;
    }

    let report = if rerun {
        tracing::info!(
            scope = ?options.scope_filter,
            limit = options.max_thoughts_per_run,
            since = ?parsed_since,
            "engram reflect --rerun starting",
        );
        engram_mcp::run_reflector_rerun(&pool, extractor.as_ref(), &options, parsed_since).await?
    } else {
        tracing::info!(
            scope = ?options.scope_filter,
            limit = options.max_thoughts_per_run,
            "engram reflect starting",
        );
        engram_mcp::run_reflector_once(&pool, extractor.as_ref(), &options).await?
    };

    tracing::info!(
        run_id = %report.run_id,
        processed = report.n_thoughts_processed,
        committed = report.n_facts_committed,
        review = report.n_review_queue,
        failures = report.n_extractor_failures,
        "reflect complete",
    );

    if report.n_extractor_failures > 0 {
        anyhow::bail!(
            "{} thoughts failed extraction (see logs); {} committed, {} routed to review",
            report.n_extractor_failures,
            report.n_facts_committed,
            report.n_review_queue,
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
        Command::Reflect { scope, limit, rerun, since } => {
            run_reflect(config, scope, limit, rerun, since).await
        }
    }
}

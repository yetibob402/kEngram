//! engram — the only binary. M1 wired `serve` and `migrate`; M2 added the
//! worker; M3 added `reflect`, `embed-backfill --target`, and the bench
//! harness. M4 collapses the facts pipeline into a thoughts-only sidecar:
//! `reflect` → `tag`, `embed-backfill` loses its `--target` flag, and the
//! worker drops its reflector cron in favour of a plain tag drainer that
//! runs on the same tick as the embed drainer.

mod bench;
mod config;

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::{Parser, Subcommand};
use engram_core::{Embedder, EmbeddingModel, Tagger};
use engram_embed::{
    OpenAICompatibleConfig, OpenAICompatibleEmbedder, Reranker, TeiReranker, TeiRerankerConfig,
};
use engram_extract::{OpenAICompatibleConfig as TaggerConfigBuilder, OpenAICompatibleTagger};
use engram_mcp::EngramServer;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use sqlx::postgres::PgPoolOptions;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, EmbedderConfig, RerankerConfig, TaggerConfig, WorkerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "engram",
    version,
    about = "Self-hosted MCP-native memory service"
)]
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
    /// Long-running worker process: drains both `pending_embeddings` and
    /// `pending_tags` on every `[worker] tick_interval_seconds` tick.
    /// Tag drainer only spawns when `[tagger]` is configured.
    Worker,
    /// Embed thoughts that don't yet have an embedding row for the active
    /// embedder model. M4 thoughts-only (the `--target` flag is gone).
    EmbedBackfill {
        /// Restrict to a single scope.
        #[arg(long)]
        scope: Option<String>,
        /// Maximum number of thoughts to embed in this run.
        #[arg(long, default_value_t = 1000)]
        limit: i64,
    },
    /// One-shot tag run. By default walks untagged thoughts (where
    /// `tags_extractor_version IS NULL`). With `--rerun`, also walks
    /// thoughts whose `tags_extractor_version < [tagger].model_version`
    /// (the tagger prompt has been bumped) — useful after upgrading the
    /// tagger model or prompt.
    Tag {
        /// Restrict to a single scope (exact match).
        #[arg(long)]
        scope: Option<String>,
        /// Max thoughts to process this run. Defaults to 200.
        #[arg(long, default_value_t = 200)]
        limit: i64,
        /// Re-tag thoughts whose `tags_extractor_version` is older than the
        /// configured `[tagger].model_version`. Without this, only
        /// never-tagged thoughts are walked.
        #[arg(long)]
        rerun: bool,
        /// Restrict to thoughts created at or after this RFC-3339 timestamp.
        /// Allowed with or without `--rerun` (unlike `engram reflect`'s
        /// `--since`, which required `--rerun`).
        #[arg(long)]
        since: Option<String>,
    },
    /// Benchmarking harness — A/B comparisons across search-pipeline
    /// configurations. Subcommand-action shape leaves room for additional
    /// bench targets without flattening the CLI.
    Bench {
        #[command(subcommand)]
        action: BenchAction,
    },
}

#[derive(Subcommand, Debug)]
enum BenchAction {
    /// Run rerank A/B (RRF-only vs cross-encoder-reranked) over a fixture
    /// corpus. Prints a markdown table to stdout. See
    /// `tests/fixtures/bench-rerank.example.json` for the fixture schema.
    /// Requires `[reranker]` to be configured.
    Rerank {
        /// Path to the fixture JSON file.
        #[arg(long)]
        corpus: PathBuf,
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
        other => anyhow::bail!("unknown embedder provider: {other:?} (valid: 'openai-compatible')"),
    }
}

/// Build the reranker per `[reranker]` config. Returns `None` when the
/// provider is empty (the silent-disable sentinel), so the search pipeline
/// falls through to the Phase B step 1 RRF + recency pipeline without
/// erroring. Logs the resolved config at INFO level — startup-config
/// observability convention from Phase A (commit 1d627e4).
fn build_reranker(c: &RerankerConfig) -> anyhow::Result<Option<Arc<dyn Reranker>>> {
    if c.provider.is_empty() {
        tracing::info!("reranker: not configured (rerank stage disabled)");
        return Ok(None);
    }
    match c.provider.as_str() {
        "tei" => {
            let reranker = TeiReranker::new(TeiRerankerConfig {
                endpoint: c.endpoint.clone(),
                model_id: c.model_id.clone(),
                timeout: Duration::from_secs(c.timeout_seconds),
            })
            .with_context(|| format!("constructing reranker for endpoint {}", c.endpoint))?;
            tracing::info!(
                provider = %c.provider,
                endpoint = %c.endpoint,
                model_id = %c.model_id,
                timeout_seconds = c.timeout_seconds,
                "reranker: resolved config",
            );
            Ok(Some(Arc::new(reranker)))
        }
        other => anyhow::bail!("unknown reranker provider: {other:?} (valid: 'tei' or empty)"),
    }
}

/// Resolved tagger plus the bits of the original config the callers need
/// to know about. `tagger` is `None` on silent-disable (empty `provider`);
/// the `version` field is the configured `model_version` either way so
/// `engram tag` and the worker tick can compare against it without
/// re-reading config.
struct ResolvedTagger {
    tagger: Option<Arc<dyn Tagger>>,
    /// Stable identity the server stamps onto `pending_tags` rows at
    /// capture time. `None` mirrors `tagger == None`.
    model_id: Option<String>,
    /// Prompt/schema version, written into `thoughts.tags_extractor_version`.
    version: i32,
}

/// Build the tagger per `[tagger]` config. Returns a `ResolvedTagger` with
/// every field present on opt-in and only `version` populated on silent-
/// disable (empty `provider`). Mirrors `build_reranker`'s silent-disable
/// pattern. Logs the resolved config at INFO level on opt-in (Phase A
/// startup-config observability convention; commit 1d627e4).
fn build_tagger(c: &TaggerConfig) -> anyhow::Result<ResolvedTagger> {
    if c.provider.is_empty() {
        tracing::info!("tagger: not configured (capture-time enqueue silently disabled)");
        return Ok(ResolvedTagger {
            tagger: None,
            model_id: None,
            version: c.model_version,
        });
    }

    // Resolve the system prompt: bundled by default; load from a file when
    // `system_prompt_file` is set. The path is anyhow-context'd so errors
    // surface with the path the operator typed.
    let system_prompt =
        match c.system_prompt_file.as_ref() {
            Some(path) => Some(std::fs::read_to_string(path).with_context(|| {
                format!("reading tagger system_prompt_file at {}", path.display())
            })?),
            None => None,
        };
    tracing::info!(
        provider = %c.provider,
        endpoint = %c.endpoint,
        system_prompt = %match c.system_prompt_file.as_ref() {
            Some(p) => format!("file:{}", p.display()),
            None => "bundled".to_string(),
        },
        model_name = %c.model_name,
        model_id = %c.model_id,
        model_version = c.model_version,
        timeout_seconds = c.timeout_seconds,
        "tagger: resolved config",
    );

    match c.provider.as_str() {
        "openai-compatible" | "openrouter" => {
            let tagger = OpenAICompatibleTagger::new(TaggerConfigBuilder {
                endpoint: c.endpoint.clone(),
                model_name: c.model_name.clone(),
                model_id: c.model_id.clone(),
                model_version: c.model_version,
                api_key: c.api_key.clone(),
                timeout: Duration::from_secs(c.timeout_seconds),
                temperature: c.temperature,
                system_prompt,
            })
            .with_context(|| format!("constructing tagger for endpoint {}", c.endpoint))?;
            Ok(ResolvedTagger {
                tagger: Some(Arc::new(tagger)),
                model_id: Some(c.model_id.clone()),
                version: c.model_version,
            })
        }
        other => anyhow::bail!(
            "unknown tagger provider: {other:?} (valid: 'openai-compatible', 'openrouter', or empty for silent-disable)"
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
    let reranker = build_reranker(&config.reranker)?;
    // Server only needs the tagger model_id (to stamp pending_tags rows).
    // The actual tagger HTTP client lives in the worker process.
    let ResolvedTagger {
        tagger: _,
        model_id: tagger_model_id,
        version: _,
    } = build_tagger(&config.tagger)?;

    let bind: SocketAddr = config
        .server
        .bind
        .parse()
        .with_context(|| format!("parsing server.bind = {:?}", config.server.bind))?;

    let pool_for_factory = pool.clone();
    let embedder_for_factory = embedder.clone();
    let reranker_for_factory = reranker.clone();
    let tagger_model_id_for_factory = tagger_model_id.clone();
    let factory = move || {
        Ok(EngramServer::new(
            pool_for_factory.clone(),
            embedder_for_factory.clone(),
            reranker_for_factory.clone(),
            tagger_model_id_for_factory.clone(),
        ))
    };

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
    // DNS-rebinding protection: rmcp ships with a safe default allowlist
    // (`localhost` / `127.0.0.1` / `::1`). Operators binding to a non-loopback
    // interface (Tailnet, LAN) must extend the allowlist via
    // `[server].allowed_hosts` in engram.toml, or the rmcp transport rejects
    // every request whose Host header isn't `localhost`. Empty config list =
    // keep rmcp's default; non-empty replaces it.
    let mut http_cfg = StreamableHttpServerConfig::default()
        .with_stateful_mode(false)
        .with_json_response(true);
    if !config.server.allowed_hosts.is_empty() {
        http_cfg = http_cfg.with_allowed_hosts(config.server.allowed_hosts.clone());
    }
    let mcp_service =
        StreamableHttpService::new(factory, LocalSessionManager::default().into(), http_cfg);

    let app = axum::Router::new().nest_service("/mcp", mcp_service);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding HTTP server to {bind}"))?;

    let allowed_hosts_summary = if config.server.allowed_hosts.is_empty() {
        "rmcp default (localhost / 127.0.0.1 / ::1)".to_string()
    } else {
        format!("custom: {:?}", config.server.allowed_hosts)
    };
    tracing::info!(
        bind = %bind,
        allowed_hosts = %allowed_hosts_summary,
        embedder_endpoint = %config.embedder.endpoint,
        model_id = %config.embedder.model_id,
        tagger = %match tagger_model_id.as_deref() {
            Some(id) => format!("enabled ({id})"),
            None => "disabled".to_string(),
        },
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
    let ResolvedTagger {
        tagger,
        model_id: tagger_model_id,
        version: _,
    } = build_tagger(&config.tagger)?;

    let cancel = CancellationToken::new();
    let mut set = tokio::task::JoinSet::new();

    let WorkerConfig {
        tick_interval_seconds,
        batch_size,
    } = config.worker;
    let interval = Duration::from_secs(tick_interval_seconds);

    let drain_pool = pool.clone();
    let drain_embedder = embedder.clone();
    let drain_cancel = cancel.clone();
    set.spawn(async move {
        embed_drainer_loop(
            drain_pool,
            drain_embedder,
            interval,
            batch_size,
            drain_cancel,
        )
        .await;
    });

    // Scope-vocabulary injection: when enabled and size > 0, the drainer
    // pre-fetches the top-N established terms in each thought's scope and
    // passes them to the tagger as controlled-vocabulary hints.
    let scope_vocab_limit: Option<i64> =
        if config.tagger.scope_vocab_enabled && config.tagger.scope_vocab_size > 0 {
            Some(i64::from(config.tagger.scope_vocab_size))
        } else {
            None
        };

    // Tag drainer is silent-disabled when [tagger] isn't configured —
    // mirrors the capture-side enqueue gate. No tag rows can exist in the
    // queue if no captures enqueued them, so even spinning the loop would
    // be wasted ticks; just don't spawn it.
    let tagger_summary = match tagger {
        Some(t) => {
            let tag_pool = pool.clone();
            let tag_tagger = t.clone();
            let tag_cancel = cancel.clone();
            let model_id = t.model_id().to_string();
            set.spawn(async move {
                tag_drainer_loop(
                    tag_pool,
                    tag_tagger,
                    interval,
                    batch_size,
                    scope_vocab_limit,
                    tag_cancel,
                )
                .await;
            });
            format!("enabled ({model_id})")
        }
        None => "disabled".to_string(),
    };

    tracing::info!(
        tick_interval_seconds,
        batch_size,
        embedder_endpoint = %config.embedder.endpoint,
        model_id = %config.embedder.model_id,
        tagger = %tagger_summary,
        tagger_model_id = ?tagger_model_id,
        scope_vocab_limit = ?scope_vocab_limit,
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

/// Tag drainer mirror of `embed_drainer_loop`. Same cadence (`[worker]
/// tick_interval_seconds`) + batch size (`[worker] batch_size`) — Q1 of
/// the M4 spec settled on "one cadence number for the operator to reason
/// about." Only spawned when `[tagger]` is configured.
async fn tag_drainer_loop(
    pool: sqlx::PgPool,
    tagger: std::sync::Arc<dyn Tagger>,
    interval: Duration,
    batch_size: i64,
    scope_vocab_limit: Option<i64>,
    cancel: CancellationToken,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    ticker.tick().await;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::info!("tag drainer shutting down");
                return;
            }
            _ = ticker.tick() => {
                match engram_mcp::drain_pending_tags(
                    &pool,
                    tagger.as_ref(),
                    batch_size,
                    scope_vocab_limit,
                )
                .await
                {
                    Ok(report) if report.processed > 0 => tracing::info!(
                        processed = report.processed,
                        completed = report.completed,
                        failed_transient = report.failed_transient,
                        failed_permanent = report.failed_permanent,
                        "tag drain tick",
                    ),
                    Ok(_) => {} // idle tick; stay quiet
                    Err(err) => tracing::error!(error = ?err, "tag drain tick failed"),
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

    // Treat `--scope ""` as "no filter" (same empty-string normalisation
    // applied elsewhere on the config side).
    let scope_filter = scope.filter(|s| !s.is_empty());

    let report =
        engram_mcp::embed_backfill(&pool, embedder.as_ref(), scope_filter.as_deref(), limit)
            .await?;

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

/// `engram tag` — one-shot tag run. Mirrors `engram reflect`'s M3 surface:
/// walks the candidate set returned by
/// `engram_storage::find_untagged_or_stale_thoughts`, calls
/// `tagger.tag(content)` on each, persists via `update_thought_tags`. No
/// queue interaction — `pending_tags` is the worker-tick path, not this
/// one. Hard-fails when `[tagger]` is unconfigured (matches `engram bench
/// rerank`'s "configured reranker required" stance).
async fn run_tag(
    config: Config,
    scope: Option<String>,
    limit: i64,
    rerun: bool,
    since: Option<String>,
) -> anyhow::Result<()> {
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

    let ResolvedTagger {
        tagger,
        model_id: _,
        version: tagger_version,
    } = build_tagger(&config.tagger)?;
    let tagger = tagger
        .context("`engram tag` requires a configured `[tagger]` section; see DEVELOPMENT.md")?;

    // Mirror the worker's scope-vocab resolution so one-shot tagging applies
    // the same controlled-vocabulary behavior the drainer would.
    let scope_vocab_limit: Option<i64> =
        if config.tagger.scope_vocab_enabled && config.tagger.scope_vocab_size > 0 {
            Some(i64::from(config.tagger.scope_vocab_size))
        } else {
            None
        };

    // Treat `--scope ""` as "no filter" (matches the empty-string-as-None
    // normalisation applied elsewhere).
    let scope_filter = scope.filter(|s| !s.is_empty());

    tracing::info!(
        scope = ?scope_filter,
        limit,
        rerun,
        since = ?parsed_since,
        target_version = tagger_version,
        scope_vocab_limit = ?scope_vocab_limit,
        "engram tag starting",
    );

    let candidates = engram_storage::find_untagged_or_stale_thoughts(
        &pool,
        tagger_version,
        rerun,
        scope_filter.as_deref(),
        parsed_since,
        limit,
    )
    .await
    .context("walking untagged-or-stale thoughts")?;

    let n_candidates = candidates.len();
    let mut tagged = 0usize;
    let mut failed = 0usize;
    let model_id = tagger.model_id().to_string();

    for t in candidates {
        let vocab = match scope_vocab_limit {
            Some(n) if n > 0 => {
                match engram_storage::fetch_scope_vocab(&pool, t.scope.as_str(), n).await {
                    Ok(v) if v.is_empty() => None,
                    Ok(v) => Some(v),
                    Err(err) => {
                        tracing::warn!(
                            thought_id = %t.id,
                            scope = %t.scope.as_str(),
                            error = ?err,
                            "engram tag: scope vocab fetch failed; tagging without vocab",
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        match tagger.tag(&t.content, vocab.as_ref()).await {
            Ok(tags) => {
                if let Err(err) = engram_storage::update_thought_tags(
                    &pool,
                    t.id,
                    &tags,
                    &model_id,
                    tagger_version,
                )
                .await
                {
                    tracing::warn!(
                        thought_id = %t.id,
                        error = ?err,
                        "engram tag: storage write failed; continuing",
                    );
                    failed += 1;
                } else {
                    tagged += 1;
                }
            }
            Err(err) => {
                tracing::warn!(
                    thought_id = %t.id,
                    error = ?err,
                    "engram tag: tagger call failed; continuing",
                );
                failed += 1;
            }
        }
    }

    tracing::info!(n_candidates, tagged, failed, "engram tag complete",);

    if failed > 0 {
        anyhow::bail!(
            "{} thoughts failed tagging (see logs); {} tagged successfully",
            failed,
            tagged,
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
        Command::Tag {
            scope,
            limit,
            rerun,
            since,
        } => run_tag(config, scope, limit, rerun, since).await,
        Command::Bench { action } => match action {
            BenchAction::Rerank { corpus } => run_bench_rerank(config, corpus).await,
        },
    }
}

async fn run_bench_rerank(config: Config, corpus: PathBuf) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    let embedder = build_embedder(&config.embedder)?;
    let reranker = build_reranker(&config.reranker)?
        .context("bench rerank requires a configured [reranker] section; see DEVELOPMENT.md")?;
    bench::run_bench_rerank(&pool, embedder, reranker, &corpus).await
}

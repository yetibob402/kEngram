//! kengram — the only binary. M1 wired `serve` and `migrate`; M2 added the
//! worker; M3 added `reflect`, `embed-backfill --target`, and the bench
//! harness. M4 collapses the facts pipeline into a thoughts-only sidecar:
//! `reflect` → `tag`, `embed-backfill` loses its `--target` flag, and the
//! worker drops its reflector cron in favour of a plain tag drainer that
//! runs on the same tick as the embed drainer.

mod backup;
mod bench;
mod config;
mod eval;

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::{Parser, Subcommand};
use kengram_core::{Embedder, EmbeddingModel, Tagger};
use kengram_embed::{
    OpenAICompatibleConfig, OpenAICompatibleEmbedder, Reranker, TeiReranker, TeiRerankerConfig,
};
use kengram_extract::{OpenAICompatibleConfig as TaggerConfigBuilder, OpenAICompatibleTagger};
use kengram_mcp::KengramServer;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use sqlx::postgres::PgPoolOptions;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, EmbedderConfig, RerankerConfig, TaggerConfig, WorkerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "kengram",
    version,
    about = "Self-hosted MCP-native memory service"
)]
struct Cli {
    /// Path to an `kengram.toml` config file. Overrides `~/.config/kengram/kengram.toml`.
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
        /// Restrict to a single scope (exact match). Mutually exclusive
        /// with `--scope-prefix`.
        #[arg(long, conflicts_with = "scope_prefix")]
        scope: Option<String>,
        /// Restrict to scopes starting with this prefix (e.g. `kengram.`
        /// matches `kengram.dogfood`, `kengram.test`, etc.). Mutually
        /// exclusive with `--scope`.
        #[arg(long, conflicts_with = "scope")]
        scope_prefix: Option<String>,
        /// Maximum number of thoughts to embed in this run.
        #[arg(long, default_value_t = 1000)]
        limit: i64,
    },
    /// One-shot tag run. By default walks untagged thoughts (where
    /// `tags_extractor_version IS NULL`). With `--rerun`, also walks
    /// thoughts whose `tags_extractor_version < [tagger].model_version`
    /// (the tagger prompt has been bumped). With `--force`, re-tags every
    /// matching thought regardless of version — useful after switching the
    /// tagger model without bumping the prompt version.
    Tag {
        /// Restrict to a single scope (exact match). Mutually exclusive
        /// with `--scope-prefix`.
        #[arg(long, conflicts_with = "scope_prefix")]
        scope: Option<String>,
        /// Restrict to scopes starting with this prefix. Mutually
        /// exclusive with `--scope`.
        #[arg(long, conflicts_with = "scope")]
        scope_prefix: Option<String>,
        /// Max thoughts to process this run. Defaults to 200.
        #[arg(long, default_value_t = 200)]
        limit: i64,
        /// Re-tag thoughts whose `tags_extractor_version` is older than the
        /// configured `[tagger].model_version`. Without this, only
        /// never-tagged thoughts are walked.
        #[arg(long)]
        rerun: bool,
        /// Re-tag every matching thought regardless of version (re-stamps the
        /// configured model_version and records the new model_id). Use after
        /// switching the tagger model. Bound the run with --scope /
        /// --scope-prefix / --since / --limit.
        #[arg(long)]
        force: bool,
        /// Restrict to thoughts created at or after this RFC-3339 timestamp.
        /// Allowed with or without `--rerun` (unlike `kengram reflect`'s
        /// `--since`, which required `--rerun`).
        #[arg(long)]
        since: Option<String>,
        /// Before tagging, snapshot current tags + provenance for ALL
        /// non-retracted thoughts to a JSON file (retag overwrites `tags` in
        /// place; there is no history table). Bare `--snapshot` writes
        /// `./kengram-tag-snapshot-<unixtime>.json`; `--snapshot=PATH` writes
        /// to PATH. Recover by hand via psql if a retag produces worse tags.
        #[arg(long, value_name = "PATH", num_args = 0..=1, require_equals = true)]
        snapshot: Option<Option<PathBuf>>,
    },
    /// Benchmarking harness — A/B comparisons across search-pipeline
    /// configurations. Subcommand-action shape leaves room for additional
    /// bench targets without flattening the CLI.
    Bench {
        #[command(subcommand)]
        action: BenchAction,
    },
    /// Operator-diagnostics queries — surface state that's normally only
    /// visible via psql. Subcommand-action shape leaves room for more
    /// audit resources without flattening the CLI.
    Audit {
        #[command(subcommand)]
        resource: AuditResource,
    },
    /// Evaluation harness — offline quality measurement (first M7 eval
    /// suite: multi-model tagger comparison). `eval tagger` NEVER touches
    /// the database: it reads a corpus file, calls tagger HTTP endpoints,
    /// and writes a report file.
    Eval {
        #[command(subcommand)]
        action: eval::EvalAction,
    },
    /// Print corpus + storage telemetry: thought counts, embeddings,
    /// links, per-scope summary, per-table heap/index/total sizes.
    /// Operator-facing snapshot of "how much am I storing?" without psql.
    Stats {
        /// Optional scope prefix. Only filters which scopes appear in the
        /// scopes summary section; corpus-global counts stay global.
        #[arg(long)]
        scope_prefix: Option<String>,
        /// Limit the scopes summary section to the N most-recently-used
        /// scopes. Defaults to 20.
        #[arg(long, default_value_t = 20)]
        top_scopes: usize,
    },
    /// Back up the database to a portable archive (pg_dump + manifest
    /// sidecar). Pair with `restore` for machine-to-machine migration.
    Backup {
        /// Output archive path. Defaults to
        /// `./kengram-backup-<timestamp>.tar.gz` in the current directory.
        #[arg(long, value_name = "PATH")]
        to: Option<PathBuf>,
        /// Exclude embeddings table data from the archive. Smaller
        /// backup; restore will require `kengram embed-backfill` to
        /// repopulate vectors. The HNSW index and table definition
        /// survive empty.
        #[arg(long)]
        skip_embeddings: bool,
    },
    /// Restore the database from a backup archive. DESTRUCTIVE — replaces
    /// existing schema and data on the target. Refuses without `--force`
    /// when the target already has thoughts. Validates the manifest
    /// against the target's schema head and embedder/tagger config before
    /// proceeding.
    Restore {
        /// Input archive path. Required.
        #[arg(long, value_name = "PATH")]
        from: PathBuf,
        /// Confirm replacement of existing data. Required when the
        /// target's `thoughts` table is non-empty; unnecessary on a
        /// freshly-migrated empty database.
        #[arg(long)]
        force: bool,
        /// Skip the manifest compatibility check (schema head, embedder,
        /// tagger). Advanced — use only when you understand the
        /// implications.
        #[arg(long)]
        skip_version_check: bool,
    },
}

#[derive(Subcommand, Debug)]
enum AuditResource {
    /// Print the `migration_audit` log: per-migration ran_at, rows_touched,
    /// notes. Most recent first.
    Migrations {
        /// Optional RFC-3339 lower bound on `ran_at`. Restrict the log to
        /// recent entries.
        #[arg(long)]
        since: Option<String>,
        /// Maximum rows to print. Defaults to 50.
        #[arg(long, default_value_t = 50)]
        limit: i64,
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
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,kengram=debug"));
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
/// `kengram tag` and the worker tick can compare against it without
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

    // Provider registry. The `Tagger` trait at kengram-core is the public
    // contract for tagger backends — anyone who implements it is pluggable.
    // The match below is the registry of *known* implementations, not the
    // contract itself. To add a new in-tree backend:
    //   1. Write a struct implementing kengram_core::Tagger.
    //   2. Add a config sub-section to TaggerConfig if it needs config
    //      beyond the flat fields openai-compatible already uses.
    //   3. Add an arm to this match mapping a new provider string to your
    //      constructor.
    // For out-of-tree backends, prefer the sidecar pattern: run an HTTP
    // service that speaks kengram's wire contract (the kengram-tagger-protocol
    // crate is the spec) and use provider = "http" to point at it.
    // See `docs/tagger-backends.md` for the full contract + recipe.
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
        "http" => {
            let http_cfg = c.http.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "provider = \"http\" requires a [tagger.http] config section (endpoint, optional api_key, optional timeout_seconds). See docs/tagger-backends.md.",
                )
            })?;
            let tagger = kengram_extract::HttpTagger::new(kengram_extract::HttpTaggerConfig {
                endpoint: http_cfg.endpoint.clone(),
                model_id: c.model_id.clone(),
                model_version: c.model_version,
                api_key: http_cfg.api_key.clone(),
                timeout: Duration::from_secs(http_cfg.timeout_seconds),
            })
            .with_context(|| {
                format!(
                    "constructing http tagger for endpoint {}",
                    http_cfg.endpoint
                )
            })?;
            Ok(ResolvedTagger {
                tagger: Some(Arc::new(tagger)),
                model_id: Some(c.model_id.clone()),
                version: c.model_version,
            })
        }
        other => anyhow::bail!(
            "unknown tagger provider: {other:?} (valid: 'openai-compatible', 'openrouter', 'http' for sidecar, or empty for silent-disable). See docs/tagger-backends.md for how to register a new backend."
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
    kengram_storage::ensure_ann_projection_ready(&pool, embedder.model())
        .await
        .context("ensuring ANN projection readiness")?;
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
        Ok(KengramServer::new(
            pool_for_factory.clone(),
            embedder_for_factory.clone(),
            reranker_for_factory.clone(),
            tagger_model_id_for_factory.clone(),
        ))
    };

    let cancel = CancellationToken::new();
    // Stateless mode (allowed by MCP Streamable HTTP spec 2025-06-18 for
    // simple request-response tools). Kengram has no per-session state — every
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
    // `[server].allowed_hosts` in kengram.toml, or the rmcp transport rejects
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
        "kengram serve started"
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
    kengram_storage::ensure_ann_projection_ready(&pool, embedder.model())
        .await
        .context("ensuring ANN projection readiness")?;
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
        "kengram worker started"
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
    let mut coverage_audit_ticks = 0_u64;
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
                match kengram_mcp::drain_pending_embeddings(&pool, embedder.as_ref(), batch_size).await {
                    Ok(report) if report.found > 0 => tracing::info!(
                        found = report.found,
                        embedded = report.embedded,
                        failed = report.failed,
                        "embed drain tick",
                    ),
                    Ok(_) => {} // idle tick; stay quiet
                    Err(err) => tracing::error!(error = ?err, "embed drain tick failed"),
                }
                coverage_audit_ticks = coverage_audit_ticks.saturating_add(1);
                if coverage_audit_ticks >= 12 {
                    coverage_audit_ticks = 0;
                    match kengram_storage::assert_ann_projection_coverage(&pool, embedder.model()).await {
                        Ok(Some(coverage)) => tracing::info!(
                            projection_id = %coverage.projection_id,
                            model_id = %coverage.model_id,
                            embedding_count = coverage.embedding_count,
                            projection_count = coverage.projection_count,
                            missing_count = coverage.missing_count,
                            "ANN projection periodic coverage assertion passed",
                        ),
                        Ok(None) => {}
                        Err(err) => tracing::error!(
                            error = ?err,
                            "ANN projection periodic coverage assertion failed",
                        ),
                    }
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
                match kengram_mcp::drain_pending_tags(
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
    scope_prefix: Option<String>,
    limit: i64,
) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    let embedder = build_embedder(&config.embedder)?;

    // Treat `--scope ""` / `--scope-prefix ""` as "no filter" (same
    // empty-string normalisation applied elsewhere on the config side).
    let scope_filter = scope.filter(|s| !s.is_empty());
    let scope_prefix_filter = scope_prefix.filter(|s| !s.is_empty());

    let report = kengram_mcp::embed_backfill(
        &pool,
        embedder.as_ref(),
        scope_filter.as_deref(),
        scope_prefix_filter.as_deref(),
        limit,
    )
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

/// `kengram tag` — one-shot tag run. Mirrors `kengram reflect`'s M3 surface:
/// walks the candidate set returned by
/// `kengram_storage::find_untagged_or_stale_thoughts`, calls
/// `tagger.tag(content)` on each, persists via `update_thought_tags`. No
/// queue interaction — `pending_tags` is the worker-tick path, not this
/// one. Hard-fails when `[tagger]` is unconfigured (matches `kengram bench
/// rerank`'s "configured reranker required" stance).
/// Shape pre-retag snapshot rows into the JSON array written by `--snapshot`.
/// `thought_id` serializes as a bare UUID string (`ThoughtId` is transparent),
/// and untagged rows carry `null` provenance — both pinned by the unit test.
fn snapshot_rows_to_json(rows: &[kengram_storage::TagSnapshotRow]) -> Vec<serde_json::Value> {
    rows.iter()
        .map(|r| {
            serde_json::json!({
                "thought_id": r.thought_id,
                "tags": r.tags,
                "tags_extractor_model": r.tags_extractor_model,
                "tags_extractor_version": r.tags_extractor_version,
            })
        })
        .collect()
}

// Args mirror the `Command::Tag` clap flags 1:1; grouping them into a struct
// would just move the same fields behind one more name. Same rationale as the
// wide query helpers in kengram-storage.
#[allow(clippy::too_many_arguments)]
async fn run_tag(
    config: Config,
    scope: Option<String>,
    scope_prefix: Option<String>,
    limit: i64,
    rerun: bool,
    force: bool,
    since: Option<String>,
    snapshot: Option<Option<PathBuf>>,
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

    // Pre-retag safety net: snapshot current tags before any destructive
    // overwrite. Runs before build_tagger so the snapshot is captured even if
    // the tagger turns out to be misconfigured. Corpus-wide (all non-retracted
    // rows) regardless of --scope/--since, so any row stays recoverable.
    if let Some(snapshot_arg) = snapshot {
        let path = snapshot_arg.unwrap_or_else(|| {
            PathBuf::from(format!(
                "kengram-tag-snapshot-{}.json",
                time::OffsetDateTime::now_utc().unix_timestamp()
            ))
        });
        let rows = kengram_storage::snapshot_nonretracted_tags(&pool)
            .await
            .context("capturing pre-retag tag snapshot")?;
        let json = snapshot_rows_to_json(&rows);
        let file = std::fs::File::create(&path)
            .with_context(|| format!("creating snapshot file {}", path.display()))?;
        serde_json::to_writer_pretty(file, &json)
            .with_context(|| format!("writing snapshot to {}", path.display()))?;
        tracing::info!(
            count = rows.len(),
            path = %path.display(),
            "kengram tag: wrote pre-retag tag snapshot",
        );
        println!(
            "Wrote pre-retag snapshot of {} non-retracted thoughts to {}",
            rows.len(),
            path.display()
        );
    }

    let ResolvedTagger {
        tagger,
        model_id: _,
        version: tagger_version,
    } = build_tagger(&config.tagger)?;
    let tagger = tagger
        .context("`kengram tag` requires a configured `[tagger]` section; see DEVELOPMENT.md")?;

    // Mirror the worker's scope-vocab resolution so one-shot tagging applies
    // the same controlled-vocabulary behavior the drainer would.
    let scope_vocab_limit: Option<i64> =
        if config.tagger.scope_vocab_enabled && config.tagger.scope_vocab_size > 0 {
            Some(i64::from(config.tagger.scope_vocab_size))
        } else {
            None
        };

    // Treat `--scope ""` / `--scope-prefix ""` as "no filter" (matches the
    // empty-string-as-None normalisation applied elsewhere).
    let scope_filter = scope.filter(|s| !s.is_empty());
    let scope_prefix_filter = scope_prefix.filter(|s| !s.is_empty());

    tracing::info!(
        scope = ?scope_filter,
        scope_prefix = ?scope_prefix_filter,
        limit,
        rerun,
        force,
        since = ?parsed_since,
        target_version = tagger_version,
        scope_vocab_limit = ?scope_vocab_limit,
        "kengram tag starting",
    );

    let candidates = kengram_storage::find_untagged_or_stale_thoughts(
        &pool,
        tagger_version,
        rerun,
        force,
        scope_filter.as_deref(),
        scope_prefix_filter.as_deref(),
        parsed_since,
        limit,
    )
    .await
    .context("walking untagged-or-stale thoughts")?;

    let n_candidates = candidates.len();
    let mut tagged = 0usize;
    let mut failed = 0usize;
    let model_id = tagger.model_id().to_string();

    // Corpus scope set, fetched once (mirrors the worker drainer) so the
    // scope-identifier filter in `finalize` runs without a per-thought query.
    let known_scopes = kengram_storage::list_scopes(&pool, None)
        .await
        .map(|s| {
            s.into_iter()
                .map(|x| x.scope.as_str().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    for t in candidates {
        let vocab = match scope_vocab_limit {
            Some(n) if n > 0 => {
                match kengram_storage::fetch_scope_vocab(&pool, t.scope.as_str(), n).await {
                    Ok(v) if v.is_empty() => None,
                    Ok(v) => Some(v),
                    Err(err) => {
                        tracing::warn!(
                            thought_id = %t.id,
                            scope = %t.scope.as_str(),
                            error = ?err,
                            "kengram tag: scope vocab fetch failed; tagging without vocab",
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        match tagger.tag(&t.content, vocab.as_ref()).await {
            Ok(mut output) => {
                // Apply the same deterministic post-tag pipeline the worker
                // drainer runs (topic-normalize + people/entities disjoint).
                // This path previously skipped it, so CLI-tagged rows could
                // diverge from worker-tagged rows for the same thought.
                kengram_mcp::finalize::finalize_tags(
                    &mut output.tags,
                    &t.metadata,
                    &t.scope,
                    vocab.as_ref(),
                    &known_scopes,
                );
                if let Err(err) = kengram_storage::update_thought_tags(
                    &pool,
                    t.id,
                    &output.tags,
                    &model_id,
                    tagger_version,
                )
                .await
                {
                    tracing::warn!(
                        thought_id = %t.id,
                        error = ?err,
                        "kengram tag: storage write failed; continuing",
                    );
                    failed += 1;
                } else {
                    // Apply tagger-extracted relations after tags persist.
                    // Mirrors the worker drainer path so synchronous
                    // `kengram tag` and async drainer behave identically.
                    // Relations go to thought_links via apply_tagger_relations,
                    // NOT into tags.relations (migration 0011 removed the
                    // JSONB key).
                    kengram_mcp::apply_tagger_relations(&pool, t.id, &output.relations).await;
                    tagged += 1;
                }
            }
            Err(err) => {
                tracing::warn!(
                    thought_id = %t.id,
                    error = ?err,
                    "kengram tag: tagger call failed; continuing",
                );
                failed += 1;
            }
        }
    }

    tracing::info!(n_candidates, tagged, failed, "kengram tag complete",);

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
        Command::EmbedBackfill {
            scope,
            scope_prefix,
            limit,
        } => run_embed_backfill(config, scope, scope_prefix, limit).await,
        Command::Tag {
            scope,
            scope_prefix,
            limit,
            rerun,
            force,
            since,
            snapshot,
        } => {
            run_tag(
                config,
                scope,
                scope_prefix,
                limit,
                rerun,
                force,
                since,
                snapshot,
            )
            .await
        }
        Command::Bench { action } => match action {
            BenchAction::Rerank { corpus } => run_bench_rerank(config, corpus).await,
        },
        Command::Audit { resource } => match resource {
            AuditResource::Migrations { since, limit } => {
                run_audit_migrations(config, since, limit).await
            }
        },
        Command::Eval { action } => match action {
            // Deliberately does NOT receive `config`: `eval tagger` must
            // never see a database URL, let alone connect (see eval/mod.rs).
            eval::EvalAction::Tagger(args) => eval::run_tagger_cli(args).await,
            eval::EvalAction::ExportCorpus(args) => {
                eval::export::run_export_cli(config, args).await
            }
        },
        Command::Stats {
            scope_prefix,
            top_scopes,
        } => run_stats(config, scope_prefix, top_scopes).await,
        Command::Backup {
            to,
            skip_embeddings,
        } => backup::run_backup(config, to, skip_embeddings).await,
        Command::Restore {
            from,
            force,
            skip_version_check,
        } => backup::run_restore(config, from, force, skip_version_check).await,
    }
}

async fn run_audit_migrations(
    config: Config,
    since: Option<String>,
    limit: i64,
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

    let rows = kengram_storage::query_migration_audit(&pool, parsed_since, limit)
        .await
        .context("querying migration_audit")?;

    if rows.is_empty() {
        println!("(no migration_audit rows)");
        return Ok(());
    }

    // Two-line-per-row "header + notes" rendering. Migration names are
    // long; a column-table would either truncate them or fan out wide.
    for r in rows {
        let ran_at = r
            .ran_at
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| r.ran_at.to_string());
        println!(
            "{ran_at}  rows_touched={rows}  {name}",
            ran_at = ran_at,
            rows = r.rows_touched,
            name = r.migration,
        );
        if let Some(notes) = r.notes {
            println!("    {notes}");
        }
    }
    Ok(())
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

/// `kengram stats` — corpus + storage telemetry snapshot. Read-only; reads
/// from the live DB and prints a sectional plain-text report. Designed for
/// "operator wants to know what's stored" without a psql session.
async fn run_stats(
    config: Config,
    scope_prefix: Option<String>,
    top_scopes: usize,
) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    let scope_prefix = scope_prefix.filter(|s| !s.is_empty());
    let stats = kengram_storage::corpus_stats(&pool, scope_prefix.as_deref())
        .await
        .context("querying corpus_stats")?;

    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "<now>".into());
    println!("kengram corpus — {now}");
    println!();

    // -- Corpus section --
    let retracted_pct = if stats.thoughts.live + stats.thoughts.retracted > 0 {
        100 * stats.thoughts.retracted / (stats.thoughts.live + stats.thoughts.retracted)
    } else {
        0
    };
    println!(
        "  Thoughts:    {} live, {} retracted ({}%), {} untagged",
        stats.thoughts.live, stats.thoughts.retracted, retracted_pct, stats.thoughts.untagged
    );
    println!(
        "  Content:     {} total (avg {}/thought)",
        humanize_bytes(stats.thoughts.content_bytes_total),
        humanize_bytes(stats.thoughts.content_bytes_avg)
    );
    if stats.embeddings.is_empty() {
        println!("  Embeddings:  none");
    } else {
        let parts: Vec<String> = stats
            .embeddings
            .iter()
            .map(|e| {
                format!(
                    "{} × {} ({}-dim, v{})",
                    e.count, e.model_id, e.dimensions, e.model_version
                )
            })
            .collect();
        println!("  Embeddings:  {}", parts.join("; "));
    }
    if !stats.ann_projections.is_empty() {
        let parts: Vec<String> = stats
            .ann_projections
            .iter()
            .map(|p| {
                format!(
                    "{} {} raw={} projected={} missing={}",
                    p.projection_id,
                    p.status,
                    p.embedding_count,
                    p.projection_count,
                    p.missing_count
                )
            })
            .collect();
        println!("  ANN cover:   {}", parts.join("; "));
    }
    println!(
        "  Links:       {} live, {} soft-deleted",
        stats.links.live, stats.links.soft_deleted
    );
    if !stats.links.by_relation.is_empty() {
        let parts: Vec<String> = stats
            .links
            .by_relation
            .iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect();
        println!("    by relation:   {}", parts.join(", "));
    }
    if !stats.links.by_kind.is_empty() {
        let parts: Vec<String> = stats
            .links
            .by_kind
            .iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect();
        println!("    by kind:       {}", parts.join(", "));
    }
    if !stats.links.by_source.is_empty() {
        let parts: Vec<String> = stats
            .links
            .by_source
            .iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect();
        println!("    by source:     {}", parts.join(", "));
    }
    println!(
        "  Queues:      embeddings: {} pending, tags: {} pending",
        stats.queues.pending_embeddings, stats.queues.pending_tags
    );

    // -- Scopes section --
    println!();
    if stats.scopes.is_empty() {
        let suffix = scope_prefix
            .as_deref()
            .map(|p| format!(" matching prefix {p:?}"))
            .unwrap_or_default();
        println!("Scopes (0){suffix}: (no scopes)");
    } else {
        let shown = stats.scopes.iter().take(top_scopes).collect::<Vec<_>>();
        let truncated = stats.scopes.len().saturating_sub(shown.len());
        let suffix = scope_prefix
            .as_deref()
            .map(|p| format!(" matching prefix {p:?}"))
            .unwrap_or_default();
        println!(
            "Scopes ({}){}:{}",
            stats.scopes.len(),
            suffix,
            if truncated > 0 {
                format!("  (showing top {top_scopes}, {truncated} more hidden)")
            } else {
                String::new()
            }
        );
        let max_scope_w = shown
            .iter()
            .map(|s| s.scope.as_str().len())
            .max()
            .unwrap_or(0);
        for s in shown {
            let last_date = s
                .last_activity_at
                .format(&time::format_description::well_known::Iso8601::DATE)
                .unwrap_or_else(|_| "?".into());
            println!(
                "  {scope:<width$}  {n} thoughts  last {last}",
                scope = s.scope.as_str(),
                width = max_scope_w,
                n = s.thought_count,
                last = last_date,
            );
        }
    }

    // -- On-disk tables section --
    println!();
    println!(
        "On-disk tables ({} total):",
        humanize_bytes(stats.database_total_bytes)
    );
    let max_table_w = stats
        .storage
        .iter()
        .map(|t| t.table.len())
        .max()
        .unwrap_or(0);
    for t in &stats.storage {
        println!(
            "  {table:<width$}  {total:>9}  (heap {heap}, indexes {idx})",
            table = t.table,
            width = max_table_w,
            total = humanize_bytes(t.total_bytes),
            heap = humanize_bytes(t.heap_bytes),
            idx = humanize_bytes(t.indexes_bytes),
        );
    }

    Ok(())
}

/// Format a byte count as a human-readable string using kibi-base units
/// (1 KB = 1024 bytes; matches what operators see in `du -h` / `psql`'s
/// `pg_size_pretty`). One decimal place for MB+ to keep precision useful
/// at small corpus sizes without overwhelming the eye.
fn humanize_bytes(n: i64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let n_f = n as f64;
    if n_f < KB {
        format!("{n} B")
    } else if n_f < MB {
        format!("{:.0} KB", n_f / KB)
    } else if n_f < GB {
        format!("{:.1} MB", n_f / MB)
    } else {
        format!("{:.2} GB", n_f / GB)
    }
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    #[test]
    fn humanize_bytes_renders_unit_scale() {
        assert_eq!(humanize_bytes(0), "0 B");
        assert_eq!(humanize_bytes(999), "999 B");
        assert_eq!(humanize_bytes(1024), "1 KB");
        assert_eq!(humanize_bytes(40 * 1024), "40 KB");
        assert_eq!(humanize_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(humanize_bytes(12 * 1024 * 1024), "12.0 MB");
        assert_eq!(humanize_bytes(1024_i64.pow(3)), "1.00 GB");
    }
}

#[cfg(test)]
mod tag_tests {
    use super::*;
    use kengram_core::ThoughtId;
    use kengram_storage::TagSnapshotRow;

    #[test]
    fn snapshot_rows_to_json_shapes_provenance_and_bare_uuid() {
        let id = ThoughtId::new();
        let rows = vec![
            TagSnapshotRow {
                thought_id: id,
                tags: serde_json::json!({"people": ["Ron"], "kind": "observation"}),
                tags_extractor_model: Some("ollama/test".to_string()),
                tags_extractor_version: Some(13),
            },
            TagSnapshotRow {
                thought_id: ThoughtId::new(),
                tags: serde_json::json!({}),
                tags_extractor_model: None,
                tags_extractor_version: None,
            },
        ];

        let json = snapshot_rows_to_json(&rows);

        assert_eq!(json.len(), 2);
        // thought_id is a bare UUID string, not a wrapper object.
        assert_eq!(
            json[0]["thought_id"].as_str().unwrap(),
            id.as_uuid().to_string()
        );
        assert_eq!(json[0]["tags"]["people"][0], "Ron");
        assert_eq!(json[0]["tags_extractor_model"], "ollama/test");
        assert_eq!(json[0]["tags_extractor_version"], 13);
        // Untagged row → null provenance.
        assert!(json[1]["tags_extractor_model"].is_null());
        assert!(json[1]["tags_extractor_version"].is_null());
    }
}

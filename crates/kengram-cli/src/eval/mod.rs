//! `kengram eval` — the M7 evaluation harness. First suite: multi-model
//! tagger comparison against a golden corpus.
//!
//! # The no-production-database invariant
//!
//! `kengram eval tagger` must NEVER touch a database. It reads a corpus
//! file, calls tagger HTTP endpoints, and writes a report file — nothing
//! else. The corpus file carries all the context the production pipeline
//! would otherwise fetch from Postgres (`known_scopes`, per-scope vocab
//! snapshots), so `finalize_tags` and vocab injection run DB-free.
//!
//! Structurally: `corpus`, `score`, `report`, `arms`, and `run` do not
//! import `sqlx`, and the `EvalAction::Tagger` dispatch arm never receives
//! a config or constructs a `PgPool`. The only eval module allowed to talk
//! to the database is `export` (corpus drafting), and it is restricted to
//! read-only SELECTs. The `no_db_tripwire` test below turns the import
//! half of that invariant into a red test rather than a review convention.

pub(crate) mod arms;
pub(crate) mod corpus;
pub(crate) mod export;
pub(crate) mod report;
pub(crate) mod run;
pub(crate) mod score;

use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};

use run::{RunArm, RunOpts};

#[derive(Subcommand, Debug)]
pub(crate) enum EvalAction {
    /// Compare candidate tagger models against a golden corpus. DB-free:
    /// reads the corpus file, calls tagger HTTP endpoints, writes a JSON
    /// report. Headline scores are computed on FINALIZED output (the same
    /// deterministic post-filters production persists).
    Tagger(TaggerArgs),
    /// Draft a corpus file from the configured database (read-only
    /// SELECTs; writes nothing to Postgres). Items carry current
    /// production tags as DRAFT labels (`"reviewed": false`) — hand-review
    /// them to make the corpus golden.
    ExportCorpus(export::ExportArgs),
}

#[derive(Args, Debug)]
pub(crate) struct TaggerArgs {
    /// Path to a golden corpus JSON file (see eval/corpora/example.json).
    #[arg(long)]
    pub corpus: PathBuf,
    /// Path to the model-arms TOML. Defaults to ./eval/models.toml
    /// (committed example: eval/models.example.toml).
    #[arg(long)]
    pub models: Option<PathBuf>,
    /// Restrict the run to named arms (repeatable). Default: all arms.
    #[arg(long = "arm", value_name = "NAME")]
    pub arms: Vec<String>,
    /// Tag each item N times per arm. N >= 2 enables stability metrics
    /// (kind drift, per-field Jaccard variance).
    #[arg(long, default_value_t = 1)]
    pub repeats: u32,
    /// Scope-vocab injection: `on` (production parity), `off`, or `both`
    /// (each arm runs twice, with `+vocab` / `-vocab` name suffixes).
    #[arg(long, value_enum, default_value_t = VocabMode::On)]
    pub vocab: VocabMode,
    /// Report output path. Defaults to eval/reports/tagger-<timestamp>.json.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Score items with "reviewed": false too (draft labels). The report
    /// is marked so the numbers can't masquerade as golden.
    #[arg(long)]
    pub include_unreviewed: bool,
    /// Cap the number of corpus items scored (first N after filtering).
    #[arg(long)]
    pub limit: Option<usize>,
    /// Concurrent tagger calls per arm. Default 1 — a local Ollama
    /// serializes requests anyway.
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,
    /// Abandon an arm after this many consecutive call failures — the
    /// looping/dead-endpoint guard for unattended runs. Its resolved calls
    /// still score; remaining calls are skipped and the arm is marked
    /// aborted in the report. 0 disables.
    #[arg(long, default_value_t = 5)]
    pub max_consecutive_failures: u32,
    /// Validate corpus + arms, print the call-count / wall-time preview,
    /// and exit without making any tagger calls.
    #[arg(long)]
    pub dry_run: bool,
    /// Emit the JSON report to stdout instead of the human-readable table.
    #[arg(long)]
    pub json: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VocabMode {
    On,
    Off,
    Both,
}

impl VocabMode {
    fn as_str(self) -> &'static str {
        match self {
            VocabMode::On => "on",
            VocabMode::Off => "off",
            VocabMode::Both => "both",
        }
    }
}

/// Print a usage/config error and exit with code 2.
fn usage_error(err: impl std::fmt::Display) -> ! {
    eprintln!("kengram eval tagger: {err}");
    std::process::exit(2);
}

/// Entry point for `kengram eval tagger`. Deliberately takes no `Config`
/// and constructs no database pool — see the module docs.
///
/// Exit codes: 0 = run completed; 1 = at least one arm had a 100%
/// call-failure rate (the report is still written); 2 = usage/config error.
pub(crate) async fn run_tagger_cli(args: TaggerArgs) -> anyhow::Result<()> {
    if args.repeats == 0 {
        usage_error("--repeats must be >= 1");
    }
    if args.concurrency == 0 {
        usage_error("--concurrency must be >= 1");
    }

    // -- corpus ---------------------------------------------------------
    let corpus = match corpus::load_corpus(&args.corpus) {
        Ok(c) => c,
        Err(e) => usage_error(format!("{e:#}")),
    };
    let corpus_item_count = corpus.items.len();

    let mut items: Vec<corpus::CorpusItem> = if args.include_unreviewed {
        corpus.items.clone()
    } else {
        let unreviewed = corpus_item_count - corpus.reviewed_count();
        if unreviewed > 0 {
            eprintln!(
                "skipping {unreviewed} unreviewed item(s) (draft labels); \
                 hand-review them and set \"reviewed\": true, or pass --include-unreviewed"
            );
        }
        corpus
            .items
            .iter()
            .filter(|i| i.reviewed)
            .cloned()
            .collect()
    };
    if let Some(limit) = args.limit {
        items.truncate(limit);
    }
    if items.is_empty() {
        usage_error("no items to score after filtering (is the corpus reviewed yet?)");
    }

    // -- arms -------------------------------------------------------------
    let models_path = args
        .models
        .clone()
        .unwrap_or_else(|| PathBuf::from("eval/models.toml"));
    let mut specs = match arms::load_arm_specs(&models_path) {
        Ok(s) => s,
        Err(e) => usage_error(format!(
            "{e:#}\n(hint: copy eval/models.example.toml to {} and edit)",
            models_path.display()
        )),
    };
    if !args.arms.is_empty() {
        let known: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
        for requested in &args.arms {
            if !known.contains(requested) {
                usage_error(format!(
                    "--arm {requested:?} not found in {} (declared arms: {})",
                    models_path.display(),
                    known.join(", ")
                ));
            }
        }
        specs.retain(|s| args.arms.contains(&s.name));
    }

    let mut run_arms: Vec<RunArm> = Vec::new();
    let max_timeout_secs = specs
        .iter()
        .map(|s| s.timeout_seconds.unwrap_or(arms::DEFAULT_TIMEOUT_SECONDS))
        .max()
        .unwrap_or(arms::DEFAULT_TIMEOUT_SECONDS);
    for spec in &specs {
        let built = match arms::build_arm(spec) {
            Ok(b) => b,
            Err(e) => usage_error(format!("{e:#}")),
        };
        let vocab_variants: &[(bool, &str)] = match args.vocab {
            VocabMode::On => &[(true, "")],
            VocabMode::Off => &[(false, "")],
            VocabMode::Both => &[(true, "+vocab"), (false, "-vocab")],
        };
        for (vocab_on, suffix) in vocab_variants {
            run_arms.push(RunArm {
                name: format!("{}{}", built.name, suffix),
                provider: built.provider.clone(),
                endpoint: built.endpoint.clone(),
                model_name: built.model_name.clone(),
                model_version: built.model_version,
                prompt: built.prompt.clone(),
                vocab_on: *vocab_on,
                tagger: built.tagger.clone(),
            });
        }
    }

    // -- dry run -----------------------------------------------------------
    let total_calls = items.len() * run_arms.len() * args.repeats as usize;
    if args.dry_run {
        println!(
            "dry run: {} item(s) x {} arm(s) x {} repeat(s) = {} tagger call(s)",
            items.len(),
            run_arms.len(),
            args.repeats,
            total_calls
        );
        println!(
            "worst-case wall time ~ {}s ({} call(s) / concurrency {} x {}s max timeout)",
            total_calls.div_ceil(args.concurrency) as u64 * max_timeout_secs,
            total_calls,
            args.concurrency,
            max_timeout_secs
        );
        println!(
            "arms: {}",
            run_arms
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Ok(());
    }

    // -- run ----------------------------------------------------------------
    // Resolve the report path up front: the run flushes the report-so-far
    // there after every completed arm, so a crashed overnight run keeps
    // its finished arms.
    let out_path = args.out.clone().unwrap_or_else(|| {
        let ts = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "unknown-time".to_string())
            .replace(':', "-");
        PathBuf::from(format!("eval/reports/tagger-{ts}.json"))
    });
    eprintln!(
        "running {} call(s) across {} arm(s); report (flushed per arm) at {}",
        total_calls,
        run_arms.len(),
        out_path.display()
    );
    let opts = RunOpts {
        repeats: args.repeats,
        concurrency: args.concurrency,
        corpus_path: args.corpus.display().to_string(),
        vocab_mode: args.vocab.as_str().to_string(),
        unreviewed_items_included: args.include_unreviewed,
        corpus_item_count,
        flush_path: Some(out_path.clone()),
        max_consecutive_failures: args.max_consecutive_failures,
    };
    let report = run::run_tagger_eval(&corpus, &items, &run_arms, &opts).await?;

    // -- emit -----------------------------------------------------------
    report::write_report(&report, &out_path)?;
    eprintln!("report written to {}", out_path.display());

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", report::render_table(&report));
    }

    let dead_arms: Vec<&str> = report
        .arms
        .iter()
        .filter(|a| a.aborted.is_some() || (a.calls.attempted > 0 && a.calls.failure_rate >= 1.0))
        .map(|a| a.name.as_str())
        .collect();
    if !dead_arms.is_empty() {
        eprintln!(
            "arm(s) aborted or with 100% call failure (looping, endpoint down, or misconfigured): {}",
            dead_arms.join(", ")
        );
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The DB-free eval modules must not so much as import sqlx. `export.rs`
    /// is deliberately absent from this list — it is the one eval module
    /// allowed a (read-only) database connection.
    #[test]
    fn no_db_tripwire() {
        for (name, src) in [
            ("mod.rs", include_str!("mod.rs")),
            ("corpus.rs", include_str!("corpus.rs")),
            ("score.rs", include_str!("score.rs")),
            ("report.rs", include_str!("report.rs")),
            ("arms.rs", include_str!("arms.rs")),
            ("run.rs", include_str!("run.rs")),
        ] {
            // Match code references (an import or a path qualification of
            // the database crate), not prose mentions in doc comments. The
            // needles are concat!-assembled so this test's own source
            // doesn't trip the check.
            assert!(
                !src.contains(concat!("sqlx", "::")) && !src.contains(concat!("use ", "sqlx")),
                "eval/{name} must stay database-free (found a sqlx reference)"
            );
        }
    }
}

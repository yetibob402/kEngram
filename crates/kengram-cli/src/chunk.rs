//! Batch chunking utilities for the recall-surgery pipeline.
//!
//! This module deliberately stays deterministic and operator-driven. The agent
//! builds the component; batch scripts run it with dry-run artifacts, apply
//! chunks idempotently, and enqueue embeddings under an explicit cap.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Args, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::config::Config;

const SOURCE_KIND: &str = "thought-chunk";
const DEFAULT_TARGET_WORDS: usize = 280;
const DEFAULT_MIN_WORDS: usize = 160;
const DEFAULT_MAX_WORDS: usize = 420;
const DEFAULT_OVERLAP_WORDS: usize = 55;

#[derive(Subcommand, Debug)]
pub(crate) enum ChunkAction {
    /// Produce a deterministic dry-run report and optional JSONL chunk artifact.
    DryRun(ChunkDryRunArgs),
    /// Persist chunks for eligible thoughts. Idempotent by source thought +
    /// chunk fingerprint.
    Apply(ChunkApplyArgs),
    /// Enqueue unembedded artifact chunks with an explicit cap.
    Enqueue(ChunkEnqueueArgs),
}

#[derive(Args, Debug, Clone)]
pub(crate) struct ChunkDryRunArgs {
    #[command(flatten)]
    pub select: ChunkSelectArgs,
    /// Write proposed chunks as JSONL for sample audit.
    #[arg(long)]
    pub chunks_out: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct ChunkApplyArgs {
    #[command(flatten)]
    pub select: ChunkSelectArgs,
    /// Artifact directory or run note recorded on corpus_pipeline_runs.
    #[arg(long)]
    pub artifact_dir: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct ChunkEnqueueArgs {
    /// Restrict through source parent thought scope. Mutually exclusive with
    /// `--scope-prefix`.
    #[arg(long, conflicts_with = "scope_prefix")]
    pub scope: Option<String>,
    /// Restrict through source parent thought scope prefix. Mutually exclusive
    /// with `--scope`.
    #[arg(long, conflicts_with = "scope")]
    pub scope_prefix: Option<String>,
    /// Maximum chunk embeddings to enqueue this run. Keep small enough to avoid
    /// live query-embed contention.
    #[arg(long, default_value_t = 40)]
    pub limit: i64,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct ChunkSelectArgs {
    /// Restrict parent thoughts to a single scope. Mutually exclusive with
    /// `--scope-prefix`.
    #[arg(long, conflicts_with = "scope_prefix")]
    pub scope: Option<String>,
    /// Restrict parent thoughts to scopes starting with this prefix. Mutually
    /// exclusive with `--scope`.
    #[arg(long, conflicts_with = "scope")]
    pub scope_prefix: Option<String>,
    /// Maximum parent thoughts to inspect.
    #[arg(long, default_value_t = 200)]
    pub limit: i64,
    /// Minimum parent word count before chunking. Short thoughts are already
    /// precise enough as thoughts.
    #[arg(long, default_value_t = DEFAULT_MIN_WORDS)]
    pub min_parent_words: usize,
    /// Newline-delimited UUIDs of parent thoughts that must never be selected.
    /// Used for mechanically enforcing protected-gold canary exclusions.
    #[arg(long, value_name = "PATH")]
    pub exclude_thought_ids_file: Option<PathBuf>,
    /// Target words per chunk.
    #[arg(long, default_value_t = DEFAULT_TARGET_WORDS)]
    pub target_words: usize,
    /// Maximum words per chunk before forcing a boundary.
    #[arg(long, default_value_t = DEFAULT_MAX_WORDS)]
    pub max_words: usize,
    /// Overlap words between adjacent chunks.
    #[arg(long, default_value_t = DEFAULT_OVERLAP_WORDS)]
    pub overlap_words: usize,
}

#[derive(Debug, Serialize)]
struct ChunkRunReport {
    mode: String,
    parent_candidates: usize,
    parent_chunkable: usize,
    parent_skipped_short_or_single: usize,
    chunks_emitted: usize,
    chunks_inserted: Option<u64>,
    embeddings_enqueued: Option<usize>,
    parameters: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, sqlx::FromRow)]
struct ParentRow {
    id: Uuid,
    scope: String,
    content: String,
    source: String,
    created_at: OffsetDateTime,
    metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct ProposedChunk {
    source_thought_id: Uuid,
    artifact_id: Uuid,
    chunk_index: i32,
    content: String,
    content_fingerprint_hex: String,
    chunker_id: String,
    chunker_version: i32,
    token_estimate: i32,
    start_char: i32,
    end_char: i32,
    metadata: serde_json::Value,
}

pub(crate) async fn run_chunk_cli(config: Config, action: ChunkAction) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    match action {
        ChunkAction::DryRun(args) => run_dry_run(&pool, args).await,
        ChunkAction::Apply(args) => run_apply(&pool, args).await,
        ChunkAction::Enqueue(args) => run_enqueue(&pool, &config.embedder.model_id, args).await,
    }
}

async fn run_dry_run(pool: &PgPool, args: ChunkDryRunArgs) -> anyhow::Result<()> {
    let parents = select_parent_rows(pool, &args.select).await?;
    let chunks = propose_chunks(&parents, &args.select);
    if let Some(path) = args.chunks_out.as_ref() {
        write_chunks_jsonl(path, &chunks)?;
    }
    let report = build_report("dry_run", &parents, &chunks, &args.select, None, None);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn run_apply(pool: &PgPool, args: ChunkApplyArgs) -> anyhow::Result<()> {
    let parents = select_parent_rows(pool, &args.select).await?;
    let chunks = propose_chunks(&parents, &args.select);
    let mut tx = pool.begin().await?;
    let run_id =
        insert_pipeline_run(&mut tx, "batch", &args.select, args.artifact_dir.as_ref()).await?;
    let inserted = persist_chunks(&mut tx, run_id, &chunks).await?;
    finish_pipeline_run(&mut tx, run_id, &parents, &chunks, inserted).await?;
    tx.commit().await?;

    let report = build_report(
        "batch",
        &parents,
        &chunks,
        &args.select,
        Some(inserted),
        None,
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn run_enqueue(pool: &PgPool, model_id: &str, args: ChunkEnqueueArgs) -> anyhow::Result<()> {
    if args.limit <= 0 {
        bail!("--limit must be positive");
    }
    let scope = args.scope.filter(|s| !s.is_empty());
    let scope_prefix = args.scope_prefix.filter(|s| !s.is_empty());
    let enqueued = kengram_storage::enqueue_unembedded_artifact_chunks(
        pool,
        model_id,
        scope.as_deref(),
        scope_prefix.as_deref(),
        args.limit,
    )
    .await
    .context("enqueueing artifact chunk embeddings")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "mode": "enqueue",
            "model_id": model_id,
            "limit": args.limit,
            "enqueued": enqueued,
        }))?
    );
    Ok(())
}

async fn select_parent_rows(
    pool: &PgPool,
    args: &ChunkSelectArgs,
) -> anyhow::Result<Vec<ParentRow>> {
    if args.limit <= 0 {
        bail!("--limit must be positive");
    }
    validate_chunk_params(args)?;

    let scope = args.scope.as_deref().filter(|s| !s.is_empty());
    let scope_prefix = args.scope_prefix.as_deref().filter(|s| !s.is_empty());
    let excluded_thought_ids = load_excluded_thought_ids(args.exclude_thought_ids_file.as_ref())?;
    let rows = sqlx::query_as::<_, ParentRow>(
        r#"
        SELECT t.id, t.scope, t.content, t.source, t.created_at, t.metadata
        FROM thoughts t
        WHERE t.retracted_at IS NULL
          AND t.scope NOT LIKE 'archive.%'
          AND ($1::text IS NULL OR t.scope = $1)
          AND ($2::text IS NULL OR t.scope LIKE $2 || '%')
          AND NOT (t.id = ANY($3::uuid[]))
          AND NOT EXISTS (
              SELECT 1
              FROM artifact_chunks ac
              WHERE ac.source_thought_id = t.id
                AND ac.retracted_at IS NULL
          )
        ORDER BY LENGTH(t.content) DESC, t.created_at ASC
        LIMIT $4
        "#,
    )
    .bind(scope)
    .bind(scope_prefix)
    .bind(excluded_thought_ids)
    .bind(args.limit)
    .fetch_all(pool)
    .await
    .context("selecting parent thoughts for chunking")?;

    Ok(rows)
}

fn load_excluded_thought_ids(path: Option<&PathBuf>) -> anyhow::Result<Vec<Uuid>> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading excluded thought IDs from {}", path.display()))?;
    let mut ids = BTreeSet::new();
    for (line_no, line) in contents.lines().enumerate() {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }
        let id = Uuid::parse_str(trimmed).with_context(|| {
            format!(
                "invalid UUID in excluded thought IDs file {}:{}",
                path.display(),
                line_no + 1
            )
        })?;
        ids.insert(id);
    }
    Ok(ids.into_iter().collect())
}

fn validate_chunk_params(args: &ChunkSelectArgs) -> anyhow::Result<()> {
    if args.min_parent_words == 0 || args.target_words == 0 || args.max_words == 0 {
        bail!("chunk word limits must be positive");
    }
    if args.target_words > args.max_words {
        bail!("--target-words must be <= --max-words");
    }
    if args.overlap_words >= args.target_words {
        bail!("--overlap-words must be smaller than --target-words");
    }
    Ok(())
}

fn propose_chunks(parents: &[ParentRow], args: &ChunkSelectArgs) -> Vec<ProposedChunk> {
    let chunker_id = format!(
        "sentence-aware-words:{}-{}-overlap{}",
        args.target_words, args.max_words, args.overlap_words
    );
    let mut out = Vec::new();
    for parent in parents {
        let spans = chunk_spans(
            &parent.content,
            args.min_parent_words,
            args.target_words,
            args.max_words,
            args.overlap_words,
        );
        if spans.len() <= 1 {
            continue;
        }
        let artifact_id = deterministic_uuid(parent.id, "chunk-artifact");
        let chunk_count = spans.len();
        for (idx, span) in spans.into_iter().enumerate() {
            let content = parent.content[span.start_char..span.end_char]
                .trim()
                .to_string();
            let fp = sha256_bytes(content.as_bytes());
            let metadata = serde_json::json!({
                "source_kind": SOURCE_KIND,
                "parent_thought_id": parent.id,
                "parent_scope": parent.scope,
                "parent_source": parent.source,
                "parent_created_at": parent.created_at,
                "parent_metadata": parent.metadata,
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "start_char": span.start_char,
                "end_char": span.end_char,
            });
            out.push(ProposedChunk {
                source_thought_id: parent.id,
                artifact_id,
                chunk_index: idx as i32,
                content_fingerprint_hex: hex_lower(&fp),
                content,
                chunker_id: chunker_id.clone(),
                chunker_version: 1,
                token_estimate: (span.word_count as f32 * 1.35).ceil() as i32,
                start_char: span.start_char as i32,
                end_char: span.end_char as i32,
                metadata,
            });
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct ChunkSpan {
    start_char: usize,
    end_char: usize,
    word_count: usize,
}

fn chunk_spans(
    text: &str,
    min_parent_words: usize,
    target_words: usize,
    max_words: usize,
    overlap_words: usize,
) -> Vec<ChunkSpan> {
    let words = word_spans(text);
    if words.len() < min_parent_words {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0_usize;
    while start < words.len() {
        let remaining = words.len() - start;
        if remaining <= max_words && remaining <= target_words + overlap_words {
            let end = words.len();
            chunks.push(span_from_words(&words, start, end));
            break;
        }

        let mut end = (start + target_words).min(words.len());
        let hard_end = (start + max_words).min(words.len());
        if let Some(boundary) = best_sentence_boundary(text, &words, start, end, hard_end) {
            end = boundary;
        } else {
            end = hard_end;
        }
        if end <= start {
            break;
        }
        chunks.push(span_from_words(&words, start, end));
        if end >= words.len() {
            break;
        }
        start = end.saturating_sub(overlap_words);
    }

    chunks
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, idx));
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

fn best_sentence_boundary(
    text: &str,
    words: &[(usize, usize)],
    start: usize,
    target_end: usize,
    hard_end: usize,
) -> Option<usize> {
    let min_end = (start + (target_end - start).saturating_mul(2) / 3).max(start + 1);
    let mut best = None;
    for idx in min_end..=hard_end {
        let end_char = words[idx - 1].1;
        let punct = text[..end_char]
            .chars()
            .next_back()
            .is_some_and(|c| matches!(c, '.' | '!' | '?' | ';' | ':'));
        if punct {
            best = Some(idx);
        }
    }
    best
}

fn span_from_words(words: &[(usize, usize)], start: usize, end: usize) -> ChunkSpan {
    ChunkSpan {
        start_char: words[start].0,
        end_char: words[end - 1].1,
        word_count: end - start,
    }
}

async fn insert_pipeline_run(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    mode: &str,
    args: &ChunkSelectArgs,
    artifact_dir: Option<&PathBuf>,
) -> anyhow::Result<Uuid> {
    let parameters = select_args_json(args);
    let row: (Uuid,) = sqlx::query_as(
        r#"
        INSERT INTO corpus_pipeline_runs (pipeline_kind, mode, status, parameters, artifact_dir)
        VALUES ('chunk', $1, 'running', $2, $3)
        RETURNING id
        "#,
    )
    .bind(mode)
    .bind(parameters)
    .bind(artifact_dir.map(|p| p.display().to_string()))
    .fetch_one(&mut **tx)
    .await
    .context("creating corpus_pipeline_runs row")?;
    Ok(row.0)
}

async fn persist_chunks(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    run_id: Uuid,
    chunks: &[ProposedChunk],
) -> anyhow::Result<u64> {
    let mut inserted = 0_u64;
    for chunk in chunks {
        ensure_artifact(tx, chunk).await?;
        let fingerprint = hex::decode(&chunk.content_fingerprint_hex)
            .map_err(anyhow::Error::msg)
            .context("decoding proposed chunk fingerprint")?;
        let result = sqlx::query(
            r#"
            INSERT INTO artifact_chunks (
                artifact_id,
                source_thought_id,
                chunk_index,
                content,
                content_fingerprint,
                chunker_id,
                chunker_version,
                token_estimate,
                start_char,
                end_char,
                metadata,
                pipeline_run_id
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (source_thought_id, content_fingerprint)
                WHERE source_thought_id IS NOT NULL AND content_fingerprint IS NOT NULL
            DO NOTHING
            "#,
        )
        .bind(chunk.artifact_id)
        .bind(chunk.source_thought_id)
        .bind(chunk.chunk_index)
        .bind(&chunk.content)
        .bind(fingerprint)
        .bind(&chunk.chunker_id)
        .bind(chunk.chunker_version)
        .bind(chunk.token_estimate)
        .bind(chunk.start_char)
        .bind(chunk.end_char)
        .bind(&chunk.metadata)
        .bind(run_id)
        .execute(&mut **tx)
        .await
        .context("inserting artifact chunk")?;
        inserted += result.rows_affected();
    }
    Ok(inserted)
}

async fn ensure_artifact(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    chunk: &ProposedChunk,
) -> anyhow::Result<()> {
    sqlx::query(
        r#"
        INSERT INTO artifacts (id, scope, kind, title, metadata)
        SELECT $1, t.scope, 'thought_chunks', 'thought:' || t.id::text, jsonb_build_object(
            'source_kind', $2,
            'source_thought_id', t.id::text
        )
        FROM thoughts t
        WHERE t.id = $3
        ON CONFLICT (id) DO NOTHING
        "#,
    )
    .bind(chunk.artifact_id)
    .bind(SOURCE_KIND)
    .bind(chunk.source_thought_id)
    .execute(&mut **tx)
    .await
    .context("ensuring chunk parent artifact")?;
    Ok(())
}

async fn finish_pipeline_run(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    run_id: Uuid,
    parents: &[ParentRow],
    chunks: &[ProposedChunk],
    inserted: u64,
) -> anyhow::Result<()> {
    let stats = serde_json::json!({
        "parent_candidates": parents.len(),
        "chunks_emitted": chunks.len(),
        "chunks_inserted": inserted,
    });
    sqlx::query(
        r#"
        UPDATE corpus_pipeline_runs
        SET status = 'completed',
            stats = $2,
            finished_at = NOW()
        WHERE id = $1
        "#,
    )
    .bind(run_id)
    .bind(stats)
    .execute(&mut **tx)
    .await
    .context("finishing corpus_pipeline_runs row")?;
    Ok(())
}

fn build_report(
    mode: &str,
    parents: &[ParentRow],
    chunks: &[ProposedChunk],
    args: &ChunkSelectArgs,
    inserted: Option<u64>,
    enqueued: Option<usize>,
) -> ChunkRunReport {
    let chunked_parent_ids = chunks
        .iter()
        .map(|c| c.source_thought_id)
        .collect::<std::collections::BTreeSet<_>>();
    ChunkRunReport {
        mode: mode.to_string(),
        parent_candidates: parents.len(),
        parent_chunkable: chunked_parent_ids.len(),
        parent_skipped_short_or_single: parents.len().saturating_sub(chunked_parent_ids.len()),
        chunks_emitted: chunks.len(),
        chunks_inserted: inserted,
        embeddings_enqueued: enqueued,
        parameters: select_args_map(args),
    }
}

fn select_args_json(args: &ChunkSelectArgs) -> serde_json::Value {
    serde_json::Value::Object(select_args_map(args).into_iter().collect())
}

fn select_args_map(args: &ChunkSelectArgs) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        ("scope".to_string(), serde_json::json!(args.scope)),
        (
            "scope_prefix".to_string(),
            serde_json::json!(args.scope_prefix),
        ),
        ("limit".to_string(), serde_json::json!(args.limit)),
        (
            "min_parent_words".to_string(),
            serde_json::json!(args.min_parent_words),
        ),
        (
            "exclude_thought_ids_file".to_string(),
            serde_json::json!(
                args.exclude_thought_ids_file
                    .as_ref()
                    .map(|p| p.display().to_string())
            ),
        ),
        (
            "target_words".to_string(),
            serde_json::json!(args.target_words),
        ),
        ("max_words".to_string(), serde_json::json!(args.max_words)),
        (
            "overlap_words".to_string(),
            serde_json::json!(args.overlap_words),
        ),
    ])
}

fn write_chunks_jsonl(path: &PathBuf, chunks: &[ProposedChunk]) -> anyhow::Result<()> {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str(&serde_json::to_string(chunk)?);
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn deterministic_uuid(seed: Uuid, suffix: &str) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    hasher.update(suffix.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

mod hex {
    pub fn decode(value: &str) -> Result<Vec<u8>, String> {
        if !value.len().is_multiple_of(2) {
            return Err("odd-length hex string".to_string());
        }
        (0..value.len())
            .step_by(2)
            .map(|idx| {
                u8::from_str_radix(&value[idx..idx + 2], 16)
                    .map_err(|e| format!("invalid hex at byte {idx}: {e}"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_spans_splits_with_overlap() {
        let text = (0..700)
            .map(|i| {
                if i % 20 == 19 {
                    format!("word{i}.")
                } else {
                    format!("word{i}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_spans(&text, 160, 280, 420, 55);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.word_count <= 420));
        assert!(chunks.windows(2).all(|w| w[1].start_char < w[0].end_char));
    }

    #[test]
    fn short_parents_do_not_chunk() {
        let text = "short thought with enough meaning";
        assert!(chunk_spans(text, 160, 280, 420, 55).is_empty());
    }
}

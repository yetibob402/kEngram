//! Operator-only contextual retrieval data prep.
//!
//! Serving flags never generate context. This command selects already-chunked
//! corpus rows through the storage-layer contamination fence, builds bounded
//! prompts, calls a configured provider, persists non-searchable rejected rows
//! or ready contextual sidecars, and optionally embeds ready contextual text.

use std::time::Duration;

use anyhow::{Context, bail};
use clap::{Args, Subcommand};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPoolOptions;

use crate::config::{Config, SearchConfig};

const DEFAULT_CONTEXTUAL_GENERATION_LIMIT: i64 = 25;
const STORAGE_MAX_CONTEXT_CHARS: usize = 1_200;

#[derive(Subcommand, Debug)]
pub(crate) enum ContextualAction {
    /// Generate contextual sidecar rows for eligible artifact chunks.
    Generate(ContextualGenerateArgs),
}

#[derive(Args, Debug, Clone)]
pub(crate) struct ContextualGenerateArgs {
    /// Restrict through source parent thought scope. Mutually exclusive with
    /// `--scope-prefix`.
    #[arg(long, conflicts_with = "scope_prefix")]
    pub scope: Option<String>,
    /// Restrict through source parent thought scope prefix. Mutually exclusive
    /// with `--scope`.
    #[arg(long, conflicts_with = "scope")]
    pub scope_prefix: Option<String>,
    /// Maximum chunks to inspect/generate in this run.
    #[arg(long, default_value_t = DEFAULT_CONTEXTUAL_GENERATION_LIMIT)]
    pub limit: i64,
    /// Selection-only proof. Does not call the provider and does not mutate DB.
    #[arg(long)]
    pub dry_run: bool,
    /// After a ready context row is persisted, embed contextual_text + raw
    /// chunk into the contextual BGE-M3 sidecar.
    #[arg(long)]
    pub embed_ready: bool,
}

#[derive(Debug, Serialize)]
struct ContextualGenerateReport {
    mode: String,
    selected: usize,
    ready: usize,
    rejected: usize,
    embedded: usize,
    failed: usize,
    generator_id: String,
    generator_version: i32,
    prompt_version: String,
    prompt_hash: String,
    max_context_chars: usize,
    max_prompt_chars: usize,
}

#[derive(Debug)]
struct OpenAICompatibleContextGenerator {
    client: Client,
    endpoint: String,
    model_name: String,
    timeout_seconds: u64,
    api_key: Option<String>,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct ChatRequestBody<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    response_format: ResponseFormat<'a>,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponseBody {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ContextResponse {
    context: String,
}

impl OpenAICompatibleContextGenerator {
    fn new(c: &SearchConfig) -> anyhow::Result<Self> {
        if c.contextual_generation_endpoint.trim().is_empty() {
            bail!("contextual generation endpoint must be non-empty");
        }
        if c.contextual_generation_model_name.trim().is_empty() {
            bail!("contextual generation model_name must be non-empty");
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(c.contextual_generation_timeout_seconds))
            .build()
            .context("constructing contextual generation HTTP client")?;
        Ok(Self {
            client,
            endpoint: c.contextual_generation_endpoint.clone(),
            model_name: c.contextual_generation_model_name.clone(),
            timeout_seconds: c.contextual_generation_timeout_seconds,
            api_key: c.contextual_generation_api_key.clone(),
            temperature: c.contextual_generation_temperature,
        })
    }

    async fn generate_context(
        &self,
        prompt: &str,
        max_context_chars: usize,
    ) -> anyhow::Result<String> {
        let url = format!("{}/chat/completions", self.endpoint.trim_end_matches('/'));
        let system = contextual_generation_system_prompt(max_context_chars);
        let body = ChatRequestBody {
            model: &self.model_name,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: &system,
                },
                ChatMessage {
                    role: "user",
                    content: prompt,
                },
            ],
            temperature: self.temperature,
            response_format: ResponseFormat {
                kind: "json_object",
            },
        };
        let mut req = self.client.post(url).json(&body);
        if let Some(key) = self.api_key.as_deref()
            && !key.is_empty()
        {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.with_context(|| {
            format!(
                "contextual generation provider request failed or timed out after {}s",
                self.timeout_seconds
            )
        })?;
        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_else(|_| String::new());
            bail!(
                "contextual generation backend returned status {}: {}",
                status.as_u16(),
                message
            );
        }
        let decoded: ChatResponseBody = resp
            .json()
            .await
            .context("decoding contextual generation chat response")?;
        let content = decoded
            .choices
            .first()
            .map(|choice| choice.message.content.trim())
            .filter(|content| !content.is_empty())
            .context("contextual generation response missing choices[0].message.content")?;
        let parsed: ContextResponse = serde_json::from_str(strip_json_code_fence(content))
            .context("parsing contextual generation JSON")?;
        Ok(parsed.context.trim().to_string())
    }
}

pub(crate) async fn run_contextual_cli(
    config: Config,
    action: ContextualAction,
) -> anyhow::Result<()> {
    match action {
        ContextualAction::Generate(args) => run_generate(config, args).await,
    }
}

async fn run_generate(config: Config, args: ContextualGenerateArgs) -> anyhow::Result<()> {
    if !config.search.contextual_generation_enabled {
        bail!("contextual generation requires [search].contextual_generation_enabled=true");
    }
    if config.search.contextual_generation_provider != "openai-compatible" {
        bail!(
            "contextual generation requires [search].contextual_generation_provider=\"openai-compatible\""
        );
    }
    if args.limit <= 0 {
        bail!("--limit must be positive");
    }
    if config.search.contextual_generation_max_context_chars == 0
        || config.search.contextual_generation_max_context_chars > STORAGE_MAX_CONTEXT_CHARS
    {
        bail!(
            "contextual_generation_max_context_chars must be 1..={}",
            STORAGE_MAX_CONTEXT_CHARS
        );
    }
    if config.search.contextual_generation_max_prompt_chars == 0 {
        bail!("contextual_generation_max_prompt_chars must be positive");
    }

    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;
    let scope = args.scope.as_deref().filter(|scope| !scope.is_empty());
    let scope_prefix = args
        .scope_prefix
        .as_deref()
        .filter(|scope_prefix| !scope_prefix.is_empty());
    let prompt_hash = contextual_prompt_hash(&config.search);
    let sources = kengram_storage::select_artifact_chunk_context_generation_sources(
        &pool,
        scope,
        scope_prefix,
        &config.search.contextual_generation_model_id,
        config.search.contextual_generation_version,
        &prompt_hash,
        args.limit,
    )
    .await
    .context("selecting contextual generation sources")?;

    if args.dry_run {
        print_report(ContextualGenerateReport {
            mode: "dry-run".to_string(),
            selected: sources.len(),
            ready: 0,
            rejected: 0,
            embedded: 0,
            failed: 0,
            generator_id: config.search.contextual_generation_model_id.clone(),
            generator_version: config.search.contextual_generation_version,
            prompt_version: config.search.contextual_generation_prompt_version.clone(),
            prompt_hash,
            max_context_chars: config.search.contextual_generation_max_context_chars,
            max_prompt_chars: config.search.contextual_generation_max_prompt_chars,
        })?;
        return Ok(());
    }

    let generator = OpenAICompatibleContextGenerator::new(&config.search)?;
    let embedder = if args.embed_ready {
        Some(super::build_embedder(&config.embedder)?)
    } else {
        None
    };
    let mut ready = 0_usize;
    let mut rejected = 0_usize;
    let mut embedded = 0_usize;
    let mut failed = 0_usize;

    for source in sources {
        let prompt = build_contextual_generation_prompt(
            &source,
            config.search.contextual_generation_max_prompt_chars,
            config.search.contextual_generation_max_context_chars,
        );
        let context_text = match generator
            .generate_context(
                &prompt,
                config.search.contextual_generation_max_context_chars,
            )
            .await
        {
            Ok(text) => text,
            Err(e) => {
                failed += 1;
                tracing::warn!(
                    chunk_id = %source.chunk_id,
                    error = ?e,
                    "contextual generation provider failed for chunk; row not persisted",
                );
                continue;
            }
        };
        let contextual_content = format!("{}\n\n{}", context_text.trim(), source.chunk_content);
        let outcome = kengram_storage::insert_artifact_chunk_context(
            &pool,
            kengram_storage::ArtifactChunkContextInsert {
                chunk_id: source.chunk_id,
                context_text,
                generator_id: config.search.contextual_generation_model_id.clone(),
                generator_version: config.search.contextual_generation_version,
                prompt_version: config.search.contextual_generation_prompt_version.clone(),
                prompt_hash: prompt_hash.clone(),
                model_id: config.search.contextual_generation_model_id.clone(),
                model_version: config.search.contextual_generation_version.to_string(),
                pipeline_run_id: None,
                metadata: serde_json::json!({
                    "provider": config.search.contextual_generation_provider,
                    "model_name": config.search.contextual_generation_model_name,
                    "prompt_chars": prompt.chars().count(),
                }),
            },
        )
        .await
        .with_context(|| format!("persisting contextual row for chunk {}", source.chunk_id))?;
        if outcome.status == "ready" {
            ready += 1;
            if let Some(embedder) = embedder.as_ref() {
                let vectors = embedder
                    .embed(&[contextual_content])
                    .await
                    .context("embedding contextual content")?;
                let embedding = kengram_core::Embedding::new(
                    embedder.model().clone(),
                    vectors
                        .into_iter()
                        .next()
                        .context("embedder returned no vectors")?,
                )
                .context("constructing contextual embedding")?;
                if kengram_storage::insert_artifact_chunk_context_embedding(
                    &pool,
                    outcome.context_id,
                    &embedding,
                )
                .await?
                {
                    embedded += 1;
                }
            }
        } else {
            rejected += 1;
        }
    }

    print_report(ContextualGenerateReport {
        mode: "generate".to_string(),
        selected: ready + rejected + failed,
        ready,
        rejected,
        embedded,
        failed,
        generator_id: config.search.contextual_generation_model_id,
        generator_version: config.search.contextual_generation_version,
        prompt_version: config.search.contextual_generation_prompt_version,
        prompt_hash,
        max_context_chars: config.search.contextual_generation_max_context_chars,
        max_prompt_chars: config.search.contextual_generation_max_prompt_chars,
    })?;

    if failed > 0 {
        bail!("{} contextual rows failed provider generation", failed);
    }
    Ok(())
}

fn contextual_generation_system_prompt(max_context_chars: usize) -> String {
    format!(
        "You generate short factual retrieval context for one immutable chunk. \
Use only the supplied parent and chunk evidence. Do not add answer keys, eval \
labels, or facts not present in the input. Return strict JSON: \
{{\"context\":\"...\"}}. The context must be at most {max_context_chars} \
characters."
    )
}

fn build_contextual_generation_prompt(
    source: &kengram_storage::ContextGenerationSource,
    max_prompt_chars: usize,
    max_context_chars: usize,
) -> String {
    let metadata = serde_json::to_string(source.parent_metadata.as_value())
        .unwrap_or_else(|_| "{}".to_string());
    let parent_budget = max_prompt_chars
        .saturating_sub(source.chunk_content.len())
        .max(500);
    let parent_excerpt = truncate_chars(&source.parent_content, parent_budget);
    let prompt = format!(
        "Goal: write a short factual context prefix for this chunk.\n\
Max context characters: {max_context_chars}\n\
Parent scope: {}\n\
Parent source: {}\n\
Parent created_at: {}\n\
Parent metadata JSON: {}\n\
Chunk index: {}\n\
Raw chunk:\n{}\n\n\
Parent excerpt:\n{}\n\n\
",
        source.scope,
        source.parent_source,
        source.parent_created_at,
        truncate_chars(&metadata, 2_000),
        source.chunk_index,
        source.chunk_content,
        parent_excerpt,
    );
    truncate_chars(&prompt, max_prompt_chars)
}

fn contextual_prompt_hash(c: &SearchConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(c.contextual_generation_prompt_version.as_bytes());
    hasher.update(b"\0");
    hasher.update(c.contextual_generation_model_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(
        c.contextual_generation_max_context_chars
            .to_string()
            .as_bytes(),
    );
    hasher.update(b"\0");
    hasher.update(
        c.contextual_generation_max_prompt_chars
            .to_string()
            .as_bytes(),
    );
    format!("{:x}", hasher.finalize())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

fn strip_json_code_fence(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        return rest.trim_end_matches("```").trim();
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        return rest.trim_end_matches("```").trim();
    }
    trimmed
}

fn print_report(report: ContextualGenerateReport) -> anyhow::Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&report).context("encoding contextual report")?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::{Metadata, Scope, Source, ThoughtId};
    use time::OffsetDateTime;
    use uuid::Uuid;

    fn source(content: &str, chunk: &str) -> kengram_storage::ContextGenerationSource {
        kengram_storage::ContextGenerationSource {
            chunk_id: Uuid::new_v4(),
            source_thought_id: ThoughtId::new(),
            scope: Scope::new("work.test").unwrap(),
            parent_source: Source::new("test").unwrap(),
            parent_created_at: OffsetDateTime::now_utc(),
            parent_metadata: Metadata::from(serde_json::json!({"source_file": "safe.md"})),
            parent_content: content.to_string(),
            chunk_index: 2,
            chunk_content: chunk.to_string(),
            chunk_metadata: serde_json::json!({}),
            raw_chunk_fingerprint: [1_u8; 32],
        }
    }

    #[test]
    fn prompt_builder_bounds_prompt_and_includes_raw_chunk() {
        let src = source(&"parent ".repeat(2_000), "small raw chunk");
        let prompt = build_contextual_generation_prompt(&src, 1_000, 300);
        assert!(prompt.chars().count() <= 1_000);
        assert!(prompt.contains("small raw chunk"));
        assert!(prompt.contains("work.test"));
    }

    #[test]
    fn prompt_hash_changes_when_prompt_contract_changes() {
        let mut c = SearchConfig::default();
        c.contextual_generation_model_id = "model-a".to_string();
        let a = contextual_prompt_hash(&c);
        c.contextual_generation_prompt_version = "other".to_string();
        let b = contextual_prompt_hash(&c);
        assert_ne!(a, b);
    }

    #[test]
    fn code_fence_stripper_accepts_json_fence() {
        assert_eq!(
            strip_json_code_fence("```json\n{\"context\":\"x\"}\n```"),
            "{\"context\":\"x\"}"
        );
    }
}

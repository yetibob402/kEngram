//! `kengram eval export-corpus` — draft a golden corpus from the configured
//! database.
//!
//! This is the ONE eval module allowed to touch the database, and it is
//! restricted to read-only SELECTs: `find_untagged_or_stale_thoughts`
//! (with `force = true`, i.e. every non-retracted thought in the filter
//! window), `list_scopes`, and `fetch_scope_vocab`. It writes nothing to
//! Postgres — the only output is the corpus JSON file.
//!
//! Exported items carry the thought's CURRENT tags as draft labels with
//! `"reviewed": false`. They are not golden until the operator hand-checks
//! each item and flips the flag.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args;
use kengram_core::Thought;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use super::corpus::{
    CORPUS_FORMAT_VERSION, CORPUS_KIND, Corpus, CorpusItem, Provenance, VocabEntry,
};
use crate::config::Config;

#[derive(Args, Debug)]
pub(crate) struct ExportArgs {
    /// Output path for the draft corpus JSON.
    #[arg(long)]
    pub out: PathBuf,
    /// Restrict to a single scope (exact match). Mutually exclusive with
    /// `--scope-prefix`.
    #[arg(long, conflicts_with = "scope_prefix")]
    pub scope: Option<String>,
    /// Restrict to scopes starting with this prefix. Mutually exclusive
    /// with `--scope`.
    #[arg(long, conflicts_with = "scope")]
    pub scope_prefix: Option<String>,
    /// Max thoughts to export (oldest first within the filters).
    #[arg(long, default_value_t = 100)]
    pub limit: i64,
    /// Restrict to thoughts created at or after this RFC-3339 timestamp.
    #[arg(long)]
    pub since: Option<String>,
}

/// Everything the draft corpus needs, gathered via read-only SELECTs.
struct ExportData {
    thoughts: Vec<Thought>,
    known_scopes: Vec<String>,
    scope_vocab: BTreeMap<String, VocabEntry>,
}

async fn gather_export_data(
    pool: &PgPool,
    scope: Option<&str>,
    scope_prefix: Option<&str>,
    since: Option<OffsetDateTime>,
    limit: i64,
    vocab_size: i64,
) -> anyhow::Result<ExportData> {
    // force = true: walk every non-retracted thought in the filter window
    // regardless of tag-version state (target_tagger_version is unused in
    // force mode). Read-only.
    let thoughts = kengram_storage::find_untagged_or_stale_thoughts(
        pool,
        0,
        false,
        true,
        scope,
        scope_prefix,
        since,
        limit,
    )
    .await
    .context("walking thoughts for export")?;

    let known_scopes: Vec<String> = kengram_storage::list_scopes(pool, None)
        .await
        .context("listing scopes")?
        .into_iter()
        .map(|s| s.scope.as_str().to_string())
        .collect();

    let mut scope_vocab = BTreeMap::new();
    for scope in thoughts
        .iter()
        .map(|t| t.scope.as_str().to_string())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let vocab = kengram_storage::fetch_scope_vocab(pool, &scope, vocab_size)
            .await
            .with_context(|| format!("fetching vocab for scope {scope:?}"))?;
        if !vocab.is_empty() {
            scope_vocab.insert(
                scope,
                VocabEntry {
                    topics: vocab.topics,
                    entities: vocab.entities,
                },
            );
        }
    }

    Ok(ExportData {
        thoughts,
        known_scopes,
        scope_vocab,
    })
}

/// Pure assembly of the draft corpus from gathered data.
fn build_draft_corpus(data: ExportData, exported_at: String, source: String) -> Corpus {
    // Draft provenance: the modal extractor model/version across the
    // exported rows (untagged rows have no provenance and don't vote).
    let mut model_votes: BTreeMap<(&str, i32), usize> = BTreeMap::new();
    for t in &data.thoughts {
        if let (Some(model), Some(version)) = (&t.tags_extractor_model, t.tags_extractor_version) {
            *model_votes.entry((model.as_str(), version)).or_insert(0) += 1;
        }
    }
    let modal = model_votes
        .iter()
        .max_by_key(|(_, count)| **count)
        .map(|((model, version), _)| (model.to_string(), *version));
    let mixed_provenance = model_votes.len() > 1;

    let items: Vec<CorpusItem> = data
        .thoughts
        .iter()
        .map(|t| CorpusItem {
            id: t.id.into_uuid().to_string(),
            content: t.content.clone(),
            scope: t.scope.as_str().to_string(),
            metadata: t.metadata.as_value().clone(),
            reviewed: false,
            notes: None,
            golden: t.tags.clone(),
        })
        .collect();

    Corpus {
        format_version: CORPUS_FORMAT_VERSION,
        kind: CORPUS_KIND.to_string(),
        provenance: Provenance {
            exported_at: Some(exported_at),
            source: Some(if mixed_provenance {
                format!("{source} (mixed extractor provenance across rows)")
            } else {
                source
            }),
            draft_tagger_model: modal.as_ref().map(|(m, _)| m.clone()),
            draft_tagger_version: modal.as_ref().map(|(_, v)| *v),
            item_count: Some(items.len()),
        },
        known_scopes: data.known_scopes,
        scope_vocab: data.scope_vocab,
        items,
    }
}

/// Entry point for `kengram eval export-corpus`.
pub(crate) async fn run_export_cli(config: Config, args: ExportArgs) -> anyhow::Result<()> {
    let since = match &args.since {
        None => None,
        Some(s) => match OffsetDateTime::parse(s, &Rfc3339) {
            Ok(t) => Some(t),
            Err(e) => super::usage_error(format!("--since must be RFC-3339: {e}")),
        },
    };

    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .context("connecting to the database")?;

    let data = gather_export_data(
        &pool,
        args.scope.as_deref(),
        args.scope_prefix.as_deref(),
        since,
        args.limit,
        i64::from(config.tagger.scope_vocab_size),
    )
    .await?;
    if data.thoughts.is_empty() {
        super::usage_error("no thoughts matched the export filters");
    }

    let exported_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("formatting export timestamp")?;
    let mut source = "kengram eval export-corpus".to_string();
    if let Some(s) = &args.scope {
        source.push_str(&format!(" --scope {s}"));
    }
    if let Some(p) = &args.scope_prefix {
        source.push_str(&format!(" --scope-prefix {p}"));
    }
    if let Some(s) = &args.since {
        source.push_str(&format!(" --since {s}"));
    }

    let corpus = build_draft_corpus(data, exported_at, source);
    let item_count = corpus.items.len();

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&corpus).context("serializing draft corpus")?;
    std::fs::write(&args.out, json)
        .with_context(|| format!("writing draft corpus to {}", args.out.display()))?;

    println!(
        "Wrote {item_count} DRAFT item(s) to {}.\n\
         Labels are the current production tags and are marked \"reviewed\": false.\n\
         Next steps:\n\
         1. Open the file and hand-check each item's `golden` labels (people,\n\
            entities, action_items, topics, dates_mentioned, kind).\n\
         2. Correct anything wrong — these become the answer key, so precision here\n\
            is what every model gets measured against.\n\
         3. Set \"reviewed\": true on each verified item.\n\
         4. Run: kengram eval tagger --corpus {}",
        args.out.display(),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::{Metadata, Scope, Source, TagKind, Tags};
    use sha2::{Digest, Sha256};

    fn fingerprint(content: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    }

    async fn seed_thought(
        pool: &PgPool,
        scope: &str,
        content: &str,
        tags: Option<&Tags>,
    ) -> kengram_core::ThoughtId {
        let (inserted, is_new) = kengram_storage::insert_thought(
            pool,
            kengram_storage::NewThought {
                scope: &Scope::new(scope).unwrap(),
                content,
                source: &Source::new("manual").unwrap(),
                metadata: &Metadata::empty(),
                content_fingerprint: fingerprint(content),
            },
        )
        .await
        .unwrap();
        assert!(is_new);
        if let Some(tags) = tags {
            kengram_storage::update_thought_tags(pool, inserted.id, tags, "test/model", 16)
                .await
                .unwrap();
        }
        inserted.id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn export_drafts_unreviewed_items_and_leaves_rows_untouched(pool: PgPool) {
        let tagged = Tags {
            people: vec!["Sarah".to_string()],
            topics: vec!["release-process".to_string()],
            kind: Some(TagKind::Task),
            ..Default::default()
        };
        seed_thought(&pool, "demo.a", "Sarah ships Thursday.", Some(&tagged)).await;
        seed_thought(
            &pool,
            "demo.a",
            "Background observation.",
            Some(&Tags::default()),
        )
        .await;
        seed_thought(&pool, "demo.b", "Untagged capture.", None).await;

        let before = kengram_storage::find_untagged_or_stale_thoughts(
            &pool, 0, false, true, None, None, None, 100,
        )
        .await
        .unwrap();

        let data = gather_export_data(&pool, None, Some("demo."), None, 100, 50)
            .await
            .unwrap();
        let corpus = build_draft_corpus(
            data,
            "2026-06-10T00:00:00Z".to_string(),
            "test export".to_string(),
        );

        // Draft schema shape.
        assert_eq!(corpus.format_version, CORPUS_FORMAT_VERSION);
        assert_eq!(corpus.kind, CORPUS_KIND);
        assert_eq!(corpus.items.len(), 3);
        assert!(
            corpus.items.iter().all(|i| !i.reviewed),
            "drafts must be unreviewed"
        );
        assert!(corpus.known_scopes.contains(&"demo.a".to_string()));
        assert!(corpus.known_scopes.contains(&"demo.b".to_string()));
        // Vocab snapshot present for the scope with established topics.
        assert!(corpus.scope_vocab.contains_key("demo.a"));
        // Draft labels are the current prod tags.
        let sarah_item = corpus
            .items
            .iter()
            .find(|i| i.content.contains("Sarah"))
            .unwrap();
        assert_eq!(sarah_item.golden.people, vec!["Sarah".to_string()]);
        assert_eq!(sarah_item.golden.kind, Some(TagKind::Task));
        // Modal provenance recorded.
        assert_eq!(
            corpus.provenance.draft_tagger_model.as_deref(),
            Some("test/model")
        );
        assert_eq!(corpus.provenance.draft_tagger_version, Some(16));
        // The corpus round-trips through the loader's validation rules.
        let json = serde_json::to_string(&corpus).unwrap();
        let reparsed: Corpus = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed.items.len(), 3);

        // Read-only behavioral check: the same walk returns identical rows.
        let after = kengram_storage::find_untagged_or_stale_thoughts(
            &pool, 0, false, true, None, None, None, 100,
        )
        .await
        .unwrap();
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.id, a.id);
            assert_eq!(b.content, a.content);
            assert_eq!(b.tags, a.tags);
            assert_eq!(b.tags_extractor_version, a.tags_extractor_version);
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn export_respects_scope_and_limit_filters(pool: PgPool) {
        seed_thought(&pool, "demo.a", "first", Some(&Tags::default())).await;
        seed_thought(&pool, "demo.a", "second", Some(&Tags::default())).await;
        seed_thought(&pool, "other.scope", "elsewhere", Some(&Tags::default())).await;

        let data = gather_export_data(&pool, Some("demo.a"), None, None, 1, 50)
            .await
            .unwrap();
        assert_eq!(data.thoughts.len(), 1, "--limit 1 within --scope demo.a");
        assert_eq!(data.thoughts[0].content, "first", "oldest first");
        // known_scopes is corpus-wide regardless of the item filter.
        assert!(data.known_scopes.contains(&"other.scope".to_string()));
    }
}

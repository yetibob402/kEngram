//! Golden-corpus file schema for the tagger eval harness.
//!
//! A corpus is a self-contained JSON file: items (content + golden labels)
//! plus the scope context (`known_scopes`, per-scope vocab snapshots) needed
//! to run the production `finalize_tags` pipeline and vocab injection
//! without any database connection. `kengram eval export-corpus` drafts one
//! from the live corpus (read-only); the operator hand-reviews labels and
//! flips `reviewed` to `true` to make them golden.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, bail};
use kengram_core::{ScopeVocab, Tags};
use serde::{Deserialize, Serialize};

pub const CORPUS_FORMAT_VERSION: u32 = 1;
pub const CORPUS_KIND: &str = "tagger-golden-corpus";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    /// Hard-checked on load; bump on breaking schema changes.
    pub format_version: u32,
    /// Discriminator so a report or other JSON file can't be fed in by
    /// accident. Must equal [`CORPUS_KIND`].
    pub kind: String,
    #[serde(default)]
    pub provenance: Provenance,
    /// Corpus-wide scope set, snapshotted at export time. Feeds the
    /// scope-identifier filter in `finalize_tags`.
    #[serde(default)]
    pub known_scopes: Vec<String>,
    /// Per-scope controlled-vocabulary snapshots (already truncated to the
    /// production `scope_vocab_size` at export time). Feeds vocab injection
    /// and topic normalization.
    #[serde(default)]
    pub scope_vocab: BTreeMap<String, VocabEntry>,
    pub items: Vec<CorpusItem>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exported_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_tagger_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_tagger_version: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_count: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VocabEntry {
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusItem {
    /// Thought UUID for exported items; any stable slug for hand-written ones.
    pub id: String,
    pub content: String,
    #[serde(default = "default_scope")]
    pub scope: String,
    /// Thought metadata; exercises the `metadata.decision_type` override in
    /// `finalize_tags`. Must be a JSON object when present.
    #[serde(default = "empty_object")]
    pub metadata: serde_json::Value,
    /// `false` = draft labels (export output, not yet hand-checked). The
    /// scorer skips unreviewed items unless `--include-unreviewed` is given.
    #[serde(default)]
    pub reviewed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// The correct post-finalize labels for this content.
    pub golden: Tags,
}

fn default_scope() -> String {
    "global".to_string()
}

fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

impl Corpus {
    /// Vocab snapshot for a scope, in the shape `Tagger::tag` expects.
    /// `None` when the corpus has no entry for the scope — the same
    /// fall-through the production drainer has for a scope with no
    /// established vocabulary.
    pub fn vocab_for(&self, scope: &str) -> Option<ScopeVocab> {
        self.scope_vocab.get(scope).map(|v| ScopeVocab {
            topics: v.topics.clone(),
            entities: v.entities.clone(),
        })
    }

    pub fn reviewed_count(&self) -> usize {
        self.items.iter().filter(|i| i.reviewed).count()
    }
}

/// Load and validate a corpus file. Validation failures are usage errors
/// (exit code 2 at the CLI layer).
pub fn load_corpus(path: &Path) -> anyhow::Result<Corpus> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading corpus file {}", path.display()))?;
    let corpus: Corpus = serde_json::from_str(&raw)
        .with_context(|| format!("parsing corpus file {}", path.display()))?;

    if corpus.format_version != CORPUS_FORMAT_VERSION {
        bail!(
            "corpus {} has format_version {} (this kengram understands {})",
            path.display(),
            corpus.format_version,
            CORPUS_FORMAT_VERSION
        );
    }
    if corpus.kind != CORPUS_KIND {
        bail!(
            "corpus {} has kind {:?} (expected {:?})",
            path.display(),
            corpus.kind,
            CORPUS_KIND
        );
    }
    if corpus.items.is_empty() {
        bail!("corpus {} contains no items", path.display());
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for item in &corpus.items {
        if item.id.is_empty() {
            bail!("corpus {} has an item with an empty id", path.display());
        }
        if !seen.insert(item.id.as_str()) {
            bail!(
                "corpus {} has duplicate item id {:?}",
                path.display(),
                item.id
            );
        }
        if item.content.is_empty() {
            bail!("corpus item {:?} has empty content", item.id);
        }
        if !item.metadata.is_object() {
            bail!("corpus item {:?}: metadata must be a JSON object", item.id);
        }
    }
    Ok(corpus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn example_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/corpora/example.json")
    }

    #[test]
    fn committed_example_corpus_loads_and_round_trips() {
        let corpus = load_corpus(&example_path()).expect("example corpus must load");
        assert_eq!(corpus.format_version, CORPUS_FORMAT_VERSION);
        assert_eq!(corpus.kind, CORPUS_KIND);
        assert!(corpus.items.len() >= 8, "example should cover all kinds");
        // Every item in the committed example is reviewed (it's synthetic).
        assert_eq!(corpus.reviewed_count(), corpus.items.len());
        // All eight kind classes (7 variants + null) are represented.
        let kinds: BTreeSet<usize> = corpus
            .items
            .iter()
            .map(|i| crate::eval::score::kind_index(i.golden.kind))
            .collect();
        assert_eq!(kinds.len(), 8, "example must cover all 7 kinds + null");
        // Scope context present for finalize_tags + vocab injection.
        assert!(!corpus.known_scopes.is_empty());
        assert!(corpus.vocab_for("demo.eval").is_some());
        assert!(corpus.vocab_for("no.such.scope").is_none());
        // Round-trip: serialize -> reparse -> equal field counts.
        let json = serde_json::to_string_pretty(&corpus).unwrap();
        let reparsed: Corpus = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed.items.len(), corpus.items.len());
        assert_eq!(reparsed.known_scopes, corpus.known_scopes);
    }

    #[test]
    fn rejects_wrong_format_version() {
        let dir = std::env::temp_dir().join("kengram-eval-corpus-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("future-version.json");
        std::fs::write(
            &path,
            r#"{"format_version":2,"kind":"tagger-golden-corpus","items":[
                {"id":"a","content":"x","golden":{}}]}"#,
        )
        .unwrap();
        let err = load_corpus(&path).unwrap_err();
        assert!(err.to_string().contains("format_version 2"));
    }

    #[test]
    fn rejects_wrong_kind_and_duplicate_ids_and_empty_items() {
        let dir = std::env::temp_dir().join("kengram-eval-corpus-tests");
        std::fs::create_dir_all(&dir).unwrap();

        let wrong_kind = dir.join("wrong-kind.json");
        std::fs::write(
            &wrong_kind,
            r#"{"format_version":1,"kind":"something-else","items":[
                {"id":"a","content":"x","golden":{}}]}"#,
        )
        .unwrap();
        assert!(
            load_corpus(&wrong_kind)
                .unwrap_err()
                .to_string()
                .contains("kind")
        );

        let dup = dir.join("dup-ids.json");
        std::fs::write(
            &dup,
            r#"{"format_version":1,"kind":"tagger-golden-corpus","items":[
                {"id":"a","content":"x","golden":{}},
                {"id":"a","content":"y","golden":{}}]}"#,
        )
        .unwrap();
        assert!(
            load_corpus(&dup)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        let empty = dir.join("empty.json");
        std::fs::write(
            &empty,
            r#"{"format_version":1,"kind":"tagger-golden-corpus","items":[]}"#,
        )
        .unwrap();
        assert!(
            load_corpus(&empty)
                .unwrap_err()
                .to_string()
                .contains("no items")
        );
    }

    #[test]
    fn unreviewed_items_counted_separately() {
        let dir = std::env::temp_dir().join("kengram-eval-corpus-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mixed-review.json");
        std::fs::write(
            &path,
            r#"{"format_version":1,"kind":"tagger-golden-corpus","items":[
                {"id":"a","content":"x","reviewed":true,"golden":{}},
                {"id":"b","content":"y","golden":{}}]}"#,
        )
        .unwrap();
        let corpus = load_corpus(&path).unwrap();
        assert_eq!(corpus.items.len(), 2);
        assert_eq!(corpus.reviewed_count(), 1);
        // Default metadata is an empty object; default scope is "global".
        assert_eq!(corpus.items[1].scope, "global");
        assert!(corpus.items[1].metadata.is_object());
    }
}

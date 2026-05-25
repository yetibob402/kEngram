//! Topic classification — embedding-space match against an operator-
//! curated taxonomy.
//!
//! The taxonomy is a TOML file (default location `topic-taxonomy.toml`
//! beside `kengram-extract/Cargo.toml`; configurable per-deployment via
//! `DeterministicTaggerConfig::topic_taxonomy_path`). Each entry has a
//! kebab-case label, a prose `description` (richer than the label alone
//! — this is what bge-m3 actually embeds for matching), and an optional
//! per-label `threshold` override.
//!
//! At `DeterministicTagger::new` time every description is embedded
//! once via the same embedder backing the rest of the system, and the
//! vectors are cached on [`Taxonomy`]. At tag time we embed the
//! thought's content, cosine-score against every cached description,
//! and return the top-3 labels above the (per-label or default)
//! threshold.
//!
//! Phase 2c wires this into the `Tagger` trait impl. Phase 2b only
//! ships the loader + the `classify_topics` function.

use std::path::Path;

use serde::Deserialize;

use kengram_core::{Embedder, TaggerError};

use super::{cosine_similarity, embedder_to_tagger_error};

/// Number of topics the tagger returns per thought. Hard-capped at 3
/// to match the existing `Tags::topics` convention from the LLM tagger
/// (the v9 prompt already restricted topics to 1–3 short labels).
pub const MAX_TOPICS_PER_THOUGHT: usize = 3;

/// One taxonomy entry as it appears in the TOML file. Public for
/// inspection by tests + tooling; constructed by `Taxonomy::load`.
#[derive(Debug, Clone)]
pub struct TaxonomyEntry {
    /// Kebab-case label persisted onto `tags.topics`.
    pub label: String,
    /// Pre-embedded description vector. Length matches the embedder's
    /// configured dimensions.
    pub description_vector: Vec<f32>,
    /// Per-label cosine threshold override. `None` means "use the
    /// caller-supplied default".
    pub threshold: Option<f32>,
}

/// Cached taxonomy with descriptions already embedded. Construct once
/// at `DeterministicTagger::new` time via [`Taxonomy::load`].
#[derive(Debug, Clone, Default)]
pub struct Taxonomy {
    pub entries: Vec<TaxonomyEntry>,
}

impl Taxonomy {
    /// Read the taxonomy TOML at `path`, embed every description via
    /// the supplied embedder, and return the populated cache. Loading
    /// happens once at startup; tag-time work is just cosine
    /// comparisons against the cached vectors.
    pub async fn load(
        path: impl AsRef<Path>,
        embedder: &dyn Embedder,
    ) -> Result<Self, TaggerError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|e| {
            TaggerError::Misconfigured(format!(
                "taxonomy file unreadable at {}: {e}",
                path.display(),
            ))
        })?;
        let parsed: TaxonomyFile = toml::from_str(&raw).map_err(|e| {
            TaggerError::Misconfigured(format!(
                "taxonomy file at {} is invalid TOML: {e}",
                path.display(),
            ))
        })?;

        if parsed.topics.is_empty() {
            return Err(TaggerError::Misconfigured(format!(
                "taxonomy file at {} contains no [topics.*] entries",
                path.display(),
            )));
        }

        // Stable ordering for both the embed batch and the output entries
        // — alphabetical by label. TOML map iteration order isn't
        // guaranteed; sorting makes startup behavior deterministic and
        // makes tests easier to write.
        let mut labels: Vec<&String> = parsed.topics.keys().collect();
        labels.sort();

        let descriptions: Vec<String> = labels
            .iter()
            .map(|label| parsed.topics[*label].description.clone())
            .collect();
        let vectors = embedder
            .embed(&descriptions)
            .await
            .map_err(embedder_to_tagger_error)?;
        if vectors.len() != labels.len() {
            return Err(TaggerError::MalformedResponse(format!(
                "taxonomy embed: expected {} vectors, got {}",
                labels.len(),
                vectors.len()
            )));
        }

        let entries = labels
            .into_iter()
            .zip(vectors)
            .map(|(label, description_vector)| {
                let raw = &parsed.topics[label];
                TaxonomyEntry {
                    label: label.clone(),
                    description_vector,
                    threshold: raw.threshold,
                }
            })
            .collect();

        Ok(Self { entries })
    }

    /// Empty taxonomy — useful for tests that exercise the "no labels
    /// configured" path. Production deployments always load from TOML.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Classify a thought's content into up to [`MAX_TOPICS_PER_THOUGHT`]
/// taxonomy labels. Returns the top-N labels whose cosine similarity
/// with the content is at or above the effective threshold for that
/// label (per-label override if present, otherwise `default_threshold`).
///
/// `Vec<String>` is returned in descending similarity order — strongest
/// match first. An empty vec means "no taxonomy entry crossed the
/// threshold," which the caller persists as `tags.topics = []`.
pub async fn classify_topics(
    content: &str,
    embedder: &dyn Embedder,
    taxonomy: &Taxonomy,
    default_threshold: f32,
) -> Result<Vec<String>, TaggerError> {
    if taxonomy.entries.is_empty() {
        return Ok(Vec::new());
    }

    let vectors = embedder
        .embed(&[content.to_string()])
        .await
        .map_err(embedder_to_tagger_error)?;
    let content_vec = vectors.into_iter().next().ok_or_else(|| {
        TaggerError::MalformedResponse("topics classify: empty embed result".into())
    })?;

    let mut scored: Vec<(&str, f32)> = taxonomy
        .entries
        .iter()
        .filter_map(|entry| {
            let score = cosine_similarity(&content_vec, &entry.description_vector);
            let threshold = entry.threshold.unwrap_or(default_threshold);
            (score >= threshold).then_some((entry.label.as_str(), score))
        })
        .collect();
    // Descending by score; stable sort preserves alphabetical order on ties
    // (Taxonomy::load already sorted entries alphabetically).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(MAX_TOPICS_PER_THOUGHT);

    Ok(scored.into_iter().map(|(l, _)| l.to_string()).collect())
}

// --- TOML deserialization shape ----------------------------------------

/// On-disk shape of `topic-taxonomy.toml`. The `[topics]` table maps
/// label -> entry; the entry has `description` and optional `threshold`.
#[derive(Debug, Deserialize)]
struct TaxonomyFile {
    topics: std::collections::BTreeMap<String, RawTaxonomyEntry>,
}

#[derive(Debug, Deserialize)]
struct RawTaxonomyEntry {
    description: String,
    #[serde(default)]
    threshold: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_embed::FakeEmbedder;
    use std::io::Write;

    fn write_temp_taxonomy(contents: &str) -> tempfile_like::TempPath {
        let mut tmp = tempfile_like::NamedTempFile::new();
        tmp.file.write_all(contents.as_bytes()).expect("write toml");
        tmp.into_path()
    }

    // Mini "tempfile" replacement — kengram doesn't currently take a
    // tempfile dev-dep and pulling one in just for these tests felt like
    // overkill. We construct a path under std::env::temp_dir() with a
    // pid+counter suffix so parallel tests don't collide, and clean up
    // on drop.
    mod tempfile_like {
        use std::fs::File;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        pub struct NamedTempFile {
            pub file: File,
            path: PathBuf,
        }

        pub struct TempPath(PathBuf);

        impl NamedTempFile {
            pub fn new() -> Self {
                let n = COUNTER.fetch_add(1, Ordering::SeqCst);
                let pid = std::process::id();
                let path =
                    std::env::temp_dir().join(format!("kengram-taxonomy-test-{pid}-{n}.toml"));
                let file = File::create(&path).expect("create temp file");
                Self { file, path }
            }

            pub fn into_path(self) -> TempPath {
                TempPath(self.path)
            }
        }

        impl TempPath {
            pub fn path(&self) -> &std::path::Path {
                &self.0
            }
        }

        impl Drop for TempPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    #[tokio::test]
    async fn loads_minimal_valid_taxonomy() {
        let toml = r#"
[topics.rust]
description = "rust programming language and ecosystem"

[topics.databases]
description = "relational and vector databases"
threshold = 0.55
"#;
        let path = write_temp_taxonomy(toml);
        let embedder = FakeEmbedder::new();
        let tax = Taxonomy::load(path.path(), &embedder)
            .await
            .expect("load taxonomy");
        assert_eq!(tax.entries.len(), 2);
        // Entries sorted alphabetically by label.
        assert_eq!(tax.entries[0].label, "databases");
        assert_eq!(tax.entries[0].threshold, Some(0.55));
        assert_eq!(tax.entries[1].label, "rust");
        assert!(tax.entries[1].threshold.is_none());
        // Each entry has an embedded description vector at bge-m3 dim.
        for entry in &tax.entries {
            assert_eq!(entry.description_vector.len(), 1024);
        }
    }

    #[tokio::test]
    async fn empty_taxonomy_returns_misconfigured() {
        let toml = "[topics]\n";
        let path = write_temp_taxonomy(toml);
        let embedder = FakeEmbedder::new();
        let err = Taxonomy::load(path.path(), &embedder)
            .await
            .expect_err("should reject empty taxonomy");
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn missing_taxonomy_file_returns_misconfigured() {
        let embedder = FakeEmbedder::new();
        let err = Taxonomy::load("/nonexistent/path/taxonomy.toml", &embedder)
            .await
            .expect_err("should reject missing file");
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn invalid_toml_returns_misconfigured() {
        let path = write_temp_taxonomy("this is not [valid toml");
        let embedder = FakeEmbedder::new();
        let err = Taxonomy::load(path.path(), &embedder)
            .await
            .expect_err("should reject invalid toml");
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn classify_topics_with_empty_taxonomy_returns_empty() {
        let embedder = FakeEmbedder::new();
        let tax = Taxonomy::empty();
        let out = classify_topics("any content", &embedder, &tax, 0.5)
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn classify_topics_with_impossible_threshold_returns_empty() {
        let toml = r#"
[topics.rust]
description = "rust programming"
"#;
        let path = write_temp_taxonomy(toml);
        let embedder = FakeEmbedder::new();
        let tax = Taxonomy::load(path.path(), &embedder).await.unwrap();
        // Threshold > 1.0 can't be crossed; expect empty result.
        let out = classify_topics("any content", &embedder, &tax, 1.5)
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn classify_topics_caps_at_max_topics_per_thought() {
        // Build a taxonomy with > MAX_TOPICS_PER_THOUGHT entries; with
        // threshold=0.0 every label crosses, so the cap is what limits
        // output length.
        let toml = r#"
[topics.a]
description = "a"
[topics.b]
description = "b"
[topics.c]
description = "c"
[topics.d]
description = "d"
[topics.e]
description = "e"
"#;
        let path = write_temp_taxonomy(toml);
        let embedder = FakeEmbedder::new();
        let tax = Taxonomy::load(path.path(), &embedder).await.unwrap();
        assert_eq!(tax.entries.len(), 5);
        let out = classify_topics("anything", &embedder, &tax, 0.0)
            .await
            .unwrap();
        assert_eq!(out.len(), MAX_TOPICS_PER_THOUGHT);
    }

    #[tokio::test]
    async fn classify_topics_returns_descending_by_score() {
        // Tests the sort order: when content == one specific description,
        // that entry should be first (highest cosine), regardless of
        // alphabetical position.
        let toml = r#"
[topics.alpha]
description = "alpha"
[topics.beta]
description = "beta"
[topics.gamma]
description = "gamma"
"#;
        let path = write_temp_taxonomy(toml);
        let embedder = FakeEmbedder::new();
        let tax = Taxonomy::load(path.path(), &embedder).await.unwrap();
        // Content equals one of the descriptions → that entry scores 1.0
        // and lands first in the result.
        let out = classify_topics("gamma", &embedder, &tax, 0.99)
            .await
            .unwrap();
        assert_eq!(out.first().map(String::as_str), Some("gamma"));
    }

    #[tokio::test]
    async fn per_label_threshold_overrides_default() {
        // Two entries; one has its own strict threshold of 0.99, the
        // other inherits the (very permissive) default of -2.0 (which
        // any cosine score crosses). FakeEmbedder produces pseudo-random
        // vectors whose cosines cluster near 0, so we need the negative
        // default to guarantee the "permissive" entry survives while the
        // strict one drops. The test pins that per-label thresholds
        // genuinely override the default, not just sit alongside it.
        let toml = r#"
[topics.permissive]
description = "anything goes here"

[topics.strict]
description = "very specific content"
threshold = 0.99
"#;
        let path = write_temp_taxonomy(toml);
        let embedder = FakeEmbedder::new();
        let tax = Taxonomy::load(path.path(), &embedder).await.unwrap();
        // Content does NOT equal the "strict" description; the strict
        // entry's 0.99 threshold gates it out.
        let out = classify_topics("unrelated thought content", &embedder, &tax, -2.0)
            .await
            .unwrap();
        assert!(out.contains(&"permissive".to_string()));
        assert!(!out.contains(&"strict".to_string()));
    }
}

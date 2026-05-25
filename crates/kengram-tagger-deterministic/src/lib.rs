//! kengram-tagger-deterministic — reference sidecar implementation for
//! the kengram HTTP-tagger wire contract.
//!
//! This crate ships two things:
//! - A Rust-native deterministic tagging pipeline ([`DeterministicTagger`])
//!   that produces a `TagOutput` without any LLM call.
//! - An HTTP server binary (`src/main.rs`) that wraps the pipeline in
//!   the `kengram-tagger-protocol` wire shape so kengram (or any other
//!   client) can point at it via `provider = "http"`.
//!
//! The library surface is exposed so the pipeline can also be reused
//! as an in-process Rust tagger by anyone willing to take on the
//! gline-rs + ort native build cost — this isn't kengram's default
//! deployment, but it's a reasonable embed for downstream consumers
//! who want zero network hop.
//!
//! # Pipeline (per thought)
//!
//! 1. [`preprocess::clean_for_ner`] — strip quoted spans, parenthetical
//!    "e.g."/"such as" examples, and normalize ALL-CAPS headings. Defangs
//!    the use-mention discourse failures the LLM tagger couldn't solve.
//! 2. [`ner::extract_ner`] — single gline-rs call with zero-shot labels
//!    `[person, product, organization, title, action item, task to do]`.
//!    Person spans are filtered against product/organization overlap so
//!    "Claude Desktop" → product, not "Claude" → person.
//! 3. [`dates::extract_dates`] — regex-only surface-form extraction.
//!    Deterministic by construction; the 1904 → 2004 transposition
//!    class of failure cannot happen.
//! 4. [`kind::classify_kind`] — bge-m3 embedding vs 6 prototype
//!    sentences, argmax over a cosine threshold.
//! 5. [`topics::classify_topics`] — bge-m3 cosine vs operator-
//!    curated taxonomy, top-3 above a per-label threshold.
//!
//! `DeterministicTagger::new` loads the gline-rs ONNX model from
//! `config.gliner_model_dir` and embeds the taxonomy + 6 kind
//! prototypes via the supplied [`kengram_core::Embedder`] at startup.
//! Per-thought work is one content embed + pure cosine math + one
//! gline-rs inference call.
//!
//! The `entities` field stays empty in the 5-field schema; the
//! `relations` vec stays empty because this backend doesn't extract
//! LLM-style relations.
//!
//! # Unit-test ergonomics
//!
//! Tests that only exercise taxonomy / prototypes / dates / preprocess
//! use [`DeterministicTagger::without_gliner`] which skips the ONNX
//! load; `tag()` on a `without_gliner` tagger returns
//! `TaggerError::Misconfigured`.
//!
//! # Empirical findings (2026-05-24)
//!
//! Calibrated against kengram's local Ollama (bge-m3) + a corpus of 25
//! fixtures, this backend scored 20/25 (80%) vs gemma3:12b v13 at
//! 24/25 (96%). The 5 LLM-only wins are all discourse pragmatics
//! (use-mention, role-vs-person, implicit action items, list
//! discrimination, nested quotation) — gline-rs is a token-level NER
//! model, not a discourse reasoner. The 1 deterministic-only win was
//! a GitHub handle case the v13 prompt didn't catch. See
//! `docs/tagger-improvements.md` v14 section for the methodology.

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

use gliner::model::GLiNER;
use gliner::model::params::Parameters;
use gliner::model::pipeline::span::SpanMode;
use kengram_core::{Embedder, EmbedderError, ScopeVocab, TagOutput, Tagger, TaggerError, Tags};
use orp::params::RuntimeParameters;

use self::kind::KindPrototypes;
use self::topics::Taxonomy;

pub mod dates;
pub mod kind;
pub mod ner;
pub mod preprocess;
pub mod topics;

/// Cosine similarity between two equal-length vectors. Returns 0.0 if
/// either vector has zero magnitude (degenerate case — shouldn't happen
/// with real embeddings but guards against division-by-zero on
/// hand-constructed test fixtures).
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "cosine_similarity: vectors must have equal length",
    );
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Bridge `EmbedderError` into `TaggerError` for use inside the
/// deterministic tagger pipeline. Embedder calls happen during `Tagger::tag`,
/// so failures surface as TaggerErrors to the drainer (which decides whether
/// to retry based on `is_transient()`). Mapping preserves transience:
/// timeouts / 5xx / unreachable stay transient on the tagger side.
pub(crate) fn embedder_to_tagger_error(e: EmbedderError) -> TaggerError {
    match e {
        EmbedderError::Timeout { seconds } => TaggerError::Timeout { seconds },
        EmbedderError::Unreachable(s) => TaggerError::Unreachable(s),
        EmbedderError::Backend { status, message } => TaggerError::Backend {
            status,
            body: message,
        },
        EmbedderError::MalformedResponse(s) => TaggerError::MalformedResponse(s),
        EmbedderError::DimensionMismatch { expected, got } => TaggerError::Misconfigured(format!(
            "embedder dimension mismatch: expected {expected}, got {got}",
        )),
        EmbedderError::EmptyBatch => TaggerError::Misconfigured(
            "deterministic tagger sent empty batch to embedder (internal bug)".into(),
        ),
    }
}

/// Configuration for the deterministic backend. Mirrors
/// `kengram_cli::config::DeterministicTaggerConfig` (the figment-parseable
/// shape) — kept separate so this crate doesn't depend on kengram-cli for
/// its own internal config.
#[derive(Debug, Clone)]
pub struct DeterministicTaggerConfig {
    /// Directory containing the gline-rs ONNX model assets:
    /// `<dir>/tokenizer.json` and `<dir>/onnx/model.onnx`.
    pub gliner_model_dir: PathBuf,
    /// Path to the operator-curated topic taxonomy TOML (Phase 2b will
    /// consume; Phase 2a holds it without using it).
    pub topic_taxonomy_path: PathBuf,
    /// Cosine-similarity threshold for `kind` classification. Phase 2b.
    pub kind_threshold: f32,
    /// Default cosine-similarity threshold for topic taxonomy matching.
    /// Phase 2b. Per-label overrides live in the taxonomy TOML.
    pub topic_threshold_default: f32,
    /// Stable model identifier stamped onto
    /// `thoughts.tags_extractor_model` for provenance. Conventionally
    /// `"deterministic/<gliner-model-name>"`.
    pub model_id: String,
    /// Schema-version of this backend's output contract. Bumped when the
    /// backend's behavior changes such that prior tags shouldn't be
    /// considered comparable. Starts at 1.
    pub model_version: i32,
}

impl Default for DeterministicTaggerConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_default();
        Self {
            gliner_model_dir: PathBuf::from(format!("{home}/models/gliner_small-v2.1")),
            // Relative path: resolves to `crates/kengram-extract/topic-taxonomy.toml`
            // when `cargo test` runs in this crate, and to `<run-dir>/topic-taxonomy.toml`
            // in production. Operators should set this explicitly in `kengram.toml`
            // (e.g. `topic_taxonomy_path = "/etc/kengram/topic-taxonomy.toml"`).
            topic_taxonomy_path: PathBuf::from("topic-taxonomy.toml"),
            kind_threshold: 0.55,
            // Calibrated 2026-05-24: dropped from 0.60 to 0.45 after local
            // eval against deterministic.json showed all three topic-taxonomy
            // fixtures empty at 0.60 (real bge-m3 cosine on kengram content
            // clusters in the 0.35-0.55 range). At 0.45 all three pass.
            // Per-label tighter overrides live in the taxonomy TOML.
            topic_threshold_default: 0.45,
            model_id: "deterministic/gliner-small-v2.1+interim+bge-m3".to_string(),
            model_version: 1,
        }
    }
}

/// Deterministic tagger backend. Holds the embedder, the pre-embedded
/// taxonomy + kind prototypes, and the gline-rs ONNX model loaded at
/// startup. Per-thought work is one content embedding (for kind +
/// topics) plus one gline-rs inference call.
pub struct DeterministicTagger {
    config: DeterministicTaggerConfig,
    /// Reusable embedder handle. Cloned into per-call references inside
    /// kind/topics classification. Held as `Arc<dyn Embedder>` so the
    /// same instance backing capture-time embedding is also used here —
    /// one TEI sidecar, one embedding model, one set of vectors.
    embedder: Arc<dyn Embedder>,
    /// Topic taxonomy with every entry's description already embedded.
    /// Loaded once at startup from `config.topic_taxonomy_path`.
    taxonomy: Taxonomy,
    /// The six TagKind prototype sentences pre-embedded via the same
    /// embedder. Cached at construction time so per-thought work is
    /// just one content embed plus six cosine comparisons.
    kind_prototypes: KindPrototypes,
    /// gline-rs ONNX model. `None` only for unit tests that exercise
    /// taxonomy/prototypes/dates without paying the ~200MB ONNX load —
    /// in that mode `tag()` returns `Misconfigured`. Production
    /// construction (`::new`) always populates this.
    gliner: Option<GLiNER<SpanMode>>,
}

impl DeterministicTagger {
    /// Production constructor. Loads the gline-rs ONNX model from
    /// `config.gliner_model_dir` (expecting `tokenizer.json` and
    /// `onnx/model.onnx`), embeds the taxonomy via `embedder`, and
    /// embeds the six kind prototypes — all synchronously during
    /// construction. The ONNX load is ~5-15 seconds first time;
    /// taxonomy + prototype embedding is typically a few hundred ms
    /// against a real TEI sidecar.
    ///
    /// All this one-time startup cost is amortized over the worker's
    /// lifetime: per-thought `tag()` is one content embed plus one
    /// gline-rs inference (~200ms total on CPU per Phase 0 spike).
    pub async fn new(
        config: DeterministicTaggerConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, TaggerError> {
        let taxonomy = Taxonomy::load(&config.topic_taxonomy_path, &*embedder).await?;
        let kind_prototypes = KindPrototypes::new(&*embedder).await?;
        let gliner = Some(load_gliner(&config.gliner_model_dir)?);
        Ok(Self {
            config,
            embedder,
            taxonomy,
            kind_prototypes,
            gliner,
        })
    }

    /// Test constructor that skips the gline-rs ONNX load. `tag()`
    /// returns `Misconfigured` until a real model is plugged in; useful
    /// for unit tests that only need the taxonomy / prototypes /
    /// dates / preprocess stages.
    pub async fn without_gliner(
        config: DeterministicTaggerConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, TaggerError> {
        let taxonomy = Taxonomy::load(&config.topic_taxonomy_path, &*embedder).await?;
        let kind_prototypes = KindPrototypes::new(&*embedder).await?;
        Ok(Self {
            config,
            embedder,
            taxonomy,
            kind_prototypes,
            gliner: None,
        })
    }

    /// Returns the model_id without requiring full backend construction.
    /// Used by `kengram-cli`'s server path, which only needs to stamp
    /// `pending_tags` rows at capture time and doesn't pay the cost of
    /// loading the gline-rs model + embedding the taxonomy.
    pub fn model_id_only(config: &DeterministicTaggerConfig) -> String {
        config.model_id.clone()
    }

    /// Borrow the pre-embedded taxonomy. Exposed for integration tests
    /// + the evaluation harness.
    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }

    /// Borrow the pre-embedded kind prototypes. Exposed for integration
    /// tests + the evaluation harness.
    pub fn kind_prototypes(&self) -> &KindPrototypes {
        &self.kind_prototypes
    }

    /// Borrow the embedder handle. Used by integration tests + the
    /// evaluation harness.
    pub fn embedder(&self) -> &Arc<dyn Embedder> {
        &self.embedder
    }
}

/// Load the gline-rs ONNX model from `model_dir`. Expects
/// `<model_dir>/tokenizer.json` and `<model_dir>/onnx/model.onnx`.
/// Failures surface as `TaggerError::Misconfigured` with the path so
/// the operator sees what was missing.
fn load_gliner(model_dir: &std::path::Path) -> Result<GLiNER<SpanMode>, TaggerError> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let model_path = model_dir.join("onnx").join("model.onnx");
    if !tokenizer_path.exists() {
        return Err(TaggerError::Misconfigured(format!(
            "gline-rs tokenizer.json not found at {}",
            tokenizer_path.display()
        )));
    }
    if !model_path.exists() {
        return Err(TaggerError::Misconfigured(format!(
            "gline-rs model.onnx not found at {}",
            model_path.display()
        )));
    }
    GLiNER::<SpanMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        tokenizer_path.to_str().ok_or_else(|| {
            TaggerError::Misconfigured("tokenizer_path is not valid UTF-8".into())
        })?,
        model_path
            .to_str()
            .ok_or_else(|| TaggerError::Misconfigured("model_path is not valid UTF-8".into()))?,
    )
    .map_err(|e| {
        TaggerError::Misconfigured(format!(
            "gline-rs model load failed for {}: {e}",
            model_dir.display()
        ))
    })
}

#[async_trait]
impl Tagger for DeterministicTagger {
    fn model_id(&self) -> &str {
        &self.config.model_id
    }

    fn version(&self) -> i32 {
        self.config.model_version
    }

    async fn tag(
        &self,
        thought_content: &str,
        _vocab: Option<&ScopeVocab>,
    ) -> Result<TagOutput, TaggerError> {
        // The `vocab` parameter is the LLM-backend's controlled-vocabulary
        // hint mechanism (a scope's most-frequent topics + entities the
        // LLM should prefer over coining new terms). The deterministic
        // backend's topic taxonomy IS the controlled vocabulary — there's
        // nothing for the drainer's vocab hint to add — so we deliberately
        // ignore it. Documented here so the unused arg isn't mistaken
        // for a bug.
        let gliner = self.gliner.as_ref().ok_or_else(|| {
            TaggerError::Misconfigured(
                "DeterministicTagger constructed without gline-rs (use ::new not ::without_gliner)"
                    .into(),
            )
        })?;

        // NER runs against the use-mention-cleaned text; everything else
        // runs against the original content so we don't accidentally strip
        // legitimate dates / topics-words inside quoted spans.
        let cleaned = preprocess::clean_for_ner(thought_content);
        let ner_out = ner::extract_ner(&cleaned, gliner)?;
        let dates = dates::extract_dates(thought_content);
        let kind = kind::classify_kind(
            thought_content,
            &*self.embedder,
            &self.kind_prototypes,
            self.config.kind_threshold,
        )
        .await?;
        let topics = topics::classify_topics(
            thought_content,
            &*self.embedder,
            &self.taxonomy,
            self.config.topic_threshold_default,
        )
        .await?;

        Ok(TagOutput {
            tags: Tags {
                kind,
                people: ner_out.people,
                action_items: ner_out.action_items,
                topics,
                dates_mentioned: dates,
                // 5-field schema: entities is intentionally empty under
                // the deterministic backend. The LLM-tagger v13 entities
                // pollution was the primary motivation for dropping the
                // field. See docs/tagger-improvements.md.
                entities: Vec::new(),
            },
            // Deterministic backend doesn't extract LLM-style relations.
            // Relations are still added by the existing graph tools
            // (link_thoughts MCP, link normalisation) at the storage layer.
            relations: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_embed::FakeEmbedder;

    /// Build a config pointing at the in-crate seed taxonomy file.
    /// `cargo test -p kengram-extract` runs with CWD at the crate root,
    /// so the relative `topic-taxonomy.toml` resolves to the seed file
    /// committed alongside this code.
    fn test_config() -> DeterministicTaggerConfig {
        DeterministicTaggerConfig::default()
    }

    #[test]
    fn default_config_has_sensible_values() {
        let c = DeterministicTaggerConfig::default();
        assert!((c.kind_threshold - 0.55).abs() < f32::EPSILON);
        assert!((c.topic_threshold_default - 0.45).abs() < f32::EPSILON);
        assert_eq!(c.model_version, 1);
        assert!(c.model_id.starts_with("deterministic/"));
    }

    #[test]
    fn model_id_only_doesnt_require_construction() {
        let c = DeterministicTaggerConfig::default();
        let id = DeterministicTagger::model_id_only(&c);
        assert_eq!(id, c.model_id);
    }

    #[tokio::test]
    async fn without_gliner_loads_taxonomy_and_prototypes() {
        let embedder: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new());
        let t = DeterministicTagger::without_gliner(test_config(), embedder)
            .await
            .expect("construct tagger");
        // Taxonomy was loaded and non-empty (seed file has ~30 entries).
        assert!(t.taxonomy().entries.len() >= 10);
        // Kind prototypes cover all six TagKind variants.
        assert_eq!(t.kind_prototypes().entries().len(), kind::PROTOTYPES.len());
    }

    #[tokio::test]
    async fn tag_without_gliner_returns_misconfigured() {
        // ::without_gliner is the test-only path; calling tag() on the
        // resulting tagger returns Misconfigured with a clear message
        // pointing operators at ::new.
        let embedder: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new());
        let t = DeterministicTagger::without_gliner(test_config(), embedder)
            .await
            .expect("construct tagger");
        let err = t
            .tag("any content", None)
            .await
            .expect_err("tag should fail without gliner");
        assert!(matches!(err, TaggerError::Misconfigured(_)));
    }

    #[tokio::test]
    async fn load_gliner_missing_dir_returns_misconfigured() {
        let mut cfg = test_config();
        cfg.gliner_model_dir = PathBuf::from("/nonexistent/path/to/gliner");
        let embedder: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new());
        // GLiNER<SpanMode> doesn't impl Debug, so we can't use expect_err
        // on the Result<DeterministicTagger, _>; pattern-match instead.
        match DeterministicTagger::new(cfg, embedder).await {
            Err(TaggerError::Misconfigured(_)) => {}
            Err(other) => panic!("expected Misconfigured, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}

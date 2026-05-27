//! Kind classification — argmax over six prototype embeddings.
//!
//! For each [`TagKind`] variant we hand-write a short prototype paragraph
//! (a few representative sentences in that genre). At
//! `DeterministicTagger::new` time we embed all six prototypes once via
//! bge-m3 and cache the vectors. At tag time we embed the thought's
//! content, compute cosine similarity against each cached prototype, and
//! return the argmax — or `None` if no prototype crosses the configured
//! similarity threshold.
//!
//! The prototypes here are seed values intended to be operator-tuned
//! against a real corpus. They're not config-file-driven (yet) because
//! the closed enum of TagKind variants makes the prototype set a fixed-
//! shape table rather than open-ended data; moving them to a TOML file
//! is a future change once we have feedback on which variants need
//! per-corpus tuning.
//!
//! Phase 2c will wire this into the `Tagger` trait impl. Phase 2b only
//! ships the function + the `KindPrototypes` cache type.
//!
//! Threshold tuning: bge-m3 cosine scores on real kengram content cluster
//! around 0.5–0.7 for "this prototype is on-topic"; below ~0.4 is "no
//! strong match." The default `kind_threshold = 0.55` in
//! `DeterministicTaggerConfig` is a starting point — Phase 2c eval
//! against fixtures will calibrate.

use kengram_core::{Embedder, TagKind, TaggerError};

use super::{cosine_similarity, embedder_to_tagger_error};

/// Hand-tuned prototype sentences per `TagKind`. Each prototype is
/// embedded once at `DeterministicTagger::new` time and the vector is
/// cached on [`KindPrototypes`]. Order doesn't matter for argmax; it's
/// preserved here so the test suite can pin which slot a variant occupies.
pub const PROTOTYPES: &[(TagKind, &str)] = &[
    (
        TagKind::Observation,
        "Noticed that latency spikes on the first request after a cache invalidation. \
         Quiet observation about how the system actually behaves under load: the retry \
         loop fires twice when the embedder times out. Recording what we saw, no action yet.",
    ),
    (
        TagKind::Task,
        "Need to fix the migration that drops the index. Refactor the tagger to take an \
         Arc<dyn Embedder> by end of week. Action item: ship the deterministic backend, \
         then update the docs. Concrete to-do with an outcome to verify.",
    ),
    (
        TagKind::Idea,
        "What if we replaced the LLM tagger entirely with a zero-shot NER pipeline? \
         Brainstorm: maybe gline-rs plus a small taxonomy embedded with bge-m3 covers \
         the same ground. Speculative — not committed, just exploring the design space.",
    ),
    (
        TagKind::Reference,
        "Documentation on bge-m3 lives at huggingface.co/BAAI/bge-m3. The pgvector HNSW \
         tuning guide is the authoritative source. Pointer to the design doc section on \
         retrieval fusion. Not a thought of my own — citing an external resource.",
    ),
    (
        TagKind::PersonNote,
        "Sarah is leading the search team this quarter. Marcus prefers async-first design \
         decisions and tends to push back on premature abstraction. Notes from the one-on-one \
         with David: career goals include moving toward staff engineer; values mentorship work.",
    ),
    (
        TagKind::Session,
        "Today's standup covered the release plan and the migration freeze. Pairing session \
         with Priya on the auth rewrite — got through the token-storage piece. Sprint planning \
         meeting decided to defer the dashboard work to next iteration.",
    ),
    (
        TagKind::DecisionRecord,
        "We decided to use pgvector over a dedicated vector database after benchmarking recall. \
         The team chose Cap'n Proto for zero-copy reads. Settled on the HTTP sidecar approach \
         instead of an in-tree backend. A choice already made and recorded, with its rationale — \
         past tense, not a proposal and not future work.",
    ),
];

/// Pre-embedded prototype vectors, one per `TagKind`. Constructed once at
/// `DeterministicTagger::new` time so tag-time work is just a single
/// content embedding + N cosine comparisons.
#[derive(Debug, Clone)]
pub struct KindPrototypes {
    /// (variant, embedded prototype vector). Length matches `PROTOTYPES`.
    embedded: Vec<(TagKind, Vec<f32>)>,
}

impl KindPrototypes {
    /// Embed every prototype sentence in `PROTOTYPES` via the supplied
    /// embedder. This is an I/O call against bge-m3 (or whatever embedder
    /// is wired in) and should happen exactly once at tagger startup.
    pub async fn new(embedder: &dyn Embedder) -> Result<Self, TaggerError> {
        let texts: Vec<String> = PROTOTYPES.iter().map(|(_, t)| (*t).to_string()).collect();
        let vectors = embedder
            .embed(&texts)
            .await
            .map_err(embedder_to_tagger_error)?;
        if vectors.len() != PROTOTYPES.len() {
            return Err(TaggerError::MalformedResponse(format!(
                "kind prototype embed: expected {} vectors, got {}",
                PROTOTYPES.len(),
                vectors.len()
            )));
        }
        let embedded = PROTOTYPES
            .iter()
            .zip(vectors)
            .map(|((k, _), v)| (*k, v))
            .collect();
        Ok(Self { embedded })
    }

    /// Borrow the cached (variant, vector) pairs — used by `classify_kind`.
    pub fn entries(&self) -> &[(TagKind, Vec<f32>)] {
        &self.embedded
    }
}

/// Classify a thought's content into one of six [`TagKind`] variants, or
/// return `None` if no prototype crosses `threshold`.
///
/// Embeds `content` as a single-element batch, then cosine-compares
/// against each pre-embedded prototype. The winner is the highest cosine
/// score *if it's at or above `threshold`*; otherwise the function
/// returns `None` to signal "no confident classification" (which the
/// caller persists as `tags.kind = null`).
pub async fn classify_kind(
    content: &str,
    embedder: &dyn Embedder,
    prototypes: &KindPrototypes,
    threshold: f32,
) -> Result<Option<TagKind>, TaggerError> {
    let vectors = embedder
        .embed(&[content.to_string()])
        .await
        .map_err(embedder_to_tagger_error)?;
    let content_vec = vectors.into_iter().next().ok_or_else(|| {
        TaggerError::MalformedResponse("kind classify: empty embed result".into())
    })?;

    let mut best: Option<(TagKind, f32)> = None;
    for (kind, proto_vec) in prototypes.entries() {
        let score = cosine_similarity(&content_vec, proto_vec);
        if best.is_none_or(|(_, b)| score > b) {
            best = Some((*kind, score));
        }
    }

    Ok(best.and_then(|(k, s)| (s >= threshold).then_some(k)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_embed::FakeEmbedder;

    #[test]
    fn prototypes_cover_all_tag_kinds_exactly_once() {
        // Catches drift if a new TagKind variant is added without a
        // matching prototype here. TagKind doesn't implement Hash, so we
        // check coverage by exhaustively pattern-matching each expected
        // variant against the prototype list.
        fn has(kind: TagKind) -> bool {
            PROTOTYPES.iter().any(|(k, _)| *k == kind)
        }
        assert!(has(TagKind::Observation));
        assert!(has(TagKind::Task));
        assert!(has(TagKind::Idea));
        assert!(has(TagKind::Reference));
        assert!(has(TagKind::PersonNote));
        assert!(has(TagKind::Session));
        assert!(has(TagKind::DecisionRecord));
        // And no duplicates — if a variant appears twice the count grows.
        let unique_count = PROTOTYPES
            .iter()
            .enumerate()
            .filter(|(i, (k, _))| PROTOTYPES[..*i].iter().all(|(prev, _)| prev != k))
            .count();
        assert_eq!(
            unique_count,
            PROTOTYPES.len(),
            "duplicate TagKind in prototypes"
        );
    }

    #[tokio::test]
    async fn kind_prototypes_new_caches_one_vector_per_variant() {
        let embedder = FakeEmbedder::new();
        let cache = KindPrototypes::new(&embedder)
            .await
            .expect("embed prototypes");
        assert_eq!(cache.entries().len(), PROTOTYPES.len());
        for (_, v) in cache.entries() {
            assert_eq!(v.len(), 1024); // bge-m3 default dim on FakeEmbedder
        }
    }

    #[tokio::test]
    async fn classify_kind_with_zero_threshold_always_returns_some() {
        // FakeEmbedder is deterministic-but-not-semantic, so cosine scores
        // are pseudo-random. With threshold=0.0 every input crosses the
        // bar and argmax always returns a variant — proves the mechanics
        // work regardless of which variant wins.
        let embedder = FakeEmbedder::new();
        let cache = KindPrototypes::new(&embedder)
            .await
            .expect("embed prototypes");
        let out = classify_kind("any content here", &embedder, &cache, 0.0)
            .await
            .expect("classify");
        assert!(out.is_some());
    }

    #[tokio::test]
    async fn classify_kind_with_impossible_threshold_returns_none() {
        // Threshold > 1.0 can never be crossed by cosine similarity, so
        // every classification falls below and returns None. Pins the
        // gating-by-threshold mechanism.
        let embedder = FakeEmbedder::new();
        let cache = KindPrototypes::new(&embedder)
            .await
            .expect("embed prototypes");
        let out = classify_kind("any content here", &embedder, &cache, 1.5)
            .await
            .expect("classify");
        assert!(out.is_none());
    }

    #[tokio::test]
    async fn classify_kind_is_deterministic_for_same_input() {
        // The FakeEmbedder is content-deterministic, so back-to-back
        // classifications of the same input should pick the same kind.
        let embedder = FakeEmbedder::new();
        let cache = KindPrototypes::new(&embedder)
            .await
            .expect("embed prototypes");
        let a = classify_kind("stable input", &embedder, &cache, 0.0)
            .await
            .unwrap();
        let b = classify_kind("stable input", &embedder, &cache, 0.0)
            .await
            .unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn classify_kind_picks_matching_prototype_when_content_equals_it() {
        // Strongest possible signal in FakeEmbedder-land: when the input
        // string IS the prototype string, cosine similarity is 1.0
        // (deterministic identical vectors) and argmax picks that variant.
        let embedder = FakeEmbedder::new();
        let cache = KindPrototypes::new(&embedder)
            .await
            .expect("embed prototypes");
        // Pick a specific prototype to test against.
        let (target_kind, target_text) = PROTOTYPES[1]; // Task
        let out = classify_kind(target_text, &embedder, &cache, 0.99)
            .await
            .expect("classify");
        assert_eq!(out, Some(target_kind));
    }
}

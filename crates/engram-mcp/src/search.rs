//! Read operations: `search_thoughts`, `recent_thoughts`, `get_thought`.
//!
//! `search_thoughts` is the hybrid retrieval path: vector kNN ∪ trigram
//! similarity, fused by RRF, then recency-boosted, then (M3 Phase B step 2)
//! optionally reranked by a cross-encoder model over the top
//! `candidate_pool` candidates. If the embedder is unreachable, the
//! vector leg is skipped and `vector_search_available` flips to `false`.
//! If the reranker is unreachable or not configured, the rerank stage is
//! skipped and `rerank_used` flips to `false`; trigram + vector + RRF
//! results still come back.

use engram_core::{
    DEFAULT_RECENCY_HALF_LIFE_DAYS, DEFAULT_RRF_K, Embedder, EmbeddingModel, EmbeddingStatus, Fact,
    Metadata, Scope, Source, Thought, ThoughtId, recency_boost, rrf_fuse,
};
use engram_embed::Reranker;
use sqlx::PgPool;
use time::OffsetDateTime;
use uuid::Uuid;

pub const DEFAULT_SEARCH_LIMIT: usize = 10;
pub const MAX_SEARCH_LIMIT: usize = 100;
pub const DEFAULT_TOP_K_PER_LEG: usize = 50;
/// Default `candidate_pool` for the rerank stage: retrieve top-32 via RRF +
/// recency, rerank those 32 to produce the final top-`limit`. Matches TEI's
/// default `--max-client-batch-size` of 32 so the default request shape
/// works out of the box. Larger pools give the reranker more material at
/// higher latency; per-call callers override via the `candidate_pool`
/// request field. Bumping this default would also require bumping TEI's
/// flag (warmup-time cost on CPU is significant — see docker-compose.yml).
pub const DEFAULT_RERANK_CANDIDATE_POOL: usize = 32;

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,
    pub recency_half_life_days: Option<f32>,
    /// Apply the cross-encoder rerank stage over the top `candidate_pool`
    /// post-RRF candidates. Defaults to `true` when omitted (M3 Phase B
    /// step 2 success criterion #3). Set `Some(false)` to skip rerank
    /// even when a reranker is configured (useful for A/B comparison).
    pub rerank: Option<bool>,
    /// Number of post-RRF candidates fed into the reranker. Ignored when
    /// rerank is off. Defaults to [`DEFAULT_RERANK_CANDIDATE_POOL`].
    pub candidate_pool: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub thought_id: ThoughtId,
    pub content: String,
    pub scope: Scope,
    pub source: Source,
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,
    /// Raw cosine similarity from the vector leg (`None` if not in that leg).
    pub vector_score: Option<f32>,
    /// Raw `word_similarity` from the trigram leg (`None` if not in that leg).
    pub trigram_score: Option<f32>,
    /// RRF aggregate (optionally adjusted by recency boost). Preserved
    /// across the rerank stage. `Some(_)` for every fused hit; `None`
    /// would only appear in a pathological pre-fusion path.
    pub rrf_score: Option<f32>,
    /// Calibrated absolute score from the reranker (`None` if rerank was
    /// off, unavailable, or this hit fell outside the candidate pool).
    pub rerank_score: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub vector_search_available: bool,
    /// `true` when the rerank stage actually ran. `false` when rerank was
    /// disabled per-request, no reranker was configured, or the reranker
    /// failed at runtime (in which case results come from the RRF + recency
    /// pipeline). Mirrors `vector_search_available`'s soft-fail semantics.
    pub rerank_used: bool,
}

#[derive(Debug, Clone)]
pub struct RecentRequest {
    pub scope: Option<Scope>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RecentResponse {
    pub results: Vec<Thought>,
}

#[derive(Debug, Clone)]
pub struct GetThoughtResponse {
    pub thought: Thought,
    pub embedding_status: EmbeddingStatus,
    pub embedded_at: Option<OffsetDateTime>,
    pub linked_facts: Vec<Fact>,
    /// `Some(_)` when the operator has retracted this thought via
    /// `retract_thought`. Retracted thoughts don't appear in
    /// `search_thoughts` / `recent_thoughts` / `search_facts`, and their
    /// `linked_facts` will always be empty (auto-superseded as part of
    /// retraction). `get_thought` is the audit path: ID lookup always
    /// returns the row regardless of retraction state.
    pub retracted_at: Option<OffsetDateTime>,
    pub retracted_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchFactsRequest {
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,
    pub recency_half_life_days: Option<f32>,
    pub rerank: Option<bool>,
    pub candidate_pool: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SearchFactHit {
    pub fact_id: Uuid,
    pub statement: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub confidence: f32,
    /// M3 Phase C: true for the middle confidence band
    /// (`review_queue_below ≤ confidence < min_confidence_to_store`).
    /// Consumers may filter or de-emphasize flagged rows.
    pub flagged: bool,
    pub source_thought_id: ThoughtId,
    pub source_thought_content: String,
    pub source_thought_scope: Scope,
    pub source_thought_created_at: OffsetDateTime,
    pub vector_score: Option<f32>,
    pub trigram_score: Option<f32>,
    pub rrf_score: Option<f32>,
    pub rerank_score: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchFactsResponse {
    pub results: Vec<SearchFactHit>,
    /// `false` when the embedder failed (couldn't embed the query, or the
    /// vector kNN query errored) and the response is trigram-only. Mirrors
    /// `SearchResponse.vector_search_available` so clients can warn about
    /// degraded retrieval.
    pub vector_search_available: bool,
    /// See [`SearchResponse::rerank_used`] for semantics.
    pub rerank_used: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("query must be non-empty")]
    EmptyQuery,

    #[error("limit out of bounds: {got} (must be 1..={max})")]
    LimitOutOfBounds { got: usize, max: usize },

    #[error("thought not found")]
    NotFound,

    #[error("storage error: {0}")]
    Storage(#[from] engram_storage::StorageError),
}

pub async fn search_thoughts(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    request: SearchRequest,
) -> Result<SearchResponse, ReadError> {
    if request.query.is_empty() {
        return Err(ReadError::EmptyQuery);
    }
    let limit = request.limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
    if limit == 0 || limit > MAX_SEARCH_LIMIT {
        return Err(ReadError::LimitOutOfBounds {
            got: limit,
            max: MAX_SEARCH_LIMIT,
        });
    }
    let scope_filter = request.scope.as_ref().map(Scope::as_str);

    // Vector leg (soft-fail to empty + flag).
    let (vector_hits, vector_search_available) = match embedder.embed(std::slice::from_ref(&request.query)).await {
        Ok(mut vectors) => {
            let v = vectors
                .pop()
                .expect("non-empty input must yield at least one vector");
            match engram_storage::search_vector_knn(
                pool,
                v,
                embedder.model(),
                scope_filter,
                DEFAULT_TOP_K_PER_LEG as i64,
            )
            .await
            {
                Ok(hits) => (hits, true),
                Err(e) => {
                    tracing::warn!(error = %e, "vector kNN query failed; falling back to trigram only");
                    (vec![], false)
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "embedder failed to embed query; falling back to trigram only");
            (vec![], false)
        }
    };

    // Trigram leg (errors propagate).
    let trigram_hits = engram_storage::search_trigram(
        pool,
        &request.query,
        scope_filter,
        DEFAULT_TOP_K_PER_LEG as i64,
    )
    .await?;

    // RRF fuse → recency boost.
    let mut fused = rrf_fuse(vec![vector_hits, trigram_hits], DEFAULT_RRF_K);
    let half_life = request
        .recency_half_life_days
        .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_DAYS);
    recency_boost(&mut fused, half_life, OffsetDateTime::now_utc());

    // Optional rerank stage: feed the top `candidate_pool` post-RRF hits
    // into the cross-encoder, replace `score` with the rerank score, and
    // re-sort by it. Soft-fails to RRF order on transient errors.
    let rerank_enabled = request.rerank.unwrap_or(true);
    let candidate_pool = request
        .candidate_pool
        .unwrap_or(DEFAULT_RERANK_CANDIDATE_POOL);
    let rerank_used = match (rerank_enabled, reranker) {
        (true, Some(rr)) => {
            apply_rerank_to_thought_hits(rr, &request.query, &mut fused, candidate_pool).await
        }
        _ => false,
    };

    let results: Vec<SearchHit> = fused
        .into_iter()
        .take(limit)
        .map(|h| SearchHit {
            thought_id: h.thought.id,
            content: h.thought.content,
            scope: h.thought.scope,
            source: h.thought.source,
            created_at: h.thought.created_at,
            metadata: h.thought.metadata,
            vector_score: h.vector_score,
            trigram_score: h.trigram_score,
            rrf_score: h.rrf_score,
            rerank_score: h.rerank_score,
        })
        .collect();

    Ok(SearchResponse {
        results,
        vector_search_available,
        rerank_used,
    })
}

/// Run the cross-encoder rerank stage over the top `candidate_pool` hits.
/// On success, mutates `hits` in place: rerank scores are written to
/// `rerank_score`; the un-reranked tail is **truncated** so the response
/// contains only the reranker's verdict on the candidate pool. Re-sorts
/// by rerank score descending. `rrf_score` is preserved (set by
/// [`engram_core::search::rrf_fuse`]) so consumers can compare RRF-only
/// ordering against rerank ordering.
///
/// Truncating is necessary because rerank scores and RRF+recency scores
/// aren't on the same scale: a small cross-encoder like MiniLM produces
/// scores in [0, 1] with most candidates well below 0.01, while RRF+recency
/// caps at ~0.033. Mixing them in a single sort lets the un-reranked tail
/// outrank the reranker's verdict — defeating the purpose of running the
/// reranker.
///
/// Returns `true` if rerank actually ran; `false` on transient failure
/// (logged + soft-fall-through to the RRF order). Non-transient errors
/// are also treated as soft failures here — the search request should
/// not 500 because the reranker is misconfigured.
async fn apply_rerank_to_thought_hits(
    reranker: &dyn Reranker,
    query: &str,
    hits: &mut Vec<engram_core::Hit>,
    candidate_pool: usize,
) -> bool {
    if hits.is_empty() {
        return false;
    }
    let pool_len = candidate_pool.min(hits.len());
    let candidates: Vec<&str> = hits[..pool_len].iter().map(|h| h.thought.content.as_str()).collect();
    let scores = match reranker.rerank(query, &candidates).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                error = %e,
                transient = e.is_transient(),
                "reranker failed; falling back to RRF + recency order",
            );
            return false;
        }
    };
    // Map back to hits via RerankScore.index. `rrf_score` was already set
    // by rrf_fuse / recency_boost upstream; we only write rerank_score.
    for s in scores {
        if let Some(hit) = hits.get_mut(s.index) {
            hit.rerank_score = Some(s.score);
        }
    }
    // Drop the un-reranked tail and sort the reranked candidates by
    // rerank_score descending. Ties fall back to rrf_score.
    hits.truncate(pool_len);
    hits.sort_by(|a, b| {
        let av = a.rerank_score.unwrap_or(f32::MIN);
        let bv = b.rerank_score.unwrap_or(f32::MIN);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    true
}

pub async fn recent_thoughts(
    pool: &PgPool,
    request: RecentRequest,
) -> Result<RecentResponse, ReadError> {
    let limit = request.limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
    if limit == 0 || limit > MAX_SEARCH_LIMIT {
        return Err(ReadError::LimitOutOfBounds {
            got: limit,
            max: MAX_SEARCH_LIMIT,
        });
    }
    let scope_filter = request.scope.as_ref().map(Scope::as_str);

    let results = engram_storage::recent_thoughts(pool, scope_filter, limit as i64).await?;
    Ok(RecentResponse { results })
}

pub async fn get_thought(
    pool: &PgPool,
    model: &EmbeddingModel,
    thought_id: ThoughtId,
) -> Result<GetThoughtResponse, ReadError> {
    let prov = engram_storage::fetch_thought_with_provenance(pool, thought_id, model).await?;
    let prov = prov.ok_or(ReadError::NotFound)?;
    let linked_facts = engram_storage::list_active_facts_for_thought(pool, thought_id).await?;
    Ok(GetThoughtResponse {
        thought: prov.thought,
        embedding_status: prov.embedding_status,
        embedded_at: prov.embedded_at,
        linked_facts,
        retracted_at: prov.retracted_at,
        retracted_reason: prov.retracted_reason,
    })
}

/// Search over `facts` with the same hybrid pipeline as `search_thoughts`:
/// vector kNN over fact embeddings ∪ trigram similarity over
/// `fact.statement` (and the (S, P, O) triple), fused via RRF, recency-
/// boosted, optionally reranked. Fact embeddings landed in M3 Phase B
/// step 1 (2026-05-14); the cross-encoder rerank stage landed in step 2
/// (2026-05-15). The recency boost is keyed on `source_thought_created_at`
/// — the thought's recency is a better freshness signal for a fact than
/// the fact's own `created_at` (which is when the reflector ran, not when
/// the underlying thought was captured). Soft-fails to trigram only when
/// the embedder is unreachable (signaled via `vector_search_available`),
/// and to RRF + recency order when the reranker fails (signaled via
/// `rerank_used`).
pub async fn search_facts(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    request: SearchFactsRequest,
) -> Result<SearchFactsResponse, ReadError> {
    if request.query.is_empty() {
        return Err(ReadError::EmptyQuery);
    }
    let limit = request.limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
    if limit == 0 || limit > MAX_SEARCH_LIMIT {
        return Err(ReadError::LimitOutOfBounds {
            got: limit,
            max: MAX_SEARCH_LIMIT,
        });
    }
    let scope_filter = request.scope.as_ref().map(Scope::as_str);

    // Vector leg (soft-fail to empty + flag), mirroring `search_thoughts`.
    let (vector_hits, vector_search_available) =
        match embedder.embed(std::slice::from_ref(&request.query)).await {
            Ok(mut vectors) => {
                let v = vectors
                    .pop()
                    .expect("non-empty input must yield at least one vector");
                match engram_storage::search_facts_vector_knn(
                    pool,
                    v,
                    embedder.model(),
                    scope_filter,
                    DEFAULT_TOP_K_PER_LEG as i64,
                )
                .await
                {
                    Ok(hits) => (hits, true),
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "fact vector kNN query failed; falling back to trigram only"
                        );
                        (vec![], false)
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "embedder failed to embed query for facts; falling back to trigram only"
                );
                (vec![], false)
            }
        };

    let trigram_hits = engram_storage::search_facts_trigram(
        pool,
        &request.query,
        scope_filter,
        DEFAULT_TOP_K_PER_LEG as i64,
    )
    .await?;

    // RRF fuse → recency boost → truncate. `rrf_fuse` in engram-core is
    // hardcoded for `Hit { thought, score }`; the fact-side fuse is the
    // same algorithm keyed on `fact.id` instead. Inline here rather than
    // generic-ifying engram-core's primitive — Phase B step 1 doesn't yet
    // need fact-aware fusion anywhere else.
    let mut fused = rrf_fuse_facts(vec![vector_hits, trigram_hits], DEFAULT_RRF_K);
    let half_life = request
        .recency_half_life_days
        .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_DAYS);
    if half_life > 0.0 {
        let now = OffsetDateTime::now_utc();
        for h in fused.iter_mut() {
            let age_secs = (now - h.source_thought_created_at).whole_seconds() as f32;
            let age_days = age_secs / 86_400.0;
            let factor = 0.5_f32.powf(age_days / half_life);
            if let Some(s) = h.rrf_score {
                h.rrf_score = Some(s * factor);
            }
        }
        fused.sort_by(|a, b| {
            let av = a.rrf_score.unwrap_or(0.0);
            let bv = b.rrf_score.unwrap_or(0.0);
            bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Optional rerank stage over the top `candidate_pool` post-recency
    // hits. Cross-encoder feeds on the fact's statement (the canonical
    // natural-language form of the claim); SPO triple isn't passed.
    let rerank_enabled = request.rerank.unwrap_or(true);
    let candidate_pool = request
        .candidate_pool
        .unwrap_or(DEFAULT_RERANK_CANDIDATE_POOL);
    let rerank_used = match (rerank_enabled, reranker) {
        (true, Some(rr)) => {
            apply_rerank_to_fact_hits(rr, &request.query, &mut fused, candidate_pool).await
        }
        _ => false,
    };

    let results: Vec<SearchFactHit> = fused
        .into_iter()
        .take(limit)
        .map(|h| SearchFactHit {
            fact_id: h.fact.id,
            statement: h.fact.statement,
            subject: h.fact.subject,
            predicate: h.fact.predicate,
            object: h.fact.object,
            confidence: h.fact.confidence,
            flagged: h.fact.flagged,
            source_thought_id: h.fact.source_thought_id,
            source_thought_content: h.source_thought_content,
            source_thought_scope: h.source_thought_scope,
            source_thought_created_at: h.source_thought_created_at,
            vector_score: h.vector_score,
            trigram_score: h.trigram_score,
            rrf_score: h.rrf_score,
            rerank_score: h.rerank_score,
        })
        .collect();

    Ok(SearchFactsResponse {
        results,
        vector_search_available,
        rerank_used,
    })
}

/// Fact-side counterpart of `apply_rerank_to_thought_hits`. Cross-encoder
/// scores each fact's `statement` against the query; on success mutates
/// hits in place (writes `rerank_score`), truncates the un-reranked tail
/// (see thought-side doc for why), and re-sorts the reranked candidate
/// pool by `rerank_score` descending. `rrf_score` was already set by
/// the upstream `rrf_fuse_facts` + recency-boost pass.
async fn apply_rerank_to_fact_hits(
    reranker: &dyn Reranker,
    query: &str,
    hits: &mut Vec<engram_storage::FactHit>,
    candidate_pool: usize,
) -> bool {
    if hits.is_empty() {
        return false;
    }
    let pool_len = candidate_pool.min(hits.len());
    let candidates: Vec<&str> = hits[..pool_len]
        .iter()
        .map(|h| h.fact.statement.as_str())
        .collect();
    let candidates_len = candidates.len();
    let scores = match reranker.rerank(query, &candidates).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                error = %e,
                transient = e.is_transient(),
                "reranker failed; falling back to RRF + recency order",
            );
            return false;
        }
    };
    let scores_len = scores.len();
    let mut applied = 0usize;
    let mut out_of_range = 0usize;
    for s in scores {
        if let Some(hit) = hits.get_mut(s.index) {
            hit.rerank_score = Some(s.score);
            applied += 1;
        } else {
            out_of_range += 1;
        }
    }
    tracing::debug!(
        candidates = candidates_len,
        scores_returned = scores_len,
        applied,
        out_of_range,
        "fact rerank stage applied",
    );
    hits.truncate(pool_len);
    hits.sort_by(|a, b| {
        let av = a.rerank_score.unwrap_or(f32::MIN);
        let bv = b.rerank_score.unwrap_or(f32::MIN);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    true
}

/// Fact-side analogue of `engram_core::search::rrf_fuse`, keyed on
/// `fact.id` instead of `thought.id`. Same reciprocal-rank-fusion algorithm
/// (`rrf_score = Σ 1 / (k + rank_i)`); same per-leg score preservation
/// rule (an input's `Some(_)` always wins over an existing `None`). Inline
/// because `rrf_fuse` in engram-core is `Hit`-specific and the fact side
/// has its own `FactHit` shape.
fn rrf_fuse_facts(
    rankings: Vec<Vec<engram_storage::FactHit>>,
    k: f32,
) -> Vec<engram_storage::FactHit> {
    use std::collections::HashMap;
    let mut acc: HashMap<uuid::Uuid, engram_storage::FactHit> = HashMap::new();
    for ranking in rankings {
        for (i, hit) in ranking.into_iter().enumerate() {
            let rank = (i + 1) as f32;
            let contribution = 1.0 / (k + rank);
            let id = hit.fact.id;
            match acc.get_mut(&id) {
                Some(existing) => {
                    let current = existing.rrf_score.unwrap_or(0.0);
                    existing.rrf_score = Some(current + contribution);
                    if existing.vector_score.is_none() {
                        existing.vector_score = hit.vector_score;
                    }
                    if existing.trigram_score.is_none() {
                        existing.trigram_score = hit.trigram_score;
                    }
                }
                None => {
                    let mut merged = hit;
                    merged.rrf_score = Some(contribution);
                    acc.insert(id, merged);
                }
            }
        }
    }
    let mut fused: Vec<engram_storage::FactHit> = acc.into_values().collect();
    fused.sort_by(|a, b| {
        let av = a.rrf_score.unwrap_or(0.0);
        let bv = b.rrf_score.unwrap_or(0.0);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use crate::drain::drain_pending_embeddings;
    use engram_embed::{FakeBehavior, FakeEmbedder};

    const TEST_MODEL_ID: &str = "bge-m3:1024";

    /// Capture a thought — Phase B leaves it queued, not embedded.
    async fn cap(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        capture(
            pool,
            TEST_MODEL_ID,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new(scope).unwrap()),
                metadata: None,
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    /// Capture and immediately drain — for tests that need vector search to
    /// work. Mirrors what `engram serve` + `engram worker` does in production.
    async fn cap_and_drain(
        pool: &PgPool,
        embedder: &dyn Embedder,
        content: &str,
        scope: &str,
    ) -> ThoughtId {
        let id = cap(pool, content, scope).await;
        drain_pending_embeddings(pool, embedder, 16).await.unwrap();
        id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_round_trip_with_fake_embedder(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        // Drain so the thoughts have embedding rows for vector kNN to find.
        let id_a = cap_and_drain(&pool, &embedder, "alpha", "global").await;
        let _id_b = cap_and_drain(&pool, &embedder, "beta", "global").await;

        let resp = search_thoughts(
            &pool,
            &embedder, None,
            SearchRequest {
                query: "alpha".to_string(),
                scope: None,
                limit: Some(10),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();

        assert!(resp.vector_search_available);
        assert!(!resp.results.is_empty());
        assert_eq!(resp.results[0].thought_id, id_a);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_degrades_when_embedder_fails(pool: PgPool) {
        // Capture (no drain) → only trigram leg can find the thought.
        let id = cap(&pool, "the tcgplayer integration was painful", "work").await;

        // Search with a failing embedder.
        let bad = FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Unreachable);
        let resp = search_thoughts(
            &pool,
            &bad, None,
            SearchRequest {
                query: "tcgplayer".to_string(),
                scope: None,
                limit: Some(10),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();

        assert!(!resp.vector_search_available);
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].thought_id, id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_empty_query_errors(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let err = search_thoughts(
            &pool,
            &embedder, None,
            SearchRequest {
                query: String::new(),
                scope: None,
                limit: None,
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReadError::EmptyQuery));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_limit_out_of_bounds_errors(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let err = search_thoughts(
            &pool,
            &embedder, None,
            SearchRequest {
                query: "x".to_string(),
                scope: None,
                limit: Some(1000),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReadError::LimitOutOfBounds { got: 1000, .. }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_respects_scope(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        cap(&pool, "tcgplayer work", "work").await;
        cap(&pool, "tcgplayer personal", "personal").await;

        let resp = search_thoughts(
            &pool,
            &embedder, None,
            SearchRequest {
                query: "tcgplayer".to_string(),
                scope: Some(Scope::new("work").unwrap()),
                limit: Some(10),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.results.iter().all(|h| h.scope.as_str() == "work"));
        assert_eq!(resp.results.len(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_returns_newest_first(pool: PgPool) {
        cap(&pool, "first", "global").await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        cap(&pool, "second", "global").await;

        let resp = recent_thoughts(
            &pool,
            RecentRequest {
                scope: None,
                limit: Some(10),
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].content, "second");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_indexed_after_drain(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap_and_drain(&pool, &embedder, "hello", "global").await;
        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.embedding_status, EmbeddingStatus::Indexed);
        assert!(resp.embedded_at.is_some());
        assert_eq!(resp.thought.content, "hello");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_pending_when_unembedded(pool: PgPool) {
        // Capture-without-drain leaves the thought pending. The Phase B
        // success criterion is exactly that: capture returns immediately;
        // vector readiness waits for the worker tick.
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "hello", "global").await;
        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.embedding_status, EmbeddingStatus::Pending);
        assert!(resp.embedded_at.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_not_found(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let err = get_thought(&pool, embedder.model(), ThoughtId::new())
            .await
            .unwrap_err();
        assert!(matches!(err, ReadError::NotFound));
    }

    // -- M2 Phase D: search_facts + linked_facts on get_thought ----------

    async fn insert_test_fact(
        pool: &PgPool,
        thought_id: ThoughtId,
        scope_str: &str,
        statement: &str,
        triple: (Option<&str>, Option<&str>, Option<&str>),
        confidence: f32,
    ) -> uuid::Uuid {
        let scope = Scope::new(scope_str).unwrap();
        let run_id = engram_storage::start_run(pool, "fake/extractor", 1, None)
            .await
            .unwrap();
        engram_storage::insert_fact(
            pool,
            engram_storage::NewFact {
                scope: &scope,
                statement,
                subject: triple.0,
                predicate: triple.1,
                object: triple.2,
                source_thought_id: thought_id,
                extractor_model: "fake/extractor",
                extractor_version: 1,
                source_run_id: Some(run_id),
                confidence,
                flagged: false,
            },
        )
        .await
        .unwrap()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_returns_results_with_source_thought_content(pool: PgPool) {
        let id = cap(&pool, "Engram uses pgvector for storage", "global").await;
        insert_test_fact(
            &pool, id, "global", "Engram uses pgvector",
            (Some("Engram"), Some("uses"), Some("pgvector")), 0.92,
        )
        .await;

        let resp = search_facts(
            &pool,
            &FakeEmbedder::new(), None,
            SearchFactsRequest {
                query: "pgvector".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0), // disable boost for deterministic ordering
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.results.len(), 1);
        let hit = &resp.results[0];
        assert_eq!(hit.statement, "Engram uses pgvector");
        assert_eq!(hit.subject.as_deref(), Some("Engram"));
        assert_eq!(hit.source_thought_content, "Engram uses pgvector for storage");
        assert_eq!(hit.source_thought_scope.as_str(), "global");
        assert_eq!(hit.source_thought_id, id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_filters_superseded_facts(pool: PgPool) {
        let id = cap(&pool, "thought about widgets", "global").await;
        let fact_id = insert_test_fact(
            &pool, id, "global", "widgets are useful", (None, None, None), 0.9,
        )
        .await;

        // Visible before supersede.
        let before = search_facts(
            &pool,
            &FakeEmbedder::new(), None,
            SearchFactsRequest {
                query: "widgets".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(before.results.len(), 1);

        engram_storage::supersede_fact(&pool, fact_id, None).await.unwrap();

        let after = search_facts(
            &pool,
            &FakeEmbedder::new(), None,
            SearchFactsRequest {
                query: "widgets".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(after.results.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_empty_query_errors(pool: PgPool) {
        let err = search_facts(
            &pool,
            &FakeEmbedder::new(), None,
            SearchFactsRequest {
                query: String::new(),
                scope: None,
                limit: None,
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReadError::EmptyQuery));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_limit_out_of_bounds_errors(pool: PgPool) {
        let err = search_facts(
            &pool,
            &FakeEmbedder::new(), None,
            SearchFactsRequest {
                query: "x".to_string(),
                scope: None,
                limit: Some(1000),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReadError::LimitOutOfBounds { got: 1000, .. }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_respects_scope(pool: PgPool) {
        let w = cap(&pool, "thought work", "work").await;
        let p = cap(&pool, "thought personal", "personal").await;
        insert_test_fact(&pool, w, "work", "widget alpha", (None, None, None), 0.9).await;
        insert_test_fact(&pool, p, "personal", "widget beta", (None, None, None), 0.9).await;

        let resp = search_facts(
            &pool,
            &FakeEmbedder::new(), None,
            SearchFactsRequest {
                query: "widget".to_string(),
                scope: Some(Scope::new("work").unwrap()),
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].statement, "widget alpha");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_returns_empty_linked_facts_when_none(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "no facts here", "global").await;
        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert!(resp.linked_facts.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_includes_active_linked_facts(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "with facts", "global").await;
        insert_test_fact(&pool, id, "global", "first fact", (None, None, None), 0.9).await;
        insert_test_fact(&pool, id, "global", "second fact", (None, None, None), 0.85).await;

        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.linked_facts.len(), 2);
        let statements: Vec<&str> = resp.linked_facts.iter().map(|f| f.statement.as_str()).collect();
        assert!(statements.contains(&"first fact"));
        assert!(statements.contains(&"second fact"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_excludes_superseded_linked_facts(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "mixed facts", "global").await;
        let _alive = insert_test_fact(&pool, id, "global", "alive", (None, None, None), 0.9).await;
        let doomed = insert_test_fact(&pool, id, "global", "doomed", (None, None, None), 0.9).await;
        engram_storage::supersede_fact(&pool, doomed, None).await.unwrap();

        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.linked_facts.len(), 1);
        assert_eq!(resp.linked_facts[0].statement, "alive");
    }

    // -- M3 Phase B step 1: search_facts vector leg ---------------------------

    /// Embed a fact via the embed pipeline: insert the fact, enqueue, drain.
    /// Returns the new fact's id.
    async fn insert_fact_and_embed(
        pool: &PgPool,
        embedder: &dyn Embedder,
        thought_id: engram_core::ThoughtId,
        scope: &str,
        statement: &str,
    ) -> uuid::Uuid {
        let fact_id = insert_test_fact(
            pool,
            thought_id,
            scope,
            statement,
            (None, None, None),
            0.9,
        )
        .await;
        engram_storage::enqueue_embedding(
            pool,
            engram_storage::target::FACT,
            fact_id,
            &embedder.model().id,
        )
        .await
        .unwrap();
        drain_pending_embeddings(pool, embedder, 16).await.unwrap();
        fact_id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_returns_results_from_vector_leg(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor thought", "global").await;
        let fact_id = insert_fact_and_embed(
            &pool,
            &embedder,
            id,
            "global",
            "semantic-only payload that wouldn't trigram-match the query terms",
        )
        .await;

        // Query has zero token overlap with the statement, so trigram alone
        // wouldn't surface this fact. FakeEmbedder produces a deterministic
        // vector per query string; if both sides go through it, the query
        // vector and fact vector live in the same space — the kNN finds it.
        let resp = search_facts(
            &pool,
            &embedder, None,
            SearchFactsRequest {
                query: "completely unrelated lexically".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.vector_search_available);
        assert!(
            resp.results.iter().any(|h| h.fact_id == fact_id),
            "vector leg should surface the embedded fact even with no lexical overlap"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_response_carries_vector_search_available_flag(pool: PgPool) {
        let healthy = FakeEmbedder::new();
        cap(&pool, "a thought", "global").await;
        let ok_resp = search_facts(
            &pool,
            &healthy, None,
            SearchFactsRequest {
                query: "anything".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(ok_resp.vector_search_available);

        let broken = FakeEmbedder::always_failing(EmbeddingModel::bge_m3(), FakeBehavior::Unreachable);
        let degraded_resp = search_facts(
            &pool,
            &broken, None,
            SearchFactsRequest {
                query: "anything".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(!degraded_resp.vector_search_available);
    }

    // -- M3 Phase B step 2: rerank stage integration -------------------------

    use engram_embed::{FakeReranker, FakeRerankerBehavior, FakeRerankerScoring};

    /// When a reranker is provided and rerank is enabled (default Some(true)),
    /// the search response carries `rerank_used: true` and every result has
    /// a populated `rerank_score`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_reranks_when_enabled(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        insert_test_fact(&pool, id, "global", "fact one", (None, None, None), 0.9).await;
        insert_test_fact(&pool, id, "global", "fact two", (None, None, None), 0.9).await;
        // Position-descending scoring: fact at index 0 wins. The exact
        // ordering doesn't matter for this test; we're verifying rerank
        // ran at all.
        let reranker = FakeReranker::new();

        let resp = search_facts(
            &pool,
            &embedder,
            Some(&reranker),
            SearchFactsRequest {
                query: "fact".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.rerank_used);
        assert!(!resp.results.is_empty());
        for hit in &resp.results {
            assert!(hit.rerank_score.is_some(), "every result should carry a rerank score");
        }
        // FakeReranker recorded the call → fact statements were passed as candidates.
        let call = reranker.last_call().unwrap();
        assert_eq!(call.query, "fact");
        assert!(!call.candidates.is_empty());
    }

    /// Explicit opt-out: `rerank: Some(false)` skips the stage even when a
    /// reranker is available. Response carries `rerank_used: false` and no
    /// hit carries a `rerank_score`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_skips_rerank_when_disabled_per_request(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        insert_test_fact(&pool, id, "global", "fact one", (None, None, None), 0.9).await;
        let reranker = FakeReranker::new();

        let resp = search_facts(
            &pool,
            &embedder,
            Some(&reranker),
            SearchFactsRequest {
                query: "fact".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(!resp.rerank_used);
        for hit in &resp.results {
            assert!(hit.rerank_score.is_none());
        }
        // Reranker was NOT called.
        assert!(reranker.last_call().is_none());
    }

    /// No reranker configured (None passed): search still works, response
    /// carries `rerank_used: false`, no error. This is the silent-disable
    /// path for `engram.toml` without a `[reranker]` section.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_falls_through_when_no_reranker_configured(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        insert_test_fact(&pool, id, "global", "fact one", (None, None, None), 0.9).await;

        let resp = search_facts(
            &pool,
            &embedder,
            None, // no reranker configured
            SearchFactsRequest {
                query: "fact".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(!resp.rerank_used);
        for hit in &resp.results {
            assert!(hit.rerank_score.is_none());
        }
    }

    /// Reranker failure soft-fails to the RRF + recency order rather than
    /// returning an error. Response carries `rerank_used: false`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_soft_fails_when_reranker_errors(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        insert_test_fact(&pool, id, "global", "fact one", (None, None, None), 0.9).await;
        let broken = FakeReranker::always_failing(FakeRerankerBehavior::Unreachable);

        let resp = search_facts(
            &pool,
            &embedder,
            Some(&broken),
            SearchFactsRequest {
                query: "fact".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(!resp.rerank_used);
        assert!(!resp.results.is_empty(), "RRF results should still come back");
    }

    /// Phase B step 2's regression target shape: when the reranker boosts
    /// a specific candidate by substring, that candidate should land at
    /// the top of the response even when RRF + trigram alone wouldn't put
    /// it there. All three facts are embedded so they all reach the
    /// vector-leg candidate pool regardless of trigram overlap with the
    /// query.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_rerank_reorders_candidate_pool(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        // Three facts; trigram favors the one matching "keyword".
        let f_lex = insert_test_fact(&pool, id, "global", "lexical match keyword", (None, None, None), 0.9).await;
        let f_alpha = insert_test_fact(&pool, id, "global", "alpha fact", (None, None, None), 0.9).await;
        let f_nix = insert_test_fact(&pool, id, "global", "Nix is reproducible", (None, None, None), 0.9).await;
        // Embed all three so they reach the vector-leg candidate pool.
        for fact_id in [f_lex, f_alpha, f_nix] {
            engram_storage::enqueue_embedding(
                &pool,
                engram_storage::target::FACT,
                fact_id,
                &embedder.model().id,
            )
            .await
            .unwrap();
        }
        drain_pending_embeddings(&pool, &embedder, 16).await.unwrap();

        // Reranker boosts whichever candidate contains "Nix".
        let reranker = FakeReranker::with_scoring(FakeRerankerScoring::SubstringBoost {
            needle: "Nix".to_string(),
        });

        let resp = search_facts(
            &pool,
            &embedder,
            Some(&reranker),
            SearchFactsRequest {
                query: "keyword".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(resp.rerank_used);
        // After rerank, the Nix fact (boosted to 1.0) wins.
        assert_eq!(resp.results[0].statement, "Nix is reproducible");
        assert_eq!(resp.results[0].rerank_score, Some(1.0));
    }

    /// Regression for the score-scale-collision bug surfaced during the
    /// 2026-05-15 step 2 dogfood: when rerank scores are all numerically
    /// smaller than the un-reranked tail's RRF+recency scores, the old
    /// sort-the-whole-vec approach let the tail outrank the reranker's
    /// verdict. The fix truncates fused to the candidate pool after rerank
    /// fires. This test uses `PositionAscending` scoring so the reranker
    /// produces *very small* scores (0.0, 0.05, 0.10, ...) — far below
    /// typical RRF-recency scores. The response must still contain only
    /// reranked candidates (every hit carries `rerank_score`), not the
    /// un-reranked tail.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_rerank_truncates_un_reranked_tail(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        // Insert 6 facts, then embed all of them so the vector leg returns
        // them all. With candidate_pool=3, the rerank fires on the top 3
        // and the bottom 3 should be dropped.
        let mut fact_ids = Vec::new();
        for i in 0..6 {
            let fid = insert_test_fact(
                &pool,
                id,
                "global",
                &format!("fact number {i}"),
                (None, None, None),
                0.9,
            )
            .await;
            fact_ids.push(fid);
            engram_storage::enqueue_embedding(
                &pool,
                engram_storage::target::FACT,
                fid,
                &embedder.model().id,
            )
            .await
            .unwrap();
        }
        drain_pending_embeddings(&pool, &embedder, 16).await.unwrap();

        // PositionAscending: rerank score for index 0 is 0.0, index 1 is 0.05,
        // etc. These are numerically *below* every plausible RRF-recency
        // score (which max out around 0.033). Pre-fix, the un-reranked tail
        // would sort above the reranked candidates.
        let reranker = FakeReranker::with_scoring(FakeRerankerScoring::PositionAscending);

        let resp = search_facts(
            &pool,
            &embedder,
            Some(&reranker),
            SearchFactsRequest {
                query: "fact".to_string(),
                scope: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: Some(3),
            },
        )
        .await
        .unwrap();

        assert!(resp.rerank_used);
        // All returned hits must be reranked (no un-reranked tail leak).
        assert_eq!(
            resp.results.len(),
            3,
            "after rerank, only the candidate pool's worth of hits should remain"
        );
        for hit in &resp.results {
            assert!(
                hit.rerank_score.is_some(),
                "every returned hit must carry a rerank_score"
            );
            assert!(
                hit.rrf_score.is_some(),
                "every reranked hit must carry rrf_score (captured pre-rerank)"
            );
        }
    }

    /// Per-leg scores are populated on the response: a hit that matched
    /// via vector should have `vector_score: Some(_)`; a hit that matched
    /// via trigram should have `trigram_score: Some(_)`.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_response_carries_per_leg_scores(pool: PgPool) {
        let embedder = FakeEmbedder::new();
        let id = cap(&pool, "anchor", "global").await;
        // Embed a fact so the vector leg has something to match.
        let fact_id = insert_test_fact(
            &pool,
            id,
            "global",
            "the lexical keyword query",
            (None, None, None),
            0.9,
        )
        .await;
        engram_storage::enqueue_embedding(
            &pool,
            engram_storage::target::FACT,
            fact_id,
            &embedder.model().id,
        )
        .await
        .unwrap();
        drain_pending_embeddings(&pool, &embedder, 16).await.unwrap();

        let resp = search_facts(
            &pool,
            &embedder,
            None, // disable rerank so we see the raw vector + trigram fields
            SearchFactsRequest {
                query: "lexical keyword query".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
            },
        )
        .await
        .unwrap();
        assert!(!resp.results.is_empty());
        let hit = &resp.results[0];
        // Either vector_score or trigram_score (or both) must be present.
        assert!(
            hit.vector_score.is_some() || hit.trigram_score.is_some(),
            "every hit must carry at least one leg's score"
        );
        // rerank_score must be None (rerank disabled).
        assert!(hit.rerank_score.is_none());
    }
}

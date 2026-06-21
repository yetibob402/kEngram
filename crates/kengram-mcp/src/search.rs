//! Read operations: `search_thoughts`, `recent_thoughts`, `get_thought`.
//!
//! `search_thoughts` is the hybrid retrieval path: vector kNN ∪ trigram
//! similarity, fused by RRF, then recency-boosted, then optionally reranked
//! by a cross-encoder model over the top `candidate_pool` candidates. If the
//! embedder is unreachable, the vector leg is skipped and
//! `vector_search_available` flips to `false`. The trigram leg is bounded and
//! opportunistic; timeout/errors soft-fail to an empty leg so the fast vector
//! path can still return. That effectively makes trigram a temporary
//! best-effort leg during the latency cutover until Phase-2 FTS replaces it as
//! the real lexical path. If the reranker is unreachable or not configured,
//! the rerank stage is skipped and `rerank_used` flips to `false`; available
//! legs still come back.
//!
//! M4: SearchRequest carries an optional `tag_filter` JSONB-containment
//! expression. The filter is applied post-fuse in Rust (mirroring Postgres'
//! `@>` semantics over JSONB objects/arrays). Less efficient than pushing
//! into SQL, but correctness-first while we ship M4. Each `SearchHit` also
//! carries the thought's `tags` so consumers can show / threshold by them
//! without a follow-up `get_thought`.

use kengram_core::{
    DEFAULT_RECENCY_HALF_LIFE_DAYS, DEFAULT_RRF_K, Embedder, EmbeddingModel, EmbeddingStatus, Hit,
    Metadata, Scope, Source, Tags, Thought, ThoughtId, recency_boost, rrf_fuse,
};
use kengram_embed::Reranker;
use sqlx::PgPool;
use time::OffsetDateTime;

pub const DEFAULT_SEARCH_LIMIT: usize = 10;
pub const MAX_SEARCH_LIMIT: usize = 100;
pub const DEFAULT_TOP_K_PER_LEG: usize = 50;
pub const DEFAULT_TRIGRAM_TOP_K: usize = DEFAULT_TOP_K_PER_LEG;
pub const DEFAULT_TRIGRAM_STATEMENT_TIMEOUT_MS: u64 = 300;
/// Historical default `candidate_pool` for the rerank stage. The bounded
/// trigram cutover deliberately does not narrow this default until the
/// 100-query GOLD sensitivity matrix selects the largest pool that preserves
/// recall while staying under the latency SLO.
pub const DEFAULT_RERANK_CANDIDATE_POOL: usize = 32;

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    /// Exact-match scope filter. Mutually exclusive with `scope_prefix`.
    pub scope: Option<Scope>,
    /// Prefix-match scope filter (matches scopes starting with this string,
    /// e.g. `"rjf."` matches `rjf.professional.cto` and `rjf.personal.health`).
    /// Mutually exclusive with `scope`; supplying both errors with
    /// [`ReadError::ScopeAndPrefixBothSet`].
    pub scope_prefix: Option<String>,
    pub limit: Option<usize>,
    pub recency_half_life_days: Option<f32>,
    /// Apply the cross-encoder rerank stage over the top `candidate_pool`
    /// post-RRF candidates. Defaults to `true` when omitted. Set
    /// `Some(false)` to skip rerank even when a reranker is configured
    /// (useful for A/B comparison).
    pub rerank: Option<bool>,
    /// Number of post-RRF candidates fed into the reranker. Ignored when
    /// rerank is off. Defaults to [`DEFAULT_RERANK_CANDIDATE_POOL`].
    pub candidate_pool: Option<usize>,
    /// JSONB containment filter against `thoughts.tags`. Examples:
    /// - `{"kind": "task"}`
    /// - `{"people": ["Sarah"]}`
    /// - `{"topics": ["rust"], "kind": "idea"}`
    ///
    /// When `None` or `{}`, no filter is applied. Applied post-fuse in
    /// Rust (containment check) rather than pushed into SQL.
    pub tag_filter: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub thought_id: ThoughtId,
    pub content: String,
    pub scope: Scope,
    pub source: Source,
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,
    /// LLM-extracted metadata for this thought. `Tags::default()` until the
    /// tag drainer has run.
    pub tags: Tags,
    /// Raw cosine similarity from the vector leg (`None` if not in that leg).
    pub vector_score: Option<f32>,
    /// Raw `similarity` (pg_trgm symmetric n-gram Jaccard) from the trigram
    /// leg (`None` if not in that leg).
    pub trigram_score: Option<f32>,
    /// RRF aggregate (optionally adjusted by recency boost).
    pub rrf_score: Option<f32>,
    /// Calibrated absolute score from the reranker (`None` if rerank was
    /// off, unavailable, or this hit fell outside the candidate pool).
    pub rerank_score: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub vector_search_available: bool,
    pub rerank_used: bool,
}

#[derive(Debug, Clone)]
pub struct RecentRequest {
    /// Exact-match scope filter. Mutually exclusive with `scope_prefix`.
    pub scope: Option<Scope>,
    /// Prefix-match scope filter. Mutually exclusive with `scope`.
    pub scope_prefix: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RecentResponse {
    pub results: Vec<Thought>,
}

/// Request for the `list_scopes` orchestrator. Optional `prefix` filters
/// scopes to those starting with the given string; `None` returns every
/// scope currently in use.
#[derive(Debug, Clone, Default)]
pub struct ListScopesRequest {
    pub prefix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ListScopesResponse {
    pub scopes: Vec<ScopeSummaryHit>,
}

/// One row in the [`ListScopesResponse`]. Wire-shape version of
/// `kengram_storage::ScopeSummary` — the `scope` is stringified for clients
/// that don't share the `Scope` newtype.
#[derive(Debug, Clone)]
pub struct ScopeSummaryHit {
    pub scope: String,
    pub thought_count: i64,
    pub first_activity_at: OffsetDateTime,
    pub last_activity_at: OffsetDateTime,
}

#[derive(Debug, Clone)]
pub struct GetThoughtResponse {
    pub thought: Thought,
    pub embedding_status: EmbeddingStatus,
    pub embedded_at: Option<OffsetDateTime>,
    /// `Some(_)` when the operator has retracted this thought via
    /// `retract_thought`. Retracted thoughts don't appear in
    /// `search_thoughts` / `recent_thoughts`. `get_thought` is the audit
    /// path: ID lookup always returns the row regardless of retraction
    /// state.
    pub retracted_at: Option<OffsetDateTime>,
    pub retracted_reason: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("query must be non-empty")]
    EmptyQuery,

    #[error("limit out of bounds: {got} (must be 1..={max})")]
    LimitOutOfBounds { got: usize, max: usize },

    #[error("thought not found")]
    NotFound,

    #[error("scope and scope_prefix are mutually exclusive; supply at most one")]
    ScopeAndPrefixBothSet,

    #[error("storage error: {0}")]
    Storage(#[from] kengram_storage::StorageError),
}

pub async fn search_thoughts(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    request: SearchRequest,
) -> Result<SearchResponse, ReadError> {
    search_thoughts_with_tuning(
        pool,
        embedder,
        reranker,
        request,
        DEFAULT_TRIGRAM_TOP_K,
        DEFAULT_TRIGRAM_STATEMENT_TIMEOUT_MS,
        DEFAULT_RERANK_CANDIDATE_POOL,
    )
    .await
}

async fn search_thoughts_with_tuning(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    request: SearchRequest,
    trigram_top_k: usize,
    trigram_timeout_ms: u64,
    default_candidate_pool: usize,
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
    if request.scope.is_some() && request.scope_prefix.is_some() {
        return Err(ReadError::ScopeAndPrefixBothSet);
    }
    let scope_filter = request.scope.as_ref().map(Scope::as_str);
    let scope_prefix_filter = request.scope_prefix.as_deref();

    // Vector leg (soft-fail to empty + flag).
    let (vector_hits, vector_search_available) = match embedder
        .embed(std::slice::from_ref(&request.query))
        .await
    {
        Ok(mut vectors) => {
            let v = vectors
                .pop()
                .expect("non-empty input must yield at least one vector");
            match kengram_storage::search_vector_knn(
                pool,
                v,
                embedder.model(),
                scope_filter,
                scope_prefix_filter,
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

    // Trigram leg: bounded and opportunistic. pg_trgm similarity over very
    // large blobs can be non-selective, so this leg must not break the SLO for
    // the already-fast vector + rerank path.
    let trigram_hits = bounded_trigram_hits(
        pool,
        &request.query,
        scope_filter,
        scope_prefix_filter,
        trigram_top_k,
        trigram_timeout_ms,
    )
    .await;

    // RRF fuse → recency boost.
    let mut fused = rrf_fuse(vec![vector_hits, trigram_hits], DEFAULT_RRF_K);
    let half_life = request
        .recency_half_life_days
        .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_DAYS);
    recency_boost(&mut fused, half_life, OffsetDateTime::now_utc());

    // Apply tag_filter (post-fuse, Rust-side). Empty objects and `None`
    // are no-ops. We do this BEFORE rerank so the reranker's candidate
    // pool is drawn from the filtered set — matches operator intent
    // ("rerank the task-kind thoughts" should rerank only those).
    if let Some(filter) = request.tag_filter.as_ref()
        && !is_empty_filter(filter)
    {
        let before = fused.len();
        fused.retain(|h| tags_match_filter(&h.thought.tags, filter));
        tracing::info!(
            tag_filter = %filter,
            retained = fused.len(),
            removed = before - fused.len(),
            "search_thoughts: tag_filter applied",
        );
    } else if request.tag_filter.is_some() {
        tracing::debug!("search_thoughts: tag_filter present but empty-object — no-op");
    }

    // Optional rerank stage.
    let rerank_enabled = request.rerank.unwrap_or(true);
    let candidate_pool = request.candidate_pool.unwrap_or(default_candidate_pool);
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
            tags: h.thought.tags,
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

async fn bounded_trigram_hits(
    pool: &PgPool,
    query: &str,
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    trigram_top_k: usize,
    trigram_timeout_ms: u64,
) -> Vec<Hit> {
    match kengram_storage::search_trigram_bounded(
        pool,
        query,
        scope_filter,
        scope_prefix_filter,
        trigram_top_k as i64,
        trigram_timeout_ms,
    )
    .await
    {
        Ok(hits) => hits,
        Err(e) => {
            tracing::warn!(
                error = %e,
                query_canceled = e.is_query_canceled(),
                timeout_ms = trigram_timeout_ms,
                "bounded trigram query failed; continuing with available search legs only",
            );
            vec![]
        }
    }
}

/// True when the supplied filter is a JSON object with no keys (i.e. `{}`)
/// or any other shape that should be treated as "no filter."
fn is_empty_filter(filter: &serde_json::Value) -> bool {
    match filter {
        serde_json::Value::Object(m) => m.is_empty(),
        // Anything other than a non-empty object is meaningless as a
        // containment filter; treat as no-op rather than failing.
        _ => true,
    }
}

/// Apply Postgres `@>`-style JSONB containment between a `Tags` haystack
/// and a JSON `needle`. Implementation strategy: serialize `tags` to a
/// `serde_json::Value` and walk both trees recursively.
///
/// Semantics:
/// - Object containment: every key in `needle` must exist in `haystack`
///   with a contained value.
/// - Array containment: every element in `needle` must have *some*
///   element in `haystack` that contains it (set-wise containment, not
///   index-wise).
/// - Scalars: equality.
/// - Null: equality only.
///
/// Matches Postgres' JSONB `@>` for the shapes we expect from the tagger
/// (objects keyed by `people`/`topics`/`action_items`/`dates_mentioned`/
/// `kind`, with string-array or string-scalar values).
pub(crate) fn tags_match_filter(tags: &Tags, filter: &serde_json::Value) -> bool {
    let haystack = match serde_json::to_value(tags) {
        Ok(v) => v,
        Err(_) => return false,
    };
    json_contains(&haystack, filter)
}

fn json_contains(haystack: &serde_json::Value, needle: &serde_json::Value) -> bool {
    use serde_json::Value as V;
    match (haystack, needle) {
        (V::Object(h), V::Object(n)) => n
            .iter()
            .all(|(k, v)| h.get(k).is_some_and(|hv| json_contains(hv, v))),
        (V::Array(h), V::Array(n)) => n.iter().all(|nv| h.iter().any(|hv| json_contains(hv, nv))),
        // Object-against-array or array-against-object: not contained.
        (V::Array(_), V::Object(_)) | (V::Object(_), V::Array(_)) => false,
        // Scalar/null equality.
        (a, b) => a == b,
    }
}

/// Run the cross-encoder rerank stage over the top `candidate_pool` hits.
/// On success, mutates `hits` in place: rerank scores are written to
/// `rerank_score`; the un-reranked tail is **truncated** so the response
/// contains only the reranker's verdict on the candidate pool. Re-sorts
/// by rerank score descending. `rrf_score` is preserved.
async fn apply_rerank_to_thought_hits(
    reranker: &dyn Reranker,
    query: &str,
    hits: &mut Vec<kengram_core::Hit>,
    candidate_pool: usize,
) -> bool {
    if hits.is_empty() {
        return false;
    }
    let pool_len = candidate_pool.min(hits.len());
    let candidates: Vec<&str> = hits[..pool_len]
        .iter()
        .map(|h| h.thought.content.as_str())
        .collect();
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
    for s in scores {
        if let Some(hit) = hits.get_mut(s.index) {
            hit.rerank_score = Some(s.score);
        }
    }
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
    if request.scope.is_some() && request.scope_prefix.is_some() {
        return Err(ReadError::ScopeAndPrefixBothSet);
    }
    let scope_filter = request.scope.as_ref().map(Scope::as_str);
    let scope_prefix_filter = request.scope_prefix.as_deref();

    let results =
        kengram_storage::recent_thoughts(pool, scope_filter, scope_prefix_filter, limit as i64)
            .await?;
    Ok(RecentResponse { results })
}

/// Enumerate scopes currently in use, optionally narrowed to a prefix.
/// Wraps [`kengram_storage::list_scopes`] and converts the storage-side
/// [`kengram_storage::ScopeSummary`] to wire-shape [`ScopeSummaryHit`].
pub async fn list_scopes(
    pool: &PgPool,
    request: ListScopesRequest,
) -> Result<ListScopesResponse, ReadError> {
    let rows = kengram_storage::list_scopes(pool, request.prefix.as_deref()).await?;
    let scopes = rows
        .into_iter()
        .map(|s| ScopeSummaryHit {
            scope: s.scope.into_string(),
            thought_count: s.thought_count,
            first_activity_at: s.first_activity_at,
            last_activity_at: s.last_activity_at,
        })
        .collect();
    Ok(ListScopesResponse { scopes })
}

pub async fn get_thought(
    pool: &PgPool,
    model: &EmbeddingModel,
    thought_id: ThoughtId,
) -> Result<GetThoughtResponse, ReadError> {
    let prov = kengram_storage::fetch_thought_with_provenance(pool, thought_id, model).await?;
    let prov = prov.ok_or(ReadError::NotFound)?;
    Ok(GetThoughtResponse {
        thought: prov.thought,
        embedding_status: prov.embedding_status,
        embedded_at: prov.embedded_at,
        retracted_at: prov.retracted_at,
        retracted_reason: prov.retracted_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureRequest, capture};
    use crate::drain::drain_pending_embeddings;
    use kengram_core::{EmbeddingModel, TagKind, Tags};
    use kengram_embed::{FakeBehavior, FakeEmbedder};

    const TEST_EMBEDDER_MODEL_ID: &str = "qwen3-embedding";

    fn test_embedding_model() -> EmbeddingModel {
        EmbeddingModel::new(TEST_EMBEDDER_MODEL_ID, 4096)
    }

    fn test_embedder() -> FakeEmbedder {
        FakeEmbedder::with_model(test_embedding_model())
    }

    /// Capture a thought — leaves it queued, not embedded.
    async fn cap(pool: &PgPool, content: &str, scope: &str) -> ThoughtId {
        capture(
            pool,
            TEST_EMBEDDER_MODEL_ID,
            None,
            CaptureRequest {
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                scope: Some(Scope::new(scope).unwrap()),
                metadata: None,
                argus_source_event: None,
            },
        )
        .await
        .unwrap()
        .thought_id
    }

    /// Capture and immediately drain — for tests that need vector search to
    /// work.
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
        let embedder = test_embedder();
        let id_a = cap_and_drain(&pool, &embedder, "alpha", "global").await;
        let _id_b = cap_and_drain(&pool, &embedder, "beta", "global").await;

        let resp = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "alpha".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
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
        let id = cap(&pool, "the tcgplayer integration was painful", "work").await;

        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let resp = search_thoughts(
            &pool,
            &bad,
            None,
            SearchRequest {
                query: "tcgplayer".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            },
        )
        .await
        .unwrap();

        assert!(!resp.vector_search_available);
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].thought_id, id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_soft_fails_timed_out_trigram_leg(pool: PgPool) {
        let embedder = test_embedder();
        let needle = "needle vector anchor";
        let needle_id = cap_and_drain(&pool, &embedder, needle, "global").await;

        for i in 0..512 {
            cap(
                &pool,
                &format!(
                    "bounded trigram load filler {i} {}",
                    "surface-noise ".repeat(350)
                ),
                "load",
            )
            .await;
        }

        let started = std::time::Instant::now();
        let trigram_hits =
            bounded_trigram_hits(&pool, needle, None, None, DEFAULT_TRIGRAM_TOP_K, 1).await;
        assert!(
            started.elapsed() < std::time::Duration::from_millis(800),
            "timed-out trigram leg should return inside its budget"
        );
        assert!(
            trigram_hits.is_empty(),
            "timed-out trigram leg must soft-fail to an empty leg"
        );

        let resp = search_thoughts_with_tuning(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: needle.to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
            },
            DEFAULT_TRIGRAM_TOP_K,
            1,
            DEFAULT_RERANK_CANDIDATE_POOL,
        )
        .await
        .unwrap();

        assert!(resp.vector_search_available);
        assert!(
            resp.results.iter().any(|hit| hit.thought_id == needle_id
                && hit.vector_score.is_some()
                && hit.trigram_score.is_none()),
            "outer search should still return vector results when trigram times out"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_empty_query_errors(pool: PgPool) {
        let embedder = test_embedder();
        let err = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: String::new(),
                scope: None,
                scope_prefix: None,
                limit: None,
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReadError::EmptyQuery));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_limit_out_of_bounds_errors(pool: PgPool) {
        let embedder = test_embedder();
        let err = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "x".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(1000),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReadError::LimitOutOfBounds { got: 1000, .. }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_respects_scope(pool: PgPool) {
        let embedder = test_embedder();
        cap(&pool, "tcgplayer work", "work").await;
        cap(&pool, "tcgplayer personal", "personal").await;

        let resp = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "tcgplayer".to_string(),
                scope: Some(Scope::new("work").unwrap()),
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
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
                scope_prefix: None,
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
        let embedder = test_embedder();
        let id = cap_and_drain(&pool, &embedder, "hello", "global").await;
        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.embedding_status, EmbeddingStatus::Indexed);
        assert!(resp.embedded_at.is_some());
        assert_eq!(resp.thought.content, "hello");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_pending_when_unembedded(pool: PgPool) {
        let embedder = test_embedder();
        let id = cap(&pool, "hello", "global").await;
        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.embedding_status, EmbeddingStatus::Pending);
        assert!(resp.embedded_at.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_not_found(pool: PgPool) {
        let embedder = test_embedder();
        let err = get_thought(&pool, embedder.model(), ThoughtId::new())
            .await
            .unwrap_err();
        assert!(matches!(err, ReadError::NotFound));
    }

    // -- M4: tag-filter coverage ---------------------------------------------

    /// Helper: capture, then directly write the given Tags via the storage
    /// path (bypassing the drainer).
    async fn cap_with_tags(pool: &PgPool, content: &str, tags: Tags) -> ThoughtId {
        let id = cap(pool, content, "global").await;
        kengram_storage::update_thought_tags(pool, id, &tags, "fake/tagger", 1)
            .await
            .unwrap();
        id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_filters_by_tag_kind(pool: PgPool) {
        let embedder = test_embedder();
        let task_id = cap_with_tags(
            &pool,
            "needs doing alpha",
            Tags {
                kind: Some(TagKind::Task),
                ..Tags::default()
            },
        )
        .await;
        let _idea_id = cap_with_tags(
            &pool,
            "needs doing beta idea",
            Tags {
                kind: Some(TagKind::Idea),
                ..Tags::default()
            },
        )
        .await;

        let resp = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "needs doing".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: Some(serde_json::json!({"kind": "task"})),
            },
        )
        .await
        .unwrap();

        let ids: Vec<ThoughtId> = resp.results.iter().map(|r| r.thought_id).collect();
        assert!(ids.contains(&task_id));
        assert_eq!(ids.len(), 1, "only the task-kind thought should match");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_filters_by_tag_array_containment(pool: PgPool) {
        let embedder = test_embedder();
        let sarah_id = cap_with_tags(
            &pool,
            "meeting with Sarah and Ron",
            Tags {
                people: vec!["Sarah".into(), "Ron".into()],
                ..Tags::default()
            },
        )
        .await;
        let _alex_id = cap_with_tags(
            &pool,
            "meeting with Alex",
            Tags {
                people: vec!["Alex".into()],
                ..Tags::default()
            },
        )
        .await;

        let resp = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "meeting".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: Some(serde_json::json!({"people": ["Sarah"]})),
            },
        )
        .await
        .unwrap();

        let ids: Vec<ThoughtId> = resp.results.iter().map(|r| r.thought_id).collect();
        assert!(ids.contains(&sarah_id));
        assert_eq!(ids.len(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_empty_tag_filter_is_noop(pool: PgPool) {
        let embedder = test_embedder();
        cap(&pool, "alpha keyword", "global").await;
        cap(&pool, "beta keyword", "global").await;

        let resp = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "keyword".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: Some(serde_json::json!({})),
            },
        )
        .await
        .unwrap();

        assert_eq!(resp.results.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_response_carries_tags_per_hit(pool: PgPool) {
        let embedder = test_embedder();
        let id = cap_with_tags(
            &pool,
            "tagged content here",
            Tags {
                topics: vec!["rust".into()],
                ..Tags::default()
            },
        )
        .await;

        let resp = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "tagged".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            },
        )
        .await
        .unwrap();
        let hit = resp
            .results
            .iter()
            .find(|h| h.thought_id == id)
            .expect("found");
        assert_eq!(hit.tags.topics, vec!["rust".to_string()]);
    }

    #[test]
    fn tags_match_filter_basic_kind() {
        let tags = Tags {
            kind: Some(TagKind::Task),
            ..Tags::default()
        };
        assert!(tags_match_filter(
            &tags,
            &serde_json::json!({"kind": "task"})
        ));
        assert!(!tags_match_filter(
            &tags,
            &serde_json::json!({"kind": "idea"})
        ));
    }

    #[test]
    fn tags_match_filter_array_containment_is_subset() {
        let tags = Tags {
            people: vec!["Sarah".into(), "Ron".into()],
            ..Tags::default()
        };
        // Needle ⊆ haystack → match.
        assert!(tags_match_filter(
            &tags,
            &serde_json::json!({"people": ["Sarah"]})
        ));
        // Needle has an element not in haystack → no match.
        assert!(!tags_match_filter(
            &tags,
            &serde_json::json!({"people": ["Alex"]})
        ));
    }

    #[test]
    fn tags_match_filter_combined_keys() {
        let tags = Tags {
            topics: vec!["rust".into()],
            kind: Some(TagKind::Idea),
            ..Tags::default()
        };
        assert!(tags_match_filter(
            &tags,
            &serde_json::json!({"topics": ["rust"], "kind": "idea"})
        ));
        assert!(!tags_match_filter(
            &tags,
            &serde_json::json!({"topics": ["rust"], "kind": "task"})
        ));
    }

    #[test]
    fn tags_match_filter_empty_object_is_trivially_true() {
        let tags = Tags::default();
        assert!(tags_match_filter(&tags, &serde_json::json!({})));
    }

    // -- M4: get_thought provenance carries tags + tagger metadata -----------

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_carries_tags_and_provenance(pool: PgPool) {
        let embedder = test_embedder();
        let tags = Tags {
            topics: vec!["meeting".into()],
            kind: Some(TagKind::Session),
            ..Tags::default()
        };
        let id = cap_with_tags(&pool, "tagged thought", tags.clone()).await;

        let resp = get_thought(&pool, embedder.model(), id).await.unwrap();
        assert_eq!(resp.thought.tags, tags);
        assert_eq!(
            resp.thought.tags_extractor_model.as_deref(),
            Some("fake/tagger")
        );
        assert_eq!(resp.thought.tags_extractor_version, Some(1));
        assert!(resp.thought.tags_extracted_at.is_some());
    }
}

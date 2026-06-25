//! Read operations: `search_thoughts`, `recent_thoughts`, `get_thought`.
//!
//! `search_thoughts` is the hybrid retrieval path: vector kNN ∪ Postgres FTS,
//! fused by RRF, then recency-boosted, then optionally reranked
//! by a cross-encoder model over the top `candidate_pool` candidates. If the
//! embedder is unreachable, the vector leg is skipped and
//! `vector_search_available` flips to `false`. The FTS lexical leg is backed
//! by a GIN index and still bounded defensively; timeout/errors soft-fail to
//! an empty leg so the fast vector path can still return. If the reranker is
//! unreachable or not configured,
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
    Metadata, Scope, Source, Tags, Thought, ThoughtId, normalize_domain_scope,
    normalize_retrieval_alias, recency_boost, rrf_fuse,
};
use kengram_embed::{RerankScore, Reranker, RerankerError};
use sqlx::PgPool;
use std::time::Instant;
use time::OffsetDateTime;

pub const DEFAULT_SEARCH_LIMIT: usize = 10;
pub const MAX_SEARCH_LIMIT: usize = 100;
pub const DEFAULT_TOP_K_PER_LEG: usize = 50;
pub const DEFAULT_LEXICAL_TOP_K: usize = DEFAULT_TOP_K_PER_LEG;
pub const DEFAULT_LEXICAL_STATEMENT_TIMEOUT_MS: u64 = 300;
/// Historical default `candidate_pool` for the rerank stage. The bounded
/// trigram matrix showed narrowing regressed recall; hold this value unless
/// the GOLD eval explicitly supports a change.
pub const DEFAULT_RERANK_CANDIDATE_POOL: usize = 32;
const RERANK_BACKEND_MAX_BATCH: usize = 32;
const PAIRWISE_MAX_SUBQUERIES: usize = 8;
const PAIRWISE_PER_SUBQUERY_TOP_K: usize = 25;

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
    /// Enables chunk dense + chunk FTS retrieval legs when the server config
    /// explicitly opts in. Defaults to false at the config/server layer.
    pub chunk_serving_enabled: bool,
    /// Master default-off gate for the composed full retrieval pipeline.
    pub full_pipeline_enabled: bool,
    /// Effective tag/domain routing gate supplied by server config. This must
    /// remain false when `full_pipeline_enabled` is false.
    pub tag_domain_routing_enabled: bool,
    /// Include per-stage latency diagnostics in the response. Defaults false
    /// so normal MCP callers do not receive internal timing metadata.
    pub include_profile: bool,
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
    /// Raw rank from the current lexical leg (`None` if not in that leg).
    pub lexical_score: Option<f32>,
    /// Legacy lexical score field from the old trigram leg. FTS hits mirror
    /// `lexical_score` here so existing clients keep seeing a lexical-leg
    /// signal while they migrate.
    pub trigram_score: Option<f32>,
    /// RRF aggregate (optionally adjusted by recency boost).
    pub rrf_score: Option<f32>,
    /// Calibrated absolute score from the reranker (`None` if rerank was
    /// off, unavailable, or this hit fell outside the candidate pool).
    pub rerank_score: Option<f32>,
    /// Matched chunk provenance when a chunk leg supplied evidence for this
    /// parent thought hit. `content` above remains the parent thought body for
    /// backward compatibility.
    pub chunk_id: Option<uuid::Uuid>,
    pub chunk_artifact_id: Option<uuid::Uuid>,
    pub chunk_source_thought_id: Option<ThoughtId>,
    pub chunk_index: Option<i32>,
    pub chunk_content: Option<String>,
    pub chunker_id: Option<String>,
    pub chunker_version: Option<i32>,
    pub chunk_token_estimate: Option<i32>,
    pub chunk_start_char: Option<i32>,
    pub chunk_end_char: Option<i32>,
    pub chunk_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub vector_search_available: bool,
    pub rerank_used: bool,
    pub profile: Option<SearchProfile>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SearchProfile {
    pub total_ms: f64,
    pub query_embedding_ms: f64,
    pub thought_vector_knn_ms: f64,
    pub chunk_vector_knn_ms: f64,
    pub thought_fts_ms: f64,
    pub chunk_fts_ms: f64,
    pub chunk_pairwise_fts_ms: f64,
    pub domain_scope_ms: f64,
    pub tag_facet_ms: f64,
    pub rrf_fusion_ms: f64,
    pub tag_filter_ms: f64,
    pub rerank_ms: f64,
    pub result_projection_ms: f64,
    pub parent_resolution_ms: f64,
    pub parent_resolution_mode: &'static str,
    pub full_pipeline_enabled: bool,
    pub tag_domain_routing_enabled: bool,
    pub planner_route: &'static str,
    pub planner_inferred_domains: Vec<String>,
    pub planner_tag_terms: Vec<String>,
    pub thought_vector_hits: usize,
    pub chunk_vector_hits: usize,
    pub thought_fts_hits: usize,
    pub chunk_fts_hits: usize,
    pub chunk_pairwise_fts_hits: usize,
    pub domain_scope_hits: usize,
    pub tag_facet_hits: usize,
    pub chunk_pairwise_subqueries: usize,
    pub fused_hit_count: usize,
    pub rerank_candidate_count: usize,
    pub result_count: usize,
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
        DEFAULT_LEXICAL_TOP_K,
        DEFAULT_LEXICAL_STATEMENT_TIMEOUT_MS,
        DEFAULT_RERANK_CANDIDATE_POOL,
    )
    .await
}

async fn search_thoughts_with_tuning(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    request: SearchRequest,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
    default_candidate_pool: usize,
) -> Result<SearchResponse, ReadError> {
    let total_started = Instant::now();
    let include_profile = request.include_profile;
    let mut profile = SearchProfile {
        parent_resolution_mode: "sql_join_in_retrieval_legs",
        ..SearchProfile::default()
    };
    let query = request.query.trim().to_string();
    if query.is_empty() {
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
    let chunk_serving_enabled = request.chunk_serving_enabled;
    let tag_domain_routing_enabled =
        request.full_pipeline_enabled && request.tag_domain_routing_enabled;
    profile.full_pipeline_enabled = request.full_pipeline_enabled;
    profile.tag_domain_routing_enabled = tag_domain_routing_enabled;
    profile.planner_route = if tag_domain_routing_enabled {
        "tag_domain_routing_v0"
    } else {
        "baseline"
    };

    // Vector leg (soft-fail to empty + flag).
    let embedding_started = Instant::now();
    let embedded_query = match embedder.embed(std::slice::from_ref(&query)).await {
        Ok(mut vectors) => Some(
            vectors
                .pop()
                .expect("non-empty input must yield at least one vector"),
        ),
        Err(e) => {
            tracing::warn!(error = %e, "embedder failed to embed query; falling back to lexical only");
            None
        }
    };
    profile.query_embedding_ms = elapsed_ms(embedding_started);

    let (vector_hits, chunk_vector_hits, vector_search_available) = match embedded_query {
        Some(v) => {
            let mut thought_vector_ok = false;
            let thought_vector_started = Instant::now();
            let vector_hits = match kengram_storage::search_vector_knn(
                pool,
                v.clone(),
                embedder.model(),
                scope_filter,
                scope_prefix_filter,
                DEFAULT_TOP_K_PER_LEG as i64,
            )
            .await
            {
                Ok(hits) => {
                    thought_vector_ok = true;
                    hits
                }
                Err(e) => {
                    tracing::warn!(error = %e, "vector kNN query failed; falling back to lexical only");
                    vec![]
                }
            };
            profile.thought_vector_knn_ms = elapsed_ms(thought_vector_started);
            let mut chunk_vector_ok = false;
            let chunk_vector_hits = if chunk_serving_enabled {
                let chunk_vector_started = Instant::now();
                let hits = match kengram_storage::search_artifact_chunks_vector_knn(
                    pool,
                    v,
                    embedder.model(),
                    scope_filter,
                    scope_prefix_filter,
                    DEFAULT_TOP_K_PER_LEG as i64,
                )
                .await
                {
                    Ok(hits) => {
                        chunk_vector_ok = true;
                        hits
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "artifact-chunk vector kNN query failed; continuing without chunk vector hits");
                        vec![]
                    }
                };
                profile.chunk_vector_knn_ms = elapsed_ms(chunk_vector_started);
                hits
            } else {
                vec![]
            };
            profile.thought_vector_hits = vector_hits.len();
            profile.chunk_vector_hits = chunk_vector_hits.len();
            let vector_available = thought_vector_ok || chunk_vector_ok;
            (vector_hits, chunk_vector_hits, vector_available)
        }
        None => (vec![], vec![], false),
    };

    // Lexical leg: FTS-backed and bounded. The GIN index should make this
    // fast; the timeout is a defensive belt so lexical failures do not turn
    // the whole hybrid search into a 5xx.
    let thought_fts_started = Instant::now();
    let lexical_hits = bounded_fts_hits(
        pool,
        &query,
        scope_filter,
        scope_prefix_filter,
        lexical_top_k,
        lexical_timeout_ms,
    )
    .await;
    profile.thought_fts_ms = elapsed_ms(thought_fts_started);
    profile.thought_fts_hits = lexical_hits.len();
    let chunk_lexical_hits = if chunk_serving_enabled {
        let chunk_fts_started = Instant::now();
        let hits = bounded_artifact_chunk_fts_hits(
            pool,
            &query,
            scope_filter,
            scope_prefix_filter,
            lexical_top_k,
            lexical_timeout_ms,
        )
        .await;
        profile.chunk_fts_ms = elapsed_ms(chunk_fts_started);
        hits
    } else {
        vec![]
    };
    profile.chunk_fts_hits = chunk_lexical_hits.len();
    let chunk_pairwise_lexical_hits = if chunk_serving_enabled {
        profile.chunk_pairwise_subqueries = pairwise_subqueries(&query).len();
        let chunk_pairwise_started = Instant::now();
        let hits = bounded_pairwise_artifact_chunk_fts_hits(
            pool,
            &query,
            scope_filter,
            scope_prefix_filter,
            lexical_top_k,
            lexical_timeout_ms,
        )
        .await;
        profile.chunk_pairwise_fts_ms = elapsed_ms(chunk_pairwise_started);
        hits
    } else {
        vec![]
    };
    profile.chunk_pairwise_fts_hits = chunk_pairwise_lexical_hits.len();

    let (domain_scope_hits, tag_facet_hits) = if tag_domain_routing_enabled {
        profile.planner_inferred_domains = domain_candidates(&query);
        profile.planner_tag_terms = tag_facet_terms(&query);

        let domain_started = Instant::now();
        let domain_hits = bounded_domain_scope_hits(
            pool,
            &profile.planner_inferred_domains,
            scope_filter,
            scope_prefix_filter,
            lexical_top_k,
            lexical_timeout_ms,
        )
        .await;
        profile.domain_scope_ms = elapsed_ms(domain_started);
        profile.domain_scope_hits = domain_hits.len();

        let tag_started = Instant::now();
        let tag_hits = bounded_tag_facet_hits(
            pool,
            &profile.planner_tag_terms,
            scope_filter,
            scope_prefix_filter,
            lexical_top_k,
            lexical_timeout_ms,
        )
        .await;
        profile.tag_facet_ms = elapsed_ms(tag_started);
        profile.tag_facet_hits = tag_hits.len();
        (domain_hits, tag_hits)
    } else {
        (vec![], vec![])
    };

    // RRF fuse → recency boost.
    let rrf_started = Instant::now();
    let mut rankings = vec![vector_hits, lexical_hits];
    if chunk_serving_enabled {
        rankings.push(chunk_vector_hits);
        rankings.push(chunk_lexical_hits);
        rankings.push(chunk_pairwise_lexical_hits);
    }
    if tag_domain_routing_enabled {
        rankings.push(domain_scope_hits);
        rankings.push(tag_facet_hits);
    }
    let mut fused = rrf_fuse(rankings, DEFAULT_RRF_K);
    let half_life = request
        .recency_half_life_days
        .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_DAYS);
    recency_boost(&mut fused, half_life, OffsetDateTime::now_utc());
    profile.rrf_fusion_ms = elapsed_ms(rrf_started);
    profile.fused_hit_count = fused.len();

    // Apply tag_filter (post-fuse, Rust-side). Empty objects and `None`
    // are no-ops. We do this BEFORE rerank so the reranker's candidate
    // pool is drawn from the filtered set — matches operator intent
    // ("rerank the task-kind thoughts" should rerank only those).
    let tag_filter_started = Instant::now();
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
    profile.tag_filter_ms = elapsed_ms(tag_filter_started);

    // Optional rerank stage.
    let rerank_enabled = request.rerank.unwrap_or(true);
    let candidate_pool = request.candidate_pool.unwrap_or(default_candidate_pool);
    profile.rerank_candidate_count = if rerank_enabled && reranker.is_some() {
        candidate_pool.min(fused.len())
    } else {
        0
    };
    let rerank_started = Instant::now();
    let rerank_used = match (rerank_enabled, reranker) {
        (true, Some(rr)) => {
            apply_rerank_to_thought_hits(
                rr,
                &query,
                &mut fused,
                candidate_pool,
                chunk_serving_enabled,
            )
            .await
        }
        _ => false,
    };
    profile.rerank_ms = elapsed_ms(rerank_started);

    let projection_started = Instant::now();
    let results: Vec<SearchHit> = fused
        .into_iter()
        .take(limit)
        .map(search_hit_from_core_hit)
        .collect();
    profile.result_projection_ms = elapsed_ms(projection_started);
    profile.result_count = results.len();
    profile.total_ms = elapsed_ms(total_started);

    Ok(SearchResponse {
        results,
        vector_search_available,
        rerank_used,
        profile: include_profile.then_some(profile),
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn search_hit_from_core_hit(h: Hit) -> SearchHit {
    let chunk = h.chunk;
    SearchHit {
        thought_id: h.thought.id,
        content: h.thought.content,
        scope: h.thought.scope,
        source: h.thought.source,
        created_at: h.thought.created_at,
        metadata: h.thought.metadata,
        tags: h.thought.tags,
        vector_score: h.vector_score,
        lexical_score: h.lexical_score,
        trigram_score: h.trigram_score,
        rrf_score: h.rrf_score,
        rerank_score: h.rerank_score,
        chunk_id: chunk.as_ref().map(|c| c.chunk_id),
        chunk_artifact_id: chunk.as_ref().map(|c| c.artifact_id),
        chunk_source_thought_id: chunk.as_ref().map(|c| c.source_thought_id),
        chunk_index: chunk.as_ref().map(|c| c.chunk_index),
        chunk_content: chunk.as_ref().map(|c| c.content.clone()),
        chunker_id: chunk.as_ref().map(|c| c.chunker_id.clone()),
        chunker_version: chunk.as_ref().map(|c| c.chunker_version),
        chunk_token_estimate: chunk.as_ref().and_then(|c| c.token_estimate),
        chunk_start_char: chunk.as_ref().and_then(|c| c.start_char),
        chunk_end_char: chunk.as_ref().and_then(|c| c.end_char),
        chunk_metadata: chunk.map(|c| c.metadata),
    }
}

async fn bounded_fts_hits(
    pool: &PgPool,
    query: &str,
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
) -> Vec<Hit> {
    match kengram_storage::search_fts_bounded(
        pool,
        query,
        scope_filter,
        scope_prefix_filter,
        lexical_top_k as i64,
        lexical_timeout_ms,
    )
    .await
    {
        Ok(hits) => hits,
        Err(e) => {
            tracing::warn!(
                error = %e,
                query_canceled = e.is_query_canceled(),
                timeout_ms = lexical_timeout_ms,
                "bounded FTS query failed; continuing with available search legs only",
            );
            vec![]
        }
    }
}

async fn bounded_domain_scope_hits(
    pool: &PgPool,
    domains: &[String],
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
) -> Vec<Hit> {
    match kengram_storage::search_domain_scope_aliases_bounded(
        pool,
        domains,
        scope_filter,
        scope_prefix_filter,
        lexical_top_k as i64,
        lexical_timeout_ms,
    )
    .await
    {
        Ok(hits) => hits,
        Err(e) => {
            tracing::warn!(
                error = %e,
                query_canceled = e.is_query_canceled(),
                timeout_ms = lexical_timeout_ms,
                "bounded domain-scope candidate leg failed; continuing with baseline search legs",
            );
            vec![]
        }
    }
}

async fn bounded_tag_facet_hits(
    pool: &PgPool,
    terms: &[String],
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
) -> Vec<Hit> {
    match kengram_storage::search_tag_facets_bounded(
        pool,
        terms,
        scope_filter,
        scope_prefix_filter,
        lexical_top_k as i64,
        lexical_timeout_ms,
    )
    .await
    {
        Ok(hits) => hits,
        Err(e) => {
            tracing::warn!(
                error = %e,
                query_canceled = e.is_query_canceled(),
                timeout_ms = lexical_timeout_ms,
                "bounded tag-facet candidate leg failed; continuing with baseline search legs",
            );
            vec![]
        }
    }
}

async fn bounded_artifact_chunk_fts_hits(
    pool: &PgPool,
    query: &str,
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
) -> Vec<Hit> {
    match kengram_storage::search_artifact_chunks_fts_bounded(
        pool,
        query,
        scope_filter,
        scope_prefix_filter,
        lexical_top_k as i64,
        lexical_timeout_ms,
    )
    .await
    {
        Ok(hits) => hits,
        Err(e) => {
            tracing::warn!(
                error = %e,
                query_canceled = e.is_query_canceled(),
                timeout_ms = lexical_timeout_ms,
                "bounded artifact-chunk FTS query failed; continuing with available search legs only",
            );
            vec![]
        }
    }
}

async fn bounded_pairwise_artifact_chunk_fts_hits(
    pool: &PgPool,
    query: &str,
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
) -> Vec<Hit> {
    let subqueries = pairwise_subqueries(query);
    if subqueries.is_empty() {
        return Vec::new();
    }

    let mut rankings = Vec::new();
    for subquery in &subqueries {
        let hits = bounded_artifact_chunk_fts_hits(
            pool,
            subquery,
            scope_filter,
            scope_prefix_filter,
            PAIRWISE_PER_SUBQUERY_TOP_K,
            lexical_timeout_ms,
        )
        .await;
        if !hits.is_empty() {
            rankings.push(hits);
        }
    }

    if rankings.is_empty() {
        return Vec::new();
    }
    tracing::debug!(
        subquery_count = subqueries.len(),
        rankings = rankings.len(),
        "artifact-chunk pairwise FTS leg produced candidate rankings",
    );
    rrf_fuse(rankings, DEFAULT_RRF_K)
        .into_iter()
        .take(lexical_top_k)
        .collect()
}

fn pairwise_subqueries(query: &str) -> Vec<String> {
    let terms = lexical_terms(query);
    let mut pairs = Vec::new();
    for window in terms.windows(2) {
        if window[0].eq_ignore_ascii_case(&window[1]) {
            continue;
        }
        pairs.push(format!("{} {}", window[0], window[1]));
    }
    for ident in identifier_terms(query).into_iter().take(4) {
        for term in terms.iter().take(8) {
            if ident.eq_ignore_ascii_case(term) {
                continue;
            }
            pairs.push(format!("{ident} {term}"));
        }
    }
    dedupe_lowercase_limited(pairs, PAIRWISE_MAX_SUBQUERIES)
}

fn domain_candidates(query: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(domain) = normalize_domain_scope(query) {
        candidates.push(domain);
    }
    for term in lexical_terms(query) {
        if let Some(domain) = normalize_domain_scope(&term) {
            candidates.push(domain);
        }
    }
    dedupe_lowercase_limited(candidates, 8)
}

fn tag_facet_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    if let Some(alias) = normalize_retrieval_alias(query) {
        terms.push(alias);
    }
    terms.extend(lexical_terms(query));
    dedupe_lowercase_limited(terms, 16)
}

fn lexical_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw in raw_pairwise_tokens(query) {
        let token = clean_token(&raw);
        if token.is_empty() {
            continue;
        }
        let parts: Vec<String> = token
            .split(['/', '_', '.', ':', '#', '-'])
            .filter(|part| !part.is_empty())
            .map(ToString::to_string)
            .collect();
        let mut candidates = vec![token];
        if parts.len() > 1 {
            candidates.extend(parts);
        }
        for candidate in candidates {
            let candidate = clean_token(&candidate);
            if candidate.is_empty() {
                continue;
            }
            let lower = candidate.to_ascii_lowercase();
            if !is_identifier(&candidate) && (is_pairwise_stopword(&lower) || lower.len() < 3) {
                continue;
            }
            if seen.insert(lower) {
                terms.push(candidate);
            }
        }
    }
    for term in terms.clone() {
        for alias in pairwise_aliases(&term.to_ascii_lowercase()) {
            let lower = alias.to_ascii_lowercase();
            if seen.insert(lower) {
                terms.push(alias.to_string());
            }
        }
    }
    terms.truncate(18);
    terms
}

fn identifier_terms(query: &str) -> Vec<String> {
    let raws = raw_pairwise_tokens(query);
    let mut terms = Vec::new();
    for (idx, raw) in raws.iter().enumerate() {
        let token = clean_token(raw);
        if token.is_empty() {
            continue;
        }
        if token.eq_ignore_ascii_case("pr")
            && let Some(next) = raws.get(idx + 1).map(|s| clean_token(s))
            && next
                .trim_start_matches('#')
                .chars()
                .all(|ch| ch.is_ascii_digit())
        {
            terms.push(format!("PR {next}"));
        }
        if (token.starts_with('#') && token[1..].chars().all(|ch| ch.is_ascii_digit()))
            || looks_like_hash(&token)
            || looks_like_file_path(&token)
            || looks_like_date(&token)
        {
            terms.push(token);
        }
    }
    for token in lexical_terms(query) {
        if is_identifier(&token) {
            terms.push(token);
        }
    }
    dedupe_lowercase_limited(terms, 12)
}

fn dedupe_lowercase_limited(values: Vec<String>, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for value in values {
        if seen.insert(value.to_ascii_lowercase()) {
            out.push(value);
        }
        if out.len() == limit {
            break;
        }
    }
    out
}

fn raw_pairwise_tokens(query: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in query.chars() {
        if current.is_empty() {
            if ch.is_alphanumeric() || ch == '_' || ch == '#' {
                current.push(ch);
            }
        } else if ch.is_alphanumeric() || matches!(ch, '_' | '.' | ':' | '/' | '#' | '-') {
            current.push(ch);
        } else {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn clean_token(token: &str) -> String {
    token
        .trim()
        .trim_matches(|ch: char| "'\"`()[]{}<>.,;!?".contains(ch))
        .to_string()
}

fn is_identifier(value: &str) -> bool {
    let mut upper_count = 0;
    let mut prev_lower = false;
    let mut camel = false;
    for ch in value.chars() {
        if ch.is_uppercase() {
            upper_count += 1;
            if prev_lower {
                camel = true;
            }
            prev_lower = false;
        } else {
            prev_lower = ch.is_lowercase();
        }
    }
    value.chars().any(|ch| ch.is_ascii_digit())
        || value.chars().any(|ch| "_/.:-#".contains(ch))
        || upper_count >= 2
        || camel
}

fn looks_like_hash(value: &str) -> bool {
    (7..=40).contains(&value.len()) && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn looks_like_file_path(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        ".md", ".json", ".toml", ".rs", ".ts", ".tsx", ".py", ".sql", ".yaml", ".yml", ".sh",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
        || value.split('/').filter(|part| !part.is_empty()).count() > 1
}

fn looks_like_date(value: &str) -> bool {
    value.len() >= 10
        && value.get(4..5) == Some("-")
        && value.get(7..8) == Some("-")
        && value[..4].chars().all(|ch| ch.is_ascii_digit())
        && value[5..7].chars().all(|ch| ch.is_ascii_digit())
        && value[8..10].chars().all(|ch| ch.is_ascii_digit())
}

fn pairwise_aliases(value: &str) -> &'static [&'static str] {
    match value {
        "a2a" => &["agent comms", "agent communication", "AGENT_COMMS"],
        "fts" => &["full text search", "lexical search", "lexical"],
        "kengram" => &["kEngram", "memory", "search_thoughts"],
        "mcp" => &["tools call", "tools list", "model context protocol"],
        "pr" => &["pull request"],
        "reranker" => &["cross encoder", "cross-encoder", "TEI"],
        "smith" => &["agents/smith"],
        _ => &[],
    }
}

fn is_pairwise_stopword(lower: &str) -> bool {
    matches!(
        lower,
        "a" | "about"
            | "after"
            | "all"
            | "an"
            | "and"
            | "are"
            | "as"
            | "ask"
            | "asked"
            | "at"
            | "be"
            | "before"
            | "by"
            | "can"
            | "could"
            | "current"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "how"
            | "in"
            | "into"
            | "is"
            | "it"
            | "its"
            | "me"
            | "of"
            | "on"
            | "or"
            | "right"
            | "should"
            | "so"
            | "that"
            | "the"
            | "their"
            | "there"
            | "this"
            | "to"
            | "use"
            | "used"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "with"
    )
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
    exact_identifier_boost_enabled: bool,
) -> bool {
    if hits.is_empty() {
        return false;
    }
    let pool_len = candidate_pool.min(hits.len());
    let candidates: Vec<&str> = hits[..pool_len]
        .iter()
        .map(|h| {
            h.chunk
                .as_ref()
                .map(|chunk| chunk.content.as_str())
                .unwrap_or_else(|| h.thought.content.as_str())
        })
        .collect();
    let scores = match rerank_candidates_batched(reranker, query, &candidates).await {
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
    if exact_identifier_boost_enabled {
        apply_exact_identifier_boost(query, &mut hits[..pool_len]);
    }
    hits.truncate(pool_len);
    hits.sort_by(|a, b| {
        let av = a.rerank_score.unwrap_or(f32::MIN);
        let bv = b.rerank_score.unwrap_or(f32::MIN);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    true
}

async fn rerank_candidates_batched(
    reranker: &dyn Reranker,
    query: &str,
    candidates: &[&str],
) -> Result<Vec<RerankScore>, RerankerError> {
    let batches: Vec<(usize, &[&str])> = candidates
        .chunks(RERANK_BACKEND_MAX_BATCH)
        .enumerate()
        .map(|(idx, batch)| (idx * RERANK_BACKEND_MAX_BATCH, batch))
        .collect();

    match batches.as_slice() {
        [] => Ok(Vec::new()),
        [(offset, batch)] => rerank_one_batch(reranker, query, batch, *offset).await,
        [(offset_a, batch_a), (offset_b, batch_b)] => {
            let (a, b) = tokio::join!(
                rerank_one_batch(reranker, query, batch_a, *offset_a),
                rerank_one_batch(reranker, query, batch_b, *offset_b),
            );
            let mut scores = Vec::with_capacity(candidates.len());
            scores.extend(a?);
            scores.extend(b?);
            Ok(scores)
        }
        _ => {
            let mut scores = Vec::with_capacity(candidates.len());
            for (offset, batch) in batches {
                scores.extend(rerank_one_batch(reranker, query, batch, offset).await?);
            }
            Ok(scores)
        }
    }
}

async fn rerank_one_batch(
    reranker: &dyn Reranker,
    query: &str,
    batch: &[&str],
    batch_offset: usize,
) -> Result<Vec<RerankScore>, RerankerError> {
    reranker.rerank(query, batch).await.map(|scores| {
        scores
            .into_iter()
            .map(|s| RerankScore {
                index: s.index + batch_offset,
                score: s.score,
            })
            .collect()
    })
}

fn apply_exact_identifier_boost(query: &str, hits: &mut [kengram_core::Hit]) {
    let terms = exact_identifier_boost_terms(query);
    if terms.is_empty() {
        return;
    }
    for hit in hits {
        let boost = exact_identifier_match_score(hit, &terms);
        if boost > 0.0 {
            hit.rerank_score = Some(hit.rerank_score.unwrap_or(0.0) + boost.min(0.75));
        }
    }
}

fn exact_identifier_match_score(hit: &kengram_core::Hit, terms: &[(String, f32)]) -> f32 {
    let haystack = hit
        .chunk
        .as_ref()
        .map(|chunk| chunk.content.as_str())
        .unwrap_or_else(|| hit.thought.content.as_str())
        .to_ascii_lowercase();
    let mut boost = 0.0_f32;
    for (term, weight) in terms {
        if haystack.contains(term) {
            boost += *weight;
        }
    }
    boost
}

fn exact_identifier_boost_terms(query: &str) -> Vec<(String, f32)> {
    let raw_tokens: Vec<String> = raw_pairwise_tokens(query)
        .into_iter()
        .map(|token| clean_token(&token))
        .filter(|token| !token.is_empty())
        .collect();
    let mut terms = Vec::new();

    for token in &raw_tokens {
        if is_strong_identifier(token) {
            terms.push((token.to_ascii_lowercase(), 0.55));
        }
    }
    for window in raw_tokens.windows(2) {
        let left = &window[0];
        let right = &window[1];
        if left.chars().any(|ch| ch.is_ascii_alphabetic())
            && right.chars().any(|ch| ch.is_ascii_digit())
        {
            terms.push((format!("{left} {right}").to_ascii_lowercase(), 0.55));
        }
        if left.eq_ignore_ascii_case("pr")
            && right
                .trim_start_matches('#')
                .chars()
                .all(|ch| ch.is_ascii_digit())
        {
            terms.push((format!("PR {right}").to_ascii_lowercase(), 0.55));
        }
    }

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (term, weight) in terms {
        if seen.insert(term.clone()) {
            out.push((term, weight));
        }
    }
    out
}

fn is_strong_identifier(value: &str) -> bool {
    if value.len() < 3 {
        return false;
    }
    let has_alpha = value.chars().any(|ch| ch.is_ascii_alphabetic());
    let has_digit = value.chars().any(|ch| ch.is_ascii_digit());
    (has_alpha && has_digit) || looks_like_hash(value) || looks_like_file_path(value)
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
    use kengram_embed::{FakeBehavior, FakeEmbedder, FakeReranker};

    const TEST_EMBEDDER_MODEL_ID: &str = "qwen3-embedding";

    fn test_embedding_model() -> EmbeddingModel {
        EmbeddingModel::new(TEST_EMBEDDER_MODEL_ID, 4096)
    }

    fn test_embedder() -> FakeEmbedder {
        FakeEmbedder::with_model(test_embedding_model())
    }

    fn test_hit(content: &str) -> kengram_core::Hit {
        kengram_core::Hit {
            thought: Thought {
                id: ThoughtId::new(),
                scope: Scope::new("global").unwrap(),
                content: content.to_string(),
                source: Source::new("test").unwrap(),
                created_at: OffsetDateTime::now_utc(),
                metadata: Metadata::default(),
                content_fingerprint: [0_u8; 32],
                tags: Tags::default(),
                tags_extractor_model: None,
                tags_extractor_version: None,
                tags_extracted_at: None,
            },
            vector_score: None,
            lexical_score: None,
            trigram_score: None,
            rrf_score: Some(1.0),
            rerank_score: None,
            chunk: None,
        }
    }

    #[tokio::test]
    async fn exact_identifier_boost_is_chunk_serving_flag_gated() {
        let reranker = FakeReranker::new();
        let query = "which note explains KGR999";

        let mut flag_off = vec![
            test_hit("plain first candidate"),
            test_hit("KGR999 exact id"),
        ];
        assert!(apply_rerank_to_thought_hits(&reranker, query, &mut flag_off, 2, false).await);
        assert_eq!(flag_off[0].thought.content, "plain first candidate");
        assert_eq!(flag_off[1].thought.content, "KGR999 exact id");

        let mut flag_on = vec![
            test_hit("plain first candidate"),
            test_hit("KGR999 exact id"),
        ];
        assert!(apply_rerank_to_thought_hits(&reranker, query, &mut flag_on, 2, true).await);
        assert_eq!(flag_on[0].thought.content, "KGR999 exact id");
        assert_eq!(flag_on[1].thought.content, "plain first candidate");
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

    async fn ensure_test_chunk_schema(pool: &PgPool) {
        for ddl in [
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS source_thought_id UUID REFERENCES thoughts(id) ON DELETE CASCADE",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS content_fingerprint BYTEA",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS chunker_id TEXT DEFAULT 'test'",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS chunker_version INT DEFAULT 1",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS token_estimate INT",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS start_char INT",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS end_char INT",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS pipeline_run_id UUID",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS retracted_at TIMESTAMPTZ",
            "ALTER TABLE artifact_chunks ADD COLUMN IF NOT EXISTS retracted_reason TEXT",
            "CREATE INDEX IF NOT EXISTS artifact_chunks_content_fts_idx ON artifact_chunks USING gin (to_tsvector('english', content))",
        ] {
            sqlx::query(ddl).execute(pool).await.unwrap();
        }
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
            },
        )
        .await
        .unwrap();

        assert!(resp.vector_search_available);
        assert!(!resp.results.is_empty());
        assert_eq!(resp.results[0].thought_id, id_a);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_excludes_eval_contamination_before_rerank_pool(pool: PgPool) {
        let clean = cap(
            &pool,
            "clean answer marker tcgplayer canonical baseline",
            "global",
        )
        .await;
        let denied = cap(
            &pool,
            "KGR024 answer marker tcgplayer canonical baseline",
            "global",
        )
        .await;
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let reranker = FakeReranker::new();

        let resp = search_thoughts(
            &pool,
            &bad,
            Some(&reranker),
            SearchRequest {
                query: "tcgplayer canonical".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(true),
                candidate_pool: Some(10),
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        assert!(resp.rerank_used, "reranker must run for pre-pool proof");
        let rerank_call = reranker
            .last_call()
            .expect("reranker should record the candidate pool");
        assert!(
            rerank_call
                .candidates
                .iter()
                .any(|content| content.contains("clean answer marker")),
            "clean control row should enter the rerank candidate pool"
        );
        assert!(
            rerank_call
                .candidates
                .iter()
                .all(|content| !content.contains("KGR024")),
            "denied eval row must be absent from the pre-rerank candidate pool"
        );

        let returned_ids = resp
            .results
            .iter()
            .map(|hit| hit.thought_id)
            .collect::<Vec<_>>();
        assert!(returned_ids.contains(&clean));
        assert!(!returned_ids.contains(&denied));
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
            },
        )
        .await
        .unwrap();

        assert!(!resp.vector_search_available);
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].thought_id, id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn chunk_serving_flag_controls_chunk_fts_parent_hits(pool: PgPool) {
        ensure_test_chunk_schema(&pool).await;
        let parent_id = cap(&pool, "parent body deliberately lacks the marker", "global").await;
        let artifact_id = uuid::Uuid::new_v4();
        let chunk_id = uuid::Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO artifacts (id, scope, kind, title, metadata)
            VALUES ($1, 'global', 'thought_chunks', 'test artifact', '{}')
            "#,
        )
        .bind(artifact_id)
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            r#"
            INSERT INTO artifact_chunks (
                id,
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
                metadata
            )
            VALUES ($1,$2,$3,0,'chunk-only-needle answer-bearing chunk',$4,'test-chunker',1,6,0,38,'{"fixture":true}')
            "#,
        )
        .bind(chunk_id)
        .bind(artifact_id)
        .bind(parent_id.into_uuid())
        .bind(vec![9_u8; 32])
        .execute(&pool)
        .await
        .unwrap();

        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let mut request = SearchRequest {
            query: "chunk-only-needle".to_string(),
            scope: None,
            scope_prefix: None,
            limit: Some(10),
            recency_half_life_days: Some(0.0),
            rerank: Some(false),
            candidate_pool: None,
            tag_filter: None,
            chunk_serving_enabled: false,
            full_pipeline_enabled: false,
            tag_domain_routing_enabled: false,
            include_profile: false,
        };
        let off = search_thoughts(&pool, &bad, None, request.clone())
            .await
            .unwrap();
        assert!(off.results.is_empty());

        request.chunk_serving_enabled = true;
        let on = search_thoughts(&pool, &bad, None, request).await.unwrap();
        assert_eq!(on.results.len(), 1);
        let hit = &on.results[0];
        assert_eq!(hit.thought_id, parent_id);
        assert_eq!(hit.content, "parent body deliberately lacks the marker");
        assert_eq!(hit.chunk_id, Some(chunk_id));
        assert_eq!(
            hit.chunk_content.as_deref(),
            Some("chunk-only-needle answer-bearing chunk")
        );
        assert_eq!(hit.chunk_index, Some(0));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_soft_fails_timed_out_fts_leg(pool: PgPool) {
        let embedder = test_embedder();
        let needle = "needle vector anchor";
        let needle_id = cap_and_drain(&pool, &embedder, needle, "global").await;

        for i in 0..512 {
            cap(
                &pool,
                &format!(
                    "bounded fts load filler {i} needle vector anchor {}",
                    "surface noise ".repeat(350)
                ),
                "load",
            )
            .await;
        }

        let started = std::time::Instant::now();
        let lexical_hits =
            bounded_fts_hits(&pool, needle, None, None, DEFAULT_LEXICAL_TOP_K, 1).await;
        assert!(
            started.elapsed() < std::time::Duration::from_millis(800),
            "timed-out FTS leg should return inside its budget"
        );
        assert!(
            lexical_hits.is_empty(),
            "timed-out FTS leg must soft-fail to an empty leg"
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
            },
            DEFAULT_LEXICAL_TOP_K,
            1,
            DEFAULT_RERANK_CANDIDATE_POOL,
        )
        .await
        .unwrap();

        assert!(resp.vector_search_available);
        assert!(
            resp.results.iter().any(|hit| hit.thought_id == needle_id
                && hit.vector_score.is_some()
                && hit.lexical_score.is_none()
                && hit.trigram_score.is_none()),
            "outer search should still return vector results when FTS times out"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_empty_query_errors(pool: PgPool) {
        let embedder = test_embedder();
        for query in ["", "   \n\t  "] {
            let err = search_thoughts(
                &pool,
                &embedder,
                None,
                SearchRequest {
                    query: query.to_string(),
                    scope: None,
                    scope_prefix: None,
                    limit: None,
                    recency_half_life_days: None,
                    rerank: None,
                    candidate_pool: None,
                    tag_filter: None,
                    chunk_serving_enabled: false,
                    full_pipeline_enabled: false,
                    tag_domain_routing_enabled: false,
                    include_profile: false,
                },
            )
            .await
            .unwrap_err();
            assert!(matches!(err, ReadError::EmptyQuery));
        }
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(resp.results.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn subordinate_tag_domain_flag_is_ineffective_when_master_gate_off(pool: PgPool) {
        let embedder = test_embedder();
        cap(&pool, "pipeline flag keyword alpha", "global").await;
        cap(&pool, "pipeline flag keyword beta", "global").await;

        let base_request = SearchRequest {
            query: "pipeline flag keyword".to_string(),
            scope: None,
            scope_prefix: None,
            limit: Some(10),
            recency_half_life_days: Some(0.0),
            rerank: Some(false),
            candidate_pool: None,
            tag_filter: None,
            chunk_serving_enabled: false,
            full_pipeline_enabled: false,
            tag_domain_routing_enabled: false,
            include_profile: true,
        };
        let mut subordinate_only = base_request.clone();
        subordinate_only.tag_domain_routing_enabled = true;

        let base = search_thoughts(&pool, &embedder, None, base_request)
            .await
            .unwrap();
        let subordinate = search_thoughts(&pool, &embedder, None, subordinate_only)
            .await
            .unwrap();
        let base_ids: Vec<ThoughtId> = base.results.iter().map(|r| r.thought_id).collect();
        let subordinate_ids: Vec<ThoughtId> =
            subordinate.results.iter().map(|r| r.thought_id).collect();
        assert_eq!(subordinate_ids, base_ids);
        let profile = subordinate
            .profile
            .expect("include_profile should emit profile");
        assert!(!profile.full_pipeline_enabled);
        assert!(!profile.tag_domain_routing_enabled);
        assert_eq!(profile.planner_route, "baseline");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn flag_true_adds_tag_candidates_without_dropping_baseline(pool: PgPool) {
        let embedder = test_embedder();
        let baseline_id = cap(&pool, "memory search baseline lexical hit", "global").await;
        let alias_id = cap_with_tags(
            &pool,
            "unrelated body that only matches through tags",
            Tags {
                retrieval_aliases: vec!["memory search".to_string()],
                domain_scope: Some("infra".to_string()),
                ..Tags::default()
            },
        )
        .await;

        let flag_off = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "memory search".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();
        let flag_on = search_thoughts(
            &pool,
            &embedder,
            None,
            SearchRequest {
                query: "memory search".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: true,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        let flag_off_ids: Vec<ThoughtId> = flag_off.results.iter().map(|r| r.thought_id).collect();
        let flag_on_ids: Vec<ThoughtId> = flag_on.results.iter().map(|r| r.thought_id).collect();
        assert!(flag_off_ids.contains(&baseline_id));
        assert!(!flag_off_ids.contains(&alias_id));
        assert!(
            flag_on_ids.contains(&baseline_id),
            "baseline lexical hit must not be dropped by soft tag/domain routing"
        );
        assert!(flag_on_ids.contains(&alias_id));
        let profile = flag_on
            .profile
            .expect("include_profile should emit profile");
        assert!(profile.full_pipeline_enabled);
        assert!(profile.tag_domain_routing_enabled);
        assert_eq!(profile.planner_route, "tag_domain_routing_v0");
        assert!(profile.tag_facet_hits >= 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn thought_scope_aliases_prevent_duplicate_active_aliases(pool: PgPool) {
        let id = cap(&pool, "alias duplicate guard", "global").await;
        let alias_id = uuid::Uuid::new_v4();
        sqlx::query(
            r#"
            INSERT INTO thought_scope_aliases
                (id, thought_id, axis, scope, confidence, source, evidence)
            VALUES
                ($1, $2, 'domain', 'infra', 1.0, 'test', '{}'::jsonb)
            "#,
        )
        .bind(alias_id)
        .bind(id.0)
        .execute(&pool)
        .await
        .unwrap();

        let duplicate = sqlx::query(
            r#"
            INSERT INTO thought_scope_aliases
                (id, thought_id, axis, scope, confidence, source, evidence)
            VALUES
                ($1, $2, 'domain', 'infra', 1.0, 'test', '{}'::jsonb)
            "#,
        )
        .bind(uuid::Uuid::new_v4())
        .bind(id.0)
        .execute(&pool)
        .await;
        assert!(
            duplicate.is_err(),
            "active partial unique index must reject duplicate active aliases"
        );

        sqlx::query("UPDATE thought_scope_aliases SET retracted_at = now() WHERE id = $1")
            .bind(alias_id)
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(
            r#"
            INSERT INTO thought_scope_aliases
                (id, thought_id, axis, scope, confidence, source, evidence)
            VALUES
                ($1, $2, 'domain', 'infra', 1.0, 'test', '{}'::jsonb)
            "#,
        )
        .bind(uuid::Uuid::new_v4())
        .bind(id.0)
        .execute(&pool)
        .await
        .unwrap();
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
                chunk_serving_enabled: false,
                full_pipeline_enabled: false,
                tag_domain_routing_enabled: false,
                include_profile: false,
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

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
    LinkDirection, Metadata, RelationKind, Scope, Source, Tags, Thought, ThoughtId,
    normalize_domain_scope, normalize_retrieval_alias, recency_boost, rrf_fuse,
};
use kengram_embed::{RerankScore, Reranker, RerankerError};
use sqlx::PgPool;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;
use time::OffsetDateTime;

use crate::query_expansion::{
    DEFAULT_QUERY_EXPANSION_MAX_HYDE_CHARS, DEFAULT_QUERY_EXPANSION_MAX_VARIANTS,
    QueryExpansionProvider, normalize_expansion_output,
};

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
pub const DEFAULT_GRAPH_SEED_COUNT: usize = 12;
pub const DEFAULT_GRAPH_PER_SEED_CAP: usize = 3;
pub const DEFAULT_GRAPH_TOTAL_CAP: usize = 24;
pub const MAX_GRAPH_SEED_COUNT: usize = 64;
pub const MAX_GRAPH_PER_SEED_CAP: usize = 16;
pub const MAX_GRAPH_TOTAL_CAP: usize = 256;

pub fn default_graph_relations() -> Vec<RelationKind> {
    vec![
        RelationKind::Supports,
        RelationKind::Requires,
        RelationKind::DecidedBy,
        RelationKind::Replaces,
        RelationKind::Refines,
    ]
}

#[derive(Debug, Clone)]
pub struct SearchRuntimeOptions {
    pub query_expansion_enabled: bool,
    pub hyde_enabled: bool,
    pub query_expansion_max_variants: usize,
    pub query_expansion_max_hyde_chars: usize,
    pub graph_augmentation_enabled: bool,
    pub graph_seed_count: usize,
    pub graph_per_seed_cap: usize,
    pub graph_total_cap: usize,
    pub graph_relations: Vec<RelationKind>,
    pub graph_direction: LinkDirection,
}

impl Default for SearchRuntimeOptions {
    fn default() -> Self {
        Self {
            query_expansion_enabled: false,
            hyde_enabled: false,
            query_expansion_max_variants: DEFAULT_QUERY_EXPANSION_MAX_VARIANTS,
            query_expansion_max_hyde_chars: DEFAULT_QUERY_EXPANSION_MAX_HYDE_CHARS,
            graph_augmentation_enabled: false,
            graph_seed_count: DEFAULT_GRAPH_SEED_COUNT,
            graph_per_seed_cap: DEFAULT_GRAPH_PER_SEED_CAP,
            graph_total_cap: DEFAULT_GRAPH_TOTAL_CAP,
            graph_relations: default_graph_relations(),
            graph_direction: LinkDirection::Both,
        }
    }
}

#[derive(Default)]
struct ExpansionRankings {
    thought_vector_rankings: Vec<Vec<Hit>>,
    chunk_vector_rankings: Vec<Vec<Hit>>,
    thought_fts_rankings: Vec<Vec<Hit>>,
    chunk_fts_rankings: Vec<Vec<Hit>>,
}

#[derive(Debug, Clone, Default)]
struct ExpansionPlan {
    route: String,
    queries: Vec<String>,
    hyde: Option<String>,
    decomposition: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct GraphExpansion {
    ranking: Vec<Hit>,
    provenance: HashMap<ThoughtId, Vec<GraphProvenance>>,
}

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
    /// Enables chunk dense + chunk FTS retrieval legs only when the server
    /// config explicitly opts in and `full_pipeline_enabled` is also true.
    /// Defaults to false at the config/server layer.
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
    pub graph_provenance: Vec<GraphProvenance>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphProvenance {
    pub seed_thought_id: String,
    pub link_id: String,
    pub relation: String,
    pub direction: String,
    pub source: String,
    pub note: Option<String>,
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
    pub query_expansion_ms: f64,
    pub query_expansion_vector_knn_ms: f64,
    pub query_expansion_fts_ms: f64,
    pub rerank_ms: f64,
    pub result_projection_ms: f64,
    pub parent_resolution_ms: f64,
    pub parent_resolution_mode: &'static str,
    pub full_pipeline_enabled: bool,
    pub chunk_serving_enabled: bool,
    pub tag_domain_routing_enabled: bool,
    pub query_expansion_enabled: bool,
    pub hyde_enabled: bool,
    pub graph_augmentation_enabled: bool,
    pub query_expansion_provider: Option<String>,
    pub query_expansion_model_id: Option<String>,
    pub query_expansion_prompt_version: Option<String>,
    pub query_expansion_fallback: bool,
    pub query_expansion_fallback_reason: Option<String>,
    pub planner_route: &'static str,
    pub query_expansion_route: Option<String>,
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
    pub query_expansion_variant_count: usize,
    pub query_expansion_decomposition_count: usize,
    pub query_expansion_hyde_used: bool,
    pub query_expansion_thought_vector_hits: usize,
    pub query_expansion_chunk_vector_hits: usize,
    pub query_expansion_thought_fts_hits: usize,
    pub query_expansion_chunk_fts_hits: usize,
    pub graph_seed_count: usize,
    pub graph_per_seed_cap: usize,
    pub graph_total_cap: usize,
    pub graph_direction: String,
    pub graph_relations: Vec<String>,
    pub graph_seeds_considered: usize,
    pub graph_candidates_before_dedupe: usize,
    pub graph_candidates_after_dedupe: usize,
    pub graph_candidates_retained_after_filters: usize,
    pub graph_merged_provenance_count: usize,
    pub graph_entity_provenance_count: i64,
    pub graph_person_provenance_count: i64,
    pub graph_url_provenance_count: i64,
    pub graph_per_relation_counts: BTreeMap<String, usize>,
    pub graph_expansion_ms: f64,
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
    search_thoughts_with_runtime(
        pool,
        embedder,
        reranker,
        None,
        SearchRuntimeOptions::default(),
        request,
    )
    .await
}

pub async fn search_thoughts_with_runtime(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: Option<&dyn Reranker>,
    query_expander: Option<&dyn QueryExpansionProvider>,
    runtime: SearchRuntimeOptions,
    request: SearchRequest,
) -> Result<SearchResponse, ReadError> {
    search_thoughts_with_tuning(
        pool,
        embedder,
        reranker,
        query_expander,
        runtime,
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
    query_expander: Option<&dyn QueryExpansionProvider>,
    runtime: SearchRuntimeOptions,
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
    let chunk_serving_enabled = request.full_pipeline_enabled && request.chunk_serving_enabled;
    let tag_domain_routing_enabled =
        request.full_pipeline_enabled && request.tag_domain_routing_enabled;
    let query_expansion_enabled = request.full_pipeline_enabled && runtime.query_expansion_enabled;
    let hyde_enabled = query_expansion_enabled && runtime.hyde_enabled;
    let graph_augmentation_enabled =
        request.full_pipeline_enabled && runtime.graph_augmentation_enabled;
    profile.full_pipeline_enabled = request.full_pipeline_enabled;
    profile.chunk_serving_enabled = chunk_serving_enabled;
    profile.tag_domain_routing_enabled = tag_domain_routing_enabled;
    profile.query_expansion_enabled = query_expansion_enabled;
    profile.hyde_enabled = hyde_enabled;
    profile.graph_augmentation_enabled = graph_augmentation_enabled;
    profile.graph_seed_count = runtime.graph_seed_count.min(MAX_GRAPH_SEED_COUNT);
    profile.graph_per_seed_cap = runtime.graph_per_seed_cap.min(MAX_GRAPH_PER_SEED_CAP);
    profile.graph_total_cap = runtime.graph_total_cap.min(MAX_GRAPH_TOTAL_CAP);
    profile.graph_direction = runtime.graph_direction.as_str().to_string();
    profile.graph_relations = runtime
        .graph_relations
        .iter()
        .map(|relation| relation.as_str().to_string())
        .collect();
    profile.planner_route = if tag_domain_routing_enabled {
        "tag_domain_routing_v0"
    } else {
        "baseline"
    };
    let expansion_plan = build_expansion_plan(
        query_expander,
        query_expansion_enabled,
        hyde_enabled,
        &runtime,
        &query,
        &mut profile,
    )
    .await;

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

    let expansion_rankings = if expansion_plan
        .as_ref()
        .is_some_and(|plan| plan.has_generated_inputs())
    {
        collect_expansion_rankings(
            pool,
            embedder,
            expansion_plan.as_ref().expect("checked above"),
            chunk_serving_enabled,
            scope_filter,
            scope_prefix_filter,
            lexical_top_k,
            lexical_timeout_ms,
            &mut profile,
        )
        .await
    } else {
        ExpansionRankings::default()
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
    rankings.extend(expansion_rankings.thought_vector_rankings);
    rankings.extend(expansion_rankings.thought_fts_rankings);
    if chunk_serving_enabled {
        rankings.extend(expansion_rankings.chunk_vector_rankings);
        rankings.extend(expansion_rankings.chunk_fts_rankings);
    }
    let mut fused = rrf_fuse(rankings, DEFAULT_RRF_K);
    let graph_expansion = collect_graph_expansion(
        pool,
        graph_augmentation_enabled,
        &runtime,
        &fused,
        scope_filter,
        scope_prefix_filter,
        &mut profile,
    )
    .await?;
    merge_graph_expansion(&mut fused, &graph_expansion, DEFAULT_RRF_K);
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
    profile.graph_candidates_retained_after_filters = fused
        .iter()
        .filter(|hit| graph_expansion.provenance.contains_key(&hit.thought.id))
        .count();
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
        .map(|hit| search_hit_from_core_hit(hit, &graph_expansion.provenance))
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

impl ExpansionPlan {
    fn has_generated_inputs(&self) -> bool {
        !self.queries.is_empty() || !self.decomposition.is_empty() || self.hyde.is_some()
    }

    fn generated_inputs(&self) -> Vec<String> {
        let mut values = Vec::new();
        values.extend(self.queries.clone());
        values.extend(self.decomposition.clone());
        if let Some(hyde) = self.hyde.clone() {
            values.push(hyde);
        }
        dedupe_lowercase_limited(values, 16)
    }
}

async fn build_expansion_plan(
    query_expander: Option<&dyn QueryExpansionProvider>,
    query_expansion_enabled: bool,
    hyde_enabled: bool,
    runtime: &SearchRuntimeOptions,
    query: &str,
    profile: &mut SearchProfile,
) -> Option<ExpansionPlan> {
    if !query_expansion_enabled {
        return None;
    }
    profile.planner_route = "query_expansion_v0";
    let started = Instant::now();
    let expander = match query_expander {
        Some(expander) => expander,
        None => {
            profile.query_expansion_fallback = true;
            profile.query_expansion_fallback_reason = Some("provider_disabled".to_string());
            profile.query_expansion_ms = elapsed_ms(started);
            return None;
        }
    };
    profile.query_expansion_provider = Some(expander.provider_name().to_string());
    profile.query_expansion_model_id = Some(expander.model_id().to_string());
    profile.query_expansion_prompt_version = Some(expander.prompt_version().to_string());

    let max_variants = runtime.query_expansion_max_variants.min(8);
    match expander
        .expand(crate::query_expansion::QueryExpansionInput {
            query: query.to_string(),
            max_variants,
            hyde_enabled,
        })
        .await
    {
        Ok(raw) => {
            let normalized = normalize_expansion_output(
                query,
                raw,
                max_variants,
                hyde_enabled,
                runtime.query_expansion_max_hyde_chars,
            );
            profile.query_expansion_route = Some(normalized.route.clone());
            profile.query_expansion_variant_count = normalized.queries.len();
            profile.query_expansion_decomposition_count = normalized.decomposition.len();
            profile.query_expansion_hyde_used = normalized.hyde.is_some();
            profile.query_expansion_ms = elapsed_ms(started);
            Some(ExpansionPlan {
                route: normalized.route,
                queries: normalized.queries,
                hyde: normalized.hyde,
                decomposition: normalized.decomposition,
            })
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                reason = e.reason_code(),
                "query expansion provider failed; falling back to original-query-only retrieval",
            );
            profile.query_expansion_fallback = true;
            profile.query_expansion_fallback_reason = Some(e.reason_code().to_string());
            profile.query_expansion_ms = elapsed_ms(started);
            None
        }
    }
}

async fn collect_expansion_rankings(
    pool: &PgPool,
    embedder: &dyn Embedder,
    plan: &ExpansionPlan,
    chunk_serving_enabled: bool,
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    lexical_top_k: usize,
    lexical_timeout_ms: u64,
    profile: &mut SearchProfile,
) -> ExpansionRankings {
    let inputs = plan.generated_inputs();
    if inputs.is_empty() {
        return ExpansionRankings::default();
    }

    let mut out = ExpansionRankings::default();
    let vector_started = Instant::now();
    match embedder.embed(&inputs).await {
        Ok(vectors) => {
            for vector in vectors {
                match kengram_storage::search_vector_knn(
                    pool,
                    vector.clone(),
                    embedder.model(),
                    scope_filter,
                    scope_prefix_filter,
                    DEFAULT_TOP_K_PER_LEG as i64,
                )
                .await
                {
                    Ok(hits) => {
                        profile.query_expansion_thought_vector_hits += hits.len();
                        if !hits.is_empty() {
                            out.thought_vector_rankings.push(hits);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "query-expansion thought vector leg failed; continuing",
                        );
                    }
                }
                if chunk_serving_enabled {
                    match kengram_storage::search_artifact_chunks_vector_knn(
                        pool,
                        vector,
                        embedder.model(),
                        scope_filter,
                        scope_prefix_filter,
                        DEFAULT_TOP_K_PER_LEG as i64,
                    )
                    .await
                    {
                        Ok(hits) => {
                            profile.query_expansion_chunk_vector_hits += hits.len();
                            if !hits.is_empty() {
                                out.chunk_vector_rankings.push(hits);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "query-expansion chunk vector leg failed; continuing",
                            );
                        }
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "embedder failed for query-expansion variants; keeping lexical expansion legs only",
            );
        }
    }
    profile.query_expansion_vector_knn_ms = elapsed_ms(vector_started);

    let fts_started = Instant::now();
    for input in &inputs {
        let thought_hits = bounded_fts_hits(
            pool,
            input,
            scope_filter,
            scope_prefix_filter,
            lexical_top_k,
            lexical_timeout_ms,
        )
        .await;
        profile.query_expansion_thought_fts_hits += thought_hits.len();
        if !thought_hits.is_empty() {
            out.thought_fts_rankings.push(thought_hits);
        }
        if chunk_serving_enabled {
            let chunk_hits = bounded_artifact_chunk_fts_hits(
                pool,
                input,
                scope_filter,
                scope_prefix_filter,
                lexical_top_k,
                lexical_timeout_ms,
            )
            .await;
            profile.query_expansion_chunk_fts_hits += chunk_hits.len();
            if !chunk_hits.is_empty() {
                out.chunk_fts_rankings.push(chunk_hits);
            }
        }
    }
    profile.query_expansion_fts_ms = elapsed_ms(fts_started);
    tracing::debug!(
        route = %plan.route,
        inputs = inputs.len(),
        "query-expansion legs collected",
    );
    out
}

async fn collect_graph_expansion(
    pool: &PgPool,
    graph_augmentation_enabled: bool,
    runtime: &SearchRuntimeOptions,
    fused: &[Hit],
    scope_filter: Option<&str>,
    scope_prefix_filter: Option<&str>,
    profile: &mut SearchProfile,
) -> Result<GraphExpansion, ReadError> {
    if !graph_augmentation_enabled || fused.is_empty() {
        return Ok(GraphExpansion::default());
    }

    let started = Instant::now();
    let seed_count = runtime.graph_seed_count.min(MAX_GRAPH_SEED_COUNT);
    let per_seed_cap = runtime.graph_per_seed_cap.min(MAX_GRAPH_PER_SEED_CAP);
    let total_cap = runtime.graph_total_cap.min(MAX_GRAPH_TOTAL_CAP);
    if seed_count == 0 || per_seed_cap == 0 || total_cap == 0 {
        profile.graph_expansion_ms = elapsed_ms(started);
        return Ok(GraphExpansion::default());
    }

    let seeds = fused
        .iter()
        .take(seed_count)
        .map(|hit| hit.thought.id)
        .collect::<Vec<_>>();
    profile.graph_seeds_considered = seeds.len();

    let non_thought = kengram_storage::count_graph_non_thought_targets(
        pool,
        &seeds,
        &runtime.graph_relations,
        runtime.graph_direction,
    )
    .await?;
    profile.graph_entity_provenance_count = non_thought.entity;
    profile.graph_person_provenance_count = non_thought.person;
    profile.graph_url_provenance_count = non_thought.url;

    let rows = kengram_storage::search_graph_neighbors(
        pool,
        &seeds,
        &runtime.graph_relations,
        runtime.graph_direction,
        per_seed_cap,
        total_cap,
        scope_filter,
        scope_prefix_filter,
    )
    .await?;
    profile.graph_candidates_before_dedupe = rows.len();

    let base_ids = fused
        .iter()
        .map(|hit| hit.thought.id)
        .collect::<HashSet<_>>();
    let mut by_thought: HashMap<ThoughtId, Hit> = HashMap::new();
    let mut provenance: HashMap<ThoughtId, Vec<GraphProvenance>> = HashMap::new();
    let mut per_relation = BTreeMap::<String, usize>::new();
    let mut merged_existing = HashSet::new();
    let mut order = Vec::new();

    for row in rows {
        let thought_id = row.thought.id;
        *per_relation
            .entry(row.relation.as_str().to_string())
            .or_default() += 1;
        if base_ids.contains(&thought_id) {
            merged_existing.insert(thought_id);
        }
        provenance
            .entry(thought_id)
            .or_default()
            .push(GraphProvenance {
                seed_thought_id: row.seed_thought_id.as_uuid().to_string(),
                link_id: row.link_id.as_uuid().to_string(),
                relation: row.relation.as_str().to_string(),
                direction: row.direction.as_str().to_string(),
                source: row.link_source.as_str().to_string(),
                note: row.note,
            });
        by_thought.entry(thought_id).or_insert_with(|| {
            order.push(thought_id);
            Hit {
                thought: row.thought,
                vector_score: None,
                lexical_score: None,
                trigram_score: None,
                rrf_score: None,
                rerank_score: None,
                chunk: None,
            }
        });
    }

    let ranking = order
        .into_iter()
        .filter_map(|id| by_thought.remove(&id))
        .collect::<Vec<_>>();

    profile.graph_candidates_after_dedupe = ranking.len();
    profile.graph_merged_provenance_count = merged_existing.len();
    profile.graph_per_relation_counts = per_relation;
    profile.graph_expansion_ms = elapsed_ms(started);

    Ok(GraphExpansion {
        ranking,
        provenance,
    })
}

fn merge_graph_expansion(fused: &mut Vec<Hit>, graph: &GraphExpansion, k: f32) {
    if graph.ranking.is_empty() {
        return;
    }

    let mut positions = fused
        .iter()
        .enumerate()
        .map(|(idx, hit)| (hit.thought.id, idx))
        .collect::<HashMap<_, _>>();
    for (idx, graph_hit) in graph.ranking.iter().cloned().enumerate() {
        let rank = (idx + 1) as f32;
        let contribution = 1.0 / (k + rank);
        if let Some(existing_idx) = positions.get(&graph_hit.thought.id).copied() {
            let current = fused[existing_idx].rrf_score.unwrap_or(0.0);
            fused[existing_idx].rrf_score = Some(current + contribution);
        } else {
            let mut inserted = graph_hit;
            inserted.rrf_score = Some(contribution);
            positions.insert(inserted.thought.id, fused.len());
            fused.push(inserted);
        }
    }
    fused.sort_by(|a, b| {
        let av = a.rrf_score.unwrap_or(0.0);
        let bv = b.rrf_score.unwrap_or(0.0);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn search_hit_from_core_hit(
    h: Hit,
    graph_provenance: &HashMap<ThoughtId, Vec<GraphProvenance>>,
) -> SearchHit {
    let chunk = h.chunk;
    let graph_provenance = graph_provenance
        .get(&h.thought.id)
        .cloned()
        .unwrap_or_default();
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
        graph_provenance,
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
    use crate::query_expansion::{
        QueryExpansionError, QueryExpansionInput, QueryExpansionOutput, QueryExpansionProvider,
    };
    use async_trait::async_trait;
    use kengram_core::{
        EmbeddingModel, LinkDirection, LinkSource, LinkTarget, RelationKind, TagKind, Tags,
    };
    use kengram_embed::{FakeBehavior, FakeEmbedder, FakeReranker};

    const TEST_EMBEDDER_MODEL_ID: &str = "qwen3-embedding";

    fn test_embedding_model() -> EmbeddingModel {
        EmbeddingModel::new(TEST_EMBEDDER_MODEL_ID, 4096)
    }

    struct StaticQueryExpander {
        output: Result<QueryExpansionOutput, QueryExpansionError>,
    }

    #[async_trait]
    impl QueryExpansionProvider for StaticQueryExpander {
        fn provider_name(&self) -> &'static str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake/query-expander"
        }

        fn prompt_version(&self) -> &str {
            "fake-v1"
        }

        async fn expand(
            &self,
            _input: QueryExpansionInput,
        ) -> Result<QueryExpansionOutput, QueryExpansionError> {
            self.output.clone()
        }
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
            include_profile: true,
        };
        let off = search_thoughts(&pool, &bad, None, request.clone())
            .await
            .unwrap();
        assert!(off.results.is_empty());
        let off_profile = off.profile.expect("profile requested");
        assert!(!off_profile.full_pipeline_enabled);
        assert!(!off_profile.chunk_serving_enabled);
        assert_eq!(off_profile.chunk_fts_hits, 0);

        request.chunk_serving_enabled = true;
        let subordinate_only = search_thoughts(&pool, &bad, None, request.clone())
            .await
            .unwrap();
        assert!(subordinate_only.results.is_empty());
        let subordinate_profile = subordinate_only.profile.expect("profile requested");
        assert!(!subordinate_profile.full_pipeline_enabled);
        assert!(!subordinate_profile.chunk_serving_enabled);
        assert_eq!(subordinate_profile.chunk_fts_hits, 0);

        request.full_pipeline_enabled = true;
        let on = search_thoughts(&pool, &bad, None, request).await.unwrap();
        assert_eq!(on.results.len(), 1);
        let on_profile = on.profile.as_ref().expect("profile requested");
        assert!(on_profile.full_pipeline_enabled);
        assert!(on_profile.chunk_serving_enabled);
        assert_eq!(on_profile.chunk_fts_hits, 1);
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
            None,
            SearchRuntimeOptions::default(),
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

    async fn cap_with_tags_in_scope(
        pool: &PgPool,
        content: &str,
        scope: &str,
        tags: Tags,
    ) -> ThoughtId {
        let id = cap(pool, content, scope).await;
        kengram_storage::update_thought_tags(pool, id, &tags, "fake/tagger", 1)
            .await
            .unwrap();
        id
    }

    async fn link_thoughts_for_test(
        from: ThoughtId,
        relation: RelationKind,
        to: ThoughtId,
        pool: &PgPool,
    ) {
        kengram_storage::insert_link(
            pool,
            from,
            relation,
            &LinkTarget::Thought(to),
            LinkSource::Agent,
            None,
        )
        .await
        .unwrap();
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
    async fn query_expansion_subordinate_flag_is_ineffective_when_master_gate_off(pool: PgPool) {
        let embedder = test_embedder();
        let baseline_id = cap(&pool, "original-only baseline marker", "global").await;
        let expansion_only_id = cap(&pool, "variant-only hidden marker", "global").await;
        let expander = StaticQueryExpander {
            output: Ok(QueryExpansionOutput {
                route: Some("semantic".to_string()),
                queries: vec!["variant-only hidden".to_string()],
                hyde: None,
                decomposition: vec![],
                facets: Default::default(),
            }),
        };
        let request = SearchRequest {
            query: "original-only baseline".to_string(),
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
        let runtime = SearchRuntimeOptions {
            query_expansion_enabled: true,
            hyde_enabled: true,
            ..SearchRuntimeOptions::default()
        };

        let resp =
            search_thoughts_with_runtime(&pool, &embedder, None, Some(&expander), runtime, request)
                .await
                .unwrap();
        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&baseline_id));
        assert!(!ids.contains(&expansion_only_id));
        let profile = resp.profile.expect("profile requested");
        assert!(!profile.full_pipeline_enabled);
        assert!(!profile.query_expansion_enabled);
        assert_eq!(profile.planner_route, "baseline");
        assert_eq!(profile.query_expansion_variant_count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn query_expansion_provider_failure_falls_back_to_original_query(pool: PgPool) {
        let embedder = test_embedder();
        let baseline_id = cap(&pool, "fallback baseline marker", "global").await;
        let expander = StaticQueryExpander {
            output: Err(QueryExpansionError::Timeout { seconds: 1 }),
        };
        let resp = search_thoughts_with_runtime(
            &pool,
            &embedder,
            None,
            Some(&expander),
            SearchRuntimeOptions {
                query_expansion_enabled: true,
                hyde_enabled: true,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "fallback baseline".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();
        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&baseline_id));
        let profile = resp.profile.expect("profile requested");
        assert!(profile.full_pipeline_enabled);
        assert!(profile.query_expansion_enabled);
        assert!(profile.query_expansion_fallback);
        assert_eq!(
            profile.query_expansion_fallback_reason.as_deref(),
            Some("timeout")
        );
        assert_eq!(profile.query_expansion_variant_count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn query_expansion_flag_true_adds_candidates_without_dropping_original(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let baseline_id = cap(&pool, "original belt four baseline", "global").await;
        let variant_id = cap(&pool, "expanded semantic marker", "global").await;
        let hyde_id = cap(&pool, "pseudo document marker", "global").await;
        let expander = StaticQueryExpander {
            output: Ok(QueryExpansionOutput {
                route: Some("semantic".to_string()),
                queries: vec!["expanded semantic marker".to_string()],
                hyde: Some("pseudo document marker".to_string()),
                decomposition: vec![],
                facets: Default::default(),
            }),
        };
        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            Some(&expander),
            SearchRuntimeOptions {
                query_expansion_enabled: true,
                hyde_enabled: true,
                query_expansion_max_variants: 4,
                query_expansion_max_hyde_chars: 600,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "original belt four".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();
        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&baseline_id));
        assert!(ids.contains(&variant_id));
        assert!(ids.contains(&hyde_id));
        let profile = resp.profile.expect("profile requested");
        assert_eq!(profile.planner_route, "query_expansion_v0");
        assert_eq!(profile.query_expansion_route.as_deref(), Some("semantic"));
        assert_eq!(profile.query_expansion_variant_count, 1);
        assert!(profile.query_expansion_hyde_used);
        assert!(profile.query_expansion_thought_fts_hits >= 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_subordinate_flag_is_ineffective_when_master_gate_off(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let seed_id = cap(&pool, "graph seed alpha marker", "global").await;
        let neighbor_id = cap(&pool, "linked neighbor only", "global").await;
        link_thoughts_for_test(seed_id, RelationKind::Supports, neighbor_id, &pool).await;

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "graph seed alpha".to_string(),
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

        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&seed_id));
        assert!(!ids.contains(&neighbor_id));
        let profile = resp.profile.expect("profile requested");
        assert!(!profile.full_pipeline_enabled);
        assert!(!profile.graph_augmentation_enabled);
        assert_eq!(profile.graph_candidates_before_dedupe, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_flag_true_adds_neighbor_with_provenance_without_dropping_base(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let seed_id = cap(&pool, "graph seed beta marker", "global").await;
        let neighbor_id = cap(&pool, "linked graph neighbor beta", "global").await;
        link_thoughts_for_test(seed_id, RelationKind::Supports, neighbor_id, &pool).await;

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                graph_seed_count: 12,
                graph_per_seed_cap: 3,
                graph_total_cap: 24,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "graph seed beta".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&seed_id));
        assert!(ids.contains(&neighbor_id));
        let neighbor = resp
            .results
            .iter()
            .find(|hit| hit.thought_id == neighbor_id)
            .expect("graph neighbor should be returned");
        assert_eq!(neighbor.graph_provenance.len(), 1);
        assert_eq!(neighbor.graph_provenance[0].relation, "supports");
        assert_eq!(neighbor.graph_provenance[0].direction, "outbound");
        let profile = resp.profile.expect("profile requested");
        assert!(profile.graph_augmentation_enabled);
        assert_eq!(profile.graph_candidates_before_dedupe, 1);
        assert_eq!(profile.graph_candidates_after_dedupe, 1);
        assert_eq!(
            profile.graph_per_relation_counts.get("supports").copied(),
            Some(1)
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_seed_count_limits_which_fused_hits_expand(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let first_seed = cap(&pool, "graph seed cap alpha primary", "global").await;
        let second_seed = cap(&pool, "graph seed cap alpha secondary", "global").await;
        let first_neighbor = cap(&pool, "first seed graph neighbor", "global").await;
        let second_neighbor = cap(&pool, "second seed graph neighbor", "global").await;
        link_thoughts_for_test(first_seed, RelationKind::Supports, first_neighbor, &pool).await;
        link_thoughts_for_test(second_seed, RelationKind::Supports, second_neighbor, &pool).await;

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                graph_seed_count: 1,
                graph_direction: LinkDirection::Outbound,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "graph seed cap alpha primary".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&first_seed));
        assert!(ids.contains(&first_neighbor));
        assert!(!ids.contains(&second_neighbor));
        let profile = resp.profile.expect("profile requested");
        assert_eq!(profile.graph_seeds_considered, 1);
        assert_eq!(profile.graph_candidates_before_dedupe, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_duplicate_base_hit_merges_provenance(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let seed_id = cap(&pool, "duplicate graph seed marker", "global").await;
        let duplicate_id = cap(&pool, "duplicate graph seed neighbor marker", "global").await;
        link_thoughts_for_test(seed_id, RelationKind::Supports, duplicate_id, &pool).await;

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                graph_direction: LinkDirection::Outbound,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "duplicate graph seed".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        let duplicate_hits = resp
            .results
            .iter()
            .filter(|hit| hit.thought_id == duplicate_id)
            .count();
        assert_eq!(duplicate_hits, 1, "base and graph duplicate must merge");
        let hit = resp
            .results
            .iter()
            .find(|hit| hit.thought_id == duplicate_id)
            .unwrap();
        assert_eq!(hit.graph_provenance.len(), 1);
        let profile = resp.profile.expect("profile requested");
        assert_eq!(profile.graph_merged_provenance_count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_tag_filter_remains_hard_before_rerank(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let seed_id = cap_with_tags_in_scope(
            &pool,
            "tag filter graph seed",
            "global",
            Tags {
                kind: Some(TagKind::Task),
                ..Tags::default()
            },
        )
        .await;
        let neighbor_id = cap_with_tags_in_scope(
            &pool,
            "graph neighbor filtered by tag",
            "global",
            Tags {
                kind: Some(TagKind::Idea),
                ..Tags::default()
            },
        )
        .await;
        link_thoughts_for_test(seed_id, RelationKind::Supports, neighbor_id, &pool).await;

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "tag filter graph seed".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: Some(serde_json::json!({"kind": "task"})),
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert_eq!(ids, vec![seed_id]);
        assert!(!ids.contains(&neighbor_id));
        let profile = resp.profile.expect("profile requested");
        assert_eq!(profile.graph_candidates_before_dedupe, 1);
        assert_eq!(profile.graph_candidates_retained_after_filters, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_neighbor_contamination_never_reaches_rerank_pool(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let reranker = FakeReranker::new();
        let seed_id = cap(&pool, "graph contamination clean seed", "global").await;
        let clean_neighbor = cap(&pool, "clean graph neighbor evidence", "global").await;
        let denied_neighbor = cap(&pool, "KGR024 denied graph neighbor evidence", "global").await;
        link_thoughts_for_test(seed_id, RelationKind::Supports, clean_neighbor, &pool).await;
        link_thoughts_for_test(seed_id, RelationKind::Supports, denied_neighbor, &pool).await;

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            Some(&reranker),
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                graph_per_seed_cap: 5,
                graph_total_cap: 5,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "graph contamination clean seed".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(true),
                candidate_pool: Some(10),
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        assert!(resp.rerank_used);
        let rerank_call = reranker
            .last_call()
            .expect("reranker should record candidate pool");
        assert!(
            rerank_call
                .candidates
                .iter()
                .any(|candidate| candidate.contains("clean graph neighbor evidence"))
        );
        assert!(
            rerank_call
                .candidates
                .iter()
                .all(|candidate| !candidate.contains("KGR024")),
            "denied graph neighbor must not enter rerank pool"
        );
        let ids: Vec<_> = resp.results.iter().map(|hit| hit.thought_id).collect();
        assert!(ids.contains(&clean_neighbor));
        assert!(!ids.contains(&denied_neighbor));
        let profile = resp.profile.expect("profile requested");
        assert_eq!(profile.graph_candidates_before_dedupe, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_non_thought_targets_are_profiled_not_returned_as_hits(pool: PgPool) {
        let bad = FakeEmbedder::always_failing(test_embedding_model(), FakeBehavior::Unreachable);
        let seed_id = cap(&pool, "non thought graph seed", "global").await;
        for target in [
            LinkTarget::Entity("Model T".to_string()),
            LinkTarget::Person("Bob Hrbek".to_string()),
            LinkTarget::Url("https://example.com/model-t".to_string()),
        ] {
            kengram_storage::insert_link(
                &pool,
                seed_id,
                RelationKind::Supports,
                &target,
                LinkSource::Agent,
                None,
            )
            .await
            .unwrap();
        }

        let resp = search_thoughts_with_runtime(
            &pool,
            &bad,
            None,
            None,
            SearchRuntimeOptions {
                graph_augmentation_enabled: true,
                ..SearchRuntimeOptions::default()
            },
            SearchRequest {
                query: "non thought graph seed".to_string(),
                scope: None,
                scope_prefix: None,
                limit: Some(10),
                recency_half_life_days: Some(0.0),
                rerank: Some(false),
                candidate_pool: None,
                tag_filter: None,
                chunk_serving_enabled: false,
                full_pipeline_enabled: true,
                tag_domain_routing_enabled: false,
                include_profile: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].thought_id, seed_id);
        let profile = resp.profile.expect("profile requested");
        assert_eq!(profile.graph_candidates_before_dedupe, 0);
        assert_eq!(profile.graph_entity_provenance_count, 1);
        assert_eq!(profile.graph_person_provenance_count, 1);
        assert_eq!(profile.graph_url_provenance_count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_storage_respects_relation_direction_scope_and_caps(pool: PgPool) {
        let seed = cap(&pool, "graph storage seed", "work.alpha").await;
        let supports_out = cap(&pool, "supports outbound", "work.alpha").await;
        let requires_out = cap(&pool, "requires outbound", "work.alpha").await;
        let supports_in = cap(&pool, "supports inbound", "work.alpha").await;
        let other_scope = cap(&pool, "supports other scope", "personal").await;
        link_thoughts_for_test(seed, RelationKind::Supports, supports_out, &pool).await;
        link_thoughts_for_test(seed, RelationKind::Requires, requires_out, &pool).await;
        link_thoughts_for_test(supports_in, RelationKind::Supports, seed, &pool).await;
        link_thoughts_for_test(seed, RelationKind::Supports, other_scope, &pool).await;

        let outbound_supports = kengram_storage::search_graph_neighbors(
            &pool,
            &[seed],
            &[RelationKind::Supports],
            LinkDirection::Outbound,
            10,
            10,
            Some("work.alpha"),
            None,
        )
        .await
        .unwrap();
        let outbound_ids = outbound_supports
            .iter()
            .map(|hit| hit.thought.id)
            .collect::<Vec<_>>();
        assert_eq!(outbound_ids, vec![supports_out]);

        let inbound_supports = kengram_storage::search_graph_neighbors(
            &pool,
            &[seed],
            &[RelationKind::Supports],
            LinkDirection::Inbound,
            10,
            10,
            None,
            Some("work."),
        )
        .await
        .unwrap();
        let inbound_ids = inbound_supports
            .iter()
            .map(|hit| hit.thought.id)
            .collect::<Vec<_>>();
        assert_eq!(inbound_ids, vec![supports_in]);

        let capped = kengram_storage::search_graph_neighbors(
            &pool,
            &[seed],
            &[RelationKind::Supports, RelationKind::Requires],
            LinkDirection::Both,
            2,
            2,
            None,
            Some("work."),
        )
        .await
        .unwrap();
        assert_eq!(capped.len(), 2, "per-seed and total cap must bind");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_storage_total_cap_binds_across_multiple_seeds(pool: PgPool) {
        let seed_a = cap(&pool, "graph total cap seed a", "work.alpha").await;
        let seed_b = cap(&pool, "graph total cap seed b", "work.alpha").await;
        let a1 = cap(&pool, "graph total cap a1", "work.alpha").await;
        let a2 = cap(&pool, "graph total cap a2", "work.alpha").await;
        let b1 = cap(&pool, "graph total cap b1", "work.alpha").await;
        let b2 = cap(&pool, "graph total cap b2", "work.alpha").await;
        link_thoughts_for_test(seed_a, RelationKind::Supports, a1, &pool).await;
        link_thoughts_for_test(seed_a, RelationKind::Supports, a2, &pool).await;
        link_thoughts_for_test(seed_b, RelationKind::Supports, b1, &pool).await;
        link_thoughts_for_test(seed_b, RelationKind::Supports, b2, &pool).await;

        let capped = kengram_storage::search_graph_neighbors(
            &pool,
            &[seed_a, seed_b],
            &[RelationKind::Supports],
            LinkDirection::Outbound,
            2,
            3,
            None,
            Some("work."),
        )
        .await
        .unwrap();

        assert_eq!(capped.len(), 3, "total cap must bind after per-seed caps");
        assert_eq!(
            capped
                .iter()
                .filter(|hit| hit.seed_thought_id == seed_a)
                .count(),
            2
        );
        assert_eq!(
            capped
                .iter()
                .filter(|hit| hit.seed_thought_id == seed_b)
                .count(),
            1
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn graph_storage_excludes_soft_deleted_and_retracted_neighbors(pool: PgPool) {
        let seed = cap(&pool, "graph exclusion seed", "global").await;
        let clean = cap(&pool, "graph clean neighbor", "global").await;
        let deleted = cap(&pool, "graph deleted neighbor", "global").await;
        let retracted = cap(&pool, "graph retracted neighbor", "global").await;
        link_thoughts_for_test(seed, RelationKind::Supports, clean, &pool).await;
        link_thoughts_for_test(seed, RelationKind::Supports, deleted, &pool).await;
        link_thoughts_for_test(seed, RelationKind::Supports, retracted, &pool).await;
        kengram_storage::delete_link(
            &pool,
            seed,
            RelationKind::Supports,
            &LinkTarget::Thought(deleted),
        )
        .await
        .unwrap();
        kengram_storage::retract_thought(&pool, retracted, Some("test retraction"))
            .await
            .unwrap();

        let hits = kengram_storage::search_graph_neighbors(
            &pool,
            &[seed],
            &[RelationKind::Supports],
            LinkDirection::Outbound,
            10,
            10,
            None,
            None,
        )
        .await
        .unwrap();
        let ids = hits.iter().map(|hit| hit.thought.id).collect::<Vec<_>>();
        assert_eq!(ids, vec![clean]);
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

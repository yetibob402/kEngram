//! Read operations: `search_thoughts`, `recent_thoughts`, `get_thought`.
//!
//! `search_thoughts` is the hybrid retrieval path: vector kNN ∪ trigram
//! similarity, fused by RRF, then recency-boosted. If the embedder is
//! unreachable, the vector leg is skipped and `vector_search_available`
//! flips to `false`; trigram-only results still come back.

use engram_core::{
    DEFAULT_RECENCY_HALF_LIFE_DAYS, DEFAULT_RRF_K, Embedder, EmbeddingModel, EmbeddingStatus, Fact,
    Metadata, Scope, Source, Thought, ThoughtId, recency_boost, rrf_fuse,
};
use sqlx::PgPool;
use time::OffsetDateTime;
use uuid::Uuid;

pub const DEFAULT_SEARCH_LIMIT: usize = 10;
pub const MAX_SEARCH_LIMIT: usize = 100;
pub const DEFAULT_TOP_K_PER_LEG: usize = 50;

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,
    pub recency_half_life_days: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub thought_id: ThoughtId,
    pub content: String,
    pub scope: Scope,
    pub source: Source,
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub vector_search_available: bool,
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
}

#[derive(Debug, Clone)]
pub struct SearchFactHit {
    pub fact_id: Uuid,
    pub statement: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub confidence: f32,
    pub source_thought_id: ThoughtId,
    pub source_thought_content: String,
    pub source_thought_scope: Scope,
    pub source_thought_created_at: OffsetDateTime,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchFactsResponse {
    pub results: Vec<SearchFactHit>,
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

    // RRF fuse → recency boost → truncate.
    let mut fused = rrf_fuse(vec![vector_hits, trigram_hits], DEFAULT_RRF_K);
    let half_life = request
        .recency_half_life_days
        .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_DAYS);
    recency_boost(&mut fused, half_life, OffsetDateTime::now_utc());

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
            score: h.score,
        })
        .collect();

    Ok(SearchResponse {
        results,
        vector_search_available,
    })
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

/// Search over `facts` — trigram-only in Phase D inside the same response
/// shape as `search_thoughts`. The vector leg lands in M3 (search quality)
/// once fact embeddings are wired through the queue. The recency boost is
/// keyed on `source_thought_created_at` — the thought's recency is a better
/// freshness signal for a fact than the fact's own `created_at` (which is
/// when the reflector ran, not when the underlying thought was captured).
pub async fn search_facts(
    pool: &PgPool,
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

    let mut hits = engram_storage::search_facts_trigram(
        pool,
        &request.query,
        scope_filter,
        DEFAULT_TOP_K_PER_LEG as i64,
    )
    .await?;

    // Recency boost keyed on the source thought's created_at. With only one
    // ranking leg, RRF normalization is identity, so we skip rrf_fuse and
    // apply the boost directly to the trigram score. When M3 adds the
    // vector leg, this becomes a real fuse-then-boost pipeline; we'll need
    // to refactor `rrf_fuse` to be generic (or fold FactHit through a
    // ranking abstraction) at that point.
    let half_life = request
        .recency_half_life_days
        .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_DAYS);
    if half_life > 0.0 {
        let now = OffsetDateTime::now_utc();
        for h in hits.iter_mut() {
            let age_secs = (now - h.source_thought_created_at).whole_seconds() as f32;
            let age_days = age_secs / 86_400.0;
            h.score *= 0.5_f32.powf(age_days / half_life);
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let results: Vec<SearchFactHit> = hits
        .into_iter()
        .take(limit)
        .map(|h| SearchFactHit {
            fact_id: h.fact.id,
            statement: h.fact.statement,
            subject: h.fact.subject,
            predicate: h.fact.predicate,
            object: h.fact.object,
            confidence: h.fact.confidence,
            source_thought_id: h.fact.source_thought_id,
            source_thought_content: h.source_thought_content,
            source_thought_scope: h.source_thought_scope,
            source_thought_created_at: h.source_thought_created_at,
            score: h.score,
        })
        .collect();

    Ok(SearchFactsResponse { results })
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
            &embedder,
            SearchRequest {
                query: "alpha".to_string(),
                scope: None,
                limit: Some(10),
                recency_half_life_days: None,
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
            &bad,
            SearchRequest {
                query: "tcgplayer".to_string(),
                scope: None,
                limit: Some(10),
                recency_half_life_days: None,
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
            &embedder,
            SearchRequest {
                query: String::new(),
                scope: None,
                limit: None,
                recency_half_life_days: None,
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
            &embedder,
            SearchRequest {
                query: "x".to_string(),
                scope: None,
                limit: Some(1000),
                recency_half_life_days: None,
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
            &embedder,
            SearchRequest {
                query: "tcgplayer".to_string(),
                scope: Some(Scope::new("work").unwrap()),
                limit: Some(10),
                recency_half_life_days: None,
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
            SearchFactsRequest {
                query: "pgvector".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0), // disable boost for deterministic ordering
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
            SearchFactsRequest {
                query: "widgets".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
            },
        )
        .await
        .unwrap();
        assert_eq!(before.results.len(), 1);

        engram_storage::supersede_fact(&pool, fact_id, None).await.unwrap();

        let after = search_facts(
            &pool,
            SearchFactsRequest {
                query: "widgets".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
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
            SearchFactsRequest {
                query: String::new(),
                scope: None,
                limit: None,
                recency_half_life_days: None,
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
            SearchFactsRequest {
                query: "x".to_string(),
                scope: None,
                limit: Some(1000),
                recency_half_life_days: None,
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
            SearchFactsRequest {
                query: "widget".to_string(),
                scope: Some(Scope::new("work").unwrap()),
                limit: None,
                recency_half_life_days: Some(0.0),
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
}

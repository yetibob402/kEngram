//! `engram bench rerank` — A/B benchmarking harness comparing RRF-only
//! vs cross-encoder-reranked search orderings on an operator-curated
//! fixture corpus. Closes M3 success criterion 1 by turning the
//! qualitative "rerank feels better" signal into nDCG@10 / MRR numbers.
//!
//! M4: thoughts-only. The `target` field on each query (and the
//! `BenchTarget` enum it selected) is gone — facts no longer exist.
//! See `tests/fixtures/bench-rerank.example.json` for the corpus schema.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

use anyhow::Context;
use engram_core::{Embedder, Scope, ScopeError, ndcg_at_k, reciprocal_rank};
use engram_embed::Reranker;
use engram_mcp::{SearchRequest, search};
use serde::Deserialize;
use sqlx::PgPool;
use uuid::Uuid;

/// nDCG@K parameter — 10 matches the M3 success criterion phrasing.
const NDCG_K: usize = 10;
/// Top-N requested per search. Matches NDCG_K so the ranking the
/// harness scores against is the same the consumer would see.
const SEARCH_LIMIT: usize = NDCG_K;

#[derive(Debug, Clone, Deserialize)]
pub struct BenchQuery {
    pub query: String,
    #[serde(default)]
    pub scope: Option<String>,
    pub relevant_ids: Vec<Uuid>,
    /// Optional explicit graded relevance (id → weight in [0, 1]). When
    /// absent, every id in `relevant_ids` is promoted to weight 1.0.
    #[serde(default)]
    pub graded_relevance: Option<HashMap<Uuid, f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BenchRerankCorpus {
    pub queries: Vec<BenchQuery>,
}

/// Per-query result row plus whether the harness saw zero relevant items
/// in both rankings (used to emit one warning per stale fixture entry).
#[derive(Debug, Clone)]
struct QueryRow {
    query: String,
    n_relevant: usize,
    ndcg_rrf: f32,
    ndcg_rerank: f32,
    mrr_rrf: f32,
    mrr_rerank: f32,
    no_match: bool,
}

pub async fn run_bench_rerank(
    pool: &PgPool,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    corpus_path: &Path,
) -> anyhow::Result<()> {
    let corpus = load_corpus(corpus_path)?;
    if corpus.queries.is_empty() {
        anyhow::bail!("bench corpus at {} has zero queries", corpus_path.display());
    }
    tracing::info!(
        path = %corpus_path.display(),
        n_queries = corpus.queries.len(),
        "bench: loaded corpus",
    );

    let mut rows: Vec<QueryRow> = Vec::with_capacity(corpus.queries.len());
    for q in &corpus.queries {
        let (ranking_rrf, ranking_rerank) = run_pair(pool, &*embedder, &*reranker, q).await?;
        let row = score_query(q, &ranking_rrf, &ranking_rerank);
        if row.no_match {
            tracing::warn!(
                query = %q.query,
                "bench: no relevant_ids found in either ranking — fixture may be stale",
            );
        }
        rows.push(row);
    }

    print_table(&rows);
    Ok(())
}

fn load_corpus(path: &Path) -> anyhow::Result<BenchRerankCorpus> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading bench corpus at {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("parsing bench corpus at {}", path.display()))
}

async fn run_pair(
    pool: &PgPool,
    embedder: &dyn Embedder,
    reranker: &dyn Reranker,
    q: &BenchQuery,
) -> anyhow::Result<(Vec<Uuid>, Vec<Uuid>)> {
    let scope =
        match &q.scope {
            Some(s) => Some(Scope::new(s.clone()).map_err(|e: ScopeError| {
                anyhow::anyhow!("invalid scope {s:?} in bench query: {e}")
            })?),
            None => None,
        };

    let mk = |rerank: bool| SearchRequest {
        query: q.query.clone(),
        scope: scope.clone(),
        limit: Some(SEARCH_LIMIT),
        recency_half_life_days: None,
        rerank: Some(rerank),
        candidate_pool: None,
        tag_filter: None,
    };
    let r_rrf = search::search_thoughts(pool, embedder, None, mk(false))
        .await
        .context("search_thoughts (rrf-only) failed")?;
    let r_rerank = search::search_thoughts(pool, embedder, Some(reranker), mk(true))
        .await
        .context("search_thoughts (reranked) failed")?;
    Ok((
        r_rrf
            .results
            .into_iter()
            .map(|h| h.thought_id.into_uuid())
            .collect(),
        r_rerank
            .results
            .into_iter()
            .map(|h| h.thought_id.into_uuid())
            .collect(),
    ))
}

/// Pure scorer: takes a query + its two rankings and produces the row.
/// Factored out so the parser + promotion logic can be unit-tested
/// without a live DB.
fn score_query(q: &BenchQuery, ranking_rrf: &[Uuid], ranking_rerank: &[Uuid]) -> QueryRow {
    let graded = resolve_graded(q);
    let relevant: HashSet<Uuid> = q.relevant_ids.iter().copied().collect();

    let ndcg_rrf = ndcg_at_k(ranking_rrf, &graded, NDCG_K);
    let ndcg_rerank = ndcg_at_k(ranking_rerank, &graded, NDCG_K);
    let mrr_rrf = reciprocal_rank(ranking_rrf, &relevant);
    let mrr_rerank = reciprocal_rank(ranking_rerank, &relevant);

    let no_match = !ranking_rrf.iter().any(|id| relevant.contains(id))
        && !ranking_rerank.iter().any(|id| relevant.contains(id));

    QueryRow {
        query: q.query.clone(),
        n_relevant: q.relevant_ids.len(),
        ndcg_rrf,
        ndcg_rerank,
        mrr_rrf,
        mrr_rerank,
        no_match,
    }
}

/// Promote bare `relevant_ids` (binary relevance) to unit-weight graded
/// when `graded_relevance` is absent. If both are present, the explicit
/// `graded_relevance` wins (the spec map is the source of truth).
fn resolve_graded(q: &BenchQuery) -> HashMap<Uuid, f32> {
    if let Some(g) = &q.graded_relevance {
        return g.clone();
    }
    q.relevant_ids.iter().copied().map(|id| (id, 1.0)).collect()
}

fn print_table(rows: &[QueryRow]) {
    // Column headers.
    println!(
        "| {q:<54} | rel | nDCG@10 (rrf) | nDCG@10 (rerank) | Δ      | MRR (rrf) | MRR (rerank) | Δ      |",
        q = "query",
    );
    println!(
        "|{q}|-----|---------------|------------------|--------|-----------|--------------|--------|",
        q = "-".repeat(56),
    );

    let mut sum_ndcg_rrf = 0.0;
    let mut sum_ndcg_rerank = 0.0;
    let mut sum_mrr_rrf = 0.0;
    let mut sum_mrr_rerank = 0.0;

    for row in rows {
        let q_trunc = truncate(&row.query, 54);
        let d_ndcg = row.ndcg_rerank - row.ndcg_rrf;
        let d_mrr = row.mrr_rerank - row.mrr_rrf;
        println!(
            "| {q:<54} | {rel:>3} | {nr:>13.3} | {nk:>16.3} | {dn:>+6.3} | {mr:>9.3} | {mk:>12.3} | {dm:>+6.3} |",
            q = q_trunc,
            rel = row.n_relevant,
            nr = row.ndcg_rrf,
            nk = row.ndcg_rerank,
            dn = d_ndcg,
            mr = row.mrr_rrf,
            mk = row.mrr_rerank,
            dm = d_mrr,
        );
        sum_ndcg_rrf += row.ndcg_rrf;
        sum_ndcg_rerank += row.ndcg_rerank;
        sum_mrr_rrf += row.mrr_rrf;
        sum_mrr_rerank += row.mrr_rerank;
    }

    let n = rows.len() as f32;
    let avg_ndcg_rrf = sum_ndcg_rrf / n;
    let avg_ndcg_rerank = sum_ndcg_rerank / n;
    let avg_mrr_rrf = sum_mrr_rrf / n;
    let avg_mrr_rerank = sum_mrr_rerank / n;

    println!(
        "|{q}|-----|---------------|------------------|--------|-----------|--------------|--------|",
        q = "-".repeat(56),
    );
    println!(
        "| {q:<54} | {rel:>3} | {nr:>13.3} | {nk:>16.3} | {dn:>+6.3} | {mr:>9.3} | {mk:>12.3} | {dm:>+6.3} |",
        q = "AVERAGE",
        rel = "-",
        nr = avg_ndcg_rrf,
        nk = avg_ndcg_rerank,
        dn = avg_ndcg_rerank - avg_ndcg_rrf,
        mr = avg_mrr_rrf,
        mk = avg_mrr_rerank,
        dm = avg_mrr_rerank - avg_mrr_rrf,
    );

    println!();
    println!(
        "rerank improved nDCG@10 by {:+.3} ({} queries); MRR by {:+.3}",
        avg_ndcg_rerank - avg_ndcg_rrf,
        rows.len(),
        avg_mrr_rerank - avg_mrr_rrf,
    );
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_parses_valid_fixture() {
        let json = r#"{
            "queries": [
                {
                    "query": "tooling for compiling codebases reproducibly",
                    "scope": null,
                    "relevant_ids": [
                        "00000000-0000-0000-0000-000000000001",
                        "00000000-0000-0000-0000-000000000002"
                    ],
                    "graded_relevance": {
                        "00000000-0000-0000-0000-000000000001": 1.0,
                        "00000000-0000-0000-0000-000000000002": 0.7
                    }
                },
                {
                    "query": "what does Engram use as a vector store",
                    "relevant_ids": ["00000000-0000-0000-0000-000000000003"]
                }
            ]
        }"#;
        let corpus: BenchRerankCorpus = serde_json::from_str(json).unwrap();
        assert_eq!(corpus.queries.len(), 2);
        assert_eq!(corpus.queries[0].relevant_ids.len(), 2);
        assert!(corpus.queries[0].graded_relevance.is_some());
        assert!(corpus.queries[1].graded_relevance.is_none());
        assert!(corpus.queries[1].scope.is_none());
    }

    #[test]
    fn corpus_promotes_binary_relevance_to_unit_weights_when_graded_absent() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let q = BenchQuery {
            query: "x".into(),
            scope: None,
            relevant_ids: vec![a, b],
            graded_relevance: None,
        };
        let graded = resolve_graded(&q);
        assert_eq!(graded.len(), 2);
        assert!((graded[&a] - 1.0).abs() < 1e-6);
        assert!((graded[&b] - 1.0).abs() < 1e-6);

        // Explicit graded_relevance wins when present.
        let q2 = BenchQuery {
            query: "x".into(),
            scope: None,
            relevant_ids: vec![a, b],
            graded_relevance: Some(HashMap::from([(a, 0.7)])),
        };
        let graded2 = resolve_graded(&q2);
        assert_eq!(graded2.len(), 1);
        assert!((graded2[&a] - 0.7).abs() < 1e-6);
    }

    /// Pins the bundled example fixture against the parser. Any schema
    /// drift will fail this test before downstream consumers hit it.
    #[test]
    fn bundled_example_fixture_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("bench-rerank.example.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let corpus: BenchRerankCorpus = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        assert!(!corpus.queries.is_empty());
        // Schema-coverage assertions: at least one with graded_relevance and
        // one without, at least one with a non-null scope.
        assert!(corpus.queries.iter().any(|q| q.graded_relevance.is_some()));
        assert!(corpus.queries.iter().any(|q| q.graded_relevance.is_none()));
        assert!(corpus.queries.iter().any(|q| q.scope.is_some()));
    }

    #[test]
    fn score_query_flags_no_match_when_neither_ranking_has_relevant() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let c = Uuid::from_u128(3);
        let q = BenchQuery {
            query: "stale fixture".into(),
            scope: None,
            relevant_ids: vec![a],
            graded_relevance: None,
        };
        // Neither ranking includes `a`.
        let row = score_query(&q, &[b, c], &[c, b]);
        assert!(row.no_match);
        assert_eq!(row.ndcg_rrf, 0.0);
        assert_eq!(row.ndcg_rerank, 0.0);
        assert_eq!(row.mrr_rrf, 0.0);
        assert_eq!(row.mrr_rerank, 0.0);

        // If at least one ranking has the relevant id, no_match flips false.
        let row2 = score_query(&q, &[a, b], &[b, c]);
        assert!(!row2.no_match);
        assert!(row2.mrr_rrf > 0.0);
        assert_eq!(row2.mrr_rerank, 0.0);
    }
}

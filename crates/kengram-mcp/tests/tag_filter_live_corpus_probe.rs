//! Ad-hoc probe replicating the 2026-05-17 dogfood report:
//! `search_thoughts(scope="kengram.m3.dogfood", tag_filter={"kind":"task"})`
//! reportedly returned 3 results all with `kind="observation"`. The
//! existing unit test `search_thoughts_filters_by_tag_kind` passes
//! against synthetic data via FakeEmbedder; this test runs the same
//! orchestrator path against the live corpus to settle whether the bug
//! reproduces.
//!
//! Reads `DATABASE_URL` at runtime. Gated behind `--ignored` so CI doesn't
//! depend on a live database. Run with:
//!   DATABASE_URL=postgres://kengram:kengram@localhost:5432/kengram \
//!     cargo test -p kengram-mcp --test dogfood_tag_filter_probe -- --ignored --nocapture

use kengram_embed::FakeEmbedder;
use kengram_mcp::{SearchRequest, search_thoughts};
use sqlx::PgPool;

#[tokio::test]
#[ignore]
async fn dogfood_kind_task_filter_returns_zero_on_observation_only_corpus() {
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&url).await.expect("connect");

    let embedder = FakeEmbedder::new();

    // Baseline: same query WITHOUT the filter should return results.
    let baseline = search_thoughts(
        &pool,
        &embedder,
        None,
        SearchRequest {
            query: "kengram".to_string(),
            scope: Some(kengram_core::Scope::new("kengram.m3.dogfood").unwrap()),
            scope_prefix: None,
            limit: Some(50),
            recency_half_life_days: None,
            rerank: Some(false),
            candidate_pool: None,
            tag_filter: None,
            chunk_serving_enabled: false,
            full_pipeline_enabled: false,
            tag_domain_routing_enabled: false,
            include_profile: false,
        },
    )
    .await
    .expect("baseline search_thoughts");
    println!("baseline (no filter): {} results", baseline.results.len());

    let resp = search_thoughts(
        &pool,
        &embedder,
        None,
        SearchRequest {
            query: "kengram".to_string(),
            scope: Some(kengram_core::Scope::new("kengram.m3.dogfood").unwrap()),
            scope_prefix: None,
            limit: Some(50),
            recency_half_life_days: None,
            rerank: Some(false),
            candidate_pool: None,
            tag_filter: Some(serde_json::json!({"kind": "task"})),
            chunk_serving_enabled: false,
            full_pipeline_enabled: false,
            tag_domain_routing_enabled: false,
            include_profile: false,
        },
    )
    .await
    .expect("search_thoughts");

    let kinds: Vec<_> = resp
        .results
        .iter()
        .map(|h| h.tags.kind.as_ref().map(|k| format!("{:?}", k)))
        .collect();
    println!(
        "kind=task filter: {} results, kinds={:?}",
        resp.results.len(),
        kinds
    );
    for hit in &resp.results {
        let short_id = hit.thought_id.to_string()[..8].to_string();
        println!(
            "  {short_id} kind={:?} content_preview={:?}",
            hit.tags.kind,
            &hit.content[..hit.content.len().min(60)]
        );
    }
}

#[tokio::test]
#[ignore]
async fn dogfood_bogus_entity_filter_returns_zero() {
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&url).await.expect("connect");

    let embedder = FakeEmbedder::new();
    let resp = search_thoughts(
        &pool,
        &embedder,
        None,
        SearchRequest {
            query: "kengram".to_string(),
            scope: Some(kengram_core::Scope::new("kengram.m3.dogfood").unwrap()),
            scope_prefix: None,
            limit: Some(50),
            recency_half_life_days: None,
            rerank: Some(false),
            candidate_pool: None,
            tag_filter: Some(
                serde_json::json!({"entities": ["DefinitelyNotARealEntityFooBarBaz"]}),
            ),
            chunk_serving_enabled: false,
            full_pipeline_enabled: false,
            tag_domain_routing_enabled: false,
            include_profile: false,
        },
    )
    .await
    .expect("search_thoughts");

    println!("bogus-entity filter: {} results", resp.results.len());
    for hit in &resp.results {
        let short_id = hit.thought_id.to_string()[..8].to_string();
        println!("  {short_id} entities={:?}", hit.tags.entities);
    }
}

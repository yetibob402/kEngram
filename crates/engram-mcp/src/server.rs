//! The rmcp `ServerHandler` wiring. `EngramServer` is the per-connection
//! service factory; it holds an `Arc<dyn Embedder>` and a `PgPool` (both
//! cheap to clone). The actual orchestration lives in [`crate::capture`]
//! and [`crate::search`].

use engram_core::{Embedder, Metadata, Scope, Source, ThoughtId};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use sqlx::PgPool;
use std::str::FromStr;
use std::sync::Arc;

use crate::capture::{self, CaptureError, CaptureRequest, MAX_CONTENT_LEN};
use crate::search::{
    self, GetThoughtResponse, ReadError, RecentRequest, RecentResponse, SearchRequest,
    SearchResponse,
};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CaptureArgs {
    #[schemars(description = "The thought text. Required, non-empty, max 1 MiB.")]
    pub content: String,

    #[schemars(description = "Provenance label. Required. Examples: 'manual', 'agent:claude-code'.")]
    pub source: String,

    #[schemars(description = "Scope label. Optional; defaults to 'global'. Convention is dotted ('work.tcgplayer').")]
    pub scope: Option<String>,

    #[schemars(description = "Optional free-form metadata object. Recommended keys: client_name, session_id, tool_name, agent_role.")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchThoughtsArgs {
    #[schemars(description = "Search query. Required, non-empty.")]
    pub query: String,

    #[schemars(description = "Scope filter. Optional; when omitted, searches across all scopes.")]
    pub scope: Option<String>,

    #[schemars(description = "Max results. Optional; defaults to 10, max 100.")]
    pub limit: Option<usize>,

    #[schemars(description = "Recency boost half-life in days. Optional; defaults to 30. Set to 0 to disable recency boost.")]
    pub recency_half_life_days: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecentThoughtsArgs {
    #[schemars(description = "Scope filter. Optional; when omitted, returns across all scopes.")]
    pub scope: Option<String>,

    #[schemars(description = "Max results. Optional; defaults to 10, max 100.")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetThoughtArgs {
    #[schemars(description = "Thought ID (UUID string).")]
    pub thought_id: String,
}

#[derive(Clone)]
pub struct EngramServer {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
    tool_router: ToolRouter<Self>,
}

impl EngramServer {
    pub fn new(pool: PgPool, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            pool,
            embedder,
            tool_router: Self::tool_router(),
        }
    }
}

impl std::fmt::Debug for EngramServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramServer")
            .field("model_id", &self.embedder.model().id)
            .finish()
    }
}

#[tool_router]
impl EngramServer {
    #[tool(description = "Capture a thought into engram's persistent memory. Returns the thought_id and embedding_status='pending'. The thought is durable and findable by trigram (lexical) search immediately; vector search picks it up on the next worker tick (default 5 seconds). To make vector-search-readiness fully synchronous, run `engram worker` alongside `engram serve`.")]
    async fn capture(
        &self,
        Parameters(args): Parameters<CaptureArgs>,
    ) -> Result<String, String> {
        let source = Source::new(args.source)
            .map_err(|e| format!("invalid source: {e}"))?;

        let scope = match args.scope {
            Some(s) => Some(Scope::new(s).map_err(|e| format!("invalid scope: {e}"))?),
            None => None,
        };

        let metadata = args.metadata.map(Metadata::from);

        let request = CaptureRequest {
            content: args.content,
            source,
            scope,
            metadata,
        };

        let resp = capture::capture(&self.pool, &self.embedder.model().id, request)
            .await
            .map_err(map_capture_error)?;

        let body = serde_json::json!({
            "thought_id": resp.thought_id.to_string(),
            "embedding_status": resp.embedding_status,
        });

        serde_json::to_string(&body)
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(description = "Hybrid search across captured thoughts. Combines vector kNN (over the active embedding model) with trigram lexical similarity via reciprocal rank fusion, then applies a recency boost. Returns the top-N matching thoughts with score. If the embedder is unreachable, results still come back from the trigram leg only and `vector_search_available` is false.")]
    async fn search_thoughts(
        &self,
        Parameters(args): Parameters<SearchThoughtsArgs>,
    ) -> Result<String, String> {
        let scope = match args.scope {
            Some(s) => Some(Scope::new(s).map_err(|e| format!("invalid scope: {e}"))?),
            None => None,
        };

        let request = SearchRequest {
            query: args.query,
            scope,
            limit: args.limit,
            recency_half_life_days: args.recency_half_life_days,
        };

        let resp = search::search_thoughts(&self.pool, self.embedder.as_ref(), request)
            .await
            .map_err(map_read_error)?;

        serde_json::to_string(&search_response_json(&resp))
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(description = "Recent thoughts in (optional) scope, ordered newest first. No retrieval scoring — just chronological browsing.")]
    async fn recent_thoughts(
        &self,
        Parameters(args): Parameters<RecentThoughtsArgs>,
    ) -> Result<String, String> {
        let scope = match args.scope {
            Some(s) => Some(Scope::new(s).map_err(|e| format!("invalid scope: {e}"))?),
            None => None,
        };

        let request = RecentRequest {
            scope,
            limit: args.limit,
        };

        let resp = search::recent_thoughts(&self.pool, request)
            .await
            .map_err(map_read_error)?;

        serde_json::to_string(&recent_response_json(&resp))
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(description = "Fetch a single thought by ID along with its provenance: whether it's been embedded ('indexed' or 'pending'), when it was embedded, and (M2+) linked extracted facts.")]
    async fn get_thought(
        &self,
        Parameters(args): Parameters<GetThoughtArgs>,
    ) -> Result<String, String> {
        let id = ThoughtId::from_str(&args.thought_id)
            .map_err(|e| format!("invalid thought_id: {e}"))?;

        let resp = search::get_thought(&self.pool, self.embedder.model(), id)
            .await
            .map_err(map_read_error)?;

        serde_json::to_string(&get_thought_response_json(&resp))
            .map_err(|e| format!("response serialization error: {e}"))
    }
}

fn map_capture_error(err: CaptureError) -> String {
    match err {
        CaptureError::EmptyContent => "content must be non-empty".to_string(),
        CaptureError::ContentTooLong { got, max } => {
            format!("content too long: {got} bytes (max {max} = {MAX_CONTENT_LEN})")
        }
        CaptureError::Storage(e) => {
            tracing::error!(error = %e, "capture storage error");
            "internal database error during capture".to_string()
        }
    }
}

fn map_read_error(err: ReadError) -> String {
    match err {
        ReadError::EmptyQuery => "query must be non-empty".to_string(),
        ReadError::LimitOutOfBounds { got, max } => {
            format!("limit out of bounds: {got} (must be 1..={max})")
        }
        ReadError::NotFound => "thought not found".to_string(),
        ReadError::Storage(e) => {
            tracing::error!(error = %e, "read storage error");
            "internal database error".to_string()
        }
    }
}

fn search_response_json(resp: &SearchResponse) -> serde_json::Value {
    let results: Vec<serde_json::Value> = resp
        .results
        .iter()
        .map(|h| {
            serde_json::json!({
                "thought_id": h.thought_id.to_string(),
                "content": h.content,
                "scope": h.scope.as_str(),
                "source": h.source.as_str(),
                "created_at": h.created_at.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
                "metadata": h.metadata.as_value(),
                "score": h.score,
            })
        })
        .collect();
    serde_json::json!({
        "results": results,
        "vector_search_available": resp.vector_search_available,
    })
}

fn recent_response_json(resp: &RecentResponse) -> serde_json::Value {
    let results: Vec<serde_json::Value> = resp
        .results
        .iter()
        .map(|t| {
            serde_json::json!({
                "thought_id": t.id.to_string(),
                "content": t.content,
                "scope": t.scope.as_str(),
                "source": t.source.as_str(),
                "created_at": t.created_at.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
                "metadata": t.metadata.as_value(),
            })
        })
        .collect();
    serde_json::json!({ "results": results })
}

fn get_thought_response_json(resp: &GetThoughtResponse) -> serde_json::Value {
    serde_json::json!({
        "thought": {
            "thought_id": resp.thought.id.to_string(),
            "content": resp.thought.content,
            "scope": resp.thought.scope.as_str(),
            "source": resp.thought.source.as_str(),
            "created_at": resp.thought.created_at.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
            "metadata": resp.thought.metadata.as_value(),
        },
        "provenance": {
            "embedding_status": resp.embedding_status,
            "embedded_at": resp.embedded_at.and_then(|t| t.format(&time::format_description::well_known::Rfc3339).ok()),
            "linked_facts": [],
        },
    })
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EngramServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Engram — self-hosted MCP-native memory service. \
                 Use `capture` to record a thought, `search_thoughts` for hybrid retrieval, \
                 `recent_thoughts` to browse by recency, and `get_thought` for full provenance \
                 of a single thought.",
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_embed::FakeEmbedder;

    fn server(pool: PgPool) -> EngramServer {
        EngramServer::new(pool, Arc::new(FakeEmbedder::new()))
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn capture_tool_returns_thought_id_and_pending_status(pool: PgPool) {
        let s = server(pool);
        let raw = s
            .capture(Parameters(CaptureArgs {
                content: "hello there".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(json["thought_id"].is_string());
        // M2 Phase B flip: pending is the normal return; worker drains on its tick.
        assert_eq!(json["embedding_status"], "pending");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn capture_tool_validation_error_reports_message(pool: PgPool) {
        let s = server(pool);
        let err = s
            .capture(Parameters(CaptureArgs {
                content: String::new(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap_err();
        assert!(err.contains("content"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_tool_returns_results_and_flag(pool: PgPool) {
        let s = server(pool);
        s.capture(Parameters(CaptureArgs {
            content: "tcgplayer integration notes".into(),
            source: "test".into(),
            scope: None,
            metadata: None,
        }))
        .await
        .unwrap();

        let raw = s
            .search_thoughts(Parameters(SearchThoughtsArgs {
                query: "tcgplayer".into(),
                scope: None,
                limit: None,
                recency_half_life_days: None,
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["vector_search_available"], true);
        let results = json["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert!(results[0]["thought_id"].is_string());
        assert!(results[0]["content"].as_str().unwrap().contains("tcgplayer"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn recent_thoughts_tool_returns_results(pool: PgPool) {
        let s = server(pool);
        s.capture(Parameters(CaptureArgs {
            content: "one".into(),
            source: "test".into(),
            scope: None,
            metadata: None,
        }))
        .await
        .unwrap();
        s.capture(Parameters(CaptureArgs {
            content: "two".into(),
            source: "test".into(),
            scope: None,
            metadata: None,
        }))
        .await
        .unwrap();

        let raw = s
            .recent_thoughts(Parameters(RecentThoughtsArgs {
                scope: None,
                limit: None,
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_tool_returns_thought_and_pending_provenance(pool: PgPool) {
        let s = server(pool);
        let cap_raw = s
            .capture(Parameters(CaptureArgs {
                content: "hello".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let cap_json: serde_json::Value = serde_json::from_str(&cap_raw).unwrap();
        let thought_id = cap_json["thought_id"].as_str().unwrap().to_string();

        let raw = s
            .get_thought(Parameters(GetThoughtArgs { thought_id }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(json["thought"].is_object());
        assert_eq!(json["thought"]["content"], "hello");
        // M2 Phase B: capture leaves the embedding pending; no worker has run.
        assert_eq!(json["provenance"]["embedding_status"], "pending");
        assert!(json["provenance"]["embedded_at"].is_null());
        assert_eq!(json["provenance"]["linked_facts"], serde_json::json!([]));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_tool_invalid_uuid_errors(pool: PgPool) {
        let s = server(pool);
        let err = s
            .get_thought(Parameters(GetThoughtArgs {
                thought_id: "not-a-uuid".into(),
            }))
            .await
            .unwrap_err();
        assert!(err.contains("invalid thought_id"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn capture_then_drain_makes_thought_indexed_via_get_thought(pool: PgPool) {
        // M2 Phase B end-to-end (success criterion #4 in m2-facts-pipeline.md):
        // capture → worker drains → get_thought reports embedding_status=indexed.
        let s = server(pool.clone());

        let cap_raw = s
            .capture(Parameters(CaptureArgs {
                content: "drain me end-to-end".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let cap_json: serde_json::Value = serde_json::from_str(&cap_raw).unwrap();
        let thought_id = cap_json["thought_id"].as_str().unwrap().to_string();
        assert_eq!(cap_json["embedding_status"], "pending");

        // Worker tick: pull the queue, embed, mark.
        let report = crate::drain::drain_pending_embeddings(&pool, s.embedder.as_ref(), 16)
            .await
            .unwrap();
        assert_eq!(report.embedded, 1);
        assert_eq!(report.failed, 0);

        // After the drain, get_thought reports indexed.
        let raw = s
            .get_thought(Parameters(GetThoughtArgs { thought_id }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["provenance"]["embedding_status"], "indexed");
        assert!(json["provenance"]["embedded_at"].is_string());
    }
}

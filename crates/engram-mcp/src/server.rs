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
use crate::correct::{self, CorrectError, CorrectFactRequest, FactReplacement};
use crate::retract::{self, RetractError, RetractThoughtRequest};
use crate::search::{
    self, GetThoughtResponse, ReadError, RecentRequest, RecentResponse, SearchFactsRequest,
    SearchFactsResponse, SearchRequest, SearchResponse,
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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchFactsArgs {
    #[schemars(description = "Search query. Required, non-empty. Matched against fact.statement via trigram similarity.")]
    pub query: String,

    #[schemars(description = "Scope filter. Optional; when omitted, searches across all scopes.")]
    pub scope: Option<String>,

    #[schemars(description = "Max results. Optional; defaults to 10, max 100.")]
    pub limit: Option<usize>,

    #[schemars(description = "Recency boost half-life in days, keyed on the source thought's created_at. Optional; defaults to 30. Set to 0 to disable.")]
    pub recency_half_life_days: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CorrectFactArgs {
    #[schemars(description = "Fact ID (UUID string) to correct or retract.")]
    pub fact_id: String,

    #[schemars(description = "Optional replacement. If present, a new fact is inserted with manual-author provenance (extractor_model='manual', extractor_version=0, confidence=1.0) and the old fact is superseded with `superseded_by` pointing at the new row. If omitted, the old fact is superseded with no replacement (delete-by-supersede).")]
    pub replacement: Option<CorrectFactReplacementArgs>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RetractThoughtArgs {
    #[schemars(description = "Thought ID (UUID string) to retract.")]
    pub thought_id: String,

    #[schemars(description = "Optional free-text reason for the retraction (e.g. 'wrong claim — see thought <new id> for correction'). Stored on thoughts.retracted_reason for audit; max 1000 chars.")]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CorrectFactReplacementArgs {
    #[schemars(description = "Natural-language statement for the corrected fact. Required, non-empty.")]
    pub statement: String,
    #[schemars(description = "Optional subject of the (S, P, O) triple.")]
    pub subject: Option<String>,
    #[schemars(description = "Optional predicate of the (S, P, O) triple.")]
    pub predicate: Option<String>,
    #[schemars(description = "Optional object of the (S, P, O) triple.")]
    pub object: Option<String>,
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

    #[tool(description = "Fetch a single thought by ID along with its provenance: whether it's been embedded ('indexed' or 'pending'), when it was embedded, and the active (non-superseded) facts derived from it.")]
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

    #[tool(description = "Search across extracted facts via trigram similarity over fact.statement, filtered to active (non-superseded) facts. Each result includes the fact's S/P/O triple, confidence, score, and the source thought's content/scope/created_at so the agent doesn't need a follow-up get_thought call.")]
    async fn search_facts(
        &self,
        Parameters(args): Parameters<SearchFactsArgs>,
    ) -> Result<String, String> {
        let scope = match args.scope {
            Some(s) => Some(Scope::new(s).map_err(|e| format!("invalid scope: {e}"))?),
            None => None,
        };

        let request = SearchFactsRequest {
            query: args.query,
            scope,
            limit: args.limit,
            recency_half_life_days: args.recency_half_life_days,
        };

        let resp = search::search_facts(&self.pool, self.embedder.as_ref(), request)
            .await
            .map_err(map_read_error)?;

        serde_json::to_string(&search_facts_response_json(&resp))
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(description = "Retract a thought as untrusted (e.g. the operator captured a wrong claim). Atomically marks `thoughts.retracted_at = NOW()` and auto-supersedes every active fact derived from the thought, so a subsequent reflector run won't re-extract from the still-untrusted source. The thought row itself stays in the DB (`get_thought` still finds it; the response carries `retracted_at` and `retracted_reason`). Use this rather than retracting facts one at a time, which is fragile if the operator misses any.")]
    async fn retract_thought(
        &self,
        Parameters(args): Parameters<RetractThoughtArgs>,
    ) -> Result<String, String> {
        let thought_id = ThoughtId::from_str(&args.thought_id)
            .map_err(|e| format!("invalid thought_id: {e}"))?;

        let resp = retract::retract_thought(
            &self.pool,
            RetractThoughtRequest { thought_id, reason: args.reason },
        )
        .await
        .map_err(map_retract_error)?;

        let body = serde_json::json!({
            "retracted": resp.retracted,
            "facts_superseded": resp.facts_superseded,
        });
        serde_json::to_string(&body)
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(description = "Correct or retract a fact. With a replacement, inserts a new fact (manual provenance: extractor_model='manual', version=0, confidence=1.0) and supersedes the old one; the audit trail (superseded_by, superseded_at) is preserved. Without a replacement, the fact is superseded with no successor — the row stays in the DB but search_facts and get_thought will no longer surface it.")]
    async fn correct_fact(
        &self,
        Parameters(args): Parameters<CorrectFactArgs>,
    ) -> Result<String, String> {
        let fact_id = uuid::Uuid::from_str(&args.fact_id)
            .map_err(|e| format!("invalid fact_id: {e}"))?;
        let replacement = args.replacement.map(|r| FactReplacement {
            statement: r.statement,
            subject: r.subject,
            predicate: r.predicate,
            object: r.object,
        });

        let resp = correct::correct_fact(&self.pool, CorrectFactRequest { fact_id, replacement })
            .await
            .map_err(map_correct_error)?;

        let body = serde_json::json!({
            "superseded": resp.superseded,
            "new_fact_id": resp.new_fact_id.map(|u| u.to_string()),
        });
        serde_json::to_string(&body)
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

fn map_correct_error(err: CorrectError) -> String {
    match err {
        CorrectError::AlreadySupersededOrMissing(id) => {
            format!("fact not found or already superseded: {id}")
        }
        CorrectError::EmptyReplacementStatement => {
            "replacement statement must not be empty".to_string()
        }
        CorrectError::Storage(e) => {
            tracing::error!(error = %e, "correct_fact storage error");
            "internal database error".to_string()
        }
    }
}

fn map_retract_error(err: RetractError) -> String {
    match err {
        RetractError::NotFoundOrAlreadyRetracted(id) => {
            format!("thought not found or already retracted: {id}")
        }
        RetractError::Storage(e) => {
            tracing::error!(error = %e, "retract_thought storage error");
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
    let linked_facts: Vec<serde_json::Value> = resp
        .linked_facts
        .iter()
        .map(|f| {
            serde_json::json!({
                "fact_id": f.id.to_string(),
                "statement": f.statement,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "extractor_model": f.extractor_model,
                "extractor_version": f.extractor_version,
                "created_at": f.created_at.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
            })
        })
        .collect();
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
            "linked_facts": linked_facts,
            "retracted_at": resp.retracted_at.and_then(|t| t.format(&time::format_description::well_known::Rfc3339).ok()),
            "retracted_reason": resp.retracted_reason,
        },
    })
}

fn search_facts_response_json(resp: &SearchFactsResponse) -> serde_json::Value {
    let results: Vec<serde_json::Value> = resp
        .results
        .iter()
        .map(|h| {
            serde_json::json!({
                "fact_id": h.fact_id.to_string(),
                "statement": h.statement,
                "subject": h.subject,
                "predicate": h.predicate,
                "object": h.object,
                "confidence": h.confidence,
                "source_thought_id": h.source_thought_id.to_string(),
                "source_thought_content": h.source_thought_content,
                "source_thought_scope": h.source_thought_scope.as_str(),
                "source_thought_created_at": h.source_thought_created_at.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
                "score": h.score,
            })
        })
        .collect();
    serde_json::json!({
        "results": results,
        "vector_search_available": resp.vector_search_available,
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

    // -- M2 Phase D tool tests --

    async fn seed_fact_for_tool_test(
        pool: &PgPool,
        thought_content: &str,
        statement: &str,
    ) -> (String, String) {
        // Capture a thought, insert a fact against it, return (thought_id, fact_id).
        let scope = Scope::global();
        let source = Source::new("test").unwrap();
        let metadata = engram_core::Metadata::empty();
        let inserted = engram_storage::insert_thought(
            pool,
            engram_storage::NewThought {
                scope: &scope,
                content: thought_content,
                source: &source,
                metadata: &metadata,
            },
        )
        .await
        .unwrap();
        let run_id = engram_storage::start_run(pool, "fake/extractor", 1, None)
            .await
            .unwrap();
        let fact_id = engram_storage::insert_fact(
            pool,
            engram_storage::NewFact {
                scope: &scope,
                statement,
                subject: None,
                predicate: None,
                object: None,
                source_thought_id: inserted.id,
                extractor_model: "fake/extractor",
                extractor_version: 1,
                source_run_id: Some(run_id),
                confidence: 0.9,
            },
        )
        .await
        .unwrap();
        (inserted.id.to_string(), fact_id.to_string())
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_facts_tool_returns_results_with_source_thought_content(pool: PgPool) {
        seed_fact_for_tool_test(&pool, "Engram uses pgvector", "pgvector is the vector store").await;

        let s = server(pool);
        let raw = s
            .search_facts(Parameters(SearchFactsArgs {
                query: "pgvector".to_string(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0]["fact_id"].is_string());
        assert_eq!(results[0]["statement"], "pgvector is the vector store");
        assert_eq!(results[0]["source_thought_content"], "Engram uses pgvector");
        assert_eq!(results[0]["source_thought_scope"], "global");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_tool_supersedes_with_replacement(pool: PgPool) {
        let (_thought_id, fact_id) =
            seed_fact_for_tool_test(&pool, "anchor", "old wording").await;
        let s = server(pool.clone());

        let raw = s
            .correct_fact(Parameters(CorrectFactArgs {
                fact_id: fact_id.clone(),
                replacement: Some(CorrectFactReplacementArgs {
                    statement: "new wording".to_string(),
                    subject: Some("S".to_string()),
                    predicate: Some("P".to_string()),
                    object: Some("O".to_string()),
                }),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["superseded"], true);
        assert!(json["new_fact_id"].is_string());

        // Old fact is superseded; new fact exists with manual sentinel.
        let old_uuid = uuid::Uuid::from_str(&fact_id).unwrap();
        let old_row = sqlx::query!(
            r#"SELECT superseded_at FROM facts WHERE id = $1"#,
            old_uuid,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(old_row.superseded_at.is_some());

        let new_uuid =
            uuid::Uuid::from_str(json["new_fact_id"].as_str().unwrap()).unwrap();
        let new_row = sqlx::query!(
            r#"SELECT extractor_model, extractor_version FROM facts WHERE id = $1"#,
            new_uuid,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(new_row.extractor_model, "manual");
        assert_eq!(new_row.extractor_version, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_tool_errors_on_unknown_id(pool: PgPool) {
        let s = server(pool);
        let err = s
            .correct_fact(Parameters(CorrectFactArgs {
                fact_id: uuid::Uuid::new_v4().to_string(),
                replacement: None,
            }))
            .await
            .unwrap_err();
        assert!(err.contains("not found or already superseded"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn correct_fact_tool_invalid_uuid_errors(pool: PgPool) {
        let s = server(pool);
        let err = s
            .correct_fact(Parameters(CorrectFactArgs {
                fact_id: "not-a-uuid".to_string(),
                replacement: None,
            }))
            .await
            .unwrap_err();
        assert!(err.contains("invalid fact_id"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_tool_includes_linked_facts(pool: PgPool) {
        let (thought_id, _fact_id) =
            seed_fact_for_tool_test(&pool, "with linked fact", "the fact").await;
        let s = server(pool);

        let raw = s
            .get_thought(Parameters(GetThoughtArgs { thought_id }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let facts = json["provenance"]["linked_facts"].as_array().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0]["statement"], "the fact");
        assert!(facts[0]["fact_id"].is_string());
        // Fresh thought has no retraction state.
        assert!(json["provenance"]["retracted_at"].is_null());
        assert!(json["provenance"]["retracted_reason"].is_null());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_tool_marks_thought_and_supersedes_facts(pool: PgPool) {
        let (thought_id, fact_id) =
            seed_fact_for_tool_test(&pool, "wrong claim", "derived false fact").await;
        let s = server(pool.clone());

        let raw = s
            .retract_thought(Parameters(RetractThoughtArgs {
                thought_id: thought_id.clone(),
                reason: Some("test reason".into()),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["retracted"], true);
        assert_eq!(json["facts_superseded"], 1);

        // get_thought now surfaces retraction state + empty linked_facts.
        let raw = s
            .get_thought(Parameters(GetThoughtArgs { thought_id: thought_id.clone() }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(json["provenance"]["retracted_at"].is_string());
        assert_eq!(json["provenance"]["retracted_reason"], "test reason");
        let linked = json["provenance"]["linked_facts"].as_array().unwrap();
        assert!(linked.is_empty());

        // The derived fact is now superseded in the DB.
        let fact_uuid = uuid::Uuid::from_str(&fact_id).unwrap();
        let row = sqlx::query!(
            r#"SELECT superseded_at FROM facts WHERE id = $1"#,
            fact_uuid,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.superseded_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_tool_errors_on_already_retracted(pool: PgPool) {
        let (thought_id, _) = seed_fact_for_tool_test(&pool, "wrong claim", "f").await;
        let s = server(pool);

        s.retract_thought(Parameters(RetractThoughtArgs {
            thought_id: thought_id.clone(),
            reason: None,
        }))
        .await
        .unwrap();

        let err = s
            .retract_thought(Parameters(RetractThoughtArgs { thought_id, reason: None }))
            .await
            .unwrap_err();
        assert!(err.contains("not found or already retracted"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_tool_errors_on_invalid_uuid(pool: PgPool) {
        let s = server(pool);
        let err = s
            .retract_thought(Parameters(RetractThoughtArgs {
                thought_id: "not-a-uuid".into(),
                reason: None,
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

//! The rmcp `ServerHandler` wiring. `EngramServer` is the per-connection
//! service factory; it holds an `Arc<dyn Embedder>`, an optional
//! `Arc<dyn Reranker>`, the configured embedder/tagger model ids, and a
//! `PgPool` (cheap to clone). The actual orchestration lives in
//! [`crate::capture`] and [`crate::search`].
//!
//! M4: no more facts. `search_facts`, `correct_fact`, `linked_facts`,
//! `reflect`, `reflect_rerun` are gone. The server's only handles are
//! `capture`, `search_thoughts`, `recent_thoughts`, `get_thought`,
//! `retract_thought`. Tag drainage lives in the worker, not here.

use engram_core::{Embedder, LinkDirection, Metadata, RelationKind, Scope, Source, ThoughtId};
use engram_embed::Reranker;
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
use crate::link::{self, LinkError, LinkThoughtsRequest, MAX_LINK_NOTE_LEN};
use crate::relate::{self, GetRelatedThoughtsRequest, RelateError};
use crate::retract::{self, RetractError, RetractThoughtRequest};
use crate::search::{
    self, GetThoughtResponse, ReadError, RecentRequest, RecentResponse, SearchRequest,
    SearchResponse,
};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CaptureArgs {
    #[schemars(description = "The thought text. Required, non-empty, max 1 MiB.")]
    pub content: String,

    #[schemars(
        description = "Provenance label. Required. Examples: 'manual', 'agent:claude-code'."
    )]
    pub source: String,

    #[schemars(
        description = "Scope label. Optional; defaults to 'global'. Convention is dotted ('work.tcgplayer')."
    )]
    pub scope: Option<String>,

    #[schemars(
        description = "Optional free-form metadata object. Recommended keys: client_name, session_id, tool_name, agent_role."
    )]
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

    #[schemars(
        description = "Recency boost half-life in days. Optional; defaults to 30. Set to 0 to disable recency boost."
    )]
    pub recency_half_life_days: Option<f32>,

    #[schemars(
        description = "Apply the cross-encoder rerank stage. Defaults to true when a reranker is configured. Set false for A/B comparison against the RRF + recency pipeline."
    )]
    pub rerank: Option<bool>,

    #[schemars(
        description = "Number of post-RRF candidates fed into the reranker. Ignored when rerank is off. Defaults to 32."
    )]
    pub candidate_pool: Option<usize>,

    #[schemars(
        description = "Optional JSONB-containment filter applied to each thought's `tags` field. Tags are LLM-extracted metadata with shape: { people: string[], entities: string[] (named proper-noun-style identifiers — projects, products, libraries, tools, e.g. \"engram\", \"pgvector\"), action_items: string[], topics: string[] (1-3 short lowercase subject categories — e.g. \"rust\", \"memory-systems\"), dates_mentioned: string[], kind: 'observation' | 'task' | 'idea' | 'reference' | 'person_note' | 'session' | null }. Distinguish `entities` (specific named things mentioned by name) from `topics` (broader subject categories the thought falls under). The `kind` enum is closed at the values listed; the array fields are open-vocabulary strings. Examples: {\"kind\": \"task\"} returns only thoughts the tagger classified as tasks; {\"people\": [\"Sarah\"]} returns thoughts whose people-tag contains Sarah; {\"entities\": [\"engram\"]} returns thoughts mentioning engram by name; {\"topics\": [\"rust\"], \"kind\": \"idea\"} combines both (top-level keys AND together; array values are subset-match). Empty object {} is a no-op. Filters compose with `scope`."
    )]
    pub tag_filter: Option<serde_json::Value>,
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
pub struct RetractThoughtArgs {
    #[schemars(description = "Thought ID (UUID string) to retract.")]
    pub thought_id: String,

    #[schemars(
        description = "Optional free-text reason for the retraction (e.g. 'wrong claim — see thought <new id> for correction'). Stored on thoughts.retracted_reason for audit; max 1000 chars."
    )]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LinkThoughtsArgs {
    #[schemars(description = "Source thought ID (the 'from' side of the relation). UUID string.")]
    pub from_thought_id: String,

    #[schemars(
        description = "Relation type. Must be one of the closed vocabulary: 'replaces' (this thought replaces an earlier one — most recent supersedes), 'requires' (this thought depends on another), 'references' (this thought points at another for context, like a citation), 'supports' (this thought confirms a claim made in another — experimental evidence, corroborating data, logical support), 'belongs_to' (this thought is a member/sub-element of another, e.g. a finding under a parent thread), 'decided_by' (this thought is a decision attributed to another, e.g. a person-note or session-anchor), 'refines' (this thought is a refinement/iteration of an earlier one — both stand, but the newer thought represents updated thinking).\n\nCommon mistakes to avoid:\n- DO NOT use `refines` for citation or evidence. `refines` means the newer thought represents updated thinking on the SAME proposition — not 'the newer thought cites the older one for context.' Use `references` (or `supports` if the newer thought confirms a claim) instead.\n- DO NOT use `belongs_to` when the target is a peer or sibling — the v1 vocabulary lacks a sibling/peer-grouping relation. Model the parent (e.g., the experiment, the session) explicitly as its own thought and use `belongs_to` against that. If the parent isn't naturally a thought, capture one for it (one-line description is fine).\n- DO NOT use `decided_by` unless there is a clear decision-maker attribution. 'The team converged on X' is `decided_by` Team; 'the research suggests X' is `supports`, not `decided_by`.\n- DO NOT use `replaces` for refinement. `replaces` means the older thought is no longer the current thinking; use `refines` when both stand and the newer one just represents updated thinking.\n- DO NOT use `references` when the newer thought confirms a claim made in the older one — use `supports`. (`references` is for prose-level mention; `supports` is for evidential / corroborative relationship.)"
    )]
    pub relation: String,

    #[schemars(description = "Target thought ID (the 'to' side of the relation). UUID string.")]
    pub to_thought_id: String,

    #[schemars(
        description = "Optional free-text annotation explaining the link (e.g. 'refines after probe 2B dogfood'). Stored on thought_links.note; max 1000 chars."
    )]
    pub note: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct UnlinkThoughtsArgs {
    #[schemars(description = "Source thought ID (the 'from' side). UUID string.")]
    pub from_thought_id: String,

    #[schemars(
        description = "Relation type (same closed vocabulary as link_thoughts: replaces, requires, references, supports, belongs_to, decided_by, refines)."
    )]
    pub relation: String,

    #[schemars(description = "Target thought ID (the 'to' side). UUID string.")]
    pub to_thought_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetRelatedThoughtsArgs {
    #[schemars(description = "Thought ID (UUID string) to traverse from.")]
    pub thought_id: String,

    #[schemars(
        description = "Optional filter to a subset of relation types (e.g. ['refines','replaces']). Each item must be in the closed vocabulary: replaces, requires, references, supports, belongs_to, decided_by, refines. Omit to return edges of every type."
    )]
    pub relations: Option<Vec<String>>,

    #[schemars(
        description = "Traversal direction: 'outbound' (edges where the queried thought is the source — 'this refines X'), 'inbound' (edges where it is the target — 'X refines this'), or 'both' (default). Response always groups results into separate `outbound` and `inbound` arrays regardless."
    )]
    pub direction: Option<String>,
}

#[derive(Clone)]
pub struct EngramServer {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
    /// `None` when no `[reranker]` config is provided; the search pipeline
    /// silently falls through to the RRF + recency pipeline.
    reranker: Option<Arc<dyn Reranker>>,
    /// `None` when `[tagger]` is unconfigured — silent-disables tag-job
    /// enqueue at capture time. `Some(model_id)` enqueues a `pending_tags`
    /// row per fresh capture; the worker's tag drainer picks it up.
    tagger_model_id: Option<String>,
    tool_router: ToolRouter<Self>,
}

impl EngramServer {
    pub fn new(
        pool: PgPool,
        embedder: Arc<dyn Embedder>,
        reranker: Option<Arc<dyn Reranker>>,
        tagger_model_id: Option<String>,
    ) -> Self {
        Self {
            pool,
            embedder,
            reranker,
            tagger_model_id,
            tool_router: Self::tool_router(),
        }
    }

    /// Convenience for tests that don't exercise the rerank stage or tagger.
    #[cfg(test)]
    pub fn new_without_reranker(pool: PgPool, embedder: Arc<dyn Embedder>) -> Self {
        Self::new(pool, embedder, None, None)
    }
}

impl std::fmt::Debug for EngramServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramServer")
            .field("model_id", &self.embedder.model().id)
            .field("tagger_model_id", &self.tagger_model_id)
            .finish()
    }
}

#[tool_router]
impl EngramServer {
    #[tool(
        description = "Capture a thought into engram's persistent memory. Returns the thought_id and embedding_status='pending'. The thought is durable and findable by trigram (lexical) search immediately; vector search picks it up on the next worker tick (default 5 seconds). Identical content (SHA-256 of the bytes) is deduplicated — the response will include `is_duplicate: true` and the pre-existing thought_id when the fingerprint collides."
    )]
    async fn capture(&self, Parameters(args): Parameters<CaptureArgs>) -> Result<String, String> {
        let source = Source::new(args.source).map_err(|e| format!("invalid source: {e}"))?;

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

        let resp = capture::capture(
            &self.pool,
            &self.embedder.model().id,
            self.tagger_model_id.as_deref(),
            request,
        )
        .await
        .map_err(map_capture_error)?;

        let body = serde_json::json!({
            "thought_id": resp.thought_id.to_string(),
            "embedding_status": resp.embedding_status,
            "is_duplicate": resp.is_duplicate,
        });

        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Hybrid search across captured thoughts. Combines vector kNN (over the active embedding model) with trigram lexical similarity via reciprocal rank fusion, then applies a recency boost. When a cross-encoder reranker is configured, the top `candidate_pool` post-RRF hits are re-scored and returned in rerank order. Optional `tag_filter` narrows to thoughts whose tags JSONB satisfies a containment query. If the embedder is unreachable, results still come back from the trigram leg and `vector_search_available` is false; if the reranker fails, results come back in RRF + recency order and `rerank_used` is false. Each hit carries the thought's tags so consumers can show / threshold without a follow-up get_thought."
    )]
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
            rerank: args.rerank,
            candidate_pool: args.candidate_pool,
            tag_filter: args.tag_filter,
        };

        let resp = search::search_thoughts(
            &self.pool,
            self.embedder.as_ref(),
            self.reranker.as_deref(),
            request,
        )
        .await
        .map_err(map_read_error)?;

        serde_json::to_string(&search_response_json(&resp))
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Recent thoughts in (optional) scope, ordered newest first. No retrieval scoring — just chronological browsing."
    )]
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

    #[tool(
        description = "Fetch a single thought by ID along with its provenance: whether it's been embedded ('indexed' or 'pending'), when it was embedded, the LLM-extracted tags (with tagger model_id/version/extracted_at), and any retraction state."
    )]
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

    #[tool(
        description = "Retract a thought as untrusted (e.g. the operator captured a wrong claim). Marks `thoughts.retracted_at = NOW()`. The thought row itself stays in the DB (`get_thought` still finds it; the response carries `retracted_at` and `retracted_reason`); retrieval (`search_thoughts`, `recent_thoughts`) skips it. To correct a wrong thought, retract it and capture a corrected one."
    )]
    async fn retract_thought(
        &self,
        Parameters(args): Parameters<RetractThoughtArgs>,
    ) -> Result<String, String> {
        let thought_id = ThoughtId::from_str(&args.thought_id)
            .map_err(|e| format!("invalid thought_id: {e}"))?;

        let resp = retract::retract_thought(
            &self.pool,
            RetractThoughtRequest {
                thought_id,
                reason: args.reason,
            },
        )
        .await
        .map_err(map_retract_error)?;

        let body = serde_json::json!({
            "retracted": resp.retracted,
        });
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Create a thought-to-thought relation in the M5 graph layer. Asserts an edge with one of six closed-vocabulary relations: replaces, requires, references, belongs_to, decided_by, refines. Idempotent on the (from, relation, to) triple — re-asserting the same edge returns is_new=false and the existing link_id. Validates that both endpoints exist and that from != to; returns clear error strings otherwise. Use this to link a thought to one it refines, replaces, references, depends on, belongs under, or that decided it. Heterogeneous targets (to-entity, to-person, to-URL) and tagger-extracted relations are not in M5 — use `link_thoughts` agent-side."
    )]
    async fn link_thoughts(
        &self,
        Parameters(args): Parameters<LinkThoughtsArgs>,
    ) -> Result<String, String> {
        let from = ThoughtId::from_str(&args.from_thought_id)
            .map_err(|e| format!("invalid from_thought_id: {e}"))?;
        let to = ThoughtId::from_str(&args.to_thought_id)
            .map_err(|e| format!("invalid to_thought_id: {e}"))?;
        let relation: RelationKind = args
            .relation
            .parse()
            .map_err(|e: engram_core::UnknownRelationKind| e.to_string())?;

        let resp = link::link_thoughts(
            &self.pool,
            LinkThoughtsRequest {
                from_thought_id: from,
                relation,
                to_thought_id: to,
                note: args.note,
            },
        )
        .await
        .map_err(map_link_error)?;

        let body = serde_json::json!({
            "link_id": resp.link_id.to_string(),
            "from_thought_id": resp.from_thought_id.to_string(),
            "relation": resp.relation.as_str(),
            "to_thought_id": resp.to_thought_id.to_string(),
            "is_new": resp.is_new,
        });
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Delete a thought-to-thought edge by its (from, relation, to) triple. Idempotent — deleting an already-missing edge returns existed=false and is not an error. Mirrors `link_thoughts`'s argument shape (without `note`)."
    )]
    async fn unlink_thoughts(
        &self,
        Parameters(args): Parameters<UnlinkThoughtsArgs>,
    ) -> Result<String, String> {
        let from = ThoughtId::from_str(&args.from_thought_id)
            .map_err(|e| format!("invalid from_thought_id: {e}"))?;
        let to = ThoughtId::from_str(&args.to_thought_id)
            .map_err(|e| format!("invalid to_thought_id: {e}"))?;
        let relation: RelationKind = args
            .relation
            .parse()
            .map_err(|e: engram_core::UnknownRelationKind| e.to_string())?;

        let resp = link::unlink_thoughts(&self.pool, from, relation, to)
            .await
            .map_err(map_link_error)?;

        let body = serde_json::json!({
            "existed": resp.existed,
        });
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Walk the M5 thought-to-thought graph from a single thought. Returns grouped `outbound` (edges where this thought is `from`) and `inbound` (edges where it's `to`) arrays. Each entry carries the related thought's id, scope, content_preview (first 400 chars), retracted-state flag, the edge's relation/note/source, and timestamps. Optional `relations` array restricts to specific relation types; optional `direction` ('outbound' | 'inbound' | 'both') is the traversal scope (default 'both'). Retracted thoughts on either side are included with `retracted: true` — the caller decides whether to show, dim, or hide them."
    )]
    async fn get_related_thoughts(
        &self,
        Parameters(args): Parameters<GetRelatedThoughtsArgs>,
    ) -> Result<String, String> {
        let thought_id = ThoughtId::from_str(&args.thought_id)
            .map_err(|e| format!("invalid thought_id: {e}"))?;

        let relations = match args.relations {
            Some(rs) => {
                let mut parsed = Vec::with_capacity(rs.len());
                for r in rs {
                    let kind: RelationKind = r
                        .parse()
                        .map_err(|e: engram_core::UnknownRelationKind| e.to_string())?;
                    parsed.push(kind);
                }
                Some(parsed)
            }
            None => None,
        };

        let direction = match args.direction.as_deref() {
            Some(s) => s
                .parse()
                .map_err(|e: engram_core::UnknownLinkDirection| e.to_string())?,
            None => LinkDirection::default(),
        };

        let resp = relate::get_related_thoughts(
            &self.pool,
            GetRelatedThoughtsRequest {
                thought_id,
                relations,
                direction,
            },
        )
        .await
        .map_err(map_relate_error)?;

        let body = related_thoughts_response_json(&resp);
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
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

fn map_link_error(err: LinkError) -> String {
    match err {
        LinkError::SelfLink => {
            "from_thought_id and to_thought_id must differ — self-links are not supported"
                .to_string()
        }
        LinkError::FromThoughtMissing(id) => format!("from_thought_id {id} not found"),
        LinkError::ToThoughtMissing(id) => format!("to_thought_id {id} not found"),
        LinkError::NoteTooLong { got, max } => {
            format!("note too long: {got} bytes (max {max} = {MAX_LINK_NOTE_LEN})")
        }
        LinkError::Storage(e) => {
            tracing::error!(error = %e, "link/unlink storage error");
            "internal database error".to_string()
        }
    }
}

fn map_relate_error(err: RelateError) -> String {
    match err {
        RelateError::ThoughtNotFound(id) => format!("thought not found: {id}"),
        RelateError::Storage(e) => {
            tracing::error!(error = %e, "get_related_thoughts storage error");
            "internal database error".to_string()
        }
    }
}

fn related_thoughts_response_json(
    resp: &crate::relate::GetRelatedThoughtsResponse,
) -> serde_json::Value {
    fn hit_to_json(h: &crate::relate::RelatedThoughtHit) -> serde_json::Value {
        serde_json::json!({
            "link_id": h.link_id.to_string(),
            "relation": h.relation.as_str(),
            "thought_id": h.thought_id.to_string(),
            "scope": h.scope.as_str(),
            "content_preview": h.content_preview,
            "content_truncated": h.content_truncated,
            "thought_created_at": h.thought_created_at
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_default(),
            "link_created_at": h.link_created_at
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_default(),
            "link_source": h.link_source.as_str(),
            "note": h.note,
            "retracted": h.retracted,
        })
    }

    let outbound: Vec<_> = resp.outbound.iter().map(hit_to_json).collect();
    let inbound: Vec<_> = resp.inbound.iter().map(hit_to_json).collect();
    serde_json::json!({
        "thought_id": resp.thought_id.to_string(),
        "outbound": outbound,
        "inbound": inbound,
    })
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
                "tags": h.tags,
                "vector_score": h.vector_score,
                "trigram_score": h.trigram_score,
                "rrf_score": h.rrf_score,
                "rerank_score": h.rerank_score,
            })
        })
        .collect();
    serde_json::json!({
        "results": results,
        "vector_search_available": resp.vector_search_available,
        "rerank_used": resp.rerank_used,
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
            "tags": resp.thought.tags,
            "tags_extractor_model": resp.thought.tags_extractor_model,
            "tags_extractor_version": resp.thought.tags_extractor_version,
            "tags_extracted_at": resp.thought.tags_extracted_at.and_then(|t| t.format(&time::format_description::well_known::Rfc3339).ok()),
            "retracted_at": resp.retracted_at.and_then(|t| t.format(&time::format_description::well_known::Rfc3339).ok()),
            "retracted_reason": resp.retracted_reason,
        },
    })
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EngramServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(SERVER_INSTRUCTIONS)
    }
}

/// Server-level instructions surfaced during the MCP initialization
/// handshake. Cross-cutting orientation that doesn't belong on any single
/// tool — primarily the `tags` shape, since `tag_filter` is the one MCP
/// argument whose valid values aren't fully derivable from the JSON Schema
/// alone (the schema admits any JSONB object; we want clients to know the
/// closed `kind` enum and the open-vocabulary array fields).
pub const SERVER_INSTRUCTIONS: &str = "\
Engram — self-hosted MCP-native memory service.

Storage model: thoughts are the unit. Each thought has:
- scope: per-thought string label (exact-match filter; keep flat).
- metadata: agent-supplied JSONB blob (e.g. {client_name, session_id, tool_name}).
- tags: LLM-extracted metadata sidecar (advisory; advisory means consumers may filter or de-emphasize, but tags don't gate retrieval).

`tags` shape (auto-extracted by the tagger, distinct from `metadata`):
  { people: string[], entities: string[], action_items: string[], topics: string[] (1-3 short lowercase tags),
    dates_mentioned: string[],
    kind: 'observation' | 'task' | 'idea' | 'reference' | 'person_note' | 'session' | null }
`entities` (proper-noun-style identifiers — projects, products, libraries, tools mentioned by name, e.g. \"engram\", \"pgvector\") is distinct from `topics` (broader subject categories the thought falls under, e.g. \"memory-systems\", \"databases\"). The `kind` enum is closed at those six values (plus null). Array fields are open-vocabulary strings.

Use `tag_filter` on `search_thoughts` for JSONB-containment filtering. Examples:
  {\"kind\": \"task\"}              → only task-classified thoughts
  {\"people\": [\"Sarah\"]}       → only thoughts mentioning Sarah
  {\"entities\": [\"engram\"]}    → only thoughts mentioning engram by name
  {\"topics\": [\"rust\"]}        → only thoughts tagged with the rust subject category
Top-level keys AND together; array values match by subset containment.

`capture` is idempotent on content via SHA-256 fingerprint: same content captured twice returns the existing `thought_id` with `is_duplicate: true` and no new embedding/tag jobs enqueue.

Relations (M5+): thoughts can be linked with a closed-vocabulary `(from, relation, to)` edge. Seven relations (M5 shipped 6; M5.1 added `supports` after dogfood):
  replaces, requires, references, supports, belongs_to, decided_by, refines
Distinguish `references` (prose-level citation / contextual mention) from `supports` (evidential / corroborative — the newer thought confirms a claim made in the older one). Endpoints are thought_ids only — heterogeneous targets (entities, people, URLs) and tagger-extracted relations are not in M5. Use:
  - `link_thoughts(from_thought_id, relation, to_thought_id, note?)` → idempotent on the (from, relation, to) triple; returns `is_new` + the link_id.
  - `unlink_thoughts(from_thought_id, relation, to_thought_id)` → idempotent on already-deleted.
  - `get_related_thoughts(thought_id, relations?, direction?)` → grouped `outbound` + `inbound` arrays with full edge metadata and a content_preview for each related thought.
Edges survive thought retraction (retracted thoughts surface with `retracted: true`); to fully sever a link, use `unlink_thoughts`.

Tools: `capture`, `search_thoughts`, `recent_thoughts`, `get_thought`, `retract_thought`, `link_thoughts`, `unlink_thoughts`, `get_related_thoughts`.";

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{TagKind, Tags};
    use engram_embed::FakeEmbedder;

    fn server(pool: PgPool) -> EngramServer {
        EngramServer::new(pool, Arc::new(FakeEmbedder::new()), None, None)
    }

    /// Regression pin: the server-level instructions surface the `tags`
    /// shape at the MCP initialization handshake so connecting clients
    /// have orientation on the closed `kind` enum + open-vocabulary array
    /// fields without having to discover them by reading per-tool schemas.
    /// If the SERVER_INSTRUCTIONS text drifts and loses this orientation,
    /// agents lose their reliable mental model for `tag_filter` values.
    #[test]
    fn server_instructions_advertise_tags_shape_and_kind_enum() {
        let s = SERVER_INSTRUCTIONS;
        // Tag fields named — including the M4.1 entities split.
        assert!(
            s.contains("people"),
            "instructions should name the `people` field"
        );
        assert!(
            s.contains("entities"),
            "instructions should name the `entities` field"
        );
        assert!(s.contains("action_items"));
        assert!(s.contains("topics"));
        assert!(s.contains("dates_mentioned"));
        // The entities-vs-topics distinction is the load-bearing M4.1 addition.
        assert!(
            s.contains("distinct from `topics`") || s.contains("distinct from"),
            "instructions should disambiguate `entities` from `topics`"
        );
        // The closed kind enum is the load-bearing part — every value must appear.
        for variant in [
            "observation",
            "task",
            "idea",
            "reference",
            "person_note",
            "session",
        ] {
            assert!(
                s.contains(variant),
                "instructions should list kind variant `{variant}`",
            );
        }
        // The agent-supplied `metadata` vs auto-extracted `tags` disambiguation
        // is the terminology landmine — keep it pinned.
        assert!(
            s.contains("metadata") && s.contains("tags"),
            "instructions should disambiguate `metadata` (agent-supplied) from `tags` (LLM-extracted)",
        );
        // Capture-time idempotency is worth advertising — agents that
        // double-capture should expect `is_duplicate: true`.
        assert!(s.contains("is_duplicate"));
        // M5+: the relation vocabulary is the load-bearing closed set; every
        // value must appear so agents have a reliable reference for the
        // link/unlink/get_related_thoughts tools. `supports` was added in
        // M5.1 after day-one dogfood showed `references` was over-firing.
        for relation in [
            "replaces",
            "requires",
            "references",
            "supports",
            "belongs_to",
            "decided_by",
            "refines",
        ] {
            assert!(
                s.contains(relation),
                "instructions should list relation `{relation}`",
            );
        }
        for tool in ["link_thoughts", "unlink_thoughts", "get_related_thoughts"] {
            assert!(
                s.contains(tool),
                "instructions should advertise tool `{tool}`"
            );
        }
    }

    fn server_with_tagger(pool: PgPool) -> EngramServer {
        EngramServer::new(
            pool,
            Arc::new(FakeEmbedder::new()),
            None,
            Some("fake/tagger".to_string()),
        )
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
        assert_eq!(json["embedding_status"], "pending");
        assert_eq!(json["is_duplicate"], false);
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
    async fn capture_tool_returns_duplicate_flag_on_repeat_content(pool: PgPool) {
        let s = server(pool);
        let first_raw = s
            .capture(Parameters(CaptureArgs {
                content: "duplicate content".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let first: serde_json::Value = serde_json::from_str(&first_raw).unwrap();
        assert_eq!(first["is_duplicate"], false);

        let second_raw = s
            .capture(Parameters(CaptureArgs {
                content: "duplicate content".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let second: serde_json::Value = serde_json::from_str(&second_raw).unwrap();
        assert_eq!(second["is_duplicate"], true);
        assert_eq!(second["thought_id"], first["thought_id"]);
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
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["vector_search_available"], true);
        let results = json["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert!(results[0]["thought_id"].is_string());
        assert!(
            results[0]["content"]
                .as_str()
                .unwrap()
                .contains("tcgplayer")
        );
        // Each hit carries a tags object (empty by default).
        assert!(results[0]["tags"].is_object());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_tool_response_carries_tags_per_hit(pool: PgPool) {
        let s = server(pool.clone());
        let cap_raw = s
            .capture(Parameters(CaptureArgs {
                content: "tag-aware tcgplayer note".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let cap_json: serde_json::Value = serde_json::from_str(&cap_raw).unwrap();
        let thought_id = ThoughtId::from_str(cap_json["thought_id"].as_str().unwrap()).unwrap();

        // Write tags directly via storage.
        let tags = Tags {
            topics: vec!["rust".into()],
            kind: Some(TagKind::Idea),
            ..Tags::default()
        };
        engram_storage::update_thought_tags(&pool, thought_id, &tags, "fake/tagger", 1)
            .await
            .unwrap();

        let raw = s
            .search_thoughts(Parameters(SearchThoughtsArgs {
                query: "tcgplayer".into(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let results = json["results"].as_array().unwrap();
        let hit = results
            .iter()
            .find(|h| h["thought_id"] == cap_json["thought_id"])
            .expect("inserted hit present");
        assert_eq!(hit["tags"]["topics"], serde_json::json!(["rust"]));
        assert_eq!(hit["tags"]["kind"], "idea");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_thoughts_tool_applies_tag_filter(pool: PgPool) {
        let s = server(pool.clone());
        // Capture two thoughts; tag one with kind=task.
        let task_raw = s
            .capture(Parameters(CaptureArgs {
                content: "task one keyword".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let _other_raw = s
            .capture(Parameters(CaptureArgs {
                content: "idea two keyword".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let task_json: serde_json::Value = serde_json::from_str(&task_raw).unwrap();
        let task_id = ThoughtId::from_str(task_json["thought_id"].as_str().unwrap()).unwrap();

        engram_storage::update_thought_tags(
            &pool,
            task_id,
            &Tags {
                kind: Some(TagKind::Task),
                ..Tags::default()
            },
            "fake/tagger",
            1,
        )
        .await
        .unwrap();

        let raw = s
            .search_thoughts(Parameters(SearchThoughtsArgs {
                query: "keyword".into(),
                scope: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: Some(serde_json::json!({"kind": "task"})),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["thought_id"], task_json["thought_id"]);
    }

    /// Regression: Phase C dropped the unified `score` field.
    #[sqlx::test(migrations = "../../migrations")]
    async fn search_response_omits_score_field(pool: PgPool) {
        let s = server(pool);
        s.capture(Parameters(CaptureArgs {
            content: "reproducible builds via Nix".into(),
            source: "test".into(),
            scope: None,
            metadata: None,
        }))
        .await
        .unwrap();

        let raw = s
            .search_thoughts(Parameters(SearchThoughtsArgs {
                query: "Nix".into(),
                scope: None,
                limit: None,
                recency_half_life_days: None,
                rerank: None,
                candidate_pool: None,
                tag_filter: None,
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let results = json["results"].as_array().unwrap();
        assert!(!results.is_empty());
        let first = &results[0];
        assert!(
            first.get("score").is_none(),
            "Phase C dropped `score` from search_thoughts hits"
        );
        assert!(first.get("rrf_score").is_some());
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
        assert_eq!(json["provenance"]["embedding_status"], "pending");
        assert!(json["provenance"]["embedded_at"].is_null());
        // No linked_facts field post-M4.
        assert!(json["provenance"].get("linked_facts").is_none());
        // Tag fields exist (empty defaults).
        assert!(json["provenance"]["tags"].is_object());
        assert!(json["provenance"]["tags_extractor_model"].is_null());
        assert!(json["provenance"]["tags_extractor_version"].is_null());
        assert!(json["provenance"]["tags_extracted_at"].is_null());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_thought_tool_carries_tags_and_tagger_provenance(pool: PgPool) {
        let s = server(pool.clone());
        let cap_raw = s
            .capture(Parameters(CaptureArgs {
                content: "tagged thought".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let cap_json: serde_json::Value = serde_json::from_str(&cap_raw).unwrap();
        let thought_id = ThoughtId::from_str(cap_json["thought_id"].as_str().unwrap()).unwrap();

        let tags = Tags {
            people: vec!["Sarah".into()],
            kind: Some(TagKind::PersonNote),
            ..Tags::default()
        };
        engram_storage::update_thought_tags(&pool, thought_id, &tags, "fake/tagger", 1)
            .await
            .unwrap();

        let raw = s
            .get_thought(Parameters(GetThoughtArgs {
                thought_id: cap_json["thought_id"].as_str().unwrap().to_string(),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            json["provenance"]["tags"]["people"],
            serde_json::json!(["Sarah"])
        );
        assert_eq!(json["provenance"]["tags"]["kind"], "person_note");
        assert_eq!(json["provenance"]["tags_extractor_model"], "fake/tagger");
        assert_eq!(json["provenance"]["tags_extractor_version"], 1);
        assert!(json["provenance"]["tags_extracted_at"].is_string());
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
    async fn retract_thought_tool_marks_thought(pool: PgPool) {
        let s = server(pool.clone());
        let cap_raw = s
            .capture(Parameters(CaptureArgs {
                content: "wrong claim".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let cap_json: serde_json::Value = serde_json::from_str(&cap_raw).unwrap();
        let thought_id = cap_json["thought_id"].as_str().unwrap().to_string();

        let raw = s
            .retract_thought(Parameters(RetractThoughtArgs {
                thought_id: thought_id.clone(),
                reason: Some("test reason".into()),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["retracted"], true);
        // No facts_superseded field post-M4.
        assert!(json.get("facts_superseded").is_none());

        // get_thought surfaces retraction state.
        let raw = s
            .get_thought(Parameters(GetThoughtArgs {
                thought_id: thought_id.clone(),
            }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(json["provenance"]["retracted_at"].is_string());
        assert_eq!(json["provenance"]["retracted_reason"], "test reason");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn retract_thought_tool_errors_on_already_retracted(pool: PgPool) {
        let s = server(pool.clone());
        let cap_raw = s
            .capture(Parameters(CaptureArgs {
                content: "wrong claim".into(),
                source: "test".into(),
                scope: None,
                metadata: None,
            }))
            .await
            .unwrap();
        let cap_json: serde_json::Value = serde_json::from_str(&cap_raw).unwrap();
        let thought_id = cap_json["thought_id"].as_str().unwrap().to_string();

        s.retract_thought(Parameters(RetractThoughtArgs {
            thought_id: thought_id.clone(),
            reason: None,
        }))
        .await
        .unwrap();

        let err = s
            .retract_thought(Parameters(RetractThoughtArgs {
                thought_id,
                reason: None,
            }))
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
        // Capture → worker drains → get_thought reports embedding_status=indexed.
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

        let report = crate::drain::drain_pending_embeddings(&pool, s.embedder.as_ref(), 16)
            .await
            .unwrap();
        assert_eq!(report.embedded, 1);
        assert_eq!(report.failed, 0);

        let raw = s
            .get_thought(Parameters(GetThoughtArgs { thought_id }))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["provenance"]["embedding_status"], "indexed");
        assert!(json["provenance"]["embedded_at"].is_string());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn server_with_tagger_enqueues_tag_job_on_capture(pool: PgPool) {
        let s = server_with_tagger(pool.clone());
        s.capture(Parameters(CaptureArgs {
            content: "needs tagging".into(),
            source: "test".into(),
            scope: None,
            metadata: None,
        }))
        .await
        .unwrap();

        let jobs = engram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].tagger_model_id, "fake/tagger");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn server_without_tagger_skips_tag_enqueue(pool: PgPool) {
        let s = server(pool.clone());
        s.capture(Parameters(CaptureArgs {
            content: "no tagger".into(),
            source: "test".into(),
            scope: None,
            metadata: None,
        }))
        .await
        .unwrap();

        let jobs = engram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(jobs.is_empty());
    }
}

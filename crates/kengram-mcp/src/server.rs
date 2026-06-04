//! The rmcp `ServerHandler` wiring. `KengramServer` is the per-connection
//! service factory; it holds an `Arc<dyn Embedder>`, an optional
//! `Arc<dyn Reranker>`, the configured embedder/tagger model ids, and a
//! `PgPool` (cheap to clone). The actual orchestration lives in
//! [`crate::capture`] and [`crate::search`].
//!
//! M4: no more facts. `search_facts`, `correct_fact`, `linked_facts`,
//! `reflect`, `reflect_rerun` are gone. The server's only handles are
//! `capture`, `search_thoughts`, `recent_thoughts`, `get_thought`,
//! `retract_thought`. Tag drainage lives in the worker, not here.

use kengram_core::{
    Embedder, LinkDirection, LinkTarget, Metadata, RelationKind, Scope, Source, ThoughtId,
};
use kengram_embed::Reranker;
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        InitializeRequestParams, InitializeResult, ProtocolVersion, ServerCapabilities, ServerInfo,
    },
    schemars,
    service::{MaybeSendFuture, RequestContext},
    tool, tool_handler, tool_router,
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
    self, GetThoughtResponse, ListScopesRequest, ListScopesResponse, ReadError, RecentRequest,
    RecentResponse, SearchRequest, SearchResponse,
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
        description = "Optional free-form metadata object. Recommended keys: client_name, session_id, tool_name, agent_role, for_audience. DO NOT use this field to encode references to other thoughts (refines, replaces, etc.) — metadata is opaque to retrieval and graph traversal. For cross-thought structure, use `link_thoughts` after capture."
    )]
    // Map<String, Value> rather than Value: ensures the JSON schema renders
    // with a concrete `type: "object"` instead of (no type). claude.ai's MCP
    // client strips fields without concrete types from outbound tool calls;
    // typing this as Map<...> keeps it forwarded. Semantically a strict
    // tightening — metadata was always supposed to be an object.
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchThoughtsArgs {
    #[schemars(description = "Search query. Required, non-empty.")]
    pub query: String,

    #[schemars(
        description = "Scope filter (exact match). Optional; when omitted, searches across all scopes. For prefix-mode filtering across a namespace of scopes (e.g. all `rjf.*` scopes), use `scope_prefix` instead. Mutually exclusive with `scope_prefix` — supplying both returns an error. Call `list_scopes` to discover what's in use."
    )]
    pub scope: Option<String>,

    #[schemars(
        description = "Prefix filter on scope. Optional; matches scopes starting with this string (e.g. `scope_prefix: \"rjf.\"` returns hits from `rjf.professional.cto`, `rjf.personal.health`, etc.). Mutually exclusive with `scope` (exact match) — supply at most one. Pair with `list_scopes(prefix=...)` for a discover-then-query workflow across a namespace of related scopes."
    )]
    pub scope_prefix: Option<String>,

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
        description = "Optional JSONB-containment filter applied to each thought's `tags` field. Tags are LLM-extracted metadata with shape: { people: string[], entities: string[] (named proper-noun-style identifiers — projects, products, libraries, tools, e.g. \"kengram\", \"pgvector\"), action_items: string[], topics: string[] (1-3 short lowercase subject categories — e.g. \"rust\", \"memory-systems\"), dates_mentioned: string[], kind: 'observation' | 'task' | 'idea' | 'reference' | 'person_note' | 'session' | 'decision_record' | null }. Distinguish `entities` (specific named things mentioned by name) from `topics` (broader subject categories the thought falls under). The `kind` enum is closed at the values listed; the array fields are open-vocabulary strings. Examples: {\"kind\": \"task\"} returns only thoughts the tagger classified as tasks; {\"people\": [\"Sarah\"]} returns thoughts whose people-tag contains Sarah; {\"entities\": [\"kengram\"]} returns thoughts mentioning kengram by name; {\"topics\": [\"rust\"], \"kind\": \"idea\"} combines both (top-level keys AND together; array values are subset-match). Empty object {} is a no-op. Filters compose with `scope` (or `scope_prefix`, whichever is set) via AND. Note: `entities` is LLM-extracted and best-effort — a known structural ceiling on adjectival-vs-name discrimination means the field may include descriptive phrases alongside legitimate names. Treat `tag_filter: {\"entities\": [...]}` as a positive signal rather than a strict membership claim."
    )]
    // Map<String, Value> rather than Value: ensures the JSON schema renders
    // with a concrete `type: "object"` instead of (no type). claude.ai's MCP
    // client strips fields without concrete types from outbound tool calls
    // — empirically reproduced 2026-05-18 against the live kengram server.
    // Tightening to Map is semantically correct (the filter must be an
    // object for JSONB-containment to make sense).
    pub tag_filter: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecentThoughtsArgs {
    #[schemars(
        description = "Scope filter (exact match). Optional; when omitted, returns across all scopes. For prefix-mode filtering, use `scope_prefix` instead. Mutually exclusive with `scope_prefix`."
    )]
    pub scope: Option<String>,

    #[schemars(
        description = "Prefix filter on scope (e.g. `scope_prefix: \"rjf.\"` matches `rjf.professional.cto`, `rjf.personal.health`, etc.). Mutually exclusive with `scope` — supplying both returns an error."
    )]
    pub scope_prefix: Option<String>,

    #[schemars(description = "Max results. Optional; defaults to 10, max 100.")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListScopesArgs {
    #[schemars(
        description = "Optional prefix filter. When supplied, only scopes starting with this string are returned (e.g. `prefix: \"rjf.\"` returns `rjf.professional.cto`, `rjf.personal.health`, etc.). Omit for the full scope set."
    )]
    pub prefix: Option<String>,
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
        description = "Relation type. Must be one of the closed vocabulary: 'replaces' (this thought replaces an earlier one — most recent supersedes), 'requires' (this thought depends on another), 'references' (this thought points at another for context, like a citation — passive mention), 'supports' (this thought makes a claim that ACTIVELY CONFIRMS a claim made in another — experimental evidence, corroborating data, logical support; direction is FROM=confirmer, TO=claim-maker; ask 'does the FROM thought itself make a confirming claim?' — if it's just adjacent or topical, use `references` instead), 'belongs_to' (this thought is a member/sub-element of another, e.g. a finding under a parent thread), 'decided_by' (this thought is a decision attributed to another, e.g. a person-note or session-anchor), 'refines' (this thought is a refinement/iteration of an earlier one — both stand, but the newer thought represents updated thinking).\n\nCommon mistakes to avoid:\n- DO NOT use `refines` for citation or evidence. `refines` means the newer thought represents updated thinking on the SAME proposition — not 'the newer thought cites the older one for context.' Use `references` (or `supports` if the newer thought makes a confirming claim) instead.\n- DO NOT use `belongs_to` when the target is a peer or sibling — the v1 vocabulary lacks a sibling/peer-grouping relation. Model the parent (e.g., the experiment, the session) explicitly as its own thought (or, if it's not a thought at all, pass it as `to_entity`) and use `belongs_to` against that.\n- DO NOT use `decided_by` unless there is a clear decision-maker attribution. 'The team converged on X' is `decided_by` Team; 'the research suggests X' is `supports`, not `decided_by`.\n- DO NOT use `replaces` for refinement. `replaces` means the older thought is no longer the current thinking; use `refines` when both stand and the newer one just represents updated thinking.\n- DO NOT use `supports` for passive citation or summarization. `supports` requires the FROM thought to itself make a claim that confirms TO's claim. Summary/aggregation edges (FROM summarizes data points TO) are `references`, not `supports`. Passive prose mentions (FROM cites TO without endorsing) are `references`, not `supports`."
    )]
    pub relation: String,

    #[schemars(
        description = "Target thought ID (UUID string) when linking to another thought. Mutually exclusive with `to_entity`, `to_person`, `to_url` — supply exactly one of the four target fields."
    )]
    pub to_thought_id: Option<String>,

    #[schemars(
        description = "Target entity name (free-text string, up to 200 chars) when the natural target of the relation isn't a thought — e.g. an experiment, a project, a session. Mutually exclusive with `to_thought_id`, `to_person`, `to_url`."
    )]
    pub to_entity: Option<String>,

    #[schemars(
        description = "Target person name (free-text string, up to 200 chars) when attributing a thought to a person (e.g. `decided_by` Ron). Mutually exclusive with `to_thought_id`, `to_entity`, `to_url`."
    )]
    pub to_person: Option<String>,

    #[schemars(
        description = "Target URL (must start with http:// or https://, up to 2048 chars) when referencing an external resource. Mutually exclusive with `to_thought_id`, `to_entity`, `to_person`."
    )]
    pub to_url: Option<String>,

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

    #[schemars(
        description = "Target thought ID (UUID string) when the edge being removed targeted a thought. Mutually exclusive with `to_entity`, `to_person`, `to_url`."
    )]
    pub to_thought_id: Option<String>,

    #[schemars(
        description = "Target entity name when the edge being removed targeted an entity. Mutually exclusive with `to_thought_id`, `to_person`, `to_url`."
    )]
    pub to_entity: Option<String>,

    #[schemars(
        description = "Target person name when the edge being removed targeted a person. Mutually exclusive with `to_thought_id`, `to_entity`, `to_url`."
    )]
    pub to_person: Option<String>,

    #[schemars(
        description = "Target URL when the edge being removed targeted a URL. Mutually exclusive with `to_thought_id`, `to_entity`, `to_person`."
    )]
    pub to_url: Option<String>,
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
        description = "Optional filter to a subset of target kinds (e.g. ['thought','url']). Each item must be one of: thought, entity, person, url. Applies to outbound edges only — inbound edges are always thought→thought by schema. Omit to return every kind."
    )]
    pub target_kinds: Option<Vec<String>>,

    #[schemars(
        description = "Traversal direction: 'outbound' (edges where the queried thought is the source — 'this refines X'), 'inbound' (edges where it is the target — 'X refines this'), or 'both' (default). Response always groups results into separate `outbound` and `inbound` arrays regardless."
    )]
    pub direction: Option<String>,
}

#[derive(Clone)]
pub struct KengramServer {
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

impl KengramServer {
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

impl std::fmt::Debug for KengramServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KengramServer")
            .field("model_id", &self.embedder.model().id)
            .field("tagger_model_id", &self.tagger_model_id)
            .finish()
    }
}

#[tool_router]
impl KengramServer {
    #[tool(
        description = "Capture a thought into kengram's persistent memory. Returns the thought_id and embedding_status='pending'. The thought is durable and findable by trigram (lexical) search immediately; vector search picks it up on the next worker tick (default 5 seconds). Identical content (SHA-256 of the bytes) is deduplicated — the response will include `is_duplicate: true` and the pre-existing thought_id when the fingerprint collides. To express that this thought refines, replaces, references, supports, depends on, belongs under, or was decided by another thought, use `link_thoughts` after capture — these relations are queryable via `get_related_thoughts`. Do NOT encode cross-thought relationships in the `metadata` field; metadata is opaque to retrieval and graph traversal. To make a term filterable as an entity or topic, put it in the opening sentence — the tagger lifts phrases from prose surface vocabulary, with extraction probability falling off after the opening."
    )]
    async fn capture(&self, Parameters(args): Parameters<CaptureArgs>) -> Result<String, String> {
        let source = Source::new(args.source).map_err(|e| format!("invalid source: {e}"))?;

        let scope = match args.scope {
            Some(s) => Some(Scope::new(s).map_err(|e| format!("invalid scope: {e}"))?),
            None => None,
        };

        // Map<String, Value> → Value::Object → Metadata. The Map type on
        // the args struct keeps the schema's `type: "object"` concrete so
        // claude.ai's MCP client forwards the field intact.
        let metadata = args
            .metadata
            .map(serde_json::Value::Object)
            .map(Metadata::from);

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
        description = "Hybrid search across captured thoughts. Combines vector kNN (over the active embedding model) with trigram lexical similarity via reciprocal rank fusion, then applies a recency boost. On natural-language queries the trigram leg's surface-similarity threshold typically gates everything out, so it primarily contributes a co-occurrence boost when a query hits exact surface terms (acronyms, proper nouns, code identifiers) and serves as the retrieval fallback when the embedder is unreachable. For typical content-search the candidate pool is dominated by vector kNN with rerank as the final discriminator. When a cross-encoder reranker is configured, the top `candidate_pool` post-RRF hits are re-scored and returned in rerank order. Scope filtering: use `scope` for exact match or `scope_prefix` for namespace match — supply at most one (mutually exclusive; supplying both returns an error). Optional `tag_filter` narrows to thoughts whose tags JSONB satisfies a containment query. Scope filter (whichever you pick), `tag_filter`, and the search query all compose via AND. If the embedder is unreachable, results still come back from the trigram leg and `vector_search_available` is false; if the reranker fails, results come back in RRF + recency order and `rerank_used` is false. Each hit carries the thought's tags so consumers can show / threshold without a follow-up get_thought. For each hit, follow up with `get_related_thoughts(thought_id)` to walk the graph layer — refinements, replacements, supports, citations, and other edges the agent has linked. The search-then-traverse pattern is how a discovery walk arrives at the relational context of a hit."
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
            scope_prefix: args.scope_prefix,
            limit: args.limit,
            recency_half_life_days: args.recency_half_life_days,
            rerank: args.rerank,
            candidate_pool: args.candidate_pool,
            // Map<String, Value> on the wire → Value::Object for the
            // orchestrator's filter logic. Keeps the schema concrete (so
            // claude.ai forwards the field) without changing the
            // SearchRequest API.
            tag_filter: args.tag_filter.map(serde_json::Value::Object),
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
            scope_prefix: args.scope_prefix,
            limit: args.limit,
        };

        let resp = search::recent_thoughts(&self.pool, request)
            .await
            .map_err(map_read_error)?;

        serde_json::to_string(&recent_response_json(&resp))
            .map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Enumerate scopes currently in use in the corpus. Returns each scope with its `thought_count`, `first_activity_at` (when the scope first appeared), and `last_activity_at` (when it was most recently used). Optional `prefix` filter matches scopes starting with the given string (e.g. `prefix: \"rjf.\"` returns all `rjf.*` scopes). Sorted by `last_activity_at` descending — most recently used scopes first. Retracted thoughts are excluded from counts (scopes whose every thought is retracted don't appear). Use this before capturing to pick an existing scope; do not invent new scopes silently. Combine with `scope_prefix` on `search_thoughts` or `recent_thoughts` to query across a namespace of related scopes."
    )]
    async fn list_scopes(
        &self,
        Parameters(args): Parameters<ListScopesArgs>,
    ) -> Result<String, String> {
        let request = ListScopesRequest {
            prefix: args.prefix,
        };
        let resp = search::list_scopes(&self.pool, request)
            .await
            .map_err(map_read_error)?;

        serde_json::to_string(&list_scopes_response_json(&resp))
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
        description = "Create a relation from a thought to a polymorphic target in the kengram graph layer. Asserts an edge with one of seven closed-vocabulary relations: replaces, requires, references, supports, belongs_to, decided_by, refines. The target side can be: another thought (`to_thought_id`), a free-text entity name (`to_entity` — for experiments, projects, sessions, abstract concepts), a person name (`to_person` — for attribution like `decided_by`), or a URL (`to_url` — for external resources). Supply exactly one of the four target fields. Idempotent on the (from, relation, to_kind, to_value) quadruple — re-asserting the same live edge returns is_new=false and the existing link_id. If the edge was previously soft-deleted via `unlink_thoughts`, a fresh live row is inserted and is_new=true. Validates that thought endpoints exist (from + to_thought_id), that from != to_thought_id when targeting a thought, that to_url starts with http:// or https://, and that to_entity/to_person aren't empty; returns clear error strings otherwise."
    )]
    async fn link_thoughts(
        &self,
        Parameters(args): Parameters<LinkThoughtsArgs>,
    ) -> Result<String, String> {
        let from = ThoughtId::from_str(&args.from_thought_id)
            .map_err(|e| format!("invalid from_thought_id: {e}"))?;
        let relation: RelationKind = args
            .relation
            .parse()
            .map_err(|e: kengram_core::UnknownRelationKind| e.to_string())?;
        let target = parse_link_target(
            args.to_thought_id.as_deref(),
            args.to_entity,
            args.to_person,
            args.to_url,
        )?;

        let resp = link::link_thoughts(
            &self.pool,
            LinkThoughtsRequest {
                from_thought_id: from,
                relation,
                target,
                note: args.note,
            },
        )
        .await
        .map_err(map_link_error)?;

        let body = serde_json::json!({
            "link_id": resp.link_id.to_string(),
            "from_thought_id": resp.from_thought_id.to_string(),
            "relation": resp.relation.as_str(),
            "to_kind": resp.target.kind_str(),
            "to_value": resp.target.value_str(),
            "is_new": resp.is_new,
        });
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Soft-delete a link by its (from, relation, target) triple. Returns a three-way status discriminator: `deleted_now` (the edge was live and was just removed), `already_deleted` (the edge previously existed but had already been soft-deleted), or `never_existed` (no edge with this triple ever existed). Soft-deleted edges sit inert in the table — re-creating the same edge via `link_thoughts` succeeds (fresh row). Supply the target the same way as `link_thoughts` (exactly one of `to_thought_id`, `to_entity`, `to_person`, `to_url`)."
    )]
    async fn unlink_thoughts(
        &self,
        Parameters(args): Parameters<UnlinkThoughtsArgs>,
    ) -> Result<String, String> {
        let from = ThoughtId::from_str(&args.from_thought_id)
            .map_err(|e| format!("invalid from_thought_id: {e}"))?;
        let relation: RelationKind = args
            .relation
            .parse()
            .map_err(|e: kengram_core::UnknownRelationKind| e.to_string())?;
        let target = parse_link_target(
            args.to_thought_id.as_deref(),
            args.to_entity,
            args.to_person,
            args.to_url,
        )?;

        let resp = link::unlink_thoughts(&self.pool, from, relation, &target)
            .await
            .map_err(map_link_error)?;

        let body = serde_json::json!({
            "status": resp.status.as_str(),
        });
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }

    #[tool(
        description = "Walk the kengram link graph from a single thought. Returns grouped `outbound` (edges where this thought is `from`) and `inbound` (edges where it's `to`) arrays. Each entry carries the edge's `link_id`, `relation`, `to_kind` (`thought` | `entity` | `person` | `url`), `to_value` (the target's UUID/name/URL string), the edge's `link_created_at`/`link_source`/`note`, plus — when `to_kind = thought` — the target thought's `thought_id`, `scope`, `content_preview` (first 400 chars), `content_truncated`, `thought_created_at`, and `retracted` flag. For non-thought targets those thought-specific fields are null. Optional `relations` array restricts to specific relation types; optional `target_kinds` array restricts outbound to specific target kinds (no effect on inbound, which is always thought→thought by schema); optional `direction` ('outbound' | 'inbound' | 'both') is the traversal scope (default 'both'). Retracted thoughts on either side are surfaced with `retracted: true` — the caller decides whether to show, dim, or hide them. Soft-deleted edges are excluded. Edges are agent-supplied via `link_thoughts` and removed via `unlink_thoughts`. If the response is empty, no live edges connect to this thought yet."
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
                        .map_err(|e: kengram_core::UnknownRelationKind| e.to_string())?;
                    parsed.push(kind);
                }
                Some(parsed)
            }
            None => None,
        };

        let direction = match args.direction.as_deref() {
            Some(s) => s
                .parse()
                .map_err(|e: kengram_core::UnknownLinkDirection| e.to_string())?,
            None => LinkDirection::default(),
        };

        let resp = relate::get_related_thoughts(
            &self.pool,
            GetRelatedThoughtsRequest {
                thought_id,
                relations,
                target_kinds: args.target_kinds,
                direction,
            },
        )
        .await
        .map_err(map_relate_error)?;

        let body = related_thoughts_response_json(&resp);
        serde_json::to_string(&body).map_err(|e| format!("response serialization error: {e}"))
    }
}

/// Parse the four optional target args into a `LinkTarget`. Returns an
/// error string when zero or more-than-one is supplied. Also validates
/// the inner string formats (UUID for thought, http(s):// for URL — though
/// the URL prefix check is also enforced in `link::validate_target` and
/// at the DB CHECK).
fn parse_link_target(
    to_thought_id: Option<&str>,
    to_entity: Option<String>,
    to_person: Option<String>,
    to_url: Option<String>,
) -> Result<LinkTarget, String> {
    let count = [
        to_thought_id.is_some(),
        to_entity.is_some(),
        to_person.is_some(),
        to_url.is_some(),
    ]
    .into_iter()
    .filter(|b| *b)
    .count();
    if count == 0 {
        return Err(
            "exactly one of to_thought_id, to_entity, to_person, to_url must be supplied".into(),
        );
    }
    if count > 1 {
        return Err(
            "to_thought_id, to_entity, to_person, and to_url are mutually exclusive — supply exactly one".into(),
        );
    }
    if let Some(id) = to_thought_id {
        let parsed = ThoughtId::from_str(id).map_err(|e| format!("invalid to_thought_id: {e}"))?;
        Ok(LinkTarget::Thought(parsed))
    } else if let Some(name) = to_entity {
        Ok(LinkTarget::Entity(name))
    } else if let Some(name) = to_person {
        Ok(LinkTarget::Person(name))
    } else if let Some(url) = to_url {
        Ok(LinkTarget::Url(url))
    } else {
        unreachable!("count check above guarantees exactly one is Some")
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
        ReadError::ScopeAndPrefixBothSet => {
            "scope and scope_prefix are mutually exclusive; supply at most one".to_string()
        }
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
        LinkError::EmptyTargetName => {
            "to_entity / to_person target must not be empty or whitespace-only".to_string()
        }
        LinkError::TargetNameTooLong { got, max } => {
            format!("to_entity / to_person target too long: {got} bytes (max {max})")
        }
        LinkError::TargetUrlTooLong { got, max } => {
            format!("to_url target too long: {got} bytes (max {max})")
        }
        LinkError::InvalidUrl => "to_url must start with http:// or https://".to_string(),
        LinkError::Storage(e) => {
            tracing::error!(error = %e, "link/unlink storage error");
            "internal database error".to_string()
        }
    }
}

fn map_relate_error(err: RelateError) -> String {
    match err {
        RelateError::ThoughtNotFound(id) => format!("thought not found: {id}"),
        RelateError::UnknownTargetKind(s) => {
            format!("unknown target_kind {s:?} (expected one of: thought, entity, person, url)")
        }
        RelateError::Storage(e) => {
            tracing::error!(error = %e, "get_related_thoughts storage error");
            "internal database error".to_string()
        }
    }
}

fn related_thoughts_response_json(
    resp: &crate::relate::GetRelatedThoughtsResponse,
) -> serde_json::Value {
    fn hit_to_json(h: &crate::relate::RelatedTargetHit) -> serde_json::Value {
        // Thought-target hits carry thought_id + scope + content_preview +
        // thought_created_at + retracted; non-thought hits leave those null.
        let thought_id = match &h.target {
            LinkTarget::Thought(id) => Some(id.to_string()),
            _ => None,
        };
        serde_json::json!({
            "link_id": h.link_id.to_string(),
            "relation": h.relation.as_str(),
            "to_kind": h.target.kind_str(),
            "to_value": h.target.value_str(),
            "thought_id": thought_id,
            "scope": h.scope.as_ref().map(|s| s.as_str()),
            "content_preview": h.content_preview,
            "content_truncated": h.content_truncated,
            "thought_created_at": h.thought_created_at.map(|t| {
                t.format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_default()
            }),
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

fn list_scopes_response_json(resp: &ListScopesResponse) -> serde_json::Value {
    let scopes: Vec<serde_json::Value> = resp
        .scopes
        .iter()
        .map(|s| {
            serde_json::json!({
                "scope": s.scope,
                "thought_count": s.thought_count,
                "first_activity_at": s
                    .first_activity_at
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_default(),
                "last_activity_at": s
                    .last_activity_at
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_default(),
            })
        })
        .collect();
    serde_json::json!({ "scopes": scopes })
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
impl ServerHandler for KengramServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(SERVER_INSTRUCTIONS)
    }

    // rmcp 1.6's default `initialize` ignores the client's requested
    // `protocolVersion` and unconditionally returns `ProtocolVersion::LATEST`.
    // That breaks clients pinned to an older known-good version (e.g. the iOS
    // ChatMcpiOSClient asks for `2025-03-26`). Override to echo any version
    // rmcp itself knows about; fall back to LATEST otherwise.
    fn initialize(
        &self,
        request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<InitializeResult, McpError>> + MaybeSendFuture + '_
    {
        let client_version = request.protocol_version.clone();
        if context.peer.peer_info().is_none() {
            context.peer.set_peer_info(request);
        }
        let negotiated = if ProtocolVersion::KNOWN_VERSIONS.contains(&client_version) {
            client_version.clone()
        } else {
            ProtocolVersion::LATEST
        };
        if negotiated != ProtocolVersion::LATEST {
            tracing::info!(
                client_requested = %client_version,
                negotiated = %negotiated,
                "mcp initialize: echoed client's protocol version"
            );
        }
        let info = self.get_info().with_protocol_version(negotiated);
        std::future::ready(Ok(info))
    }
}

/// Server-level instructions surfaced during the MCP initialization
/// handshake. Cross-cutting orientation that doesn't belong on any single
/// tool — primarily the `tags` shape, since `tag_filter` is the one MCP
/// argument whose valid values aren't fully derivable from the JSON Schema
/// alone (the schema admits any JSONB object; we want clients to know the
/// closed `kind` enum and the open-vocabulary array fields).
pub const SERVER_INSTRUCTIONS: &str = "\
kEngram — self-hosted MCP-native memory service.

Storage model: thoughts are the unit. Each thought has:
- scope: per-thought string label (exact-match filter; keep flat).
- metadata: agent-supplied JSONB blob (e.g. {client_name, session_id, tool_name}).
- tags: LLM-extracted metadata sidecar (advisory; advisory means consumers may filter or de-emphasize, but tags don't gate retrieval).

`tags` shape (auto-extracted by the tagger, distinct from `metadata`):
  { people: string[], entities: string[], action_items: string[], topics: string[] (1-3 short lowercase tags),
    dates_mentioned: string[],
    kind: 'observation' | 'task' | 'idea' | 'reference' | 'person_note' | 'session' | 'decision_record' | null }
`entities` (proper-noun-style identifiers — projects, products, libraries, tools mentioned by name, e.g. \"kengram\", \"pgvector\") is distinct from `topics` (broader subject categories the thought falls under, e.g. \"memory-systems\", \"databases\"). The `kind` enum is closed at those six values (plus null). Array fields are open-vocabulary strings.

Use `tag_filter` on `search_thoughts` for JSONB-containment filtering. Examples:
  {\"kind\": \"task\"}              → only task-classified thoughts
  {\"people\": [\"Sarah\"]}       → only thoughts mentioning Sarah
  {\"entities\": [\"kengram\"]}    → only thoughts mentioning kengram by name
  {\"topics\": [\"rust\"]}        → only thoughts tagged with the rust subject category
Top-level keys AND together; array values match by subset containment.

`capture` is idempotent on content via SHA-256 fingerprint: same content captured twice returns the existing `thought_id` with `is_duplicate: true` and no new embedding/tag jobs enqueue.

Relations (M5+): thoughts can be linked with a closed-vocabulary `(from, relation, target)` edge. Seven relations (M5 shipped 6; M5.1 added `supports` after dogfood):
  replaces, requires, references, supports, belongs_to, decided_by, refines
Distinguish `references` (prose-level citation / contextual mention) from `supports` (evidential / corroborative — the newer thought confirms a claim made in the older one). The `from` side is always a thought; the `to` side can be a thought, an entity name, a person name, or a URL (M5.2). Use:
  - `link_thoughts(from_thought_id, relation, {to_thought_id | to_entity | to_person | to_url}, note?)` → supply exactly one of the four target fields. Idempotent on the (from, relation, to_kind, to_value) quadruple; returns `is_new` + the `link_id` + the `to_kind`/`to_value` discriminator.
  - `unlink_thoughts(from_thought_id, relation, {one-of-four-targets})` → soft-delete; returns a three-way `status`: `deleted_now`, `already_deleted`, or `never_existed`.
  - `get_related_thoughts(thought_id, relations?, target_kinds?, direction?)` → grouped `outbound` + `inbound` arrays. Each hit carries `to_kind`/`to_value`; thought-target hits also include `thought_id`, `scope`, `content_preview`, and `retracted`. Non-thought-target hits leave those fields null.
Edges survive thought retraction (retracted thoughts surface with `retracted: true`); to fully sever a link, use `unlink_thoughts`.

Workflow shape: after capturing a thought, use `link_thoughts` to express structural relationships — these are queryable via `get_related_thoughts` and not encodable in `metadata`. After a search, follow up with `get_related_thoughts(thought_id)` on a hit to walk the graph from there. When the natural target of a relation isn't a thought, prefer the typed `to_entity`/`to_person`/`to_url` over capturing a placeholder thought. Before capturing, call `list_scopes(prefix?)` to discover what scopes are in use — don't invent new scope names silently. Pass the same prefix to `search_thoughts(scope_prefix=...)` or `recent_thoughts(scope_prefix=...)` to query across a namespace of related scopes (exact-match `scope` and prefix-match `scope_prefix` are mutually exclusive).

Tools: `capture`, `search_thoughts`, `recent_thoughts`, `list_scopes`, `get_thought`, `retract_thought`, `link_thoughts`, `unlink_thoughts`, `get_related_thoughts`.";

#[cfg(test)]
mod tests {
    use super::*;
    use kengram_core::{TagKind, Tags};
    use kengram_embed::FakeEmbedder;

    fn server(pool: PgPool) -> KengramServer {
        KengramServer::new(pool, Arc::new(FakeEmbedder::new()), None, None)
    }

    /// Regression pin: the server-level instructions surface the `tags`
    /// shape at the MCP initialization handshake so connecting clients
    /// have orientation on the closed `kind` enum + open-vocabulary array
    /// fields without having to discover them by reading per-tool schemas.
    /// If the SERVER_INSTRUCTIONS text drifts and loses this orientation,
    /// agents lose their reliable mental model for `tag_filter` values.
    /// Regression pin: the JSON schema for `search_thoughts.tag_filter` and
    /// `capture.metadata` must declare a concrete `type` ("object" + "null"),
    /// not just `description`. Without a concrete type, claude.ai's MCP
    /// client silently strips the field from outbound tool calls
    /// (empirically reproduced 2026-05-18). Pinning the schema shape here
    /// catches a regression to `Option<serde_json::Value>` (no type) before
    /// it ships.
    #[test]
    fn tool_args_object_fields_have_concrete_schema_type() {
        let s = schemars::schema_for!(SearchThoughtsArgs);
        let v = serde_json::to_value(&s).unwrap();
        let tag_filter = v
            .get("properties")
            .and_then(|p| p.get("tag_filter"))
            .expect("tag_filter property must exist in SearchThoughtsArgs schema");
        let type_field = tag_filter
            .get("type")
            .expect("tag_filter must declare a concrete `type` (else clients strip it)");
        // schemars renders `Option<Map<String, Value>>` as `type: ["object", "null"]`.
        let types: Vec<&str> = type_field
            .as_array()
            .expect("type should be an array of strings")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(
            types.contains(&"object"),
            "tag_filter type must include `object`; got {types:?}"
        );
        assert!(
            types.contains(&"null"),
            "tag_filter type must include `null` (it's optional); got {types:?}"
        );

        let s = schemars::schema_for!(CaptureArgs);
        let v = serde_json::to_value(&s).unwrap();
        let metadata = v
            .get("properties")
            .and_then(|p| p.get("metadata"))
            .expect("metadata property must exist in CaptureArgs schema");
        let types: Vec<&str> = metadata
            .get("type")
            .expect("metadata must declare a concrete `type`")
            .as_array()
            .expect("type should be an array of strings")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(types.contains(&"object"));
        assert!(types.contains(&"null"));
    }

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
            "decision_record",
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
        for tool in [
            "link_thoughts",
            "unlink_thoughts",
            "get_related_thoughts",
            "list_scopes",
        ] {
            assert!(
                s.contains(tool),
                "instructions should advertise tool `{tool}`"
            );
        }
        // M5.2: the polymorphic target-kind enum is the load-bearing new
        // surface; every kind must appear so agents know they can target
        // non-thoughts. The three-way unlink status enum is also pinned —
        // it's the operator-facing semantic upgrade from the old boolean.
        for kind in ["to_thought_id", "to_entity", "to_person", "to_url"] {
            assert!(
                s.contains(kind),
                "instructions should advertise target field `{kind}`",
            );
        }
        for status in ["deleted_now", "already_deleted", "never_existed"] {
            assert!(
                s.contains(status),
                "instructions should advertise unlink status `{status}`",
            );
        }
    }

    fn server_with_tagger(pool: PgPool) -> KengramServer {
        KengramServer::new(
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
                scope_prefix: None,
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
        kengram_storage::update_thought_tags(&pool, thought_id, &tags, "fake/tagger", 1)
            .await
            .unwrap();

        let raw = s
            .search_thoughts(Parameters(SearchThoughtsArgs {
                query: "tcgplayer".into(),
                scope: None,
                scope_prefix: None,
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

        kengram_storage::update_thought_tags(
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
                scope_prefix: None,
                limit: None,
                recency_half_life_days: Some(0.0),
                rerank: None,
                candidate_pool: None,
                tag_filter: serde_json::json!({"kind": "task"}).as_object().cloned(),
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
                scope_prefix: None,
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
                scope_prefix: None,
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
        kengram_storage::update_thought_tags(&pool, thought_id, &tags, "fake/tagger", 1)
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

        let jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
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

        let jobs = kengram_storage::fetch_pending_tag_jobs(&pool, 10)
            .await
            .unwrap();
        assert!(jobs.is_empty());
    }
}

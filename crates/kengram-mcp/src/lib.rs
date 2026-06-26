//! kengram-mcp: rmcp tool descriptors and orchestration logic for kengram's
//! MCP surface.
//!
//! The orchestration functions (`capture`, `search_thoughts`,
//! `recent_thoughts`, `get_thought`, `retract_thought`) are testable Rust
//! functions that take `&PgPool` + a request struct. The [`KengramServer`]
//! type wires them into rmcp's `ServerHandler` trait so they're invokable
//! over an MCP transport. Background drainers (`drain_pending_embeddings`,
//! `drain_pending_tags`) live in [`drain`] and are driven by `kengram worker`.

pub mod backfill;
pub mod capture;
pub mod drain;
pub mod filters;
pub mod finalize;
pub mod link;
mod normalize;
pub mod query_expansion;
pub mod relate;
pub mod retract;
pub mod search;
pub mod server;
mod validate;

pub use backfill::{BackfillError, BackfillReport, embed_backfill};
pub use capture::{CaptureError, CaptureRequest, CaptureResponse, MAX_CONTENT_LEN, capture};
pub use drain::{
    DrainError, DrainReport, DrainTagsReport, MAX_TAG_ATTEMPTS, apply_tagger_relations,
    drain_pending_embeddings, drain_pending_tags,
};
pub use link::{
    LinkError, LinkThoughtsRequest, LinkThoughtsResponse, MAX_LINK_NOTE_LEN, MAX_TARGET_NAME_LEN,
    MAX_TARGET_URL_LEN, UnlinkStatus, UnlinkThoughtsResponse, link_thoughts, unlink_thoughts,
};
pub use query_expansion::{
    DEFAULT_QUERY_EXPANSION_MAX_HYDE_CHARS, DEFAULT_QUERY_EXPANSION_MAX_VARIANTS,
    DEFAULT_QUERY_EXPANSION_PROMPT_VERSION, NormalizedQueryExpansion,
    OpenAICompatibleQueryExpansionProvider, QueryExpansionConfig, QueryExpansionError,
    QueryExpansionInput, QueryExpansionOutput, QueryExpansionProvider, normalize_expansion_output,
};
pub use relate::{
    GetRelatedThoughtsRequest, GetRelatedThoughtsResponse, RELATED_CONTENT_PREVIEW_LEN,
    RelateError, RelatedTargetHit, get_related_thoughts,
};
pub use retract::{RetractError, RetractThoughtRequest, RetractThoughtResponse, retract_thought};
pub use search::{
    DEFAULT_GRAPH_PER_SEED_CAP, DEFAULT_GRAPH_SEED_COUNT, DEFAULT_GRAPH_TOTAL_CAP,
    DEFAULT_SEARCH_LIMIT, DEFAULT_TOP_K_PER_LEG, GetThoughtResponse, GraphProvenance,
    ListScopesRequest, ListScopesResponse, MAX_GRAPH_PER_SEED_CAP, MAX_GRAPH_SEED_COUNT,
    MAX_GRAPH_TOTAL_CAP, MAX_SEARCH_LIMIT, ReadError, RecentRequest, RecentResponse,
    ScopeSummaryHit, SearchHit, SearchRequest, SearchResponse, SearchRuntimeOptions,
    default_graph_relations, get_thought, list_scopes, recent_thoughts, search_thoughts,
    search_thoughts_with_runtime,
};
pub use server::{
    CaptureArgs, GetRelatedThoughtsArgs, GetThoughtArgs, KengramServer, LinkThoughtsArgs,
    ListScopesArgs, RecentThoughtsArgs, RetractThoughtArgs, SearchThoughtsArgs, UnlinkThoughtsArgs,
};

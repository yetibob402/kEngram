//! engram-mcp: rmcp tool descriptors and orchestration logic for engram's
//! MCP surface.
//!
//! The orchestration functions (`capture`, `search_thoughts`,
//! `recent_thoughts`, `get_thought`, `retract_thought`) are testable Rust
//! functions that take `&PgPool` + a request struct. The [`EngramServer`]
//! type wires them into rmcp's `ServerHandler` trait so they're invokable
//! over an MCP transport. Background drainers (`drain_pending_embeddings`,
//! `drain_pending_tags`) live in [`drain`] and are driven by `engram worker`.

pub mod backfill;
pub mod capture;
pub mod drain;
pub mod link;
pub mod relate;
pub mod retract;
pub mod search;
pub mod server;

pub use backfill::{BackfillError, BackfillReport, embed_backfill};
pub use capture::{CaptureError, CaptureRequest, CaptureResponse, MAX_CONTENT_LEN, capture};
pub use drain::{
    DrainError, DrainReport, DrainTagsReport, MAX_TAG_ATTEMPTS, drain_pending_embeddings,
    drain_pending_tags,
};
pub use link::{
    LinkError, LinkThoughtsRequest, LinkThoughtsResponse, MAX_LINK_NOTE_LEN,
    UnlinkThoughtsResponse, link_thoughts, unlink_thoughts,
};
pub use relate::{
    GetRelatedThoughtsRequest, GetRelatedThoughtsResponse, RELATED_CONTENT_PREVIEW_LEN,
    RelateError, RelatedThoughtHit, get_related_thoughts,
};
pub use retract::{RetractError, RetractThoughtRequest, RetractThoughtResponse, retract_thought};
pub use search::{
    DEFAULT_SEARCH_LIMIT, DEFAULT_TOP_K_PER_LEG, GetThoughtResponse, MAX_SEARCH_LIMIT, ReadError,
    RecentRequest, RecentResponse, SearchHit, SearchRequest, SearchResponse, get_thought,
    recent_thoughts, search_thoughts,
};
pub use server::{
    CaptureArgs, EngramServer, GetRelatedThoughtsArgs, GetThoughtArgs, LinkThoughtsArgs,
    RecentThoughtsArgs, RetractThoughtArgs, SearchThoughtsArgs, UnlinkThoughtsArgs,
};

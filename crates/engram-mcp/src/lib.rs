//! engram-mcp: rmcp tool descriptors and orchestration logic for engram's
//! MCP surface.
//!
//! The orchestration functions (`capture`, plus `search_thoughts`,
//! `recent_thoughts`, `get_thought` in Phase C) are testable Rust functions
//! that take `&PgPool` + `&dyn Embedder` + a request struct. The
//! [`EngramServer`] type wires them into rmcp's `ServerHandler` trait so
//! they're invokable over an MCP transport.

pub mod backfill;
pub mod capture;
pub mod drain;
pub mod search;
pub mod server;

pub use backfill::{BackfillError, BackfillReport, embed_backfill};
pub use capture::{capture, CaptureError, CaptureRequest, CaptureResponse, MAX_CONTENT_LEN};
pub use drain::{drain_pending_embeddings, DrainError, DrainReport};
pub use search::{
    get_thought, recent_thoughts, search_thoughts, GetThoughtResponse, ReadError, RecentRequest,
    RecentResponse, SearchHit, SearchRequest, SearchResponse, DEFAULT_SEARCH_LIMIT,
    DEFAULT_TOP_K_PER_LEG, MAX_SEARCH_LIMIT,
};
pub use server::{CaptureArgs, EngramServer, GetThoughtArgs, RecentThoughtsArgs, SearchThoughtsArgs};

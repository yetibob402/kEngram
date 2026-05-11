//! The rmcp `ServerHandler` wiring. `EngramServer` is the per-connection
//! service factory; it holds an `Arc<dyn Embedder>` and a `PgPool` (both
//! cheap to clone). The actual orchestration lives in [`crate::capture`].

use engram_core::{Embedder, Metadata, Scope, Source};
use rmcp::{
    ServerHandler, model::ServerCapabilities, model::ServerInfo, schemars, tool,
};
use serde::Deserialize;
use sqlx::PgPool;
use std::sync::Arc;

use crate::capture::{self, CaptureError, CaptureRequest, MAX_CONTENT_LEN};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CaptureArgs {
    /// The thought text to capture. Required; non-empty; max 1 MiB.
    #[schemars(description = "The thought text. Required, non-empty, max 1 MiB.")]
    pub content: String,

    /// Provenance label. Required. Convention: `manual`, `agent:claude-code`,
    /// `agent:opencode`, `reflector`, etc.
    #[schemars(description = "Provenance label. Required. Examples: 'manual', 'agent:claude-code'.")]
    pub source: String,

    /// Scope of the thought. Defaults to `"global"` when omitted.
    #[schemars(description = "Scope label. Optional; defaults to 'global'. Convention is dotted ('work.tcgplayer').")]
    pub scope: Option<String>,

    /// Free-form metadata object. Defaults to `{}`. Recommended keys:
    /// `client_name`, `session_id`, `tool_name`, `agent_role`.
    #[schemars(description = "Optional free-form metadata object. Recommended keys: client_name, session_id, tool_name, agent_role.")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct EngramServer {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
}

impl EngramServer {
    pub fn new(pool: PgPool, embedder: Arc<dyn Embedder>) -> Self {
        Self { pool, embedder }
    }
}

impl std::fmt::Debug for EngramServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramServer")
            .field("model_id", &self.embedder.model().id)
            .finish()
    }
}

#[tool(tool_box)]
impl EngramServer {
    #[tool(description = "Capture a thought into engram's persistent memory. Returns the thought_id and an embedding_status indicating whether vector search will immediately surface it ('indexed') or whether the embedding is deferred to a backfill ('pending').")]
    async fn capture(
        &self,
        #[tool(aggr)] args: CaptureArgs,
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

        let resp = capture::capture(&self.pool, self.embedder.as_ref(), request)
            .await
            .map_err(map_capture_error)?;

        let body = serde_json::json!({
            "thought_id": resp.thought_id.to_string(),
            "embedding_status": resp.embedding_status,
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

#[tool(tool_box)]
impl ServerHandler for EngramServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Engram — self-hosted MCP-native memory service. Use `capture` to record \
                 a thought. Search tools (`search_thoughts`, `recent_thoughts`, `get_thought`) \
                 land in subsequent commits."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

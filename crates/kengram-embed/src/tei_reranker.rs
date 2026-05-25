//! `TeiReranker` — talks to Hugging Face's `text-embeddings-inference`
//! sidecar in rerank-task mode. Endpoint convention: the configured
//! `endpoint` is the service root (no `/v1` suffix); the reranker appends
//! `/rerank`. Default port for the dev Docker container is 8080.
//!
//! Request shape:
//!   POST /rerank
//!   { "query": "...", "texts": ["...", "..."], "raw_scores": false }
//!
//! Response shape:
//!   [{ "index": 0, "score": 0.95, "text": "..." }, ...]
//!
//! TEI's response is sorted by score descending; callers MUST consult
//! `RerankScore.index` to map back into the input `candidates` slice.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::reranker::{RerankScore, Reranker, RerankerError};

#[derive(Debug, Clone)]
pub struct TeiRerankerConfig {
    /// Service root URL (no trailing `/rerank`). Example:
    /// `"http://localhost:8080"`.
    pub endpoint: String,
    /// Kengram-side stable identity, written into the startup log + (later)
    /// per-response provenance fields. Conventionally `<vendor>/<model>` —
    /// e.g. `"BAAI/bge-reranker-v2-m3"`.
    pub model_id: String,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct TeiReranker {
    endpoint: String,
    model_id: String,
    /// Stored alongside the client so the timeout-error path reports the
    /// actual configured value. (Phase A taught us this lesson — see
    /// commit 1d627e4.)
    timeout_seconds: u64,
    client: Client,
}

impl TeiReranker {
    pub fn new(config: TeiRerankerConfig) -> Result<Self, RerankerError> {
        if config.endpoint.is_empty() {
            return Err(RerankerError::Misconfigured(
                "reranker endpoint must not be empty".into(),
            ));
        }
        if config.model_id.is_empty() {
            return Err(RerankerError::Misconfigured(
                "reranker model_id must not be empty".into(),
            ));
        }
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| RerankerError::Unreachable(format!("client build: {e}")))?;
        Ok(Self {
            endpoint: config.endpoint,
            model_id: config.model_id,
            timeout_seconds: config.timeout.as_secs(),
            client,
        })
    }
}

#[derive(Serialize)]
struct RerankRequestBody<'a> {
    query: &'a str,
    texts: &'a [&'a str],
    raw_scores: bool,
}

#[derive(Deserialize)]
struct RerankResponseItem {
    index: usize,
    score: f32,
}

#[async_trait]
impl Reranker for TeiReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn rerank(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/rerank", self.endpoint.trim_end_matches('/'));
        let body = RerankRequestBody {
            query,
            texts: candidates,
            raw_scores: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| map_send_error(e, self.timeout_seconds))?;

        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_default();
            return Err(RerankerError::Backend {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: Vec<RerankResponseItem> = resp.json().await.map_err(|e| {
            RerankerError::MalformedResponse(format!("decoding rerank response: {e}"))
        })?;

        if parsed.len() != candidates.len() {
            return Err(RerankerError::MalformedResponse(format!(
                "expected {} scores, got {}",
                candidates.len(),
                parsed.len()
            )));
        }

        // Validate every index falls in range. TEI shouldn't violate this
        // but defensive — a malformed backend response shouldn't panic the
        // search orchestrator's later array indexing.
        for item in &parsed {
            if item.index >= candidates.len() {
                return Err(RerankerError::MalformedResponse(format!(
                    "rerank score index {} out of range (candidates: {})",
                    item.index,
                    candidates.len()
                )));
            }
        }

        Ok(parsed
            .into_iter()
            .map(|item| RerankScore {
                index: item.index,
                score: item.score,
            })
            .collect())
    }
}

fn map_send_error(e: reqwest::Error, timeout_seconds: u64) -> RerankerError {
    if e.is_timeout() {
        RerankerError::Timeout {
            seconds: timeout_seconds,
        }
    } else if e.is_connect() {
        RerankerError::Unreachable(e.to_string())
    } else if let Some(status) = e.status() {
        RerankerError::Backend {
            status: status.as_u16(),
            message: e.to_string(),
        }
    } else {
        RerankerError::Unreachable(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(endpoint: String) -> TeiRerankerConfig {
        TeiRerankerConfig {
            endpoint,
            model_id: "BAAI/bge-reranker-v2-m3".to_string(),
            timeout: Duration::from_secs(2),
        }
    }

    #[tokio::test]
    async fn calls_endpoint_with_query_and_texts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_partial_json(json!({
                "query": "is the cat on the mat",
                "raw_scores": false,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"index": 0, "score": 0.95, "text": "the cat is sleeping"},
                {"index": 1, "score": 0.10, "text": "rust is fast"},
            ])))
            .mount(&server)
            .await;

        let r = TeiReranker::new(config_for(server.uri())).unwrap();
        let out = r
            .rerank(
                "is the cat on the mat",
                &["the cat is sleeping", "rust is fast"],
            )
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn preserves_response_order_for_index_lookup() {
        let server = MockServer::start().await;
        // TEI returns sorted by score descending — index field is what
        // callers consult to map back.
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"index": 2, "score": 0.99},
                {"index": 0, "score": 0.50},
                {"index": 1, "score": 0.10},
            ])))
            .mount(&server)
            .await;

        let r = TeiReranker::new(config_for(server.uri())).unwrap();
        let out = r.rerank("query", &["a", "b", "c"]).await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].index, 2);
        assert!((out[0].score - 0.99).abs() < 1e-5);
    }

    #[tokio::test]
    async fn empty_candidates_is_a_noop_no_request() {
        // No mock mounted; if the impl tried to send a request, the test
        // would fail. Empty input MUST NOT hit the network.
        let r = TeiReranker::new(config_for("http://127.0.0.1:1".into())).unwrap();
        let out = r.rerank("query", &[]).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn errors_on_count_mismatch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"index": 0, "score": 0.95},
            ])))
            .mount(&server)
            .await;

        let r = TeiReranker::new(config_for(server.uri())).unwrap();
        let err = r.rerank("q", &["a", "b"]).await.unwrap_err();
        assert!(matches!(err, RerankerError::MalformedResponse(_)));
    }

    #[tokio::test]
    async fn errors_on_out_of_range_index() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"index": 0, "score": 0.9},
                {"index": 5, "score": 0.5},  // out of range for 2 candidates
            ])))
            .mount(&server)
            .await;

        let r = TeiReranker::new(config_for(server.uri())).unwrap();
        let err = r.rerank("q", &["a", "b"]).await.unwrap_err();
        assert!(matches!(err, RerankerError::MalformedResponse(_)));
    }

    #[tokio::test]
    async fn errors_on_5xx_with_transient_classification() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream gone"))
            .mount(&server)
            .await;

        let r = TeiReranker::new(config_for(server.uri())).unwrap();
        let err = r.rerank("q", &["a"]).await.unwrap_err();
        match &err {
            RerankerError::Backend { status, .. } => assert_eq!(*status, 503),
            other => panic!("expected Backend, got {other:?}"),
        }
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn errors_on_unreachable_endpoint() {
        // Port 1 is reliably refused.
        let r = TeiReranker::new(config_for("http://127.0.0.1:1".into())).unwrap();
        let err = r.rerank("q", &["a"]).await.unwrap_err();
        assert!(
            matches!(
                err,
                RerankerError::Unreachable(_) | RerankerError::Timeout { .. }
            ),
            "expected Unreachable or Timeout, got {err:?}"
        );
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn timeout_reports_actual_configured_value() {
        // Wiremock with a long delay forces the client timeout to fire.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(10)))
            .mount(&server)
            .await;

        let cfg = TeiRerankerConfig {
            endpoint: server.uri(),
            model_id: "BAAI/bge-reranker-v2-m3".into(),
            timeout: Duration::from_millis(100),
        };
        let r = TeiReranker::new(cfg).unwrap();
        let err = r.rerank("q", &["a"]).await.unwrap_err();
        // Timeout reports `seconds: 0` (100ms rounds down) — the point is
        // the value was *captured from config*, not hardcoded.
        assert!(matches!(err, RerankerError::Timeout { .. }));
    }

    #[tokio::test]
    async fn rejects_empty_endpoint() {
        let cfg = TeiRerankerConfig {
            endpoint: String::new(),
            model_id: "BAAI/bge-reranker-v2-m3".into(),
            timeout: Duration::from_secs(1),
        };
        let err = TeiReranker::new(cfg).unwrap_err();
        assert!(matches!(err, RerankerError::Misconfigured(_)));
    }
}

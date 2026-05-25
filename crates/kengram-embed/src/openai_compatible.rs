//! `OpenAICompatibleEmbedder` — talks to any backend that implements the
//! OpenAI `/v1/embeddings` API shape. Covers Ollama (dev), TEI (production
//! sidecar), OpenAI (cloud), and Voyage AI (cloud) by config alone.
//!
//! Endpoint convention: the configured `endpoint` is the `/v1` base, and the
//! embedder appends `/embeddings`. For Ollama running locally, that is
//! `http://localhost:11434/v1`.

use async_trait::async_trait;
use kengram_core::{Embedder, EmbedderError, EmbeddingModel};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OpenAICompatibleEmbedder {
    endpoint: String,
    model_name: String,
    model: EmbeddingModel,
    api_key: Option<String>,
    client: Client,
}

#[derive(Debug, Clone)]
pub struct OpenAICompatibleConfig {
    /// Base URL ending in `/v1`. Example: `"http://localhost:11434/v1"`.
    pub endpoint: String,
    /// The model name as the backend understands it. For Ollama: `"bge-m3"`.
    /// For OpenAI: `"text-embedding-3-small"`.
    pub model_name: String,
    /// The kengram-side `EmbeddingModel` identity (`"bge-m3:1024"` is the
    /// default for M1). Used for dimension validation and to identify which
    /// HNSW partial index the resulting vectors should land in.
    pub model: EmbeddingModel,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

impl OpenAICompatibleConfig {
    /// Defaults for the local Ollama dev path: `bge-m3:1024` at port 11434, 5s timeout, no key.
    pub fn ollama_local() -> Self {
        Self {
            endpoint: "http://localhost:11434/v1".to_string(),
            model_name: "bge-m3".to_string(),
            model: EmbeddingModel::bge_m3(),
            api_key: None,
            timeout: Duration::from_secs(5),
        }
    }
}

impl OpenAICompatibleEmbedder {
    pub fn new(config: OpenAICompatibleConfig) -> Result<Self, EmbedderError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbedderError::Unreachable(format!("client build: {e}")))?;
        Ok(Self {
            endpoint: config.endpoint,
            model_name: config.model_name,
            model: config.model,
            api_key: config.api_key,
            client,
        })
    }
}

#[derive(Serialize)]
struct EmbeddingRequestBody<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponseBody {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for OpenAICompatibleEmbedder {
    fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Err(EmbedderError::EmptyBatch);
        }

        let url = format!("{}/embeddings", self.endpoint.trim_end_matches('/'));
        let body = EmbeddingRequestBody {
            model: &self.model_name,
            input: texts,
        };

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await.map_err(map_send_error)?;

        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_default();
            return Err(EmbedderError::Backend {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: EmbeddingResponseBody = resp.json().await.map_err(|e| {
            EmbedderError::MalformedResponse(format!("decoding embeddings response: {e}"))
        })?;

        if parsed.data.len() != texts.len() {
            return Err(EmbedderError::MalformedResponse(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                parsed.data.len()
            )));
        }

        let expected_dims = self.model.dimensions;
        for (i, datum) in parsed.data.iter().enumerate() {
            if datum.embedding.len() != expected_dims {
                return Err(EmbedderError::DimensionMismatch {
                    expected: expected_dims,
                    got: datum.embedding.len(),
                });
            }
            let _ = i; // silence unused (kept for future logging)
        }

        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

fn map_send_error(e: reqwest::Error) -> EmbedderError {
    if e.is_timeout() {
        EmbedderError::Timeout { seconds: 5 }
    } else if e.is_connect() {
        EmbedderError::Unreachable(e.to_string())
    } else if let Some(status) = e.status() {
        EmbedderError::Backend {
            status: status.as_u16(),
            message: e.to_string(),
        }
    } else {
        EmbedderError::Unreachable(e.to_string())
    }
}

// silence dead-code warning for StatusCode import only used in tests
const _: Option<StatusCode> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(endpoint: String, dims: usize) -> OpenAICompatibleConfig {
        OpenAICompatibleConfig {
            endpoint,
            model_name: "bge-m3".to_string(),
            model: EmbeddingModel::new("bge-m3:test", dims),
            api_key: None,
            timeout: Duration::from_secs(2),
        }
    }

    fn good_response(dims: usize, count: usize) -> serde_json::Value {
        let data: Vec<_> = (0..count)
            .map(|i| {
                let v: Vec<f32> = (0..dims).map(|j| (i + j) as f32 * 0.001).collect();
                json!({"object": "embedding", "index": i, "embedding": v})
            })
            .collect();
        json!({"object": "list", "data": data, "model": "bge-m3"})
    }

    #[tokio::test]
    async fn posts_correct_json_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(body_partial_json(json!({"model": "bge-m3"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(good_response(8, 1)))
            .mount(&server)
            .await;

        let e =
            OpenAICompatibleEmbedder::new(config_for(format!("{}/v1", server.uri()), 8)).unwrap();
        e.embed(&["hello".to_string()]).await.unwrap();
        // If the mock matched, the request body had the expected shape.
    }

    #[tokio::test]
    async fn parses_valid_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(good_response(8, 2)))
            .mount(&server)
            .await;

        let e =
            OpenAICompatibleEmbedder::new(config_for(format!("{}/v1", server.uri()), 8)).unwrap();
        let out = e.embed(&["a".to_string(), "b".to_string()]).await.unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 8);
        assert_eq!(out[1].len(), 8);
    }

    #[tokio::test]
    async fn errors_on_dimension_mismatch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(good_response(4, 1)))
            .mount(&server)
            .await;

        // We expect 8 dims but the server returns 4.
        let e =
            OpenAICompatibleEmbedder::new(config_for(format!("{}/v1", server.uri()), 8)).unwrap();
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        assert!(matches!(
            err,
            EmbedderError::DimensionMismatch {
                expected: 8,
                got: 4
            }
        ));
    }

    #[tokio::test]
    async fn errors_on_malformed_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let e =
            OpenAICompatibleEmbedder::new(config_for(format!("{}/v1", server.uri()), 8)).unwrap();
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        assert!(matches!(err, EmbedderError::MalformedResponse(_)));
    }

    #[tokio::test]
    async fn errors_on_http_5xx() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream gone"))
            .mount(&server)
            .await;

        let e =
            OpenAICompatibleEmbedder::new(config_for(format!("{}/v1", server.uri()), 8)).unwrap();
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        match err {
            EmbedderError::Backend { status, .. } => assert_eq!(status, 503),
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn errors_on_http_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorised"))
            .mount(&server)
            .await;

        let e =
            OpenAICompatibleEmbedder::new(config_for(format!("{}/v1", server.uri()), 8)).unwrap();
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        match err {
            EmbedderError::Backend { status, .. } => assert_eq!(status, 401),
            other => panic!("expected Backend error, got {other:?}"),
        }
        assert!(!err.is_transient());
    }

    #[tokio::test]
    async fn errors_on_unreachable_endpoint() {
        // Port 1 is reliably refused on macOS and Linux.
        let e = OpenAICompatibleEmbedder::new(config_for("http://127.0.0.1:1/v1".to_string(), 8))
            .unwrap();
        let err = e.embed(&["hello".to_string()]).await.unwrap_err();
        assert!(
            matches!(
                err,
                EmbedderError::Unreachable(_) | EmbedderError::Timeout { .. }
            ),
            "expected Unreachable or Timeout, got {err:?}"
        );
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn empty_batch_errors_before_http_call() {
        let e = OpenAICompatibleEmbedder::new(config_for("http://127.0.0.1:1/v1".to_string(), 8))
            .unwrap();
        assert!(matches!(e.embed(&[]).await, Err(EmbedderError::EmptyBatch)));
    }

    /// Hits a real Ollama at localhost:11434 with `bge-m3` pulled. Gated on
    /// the `integration` feature. Run with `cargo test -p kengram-embed
    /// --features integration -- live_ollama`.
    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn live_ollama_returns_1024_dim_vector() {
        let cfg = OpenAICompatibleConfig::ollama_local();
        let dims = cfg.model.dimensions;
        let e = OpenAICompatibleEmbedder::new(cfg).unwrap();
        let out = e
            .embed(&["the quick brown fox".to_string()])
            .await
            .expect("ollama unreachable — is the daemon up and 'bge-m3' pulled?");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), dims);
        // Sanity: the vector shouldn't be all zeros.
        assert!(out[0].iter().any(|x| x.abs() > 1e-6));
    }
}

//! Gemini embeddings client.
//!
//! Uses `batchEmbedContents` so the document backfill can send
//! `RETRIEVAL_DOCUMENT` while query search sends `RETRIEVAL_QUERY`.

use async_trait::async_trait;
use kengram_core::{Embedder, EmbedderError, EmbeddingModel};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const MAX_RATE_LIMIT_RETRIES: usize = 8;
const RATE_LIMIT_FALLBACK_DELAY: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct GeminiEmbedder {
    endpoint: String,
    api_model: String,
    model: EmbeddingModel,
    api_key: String,
    client: Client,
}

#[derive(Debug, Clone)]
pub struct GeminiConfig {
    /// API base URL, default `"https://generativelanguage.googleapis.com/v1beta"`.
    pub endpoint: String,
    /// Gemini model resource, e.g. `"models/gemini-embedding-001"`.
    pub api_model: String,
    /// Kengram-side model identity, e.g. `"gemini-embedding-001"`.
    pub model: EmbeddingModel,
    pub api_key: String,
    pub timeout: Duration,
}

impl GeminiConfig {
    pub fn gemini_embedding_001(api_key: String) -> Self {
        Self {
            endpoint: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            api_model: "models/gemini-embedding-001".to_string(),
            model: EmbeddingModel::new("gemini-embedding-001", 3072),
            api_key,
            timeout: Duration::from_secs(60),
        }
    }
}

impl GeminiEmbedder {
    pub fn new(config: GeminiConfig) -> Result<Self, EmbedderError> {
        if config.api_key.trim().is_empty() {
            return Err(EmbedderError::Unreachable(
                "Gemini API key is empty".to_string(),
            ));
        }
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbedderError::Unreachable(format!("client build: {e}")))?;
        Ok(Self {
            endpoint: config.endpoint,
            api_model: config.api_model,
            model: config.model,
            api_key: config.api_key,
            client,
        })
    }

    async fn embed_with_task_type(
        &self,
        texts: &[String],
        task_type: GeminiTaskType,
    ) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Err(EmbedderError::EmptyBatch);
        }

        let mut url = reqwest::Url::parse(&format!(
            "{}/{}:batchEmbedContents",
            self.endpoint.trim_end_matches('/'),
            self.api_model.trim_start_matches('/')
        ))
        .map_err(|e| EmbedderError::MalformedResponse(format!("invalid Gemini URL: {e}")))?;
        url.query_pairs_mut().append_pair("key", &self.api_key);
        let body = BatchEmbedRequest {
            requests: texts
                .iter()
                .map(|text| EmbedContentRequest {
                    model: &self.api_model,
                    content: GeminiContent {
                        parts: vec![GeminiPart { text }],
                    },
                    task_type,
                })
                .collect(),
        };

        let mut attempt = 0usize;
        let resp = loop {
            let resp = self
                .client
                .post(url.clone())
                .json(&body)
                .send()
                .await
                .map_err(map_send_error)?;

            let status = resp.status();
            if status.is_success() {
                break resp;
            }

            let message = resp.text().await.unwrap_or_default();
            if status == StatusCode::TOO_MANY_REQUESTS && attempt < MAX_RATE_LIMIT_RETRIES {
                attempt += 1;
                let delay =
                    retry_delay_from_gemini_error(&message).unwrap_or(RATE_LIMIT_FALLBACK_DELAY);
                let quota = gemini_quota_metric(&message).unwrap_or_else(|| "unknown".to_string());
                tracing::warn!(
                    model_id = %self.model.id,
                    quota_metric = %quota,
                    retry_delay_ms = delay.as_millis(),
                    attempt,
                    max_attempts = MAX_RATE_LIMIT_RETRIES,
                    "Gemini embedding rate limited; honoring retry delay"
                );
                tokio::time::sleep(delay).await;
                continue;
            }

            return Err(EmbedderError::Backend {
                status: status.as_u16(),
                message,
            });
        };

        let parsed: BatchEmbedResponse = resp.json().await.map_err(|e| {
            EmbedderError::MalformedResponse(format!("decoding Gemini embeddings response: {e}"))
        })?;

        if parsed.embeddings.len() != texts.len() {
            return Err(EmbedderError::MalformedResponse(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                parsed.embeddings.len()
            )));
        }

        let expected = self.model.dimensions;
        for embedding in &parsed.embeddings {
            if embedding.values.len() != expected {
                return Err(EmbedderError::DimensionMismatch {
                    expected,
                    got: embedding.values.len(),
                });
            }
        }

        Ok(parsed
            .embeddings
            .into_iter()
            .map(|embedding| embedding.values)
            .collect())
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
enum GeminiTaskType {
    #[serde(rename = "RETRIEVAL_DOCUMENT")]
    RetrievalDocument,
    #[serde(rename = "RETRIEVAL_QUERY")]
    RetrievalQuery,
}

#[derive(Serialize)]
struct BatchEmbedRequest<'a> {
    requests: Vec<EmbedContentRequest<'a>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest<'a> {
    model: &'a str,
    content: GeminiContent<'a>,
    task_type: GeminiTaskType,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    parts: Vec<GeminiPart<'a>>,
}

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<GeminiEmbedding>,
}

#[derive(Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

#[async_trait]
impl Embedder for GeminiEmbedder {
    fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.embed_documents(texts).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.embed_with_task_type(texts, GeminiTaskType::RetrievalDocument)
            .await
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedderError> {
        let vectors = self
            .embed_with_task_type(&[query.to_string()], GeminiTaskType::RetrievalQuery)
            .await?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::MalformedResponse("missing query embedding".to_string()))
    }
}

fn map_send_error(e: reqwest::Error) -> EmbedderError {
    if e.is_timeout() {
        EmbedderError::Timeout { seconds: 60 }
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

fn retry_delay_from_gemini_error(message: &str) -> Option<Duration> {
    let value: serde_json::Value = serde_json::from_str(message).ok()?;
    let details = value.pointer("/error/details")?.as_array()?;
    details.iter().find_map(|detail| {
        detail
            .get("retryDelay")
            .and_then(|value| value.as_str())
            .and_then(parse_google_duration)
            .map(|delay| delay + Duration::from_secs(2))
    })
}

fn gemini_quota_metric(message: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(message).ok()?;
    let details = value.pointer("/error/details")?.as_array()?;
    details
        .iter()
        .filter_map(|detail| detail.get("violations")?.as_array())
        .flat_map(|violations| violations.iter())
        .find_map(|violation| violation.get("quotaMetric")?.as_str())
        .map(str::to_string)
}

fn parse_google_duration(value: &str) -> Option<Duration> {
    let seconds = value.strip_suffix('s')?.parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }
    Some(Duration::from_secs_f64(seconds.min(300.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(endpoint: String, dims: usize) -> GeminiConfig {
        GeminiConfig {
            endpoint,
            api_model: "models/gemini-embedding-001".to_string(),
            model: EmbeddingModel::new("gemini-embedding-001", dims),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(2),
        }
    }

    fn good_response(dims: usize, count: usize) -> serde_json::Value {
        let embeddings: Vec<_> = (0..count)
            .map(|i| {
                let values: Vec<f32> = (0..dims).map(|j| (i + j) as f32 * 0.001).collect();
                json!({"values": values})
            })
            .collect();
        json!({"embeddings": embeddings})
    }

    #[tokio::test]
    async fn document_embeddings_use_document_task_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-embedding-001:batchEmbedContents",
            ))
            .and(query_param("key", "test-key"))
            .and(body_partial_json(json!({
                "requests": [{
                    "model": "models/gemini-embedding-001",
                    "taskType": "RETRIEVAL_DOCUMENT"
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(good_response(8, 1)))
            .mount(&server)
            .await;

        let e = GeminiEmbedder::new(config_for(format!("{}/v1beta", server.uri()), 8)).unwrap();
        let out = e.embed_documents(&["body".to_string()]).await.unwrap();
        assert_eq!(out[0].len(), 8);
    }

    #[tokio::test]
    async fn query_embeddings_use_query_task_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-embedding-001:batchEmbedContents",
            ))
            .and(query_param("key", "test-key"))
            .and(body_partial_json(json!({
                "requests": [{
                    "taskType": "RETRIEVAL_QUERY"
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(good_response(8, 1)))
            .mount(&server)
            .await;

        let e = GeminiEmbedder::new(config_for(format!("{}/v1beta", server.uri()), 8)).unwrap();
        let out = e.embed_query("question").await.unwrap();
        assert_eq!(out.len(), 8);
    }

    #[tokio::test]
    async fn errors_on_dimension_mismatch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-embedding-001:batchEmbedContents",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(good_response(4, 1)))
            .mount(&server)
            .await;

        let e = GeminiEmbedder::new(config_for(format!("{}/v1beta", server.uri()), 8)).unwrap();
        let err = e.embed_documents(&["body".to_string()]).await.unwrap_err();
        assert!(matches!(
            err,
            EmbedderError::DimensionMismatch {
                expected: 8,
                got: 4
            }
        ));
    }

    #[test]
    fn parses_gemini_retry_info_delay() {
        let message = json!({
            "error": {
                "code": 429,
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "39s"
                }, {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{
                        "quotaMetric": "generativelanguage.googleapis.com/embed_content_paid_tier_requests"
                    }]
                }]
            }
        })
        .to_string();

        assert_eq!(
            retry_delay_from_gemini_error(&message),
            Some(Duration::from_secs(41))
        );
        assert_eq!(
            gemini_quota_metric(&message).as_deref(),
            Some("generativelanguage.googleapis.com/embed_content_paid_tier_requests")
        );
    }
}

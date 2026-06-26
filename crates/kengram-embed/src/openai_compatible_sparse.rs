//! OpenAI-compatible sparse lexical embedder for BGE-M3 lexical weights.
//!
//! This is intentionally separate from the dense OpenAI-compatible embedder:
//! many `/v1/embeddings` providers return dense vectors only. Sparse serving
//! must fail closed when the configured backend does not emit lexical weights.

use async_trait::async_trait;
use kengram_core::{
    EmbedderError, SparseEmbedder, SparseEmbeddingModel, SparseLexicalVector, SparseWeight,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OpenAICompatibleSparseEmbedder {
    endpoint: String,
    model_name: String,
    model: SparseEmbeddingModel,
    api_key: Option<String>,
    client: Client,
}

#[derive(Debug, Clone)]
pub struct OpenAICompatibleSparseConfig {
    /// Base URL ending in `/v1`.
    pub endpoint: String,
    /// Backend model name as the provider understands it.
    pub model_name: String,
    /// Kengram-side sparse model identity, normally `bge-m3:sparse`.
    pub model: SparseEmbeddingModel,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

impl OpenAICompatibleSparseEmbedder {
    pub fn new(config: OpenAICompatibleSparseConfig) -> Result<Self, EmbedderError> {
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
struct SparseEmbeddingResponseBody {
    data: Vec<SparseEmbeddingDatum>,
}

#[derive(Deserialize)]
struct SparseEmbeddingDatum {
    #[serde(default)]
    lexical_weights: Option<Value>,
    #[serde(default)]
    sparse_embedding: Option<Value>,
}

#[async_trait]
impl SparseEmbedder for OpenAICompatibleSparseEmbedder {
    fn sparse_model(&self) -> &SparseEmbeddingModel {
        &self.model
    }

    async fn encode_sparse(
        &self,
        texts: &[String],
    ) -> Result<Vec<SparseLexicalVector>, EmbedderError> {
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

        let parsed: SparseEmbeddingResponseBody = resp.json().await.map_err(|e| {
            EmbedderError::MalformedResponse(format!("decoding sparse embeddings response: {e}"))
        })?;

        if parsed.data.len() != texts.len() {
            return Err(EmbedderError::MalformedResponse(format!(
                "expected {} sparse embeddings, got {}",
                texts.len(),
                parsed.data.len()
            )));
        }

        parsed
            .data
            .into_iter()
            .map(|datum| {
                let weights_value = datum
                    .lexical_weights
                    .or(datum.sparse_embedding)
                    .ok_or_else(|| {
                        EmbedderError::MalformedResponse(
                            "sparse response missing lexical_weights".to_string(),
                        )
                    })?;
                let weights = parse_sparse_weights(&weights_value)?;
                SparseLexicalVector::new(self.model.clone(), weights).map_err(|e| {
                    EmbedderError::MalformedResponse(format!("invalid sparse vector: {e}"))
                })
            })
            .collect()
    }
}

fn parse_sparse_weights(value: &Value) -> Result<Vec<SparseWeight>, EmbedderError> {
    match value {
        Value::Object(map) => map
            .iter()
            .map(|(token_id, weight)| {
                let token_id = token_id.parse::<u32>().map_err(|e| {
                    EmbedderError::MalformedResponse(format!(
                        "sparse lexical_weights key {token_id:?} is not a token id: {e}"
                    ))
                })?;
                let weight = value_to_f32(weight, "lexical_weights value")?;
                Ok(SparseWeight::new(token_id, weight))
            })
            .collect(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                let obj = item.as_object().ok_or_else(|| {
                    EmbedderError::MalformedResponse(
                        "sparse lexical_weights array entries must be objects".to_string(),
                    )
                })?;
                let token_value = obj
                    .get("token_id")
                    .or_else(|| obj.get("token"))
                    .or_else(|| obj.get("id"))
                    .or_else(|| obj.get("index"))
                    .ok_or_else(|| {
                        EmbedderError::MalformedResponse(
                            "sparse lexical_weights entry missing token_id".to_string(),
                        )
                    })?;
                let weight_value =
                    obj.get("weight")
                        .or_else(|| obj.get("value"))
                        .ok_or_else(|| {
                            EmbedderError::MalformedResponse(
                                "sparse lexical_weights entry missing weight".to_string(),
                            )
                        })?;
                let token_id = value_to_u32(token_value, "token_id")?;
                let weight = value_to_f32(weight_value, "weight")?;
                Ok(SparseWeight::new(token_id, weight))
            })
            .collect(),
        _ => Err(EmbedderError::MalformedResponse(
            "sparse lexical_weights must be an object or array".to_string(),
        )),
    }
}

fn value_to_u32(value: &Value, label: &str) -> Result<u32, EmbedderError> {
    if let Some(n) = value.as_u64() {
        return u32::try_from(n).map_err(|e| {
            EmbedderError::MalformedResponse(format!("{label} {n} does not fit u32: {e}"))
        });
    }
    if let Some(s) = value.as_str() {
        return s.parse::<u32>().map_err(|e| {
            EmbedderError::MalformedResponse(format!("{label} {s:?} is not a token id: {e}"))
        });
    }
    Err(EmbedderError::MalformedResponse(format!(
        "{label} must be a non-negative integer or string"
    )))
}

fn value_to_f32(value: &Value, label: &str) -> Result<f32, EmbedderError> {
    let Some(n) = value.as_f64() else {
        return Err(EmbedderError::MalformedResponse(format!(
            "{label} must be numeric"
        )));
    };
    let f = n as f32;
    if !f.is_finite() {
        return Err(EmbedderError::MalformedResponse(format!(
            "{label} is not finite"
        )));
    }
    Ok(f)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(endpoint: String) -> OpenAICompatibleSparseConfig {
        OpenAICompatibleSparseConfig {
            endpoint,
            model_name: "bge-m3".to_string(),
            model: SparseEmbeddingModel::bge_m3_sparse(),
            api_key: None,
            timeout: Duration::from_secs(2),
        }
    }

    #[tokio::test]
    async fn parses_lexical_weights_object() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(body_partial_json(json!({"model": "bge-m3"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.0],
                    "lexical_weights": {"5": 0.25, "8": 1.5}
                }]
            })))
            .mount(&server)
            .await;

        let e = OpenAICompatibleSparseEmbedder::new(config_for(format!("{}/v1", server.uri())))
            .unwrap();
        let out = e.encode_sparse(&["hello".to_string()]).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].weights().len(), 2);
        assert_eq!(out[0].model.id, "bge-m3:sparse");
    }

    #[tokio::test]
    async fn fails_closed_when_backend_omits_lexical_weights() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.0]
                }]
            })))
            .mount(&server)
            .await;

        let e = OpenAICompatibleSparseEmbedder::new(config_for(format!("{}/v1", server.uri())))
            .unwrap();
        let err = e.encode_sparse(&["hello".to_string()]).await.unwrap_err();
        assert!(matches!(err, EmbedderError::MalformedResponse(_)));
    }
}

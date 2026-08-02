//! Stable-source supersession transaction caller + disposable-PG acceptance.
//! Spec: kengram-supersession-transactional-capability r1+r2.
//! No plaintext Debug on request content.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Clone)]
pub struct SupersessionRequest {
    pub request_id: Uuid,
    pub namespace: String,
    pub source_ref: String,
    pub expected_status: String,
    pub expected_old_payload_hash: String,
    pub expected_old_thought_id: Option<Uuid>,
    pub new_payload_canonical_json: String,
    pub new_payload_hash: String,
    pub new_content: String,
    pub new_metadata: Value,
    pub embedding_model_id: String,
    pub tagger_model_id: String,
    pub actor: String,
    pub lane: String,
    pub approval_ref: String,
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SupersessionError {
    #[error("database error")]
    Db(#[from] sqlx::Error),
    #[error("sqlstate {code}: {message}")]
    SqlState { code: String, message: String },
    #[error("invalid response")]
    InvalidResponse,
}

/// Execute exactly one SQL call — no multi-step mutation.
pub async fn supersede_argus_source_event(
    pool: &PgPool,
    req: &SupersessionRequest,
) -> Result<Value, SupersessionError> {
    let row: (Value,) = sqlx::query_as(
        r#"
        SELECT public.supersede_argus_source_event(
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16
        )
        "#,
    )
    .bind(req.request_id)
    .bind(&req.namespace)
    .bind(&req.source_ref)
    .bind(&req.expected_status)
    .bind(&req.expected_old_payload_hash)
    .bind(req.expected_old_thought_id)
    .bind(&req.new_payload_canonical_json)
    .bind(&req.new_payload_hash)
    .bind(&req.new_content)
    .bind(&req.new_metadata)
    .bind(&req.embedding_model_id)
    .bind(&req.tagger_model_id)
    .bind(&req.actor)
    .bind(&req.lane)
    .bind(&req.approval_ref)
    .bind(&req.reason)
    .fetch_one(pool)
    .await
    .map_err(map_sqlx)?;
    Ok(row.0)
}

fn map_sqlx(err: sqlx::Error) -> SupersessionError {
    if let sqlx::Error::Database(db) = &err {
        if let Some(code) = db.code() {
            return SupersessionError::SqlState {
                code: code.to_string(),
                message: db.message().to_string(),
            };
        }
    }
    SupersessionError::Db(err)
}

/// Independent oracle — must not call producer helpers.
pub fn independent_request_digest_hex(req: &SupersessionRequest) -> (String, String) {
    let envelope = json!({
        "v": 1,
        "request_id": req.request_id.to_string(),
        "namespace": req.namespace,
        "source_ref": req.source_ref,
        "expected_status": req.expected_status,
        "expected_old_payload_hash": req.expected_old_payload_hash,
        "expected_old_thought_id": req.expected_old_thought_id.map(|u| u.to_string()),
        "new_payload_canonical_json": req.new_payload_canonical_json,
        "new_payload_hash": req.new_payload_hash,
        "new_content": req.new_content,
        "new_metadata": req.new_metadata,
        "embedding_model_id": req.embedding_model_id,
        "tagger_model_id": req.tagger_model_id,
        "actor": req.actor,
        "lane": req.lane,
        "approval_ref": req.approval_ref,
        "reason": req.reason,
    });
    // Note: Postgres jsonb_build_object text form may differ from serde; tests
    // that compare preimage use SQL-side construction. This helper is for
    // collision/adversarial framing checks in pure Rust where possible.
    let preimage = envelope.to_string();
    let digest = Sha256::digest(preimage.as_bytes());
    (preimage, to_hex(&digest))
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn payload_hash_hex(canonical_json: &str) -> String {
    to_hex(&Sha256::digest(canonical_json.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::postgres::PgPoolOptions;
    use std::time::Duration;

    fn require_database_url() -> String {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL required");
        assert!(
            !url.contains("kengram_prod") && !url.contains("prod"),
            "refusing production-looking DATABASE_URL"
        );
        url
    }

    async fn pool() -> PgPool {
        let url = require_database_url();
        PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(10))
            .connect(&url)
            .await
            .expect("connect")
    }

    #[tokio::test]
    async fn invalid_expected_status_raises_zsi01_zero_receipt() {
        let pool = pool().await;
        let before: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM public.argus_source_event_supersession_receipts",
        )
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let content = "corrected body for invalid status fixture";
        let canon = format!(r#"{{"text":"{content}"}}"#);
        let hash = payload_hash_hex(&canon);
        let old_hash = payload_hash_hex(r#"{"text":"old"}"#);
        let req = SupersessionRequest {
            request_id: Uuid::new_v4(),
            namespace: "test".into(),
            source_ref: "invalid-status-1".into(),
            expected_status: "stored".into(),
            expected_old_payload_hash: old_hash,
            expected_old_thought_id: Some(Uuid::new_v4()),
            new_payload_canonical_json: canon,
            new_payload_hash: hash,
            new_content: content.into(),
            new_metadata: json!({
                "namespace": "test",
                "source_ref": "invalid-status-1",
                "payload_sha256": payload_hash_hex(&format!(r#"{{"text":"{content}"}}"#)),
            }),
            embedding_model_id: "bge-m3:1024".into(),
            tagger_model_id: "test-tagger".into(),
            actor: "diesel".into(),
            lane: "724808".into(),
            approval_ref: "test".into(),
            reason: "invalid status fixture".into(),
        };
        // fix metadata hash to match
        let mut req = req;
        req.new_metadata = json!({
            "namespace": "test",
            "source_ref": "invalid-status-1",
            "payload_sha256": req.new_payload_hash,
        });

        // Connect as supersession role if possible
        let url = require_database_url();
        let res = supersede_as_role(&url, "kengram_rt_supersession", &req).await;
        match res {
            Err(SupersessionError::SqlState { code, message }) => {
                assert_eq!(code, "ZSI01", "message={message}");
            }
            other => panic!("expected ZSI01, got {other:?}"),
        }
        let after: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM public.argus_source_event_supersession_receipts",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(before, after, "invalid input must write zero receipts");
    }

    async fn supersede_as_role(
        base_url: &str,
        user: &str,
        req: &SupersessionRequest,
    ) -> Result<Value, SupersessionError> {
        // Rewrite user in URL for role tests when password matches fixture
        let url = if base_url.contains("://") {
            // postgres://kengram:kengram@host/db -> swap user
            let mut u = base_url.to_string();
            if let Some(at) = u.find('@') {
                if let Some(scheme_end) = u.find("://") {
                    let pass_part = &u[scheme_end + 3..at];
                    if let Some(colon) = pass_part.find(':') {
                        let pass = &pass_part[colon + 1..];
                        u = format!(
                            "{}{}:{}{}",
                            &u[..scheme_end + 3],
                            user,
                            pass,
                            &u[at..]
                        );
                    }
                }
            }
            u
        } else {
            base_url.to_string()
        };
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect(&url)
            .await
            .map_err(SupersessionError::Db)?;
        supersede_argus_source_event(&pool, req).await
    }

    #[tokio::test]
    async fn missing_stable_row_refused_expected_state_null_id() {
        let url = require_database_url();
        let content = "missing row correction body";
        let canon = format!(r#"{{"text":"{content}"}}"#);
        let hash = payload_hash_hex(&canon);
        let old_hash = payload_hash_hex(r#"{"text":"ghost-old"}"#);
        let req = SupersessionRequest {
            request_id: Uuid::new_v4(),
            namespace: "test".into(),
            source_ref: format!("missing-{}", Uuid::new_v4()),
            expected_status: "conflict".into(),
            expected_old_payload_hash: old_hash,
            expected_old_thought_id: Some(Uuid::new_v4()),
            new_payload_canonical_json: canon.clone(),
            new_payload_hash: hash.clone(),
            new_content: content.into(),
            new_metadata: json!({
                "namespace": "test",
                "source_ref": "placeholder",
                "payload_sha256": hash,
            }),
            embedding_model_id: "bge-m3:1024".into(),
            tagger_model_id: "test-tagger".into(),
            actor: "diesel".into(),
            lane: "724808".into(),
            approval_ref: "test".into(),
            reason: "missing row".into(),
        };
        let mut req = req;
        req.new_metadata = json!({
            "namespace": req.namespace,
            "source_ref": req.source_ref,
            "payload_sha256": req.new_payload_hash,
        });
        let out = supersede_as_role(&url, "kengram_rt_supersession", &req)
            .await
            .expect("missing row should refuse not error");
        assert_eq!(out["outcome"], "refused_expected_state");
        assert_eq!(out["observed_missing"], true);
        assert!(out.get("stable_source_event_id").unwrap().is_null());
        assert_eq!(out["replayed"], false);
        // second apply replay
        let out2 = supersede_as_role(&url, "kengram_rt_supersession", &req)
            .await
            .expect("replay");
        assert_eq!(out2["outcome"], "refused_expected_state");
        assert_eq!(out2["replayed"], true);
        assert_eq!(out2["receipt_hash"], out["receipt_hash"]);
    }

    #[tokio::test]
    async fn break_glass_kengram_gets_zsa01() {
        let url = require_database_url();
        let content = "break glass fixture";
        let canon = format!(r#"{{"text":"{content}"}}"#);
        let hash = payload_hash_hex(&canon);
        let req = SupersessionRequest {
            request_id: Uuid::new_v4(),
            namespace: "test".into(),
            source_ref: "zsa01".into(),
            expected_status: "conflict".into(),
            expected_old_payload_hash: payload_hash_hex(r#"{"text":"o"}"#),
            expected_old_thought_id: Some(Uuid::new_v4()),
            new_payload_canonical_json: canon,
            new_payload_hash: hash.clone(),
            new_content: content.into(),
            new_metadata: json!({
                "namespace": "test",
                "source_ref": "zsa01",
                "payload_sha256": hash,
            }),
            embedding_model_id: "bge-m3:1024".into(),
            tagger_model_id: "test-tagger".into(),
            actor: "diesel".into(),
            lane: "724808".into(),
            approval_ref: "test".into(),
            reason: "zsa01".into(),
        };
        let res = supersede_as_role(&url, "kengram", &req).await;
        match res {
            Err(SupersessionError::SqlState { code, .. }) => {
                assert_eq!(code, "ZSA01");
            }
            other => panic!("expected ZSA01 got {other:?}"),
        }
    }
}

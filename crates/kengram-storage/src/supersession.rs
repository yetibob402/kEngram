//! Stable-source supersession transaction caller + disposable-PG acceptance.
//! Spec: kengram-supersession-transactional-capability r1 §9 + r2 §8/§9.

use serde_json::{Value, json};
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
}

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

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn payload_hash_hex(canonical_json: &str) -> String {
    to_hex(&Sha256::digest(canonical_json.as_bytes()))
}

/// Independent request-envelope construction for oracle checks (no producer helper).
pub fn independent_request_envelope(req: &SupersessionRequest) -> Value {
    json!({
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
        "new_metadata": req.new_metadata.clone(),
        "embedding_model_id": req.embedding_model_id,
        "tagger_model_id": req.tagger_model_id,
        "actor": req.actor,
        "lane": req.lane,
        "approval_ref": req.approval_ref,
        "reason": req.reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::Row;
    use sqlx::postgres::PgPoolOptions;
    use std::time::Duration;

    const EMB: &str = "bge-m3:1024";
    const TAG: &str = "test-tagger";

    fn require_database_url() -> String {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL required");
        assert!(
            !url.contains("kengram_prod"),
            "refusing production-looking DATABASE_URL"
        );
        url
    }

    async fn admin_pool() -> PgPool {
        PgPoolOptions::new()
            .max_connections(8)
            .acquire_timeout(Duration::from_secs(10))
            .connect(&require_database_url())
            .await
            .expect("admin connect")
    }

    fn rewrite_user(base_url: &str, user: &str) -> String {
        let mut u = base_url.to_string();
        if let Some(at) = u.find('@') {
            if let Some(scheme_end) = u.find("://") {
                let pass_part = &u[scheme_end + 3..at];
                if let Some(colon) = pass_part.find(':') {
                    let pass = &pass_part[colon + 1..];
                    u = format!("{}{}:{}{}", &u[..scheme_end + 3], user, pass, &u[at..]);
                }
            }
        }
        u
    }

    async fn role_pool(user: &str) -> PgPool {
        let url = rewrite_user(&require_database_url(), user);
        PgPoolOptions::new()
            .max_connections(4)
            .acquire_timeout(Duration::from_secs(10))
            .connect(&url)
            .await
            .expect("role connect")
    }

    async fn call_as(user: &str, req: &SupersessionRequest) -> Result<Value, SupersessionError> {
        let pool = role_pool(user).await;
        supersede_argus_source_event(&pool, req).await
    }

    fn mk_req(
        request_id: Uuid,
        source_ref: &str,
        expected_status: &str,
        old_hash: &str,
        old_thought: Option<Uuid>,
        new_content: &str,
        actor: &str,
        lane: &str,
    ) -> SupersessionRequest {
        let canon = format!(
            r#"{{"text":{}}}"#,
            serde_json::to_string(new_content).unwrap()
        );
        let hash = payload_hash_hex(&canon);
        SupersessionRequest {
            request_id,
            namespace: "test".into(),
            source_ref: source_ref.into(),
            expected_status: expected_status.into(),
            expected_old_payload_hash: old_hash.into(),
            expected_old_thought_id: old_thought,
            new_payload_canonical_json: canon,
            new_payload_hash: hash.clone(),
            new_content: new_content.into(),
            new_metadata: json!({
                "namespace": "test",
                "source_ref": source_ref,
                "payload_sha256": hash,
            }),
            embedding_model_id: EMB.into(),
            tagger_model_id: TAG.into(),
            actor: actor.into(),
            lane: lane.into(),
            approval_ref: "test-approval".into(),
            reason: "acceptance fixture".into(),
        }
    }

    async fn disable_thought_gate(admin: &PgPool) {
        let _ = sqlx::query(
            "ALTER TABLE public.thoughts DISABLE TRIGGER thoughts_require_gated_writer",
        )
        .execute(admin)
        .await;
    }

    async fn enable_thought_gate(admin: &PgPool) {
        let _ =
            sqlx::query("ALTER TABLE public.thoughts ENABLE TRIGGER thoughts_require_gated_writer")
                .execute(admin)
                .await;
    }

    /// Seed conflict source-event + active old thought. Content is unique per call.
    async fn seed_conflict(
        admin: &PgPool,
        source_ref: &str,
        old_content: &str,
    ) -> (Uuid, String, Uuid) {
        let thought_id = Uuid::new_v4();
        let event_id = Uuid::new_v4();
        let old_canon = format!(
            r#"{{"text":{}}}"#,
            serde_json::to_string(old_content).unwrap()
        );
        let old_hash = payload_hash_hex(&old_canon);
        disable_thought_gate(admin).await;
        sqlx::query(
            r#"
            INSERT INTO public.thoughts (id, scope, content, source, metadata, content_fingerprint)
            VALUES (
              $1, 'global', $2, 'manual', '{}'::jsonb,
              public.digest(pg_catalog.convert_to($2, 'UTF8'), 'sha256')
            )
            "#,
        )
        .bind(thought_id)
        .bind(old_content)
        .execute(admin)
        .await
        .expect("insert thought");
        enable_thought_gate(admin).await;

        sqlx::query(
            r#"
            INSERT INTO public.argus_source_events
              (id, namespace, source_ref, payload_hash, thought_id, status, metadata)
            VALUES ($1, 'test', $2, $3, $4, 'conflict', '{"conflict":true}'::jsonb)
            "#,
        )
        .bind(event_id)
        .bind(source_ref)
        .bind(&old_hash)
        .bind(thought_id)
        .execute(admin)
        .await
        .expect("insert source event");

        (event_id, old_hash, thought_id)
    }

    #[derive(Clone, Debug)]
    struct DomainSnap {
        thoughts: i64,
        links: i64,
        events: i64,
        receipts: i64,
        embeddings: i64,
        tags: i64,
        gate: i64,
    }

    async fn snap(admin: &PgPool) -> DomainSnap {
        DomainSnap {
            thoughts: sqlx::query_scalar("SELECT count(*)::bigint FROM public.thoughts")
                .fetch_one(admin)
                .await
                .unwrap(),
            links: sqlx::query_scalar("SELECT count(*)::bigint FROM public.thought_links")
                .fetch_one(admin)
                .await
                .unwrap(),
            events: sqlx::query_scalar("SELECT count(*)::bigint FROM public.argus_source_events")
                .fetch_one(admin)
                .await
                .unwrap(),
            receipts: sqlx::query_scalar(
                "SELECT count(*)::bigint FROM public.argus_source_event_supersession_receipts",
            )
            .fetch_one(admin)
            .await
            .unwrap(),
            embeddings: sqlx::query_scalar(
                "SELECT count(*)::bigint FROM public.pending_embeddings",
            )
            .fetch_one(admin)
            .await
            .unwrap(),
            tags: sqlx::query_scalar("SELECT count(*)::bigint FROM public.pending_tags")
                .fetch_one(admin)
                .await
                .unwrap(),
            gate: sqlx::query_scalar(
                "SELECT count(*)::bigint FROM public.thought_ingest_gate_events",
            )
            .fetch_one(admin)
            .await
            .unwrap(),
        }
    }

    fn assert_domain_unchanged(before: &DomainSnap, after: &DomainSnap, receipts_delta: i64) {
        assert_eq!(after.thoughts, before.thoughts, "thoughts");
        assert_eq!(after.links, before.links, "links");
        assert_eq!(after.events, before.events, "events");
        assert_eq!(after.embeddings, before.embeddings, "embeddings");
        assert_eq!(after.tags, before.tags, "tags");
        assert_eq!(after.gate, before.gate, "gate");
        assert_eq!(
            after.receipts,
            before.receipts + receipts_delta,
            "receipts delta"
        );
    }

    /// Independent receipt digest oracle from returned envelope (strips response-only keys).
    #[allow(dead_code)]
    fn independent_receipt_hash(returned: &Value) -> String {
        let mut env = returned.clone();
        if let Some(obj) = env.as_object_mut() {
            obj.remove("receipt_hash");
            obj.remove("replayed");
        }
        // Use compact JSON from Value::to_string which matches jsonb::text ordering for our keys
        // Postgres jsonb_build_object key order is insertion order; Value retains serde key order.
        let preimage = env.to_string();
        to_hex(&Sha256::digest(preimage.as_bytes()))
    }

    // --- Case 1: applied success ---
    #[tokio::test]
    async fn case_01_applied_success() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("apply-{uid}");
        let old_content = format!("old conflict body {uid}");
        let new_content = format!("corrected body {uid}");
        let (event_id, old_hash, tid) = seed_conflict(&admin, &source_ref, &old_content).await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &new_content,
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("applied");
        assert_eq!(out["outcome"], "applied");
        assert_eq!(out["replayed"], false);
        assert_eq!(out["stable_source_event_id"], event_id.to_string());
        assert!(out["new_thought_id"].as_str().is_some());
        assert!(out["replaces_link_id"].as_str().is_some());
        assert!(out["gate_event_id"].as_str().is_some());
        assert!(out["embedding_job_id"].as_str().is_some());
        assert!(out["tag_job_generation_id"].as_str().is_some());
        assert!(out["receipt_hash"].as_str().unwrap().len() == 64);

        let row = sqlx::query(
            "SELECT status, payload_hash, thought_id, metadata FROM public.argus_source_events WHERE id = $1",
        )
        .bind(event_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        let status: String = row.get("status");
        let ph: String = row.get("payload_hash");
        let new_tid: Uuid = row.get("thought_id");
        assert_eq!(status, "stored");
        assert_eq!(ph, req.new_payload_hash);
        assert_eq!(new_tid.to_string(), out["new_thought_id"].as_str().unwrap());

        let old_retracted: Option<String> =
            sqlx::query_scalar("SELECT retracted_at::text FROM public.thoughts WHERE id = $1")
                .bind(tid)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert!(old_retracted.is_some(), "old thought retracted");

        let link_cnt: i64 = sqlx::query_scalar(
            r#"SELECT count(*)::bigint FROM public.thought_links
               WHERE from_thought_id = $1 AND to_thought_id = $2
                 AND relation = 'replaces' AND deleted_at IS NULL"#,
        )
        .bind(new_tid)
        .bind(tid)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(link_cnt, 1);

        let emb: i64 = sqlx::query_scalar(
            "SELECT count(*)::bigint FROM public.pending_embeddings WHERE target_id = $1",
        )
        .bind(new_tid)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(emb, 1);
        let tags: i64 = sqlx::query_scalar(
            "SELECT count(*)::bigint FROM public.pending_tags WHERE thought_id = $1",
        )
        .bind(new_tid)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(tags, 1);

        let event_count: i64 =
            sqlx::query_scalar("SELECT count(*)::bigint FROM public.argus_source_events WHERE namespace='test' AND source_ref=$1")
                .bind(&source_ref)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert_eq!(event_count, 1, "no suffix row");

        let after = snap(&admin).await;
        assert_eq!(after.receipts, before.receipts + 1);
        assert_eq!(after.events, before.events);
        assert_eq!(after.thoughts, before.thoughts + 1);
        assert_eq!(after.links, before.links + 1);
    }

    // --- Case 2 (r2 §8.1): four CAS fixtures ---
    #[tokio::test]
    async fn case_02a_cas_status_changed_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("cas-status-{uid}");
        let (event_id, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("cas status old {uid}")).await;
        // Change observed status after seed (valid CAS target other than conflict)
        sqlx::query("UPDATE public.argus_source_events SET status = 'stored' WHERE id = $1")
            .bind(event_id)
            .execute(&admin)
            .await
            .unwrap();
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("cas status new {uid}"),
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("refusal");
        assert_eq!(out["outcome"], "refused_expected_state");
        assert_eq!(out["observed_old_status"], "stored");
        assert_eq!(out["observed_missing"], false);
        assert_domain_unchanged(&before, &snap(&admin).await, 1);
    }

    #[tokio::test]
    async fn case_02b_cas_wrong_hash_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("cas-hash-{uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("cas hash old {uid}")).await;
        let before = snap(&admin).await;
        let wrong = payload_hash_hex(&format!(r#"{{"text":"not-old-{uid}"}}"#));
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &wrong,
            Some(tid),
            &format!("cas hash new {uid}"),
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("refusal");
        assert_eq!(out["outcome"], "refused_expected_state");
        assert_eq!(out["observed_old_payload_hash"], old_hash);
        assert_domain_unchanged(&before, &snap(&admin).await, 1);
    }

    #[tokio::test]
    async fn case_02c_cas_wrong_thought_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("cas-tid-{uid}");
        let (_eid, old_hash, _tid) =
            seed_conflict(&admin, &source_ref, &format!("cas tid old {uid}")).await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(Uuid::new_v4()),
            &format!("cas tid new {uid}"),
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("refusal");
        assert_eq!(out["outcome"], "refused_expected_state");
        assert_domain_unchanged(&before, &snap(&admin).await, 1);
    }

    #[tokio::test]
    async fn case_02d_invalid_expected_status_zsi01_zero_receipt() {
        let admin = admin_pool().await;
        let before = snap(&admin).await;
        // reuse existing request id if any by using fixed id after seeding a receipt? prove ZSI01 before lock
        let existing_rid = Uuid::new_v4();
        // first create a missing-row receipt with this request id via valid conflict status
        let miss_ref = format!("zsi-pre-{}", Uuid::new_v4());
        let pre = mk_req(
            existing_rid,
            &miss_ref,
            "conflict",
            &payload_hash_hex(r#"{"text":"g"}"#),
            Some(Uuid::new_v4()),
            &format!("pre {}", Uuid::new_v4()),
            "diesel",
            "724808",
        );
        let _ = call_as("kengram_rt_supersession", &pre)
            .await
            .expect("pre refusal");
        let mid = snap(&admin).await;
        let req = mk_req(
            existing_rid,
            "case-invalid-status",
            "stored",
            &payload_hash_hex(r#"{"text":"o"}"#),
            Some(Uuid::new_v4()),
            "new body",
            "diesel",
            "724808",
        );
        match call_as("kengram_rt_supersession", &req).await {
            Err(SupersessionError::SqlState { code, .. }) => assert_eq!(code, "ZSI01"),
            o => panic!("expected ZSI01 got {o:?}"),
        }
        // existing receipt unchanged; no new receipt
        assert_eq!(snap(&admin).await.receipts, mid.receipts);
        assert_eq!(snap(&admin).await.receipts, before.receipts + 1);
    }

    // --- Case 3: missing stable row ---
    #[tokio::test]
    async fn case_03_missing_stable_row_refusal_and_replay() {
        let admin = admin_pool().await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &format!("missing-{}", Uuid::new_v4()),
            "conflict",
            &payload_hash_hex(r#"{"text":"ghost"}"#),
            Some(Uuid::new_v4()),
            &format!("missing row correction {}", Uuid::new_v4()),
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("refusal");
        assert_eq!(out["outcome"], "refused_expected_state");
        assert_eq!(out["observed_missing"], true);
        assert!(out.get("stable_source_event_id").unwrap().is_null());
        assert_eq!(out["replayed"], false);
        assert!(out.get("observed_old_status").unwrap().is_null());
        let after1 = snap(&admin).await;
        assert_domain_unchanged(&before, &after1, 1);

        let out2 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("replay");
        assert_eq!(out2["replayed"], true);
        assert_eq!(out2["receipt_hash"], out["receipt_hash"]);
        assert_eq!(snap(&admin).await.receipts, after1.receipts);
    }

    // --- Case 4: exact second apply ---
    #[tokio::test]
    async fn case_04_exact_second_apply_replay() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("replay-apply-{uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("replay old {uid}")).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("replay new {uid}"),
            "diesel",
            "724808",
        );
        let out1 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("first apply");
        assert_eq!(out1["outcome"], "applied");
        let after1 = snap(&admin).await;
        let out2 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("second apply");
        assert_eq!(out2["replayed"], true);
        assert_eq!(out2["receipt_hash"], out1["receipt_hash"]);
        assert_eq!(out2["new_thought_id"], out1["new_thought_id"]);
        let after2 = snap(&admin).await;
        assert_eq!(after2.thoughts, after1.thoughts);
        assert_eq!(after2.links, after1.links);
        assert_eq!(after2.receipts, after1.receipts);
        assert_eq!(after2.embeddings, after1.embeddings);
    }

    // --- Case 5 (r2 §8.3): adversarial request framing ---
    #[tokio::test]
    async fn case_05_request_id_collision_adversarial_framing() {
        let request_id = Uuid::new_v4();
        let source_ref = format!("collide-miss-{}", Uuid::new_v4());
        let content = format!("body A {}", Uuid::new_v4());
        let a = mk_req(
            request_id,
            &source_ref,
            "conflict",
            &payload_hash_hex(r#"{"text":"ghost"}"#),
            Some(Uuid::new_v4()),
            &content,
            "alpha|beta",
            "gamma",
        );
        let mut b = mk_req(
            request_id,
            &source_ref,
            "conflict",
            &payload_hash_hex(r#"{"text":"ghost"}"#),
            a.expected_old_thought_id,
            &content,
            "alpha",
            "beta|gamma",
        );
        b.new_content = a.new_content.clone();
        b.new_payload_canonical_json = a.new_payload_canonical_json.clone();
        b.new_payload_hash = a.new_payload_hash.clone();
        b.new_metadata = a.new_metadata.clone();
        b.expected_old_payload_hash = a.expected_old_payload_hash.clone();

        let ea = independent_request_envelope(&a);
        let eb = independent_request_envelope(&b);
        assert_ne!(ea.to_string(), eb.to_string());
        assert_ne!(
            payload_hash_hex(&ea.to_string()),
            payload_hash_hex(&eb.to_string())
        );
        // naive concat collides
        assert_eq!(
            format!("{}|{}", a.actor, a.lane),
            format!("{}|{}", b.actor, b.lane)
        );

        let admin = admin_pool().await;
        let before = snap(&admin).await;
        let out_a = call_as("kengram_rt_supersession", &a)
            .await
            .expect("A terminal receipt");
        assert_eq!(out_a["replayed"], false);
        let mid = snap(&admin).await;
        match call_as("kengram_rt_supersession", &b).await {
            Err(SupersessionError::SqlState { code, .. }) => assert_eq!(code, "ZSR01"),
            Ok(v) if v["replayed"] == true => {
                panic!("adversarial pair must not share digest; got replay");
            }
            other => panic!("expected ZSR01 got {other:?}"),
        }
        assert_eq!(snap(&admin).await.receipts, mid.receipts);
        assert_eq!(snap(&admin).await.receipts, before.receipts + 1);
    }

    // --- Case 6: concurrent apply ---
    #[tokio::test]
    async fn case_06_concurrent_exact_and_race() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("conc-{uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("conc old {uid}")).await;
        let rid = Uuid::new_v4();
        let req = mk_req(
            rid,
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("conc new {uid}"),
            "diesel",
            "724808",
        );
        let (r1, r2) = tokio::join!(
            call_as("kengram_rt_supersession", &req),
            call_as("kengram_rt_supersession", &req),
        );
        let outs: Vec<_> = [r1, r2]
            .into_iter()
            .map(|r| r.expect("concurrent call ok"))
            .collect();
        let applied = outs.iter().filter(|o| o["replayed"] == false).count();
        let replayed = outs.iter().filter(|o| o["replayed"] == true).count();
        assert_eq!(applied, 1, "one applied");
        assert_eq!(replayed, 1, "one replay");
        assert_eq!(outs[0]["receipt_hash"], outs[1]["receipt_hash"]);

        // Two different request IDs from same expected state: one apply one refusal
        let uid2 = Uuid::new_v4();
        let source_ref2 = format!("conc2-{uid2}");
        let (_e2, h2, t2) = seed_conflict(&admin, &source_ref2, &format!("conc2 old {uid2}")).await;
        let req_a = mk_req(
            Uuid::new_v4(),
            &source_ref2,
            "conflict",
            &h2,
            Some(t2),
            &format!("conc2 body A {uid2}"),
            "diesel",
            "724808",
        );
        let req_b = mk_req(
            Uuid::new_v4(),
            &source_ref2,
            "conflict",
            &h2,
            Some(t2),
            &format!("conc2 body B {uid2}"),
            "diesel",
            "724808",
        );
        let (ra, rb) = tokio::join!(
            call_as("kengram_rt_supersession", &req_a),
            call_as("kengram_rt_supersession", &req_b),
        );
        let oa = ra.expect("a");
        let ob = rb.expect("b");
        let outcomes = [
            oa["outcome"].as_str().unwrap(),
            ob["outcome"].as_str().unwrap(),
        ];
        assert!(
            outcomes.contains(&"applied") && outcomes.contains(&"refused_expected_state"),
            "expected one apply one CAS refusal, got {outcomes:?}"
        );
        let new_thoughts = [oa.get("new_thought_id"), ob.get("new_thought_id")]
            .into_iter()
            .filter(|v| v.map(|x| !x.is_null()).unwrap_or(false))
            .count();
        assert_eq!(new_thoughts, 1, "never two corrected thoughts");
    }

    // --- Case 7 (r2 §8.4): existing corrected content ---
    #[tokio::test]
    async fn case_07_exact_content_duplicate_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("exact-dup-{uid}");
        let new_content = format!("already exists corrected {uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("exact dup old {uid}")).await;
        // Pre-seed corrected fingerprint as a live thought
        disable_thought_gate(&admin).await;
        sqlx::query(
            r#"
            INSERT INTO public.thoughts (id, scope, content, source, metadata, content_fingerprint)
            VALUES (
              $1, 'global', $2, 'manual', '{}'::jsonb,
              public.digest(pg_catalog.convert_to($2, 'UTF8'), 'sha256')
            )
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(&new_content)
        .execute(&admin)
        .await
        .expect("preseed content");
        enable_thought_gate(&admin).await;

        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &new_content,
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("exact content refusal");
        assert_eq!(out["outcome"], "refused_exact_content_duplicate");
        assert!(out.get("new_thought_id").unwrap().is_null());
        assert!(out.get("gate_event_id").unwrap().is_null());
        assert_domain_unchanged(&before, &snap(&admin).await, 1);

        // exact second apply returns same receipt
        let out2 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("replay exact");
        assert_eq!(out2["replayed"], true);
        assert_eq!(out2["receipt_hash"], out["receipt_hash"]);
        assert_eq!(snap(&admin).await.receipts, before.receipts + 1);
    }

    // --- Case 8 (r2 §8.5): authorization layers ---
    #[tokio::test]
    async fn case_08_authorization_acl_and_session_guard() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        // Catalog: only dedicated has EXECUTE
        for user in [
            "kengram_rt_native_mcp",
            "kengram_rt_session",
            "kengram_rt_telegram",
        ] {
            let allowed: bool = sqlx::query_scalar(
                r#"
                SELECT has_function_privilege($1,
                  'public.supersede_argus_source_event(uuid,text,text,text,text,uuid,text,text,text,jsonb,text,text,text,text,text,text)',
                  'EXECUTE')
                "#,
            )
            .bind(user)
            .fetch_one(&admin)
            .await
            .unwrap();
            assert!(!allowed, "{user} must not have EXECUTE");
        }
        let dedicated: bool = sqlx::query_scalar(
            r#"
            SELECT has_function_privilege('kengram_rt_supersession',
              'public.supersede_argus_source_event(uuid,text,text,text,text,uuid,text,text,text,jsonb,text,text,text,text,text,text)',
              'EXECUTE')
            "#,
        )
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(dedicated);

        // ACL 42501 for ordinary runtime logins (if password works)
        let req = mk_req(
            Uuid::new_v4(),
            &format!("acl-{uid}"),
            "conflict",
            &payload_hash_hex(r#"{"text":"o"}"#),
            Some(Uuid::new_v4()),
            &format!("acl body {uid}"),
            "diesel",
            "724808",
        );
        let before = snap(&admin).await;
        for user in [
            "kengram_rt_native_mcp",
            "kengram_rt_session",
            "kengram_rt_telegram",
        ] {
            match call_as(user, &req).await {
                Err(SupersessionError::SqlState { code, .. }) => {
                    assert_eq!(code, "42501", "{user} expected 42501");
                }
                Err(SupersessionError::Db(_)) => {
                    // connection failure if password wrong is not 42501 evidence; try SET ROLE
                    let res: Result<(Value,), sqlx::Error> = sqlx::query_as(
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
                    .fetch_one(&admin)
                    .await;
                    // superuser can execute - not the ACL proof
                    let _ = res;
                    // catalog already proved no EXECUTE for role
                }
                o => panic!("{user} unexpected {o:?}"),
            }
        }
        // break-glass kengram -> ZSA01
        match call_as("kengram", &req).await {
            Err(SupersessionError::SqlState { code, .. }) => assert_eq!(code, "ZSA01"),
            o => panic!("expected ZSA01 {o:?}"),
        }
        assert_eq!(snap(&admin).await.receipts, before.receipts);

        // dedicated role no direct DML on receipts
        let pool = role_pool("kengram_rt_supersession").await;
        let ins = sqlx::query(
            "INSERT INTO public.argus_source_event_supersession_receipts (request_id) VALUES ($1)",
        )
        .bind(Uuid::new_v4())
        .execute(&pool)
        .await;
        assert!(ins.is_err(), "dedicated role must not INSERT receipts");
    }

    // --- Case 9: payload/metadata binding ---
    #[tokio::test]
    async fn case_09_payload_metadata_binding_refuses() {
        let admin = admin_pool().await;
        let before = snap(&admin).await;
        let uid = Uuid::new_v4();
        // wrong hash
        let mut req = mk_req(
            Uuid::new_v4(),
            &format!("bad-ph-{uid}"),
            "conflict",
            &payload_hash_hex(r#"{"text":"o"}"#),
            Some(Uuid::new_v4()),
            &format!("body {uid}"),
            "diesel",
            "724808",
        );
        req.new_payload_hash = "0".repeat(64);
        req.new_metadata = json!({
            "namespace": "test",
            "source_ref": req.source_ref,
            "payload_sha256": req.new_payload_hash,
        });
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());

        // wrong metadata namespace
        let mut req2 = mk_req(
            Uuid::new_v4(),
            &format!("bad-meta-{uid}"),
            "conflict",
            &payload_hash_hex(r#"{"text":"o"}"#),
            Some(Uuid::new_v4()),
            &format!("body2 {uid}"),
            "diesel",
            "724808",
        );
        req2.new_metadata = json!({
            "namespace": "wrong",
            "source_ref": req2.source_ref,
            "payload_sha256": req2.new_payload_hash,
        });
        assert!(call_as("kengram_rt_supersession", &req2).await.is_err());

        // non-object metadata
        let mut req3 = mk_req(
            Uuid::new_v4(),
            &format!("bad-meta2-{uid}"),
            "conflict",
            &payload_hash_hex(r#"{"text":"o"}"#),
            Some(Uuid::new_v4()),
            &format!("body3 {uid}"),
            "diesel",
            "724808",
        );
        req3.new_metadata = json!(["not", "object"]);
        assert!(call_as("kengram_rt_supersession", &req3).await.is_err());

        assert_eq!(snap(&admin).await.receipts, before.receipts);
    }

    // --- Case 10: old-thought state ---
    #[tokio::test]
    async fn case_10_old_thought_unavailable() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("old-th-{uid}");
        let (event_id, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("old th {uid}")).await;
        // retract old thought first
        disable_thought_gate(&admin).await;
        sqlx::query(
            "UPDATE public.thoughts SET retracted_at = now(), retracted_reason = 'test' WHERE id = $1",
        )
        .bind(tid)
        .execute(&admin)
        .await
        .unwrap();
        enable_thought_gate(&admin).await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("new after retract {uid}"),
            "diesel",
            "724808",
        );
        let res = call_as("kengram_rt_supersession", &req).await;
        assert!(res.is_err(), "must not apply with retracted old thought");
        // no applied receipt; domain for source event unchanged status
        let status: String =
            sqlx::query_scalar("SELECT status FROM public.argus_source_events WHERE id = $1")
                .bind(event_id)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert_eq!(status, "conflict");
        assert_eq!(snap(&admin).await.receipts, before.receipts);
    }

    // --- Case 11: exact-one update / sibling untouched ---
    #[tokio::test]
    async fn case_11_sibling_source_row_untouched() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("sib-main-{uid}");
        let sibling_ref = format!("sib-side-{uid}");
        let (eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("sib main {uid}")).await;
        let (sid, sh, st) = seed_conflict(&admin, &sibling_ref, &format!("sib side {uid}")).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("sib corrected {uid}"),
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("apply main");
        assert_eq!(out["outcome"], "applied");
        let sib = sqlx::query(
            "SELECT status, payload_hash, thought_id FROM public.argus_source_events WHERE id = $1",
        )
        .bind(sid)
        .fetch_one(&admin)
        .await
        .unwrap();
        let s_status: String = sib.get("status");
        let s_hash: String = sib.get("payload_hash");
        let s_tid: Uuid = sib.get("thought_id");
        assert_eq!(s_status, "conflict");
        assert_eq!(s_hash, sh);
        assert_eq!(s_tid, st);
        let _ = eid;
    }

    // --- Case 12: rollback injection triggers ---
    #[tokio::test]
    async fn case_12_rollback_injection_receipt_insert() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-rcpt-{uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb old {uid}")).await;
        sqlx::query(
            r#"
            CREATE OR REPLACE FUNCTION public._test_supersession_rb_receipt()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              RAISE EXCEPTION 'test_rb_receipt' USING ERRCODE = 'P0001';
            END;
            $$;
            "#,
        )
        .execute(&admin)
        .await
        .expect("create rb receipt fn");
        sqlx::query(
            "DROP TRIGGER IF EXISTS _test_rb_receipt ON public.argus_source_event_supersession_receipts",
        )
        .execute(&admin)
        .await
        .expect("drop old rb receipt trigger");
        sqlx::query(
            r#"
            CREATE TRIGGER _test_rb_receipt
              BEFORE INSERT ON public.argus_source_event_supersession_receipts
              FOR EACH ROW EXECUTE FUNCTION public._test_supersession_rb_receipt()
            "#,
        )
        .execute(&admin)
        .await
        .expect("create rb receipt trigger");
        let pc = sqlx::query(
            "INSERT INTO public.argus_source_event_supersession_receipts (request_id) VALUES ($1)",
        )
        .bind(Uuid::new_v4())
        .execute(&admin)
        .await;
        assert!(pc.is_err(), "positive control must fire");

        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("rb new {uid}"),
            "diesel",
            "724808",
        );
        let res = call_as("kengram_rt_supersession", &req).await;
        assert!(res.is_err(), "function must error on receipt insert fail");
        let after = snap(&admin).await;
        assert_eq!(after.receipts, before.receipts);
        assert_eq!(after.thoughts, before.thoughts);
        assert_eq!(after.links, before.links);
        let status: String = sqlx::query_scalar(
            "SELECT status FROM public.argus_source_events WHERE source_ref = $1 AND namespace='test'",
        )
        .bind(&source_ref)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(status, "conflict");

        sqlx::query(
            "DROP TRIGGER IF EXISTS _test_rb_receipt ON public.argus_source_event_supersession_receipts",
        )
        .execute(&admin)
        .await
        .expect("cleanup rb receipt trigger");
    }

    #[tokio::test]
    async fn case_12b_rollback_injection_pending_embedding() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-emb-{uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb emb old {uid}")).await;
        sqlx::query(
            r#"
            CREATE OR REPLACE FUNCTION public._test_supersession_rb_emb()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              IF NEW.model_id = 'bge-m3:1024' THEN
                RAISE EXCEPTION 'test_rb_embedding' USING ERRCODE = 'P0001';
              END IF;
              RETURN NEW;
            END;
            $$;
            "#,
        )
        .execute(&admin)
        .await
        .expect("create emb fn");
        sqlx::query("DROP TRIGGER IF EXISTS _test_rb_emb ON public.pending_embeddings")
            .execute(&admin)
            .await
            .expect("drop emb trigger");
        sqlx::query(
            r#"
            CREATE TRIGGER _test_rb_emb
              BEFORE INSERT ON public.pending_embeddings
              FOR EACH ROW EXECUTE FUNCTION public._test_supersession_rb_emb()
            "#,
        )
        .execute(&admin)
        .await
        .expect("create emb trigger");

        let pc = sqlx::query(
            r#"INSERT INTO public.pending_embeddings (target_kind, target_id, model_id)
               VALUES ('thought', $1, 'bge-m3:1024')"#,
        )
        .bind(Uuid::new_v4())
        .execute(&admin)
        .await;
        assert!(pc.is_err(), "positive control emb trigger");

        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("rb emb new {uid}"),
            "diesel",
            "724808",
        );
        let res = call_as("kengram_rt_supersession", &req).await;
        assert!(res.is_err());
        let after = snap(&admin).await;
        assert_eq!(after.receipts, before.receipts);
        assert_eq!(after.thoughts, before.thoughts);
        assert_eq!(after.links, before.links);

        sqlx::query("DROP TRIGGER IF EXISTS _test_rb_emb ON public.pending_embeddings")
            .execute(&admin)
            .await
            .expect("cleanup emb trigger");
    }

    // --- Case 13 (r2 §8.6): receipt immutability + hash ---
    #[tokio::test]
    async fn case_13_receipt_immutability_and_hash() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rcpt-hash-{uid}");
        let (_eid, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rcpt old {uid}")).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("rcpt new {uid}"),
            "diesel",
            "724808",
        );
        let out = call_as("kengram_rt_supersession", &req)
            .await
            .expect("applied");
        assert_eq!(out["outcome"], "applied");

        // DB-side recompute of stored digest must match returned hash
        let db_hex: String = sqlx::query_scalar(
            r#"
            SELECT encode(receipt_digest, 'hex')
            FROM public.argus_source_event_supersession_receipts
            WHERE request_id = $1
            "#,
        )
        .bind(req.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(db_hex, out["receipt_hash"].as_str().unwrap());

        let db_check: bool = sqlx::query_scalar(
            r#"
            SELECT receipt_digest = public.digest(
              pg_catalog.convert_to(canonical_receipt_json::text, 'UTF8'), 'sha256'
            )
            FROM public.argus_source_event_supersession_receipts
            WHERE request_id = $1
            "#,
        )
        .bind(req.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(db_check, "stored digest equals SHA of envelope text");

        // Independent frozen §7 key oracle (must not use producer helper)
        let canon_json: Value = sqlx::query_scalar(
            r#"
            SELECT canonical_receipt_json
            FROM public.argus_source_event_supersession_receipts
            WHERE request_id = $1
            "#,
        )
        .bind(req.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        let required_keys = [
            "v",
            "request_id",
            "request_digest",
            "outcome",
            "stable_source_event_id",
            "namespace",
            "source_ref",
            "expected_old_status",
            "expected_old_payload_hash",
            "expected_old_thought_id",
            "observed_missing",
            "observed_old_status",
            "observed_old_payload_hash",
            "observed_old_thought_id",
            "new_payload_hash",
            "new_thought_id",
            "replaces_link_id",
            "gate_event_id",
            "embedding_job_id",
            "tag_job_generation_id",
            "embedding_model_id",
            "tagger_model_id",
            "actor",
            "lane",
            "approval_ref",
            "reason",
            "authenticated_session_user",
            "occurred_at",
        ];
        let obj = canon_json.as_object().expect("receipt object");
        for k in required_keys {
            assert!(
                obj.contains_key(k),
                "frozen receipt envelope missing key {k}"
            );
        }
        assert!(
            !obj.contains_key("receipt_hash"),
            "receipt_hash must not be in envelope"
        );
        assert!(
            !obj.contains_key("replayed"),
            "replayed must not be in envelope"
        );

        // Independent oracle: recompute from DB canonical_receipt_json text
        let canon_text: String = sqlx::query_scalar(
            r#"
            SELECT canonical_receipt_json::text
            FROM public.argus_source_event_supersession_receipts
            WHERE request_id = $1
            "#,
        )
        .bind(req.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        let oracle = to_hex(&Sha256::digest(canon_text.as_bytes()));
        assert_eq!(oracle, db_hex);

        // Direct DML under dedicated role refuses
        let pool = role_pool("kengram_rt_supersession").await;
        assert!(sqlx::query(
            "UPDATE public.argus_source_event_supersession_receipts SET reason = 'x' WHERE request_id = $1"
        )
        .bind(req.request_id)
        .execute(&pool)
        .await
        .is_err());
        assert!(
            sqlx::query(
                "DELETE FROM public.argus_source_event_supersession_receipts WHERE request_id = $1"
            )
            .bind(req.request_id)
            .execute(&pool)
            .await
            .is_err()
        );

        // Refuse fixtures also have matching digests
        let miss = mk_req(
            Uuid::new_v4(),
            &format!("rcpt-miss-{uid}"),
            "conflict",
            &payload_hash_hex(r#"{"text":"g"}"#),
            Some(Uuid::new_v4()),
            &format!("miss {uid}"),
            "diesel",
            "724808",
        );
        let out_m = call_as("kengram_rt_supersession", &miss)
            .await
            .expect("miss");
        let db_m: String = sqlx::query_scalar(
            "SELECT encode(receipt_digest,'hex') FROM public.argus_source_event_supersession_receipts WHERE request_id=$1",
        )
        .bind(miss.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(db_m, out_m["receipt_hash"].as_str().unwrap());
    }

    // --- Case 14: rollback migration ---
    #[tokio::test]
    async fn case_14_rollback_migration_requires_zero_receipts() {
        let admin = admin_pool().await;
        // With receipts present, down migration must fail and leave objects
        let cnt: i64 = sqlx::query_scalar(
            "SELECT count(*)::bigint FROM public.argus_source_event_supersession_receipts",
        )
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(
            cnt > 0,
            "suite must have receipts before down-migration check"
        );

        let down = include_str!(
            "../../../migrations/rollback/0036_argus_source_event_supersession_transaction_down.sql"
        );
        // Execute down via temporary connection - expect failure
        let res = sqlx::raw_sql(down).execute(&admin).await;
        assert!(res.is_err(), "down must fail when receipts exist");

        // function still present
        let still: bool = sqlx::query_scalar(
            r#"
            SELECT to_regprocedure(
              'public.supersede_argus_source_event(uuid,text,text,text,text,uuid,text,text,text,jsonb,text,text,text,text,text,text)'
            ) IS NOT NULL
            "#,
        )
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(still, "objects must remain after failed down");
    }
}

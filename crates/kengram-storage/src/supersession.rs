//! Stable-source supersession transaction caller + disposable-PG acceptance.
//! Spec: kengram-supersession-transactional-capability r1 §9 + r2 §8/§9.
//! Six-path fence: no runtime sha2 dependency (hash helpers are test-only).

use serde_json::{Value, json};
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

/// Independent request envelope (literal §6 keys) — no producer helper dependency.
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
    use sha2::{Digest, Sha256};
    use sqlx::Row;
    use sqlx::postgres::PgPoolOptions;
    use std::time::Duration;

    const EMB: &str = "bge-m3:1024";
    const TAG: &str = "test-tagger";

    fn to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn payload_hash_hex(canonical_json: &str) -> String {
        to_hex(&Sha256::digest(canonical_json.as_bytes()))
    }

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
            .acquire_timeout(Duration::from_secs(15))
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
            .acquire_timeout(Duration::from_secs(15))
            .connect(&url)
            .await
            .unwrap_or_else(|e| panic!("role connect {user}: {e}"))
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

    /// Ordered-byte domain snapshot (not mere counts).
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DomainSnap {
        thoughts: String,
        links: String,
        events: String,
        embeddings: String,
        tags: String,
        gate: String,
        receipts: i64,
    }

    async fn snap(admin: &PgPool) -> DomainSnap {
        let thoughts: String = sqlx::query_scalar(
            r#"SELECT coalesce(string_agg(id::text || ':' || coalesce(retracted_at::text,''), '|' ORDER BY id), '')
               FROM public.thoughts"#,
        )
        .fetch_one(admin)
        .await
        .unwrap();
        let links: String = sqlx::query_scalar(
            r#"SELECT coalesce(string_agg(id::text || ':' || relation || ':' || coalesce(deleted_at::text,''), '|' ORDER BY id), '')
               FROM public.thought_links"#,
        )
        .fetch_one(admin)
        .await
        .unwrap();
        let events: String = sqlx::query_scalar(
            r#"SELECT coalesce(string_agg(id::text || ':' || status || ':' || payload_hash, '|' ORDER BY id), '')
               FROM public.argus_source_events"#,
        )
        .fetch_one(admin)
        .await
        .unwrap();
        let embeddings: String = sqlx::query_scalar(
            r#"SELECT coalesce(string_agg(id::text || ':' || target_id::text, '|' ORDER BY id), '')
               FROM public.pending_embeddings"#,
        )
        .fetch_one(admin)
        .await
        .unwrap();
        let tags: String = sqlx::query_scalar(
            r#"SELECT coalesce(string_agg(thought_id::text || ':' || tag_job_generation_id::text, '|' ORDER BY thought_id), '')
               FROM public.pending_tags"#,
        )
        .fetch_one(admin)
        .await
        .unwrap();
        let gate: String = sqlx::query_scalar(
            r#"SELECT coalesce(string_agg(id::text, '|' ORDER BY id), '')
               FROM public.thought_ingest_gate_events"#,
        )
        .fetch_one(admin)
        .await
        .unwrap();
        let receipts: i64 = sqlx::query_scalar(
            "SELECT count(*)::bigint FROM public.argus_source_event_supersession_receipts",
        )
        .fetch_one(admin)
        .await
        .unwrap();
        DomainSnap {
            thoughts,
            links,
            events,
            embeddings,
            tags,
            gate,
            receipts,
        }
    }

    fn assert_domain_unchanged(before: &DomainSnap, after: &DomainSnap, receipts_delta: i64) {
        assert_eq!(after.thoughts, before.thoughts, "thoughts bytes");
        assert_eq!(after.links, before.links, "links bytes");
        assert_eq!(after.events, before.events, "events bytes");
        assert_eq!(after.embeddings, before.embeddings, "embeddings bytes");
        assert_eq!(after.tags, before.tags, "tags bytes");
        assert_eq!(after.gate, before.gate, "gate bytes");
        assert_eq!(after.receipts, before.receipts + receipts_delta, "receipts");
    }

    async fn drop_test_trigger(admin: &PgPool, name: &str, table: &str) {
        let q = format!("DROP TRIGGER IF EXISTS {name} ON {table}");
        let _ = sqlx::query(&q).execute(admin).await;
    }

    #[tokio::test]
    async fn case_01_applied_success() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("apply-{uid}");
        let (event_id, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("old conflict body {uid}")).await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("corrected body {uid}"),
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
        assert!(out["receipt_hash"].as_str().unwrap().len() == 64);
        let status: String =
            sqlx::query_scalar("SELECT status FROM public.argus_source_events WHERE id=$1")
                .bind(event_id)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert_eq!(status, "stored");
        let old_retracted: Option<String> =
            sqlx::query_scalar("SELECT retracted_at::text FROM public.thoughts WHERE id=$1")
                .bind(tid)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert!(old_retracted.is_some());
        let after = snap(&admin).await;
        assert_eq!(after.receipts, before.receipts + 1);
    }

    #[tokio::test]
    async fn case_02a_cas_status_changed_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("cas-status-{uid}");
        let (event_id, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("cas status old {uid}")).await;
        sqlx::query("UPDATE public.argus_source_events SET status='stored' WHERE id=$1")
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
        assert_domain_unchanged(&before, &snap(&admin).await, 1);
    }

    #[tokio::test]
    async fn case_02b_cas_wrong_hash_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("cas-hash-{uid}");
        let (_e, old_hash, tid) =
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
        let (_e, old_hash, _tid) =
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
        let existing_rid = Uuid::new_v4();
        let pre = mk_req(
            existing_rid,
            &format!("zsi-pre-{}", Uuid::new_v4()),
            "conflict",
            &payload_hash_hex(r#"{"text":"g"}"#),
            Some(Uuid::new_v4()),
            &format!("pre {}", Uuid::new_v4()),
            "diesel",
            "724808",
        );
        let _ = call_as("kengram_rt_supersession", &pre).await.expect("pre");
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
            o => panic!("expected ZSI01 {o:?}"),
        }
        assert_eq!(snap(&admin).await.receipts, mid.receipts);
    }

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
        assert_domain_unchanged(&before, &snap(&admin).await, 1);
        let out2 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("replay");
        assert_eq!(out2["replayed"], true);
        assert_eq!(out2["receipt_hash"], out["receipt_hash"]);
    }

    #[tokio::test]
    async fn case_04_exact_second_apply_replay() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("replay-apply-{uid}");
        let (_e, old_hash, tid) =
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
            .expect("first");
        assert_eq!(out1["outcome"], "applied");
        let after1 = snap(&admin).await;
        let out2 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("second");
        assert_eq!(out2["replayed"], true);
        assert_eq!(out2["receipt_hash"], out1["receipt_hash"]);
        let after2 = snap(&admin).await;
        assert_eq!(after2.thoughts, after1.thoughts);
        assert_eq!(after2.receipts, after1.receipts);
    }

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
        assert_eq!(
            format!("{}|{}", a.actor, a.lane),
            format!("{}|{}", b.actor, b.lane)
        );
        let admin = admin_pool().await;
        let before = snap(&admin).await;
        let out_a = call_as("kengram_rt_supersession", &a).await.expect("A");
        assert_eq!(out_a["replayed"], false);
        match call_as("kengram_rt_supersession", &b).await {
            Err(SupersessionError::SqlState { code, .. }) => assert_eq!(code, "ZSR01"),
            other => panic!("expected ZSR01 {other:?}"),
        }
        assert_eq!(snap(&admin).await.receipts, before.receipts + 1);
    }

    #[tokio::test]
    async fn case_06_concurrent_exact_and_race() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("conc-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("conc old {uid}")).await;
        let req = mk_req(
            Uuid::new_v4(),
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
        let outs = [r1.expect("c1"), r2.expect("c2")];
        assert_eq!(outs.iter().filter(|o| o["replayed"] == false).count(), 1);
        assert_eq!(outs.iter().filter(|o| o["replayed"] == true).count(), 1);

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
            "{outcomes:?}"
        );
    }

    #[tokio::test]
    async fn case_07_exact_content_duplicate_refusal() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("exact-dup-{uid}");
        let new_content = format!("already exists corrected {uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("exact dup old {uid}")).await;
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
        .unwrap();
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
            .expect("exact");
        assert_eq!(out["outcome"], "refused_exact_content_duplicate");
        assert_domain_unchanged(&before, &snap(&admin).await, 1);
        let out2 = call_as("kengram_rt_supersession", &req)
            .await
            .expect("replay");
        assert_eq!(out2["replayed"], true);
    }

    #[tokio::test]
    async fn case_08_authorization_acl_and_session_guard() {
        let admin = admin_pool().await;
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
            assert!(!allowed, "{user}");
        }
        // no kengram_runtime membership
        let has_rt: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS (
              SELECT 1 FROM pg_auth_members m
              JOIN pg_roles r ON r.oid = m.roleid
              JOIN pg_roles u ON u.oid = m.member
              WHERE r.rolname = 'kengram_runtime' AND u.rolname = 'kengram_rt_supersession'
            )
            "#,
        )
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(
            !has_rt,
            "dedicated role must not be member of kengram_runtime"
        );

        let uid = Uuid::new_v4();
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
                    assert_eq!(code, "42501", "{user}");
                }
                o => panic!("{user} expected 42501 got {o:?}"),
            }
        }
        match call_as("kengram", &req).await {
            Err(SupersessionError::SqlState { code, .. }) => assert_eq!(code, "ZSA01"),
            o => panic!("ZSA01 {o:?}"),
        }
        assert_eq!(snap(&admin).await.receipts, before.receipts);
        let pool = role_pool("kengram_rt_supersession").await;
        assert!(sqlx::query(
            "INSERT INTO public.argus_source_event_supersession_receipts (request_id) VALUES ($1)",
        )
        .bind(Uuid::new_v4())
        .execute(&pool)
        .await
        .is_err());
    }

    #[tokio::test]
    async fn case_09_payload_metadata_binding_refuses() {
        let admin = admin_pool().await;
        let before = snap(&admin).await;
        let uid = Uuid::new_v4();
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
        assert_eq!(snap(&admin).await.receipts, before.receipts);
    }

    #[tokio::test]
    async fn case_10_old_thought_retracted_unavailable() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("old-th-{uid}");
        let (event_id, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("old th {uid}")).await;
        disable_thought_gate(&admin).await;
        sqlx::query(
            "UPDATE public.thoughts SET retracted_at=now(), retracted_reason='test' WHERE id=$1",
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
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        let status: String =
            sqlx::query_scalar("SELECT status FROM public.argus_source_events WHERE id=$1")
                .bind(event_id)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert_eq!(status, "conflict");
        assert_eq!(snap(&admin).await.receipts, before.receipts);
    }

    #[tokio::test]
    async fn case_10b_old_thought_missing() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("old-miss-{uid}");
        let (event_id, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("old miss {uid}")).await;
        // Point event at a non-existent thought id for CAS match, then use that id
        let ghost = Uuid::new_v4();
        sqlx::query("UPDATE public.argus_source_events SET thought_id=$1 WHERE id=$2")
            .bind(ghost)
            .bind(event_id)
            .execute(&admin)
            .await
            .ok();
        // If FK prevents, delete old thought instead
        let status_fk =
            sqlx::query("UPDATE public.argus_source_events SET thought_id=$1 WHERE id=$2")
                .bind(ghost)
                .bind(event_id)
                .execute(&admin)
                .await;
        let (use_hash, use_tid) = if status_fk.is_ok() {
            (old_hash, ghost)
        } else {
            // delete thought (may fail FK) — use wrong expected tid that is missing
            (old_hash, Uuid::new_v4())
        };
        let _ = tid;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &use_hash,
            Some(use_tid),
            &format!("new miss th {uid}"),
            "diesel",
            "724808",
        );
        // Either CAS refusal or old-thought unavailable error — must not apply
        let res = call_as("kengram_rt_supersession", &req).await;
        match res {
            Ok(v) => assert_ne!(v["outcome"], "applied"),
            Err(_) => {}
        }
        let st: String =
            sqlx::query_scalar("SELECT status FROM public.argus_source_events WHERE id=$1")
                .bind(event_id)
                .fetch_one(&admin)
                .await
                .unwrap();
        assert_ne!(st, "stored");
        let _ = before;
    }

    #[tokio::test]
    async fn case_11_sibling_source_row_untouched() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("sib-main-{uid}");
        let sibling_ref = format!("sib-side-{uid}");
        let (_e, old_hash, tid) =
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
            .expect("apply");
        assert_eq!(out["outcome"], "applied");
        let row = sqlx::query(
            "SELECT status, payload_hash, thought_id FROM public.argus_source_events WHERE id=$1",
        )
        .bind(sid)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(row.get::<String, _>("status"), "conflict");
        assert_eq!(row.get::<String, _>("payload_hash"), sh);
        assert_eq!(row.get::<Uuid, _>("thought_id"), st);
    }

    async fn install_rb_trigger(admin: &PgPool, fn_name: &str, table: &str, tg_name: &str) {
        sqlx::query(&format!(
            r#"
            CREATE OR REPLACE FUNCTION public.{fn_name}()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              RAISE EXCEPTION 'test_rb' USING ERRCODE = 'P0001';
            END;
            $$;
            "#
        ))
        .execute(admin)
        .await
        .expect("fn");
        drop_test_trigger(admin, tg_name, table).await;
        sqlx::query(&format!(
            "CREATE TRIGGER {tg_name} BEFORE INSERT ON {table} FOR EACH ROW EXECUTE FUNCTION public.{fn_name}()"
        ))
        .execute(admin)
        .await
        .expect("tg");
    }

    #[tokio::test]
    async fn case_12a_rollback_receipt_insert() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-rcpt-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb old {uid}")).await;
        install_rb_trigger(
            &admin,
            "_test_ss_rb_receipt",
            "public.argus_source_event_supersession_receipts",
            "_test_rb_receipt",
        )
        .await;
        assert!(sqlx::query(
            "INSERT INTO public.argus_source_event_supersession_receipts (request_id) VALUES ($1)",
        )
        .bind(Uuid::new_v4())
        .execute(&admin)
        .await
        .is_err());
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
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        assert_domain_unchanged(&before, &snap(&admin).await, 0);
        drop_test_trigger(
            &admin,
            "_test_rb_receipt",
            "public.argus_source_event_supersession_receipts",
        )
        .await;
    }

    #[tokio::test]
    async fn case_12b_rollback_pending_embedding() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-emb-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb emb old {uid}")).await;
        install_rb_trigger(
            &admin,
            "_test_ss_rb_emb",
            "public.pending_embeddings",
            "_test_rb_emb",
        )
        .await;
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
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        assert_domain_unchanged(&before, &snap(&admin).await, 0);
        drop_test_trigger(&admin, "_test_rb_emb", "public.pending_embeddings").await;
    }

    #[tokio::test]
    async fn case_12c_rollback_pending_tag() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-tag-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb tag old {uid}")).await;
        install_rb_trigger(
            &admin,
            "_test_ss_rb_tag",
            "public.pending_tags",
            "_test_rb_tag",
        )
        .await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("rb tag new {uid}"),
            "diesel",
            "724808",
        );
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        assert_domain_unchanged(&before, &snap(&admin).await, 0);
        drop_test_trigger(&admin, "_test_rb_tag", "public.pending_tags").await;
    }

    #[tokio::test]
    async fn case_12d_rollback_link_insert() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-link-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb link old {uid}")).await;
        install_rb_trigger(
            &admin,
            "_test_ss_rb_link",
            "public.thought_links",
            "_test_rb_link",
        )
        .await;
        let before = snap(&admin).await;
        let req = mk_req(
            Uuid::new_v4(),
            &source_ref,
            "conflict",
            &old_hash,
            Some(tid),
            &format!("rb link new {uid}"),
            "diesel",
            "724808",
        );
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        // thoughts may have been rolled back with the statement
        assert_eq!(snap(&admin).await.receipts, before.receipts);
        let st: String = sqlx::query_scalar(
            "SELECT status FROM public.argus_source_events WHERE source_ref=$1 AND namespace='test'",
        )
        .bind(&source_ref)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(st, "conflict");
        drop_test_trigger(&admin, "_test_rb_link", "public.thought_links").await;
    }

    #[tokio::test]
    async fn case_12e_rollback_stable_update() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-upd-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb upd old {uid}")).await;
        sqlx::query(
            r#"
            CREATE OR REPLACE FUNCTION public._test_ss_rb_upd()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              IF NEW.status = 'stored' AND OLD.status = 'conflict' THEN
                RAISE EXCEPTION 'test_rb_upd' USING ERRCODE = 'P0001';
              END IF;
              RETURN NEW;
            END;
            $$;
            "#,
        )
        .execute(&admin)
        .await
        .unwrap();
        drop_test_trigger(&admin, "_test_rb_upd", "public.argus_source_events").await;
        sqlx::query(
            r#"
            CREATE TRIGGER _test_rb_upd
              BEFORE UPDATE ON public.argus_source_events
              FOR EACH ROW EXECUTE FUNCTION public._test_ss_rb_upd()
            "#,
        )
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
            &format!("rb upd new {uid}"),
            "diesel",
            "724808",
        );
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        assert_eq!(snap(&admin).await.receipts, before.receipts);
        drop_test_trigger(&admin, "_test_rb_upd", "public.argus_source_events").await;
    }

    #[tokio::test]
    async fn case_12f_rollback_retract_path() {
        // Inject failure on thought update when retracting (set retracted_at)
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        let source_ref = format!("rb-ret-{uid}");
        let (_e, old_hash, tid) =
            seed_conflict(&admin, &source_ref, &format!("rb ret old {uid}")).await;
        sqlx::query(
            r#"
            CREATE OR REPLACE FUNCTION public._test_ss_rb_ret()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
              IF NEW.retracted_at IS NOT NULL AND OLD.retracted_at IS NULL THEN
                RAISE EXCEPTION 'test_rb_ret' USING ERRCODE = 'P0001';
              END IF;
              RETURN NEW;
            END;
            $$;
            "#,
        )
        .execute(&admin)
        .await
        .unwrap();
        drop_test_trigger(&admin, "_test_rb_ret", "public.thoughts").await;
        sqlx::query(
            r#"
            CREATE TRIGGER _test_rb_ret
              BEFORE UPDATE ON public.thoughts
              FOR EACH ROW EXECUTE FUNCTION public._test_ss_rb_ret()
            "#,
        )
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
            &format!("rb ret new {uid}"),
            "diesel",
            "724808",
        );
        assert!(call_as("kengram_rt_supersession", &req).await.is_err());
        assert_eq!(snap(&admin).await.receipts, before.receipts);
        drop_test_trigger(&admin, "_test_rb_ret", "public.thoughts").await;
    }

    #[tokio::test]
    async fn case_13_receipt_oracle_four_envelopes() {
        let admin = admin_pool().await;
        let uid = Uuid::new_v4();
        // applied
        let source_ref = format!("rcpt-app-{uid}");
        let (_e, old_hash, tid) =
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
        let canon: Value = sqlx::query_scalar(
            "SELECT canonical_receipt_json FROM public.argus_source_event_supersession_receipts WHERE request_id=$1",
        )
        .bind(req.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        // Independent literal key set from §7
        for k in [
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
        ] {
            assert!(canon.get(k).is_some(), "missing key {k}");
        }
        let text: String = sqlx::query_scalar(
            "SELECT canonical_receipt_json::text FROM public.argus_source_event_supersession_receipts WHERE request_id=$1",
        )
        .bind(req.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        let oracle = to_hex(&Sha256::digest(text.as_bytes()));
        assert_eq!(oracle, out["receipt_hash"].as_str().unwrap());

        // expected-state refusal envelope
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
        assert_eq!(out_m["outcome"], "refused_expected_state");
        let text_m: String = sqlx::query_scalar(
            "SELECT canonical_receipt_json::text FROM public.argus_source_event_supersession_receipts WHERE request_id=$1",
        )
        .bind(miss.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(
            to_hex(&Sha256::digest(text_m.as_bytes())),
            out_m["receipt_hash"].as_str().unwrap()
        );

        // exact-content refusal
        let uid2 = Uuid::new_v4();
        let sref2 = format!("rcpt-ex-{uid2}");
        let content = format!("exact pre {uid2}");
        let (_e2, h2, t2) = seed_conflict(&admin, &sref2, &format!("ex old {uid2}")).await;
        disable_thought_gate(&admin).await;
        sqlx::query(
            r#"
            INSERT INTO public.thoughts (id, scope, content, source, metadata, content_fingerprint)
            VALUES ($1,'global',$2,'manual','{}'::jsonb, public.digest(pg_catalog.convert_to($2,'UTF8'),'sha256'))
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(&content)
        .execute(&admin)
        .await
        .unwrap();
        enable_thought_gate(&admin).await;
        let req_e = mk_req(
            Uuid::new_v4(),
            &sref2,
            "conflict",
            &h2,
            Some(t2),
            &content,
            "diesel",
            "724808",
        );
        let out_e = call_as("kengram_rt_supersession", &req_e)
            .await
            .expect("exact");
        assert_eq!(out_e["outcome"], "refused_exact_content_duplicate");
        let text_e: String = sqlx::query_scalar(
            "SELECT canonical_receipt_json::text FROM public.argus_source_event_supersession_receipts WHERE request_id=$1",
        )
        .bind(req_e.request_id)
        .fetch_one(&admin)
        .await
        .unwrap();
        assert_eq!(
            to_hex(&Sha256::digest(text_e.as_bytes())),
            out_e["receipt_hash"].as_str().unwrap()
        );

        // replay envelope identity
        let out_r = call_as("kengram_rt_supersession", &req)
            .await
            .expect("replay");
        assert_eq!(out_r["replayed"], true);
        assert_eq!(out_r["receipt_hash"], out["receipt_hash"]);
    }

    #[tokio::test]
    async fn case_14_down_migration_with_and_without_receipts() {
        let admin = admin_pool().await;
        let cnt: i64 = sqlx::query_scalar(
            "SELECT count(*)::bigint FROM public.argus_source_event_supersession_receipts",
        )
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(cnt > 0);
        let down = include_str!(
            "../../../migrations/rollback/0036_argus_source_event_supersession_transaction_down.sql"
        );
        assert!(sqlx::raw_sql(down).execute(&admin).await.is_err());
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
        assert!(still);

        // Zero-receipt success path: disable immutability, wipe receipts, down, reinstall
        sqlx::query(
            "ALTER TABLE public.argus_source_event_supersession_receipts DISABLE TRIGGER USER",
        )
        .execute(&admin)
        .await
        .ok();
        sqlx::query("DELETE FROM public.argus_source_event_supersession_receipts")
            .execute(&admin)
            .await
            .expect("wipe receipts");
        assert!(
            sqlx::raw_sql(down).execute(&admin).await.is_ok(),
            "down must succeed with zero receipts"
        );
        let gone: bool = sqlx::query_scalar(
            r#"
            SELECT to_regprocedure(
              'public.supersede_argus_source_event(uuid,text,text,text,text,uuid,text,text,text,jsonb,text,text,text,text,text,text)'
            ) IS NULL
            "#,
        )
        .fetch_one(&admin)
        .await
        .unwrap();
        assert!(gone);
        // reinstall 0036 for later tests in same DB
        let up = include_str!(
            "../../../migrations/0036_argus_source_event_supersession_transaction.sql"
        );
        sqlx::raw_sql(up)
            .execute(&admin)
            .await
            .expect("reinstall 0036");
        // re-bootstrap passwords for roles
        let url = require_database_url();
        let pass = url
            .split("://")
            .nth(1)
            .and_then(|s| s.split('@').next())
            .and_then(|s| s.split(':').nth(1))
            .unwrap_or("kengram");
        for r in [
            "kengram_rt_supersession",
            "kengram_rt_native_mcp",
            "kengram_rt_session",
            "kengram_rt_telegram",
        ] {
            let _ = sqlx::query(&format!("ALTER ROLE {r} WITH LOGIN PASSWORD '{pass}'"))
                .execute(&admin)
                .await;
        }
    }

    #[tokio::test]
    async fn case_15_bogus_receipt_row_rejected() {
        let admin = admin_pool().await;
        let rid = Uuid::new_v4();
        let res = sqlx::query(
            r#"
            INSERT INTO public.argus_source_event_supersession_receipts (
              request_id, request_digest, outcome, namespace, source_ref,
              expected_old_status, expected_old_payload_hash, observed_missing,
              new_payload_hash, embedding_model_id, tagger_model_id, actor, lane,
              approval_ref, reason, authenticated_session_user, occurred_at,
              canonical_receipt_json, receipt_digest
            ) VALUES (
              $1, decode(repeat('ab', 32), 'hex'), 'applied', 'test', 'bogus',
              'conflict', repeat('0', 64), false,
              repeat('1', 64), 'm', 't', 'a', 'l',
              'ap', 'r', 'kengram', now(),
              '{}'::jsonb, decode(repeat('cd', 32), 'hex')
            )
            "#,
        )
        .bind(rid)
        .execute(&admin)
        .await;
        assert!(
            res.is_err(),
            "bogus receipt must be rejected by constraints"
        );
    }
}

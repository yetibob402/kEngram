//! Portable migration-0030 security-boundary manifest and reconciliation.
//!
//! `pg_dump --no-owner --no-acl` cannot preserve cluster roles, ownership,
//! grants, or membership options.  This module treats those objects as one
//! exact-set contract, refuses backup on drift, and emits reviewed static SQL
//! that can repair a fresh restore before any writer starts.

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};

use crate::config::Config;

pub const SECURITY_MANIFEST_VERSION: u32 = 1;

pub const PRODUCER_ROLES: [&str; 11] = [
    "kengram_rt_native_mcp",
    "kengram_rt_session",
    "kengram_rt_telegram",
    "kengram_rt_agent_comms",
    "kengram_rt_reviews",
    "kengram_rt_specs",
    "kengram_rt_openclaw",
    "kengram_rt_hive",
    "kengram_rt_mba_archive",
    "kengram_rt_phase4",
    "kengram_rt_maintenance_import",
];

const HYGIENE_ROLES: [&str; 14] = [
    "kengram",
    "kengram_gate_owner",
    "kengram_runtime",
    "kengram_rt_native_mcp",
    "kengram_rt_session",
    "kengram_rt_telegram",
    "kengram_rt_agent_comms",
    "kengram_rt_reviews",
    "kengram_rt_specs",
    "kengram_rt_openclaw",
    "kengram_rt_hive",
    "kengram_rt_mba_archive",
    "kengram_rt_phase4",
    "kengram_rt_maintenance_import",
];

const SECURITY_FUNCTIONS: [&str; 7] = [
    "capture_thought_gated",
    "capture_thought_gated_passthrough",
    "mutate_thought_relations_serialized",
    "retract_thought_serialized",
    "lock_thought_relation_endpoints",
    "thoughts_require_gated_writer",
    "thought_links_require_serialized_writer",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusHygieneSecurityManifest {
    pub manifest_version: u32,
    pub migration_head: String,
    pub server_version: String,
    pub roles: serde_json::Value,
    pub memberships: serde_json::Value,
    pub producer_principals: serde_json::Value,
    pub producer_principals_sha256: String,
    pub functions: serde_json::Value,
    pub triggers: serde_json::Value,
    pub objects_and_acls: serde_json::Value,
    pub privilege_proof: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct SecurityPacket {
    pub manifest: CorpusHygieneSecurityManifest,
    pub manifest_json: String,
    pub manifest_sha256: String,
    pub reconcile_sql: String,
    pub reconcile_sha256: String,
}

pub fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes.as_ref())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub async fn is_installed(pool: &PgPool) -> anyhow::Result<bool> {
    let installed: bool = sqlx::query_scalar(
        "SELECT to_regclass('public.corpus_hygiene_producer_principals') IS NOT NULL",
    )
    .fetch_one(pool)
    .await?;
    Ok(installed)
}

pub async fn build_verified_packet(pool: &PgPool) -> anyhow::Result<SecurityPacket> {
    let roles = query_roles(pool).await?;
    verify_roles(&roles)?;
    let memberships = query_memberships(pool).await?;
    verify_memberships(&memberships)?;
    let principals = query_principals(pool).await?;
    verify_principals(&principals)?;
    let functions = query_functions(pool).await?;
    verify_functions(&functions)?;
    let triggers = query_triggers(pool).await?;
    verify_triggers(&triggers)?;
    let objects_and_acls = query_objects_and_acls(pool).await?;
    verify_objects_and_acls(&objects_and_acls)?;
    let privilege_proof = query_and_verify_privileges(pool).await?;

    let migration_head: String = sqlx::query_scalar(
        "SELECT version::text || '_' || description FROM _sqlx_migrations ORDER BY version DESC LIMIT 1",
    )
    .fetch_one(pool)
    .await
    .context("querying corpus-hygiene migration head")?;
    let head_version = migration_head
        .split_once('_')
        .and_then(|(version, _)| version.parse::<i64>().ok())
        .unwrap_or_default();
    if head_version < 30 {
        bail!("corpus-hygiene security backup requires migration 0030 head, got {migration_head}");
    }
    let server_version: String = sqlx::query_scalar("SHOW server_version")
        .fetch_one(pool)
        .await?;
    let principals_canonical = serde_json::to_vec(&principals)?;

    let manifest = CorpusHygieneSecurityManifest {
        manifest_version: SECURITY_MANIFEST_VERSION,
        migration_head,
        server_version,
        roles,
        memberships,
        producer_principals: principals,
        producer_principals_sha256: sha256_hex(principals_canonical),
        functions,
        triggers,
        objects_and_acls,
        privilege_proof,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    let reconcile_sql = RECONCILE_SQL.to_string();
    Ok(SecurityPacket {
        manifest_sha256: sha256_hex(manifest_json.as_bytes()),
        reconcile_sha256: sha256_hex(reconcile_sql.as_bytes()),
        manifest,
        manifest_json,
        reconcile_sql,
    })
}

pub async fn apply_reconciliation(pool: &PgPool, sql: &str) -> anyhow::Result<()> {
    if sha256_hex(sql.as_bytes()) != sha256_hex(RECONCILE_SQL.as_bytes()) {
        bail!("refusing non-canonical corpus-hygiene reconciliation SQL");
    }
    sqlx::raw_sql(sql)
        .execute(pool)
        .await
        .context("applying corpus-hygiene security reconciliation")?;
    build_verified_packet(pool).await?;
    Ok(())
}

pub async fn restore_gate_event(config: Config, gate_event: uuid::Uuid) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;
    let mut tx = pool.begin().await?;
    let row = sqlx::query(
        r#"
        SELECT action, scope, source, candidate_content, candidate_metadata,
               candidate_fingerprint, effective_created_at, restored_thought_id
        FROM public.thought_ingest_gate_events
        WHERE id = $1
        FOR UPDATE
        "#,
    )
    .bind(gate_event)
    .fetch_optional(&mut *tx)
    .await?
    .with_context(|| format!("gate event not found: {gate_event}"))?;

    let action: String = row.try_get("action")?;
    if action != "semantic_duplicate" {
        bail!("gate event {gate_event} is {action}, not a restorable semantic_duplicate");
    }
    let already_restored: Option<uuid::Uuid> = row.try_get("restored_thought_id")?;
    if let Some(thought_id) = already_restored {
        bail!("gate event {gate_event} was already restored as thought {thought_id}");
    }
    let content: Option<String> = row.try_get("candidate_content")?;
    let content = content.context("semantic duplicate gate event is missing recovery content")?;
    let scope: String = row.try_get("scope")?;
    let source: String = row.try_get("source")?;
    let metadata: serde_json::Value = row.try_get("candidate_metadata")?;
    let fingerprint: Vec<u8> = row.try_get("candidate_fingerprint")?;
    if fingerprint.len() != 32 {
        bail!("gate event {gate_event} has an invalid candidate fingerprint");
    }
    let payload_hash = fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let effective_created_at: time::OffsetDateTime = row.try_get("effective_created_at")?;
    let restore_ref = gate_event.to_string();
    let restore_metadata = serde_json::json!({
        "restored_from_gate_event": gate_event.to_string(),
        "operator_path": "kengram corpus-gate restore",
    });
    let bypass = serde_json::json!({
        "code": "operator_gate_restore",
        "gate_event_id": gate_event,
    });
    let tagger_model_id =
        (!config.tagger.provider.is_empty()).then(|| config.tagger.model_id.clone());

    let restored = sqlx::query(
        r#"
        SELECT thought_id, action
        FROM public.capture_thought_gated(
          $1, $2, $3, $4, $5,
          NULL::vector(1024), 'bge-m3:1024', NULL::integer, $6,
          'corpus-gate/restore', $7, $8, $9,
          '[]'::jsonb, $10, NULL::text, $11, $12
        )
        "#,
    )
    .bind(&scope)
    .bind(&content)
    .bind(&source)
    .bind(&metadata)
    .bind(effective_created_at)
    .bind(&bypass)
    .bind(&restore_ref)
    .bind(&payload_hash)
    .bind(&restore_metadata)
    .bind(tagger_model_id.as_deref())
    .bind(format!("restore:{gate_event}"))
    .bind(format!("restore:{gate_event}"))
    .fetch_one(&mut *tx)
    .await?;
    let restored_action: String = restored.try_get("action")?;
    if !matches!(
        restored_action.as_str(),
        "inserted" | "out_of_family_insert"
    ) {
        bail!(
            "gate event {gate_event} restore refused because the gate returned {restored_action}"
        );
    }
    let thought_id: uuid::Uuid = restored.try_get("thought_id")?;
    let updated = sqlx::query(
        r#"
        UPDATE public.thought_ingest_gate_events
        SET restored_thought_id = $2, restored_at = transaction_timestamp()
        WHERE id = $1 AND restored_thought_id IS NULL
        "#,
    )
    .bind(gate_event)
    .bind(thought_id)
    .execute(&mut *tx)
    .await?;
    if updated.rows_affected() != 1 {
        bail!("gate event {gate_event} restore lost its single-writer guard");
    }
    tx.commit().await?;
    println!("restored gate event {gate_event} as thought {thought_id}");
    Ok(())
}

async fn query_roles(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let rows = sqlx::query(
        r#"
        SELECT rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb,
               rolcanlogin, rolreplication, rolbypassrls
        FROM pg_roles
        WHERE rolname = ANY($1::text[])
        ORDER BY rolname
        "#,
    )
    .bind(HYGIENE_ROLES.to_vec())
    .fetch_all(pool)
    .await?;
    Ok(serde_json::Value::Array(
        rows.into_iter()
            .map(|row| {
                Ok(serde_json::json!({
                    "name": row.try_get::<String, _>("rolname")?,
                    "superuser": row.try_get::<bool, _>("rolsuper")?,
                    "inherit": row.try_get::<bool, _>("rolinherit")?,
                    "createrole": row.try_get::<bool, _>("rolcreaterole")?,
                    "createdb": row.try_get::<bool, _>("rolcreatedb")?,
                    "login": row.try_get::<bool, _>("rolcanlogin")?,
                    "replication": row.try_get::<bool, _>("rolreplication")?,
                    "bypassrls": row.try_get::<bool, _>("rolbypassrls")?,
                }))
            })
            .collect::<Result<Vec<_>, sqlx::Error>>()?,
    ))
}

fn verify_roles(value: &serde_json::Value) -> anyhow::Result<()> {
    let rows = value.as_array().context("role manifest is not an array")?;
    if rows.len() != HYGIENE_ROLES.len() {
        bail!(
            "corpus-hygiene role drift: expected {} roles, got {}",
            HYGIENE_ROLES.len(),
            rows.len()
        );
    }
    for row in rows {
        let name = row["name"].as_str().context("role name missing")?;
        let is_gate_or_runtime = matches!(name, "kengram_gate_owner" | "kengram_runtime");
        let is_break_glass = name == "kengram";
        let expected_login = !is_gate_or_runtime;
        if row["login"] != expected_login
            || row["replication"] != false
            || (!is_break_glass
                && (row["superuser"] != false
                    || row["createrole"] != false
                    || row["createdb"] != false
                    || row["bypassrls"] != false))
            || (is_gate_or_runtime && row["inherit"] != false)
            || (!is_gate_or_runtime && row["inherit"] != true)
        {
            bail!("unsafe corpus-hygiene role attributes for {name}: {row}");
        }
        if is_break_glass
            && (row["superuser"] != true || row["createrole"] != true || row["createdb"] != true)
        {
            bail!("retained kengram break-glass role lost admin attributes: {row}");
        }
    }
    Ok(())
}

async fn query_memberships(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let rows = sqlx::query(
        r#"
        SELECT parent.rolname AS parent_name, member.rolname AS member_name,
               m.admin_option, m.inherit_option, m.set_option
        FROM pg_auth_members m
        JOIN pg_roles parent ON parent.oid = m.roleid
        JOIN pg_roles member ON member.oid = m.member
        WHERE parent.rolname = ANY($1::text[]) OR member.rolname = ANY($1::text[])
        ORDER BY parent.rolname, member.rolname
        "#,
    )
    .bind(HYGIENE_ROLES[1..].to_vec())
    .fetch_all(pool)
    .await?;
    Ok(serde_json::Value::Array(
        rows.into_iter()
            .map(|row| {
                Ok(serde_json::json!({
                    "parent": row.try_get::<String, _>("parent_name")?,
                    "member": row.try_get::<String, _>("member_name")?,
                    "admin": row.try_get::<bool, _>("admin_option")?,
                    "inherit": row.try_get::<bool, _>("inherit_option")?,
                    "set": row.try_get::<bool, _>("set_option")?,
                }))
            })
            .collect::<Result<Vec<_>, sqlx::Error>>()?,
    ))
}

fn verify_memberships(value: &serde_json::Value) -> anyhow::Result<()> {
    let rows = value
        .as_array()
        .context("membership manifest is not an array")?;
    if rows.len() != PRODUCER_ROLES.len() {
        bail!(
            "corpus-hygiene membership drift: expected {} edges, got {}: {value}",
            PRODUCER_ROLES.len(),
            rows.len()
        );
    }
    for expected_member in PRODUCER_ROLES {
        let row = rows
            .iter()
            .find(|row| row["member"] == expected_member)
            .with_context(|| format!("missing runtime membership for {expected_member}"))?;
        if row["parent"] != "kengram_runtime"
            || row["admin"] != false
            || row["inherit"] != true
            || row["set"] != false
        {
            bail!("corpus-hygiene membership drift: {row}");
        }
    }
    Ok(())
}

async fn query_principals(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let rows = sqlx::query(
        r#"
        SELECT principal_name::text, producer_class, profile_revision, enabled,
               requires_source_created_at, keep_only, enforce_eligible, relation_allowed
        FROM corpus_hygiene_producer_principals
        ORDER BY principal_name
        "#,
    )
    .fetch_all(pool)
    .await?;
    Ok(serde_json::Value::Array(
        rows.into_iter()
            .map(|row| {
                Ok(serde_json::json!({
                    "principal": row.try_get::<String, _>("principal_name")?,
                    "producer_class": row.try_get::<String, _>("producer_class")?,
                    "profile_revision": row.try_get::<i32, _>("profile_revision")?,
                    "enabled": row.try_get::<bool, _>("enabled")?,
                    "requires_source_created_at": row.try_get::<bool, _>("requires_source_created_at")?,
                    "keep_only": row.try_get::<bool, _>("keep_only")?,
                    "enforce_eligible": row.try_get::<bool, _>("enforce_eligible")?,
                    "relation_allowed": row.try_get::<bool, _>("relation_allowed")?,
                }))
            })
            .collect::<Result<Vec<_>, sqlx::Error>>()?,
    ))
}

fn verify_principals(value: &serde_json::Value) -> anyhow::Result<()> {
    let rows = value.as_array().context("principal map is not an array")?;
    if rows.len() != 12 {
        bail!(
            "producer-principal map drift: expected 12 rows, got {}",
            rows.len()
        );
    }
    if rows
        .iter()
        .any(|row| row["enabled"] != true || row["profile_revision"] != 1)
    {
        bail!("producer-principal map contains disabled or wrong-revision row: {value}");
    }
    let expected = [
        (
            "kengram",
            "break_glass_passthrough",
            true,
            true,
            false,
            true,
        ),
        (
            "kengram_rt_agent_comms",
            "agent_comms_realtime",
            false,
            false,
            true,
            false,
        ),
        (
            "kengram_rt_hive",
            "hive_historical",
            true,
            false,
            true,
            true,
        ),
        (
            "kengram_rt_maintenance_import",
            "maintenance_historical_keep_only",
            true,
            true,
            false,
            true,
        ),
        (
            "kengram_rt_mba_archive",
            "mba_archive_historical",
            true,
            false,
            true,
            false,
        ),
        (
            "kengram_rt_native_mcp",
            "native_mcp",
            false,
            true,
            false,
            true,
        ),
        (
            "kengram_rt_openclaw",
            "openclaw_historical",
            true,
            false,
            true,
            false,
        ),
        (
            "kengram_rt_phase4",
            "phase4_derived",
            true,
            false,
            true,
            true,
        ),
        (
            "kengram_rt_reviews",
            "review_historical",
            true,
            false,
            true,
            false,
        ),
        (
            "kengram_rt_session",
            "session_realtime",
            false,
            false,
            true,
            false,
        ),
        (
            "kengram_rt_specs",
            "spec_historical",
            true,
            false,
            true,
            false,
        ),
        (
            "kengram_rt_telegram",
            "telegram_realtime",
            false,
            false,
            true,
            false,
        ),
    ];
    for (row, (principal, class, historical, keep_only, enforce, relations)) in
        rows.iter().zip(expected)
    {
        if row["principal"] != principal
            || row["producer_class"] != class
            || row["requires_source_created_at"] != historical
            || row["keep_only"] != keep_only
            || row["enforce_eligible"] != enforce
            || row["relation_allowed"] != relations
        {
            bail!("producer-principal profile drift for {principal}->{class}: {row}");
        }
    }
    Ok(())
}

async fn query_functions(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let rows = sqlx::query(
        r#"
        SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS identity_arguments,
               owner.rolname AS owner, p.prosecdef,
               COALESCE(to_jsonb(p.proconfig), '[]'::jsonb) AS proconfig,
               COALESCE(to_jsonb(p.proacl), '[]'::jsonb) AS proacl,
               encode(digest(pg_get_functiondef(p.oid), 'sha256'), 'hex') AS definition_sha256
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        JOIN pg_roles owner ON owner.oid = p.proowner
        WHERE n.nspname = 'public' AND p.proname = ANY($1::text[])
        ORDER BY p.proname, identity_arguments
        "#,
    )
    .bind(SECURITY_FUNCTIONS.to_vec())
    .fetch_all(pool)
    .await?;
    Ok(serde_json::Value::Array(
        rows.into_iter()
            .map(|row| {
                Ok(serde_json::json!({
                    "name": row.try_get::<String, _>("proname")?,
                    "identity_arguments": row.try_get::<String, _>("identity_arguments")?,
                    "owner": row.try_get::<String, _>("owner")?,
                    "security_definer": row.try_get::<bool, _>("prosecdef")?,
                    "proconfig": row.try_get::<serde_json::Value, _>("proconfig")?,
                    "acl": row.try_get::<serde_json::Value, _>("proacl")?,
                    "definition_sha256": row.try_get::<String, _>("definition_sha256")?,
                }))
            })
            .collect::<Result<Vec<_>, sqlx::Error>>()?,
    ))
}

fn verify_functions(value: &serde_json::Value) -> anyhow::Result<()> {
    let rows = value
        .as_array()
        .context("function manifest is not an array")?;
    if rows.len() != SECURITY_FUNCTIONS.len() {
        bail!(
            "security function drift: expected {} rows, got {}",
            SECURITY_FUNCTIONS.len(),
            rows.len()
        );
    }
    for row in rows {
        let name = row["name"].as_str().unwrap_or_default();
        let trigger = name.ends_with("gated_writer") || name.ends_with("serialized_writer");
        if row["owner"] != "kengram_gate_owner"
            || row["security_definer"] != !trigger
            || (!trigger
                && !row["proconfig"]
                    .to_string()
                    .contains("search_path=pg_catalog, public"))
        {
            bail!("security function owner/definer/search_path drift: {row}");
        }
    }
    Ok(())
}

async fn query_triggers(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let rows = sqlx::query(
        r#"
        SELECT t.tgname, c.relname AS table_name, t.tgenabled::text AS tgenabled,
               p.proname AS function_name,
               encode(digest(pg_get_triggerdef(t.oid, true), 'sha256'), 'hex') AS definition_sha256
        FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_proc p ON p.oid = t.tgfoid
        WHERE n.nspname = 'public'
          AND t.tgname IN ('thoughts_require_gated_writer','thought_links_require_serialized_writer')
        ORDER BY t.tgname
        "#,
    )
    .fetch_all(pool)
    .await?;
    Ok(serde_json::Value::Array(
        rows.into_iter()
            .map(|row| {
                Ok(serde_json::json!({
                    "name": row.try_get::<String, _>("tgname")?,
                    "table": row.try_get::<String, _>("table_name")?,
                    "enabled": row.try_get::<String, _>("tgenabled")?,
                    "function": row.try_get::<String, _>("function_name")?,
                    "definition_sha256": row.try_get::<String, _>("definition_sha256")?,
                }))
            })
            .collect::<Result<Vec<_>, sqlx::Error>>()?,
    ))
}

fn verify_triggers(value: &serde_json::Value) -> anyhow::Result<()> {
    let rows = value
        .as_array()
        .context("trigger manifest is not an array")?;
    if rows.len() != 2 || rows.iter().any(|row| row["enabled"] != "O") {
        bail!("both corpus-hygiene triggers must be enabled before backup: {value}");
    }
    Ok(())
}

async fn query_objects_and_acls(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let rows: serde_json::Value = sqlx::query_scalar(
        r#"
        SELECT COALESCE(jsonb_agg(object_row ORDER BY object_row->>'kind', object_row->>'name'), '[]'::jsonb)
        FROM (
          SELECT jsonb_build_object(
            'kind', CASE c.relkind WHEN 'S' THEN 'sequence' ELSE 'table' END,
            'name', c.relname,
            'owner', owner.rolname,
            'acl', COALESCE(to_jsonb(c.relacl), '[]'::jsonb)
          ) AS object_row
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          JOIN pg_roles owner ON owner.oid = c.relowner
          WHERE n.nspname = 'public'
            AND c.relname IN (
              'thoughts','thought_links','argus_source_events','pending_embeddings','pending_tags',
              'thought_embeddings_bge_m3','corpus_hygiene_producer_principals',
              'corpus_hygiene_gate_settings','thought_ingest_gate_events',
              'thought_relation_request_events'
            )
          UNION ALL
          SELECT jsonb_build_object(
            'kind','column','name',c.relname || '.' || a.attname,
            'owner',owner.rolname,'acl',to_jsonb(a.attacl)
          ) AS object_row
          FROM pg_attribute a
          JOIN pg_class c ON c.oid = a.attrelid
          JOIN pg_namespace n ON n.oid = c.relnamespace
          JOIN pg_roles owner ON owner.oid = c.relowner
          WHERE n.nspname = 'public' AND a.attacl IS NOT NULL
            AND c.relname IN (
              'thoughts','thought_links','argus_source_events','pending_embeddings','pending_tags',
              'thought_embeddings_bge_m3','corpus_hygiene_producer_principals',
              'corpus_hygiene_gate_settings','thought_ingest_gate_events',
              'thought_relation_request_events'
            )
          UNION ALL
          SELECT jsonb_build_object(
            'kind','schema','name','public','owner',owner.rolname,
            'acl',COALESCE(to_jsonb(n.nspacl), '[]'::jsonb)
          ) AS object_row
          FROM pg_namespace n JOIN pg_roles owner ON owner.oid=n.nspowner
          WHERE n.nspname='public'
        ) objects
        "#,
    )
    .fetch_one(pool)
    .await?;
    Ok(rows)
}

fn verify_objects_and_acls(value: &serde_json::Value) -> anyhow::Result<()> {
    let rows = value
        .as_array()
        .context("security object manifest is not an array")?;
    let expected_tables = [
        "argus_source_events",
        "corpus_hygiene_gate_settings",
        "corpus_hygiene_producer_principals",
        "pending_embeddings",
        "pending_tags",
        "thought_embeddings_bge_m3",
        "thought_ingest_gate_events",
        "thought_links",
        "thought_relation_request_events",
        "thoughts",
    ];
    for table in expected_tables {
        let row = rows
            .iter()
            .find(|row| row["kind"] == "table" && row["name"] == table)
            .with_context(|| format!("security object missing from manifest: {table}"))?;
        if row["owner"] != "kengram" {
            bail!("security object owner drift for {table}: {row}");
        }
    }

    let schema = rows
        .iter()
        .find(|row| row["kind"] == "schema" && row["name"] == "public")
        .context("public schema missing from security manifest")?;
    if schema["owner"] != "pg_database_owner" {
        bail!("public schema owner drift: {schema}");
    }

    let expected_columns = [
        "thoughts.retracted_at",
        "thoughts.retracted_reason",
        "thoughts.tags",
        "thoughts.tags_extracted_at",
        "thoughts.tags_extractor_model",
        "thoughts.tags_extractor_version",
    ];
    let actual_columns = rows
        .iter()
        .filter(|row| row["kind"] == "column")
        .map(|row| row["name"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    if actual_columns != expected_columns {
        bail!("column ACL object drift: expected {expected_columns:?}, got {actual_columns:?}");
    }
    if rows.len() != expected_tables.len() + expected_columns.len() + 1 {
        bail!("unexpected object in corpus-hygiene security manifest: {value}");
    }
    Ok(())
}

async fn query_and_verify_exact_acls(pool: &PgPool) -> anyhow::Result<Vec<String>> {
    let actual = sqlx::query_scalar::<_, String>(
        r#"
        SELECT acl_key FROM (
          SELECT 'table:' || c.relname || ':' || COALESCE(grantee.rolname,'PUBLIC') || ':' ||
                 COALESCE(grantor.rolname,'PUBLIC') || ':' || x.privilege_type || ':' || x.is_grantable AS acl_key
          FROM pg_class c
          JOIN pg_namespace n ON n.oid=c.relnamespace
          CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) x
          LEFT JOIN pg_roles grantee ON grantee.oid=x.grantee
          LEFT JOIN pg_roles grantor ON grantor.oid=x.grantor
          WHERE n.nspname='public'
            AND c.relname IN ('thoughts','thought_links','argus_source_events','pending_embeddings','pending_tags','thought_embeddings_bge_m3','corpus_hygiene_producer_principals','corpus_hygiene_gate_settings','thought_ingest_gate_events','thought_relation_request_events')
            AND x.grantee<>c.relowner
          UNION ALL
          SELECT 'column:' || c.relname || '.' || a.attname || ':' || COALESCE(grantee.rolname,'PUBLIC') || ':' ||
                 COALESCE(grantor.rolname,'PUBLIC') || ':' || x.privilege_type || ':' || x.is_grantable AS acl_key
          FROM pg_attribute a
          JOIN pg_class c ON c.oid=a.attrelid
          JOIN pg_namespace n ON n.oid=c.relnamespace
          CROSS JOIN LATERAL aclexplode(a.attacl) x
          LEFT JOIN pg_roles grantee ON grantee.oid=x.grantee
          LEFT JOIN pg_roles grantor ON grantor.oid=x.grantor
          WHERE n.nspname='public' AND a.attacl IS NOT NULL
            AND c.relname IN ('thoughts','thought_links','argus_source_events','pending_embeddings','pending_tags','thought_embeddings_bge_m3','corpus_hygiene_producer_principals','corpus_hygiene_gate_settings','thought_ingest_gate_events','thought_relation_request_events')
            AND x.grantee<>c.relowner
          UNION ALL
          SELECT 'function:' || p.proname || ':' || COALESCE(grantee.rolname,'PUBLIC') || ':' ||
                 COALESCE(grantor.rolname,'PUBLIC') || ':' || x.privilege_type || ':' || x.is_grantable AS acl_key
          FROM pg_proc p
          JOIN pg_namespace n ON n.oid=p.pronamespace
          CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) x
          LEFT JOIN pg_roles grantee ON grantee.oid=x.grantee
          LEFT JOIN pg_roles grantor ON grantor.oid=x.grantor
          WHERE n.nspname='public'
            AND p.proname IN ('capture_thought_gated','capture_thought_gated_passthrough','mutate_thought_relations_serialized','retract_thought_serialized','lock_thought_relation_endpoints','thoughts_require_gated_writer','thought_links_require_serialized_writer')
            AND x.grantee<>p.proowner
          UNION ALL
          SELECT 'schema:' || n.nspname || ':' || COALESCE(grantee.rolname,'PUBLIC') || ':' ||
                 COALESCE(grantor.rolname,'PUBLIC') || ':' || x.privilege_type || ':' || x.is_grantable AS acl_key
          FROM pg_namespace n
          CROSS JOIN LATERAL aclexplode(COALESCE(n.nspacl,acldefault('n',n.nspowner))) x
          LEFT JOIN pg_roles grantee ON grantee.oid=x.grantee
          LEFT JOIN pg_roles grantor ON grantor.oid=x.grantor
          WHERE n.nspname='public' AND x.grantee<>n.nspowner
        ) acl_rows
        ORDER BY acl_key
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut expected = Vec::new();
    let mut add = |kind: &str, object: &str, grantee: &str, grantor: &str, privileges: &[&str]| {
        for privilege in privileges {
            expected.push(format!(
                "{kind}:{object}:{grantee}:{grantor}:{privilege}:false"
            ));
        }
    };
    add(
        "table",
        "argus_source_events",
        "kengram_gate_owner",
        "kengram",
        &["INSERT", "SELECT", "UPDATE"],
    );
    add(
        "table",
        "argus_source_events",
        "kengram_runtime",
        "kengram",
        &["SELECT"],
    );
    for table in [
        "corpus_hygiene_gate_settings",
        "corpus_hygiene_producer_principals",
    ] {
        add("table", table, "kengram_gate_owner", "kengram", &["SELECT"]);
    }
    for table in [
        "pending_embeddings",
        "pending_tags",
        "thought_embeddings_bge_m3",
    ] {
        add(
            "table",
            table,
            "kengram_gate_owner",
            "kengram",
            &["INSERT", "SELECT"],
        );
        add(
            "table",
            table,
            "kengram_runtime",
            "kengram",
            &["DELETE", "INSERT", "SELECT", "UPDATE"],
        );
    }
    for table in [
        "thought_ingest_gate_events",
        "thought_relation_request_events",
    ] {
        add(
            "table",
            table,
            "kengram_gate_owner",
            "kengram",
            &["INSERT", "SELECT", "UPDATE"],
        );
    }
    add(
        "table",
        "thought_links",
        "kengram_gate_owner",
        "kengram",
        &["DELETE", "INSERT", "SELECT", "UPDATE"],
    );
    add(
        "table",
        "thought_links",
        "kengram_runtime",
        "kengram",
        &["SELECT"],
    );
    add(
        "table",
        "thoughts",
        "kengram_gate_owner",
        "kengram",
        &["INSERT", "SELECT"],
    );
    add(
        "table",
        "thoughts",
        "kengram_runtime",
        "kengram",
        &["SELECT"],
    );
    for column in ["thoughts.retracted_at", "thoughts.retracted_reason"] {
        add(
            "column",
            column,
            "kengram_gate_owner",
            "kengram",
            &["UPDATE"],
        );
    }
    for column in [
        "thoughts.tags",
        "thoughts.tags_extracted_at",
        "thoughts.tags_extractor_model",
        "thoughts.tags_extractor_version",
    ] {
        add("column", column, "kengram_runtime", "kengram", &["UPDATE"]);
    }
    for function in [
        "capture_thought_gated",
        "mutate_thought_relations_serialized",
        "retract_thought_serialized",
    ] {
        add(
            "function",
            function,
            "kengram_runtime",
            "kengram_gate_owner",
            &["EXECUTE"],
        );
    }
    add(
        "schema",
        "public",
        "PUBLIC",
        "pg_database_owner",
        &["USAGE"],
    );
    expected.sort();
    if actual != expected {
        bail!("exact non-owner ACL drift: expected {expected:?}, got {actual:?}");
    }
    Ok(actual)
}

async fn query_and_verify_privileges(pool: &PgPool) -> anyhow::Result<serde_json::Value> {
    let exact_non_owner_acls = query_and_verify_exact_acls(pool).await?;
    let mut rows = Vec::new();
    for role in PRODUCER_ROLES {
        let row = sqlx::query(
            r#"
            SELECT
              has_table_privilege($1, 'public.thoughts', 'INSERT') AS thought_insert,
              has_column_privilege($1, 'public.thoughts', 'retracted_at', 'UPDATE') AS retract_update,
              has_table_privilege($1, 'public.thought_links', 'INSERT,UPDATE,DELETE,TRUNCATE') AS relation_write,
              has_function_privilege($1, 'public.capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text)', 'EXECUTE') AS gate_execute,
              has_function_privilege($1, 'public.mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text)', 'EXECUTE') AS relation_execute,
              pg_has_role($1, 'kengram_gate_owner', 'SET') AS can_set_gate_owner,
              pg_has_role($1, 'kengram_runtime', 'SET') AS can_set_runtime
            "#,
        )
        .bind(role)
        .fetch_one(pool)
        .await?;
        let proof = serde_json::json!({
            "role": role,
            "thought_insert": row.try_get::<bool, _>("thought_insert")?,
            "retract_update": row.try_get::<bool, _>("retract_update")?,
            "relation_write": row.try_get::<bool, _>("relation_write")?,
            "gate_execute": row.try_get::<bool, _>("gate_execute")?,
            "relation_execute": row.try_get::<bool, _>("relation_execute")?,
            "can_set_gate_owner": row.try_get::<bool, _>("can_set_gate_owner")?,
            "can_set_runtime": row.try_get::<bool, _>("can_set_runtime")?,
        });
        if proof["thought_insert"] != false
            || proof["retract_update"] != false
            || proof["relation_write"] != false
            || proof["gate_execute"] != true
            || proof["relation_execute"] != true
            || proof["can_set_gate_owner"] != false
            || proof["can_set_runtime"] != false
        {
            bail!("runtime privilege drift for {role}: {proof}");
        }
        rows.push(proof);
    }
    let public_create: bool =
        sqlx::query_scalar("SELECT has_schema_privilege('public','public','CREATE')")
            .fetch_one(pool)
            .await?;
    if public_create {
        bail!("PUBLIC unexpectedly has CREATE on schema public");
    }
    Ok(serde_json::json!({
        "runtime_roles": rows,
        "public_schema_create": public_create,
        "exact_non_owner_acls": exact_non_owner_acls,
    }))
}

/// Static, reviewed reconciliation source.  It is intentionally generated
/// from the migration contract rather than from dump text.
pub const RECONCILE_SQL: &str = r#"
BEGIN;
DO $roles$
DECLARE r text;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='kengram_gate_owner') THEN CREATE ROLE kengram_gate_owner NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='kengram_runtime') THEN CREATE ROLE kengram_runtime NOLOGIN; END IF;
  ALTER ROLE kengram_gate_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
  ALTER ROLE kengram_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
  FOREACH r IN ARRAY ARRAY['kengram_rt_native_mcp','kengram_rt_session','kengram_rt_telegram','kengram_rt_agent_comms','kengram_rt_reviews','kengram_rt_specs','kengram_rt_openclaw','kengram_rt_hive','kengram_rt_mba_archive','kengram_rt_phase4','kengram_rt_maintenance_import'] LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname=r) THEN EXECUTE format('CREATE ROLE %I LOGIN',r); END IF;
    EXECUTE format('ALTER ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT NOREPLICATION NOBYPASSRLS',r);
  END LOOP;
END $roles$;
DO $memberships$
DECLARE e record;
BEGIN
  FOR e IN
    SELECT p.rolname parent_name, m.rolname member_name
    FROM pg_auth_members a JOIN pg_roles p ON p.oid=a.roleid JOIN pg_roles m ON m.oid=a.member
    WHERE p.rolname = ANY(ARRAY['kengram_gate_owner','kengram_runtime','kengram_rt_native_mcp','kengram_rt_session','kengram_rt_telegram','kengram_rt_agent_comms','kengram_rt_reviews','kengram_rt_specs','kengram_rt_openclaw','kengram_rt_hive','kengram_rt_mba_archive','kengram_rt_phase4','kengram_rt_maintenance_import'])
       OR m.rolname = ANY(ARRAY['kengram_gate_owner','kengram_runtime','kengram_rt_native_mcp','kengram_rt_session','kengram_rt_telegram','kengram_rt_agent_comms','kengram_rt_reviews','kengram_rt_specs','kengram_rt_openclaw','kengram_rt_hive','kengram_rt_mba_archive','kengram_rt_phase4','kengram_rt_maintenance_import'])
  LOOP EXECUTE format('REVOKE %I FROM %I',e.parent_name,e.member_name); END LOOP;
END $memberships$;
GRANT kengram_runtime TO kengram_rt_native_mcp,kengram_rt_session,kengram_rt_telegram,kengram_rt_agent_comms,kengram_rt_reviews,kengram_rt_specs,kengram_rt_openclaw,kengram_rt_hive,kengram_rt_mba_archive,kengram_rt_phase4,kengram_rt_maintenance_import WITH INHERIT TRUE, SET FALSE;
DELETE FROM corpus_hygiene_gate_settings;
DELETE FROM corpus_hygiene_producer_principals;
INSERT INTO corpus_hygiene_producer_principals
  (principal_name,producer_class,profile_revision,enabled,requires_source_created_at,keep_only,enforce_eligible,relation_allowed)
VALUES
  ('kengram','break_glass_passthrough',1,true,true,true,false,true),
  ('kengram_rt_native_mcp','native_mcp',1,true,false,true,false,true),
  ('kengram_rt_session','session_realtime',1,true,false,false,true,false),
  ('kengram_rt_telegram','telegram_realtime',1,true,false,false,true,false),
  ('kengram_rt_agent_comms','agent_comms_realtime',1,true,false,false,true,false),
  ('kengram_rt_reviews','review_historical',1,true,true,false,true,false),
  ('kengram_rt_specs','spec_historical',1,true,true,false,true,false),
  ('kengram_rt_openclaw','openclaw_historical',1,true,true,false,true,false),
  ('kengram_rt_hive','hive_historical',1,true,true,false,true,true),
  ('kengram_rt_mba_archive','mba_archive_historical',1,true,true,false,true,false),
  ('kengram_rt_phase4','phase4_derived',1,true,true,false,true,true),
  ('kengram_rt_maintenance_import','maintenance_historical_keep_only',1,true,true,true,false,true);
INSERT INTO corpus_hygiene_gate_settings(principal_name,producer_class,profile_revision,mode)
SELECT principal_name,producer_class,profile_revision,'off'
FROM corpus_hygiene_producer_principals;
ALTER TABLE corpus_hygiene_producer_principals OWNER TO kengram;
ALTER TABLE corpus_hygiene_gate_settings OWNER TO kengram;
ALTER TABLE thought_ingest_gate_events OWNER TO kengram;
ALTER TABLE thought_relation_request_events OWNER TO kengram;
ALTER TABLE thoughts OWNER TO kengram;
ALTER TABLE thought_links OWNER TO kengram;
ALTER TABLE argus_source_events OWNER TO kengram;
ALTER TABLE pending_embeddings OWNER TO kengram;
ALTER TABLE pending_tags OWNER TO kengram;
ALTER TABLE thought_embeddings_bge_m3 OWNER TO kengram;
ALTER SCHEMA public OWNER TO pg_database_owner;
ALTER FUNCTION thoughts_require_gated_writer() OWNER TO kengram_gate_owner;
ALTER FUNCTION thought_links_require_serialized_writer() OWNER TO kengram_gate_owner;
ALTER FUNCTION lock_thought_relation_endpoints(uuid[],boolean) OWNER TO kengram_gate_owner;
ALTER FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION retract_thought_serialized(uuid,text,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION capture_thought_gated_passthrough(text,text,text,jsonb,timestamptz,text,jsonb,text,text,text,jsonb,jsonb,text,text,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION thoughts_require_gated_writer() SECURITY INVOKER SET search_path = pg_catalog, public;
ALTER FUNCTION thought_links_require_serialized_writer() SECURITY INVOKER SET search_path = pg_catalog, public;
ALTER FUNCTION lock_thought_relation_endpoints(uuid[],boolean) SECURITY DEFINER SET search_path = pg_catalog, public;
ALTER FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) SECURITY DEFINER SET search_path = pg_catalog, public;
ALTER FUNCTION retract_thought_serialized(uuid,text,text) SECURITY DEFINER SET search_path = pg_catalog, public;
ALTER FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) SECURITY DEFINER SET search_path = pg_catalog, public;
ALTER FUNCTION capture_thought_gated_passthrough(text,text,text,jsonb,timestamptz,text,jsonb,text,text,text,jsonb,jsonb,text,text,text) SECURITY DEFINER SET search_path = pg_catalog, public;
REVOKE INSERT ON thoughts FROM PUBLIC,kengram_runtime;
REVOKE UPDATE(retracted_at,retracted_reason) ON thoughts FROM PUBLIC,kengram_runtime;
REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON thought_links FROM PUBLIC,kengram_runtime;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON FUNCTION lock_thought_relation_endpoints(uuid[],boolean) FROM PUBLIC,kengram_runtime;
REVOKE ALL ON FUNCTION thoughts_require_gated_writer() FROM PUBLIC;
REVOKE ALL ON FUNCTION thought_links_require_serialized_writer() FROM PUBLIC;
REVOKE ALL ON FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION capture_thought_gated_passthrough(text,text,text,jsonb,timestamptz,text,jsonb,text,text,text,jsonb,jsonb,text,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION retract_thought_serialized(uuid,text,text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) TO kengram_runtime;
GRANT EXECUTE ON FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) TO kengram_runtime;
GRANT EXECUTE ON FUNCTION retract_thought_serialized(uuid,text,text) TO kengram_runtime;
GRANT SELECT ON corpus_hygiene_producer_principals,corpus_hygiene_gate_settings TO kengram_gate_owner;
GRANT SELECT,INSERT ON thoughts TO kengram_gate_owner;
GRANT UPDATE(retracted_at,retracted_reason) ON thoughts TO kengram_gate_owner;
GRANT SELECT,INSERT,UPDATE ON argus_source_events,thought_ingest_gate_events,thought_relation_request_events TO kengram_gate_owner;
GRANT SELECT,INSERT ON thought_embeddings_bge_m3,pending_embeddings,pending_tags TO kengram_gate_owner;
GRANT SELECT,INSERT,UPDATE,DELETE ON thought_links TO kengram_gate_owner;
GRANT SELECT ON thoughts,thought_links,argus_source_events TO kengram_runtime;
GRANT SELECT,INSERT,UPDATE,DELETE ON pending_embeddings,pending_tags,thought_embeddings_bge_m3 TO kengram_runtime;
GRANT UPDATE(tags,tags_extractor_model,tags_extractor_version,tags_extracted_at) ON thoughts TO kengram_runtime;
ALTER TABLE thoughts ENABLE TRIGGER thoughts_require_gated_writer;
ALTER TABLE thought_links ENABLE TRIGGER thought_links_require_serialized_writer;
COMMIT;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_manifest_constants_have_exact_cardinality() {
        assert_eq!(PRODUCER_ROLES.len(), 11);
        assert_eq!(HYGIENE_ROLES.len(), 14);
    }

    #[test]
    fn reconciliation_never_disables_triggers_or_grants_direct_writes() {
        let normalized = RECONCILE_SQL.to_ascii_lowercase();
        assert!(!normalized.contains("disable trigger"));
        assert!(!normalized.contains("grant insert on thoughts"));
        assert!(!normalized.contains("grant insert on thought_links"));
        assert!(normalized.contains("enable trigger thoughts_require_gated_writer"));
        assert!(normalized.contains("enable trigger thought_links_require_serialized_writer"));
        assert!(normalized.contains("set false"));
    }
}

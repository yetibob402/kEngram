//! `kengram backup` / `kengram restore` — portable, manifest-validated
//! machine-to-machine corpus migration. M7 operator subcommands.
//!
//! Both commands shell out to system `pg_dump` / `pg_restore`; kengram does
//! not reimplement them. The value-add over a bare `pg_dump | pg_restore`
//! pipeline is the `manifest.json` sidecar (kengram version, schema head,
//! embedder model, tagger version, corpus counts) that travels in the same
//! tarball — restore validates it against the target's state before doing
//! anything destructive.
//!
//! See DEVELOPMENT.md "Migrating between machines" for the operator
//! walkthrough.

use std::{fs::File, io::Read, path::PathBuf, process::Stdio};

use anyhow::{Context, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPoolOptions;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::process::Command;

use crate::config::Config;
use crate::corpus_hygiene_security::{self, CorpusHygieneSecurityManifest, SecurityPacket};

/// Schema version of the manifest itself. Bump only on incompatible shape
/// changes (added fields are backward-compatible via serde defaults;
/// removed or renamed fields aren't). Restore refuses on a higher value
/// than this binary understands.
pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub manifest_version: u32,
    pub kengram_version: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    pub source_database: String,
    pub schema: SchemaInfo,
    pub embedder: EmbedderInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tagger: Option<TaggerInfo>,
    pub corpus: CorpusSummary,
    pub includes_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    /// Numeric version of the head migration (the `version` column from
    /// sqlx's `_sqlx_migrations` table). Compatibility comparison is
    /// numeric, not lexicographic — string compare breaks on `11` vs `9`.
    pub head_version: i64,
    /// Human-readable name of the head migration, formatted
    /// `<version>_<description>` for display only. Not used for
    /// compatibility decisions.
    pub head_name: String,
    pub all_migrations: Vec<String>,
    pub audit_entries: Vec<ManifestAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestAuditEntry {
    pub migration: String,
    #[serde(with = "time::serde::rfc3339")]
    pub ran_at: OffsetDateTime,
    pub rows_touched: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderInfo {
    pub model_id: String,
    pub dimensions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggerInfo {
    pub model_id: String,
    pub version: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusSummary {
    pub thoughts_live: i64,
    pub thoughts_retracted: i64,
    pub embeddings_count: i64,
    pub thought_links_live: i64,
    pub scopes_count: i64,
    /// ANN projection coverage posture ("present" or the documented
    /// absent marker). `default` keeps pre-field archives deserializable.
    #[serde(default)]
    pub ann_projection_posture: String,
}

/// Comparison outcome between the source's schema head version and the
/// target's schema head version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaCompat {
    /// Heads match — safe to restore.
    Match,
    /// Target is behind source. Operator must `sqlx migrate run` first.
    TargetBehind { source: i64, target: i64 },
    /// Target is ahead of source. Operator must restore on a matching
    /// version, or pass --skip-version-check.
    TargetAhead { source: i64, target: i64 },
    /// Target has no migrations applied (fresh DB or missing
    /// `_sqlx_migrations` table). Operator should migrate first.
    TargetEmpty { source: i64 },
}

/// Compare schema-head versions numerically. Sqlx's `version` column is
/// `i64`, so the comparison is just integer ordering — no string sorting
/// pitfalls (e.g. `"11" < "9"` lexicographically).
pub fn compare_schema_heads(source: i64, target: Option<i64>) -> SchemaCompat {
    match target {
        None => SchemaCompat::TargetEmpty { source },
        Some(t) if t == source => SchemaCompat::Match,
        Some(t) if t < source => SchemaCompat::TargetBehind { source, target: t },
        Some(t) => SchemaCompat::TargetAhead { source, target: t },
    }
}

/// Redact the password component of a postgres URL for storage in the
/// manifest. Best-effort — anything we don't recognize, we leave alone.
fn redact_db_url(url: &str) -> String {
    // Format: postgres://user:password@host:port/dbname?...
    // We just zap whatever sits between the first ':' after `://` and
    // the `@`.
    if let Some(scheme_end) = url.find("://") {
        let after = &url[scheme_end + 3..];
        if let Some(at_pos) = after.find('@') {
            let creds = &after[..at_pos];
            if let Some(colon) = creds.find(':') {
                let user = &creds[..colon];
                let host_and_rest = &after[at_pos..];
                return format!(
                    "{}://{}:<redacted>{}",
                    &url[..scheme_end],
                    user,
                    host_and_rest
                );
            }
        }
    }
    url.to_string()
}

/// `kengram backup` entry point.
pub async fn run_backup(
    config: Config,
    to: Option<PathBuf>,
    skip_embeddings: bool,
) -> anyhow::Result<()> {
    ensure_tool_on_path("pg_dump").await?;
    ensure_tool_on_path("tar").await?;

    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .connect(&config.database.url)
        .await
        .with_context(|| format!("connecting to {}", config.database.url))?;

    let manifest = build_manifest(&pool, &config, !skip_embeddings).await?;
    let security_packet = if corpus_hygiene_security::is_installed(&pool).await? {
        Some(corpus_hygiene_security::build_verified_packet(&pool).await?)
    } else {
        None
    };

    // Default output path: ./kengram-backup-<RFC3339-compact>.tar.gz in CWD.
    let output_path = match to {
        Some(p) => p,
        None => {
            let stamp: String = manifest
                .created_at
                .format(&Rfc3339)
                .unwrap_or_else(|_| "unknown".into())
                // Filesystem-friendly: colons in RFC3339 are illegal on
                // some FSes; drop them and the millisecond fraction.
                .chars()
                .map(|c| if c == ':' || c == '.' { '-' } else { c })
                .collect();
            PathBuf::from(format!("./kengram-backup-{stamp}.tar.gz"))
        }
    };

    // Use a unique tempdir under the system temp root. Manual cleanup at
    // function end; no Drop guard because we don't want to bring in
    // tempfile just for this. Crash-leak is recoverable (just a stale
    // tempdir).
    let tmp = std::env::temp_dir().join(format!("kengram-backup-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp).with_context(|| format!("creating tempdir {}", tmp.display()))?;

    let result = (async {
        // 1. pg_dump.
        let dump_path = tmp.join("kengram.dump");
        let mut cmd = Command::new("pg_dump");
        cmd.arg("--format=custom")
            .arg("--no-owner")
            .arg("--no-acl")
            .arg("--file")
            .arg(&dump_path);
        if skip_embeddings {
            // Drop the rows, keep the table. HNSW index survives over an
            // empty table; embed-backfill repopulates.
            cmd.arg("--exclude-table-data=embeddings");
        }
        cmd.arg(&config.database.url);
        run_subprocess(cmd, "pg_dump").await?;

        // 2. Manifest JSON.
        let manifest_path = tmp.join("manifest.json");
        let manifest_json =
            serde_json::to_string_pretty(&manifest).context("serializing backup manifest")?;
        std::fs::write(&manifest_path, manifest_json)
            .with_context(|| format!("writing {}", manifest_path.display()))?;

        // 3. Migration-0030's cluster-global security boundary.  Backups
        // refuse before this point if roles, membership options, owners,
        // triggers, principal map, or direct-write denials drifted.
        let mut checksums = vec![
            format!("{}  kengram.dump", sha256_file(&dump_path)?),
            format!("{}  manifest.json", sha256_file(&manifest_path)?),
        ];
        if let Some(packet) = &security_packet {
            let security_path = tmp.join("corpus_hygiene_security.json");
            let reconcile_path = tmp.join("reconcile-corpus-hygiene-security.sql");
            std::fs::write(&security_path, &packet.manifest_json)?;
            std::fs::write(&reconcile_path, &packet.reconcile_sql)?;
            checksums.push(format!(
                "{}  corpus_hygiene_security.json",
                packet.manifest_sha256
            ));
            checksums.push(format!(
                "{}  reconcile-corpus-hygiene-security.sql",
                packet.reconcile_sha256
            ));
        }
        std::fs::write(tmp.join("checksums.sha256"), checksums.join("\n") + "\n")?;

        // 4. tar -czf.
        let mut tar = Command::new("tar");
        tar.arg("-czf")
            .arg(&output_path)
            .arg("-C")
            .arg(&tmp)
            .arg(".");
        run_subprocess(tar, "tar").await?;

        Ok::<_, anyhow::Error>(())
    })
    .await;

    // Best-effort cleanup; don't shadow the original error.
    let _ = std::fs::remove_dir_all(&tmp);

    result?;

    // The archive cannot contain a digest of itself; write the whole-archive
    // digest as an adjacent, deterministic sidecar.
    let archive_sha = sha256_file(&output_path)?;
    let archive_sha_path = PathBuf::from(format!("{}.sha256", output_path.display()));
    std::fs::write(
        &archive_sha_path,
        format!(
            "{}  {}\n",
            archive_sha,
            output_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ),
    )?;

    // 4. Print summary.
    let bytes = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or_default();
    println!(
        "kengram backup → {} ({})",
        output_path.display(),
        humanize_bytes(bytes as i64)
    );
    println!("  schema:    {}", manifest.schema.head_name);
    println!(
        "  thoughts:  {} live, {} retracted",
        manifest.corpus.thoughts_live, manifest.corpus.thoughts_retracted
    );
    println!(
        "  embeddings: {}{}",
        manifest.corpus.embeddings_count,
        if manifest.includes_embeddings {
            ""
        } else {
            " (definition only — table data excluded)"
        }
    );
    println!("  links:     {} live", manifest.corpus.thought_links_live);
    println!("  scopes:    {}", manifest.corpus.scopes_count);
    println!(
        "  embedder:  {} ({}d)",
        manifest.embedder.model_id, manifest.embedder.dimensions
    );
    if let Some(t) = &manifest.tagger {
        println!("  tagger:    {} v{}", t.model_id, t.version);
    }
    if security_packet.is_some() {
        println!("  security:  exact corpus-hygiene role/owner/ACL manifest included");
        println!("  archive:   sha256 {}", archive_sha);
    }
    Ok(())
}

/// `kengram restore` entry point.
pub async fn run_restore(
    config: Config,
    from: PathBuf,
    force: bool,
    skip_version_check: bool,
) -> anyhow::Result<()> {
    ensure_tool_on_path("pg_restore").await?;
    ensure_tool_on_path("tar").await?;

    if !from.exists() {
        bail!("backup archive not found: {}", from.display());
    }

    // 1. Extract tarball.
    let tmp = std::env::temp_dir().join(format!("kengram-restore-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp).with_context(|| format!("creating tempdir {}", tmp.display()))?;

    let result = (async {
        let mut tar = Command::new("tar");
        tar.arg("-xzf").arg(&from).arg("-C").arg(&tmp);
        run_subprocess(tar, "tar").await?;

        // 2. Read manifest.
        let manifest_path = tmp.join("manifest.json");
        if !manifest_path.exists() {
            bail!(
                "archive missing manifest.json — not a valid kengram backup: {}",
                from.display()
            );
        }
        let dump_path = tmp.join("kengram.dump");
        if !dump_path.exists() {
            bail!(
                "archive missing kengram.dump — not a valid kengram backup: {}",
                from.display()
            );
        }
        let manifest_bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest: BackupManifest = serde_json::from_slice(&manifest_bytes)
            .context("parsing manifest.json — corrupt or wrong format")?;

        let checksums_path = tmp.join("checksums.sha256");
        if checksums_path.exists() {
            verify_component_checksums(&tmp)?;
        } else if manifest.schema.head_version >= 30 {
            bail!("migration-0030+ archive is missing checksums.sha256");
        }

        let security_path = tmp.join("corpus_hygiene_security.json");
        let reconcile_path = tmp.join("reconcile-corpus-hygiene-security.sql");
        let source_security = if manifest.schema.head_version >= 30 {
            if !security_path.exists() || !reconcile_path.exists() {
                bail!("migration-0030+ archive is missing corpus-hygiene security artifacts");
            }
            Some(
                serde_json::from_slice::<CorpusHygieneSecurityManifest>(&std::fs::read(
                    &security_path,
                )?)
                .context("parsing corpus_hygiene_security.json")?,
            )
        } else {
            None
        };

        if manifest.manifest_version > MANIFEST_VERSION {
            bail!(
                "manifest version {} is newer than this binary understands ({}); \
                 upgrade kengram on the target machine first",
                manifest.manifest_version,
                MANIFEST_VERSION
            );
        }

        // 3. Connect to target DB.
        let pool = PgPoolOptions::new()
            .max_connections(config.database.max_connections)
            .connect(&config.database.url)
            .await
            .with_context(|| format!("connecting to {}", config.database.url))?;

        // 4. --force guard — only required when target has data.
        let existing_thoughts = count_existing_thoughts(&pool).await?;
        if existing_thoughts > 0 && !force {
            println!("kengram restore — DRY RUN (target has existing data)");
            println!();
            print_manifest_summary(&manifest);
            println!();
            println!(
                "target database has {} existing thoughts. Re-run with --force \
                 to replace them.",
                existing_thoughts
            );
            std::process::exit(1);
        }

        // 5. Compatibility check.
        if !skip_version_check {
            let target_head = query_schema_head_version(&pool).await?;
            match compare_schema_heads(manifest.schema.head_version, target_head) {
                SchemaCompat::Match => {}
                SchemaCompat::TargetEmpty { .. } => {
                    bail!(
                        "target database has no schema applied. Run \
                         `sqlx migrate run` (or `kengram migrate`) on the target first."
                    );
                }
                SchemaCompat::TargetBehind { source, target } => {
                    bail!(
                        "target schema head (v{}) is behind source (v{} — {}). \
                         Run `sqlx migrate run` on the target first.",
                        target,
                        source,
                        manifest.schema.head_name,
                    );
                }
                SchemaCompat::TargetAhead { source, target } => {
                    bail!(
                        "target schema head (v{}) is ahead of source (v{} — {}). \
                         Either restore on a matching version, or pass \
                         --skip-version-check (advanced; may leave the \
                         database in an inconsistent state).",
                        target,
                        source,
                        manifest.schema.head_name,
                    );
                }
            }

            // Embedder + tagger mismatch are warnings, not refusals.
            if manifest.embedder.model_id != config.embedder.model_id {
                eprintln!(
                    "warning: backup embedder is `{}` ({}d); target is configured \
                     for `{}` ({}d). Mismatched embeddings will be restored; \
                     consider `kengram embed-backfill` afterward.",
                    manifest.embedder.model_id,
                    manifest.embedder.dimensions,
                    config.embedder.model_id,
                    config.embedder.dimensions,
                );
            }
            if let Some(source_tagger) = &manifest.tagger
                && !config.tagger.provider.is_empty()
                && (source_tagger.model_id != config.tagger.model_id
                    || source_tagger.version != config.tagger.model_version)
            {
                eprintln!(
                    "warning: backup tagger is `{}` v{}; target is configured \
                     for `{}` v{}. Tag provenance will reflect the source; \
                     `kengram tag --rerun` to refresh.",
                    source_tagger.model_id,
                    source_tagger.version,
                    config.tagger.model_id,
                    config.tagger.model_version,
                );
            }
        }

        // 6. Confirmation summary.
        println!("kengram restore — proceeding");
        println!();
        print_manifest_summary(&manifest);
        println!();

        // 7. pg_restore.
        let mut restore = Command::new("pg_restore");
        restore
            .arg("--clean")
            .arg("--if-exists")
            .arg("--no-owner")
            .arg("--no-acl")
            .arg("--dbname")
            .arg(&config.database.url)
            .arg(&dump_path);
        run_subprocess(restore, "pg_restore").await?;

        // 8. Reconcile cluster-global roles/owners/ACLs before any restored
        // writer can start, then read back the exact boundary.
        if let Some(source_security) = source_security {
            let reconcile_sql = std::fs::read_to_string(&reconcile_path)?;
            corpus_hygiene_security::apply_reconciliation(&pool, &reconcile_sql).await?;
            let target_packet = corpus_hygiene_security::build_verified_packet(&pool).await?;
            ensure_security_equivalent(&source_security, &target_packet)?;
        }

        // 9. Done.
        println!();
        println!(
            "restored from {} (backup created {}). Run `kengram stats` to verify.",
            manifest.source_database,
            manifest
                .created_at
                .format(&Rfc3339)
                .unwrap_or_else(|_| "<unknown>".into())
        );
        Ok::<_, anyhow::Error>(())
    })
    .await;

    let _ = std::fs::remove_dir_all(&tmp);
    result
}

fn sha256_file(path: &std::path::Path) -> anyhow::Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("opening {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn verify_component_checksums(root: &std::path::Path) -> anyhow::Result<()> {
    let path = root.join("checksums.sha256");
    if !path.exists() {
        bail!("backup archive missing checksums.sha256");
    }
    for (line_number, line) in std::fs::read_to_string(&path)?.lines().enumerate() {
        let (expected, relative) = line
            .split_once("  ")
            .with_context(|| format!("invalid checksums.sha256 line {}", line_number + 1))?;
        if relative.contains('/') || relative.contains("..") {
            bail!("unsafe checksum path: {relative}");
        }
        let actual = sha256_file(&root.join(relative))?;
        if actual != expected {
            bail!(
                "backup component checksum mismatch for {relative}: expected {expected}, got {actual}"
            );
        }
    }
    Ok(())
}

fn ensure_security_equivalent(
    source: &CorpusHygieneSecurityManifest,
    target: &SecurityPacket,
) -> anyhow::Result<()> {
    let restored = &target.manifest;
    // PostgreSQL 17 added the table MAINTAIN privilege to the owner's
    // implicit ACL (`m`).  Owners are compared independently and remain the
    // same canonical role, so cross-major restore compares every non-owner
    // grant exactly while allowing that platform-owned capability delta.
    let object_acls_match = if source.server_version == restored.server_version {
        source.objects_and_acls == restored.objects_and_acls
    } else {
        non_owner_acl_view(&source.objects_and_acls)
            == non_owner_acl_view(&restored.objects_and_acls)
    };
    if source.roles != restored.roles
        || source.memberships != restored.memberships
        || source.producer_principals_sha256 != restored.producer_principals_sha256
        || source.functions != restored.functions
        || source.triggers != restored.triggers
        || !object_acls_match
        || source.privilege_proof != restored.privilege_proof
    {
        bail!("restored corpus-hygiene security boundary differs from the signed source manifest");
    }
    Ok(())
}

fn non_owner_acl_view(value: &serde_json::Value) -> serde_json::Value {
    let mut normalized = value.clone();
    if let Some(objects) = normalized.as_array_mut() {
        for object in objects {
            let owner_prefix = object["owner"]
                .as_str()
                .map(|owner| format!("{owner}="))
                .unwrap_or_default();
            if let Some(acls) = object
                .get_mut("acl")
                .and_then(serde_json::Value::as_array_mut)
            {
                acls.retain(|acl| {
                    !acl.as_str()
                        .is_some_and(|entry| entry.starts_with(&owner_prefix))
                });
            }
        }
    }
    normalized
}

/// Build the manifest by querying the live DB and reading config.
async fn build_manifest(
    pool: &sqlx::PgPool,
    config: &Config,
    includes_embeddings: bool,
) -> anyhow::Result<BackupManifest> {
    let migrations = query_all_migrations(pool).await?;
    let (head_version, head_name) = migrations.last().cloned().ok_or_else(|| {
        anyhow!("target has no migrations applied; cannot back up an empty schema")
    })?;
    let all_migration_names: Vec<String> =
        migrations.iter().map(|(_, name)| name.clone()).collect();

    let audit_entries = kengram_storage::query_migration_audit(pool, None, 1000)
        .await
        .context("querying migration_audit")?
        .into_iter()
        .map(|r| ManifestAuditEntry {
            migration: r.migration,
            ran_at: r.ran_at,
            rows_touched: r.rows_touched,
            notes: r.notes,
        })
        .collect();

    let stats = kengram_storage::corpus_stats(pool, None)
        .await
        .context("querying corpus_stats")?;

    let embeddings_count: i64 = stats.embeddings.iter().map(|e| e.count).sum();

    let tagger = if config.tagger.provider.is_empty() {
        None
    } else {
        Some(TaggerInfo {
            model_id: config.tagger.model_id.clone(),
            version: config.tagger.model_version,
        })
    };

    Ok(BackupManifest {
        manifest_version: MANIFEST_VERSION,
        kengram_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: OffsetDateTime::now_utc(),
        source_database: redact_db_url(&config.database.url),
        schema: SchemaInfo {
            head_version,
            head_name,
            all_migrations: all_migration_names,
            audit_entries,
        },
        embedder: EmbedderInfo {
            model_id: config.embedder.model_id.clone(),
            dimensions: config.embedder.dimensions as u32,
        },
        tagger,
        corpus: CorpusSummary {
            thoughts_live: stats.thoughts.live,
            thoughts_retracted: stats.thoughts.retracted,
            embeddings_count,
            thought_links_live: stats.links.live,
            scopes_count: stats.scopes.len() as i64,
            ann_projection_posture: stats.ann_projection_posture.clone(),
        },
        includes_embeddings,
    })
}

/// Read all migration rows from `_sqlx_migrations`, sorted by version
/// ascending. Each tuple is `(version, "<version>_<description>")` — the
/// version for numeric compatibility comparison, the formatted name for
/// human display.
async fn query_all_migrations(pool: &sqlx::PgPool) -> anyhow::Result<Vec<(i64, String)>> {
    let rows = sqlx::query!(
        r#"
        SELECT version, description
        FROM _sqlx_migrations
        ORDER BY version ASC
        "#
    )
    .fetch_all(pool)
    .await
    .context("querying _sqlx_migrations")?;

    Ok(rows
        .into_iter()
        .map(|r| (r.version, format!("{}_{}", r.version, r.description)))
        .collect())
}

/// Schema head version as the last (highest-version) row of
/// `_sqlx_migrations`. Returns `None` when the table doesn't exist or is
/// empty (fresh database).
async fn query_schema_head_version(pool: &sqlx::PgPool) -> anyhow::Result<Option<i64>> {
    // Existence check first — restore target may be a fresh DB without the
    // migrations table.
    let exists = sqlx::query_scalar!(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = '_sqlx_migrations'
        ) AS "exists!"
        "#
    )
    .fetch_one(pool)
    .await
    .context("checking _sqlx_migrations existence")?;

    if !exists {
        return Ok(None);
    }

    let migrations = query_all_migrations(pool).await?;
    Ok(migrations.last().map(|(v, _)| *v))
}

/// COUNT of rows in `thoughts`. Returns 0 if the table doesn't exist yet
/// (fresh DB before migrations).
async fn count_existing_thoughts(pool: &sqlx::PgPool) -> anyhow::Result<i64> {
    let exists = sqlx::query_scalar!(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'thoughts'
        ) AS "exists!"
        "#
    )
    .fetch_one(pool)
    .await
    .context("checking thoughts table existence")?;

    if !exists {
        return Ok(0);
    }

    let count = sqlx::query_scalar!(r#"SELECT COUNT(*) AS "count!" FROM thoughts"#)
        .fetch_one(pool)
        .await
        .context("counting thoughts")?;
    Ok(count)
}

fn print_manifest_summary(m: &BackupManifest) {
    println!("  source:    {}", m.source_database);
    println!(
        "  created:   {}",
        m.created_at
            .format(&Rfc3339)
            .unwrap_or_else(|_| "<unknown>".into())
    );
    println!("  kengram:    v{}", m.kengram_version);
    println!("  schema:    {}", m.schema.head_name);
    println!(
        "  thoughts:  {} live, {} retracted",
        m.corpus.thoughts_live, m.corpus.thoughts_retracted
    );
    println!(
        "  embeddings: {}{}",
        m.corpus.embeddings_count,
        if m.includes_embeddings {
            ""
        } else {
            " (definition only)"
        }
    );
    println!("  links:     {} live", m.corpus.thought_links_live);
    println!("  scopes:    {}", m.corpus.scopes_count);
    println!(
        "  embedder:  {} ({}d)",
        m.embedder.model_id, m.embedder.dimensions
    );
    if let Some(t) = &m.tagger {
        println!("  tagger:    {} v{}", t.model_id, t.version);
    }
}

/// Spawn a subprocess; bail with stderr if it exits non-zero.
async fn run_subprocess(mut cmd: Command, name: &str) -> anyhow::Result<()> {
    let output = cmd
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("spawning {name}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{name} failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        );
    }
    Ok(())
}

/// Best-effort PATH check for an external tool. Surfaces a clear error
/// before the operator has to decode "No such file or directory" from
/// tokio's process spawn failure.
async fn ensure_tool_on_path(tool: &str) -> anyhow::Result<()> {
    let status = Command::new("which")
        .arg(tool)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => Err(anyhow!(
            "`{tool}` not found on PATH. Install Postgres client tools \
             (`brew install postgresql@16` on macOS; `apt install \
             postgresql-client-16` on Debian/Ubuntu)."
        )),
    }
}

fn humanize_bytes(n: i64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < units.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", n, units[i])
    } else {
        format!("{:.1} {}", v, units[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> BackupManifest {
        BackupManifest {
            manifest_version: 1,
            kengram_version: "0.1.0".into(),
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
            source_database: "postgres://kengram:<redacted>@localhost:5432/kengram".into(),
            schema: SchemaInfo {
                head_version: 11,
                head_name: "11_drop_tags_relations".into(),
                all_migrations: vec!["1_initial".into(), "11_drop_tags_relations".into()],
                audit_entries: vec![],
            },
            embedder: EmbedderInfo {
                model_id: "bge-m3:1024".into(),
                dimensions: 1024,
            },
            tagger: Some(TaggerInfo {
                model_id: "vllm/qwen3-coder:30b".into(),
                version: 7,
            }),
            corpus: CorpusSummary {
                thoughts_live: 42,
                thoughts_retracted: 3,
                embeddings_count: 42,
                thought_links_live: 15,
                scopes_count: 7,
            },
            includes_embeddings: true,
        }
    }

    #[test]
    fn manifest_json_round_trip() {
        let original = sample_manifest();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let parsed: BackupManifest = serde_json::from_str(&json).unwrap();
        // Compare via JSON re-serialization since BackupManifest doesn't
        // derive PartialEq (OffsetDateTime equality is tricky).
        let reserialized = serde_json::to_string_pretty(&parsed).unwrap();
        assert_eq!(json, reserialized);
    }

    #[test]
    fn manifest_without_tagger_omits_field() {
        let mut m = sample_manifest();
        m.tagger = None;
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            !json.contains("\"tagger\""),
            "tagger should be omitted: {json}"
        );
    }

    #[test]
    fn compare_schema_heads_match() {
        assert_eq!(compare_schema_heads(11, Some(11)), SchemaCompat::Match);
    }

    #[test]
    fn compare_schema_heads_target_behind() {
        assert!(matches!(
            compare_schema_heads(11, Some(9)),
            SchemaCompat::TargetBehind {
                source: 11,
                target: 9,
            },
        ));
    }

    #[test]
    fn compare_schema_heads_target_ahead() {
        assert!(matches!(
            compare_schema_heads(9, Some(11)),
            SchemaCompat::TargetAhead {
                source: 9,
                target: 11,
            },
        ));
    }

    #[test]
    fn compare_schema_heads_target_empty() {
        assert!(matches!(
            compare_schema_heads(11, None),
            SchemaCompat::TargetEmpty { source: 11 },
        ));
    }

    /// Regression: lex compare on raw version strings flips 9 vs 11. This
    /// test pins the numeric ordering invariant.
    #[test]
    fn compare_schema_heads_handles_two_digit_versions_correctly() {
        // If we did lex compare on the rendered strings, "9" > "11", so
        // target=9 vs source=11 would be (incorrectly) TargetAhead. Numeric
        // compare reports TargetBehind, which is correct.
        assert!(matches!(
            compare_schema_heads(11, Some(9)),
            SchemaCompat::TargetBehind { .. },
        ));
    }

    #[test]
    fn redact_db_url_replaces_password() {
        let url = "postgres://kengram:secret@host:5432/db";
        let redacted = redact_db_url(url);
        assert!(!redacted.contains("secret"), "password leaked: {redacted}");
        assert!(
            redacted.contains("kengram"),
            "user should survive: {redacted}"
        );
        assert!(
            redacted.contains("host:5432/db"),
            "host should survive: {redacted}"
        );
    }

    #[test]
    fn redact_db_url_passes_through_when_no_credentials() {
        let url = "postgres:///dbname";
        // Best-effort: just don't crash. The output may or may not be
        // modified.
        let _ = redact_db_url(url);
    }

    #[test]
    fn manifest_version_constant_is_one() {
        assert_eq!(MANIFEST_VERSION, 1);
    }
}

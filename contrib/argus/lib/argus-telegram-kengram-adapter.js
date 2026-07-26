"use strict";
/*
 * argus-telegram-kengram-adapter (productionized module)
 *
 * Source of truth for semantics: Trinity's prototype
 *   trinity/artifacts/telegram-kengram-ingest-20260606/argus-telegram-kengram-adapter.js
 *
 * This file is a MECHANICAL productionization of that prototype:
 *   - restructured into a require-safe module (no CLI / DB / FS side effects at import)
 *   - the spike-DB DEFAULT_DB hardcode is REMOVED; the API now requires an explicit
 *     dbUrl and FAILS CLOSED if absent (never defaults to a local production DB)
 *   - secret hygiene: only a sanitized error_class reaches logs/DLQ; raw err / raw
 *     envelope live only in 0600 files; raw err.message never reaches log/DLQ/DB-error
 *
 * ZERO semantic change to the write logic vs the prototype:
 *   - rendered thought content (buildThoughtContent)
 *   - scope mapping agents/{author}
 *   - the argus_source_events (namespace, source_ref) idempotency gate
 *   - duplicate_skip / conflict DLQ (metadata-only) / pending_embeddings / 3-table txn
 *
 * Compact (payload:null) events are kept OUT of this write path. The NATS->kEngram
 * CONSUMER is responsible for DLQ-deferring compact events; the adapter never sees
 * them. validate() rejects a null/missing payload before any DB work as a backstop.
 */

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const EMBEDDER_MODEL_ID = process.env.KENGRAM_EMBEDDER_MODEL_ID || "bge-m3:1024";
const TAGGER_MODEL_ID = process.env.KENGRAM_TAGGER_MODEL_ID || "ollama/gemma3:12b";
const ADAPTER_VERSION = "trinity-telegram-kengram-adapter-v0.1";
const TELEGRAM_NAMESPACE = "conversations/telegram-yetiwerks";
const FUTURE_SKEW_MS = 5 * 60 * 1000;

function resolveNowMs(now) {
  if (now == null) return Date.now();
  if (typeof now === "function") return resolveNowMs(now());
  if (typeof now === "number" && Number.isFinite(now)) return now;
  if (now instanceof Date) {
    const ms = now.getTime();
    if (Number.isFinite(ms)) return ms;
  }
  if (typeof now === "string" && now.trim()) {
    const ms = Date.parse(now);
    if (Number.isFinite(ms)) return ms;
  }
  return Date.now();
}

function throwTimestampError(errorClass) {
  const e = new Error(errorClass);
  e.error_class = errorClass;
  throw e;
}

// ECMAScript TimeClip limit: abs(ms) > 8.64e15 is unrepresentable as Date.
const MAX_REPRESENTABLE_DATE_MS = 8.64e15;

function isoFromRepresentableMs(ms) {
  if (!Number.isFinite(ms) || Math.abs(ms) > MAX_REPRESENTABLE_DATE_MS) {
    throwTimestampError("invalid_source_created_at");
  }
  try {
    const iso = new Date(ms).toISOString();
    if (typeof iso !== "string" || !Number.isFinite(Date.parse(iso))) {
      throwTimestampError("invalid_source_created_at");
    }
    return iso;
  } catch (err) {
    if (err && err.error_class) throw err;
    throwTimestampError("invalid_source_created_at");
  }
}

function validateSourceCreatedAt(raw, now) {
  const nowMs = resolveNowMs(now);
  if (raw == null) throwTimestampError("missing_source_created_at");
  if (typeof raw === "string" && raw.trim() === "") throwTimestampError("missing_source_created_at");
  let ms;
  if (typeof raw === "number") {
    ms = raw;
  } else {
    const s = String(raw).trim();
    if (!s) throwTimestampError("missing_source_created_at");
    ms = Date.parse(s);
  }
  if (!Number.isFinite(ms)) throwTimestampError("invalid_source_created_at");
  const iso = isoFromRepresentableMs(ms);
  if (ms > nowMs + FUTURE_SKEW_MS) throwTimestampError("future_source_created_at");
  return iso;
}

// ---------------------------------------------------------------------------
// canonicalJson — byte-identical to the producer
// (argus-telegram-hive-batcher.js:1550-1572 / nats-shadow:421-436). The leaf
// rule maps undefined -> null so the hash is invariant the same way the
// producer computes payload_sha256.
// ---------------------------------------------------------------------------
function canonicalJson(value) {
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalJson).join(",") + "]";
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return (
      "{" +
      keys
        .map((k) => JSON.stringify(k) + ":" + canonicalJson(value[k]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(value === undefined ? null : value);
}

function sha256Hex(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

// payloadSha256 hashes the canonical PAYLOAD, matching the producer exactly.
function payloadSha256(payload) {
  return sha256Hex(canonicalJson(payload));
}

function sqlString(value) {
  return "'" + String(value).replace(/'/g, "''") + "'";
}

// ---------------------------------------------------------------------------
// Secret hygiene helpers
//   errorClass(err): a stable, sanitized error class token (no raw message,
//   no payload, no url/password). This is the ONLY error text allowed into
//   logs / DLQ rows / DB error fields.
// ---------------------------------------------------------------------------
function errorClass(err) {
  if (!err) return "unknown_error";
  // Prefer an explicit code tag if the thrower set one.
  if (err.error_class && typeof err.error_class === "string") return err.error_class;
  const msg = String((err && err.message) || err || "");
  // Map known validation failures to stable classes WITHOUT echoing values.
  if (/payload_hash mismatch/.test(msg)) return "payload_hash_conflict";
  if (/payload_sha256 mismatch/.test(msg)) return "payload_sha256_mismatch";
  if (/missing\/invalid namespace/.test(msg)) return "invalid_namespace";
  if (/missing\/invalid kind/.test(msg) || /unsupported kind/.test(msg)) return "invalid_kind";
  if (/author\/agent mismatch/.test(msg)) return "author_agent_mismatch";
  if (/missing\/invalid author/.test(msg)) return "invalid_author";
  if (/missing\/invalid agent/.test(msg)) return "invalid_agent";
  if (/missing\/invalid (summary|source_ref|topic_key)/.test(msg)) return "invalid_required_field";
  if (/unsupported source_kind/.test(msg)) return "unsupported_source_kind";
  if (/source_ref must be agent-scoped/.test(msg)) return "source_ref_scope";
  if (/author does not produce a safe scope/.test(msg)) return "invalid_scope_token";
  if (/missing\/invalid payload/.test(msg) || /compact/.test(msg)) return "compact_or_missing_payload";
  if (/dbUrl/.test(msg)) return "missing_db_url";
  return "adapter_error";
}

// requireDbUrl: FAIL CLOSED. No hardcoded default DB ever.
function requireDbUrl(dbUrl) {
  if (!dbUrl || typeof dbUrl !== "string" || !dbUrl.trim()) {
    const e = new Error("dbUrl is required (no default DB; adapter fails closed)");
    e.error_class = "missing_db_url";
    throw e;
  }
  return dbUrl;
}

function makePsql(dbUrl) {
  const db = requireDbUrl(dbUrl);
  return function psql(sql, tuplesOnly = true) {
    const args = [db, "-X", "-v", "ON_ERROR_STOP=1"];
    if (tuplesOnly) args.push("-t", "-A", "-F", "\t");
    args.push("-c", sql);
    return execFileSync("psql", args, {
      encoding: "utf8",
      env: Object.assign({}, process.env, {
        PATH: "/opt/homebrew/bin:" + (process.env.PATH || ""),
      }),
    }).trim();
  };
}

// ---------------------------------------------------------------------------
// validate / normalize — Trinity's validateRecord, verbatim semantics.
// The argus.ingest.v1 branch normalizes the envelope then recurses.
// payload:null / missing payload is rejected (compact kept out of write path).
// ---------------------------------------------------------------------------
function validate(record, options) {
  options = options || {};
  if (record.schema_version === "argus.ingest.v1") {
    if (record.source_kind !== "telegram") throw new Error("unsupported source_kind: " + record.source_kind);
    if (record.namespace !== TELEGRAM_NAMESPACE) {
      throw new Error("missing/invalid namespace: expected " + TELEGRAM_NAMESPACE);
    }
    if (record.kind !== "note") throw new Error("missing/invalid kind: expected note");
    if (!record.author || typeof record.author !== "string") throw new Error("missing/invalid author");
    if (!record.agent || typeof record.agent !== "string") throw new Error("missing/invalid agent");
    if (record.author !== record.agent) throw new Error("author/agent mismatch: " + record.author + " != " + record.agent);
    // Compact / payload-null backstop: the consumer DLQ-defers compact events,
    // but if one ever reaches the adapter we reject it BEFORE any DB work.
    if (record.payload === null || record.payload === undefined) {
      const e = new Error("missing/invalid payload (compact events are not adapter input)");
      e.error_class = "compact_or_missing_payload";
      throw e;
    }
    // FIX 1 (Smith HIGH): payload_sha256 integrity gate. Mirrors the producer
    // invariant (nats-shadow:532-548): payload_sha256 MUST be 64-lowercase-hex
    // AND MUST equal sha256(canonicalJson(record.payload)) recomputed with THIS
    // module's own canonicalJson (byte-identical to the producer). This runs in
    // the validate/normalize path BEFORE the recursion and BEFORE any DB call
    // (readExisting/storeRecord), so a forged hash is rejected with zero psql
    // calls. Valid envelopes (hash matches) are unaffected.
    {
      const claimed = record.payload_sha256;
      if (typeof claimed !== "string" || !/^[0-9a-f]{64}$/.test(claimed)) {
        const e = new Error("payload_sha256 mismatch: missing or not 64-hex-lowercase");
        e.error_class = "payload_sha256_mismatch";
        throw e;
      }
      const recomputed = payloadSha256(record.payload);
      if (recomputed !== claimed) {
        const e = new Error("payload_sha256 mismatch: recomputed canonical payload hash does not match envelope.payload_sha256");
        e.error_class = "payload_sha256_mismatch";
        throw e;
      }
    }
    const sourceCreatedAt = validateSourceCreatedAt(record.created_at, options.now);
    const payload = record.payload || {};
    const normalized = {
      namespace: record.namespace,
      source_ref: record.source_ref,
      topic_key: record.topic_key,
      kind: record.kind,
      author: record.author,
      summary: payload.summary,
      key_facts: payload.key_facts || [],
      decisions: payload.decisions || [],
      intents: payload.intents || [],
      action_items: payload.action_items || [],
      open_questions: payload.open_questions || [],
      blockers: payload.blockers || [],
      artifacts: payload.artifacts || [],
      corrections: payload.corrections || [],
      topics: payload.topics || [],
      participants: payload.participants || [],
      is_noise: payload.is_noise === true,
      payload_sha256: record.payload_sha256,
      envelope: {
        schema_version: record.schema_version,
        event_id: record.event_id,
        dedupe_key: record.dedupe_key,
        subject: record.subject,
        source_kind: record.source_kind,
        producer: record.producer,
        direction: record.direction,
        chat_id: record.chat_id,
        message_id: record.message_id,
        session_id: record.session_id,
        batch_id: record.batch_id,
        created_at: sourceCreatedAt,
        published_at: record.published_at,
      },
      provenance: record.provenance || {},
    };
    return validate(normalized, options);
  }

  const required = ["namespace", "source_ref", "topic_key", "kind", "author", "summary"];
  for (const key of required) {
    if (!record[key] || typeof record[key] !== "string") throw new Error("missing/invalid " + key);
  }
  if (record.kind !== "note") throw new Error("unsupported kind: " + record.kind);
  if (record.is_noise === true) return { skip: true, reason: "noise" };
  const expectedPrefix = "telegram:" + record.author + ":";
  if (!record.source_ref.startsWith(expectedPrefix)) {
    throw new Error("source_ref must be agent-scoped and start with " + expectedPrefix);
  }
  return { skip: false, record };
}

// Alias retained for callers preferring normalize() naming.
function normalize(record, options) {
  return validate(record, options);
}

function stringList(record, key) {
  const value = record[key];
  if (!Array.isArray(value) || value.length === 0) return "";
  return "\\n" + title(key) + ":\\n" + value.map((item) => "- " + String(item)).join("\\n") + "\\n";
}

function title(key) {
  return key.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function buildThoughtContent(record) {
  const sections = [
    "Telegram distilled batch for " + record.author + ".",
    "Summary: " + record.summary,
    stringList(record, "key_facts"),
    stringList(record, "decisions"),
    stringList(record, "intents"),
    stringList(record, "action_items"),
    stringList(record, "open_questions"),
    stringList(record, "blockers"),
    stringList(record, "artifacts"),
    stringList(record, "corrections"),
    stringList(record, "topics"),
    stringList(record, "participants"),
    "Source: " + record.source_ref,
    "Topic: " + record.topic_key,
  ];
  return sections.filter(Boolean).join("\\n").trim();
}

function scopeFor(record) {
  const safe = record.author.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  if (!safe) throw new Error("author does not produce a safe scope token");
  return "agents/" + safe;
}

// DLQ + raw-error files are 0600. Only metadata-only rows go to the DLQ jsonl;
// raw envelope / raw error text go to a separate 0600 sidecar.
function writeDlq(dlqPath, row) {
  fs.mkdirSync(path.dirname(dlqPath), { recursive: true });
  fs.appendFileSync(dlqPath, JSON.stringify(row) + "\\n", { mode: 0o600 });
}

function writeRawSidecar(dlqPath, rawObj) {
  // raw envelope + raw error live ONLY here, 0600, alongside the DLQ.
  const rawPath = dlqPath + ".raw.jsonl";
  fs.mkdirSync(path.dirname(rawPath), { recursive: true });
  fs.appendFileSync(rawPath, JSON.stringify(rawObj) + "\\n", { mode: 0o600 });
}

function readExisting(psql, namespace, sourceRef) {
  const sql = [
    "SELECT payload_hash, status, COALESCE(thought_id::text, '')",
    "FROM argus_source_events",
    "WHERE namespace = " + sqlString(namespace) + " AND source_ref = " + sqlString(sourceRef),
    "LIMIT 1",
  ].join(" ");
  const out = psql(sql);
  if (!out) return null;
  const [hash, status, thoughtId] = out.split("\t");
  return { hash, status, thoughtId: thoughtId || null };
}

function markConflict(psql, record, payloadHash, priorHash, dlqPath) {
  // FIX 2 (Smith MEDIUM): the argus_source_events.error column gets a
  // closed-vocabulary class token ONLY — no prose, no namespace/source_ref,
  // no hashes. The identifiers + prior/incoming hashes live exclusively in the
  // conflict metadata jsonb (below) and the metadata-only DLQ row, matching the
  // "error_class only to DB-error" contract.
  const error = "payload_hash_conflict";
  const conflictMeta = {
    adapter_version: ADAPTER_VERSION,
    conflict_at: new Date().toISOString(),
    incoming_payload_sha256: payloadHash,
    prior_payload_sha256: priorHash,
  };
  const sql = [
    "UPDATE argus_source_events",
    "SET status = 'conflict', last_seen_at = NOW(), error = " + sqlString(error) + ", metadata = metadata || " + sqlString(JSON.stringify({ conflict: conflictMeta })) + "::jsonb",
    "WHERE namespace = " + sqlString(record.namespace) + " AND source_ref = " + sqlString(record.source_ref),
  ].join(" ");
  psql(sql, false);
  // DLQ row is metadata-only: identifiers + hashes, never payload content.
  writeDlq(dlqPath, {
    ts: new Date().toISOString(),
    reason: "payload_hash_conflict",
    namespace: record.namespace,
    source_ref: record.source_ref,
    prior_payload_sha256: priorHash,
    incoming_payload_sha256: payloadHash,
  });
}

function storeRecord(psql, record, payloadHash) {
  const content = buildThoughtContent(record);
  const scope = scopeFor(record);
  const createdAt =
    record.envelope && record.envelope.created_at
      ? validateSourceCreatedAt(record.envelope.created_at)
      : (() => {
          throwTimestampError("missing_source_created_at");
        })();
  const metadata = {
    adapter_version: ADAPTER_VERSION,
    namespace: record.namespace,
    source_ref: record.source_ref,
    topic_key: record.topic_key,
    kind: record.kind,
    author: record.author,
    payload_sha256: payloadHash,
    structured_payload: {
      summary: record.summary,
      key_facts: record.key_facts || [],
      decisions: record.decisions || [],
      intents: record.intents || [],
      action_items: record.action_items || [],
      open_questions: record.open_questions || [],
      blockers: record.blockers || [],
      artifacts: record.artifacts || [],
      corrections: record.corrections || [],
      topics: record.topics || [],
      participants: record.participants || [],
    },
    envelope: record.envelope || {},
    provenance: record.provenance || {},
  };

  const sql =
"\nBEGIN;\n" +
"INSERT INTO argus_source_events (namespace, source_ref, payload_hash, status, metadata)\n" +
"VALUES (" + sqlString(record.namespace) + ", " + sqlString(record.source_ref) + ", " + sqlString(payloadHash) + ", 'pending', " + sqlString(JSON.stringify(metadata)) + "::jsonb);\n" +
"\n" +
"WITH upserted AS (\n" +
"  INSERT INTO thoughts (scope, content, source, metadata, content_fingerprint, created_at)\n" +
"  VALUES (" + sqlString(scope) + ", " + sqlString(content) + ", 'telegram-batcher', " + sqlString(JSON.stringify(metadata)) + "::jsonb, digest(" + sqlString(content) + ", 'sha256'), " + sqlString(createdAt) + "::timestamptz)\n" +
"  ON CONFLICT (content_fingerprint) DO UPDATE SET metadata = thoughts.metadata\n" +
"  RETURNING id\n" +
"),\n" +
"queued AS (\n" +
"  INSERT INTO pending_embeddings (target_kind, target_id, model_id)\n" +
"  SELECT 'thought', id, " + sqlString(EMBEDDER_MODEL_ID) + " FROM upserted\n" +
"  ON CONFLICT (target_kind, target_id, model_id) DO NOTHING\n" +
"  RETURNING 1\n" +
"),\n" +
"queued_tags AS (\n" +
"  INSERT INTO pending_tags (thought_id, tagger_model_id)\n" +
"  SELECT u.id, " + sqlString(TAGGER_MODEL_ID) + " FROM upserted u\n" +
"  WHERE NOT EXISTS (\n" +
"    SELECT 1 FROM thoughts t\n" +
"    WHERE t.id = u.id\n" +
"      AND (t.tags_extractor_model IS NOT NULL OR COALESCE(t.tags, '{}'::jsonb) <> '{}'::jsonb)\n" +
"  )\n" +
"  ON CONFLICT (thought_id) DO NOTHING\n" +
"  RETURNING 1\n" +
"),\n" +
"queue_counts AS (\n" +
"  SELECT (SELECT count(*) FROM queued) AS embedding_rows, (SELECT count(*) FROM queued_tags) AS tag_rows\n" +
")\n" +
"UPDATE argus_source_events\n" +
"SET thought_id = (SELECT id FROM upserted), status = 'stored', last_seen_at = NOW()\n" +
"FROM queue_counts\n" +
"WHERE namespace = " + sqlString(record.namespace) + " AND source_ref = " + sqlString(record.source_ref) + ";\n" +
"COMMIT;\n" +
"SELECT thought_id::text FROM argus_source_events WHERE namespace = " + sqlString(record.namespace) + " AND source_ref = " + sqlString(record.source_ref) + ";\n";
  const out = psql(sql);
  return out.split("\n").filter(Boolean).pop();
}

// processRecord: explicit options. dbUrl is REQUIRED (fail closed). dlqPath
// optional; defaults are NOT a DB, just a local DLQ file path the caller may
// override. No hidden DB default anywhere.
function processRecord(record, options) {
  options = options || {};
  const psql = options.psql || makePsql(options.dbUrl); // makePsql enforces dbUrl
  const dlqPath = options.dlqPath || null;

  const validation = validate(record, options);
  record = validation.record || record;
  if (validation.skip) return { action: "skipped", reason: validation.reason, source_ref: record.source_ref };

  const payloadHash = record.payload_sha256 || sha256Hex(canonicalJson(record));
  const existing = readExisting(psql, record.namespace, record.source_ref);
  if (existing) {
    if (existing.hash === payloadHash) {
      psql("UPDATE argus_source_events SET last_seen_at = NOW() WHERE namespace = " + sqlString(record.namespace) + " AND source_ref = " + sqlString(record.source_ref), false);
      return { action: "duplicate_skip", source_ref: record.source_ref, thought_id: existing.thoughtId };
    }
    if (!dlqPath) {
      const e = new Error("dlqPath required to record a conflict");
      e.error_class = "missing_dlq_path";
      throw e;
    }
    markConflict(psql, record, payloadHash, existing.hash, dlqPath);
    return { action: "conflict_dlq", source_ref: record.source_ref };
  }

  const thoughtId = storeRecord(psql, record, payloadHash);
  return { action: "stored", source_ref: record.source_ref, thought_id: thoughtId, payload_sha256: payloadHash };
}

module.exports = {
  ADAPTER_VERSION,
  EMBEDDER_MODEL_ID,
  TAGGER_MODEL_ID,
  TELEGRAM_NAMESPACE,
  canonicalJson,
  sha256Hex,
  payloadSha256,
  sqlString,
  errorClass,
  requireDbUrl,
  makePsql,
  validateSourceCreatedAt,
  validate,
  normalize,
  buildThoughtContent,
  scopeFor,
  stringList,
  title,
  writeDlq,
  writeRawSidecar,
  readExisting,
  markConflict,
  storeRecord,
  processRecord,
};

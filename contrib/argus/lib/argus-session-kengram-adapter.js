"use strict";
/*
 * argus-session-kengram-adapter
 *
 * kEngram write adapter for distilled agent session envelopes.
 * This module is require-safe and never opens a DB connection on import.
 * Callers must provide dbUrl or psql; missing DB config fails closed.
 */

const base = require("./argus-telegram-kengram-adapter");

const ADAPTER_VERSION = "argus-session-kengram-adapter-v0.1";
const SESSION_SCHEMA_VERSION = "argus.ingest.v1";
const SESSION_SOURCE_KIND = "session";
const EMBEDDER_MODEL_ID = base.EMBEDDER_MODEL_ID;
const TAGGER_MODEL_ID = base.TAGGER_MODEL_ID;

function expectedSubjectPrefix() {
  return process.env.SESSION_KENGRAM_EXPECTED_SUBJECT_PREFIX || "ingest.session";
}

function errorClass(err) {
  if (!err) return "unknown_error";
  if (typeof err.error_class === "string" && err.error_class) return err.error_class;
  const msg = String((err && err.message) || err || "");
  if (/payload_sha256/i.test(msg)) return "payload_sha256_mismatch";
  if (/missing\/invalid namespace/i.test(msg)) return "invalid_namespace";
  if (/missing\/invalid source_ref/i.test(msg)) return "invalid_source_ref";
  if (/missing\/invalid payload/i.test(msg)) return "invalid_payload";
  if (/missing\/invalid agent/i.test(msg)) return "invalid_agent";
  if (/missing\/invalid subject/i.test(msg)) return "invalid_subject";
  if (/unsupported source_kind/i.test(msg)) return "unsupported_source_kind";
  if (/dbUrl/i.test(msg)) return "missing_db_url";
  return "session_adapter_error";
}

function fail(message, errorClassValue) {
  const e = new Error(message);
  e.error_class = errorClassValue;
  throw e;
}

function validateRecord(record) {
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    fail("missing/invalid envelope", "invalid_envelope");
  }
  if (record.schema_version !== SESSION_SCHEMA_VERSION) {
    fail("bad schema_version", "bad_schema_version");
  }
  if (record.source_kind !== SESSION_SOURCE_KIND) {
    fail("unsupported source_kind", "unsupported_source_kind");
  }
  if (record.kind !== "note") fail("missing/invalid kind", "invalid_kind");

  const agent = String(record.agent || "").trim();
  if (!agent || !/^[A-Za-z0-9_-]+$/.test(agent)) {
    fail("missing/invalid agent", "invalid_agent");
  }
  if (record.author !== agent) fail("author/agent mismatch", "author_agent_mismatch");

  const namespace = String(record.namespace || "").trim();
  if (namespace !== `sessions/${agent}`) {
    fail("missing/invalid namespace", "invalid_namespace");
  }

  const sourceRef = String(record.source_ref || "").trim();
  const sourceRefRe = new RegExp(
    `^session:${agent}:[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:bytes:\\d+-\\d+(?::gen:[0-9a-f]{12})?$`
  );
  if (!sourceRefRe.test(sourceRef)) {
    fail("missing/invalid source_ref", "invalid_source_ref");
  }

  if (record.event_id !== sourceRef) fail("event_id mismatch", "event_id_mismatch");
  if (record.dedupe_key !== sourceRef) fail("dedupe_key mismatch", "dedupe_key_mismatch");
  if (record.batch_id !== sourceRef) fail("batch_id mismatch", "batch_id_mismatch");

  const subject = String(record.subject || "").trim();
  const expectedPrefix = `${expectedSubjectPrefix()}.${agent}.`;
  if (!subject.startsWith(expectedPrefix)) {
    fail("missing/invalid subject", "invalid_subject");
  }

  const payload = record.payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    fail("missing/invalid payload", "invalid_payload");
  }
  if (payload.is_noise === true) return { skip: true, reason: "noise" };

  const claimed = String(record.payload_sha256 || "");
  if (!/^[0-9a-f]{64}$/.test(claimed)) {
    fail("payload_sha256 mismatch", "payload_sha256_mismatch");
  }
  const actual = base.sha256Hex(base.canonicalJson(payload));
  if (actual !== claimed) {
    fail("payload_sha256 mismatch", "payload_sha256_mismatch");
  }

  const arrays = [
    "key_facts",
    "decisions",
    "intents",
    "action_items",
    "open_questions",
    "blockers",
    "artifacts",
    "corrections",
  ];
  const signalCount = arrays.reduce(
    (n, key) => n + (Array.isArray(payload[key]) ? payload[key].length : 0),
    0
  );
  if (signalCount === 0) return { skip: true, reason: "empty_signal" };

  return { skip: false, record };
}

function listSection(title, arr) {
  if (!Array.isArray(arr) || arr.length === 0) return "";
  return `\n${title}:\n${arr.map((item) => `- ${String(item)}`).join("\n")}\n`;
}

function buildThoughtContent(record) {
  // Embedded content = clean semantic text only. agent/source_ref/kind/namespace
  // live in metadata (see storeRecord) — never duplicate the envelope into
  // content or it poisons the embedding/FTS with the repeated preamble + slugs.
  // (Bob 2026-06-22 memory de-pollution.)
  const payload = record.payload || {};
  const parts = [
    payload.summary ? String(payload.summary).trim() : "",
    listSection("Key facts", payload.key_facts),
    listSection("Decisions", payload.decisions),
    listSection("Corrections", payload.corrections),
    listSection("Intents", payload.intents),
    listSection("Action items", payload.action_items),
    listSection("Open questions", payload.open_questions),
    listSection("Blockers", payload.blockers),
    listSection("Artifacts", payload.artifacts),
    listSection("Topics", payload.topics),
    listSection("Participants", payload.participants),
  ];
  return parts.filter(Boolean).join("\n").trim();
}

function storeRecord(psql, record, payloadHash) {
  const content = buildThoughtContent(record);
  const scope = record.namespace;
  const metadata = {
    adapter_version: ADAPTER_VERSION,
    namespace: record.namespace,
    source_ref: record.source_ref,
    kind: record.kind,
    author: record.author,
    agent: record.agent,
    source_kind: SESSION_SOURCE_KIND,
    subject: record.subject,
    payload_sha256: payloadHash,
    structured_payload: record.payload,
    envelope: {
      schema_version: record.schema_version,
      event_id: record.event_id,
      dedupe_key: record.dedupe_key,
      subject: record.subject,
      source_kind: record.source_kind,
      producer: record.producer,
      host: record.host,
      session_id: record.session_id,
      batch_id: record.batch_id,
      created_at: record.created_at,
      published_at: record.published_at,
    },
    provenance: record.provenance || {},
  };

  const sql =
    "\nBEGIN;\n" +
    "INSERT INTO argus_source_events (namespace, source_ref, payload_hash, status, metadata)\n" +
    "VALUES (" +
    base.sqlString(record.namespace) +
    ", " +
    base.sqlString(record.source_ref) +
    ", " +
    base.sqlString(payloadHash) +
    ", 'pending', " +
    base.sqlString(JSON.stringify(metadata)) +
    "::jsonb);\n" +
    "\nWITH upserted AS (\n" +
    "  INSERT INTO thoughts (scope, content, source, metadata, content_fingerprint)\n" +
    "  VALUES (" +
    base.sqlString(scope) +
    ", " +
    base.sqlString(content) +
    ", 'session-batcher', " +
    base.sqlString(JSON.stringify(metadata)) +
    "::jsonb, digest(" +
    base.sqlString(content) +
    ", 'sha256'))\n" +
    "  ON CONFLICT (content_fingerprint) DO UPDATE SET metadata = thoughts.metadata\n" +
    "  RETURNING id\n" +
    "), queued AS (\n" +
    "  INSERT INTO pending_embeddings (target_kind, target_id, model_id)\n" +
    "  SELECT 'thought', id, " +
    base.sqlString(EMBEDDER_MODEL_ID) +
    " FROM upserted\n" +
    "  ON CONFLICT (target_kind, target_id, model_id) DO NOTHING\n" +
    "  RETURNING 1\n" +
    "), queued_tags AS (\n" +
    "  INSERT INTO pending_tags (thought_id, tagger_model_id)\n" +
    "  SELECT u.id, " +
    base.sqlString(TAGGER_MODEL_ID) +
    " FROM upserted u\n" +
    "  WHERE NOT EXISTS (\n" +
    "    SELECT 1 FROM thoughts t\n" +
    "    WHERE t.id = u.id\n" +
    "      AND (t.tags_extractor_model IS NOT NULL OR COALESCE(t.tags, '{}'::jsonb) <> '{}'::jsonb)\n" +
    "  )\n" +
    "  ON CONFLICT (thought_id) DO NOTHING\n" +
    "  RETURNING 1\n" +
    "), queue_counts AS (\n" +
    "  SELECT (SELECT count(*) FROM queued) AS embedding_rows, (SELECT count(*) FROM queued_tags) AS tag_rows\n" +
    ")\n" +
    "UPDATE argus_source_events\n" +
    "SET thought_id = (SELECT id FROM upserted), status = 'stored', last_seen_at = NOW()\n" +
    "FROM queue_counts\n" +
    "WHERE namespace = " +
    base.sqlString(record.namespace) +
    " AND source_ref = " +
    base.sqlString(record.source_ref) +
    ";\n" +
    "COMMIT;\n" +
    "SELECT thought_id::text FROM argus_source_events WHERE namespace = " +
    base.sqlString(record.namespace) +
    " AND source_ref = " +
    base.sqlString(record.source_ref) +
    ";\n";
  const out = psql(sql);
  return out.split("\n").filter(Boolean).pop();
}

function processRecord(record, options) {
  options = options || {};
  const psql = options.psql || base.makePsql(options.dbUrl);
  const dlqPath = options.dlqPath || null;
  const validation = validateRecord(record);
  if (validation.skip) {
    return { action: "skipped", reason: validation.reason, source_ref: record.source_ref };
  }
  record = validation.record;
  const payloadHash = record.payload_sha256 || base.sha256Hex(base.canonicalJson(record.payload));
  const existing = base.readExisting(psql, record.namespace, record.source_ref);
  if (existing) {
    if (existing.hash === payloadHash) {
      psql(
        "UPDATE argus_source_events SET last_seen_at = NOW() WHERE namespace = " +
          base.sqlString(record.namespace) +
          " AND source_ref = " +
          base.sqlString(record.source_ref),
        false
      );
      return { action: "duplicate_skip", source_ref: record.source_ref, thought_id: existing.thoughtId };
    }
    if (!dlqPath) {
      const e = new Error("dlqPath required to record a conflict");
      e.error_class = "missing_dlq_path";
      throw e;
    }
    base.markConflict(psql, record, payloadHash, existing.hash, dlqPath);
    return { action: "conflict_dlq", source_ref: record.source_ref };
  }
  const thoughtId = storeRecord(psql, record, payloadHash);
  return { action: "stored", source_ref: record.source_ref, thought_id: thoughtId, payload_sha256: payloadHash };
}

module.exports = {
  ADAPTER_VERSION,
  SESSION_SCHEMA_VERSION,
  SESSION_SOURCE_KIND,
  EMBEDDER_MODEL_ID,
  TAGGER_MODEL_ID,
  expectedSubjectPrefix,
  errorClass,
  validateRecord,
  buildThoughtContent,
  storeRecord,
  processRecord,
};

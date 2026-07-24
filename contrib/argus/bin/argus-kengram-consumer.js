#!/usr/bin/env node
"use strict";
/*
 * argus-kengram-consumer
 *
 * Generalized one-shot NATS -> kEngram consumer. This is the replacement
 * contract for running telegram + session lanes from one yeti-side writer.
 * It intentionally does not run as a daemon loop; launchd StartInterval calls
 * --once just like the existing telegram consumer.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const NATS_LIB_DIR =
  process.env.NATS_LIB_DIR || path.join(__dirname, "nats-shadow/node_modules/nats");
const {
  connect,
  AckPolicy,
  DeliverPolicy,
  RetentionPolicy,
  StorageType,
  DiscardPolicy,
  nanos,
  StringCodec,
} = require(NATS_LIB_DIR);
const SC = StringCodec();

const telegramConsumer = require("./argus-telegram-kengram-consumer.js");
const sessionAdapter = require("../lib/argus-session-kengram-adapter.js");
const baseAdapter = require("../lib/argus-telegram-kengram-adapter.js");

let redactSecrets;
(function loadRedactor() {
  const candidates = [
    process.env.SUMMARIZER_MODULE,
    "/Users/yetibob/argus/bin/argus-telegram-hive-summarizer.js",
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      const mod = require(candidate);
      if (mod && typeof mod.redactSecrets === "function") {
        redactSecrets = mod.redactSecrets;
        return;
      }
    } catch (_) {
      /* try next */
    }
  }
  const secretPatterns = [
    /\b\d{6,12}:[A-Za-z0-9_-]{30,}\b/g,
    /\b(?:sbp|sb|sk|pk|rk|ghp|gho|ghs|ghr|glpat|xox[baprs]|AKIA|ASIA)[_-][A-Za-z0-9_-]{12,}\b/g,
    /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g,
    /\bsk-[A-Za-z0-9]{20,}\b/g,
    /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g,
    /\b(?:token|secret|password|passwd|api[_-]?key|apikey|access[_-]?key|private[_-]?key|service[_-]?key|bearer)\b\s*[:=]\s*["']?[A-Za-z0-9_\-./+]{12,}["']?/gi,
    /\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{24,}\b/g,
  ];
  redactSecrets = function redactSecretsFallback(value) {
    if (typeof value !== "string" || !value) return value;
    let out = value;
    for (const pattern of secretPatterns) out = out.replace(pattern, "[redacted secret]");
    return out;
  };
})();

const ONCE = process.argv.includes("--once") || process.env.KENGRAM_CONSUMER_ONCE === "1";
const HOME = process.env.HOME || require("os").homedir();

function loadEnvFile(file) {
  if (!file) return {};
  try {
    const out = {};
    for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
      if (!match) continue;
      out[match[1]] = match[2].replace(/^['"]|['"]$/g, "");
    }
    return out;
  } catch (err) {
    if (err.code === "ENOENT") return {};
    throw err;
  }
}

const NATS_AUTH_ENV = loadEnvFile(
  process.env.NATS_AUTH_ENV_FILE || path.join(HOME, "argus/state/nats-client/client.env")
);

function authUser(defaultUser = "yeti") {
  return process.env.KENGRAM_CONSUMER_NATS_USER || process.env.NATS_AUTH_USER || NATS_AUTH_ENV.NATS_AUTH_USER || defaultUser;
}

function authToken(user) {
  const key = `CRED_NATS_TOKEN_${String(user || "").toUpperCase()}`;
  return (
    process.env.KENGRAM_CONSUMER_NATS_TOKEN ||
    process.env.NATS_AUTH_TOKEN ||
    process.env[key] ||
    NATS_AUTH_ENV[key] ||
    NATS_AUTH_ENV.CRED_NATS_TOKEN
  );
}

const CONFIG = {
  natsUrl: process.env.KENGRAM_CONSUMER_NATS_URL || "127.0.0.1:4222",
  dbUrl: process.env.KENGRAM_CONSUMER_DB_URL || null,
  connectTimeoutMs: Number(process.env.KENGRAM_CONSUMER_CONNECT_TIMEOUT_MS || 8000),
  runTelegram: process.env.KENGRAM_GENERAL_RUN_TELEGRAM !== "0",
  runSession: process.env.KENGRAM_GENERAL_RUN_SESSION !== "0",
  runA2A: process.env.KENGRAM_GENERAL_RUN_A2A !== "0",
  dryRun: process.env.KENGRAM_GENERAL_DRY_RUN === "1",
  sessionStream: process.env.SESSION_KENGRAM_CONSUMER_STREAM || "ARGUS_SESSION_INGEST",
  sessionSubject: process.env.SESSION_KENGRAM_CONSUMER_FILTER || "ingest.session.>",
  sessionDurable:
    process.env.SESSION_KENGRAM_CONSUMER_DURABLE || "argus_session_kengram_writer",
  sessionBatch: Number(process.env.SESSION_KENGRAM_CONSUMER_BATCH || 100),
  sessionAckWaitMs: Number(process.env.SESSION_KENGRAM_CONSUMER_ACK_WAIT_MS || 120000),
  sessionMaxDeliver: Number(process.env.SESSION_KENGRAM_CONSUMER_MAX_DELIVER || 5),
  sessionMaxAckPending: Number(process.env.SESSION_KENGRAM_CONSUMER_MAX_ACK_PENDING || 128),
  sessionFetchExpiresMs: Number(process.env.SESSION_KENGRAM_CONSUMER_FETCH_EXPIRES_MS || 5000),
  sessionNakBackoffMs: Number(process.env.SESSION_KENGRAM_CONSUMER_NAK_BACKOFF_MS || 30000),
  a2aStream: process.env.A2A_KENGRAM_CONSUMER_STREAM || "AGENT_COMMS",
  a2aSubject: process.env.A2A_KENGRAM_CONSUMER_FILTER || "agent.>",
  a2aDurable: process.env.A2A_KENGRAM_CONSUMER_DURABLE || "kengram-ingest-a2a",
  a2aBatch: Number(process.env.A2A_KENGRAM_CONSUMER_BATCH || 100),
  a2aAckWaitMs: Number(process.env.A2A_KENGRAM_CONSUMER_ACK_WAIT_MS || 120000),
  a2aMaxDeliver: Number(process.env.A2A_KENGRAM_CONSUMER_MAX_DELIVER || 5),
  a2aMaxAckPending: Number(process.env.A2A_KENGRAM_CONSUMER_MAX_ACK_PENDING || 128),
  a2aFetchExpiresMs: Number(process.env.A2A_KENGRAM_CONSUMER_FETCH_EXPIRES_MS || 5000),
  a2aNakBackoffMs: Number(process.env.A2A_KENGRAM_CONSUMER_NAK_BACKOFF_MS || 30000),
  stateDir:
    process.env.KENGRAM_GENERAL_STATE_DIR ||
    path.join(HOME, "argus/state/kengram-consumer"),
};
CONFIG.natsUser = authUser();
CONFIG.natsToken = authToken(CONFIG.natsUser);

const SESSION_METRICS_FILE =
  process.env.SESSION_KENGRAM_CONSUMER_METRICS_FILE ||
  path.join(CONFIG.stateDir, "session-metrics.json");
const SESSION_DLQ_FILE =
  process.env.SESSION_KENGRAM_CONSUMER_DLQ_FILE ||
  path.join(CONFIG.stateDir, "session-dlq.jsonl");
const A2A_METRICS_FILE =
  process.env.A2A_KENGRAM_CONSUMER_METRICS_FILE ||
  path.join(CONFIG.stateDir, "a2a-metrics.json");
const A2A_DLQ_FILE =
  process.env.A2A_KENGRAM_CONSUMER_DLQ_FILE ||
  path.join(CONFIG.stateDir, "a2a-dlq.jsonl");

const A2A_NAMESPACE = "conversations/agents";
const A2A_SOURCE_PREFIX = "a2a:";

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
}

function chmodSafe(file, mode) {
  try {
    fs.chmodSync(file, mode);
  } catch (_) {
    /* best effort */
  }
}

function append0600(file, row) {
  ensureDir(path.dirname(file));
  const fd = fs.openSync(file, "a", 0o600);
  try {
    fs.writeSync(fd, JSON.stringify(row) + "\n");
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  chmodSafe(file, 0o600);
}

function sha256Hex(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

function classifyError(err) {
  const explicit = err && typeof err.error_class === "string" ? err.error_class : null;
  if (explicit) return { transient: err.transient === true, error_class: explicit };
  const msg = String((err && (err.stderr || err.message)) || err || "");
  if (/ECONNREFUSED|ETIMEDOUT|timeout|deadlock|too many clients|could not connect/i.test(msg)) {
    return { transient: true, error_class: "db_unavailable" };
  }
  return { transient: false, error_class: sessionAdapter.errorClass(err) };
}

function sessionStats() {
  return {
    fetched: 0,
    stored: 0,
    dup: 0,
    skipped: 0,
    conflict: 0,
    invalid: 0,
    dry_run_valid: 0,
    held: 0,
    acked: 0,
    termed: 0,
    naked: 0,
  };
}

function sessionConsumerConfig() {
  return {
    durable_name: CONFIG.sessionDurable,
    name: CONFIG.sessionDurable,
    filter_subject: CONFIG.sessionSubject,
    ack_policy: AckPolicy.Explicit,
    deliver_policy: DeliverPolicy.All,
    ack_wait: nanos(CONFIG.sessionAckWaitMs),
    max_deliver: CONFIG.sessionMaxDeliver,
    max_ack_pending: CONFIG.sessionMaxAckPending,
  };
}

async function ensureSessionStream(jsm) {
  try {
    await jsm.streams.info(CONFIG.sessionStream);
  } catch (_) {
    await jsm.streams.add({
      name: CONFIG.sessionStream,
      subjects: [CONFIG.sessionSubject],
      retention: RetentionPolicy.Limits,
      storage: StorageType.File,
      discard: DiscardPolicy.Old,
    });
  }
  try {
    await jsm.consumers.info(CONFIG.sessionStream, CONFIG.sessionDurable);
  } catch (_) {
    await jsm.consumers.add(CONFIG.sessionStream, sessionConsumerConfig());
  }
}

function validSubjectForEnvelope(envelope, subject) {
  const prefix = process.env.SESSION_KENGRAM_EXPECTED_SUBJECT_PREFIX || "ingest.session";
  return (
    envelope &&
    typeof envelope.subject === "string" &&
    envelope.subject === subject &&
    subject.startsWith(`${prefix}.${String(envelope.agent || "")}.`)
  );
}

async function processSessionMessage(msg, stats) {
  let envelope;
  try {
    envelope = JSON.parse(SC.decode(msg.data));
  } catch (_) {
    append0600(SESSION_DLQ_FILE, {
      ts: new Date().toISOString(),
      reason: "invalid",
      error_class: "json_parse_failed",
      subject: msg.subject,
      stream_seq: msg.seq,
    });
    stats.invalid++;
    stats.termed++;
    msg.term();
    return;
  }

  const sourceRef = typeof envelope.source_ref === "string" ? envelope.source_ref : undefined;
  const natsMsgId =
    sourceRef && typeof envelope.payload_sha256 === "string"
      ? `${sourceRef}:${envelope.payload_sha256}`
      : undefined;

  if (!validSubjectForEnvelope(envelope, msg.subject)) {
    append0600(SESSION_DLQ_FILE, {
      ts: new Date().toISOString(),
      reason: "invalid",
      error_class: "subject_mismatch",
      source_ref: sourceRef,
      nats_msg_id: natsMsgId,
      subject: msg.subject,
      stream_seq: msg.seq,
    });
    stats.invalid++;
    stats.termed++;
    msg.term();
    return;
  }

  if (CONFIG.dryRun) {
    try {
      sessionAdapter.validateRecord(envelope);
    } catch (err) {
      append0600(SESSION_DLQ_FILE, {
        ts: new Date().toISOString(),
        reason: "invalid",
        error_class: sessionAdapter.errorClass(err),
        source_ref: sourceRef,
        nats_msg_id: natsMsgId,
        subject: msg.subject,
        stream_seq: msg.seq,
      });
      stats.invalid++;
      stats.termed++;
      msg.term();
      return;
    }
    stats.dry_run_valid++;
    stats.acked++;
    msg.ack();
    return;
  }

  let result;
  try {
    result = sessionAdapter.processRecord(envelope, {
      dbUrl: CONFIG.dbUrl,
      dlqPath: path.join(CONFIG.stateDir, "session-adapter-dlq.jsonl"),
    });
  } catch (err) {
    const c = classifyError(err);
    append0600(SESSION_DLQ_FILE, {
      ts: new Date().toISOString(),
      reason: c.transient ? "held" : "unpublishable",
      error_class: c.error_class,
      source_ref: sourceRef,
      nats_msg_id: natsMsgId,
      subject: msg.subject,
      stream_seq: msg.seq,
    });
    if (c.transient) {
      stats.held++;
      stats.naked++;
      msg.nak(CONFIG.sessionNakBackoffMs);
    } else {
      stats.invalid++;
      stats.termed++;
      msg.term();
    }
    return;
  }

  if (!result || result.action === "stored") {
    stats.stored++;
    stats.acked++;
    msg.ack();
    return;
  }
  if (result.action === "duplicate_skip") {
    stats.dup++;
    stats.acked++;
    msg.ack();
    return;
  }
  if (result.action === "skipped") {
    stats.skipped++;
    stats.acked++;
    msg.ack();
    return;
  }
  if (result.action === "conflict_dlq") {
    stats.conflict++;
    stats.acked++;
    msg.ack();
    return;
  }

  append0600(SESSION_DLQ_FILE, {
    ts: new Date().toISOString(),
    reason: "held",
    error_class: "adapter_unknown_action",
    source_ref: sourceRef,
    nats_msg_id: natsMsgId,
    subject: msg.subject,
    stream_seq: msg.seq,
  });
  stats.held++;
  stats.naked++;
  msg.nak(CONFIG.sessionNakBackoffMs);
}

function writeSessionMetrics(stats, info, startedAt) {
  const metrics = {
    last_run_at: new Date().toISOString(),
    run_duration_ms: Date.now() - startedAt,
    ...stats,
    consumer_num_pending: info ? info.num_pending : null,
    consumer_num_ack_pending: info ? info.num_ack_pending : null,
    consumer_ack_floor: info && info.ack_floor ? info.ack_floor.stream_seq : null,
  };
  ensureDir(path.dirname(SESSION_METRICS_FILE));
  const tmp = `${SESSION_METRICS_FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(metrics, null, 2), { mode: 0o600 });
  fs.renameSync(tmp, SESSION_METRICS_FILE);
  chmodSafe(SESSION_METRICS_FILE, 0o600);
  return metrics;
}

async function runSessionOnce() {
  const startedAt = Date.now();
  const stats = sessionStats();
  ensureDir(CONFIG.stateDir);
  chmodSafe(CONFIG.stateDir, 0o700);
  const nc = await connect({
    servers: CONFIG.natsUrl,
    name: "argus-kengram-consumer-session",
    timeout: CONFIG.connectTimeoutMs,
    reconnect: false,
    maxReconnectAttempts: 0,
    ...(CONFIG.natsToken ? { user: CONFIG.natsUser, pass: CONFIG.natsToken } : {}),
  });
  let info = null;
  try {
    const jsm = await nc.jetstreamManager();
    await ensureSessionStream(jsm);
    const js = nc.jetstream();
    const consumer = await js.consumers.get(CONFIG.sessionStream, CONFIG.sessionDurable);
    const iter = await consumer.fetch({
      max_messages: CONFIG.sessionBatch,
      expires: CONFIG.sessionFetchExpiresMs,
    });
    for await (const msg of iter) {
      stats.fetched++;
      await processSessionMessage(msg, stats);
    }
    try {
      info = await jsm.consumers.info(CONFIG.sessionStream, CONFIG.sessionDurable);
    } catch (_) {
      info = null;
    }
  } finally {
    await nc.drain();
  }
  const metrics = writeSessionMetrics(stats, info, startedAt);
  console.log(
    `[argus-kengram-consumer] session once: fetched=${metrics.fetched} stored=${metrics.stored} ` +
      `dup=${metrics.dup} skipped=${metrics.skipped} conflict=${metrics.conflict} invalid=${metrics.invalid} ` +
      `dry_run_valid=${metrics.dry_run_valid} ` +
      `held=${metrics.held} acked=${metrics.acked} termed=${metrics.termed} naked=${metrics.naked} ` +
      `pending=${metrics.consumer_num_pending} ack_floor=${metrics.consumer_ack_floor}`
  );
  return stats;
}

function a2aStats() {
  return {
    fetched: 0,
    stored: 0,
    dup: 0,
    skipped: 0,
    skipped_system_events: 0,
    skipped_telegram_passthrough: 0,
    conflict: 0,
    invalid: 0,
    dry_run_valid: 0,
    held: 0,
    acked: 0,
    termed: 0,
    naked: 0,
  };
}

function failA2A(message, errorClassValue) {
  const e = new Error(message);
  e.error_class = errorClassValue;
  throw e;
}

function assertA2AAgentToken(value, label) {
  if (!value || typeof value !== "string" || !/^[a-z][a-z0-9_-]*$/.test(value)) {
    failA2A(`${label} must be a single lowercase agent token`, `invalid_${label}`);
  }
}

function stringifyHumanField(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch (_) {
    return String(value);
  }
}

function redactedHumanField(value) {
  return redactSecrets(stringifyHumanField(value)).trim();
}

function msgIDForA2AMessage(msg, envelope) {
  const headerID = msg.headers && msg.headers.get("Nats-Msg-Id");
  if (headerID) return headerID;
  if (envelope && typeof envelope.__a2a_msg_id === "string" && envelope.__a2a_msg_id) {
    return envelope.__a2a_msg_id;
  }
  return sha256Hex(baseAdapter.canonicalJson(envelope));
}

function normalizeA2AEnvelope(envelope, msgID, subject) {
  if (!envelope || typeof envelope !== "object" || Array.isArray(envelope)) {
    failA2A("missing/invalid a2a envelope", "invalid_envelope");
  }
  if (envelope.__no_ingest === true) {
    return { skip: true, reason: "__no_ingest" };
  }
  // System events (relay job results, watchdog/heartbeat probes, digests) carry
  // job_name and no from — pane deliveries, not a2a conversation records. Skip
  // capture, counted (skipped_system_events), per Knox contract 2026-07-20.
  // Strictly conjunctive: anything WITH a from still validates below.
  if (envelope.job_name !== undefined && !String(envelope.from || "").trim()) {
    return { skip: true, reason: "system_event" };
  }
  // Telegram passthrough (e.g. Bob replies relayed onto inbox subjects) carries
  // telegram identity (from_user_id/chat_id) and no a2a from — the telegram
  // consumer owns that conversation's ingestion (distilled), so capturing here
  // would dupe (raw vs distilled never dedups). Skip capture, counted
  // (skipped_telegram_passthrough), per Knox GO 2026-07-20.
  // Strictly conjunctive: telegram marker AND no from AND inbox subject.
  if (
    (envelope.from_user_id !== undefined || envelope.chat_id !== undefined) &&
    !String(envelope.from || "").trim() &&
    /^agent\.[a-z][a-z0-9_-]*\.inbox$/.test(String(subject || ""))
  ) {
    return { skip: true, reason: "telegram_passthrough" };
  }

  const from = String(envelope.from || "").trim();
  const to = String(envelope.to || envelope.__inbox || "").trim();
  assertA2AAgentToken(from, "from");
  assertA2AAgentToken(to, "to");
  if (envelope.__inbox !== undefined && envelope.__inbox !== to) {
    failA2A("__inbox/to mismatch", "inbox_to_mismatch");
  }
  const expectedSubject = `agent.${to}.inbox`;
  if (subject !== expectedSubject) {
    failA2A("subject/to mismatch", "subject_to_mismatch");
  }

  const type = redactedHumanField(envelope.type || "message");
  const priority = redactedHumanField(envelope.priority || "");
  const re = redactedHumanField(envelope.re || "");
  const text = redactedHumanField(envelope.text || envelope.summary || "");
  const sourceRef = `${A2A_SOURCE_PREFIX}${msgID}`;
  const payloadHash = baseAdapter.sha256Hex(
    baseAdapter.canonicalJson({
      subject,
      msg_id: msgID,
      envelope,
    })
  );

  return {
    skip: false,
    record: {
      namespace: A2A_NAMESPACE,
      source_ref: sourceRef,
      source_kind: "a2a",
      kind: "note",
      from,
      to,
      type,
      priority,
      re,
      text,
      subject,
      msg_id: msgID,
      payload_sha256: payloadHash,
      created_at: redactedHumanField(envelope.ts || ""),
    },
  };
}

function buildA2AThoughtContent(record) {
  // Embedded content = message substance only (subject + text). from/to/type/
  // priority/source_ref live in metadata (see storeA2ARecord) — never duplicate
  // the envelope into content or it poisons the embedding/FTS with repeated
  // "Agent-to-agent…/From:/To:" boilerplate + slugs. (Bob 2026-06-22.)
  const parts = [
    record.re ? String(record.re).trim() : "",
    record.text ? String(record.text).trim() : "",
  ];
  return parts.filter(Boolean).join("\n\n").trim();
}

function storeA2ARecord(psql, record, payloadHash) {
  const content = buildA2AThoughtContent(record);
  const scope = `agents/${record.from}`;
  const metadata = {
    adapter_version: "argus-a2a-kengram-adapter-v0.1",
    namespace: record.namespace,
    source_ref: record.source_ref,
    kind: record.kind,
    source_kind: record.source_kind,
    from: record.from,
    to: record.to,
    type: record.type,
    priority: record.priority || null,
    re: record.re || null,
    delivery: "nats",
    msg_id: record.msg_id,
    subject: record.subject,
    payload_sha256: payloadHash,
    envelope: {
      created_at: record.created_at || null,
    },
  };

  const sql =
    "\nBEGIN;\n" +
    "INSERT INTO argus_source_events (namespace, source_ref, payload_hash, status, metadata)\n" +
    "VALUES (" +
    baseAdapter.sqlString(record.namespace) +
    ", " +
    baseAdapter.sqlString(record.source_ref) +
    ", " +
    baseAdapter.sqlString(payloadHash) +
    ", 'pending', " +
    baseAdapter.sqlString(JSON.stringify(metadata)) +
    "::jsonb);\n" +
    "\nWITH upserted AS (\n" +
    "  INSERT INTO thoughts (scope, content, source, metadata, content_fingerprint)\n" +
    "  VALUES (" +
    baseAdapter.sqlString(scope) +
    ", " +
    baseAdapter.sqlString(content) +
    ", 'agent-comms', " +
    baseAdapter.sqlString(JSON.stringify(metadata)) +
    "::jsonb, digest(" +
    baseAdapter.sqlString(content) +
    ", 'sha256'))\n" +
    "  ON CONFLICT (content_fingerprint) DO UPDATE SET metadata = thoughts.metadata\n" +
    "  RETURNING id\n" +
    "), queued AS (\n" +
    "  INSERT INTO pending_embeddings (target_kind, target_id, model_id)\n" +
    "  SELECT 'thought', id, " +
    baseAdapter.sqlString(baseAdapter.EMBEDDER_MODEL_ID) +
    " FROM upserted\n" +
    "  ON CONFLICT (target_kind, target_id, model_id) DO NOTHING\n" +
    "  RETURNING 1\n" +
    "), queued_tags AS (\n" +
    "  INSERT INTO pending_tags (thought_id, tagger_model_id)\n" +
    "  SELECT u.id, " +
    baseAdapter.sqlString(baseAdapter.TAGGER_MODEL_ID) +
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
    baseAdapter.sqlString(record.namespace) +
    " AND source_ref = " +
    baseAdapter.sqlString(record.source_ref) +
    ";\n" +
    "COMMIT;\n" +
    "SELECT thought_id::text FROM argus_source_events WHERE namespace = " +
    baseAdapter.sqlString(record.namespace) +
    " AND source_ref = " +
    baseAdapter.sqlString(record.source_ref) +
    ";\n";
  const out = psql(sql);
  return out.split("\n").filter(Boolean).pop();
}

function processA2ARecord(record, options) {
  options = options || {};
  const psql = options.psql || baseAdapter.makePsql(options.dbUrl);
  const dlqPath = options.dlqPath || null;
  const existing = baseAdapter.readExisting(psql, record.namespace, record.source_ref);
  if (existing) {
    if (existing.hash === record.payload_sha256) {
      psql(
        "UPDATE argus_source_events SET last_seen_at = NOW() WHERE namespace = " +
          baseAdapter.sqlString(record.namespace) +
          " AND source_ref = " +
          baseAdapter.sqlString(record.source_ref),
        false
      );
      return { action: "duplicate_skip", source_ref: record.source_ref, thought_id: existing.thoughtId };
    }
    if (!dlqPath) {
      const e = new Error("dlqPath required to record a conflict");
      e.error_class = "missing_dlq_path";
      throw e;
    }
    baseAdapter.markConflict(psql, record, record.payload_sha256, existing.hash, dlqPath);
    return { action: "conflict_dlq", source_ref: record.source_ref };
  }
  const thoughtId = storeA2ARecord(psql, record, record.payload_sha256);
  return {
    action: "stored",
    source_ref: record.source_ref,
    thought_id: thoughtId,
    payload_sha256: record.payload_sha256,
  };
}

function a2aConsumerConfig() {
  return {
    durable_name: CONFIG.a2aDurable,
    name: CONFIG.a2aDurable,
    filter_subject: CONFIG.a2aSubject,
    ack_policy: AckPolicy.Explicit,
    deliver_policy: DeliverPolicy.StartTime,
    opt_start_time: new Date().toISOString(),
    ack_wait: nanos(CONFIG.a2aAckWaitMs),
    max_deliver: CONFIG.a2aMaxDeliver,
    max_ack_pending: CONFIG.a2aMaxAckPending,
  };
}

async function ensureA2AStream(jsm) {
  try {
    await jsm.streams.info(CONFIG.a2aStream);
  } catch (_) {
    await jsm.streams.add({
      name: CONFIG.a2aStream,
      subjects: [CONFIG.a2aSubject],
      retention: RetentionPolicy.Limits,
      storage: StorageType.File,
      discard: DiscardPolicy.Old,
      max_age: 90 * 24 * 60 * 60 * 1000000000,
      duplicate_window: 2 * 60 * 1000000000,
    });
  }
  try {
    await jsm.consumers.info(CONFIG.a2aStream, CONFIG.a2aDurable);
  } catch (_) {
    await jsm.consumers.add(CONFIG.a2aStream, a2aConsumerConfig());
  }
}

async function processA2AMessage(msg, stats) {
  let envelope;
  try {
    envelope = JSON.parse(SC.decode(msg.data));
  } catch (_) {
    append0600(A2A_DLQ_FILE, {
      ts: new Date().toISOString(),
      reason: "invalid",
      error_class: "json_parse_failed",
      subject: msg.subject,
      stream_seq: msg.seq,
    });
    stats.invalid++;
    stats.termed++;
    msg.term();
    return;
  }

  let normalized;
  let sourceRef = null;
  let msgID = null;
  try {
    msgID = msgIDForA2AMessage(msg, envelope);
    normalized = normalizeA2AEnvelope(envelope, msgID, msg.subject);
    if (normalized.skip) {
      stats.skipped++;
      if (normalized.reason === "system_event") stats.skipped_system_events++;
      if (normalized.reason === "telegram_passthrough") stats.skipped_telegram_passthrough++;
      stats.acked++;
      msg.ack();
      return;
    }
    sourceRef = normalized.record.source_ref;
  } catch (err) {
    append0600(A2A_DLQ_FILE, {
      ts: new Date().toISOString(),
      reason: "invalid",
      error_class: classifyError(err).error_class,
      source_ref: sourceRef,
      nats_msg_id: msgID,
      subject: msg.subject,
      stream_seq: msg.seq,
    });
    stats.invalid++;
    stats.termed++;
    msg.term();
    return;
  }

  if (CONFIG.dryRun) {
    stats.dry_run_valid++;
    stats.acked++;
    msg.ack();
    return;
  }

  let result;
  try {
    result = processA2ARecord(normalized.record, {
      dbUrl: CONFIG.dbUrl,
      dlqPath: path.join(CONFIG.stateDir, "a2a-adapter-dlq.jsonl"),
    });
  } catch (err) {
    const c = classifyError(err);
    append0600(A2A_DLQ_FILE, {
      ts: new Date().toISOString(),
      reason: c.transient ? "held" : "unpublishable",
      error_class: c.error_class,
      source_ref: sourceRef,
      nats_msg_id: msgID,
      subject: msg.subject,
      stream_seq: msg.seq,
    });
    if (c.transient) {
      stats.held++;
      stats.naked++;
      msg.nak(CONFIG.a2aNakBackoffMs);
    } else {
      stats.invalid++;
      stats.termed++;
      msg.term();
    }
    return;
  }

  if (!result || result.action === "stored") {
    stats.stored++;
    stats.acked++;
    msg.ack();
    return;
  }
  if (result.action === "duplicate_skip") {
    stats.dup++;
    stats.acked++;
    msg.ack();
    return;
  }
  if (result.action === "skipped") {
    stats.skipped++;
    stats.acked++;
    msg.ack();
    return;
  }
  if (result.action === "conflict_dlq") {
    stats.conflict++;
    stats.acked++;
    msg.ack();
    return;
  }

  append0600(A2A_DLQ_FILE, {
    ts: new Date().toISOString(),
    reason: "held",
    error_class: "adapter_unknown_action",
    source_ref: sourceRef,
    nats_msg_id: msgID,
    subject: msg.subject,
    stream_seq: msg.seq,
  });
  stats.held++;
  stats.naked++;
  msg.nak(CONFIG.a2aNakBackoffMs);
}

function writeA2AMetrics(stats, info, startedAt) {
  const metrics = {
    last_run_at: new Date().toISOString(),
    run_duration_ms: Date.now() - startedAt,
    ...stats,
    consumer_num_pending: info ? info.num_pending : null,
    consumer_num_ack_pending: info ? info.num_ack_pending : null,
    consumer_ack_floor: info && info.ack_floor ? info.ack_floor.stream_seq : null,
  };
  ensureDir(path.dirname(A2A_METRICS_FILE));
  const tmp = `${A2A_METRICS_FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(metrics, null, 2), { mode: 0o600 });
  fs.renameSync(tmp, A2A_METRICS_FILE);
  chmodSafe(A2A_METRICS_FILE, 0o600);
  return metrics;
}

async function runA2AOnce() {
  const startedAt = Date.now();
  const stats = a2aStats();
  ensureDir(CONFIG.stateDir);
  chmodSafe(CONFIG.stateDir, 0o700);
  const nc = await connect({
    servers: CONFIG.natsUrl,
    name: "argus-kengram-consumer-a2a",
    timeout: CONFIG.connectTimeoutMs,
    reconnect: false,
    maxReconnectAttempts: 0,
    ...(CONFIG.natsToken ? { user: CONFIG.natsUser, pass: CONFIG.natsToken } : {}),
  });
  let info = null;
  try {
    const jsm = await nc.jetstreamManager();
    await ensureA2AStream(jsm);
    const js = nc.jetstream();
    const consumer = await js.consumers.get(CONFIG.a2aStream, CONFIG.a2aDurable);
    const iter = await consumer.fetch({
      max_messages: CONFIG.a2aBatch,
      expires: CONFIG.a2aFetchExpiresMs,
    });
    for await (const msg of iter) {
      stats.fetched++;
      await processA2AMessage(msg, stats);
    }
    try {
      info = await jsm.consumers.info(CONFIG.a2aStream, CONFIG.a2aDurable);
    } catch (_) {
      info = null;
    }
  } finally {
    await nc.drain();
  }
  const metrics = writeA2AMetrics(stats, info, startedAt);
  console.log(
    `[argus-kengram-consumer] a2a once: fetched=${metrics.fetched} stored=${metrics.stored} ` +
      `dup=${metrics.dup} skipped=${metrics.skipped} skipped_system_events=${metrics.skipped_system_events} skipped_telegram_passthrough=${metrics.skipped_telegram_passthrough} conflict=${metrics.conflict} invalid=${metrics.invalid} ` +
      `dry_run_valid=${metrics.dry_run_valid} held=${metrics.held} acked=${metrics.acked} ` +
      `termed=${metrics.termed} naked=${metrics.naked} pending=${metrics.consumer_num_pending} ` +
      `ack_floor=${metrics.consumer_ack_floor}`
  );
  return stats;
}

async function run() {
  if (!CONFIG.dryRun && !CONFIG.dbUrl) {
    const e = new Error("KENGRAM_CONSUMER_DB_URL is required");
    e.error_class = "missing_db_url";
    throw e;
  }
  const result = {};
  if (CONFIG.runTelegram) {
    result.telegram = await telegramConsumer.run();
  }
  if (CONFIG.runSession) {
    result.session = await runSessionOnce();
  }
  if (CONFIG.runA2A) {
    result.a2a = await runA2AOnce();
  }
  return result;
}

module.exports = {
  CONFIG,
  SESSION_METRICS_FILE,
  SESSION_DLQ_FILE,
  A2A_METRICS_FILE,
  A2A_DLQ_FILE,
  A2A_NAMESPACE,
  sha256Hex,
  validSubjectForEnvelope,
  ensureSessionStream,
  processSessionMessage,
  runSessionOnce,
  normalizeA2AEnvelope,
  buildA2AThoughtContent,
  processA2ARecord,
  ensureA2AStream,
  processA2AMessage,
  runA2AOnce,
  run,
  baseAdapter,
  sessionAdapter,
};

if (require.main === module) {
  if (!ONCE) {
    console.error("[argus-kengram-consumer] pass --once; launchd owns cadence");
    process.exit(2);
  }
  run()
    .then(() => process.exit(0))
    .catch((err) => {
      const c = classifyError(err);
      console.error(`[argus-kengram-consumer] FATAL error_class=${c.error_class}`);
      process.exit(1);
    });
}

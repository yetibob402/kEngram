#!/usr/bin/env node
"use strict";
/*
 * argus-session-nats-producer
 *
 * Reads Claude session JSONL deltas, distills durable signal with the existing
 * telegram summarizer model/prompt, and publishes session envelopes to NATS.
 * It never imports a DB client, never opens kEngram, and never writes
 * Hive. launchd should run it with --once.
 */

const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");

const NATS_LIB_DIR =
  process.env.NATS_LIB_DIR || path.join(__dirname, "nats-shadow/node_modules/nats");

// Production-only NATS + distiller loads are deferred until the direct-execution
// / run path so pure helpers can be required from contrib/argus/test without
// sibling nats-shadow or argus-session-distiller.js.
let _nats = null;
let _summarizer = null;

function loadNats() {
  if (!_nats) {
    _nats = require(NATS_LIB_DIR);
  }
  return _nats;
}

function loadSummarizer() {
  if (!_summarizer) {
    _summarizer = require("./argus-session-distiller.js");
  }
  return _summarizer;
}

const SESSION_SCHEMA_VERSION = "argus.ingest.v1";
const SESSION_SOURCE_KIND = "session";
const FUTURE_SKEW_MS = 5 * 60 * 1000;
const TIMESTAMP_ERROR_CLASSES = new Set([
  "missing_source_created_at",
  "invalid_source_created_at",
  "future_source_created_at",
]);
const SECRET_PATTERNS = [
  { name: "openai_key", re: /\bsk-[A-Za-z0-9_-]{20,}\b/g },
  { name: "anthropic_key", re: /\bsk-ant-[A-Za-z0-9_-]{20,}\b/g },
  { name: "github_token", re: /\bgh[pousr]_[A-Za-z0-9_]{20,}\b/g },
  { name: "telegram_bot_token", re: /\b\d{6,}:[A-Za-z0-9_-]{20,}\b/g },
  { name: "jwt_like", re: /\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b/g },
  { name: "hive_key", re: /\bhive_sk[a-z0-9_]{10,}\b/gi },
  { name: "supabase_project_token", re: /\bsbp_[A-Za-z0-9]{20,}\b/g },
  { name: "postgres_url", re: /\bpostgres(?:ql)?:\/\/[^:\s/@]+:[^@\s]+@[^\s)>\]"']+/gi },
  { name: "password_assignment", re: /\b(password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*['"]?[^'"\s]{8,}/gi },
];

const HOME = process.env.HOME || os.homedir();
const ONCE = process.argv.includes("--once") || process.env.SESSION_NATS_PRODUCER_ONCE === "1";
const DRY_RUN = process.argv.includes("--dry-run") || process.env.SESSION_NATS_PRODUCER_DRY_RUN === "1";
const BACKFILL = process.argv.includes("--backfill") || process.env.SESSION_NATS_PRODUCER_BACKFILL === "1";

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
  return process.env.SESSION_NATS_PRODUCER_NATS_USER || process.env.NATS_AUTH_USER || NATS_AUTH_ENV.NATS_AUTH_USER || defaultUser;
}

function authToken(user) {
  const key = `CRED_NATS_TOKEN_${String(user || "").toUpperCase()}`;
  return (
    process.env.SESSION_NATS_PRODUCER_NATS_TOKEN ||
    process.env.NATS_AUTH_TOKEN ||
    process.env[key] ||
    NATS_AUTH_ENV[key] ||
    NATS_AUTH_ENV.CRED_NATS_TOKEN
  );
}

const CONFIG = {
  natsUrl: process.env.SESSION_NATS_PRODUCER_NATS_URL || "127.0.0.1:4222",
  stream: process.env.SESSION_NATS_PRODUCER_STREAM || "ARGUS_SESSION_INGEST",
  subjectPrefix: process.env.SESSION_NATS_PRODUCER_SUBJECT_PREFIX || "ingest.session",
  profileDir:
    process.env.SESSION_NATS_PRODUCER_PROFILE_DIR ||
    path.join(HOME, "argus/config/agents"),
  stateFile:
    process.env.SESSION_NATS_PRODUCER_STATE_FILE ||
    path.join(HOME, "argus/state/session-kengram-producer/offsets.json"),
  metricsFile:
    process.env.SESSION_NATS_PRODUCER_METRICS_FILE ||
    path.join(HOME, "argus/state/session-kengram-producer/metrics.json"),
  dlqFile:
    process.env.SESSION_NATS_PRODUCER_DLQ_FILE ||
    path.join(HOME, "argus/state/session-kengram-producer/dlq.jsonl"),
  maxLinesPerBatch: Number(process.env.SESSION_NATS_PRODUCER_MAX_LINES || 40),
  maxBytesPerBatch: Number(process.env.SESSION_NATS_PRODUCER_MAX_BYTES || 180000),
  minUserLen: Number(process.env.SESSION_NATS_PRODUCER_MIN_USER_LEN || 40),
  minAssistantLen: Number(process.env.SESSION_NATS_PRODUCER_MIN_ASSISTANT_LEN || 120),
  connectTimeoutMs: Number(process.env.SESSION_NATS_PRODUCER_CONNECT_TIMEOUT_MS || 8000),
  host: process.env.SESSION_NATS_PRODUCER_HOST || os.hostname().replace(/[^A-Za-z0-9_.-]+/g, "-"),
  producer: "argus-session-nats-producer-v0.1",
};
CONFIG.natsUser = authUser();
CONFIG.natsToken = authToken(CONFIG.natsUser);

const PERMANENT_REJECT_ERROR_CLASSES = new Set([
  "secret_scan_failed",
  "invalid_envelope",
  "invalid_agent",
  "invalid_source_identity",
  "bad_schema_version",
  "bad_source_kind",
  "invalid_namespace",
  "invalid_source_ref",
  "invalid_subject",
  "invalid_payload",
  "payload_sha256_mismatch",
  "identity_mismatch",
  "missing_source_created_at",
  "invalid_source_created_at",
  "future_source_created_at",
]);

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
  throw Object.assign(new Error(errorClass), { error_class: errorClass });
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

function parseUsableSourceInstant(raw, nowMs) {
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
  return { ms, iso };
}

function sourceCreatedAtFromRecords(records, now) {
  const nowMs = resolveNowMs(now);
  if (!Array.isArray(records) || records.length === 0) {
    throwTimestampError("missing_source_created_at");
  }
  let maxMs = -Infinity;
  let maxIso = null;
  for (const record of records) {
    const { ms, iso } = parseUsableSourceInstant(record && record.capture_ts, nowMs);
    if (ms > maxMs) {
      maxMs = ms;
      maxIso = iso;
    }
  }
  return maxIso;
}

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

function loadJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (err) {
    if (err.code === "ENOENT") return fallback;
    throw err;
  }
}

function writeJsonAtomic(file, value, mode = 0o600) {
  ensureDir(path.dirname(file));
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value, null, 2) + "\n", { mode });
  fs.renameSync(tmp, file);
  chmodSafe(file, mode);
}

function append0600(file, row) {
  ensureDir(path.dirname(file));
  fs.appendFileSync(file, JSON.stringify(row) + "\n", { mode: 0o600 });
  chmodSafe(file, 0o600);
}

function canonicalJson(value) {
  if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
  if (value && typeof value === "object") {
    return (
      "{" +
      Object.keys(value)
        .sort()
        .map((key) => JSON.stringify(key) + ":" + canonicalJson(value[key]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(value === undefined ? null : value);
}

function sha256Hex(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

function secretHitPaths(value, prefix = "$", hits = []) {
  if (typeof value === "string") {
    for (const { name, re } of SECRET_PATTERNS) {
      re.lastIndex = 0;
      if (re.test(value)) hits.push(`${prefix}:${name}`);
    }
    return hits;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => secretHitPaths(item, `${prefix}[${index}]`, hits));
    return hits;
  }
  if (value && typeof value === "object") {
    for (const key of Object.keys(value)) secretHitPaths(value[key], `${prefix}.${key}`, hits);
  }
  return hits;
}

function assertNoSecrets(value) {
  const hits = secretHitPaths(value);
  if (hits.length) {
    const e = new Error(`secret-shaped string in session envelope at ${hits.slice(0, 8).join(",")}`);
    e.error_class = "secret_scan_failed";
    throw e;
  }
}

function loadProfiles() {
  if (!fs.existsSync(CONFIG.profileDir)) return [];
  return fs
    .readdirSync(CONFIG.profileDir)
    .filter((file) => file.endsWith(".json"))
    .sort()
    .map((file) => {
      const profilePath = path.join(CONFIG.profileDir, file);
      const profile = JSON.parse(fs.readFileSync(profilePath, "utf8"));
      return { ...profile, profilePath };
    })
    .filter((profile) => profile.runtime === "claude")
    .filter((profile) => profile.name && profile.session_path && profile.namespace);
}

function stateKey(profile, file) {
  return `${profile.name}/${path.basename(file)}`;
}

function listSessionFiles(profile) {
  if (!fs.existsSync(profile.session_path)) return [];
  return fs
    .readdirSync(profile.session_path)
    .filter((file) => file.endsWith(".jsonl"))
    .map((file) => path.join(profile.session_path, file))
    .sort();
}

function extractText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((block) => block && block.type === "text")
      .map((block) => block.text || "")
      .join("\n");
  }
  return "";
}

function parseSessionLine(line, profile, sessionFile, startByte, endByte) {
  let obj;
  try {
    obj = JSON.parse(line);
  } catch (_) {
    return null;
  }
  const type = obj.type;
  if (type !== "user" && type !== "assistant") return null;
  const msg = obj.message;
  if (!msg || typeof msg !== "object") return null;
  const text = loadSummarizer().redactSecrets(extractText(msg.content).trim());
  if (!text) return null;
  if (text.startsWith("<local-command") || text.startsWith("<command-name")) return null;
  if (type === "user" && text.length < CONFIG.minUserLen) return null;
  if (type === "assistant" && text.length < CONFIG.minAssistantLen) return null;
  return {
    agent: profile.name,
    direction: type === "user" ? "inbound" : "outbound",
    text,
    capture_ts: obj.timestamp || obj.created_at || "",
    dedupe_key: `session-line:${profile.name}:${path.basename(sessionFile)}:${startByte}-${endByte}`,
    session_file: sessionFile,
    byte_start: startByte,
    byte_end: endByte,
  };
}

function readNewLines(file, fromOffset, maxBytes) {
  const stat = fs.statSync(file);
  if (stat.size <= fromOffset) return { lines: [], nextOffset: fromOffset, stat };
  const readLen = Math.min(stat.size - fromOffset, maxBytes);
  const fd = fs.openSync(file, "r");
  const buf = Buffer.alloc(readLen);
  try {
    fs.readSync(fd, buf, 0, readLen, fromOffset);
  } finally {
    fs.closeSync(fd);
  }
  const lines = [];
  let lineStart = 0;
  let nextOffset = fromOffset;
  for (let idx = 0; idx < buf.length; idx++) {
    if (buf[idx] !== 0x0a) continue;
    const startByte = fromOffset + lineStart;
    const endByte = fromOffset + idx + 1;
    lines.push({
      text: buf.toString("utf8", lineStart, idx),
      startByte,
      endByte,
    });
    lineStart = idx + 1;
    nextOffset = endByte;
    if (lines.length >= CONFIG.maxLinesPerBatch) break;
  }
  return { lines, nextOffset, stat };
}

function topicKey(summary) {
  const topics = Array.isArray(summary.topics) ? summary.topics : [];
  if (topics.length) {
    return String(topics[0])
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 80);
  }
  return "session-distill";
}

function generationTag() {
  return sha256Hex(JSON.stringify({
    producer: CONFIG.producer,
    summarizer_model: process.env.SUMMARIZER_MODEL || "default",
    minUserLen: CONFIG.minUserLen,
    minAssistantLen: CONFIG.minAssistantLen,
  })).slice(0, 12);
}

function sessionSourceRef(profile, sessionFile, startByte, endByte, generation) {
  const agent = profile.name;
  const sessionId = path.basename(sessionFile, ".jsonl").replace(/[^A-Za-z0-9_.-]+/g, "-");
  const gen = generation || generationTag();
  return `session:${agent}:${CONFIG.host}:${sessionId}:bytes:${startByte}-${endByte}:gen:${gen}`;
}

function buildEnvelope(profile, sessionFile, startByte, endByte, records, summary, now) {
  const agent = profile.name;
  const sessionId = path.basename(sessionFile, ".jsonl").replace(/[^A-Za-z0-9_.-]+/g, "-");
  const payload = {
    summary: summary.summary || "",
    key_facts: summary.key_facts || [],
    decisions: summary.decisions || [],
    intents: summary.intents || [],
    action_items: summary.action_items || [],
    open_questions: summary.open_questions || [],
    blockers: summary.blockers || [],
    artifacts: summary.artifacts || [],
    corrections: summary.corrections || [],
    topics: summary.topics || [],
    participants: summary.participants || [],
    is_noise: summary.is_noise === true,
    record_count: records.length,
    source_line_refs: records.map((record) => record.dedupe_key),
  };
  const payloadHash = sha256Hex(canonicalJson(payload));
  const generation = generationTag();
  const sourceRef = sessionSourceRef(profile, sessionFile, startByte, endByte, generation);
  const subject = `${CONFIG.subjectPrefix}.${agent}.${CONFIG.host}`;
  const nowMs = resolveNowMs(now);
  const createdAt = sourceCreatedAtFromRecords(records, nowMs);
  const publishedAt = new Date(nowMs).toISOString();
  return {
    schema_version: SESSION_SCHEMA_VERSION,
    event_id: sourceRef,
    dedupe_key: sourceRef,
    batch_id: sourceRef,
    subject,
    namespace: `sessions/${agent}`,
    kind: "note",
    source_kind: SESSION_SOURCE_KIND,
    producer: CONFIG.producer,
    author: agent,
    agent,
    host: CONFIG.host,
    session_id: sessionId,
    source_ref: sourceRef,
    topic_key: topicKey(summary),
    payload,
    payload_sha256: payloadHash,
    created_at: createdAt,
    published_at: publishedAt,
    provenance: {
      raw_path: sessionFile,
      byte_start: startByte,
      byte_end: endByte,
      line_count: records.length,
      profile: profile.profilePath,
      generation,
    },
  };
}

function validateEnvelope(envelope) {
  if (!envelope || typeof envelope !== "object" || Array.isArray(envelope)) {
    throw Object.assign(new Error("bad envelope"), { error_class: "invalid_envelope" });
  }
  const agent = String(envelope.agent || "");
  const host = String(envelope.host || "");
  const sessionId = String(envelope.session_id || "");
  if (!/^[A-Za-z0-9_-]+$/.test(agent)) {
    throw Object.assign(new Error("bad agent"), { error_class: "invalid_agent" });
  }
  if (!/^[A-Za-z0-9_.-]+$/.test(host) || !/^[A-Za-z0-9_.-]+$/.test(sessionId)) {
    throw Object.assign(new Error("bad host/session"), { error_class: "invalid_source_identity" });
  }
  if (envelope.schema_version !== SESSION_SCHEMA_VERSION) {
    throw Object.assign(new Error("bad schema"), { error_class: "bad_schema_version" });
  }
  if (envelope.source_kind !== SESSION_SOURCE_KIND) {
    throw Object.assign(new Error("bad source_kind"), { error_class: "bad_source_kind" });
  }
  if (envelope.namespace !== `sessions/${agent}` || envelope.author !== agent) {
    throw Object.assign(new Error("bad namespace/author"), { error_class: "invalid_namespace" });
  }
  const re = new RegExp(
    `^session:${agent}:${host}:${sessionId}:bytes:\\d+-\\d+(?::gen:[0-9a-f]{12})?$`
  );
  if (!re.test(String(envelope.source_ref || ""))) {
    throw Object.assign(new Error("bad source_ref"), { error_class: "invalid_source_ref" });
  }
  if (
    envelope.event_id !== envelope.source_ref ||
    envelope.dedupe_key !== envelope.source_ref ||
    envelope.batch_id !== envelope.source_ref
  ) {
    throw Object.assign(new Error("identity mismatch"), { error_class: "identity_mismatch" });
  }
  if (envelope.subject !== `${CONFIG.subjectPrefix}.${agent}.${host}`) {
    throw Object.assign(new Error("bad subject"), { error_class: "invalid_subject" });
  }
  const payload = envelope.payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw Object.assign(new Error("bad payload"), { error_class: "invalid_payload" });
  }
  const claimed = String(envelope.payload_sha256 || "");
  if (!/^[0-9a-f]{64}$/.test(claimed) || sha256Hex(canonicalJson(payload)) !== claimed) {
    throw Object.assign(new Error("bad payload hash"), { error_class: "payload_sha256_mismatch" });
  }
  return true;
}

async function ensureStream(jsm) {
  const { RetentionPolicy, StorageType, DiscardPolicy } = loadNats();
  try {
    await jsm.streams.info(CONFIG.stream);
  } catch (_) {
    await jsm.streams.add({
      name: CONFIG.stream,
      subjects: [`${CONFIG.subjectPrefix}.>`],
      retention: RetentionPolicy.Limits,
      storage: StorageType.File,
      discard: DiscardPolicy.Old,
    });
  }
}

async function publishEnvelope(js, envelope) {
  const SC = loadNats().StringCodec();
  const msgID = `${envelope.source_ref}:${envelope.payload_sha256}`;
  await js.publish(envelope.subject, SC.encode(JSON.stringify(envelope)), { msgID });
}

function attachReplayProvenance(err, provenance) {
  if (err && typeof err === "object") {
    err.session_provenance = provenance;
    return err;
  }
  const wrapped = new Error("session processing failed");
  wrapped.error_class = "session_processing_failed";
  wrapped.session_provenance = provenance;
  return wrapped;
}

function isPermanentReject(err) {
  return !!(err && PERMANENT_REJECT_ERROR_CLASSES.has(err.error_class));
}

function replayProvenance(profile, file, key, heldOffset, nextOffset, records) {
  return {
    profile: profile.name,
    state_key: key,
    session_file: path.basename(file),
    file_sha256: sha256Hex(file),
    byte_start: records[0].byte_start,
    byte_end: records[records.length - 1].byte_end,
    held_offset: heldOffset,
    next_offset: nextOffset,
    source_ref: sessionSourceRef(profile, file, records[0].byte_start, records[records.length - 1].byte_end),
  };
}

function appendPermanentRejectDlq(profile, file, err, provenance) {
  append0600(CONFIG.dlqFile, {
    ts: new Date().toISOString(),
    reason: "permanent_reject",
    error_class: err && err.error_class ? err.error_class : err && err.name ? err.name : "Error",
    profile: profile.name,
    file_sha256: sha256Hex(file),
    replay: provenance,
  });
}

function combineResults(left, right) {
  return {
    scanned: 0,
    accepted: left.accepted + right.accepted,
    published: left.published + right.published,
    skipped: left.skipped + right.skipped,
    seeded: false,
    dlq: (left.dlq || 0) + (right.dlq || 0),
  };
}

async function processRecordWindow(profile, file, key, state, js, records, heldOffset, nextOffset, now) {
  let envelope = null;
  const provenance = replayProvenance(profile, file, key, heldOffset, nextOffset, records);
  try {
    // Fail closed on source times before production-only summarizer/NATS loads
    // so permanent-reject can advance offset without sibling dependencies.
    sourceCreatedAtFromRecords(records, now);
    const summary = await loadSummarizer().summarizeThread(records, {});
    if (summary.is_noise === true) {
      state[key].offset = records[records.length - 1].byte_end;
      return { scanned: 0, accepted: records.length, published: 0, skipped: records.length, seeded: false, dlq: 0 };
    }
    envelope = buildEnvelope(
      profile,
      file,
      records[0].byte_start,
      records[records.length - 1].byte_end,
      records,
      summary,
      now
    );
    provenance.source_ref = envelope.source_ref;
    validateEnvelope(envelope);
    assertNoSecrets(envelope);
    if (!DRY_RUN) await publishEnvelope(js, envelope);
    state[key].offset = records[records.length - 1].byte_end;
    return {
      scanned: 0,
      accepted: records.length,
      published: DRY_RUN ? 0 : 1,
      skipped: 0,
      seeded: false,
      dlq: 0,
      source_ref: envelope.source_ref,
    };
  } catch (err) {
    if (!isPermanentReject(err)) throw attachReplayProvenance(err, provenance);
    if (records.length > 1) {
      const mid = Math.max(1, Math.floor(records.length / 2));
      const left = await processRecordWindow(
        profile,
        file,
        key,
        state,
        js,
        records.slice(0, mid),
        heldOffset,
        records[mid - 1].byte_end,
        now
      );
      const right = await processRecordWindow(
        profile,
        file,
        key,
        state,
        js,
        records.slice(mid),
        records[mid - 1].byte_end,
        nextOffset,
        now
      );
      return combineResults(left, right);
    }
    appendPermanentRejectDlq(profile, file, err, provenance);
    state[key].offset = records[0].byte_end;
    return { scanned: 0, accepted: 0, published: 0, skipped: 1, seeded: false, dlq: 1 };
  }
}

async function processFile(profile, file, state, js) {
  const key = stateKey(profile, file);
  const stat = fs.statSync(file);
  if (!state[key]) {
    state[key] = { offset: BACKFILL ? 0 : stat.size };
    if (BACKFILL) {
      // Continue below and process from byte 0 on the first explicit backfill run.
    } else {
      return { scanned: 0, accepted: 0, published: 0, skipped: 0, seeded: true };
    }
  }
  let offset = Number(state[key].offset || 0);
  if (stat.size < offset) offset = 0;
  const { lines, nextOffset } = readNewLines(file, offset, CONFIG.maxBytesPerBatch);
  if (lines.length === 0) {
    state[key].offset = nextOffset;
    return { scanned: 0, accepted: 0, published: 0, skipped: 0, seeded: false };
  }
  const records = lines
    .map((line) => parseSessionLine(line.text, profile, file, line.startByte, line.endByte))
    .filter(Boolean);
  if (records.length === 0) {
    state[key].offset = nextOffset;
    return { scanned: lines.length, accepted: 0, published: 0, skipped: lines.length, seeded: false };
  }

  try {
    const result = await processRecordWindow(profile, file, key, state, js, records, offset, nextOffset);
    state[key].offset = nextOffset;
    return {
      scanned: lines.length,
      accepted: result.accepted,
      published: result.published,
      skipped: result.skipped,
      seeded: false,
      dlq: result.dlq || 0,
      source_ref: result.source_ref,
    };
  } catch (err) {
    throw err;
  }
}

async function runOnce() {
  const profiles = loadProfiles();
  const state = loadJson(CONFIG.stateFile, {});
  const totals = { profiles: profiles.length, files: 0, scanned: 0, accepted: 0, published: 0, skipped: 0, seeded: 0, dlq: 0, errors: 0 };
  const { connect } = loadNats();
  const nc = DRY_RUN
    ? null
    : await connect({
        servers: CONFIG.natsUrl,
        name: "argus-session-nats-producer",
        timeout: CONFIG.connectTimeoutMs,
        reconnect: false,
        maxReconnectAttempts: 0,
        ...(CONFIG.natsToken ? { user: CONFIG.natsUser, pass: CONFIG.natsToken } : {}),
      });
  try {
    let js = null;
    if (nc) {
      const jsm = await nc.jetstreamManager();
      await ensureStream(jsm);
      js = nc.jetstream();
    }
    for (const profile of profiles) {
      for (const file of listSessionFiles(profile)) {
        totals.files++;
        try {
          const res = await processFile(profile, file, state, js);
          totals.scanned += res.scanned;
          totals.accepted += res.accepted;
          totals.published += res.published;
          totals.skipped += res.skipped;
          if (res.seeded) totals.seeded++;
          totals.dlq += res.dlq || 0;
        } catch (err) {
          totals.errors++;
          append0600(CONFIG.dlqFile, {
            ts: new Date().toISOString(),
            reason: "process_file_error",
            error_class: err && err.error_class ? err.error_class : err && err.name ? err.name : "Error",
            profile: profile.name,
            file_sha256: sha256Hex(file),
            replay: err && err.session_provenance ? err.session_provenance : null,
          });
        }
      }
    }
  } finally {
    if (nc) await nc.drain();
  }
  writeJsonAtomic(CONFIG.stateFile, state);
  writeJsonAtomic(CONFIG.metricsFile, { last_run_at: new Date().toISOString(), dry_run: DRY_RUN, backfill: BACKFILL, ...totals });
  console.log(
    `[session-producer] once: profiles=${totals.profiles} files=${totals.files} scanned=${totals.scanned} ` +
      `accepted=${totals.accepted} published=${totals.published} skipped=${totals.skipped} seeded=${totals.seeded} dlq=${totals.dlq} errors=${totals.errors}`
  );
  return totals;
}

module.exports = {
  CONFIG,
  TIMESTAMP_ERROR_CLASSES,
  canonicalJson,
  sha256Hex,
  loadProfiles,
  extractText,
  parseSessionLine,
  readNewLines,
  sessionSourceRef,
  sourceCreatedAtFromRecords,
  parseUsableSourceInstant,
  buildEnvelope,
  validateEnvelope,
  isPermanentReject,
  processRecordWindow,
  processFile,
  ensureStream,
  runOnce,
};

if (require.main === module) {
  if (!ONCE) {
    console.error("[session-producer] pass --once; launchd owns cadence");
    process.exit(2);
  }
  runOnce()
    .then((totals) => process.exit(totals.errors ? 1 : 0))
    .catch((err) => {
      console.error(`[session-producer] FATAL error_class=${err && err.error_class ? err.error_class : err && err.name ? err.name : "Error"}`);
      process.exit(1);
    });
}

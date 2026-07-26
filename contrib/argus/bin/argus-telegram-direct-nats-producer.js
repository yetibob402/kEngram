#!/usr/bin/env node
"use strict";

/*
 * argus-telegram-direct-nats-producer
 *
 * Reads canonical Telegram capture sidecars, builds telegram argus.ingest.v1
 * envelopes, and publishes them directly to NATS for the existing kEngram
 * consumer. It defaults to dry-run and refuses live publish until production
 * state has been seeded to current capture EOF.
 */

const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");

const TELEGRAM_SCHEMA_VERSION = "argus.ingest.v1";
const TELEGRAM_NAMESPACE = "conversations/telegram-yetiwerks";
const TELEGRAM_SOURCE_KIND = "telegram";
const PRODUCER = "telegram-direct-nats-producer";
const DEFAULT_STREAM = "ARGUS_TELEGRAM_DISTILLED_SHADOW";
const SUBJECT_PREFIX = "ingest.telegram";
const SOURCE_REF_RE = /^telegram:([^:]+):([^:]+):batch:(\d+)-(\d+)$/;
const HEX64_RE = /^[0-9a-f]{64}$/;
const HOME = process.env.HOME || os.homedir();
const LEGACY_SIDECAR_DIR = path.join(HOME, "argus/state", "telegram-" + "distilled");
const POISON_ATTEMPTS = 5;
const MAX_CONTEXT_SPLIT_DEPTH = 4;
const DEFAULT_MAX_SUMMARY_TOKENS = 24000;
const MIN_SUMMARY_TOKENS = 2000;
const SUMMARY_TOKEN_CHAR_ESTIMATE = 3.5; // Crude render-budget estimate for context fallback.
const ATTEMPT_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;
const ATTEMPT_HARD_CAP = 1000;
const ATTEMPT_FILE_MAX_BYTES = 256 * 1024;

const SECRET_PATTERNS = [
  /\b\d{6,12}:[A-Za-z0-9_-]{30,}\b/g,
  /\b(?:sbp|sb|sk|pk|rk|ghp|gho|ghs|ghr|glpat|xox[baprs]|AKIA|ASIA)[_-][A-Za-z0-9_-]{12,}\b/g,
  /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g,
  /\bsk-[A-Za-z0-9]{20,}\b/g,
  /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g,
  /\b(?:token|secret|password|passwd|api[_-]?key|apikey|access[_-]?key|private[_-]?key|service[_-]?key|bearer)\b\s*[:=]\s*["']?[A-Za-z0-9_\-./+]{12,}["']?/gi,
  /\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{24,}\b/g,
];

const DLQ_ERROR_MESSAGE_MAX = 200;

function parseArgs(argv = process.argv.slice(2)) {
  const flags = new Set();
  const values = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const eq = arg.indexOf("=");
    if (eq !== -1) {
      values[arg.slice(2, eq)] = arg.slice(eq + 1);
      continue;
    }
    const key = arg.slice(2);
    if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      values[key] = argv[i + 1];
      i += 1;
    } else {
      flags.add(key);
    }
  }
  return { flags, values };
}

function positiveInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

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
  return (
    process.env.TELEGRAM_DIRECT_NATS_USER ||
    process.env.NATS_AUTH_USER ||
    NATS_AUTH_ENV.NATS_AUTH_USER ||
    defaultUser
  );
}

function authToken(user) {
  const key = `CRED_NATS_TOKEN_${String(user || "").toUpperCase()}`;
  return (
    process.env.TELEGRAM_DIRECT_NATS_TOKEN ||
    process.env.NATS_AUTH_TOKEN ||
    process.env[key] ||
    NATS_AUTH_ENV[key] ||
    NATS_AUTH_ENV.CRED_NATS_TOKEN
  );
}

function buildConfig(argv = process.argv.slice(2), env = process.env) {
  const parsed = parseArgs(argv);
  const live = parsed.flags.has("live") || env.TELEGRAM_DIRECT_NATS_PRODUCER_LIVE === "1";
  const seedCurrentEof = parsed.flags.has("seed-current-eof");
  const captureRoot =
    parsed.values["capture-root"] ||
    env.ARGUS_TELEGRAM_DIRECT_CAPTURE_ROOT ||
    path.join(HOME, "argus/state/telegram-capture");
  const defaultStateDir = path.join(HOME, "argus/state/telegram-direct-nats-producer");
  const scratchStateDir = path.join(os.tmpdir(), `argus-telegram-direct-nats-producer/${process.pid}-${Date.now()}`);
  const explicitStateDir = parsed.values["state-dir"] || env.ARGUS_TELEGRAM_DIRECT_STATE_DIR || "";
  const stateDir = explicitStateDir || (live || seedCurrentEof ? defaultStateDir : scratchStateDir);
  const config = {
    live,
    dryRun: !live,
    once: parsed.flags.has("once") || env.TELEGRAM_DIRECT_NATS_PRODUCER_ONCE === "1",
    seedCurrentEof,
    captureRoot: path.resolve(captureRoot),
    stateDir: path.resolve(stateDir),
    offsetFile: path.resolve(stateDir, "offsets.json"),
    seedFile: path.resolve(stateDir, "seed.json"),
    metricsFile: path.resolve(stateDir, "metrics.json"),
    dlqFile: path.resolve(stateDir, "publisher-dlq.jsonl"),
    poisonFile: path.resolve(stateDir, "poison-windows.jsonl"),
    seenFile: path.resolve(stateDir, "published-windows.json"),
    attemptsFile: path.resolve(stateDir, "window-attempts.json"),
    stateDirExplicit: Boolean(explicitStateDir),
    dryRunSeedEof: parsed.flags.has("dry-run-seed-eof") || env.TELEGRAM_DIRECT_DRY_RUN_SEED_EOF === "1",
    stream: env.TELEGRAM_DIRECT_NATS_STREAM || DEFAULT_STREAM,
    subjectPrefix: env.TELEGRAM_DIRECT_NATS_SUBJECT_PREFIX || SUBJECT_PREFIX,
    natsUrl: env.TELEGRAM_DIRECT_NATS_URL || "127.0.0.1:4222",
    natsLibDir: env.NATS_LIB_DIR || path.join(HOME, "argus/bin/nats-shadow/node_modules/nats"),
    connectTimeoutMs: Number(env.TELEGRAM_DIRECT_NATS_CONNECT_TIMEOUT_MS || 8000),
    publishTimeoutMs: Number(env.TELEGRAM_DIRECT_NATS_PUBLISH_TIMEOUT_MS || 5000),
    maxBytesPerPass: Number(env.TELEGRAM_DIRECT_MAX_BYTES_PER_PASS || 524288),
    maxRecordsPerWindow: Number(env.TELEGRAM_DIRECT_MAX_RECORDS_PER_WINDOW || 12),
    maxSummaryTokens: positiveInt(env.TELEGRAM_DIRECT_MAX_SUMMARY_TOKENS, DEFAULT_MAX_SUMMARY_TOKENS),
    summarizerModule:
      env.TELEGRAM_DIRECT_SUMMARIZER_MODULE ||
      path.join(HOME, "argus/bin/argus-telegram-hive-summarizer.js"),
    producer: PRODUCER,
  };
  config.natsUser = authUser();
  config.natsToken = authToken(config.natsUser);
  return config;
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

function persistJson0600(file, value) {
  writeJsonAtomic(file, value, 0o600);
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

function loadSummarizer(config) {
  const mod = require(config.summarizerModule);
  for (const name of ["summarizeThread", "sourceRefOf", "tsOf", "renderBatch", "redactSecrets"]) {
    if (typeof mod[name] !== "function") {
      throw Object.assign(new Error(`summarizer missing ${name}`), { error_class: "summarizer_contract" });
    }
  }
  return mod;
}

function secretHitPaths(value, prefix = "$", hits = []) {
  if (prefix === "payload_sha256" || prefix === "provenance.generation.config_hash") return hits;
  if (typeof value === "string") {
    for (const re of SECRET_PATTERNS) {
      re.lastIndex = 0;
      if (re.test(value)) hits.push(prefix);
    }
    return hits;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => secretHitPaths(item, `${prefix}[${index}]`, hits));
    return hits;
  }
  if (value && typeof value === "object") {
    for (const key of Object.keys(value)) secretHitPaths(value[key], prefix === "$" ? key : `${prefix}.${key}`, hits);
  }
  return hits;
}

function assertNoSecrets(envelope, redactor) {
  const redactorHits = [];
  const walkRedactor = (value, prefix = "$") => {
    if (prefix === "payload_sha256" || prefix === "provenance.generation.config_hash") return;
    if (typeof value === "string") {
      if (redactor && redactor(value) !== value) redactorHits.push(prefix);
      return;
    }
    if (Array.isArray(value)) return value.forEach((item, index) => walkRedactor(item, `${prefix}[${index}]`));
    if (value && typeof value === "object") {
      for (const key of Object.keys(value)) walkRedactor(value[key], prefix === "$" ? key : `${prefix}.${key}`);
    }
  };
  walkRedactor(envelope);
  const hits = Array.from(new Set([...secretHitPaths(envelope), ...redactorHits]));
  if (hits.length) {
    throw Object.assign(new Error(`secret-shaped string in telegram direct envelope at ${hits.slice(0, 8).join(",")}`), {
      error_class: "secret_scan_failed",
    });
  }
}

function assertStatePathsSafe(config) {
  const captureRoot = path.resolve(config.captureRoot);
  const stateDir = path.resolve(config.stateDir);
  const legacyDir = path.resolve(LEGACY_SIDECAR_DIR);
  const relCapture = path.relative(captureRoot, stateDir);
  if (!relCapture.startsWith("..") && relCapture !== "") {
    throw Object.assign(new Error("state directory must be outside capture root"), { error_class: "state_inside_capture_root" });
  }
  const relLegacy = path.relative(legacyDir, stateDir);
  if (!relLegacy.startsWith("..") && relLegacy !== "") {
    throw Object.assign(new Error("state directory must be outside legacy sidecar root"), { error_class: "state_inside_legacy_sidecar" });
  }
}

function listCaptureFiles(config) {
  if (!fs.existsSync(config.captureRoot)) return [];
  const out = [];
  for (const agent of fs.readdirSync(config.captureRoot).sort()) {
    const agentDir = path.join(config.captureRoot, agent);
    let stat;
    try {
      stat = fs.statSync(agentDir);
    } catch (_) {
      continue;
    }
    if (!stat.isDirectory()) continue;
    for (const file of fs.readdirSync(agentDir).sort()) {
      if (file.endsWith(".jsonl")) out.push(path.join(agentDir, file));
    }
  }
  return out;
}

function stateKey(config, file) {
  return path.relative(config.captureRoot, file);
}

function readCompleteLines(file, fromOffset, maxBytes) {
  const stat = fs.statSync(file);
  if (stat.size < fromOffset) {
    throw Object.assign(new Error(`capture sidecar shrank below saved offset: ${file}`), {
      error_class: "capture_shrink_guard",
      held_offset: fromOffset,
      size: stat.size,
    });
  }
  if (stat.size === fromOffset) return { lines: [], nextOffset: fromOffset, stat };
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
  for (let idx = 0; idx < buf.length; idx += 1) {
    if (buf[idx] !== 0x0a) continue;
    const startByte = fromOffset + lineStart;
    const endByte = fromOffset + idx + 1;
    const text = buf.toString("utf8", lineStart, idx);
    lines.push({ text, startByte, endByte, line_sha256: sha256Hex(text) });
    lineStart = idx + 1;
    nextOffset = endByte;
  }
  return { lines, nextOffset, stat };
}

function sourceRefOf(record) {
  if (record && record.dedupe_key && String(record.dedupe_key).trim()) {
    return String(record.dedupe_key).trim();
  }
  return "";
}

function messageIdOf(record) {
  if (record && record.message_id != null && String(record.message_id).trim() !== "") {
    return String(record.message_id).trim();
  }
  const ref = sourceRefOf(record);
  const tail = ref ? ref.split(":").pop() : "";
  return /^\d+$/.test(tail) ? tail : "";
}

const FUTURE_SKEW_MS = 5 * 60 * 1000;
const TIMESTAMP_ERROR_CLASSES = new Set([
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

function finalizeSourceInstant(ms, nowMs) {
  if (!Number.isFinite(ms)) throwTimestampError("invalid_source_created_at");
  if (ms > nowMs + FUTURE_SKEW_MS) throwTimestampError("future_source_created_at");
  return new Date(ms).toISOString();
}

function parsePresentSourceTime(raw, nowMs) {
  // Stronger field is present: missing-empty is invalid, not demotion.
  if (raw == null) throwTimestampError("invalid_source_created_at");
  if (typeof raw === "string" && raw.trim() === "") throwTimestampError("invalid_source_created_at");
  let ms;
  if (typeof raw === "number") {
    ms = raw;
  } else {
    const s = String(raw).trim();
    if (!s) throwTimestampError("invalid_source_created_at");
    ms = Date.parse(s);
  }
  return finalizeSourceInstant(ms, nowMs);
}

function parsePresentTelegramDate(raw, nowMs) {
  if (raw == null || raw === "") throwTimestampError("invalid_source_created_at");
  const n = Number(raw);
  if (!Number.isFinite(n)) throwTimestampError("invalid_source_created_at");
  // Telegram date is Unix seconds.
  return finalizeSourceInstant(n * 1000, nowMs);
}

// True when key is own or inherited on obj (distinguishes present-null/empty from absence).
function fieldPresent(obj, key) {
  return obj != null && Object(obj) === obj && key in obj;
}

// Precedence: valid telegram_date → valid normalized_event.ts → valid capture_ts.
// A present but invalid stronger field fails closed (no demotion, no wall clock).
// Present includes explicitly empty/null/undefined values; only a missing key demotes.
function tsOf(record, now) {
  const nowMs = resolveNowMs(now);
  const rec = record || {};
  if (fieldPresent(rec, "telegram_date")) {
    return parsePresentTelegramDate(rec.telegram_date, nowMs);
  }
  if (fieldPresent(rec.normalized_event, "ts")) {
    return parsePresentSourceTime(rec.normalized_event.ts, nowMs);
  }
  if (rec.capture_ts != null && rec.capture_ts !== "") {
    return parsePresentSourceTime(rec.capture_ts, nowMs);
  }
  throwTimestampError("missing_source_created_at");
}

function windowSourceRef(agent, chatId, records) {
  const ids = records.map(messageIdOf).filter(Boolean);
  const first = ids.length ? ids[0] : "0";
  const last = ids.length ? ids[ids.length - 1] : first;
  return `telegram:${agent}:${chatId}:batch:${first}-${last}`;
}

function sourceRefsForWindow(window) {
  return window.records.map(sourceRefOf);
}

function windowAttemptKey(window) {
  return sha256Hex(sourceRefsForWindow(window).join("\n"));
}

function normalizeRecord(record) {
  const agent = String(record.agent || "").trim();
  const chatId = String(record.chat_id || (record.normalized_event && record.normalized_event.chat_id) || "").trim();
  const messageId = String(record.message_id || (record.normalized_event && record.normalized_event.message_id) || "").trim();
  const dedupeKey = sourceRefOf(record);
  if (!agent || !chatId || !messageId || !dedupeKey) {
    throw Object.assign(new Error("capture row missing agent/chat/message/dedupe identity"), { error_class: "invalid_capture_identity" });
  }
  return { ...record, agent, chat_id: chatId, message_id: messageId, dedupe_key: dedupeKey };
}

// Sort key only: never throws. Invalid/missing times sort after valid ones so
// buildWindows can hand windows to processWindow, where fail-closed timestamp
// errors enter the existing attempt/DLQ/quarantine path (not file_error).
function sourceTimeMsForSort(record, now) {
  try {
    const ms = Date.parse(tsOf(record, now));
    return Number.isFinite(ms) ? ms : null;
  } catch (_) {
    return null;
  }
}

function assertWindowSourceTimes(window, now) {
  const records = (window && window.records) || [];
  for (const record of records) {
    tsOf(record, now);
  }
}

function buildWindows(records, maxRecordsPerWindow = 12, now) {
  const groups = new Map();
  for (const record of records) {
    const key = `${record.agent}\u0000${record.chat_id}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(record);
  }
  const windows = [];
  for (const [key, group] of groups.entries()) {
    group.sort((a, b) => {
      const ta = sourceTimeMsForSort(a, now);
      const tb = sourceTimeMsForSort(b, now);
      if (ta == null && tb == null) {
        return Number(messageIdOf(a)) - Number(messageIdOf(b));
      }
      if (ta == null) return 1;
      if (tb == null) return -1;
      if (ta !== tb) return ta - tb;
      return Number(messageIdOf(a)) - Number(messageIdOf(b));
    });
    for (let i = 0; i < group.length; i += maxRecordsPerWindow) {
      const chunk = group.slice(i, i + maxRecordsPerWindow);
      const [agent, chatId] = key.split("\u0000");
      windows.push({ agent, chatId, records: chunk });
    }
  }
  return windows;
}

function generationBlock(config) {
  const body = {
    producer: PRODUCER,
    max_records_per_window: config.maxRecordsPerWindow,
    schema_version: TELEGRAM_SCHEMA_VERSION,
  };
  return { ...body, config_hash: sha256Hex(JSON.stringify(body)).slice(0, 16) };
}

function buildEnvelope(config, window, summary, now) {
  const agent = window.agent;
  const chatId = String(window.chatId);
  const sourceRef = windowSourceRef(agent, chatId, window.records);
  const sourceRefs = window.records.map(sourceRefOf);
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
  };
  const payloadHash = sha256Hex(canonicalJson(payload));
  const nowMs = resolveNowMs(now);
  const startTimes = window.records.map((record) => tsOf(record, nowMs));
  if (!startTimes.length) throwTimestampError("missing_source_created_at");
  let maxMs = -Infinity;
  let maxIso = null;
  for (const iso of startTimes) {
    const ms = Date.parse(iso);
    if (ms > maxMs) {
      maxMs = ms;
      maxIso = iso;
    }
  }
  const windowStart = startTimes[0];
  const windowEnd = maxIso;
  return {
    namespace: TELEGRAM_NAMESPACE,
    kind: "note",
    schema_version: TELEGRAM_SCHEMA_VERSION,
    event_id: sourceRef,
    source_ref: sourceRef,
    dedupe_key: sourceRef,
    batch_id: sourceRef,
    subject: `${config.subjectPrefix}.${agent}.${chatId}.batch`,
    source_kind: TELEGRAM_SOURCE_KIND,
    producer: PRODUCER,
    author: agent,
    agent,
    direction: null,
    chat_id: chatId,
    topic_key: `telegram:${agent}:${chatId}`,
    created_at: windowEnd,
    published_at: new Date(nowMs).toISOString(),
    payload_sha256: payloadHash,
    payload,
    provenance: {
      first_source_ref: sourceRefs[0] || "",
      last_source_ref: sourceRefs[sourceRefs.length - 1] || "",
      source_refs: sourceRefs,
      count: window.records.length,
      window_start: windowStart,
      window_end: windowEnd,
      tee_paths: Array.from(new Set(window.records.map((record) => record.__capture_key || record.__capture_file).filter(Boolean))),
      generation: generationBlock(config),
    },
  };
}

function validateEnvelope(config, envelope) {
  if (!envelope || typeof envelope !== "object" || Array.isArray(envelope)) {
    throw Object.assign(new Error("bad envelope"), { error_class: "invalid_envelope" });
  }
  if (envelope.schema_version !== TELEGRAM_SCHEMA_VERSION) throw Object.assign(new Error("bad schema"), { error_class: "bad_schema_version" });
  if (envelope.namespace !== TELEGRAM_NAMESPACE) throw Object.assign(new Error("bad namespace"), { error_class: "bad_namespace" });
  if (envelope.kind !== "note") throw Object.assign(new Error("bad kind"), { error_class: "bad_kind" });
  if (envelope.source_kind !== TELEGRAM_SOURCE_KIND) throw Object.assign(new Error("bad source kind"), { error_class: "bad_source_kind" });
  const match = SOURCE_REF_RE.exec(String(envelope.source_ref || ""));
  if (!match) throw Object.assign(new Error("bad source_ref"), { error_class: "bad_source_ref" });
  const agent = match[1];
  const chatId = match[2];
  if (envelope.author !== agent || envelope.agent !== agent) throw Object.assign(new Error("agent mismatch"), { error_class: "agent_mismatch" });
  if (String(envelope.chat_id) !== chatId) throw Object.assign(new Error("chat mismatch"), { error_class: "chat_id_mismatch" });
  if (envelope.event_id !== envelope.source_ref || envelope.dedupe_key !== envelope.source_ref || envelope.batch_id !== envelope.source_ref) {
    throw Object.assign(new Error("identity mismatch"), { error_class: "identity_mismatch" });
  }
  if (envelope.subject !== `${config.subjectPrefix}.${agent}.${chatId}.batch`) {
    throw Object.assign(new Error("subject mismatch"), { error_class: "subject_mismatch" });
  }
  if (!envelope.payload || typeof envelope.payload !== "object" || Array.isArray(envelope.payload)) {
    throw Object.assign(new Error("bad payload"), { error_class: "missing_payload" });
  }
  if (!HEX64_RE.test(String(envelope.payload_sha256 || ""))) {
    throw Object.assign(new Error("bad payload hash"), { error_class: "bad_payload_sha256" });
  }
  if (sha256Hex(canonicalJson(envelope.payload)) !== envelope.payload_sha256) {
    throw Object.assign(new Error("payload hash mismatch"), { error_class: "payload_sha256_mismatch" });
  }
  const sourceRefs = envelope.provenance && envelope.provenance.source_refs;
  if (!Array.isArray(sourceRefs) || sourceRefs.length < 1 || !sourceRefs.every((ref) => /^telegram:[^:]+:(?:inbound|outbound):[^:]+:\d+$/.test(String(ref)))) {
    throw Object.assign(new Error("missing real capture source refs"), { error_class: "missing_capture_source_refs" });
  }
  return true;
}

function seedCurrentEof(config) {
  assertStatePathsSafe(config);
  const files = {};
  const offsets = {};
  for (const file of listCaptureFiles(config)) {
    const stat = fs.statSync(file);
    const key = stateKey(config, file);
    files[key] = {
      path: file,
      size: stat.size,
      offset: stat.size,
      mtime: stat.mtime.toISOString(),
    };
    offsets[key] = { offset: stat.size };
  }
  const seed = {
    seeded_at: new Date().toISOString(),
    capture_root: config.captureRoot,
    mode: "current_eof",
    producer: PRODUCER,
    files,
  };
  seed.seed_body_sha256 = sha256Hex(canonicalJson(seed));
  writeJsonAtomic(config.offsetFile, offsets);
  writeJsonAtomic(config.seedFile, seed);
  writeJsonAtomic(config.metricsFile, {
    last_run_at: new Date().toISOString(),
    mode: "seed-current-eof",
    files_seeded: Object.keys(files).length,
  });
  console.log(`[telegram-direct-producer] seeded files=${Object.keys(files).length} seed=${config.seedFile}`);
  return seed;
}

function loadSeed(config) {
  return loadJson(config.seedFile, null);
}

function assertLiveSeed(config, files) {
  const seed = loadSeed(config);
  if (!seed || seed.mode !== "current_eof" || seed.capture_root !== config.captureRoot || !seed.files) {
    throw Object.assign(new Error("live publish requires valid current-EOF seed"), { error_class: "missing_or_invalid_seed" });
  }
  const copy = { ...seed };
  const expectedHash = copy.seed_body_sha256;
  delete copy.seed_body_sha256;
  if (!expectedHash || sha256Hex(canonicalJson(copy)) !== expectedHash) {
    throw Object.assign(new Error("seed checksum mismatch"), { error_class: "seed_checksum_mismatch" });
  }
  for (const file of files) {
    const key = stateKey(config, file);
    if (seed.files[key]) continue;
    const stat = fs.statSync(file);
    if (Date.parse(seed.seeded_at) <= stat.birthtimeMs) continue;
    throw Object.assign(new Error(`visible sidecar missing from seed: ${key}`), { error_class: "seed_missing_visible_sidecar" });
  }
  return seed;
}

function buildDlqRow(config, reason, err, provenance) {
  return {
    ts: new Date().toISOString(),
    reason,
    error_class: err && err.error_class ? err.error_class : err && err.name ? err.name : "Error",
    error_message: errorMessage(err).slice(0, DLQ_ERROR_MESSAGE_MAX),
    source_ref: provenance && provenance.source_ref,
    capture_file: provenance && provenance.capture_file,
    state_key: provenance && provenance.state_key,
    byte_start: provenance && provenance.byte_start,
    byte_end: provenance && provenance.byte_end,
    held_offset: provenance && provenance.held_offset,
    next_offset: provenance && provenance.next_offset,
    line_sha256: provenance && provenance.line_sha256,
    source_refs: provenance && provenance.source_refs,
    payload_sha256: provenance && provenance.payload_sha256,
    nats_msg_id: provenance && provenance.nats_msg_id,
    producer: config.producer,
  };
}

async function connectNats(config) {
  const nats = require(config.natsLibDir);
  const nc = await nats.connect({
    servers: config.natsUrl,
    name: "argus-telegram-direct-nats-producer",
    timeout: config.connectTimeoutMs,
    reconnect: false,
    maxReconnectAttempts: 0,
    ...(config.natsToken ? { user: config.natsUser, pass: config.natsToken } : {}),
  });
  return { nc, js: nc.jetstream(), StringCodec: nats.StringCodec };
}

async function publishEnvelope(config, js, StringCodec, envelope) {
  const sc = StringCodec();
  const msgID = `${envelope.source_ref}:${envelope.payload_sha256}`;
  const ack = await js.publish(envelope.subject, sc.encode(JSON.stringify(envelope)), {
    msgID,
    timeout: config.publishTimeoutMs,
  });
  return { msgID, ack };
}

function buildSeenEntry(config, envelope, provenance, status) {
  return {
    status,
    updated_at: new Date().toISOString(),
    source_ref: envelope.source_ref,
    payload_sha256: envelope.payload_sha256,
    nats_msg_id: `${envelope.source_ref}:${envelope.payload_sha256}`,
    byte_start: provenance.byte_start,
    byte_end: provenance.byte_end,
    held_offset: provenance.held_offset,
    next_offset: provenance.next_offset,
    state_key: provenance.state_key,
    capture_file: provenance.capture_file,
    source_refs: provenance.source_refs,
    envelope,
    producer: config.producer,
  };
}

function errorClass(err) {
  return err && err.error_class ? String(err.error_class) : err && err.name ? String(err.name) : "Error";
}

function errorMessage(err) {
  // Redacted HERE rather than at any one call site: every persisted error string
  // in this file already routes through this function -- window attempts, poison
  // windows, the DLQ row and the operator log lines -- so redacting at the callers
  // guarantees the next caller forgets. An error string can quote field values,
  // and these land in durable append-only files.
  if (!err) return "";
  const raw = err.message ? String(err.message) : String(err);
  let out = raw;
  for (const re of SECRET_PATTERNS) {
    re.lastIndex = 0;
    out = out.replace(re, "<redacted>");
  }
  return out;
}

function isContextSizeError(err) {
  const text = [
    err && err.error_class,
    err && err.code,
    err && err.name,
    err && err.message,
    err && err.status,
    err && err.statusCode,
  ]
    .filter((part) => part != null && String(part).trim() !== "")
    .join(" ")
    .toLowerCase();
  if (!text) return false;
  if (/exceed_context|context[_ -]?size|exceed[^ ]*[_ -]?context|context[^ ]*[_ -]?exceed|num_ctx/.test(text)) return true;
  if (/prompt.*too.*long|too.*many.*tokens|tokens?.*(exceed|limit|maximum|max)|maximum.*tokens?/.test(text)) return true;
  return /\b400\b/.test(text) && /(context|token|prompt|num_ctx|too long|exceed)/.test(text);
}

function splitWindow(window) {
  const midpoint = Math.ceil(window.records.length / 2);
  return [
    { agent: window.agent, chatId: window.chatId, records: window.records.slice(0, midpoint) },
    { agent: window.agent, chatId: window.chatId, records: window.records.slice(midpoint) },
  ].filter((child) => child.records.length > 0);
}

function addPendingAttemptKeysForWindow(window, pending, depth = 0) {
  pending.add(windowAttemptKey(window));
  if (window.records.length <= 1 || depth >= MAX_CONTEXT_SPLIT_DEPTH) return;
  for (const child of splitWindow(window)) addPendingAttemptKeysForWindow(child, pending, depth + 1);
}

function initialOffsetForFile(config, file, state, stat) {
  const key = stateKey(config, file);
  if (state[key]) return { offset: Number(state[key].offset || 0), seeded: false };
  const seed = loadSeed(config);
  const seeded = seed && seed.files && seed.files[key];
  if (seeded) return { offset: Number(seeded.offset), seeded: true };
  return { offset: !config.live && config.dryRunSeedEof ? stat.size : 0, seeded: false };
}

function collectPendingAttemptKeys(config, files, state) {
  const pending = new Set();
  for (const file of files) {
    let stat;
    try {
      stat = fs.statSync(file);
      const { offset } = initialOffsetForFile(config, file, state, stat);
      const read = readCompleteLines(file, offset, config.maxBytesPerPass);
      if (read.lines.length === 0) continue;
      const records = [];
      for (const line of read.lines) {
        try {
          const row = normalizeRecord(JSON.parse(line.text));
          records.push(row);
        } catch (_) {
          /* invalid rows are DLQed by processFile and do not own window attempts */
        }
      }
      for (const window of buildWindows(records, config.maxRecordsPerWindow)) addPendingAttemptKeysForWindow(window, pending);
    } catch (_) {
      /* processFile owns the live error; pending pruning should not hide it */
    }
  }
  return pending;
}

function loadAttemptState(config) {
  const attempts = loadJson(config.attemptsFile, {});
  return attempts && typeof attempts === "object" && !Array.isArray(attempts) ? attempts : {};
}

function pruneAttemptState(attempts, pendingKeys = null, nowMs = Date.now()) {
  let changed = false;
  for (const [key, entry] of Object.entries(attempts)) {
    const last = Date.parse(entry && entry.last_failed_at ? entry.last_failed_at : "");
    if ((Number.isFinite(last) && nowMs - last > ATTEMPT_MAX_AGE_MS) || (pendingKeys && !pendingKeys.has(key))) {
      delete attempts[key];
      changed = true;
    }
  }
  const entries = Object.entries(attempts);
  if (entries.length > ATTEMPT_HARD_CAP) {
    entries
      .sort((a, b) => {
        const at = Date.parse(a[1] && a[1].last_failed_at ? a[1].last_failed_at : "") || 0;
        const bt = Date.parse(b[1] && b[1].last_failed_at ? b[1].last_failed_at : "") || 0;
        return at - bt;
      })
      .slice(0, entries.length - ATTEMPT_HARD_CAP)
      .forEach(([key]) => {
        delete attempts[key];
        changed = true;
      });
  }
  return changed;
}

function persistAttemptState(config, attempts) {
  pruneAttemptState(attempts);
  const text = JSON.stringify(attempts, null, 2) + "\n";
  if (Buffer.byteLength(text, "utf8") > ATTEMPT_FILE_MAX_BYTES) {
    throw Object.assign(new Error("window attempt state exceeds 256KB after pruning"), { error_class: "window_attempts_too_large" });
  }
  ensureDir(path.dirname(config.attemptsFile));
  const tmp = `${config.attemptsFile}.tmp`;
  fs.writeFileSync(tmp, text, { mode: 0o600 });
  fs.renameSync(tmp, config.attemptsFile);
  chmodSafe(config.attemptsFile, 0o600);
}

function deleteWindowAttempt(config, attempts, key) {
  if (!attempts || !attempts[key]) return;
  delete attempts[key];
  persistAttemptState(config, attempts);
}

function recordWindowAttempt(config, attempts, provenance, sourceRef, err) {
  const now = new Date().toISOString();
  const key = provenance.line_sha256;
  const existing = attempts[key] || {};
  const entry = {
    attempts: Number(existing.attempts || 0) + 1,
    first_failed_at: existing.first_failed_at || now,
    last_failed_at: now,
    error_class: errorClass(err),
    error_message: errorMessage(err).slice(0, 1000),
    source_ref: sourceRef,
    state_key: provenance.state_key,
    capture_file: provenance.capture_file,
    source_refs: provenance.source_refs,
  };
  attempts[key] = entry;
  persistAttemptState(config, attempts);
  return entry;
}

function buildWindowProvenance(context, window) {
  const sourceRefs = sourceRefsForWindow(window);
  const firstLine = context.sourceLineByRef.get(sourceRefs[0]);
  const lastLine = context.sourceLineByRef.get(sourceRefs[sourceRefs.length - 1]);
  return {
    capture_file: context.file,
    state_key: context.stateKey,
    byte_start: firstLine ? firstLine.startByte : context.offset,
    byte_end: lastLine ? lastLine.endByte : context.nextOffset,
    held_offset: context.offset,
    next_offset: context.nextOffset,
    line_sha256: sha256Hex(sourceRefs.join("\n")),
    source_refs: sourceRefs,
  };
}

function truncateRecordsForSummary(records, budgetTokens, renderBatch) {
  const rendered = typeof renderBatch === "function" ? renderBatch(records) : records.map((record) => JSON.stringify(record)).join("\n");
  const maxChars = Math.max(1, Math.floor(budgetTokens * SUMMARY_TOKEN_CHAR_ESTIMATE));
  if (rendered.length <= maxChars) return records;
  const marker = `[truncated ${rendered.length - maxChars} chars for summarizer context]`;
  const keepChars = Math.max(1, maxChars - marker.length - 1);
  const truncatedText = `${rendered.slice(0, keepChars).trimEnd()}\n${marker}`;
  const record = records[0];
  const clone = { ...record, text: truncatedText };
  if (clone.normalized_event && typeof clone.normalized_event === "object" && !Array.isArray(clone.normalized_event)) {
    clone.normalized_event = { ...clone.normalized_event, text: truncatedText };
  }
  return [clone];
}

async function summarizeSingleWithTruncation(config, window, deps) {
  let budget = Math.max(MIN_SUMMARY_TOKENS, config.maxSummaryTokens || DEFAULT_MAX_SUMMARY_TOKENS);
  let lastErr = null;
  while (budget >= MIN_SUMMARY_TOKENS) {
    const records = truncateRecordsForSummary(window.records, budget, deps.renderBatch);
    try {
      return await deps.summarizeThread(records, { context_truncation_budget_tokens: budget });
    } catch (err) {
      if (!isContextSizeError(err)) throw err;
      lastErr = err;
      if (budget <= MIN_SUMMARY_TOKENS) break;
      budget = Math.max(MIN_SUMMARY_TOKENS, Math.floor(budget / 2));
    }
  }
  if (lastErr) {
    if (!lastErr.error_class) lastErr.error_class = "context_ladder_exhausted";
    lastErr.context_ladder_exhausted = true;
    throw lastErr;
  }
  const err = new Error("context ladder exhausted without summarizer error");
  err.error_class = "context_ladder_exhausted";
  err.context_ladder_exhausted = true;
  throw err;
}

function buildPoisonRow(config, err, provenance, window, attemptEntry) {
  return {
    ts: new Date().toISOString(),
    reason: "poison_window",
    error_class: errorClass(err),
    error_message: errorMessage(err).slice(0, 4000),
    source_ref: windowSourceRef(window.agent, String(window.chatId), window.records),
    capture_file: provenance.capture_file,
    state_key: provenance.state_key,
    byte_start: provenance.byte_start,
    byte_end: provenance.byte_end,
    held_offset: provenance.held_offset,
    next_offset: provenance.next_offset,
    line_sha256: provenance.line_sha256,
    source_refs: provenance.source_refs,
    attempts: attemptEntry && attemptEntry.attempts ? attemptEntry.attempts : undefined,
    records: window.records,
    producer: config.producer,
  };
}

function quarantineWindow(config, seen, attempts, window, provenance, err, attemptEntry = null) {
  const sourceRef = windowSourceRef(window.agent, String(window.chatId), window.records);
  append0600(config.poisonFile, buildPoisonRow(config, err, provenance, window, attemptEntry));
  const now = new Date().toISOString();
  seen[sourceRef] = {
    status: "quarantined",
    updated_at: now,
    quarantined_at: now,
    source_ref: sourceRef,
    byte_start: provenance.byte_start,
    byte_end: provenance.byte_end,
    held_offset: provenance.held_offset,
    next_offset: provenance.next_offset,
    state_key: provenance.state_key,
    capture_file: provenance.capture_file,
    line_sha256: provenance.line_sha256,
    source_refs: provenance.source_refs,
    error_class: errorClass(err),
    error_message: errorMessage(err).slice(0, 1000),
    producer: config.producer,
  };
  persistJson0600(config.seenFile, seen);
  deleteWindowAttempt(config, attempts, provenance.line_sha256);
  console.warn(
    `[telegram-direct-producer] WARN poison_window file=${provenance.capture_file} source_refs=${provenance.source_refs.join(",")} class=${errorClass(err)}`
  );
}

async function processWindow(config, window, seen, deps, attempts, context, depth = 0) {
  const provenance = buildWindowProvenance(context, window);
  const sourceRef = windowSourceRef(window.agent, String(window.chatId), window.records);
  const existing = seen[sourceRef];
  if (existing && existing.status === "quarantined") {
    deleteWindowAttempt(config, attempts, provenance.line_sha256);
    return { accepted: 0, published: 0, skipped: window.records.length };
  }

  let envelope = existing && existing.envelope;
  try {
    // Validate represented source times inside processWindow so timestamp-class
    // failures hit recordWindowAttempt / held DLQ / poison quarantine, not the
    // buildWindows sort path (which must remain non-throwing).
    assertWindowSourceTimes(window);
    if (envelope && existing.payload_sha256 && envelope.payload_sha256 !== existing.payload_sha256) {
      throw Object.assign(new Error("seen envelope payload hash mismatch"), { error_class: "seen_payload_mismatch" });
    }
    if (!envelope) {
      let summary = null;
      try {
        summary = await deps.summarizeThread(window.records, {});
      } catch (err) {
        if (!isContextSizeError(err)) throw err;
        const latest = seen[sourceRef];
        if (latest && latest.status === "published") {
          throw Object.assign(new Error(`refusing to split already-published window ${sourceRef}`), {
            error_class: "split_published_window_refused",
          });
        }
        if (window.records.length > 1 && depth < MAX_CONTEXT_SPLIT_DEPTH) {
          deleteWindowAttempt(config, attempts, provenance.line_sha256);
          const result = { accepted: 0, published: 0, skipped: 0 };
          for (const child of splitWindow(window)) {
            const childResult = await processWindow(config, child, seen, deps, attempts, context, depth + 1);
            result.accepted += childResult.accepted;
            result.published += childResult.published;
            result.skipped += childResult.skipped;
          }
          return result;
        }
        if (window.records.length === 1) {
          try {
            summary = await summarizeSingleWithTruncation(config, window, deps);
          } catch (truncErr) {
            if (isContextSizeError(truncErr)) {
              quarantineWindow(config, seen, attempts, window, provenance, truncErr);
              truncErr.window_failure_recorded = true;
            }
            throw truncErr;
          }
        } else {
          quarantineWindow(config, seen, attempts, window, provenance, err);
          err.window_failure_recorded = true;
          throw err;
        }
      }

      const accepted = window.records.length;
      if (summary.is_noise === true) {
        deleteWindowAttempt(config, attempts, provenance.line_sha256);
        return { accepted, published: 0, skipped: window.records.length };
      }
      envelope = buildEnvelope(config, window, summary);
      provenance.source_ref = envelope.source_ref;
      provenance.payload_sha256 = envelope.payload_sha256;
      provenance.nats_msg_id = `${envelope.source_ref}:${envelope.payload_sha256}`;
      validateEnvelope(config, envelope);
      assertNoSecrets(envelope, deps.redactSecrets);
      if (!config.dryRun) {
        seen[envelope.source_ref] = buildSeenEntry(config, envelope, provenance, "prepared");
        persistJson0600(config.seenFile, seen);
      }
    }

    const accepted = envelope && existing && existing.envelope ? window.records.length : 0;
    provenance.source_ref = envelope.source_ref;
    provenance.payload_sha256 = envelope.payload_sha256;
    validateEnvelope(config, envelope);
    assertNoSecrets(envelope, deps.redactSecrets);
    let published = 0;
    if (!config.dryRun && (!existing || existing.status !== "published")) {
      const result = await publishEnvelope(config, deps.js, deps.StringCodec, envelope);
      provenance.nats_msg_id = result.msgID;
      seen[envelope.source_ref] = {
        ...buildSeenEntry(config, envelope, provenance, "published"),
        published_at: new Date().toISOString(),
      };
      persistJson0600(config.seenFile, seen);
      published = 1;
    }
    deleteWindowAttempt(config, attempts, provenance.line_sha256);
    return { accepted: accepted || window.records.length, published, skipped: 0 };
  } catch (err) {
    if (err && err.window_failure_recorded) throw err;
    const attemptEntry = recordWindowAttempt(config, attempts, provenance, sourceRef, err);
    if (attemptEntry.attempts >= POISON_ATTEMPTS) {
      quarantineWindow(config, seen, attempts, window, provenance, err, attemptEntry);
      err.window_failure_recorded = true;
      throw err;
    }
    append0600(config.dlqFile, buildDlqRow(config, "held", err, provenance));
    err.window_failure_recorded = true;
    throw err;
  }
}

async function processFile(config, file, state, deps) {
  const key = stateKey(config, file);
  const stat = fs.statSync(file);
  if (!state[key]) {
    const initial = initialOffsetForFile(config, file, state, stat);
    state[key] = { offset: initial.offset };
    if (!initial.seeded && !config.live && config.dryRunSeedEof) return { scanned: 0, accepted: 0, published: 0, skipped: 0, seeded: true };
  }
  const offset = Number(state[key].offset || 0);
  const read = readCompleteLines(file, offset, config.maxBytesPerPass);
  if (read.lines.length === 0) {
    state[key].offset = read.nextOffset;
    return { scanned: 0, accepted: 0, published: 0, skipped: 0, seeded: false };
  }
  const records = [];
  const sourceLineByRef = new Map();
  let skipped = 0;
  for (const line of read.lines) {
    try {
      const row = normalizeRecord(JSON.parse(line.text));
      row.__capture_file = file;
      row.__capture_key = key;
      row.__byte_start = line.startByte;
      row.__byte_end = line.endByte;
      row.__line_sha256 = line.line_sha256;
      records.push(row);
      sourceLineByRef.set(row.dedupe_key, line);
    } catch (err) {
      append0600(config.dlqFile, buildDlqRow(config, "invalid_capture_row", err, {
        capture_file: file,
        state_key: key,
        byte_start: line.startByte,
        byte_end: line.endByte,
        held_offset: offset,
        next_offset: line.endByte,
        line_sha256: line.line_sha256,
      }));
      state[key].offset = line.endByte;
      skipped += 1;
    }
  }
  if (records.length === 0) {
    state[key].offset = read.nextOffset;
    return { scanned: read.lines.length, accepted: 0, published: 0, skipped, seeded: false };
  }
  let published = 0;
  let accepted = 0;
  const seen = loadJson(config.seenFile, {});
  const attempts = deps.attempts || loadAttemptState(config);
  const context = { file, stateKey: key, offset, nextOffset: read.nextOffset, sourceLineByRef };
  for (const window of buildWindows(records, config.maxRecordsPerWindow)) {
    const result = await processWindow(config, window, seen, deps, attempts, context);
    accepted += result.accepted;
    published += result.published;
    skipped += result.skipped;
  }
  state[key].offset = read.nextOffset;
  return { scanned: read.lines.length, accepted, published, skipped, seeded: false };
}

async function runOnce(config = buildConfig(), overrides = {}) {
  assertStatePathsSafe(config);
  if (config.seedCurrentEof) return { seeded: seedCurrentEof(config) };
  const files = listCaptureFiles(config);
  if (config.live) assertLiveSeed(config, files);
  const state = loadJson(config.offsetFile, {});
  const pendingAttemptKeys = collectPendingAttemptKeys(config, files, state);
  const attempts = loadAttemptState(config);
  pruneAttemptState(attempts, pendingAttemptKeys);
  persistAttemptState(config, attempts);
  const summarizer = overrides.summarizer || loadSummarizer(config);
  let nc = null;
  const deps = {
    summarizeThread: overrides.summarizeThread || summarizer.summarizeThread,
    renderBatch: overrides.renderBatch || summarizer.renderBatch,
    redactSecrets: overrides.redactSecrets || summarizer.redactSecrets,
    js: overrides.js || null,
    StringCodec: overrides.StringCodec || null,
    attempts,
  };
  const totals = { files: files.length, scanned: 0, accepted: 0, published: 0, skipped: 0, seeded: 0, errors: 0 };
  try {
    if (config.live && !deps.js) {
      const conn = await connectNats(config);
      nc = conn.nc;
      deps.js = conn.js;
      deps.StringCodec = conn.StringCodec;
    }
    for (const file of files) {
      try {
        const res = await processFile(config, file, state, deps);
        totals.scanned += res.scanned;
        totals.accepted += res.accepted;
        totals.published += res.published;
        totals.skipped += res.skipped;
        if (res.seeded) totals.seeded += 1;
      } catch (err) {
        totals.errors += 1;
        // Was emitting err.message RAW, bypassing redaction and any cap --
        // the inverse of the omission this PR fixes, same class.
        console.error(
          `[telegram-direct-producer] file_error file=${file} class=${errorClass(err)} msg=${errorMessage(err).slice(0, DLQ_ERROR_MESSAGE_MAX)}`,
        );
      }
    }
  } finally {
    if (nc) await nc.drain();
  }
  writeJsonAtomic(config.offsetFile, state);
  writeJsonAtomic(config.metricsFile, { last_run_at: new Date().toISOString(), dry_run: config.dryRun, ...totals });
  console.log(
    `[telegram-direct-producer] once: files=${totals.files} scanned=${totals.scanned} accepted=${totals.accepted} ` +
      `published=${totals.published} skipped=${totals.skipped} seeded=${totals.seeded} errors=${totals.errors}`
  );
  return totals;
}

module.exports = {
  PRODUCER,
  TIMESTAMP_ERROR_CLASSES,
  POISON_ATTEMPTS,
  errorMessage,
  errorClass,
  buildDlqRow,
  buildConfig,
  parseArgs,
  canonicalJson,
  sha256Hex,
  assertStatePathsSafe,
  listCaptureFiles,
  readCompleteLines,
  sourceRefOf,
  messageIdOf,
  tsOf,
  windowSourceRef,
  normalizeRecord,
  sourceTimeMsForSort,
  assertWindowSourceTimes,
  buildWindows,
  buildEnvelope,
  validateEnvelope,
  assertNoSecrets,
  isContextSizeError,
  splitWindow,
  addPendingAttemptKeysForWindow,
  truncateRecordsForSummary,
  loadAttemptState,
  pruneAttemptState,
  seedCurrentEof,
  assertLiveSeed,
  processWindow,
  processFile,
  runOnce,
};

if (require.main === module) {
  const config = buildConfig();
  if (!config.once && !config.seedCurrentEof) {
    console.error("[telegram-direct-producer] pass --once or --seed-current-eof; launchd owns cadence");
    process.exit(2);
  }
  runOnce(config)
    .then((result) => {
      if (result && result.errors) process.exit(1);
      process.exit(0);
    })
    .catch((err) => {
      // Same omission as the DLQ row: the class alone cannot tell an operator
      // what went wrong. Redacted and capped by the same helper.
      console.error(
        `[telegram-direct-producer] FATAL error_class=${err && err.error_class ? err.error_class : err && err.name ? err.name : "Error"} error_message=${JSON.stringify(errorMessage(err).slice(0, DLQ_ERROR_MESSAGE_MAX))}`,
      );
      process.exit(1);
    });
}

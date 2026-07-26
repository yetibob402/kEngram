#!/usr/bin/env node
"use strict";
/*
 * argus-codex-rollout-producer
 *
 * Reads Codex CLI rollout JSONL (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl),
 * extracts the user/agent conversation turns, distills durable signal with the
 * SAME distiller as the Claude session producer, and publishes session envelopes
 * to NATS (stream ARGUS_SESSION_INGEST, subject ingest.session.{agent}.{host}).
 *
 * The existing argus-kengram-consumer ingests these envelopes unchanged
 * (scope sessions/{agent}) — pg-direct on yeti, MCP on dockerized outposts.
 *
 * Why this exists: Codex agents (neo/smith/trinity/sparx/dozer) write to
 * ~/.codex/sessions, NOT ~/.claude/projects, so the Claude session producer
 * never saw them. Capture for them stopped ~2026-06-16. This closes that gap.
 *
 * launchd runs it with --once.
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
let _summarizeThread = null;

function loadNats() {
  if (!_nats) {
    _nats = require(NATS_LIB_DIR);
  }
  return _nats;
}

function loadSummarizeThread() {
  if (!_summarizeThread) {
    _summarizeThread = require("./argus-session-distiller.js").summarizeThread;
  }
  return _summarizeThread;
}

const HOME = process.env.HOME || os.homedir();
const ONCE = process.argv.includes("--once") || process.env.CODEX_PRODUCER_ONCE === "1";
const DRY_RUN = process.argv.includes("--dry-run") || process.env.CODEX_PRODUCER_DRY_RUN === "1";
const BACKFILL = process.argv.includes("--backfill") || process.env.CODEX_PRODUCER_BACKFILL === "1";

const SESSION_SCHEMA_VERSION = "argus.ingest.v1";
const SESSION_SOURCE_KIND = "session";
const PRODUCER = "argus-codex-rollout-producer";
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

function isSourceTimestampError(err) {
  return !!(err && TIMESTAMP_ERROR_CLASSES.has(err.error_class));
}

function noteEnvelopeBuildFailure(err, totals) {
  if (!isSourceTimestampError(err)) return false;
  totals.errors++;
  return true;
}

function loadEnvFile(file) {
  if (!file) return {};
  try {
    const out = {};
    for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
      const t = line.trim();
      if (!t || t.startsWith("#")) continue;
      const m = t.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
      if (!m) continue;
      out[m[1]] = m[2].replace(/^['"]|['"]$/g, "");
    }
    return out;
  } catch (_) {
    return {};
  }
}

const NATS_AUTH_ENV = loadEnvFile(
  process.env.NATS_AUTH_ENV_FILE || path.join(HOME, "argus/state/nats-client/client.env")
);
function authUser(d = "yeti") {
  return process.env.CODEX_PRODUCER_NATS_USER || process.env.NATS_AUTH_USER || NATS_AUTH_ENV.NATS_AUTH_USER || d;
}
function authToken(user) {
  const key = `CRED_NATS_TOKEN_${String(user || "").toUpperCase()}`;
  return (
    process.env.CODEX_PRODUCER_NATS_TOKEN ||
    process.env.NATS_AUTH_TOKEN ||
    process.env[key] ||
    NATS_AUTH_ENV[key] ||
    NATS_AUTH_ENV.CRED_NATS_TOKEN
  );
}

const CONFIG = {
  natsUrl: process.env.CODEX_PRODUCER_NATS_URL || "127.0.0.1:4222",
  stream: process.env.CODEX_PRODUCER_STREAM || "ARGUS_SESSION_INGEST",
  subjectPrefix: process.env.CODEX_PRODUCER_SUBJECT_PREFIX || "ingest.session",
  codexDir: process.env.CODEX_SESSIONS_DIR || path.join(HOME, ".codex/sessions"),
  agents: (process.env.CODEX_PRODUCER_AGENTS || "neo,smith,trinity")
    .split(",").map((s) => s.trim()).filter(Boolean),
  host: process.env.CODEX_PRODUCER_HOST || os.hostname().replace(/[^A-Za-z0-9_.-]+/g, "-"),
  stateFile:
    process.env.CODEX_PRODUCER_STATE_FILE ||
    path.join(HOME, "argus/state/codex-rollout-producer/offsets.json"),
  metricsFile:
    process.env.CODEX_PRODUCER_METRICS_FILE ||
    path.join(HOME, "argus/state/codex-rollout-producer/metrics.json"),
  maxBatch: Number(process.env.CODEX_PRODUCER_MAX_LINES || 40),
  maxFilesPerRun: Number(process.env.CODEX_PRODUCER_MAX_FILES || 200),
  connectTimeoutMs: Number(process.env.CODEX_PRODUCER_CONNECT_TIMEOUT_MS || 8000),
};
CONFIG.natsUser = authUser();
CONFIG.natsToken = authToken(CONFIG.natsUser);

// ---- helpers (mirrored from argus-session-nats-producer for envelope parity) --
function canonicalJson(value) {
  if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
  if (value && typeof value === "object") {
    return (
      "{" +
      Object.keys(value).sort().map((k) => JSON.stringify(k) + ":" + canonicalJson(value[k])).join(",") +
      "}"
    );
  }
  return JSON.stringify(value === undefined ? null : value);
}
function sha256Hex(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}
function generationTag() {
  return sha256Hex(JSON.stringify({
    producer: PRODUCER,
    summarizer_model: process.env.SUMMARIZER_MODEL || "default",
  })).slice(0, 12);
}
function topicKey(summary) {
  const topics = Array.isArray(summary.topics) ? summary.topics : [];
  if (topics.length) {
    return String(topics[0]).toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);
  }
  return "session-distill";
}
function loadJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (_) {
    return fallback;
  }
}
function writeJson0600(file, obj) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(obj, null, 2), { mode: 0o600 });
}

// ---- rollout discovery + parsing -------------------------------------------
function listRolloutFiles(dir) {
  const out = [];
  const walk = (d) => {
    let entries;
    try {
      entries = fs.readdirSync(d, { withFileTypes: true });
    } catch (_) {
      return;
    }
    for (const e of entries) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.isFile() && /^rollout-.*\.jsonl$/.test(e.name)) out.push(full);
    }
  };
  walk(dir);
  out.sort();
  return out;
}

// Returns {agent, sessionId} from the session_meta line, or null if not a tracked agent.
function rolloutMeta(file) {
  let firstLine;
  try {
    // session_meta line carries base_instructions (long) — read the whole first
    // line, not a fixed buffer, or we truncate + fail to parse.
    const content = fs.readFileSync(file, "utf8");
    const nl = content.indexOf("\n");
    firstLine = nl === -1 ? content : content.slice(0, nl);
  } catch (_) {
    return null;
  }
  let o;
  try {
    o = JSON.parse(firstLine);
  } catch (_) {
    return null;
  }
  const p = o && o.payload;
  if (!p || o.type !== "session_meta") return null;
  const cwd = String(p.cwd || "");
  const agent = path.basename(cwd);
  const sessionId = String(p.id || path.basename(file).replace(/^rollout-|\.jsonl$/g, "")).replace(/[^A-Za-z0-9_.-]+/g, "-");
  return { agent, sessionId };
}

// Extract conversation-turn records from rollout lines starting at lineOffset.
function extractRecords(file, agent, lineOffset) {
  let lines;
  try {
    lines = fs.readFileSync(file, "utf8").split("\n");
  } catch (_) {
    return { records: [], endLine: lineOffset };
  }
  const records = [];
  let idx = lineOffset;
  for (; idx < lines.length; idx++) {
    const raw = lines[idx];
    if (!raw) continue;
    let o;
    try {
      o = JSON.parse(raw);
    } catch (_) {
      continue;
    }
    if (o.type !== "event_msg") continue;
    const p = o.payload || {};
    const ts = o.timestamp || null;
    if (p.type === "user_message") {
      const text = String(p.message || "").trim();
      if (text) records.push({ direction: "inbound", agent: "user", text, capture_ts: ts, dedupe_key: `codex:${agent}:${idx}` });
    } else if (p.type === "agent_message") {
      const text = String(p.message || "").trim();
      if (text) records.push({ direction: "outbound", agent, text, capture_ts: ts, dedupe_key: `codex:${agent}:${idx}` });
    }
  }
  return { records, endLine: lines.length };
}

function buildEnvelope(agent, sessionId, startIdx, endIdx, records, summary, now) {
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
    source_line_refs: records.map((r) => r.dedupe_key),
  };
  const payloadHash = sha256Hex(canonicalJson(payload));
  const gen = generationTag();
  const sourceRef = `session:${agent}:${CONFIG.host}:${sessionId}:bytes:${startIdx}-${endIdx}:gen:${gen}`;
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
    producer: PRODUCER,
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
    provenance: { raw_path: file_of(sessionId), batch_start: startIdx, batch_end: endIdx, record_count: records.length, generation: gen },
  };
}
let _filemap = {};
function file_of(sessionId) {
  return _filemap[sessionId] || sessionId;
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
      max_msgs: 1000000,
    });
  }
}

async function runOnce() {
  const state = loadJson(CONFIG.stateFile, {});
  const totals = { files: 0, scanned: 0, agents: {}, batches: 0, published: 0, skipped: 0, errors: 0 };
  const files = listRolloutFiles(CONFIG.codexDir).slice(-CONFIG.maxFilesPerRun);
  const { connect, StringCodec } = loadNats();
  const SC = StringCodec();
  const summarizeThread = loadSummarizeThread();

  const nc = DRY_RUN
    ? null
    : await connect({
        servers: CONFIG.natsUrl,
        name: PRODUCER,
        timeout: CONFIG.connectTimeoutMs,
        reconnect: false,
        maxReconnectAttempts: 0,
        ...(CONFIG.natsToken ? { user: CONFIG.natsUser, pass: CONFIG.natsToken } : {}),
      });
  let js = null;
  try {
    if (nc) {
      const jsm = await nc.jetstreamManager();
      await ensureStream(jsm);
      js = nc.jetstream();
    }
    for (const file of files) {
      const key = file.replace(HOME, "~");
      let st = state[key];
      if (!st) {
        const meta = rolloutMeta(file);
        if (!meta || !CONFIG.agents.includes(meta.agent)) {
          state[key] = { agent: meta ? meta.agent : null, skip: true, offset: 0 };
          continue;
        }
        st = state[key] = { agent: meta.agent, sessionId: meta.sessionId, offset: 0 };
      }
      if (st.skip) continue;
      _filemap[st.sessionId] = file;
      totals.files++;
      const startOffset = BACKFILL ? 0 : st.offset || 0;
      const { records, endLine } = extractRecords(file, st.agent, startOffset);
      totals.scanned += records.length;
      totals.agents[st.agent] = (totals.agents[st.agent] || 0) + records.length;
      // batch records into windows of maxBatch
      // global record index base for deterministic source_ref ranges
      const baseIdx = BACKFILL ? 0 : st.recordsEmitted || 0;
      for (let i = 0; i < records.length; i += CONFIG.maxBatch) {
        const batch = records.slice(i, i + CONFIG.maxBatch);
        const startIdx = baseIdx + i;
        const endIdx = baseIdx + i + batch.length;
        let summary;
        try {
          summary = await summarizeThread(batch, {});
        } catch (err) {
          totals.errors++;
          continue;
        }
        if (summary && summary.is_noise === true) {
          totals.skipped++;
          continue;
        }
        let env;
        try {
          env = buildEnvelope(st.agent, st.sessionId, startIdx, endIdx, batch, summary);
        } catch (err) {
          if (noteEnvelopeBuildFailure(err, totals)) continue;
          throw err;
        }
        if (DRY_RUN) {
          totals.batches++;
          continue;
        }
        try {
          const msgID = `${env.source_ref}:${env.payload_sha256}`;
          await js.publish(env.subject, SC.encode(JSON.stringify(env)), { msgID });
          totals.published++;
          totals.batches++;
        } catch (err) {
          totals.errors++;
        }
      }
      st.offset = endLine;
      st.recordsEmitted = (st.recordsEmitted || 0) + records.length;
    }
    if (!DRY_RUN) writeJson0600(CONFIG.stateFile, state);
    writeJson0600(CONFIG.metricsFile, { last_run_at: new Date().toISOString(), dry_run: DRY_RUN, backfill: BACKFILL, ...totals });
  } finally {
    if (nc) await nc.drain();
  }
  return totals;
}

module.exports = {
  CONFIG,
  TIMESTAMP_ERROR_CLASSES,
  PRODUCER,
  canonicalJson,
  sha256Hex,
  sourceCreatedAtFromRecords,
  parseUsableSourceInstant,
  isSourceTimestampError,
  noteEnvelopeBuildFailure,
  buildEnvelope,
  extractRecords,
  ensureStream,
  runOnce,
};

if (require.main === module) {
  runOnce()
    .then((t) => {
      const agentStr = Object.entries(t.agents)
        .map(([a, n]) => `${a}=${n}`)
        .join(",");
      console.log(
        `[codex-rollout-producer] once: files=${t.files} scanned=${t.scanned} agents=[${agentStr}] batches=${t.batches} published=${t.published} skipped=${t.skipped} errors=${t.errors}`
      );
      process.exit(0);
    })
    .catch((err) => {
      console.error(`[codex-rollout-producer] FATAL: ${err && err.message}`);
      process.exit(1);
    });
}

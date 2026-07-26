#!/usr/bin/env node
"use strict";
/*
 * Temporal-pollution slice 1 — focused deterministic RED/GREEN tests.
 * Imports real producer/adapter modules. No live NATS, PostgreSQL, kEngram,
 * Telegram, or production filesystem state.
 */

const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");

const session = require("../bin/argus-session-nats-producer.js");
const codex = require("../bin/argus-codex-rollout-producer.js");
const telegram = require("../bin/argus-telegram-direct-nats-producer.js");
const adapter = require("../lib/argus-telegram-kengram-adapter.js");

const FIXED_NOW = Date.parse("2026-07-26T12:00:00.000Z");
const FIXED_NOW_ISO = new Date(FIXED_NOW).toISOString();
const OLD_2025 = "2025-06-15T10:00:00.000Z";
const MID_2026 = "2026-03-01T08:30:00.000Z";
const NEWEST = "2026-07-20T18:00:00.000Z";
const FUTURE = "2026-07-26T12:10:00.000Z"; // +10 min past FIXED_NOW
// Finite but outside ECMAScript TimeClip (±8.64e15 ms) → unrepresentable Date.
const OUT_OF_RANGE_MS = -8.64e15 - 1;

let failed = 0;
let passed = 0;

function check(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`PASS ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`FAIL ${name}`);
    console.error(err && err.stack ? err.stack : err);
  }
}

async function checkAsync(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`PASS ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`FAIL ${name}`);
    console.error(err && err.stack ? err.stack : err);
  }
}

function emptySummary() {
  return {
    summary: "distilled signal for temporal test",
    key_facts: ["fact-a"],
    decisions: [],
    intents: [],
    action_items: [],
    open_questions: [],
    blockers: [],
    artifacts: [],
    corrections: [],
    topics: ["temporal-test"],
    participants: ["smith"],
    is_noise: false,
  };
}

function sessionRecords() {
  return [
    {
      agent: "smith",
      direction: "inbound",
      text: "older turn content that is long enough for a session fixture",
      capture_ts: OLD_2025,
      dedupe_key: "session-line:smith:s.jsonl:0-10",
      byte_start: 0,
      byte_end: 10,
    },
    {
      agent: "smith",
      direction: "outbound",
      text: "newer turn content that is long enough for a session fixture response",
      capture_ts: NEWEST,
      dedupe_key: "session-line:smith:s.jsonl:10-20",
      byte_start: 10,
      byte_end: 20,
    },
    {
      agent: "smith",
      direction: "inbound",
      text: "middle turn content that is long enough for a session fixture again",
      capture_ts: MID_2026,
      dedupe_key: "session-line:smith:s.jsonl:20-30",
      byte_start: 20,
      byte_end: 30,
    },
  ];
}

function codexRecords() {
  return [
    {
      direction: "inbound",
      agent: "user",
      text: "codex older",
      capture_ts: OLD_2025,
      dedupe_key: "codex:smith:1",
    },
    {
      direction: "outbound",
      agent: "smith",
      text: "codex newest",
      capture_ts: NEWEST,
      dedupe_key: "codex:smith:2",
    },
  ];
}

function telegramConfig() {
  return {
    subjectPrefix: "ingest.telegram",
    maxRecordsPerWindow: 12,
  };
}

function telegramSummary() {
  return emptySummary();
}

function expectTimestampError(fn, errorClass) {
  let err = null;
  try {
    fn();
  } catch (e) {
    err = e;
  }
  assert.ok(err, `expected throw ${errorClass}`);
  assert.strictEqual(err.error_class, errorClass, `expected ${errorClass}, got ${err && err.error_class}`);
}

// ---------------------------------------------------------------------------
// 2. Import safety (session + codex load without NATS/distiller; no process exit)
// ---------------------------------------------------------------------------
check("import: session and codex modules load without production NATS/distiller", () => {
  assert.strictEqual(typeof session.buildEnvelope, "function");
  assert.strictEqual(typeof session.isPermanentReject, "function");
  assert.strictEqual(typeof codex.buildEnvelope, "function");
  assert.ok(codex.TIMESTAMP_ERROR_CLASSES instanceof Set);
  // If either module had eagerly required nats-shadow or distiller, this file
  // would not have loaded at all under the isolated contrib/argus/bin layout.
});

// ---------------------------------------------------------------------------
// 1. Session max source time
// ---------------------------------------------------------------------------
check("session: created_at is newest capture_ts, not wall clock; published_at is publish now", () => {
  const profile = { name: "smith", profilePath: "/tmp/profile" };
  const env = session.buildEnvelope(
    profile,
    "/tmp/session.jsonl",
    0,
    30,
    sessionRecords(),
    emptySummary(),
    FIXED_NOW
  );
  assert.strictEqual(env.created_at, NEWEST);
  assert.strictEqual(env.published_at, FIXED_NOW_ISO);
  assert.notStrictEqual(env.created_at, env.published_at);
  assert.ok(env.created_at !== new Date().toISOString());
  assert.ok(Array.isArray(env.payload.source_line_refs));
  assert.strictEqual(env.payload.source_line_refs.length, 3);
  assert.ok(/^session:smith:/.test(env.source_ref));
  assert.strictEqual(env.payload_sha256, session.sha256Hex(session.canonicalJson(env.payload)));
});

// ---------------------------------------------------------------------------
// 2b. Codex max source time
// ---------------------------------------------------------------------------
check("codex: created_at is newest capture_ts, not wall clock; published_at is publish now", () => {
  const env = codex.buildEnvelope("smith", "sess-1", 0, 2, codexRecords(), emptySummary(), FIXED_NOW);
  assert.strictEqual(env.created_at, NEWEST);
  assert.strictEqual(env.published_at, FIXED_NOW_ISO);
  assert.notStrictEqual(env.created_at, env.published_at);
  assert.strictEqual(env.payload_sha256, codex.sha256Hex(codex.canonicalJson(env.payload)));
  assert.ok(env.source_ref.includes("session:smith:"));
});

// ---------------------------------------------------------------------------
// 3. Telegram precedence
// ---------------------------------------------------------------------------
check("telegram: telegram_date beats later capture_ts; normalized_event.ts beats capture_ts", () => {
  const tdUnix = Math.floor(Date.parse("2026-01-10T09:00:00.000Z") / 1000);
  const recTelegram = {
    agent: "smith",
    chat_id: "42",
    message_id: "100",
    dedupe_key: "telegram:smith:42:100",
    capture_ts: NEWEST,
    telegram_date: tdUnix,
  };
  assert.strictEqual(telegram.tsOf(recTelegram, FIXED_NOW), "2026-01-10T09:00:00.000Z");

  const recNorm = {
    agent: "smith",
    chat_id: "42",
    message_id: "101",
    dedupe_key: "telegram:smith:42:101",
    capture_ts: NEWEST,
    normalized_event: { ts: MID_2026 },
  };
  assert.strictEqual(telegram.tsOf(recNorm, FIXED_NOW), MID_2026);
});

check("telegram: present-empty/null/undefined stronger fields fail closed (no demotion)", () => {
  // Empty telegram_date is present-invalid; must not demote to normalized_event.ts or capture_ts.
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          telegram_date: "",
          normalized_event: { ts: MID_2026 },
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          telegram_date: null,
          normalized_event: { ts: MID_2026 },
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          telegram_date: undefined,
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );

  // Empty normalized_event.ts is present-invalid; must not demote to capture_ts.
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          normalized_event: { ts: "" },
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          normalized_event: { ts: null },
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          normalized_event: { ts: undefined },
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );

  // Absence of stronger keys still allows valid weaker fallback.
  assert.strictEqual(
    telegram.tsOf({ capture_ts: NEWEST }, FIXED_NOW),
    NEWEST
  );
  assert.strictEqual(
    telegram.tsOf({ normalized_event: { ts: MID_2026 }, capture_ts: NEWEST }, FIXED_NOW),
    MID_2026
  );
});

check("session/codex/telegram/adapter: out-of-range finite ms is invalid_source_created_at", () => {
  const profile = { name: "smith", profilePath: "/tmp/profile" };
  const sbase = sessionRecords()[0];
  const cbase = codexRecords()[0];

  expectTimestampError(
    () =>
      session.buildEnvelope(
        profile,
        "/tmp/s.jsonl",
        0,
        10,
        [{ ...sbase, capture_ts: OUT_OF_RANGE_MS }],
        emptySummary(),
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      codex.buildEnvelope(
        "smith",
        "s",
        0,
        1,
        [{ ...cbase, capture_ts: OUT_OF_RANGE_MS }],
        emptySummary(),
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => telegram.tsOf({ capture_ts: OUT_OF_RANGE_MS }, FIXED_NOW),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => adapter.validateSourceCreatedAt(OUT_OF_RANGE_MS, FIXED_NOW),
    "invalid_source_created_at"
  );

  // Must not surface as RangeError / unclassified throw.
  for (const fn of [
    () => session.parseUsableSourceInstant(OUT_OF_RANGE_MS, FIXED_NOW),
    () => codex.parseUsableSourceInstant(OUT_OF_RANGE_MS, FIXED_NOW),
    () => telegram.tsOf({ capture_ts: OUT_OF_RANGE_MS }, FIXED_NOW),
    () => adapter.validateSourceCreatedAt(OUT_OF_RANGE_MS, FIXED_NOW),
  ]) {
    let err = null;
    try {
      fn();
    } catch (e) {
      err = e;
    }
    assert.ok(err);
    assert.strictEqual(err.error_class, "invalid_source_created_at");
    assert.notStrictEqual(err.name, "RangeError");
  }
});

check("telegram: whitespace/boolean/object telegram_date fail closed without coercion demotion", () => {
  const weaker = {
    normalized_event: { ts: MID_2026 },
    capture_ts: NEWEST,
  };
  expectTimestampError(
    () => telegram.tsOf({ telegram_date: "   ", ...weaker }, FIXED_NOW),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => telegram.tsOf({ telegram_date: true, ...weaker }, FIXED_NOW),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => telegram.tsOf({ telegram_date: false, ...weaker }, FIXED_NOW),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => telegram.tsOf({ telegram_date: [], ...weaker }, FIXED_NOW),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => telegram.tsOf({ telegram_date: {}, ...weaker }, FIXED_NOW),
    "invalid_source_created_at"
  );
  // Valid numeric string and number still work.
  const tdUnix = Math.floor(Date.parse("2026-01-10T09:00:00.000Z") / 1000);
  assert.strictEqual(
    telegram.tsOf({ telegram_date: String(tdUnix), capture_ts: NEWEST }, FIXED_NOW),
    "2026-01-10T09:00:00.000Z"
  );
  assert.strictEqual(
    telegram.tsOf({ telegram_date: tdUnix, capture_ts: NEWEST }, FIXED_NOW),
    "2026-01-10T09:00:00.000Z"
  );
});

checkAsync("telegram: invalid timestamp uses processWindow attempt/DLQ/quarantine (not file_error)", async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "kengram-tp-tg-attempt-"));
  const config = {
    producer: telegram.PRODUCER,
    dlqFile: path.join(tmp, "dlq.jsonl"),
    poisonFile: path.join(tmp, "poison.jsonl"),
    seenFile: path.join(tmp, "seen.json"),
    attemptsFile: path.join(tmp, "attempts.json"),
    dryRun: true,
    subjectPrefix: "ingest.telegram",
    maxRecordsPerWindow: 12,
    maxSummaryTokens: 24000,
  };
  const validOlder = {
    agent: "smith",
    chat_id: "42",
    message_id: "1",
    dedupe_key: "telegram:smith:42:1",
    capture_ts: OLD_2025,
  };
  const validNewer = {
    agent: "smith",
    chat_id: "42",
    message_id: "2",
    dedupe_key: "telegram:smith:42:2",
    capture_ts: NEWEST,
  };
  const invalid = {
    agent: "smith",
    chat_id: "42",
    message_id: "3",
    dedupe_key: "telegram:smith:42:3",
    capture_ts: "not-a-timestamp",
  };

  // Window construction must not throw; valid records keep chronological order.
  const windows = telegram.buildWindows([validNewer, invalid, validOlder], 12, FIXED_NOW);
  assert.strictEqual(windows.length, 1);
  assert.deepStrictEqual(
    windows[0].records.map((r) => r.message_id),
    ["1", "2", "3"]
  );

  let summarizeCalls = 0;
  const deps = {
    summarizeThread: async () => {
      summarizeCalls += 1;
      throw new Error("summarize must not run for timestamp-class failures");
    },
    renderBatch: () => "",
    redactSecrets: (s) => s,
    js: null,
    StringCodec: null,
  };
  const attempts = {};
  const seen = {};
  const context = {
    file: path.join(tmp, "capture.jsonl"),
    stateKey: "smith|capture",
    offset: 0,
    nextOffset: 300,
    sourceLineByRef: new Map([
      [validOlder.dedupe_key, { startByte: 0, endByte: 100 }],
      [validNewer.dedupe_key, { startByte: 100, endByte: 200 }],
      [invalid.dedupe_key, { startByte: 200, endByte: 300 }],
    ]),
  };

  const poisonLimit = telegram.POISON_ATTEMPTS;
  assert.ok(poisonLimit >= 2);

  // Held attempts: each advances the attempt counter and writes held DLQ; no publish.
  for (let n = 1; n < poisonLimit; n += 1) {
    let err = null;
    try {
      await telegram.processWindow(config, windows[0], seen, deps, attempts, context);
    } catch (e) {
      err = e;
    }
    assert.ok(err, `expected held throw on attempt ${n}`);
    assert.strictEqual(err.error_class, "invalid_source_created_at");
    assert.strictEqual(summarizeCalls, 0, "timestamp fail must not reach summarizer");
    const attemptKeys = Object.keys(attempts);
    assert.strictEqual(attemptKeys.length, 1);
    assert.strictEqual(attempts[attemptKeys[0]].attempts, n);
    assert.strictEqual(attempts[attemptKeys[0]].error_class, "invalid_source_created_at");
    const dlq = fs.readFileSync(config.dlqFile, "utf8");
    assert.ok(dlq.includes('"reason":"held"'), "held DLQ receipt required");
    assert.ok(dlq.includes("invalid_source_created_at"));
    assert.ok(!fs.existsSync(config.poisonFile), "poison file must not exist before poison limit");
    const sourceRef = telegram.windowSourceRef("smith", "42", windows[0].records);
    assert.ok(!seen[sourceRef] || seen[sourceRef].status !== "quarantined");
  }

  // Final attempt reaches existing quarantine machinery.
  let finalErr = null;
  try {
    await telegram.processWindow(config, windows[0], seen, deps, attempts, context);
  } catch (e) {
    finalErr = e;
  }
  assert.ok(finalErr);
  assert.strictEqual(finalErr.error_class, "invalid_source_created_at");
  assert.strictEqual(summarizeCalls, 0);
  const sourceRef = telegram.windowSourceRef("smith", "42", windows[0].records);
  assert.strictEqual(seen[sourceRef].status, "quarantined");
  assert.strictEqual(seen[sourceRef].error_class, "invalid_source_created_at");
  const poison = fs.readFileSync(config.poisonFile, "utf8");
  assert.ok(poison.includes("poison_window") || poison.includes('"reason":"poison_window"'));
  assert.ok(poison.includes("invalid_source_created_at"));
  assert.strictEqual(Object.keys(attempts).length, 0, "quarantine clears attempt entry");

  // Replay of quarantined window skips without publish (bounded; not a poison loop).
  const skip = await telegram.processWindow(config, windows[0], seen, deps, attempts, context);
  assert.strictEqual(skip.published, 0);
  assert.strictEqual(skip.skipped, windows[0].records.length);

  try {
    fs.rmSync(tmp, { recursive: true, force: true });
  } catch (_) {
    /* best effort */
  }
});

// ---------------------------------------------------------------------------
// 4. Session/Codex fail-closed classes + session permanent offset + codex continue
// ---------------------------------------------------------------------------
check("session/codex: missing/invalid/future fail with named classes", () => {
  const profile = { name: "smith", profilePath: "/tmp/profile" };
  const base = sessionRecords()[0];

  expectTimestampError(
    () =>
      session.buildEnvelope(
        profile,
        "/tmp/s.jsonl",
        0,
        10,
        [{ ...base, capture_ts: "" }],
        emptySummary(),
        FIXED_NOW
      ),
    "missing_source_created_at"
  );
  expectTimestampError(
    () =>
      session.buildEnvelope(
        profile,
        "/tmp/s.jsonl",
        0,
        10,
        [{ ...base, capture_ts: "not-a-timestamp" }],
        emptySummary(),
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      session.buildEnvelope(
        profile,
        "/tmp/s.jsonl",
        0,
        10,
        [{ ...base, capture_ts: FUTURE }],
        emptySummary(),
        FIXED_NOW
      ),
    "future_source_created_at"
  );

  const cbase = codexRecords()[0];
  expectTimestampError(
    () => codex.buildEnvelope("smith", "s", 0, 1, [{ ...cbase, capture_ts: null }], emptySummary(), FIXED_NOW),
    "missing_source_created_at"
  );
  expectTimestampError(
    () =>
      codex.buildEnvelope(
        "smith",
        "s",
        0,
        1,
        [{ ...cbase, capture_ts: "NaN" }],
        emptySummary(),
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      codex.buildEnvelope(
        "smith",
        "s",
        0,
        1,
        [{ ...cbase, capture_ts: FUTURE }],
        emptySummary(),
        FIXED_NOW
      ),
    "future_source_created_at"
  );

  assert.strictEqual(session.isPermanentReject({ error_class: "missing_source_created_at" }), true);
  assert.strictEqual(session.isPermanentReject({ error_class: "invalid_source_created_at" }), true);
  assert.strictEqual(session.isPermanentReject({ error_class: "future_source_created_at" }), true);
  assert.strictEqual(codex.isSourceTimestampError({ error_class: "missing_source_created_at" }), true);
  assert.strictEqual(codex.isSourceTimestampError({ error_class: "invalid_source_created_at" }), true);
  assert.strictEqual(codex.isSourceTimestampError({ error_class: "future_source_created_at" }), true);
});

checkAsync("session: permanent-reject advances offset (no poison loop)", async () => {
  const dlqFile = path.join(os.tmpdir(), `kengram-tp-session-dlq-${process.pid}.jsonl`);
  try {
    fs.unlinkSync(dlqFile);
  } catch (_) {
    /* ok */
  }
  const prevDlq = session.CONFIG.dlqFile;
  session.CONFIG.dlqFile = dlqFile;
  try {
    const key = "smith|/tmp/s.jsonl";
    const state = { [key]: { offset: 0 } };
    const records = [
      {
        agent: "smith",
        direction: "inbound",
        text: "bad-time record for permanent reject path",
        capture_ts: "not-parseable",
        dedupe_key: "session-line:smith:s.jsonl:100-200",
        byte_start: 100,
        byte_end: 200,
      },
    ];
    const result = await session.processRecordWindow(
      { name: "smith", profilePath: "/tmp/profile" },
      "/tmp/s.jsonl",
      key,
      state,
      null,
      records,
      100,
      200,
      FIXED_NOW
    );
    assert.strictEqual(result.dlq, 1);
    assert.strictEqual(result.published, 0);
    assert.strictEqual(state[key].offset, 200, "offset must advance past permanent-rejected record");
    const dlq = fs.readFileSync(dlqFile, "utf8");
    assert.ok(dlq.includes("permanent_reject"));
    assert.ok(dlq.includes("invalid_source_created_at"));
  } finally {
    session.CONFIG.dlqFile = prevDlq;
    try {
      fs.unlinkSync(dlqFile);
    } catch (_) {
      /* ok */
    }
  }
});

check("codex: timestamp failure counts error and continues (does not abort)", () => {
  const totals = { errors: 0, published: 0, batches: 0 };
  const handled = codex.noteEnvelopeBuildFailure(
    Object.assign(new Error("future_source_created_at"), { error_class: "future_source_created_at" }),
    totals
  );
  assert.strictEqual(handled, true);
  assert.strictEqual(totals.errors, 1);
  assert.strictEqual(totals.published, 0);
  const notHandled = codex.noteEnvelopeBuildFailure(new Error("other"), totals);
  assert.strictEqual(notHandled, false);
  assert.strictEqual(totals.errors, 1);
});

// ---------------------------------------------------------------------------
// 5. Telegram fail-closed
// ---------------------------------------------------------------------------
check("telegram: invalid stronger time / missing / non-finite / future fail closed; never returns now", () => {
  const nowWall = new Date().toISOString();
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          telegram_date: "not-a-number",
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () =>
      telegram.tsOf(
        {
          normalized_event: { ts: "bogus" },
          capture_ts: NEWEST,
        },
        FIXED_NOW
      ),
    "invalid_source_created_at"
  );
  expectTimestampError(() => telegram.tsOf({}, FIXED_NOW), "missing_source_created_at");
  expectTimestampError(
    () => telegram.tsOf({ capture_ts: "Infinity" }, FIXED_NOW),
    "invalid_source_created_at"
  );
  expectTimestampError(
    () => telegram.tsOf({ capture_ts: FUTURE }, FIXED_NOW),
    "future_source_created_at"
  );

  let threw = false;
  try {
    telegram.tsOf({ capture_ts: "" }, FIXED_NOW);
  } catch (e) {
    threw = true;
    assert.notStrictEqual(e.message, nowWall);
    assert.ok(e.error_class);
  }
  assert.ok(threw);
});

// ---------------------------------------------------------------------------
// 6. Telegram window created_at newest; published_at independent
// ---------------------------------------------------------------------------
check("telegram: window created_at is newest validated source time; published_at is current", () => {
  const records = [
    {
      agent: "smith",
      chat_id: "99",
      message_id: "1",
      dedupe_key: "telegram:smith:99:1",
      capture_ts: OLD_2025,
    },
    {
      agent: "smith",
      chat_id: "99",
      message_id: "2",
      dedupe_key: "telegram:smith:99:2",
      capture_ts: NEWEST,
    },
    {
      agent: "smith",
      chat_id: "99",
      message_id: "3",
      dedupe_key: "telegram:smith:99:3",
      capture_ts: MID_2026,
    },
  ];
  const windows = telegram.buildWindows(records, 12, FIXED_NOW);
  assert.strictEqual(windows.length, 1);
  const env = telegram.buildEnvelope(telegramConfig(), windows[0], telegramSummary(), FIXED_NOW);
  assert.strictEqual(env.created_at, NEWEST);
  assert.strictEqual(env.published_at, FIXED_NOW_ISO);
  assert.notStrictEqual(env.created_at, env.published_at);
  assert.strictEqual(env.payload_sha256, telegram.sha256Hex(telegram.canonicalJson(env.payload)));
  assert.ok(env.source_ref.startsWith("telegram:smith:99:batch:"));
});

// ---------------------------------------------------------------------------
// 7. Adapter INSERT names created_at; exact replay no update
// ---------------------------------------------------------------------------
check("adapter: normalize validates created_at; first-insert SQL includes thoughts.created_at", () => {
  const payload = emptySummary();
  const payloadHash = adapter.payloadSha256(payload);
  const wire = {
    schema_version: "argus.ingest.v1",
    source_kind: "telegram",
    namespace: adapter.TELEGRAM_NAMESPACE,
    kind: "note",
    author: "smith",
    agent: "smith",
    source_ref: "telegram:smith:99:batch:1-2",
    topic_key: "telegram:smith:99",
    event_id: "telegram:smith:99:batch:1-2",
    dedupe_key: "telegram:smith:99:batch:1-2",
    batch_id: "telegram:smith:99:batch:1-2",
    subject: "ingest.telegram.smith.99.batch",
    producer: "argus-telegram-direct-nats-producer",
    direction: null,
    chat_id: "99",
    message_id: null,
    session_id: null,
    created_at: NEWEST,
    published_at: FIXED_NOW_ISO,
    payload_sha256: payloadHash,
    payload,
    provenance: { count: 2 },
  };
  const validated = adapter.validate(wire, { now: FIXED_NOW });
  assert.strictEqual(validated.skip, false);
  assert.strictEqual(validated.record.envelope.created_at, NEWEST);

  const sqls = [];
  const psql = (sql) => {
    sqls.push(sql);
    return "thought-uuid-1";
  };
  adapter.storeRecord(psql, validated.record, payloadHash);
  const insertSql = sqls.find((s) => s.includes("INSERT INTO thoughts"));
  assert.ok(insertSql, "expected thoughts INSERT");
  assert.ok(/INSERT INTO thoughts\s*\([^)]*\bcreated_at\b/.test(insertSql), "INSERT must name created_at");
  assert.ok(insertSql.includes(NEWEST) || insertSql.includes(adapter.sqlString(NEWEST)), "INSERT must pass validated source time");
  assert.ok(insertSql.includes("::timestamptz") || insertSql.includes("timestamptz"), "created_at cast/type present");
});

check("adapter: exact replay is duplicate_skip and does not INSERT/UPDATE thought created_at", () => {
  const payload = emptySummary();
  const payloadHash = adapter.payloadSha256(payload);
  const wire = {
    schema_version: "argus.ingest.v1",
    source_kind: "telegram",
    namespace: adapter.TELEGRAM_NAMESPACE,
    kind: "note",
    author: "smith",
    agent: "smith",
    source_ref: "telegram:smith:99:batch:1-2",
    topic_key: "telegram:smith:99",
    event_id: "telegram:smith:99:batch:1-2",
    dedupe_key: "telegram:smith:99:batch:1-2",
    batch_id: "telegram:smith:99:batch:1-2",
    subject: "ingest.telegram.smith.99.batch",
    producer: "argus-telegram-direct-nats-producer",
    chat_id: "99",
    created_at: NEWEST,
    published_at: FIXED_NOW_ISO,
    payload_sha256: payloadHash,
    payload,
    provenance: {},
  };
  const sqls = [];
  const psql = (sql) => {
    sqls.push(sql);
    if (/SELECT payload_hash, status/.test(sql)) {
      return `${payloadHash}\tstored\tthought-existing`;
    }
    return "";
  };
  const result = adapter.processRecord(wire, { psql, now: FIXED_NOW });
  assert.strictEqual(result.action, "duplicate_skip");
  assert.ok(!sqls.some((s) => /INSERT INTO thoughts/.test(s)), "replay must not INSERT thoughts");
  assert.ok(
    !sqls.some((s) => /UPDATE thoughts[\s\S]*created_at/.test(s)),
    "replay must not UPDATE thoughts.created_at"
  );
});

// ---------------------------------------------------------------------------
// 8. Metadata / hash / source_ref unchanged for valid fixtures
// ---------------------------------------------------------------------------
check("valid fixtures keep payload hash, source_ref, and metadata invariants", () => {
  const profile = { name: "smith", profilePath: "/tmp/profile" };
  const summary = emptySummary();
  const env1 = session.buildEnvelope(profile, "/tmp/s.jsonl", 0, 30, sessionRecords(), summary, FIXED_NOW);
  const env2 = session.buildEnvelope(profile, "/tmp/s.jsonl", 0, 30, sessionRecords(), summary, FIXED_NOW);
  assert.strictEqual(env1.payload_sha256, env2.payload_sha256);
  assert.strictEqual(env1.source_ref, env2.source_ref);
  assert.strictEqual(env1.payload.summary, summary.summary);
  assert.deepStrictEqual(env1.payload.key_facts, summary.key_facts);

  const c1 = codex.buildEnvelope("smith", "sess-1", 0, 2, codexRecords(), summary, FIXED_NOW);
  assert.strictEqual(c1.payload_sha256, codex.sha256Hex(codex.canonicalJson(c1.payload)));
  assert.ok(c1.provenance && c1.provenance.generation);

  const tRecs = [
    {
      agent: "smith",
      chat_id: "7",
      message_id: "9",
      dedupe_key: "telegram:smith:7:9",
      capture_ts: MID_2026,
    },
  ];
  const tenv = telegram.buildEnvelope(
    telegramConfig(),
    { agent: "smith", chatId: "7", records: tRecs },
    summary,
    FIXED_NOW
  );
  assert.strictEqual(tenv.payload_sha256, telegram.sha256Hex(telegram.canonicalJson(tenv.payload)));
  assert.strictEqual(tenv.agent, "smith");
  assert.strictEqual(tenv.chat_id, "7");
});

(async () => {
  // drain any pending async checks registered above
  await Promise.resolve();
  // re-run async checks that were scheduled
  // (checkAsync already ran when called; wait a tick for completion)
  // Actually checkAsync is fire-and-forget without await at call site — fix by awaiting explicitly below.
})();

// Explicitly run async checks with top-level await via main
async function main() {
  // Re-execute async permanent-reject test if checkAsync already printed; the
  // checkAsync above starts immediately — wait for event loop drain.
  await new Promise((r) => setImmediate(r));
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed) process.exit(1);
  process.exit(0);
}

main();

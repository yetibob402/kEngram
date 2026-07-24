#!/usr/bin/env node
"use strict";
/*
 * M4 truth-trio rev4 slice-C test harness (spec dd21d8f9, Neo round-5 notes).
 *
 * Drives the general consumer's session + a2a per-message handlers end to end
 * against a REAL local PostgreSQL (migrations 0001..0030 + 0032) under the
 * REAL runtime role kengram_rt_session — never the owner.  NATS is stubbed
 * (fake msg objects recording ack/nak/term); the embed service is a local
 * HTTP stub speaking the openai-compatible /v1/embeddings shape.
 *
 * Zero production contact: everything runs against a disposable local
 * database; no live kEngram service, prod DB, or NATS subject is touched.
 *
 * Scenarios (spec C1c 1-4 + privilege assertion, C3 receipt trio, Neo note
 * ACK assertions):
 *   S1 first-insert -> stored, source-event row written by the gate, ACK
 *   S2 exact replay -> raw 'replay' normalized to duplicate_skip, ACK,
 *      no new thought
 *   S3 payload-hash conflict -> conflict surfaced to DLQ handling, ACK
 *   S4 concurrent identical delivery -> exactly one stored + one
 *      duplicate_skip, no deadlock
 *   S5 privilege assertions: outer source-event/thought DML fails loudly
 *      under kengram_rt_session; owner-side direct insert stays allowed
 *   S6 shadow_candidate -> BINDING: first delivery ACKed (Neo note c)
 *   S7 enforce semantic_duplicate -> BINDING: first delivery ACKed, no
 *      thought created (Neo note c)
 *   S8 controlled embed-down window -> structured fail_open_insert, ACK
 *   S9 a2a lane stored + replay through the gate
 *   S10 ledger receipts (shadow_candidate / fail_open_insert /
 *       semantic_duplicate all present) + vector coverage
 *
 * Run:  node contrib/argus/test/truth-trio-consumer.test.js
 * Env:  M4_PG_HOST (127.0.0.1)  M4_PG_PORT (55440)  M4_PG_SUPERUSER (kengram)
 *       M4_TEST_DB (kengram_m4_test)
 */

const assert = require("assert");
const { execFileSync } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const PG_HOST = process.env.M4_PG_HOST || "127.0.0.1";
const PG_PORT = process.env.M4_PG_PORT || "55440";
const PG_SUPERUSER = process.env.M4_PG_SUPERUSER || "kengram";
const TEST_DB = process.env.M4_TEST_DB || "kengram_m4_test";
const RT_ROLE = "kengram_rt_session";

const ADMIN_BOOT_DSN = `postgres://${PG_SUPERUSER}@${PG_HOST}:${PG_PORT}/postgres`;
const ADMIN_DSN = `postgres://${PG_SUPERUSER}@${PG_HOST}:${PG_PORT}/${TEST_DB}`;
const RT_DSN = `postgres://${RT_ROLE}@${PG_HOST}:${PG_PORT}/${TEST_DB}`;

const TEST_DIR = __dirname;
const REPO_ROOT = path.resolve(TEST_DIR, "../../..");
const MIGRATIONS_DIR = path.join(REPO_ROOT, "migrations");
const TMP_DIR = path.join(TEST_DIR, ".tmp");

function psql(dsn, sql, opts) {
  return execFileSync(
    "psql",
    [dsn, "-X", "-v", "ON_ERROR_STOP=1", "-t", "-A", "-F", "\t", "-c", sql],
    {
      encoding: "utf8",
      env: Object.assign({}, process.env, {
        PATH: "/opt/homebrew/bin:" + (process.env.PATH || ""),
      }),
      stdio: opts && opts.quietStderr ? ["ignore", "pipe", "pipe"] : undefined,
    }
  ).trim();
}

function psqlExpectError(dsn, sql) {
  try {
    psql(dsn, sql, { quietStderr: true });
  } catch (err) {
    return String(err.stderr || err.message || "");
  }
  return null;
}

function psqlFile(dsn, file) {
  // One transaction per migration file (how the migrator applies them),
  // except files using CREATE INDEX CONCURRENTLY, which forbid it.
  const oneTx = !/CONCURRENTLY/.test(fs.readFileSync(file, "utf8"));
  const args = [dsn, "-X", "-q", "-v", "ON_ERROR_STOP=1"];
  if (oneTx) args.push("-1");
  args.push("-f", file);
  execFileSync("psql", args, {
    encoding: "utf8",
    env: Object.assign({}, process.env, {
      PATH: "/opt/homebrew/bin:" + (process.env.PATH || ""),
    }),
  });
}

// --- embed stub -------------------------------------------------------------

const vectorByContent = new Map();
let embedDown = false;
let embedCalls = 0;

function axisVector(axis, value, spillAxis) {
  const v = new Array(1024).fill(0);
  if (spillAxis === undefined) {
    v[axis] = 1;
  } else {
    v[axis] = value;
    v[spillAxis] = Math.sqrt(1 - value * value);
  }
  return v;
}

function defaultVector(content) {
  // Deterministic, mutually near-orthogonal fallback: one hot axis by hash.
  let h = 0;
  for (const ch of content) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return axisVector(512 + (h % 500));
}

function startEmbedStub() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (embedDown) {
        res.statusCode = 500;
        res.end("embed stub down (controlled fail-open window)");
        return;
      }
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        embedCalls++;
        const input = JSON.parse(body).input[0];
        const vector = vectorByContent.get(input) || defaultVector(input);
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ data: [{ embedding: vector }] }));
      });
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

// --- envelope builders ------------------------------------------------------

let refCounter = 1000;

function sessionEnvelope(base, agent, payload) {
  const refStart = (refCounter += 100);
  const sourceRef = `session:${agent}:studio:harness:bytes:${refStart}-${refStart + 99}`;
  const fullPayload = Object.assign(
    {
      summary: "",
      key_facts: [],
      decisions: [],
      intents: [],
      action_items: [],
      open_questions: [],
      blockers: [],
      artifacts: [],
      corrections: [],
      topics: [],
      participants: [],
    },
    payload
  );
  return {
    schema_version: "argus.ingest.v1",
    source_kind: "session",
    kind: "note",
    agent,
    author: agent,
    namespace: `sessions/${agent}`,
    source_ref: sourceRef,
    event_id: sourceRef,
    dedupe_key: sourceRef,
    batch_id: sourceRef,
    subject: `ingest.session.${agent}.studio`,
    payload: fullPayload,
    payload_sha256: base.sha256Hex(base.canonicalJson(fullPayload)),
    created_at: new Date().toISOString(),
    published_at: new Date().toISOString(),
    producer: "argus-session-nats-producer",
    host: "studio",
    session_id: "harness",
    provenance: {},
  };
}

let seqCounter = 0;

function fakeMsg(subject, envelope) {
  return {
    subject,
    seq: ++seqCounter,
    data: Buffer.from(JSON.stringify(envelope), "utf8"),
    calls: [],
    ack() {
      this.calls.push("ack");
    },
    nak() {
      this.calls.push("nak");
    },
    term() {
      this.calls.push("term");
    },
  };
}

function count(sql) {
  return Number(psql(ADMIN_DSN, sql));
}

// --- setup ------------------------------------------------------------------

async function main() {
  fs.rmSync(TMP_DIR, { recursive: true, force: true });
  fs.mkdirSync(TMP_DIR, { recursive: true });

  console.log(`[setup] recreating ${TEST_DB} on ${PG_HOST}:${PG_PORT}`);
  psql(ADMIN_BOOT_DSN, `DROP DATABASE IF EXISTS ${TEST_DB} WITH (FORCE)`);
  psql(ADMIN_BOOT_DSN, `CREATE DATABASE ${TEST_DB}`);

  const migrations = fs
    .readdirSync(MIGRATIONS_DIR)
    .filter((f) => /^\d{4}_.*\.sql$/.test(f))
    .sort();
  console.log(`[setup] applying ${migrations.length} migrations (${migrations[0]} .. ${migrations[migrations.length - 1]})`);
  for (const file of migrations) {
    psqlFile(ADMIN_DSN, path.join(MIGRATIONS_DIR, file));
  }
  assert.strictEqual(
    psql(ADMIN_DSN, "SELECT tgenabled FROM pg_trigger WHERE tgname = 'thoughts_require_gated_writer'"),
    "O",
    "0032 must leave thoughts_require_gated_writer ENABLED"
  );

  const embedServer = await startEmbedStub();
  const embedPort = embedServer.address().port;
  const configPath = path.join(TMP_DIR, "kengram.test.toml");
  fs.writeFileSync(
    configPath,
    [
      "[embedder]",
      'provider = "openai-compatible"',
      `endpoint = "http://127.0.0.1:${embedPort}/v1"`,
      'model = "bge-m3"',
      'model_id = "bge-m3:1024"',
      "dimensions = 1024",
      "timeout_seconds = 5",
      "",
    ].join("\n")
  );

  process.env.KENGRAM_CONFIG_TOML = configPath;
  process.env.KENGRAM_CONSUMER_DB_URL = RT_DSN;
  process.env.KENGRAM_GENERAL_STATE_DIR = path.join(TMP_DIR, "state");
  process.env.NATS_LIB_DIR = path.join(TEST_DIR, "nats-stub");
  process.env.KENGRAM_GENERAL_RUN_TELEGRAM = "0";

  // Deployed-client contract (Neo B1): the consumer's gate calls must run on
  // the same psql major as the deployment host (yeti ships psql 14.x).
  const psql14 = process.env.M4_PSQL14 || "/opt/homebrew/opt/postgresql@14/bin/psql";
  if (!process.env.KENGRAM_PSQL_BIN && fs.existsSync(psql14)) {
    process.env.KENGRAM_PSQL_BIN = psql14;
  }
  const gateClient = process.env.KENGRAM_PSQL_BIN || "psql";
  const gateClientVersion = execFileSync(gateClient, ["--version"], {
    encoding: "utf8",
    env: Object.assign({}, process.env, { PATH: "/opt/homebrew/bin:" + (process.env.PATH || "") }),
  }).trim();
  console.log(`[client] consumer gate calls use: ${gateClientVersion} (${gateClient})`);
  if (!/PostgreSQL\) 14\./.test(gateClientVersion)) {
    console.log(
      "[client] EMULATION DISCLOSURE: no psql 14 client available — receipts were NOT produced on the deployed client generation"
    );
  }

  const consumer = require(path.join(REPO_ROOT, "contrib/argus/bin/argus-kengram-consumer.js"));
  const base = consumer.baseAdapter;
  const sessionDlq = path.join(process.env.KENGRAM_GENERAL_STATE_DIR, "session-adapter-dlq.jsonl");

  const passed = [];
  async function scenario(name, fn) {
    await fn();
    passed.push(name);
    console.log(`PASS ${name}`);
  }

  // --- S1 first insert ------------------------------------------------------
  const e1 = sessionEnvelope(base, "testa", {
    summary: "harness first insert for the truth trio gate path",
    key_facts: ["the consumer now routes every validated event through the gate"],
  });
  await scenario("S1 first-insert -> stored + ACK, gate writes the source-event row", async () => {
    const msg = fakeMsg(e1.subject, e1);
    const s = freshSessionStats(consumer);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"], "first insert must ACK");
    assert.strictEqual(s.stored, 1);
    const row = psql(
      ADMIN_DSN,
      `SELECT status, (thought_id IS NOT NULL) FROM argus_source_events WHERE source_ref = '${e1.source_ref}'`
    );
    assert.strictEqual(row, "stored\tt", "gate must write the source-event row as stored");
    assert.strictEqual(count(`SELECT count(*) FROM thoughts WHERE scope = 'sessions/testa'`), 1);
    assert.strictEqual(
      count(`SELECT count(*) FROM thought_ingest_gate_events WHERE source_event_ref = '${e1.source_ref}'`),
      1,
      "gate event inserted exactly once for the ACKed first delivery"
    );
  });

  // --- S2 exact replay ------------------------------------------------------
  await scenario("S2 exact replay -> raw 'replay' normalized to duplicate_skip + ACK", async () => {
    const direct = await consumer.processSessionRecordGated(e1, { dbUrl: RT_DSN, dlqPath: sessionDlq });
    assert.strictEqual(direct.action, "duplicate_skip", "capture.rs:213-236 vocab: replay -> duplicate_skip");
    const msg = fakeMsg(e1.subject, e1);
    const s = freshSessionStats(consumer);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"]);
    assert.strictEqual(s.dup, 1);
    assert.strictEqual(count(`SELECT count(*) FROM thoughts WHERE scope = 'sessions/testa'`), 1, "replay must not create a thought");
  });

  // --- S3 payload-hash conflict ---------------------------------------------
  await scenario("S3 payload-hash conflict -> conflict status + DLQ row + ACK", async () => {
    const conflicting = JSON.parse(JSON.stringify(e1));
    conflicting.payload.summary = "harness conflicting re-distillation of the same source ref";
    conflicting.payload_sha256 = base.sha256Hex(base.canonicalJson(conflicting.payload));
    const msg = fakeMsg(conflicting.subject, conflicting);
    const s = freshSessionStats(consumer);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"], "conflict rides the existing ack+DLQ path");
    assert.strictEqual(s.conflict, 1);
    assert.strictEqual(
      psql(ADMIN_DSN, `SELECT status FROM argus_source_events WHERE source_ref = '${e1.source_ref}'`),
      "conflict"
    );
    const dlqRows = fs
      .readFileSync(sessionDlq, "utf8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l));
    assert.ok(
      dlqRows.some((r) => r.reason === "payload_hash_conflict" && r.source_ref === e1.source_ref),
      "metadata-only conflict DLQ row required"
    );
  });

  // --- S4 concurrent first delivery -----------------------------------------
  await scenario("S4 concurrent identical delivery -> exactly one stored + one duplicate_skip", async () => {
    const e4 = sessionEnvelope(base, "testa", {
      summary: "harness concurrent delivery event for the atomic claim check",
      key_facts: ["two simultaneous identical deliveries must resolve deterministically"],
    });
    const [r1, r2] = await Promise.all([
      consumer.processSessionRecordGated(e4, { dbUrl: RT_DSN, dlqPath: sessionDlq }),
      consumer.processSessionRecordGated(e4, { dbUrl: RT_DSN, dlqPath: sessionDlq }),
    ]);
    const actions = [r1.action, r2.action].sort();
    assert.deepStrictEqual(actions, ["duplicate_skip", "stored"], `got ${actions}`);
    assert.strictEqual(
      count(`SELECT count(*) FROM argus_source_events WHERE source_ref = '${e4.source_ref}'`),
      1
    );
  });

  // --- S5 privilege assertions ----------------------------------------------
  await scenario("S5 runtime-role privilege assertions (outer DML dead by construction)", async () => {
    const insErr = psqlExpectError(
      RT_DSN,
      "INSERT INTO argus_source_events (namespace, source_ref, payload_hash, status) VALUES ('x','y','z','pending')"
    );
    assert.ok(insErr && /permission denied/i.test(insErr), "rt INSERT on argus_source_events must fail loudly");
    const updErr = psqlExpectError(RT_DSN, "UPDATE argus_source_events SET last_seen_at = NOW()");
    assert.ok(updErr && /permission denied/i.test(updErr), "rt UPDATE on argus_source_events must fail loudly");
    const thoughtErr = psqlExpectError(
      RT_DSN,
      "INSERT INTO thoughts (scope, content, source, content_fingerprint) VALUES ('agents/x','c','s', digest('c','sha256'))"
    );
    assert.ok(thoughtErr && /permission denied|thought_insert_requires_capture_thought_gated/i.test(thoughtErr));
    // 0032 scoping check: the deny is scoped to the runtime role family only —
    // an owner-side (unmigrated legacy writer) direct insert must still work:
    psql(
      ADMIN_DSN,
      "INSERT INTO thoughts (scope, content, source, content_fingerprint) VALUES ('agents/legacy','legacy writer parity probe','harness', digest('legacy writer parity probe','sha256'))"
    );
    assert.strictEqual(count("SELECT count(*) FROM thoughts WHERE scope = 'agents/legacy'"), 1, "owner/unmigrated writer path must remain open this slice");
    const rtSelect = psqlExpectError(RT_DSN, "SELECT count(*) FROM argus_source_events");
    console.log(`  [info] rt SELECT on argus_source_events: ${rtSelect ? "denied in migration-built DB (live role holds SELECT per Neo read)" : "allowed"}`);
  });

  // --- S6 shadow_candidate (BINDING ACK assertion) --------------------------
  psql(ADMIN_DSN, `UPDATE corpus_hygiene_gate_settings SET mode = 'shadow' WHERE principal_name = '${RT_ROLE}'`);
  await scenario("S6 shadow_candidate -> BINDING first-delivery ACK", async () => {
    const f1 = sessionEnvelope(base, "testa", {
      summary:
        "the westshore pipeline finished the nightly baseline regeneration for every resident site and the output matched the previous run exactly",
      key_facts: ["baseline regeneration completed for all resident sites"],
    });
    const f2 = sessionEnvelope(base, "testa", {
      summary:
        "the westshore pipeline finished the overnight baseline regeneration for every resident site and the output matched the previous run exactly",
      key_facts: ["baseline regeneration completed for all resident sites"],
    });
    const f1Content = consumer.sessionAdapter.buildThoughtContent({ payload: f1.payload });
    const f2Content = consumer.sessionAdapter.buildThoughtContent({ payload: f2.payload });
    vectorByContent.set(f1Content, axisVector(2));
    vectorByContent.set(f2Content, axisVector(2, 0.85, 3)); // cosine 0.85 >= 0.80 observation floor

    let s = freshSessionStats(consumer);
    let msg = fakeMsg(f1.subject, f1);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"]);

    s = freshSessionStats(consumer);
    msg = fakeMsg(f2.subject, f2);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"], "BINDING: shadow_candidate first delivery must ACK");
    assert.strictEqual(s.gate_shadow_candidate, 1, "gate must classify shadow_candidate");
    assert.strictEqual(s.stored, 1, "shadow mode still inserts");
    assert.strictEqual(
      count(
        `SELECT count(*) FROM thought_ingest_gate_events WHERE action = 'shadow_candidate' AND producer_principal = '${RT_ROLE}'`
      ),
      1,
      "shadow_candidate ledger receipt"
    );
    console.log("  BINDING ACK ASSERT shadow_candidate: first delivery ACKed (msg.calls === ['ack'])");
  });

  // --- S7 enforce semantic_duplicate (BINDING ACK assertion) ----------------
  psql(ADMIN_DSN, `UPDATE corpus_hygiene_gate_settings SET mode = 'enforce' WHERE principal_name = '${RT_ROLE}'`);
  await scenario("S7 enforce semantic_duplicate -> BINDING first-delivery ACK, no thought created", async () => {
    const g1 = sessionEnvelope(base, "testa", {
      summary:
        "the lake association crawler completed the weekly classification sweep across all member communities and every generated page passed review without further edits from the maintainers",
      key_facts: ["classification sweep completed cleanly across all member communities"],
    });
    const g2 = sessionEnvelope(base, "testa", {
      summary:
        "the lake association crawler completed the weekly classification sweep across all member communities and every generated page passed inspection without further edits from the maintainers",
      key_facts: ["classification sweep completed cleanly across all member communities"],
    });
    const g1Content = consumer.sessionAdapter.buildThoughtContent({ payload: g1.payload });
    const g2Content = consumer.sessionAdapter.buildThoughtContent({ payload: g2.payload });
    assert.ok(g2Content.length >= 120, "enforce skip requires len >= 120");
    vectorByContent.set(g1Content, axisVector(4));
    vectorByContent.set(g2Content, axisVector(4, 0.97, 5)); // cosine 0.97 > 0.90 threshold

    let s = freshSessionStats(consumer);
    let msg = fakeMsg(g1.subject, g1);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"]);
    const thoughtsBefore = count("SELECT count(*) FROM thoughts");

    s = freshSessionStats(consumer);
    msg = fakeMsg(g2.subject, g2);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"], "BINDING: semantic_duplicate first delivery must ACK");
    assert.strictEqual(s.semantic_skip, 1, "semantic skip ACKs without creating a thought");
    assert.strictEqual(count("SELECT count(*) FROM thoughts"), thoughtsBefore, "no thought may be created");
    const eventRow = psql(
      ADMIN_DSN,
      `SELECT status, (thought_id IS NOT NULL) FROM argus_source_events WHERE source_ref = '${g2.source_ref}'`
    );
    assert.strictEqual(eventRow, "skipped\tt", "semantic skip must record status=skipped pointing at the match");
    assert.strictEqual(
      count(
        `SELECT count(*) FROM thought_ingest_gate_events WHERE action = 'semantic_duplicate' AND producer_principal = '${RT_ROLE}'`
      ),
      1,
      "semantic_duplicate ledger receipt"
    );
    console.log("  BINDING ACK ASSERT semantic_duplicate: first delivery ACKed (msg.calls === ['ack'])");
  });

  // --- S8 controlled embed-down window -> structured fail-open --------------
  await scenario("S8 embed-down window -> structured fail_open_insert + ACK (never silent, never dropped)", async () => {
    embedDown = true;
    try {
      const h1 = sessionEnvelope(base, "testa", {
        summary: "harness event captured while the embedding service is deliberately unavailable",
        key_facts: ["fail open must keep the thought and record a structured bypass"],
      });
      const s = freshSessionStats(consumer);
      const msg = fakeMsg(h1.subject, h1);
      await consumer.processSessionMessage(msg, s);
      assert.deepStrictEqual(msg.calls, ["ack"], "fail-open insert must ACK");
      assert.strictEqual(s.stored, 1);
      assert.strictEqual(s.gate_fail_open, 1, "gate must classify fail_open_insert");
      assert.strictEqual(s.embed_bypass, 1, "consumer must count the bypass for vector coverage");
      const bypass = psql(
        ADMIN_DSN,
        `SELECT bypass_reason->>'code' FROM thought_ingest_gate_events WHERE action = 'fail_open_insert' AND producer_principal = '${RT_ROLE}'`
      );
      assert.strictEqual(bypass, "embedding_unavailable", "structured bypass_reason receipt");
    } finally {
      embedDown = false;
    }
  });

  // --- S8b malformed embed response (Neo B2 RED/GREEN) ----------------------
  await scenario("S8b malformed embed response (1024 nulls) -> structured fail_open_insert + ACK", async () => {
    const m1 = sessionEnvelope(base, "testa", {
      summary: "harness event whose embedder response is a full vector of nulls",
      key_facts: ["malformed embedder output must fail open with a structured bypass"],
    });
    const mContent = consumer.sessionAdapter.buildThoughtContent({ payload: m1.payload });
    vectorByContent.set(mContent, new Array(1024).fill(null)); // right length, no finite numbers
    const s = freshSessionStats(consumer);
    const msg = fakeMsg(m1.subject, m1);
    await consumer.processSessionMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"], "malformed embed vector must fail open and ACK, never term");
    assert.strictEqual(s.stored, 1);
    assert.strictEqual(s.gate_fail_open, 1, "gate must classify fail_open_insert");
    assert.strictEqual(s.embed_bypass, 1, "malformed response counts as a bypass for coverage");
    const bypass = psql(
      ADMIN_DSN,
      `SELECT bypass_reason->>'code' || ':' || (bypass_reason->>'detail') FROM thought_ingest_gate_events WHERE source_event_ref = '${m1.source_ref}'`
    );
    assert.strictEqual(bypass, "embedding_unavailable:malformed_vector_elements", "structured bypass receipt");
  });

  // --- S9 a2a lane through the gate -----------------------------------------
  psql(ADMIN_DSN, `UPDATE corpus_hygiene_gate_settings SET mode = 'off' WHERE principal_name = '${RT_ROLE}'`);
  await scenario("S9 a2a lane -> gate-routed stored + replay duplicate_skip", async () => {
    const a2aEnvelope = {
      from: "carl",
      to: "knox",
      type: "status",
      re: "m4 truth trio harness",
      text: "the a2a lane now routes through the capture gate under the runtime role",
      ts: new Date().toISOString(),
    };
    let s = freshA2AStats(consumer);
    let msg = fakeMsg("agent.knox.inbox", a2aEnvelope);
    await consumer.processA2AMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"]);
    assert.strictEqual(s.stored, 1);
    assert.strictEqual(count("SELECT count(*) FROM thoughts WHERE scope = 'agents/carl'"), 1);

    s = freshA2AStats(consumer);
    msg = fakeMsg("agent.knox.inbox", a2aEnvelope);
    await consumer.processA2AMessage(msg, s);
    assert.deepStrictEqual(msg.calls, ["ack"]);
    assert.strictEqual(s.dup, 1, "a2a replay must normalize to duplicate_skip");
    assert.strictEqual(count("SELECT count(*) FROM thoughts WHERE scope = 'agents/carl'"), 1);
  });

  // --- S10 ledger receipts + vector coverage --------------------------------
  await scenario("S10 C3 receipt trio present in the ledger + vector coverage", async () => {
    const byAction = psql(
      ADMIN_DSN,
      `SELECT action, count(*) FROM thought_ingest_gate_events WHERE producer_principal = '${RT_ROLE}' GROUP BY action ORDER BY action`
    );
    console.log(`  [ledger] ${byAction.replace(/\n/g, "  ")}`);
    for (const needed of ["shadow_candidate", "fail_open_insert", "semantic_duplicate"]) {
      assert.ok(new RegExp(`(^|\\n)${needed}\t[1-9]`).test(byAction), `ledger must hold >= 1 ${needed}`);
    }
    const cov = psql(
      ADMIN_DSN,
      `SELECT count(*) FILTER (WHERE bypass_reason IS NULL), count(*) FROM thought_ingest_gate_events WHERE producer_principal = '${RT_ROLE}'`
    ).split("\t");
    const coverage = Number(cov[0]) / Number(cov[1]);
    console.log(
      `  [coverage] candidate-vector coverage = ${cov[0]}/${cov[1]} = ${(coverage * 100).toFixed(1)}% ` +
        `(enforce flip requires >= 95%; fail-open rows carry no semantic verdict and are excluded from the would-be-skip denominator)`
    );
    assert.ok(coverage > 0 && coverage < 1, "harness window must show both covered and bypassed calls");
    assert.strictEqual(
      Number(cov[1]) - Number(cov[0]),
      2,
      "exactly the embed-down and malformed-vector calls are uncovered"
    );
  });

  embedServer.close();
  fs.rmSync(TMP_DIR, { recursive: true, force: true });
  console.log(`\nRESULT: ${passed.length}/${passed.length} scenarios passed (${embedCalls} embed-stub calls)`);
  console.log("BINDING (Neo round-5 note c): shadow_candidate first-delivery ACK asserted — PASS");
  console.log("BINDING (Neo round-5 note c): semantic_duplicate first-delivery ACK asserted — PASS");
}

function freshSessionStats(consumer) {
  // sessionStats() is not exported; reconstruct via a metrics-shaped object.
  return {
    fetched: 0,
    stored: 0,
    dup: 0,
    skipped: 0,
    semantic_skip: 0,
    conflict: 0,
    invalid: 0,
    dry_run_valid: 0,
    held: 0,
    acked: 0,
    termed: 0,
    naked: 0,
    embed_ok: 0,
    embed_bypass: 0,
    gate_fail_open: 0,
    gate_shadow_candidate: 0,
  };
}

function freshA2AStats(consumer) {
  return Object.assign(freshSessionStats(consumer), {
    skipped_system_events: 0,
    skipped_telegram_passthrough: 0,
  });
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("FAIL", err && err.stack ? err.stack : err);
    process.exit(1);
  });

# Kengram M2 smoke test

An operator-driven checklist that exercises every MCP tool end-to-end through a real chat client. Closes the manual MCP-smoke item in `docs/milestones/m2-progress.md`.

Pair the chat steps with the `psql` one-liners so you can confirm what landed in the database after each call. The same prompts work in **Claude Desktop**, **Claude Code**, or **opencode** — anywhere the MCP server has been wired per the README.

## Prerequisites

- `docker compose up -d postgres` and migrations applied.
- `kengram serve` running in one terminal.
- `kengram worker` running in another (the Tier-1 steps need the embed drainer; Tier 2 also needs `KENGRAM_REFLECTOR__ENABLED=true`).
- MCP client configured to point at `http://127.0.0.1:8080/mcp` (see README "Connecting MCP clients").
- **Tier 2 only:** vLLM (or another OpenAI-compatible chat endpoint) reachable at `[extractor].endpoint` (default `http://localhost:8000/v1`).

**One-time setup** — paste this **once** into the shell you'll run the verifications from. It defines a function (more robust than an alias for quoted SQL) that survives for the rest of the session:

```bash
kengram-psql() { docker exec -i kengram-postgres psql -U kengram -d kengram -t -c "$1"; }
```

Sanity check: `kengram-psql "SELECT 1"` should print `1`.

If you'd rather not define a helper, substitute the full command wherever the runbook says `kengram-psql "..."`:

```bash
docker exec -i kengram-postgres psql -U kengram -d kengram -t -c "SELECT 1"
```

---

## Tier 1 — Thoughts only (Phase B integration; no vLLM)

Verifies capture → embed-drainer → search/recent/get_thought. Should pass with just Postgres + Ollama running.

### 1. Capture a thought

**Prompt:**

> Use kengram's `capture` tool to save this thought, with `scope: "smoke-test"` and `source: "manual"`:
> _"Kengram uses Postgres with pgvector for vector storage and pg_trgm for trigram search. M2 ships six MCP tools."_

**Expected:** the client reports a `thought_id` (UUID) and `embedding_status: "pending"`. **Copy the `thought_id`** — Step 3 uses it.

**Verify:**

```bash
kengram-psql "SELECT id, scope, source, length(content) FROM thoughts WHERE scope='smoke-test' ORDER BY created_at DESC LIMIT 1"
kengram-psql "SELECT count(*) FROM pending_embeddings"  # ≥ 1 immediately after capture
```

### 2. Wait ~10 seconds for the worker tick

The default `[worker].tick_interval_seconds = 5`, so two ticks give a comfortable margin. You can also force-drain with `kengram embed-backfill --limit 10` if you don't want to wait.

**Verify:**

```bash
kengram-psql "SELECT count(*) FROM pending_embeddings WHERE target_kind='thought'"  # → 0
kengram-psql "SELECT target_kind, model_id, length(vector::text) > 0 FROM embeddings ORDER BY created_at DESC LIMIT 1"
```

### 3. `get_thought` should now report `indexed`

**Prompt:**

> Use kengram's `get_thought` tool on `<thought_id from step 1>`. What does the `provenance.embedding_status` say?

**Expected:** `embedding_status: "indexed"`, `embedded_at` is a non-null RFC-3339 timestamp, `linked_facts: []` (no facts yet — Tier 2 populates them).

### 4. `search_thoughts` hybrid hit

**Prompt:**

> Use kengram's `search_thoughts` tool with query `"pgvector"`, scope `"smoke-test"`. What does it return?

**Expected:** the thought from step 1 in `results`, `vector_search_available: true`, a positive `score`. If vector_search_available is false, the embedder is unreachable (check Ollama).

### 5. `recent_thoughts`

Capture one more thought to exercise ordering:

**Prompt:**

> Capture a second thought in `scope: "smoke-test"`, source `"manual"`: _"This is the second smoke-test thought, captured a moment later."_ Then call `recent_thoughts` with `scope: "smoke-test"` to confirm both show up newest-first.

**Expected:** two results, the second-captured one first by `created_at`.

---

## Tier 2 — Facts pipeline (Phase C/D, requires vLLM)

Verifies reflector → `search_facts` → `correct_fact` → `linked_facts` on `get_thought`. Skip this section if vLLM isn't running.

### 6. Force a reflector pass over the smoke-test scope

Rather than wait for cron (default `0 0 3 * * *` — 03:00 daily), drive it manually:

```bash
DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram' \
KENGRAM_REFLECTOR__ENABLED=true \
  cargo run --bin kengram -- reflect --scope smoke-test --limit 10
```

**Expected log line:** `reflect complete processed=2 committed=N review=M failures=0` (N + M ≥ 1 if the model produced anything).

**Verify:**

```bash
kengram-psql "SELECT count(*) FROM facts WHERE scope='smoke-test' AND superseded_at IS NULL"
kengram-psql "SELECT statement, confidence FROM facts WHERE scope='smoke-test' ORDER BY created_at DESC LIMIT 5"
kengram-psql "SELECT started_at, finished_at, n_thoughts_processed, n_facts_committed, n_review_queue, error FROM reflector_runs ORDER BY started_at DESC LIMIT 1"
```

### 7. `search_facts`

**Prompt:**

> Use kengram's `search_facts` tool with query `"pgvector"`, scope `"smoke-test"`. What does it return?

**Expected:** at least one fact whose `statement` mentions pgvector. Each result must include `source_thought_content`, `source_thought_scope: "smoke-test"`, `source_thought_created_at`. **Copy a `fact_id`** — step 9 needs it.

### 8. `get_thought` now carries `linked_facts`

**Prompt:**

> Call `get_thought` again on `<thought_id from step 1>`. What's in `provenance.linked_facts`?

**Expected:** a non-empty array. Each entry has a `fact_id`, `statement`, `extractor_model` (the configured vLLM model id, e.g. `"vllm/qwen2.5-7b-instruct"`), and `extractor_version`.

### 9. `correct_fact` — replace path

**Prompt:**

> Use kengram's `correct_fact` tool on `<fact_id from step 7>`, replacement `{ statement: "Kengram uses pgvector for vector storage and pg_trgm for trigram search.", subject: "Kengram", predicate: "uses", object: "pgvector + pg_trgm" }`.

**Expected:** `{ superseded: true, new_fact_id: "<UUID>" }`.

**Verify the audit trail:**

```bash
# Old row stays in `facts`, superseded_at populated, superseded_by → new fact
kengram-psql "SELECT id, superseded_at IS NOT NULL AS gone, superseded_by FROM facts WHERE id = '<fact_id from step 7>'"

# New row has manual-sentinel provenance
kengram-psql "SELECT extractor_model, extractor_version, source_run_id IS NULL AS no_run, confidence FROM facts WHERE id = '<new_fact_id from step 9>'"
# expect: manual | 0 | t | 1
```

### 10. `search_facts` after correction

**Prompt:** repeat step 7's query. **Expected:** the superseded fact is gone; the new (manual-author) fact appears in its place.

### 11. `correct_fact` — retract path

**Prompt:**

> Use `correct_fact` on `<new_fact_id from step 9>` with no replacement (omit the `replacement` field).

**Expected:** `{ superseded: true, new_fact_id: null }`. `search_facts` from step 10 now returns one fewer result; `get_thought`'s `linked_facts` shrinks accordingly.

### 12. Rerun is idempotent

```bash
DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram' \
  cargo run --bin kengram -- reflect --rerun --scope smoke-test
# Run twice — second run should report committed=0 because every fact still matches (S,P,O,statement).
```

---

## Via Claude Code CLI (one-off mode)

If you'd rather drive this from `claude -p` without an interactive session (e.g. for a scripted runbook), each numbered step works as a single prompt. Example:

```bash
claude -p "Use kengram's capture tool to save 'smoke test' in scope smoke-test, source manual. Report the thought_id verbatim."
```

You need an `.mcp.json` in the current directory (or a user-scoped `~/.claude.json` entry) pointing at `http://127.0.0.1:8080/mcp` — same shape as in the README's Claude Code section.

---

## Cleanup

```bash
kengram-psql "DELETE FROM thoughts WHERE scope = 'smoke-test'"
# ON DELETE CASCADE on facts.source_thought_id wipes the derived facts;
# review-queue rows and the embedding row (FK on target_id) need a manual sweep:
kengram-psql "DELETE FROM pending_embeddings WHERE target_kind='thought' AND target_id NOT IN (SELECT id FROM thoughts)"
kengram-psql "DELETE FROM embeddings WHERE target_kind='thought' AND target_id NOT IN (SELECT id FROM thoughts)"
kengram-psql "DELETE FROM facts_review_queue WHERE source_thought_id IS NULL"
```

# M4 — Collapse to thoughts-only (Path B-OB1)

## Context

M3's Phase D dogfood produced negative knowledge: the `facts` pipeline as currently designed isn't earning its complexity for the operator's use case. Across 7 dogfood rounds the recurring pattern was statements come back faithful, triples come back broken, and each prompt patch trades one failure mode for another. The consumer (LLM agents reading prose) doesn't query by (S, P, O); the producer (a local 30B coding model) can't reliably emit triples. The substrate that generates the failure modes is the wrong abstraction for the use case.

Independent corroboration came from Nate B. Jones's [Open Brain (OB1)](https://github.com/NateBJones-Projects/OB1) released 2026-03-11: same problem space (MCP-native personal memory), single `thoughts` table, LLM extraction collapsed into a JSONB `metadata` column on the same row rather than a separate `facts` entity. Stated philosophy: *"raw data is permanent, embeddings are derived."*

**M4 collapses kEngram to thoughts-only with a metadata sidecar** ([Path B-OB1](../../DESIGN.md#path-b-ob1), decided 2026-05-16). The `facts` table goes away. The reflector's role narrows from "extract structured atomic claims with (S, P, O, confidence)" to "tag each thought with metadata (people, action_items, topics, type)." Retrieval continues to use the M3-shipped hybrid + cross-encoder rerank pipeline — operating only on thoughts, with tags as an optional JSONB filter signal. Content-fingerprint dedup is added at the thought level (SHA-256, unique-indexed) so we don't regress vs OB1 on duplicate-handling.

**What we keep that OB1 doesn't have:**
- `pg_trgm` + RRF hybrid + cross-encoder rerank (OB1 is pure cosine).
- Provenance on extractor outputs (`tags_extractor_model` / `tags_extractor_version` / `tags_extracted_at`) so selective re-tagging on prompt/model changes works.

**What we drop (deliberately):**
- Entire `facts` retrieval surface (table, embedding fleet, MCP tools, dedup machinery).
- Confidence-band routing for facts (was Phase C three-band; thoughts have no confidence).
- All SPO infrastructure (subject/predicate/object columns, anchor rules, comparative few-shots, subsumption, quality-aware pick, retraction durability for facts).

## Architectural target (end state)

### Schema

`migrations/0006_collapse_to_thoughts.sql`:

```sql
-- Drop the facts pipeline entirely.
DROP TABLE IF EXISTS facts_review_queue CASCADE;
DROP TABLE IF EXISTS reflector_runs CASCADE;
DROP TABLE IF EXISTS facts CASCADE;

-- Clean up fact-targeted embedding rows + pending queue entries.
DELETE FROM embeddings WHERE target_kind = 'fact';
DELETE FROM pending_embeddings WHERE target_kind = 'fact';

-- (Optional: drop the 'fact' value from target_kind enum/check constraint.
-- Defer — leaving the value in lets us add it back later without a migration.)

-- Add thought-level dedup and tag sidecar.
ALTER TABLE thoughts
    ADD COLUMN content_fingerprint BYTEA,                    -- SHA-256 of content; nullable until backfilled
    ADD COLUMN tags JSONB NOT NULL DEFAULT '{}',             -- LLM-tagger output (see schema below)
    ADD COLUMN tags_extractor_model TEXT,                    -- provenance: tagger model_id
    ADD COLUMN tags_extractor_version INT,                   -- provenance: tagger prompt version
    ADD COLUMN tags_extracted_at TIMESTAMPTZ;                -- provenance: when tagged

-- Backfill content_fingerprint for existing thoughts (SHA-256 of content).
UPDATE thoughts
SET content_fingerprint = digest(content, 'sha256')
WHERE content_fingerprint IS NULL;

-- Lock content_fingerprint NOT NULL + UNIQUE post-backfill.
ALTER TABLE thoughts
    ALTER COLUMN content_fingerprint SET NOT NULL,
    ADD CONSTRAINT thoughts_content_fingerprint_unique UNIQUE (content_fingerprint);

-- GIN index on tags JSONB for containment queries.
CREATE INDEX thoughts_tags_gin ON thoughts USING gin (tags);
```

### Thought struct (kengram-core)

```rust
pub struct Thought {
    pub id: ThoughtId,
    pub scope: Scope,
    pub content: String,
    pub source: Source,
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,                  // client-provided (unchanged)
    pub content_fingerprint: [u8; 32],       // SHA-256 of content; NEW
    pub tags: Tags,                          // LLM-extracted sidecar; NEW
    pub tags_extractor_model: Option<String>,
    pub tags_extractor_version: Option<i32>,
    pub tags_extracted_at: Option<OffsetDateTime>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Tags {
    pub people: Vec<String>,
    pub action_items: Vec<String>,
    pub topics: Vec<String>,           // 1-3 short tags
    pub dates_mentioned: Vec<String>,  // free-text dates as the LLM emits them
    pub kind: Option<TagKind>,         // observation | task | idea | reference | person_note | session
}

pub enum TagKind {
    Observation, Task, Idea, Reference, PersonNote, Session,
}
```

`Tags` is `JSONB-shaped`; an empty `Tags::default()` round-trips as `{}`. Untagged thoughts (newly captured, pre-drainer) carry the default.

### Tagger (kengram-extract, repurposed)

Replace `Extractor` trait with `Tagger`. Output shape:

```rust
#[async_trait]
pub trait Tagger: Send + Sync {
    fn model_id(&self) -> &str;
    fn version(&self) -> i32;
    async fn tag(&self, thought_content: &str) -> Result<Tags, TaggerError>;
}
```

JSON Schema sent to the LLM:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["people", "action_items", "topics", "dates_mentioned", "kind"],
  "properties": {
    "people": { "type": "array", "items": { "type": "string" } },
    "action_items": { "type": "array", "items": { "type": "string" } },
    "topics": { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
    "dates_mentioned": { "type": "array", "items": { "type": "string" } },
    "kind": { "enum": ["observation", "task", "idea", "reference", "person_note", "session"] }
  }
}
```

Prompt (v1 of tagger; numbered separately from extractor v4):

```
You are a tagging assistant. Given a single thought, return its metadata tags as JSON.

# Output shape
{ "people": [...], "action_items": [...], "topics": [...], "dates_mentioned": [...], "kind": "..." }

# Field semantics
- people: bare names of people mentioned. Empty array if none.
- action_items: short imperative phrases describing tasks the thought commits to or implies.
- topics: 1-3 short tag-like topics (lowercase, no punctuation). What is this thought ABOUT?
- dates_mentioned: any dates or temporal references appearing in the prose ("next Thursday", "Q3", "2026-05-15"). Free-form strings.
- kind: a single classification — observation (factual claim), task (a thing to do), idea (a proposal), reference (a pointer to external resource), person_note (a fact about a specific person), or session (transient session/test narrative).

# Rules
- Only extract what is explicitly there. Do not infer.
- A session-narrative thought ("the test passed", "search returned X") should be kind=session with otherwise-empty arrays.
- Empty arrays are fine for any field that has no content.
```

That's the whole prompt. Order-of-magnitude shorter than v4. No SPO rules, no anchor checks, no confidence rubric, no comparatives. Tagger output is *advisory* — it doesn't gate storage; even a totally-wrong tag is low-impact because retrieval still works on the statement content.

### Retrieval

`search_thoughts` is the only retrieval surface. The MCP tool gains one new optional argument:

```rust
pub struct SearchRequest {
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,
    pub recency_half_life_days: Option<f32>,
    pub rerank: Option<bool>,
    pub candidate_pool: Option<usize>,
    pub tag_filter: Option<serde_json::Value>,   // NEW: JSONB containment filter against thoughts.tags
}
```

`tag_filter` example: `{"kind": "task"}` returns only thoughts the tagger classified as tasks. `{"people": ["Sarah"]}` returns thoughts mentioning Sarah. Implemented as `WHERE tags @> $tag_filter` (JSONB containment, GIN-indexed). When `tag_filter` is None, no tag filter applied.

`SearchHit` gains `tags: Tags` so agents can see the tagger's output per result.

### Capture (with fingerprint dedup)

Capture flow:

1. Agent calls `capture(content, source, scope?, metadata?)`.
2. Compute `content_fingerprint = sha256(content)`.
3. `INSERT INTO thoughts (...) VALUES (..., $fingerprint, '{}'::jsonb, NULL, NULL, NULL) ON CONFLICT (content_fingerprint) DO NOTHING RETURNING id`.
4. If insert succeeded: enqueue embedding job + enqueue tag job; return new `thought_id`.
5. If insert conflict: SELECT existing `thought_id` by fingerprint; return it with `embedding_status` reflecting the existing row's state; no new jobs enqueued.

Same scope semantics whether new or duplicate — agents see a stable `thought_id` for the same content regardless of how many times they capture it.

### Drainer (worker)

`kengram worker` drains two queue tables:

- `pending_embeddings` (unchanged): worker calls embedder, inserts into `embeddings`.
- `pending_tags` (NEW): worker calls tagger, updates `thoughts.tags` + provenance columns.

Both drain in parallel ticks. Same `tick_interval_seconds` + `batch_size` knobs.

### MCP surface delta

| Tool | M3 state | M4 state |
|---|---|---|
| `capture` | Returns `thought_id` + `embedding_status` | Same, plus duplicate handling via fingerprint (returns existing id on duplicate content); response gains `is_duplicate: bool` |
| `search_thoughts` | Hybrid retrieval over thoughts | Same, plus `tag_filter` parameter and `tags` field per hit |
| `recent_thoughts` | Browse by recency | Same |
| `get_thought` | Thought + provenance + `linked_facts` | Same minus `linked_facts` (facts no longer exist); gains `tags` + tag provenance |
| `search_facts` | Hybrid retrieval over facts | **Removed** |
| `correct_fact` | Operator-driven fact correction | **Removed** (operators can `retract_thought` + capture a corrected thought; tags are advisory and re-derivable) |
| `retract_thought` | Atomically retract thought + cascade-supersede derived facts | Simplified — no cascade needed since facts don't exist |

### CLI surface delta

| Subcommand | M3 state | M4 state |
|---|---|---|
| `kengram serve` | MCP server | Unchanged |
| `kengram worker` | Embed drainer + reflector cron | Embed drainer + **tag drainer** (reflector cron gone) |
| `kengram migrate` | Apply migrations | Unchanged |
| `kengram embed-backfill` | Heal-then-drain for thoughts + facts | Thoughts only |
| `kengram reflect [--rerun --since X]` | One-shot facts extraction | **Removed**; replaced by `kengram tag [--rerun --since X]` (one-shot tagger run; reruns re-tag thoughts) |
| `kengram bench rerank --corpus X` | A/B harness over thoughts + facts | Thoughts only (fixture format simplifies — `target` field goes away) |

### Config surface delta

| Section | M3 state | M4 state |
|---|---|---|
| `[server]` | bind | Unchanged |
| `[database]` | url, max_connections | Unchanged |
| `[embedder]` | provider/endpoint/model/model_id/dimensions/api_key/timeout | Unchanged |
| `[reranker]` | provider/endpoint/model_id/timeout | Unchanged |
| `[worker]` | tick_interval_seconds, batch_size | Unchanged |
| `[extractor]` | provider/endpoint/model_name/model_id/model_version/api_key/timeout/temperature/max_facts_per_thought/system_prompt_file | **Renamed to `[tagger]`**; `model_version` resets to 1; `max_facts_per_thought` removed |
| `[reflector]` | enabled/schedule/scope_filter/max_thoughts_per_run/max_facts_per_thought/review_queue_below/min_confidence_to_store/subsumption_keep | **Removed**; tag drainer is always-on (no opt-in); no confidence-band routing (tags don't have confidence) |

The `[reranker]` config is the same. Hybrid retrieval and the rerank stage are untouched.

## Task graph

Eight agents, five waves. Each agent works in an isolated git worktree (`isolation: "worktree"` on dispatch) and produces a focused diff. Wave boundaries are sequential merge points where the coordinator integrates parallel work before the next wave starts.

```
Wave 1: SPEC (1 agent, sequential)
   │
   ▼
Wave 2: CORE (1 agent)
   │
   ▼
Wave 3: STORAGE, MIGRATION, EXTRACT (3 agents, parallel)
   │
   ▼
Wave 4: MCP (1 agent)
   │
   ▼
Wave 5: CLI, DOCS (2 agents, parallel) → INTEGRATE (coordinator)
```

### Wave 1 — SPEC (1 agent, sequential, ~1 hour)

**Owner:** SPEC agent (or human if preferred).

**Goal:** lock the contract every downstream agent works against. Output: a sub-document at `docs/milestones/m4-spec.md` containing the final Thought struct, Tags struct, Tagger trait, JSON schema, prompt v1, MCP request/response shapes, and migration SQL.

**Inputs:** this milestone doc.

**Deliverables:**
- `docs/milestones/m4-spec.md`: written reference for all parallel agents.
- No code changes.

**Acceptance:** the spec is concrete enough that two downstream agents reading it independently would write compatible code.

### Wave 2 — CORE (1 agent, sequential, ~2 hours)

**Owner:** CORE agent.

**Goal:** establish the kengram-core type contract.

**Inputs:** SPEC doc.

**Files changed:**
- `crates/kengram-core/src/thought.rs`: add fields per spec.
- `crates/kengram-core/src/tags.rs` (NEW): `Tags`, `TagKind` types with serde + tests.
- `crates/kengram-core/src/lib.rs`: re-exports.
- `crates/kengram-core/src/extractor.rs` → repurpose to `tagger.rs`: drop `ExtractedFact`, `ExtractionContext`, `ExtractMode`, `Extractor` trait; add `Tagger` trait + `TaggerError`.
- `crates/kengram-core/src/fact.rs`: **delete entirely**.

**Tests:** Tags serde roundtrip; default Tags is `{}`-equivalent; TagKind enum lowercases; Thought struct serde roundtrip with full + minimal tags.

**Acceptance:** `cargo build -p kengram-core` clean; `cargo test -p kengram-core` clean; `cargo clippy -p kengram-core` clean. Other crates won't build yet — that's expected.

### Wave 3 — STORAGE, MIGRATION, EXTRACT (3 agents, parallel, ~3 hours each)

#### Wave 3a — STORAGE

**Owner:** STORAGE agent.

**Goal:** rewrite kengram-storage against the new Thought shape.

**Files changed:**
- `crates/kengram-storage/src/lib.rs`:
  - **Delete:** `NewFact`, `FactHit`, all fact functions (`insert_fact`, `fetch_fact`, `supersede_fact`, `find_matching_active_facts`, `find_matching_superseded_facts`, `find_subsuming_active_facts`, `list_active_facts_for_thought`, `search_facts_trigram`, `search_facts_vector_knn`, `enqueue_unembedded_facts`, `insert_fact_embedding`), `RunId`, `start_run`, `finish_run`, all `reflector_runs` queries, `NewReviewRow`, `insert_review_queue_row`, `fact_from_columns`, `FactVectorSearchRow`.
  - **Modify `NewThought`**: add `content_fingerprint: [u8; 32]`. (Tags are not provided at capture time; they're written by the tagger drainer later.)
  - **Modify `insert_thought`**: compute fingerprint, INSERT with ON CONFLICT (content_fingerprint) returning either new id + `is_new=true` or existing id + `is_new=false`.
  - **Modify `retract_thought`**: drop the fact-cascade UPDATE. Just set `thoughts.retracted_at` + reason. Update `RetractThoughtOutcome` to drop `facts_superseded` field.
  - **Add `update_thought_tags(pool, thought_id, tags, model_id, version)`**: writes tags JSONB + provenance columns; updates `updated_at`. (No supersede semantics — tags are overwriteable.)
  - **Add `enqueue_tag_job(pool, thought_id, tagger_model_id) -> ()`**: inserts into a new `pending_tags` table (created by MIGRATION).
  - **Add `fetch_thought_tags(pool, thought_id) -> Option<Tags>`** for `get_thought`.
  - **Update all SELECT paths** on `thoughts`: include the new columns (content_fingerprint, tags, tags_extractor_*).
- All `NewThought { ... }` construction sites in tests get the new fingerprint field (compute via `digest(content, 'sha256')`).

**Tests:**
- `insert_thought_returns_existing_id_on_duplicate_content_fingerprint`
- `insert_thought_with_distinct_content_returns_distinct_ids`
- `update_thought_tags_persists_jsonb_and_provenance`
- `retract_thought_drops_fact_cascade_assertion` (replaces the prior cascade test)
- `enqueue_tag_job_inserts_into_pending_tags`
- All existing thought-search tests continue to pass; their fixtures may need to recompute fingerprints.

**Acceptance:** `cargo build -p kengram-storage` clean; `cargo test -p kengram-storage` clean (subset of M3 tests survives + 4-5 new tests); clippy clean.

#### Wave 3b — MIGRATION

**Owner:** MIGRATION agent.

**Goal:** write the SQL migration + verify it runs against a snapshot of the current local DB.

**Files changed:**
- `migrations/0006_collapse_to_thoughts.sql` (NEW): the SQL from the "Schema" section above.
- (No Rust changes — sqlx picks up the migration on next build.)

**Verification:**
- Apply against a fresh `kengram` test database after `kengram migrate` has applied 0001-0005.
- `\d thoughts` shows the new columns + index + unique constraint.
- `\dt` confirms `facts`, `facts_review_queue`, `reflector_runs` are gone.
- `SELECT COUNT(*) FROM embeddings WHERE target_kind = 'fact'` returns 0.

**Acceptance:** clean apply; idempotent re-run is a no-op (or fails clearly without partial corruption).

#### Wave 3c — EXTRACT (repurposed to TAGGER)

**Owner:** EXTRACT agent.

**Goal:** turn kengram-extract into the tagger module.

**Files changed:**
- `crates/kengram-extract/src/lib.rs`: re-exports for new types.
- `crates/kengram-extract/src/openai_compatible.rs`:
  - Replace `OpenAICompatibleExtractor` → `OpenAICompatibleTagger`.
  - Replace `OpenAICompatibleConfig`'s `max_facts_per_thought` with no replacement.
  - Replace `BUNDLED_SYSTEM_PROMPT` with the tagger prompt (above).
  - Replace `facts_response_format()` with `tags_response_format()` per the JSON schema.
  - `tag()` method: POST to chat-completions with the new prompt + schema; deserialize into `Tags`.
- `crates/kengram-extract/src/fake_extractor.rs` → `fake_tagger.rs`: `FakeTagger` returns canned `Tags` from a deterministic mapping or operator-set behavior.
- Tests in this crate update to match.

**Tests:**
- `valid_response_parses_to_tags` (wiremock).
- `malformed_response_returns_malformed_error`.
- `timeout_returns_transient_error`.
- `tagger_v1_prompt_contains_field_semantics_section` (regression pin).
- FakeTagger tests for the determinism modes (`Empty`, `Canned(Tags)`, `Substring(map<&str, Tags>)`).

**Acceptance:** `cargo build -p kengram-extract` clean; tests pass; clippy clean.

### Wave 4 — MCP (1 agent, sequential after Wave 3, ~4 hours)

**Owner:** MCP agent.

**Goal:** rewrite kengram-mcp against the new kengram-storage + kengram-extract.

**Files changed:**
- `crates/kengram-mcp/src/reflect.rs`: **delete entirely**.
- `crates/kengram-mcp/src/correct.rs`: **delete entirely**.
- `crates/kengram-mcp/src/search.rs`:
  - **Delete:** `search_facts`, `SearchFactHit`, `SearchFactsRequest`, `SearchFactsResponse`, `rrf_fuse_facts`, `apply_rerank_to_fact_hits`, all fact-related code.
  - **Modify `search_thoughts`**: add `tag_filter` field; thread into `search_vector_knn` + `search_trigram` via a new WHERE clause `AND tags @> $tag_filter`.
  - **Modify `SearchHit`**: add `tags: Tags` field.
- `crates/kengram-mcp/src/server.rs`:
  - **Delete:** `SearchFactsArgs`, `CorrectFactArgs`, `CorrectFactReplacementArgs`, `search_facts_response_json`, `search_facts` + `correct_fact` tool handlers.
  - **Modify `SearchThoughtsArgs`**: add `tag_filter: Option<serde_json::Value>` with schemars description.
  - **Modify `search_thoughts_response_json`**: emit `tags` per hit.
  - **Modify `get_thought_response_json`**: drop `linked_facts` from `provenance`; emit `tags` + tag provenance.
  - **Modify `KengramServer::new`**: drop the reranker-only signature confusion — wait, reranker is unchanged; it's just the reflector/extractor wiring that goes. KengramServer no longer holds an `Arc<dyn Extractor>` — that lives only in the worker now.
- `crates/kengram-mcp/src/capture.rs`:
  - Compute SHA-256 of content; pass to NewThought; handle `is_new` in the response (`is_duplicate: !is_new`).
  - On successful insert, enqueue both embedding job and tag job.
- `crates/kengram-mcp/src/retract.rs`: drop the fact-cascade reporting.
- `crates/kengram-mcp/src/drain.rs`:
  - Split into two drainer functions: `drain_pending_embeddings` (unchanged shape) and `drain_pending_tags` (NEW).
  - Tag drainer fetches the thought, calls `tagger.tag(content)`, calls `update_thought_tags`. Soft-fail per the embed-drainer's Q9 pattern.
- `crates/kengram-mcp/src/backfill.rs`: drop the fact-target arm; embed-backfill is thoughts-only.
- `crates/kengram-mcp/src/lib.rs`: update re-exports.

**Tests:**
- `search_thoughts_filters_by_tag_containment`
- `search_thoughts_response_carries_tags`
- `capture_returns_existing_id_on_duplicate_content` (and `is_duplicate: true`)
- `capture_enqueues_both_embedding_and_tag_jobs_on_new_insert`
- `capture_does_not_enqueue_jobs_on_duplicate_insert`
- `drain_pending_tags_updates_thought_tags_and_provenance`
- `retract_thought_no_longer_reports_facts_superseded`
- `get_thought_response_carries_tags_and_provenance`
- Delete all reflect / correct / search_facts tests.

**Acceptance:** `cargo build -p kengram-mcp` clean; tests pass (test count substantially lower than M3's 105); clippy clean.

### Wave 5a — CLI (1 agent, parallel with DOCS, ~3 hours)

**Owner:** CLI agent.

**Goal:** rewrite kengram-cli against the new MCP surface.

**Files changed:**
- `crates/kengram-cli/src/main.rs`:
  - `Command::Reflect { ... }` → `Command::Tag { scope, limit, rerun, since }`. Subcommand semantics: like `kengram reflect` but for the tagger.
  - `run_reflect` → `run_tag`: builds the tagger (via `build_tagger(&config.tagger)`), iterates thoughts (unfacted-or-rerun semantics adapted to "untagged-or-rerun"), calls `tagger.tag()` per thought, writes via `update_thought_tags`.
  - Worker: drop reflector cron. Just runs embed-drainer + tag-drainer in tick loops.
  - `run_embed_backfill`: drop the `--target` flag (thoughts only).
- `crates/kengram-cli/src/config.rs`:
  - **Rename `ExtractorConfig` → `TaggerConfig`**: drop `max_facts_per_thought`. Reset `model_version` default to 1.
  - **Delete `ReflectorOptions`/`ReflectorConfig`** entirely. Tag drainer is always-on with no cron (drains on every worker tick).
  - Config struct: drop `reflector` field; rename `extractor` to `tagger`.
- `crates/kengram-cli/src/bench.rs`:
  - Drop `BenchTarget` enum.
  - `BenchQuery`: drop `target` field; all queries are thought queries.
  - `run_pair`: drop the fact-target dispatch.
- `tests/fixtures/bench-rerank.example.json`: drop the `target` field from each entry; convert any `target: "facts"` examples to thought-shaped examples.

**Tests:**
- `tagger_config_loads_from_toml` (regression for the rename).
- `bench_query_parses_without_target_field`.
- Bench harness tests: update to drop target dispatch.
- Delete reflect-related CLI tests.

**Acceptance:** `cargo build -p kengram-cli` clean; `kengram --help` shows the new subcommand set; tests pass; clippy clean.

### Wave 5b — DOCS (1 agent, parallel with CLI, ~3 hours)

**Owner:** DOCS agent.

**Goal:** bring the operator-facing docs and design narratives in line with the new architecture.

**Files changed:**
- `README.md`:
  - Replace `## How fact extraction works` with `## How tagging works` (much shorter — describes the tagger output, content_fingerprint dedup, retrieval-with-tag-filter).
  - Update `## What you get (MCP surface)` table: drop `search_facts` + `correct_fact` rows; update `capture` row (mentions fingerprint dedup); update `search_thoughts` row (mentions tag_filter); update `retract_thought` row (drops fact-cascade detail).
  - Update `## Configuring the extractor backend (M2+)` → `## Configuring the tagger backend` with shorter content reflecting the new prompt + schema.
  - Update `## Reranking search results`: confirm it still applies; the section is largely unchanged.
  - Update `## Configuration reference`: rename `[extractor]` block to `[tagger]`; drop `[reflector]` block; update field tables.
  - Update Status line + Roadmap.
- `DEVELOPMENT.md`:
  - Drop `kengram reflect` examples; add `kengram tag` examples.
  - Drop `[extractor]`/`[reflector]` config blocks; add `[tagger]` block.
  - Drop "Tier 2" reflector setup section; add "Tagger backend setup" (short).
- `DESIGN.md`: major revision. The "facts pipeline" section (§6 and §10) gets rewritten as "tagging sidecar." Decisions reset: no SPO, no confidence routing, no review queue. Content_fingerprint dedup added.
- `docs/milestones/m3-progress.md`: close-out entry — "M3 retrieval improvements shipped; extraction work produced negative knowledge that motivates M4."
- `docs/milestones/m3-search-quality.md`: status flipped to ✅ (retrieval portion) with a final paragraph linking to M4 for the extraction-side outcome.
- `docs/milestones/m4-collapse-to-thoughts.md`: this file. Progress section added (mirroring m3-progress.md format).
- `docs/milestones/m4-artifacts.md` → **rename to `m5-artifacts.md`** (artifact ingestion moves out by one).
- `docs/milestones/m5-operational-maturity.md` → **rename to `m6-operational-maturity.md`**.
- `scripts/bench-rerank.md`: drop fact-target instructions; update fixture authoring section.

**Acceptance:** docs build cleanly (no broken cross-references); README and DEVELOPMENT.md walk through a coherent end-to-end story; the design doc no longer references SPO or confidence routing.

### Wave 6 — INTEGRATE (coordinator, sequential, ~2 hours)

**Owner:** coordinator (human or one final agent).

**Goal:** assemble all parallel work; resolve conflicts; verify end-to-end.

**Steps:**
1. Merge Wave 3 worktrees (STORAGE, MIGRATION, EXTRACT) onto Wave 2 (CORE) branch.
2. Merge Wave 4 (MCP) on top.
3. Merge Wave 5a (CLI) and Wave 5b (DOCS) on top.
4. Resolve conflicts (mostly in `Cargo.toml`, `Cargo.lock`, and lib re-export sites).
5. Run full workspace: `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --all-targets -- -D warnings`.
6. Apply migration against the local DB: `kengram migrate`.
7. Smoke-test:
   - `kengram serve` starts; `kengram worker` starts.
   - Capture a thought via Claude Desktop / MCP inspector.
   - Confirm `thoughts.tags` is `{}` initially, becomes populated within a worker tick.
   - Capture the same content again — same `thought_id` returned, no new row.
   - `search_thoughts` returns results with `tags` field populated.
   - `search_thoughts` with `tag_filter: {"kind": "task"}` filters correctly.
   - `kengram reflect --rerun` is gone; `kengram tag --rerun` works.
   - `kengram bench rerank --corpus tests/fixtures/bench-rerank.example.json` runs (with no `target` field) and prints the table.
8. Confirm no orphaned references: `git grep -i "search_facts\|correct_fact\|reflector\|ExtractedFact\|reflect.rs\|correct.rs"` returns nothing in code (only legitimate hits in m3 archive docs).

**Acceptance:** all of the above. Then commit as a single bundled commit `M4: collapse to thoughts-only with metadata-tagging sidecar (Path B-OB1)`. Body covers what M4 shipped, what M3 close-out looked like, the test count delta, and the architectural narrative.

## Risks and mitigations

- **Foreign key ordering in migration.** The migration drops three tables with cross-references (facts → facts_review_queue? facts → reflector_runs?). Use `CASCADE` on the DROPs and order them from leaves to roots. MIGRATION agent validates against an existing DB before merging.
- **Lossy data**: existing facts (Ron's dogfood corpus) are deleted by the migration. Operator-acceptable per the Path B decision (2026-05-16). If hedge desired, MIGRATION agent can `pg_dump facts > /tmp/facts-snapshot-pre-m4.sql` as a pre-migration step. Recommendation: take the snapshot, never restore.
- **Existing thoughts arrive untagged**: `content_fingerprint` is backfilled in the migration via `digest(content, 'sha256')`; `tags` defaults to `{}`. Tags get populated by `kengram tag --rerun --since <epoch>` as a one-time operator step post-migration.
- **Concurrent agent edits to shared files**: Cargo.toml at workspace level changes minimally (only if a crate is added/removed; we're not adding any). Each agent's worktree contains only their crate's edits. INTEGRATE handles Cargo.lock by regenerating (`cargo update -p kengram-core` etc.).
- **MCP wire-shape change is breaking**: `search_facts` and `correct_fact` tools disappear. Any in-flight MCP client that calls them errors with "unknown tool." Acceptable — kEngram is single-user, single-operator; Ron updates his MCP client configs.
- **Test count drops sharply**: M3 ended at 307 tests; M4 will land closer to ~150-180. The dropped tests were exercising removed code paths, not real regressions. Coordinator confirms no orphan removals (every deleted test maps to a removed function).
- **Tagger output quality**: the tagger is a new prompt against the same local model that struggled with SPO. Mitigation: tags are advisory metadata, not load-bearing. A wrong tag on a thought that you'd find by content anyway is low-impact. The dogfood signal for the tagger should be much cleaner than for the extractor was — but the same model brittleness exists.

## Verification (final acceptance criteria for M4)

1. `cargo build --workspace` clean.
2. `cargo test --workspace` clean (~150-180 tests).
3. `cargo clippy --all-targets -- -D warnings` clean.
4. `kengram migrate` applies cleanly against the current state of the DB.
5. End-to-end smoke (steps in Wave 6 step 7).
6. `git grep` audit confirms no code references to the removed entities.
7. README + DEVELOPMENT.md + DESIGN.md tell a consistent story.
8. **Single bundled commit** `M4: collapse to thoughts-only with metadata-tagging sidecar (Path B-OB1)` on `main`, with body summarizing the work + the architectural narrative.

## Out of scope (deferred)

- **Backfill existing thoughts' tags**: post-merge operator step (`kengram tag --rerun --since 1970-01-01T00:00:00Z` or similar). Not part of the M4 commit.
- **Tagger quality dogfood loop**: M4 ships the new pipeline; dogfood validation comes after, as a new milestone-D-shaped phase. Don't pre-iterate.
- **Schema-extension for richer tag semantics** (relations between tags, hierarchical topics, etc.): M5+. M4 ships the OB1-equivalent shape; richer comes later if dogfood demands.
- **Removing the `target_kind` enum value `'fact'`**: defer indefinitely. Leaving the value in lets us add facts back without a schema change if Path B ever proves insufficient.
- **Original M4 (artifacts) → M5; original M5 (operational maturity) → M6**: rename in DOCS. Renumbering is bookkeeping, not work.

## Open questions to settle in Wave 1 (SPEC)

1. **Tag drainer cadence**: same `[worker]` knobs as embed drainer, or separate `[tagger] tick_interval_seconds`? Default: same knobs, cheapest path.
2. **Tagger backfill UX**: `kengram tag --rerun --since <epoch>` for "tag everything that's untagged" — same shape as `kengram reflect --rerun` had. Confirm.
3. **`is_duplicate` field on capture response**: surface explicitly, or just return the existing thought_id silently? Recommendation: surface — agents may want to handle "I already captured this" differently from "freshly captured."
4. **Tag schema field names**: `people` / `action_items` / `topics` / `dates_mentioned` / `kind` matches OB1. Any preference to rename? `kind` could be `classification` or `category`; the others are conventional.
5. **Empty `[tagger]` config behavior**: silently disable (no tag drainer runs)? Or refuse to boot? Recommendation: silent-disable, matches `[reranker]` pattern from Phase B.

These are settled by the SPEC agent (or human in lieu of), and the answers go into `docs/milestones/m4-spec.md` for downstream agents to reference.

## Progress

Living checklist tracking M4 implementation; **History** at the bottom captures dated notes per wave merge. Wave 5a (CLI) and Wave 5b (DOCS) run in parallel on isolated worktrees; commit hashes for those waves are filled in once the coordinator merges them.

### Wave 1 — SPEC
- [x] `docs/milestones/m4-spec.md` written; 5 open questions answered; types, JSON schema, prompt v1, migration SQL all locked.

### Wave 2 — CORE (`kengram-core`)
- [x] `Thought` extended with `content_fingerprint: [u8; 32]` + 4 tag-related fields.
- [x] `tags.rs` new module: `Tags` struct + `TagKind` enum with serde + tests.
- [x] `extractor.rs` repurposed to `tagger.rs`: `Tagger` trait + `TaggerError` (with `is_transient`).
- [x] `fact.rs` deleted.
- [x] Re-exports in `lib.rs` updated.
- [x] `cargo build -p kengram-core` / `cargo test -p kengram-core` / clippy clean.

### Wave 3a — STORAGE (`kengram-storage`)
- [x] Drop all fact functions, `RunId`, `start_run`/`finish_run`, review-queue helpers, `NewReviewRow`.
- [x] `NewThought` gains `content_fingerprint`; `insert_thought` returns `(InsertedThought, is_new)` via ON CONFLICT.
- [x] `retract_thought` drops fact-cascade UPDATE.
- [x] New: `update_thought_tags`, `enqueue_tag_job`, `fetch_thought_tags`, `fetch_pending_tag_jobs`, `complete_tag_job`, `increment_tag_job_attempts`, `find_untagged_or_stale_thoughts`.
- [x] All SELECT paths on `thoughts` updated for new columns.
- [x] Tests: dedup-by-fingerprint paths, tag update, queue mechanics.

### Wave 3b — MIGRATION (`migrations/0006_collapse_to_thoughts.sql`)
- [x] DROP `facts_review_queue`, `reflector_runs`, `facts` CASCADE.
- [x] DELETE fact-targeted rows from `embeddings` + `pending_embeddings`.
- [x] ALTER `thoughts` to add `content_fingerprint`, `tags`, `tags_extractor_*`.
- [x] Backfill `content_fingerprint = digest(content, 'sha256')`.
- [x] Lock NOT NULL + UNIQUE on fingerprint; GIN index on `tags`.
- [x] CREATE `pending_tags` queue table.
- [x] Clean apply against fresh DB; idempotent re-run.

### Wave 3c — EXTRACT (`kengram-extract`)
- [x] `OpenAICompatibleExtractor` → `OpenAICompatibleTagger`; new `BUNDLED_TAGGER_PROMPT` + `tags_response_format`.
- [x] `FakeExtractor` → `FakeTagger` with `Empty` / `Canned(Tags)` / `Substring(map)` / soft-fail behaviors.
- [x] Lib re-exports updated; tests rewritten.
- [x] `cargo build -p kengram-extract` / tests / clippy clean.

### Wave 4 — MCP (`kengram-mcp`)
- [x] Delete `reflect.rs`, `correct.rs`, all `search_facts` code in `search.rs`.
- [x] `search_thoughts` gains `tag_filter`; `SearchHit` gains `tags`.
- [x] `capture` computes SHA-256, threads `is_new` → `is_duplicate`; enqueues both embedding and tag jobs on insert.
- [x] `retract_thought` drops fact-cascade reporting.
- [x] `drain.rs` split into embed-drainer + tag-drainer.
- [x] `backfill.rs` drops fact-target arm (thoughts-only).
- [x] Server tool wiring: drop `search_facts` / `correct_fact` tools; add `tag_filter` arg + `tags` in response shapes.
- [x] `cargo build -p kengram-mcp` / tests / clippy clean.

### Wave 5a — CLI (`kengram-cli`)
- [ ] `Command::Reflect` → `Command::Tag`; `run_reflect` → `run_tag` against tagger.
- [ ] Worker drops reflector cron; runs embed + tag drainers in tick loops.
- [ ] `embed-backfill` drops `--target` flag.
- [ ] Config: rename `ExtractorConfig` → `TaggerConfig` (drop `max_facts_per_thought`); delete `ReflectorConfig`.
- [ ] Bench harness: drop `BenchTarget`, fact-target dispatch; fixture-JSON drops `target` field.
- [ ] Tests + clippy clean.

### Wave 5b — DOCS
- [x] `README.md`: "How fact extraction works" → "How tagging works"; MCP surface table updated; `[extractor]`/`[reflector]` config replaced with `[tagger]`; status + roadmap reflect M4-shipped, M5/M6 planned.
- [x] `DEVELOPMENT.md`: `kengram reflect` examples → `kengram tag`; `[tagger]` config block; tagger backend setup notes.
- [x] `DESIGN.md`: §6 + §10 rewritten as tagging sidecar + operational shape; §5 schema rewritten; SPO / confidence-routing / supersede-facts / correct-fact / dedup-via-supersession discussion all dropped.
- [x] `docs/milestones/m3-progress.md` + `m3-search-quality.md`: M3 ✅ for retrieval; extraction-side close-out forward-references M4.
- [x] `docs/milestones/m4-collapse-to-thoughts.md`: this Progress section.
- [x] Renames: `m4-artifacts.md` → `m5-artifacts.md`; `m5-operational-maturity.md` → `m6-operational-maturity.md` (via `git mv`); cross-references updated.
- [x] `scripts/bench-rerank.md`: fact-target instructions dropped.

### Wave 6 — INTEGRATE
- [x] Wave 5a CLI committed directly to `m4-collapse-to-thoughts` (commit `80842c7`); Wave 5b DOCS merged via `a5e9ba1`.
- [x] Full workspace clean: `cargo build --workspace`, `cargo test --workspace` (234 passing across all 6 crates), `cargo clippy --all-targets -- -D warnings`.
- [x] Migration `0006_collapse_to_thoughts.sql` applied to dev DB during Wave 3b; verified post-merge that `\d thoughts` shows the new schema and the facts tables are gone.
- [x] `git grep` audit clean — only 3 hits, all in source-file doc-comments framing the M4 removal historically (no live code references).
- [x] CLI surface verified: `kengram --help` shows `tag` (replaces `reflect`); `embed-backfill` has no `--target` flag; `worker` description mentions both drainers; tag drainer silent-disables when `[tagger]` provider empty.
- [x] M4 ships as a sequence of wave commits on `m4-collapse-to-thoughts` rather than a single squashed commit — the wave-by-wave audit trail captures the parallel-team-of-agents execution shape and is more useful than a flat squash. Final close-out commit on the branch documents the integration.

## History

Format: `YYYY-MM-DD — <one-line summary>`. Multi-line entries fine for decisions that need explanation. **Most recent entry first.**

- **2026-05-16** — **M4 sealed.** Wave 6 INTEGRATE: Wave 5a's CLI commit (`80842c7`) was already on `m4-collapse-to-thoughts` (worktree isolation didn't trigger); Wave 5b's DOCS branch merged via `a5e9ba1`. Full workspace verification passed: `cargo build --workspace` clean, `cargo test --workspace` 234 passing across all six crates (engram-cli 10, engram-core 65, engram-embed 37, engram-extract 21, engram-mcp 64, engram-storage 37), `cargo clippy --all-targets -- -D warnings` clean. Orphan-reference audit returned 3 hits — all in source-file doc-comments framing the M4 removal historically, no live code references. CLI surface verified: `tag` subcommand replaces `reflect`; `embed-backfill` has no `--target` flag; `worker` drains both queues with tag-drainer silent-disabled when `[tagger]` provider is empty. The facts pipeline is gone end-to-end. Operator workflow now: capture → embed + tag (async, via worker) → search (hybrid + rerank + optional tag_filter) → retract when wrong; dedup is content-fingerprint-based at capture time. M4 ships as a wave-by-wave audit trail rather than a squashed commit — `git log m4-collapse-to-thoughts ^main` shows the team-of-agents execution shape.
- **2026-05-16** — **Wave 5a CLI landed.** Commit `80842c7`. `engram-cli` refactor: `Command::Reflect` → `Command::Tag` (same flag shape, new semantics — walks `tags_extractor_version IS NULL` or stale-on-rerun); worker drops reflector cron, adds tag drainer alongside embed drainer on every tick (tag drainer silent-disabled when `[tagger]` provider is empty); `embed-backfill` drops `--target` flag; `ExtractorConfig` → `TaggerConfig` (rename + drops `max_facts_per_thought` + resets `model_version` default to 1); `ReflectorOptions` / `ReflectorConfig` deleted entirely; `BenchTarget` enum and `target` field on `BenchQuery` dropped (bench is thoughts-only); `tests/fixtures/bench-rerank.example.json` updated. `tokio-cron-scheduler` workspace dep dropped (was only used by the reflector cron). New `ResolvedTagger` struct on the CLI side (clippy's `type_complexity` lint nudged this over a bare tuple). Workspace builds + tests + clippy clean for the first time since Wave 2.
- **2026-05-16** — **Wave 5b DOCS landed (this commit).** Major documentation overhaul: README's "How fact extraction works" section replaced with shorter "How tagging works"; MCP-surface table updated (dropped `search_facts` / `correct_fact` rows; `capture` row now mentions content-fingerprint dedup + `is_duplicate`; `search_thoughts` row now mentions `tag_filter` + per-hit `tags`; `retract_thought` simplified). "Configuring the extractor backend" → "Configuring the tagger backend." `[extractor]` and `[reflector]` config blocks replaced with `[tagger]` everywhere. Roadmap table renumbered: M2 marked superseded-by-M4; M3 ✅ retrieval; **M4 ✅ collapse to thoughts-only**; M5 ⏳ artifacts (was M4); M6 ⏳ operational maturity (was M5). `DEVELOPMENT.md`: `engram reflect` examples replaced with `engram tag`; reflect-rerun runbook block dropped; `--target` flag for embed-backfill dropped (thoughts-only). `DESIGN.md`: §6 rewritten as "Ingest path + Tagging sidecar"; §10 rewritten as "Operational shape — what makes the store honest" (operational guarantees, no drift-defense ceremony); §5 schema rewritten; §8 MCP surface table updated; §9 traits updated (`Tagger` replaces `Extractor`); §11 / §12 / §13 milestone callouts updated for the M5→M6 shift; revision history extended with the M4 entry. Renames: `m4-artifacts.md` → `m5-artifacts.md`; `m5-operational-maturity.md` → `m6-operational-maturity.md` via `git mv`; cross-references inside renamed files updated (M4→M5 / M5→M6). `m3-progress.md` + `m3-search-quality.md` final close-out: M3 status ✅ retrieval; extraction findings forward-reference M4. `scripts/bench-rerank.md`: dropped `target: "facts"` instructions; thoughts-only authoring guidance. Doc-only commit; no code touched.
- **2026-05-16** — **Wave 4 MCP landed.** `crates/engram-mcp` simplified for thoughts-only: `reflect.rs` + `correct.rs` deleted; `search.rs` drops all `search_facts` / `SearchFactHit` / `rrf_fuse_facts` code and gains `tag_filter` threading; `SearchHit` gains `tags`. `capture.rs` computes SHA-256 of content, threads `is_new` → `is_duplicate`, and enqueues both embedding and tag jobs on a fresh insert. `retract.rs` drops the fact-cascade reporting (no `facts_superseded` field on response). `drain.rs` split into `drain_pending_embeddings` (unchanged shape) + `drain_pending_tags` (NEW). `backfill.rs` drops the fact-target arm. `server.rs` tool wiring drops `search_facts` + `correct_fact`; `SearchThoughtsArgs` gains `tag_filter`; both `search_thoughts_response_json` and `get_thought_response_json` emit `tags` + tag provenance. Test count drops substantially from M3's 105. Commit `358f86b`.
- **2026-05-16** — **Wave 3c EXTRACT landed (via merge `2000059`).** `engram-extract` repurposed: `OpenAICompatibleExtractor` → `OpenAICompatibleTagger`; bundled prompt replaced with the v1 tagger prompt; `facts_response_format` → `tags_response_format` per the locked JSON schema. `FakeExtractor` → `FakeTagger` with `Empty` / `Canned(Tags)` / `Substring(map)` / soft-fail behaviors. Lib re-exports updated. Commit `d7892ac`.
- **2026-05-16** — **Wave 3b MIGRATION landed.** `migrations/0006_collapse_to_thoughts.sql`: drops `facts_review_queue`, `reflector_runs`, `facts` CASCADE; deletes fact-targeted rows from `embeddings` + `pending_embeddings`; extends `thoughts` with `content_fingerprint`, `tags`, `tags_extractor_*`; backfills fingerprint via `digest(content, 'sha256')` (requires pgcrypto, already enabled); locks NOT NULL + UNIQUE on fingerprint; creates GIN index on `tags`; creates `pending_tags` queue table. Verified clean against snapshot of current dev DB; idempotent re-run check passed. Commit `ca2fcd0`.
- **2026-05-16** — **Wave 3a STORAGE landed.** `engram-storage` rewritten for thoughts-only: all fact storage functions dropped; `RunId` / `start_run` / `finish_run` / review-queue helpers gone; `NewThought` gains `content_fingerprint`; `insert_thought` returns `(InsertedThought, is_new)` via ON CONFLICT; `retract_thought` drops the fact-cascade UPDATE. New: `update_thought_tags`, `enqueue_tag_job`, `fetch_thought_tags`, `fetch_pending_tag_jobs`, `complete_tag_job`, `increment_tag_job_attempts`, `find_untagged_or_stale_thoughts`. Commit `4a93571`.
- **2026-05-16** — **Wave 2 CORE landed.** `engram-core` type contract for thoughts-only: `Thought` extended with `content_fingerprint: [u8; 32]` plus 4 tag-related fields; new `tags.rs` module with `Tags` + `TagKind`; `extractor.rs` repurposed to `tagger.rs` (`Tagger` trait + `TaggerError`); `fact.rs` deleted; re-exports updated. Commit `b7a22c4`.
- **2026-05-16** — **Wave 1 SPEC landed.** `docs/milestones/m4-spec.md` written: 5 open questions answered (drainer cadence reuses `[worker]` knobs; backfill UX is `engram tag --rerun`; `is_duplicate` surfaced; tag fields match OB1; empty `[tagger]` silent-disables); types + JSON schema + v1 prompt + migration SQL + MCP wire shapes + storage function signatures locked as the prescriptive contract for downstream waves. Commit `34ba756`.

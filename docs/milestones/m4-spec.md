# M4 spec — locked contract for downstream waves

Reference: [`m4-collapse-to-thoughts.md`](./m4-collapse-to-thoughts.md) for the architectural narrative; this file is the prescriptive contract that Waves 2-5 work against. Locked 2026-05-16; changes require coordinator approval.

## Decisions on the 5 open questions (from milestone doc)

### Q1 — Tag drainer cadence

**Decided: reuse `[worker]` knobs.** Both `pending_embeddings` and `pending_tags` are drained on every `[worker] tick_interval_seconds` tick, each batched at `[worker] batch_size`. No separate `[tagger] tick_interval_seconds`. Justification: agents capturing thoughts shouldn't need two cadence configs to reason about; they care about "how fresh is my tagged-and-embedded memory," which is one number.

### Q2 — Tagger backfill UX

**Decided: `kengram tag --rerun [--since RFC3339] [--scope X] [--limit N]`.** Mirrors `kengram reflect --rerun`'s surface from M3 (same operator muscle memory). Default behavior without `--rerun`: tag thoughts that are currently untagged (`tags_extractor_version IS NULL`). With `--rerun`: re-tag thoughts whose `tags_extractor_version < current_tagger_version` (i.e. tagger prompt has changed). `--since` filters by `thoughts.created_at`. `--scope` filters by scope. `--limit` caps per-run.

### Q3 — `is_duplicate` field on capture response

**Decided: surface explicitly.** `CaptureResponse` gains `is_duplicate: bool`. Returns `true` when the inserted fingerprint conflicted with an existing row (returning the existing `thought_id`); `false` when a new row was inserted. Agents may want different behavior on duplicate (e.g., "you already told me that").

### Q4 — Tag schema field names

**Decided: match OB1 verbatim.** Five fields:
- `people: Vec<String>` — bare names of people mentioned.
- `action_items: Vec<String>` — short imperative phrases for tasks the thought commits to or implies.
- `topics: Vec<String>` — 1-3 short lowercase tag-like topics.
- `dates_mentioned: Vec<String>` — free-text dates/temporal references appearing in the prose.
- `kind: Option<TagKind>` — single classification (see `TagKind` enum below).

No extensions beyond OB1 in v1. If dogfood shows the need for `urls`, `code_snippets_mentioned`, `mentioned_files`, etc., they get added in a later v2 of the tagger; the JSONB shape can accept new keys without a schema change.

### Q5 — Empty `[tagger]` config behavior

**Decided: silent-disable at capture time.** When `[tagger].provider` is empty (or the section is missing), the tag-job enqueue at capture is a no-op. No rows go into `pending_tags`. The tag drainer task in `kengram worker` doesn't spawn. Thoughts capture cleanly, embed normally, search normally — they just stay with `tags = '{}'` forever. Matches the `[reranker]` silent-disable pattern from Phase B. Operator can flip `[tagger].provider = "openai-compatible"` later and run `kengram tag --rerun --since 1970-01-01T00:00:00Z` to tag the backlog.

## Final types (kengram-core)

### `Thought` struct (extended; existing fields unchanged)

```rust
pub struct Thought {
    pub id: ThoughtId,
    pub scope: Scope,
    pub content: String,
    pub source: Source,
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,                          // client-provided (unchanged)
    pub content_fingerprint: [u8; 32],               // SHA-256 of content; NEW
    pub tags: Tags,                                  // LLM-extracted sidecar; NEW
    pub tags_extractor_model: Option<String>,        // None until first tag pass; NEW
    pub tags_extractor_version: Option<i32>,         // None until first tag pass; NEW
    pub tags_extracted_at: Option<OffsetDateTime>,   // None until first tag pass; NEW
}
```

Serde: `content_fingerprint` serializes as a lowercase 64-char hex string; deserializes from either hex or base64. Implementation note: a small `with` module on the field, similar to `time::serde::rfc3339`.

### `Tags` struct + `TagKind` enum

```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tags {
    #[serde(default)]
    pub people: Vec<String>,
    #[serde(default)]
    pub action_items: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub dates_mentioned: Vec<String>,
    #[serde(default)]
    pub kind: Option<TagKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagKind {
    Observation,
    Task,
    Idea,
    Reference,
    PersonNote,    // serializes as "person_note"
    Session,
}
```

`Tags::default()` round-trips as `{}` in JSON (every field is empty/None). The `#[serde(default)]` on each field means deserializing `{}` yields a `Tags` with empty vectors and `kind: None`. New tagger versions can add fields without breaking older readers.

### `Tagger` trait + `TaggerError` (replacing `Extractor` / `ExtractorError`)

```rust
#[async_trait]
pub trait Tagger: Send + Sync {
    fn model_id(&self) -> &str;
    fn version(&self) -> i32;
    async fn tag(&self, thought_content: &str) -> Result<Tags, TaggerError>;
}

#[derive(Debug, thiserror::Error)]
pub enum TaggerError {
    #[error("tagger endpoint unreachable: {0}")]
    Unreachable(String),
    #[error("tagger timed out after {seconds}s")]
    Timeout { seconds: u64 },
    #[error("tagger returned malformed response: {0}")]
    MalformedResponse(String),
    #[error("tagger misconfigured: {0}")]
    Misconfigured(String),
    #[error("tagger backend error (status {status}): {body}")]
    Backend { status: u16, body: String },
}

impl TaggerError {
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Unreachable(_) | Self::Timeout { .. } | Self::Backend { status: 500..=599, .. }
        )
    }
}
```

Same is-transient discipline as `EmbedderError` / `RerankerError` / (defunct) `ExtractorError`. Drainer uses this to decide retry vs skip.

## Tagger JSON schema (sent to LLM via `response_format`)

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "kengram_tags",
    "strict": true,
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["people", "action_items", "topics", "dates_mentioned", "kind"],
      "properties": {
        "people": { "type": "array", "items": { "type": "string" } },
        "action_items": { "type": "array", "items": { "type": "string" } },
        "topics": { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
        "dates_mentioned": { "type": "array", "items": { "type": "string" } },
        "kind": {
          "type": ["string", "null"],
          "enum": ["observation", "task", "idea", "reference", "person_note", "session", null]
        }
      }
    }
  }
}
```

Note: `kind` is nullable in the wire shape (LLM can return `null` if uncertain); deserializes into `Option<TagKind>`. `topics` is capped at 3 items.

## v1 Tagger prompt (`BUNDLED_TAGGER_PROMPT`)

```
You are a tagging assistant. Given a single thought from a memory service, return its metadata tags as JSON.

# Output shape
{ "people": [...], "action_items": [...], "topics": [...], "dates_mentioned": [...], "kind": "..." }

# Field semantics
- people: bare names of people mentioned. Empty array if none.
- action_items: short imperative phrases describing tasks the thought commits to or implies (e.g., "fix the login bug", "review the migration plan"). Empty array if none.
- topics: 1-3 short tag-like topics, lowercase, no punctuation. What is this thought ABOUT at a high level? Examples: "rust", "build-systems", "team-management".
- dates_mentioned: any dates or temporal references appearing in the prose ("next Thursday", "Q3", "2026-05-15", "before the release"). Free-form strings, copied roughly as they appear. Empty array if none.
- kind: a single classification. Use null if uncertain. Categories:
  - observation: a factual claim about the world ("Rust has stronger memory safety than C").
  - task: a thing the writer or someone else needs to do ("fix the login bug").
  - idea: a proposal or hypothesis ("we could use Bloom filters here").
  - reference: a pointer to an external resource (a URL, a paper, a tool).
  - person_note: a fact about a specific person ("Sarah prefers async meetings").
  - session: transient session/test narrative ("the search returned 3 results", "I just ran the migration"). These should also have otherwise-empty arrays.

# Rules
- Only extract what is explicitly present in the thought. Do not infer.
- Empty arrays are correct for any field that has no content.
- One classification only; pick the most-load-bearing category. If genuinely ambiguous, return null.
- This is a tagging pass, not a paraphrase or rewrite. Do not rephrase the thought's content; only emit metadata.
```

Tagger prompt version: starts at `1`. Bumped when the prompt or schema changes such that prior tags shouldn't be considered comparable.

## MCP wire shapes

### `SearchRequest` (modified)

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

`tag_filter` is passed through to the SQL as `WHERE tags @> $tag_filter`. JSON-shape examples that work as filters:
- `{"kind": "task"}` — only thoughts the tagger classified as tasks.
- `{"people": ["Sarah"]}` — only thoughts where the `people` array contains Sarah (JSONB containment on arrays).
- `{"topics": ["rust"], "kind": "idea"}` — combined.

When `tag_filter` is `None` or `{}`, no filter applied.

### `SearchHit` (modified)

```rust
pub struct SearchHit {
    pub thought_id: ThoughtId,
    pub content: String,
    pub scope: Scope,
    pub source: Source,
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,
    pub tags: Tags,                          // NEW
    pub vector_score: Option<f32>,
    pub trigram_score: Option<f32>,
    pub rrf_score: Option<f32>,
    pub rerank_score: Option<f32>,
}
```

JSON serializer emits `tags` as a nested object: `"tags": {"people": [...], "topics": [...], ...}`. Empty `Tags` serializes as `"tags": {"people": [], "action_items": [], "topics": [], "dates_mentioned": [], "kind": null}`.

### `CaptureResponse` (modified)

```rust
pub struct CaptureResponse {
    pub thought_id: ThoughtId,
    pub embedding_status: EmbeddingStatus,
    pub is_duplicate: bool,                  // NEW — true when fingerprint collided
}
```

When `is_duplicate: true`, `embedding_status` reflects the existing row's state (could be `"indexed"` if the original embedded long ago, or `"pending"` if still in the queue). No new embedding/tag jobs are enqueued.

### `GetThoughtResponse` (modified)

```rust
pub struct GetThoughtResponse {
    pub thought: Thought,
    pub embedding_status: EmbeddingStatus,
    pub embedded_at: Option<OffsetDateTime>,
    pub tags: Tags,                          // NEW (moved out of thought.tags for wire-shape clarity)
    pub tags_extractor_model: Option<String>,
    pub tags_extractor_version: Option<i32>,
    pub tags_extracted_at: Option<OffsetDateTime>,
    pub retracted_at: Option<OffsetDateTime>,
    pub retracted_reason: Option<String>,
    // DROPPED: linked_facts (facts no longer exist)
}
```

### `RetractThoughtResponse` (simplified)

```rust
pub struct RetractThoughtResponse {
    pub retracted: bool,
    // DROPPED: facts_superseded (no facts to cascade-supersede)
}
```

## Migration 0006 SQL (`migrations/0006_collapse_to_thoughts.sql`)

```sql
-- M4: collapse facts pipeline to thoughts-only with metadata-tagging sidecar.
-- See docs/milestones/m4-collapse-to-thoughts.md for the architectural rationale.

-- 1. Drop the facts pipeline tables.
DROP TABLE IF EXISTS facts_review_queue CASCADE;
DROP TABLE IF EXISTS reflector_runs CASCADE;
DROP TABLE IF EXISTS facts CASCADE;

-- 2. Clean up fact-targeted rows in shared tables.
DELETE FROM embeddings WHERE target_kind = 'fact';
DELETE FROM pending_embeddings WHERE target_kind = 'fact';

-- 3. Extend thoughts with content-fingerprint dedup + tags sidecar.
ALTER TABLE thoughts
    ADD COLUMN content_fingerprint BYTEA,
    ADD COLUMN tags JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN tags_extractor_model TEXT,
    ADD COLUMN tags_extractor_version INT,
    ADD COLUMN tags_extracted_at TIMESTAMPTZ;

-- 4. Backfill content_fingerprint for existing thoughts.
UPDATE thoughts
SET content_fingerprint = digest(content, 'sha256')
WHERE content_fingerprint IS NULL;

-- 5. Lock content_fingerprint NOT NULL + UNIQUE post-backfill.
ALTER TABLE thoughts
    ALTER COLUMN content_fingerprint SET NOT NULL,
    ADD CONSTRAINT thoughts_content_fingerprint_unique UNIQUE (content_fingerprint);

-- 6. GIN index on tags JSONB for containment queries.
CREATE INDEX thoughts_tags_gin ON thoughts USING gin (tags);

-- 7. Queue table for the tag drainer (mirrors pending_embeddings shape).
CREATE TABLE pending_tags (
    thought_id UUID PRIMARY KEY REFERENCES thoughts(id) ON DELETE CASCADE,
    tagger_model_id TEXT NOT NULL,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    attempts INT NOT NULL DEFAULT 0
);
```

**Notes:**
- `digest(content, 'sha256')` is provided by the `pgcrypto` extension (already enabled in migration 0001).
- The `target_kind = 'fact'` enum value (or CHECK constraint) is **not** removed by this migration. Leaving the value in lets us add the facts table back later without a schema migration if Path B-OB1 ever proves insufficient.
- `pending_tags` uses `thought_id` as primary key (not a separate `id`) so re-enqueueing for the same thought is idempotent (ON CONFLICT (thought_id) DO NOTHING). One pending tag job per thought at a time.

## Storage function signatures (kengram-storage)

```rust
// NewThought gains content_fingerprint; everything else as-is.
pub struct NewThought<'a> {
    pub scope: &'a Scope,
    pub content: &'a str,
    pub source: &'a Source,
    pub metadata: &'a Metadata,
    pub content_fingerprint: [u8; 32],   // NEW; callers compute via sha2::Sha256
}

// Returns (thought_id, is_new). is_new = false means the fingerprint collided
// and the returned id is the pre-existing row.
pub async fn insert_thought(
    pool: &PgPool,
    new: NewThought<'_>,
) -> Result<(InsertedThought, bool), StorageError>;

// Overwrites the tags JSONB + provenance on the thought. Called by the tag drainer.
pub async fn update_thought_tags(
    pool: &PgPool,
    thought_id: ThoughtId,
    tags: &Tags,
    tagger_model_id: &str,
    tagger_version: i32,
) -> Result<(), StorageError>;

// Inserts a row into pending_tags. ON CONFLICT (thought_id) DO NOTHING.
pub async fn enqueue_tag_job(
    pool: &PgPool,
    thought_id: ThoughtId,
    tagger_model_id: &str,
) -> Result<(), StorageError>;

// Read-side: fetches the current tags + provenance for a single thought.
// Used by get_thought.
pub async fn fetch_thought_tags(
    pool: &PgPool,
    thought_id: ThoughtId,
) -> Result<Option<ThoughtTags>, StorageError>;

pub struct ThoughtTags {
    pub tags: Tags,
    pub tagger_model_id: Option<String>,
    pub tagger_version: Option<i32>,
    pub tagged_at: Option<OffsetDateTime>,
}

// Drainer-side: fetch up-to-N pending tag jobs.
pub async fn fetch_pending_tag_jobs(
    pool: &PgPool,
    batch_size: i64,
) -> Result<Vec<PendingTagJob>, StorageError>;

pub struct PendingTagJob {
    pub thought_id: ThoughtId,
    pub tagger_model_id: String,
    pub attempts: i32,
}

// Drainer-side: remove a pending tag job after successful processing.
// (On failure, attempts is incremented; the job stays in the queue.)
pub async fn complete_tag_job(
    pool: &PgPool,
    thought_id: ThoughtId,
) -> Result<(), StorageError>;

pub async fn increment_tag_job_attempts(
    pool: &PgPool,
    thought_id: ThoughtId,
) -> Result<(), StorageError>;

// Updated: drops the fact-cascade UPDATE. Just sets thoughts.retracted_at.
pub async fn retract_thought(
    pool: &PgPool,
    thought_id: ThoughtId,
    reason: Option<&str>,
) -> Result<RetractThoughtOutcome, StorageError>;

pub struct RetractThoughtOutcome {
    pub retracted: bool,
    // DROPPED: facts_superseded (no facts to cascade)
}

// New: walk thoughts whose tags_extractor_version < target_version (rerun) or IS NULL (first-time).
// Returns oldest-first per the find_unfacted_thoughts convention.
pub async fn find_untagged_or_stale_thoughts(
    pool: &PgPool,
    target_tagger_version: i32,
    rerun: bool,                          // true = stale-version included; false = only NULL
    scope: Option<&str>,
    since: Option<OffsetDateTime>,
    limit: i64,
) -> Result<Vec<Thought>, StorageError>;
```

`fetch_thought_with_provenance` (used by `get_thought`) is extended to also return the tag provenance fields. All other SELECT paths on `thoughts` add the five new columns to the SELECT list.

**Functions deleted** (from M3): `NewFact`, `FactHit`, `insert_fact`, `fetch_fact`, `supersede_fact`, `find_matching_active_facts`, `find_matching_superseded_facts`, `find_subsuming_active_facts`, `list_active_facts_for_thought`, `search_facts_trigram`, `search_facts_vector_knn`, `enqueue_unembedded_facts`, `insert_fact_embedding`, `fact_from_columns`, `FactVectorSearchRow`, `RunId`, `start_run`, `finish_run`, `NewReviewRow`, `insert_review_queue_row`.

## CLI subcommand shape

```
kengram tag [--scope X] [--limit N] [--rerun] [--since RFC3339]
```

Semantics (mirrors `kengram reflect` from M3):

- Without `--rerun`: walks `find_untagged_or_stale_thoughts(target_version, rerun=false, ...)` — thoughts where `tags_extractor_version IS NULL`. Tags each one via the configured `[tagger]`. Skips thoughts whose `tags_extractor_version` is already set.
- With `--rerun`: walks `find_untagged_or_stale_thoughts(target_version, rerun=true, ...)` — thoughts where `tags_extractor_version IS NULL OR tags_extractor_version < target_version`. Re-tags any stale ones; overwrites the `tags` column + provenance.
- `--scope X`: restricts to one scope (exact match; same scope-filter semantics as M3 search).
- `--since RFC3339`: restricts to `thoughts.created_at >= since`. Allowed with or without `--rerun` (unlike `kengram reflect --since`, which required `--rerun`).
- `--limit N`: caps how many thoughts to process this run.

Empty/missing `[tagger]` config → errors at startup with a clear message ("`kengram tag` requires a configured `[tagger]` section; see DEVELOPMENT.md"). Same hard-fail shape as `kengram bench rerank` when no reranker.

## `[tagger]` config section

```toml
[tagger]
provider              = "openai-compatible"   # also "openrouter"; "" = disabled
endpoint              = "http://localhost:8000/v1"
model_name            = "qwen2.5-7b-instruct"
model_id              = "vllm/qwen2.5-7b-instruct"
model_version         = 1                     # tagger prompt version; bump on prompt/schema change
api_key               = ""                    # optional bearer token
timeout_seconds       = 60
temperature           = 0.2
# system_prompt_file = "~/.config/kengram/tagger-prompt.txt"
# Optional: replace the bundled v1 prompt. Operator responsible for bumping
# model_version when overriding.
```

Fields dropped from the M3 `[extractor]` section: `max_facts_per_thought` (no per-thought emission cap; tagger always emits one tags object).

## Queue table choice

`pending_tags` (separate queue table) selected over `tags_pending BOOLEAN` (column on thoughts). Rationale:

- Symmetric with `pending_embeddings` — drainer code structure is one shape, not two.
- Idempotent enqueue via ON CONFLICT (thought_id) DO NOTHING.
- Attempt counter for retry-with-backoff observability.
- Capacity to add per-job metadata later (priority, scheduled_at) without further migration.

## Test count target

M3 ended at 307 tests. M4 target: ~160-200. The drop accounts for:
- All fact-side storage tests (estimate ~30 tests)
- reflect.rs tests (estimate ~30 tests)
- correct.rs tests (estimate ~5 tests)
- search_facts tests in mcp/search (estimate ~10 tests)
- search_facts tests in mcp/server (estimate ~5 tests)
- Various other extractor + reflector + facts tests scattered across crates (estimate ~30 tests)

Total estimated deletions: ~110 tests. Net target with new tagger/fingerprint/tag_filter tests: 307 - 110 + 15-25 = ~180.

## File-modification matrix (cheat-sheet for downstream agents)

| Crate / file | Wave | Action |
|---|---|---|
| `crates/kengram-core/src/thought.rs` | 2 (CORE) | Modify (add fields) |
| `crates/kengram-core/src/tags.rs` | 2 (CORE) | New |
| `crates/kengram-core/src/extractor.rs` | 2 (CORE) | Rename to `tagger.rs`; replace contents |
| `crates/kengram-core/src/fact.rs` | 2 (CORE) | **Delete** |
| `crates/kengram-core/src/lib.rs` | 2 (CORE) | Modify re-exports |
| `migrations/0006_collapse_to_thoughts.sql` | 3b (MIGRATION) | New |
| `crates/kengram-storage/src/lib.rs` | 3a (STORAGE) | Major rewrite (drop ~30%, add ~20%) |
| `crates/kengram-extract/src/lib.rs` | 3c (EXTRACT) | Modify re-exports |
| `crates/kengram-extract/src/openai_compatible.rs` | 3c (EXTRACT) | Rewrite to tagger |
| `crates/kengram-extract/src/fake_extractor.rs` | 3c (EXTRACT) | Rename to `fake_tagger.rs`; rewrite |
| `crates/kengram-mcp/src/reflect.rs` | 4 (MCP) | **Delete** |
| `crates/kengram-mcp/src/correct.rs` | 4 (MCP) | **Delete** |
| `crates/kengram-mcp/src/search.rs` | 4 (MCP) | Major simplify (drop fact code, add tag_filter) |
| `crates/kengram-mcp/src/server.rs` | 4 (MCP) | Drop fact tools, add tags serialization |
| `crates/kengram-mcp/src/capture.rs` | 4 (MCP) | Fingerprint dedup + dual enqueue |
| `crates/kengram-mcp/src/retract.rs` | 4 (MCP) | Drop fact-cascade reporting |
| `crates/kengram-mcp/src/drain.rs` | 4 (MCP) | Split into embed + tag drainers |
| `crates/kengram-mcp/src/backfill.rs` | 4 (MCP) | Drop fact target |
| `crates/kengram-mcp/src/lib.rs` | 4 (MCP) | Modify re-exports |
| `crates/kengram-cli/src/main.rs` | 5a (CLI) | Reflect→Tag subcommand; drop reflector cron |
| `crates/kengram-cli/src/config.rs` | 5a (CLI) | Rename [extractor]→[tagger], drop [reflector] |
| `crates/kengram-cli/src/bench.rs` | 5a (CLI) | Drop BenchTarget, fact dispatch |
| `tests/fixtures/bench-rerank.example.json` | 5a (CLI) | Drop `target` field |
| `README.md` | 5b (DOCS) | Major rewrite of extraction sections |
| `DEVELOPMENT.md` | 5b (DOCS) | Drop reflect/extractor, add tag/tagger |
| `DESIGN.md` | 5b (DOCS) | Major revision §6 + §10 |
| `docs/milestones/m3-progress.md` | 5b (DOCS) | Final close-out entry |
| `docs/milestones/m3-search-quality.md` | 5b (DOCS) | Status ✅ on retrieval items |
| `docs/milestones/m4-collapse-to-thoughts.md` | 5b (DOCS) | Add Progress section |
| `docs/milestones/m4-artifacts.md` | 5b (DOCS) | Rename to `m5-artifacts.md` |
| `docs/milestones/m5-operational-maturity.md` | 5b (DOCS) | Rename to `m6-operational-maturity.md` |
| `scripts/bench-rerank.md` | 5b (DOCS) | Drop fact-target instructions |

## Reused patterns (don't reinvent)

- **`Embedder` / `Reranker` HTTP-client construction + reqwest error mapping** in kengram-embed — direct template for `OpenAICompatibleTagger` and `TaggerError`.
- **`pending_embeddings` queue + drainer pattern** in kengram-mcp/drain.rs and kengram-storage — direct template for `pending_tags` + tag drainer.
- **`FakeEmbedder` / `FakeReranker` mock pattern with last-call recording** — direct template for `FakeTagger`.
- **`build_embedder` / `build_reranker`** in kengram-cli/main.rs — template for `build_tagger`.
- **Phase A startup config log** (commit 1d627e4) — template for tagger startup log.
- **Phase C `flagged` JSONB-shaped surface** in `SearchFactHit` — comparable to surfacing `tags` on `SearchHit`.

## Sign-off

This spec is the contract. Downstream waves treat anything in this file as authoritative. Changes require the coordinator (Ron) to approve.

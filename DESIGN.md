# Engram — Local Agent Memory Service

**Status:** Draft v0.1 · for review
**Working name:** Engram (placeholder; trivial to rename)
**Author:** [you]
**Reviewers:** [TBD]
**Last updated:** 2026-05-16

---

## 1. Summary

Engram is a self-hosted, MCP-native memory service for AI agents. It runs alongside vLLM (or equivalent) on a personal headless inference server, reachable from the operator's devices over Tailscale wherever they happen to be. It provides a persistent, model-agnostic backing store that any MCP-capable client (Claude Code, Claude Desktop, opencode, ChatGPT, Cursor, Gemini CLI, custom Rust agents) can read from and write to.

It is OB1's architectural shape — Postgres + pgvector + a thin MCP gateway — implemented as a single Rust binary, with the local vLLM endpoint serving as the embedding and extraction backend, designed so that swapping the underlying embedding or extraction model is a routine operation rather than a migration.

The deployment target is single-user, single-active-session. Concurrent multi-user serving is explicitly not in scope.

The system is built incrementally across six milestones (§3.5). The remainder of this document describes the *terminal* state — all six milestones complete. Inline milestone callouts (e.g. `[M1]`, `[M2+]`) flag features that arrive at a specific milestone. §3.5 is the source of truth for what ships when, and supersedes anything elsewhere in the document that reads as if a feature is "v0."

## 2. Goals

- **Single source of memory** across every agent and model the operator uses.
- **Model-independence** at the storage layer: changing embedding or extraction model must not invalidate captured content.
- **Local-first**: defaults run with no cloud dependency. Cloud is a configurable opt-in per provider.
- **Raw data is permanent, derived signals are recomputable**: thoughts are immutable; embeddings and tags are derived from them and can be regenerated when models or prompts change. The raw capture is the source of truth.
- **Tiered exposure**: localhost / mesh / public, configurable, with auth that scales accordingly.
- **Operationally simple**: single Postgres, single Rust binary, runs under systemd.

## 3. Non-goals

- Not an agent runtime (cf. Letta). Engram stores and retrieves; agents live elsewhere.
- Not a temporal knowledge graph (cf. Graphiti). Facts are timestamped and supersedable, but we do not model validity windows as first-class entities.
- Not a vector database product. We use pgvector and we are happy.
- Not multi-tenant SaaS. Single operator, optional shared with trusted humans.
- No ML training. We use existing embedding / instruct models as black boxes.

## 3.5 Milestone roadmap

The system is built in six capability milestones, preceded by a small environment-setup milestone (M0). Each capability milestone is independently shippable: at the end of M1 the operator has a usable memory service; subsequent milestones add capability without invalidating prior ones.

**M0 — Development environment.** *The floor under the floor.*
- Postgres 16 running in Docker via `docker-compose.yml` at the repo root, using the `pgvector/pgvector:pg16` image (bundles `vector`, `pg_trgm`, `pgcrypto`).
- Ollama (already installed on the operator's box) serves as the dev-mode embedder via its OpenAI-compatible endpoint (`http://localhost:11434/v1/embeddings`, model `bge-m3`). Production retains the TEI sidecar.
- `DEVELOPMENT.md` runbook for first-time setup. No code is written; M0 only ensures M1's code has somewhere to run.

**M1 — Capture and search.** *The floor.*
- Schema ships with `thoughts` and `embeddings`. Migration 0001 also ships `artifacts` / `artifact_chunks` tables for what was then a planned long-form-ingestion milestone; the artifacts plan was dropped in the M6 reshape (see revision history 2026-05-17), and those two tables remain inert pending a future drop migration.
- Sync embedding on `capture` via TEI sidecar (BGE-M3, 1024-dim).
- Hybrid retrieval: vector kNN ∪ trigram lexical search, fused via reciprocal rank fusion (RRF). No reranker.
- Four MCP tools: `capture`, `search_thoughts`, `recent_thoughts`, `get_thought`.
- Single binary; subcommands `serve` and `migrate`. No worker process.
- Tier 0 auth (localhost-only). Tier 1 (Tailnet) is a config change, not a code change.

**M2 — Facts pipeline.** *(Retired in M4; documented here for history.)*
- `engram-extract` crate became real with a vLLM client; `Extractor` trait gained its first implementation.
- Worker process appeared (`engram worker` subcommand). Reflector cron job ran.
- `facts` table populated; new MCP tools `search_facts`, `correct_fact`.
- The async-embedding seam designed at M1 was exercised: `capture` posts a job; the worker computes the embedding.

**M3 — Search quality.**
- Cross-encoder reranker (via TEI) plugged in after RRF fusion. Retrieve top-50, rerank to top-N.
- MCP search interface unchanged; quality goes up. A/B benchmarking harness (`engram bench rerank`) added so the latency/quality tradeoff is measurable.

**M4 — Collapse to thoughts-only (Path B-OB1).**
- The M3 Phase D dogfood showed the facts pipeline's structured-triple abstraction was the wrong shape for the operator's use case (statements faithful, triples broken; 7 dogfood rounds). M4 collapses the schema: facts table goes away, replaced by a JSONB `tags` sidecar column on `thoughts` populated by an LLM tagger drainer.
- Content-fingerprint dedup at the thought level (SHA-256 unique constraint) so duplicate captures collapse to the same `thought_id`.
- `engram-extract` repurposed: `Extractor`/`ExtractedFact`/SPO machinery gone; `Tagger`/`Tags`/JSONB output. Initial M4 tag fields are `people`, `action_items`, `topics`, `dates_mentioned`, `kind` (M4.1 adds `entities`).
- MCP surface shrinks: `search_facts` and `correct_fact` removed; `search_thoughts` gains an optional `tag_filter` (JSONB containment); `capture` response gains `is_duplicate`.
- CLI surface: `engram reflect` → `engram tag` (same shape; tags are advisory and overwritten on `--rerun` rather than supersede-chained).

**M4.1 — v2 tagging.**
- Dogfood on the M4 v1 tagger surfaced two patterns: (1) the model already half-distinguished named-entities from inferred-categories but the v1 schema collapsed them; (2) topics were phrase-driven, producing divergent terms across paraphrases of the same concept. M4.1 ships a v2 prompt + small trait/storage/drainer adjustments to address both.
- `Tags` gains an `entities` field separate from `topics`. Schema is additive (JSONB-backed; no migration). The `Tagger` trait gains an optional `vocab: &ScopeVocab` parameter; the drainer pre-fetches the top-N most-frequent topic + entity terms in the thought's scope and renders them into the prompt as a controlled-vocabulary hint.
- Tagger version bumps 1→2; operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill existing rows.

**M5 — Selective relations.**
- Thought-to-thought graph layer on top of the M4 substrate. Closed relation vocabulary (initially `replaces`, `requires`, `references`, `belongs_to`, `decided_by`, `refines`; M5.1 added `supports` after day-one dogfood). Thought-to-thought edges at M5.
- New `thought_links` table; new MCP tools `link_thoughts`, `unlink_thoughts`, `get_related_thoughts`.
- Captures the relational structure that actually shows up in conversation memory (decisions, references, dependencies, refinements, evidential support) without trying to be a general knowledge graph.

**M5.2 — Heterogeneous targets + audit + CLI scope-prefix (close-the-M5-loop).**
- Polymorphic targets on `thought_links`: the `to` side can be a thought, an entity (free-text), a person (free-text), or a URL. Migration 0009 adds discriminator + per-kind columns + generated `to_value`.
- Soft-delete on `thought_links` (`deleted_at`); `unlink_thoughts` returns three-way status (`deleted_now` / `already_deleted` / `never_existed`). Partial unique index makes re-creating a previously-removed edge succeed cleanly.
- `migration_audit` table + `engram audit migrations` CLI subcommand for operator-visible diagnostics on per-migration row impact.
- `engram tag` and `engram embed-backfill` gain `--scope-prefix` (mutex with `--scope`), completing the M5.x retrieval-side scope_prefix work.

**M6 — Corpus stats CLI + tagger-extracted relations (v1, non-thought targets).** *(The original M6 was artifacts/long-form ingestion; that plan was dropped after a 2026-05-17 live-corpus measurement showed engram occupies a high-signal-density sweet spot that long-form ingestion would dilute. M5.2's `to_url` link target already covers the "reference external doc" case. See revision history for the pivot.)*
- New `engram stats` CLI subcommand prints corpus + storage telemetry (thought counts, embeddings, links, queues, per-scope summary, per-table sizes) without requiring psql. Operator-facing; no MCP surface in v1.
- v5 tagger prompt + schema add a `relations` field — the LLM emits closed-vocabulary `(relation, to_kind, to_value)` edges from prose. v1 ships non-thought targets only (`url` / `entity` / `person`); thought-to-thought tagger relations require entity resolution and are deferred.
- Drainer inserts emissions into `thought_links` with `source = 'tagger'`. Re-tag soft-deletes prior tagger emissions from the thought before fresh inserts (agent-supplied edges untouched). Bypass-on-error: malformed individual emissions are logged + skipped.

**M7 — Operational maturity.**
- Prometheus `/metrics` endpoint.
- Tier 2 bearer-token auth + audit log.
- Backup tooling (scripts, retention policy).
- Eval suite (capture-recall, cross-model retrieval consistency, LongMemEval-style).
- The `stats` MCP tool.

**Order rationale.** M1 is the floor: nothing else makes sense without capture and retrieval. M2 (facts) before M3 (rerank) because at the time facts added capability and rerank improved quality, and quality without capability felt unmotivated. M3 dogfood produced negative knowledge that motivated M4 (collapse to thoughts-only) — the facts pipeline didn't earn its complexity for the operator's actual queries. M5 (selective relations) comes after M4 because the citation-chain pattern that emerged from the M4.1 dogfood (thoughts referencing and refining each other in prose) was the strongest signal for the next architectural addition — making implicit graph structure first-class. M6's original "artifacts/long-form ingestion" plan was dropped after a live-corpus measurement showed engram occupies a high-signal-density sweet spot that long-form ingestion would dilute; the reshaped M6 ships `engram stats` and tagger-extracted relations instead (see revision history 2026-05-17). M7 (operational maturity) closes out the v0 plan.

## 4. High-level architecture

```
                   ┌──────────────────────────────────────────┐
                   │             Engram (single binary)       │
                   │                                          │
  MCP clients      │   ┌──────────┐    ┌────────────────┐     │
  (Claude Code, ──→│──→│ MCP/HTTP │───→│   Core service │     │──┐
   Desktop, etc.)  │   │  surface │    │  (capture,     │     │  │
   over Tailscale  │   └──────────┘    │   retrieval,   │     │  │
                   │                   │   tagging)     │     │  │
                   │   ┌──────────┐    └────────────────┘     │  │
                   │   │  Worker  │            │              │  │
                   │   │ (drainer)│────────────┘              │  │
                   │   └──────────┘   [M2+]                   │  │
                   │         │                                │  │
                   │         ▼                                │  │
                   │   ┌──────────────────────────────────┐   │  │
                   │   │ Embedder + Reranker + Tagger     │   │  │
                   │   │ (traits) — OpenAI-compatible /   │   │  │
                   │   │ TEI defaults                     │   │  │
                   │   └──────────────────────────────────┘   │  │
                   │                  │                       │  │
                   └──────────────────┼───────────────────────┘  │
                                      ▼                          ▼
                            ┌──────────────────┐         ┌────────────┐
                            │  vLLM endpoint   │         │ Postgres   │
                            │  (instruct +     │         │ + pgvector │
                            │   embedding,     │         │            │
                            │   localhost:8000)│         └────────────┘
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  RTX 3090(s)     │
                            └──────────────────┘
```

Engram is a *client* of the local vLLM endpoint, not the operator of it. vLLM is presumed to be serving primary inference traffic to other Tailscale-connected devices anyway; Engram piggybacks on that infrastructure. Three logical components, one binary:

- **MCP/HTTP surface.** Streamable HTTP transport speaking MCP. Same binary also exposes an admin HTTP API.
- **Core service.** Capture, search, retraction, scope management.
- **Worker.** [M2+] Runs two drainer tasks in a Tokio process when the binary is launched in `worker` mode: the embed drainer (always on; pulls from `pending_embeddings`) and the tag drainer ([M4+]; pulls from `pending_tags` when `[tagger].provider` is non-empty). **The worker process does not exist in M1**; capture-side embedding is synchronous in the server process. From M2 onward all derived-signal production is async.

## 5. Data model

The model is deliberately small. Two primary entities — thoughts and embeddings — plus an artifacts table for long-form content reserved for M5. Embeddings are intentionally a separate first-class table so model swaps are routine rather than migrations. From M4 onward, thoughts carry a JSONB `tags` sidecar populated by the tagger drainer; tagger output is overwritten on re-tag and is *advisory metadata*, not load-bearing.

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Raw, immutable captures. Single source of truth. [M1 + M3 retraction + M4 tags/fingerprint]
CREATE TABLE thoughts (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope                   TEXT NOT NULL DEFAULT 'global',
    content                 TEXT NOT NULL,
    source                  TEXT NOT NULL,           -- 'manual', 'agent:claude-code', etc.
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata                JSONB NOT NULL DEFAULT '{}',
    -- M3 retraction
    retracted_at            TIMESTAMPTZ,
    retracted_reason        TEXT,
    -- M4 dedup + tagging sidecar
    content_fingerprint     BYTEA NOT NULL,          -- SHA-256 of content; UNIQUE
    tags                    JSONB NOT NULL DEFAULT '{}',
    tags_extractor_model    TEXT,                    -- NULL until first tag pass
    tags_extractor_version  INT,
    tags_extracted_at       TIMESTAMPTZ,
    UNIQUE (content_fingerprint)
);

CREATE INDEX thoughts_scope_recent_idx
    ON thoughts (scope, created_at DESC) WHERE retracted_at IS NULL;
CREATE INDEX thoughts_content_trgm_idx
    ON thoughts USING gin (content gin_trgm_ops);
CREATE INDEX thoughts_tags_gin
    ON thoughts USING gin (tags);            -- JSONB containment for tag_filter

-- Migration 0001 also ships `artifacts` and `artifact_chunks` tables for
-- what was then a planned long-form-ingestion milestone. That plan was
-- dropped in the M6 reshape (2026-05-17) after a live-corpus measurement
-- showed engram occupies a high-signal-density sweet spot that long-form
-- ingestion would dilute. Those two tables are inert and slated for
-- removal by a future drop migration; they are intentionally omitted
-- from the schema block above.

-- Embeddings are first-class. Multiple per target during model migration.
CREATE TABLE embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind     TEXT NOT NULL CHECK (target_kind IN ('thought','artifact_chunk','fact')),
    target_id       UUID NOT NULL,
    model_id        TEXT NOT NULL,           -- e.g. 'bge-m3:1024'
    model_version   INT NOT NULL DEFAULT 1,
    vector          vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (target_kind, target_id, model_id, model_version)
);

-- One HNSW partial index per active embedding model. M1 ships this one.
-- Adding a new model = a future migration adds a new partial index over
-- the same table; old rows stay; the active-model concept lives in config
-- (see §9), not in a Postgres GUC.
CREATE INDEX embeddings_bge_m3_hnsw
    ON embeddings USING hnsw (vector vector_cosine_ops)
    WHERE model_id = 'bge-m3:1024';

-- M2 added pending_embeddings to back the async embedding seam.
-- M4 added pending_tags as the second drain queue.
-- The facts / reflector_runs / facts_review_queue tables shipped in M2
-- and were dropped by M4's migration 0006 — see docs/milestones/m4-spec.md
-- for the contract and docs/milestones/m4-collapse-to-thoughts.md for the
-- architectural rationale.

-- Durable FIFO queue backing the async embedding seam. Capture inserts a
-- row; `engram worker`'s drainer task pulls batches via
-- `UPDATE ... FROM (SELECT ... FOR UPDATE SKIP LOCKED LIMIT $1)`.
-- The UNIQUE constraint makes enqueue idempotent.
CREATE TABLE pending_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind     TEXT NOT NULL CHECK (target_kind IN ('thought','artifact_chunk','fact')),
    target_id       UUID NOT NULL,
    model_id        TEXT NOT NULL,                    -- pairs the job with the right embedder
    enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    attempts        INT NOT NULL DEFAULT 0,
    last_attempt_at TIMESTAMPTZ,
    last_error      TEXT,
    UNIQUE (target_kind, target_id, model_id)
);
CREATE INDEX pending_embeddings_dequeue_idx ON pending_embeddings (enqueued_at ASC);
-- The `'fact'` value in the target_kind CHECK constraint is intentionally
-- retained post-M4 so a future re-introduction of a facts table wouldn't
-- need a schema migration.

-- M4: queue table feeding the tag drainer. One pending tag job per
-- thought at a time (PK on thought_id makes re-enqueue idempotent).
CREATE TABLE pending_tags (
    thought_id      UUID PRIMARY KEY REFERENCES thoughts(id) ON DELETE CASCADE,
    tagger_model_id TEXT NOT NULL,
    enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    attempts        INT NOT NULL DEFAULT 0
);
```

**Why embeddings are a separate table.** A model swap is a re-index, not a re-write. With this layout we insert a new row in `embeddings` per `(target, new model)`, build a new HNSW partial index for the new model, and once the operator is satisfied with retrieval quality, optionally drop the old rows and old index. No data is lost during the swap.

**One HNSW index per model.** Each active embedding model gets its own partial index, predicated on a literal `model_id` string. This is required for correctness: Postgres demands partial-index predicates be `IMMUTABLE`, and `current_setting()` is `STABLE`. The "active embedder" is therefore a config concern (see §9), not a database GUC. Operationally, swapping the active model means: ship a migration that adds the new partial index, update config to point at the new `model_id`, restart.

**Scoping.** Free-form string, default `global`. Convention rather than enforcement: `work.tcgplayer`, `personal`, `project.engram`, etc. A `scopes` registry table can come later if introspection is wanted.

## 6. Ingest path

There is one write path. It terminates in a `thoughts` row plus an embedding plus (when the tagger is configured) a tags JSONB sidecar.

**Direct capture.** [M1 + M4 dedup] Agent calls `capture(content, scope?, source?, metadata?)`. The handler computes `content_fingerprint = sha256(content)` and runs `INSERT INTO thoughts (..., content_fingerprint, tags) VALUES (..., $fp, '{}') ON CONFLICT (content_fingerprint) DO NOTHING RETURNING id`. On insert: enqueue an embedding job and (if `[tagger]` is configured) a tag job; return `{thought_id, embedding_status: "pending", is_duplicate: false}`. On conflict: SELECT the existing `thought_id` by fingerprint and return `{thought_id, embedding_status: <existing state>, is_duplicate: true}` — no new jobs enqueued.

(An artifact-ingestion write path was on the original M6 surface; M6 was reshaped and the artifacts plan was dropped — see revision history 2026-05-17.)

**Designed-in seam for async embedding.** [M2+] In M1 the capture handler called `Embedder::embed(...)` directly. From M2, the worker process exists, and the capture handler enqueues a row in `pending_embeddings`; the worker drains the queue. The MCP contract is identical to M1; the embedding row becomes available shortly after capture returns (with a brief window during which `search_thoughts` may not surface the brand-new thought via vector — trigram still finds it).

**Tagging sidecar.** [M4+] The same worker process runs a second drainer task against `pending_tags`. The tag drainer pulls batches the same way the embed drainer does, calls `Tagger::tag(content, vocab)`, and writes the persisted `Tags` half into `thoughts.tags` plus the three provenance columns (`tags_extractor_model`, `tags_extractor_version`, `tags_extracted_at`). From M6.1, the same tagger call also returns `relations` — closed-vocabulary edges from prose that the drainer routes into `thought_links` with `source='tagger'` (see §6.7). Both drainers share the `[worker]` knobs (`tick_interval_seconds`, `batch_size`); tagging is opt-in via `[tagger].provider` non-empty. The tagging-sidecar shape is the subject of §6.5; §10 covers the operational shape and the rationale for *not* having drift-defense machinery.

## 6.5 Tagging sidecar

[M4+] The tagger reads each new thought and writes a JSONB metadata blob onto the same row. Six fields: `people`, `entities`, `action_items`, `topics`, `dates_mentioned`, `kind`. `entities` and `topics` are separate slots ([M4.1+]): `entities` lists proper-noun-style identifiers the prose mentions by name (projects, products, libraries, named concepts); `topics` lists broader subject categories the thought falls under. Keeping them separate lets `tag_filter` distinguish "thoughts that mention engram by name" from "thoughts categorized under memory-systems." Tags are advisory metadata — they don't gate storage or supersede each other; they're an optional filtering signal at retrieval time and a UX-time annotation in `search_thoughts` responses.

**Why a sidecar, not a separate table.** M2 shipped a `facts` pipeline that decomposed each thought into structured `(subject, predicate, object, confidence, statement)` rows. M3 Phase D dogfood (7 rounds, 2026-05-13 → 2026-05-16) produced a consistent finding: the *statement* field came back faithful to the source thought, but the *triples* came back broken — comparative S/O inversion, self-referential subjects, conditional-as-subject, predicate verbosity, polarity contradictions, triple-semantic drift. The producer (local 30B-class coding model) couldn't reliably emit triples; the consumer (LLM agents reading prose) didn't query by `(S, P, O)`. M4 collapsed the pipeline: drop the `facts` table, write a JSONB sidecar on the thought instead, and treat tagger output as *overwriteable* rather than supersede-chained. The architectural antecedent is OB1's `metadata` column; the design philosophy is *raw data is permanent, derived signals are recomputable*.

**The pipeline.** Four steps, all in `engram worker`'s tag-drainer task:

1. **Drain.** `SELECT thought_id, tagger_model_id FROM pending_tags ORDER BY enqueued_at ASC FOR UPDATE SKIP LOCKED LIMIT $batch_size`. Fetched in the same idempotent style as `pending_embeddings`.

2. **Tag.** Call `Tagger::tag(content, vocab)`. The default impl (`OpenAICompatibleTagger`) POSTs to `/v1/chat/completions` with the bundled prompt (currently v7) + `response_format: { type: "json_schema", strict: true, schema: { ... } }`. Schema (live in `crates/engram-extract/src/openai_compatible.rs`):

    ```json
    {
      "type": "object",
      "additionalProperties": false,
      "required": ["people", "entities", "action_items", "topics", "dates_mentioned", "kind", "relations"],
      "properties": {
        "people":          { "type": "array", "items": { "type": "string" } },
        "entities":        { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
        "action_items":    { "type": "array", "items": { "type": "string" } },
        "topics":          { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
        "dates_mentioned": { "type": "array", "items": { "type": "string" } },
        "kind":            { "type": ["string", "null"],
                             "enum": ["observation","task","idea","reference","person_note","session", null] },
        "relations":       { "type": "array", "maxItems": 5,
                             "items": { "type": "object",
                                        "required": ["relation","to_kind","to_value","note"],
                                        "properties": {
                                          "relation": { "type": "string",
                                                        "enum": ["replaces","requires","references","supports","belongs_to","decided_by","refines"] },
                                          "to_kind":  { "type": "string", "enum": ["entity","person","url"] },
                                          "to_value": { "type": "string" },
                                          "note":     { "type": ["string","null"] } } } }
      }
    }
    ```

    [M6.1+] The LLM emits both `tags` fields (persisted to `thoughts.tags` JSONB) and `relations` (routed to `thought_links` with `source='tagger'`, not persisted into the JSONB — see §6.7). `Tagger::tag` returns `TagOutput { tags, relations }`; the drainer destructures and writes the two halves through separate code paths.

    [M4.1+] Before the tagger call, the drainer optionally pre-fetches the top-N most-frequent topic and entity terms from the thought's scope (via `engram_storage::fetch_scope_vocab`) and passes them to `Tagger::tag(content, Some(&vocab))`. The default `OpenAICompatibleTagger` renders the vocab into a "controlled vocabulary" section appended to the system prompt — the model is told to prefer established terms when they fit and coin new ones only for genuinely unseen concepts. This produces consistent topic vocabulary at the corpus level: the same author writing about the same subject in different prose now lands in overlapping topic terms, addressing v1's phrase-driven divergence. Controlled by `[tagger].scope_vocab_enabled` (default `true`) and `[tagger].scope_vocab_size` (default `50`).

    On per-thought tagger failure (`Timeout`, `Unreachable`, `Backend 5xx`, `MalformedResponse`), the drainer **soft-fails**: logs a warning with `transient = err.is_transient()`, increments the row's `attempts` counter, and leaves it in `pending_tags`. Next tick retries. Vocab-fetch failure folds into the same transient bucket.

3. **Write.** `UPDATE thoughts SET tags = $tags, tags_extractor_model = $model_id, tags_extractor_version = $version, tags_extracted_at = NOW() WHERE id = $thought_id`; then `DELETE FROM pending_tags WHERE thought_id = $thought_id`. The two statements run in one transaction. There is no supersede chain — the tags column is overwritten on every successful tagger pass.

4. **(Optional) Re-tag.** `engram tag --rerun [--since <RFC3339>] [--scope X] [--limit N]` walks thoughts whose `tags_extractor_version < current_tagger_version` (or `IS NULL`, for the first pass on a previously-untagged thought), calls the tagger, and overwrites. Use this after bumping `[tagger].model_version` on a prompt or schema change.

**Concrete example.** A thought: *"Talked to Sarah today about the PR backlog. She wants migration #0042 fast-tracked because the mobile freeze starts Thursday."* The tagger returns:

```json
{
  "people": ["Sarah"],
  "entities": ["migration #0042"],
  "action_items": ["fast-track migration #0042"],
  "topics": ["pr-backlog", "release-process"],
  "dates_mentioned": ["Thursday"],
  "kind": "task"
}
```

The blob lands in `thoughts.tags` via the drainer. A subsequent `search_thoughts(query, tag_filter = {"people": ["Sarah"]})` filters to thoughts whose tags JSONB contains `{"people": ["Sarah"]}` (JSONB `@>` containment, GIN-indexed). `get_thought` surfaces the tags + provenance alongside the content.

**What this gives you, in plain English.** A single source of truth (the thought) with two derived layers — embeddings for retrieval, tags for filtering — both recomputable independently from the raw text. No drift-defense, no supersession chain, no review queue, no audit trail on tagger output. The raw thought stays immutable (§10); the tagger output is overwritten as model and prompt drift. A wrong tag on a single thought is a low-impact failure mode because retrieval still works on the content.

## 6.6 Selective relations (graph layer)

[M5+] On top of the tagging sidecar, M5 adds a graph layer of edges over thoughts in a small closed vocabulary. M5 shipped six relations; M5.1 added a seventh after day-one dogfood (see Revision history): `replaces`, `requires`, `references`, `supports`, `belongs_to`, `decided_by`, `refines`. `supports` separates evidential / corroborative relationships ("this confirms a claim made there") from `references` (prose-level citation). M5 launched thought-to-thought only; M5.2 generalized the target to a polymorphic shape — an edge can now point at another thought, a free-text entity name, a free-text person name, or a URL — and added soft-delete so a removed edge can be re-created cleanly.

**Why selective, not general.** The M3 facts pipeline tried full open-vocabulary `(subject, predicate, object)` extraction and the dogfood (see Revision history 2026-05-16 entries) showed the predicate slot broke under small-model limitations. M5's closed vocabulary trades coverage for tractability — seven relations the operator can predict (six at M5 launch; `supports` added in M5.1), queries that always have a fixed-cardinality dispatch, and no extraction prompt to break under load. The vocabulary was picked from observation of the M4.1 dogfood corpus: the citation chain `137dba1d → 6d2ef58e → 8a533e15` is exactly the `refines`-style structure operators kept building in prose.

**Schema.** Migration 0007 introduces the table; 0009 generalizes the target shape; 0010 adds soft-delete + the partial unique index.

```sql
CREATE TABLE thought_links (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_thought_id UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    relation        TEXT NOT NULL,
    -- Polymorphic target [M5.2]: exactly one of the four columns is populated;
    -- to_kind is the discriminator; to_value is a generated convenience column
    -- (canonical string form of whichever target column is non-null).
    to_kind         TEXT NOT NULL,                          -- 'thought'|'entity'|'person'|'url'
    to_thought_id   UUID REFERENCES thoughts(id) ON DELETE CASCADE,
    to_entity       TEXT,
    to_person       TEXT,
    to_url          TEXT,
    to_value        TEXT GENERATED ALWAYS AS (
                        COALESCE(to_thought_id::text, to_entity, to_person, to_url)
                    ) STORED,
    source          TEXT NOT NULL DEFAULT 'agent',
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ,                            -- soft-delete [M5.2]
    CHECK (relation IN ('replaces','requires','references','supports','belongs_to','decided_by','refines')),
    CHECK (source IN ('agent','tagger')),
    CHECK (to_kind IN ('thought','entity','person','url')),
    -- Exactly-one-target invariant.
    CHECK (
        (to_kind = 'thought' AND to_thought_id IS NOT NULL AND to_entity IS NULL AND to_person IS NULL AND to_url IS NULL) OR
        (to_kind = 'entity'  AND to_entity     IS NOT NULL AND to_thought_id IS NULL AND to_person IS NULL AND to_url IS NULL) OR
        (to_kind = 'person'  AND to_person     IS NOT NULL AND to_thought_id IS NULL AND to_entity IS NULL AND to_url IS NULL) OR
        (to_kind = 'url'     AND to_url        IS NOT NULL AND to_thought_id IS NULL AND to_entity IS NULL AND to_person IS NULL)
    ),
    -- No self-links for thought targets.
    CHECK (NOT (to_kind = 'thought' AND from_thought_id = to_thought_id))
);

-- Idempotency on the live (non-deleted) triple. Soft-deleted rows don't
-- conflict, so re-creating a previously-removed edge succeeds.
CREATE UNIQUE INDEX thought_links_live_uq
    ON thought_links (from_thought_id, relation, to_kind, to_value)
    WHERE deleted_at IS NULL;
```

`source` distinguishes agent-supplied (M5) from tagger-extracted (M6.1). `note` is an optional free-text annotation (max 1000 chars enforced at the MCP layer). `ON DELETE CASCADE` is safe because retraction is soft — edges resolve against retracted thoughts and just surface the `retracted: true` flag rather than disappear. Soft-delete on the edge itself (the `deleted_at` column) preserves an audit trail of removed edges while making re-creation idempotent.

**Pipeline.** Two write paths land in this table — agent-supplied via MCP and tagger-extracted via the M6.1 drainer (covered in §6.7). Read path is shared.

1. **`link_thoughts(from, relation, to, note?)`** [M5+, M5.2 polymorphic] takes exactly one of `to_thought_id` / `to_entity` / `to_person` / `to_url`. Validates self-link (for thought targets), note length, and endpoint existence (for thought targets), then calls `engram_storage::insert_link` with ON CONFLICT idempotency against the live unique index. Returns `is_new: bool` (mirrors `capture`'s `is_duplicate` polarity) plus the `link_id`.
2. **`unlink_thoughts(from, relation, to)`** [M5+, M5.2 three-way status] soft-deletes by stamping `deleted_at`. Returns a three-way `status: deleted_now | already_deleted | never_existed` so the caller can distinguish a no-op-because-already-gone from a no-op-because-never-there. A subsequent `link_thoughts` of the same triple succeeds cleanly.
3. **`get_related_thoughts(thought_id, relations?, target_kinds?, direction?)`** walks the graph from a single thought. Returns grouped `outbound` (edges where this thought is `from`) and `inbound` (edges where it's `to`, thought-target rows only). Each edge carries `to_kind`, `to_value`, `link_source` (`agent` | `tagger`), and `note`. Thought-target hits additionally surface the related thought's content_preview, scope, and retraction state. Filters: `relations` (closed-vocab subset), `target_kinds` (outbound only — restrict to specific `to_kind` values), `direction` (`outbound` | `inbound` | `both`).

**Concrete example.** Three thoughts in a citation chain plus a URL citation:

```text
link_thoughts(137dba1d, "refines",    to_thought_id: 6d2ef58e, note: "post-Probe-2 refinement")
link_thoughts(6d2ef58e, "refines",    to_thought_id: 8a533e15, note: "first refinement")
link_thoughts(137dba1d, "references", to_url: "https://example.org/probe-2", note: "source")
```

`get_related_thoughts(8a533e15, direction: "inbound")` returns the inbound `refines` edge from `6d2ef58e`. `get_related_thoughts(137dba1d, direction: "outbound")` returns the `refines→6d2ef58e` (a thought hit, with content_preview) and the `references→<url>` (a URL hit). `unlink_thoughts(137dba1d, "references", to_url: "...")` returns `deleted_now`; a re-issued `unlink_thoughts` on the same triple returns `already_deleted`; a third `link_thoughts` on the same triple succeeds because the soft-deleted row is excluded from the live unique index.

**Migration audit (M5.2 ancillary).** Migration 0010 also introduces a `migration_audit` table that any row-touching migration populates by convention. `engram audit migrations` surfaces the log so an operator can answer "what did each migration actually touch on my corpus" without psql. Currently populated by 0009, 0010, and 0011.

**Out of scope at v0.** First-class `entities` / `persons` tables (the free-text columns on `thought_links` are the v0 representation; entity resolution is its own multi-conversation problem). Reverse traversal from non-thought targets, restore-link tool, hard-purge of soft-deleted edges, sibling `engram audit links` / `engram audit thoughts` resources, bulk-link tooling, multi-hop traversal (`get_thoughts_n_hops_away`), an `engram link` CLI shortcut — deferred per usage demand.

## 6.7 Tagger-extracted relations (M6.1)

[M6.1+] The graph layer in §6.6 was agent-supplied at M5 launch. M6.1 adds a second writer: the tagger emits relations from prose, and the drainer routes them into the same `thought_links` table with `source = 'tagger'`.

**Why the tagger writes graph rows.** Some relational structure is observable in the surface form of the thought itself ("This finding supports earlier observation X"; "Decision per Ron"; "See arxiv.org/abs/..."). When the LLM can spot it during the existing tagging pass, there is no reason to make the agent re-derive the same edge at capture time. The closed seven-relation vocabulary already trained on §6.6 carries over; only the target shape narrows (no thought-to-thought tagger relations until entity resolution lands — the tagger can read prose but cannot reliably resolve "the earlier Probe 2 finding" to a specific UUID).

**Schema reuse.** No schema changes for M6.1. M5.2's polymorphic targets (`to_kind ∈ {entity, person, url}`), `LinkSource::Tagger` discriminator, and soft-delete machinery were the exact substrate this milestone needed.

**Pipeline.** The tag drainer (§6.5 step 2) already calls `Tagger::tag(content, vocab)`. Post-M6.1 that call returns `TagOutput { tags, relations }`. The drainer destructures:

1. **Write tags** (unchanged from §6.5): `update_thought_tags` writes `tags` into `thoughts.tags` JSONB.
2. **Apply relations** ([M6.1]): `engram_mcp::apply_tagger_relations(thought_id, &output.relations, pool)` first calls `soft_delete_tagger_edges_for_thought(thought_id)` — every existing `source='tagger'` edge from the thought is stamped with `deleted_at = NOW()`. Then each emission in `output.relations` is validated (`link::validate_target` — URL must be `http(s)://`, names ≤200 chars) and inserted via the standard `insert_link` path with `source = 'tagger'`. Bypass-on-error: a single malformed emission is logged and skipped; the rest of the batch still lands; the tag job is never failed by a bad relation.

Re-tag cycles (operator-initiated via `engram tag --rerun`) replay step 2: prior tagger edges are soft-deleted and fresh ones inserted. Agent-supplied edges (`source = 'agent'`) are untouched — only the tagger-owned subset churns.

**Why not persist into the `tags` JSONB.** v5/M6.1 originally wrote tagger emissions into BOTH `thoughts.tags.relations` (raw frozen JSONB) AND `thought_links` rows. The live-corpus dogfood revealed every persisted JSONB entry had a corresponding `thought_links` row — pure DRY violation. v9 (2026-05-18, migration 0011) dropped the `tags.relations` JSONB field; `thought_links` is now the single canonical store. The LLM-side response schema (§6.5) still asks for a top-level `relations` array — `OpenAICompatibleTagger` parses it into the transient `TagOutput.relations` field; only the persistence shape changed.

**Consumer-side discriminator.** `get_related_thoughts` exposes `link_source` on every edge so agents downstream can treat tagger-emitted edges as advisory (re-tag may add or remove them) versus agent-supplied edges as deliberate. `unlink_thoughts` (M5.2 soft-delete) is the operator-correction tool for a wrong tagger edge; the next re-tag cycle may re-emit it, which is the signal that the prompt or the source thought needs work (operator's call, not the agent's).

## 7. Retrieval path

Three retrieval primitives, composable:

- **Semantic** — vector kNN over the active embedding model.
- **Lexical** — `pg_trgm` similarity over `content`. Cheap; complements vector search for proper nouns, acronyms, and code identifiers — exactly the queries pure embeddings are notoriously bad at.
- **Recency** — `ORDER BY created_at DESC` with a scope filter.

**Default `search_thoughts` from M1 is a hybrid.** Concretely:

1. Run two SQL queries in parallel: a vector kNN limited to top-K against the active model's HNSW index; a trigram similarity query limited to top-K against `thoughts_content_trgm_idx`.
2. Fuse the result sets with reciprocal rank fusion (RRF): `score(d) = Σᵢ 1 / (k + rankᵢ(d))` over the two rankings, with `k` typically 60.
3. Apply scope filtering and a recency boost (multiplicative `exp(-age/τ)` with `τ` = 30 days, configurable per call). Return the top N.

Why RRF over a weighted-score blend like `α·cos_sim + β·bm25 + γ·exp(-age/τ)`: RRF is parameter-light, robust to score-distribution differences between heterogeneous rankers, and is the de-facto choice for vector + lexical hybrids in current information-retrieval literature. It also generalizes cleanly to a third ranker when the M3 reranker is added.

```rust
pub struct SearchRequest {                       // [M1 + M3 rerank + M4 tag_filter]
    pub query: String,
    pub scope: Option<Scope>,
    pub limit: Option<usize>,                    // defaults to 10; max 100
    pub recency_half_life_days: Option<f32>,     // default 30; 0 disables
    pub rerank: Option<bool>,                    // M3
    pub candidate_pool: Option<usize>,           // M3; default 32
    pub tag_filter: Option<serde_json::Value>,   // M4; JSONB containment against thoughts.tags
}
```

**Reranker.** [M3] M3 adds a cross-encoder rerank pass after RRF fusion: retrieve a wider candidate set (typically top-32), rerank with a cross-encoder via TEI to get the final top-N. The MCP search interface stays additive-compatible — the `rerank` and `candidate_pool` fields are optional with sensible defaults; clients written against M1 keep working.

**Tag filter.** [M4] When `tag_filter` is present, the SQL adds `AND tags @> $tag_filter` to both legs (vector + trigram), using the `thoughts_tags_gin` GIN index. Examples: `{"kind": "task"}`, `{"people": ["Sarah"]}`, `{"topics": ["rust"], "kind": "idea"}`. When omitted or `{}`, no filter applies. Each `SearchHit` carries the full `Tags` object so consumers can see the tagger's classification per result.

## 8. MCP surface

Tools and the milestone in which each ships. Names and signatures are part of the contract once shipped.

| Tool | Milestone | Purpose |
|---|---|---|
| `capture` | M1 (M2 async; M4 dedup) | Store a thought. Computes SHA-256 fingerprint; duplicate content returns the existing `thought_id` with `is_duplicate: true` and no new embed/tag jobs. New captures enqueue both an embedding job and (if `[tagger]` is configured) a tag job; response is `{thought_id, embedding_status: "pending", is_duplicate}`. |
| `search_thoughts` | M1 (M3 rerank; M4 tag_filter) | Hybrid retrieval over thoughts. RRF-fused vector + trigram + recency boost; M3 cross-encoder rerank stage on top; M4 adds an optional `tag_filter` (JSONB containment against `thoughts.tags`). Each hit carries `tags`. Gracefully degrades to trigram-only when the embedder is unreachable; excludes retracted thoughts. |
| `recent_thoughts` | M1 | Browse by recency in a scope. Excludes retracted thoughts. |
| `get_thought` | M1 (M3 retraction; M4 tags) | Full thought + provenance (`embedding_status`, `embedded_at`, `tags`, `tags_extractor_model`, `tags_extractor_version`, `tags_extracted_at`, `retracted_at`, `retracted_reason`). Direct lookup returns the row even if retracted — this is the audit path. |
| `retract_thought` | M3 | Mark a thought as untrusted. Sets `thoughts.retracted_at`; the row is excluded from retrieval but stays in the DB for audit. |
| `list_scopes` | M5.x | Enumerate scopes currently in use with `thought_count` / `first_activity_at` / `last_activity_at`. Optional `prefix` filter. Sorted most-recently-used first. |
| `link_thoughts` | M5 (M5.2 polymorphic targets) | Create a link from a thought to a polymorphic target — another thought (`to_thought_id`), entity name (`to_entity`), person name (`to_person`), or URL (`to_url`). Exactly one of the four target fields. Seven closed-vocabulary relations (`replaces`, `requires`, `references`, `supports`, `belongs_to`, `decided_by`, `refines`). Idempotent on `(from, relation, to_kind, to_value)`. |
| `unlink_thoughts` | M5 (M5.2 soft-delete + three-way status) | Soft-delete a link by its `(from, relation, target)` triple. Returns `status: deleted_now | already_deleted | never_existed`. Re-creating a soft-deleted edge succeeds. |
| `get_related_thoughts` | M5 (M5.2 polymorphic) | Walk the link graph. Returns grouped `outbound` + `inbound` arrays. Each hit carries `to_kind` / `to_value` / `link_source`; thought-target hits additionally surface the target's content_preview, scope, retraction state. Optional filters: `relations`, `target_kinds` (outbound only), `direction`. |

Operator-only surfaces (not exposed as MCP tools): `engram stats` (corpus + storage telemetry, shipped at M6.0) and `engram audit migrations` (per-migration row impact log, shipped at M5.2) are CLI subcommands; an MCP `stats` tool is deferred to M7+. `ingest_artifact` was on the original M6 surface and was dropped when M6 was reshaped (see revision history 2026-05-17).

`search_facts` and `correct_fact` shipped in M2 and were removed in M4 when the facts pipeline was retired. Operators correcting a wrong claim now `retract_thought` and capture a corrected one; tags are advisory and re-derivable, so per-tag operator correction was unnecessary.

## 9. Embedding, reranking & tagging abstractions

Three traits, no other architectural concession to model choice.

```rust
#[async_trait]                      // [M1]
pub trait Embedder: Send + Sync {
    fn model(&self) -> &EmbeddingModel;          // { id: String, dimensions: usize }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError>;
}

#[async_trait]                      // [M3]
pub trait Reranker: Send + Sync {
    fn model_id(&self) -> &str;
    async fn rerank(
        &self,
        query: &str,
        candidates: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError>;
}

#[async_trait]                      // [M4]
pub trait Tagger: Send + Sync {
    fn model_id(&self) -> &str;                  // e.g. "vllm/qwen2.5-7b-instruct"
    fn version(&self) -> i32;                    // bumped when tagger prompt/schema changes
    // [M4.1] vocab is the top-N established topic + entity terms in the
    // thought's scope; rendered into the prompt as a controlled-vocabulary hint.
    // [M6.1+] returns TagOutput { tags, relations }; the drainer writes tags
    // to thoughts.tags JSONB and routes relations into thought_links with
    // source='tagger' (see §6.7).
    async fn tag(&self, thought_content: &str, vocab: Option<&ScopeVocab>) -> Result<TagOutput, TaggerError>;
}
```

Data shapes (`Tags`, `ScopeVocab`, `TagOutput`, `ExtractedRelation`, `RelationKind`, `LinkTarget`, `LinkSource`, `ThoughtLink`) live in `crates/engram-core`; the SQL forms in §5 and §6.6 are the contract, and the Rust shapes follow them. All three trait error enums carry `is_transient(&self) -> bool` so capture/drain orchestrators can decide retry vs. mark-failed per call. `Timeout`, `Unreachable`, and `Backend { status: 500..=599, .. }` are transient; everything else is not.

**Default implementations:**

- **`OpenAICompatibleEmbedder`** [M1] — one type covering every OpenAI-`/v1/embeddings`-shaped backend by config alone: Ollama (dev default, `http://localhost:11434/v1` + `bge-m3`), Hugging Face TEI (production sidecar), OpenAI, Voyage. Dimension validation against the declared `model.dimensions` is built in.
- **`FakeEmbedder`** [M1] — deterministic in-memory embedder for tests. Configurable via `FakeBehavior` to always fail (`Timeout` / `Unreachable`) so soft-fail paths can be exercised without standing up TEI/Ollama.
- **`TeiReranker`** [M3] — POSTs to TEI's `/rerank` endpoint. `FakeReranker` mirrors `FakeEmbedder` for tests.
- **`OpenAICompatibleTagger`** [M4] — one type covering every OpenAI-`/v1/chat/completions`-shaped backend that supports `response_format: { type: "json_schema", strict: true }`. Named-constructor presets `OpenAICompatibleConfig::vllm_local()` (default `http://localhost:8000/v1`, no key) and `::open_router(api_key, model)` cover the production-sidecar and cloud-fallback paths.
- **`FakeTagger`** [M4] — deterministic in-memory tagger mirroring `FakeEmbedder`. `Empty` / `Canned(Tags)` / `Substring(map<&str, Tags>)` behaviors for unit-test control; `always_failing(FakeBehavior)` exercises the soft-fail path.

The trait boundary is the buffer-from-model-changes guarantee. Swapping vLLM's served model, swapping to SGLang, swapping to a cloud provider — all happen behind the same interface. The only operation that propagates beyond the trait is a re-embed when the *embedder* changes (which is why §5 makes embeddings a separate table); a tagger swap is just `engram tag --rerun --since 1970-01-01T00:00:00Z` and tags overwrite in place.

**Active-embedder selection.** From M1 onward the active embedder is identified by `model_id` (e.g. `bge-m3:1024`) and is a config field — the engram TOML declares which model is active; that string must match the predicate of an existing HNSW partial index. There is intentionally no Postgres-side GUC.

Configuration is a TOML file:

```toml
[database]
# Postgres connection. Overridden by DATABASE_URL env var if set (sqlx convention).
url = "postgres://engram:engram@localhost:5432/engram"
max_connections = 10

[server]                                        # [M1]
bind = "127.0.0.1:8080"                         # Tier 0 default — see §12
allowed_hosts = []                              # [M5.x] DNS-rebinding allowlist;
                                                # empty = rmcp default (localhost/127.0.0.1/::1);
                                                # non-empty REPLACES the default (Tier 1 mesh access)

[embedder]                                      # [M1]
provider     = "openai-compatible"
endpoint     = "http://localhost:11434/v1"      # Ollama in dev; TEI in production
model        = "bge-m3"                         # backend-side model name
model_id     = "bge-m3:1024"                    # Engram-side identity; must match an HNSW index
dimensions   = 1024
timeout_seconds = 5

[worker]                                        # [M2 + M4]
tick_interval_seconds = 5                       # embed-drainer AND tag-drainer wakeup cadence
batch_size            = 16                      # max jobs per tick (per queue)

[reranker]                                      # [M3]
provider        = "tei"                         # "" = disabled (default)
endpoint        = "http://localhost:8080"       # TEI; no /v1 suffix (impl appends /rerank)
model_id        = "cross-encoder/ms-marco-MiniLM-L-6-v2"
timeout_seconds = 30

[tagger]                                        # [M4]; empty provider = silent disable
provider        = "openai-compatible"           # alternative: "openrouter"; "" = disabled
endpoint        = "http://localhost:8000/v1"    # vLLM default
model_name      = "qwen2.5-7b-instruct"         # backend-side model name
model_id        = "vllm/qwen2.5-7b-instruct"    # provenance label → thoughts.tags_extractor_model
model_version   = 7                             # tracks BUNDLED_TAGGER_VERSION; bump on prompt/schema change
scope_vocab_enabled = true                      # [M4.1] controlled-vocabulary hint per scope
scope_vocab_size    = 50                        # [M4.1] top-N established terms per scope
timeout_seconds = 60
temperature     = 0.2
```

The `[extractor]` and `[reflector]` sections that shipped in M2 were removed by M4. The tagger drainer is always-on when `[tagger].provider` is non-empty — no cron, no opt-in flag, no confidence-band routing.

**Hardware sizing — concrete on the Phase 1 / Phase 2 BOM, single-user.**

The box is a personal inference server: one operator, one active session at a time, accessed over Tailscale from wherever the operator is. There is no concurrent multi-user load to budget for. The binding constraint is fitting the served instruct model + embedder + a single session's KV cache in available VRAM.

**Phase 1 (single RTX 3090, 24 GB VRAM):**

The default optimizes for tool-use quality, since the operator's stated use case is opencode / Claude Code against the local endpoint:

| Component | Choice | VRAM |
|---|---|---|
| vLLM-served instruct | Qwen2.5-Coder-32B-Instruct AWQ-int4 | ~19 GB |
| Embedder | BGE-M3 in TEI, **CPU build** | 0 GB (system RAM) |
| **KV cache headroom** | | **~5 GB → ~32K tokens single-session** |

CPU embeddings via TEI on the 9800X3D run at ~50–150 ms per call. Engram's actual call rate is a few embeddings per minute at peak personal use, not thousands, so the latency is invisible. The trade is real: capture latency goes from ~10 ms (GPU TEI) to ~100 ms (CPU TEI), and ~5 GB of KV cache headroom comes back to vLLM. For single-user code-agent work that almost always stays under 32K tokens, this is the right deal.

**Why Coder-32B over a smaller model.** For strong tool use against opencode / Claude Code, model quality at the tool-call schema and multi-step planning level matters more than peak throughput. Qwen2.5-Coder-32B is one of the few open models where tool calling holds up under real agent loops — error recovery, multi-step planning, long tool-result reasoning. A 14B class model is sufficient for Engram's *own* tagger needs but underperforms on the operator's primary use case.

**Tagger cost** [M4+]. A tagger call on a single thought is ~200–500 input tokens → ~50–150 structured output. At Coder-32B's vLLM throughput on a 3090 (≈75 tok/s per stream per the BOM), one thought tags in 1–2 seconds. The drainer runs on the `[worker] tick_interval_seconds` cadence (default 5 s) and processes up to `batch_size` thoughts per tick (default 16). For dev/test, drive a one-shot batch with `engram tag [--scope <s>] [--limit <n>]`. Unlike M2's nightly reflector cron, there's no scheduled cron — the drainer simply catches up as the queue fills.

**Embedder placement is a deployment-time choice, not a code change.** TEI ships CPU and CUDA builds with identical HTTP APIs (`ghcr.io/huggingface/text-embeddings-inference:cpu-1.x` vs `:1.x`). Switching is a systemd unit edit; the Engram TOML doesn't change. CPU is the v0 default; GPU is appropriate later if capture rate grows or the operator wants sub-100ms capture latency for some interaction pattern.

**Phase 2 (dual RTX 3090, 48 GB VRAM):**

Phase 2 is a quality upgrade rather than a necessity-driven one — Phase 1 single-user is genuinely a credible primary daily driver. The upgrade unlocks:

- Qwen2.5-Coder-32B at Q6/Q8 (better quality than AWQ-int4) with full KV cache via tensor-parallel
- 70B-class general models at Q4 (Llama 3.3 70B, Qwen 2.5 72B) for harder reasoning tasks, ~32 tok/s per the BOM
- DeepSeek-V2.5/V3 (235B MoE, ~21B active) at Q4 — explicitly strong at agentic work, ~25 tok/s per the BOM

vLLM's `--tensor-parallel-size 2` is the obvious deployment shape. The embedder either stays on CPU or moves to a single card via TEI's CUDA build; both are easy.

**System RAM and storage.** Postgres + pgvector will be MB-to-low-GB scale even with 100k+ thoughts; the 64 GB system RAM is overprovisioned for Engram's purposes (and is there for vLLM's CPU offload / weights loading anyway). With CPU embedding the embedder also runs out of system RAM — BGE-M3 is ~2 GB resident — well within budget. On the 2 TB NVMe, Engram's footprint is dominated by the database (single-digit GB at realistic scale); vLLM model weights are the actual storage hog.

## 10. Operational shape — what makes the store honest

§6.5 describes the tagging sidecar as a capability — what it produces and how. This section is the operational counterweight: what guarantees the store gives the operator, what it explicitly does *not* guarantee, and why the M4 architecture deliberately stops short of drift-defense ceremony that the M2 facts pipeline carried.

**The five operational properties Engram guarantees:**

1. **Raw thoughts are immutable.** Capture writes a `thoughts` row once and never updates the `content` column. The only mutations on `thoughts` are state flips — `retracted_at` (M3), the `tags` JSONB (M4), and the tagger provenance triplet — none of which touch content. If a derived signal (embedding, tag) drifts, the truth is recoverable: re-embed, re-tag.

2. **Content-fingerprint dedup.** [M4] `thoughts.content_fingerprint` is SHA-256 of `content` with a `UNIQUE` constraint. Capture is `INSERT ... ON CONFLICT (content_fingerprint) DO NOTHING RETURNING id`; duplicate content from any source (same agent retrying, two agents capturing the same observation, an explicit re-capture) collapses to the existing `thought_id`. Agents see a stable id for a given content, and `is_duplicate: true` on the response so they can react to "I already told you this" if they care.

3. **Derived signals are recomputable.** The `embeddings` table is per-model and per-version; a new embedding model is a new partial HNSW index + a re-embed pass — no data loss, no migration on `thoughts`. The `tags` column is overwritten in place by `engram tag --rerun` whenever `[tagger].model_version` advances; no supersede chain to walk, no audit trail to reconcile, because tags are advisory metadata, not load-bearing.

4. **Retraction is durable.** [M3] `retract_thought` sets `retracted_at` and `retracted_reason` on the row. Every retrieval path (`search_thoughts`, `recent_thoughts`) filters `WHERE retracted_at IS NULL`. `get_thought` is the audit path — it returns the row with retraction state visible. The row never leaves the database; the retracted-thought UX is "untrusted but inspectable."

5. **JSONB tag containment is GIN-indexed and additive-by-default.** [M4] `search_thoughts(..., tag_filter)` runs `WHERE tags @> $tag_filter` against the `thoughts_tags_gin` index. Containment is one-sided: a thought with tags `{"kind": "task", "people": ["Sarah", "Bob"]}` matches a filter `{"people": ["Sarah"]}` (the filter is a subset of the row's tags). Operators can build precision queries (`{"kind": "task", "topics": ["release"]}`) or coarse ones (`{"kind": "task"}`) without retrieval-side changes.

**What this architecture deliberately does NOT have, post-M4:**

- **No confidence-band routing.** Tags don't carry confidence. The tagger emits a single object per thought; if it's wrong, re-tagging overwrites it. (The M2 `review_queue_below` / `min_confidence_to_store` machinery is gone with the facts table.)
- **No supersede chain on tagger output.** Tags are overwritten on `engram tag --rerun`. There's no `tags_superseded_by`, no history table, no audit trail on what the tagger said last week. The provenance triplet (`tags_extractor_model`, `tags_extractor_version`, `tags_extracted_at`) tells you what produced the *current* tags; if the operator wants pre-`--rerun` state, they restore from a backup.
- **No fact-review queue.** The `facts_review_queue` table was dropped by migration 0006. Tagger output goes straight onto the row; there's no operator-review gate.
- **No `correct_fact` MCP tool.** Operators who notice a wrong tag don't correct it; they ignore it (tags are advisory) or `retract_thought` if the underlying content is wrong. The cost of being wrong about a tag is small enough that operator-correction infrastructure isn't worth the complexity.
- **No drift-defense `engram audit` job.** M2's audit story keyed on extractor drift across model versions; M4's tags are recomputable from raw text whenever the operator wants. The corresponding `stats` MCP tool ([M7]) surfaces current state, not drift.

**The pre-M4 design (preserved here for history).** M2 shipped a `facts` table with `(subject, predicate, object, confidence, statement)` rows, a reflector cron, a confidence-band review queue, `correct_fact` MCP tool, and a `--rerun` flow with supersede-via-statement-or-triple-match dedup. M3 Phase D dogfood (commits `34ba756` → `2000059` on the m4-collapse-to-thoughts branch tell the full story) revealed that the (S, P, O) abstraction was generating most of the operator-visible failure modes (inverted comparatives, self-referential subjects, conditional-as-subject, predicate verbosity, polarity contradictions, triple-semantic drift). Statements were faithful; triples were brittle. **None of the M2-era drift-defense machinery was the wrong design for a fact store — it was the right design for the wrong abstraction.** M4 swapped the abstraction; the defensive machinery went with it.

## 11. Deployment & ops

**Target hardware:** Phase 1 of the BOM — RTX 3090 (24 GB), Ryzen 7 9800X3D, 64 GB DDR5-6000, 2 TB PCIe 5.0 NVMe, Ubuntu 24.04 LTS, NVIDIA driver 560+, CUDA 12.6+. Postgres 16+ with `pgvector` ≥ 0.7 (HNSW required), `pg_trgm`, `pgcrypto`. Phase 2 (dual 3090) is fully supported by the same software stack with one config change (`CUDA_VISIBLE_DEVICES`).

**Components:**

- `engram` — the single Rust binary. M1 supports `serve` and `migrate` subcommands; `worker` and `embed-backfill` join at M2; `bench` joins at M3; `tag` joins at M4 (replacing M2's `reflect`).
- Postgres 16 with `pgvector` ≥ 0.7, `pg_trgm`, `pgcrypto`. Connection is configured by URL (TOML or `DATABASE_URL` env). Local Unix socket is the simplest deployment; remote TCP — same Tailnet, separate NAS or DB host, or anywhere reachable — is fully supported. **Extensions must be installed on the Postgres server**, not the Engram host. At personal-scale data with HNSW indexes, network round-trip on a LAN adds negligible latency to queries.
- `text-embeddings-inference` HTTP server for BGE-M3, sidecar pattern. **CPU build by default** for v0; swap to CUDA build by changing the systemd unit's container image (no Engram code or config change needed). Required from M1. From M3 onward, TEI also serves the cross-encoder reranker (separate model on the same HTTP shape).
- vLLM serving an instruct model — required from M4 onward when the tagger is configured (no tagger backend in M1). **Operated independently of Engram.** Engram is a client; the operator manages vLLM's lifecycle, model choice, and serving config. Engram only requires the OpenAI-compatible endpoint to be reachable.

**Process model:** systemd units. `engram-server.service` exists from M1. `engram-tei.service` (the embeddings sidecar; CPU build by default — see §9) is required from M1. `engram-worker.service` joins at M2 and from M4 onward runs *two* drainer tasks inside one process: the embed drainer (always on, pulls jobs off `pending_embeddings`) and the tag drainer (M4; pulls off `pending_tags` when `[tagger].provider` is non-empty). vLLM and Postgres run as their own units, managed independently.

**Why two drainers, no cron** [M4+]. The M2 reflector ran on a cron schedule (default `0 0 3 * * *`) to batch fact extraction overnight and avoid contending with the operator's daytime agent loads. M4's tagger is a single chat-completion per thought, runs in the 1–2 s range, and produces 100–300 output tokens — small enough that ticking through `pending_tags` continuously alongside the embed drainer is cheap. No nightly scheduled run, no missed-cron catch-up logic, no time-of-day contention question.

**Backups:** `pg_dump --format=custom` nightly to a separate disk; weekly to a remote (Backblaze B2 or rsync.net). Embeddings are derived data and don't strictly need backing up — re-running `engram embed-backfill` regenerates them — but including them speeds disaster recovery.

**Migrations:** `sqlx migrate`. Schema changes ship with the binary.

**Observability** [M7]. Structured `tracing` logs to journald are present from M1. The Prometheus `/metrics` endpoint exposing capture-rate, search-latency P50/P95/P99, embedding-queue depth, embed/tag failure counts, and queue ages lands at M7.

## 12. Auth & network exposure

Three relevant tiers. They map to milestones, not to deployment options offered all at once.

| Tier | Network | Auth | Milestone | Use case |
|---|---|---|---|---|
| **0 — Localhost** | `127.0.0.1` only | None | M1 | First-run validation; the development default. |
| **1 — Mesh** | Tailscale / WireGuard | None (mesh = auth) | M1 (config change) | Personal devices already on the Tailnet. The ops-recommended endpoint for single-user deployment. |
| **2 — Tunnel** | Cloudflare Tunnel / Caddy + LE | Bearer token | M6 | Non-Tailnet clients (Claude Desktop, ChatGPT) that need a public HTTPS MCP URL. |

A "Tier 3 — public + multi-user" option exists in principle but is **explicitly out of scope** for the current roadmap. It would require OAuth2, per-client tokens, and audit log; implementable later if the system is genuinely shared with another person, which is not a current requirement.

**Tier 1 is the recommended endpoint for single-user deployment.** Engram binds to the Tailnet interface and is reachable as `engram.tailXXXX.ts.net` from every personal device, using the same MagicDNS pattern as vLLM. No code change vs. Tier 0; only the bind address.

**Auth at Tier 2** [M7]. Bearer token validated against a hashed allowlist in `engram_tokens`. Tokens carry a scope-list — a token can be locked to `work.*` and not see `personal.*`. Audit log records `(token_id, tool, args_hash, ts)` for every call.

## 13. Evaluation

[M7] — eval suite ships at the operational-maturity milestone. We don't ship without it because "did the model swap regress retrieval" is the kind of question we'll ask ourselves often.

**Three suites, all reproducible from a fixture corpus:**

1. **Capture-recall.** Synthetic conversations seeded with target thoughts; check that subsequent semantically-relevant queries surface the right thoughts.
2. **Cross-model retrieval consistency.** Re-embed the same fixture with a new embedder; measure overlap of top-10 results vs. baseline. Drop > 30% triggers a manual review before the swap is committed in production scopes.
3. **LongMemEval-style.** Subset of the public benchmark adapted to our schema. Apples-to-apples comparison against published Mem0 / Zep / Letta numbers.

Eval runs end-to-end in `engram eval --suite <name>` and dumps a JSON report.

## 14. Open questions

Resolved during the milestone-roadmap planning conversation (see Revision history):

1. ~~**Inference box specs.**~~ Resolved: Phase 1 RTX 3090 / 9800X3D / 64 GB; Phase 2 adds a second 3090.
2. ~~**v0 scope.**~~ Resolved: see §3.5 milestone roadmap. M1 = capture + hybrid search + MCP. M2 added the facts pipeline; M4 retired it in favor of a tagging sidecar after M3 dogfood demonstrated the (S, P, O) abstraction wasn't earning its complexity.
3. ~~**Search architecture.**~~ Resolved: hybrid (vector ∪ trigram, RRF) at M1; reranker at M3.
4. ~~**Active-embedder mechanism.**~~ Resolved: config-driven `model_id`, one HNSW partial index per model.

Carrying forward:

5. **Naming.** Engram is a placeholder. (Hippocampus, Cortex, Lattice, Mneme are all in the drawer.)
6. **Sync.** Do we ever want multi-machine replication? Logical replication on Postgres is straightforward, but only worth doing if you'll actually use it. Defer.
7. **Capture UX.** OB1's Slack capture is clever. Equivalents: a Telegram bot, a CLI `engram capture`, a Raycast/Alfred extension, a browser extension. Out of scope until at least M6.
8. **Embedding model default.** v0 commits to BGE-M3 (well-established, multilingual, runs in ~1.5 GB, supports rerank). A future milestone should bake off Qwen3-Embedding-4B and Qwen3-Embedding-8B against our own eval fixture before any production-scope re-embed. The embeddings table design (§5) makes this a routine swap rather than a migration.
9. **Are we storing agent transcripts?** Currently artifacts can hold them (M5+); we haven't decided whether agents auto-capture session transcripts on close or whether that's an explicit flush.
10. **Tagger model: dense vs. MoE.** Phase 2 unlocks Qwen3-30B-A3B (MoE, 3B active) as an alternative to Qwen2.5-32B (dense). The MoE option likely wins on throughput; quality on tagger output is unmeasured against M4's v1 prompt. Decide via the eval suite (M6).

## 15. Out of scope (for the foreseeable future)

- Knowledge-graph reasoning (Cognee/Graphiti territory). Retired with the M4 collapse; structured `(S, P, O)` triples were the wrong abstraction for this use case.
- Memory forgetting / TTL policies (everything is forever; pruning is a post-M6 conversation).
- Multi-modal memory (images, audio).
- Federated query across multiple Engram instances.
- A web UI. Postgres + `psql` is the admin interface.
- Public + multi-user deployment ("Tier 3" in §12).

## Revision history

- **2026-05-09** — Initial v0 draft by Claude Desktop in a "technical PM" capacity.
- **2026-05-09** — Revised by engineer + architect after the milestone-roadmap brainstorm. Added §3.5 milestone roadmap. Corrected schema in §5: added `CREATE EXTENSION` lines for `pgcrypto`/`vector`/`pg_trgm`; removed trailing comma in `thoughts`; replaced the `current_setting`-based partial HNSW index (which the Postgres planner rejects, since `current_setting` is `STABLE` not `IMMUTABLE`) with a literal-model partial index (`embeddings_bge_m3_hnsw`); added `thoughts_scope_recent_idx` and `thoughts_content_trgm_idx`; added `target_kind` CHECK on `embeddings`. Reframed §6 (M1 sync embedding via TEI; M2+ async seam), §7 (RRF hybrid; reranker M3), §8 (per-tool milestone column), §9 (Embedder M1, Extractor M2; `CloudEmbedder` added; active-embedder via config). Reframed §12 auth tiers as a milestone progression and dropped Tier 3 from the table. Pruned resolved open questions in §14. Doc now describes the M5-complete terminal state with milestone callouts inline.
- **2026-05-13** — **M2 complete.** Shipped in four phases A–D (see `docs/milestones/m2-progress.md`). Facts pipeline live: async embedding seam (capture enqueues; `engram worker` drains), reflector cron via `tokio-cron-scheduler` 0.15 (default off — opt-in via `[reflector] enabled = true`), `OpenAICompatibleExtractor` covering vLLM and OpenRouter via named-constructor presets, two new MCP tools (`search_facts`, `correct_fact`), `get_thought` now carries active `linked_facts`, and a new `engram reflect` subcommand with `--rerun [--since <RFC3339>]` for re-extracting historical thoughts (idempotent; supersedes on (S,P,O)-match-but-statement-differs; additive only). **Phase D simplification:** `search_facts` ships trigram-only inside an RRF-shaped pipeline — fact embeddings are wired through migration 0001's `target_kind = 'fact'` enum but the worker doesn't yet enqueue facts; the vector leg lands in M3 (search quality) alongside the cross-encoder reranker. **`correct_fact` provenance:** manual rows use the sentinel `extractor_model = "manual"`, `extractor_version = 0`, `source_run_id = NULL`, `confidence = 1.0`. Three-band confidence routing (the "flagged but committed" middle band from §10) is deferred — needs a `flagged` column on `facts` that doesn't exist yet. M2 success criteria #1–#5 met by code; #6 (operator dogfood ≥ 1 week) is the only remaining open item.
- **2026-05-13** — Reconciled doc against shipped M1 + M2 code. §5 schema block extended with migration 0002's three tables (`pending_embeddings`, `reflector_runs`, `facts_review_queue`) and `facts.source_run_id`. §7 `SearchRequest` snippet matches the shipped struct (no `mode: SearchMode` or `include_facts: bool` — neither was implemented); added a `SearchFactsRequest` peer. §8 tool descriptions tightened: `capture` documents the always-pending return; `get_thought` calls out `linked_facts`; `search_facts` notes the trigram-only / M3-vector-leg state; `correct_fact` documents the manual-sentinel provenance and retract-via-no-replacement variant. §9 trait signatures match the code (`Embedder::model() -> &EmbeddingModel`, `Extractor` takes `&Thought`, both return typed errors with `is_transient()`); added the `Fact` read-shape struct; replaced the fictional `TeiEmbedder` / `CloudEmbedder` / `OpenRouterExtractor` defaults with the actual `OpenAICompatibleEmbedder` / `FakeEmbedder` / `OpenAICompatibleExtractor` / `FakeExtractor` set. §9 default-config TOML and §11 process-model paragraph match shipped fields and values (6-field cron, opt-in reflector, single-threshold routing). §10 mechanism #3 reframed as single-band with the three-band band noted as deferred; mechanism #5 documents `--rerun` as additive-only.
- **2026-05-13** — Added §6.5 "Fact extraction pipeline" as the affirmative companion to §10. §6.5 leads with *why facts matter* (the structured-second-layer story: same captures, two queryable surfaces, thought stays source-of-truth), walks the six-step pipeline (open run → walk unfacted thoughts → extract via JSON-Schema-guided decoding → route by confidence → close run → optional operator review/rerun), shows the exact `response_format` JSON Schema, gives a worked example (a casual conversation capture becoming two facts), and ends with operator-facing SQL ("here are the queries that become trivial once you have a facts table"). §10 reframed as the drift-defense counterweight — same content, but explicitly positioned as the defensive complement to §6.5 rather than the only place facts are discussed.
- **2026-05-13** — M3 starter shipped early in response to M2 dogfood: first-class thought retraction. Migration `0003_thoughts_retraction.sql` adds `thoughts.retracted_at` + `thoughts.retracted_reason` and an `(scope, created_at DESC) WHERE retracted_at IS NULL` partial index. New `retract_thought(thought_id, reason?)` MCP tool and `engram-storage::retract_thought` fn atomically (a) sets the trust-state column and (b) auto-supersedes every active fact derived from the thought. All retrieval paths (`recent_thoughts`, `search_trigram`, `search_vector_knn`, `search_facts_trigram`) and reflector paths (`find_unfacted_thoughts`, `find_facted_thoughts`, `enqueue_unembedded_thoughts`) now filter `retracted_at IS NULL`; `get_thought` is the audit path and still returns the row with `retracted_at` / `retracted_reason` exposed on the response. Motivation: M2 dogfood (see `docs/milestones/m2-progress.md` 2026-05-13 history) showed that the previous workaround — retract every derived fact one at a time via `correct_fact` — fails as soon as the operator misses any fact, because the unretracted-thought-with-one-active-fact stays in the reflector's `find_facted_thoughts` set and gets re-extracted on the next `engram reflect --rerun`. The atomic supersede + DB-invariant filter closes that gap. Note: this expands the M3 scope (M3 was originally search-quality only) but the work was pulled in early because it gates honest dogfood — operators iterating on captures will inevitably need a way to mark wrong claims as untrusted. The reranker + fact embeddings remain the rest of M3.
- **2026-05-16** — **M4 collapse to thoughts-only (Path B-OB1).** Major doc revision following the M3 Phase D dogfood negative-knowledge outcome (statements faithful, triples broken across 7 rounds). Roadmap renumbered: M2 (facts pipeline) marked retired-by-M4; M3 closed out at retrieval portion; **M4 = collapse to thoughts-only with metadata-tagging sidecar**; what was M4 (artifacts) shifts to M5; what was M5 (operational maturity) shifts to M6. Schema: `facts`, `facts_review_queue`, `reflector_runs` dropped via migration 0006; `thoughts` extended with `content_fingerprint BYTEA UNIQUE` (SHA-256 dedup), `tags JSONB` (LLM-tagger output: people / action_items / topics / dates_mentioned / kind), and the three `tags_extractor_*` provenance columns. New `pending_tags` queue table feeding a tag drainer in `engram worker`. §6 rewritten as "Ingest path" + "Tagging sidecar"; §10 rewritten as "Operational shape" — operational guarantees rather than drift-defense ceremony (no confidence-band routing, no supersede chain on tagger output, no `facts_review_queue`, no `correct_fact`). §8 MCP surface: `search_facts` and `correct_fact` removed; `search_thoughts` gains `tag_filter`; `capture` gains `is_duplicate`; `retract_thought` simplified (no fact-cascade). §9: `Tagger` trait replaces `Extractor`; config `[tagger]` replaces `[extractor]` + `[reflector]` (drainer is always-on when `[tagger].provider` is non-empty, silent-disable when empty). CLI: `engram reflect` → `engram tag`. The full M4 contract lives in `docs/milestones/m4-spec.md`; architectural narrative in `docs/milestones/m4-collapse-to-thoughts.md`.
- **2026-05-16** — **M4.1 v2 tagging.** Dogfood on the M4 v1 tagger surfaced two patterns ("memory-systems" inference: model already half-distinguished entities from category-inferred terms; Probe 2-style runs: topics were phrase-driven, producing divergent terms across paraphrases of the same concept). M4.1 splits `Tags.topics` into `Tags.entities` (proper-noun-style identifiers mentioned by name) + `Tags.topics` (broader subject categories) and adds an optional `ScopeVocab` parameter to `Tagger::tag()` — the drainer pre-fetches the top-N most-frequent topic + entity terms in the thought's scope (via the new `engram_storage::fetch_scope_vocab` helper) and renders them into the prompt as a controlled-vocabulary section so the model prefers established terms over coining new ones. Tagger version bumps 1→2; operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill. No schema migration (tags is JSONB; the `entities` key is additive). Config: `[tagger]` gains `scope_vocab_enabled` (default `true`) and `scope_vocab_size` (default `50`). MCP wire surface: `SearchHit.tags` carries `entities` for free; `SearchThoughtsArgs.tag_filter` and `SERVER_INSTRUCTIONS` updated to advertise the new shape and document the entities-vs-topics distinction. Selective relations (the M5 candidate) and Probes 1-3 remain deferred. The M4.1 contract + dogfood plan live in `docs/milestones/m4.1-tagging-v2.md`.
- **2026-05-17** — **M4.1 v3 prompt iteration.** Dogfood on the v2 tagger surfaced two failure modes: (1) entities degraded to noun-phrase extraction when no proper nouns were present (the model padded the slot with descriptive phrases like "agent memory protocol", "cross-encoder", "lexical signals" to fill a required field); (2) kind classification drifted across re-tag cycles. A kind-stability diagnostic (N=10 per fixture on six fixtures) showed within-tagger kind is deterministic at temperature 0.2 (10/10 same kind per fixture) but vocab presence shifts the kind prior on bistable content (8a533e15: task→observation at 8/10 with vocab) and other fixtures' stored kinds aren't reachable from current model+vocab state (22bccb3a stored as `reference`; both vocab-off and vocab-on diagnostics produce `observation` 10/10). v3 ships three prompt changes: (a) tightens `entities` definition to "canonical proper names of specific named things ... recognizable as named entities to someone outside the conversation," dropping the v2 "named concepts" phrasing that was the padding vector; (b) adds an explicit `RETURN [] IF THE THOUGHT CONTAINS NO SUCH NAMED ENTITIES` anti-padding rule with a verbatim list of v2-padding negative examples; (c) adds a kind-isolation clause framing kind as intrinsic-shape classification with an explicit Rules-section line forbidding the controlled vocabulary from influencing kind. Schema unchanged. `BUNDLED_TAGGER_VERSION` bumps 2→3; operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill. The kind-stability diagnostics in `crates/engram-extract/src/openai_compatible.rs` (`kind_stability_diagnostic` and `kind_stability_diagnostic_with_vocab`, both gated on `--features integration` and `--ignored`) are operator-runnable for post-v3 verification.
- **2026-05-17** — **M4.1 v4 prompt iteration.** Dogfood on the v3 tagger across 23 v3-tagged thoughts revealed the v3 entities anti-padding fix was half-landed (works for proper-noun-anchored content sometimes, fails for proper-noun-free content) and the v3 negative-example list was *counterproductive* — `047d0ce8` (a proper-noun-free thought) emitted `["agent memory protocol", "embedding-based", "lexical signals", "cross-encoder"]`, four of the seven verbatim items in the v3 prompt's "NOT entities" list, the model substituting them precisely because they appeared as noun phrases in the content. Additionally, scope-aware vocabulary injection began sacrificing topic precision in practice — `74eb781c`, `45cd2001` slot-padded entities even on proper-noun-anchored content (maxItems=5 read as "fill 5" not "up to 5"). v4 ships three prompt changes: (a) restructures the entities description to lead with `entities: default to []` (empty-as-default instead of empty-as-constraint) and replaces the v3 negative-example list with a structural NAME-vs-DESCRIBE test ("does this phrase NAME a specific thing or does the thought DESCRIBE an action using a noun phrase?"); (b) lowers `entities.maxItems` from 5 to 3 to force selectivity; (c) softens the scope-vocabulary section from "prefer the established form ... coin new terms only when genuinely unseen" to "use a vocab term when it accurately describes the thought's subject ... precision over consistency" — vocab moves from dominator to tie-breaker. Kind isolation from v3 retained. `BUNDLED_TAGGER_VERSION` bumps 3→4; operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill. The v3 negative-example list lesson — don't list the phrases you want excluded; the listing itself reinforces them — is itself a noteworthy finding about prompt-engineering this model class.
- **2026-05-17** — **M5 selective relations.** New milestone adding a thought-to-thought graph layer on top of the M4 substrate. Six-relation closed vocabulary (`replaces`, `requires`, `references`, `belongs_to`, `decided_by`, `refines`) — chosen by intuition about conversation-memory structure and validated against the M4.1 dogfood corpus where the citation chain `137dba1d → 6d2ef58e → 8a533e15` was exactly the implicit-`refines` pattern. New `thought_links` table (migration 0007) with `(from, relation, to)` UNIQUE constraint for idempotency, CHECK constraints on the closed vocab + self-link prohibition + `source` enum (`agent`|`tagger`), ON DELETE CASCADE on both endpoint FKs. New core types `RelationKind`, `LinkSource`, `LinkDirection`, `LinkId`, `ThoughtLink`. New storage helpers `insert_link`, `delete_link`, `fetch_related_thoughts` returning `RelatedThought` enrichment rows. New MCP tools `link_thoughts(from, relation, to, note?)` (idempotent on triple; pre-validates endpoint existence with actionable errors for SelfLink / FromThoughtMissing / ToThoughtMissing / NoteTooLong), `unlink_thoughts(from, relation, to)` (idempotent on already-deleted), and `get_related_thoughts(thought_id, relations?, direction?)` (grouped `outbound` + `inbound` arrays with content_preview, retraction state, and edge metadata). `SERVER_INSTRUCTIONS` extended with the relation vocab + tool listing; regression test pins the documentation. Roadmap renumbered: M5 = selective relations, M6 = artifacts (was M5), M7 = operational maturity (was M6); milestone docs renamed via `git mv`. M5 is agent-supplied and thought-to-thought only at this milestone; tagger-extracted relations (M5.x — requires entity resolution) and heterogeneous targets (M5.x or M6 — polymorphic schema work) are explicit deferrals. The M5 milestone doc lives at `docs/milestones/m5-selective-relations.md`.
- **2026-05-17** — **M5.1 — `supports` relation + tool-description anti-patterns.** Day-one M5 dogfood on 17 agent-supplied edges + 2 captured findings revealed `references` was over-firing on what was actually evidence / corroboration. Four functionally distinct edge types collapsed into it: weak prose cite, experimental evidence, summary cite, and sibling grouping. The `note` field disambiguated by inspection but any aggregation tooling that filtered by `relation` couldn't tell evidential support from prose citation. M5.1 adds a seventh relation `supports` to split "I cite for context" (`references`) from "I confirm a claim" (`supports` — experimental evidence, corroborating data, logical support). Migration 0008 is a pure CHECK constraint relax (no data migration needed). Same dogfood also revealed `refines` being misapplied where `references` was correct (the proposed citation chain `6d2ef58e refines 8a533e15` was wrong because the bootstrap is a charter, not a proposition with updated thinking). The `link_thoughts.relation` schemars description gains a "Common mistakes to avoid" decision-tree block flagging the five anti-patterns: don't-use-refines-for-citation, don't-use-belongs_to-for-peer, don't-use-decided_by-without-attribution, don't-use-replaces-for-refinement, and don't-use-references-when-supports-fits. SERVER_INSTRUCTIONS updated with the seven-relation list + the `references`/`supports` distinction; regression test pins the documentation. Heterogeneous targets (the other dogfood-promoted concern — "Probe 2A and 2B are sibling variants under an experiment that isn't a thought") promoted from M5.x to a near-term M5.2 iteration. Tagger-extracted relations remain M5.x. The v1 dogfood lesson — closed-vocab relation design needs the same anti-pattern documentation discipline as closed-enum kind classification (cf. the M4.1 v3 negative-example backfire) — is itself a noteworthy methodology finding.

- **2026-05-17** — **M6 reshape + ship.** Original M6 (artifacts / long-form document ingestion) was dropped after a same-day live-corpus measurement: 12 MB DB / 41 thoughts / ~1.5 KB user-data-per-thought / 15-20× index amplification — engram sits in a high-signal-density sweet spot between transcripts and tags, and arbitrary long-form ingestion would dilute that. M5.2 already shipped `to_url` link targets covering the "this thought references that external doc" case. New M6 = corpus stats CLI + tagger-extracted relations v1. **M6.0:** new top-level `engram stats` subcommand prints thought counts (live/retracted/untagged), content/tags/metadata byte totals, embeddings by model + dims, link counts (by relation / kind / source), queue depths, per-scope summary, per-table `pg_relation_size` / `pg_indexes_size` / `pg_total_relation_size`, and `pg_database_size`. Plain-println rendering mirroring `engram audit migrations` style; `humanize_bytes` helper at 1024-base. New storage helper `corpus_stats(pool, scope_prefix)` returning a `CorpusStats` struct (six SQL round-trips, the last via `sqlx::query()` runtime-checked against `pg_class`). CLI-only for v1 — no MCP `stats` tool. **M6.1:** v5 tagger prompt + JSON schema add a `relations` field. Tagger emits closed-vocabulary `(relation, to_kind, to_value, note?)` triples from explicit relational claims in prose; `to_kind ∈ {url, entity, person}` (no thought targets — entity resolution is the deferred sub-problem). Schema enforces `maxItems: 5`, closed enums on `relation` (7 values) and `to_kind` (3 values). `BUNDLED_TAGGER_VERSION` bumps 4→5; operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill. Drainer wiring: new `engram_mcp::apply_tagger_relations` helper soft-deletes prior `source=tagger` edges from the thought via the new `soft_delete_tagger_edges_for_thought` storage helper, then `insert_link`s each emission with `source = tagger`. Validation reuses `link::validate_target` (visibility bumped `fn` → `pub(crate) fn`). Bypass-on-error in the drainer — a single malformed emission (non-http URL, empty entity name) is logged and skipped, never fails the whole tag job. `run_tag` CLI mirrors the loop for synchronous re-tag runs. Core types: `Tags.relations: Vec<ExtractedRelation>` (serde-default empty for v1-v4 backward compat), new `ExtractedRelation { relation, target, note? }` + `ExtractedTarget` enum (`Entity | Person | Url`). No schema migration — M5.2 polymorphic targets + `LinkSource::Tagger` + soft-delete already covered the data shape. The original M6 (artifacts) plan is archived in `docs/milestones/m6-artifacts.md` (with a SUPERSEDED header); new milestone doc at `docs/milestones/m6-stats-and-tagger-relations.md`. 334 tests passing post-M6.

- **2026-05-17** — **v6 tagger prompt iteration.** A 2026-05-17 dogfood pass on the v5 tagger (post-M6.1 corpus-wide re-tag) surfaced three regressions: (1) kind classification collapsed to `observation` 17/17 across the `engram.m3.dogfood` scope, including mission/charter statements that should have been `task`, definitional thoughts that should have been `reference`, and finding-shaped thoughts that should have been `idea` — the closed 6-value kind enum was empirically reduced to 1 in practice; (2) the entity field regressed on world-knowledge hallucination (`63ad01e0` Probe 2A extracted `pg_trgm` from prose containing only "trigram retrieval"); (3) the entity field regressed on adjectival miscategorization (`047d0ce8` Probe 2B extracted `embedding-based` and `lexical signals` as entities — the same v3 regression class v4 was supposed to close). Plus a fourth dogfood observation that v6 addresses preemptively: 2/2 URL emissions in `relations` were rejected by the app-side validator for missing `http(s)://` prefix. v6 ships four prompt changes: (a) kind reframed as a 5-step decision tree (define/point-at → reference|person_note; commit-to-action → task; propose/hypothesize/report → idea; narrate-current-activity → session; otherwise → observation), with explicit "observation is the CATCHALL, not the default" anti-default framing and worked examples for each step — the model now walks the tree in order, only the catchall lands on observation; (b) entities gain a "Surface-only rule (load-bearing)" with explicit "Do NOT infer entities from world knowledge" and `pg_trgm` cited as the failure case, plus a "re-read the thought and verify each entity appears in the prose" final-pass instruction; (c) adjectival re-tightening via pattern-based negative examples (adjectives, descriptive noun phrases) rather than literal phrase lists (preserving the v3→v4 lesson — listing forbidden phrases backfires); (d) URL emission tightening: "to_value MUST start with `http://` or `https://`" with the `arxiv.org/abs/...` partial-URL case as the failure example. Structural tweaks: kind reordered to sit next to entities (the two highest-signal classification fields adjacent); relations block shortened to free attention budget; closing "Before you emit" final-pass review section. Schema unchanged. `BUNDLED_TAGGER_VERSION` bumps 5→6; `TaggerConfig::default().model_version` likewise; operator runs `engram tag --rerun --since 1970-01-01T00:00:00Z` to backfill. The `kind_stability_diagnostic*` integration tests gain a 7th fixture (`63ad01e0`) and capture+print entities alongside kind so the dogfood failure cases are visible in one run. A separate dogfood finding — `tag_filter` silently ignored — was investigated and proven a false positive (orchestrator filters correctly against live corpus); tracing instrumentation on the tag_filter retain step makes future false claims diagnosable from server logs. The methodology lesson — v3 negative-example lists backfired, v6 uses structural patterns instead — is itself a noteworthy finding about prompt-engineering this model class.

- **2026-05-17** — **v7 tagger prompt iteration + JSON schema concrete-type fix.** A second dogfood pass on the WIP v6 prompt (before commit) confirmed the entities section regressed on `047d0ce8`: `["agent memory protocol", "embedding-based", "lexical signals"]` — same shape as the v5 dogfood failure. Root cause: v6 had added a "Patterns that are NOT entities" block listing adjectival phrases as examples (e.g., `embedding-based` shown as an adjectival modifier failure), repeating the v3→v4 backfire — listing forbidden phrases reinforces them. v7 drops the entire NOT-entities block; the entities section now contains only the lead-with-empty framing (v4), the surface-only rule citing `pg_trgm` hallucination (v6), the NAME-vs-DESCRIBE structural test (v4), and the final-pass re-read verification (v6). Positive examples (`engram`, `pgvector`, ...) retained — they reinforce desired behavior. v7 also explicitly states topics-as-concept-mapping intent ("Topics map prose to canonical subject categories — they may be inferred from context when the subject is clear ... This is concept-mapping behavior, not surface-lexeme lifting"), which had been de-facto behavior since v4 vocab-softening but wasn't documented; four corpus findings (`6d2ef58e`, `74eb781c`, `137dba1d`, `ce83b7ba`) claim phrase-driven topics extraction with empirical support that is now empirically false (Probe 2 disjoint-vocab pair has 2/3 topic overlap). Operator action: retract-and-replace those four findings at leisure. **Bundled JSON-schema concrete-type fix:** the same dogfood pass surfaced that `tag_filter` and `metadata` were being stripped by claude.ai's MCP client before reaching engram. Root cause: both fields were typed `Option<serde_json::Value>` in the rmcp tool-args structs, which schemars renders without a concrete `type` field in the published JSON schema. Strict client-side validators drop fields they cannot match to a declared type. Wire-tested with raw curl: orchestrator filters correctly when field arrives. Audited via claude.ai client: field never arrives. Fix: change Rust types to `Option<serde_json::Map<String, serde_json::Value>>` (semantically a tightening; both fields were always supposed to be objects) so schemars renders `type: ["object", "null"]`. Boundary conversion to `Value::Object` at the orchestrator call site keeps the inner API unchanged. New regression test `tool_args_object_fields_have_concrete_schema_type` (engram-mcp/src/server.rs) pins the schema shape — any regression to `Option<Value>` fails CI before ship. `BUNDLED_TAGGER_VERSION` bumps 6→7; `TaggerConfig::default().model_version` likewise. Operator workflow after pull: (a) restart `engram serve` to apply the schema fix; (b) `engram tag --rerun --since 1970-01-01T00:00:00Z` to refresh the WIP-v6-tagged corpus under v7; (c) re-audit `search_thoughts(tag_filter={"entities": ["embedding-based"]})` should return 0 results. The v3→v4 lesson is recorded twice in revision history now (v3→v4, and v6→v7); future tagger iterations should default to the v4/v7 pattern (structural questions only, zero phrase hints for what to exclude). The diagnostic doc `tag_filter-strip-diagnostic.md` (operator-supplied, not committed) is the canonical record of the schema-strip bracketing.

- **2026-05-18** — **v8 entities adjectival regression: accept + document (no prompt iteration).** The four-iteration prompt arc on entities (v3 negative-example list → v4 NAME-vs-DESCRIBE structural test → v6 pattern descriptions → v7 pure structural framing, zero phrase hints) hit diminishing returns on `047d0ce8`: at `tags_extractor_version: 7` the thought still emits entities `["agent memory protocol", "embedding-based", "lexical signals"]`. Diagnosis: the NAME-vs-DESCRIBE test asks the model to verify a fact it can't reliably verify — "does this phrase NAME a specific thing that has its own canonical identity outside this thought" requires world-knowledge the 30B model lacks reliable access to. When uncertain, the model defaults to "include," producing over-extraction on technical-prose noun phrases. Three reinforcing model-level biases compound this: (a) surface-pattern over-generalization from pre-training on entity-extraction tasks where multi-word noun phrases were correctly extracted; (b) definite-article-as-name signal ("The X provides..." reads as a definition); (c) coordination spillover (once one of `embedding-based` or `lexical signals` is included, the other follows by parallel structure). Lowering `entities.maxItems` (3→2) was rejected — drops legitimate 3-entity cases as collateral. v8 ships zero prompt changes and zero code changes; instead it documents the structural ceiling and ratifies operator-correction as the design intent. Engram's M5+ machinery (soft-delete on `thought_links`, `link_source` discriminator on `get_related_thoughts` responses, `unlink_thoughts` MCP tool) was designed for operator correction of tagger output — the imperfect tagger feeding into that correction layer is by design, not a bug. **Methodology lesson** (worth keeping for future prompt iterations on any closed-vocabulary or surface-discrimination LLM task): when prompt-engineering hits a structural ceiling — i.e., the prompt asks the model to verify a fact (world existence, factuality, ground truth) the model can't reliably check — further prompt iterations have diminishing returns. The next lever is architectural: closed vocabulary (model becomes classifier over a known set rather than free-form extractor), two-pass verification (verify pass after emission), or model swap (larger or specialized model with better world-knowledge or instruction-following). If continued dogfood reveals the residual entities imperfection is intolerable, escalate to closed-vocabulary mode (the `tagger.scope_vocab` already exists as tie-breaker; promote to gate) or model swap. `BUNDLED_TAGGER_VERSION` stays at 7 — no prompt change, no re-tag needed.

- **2026-05-18** — **Rename + relocate.** The design doc moved from `docs/engram-design-v0.md` to `DESIGN.md` at the project root, joining `README.md` / `DEVELOPMENT.md` / `AGENTS.md` / `CLAUDE.md` as a top-level operator-facing document. The `v0` filename had outlived its purpose — this document is the canonical record of the design as it stands now, with the revision history below capturing past architectural decisions. Cross-references in `README.md`, `DEVELOPMENT.md`, `CLAUDE.md`, `docker-compose.yml`, `migrations/0001_initial.sql`, `crates/engram-extract/src/openai_compatible.rs`, and the milestone docs in `docs/milestones/` were updated; the doc body itself is unchanged from the consistency pass below.

- **2026-05-18** — **Doc-only consistency pass.** Section bodies were brought back into alignment with the M5.2/M6.1/v9 reality. §3.5 line 48: clarified that the `artifacts` / `artifact_chunks` tables remain on disk but are inert pending a future drop migration (M6 reshape dropped the artifacts plan). §3.5 "Order rationale" tail: removed the obsolete "M6 = artifacts" framing. §5 schema block: dropped the `artifacts` / `artifact_chunks` `CREATE TABLE` snippets in favour of a one-paragraph note about the inert tables — the doc represents the *design* as it stands today and points at the cleanup migration. §6.5: bumped "currently v3" to "currently v7"; added the `relations` field to the tagger JSON-schema example and a one-line note about the M6.1/v9 split into `TagOutput { tags, relations }`. §6.6: replaced the M5-era `thought_links` SQL with the current M5.2 schema (polymorphic `to_kind` + per-kind columns + generated `to_value` + `deleted_at` + partial unique index `WHERE deleted_at IS NULL`); rewrote the pipeline paragraph (three-way `UnlinkStatus`, polymorphic `link_thoughts`, `link_source` on every edge); reworked the worked example to span thought + URL targets and show the soft-delete lifecycle; deleted the redundant M5.2-summary paragraph (now woven into the body); pruned the out-of-scope list (tagger-extracted relations moved to §6.7). New §6.7 "Tagger-extracted relations (M6.1)" — covers why the tagger writes graph rows, schema reuse, the drainer pipeline (`apply_tagger_relations` + `soft_delete_tagger_edges_for_thought`), why the JSONB persistence path was dropped in v9, and the agent-facing discriminator (`link_source`). §8: removed the `ingest_artifact` row (artifacts dropped) and the `stats` row (CLI-only at M6.0, no MCP tool); added an operator-only-surfaces note covering `engram stats` + `engram audit migrations`; expanded `get_related_thoughts` description to surface `link_source`. §9: trimmed the Rust data-shape struct snippets (`Tags`, `ScopeVocab`, `RelationKind`, `ThoughtLink`) — the SQL forms in §5 and §6.6 are now the contract, with one line pointing at `crates/engram-core`; updated the `Tagger` trait signature to return `TagOutput` with an inline note. §9 config TOML: bumped `[tagger].model_version` 1 → 7, added `scope_vocab_enabled` / `scope_vocab_size`, added `[server].allowed_hosts` with a one-line explainer pointing at §12. No code changes, no migrations.

- **2026-05-18** — **Drop `tags.relations` from persisted JSONB; thought_links becomes the single canonical store for the link graph.** Dogfood thought `b533ebac` raised a naming-collision concern: engram used the word "relations" for two distinct things — the link graph (`thought_links`, `link_thoughts`, `get_related_thoughts`) and the tagger's `tags.relations` JSONB field. Investigation against the live corpus confirmed that the M6.1 drainer (`apply_tagger_relations`) was writing tagger emissions to BOTH stores: thoughts.tags.relations (raw frozen emission JSONB, preserving duplicates) AND thought_links rows with source='tagger' (deduplicated queryable graph). Live evidence: thought `15533025` had 3 entries in tags.relations and 3 corresponding thought_links rows; thought `b533ebac` had 2 duplicate entries in tags.relations and 1 thought_links row (deduped by the partial unique index). The duplication was pure DRY violation — every persisted tags.relations entry had a corresponding thought_links row. Operator pushback against "accept-the-duplication-because-low-scale": engram is planned OSS, larger-scale operators will hit this, and the duplication doesn't provide better data. **Resolution:** drop `tags.relations` from the persisted JSONB; thought_links is the single canonical store. Engineering changes (all in one ship): (a) Move `ExtractedRelation` + `ExtractedTarget` from engram-core/tags.rs to engram-core/tagger.rs (they're tagger-output shape, not Tags shape). (b) Add `TagOutput { tags: Tags, relations: Vec<ExtractedRelation> }` to engram-core/tagger.rs; `Tagger::tag` returns `TagOutput`. (c) `Tags` struct loses the `relations` field; the `tags` JSONB column written by `update_thought_tags` is automatically narrower. (d) `OpenAICompatibleTagger::tag` parses the LLM response into a transient `TaggerResponseDoc` struct via serde flatten, splitting into the TagOutput shape. The LLM-side `tags_response_format()` schema is **unchanged** — the LLM still emits a top-level `relations` field; only the Rust-side persistence shape changes. (e) Drainer + CLI destructure TagOutput: `update_thought_tags(&output.tags, ...)` then `apply_tagger_relations(&output.relations, ...)`. (f) `FakeTagger::with_canned` keeps its `(tags: Tags)` shape (convenience for the common case); new `with_canned_output(output: TagOutput)` for tests needing both. **Migration `0011_drop_tags_relations.sql`** removes the JSONB key from existing rows: `UPDATE thoughts SET tags = tags - 'relations' WHERE tags ? 'relations'`; CTE captures the rows_touched count for the audit row. Applied 2026-05-18 against the live corpus; 45 rows touched. No tagger version bump — the LLM emission content is unchanged, only the Rust-side persistence path changes. The naming collision is resolved entirely: only `thought_links` / `link_thoughts` / `get_related_thoughts` exist for relations; the `tags` JSONB is metadata-only. AGENTS.md updated to reflect the new mental model. The methodology lesson: when two persistence paths exist for the same data, the DRY violation is the load-bearing concern even at small scale — the cost-now framing ignores the OSS-future + reader-confusion costs that matter long-term.

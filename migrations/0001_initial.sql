-- Engram initial schema.
-- Mirrors design doc §5 (DESIGN.md). Future migrations add
-- per-model HNSW partial indexes when new embedders are introduced.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Raw, immutable captures. Single source of truth.
CREATE TABLE thoughts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope           TEXT NOT NULL DEFAULT 'global',
    content         TEXT NOT NULL,
    source          TEXT NOT NULL,           -- 'manual', 'agent:claude-code', 'reflector', etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX thoughts_scope_recent_idx
    ON thoughts (scope, created_at DESC);
CREATE INDEX thoughts_content_trgm_idx
    ON thoughts USING gin (content gin_trgm_ops);

-- Long-form content. Reserved for M4.
CREATE TABLE artifacts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope           TEXT NOT NULL DEFAULT 'global',
    kind            TEXT NOT NULL,           -- 'document'|'transcript'|'code'|'web'|...
    title           TEXT,
    content_uri     TEXT,                    -- file:// or s3:// for blobs
    content_text    TEXT,                    -- inline if small
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE artifact_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id     UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    UNIQUE (artifact_id, chunk_index)
);

-- Embeddings are first-class. Multiple per target during model migration.
CREATE TABLE embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind     TEXT NOT NULL CHECK (target_kind IN ('thought','artifact_chunk','fact')),
    target_id       UUID NOT NULL,
    model_id        TEXT NOT NULL,           -- e.g. 'qwen3-embedding'
    model_version   INT NOT NULL DEFAULT 1,
    vector          vector(4096) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (target_kind, target_id, model_id, model_version)
);

-- Fast-path 4096-dim MVP note:
-- pgvector 0.8.2 rejects HNSW/IVFFlat indexes on vector columns above 2000
-- dimensions, and halfvec HNSW caps at 4000 dimensions. The first
-- Telegram->kEngram flow keeps exact 4096 vectors unindexed and relies on
-- model/status btree filters while corpus size is small. A production ANN
-- path needs a deliberate follow-up: projection, lower-dim embedder,
-- binary quantization, or query rewrite to a supported pgvector type.
CREATE INDEX embeddings_model_target_idx
    ON embeddings (model_id, target_kind, target_id);

-- Reserved for M2.
CREATE TABLE facts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope               TEXT NOT NULL,
    statement           TEXT NOT NULL,       -- natural-language fact
    subject             TEXT,                -- optional structured triple
    predicate           TEXT,
    object              TEXT,
    source_thought_id   UUID REFERENCES thoughts(id) ON DELETE CASCADE,
    source_chunk_id     UUID REFERENCES artifact_chunks(id) ON DELETE CASCADE,
    extractor_model     TEXT NOT NULL,
    extractor_version   INT NOT NULL,
    confidence          REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    superseded_by       UUID REFERENCES facts(id),
    superseded_at       TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (source_thought_id IS NOT NULL OR source_chunk_id IS NOT NULL)
);

CREATE INDEX facts_active_idx
    ON facts (scope, created_at DESC)
    WHERE superseded_at IS NULL;

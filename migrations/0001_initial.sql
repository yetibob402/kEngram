-- Engram initial schema.
-- Mirrors design doc §5 (docs/engram-design-v0.md). Future migrations add
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
    model_id        TEXT NOT NULL,           -- e.g. 'bge-m3:1024'
    model_version   INT NOT NULL DEFAULT 1,
    vector          vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (target_kind, target_id, model_id, model_version)
);

-- One HNSW partial index per active embedding model. M1 ships this one.
-- Adding a new model = a future migration adds a new partial index over
-- the same table; old rows stay; the active-model concept lives in config
-- (see design doc §9), not in a Postgres GUC.
CREATE INDEX embeddings_bge_m3_hnsw
    ON embeddings USING hnsw (vector vector_cosine_ops)
    WHERE model_id = 'bge-m3:1024';

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

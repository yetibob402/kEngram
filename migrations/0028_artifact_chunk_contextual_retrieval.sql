-- Stage 6 contextual retrieval data-prep sidecars.
-- Additive only: raw artifact_chunks.content and existing raw chunk embeddings
-- remain immutable. Serving stays default-off behind full-pipeline flags.

CREATE EXTENSION IF NOT EXISTS vector;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

CREATE TABLE IF NOT EXISTS artifact_chunk_contexts (
    id                             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id                       UUID NOT NULL REFERENCES artifact_chunks(id) ON DELETE CASCADE,
    source_thought_id              UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    context_text                   TEXT NOT NULL DEFAULT '',
    contextual_content             TEXT NOT NULL DEFAULT '',
    raw_chunk_fingerprint          BYTEA NOT NULL,
    contextual_content_fingerprint BYTEA NOT NULL,
    generator_id                   TEXT NOT NULL,
    generator_version              INT NOT NULL,
    prompt_version                 TEXT NOT NULL,
    prompt_hash                    TEXT NOT NULL,
    model_id                       TEXT NOT NULL,
    model_version                  TEXT NOT NULL,
    contamination_filter_version   TEXT NOT NULL,
    pipeline_run_id                UUID REFERENCES corpus_pipeline_runs(id),
    status                         TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready', 'rejected')),
    rejection_reason               TEXT,
    metadata                       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retracted_at                   TIMESTAMPTZ,
    CHECK (
        (status = 'ready' AND context_text <> '' AND contextual_content <> '' AND rejection_reason IS NULL)
        OR
        (status = 'rejected' AND rejection_reason IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS artifact_chunk_contexts_active_unique_idx
    ON artifact_chunk_contexts (
        chunk_id,
        generator_id,
        generator_version,
        prompt_hash,
        raw_chunk_fingerprint
    )
    WHERE retracted_at IS NULL;

CREATE INDEX IF NOT EXISTS artifact_chunk_contexts_chunk_idx
    ON artifact_chunk_contexts (chunk_id)
    WHERE retracted_at IS NULL;

CREATE INDEX IF NOT EXISTS artifact_chunk_contexts_source_thought_idx
    ON artifact_chunk_contexts (source_thought_id)
    WHERE retracted_at IS NULL;

CREATE INDEX IF NOT EXISTS artifact_chunk_contexts_ready_fts_idx
    ON artifact_chunk_contexts
    USING gin (to_tsvector('english', contextual_content))
    WHERE status = 'ready' AND retracted_at IS NULL;

CREATE TABLE IF NOT EXISTS artifact_chunk_context_embeddings_bge_m3 (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_id      UUID NOT NULL REFERENCES artifact_chunk_contexts(id) ON DELETE CASCADE,
    model_id        TEXT NOT NULL CHECK (model_id = 'bge-m3:1024'),
    model_version   INT NOT NULL DEFAULT 1 CHECK (model_version = 1),
    dimensions      INT NOT NULL DEFAULT 1024 CHECK (dimensions = 1024),
    embedding       vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (context_id, model_id, model_version)
);

CREATE INDEX IF NOT EXISTS artifact_chunk_context_embeddings_bge_m3_model_idx
    ON artifact_chunk_context_embeddings_bge_m3 (model_id, model_version, context_id);

CREATE INDEX IF NOT EXISTS artifact_chunk_context_embeddings_bge_m3_hnsw
    ON artifact_chunk_context_embeddings_bge_m3
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 100);

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0028_artifact_chunk_contextual_retrieval',
    0,
    'Added additive contextual chunk sidecars plus contextual BGE-M3 dense and FTS indexes. No raw chunk content or existing embedding rows are mutated.'
);

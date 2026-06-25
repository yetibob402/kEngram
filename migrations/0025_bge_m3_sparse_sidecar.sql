-- BGE-M3 sparse lexical sidecars.
-- Data-prep only: no serving path changes and no default retrieval change.

CREATE EXTENSION IF NOT EXISTS vector;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type WHERE typname = 'sparsevec'
    ) THEN
        RAISE EXCEPTION 'pgvector sparsevec type is required for BGE-M3 sparse sidecars';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_opclass WHERE opcname = 'sparsevec_ip_ops'
    ) THEN
        RAISE EXCEPTION 'pgvector sparsevec_ip_ops opclass is required for BGE-M3 sparse sidecars';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS thought_sparse_embeddings_bge_m3 (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thought_id           UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    model_id             TEXT NOT NULL CHECK (model_id = 'bge-m3:sparse'),
    model_version        INT NOT NULL DEFAULT 1 CHECK (model_version = 1),
    source_model         TEXT NOT NULL DEFAULT 'BAAI/bge-m3',
    vocab_size           INT NOT NULL DEFAULT 250002 CHECK (vocab_size = 250002),
    nonzero_count        INT NOT NULL CHECK (nonzero_count > 0 AND nonzero_count <= vocab_size),
    content_fingerprint  BYTEA NOT NULL,
    source_content_chars INT NOT NULL CHECK (source_content_chars >= 0),
    generator            TEXT NOT NULL,
    generator_version    TEXT NOT NULL,
    pipeline_run_id      UUID REFERENCES corpus_pipeline_runs(id),
    producer_metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding            sparsevec(250002) NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (thought_id, model_id, model_version)
);

CREATE TABLE IF NOT EXISTS artifact_chunk_sparse_embeddings_bge_m3 (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id             UUID NOT NULL REFERENCES artifact_chunks(id) ON DELETE CASCADE,
    model_id             TEXT NOT NULL CHECK (model_id = 'bge-m3:sparse'),
    model_version        INT NOT NULL DEFAULT 1 CHECK (model_version = 1),
    source_model         TEXT NOT NULL DEFAULT 'BAAI/bge-m3',
    vocab_size           INT NOT NULL DEFAULT 250002 CHECK (vocab_size = 250002),
    nonzero_count        INT NOT NULL CHECK (nonzero_count > 0 AND nonzero_count <= vocab_size),
    content_fingerprint  BYTEA NOT NULL,
    source_content_chars INT NOT NULL CHECK (source_content_chars >= 0),
    generator            TEXT NOT NULL,
    generator_version    TEXT NOT NULL,
    pipeline_run_id      UUID REFERENCES corpus_pipeline_runs(id),
    producer_metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding            sparsevec(250002) NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (chunk_id, model_id, model_version)
);

CREATE INDEX IF NOT EXISTS thought_sparse_embeddings_bge_m3_model_idx
    ON thought_sparse_embeddings_bge_m3 (model_id, model_version, thought_id);

CREATE INDEX IF NOT EXISTS artifact_chunk_sparse_embeddings_bge_m3_model_idx
    ON artifact_chunk_sparse_embeddings_bge_m3 (model_id, model_version, chunk_id);

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0025_bge_m3_sparse_sidecar',
    0,
    'Added additive BGE-M3 sparsevec sidecars for thoughts and artifact chunks. HNSW indexes are isolated in follow-up no-transaction migrations.'
);

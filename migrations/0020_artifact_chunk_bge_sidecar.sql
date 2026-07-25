-- Direct bge-m3 sidecar table for artifact chunks. Index builds are isolated
-- in follow-up no-transaction migrations so CREATE INDEX CONCURRENTLY is never
-- wrapped in a transaction.

CREATE EXTENSION IF NOT EXISTS vector;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

CREATE TABLE IF NOT EXISTS artifact_chunk_embeddings_bge_m3 (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL REFERENCES artifact_chunks(id) ON DELETE CASCADE,
    model_id        TEXT NOT NULL CHECK (model_id = 'bge-m3:1024'),
    model_version   INT NOT NULL DEFAULT 1 CHECK (model_version = 1),
    dimensions      INT NOT NULL DEFAULT 1024 CHECK (dimensions = 1024),
    embedding       vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (chunk_id, model_id, model_version)
);

CREATE INDEX IF NOT EXISTS artifact_chunk_embeddings_bge_m3_model_idx
    ON artifact_chunk_embeddings_bge_m3 (model_id, model_version, chunk_id);

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0020_artifact_chunk_bge_sidecar',
    0,
    'Added typed vector(1024) bge-m3 sidecar for artifact chunks. HNSW and FTS indexes are isolated in follow-up migrations.'
);

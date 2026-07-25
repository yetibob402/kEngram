-- no-transaction

SET lock_timeout = '5s';
SET statement_timeout = '30min';

DROP INDEX CONCURRENTLY IF EXISTS artifact_chunks_content_fts_idx;
DROP INDEX CONCURRENTLY IF EXISTS artifact_chunk_embeddings_bge_m3_hnsw;
DROP INDEX IF EXISTS artifact_chunk_embeddings_bge_m3_model_idx;
DROP TABLE IF EXISTS artifact_chunk_embeddings_bge_m3;

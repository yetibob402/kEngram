-- no-transaction

CREATE INDEX CONCURRENTLY IF NOT EXISTS artifact_chunk_embeddings_bge_m3_hnsw
    ON artifact_chunk_embeddings_bge_m3
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 100)
    WHERE model_id = 'bge-m3:1024'
      AND model_version = 1;

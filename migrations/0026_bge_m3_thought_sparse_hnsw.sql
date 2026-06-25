-- no-transaction

CREATE INDEX CONCURRENTLY IF NOT EXISTS thought_sparse_embeddings_bge_m3_hnsw
    ON thought_sparse_embeddings_bge_m3
    USING hnsw (embedding sparsevec_ip_ops)
    WITH (m = 16, ef_construction = 100)
    WHERE model_id = 'bge-m3:sparse'
      AND model_version = 1;

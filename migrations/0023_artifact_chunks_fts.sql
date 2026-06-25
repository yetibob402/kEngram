-- no-transaction

CREATE INDEX CONCURRENTLY IF NOT EXISTS artifact_chunks_content_fts_idx
    ON artifact_chunks
    USING gin (to_tsvector('english', content))
    WHERE retracted_at IS NULL;

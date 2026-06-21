-- no-transaction

-- Fast lexical retrieval for hybrid search.
--
-- The old pg_trgm similarity leg was too non-selective on large legacy blobs
-- and forced multi-second scans. Postgres FTS gives the lexical leg an
-- inverted index while keeping raw thought content as the source of truth.

CREATE INDEX CONCURRENTLY IF NOT EXISTS thoughts_content_fts_idx
    ON thoughts
    USING gin (to_tsvector('english', content))
    WHERE retracted_at IS NULL;

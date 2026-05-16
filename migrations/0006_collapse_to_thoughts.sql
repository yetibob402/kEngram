-- M4: collapse facts pipeline to thoughts-only with metadata-tagging sidecar.
-- See docs/milestones/m4-collapse-to-thoughts.md for the architectural rationale.

-- 1. Drop the facts pipeline tables.
DROP TABLE IF EXISTS facts_review_queue CASCADE;
DROP TABLE IF EXISTS reflector_runs CASCADE;
DROP TABLE IF EXISTS facts CASCADE;

-- 2. Clean up fact-targeted rows in shared tables.
DELETE FROM embeddings WHERE target_kind = 'fact';
DELETE FROM pending_embeddings WHERE target_kind = 'fact';

-- 3. Extend thoughts with content-fingerprint dedup + tags sidecar.
ALTER TABLE thoughts
    ADD COLUMN content_fingerprint BYTEA,
    ADD COLUMN tags JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN tags_extractor_model TEXT,
    ADD COLUMN tags_extractor_version INT,
    ADD COLUMN tags_extracted_at TIMESTAMPTZ;

-- 4. Backfill content_fingerprint for existing thoughts. `digest(...)` is
-- provided by pgcrypto (enabled in migration 0001).
UPDATE thoughts
SET content_fingerprint = digest(content, 'sha256')
WHERE content_fingerprint IS NULL;

-- 5. Lock content_fingerprint NOT NULL + UNIQUE post-backfill.
ALTER TABLE thoughts
    ALTER COLUMN content_fingerprint SET NOT NULL,
    ADD CONSTRAINT thoughts_content_fingerprint_unique UNIQUE (content_fingerprint);

-- 6. GIN index on tags JSONB for containment queries.
CREATE INDEX thoughts_tags_gin ON thoughts USING gin (tags);

-- 7. Queue table for the tag drainer (mirrors pending_embeddings shape).
-- thought_id is the primary key so re-enqueueing for the same thought is
-- idempotent (ON CONFLICT (thought_id) DO NOTHING).
CREATE TABLE pending_tags (
    thought_id UUID PRIMARY KEY REFERENCES thoughts(id) ON DELETE CASCADE,
    tagger_model_id TEXT NOT NULL,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    attempts INT NOT NULL DEFAULT 0
);

-- Note: the `'fact'` value in the target_kind enum/CHECK constraint is
-- deliberately NOT removed. Leaving it lets us re-add the facts table
-- without another schema migration if Path B-OB1 ever proves insufficient.

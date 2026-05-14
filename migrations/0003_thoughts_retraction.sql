-- M3 prep: first-class thought retraction.
--
-- The M2 dogfood (2026-05-13) surfaced the gap that the design doc's
-- "raw thoughts are immutable" claim ran headfirst into: thoughts can be
-- *wrong*, and the operator needs a way to mark them untrusted without
-- (a) deleting the row (we want the audit trail) or (b) retracting each
-- derived fact individually (the previous workaround, which fails as
-- soon as the operator misses any fact and a subsequent
-- `engram reflect --rerun` re-extracts from the still-untrusted source).
--
-- The fix is a soft-state column on `thoughts`. Immutability of the
-- claim content (`content`, `scope`, `source`, `metadata`, `created_at`)
-- is preserved; we add a separate trust-state marker. The reflector and
-- retrieval paths now filter `WHERE retracted_at IS NULL` so retracted
-- thoughts are invisible to extraction and to search.

ALTER TABLE thoughts
    ADD COLUMN retracted_at      TIMESTAMPTZ,
    ADD COLUMN retracted_reason  TEXT;

-- Index for "active thoughts" filters used by recent_thoughts, search_trigram,
-- search_vector_knn (the join), find_unfacted_thoughts, find_facted_thoughts.
-- A partial index keeps it small at single-user volumes.
CREATE INDEX thoughts_active_scope_recent_idx
    ON thoughts (scope, created_at DESC)
    WHERE retracted_at IS NULL;

-- M3 Phase C: three-band confidence routing. The pre-M3 routing was binary —
-- `confidence < review_queue_below` (default 0.7) lands in `facts_review_queue`,
-- everything else lands in `facts`. This column adds the middle band: rows
-- whose confidence is `review_queue_below ≤ confidence < min_confidence_to_store`
-- (default 0.85) commit to `facts` but with `flagged = true`, so downstream
-- consumers can filter or de-emphasize them. Kill-switch: setting
-- `min_confidence_to_store = review_queue_below` in `[reflector]` collapses
-- back to two-band semantics — every committed row gets `flagged = false`.
-- See `docs/milestones/m3-search-quality.md` Success Criterion 5.
ALTER TABLE facts
  ADD COLUMN flagged BOOLEAN NOT NULL DEFAULT FALSE;

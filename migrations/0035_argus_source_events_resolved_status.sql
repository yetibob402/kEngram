-- Allow terminal resolution of payload_hash_conflict without deleting quarantine history.
-- Gauge kengram_source_conflicts counts status='conflict' only (stack-monitor kengram-health.js).
-- Applied on prod 2026-07-31 for row 1623cb17 (diesel residual after consumer v0.2 recovery).

ALTER TABLE argus_source_events DROP CONSTRAINT IF EXISTS argus_source_events_status_check;
ALTER TABLE argus_source_events ADD CONSTRAINT argus_source_events_status_check
  CHECK (status = ANY (ARRAY[
    'pending'::text,
    'stored'::text,
    'conflict'::text,
    'dlq'::text,
    'skipped'::text,
    'resolved'::text
  ]));

COMMENT ON CONSTRAINT argus_source_events_status_check ON argus_source_events IS
  'pending|stored|conflict|dlq|skipped|resolved — resolved is terminal resolution of conflict; keep row + error + metadata.conflict evidence';

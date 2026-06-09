-- Argus Telegram/NATS source_ref idempotency gate.
--
-- kEngram's content_fingerprint is intentionally content-dedupe, not source
-- identity. Telegram, session, and bridge ingest need replay safety keyed by
-- producer source_ref. This adapter-owned table is the gate before capture.

CREATE TABLE argus_source_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace       TEXT NOT NULL,
    source_ref      TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    thought_id      UUID REFERENCES thoughts(id) ON DELETE SET NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','stored','conflict','dlq','skipped')),
    error           TEXT,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}',
    UNIQUE (namespace, source_ref)
);

CREATE INDEX argus_source_events_status_idx
    ON argus_source_events (status, first_seen_at ASC);

CREATE INDEX argus_source_events_thought_idx
    ON argus_source_events (thought_id)
    WHERE thought_id IS NOT NULL;

-- Stage 7 ingest hygiene audit.
--
-- Repair/apply runs are explicit, bounded, and audited. Dry-run checks do
-- not write these tables; only operator-confirmed apply runs persist rows.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

CREATE TABLE IF NOT EXISTS ingest_hygiene_runs (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode                  TEXT NOT NULL CHECK (mode IN ('apply')),
    status                TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
    parameters            JSONB NOT NULL DEFAULT '{}'::jsonb,
    stats                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at           TIMESTAMPTZ,
    error                 TEXT
);

CREATE INDEX IF NOT EXISTS ingest_hygiene_runs_recent_idx
    ON ingest_hygiene_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS ingest_hygiene_mutations (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id                UUID NOT NULL REFERENCES ingest_hygiene_runs(id) ON DELETE CASCADE,
    mutation_kind         TEXT NOT NULL CHECK (mutation_kind IN ('delete','retract')),
    target_table          TEXT NOT NULL,
    target_kind           TEXT,
    target_id             UUID,
    reason                TEXT NOT NULL,
    prior_state           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ingest_hygiene_mutations_run_idx
    ON ingest_hygiene_mutations (run_id, created_at ASC);

CREATE INDEX IF NOT EXISTS ingest_hygiene_mutations_target_idx
    ON ingest_hygiene_mutations (target_table, target_kind, target_id, created_at DESC);

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0029_ingest_hygiene_audit',
    0,
    'Added durable audit tables for bounded Stage-7 ingest hygiene repair/apply runs. Dry-run checks remain non-mutating.'
);

-- Part 2 corpus hygiene: document source_ref v2 reconciliation ledger.
--
-- Additive only. This migration creates the run and exception ledger for the
-- reviews/specs source_ref v2 alias backfill. It never rewrites or deletes a
-- v1 argus_source_events row; v2 aliases themselves are ordinary
-- argus_source_events rows inserted by the operator-driven apply transaction
-- together with exactly one run row here. Dry-run and read-only
-- reconciliation phases do not write these tables.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

CREATE TABLE IF NOT EXISTS document_source_ref_v2_runs (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status                      TEXT NOT NULL
                                CHECK (status IN ('applied','aborted','rolled_back')),
    -- Frozen inventory identity (§6.4 steps 2-4): counts and hashes recorded
    -- at apply time are immutable; the freeze trigger below enforces it.
    file_count                  INTEGER NOT NULL CHECK (file_count >= 0),
    staged_row_count            INTEGER NOT NULL CHECK (staged_row_count >= 0),
    class_a_count               INTEGER NOT NULL CHECK (class_a_count >= 0),
    class_b_count               INTEGER NOT NULL CHECK (class_b_count >= 0),
    class_c_count               INTEGER NOT NULL CHECK (class_c_count >= 0),
    class_d_count               INTEGER NOT NULL CHECK (class_d_count >= 0),
    class_counts_by_namespace   JSONB NOT NULL,
    inventory_sha256            TEXT NOT NULL
                                CHECK (inventory_sha256 ~ '^[0-9a-f]{64}$'),
    normalized_content_sha256   TEXT NOT NULL
                                CHECK (normalized_content_sha256 ~ '^[0-9a-f]{64}$'),
    backup_archive_sha256       TEXT
                                CHECK (backup_archive_sha256 IS NULL
                                       OR backup_archive_sha256 ~ '^[0-9a-f]{64}$'),
    code_head                   TEXT NOT NULL,
    importer_profile            JSONB NOT NULL,
    approvals                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
    notes                       TEXT
);

CREATE INDEX IF NOT EXISTS document_source_ref_v2_runs_recent_idx
    ON document_source_ref_v2_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS document_source_ref_v2_exceptions (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id                  UUID NOT NULL
                            REFERENCES document_source_ref_v2_runs(id) ON DELETE RESTRICT,
    class                   TEXT NOT NULL
                            CHECK (class IN ('legacy_payload_thought_mismatch_quarantined',
                                             'current_unresolved',
                                             'unexpected_inconsistency')),
    reason                  TEXT NOT NULL,
    namespace               TEXT NOT NULL
                            CHECK (namespace IN ('documents/reviews','documents/specs')),
    source_event_id         UUID
                            REFERENCES argus_source_events(id) ON DELETE RESTRICT,
    v1_source_ref           TEXT,
    source_file             TEXT,
    staged_key              TEXT,
    legacy_payload_hash     TEXT
                            CHECK (legacy_payload_hash IS NULL
                                   OR legacy_payload_hash ~ '^[0-9a-f]{64}$'),
    linked_thought_id       UUID,
    linked_thought_digest   TEXT
                            CHECK (linked_thought_digest IS NULL
                                   OR linked_thought_digest ~ '^[0-9a-f]{64}$'),
    first_seen_at           TIMESTAMPTZ,
    last_seen_at            TIMESTAMPTZ,
    details                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Every exception is keyed by the legacy event or by the staged v2 key.
    CHECK (source_event_id IS NOT NULL OR staged_key IS NOT NULL),
    -- Class B always quarantines a concrete legacy event (§6.3 B).
    CHECK (class <> 'legacy_payload_thought_mismatch_quarantined'
           OR (source_event_id IS NOT NULL
               AND legacy_payload_hash IS NOT NULL
               AND linked_thought_id IS NOT NULL)),
    -- Class C always names the staged current chunk it refused to alias (§6.3 C).
    CHECK (class <> 'current_unresolved' OR staged_key IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS document_source_ref_v2_exceptions_event_uidx
    ON document_source_ref_v2_exceptions (run_id, source_event_id)
    WHERE source_event_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS document_source_ref_v2_exceptions_staged_key_uidx
    ON document_source_ref_v2_exceptions (run_id, namespace, staged_key)
    WHERE staged_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS document_source_ref_v2_exceptions_run_idx
    ON document_source_ref_v2_exceptions (run_id, class);

CREATE INDEX IF NOT EXISTS document_source_ref_v2_exceptions_namespace_idx
    ON document_source_ref_v2_exceptions (namespace, class, created_at DESC);

-- Immutability enforcement. Run count/hash/provenance fields freeze at
-- insert; only status, completed_at, and notes may change afterward.
-- Exception rows are a permanent quarantine ledger: no UPDATE, no DELETE.
CREATE OR REPLACE FUNCTION document_source_ref_v2_runs_freeze()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION
            'document_source_ref_v2_runs is a permanent ledger; DELETE is refused';
    END IF;
    IF (to_jsonb(NEW) - 'status' - 'completed_at' - 'notes')
       IS DISTINCT FROM
       (to_jsonb(OLD) - 'status' - 'completed_at' - 'notes') THEN
        RAISE EXCEPTION
            'document_source_ref_v2_runs count/hash/provenance fields are immutable; only status, completed_at, notes may change';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION document_source_ref_v2_exceptions_freeze()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'document_source_ref_v2_exceptions is a permanent quarantine ledger; % is refused', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS document_source_ref_v2_runs_freeze_trg
    ON document_source_ref_v2_runs;
CREATE TRIGGER document_source_ref_v2_runs_freeze_trg
    BEFORE UPDATE OR DELETE ON document_source_ref_v2_runs
    FOR EACH ROW EXECUTE FUNCTION document_source_ref_v2_runs_freeze();

DROP TRIGGER IF EXISTS document_source_ref_v2_exceptions_freeze_trg
    ON document_source_ref_v2_exceptions;
CREATE TRIGGER document_source_ref_v2_exceptions_freeze_trg
    BEFORE UPDATE OR DELETE ON document_source_ref_v2_exceptions
    FOR EACH ROW EXECUTE FUNCTION document_source_ref_v2_exceptions_freeze();

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0031_doc_source_ref_v2_aliases',
    0,
    'Added document source_ref v2 run/exception reconciliation ledger (additive; no v1 source event is rewritten or deleted; count/hash fields frozen by trigger).'
);

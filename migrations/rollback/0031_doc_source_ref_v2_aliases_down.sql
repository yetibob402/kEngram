-- Rollback for 0031_doc_source_ref_v2_aliases.
--
-- Fail-closed: refuses to erase a populated permanent ledger, and refuses to
-- run while any v2 alias rows exist in argus_source_events (an applied run
-- implies aliases; dropping the ledger would orphan their audit trail).

SET lock_timeout = '5s';
SET statement_timeout = '30min';

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM document_source_ref_v2_exceptions)
        OR EXISTS (SELECT 1 FROM document_source_ref_v2_runs)
        OR EXISTS (
            SELECT 1 FROM argus_source_events
            WHERE namespace IN ('documents/reviews','documents/specs')
              AND (source_ref LIKE 'reviews:v2:%' OR source_ref LIKE 'specs:v2:%')
        )
    THEN
        RAISE EXCEPTION
            '0031 rollback refused: document source_ref v2 ledger or v2 aliases are populated';
    END IF;
END;
$$;

DROP TRIGGER IF EXISTS document_source_ref_v2_exceptions_freeze_trg
    ON document_source_ref_v2_exceptions;
DROP TRIGGER IF EXISTS document_source_ref_v2_runs_freeze_trg
    ON document_source_ref_v2_runs;
DROP FUNCTION IF EXISTS document_source_ref_v2_exceptions_freeze();
DROP FUNCTION IF EXISTS document_source_ref_v2_runs_freeze();
DROP TABLE IF EXISTS document_source_ref_v2_exceptions;
DROP TABLE IF EXISTS document_source_ref_v2_runs;

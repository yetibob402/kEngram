SET lock_timeout = '5s';
SET statement_timeout = '30min';

DROP TABLE IF EXISTS eval_equivalence_mappings;
DROP TABLE IF EXISTS protected_gold_impact_report;
DROP TABLE IF EXISTS gold_protection_manifest;
DROP TABLE IF EXISTS thought_scope_aliases;
DROP TABLE IF EXISTS thought_dedup_candidates;
DROP TABLE IF EXISTS thought_archive_events;

DROP INDEX IF EXISTS artifact_chunks_pipeline_idx;
DROP INDEX IF EXISTS artifact_chunks_source_order_idx;
DROP INDEX IF EXISTS artifact_chunks_source_fingerprint_idx;

ALTER TABLE artifact_chunks
    DROP COLUMN IF EXISTS retracted_reason,
    DROP COLUMN IF EXISTS retracted_at,
    DROP COLUMN IF EXISTS created_at,
    DROP COLUMN IF EXISTS pipeline_run_id,
    DROP COLUMN IF EXISTS metadata,
    DROP COLUMN IF EXISTS end_char,
    DROP COLUMN IF EXISTS start_char,
    DROP COLUMN IF EXISTS token_estimate,
    DROP COLUMN IF EXISTS chunker_version,
    DROP COLUMN IF EXISTS chunker_id,
    DROP COLUMN IF EXISTS content_fingerprint,
    DROP COLUMN IF EXISTS source_thought_id;

DROP TABLE IF EXISTS corpus_pipeline_runs;
DROP TABLE IF EXISTS corpus_snapshots;

-- Corpus-surgery audit, chunk lineage, gold protection, and reversible archive state.
--
-- Phase 1 is dedup/chunk/rescope only. It does not create fact extraction or
-- promoted-fact tables.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

CREATE TABLE IF NOT EXISTS corpus_snapshots (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_name         TEXT NOT NULL UNIQUE,
    dump_path             TEXT,
    dump_sha256           TEXT,
    repo_sha              TEXT,
    config_sha256         TEXT,
    active_model_id       TEXT,
    ollama_digest         TEXT,
    counts                JSONB NOT NULL DEFAULT '{}',
    index_validity        JSONB NOT NULL DEFAULT '{}',
    eval_artifact_path    TEXT,
    writer_freeze_started_at TIMESTAMPTZ,
    writer_freeze_ended_at   TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS corpus_pipeline_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id         UUID REFERENCES corpus_snapshots(id),
    pipeline_kind       TEXT NOT NULL CHECK (pipeline_kind IN ('dedup','chunk','archive','rescope','rollback')),
    mode                TEXT NOT NULL CHECK (mode IN ('dry_run','canary','batch','rollback')),
    status              TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
    parameters          JSONB NOT NULL DEFAULT '{}',
    input_cohort_hash   TEXT,
    stats               JSONB NOT NULL DEFAULT '{}',
    artifact_dir        TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS corpus_pipeline_runs_recent_idx
    ON corpus_pipeline_runs (started_at DESC);

ALTER TABLE artifact_chunks
    ADD COLUMN IF NOT EXISTS source_thought_id UUID REFERENCES thoughts(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS content_fingerprint BYTEA,
    ADD COLUMN IF NOT EXISTS chunker_id TEXT,
    ADD COLUMN IF NOT EXISTS chunker_version INT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS token_estimate INT,
    ADD COLUMN IF NOT EXISTS start_char INT,
    ADD COLUMN IF NOT EXISTS end_char INT,
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS pipeline_run_id UUID REFERENCES corpus_pipeline_runs(id),
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS retracted_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS retracted_reason TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS artifact_chunks_source_fingerprint_idx
    ON artifact_chunks (source_thought_id, content_fingerprint)
    WHERE source_thought_id IS NOT NULL
      AND content_fingerprint IS NOT NULL;

CREATE INDEX IF NOT EXISTS artifact_chunks_source_order_idx
    ON artifact_chunks (source_thought_id, chunk_index)
    WHERE source_thought_id IS NOT NULL
      AND retracted_at IS NULL;

CREATE INDEX IF NOT EXISTS artifact_chunks_pipeline_idx
    ON artifact_chunks (pipeline_run_id, chunk_index)
    WHERE pipeline_run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS thought_archive_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thought_id           UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    pipeline_run_id      UUID REFERENCES corpus_pipeline_runs(id),
    original_scope       TEXT NOT NULL,
    archive_scope        TEXT NOT NULL,
    reason              TEXT NOT NULL,
    archived_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    restored_at          TIMESTAMPTZ,
    restore_reason       TEXT,
    UNIQUE (thought_id, archive_scope, archived_at)
);

CREATE INDEX IF NOT EXISTS thought_archive_events_active_idx
    ON thought_archive_events (archive_scope, archived_at DESC)
    WHERE restored_at IS NULL;

CREATE TABLE IF NOT EXISTS thought_dedup_candidates (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id       UUID REFERENCES corpus_pipeline_runs(id),
    cluster_id            UUID NOT NULL DEFAULT gen_random_uuid(),
    canonical_thought_id  UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    duplicate_thought_id  UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    similarity            DOUBLE PRECISION,
    reasons               JSONB NOT NULL DEFAULT '{}',
    guard_outputs         JSONB NOT NULL DEFAULT '{}',
    status                TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','review_only','approved','rejected','archived')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pipeline_run_id, canonical_thought_id, duplicate_thought_id)
);

CREATE TABLE IF NOT EXISTS thought_scope_aliases (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thought_id           UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    axis                TEXT NOT NULL CHECK (axis IN ('agent','domain','archive','legacy')),
    scope               TEXT NOT NULL,
    confidence          REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    classifier_id       TEXT,
    classifier_version  INT,
    pipeline_run_id     UUID REFERENCES corpus_pipeline_runs(id),
    source              TEXT NOT NULL DEFAULT 'pipeline',
    evidence            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retracted_at        TIMESTAMPTZ,
    UNIQUE (thought_id, axis, scope, retracted_at)
);

CREATE INDEX IF NOT EXISTS thought_scope_aliases_active_scope_idx
    ON thought_scope_aliases (axis, scope, thought_id)
    WHERE retracted_at IS NULL;

CREATE TABLE IF NOT EXISTS gold_protection_manifest (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id           UUID REFERENCES corpus_snapshots(id),
    query_id              TEXT NOT NULL,
    gold_thought_id       UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    selectors             JSONB NOT NULL DEFAULT '{}',
    content_hash          TEXT,
    source_scope          TEXT,
    active_state          TEXT NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (snapshot_id, query_id, gold_thought_id)
);

CREATE TABLE IF NOT EXISTS protected_gold_impact_report (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id       UUID REFERENCES corpus_pipeline_runs(id),
    query_id              TEXT NOT NULL,
    gold_thought_id       UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    proposed_action       TEXT NOT NULL,
    reason                TEXT,
    rollback_action       TEXT,
    review_status         TEXT NOT NULL DEFAULT 'pending' CHECK (review_status IN ('pending','approved','blocked','not_applicable')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_equivalence_mappings (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id              TEXT NOT NULL,
    original_gold_id      UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    equivalent_target_kind TEXT NOT NULL CHECK (equivalent_target_kind IN ('thought','artifact_chunk','fact')),
    equivalent_target_id  UUID NOT NULL,
    relationship          TEXT NOT NULL CHECK (relationship IN ('chunk_of','approved_supersedes','approved_replaces','approved_fact')),
    answer_evidence       JSONB NOT NULL DEFAULT '{}',
    reviewer              TEXT NOT NULL,
    reviewed_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (query_id, original_gold_id, equivalent_target_kind, equivalent_target_id)
);

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0019_corpus_surgery_audit_and_chunks',
    0,
    'Added corpus snapshot/run audit, chunk lineage, archive events, dedup candidates, scope aliases, and eval gold-protection tables. No fact-promotion tables.'
);

ANALYZE artifact_chunks;
ANALYZE corpus_pipeline_runs;
ANALYZE thought_archive_events;

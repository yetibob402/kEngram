-- Adopt the live/manual thought_scope_aliases table into tracked migrations.
-- This migration is intentionally defensive:
--   * absent table: create the expected shape
--   * present matching table: no-op except missing constraints/indexes
--   * present incompatible table: fail closed before backfill or serving legs

DO $$
BEGIN
    IF to_regclass('public.thought_scope_aliases') IS NULL THEN
        CREATE TABLE public.thought_scope_aliases (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            thought_id          UUID NOT NULL,
            axis                TEXT NOT NULL,
            scope               TEXT NOT NULL,
            confidence          REAL NOT NULL,
            classifier_id       TEXT,
            classifier_version  INTEGER,
            pipeline_run_id     UUID,
            source              TEXT NOT NULL DEFAULT 'pipeline',
            evidence            JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            retracted_at        TIMESTAMPTZ
        );
    END IF;
END $$;

DO $$
DECLARE
    missing_column TEXT;
BEGIN
    SELECT required.column_name
      INTO missing_column
      FROM (
          VALUES
              ('id'),
              ('thought_id'),
              ('axis'),
              ('scope'),
              ('confidence'),
              ('classifier_id'),
              ('classifier_version'),
              ('pipeline_run_id'),
              ('source'),
              ('evidence'),
              ('created_at'),
              ('retracted_at')
      ) AS required(column_name)
      WHERE NOT EXISTS (
          SELECT 1
            FROM information_schema.columns c
           WHERE c.table_schema = 'public'
             AND c.table_name = 'thought_scope_aliases'
             AND c.column_name = required.column_name
      )
      LIMIT 1;

    IF missing_column IS NOT NULL THEN
        RAISE EXCEPTION 'thought_scope_aliases is present but missing required column %', missing_column;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'thought_scope_aliases'
           AND column_name = 'thought_id'
           AND data_type = 'uuid'
           AND is_nullable = 'NO'
    ) THEN
        RAISE EXCEPTION 'thought_scope_aliases.thought_id must be NOT NULL uuid';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'thought_scope_aliases'
           AND column_name = 'axis'
           AND data_type = 'text'
           AND is_nullable = 'NO'
    ) THEN
        RAISE EXCEPTION 'thought_scope_aliases.axis must be NOT NULL text';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'thought_scope_aliases'
           AND column_name = 'scope'
           AND data_type = 'text'
           AND is_nullable = 'NO'
    ) THEN
        RAISE EXCEPTION 'thought_scope_aliases.scope must be NOT NULL text';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'thought_scope_aliases'
           AND column_name = 'confidence'
           AND data_type = 'real'
           AND is_nullable = 'NO'
    ) THEN
        RAISE EXCEPTION 'thought_scope_aliases.confidence must be NOT NULL real';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'thought_scope_aliases'
           AND column_name = 'evidence'
           AND data_type = 'jsonb'
           AND is_nullable = 'NO'
    ) THEN
        RAISE EXCEPTION 'thought_scope_aliases.evidence must be NOT NULL jsonb';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.thought_scope_aliases'::regclass
           AND conname = 'thought_scope_aliases_axis_check'
    ) THEN
        ALTER TABLE public.thought_scope_aliases
            ADD CONSTRAINT thought_scope_aliases_axis_check
            CHECK (axis = ANY (ARRAY['agent'::text, 'domain'::text, 'archive'::text, 'legacy'::text]));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.thought_scope_aliases'::regclass
           AND conname = 'thought_scope_aliases_confidence_check'
    ) THEN
        ALTER TABLE public.thought_scope_aliases
            ADD CONSTRAINT thought_scope_aliases_confidence_check
            CHECK (confidence >= 0 AND confidence <= 1);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.thought_scope_aliases'::regclass
           AND conname = 'thought_scope_aliases_thought_id_fkey'
    ) THEN
        ALTER TABLE public.thought_scope_aliases
            ADD CONSTRAINT thought_scope_aliases_thought_id_fkey
            FOREIGN KEY (thought_id) REFERENCES public.thoughts(id) ON DELETE CASCADE;
    END IF;

    IF to_regclass('public.corpus_pipeline_runs') IS NOT NULL
       AND NOT EXISTS (
           SELECT 1 FROM pg_constraint
            WHERE conrelid = 'public.thought_scope_aliases'::regclass
              AND conname = 'thought_scope_aliases_pipeline_run_id_fkey'
       )
    THEN
        ALTER TABLE public.thought_scope_aliases
            ADD CONSTRAINT thought_scope_aliases_pipeline_run_id_fkey
            FOREIGN KEY (pipeline_run_id) REFERENCES public.corpus_pipeline_runs(id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.thought_scope_aliases'::regclass
           AND conname = 'thought_scope_aliases_thought_id_axis_scope_retracted_at_key'
    ) THEN
        ALTER TABLE public.thought_scope_aliases
            ADD CONSTRAINT thought_scope_aliases_thought_id_axis_scope_retracted_at_key
            UNIQUE (thought_id, axis, scope, retracted_at);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS thought_scope_aliases_active_scope_idx
    ON public.thought_scope_aliases (axis, scope, thought_id)
    WHERE retracted_at IS NULL;

-- The live/manual unique constraint includes nullable retracted_at and does
-- not prevent duplicate active aliases. This partial unique index is the
-- enforceable active-row guarantee required before any alias backfill.
CREATE UNIQUE INDEX IF NOT EXISTS thought_scope_aliases_active_unique_idx
    ON public.thought_scope_aliases (thought_id, axis, scope)
    WHERE retracted_at IS NULL;

CREATE INDEX IF NOT EXISTS thoughts_tags_domain_scope_string_idx
    ON public.thoughts ((tags->>'domain_scope'))
    WHERE retracted_at IS NULL
      AND jsonb_typeof(tags->'domain_scope') = 'string';

CREATE INDEX IF NOT EXISTS thoughts_tags_retrieval_aliases_gin_idx
    ON public.thoughts USING gin ((tags->'retrieval_aliases'))
    WHERE retracted_at IS NULL
      AND jsonb_typeof(tags->'retrieval_aliases') = 'array';

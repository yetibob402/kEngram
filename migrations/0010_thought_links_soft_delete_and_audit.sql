-- M5.2: soft-delete on thought_links + migration_audit table.
--
-- Two operator-diagnostics changes bundled into one migration.
--
-- (1) Soft-delete on thought_links. Previously `delete_link` did a hard
--     DELETE, so `unlink_thoughts` returned `existed: false` for both
--     "never existed" and "previously removed" — indistinguishable from
--     the operator's seat. With `deleted_at TIMESTAMPTZ NULL`, the
--     orchestrator can distinguish three states: deleted_now,
--     already_deleted, never_existed. Re-creating a previously-deleted
--     edge with the same triple inserts a fresh live row (the partial
--     unique index ignores soft-deleted rows).
--
-- (2) migration_audit table. New table tracking what each migration did
--     (rows touched, free-text notes). Sqlx's internal _sqlx_migrations
--     table records *that* a migration ran but not its operator-facing
--     impact. Going forward, any row-touching migration ends with an
--     INSERT INTO migration_audit statement. Surfaced to operators via
--     `engram audit migrations`.

-- (1) Soft-delete.

ALTER TABLE thought_links
    ADD COLUMN deleted_at TIMESTAMPTZ NULL;

-- Replace the table-level unique constraint with a partial unique index
-- so the constraint only applies to live rows. Re-creating an edge with
-- the same (from, relation, to_kind, to_value) triple after a soft-delete
-- succeeds; the previously-deleted row sits inert in the table.
ALTER TABLE thought_links DROP CONSTRAINT thought_links_unique_edge;
CREATE UNIQUE INDEX thought_links_unique_edge
    ON thought_links (from_thought_id, relation, to_kind, to_value)
    WHERE deleted_at IS NULL;

-- Index for "find soft-deleted edges" diagnostic queries. Partial keeps
-- it tiny — most rows have deleted_at IS NULL.
CREATE INDEX thought_links_deleted_at_idx
    ON thought_links (deleted_at)
    WHERE deleted_at IS NOT NULL;

-- (2) Migration audit.

CREATE TABLE migration_audit (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    migration    TEXT        NOT NULL,
    ran_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rows_touched BIGINT      NOT NULL DEFAULT 0,
    notes        TEXT
);

CREATE INDEX migration_audit_ran_at_idx ON migration_audit (ran_at DESC);

-- Seed entries for the two M5.2 migrations.
INSERT INTO migration_audit (migration, rows_touched, notes) VALUES
    ('0009_thought_links_heterogeneous_targets', 0,
     'Added polymorphic target columns (to_kind, to_entity, to_person, to_url, to_value). Existing rows defaulted to to_kind=thought. Replaced unique_edge + no_self_reference constraints; added target_valid and url_format CHECKs. Non-destructive.'),
    ('0010_thought_links_soft_delete_and_audit', 0,
     'Added thought_links.deleted_at; unique edge became a partial index excluding soft-deleted rows. Created migration_audit table.');

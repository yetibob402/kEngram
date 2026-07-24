-- M4 truth-trio rev4 C2: consumer-scoped writer enforcement.
--
-- Re-enables the 0030 thoughts_require_gated_writer trigger in a SCOPED form:
-- a direct INSERT INTO thoughts is denied for the runtime ingest principals
-- (kengram_runtime and every kengram_rt_* role), so the general consumer on
-- kengram_rt_session cannot regress to its raw-INSERT bypass.  The gate owner
-- keeps the 0030 fingerprint integrity check, and every other writer — owner
-- `kengram` and the unmigrated chokepoint-manifest writers — is UNAFFECTED
-- this slice.  Migrating those writers is the boarded follow-on slice.
--
-- Numbering note: 0031 is reserved for the truth-trio A-slice source_created_at
-- column (separate branch, kengram Rust/SQL); this migration is independent of
-- it and applies in either order.

SET lock_timeout = '5s';
SET statement_timeout = '5min';

CREATE OR REPLACE FUNCTION thoughts_require_gated_writer()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $fn$
BEGIN
    IF current_user = 'kengram_gate_owner' THEN
        -- Unchanged 0030 invariant for the gate's own inserts.
        IF NEW.content_fingerprint IS DISTINCT FROM public.digest(NEW.content, 'sha256') THEN
            RAISE EXCEPTION 'thought_content_fingerprint_mismatch'
                USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
    END IF;
    -- ponytail: name-scoped deny (the runtime role family only) this slice;
    -- move to membership-based scoping when the remaining chokepoint-manifest
    -- writers migrate in the follow-on slice.  Name matching is deliberate:
    -- pg_has_role() answers true for superusers, which would wrongly deny the
    -- unmigrated owner-side writers on clusters where the owner is superuser.
    IF current_user::text = 'kengram_runtime'
       OR current_user::text LIKE 'kengram\_rt\_%' THEN
        RAISE EXCEPTION 'thought_insert_requires_capture_thought_gated'
            USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
END
$fn$;

ALTER FUNCTION thoughts_require_gated_writer() OWNER TO kengram_gate_owner;
REVOKE ALL ON FUNCTION thoughts_require_gated_writer() FROM PUBLIC;

ALTER TABLE thoughts ENABLE TRIGGER thoughts_require_gated_writer;

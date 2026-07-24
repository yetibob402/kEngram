-- Rollback for 0032: re-disable the trigger and restore the exact 0030
-- gate-owner-only function body.

ALTER TABLE thoughts DISABLE TRIGGER thoughts_require_gated_writer;

CREATE OR REPLACE FUNCTION thoughts_require_gated_writer()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $fn$
BEGIN
    IF current_user <> 'kengram_gate_owner' THEN
        RAISE EXCEPTION 'thought_insert_requires_capture_thought_gated'
            USING ERRCODE = '42501';
    END IF;
    IF NEW.content_fingerprint IS DISTINCT FROM public.digest(NEW.content, 'sha256') THEN
        RAISE EXCEPTION 'thought_content_fingerprint_mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END
$fn$;

ALTER FUNCTION thoughts_require_gated_writer() OWNER TO kengram_gate_owner;
REVOKE ALL ON FUNCTION thoughts_require_gated_writer() FROM PUBLIC;

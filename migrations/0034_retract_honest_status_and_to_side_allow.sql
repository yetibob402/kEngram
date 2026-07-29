-- Migration 0034: retract contract honesty + supersession TO may retract
--
-- Board 547350 / knox train: retract_thought returned false with status
-- thought_chain_participant_requires_repoint (or similar), but MCP mapped
-- every !retracted to "not found or already retracted" while get_thought
-- still showed the row live. Contract lie, not missing row.
--
-- Also: guard blocked ANY replaces/refines participant (from OR to). After
-- PR9, predecessors that are TO of replaces are meant to be retractable
-- (edges survive retraction). Only FROM-side of live supersession edges
-- still require unlink/repoint before retract.

CREATE OR REPLACE FUNCTION public.retract_thought_serialized(
    p_thought_id uuid,
    p_reason text DEFAULT NULL,
    p_claimed_producer_class text DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $fn$
DECLARE
    v_principal name := session_user;
    v_profile corpus_hygiene_producer_principals%ROWTYPE;
    v_retracted_at timestamptz;
    v_exists boolean;
BEGIN
    SELECT * INTO STRICT v_profile
    FROM public.corpus_hygiene_producer_principals
    WHERE principal_name = v_principal AND enabled;
    IF p_claimed_producer_class IS NOT NULL
       AND p_claimed_producer_class <> v_profile.producer_class THEN
        RAISE EXCEPTION 'producer_class_mismatch' USING ERRCODE = '42501';
    END IF;
    IF p_reason IS NOT NULL AND length(p_reason) > 1000 THEN
        RAISE EXCEPTION 'retraction_reason_too_long' USING ERRCODE = '22023';
    END IF;

    PERFORM public.lock_thought_relation_endpoints(ARRAY[p_thought_id], false);

    SELECT EXISTS(SELECT 1 FROM public.thoughts WHERE id = p_thought_id)
    INTO v_exists;
    IF NOT v_exists THEN
        RETURN jsonb_build_object('retracted', false, 'status', 'not_found');
    END IF;

    SELECT retracted_at INTO v_retracted_at
    FROM public.thoughts WHERE id = p_thought_id;
    IF v_retracted_at IS NOT NULL THEN
        RETURN jsonb_build_object('retracted', false, 'status', 'already_retracted');
    END IF;

    -- Only FROM-side of live supersession edges blocks retract.
    -- TO-side (predecessor) may retract; edges survive (m5 / PR9).
    IF EXISTS (
        SELECT 1
        FROM public.thought_links
        WHERE deleted_at IS NULL
          AND relation IN ('replaces','refines')
          AND to_kind = 'thought'
          AND from_thought_id = p_thought_id
    ) THEN
        RETURN jsonb_build_object(
            'retracted', false,
            'status', 'thought_chain_from_requires_unlink'
        );
    END IF;

    UPDATE public.thoughts
    SET retracted_at = transaction_timestamp(), retracted_reason = p_reason
    WHERE id = p_thought_id AND retracted_at IS NULL;
    IF NOT FOUND THEN
        -- Race: became retracted between check and update
        RETURN jsonb_build_object('retracted', false, 'status', 'already_retracted');
    END IF;
    RETURN jsonb_build_object('retracted', true, 'status', 'retracted');
EXCEPTION
    WHEN foreign_key_violation THEN
        RETURN jsonb_build_object('retracted', false, 'status', 'not_found');
    WHEN no_data_found THEN
        RAISE EXCEPTION 'producer_principal_unmapped:%', session_user USING ERRCODE = '42501';
END
$fn$;

ALTER FUNCTION public.retract_thought_serialized(uuid, text, text) OWNER TO kengram_gate_owner;
REVOKE ALL ON FUNCTION public.retract_thought_serialized(uuid, text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.retract_thought_serialized(uuid, text, text) TO kengram_runtime;

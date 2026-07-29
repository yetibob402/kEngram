-- Migration 0033: allow link_thoughts targets to be retracted thoughts.
--
-- Root cause (board kengram-replaces-link-to-retracted-thought-returns-internal-error,
-- jones 2x repro 2026-07-29): mutate_thought_relations_serialized locked EVERY
-- endpoint with require_active=true on create. A replaces edge from a live
-- corrected thought to a retracted predecessor raised relation_endpoint_retracted
-- (23514), which MCP map_link_error collapsed to opaque "internal database error".
-- supports edges from the same source to live thoughts succeeded.
--
-- Design (m5-selective-relations): edges survive thought retraction; soft-retract
-- keeps the row so FK stays valid. Correction provenance needs replaces -> retracted.
--
-- Fix: require *from* of create/replace_tagger_set active always.
-- TO thought may be retracted ONLY for ruled relations replaces|refines
-- (supersession provenance; Knox design ruling + jones sealed RED).
-- Unruled kinds (requires, references, supports, belongs_to, decided_by)
-- still require an active TO — jones PR9 blocker seq 538210 / probe at ae74f45.

-- Knox DESIGN RULING 2026-07-29 (a2a 3ea8ed16, hunter two-sided evidence):
-- replaces/refines TO retracted targets MUST be VALID (supersession provenance).
-- Do NOT enforce link-first ordering — that fights the documented
-- retract-then-capture-corrected + link_thoughts correction flow.
-- Discriminator: same payload_hash fails on retracted TO, succeeds on live TO
-- (hunter positive control link_id 9041c982). Jones sealed RED bound.
--

-- 0033: relation targets may be retracted (m5 edges-survive-retraction).
-- From-endpoints of create/replace_tagger_set still require active.
-- Jones repro: replaces from live corrected thought -> retracted predecessor
-- raised relation_endpoint_retracted and MCP mapped it to opaque
-- "internal database error". supports to live targets succeeded.

DROP FUNCTION IF EXISTS public.lock_thought_relation_endpoints(uuid[], boolean);

CREATE FUNCTION public.lock_thought_relation_endpoints(
    p_thought_ids uuid[],
    p_require_active boolean DEFAULT true,
    p_active_required_ids uuid[] DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $fn$
DECLARE
    v_id uuid;
    v_retracted_at timestamptz;
    v_must_be_active boolean;
BEGIN
    IF current_user <> 'kengram_gate_owner' THEN
        RAISE EXCEPTION 'endpoint_lock_owner_mismatch' USING ERRCODE = '42501';
    END IF;

    FOR v_id IN
        SELECT DISTINCT id
        FROM pg_catalog.unnest(COALESCE(p_thought_ids, ARRAY[]::uuid[])) AS id
        WHERE id IS NOT NULL
        ORDER BY id
    LOOP
        SELECT retracted_at
        INTO v_retracted_at
        FROM public.thoughts
        WHERE id = v_id
        FOR UPDATE;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'relation_endpoint_missing:%', v_id USING ERRCODE = '23503';
        END IF;

        IF p_require_active THEN
            IF p_active_required_ids IS NULL THEN
                -- Legacy 2-arg semantics: every locked id must be active.
                v_must_be_active := true;
            ELSE
                -- Only listed ids must be active (from-side always; TO unless
                -- relation is ruled replaces|refines supersession provenance).
                v_must_be_active := v_id = ANY (p_active_required_ids);
            END IF;
            IF v_must_be_active AND v_retracted_at IS NOT NULL THEN
                RAISE EXCEPTION 'relation_endpoint_retracted:%', v_id USING ERRCODE = '23514';
            END IF;
        END IF;
    END LOOP;
END
$fn$;

ALTER FUNCTION public.lock_thought_relation_endpoints(uuid[], boolean, uuid[]) OWNER TO kengram_gate_owner;
REVOKE ALL ON FUNCTION public.lock_thought_relation_endpoints(uuid[], boolean, uuid[]) FROM PUBLIC, kengram_runtime;

CREATE OR REPLACE FUNCTION public.mutate_thought_relations_serialized(
    p_operations jsonb,
    p_source_event_namespace text,
    p_source_event_ref text,
    p_source_event_payload_hash text,
    p_request_metadata jsonb DEFAULT '{}'::jsonb,
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
    v_existing thought_relation_request_events%ROWTYPE;
    v_canonical jsonb;
    v_intent_hash bytea;
    v_endpoint_ids uuid[] := ARRAY[]::uuid[];
    v_op jsonb;
    v_rel jsonb;
    v_action text;
    v_from uuid;
    v_to_kind text;
    v_to_value text;
    v_to_thought uuid;
    v_relation text;
    v_source text;
    v_note text;
    v_link_id uuid;
    v_result_ids uuid[] := ARRAY[]::uuid[];
    v_operation_ids uuid[];
    v_operation_results jsonb := '[]'::jsonb;
    v_operation_outcome text;
    v_require_active boolean;
    v_active_required_ids uuid[] := ARRAY[]::uuid[];
BEGIN
    IF NULLIF(btrim(p_source_event_namespace), '') IS NULL
       OR NULLIF(btrim(p_source_event_ref), '') IS NULL
       OR NULLIF(btrim(p_source_event_payload_hash), '') IS NULL THEN
        RAISE EXCEPTION 'relation_source_event_required' USING ERRCODE = '22023';
    END IF;
    IF p_operations IS NULL OR jsonb_typeof(p_operations) <> 'array' THEN
        RAISE EXCEPTION 'relation_operations_must_be_array' USING ERRCODE = '22023';
    END IF;

    SELECT * INTO STRICT v_profile
    FROM public.corpus_hygiene_producer_principals
    WHERE principal_name = v_principal AND enabled;

    IF p_claimed_producer_class IS NOT NULL
       AND p_claimed_producer_class <> v_profile.producer_class THEN
        RAISE EXCEPTION 'producer_class_mismatch' USING ERRCODE = '42501';
    END IF;
    IF NOT v_profile.relation_allowed THEN
        RAISE EXCEPTION 'producer_profile_relation_denied' USING ERRCODE = '42501';
    END IF;

    SELECT COALESCE(jsonb_agg(value ORDER BY
               value->>'action', value->>'from_thought_id', value->>'relation',
               value->>'to_kind', value->>'to_value', value->>'source', value->>'note',
               value::text),
               '[]'::jsonb)
    INTO v_canonical
    FROM jsonb_array_elements(p_operations);
    v_intent_hash := public.digest(convert_to(v_canonical::text, 'UTF8'), 'sha256');

    -- Claim the replay key before endpoint locking or relation mutation.
    -- ON CONFLICT waits for a concurrent claimant; the re-read below then
    -- returns that claimant's completed result instead of leaking 23505.
    INSERT INTO public.thought_relation_request_events (
        source_event_namespace, source_event_ref, source_event_payload_hash,
        producer_principal, producer_class, profile_revision,
        canonical_intent_hash, operations, request_metadata, status
    ) VALUES (
        p_source_event_namespace, p_source_event_ref, p_source_event_payload_hash,
        v_principal, v_profile.producer_class, v_profile.profile_revision,
        v_intent_hash, v_canonical, COALESCE(p_request_metadata, '{}'::jsonb), 'pending'
    )
    ON CONFLICT (source_event_namespace, source_event_ref) DO NOTHING
    RETURNING * INTO v_existing;

    IF NOT FOUND THEN
        SELECT * INTO STRICT v_existing
        FROM public.thought_relation_request_events
        WHERE source_event_namespace = p_source_event_namespace
          AND source_event_ref = p_source_event_ref
        FOR UPDATE;

        IF v_existing.source_event_payload_hash = p_source_event_payload_hash
           AND v_existing.canonical_intent_hash = v_intent_hash THEN
            RETURN jsonb_build_object(
                'status', v_existing.status,
                'replayed', true,
                'request_id', v_existing.id,
                'link_ids', to_jsonb(v_existing.result_link_ids),
                'operation_results', v_existing.operation_results
            );
        END IF;
        RETURN jsonb_build_object(
            'status', 'source_event_conflict',
            'replayed', false,
            'request_id', v_existing.id,
            'link_ids', '[]'::jsonb,
            'operation_results', '[]'::jsonb
        );
    END IF;

    FOR v_op IN SELECT value FROM jsonb_array_elements(v_canonical)
    LOOP
        v_action := COALESCE(v_op->>'action', 'create');
        v_from := NULLIF(v_op->>'from_thought_id', '')::uuid;
        v_to_kind := COALESCE(NULLIF(v_op->>'to_kind', ''), 'thought');
        v_to_value := NULLIF(v_op->>'to_value', '');
        IF v_from IS NULL THEN
            RAISE EXCEPTION 'invalid_relation_from_endpoint' USING ERRCODE = '22023';
        END IF;
        IF v_action NOT IN ('create','delete','replace_tagger_set') THEN
            RAISE EXCEPTION 'invalid_relation_action:%', v_action USING ERRCODE = '22023';
        END IF;
        IF v_action = 'replace_tagger_set' THEN
            IF jsonb_typeof(v_op->'relations') <> 'array' THEN
                RAISE EXCEPTION 'tagger_relation_set_must_be_array' USING ERRCODE = '22023';
            END IF;
            v_endpoint_ids := array_append(v_endpoint_ids, v_from);
            v_active_required_ids := array_append(v_active_required_ids, v_from);
            SELECT v_endpoint_ids || COALESCE(array_agg(to_thought_id), ARRAY[]::uuid[])
            INTO v_endpoint_ids
            FROM public.thought_links
            WHERE from_thought_id = v_from
              AND source = 'tagger'
              AND to_kind = 'thought'
              AND deleted_at IS NULL;
            SELECT v_endpoint_ids || COALESCE(array_agg((value->>'to_value')::uuid), ARRAY[]::uuid[])
            INTO v_endpoint_ids
            FROM jsonb_array_elements(v_op->'relations')
            WHERE value->>'to_kind' = 'thought';
            -- Tagger TOs: only replaces|refines may target retracted thoughts.
            SELECT v_active_required_ids || COALESCE(array_agg((value->>'to_value')::uuid), ARRAY[]::uuid[])
            INTO v_active_required_ids
            FROM jsonb_array_elements(v_op->'relations')
            WHERE value->>'to_kind' = 'thought'
              AND COALESCE(value->>'relation', '') NOT IN ('replaces', 'refines');
            CONTINUE;
        END IF;
        IF v_to_value IS NULL THEN
            RAISE EXCEPTION 'invalid_relation_endpoint' USING ERRCODE = '22023';
        END IF;
        IF v_to_kind NOT IN ('thought','entity','person','url') THEN
            RAISE EXCEPTION 'invalid_relation_target_kind:%', v_to_kind USING ERRCODE = '22023';
        END IF;
        v_endpoint_ids := array_append(v_endpoint_ids, v_from);
        v_relation := v_op->>'relation';
        IF v_action = 'create' THEN
            v_active_required_ids := array_append(v_active_required_ids, v_from);
        END IF;
        IF v_to_kind = 'thought' THEN
            v_to_thought := v_to_value::uuid;
            IF v_from = v_to_thought THEN
                RAISE EXCEPTION 'relation_intent_self_reference' USING ERRCODE = '23514';
            END IF;
            v_endpoint_ids := array_append(v_endpoint_ids, v_to_thought);
            -- Ruled set only: replaces|refines may point at a retracted TO.
            -- Jones blocker 538210: unruled kinds must still require active TO.
            IF v_action = 'create'
               AND COALESCE(v_relation, '') NOT IN ('replaces', 'refines') THEN
                v_active_required_ids := array_append(v_active_required_ids, v_to_thought);
            END IF;
        END IF;
    END LOOP;

    v_require_active := EXISTS (
        SELECT 1 FROM jsonb_array_elements(v_canonical) x
        WHERE COALESCE(x->>'action', 'create') IN ('create','replace_tagger_set')
    );
    IF NOT v_require_active AND v_profile.producer_class <> 'break_glass_passthrough' THEN
        v_require_active := true;
    END IF;
    -- Active: from-side always on create/tagger; TO also unless relation is
    -- ruled replaces|refines (supersession provenance to retracted predecessor).
    PERFORM public.lock_thought_relation_endpoints(
        v_endpoint_ids,
        v_require_active,
        CASE WHEN v_require_active THEN v_active_required_ids ELSE NULL END
    );

    FOR v_op IN SELECT value FROM jsonb_array_elements(v_canonical)
    LOOP
        v_link_id := NULL;
        v_operation_ids := ARRAY[]::uuid[];
        v_action := COALESCE(v_op->>'action', 'create');
        v_from := (v_op->>'from_thought_id')::uuid;
        v_relation := v_op->>'relation';
        v_to_kind := COALESCE(v_op->>'to_kind', 'thought');
        v_to_value := v_op->>'to_value';
        v_to_thought := CASE WHEN v_to_kind = 'thought' THEN v_to_value::uuid ELSE NULL END;
        v_source := COALESCE(v_op->>'source', 'agent');
        v_note := v_op->>'note';

        IF v_action = 'replace_tagger_set' THEN
            UPDATE public.thought_links
            SET deleted_at = transaction_timestamp()
            WHERE from_thought_id = v_from
              AND source = 'tagger'
              AND deleted_at IS NULL;

            FOR v_rel IN SELECT value FROM jsonb_array_elements(v_op->'relations')
            LOOP
                v_relation := v_rel->>'relation';
                v_to_kind := COALESCE(v_rel->>'to_kind', 'thought');
                v_to_value := v_rel->>'to_value';
                v_to_thought := CASE WHEN v_to_kind = 'thought' THEN v_to_value::uuid ELSE NULL END;
                v_note := v_rel->>'note';
                IF v_relation NOT IN ('replaces','requires','references','supports','belongs_to','decided_by','refines')
                   OR v_to_kind NOT IN ('thought','entity','person','url')
                   OR NULLIF(v_to_value, '') IS NULL
                   OR (v_to_kind = 'thought' AND v_from = v_to_thought) THEN
                    RAISE EXCEPTION 'invalid_tagger_relation_intent' USING ERRCODE = '22023';
                END IF;
                INSERT INTO public.thought_links (
                    from_thought_id, relation, to_kind,
                    to_thought_id, to_entity, to_person, to_url,
                    source, note
                ) VALUES (
                    v_from, v_relation, v_to_kind,
                    CASE WHEN v_to_kind = 'thought' THEN v_to_value::uuid END,
                    CASE WHEN v_to_kind = 'entity' THEN v_to_value END,
                    CASE WHEN v_to_kind = 'person' THEN v_to_value END,
                    CASE WHEN v_to_kind = 'url' THEN v_to_value END,
                    'tagger', v_note
                ) RETURNING id INTO v_link_id;
                v_result_ids := array_append(v_result_ids, v_link_id);
                v_operation_ids := array_append(v_operation_ids, v_link_id);
            END LOOP;
            v_operation_results := v_operation_results || jsonb_build_array(
                jsonb_build_object(
                    'action', 'replace_tagger_set',
                    'outcome', 'replaced',
                    'link_ids', to_jsonb(v_operation_ids)
                )
            );
            CONTINUE;
        END IF;

        IF v_relation NOT IN ('replaces','requires','references','supports','belongs_to','decided_by','refines')
           OR v_source NOT IN ('agent','tagger') THEN
            RAISE EXCEPTION 'invalid_relation_intent' USING ERRCODE = '22023';
        END IF;

        IF v_action = 'create' THEN
            IF v_relation IN ('replaces','refines') AND v_to_kind = 'thought' AND EXISTS (
                WITH RECURSIVE walk(id) AS (
                    SELECT l.to_thought_id
                    FROM public.thought_links l
                    WHERE l.from_thought_id = v_to_thought
                      AND l.to_kind = 'thought'
                      AND l.relation IN ('replaces','refines')
                      AND l.deleted_at IS NULL
                    UNION
                    SELECT l.to_thought_id
                    FROM public.thought_links l
                    JOIN walk w ON l.from_thought_id = w.id
                    WHERE l.to_kind = 'thought'
                      AND l.relation IN ('replaces','refines')
                      AND l.deleted_at IS NULL
                )
                SELECT 1 FROM walk WHERE id = v_from
            ) THEN
                RAISE EXCEPTION 'relation_cycle_detected' USING ERRCODE = '23514';
            END IF;

            SELECT id INTO v_link_id
            FROM public.thought_links
            WHERE from_thought_id = v_from
              AND relation = v_relation
              AND to_kind = v_to_kind
              AND to_value = v_to_value
              AND deleted_at IS NULL;

            IF v_link_id IS NULL THEN
                v_operation_outcome := 'created';
                INSERT INTO public.thought_links (
                    from_thought_id, relation, to_kind,
                    to_thought_id, to_entity, to_person, to_url,
                    source, note
                ) VALUES (
                    v_from, v_relation, v_to_kind,
                    CASE WHEN v_to_kind = 'thought' THEN v_to_value::uuid END,
                    CASE WHEN v_to_kind = 'entity' THEN v_to_value END,
                    CASE WHEN v_to_kind = 'person' THEN v_to_value END,
                    CASE WHEN v_to_kind = 'url' THEN v_to_value END,
                    v_source, v_note
                ) RETURNING id INTO v_link_id;
            ELSE
                v_operation_outcome := 'already_live';
            END IF;
            v_result_ids := array_append(v_result_ids, v_link_id);
            v_operation_ids := array_append(v_operation_ids, v_link_id);
        ELSE
            UPDATE public.thought_links
            SET deleted_at = transaction_timestamp()
            WHERE from_thought_id = v_from
              AND relation = v_relation
              AND to_kind = v_to_kind
              AND to_value = v_to_value
              AND deleted_at IS NULL
            RETURNING id INTO v_link_id;
            IF v_link_id IS NOT NULL THEN
                v_operation_outcome := 'deleted_now';
                v_result_ids := array_append(v_result_ids, v_link_id);
                v_operation_ids := array_append(v_operation_ids, v_link_id);
            ELSIF EXISTS (
                SELECT 1
                FROM public.thought_links
                WHERE from_thought_id = v_from
                  AND relation = v_relation
                  AND to_kind = v_to_kind
                  AND to_value = v_to_value
                  AND deleted_at IS NOT NULL
            ) THEN
                v_operation_outcome := 'already_deleted';
            ELSE
                v_operation_outcome := 'never_existed';
            END IF;
        END IF;
        v_operation_results := v_operation_results || jsonb_build_array(
            jsonb_build_object(
                'action', v_action,
                'outcome', v_operation_outcome,
                'link_ids', to_jsonb(v_operation_ids)
            )
        );
    END LOOP;

    UPDATE public.thought_relation_request_events
    SET status = 'completed', result_link_ids = v_result_ids,
        operation_results = v_operation_results,
        completed_at = transaction_timestamp()
    WHERE source_event_namespace = p_source_event_namespace
      AND source_event_ref = p_source_event_ref;

    RETURN jsonb_build_object(
        'status', 'completed',
        'replayed', false,
        'link_ids', to_jsonb(v_result_ids),
        'operation_results', v_operation_results
    );
EXCEPTION
    WHEN no_data_found THEN
        RAISE EXCEPTION 'producer_principal_unmapped:%', session_user USING ERRCODE = '42501';
END
$fn$;

ALTER FUNCTION public.mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) OWNER TO kengram_gate_owner;
REVOKE ALL ON FUNCTION public.mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) TO kengram_runtime;

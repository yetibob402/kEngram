-- Delivery 1: one enforced capture chokepoint and one serialized relation
-- mutation primitive.  This migration deliberately lands both enforcement
-- triggers disabled; the reviewed cutover enables them only after every
-- caller has moved to the functions below.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

SET lock_timeout = '5s';
SET statement_timeout = '30min';

-- Roles are cluster-global while sqlx test databases run migrations in
-- parallel. pg_authid is a shared catalog, so this short transaction lock
-- serializes the role/membership reconciliation across databases too.
LOCK TABLE pg_catalog.pg_authid IN EXCLUSIVE MODE;

-- -------------------------------------------------------------------------
-- Principals.  Passwords are provisioned out of band and never appear here.
-- -------------------------------------------------------------------------

DO $roles$
DECLARE
    v_role text;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kengram_gate_owner') THEN
        BEGIN
            CREATE ROLE kengram_gate_owner NOLOGIN;
        EXCEPTION WHEN duplicate_object THEN
            NULL; -- concurrent disposable-test database created it
        END;
    END IF;
    ALTER ROLE kengram_gate_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kengram_runtime') THEN
        BEGIN
            CREATE ROLE kengram_runtime NOLOGIN;
        EXCEPTION WHEN duplicate_object THEN
            NULL; -- concurrent disposable-test database created it
        END;
    END IF;
    ALTER ROLE kengram_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

    FOREACH v_role IN ARRAY ARRAY[
        'kengram_rt_native_mcp',
        'kengram_rt_session',
        'kengram_rt_telegram',
        'kengram_rt_agent_comms',
        'kengram_rt_reviews',
        'kengram_rt_specs',
        'kengram_rt_openclaw',
        'kengram_rt_hive',
        'kengram_rt_mba_archive',
        'kengram_rt_phase4',
        'kengram_rt_maintenance_import'
    ] LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = v_role) THEN
            BEGIN
                EXECUTE format('CREATE ROLE %I LOGIN', v_role);
            EXCEPTION WHEN duplicate_object THEN
                NULL; -- concurrent disposable-test database created it
            END;
        END IF;
        EXECUTE format(
            'ALTER ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT NOREPLICATION NOBYPASSRLS',
            v_role
        );
    END LOOP;
END
$roles$;

-- Reconcile every edge touching a hygiene role before installing the exact
-- graph.  All producer roles are newly introduced by this migration, so this
-- is both idempotent and fail-safe on a retried cutover.
DO $memberships$
DECLARE
    v_edge record;
BEGIN
    FOR v_edge IN
        SELECT parent.rolname AS parent_name, member.rolname AS member_name
        FROM pg_auth_members m
        JOIN pg_roles parent ON parent.oid = m.roleid
        JOIN pg_roles member ON member.oid = m.member
        WHERE parent.rolname = ANY (ARRAY[
                  'kengram_gate_owner','kengram_runtime',
                  'kengram_rt_native_mcp','kengram_rt_session','kengram_rt_telegram',
                  'kengram_rt_agent_comms','kengram_rt_reviews','kengram_rt_specs',
                  'kengram_rt_openclaw','kengram_rt_hive','kengram_rt_mba_archive',
                  'kengram_rt_phase4','kengram_rt_maintenance_import'
              ])
           OR member.rolname = ANY (ARRAY[
                  'kengram_gate_owner','kengram_runtime',
                  'kengram_rt_native_mcp','kengram_rt_session','kengram_rt_telegram',
                  'kengram_rt_agent_comms','kengram_rt_reviews','kengram_rt_specs',
                  'kengram_rt_openclaw','kengram_rt_hive','kengram_rt_mba_archive',
                  'kengram_rt_phase4','kengram_rt_maintenance_import'
              ])
    LOOP
        EXECUTE format('REVOKE %I FROM %I', v_edge.parent_name, v_edge.member_name);
    END LOOP;
END
$memberships$;

GRANT kengram_runtime TO kengram_rt_native_mcp WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_session WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_telegram WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_agent_comms WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_reviews WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_specs WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_openclaw WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_hive WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_mba_archive WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_phase4 WITH INHERIT TRUE, SET FALSE;
GRANT kengram_runtime TO kengram_rt_maintenance_import WITH INHERIT TRUE, SET FALSE;

-- -------------------------------------------------------------------------
-- Authoritative principal/profile map and mode settings.
-- -------------------------------------------------------------------------

CREATE TABLE corpus_hygiene_producer_principals (
    principal_name              name PRIMARY KEY,
    producer_class              text NOT NULL UNIQUE,
    profile_revision            integer NOT NULL CHECK (profile_revision > 0),
    enabled                     boolean NOT NULL DEFAULT true,
    requires_source_created_at  boolean NOT NULL,
    keep_only                   boolean NOT NULL,
    enforce_eligible            boolean NOT NULL,
    relation_allowed            boolean NOT NULL,
    created_at                  timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CHECK (NOT keep_only OR NOT enforce_eligible)
);

ALTER TABLE corpus_hygiene_producer_principals OWNER TO kengram;

INSERT INTO corpus_hygiene_producer_principals
    (principal_name, producer_class, profile_revision,
     requires_source_created_at, keep_only, enforce_eligible, relation_allowed)
VALUES
    ('kengram_rt_native_mcp', 'native_mcp', 1, false, true, false, true),
    ('kengram_rt_session', 'session_realtime', 1, false, false, true, false),
    ('kengram_rt_telegram', 'telegram_realtime', 1, false, false, true, false),
    ('kengram_rt_agent_comms', 'agent_comms_realtime', 1, false, false, true, false),
    ('kengram_rt_reviews', 'review_historical', 1, true, false, true, false),
    ('kengram_rt_specs', 'spec_historical', 1, true, false, true, false),
    ('kengram_rt_openclaw', 'openclaw_historical', 1, true, false, true, false),
    ('kengram_rt_hive', 'hive_historical', 1, true, false, true, true),
    ('kengram_rt_mba_archive', 'mba_archive_historical', 1, true, false, true, false),
    ('kengram_rt_phase4', 'phase4_derived', 1, true, false, true, true),
    ('kengram_rt_maintenance_import', 'maintenance_historical_keep_only', 1, true, true, false, true),
    ('kengram', 'break_glass_passthrough', 1, true, true, false, true);

CREATE TABLE corpus_hygiene_gate_settings (
    principal_name      name NOT NULL,
    producer_class      text NOT NULL,
    profile_revision    integer NOT NULL,
    mode                text NOT NULL DEFAULT 'off' CHECK (mode IN ('off','shadow','enforce')),
    observation_floor   double precision NOT NULL DEFAULT 0.80 CHECK (observation_floor >= 0 AND observation_floor <= 1),
    semantic_threshold  double precision NOT NULL DEFAULT 0.90 CHECK (semantic_threshold >= 0 AND semantic_threshold <= 1),
    window_days         integer NOT NULL DEFAULT 14 CHECK (window_days BETWEEN 1 AND 30),
    novelty_bound       double precision NOT NULL DEFAULT 0.15 CHECK (novelty_bound >= 0 AND novelty_bound <= 1),
    config_revision     integer NOT NULL DEFAULT 1 CHECK (config_revision > 0),
    updated_at          timestamptz NOT NULL DEFAULT transaction_timestamp(),
    PRIMARY KEY (principal_name, producer_class, profile_revision),
    FOREIGN KEY (principal_name) REFERENCES corpus_hygiene_producer_principals(principal_name),
    CHECK (producer_class <> 'break_glass_passthrough' OR mode = 'off')
);

ALTER TABLE corpus_hygiene_gate_settings OWNER TO kengram;

INSERT INTO corpus_hygiene_gate_settings (principal_name, producer_class, profile_revision)
SELECT principal_name, producer_class, profile_revision
FROM corpus_hygiene_producer_principals;

-- -------------------------------------------------------------------------
-- Durable gate and relation request ledgers.
-- -------------------------------------------------------------------------

CREATE TABLE thought_ingest_gate_events (
    id                       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_identity         text NOT NULL,
    producer_principal       name NOT NULL,
    producer_class           text NOT NULL,
    profile_revision         integer NOT NULL,
    correlation_id           text,
    scope                    text NOT NULL,
    source                   text NOT NULL,
    source_event_namespace   text,
    source_event_ref         text,
    source_event_payload_hash text,
    candidate_fingerprint    bytea NOT NULL CHECK (octet_length(candidate_fingerprint) = 32),
    candidate_content        text,
    candidate_metadata       jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding_model_id       text,
    embedding_model_version  integer,
    mode                     text NOT NULL,
    action                   text NOT NULL,
    bypass_reason            jsonb,
    matched_thought_id       uuid REFERENCES thoughts(id) ON DELETE SET NULL,
    similarity               double precision,
    threshold                double precision,
    window_days              integer,
    effective_created_at     timestamptz NOT NULL,
    observed_at              timestamptz NOT NULL,
    protected_atoms          text[] NOT NULL DEFAULT '{}',
    missing_protected_atoms  text[] NOT NULL DEFAULT '{}',
    novelty_ratio            double precision,
    polarity_safe            boolean,
    relation_intent_guard    boolean NOT NULL DEFAULT false,
    restored_thought_id      uuid REFERENCES thoughts(id) ON DELETE SET NULL,
    restored_at              timestamptz,
    created_at               timestamptz NOT NULL DEFAULT transaction_timestamp()
);

CREATE INDEX thought_ingest_gate_events_request_idx
    ON thought_ingest_gate_events (request_identity, created_at DESC);
CREATE INDEX thought_ingest_gate_events_action_idx
    ON thought_ingest_gate_events (action, created_at DESC);

ALTER TABLE thought_ingest_gate_events OWNER TO kengram;

CREATE TABLE thought_relation_request_events (
    id                        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_event_namespace    text NOT NULL,
    source_event_ref          text NOT NULL,
    source_event_payload_hash text NOT NULL,
    producer_principal        name NOT NULL,
    producer_class            text NOT NULL,
    profile_revision          integer NOT NULL,
    canonical_intent_hash     bytea NOT NULL CHECK (octet_length(canonical_intent_hash) = 32),
    operations                jsonb NOT NULL,
    request_metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
    status                    text NOT NULL CHECK (status IN ('pending','completed')),
    result_link_ids           uuid[] NOT NULL DEFAULT '{}',
    created_at                timestamptz NOT NULL DEFAULT transaction_timestamp(),
    completed_at              timestamptz,
    UNIQUE (source_event_namespace, source_event_ref)
);

CREATE INDEX thought_relation_request_events_created_idx
    ON thought_relation_request_events (created_at DESC);

ALTER TABLE thought_relation_request_events OWNER TO kengram;

ALTER TABLE pending_tags
    ADD COLUMN tag_job_generation_id uuid DEFAULT gen_random_uuid();
UPDATE pending_tags
SET tag_job_generation_id = gen_random_uuid()
WHERE tag_job_generation_id IS NULL;
ALTER TABLE pending_tags
    ALTER COLUMN tag_job_generation_id SET NOT NULL,
    ALTER COLUMN tag_job_generation_id SET DEFAULT gen_random_uuid();
CREATE UNIQUE INDEX pending_tags_generation_idx
    ON pending_tags (thought_id, tag_job_generation_id);

-- -------------------------------------------------------------------------
-- Small deterministic guard helpers.  They are intentionally conservative:
-- uncertainty keeps the candidate.
-- -------------------------------------------------------------------------

CREATE FUNCTION corpus_hygiene_protected_atoms(p_content text)
RETURNS text[]
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $fn$
    SELECT COALESCE(array_agg(DISTINCT lower(m[1]) ORDER BY lower(m[1])), ARRAY[]::text[])
    FROM pg_catalog.regexp_matches(
        p_content,
        '(https?://[^[:space:]<>]+|[0-9a-fA-F]{64}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|#[0-9]+|v[0-9]+([.][0-9]+)+|[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?|[$][0-9]+([.][0-9]+)?|[0-9]+([.][0-9]+)?%|/[A-Za-z0-9_.~/-]+|"[^"]+"|''[^'']+''|[0-9]+([.][0-9]+)?)',
        'gi'
    ) AS m;
$fn$;

CREATE FUNCTION corpus_hygiene_novelty_ratio(p_candidate text, p_match text)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $fn$
    WITH candidate_tokens AS (
        SELECT DISTINCT token
        FROM pg_catalog.regexp_split_to_table(lower(p_candidate), '[^[:alnum:]_./:-]+') AS token
        WHERE length(token) >= 3
    ), match_tokens AS (
        SELECT DISTINCT token
        FROM pg_catalog.regexp_split_to_table(lower(p_match), '[^[:alnum:]_./:-]+') AS token
        WHERE length(token) >= 3
    )
    SELECT CASE WHEN count(*) = 0 THEN 0::double precision
                ELSE count(*) FILTER (WHERE m.token IS NULL)::double precision / count(*)::double precision
           END
    FROM candidate_tokens c
    LEFT JOIN match_tokens m USING (token);
$fn$;

-- -------------------------------------------------------------------------
-- Enforcement triggers.  SECURITY INVOKER is part of the invariant.
-- -------------------------------------------------------------------------

CREATE FUNCTION thoughts_require_gated_writer()
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

CREATE TRIGGER thoughts_require_gated_writer
BEFORE INSERT ON thoughts
FOR EACH ROW EXECUTE FUNCTION thoughts_require_gated_writer();
ALTER TABLE thoughts DISABLE TRIGGER thoughts_require_gated_writer;

CREATE FUNCTION thought_links_require_serialized_writer()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $fn$
BEGIN
    IF current_user <> 'kengram_gate_owner' THEN
        RAISE EXCEPTION 'thought_link_mutation_requires_serialized_writer'
            USING ERRCODE = '42501';
    END IF;
    RETURN COALESCE(NEW, OLD);
END
$fn$;

CREATE TRIGGER thought_links_require_serialized_writer
BEFORE INSERT OR UPDATE OR DELETE ON thought_links
FOR EACH ROW EXECUTE FUNCTION thought_links_require_serialized_writer();
ALTER TABLE thought_links DISABLE TRIGGER thought_links_require_serialized_writer;

-- -------------------------------------------------------------------------
-- Shared endpoint-row lock, relation mutation, and runtime retraction.
-- -------------------------------------------------------------------------

CREATE FUNCTION lock_thought_relation_endpoints(
    p_thought_ids uuid[],
    p_require_active boolean DEFAULT true
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $fn$
DECLARE
    v_id uuid;
    v_retracted_at timestamptz;
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
        IF p_require_active AND v_retracted_at IS NOT NULL THEN
            RAISE EXCEPTION 'relation_endpoint_retracted:%', v_id USING ERRCODE = '23514';
        END IF;
    END LOOP;
END
$fn$;

CREATE FUNCTION mutate_thought_relations_serialized(
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
    v_require_active boolean;
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

    SELECT * INTO v_existing
    FROM public.thought_relation_request_events
    WHERE source_event_namespace = p_source_event_namespace
      AND source_event_ref = p_source_event_ref
    FOR UPDATE;

    IF FOUND THEN
        IF v_existing.source_event_payload_hash = p_source_event_payload_hash
           AND v_existing.canonical_intent_hash = v_intent_hash THEN
            RETURN jsonb_build_object(
                'status', v_existing.status,
                'replayed', true,
                'request_id', v_existing.id,
                'link_ids', to_jsonb(v_existing.result_link_ids)
            );
        END IF;
        RETURN jsonb_build_object(
            'status', 'source_event_conflict',
            'replayed', false,
            'request_id', v_existing.id,
            'link_ids', '[]'::jsonb
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
            CONTINUE;
        END IF;
        IF v_to_value IS NULL THEN
            RAISE EXCEPTION 'invalid_relation_endpoint' USING ERRCODE = '22023';
        END IF;
        IF v_to_kind NOT IN ('thought','entity','person','url') THEN
            RAISE EXCEPTION 'invalid_relation_target_kind:%', v_to_kind USING ERRCODE = '22023';
        END IF;
        v_endpoint_ids := array_append(v_endpoint_ids, v_from);
        IF v_to_kind = 'thought' THEN
            v_to_thought := v_to_value::uuid;
            IF v_from = v_to_thought THEN
                RAISE EXCEPTION 'relation_intent_self_reference' USING ERRCODE = '23514';
            END IF;
            v_endpoint_ids := array_append(v_endpoint_ids, v_to_thought);
        END IF;
    END LOOP;

    v_require_active := EXISTS (
        SELECT 1 FROM jsonb_array_elements(v_canonical) x
        WHERE COALESCE(x->>'action', 'create') IN ('create','replace_tagger_set')
    );
    IF NOT v_require_active AND v_profile.producer_class <> 'break_glass_passthrough' THEN
        v_require_active := true;
    END IF;
    PERFORM public.lock_thought_relation_endpoints(v_endpoint_ids, v_require_active);

    INSERT INTO public.thought_relation_request_events (
        source_event_namespace, source_event_ref, source_event_payload_hash,
        producer_principal, producer_class, profile_revision,
        canonical_intent_hash, operations, request_metadata, status
    ) VALUES (
        p_source_event_namespace, p_source_event_ref, p_source_event_payload_hash,
        v_principal, v_profile.producer_class, v_profile.profile_revision,
        v_intent_hash, v_canonical, COALESCE(p_request_metadata, '{}'::jsonb), 'pending'
    )
    RETURNING id INTO v_link_id;

    FOR v_op IN SELECT value FROM jsonb_array_elements(v_canonical)
    LOOP
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
            END LOOP;
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
            END IF;
            v_result_ids := array_append(v_result_ids, v_link_id);
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
                v_result_ids := array_append(v_result_ids, v_link_id);
            END IF;
        END IF;
    END LOOP;

    UPDATE public.thought_relation_request_events
    SET status = 'completed', result_link_ids = v_result_ids,
        completed_at = transaction_timestamp()
    WHERE source_event_namespace = p_source_event_namespace
      AND source_event_ref = p_source_event_ref;

    RETURN jsonb_build_object(
        'status', 'completed',
        'replayed', false,
        'link_ids', to_jsonb(v_result_ids)
    );
EXCEPTION
    WHEN no_data_found THEN
        RAISE EXCEPTION 'producer_principal_unmapped:%', session_user USING ERRCODE = '42501';
END
$fn$;

CREATE FUNCTION retract_thought_serialized(
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
    SELECT retracted_at INTO v_retracted_at
    FROM public.thoughts WHERE id = p_thought_id;
    IF v_retracted_at IS NOT NULL THEN
        RETURN jsonb_build_object('retracted', false, 'status', 'already_retracted');
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.thought_links
        WHERE deleted_at IS NULL
          AND relation IN ('replaces','refines')
          AND to_kind = 'thought'
          AND (from_thought_id = p_thought_id OR to_thought_id = p_thought_id)
    ) THEN
        RETURN jsonb_build_object(
            'retracted', false,
            'status', 'thought_chain_participant_requires_repoint'
        );
    END IF;

    UPDATE public.thoughts
    SET retracted_at = transaction_timestamp(), retracted_reason = p_reason
    WHERE id = p_thought_id AND retracted_at IS NULL;
    RETURN jsonb_build_object('retracted', true, 'status', 'retracted');
EXCEPTION
    WHEN foreign_key_violation THEN
        RETURN jsonb_build_object('retracted', false, 'status', 'not_found');
    WHEN no_data_found THEN
        RAISE EXCEPTION 'producer_principal_unmapped:%', session_user USING ERRCODE = '42501';
END
$fn$;

-- -------------------------------------------------------------------------
-- Capture gate.  Fingerprints, profile, mode, and timestamps are all derived
-- or validated here; callers cannot select a safer policy.
-- -------------------------------------------------------------------------

CREATE FUNCTION capture_thought_gated(
    p_scope text,
    p_content text,
    p_source text,
    p_metadata jsonb DEFAULT '{}'::jsonb,
    p_source_created_at timestamptz DEFAULT NULL,
    p_candidate_embedding vector(1024) DEFAULT NULL,
    p_embedding_model_id text DEFAULT NULL,
    p_embedding_model_version integer DEFAULT NULL,
    p_bypass_reason jsonb DEFAULT NULL,
    p_source_event_namespace text DEFAULT NULL,
    p_source_event_ref text DEFAULT NULL,
    p_source_event_payload_hash text DEFAULT NULL,
    p_source_event_metadata jsonb DEFAULT NULL,
    p_relation_intents jsonb DEFAULT '[]'::jsonb,
    p_tagger_model_id text DEFAULT NULL,
    p_claimed_producer_class text DEFAULT NULL,
    p_correlation_id text DEFAULT NULL,
    p_force_keep_token text DEFAULT NULL
)
RETURNS TABLE (
    thought_id uuid,
    action text,
    matched_thought_id uuid,
    similarity double precision,
    threshold double precision,
    effective_created_at timestamptz,
    observed_at timestamptz,
    source_event_status text,
    source_event_action text,
    relation_results jsonb,
    gate_event_id uuid
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $fn$
DECLARE
    v_principal name := session_user;
    v_profile corpus_hygiene_producer_principals%ROWTYPE;
    v_settings corpus_hygiene_gate_settings%ROWTYPE;
    v_observed timestamptz := transaction_timestamp();
    v_effective timestamptz;
    v_fingerprint bytea;
    v_existing_event argus_source_events%ROWTYPE;
    v_existing_relation thought_relation_request_events%ROWTYPE;
    v_existing_thought uuid;
    v_new_thought uuid;
    v_match_id uuid;
    v_match_content text;
    v_similarity double precision;
    v_agent_key text;
    v_mode text;
    v_action text;
    v_gate_id uuid;
    v_event_status text;
    v_event_action text;
    v_relation_results jsonb := '[]'::jsonb;
    v_relation_ops jsonb := '[]'::jsonb;
    v_canonical_relation_intents jsonb := '[]'::jsonb;
    v_existing_relation_intents jsonb := '[]'::jsonb;
    v_has_relations boolean := false;
    v_relation_replay_conflict boolean := false;
    v_force_keep boolean := false;
    v_vector_valid boolean := false;
    v_comparison_available boolean := true;
    v_effective_bypass jsonb;
    v_queue_model_id text;
    v_atoms text[] := ARRAY[]::text[];
    v_match_atoms text[] := ARRAY[]::text[];
    v_missing_atoms text[] := ARRAY[]::text[];
    v_novelty double precision;
    v_polarity_safe boolean := true;
    v_request_identity text;
    v_status_token text;
    v_comparison_sqlstate text;
    v_comparison_error text;
BEGIN
    IF p_content IS NULL OR p_content = '' THEN
        RAISE EXCEPTION 'content_must_be_nonempty' USING ERRCODE = '22023';
    END IF;
    IF p_scope IS NULL OR p_source IS NULL THEN
        RAISE EXCEPTION 'scope_and_source_required' USING ERRCODE = '22023';
    END IF;
    IF p_relation_intents IS NULL OR jsonb_typeof(p_relation_intents) <> 'array' THEN
        RAISE EXCEPTION 'relation_intents_must_be_array' USING ERRCODE = '22023';
    END IF;
    v_has_relations := jsonb_array_length(p_relation_intents) > 0;

    SELECT * INTO STRICT v_profile
    FROM public.corpus_hygiene_producer_principals
    WHERE principal_name = v_principal AND enabled;

    IF p_claimed_producer_class IS NOT NULL
       AND p_claimed_producer_class <> v_profile.producer_class THEN
        RAISE EXCEPTION 'producer_class_mismatch' USING ERRCODE = '42501';
    END IF;
    IF p_force_keep_token IS NOT NULL
       AND v_profile.producer_class <> 'break_glass_passthrough' THEN
        RAISE EXCEPTION 'force_keep_requires_break_glass' USING ERRCODE = '42501';
    END IF;

    SELECT * INTO STRICT v_settings
    FROM public.corpus_hygiene_gate_settings
    WHERE principal_name = v_profile.principal_name
      AND producer_class = v_profile.producer_class
      AND profile_revision = v_profile.profile_revision;
    v_mode := CASE WHEN v_profile.keep_only THEN 'off' ELSE v_settings.mode END;

    v_effective := COALESCE(p_source_created_at, v_observed);
    IF p_source_created_at > v_observed + interval '5 minutes' THEN
        RAISE EXCEPTION 'source_created_at_too_far_in_future' USING ERRCODE = '22007';
    END IF;
    IF v_profile.requires_source_created_at AND p_source_created_at IS NULL THEN
        RAISE EXCEPTION 'source_created_at_required' USING ERRCODE = '22007';
    END IF;

    IF (p_candidate_embedding IS NULL) <> (p_bypass_reason IS NOT NULL) THEN
        RAISE EXCEPTION 'candidate_vector_or_structured_bypass_required' USING ERRCODE = '22023';
    END IF;
    IF p_bypass_reason IS NOT NULL AND jsonb_typeof(p_bypass_reason) <> 'object' THEN
        RAISE EXCEPTION 'bypass_reason_must_be_object' USING ERRCODE = '22023';
    END IF;

    v_vector_valid := p_candidate_embedding IS NOT NULL
        AND p_embedding_model_id = 'bge-m3:1024'
        AND p_embedding_model_version = 1
        AND vector_dims(p_candidate_embedding) = 1024;
    v_effective_bypass := CASE
        WHEN v_vector_valid THEN NULL
        WHEN p_bypass_reason IS NOT NULL THEN p_bypass_reason
        ELSE jsonb_build_object(
            'code', 'candidate_embedding_contract_mismatch',
            'model_id', p_embedding_model_id,
            'model_version', p_embedding_model_version
        )
    END;
    v_queue_model_id := CASE
        WHEN p_candidate_embedding IS NOT NULL THEN 'bge-m3:1024'
        ELSE COALESCE(p_embedding_model_id, 'bge-m3:1024')
    END;

    IF (p_source_event_namespace IS NULL)::integer
       + (p_source_event_ref IS NULL)::integer
       + (p_source_event_payload_hash IS NULL)::integer NOT IN (0, 3) THEN
        RAISE EXCEPTION 'source_event_identity_must_be_complete' USING ERRCODE = '22023';
    END IF;
    IF v_has_relations AND p_source_event_namespace IS NULL THEN
        RAISE EXCEPTION 'relation_source_event_required' USING ERRCODE = '22023';
    END IF;
    IF v_has_relations AND NOT v_profile.relation_allowed THEN
        RAISE EXCEPTION 'producer_profile_relation_denied' USING ERRCODE = '42501';
    END IF;
    IF v_has_relations AND EXISTS (
        SELECT 1
        FROM jsonb_array_elements(p_relation_intents) AS intent(value)
        WHERE jsonb_typeof(intent.value) <> 'object'
           OR intent.value ? 'from_thought_id'
           OR COALESCE(intent.value->>'action', 'create') <> 'create'
           OR COALESCE(intent.value->>'source', 'agent') <> 'agent'
           OR COALESCE(intent.value->>'relation', '') NOT IN (
                  'replaces','requires','references','supports',
                  'belongs_to','decided_by','refines'
              )
           OR COALESCE(intent.value->>'to_kind', 'thought') NOT IN (
                  'thought','entity','person','url'
              )
           OR NULLIF(intent.value->>'to_value', '') IS NULL
           OR (
                  COALESCE(intent.value->>'to_kind', 'thought') = 'url'
                  AND intent.value->>'to_value' !~ '^https?://'
              )
    ) THEN
        RAISE EXCEPTION 'invalid_capture_relation_intent' USING ERRCODE = '22023';
    END IF;

    SELECT COALESCE(jsonb_agg(
               value ORDER BY
               value->>'action', value->>'relation', value->>'to_kind',
               value->>'to_value', value->>'source', value->>'note', value::text
           ), '[]'::jsonb)
    INTO v_canonical_relation_intents
    FROM jsonb_array_elements(p_relation_intents);

    v_fingerprint := public.digest(p_content, 'sha256');
    v_request_identity := COALESCE(
        p_source_event_namespace || ':' || p_source_event_ref,
        p_correlation_id,
        encode(v_fingerprint, 'hex')
    );

    IF p_source_event_namespace IS NOT NULL THEN
        SELECT * INTO v_existing_event
        FROM public.argus_source_events
        WHERE namespace = p_source_event_namespace AND source_ref = p_source_event_ref
        FOR UPDATE;

        IF FOUND THEN
            IF v_existing_event.payload_hash <> p_source_event_payload_hash THEN
                UPDATE public.argus_source_events
                SET status = 'conflict', error = 'payload_hash_conflict',
                    last_seen_at = v_observed,
                    metadata = metadata || jsonb_build_object(
                        'conflict', jsonb_build_object(
                            'incoming_payload_sha256', p_source_event_payload_hash,
                            'prior_payload_sha256', v_existing_event.payload_hash
                        )
                    )
                WHERE id = v_existing_event.id;
                RETURN QUERY SELECT NULL::uuid, 'source_event_conflict'::text,
                    v_existing_event.thought_id, NULL::double precision,
                    v_settings.semantic_threshold, v_effective, v_observed,
                    'conflict'::text, 'conflict'::text, '[]'::jsonb, NULL::uuid;
                RETURN;
            END IF;

            -- A source replay is valid only when its canonical relation intent
            -- is the same request that completed with the source event.  The
            -- relation ledger stores the gate-supplied from_thought_id, so
            -- compare the canonical producer intent after removing that one
            -- derived field.  This also rejects adding or removing all intents
            -- under an already-completed source identity.
            SELECT * INTO v_existing_relation
            FROM public.thought_relation_request_events
            WHERE source_event_namespace = p_source_event_namespace
              AND source_event_ref = p_source_event_ref
            FOR UPDATE;

            IF v_has_relations THEN
                -- Force UUID parsing and self-reference validation before a
                -- replay can return success without reaching relation DML.
                IF EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(v_canonical_relation_intents) AS intent(value)
                    WHERE COALESCE(intent.value->>'to_kind', 'thought') = 'thought'
                      AND (intent.value->>'to_value')::uuid = v_existing_event.thought_id
                ) THEN
                    RAISE EXCEPTION 'relation_intent_self_reference' USING ERRCODE = '23514';
                END IF;

                IF v_existing_relation.id IS NULL THEN
                    v_relation_replay_conflict := true;
                ELSE
                    SELECT COALESCE(jsonb_agg(
                               value - 'from_thought_id' ORDER BY
                               value->>'action', value->>'relation', value->>'to_kind',
                               value->>'to_value', value->>'source', value->>'note',
                               (value - 'from_thought_id')::text
                           ), '[]'::jsonb)
                    INTO v_existing_relation_intents
                    FROM jsonb_array_elements(v_existing_relation.operations);

                    v_relation_replay_conflict :=
                        v_existing_relation.status <> 'completed'
                        OR v_existing_relation.source_event_payload_hash
                           <> p_source_event_payload_hash
                        OR v_existing_relation_intents
                           IS DISTINCT FROM v_canonical_relation_intents;
                END IF;
            ELSE
                v_relation_replay_conflict := v_existing_relation.id IS NOT NULL;
            END IF;

            IF v_relation_replay_conflict THEN
                RETURN QUERY SELECT NULL::uuid, 'source_event_conflict'::text,
                    v_existing_event.thought_id, NULL::double precision,
                    v_settings.semantic_threshold, v_effective, v_observed,
                    'conflict'::text, 'conflict'::text, '[]'::jsonb, NULL::uuid;
                RETURN;
            END IF;

            UPDATE public.argus_source_events
            SET last_seen_at = v_observed
            WHERE id = v_existing_event.id;
            RETURN QUERY SELECT v_existing_event.thought_id,
                CASE WHEN v_existing_event.status = 'skipped' THEN 'semantic_duplicate' ELSE 'exact_duplicate' END,
                v_existing_event.thought_id, NULL::double precision,
                v_settings.semantic_threshold, v_effective, v_observed,
                v_existing_event.status, 'replay'::text,
                COALESCE(v_existing_event.metadata->'corpus_hygiene'->'relation_results', '[]'::jsonb),
                NULLIF(v_existing_event.metadata->'corpus_hygiene'->>'gate_event_id','')::uuid;
            RETURN;
        END IF;

        INSERT INTO public.argus_source_events (
            namespace, source_ref, payload_hash, status, metadata
        ) VALUES (
            p_source_event_namespace, p_source_event_ref,
            p_source_event_payload_hash, 'pending',
            COALESCE(p_source_event_metadata, '{}'::jsonb)
        );
    END IF;

    SELECT id INTO v_existing_thought
    FROM public.thoughts
    WHERE content_fingerprint = v_fingerprint;

    IF v_existing_thought IS NOT NULL AND v_has_relations THEN
        RAISE EXCEPTION 'exact_content_requires_adjudication' USING ERRCODE = '23505';
    END IF;

    IF v_existing_thought IS NOT NULL THEN
        v_action := 'exact_duplicate';
        v_new_thought := v_existing_thought;
        v_match_id := v_existing_thought;
    ELSE
        v_agent_key := (pg_catalog.regexp_match(p_scope, '^(agents|sessions)/([^/]+)$'))[2];

        IF v_agent_key IS NOT NULL THEN
            -- Capture availability must not wait indefinitely behind another
            -- family comparison.  A contended lock is a comparison failure,
            -- so keep the thought, queue ordinary async work, and audit the
            -- fail-open signal instead of blocking the caller.
            IF pg_catalog.pg_try_advisory_xact_lock(
                pg_catalog.hashtextextended(v_agent_key, 727061667331::bigint)
            ) THEN
                -- The fingerprint may have appeared before this transaction
                -- acquired the family lock.
                SELECT id INTO v_existing_thought
                FROM public.thoughts
                WHERE content_fingerprint = v_fingerprint;
            ELSE
                v_comparison_available := false;
                v_vector_valid := false;
                v_effective_bypass := jsonb_build_object(
                    'code', 'similarity_lock_contended',
                    'agent_key', v_agent_key
                );
            END IF;
        END IF;

        IF v_existing_thought IS NOT NULL THEN
            v_action := 'exact_duplicate';
            v_new_thought := v_existing_thought;
            v_match_id := v_existing_thought;
        ELSE
            v_force_keep := v_has_relations OR v_profile.keep_only OR p_force_keep_token IS NOT NULL;
            IF v_agent_key IS NULL THEN
                v_action := 'out_of_family_insert';
            ELSIF NOT v_comparison_available THEN
                v_action := 'fail_open_insert';
            ELSIF NOT v_vector_valid AND v_mode IN ('shadow','enforce') THEN
                v_action := 'fail_open_insert';
            ELSIF v_vector_valid THEN
                BEGIN
                    SELECT candidate.id, candidate.content, candidate.similarity
                    INTO v_match_id, v_match_content, v_similarity
                    FROM (
                        SELECT t.id, t.content,
                               1 - (e.embedding <=> p_candidate_embedding) AS similarity
                        FROM public.thoughts t
                        JOIN public.thought_embeddings_bge_m3 e ON e.thought_id = t.id
                        WHERE t.retracted_at IS NULL
                          AND t.scope IN ('agents/' || v_agent_key, 'sessions/' || v_agent_key)
                          AND t.created_at >= v_effective - make_interval(days => v_settings.window_days)
                          AND t.created_at <= v_effective + interval '5 minutes'
                          AND e.model_id = 'bge-m3:1024' AND e.model_version = 1
                        ORDER BY e.embedding <=> p_candidate_embedding
                        LIMIT 20
                    ) candidate
                    ORDER BY candidate.similarity DESC
                    LIMIT 1;
                EXCEPTION
                    -- These conditions arise only inside the similarity read.
                    -- Principal, timestamp, source identity, relation, insert,
                    -- queue, and integrity failures remain outside this catch
                    -- and therefore stay fail-closed.
                    WHEN query_canceled OR lock_not_available
                         OR object_not_in_prerequisite_state OR data_exception THEN
                        GET STACKED DIAGNOSTICS
                            v_comparison_sqlstate = RETURNED_SQLSTATE,
                            v_comparison_error = MESSAGE_TEXT;
                        v_comparison_available := false;
                        v_vector_valid := false;
                        v_effective_bypass := jsonb_build_object(
                            'code', 'similarity_comparison_unavailable',
                            'sqlstate', v_comparison_sqlstate,
                            'detail', v_comparison_error
                        );
                END;

                IF NOT v_comparison_available THEN
                    v_action := 'fail_open_insert';
                ELSE
                    IF v_match_id IS NOT NULL THEN
                        v_atoms := public.corpus_hygiene_protected_atoms(p_content);
                        v_match_atoms := public.corpus_hygiene_protected_atoms(v_match_content);
                        SELECT COALESCE(array_agg(atom ORDER BY atom), ARRAY[]::text[])
                        INTO v_missing_atoms
                        FROM unnest(v_atoms) atom
                        WHERE NOT atom = ANY(v_match_atoms);
                        v_novelty := public.corpus_hygiene_novelty_ratio(p_content, v_match_content);

                        FOREACH v_status_token IN ARRAY ARRAY[
                            'not','failed','blocked','reverted','superseded','approved','denied',
                            'open','closed','enabled','disabled'
                        ] LOOP
                            IF (lower(p_content) ~ ('(^|[^[:alnum:]_])' || v_status_token || '([^[:alnum:]_]|$)'))
                               <> (lower(v_match_content) ~ ('(^|[^[:alnum:]_])' || v_status_token || '([^[:alnum:]_]|$)')) THEN
                                v_polarity_safe := false;
                            END IF;
                        END LOOP;
                    END IF;

                    IF NOT v_force_keep
                       AND v_mode = 'enforce'
                       AND v_profile.enforce_eligible
                       AND v_match_id IS NOT NULL
                       AND v_similarity > v_settings.semantic_threshold
                       AND length(p_content) >= 120
                       AND cardinality(v_missing_atoms) = 0
                       AND v_polarity_safe
                       AND v_novelty <= v_settings.novelty_bound THEN
                        v_action := 'semantic_duplicate';
                        v_new_thought := v_match_id;
                    ELSIF v_match_id IS NOT NULL
                          AND v_similarity >= v_settings.observation_floor
                          AND v_mode = 'shadow' THEN
                        v_action := 'shadow_candidate';
                    ELSIF v_has_relations THEN
                        v_action := 'relation_intent_keep';
                    ELSE
                        v_action := 'inserted';
                    END IF;
                END IF;
            ELSIF v_has_relations THEN
                v_action := 'relation_intent_keep';
            ELSE
                v_action := 'inserted';
            END IF;

            IF v_action <> 'semantic_duplicate' THEN
                INSERT INTO public.thoughts (
                    scope, content, source, metadata, created_at, content_fingerprint
                ) VALUES (
                    p_scope, p_content, p_source, COALESCE(p_metadata, '{}'::jsonb),
                    v_effective, v_fingerprint
                ) RETURNING id INTO v_new_thought;

                IF v_vector_valid THEN
                    INSERT INTO public.thought_embeddings_bge_m3 (
                        thought_id, model_id, model_version, dimensions, embedding
                    ) VALUES (v_new_thought, 'bge-m3:1024', 1, 1024, p_candidate_embedding)
                    ON CONFLICT ON CONSTRAINT thought_embeddings_bge_m3_thought_id_model_id_model_version_key DO NOTHING;
                ELSE
                    INSERT INTO public.pending_embeddings (target_kind, target_id, model_id)
                    VALUES ('thought', v_new_thought, v_queue_model_id)
                    ON CONFLICT (target_kind, target_id, model_id) DO NOTHING;
                END IF;

                IF p_tagger_model_id IS NOT NULL THEN
                    INSERT INTO public.pending_tags (thought_id, tagger_model_id)
                    VALUES (v_new_thought, p_tagger_model_id)
                    ON CONFLICT ON CONSTRAINT pending_tags_pkey DO NOTHING;
                END IF;
            END IF;
        END IF;
    END IF;

    INSERT INTO public.thought_ingest_gate_events (
        request_identity, producer_principal, producer_class, profile_revision,
        correlation_id, scope, source, source_event_namespace, source_event_ref,
        source_event_payload_hash, candidate_fingerprint, candidate_content,
        candidate_metadata, embedding_model_id, embedding_model_version,
        mode, action, bypass_reason, matched_thought_id, similarity, threshold,
        window_days, effective_created_at, observed_at, protected_atoms,
        missing_protected_atoms, novelty_ratio, polarity_safe, relation_intent_guard
    ) VALUES (
        v_request_identity, v_principal, v_profile.producer_class, v_profile.profile_revision,
        p_correlation_id, p_scope, p_source, p_source_event_namespace, p_source_event_ref,
        p_source_event_payload_hash, v_fingerprint,
        CASE WHEN v_action IN ('semantic_duplicate','shadow_candidate') THEN p_content END,
        CASE WHEN v_action IN ('semantic_duplicate','shadow_candidate') THEN COALESCE(p_metadata,'{}'::jsonb) ELSE '{}'::jsonb END,
        p_embedding_model_id, p_embedding_model_version, v_mode, v_action,
        v_effective_bypass, v_match_id, v_similarity, v_settings.semantic_threshold,
        v_settings.window_days, v_effective, v_observed, v_atoms, v_missing_atoms,
        v_novelty, v_polarity_safe, v_has_relations
    ) RETURNING id INTO v_gate_id;

    IF v_has_relations THEN
        SELECT COALESCE(jsonb_agg(
            value || jsonb_build_object('from_thought_id', v_new_thought::text)
            ORDER BY value->>'action', value->>'relation', value->>'to_kind',
                     value->>'to_value', value->>'source', value->>'note', value::text
        ), '[]'::jsonb)
        INTO v_relation_ops
        FROM jsonb_array_elements(v_canonical_relation_intents);

        v_relation_results := public.mutate_thought_relations_serialized(
            v_relation_ops,
            p_source_event_namespace,
            p_source_event_ref,
            p_source_event_payload_hash,
            jsonb_build_object('capture_gate_event_id', v_gate_id),
            p_claimed_producer_class
        );
        IF v_relation_results->>'status' <> 'completed' THEN
            RAISE EXCEPTION 'atomic_relation_intent_failed:%', v_relation_results->>'status'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    IF p_source_event_namespace IS NOT NULL THEN
        v_event_status := CASE WHEN v_action = 'semantic_duplicate' THEN 'skipped' ELSE 'stored' END;
        v_event_action := CASE WHEN v_action = 'semantic_duplicate' THEN 'semantic_skip' ELSE 'stored' END;
        UPDATE public.argus_source_events
        SET thought_id = v_new_thought,
            status = v_event_status,
            error = NULL,
            last_seen_at = v_observed,
            metadata = metadata || jsonb_build_object(
                'corpus_hygiene', jsonb_build_object(
                    'gate_event_id', v_gate_id,
                    'action', v_action,
                    'relation_results', v_relation_results
                )
            )
        WHERE namespace = p_source_event_namespace AND source_ref = p_source_event_ref;
    END IF;

    RETURN QUERY SELECT v_new_thought, v_action, v_match_id, v_similarity,
        v_settings.semantic_threshold, v_effective, v_observed,
        v_event_status, v_event_action, v_relation_results, v_gate_id;
EXCEPTION
    WHEN no_data_found THEN
        RAISE EXCEPTION 'producer_principal_unmapped:%', session_user USING ERRCODE = '42501';
END
$fn$;

-- A separately addressable, pre-reviewed rollback surface.  The normal gate
-- already treats break-glass and mode=off as passthrough, so this wrapper is
-- intentionally tiny and retains every principal/time/fingerprint check.
CREATE FUNCTION capture_thought_gated_passthrough(
    p_scope text,
    p_content text,
    p_source text,
    p_metadata jsonb DEFAULT '{}'::jsonb,
    p_source_created_at timestamptz DEFAULT NULL,
    p_embedding_model_id text DEFAULT NULL,
    p_bypass_reason jsonb DEFAULT jsonb_build_object('code','operator_passthrough'),
    p_source_event_namespace text DEFAULT NULL,
    p_source_event_ref text DEFAULT NULL,
    p_source_event_payload_hash text DEFAULT NULL,
    p_source_event_metadata jsonb DEFAULT NULL,
    p_relation_intents jsonb DEFAULT '[]'::jsonb,
    p_tagger_model_id text DEFAULT NULL,
    p_claimed_producer_class text DEFAULT NULL,
    p_correlation_id text DEFAULT NULL
)
RETURNS TABLE (
    thought_id uuid,
    action text,
    matched_thought_id uuid,
    similarity double precision,
    threshold double precision,
    effective_created_at timestamptz,
    observed_at timestamptz,
    source_event_status text,
    source_event_action text,
    relation_results jsonb,
    gate_event_id uuid
)
LANGUAGE sql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $fn$
    SELECT *
    FROM public.capture_thought_gated(
        p_scope, p_content, p_source, p_metadata, p_source_created_at,
        NULL::vector(1024), p_embedding_model_id, NULL::integer,
        COALESCE(p_bypass_reason, jsonb_build_object('code','operator_passthrough')),
        p_source_event_namespace, p_source_event_ref, p_source_event_payload_hash,
        p_source_event_metadata, p_relation_intents, p_tagger_model_id,
        p_claimed_producer_class, p_correlation_id, 'passthrough'
    );
$fn$;

-- -------------------------------------------------------------------------
-- Ownership and privileges.  Runtime roles inherit only function execution
-- plus queue/read columns needed by the server/worker; they never inherit a
-- direct thought insert, relation write, or retraction-column update.
-- -------------------------------------------------------------------------

ALTER FUNCTION thoughts_require_gated_writer() OWNER TO kengram_gate_owner;
ALTER FUNCTION thought_links_require_serialized_writer() OWNER TO kengram_gate_owner;
ALTER FUNCTION lock_thought_relation_endpoints(uuid[], boolean) OWNER TO kengram_gate_owner;
ALTER FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION retract_thought_serialized(uuid,text,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) OWNER TO kengram_gate_owner;
ALTER FUNCTION capture_thought_gated_passthrough(text,text,text,jsonb,timestamptz,text,jsonb,text,text,text,jsonb,jsonb,text,text,text) OWNER TO kengram_gate_owner;

REVOKE ALL ON FUNCTION thoughts_require_gated_writer() FROM PUBLIC;
REVOKE ALL ON FUNCTION thought_links_require_serialized_writer() FROM PUBLIC;
REVOKE ALL ON FUNCTION lock_thought_relation_endpoints(uuid[], boolean) FROM PUBLIC, kengram_runtime;
REVOKE ALL ON FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION retract_thought_serialized(uuid,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION capture_thought_gated_passthrough(text,text,text,jsonb,timestamptz,text,jsonb,text,text,text,jsonb,jsonb,text,text,text) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text) TO kengram_runtime;
GRANT EXECUTE ON FUNCTION retract_thought_serialized(uuid,text,text) TO kengram_runtime;
GRANT EXECUTE ON FUNCTION capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text) TO kengram_runtime;

REVOKE ALL ON corpus_hygiene_producer_principals FROM PUBLIC, kengram_runtime;
REVOKE ALL ON corpus_hygiene_gate_settings FROM PUBLIC, kengram_runtime;
GRANT SELECT ON corpus_hygiene_producer_principals, corpus_hygiene_gate_settings TO kengram_gate_owner;

REVOKE INSERT ON thoughts FROM PUBLIC, kengram_runtime;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON thought_links FROM PUBLIC, kengram_runtime;
REVOKE UPDATE (retracted_at, retracted_reason) ON thoughts FROM PUBLIC, kengram_runtime;

GRANT SELECT, INSERT ON thoughts TO kengram_gate_owner;
GRANT UPDATE (retracted_at, retracted_reason) ON thoughts TO kengram_gate_owner;
GRANT SELECT, INSERT, UPDATE ON argus_source_events TO kengram_gate_owner;
GRANT SELECT, INSERT, UPDATE ON thought_ingest_gate_events TO kengram_gate_owner;
GRANT SELECT, INSERT, UPDATE ON thought_relation_request_events TO kengram_gate_owner;
GRANT SELECT, INSERT ON thought_embeddings_bge_m3 TO kengram_gate_owner;
GRANT SELECT, INSERT ON pending_embeddings TO kengram_gate_owner;
GRANT SELECT, INSERT ON pending_tags TO kengram_gate_owner;
GRANT SELECT, INSERT, UPDATE, DELETE ON thought_links TO kengram_gate_owner;

GRANT SELECT ON thoughts, thought_links, argus_source_events TO kengram_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON pending_embeddings, pending_tags TO kengram_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON thought_embeddings_bge_m3 TO kengram_runtime;
GRANT UPDATE (tags, tags_extractor_model, tags_extractor_version, tags_extracted_at) ON thoughts TO kengram_runtime;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;

INSERT INTO migration_audit (migration, rows_touched, notes)
VALUES (
    '0030_corpus_hygiene_gate',
    0,
    'Delivery 1 roles, server-derived capture gate, serialized relation/retraction functions, retry-stable tag generation, and disabled-at-land enforcement triggers.'
);

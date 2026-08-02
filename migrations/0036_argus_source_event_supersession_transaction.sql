-- Migration 0036: argus source-event supersession transaction
-- Spec: kengram-supersession-transactional-capability r1 §4-8 + r2 F1-F4
-- Six-path allowlist only; disposable-DB acceptance first.

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;

-- Reserved SQLSTATEs (ZSD01 exact content, ZSI01 invalid status, ZSA01 unauthorized session, ZSR01 request reuse)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kengram_rt_supersession') THEN
    IF EXISTS (
      SELECT 1 FROM pg_roles
      WHERE rolname = 'kengram_rt_supersession'
        AND (rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls)
    ) THEN
      RAISE EXCEPTION 'kengram_rt_supersession exists with unexpected attributes';
    END IF;
  ELSE
    CREATE ROLE kengram_rt_supersession LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
  END IF;
END$$;

-- Corpus hygiene principal + settings (exact intended state only)
INSERT INTO public.corpus_hygiene_producer_principals (
  principal_name, producer_class, profile_revision, enabled,
  requires_source_created_at, keep_only, enforce_eligible, relation_allowed
) VALUES (
  'kengram_rt_supersession', 'source_event_supersession', 1, true,
  true, true, false, true
) ON CONFLICT (principal_name) DO UPDATE SET
  producer_class = EXCLUDED.producer_class,
  profile_revision = EXCLUDED.profile_revision,
  enabled = EXCLUDED.enabled,
  requires_source_created_at = EXCLUDED.requires_source_created_at,
  keep_only = EXCLUDED.keep_only,
  enforce_eligible = EXCLUDED.enforce_eligible,
  relation_allowed = EXCLUDED.relation_allowed;

INSERT INTO public.corpus_hygiene_gate_settings (
  principal_name, producer_class, profile_revision, mode
) VALUES (
  'kengram_rt_supersession', 'source_event_supersession', 1, 'off'
) ON CONFLICT (principal_name, producer_class, profile_revision) DO UPDATE SET
  mode = EXCLUDED.mode;

CREATE TABLE IF NOT EXISTS public.argus_source_event_supersession_receipts (
  request_id uuid PRIMARY KEY,
  request_digest bytea NOT NULL CHECK (octet_length(request_digest) = 32),
  outcome text NOT NULL CHECK (outcome = ANY (ARRAY[
    'applied'::text,
    'refused_expected_state'::text,
    'refused_exact_content_duplicate'::text
  ])),
  stable_source_event_id uuid NULL,
  namespace text NOT NULL,
  source_ref text NOT NULL,
  expected_old_status text NOT NULL,
  expected_old_payload_hash text NOT NULL,
  expected_old_thought_id uuid NULL,
  observed_missing boolean NOT NULL,
  observed_old_status text NULL,
  observed_old_payload_hash text NULL,
  observed_old_thought_id uuid NULL,
  new_payload_hash text NOT NULL,
  new_thought_id uuid NULL,
  replaces_link_id uuid NULL,
  gate_event_id uuid NULL,
  embedding_job_id uuid NULL,
  tag_job_generation_id uuid NULL,
  embedding_model_id text NOT NULL,
  tagger_model_id text NOT NULL,
  actor text NOT NULL,
  lane text NOT NULL,
  approval_ref text NOT NULL,
  reason text NOT NULL,
  authenticated_session_user text NOT NULL,
  occurred_at timestamptz NOT NULL,
  canonical_receipt_json jsonb NOT NULL,
  receipt_digest bytea NOT NULL CHECK (octet_length(receipt_digest) = 32),
  CONSTRAINT supersession_receipt_missing_consistency CHECK (
    (observed_missing = true AND stable_source_event_id IS NULL
      AND observed_old_status IS NULL AND observed_old_payload_hash IS NULL
      AND observed_old_thought_id IS NULL)
    OR
    (observed_missing = false AND stable_source_event_id IS NOT NULL)
  ),
  CONSTRAINT supersession_receipt_applied_ids CHECK (
    (outcome = 'applied' AND new_thought_id IS NOT NULL AND replaces_link_id IS NOT NULL
      AND gate_event_id IS NOT NULL AND embedding_job_id IS NOT NULL
      AND tag_job_generation_id IS NOT NULL)
    OR
    (outcome <> 'applied' AND new_thought_id IS NULL AND replaces_link_id IS NULL
      AND gate_event_id IS NULL AND embedding_job_id IS NULL
      AND tag_job_generation_id IS NULL)
  )
);

ALTER TABLE public.argus_source_event_supersession_receipts OWNER TO kengram_gate_owner;
GRANT ALL ON public.argus_source_event_supersession_receipts TO kengram_gate_owner;

CREATE OR REPLACE FUNCTION public.argus_source_event_supersession_receipts_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  RAISE EXCEPTION 'supersession_receipt_immutable' USING ERRCODE = '25006';
END;
$$;

DROP TRIGGER IF EXISTS argus_source_event_supersession_receipts_no_update
  ON public.argus_source_event_supersession_receipts;
CREATE TRIGGER argus_source_event_supersession_receipts_no_update
  BEFORE UPDATE OR DELETE ON public.argus_source_event_supersession_receipts
  FOR EACH ROW EXECUTE PROCEDURE public.argus_source_event_supersession_receipts_immutable();

REVOKE ALL ON public.argus_source_event_supersession_receipts FROM PUBLIC;
GRANT SELECT ON public.argus_source_event_supersession_receipts TO kengram_rt_supersession;

CREATE OR REPLACE FUNCTION public.supersede_argus_source_event(
  p_request_id uuid,
  p_namespace text,
  p_source_ref text,
  p_expected_status text,
  p_expected_old_payload_hash text,
  p_expected_old_thought_id uuid,
  p_new_payload_canonical_json text,
  p_new_payload_hash text,
  p_new_content text,
  p_new_metadata jsonb,
  p_embedding_model_id text,
  p_tagger_model_id text,
  p_actor text,
  p_lane text,
  p_approval_ref text,
  p_reason text
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $fn$
DECLARE
  v_request_envelope jsonb;
  v_request_preimage text;
  v_request_digest bytea;
  v_existing public.argus_source_event_supersession_receipts%ROWTYPE;
  v_row public.argus_source_events%ROWTYPE;
  v_old_thought public.thoughts%ROWTYPE;
  v_lock_key bigint;
  v_receipt_occurred_at timestamptz;
  v_occurred_text text;
  v_receipt_envelope jsonb;
  v_receipt_digest bytea;
  v_outcome text;
  v_capture RECORD;
  v_retract jsonb;
  v_link_id uuid;
  v_emb_id uuid;
  v_tag_gen uuid;
  v_rc integer;
  v_exact_content_refusal boolean := false;
  v_new_meta jsonb;
  v_hex_re text := '^[0-9a-f]{64}$';
  v_computed_hash text;
BEGIN
  IF session_user IS DISTINCT FROM 'kengram_rt_supersession' THEN
    RAISE EXCEPTION 'supersession_unauthorized_session_user' USING ERRCODE = 'ZSA01';
  END IF;

  IF p_request_id IS NULL THEN
    RAISE EXCEPTION 'supersession_invalid_request_id' USING ERRCODE = '22023';
  END IF;
  IF p_namespace IS NULL OR btrim(p_namespace) = '' OR length(p_namespace) > 512 THEN
    RAISE EXCEPTION 'supersession_invalid_namespace' USING ERRCODE = '22023';
  END IF;
  IF p_source_ref IS NULL OR btrim(p_source_ref) = '' OR length(p_source_ref) > 1024 THEN
    RAISE EXCEPTION 'supersession_invalid_source_ref' USING ERRCODE = '22023';
  END IF;
  IF p_new_content IS NULL OR btrim(p_new_content) = '' THEN
    RAISE EXCEPTION 'supersession_invalid_content' USING ERRCODE = '22023';
  END IF;
  IF p_actor IS NULL OR btrim(p_actor) = '' OR p_lane IS NULL OR btrim(p_lane) = '' THEN
    RAISE EXCEPTION 'supersession_invalid_provenance' USING ERRCODE = '22023';
  END IF;
  IF p_approval_ref IS NULL OR btrim(p_approval_ref) = '' OR p_reason IS NULL OR btrim(p_reason) = '' THEN
    RAISE EXCEPTION 'supersession_invalid_provenance' USING ERRCODE = '22023';
  END IF;
  IF p_new_metadata IS NULL OR jsonb_typeof(p_new_metadata) <> 'object' THEN
    RAISE EXCEPTION 'supersession_invalid_metadata' USING ERRCODE = '22023';
  END IF;
  IF p_expected_status IS DISTINCT FROM 'conflict' THEN
    RAISE EXCEPTION 'supersession_invalid_expected_status' USING ERRCODE = 'ZSI01';
  END IF;
  IF p_expected_old_payload_hash IS NULL OR p_new_payload_hash IS NULL
     OR p_expected_old_payload_hash !~ v_hex_re OR p_new_payload_hash !~ v_hex_re THEN
    RAISE EXCEPTION 'supersession_invalid_payload_hash' USING ERRCODE = '22023';
  END IF;
  IF p_expected_old_payload_hash = p_new_payload_hash THEN
    RAISE EXCEPTION 'supersession_payload_hashes_must_differ' USING ERRCODE = '22023';
  END IF;
  IF p_new_payload_canonical_json IS NULL THEN
    RAISE EXCEPTION 'supersession_invalid_payload_json' USING ERRCODE = '22023';
  END IF;
  v_computed_hash := encode(public.digest(pg_catalog.convert_to(p_new_payload_canonical_json, 'UTF8'), 'sha256'), 'hex');
  IF v_computed_hash IS DISTINCT FROM p_new_payload_hash THEN
    RAISE EXCEPTION 'supersession_payload_hash_mismatch' USING ERRCODE = '22023';
  END IF;
  IF (p_new_metadata ->> 'namespace') IS DISTINCT FROM p_namespace
     OR (p_new_metadata ->> 'source_ref') IS DISTINCT FROM p_source_ref
     OR (p_new_metadata ->> 'payload_sha256') IS DISTINCT FROM p_new_payload_hash THEN
    RAISE EXCEPTION 'supersession_metadata_identity_mismatch' USING ERRCODE = '22023';
  END IF;
  IF p_embedding_model_id IS NULL OR btrim(p_embedding_model_id) = ''
     OR p_tagger_model_id IS NULL OR btrim(p_tagger_model_id) = '' THEN
    RAISE EXCEPTION 'supersession_invalid_model_ids' USING ERRCODE = '22023';
  END IF;

  v_request_envelope := pg_catalog.jsonb_build_object(
    'v', 1,
    'request_id', p_request_id::text,
    'namespace', p_namespace,
    'source_ref', p_source_ref,
    'expected_status', p_expected_status,
    'expected_old_payload_hash', p_expected_old_payload_hash,
    'expected_old_thought_id', to_jsonb(p_expected_old_thought_id),
    'new_payload_canonical_json', p_new_payload_canonical_json,
    'new_payload_hash', p_new_payload_hash,
    'new_content', p_new_content,
    'new_metadata', p_new_metadata,
    'embedding_model_id', p_embedding_model_id,
    'tagger_model_id', p_tagger_model_id,
    'actor', p_actor,
    'lane', p_lane,
    'approval_ref', p_approval_ref,
    'reason', p_reason
  );
  v_request_preimage := v_request_envelope::text;
  v_request_digest := public.digest(pg_catalog.convert_to(v_request_preimage, 'UTF8'), 'sha256');

  v_lock_key := ('x' || substr(encode(v_request_digest, 'hex'), 1, 16))::bit(64)::bigint;
  PERFORM pg_advisory_xact_lock(v_lock_key);

  SELECT * INTO v_existing
  FROM public.argus_source_event_supersession_receipts
  WHERE request_id = p_request_id;
  IF FOUND THEN
    IF v_existing.request_digest = v_request_digest THEN
      RETURN v_existing.canonical_receipt_json
        || jsonb_build_object(
          'receipt_hash', encode(v_existing.receipt_digest, 'hex'),
          'replayed', true
        );
    END IF;
    RAISE EXCEPTION 'supersession_request_id_reuse' USING ERRCODE = 'ZSR01';
  END IF;

  -- Lock stable row
  SELECT * INTO v_row
  FROM public.argus_source_events
  WHERE namespace = p_namespace AND source_ref = p_source_ref
  FOR UPDATE;

  v_receipt_occurred_at := transaction_timestamp();
  v_occurred_text := pg_catalog.to_char(
    v_receipt_occurred_at AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
  );

  IF NOT FOUND THEN
    v_outcome := 'refused_expected_state';
    v_receipt_envelope := pg_catalog.jsonb_build_object(
      'v', 1,
      'request_id', p_request_id::text,
      'request_digest', encode(v_request_digest, 'hex'),
      'outcome', v_outcome,
      'stable_source_event_id', NULL,
      'namespace', p_namespace,
      'source_ref', p_source_ref,
      'expected_old_status', p_expected_status,
      'expected_old_payload_hash', p_expected_old_payload_hash,
      'expected_old_thought_id', to_jsonb(p_expected_old_thought_id),
      'observed_missing', true,
      'observed_old_status', NULL,
      'observed_old_payload_hash', NULL,
      'observed_old_thought_id', NULL,
      'new_payload_hash', p_new_payload_hash,
      'new_thought_id', NULL,
      'replaces_link_id', NULL,
      'gate_event_id', NULL,
      'embedding_job_id', NULL,
      'tag_job_generation_id', NULL,
      'embedding_model_id', p_embedding_model_id,
      'tagger_model_id', p_tagger_model_id,
      'actor', p_actor,
      'lane', p_lane,
      'approval_ref', p_approval_ref,
      'reason', p_reason,
      'authenticated_session_user', session_user::text,
      'occurred_at', v_occurred_text
    );
    v_receipt_digest := public.digest(pg_catalog.convert_to(v_receipt_envelope::text, 'UTF8'), 'sha256');
    INSERT INTO public.argus_source_event_supersession_receipts (
      request_id, request_digest, outcome, stable_source_event_id, namespace, source_ref,
      expected_old_status, expected_old_payload_hash, expected_old_thought_id,
      observed_missing, observed_old_status, observed_old_payload_hash, observed_old_thought_id,
      new_payload_hash, new_thought_id, replaces_link_id, gate_event_id, embedding_job_id,
      tag_job_generation_id, embedding_model_id, tagger_model_id, actor, lane, approval_ref,
      reason, authenticated_session_user, occurred_at, canonical_receipt_json, receipt_digest
    ) VALUES (
      p_request_id, v_request_digest, v_outcome, NULL, p_namespace, p_source_ref,
      p_expected_status, p_expected_old_payload_hash, p_expected_old_thought_id,
      true, NULL, NULL, NULL,
      p_new_payload_hash, NULL, NULL, NULL, NULL, NULL,
      p_embedding_model_id, p_tagger_model_id, p_actor, p_lane, p_approval_ref,
      p_reason, session_user::text, v_receipt_occurred_at, v_receipt_envelope, v_receipt_digest
    );
    RETURN v_receipt_envelope || jsonb_build_object(
      'receipt_hash', encode(v_receipt_digest, 'hex'),
      'replayed', false
    );
  END IF;

  IF v_row.status IS DISTINCT FROM p_expected_status
     OR v_row.payload_hash IS DISTINCT FROM p_expected_old_payload_hash
     OR v_row.thought_id IS DISTINCT FROM p_expected_old_thought_id THEN
    v_outcome := 'refused_expected_state';
    v_receipt_envelope := pg_catalog.jsonb_build_object(
      'v', 1,
      'request_id', p_request_id::text,
      'request_digest', encode(v_request_digest, 'hex'),
      'outcome', v_outcome,
      'stable_source_event_id', v_row.id::text,
      'namespace', p_namespace,
      'source_ref', p_source_ref,
      'expected_old_status', p_expected_status,
      'expected_old_payload_hash', p_expected_old_payload_hash,
      'expected_old_thought_id', to_jsonb(p_expected_old_thought_id),
      'observed_missing', false,
      'observed_old_status', v_row.status,
      'observed_old_payload_hash', v_row.payload_hash,
      'observed_old_thought_id', to_jsonb(v_row.thought_id),
      'new_payload_hash', p_new_payload_hash,
      'new_thought_id', NULL,
      'replaces_link_id', NULL,
      'gate_event_id', NULL,
      'embedding_job_id', NULL,
      'tag_job_generation_id', NULL,
      'embedding_model_id', p_embedding_model_id,
      'tagger_model_id', p_tagger_model_id,
      'actor', p_actor,
      'lane', p_lane,
      'approval_ref', p_approval_ref,
      'reason', p_reason,
      'authenticated_session_user', session_user::text,
      'occurred_at', v_occurred_text
    );
    v_receipt_digest := public.digest(pg_catalog.convert_to(v_receipt_envelope::text, 'UTF8'), 'sha256');
    INSERT INTO public.argus_source_event_supersession_receipts (
      request_id, request_digest, outcome, stable_source_event_id, namespace, source_ref,
      expected_old_status, expected_old_payload_hash, expected_old_thought_id,
      observed_missing, observed_old_status, observed_old_payload_hash, observed_old_thought_id,
      new_payload_hash, new_thought_id, replaces_link_id, gate_event_id, embedding_job_id,
      tag_job_generation_id, embedding_model_id, tagger_model_id, actor, lane, approval_ref,
      reason, authenticated_session_user, occurred_at, canonical_receipt_json, receipt_digest
    ) VALUES (
      p_request_id, v_request_digest, v_outcome, v_row.id, p_namespace, p_source_ref,
      p_expected_status, p_expected_old_payload_hash, p_expected_old_thought_id,
      false, v_row.status, v_row.payload_hash, v_row.thought_id,
      p_new_payload_hash, NULL, NULL, NULL, NULL, NULL,
      p_embedding_model_id, p_tagger_model_id, p_actor, p_lane, p_approval_ref,
      p_reason, session_user::text, v_receipt_occurred_at, v_receipt_envelope, v_receipt_digest
    );
    RETURN v_receipt_envelope || jsonb_build_object(
      'receipt_hash', encode(v_receipt_digest, 'hex'),
      'replayed', false
    );
  END IF;

  SELECT * INTO v_old_thought FROM public.thoughts WHERE id = p_expected_old_thought_id FOR UPDATE;
  IF NOT FOUND OR v_old_thought.retracted_at IS NOT NULL THEN
    RAISE EXCEPTION 'supersession_old_thought_unavailable' USING ERRCODE = 'P0001';
  END IF;

  v_new_meta := p_new_metadata;

  BEGIN
    SELECT * INTO v_capture
    FROM public.capture_thought_gated(
      v_old_thought.scope,
      p_new_content,
      v_old_thought.source,
      v_new_meta,
      v_old_thought.created_at,
      NULL::vector,
      p_embedding_model_id,
      1,
      jsonb_build_object('supersession', true, 'request_id', p_request_id::text),
      NULL, NULL, NULL, NULL,
      '[]'::jsonb,
      p_tagger_model_id,
      'source_event_supersession',
      p_request_id::text,
      NULL
    );
    IF v_capture.action = 'exact_duplicate' OR v_capture.thought_id IS NULL THEN
      RAISE EXCEPTION USING ERRCODE = 'ZSD01', MESSAGE = 'supersession_exact_content_duplicate';
    END IF;
  EXCEPTION
    WHEN SQLSTATE 'ZSD01' THEN
      v_exact_content_refusal := true;
  END;

  IF v_exact_content_refusal THEN
    v_outcome := 'refused_exact_content_duplicate';
    v_receipt_envelope := pg_catalog.jsonb_build_object(
      'v', 1,
      'request_id', p_request_id::text,
      'request_digest', encode(v_request_digest, 'hex'),
      'outcome', v_outcome,
      'stable_source_event_id', v_row.id::text,
      'namespace', p_namespace,
      'source_ref', p_source_ref,
      'expected_old_status', p_expected_status,
      'expected_old_payload_hash', p_expected_old_payload_hash,
      'expected_old_thought_id', to_jsonb(p_expected_old_thought_id),
      'observed_missing', false,
      'observed_old_status', v_row.status,
      'observed_old_payload_hash', v_row.payload_hash,
      'observed_old_thought_id', to_jsonb(v_row.thought_id),
      'new_payload_hash', p_new_payload_hash,
      'new_thought_id', NULL,
      'replaces_link_id', NULL,
      'gate_event_id', NULL,
      'embedding_job_id', NULL,
      'tag_job_generation_id', NULL,
      'embedding_model_id', p_embedding_model_id,
      'tagger_model_id', p_tagger_model_id,
      'actor', p_actor,
      'lane', p_lane,
      'approval_ref', p_approval_ref,
      'reason', p_reason,
      'authenticated_session_user', session_user::text,
      'occurred_at', v_occurred_text
    );
    v_receipt_digest := public.digest(pg_catalog.convert_to(v_receipt_envelope::text, 'UTF8'), 'sha256');
    INSERT INTO public.argus_source_event_supersession_receipts (
      request_id, request_digest, outcome, stable_source_event_id, namespace, source_ref,
      expected_old_status, expected_old_payload_hash, expected_old_thought_id,
      observed_missing, observed_old_status, observed_old_payload_hash, observed_old_thought_id,
      new_payload_hash, new_thought_id, replaces_link_id, gate_event_id, embedding_job_id,
      tag_job_generation_id, embedding_model_id, tagger_model_id, actor, lane, approval_ref,
      reason, authenticated_session_user, occurred_at, canonical_receipt_json, receipt_digest
    ) VALUES (
      p_request_id, v_request_digest, v_outcome, v_row.id, p_namespace, p_source_ref,
      p_expected_status, p_expected_old_payload_hash, p_expected_old_thought_id,
      false, v_row.status, v_row.payload_hash, v_row.thought_id,
      p_new_payload_hash, NULL, NULL, NULL, NULL, NULL,
      p_embedding_model_id, p_tagger_model_id, p_actor, p_lane, p_approval_ref,
      p_reason, session_user::text, v_receipt_occurred_at, v_receipt_envelope, v_receipt_digest
    );
    RETURN v_receipt_envelope || jsonb_build_object(
      'receipt_hash', encode(v_receipt_digest, 'hex'),
      'replayed', false
    );
  END IF;

  v_retract := public.retract_thought_serialized(
    p_expected_old_thought_id,
    'supersession:' || p_request_id::text,
    'source_event_supersession'
  );
  IF (v_retract ->> 'status') IS DISTINCT FROM 'retracted' THEN
    RAISE EXCEPTION 'supersession_retract_failed:%', v_retract ->> 'status' USING ERRCODE = 'P0001';
  END IF;

  -- Insert replaces link via direct insert as gate owner (security definer)
  INSERT INTO public.thought_links (
    from_thought_id, relation, to_kind, to_thought_id, source, note
  ) VALUES (
    v_capture.thought_id, 'replaces', 'thought', p_expected_old_thought_id, 'agent',
    left('supersession request=' || p_request_id::text || ' lane=' || p_lane, 500)
  ) RETURNING id INTO v_link_id;

  SELECT id INTO v_emb_id FROM public.pending_embeddings
  WHERE target_kind = 'thought' AND target_id = v_capture.thought_id
  ORDER BY enqueued_at DESC NULLS LAST LIMIT 1;
  SELECT tag_job_generation_id INTO v_tag_gen FROM public.pending_tags
  WHERE thought_id = v_capture.thought_id
  ORDER BY enqueued_at DESC NULLS LAST LIMIT 1;
  IF v_emb_id IS NULL OR v_tag_gen IS NULL OR v_capture.gate_event_id IS NULL THEN
    RAISE EXCEPTION 'supersession_queue_or_gate_missing' USING ERRCODE = 'P0001';
  END IF;

  UPDATE public.argus_source_events
  SET payload_hash = p_new_payload_hash,
      thought_id = v_capture.thought_id,
      status = 'stored',
      error = NULL,
      last_seen_at = transaction_timestamp(),
      metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
        'supersession', jsonb_build_object(
          'request_id', p_request_id::text,
          'old_payload_hash', p_expected_old_payload_hash,
          'new_payload_hash', p_new_payload_hash,
          'old_thought_id', p_expected_old_thought_id::text,
          'new_thought_id', v_capture.thought_id::text,
          'link_id', v_link_id::text,
          'gate_event_id', v_capture.gate_event_id::text,
          'actor', p_actor,
          'lane', p_lane,
          'approval_ref', p_approval_ref,
          'occurred_at', v_occurred_text
        )
      )
  WHERE id = v_row.id
    AND status IS NOT DISTINCT FROM p_expected_status
    AND payload_hash IS NOT DISTINCT FROM p_expected_old_payload_hash
    AND thought_id IS NOT DISTINCT FROM p_expected_old_thought_id;
  GET DIAGNOSTICS v_rc = ROW_COUNT;
  IF v_rc <> 1 THEN
    RAISE EXCEPTION 'supersession_stable_row_update_count:%', v_rc USING ERRCODE = 'P0001';
  END IF;

  v_outcome := 'applied';
  v_receipt_envelope := pg_catalog.jsonb_build_object(
    'v', 1,
    'request_id', p_request_id::text,
    'request_digest', encode(v_request_digest, 'hex'),
    'outcome', v_outcome,
    'stable_source_event_id', v_row.id::text,
    'namespace', p_namespace,
    'source_ref', p_source_ref,
    'expected_old_status', p_expected_status,
    'expected_old_payload_hash', p_expected_old_payload_hash,
    'expected_old_thought_id', to_jsonb(p_expected_old_thought_id),
    'observed_missing', false,
    'observed_old_status', v_row.status,
    'observed_old_payload_hash', v_row.payload_hash,
    'observed_old_thought_id', to_jsonb(v_row.thought_id),
    'new_payload_hash', p_new_payload_hash,
    'new_thought_id', v_capture.thought_id::text,
    'replaces_link_id', v_link_id::text,
    'gate_event_id', v_capture.gate_event_id::text,
    'embedding_job_id', v_emb_id::text,
    'tag_job_generation_id', v_tag_gen::text,
    'embedding_model_id', p_embedding_model_id,
    'tagger_model_id', p_tagger_model_id,
    'actor', p_actor,
    'lane', p_lane,
    'approval_ref', p_approval_ref,
    'reason', p_reason,
    'authenticated_session_user', session_user::text,
    'occurred_at', v_occurred_text
  );
  v_receipt_digest := public.digest(pg_catalog.convert_to(v_receipt_envelope::text, 'UTF8'), 'sha256');
  INSERT INTO public.argus_source_event_supersession_receipts (
    request_id, request_digest, outcome, stable_source_event_id, namespace, source_ref,
    expected_old_status, expected_old_payload_hash, expected_old_thought_id,
    observed_missing, observed_old_status, observed_old_payload_hash, observed_old_thought_id,
    new_payload_hash, new_thought_id, replaces_link_id, gate_event_id, embedding_job_id,
    tag_job_generation_id, embedding_model_id, tagger_model_id, actor, lane, approval_ref,
    reason, authenticated_session_user, occurred_at, canonical_receipt_json, receipt_digest
  ) VALUES (
    p_request_id, v_request_digest, v_outcome, v_row.id, p_namespace, p_source_ref,
    p_expected_status, p_expected_old_payload_hash, p_expected_old_thought_id,
    false, v_row.status, v_row.payload_hash, v_row.thought_id,
    p_new_payload_hash, v_capture.thought_id, v_link_id, v_capture.gate_event_id, v_emb_id,
    v_tag_gen, p_embedding_model_id, p_tagger_model_id, p_actor, p_lane, p_approval_ref,
    p_reason, session_user::text, v_receipt_occurred_at, v_receipt_envelope, v_receipt_digest
  );

  RETURN v_receipt_envelope || jsonb_build_object(
    'receipt_hash', encode(v_receipt_digest, 'hex'),
    'replayed', false
  );
END;
$fn$;

ALTER FUNCTION public.supersede_argus_source_event(
  uuid, text, text, text, text, uuid, text, text, text, jsonb, text, text, text, text, text, text
) OWNER TO kengram_gate_owner;

REVOKE ALL ON FUNCTION public.supersede_argus_source_event(
  uuid, text, text, text, text, uuid, text, text, text, jsonb, text, text, text, text, text, text
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.supersede_argus_source_event(
  uuid, text, text, text, text, uuid, text, text, text, jsonb, text, text, text, text, text, text
) TO kengram_rt_supersession;

-- Gate owner needs EXECUTE on nested functions (already has as owner typically)
GRANT kengram_runtime TO kengram_rt_supersession WITH INHERIT TRUE, SET FALSE;

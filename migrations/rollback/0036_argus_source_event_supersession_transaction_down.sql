-- Fail-closed rollback for 0036 supersession transaction
DO $$
DECLARE
  n bigint;
BEGIN
  SELECT count(*) INTO n FROM public.argus_source_event_supersession_receipts;
  IF n > 0 THEN
    RAISE EXCEPTION 'supersession_rollback_refused_receipts_present:%', n;
  END IF;
END$$;

DROP FUNCTION IF EXISTS public.supersede_argus_source_event(
  uuid, text, text, text, text, uuid, text, text, text, jsonb, text, text, text, text, text, text
);
DROP TRIGGER IF EXISTS argus_source_event_supersession_receipts_no_update
  ON public.argus_source_event_supersession_receipts;
DROP FUNCTION IF EXISTS public.argus_source_event_supersession_receipts_immutable();
DROP TABLE IF EXISTS public.argus_source_event_supersession_receipts;
-- B5: remove successor key-count helper if present (idempotent; residue-free rollback)
DROP FUNCTION IF EXISTS public.supersession_receipt_json_key_count(jsonb);
DELETE FROM public.corpus_hygiene_gate_settings
 WHERE principal_name = 'kengram_rt_supersession'
   AND producer_class = 'source_event_supersession';
DELETE FROM public.corpus_hygiene_producer_principals
 WHERE principal_name = 'kengram_rt_supersession';
DROP ROLE IF EXISTS kengram_rt_supersession;

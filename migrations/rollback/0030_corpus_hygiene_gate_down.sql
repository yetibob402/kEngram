-- Fail closed once Delivery-1 permanent ledgers contain evidence.
DO $down$
BEGIN
    IF EXISTS (SELECT 1 FROM thought_ingest_gate_events)
       OR EXISTS (SELECT 1 FROM thought_relation_request_events) THEN
        RAISE EXCEPTION '0030 rollback refused: permanent corpus-hygiene ledger is populated';
    END IF;
END
$down$;

DROP TRIGGER IF EXISTS thought_links_require_serialized_writer ON thought_links;
DROP TRIGGER IF EXISTS thoughts_require_gated_writer ON thoughts;
DROP FUNCTION IF EXISTS capture_thought_gated_passthrough(text,text,text,jsonb,timestamptz,text,jsonb,text,text,text,jsonb,jsonb,text,text,text);
DROP FUNCTION IF EXISTS capture_thought_gated(text,text,text,jsonb,timestamptz,vector,text,integer,jsonb,text,text,text,jsonb,jsonb,text,text,text,text);
DROP FUNCTION IF EXISTS retract_thought_serialized(uuid,text,text);
DROP FUNCTION IF EXISTS mutate_thought_relations_serialized(jsonb,text,text,text,jsonb,text);
DROP FUNCTION IF EXISTS lock_thought_relation_endpoints(uuid[],boolean);
DROP FUNCTION IF EXISTS thought_links_require_serialized_writer();
DROP FUNCTION IF EXISTS thoughts_require_gated_writer();
DROP FUNCTION IF EXISTS corpus_hygiene_novelty_ratio(text,text);
DROP FUNCTION IF EXISTS corpus_hygiene_protected_atoms(text);

DROP TABLE IF EXISTS thought_relation_request_events;
DROP TABLE IF EXISTS thought_ingest_gate_events;
DROP TABLE IF EXISTS corpus_hygiene_gate_settings;
DROP TABLE IF EXISTS corpus_hygiene_producer_principals;

DROP INDEX IF EXISTS pending_tags_generation_idx;
ALTER TABLE pending_tags DROP COLUMN IF EXISTS tag_job_generation_id;

REVOKE kengram_runtime FROM kengram_rt_native_mcp;
REVOKE kengram_runtime FROM kengram_rt_session;
REVOKE kengram_runtime FROM kengram_rt_telegram;
REVOKE kengram_runtime FROM kengram_rt_agent_comms;
REVOKE kengram_runtime FROM kengram_rt_reviews;
REVOKE kengram_runtime FROM kengram_rt_specs;
REVOKE kengram_runtime FROM kengram_rt_openclaw;
REVOKE kengram_runtime FROM kengram_rt_hive;
REVOKE kengram_runtime FROM kengram_rt_mba_archive;
REVOKE kengram_runtime FROM kengram_rt_phase4;
REVOKE kengram_runtime FROM kengram_rt_maintenance_import;

-- Remove ACL dependencies installed on pre-existing tables/functions before
-- dropping the cluster roles. These roles are created exclusively by 0030.
DROP OWNED BY kengram_gate_owner;
DROP OWNED BY kengram_runtime;

DROP ROLE IF EXISTS kengram_rt_native_mcp;
DROP ROLE IF EXISTS kengram_rt_session;
DROP ROLE IF EXISTS kengram_rt_telegram;
DROP ROLE IF EXISTS kengram_rt_agent_comms;
DROP ROLE IF EXISTS kengram_rt_reviews;
DROP ROLE IF EXISTS kengram_rt_specs;
DROP ROLE IF EXISTS kengram_rt_openclaw;
DROP ROLE IF EXISTS kengram_rt_hive;
DROP ROLE IF EXISTS kengram_rt_mba_archive;
DROP ROLE IF EXISTS kengram_rt_phase4;
DROP ROLE IF EXISTS kengram_rt_maintenance_import;
DROP ROLE IF EXISTS kengram_runtime;
DROP ROLE IF EXISTS kengram_gate_owner;

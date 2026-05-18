-- M6.x: drop tags.relations from persisted JSONB on existing rows.
--
-- The tagger drainer (apply_tagger_relations in engram-mcp/src/drain.rs)
-- wrote LLM-emitted relations to TWO stores: thoughts.tags.relations
-- (raw frozen emission JSONB, preserving duplicates) AND thought_links
-- rows with source='tagger' (deduplicated queryable graph). Verified
-- 2026-05-18 on the live corpus: thought 15533025 had 3 entries in
-- tags.relations and 3 corresponding thought_links rows with
-- source='tagger'; thought b533ebac had 2 duplicate entries in
-- tags.relations that the partial unique index on thought_links
-- collapsed to 1 row.
--
-- thought_links is the canonical store (queryable, deduplicated, with
-- soft-delete + link_source discriminator). The tags.relations copy is
-- pure duplication. This migration removes the key from existing rows;
-- the Rust-side Tags struct also loses the field in the same ship, so
-- new captures will not re-populate it.
--
-- The LLM emission shape is unchanged — the prompt + JSON schema still
-- ask for relations. Drainer-side parsing splits the response into
-- (Tags, Vec<ExtractedRelation>); only the tags portion is persisted to
-- the JSONB column. apply_tagger_relations continues to write to
-- thought_links from the relations portion.
--
-- No tagger version bump: the tagging output content (people, entities,
-- topics, kind, etc., and the relations that land in thought_links) is
-- identical under the old and new code paths. tags_extractor_version
-- stays at whatever it was on each row.

WITH updated AS (
    UPDATE thoughts
    SET tags = tags - 'relations'
    WHERE tags ? 'relations'
    RETURNING 1
)
INSERT INTO migration_audit (migration, rows_touched, notes)
SELECT
    '0011_drop_tags_relations',
    COUNT(*),
    'Removed tags.relations key from persisted JSONB. thought_links source=tagger remains the canonical store. No tagger version bump; LLM emission shape unchanged.'
FROM updated;

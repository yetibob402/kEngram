-- M5.1: extend the closed relation vocabulary with `supports`.
--
-- Day-one M5 dogfood revealed `references` was over-firing on what was
-- actually evidence / corroboration — "experimental result confirming a
-- claim" got conflated with "weak prose cite" and "summary cite." The
-- split separates "I cite for context" (`references`) from "I confirm a
-- claim" (`supports`).
--
-- This migration is a pure CHECK constraint relax. Existing rows are
-- unaffected — every value in the old set is still valid. Operators don't
-- need to re-link anything; the new `supports` value just becomes
-- available for new (or re-asserted) edges.

ALTER TABLE thought_links
    DROP CONSTRAINT thought_links_relation_check;

ALTER TABLE thought_links
    ADD CONSTRAINT thought_links_relation_check
        CHECK (relation IN (
            'replaces',
            'requires',
            'references',
            'supports',
            'belongs_to',
            'decided_by',
            'refines'
        ));

-- M5: selective relations — thought-to-thought graph layer with a closed
-- relation vocabulary. See docs/milestones/m5-selective-relations.md for
-- the architectural rationale.

-- 1. Create the thought_links edge table. Agent-supplied at link time;
--    tagger-extracted relations are M5.x (the `source` column has the
--    extension point pre-baked). Heterogeneous targets (to-entity,
--    to-person, to-URL) are also deferred — M5 is thought-to-thought
--    only, hence both endpoints REFERENCE thoughts(id).
CREATE TABLE thought_links (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    from_thought_id UUID        NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    relation        TEXT        NOT NULL,
    to_thought_id   UUID        NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    source          TEXT        NOT NULL DEFAULT 'agent',
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Closed relation vocabulary; TEXT + CHECK (not Postgres ENUM) so the
    -- set can be extended/revised by a future migration without an
    -- ALTER TYPE dance.
    CONSTRAINT thought_links_relation_check
        CHECK (relation IN (
            'replaces',
            'requires',
            'references',
            'belongs_to',
            'decided_by',
            'refines'
        )),
    -- Source enum: M5 inserts 'agent' only; 'tagger' reserved for M5.x.
    CONSTRAINT thought_links_source_check
        CHECK (source IN ('agent', 'tagger')),
    -- Self-loops are nonsensical for this vocabulary.
    CONSTRAINT thought_links_no_self_reference
        CHECK (from_thought_id <> to_thought_id),
    -- Idempotency: re-asserting the same edge returns the existing row
    -- rather than creating a duplicate (matches `capture`'s fingerprint
    -- dedup pattern).
    CONSTRAINT thought_links_unique_edge
        UNIQUE (from_thought_id, relation, to_thought_id)
);

-- 2. Direction-specific indexes. Both directions are common at retrieval
--    time ("what does this thought refine?" + "what refines this thought?"),
--    so we index both endpoints. The (endpoint, relation) composite lets
--    relation-filtered traversal go straight to the index without a
--    secondary filter pass.
CREATE INDEX thought_links_from_idx ON thought_links (from_thought_id, relation);
CREATE INDEX thought_links_to_idx   ON thought_links (to_thought_id, relation);

-- M5.2: heterogeneous link targets.
--
-- Day-one M5 dogfood surfaced the case where the natural target of a
-- relation isn't a thought — Probe 2A and 2B were sibling variants under
-- "Probe 2 experiment," which isn't itself captured as a thought. The
-- `belongs_to` relation had nothing to point at. Workarounds (capture a
-- placeholder thought first) impose a capture-time tax and don't
-- retroactively fix existing data.
--
-- The fix is polymorphic targets on `thought_links`. A link's `from` side
-- is still always a thought (engram's graph layer is anchored on the
-- thoughts table); the `to` side can be a thought, an entity, a person,
-- or a URL. The discriminator column `to_kind` + per-kind value columns +
-- a generated `to_value` column anchor the unique-edge constraint across
-- all four kinds.
--
-- Existing rows keep `to_kind = 'thought'` and the new columns NULL —
-- this migration is non-destructive.

ALTER TABLE thought_links
    ADD COLUMN to_kind   TEXT NOT NULL DEFAULT 'thought'
        CHECK (to_kind IN ('thought', 'entity', 'person', 'url')),
    ADD COLUMN to_entity TEXT NULL,
    ADD COLUMN to_person TEXT NULL,
    ADD COLUMN to_url    TEXT NULL,
    ALTER COLUMN to_thought_id DROP NOT NULL,
    ADD COLUMN to_value  TEXT GENERATED ALWAYS AS (
        COALESCE(to_thought_id::text, to_entity, to_person, to_url)
    ) STORED;

-- Drop the constraints we're replacing.
ALTER TABLE thought_links
    DROP CONSTRAINT thought_links_unique_edge,
    DROP CONSTRAINT thought_links_no_self_reference;

-- Replacement constraints.
ALTER TABLE thought_links
    -- Uniqueness now includes to_kind so the same (from, relation, value)
    -- triple is allowed across different target kinds (e.g., link to a
    -- thought whose UUID happens to be a URL — pathological but legal).
    ADD CONSTRAINT thought_links_unique_edge
        UNIQUE (from_thought_id, relation, to_kind, to_value),

    -- Target validity: exactly one per-kind column is set, matching to_kind.
    ADD CONSTRAINT thought_links_target_valid CHECK (
        (to_kind = 'thought' AND to_thought_id IS NOT NULL AND to_entity IS NULL AND to_person IS NULL AND to_url IS NULL) OR
        (to_kind = 'entity'  AND to_entity      IS NOT NULL AND to_thought_id IS NULL AND to_person IS NULL AND to_url IS NULL) OR
        (to_kind = 'person'  AND to_person      IS NOT NULL AND to_thought_id IS NULL AND to_entity IS NULL AND to_url IS NULL) OR
        (to_kind = 'url'     AND to_url         IS NOT NULL AND to_thought_id IS NULL AND to_entity IS NULL AND to_person IS NULL)
    ),

    -- Self-loops still nonsensical for thought→thought, but allowed (or
    -- rather, irrelevant) for non-thought targets.
    ADD CONSTRAINT thought_links_no_self_reference CHECK (
        to_kind <> 'thought' OR from_thought_id <> to_thought_id
    ),

    -- Lightweight URL validation. Not RFC-grade; just keeps obvious
    -- non-URLs (free-form text, javascript:, etc.) out of the column.
    ADD CONSTRAINT thought_links_url_format CHECK (
        to_url IS NULL OR to_url ~ '^https?://'
    );

-- Index supporting outbound traversal filtered by target kind
-- ("what URLs has this thought referenced?", etc.).
CREATE INDEX thought_links_from_kind_idx
    ON thought_links (from_thought_id, to_kind, relation);

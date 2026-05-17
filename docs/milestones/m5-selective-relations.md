# M5 — Selective relations (thought-to-thought graph layer)

**Status:** ✅ shipped 2026-05-17. **M5.1 iteration** shipped 2026-05-17 (see end of doc); vocabulary now has seven relations.

**One-line:** thought-to-thought edges in a closed six-relation vocabulary, agent-supplied via three new MCP tools.

## Motivation

The M4.1 dogfood produced a citation chain: `137dba1d` refined `6d2ef58e` refined `8a533e15` (and `137dba1d` cited `6d2ef58e` explicitly in its prose). That structure was implicit — encoded in the body text of each thought, discoverable only by reading. M5 makes it first-class: edges in a closed vocabulary that captures the relational structure that actually shows up in conversation memory.

Six relations cover what M4.1 dogfood revealed operators wanting to express:

- **`refines`** — newer thought refines an earlier one (both stand; the newer represents updated thinking). The citation-chain pattern.
- **`replaces`** — newer thought supersedes an earlier one (decision changed; retrieval prefers newer).
- **`references`** — this thought points at another for context (citation, follow-up, related observation).
- **`requires`** — dependency relation (decision presupposes a constraint; refinement presupposes an earlier finding).
- **`belongs_to`** — membership/containment (a finding under a parent thread; a decision under a session).
- **`decided_by`** — provenance: this thought is a decision attributable to another (person-note, session-anchor).

The vocabulary is selective, not general. M3's facts pipeline tried open-vocabulary `(subject, predicate, object)` extraction; the M3 Phase D dogfood showed predicates broke under small-model limitations (open relation slot is too hard). M5 commits to a closed six-element vocabulary that's predictable for queries, tractable for downstream tooling, and avoids the failure mode that retired M3.

## Architecture

### Schema (migration 0007)

```sql
CREATE TABLE thought_links (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    from_thought_id UUID        NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    relation        TEXT        NOT NULL,
    to_thought_id   UUID        NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    source          TEXT        NOT NULL DEFAULT 'agent',
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (relation IN ('replaces','requires','references','belongs_to','decided_by','refines')),
    CHECK (source IN ('agent','tagger')),
    CHECK (from_thought_id <> to_thought_id),
    UNIQUE (from_thought_id, relation, to_thought_id)
);

CREATE INDEX thought_links_from_idx ON thought_links (from_thought_id, relation);
CREATE INDEX thought_links_to_idx   ON thought_links (to_thought_id, relation);
```

Decisions:
- **TEXT + CHECK** for the relation enum rather than Postgres ENUM. Easier to extend/revise in a future migration without ALTER TYPE.
- **UNIQUE (from, relation, to)** makes `link_thoughts` idempotent on the triple, mirroring `capture`'s fingerprint dedup.
- **ON DELETE CASCADE** is safe because retraction is soft. Hard-delete of a thought (not currently exposed by the system) would CASCADE its edges; soft-retraction leaves edges intact with `retracted: true` surfaced in responses.
- **`source` column** distinguishes agent-supplied (M5) from tagger-extracted (M5.x). M5 inserts only `'agent'`.
- **`note` column** parallels `thoughts.retracted_reason` — optional free-text rationale capped at 1000 chars by the MCP layer.

### Core types

`crates/engram-core/src/relation.rs`:

```rust
pub enum RelationKind { Replaces, Requires, References, BelongsTo, DecidedBy, Refines }
pub enum LinkSource { Agent, Tagger }                  // Tagger reserved for M5.x
pub enum LinkDirection { Outbound, Inbound, Both }     // default Both
pub struct LinkId(uuid::Uuid);
pub struct ThoughtLink { id, from, relation, to, source, note, created_at }
```

All snake_case-serialized; closed enums; `Display` + `FromStr` impls for clean MCP wire conversion.

### Storage helpers

`crates/engram-storage/src/lib.rs`:

- `insert_link(pool, from, relation, to, source, note) -> (LinkId, is_new)` — `ON CONFLICT ON CONSTRAINT thought_links_unique_edge DO NOTHING` then re-fetch on conflict to return the stable `link_id`.
- `delete_link(pool, from, relation, to) -> existed` — bare DELETE.
- `fetch_related_thoughts(pool, thought_id, relations?, direction) -> Vec<RelatedThought>` — LEFT JOIN to `thoughts` per direction; relation filter via `cardinality($N::text[]) = 0 OR relation = ANY($N::text[])` (avoids sqlx Option<Vec<String>> compile-time pain); `Both` direction concatenates outbound + inbound and re-sorts by `link_created_at DESC`.

`RelatedThought` is the enrichment struct — carries the edge metadata plus the related thought's `scope`, `content`, `created_at`, and `retracted` flag.

### MCP surface

Three new tools wired through `crates/engram-mcp/src/{link.rs,relate.rs,server.rs}`:

- **`link_thoughts(from_thought_id, relation, to_thought_id, note?)`** — pre-validates self-link, note length (≤1000 chars), and endpoint existence with distinct `LinkError` variants for actionable error messages. Idempotent on the triple. Returns `{link_id, from_thought_id, relation, to_thought_id, is_new}`.
- **`unlink_thoughts(from_thought_id, relation, to_thought_id)`** — DELETE the edge. Idempotent on already-deleted. Returns `{existed}`.
- **`get_related_thoughts(thought_id, relations?, direction?)`** — grouped `outbound` + `inbound` arrays. Each entry: `{link_id, relation, thought_id, scope, content_preview, content_truncated, thought_created_at, link_created_at, link_source, note, retracted}`. Content preview is capped at 400 chars (char-boundary-safe).

`SERVER_INSTRUCTIONS` const documents the closed vocabulary and the three tools; regression test pins the documentation so future edits don't accidentally drop the orientation block.

### CLI

No new subcommands. Linking is MCP-only — operators link via Claude Code/Desktop. If future dogfood reveals an `engram link` CLI shortcut is wanted, it's a small additional iteration.

## Files

**New:**
- `migrations/0007_thought_links.sql`
- `crates/engram-core/src/relation.rs`
- `crates/engram-mcp/src/link.rs`
- `crates/engram-mcp/src/relate.rs`
- `docs/milestones/m5-selective-relations.md` (this file)

**Modified:**
- `crates/engram-core/src/lib.rs` — re-exports
- `crates/engram-storage/src/lib.rs` — storage helpers + 12 integration tests
- `crates/engram-mcp/src/server.rs` — three `#[tool]` methods, SERVER_INSTRUCTIONS, regression test
- `crates/engram-mcp/src/lib.rs` — module declarations + re-exports
- `README.md` — roadmap table (M5/M6/M7); MCP tool surface table; new "How relations work" section
- `docs/engram-design-v0.md` — §3.5 roadmap renumbered; §6.6 selective relations sidecar added; §8 MCP surface table updated; §9 type listing; revision history

**Renamed (via `git mv`):**
- `docs/milestones/m5-artifacts.md` → `m6-artifacts.md`
- `docs/milestones/m6-operational-maturity.md` → `m7-operational-maturity.md`

## Verification

- `cargo build --workspace` — clean.
- `cargo test --workspace` — clean. New test count: 9 in `engram-core` (relation module), 12 in `engram-storage` (insert/delete/fetch + cascade/retracted edge cases), 7 in `engram-mcp::link` (happy path + each `LinkError` variant), 6 in `engram-mcp::relate` (direction filter, relation filter, content truncation, retracted state, missing-thought error), 1 extended in `engram-mcp::server` (SERVER_INSTRUCTIONS pin).
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo fmt --all -- --check` — clean.
- Migration 0007 applies cleanly; `\d thought_links` shows the constraints + indexes documented above.
- Smoke (post-merge):
  - Capture two thoughts; link them via `link_thoughts`; confirm `is_new: true`, idempotency on second call.
  - Self-link rejected with `SelfLink` error.
  - Non-existent endpoint rejected with `FromThoughtMissing` / `ToThoughtMissing`.
  - `get_related_thoughts` returns `outbound`/`inbound` groups correctly.
  - Retract one endpoint; edge still surfaces with `retracted: true`.
  - `unlink_thoughts` returns `existed: true` then `existed: false`.

## Dogfood plan (post-merge)

1. **Retroactively link the M4.1 citation chain.** Call `link_thoughts(137dba1d, "refines", 6d2ef58e)` and `link_thoughts(6d2ef58e, "refines", 8a533e15)`. Verify `get_related_thoughts(8a533e15, direction: "inbound")` walks the chain.
2. **Observe relation usage frequency.** Over the next dogfood week, watch which of the six relations get used and which don't. The vocabulary was picked by intuition; usage will validate or refine. Likely outcomes:
   - `refines` and `replaces` will be the most common.
   - `references` and `belongs_to` will be moderately common.
   - `requires` and `decided_by` may be rare — if they're never used, candidates for removal in a future iteration.
3. **Watch for the "I want to link to a non-thought" failure mode.** Every time you wish you could link to an entity-tag, a person, or a URL, that's a vote for the M5.x heterogeneous-targets work.
4. **Watch for the "I want the LLM to auto-link" failure mode.** Every time you manually link in a way the prose already implied, that's a vote for M5.x tagger-extracted relations.
5. **Capture findings as thoughts in `engram.m3.dogfood`** — and link them. Meta-dogfood: the relations layer documenting itself via its own substrate.

## Risks

- **Closed vocab is a guess.** The six relations come from intuition + the M4.1 citation chain. Dogfood may reveal redundancy (`references` vs `refines`?) or missing relations (`contradicts`? `extends`?). Mitigation: schema uses TEXT + CHECK rather than Postgres ENUM, so adding/removing relations is a single ALTER + data-migration migration.
- **No backfill.** Existing thoughts have no edges; the dogfood-week graph is whatever you retroactively link. Acceptable — relations are forward-looking.
- **Single-edge link tool may chafe at scale.** If dogfood reveals operators wanting to bulk-link, array-shaped tools follow in a small M5.x iteration.
- **`get_related_thoughts` returns 400-char previews.** At single-thought traversal scale (rarely > ~10 direct edges), the response stays bounded. If walks grow large enough to matter, paginate in a follow-up.

## Out of scope (deliberate)

- Tagger-extracted relations → M5.x (LLM finds the edge from prose; requires entity resolution + its own dogfood loop).
- Heterogeneous targets (to-entity, to-person, to-URL) → M5.x or M6 (polymorphic schema work).
- Bulk-link/unlink MCP tools → M5.x if usage demands.
- Multi-hop traversal (`get_thoughts_n_hops_away`) → M5.x.
- `engram link` CLI subcommand → if dogfood demands.
- Relation confidence/provenance metadata (M3-style) → not needed for agent-supplied; revisit when tagger-extraction lands.

## Decision log

- **Six relations, not five or seven.** Picked by reflecting on the M4.1 dogfood (citation chain → `refines`; supersession of v3 prompt by v4 → `replaces`; the M3 Phase D dogfood → `references`; "this depends on that finding" → `requires`; "this finding belongs to the dogfood thread" → `belongs_to`; "Ron decided this in session X" → `decided_by`). The set is large enough to cover what showed up, small enough to be enumerable.
- **TEXT + CHECK over Postgres ENUM.** ENUM types are awkward to evolve; TEXT + CHECK lets a future migration revise the vocabulary by simple ALTER TABLE.
- **UNIQUE on (from, relation, to).** Idempotency at the schema level mirrors `capture`'s content-fingerprint dedup. Re-asserting the same edge isn't an error; it's a no-op that returns the existing link_id.
- **Agent-supplied only at M5.** Tagger extraction is the more differentiating capability long-term, but it's its own design problem (entity resolution: which thought_id does "the earlier finding" refer to?). Shipping agent-supplied first gets the graph queries usable immediately while keeping the M5 scope tight.
- **Thought-to-thought only at M5.** Heterogeneous targets (to-entity, to-person, to-URL) are real future work but they multiply the schema surface (polymorphic target column or multiple FK columns) without paying off the citation-chain use case that motivated M5.
- **`get_related_thoughts` returns content_preview, not full content.** Previews keep response sizes bounded for callers building UI; the full content is one `get_thought` call away if needed.
- **Edges survive thought retraction.** Soft-retraction sets `retracted_at` but doesn't delete the row, so FK is still valid. Surfacing `retracted: true` in responses lets consumers decide whether to show, dim, or hide; we don't gate retrieval at the storage layer because that's a UX policy, not a data property.
- **No `link_id`-based deletion.** Edges are identified by their (from, relation, to) triple, not by link_id. Operators see edges as relationships, not as rows with surrogate keys. `unlink_thoughts(from, relation, to)` matches that mental model.

---

## M5.1 iteration (2026-05-17)

Day-one dogfood on the v1 vocabulary (17 agent-supplied edges + 2 captured findings) surfaced two priorities. M5.1 is a small additive iteration that addresses Priority 1 (vocabulary gap) and Priority 3 (tool description anti-patterns) from the dogfood notes. Priority 2 (heterogeneous targets) earned promotion from M5.x to a near-term M5.2 iteration; tracked separately.

### Priority 1: `references` was over-firing

Four functionally distinct edge types collapsed into `references` across 17 edges:
- **Weak cite** — passing prose mention (`6d2ef58e → 8a533e15`).
- **Experimental evidence / corroboration** — `74eb781c → 6d2ef58e`.
- **Summary cite** — result aggregates source data points (`74eb781c → 63ad01e0`, `74eb781c → 047d0ce8`).
- **Sibling grouping** — peer findings (`618f5a6b → 0ce53ec2`).

The `note` field carried the semantics in practice and `get_related_thoughts` returns notes, so careful consumers can disambiguate by inspection. But aggregation tooling that counts edges by relation type can't ask `relations: ["evidences"]` — it has to ask `relations: ["references"]` and string-match notes, which is fragile.

The cleanest factoring: split the evidence-vs-context divide. M5.1 adds **`supports`** as the seventh relation, separating "I cite for context" (`references`) from "I confirm a claim" (`supports` — experimental evidence, corroborating data, logical support). The other three overloads (weak cite, summary cite, sibling) are tolerable; the evidence-vs-context conflation was the one that bit hardest.

Why `supports` over `evidences` / `corroborates`:
- Reads cleanly in agent-natural sentence shape ("B supports A").
- Covers both experimental and logical support.
- Matches the closed-vocab pattern of existing relations (single English verb, snake_case-friendly).
- `evidences` is slightly stilted as a verb; `corroborates` is too narrow (only fits experimental confirmation).

### Priority 3: tool-description anti-patterns

The dogfood revealed the proposed citation chain `137dba1d refines 6d2ef58e refines 8a533e15` collapsed citation-of-evidence into a refinement chain — `6d2ef58e refines 8a533e15` is wrong because the bootstrap is a charter, not a proposition with updated thinking. The mistake is natural (both feel chain-shaped) but nothing in the v1 description actively flagged the anti-pattern.

M5.1 extends `link_thoughts.relation`'s schemars description with a "Common mistakes to avoid" decision-tree block listing five anti-patterns:
- Don't use `refines` for citation or evidence — use `references` (or `supports` if it confirms a claim).
- Don't use `belongs_to` when the target is a peer or sibling — model the parent (e.g., the experiment, the session) explicitly as its own thought.
- Don't use `decided_by` without a clear decision-maker attribution — "the research suggests X" is `supports`, not `decided_by`.
- Don't use `replaces` for refinement — `replaces` means the older thought is no longer the current thinking; `refines` is when both stand.
- Don't use `references` when the newer thought confirms a claim made in the older one — use `supports`.

Cost: ~600 chars of additional prompt context in the `relation` schemars description. Tradeoff: longer prompt vs. fewer agent mistakes. For LLM-agent callers (the primary caller class), the cost is worth it.

### Schema impact

Migration 0008 is a pure CHECK constraint relax — drops the old `thought_links_relation_check` and adds a new one including `supports`. No data migration needed; existing rows are unaffected. Operators don't need to re-link anything; the new `supports` value just becomes available for new (or re-asserted) edges.

### Files changed

- `crates/engram-core/src/relation.rs` — added `Supports` variant to `RelationKind` enum; `as_str`, `FromStr`, `ALL`, error message text all updated. `ALL` is now `[RelationKind; 7]`.
- `migrations/0008_relation_supports.sql` — CHECK constraint extension.
- `crates/engram-mcp/src/server.rs` — `LinkThoughtsArgs.relation` description gains the `supports` entry + "Common mistakes" block; `UnlinkThoughtsArgs.relation` and `GetRelatedThoughtsArgs.relations` descriptions mention `supports`; `SERVER_INSTRUCTIONS` lists all seven relations + the `references`/`supports` distinction; regression test `server_instructions_advertise_*` pins `supports`.
- `README.md`, `docs/engram-design-v0.md` — vocabulary table / §6.6 / §9 / revision history updated.

### Dogfood plan (post-merge)

1. Re-link `74eb781c → 6d2ef58e` and `74eb781c → 63ad01e0` / `74eb781c → 047d0ce8` from `references` to `supports` (where they're evidential/aggregation cases). Verify `get_related_thoughts(74eb781c, relations: ["supports"])` returns the expected edges.
2. Observe agent behavior under the new vocab + anti-pattern doc: do new captures pick the right relation more often? (Particularly: do refinement-vs-citation distinctions land correctly?)
3. Exercise the dedup path: re-call an existing edge and verify the response is `is_new: false` with the same `link_id` (Priority 4 from the dogfood notes — spec was right but never exercised live).
4. Capture findings as thoughts; link them using the new vocabulary.

### Priority 2 — promoted to M5.2 (not in this iteration)

Heterogeneous targets (link-to-entity, link-to-person, link-to-URL) materialized as a real need on day one — Probe 2A and 2B are sibling variants under "Probe 2 experiment" which is not a thought. The tactical workaround (capture an experiment-thought, then `belongs_to`) is a capture-overhead tax that doesn't apply retroactively. The architectural fix is heterogeneous targets, promoted from M5.x to a near-term M5.2 iteration. M5.2 deserves its own plan-mode conversation — the schema change is non-trivial (polymorphic target column with no FK guarantee for non-thought rows) and the design space has real choices (polymorphic single table vs. separate `thought_external_links` table). Out of scope for M5.1.

### Lessons

- **Closed-vocab relation design needs the same anti-pattern documentation discipline as closed-enum kind classification.** The M4.1 v3 negative-example backfire taught us that listing forbidden phrases can reinforce them; the M5 v1 dogfood taught us that *not* listing anti-patterns lets natural-but-wrong applications slip through. The right shape is decision-tree rules ("don't use X for Y; use Z instead") in the tool description, not negative-example lists.
- **Day-one dogfood is the right cadence.** v1 had been live for hours; 17 edges across the operator's existing corpus was enough to falsify the closed-vocab choice. Vocabulary additions are cheap (one CHECK constraint relax); the cost of *not* iterating on the vocab is downstream brittleness in aggregation tooling.

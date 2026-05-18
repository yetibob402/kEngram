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

---

## M5.1.1 follow-up (2026-05-17)

Day-one M5.1 dogfood (immediately after the M5.1 commit) executed the suggested first moves cleanly — `supports` conversion landed, idempotency verified on an existing edge, filtered walk works as designed. Two findings + three smaller items returned for follow-up.

### Refined `supports` semantic

The original M5.1 description framed `supports` as "this thought confirms a claim made in another." Dogfood revealed this was under-specified — operators read it ambiguously. The clarification: `supports` is **active confirmation, not passive citation**. The FROM thought must itself make a claim that confirms TO's claim. Summary/aggregation edges and passive prose mentions stay as `references`.

Direction convention is **FROM=confirmer, TO=claim-maker** — consistent with the "this thought does X to another" shape of the other relations (the FROM thought is the actor; the TO thought is the target).

The agent decision rule that disambiguates cleanly: ask *"does the FROM thought itself make a confirming claim?"* — not *"is this thought evidence-shaped?"* The evidence-shaped framing pulls operators toward false positives on summary/aggregation cases.

M5.1.1 tightens the `link_thoughts.relation` tool description with this clarification plus an explicit anti-pattern: "DO NOT use `supports` for passive citation or summarization. ... Summary/aggregation edges (FROM summarizes data points TO) are `references`, not `supports`."

The dogfood scope: M5.1's suggested "convert the summary cites too" was wrong. Only one edge (`74eb781c supports 6d2ef58e`) was a true `supports` conversion; the summary cites (`74eb781c → 63ad01e0`, `74eb781c → 047d0ce8`) stay as `references`.

### Silent-edge-removal investigation

Day-one dogfood reported a `74eb781c references 6d2ef58e` edge (link_id `9b1c0d78-...`, created in-session) as missing after the M5.1 restart, with the hypothesis that migration 0008 may have pattern-matched its note text and removed it.

Investigation rules this out. Migration 0008 is a four-line `ALTER TABLE thought_links DROP CONSTRAINT ... ADD CONSTRAINT ...` — a pure CHECK relax with no DELETE / UPDATE clauses. Verified by inspecting the file. Direct DB inspection confirms 6 surviving edges involving `74eb781c` (3 outbound, 3 inbound), with `74eb781c references 6d2ef58e` absent. The new `74eb781c supports 6d2ef58e` (link_id `b5bfa573-...`) carries note "converted from references — ...", indicating an intentional unlink-then-link conversion procedure.

Most parsimonious diagnosis: the conversion was a two-step (`unlink_thoughts` then `link_thoughts`) procedure where the unlink step was performed but not retained in operator memory. This isn't a bug; it's a consequence of conversion-as-unlink+link being the only path under the current MCP surface.

Two operator-visible improvements would prevent this class of confusion in M5.x:

1. **Migration audit signal.** When `engram migrate` applies an edge-affecting migration (delete / update of `thought_links` rows), record the row count delta into a `migration_audit` table or emit a structured tracing event with the migration version + delta. Operators can then distinguish "the migration touched my edges" from "the migration was schema-only." Migration 0008 has zero edge-affecting clauses; future migrations should self-classify so the operator-visible signal is unambiguous.

2. **`unlink_thoughts` discriminator.** Distinguish "never-existed" from "previously-removed" in the response. Cheapest implementation: a soft-delete model on `thought_links` (`deleted_at TIMESTAMPTZ`). Edges resolve in queries by filtering `WHERE deleted_at IS NULL`; `unlink_thoughts` returns `existed: false, previously_removed: true | null`. More expensive: a `thought_links_history` table tracking inserts + deletes with timestamps and source. Either gives operators an "I unlinked this myself" vs. "something else happened" diagnostic bit.

Both promoted to the **M5.x backlog** under "audit & operator diagnostics." Not load-bearing for M5.2's heterogeneous-targets work; can land independently.

### Doc-bug fixes

Three places still said "one of six closed-vocabulary relations" while enumerating seven (live `link_thoughts` tool description; README MCP surface table; design-v0 §8 MCP surface table). Plus the §6.6 rationale paragraph said "six relations the operator can predict." All fixed in M5.1.1.

### Sqlx migrations table state (operator action item)

`_sqlx_migrations` records only versions 1-6. Migrations 0007 and 0008 were applied directly via `docker exec psql` during M5 / M5.1 development sessions, so the migrations table doesn't reflect them. The schema state is correct (verified via `\d thought_links`) but the migrations table is stale.

The fix is operator-side: run `engram migrate` to record the applied migrations. sqlx::migrate should recognize the existing schema state and update the table accordingly (or, if it complains about applied-but-not-recorded migrations, the resolution is a manual `INSERT INTO _sqlx_migrations (version, description, success, checksum, execution_time) VALUES ...` for the two missing entries).

Tracked as an M5.x cleanup item; not urgent.

### Notable edge structure observed

The dogfood ran the refinement chain in the wild: `82256109` (silent-edge-removal finding) and `8751f1aa` (supports-semantic finding) both `references` the M5 milestone thought `74eb781c`; `8751f1aa` then `refines` `618f5a6b` (the original references-ambiguity finding from earlier in the dogfood). The dialectical chain is first-class: references-ambiguity → `supports` lands → refined finding on `supports` semantics. Walkable via `get_related_thoughts(618f5a6b, relations: ["refines"])` from the inbound direction.

This is the citation-chain pattern the M5 milestone was motivated by, now visible in the live graph and queryable by relation type — the M4.1 dogfood's implicit-in-prose chain made explicit at M5+.

---

# M5.2 — Heterogeneous targets + audit + CLI scope-prefix

**Status:** shipped 2026-05-17 as the M5 close-out iteration. Bundles three M5.x backlog items that emerged from M5 / M5.1 / M5.1.1 dogfood.

## Scope

1. **Heterogeneous link targets.** The `from` side of a `thought_links` row stays anchored on a thought; the `to` side gains three new shapes — entity (free-text), person (free-text), URL (`http(s)://...`). Discriminator column `to_kind` + three per-kind columns + a generated `to_value` column anchoring the unique-edge constraint.

2. **Soft-delete + three-way unlink status.** `thought_links.deleted_at TIMESTAMPTZ NULL`. `delete_link` becomes UPDATE-with-RETURNING. `unlink_thoughts` returns a three-way status enum: `deleted_now` / `already_deleted` / `never_existed`. The partial unique index ignores soft-deleted rows, so re-creating a previously-removed edge inserts a fresh live row.

3. **Migration audit table + CLI subcommand.** `migration_audit (id, migration, ran_at, rows_touched, notes)` populated by per-migration `INSERT` statements (process discipline; migration 0010 seeds the table with rows for 0009 + 0010). `engram audit migrations [--since X] [--limit N]` prints the log most-recent-first.

4. **CLI scope-prefix flags.** `engram tag` and `engram embed-backfill` gain `--scope-prefix` (mutually exclusive with `--scope` via clap `conflicts_with`), threaded through to the corresponding storage helpers.

5. **MCP surface deltas.** `link_thoughts` / `unlink_thoughts` args gain `to_entity` / `to_person` / `to_url` alongside `to_thought_id` (mutex; exactly-one validated server-side). `get_related_thoughts` args gain `target_kinds`; response gains `to_kind` / `to_value`, with thought-target hits retaining the existing scope / content_preview / retracted fields and non-thought hits leaving them null. SERVER_INSTRUCTIONS updated with the polymorphic-target surface; regression test extended to pin the four target-kind fields and the three-way unlink status enum.

## Schema deltas

- **Migration 0009** — heterogeneous targets. Adds `to_kind` (CHECK enum), `to_entity` / `to_person` / `to_url` (nullable), `to_value` (generated, COALESCE). Replaces `thought_links_unique_edge` (table-level UNIQUE) and `thought_links_no_self_reference`. Adds `thought_links_target_valid` (exactly-one-per-kind CHECK) and `thought_links_url_format` (http(s):// CHECK). New index `thought_links_from_kind_idx`. Non-destructive; existing rows default to `to_kind = 'thought'`.

- **Migration 0010** — soft-delete + audit. Adds `thought_links.deleted_at`. Drops the table-level `thought_links_unique_edge` constraint added in 0009 and recreates it as a partial unique index (`WHERE deleted_at IS NULL`). Adds `thought_links_deleted_at_idx` partial index. Creates `migration_audit` table with `ran_at` index. Seeds audit rows for 0009 and 0010.

## Decision log

- **Discriminator-column over separate-tables (heterogeneous targets).** Postgres handles polymorphic targets fine via a discriminator + per-kind columns; the alternative (`thought_to_entity_links`, `thought_to_url_links`, etc.) fans out queries across tables and complicates traversal. The `to_value` generated column anchors a single unique-edge constraint covering all kinds — cleaner than per-kind unique indexes.

- **Partial unique index for soft-delete idempotency.** `CREATE UNIQUE INDEX ... WHERE deleted_at IS NULL` lets a previously-soft-deleted edge sit inert while a fresh insert with the same triple succeeds. ON CONFLICT can target a partial unique index by `ON CONFLICT (cols) WHERE predicate` — works as expected; verified in `insert_after_soft_delete_creates_fresh_live_row` integration test.

- **Free-text entities and persons (no first-class table).** Entities and persons are stored as strings on `thought_links.to_entity` / `to_person` rather than as FKs to dedicated tables. Engram has no entity-resolution layer in v0; first-class entity tables would precede tagger-extracted relations (still M5.x), and over-engineering them now would block M5.2 ship. Free-text is consistent with how the tagger already represents entities (in the `tags.entities[]` array — free-form strings).

- **Three-way unlink status, not boolean discriminator.** `{ existed: bool, previously_removed: bool }` would work but composes awkwardly with future states (e.g., a `restored` status if `restore_link` ever ships). A tagged enum (`status`) leaves room for that without breaking the response shape further.

- **`migration_audit` as a schema artifact + CLI surface, not a code helper.** Every future row-touching migration ends with an `INSERT INTO migration_audit` statement by convention (DEVELOPMENT.md note). A code-side `migration_audit_record!` helper would be over-engineered for one-line-per-migration usage. The CLI subcommand prints the log; no MCP-side surface for v1.

- **`engram audit migrations` resource-as-positional, not `engram audit-migrations`.** Future audit resources (`engram audit links`, `engram audit thoughts`) compose under `engram audit <resource>`; a flat subcommand would have to add new top-level subcommands per resource.

- **Schema breaking change accepted on `unlink_thoughts` response.** The shape moves from `{ existed: bool }` to `{ status: enum }`. Acceptable in single-operator dogfood; called out here so a future operator-facing changelog can pick it up.

## Tests added

Eleven new integration tests in engram-storage (heterogeneous-target writes, URL format CHECK, unique-edge across kinds, soft-delete + lookup_link_status three-way + fresh-row-after-soft-delete, target_kinds filter, migration_audit presence + ordering), seven new in engram-mcp (entity / URL / non-http rejection writes via the orchestrator, empty-name rejection, three-way unlink status, link-after-unlink, heterogeneous outbound retrieval, target_kinds filter), one CLI regression-shape extension. 317 total tests passing post-M5.2.

## Out of scope (deferred)

- Indexed search of `to_entity` / `to_person` / `to_url` columns. No GIN/B-tree on these in v1.
- Entity/person resolution (would precede tagger-extracted relations).
- Reverse traversal from non-thought targets ("what thoughts link to this URL?").
- A `restore_link` tool. Operator can re-link via `link_thoughts` (fresh row) or UPDATE via psql.
- Hard-purge / retention policy for soft-deleted edges.
- Backfilling `migration_audit` for migrations 0001-0008.
- `engram audit links`, `engram audit thoughts`, etc. — only `migrations` ships in M5.2.

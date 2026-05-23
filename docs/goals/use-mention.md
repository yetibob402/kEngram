# Goal: close the use-mention gap in the tagger prompt

## Goal

Close the use-mention failure in the engram tagger prompt so that
meta-content thoughts no longer pollute their own `people` and
`action_items` tags by extracting names/phrases mentioned as linguistic
examples, quoted directives, or items in demonstrative lists.

## Success criterion

```sh
cargo run --example tagger_eval -p engram-extract -- \
  crates/engram-extract/tests/fixtures/use_mention.json
```

Pass condition (parse from the stdout summary line `N/12 passed` or the
`--json` `passed` field):

- **6 of 6 fixtures in `category=control` PASS** (no false negatives —
  real name references must still extract correctly).
- **At least 5 of 6 fixtures in `category=use_mention` PASS** (one
  LLM-noise slip is acceptable; the goal is reducing pollution, not
  perfection).
- **Net total: ≥ 11/12 PASS.**

The harness exits 0 if all 12 pass, non-zero otherwise. The session
should re-run the harness after each prompt change and track which
fixtures move from FAIL to PASS (or vice versa) per iteration.

## Iteration loop

**File to modify:** `crates/engram-extract/src/openai_compatible.rs`,
the `BUNDLED_TAGGER_PROMPT` const (search for `pub const
BUNDLED_TAGGER_PROMPT`).

**Action space, in order of preference:**

1. **Add a `# Examples` section** between `# Rules` and `# Before you
   emit — final pass`. Each example is a worked input → output pair
   demonstrating the discipline. The use-mention shape that the corpus
   has documented evidence for:

   - Parenthetical example: `'The Bob-as-verb pattern (e.g., "Bob the
     index rebuild") needs investigation.'` → empty `people`, empty
     `action_items`.
   - Demonstrative list: `'Common verb-as-name first names include:
     Bob, Mark, Rob.'` → empty `people`.
   - Quoted directive: `'The probe should use prompts like "evaluate
     options" and "pick one".'` → empty `action_items`.

   Frame as `Thought: ... → Output: ...` pairs, positive guidance only.

2. **If examples alone don't move the score**, strengthen the `# Rules`
   section with a positively-framed use-mention rule. Sample wording:
   *"Text inside quotation marks, parentheticals introduced by 'e.g.'
   or 'such as,' and bullet/comma-list items presented as candidates
   for selection are MENTIONED, not USED. Do not extract their contents
   as people, entities, or action_items."*

3. **If still stuck**, revisit the per-field instructions (`people`,
   `entities`, `action_items`) and add field-local guidance.

**Per-iteration measurement:**

- Re-run the harness command above against the local Ollama:
  `OLLAMA_ENDPOINT=http://localhost:11434/v1 TAGGER_MODEL=gemma3:12b`.
- Iteration runs against the production model locally — this is the
  same model deployed on rons-imac, just running on the iterator's
  workstation. No proxy / smaller-model substitution: prompt
  improvements measured here transfer directly to production.
- Score the result (count of passes by category).
- Keep changes that improved the score; revert changes that regressed.
- Track running best across iterations.

**On LLM variance:** at temperature 0.2 the score may drift ±1-2 between
runs on the same prompt. If a change appears to help by exactly 1 point,
re-run 2-3 times before treating it as a real improvement vs. noise.

**Ship-time deployment (separate from iteration):** once the criterion
is met, the version-bump + commit + push + retag-on-rons-imac is the
ship step. That's outside the iteration loop and gets handled per the
existing v8/v9/v11/v12 commit-and-push pattern.

## Constraints

- **Do not change `BUNDLED_TAGGER_VERSION`.** The version bump (12 →
  13) is the *ship* step and only happens after the criterion is met.
  Iterating with the same version stamped means the harness keeps
  measuring against the same baseline.
- **Do not change `Tags` schema** (`crates/engram-core/src/tags.rs`)
  or the JSON shape the tagger emits.
- **Do not introduce new crate dependencies.** Prompt-only changes.
- **Do not break `cargo test -p engram-extract`** (currently 30/30 pass).
- **Do not change existing fixture files.** If a fixture turns out to
  be wrongly specified, stop and report rather than edit it.
- **Do not commit / push.** Stop at the winning prompt diff and let the
  operator review before shipping.

## Stop conditions

- **Success:** harness reports ≥ 11/12 PASS with the category
  breakdown required above. Stop, present the winning prompt diff and
  the per-iteration score trajectory.
- **Bounded failure:** 8 iterations without net improvement over the
  baseline. Stop, present the best attempt, the failing fixtures, and
  hypotheses for why they resist (likely candidates: model-capacity
  ceiling, fixture-specification issue, or prompt action space
  exhausted).
- **External failure:** harness fails to connect to Ollama, tagger
  returns malformed JSON repeatedly, or `cargo build` breaks for
  reasons unrelated to the prompt content. Stop, report.

## Reporting

When stopping, produce:

1. **Diff** of `BUNDLED_TAGGER_PROMPT` against the v12 baseline.
2. **Per-iteration score history** (`iteration N: 9/12 (6 control, 3
   use_mention)` style).
3. **Per-fixture final state** highlighting any fixtures that flipped
   between PASS and FAIL across iterations.
4. **Recommendation:** ship as v13 (prompt bump + retag), or escalate
   to a structural fix (e.g., pre-process layer that strips quoted
   spans before tagging), or defer to the post-3090 larger model.

## Out of scope

- Other gaps from the v12 evaluation: Bob-as-verb misclassification,
  product-name token extraction (probe D's "Claude" from "Claude
  Desktop"), role descriptors in `people`, GitHub-handle
  misclassification. Each gets its own goal artifact with its own
  fixture set when prioritized.
- Structural pre-process layers (e.g., a quoted-span stripper). If the
  prompt-only action space is insufficient, surface this as a
  recommendation in the report — don't bolt it on in this goal.
- Model swap. Same — surface as a recommendation, don't change the
  configured model.

## Risk + rollback

- Prompt changes are scoped to one Rust string constant. `git restore
  crates/engram-extract/src/openai_compatible.rs` is a clean rollback.
- If a winning prompt ships as v13 and turns out to over-suppress
  legitimate emissions in the real corpus, the next operator retag
  surfaces it; revert and re-iterate with stronger control-fixture
  coverage.

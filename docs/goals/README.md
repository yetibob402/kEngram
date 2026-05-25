# Goal artifacts

A *goal artifact* is a self-contained markdown brief that a future Claude
session reads to autonomously work on a specific gap, with an
objectively-measurable success criterion and bounded stop conditions.

Goal artifacts exist so prompt iteration, fixture-driven prompt tuning,
or any bounded improvement work can be delegated cleanly:

1. Build the measurement tool first (an eval harness, a test script, a
   benchmark) and a fixture set that defines "good."
2. Write a goal artifact that points at the measurement, declares
   success in terms of it, and specifies what to modify, what's
   off-limits, and when to stop.
3. In a future session, hand the artifact path to Claude (e.g.,
   `/goal docs/goals/use-mention.md`). Claude iterates without further
   human direction until success or a stop condition.

The point is to keep the human in the **goal-setting** loop, not the
**iteration** loop. Iteration loops are the failure-rich, time-consuming
part; goal-setting is where human judgment lives.

## Required sections

A goal artifact must have these six sections, in order:

### 1. Goal

One sentence. What gap is being closed? Frame it in terms of an
observable behavior, not an implementation. ("Close the use-mention
failure" not "add a few-shot example.")

### 2. Success criterion

The exact command (or commands) to run, and the exit / output condition
that means "done." Must be:

- **Objective.** No human judgment in the pass/fail loop. If a number is
  involved, name the number and the comparison.
- **Reproducible.** Same command, same fixture, same prompt → same
  result modulo LLM variance documented separately.
- **Cheap.** Should run in seconds-to-minutes, not the multi-step retag
  cycles we're trying to escape.

Example:
> Running `cargo run --example tagger_eval -- crates/kengram-extract/tests/fixtures/use_mention.json` reports 6/6 PASS in `category=control` and ≥5/6 PASS in `category=use_mention` (net 11/12 or better).

### 3. Iteration loop

What to modify, in what order, and how to measure progress between
attempts. Should be specific enough that the iterating session doesn't
guess at the action space. Include:

- **What file(s) to modify.** Name them explicitly.
- **What aspects to vary.** "The wording of rule N," "the order of
  examples," "the section the rule lives in" — whatever the action
  space is.
- **How to score.** Usually "re-run the success-criterion command."
- **How to decide between attempts.** "Keep if score improved, revert
  if regressed. Track running best."

### 4. Constraints

What's off-limits. Common entries:

- No new crate dependencies.
- No schema changes (the `Tags` struct, the JSON shape, etc.).
- No changes to existing tests (other than version bumps tied to the
  change being shipped).
- Don't ship until the criterion is met (no version bump, no commit to
  `main`).
- Don't iterate beyond N attempts.

The constraints section is where you encode the operator's risk
appetite. Without it, the iterating session might over-reach.

### 5. Stop conditions

When to stop iterating and report. Three categories:

- **Success.** Criterion met. Stop. Present the winning change and
  metrics.
- **Bounded failure.** N iterations without net improvement (typically
  6-10 depending on the action space). Stop. Present best attempt and
  what's stubbornly resisting.
- **External failure.** Harness/tool errors that aren't caused by the
  changes being made. Stop. Report so the operator can fix the
  infrastructure.

### 6. Reporting

What to summarize when stopping. At minimum:

- The diff of the change(s) attempted (or the running best).
- The score trajectory across iterations (so the operator can see what
  helped and what didn't).
- A short analysis of which fixtures still fail (if any) and a
  recommendation on next-action.

## Optional sections

- **Background / motivation** — short context if the goal isn't
  self-evident from the gap name.
- **Out of scope** — explicit non-goals to keep the iterating session
  focused.
- **Risk + rollback** — what happens if the change ships and turns out
  to be wrong.

## Conventions

- Goal artifacts live in `docs/goals/<gap-name>.md`.
- One artifact per gap. If a gap has multiple sub-gaps, prefer separate
  artifacts over a sprawling single brief.
- Reference the fixture file(s) and the harness command verbatim — no
  paraphrasing. The artifact must be runnable as written.
- Keep the artifact under ~300 lines. Beyond that, split or move
  background into a sibling design doc.

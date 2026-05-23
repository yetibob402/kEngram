# Tagger improvement ideas

Notes from the M5 dogfood round (2026-05-22) where we chased a verb-as-name
failure mode by swapping tagger models. Captures what was tried, what's still
open, and where to take it after the next hardware upgrade.

## Context

Original failure observed in the `rjf.tech` scope: a thought starting with
"Calibrate daily Rust challenges..." had **Calibrate** extracted into the
`people` field. Tagger was `ollama/qwen3-coder:30b` at prompt v7.

Hypothesis: the code-specialized fine-tune was weakening general NER. Swapped
to general-purpose models and ran a probe set in the `engram.tagger-test`
scope covering imperative verbs as sentence starts, words that are both
common verbs and real names (Mark / Will / Bob), pure imperatives with no
people, mixed people + entities, pure people, and field-separation pressure
tests.

Two general-purpose models exercised: `ollama/qwen3:30b` (probes A, B) and
`ollama/gemma3:12b` (probes C, D, E, F, G, H).

## Resolved

### Verb-as-name in `people` — fixed by leaving the coder model

Across 8 probes, neither qwen3:30b nor gemma3:12b ever put an imperative verb
(Calibrate, Refactor, Document, Tune, Profile, Configure, Build, Audit,
Review, Verify, Patch, Confirm) into `people`. The original failure was
specifically a code-specialized fine-tune artifact, not a small-model ceiling.

Concrete change: `[tagger].model_name` and `model_id` swapped to `gemma3:12b`
in `~/.config/engram/engram.toml`. Prompt version unchanged.

### Topic overreach — fixed by v9 (after v8 only rotated the target)

Root cause confirmed in source review: the bundled tagger prompt's `topics`
example list (`Examples: ...`) was the priming mechanism. First-item
priming in few-shot example lists disproportionately shapes what the LLM
treats as canonical vocabulary.

**v8 attempt (partial fix):** removed `"rust"` from the topics examples and
the kind=observation exemplar. Topics examples re-led with `memory-systems`
then `team-management`, `databases`, `information-retrieval`. Observation
exemplar swapped the Rust-vs-C claim for a Postgres autovacuum fact.
Post-v8 corpus retag confirmed `"rust"` overreach was gone (only 2 of 58
thoughts kept `"rust"` topic, both genuinely about Rust). **But:**
`"databases"` over-emission appeared in 13 of 18 thoughts that got it as
a topic — engram API design, branding, capture-discipline observations,
etc. v8 had simply rotated the priming target from `"rust"` to `"databases"`.

**v9 fix:** dropped the standalone `Examples: ...` clause from the topics
field instruction entirely. The topics paragraph now relies on the prose
example at the end ("a thought naming engram and pgvector might have
topics [memory-systems, databases]") plus the concept-mapping intent
statement to teach the field — no free-floating list of canonical items
for the model to over-anchor to. `BUNDLED_TAGGER_VERSION` bumped 8 → 9.
Lesson: the structural issue was the example list itself, not its
contents. Entities example list still carries one (it's needed for the
surface-only rule's clarity) but topics now has none.

### Topic overreach via scope_vocab feedback loop — partial fix at v11; ceiling-bound by model capacity

**Diagnosis.** Post-v9 corpus-wide retag revealed that v9 only addressed the
*in-prompt* priming mechanism. A second mechanism persisted: the worker
drainer fetches the top-N established topic/entity terms from each
thought's scope (`scope_vocab_limit = Some(50)`) and injects them into the
prompt as a controlled-vocabulary hint. The intent (v4 design) was
"tie-break to canonical forms when a thought is genuinely about an
established subject." In practice the LLM treats the hint list as a menu
of canonical options and fits new thoughts to whichever ones plausibly
apply.

This creates a positive feedback loop: noise in early tags (e.g. v7's
`databases` overreach) becomes "established" in the vocab; subsequent
retags see those terms as canonical and reapply them; the noise
self-reinforces indefinitely.

**Empirical evidence (2026-05-22 / 2026-05-23 retag cycles):**

| Configuration | `"databases"` topic count | Overreach (not really about DBs) |
|---|---:|---:|
| v8 + scope_vocab on (size=50) | 18 | 13 |
| v9 + scope_vocab on (size=50) | 16 | 11 |
| v9 + scope_vocab off          | 10 | 3 |

Disabling vocab dropped overreach by ~77%. But it also killed the
legitimate benefit: `"rust"` topic disappeared from probe A (which is
genuinely about Rust) and from `307b0039` (Ron's Rust-preference note),
because the LLM no longer had a canonical-form hint to reach for.

**Why this isn't fixable with prompt edits alone.** Tagger history (v3→v4
backfire, v6→v7 repeat) shows that small LLMs don't reliably honor
"qualifier" instructions like "use these only when the thought is centrally
about them." The mechanism is sound in principle but the LLM's execution
is unreliable. Mechanism-level changes are needed.

**Approaches considered:**

- **A. Post-process topic normalization (chosen).** LLM emits topics fresh
  from prose (no topic vocab in prompt). After response, normalize each
  emitted topic against the scope vocab using embedding similarity — close
  matches get substituted with canonical forms; novel emissions pass
  through. Separates emission (LLM's job) from normalization (post-process).
- B. Embedding-similarity-filtered vocab in prompt. Per-thought, embed the
  content and pass only vocab terms semantically close to the thought as
  hints. More invasive (LLM still in the loop); more code surface.
- C. Reduce `scope_vocab_size` from 50 to 10. Cheap empirical test; loses
  long-tail canonical-form convergence; doesn't address the mechanism.
  Acceptable only if `tag_filter` usage is light.

**Decision: implement Option A** with one refinement — keep *entities*
in the prompt vocab. Entities are surface-bound (the prose must contain
the name) so the overreach failure mode doesn't apply to them, and the
casing/spelling consistency benefit of the in-prompt entity hint is
worth preserving. Only *topics* get the Option A treatment.

**Concrete change shape:**

1. `render_vocab_section` in `engram-extract` stops rendering the topics
   sub-section. The entities sub-section keeps its current behavior.
2. New post-processing step in the tagger flow: after the LLM emits topics,
   embed each emitted topic + each scope-vocab topic; for each emitted
   topic, find the best embedding-similarity match in the vocab. If above
   threshold (initial guess: cosine ≥ 0.85), substitute the canonical form.
3. `BUNDLED_TAGGER_VERSION` will bump (10 → 11) to mark the behavioral
   change in provenance.
4. The `scope_vocab_enabled` flag remains the master switch: when false,
   no vocab feed AND no normalization. When true, entities-in-prompt +
   topics-normalized-post.

**The general design lesson (worth recording):** any LLM-driven extraction
system that uses its own corpus output as feedback into subsequent
extractions is at risk of self-reinforcing overreach. Separating emission
from normalization is the structural fix. Not gemma3-specific; applies to
any controlled-vocabulary-with-LLM pipeline.

**v11 empirical outcome (2026-05-23 retag, gemma3:12b).** The architecture
landed cleanly but the empirical gains over the v9 vocab-off baseline are
smaller than the diagnosis predicted. Across the same 58-thought corpus:

| Run | `"databases"` topic count | Overreach (not really about DBs) | `"rust"` on Rust-about thoughts |
|---|---:|---:|---:|
| v8 + vocab on (size=50) | 18 | 13 | 1 of 2 (correct) |
| v9 + vocab on (size=50) | 16 | 11 | 1 of 2 (correct) |
| v9 + vocab off          | 10 | 3 | 0 of 2 (lost) |
| **v11 (vocab off in prompt + normalize post)** | **14** | **~8** | **0 of 2 (still lost)** |

Worst-case overreach is gone — the Engram-branding thought
(`1bfcc158`) is now `[branding, ai-agents, memory-systems]` and the
AI-Agile manifesto (`62495576`) is `[ai, agile-methodology,
software-development]`. Both were clear `databases`/`rust` overreach
before. Legitimate canonical convergence works where the corpus has
established a canonical form (`search-engines`, `ai-agents`,
`memory-systems` appearing consistently across multiple thoughts).

But the remaining `databases` cases (~8) are mostly thoughts about
engram's storage/graph layer where the LLM emits `databases` or close
variants on its own initiative, regardless of prompt. The normalizer
dutifully converges variants (`database` → `databases`) which can amplify
the count slightly versus vocab-off. And the `rust` topic on actual
Rust-about content is still missing — without vocab in the prompt,
gemma3:12b doesn't emit `rust` spontaneously on probe A or the
Ron-prefers-Rust thought, and the normalizer can't introduce a term the
LLM didn't emit.

**The real takeaway from this exploration arc.** gemma3:12b's topic
emission is the dominant variance source, not vocab feedback. v8/v9
prompt edits, scope_vocab toggle, and v11 normalization each gave us
modest wins on different axes, but none broke the model-capacity
ceiling. The architectural cleanups (separating emission from
normalization, entities-only-in-prompt) are sound and worth keeping
regardless — they compound with model improvements rather than
substituting for them. **Further tagger iteration is parked until the
3090 build lands a larger model.** Re-run the probe set then.

### How v11 compounds with a larger model (post-3090)

The v11 architecture is model-agnostic. When a larger tagger model
(Qwen 2.5 32B Instruct or similar — see "Future model upgrades" below)
runs through the same v11 pipeline, expected additional benefits:

- **Less noise in emissions.** A 32B model follows the topic
  instruction more reliably — fewer spontaneous `databases`/`rust`
  emissions on tangentially-related content. Fewer false positives
  entering the corpus → fewer false canonical forms for the
  normalizer to reinforce.
- **Better legitimate emission.** A 32B model is more likely to
  emit `rust` on probe A and the Ron-prefers-Rust thought when
  appropriate, restoring the canonical-form benefit we lost at v9.
- **Tighter normalizer convergence.** With richer per-scope vocabs
  (from accurate spontaneous emissions), the normalizer has more
  canonical targets to converge against. The current sparse vocab
  in some scopes (`engram.tagger-test`, `rjf.tech`) means
  normalization is a no-op there.
- **Vocab-in-prompt may become safe again.** A future v12 could
  reasonably re-enable in-prompt topic vocab if the larger model
  treats it as a tie-breaker hint rather than a menu. That's a
  tunable to revisit empirically with the new model — keep both
  mechanisms (prompt vocab + post-process normalize) available as
  config knobs and measure.

Net: the work done at v8-v11 was the right architecture; the larger
model is the lever that makes it deliver on the promise.

**v12 follow-up (2026-05-23):** two small additions on top of v11
that don't depend on a model swap.

1. **Positive syntactic-disambiguation rule in the `people` field
   instruction.** Targets the "Bob the worker batch limit" failure
   mode where probes B and E mis-route a sentence-start imperative
   verb into `people`. Framed positively ("when X, do Y") to dodge
   the v3→v4/v6→v7 backfire pattern; prior is ~30% probability of
   landing on gemma3:12b. Non-load-bearing — measured but not
   gated. If probes B/E still mis-tag "Bob" at v12, the next
   escape hatch is a deterministic syntactic check, deferred until
   after the 3090 model swap.
2. **Post-process disjointness validator.** See next subsection.

### Field separation (people-in-entities) — fixed by v12 disjointness validator

**Symptom.** Probe E at v11 emitted `people: ["Bob", "Sarah"]` and
`entities: ["Sarah"]` — same string in both arrays. Earlier samples
showed the same pattern intermittently on other thoughts. Small LLMs
occasionally route the same name into both fields; no prompt
instruction reliably prevents it.

**Why it's not a prompt fix.** This is a structural invariant of valid
`Tags` (a string cannot legitimately occupy both fields), so the right
place to enforce it is a deterministic post-process step, not the LLM.
Same architectural rationale as the topic normalizer at v11: separate
the LLM's emission from the structural cleanup.

**Fix (v12).** New module `crates/engram-mcp/src/validate.rs` exposes
`enforce_people_entities_disjoint(&mut tags)`. Behavior:

- Build a case-insensitive set of `tags.people`.
- Retain only those `tags.entities` whose lowercased form is NOT in
  that set. Person wins on tie (the `people` classification is more
  semantically constrained).
- Pure function over `&mut Tags`; mutates entities in place.

Called unconditionally in `process_tag_job` after the topic normalize
step, before `update_thought_tags` persists. Eight unit tests cover
disjoint pass-through, exact and case-insensitive duplicate stripping,
empty-array no-ops, order preservation, and people-array
untouchability.

Belt-and-suspenders pairing: the v12 prompt also has a redundant rule
in the `# Rules` section ("a name belongs in EITHER `people` OR
`entities`, never both") so the LLM has a fair chance of getting it
right on first emission; the validator backstops when it doesn't.

### Use-mention pollution — fixed by v13 prompt edit (gemma3:12b)

**Diagnosis.** Post-v12 evaluation report identified use-mention as the
#1 G1/G3 gap: meta-content thoughts (engram.m3.dogfood scope, v11/v12
recommendation thoughts in engram.tagger-test) were extracting tokens
that appeared in quoted strings, parenthetical examples, demonstrative
lists, and brainstorm enumerations as if they were object-level content.
Probe data showed Bob / Mark / Rob / Frank / Pat all extracted as
`people` on `de1411c1`; `"evaluate options A through F"` extracted as
`action_item` on `74b46543`; etc.

**Fix (v13).** Iterated `BUNDLED_TAGGER_PROMPT` locally against
gemma3:12b using the new `examples/tagger_eval.rs` harness +
12-fixture set at `crates/engram-extract/tests/fixtures/use_mention.json`.
Result: 11/12 stable PASS (6/6 control + 5/6 use_mention) over two
runs. Prompt additions (all in `BUNDLED_TAGGER_PROMPT`):

- `# First discipline: USE vs MENTION` section at the top of the
  prompt — defines USED (extract) vs MENTIONED (don't), calls out
  meta-discussion as the case where every name is typically a mention.
- One use-mention bullet appended to `# Rules` restating the
  discipline at emission time.
- `# Examples` section between `# Rules` and `# Before you emit —
  final pass` with 6 worked input→output pairs covering parenthetical
  mentions, demonstrative lists, real-references contrast, quoted
  directives, meta-discussion of other thoughts, and meta-discussion
  of tagger behavior.

**Residual.** `meta-discussion-of-contamination` fixture still fails:
when prose says "Sarah was emitted in both people and entities" the
model puts Sarah in `entities`. The v12 disjointness validator can't
help because Sarah isn't in `people` (LLM never emits her there).
Closing this would require either a structural pre-process layer that
strips quoted/cited spans before tagging, or a larger model
(post-3090). Documented as a known residual rather than blocking
ship.

**Empirical-method note.** The previous /goal iteration against
gemma3:4b mis-attributed two control failures to "fixture-spec issues"
(HNSW recall on `maria-david-pgvector-control`, Postgres
canonicalization on `postgres-no-people-control`). Re-baseline against
gemma3:12b showed both PASS at v12 baseline with zero changes — they
were model-capacity issues on 4b, not fixture issues. Lesson for
future goal artifacts: iterate against the production model when
feasible; small-model headroom-for-discrimination is real but
ambiguity-of-failure-class is also real.

## Open issues

### 1. "Bob-as-verb" ambiguity (residual)

**Symptom.** Both qwen3:30b (probe B) and gemma3:12b (probe E) read "Bob the
index rebuild for off-hours" with "Bob" as a person name. Other ambiguous
verb/names (Mark, Will) were filtered correctly — only "Bob" stuck.

**Why it's hard.** Bob is a real common name and "to bob something" is
unusual usage. Even a human reader pauses. This is syntactic ambiguity at
the small-model ceiling, not a clean bug.

**Options, weakest to strongest:**

- **Accept.** Tags are advisory — search doesn't gate on `people`. Cost:
  bogus Bobs appear in `people`-filtered search. Effort: zero.
- **Post-hoc denylist filter.** After the LLM returns tags, drop any
  `people` entry whose lowercase form is in a small denylist of clearly
  verb-or-noun words (`calibrate`, `ensure`, `review`, `configure`,
  `build`, ...). Easy in `engram-extract`'s tagger output adapter. Risk:
  false positives for legit names that share with verbs. Mitigation:
  restrict denylist to *unambiguous* verbs (excludes Mark, Will, Bob,
  Rose, etc.).
- **Constrained-vocabulary `people` mode.** Mirror the existing
  `scope_vocab_limit` (already used for topics): per-scope, pass the top-N
  established names from prior thoughts as a hint, instructing the tagger
  to prefer those names. Genuine new names still get through. Higher
  effort; closer to the design doc's spirit.
- **Larger model.** A 32B/70B model will be more consistent on syntactic
  role assignment — but won't be perfect.

**Recommendation:** Accept for now. Revisit when the 3090 build is up and a
larger model is running, then measure whether it's still worth the post-hoc
filter or the constrained-vocab work.

### 2. `action_items` coverage is uneven

**Symptom.** gemma3:12b on probe E captured 2 of 3 imperatives, missing the
"Bob the worker batch limit" item — plausibly because "Bob" got routed into
`people` and the imperative got eaten with it. qwen3:30b on probe B captured
all 3 even with the same Bob-in-people miss.

**Status.** Bound to the Bob-as-verb issue. Likely resolves itself once that
underlying ambiguity is addressed.

## Future model upgrades (post-3090 build)

Once a 3090 (24GB VRAM) is available for the tagger:

| Model | Quant | VRAM | Notes |
|---|---|---|---|
| **Qwen 2.5 32B Instruct** | Q4_K_M | ~18-20GB | Strong NER + schema-adherence reputation. Likely top pick. |
| **Gemma 2 27B** | Q4_K_M | ~16-18GB | Solid mid-tier; similar quality bar to Qwen 2.5 32B. |
| **Qwen3 32B / Qwen3.6 27B** | Q4_K_M | ~16-20GB | Newer Qwen generation; reports good NER. |
| **Llama 3.3 70B** | Q3_K_M | ~30GB | Tight — needs partial CPU offload or aggressive quant. Probably slow on a single 3090. |

**What bigger should fix:**

- Field-separation discipline (people-vs-entities)
- `action_items` coverage
- Schema robustness in general
- Cold-load timeout pressure (gemma3:12b currently ~27s/thought on the iMac;
  Qwen 2.5 32B on a 3090 should hit ~5-10s)
- **Topic emission consistency.** The v11 retag showed gemma3:12b's
  spontaneous emissions are the dominant variance source — `database`
  vs `databases`, missing `rust` on Rust-about content, `databases`
  on tangentially-DB-related content. Larger models follow the topic
  instruction more reliably; the v11 normalize layer compounds with
  this rather than substituting for it.

**What bigger probably WON'T fix:**

- "Bob-as-verb" syntactic ambiguity (still benefits from constrained-vocab
  or denylist regardless of model size)

## How to re-run the probe set

The eight probes live in `engram.tagger-test`. To benchmark a new tagger
model against this corpus:

1. Edit `~/.config/engram/engram.toml`: change `[tagger].model_name` and
   `model_id`. Bump `model_version` to force re-tagging.
2. Restart server + worker.
3. `engram tag --scope engram.tagger-test --rerun`
4. Fetch each probe via `mcp__engram__get_thought` (or `psql`) and compare
   against the expected results below.

| Probe ID | Label | Expected `people` | Expected `entities` (key ones) | Failure mode tested |
|---|---|---|---|---|
| `d9d20425…` | A | `[Ron]` | `[Rust, Fn, FnMut, FnOnce]` | Calibrate / Refactor / Document as verbs |
| `3e977745…` | B | `[Sarah]` | `[M5.1, M6]` | Mark / Will / Bob as verbs alongside Sarah |
| `423a19fb…` | C | `[]` | `[pgvector, HNSW, migration_audit]` | Configure / Build / Audit, no people |
| `88fbcdbe…` | D | `[Ron]` | `[iOS, mcp-remote, Claude Desktop]` | Review / Verify / Patch + real entities |
| `0c7874da…` | E | `[Sarah]` | (none expected — Sarah is the only proper noun) | Bob-as-verb rematch |
| `f5450659…` | F | `[Maria, David]` | `[pgvector, HNSW, engram-cli, cross-encoder]` | Field separation, multi-person + entity |
| `fef5bdd4…` | G | `[Priya]` | `[M7, trigram]` | Calibrate-style under gemma3:12b |
| `8acae5c9…` | H | `[Priya, Marcus, Elena]` | `[M7]` | Pure people |

Probes are pre-tagged from the 2026-05-22 round; the table records the
canonical "good" answers for re-running on new tagger configs.

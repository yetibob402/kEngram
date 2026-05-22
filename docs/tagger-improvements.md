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

### 2. Field separation (people-in-entities) — likely a one-off

**Symptom.** Probe E (gemma3:12b) put "Sarah" in BOTH `people` and
`entities`. Did NOT recur on F or H (also gemma3:12b, multi-person + entity
content).

**Status.** Insufficient evidence to call it systematic. Watch on real
corpus.

**If it recurs:**

- **Tighten schema prompt** with explicit "a person's name belongs in
  `people`, not `entities`" guidance.
- **Post-hoc dedup** in the tagger output adapter: anything appearing in
  both arrays gets removed from `entities` (people-wins-tie).

### 3. `action_items` coverage is uneven

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

**What bigger probably WON'T fix:**

- "Bob-as-verb" syntactic ambiguity (still benefits from constrained-vocab
  or denylist regardless of model size)
- Topic overreach (prompt issue, not capacity — larger models tend to
  hallucinate well-attested topics more confidently, not less)

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

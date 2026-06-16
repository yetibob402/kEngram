# Tagger Eval Harness

Reference for `kengram eval tagger` and `kengram eval export-corpus` — the offline
harness for measuring and comparing tagger models against a golden corpus.

It is the source of truth for *how the tagger is chosen*. If you are about to swap the
production tagger model or prompt, you run this first. For the prompt-iteration history
see [`tagger-improvements.md`](./tagger-improvements.md); for the backend contract see
[`tagger-backends.md`](./tagger-backends.md) and
[`tagger-sidecar-protocol.md`](./tagger-sidecar-protocol.md); for the milestone scope see
[`milestones/m7-operational-maturity.md`](./milestones/m7-operational-maturity.md).

---

## 1. What it is and why

The tagger turns a thought into structured tags (`people`, `entities`, `action_items`,
`topics`, `dates_mentioned`, `kind`). Which model and prompt do that *best* was, before
this harness, an unmeasured judgment call. `kengram eval tagger` makes it an objective,
repeatable measurement:

- **`export-corpus`** drafts a golden corpus from your live thoughts (read-only).
- **`eval tagger`** runs one or more candidate models ("arms") over that corpus and scores
  each against the golden labels — per-field precision/recall/F1, `kind` accuracy with a
  confusion matrix, optional stability across repeats, and latency.

Two design commitments make the results trustworthy:

- **Scored on *finalized* output.** Every headline number is computed *after* the same
  deterministic post-filters production persists (`finalize_tags`), so the score reflects
  what would actually land in the store — not raw model output. A per-field `finalize_delta`
  shows what the filters bought.
- **The eval never touches the database** (see §2). You can sweep arbitrary models while a
  live kengram keeps serving, with zero risk to the corpus.

---

## 2. Architecture and the no-DB invariant

The harness lives in `crates/kengram-cli/src/eval/`, seven small modules:

| module | role |
|---|---|
| `mod.rs` | CLI entry, dispatch, dry-run, exit codes |
| `corpus.rs` | golden-corpus format + load/validate |
| `arms.rs` | `models.toml` → constructed `Tagger` trait objects |
| `run.rs` | the run loop: dispatch, concurrency, progress, failure guard, per-arm flush |
| `score.rs` | pure scoring math (no I/O) |
| `report.rs` | the JSON report schema + human table |
| `export.rs` | drafts a corpus from the live DB (the **only** DB-touching module) |

Arms are constructed through the **same production builders** (`OpenAICompatibleTagger`,
`HttpTagger`) behind the `Tagger` trait, so the harness is backend-agnostic and anything
production enforces (e.g. the prompt/version provenance binding) applies to eval arms for free.

### The no-production-DB invariant

The `eval tagger` path is **database-free by construction**. The corpus file carries the
scope/vocab context the production pipeline would otherwise fetch from Postgres
(`known_scopes`, `scope_vocab`), so `finalize_tags` and vocab injection run without a DB. The
dispatch arm never receives a `Config` or `PgPool`; only `export-corpus` is given a read-only
connection.

This is enforced structurally **and pinned by a test**: `no_db_tripwire` (in `mod.rs`) asserts
that none of `mod/corpus/score/report/arms/run` contain `sqlx::` or `use sqlx`. If the eval
ever accidentally imports database code, that test fails.

*Why it matters:* earlier `kengram tag --rerun` dogfood runs wrote to the live corpus and had
to be manually unwound. The eval must be safe to run against any model list at any time.

---

## 3. The golden corpus

A corpus is a JSON file (`format_version: 1`, `kind: "tagger-golden-corpus"`):

```jsonc
{
  "format_version": 1,
  "kind": "tagger-golden-corpus",
  "provenance": { "exported_at": "...", "source": "...",
                  "draft_tagger_model": "...", "draft_tagger_version": 16, "item_count": 116 },
  "known_scopes": ["rjf.tech", "project.kengram", "..."],
  "scope_vocab": {
    "rjf.tech": { "topics": ["llm-inference", "..."], "entities": ["kengram", "..."] }
  },
  "items": [
    {
      "id": "<uuid-or-slug>",
      "content": "the thought text",
      "scope": "rjf.tech",
      "metadata": {},
      "reviewed": true,
      "notes": "optional adjudication note",
      "golden": { "people": [], "entities": ["..."], "action_items": ["..."],
                  "topics": ["..."], "dates_mentioned": ["..."], "kind": "idea" }
    }
  ]
}
```

- **`golden`** holds the *post-finalize* ground-truth tags an item should produce.
- **`scope_vocab`** is per-scope controlled-vocabulary snapshots (truncated to production's
  `scope_vocab_size` at export). It feeds vocab injection and topic normalization so the eval
  matches production behavior.
- **`reviewed`** gates scoring: `false` = draft (exported, not yet hand-checked), `true` =
  golden. Unreviewed items are **skipped by default** (with an stderr notice); pass
  `--include-unreviewed` to score them anyway.
- Load is validated (version/kind match, non-empty items, unique ids, non-empty content,
  object-valued metadata); a bad corpus is a usage error (exit 2).

### Building one

```bash
# 1. Draft from live thoughts (READ-ONLY; writes only the JSON). Drafts carry current
#    production tags as labels with reviewed:false.
kengram eval export-corpus --out eval/corpora/mine.json --scope-prefix rjf. --limit 100

# 2. Make labels golden — two ways:
#    (a) hand-review: fix each item's `golden`, set "reviewed": true.
#    (b) frontier-assisted (used for golden-v1): a capable model BLIND-labels each item from
#        content/scope/metadata only (never shown the drafts — avoids anchoring), bound to the
#        prompt + finalize semantics; flips reviewed:true on clear items, leaves bistable ones
#        for you to adjudicate. NEVER score a model family against its own answer key.
#        Worked example: eval/corpora/golden-v1-labeling-notes.md.
```

Corpora live gitignored in `eval/corpora/` (except the committed synthetic `example.json`) —
personal thought content never enters the repo.

---

## 4. Model arms (`models.toml`)

Each `[[arm]]` is one tagger under test (committed template: `eval/models.example.toml`):

```toml
[[arm]]
name            = "gemma4-31b-ctx16k"            # required, unique
provider        = "openai-compatible"            # or "http"
endpoint        = "http://localhost:11434/v1"
model_name      = "gemma4:31b-qat-ctx16k"        # required for openai-compatible; ignored for http
temperature     = 0.2                            # default 0.2 — NEVER 0 (see §8)
timeout_seconds = 300                            # default 180
# api_key_env   = "OPENROUTER_API_KEY"           # NAME of an env var holding the token — never the token
# system_prompt_file = "prompts/v17.txt"         # prompt A/B; REQUIRES model_version (provenance binding)
# model_version = 17
```

- **Providers:** `openai-compatible` (Ollama, vLLM, OpenRouter — any `/v1/chat/completions` with
  strict `json_schema`) and `http` (the kengram tagger-sidecar protocol, e.g. the deterministic
  GLiNER sidecar).
- **Defaults:** `temperature` 0.2, `timeout_seconds` 180. `model_version` defaults to the bundled
  tagger version for `openai-compatible`.
- **What the harness sends vs. what it can't:** the harness sends `temperature` and (if the arm
  runs vocab-on) the scope vocab, per request. It **cannot** set `num_ctx` or `presence_penalty` —
  those are Ollama Modelfile parameters and must be baked into the model (e.g. a
  `gemma4:31b-qat-ctx16k` built `FROM` the base with `PARAMETER num_ctx 16384` /
  `presence_penalty 1.5`). This is why context-capped, trace-bounded variants are pre-built.
- **Prompt A/B + provenance binding:** to test a custom prompt, set `system_prompt_file` *and* an
  explicit `model_version`. The production provenance binding rejects a custom prompt under the
  bundled version number, so a prompt change must carry its own version — the harness enforces the
  same rule (pinned by test).

---

## 5. Scoring methodology

All scoring is pure, deterministic math in `score.rs` (no I/O). Headline numbers are computed on
**finalized** output.

### Normalization
Every tag string is normalized before comparison: Unicode-lowercase, trim, collapse internal
whitespace. Sets dedupe and drop empties after normalization.

### Set fields — `people`, `entities`, `topics`, `dates_mentioned`
Compared as normalized sets vs. golden:
- TP = intersection, FP = predicted∖golden, FN = golden∖predicted.
- Precision `TP/(TP+FP)`, Recall `TP/(TP+FN)` (each 1.0 when the denominator is 0), F1
  `2PR/(P+R)` (0.0 when `P+R=0`). **Both-empty scores as perfect (F1 = 1.0).**

### `action_items` — fuzzy, greedy token-F1
Free-text items aren't exact-matchable, so:
- Each item is tokenized (normalize → non-alphanumerics to spaces → split → dedupe to a set).
- A predicted/golden pair's **token-F1** = `2·|A∩B|/(|A|+|B|)` over token sets.
- A pair is *eligible* if token-F1 ≥ **0.6** (`ACTION_ITEM_MATCH_THRESHOLD`). As a fallback, if
  token-F1 < 0.6 but one string contains the other, the pair score is floored to exactly 0.6.
- Eligible pairs are matched **greedily**, sorted by `(score desc, pred_idx asc, gold_idx asc)`;
  each side matches at most once. Matched pairs = TP, unmatched predictions = FP, unmatched
  goldens = FN → P/R/F1 as above.

### `kind` — 8×8 confusion matrix
Closed label set, fixed order: `observation, task, idea, reference, person_note, session,
decision_record, null`. The report carries the full matrix (rows = golden, cols = predicted) and
`accuracy = trace / total`. **`kind` accuracy is the gate** — it's the dominant production
failure mode the harness was built to discriminate.

### Micro vs. macro
- **Micro** (the headline `micro.f1`): TP/FP/FN aggregated across all calls, then P/R/F1.
- **Macro** (`macro_f1`): the unweighted mean of per-call F1s.

### `finalize_delta`
Each call is scored twice — on raw model output and on `finalize_tags(raw)`. The headline numbers
are the **finalized** ones; `finalize_delta = finalized_f1 − raw_micro_f1` shows what the
deterministic post-filters contributed for that field.

### Stability (only when `--repeats ≥ 2`)
- **`mean_kind_agreement`** — per item, the fraction of repeats agreeing with that item's modal
  `kind` (ties broken by label order), averaged over items.
- **`unanimous_kind_pct`** — fraction of items where all repeats agreed.
- **`mean_pairwise_jaccard`** — per field, the mean Jaccard over all repeat-pairs' normalized tag
  sets (empty-vs-empty = 1.0).
- **`unstable_items`** — item ids whose kind agreement < 1.0.

### Latency
`p50`/`p95` via nearest-rank (`ceil(p/100 · n)`), plus `mean`, in milliseconds.

### Determinism
Outcomes are sorted by `(item, repeat)` before aggregation, all tie-breaks are fixed (label order,
index order, the 0.6 substring floor), so the same corpus + arms + calls always yield the same
report.

---

## 6. Running it

```bash
# 0. (once) declare arms
cp eval/models.example.toml eval/models.toml      # then edit (gitignored)

# 1. preview cost — validates corpus+arms, prints call count + worst-case wall time, makes NO calls
kengram eval tagger --corpus eval/corpora/mine.json --dry-run

# 2. run
kengram eval tagger --corpus eval/corpora/mine.json --repeats 3 --vocab on \
  --concurrency 1 --max-consecutive-failures 10
# → JSON report in eval/reports/tagger-<RFC3339>.json ; summary table on stdout
```

### Flags

| flag | meaning |
|---|---|
| `--corpus PATH` | golden corpus (required) |
| `--repeats N` | N≥2 adds the stability metrics |
| `--vocab on\|off\|both` | scope-vocab injection; `both` runs each arm twice (`+vocab`/`-vocab` name suffixes) |
| `--limit N` | cap items (after the reviewed filter) — quick runs |
| `--arm NAME` | restrict to named arm(s); validated against the models file |
| `--concurrency N` | parallel calls (default 1; **1 makes the consecutive-failure guard exact**) |
| `--max-consecutive-failures N` | abandon an arm after N consecutive failures (default 5; 0 disables) |
| `--include-unreviewed` | score `reviewed:false` items too |
| `--out PATH` | report path (else auto-named under `eval/reports/`) |
| `--json` | also emit the report JSON to stdout |
| `--since RFC3339` | (`export-corpus` only) thoughts created at/after the timestamp |

### Unattended-run hardening
- **Per-arm flush:** after each arm the report is written with `"complete": false`; the final write
  sets `"complete": true`. A crash/Ctrl-C keeps every finished arm.
- **Consecutive-failure guard:** abandons a looping model rather than burning hours of timeouts;
  aborted arms are marked (`aborted`), their resolved calls still score.
- **Progress line** to stderr, e.g.
  `[arm 2/4 …] 120/348 done · 3 failed total (2.5%) · avg 13.8s/call · ETA 52m`.
- **Exit codes:** `0` success · `1` an arm hit 100% failure or was aborted (report still written) ·
  `2` usage/config error.

### Multi-arm Ollama runs — set `OLLAMA_MAX_LOADED_MODELS=1`
Launch multi-arm Ollama sweeps with `OLLAMA_MAX_LOADED_MODELS=1` (and/or `OLLAMA_KEEP_ALIVE=0`) so
each arm loads with the *previous* arm's model evicted. Otherwise arm N+1 is placed while arm N is
still resident (keep-alive), Ollama underestimates free VRAM and strands layers on CPU (~2×
slowdown, with VRAM left idle), and it does **not** re-balance after arm N later evicts. Single-arm
runs and non-Ollama backends are unaffected. (This is an Ollama scheduling behavior; the harness is
backend-agnostic, so the fix is at the Ollama layer, not in the harness.)

---

## 7. Reading a report

```jsonc
{
  "format_version": 1,
  "run":   { "started_at", "corpus_path", "corpus_item_count", "scored_item_count",
             "unreviewed_items_included", "repeats", "vocab_mode", "kengram_version", "complete" },
  "arms":  [ { "name", "provider", "endpoint", "model_name", "model_version", "prompt",
               "vocab_used", "aborted",
               "calls":   { "attempted", "succeeded", "failed", "skipped", "failure_rate" },
               "latency_ms": { "p50", "p95", "mean" },
               "kind":    { "accuracy", "labels", "matrix" },
               "fields":  { "<field>": { "micro": {precision,recall,f1,tp,fp,fn},
                                          "macro_f1", "raw_micro_f1", "finalize_delta" } },
               "stability": { "mean_kind_agreement", "unanimous_kind_pct",
                              "mean_pairwise_jaccard", "unstable_items" },   // only if repeats≥2
               "worst_items": [ { "id", "mean_f1", "golden", "predicted" } ] } ],
  "pairwise": [ { "a", "b", "kind_accuracy_delta", "micro_f1_delta": {"<field>": delta} } ],
  "calls":  [ { "arm", "item_id", "repeat", "vocab_used", "latency_ms",
                "raw_tags", "finalized_tags", "error" } ]
}
```

How to read it:
- **`kind.accuracy` is the headline gate.** Then per-field `fields.<field>.micro.f1`. Watch
  `action_items` — historically the field that separates models.
- **`finalize_delta`** tells you whether the win came from the model or the post-filters.
- **`calls.failure_rate`** + the `calls[].error` strings tell you *how* a model fails (timeout vs
  truncation vs empty response — see §8). Distinguish per-*call* rate (diluted by repeats) from
  per-*item* failures.
- **`stability`** separates a genuinely-better model from a noisy one.
- **`worst_items`** are the bottom items by mean F1 — the concrete failures to inspect.
- **Everything is re-derivable from `calls[]`** — the full raw+finalized output per call is the
  audit trail; aggregates are reproducible from it alone.

---

## 8. Operational lessons (hard-won)

These are the traps that cost real time. Dated specifics (model-selection decisions, eval reports)
live in kengram (`project.kengram` scope, `decision_record` kind); the principles:

- **Never run a reasoning tagger at `temperature 0`.** Greedy decoding makes degenerate repetition
  loops *deterministic* (~36% failures vs near-zero at 0.2). `0.2` is the validated floor.
- **Reasoning-model failure modes are distinct — read the error string.** A runaway reasoning trace
  manifests as a **300s timeout** (`truncated=0`; ran past the wall) at large context, or as
  **context-truncation / malformed JSON** at small context (the trace overruns `num_ctx`), or as an
  **empty response** (`content=""`) on models that bail. Same item, different mode by context size.
- **The keep-alive arm-placement trap.** With `OLLAMA_KEEP_ALIVE` keeping arm N resident, arm N+1
  loads under a memory shadow, gets layers stranded on CPU (~2× slow with VRAM idle), and Ollama
  never re-balances. Fix: `OLLAMA_MAX_LOADED_MODELS=1` for multi-arm sweeps (see §6).
- **`--vocab on` vs production `scope_vocab_enabled`.** The eval defaults to vocab-on for production
  parity, but `topics` is then circular — golden topics derive from production's own historical
  emissions, so `topics` F1 flatters the incumbent. Treat `kind`/`people`/`entities`/`action_items`
  as the real discriminators; if production runs `scope_vocab_enabled = false`, the eval's `topics`
  number won't transfer.
- **Residence / context / KV interplay.** A model that fits one GPU at short context spills (or
  splits across cards) at long context — the KV cache counts against VRAM. `num_ctx` is the
  footprint lever; check `ollama ps` + `nvidia-smi` (per-card), not just "100% GPU" (which means
  *no CPU*, not *one card*). Bounding `num_ctx` to what the tagger actually needs can free a whole
  card.
- **Output-length guard (planned hardening).** The runaway-trace failures (timeout *or* truncation)
  are the same root cause; inspecting the tagger response for a length-stop and raising a distinct
  `TaggerTruncated` error would convert a 300s hang into a fast, diagnosable skip.
- **Capability ≠ on-task performance.** Reasoning-leaderboard strength does not predict tagging
  quality — tagging rewards instruction-following, schema discipline, and taxonomic judgment, not
  chain-of-thought. Measure on the golden corpus; don't infer from benchmarks.

---

## 9. See also

- [`tagger-improvements.md`](./tagger-improvements.md) — prompt-version iteration log.
- `tagger-sweep.sh` + `crates/kengram-extract/examples/tagger_eval.rs` — the fast, assertion-fixture
  lane for quick prompt iteration (distinct from this scored golden-corpus harness).
- [`tagger-backends.md`](./tagger-backends.md), [`tagger-sidecar-protocol.md`](./tagger-sidecar-protocol.md)
  — the tagger backend/sidecar contracts.
- [`milestones/m7-operational-maturity.md`](./milestones/m7-operational-maturity.md) — M7.1 scope.
- Code: `crates/kengram-cli/src/eval/` (and `eval/corpora/golden-v1-labeling-notes.md`).

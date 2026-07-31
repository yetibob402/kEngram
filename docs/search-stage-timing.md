# Search stage-timing instrumentation

## What

Every `search_thoughts` request emits **one** structured log line
(`target = kengram_search_stage_timing`) with per-stage elapsed ms, keyed by
`request_id`.

`include_profile` already returned the same timings in the MCP response when
requested. This change **reuses** that profile computation (always computed
internally) and:

1. Logs one line per request (bounded volume — not per-stage spam)
2. Accepts optional client `request_tag` (e.g. eval `KGR-*` id), echoes as `request_id`
3. Generates a UUID when `request_tag` is absent/empty

## Zero behavior change

- No change to candidate sizes, ranking, filters, or config defaults
- Profile still only **returned** when `include_profile=true`
- Logging is additive

## Correlation with eval

Pass `request_tag: <query_id>` on each `search_thoughts` call. Service logs and
response both carry `request_id` for join against eval rows.

## Gold protection (folded lane)

- Tables: `gold_protection_manifest` (migration 0019)
- Populate offline: `scripts/populate_gold_protection_manifest.py --corpus eval/corpora/gold100.json --sql-only`
- Per-question regression: `scripts/eval_gold_protection_check.py --corpus ... --results ... --k 10`
- **Do not apply migrations/restarts to live fleet memory without knox gate.**

## Deploy note

Code change only. Live kEngram restart/deploy requires explicit knox gate.

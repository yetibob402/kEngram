#!/usr/bin/env bash
# schema-binary-migrate-gate.sh — bidirectional deploy hygiene for kEngram
#
# Compares:
#   - main_max: highest top-level NNNN_*.sql version on immutable origin/main
#   - prod_max: max(version) from _sqlx_migrations where success (requires DATABASE_URL)
#
# Exit codes:
#   0  equal (green) — or SCHEMA_AHEAD_OK=1 when prod > main with explicit allow
#   2  schema behind main (prod < main_max) — deploy-without-migrate class
#   3  schema ahead of main (prod > main_max) without SCHEMA_AHEAD_OK=1
#   4  usage / resolve failure
#
# Does NOT use `ls migrations | sort | tail` (includes rollback/, wrong tip).
set -euo pipefail

REMOTE_REF="${REMOTE_REF:-origin/main}"
ALLOW_AHEAD="${SCHEMA_AHEAD_OK:-0}"

usage() {
  cat <<'EOF'
Usage: schema-binary-migrate-gate.sh [--print-main-max] [--compare]
  --print-main-max   print max numeric migration version on REMOTE_REF (default origin/main)
  --compare          compare prod DATABASE_URL successful max vs main_max (default if DATABASE_URL set)

Env:
  REMOTE_REF         git ref for immutable main tree (default: origin/main)
  DATABASE_URL       prod/live DSN for _sqlx_migrations
  SCHEMA_AHEAD_OK=1  allow prod > main (must be documented on owning PR)
EOF
}

main_max_from_ref() {
  local ref="$1"
  # Immutable tree: only top-level migrations/NNNN_*.sql (exclude rollback/, non-numeric)
  git ls-tree --name-only "$ref" migrations/ \
    | sed -n 's|^migrations/\([0-9][0-9][0-9][0-9]\)_.*\.sql$|\1|p' \
    | sed 's/^0*//' \
    | awk 'NF{print $1+0}' \
    | sort -n \
    | tail -1
}

prod_max_from_db() {
  local url="$1"
  psql "$url" -v ON_ERROR_STOP=1 -tAc \
    "SELECT COALESCE(MAX(version), 0) FROM _sqlx_migrations WHERE success"
}

cmd_print=
cmd_compare=
if [[ $# -eq 0 ]]; then
  if [[ -n "${DATABASE_URL:-}" ]]; then
    cmd_compare=1
  else
    cmd_print=1
  fi
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-main-max) cmd_print=1; shift ;;
    --compare) cmd_compare=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 4 ;;
  esac
done

if ! git rev-parse -q --verify "$REMOTE_REF" >/dev/null; then
  echo "error: cannot resolve REMOTE_REF=$REMOTE_REF (fetch origin first)" >&2
  exit 4
fi

main_max="$(main_max_from_ref "$REMOTE_REF")"
if [[ -z "${main_max:-}" ]]; then
  echo "error: no numeric migrations found under $REMOTE_REF migrations/" >&2
  exit 4
fi

if [[ -n "$cmd_print" && -z "$cmd_compare" ]]; then
  echo "main_max=$main_max ref=$REMOTE_REF"
  exit 0
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "error: DATABASE_URL required for --compare" >&2
  exit 4
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "error: psql not on PATH" >&2
  exit 4
fi

prod_max="$(prod_max_from_db "$DATABASE_URL" | tr -d '[:space:]')"
if [[ -z "$prod_max" || ! "$prod_max" =~ ^[0-9]+$ ]]; then
  echo "error: could not read successful prod max from _sqlx_migrations" >&2
  exit 4
fi

echo "main_max=$main_max ref=$REMOTE_REF"
echo "prod_max=$prod_max (success rows only)"

if [[ "$prod_max" -eq "$main_max" ]]; then
  echo "state=equal GREEN"
  exit 0
elif [[ "$prod_max" -lt "$main_max" ]]; then
  echo "state=schema_behind_binary_or_main RED (deploy-without-migrate class)"
  echo "prod is behind immutable main; apply pending migrations before claiming deploy complete"
  exit 2
else
  echo "state=schema_ahead_of_main"
  if [[ "$ALLOW_AHEAD" == "1" ]]; then
    echo "SCHEMA_AHEAD_OK=1 — allowed only if owning PR body records the delta"
    exit 0
  fi
  echo "RED: prod > main without SCHEMA_AHEAD_OK=1 (record on owning PR or merge the migration)"
  exit 3
fi

#!/usr/bin/env bash
# Registered acceptance: supersession::tests on disposable non-prod PostgreSQL.
# Provisions disposable LOGIN passwords for dedicated/ACL roles from DATABASE_URL
# so a clean migrate 1..36 is executable without undocumented manual setup.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SQLX_OFFLINE=true

if test -z "${DATABASE_URL:-}"; then
  echo "FAIL source-event-supersession: DATABASE_URL required (disposable non-prod)" >&2
  exit 1
fi
case "$DATABASE_URL" in
  *kengram_prod*|*PROD*|*prod-write*)
    echo "FAIL source-event-supersession: refusing production-looking DATABASE_URL" >&2
    exit 1
    ;;
esac

# Exact selected denominator for this head (must match #[tokio::test] count).
SELECTED=24

# Parse password from postgres URL user:pass@host
# Supports postgres://user:pass@host:port/db
PASS="$(printf '%s' "$DATABASE_URL" | sed -n 's#.*://[^:]*:\([^@]*\)@.*#\1#p')"
if test -z "$PASS"; then
  echo "FAIL source-event-supersession: cannot parse password from DATABASE_URL for disposable role bootstrap" >&2
  exit 1
fi

# Disposable-only role password bootstrap (test contract; never for prod URLs above).
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<SQL
DO \$\$
DECLARE
  r text;
  roles text[] := ARRAY[
    'kengram_rt_supersession',
    'kengram_rt_native_mcp',
    'kengram_rt_session',
    'kengram_rt_telegram'
  ];
BEGIN
  FOREACH r IN ARRAY roles LOOP
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('ALTER ROLE %I WITH LOGIN PASSWORD %L', r, '$PASS');
    END IF;
  END LOOP;
END\$\$;
SQL

set +e
OUT="$(cargo test -p kengram-storage supersession::tests -- --test-threads=1 --nocapture 2>&1)"
RC=$?
set -e
printf '%s\n' "$OUT"

# Require cargo actually ran tests
if ! printf '%s\n' "$OUT" | grep -qE '^running[[:space:]]+[0-9]+[[:space:]]+tests?$'; then
  echo "FAIL source-event-supersession: cargo did not report running tests (marker-only forbidden)" >&2
  exit 1
fi
RUNNING="$(printf '%s\n' "$OUT" | sed -n 's/^running \([0-9][0-9]*\) tests*$/\1/p' | tail -1)"
if test -z "$RUNNING" || test "$RUNNING" -ne "$SELECTED"; then
  echo "FAIL source-event-supersession: running=$RUNNING selected=$SELECTED mismatch" >&2
  exit 1
fi

# Reject ignored/skipped
if printf '%s\n' "$OUT" | grep -qE 'ignored|filtered out.*ignored'; then
  # cargo always prints "filtered out" for other lib tests - allow that
  :
fi
# Parse final summary line only
SUMMARY="$(printf '%s\n' "$OUT" | grep -E '^test result:' | tail -1 || true)"
if test -z "$SUMMARY"; then
  echo "FAIL source-event-supersession: missing cargo test result summary" >&2
  exit 1
fi
if ! printf '%s\n' "$SUMMARY" | grep -q 'test result: ok'; then
  echo "FAIL source-event-supersession: $SUMMARY" >&2
  exit 1
fi
PASSED="$(printf '%s\n' "$SUMMARY" | sed -n 's/.*ok\. \([0-9][0-9]*\) passed.*/\1/p')"
FAILED_N="$(printf '%s\n' "$SUMMARY" | sed -n 's/.*passed; \([0-9][0-9]*\) failed.*/\1/p')"
IGNORED_N="$(printf '%s\n' "$SUMMARY" | sed -n 's/.*failed; \([0-9][0-9]*\) ignored.*/\1/p')"
if test -z "$PASSED" || test "$PASSED" -ne "$SELECTED"; then
  echo "FAIL source-event-supersession: passed=$PASSED selected=$SELECTED" >&2
  exit 1
fi
if test "${FAILED_N:-1}" -ne 0; then
  echo "FAIL source-event-supersession: failed=$FAILED_N" >&2
  exit 1
fi
if test "${IGNORED_N:-1}" -ne 0; then
  echo "FAIL source-event-supersession: ignored=$IGNORED_N (skips forbidden)" >&2
  exit 1
fi
if test "$RC" -ne 0; then
  echo "FAIL source-event-supersession cargo_rc=$RC" >&2
  exit 1
fi

printf 'PASS source-event-supersession selected=%s executed=%s failed=0 skipped=0\n' "$SELECTED" "$PASSED"

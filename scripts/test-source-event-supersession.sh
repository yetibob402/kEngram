#!/usr/bin/env bash
# Registered acceptance: supersession::tests on disposable non-prod PostgreSQL.
# R1 successor: real Cargo child status + fresh test-binary execution receipt.
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

# Disposable execution-proof surface (mode 0700).
RECEIPT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kengram-ss-exec.XXXXXX")"
chmod 700 "$RECEIPT_DIR"
RECEIPT_PATH="${RECEIPT_DIR}/execution.receipt"
NONCE="$(openssl rand -hex 16)"
export KENGRAM_SS_EXEC_RECEIPT_PATH="$RECEIPT_PATH"
export KENGRAM_SS_EXEC_NONCE="$NONCE"
cleanup() {
  local st=$?
  rm -rf "$RECEIPT_DIR" || st=1
  exit "$st"
}
trap cleanup EXIT

# Parse password from postgres URL user:pass@host
PASS="$(printf '%s' "$DATABASE_URL" | sed -n 's#.*://[^:]*:\([^@]*\)@.*#\1#p')"
if test -z "$PASS"; then
  echo "FAIL source-event-supersession: cannot parse password from DATABASE_URL for disposable role bootstrap" >&2
  exit 1
fi

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

# Dynamic selected denominator from Cargo list (not hard-coded).
LIST_OUT="$(cargo test -p kengram-storage supersession::tests -- --list 2>&1 || true)"
SELECTED="$(printf '%s\n' "$LIST_OUT" | grep -cE ': test$' || true)"
if test -z "$SELECTED" || test "$SELECTED" -lt 1; then
  echo "FAIL source-event-supersession: cargo list produced non-positive selection" >&2
  printf '%s\n' "$LIST_OUT" >&2
  exit 1
fi
# Require execution-receipt test and successor-named cases in the selected list.
for need in \
  case_00_execution_receipt_writes_once \
  case_10b_old_thought_missing \
  case_10c_old_thought_chain_blocked \
  case_13_receipt_oracle_four_envelopes \
  case_snapshot_metadata_sensitivity
do
  if ! printf '%s\n' "$LIST_OUT" | grep -q "${need}: test"; then
    echo "FAIL source-event-supersession: required test missing from list: $need" >&2
    exit 1
  fi
done

CARGO_BIN="$(command -v cargo)"
if test -z "$CARGO_BIN"; then
  echo "FAIL source-event-supersession: cargo not found" >&2
  exit 1
fi

set +e
set -o pipefail
"$CARGO_BIN" test -p kengram-storage supersession::tests -- --test-threads=1 --nocapture 2>&1 \
  | tee "${RECEIPT_DIR}/cargo.out"
RC=${PIPESTATUS[0]}
set +o pipefail
set -e
OUT="$(cat "${RECEIPT_DIR}/cargo.out")"
printf '%s\n' "$OUT"

if test "$RC" -ne 0; then
  echo "FAIL source-event-supersession: cargo child exit $RC" >&2
  exit 1
fi

# Fresh execution receipt from the selected Rust test binary (not stdout text).
if test ! -f "$RECEIPT_PATH"; then
  echo "FAIL source-event-supersession: missing execution receipt (fabricated cargo text cannot PASS)" >&2
  exit 1
fi
# Exactly one line: nonce pid
LINE_COUNT="$(wc -l <"$RECEIPT_PATH" | tr -d ' ')"
if test "$LINE_COUNT" -ne 1; then
  echo "FAIL source-event-supersession: execution receipt must be exactly one line (got $LINE_COUNT)" >&2
  exit 1
fi
read -r GOT_NONCE GOT_PID _rest <"$RECEIPT_PATH" || true
if test -n "${_rest:-}"; then
  echo "FAIL source-event-supersession: execution receipt has trailing fields (exact nonce pid only)" >&2
  exit 1
fi
if test "$GOT_NONCE" != "$NONCE"; then
  echo "FAIL source-event-supersession: execution receipt nonce mismatch" >&2
  exit 1
fi
if ! printf '%s' "$GOT_PID" | grep -qE '^[1-9][0-9]*$'; then
  echo "FAIL source-event-supersession: execution receipt PID invalid" >&2
  exit 1
fi

# Diagnostic cargo summary only after child+receipt proof.
if ! printf '%s\n' "$OUT" | grep -qE '^running[[:space:]]+[0-9]+[[:space:]]+tests?$'; then
  echo "FAIL source-event-supersession: cargo did not report running tests" >&2
  exit 1
fi
RUNNING="$(printf '%s\n' "$OUT" | sed -n 's/^running \([0-9][0-9]*\) tests*$/\1/p' | tail -1)"
if test -z "$RUNNING" || test "$RUNNING" -ne "$SELECTED"; then
  echo "FAIL source-event-supersession: running=$RUNNING selected=$SELECTED mismatch" >&2
  exit 1
fi
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

printf 'PASS source-event-supersession selected=%s executed=%s failed=0 skipped=0\n' "$SELECTED" "$PASSED"

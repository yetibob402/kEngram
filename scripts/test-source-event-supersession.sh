#!/usr/bin/env bash
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

SELECTED=3
# Run only supersession tests serially
set +e
OUT="$(cargo test -p kengram-storage supersession::tests -- --test-threads=1 --nocapture 2>&1)"
RC=$?
set -e
printf '%s\n' "$OUT"

# Count executed tests from cargo output
EXECUTED="$(printf '%s\n' "$OUT" | grep -c 'test supersession::tests::' || true)"
FAILED="$(printf '%s\n' "$OUT" | grep -c 'FAILED\|failed' || true)"
# Prefer summary line
if printf '%s\n' "$OUT" | grep -q 'test result: ok'; then
  # extract from "test result: ok. N passed"
  PASSED="$(printf '%s\n' "$OUT" | sed -n 's/.*test result: ok\. \([0-9][0-9]*\) passed.*/\1/p' | tail -1)"
  EXECUTED="${PASSED:-$EXECUTED}"
  FAILED=0
elif printf '%s\n' "$OUT" | grep -q 'test result: FAILED'; then
  FAILED=1
fi

if test "$RC" -ne 0 || test "$FAILED" -ne 0; then
  echo "FAIL source-event-supersession cargo_rc=$RC" >&2
  exit 1
fi
if test -z "$EXECUTED" || test "$EXECUTED" -eq 0; then
  echo "FAIL source-event-supersession zero executed" >&2
  exit 1
fi
if test "$EXECUTED" -lt "$SELECTED"; then
  echo "FAIL source-event-supersession selected=$SELECTED executed=$EXECUTED count mismatch" >&2
  exit 1
fi

printf 'PASS source-event-supersession selected=%s executed=%s failed=0 skipped=0\n' "$EXECUTED" "$EXECUTED"

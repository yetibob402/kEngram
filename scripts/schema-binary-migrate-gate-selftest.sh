#!/usr/bin/env bash
# Watched-to-fail self-test for schema-binary-migrate-gate.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GATE="$ROOT/scripts/schema-binary-migrate-gate.sh"
fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }

git fetch origin -q 2>/dev/null || true
if ! git rev-parse -q --verify origin/main >/dev/null; then
  fail "origin/main missing"
fi

main_max="$("$GATE" --print-main-max)"
echo "$main_max" | grep -q '^main_max=[0-9]' || fail "print-main-max format: $main_max"
ver="${main_max#main_max=}"
ver="${ver%% *}"
[[ "$ver" =~ ^[0-9]+$ ]] || fail "non-numeric main_max: $ver"
# Must not be confused with rollback dir (lexicographic tail)
[[ "$ver" -ge 30 ]] || fail "main_max unexpectedly low: $ver"

# Prove we ignore rollback/: max from ls|sort|tail style is wrong
bad_tail="$(git ls-tree --name-only origin/main migrations/ | sed 's|migrations/||' | sort | tail -1)"
[[ "$bad_tail" == "rollback" ]] || echo "note: tip name is $bad_tail (still must not use raw sort|tail alone)"

# Watched schema-behind: fake prod max = main-1
export DATABASE_URL="postgres://unused"
# inject by wrapping — call functions via env mock not available; use bash subshell with stub psql
tmpdir="$(mktemp -d)"
cat > "$tmpdir/psql" <<'PSQL'
#!/bin/bash
# emit fake prod max from FAKE_PROD_MAX
echo "${FAKE_PROD_MAX}"
PSQL
chmod +x "$tmpdir/psql"
export PATH="$tmpdir:$PATH"

# equal green
export FAKE_PROD_MAX="$ver"
export SCHEMA_AHEAD_OK=0
out="$("$GATE" --compare 2>&1)" || rc=$?
rc=${rc:-0}
echo "$out" | grep -q 'state=equal' || fail "equal expected, got rc=$rc out=$out"
[[ "$rc" -eq 0 ]] || fail "equal exit 0, got $rc"

# schema-behind red
export FAKE_PROD_MAX=$((ver - 1))
set +e
out="$("$GATE" --compare 2>&1)"
rc=$?
set -e
echo "$out" | grep -q 'schema_behind' || fail "behind expected: $out"
[[ "$rc" -eq 2 ]] || fail "behind exit 2, got $rc"

# schema-ahead red without allow
export FAKE_PROD_MAX=$((ver + 1))
set +e
out="$("$GATE" --compare 2>&1)"
rc=$?
set -e
echo "$out" | grep -q 'schema_ahead_of_main' || fail "ahead expected: $out"
[[ "$rc" -eq 3 ]] || fail "ahead exit 3, got $rc"

# schema-ahead green with allow
export SCHEMA_AHEAD_OK=1
set +e
out="$("$GATE" --compare 2>&1)"
rc=$?
set -e
[[ "$rc" -eq 0 ]] || fail "ahead allowed exit 0, got $rc out=$out"

rm -rf "$tmpdir"
echo "SELFTEST PASS: equal green, behind RED(2), ahead RED(3), ahead+OK GREEN"

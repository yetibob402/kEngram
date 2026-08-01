#!/bin/bash
set -euo pipefail

export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATIONS="$ROOT/migrations"
SELECTED=8
EXECUTED=0
IMAGE="pgvector/pgvector:pg16"
ROW_ID="1623cb17-62cc-4159-9afd-9b999e9e792e"
SHA31="2cd51f7d9b960c31361053bd4650d3a7fba64495cc6ae5bf8516b2a83c527894"
SHA384_31="064ac07479779aae1fc349038ec3f54a61eb5d304c3f6c5e954e6a08deff12bc8ff621bb96f0904eb28d346d58aa9b73"
BLOB31="7a4f62b91beef4cf83013c1a2cd8fe0cc4d5e253"
SHA35="a5e8a91febc5fdbcf97d9fb6d9d471e8cef8d6a4b70b6a681e35c1ea89340986"
SHA384_35="abbb076e8970608fa26ae6f049843ddc3cfff9269beba13d476a11ecb026e9f1e6e0062028ac4c786f964c47431ca15e"
BLOB35="c2986af1225b9fa14f697891df229476120536db"
EXPECTED_DEF="CHECK ((status = ANY (ARRAY['pending'::text, 'stored'::text, 'conflict'::text, 'dlq'::text, 'skipped'::text, 'resolved'::text])))"
EXPECTED_COMMENT="pending|stored|conflict|dlq|skipped|resolved — resolved is terminal resolution of conflict; keep row + error + metadata.conflict evidence"

fail() {
  printf 'FAIL migration-0035-reconciliation: %s\n' "$*" >&2
  exit 1
}

pass_case() {
  EXECUTED=$((EXECUTED + 1))
  printf 'PASS migration-0035-reconciliation case=%s selected=%s executed=%s\n' "$1" "$SELECTED" "$EXECUTED"
}

for tool in awk cmp cp docker find git grep mktemp mv sed shasum sort sqlx wc; do
  command -v "$tool" >/dev/null 2>&1 || fail "missing required tool: $tool"
done

WORK="$(mktemp -d /tmp/kengram-0035-accept.XXXXXX)"
case "$WORK" in
  /tmp/kengram-0035-accept.*) ;;
  *) fail "unexpected temporary directory: $WORK" ;;
esac
test -d "$WORK" && test ! -L "$WORK" || fail "temporary directory is not a real directory"

CONTAINER="kengram-0035-accept-$$-${RANDOM}"
case "$CONTAINER" in
  kengram-0035-accept-[0-9]*-[0-9]*) ;;
  *) fail "unexpected container name: $CONTAINER" ;;
esac
CONTAINER_ID=""

cleanup() {
  prior_rc=$?
  cleanup_rc=0
  trap - EXIT INT TERM

  if test -n "$CONTAINER_ID" && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    actual_id="$(docker inspect --format '{{.Id}}' "$CONTAINER" 2>/dev/null || true)"
    actual_label="$(docker inspect --format '{{index .Config.Labels "io.yetiwerks.kengram-0035-accept"}}' "$CONTAINER" 2>/dev/null || true)"
    if test "$actual_id" = "$CONTAINER_ID" && test "$actual_label" = "$CONTAINER"; then
      docker stop -t 5 "$CONTAINER" >/dev/null 2>&1 || cleanup_rc=1
    else
      printf 'FAIL cleanup refused unexpected container identity name=%s\n' "$CONTAINER" >&2
      cleanup_rc=1
    fi
  fi

  if test -d "$WORK" && test ! -L "$WORK"; then
    if test -n "$(find "$WORK" -type l -print -quit)"; then
      printf 'FAIL cleanup refused temporary tree containing a symlink: %s\n' "$WORK" >&2
      cleanup_rc=1
    else
      /bin/rm -rf -- "$WORK" || cleanup_rc=1
    fi
  fi

  if test "$prior_rc" -eq 0 && test "$cleanup_rc" -ne 0; then
    exit "$cleanup_rc"
  fi
  exit "$prior_rc"
}
trap cleanup EXIT INT TERM

source_manifest() {
  source_dir="$1"
  output="$2"
  : >"$output"
  count=0
  for file in "$source_dir"/[0-9][0-9][0-9][0-9]_*.sql; do
    test -f "$file" && test ! -L "$file" || fail "invalid migration file: $file"
    base="${file##*/}"
    case "$base" in
      [0-9][0-9][0-9][0-9]_*.sql) ;;
      *) fail "unparseable migration filename: $base" ;;
    esac
    padded="${base%%_*}"
    version=$((10#$padded))
    checksum="$(shasum -a 384 "$file")" || fail "SHA-384 producer failed: $base"
    checksum="${checksum%% *}"
    case "$checksum" in
      [0-9a-f][0-9a-f]*) ;;
      *) fail "invalid SHA-384 output: $base" ;;
    esac
    test "${#checksum}" -eq 96 || fail "wrong SHA-384 length: $base"
    printf '%s|%s\n' "$version" "$checksum" >>"$output"
    count=$((count + 1))
  done
  test "$count" -gt 0 || fail "source manifest selected zero files"
  sort -t '|' -k1,1n "$output" -o "$output" || fail "source manifest sort failed"
  awk -F '|' 'NR > 1 && $1 == previous { exit 7 } { previous = $1 }' "$output" || fail "duplicate migration version"
}

assert_exact_1_35() {
  manifest="$1"
  awk -F '|' 'NR != $1 { exit 7 } END { if (NR != 35) exit 8 }' "$manifest" || fail "source versions are not exactly 1..35"
}

psql_query() {
  query="$1"
  docker exec "$CONTAINER" psql -X -v ON_ERROR_STOP=1 -U kengram_accept -d kengram_accept -At -F '|' -c "$query"
}

ledger_manifest() {
  output="$1"
  psql_query "SELECT version, encode(checksum, 'hex') FROM _sqlx_migrations WHERE success ORDER BY version" >"$output" || fail "ledger manifest producer failed"
}

capture_state() {
  prefix="$1"
  psql_query "SELECT oid::text, xmin::text, pg_get_constraintdef(oid), COALESCE(obj_description(oid, 'pg_constraint'), '<NULL>') FROM pg_constraint WHERE conrelid='argus_source_events'::regclass AND conname='argus_source_events_status_check'" >"${prefix}.constraint"
  psql_query "SELECT xmin::text,id::text,namespace,source_ref,payload_hash,COALESCE(thought_id::text,'<NULL>'),status,COALESCE(error,'<NULL>'),first_seen_at::text,last_seen_at::text,metadata::text FROM argus_source_events WHERE id='${ROW_ID}'" >"${prefix}.source-row"
  psql_query "SELECT xmin::text,version::text,description,installed_on::text,success::text,encode(checksum,'hex'),execution_time::text FROM _sqlx_migrations ORDER BY version" >"${prefix}.ledger-all"
  psql_query "SELECT xmin::text,version::text,description,installed_on::text,success::text,encode(checksum,'hex'),execution_time::text FROM _sqlx_migrations WHERE version=35" >"${prefix}.ledger-35"
  test "$(wc -l <"${prefix}.constraint" | tr -d ' ')" = 1 || fail "constraint identity count is not one"
  test "$(wc -l <"${prefix}.source-row" | tr -d ' ')" = 1 || fail "source-row identity count is not one"
}

assert_state_equal() {
  before="$1"
  after="$2"
  for suffix in constraint source-row ledger-all ledger-35; do
    cmp -s "${before}.${suffix}" "${after}.${suffix}" || fail "state changed unexpectedly: $suffix"
  done
}

FULL_MANIFEST="$WORK/source.manifest"
source_manifest "$MIGRATIONS" "$FULL_MANIFEST"
assert_exact_1_35 "$FULL_MANIFEST"

test "$(shasum -a 256 "$MIGRATIONS/0031_doc_source_ref_v2_aliases.sql" | awk '{print $1}')" = "$SHA31" || fail "0031 SHA-256 mismatch"
test "$(shasum -a 384 "$MIGRATIONS/0031_doc_source_ref_v2_aliases.sql" | awk '{print $1}')" = "$SHA384_31" || fail "0031 SHA-384 mismatch"
test "$(git -C "$ROOT" hash-object migrations/0031_doc_source_ref_v2_aliases.sql)" = "$BLOB31" || fail "0031 blob mismatch"
test "$(shasum -a 256 "$MIGRATIONS/0035_argus_source_events_resolved_status.sql" | awk '{print $1}')" = "$SHA35" || fail "0035 SHA-256 mismatch"
test "$(shasum -a 384 "$MIGRATIONS/0035_argus_source_events_resolved_status.sql" | awk '{print $1}')" = "$SHA384_35" || fail "0035 SHA-384 mismatch"
test "$(git -C "$ROOT" hash-object migrations/0035_argus_source_events_resolved_status.sql)" = "$BLOB35" || fail "0035 blob mismatch"
test "$(grep -Fxc './scripts/test-migration-0035-reconciliation.sh' "$ROOT/AGENTS.md")" = 1 || fail "AGENTS exact command registration count was not one"
pass_case exact-source-identities

CURRENT_SOURCE="$WORK/current-source"
ROLLBACK_SOURCE="$WORK/rollback-source"
MUTANT_SOURCE="$WORK/checksum-mutant-source"
mkdir "$CURRENT_SOURCE" "$ROLLBACK_SOURCE" "$MUTANT_SOURCE"
cp "$MIGRATIONS"/[0-9][0-9][0-9][0-9]_*.sql "$CURRENT_SOURCE/"
cp "$MIGRATIONS"/[0-9][0-9][0-9][0-9]_*.sql "$ROLLBACK_SOURCE/"
cp "$MIGRATIONS"/[0-9][0-9][0-9][0-9]_*.sql "$MUTANT_SOURCE/"
test -f "$CURRENT_SOURCE/0031_doc_source_ref_v2_aliases.sql" && test ! -L "$CURRENT_SOURCE/0031_doc_source_ref_v2_aliases.sql" || fail "current-source 0031 removal target invalid"
/bin/rm -- "$CURRENT_SOURCE/0031_doc_source_ref_v2_aliases.sql"
printf '\n-- Watched rollback mutant: must fail after transactional DDL.\nSELECT * FROM kengram_acceptance_force_rollback;\n' >>"$ROLLBACK_SOURCE/0035_argus_source_events_resolved_status.sql"

PREIMAGE="-- Allow terminal resolution of payload_hash_conflict without deleting quarantine history."
POSTIMAGE="-- Allow terminal resolution of payload_hash_conflict without deleting quarantine history (checksum mutant)."
awk -v pre="$PREIMAGE" -v post="$POSTIMAGE" '
  $0 == pre { count += 1; print post; next }
  { print }
  END { if (count != 1) exit 7 }
' "$MUTANT_SOURCE/0035_argus_source_events_resolved_status.sql" >"$WORK/0035.mutated" || fail "checksum mutant preimage count was not one"
mv "$WORK/0035.mutated" "$MUTANT_SOURCE/0035_argus_source_events_resolved_status.sql"
test "$(grep -Fxc -- "$POSTIMAGE" "$MUTANT_SOURCE/0035_argus_source_events_resolved_status.sql")" = 1 || fail "checksum mutant postimage count was not one"
test "$(shasum -a 384 "$MUTANT_SOURCE/0035_argus_source_events_resolved_status.sql" | awk '{print $1}')" != "$SHA384_35" || fail "checksum mutant did not change SHA-384"

CONTAINER_ID="$(docker run --rm -d \
  --name "$CONTAINER" \
  --label "io.yetiwerks.kengram-0035-accept=$CONTAINER" \
  -e POSTGRES_USER=kengram_accept \
  -e POSTGRES_PASSWORD=acceptance-only \
  -e POSTGRES_DB=kengram_accept \
  -p 127.0.0.1::5432 \
  "$IMAGE")" || fail "disposable PostgreSQL container failed to start"
case "$CONTAINER_ID" in
  [0-9a-f][0-9a-f]*) ;;
  *) fail "invalid disposable container id" ;;
esac

ready=0
attempt=0
while test "$attempt" -lt 30; do
  if docker exec "$CONTAINER" pg_isready -U kengram_accept -d kengram_accept >/dev/null 2>&1; then
    ready=1
    break
  fi
  attempt=$((attempt + 1))
  sleep 1
done
test "$ready" -eq 1 || fail "disposable PostgreSQL did not become ready"

PORT_LINE="$(docker port "$CONTAINER" 5432/tcp)" || fail "disposable port lookup failed"
PORT="${PORT_LINE##*:}"
case "$PORT" in
  ''|*[!0-9]*) fail "invalid disposable PostgreSQL port: $PORT" ;;
esac
DATABASE_URL="postgres://kengram_accept:acceptance-only@127.0.0.1:${PORT}/kengram_accept"
case "$DATABASE_URL" in
  *kengram_prod*) fail "production database target refused" ;;
esac
export DATABASE_URL

docker exec -i "$CONTAINER" psql -X -v ON_ERROR_STOP=1 -U kengram_accept -d kengram_accept >/dev/null <<'SQL'
CREATE TABLE _sqlx_migrations (
    version BIGINT PRIMARY KEY,
    description TEXT NOT NULL,
    installed_on TIMESTAMPTZ NOT NULL DEFAULT now(),
    success BOOLEAN NOT NULL,
    checksum BYTEA NOT NULL,
    execution_time BIGINT NOT NULL
);

CREATE TABLE argus_source_events (
    id UUID PRIMARY KEY,
    namespace TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    thought_id UUID,
    status TEXT NOT NULL,
    error TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL,
    CONSTRAINT argus_source_events_status_check CHECK (
        status = ANY (ARRAY['pending'::text,'stored'::text,'conflict'::text,'dlq'::text,'skipped'::text,'resolved'::text])
    )
);

INSERT INTO argus_source_events (
    id, namespace, source_ref, payload_hash, thought_id, status, error,
    first_seen_at, last_seen_at, metadata
) VALUES (
    '1623cb17-62cc-4159-9afd-9b999e9e792e',
    'conversations/telegram-capture-consumer',
    'telegram-capture:dojo:diesel:8032057844:2026-07-31:165-980003',
    '3cd2746c48b79f9008291d8f2416e2ea37617b1b6f9eb455dd14ed65a90eb5a9',
    '480a74b6-dbed-4d48-a11a-bc532dc6036e',
    'resolved',
    'payload_hash_conflict',
    '2026-07-31 05:09:22.038221+00',
    '2026-07-31 07:14:59.018647+00',
    '{"conflict":{"prior_payload_sha256":"3cd2746c48b79f9008291d8f2416e2ea37617b1b6f9eb455dd14ed65a90eb5a9"},"resolution":{"resolver":"agent:diesel","resolved_at":"2026-07-31T07:21:29Z","receipt":"agents/diesel/state/receipts/kengram-source-conflict-1623cb17-20260731.md"}}'::jsonb
);
SQL

LEDGER_SQL="$WORK/ledger-1-34.sql"
printf 'BEGIN;\n' >"$LEDGER_SQL"
while IFS='|' read -r version checksum; do
  test "$version" -le 34 || continue
  padded="$(printf '%04d' "$version")"
  file="$(find "$MIGRATIONS" -maxdepth 1 -type f -name "${padded}_*.sql" -print)" || fail "migration lookup failed: $version"
  test "$(printf '%s\n' "$file" | awk 'NF { count += 1 } END { print count + 0 }')" = 1 || fail "migration lookup count was not one: $version"
  base="${file##*/}"
  description="${base#"${padded}"_}"
  description="${description%.sql}"
  case "$description" in
    *[!a-z0-9_]*) fail "unsafe migration description: $description" ;;
  esac
  description="${description//_/ }"
  printf "INSERT INTO _sqlx_migrations(version,description,installed_on,success,checksum,execution_time) VALUES (%s,'%s','2026-07-23 06:07:33+00',true,decode('%s','hex'),1);\n" "$version" "$description" "$checksum" >>"$LEDGER_SQL"
done <"$FULL_MANIFEST"
printf 'COMMIT;\n' >>"$LEDGER_SQL"
docker exec -i "$CONTAINER" psql -X -v ON_ERROR_STOP=1 -U kengram_accept -d kengram_accept <"$LEDGER_SQL" >/dev/null || fail "fixture ledger seed failed"

set +e
sqlx migrate run --source "$CURRENT_SOURCE" --no-dotenv --target-version 35 --dry-run >"$WORK/current-red.out" 2>&1
current_rc=$?
set -e
test "$current_rc" -ne 0 || fail "current source unexpectedly accepted missing 0031"
test "$(grep -Fxc 'error: migration 31 was previously applied but is missing in the resolved migrations' "$WORK/current-red.out")" = 1 || fail "current source did not return exact VersionMissing31"
pass_case current-version-missing-31-red

LEDGER_PRE="$WORK/ledger.pre"
EXPECTED_PRE="$WORK/source-1-34.manifest"
ledger_manifest "$LEDGER_PRE"
awk -F '|' '$1 <= 34 { print }' "$FULL_MANIFEST" >"$EXPECTED_PRE" || fail "preapply expected-manifest producer failed"
cmp -s "$EXPECTED_PRE" "$LEDGER_PRE" || fail "preapply source/ledger checksum equality failed"
test "$(psql_query "SELECT count(*) FROM _sqlx_migrations WHERE NOT success")" = 0 || fail "fixture has failed ledger rows"
sqlx migrate info --source "$MIGRATIONS" --no-dotenv >"$WORK/reconciled.info" 2>&1 || fail "reconciled migrate info failed"
test "$(grep -Fxc '35/pending argus source events resolved status' "$WORK/reconciled.info")" = 1 || fail "reconciled source did not report exact pending35"
sqlx migrate run --source "$MIGRATIONS" --no-dotenv --target-version 35 --dry-run >"$WORK/reconciled-dry-run.out" 2>&1 || fail "reconciled dry-run failed"
test "$(grep -Fxc 'Can apply 35/migrate argus source events resolved status (0ns)' "$WORK/reconciled-dry-run.out")" = 1 || fail "reconciled dry-run marker missing"
test "$(psql_query "SELECT count(*)::text || '|' || max(version)::text || '|' || count(*) FILTER (WHERE version=35)::text FROM _sqlx_migrations WHERE success")" = '34|34|0' || fail "dry-run changed ledger"
pass_case reconciled-pending-35

capture_state "$WORK/rollback.before"
set +e
sqlx migrate run --source "$ROLLBACK_SOURCE" --no-dotenv --target-version 35 >"$WORK/rollback.out" 2>&1
rollback_rc=$?
set -e
test "$rollback_rc" -ne 0 || fail "rollback mutant unexpectedly succeeded"
grep -Fq 'kengram_acceptance_force_rollback' "$WORK/rollback.out" || fail "rollback mutant did not reach named failing statement"
capture_state "$WORK/rollback.after"
assert_state_equal "$WORK/rollback.before" "$WORK/rollback.after"
test "$(psql_query "SELECT count(*) FROM _sqlx_migrations WHERE version=35")" = 0 || fail "rollback mutant left ledger35"
pass_case transactional-rollback

cp "$WORK/rollback.before.source-row" "$WORK/preapply.source-row"
sqlx migrate run --source "$MIGRATIONS" --no-dotenv --target-version 35 >"$WORK/apply.out" 2>&1 || fail "exact migration35 apply failed"
capture_state "$WORK/apply.after"
cmp -s "$WORK/preapply.source-row" "$WORK/apply.after.source-row" || fail "exact apply changed source row"
actual_constraint="$(psql_query "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid='argus_source_events'::regclass AND conname='argus_source_events_status_check'")"
actual_comment="$(psql_query "SELECT COALESCE(obj_description(oid, 'pg_constraint'), '<NULL>') FROM pg_constraint WHERE conrelid='argus_source_events'::regclass AND conname='argus_source_events_status_check'")"
test "$actual_constraint" = "$EXPECTED_DEF" || fail "postapply constraint definition mismatch"
test "$actual_comment" = "$EXPECTED_COMMENT" || fail "postapply constraint comment mismatch"
ledger35_identity="$(psql_query "SELECT version::text || '|' || description || '|' || success::text || '|' || encode(checksum,'hex') FROM _sqlx_migrations WHERE version=35")"
test "$ledger35_identity" = "35|argus source events resolved status|true|${SHA384_35}" || fail "postapply ledger35 identity mismatch"
test "$(psql_query "SELECT count(*) FROM argus_source_events WHERE status='conflict'")" = 0 || fail "postapply conflict gauge is nonzero"
pass_case exact-apply-a1

LEDGER_POST="$WORK/ledger.post"
ledger_manifest "$LEDGER_POST"
cmp -s "$FULL_MANIFEST" "$LEDGER_POST" || fail "postapply per-version source/ledger equality failed"
test "$(psql_query "SELECT count(*) FROM _sqlx_migrations WHERE NOT success")" = 0 || fail "postapply failed ledger row exists"
sqlx migrate info --source "$MIGRATIONS" --no-dotenv >"$WORK/postapply.info" 2>&1 || fail "postapply migrate info failed"
test "$(grep -Fxc '35/installed argus source events resolved status' "$WORK/postapply.info")" = 1 || fail "postapply installed35 marker missing"
pass_case postapply-manifest-equality

capture_state "$WORK/noop.before"
sqlx migrate run --source "$MIGRATIONS" --no-dotenv --target-version 35 >"$WORK/noop.out" 2>&1 || fail "second exact apply failed"
capture_state "$WORK/noop.after"
assert_state_equal "$WORK/noop.before" "$WORK/noop.after"
pass_case second-apply-identity-noop

capture_state "$WORK/mismatch.before"
set +e
sqlx migrate run --source "$MUTANT_SOURCE" --no-dotenv --target-version 35 >"$WORK/mismatch.out" 2>&1
mismatch_rc=$?
set -e
test "$mismatch_rc" -ne 0 || fail "checksum mutant unexpectedly succeeded"
test "$(grep -Fxc 'error: migration 35 was previously applied but has been modified' "$WORK/mismatch.out")" = 1 || fail "checksum mutant did not return exact VersionMismatch35"
capture_state "$WORK/mismatch.after"
assert_state_equal "$WORK/mismatch.before" "$WORK/mismatch.after"
pass_case checksum-version-35-refusal

test "$EXECUTED" -eq "$SELECTED" || fail "selected/executed mismatch"
printf 'PASS kengram-migration-0035-reconciliation selected=%s executed=%s failed=0 skipped=0\n' "$SELECTED" "$EXECUTED"

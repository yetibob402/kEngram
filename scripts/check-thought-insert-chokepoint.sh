#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <kengram-root> <argus-bin-dir> <argus-lib-dir>" >&2
  exit 64
fi

kengram_root=$1
argus_bin=$2
argus_lib=$3
argus_root=$(cd "$argus_bin/.." && pwd)

for path in "$kengram_root" "$argus_bin" "$argus_lib"; do
  [[ -d "$path" ]] || { echo "missing scan root: $path" >&2; exit 2; }
done

matches=$(rg -n -U -i \
  --glob '*.rs' --glob '*.js' --glob '*.py' \
  --glob '!tests/**' --glob '!target/**' --glob '!migrations/**' \
  '(^|[^A-Za-z_])(insert[[:space:]]+into|copy)[[:space:]]+(public[.])?thoughts\b' \
  "$kengram_root/crates" "$argus_bin" "$argus_lib" || true)
if [[ -n "$matches" ]]; then
  echo "direct application thought write bypasses capture_thought_gated:" >&2
  echo "$matches" >&2
  exit 1
fi

declare -a writer_contracts=(
  "$kengram_root/crates/kengram-storage/src/lib.rs|capture_thought_gated"
  "$kengram_root/crates/kengram-mcp/src/capture.rs|capture_thought_gated"
  "$argus_lib/argus-session-kengram-adapter.js|captureWithPsql"
  "$argus_lib/argus-telegram-kengram-adapter.js|captureWithPsql"
  "$argus_bin/argus-kengram-consumer.js|captureWithPsql"
  "$argus_bin/argus-telegram-kengram-tailer.js|captureWithPsql"
  "$argus_bin/argus-reviews-kengram-ingest.py|capture_call_from_expressions"
  "$argus_bin/argus-docs-kengram-ingest.py|capture_call_from_expressions"
  "$argus_bin/argus-openclaw-memory-import.py|capture_call_from_expressions"
  "$argus_bin/argus-kengram-hive-bulk-import.py|capture_call_from_expressions"
  "$argus_bin/argus-mba-archive-ingest.py|capture_call_from_expressions"
  "$argus_bin/argus-kengram-phase4-promote-current-truth.py|capture_call_from_expressions"
  "$argus_bin/argus-session-kengram-ingest-live-once.py|capture_call_from_expressions"
)

for contract in "${writer_contracts[@]}"; do
  file=${contract%%|*}
  token=${contract#*|}
  [[ -f "$file" ]] || { echo "missing accepted writer: $file" >&2; exit 3; }
  rg -q "$token" "$file" || {
    echo "writer is not routed through the gate: $file (missing $token)" >&2
    exit 4
  }
done

[[ ${#writer_contracts[@]} -eq 13 ]] || {
  echo "internal writer manifest error: expected 13" >&2
  exit 5
}

rg -q 'FROM capture_thought_gated|capture_thought_gated\(' \
  "$argus_lib/argus-kengram-gated-writer.js" \
  "$argus_lib/argus_kengram_gated_writer.py"

echo "PASS thought insert chokepoint: 13/13 writers gated ($argus_root)"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <kengram-root> <argus-bin-dir> <argus-lib-dir>" >&2
  exit 64
fi

kengram_root=$1
argus_bin=$2
argus_lib=$3

matches=$(rg -n -U -i \
  --glob '*.rs' --glob '*.js' --glob '*.py' \
  --glob '!tests/**' --glob '!target/**' --glob '!migrations/**' \
  '(^|[^A-Za-z_])((insert[[:space:]]+into|update|delete[[:space:]]+from|copy)[[:space:]]+(public[.])?thought_links\b|update[[:space:]]+(public[.])?thoughts[[:space:][:print:]]{0,240}(retracted_at|retracted_reason))' \
  "$kengram_root/crates" "$argus_bin" "$argus_lib" || true)
if [[ -n "$matches" ]]; then
  echo "direct application relation/retraction mutation bypasses serialized functions:" >&2
  echo "$matches" >&2
  exit 1
fi

rg -q 'mutate_thought_relations_serialized' "$kengram_root/crates/kengram-mcp/src/link.rs"
rg -q 'mutate_thought_relations_serialized' "$kengram_root/crates/kengram-mcp/src/drain.rs"
rg -q 'retract_thought_serialized' "$kengram_root/crates/kengram-storage/src/lib.rs"
rg -q 'relation_mutation_call_from_expressions' "$argus_bin/argus-kengram-hive-bulk-import.py"
rg -q 'p_relation_intents' "$argus_lib/argus_kengram_gated_writer.py"

echo "PASS relation/retraction chokepoint: serialized mutation and shared endpoint lock callers present"

#!/usr/bin/env bash
# Cross-model tagger-eval sweep. Runs the `tagger_eval` example over one or more
# fixture files against each model in turn, so tag quality can be compared
# across the 12B->397B range (the v14 goal: relatively-quality data on all of
# them, with deterministic filters holding the floor where small models can't).
# No new deps — just the existing example + env vars + jq for the summary.
#
# Usage:
#   ./tagger-sweep.sh [fixture.json ...]
#     (defaults to the forward_looking + use_mention fixtures)
#
# Env:
#   SWEEP_MODELS    space-separated model list
#                   (default: a local gemma3:12b / qwen3.6:27b / qwen3-coder:30b range)
#   OLLAMA_ENDPOINT OpenAI-compatible /v1 base (default: http://localhost:11434/v1)
#   TAGGER_API_KEY  bearer token for an authenticated cloud endpoint (optional)
#   TAGGER_TIMEOUT_SECONDS  per-request timeout (default 180); bump for slow
#                   models — a cold 27B+ can otherwise time out every fixture
#
# Cloud arm example (a hosted model via OpenRouter):
#   OLLAMA_ENDPOINT=https://openrouter.ai/api/v1 TAGGER_API_KEY=sk-... \
#     SWEEP_MODELS="qwen/qwen-2.5-72b-instruct" ./tagger-sweep.sh
set -euo pipefail

cd "$(dirname "$0")"

FIXTURES=("$@")
if [[ ${#FIXTURES[@]} -eq 0 ]]; then
  FIXTURES=(
    crates/kengram-extract/tests/fixtures/forward_looking.json
    crates/kengram-extract/tests/fixtures/use_mention.json
  )
fi

: "${OLLAMA_ENDPOINT:=http://localhost:11434/v1}"
read -r -a MODELS <<< "${SWEEP_MODELS:-gemma3:12b qwen3.6:27b qwen3-coder:30b}"
export OLLAMA_ENDPOINT

# Build once so per-model timing reflects inference, not compilation.
cargo build -q --example tagger_eval

for model in "${MODELS[@]}"; do
  for fx in "${FIXTURES[@]}"; do
    # tagger_eval exits 1 when any fixture fails — that's expected sweep data,
    # not a script error, so capture stdout and tolerate the non-zero exit
    # rather than letting `set -e`/pipefail abort the whole sweep.
    out="$(TAGGER_MODEL="$model" cargo run -q --example tagger_eval -- --json "$fx" || true)"
    if [[ -n "$out" ]]; then
      printf '%s\n' "$out" | jq -c --arg model "$model" --arg fx "$(basename "$fx")" \
        '{model: $model, fixtures: $fx, passed, failed, total,
          failed_names: [.results[] | select(.passed == false) | .name]}'
    else
      echo "{\"model\":\"$model\",\"fixtures\":\"$(basename "$fx")\",\"error\":\"no output (model unreachable?)\"}"
    fi
  done
done

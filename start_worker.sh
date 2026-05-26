#!/usr/bin/env bash
# Run the Kengram worker (foreground): drains pending_embeddings, and — when a
# tagger provider is configured — pending_tags. Run ./start_stack.sh first and
# ./start_server.sh in another terminal.
#
# Config precedence (later wins): built-in defaults < ~/.config/kengram/kengram.toml
# < KENGRAM_* env. This script honors your config: if you have a kengram.toml
# (or have set KENGRAM_TAGGER__* yourself), it is used as-is — the script injects
# nothing. ONLY when no tagger config is present does it enable a zero-config
# default — tagging via local Ollama (qwen2.5:7b-instruct on :11434) — so a fresh
# checkout tags out of the box.
#
#   ./start_worker.sh            # honor kengram.toml; else default to Ollama tagging
#   ./start_worker.sh off        # force embed-only (no tagging) this run
#   KENGRAM_TAGGER__MODEL_NAME=… ./start_worker.sh   # one-off override (wins over the file)
set -euo pipefail

# Force embed-only for this run, overriding whatever the config says. An empty
# provider is the silent-disable sentinel — the tag drainer never spawns.
if [[ "${1:-}" == "off" ]]; then
  exec env KENGRAM_TAGGER__PROVIDER="" cargo run --bin kengram -- worker
fi

# Honor the operator's config. If a config file exists, or a tagger provider is
# already set in the environment, inject nothing — let kengram resolve [tagger]
# (and [database]) from the file / env / built-in defaults on its own.
if [[ -f "${HOME}/.config/kengram/kengram.toml" || -n "${KENGRAM_TAGGER__PROVIDER:-}" ]]; then
  exec cargo run --bin kengram -- worker
fi

# Zero-config fallback: no config file and no tagger provider set, so enable
# tagging via local Ollama out of the box. Any individual value is still
# overridable from the environment.
exec env \
  KENGRAM_TAGGER__PROVIDER=openai-compatible \
  KENGRAM_TAGGER__ENDPOINT="${KENGRAM_TAGGER__ENDPOINT:-http://localhost:11434/v1}" \
  KENGRAM_TAGGER__MODEL_NAME="${KENGRAM_TAGGER__MODEL_NAME:-qwen2.5:7b-instruct}" \
  KENGRAM_TAGGER__MODEL_ID="${KENGRAM_TAGGER__MODEL_ID:-ollama/qwen2.5:7b-instruct}" \
  cargo run --bin kengram -- worker

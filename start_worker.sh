#!/usr/bin/env bash
# Run the Kengram worker (foreground): drains pending_embeddings, and — when a
# tagger provider is configured — pending_tags. Run ./start_stack.sh first and
# ./start_server.sh in another terminal.
#
# Tagging defaults to ON via local Ollama (qwen2.5:7b-instruct on :11434), the
# instruction-tuned model the bundled tagger prompt was iterated against. Every
# value below is overridable from the environment; pass `off` as the first arg
# to run an embed-only worker (no tagging).
#
#   ./start_worker.sh            # embed + tag (Ollama qwen2.5:7b-instruct)
#   ./start_worker.sh off        # embed only, no tagger
#   KENGRAM_TAGGER__MODEL_NAME=qwen3-coder:30b ./start_worker.sh   # override model
set -euo pipefail

DB_URL='postgres://kengram:kengram@localhost:5432/kengram'

# Tagger config. The [tagger] section's REAL keys (config.rs): a non-empty
# provider is what ENABLES tagging — there is no separate `enabled` knob, and
# there is no [reflector]/[extractor] section (renamed to [tagger] at M4).
# "openai-compatible" covers Ollama's /v1 endpoint. Override any of these from
# the environment; the defaults below match the config-file defaults.
TAGGER_PROVIDER="${KENGRAM_TAGGER__PROVIDER:-openai-compatible}"
TAGGER_ENDPOINT="${KENGRAM_TAGGER__ENDPOINT:-http://localhost:11434/v1}"
TAGGER_MODEL_NAME="${KENGRAM_TAGGER__MODEL_NAME:-qwen2.5:7b-instruct}"
TAGGER_MODEL_ID="${KENGRAM_TAGGER__MODEL_ID:-ollama/qwen2.5:7b-instruct}"
TAGGER_TIMEOUT="${KENGRAM_TAGGER__TIMEOUT_SECONDS:-60}"

# `off` → embed-only: leave provider empty so the tag drainer never spawns
# (silent-disable sentinel, config.rs:302).
if [[ "${1:-}" == "off" ]]; then
  TAGGER_PROVIDER=""
fi

# KENGRAM_DATABASE__URL is what the running binary reads; DATABASE_URL is what
# the build-time sqlx::query! macros / sqlx-cli read. Set both from one value.
KENGRAM_DATABASE__URL="$DB_URL" \
DATABASE_URL="$DB_URL" \
KENGRAM_TAGGER__PROVIDER="$TAGGER_PROVIDER" \
KENGRAM_TAGGER__ENDPOINT="$TAGGER_ENDPOINT" \
KENGRAM_TAGGER__MODEL_NAME="$TAGGER_MODEL_NAME" \
KENGRAM_TAGGER__MODEL_ID="$TAGGER_MODEL_ID" \
KENGRAM_TAGGER__TIMEOUT_SECONDS="$TAGGER_TIMEOUT" \
  cargo run --bin kengram -- worker

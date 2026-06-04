#!/usr/bin/env bash
# Bring up kEngram's backing containers and wait until Postgres accepts
# connections. Postgres holds the corpus; the TEI container is the reranker
# ([reranker] provider = "tei", :8080); the ollama-embed container is the
# embedder ([embedder] points at :11435), kept off the host GPU so it never
# competes with the tagger model on the host Ollama (:11434). Tagging still
# runs via host Ollama. Pass --tagger to also start the opt-in deterministic
# tagger sidecar.
set -euo pipefail

if [[ "${1:-}" == "--tagger" ]]; then
  docker compose --profile tagger up -d postgres tei ollama-embed tagger-deterministic
else
  docker compose up -d postgres tei ollama-embed
fi

printf 'waiting for kengram-postgres'
until docker exec kengram-postgres pg_isready -U kengram >/dev/null 2>&1; do
  printf '.'; sleep 1
done
printf ' ready\n'

printf 'waiting for kengram-ollama-embed'
until docker exec kengram-ollama-embed ollama list >/dev/null 2>&1; do
  printf '.'; sleep 1
done
printf ' ready\n'

if ! docker exec kengram-ollama-embed ollama list | awk 'NR>1 {print $1}' | grep -q '^bge-m3'; then
  echo "pulling bge-m3 into kengram-ollama-embed (one-time, ~1.3GB)…"
  docker exec kengram-ollama-embed ollama pull bge-m3
fi

# TEI warms its model on first boot (~60-90s on Apple Silicon CPU). serve/worker
# can start before then — only reranked search waits on it.
echo "note: kengram-tei warms up in the background; rerank is ready once its healthcheck passes."

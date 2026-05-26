 KENGRAM_REFLECTOR__ENABLED=true \
  KENGRAM_EXTRACTOR__PROVIDER=openai-compatible \
  KENGRAM_EXTRACTOR__ENDPOINT='http://localhost:11434/v1' \
  KENGRAM_EXTRACTOR__MODEL_NAME='qwen3-coder:30b' \
  KENGRAM_EXTRACTOR__MODEL_ID='ollama/qwen3-coder:30b' \
  KENGRAM_EXTRACTOR__TIMEOUT_SECONDS=180 \
  DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram' \
    cargo run --bin kengram -- worker


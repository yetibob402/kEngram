# kEngram MLX Reranker

Native Apple Metal reranker service for kEngram. It uses the same
`cross-encoder/ms-marco-MiniLM-L-6-v2` weights as the CPU TEI reranker and
serves TEI-compatible `POST /rerank`.

Default service URL:

```text
http://127.0.0.1:8097/rerank
```

Run locally with the spike venv:

```sh
/Users/yetibob/argus/neo/artifacts/kengram-mlx-rerank-spike-20260624/.venv/bin/python \
  services/mlx-reranker/mlx_minilm_reranker_service.py --host 127.0.0.1 --port 8097
```

Point kEngram at it by keeping `provider = "tei"` and changing only the
reranker endpoint:

```toml
[reranker]
provider = "tei"
endpoint = "http://127.0.0.1:8097"
model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
timeout_seconds = 30
```

The existing kEngram search path soft-fails reranker errors to RRF + recency
with `query_errors=0`; TEI can remain running as the manual rollback endpoint.


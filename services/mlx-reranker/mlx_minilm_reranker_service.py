#!/usr/bin/env python3
"""TEI-compatible MLX/Metal reranker service for kEngram.

This wraps the same MiniLM cross-encoder weights used by the CPU TEI sidecar
and exposes the subset of TEI's /rerank API that kEngram uses:

    POST /rerank
    {"query": "...", "texts": ["...", "..."], "raw_scores": false}

Response:

    [{"index": 0, "score": 0.95, "text": "..."}, ...]

Scores are sorted descending, as TEI does. The `index` field is always the
input index and is the only field kEngram relies on for mapping scores back to
candidate hits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer


DEFAULT_MODEL_ROOT = Path(
    "/Users/yetibob/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8097
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def resolve_snapshot(model_root: Path) -> Path:
    refs_main = model_root / "refs" / "main"
    if refs_main.exists():
        ref = refs_main.read_text().strip()
        snapshot = model_root / "snapshots" / ref
        if snapshot.exists():
            return snapshot
    snapshots = sorted((model_root / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"no Hugging Face snapshot under {model_root}")
    return snapshots[-1]


@dataclass
class EncodedBatch:
    input_ids: mx.array
    token_type_ids: mx.array
    attention_mask: mx.array


class MlxMiniLmReranker:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512, strategy="longest_first")
        self.weights: dict[str, mx.array] = {}
        with safe_open(str(snapshot / "model.safetensors"), framework="np") as tensors:
            for key in tensors.keys():
                if key == "bert.embeddings.position_ids":
                    continue
                self.weights[key] = mx.array(tensors.get_tensor(key), dtype=mx.float32)

        self.hidden_size = 384
        self.num_heads = 12
        self.head_dim = self.hidden_size // self.num_heads
        self.num_layers = 6
        self.layer_norm_eps = 1e-12

    def encode(self, query: str, texts: list[str]) -> EncodedBatch:
        encoded = [self.tokenizer.encode(query, text) for text in texts]
        max_len = max(len(item.ids) for item in encoded)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for item in encoded:
            pad = max_len - len(item.ids)
            input_ids.append(item.ids + [0] * pad)
            token_type_ids.append(item.type_ids + [0] * pad)
            attention_mask.append(item.attention_mask + [0] * pad)
        return EncodedBatch(
            input_ids=mx.array(np.array(input_ids, dtype=np.int32)),
            token_type_ids=mx.array(np.array(token_type_ids, dtype=np.int32)),
            attention_mask=mx.array(np.array(attention_mask, dtype=np.float32)),
        )

    def weight(self, name: str) -> mx.array:
        return self.weights[name]

    def linear(self, x: mx.array, prefix: str) -> mx.array:
        weight = self.weight(f"{prefix}.weight")
        bias = self.weight(f"{prefix}.bias")
        return mx.matmul(x, mx.transpose(weight)) + bias

    def layer_norm(self, x: mx.array, prefix: str) -> mx.array:
        weight = self.weight(f"{prefix}.weight")
        bias = self.weight(f"{prefix}.bias")
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        return centered * mx.rsqrt(var + self.layer_norm_eps) * weight + bias

    @staticmethod
    def gelu(x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))

    def forward_logits(self, batch: EncodedBatch) -> mx.array:
        bsz, seq_len = batch.input_ids.shape
        positions = mx.broadcast_to(mx.arange(seq_len)[None, :], batch.input_ids.shape)
        hidden = (
            self.weight("bert.embeddings.word_embeddings.weight")[batch.input_ids]
            + self.weight("bert.embeddings.position_embeddings.weight")[positions]
            + self.weight("bert.embeddings.token_type_embeddings.weight")[batch.token_type_ids]
        )
        hidden = self.layer_norm(hidden, "bert.embeddings.LayerNorm")

        attention_mask = (1.0 - batch.attention_mask)[:, None, None, :] * -10000.0

        for layer in range(self.num_layers):
            prefix = f"bert.encoder.layer.{layer}"
            residual = hidden
            q = self.linear(hidden, f"{prefix}.attention.self.query")
            k = self.linear(hidden, f"{prefix}.attention.self.key")
            v = self.linear(hidden, f"{prefix}.attention.self.value")

            q = mx.transpose(q.reshape(bsz, seq_len, self.num_heads, self.head_dim), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(bsz, seq_len, self.num_heads, self.head_dim), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(bsz, seq_len, self.num_heads, self.head_dim), (0, 2, 1, 3))

            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
            probs = mx.softmax(scores + attention_mask, axis=-1)
            context = mx.matmul(probs, v)
            context = mx.transpose(context, (0, 2, 1, 3)).reshape(bsz, seq_len, self.hidden_size)

            attn_out = self.linear(context, f"{prefix}.attention.output.dense")
            hidden = self.layer_norm(residual + attn_out, f"{prefix}.attention.output.LayerNorm")

            residual = hidden
            intermediate = self.gelu(self.linear(hidden, f"{prefix}.intermediate.dense"))
            layer_out = self.linear(intermediate, f"{prefix}.output.dense")
            hidden = self.layer_norm(residual + layer_out, f"{prefix}.output.LayerNorm")

        pooled = mx.tanh(self.linear(hidden[:, 0, :], "bert.pooler.dense"))
        return self.linear(pooled, "classifier")[:, 0]

    def rerank_scores(self, query: str, texts: list[str], *, raw_scores: bool = False) -> np.ndarray:
        batch = self.encode(query, texts)
        logits = self.forward_logits(batch)
        mx.eval(logits)
        scores = np.array(logits, dtype=np.float32)
        return scores if raw_scores else sigmoid(scores)


class RerankerState:
    def __init__(self, model: MlxMiniLmReranker, max_batch_size: int):
        self.model = model
        self.max_batch_size = max_batch_size
        self.started_at = time.time()
        self.lock = threading.Lock()
        self.requests = 0
        self.failures = 0


class Handler(BaseHTTPRequestHandler):
    server_version = "kengram-mlx-reranker/1.0"

    @property
    def state(self) -> RerankerState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/health":
            self.write_json(404, {"ok": False, "error": "not found"})
            return
        self.write_json(
            200,
            {
                "ok": True,
                "model_id": MODEL_ID,
                "snapshot": str(self.state.model.snapshot),
                "device": str(mx.default_device()),
                "max_batch_size": self.state.max_batch_size,
                "pid": os.getpid(),
                "uptime_seconds": round(time.time() - self.state.started_at, 3),
                "requests": self.state.requests,
                "failures": self.state.failures,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/rerank":
            self.write_json(404, {"ok": False, "error": "not found"})
            return
        try:
            payload = self.read_json()
            query = payload.get("query")
            texts = payload.get("texts")
            raw_scores = bool(payload.get("raw_scores", False))
            if not isinstance(query, str) or not query.strip():
                self.write_json(400, {"error": "query must be a non-empty string"})
                return
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                self.write_json(400, {"error": "texts must be a list of strings"})
                return
            if not texts:
                self.write_json(200, [])
                return
            if len(texts) > self.state.max_batch_size:
                self.write_json(
                    413,
                    {
                        "error": "batch too large",
                        "max_batch_size": self.state.max_batch_size,
                        "got": len(texts),
                    },
                )
                return

            started = time.perf_counter()
            with self.state.lock:
                scores = self.state.model.rerank_scores(query, texts, raw_scores=raw_scores)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.state.requests += 1

            items = [
                {"index": idx, "score": float(score), "text": texts[idx]}
                for idx, score in enumerate(scores)
            ]
            items.sort(key=lambda item: item["score"], reverse=True)
            self.write_json(200, items, extra_headers={"X-Rerank-Elapsed-Ms": f"{elapsed_ms:.3f}"})
        except Exception as exc:  # noqa: BLE001
            self.state.failures += 1
            logging.exception("rerank request failed")
            self.write_json(500, {"error": str(exc)})

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("content-length") or "0")
        body = self.rfile.read(content_length)
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed

    def write_json(
        self,
        status: int,
        payload: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)


class RerankerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    state: RerankerState


def warm_model(model: MlxMiniLmReranker) -> None:
    model.rerank_scores("warmup", ["warmup text", "irrelevant text"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    snapshot = resolve_snapshot(args.model_root)
    logging.info("loading MLX MiniLM reranker snapshot=%s device=%s", snapshot, mx.default_device())
    model = MlxMiniLmReranker(snapshot)
    warm_model(model)
    logging.info("model warm and resident")

    server = RerankerHTTPServer((args.host, args.port), Handler)
    server.state = RerankerState(model, max_batch_size=args.max_batch_size)

    def shutdown(signum: int, _frame: Any) -> None:
        logging.info("signal %s received; shutting down", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logging.info("serving TEI-compatible /rerank on http://%s:%s", args.host, args.port)
    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

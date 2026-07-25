#!/usr/bin/env python3
"""Gate harness for the kEngram MLX reranker rollout."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id") or not row.get("query"):
                raise ValueError(f"{path}:{line_no}: missing id/query")
            rows.append(row)
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def rank_positions(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 1.0
    ar = [float(x) for x in rank_positions(a)]
    br = [float(x) for x in rank_positions(b)]
    am = statistics.fmean(ar)
    bm = statistics.fmean(br)
    av = [x - am for x in ar]
    bv = [x - bm for x in br]
    denom = math.sqrt(sum(x * x for x in av) * sum(x * x for x in bv))
    if denom == 0.0:
        return 1.0
    return sum(x * y for x, y in zip(av, bv)) / denom


def http_json(url: str, body: dict[str, Any], timeout: float = 120.0) -> tuple[Any, float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload), (time.perf_counter() - started) * 1000.0


def mcp_search(endpoint: str, args: dict[str, Any], request_id: str) -> tuple[dict[str, Any], float]:
    payload, elapsed = http_json(
        endpoint,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "search_thoughts", "arguments": args},
        },
    )
    if "error" in payload:
        raise RuntimeError(json.dumps(payload["error"]))
    content = payload["result"]["content"][0]["text"]
    return json.loads(content), elapsed


def rerank_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/rerank") else f"{endpoint}/rerank"


def rerank_scores(endpoint: str, query: str, texts: list[str], batch_size: int = 32) -> list[float]:
    scores = [0.0] * len(texts)
    url = rerank_url(endpoint)
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        payload, _ = http_json(
            url,
            {"query": query, "texts": batch, "raw_scores": False, "truncate": True},
            timeout=120,
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"{endpoint} returned non-list payload: {payload!r}")
        if len(payload) != len(batch):
            raise RuntimeError(f"{endpoint} returned {len(payload)} scores for {len(batch)} texts")
        for item in payload:
            scores[offset + int(item["index"])] = float(item["score"])
    return scores


def collect_pool(endpoint: str, row: dict[str, Any], pool_size: int, recency_half_life_days: float) -> dict[str, Any]:
    search_args = {
        "query": row["query"],
        "limit": pool_size,
        "candidate_pool": pool_size,
        "rerank": False,
        "recency_half_life_days": recency_half_life_days,
    }
    search_args.update(row.get("search", {}))
    result, elapsed_ms = mcp_search(endpoint, search_args, f"{row['id']}-pool")
    hits = result.get("results", [])
    texts = []
    ids = []
    for hit in hits:
        text = hit.get("chunk_content") or hit.get("content")
        thought_id = hit.get("thought_id")
        if text and thought_id:
            texts.append(text)
            ids.append(thought_id)
    return {
        "id": row["id"],
        "query": row["query"],
        "n": len(texts),
        "texts": texts,
        "thought_ids": ids,
        "pool_elapsed_ms": elapsed_ms,
    }


def parity(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.corpus)
    if args.limit_items:
        rows = rows[: args.limit_items]
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        print(f"[parity {idx}/{len(rows)}] {row['id']}", flush=True)
        pool = collect_pool(args.mcp_endpoint, row, args.pool_size, args.recency_half_life_days)
        if not pool["texts"]:
            raise RuntimeError(f"{row['id']} produced empty raw rerank pool")
        tei = rerank_scores(args.tei_endpoint, row["query"], pool["texts"], args.backend_batch_size)
        mlx = rerank_scores(args.mlx_endpoint, row["query"], pool["texts"], args.backend_batch_size)
        order_tei = sorted(range(len(tei)), key=lambda i: tei[i], reverse=True)
        order_mlx = sorted(range(len(mlx)), key=lambda i: mlx[i], reverse=True)
        deltas = [abs(a - b) for a, b in zip(tei, mlx)]
        out_rows.append(
            {
                "id": row["id"],
                "n": len(tei),
                "pool_elapsed_ms": pool["pool_elapsed_ms"],
                "max_abs_delta": max(deltas),
                "mean_abs_delta": statistics.fmean(deltas),
                "spearman": spearman(tei, mlx),
                "top1_match": int(order_tei[0] == order_mlx[0]),
                "order_match": int(order_tei == order_mlx),
                "tei_top1": order_tei[0],
                "mlx_top1": order_mlx[0],
            }
        )
    return {
        "gate": "parity",
        "generated_at": now(),
        "mcp_endpoint": args.mcp_endpoint,
        "tei_endpoint": args.tei_endpoint,
        "mlx_endpoint": args.mlx_endpoint,
        "pool_size": args.pool_size,
        "rows": out_rows,
        "summary": {
            "rows": len(out_rows),
            "candidates": sum(r["n"] for r in out_rows),
            "max_abs_delta": max(r["max_abs_delta"] for r in out_rows),
            "mean_abs_delta": statistics.fmean(r["mean_abs_delta"] for r in out_rows),
            "mean_spearman": statistics.fmean(r["spearman"] for r in out_rows),
            "min_spearman": min(r["spearman"] for r in out_rows),
            "top1_agreement": sum(r["top1_match"] for r in out_rows) / len(out_rows),
            "order_agreement": sum(r["order_match"] for r in out_rows) / len(out_rows),
        },
    }


def latency(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.corpus)
    if args.limit_items:
        rows = rows[: args.limit_items]
    results = []
    for idx, row in enumerate(rows, start=1):
        print(f"[latency {idx}/{len(rows)}] {row['id']}", flush=True)
        search_args = {
            "query": row["query"],
            "limit": args.limit,
            "candidate_pool": args.pool_size,
            "rerank": True,
            "recency_half_life_days": args.recency_half_life_days,
        }
        search_args.update(row.get("search", {}))
        try:
            result, elapsed_ms = mcp_search(args.mcp_endpoint, search_args, row["id"])
            results.append(
                {
                    "id": row["id"],
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "rerank_used": result.get("rerank_used"),
                    "result_count": len(result.get("results", [])),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"id": row["id"], "ok": False, "elapsed_ms": 0.0, "error": str(exc)})
    elapsed = [r["elapsed_ms"] for r in results if r["ok"]]
    return {
        "gate": "mcp-latency",
        "generated_at": now(),
        "mcp_endpoint": args.mcp_endpoint,
        "pool_size": args.pool_size,
        "rows": results,
        "summary": summarize_elapsed(results, elapsed),
    }


def concurrency(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.corpus)
    if args.limit_items:
        rows = rows[: args.limit_items]

    def one(row: dict[str, Any]) -> dict[str, Any]:
        search_args = {
            "query": row["query"],
            "limit": args.limit,
            "candidate_pool": args.pool_size,
            "rerank": True,
            "recency_half_life_days": args.recency_half_life_days,
        }
        search_args.update(row.get("search", {}))
        try:
            result, elapsed_ms = mcp_search(args.mcp_endpoint, search_args, row["id"])
            return {
                "id": row["id"],
                "ok": True,
                "elapsed_ms": elapsed_ms,
                "rerank_used": result.get("rerank_used"),
                "result_count": len(result.get("results", [])),
            }
        except Exception as exc:  # noqa: BLE001
            return {"id": row["id"], "ok": False, "elapsed_ms": 0.0, "error": str(exc)}

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, row) for row in rows]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = [r["elapsed_ms"] for r in results if r["ok"]]
    report = {
        "gate": "mcp-concurrency",
        "generated_at": now(),
        "mcp_endpoint": args.mcp_endpoint,
        "pool_size": args.pool_size,
        "concurrency": args.concurrency,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "rows": sorted(results, key=lambda r: r["id"]),
        "summary": summarize_elapsed(results, elapsed),
    }
    return report


def fallback(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.corpus)
    if args.limit_items:
        rows = rows[: args.limit_items]

    results: list[dict[str, Any]] = []

    def stopper() -> None:
        time.sleep(args.stop_after_seconds)
        if args.stop_command:
            subprocess.run(args.stop_command, shell=True, check=False)

    stop_thread = None
    if args.stop_command:
        stop_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        stop_thread.submit(stopper)

    for idx, row in enumerate(rows, start=1):
        print(f"[fallback {idx}/{len(rows)}] {row['id']}", flush=True)
        search_args = {
            "query": row["query"],
            "limit": args.limit,
            "candidate_pool": args.pool_size,
            "rerank": True,
            "recency_half_life_days": args.recency_half_life_days,
        }
        search_args.update(row.get("search", {}))
        try:
            result, elapsed_ms = mcp_search(args.mcp_endpoint, search_args, row["id"])
            results.append(
                {
                    "id": row["id"],
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "rerank_used": result.get("rerank_used"),
                    "result_count": len(result.get("results", [])),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"id": row["id"], "ok": False, "elapsed_ms": 0.0, "error": str(exc)})
    if stop_thread:
        stop_thread.shutdown(wait=False)
    elapsed = [r["elapsed_ms"] for r in results if r["ok"]]
    return {
        "gate": "fallback",
        "generated_at": now(),
        "mcp_endpoint": args.mcp_endpoint,
        "pool_size": args.pool_size,
        "stop_command": args.stop_command,
        "rows": results,
        "summary": summarize_elapsed(results, elapsed),
    }


def summarize_elapsed(results: list[dict[str, Any]], elapsed: list[float]) -> dict[str, Any]:
    return {
        "rows": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "errors": sum(1 for r in results if not r["ok"]),
        "rerank_used": sum(1 for r in results if r.get("rerank_used") is True),
        "rerank_fallback": sum(1 for r in results if r.get("rerank_used") is False),
        "latency_ms": {
            "mean": statistics.fmean(elapsed) if elapsed else 0.0,
            "p50": percentile(elapsed, 50),
            "p95": percentile(elapsed, 95),
            "max": max(elapsed) if elapsed else 0.0,
        },
    }


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=["parity", "mcp-latency", "mcp-concurrency", "fallback"])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mcp-endpoint", default="http://127.0.0.1:8088/mcp")
    parser.add_argument("--tei-endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--mlx-endpoint", default="http://127.0.0.1:8097")
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--backend-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--recency-half-life-days", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--stop-command")
    parser.add_argument("--stop-after-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.gate == "parity":
        report = parity(args)
    elif args.gate == "mcp-latency":
        report = latency(args)
    elif args.gate == "mcp-concurrency":
        report = concurrency(args)
    else:
        report = fallback(args)
    write_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

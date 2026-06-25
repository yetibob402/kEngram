#!/usr/bin/env python3
"""Fail if MCP search_thoughts returns eval-contamination denylist rows.

This is a service-level preflight: it calls an MCP endpoint and verifies the
returned candidate IDs, so it catches missing storage/search-path enforcement.
It is not an eval-score filter and must pass before a clean gold baseline is
accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def call_search(endpoint: str, args: dict[str, Any], request_id: str) -> dict[str, Any]:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "search_thoughts", "arguments": args},
        }
    ).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read().decode()
    outer = json.loads(payload)
    if "error" in outer:
        raise RuntimeError(json.dumps(outer["error"]))
    return json.loads(outer["result"]["content"][0]["text"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8088/mcp")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--denylist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--candidate-pool", type=int, default=32)
    ap.add_argument("--rerank", choices=["true", "false"], default="true")
    ap.add_argument("--recency-half-life-days", type=float, default=30.0)
    ap.add_argument(
        "--require-denylist-sha-prefix",
        help="Fail if the denylist file SHA-256 does not start with this prefix.",
    )
    args = ap.parse_args()

    denylist_bytes = args.denylist.read_bytes()
    denylist_sha256 = hashlib.sha256(denylist_bytes).hexdigest()
    denylist_sha_error = None
    if args.require_denylist_sha_prefix and not denylist_sha256.startswith(
        args.require_denylist_sha_prefix
    ):
        denylist_sha_error = (
            f"denylist sha256 {denylist_sha256} does not match required prefix "
            f"{args.require_denylist_sha_prefix}"
        )

    corpus = load_jsonl(args.corpus)
    deny_rows = load_jsonl(args.denylist)
    denied_ids = {row["thought_id"] for row in deny_rows}
    defaults = {
        "limit": args.limit,
        "candidate_pool": args.candidate_pool,
        "rerank": args.rerank == "true",
        "recency_half_life_days": args.recency_half_life_days,
    }

    leaks: list[dict[str, Any]] = []
    query_errors: list[dict[str, str]] = []
    started = time.perf_counter()
    for idx, row in enumerate(corpus, start=1):
        search_args = dict(defaults)
        search_args.update(row.get("search", {}))
        search_args["query"] = row["query"]
        print(f"[{idx}/{len(corpus)}] {row['id']}", flush=True)
        try:
            result = call_search(args.endpoint, search_args, row["id"])
        except Exception as exc:  # noqa: BLE001 - preflight report should keep exact error text
            query_errors.append({"id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        returned = [hit["thought_id"] for hit in result.get("results", [])]
        deny_hits = [tid for tid in returned if tid in denied_ids]
        if deny_hits:
            leaks.append(
                {
                    "id": row["id"],
                    "category": row.get("category"),
                    "query": row["query"],
                    "deny_hits": deny_hits,
                    "returned_ids": returned,
                }
            )

    report = {
        "endpoint": args.endpoint,
        "corpus": str(args.corpus),
        "denylist": str(args.denylist),
        "denylist_sha256": denylist_sha256,
        "required_denylist_sha_prefix": args.require_denylist_sha_prefix,
        "denylist_count": len(denied_ids),
        "queries": len(corpus),
        "leak_count_queries": len(leaks),
        "query_errors": query_errors
        + ([{"id": "denylist_sha", "error": denylist_sha_error}] if denylist_sha_error else []),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "ok": not leaks and not query_errors and denylist_sha_error is None,
        "leaks": leaks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ["ok", "queries", "denylist_count", "leak_count_queries"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

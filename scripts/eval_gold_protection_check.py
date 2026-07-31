#!/usr/bin/env python3
"""Per-question gold-protection check against a retrieval eval results JSON.

Hooks an eval rerun: for each query_id in the gold corpus, verify at least one
gold_thought_id appears in the top-k hits. Flags regressions per-question,
not only via aggregate pass@10.

Usage:
  python3 scripts/eval_gold_protection_check.py --corpus gold100.json --results run.json --k 10
  # exit 1 if any previously-passing gold question fails (with --baseline-results)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_list(path: Path, keys: tuple[str, ...]) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    for k in keys:
        if isinstance(data.get(k), list):
            return data[k]
    raise SystemExit(f"unrecognized shape in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="prior run; only flag regressions vs prior pass",
    )
    args = ap.parse_args()

    corpus = load_list(args.corpus, ("items", "queries", "questions", "gold"))
    gold_map: dict[str, set[str]] = {}
    for it in corpus:
        qid = str(it.get("query_id") or it.get("id") or it.get("kgr_id"))
        golds = it.get("gold_thought_ids") or it.get("gold_ids") or it.get("thought_ids") or []
        if it.get("gold_thought_id"):
            golds = list(golds) + [it["gold_thought_id"]]
        gold_map[qid] = {str(g) for g in golds}

    results = load_list(args.results, ("results", "queries", "items"))
    by_id: dict[str, dict] = {}
    for r in results:
        qid = str(r.get("query_id") or r.get("id") or r.get("kgr_id"))
        hits = r.get("hit_ids") or r.get("thought_ids") or []
        if not hits and r.get("hits"):
            hits = [
                h.get("thought_id") or h.get("id")
                for h in r["hits"]
                if isinstance(h, dict)
            ]
        by_id[qid] = {"hits": [str(h) for h in hits[: args.k] if h]}

    prior_pass: dict[str, bool] = {}
    if args.baseline_results:
        for r in load_list(args.baseline_results, ("results", "queries", "items")):
            qid = str(r.get("query_id") or r.get("id") or r.get("kgr_id"))
            prior_pass[qid] = bool(r.get("pass", True))

    failures = []
    for qid, golds in sorted(gold_map.items()):
        hits = set(by_id.get(qid, {}).get("hits", []))
        ok = bool(golds & hits)
        if args.baseline_results and not prior_pass.get(qid, False):
            continue
        if not ok:
            failures.append({"query_id": qid, "gold": sorted(golds), "hits": sorted(hits)})

    report = {
        "k": args.k,
        "queries": len(gold_map),
        "failures": failures,
        "failed_count": len(failures),
        "pass": len(failures) == 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

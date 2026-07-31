#!/usr/bin/env python3
"""Populate gold_protection_manifest from a smith-adjudicated gold-100 corpus JSON.

ADDITIVE / offline tooling. Does NOT restart the live service.
Requires DATABASE_URL to a non-prod instance unless --sql-only.

Corpus schema (flexible):
  { "items": [ { "query_id": "KGR-001", "query": "...", "gold_thought_ids": ["uuid", ...],
                 "content_hash": "...", "source_scope": "..." }, ... ] }
or a list of such objects.

Usage:
  python3 scripts/populate_gold_protection_manifest.py --corpus path/to/gold100.json --sql-only > populate.sql
  python3 scripts/populate_gold_protection_manifest.py --corpus path/to/gold100.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    for key in ("items", "queries", "questions", "gold"):
        if isinstance(data.get(key), list):
            return data[key]
    raise SystemExit(f"unrecognized corpus shape in {path}")


def normalize(item: dict) -> list[tuple[str, str, str | None, str | None]]:
    qid = item.get("query_id") or item.get("id") or item.get("kgr_id")
    if not qid:
        raise SystemExit(f"item missing query_id: {item!r}")
    golds = item.get("gold_thought_ids") or item.get("gold_ids") or item.get("thought_ids")
    if not golds and item.get("gold_thought_id"):
        golds = [item["gold_thought_id"]]
    if not golds:
        raise SystemExit(f"item {qid} missing gold thought ids")
    ch = item.get("content_hash")
    scope = item.get("source_scope") or item.get("scope")
    return [(str(qid), str(g), ch, scope) for g in golds]


def sql_for_rows(rows: list[tuple[str, str, str | None, str | None]], snapshot_id: str | None) -> str:
    lines = [
        "-- populate gold_protection_manifest (additive; offline apply)",
        "BEGIN;",
    ]
    for qid, gid, ch, scope in rows:
        ch_sql = "NULL" if not ch else "'" + str(ch).replace("'", "''") + "'"
        scope_sql = "NULL" if not scope else "'" + str(scope).replace("'", "''") + "'"
        snap = "NULL" if not snapshot_id else f"'{snapshot_id}'::uuid"
        qid_e = str(qid).replace("'", "''")
        lines.append(
            "INSERT INTO gold_protection_manifest "
            "(snapshot_id, query_id, gold_thought_id, selectors, content_hash, source_scope, active_state) "
            f"VALUES ({snap}, '{qid_e}', '{gid}'::uuid, '{{}}'::jsonb, {ch_sql}, {scope_sql}, 'active') "
            "ON CONFLICT (snapshot_id, query_id, gold_thought_id) DO UPDATE SET "
            "content_hash = EXCLUDED.content_hash, source_scope = EXCLUDED.source_scope, "
            "active_state = EXCLUDED.active_state;"
        )
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--snapshot-id", default=None)
    ap.add_argument("--sql-only", action="store_true")
    args = ap.parse_args()
    items = load_items(args.corpus)
    rows: list[tuple[str, str, str | None, str | None]] = []
    for it in items:
        rows.extend(normalize(it))
    sql = sql_for_rows(rows, args.snapshot_id)
    if args.sql_only:
        sys.stdout.write(sql)
        print(f"-- rows={len(rows)} queries={len({r[0] for r in rows})}", file=sys.stderr)
        return 0
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL required unless --sql-only", file=sys.stderr)
        return 2
    if "prod" in url.lower() and os.environ.get("KENGRAM_ALLOW_PROD_GOLD_WRITE") != "1":
        print(
            "refusing DATABASE_URL that looks like prod; set KENGRAM_ALLOW_PROD_GOLD_WRITE=1 to override",
            file=sys.stderr,
        )
        return 3
    subprocess.run(["psql", url, "-v", "ON_ERROR_STOP=1", "-f", "-"], input=sql, text=True, check=True)
    print(f"applied rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

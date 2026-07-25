#!/usr/bin/env python3
"""Backfill real BGE-M3 sparse lexical weights into sparsevec sidecars.

This is the Stage-3 data-prep producer path. It intentionally does not change
serving behavior. Run it with the FlagEmbedding venv that has `BAAI/bge-m3`
cached locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_DB_URL = "postgres://kengram:kengram@127.0.0.1:5432/kengram_prod"
DEFAULT_DENYLIST = Path(
    "/Users/yetibob/argus/trinity/reports/kengram-phase0-contamination-fence-20260625/eval-contamination-denylist.jsonl"
)
MODEL_ID = "bge-m3:sparse"
MODEL_VERSION = 1
SOURCE_MODEL = "BAAI/bge-m3"
VOCAB_SIZE = 250_002
SPARSEVEC_INDEX_MAX_NONZERO = 1_000
CAP_ALARM_EXIT_CODE = 42
DEFAULT_CAP_ALARM_BATCH_RATE = 0.10
DEFAULT_CAP_ALARM_BATCH_MIN_SIZE = 20
DEFAULT_CAP_ALARM_CUMULATIVE_RATE = 0.02
DEFAULT_CAP_ALARM_CUMULATIVE_MIN_PROCESSED = 200
GENERATOR = "FlagEmbedding.BGEM3FlagModel"
SOURCE_FILE_DENY_REGEX = (
    "kengram-recall-97|kengram-gold|gold100|gold-100|miss-analysis|"
    "label-repair|adjudication|answer-key|retrieval-baseline"
)
CONTENT_DENY_REGEX = "KGR[0-9]{3}"


def psql(db_url: str, sql: str, *, capture: bool = True, check: bool = True) -> str:
    proc = subprocess.run(
        ["psql", db_url, "-X", "-q", "-v", "ON_ERROR_STOP=1", "-t", "-A"],
        input=sql,
        text=True,
        capture_output=capture,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout if capture else ""


def load_denylist(path: Path, require_sha_prefix: str) -> tuple[list[str], str]:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if require_sha_prefix and not sha.startswith(require_sha_prefix):
        raise SystemExit(
            f"denylist sha256 {sha} does not match required prefix {require_sha_prefix}"
        )
    ids: list[str] = []
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        thought_id = row.get("thought_id")
        if thought_id:
            ids.append(str(thought_id))
    return sorted(set(ids)), sha


def uuid_array(ids: list[str]) -> str:
    if not ids:
        return "ARRAY[]::uuid[]"
    quoted = ",".join("'" + i.replace("'", "''") + "'::uuid" for i in ids)
    return f"ARRAY[{quoted}]::uuid[]"


def assert_schema(db_url: str) -> None:
    out = psql(
        db_url,
        """
        SELECT jsonb_build_object(
          'thought_table', to_regclass('public.thought_sparse_embeddings_bge_m3') IS NOT NULL,
          'chunk_table', to_regclass('public.artifact_chunk_sparse_embeddings_bge_m3') IS NOT NULL,
          'sparsevec', EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sparsevec')
        )::text;
        """,
    ).strip()
    status = json.loads(out)
    if not all(status.values()):
        raise SystemExit(f"sparse schema is not ready: {status}")


def fetch_docs(db_url: str, target: str, denied_ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    denied = uuid_array(denied_ids)
    if target == "thoughts":
        sql = f"""
        SELECT jsonb_build_object(
          'target_kind', 'thought',
          'id', t.id,
          'content', t.content,
          'content_fingerprint_hex', encode(t.content_fingerprint, 'hex'),
          'source_content_chars', char_length(t.content)
        )::text
        FROM thoughts t
        LEFT JOIN thought_sparse_embeddings_bge_m3 s
          ON s.thought_id = t.id
         AND s.model_id = '{MODEL_ID}'
         AND s.model_version = {MODEL_VERSION}
        WHERE t.retracted_at IS NULL
          AND s.thought_id IS NULL
          AND t.id <> ALL({denied})
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ '{SOURCE_FILE_DENY_REGEX}'
          AND t.content !~ '{CONTENT_DENY_REGEX}'
        ORDER BY t.created_at ASC, t.id ASC
        {limit_sql};
        """
    elif target == "chunks":
        sql = f"""
        SELECT jsonb_build_object(
          'target_kind', 'artifact_chunk',
          'id', ac.id,
          'content', ac.content,
          'content_fingerprint_hex', encode(ac.content_fingerprint, 'hex'),
          'source_content_chars', char_length(ac.content)
        )::text
        FROM artifact_chunks ac
        JOIN thoughts t ON t.id = ac.source_thought_id
        LEFT JOIN artifact_chunk_sparse_embeddings_bge_m3 s
          ON s.chunk_id = ac.id
         AND s.model_id = '{MODEL_ID}'
         AND s.model_version = {MODEL_VERSION}
        WHERE ac.retracted_at IS NULL
          AND ac.source_thought_id IS NOT NULL
          AND t.retracted_at IS NULL
          AND s.chunk_id IS NULL
          AND t.id <> ALL({denied})
          AND lower(coalesce(t.metadata->>'source_file', '')) !~ '{SOURCE_FILE_DENY_REGEX}'
          AND t.content !~ '{CONTENT_DENY_REGEX}'
          AND ac.content !~ '{CONTENT_DENY_REGEX}'
        ORDER BY ac.created_at ASC, ac.id ASC
        {limit_sql};
        """
    else:
        raise ValueError(target)
    return [json.loads(line) for line in psql(db_url, sql).splitlines() if line.strip()]


def sparsevec_literal(weights: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    merged: dict[int, float] = {}
    for token_id_raw, weight_raw in weights.items():
        token_id = int(token_id_raw)
        weight = float(weight_raw)
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise ValueError(f"token id {token_id} outside vocab size {VOCAB_SIZE}")
        if not math.isfinite(weight):
            raise ValueError(f"non-finite sparse weight for token id {token_id}")
        if weight:
            merged[token_id + 1] = merged.get(token_id + 1, 0.0) + weight
    items = [(idx, weight) for idx, weight in sorted(merged.items()) if weight]
    if not items:
        raise ValueError("producer emitted no nonzero sparse weights")
    original_nonzero_count = len(items)
    cap_applied = original_nonzero_count > SPARSEVEC_INDEX_MAX_NONZERO
    if cap_applied:
        # pgvector sparse HNSW/IVFFlat indexes reject rows above 1000 nnz.
        # Keep the highest-signal SPLADE terms by absolute weight magnitude.
        top_items = sorted(items, key=lambda item: (-abs(item[1]), item[0]))[
            :SPARSEVEC_INDEX_MAX_NONZERO
        ]
        items = sorted(top_items)
    body = ",".join(f"{idx}:{weight:.9g}" for idx, weight in items)
    cap_metadata = {
        "sparse_original_nonzero_count": original_nonzero_count,
        "sparse_capped_nonzero_count": len(items),
        "sparse_cap_applied": cap_applied,
        "sparse_cap_limit": SPARSEVEC_INDEX_MAX_NONZERO,
        "sparse_cap_rule": "top_abs_weight",
        "sparse_dropped_nonzero_count": original_nonzero_count - len(items),
    }
    return f"{{{body}}}/{VOCAB_SIZE}", len(items), cap_metadata


def encode_sparse(
    model: Any, docs: list[dict[str, Any]], batch_size: int
) -> list[tuple[str, int, dict[str, Any]]]:
    texts = [doc["content"] for doc in docs]
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=8192,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return [sparsevec_literal(weights) for weights in out["lexical_weights"]]


def write_batch(
    db_url: str,
    docs: list[dict[str, Any]],
    encoded: list[tuple[str, int, dict[str, Any]]],
    generator_version: str,
    producer_metadata: dict[str, Any],
) -> None:
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        writer = csv.writer(tmp, delimiter="\t", lineterminator="\n")
        for doc, (literal, nonzero_count, cap_metadata) in zip(docs, encoded):
            metadata_json = json.dumps(
                {**producer_metadata, **cap_metadata},
                sort_keys=True,
                separators=(",", ":"),
            )
            writer.writerow(
                [
                    doc["target_kind"],
                    doc["id"],
                    "\\x" + doc["content_fingerprint_hex"],
                    int(doc["source_content_chars"]),
                    nonzero_count,
                    generator_version,
                    metadata_json,
                    literal,
                ]
            )
        tmp.close()
        psql(
            db_url,
            f"""
            CREATE TEMP TABLE tmp_bge_m3_sparsevec_backfill (
              target_kind text,
              target_id uuid,
              content_fingerprint bytea,
              source_content_chars integer,
              nonzero_count integer,
              generator_version text,
              producer_metadata jsonb,
              embedding_text text
            );

            \\copy tmp_bge_m3_sparsevec_backfill FROM '{tmp.name}' WITH (FORMAT csv, DELIMITER E'\\t')

            INSERT INTO thought_sparse_embeddings_bge_m3 (
              thought_id, model_id, model_version, source_model, vocab_size,
              nonzero_count, content_fingerprint, source_content_chars,
              generator, generator_version, producer_metadata, embedding
            )
            SELECT target_id, '{MODEL_ID}', {MODEL_VERSION}, '{SOURCE_MODEL}', {VOCAB_SIZE},
                   nonzero_count, content_fingerprint, source_content_chars,
                   '{GENERATOR}', generator_version, producer_metadata,
                   embedding_text::sparsevec
              FROM tmp_bge_m3_sparsevec_backfill
             WHERE target_kind = 'thought'
            ON CONFLICT (thought_id, model_id, model_version)
            DO UPDATE SET
              nonzero_count = EXCLUDED.nonzero_count,
              content_fingerprint = EXCLUDED.content_fingerprint,
              source_content_chars = EXCLUDED.source_content_chars,
              generator = EXCLUDED.generator,
              generator_version = EXCLUDED.generator_version,
              producer_metadata = EXCLUDED.producer_metadata,
              embedding = EXCLUDED.embedding,
              updated_at = now();

            INSERT INTO artifact_chunk_sparse_embeddings_bge_m3 (
              chunk_id, model_id, model_version, source_model, vocab_size,
              nonzero_count, content_fingerprint, source_content_chars,
              generator, generator_version, producer_metadata, embedding
            )
            SELECT target_id, '{MODEL_ID}', {MODEL_VERSION}, '{SOURCE_MODEL}', {VOCAB_SIZE},
                   nonzero_count, content_fingerprint, source_content_chars,
                   '{GENERATOR}', generator_version, producer_metadata,
                   embedding_text::sparsevec
              FROM tmp_bge_m3_sparsevec_backfill
             WHERE target_kind = 'artifact_chunk'
            ON CONFLICT (chunk_id, model_id, model_version)
            DO UPDATE SET
              nonzero_count = EXCLUDED.nonzero_count,
              content_fingerprint = EXCLUDED.content_fingerprint,
              source_content_chars = EXCLUDED.source_content_chars,
              generator = EXCLUDED.generator,
              generator_version = EXCLUDED.generator_version,
              producer_metadata = EXCLUDED.producer_metadata,
              embedding = EXCLUDED.embedding,
              updated_at = now();
            """,
            capture=True,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def candidate_key(doc: dict[str, Any]) -> tuple[str, str]:
    return str(doc["target_kind"]), str(doc["id"])


def load_candidate_ids(path: Path) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("{"):
            row = json.loads(raw)
            key = str(row["target_kind"]), str(row["id"])
        else:
            parts = raw.replace(",", "\t").split()
            if len(parts) != 2:
                raise SystemExit(
                    f"ids-file line must be JSON or two fields target_kind id: {raw}"
                )
            key = parts[0], parts[1]
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def apply_candidate_ids(
    docs: list[dict[str, Any]], ids_file: Path | None
) -> list[dict[str, Any]]:
    if not ids_file:
        return docs
    requested = load_candidate_ids(ids_file)
    by_key = {candidate_key(doc): doc for doc in docs}
    return [by_key[key] for key in requested if key in by_key]


def dump_candidate_ids(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for doc in docs:
            f.write(
                json.dumps(
                    {"target_kind": doc["target_kind"], "id": doc["id"]},
                    sort_keys=True,
                )
                + "\n"
            )


def default_jsonl_sidecar(progress: Path, suffix: str) -> Path:
    base = progress.name[: -len(progress.suffix)] if progress.suffix else progress.name
    return progress.with_name(f"{base}-{suffix}.jsonl")


def cap_alarm_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_rate_threshold": args.cap_alarm_batch_rate,
        "batch_min_size": args.cap_alarm_batch_min_size,
        "cumulative_rate_threshold": args.cap_alarm_cumulative_rate,
        "cumulative_min_processed": args.cap_alarm_cumulative_min_processed,
        "halt_exit_code": CAP_ALARM_EXIT_CODE,
    }


def capped_row_events(
    batch_no: int,
    docs: list[dict[str, Any]],
    encoded: list[tuple[str, int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for doc, (_, _, metadata) in zip(docs, encoded):
        if not metadata["sparse_cap_applied"]:
            continue
        events.append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "sparse_cap_row",
                "batch_no": batch_no,
                "target_kind": doc["target_kind"],
                "target_id": doc["id"],
                "original_nnz": metadata["sparse_original_nonzero_count"],
                "capped_nnz": metadata["sparse_capped_nonzero_count"],
                "dropped_nnz": metadata["sparse_dropped_nonzero_count"],
                "cap_rule": metadata["sparse_cap_rule"],
                "cap_limit": metadata["sparse_cap_limit"],
            }
        )
    return events


def build_cap_alarm(
    args: argparse.Namespace,
    batch_no: int,
    batch_size: int,
    capped_rows_batch: int,
    processed_before: int,
    capped_rows_before: int,
) -> dict[str, Any] | None:
    processed_after = processed_before + batch_size
    capped_rows_after = capped_rows_before + capped_rows_batch
    batch_rate = capped_rows_batch / batch_size if batch_size else 0.0
    cumulative_rate = capped_rows_after / processed_after if processed_after else 0.0
    reasons: list[str] = []
    if (
        batch_size >= args.cap_alarm_batch_min_size
        and batch_rate > args.cap_alarm_batch_rate
    ):
        reasons.append("batch_cap_rate_exceeded")
    if (
        processed_after >= args.cap_alarm_cumulative_min_processed
        and cumulative_rate > args.cap_alarm_cumulative_rate
    ):
        reasons.append("cumulative_cap_rate_exceeded")
    if not reasons:
        return None
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": "sparse_cap_alarm",
        "status": "halt_before_write",
        "reasons": reasons,
        "batch_no": batch_no,
        "batch_size": batch_size,
        "capped_rows_batch": capped_rows_batch,
        "batch_cap_rate": batch_rate,
        "processed_before": processed_before,
        "processed_after_if_written": processed_after,
        "capped_rows_before": capped_rows_before,
        "capped_rows_after_if_written": capped_rows_after,
        "cumulative_cap_rate_if_written": cumulative_rate,
        "thresholds": cap_alarm_config(args),
    }


def run_alarm_alert_command(command: str | None, alarm: dict[str, Any]) -> dict[str, Any]:
    if not command:
        return {"configured": False}
    payload = json.dumps(alarm, sort_keys=True)
    env = os.environ.copy()
    env["CAP_ALARM_JSON"] = payload
    proc = subprocess.run(
        command,
        input=payload + "\n",
        text=True,
        shell=True,
        env=env,
        capture_output=True,
        check=False,
    )
    return {
        "configured": True,
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=DEFAULT_DB_URL)
    ap.add_argument("--target", choices=["thoughts", "chunks", "both"], default="both")
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many candidates after the target query and before --limit.",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--ids-file",
        type=Path,
        help="Process only target_kind/id pairs listed in this JSONL or two-column file.",
    )
    ap.add_argument(
        "--dump-candidates",
        type=Path,
        help="Write the selected candidates as target_kind/id JSONL before encoding.",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    ap.add_argument("--require-denylist-sha-prefix", default="a9d0fae5")
    ap.add_argument("--progress", type=Path, default=Path("artifacts/bge-m3-sparsevec-backfill-progress.jsonl"))
    ap.add_argument("--cap-events", type=Path)
    ap.add_argument("--cap-alarm-log", type=Path)
    ap.add_argument("--cap-alarm-batch-rate", type=float, default=DEFAULT_CAP_ALARM_BATCH_RATE)
    ap.add_argument("--cap-alarm-batch-min-size", type=int, default=DEFAULT_CAP_ALARM_BATCH_MIN_SIZE)
    ap.add_argument("--cap-alarm-cumulative-rate", type=float, default=DEFAULT_CAP_ALARM_CUMULATIVE_RATE)
    ap.add_argument(
        "--cap-alarm-cumulative-min-processed",
        type=int,
        default=DEFAULT_CAP_ALARM_CUMULATIVE_MIN_PROCESSED,
    )
    ap.add_argument("--cap-alarm-alert-command")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.offset < 0:
        raise SystemExit("offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive when provided")
    if args.ids_file and not args.ids_file.exists():
        raise SystemExit(f"ids-file does not exist: {args.ids_file}")
    if args.cap_alarm_batch_rate < 0 or args.cap_alarm_cumulative_rate < 0:
        raise SystemExit("cap alarm rates must be non-negative")
    if args.cap_alarm_batch_min_size < 1 or args.cap_alarm_cumulative_min_processed < 1:
        raise SystemExit("cap alarm minimum sample sizes must be positive")
    args.cap_events = args.cap_events or default_jsonl_sidecar(args.progress, "cap-events")
    args.cap_alarm_log = args.cap_alarm_log or default_jsonl_sidecar(args.progress, "cap-alarm")

    denied_ids, denylist_sha = load_denylist(args.denylist, args.require_denylist_sha_prefix)
    assert_schema(args.db_url)
    append_jsonl(
        args.cap_alarm_log,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "sparse_cap_alarm_config",
            "status": "armed",
            "thresholds": cap_alarm_config(args),
            "cap_events": str(args.cap_events),
            "progress": str(args.progress),
        },
    )

    targets = ["thoughts", "chunks"] if args.target == "both" else [args.target]
    docs: list[dict[str, Any]] = []
    per_target_limit = args.limit if len(targets) == 1 else None
    for target in targets:
        docs.extend(fetch_docs(args.db_url, target, denied_ids, per_target_limit))
    fetched_docs_count = len(docs)
    docs = apply_candidate_ids(docs, args.ids_file)
    if args.offset:
        docs = docs[args.offset :]
    if args.limit and len(targets) > 1:
        docs = docs[: args.limit]
    if args.dump_candidates:
        dump_candidate_ids(args.dump_candidates, docs)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "candidate_count": len(docs),
                    "fetched_candidate_count": fetched_docs_count,
                    "ids_file": str(args.ids_file) if args.ids_file else None,
                    "offset": args.offset,
                    "limit": args.limit,
                    "dump_candidates": str(args.dump_candidates)
                    if args.dump_candidates
                    else None,
                    "first_ids": [d["id"] for d in docs[:5]],
                }
            )
        )
        return 0

    from FlagEmbedding import BGEM3FlagModel
    import FlagEmbedding

    generator_version = getattr(FlagEmbedding, "__version__", "unknown")
    model = BGEM3FlagModel(SOURCE_MODEL, use_fp16=True, device="mps")
    producer_metadata = {
        "source_model": SOURCE_MODEL,
        "generator": GENERATOR,
        "generator_version": generator_version,
        "device": "mps",
        "denylist_sha256": denylist_sha,
        "ids_file": str(args.ids_file) if args.ids_file else None,
        "candidate_offset": args.offset,
        "candidate_limit": args.limit,
    }

    started = time.time()
    processed = 0
    capped_rows_total = 0
    dropped_nonzero_total = 0
    max_original_nonzero_count = 0
    for batch_no, batch in enumerate(chunks(docs, args.batch_size), start=1):
        t0 = time.time()
        encoded = encode_sparse(model, batch, args.batch_size)
        batch_cap_metadata = [row[2] for row in encoded]
        capped_rows_batch = sum(1 for row in batch_cap_metadata if row["sparse_cap_applied"])
        dropped_nonzero_batch = sum(row["sparse_dropped_nonzero_count"] for row in batch_cap_metadata)
        batch_max_original = max(
            (row["sparse_original_nonzero_count"] for row in batch_cap_metadata),
            default=0,
        )
        for event in capped_row_events(batch_no, batch, encoded):
            append_jsonl(args.cap_events, event)
        alarm = build_cap_alarm(
            args,
            batch_no,
            len(batch),
            capped_rows_batch,
            processed,
            capped_rows_total,
        )
        if alarm:
            alarm["alert"] = run_alarm_alert_command(args.cap_alarm_alert_command, alarm)
            append_jsonl(args.cap_alarm_log, alarm)
            append_jsonl(
                args.progress,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "batch_no": batch_no,
                    "batch_size": len(batch),
                    "processed": processed,
                    "total": len(docs),
                    "fetched_candidate_count": fetched_docs_count,
                    "ids_file": str(args.ids_file) if args.ids_file else None,
                    "candidate_offset": args.offset,
                    "candidate_limit": args.limit,
                    "halted": True,
                    "halt_reason": "sparse_cap_alarm",
                    "alarm_log": str(args.cap_alarm_log),
                    "cap_events": str(args.cap_events),
                    "alarm": alarm,
                },
            )
            print(json.dumps({"ok": False, "halted": True, "alarm": alarm}), file=sys.stderr)
            return CAP_ALARM_EXIT_CODE
        write_batch(args.db_url, batch, encoded, generator_version, producer_metadata)
        capped_rows_total += capped_rows_batch
        dropped_nonzero_total += dropped_nonzero_batch
        max_original_nonzero_count = max(max_original_nonzero_count, batch_max_original)
        processed += len(batch)
        append_jsonl(
            args.progress,
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "batch_no": batch_no,
                "batch_size": len(batch),
                "processed": processed,
                "total": len(docs),
                "fetched_candidate_count": fetched_docs_count,
                "ids_file": str(args.ids_file) if args.ids_file else None,
                "candidate_offset": args.offset,
                "candidate_limit": args.limit,
                "batch_elapsed_s": round(time.time() - t0, 3),
                "rate_per_hour": round(processed / max(1.0, time.time() - started) * 3600, 2),
                "denylist_sha256": denylist_sha,
                "sparse_cap_limit": SPARSEVEC_INDEX_MAX_NONZERO,
                "sparse_cap_rule": "top_abs_weight",
                "capped_rows_batch": capped_rows_batch,
                "capped_rows_cumulative": capped_rows_total,
                "dropped_nonzero_batch": dropped_nonzero_batch,
                "dropped_nonzero_cumulative": dropped_nonzero_total,
                "max_original_nonzero_count_batch": batch_max_original,
                "max_original_nonzero_count_cumulative": max_original_nonzero_count,
                "cap_events": str(args.cap_events),
                "cap_alarm_log": str(args.cap_alarm_log),
                "cap_alarm_thresholds": cap_alarm_config(args),
            },
        )

    cap_fraction = capped_rows_total / processed if processed else 0.0
    print(
        json.dumps(
            {
                "ok": True,
                "processed": processed,
                "total": len(docs),
                "fetched_candidate_count": fetched_docs_count,
                "ids_file": str(args.ids_file) if args.ids_file else None,
                "candidate_offset": args.offset,
                "candidate_limit": args.limit,
                "denylist_sha256": denylist_sha,
                "sparse_cap_limit": SPARSEVEC_INDEX_MAX_NONZERO,
                "sparse_cap_rule": "top_abs_weight",
                "capped_rows": capped_rows_total,
                "cap_fraction": cap_fraction,
                "dropped_nonzero_count": dropped_nonzero_total,
                "max_original_nonzero_count": max_original_nonzero_count,
                "cap_events": str(args.cap_events),
                "cap_alarm_log": str(args.cap_alarm_log),
                "cap_alarm_thresholds": cap_alarm_config(args),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

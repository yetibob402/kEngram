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


def sparsevec_literal(weights: dict[str, Any]) -> tuple[str, int]:
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
    body = ",".join(f"{idx}:{weight:.9g}" for idx, weight in items)
    return f"{{{body}}}/{VOCAB_SIZE}", len(items)


def encode_sparse(model: Any, docs: list[dict[str, Any]], batch_size: int) -> list[tuple[str, int]]:
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
    encoded: list[tuple[str, int]],
    generator_version: str,
    producer_metadata: dict[str, Any],
) -> None:
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        writer = csv.writer(tmp, delimiter="\t", lineterminator="\n")
        metadata_json = json.dumps(producer_metadata, sort_keys=True, separators=(",", ":"))
        for doc, (literal, nonzero_count) in zip(docs, encoded):
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


def append_progress(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=DEFAULT_DB_URL)
    ap.add_argument("--target", choices=["thoughts", "chunks", "both"], default="both")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    ap.add_argument("--require-denylist-sha-prefix", default="a9d0fae5")
    ap.add_argument("--progress", type=Path, default=Path("artifacts/bge-m3-sparsevec-backfill-progress.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    denied_ids, denylist_sha = load_denylist(args.denylist, args.require_denylist_sha_prefix)
    assert_schema(args.db_url)

    targets = ["thoughts", "chunks"] if args.target == "both" else [args.target]
    docs: list[dict[str, Any]] = []
    per_target_limit = args.limit if len(targets) == 1 else None
    for target in targets:
        docs.extend(fetch_docs(args.db_url, target, denied_ids, per_target_limit))
    if args.limit and len(targets) > 1:
        docs = docs[: args.limit]

    if args.dry_run:
        print(json.dumps({"dry_run": True, "candidate_count": len(docs), "first_ids": [d["id"] for d in docs[:5]]}))
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
    }

    started = time.time()
    processed = 0
    for batch_no, batch in enumerate(chunks(docs, args.batch_size), start=1):
        t0 = time.time()
        encoded = encode_sparse(model, batch, args.batch_size)
        write_batch(args.db_url, batch, encoded, generator_version, producer_metadata)
        processed += len(batch)
        append_progress(
            args.progress,
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "batch_no": batch_no,
                "batch_size": len(batch),
                "processed": processed,
                "total": len(docs),
                "batch_elapsed_s": round(time.time() - t0, 3),
                "rate_per_hour": round(processed / max(1.0, time.time() - started) * 3600, 2),
                "denylist_sha256": denylist_sha,
            },
        )

    print(json.dumps({"ok": True, "processed": processed, "total": len(docs), "denylist_sha256": denylist_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# M6 — Artifacts (SUPERSEDED)

> **Status:** dropped 2026-05-17. The original M6 (artifacts) was reshaped after a live-corpus measurement and dogfood conversation; kengram occupies a high-signal-density "sweet spot" between transcripts and tags, and storing arbitrary long-form documents would dilute that. M5.2's `to_url` link target already covers the "this thought references that external doc" case.
>
> The new M6 ships corpus stats CLI + tagger-extracted relations. See `m6-stats-and-tagger-relations.md`.
>
> This file is kept as historical reference for the original artifacts design and is no longer the active milestone plan.

---

## (Original) M5 — Artifacts

## Goal

Kengram ingests long-form content — PDFs, transcripts, code files, web pages — and unifies it with thoughts in a single search index. The operator can point kengram at existing material (past notes, agent transcripts, project documents) and have it become part of the same memory store thoughts live in.

This is the milestone that turns kengram from "memory of agent interactions" into "memory of everything the operator wants kengram to remember."

## In scope

- Chunking strategy: tokenizer-based chunks of ~500 tokens with configurable overlap. Configurable via `[chunker]` section in TOML.
- `ingest_artifact(uri, kind, scope?, metadata?)` MCP tool. Async; returns `artifact_id` and a `status: "queued"` indicator. Worker picks up, fetches, chunks, embeds, persists.
- Fetchers for `file://`, `https://` (with content-type sniffing for HTML / PDF / plain-text), and a `raw_text` path for inline content the caller has already loaded.
- `artifact_chunks` populated with embeddings on the same `model_id` as thoughts (the existing HNSW partial index covers them — no schema change needed).
- `search_thoughts` extended to "search across thoughts AND artifact_chunks." Both target kinds fused under one RRF ranking. Response shape gains a `target_kind` field per result.
- New CLI subcommand: `kengram ingest <uri> [--kind ...] [--scope ...]` for ad-hoc shell-driven ingestion.

## Out of scope (deferred to which milestone)

- Audio / video transcription — operator runs an external pipeline and ingests the resulting transcript. Indefinitely.
- Image OCR. Indefinitely.
- Web crawling beyond a single page. Indefinitely.
- The facts pipeline was removed in M4 (collapse to thoughts-only); artifact chunks share the thoughts retrieval surface — they don't need a separate facts surface.
- Auth, observability, eval suite, `stats` MCP tool → **M6**

## Schema impact

No new tables. The existing `artifacts` and `artifact_chunks` (shipped empty in M1) are now populated. The `embeddings` table now contains `target_kind = 'artifact_chunk'` rows; the existing HNSW partial index covers them because the predicate is on `model_id`, not `target_kind`.

If the chunker needs persistent state (e.g. tokenizer-version tracking), that may motivate an `artifact_chunks.chunker_version` column added in the M5 artifacts migration. To be decided in M5 planning.

## MCP surface delta

- `ingest_artifact(uri: string, kind: string, scope?: string, metadata?: object) -> { artifact_id: uuid, status: "queued" }`. Returns immediately; worker handles the fetch + chunk + embed pipeline.
- `search_thoughts` results now include items where `target_kind = 'artifact_chunk'`. Response shape gains `target_kind: "thought" | "artifact_chunk"` and, for chunks, a `chunk_index` and `artifact_id` so callers can resolve back to the parent artifact.

## Crate structure delta

- **`kengram-core`** gains a chunking module: `Chunker` trait, fixed-token impl (using `tiktoken-rs` or comparable), sentence-aware impl optional.
- **`kengram-storage`** gains repository functions for artifacts and chunks.
- **`kengram-embed`** is unchanged in shape but exercises the `Embedder` against chunk content (which is functionally identical to thought content).
- **`kengram-cli`** gains the `ingest` subcommand and the worker now drains an artifact-ingestion job in addition to the embedding queue from M2.
- **No new crate.**

## Dependencies

- **Prior milestones:** M1 (storage, embedder, search), M2 (worker process for async ingestion).
- **External services:** none new; TEI continues to provide embeddings.
- **Optional external services:** for HTTPS fetching, the operator may want a configurable user-agent and timeout policy. No external fetcher service required.

## Success criteria

1. **Round-trip:** ingest a 50-page PDF and a 2-hour transcript. Within reasonable wall-clock time (e.g. < 5 minutes on the BOM hardware), `search_thoughts` for queries derived from the documents surface relevant chunks in the top 10.
2. **Mixed ranking:** a search that hits both thoughts and artifact chunks returns them mixed by relevance, not segregated by target kind.
3. **Idempotency:** re-ingesting the same URI does not produce duplicate chunks. Either it's a no-op (default), it replaces, or it errors — the behavior is deterministic and documented.
4. **Failure modes:** an unreachable URI fails the ingest job cleanly (no orphaned `artifacts` row in a half-ingested state, or there's an explicit `status: "failed"` and a retry path).
5. **Operator dogfood:** the operator ingests at least one substantive past corpus (their own notes folder, prior agent transcripts) and confirms that searches against it return results they consider correct.

## Open questions

- **Chunker selection.** Fixed-token (`tiktoken-rs`), sentence-aware (`text-splitter` crate), semantic (chunk on topic shifts via the embedder)? Token-count target — 500? 1000? Overlap — 50? 100? 0? These affect retrieval quality and storage cost.
- **Deduplication policy.** Re-ingesting the same URI: refuse, replace-by-version, or replace-in-place? What about same-content-different-URI?
- **Large files.** Hard cap (e.g. 50 MB) with a clear error, or stream-and-chunk so size is unbounded? The latter is more complex; the former is fine for the M5 v0.
- **Chunk metadata.** Position info (line / page / offset) preserved in chunk metadata? Useful for "open the source" UX but the UX doesn't exist yet.
- **HTML extraction.** Use `html2text` / `readability-rs` to strip nav and ads before chunking? Or chunk raw HTML and let the embedder figure it out? Quality vs. preprocessing-cost trade.
- **Transcript ingestion.** Special case timestamps as chunk boundaries, or treat as plain text? Hinges on whether the operator wants "find the moment when X was discussed" as a use case.

# Changelog

All notable changes to Kengram are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-06-03

Initial public release. A self-hosted, MCP-native memory service: it stores agent
thoughts in Postgres + pgvector and exposes them to any MCP-capable client.

### Added
- **Capture & hybrid search (M1).** Immutable `thoughts` store; retrieval fuses
  vector kNN (BGE-M3) and trigram lexical search via reciprocal rank fusion with a
  recency boost. Four MCP tools: `capture`, `search_thoughts`, `recent_thoughts`,
  `list_scopes`.
- **Async embedding pipeline.** `pending_embeddings` queue drained by a worker;
  pluggable embedder backend (Ollama / TEI / any OpenAI-compatible `/v1`).
- **Cross-encoder reranking (M3).** Optional TEI reranker stage with an A/B
  benchmark harness (nDCG@10 / MRR); silently degrades if unavailable.
- **Tagging sidecar (M4 / M4.1).** JSONB tags (people, entities, topics,
  action_items, dates_mentioned, kind) written per thought; advisory, do not gate
  retrieval. Scope-aware controlled-vocabulary hinting. Content-fingerprint
  (SHA-256) dedup makes `capture` idempotent.
- **Relations graph (M5 / M5.2).** Thought-to-thought and polymorphic
  (entity / person / URL) links over a seven-relation closed vocabulary
  (`replaces`, `requires`, `references`, `supports`, `belongs_to`, `decided_by`,
  `refines`); soft-delete with audit. MCP tools `link_thoughts`,
  `unlink_thoughts`, `get_related_thoughts`, `get_thought`, `retract_thought`.
- **Stats CLI + tagger-extracted relations (M6 / M6.1).** `kengram stats` corpus
  and storage telemetry; the tagger auto-emits non-thought relations with
  `source='tagger'`.
- **Backup & restore (M7.0).** `kengram backup` / `kengram restore` with a
  manifest sidecar that validates schema head and warns on embedder/tagger drift.
- **Operator tooling.** Layered TOML + env config (`config/kengram.example.toml`),
  numbered sqlx migrations with a `migration_audit` table, and the
  `kengram serve` / `worker` / `migrate` / `embed-backfill` / `tag` / `bench` /
  `audit migrations` subcommands.

### Known limitations
- Single-user, single-session by design. No multi-tenant support and no web UI.
- No application-level auth yet (Tier 2 bearer tokens are planned). Default
  deployment is localhost-only (Tier 0).
- No Prometheus metrics or eval suite yet (remaining M7 scope).
- Tagger output quality varies by base model; tags are advisory only.

[Unreleased]: https://github.com/muckers/kengram/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/muckers/kengram/releases/tag/v0.1.0

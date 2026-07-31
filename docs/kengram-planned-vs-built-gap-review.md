# kEngram planned-vs-built gap review

| Field | Value |
| --- | --- |
| **Dispatch** | Knox → diesel 2026-07-27T16:30Z (ACTIVE half of dual lane) |
| **Lane** | READ-ONLY — no product code changes; no kEngram MEMORY queries |
| **Repo** | `github.com/yetibob402/kEngram` |
| **Base / head reviewed** | `origin/main` @ `786088bdf33aef34f6a2253d720e67e8d8fb8e13` (merge PR #4) |
| **Studio worktree** | `~/yetiwerks/worktrees/kEngram-diesel-gap-review` |
| **Branch** | `diesel/kengram-gap-review` |
| **Method** | Repo plans only: README, DESIGN, DEVELOPMENT, CHANGELOG, `docs/milestones/*`, TODO, code registration (tools/CLI/migrations). Not fleet memory. |

**Status codes:** **BUILT** = present and wired on `main` with file+line evidence · **PARTIAL** = present but incomplete vs plan or interim · **ABSENT** = planned/documented but not registered/enforced/wired on `main`.

---

## 0. Known-partial anchors (dispatch-required)

### 0.1 Open PR #5 — production interim timeout fix

| | |
| --- | --- |
| **PR** | https://github.com/yetibob402/kEngram/pull/5 **OPEN** |
| **Title** | `fix(kengram-mcp): capture-timeout response-deadline preemption (PRODUCTION INTERIM — not frozen-spec completion)` |
| **Head branch** | `spike/kengram-timeout-interim-5d7e618` |
| **Commits (PR)** | `f06595d8` → `d6ef0cc9` → `5d7e6185` (floor) |
| **Files** | `crates/kengram-mcp/src/server.rs`, `crates/kengram-storage/src/corpus_hygiene.rs` |
| **PR body claim** | Does **not** complete frozen spec `0a58e186`; full-scope CHANGES_NEEDED remains open. |

| Planned / claimed | Status on `main` @ `786088b` | Evidence |
| --- | --- | --- |
| Response-deadline preemption; phase budgets; `capture_outcome_unknown`; late-success rejection | **PARTIAL** — interim lives on open PR #5, **not merged** to `main` | `gh pr view 5` state=OPEN; on `main`, capture still has embedding-timeout path (`server.rs` ~561–621, test `capture_total_deadline_reserves_gate_time_after_embedding_timeout` ~1758) without PR5’s dual-reviewed full response-deadline package |
| Dual-reviewed production-only fence (server + corpus_hygiene) | **PARTIAL** | Documented on PR #5 body; Smith/Trinity artifacts cited there; not on `main` |
| Slow-response test exercises real serializer elapsed bound | **PARTIAL** (explicit PR evidence note) | PR #5 body: slow-response test sleeps before `spawn_blocking` — part of why B4/B5 stay boarded |

### 0.2 Boarded B4 / B5 — causal completion (`kengram-timeout-causal-completion`)

Per PR #5 body (production interim caveat):

| Anchor | Plan intent | Status | Evidence |
| --- | --- | --- | --- |
| **B4** registered-router dispatch | Full-scope timeout dispatch through registered router (vs interim two-file fence) | **PARTIAL / ABSENT on main** | Named only as boarded follow-on in PR #5; no B4 module or docs milestone file in-repo under that name |
| **B5** relation/queue/zero-write causal completion | Causal completion across relation/queue/zero-write paths | **PARTIAL / ABSENT on main** | Same boarding note; not implemented as a named completion on `main` |

These are **fleet-boarded known-partials**, not green checkmarks in milestone docs. Treat as open causal-completion debt after PR #5 interim.

---

## 1. MCP tools: docs vs registered handlers

**README claim:** “All nine tools” (`README.md` ~97, table ~186–195).  
**DESIGN §8** lists the same nine shipped tools; MCP `stats` deferred to M7+ (`DESIGN.md` ~420–440).  
**SERVER_INSTRUCTIONS** lists nine tools (`crates/kengram-mcp/src/server.rs` ~1394).

| Tool (planned name) | Status | Registration evidence |
| --- | --- | --- |
| `capture` | **BUILT** | `#[tool]` + `async fn capture` — `server.rs` ~494–497 |
| `search_thoughts` | **BUILT** | `server.rs` ~676–679; orchestrator `search.rs` ~393+ |
| `recent_thoughts` | **BUILT** | `server.rs` ~727–730 |
| `list_scopes` | **BUILT** | `server.rs` ~753–756 |
| `get_thought` | **BUILT** | `server.rs` ~771–774 |
| `retract_thought` | **BUILT** | `server.rs` ~789–792; `retract.rs` ~41 |
| `link_thoughts` | **BUILT** | `server.rs` ~815–818; `link.rs` ~132 |
| `unlink_thoughts` | **BUILT** | `server.rs` ~869–872; `link.rs` ~232 |
| `get_related_thoughts` | **BUILT** | `server.rs` ~915–918; `relate.rs` ~81 |
| MCP `stats` (M7 plan) | **ABSENT** (CLI only) | M7 doc MCP surface delta (`docs/milestones/m7-operational-maturity.md`); DESIGN defers; no `#[tool]` named stats in `server.rs` (nine `#[tool(` only) |
| `list_edges` (TODO proposal) | **ABSENT** | `TODO/graph_walking_issues.md` §1 |
| `get_thoughts` batch | **ABSENT** | same TODO §2 |
| `list_orphans` / `has_edges` filter | **ABSENT** | same TODO summary |
| `ingest_artifact` (historical M6) | **ABSENT** (dropped) | DESIGN §8: dropped when M6 reshaped |
| `search_facts` / `correct_fact` (M2) | **ABSENT** (retired M4) | DESIGN §8 removal note |

**Verdict:** Doc/tool list alignment for the **nine core tools is BUILT**. M7 MCP `stats` and graph-scale tools are the main **ABSENT** MCP deltas vs written plans/TODOs.

---

## 2. Relations vocabulary: documented vs enforced

| Plan element | Status | Evidence |
| --- | --- | --- |
| Seven closed relations: `replaces`, `requires`, `references`, `supports`, `belongs_to`, `decided_by`, `refines` | **BUILT** | README ~231–243; `RelationKind` enum + `ALL` — `crates/kengram-core/src/relation.rs` ~1–68; `FromStr` rejects unknowns ~74+ |
| DB CHECK on relation vocab | **BUILT** | `migrations/0007_thought_links.sql` ~21–22 `thought_links_relation_check` |
| M5.1 `supports` addition | **BUILT** | `migrations/0008_relation_supports.sql`; core enum comments ~22–26 |
| Polymorphic targets thought/entity/person/url | **BUILT** | `migrations/0009_thought_links_heterogeneous_targets.sql` `to_kind` CHECK; `link_thoughts` tests in `link.rs` ~690+ |
| Soft-delete + three-way unlink | **BUILT** | `migrations/0010_thought_links_soft_delete_and_audit.sql`; `unlink_thoughts` tests ~768+ |
| `source` agent\|tagger | **BUILT** | migration CHECK ~31–32; README ~247 |
| Tagger auto-emits non-thought relations (M6.1) | **BUILT** (by design + worker/tag path) | README M6 row ~304; DESIGN §6.7; tagger protocol docs under `docs/tagger-*` |
| Multi-hop / corpus edge enumeration | **ABSENT** | `TODO/graph_walking_issues.md` — only 1-hop `get_related_thoughts` |

**Verdict:** Closed vocab is **documented and enforced** (Rust + SQL). Graph **scale** operators remain **ABSENT**.

---

## 3. Workers / queues: designed vs wired

| Plan element | Status | Evidence |
| --- | --- | --- |
| `pending_embeddings` + worker drain | **BUILT** | README ~204–206; CLI `Command::Worker` — `crates/kengram-cli/src/main.rs` ~68, dispatch ~1268; `run_worker` / embed drainer ~761–802 |
| `pending_tags` + tag drainer | **BUILT** | same; tag drainer only when `[tagger]` configured ~67–68, ~802 |
| `start_worker.sh` / stack scripts | **BUILT** | repo root `start_worker.sh`, `start_stack.sh` |
| Capture enqueues embed (+ tag) on new thought | **BUILT** | README ~188, ~204; capture path in `server.rs` / `capture.rs` |
| Async “pending” embedding_status UX | **BUILT** | README ~64, tool description |
| Capture timeout causal completion across queue/relation/zero-write (B5) | **PARTIAL** | Boarded; PR #5 interim only addresses response-side class, not full B5 |

---

## 4. Milestone roadmap (repo-declared)

From `README.md` Roadmap table (~295–305) cross-checked to code:

| Milestone | README status | Gap review status | Notes / evidence |
| --- | --- | --- | --- |
| M0 dev environment | ✅ | **BUILT** | `docker-compose.yml`, `docs/milestones/m0-dev-environment.md` |
| M1 capture & search | ✅ | **BUILT** | Core tools + hybrid search crates |
| M2 facts pipeline | (historical) | **ABSENT** (retired) | M2 docs remain; DESIGN: facts removed in M4 collapse |
| M3 search quality / rerank | ✅ | **BUILT** | TEI rerank path, `kengram bench rerank` CLI |
| M4 collapse to thoughts + tagging | ✅ | **BUILT** | JSONB tags, fingerprint dedup, tagger crates |
| M4.1 tagging v2 | ✅ | **BUILT** | `docs/milestones/m4.1-tagging-v2.md` + extract crates |
| M5 selective relations | ✅ | **BUILT** | tools + migrations 0007–0010 |
| M6 stats CLI + tagger relations | ✅ | **BUILT** (CLI stats) | `Command::Stats` `main.rs` ~188; **not** MCP stats |
| M7.0 backup/restore | 🚧 partial | **BUILT** | `Command::Backup` / `Restore` ~200–217; m7 doc history |
| M7.1 tagger eval | (in m7 doc) | **BUILT** | `Command::Eval` ~145; `crates/kengram-cli/src/eval/`; m7 doc 2026-06-09/11 notes |
| M7 Prometheus `/metrics` | open | **ABSENT** | Planned in `m7-operational-maturity.md` ~11; no prometheus endpoint in serve path |
| M7 Tier 2 bearer tokens + audit tables | open | **ABSENT** | Planned `kengram_tokens` / `kengram_audit` in m7 doc; no such migration in `migrations/` through 0032 |
| M7 remaining eval suites (capture-recall, cross-model, LongMemEval-style) | open | **ABSENT** / **PARTIAL** | Only tagger eval shipped; m7 lists four suites |
| M7 backup retention cron / tunnel guide | open | **ABSENT** / **PARTIAL** | Retention remaining in m7; `docs/linux-autostart.md` exists for host services |
| M7 MCP `stats` | open | **ABSENT** | §1 |

---

## 5. CLI surface (operator plans)

| Subcommand (planned/docs) | Status | Evidence |
| --- | --- | --- |
| `serve` | **BUILT** | `Command::Serve` `main.rs` ~62 |
| `migrate` | **BUILT** | ~64 |
| `worker` | **BUILT** | ~68 |
| `embed-backfill` | **BUILT** | ~71 |
| `tag` | **BUILT** | ~91 |
| `bench` | **BUILT** | ~130 |
| `audit migrations` | **BUILT** | ~137–245 |
| `eval` (tagger / export-corpus) | **BUILT** | ~145–147, ~1303–1307 |
| `stats` | **BUILT** | ~188 |
| `backup` / `restore` | **BUILT** | ~200–217 |
| `chunk` / `contextual` / `ingest-hygiene` / `corpus-gate` | **BUILT** (extra ops) | dispatch ~1311–1328 |

---

## 6. TODO / FIXME / designed-but-absent UX

### 6.1 `TODO/graph_walking_issues.md` (primary written backlog)

| Item | Status | Evidence |
| --- | --- | --- |
| `list_edges` corpus edge enumeration | **ABSENT** | TODO §1 |
| `get_thoughts` batch fetch | **ABSENT** | TODO §2 |
| `recent_thoughts` include tags | **ABSENT** | TODO §3 |
| `list_orphans` / has_edges filter | **ABSENT** | TODO summary |
| Trigram fallback / short-query gating | **PARTIAL** | TODO §4; FTS migrations 0014–0015 exist but short-query gap called out |
| `list_scopes` stale aggregates | **PARTIAL** | TODO §5 |
| Tag field shape consistency (`provenance.tags` vs hit `.tags`) | **PARTIAL** | TODO §6 |

### 6.2 CHANGELOG “Known limitations” (`CHANGELOG.md` ~0.1.0)

| Limitation | Status |
| --- | --- |
| No multi-tenant / no web UI | **ABSENT** by design (DESIGN out of scope) |
| No application-level auth (Tier 2 planned) | **ABSENT** |
| No Prometheus metrics / full eval suite | **PARTIAL** (tagger eval only; no Prometheus) |

### 6.3 DESIGN open questions (§14) / out of scope (§15)

Open questions are process items. Out-of-scope (web UI, Tier 3 multi-user, cross-instance replication) intentionally **ABSENT**.

---

## 7. Architecture extras present beyond early milestone docs

| Capability | Status | Evidence |
| --- | --- | --- |
| Argus source-event / temporal hygiene | **BUILT** | `0012_argus_source_events.sql`; PR #4 on base |
| BGE-M3 dense + sparse sidecars / HNSW | **BUILT** | migrations 0016–0018, 0025–0027 |
| Artifact chunk path + contextual retrieval | **BUILT** | migrations 0019–0028; CLI `chunk` / `contextual` |
| Corpus hygiene gate / gated writer | **BUILT** | migrations 0029–0032; `corpus_hygiene.rs` |

These are **BUILT** hardening layers; they do not complete M7 Tier2/auth/metrics or PR5/B4/B5 timeout causal completion.

---

## 8. Summary matrix (highest-signal gaps)

| Area | Overall | Top residual |
| --- | --- | --- |
| Nine MCP tools | **BUILT** | Matches README/DESIGN |
| Relations closed vocab | **BUILT** | Enum + SQL CHECK |
| Embed/tag worker queues | **BUILT** | Dual drainers |
| M7.0 backup/restore | **BUILT** | CLI |
| M7.1 tagger eval | **BUILT** | CLI eval |
| Capture timeout production interim | **PARTIAL** | **PR #5 open**; not on `main` |
| B4/B5 causal completion | **PARTIAL/ABSENT** | Boarded `kengram-timeout-causal-completion` |
| MCP `stats` | **ABSENT** | CLI only |
| Prometheus `/metrics` | **ABSENT** | M7 plan |
| Tier 2 tokens + audit tables | **ABSENT** | No migrations |
| Full multi-suite eval | **PARTIAL** | Tagger only |
| Graph-scale MCP tools | **ABSENT** | `TODO/graph_walking_issues.md` |

---

## 9. Method limits

- Read-only static inventory of **this repo** at `786088b`; no live MCP/DB probe.
- Did **not** query kEngram MEMORY (Knox owns planned-vs-decided fold).
- PR #5 body used as source for B4/B5 boarding labels; no board API query.
- Line numbers are approximate anchors at review head; re-grep after merges.

---

*Generated by diesel for Knox bridge. Branch `diesel/kengram-gap-review`.*

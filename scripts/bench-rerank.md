# Kengram — rerank A/B bench runbook

End-to-end procedure for the operator to verify the `kengram bench rerank` harness against a live Kengram instance, and to use it to settle "does the cross-encoder reranker earn its latency on my actual corpus?" Shipped as M3 Phase B step 3 (closes M3 success criterion 1); post-M4 the harness operates on thoughts only (the facts pipeline was retired in M4 — see `docs/milestones/m4-collapse-to-thoughts.md`).

## Prerequisites

All four services up:

```bash
# Postgres + TEI (reranker sidecar)
docker compose up -d postgres tei
docker compose ps   # both should be "healthy"

# Ollama for embeddings (separate from docker compose)
ollama serve &       # if not already running
ollama list | grep bge-m3

# Kengram server + worker
DATABASE_URL=... cargo run --bin kengram -- serve &
DATABASE_URL=... cargo run --bin kengram -- worker &
```

Verify `[reranker]` is configured in `~/.config/kengram/kengram.toml`:

```toml
[reranker]
provider = "tei"
endpoint = "http://localhost:8080"
model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
timeout_seconds = 30
```

Smoke the reranker endpoint directly:

```bash
curl -s http://localhost:8080/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"reproducibility","texts":["Nix is reproducible","Redis is fast"]}' | jq .
# expect: scored results, Nix first
```

Define the psql helper from `scripts/smoke.md`:

```bash
kengram-psql() { docker exec -i kengram-postgres psql -U kengram -d kengram -t -c "$1"; }
```

## 1. Smoke the harness with the bundled example fixture

This first run confirms the parser, the CLI wiring, the reranker config, and the soft-fail warning path — without depending on your real corpus shape.

```bash
DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram' \
  cargo run --bin kengram -- bench rerank \
    --corpus tests/fixtures/bench-rerank.example.json
```

**Expected output:**

- Per-query `WARN bench: no relevant_ids found in either ranking — fixture may be stale` (the bundled fixture uses placeholder UUIDs that don't exist in your DB).
- A markdown table with all rows showing `nDCG@10 = 0.000` and `MRR = 0.000` for both columns.
- A summary line: `rerank improved nDCG@10 by +0.000 (4 queries); MRR by +0.000`.

If you see that, the harness is wired correctly. If you see a configuration error (`bench rerank requires a configured [reranker] section`) or a 5xx from TEI, fix that before continuing.

## 2. Author your fixture

You need ~10–30 query/relevant_ids pairs drawn from your actual corpus. Two authoring strategies:

### 2a. Browse-and-note via Claude Desktop / Code (recommended for first pass)

For each query you want to benchmark:

1. **Pick a query** that represents a real retrieval pattern — phrasings you'd actually type, not contrived ones. Aim for a mix:
   - Queries you know have a clear "right answer" (regression targets — e.g., "tooling for compiling codebases reproducibly" → the Nix-reproducibility thought).
   - Queries with multiple relevant hits (gradient relevance).
   - Queries that previously returned the wrong thing (debug targets — anywhere rerank should help).
   - One or two queries with no good answer (negative controls — both rankings should score 0).

2. **Run the query through `search_thoughts`** via Claude Desktop with `rerank: false` to see the RRF-only top-10. Copy the `thought_id` of any result you consider a "correct" hit.

3. **Run again with `rerank: true`** to see if the reranker brings additional relevant hits into the top-10 that weren't there in the RRF list. Add those IDs too.

4. **Record** in the fixture JSON.

### 2b. Direct from psql (faster for known regression targets)

If you know the thought you want to surface, grab its ID directly:

```bash
kengram-psql "SELECT id, content FROM thoughts WHERE content ILIKE '%Nix%reproducible%' AND retracted_at IS NULL"
kengram-psql "SELECT id, content FROM thoughts WHERE content ILIKE '%TCGPlayer%' AND retracted_at IS NULL"
```

Copy the UUIDs into `relevant_ids` in your fixture.

### 2c. Fixture file shape

Save to `~/.kengram/bench-rerank.json` (or anywhere outside the repo so it doesn't get committed):

```json
{
  "queries": [
    {
      "query": "tooling for compiling codebases reproducibly",
      "relevant_ids": [
        "8da1fa45-...",
        "fb38bf42-..."
      ]
    },
    {
      "query": "Postgres tuning for TCGPlayer pricing",
      "scope": "work",
      "relevant_ids": ["..."],
      "graded_relevance": {
        "...": 1.0,
        "...": 0.6
      }
    }
  ]
}
```

`graded_relevance` is optional — omit it and every id in `relevant_ids` is treated as weight 1.0. Use graded weights when you have a clear "primary hit vs supporting evidence" distinction; binary is fine for most queries.

All IDs are `thought_id`s — M4 retired the facts pipeline, so the harness is thoughts-only.

## 3. Run the bench against your real fixture

```bash
DATABASE_URL='postgres://kengram:kengram@localhost:5432/kengram' \
  cargo run --bin kengram -- bench rerank \
    --corpus ~/.kengram/bench-rerank.json
```

**Expected output:** a markdown table with one row per query plus an `AVERAGE` row, followed by a summary line like:

```
rerank improved nDCG@10 by +0.235 (12 queries); MRR by +0.180
```

Per-query rows show `Δ` columns (rerank − RRF) for both metrics; positive Δ means rerank moved the right answer up.

## 4. Interpret the output

Three diagnostic questions, in order:

1. **Are most `WARN bench: no relevant_ids found in either ranking` warnings cleared?** If many fixture queries still warn, the fixture needs revising — either the queries don't retrieve their stated relevant_ids at all (too narrow), or the IDs are stale (the underlying rows were superseded). Fix and re-run.

2. **Is the average Δ positive and material?** "Material" is operator-felt — there's no universal threshold. Reference points:
   - **+0.05 or less**: rerank is barely moving the needle; the latency cost may not be worth it. Consider rerank-off-by-default.
   - **+0.10 to +0.20**: meaningful improvement; rerank-on-by-default is defensible.
   - **+0.25 or more**: large gain; rerank-on-by-default is the obvious call. Investigate any negative-Δ queries — those are cases where rerank actively hurts.

3. **Where are the negative-Δ outliers?** Per-query rows with `Δ < 0` are queries where rerank made retrieval worse than RRF alone. A few outliers are normal (the reranker isn't oracular); a clustered pattern (e.g., all comparative-claim queries regress) is a signal to revisit the reranker model or candidate_pool size.

If the average is positive and material, that's the number for the Phase D rerank-on-by-default decision.

## 5. Iterate

Update the fixture as your corpus grows. Re-run periodically — especially after:

- Switching the reranker model (e.g., MiniLM → BGE-reranker-v2-m3 on a GPU host).
- Bumping the embedder (`[embedder].model_id` changes alter the vector leg's calibration; the trigram leg is unaffected).
- Adding new captured material that introduces previously-unrepresented query patterns.

The fixture is operator-owned — keep it outside the repo so commits don't leak the UUIDs of your private corpus.

## Troubleshooting

- **`bench rerank requires a configured [reranker] section`** — add `[reranker] provider = "tei" …` to `kengram.toml` and restart. The harness deliberately refuses to silently run RRF-only twice; the comparison would be meaningless.
- **All rows show 0.000 nDCG / MRR** — every fixture entry warned about no-match. The `relevant_ids` you authored aren't being retrieved by either ranking. Verify the IDs exist (`kengram-psql "SELECT id FROM thoughts WHERE id = '<uuid>'"`), the rows aren't retracted (`AND retracted_at IS NULL`), and the queries you wrote are likely to retrieve them at all (try the query in Claude Desktop first).
- **TEI errors mid-run** — the harness doesn't soft-fail; the comparison would be invalid. Check `docker compose logs tei`; restart if needed.
- **Embedder unreachable** — Ollama isn't running or `bge-m3` isn't pulled. `ollama list` should show `bge-m3`. The harness propagates the error rather than running trigram-only on both sides, since that defeats the point.

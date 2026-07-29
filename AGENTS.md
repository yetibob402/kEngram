# kEngram — agent / builder notes

Repo-root instructions for agents working **in this repository** (build, review,
gate). For the optional **client memory** template (how consumers use the
kengram MCP tools), see `AGENTS.md.example`.

## Gate commands (exact)

These are the verified commands used by fleet review (Jones/Trinity exact-head
gates). Run them from the workspace root after `git fetch` so `origin/main` is
current when a gate reads it.

**Binding rule:** every PR that adds or changes a test suite must update the
matching exact command in this section **in the same PR**.

### 1. Serial lib suite (`kengram-mcp`)

`sqlx::test` requires a live Postgres URL at **runtime**. Skipping
`DATABASE_URL` panics before test bodies run (board recurrence **549772** —
setup-only RED is not product evidence).

Dev DB default from `DEVELOPMENT.md` / `README.md`:

```bash
export SQLX_OFFLINE=true
export DATABASE_URL="postgres://kengram:kengram@localhost:5432/kengram"
cargo test -p kengram-mcp --lib -- --test-threads=1
```

- `SQLX_OFFLINE=true` — use committed `.sqlx/` metadata (workspace default in
  `.cargo/config.toml`; set explicitly so the invocation is self-contained).
- `DATABASE_URL` — **required** for the serial suite; point at a non-production
  database (`kengram`, not `kengram_prod`).
- `--test-threads=1` — serial execution as measured in review artifacts.

Focused selectors (examples; still require the same env bindings):

```bash
cargo test -p kengram-mcp --lib retract -- --test-threads=1
cargo test -p kengram-mcp --lib server::tests::retract_thought_tool_reports_chain_from_not_not_found -- --exact --test-threads=1
# Board 550689 payload_hash shape — unit + production callers (Jones 555325/26, 555885 exact-shape)
cargo test -p kengram-mcp --lib link::tests::payload_hash_rejects_uuid_with_hyphens -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::payload_hash_accepts_64_lowercase_hex -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::payload_hash_rejects_63_lowercase_hex -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::payload_hash_rejects_65_lowercase_hex -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::payload_hash_rejects_63_a_plus_g -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::link_thoughts_rejects_nonhex_payload_hash -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::unlink_thoughts_rejects_nonhex_payload_hash -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::link_thoughts_rejects_exact_shape_payload_hash -- --exact --test-threads=1
cargo test -p kengram-mcp --lib link::tests::unlink_thoughts_rejects_exact_shape_payload_hash -- --exact --test-threads=1
cargo test -p kengram-mcp --lib payload_hash -- --test-threads=1
```

### 2. Format gate

```bash
cargo fmt --all -- --check
```

Exit `0` = clean. Exit `1` = unformatted sources (fix with `cargo fmt --all`).

### 3. Workspace check

```bash
cargo check --workspace
```

Compile-only workspace validation (no test DB required when offline sqlx metadata is present).

### 4. Schema / binary migrate gate

Bidirectional hygiene: compare **immutable** `origin/main` top-level numeric
migration filenames (`migrations/NNNN_*.sql` only — not `ls | sort | tail`, which
can land on `rollback/`) against successful max in live `_sqlx_migrations`.

```bash
# Print main tip version only (no DB)
./scripts/schema-binary-migrate-gate.sh --print-main-max

# Compare against DATABASE_URL successful max
export DATABASE_URL="postgres://…"   # target DB (often prod for ops drill)
./scripts/schema-binary-migrate-gate.sh --compare
```

Exit codes (equal-green semantics):

| Code | Meaning |
|------|---------|
| `0` | **Equal-green** — `prod_max == main_max`. Also `0` when `prod > main` **and** `SCHEMA_AHEAD_OK=1` (must be documented on the owning change). |
| `2` | Schema **behind** main (`prod < main_max`) — deploy-without-migrate class. |
| `3` | Schema **ahead** of main without allow. |
| `4` | Usage / resolve failure (e.g. missing `origin/main`). |

Watched self-test (no live prod required; fakes prod max via PATH `psql` stub):

```bash
./scripts/schema-binary-migrate-gate-selftest.sh
```

Expect: equal green, behind RED(2), ahead RED(3), ahead+OK green.

Ops smoke notes: `scripts/post-deploy-replaces-retracted-smoke.md`.

### 5. Clippy (not a green-claim gate until baseline is clean)

Workspace Clippy may still be RED on base-identical out-of-scope debt (e.g.
`search_graph_neighbors` arity). Do **not** claim Clippy green unless the exact
command is green at the reviewed head:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

## PR hygiene reminders

- Author commits as **YetiWerks** / `bhrbek@gmail.com` on Yetiwerks-governed clones.
- Builders do not deploy prod; merge owners run the ordered post-merge drill
  (merge → **build from an immutable clone/worktree at the merge commit** →
  schema gate equal-green → restart → native probe).
  **Never** `cargo build --release` from the live deploy checkout if it can
  carry untracked files under `migrations/` — `sqlx` compile-time migrate
  embedding will bake those into the binary (board 552086 dirty-migration-embed
  class; clean pattern: detached worktree like `kengram-clean-build-<sha>`).
  Prove tracked-only set (e.g. 33 files / manifest `ad33eff8…` at d1dc680) and
  absence of known untracked needles before restart. Do **not** run `migrate`
  from a dirty-built binary.

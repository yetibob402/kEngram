# Post-deploy smoke: replaces → retracted (native MCP)

**Incident:** PR9 (9e5643a) shipped binary + per-kind error text while
`_sqlx_migrations` stopped at 32. Live `link_thoughts replaces` refused
retracted targets (knox P1 2026-07-29). Fixture: from `e765481c-…`, to `f9716202-…`.

## 1. Bidirectional schema gate (required)

Do **not** use `ls migrations/ | sort | tail` — that ends on the `rollback/`
directory name and reads the **active checkout**, not immutable `origin/main`.

Use the committed gate (extracts only top-level `NNNN_*.sql` from **immutable**
`origin/main`, compares to **successful** prod max):

```bash
git fetch origin
# main max only (no DB):
./scripts/schema-binary-migrate-gate.sh --print-main-max

# full compare (needs live DSN):
export DATABASE_URL=postgres://.../kengram_prod
./scripts/schema-binary-migrate-gate.sh --compare
# exit 0 = equal GREEN
# exit 2 = schema behind (deploy-without-migrate class)
# exit 3 = schema ahead of main without SCHEMA_AHEAD_OK=1

# deliberate ops advance (must be documented on owning PR body):
SCHEMA_AHEAD_OK=1 ./scripts/schema-binary-migrate-gate.sh --compare
```

Self-test (watched-to-fail, no live DB):

```bash
./scripts/schema-binary-migrate-gate-selftest.sh
# proves: equal green; behind RED; ahead RED; ahead+OK green
```

| Direction | Detection | Example |
| --- | --- | --- |
| **Schema behind** | prod_max < main_max | PR9 binary without 0033 |
| **Schema ahead** | prod_max > main_max | 0034 on prod before PR11 merge |
| **Equal** | prod_max == main_max | healthy |

`sqlx migrate info` alone is **not** sufficient for schema-ahead: it reflects the
**checkout** tree, so a main checkout without 0034 reports through 0033 while
live `_sqlx_migrations` max is already 34.

## 2. Apply pending if behind

```bash
cd /path/to/kEngram   # at release commit
export DATABASE_URL=postgres://.../kengram_prod
sqlx migrate run      # resolve checksum gaps carefully if present
```

## 3. Native MCP functional smoke

1. `replaces` live→retracted → success  
2. `supports` same pair → EndpointRetracted per-kind text  

Binary restart alone is **not** deploy complete.

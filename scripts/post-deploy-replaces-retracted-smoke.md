# Post-deploy smoke: replaces → retracted (native MCP)

**Incident:** PR9 (9e5643a) shipped binary + per-kind error text while
`_sqlx_migrations` stopped at 32. `lock_thought_relation_endpoints` stayed
2-arg; live `link_thoughts replaces` refused retracted targets (knox P1
2026-07-29). Fixture pair preserved: from `e765481c-…`, to `f9716202-…`.

## Required after every kengram release

```bash
cd /path/to/kEngram   # at release commit
export DATABASE_URL=postgres://.../kengram_prod
sqlx migrate info     # pending must be empty; version >= 33
sqlx migrate run      # if pending (resolve checksum gaps carefully)
```

Verify gate:

```sql
SELECT version FROM _sqlx_migrations WHERE version = 33;
SELECT pg_get_function_identity_arguments(p.oid)
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname = 'lock_thought_relation_endpoints';
-- expect 3-arg form including p_active_required_ids
```

Native MCP:

1. `replaces` live→retracted → success
2. `supports` same pair → EndpointRetracted per-kind text

Binary restart alone is **not** deploy complete.

## Bidirectional schema/binary check (knox 2ca9c15b)

Tonight produced **both** failure modes of incomplete deploy hygiene:

| Direction | Symptom | Example |
| --- | --- | --- |
| **Schema behind binary** | New binary + old gate | PR9 restart without 0033 → replaces→retracted still refused |
| **Schema ahead of main** | Ops-applied migration before merge | 0034 on prod before PR11 lands (low risk if additive; must be recorded) |

Smoke / release checklist should assert **both**:

1. **Behind:** live `_sqlx_migrations` max version ≥ version expected by the released binary/migrations tree (no pending required for that commit).
2. **Ahead:** if live version > max version on the **merged** default branch tip, that is a **deliberate** ops advance — document it on the PR that introduces the migration (do not silently leave prod ahead of main).

```bash
# Behind (binary/release commit checkout):
sqlx migrate info   # pending empty for this tree

# Ahead (compare prod to origin/main):
# prod:  SELECT max(version) FROM _sqlx_migrations;
# main:  ls migrations/ | sort | tail
# if prod > main tip files: require open PR that owns the row + body note
```

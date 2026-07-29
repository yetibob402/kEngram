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

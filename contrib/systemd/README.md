# kEngram systemd units

Example systemd units that bring the kEngram stack up automatically on boot:
the Docker backing services (Postgres + TEI reranker + ollama-embed), the MCP
server, and the worker.

**These are templates.** They contain `@@PLACEHOLDER@@` tokens you must substitute
before installing. The full runbook — prerequisites, install steps, enabling,
verifying, and troubleshooting — lives in
[`docs/linux-autostart.md`](../../docs/linux-autostart.md). Start there.

## Files

| Unit | Type | Purpose |
|---|---|---|
| `kengram-stack.service` | oneshot | Brings up the docker-compose backing services (wraps `start_stack.sh`). |
| `kengram-migrate.service` | oneshot | **Opt-in.** Applies schema migrations (`kengram migrate`) before server/worker. Enable only if you want boot-time migration; otherwise run `kengram migrate` manually. |
| `kengram-server.service` | exec | Runs `kengram serve` (the MCP/HTTP server). |
| `kengram-worker.service` | exec | Runs `kengram worker` (embed + tag drainers). |

The units ship in **user-service** form (run as your login user, read
`~/.config/kengram/kengram.toml` via XDG). The doc shows the small deltas for the
**system-service** form (`User=`/`Group=docker`/`HOME=` + an explicit `--config`).

## Placeholders

| Token | Meaning | Example |
|---|---|---|
| `@@KENGRAM_REPO@@` | Absolute path to the checkout (`WorkingDirectory`). | `/home/rjf/code/sandbox/kEngram` |
| `@@KENGRAM_BIN@@` | Absolute path to the release binary. | `/home/rjf/code/sandbox/kEngram/target/release/kengram` |
| `@@RUST_LOG@@` | Tracing log filter. | `info` |
| `@@KENGRAM_USER@@` | Run-as user (system-service variant only). | `rjf` |
| `@@KENGRAM_CONFIG@@` | Absolute config path (system-service variant only). | `/home/rjf/.config/kengram/kengram.toml` |

## Quick substitution (user services)

From the repo root, copy the templates into your user unit directory with the
placeholders filled in:

```bash
mkdir -p ~/.config/systemd/user
for unit in contrib/systemd/kengram-*.service; do
  sed \
    -e "s|@@KENGRAM_REPO@@|$PWD|g" \
    -e "s|@@KENGRAM_BIN@@|$PWD/target/release/kengram|g" \
    -e "s|@@RUST_LOG@@|info|g" \
    "$unit" > "$HOME/.config/systemd/user/$(basename "$unit")"
done
systemctl --user daemon-reload
```

Then follow [`docs/linux-autostart.md`](../../docs/linux-autostart.md) to enable
linger and start the units.

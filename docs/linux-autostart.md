# Linux autostart (systemd)

This runbook makes a kEngram deployment come back automatically after a reboot:
the Docker backing services, the MCP **server**, and the **worker**. It targets a
**headless single-server** Linux host running the backing services through
`docker-compose.yml` and the two kEngram processes as native systemd units. This
is the pragmatic homelab shape — not the "true production" setup where Postgres
and TEI are themselves systemd-managed (that's operator-managed and out of scope;
see `DESIGN.md` §11).

Example unit files live in [`contrib/systemd/`](../contrib/systemd/). They are
templates with `@@PLACEHOLDER@@` tokens; this doc walks through filling them in
and installing them.

## How the pieces fit

Five units, ordered so nothing starts before its dependencies:

```
docker.service            (the Docker daemon — already on your system)
   │
   ▼
kengram-stack.service     (docker compose up: postgres, tei, ollama-embed)
   │
   ▼
kengram-migrate.service   (optional — apply schema migrations)
   │
   ▼
kengram-server.service    (kengram serve — the MCP/HTTP endpoint)
   │
   ▼
kengram-worker.service    (kengram worker — embed + tag drainers)
```

`kengram-stack.service` wraps `start_stack.sh`, which waits for Postgres to accept
connections before reporting success — so the server and worker order *after* the
database is actually ready, not merely after the container launched.

**Two flavors.** The templates ship as systemd **user** services (recommended for
single-user hosts — they run as your account and read `~/.config/kengram/kengram.toml`
with no extra wiring). A **system**-service variant is documented inline at each
step for hosts where you'd rather the units be root-owned.

**Distro support.** Anything with systemd: Ubuntu, Debian, Fedora, RHEL / Rocky /
AlmaLinux, Arch, openSUSE, and friends. The unit syntax is identical across them;
only the `docker` binary path can differ (Snap installs in particular — check
`which docker`). Non-systemd distros (Alpine/OpenRC, Void/runit, Devuan) are out of
scope.

## Prerequisites

1. **Docker daemon enabled at boot.** The backing services run in Docker, so the
   daemon must start on boot:
   ```bash
   sudo systemctl enable --now docker
   ```
2. **Your user is in the `docker` group** (so user services can talk to the
   daemon without sudo):
   ```bash
   sudo usermod -aG docker "$USER"   # then log out/in, or `newgrp docker`
   ```
3. **Release binary built.** The units run a prebuilt binary, never `cargo run`:
   ```bash
   cargo build --release            # produces target/release/kengram
   ```
   You can also install it somewhere stable (e.g. `sudo install -m755
   target/release/kengram /usr/local/bin/kengram`) and point `@@KENGRAM_BIN@@`
   there instead.
4. **Config present** at `~/.config/kengram/kengram.toml` (copy
   `config/kengram.example.toml` and edit). Note `[server].bind` (default
   `127.0.0.1:8080`) — that's the endpoint you'll verify against. The running
   binary reads this file plus `KENGRAM_*` env vars; it does **not** read plain
   `DATABASE_URL`.
5. **Stack initialized once, interactively:**
   ```bash
   ./start_stack.sh
   ```
   The first run writes the arch-specific `KENGRAM_TEI_IMAGE_TAG` to `.env` and
   pulls `bge-m3` into the `ollama-embed` container (one-time, ~1.3 GB). Doing
   this once by hand keeps the first boot fast and predictable.
6. **Schema migrated once:**
   ```bash
   ./target/release/kengram migrate
   ```
   (Or enable `kengram-migrate.service` below to do this on every boot — see
   "Enabling and starting".)

## Choosing user vs system services

| | User services (recommended) | System services |
|---|---|---|
| Location | `~/.config/systemd/user/` | `/etc/systemd/system/` |
| Runs as | your login user | root (or `User=`) |
| Finds `~/.config/kengram/kengram.toml` | automatically (XDG) | only if you pass `--config` or set `HOME=` |
| Starts at boot | **needs `loginctl enable-linger`** | yes, unconditionally |
| Order against `docker.service` | not directly (cross-manager) | `After=`/`Requires=docker.service` works |
| Manage with | `systemctl --user …` | `sudo systemctl …` |

On a **headless** host, the one thing to remember for user services is
`loginctl enable-linger`: without it, your user's systemd manager only runs while
you have an active login session, so the services would not start at boot and
would stop when you log out. The system-service variant avoids linger entirely but
must be told where your config lives.

## Installing — user services (recommended)

The unit files in `contrib/systemd/` contain `@@PLACEHOLDER@@` tokens (e.g.
`@@KENGRAM_BIN@@`) rather than hardcoded paths. These are **not** environment
variables you export anywhere — they're substituted at install time. The `sed`
command below copies each template into `~/.config/systemd/user/` with the tokens
replaced by the values on the right-hand side of each `s|@@TOKEN@@|value|g`
expression, producing units with concrete paths (e.g.
`ExecStart=/home/you/.../target/release/kengram serve`).

The command assumes you run it **from the repo root**, so `@@KENGRAM_REPO@@`
becomes the current directory (`$PWD`) and `@@KENGRAM_BIN@@` becomes
`$PWD/target/release/kengram`. If your binary lives elsewhere, replace that
**whole** value — including the `$PWD/` prefix — with the correct path. For an
installed binary that means an absolute path with no `$PWD`:

```bash
    -e "s|@@KENGRAM_BIN@@|/usr/local/bin/kengram|g" \
```

(`@@KENGRAM_REPO@@` should still map to `$PWD` — the stack unit and `docker
compose` run from the repo root.) After installing, sanity-check the result:

```bash
grep -H '^ExecStart=' ~/.config/systemd/user/kengram-*.service
```

Every `ExecStart=` should name a path that actually exists — no doubled segments
like `…/kEngram/usr/local/bin/kengram`. You can also just edit the installed
`.service` files directly afterward; they're plain text in
`~/.config/systemd/user/`.

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

Enable linger so the units start at boot without a login session:

```bash
loginctl enable-linger "$USER"
```

## Installing — system services (alternative)

Substitute the same placeholders plus `@@KENGRAM_USER@@` and `@@KENGRAM_CONFIG@@`,
and write to the system directory:

```bash
for unit in contrib/systemd/kengram-*.service; do
  sudo sed \
    -e "s|@@KENGRAM_REPO@@|$PWD|g" \
    -e "s|@@KENGRAM_BIN@@|$PWD/target/release/kengram|g" \
    -e "s|@@KENGRAM_USER@@|$USER|g" \
    -e "s|@@KENGRAM_CONFIG@@|$HOME/.config/kengram/kengram.toml|g" \
    -e "s|@@RUST_LOG@@|info|g" \
    "$unit" | sudo tee "/etc/systemd/system/$(basename "$unit")" >/dev/null
done
```

Then, in each installed unit, apply the changes the templates document in their
`# System variant:` comments:

- **`ExecStart`** — add the explicit config path, e.g.
  `ExecStart=/home/you/.../kengram --config /home/you/.config/kengram/kengram.toml serve`
  (the `--config` flag is global and goes *before* the subcommand).
- **`[Service]`** — add `User=@@KENGRAM_USER@@`, `Group=docker`, and
  `Environment=HOME=/home/@@KENGRAM_USER@@`.
- **`[Install]`** — change `WantedBy=default.target` to `WantedBy=multi-user.target`.
- **`kengram-stack.service` `[Unit]`** — add `After=docker.service` and
  `Requires=docker.service` (a system unit *can* order against the daemon).

Reload:

```bash
sudo systemctl daemon-reload
```

The rest of this doc uses the `systemctl --user` forms; for system services drop
`--user` and prefix with `sudo`.

## Enabling and starting

Optionally enable boot-time migration (off by default — kEngram does not
auto-migrate, see `DESIGN.md` §11). Enable this only if you want a freshly-rebuilt
binary to apply its own schema on boot:

```bash
systemctl --user enable kengram-migrate.service
```

Enable and start the rest:

```bash
systemctl --user enable --now \
  kengram-stack.service kengram-server.service kengram-worker.service
```

`--now` starts them immediately as well as enabling them for future boots.
(`kengram-migrate.service` is pulled in automatically before the server when
enabled; you don't list it here.)

## Verifying

```bash
# Unit health
systemctl --user status kengram-stack kengram-server kengram-worker

# Live logs (RUST_LOG controls verbosity; output goes to journald)
journalctl --user -u kengram-server -f

# Backing containers up
docker compose ps        # kengram-postgres / kengram-tei / kengram-ollama-embed

# The MCP endpoint answers (use the port from [server].bind; 8080 is the default)
curl -sS http://127.0.0.1:8080/mcp
```

A healthy server logs a line like `kengram serve started … tagger=enabled` (or
`disabled`); the worker logs `kengram worker started … tagger=…`.

To confirm it truly survives a reboot, `sudo reboot` and re-run the checks above
once the box is back.

## Boot ordering, honestly

The `After=`/`Wants=` chain orders the units (stack → migrate → server → worker).
The stack unit wraps `start_stack.sh`, which blocks until Postgres accepts
connections, so the server and worker normally start against a ready database.

If you swapped the stack `ExecStart` for the lighter raw `docker compose up -d`
(no readiness wait), the server may exit once or twice on a cold boot before
Postgres is listening — `Restart=on-failure` brings it back within a few seconds.
That's expected and self-healing.

Server and worker use `Wants=` (not `Requires=`) on the stack: a transient stack
hiccup won't forcibly tear them down; they ride it out via `Restart=on-failure`.

**User-service limitation:** a user unit can't order against the system-level
`docker.service` (they live in different managers). On a normal host the Docker
daemon is up well before your lingered user manager, so this is a non-issue. If
you need a hard guarantee, use the system-service variant, which can declare
`After=docker.service` / `Requires=docker.service`.

## Stopping and disabling

```bash
# Stop now (leaves them enabled for next boot)
systemctl --user stop kengram-worker kengram-server kengram-stack

# Stop and prevent autostart
systemctl --user disable --now kengram-worker kengram-server kengram-stack
```

Stopping `kengram-stack.service` runs its `ExecStop` (`docker compose stop`),
which halts the containers but **preserves the Postgres data volume** — the same
behavior as `./stop_stack.sh`. To fully remove the containers run
`docker compose down` yourself; to wipe the corpus, `docker compose down -v`.

To undo boot-without-login for user services:

```bash
loginctl disable-linger "$USER"
```

## Troubleshooting

- **Server crash-loops right after a reboot.** Postgres wasn't accepting
  connections yet. Expected with the lightweight `docker compose up` stack
  `ExecStart`; it recovers within a few `Restart=on-failure` cycles. Use the
  default `start_stack.sh` `ExecStart` (it waits for readiness) to avoid the churn.
- **Nothing starts at boot (user services).** Linger isn't enabled — the units
  only run while you're logged in. Fix: `loginctl enable-linger "$USER"`.
- **Server connects to the wrong database / uses defaults (system service).**
  The unit ran without your `HOME`, so it never found
  `~/.config/kengram/kengram.toml` and fell back to built-in defaults
  (`postgres://kengram:kengram@localhost:5432/kengram`, bind `127.0.0.1:8080`).
  Fix: add `--config /home/you/.config/kengram/kengram.toml` to `ExecStart` (or set
  `Environment=HOME=/home/you` with `User=you`).
- **`permission denied` talking to Docker.** Your user isn't in the `docker`
  group (user services), or the system unit lacks `Group=docker`. Fix the group
  membership / unit, then `daemon-reload`.
- **`docker: command not found` in the unit.** The unit uses `/usr/bin/docker`;
  Snap and some installs put it elsewhere. Run `which docker` and adjust the path
  in `kengram-stack.service`.
- **Port already in use on start.** Another process holds `[server].bind`. Pick a
  free port in your config, or see the port-collision note in `DEVELOPMENT.md`.
- **`SQLX_OFFLINE` / `DATABASE_URL`?** Neither matters at runtime — those are
  build-time concerns for compiling the binary. A prebuilt binary needs only the
  config file (or `KENGRAM_*` env). `kengram migrate` likewise reads `[database].url`
  from config, not `DATABASE_URL`.
- **`kengram-migrate.service` fails with `migration N was previously applied but
  has been modified`.** sqlx records a checksum of each migration when it's
  applied; this error means the bytes of `migrations/000N_*.sql` in your checkout
  no longer match what was applied to the database (even a comment or whitespace
  edit changes the checksum). If the database is already fully migrated — the
  common case on an existing host — boot-time migration is unnecessary: leave the
  unit disabled (`systemctl --user disable kengram-migrate.service`) and the
  server/worker run fine without it (they only `After=` it, not `Requires=`). The
  drift itself is worth resolving before you ever need to apply a *new* migration;
  that's a database-maintenance task outside this runbook.
- **A backing container crash-loops with `Temporary failure in name resolution`
  / `Could not download model artifacts`.** The container can't resolve DNS. This
  is common on hosts running **systemd-resolved**: `/etc/resolv.conf` points at the
  loopback stub `127.0.0.53`, which a container can't use, and with no Docker DNS
  override the container has no working resolver (Tailscale/`*.ts.net` setups hit
  this too). The host itself is usually fine — test with `getent hosts
  huggingface.co` on the host vs. inside a container. Two fixes:
  - **Per-stack, no sudo:** create a local `docker-compose.override.yml` (it's
    gitignored) giving the affected service a real resolver, then
    `docker compose up -d --force-recreate <service>`:
    ```yaml
    services:
      tei:
        dns: ["1.1.1.1", "8.8.8.8"]
    ```
    Compose auto-merges the override on every invocation — including the one the
    `kengram-stack.service` unit runs — so it persists across reboots. (To *replace*
    a base list like `ports` rather than append to it, tag it `!override`; needs
    Compose ≥ 2.24.)
  - **Host-wide, needs sudo:** create `/etc/docker/daemon.json` with
    `{"dns": ["1.1.1.1", "8.8.8.8"]}` and `sudo systemctl restart docker`.

  Once a model has downloaded into its named volume, it's cached — subsequent
  boots don't need the network for that service.
- **A backing container fails to start with `address already in use`.** Some host
  port the stack publishes (e.g. TEI's `8080`) is taken by another process. Find
  it (`ss -ltnp | grep :8080`), then remap the service to a free host port in your
  local `docker-compose.override.yml` and update the matching kEngram config:
  ```yaml
  services:
    tei:
      ports: !override   # replace the base 8080 mapping, don't append
        - "8090:80"
  ```
  Then point `[reranker].endpoint` (in `~/.config/kengram/kengram.toml`) at the new
  port and restart `kengram-server`. The container-internal port (80) is unchanged,
  so the healthcheck still works.
- **`journalctl --user` shows nothing.** Either linger is off, or you're looking
  at the system journal for a user service — use `journalctl --user -u <unit>`
  (or, for the system variant, `sudo journalctl -u <unit>`).

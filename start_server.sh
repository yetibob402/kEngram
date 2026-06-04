#!/usr/bin/env bash
# Run the kEngram MCP server (foreground). Binds 127.0.0.1:8080, MCP at /mcp.
# Run ./start_stack.sh first (Postgres must be up). Pair with ./start_worker.sh
# in a second terminal to drain the embed (and tag) queues.
#
# Honors your config: the database URL comes from ~/.config/kengram/kengram.toml
# (`[database].url`), the `KENGRAM_DATABASE__URL` env var, or the built-in
# default (postgres://kengram:kengram@localhost:5432/kengram) — in that
# precedence. Builds are offline by default (.cargo/config.toml sets
# SQLX_OFFLINE), so no DATABASE_URL is needed to compile.
set -euo pipefail

exec cargo run --bin kengram -- serve

#!/usr/bin/env bash
# Run the Kengram MCP server (foreground). Binds 127.0.0.1:8080, MCP at /mcp.
# Run ./start_stack.sh first (Postgres must be up). Pair with ./start_worker.sh
# in a second terminal to drain the embed (and tag) queues.
set -euo pipefail

DB_URL='postgres://kengram:kengram@localhost:5432/kengram'

# KENGRAM_DATABASE__URL is what the running binary reads (figment, KENGRAM_
# prefix, __ nesting). DATABASE_URL is what sqlx-cli and the build-time
# sqlx::query! macros read. Set both from one value so runtime and build never
# diverge.
KENGRAM_DATABASE__URL="$DB_URL" \
DATABASE_URL="$DB_URL" \
  cargo run --bin kengram -- serve

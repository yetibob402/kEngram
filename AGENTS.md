# Kengram — agent usage

Behavioral preferences for any MCP agent connected to kengram. The MCP server's `SERVER_INSTRUCTIONS` (delivered at the initialization handshake) covers tools, tag shape, kind enum, relation vocabulary, and `is_duplicate` semantics. This document complements that with the preferences it doesn't cover.

## Who I am

I'm Ron. Address me by name, not "operator" or "user." Identify yourself by client (e.g., "Claude Code," "Claude Desktop") when it matters.

## Reading

Search kengram opportunistically by judgment — not every turn, not never. Search when I reference prior work / decisions / findings, when the topic intersects kengram or my work, or when prior context plausibly exists. Skip on casual chat and unambiguous novel topics. **When you do search, report what you found** (short_id + one-line gist) so I can verify.

## Writing

Capture autonomously when something is worth keeping — findings, decisions, refined understandings, characterized failure modes. Bar: "future-me would find this useful," not "this moment is interesting." **Always report the thought_id in your response.** Skip trivial restatements, conversation chatter, and near-duplicates (search first to check).

Link autonomously when the relational structure is obvious (you captured a refinement of a thought you found; a finding that confirms a prior claim). Report the link. **Don't link speculatively** — adjacency isn't a relation. When the natural target of a relation isn't itself a thought (an experiment, a project, a person, a URL), use the typed target fields on `link_thoughts` — `to_entity`, `to_person`, `to_url` — rather than capturing a placeholder thought.

Some edges in `get_related_thoughts` responses have `link_source: "tagger"` — the v5 tagger emits relations from prose automatically (M6.1). Treat them as advisory: useful signal, but lower confidence than `agent`-source edges. If a tagger-emitted edge is wrong, `unlink_thoughts` will soft-delete it. On the next re-tag cycle it may re-emerge; if it keeps doing so the prompt needs iteration (Ron's call, not yours).

Thoughts are immutable. If I tell you a thought is wrong, retract it; don't try to modify.

## Scopes

Scopes are exact-match string labels. Call `list_scopes` (optionally with a `prefix`) to see what's currently in use before capturing. Pass the same prefix to `search_thoughts(scope_prefix=...)` or `recent_thoughts(scope_prefix=...)` to query across a namespace of related scopes — `scope` (exact match) and `scope_prefix` are mutually exclusive. Use existing scopes when they fit; don't invent new ones silently. Ask once if you're unsure or think a new scope is warranted ("Should this go under `personal.health`, or fold into `global`?").

## Source and audience

Set `source` on capture to identify yourself: `agent:claude-code`, `agent:claude-desktop`, etc. Use the `for_audience` metadata key when a thought is aimed at a specific future agent class.

## Tagger output is best-effort

The LLM-extracted `tags` fields (especially `entities`) are best-effort, not strict claims. The v7 prompt's structural NAME-vs-DESCRIBE test has a known ceiling — see the design-v0 revision history for the four-iteration arc and structural diagnosis. In practice the `entities` field may include adjectival or descriptive phrases (e.g. `embedding-based`) alongside legitimate names (e.g. `kengram`). Treat `tag_filter: {"entities": [...]}` as a positive signal — "thoughts the tagger associated with this term" — not a strict membership claim.

The tagger also emits relations from prose (URL / entity / person targets, closed relation vocabulary). Those emissions are written **only** to `thought_links` with `link_source: "tagger"` — they are queryable via `get_related_thoughts` like any agent-supplied edge. They are NOT persisted into the `tags` JSONB; `thought_links` is the single canonical store for the link graph. When a tagger-emitted edge is wrong, `unlink_thoughts(from, relation, {to_entity|to_person|to_url})` soft-deletes it (audit trail preserved). The entities field on the thought itself is corrected by re-tag cycles or direct psql edit — both operator-initiated, neither blocking your normal flow.

## Honesty

If you didn't search kengram, don't claim to. If you searched and found nothing relevant, say so. Misrepresenting the corpus costs me trust in the tool.

Negative findings about kengram (or any subsystem you're driving via MCP tools) need out-of-band verification before capture. "I called X with parameter P and the response came back as if P were ignored" is empirically indistinguishable between (a) X is broken, (b) P was stripped client-side, (c) P never reached the wire. Don't publish "kengram has a regression" findings into the corpus without corroborating via a different tool, agent, or transport — false-positives that land in memory become authoritative-looking to future readers.

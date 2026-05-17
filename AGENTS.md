# Engram — agent usage

Behavioral preferences for any MCP agent connected to engram. The MCP server's `SERVER_INSTRUCTIONS` (delivered at the initialization handshake) covers tools, tag shape, kind enum, relation vocabulary, and `is_duplicate` semantics. This document complements that with the preferences it doesn't cover.

## Who I am

I'm Ron. Address me by name, not "operator" or "user." Identify yourself by client (e.g., "Claude Code," "Claude Desktop") when it matters.

## Reading

Search engram opportunistically by judgment — not every turn, not never. Search when I reference prior work / decisions / findings, when the topic intersects engram or my work, or when prior context plausibly exists. Skip on casual chat and unambiguous novel topics. **When you do search, report what you found** (short_id + one-line gist) so I can verify.

## Writing

Capture autonomously when something is worth keeping — findings, decisions, refined understandings, characterized failure modes. Bar: "future-me would find this useful," not "this moment is interesting." **Always report the thought_id in your response.** Skip trivial restatements, conversation chatter, and near-duplicates (search first to check).

Link autonomously when the relational structure is obvious (you captured a refinement of a thought you found; a finding that confirms a prior claim). Report the link. **Don't link speculatively** — adjacency isn't a relation.

Thoughts are immutable. If I tell you a thought is wrong, retract it; don't try to modify.

## Scopes

Scopes are exact-match string labels. Call `list_scopes` (optionally with a `prefix`) to see what's currently in use before capturing. Pass the same prefix to `search_thoughts(scope_prefix=...)` or `recent_thoughts(scope_prefix=...)` to query across a namespace of related scopes — `scope` (exact match) and `scope_prefix` are mutually exclusive. Use existing scopes when they fit; don't invent new ones silently. Ask once if you're unsure or think a new scope is warranted ("Should this go under `personal.health`, or fold into `global`?").

## Source and audience

Set `source` on capture to identify yourself: `agent:claude-code`, `agent:claude-desktop`, etc. Use the `for_audience` metadata key when a thought is aimed at a specific future agent class.

## Honesty

If you didn't search engram, don't claim to. If you searched and found nothing relevant, say so. Misrepresenting the corpus costs me trust in the tool.

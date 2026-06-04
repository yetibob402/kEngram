# Security Policy

## Supported versions

kEngram is pre-1.0. Security fixes are applied to the latest `main` and the most
recent tagged release only.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for
anything exploitable.

- Preferred: open a [GitHub private security advisory](https://github.com/muckers/kEngram/security/advisories/new).
- Or email **ron.forrester@me.com** with `KENGRAM SECURITY` in the subject.

Please include enough detail to reproduce (affected version/commit, configuration,
and steps). You can expect an acknowledgement within a few days. As a single-maintainer
project, fix timelines are best-effort; coordinated disclosure is appreciated.

## Deployment security model

kEngram is **single-user and local-first** by design. Its trust tiers:

- **Tier 0 (default):** binds to `127.0.0.1` — only local processes can reach it.
- **Tier 1:** exposure over a private mesh (e.g. Tailscale) via `[server].bind` +
  `[server].allowed_hosts`. There is **no application-level authentication** at this
  tier — network reachability is the access boundary. Do not expose a Tier 1
  instance to an untrusted network.
- **Tier 2 (planned):** bearer-token auth with a hashed allowlist and an audit log.
  Not yet shipped — see the roadmap.

Treat the Postgres database, the embedding/tagging sidecars, and the captured
corpus as sensitive: thoughts may contain personal or proprietary content.

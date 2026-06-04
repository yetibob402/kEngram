# Tagger sidecar protocol — wire contract

The human-readable companion to
[`crates/kengram-tagger-protocol/src/lib.rs`](../crates/kengram-tagger-protocol/src/lib.rs)
(the serde-derived types in that crate ARE the spec; this doc targets
non-Rust implementers who want to read the JSON shape without parsing
Rust source).

When kengram's `[tagger].provider = "http"`, the worker POSTs to a
sidecar that implements the protocol below. Sidecars can be written in
any language; the only requirement is speaking the JSON wire shape and
returning HTTP status codes that match the transient-vs-non-transient
classification kengram's drainer relies on.

The reference sidecar lives at
[`crates/kengram-tagger-deterministic/`](../crates/kengram-tagger-deterministic/);
its Dockerfile + README cover the asset setup. Anyone writing a sidecar
in another language can copy the wire shape from this doc and ignore
the Rust implementation entirely.

## Protocol version

Current: `"1"`.

Bumped when the JSON shapes change in a way that older sidecars can't
honor. Sent on every request as a string field; sidecars may echo it
in the response so kengram can detect mismatches at the first call
rather than as silent misbehavior.

## Endpoints

### `POST /tag`

Tag a single thought's content.

**Request:**

```http
POST /tag HTTP/1.1
Content-Type: application/json

{
  "protocol_version": "1",
  "content": "Sarah pushed the new bge-m3 reranker config. Need to verify the latency improvement holds.",
  "vocab": {
    "topics":   ["memory-systems", "rust"],
    "entities": ["kengram", "pgvector"]
  }
}
```

Field reference:

| Field | Type | Required | Notes |
|---|---|---|---|
| `protocol_version` | string | yes | Always `"1"` for current protocol. Sidecar should `400` an unknown version. |
| `content` | string | yes | The thought's text. Sidecar runs its tagger against this. |
| `vocab` | object\|null | no | Optional controlled-vocabulary hint. Sidecar may use as nudge toward consistent terms; should not treat as closed vocabulary. Omit or send `null` if not used. |
| `vocab.topics` | string[] | yes (if vocab present) | Most-used topic labels in the thought's scope. Default `[]`. |
| `vocab.entities` | string[] | yes (if vocab present) | Most-used entity strings in the thought's scope. Default `[]`. |

**Response (success):**

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "protocol_version": "1",
  "tags": {
    "people":          ["Sarah"],
    "entities":        [],
    "action_items":    ["verify the latency improvement holds"],
    "topics":          ["memory-systems"],
    "dates_mentioned": [],
    "kind":            "task"
  },
  "relations": []
}
```

Field reference:

| Field | Type | Required | Notes |
|---|---|---|---|
| `protocol_version` | string\|null | no | Optional. If present, must equal the request's version or kengram rejects with `Misconfigured`. |
| `tags` | object | yes | Persisted metadata. All fields are arrays except `kind` (string\|null). An empty `tags` (every field empty/null) is a valid "nothing extractable" response. |
| `tags.people` | string[] | no | Default `[]`. Person names extracted from the thought. |
| `tags.entities` | string[] | no | Default `[]`. Proper-noun-style identifiers (projects, products, libraries). Deprecated under the 5-field schema; non-LLM sidecars typically leave empty. |
| `tags.action_items` | string[] | no | Default `[]`. Imperatives or to-do items in the thought. |
| `tags.topics` | string[] | no | Default `[]`. Broad subject categories. Typically 0-3 short kebab-case labels. |
| `tags.dates_mentioned` | string[] | no | Default `[]`. Surface forms of dates / temporal references. No interpretation expected. |
| `tags.kind` | string\|null | no | Default `null`. One of: `"observation"`, `"task"`, `"idea"`, `"reference"`, `"person_note"`, `"session"`, `"decision_record"`. |
| `relations` | array | no | Default `[]`. Tagger-extracted relations. See "Relations" below. |

**Relations** (optional, often empty for non-LLM sidecars):

```json
{
  "relation":  "supports",
  "to_kind":   "entity",
  "to_value":  "kengram",
  "note":      "optional commentary"
}
```

- `relation`: one of `"replaces"`, `"requires"`, `"references"`,
  `"supports"`, `"belongs_to"`, `"decided_by"`, `"refines"`.
- `to_kind`: one of `"entity"`, `"person"`, `"url"`. (Thought-target
  relations are out of scope at the sidecar layer.)
- `to_value`: the target string (entity name, person name, or URL).
- `note`: optional commentary, often omitted.

LLM-shaped taggers populate this when prose has explicit relational
claims. Token-level NER pipelines (like the reference sidecar) leave
it empty.

### `GET /health`

Liveness probe. kEngram doesn't call this directly, but Docker
healthchecks + operator scripts often do.

**Response:**

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "ok"}
```

The reference sidecar returns 200 as soon as startup completes (model
loaded, taxonomy embedded). Sidecars are free to define their own
liveness criteria — kengram doesn't depend on this endpoint.

## Error semantics

kEngram's drainer treats responses according to HTTP status:

| Status range | Drainer behavior |
|---|---|
| `2xx` | Parse body as `TagResponse`; on parse failure, treat as non-transient. |
| `4xx` (incl. 400, 422) | Non-transient: log the response body, skip this thought, don't retry. |
| `5xx` | Transient: leave the `pending_tags` row in the queue with `attempts++`; retry on the next worker tick. |
| Connection failures (refused, reset, timeout) | Transient: same as 5xx. |
| `MalformedResponse` (response body isn't valid JSON or doesn't match the schema) | Non-transient. |

Recommended sidecar conventions (the reference sidecar follows these,
but they're not mandated by the protocol):

- `400 Bad Request` for unsupported `protocol_version` or malformed
  request bodies.
- `422 Unprocessable Entity` for non-transient pipeline failures (e.g.
  a model file is missing — kengram retrying won't help).
- `503 Service Unavailable` for transient pipeline failures (embedder
  unreachable, ONNX runtime hiccup — retry might succeed).
- `500 Internal Server Error` for unexpected exceptions.

Body for non-200 responses is conventionally `{"error": "..."}` but
kengram's client only reads the status code + raw body string for the
error message; the exact shape doesn't matter.

## Writing a sidecar in another language

Minimum viable sidecar in pseudocode:

```python
@app.post("/tag")
def tag(req):
    if req["protocol_version"] != "1":
        return JSONResponse({"error": "bad protocol_version"}, status_code=400)
    content = req["content"]
    vocab = req.get("vocab")

    # ... your tagger logic ...
    tags = my_tagger.extract(content, vocab=vocab)

    return {
        "protocol_version": "1",
        "tags": {
            "people":          tags.people,
            "entities":        [],
            "action_items":    tags.actions,
            "topics":          tags.topics,
            "dates_mentioned": tags.dates,
            "kind":            tags.kind  # or None
        },
        "relations": []
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

That's the entire contract on the sidecar side. Anything that speaks
this JSON over HTTP works with kengram's `provider = "http"` arm.

## Compatibility commitments

The protocol crate's serde derives use `#[serde(default)]` for every
optional field, so:

- **Adding a new optional field to `tags`** is backward-compatible —
  older sidecars omit the field; kengram parses with the default.
- **Adding a new variant to `kind`** is backward-compatible at the
  Rust level (the variants are an enum so unknown variants fail
  deserialization). Wire-protocol-wise, treat as a breaking change
  that requires a `protocol_version` bump.
- **Renaming or removing a field** is a breaking change → bump
  `protocol_version`.
- **Changing the meaning of an existing field** is a breaking change
  → bump `protocol_version`.

When the protocol version bumps, kengram's HTTP-tagger client and the
reference sidecar bump together. Non-Rust sidecars must update to the
new version; the response's optional `protocol_version` field lets
kengram detect a mismatch at the first call rather than later as
malformed-data drift.

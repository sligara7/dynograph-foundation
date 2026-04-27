# `/v1/*` REST API

This is the wire reference for the HTTP API the `dynograph` binary
serves. The Rust client (`dynograph-client`) wraps every endpoint
listed here; consumers in other languages can hit the same routes
directly with any HTTP client.

All request and response bodies are JSON. YAML schema input is
accepted only at config / TOML time — `/v1/*` itself is JSON-only.

## Conventions

- Non-2xx responses carry the failure reason as a plain-text body.
  Don't parse JSON from error responses; the body is the
  human-readable message (e.g. `"missing Authorization header"`,
  `"schema evolution rejected: removed node type: Item; …"`).
- `wire_version` and `content_hash` appear on every response that
  embeds or describes a `Schema`. Consumers compare them against
  compiled-in constants and fail-fast on mismatch (drift detection).
- `node_id`, `graph_id`, `node_type`, `edge_type` in URLs must
  match `[A-Za-z0-9_-]+`, max 100 chars; foundation rejects path-
  traversal-style ids before any filesystem op.

## Auth

When `[auth].provider = "bearer_jwt"`, every `/v1/*` request must
carry `Authorization: Bearer <jwt>`. `/health`, `/ready`, and
`/metrics` are public regardless of provider. See
[`docs/service.md`](service.md#bearerjwt) for token requirements.

A failed authentication is `401 Unauthorized` with the reason
in the body (`"missing Authorization header"`, `"invalid token: …"`,
etc).

## Operational

### `GET /health`

Liveness. Always returns `200 ok`.

### `GET /ready`

Readiness. `200 ready` once startup work (`rehydrate` on `OnDisk`)
is complete; `503 starting` before that. In-memory mode is ready
immediately.

### `GET /metrics`

Prometheus text format (0.0.4). Series:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynograph_build_info` | gauge | `version` | Always `1`; `version` is the `CARGO_PKG_VERSION`. |
| `dynograph_uptime_seconds` | gauge | — | Process uptime. |
| `dynograph_http_requests_total` | counter | `method`, `path`, `status` | Per-route request count. `path` is the matched-route pattern (`/v1/graphs/{id}`), not the literal URL — cardinality is bounded by route count. |
| `dynograph_http_request_duration_microseconds_sum` | counter | `method`, `path`, `status` | Cumulative latency. Pair with the counter via `rate(sum) / rate(count)` for avg latency. |
| `dynograph_hnsw_index_size` | gauge | `graph`, `node_type` | Live (non-tombstoned) embeddings per per-(graph, type) index. |
| `dynograph_hnsw_searches_total` | counter | `graph`, `node_type` | HNSW search calls per index. |
| `dynograph_hnsw_inserts_total` | counter | `graph`, `node_type` | HNSW insert calls per index. |
| `dynograph_hnsw_removes_total` | counter | `graph`, `node_type` | HNSW remove calls per index. |

## Graph lifecycle

### `POST /v1/graphs`

Create a graph.

```json
{ "id": "my_graph", "schema": { "name": "...", "version": 1, "node_types": {...}, "edge_types": {...} } }
```

Returns `201 Created` + `SchemaResponse`:

```json
{ "id": "my_graph", "wire_version": "0.3.0", "content_hash": "<sha256-hex>", "schema": {...} }
```

Errors: `400` invalid id; `409` duplicate.

### `GET /v1/graphs`

List graph ids.

```json
{ "graphs": ["alpha", "beta", "gamma"] }
```

### `GET /v1/graphs/{id}`

Metadata-only: `{id, wire_version, content_hash}`. Cheap existence
check + content-hash drift comparison. Use `/schema` for the full
schema body.

Errors: `404` not found.

### `DELETE /v1/graphs/{id}`

Returns `204 No Content`. On `OnDisk`, the per-graph dir is removed.
In-flight handlers holding a cloned `Arc<GraphEntry>` complete
their op; subsequent ops surface storage errors.

Errors: `404` not found.

## Schema

### `GET /v1/graphs/{id}/schema`

Full schema body — same shape codegen / drift-detection consumers
read.

```json
{ "id": "my_graph", "wire_version": "0.3.0", "content_hash": "...", "schema": {...} }
```

### `PUT /v1/graphs/{id}/schema`

Replace the schema. Body is the new `Schema` directly (no envelope).

The new schema must be **additively compatible** — no removed node
or edge types, no removed properties, no changed property types,
no edge-endpoint narrowing, no optional → required-without-default
transitions. All violations are reported in one response (not
first-wins) so callers see the full incompat set.

Returns `200` + new `SchemaResponse` (with the new `content_hash`)
on success. Returns `400` with all violations joined by `; ` on
incompatible changes:

```
schema evolution rejected: removed node type: Item; changed edge property type: Likes.weight Float -> String
```

Validation + persist + in-memory swap run under one engine
write-lock; concurrent observers never see a torn `(schema, hash)`
pair, and on `OnDisk` a disk-write failure leaves in-memory state
untouched.

Errors: `400` evolution rejected; `404` graph not found.

## Nodes

### `POST /v1/graphs/{id}/nodes`

```json
{ "node_type": "Item", "node_id": "n1", "properties": { "name": "widget" } }
```

Schema defaults are applied on write (e.g. a property with
`default: "standard"` is materialized when the request omits it).

Returns `201` + `NodeResponse`:

```json
{ "node_type": "Item", "node_id": "n1", "properties": { "name": "widget", "tier": "standard" } }
```

Errors: `400` validation / unknown type / missing required;
`404` graph not found.

### `GET /v1/graphs/{id}/nodes?type=X[&prop=Y&value=Z]`

List nodes of `type=X`, optionally filtered by an indexed property.
The `(prop, value)` pair must come together — half-supplied is a
`400`.

`value` is coerced to the schema-declared `PropertyType`. Coerce
failures (e.g. `value=abc` against an `int` property) are `400`,
not silent empty results.

Returns `200` + `NodeListResponse`:

```json
{ "nodes": [ {...}, {...} ] }
```

Filter requires `indexed: true` on the property. `Float` and
`list:string` aren't indexable; filtering by them is `400`.

### `GET /v1/graphs/{id}/nodes/{type}/{node_id}`

Returns `NodeResponse`. `404` if the node doesn't exist; `400` if
the type isn't in the schema.

### `PUT /v1/graphs/{id}/nodes/{type}/{node_id}`

REPLACE the property map. The new map is the complete new state;
properties absent from the body are dropped (subject to schema
defaults re-applying). PATCH is not exposed; storage's
`replace_node_properties` primitive is REPLACE-only.

```json
{ "properties": { "name": "gadget" } }
```

Returns `200` + `NodeResponse`. `404` if the node is missing.

### `DELETE /v1/graphs/{id}/nodes/{type}/{node_id}`

Cascades: drops the node's outgoing + incoming edges (with peer-
adjacency cleanup), drops the sidecar embedding (if any), removes
the node from the per-(graph, type) HNSW index (if any).

Returns `204` if the node existed, `404` if it didn't.

## Edges

### `POST /v1/graphs/{id}/edges`

```json
{
    "edge_type": "Likes",
    "from_type": "Item", "from_id": "a",
    "to_type":   "Item", "to_id":   "b",
    "properties": { "weight": 0.5 }
}
```

`from_type`/`to_type` are validation arguments (passed to
`validate_edge`); they're not part of the edge identity (URL keys
are `(edge_type, from_id, to_id)`).

Returns `201` + `EdgeResponse`:

```json
{ "edge_type": "Likes", "from_id": "a", "to_id": "b", "properties": {"weight": 0.5} }
```

### `GET /v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}`

Returns `EdgeResponse`. `404` if missing.

### `PATCH /v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}`

MERGE the property map into the existing edge — properties absent
from the body are preserved. PATCH (not PUT) is the verb here; it
mirrors storage's `merge_edge_properties` primitive.

```json
{ "properties": { "weight": 0.9 } }
```

Returns `200` + `EdgeResponse`. `404` if missing.

### `DELETE /v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}`

Returns `204` / `404`.

## Embeddings (sidecar)

Embeddings are managed via dedicated routes rather than riding on
node `properties`. The storage representation is one column family
keyed by `(graph_id, node_type, node_id)` → raw f32-LE bytes.

### `PUT /v1/graphs/{id}/nodes/{type}/{node_id}/embedding`

```json
{ "embedding": [0.1, 0.2, 0.3, ...] }
```

Sets the embedding AND inserts into the per-(graph, type) HNSW
index. The first insert per type fixes the index dimension; later
PUTs with mismatched dim are `400 EmbeddingDimMismatch`.

The node must exist (`404` otherwise). Empty embeddings are `400`.

Returns `200` + `EmbeddingResponse`:

```json
{ "node_type": "Item", "node_id": "n1", "embedding": [0.1, 0.2, 0.3] }
```

### `GET /v1/graphs/{id}/nodes/{type}/{node_id}/embedding`

Returns `200` + `EmbeddingResponse`. `404` if no embedding has been
set (regardless of whether the node itself exists; the route
treats both as a single "no embedding here" case).

### `DELETE /v1/graphs/{id}/nodes/{type}/{node_id}/embedding`

Removes from storage AND HNSW. `204` if existed; `404` if not.
`delete_node` cascades through this path automatically.

## Similarity

### `POST /v1/graphs/{id}/similar`

```json
{ "embedding": [0.1, 0.2, 0.3], "top_k": 10, "node_type": "Item" }
```

`node_type` is required — per-type indexes can have different
dimensions, so a merged "search all types" answer is ambiguous
about score comparability. Cross-type search would be a separate
route when a real consumer asks.

`embedding.len()` must match the indexed dim (`400` otherwise).
`top_k` must be `> 0`.

Returns `200`:

```json
{
    "results": [
        { "node_id": "alice", "score": 0.95 },
        { "node_id": "bob",   "score": 0.87 }
    ]
}
```

If the type exists in the schema but no embedding has been set
yet, returns `200` with an empty `results` array — the type is
honest, just no data to search. Schema-unknown type is `400`.

## Wire-format stability

The `wire_version` field on `SchemaResponse` and
`GraphMetadataResponse` is the foundation crate's `Cargo.toml`
version (e.g. `"0.3.0"`). Consumers compare it against a
compiled-in constant; mismatch should fail-fast (the consumer was
built against a different foundation version).

The `content_hash` field is SHA256 of the schema's canonical JSON
(via serde_json `to_value` → `BTreeMap`-ordered keys → byte-stable
hash). Consumers cache codegen output keyed by this hash;
mismatch means the schema has changed and code regen is needed.

Both fields are load-bearing — the drift-detection contract is
the same one storyflow's C-partial substrate uses. Don't strip
them on the consumer side.

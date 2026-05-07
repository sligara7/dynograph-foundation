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
| `dynograph_build_info` | gauge | `version`, `git_sha`, `git_dirty` | Always `1`. `version` is `CARGO_PKG_VERSION`. `git_sha` is the short HEAD sha at build time (`"unknown"` if built outside a git checkout). `git_dirty` is `"true"`/`"false"` for whether the working tree was clean when the binary was built. Same triple is also exposed as JSON at `GET /buildinfo`. |
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
{ "id": "my_graph", "wire_version": "0.5.5", "content_hash": "<sha256-hex>", "schema": {...} }
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
{ "id": "my_graph", "wire_version": "0.5.5", "content_hash": "...", "schema": {...} }
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

## Batch (atomic multi-op)

### `POST /v1/graphs/{id}/batch`

Run a list of node/edge mutations as one atomic transaction —
all-or-nothing. Mirrors the in-process write-lock atomicity that
single handlers can't compose across HTTP. Closes the dominant
gap from the storyflow audit (2026-05-04).

```json
{
    "ops": [
        {"op": "create_node", "node_type": "Item", "node_id": "a", "properties": {"name": "a"}},
        {"op": "create_edge", "edge_type": "Likes", "from_type": "Item", "from_id": "a",
                              "to_type": "Item", "to_id": "b", "properties": {"weight": 0.5}},
        {"op": "merge_edge",  "edge_type": "Likes", "from_id": "x", "to_id": "y", "properties": {"weight": 0.9}},
        {"op": "replace_node","node_type": "Item", "node_id": "x", "properties": {"name": "renamed"}},
        {"op": "delete_edge", "edge_type": "Likes", "from_id": "x", "to_id": "y"},
        {"op": "delete_node", "node_type": "Item", "node_id": "x"}
    ]
}
```

Field names per op variant match the existing single-handler
bodies — translate single calls into batch entries with no field
renames. `properties` is optional (defaults to `{}`).

**Atomicity:** entire batch runs under one engine write lock +
storage `begin_batch` / `commit_batch`. Any per-op failure
discards the batch (zero state changes) and returns `400` with a
structured JSON error identifying the failing op:

```json
{ "error": "node not found: Item/x", "op_index": 3, "op_type": "replace_node" }
```

This is the one place the service deviates from the plain-text
error convention — batch callers need `op_index` to locate the
failing op in their request.

**Success (200):**

```json
{
    "ops_applied": 6,
    "nodes_created": 1, "nodes_replaced": 1, "nodes_deleted": 1,
    "edges_created": 1, "edges_merged":   1, "edges_deleted": 1
}
```

**Limits:**

- Empty `ops` → `400 "ops must be non-empty"`.
- `ops.len() > 1000` → `400 "ops length N exceeds maximum 1000"`.

**Read-your-own-writes within a batch (v0.5.5+):**

Reads inside an active batch see the post-buffer view. Sequences
that compose naturally:

- `create_node X` then `replace_node X` — the replace sees the
  buffered create and updates it. Both apply at commit.
- `create_edge X→Y` then `merge_edge X→Y` — the merge sees the
  buffered create and rolls property changes on top.
- `create_edge X→Y` then `delete_node X` — the cascade sees the
  buffered edge and tombstones it; no orphan survives.

A discarded batch (`discard_batch()`) leaves the backend untouched —
the in-batch view never lands.

Pre-v0.5.5 the buffer was write-only. Hazard tests in
`crates/dynograph-service/tests/integration.rs` lock the new
semantics in (replace-after-create succeeds; cascade-after-create
cleans up).

## Resolve-or-create (fuzzy/vector entity resolution)

### `POST /v1/graphs/{id}/resolve-or-create`

Fuzzy/vector entity resolution with create-on-miss. Composes the
`dynograph-resolution` crate (token_sort_ratio + jaro_winkler with
cosine-similarity tiebreaker) over candidates fetched from storage.
Storyflow's LLM extraction funnels every Character through this
gate; the route exists to make the same path available without
embedding the foundation crates in-process.

```json
{
    "node_type": "Character",
    "properties": {"name": "Mira Sandgrove", "story_id": "X"},
    "embedding": [0.1, 0.2, ...],
    "scope": {"prop": "story_id", "value": "X"}
}
```

`embedding` and `scope` are optional. The query name is
`properties.name` (string, required) — same property the route
extracts from candidates for fuzzy comparison, so "name" lives in
one place.

**Returns 200:**

```json
{
    "id": "8a72...uuid",
    "was_created": true,
    "match_kind": "auto_merge" | "vector_merge" | "created_new"
}
```

`match_kind` distinguishes auto-merge (fuzzy ≥ auto_merge_threshold)
from vector-merge (fuzzy in [fuzzy_threshold, auto_merge_threshold)
+ embedding cosine ≥ vector_threshold) — useful for storyflow-side
threshold tuning. `id` is the existing node's id on either merge,
or a fresh UUIDv4 on create.

**400 on any of these (pre-flight, no state changes):**

- `node_type` unknown to schema
- `node_type` has no `resolution` block declared in schema (no
  silent fallback to defaults — explicit-is-better)
- `properties.name` missing or non-string
- `scope.prop` not declared on the node type, OR not declared with
  `indexed: true` (otherwise the candidate scan silently returns
  empty, which would mask a misconfiguration as "everything was a
  new entity")
- `embedding` empty
- `embedding.len()` ≠ existing HNSW index dim for this type

**Atomicity:** the whole call runs under one engine write lock so
candidate scan + resolve + (create + set_embedding + HNSW insert)
compose. The CreateNew path is currently sequential, not batched —
v0.5.5+ buffer-aware reads would make it safe to wrap in a batch
(`set_embedding`'s existence check sees the buffered create), but
that refactor is a follow-up. Today, pre-flight handles every
checkable failure; only a pure storage-I/O fault between the
node-create and embedding-set can tear the pair, which is
caller-retry-safe (a retry will auto-merge to the just-created node
and `set_embedding` again).

**Aliases** (mentioned in the audit memo) are not supported in v1 —
the underlying resolver doesn't natively accept multi-name queries,
and orchestrating them at the HTTP layer would re-implement the
threshold logic. Extend the resolver crate properly if a real
workload needs them.

## Edges (fan-out collection)

### `POST /v1/graphs/{id}/edges:collect`

Walk a typed source set + collect their outgoing edges of selected
types in one call. Replaces the per-node `outgoing_edges`
round-trip pattern that consumers (e.g. storyflow's
`collect_story_edges`) use to project a graph slice for relationship
listings, hierarchy queries, and algorithm input. Read-only — single
`with_engine_read` lock for the whole scan.

```json
{
    "source": {
        "type": "*" | "Character" | ["Character", "Event"],
        "filter": {"prop": "story_id", "value": "X"}
    },
    "edge_types": ["MENTIONS", "EXPLORES"],
    "format": "edges" | "adjacency",
    "resolve_target": false,
    "limit": 200
}
```

`source.type` is required and accepts a single name, an array of
names, or `"*"` (every type in the schema). `source.filter` is
optional; if present, `prop` must be declared `indexed: true` on
every covered source type. `edge_types` is required and non-empty;
each entry must be in the schema. `limit` is required, in
`1..=10_000`. `format` and `resolve_target` are optional; defaults
shown.

**Returns 200, shape `{edges: [...], truncated: bool}` for `format = "edges"`:**

```json
{
    "edges": [
        {
            "edge_type": "MENTIONS",
            "from_type": "Character",
            "from_id": "char-1",
            "to_id":   "char-2",
            "properties": {"weight": 0.5},
            "target": {                              // present iff resolve_target=true
                "node_type": "Character",
                "node_id": "char-2",
                "properties": {"name": "Bob", ...}
            }
        }
    ],
    "truncated": false
}
```

**Returns 200, shape `{adjacency: {from_id: [...]}, truncated: bool}` for `format = "adjacency"`:**

```json
{
    "adjacency": {
        "char-1": [
            {"edge_type": "MENTIONS", "to_id": "char-2", "properties": {...}}
        ]
    },
    "truncated": false
}
```

Adjacency entries omit `from_id` (it's the key) and `from_type`
(uniform across a source-typed scan). Use this format for client-
side projections (pagerank/louvain/shortest-path).

`truncated` is `true` when the scan stopped early because `limit`
was reached. Iteration order is unspecified — depends on schema
HashMap iter order across types and storage scan order within each
type. Algorithm consumers don't care about order; "show me a few
sample edges" callers shouldn't either with a `limit` cap.

**400 on any of these (pre-flight, no scans on failure):**

- `edge_types` empty
- `edge_types` references an unknown edge type
- `source.type` references an unknown node type (single, list, or
  post-wildcard expansion)
- `limit` outside `1..=10_000`
- `source.filter.prop` not declared as `indexed: true` on every
  covered source type

**Two cost notes worth knowing:**

- **No edge-type-prefix scan in storage.** Adjacency CFs are keyed
  `(graph_id, node_id, edge_type, peer_id)` — by node, not edge
  type. Wildcard `source.type` without filter walks O(N) prefix
  scans (one per node). Filter-narrowed sources use the indexed
  fast scan. A future CF
  `(graph_id, edge_type, from_id, to_id)` could enable true
  edge-type scans; out of scope until profiling shows a real
  bottleneck.
- **`StoredEdge` carries no `to_type`.** With `resolve_target=true`,
  foundation discovers each target's type by walking the schema's
  `EdgeTypeDef.to` candidates (`Single` → 1 lookup; `Multiple` →
  try each; `Wildcard` → try every node type). Bounded by the
  schema; wildcard endpoints pay full per-edge fanout.

When the candidate walk finds nothing for a given edge,
`target` is omitted from the response — the edge is an orphan
(referential-integrity gap), and absence is the caller-visible
signal.

## Traversal (typed BFS)

### `POST /v1/graphs/{id}/traverse`

Typed BFS over one or more edge-type steps from a single start
node. Backs the `compute_predecessors` shape — transitive walk
along a single edge type from a start node, scoped by some node
property — and its multi-step generalization. Read-only; the whole
BFS runs under one `with_engine_read` lock.

```json
{
    "start": {"type": "NarrativeEpoch", "id": "epoch-X"},
    "traverse": [
        {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
    ],
    "scope":  {"prop": "story_id", "value": "Y"},
    "return": "ids" | "nodes",
    "limit":  1000
}
```

`start.type` must be in the schema; `start.id` non-empty. `traverse`
is a non-empty array (max 10 steps). Each step has:

- `edge_type` — must be in the schema
- `direction` — `"outgoing"` (walk `from→to`), `"incoming"` (walk
  `to→from`), or `"both"` (union)
- `transitive` (optional, default `false`) — when true, the step's
  edges are walked BFS-style until exhausted before advancing to
  the next step; when false, exactly one hop per visited node

`scope` (optional) restricts traversal to nodes whose `prop` equals
`value`. Same indexed-property guarantee as `/edges:collect`:
`scope.prop` must be declared `indexed: true` on the start type AND
on every per-step candidate type.

`return` (optional, default `"ids"`) controls whether response
nodes carry `properties`; `"nodes"` includes them, `"ids"` omits
them via `skip_serializing_if`.

`limit` is required, in `1..=10_000`.

**Returns 200:**

```json
{
    "nodes": [
        {"node_type": "NarrativeEpoch", "node_id": "epoch-X1"},
        {"node_type": "NarrativeEpoch", "node_id": "epoch-X2"}
    ],
    "truncated": false
}
```

`truncated` is `true` when the BFS stopped early because `limit`
was reached. Iteration order is unspecified.

**Two semantic decisions worth knowing:**

- **Start is never in `nodes`.** Caller already has the start id;
  mirrors `compute_predecessors`'s "predecessors of X (not
  including X)" shape. The visited set still holds start so a
  cycle back through it short-circuits.
- **Start not found in storage → 404.** Loud failure per the
  no-silent-fallback principle. Start exists but fails the scope
  filter → `200` with empty `nodes` (legitimate "no matches", not
  a misconfiguration).

**Two dedup sets, two purposes:**

- **`emitted` keyed by `(node_type, node_id)`** — result-set dedup
  with UNION semantics across steps (closer to SQL UNION than
  JOIN): a peer reached at step 0 stays in results even if step
  1's edges would also visit it.
- **`queued_steps` keyed by `(node_id, step_idx)`** — work-item
  cycle guard. The same id at a different `step_idx` is a
  different work item, which is how multi-step chains advance.

**Transitive fan-out semantics:** a `transitive: true` step
re-enqueues each peer at BOTH `step_idx` (continue transitively)
AND `step_idx + 1` (advance to the next step, if any). Without
the latter, a chain like `[transitive A, then B]` would starve
step B. `transitive: false` re-enqueues only at `step_idx + 1`.

**400 on any of these (pre-flight, no scans on failure):**

- Unknown `start.type`; empty `start.id`
- Empty `traverse`; chain length > 10
- Unknown `edge_type` in any step
- `limit` outside `1..=10_000`
- When `scope` is supplied: `scope.prop` not declared
  `indexed: true` on the start type AND every per-step candidate
  type

**Performance notes:**

- Per visit: one `scan_outgoing_edges` and/or `scan_incoming_edges`
  (storage prefix-scans on `(graph, node)` and post-filters on
  `edge_type` in memory — no edge-type prefix CF exists today)
  plus one `get_node` per peer to resolve type + read properties
  for the scope check.
- A `peer_cache` keyed by `peer_id` collapses repeat lookups on
  the same id across transitive iterations and across multiple
  sources pointing at the same target. Cache hits are validated
  against the current candidate set so `direction: "both"` with
  different per-direction candidates stays correct.
- Cost is `O(unique_peers × avg_candidates + visits)`, capped by
  `limit`.

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
version (e.g. `"0.5.5"`). Consumers compare it against a
compiled-in constant; mismatch should fail-fast (the consumer was
built against a different foundation version).

The `content_hash` field is SHA256 of the schema's canonical JSON
(via serde_json `to_value` → `BTreeMap`-ordered keys → byte-stable
hash). Consumers cache codegen output keyed by this hash;
mismatch means the schema has changed and code regen is needed.

Both fields are load-bearing — the drift-detection contract is
the same one storyflow's C-partial substrate uses. Don't strip
them on the consumer side.

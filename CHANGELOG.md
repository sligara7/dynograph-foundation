# Changelog

Notable changes to `dynograph-foundation`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com); versions match the
workspace `version` in `Cargo.toml`.

## v0.3.0 — 2026-04-27

The "embedded → service" release. Foundation gains an HTTP service,
an async client crate, a Docker image, and a sidecar embedding
store. A handful of correctness fixes in storage and the vector
index landed at the end of the cycle (TD-1/2/3).

### Added

- **`dynograph` HTTP service** (`crates/dynograph-service`) with
  multi-graph `GraphRegistry`, RocksDB persistence + restart
  rehydration, node/edge CRUD under `/v1/graphs/{id}`, schema
  split (`POST /v1/graphs`, `PUT /v1/graphs/{id}/schema`) with
  additive-only evolution enforcement, `/ready` + `/metrics`
  (Prometheus), and pluggable auth (`NoAuth` / `BearerJwt`).
- **`dynograph-client`** async Rust HTTP client crate (`reqwest` +
  `rustls-tls`) covering every `/v1/*` route.
- **Sidecar embedding store** + **HNSW similarity search**
  exposed as `POST /v1/graphs/{id}/similar`. Embeddings cascade
  with their owning node on delete.
- **Docker image** + release artifacts (`ghcr.io/sligara7/dynograph-foundation:0.3.0`).
- **`docs/migration.md`** — embedded → service playbook.

### Changed (behavioral)

- **`Storage::delete_node` now cascades** to incident edges *and*
  peer-side adjacency entries (TD/C1). Previously it left
  dangling edges that `get_edge` would still resolve and that
  `scan_incoming_edges` on the peer would still return.
- **`update_node_properties` → `replace_node_properties`** and
  **`update_edge_properties` → `merge_edge_properties`** (S1).
  The rename makes REPLACE-vs-MERGE semantics explicit at the
  storage layer. Behavior of the underlying calls is unchanged;
  only the names moved.
- **`validate_node` now takes `&mut`** because it applies schema
  defaults inline (C3). Callers passing `&` need a one-character
  bump.
- **`commit_batch` is atomic across deletes too** (C4). Mixed
  put/delete batches no longer split into two rocksdb writes.
- **Datetime property validation** is now strict per RFC 3339 (C2).

### Fixed (TD-1/2/3, post-tag tightening)

- **HNSW correctness + perf** (TD-1) — vector index returned
  approximate neighbors that occasionally missed exact matches at
  small `M`; bound + heuristics corrected.
- **Storage cache + adjacency + scan-decode + lifecycle** (TD-2) —
  several edge cases around cache invalidation on delete and
  msgpack decode reuse on hot-path scans.
- **Schema + resolver tightening** (TD-3) — surfaces stricter
  errors on malformed schema input rather than silent partial
  acceptance.

### Migration notes

#### `delete_node` cascade

If you have code that **deletes a node and immediately creates a
new node with the same id** (typically as a workaround for
"update properties"), replace the pair with
`replace_node_properties`:

```rust
// Before — silently relied on edges hanging around through the
// delete+recreate gap. With v0.3.0's cascading delete those
// edges are now correctly destroyed, so this pattern drops every
// edge attached to the node.
storage.delete_node(graph_id, "Item", id)?;
storage.create_node(graph_id, "Item", id, new_props)?;

// After — full property replacement, edges + adjacency
// untouched.
storage.replace_node_properties(graph_id, "Item", id, new_props)?;
```

If you actually want partial-update / merge semantics on a node,
do a `get_node` + caller-side merge + `replace_node_properties`
round-trip. (Edges have native `merge_edge_properties`; nodes
deliberately don't, to keep the storage layer's REPLACE-vs-MERGE
distinction explicit.)

#### Method renames

```text
update_node_properties → replace_node_properties
update_edge_properties → merge_edge_properties
```

Mechanical rename; no semantic change.

#### `validate_node` signature

```rust
// Before
schema.validate_node(node_type, &props)?;
// After
schema.validate_node(node_type, &mut props)?;
```

The mutation is schema defaults being applied inline.

## v0.2.1

- `feat(schema)`: optional `description` field on `PropertyDef`.

## v0.2.0

- Initial public-ish baseline of the foundation crates
  (`dynograph-core`, `dynograph-storage`, `dynograph-resolution`,
  `dynograph-vector`). Embedded-only; no service.

## v0.1.0

- Initial workspace skeleton.

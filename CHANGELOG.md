# Changelog

Notable changes to `dynograph-foundation`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com); versions match the
workspace `version` in `Cargo.toml`.

## v0.5.1 — 2026-05-05

First of four primitives identified by the storyflow→foundation
audit (2026-05-04). Closes the dominant atomicity gap: every
multi-write storyflow handler today depends on the in-process
write lock making the sequence atomic, which doesn't survive the
move to HTTP.

### Added

- **`POST /v1/graphs/{id}/batch`** — atomic multi-op transaction.
  Body `{"ops": [...]}` accepts any combination of `create_node`,
  `replace_node`, `delete_node`, `create_edge`, `merge_edge`,
  `delete_edge` ops; field shapes mirror the existing
  single-handler request bodies. Whole batch runs under one
  engine write lock + storage `begin_batch` / `commit_batch`.
  All-or-nothing: any per-op failure discards the batch and
  returns `400` with a structured JSON error
  (`{error, op_index, op_type}`) identifying the failing op —
  the one place the service deviates from the plain-text error
  convention, because batch callers need the index to debug a
  partial rejection. Success returns 200 + per-kind counts +
  `ops_applied`. Soft cap: 1000 ops/batch (audit's heaviest
  known case is ~67). Empty `ops` and `> 1000 ops` are both 400.

  Two storage-layer constraints documented in
  `crates/dynograph-service/src/batch.rs` module doc and locked
  in by hazard tests in `tests/integration.rs`:

  1. *No read-your-own-writes within a batch.* The engine batch
     buffer is write-only — `engine.put()` buffers but
     `engine.get()` reads the backend. Ops whose precondition is
     a `get()` (`merge_edge`, `replace_node`, `delete_*`) see
     pre-batch state. `create_node X` then `replace_node X` in
     one batch fails with "node not found" → rollback.
  2. *Cascade-delete misses in-batch creates.* `delete_node X`
     in the same batch as `create_edge X→Y` leaves an orphaned
     edge — cascade reads pre-batch adjacency.

  Neither blocks any audit-enumerated workload (`integrate_fragment`
  and friends are all-creates or modifications-of-pre-existing).
  Lifting either would require a buffer-aware `engine.get()`;
  out of scope for this release.

  PR: [#3](https://github.com/sligara7/dynograph-foundation/pull/3).
  Storyflow side-B acceptance gate: `mutation.integrate_fragment_atomic`
  (storyflow commit `37b34717`) — must stay green when storyflow
  rewrites `integrate_fragment` to call `/batch`.

## v0.5.0 — 2026-05-04

Automation + safety release. Locks down the drift classes that bit
the v0.3.x line, ships build provenance so deployments are
self-identifying, and finishes the public-enum non-exhaustive pass
v0.4.0 started.

### Breaking

- **`Value`, `PropertyType`, `EdgeEndpoint` are now `#[non_exhaustive]`.**
  Same discipline v0.4.0 applied to `DynoError`. External callers
  doing exhaustive `match` on any of these need to add a wildcard arm
  (`_ => …`). Internal patterns within `dynograph-core` are unaffected.
  Future variant additions to these enums no longer require a major
  bump.

### Added

- **`GET /buildinfo`** — JSON build provenance:
  ```json
  {"version": "0.5.0", "git_sha": "abc1234", "git_dirty": false, "uptime_seconds": 142.391}
  ```
  Public endpoint, sibling of `/metrics`/`/health`/`/ready`.
- **`dynograph_build_info` gauge** gains `git_sha` and `git_dirty`
  labels. After v0.5.0, "what code is running on this host?" is one
  curl: `curl /metrics | grep build_info` or `curl /buildinfo`.
- **GHCR publish workflow** (`.github/workflows/release.yml`) runs on
  every `v*` tag push, builds the Docker image, and pushes to
  `ghcr.io/sligara7/dynograph-foundation:${tag}` + `:latest`. README
  and `docs/service.md` now lead with the docker-pull example.
- **CI: doc-version-drift guard** (`scripts/check-doc-versions.sh`).
  Mechanically catches the v0.3.x drift class — Cargo.toml advances
  but README/docs still advertise the previous tag.
- **CI: `cargo-deny`** — security advisories (RUSTSEC), license
  compliance, dup-version detection, unknown-registry detection.
  Config in `deny.toml`.
- **CI: `typos`** — spell-checks source/docs/comments.
- **CI: `msrv-check`** job — builds against rust 1.94 (the declared
  `rust-version`). Catches drift where we accidentally use a feature
  stabilized after MSRV, or a transitive dep silently raises its own.

### Changed

- **CI stable toolchain pinned to 1.95.0** (was `@stable`, the moving
  pointer). The `clippy::unnecessary-sort-by` expansion in 1.95 broke
  CI on the v0.4.0 release branch with no source change. Bump
  `RUST_TOOLCHAIN` env var deliberately when reviewing rustc release
  notes.
- **`SEMVER_BASELINE_REV` bumped v0.3.1 → v0.4.0.** v0.4.0 is now the
  most-recent compilable, properly-versioned release tag.

## v0.4.0 — 2026-05-04

Cleanup release. Bumps minor (in 0.x convention) to honor a breaking
change shipped under v0.3.2's patch tag.

### Breaking

- **`DynoError` is now `#[non_exhaustive]`.** Callers doing exhaustive
  match on `DynoError` must add a wildcard arm (`_ => …`). This formalizes
  the v0.3.2 addition of `DynoError::EdgeValidation` (which itself was a
  breaking change shipped under a patch bump) and prevents future variant
  additions from repeating the same semver mistake.

### Fixed

- **Workspace compiles again.** v0.3.2 added `DynoError::EdgeValidation`
  but did not extend the exhaustive match in
  `dynograph-service::registry::status_for_dyno_error`, so the v0.3.2
  tag did not build. `EdgeValidation` is now mapped to `400 Bad Request`
  alongside the other client-validation variants.
- **`cargo fmt --check` passes again.** Edge-validation code in
  `dynograph-core/src/schema.rs` shipped unformatted in v0.3.2.
  Reformatted.
- **CI `semver-checks` job actually runs.** The previous setup invoked
  `cargo-semver-checks-action@v2` with no baseline configuration, which
  defaults to crates.io — but no foundation crate is published there,
  so the job failed on every PR with `not found in registry`. Replaced
  with a manual `cargo semver-checks --baseline-rev v0.3.1` invocation
  (v0.3.1 = most recent compilable tag).

### Changed

- **Workspace `version` advances `0.3.0` → `0.4.0`.** `Cargo.toml` was
  frozen at `"0.3.0"` across v0.3.0/0.3.1/0.3.2, so binaries from any
  of those tags self-reported `wire_version` as `"0.3.0"` regardless of
  the commit. The `Cargo.toml` version is now the single source of
  truth and tracks each release tag.
- **README + `docs/*` rewritten** for accuracy. Removed references to a
  published GHCR image (no such image exists; consumers build locally
  from this repo's `Dockerfile`).

## v0.3.2 — 2026-04-30

### Fixed

- **`engine::create_edge` now validates the property bag** against the
  edge type's declared properties. Until v0.3.2, edge endpoint validation
  ran but property validation was skipped — required properties could be
  missing, enum values could fall outside the declared set, and the
  handler still returned `200`. Surfaced by storyflow's `SUBTEXT_OF`
  lifecycle probe returning HTTP 200 on `relationship_type="totally_made_up"`.
- New `DynoError::EdgeValidation { edge_type, property, message }`
  variant so edge-property failures name the offending edge instead of
  overloading the node-scoped `Validation`.
- `Schema::validate_edge_properties(edge_type, &mut HashMap)` mirrors
  `validate_node`'s shape — applies declared defaults, enforces
  required-presence, validates each value.

> **Note:** the v0.3.2 tag does not compile in `dynograph-service`
> (missing match arm; fixed in v0.3.3). Library-only consumers of
> `dynograph-core` / `dynograph-storage` are unaffected.

## v0.3.1 — 2026-04-27

### Fixed

- **`PropertyDef`, `NodeTypeDef`, `EdgeTypeDef` are now externally
  constructible** via `..Default::default()` syntax. v0.3.0's
  `#[non_exhaustive]` annotation prevented external struct-literal
  construction; discoverable only on consumer attempt. Drops
  `#[non_exhaustive]` from those three structs and adds `Default`
  derives. `Schema` and `ResolutionConfig` keep `#[non_exhaustive]`
  (they enter via deserialization, not struct literals).
- `PropertyType` gains `#[derive(Default)]` with `String` as the
  default (dominant type in real schemas). `EdgeEndpoint::default()`
  returns `Single("*")`.
- Strictly additive on the wire: serde shape unchanged,
  `content_hash` unchanged.

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
- **Docker image** built from the in-tree `Dockerfile` / `docker-compose.yml`. No published image; build locally.
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

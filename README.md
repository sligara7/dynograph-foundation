# dynograph-foundation

Schema-driven graph foundation with entity resolution and HNSW vector
similarity. Ships as both a Rust library you embed and a deployable
HTTP service. Domain-neutral by design — consumers add their own
schemas, extractors, and authorization on top.

## Crates

| Crate | Role |
|---|---|
| `dynograph-core` | Schema model: `Schema`, `NodeTypeDef`, `EdgeTypeDef`, `Value`, `DynoError`. YAML / JSON schema parsing + validation. |
| `dynograph-storage` | RocksDB-backed node/edge persistence. Schema validation, MessagePack serialization, sidecar embedding store, atomic batch writes. |
| `dynograph-resolution` | Three-tier entity resolution: fuzzy match → vector tiebreaker → new-entity. |
| `dynograph-vector` | f32 vector ops + HNSW (hierarchical navigable small world) approximate nearest-neighbor index. |
| `dynograph-service` | axum HTTP service over the above. Multi-graph `GraphRegistry`, pluggable `AuthProvider`, `/v1/*` REST API, `/metrics` Prometheus. |
| `dynograph-client` | Async HTTP client wrapping the `/v1/*` API. Reuses `dynograph-core` types; thin reqwest-based wrapper. |

## As a service

```bash
docker compose up
curl http://localhost:8080/health   # → ok
curl http://localhost:8080/ready    # → ready
curl http://localhost:8080/metrics  # → Prometheus text
```

Or point the published image at a persistent volume:

```bash
docker run --rm -p 8080:8080 -v dynograph-data:/data \
    -e DYNOGRAPH_STORAGE_ROOT=/data \
    ghcr.io/sligara7/dynograph-foundation:0.3.0
```

Config (`dynograph.toml`) covers HTTP bind, RocksDB root, and the
auth provider (`noauth` or `bearer_jwt`). See
[`dynograph.example.toml`](dynograph.example.toml) and
[`docs/service.md`](docs/service.md) for the full surface.

## As a library

```toml
[dependencies]
dynograph-core    = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.3.0" }
dynograph-storage = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.3.0" }
```

```rust
use dynograph_core::Schema;
use dynograph_storage::StorageEngine;

let schema = Schema::from_yaml(include_str!("schema.yaml"))?;
let mut engine = StorageEngine::new_in_memory(schema);
engine.create_node("graph1", "Person", "alice", properties)?;
```

The Rust HTTP client (for talking to a running `dynograph` service):

```toml
[dependencies]
dynograph-client = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.3.0" }
```

```rust
use dynograph_client::DynographClient;

let client = DynographClient::new("http://localhost:8080")
    .with_bearer(jwt_token);
let metadata = client.get_graph("g1").await?;
```

## Build

```
cargo build --workspace
cargo test --workspace
```

Cold first build pulls in RocksDB (heavy C++ compile, ~10 minutes
inside Docker, ~5 minutes locally). Incremental rebuilds are
sub-second. MSRV 1.94 — see `rust-version` in `Cargo.toml`.

## Public surface

The `pub use` block at the top of each crate's `lib.rs` is the
stable contract. Internals are free to change between minor
versions; foundation follows semver.

## Docs

- [`docs/service.md`](docs/service.md) — running the binary, config, deployment patterns
- [`docs/api.md`](docs/api.md) — REST endpoint reference (all `/v1/*` routes)
- [`docs/migration.md`](docs/migration.md) — moving an embedded consumer to the HTTP service

## License

MIT — see [LICENSE](LICENSE).

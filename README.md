# dynograph-foundation

Rust workspace: schema-driven graph storage on RocksDB, HNSW vector
search, and an HTTP service over both. Usable as embedded library
crates or as the `dynograph` binary.

## Crates

| Crate | Role |
|---|---|
| `dynograph-core` | Schema model (`Schema`, `NodeTypeDef`, `EdgeTypeDef`, `Value`, `DynoError`); YAML/JSON parse + validate. |
| `dynograph-storage` | RocksDB-backed node/edge persistence; sidecar embedding store; atomic batches. |
| `dynograph-resolution` | Three-tier entity resolution: fuzzy → vector tiebreaker → new. |
| `dynograph-vector` | f32 vector ops + HNSW index. |
| `dynograph-service` | axum HTTP service; multi-graph registry; pluggable auth; `/v1/*` REST + `/metrics`. |
| `dynograph-client` | Async `reqwest` client for `/v1/*`. |

No crates are published to crates.io; consume by git tag (below).

## Build

```
cargo build --workspace
cargo test  --workspace
```

MSRV 1.94. First build compiles vendored RocksDB (~5–10 min);
incremental rebuilds are sub-second.

## Run the service

The `dynograph` binary lives in `dynograph-service`. Three ways to
run it:

```bash
# pull published image
docker run --rm -p 8080:8080 -v dynograph-data:/data \
    -e DYNOGRAPH_STORAGE_ROOT=/data \
    ghcr.io/sligara7/dynograph-foundation:0.5.1

# native
cargo run --release --bin dynograph -- --config dynograph.example.toml

# build via docker compose (from a checkout)
docker compose up

curl http://localhost:8080/health     # ok
curl http://localhost:8080/ready      # ready
curl http://localhost:8080/metrics    # Prometheus text
curl http://localhost:8080/buildinfo  # JSON: version + git SHA
```

Published images are tagged per release (`:0.5.1`) plus `:latest`.
The publish workflow (`.github/workflows/release.yml`) runs on every
`v*` git tag push.

CLI: `dynograph [--config <path>]`. Env-var overrides:
`DYNOGRAPH_BIND`, `DYNOGRAPH_STORAGE_ROOT`, `RUST_LOG`. Defaults:
in-memory storage, `127.0.0.1:8080`, `noauth`. See
[`dynograph.example.toml`](dynograph.example.toml).

## Use as a library

Git dependency on the latest tag (`v0.5.1`):

```toml
[dependencies]
dynograph-core    = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.5.1" }
dynograph-storage = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.5.1" }
```

```rust
use dynograph_core::Schema;
use dynograph_storage::StorageEngine;

let schema = Schema::from_yaml(include_str!("schema.yaml"))?;
let mut engine = StorageEngine::new_in_memory(schema);
engine.create_node("graph1", "Person", "alice", properties)?;
```

`dynograph-client` against a running service:

```toml
[dependencies]
dynograph-client = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.5.1" }
```

```rust
let client = dynograph_client::DynographClient::new("http://localhost:8080")
    .with_bearer(jwt_token);
let metadata = client.get_graph("g1").await?;
```

## Docs

- [`docs/service.md`](docs/service.md) — config, deployment, probes
- [`docs/api.md`](docs/api.md) — `/v1/*` REST reference
- [`docs/migration.md`](docs/migration.md) — embedded → service
- [`CHANGELOG.md`](CHANGELOG.md) — release notes

## License

MIT — see [LICENSE](LICENSE).

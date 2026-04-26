# dynograph-foundation

Shared, domain-agnostic foundation crates for graph-based projects.
A schema-driven graph engine with entity resolution and vector
similarity. Consumers add their own domain schemas, extractors, and
servers on top.

## Crates

| Crate | Role |
|---|---|
| `dynograph-core` | Schema model: `Schema`, `NodeTypeDef`, `EdgeTypeDef`, `Value`, `DynoError`. Runtime YAML schema parsing + validation. |
| `dynograph-storage` | RocksDB-backed node/edge persistence with schema validation, MessagePack serialization, and graph isolation. |
| `dynograph-resolution` | Three-tier entity resolution: fuzzy match, vector tiebreaker, or new-entity. |
| `dynograph-vector` | f32 vector storage with HNSW (hierarchical navigable small world) approximate nearest-neighbor index. |
| `dynograph-introspection` | Schema for an LLM's persistent self-knowledge graph: `Concept`, `Assumption`, `Correction`, `Blindspot`, `Attempt`. The empirical-self-knowledge layer; not a soul. |

## Public surface

The `pub use` block at the top of each crate's `lib.rs` is the
entire stable contract. Internals are free to change between minor
versions; `cargo-semver-checks` runs on pull requests to enforce this.

## Build

```
cargo build --workspace
cargo test --workspace
```

Cold first build pulls in RocksDB (heavy C++ compile, ~7 minutes).
Incremental rebuilds are sub-second.

## Use

```toml
[dependencies]
dynograph-core    = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.1.0" }
dynograph-storage = { git = "https://github.com/sligara7/dynograph-foundation.git", tag = "v0.1.0" }
```

## License

MIT — see [LICENSE](LICENSE).

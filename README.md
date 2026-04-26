# dynograph-foundation

Shared, domain-agnostic foundation crates for graph-based projects.
Consumers depend on these by path / git / version; domain-specific
schemas, extractors, and servers live in the consumer projects.

## Crates

| Crate | Role |
|---|---|
| `dynograph-core` | Schema model: `Schema`, `NodeTypeDef`, `EdgeTypeDef`, `Value`, `DynoError`. Runtime YAML schema parsing + validation. |
| `dynograph-storage` | RocksDB-backed node/edge persistence. |
| `dynograph-resolution` | Entity resolution (fuzzy + vector). |
| `dynograph-vector` | Vector embeddings, HNSW similarity search. |
| `dynograph-introspection` | Schema for an LLM's persistent self-knowledge graph: `Concept`, `Assumption`, `Correction`, `Blindspot`. The empirical-self-knowledge layer; not a soul. |

## Known consumers

- [ir2](https://github.com/sligara7/ir2) — system-of-systems engineering knowledge graph for graph-informed coding with Claude Code.
- (Planned) market_graph, cure, and the LLM-memory application currently at `~/project/dynograph/`.
- Storyflow currently maintains its own in-tree copy at `services/dynograph/`; eventual migration to depend on this foundation is decoupled.

## Build

```
cargo build --workspace
cargo test --workspace
```

Cold first build pulls in RocksDB (heavy C++ compile, ~7 minutes).
Incremental rebuilds are sub-second.

## Origin

Extracted 2026-04-25 from `/home/ajs7/project/storyflow/services/dynograph/` (the newer in-tree copy). The `dynograph-introspection` crate is a slimmed-down lift from `dynograph-self`'s schema layer — the empirical Memory layer (Concept / Assumption / Correction / Blindspot) only. The Soul layer (Value / Tension / Wonder / Regret) and the Creative layer (Hallucination / Pattern) were intentionally dropped; they belong to an application above the foundation, not in it.

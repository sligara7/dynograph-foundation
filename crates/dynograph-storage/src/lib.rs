//! DynoGraph Storage — RocksDB-backed graph storage engine.
//!
//! Provides schema-validated node and edge storage with:
//! - Column families for nodes, edges, adjacency lists, and indexes
//! - MessagePack serialization for compact storage
//! - Graph isolation (multiple graphs in one instance)
//! - Batch writes for extraction integration
//! - Iterator-based scans

mod backend;
mod cache;
mod engine;
mod keys;

pub use cache::{CacheConfig, ReadCache};
#[cfg(feature = "fulltext")]
pub use engine::FulltextHit;
pub use engine::{StorageEngine, StoredEdge, StoredNode};

// Types owned by dynograph-core that cross this crate's public boundary:
// `Schema` is taken by both constructors, `Value` is the type of every
// property in `StoredNode` / `StoredEdge`, and `DynoError` is the error of
// every fallible method. A caller could not name any of them without adding
// dynograph-core to its own manifest at a version nobody stated.
//
// Re-exported rather than wrapped: core is a required dependency and is the
// shared vocabulary every crate here speaks by design (`ifc:core-api`), so
// duplicating these types would fork the vocabulary rather than protect it.
// Contrast `FulltextHit`, which IS a new type because dynograph-text reaches
// consumers only through an optional feature. See
// `req:boundaries-own-their-types`.
pub use dynograph_core::{DynoError, Schema, Value};

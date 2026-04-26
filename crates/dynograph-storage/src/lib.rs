//! DynoGraph Storage — RocksDB-backed graph storage engine.
//!
//! Provides schema-validated node and edge storage with:
//! - Column families for nodes, edges, adjacency lists, and indexes
//! - MessagePack serialization for compact storage
//! - Graph isolation (multiple graphs in one instance)
//! - Batch writes for extraction integration
//! - Iterator-based scans

mod cache;
mod engine;
mod keys;

pub use cache::{CacheConfig, ReadCache};
pub use engine::{StorageEngine, StoredNode, StoredEdge};

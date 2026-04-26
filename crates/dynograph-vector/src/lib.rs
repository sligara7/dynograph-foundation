//! DynoGraph Vector — HNSW index and SIMD-accelerated similarity.
//!
//! Provides:
//! - f32 vector storage with configurable dimensionality
//! - HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index
//! - SIMD-accelerated cosine similarity
//! - Batch insert and search
//! - Snapshot persistence

pub mod distance;
pub mod hnsw;

pub use distance::{cosine_similarity, dot_product, euclidean_distance};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats, SearchResult};

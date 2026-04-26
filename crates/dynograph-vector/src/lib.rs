//! DynoGraph Vector — HNSW index, SIMD-accelerated similarity, and
//! basic statistics over numeric slices.
//!
//! Provides:
//! - f32 vector storage with configurable dimensionality
//! - HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index
//! - SIMD-accelerated cosine similarity (f32 primary path)
//! - f64 variants of distance/similarity for non-embedding consumers
//! - Element-wise (Hadamard) product
//! - Linear regression slope and Pearson correlation
//! - Batch insert and search
//! - Snapshot persistence

mod distance;
mod hnsw;
mod stats;

pub use distance::{
    cosine_similarity, cosine_similarity_f64, dot_product, dot_product_f64, euclidean_distance,
    euclidean_distance_f64, hadamard, hadamard_f64, l2_norm, l2_norm_f64,
};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats, SearchResult};
pub use stats::{linear_regression_slope, pearson_correlation};

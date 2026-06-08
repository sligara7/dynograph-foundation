//! DynoGraph Vector — HNSW index, SIMD-accelerated similarity, and
//! basic statistics over numeric slices.
//!
//! Provides:
//! - f32 vector storage with configurable dimensionality
//! - HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index
//! - SIMD-accelerated cosine similarity (f32 primary path)
//! - f64 variants of distance/similarity for non-embedding consumers
//! - Distances: cosine, dot, (squared) Euclidean, Manhattan
//! - Element-wise algebra: add, subtract, scale, negate, Hadamard
//!   product/division, element-wise power
//! - Vector transforms: L2 normalization, centroid
//! - Statistics: mean, variance/std-dev, median/percentile, softmax,
//!   linear-regression slope, Pearson + Spearman correlation
//! - Filtered, batch insert and search
//! - Snapshot persistence

mod distance;
mod hnsw;
mod stats;

pub use distance::{
    add, add_f64, centroid, centroid_f64, cosine_similarity, cosine_similarity_f64, dot_product,
    dot_product_f64, elementwise_power, elementwise_power_f64, euclidean_distance,
    euclidean_distance_f64, hadamard, hadamard_division, hadamard_division_f64, hadamard_f64,
    l2_norm, l2_norm_f64, l2_normalize, l2_normalize_f64, manhattan_distance,
    manhattan_distance_f64, negate, negate_f64, scale, scale_f64, squared_euclidean_distance,
    squared_euclidean_distance_f64, subtract, subtract_f64, validate_similarity_vector,
};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats, SearchResult};
pub use stats::{
    linear_regression_slope, mean, median, pearson_correlation, percentile, softmax,
    spearman_rank_correlation, std_dev, variance,
};

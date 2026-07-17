//! DynoGraph Cluster — domain-neutral, density-based clustering over a
//! precomputed distance matrix.
//!
//! The clustering sibling of `dynograph-vector` / `dynograph-graph` /
//! `dynograph-game`: pure, dependency-free math with no storage, service, or
//! core dependency. The caller supplies a precomputed N×N distance matrix and
//! runs [`dbscan`]; the result labels each point with a cluster id or noise.
//!
//! The input is a **precomputed distance matrix** — the caller computes the
//! distances (embeddings and vectors aren't this crate's to compute, the same
//! way it doesn't produce dimension vectors), so this is matrix in / labels
//! out, the same stateless shape as the `util/pairwise_*` family (which
//! produces exactly the kind of matrix DBSCAN consumes). It is **not** a
//! graph algorithm — it operates on points and their pairwise distances, not on
//! a stored graph's topology, which is why it lives beside the other
//! `util/*` pure-math endpoints rather than under `algo/*`.
//!
//! Scope is **DBSCAN** (density-based, finds arbitrarily-shaped clusters and
//! flags noise) — distinct from `dynograph-graph`'s Leiden (community
//! detection on graph edges) and clustering-coefficient. Other clustering
//! families (k-means, hierarchical/agglomerative) are out of scope until a
//! consumer demand-pulls them.
//!
//! Following the foundation's fail-loud principle, a malformed distance matrix
//! (empty, non-square, non-finite or negative entry) or parameter is rejected
//! via [`ClusterError`] rather than clustered half-specified.

mod dbscan;
mod error;

pub use dbscan::dbscan;
pub use error::ClusterError;

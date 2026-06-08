//! DynoGraph Graph — domain-neutral graph-theory algorithms over a generic,
//! densely-indexed in-memory graph.
//!
//! This crate is the graph-topology sibling of `dynograph-vector`: pure,
//! dependency-free math with no storage, service, or core dependency. The
//! caller builds a [`Graph`] via [`GraphBuilder`] (interning string node ids,
//! supplying finite `f64` edge weights and a directed/undirected flag) and runs
//! algorithms over it; results come back keyed by dense node index for the
//! caller to map back to ids.
//!
//! Consumer graphs are small (10^2-10^3 nodes), so every algorithm here is
//! **exact** — no approximate or streaming variants.
//!
//! Provides (PR A):
//! - [`Graph`] / [`GraphBuilder`] with self-loop and parallel-edge policies
//! - Connected / weakly-connected [`components`]
//! - Degree [`centrality`]
//!
//! Following the foundation's fail-loud principle, malformed input (e.g. a
//! non-finite edge weight) is rejected via [`GraphError`] rather than silently
//! coerced.

mod centrality;
mod components;
mod error;
mod graph;

pub use centrality::{degree_centrality, DegreeMode};
pub use components::{connected_components, Components};
pub use error::GraphError;
pub use graph::{Graph, GraphBuilder, ParallelEdgePolicy, SelfLoopPolicy};

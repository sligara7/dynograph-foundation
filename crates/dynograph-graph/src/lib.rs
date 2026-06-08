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
//! Provides:
//! - [`Graph`] / [`GraphBuilder`] with self-loop and parallel-edge policies
//! - Connected / weakly-connected [`components`](connected_components) and
//!   strongly-connected components ([`strongly_connected_components`])
//! - Centrality: [`degree_centrality`], [`pagerank`], [`personalized_pagerank`],
//!   [`eigenvector_centrality`], [`closeness_centrality`], [`betweenness_centrality`]
//! - Structure: articulation points & bridges ([`cut_structure`]), directed
//!   cycle detection ([`find_cycle`]), [`topological_sort`], clustering
//!   coefficient & transitivity ([`clustering`])
//! - Paths & prediction: single-pair [`shortest_path`], neighborhood-overlap
//!   link prediction ([`link_prediction_from`] / [`link_prediction_all`]),
//!   max-flow / min-cut ([`max_flow_min_cut`])
//!
//! Edge weights carry one of two meanings depending on the algorithm, and the
//! caller must supply the right kind:
//! - **strength** (higher = stronger tie) for degree / PageRank / eigenvector;
//! - **cost** (higher = farther) for the shortest-path measures closeness and
//!   betweenness, which require strictly positive weights.
//!
//! Following the foundation's fail-loud principle, malformed input (a non-finite
//! edge weight, a negative strength, a non-positive cost, non-convergence) is
//! reported via [`GraphError`] rather than silently coerced.

mod betweenness;
mod centrality;
mod closeness;
mod clustering;
mod components;
mod cuts;
mod cycles;
mod eigenvector;
mod error;
mod graph;
mod link_prediction;
mod maxflow;
mod pagerank;
mod paths;
mod scc;
mod shortest_path;
mod toposort;

pub use betweenness::betweenness_centrality;
pub use centrality::{DegreeMode, degree_centrality};
pub use closeness::closeness_centrality;
pub use clustering::{Clustering, clustering};
pub use components::{Components, connected_components};
pub use cuts::{Cuts, cut_structure};
pub use cycles::find_cycle;
pub use eigenvector::{EigenvectorConfig, eigenvector_centrality};
pub use error::GraphError;
pub use graph::{Graph, GraphBuilder, ParallelEdgePolicy, SelfLoopPolicy};
pub use link_prediction::{LinkPredictionMethod, link_prediction_all, link_prediction_from};
pub use maxflow::{MinCut, max_flow_min_cut};
pub use pagerank::{PageRankConfig, pagerank, personalized_pagerank};
pub use scc::strongly_connected_components;
pub use shortest_path::{Path, shortest_path};
pub use toposort::topological_sort;

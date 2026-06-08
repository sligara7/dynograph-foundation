//! Error type for graph construction and algorithms.
//!
//! Following the foundation's core principle (fail loudly, never swallow a
//! semantic failure), graph construction rejects malformed input rather than
//! silently coercing it. The service layer maps these to `400`s.

use std::fmt;

/// Things that can go wrong building or running over a [`Graph`](crate::Graph).
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// An edge weight was NaN or infinite. Weights must be finite so that
    /// distance/centrality accumulators stay well-defined.
    NonFiniteWeight {
        from: String,
        to: String,
        weight: f64,
    },
    /// An algorithm that requires at least one node was run on an empty graph.
    EmptyGraph,
    /// A power-iteration algorithm (PageRank, eigenvector centrality) failed to
    /// converge within the configured iteration budget. Surfaced rather than
    /// returning a half-converged vector that reads like a real answer.
    NotConverged { iterations: usize },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NonFiniteWeight { from, to, weight } => write!(
                f,
                "edge ({from} -> {to}) has non-finite weight {weight}; weights must be finite"
            ),
            GraphError::EmptyGraph => {
                write!(f, "algorithm requires at least one node but the graph is empty")
            }
            GraphError::NotConverged { iterations } => {
                write!(f, "iterative algorithm did not converge within {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for GraphError {}

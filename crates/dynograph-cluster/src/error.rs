//! Error type for clustering input validation.
//!
//! Following the foundation's core principle (fail loudly, never swallow a
//! semantic failure), [`dbscan`](crate::dbscan) rejects a malformed distance
//! matrix or parameter rather than clustering garbage. The service layer maps
//! these to `400`s.

use std::fmt;

/// Things that can go wrong running [`dbscan`](crate::dbscan).
#[derive(Debug, Clone, PartialEq)]
pub enum ClusterError {
    /// The distance matrix has no rows — there is nothing to cluster.
    Empty,
    /// The matrix is not square: some row's length differs from the number of
    /// rows. The string names the offending row and its length.
    NotSquare(String),
    /// A distance entry was NaN or infinite; distances must be finite so the
    /// `<= eps` neighborhood test is well-defined.
    NonFiniteDistance,
    /// A distance entry was negative; a distance matrix must be non-negative.
    NegativeDistance,
    /// `eps` was NaN, infinite, or negative; the neighborhood radius must be a
    /// finite, non-negative number.
    InvalidEps,
    /// `min_points` was zero; a core point must require at least one point in
    /// its neighborhood (and the point counts itself, so the floor is 1).
    InvalidMinPoints,
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterError::Empty => write!(f, "distance matrix must have at least one row"),
            ClusterError::NotSquare(detail) => {
                write!(f, "distance matrix must be square: {detail}")
            }
            ClusterError::NonFiniteDistance => {
                write!(f, "distance entries must be finite (no NaN/infinity)")
            }
            ClusterError::NegativeDistance => {
                write!(f, "distance entries must be non-negative")
            }
            ClusterError::InvalidEps => {
                write!(f, "eps must be a finite, non-negative number")
            }
            ClusterError::InvalidMinPoints => {
                write!(f, "min_points must be at least 1")
            }
        }
    }
}

impl std::error::Error for ClusterError {}

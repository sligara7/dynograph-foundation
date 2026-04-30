//! Error types for DynoGraph.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DynoError {
    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Validation error on {node_type}.{property}: {message}")]
    Validation {
        node_type: String,
        property: String,
        message: String,
    },

    #[error("Validation error on edge {edge_type}.{property}: {message}")]
    EdgeValidation {
        edge_type: String,
        property: String,
        message: String,
    },

    #[error("Node not found: {node_type} {node_id}")]
    NodeNotFound { node_type: String, node_id: String },

    #[error("Edge not found: {edge_type} from {from_id} to {to_id}")]
    EdgeNotFound {
        edge_type: String,
        from_id: String,
        to_id: String,
    },

    #[error("Invalid edge: {edge_type} cannot connect {from_type} to {to_type}")]
    InvalidEdge {
        edge_type: String,
        from_type: String,
        to_type: String,
    },

    #[error("Unknown node type: {0}")]
    UnknownNodeType(String),

    #[error("Unknown edge type: {0}")]
    UnknownEdgeType(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Resolution error: {0}")]
    Resolution(String),

    #[error("Extraction error: {0}")]
    Extraction(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

//! HTTP REST API over the foundation crates. Hosts N independent
//! graphs, each backed by its own `StorageEngine`, behind a `/v1/`
//! surface.

mod app;
mod auth;
mod config;
mod edge_response;
mod embedding_response;
mod metadata_response;
mod node_response;
mod readiness;
mod registry;
mod schema_evolution;
mod schema_response;

pub use app::{AppState, app};
pub use auth::{AuthProvider, Identity, NoAuth};
pub use config::{Config, ConfigError, ServerConfig, StorageConfig};
pub use edge_response::EdgeResponse;
pub use embedding_response::EmbeddingResponse;
pub use metadata_response::GraphMetadataResponse;
pub use node_response::{NodeListResponse, NodeResponse};
pub use readiness::Readiness;
pub use registry::{GraphEntry, GraphRegistry, RegistryError, StorageBackend, validate_graph_id};
pub use schema_evolution::{EvolutionError, validate_compatible};
pub use schema_response::{SchemaResponse, WIRE_VERSION, content_hash};

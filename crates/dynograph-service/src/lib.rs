//! HTTP REST API over the foundation crates. Hosts N independent
//! graphs, each backed by its own `StorageEngine`, behind a `/v1/`
//! surface.

mod app;
mod auth;
mod batch;
mod buildinfo_response;
mod config;
mod edge_response;
mod edges_collect;
mod embedding_response;
mod metadata_response;
mod metrics_state;
mod node_response;
mod nodes_exists;
mod nodes_scan;
mod readiness;
mod registry;
mod resolve_or_create;
mod schema_evolution;
mod schema_response;
mod similar_response;
mod traverse;
mod validation;
mod welford_update;

pub use app::{AppState, app};
pub use auth::{AuthProvider, BearerJwt, Identity, NoAuth};
pub use buildinfo_response::{BuildInfoResponse, GIT_DIRTY, GIT_SHA};
pub use config::{AuthConfig, Config, ConfigError, ServerConfig, StorageConfig};
pub use edge_response::EdgeResponse;
pub use embedding_response::EmbeddingResponse;
pub use metadata_response::GraphMetadataResponse;
pub use node_response::{NodeListResponse, NodeResponse};
pub use readiness::Readiness;
pub use registry::{GraphEntry, GraphRegistry, RegistryError, StorageBackend, validate_graph_id};
pub use schema_evolution::{EvolutionError, validate_compatible};
pub use schema_response::{SchemaResponse, WIRE_VERSION, content_hash};
pub use similar_response::{SimilarHit, SimilarResponse};

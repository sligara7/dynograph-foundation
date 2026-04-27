//! HTTP REST API over the foundation crates. Hosts N independent
//! graphs, each backed by its own `StorageEngine`, behind a `/v1/`
//! surface.

mod app;
mod auth;
mod edge_response;
mod node_response;
mod registry;
mod schema_response;

pub use app::{AppState, app};
pub use auth::{AuthProvider, Identity, NoAuth};
pub use edge_response::EdgeResponse;
pub use node_response::NodeResponse;
pub use registry::{GraphEntry, GraphRegistry, RegistryError, StorageBackend, validate_graph_id};
pub use schema_response::{SchemaResponse, WIRE_VERSION, content_hash};

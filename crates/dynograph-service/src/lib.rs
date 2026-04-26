//! HTTP REST API over the foundation crates. Hosts N independent
//! graphs, each backed by its own `StorageEngine`, behind a `/v1/`
//! surface.

mod app;
mod auth;
mod registry;
mod schema_response;

pub use app::{AppState, app};
pub use auth::{AuthProvider, Identity, NoAuth};
pub use registry::{GraphEntry, GraphRegistry, RegistryError};
pub use schema_response::{SchemaResponse, WIRE_VERSION, content_hash};

//! DynoGraph Service — HTTP REST API over the foundation crates.
//!
//! Hosts N independent graphs (one `StorageEngine` each) behind a `/v1/`
//! REST surface. The first slice exposes graph lifecycle only — node, edge,
//! query, and similar routes follow in subsequent commits.

mod app;
mod auth;
mod registry;
mod schema_response;

pub use app::{AppState, app};
pub use auth::{AuthProvider, NoAuth};
pub use registry::{GraphRegistry, RegistryError};
pub use schema_response::{SchemaResponse, WIRE_VERSION, content_hash};

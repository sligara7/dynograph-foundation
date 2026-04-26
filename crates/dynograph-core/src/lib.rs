//! DynoGraph Core — Schema, types, and validation.
//!
//! This crate defines the schema model that drives the entire engine.
//! A schema is a runtime description of what nodes and edges can exist,
//! what properties they carry, and how entity resolution works.

mod schema;
mod value;
mod error;

pub use schema::{Schema, NodeTypeDef, EdgeTypeDef, EdgeEndpoint, PropertyDef, PropertyType, ResolutionConfig, ExtractionInclude};
pub use value::Value;
pub use error::DynoError;

//! DynoGraph Core — Schema, types, and validation.
//!
//! This crate defines the schema model that drives the entire engine.
//! A schema is a runtime description of what nodes and edges can exist,
//! what properties they carry, and how entity resolution works.

mod error;
mod schema;
mod value;

pub use error::DynoError;
pub use schema::{
    EdgeEndpoint, EdgeTypeDef, ExtractionInclude, NodeTypeDef, PropertyDef, PropertyType,
    ResolutionConfig, ResolutionStrategy, Schema,
};
pub use value::Value;

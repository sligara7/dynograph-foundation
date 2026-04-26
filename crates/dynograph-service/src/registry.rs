//! In-process registry of named graphs.
//!
//! Each graph owns an independent `StorageEngine`. Lock topology:
//! - Outer `RwLock` on the registry: rare writes (create / destroy),
//!   frequent reads (handler lookups).
//! - Inner per-graph `RwLock<StorageEngine>`: matches the engine's
//!   `&self` reads vs `&mut self` writes — concurrent reads, serialized
//!   writes per graph. Standard, no extra deps. Swap to `DashMap` or a
//!   per-graph actor model in v0.4 if contention shows up.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use dynograph_core::Schema;
use dynograph_storage::StorageEngine;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("graph already exists: {0}")]
    AlreadyExists(String),
    #[error("graph not found: {0}")]
    NotFound(String),
}

pub struct GraphRegistry {
    graphs: RwLock<HashMap<String, Arc<RwLock<StorageEngine>>>>,
}

impl GraphRegistry {
    pub fn new() -> Self {
        Self {
            graphs: RwLock::new(HashMap::new()),
        }
    }

    /// Create a graph with the given id and schema. Engine is in-memory
    /// for slice 1 — RocksDB-backed engines land with the `dynograph`
    /// binary slice that wires up config + per-graph paths.
    pub fn create_graph(&self, id: &str, schema: Schema) -> Result<(), RegistryError> {
        let mut graphs = self.graphs.write().expect("registry write lock poisoned");
        if graphs.contains_key(id) {
            return Err(RegistryError::AlreadyExists(id.to_string()));
        }
        let engine = StorageEngine::new_in_memory(schema);
        graphs.insert(id.to_string(), Arc::new(RwLock::new(engine)));
        Ok(())
    }

    /// Get a handle to a graph's engine. Returns `None` if the id
    /// doesn't exist. The Arc lets handlers drop the registry read-lock
    /// before doing engine work.
    pub fn get(&self, id: &str) -> Option<Arc<RwLock<StorageEngine>>> {
        self.graphs
            .read()
            .expect("registry read lock poisoned")
            .get(id)
            .cloned()
    }

    pub fn len(&self) -> usize {
        self.graphs
            .read()
            .expect("registry read lock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for GraphRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_schema() -> Schema {
        Schema::from_yaml(
            r#"
schema:
  name: t
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string, required: true }
  edge_types: {}
"#,
        )
        .unwrap()
    }

    #[test]
    fn create_graph_inserts() {
        let r = GraphRegistry::new();
        r.create_graph("g1", tiny_schema()).unwrap();
        assert_eq!(r.len(), 1);
        assert!(r.get("g1").is_some());
        assert!(r.get("missing").is_none());
    }

    #[test]
    fn create_graph_rejects_duplicate() {
        let r = GraphRegistry::new();
        r.create_graph("g1", tiny_schema()).unwrap();
        let err = r.create_graph("g1", tiny_schema()).unwrap_err();
        assert!(matches!(err, RegistryError::AlreadyExists(ref id) if id == "g1"));
    }

    #[test]
    fn distinct_graphs_coexist() {
        let r = GraphRegistry::new();
        r.create_graph("a", tiny_schema()).unwrap();
        r.create_graph("b", tiny_schema()).unwrap();
        assert_eq!(r.len(), 2);
    }
}

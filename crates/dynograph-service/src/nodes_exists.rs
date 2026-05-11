//! POST /v1/graphs/{id}/nodes:exists — batch (type, name) existence check.
//!
//! Surfaced by market_graph's two-pass extraction pipeline (Pass 1b
//! relevance gate): after Pass 1 extracts N entities from a fragment,
//! the gate asks "do any of these already exist?" — if yes, run Pass
//! 1b with graph context; if no, skip the second LLM call. Today that
//! decision needs N round-trips via `list_nodes` (or N `get_node`s
//! when the natural-key id is known); after migration it is one HTTP
//! call.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/graphs/{id}/nodes:exists
//! {
//!   "queries": [
//!     {"type": "Geography",  "name": "Iran"},
//!     {"type": "Commodity",  "name": "Oil"}
//!   ]
//! }
//! → {
//!   "results": [
//!     {"type": "Geography", "name": "Iran", "exists": true,  "id": "geo:iran"},
//!     {"type": "Commodity", "name": "Oil",  "exists": false, "id": null}
//!   ]
//! }
//! ```
//!
//! Response order matches request order so the caller can zip queries
//! with results without rebuilding a `(type, name) → result` map.
//!
//! ## Validation (all 400, pre-flight)
//!
//! - `queries` non-empty and `<= MAX_QUERIES`
//! - Every `type` referenced is declared in the schema
//! - Every referenced `type` declares a `name` property flagged
//!   `indexed: true`. Without the index, `scan_nodes_by_property`
//!   silently returns empty — every entity would look "absent" and
//!   the relevance gate would mask a schema misconfiguration as
//!   "everything is new". Same un-indexed-rejection policy
//!   `/resolve-or-create` and `/edges:collect` use.
//!
//! ## Ambiguity
//!
//! If two nodes of the same type happen to share a `name` value
//! (because the consumer never ran resolve-or-create, or set
//! `name` outside the resolver), this route returns one of the
//! matches non-deterministically and reports `exists: true`. Callers
//! that need uniqueness should funnel writes through
//! `/v1/graphs/{id}/resolve-or-create`, which enforces fuzzy
//! deduplication on the way in.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use dynograph_core::Value;
use dynograph_storage::StorageEngine;

use crate::registry::RegistryError;
use crate::validation::validate_indexed_property;

/// Hard upper bound on a single batch. Above this, callers should
/// shard. Picked at ~1000 to comfortably cover a single LLM-extracted
/// fragment's entity-set (market_graph's two-pass gate is tens of
/// entities per call) without leaving room for accidental
/// graph-wide scans.
pub(crate) const MAX_QUERIES: usize = 1000;

#[derive(Debug, Deserialize)]
pub(crate) struct NodesExistsRequest {
    pub queries: Vec<NodeQuery>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NodeQuery {
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct NodeExistence {
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    pub exists: bool,
    pub id: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct NodesExistsResponse {
    pub results: Vec<NodeExistence>,
}

/// Run a batch existence check. Caller holds the `with_engine_read`
/// lock — pure read path, one consistent snapshot for the whole
/// batch.
pub(crate) fn run(
    engine: &StorageEngine,
    graph_id: &str,
    req: NodesExistsRequest,
) -> Result<NodesExistsResponse, RegistryError> {
    if req.queries.is_empty() {
        return Err(RegistryError::BadRequest(
            "queries must be non-empty".to_string(),
        ));
    }
    if req.queries.len() > MAX_QUERIES {
        return Err(RegistryError::BadRequest(format!(
            "queries length {} exceeds maximum {MAX_QUERIES}",
            req.queries.len()
        )));
    }

    // Validate each distinct type once (a fragment's entity-set
    // typically has dozens of entries spread across a handful of types
    // — no need to re-walk the schema for every entry).
    let schema = engine.schema();
    let mut validated_types: HashSet<&str> = HashSet::new();
    for q in &req.queries {
        if !validated_types.insert(q.node_type.as_str()) {
            continue;
        }
        validate_indexed_property(schema, &q.node_type, "name", "queries")?;
    }

    let mut results = Vec::with_capacity(req.queries.len());
    for q in req.queries {
        let value = Value::String(q.name.clone());
        let matches = engine.scan_nodes_by_property(graph_id, &q.node_type, "name", &value)?;
        let first = matches.into_iter().next();
        let (exists, id) = match first {
            Some(n) => (true, Some(n.node_id)),
            None => (false, None),
        };
        results.push(NodeExistence {
            node_type: q.node_type,
            name: q.name,
            exists,
            id,
        });
    }

    Ok(NodesExistsResponse { results })
}

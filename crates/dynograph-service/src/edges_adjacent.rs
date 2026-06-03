//! POST /v1/graphs/{id}/edges:adjacent — single-node 1-hop adjacency.
//!
//! Returns the edges incident on ONE node, addressed by id. This is the
//! per-node adjacency that `edges:collect` (fan-out by source *type*) does not
//! cover: `edges:collect` scans all nodes of a type, whereas a great many
//! callers want "the edges of *this* node" — `outgoing_edges(id)` /
//! `incoming_edges(id)`. Storage already keys adjacency by node id
//! (`CF_ADJ_OUT` / `CF_ADJ_IN`), so this is a thin, O(K) wrapper over
//! `scan_outgoing_edges` / `scan_incoming_edges` — no type or schema needed.
//!
//! ```jsonc
//! {
//!   "node_id": "abc-123",          // required
//!   "direction": "both",            // optional: outgoing | incoming | both (default both)
//!   "edge_type": "MENTIONS",        // optional: filter to one edge type
//!   "limit": 200                    // optional: 1..=10_000, default 10_000
//! }
//! ```
//!
//! Response: `{ "edges": [ {edge_type, from_id, to_id, properties} ], "truncated": bool }`.
//! For `direction = both`, outgoing edges come first, then incoming. `truncated`
//! is `true` when a super-node's adjacency exceeded `limit` (some edges omitted).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use dynograph_core::Value;
use dynograph_storage::StorageEngine;

use crate::registry::RegistryError;
use crate::validation::{MAX_LIMIT, validate_limit};

#[derive(Debug, Deserialize)]
pub(crate) struct EdgesAdjacentRequest {
    pub node_id: String,
    #[serde(default)]
    pub direction: Direction,
    #[serde(default)]
    pub edge_type: Option<String>,
    /// Cap on the total edges returned (outgoing + incoming combined).
    /// Optional and defaults to `MAX_LIMIT` — unlike the *required*
    /// `limit` on `edges:collect` / `nodes:scan`, because this route
    /// shipped without one and defaulting keeps existing callers
    /// (which send no `limit`) working while still bounding the
    /// super-node response.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    MAX_LIMIT
}

#[derive(Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Direction {
    Outgoing,
    Incoming,
    #[default]
    Both,
}

#[derive(Debug, Serialize)]
pub(crate) struct AdjacentEdge {
    pub edge_type: String,
    pub from_id: String,
    pub to_id: String,
    pub properties: HashMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct EdgesAdjacentResponse {
    pub edges: Vec<AdjacentEdge>,
    /// `true` when more edges existed than `limit` allowed — some were
    /// omitted from `edges`.
    pub truncated: bool,
}

/// Wraps `scan_outgoing_edges` / `scan_incoming_edges`. Caller holds the
/// `with_engine_read` lock. No schema validation — adjacency is keyed by node
/// id, not type, so an unknown/missing node id simply yields an empty list.
pub(crate) fn run(
    engine: &StorageEngine,
    graph_id: &str,
    req: EdgesAdjacentRequest,
) -> Result<EdgesAdjacentResponse, RegistryError> {
    validate_limit(req.limit, "limit")?;

    let filter = req.edge_type.as_deref();
    let mut edges = Vec::new();
    let mut truncated = false;

    if req.direction != Direction::Incoming {
        for e in engine.scan_outgoing_edges(graph_id, &req.node_id, filter)? {
            if edges.len() >= req.limit {
                truncated = true;
                break;
            }
            edges.push(AdjacentEdge {
                edge_type: e.edge_type,
                from_id: e.from_id,
                to_id: e.to_id,
                properties: e.properties,
            });
        }
    }
    // Skip the incoming scan entirely once the cap is hit on a
    // super-node — there's no room left for more edges.
    if !truncated && req.direction != Direction::Outgoing {
        for e in engine.scan_incoming_edges(graph_id, &req.node_id, filter)? {
            if edges.len() >= req.limit {
                truncated = true;
                break;
            }
            edges.push(AdjacentEdge {
                edge_type: e.edge_type,
                from_id: e.from_id,
                to_id: e.to_id,
                properties: e.properties,
            });
        }
    }

    Ok(EdgesAdjacentResponse { edges, truncated })
}

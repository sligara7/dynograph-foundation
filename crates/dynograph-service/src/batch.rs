//! POST /v1/graphs/{id}/batch — atomic multi-op transaction.
//!
//! Closes the dominant atomicity gap surfaced by the storyflow audit
//! (2026-05-04): every multi-write storyflow handler currently relies
//! on the in-process write lock to make a sequence of mutations
//! atomic, which doesn't survive the move to HTTP. This route lets a
//! caller submit a list of node/edge mutations and have them either
//! all apply or all roll back.
//!
//! Mechanism: one `with_state_write` lock + storage's existing
//! `begin_batch` / `commit_batch` / `discard_batch` primitives. On any
//! per-op failure the batch is discarded and a structured 4xx is
//! returned identifying the failing op by index — mirrors the
//! single-handler error shape but adds the `op_index` and `op_type`
//! fields that callers need to debug a partial-rejection.
//!
//! Op JSON shape mirrors the existing single-handler request bodies
//! (`CreateNodeBody`, `CreateEdgeBody`, etc.) so callers can
//! mechanically translate single calls into batch entries; we don't
//! introduce a parallel JSON dialect to maintain.
//!
//! ## Read-your-own-writes within a batch
//!
//! Reads inside an active batch see the post-buffer view: a `Put` or
//! `Delete` queued earlier in the same batch is visible to subsequent
//! `get` and `prefix_scan` calls. Compositions like `create_node X`
//! then `replace_node X`, or `create_edge X→Y` then `delete_node X`
//! (which cascades), apply atomically at commit. A discarded batch
//! leaves the backend untouched — the buffer drops without a flush.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use dynograph_core::Value;
use dynograph_storage::StorageEngine;

/// Soft cap on ops per batch. Storyflow's heaviest known case
/// (`integrate_fragment`) runs ~67 writes; this leaves ample headroom
/// while preventing a runaway request from holding the write lock for
/// arbitrarily long. Lift if real workloads push past.
pub(crate) const MAX_BATCH_OPS: usize = 1000;

#[derive(Debug, Deserialize)]
pub(crate) struct BatchRequest {
    pub ops: Vec<BatchOp>,
}

/// One mutation. Field names match the existing single-handler bodies
/// so callers can translate `POST /v1/graphs/{id}/nodes` payloads into
/// `{"op": "create_node", ...}` entries with no field renames.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum BatchOp {
    CreateNode {
        node_type: String,
        node_id: String,
        #[serde(default)]
        properties: HashMap<String, Value>,
    },
    ReplaceNode {
        node_type: String,
        node_id: String,
        #[serde(default)]
        properties: HashMap<String, Value>,
    },
    DeleteNode {
        node_type: String,
        node_id: String,
    },
    CreateEdge {
        edge_type: String,
        from_type: String,
        from_id: String,
        to_type: String,
        to_id: String,
        #[serde(default)]
        properties: HashMap<String, Value>,
    },
    MergeEdge {
        edge_type: String,
        from_id: String,
        to_id: String,
        #[serde(default)]
        properties: HashMap<String, Value>,
    },
    DeleteEdge {
        edge_type: String,
        from_id: String,
        to_id: String,
    },
}

impl BatchOp {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            BatchOp::CreateNode { .. } => "create_node",
            BatchOp::ReplaceNode { .. } => "replace_node",
            BatchOp::DeleteNode { .. } => "delete_node",
            BatchOp::CreateEdge { .. } => "create_edge",
            BatchOp::MergeEdge { .. } => "merge_edge",
            BatchOp::DeleteEdge { .. } => "delete_edge",
        }
    }
}

/// Per-op effect captured during the apply loop. Drives the response
/// counts; `NodeDeleted` also carries (node_type, node_id) so the
/// handler can flush the matching HNSW index entry after commit.
pub(crate) enum OpEffect {
    NodeCreated,
    NodeReplaced,
    NodeDeleted { node_type: String, node_id: String },
    EdgeCreated,
    EdgeMerged,
    EdgeDeleted,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct BatchResponse {
    pub ops_applied: usize,
    pub nodes_created: usize,
    pub nodes_replaced: usize,
    pub nodes_deleted: usize,
    pub edges_created: usize,
    pub edges_merged: usize,
    pub edges_deleted: usize,
}

/// Structured per-op error body. Plain-text errors (the convention for
/// single handlers) lose the index of the failing op, which is the one
/// thing a batch caller needs to diagnose a rejection.
#[derive(Debug, Serialize)]
pub(crate) struct BatchOpError {
    pub error: String,
    pub op_index: usize,
    pub op_type: &'static str,
}

/// Apply one op against `engine` with batching active. The "missing
/// resource" returns from `replace_node_properties` / `merge_edge_properties`
/// (`Ok(None)`) and the `false` returns from `delete_*` are translated
/// into per-op errors so the batch rolls back — silently treating a
/// missing-target replace/delete as a no-op would violate the
/// no-silent-fallbacks rule.
fn apply_op(engine: &mut StorageEngine, graph_id: &str, op: BatchOp) -> Result<OpEffect, String> {
    match op {
        BatchOp::CreateNode {
            node_type,
            node_id,
            properties,
        } => engine
            .create_node(graph_id, &node_type, &node_id, properties)
            .map(|_| OpEffect::NodeCreated)
            .map_err(|e| e.to_string()),
        BatchOp::ReplaceNode {
            node_type,
            node_id,
            properties,
        } => match engine.replace_node_properties(graph_id, &node_type, &node_id, properties) {
            Ok(Some(_)) => Ok(OpEffect::NodeReplaced),
            Ok(None) => Err(format!("node not found: {node_type}/{node_id}")),
            Err(e) => Err(e.to_string()),
        },
        BatchOp::DeleteNode { node_type, node_id } => {
            match engine.delete_node(graph_id, &node_type, &node_id) {
                Ok(true) => Ok(OpEffect::NodeDeleted { node_type, node_id }),
                Ok(false) => Err(format!("node not found: {node_type}/{node_id}")),
                Err(e) => Err(e.to_string()),
            }
        }
        BatchOp::CreateEdge {
            edge_type,
            from_type,
            from_id,
            to_type,
            to_id,
            properties,
        } => engine
            .create_edge(
                graph_id, &edge_type, &from_type, &from_id, &to_type, &to_id, properties,
            )
            .map(|_| OpEffect::EdgeCreated)
            .map_err(|e| e.to_string()),
        BatchOp::MergeEdge {
            edge_type,
            from_id,
            to_id,
            properties,
        } => match engine.merge_edge_properties(graph_id, &edge_type, &from_id, &to_id, properties)
        {
            Ok(Some(_)) => Ok(OpEffect::EdgeMerged),
            Ok(None) => Err(format!("edge not found: {edge_type} {from_id}->{to_id}")),
            Err(e) => Err(e.to_string()),
        },
        BatchOp::DeleteEdge {
            edge_type,
            from_id,
            to_id,
        } => match engine.delete_edge(graph_id, &edge_type, &from_id, &to_id) {
            Ok(true) => Ok(OpEffect::EdgeDeleted),
            Ok(false) => Err(format!("edge not found: {edge_type} {from_id}->{to_id}")),
            Err(e) => Err(e.to_string()),
        },
    }
}

/// Run every op in order against `engine` (which must already have
/// `begin_batch()` active). Returns the response counts plus the list
/// of successfully-deleted nodes so the caller can flush their HNSW
/// entries after commit. On the first per-op failure, returns the
/// structured error — the caller is responsible for `discard_batch()`.
pub(crate) fn run_ops(
    engine: &mut StorageEngine,
    graph_id: &str,
    ops: Vec<BatchOp>,
) -> Result<(BatchResponse, Vec<(String, String)>), BatchOpError> {
    let mut response = BatchResponse::default();
    let mut deleted_nodes: Vec<(String, String)> = Vec::new();

    for (op_index, op) in ops.into_iter().enumerate() {
        let op_type = op.kind();
        let effect = apply_op(engine, graph_id, op).map_err(|error| BatchOpError {
            error,
            op_index,
            op_type,
        })?;
        match effect {
            OpEffect::NodeCreated => response.nodes_created += 1,
            OpEffect::NodeReplaced => response.nodes_replaced += 1,
            OpEffect::NodeDeleted { node_type, node_id } => {
                response.nodes_deleted += 1;
                deleted_nodes.push((node_type, node_id));
            }
            OpEffect::EdgeCreated => response.edges_created += 1,
            OpEffect::EdgeMerged => response.edges_merged += 1,
            OpEffect::EdgeDeleted => response.edges_deleted += 1,
        }
        response.ops_applied += 1;
    }

    Ok((response, deleted_nodes))
}

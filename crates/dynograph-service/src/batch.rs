//! POST /v1/graphs/{id}/batch — atomic multi-op transaction.
//!
//! Closes the dominant atomicity gap surfaced by the 2026-05-04 audit:
//! every multi-write handler currently relies
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
use utoipa::ToSchema;

use dynograph_core::Value;
use dynograph_storage::StorageEngine;

/// Soft cap on ops per batch. The heaviest known consumer case
/// runs ~67 writes; this leaves ample headroom
/// while preventing a runaway request from holding the write lock for
/// arbitrarily long. Lift if real workloads push past.
pub(crate) const MAX_BATCH_OPS: usize = 1000;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct BatchRequest {
    pub ops: Vec<BatchOp>,
    /// Validate-only: run every op against the batch buffer (read-your-own-
    /// writes intact) to compute a per-op pass/fail report, then discard
    /// without committing. The graph is never mutated. Defaults to false
    /// (commit). See [`BatchValidation`].
    #[serde(default)]
    pub dry_run: bool,
    /// With `dry_run`, keep evaluating after an op fails so the report covers
    /// EVERY op instead of stopping at the first failure. Ignored unless
    /// `dry_run` is set. Costs a buffer rebuild + replay per failing op — see
    /// [`dry_run_ops_exhaustive`] for why a failure cannot simply be skipped.
    /// Defaults to false (stop at the first failure).
    #[serde(default)]
    pub exhaustive: bool,
}

/// One mutation. Field names match the existing single-handler bodies
/// so callers can translate `POST /v1/graphs/{id}/nodes` payloads into
/// `{"op": "create_node", ...}` entries with no field renames.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum BatchOp {
    CreateNode {
        node_type: String,
        node_id: String,
        #[serde(default)]
        #[schema(value_type = Object)]
        properties: HashMap<String, Value>,
    },
    ReplaceNode {
        node_type: String,
        node_id: String,
        #[serde(default)]
        #[schema(value_type = Object)]
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
        #[schema(value_type = Object)]
        properties: HashMap<String, Value>,
    },
    MergeEdge {
        edge_type: String,
        from_id: String,
        to_id: String,
        #[serde(default)]
        #[schema(value_type = Object)]
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

#[derive(Debug, Default, Serialize, ToSchema)]
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
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct BatchOpError {
    pub error: String,
    pub op_index: usize,
    pub op_type: &'static str,
}

/// One op's outcome in a `dry_run` validation report.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct BatchOpResult {
    pub index: usize,
    pub op: &'static str,
    pub ok: bool,
    /// The would-be failure message; absent when `ok`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response for a `dry_run` batch: a per-op pass/fail report plus an overall
/// flag.
///
/// In the DEFAULT mode evaluation **stops at the first failing op** (a commit
/// aborts there, so reporting failures past it would describe a sequence that
/// can never run), and `results` covers the ops up to and including that
/// failure.
///
/// With `exhaustive` set, evaluation continues past a failure so `results`
/// covers every op — the preview-a-heal-pass case, where the caller wants all
/// the reasons at once rather than one per round-trip.
///
/// READ `truncated` BEFORE TREATING `results` AS COMPLETE. An exhaustive run
/// rebuilds its buffer after each failure (see [`dry_run_ops_exhaustive`]) and
/// gives up after [`MAX_DRY_RUN_RESTARTS`]; `truncated` says so explicitly
/// rather than letting a short `results` read as "those were all the ops".
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct BatchValidation {
    /// True iff every op would apply. Never true when `truncated`, because a
    /// truncated run has not looked at every op.
    pub valid: bool,
    pub results: Vec<BatchOpResult>,
    /// Which mode produced this report — echoes the request so a caller that
    /// forgot to set `exhaustive` cannot misread a stop-at-first report as a
    /// complete one.
    pub exhaustive: bool,
    /// True when evaluation gave up before covering every op. `results` is then
    /// SHORTER than the submitted op list and says nothing about the remainder.
    pub truncated: bool,
}

/// The `200 OK` body of `POST /batch`: a commit summary on the normal path, or
/// a per-op validation report for a `dry_run`. Untagged so the serialized bytes
/// are exactly the inner struct (no added discriminant — the commit response is
/// byte-for-byte unchanged) while utoipa still emits an accurate `oneOf`.
#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum BatchOk {
    Commit(BatchResponse),
    DryRun(BatchValidation),
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

/// Run ops against `engine` (which must already have `begin_batch()` active)
/// for a `dry_run`, recording each op's pass/fail, and **stop at the first
/// failure** — exactly where a real commit would abort. Stopping there is both
/// faithful (a commit never reaches ops past the first failure) and necessary
/// for correctness: a node write can buffer its RocksDB put *before* a fallible
/// full-text mirror step (`StorageEngine` documents this authoritative-then-
/// mirror order), so a failed op may leave a partial buffer entry; not
/// evaluating later ops against it avoids mislabeling them. The caller must
/// `discard_batch()` afterwards — a dry run never commits, so the partial entry
/// of a failed op is thrown away with the rest.
/// Cap on how many times an exhaustive dry run will rebuild its buffer after a
/// failing op. Each rebuild replays every previously-successful op, so an input
/// that fails on most of its ops is quadratic; this bounds it. Hitting the cap
/// sets `truncated` — the report never quietly stops short.
pub(crate) const MAX_DRY_RUN_RESTARTS: usize = 64;

/// Exhaustive `dry_run`: evaluate EVERY op, not just up to the first failure.
///
/// Why this cannot simply keep looping past a failure — the same reason
/// [`dry_run_ops`] stops: a node write can buffer its RocksDB put *before* a
/// fallible full-text mirror step (`StorageEngine`'s authoritative-then-mirror
/// order), so a failed op may leave a PARTIAL entry in the buffer. Evaluating
/// later ops against that entry would mislabel them — the exact silent-wrong
/// answer the stop-at-first rule exists to prevent.
///
/// So a failure is followed by a REBUILD: discard the poisoned buffer, begin a
/// fresh one, and replay the ops that already passed (skipping the failed one).
/// Later ops are then judged against a clean buffer that still carries the
/// read-your-own-writes effects of everything that genuinely succeeded — which
/// is what a caller previewing a heal pass is asking about.
///
/// Two honest limits, both reported rather than hidden:
/// - after [`MAX_DRY_RUN_RESTARTS`] rebuilds it stops and sets `truncated`;
/// - if replaying an op that previously PASSED fails, the buffer is not
///   reproducible, so it stops and sets `truncated` rather than emitting a
///   verdict it cannot stand behind.
///
/// The caller must `discard_batch()` afterwards, exactly as for [`dry_run_ops`].
pub(crate) fn dry_run_ops_exhaustive(
    engine: &mut StorageEngine,
    graph_id: &str,
    ops: Vec<BatchOp>,
) -> BatchValidation {
    let mut results = Vec::with_capacity(ops.len());
    // Ops that applied cleanly, in order — the replay script for a rebuild.
    let mut passed: Vec<BatchOp> = Vec::new();
    let mut restarts = 0usize;
    let mut truncated = false;

    for (index, op) in ops.into_iter().enumerate() {
        let op_type = op.kind();
        match apply_op(engine, graph_id, op.clone()) {
            Ok(_) => {
                passed.push(op);
                results.push(BatchOpResult {
                    index,
                    op: op_type,
                    ok: true,
                    error: None,
                });
            }
            Err(e) => {
                results.push(BatchOpResult {
                    index,
                    op: op_type,
                    ok: false,
                    error: Some(e),
                });
                if restarts >= MAX_DRY_RUN_RESTARTS {
                    truncated = true;
                    break;
                }
                restarts += 1;
                engine.discard_batch();
                engine.begin_batch();
                let mut replay_ok = true;
                for prior in &passed {
                    if apply_op(engine, graph_id, prior.clone()).is_err() {
                        replay_ok = false;
                        break;
                    }
                }
                if !replay_ok {
                    truncated = true;
                    break;
                }
            }
        }
    }

    BatchValidation {
        // A truncated run has not seen every op, so it can never claim validity.
        valid: !truncated && results.iter().all(|r| r.ok),
        results,
        exhaustive: true,
        truncated,
    }
}

pub(crate) fn dry_run_ops(
    engine: &mut StorageEngine,
    graph_id: &str,
    ops: Vec<BatchOp>,
) -> BatchValidation {
    let mut results = Vec::with_capacity(ops.len());
    for (index, op) in ops.into_iter().enumerate() {
        let op_type = op.kind();
        match apply_op(engine, graph_id, op) {
            Ok(_) => results.push(BatchOpResult {
                index,
                op: op_type,
                ok: true,
                error: None,
            }),
            Err(e) => {
                results.push(BatchOpResult {
                    index,
                    op: op_type,
                    ok: false,
                    error: Some(e),
                });
                return BatchValidation {
                    valid: false,
                    results,
                    exhaustive: false,
                    truncated: false,
                };
            }
        }
    }
    BatchValidation {
        valid: true,
        results,
        exhaustive: false,
        truncated: false,
    }
}

//! `StorageEngine` — public full-text search/reindex. Split out of `engine.rs`; `use super::*`
//! inherits the shared imports and types from the parent `engine` module.
//! Private helper methods live in `engine/mod.rs` (a parent module, so
//! these methods reach them as descendants).

use super::*;

impl StorageEngine {
    /// BM25 keyword search over the full-text index, scoped to `graph_id` and
    /// optionally one `node_type`. Returns up to `limit` hits, highest score
    /// first. Empty when the schema declares no `fulltext` property. Fails loud
    /// if full-text was enabled on a live engine that opened without an index
    /// (see `fulltext_unavailable`).
    #[cfg(feature = "fulltext")]
    pub fn search_fulltext(
        &self,
        graph_id: &str,
        query: &str,
        node_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<dynograph_text::TextHit>, DynoError> {
        if let Some(ti) = &self.text_index {
            return ti
                .search(graph_id, query, node_type, limit)
                .map_err(|e| DynoError::Storage(format!("full-text search failed: {e}")));
        }
        match self.fulltext_unavailable() {
            Some(err) => Err(err),
            None => Ok(Vec::new()),
        }
    }

    /// Rebuild the full-text index for `graph_id` from the authoritative node
    /// store: drop the graph's existing documents, then re-index every
    /// `fulltext` node. Use to recover from drift. Returns the number of nodes
    /// indexed; `Ok(0)` only when the schema genuinely declares no full-text.
    ///
    /// Fails loud (not `Ok(0)`) if full-text was enabled on a live engine that
    /// opened without an index — reopen the engine to build it first.
    ///
    /// Cost: materializes every full-text node and rebuilds in a single pass
    /// under the caller's lock (the service holds the per-graph write lock), so
    /// a reindex of a large graph blocks its reads and writes for the duration.
    /// An incremental / double-buffered rebuild is tracked as a follow-up.
    #[cfg(feature = "fulltext")]
    pub fn reindex_fulltext(&self, graph_id: &str) -> Result<usize, DynoError> {
        // A rebuild can't run inside an open batch: `scan_nodes` would see
        // uncommitted node state, and the final `ti.commit()` would flush the
        // batch's buffered text ops, breaking its all-or-nothing semantics.
        if self.is_batching() {
            return Err(DynoError::Storage(
                "reindex_fulltext cannot run inside an open batch".to_string(),
            ));
        }
        let ti = match &self.text_index {
            Some(ti) => ti,
            None => {
                return match self.fulltext_unavailable() {
                    Some(err) => Err(err),
                    None => Ok(0),
                };
            }
        };
        // Clear-then-rebuild buffers a `delete_graph` plus a series of
        // `upsert`s into the Tantivy writer before the final `commit()`. If any
        // step fails partway through, the half-built batch must be rolled back
        // before returning — otherwise those ops stay queued in the shared
        // writer and a later unrelated `commit()` (e.g. from a node write)
        // would flush a *partial* rebuild, silently dropping the graph's prior
        // index contents and leaving hard-to-debug drift.
        let rebuild = || -> Result<usize, DynoError> {
            ti.delete_graph(graph_id)
                .map_err(|e| DynoError::Storage(format!("full-text reindex clear failed: {e}")))?;
            // Snapshot the fulltext node types first — `scan_nodes` borrows &self.
            let node_types: Vec<String> = self
                .schema
                .node_types
                .keys()
                .filter(|nt| self.schema.has_fulltext_properties(nt))
                .cloned()
                .collect();
            let mut count = 0usize;
            for node_type in node_types {
                for node in self.scan_nodes(graph_id, &node_type)? {
                    let fields = self.fulltext_fields(&node_type, &node.properties);
                    ti.upsert(graph_id, &node_type, &node.node_id, &fields)
                        .map_err(|e| {
                            DynoError::Storage(format!("full-text reindex upsert failed: {e}"))
                        })?;
                    count += 1;
                }
            }
            ti.commit()
                .map_err(|e| DynoError::Storage(format!("full-text reindex commit failed: {e}")))?;
            Ok(count)
        };
        // Best-effort rollback to drop the uncommitted batch on failure;
        // surface the original rebuild error regardless of the rollback result.
        rebuild().inspect_err(|_| {
            let _ = ti.rollback();
        })
    }
}

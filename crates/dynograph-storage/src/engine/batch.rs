//! `StorageEngine` — write-buffer batching. Split out of `engine.rs`; `use super::*`
//! inherits the shared imports and types from the parent `engine` module.
//! Private helper methods live in `engine/mod.rs` (a parent module, so
//! these methods reach them as descendants).

use super::*;

impl StorageEngine {
    /// Begin buffering writes. All subsequent `put()` calls will be buffered
    /// instead of committed immediately. Call `commit_batch()` to write all
    /// buffered operations atomically.
    pub fn begin_batch(&mut self) {
        if self.write_buffer.is_some() {
            tracing::warn!(
                "begin_batch() called while batch already active — committing previous batch"
            );
            let _ = self.commit_batch();
        }
        // Establish a clean full-text baseline before the batch buffers its own
        // ops. `discard_batch` reverts via a writer-global `rollback()`, which is
        // only correct if the writer holds *exactly* this batch's ops. A prior
        // non-batched write whose index commit failed (the node is in RocksDB,
        // its index op left uncommitted) would otherwise be dropped by that
        // rollback — silent, permanent drift. Committing here flushes any such
        // stranded op (it matches a durable RocksDB write, so committing is the
        // correct heal) and guarantees the writer is clean when buffering starts.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index
            && let Err(e) = ti.commit()
        {
            tracing::error!("full-text commit at begin_batch failed: {e}");
        }
        self.write_buffer = Some(Vec::new());
    }

    /// Returns true if write batching is currently active.
    pub fn is_batching(&self) -> bool {
        self.write_buffer.is_some()
    }

    /// Commit all buffered writes as a single atomic operation.
    /// For RocksDB, this uses `WriteBatch` for atomic multi-CF writes.
    /// For in-memory backend, applies writes directly.
    pub fn commit_batch(&mut self) -> Result<usize, DynoError> {
        let buffer = match self.write_buffer.take() {
            Some(b) => b,
            None => return Ok(0),
        };

        let count = buffer.len();
        if count == 0 {
            return Ok(0);
        }

        // Invalidate cache before applying the batch so a concurrent
        // reader either sees pre-batch + cache-miss (re-fetches) or
        // post-batch + cache-miss — never stale data.
        {
            let mut cache = self.read_cache.lock().expect("read_cache lock poisoned");
            for op in &buffer {
                match op {
                    BufferedOp::Put { key, .. } | BufferedOp::Delete { key, .. } => {
                        cache.invalidate(key);
                    }
                    BufferedOp::PrefixDelete { prefix, .. } => {
                        cache.invalidate_prefix(prefix);
                    }
                }
            }
        }

        // Apply the buffered ops atomically. The backend owns the
        // all-or-nothing semantics (RocksDB `WriteBatch`; the in-memory
        // backend an in-order loop) and the `PrefixDelete`-supersedes-
        // earlier-puts ordering.
        self.backend.commit_batch(buffer)?;

        // Make this batch's buffered full-text writes visible now that the
        // authoritative backend write has landed. Reached only when count > 0
        // (the empty-buffer case returns early above), and any full-text op in
        // the batch rode alongside a node put/delete that's part of `count`.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index {
            ti.commit()
                .map_err(|e| DynoError::Storage(format!("full-text batch commit failed: {e}")))?;
        }

        Ok(count)
    }

    /// Discard all buffered writes without committing.
    pub fn discard_batch(&mut self) {
        self.write_buffer = None;
        // Revert the batch's buffered full-text writes too. Best-effort: a
        // rollback failure can't be surfaced through this infallible signature,
        // and the index is rebuildable via `reindex_fulltext` regardless.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index
            && let Err(e) = ti.rollback()
        {
            tracing::error!("full-text rollback after discard_batch failed: {e}");
        }
    }
}

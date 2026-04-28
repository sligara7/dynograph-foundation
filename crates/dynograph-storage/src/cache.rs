//! Read cache — LRU cache for RocksDB reads.
//!
//! Caches individual `get()` results to avoid repeated disk reads for hot
//! data. Invalidated synchronously on writes so reads are never stale.
//!
//! Backed by the `lru` crate's `LruCache`, which gives O(1) eviction
//! (the previous implementation walked the entire entries map on every
//! cache-full insert to find the least-recently-accessed key — quadratic
//! in the eviction-heavy workloads we see during bulk loads).
//!
//! Values are stored as `Arc<[u8]>`. A cache hit clones the arc (refcount
//! bump) instead of cloning the underlying bytes. The engine's read
//! path threads `Arc<[u8]>` through to the public-API decoders, which
//! deref to `&[u8]` for `rmp_serde::from_slice`.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lru::LruCache;

const DEFAULT_MAX_ENTRIES: usize = 10_000;
const DEFAULT_TTL_SECS: u64 = 300; // 5 minutes

/// A single cached entry. `created` is the wall-clock time the entry was
/// inserted; expiry is checked against `created + ttl` on `get`. The
/// LRU promotion order tracks recency separately inside `LruCache`, so
/// no per-entry `accessed` field is needed any more.
struct CacheEntry {
    data: Arc<[u8]>,
    created: Instant,
}

/// Configuration for the read cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub ttl: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_MAX_ENTRIES,
            ttl: Duration::from_secs(DEFAULT_TTL_SECS),
        }
    }
}

/// LRU read cache with TTL expiration and prefix invalidation.
pub struct ReadCache {
    entries: LruCache<Vec<u8>, CacheEntry>,
    ttl: Duration,
    hits: u64,
    misses: u64,
}

impl ReadCache {
    pub fn new(config: CacheConfig) -> Self {
        let cap = NonZeroUsize::new(config.max_entries.max(1))
            .expect("max_entries.max(1) is always >= 1");
        Self {
            entries: LruCache::new(cap),
            ttl: config.ttl,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached value. Returns `None` on miss or TTL expiry.
    /// On a hit, the entry is promoted to MRU; on a TTL expiry the
    /// stale entry is evicted lazily.
    pub fn get(&mut self, key: &[u8]) -> Option<Arc<[u8]>> {
        let now = Instant::now();
        // `LruCache::get` returns `Option<&V>` and promotes to MRU. We
        // need to either return a clone of the arc or evict on expiry,
        // so check expiry first via `peek` (no promote), then `get` to
        // promote on a real hit.
        match self.entries.peek(key) {
            Some(entry) if now.duration_since(entry.created) >= self.ttl => {
                self.entries.pop(key);
                self.misses += 1;
                None
            }
            Some(_) => {
                let entry = self.entries.get(key).expect("present from peek");
                self.hits += 1;
                Some(Arc::clone(&entry.data))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert a value into the cache. Eviction of the LRU entry happens
    /// inside `LruCache::put` when the cache is at capacity — O(1).
    pub fn put(&mut self, key: Vec<u8>, data: Arc<[u8]>) {
        self.entries.put(
            key,
            CacheEntry {
                data,
                created: Instant::now(),
            },
        );
    }

    /// Invalidate a single cache entry.
    pub fn invalidate(&mut self, key: &[u8]) {
        self.entries.pop(key);
    }

    /// Invalidate all entries whose key starts with the given prefix.
    ///
    /// Walks the entries store once to collect matching keys, then pops
    /// each. O(n) in the cache size — there is no prefix index. This is
    /// only called from `commit_batch` for `PrefixDelete` ops, which in
    /// the current codebase fires on `delete_node` and (TD-2) `clear_graph`
    /// — operations that already rewrite multiple CFs, so the cache scan
    /// is a small fraction of the work.
    pub fn invalidate_prefix(&mut self, prefix: &[u8]) {
        let to_remove: Vec<Vec<u8>> = self
            .entries
            .iter()
            .filter_map(|(k, _)| {
                if k.starts_with(prefix) {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        for k in &to_remove {
            self.entries.pop(k);
        }
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get cache statistics: (hits, misses, current_size).
    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.entries.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc_bytes(s: &[u8]) -> Arc<[u8]> {
        Arc::from(s.to_vec().into_boxed_slice())
    }

    #[test]
    fn test_get_put() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"key1".to_vec(), arc_bytes(b"value1"));
        assert_eq!(cache.get(b"key1").as_deref(), Some(b"value1".as_slice()));
        assert!(cache.get(b"key2").is_none());
    }

    #[test]
    fn test_invalidate() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"key1".to_vec(), arc_bytes(b"value1"));
        cache.invalidate(b"key1");
        assert!(cache.get(b"key1").is_none());
    }

    #[test]
    fn test_invalidate_prefix() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"graph1\x00node1".to_vec(), arc_bytes(b"v1"));
        cache.put(b"graph1\x00node2".to_vec(), arc_bytes(b"v2"));
        cache.put(b"graph2\x00node1".to_vec(), arc_bytes(b"v3"));

        cache.invalidate_prefix(b"graph1\x00");
        assert!(cache.get(b"graph1\x00node1").is_none());
        assert!(cache.get(b"graph1\x00node2").is_none());
        assert_eq!(
            cache.get(b"graph2\x00node1").as_deref(),
            Some(b"v3".as_slice())
        );
    }

    #[test]
    fn test_eviction() {
        let mut cache = ReadCache::new(CacheConfig {
            max_entries: 2,
            ttl: Duration::from_secs(300),
        });
        cache.put(b"key1".to_vec(), arc_bytes(b"v1"));
        cache.put(b"key2".to_vec(), arc_bytes(b"v2"));
        // Access key1 to make key2 the LRU
        cache.get(b"key1");
        // This should evict key2
        cache.put(b"key3".to_vec(), arc_bytes(b"v3"));

        assert_eq!(cache.get(b"key1").as_deref(), Some(b"v1".as_slice()));
        assert!(cache.get(b"key2").is_none());
        assert_eq!(cache.get(b"key3").as_deref(), Some(b"v3".as_slice()));
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = ReadCache::new(CacheConfig {
            max_entries: 100,
            ttl: Duration::from_millis(1),
        });
        cache.put(b"key1".to_vec(), arc_bytes(b"v1"));
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get(b"key1").is_none());
    }

    #[test]
    fn test_stats() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"key1".to_vec(), arc_bytes(b"v1"));
        cache.get(b"key1"); // hit
        cache.get(b"key2"); // miss
        let (hits, misses, size) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_arc_value_is_shared_on_hit() {
        // The value bytes shouldn't be copied on a cache hit — both the
        // stored entry and the returned arc must point at the same
        // allocation.
        let mut cache = ReadCache::new(CacheConfig::default());
        let payload = arc_bytes(b"shared");
        cache.put(b"k".to_vec(), Arc::clone(&payload));
        let hit = cache.get(b"k").expect("present");
        assert!(Arc::ptr_eq(&payload, &hit));
    }
}

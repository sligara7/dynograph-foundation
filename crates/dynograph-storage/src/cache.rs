//! Read cache — LRU cache for RocksDB reads.
//!
//! Caches individual `get()` results and `prefix_scan()` results to avoid
//! repeated disk reads for hot data. Invalidated synchronously on writes
//! so reads are never stale.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Default configuration.
const DEFAULT_MAX_ENTRIES: usize = 10_000;
const DEFAULT_TTL_SECS: u64 = 300; // 5 minutes

/// A single cached entry.
struct CacheEntry {
    data: Vec<u8>,
    accessed: Instant,
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
    entries: HashMap<Vec<u8>, CacheEntry>,
    config: CacheConfig,
    hits: u64,
    misses: u64,
}

impl ReadCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::with_capacity(config.max_entries / 2),
            config,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached value. Returns `None` on miss or expiry.
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let now = Instant::now();
        if let Some(entry) = self.entries.get_mut(key) {
            if now.duration_since(entry.created) < self.config.ttl {
                entry.accessed = now;
                self.hits += 1;
                return Some(entry.data.clone());
            }
            // Expired — remove lazily
            self.entries.remove(key);
        }
        self.misses += 1;
        None
    }

    /// Insert a value into the cache, evicting LRU entries if at capacity.
    pub fn put(&mut self, key: Vec<u8>, data: Vec<u8>) {
        if self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }
        let now = Instant::now();
        self.entries.insert(
            key,
            CacheEntry {
                data,
                accessed: now,
                created: now,
            },
        );
    }

    /// Invalidate a single cache entry.
    pub fn invalidate(&mut self, key: &[u8]) {
        self.entries.remove(key);
    }

    /// Invalidate all entries whose key starts with the given prefix.
    pub fn invalidate_prefix(&mut self, prefix: &[u8]) {
        self.entries.retain(|k, _| !k.starts_with(prefix));
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.entries.len())
    }

    /// Evict the least recently accessed entry.
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        // Find the key with the oldest access time
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_put() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"key1".to_vec(), b"value1".to_vec());
        assert_eq!(cache.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(cache.get(b"key2"), None);
    }

    #[test]
    fn test_invalidate() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"key1".to_vec(), b"value1".to_vec());
        cache.invalidate(b"key1");
        assert_eq!(cache.get(b"key1"), None);
    }

    #[test]
    fn test_invalidate_prefix() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"graph1\x00node1".to_vec(), b"v1".to_vec());
        cache.put(b"graph1\x00node2".to_vec(), b"v2".to_vec());
        cache.put(b"graph2\x00node1".to_vec(), b"v3".to_vec());

        cache.invalidate_prefix(b"graph1\x00");
        assert_eq!(cache.get(b"graph1\x00node1"), None);
        assert_eq!(cache.get(b"graph1\x00node2"), None);
        assert_eq!(cache.get(b"graph2\x00node1"), Some(b"v3".to_vec()));
    }

    #[test]
    fn test_eviction() {
        let mut cache = ReadCache::new(CacheConfig {
            max_entries: 2,
            ttl: Duration::from_secs(300),
        });
        cache.put(b"key1".to_vec(), b"v1".to_vec());
        cache.put(b"key2".to_vec(), b"v2".to_vec());
        // Access key1 to make key2 the LRU
        cache.get(b"key1");
        // This should evict key2
        cache.put(b"key3".to_vec(), b"v3".to_vec());

        assert_eq!(cache.get(b"key1"), Some(b"v1".to_vec()));
        assert_eq!(cache.get(b"key2"), None);
        assert_eq!(cache.get(b"key3"), Some(b"v3".to_vec()));
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = ReadCache::new(CacheConfig {
            max_entries: 100,
            ttl: Duration::from_millis(1),
        });
        cache.put(b"key1".to_vec(), b"v1".to_vec());
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(cache.get(b"key1"), None);
    }

    #[test]
    fn test_stats() {
        let mut cache = ReadCache::new(CacheConfig::default());
        cache.put(b"key1".to_vec(), b"v1".to_vec());
        cache.get(b"key1"); // hit
        cache.get(b"key2"); // miss
        let (hits, misses, size) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);
    }
}

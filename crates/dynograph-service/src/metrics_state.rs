//! Service-side metrics aggregation for `GET /metrics`.
//!
//! Per-(method, matched_path, status) request counters + latency
//! sums. Atomically incremented per request by `metrics_middleware`;
//! snapshot-emitted in Prometheus text format on each scrape. The
//! matched-path extraction (axum's `MatchedPath`) keeps cardinality
//! bounded by the static route count rather than ballooning with
//! per-graph-id URLs.
//!
//! Latency is recorded as `count + microseconds_sum` (Prometheus
//! "counter pair" — enables avg-latency dashboards via `rate(sum) /
//! rate(count)`). Histogram buckets deferred until a consumer asks
//! for percentiles; the bucket schedule is a binding contract.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestKey {
    pub method: String,
    pub path: String,
    pub status: u16,
}

#[derive(Debug, Default)]
pub struct RequestCounter {
    pub count: AtomicU64,
    pub latency_micros_sum: AtomicU64,
}

impl RequestCounter {
    pub fn record(&self, latency_micros: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.latency_micros_sum
            .fetch_add(latency_micros, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.count.load(Ordering::Relaxed),
            self.latency_micros_sum.load(Ordering::Relaxed),
        )
    }
}

pub struct MetricsState {
    counters: Mutex<HashMap<RequestKey, RequestCounter>>,
    started_at: Instant,
}

impl MetricsState {
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            started_at: Instant::now(),
        }
    }

    pub fn record(&self, method: &str, path: &str, status: u16, latency_micros: u64) {
        let key = RequestKey {
            method: method.to_string(),
            path: path.to_string(),
            status,
        };
        let counters = self.counters.lock().expect("metrics counters poisoned");
        // Look up existing entry under the lock; if absent, drop and
        // re-acquire a write to insert. The fast path (entry exists)
        // does one lookup + two atomic increments without re-locking.
        if let Some(c) = counters.get(&key) {
            c.record(latency_micros);
            return;
        }
        drop(counters);
        let mut counters = self.counters.lock().expect("metrics counters poisoned");
        let c = counters.entry(key).or_default();
        c.record(latency_micros);
    }

    /// Snapshot of all counters. Sorted by key for stable output —
    /// Prometheus tolerates unordered series but stable output makes
    /// scrape-diff debugging less painful.
    pub fn snapshot(&self) -> Vec<(RequestKey, u64, u64)> {
        let counters = self.counters.lock().expect("metrics counters poisoned");
        let mut out: Vec<(RequestKey, u64, u64)> = counters
            .iter()
            .map(|(k, c)| {
                let (count, sum) = c.snapshot();
                (k.clone(), count, sum)
            })
            .collect();
        out.sort_by(|a, b| {
            a.0.method
                .cmp(&b.0.method)
                .then(a.0.path.cmp(&b.0.path))
                .then(a.0.status.cmp(&b.0.status))
        });
        out
    }

    pub fn uptime_secs(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64()
    }
}

impl Default for MetricsState {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MetricsState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsState")
            .field("started_at", &self.started_at)
            .field("counter_count", &self.counters.lock().map(|c| c.len()).ok())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_increments_count_and_latency_sum() {
        let m = MetricsState::new();
        m.record("GET", "/v1/graphs", 200, 50);
        m.record("GET", "/v1/graphs", 200, 70);
        let snap = m.snapshot();
        assert_eq!(snap.len(), 1);
        let (key, count, sum) = &snap[0];
        assert_eq!(key.method, "GET");
        assert_eq!(key.path, "/v1/graphs");
        assert_eq!(key.status, 200);
        assert_eq!(*count, 2);
        assert_eq!(*sum, 120);
    }

    #[test]
    fn distinct_keys_dont_merge() {
        let m = MetricsState::new();
        m.record("GET", "/v1/graphs", 200, 1);
        m.record("POST", "/v1/graphs", 201, 1);
        m.record("GET", "/v1/graphs", 404, 1);
        assert_eq!(m.snapshot().len(), 3);
    }

    #[test]
    fn snapshot_is_sorted_by_method_then_path_then_status() {
        let m = MetricsState::new();
        m.record("POST", "/zeta", 201, 1);
        m.record("GET", "/alpha", 200, 1);
        m.record("GET", "/alpha", 404, 1);
        m.record("GET", "/beta", 200, 1);
        let keys: Vec<(String, String, u16)> = m
            .snapshot()
            .into_iter()
            .map(|(k, _, _)| (k.method, k.path, k.status))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("GET".into(), "/alpha".into(), 200),
                ("GET".into(), "/alpha".into(), 404),
                ("GET".into(), "/beta".into(), 200),
                ("POST".into(), "/zeta".into(), 201),
            ]
        );
    }

    #[test]
    fn uptime_is_monotonic() {
        let m = MetricsState::new();
        let a = m.uptime_secs();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let b = m.uptime_secs();
        assert!(b > a, "{a} → {b}");
    }
}

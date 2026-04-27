//! Readiness signal exposed via `GET /ready`.
//!
//! Distinct from `/health`, which only confirms the process is up.
//! `/ready` flips to 200 once the service has finished startup work
//! that must complete before traffic is safe — for slice 4+ that's
//! `GraphRegistry::rehydrate()` returning successfully on the
//! on-disk backend (in-memory mode has nothing to load, so the
//! binary marks ready immediately).
//!
//! Not-ready is the safe default: a coordinator's load balancer
//! sees 503 until something explicitly calls `mark_ready()`. The
//! library surface (`AppState::with_no_auth`) defaults to *ready*
//! for back-compat with the slice 1–3 tests, which never imagined
//! a readiness gate; the binary uses the lower-level `AppState::new`
//! constructor with an explicit not-ready `Readiness`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub struct Readiness {
    ready: AtomicBool,
}

impl Readiness {
    pub fn ready() -> Arc<Self> {
        Arc::new(Self {
            ready: AtomicBool::new(true),
        })
    }

    pub fn not_ready() -> Arc<Self> {
        Arc::new(Self {
            ready: AtomicBool::new(false),
        })
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }
}

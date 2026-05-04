//! Response shape for `GET /buildinfo`.
//!
//! Same provenance triple the `dynograph_build_info` gauge surfaces in
//! `/metrics`, exposed as JSON for callers that don't want to parse
//! Prometheus text format.

use serde::Serialize;

pub const GIT_SHA: &str = env!("DYNOGRAPH_GIT_SHA");
pub const GIT_DIRTY: &str = env!("DYNOGRAPH_GIT_DIRTY");

#[derive(Debug, Serialize)]
pub struct BuildInfoResponse {
    pub version: &'static str,
    pub git_sha: &'static str,
    pub git_dirty: bool,
    pub uptime_seconds: f64,
}

impl BuildInfoResponse {
    pub fn new(uptime_seconds: f64) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            git_sha: GIT_SHA,
            git_dirty: GIT_DIRTY == "true",
            uptime_seconds,
        }
    }
}

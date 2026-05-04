//! Captures git provenance at compile time so the running binary can
//! report exactly which commit it was built from.
//!
//! Sets two `cargo:rustc-env` vars:
//!
//! - `DYNOGRAPH_GIT_SHA` — short HEAD sha, or `"unknown"` if `git`
//!   isn't available (release tarball, etc).
//! - `DYNOGRAPH_GIT_DIRTY` — `"true"` or `"false"`. `"true"` means
//!   the working tree had uncommitted changes at build time;
//!   `"false"` means clean.
//!
//! Both are read in `app.rs` via `env!()` for the `dynograph_build_info`
//! Prometheus gauge labels and the `GET /buildinfo` JSON response.

use std::process::Command;

fn main() {
    // Trigger rebuilds when HEAD or refs/index change so the captured
    // sha tracks reality.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
    println!("cargo:rerun-if-changed=../../.git/index");

    let sha = run_git(&["rev-parse", "--short=7", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = run_git(&["status", "--porcelain"])
        .map(|out| !out.is_empty())
        .unwrap_or(false);

    println!("cargo:rustc-env=DYNOGRAPH_GIT_SHA={sha}");
    println!("cargo:rustc-env=DYNOGRAPH_GIT_DIRTY={dirty}");
}

fn run_git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

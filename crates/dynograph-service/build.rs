//! Captures git provenance at compile time so the running binary can
//! report exactly which commit it was built from.
//!
//! Sets two `cargo:rustc-env` vars:
//!
//! - `DYNOGRAPH_GIT_SHA` — short HEAD sha, or `"unknown"` if neither
//!   the build-time env var nor `git` is available.
//! - `DYNOGRAPH_GIT_DIRTY` — `"true"` or `"false"`.
//!
//! Source precedence:
//!
//! 1. `DYNOGRAPH_BUILD_GIT_SHA` / `DYNOGRAPH_BUILD_GIT_DIRTY` env vars
//!    if set. Used by the Dockerfile (which doesn't have `.git/` in
//!    its build context — `.dockerignore` excludes it) to inject
//!    provenance from GitHub Actions at image-build time.
//! 2. `git rev-parse` / `git status --porcelain` shelled out to.
//!    Used for local `cargo build` and any CI build that has the
//!    repo checked out with `.git/` intact.
//! 3. Fallback `"unknown"` / `"false"`.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!("cargo:rerun-if-env-changed=DYNOGRAPH_BUILD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=DYNOGRAPH_BUILD_GIT_DIRTY");

    let sha = std::env::var("DYNOGRAPH_BUILD_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| run_git(&["rev-parse", "--short=7", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = std::env::var("DYNOGRAPH_BUILD_GIT_DIRTY")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s == "true")
        .or_else(|| run_git(&["status", "--porcelain"]).map(|out| !out.is_empty()))
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

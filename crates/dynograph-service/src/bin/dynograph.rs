//! `dynograph` — the deployable binary entry point.
//!
//! Sequence:
//! 1. Parse CLI: just `--config <path>` (everything else lives in
//!    the TOML file or env vars).
//! 2. Load `Config` (TOML, then env-var overrides).
//! 3. Build `GraphRegistry` per `storage.root` (in-memory if absent).
//! 4. Call `rehydrate()` on the on-disk path; the readiness signal
//!    flips to ready only after rehydrate succeeds. In-memory mode
//!    is ready immediately.
//! 5. Serve on `server.bind` (and, when `server.uds_path` is set, on
//!    that Unix socket too) until SIGINT or SIGTERM. Graceful
//!    shutdown lets in-flight requests drain across every transport.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use tokio::signal::unix::{SignalKind, signal};
use tracing::{error, info, warn};

use dynograph_service::{AppState, Config, GraphRegistry, Readiness, app, bind_uds};

#[tokio::main]
async fn main() -> ExitCode {
    // Default tracing subscriber so info!/error! land on stderr.
    tracing_subscriber_init();

    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("dynograph: {e}");
            ExitCode::FAILURE
        }
    }
}

fn tracing_subscriber_init() {
    // `set_global_default` errors if called twice; in `main` we call
    // it once at startup, so warn and continue if it somehow fires
    // again rather than swallow the failure silently.
    if let Err(e) = tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_writer(std::io::stderr)
            .with_max_level(tracing::Level::INFO)
            .finish(),
    ) {
        eprintln!("warning: tracing subscriber already initialized: {e}");
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = parse_config_arg()?;
    let cfg = Config::load(config_path.as_deref())?;
    info!(?cfg, "loaded config");

    // The storage-root presence is the single axis that decides
    // both the registry backend and whether the service starts
    // not-ready (waiting for rehydrate).
    let (registry, readiness) = match &cfg.storage.root {
        Some(root) => (
            Arc::new(GraphRegistry::on_disk(root.clone())),
            Arc::new(Readiness::not_ready()),
        ),
        None => (
            Arc::new(GraphRegistry::in_memory()),
            Arc::new(Readiness::ready()),
        ),
    };
    let state = AppState::new(
        registry.clone(),
        cfg.auth.build_provider()?,
        readiness.clone(),
    )
    .with_limits(cfg.server.limits());

    if cfg.storage.root.is_some() {
        let rehydrated = registry.rehydrate()?;
        info!(count = rehydrated.len(), ids = ?rehydrated, "rehydrated graphs");
        readiness.mark_ready();
    }

    // Build the router once and serve it on every configured transport.
    // axum's Router is Clone, so the TCP and (optional) UDS listeners
    // share one handler stack — same routes, auth, limits, OpenAPI.
    let router = app(state);

    // One shutdown signal fans out to all listeners via a watch channel:
    // a single task awaits SIGINT/SIGTERM and flips it; every listener's
    // `with_graceful_shutdown` future (`shutdown_on`) resolves when it
    // changes. Without this, only one listener would observe the signal.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    // Each transport runs as its own task in a JoinSet so the *first* one
    // to finish surfaces immediately — if the optional UDS listener errors
    // while TCP keeps serving, we observe it at once rather than blocking
    // on TCP. On graceful shutdown every task returns ~together and the
    // loop drains them all.
    let mut servers = tokio::task::JoinSet::new();

    let tcp = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
    info!(bind = %cfg.server.bind, "dynograph listening (tcp)");
    {
        let router = router.clone();
        let rx = shutdown_rx.clone();
        servers.spawn(async move {
            axum::serve(tcp, router)
                .with_graceful_shutdown(shutdown_on(rx))
                .await
        });
    }

    // Optional Unix-domain-socket listener, bound alongside TCP.
    if let Some(path) = &cfg.server.uds_path {
        let listener = bind_uds(path)?;
        info!(uds_path = %path.display(), "dynograph listening (uds)");
        let path = path.clone();
        let router = router.clone();
        let rx = shutdown_rx.clone();
        servers.spawn(async move {
            let result = axum::serve(listener, router)
                .with_graceful_shutdown(shutdown_on(rx))
                .await;
            // Best-effort unlink so a clean shutdown doesn't leave a
            // stale socket that blocks the next bind. A failure here
            // can't fail the process, but it must not be silent.
            if let Err(e) = std::fs::remove_file(&path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                warn!(uds_path = %path.display(), "failed to remove socket on shutdown: {e}");
            }
            result
        });
    }

    // Surface a listener failure (not just shutdown) from whichever task
    // finishes first. Dropping the JoinSet on an early return aborts the
    // still-running listeners.
    while let Some(joined) = servers.join_next().await {
        joined??;
    }
    info!("shutdown complete");
    Ok(())
}

/// Graceful-shutdown future shared by every listener: resolves once the
/// shutdown watch channel flips to `true`, so a single SIGINT/SIGTERM
/// drains all transports together. `wait_for` checks the current value
/// first, so a signal that arrives before this future is polled is still
/// observed (no missed-edge hang).
async fn shutdown_on(mut rx: tokio::sync::watch::Receiver<bool>) {
    let _ = rx.wait_for(|&signalled| signalled).await;
}

/// Parse `--config <path>` from argv. Anything else is rejected.
/// Returns `None` if no `--config` was supplied (caller falls back
/// to defaults + env vars).
fn parse_config_arg() -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut config = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                let path = args.next().ok_or("--config requires a path argument")?;
                config = Some(PathBuf::from(path));
            }
            "-h" | "--help" => {
                println!("usage: dynograph [--config <path>]");
                println!();
                println!("Env vars (override TOML):");
                println!("  DYNOGRAPH_BIND          server bind address (default 127.0.0.1:8080)");
                println!(
                    "  DYNOGRAPH_UDS_PATH      extra Unix-socket path to serve on (default: TCP only)"
                );
                println!("  DYNOGRAPH_STORAGE_ROOT  RocksDB root dir; absent = in-memory");
                println!("  RUST_LOG                tracing filter (e.g. info, debug)");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    Ok(config)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            warn!("ctrl_c handler install failed: {e}");
        }
    };
    let term = async {
        match signal(SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => warn!("SIGTERM handler install failed: {e}"),
        }
    };
    tokio::select! {
        _ = ctrl_c => info!("received SIGINT"),
        _ = term => info!("received SIGTERM"),
    }
}

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
//! 5. Serve on `server.bind` until SIGINT or SIGTERM. Graceful
//!    shutdown lets in-flight requests drain.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use tokio::signal::unix::{SignalKind, signal};
use tracing::{error, info, warn};

use dynograph_service::{AppState, Config, GraphRegistry, NoAuth, Readiness, app};

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
    // Minimal — env-var driven. RUST_LOG=info is the typical setting.
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = tracing::subscriber::set_global_default(
            tracing_subscriber::FmtSubscriber::builder()
                .with_writer(std::io::stderr)
                .with_max_level(tracing::Level::INFO)
                .finish(),
        );
    });
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = parse_config_arg()?;
    let cfg = Config::load(config_path.as_deref())?;
    info!(?cfg, "loaded config");

    let registry = match &cfg.storage.root {
        Some(root) => Arc::new(GraphRegistry::on_disk(root.clone())),
        None => Arc::new(GraphRegistry::in_memory()),
    };

    let readiness = match &cfg.storage.root {
        Some(_) => Readiness::not_ready(),
        None => Readiness::ready(),
    };
    let state = AppState::new(registry.clone(), Arc::new(NoAuth::new()), readiness.clone());

    if cfg.storage.root.is_some() {
        let rehydrated = registry.rehydrate()?;
        info!(count = rehydrated.len(), ids = ?rehydrated, "rehydrated graphs");
        readiness.mark_ready();
    }

    let listener = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
    info!(bind = %cfg.server.bind, "dynograph listening");
    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    info!("shutdown complete");
    Ok(())
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

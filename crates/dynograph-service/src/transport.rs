//! Transport helpers shared by the binary.
//!
//! The TCP listener is bound inline in `main`; the Unix-domain-socket
//! path needs a little more care (clearing a stale socket from a prior
//! run), so it lives here where it can be unit-tested without spinning
//! up the whole binary.

use std::os::unix::fs::FileTypeExt;
use std::path::Path;

use thiserror::Error;
use tokio::net::UnixListener;

#[derive(Debug, Error)]
pub enum UdsBindError {
    /// The path already exists and is *not* a socket. We refuse to
    /// unlink it — deleting a regular file the operator may not have
    /// meant to point us at would be exactly the kind of silent,
    /// destructive fallback we avoid.
    #[error("uds_path {path} exists and is not a socket; refusing to overwrite")]
    NotASocket { path: String },
    #[error("uds_path {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

/// Bind a `UnixListener` at `path`, clearing a stale socket left by a
/// previous run.
///
/// - Path absent → bind fresh.
/// - Path is a leftover socket → unlink and rebind (the common
///   crash-and-restart case; `UnixListener::bind` itself errors on an
///   existing path, so this is required for restarts to work).
/// - Path is any other file type → loud [`UdsBindError::NotASocket`].
pub fn bind_uds(path: &Path) -> Result<UnixListener, UdsBindError> {
    let io_err = |source| UdsBindError::Io {
        path: path.display().to_string(),
        source,
    };
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => {
            std::fs::remove_file(path).map_err(io_err)?;
        }
        Ok(_) => {
            return Err(UdsBindError::NotASocket {
                path: path.display().to_string(),
            });
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_err(e)),
    }
    UnixListener::bind(path).map_err(io_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn binds_when_path_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fresh.sock");
        let listener = bind_uds(&path).expect("bind on a clean path");
        assert!(path.exists());
        drop(listener);
    }

    #[tokio::test]
    async fn rebinds_over_a_stale_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stale.sock");
        // First bind creates the socket file; dropping the listener
        // leaves it on disk (the kernel does not unlink it), mimicking a
        // crashed prior run.
        let first = bind_uds(&path).expect("first bind");
        drop(first);
        assert!(path.exists(), "socket file should persist after drop");
        // Second bind must succeed by unlinking the stale socket.
        let _second = bind_uds(&path).expect("rebind over stale socket");
    }

    #[tokio::test]
    async fn refuses_to_clobber_a_non_socket_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("regular.file");
        std::fs::write(&path, b"important").unwrap();
        let err = bind_uds(&path).expect_err("must refuse a regular file");
        assert!(matches!(err, UdsBindError::NotASocket { .. }), "{err}");
        // The file must be left intact, not deleted.
        assert_eq!(std::fs::read(&path).unwrap(), b"important");
    }
}

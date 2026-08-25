//! Cross-version on-disk round-trip probe.
//!
//! Answers "can a store written by one RocksDB engine be reopened by
//! another, and can it be rolled BACK afterwards?" Run it from two
//! checkouts differing only in the `rocksdb` crate version, against one
//! store path. The two engines are needed in sequence, in separate
//! processes — not simultaneously.
//!
//! Usage:
//!   cargo run --example rocksdb_roundtrip -- write <path> <tag>...
//!   cargo run --example rocksdb_roundtrip -- read  <path> <tag>...
//!
//! `write` appends a node per tag, stamped with the size of the binary
//! that wrote it. `read` asserts every named tag is still readable and
//! reports whether the reading binary differs from the writing one.
//!
//! WHY THE STAMP: cargo reuses cached artifacts aggressively, so a
//! rebuild after changing the `rocksdb` version can hand back the SAME
//! binary. That produces a green round-trip which never crossed a
//! version boundary at all — a false pass indistinguishable from a real
//! one. The size is recorded so the transcript carries the evidence
//! instead of depending on the operator remembering to check.

use dynograph_core::{Schema, props};
use dynograph_storage::StorageEngine;

fn schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: roundtrip
  version: 1
  node_types:
    Marker:
      properties:
        tag:
          type: string
          required: true
        writer_bytes:
          type: int
          required: true
  edge_types: {}
"#,
    )
    .unwrap()
}

/// Size of the running binary, as a cheap identity for the engine it
/// links. Two builds differing only in the RocksDB version differ here
/// by tens of megabytes; `None` if it cannot be determined, which is
/// reported rather than silently treated as a match.
fn own_size() -> Option<i64> {
    let exe = std::env::current_exe().ok()?;
    let len = std::fs::metadata(exe).ok()?.len();
    i64::try_from(len).ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: rocksdb_roundtrip <write|read> <path> <tag>...");
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let path = args[2].as_str();
    let tags = &args[3..];

    let size = own_size();
    match size {
        Some(n) => println!("this binary: {n} bytes"),
        None => println!("this binary: SIZE UNKNOWN (cannot compare against writer)"),
    }

    // Fail loud: an error opening the store IS the finding.
    let mut engine = match StorageEngine::new_rocksdb(schema(), path) {
        Ok(e) => e,
        Err(e) => {
            println!("OPEN FAILED: {e}");
            std::process::exit(1);
        }
    };
    println!("opened {path}");

    match mode {
        "write" => {
            for tag in tags {
                // -1 encodes "writer size unknown" so `read` can say so
                // rather than comparing against a fabricated value.
                let stamp = size.unwrap_or(-1);
                match engine.create_node(
                    "g",
                    "Marker",
                    tag,
                    props! { "tag" => tag.clone(), "writer_bytes" => stamp },
                ) {
                    Ok(_) => println!("WROTE {tag} (writer {stamp} bytes)"),
                    Err(e) => {
                        println!("WRITE FAILED {tag}: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        "read" => {
            let mut ok = true;
            for tag in tags {
                match engine.get_node("g", "Marker", tag) {
                    Ok(Some(n)) => {
                        let writer = n.properties.get("writer_bytes").and_then(|v| v.as_i64());
                        println!("READ OK {tag} -> writer {}", describe(writer, size));
                    }
                    Ok(None) => {
                        println!("READ MISSING {tag}");
                        ok = false;
                    }
                    Err(e) => {
                        println!("READ FAILED {tag}: {e}");
                        ok = false;
                    }
                }
            }
            if !ok {
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
    println!("OK");
}

/// Say whether the binary that wrote a marker is the same one reading
/// it. SAME BINARY means the run proved nothing about cross-version
/// compatibility, however green the rest of the output looks.
fn describe(writer: Option<i64>, reader: Option<i64>) -> String {
    match (writer, reader) {
        (Some(w), _) if w < 0 => "size was unknown at write time".to_string(),
        (Some(w), Some(r)) if w == r => {
            format!("{w} bytes — SAME BINARY, this crossed no version boundary")
        }
        (Some(w), Some(r)) => format!("{w} bytes vs reader {r} — DIFFERENT BINARY"),
        (Some(w), None) => format!("{w} bytes, reader size unknown — CANNOT COMPARE"),
        (None, _) => "not recorded".to_string(),
    }
}

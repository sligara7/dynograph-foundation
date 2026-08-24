//! Cross-version on-disk round-trip probe.
//!
//! Not a test and not shipped: a scratch harness for answering
//! "can a store written by one RocksDB engine be reopened by another,
//! and can it be rolled BACK afterwards?" Run the same binary from two
//! different checkouts (differing only in the `rocksdb` crate version)
//! against one store path.
//!
//! Usage:
//!   cargo run --example rocksdb_roundtrip -- write <path> <tag>
//!   cargo run --example rocksdb_roundtrip -- read  <path> <tag>...
//!
//! `write` appends a node tagged with the engine that wrote it.
//! `read` asserts every named tag is still readable.

use std::collections::HashMap;

use dynograph_core::{Schema, Value};
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
  edge_types: {}
"#,
    )
    .unwrap()
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
                let mut props = HashMap::new();
                props.insert("tag".to_string(), Value::String(tag.clone()));
                match engine.create_node("g", "Marker", tag, props) {
                    Ok(_) => println!("WROTE {tag}"),
                    Err(e) => {
                        println!("WRITE FAILED {tag}: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        "read" => {
            let mut missing = 0;
            for tag in tags {
                match engine.get_node("g", "Marker", tag) {
                    Ok(Some(n)) => println!("READ OK {tag} -> {:?}", n.properties.get("tag")),
                    Ok(None) => {
                        println!("READ MISSING {tag}");
                        missing += 1;
                    }
                    Err(e) => {
                        println!("READ FAILED {tag}: {e}");
                        missing += 1;
                    }
                }
            }
            if missing > 0 {
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

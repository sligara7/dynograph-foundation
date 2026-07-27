//! Can other processes read this index while one process holds the writer?
//!
//! Tantivy's own source says yes — `INDEX_WRITER_LOCK` exists to allow exactly
//! one *writer*, and the neighbouring `META_LOCK` is documented as making it
//! "possible for another process to safely consume our index in-writing". This
//! crate has never done it, so the claim was documentation rather than
//! evidence, and a whole feature was about to be built on top of it
//! (reflow2's `req:read-while-held`: a session that cannot take the write lock
//! should still be able to READ).
//!
//! Why it matters where it does: reflow2 keeps its full-text index INSIDE the
//! store directory, and `TextIndex::open` eagerly builds an `IndexWriter`. So
//! even once RocksDB's exclusive lock is dealt with, a second reader would fail
//! here — and nobody had noticed, because RocksDB's lock fires first.
//!
//! Separate processes on purpose. A lock that is merely per-`IndexWriter` would
//! pass an in-process test and still fail the case we care about, which is a
//! second reflow2 running beside the first.

use std::path::Path;
use std::process::Command;

use dynograph_text::TextIndex;

/// The child half. Ignored so the normal run skips it; the parent invokes it by
/// name in a separate process. It opens the index with NO writer, exactly as a
/// read-only seat would, and searches.
#[test]
#[ignore = "spawned as a child process by readers_can_open_while_a_writer_is_live"]
fn reader_child() {
    let path =
        std::env::var("DYNOGRAPH_TEXT_PROBE_PATH").expect("child needs DYNOGRAPH_TEXT_PROBE_PATH");
    let dir = tantivy::directory::MmapDirectory::open(Path::new(&path))
        .expect("a reader must be able to mmap a directory a writer holds");
    let index = tantivy::Index::open(dir).expect("a reader must be able to open the index");
    let reader = index
        .reader_builder()
        .reload_policy(tantivy::ReloadPolicy::Manual)
        .try_into()
        .expect("a reader must be constructible with no writer");
    let searcher: tantivy::IndexReader = reader;
    let n = searcher.searcher().num_docs();
    // The parent asserts on this line; a panic above fails the child, which the
    // parent then reports with the child's own stderr rather than a bare code.
    println!("PROBE_DOCS={n}");
    assert!(n > 0, "the child must see the committed documents");
}

#[test]
fn readers_can_open_while_a_writer_is_live() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fulltext");

    // The writer, held for the whole test — this is the seat that won the race.
    let writer = TextIndex::open(&path).expect("first open takes the writer");
    writer
        .upsert(
            "g",
            "Requirement",
            "req:one",
            &[(
                "statement".to_string(),
                "the design must survive a reader".to_string(),
            )],
        )
        .expect("upsert");
    writer.commit().expect("commit");

    // Two children, because "a second process" and "several processes" are
    // different claims and the fleet case is the second one.
    let mut kids = Vec::new();
    for i in 0..2 {
        let child = Command::new(std::env::current_exe().expect("current_exe"))
            .args(["--exact", "reader_child", "--ignored", "--nocapture"])
            .env("DYNOGRAPH_TEXT_PROBE_PATH", &path)
            .output()
            .unwrap_or_else(|e| panic!("could not spawn reader {i}: {e}"));
        kids.push(child);
    }

    for (i, out) in kids.iter().enumerate() {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "reader {i} failed while a writer was live.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("PROBE_DOCS=1"),
            "reader {i} did not see the committed document.\nstdout:\n{stdout}"
        );
    }

    // The writer must be unharmed by having been read — the point is coexistence,
    // not that one side survives.
    writer
        .upsert(
            "g",
            "Requirement",
            "req:two",
            &[(
                "statement".to_string(),
                "and keep writing afterwards".to_string(),
            )],
        )
        .expect("the writer still works after readers attached");
    writer.commit().expect("commit after readers");
}

/// The negative half, so the positive result cannot be read as "locks do not
/// apply here". A second WRITER must still be refused — that is the invariant
/// the read-only path is threading, not one it removes.
#[test]
fn a_second_writer_is_still_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fulltext");
    let _held = TextIndex::open(&path).expect("first open takes the writer");

    let second = TextIndex::open(&path);
    assert!(
        second.is_err(),
        "a second writer on one index must be refused; if this ever passes, the \
         single-writer guarantee the read-only path relies on has gone"
    );
}

/// The child half for `open_read_only`, spawned the same way — this is the API
/// a locked-out reflow2 seat would actually call, rather than raw tantivy.
#[test]
#[ignore = "spawned as a child process by open_read_only_works_while_a_writer_is_live"]
fn read_only_child() {
    let path =
        std::env::var("DYNOGRAPH_TEXT_PROBE_PATH").expect("child needs DYNOGRAPH_TEXT_PROBE_PATH");
    let ro = TextIndex::open_read_only(Path::new(&path))
        .expect("open_read_only must succeed while another process holds the writer");

    let hits = ro
        .search("g", "reader", None, 10)
        .expect("a read-only index must still search");
    println!("PROBE_HITS={}", hits.len());
    assert!(
        !hits.is_empty(),
        "the read-only seat must see committed docs"
    );

    // The half that matters more than the search: a write must be REFUSED, and
    // say so. A read-only index that swallowed writes would report success
    // while the caller's data went nowhere.
    let refused = ro.upsert(
        "g",
        "Requirement",
        "req:nope",
        &[("statement".to_string(), "should never land".to_string())],
    );
    let err = refused.expect_err("a write on a read-only index must be refused");
    let msg = err.to_string();
    println!("PROBE_REFUSAL={msg}");
    assert!(msg.contains("READ-ONLY"), "the refusal must say why: {msg}");
    assert!(msg.contains("upsert"), "and which operation: {msg}");
}

#[test]
fn open_read_only_works_while_a_writer_is_live() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fulltext");

    let writer = TextIndex::open(&path).expect("first open takes the writer");
    writer
        .upsert(
            "g",
            "Requirement",
            "req:one",
            &[(
                "statement".to_string(),
                "the design must survive a reader".to_string(),
            )],
        )
        .expect("upsert");
    writer.commit().expect("commit");

    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", "read_only_child", "--ignored", "--nocapture"])
        .env("DYNOGRAPH_TEXT_PROBE_PATH", &path)
        .output()
        .expect("spawn read-only child");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "open_read_only failed beside a live writer.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("PROBE_HITS=1"), "stdout:\n{stdout}");
    assert!(stdout.contains("PROBE_REFUSAL="), "stdout:\n{stdout}");
}

#[test]
fn a_read_only_open_never_creates_an_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("not-an-index");
    std::fs::create_dir_all(&missing).expect("mkdir");

    let opened = TextIndex::open_read_only(&missing);
    assert!(
        opened.is_err(),
        "a read-only open of a directory with no index must fail"
    );
    // Silently inventing an empty index would answer "no results" for a design
    // that exists and is merely somewhere else — the worst possible reply.
    assert!(
        !missing.join("meta.json").exists(),
        "a read-only open must not have created an index"
    );
}

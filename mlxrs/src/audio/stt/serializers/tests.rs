//! Hand-traced unit tests — exercise the python-equivalent formatting
//! decisions without going through the filesystem (those live in the
//! `tests/audio_stt_serializers.rs` integration test).

use super::*;

#[test]
fn format_timestamp_zero() {
  assert_eq!(format_timestamp(0.0), "00:00:00,000");
}

#[test]
fn format_timestamp_subsecond() {
  assert_eq!(format_timestamp(1.234), "00:00:01,234");
}

#[test]
fn format_timestamp_pre_minute_rollover() {
  assert_eq!(format_timestamp(59.999), "00:00:59,999");
}

#[test]
fn format_timestamp_minute_rollover() {
  assert_eq!(format_timestamp(60.0), "00:01:00,000");
}

#[test]
fn format_timestamp_pre_hour_rollover() {
  assert_eq!(format_timestamp(3599.999), "00:59:59,999");
}

#[test]
fn format_timestamp_hour_rollover() {
  assert_eq!(format_timestamp(3600.0), "01:00:00,000");
}

#[test]
fn format_timestamp_compound() {
  assert_eq!(format_timestamp(3661.123), "01:01:01,123");
}

#[test]
fn format_vtt_timestamp_uses_dot() {
  // mlx-audio python: `format_timestamp(s).replace(",", ".")`.
  // Same table as `format_timestamp_*` but with `.` instead of `,`.
  assert_eq!(format_vtt_timestamp(0.0), "00:00:00.000");
  assert_eq!(format_vtt_timestamp(1.234), "00:00:01.234");
  assert_eq!(format_vtt_timestamp(3661.123), "01:01:01.123");
}

#[test]
fn get_cues_segments_no_words() {
  let t = Transcript::Segments(SegmentsPayload::new(
    "hello world",
    vec![
      Segment::new(0.0, 1.0, "hello", vec![], ""),
      Segment::new(1.0, 2.0, "world", vec![], ""),
    ],
  ));
  let cues = get_cues(&t);
  assert_eq!(cues.len(), 2);
  assert_eq!(cues[0].text(), "hello");
  assert_eq!(cues[1].text(), "world");
}

#[test]
fn get_cues_segments_with_words() {
  // Python `_get_cues` emits segment-level cue THEN per-word cues.
  let t = Transcript::Segments(SegmentsPayload::new(
    "hi there",
    vec![Segment::new(
      0.0,
      1.0,
      "hi there",
      vec![
        Word::new(0.0, 0.5, "hi", BTreeMap::new()),
        Word::new(0.5, 1.0, "there", BTreeMap::new()),
      ],
      "",
    )],
  ));
  let cues = get_cues(&t);
  // 1 segment cue + 2 word cues = 3 total
  assert_eq!(cues.len(), 3);
  assert_eq!(cues[0].text(), "hi there");
  assert_eq!(cues[1].text(), "hi");
  assert_eq!(cues[2].text(), "there");
}

#[test]
fn get_cues_sentences_one_per_sentence() {
  // Python sentence branch: one cue per sentence, NO per-token cues.
  let t = Transcript::Sentences(SentencesPayload::new(
    "hi world",
    vec![
      Sentence::new(
        "hi",
        0.0,
        1.0,
        1.0,
        vec![SentenceToken::new("h", 0.0, 0.5, 0.5)],
        "",
      ),
      Sentence::new("world", 1.0, 2.0, 1.0, vec![], ""),
    ],
  ));
  let cues = get_cues(&t);
  // 2 sentences ⇒ 2 cues (NO per-token cues for the Sentences variant).
  assert_eq!(cues.len(), 2);
  assert_eq!(cues[0].text(), "hi");
  assert_eq!(cues[1].text(), "world");
}

#[test]
fn transcript_text_accessor() {
  let t1 = Transcript::Segments(SegmentsPayload::new("alpha", vec![]));
  let t2 = Transcript::Sentences(SentencesPayload::new("beta", vec![]));
  assert_eq!(t1.text(), "alpha");
  assert_eq!(t2.text(), "beta");
}

#[test]
fn with_extension_appends_not_replaces() {
  use std::path::PathBuf;
  assert_eq!(
    super::with_extension(Path::new("out"), "txt"),
    PathBuf::from("out.txt"),
  );
  // Crucially NOT `out.txt` — python `f"{output_path}.{ext}"` appends, so
  // `output_path = "out.draft"` becomes `"out.draft.txt"`.
  assert_eq!(
    super::with_extension(Path::new("out.draft"), "txt"),
    PathBuf::from("out.draft.txt"),
  );
}

/// Hand-traced fixture matching the SRT/VTT byte-exact integration tests
/// for the dash-path stdout-writer round-trip — three contiguous Whisper
/// segments, no per-word alignment.
fn three_segments_fixture() -> Transcript {
  Transcript::Segments(SegmentsPayload::new(
    "hello world foo",
    vec![
      Segment::new(0.0, 1.234, "hello", vec![], ""),
      Segment::new(1.234, 2.500, "world", vec![], ""),
      Segment::new(2.500, 4.000, "foo", vec![], ""),
    ],
  ))
}

// ---------- regression: writer-helper round-trip ----------

#[test]
fn save_as_txt_to_writer_matches_python_body() {
  // mlxrs writer helper that's shared by `save_as_txt`'s file branch AND
  // the `path == "-"` stdout branch — assert the python-shape body is
  // identical regardless of sink (so the dash-path stdout output is
  // byte-exact-equivalent to the on-disk `.txt` body).
  let t = three_segments_fixture();
  let mut buf: Vec<u8> = Vec::new();
  super::save_as_txt_to_writer(&t, &mut buf).unwrap();
  assert_eq!(
    std::str::from_utf8(&buf).unwrap(),
    "hello world foo",
    "writer helper writes the same body the on-disk `.txt` test asserts"
  );
}

#[test]
fn save_as_srt_to_writer_matches_python_body() {
  let t = three_segments_fixture();
  let mut buf: Vec<u8> = Vec::new();
  super::save_as_srt_to_writer(&t, &mut buf).unwrap();
  let expected = "1\n00:00:00,000 --> 00:00:01,234\nhello\n\n\
                    2\n00:00:01,234 --> 00:00:02,500\nworld\n\n\
                    3\n00:00:02,500 --> 00:00:04,000\nfoo\n\n";
  assert_eq!(
    std::str::from_utf8(&buf).unwrap(),
    expected,
    "writer helper writes the same body the on-disk `.srt` test asserts"
  );
}

#[test]
fn save_as_vtt_to_writer_matches_python_body() {
  let t = three_segments_fixture();
  let mut buf: Vec<u8> = Vec::new();
  super::save_as_vtt_to_writer(&t, &mut buf).unwrap();
  let expected = "WEBVTT\n\n\
                    1\n00:00:00.000 --> 00:00:01.234\nhello\n\n\
                    2\n00:00:01.234 --> 00:00:02.500\nworld\n\n\
                    3\n00:00:02.500 --> 00:00:04.000\nfoo\n\n";
  assert_eq!(
    std::str::from_utf8(&buf).unwrap(),
    expected,
    "writer helper writes the same body the on-disk `.vtt` test asserts"
  );
}

#[test]
fn save_as_json_to_writer_matches_python_body() {
  // `save_as_json_to_writer` is the shared sink for `save_as_json`'s file
  // branch + dash-path stdout branch. The python output is 2-space
  // pretty-printed; we round-trip the buffer through `serde_json` to
  // assert structural equality (the byte-exact assertion is exercised by
  // the existing integration test `save_as_json_round_trips_transcript`).
  let t = three_segments_fixture();
  let mut buf: Vec<u8> = Vec::new();
  super::save_as_json_to_writer(&t, &mut buf).unwrap();
  let parsed: Transcript = serde_json::from_slice(&buf).unwrap();
  assert_eq!(
    parsed, t,
    "writer helper writes the same python-shape JSON the on-disk `.json` test asserts"
  );
  // Sanity: 2-space indent is present (python `indent=2`).
  assert!(
    std::str::from_utf8(&buf).unwrap().contains("\n  \"text\":"),
    "writer JSON output uses 2-space indent"
  );
}

// ---------- dash-stdout flush coverage ----------

/// Test-only [`Write`] adapter that records whether [`flush`] was called +
/// (optionally) returns an [`io::Error`] from [`flush`]. Used to exercise
/// the dash-stdout `save_as_*_stdout` delegate paths without going through
/// a real stdout fd:
///
/// - `fail_flush = false` → buffers the writes, counts the flush calls.
///   Used by [`save_as_txt_writer_helper_flushes_via_explicit_call`] to
///   assert that the dash-stdout delegate actually calls `.flush()` (not
///   relying on the writer's drop).
/// - `fail_flush = true` → returns [`io::Error::other`] from `.flush()`.
///   Used by the per-format `*_stdout_flush_failure_surfaces_as_backend_error`
///   tests to assert the flush-error path surfaces as [`Error::FileIo`]
///   with the `"stdout flush failed"` marker.
struct FailingFlushWriter {
  buf: Vec<u8>,
  flush_calls: usize,
  fail_flush: bool,
}

impl FailingFlushWriter {
  fn ok() -> Self {
    Self {
      buf: Vec::new(),
      flush_calls: 0,
      fail_flush: false,
    }
  }

  fn failing() -> Self {
    Self {
      buf: Vec::new(),
      flush_calls: 0,
      fail_flush: true,
    }
  }
}

impl Write for FailingFlushWriter {
  fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
    self.buf.extend_from_slice(bytes);
    Ok(bytes.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
    self.flush_calls += 1;
    if self.fail_flush {
      Err(std::io::Error::other("induced flush failure"))
    } else {
      Ok(())
    }
  }
}

#[test]
fn save_as_txt_writer_helper_flushes_via_explicit_call() {
  // The dash-stdout `save_as_*_stdout` delegate MUST call `.flush()`
  // explicitly — relying on the writer's `Drop` is insufficient because
  // `StdoutLock`'s drop does NOT flush the underlying stdout buffer
  // (the lock guard only releases the lock; the buffer outlives it).
  // Assert one flush call per save_as_*_stdout invocation across all
  // four serializers.
  let t = three_segments_fixture();
  for (name, run) in [
    (
      "save_as_txt_stdout",
      Box::new(|t: &Transcript, w: &mut FailingFlushWriter| super::save_as_txt_stdout(t, w))
        as Box<dyn Fn(&Transcript, &mut FailingFlushWriter) -> Result<()>>,
    ),
    (
      "save_as_srt_stdout",
      Box::new(|t: &Transcript, w: &mut FailingFlushWriter| super::save_as_srt_stdout(t, w)),
    ),
    (
      "save_as_vtt_stdout",
      Box::new(|t: &Transcript, w: &mut FailingFlushWriter| super::save_as_vtt_stdout(t, w)),
    ),
    (
      "save_as_json_stdout",
      Box::new(|t: &Transcript, w: &mut FailingFlushWriter| super::save_as_json_stdout(t, w)),
    ),
  ] {
    let mut w = FailingFlushWriter::ok();
    run(&t, &mut w).unwrap_or_else(|e| panic!("{name} on ok-writer must succeed: {e}"));
    assert_eq!(
      w.flush_calls, 1,
      "{name} must call .flush() exactly once on the writer (saw {})",
      w.flush_calls
    );
    assert!(
      !w.buf.is_empty(),
      "{name} must have written body bytes before flushing"
    );
  }
}

#[test]
fn save_as_txt_stdout_flush_failure_surfaces_as_backend_error() {
  let t = three_segments_fixture();
  let mut w = FailingFlushWriter::failing();
  let err =
    super::save_as_txt_stdout(&t, &mut w).expect_err("flush-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert!(
        p.context().contains("save_as_txt") && p.context().contains("stdout flush failed"),
        "FileIo context must mention save_as_txt + stdout flush failure (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Flush, "op kind must be Flush");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert_eq!(
    w.flush_calls, 1,
    "flush must have been attempted exactly once"
  );
}

#[test]
fn save_as_srt_stdout_flush_failure_surfaces_as_backend_error() {
  let t = three_segments_fixture();
  let mut w = FailingFlushWriter::failing();
  let err =
    super::save_as_srt_stdout(&t, &mut w).expect_err("flush-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert!(
        p.context().contains("save_as_srt") && p.context().contains("stdout flush failed"),
        "FileIo context must mention save_as_srt + stdout flush failure (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Flush, "op kind must be Flush");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert_eq!(
    w.flush_calls, 1,
    "flush must have been attempted exactly once"
  );
}

#[test]
fn save_as_vtt_stdout_flush_failure_surfaces_as_backend_error() {
  let t = three_segments_fixture();
  let mut w = FailingFlushWriter::failing();
  let err =
    super::save_as_vtt_stdout(&t, &mut w).expect_err("flush-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert!(
        p.context().contains("save_as_vtt") && p.context().contains("stdout flush failed"),
        "FileIo context must mention save_as_vtt + stdout flush failure (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Flush, "op kind must be Flush");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert_eq!(
    w.flush_calls, 1,
    "flush must have been attempted exactly once"
  );
}

#[test]
fn save_as_json_stdout_flush_failure_surfaces_as_backend_error() {
  let t = three_segments_fixture();
  let mut w = FailingFlushWriter::failing();
  let err =
    super::save_as_json_stdout(&t, &mut w).expect_err("flush-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert!(
        p.context().contains("save_as_json") && p.context().contains("stdout flush failed"),
        "FileIo context must mention save_as_json + stdout flush failure (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Flush, "op kind must be Flush");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert_eq!(
    w.flush_calls, 1,
    "flush must have been attempted exactly once"
  );
}

// ---------- dash-stdout WRITE-failure coverage ----------

/// Test-only [`Write`] adapter that fails on the FIRST `write` call (returning
/// [`io::Error::other`]) — the complement of [`FailingFlushWriter`], which
/// only fails `flush`. Used to drive the `save_as_*_stdout` delegates'
/// *body-write* error arm (the `save_as_*_to_writer(..).map_err(..)` rather
/// than the `flush().map_err(..)`), surfacing as [`Error::FileIo`] with the
/// `"write to stdout failed"` / `"serialize to stdout failed"` marker and
/// [`FileOp::Write`].
///
/// `flush` is left succeeding so the test isolates the WRITE failure: the
/// delegate must short-circuit on the failed body write and never reach the
/// (succeeding) flush. `write_calls` records that exactly one write was
/// attempted; for the JSON sink `serde_json::to_writer_pretty` issues its
/// first byte (`{`) through this writer, so the very first write fails and is
/// folded into an io::Error by `save_as_json_to_writer`.
struct FailingWriteWriter {
  write_calls: usize,
  flush_calls: usize,
}

impl FailingWriteWriter {
  fn new() -> Self {
    Self {
      write_calls: 0,
      flush_calls: 0,
    }
  }
}

impl Write for FailingWriteWriter {
  fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
    self.write_calls += 1;
    Err(std::io::Error::other("induced write failure"))
  }

  fn flush(&mut self) -> std::io::Result<()> {
    self.flush_calls += 1;
    Ok(())
  }
}

#[test]
fn save_as_txt_stdout_write_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let mut w = FailingWriteWriter::new();
  let err =
    super::save_as_txt_stdout(&t, &mut w).expect_err("write-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.context(),
        "save_as_txt: write to stdout failed",
        "FileIo context must be the txt stdout WRITE marker (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Write, "op kind must be Write");
      assert_eq!(
        p.inner().to_string(),
        "induced write failure",
        "the underlying induced io::Error must be threaded through verbatim"
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert_eq!(w.write_calls, 1, "exactly one write must be attempted");
  assert_eq!(
    w.flush_calls, 0,
    "the failed body write must short-circuit before flush"
  );
}

#[test]
fn save_as_srt_stdout_write_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let mut w = FailingWriteWriter::new();
  let err =
    super::save_as_srt_stdout(&t, &mut w).expect_err("write-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.context(),
        "save_as_srt: write to stdout failed",
        "FileIo context must be the srt stdout WRITE marker (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Write, "op kind must be Write");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert_eq!(w.write_calls, 1, "exactly one write must be attempted");
  assert_eq!(
    w.flush_calls, 0,
    "the failed body write must short-circuit before flush"
  );
}

#[test]
fn save_as_vtt_stdout_write_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let mut w = FailingWriteWriter::new();
  let err =
    super::save_as_vtt_stdout(&t, &mut w).expect_err("write-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.context(),
        "save_as_vtt: write to stdout failed",
        "FileIo context must be the vtt stdout WRITE marker (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Write, "op kind must be Write");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  // The VTT body's first write is the `WEBVTT\n\n` header — that single write
  // fails, short-circuiting before any cue block or the flush.
  assert_eq!(w.write_calls, 1, "exactly one write must be attempted");
  assert_eq!(
    w.flush_calls, 0,
    "the failed header write must short-circuit before flush"
  );
}

#[test]
fn save_as_json_stdout_write_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let mut w = FailingWriteWriter::new();
  let err =
    super::save_as_json_stdout(&t, &mut w).expect_err("write-failing writer must produce an Err");
  match err {
    Error::FileIo(p) => {
      // `save_as_json_to_writer` folds the `serde_json::Error` raised when the
      // first emitted byte (`{`) fails into an `io::Error` via
      // `io::Error::other`, and `save_as_json_stdout` re-wraps it with the
      // serialize-to-stdout marker + `FileOp::Write`.
      assert_eq!(
        p.context(),
        "save_as_json: serialize to stdout failed",
        "FileIo context must be the json stdout SERIALIZE marker (got: {})",
        p.context()
      );
      assert_eq!(p.op(), FileOp::Write, "op kind must be Write");
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
  assert!(
    w.write_calls >= 1,
    "serde must have attempted at least one write before failing"
  );
  assert_eq!(
    w.flush_calls, 0,
    "the failed serialize write must short-circuit before flush"
  );
}

// ---------- file-branch File::create-failure coverage ----------

/// A base path whose PARENT directory does not exist, so that the file-branch
/// `File::create(with_extension(path, ext))` fails deterministically with
/// [`std::io::ErrorKind::NotFound`] — exercising the `save_as_*`
/// Create-error `map_err` arm without depending on permissions / quota.
///
/// The non-existent middle component is process- + name- scoped so concurrent
/// test binaries never accidentally materialize it.
fn nonexistent_parent_base(name: &str) -> std::path::PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!(
    "mlxrs_serializers_unit_{}_{}_nonexistent_dir",
    std::process::id(),
    name
  ));
  // `p` (the middle dir) is never created; pushing a leaf makes
  // `File::create` fail at the missing parent.
  p.push("out");
  p
}

#[test]
fn save_as_txt_create_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let base = nonexistent_parent_base("txt");
  let err = super::save_as_txt(&t, &base)
    .expect_err("File::create under a missing parent dir must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.context(),
        "save_as_txt",
        "FileIo context must be the bare fn label for the file-branch create arm"
      );
      assert_eq!(p.op(), FileOp::Create, "op kind must be Create");
      assert_eq!(
        p.inner().kind(),
        std::io::ErrorKind::NotFound,
        "missing parent dir yields NotFound"
      );
      // The payload path is the EXTENSION-APPENDED final path (`out.txt`),
      // never the caller's bare base (faithful to `with_extension`).
      assert!(
        p.path().to_string_lossy().ends_with("out.txt"),
        "payload path must be the `.txt`-appended final path (got {})",
        p.path().display()
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
}

#[test]
fn save_as_srt_create_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let base = nonexistent_parent_base("srt");
  let err = super::save_as_srt(&t, &base)
    .expect_err("File::create under a missing parent dir must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(p.context(), "save_as_srt");
      assert_eq!(p.op(), FileOp::Create, "op kind must be Create");
      assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
      assert!(
        p.path().to_string_lossy().ends_with("out.srt"),
        "payload path must be the `.srt`-appended final path (got {})",
        p.path().display()
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
}

#[test]
fn save_as_vtt_create_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let base = nonexistent_parent_base("vtt");
  let err = super::save_as_vtt(&t, &base)
    .expect_err("File::create under a missing parent dir must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(p.context(), "save_as_vtt");
      assert_eq!(p.op(), FileOp::Create, "op kind must be Create");
      assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
      assert!(
        p.path().to_string_lossy().ends_with("out.vtt"),
        "payload path must be the `.vtt`-appended final path (got {})",
        p.path().display()
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
}

#[test]
fn save_as_json_create_failure_surfaces_as_file_io() {
  let t = three_segments_fixture();
  let base = nonexistent_parent_base("json");
  let err = super::save_as_json(&t, &base)
    .expect_err("File::create under a missing parent dir must produce an Err");
  match err {
    Error::FileIo(p) => {
      assert_eq!(p.context(), "save_as_json");
      assert_eq!(p.op(), FileOp::Create, "op kind must be Create");
      assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
      assert!(
        p.path().to_string_lossy().ends_with("out.json"),
        "payload path must be the `.json`-appended final path (got {})",
        p.path().display()
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
  }
}

// ---------- file-branch happy-path round-trip (unit, no integration gate) ----------

/// Exercise the `save_as_*` ON-DISK success path (`File::create` →
/// `*_to_writer` → `BufWriter::flush` → `Ok(())`) end-to-end through a real
/// temp file, asserting byte-exact bodies against hand-written oracles. This
/// complements the integration suite (`tests/audio_stt_serializers.rs`, gated
/// behind `feature = "audio"`) so the file-branch success arms are covered by
/// the crate's own `mod tests` as well.
#[test]
fn save_as_all_formats_write_byte_exact_files() {
  let t = three_segments_fixture();
  let mut base = std::env::temp_dir();
  base.push(format!(
    "mlxrs_serializers_unit_{}_roundtrip",
    std::process::id()
  ));

  // ---- TXT: `segments.text` verbatim, no trailing newline. ----
  super::save_as_txt(&t, &base).expect("save_as_txt must succeed on a writable temp path");
  let txt_path = super::with_extension(&base, "txt");
  let txt = std::fs::read_to_string(&txt_path).expect("txt file must exist");
  assert_eq!(txt, "hello world foo");
  let _ = std::fs::remove_file(&txt_path);

  // ---- SRT: 1-based index, `,` separator, `\n\n` cue separator. ----
  super::save_as_srt(&t, &base).expect("save_as_srt must succeed");
  let srt_path = super::with_extension(&base, "srt");
  let srt = std::fs::read_to_string(&srt_path).expect("srt file must exist");
  let srt_expected = "1\n00:00:00,000 --> 00:00:01,234\nhello\n\n\
                      2\n00:00:01,234 --> 00:00:02,500\nworld\n\n\
                      3\n00:00:02,500 --> 00:00:04,000\nfoo\n\n";
  assert_eq!(srt, srt_expected);
  let _ = std::fs::remove_file(&srt_path);

  // ---- VTT: `WEBVTT\n\n` header + `.` separator. ----
  super::save_as_vtt(&t, &base).expect("save_as_vtt must succeed");
  let vtt_path = super::with_extension(&base, "vtt");
  let vtt = std::fs::read_to_string(&vtt_path).expect("vtt file must exist");
  let vtt_expected = "WEBVTT\n\n\
                      1\n00:00:00.000 --> 00:00:01.234\nhello\n\n\
                      2\n00:00:01.234 --> 00:00:02.500\nworld\n\n\
                      3\n00:00:02.500 --> 00:00:04.000\nfoo\n\n";
  assert_eq!(vtt, vtt_expected);
  let _ = std::fs::remove_file(&vtt_path);

  // ---- JSON: round-trips losslessly back into the same Transcript. ----
  super::save_as_json(&t, &base).expect("save_as_json must succeed");
  let json_path = super::with_extension(&base, "json");
  let raw = std::fs::read_to_string(&json_path).expect("json file must exist");
  let parsed: Transcript = serde_json::from_str(&raw).expect("json parses back into Transcript");
  assert_eq!(
    parsed, t,
    "on-disk JSON round-trips to the original Transcript"
  );
  let _ = std::fs::remove_file(&json_path);
}

// ---------- transcript_to_python_shape: extra word fields + segment speaker_id ----------

#[test]
fn transcript_to_python_shape_segments_emits_extra_word_fields_and_speaker_id() {
  use serde_json::json;

  // A Whisper-style segment carrying BOTH a non-empty `speaker_id` (exercises
  // the `Segments`-branch `speaker_id` insertion) AND a word with non-empty
  // `extra` fields (exercises the per-word `for (k, v) in w.extra()`
  // pass-through loop). The integration suite never populates either, so this
  // is the only coverage of those two arms.
  let mut extra = BTreeMap::new();
  extra.insert("probability".to_owned(), json!(0.875));
  extra.insert("scored".to_owned(), json!(true));

  let t = Transcript::Segments(SegmentsPayload::new(
    "hi",
    vec![Segment::new(
      0.0,
      1.5,
      "hi",
      vec![Word::new(0.0, 0.5, "hi", extra)],
      "spk_7",
    )],
  ));

  // Independent, closed-form oracle: hand-build the exact python-shape Value.
  // Per-segment dict order is {text, start, end, duration, words, speaker_id};
  // `duration = end - start = 1.5 - 0.0`; per-word dict is {start, end, word}
  // then the `extra` keys in BTreeMap (lexicographic) order:
  // `probability` < `scored`.
  let expected = json!({
    "text": "hi",
    "segments": [
      {
        "text": "hi",
        "start": 0.0,
        "end": 1.5,
        "duration": 1.5,
        "words": [
          {
            "start": 0.0,
            "end": 0.5,
            "word": "hi",
            "probability": 0.875,
            "scored": true
          }
        ],
        "speaker_id": "spk_7"
      }
    ]
  });

  let got = super::transcript_to_python_shape(&t);
  assert_eq!(
    got, expected,
    "Segments python-shape must inline extra per-word fields + the segment speaker_id"
  );

  // Key-ORDER assertions (serde_json::Value compares structurally, so also
  // pin the deterministic insertion order the writer relies on for byte-exact
  // python parity). The word object's keys must be start, end, word, then the
  // sorted extra keys.
  let word_obj = got["segments"][0]["words"][0]
    .as_object()
    .expect("word is an object");
  let word_keys: Vec<&str> = word_obj.keys().map(String::as_str).collect();
  assert_eq!(
    word_keys,
    vec!["start", "end", "word", "probability", "scored"],
    "per-word key order: fixed {{start,end,word}} then BTreeMap-sorted extra keys"
  );
  let seg_obj = got["segments"][0]
    .as_object()
    .expect("segment is an object");
  let seg_keys: Vec<&str> = seg_obj.keys().map(String::as_str).collect();
  assert_eq!(
    seg_keys,
    vec!["text", "start", "end", "duration", "words", "speaker_id"],
    "per-segment key order matches the python dict-insertion order"
  );
}

#[test]
fn transcript_to_python_shape_sentences_emits_speaker_id_and_token_order() {
  use serde_json::json;

  // Parakeet-style sentence carrying a non-empty `speaker_id` (the
  // `Sentences`-branch speaker_id arm) + a token, to pin the {text, start,
  // end, duration, tokens, speaker_id} key order against a closed-form oracle.
  let t = Transcript::Sentences(SentencesPayload::new(
    "hi",
    vec![Sentence::new(
      "hi",
      0.0,
      0.5,
      0.5,
      vec![SentenceToken::new("h", 0.0, 0.25, 0.25)],
      "spk_3",
    )],
  ));

  let expected = json!({
    "text": "hi",
    "sentences": [
      {
        "text": "hi",
        "start": 0.0,
        "end": 0.5,
        "duration": 0.5,
        "tokens": [
          { "text": "h", "start": 0.0, "end": 0.25, "duration": 0.25 }
        ],
        "speaker_id": "spk_3"
      }
    ]
  });

  let got = super::transcript_to_python_shape(&t);
  assert_eq!(got, expected);

  let sent_obj = got["sentences"][0]
    .as_object()
    .expect("sentence is an object");
  let sent_keys: Vec<&str> = sent_obj.keys().map(String::as_str).collect();
  assert_eq!(
    sent_keys,
    vec!["text", "start", "end", "duration", "tokens", "speaker_id"],
    "per-sentence key order matches the python dict-insertion order"
  );
}

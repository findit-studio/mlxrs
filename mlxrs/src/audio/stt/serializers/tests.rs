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

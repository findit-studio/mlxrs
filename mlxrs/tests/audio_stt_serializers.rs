//! Integration tests for the [`mlxrs::audio::stt::serializers`] surface
//! (issue #176, AUDIO-A13): transcript file writers (TXT / SRT / WebVTT /
//! JSON) and the `format_timestamp` / `format_vtt_timestamp` helpers.
//!
//! Hand-traced fixtures + byte-exact file-content asserts so a regression in
//! the python-port shape (1-based index, `,` vs `.` separator, trailing
//! blank line, JSON key order) is caught here rather than at downstream
//! tooling that string-matches the SRT/VTT/JSON output.
#![cfg(feature = "audio")]

use std::{collections::BTreeMap, fs, path::PathBuf, process};

use mlxrs::audio::stt::serializers::{
  Segment, SegmentsPayload, Sentence, SentenceToken, SentencesPayload, Transcript, Word,
  save_as_json, save_as_srt, save_as_txt, save_as_vtt,
};

/// Process-scoped + named tempfile so parallel test binaries / cases never
/// collide. Mirrors the convention `tests/audio_stt.rs` uses.
fn temp_base(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!(
    "mlxrs_audio_stt_serializers_{}_{}",
    process::id(),
    name
  ));
  p
}

/// A 3-segment Whisper-style transcript fixture — three contiguous segments,
/// no word-level alignment, no speaker_id. Hand-traced timestamps so the
/// SRT/VTT/TXT byte-exact asserts are unambiguous.
fn fixture_3_segments() -> Transcript {
  Transcript::Segments(SegmentsPayload::new(
    "hello world foo",
    vec![
      Segment::new(0.0, 1.234, "hello", vec![], ""),
      Segment::new(1.234, 2.500, "world", vec![], ""),
      Segment::new(2.500, 4.000, "foo", vec![], ""),
    ],
  ))
}

#[test]
fn save_as_txt_writes_plain_lines() {
  // python `save_as_txt` (stt/generate.py:135-141) writes `segments.text`
  // verbatim to `f"{output_path}.txt"` — no per-segment newlines, no
  // trailing newline, no transformations. mlxrs mirrors that exactly.
  let base = temp_base("txt_plain");
  let t = fixture_3_segments();
  save_as_txt(&t, &base).unwrap();
  let txt_path = base.with_file_name(format!(
    "{}.txt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&txt_path).unwrap();
  assert_eq!(
    contents, "hello world foo",
    "save_as_txt writes `segments.text` verbatim (no per-segment lines, no trailing \\n)"
  );
  let _ = fs::remove_file(&txt_path);
}

#[test]
fn save_as_srt_writes_subrip_format() {
  // python `save_as_srt` per-cue format:
  //   {idx}\n{HH:MM:SS,mmm} --> {HH:MM:SS,mmm}\n{text}\n\n
  // Indexing 1-based; `,` separator (SRT spec).
  let base = temp_base("srt_format");
  let t = fixture_3_segments();
  save_as_srt(&t, &base).unwrap();
  let srt_path = base.with_file_name(format!(
    "{}.srt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&srt_path).unwrap();
  let expected = "1\n00:00:00,000 --> 00:00:01,234\nhello\n\n\
                  2\n00:00:01,234 --> 00:00:02,500\nworld\n\n\
                  3\n00:00:02,500 --> 00:00:04,000\nfoo\n\n";
  assert_eq!(
    contents, expected,
    "save_as_srt: 1-based index, `,` separator, double-newline cue separator"
  );
  let _ = fs::remove_file(&srt_path);
}

#[test]
fn save_as_vtt_writes_webvtt_format() {
  // python `save_as_vtt` per-cue format:
  //   WEBVTT\n\n{idx}\n{HH:MM:SS.mmm} --> {HH:MM:SS.mmm}\n{text}\n\n
  // Indexing 1-based; `.` separator (WebVTT spec); WEBVTT header required.
  let base = temp_base("vtt_format");
  let t = fixture_3_segments();
  save_as_vtt(&t, &base).unwrap();
  let vtt_path = base.with_file_name(format!(
    "{}.vtt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&vtt_path).unwrap();
  let expected = "WEBVTT\n\n\
                  1\n00:00:00.000 --> 00:00:01.234\nhello\n\n\
                  2\n00:00:01.234 --> 00:00:02.500\nworld\n\n\
                  3\n00:00:02.500 --> 00:00:04.000\nfoo\n\n";
  assert_eq!(
    contents, expected,
    "save_as_vtt: WEBVTT header, 1-based index, `.` separator, double-newline cue separator"
  );
  let _ = fs::remove_file(&vtt_path);
}

#[test]
fn save_as_json_round_trips_transcript() {
  // python `save_as_json` (stt/generate.py:173-225) writes a 2-space-indent
  // JSON tree we can round-trip via serde_json. The python Whisper shape is:
  //   {"text": ..., "segments": [{"text", "start", "end", "duration",
  //                               [optional "words"], [optional "speaker_id"]}, ...]}
  // duration is `end - start` (computed, not carried on Segment).
  let base = temp_base("json_roundtrip");
  let t = fixture_3_segments();
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  // Sanity: file is 2-space-indented (python `indent=2` matches
  // `serde_json::to_writer_pretty`'s 2-space default).
  assert!(raw.contains("\n  \"text\":"), "JSON uses 2-space indent");
  assert!(
    raw.contains("\"segments\""),
    "JSON top-level has `segments` key"
  );
  // Round-trip into a `Transcript` and assert structural equality. Note:
  // the JSON shape mlxrs writes includes a computed `duration` field per
  // segment that `Segment` doesn't carry — `Transcript::Deserialize` will
  // see + ignore it (serde_json default behavior for unknown fields).
  let parsed: Transcript = serde_json::from_str(&raw).expect("JSON parses back into Transcript");
  // Equality holds because `Segment` doesn't carry `duration` (and the
  // input had no `words` / `speaker_id`), so the round-trip is lossless
  // for the fields `Transcript` declares.
  assert_eq!(
    parsed, t,
    "JSON round-trips back to the original Transcript"
  );
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_json_sentence_shape_includes_tokens() {
  // python sentence branch (`hasattr(segments, "sentences")`) — output JSON
  // is `{"text": ..., "sentences": [{"text", "start", "end", "duration",
  //                                  "tokens": [{"text", "start", "end", "duration"}, ...],
  //                                  [optional "speaker_id"]}, ...]}`.
  let base = temp_base("json_sentence");
  let t = Transcript::Sentences(SentencesPayload::new(
    "hi",
    vec![Sentence::new(
      "hi",
      0.0,
      0.5,
      0.5,
      vec![SentenceToken::new("h", 0.0, 0.25, 0.25)],
      "spk_0",
    )],
  ));
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  // Sanity: the `sentences` top-level key + per-sentence `tokens` array +
  // `speaker_id` are all present.
  assert!(raw.contains("\"sentences\""));
  assert!(raw.contains("\"tokens\""));
  assert!(raw.contains("\"speaker_id\": \"spk_0\""));
  // Round-trip.
  let parsed: Transcript = serde_json::from_str(&raw).unwrap();
  assert_eq!(parsed, t);
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_json_segments_with_words_emits_words_array() {
  // python `seg["words"] = s["words"]` pass-through when the segment has
  // word-level alignment. mlxrs writes `{start, end, word}` per word; we
  // assert the JSON contains the words array exactly.
  let base = temp_base("json_words");
  let t = Transcript::Segments(SegmentsPayload::new(
    "hi",
    vec![Segment::new(
      0.0,
      1.0,
      "hi",
      vec![Word::new(0.0, 0.5, "hi", BTreeMap::new())],
      "",
    )],
  ));
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  assert!(raw.contains("\"words\""), "JSON includes words array");
  assert!(
    raw.contains("\"word\": \"hi\""),
    "JSON includes per-word entry"
  );
  // Round-trip.
  let parsed: Transcript = serde_json::from_str(&raw).unwrap();
  assert_eq!(parsed, t);
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_srt_emits_per_word_cues_when_words_present() {
  // python `_get_cues` emits one cue per segment THEN one cue per word —
  // mlxrs preserves that exactly. Verifies the segment-level cue comes
  // BEFORE the word-level cues (and indices are 1-based and contiguous).
  let base = temp_base("srt_with_words");
  let t = Transcript::Segments(SegmentsPayload::new(
    "hi",
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
  save_as_srt(&t, &base).unwrap();
  let srt_path = base.with_file_name(format!(
    "{}.srt",
    base.file_name().unwrap().to_string_lossy()
  ));
  let contents = fs::read_to_string(&srt_path).unwrap();
  let expected = "1\n00:00:00,000 --> 00:00:01,000\nhi there\n\n\
                  2\n00:00:00,000 --> 00:00:00,500\nhi\n\n\
                  3\n00:00:00,500 --> 00:00:01,000\nthere\n\n";
  assert_eq!(contents, expected);
  let _ = fs::remove_file(&srt_path);
}

#[test]
fn save_as_txt_appends_extension_does_not_replace() {
  // Sanity: mlxrs faithful-port uses python's `f"{output_path}.txt"`
  // convention (append, not replace). `out.draft` becomes `out.draft.txt`,
  // NOT `out.txt`.
  let base = temp_base("ext_append.draft");
  let t = Transcript::Segments(SegmentsPayload::new("x", vec![]));
  save_as_txt(&t, &base).unwrap();
  let appended = base.with_file_name(format!(
    "{}.txt",
    base.file_name().unwrap().to_string_lossy()
  ));
  assert!(appended.exists(), "extension was appended, not replaced");
  let _ = fs::remove_file(&appended);
}

// ---------- Finding 1 regression: empty `words` list is python-falsy ----------

#[test]
fn save_as_json_omits_empty_words_field() {
  // python `if "words" in s and s["words"]` (stt/generate.py:213): the
  // `words` key is dropped when the per-segment word list is empty (python
  // truthy check). A `Some(vec![])` MUST NOT produce a `"words": []` entry
  // in the JSON — downstream tooling that `"words" in seg` checks would
  // diverge from the python reference otherwise.
  let base = temp_base("json_empty_words_omitted");
  let t = Transcript::Segments(SegmentsPayload::new(
    "hi",
    vec![Segment::new(
      0.0,
      1.0,
      "hi",
      // The defect-trigger: empty Vec — must NOT produce `"words": []` in
      // JSON; python omits the key entirely (truthy check).
      vec![],
      "",
    )],
  ));
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  // Parse + introspect: the segment object must NOT carry a `"words"` key.
  let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
  let segs = parsed
    .get("segments")
    .and_then(|v| v.as_array())
    .expect("JSON has `segments` array");
  assert_eq!(segs.len(), 1, "fixture has one segment");
  let seg = segs[0].as_object().expect("segment is an object");
  assert!(
    !seg.contains_key("words"),
    "empty `words: Some(vec![])` must omit the `words` key entirely (python truthy semantics); \
     got JSON keys {:?}",
    seg.keys().collect::<Vec<_>>()
  );
  // String-level sanity: the literal `"words"` substring must not appear
  // anywhere in the rendered JSON.
  assert!(
    !raw.contains("\"words\""),
    "rendered JSON must not contain the `\"words\"` key when the list is empty; got: {raw}"
  );
  let _ = fs::remove_file(&json_path);
}

#[test]
fn save_as_json_emits_words_when_non_empty() {
  // Regression-guard the happy path: a non-empty `Some(vec![Word])` MUST
  // still emit the `"words"` array (so the fix only drops the empty case).
  let base = temp_base("json_words_present_when_nonempty");
  let t = Transcript::Segments(SegmentsPayload::new(
    "hi",
    vec![Segment::new(
      0.0,
      1.0,
      "hi",
      vec![Word::new(0.0, 1.0, "hi", BTreeMap::new())],
      "",
    )],
  ));
  save_as_json(&t, &base).unwrap();
  let json_path = base.with_file_name(format!(
    "{}.json",
    base.file_name().unwrap().to_string_lossy()
  ));
  let raw = fs::read_to_string(&json_path).unwrap();
  let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
  let seg = parsed["segments"][0]
    .as_object()
    .expect("segment is an object");
  let words = seg
    .get("words")
    .and_then(|v| v.as_array())
    .expect("non-empty `words` must produce the JSON `words` array");
  assert_eq!(
    words.len(),
    1,
    "non-empty `Some(vec![Word])` produces a one-entry `words` array"
  );
  let _ = fs::remove_file(&json_path);
}

// ---------- Finding 2 regression: `Path::new("-")` → stdout, NOT a file ----------

/// Stdout-passthrough fixture used by all four `save_as_*_writes_to_stdout`
/// regression tests below.
fn dash_path_fixture() -> Transcript {
  Transcript::Segments(SegmentsPayload::new(
    "hello world foo",
    vec![
      Segment::new(0.0, 1.234, "hello", vec![], ""),
      Segment::new(1.234, 2.500, "world", vec![], ""),
      Segment::new(2.500, 4.000, "foo", vec![], ""),
    ],
  ))
}

/// Assert that calling `save_as_*` with `Path::new("-")` did NOT create a
/// `-.{ext}` file on disk anywhere a python user could have ended up
/// (cwd + `std::env::temp_dir()`). The python `contextlib.nullcontext(
/// sys.stdout)` branch never opens a file; mlxrs must mirror that exactly.
fn assert_no_dash_file_on_disk(ext: &str) {
  let cwd_candidate = std::path::PathBuf::from(format!("-.{ext}"));
  assert!(
    !cwd_candidate.exists(),
    "dash-path stdout branch must NOT create `-.{ext}` in cwd (got {})",
    cwd_candidate.display()
  );
  let mut tmp_candidate = std::env::temp_dir();
  tmp_candidate.push(format!("-.{ext}"));
  assert!(
    !tmp_candidate.exists(),
    "dash-path stdout branch must NOT create `-.{ext}` in tmpdir (got {})",
    tmp_candidate.display()
  );
}

#[test]
fn save_as_txt_writes_to_stdout_for_dash_path() {
  // python `output_path != "-"` branch in `save_as_txt`
  // (stt/generate.py:135-141): the `-` literal selects
  // `contextlib.nullcontext(sys.stdout)`, which writes the body to stdout
  // WITHOUT appending the `.txt` extension. mlxrs mirrors that by routing
  // through `std::io::stdout()` in the same branch.
  //
  // Capturing stdout from the test process is awkward + racy in cargo's
  // multi-test runner (and `--test-threads=1` doesn't help with the
  // outer-process pipe), so we assert the *observable disk side-effect*
  // (no `-.txt` file created on disk anywhere) plus the writer-helper
  // unit tests above (`save_as_txt_to_writer_matches_python_body`) which
  // exercise the byte-exact body the stdout branch writes.
  let t = dash_path_fixture();
  save_as_txt(&t, std::path::Path::new("-")).unwrap();
  assert_no_dash_file_on_disk("txt");
}

#[test]
fn save_as_srt_writes_to_stdout_for_dash_path() {
  let t = dash_path_fixture();
  save_as_srt(&t, std::path::Path::new("-")).unwrap();
  assert_no_dash_file_on_disk("srt");
}

#[test]
fn save_as_vtt_writes_to_stdout_for_dash_path() {
  let t = dash_path_fixture();
  save_as_vtt(&t, std::path::Path::new("-")).unwrap();
  assert_no_dash_file_on_disk("vtt");
}

#[test]
fn save_as_json_writes_to_stdout_for_dash_path() {
  let t = dash_path_fixture();
  save_as_json(&t, std::path::Path::new("-")).unwrap();
  assert_no_dash_file_on_disk("json");
}

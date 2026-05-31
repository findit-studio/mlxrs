//! Unit tests for the **model-free** surface of the TTS generate
//! pipeline: the private text-segmentation helpers
//! ([`segment_ranges`] / [`push_if_nonblank`] — unreachable from the
//! integration `tests/audio_tts.rs`, which can only drive them
//! indirectly through [`tts_generate`]), the config / segment / chunk
//! DTO builders + accessors, and the [`AudioFormat`] /
//! [`TextSegmentation`] enum helpers ([`as_str`](AudioFormat::as_str),
//! `Display`, the `derive_more` `IsVariant` `is_*` predicates). The
//! model-driven flows (iteration, fusing, dtype/shape guards) live in
//! `tests/audio_tts.rs` and are not duplicated here.
use super::*;

// ───────────────── segment_ranges: hand-derived byte ranges ─────────────────

/// Resolve `segment_ranges` to the segment *strings* so a test can assert
/// on the sliced content directly (the driver does `&text[start..end]`).
fn segments_of(text: &str, mode: TextSegmentation) -> Vec<&str> {
  segment_ranges(text, mode)
    .into_iter()
    .map(|(s, e)| &text[s..e])
    .collect()
}

/// `Newlines`: a maximal run of non-`\n` bytes is one segment; the
/// `(start, end)` ranges index exactly the line bytes (no newline
/// included), with no per-segment allocation.
#[test]
fn segment_ranges_newlines_exact_byte_ranges() {
  let text = "first\nsecond\nthird";
  let ranges = segment_ranges(text, TextSegmentation::Newlines);
  // "first" = [0,5), "second" = [6,12), "third" = [13,18).
  assert_eq!(ranges, vec![(0, 5), (6, 12), (13, 18)]);
  assert_eq!(
    segments_of(text, TextSegmentation::Newlines),
    ["first", "second", "third"]
  );
}

/// Consecutive newlines collapse (the `\n+` semantics) and leading /
/// trailing / whitespace-only segments are dropped — only the two
/// non-blank runs survive, with byte ranges into the *original* string.
#[test]
fn segment_ranges_newlines_collapses_and_drops_blanks() {
  // leading \n, doubled \n, a whitespace-only line "   ", trailing \n\n.
  let text = "\nalpha\n\n   \nbeta\n\n";
  let ranges = segment_ranges(text, TextSegmentation::Newlines);
  // "alpha" starts at byte 1: [1,6). After "\n\n   \n" the next non-blank
  // run "beta" is at byte 12: [12,16). The "   " run is blank → dropped.
  assert_eq!(ranges, vec![(1, 6), (12, 16)]);
  assert_eq!(
    segments_of(text, TextSegmentation::Newlines),
    ["alpha", "beta"]
  );
}

/// Interior whitespace inside a non-blank line is preserved verbatim —
/// only the *blank-drop* uses `trim`; the kept range is the whole line.
#[test]
fn segment_ranges_newlines_preserves_interior_whitespace() {
  let text = "  hello   world  \nx";
  let segs = segments_of(text, TextSegmentation::Newlines);
  // The line is non-blank (has 'hello'), so it's kept whole — leading and
  // interior and trailing spaces all intact (no trim applied to content).
  assert_eq!(segs, ["  hello   world  ", "x"]);
}

/// A lone `\r` is NOT a segment separator (only `\n` splits) — a CRLF
/// line keeps its trailing `\r` inside the segment content.
#[test]
fn segment_ranges_newlines_does_not_split_on_carriage_return() {
  let text = "a\r\nb";
  let segs = segments_of(text, TextSegmentation::Newlines);
  // Split only on '\n': "a\r" and "b". The '\r' stays with the first line
  // (it is non-whitespace-significant content as far as the splitter cares,
  // though trim would strip it — but trim only gates the blank check, the
  // kept range is the full "a\r").
  assert_eq!(segs, ["a\r", "b"]);
}

/// Multibyte UTF-8: the `(start, end)` ranges are valid char boundaries,
/// so slicing never panics and recovers the exact multibyte segment.
#[test]
fn segment_ranges_newlines_multibyte_utf8_boundaries() {
  // "héllo" (é = 2 bytes) then "wörld" (ö = 2 bytes).
  let text = "héllo\nwörld";
  let ranges = segment_ranges(text, TextSegmentation::Newlines);
  // "héllo" = 6 bytes [0,6); '\n' at 6; "wörld" = 6 bytes [7,13).
  assert_eq!(ranges, vec![(0, 6), (7, 13)]);
  assert_eq!(
    segments_of(text, TextSegmentation::Newlines),
    ["héllo", "wörld"]
  );
}

/// A single line with no newline is exactly one segment spanning the whole
/// string.
#[test]
fn segment_ranges_newlines_single_line() {
  let text = "no newline here";
  assert_eq!(
    segment_ranges(text, TextSegmentation::Newlines),
    vec![(0, text.len())]
  );
}

/// An all-blank input (empty, whitespace, only newlines) yields an empty
/// `Vec` under `Newlines` — the signal [`tts_generate`] turns into a
/// recoverable error.
#[test]
fn segment_ranges_newlines_all_blank_is_empty() {
  assert!(segment_ranges("", TextSegmentation::Newlines).is_empty());
  assert!(segment_ranges("\n\n\n", TextSegmentation::Newlines).is_empty());
  assert!(
    segment_ranges("   \n \t \n  ", TextSegmentation::Newlines).is_empty(),
    "whitespace-only lines all dropped"
  );
}

/// `Whole`: the entire input is one `(0, len)` segment — embedded
/// newlines and surrounding whitespace are part of that single span (no
/// split, no trim of the content).
#[test]
fn segment_ranges_whole_is_single_full_span() {
  let text = "  line one\nline two  ";
  let ranges = segment_ranges(text, TextSegmentation::Whole);
  assert_eq!(ranges, vec![(0, text.len())]);
  // The single segment is the verbatim input, leading/trailing spaces and
  // the embedded newline included.
  assert_eq!(
    segments_of(text, TextSegmentation::Whole),
    ["  line one\nline two  "]
  );
}

/// `Whole` on an all-blank input is still empty (the only place `Whole`
/// consults `trim`: a blank whole-input has nothing to synthesize).
#[test]
fn segment_ranges_whole_all_blank_is_empty() {
  assert!(segment_ranges("", TextSegmentation::Whole).is_empty());
  assert!(segment_ranges("   \n\t ", TextSegmentation::Whole).is_empty());
}

/// `Whole` keeps a single non-blank input even if it has leading/trailing
/// whitespace (the blank check is on `trim`, the kept range is not
/// trimmed).
#[test]
fn segment_ranges_whole_keeps_padded_nonblank() {
  let text = "   hi   ";
  assert_eq!(
    segment_ranges(text, TextSegmentation::Whole),
    vec![(0, text.len())]
  );
}

// ───────────────── push_if_nonblank: the blank-drop predicate ─────────────────

/// `push_if_nonblank` appends the range iff `text[start..end]` is not all
/// whitespace.
#[test]
fn push_if_nonblank_keeps_nonblank_drops_blank() {
  let text = "ab   cd";
  let mut out = Vec::new();
  // [0,2) = "ab" → kept.
  push_if_nonblank(&mut out, text, 0, 2);
  // [2,5) = "   " → blank, dropped.
  push_if_nonblank(&mut out, text, 2, 5);
  // [5,7) = "cd" → kept.
  push_if_nonblank(&mut out, text, 5, 7);
  assert_eq!(out, vec![(0, 2), (5, 7)]);
}

/// An empty range `[i, i)` is blank (trims to "") and is dropped.
#[test]
fn push_if_nonblank_drops_empty_range() {
  let text = "xyz";
  let mut out = Vec::new();
  push_if_nonblank(&mut out, text, 1, 1);
  assert!(out.is_empty());
}

// ───────────────── AudioFormat: as_str / Display / IsVariant ─────────────────

/// `AudioFormat::as_str` is the lowercase mlx-audio `audio_format` string.
#[test]
fn audio_format_as_str_and_display() {
  assert_eq!(AudioFormat::Wav.as_str(), "wav");
  assert_eq!(AudioFormat::Flac.as_str(), "flac");
  // `#[display("{}", self.as_str())]` ⇒ Display == as_str.
  assert_eq!(AudioFormat::Wav.to_string(), "wav");
  assert_eq!(format!("{}", AudioFormat::Flac), "flac");
}

/// `derive_more::IsVariant` generates `is_wav` / `is_flac` predicates.
#[test]
fn audio_format_is_variant_predicates() {
  assert!(AudioFormat::Wav.is_wav());
  assert!(!AudioFormat::Wav.is_flac());
  assert!(AudioFormat::Flac.is_flac());
  assert!(!AudioFormat::Flac.is_wav());
}

/// `AudioFormat::default()` is `Wav` (mlx-audio's `audio_format="wav"`).
#[test]
fn audio_format_default_is_wav() {
  assert_eq!(AudioFormat::default(), AudioFormat::Wav);
  assert!(AudioFormat::default().is_wav());
}

// ────────────── TextSegmentation: as_str / Display / IsVariant ──────────────

/// `TextSegmentation::as_str` is the lowercase mode name; Display mirrors
/// it.
#[test]
fn text_segmentation_as_str_and_display() {
  assert_eq!(TextSegmentation::Newlines.as_str(), "newlines");
  assert_eq!(TextSegmentation::Whole.as_str(), "whole");
  assert_eq!(TextSegmentation::Newlines.to_string(), "newlines");
  assert_eq!(format!("{}", TextSegmentation::Whole), "whole");
}

/// `IsVariant` predicates for the segmentation mode.
#[test]
fn text_segmentation_is_variant_predicates() {
  assert!(TextSegmentation::Newlines.is_newlines());
  assert!(!TextSegmentation::Newlines.is_whole());
  assert!(TextSegmentation::Whole.is_whole());
  assert!(!TextSegmentation::Whole.is_newlines());
}

/// `TextSegmentation::default()` is `Newlines` (the mlx-audio kokoro
/// `split_pattern=r"\n+"` default).
#[test]
fn text_segmentation_default_is_newlines() {
  assert_eq!(TextSegmentation::default(), TextSegmentation::Newlines);
  assert!(TextSegmentation::default().is_newlines());
}

// ───────────────── TtsGenConfig: defaults / builders / accessors ─────────────────

/// `TtsGenConfig::new()` equals `TtsGenConfig::default()` and carries the
/// documented mlx-audio `generate_audio` defaults across every field.
#[test]
fn tts_gen_config_new_equals_default_and_carries_defaults() {
  let c = TtsGenConfig::new();
  assert_eq!(c, TtsGenConfig::default());
  assert_eq!(c.voice(), DEFAULT_VOICE);
  assert_eq!(c.language(), DEFAULT_LANGUAGE);
  assert!((c.speed() - 1.0).abs() < 1e-6);
  assert!((c.temperature() - DEFAULT_TEMPERATURE).abs() < 1e-6);
  assert!((c.top_p() - 0.0).abs() < 1e-6);
  assert_eq!(c.top_k(), 0);
  assert_eq!(c.repetition_penalty(), None);
  assert_eq!(c.max_tokens(), DEFAULT_MAX_TOKENS);
  assert_eq!(c.segmentation(), TextSegmentation::Newlines);
  assert_eq!(c.audio_format(), AudioFormat::Wav);
  assert!((c.streaming_interval() - DEFAULT_STREAMING_INTERVAL).abs() < 1e-6);
}

/// Every `with_*` builder sets exactly its field and the matching accessor
/// reads it back — covering the sampling knobs (`top_p`, `top_k`,
/// `repetition_penalty`, `max_tokens`, `audio_format`,
/// `streaming_interval`) the integration tests do not plumb.
#[test]
fn tts_gen_config_builders_round_trip_all_fields() {
  let c = TtsGenConfig::new()
    .with_voice("bf_emma")
    .with_language("en-gb")
    .with_speed(1.25)
    .with_temperature(0.4)
    .with_top_p(0.9)
    .with_top_k(40)
    .with_repetition_penalty(Some(1.1))
    .with_max_tokens(256)
    .with_segmentation(TextSegmentation::Whole)
    .with_audio_format(AudioFormat::Flac)
    .with_streaming_interval(3.5);
  assert_eq!(c.voice(), "bf_emma");
  assert_eq!(c.language(), "en-gb");
  assert!((c.speed() - 1.25).abs() < 1e-6);
  assert!((c.temperature() - 0.4).abs() < 1e-6);
  assert!((c.top_p() - 0.9).abs() < 1e-6);
  assert_eq!(c.top_k(), 40);
  assert_eq!(c.repetition_penalty(), Some(1.1));
  assert_eq!(c.max_tokens(), 256);
  assert!(c.segmentation().is_whole());
  assert!(c.audio_format().is_flac());
  assert!((c.streaming_interval() - 3.5).abs() < 1e-6);
}

/// A `with_*` builder mutates only its own field (the others keep their
/// defaults) — guards against a copy/paste setter writing the wrong field.
#[test]
fn tts_gen_config_builder_is_field_isolated() {
  let base = TtsGenConfig::default();
  let only_topk = TtsGenConfig::default().with_top_k(7);
  assert_eq!(only_topk.top_k(), 7);
  // Nothing else moved.
  assert_eq!(only_topk.voice(), base.voice());
  assert_eq!(only_topk.language(), base.language());
  assert!((only_topk.speed() - base.speed()).abs() < 1e-6);
  assert!((only_topk.temperature() - base.temperature()).abs() < 1e-6);
  assert!((only_topk.top_p() - base.top_p()).abs() < 1e-6);
  assert_eq!(only_topk.repetition_penalty(), base.repetition_penalty());
  assert_eq!(only_topk.max_tokens(), base.max_tokens());
  assert_eq!(only_topk.segmentation(), base.segmentation());
  assert_eq!(only_topk.audio_format(), base.audio_format());
  assert!((only_topk.streaming_interval() - base.streaming_interval()).abs() < 1e-6);
}

/// `with_repetition_penalty(None)` clears a previously-set penalty (the
/// field is `Option<f32>`, so `None` is a meaningful reset).
#[test]
fn tts_gen_config_repetition_penalty_can_be_cleared() {
  let c = TtsGenConfig::new()
    .with_repetition_penalty(Some(1.3))
    .with_repetition_penalty(None);
  assert_eq!(c.repetition_penalty(), None);
}

/// `TtsGenConfig` is `Clone` + `PartialEq`: a clone equals its source, and
/// changing one field makes them unequal (the type owns no `Array`, so it
/// is cheap to clone and value-comparable — the `default_config` contract).
#[test]
fn tts_gen_config_clone_and_partial_eq() {
  let a = TtsGenConfig::new().with_voice("v").with_top_k(3);
  let b = a.clone();
  assert_eq!(a, b);
  let c = a.clone().with_top_k(4);
  assert_ne!(a, c, "differing top_k ⇒ unequal");
}

// ───────────────── TtsReference: accessors + default ─────────────────

/// `TtsReference::new` stores both optional fields and the accessors read
/// them back.
#[test]
fn tts_reference_new_accessors() {
  let wav = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3], &[3]).unwrap();
  let r = TtsReference::new(Some(&wav), Some("caption"));
  assert!(r.ref_audio().is_some());
  assert_eq!(r.ref_text(), Some("caption"));
  // The borrowed audio is the same [3] array.
  assert_eq!(r.ref_audio().unwrap().shape(), vec![3]);
}

/// `TtsReference::default()` is both-`None` (a non-cloning reference) — the
/// value [`tts_generate`] forwards on the plain (no-reference) path.
#[test]
fn tts_reference_default_is_both_none() {
  let r = TtsReference::default();
  assert!(r.ref_audio().is_none());
  assert!(r.ref_text().is_none());
}

/// The two fields are independently optional: audio-only (`Some`/`None`)
/// and text-only (`None`/`Some`) references are both representable.
#[test]
fn tts_reference_fields_are_independent() {
  let wav = Array::from_slice::<f32>(&[0.0_f32], &[1]).unwrap();
  let audio_only = TtsReference::new(Some(&wav), None);
  assert!(audio_only.ref_audio().is_some() && audio_only.ref_text().is_none());
  let text_only = TtsReference::new(None, Some("t"));
  assert!(text_only.ref_audio().is_none() && text_only.ref_text() == Some("t"));
}

// ───────────────── TtsSegment: direct construction + accessors ─────────────────

/// `TtsSegment::new` stores all 13 fields and every accessor reads its own
/// back — including the sampling knobs (`top_p`, `top_k`,
/// `repetition_penalty`, `max_tokens`, `streaming_interval`) the
/// integration mock does not record.
#[test]
fn tts_segment_new_all_accessors() {
  let wav = Array::from_slice::<f32>(&[0.5_f32, -0.5], &[2]).unwrap();
  let seg = TtsSegment::new(
    "the text",
    "af_heart",
    "en",
    1.5,
    0.6,
    0.85,
    50,
    Some(1.2),
    900,
    2.5,
    4,
    Some(&wav),
    Some("ref transcript"),
  );
  assert_eq!(seg.text(), "the text");
  assert_eq!(seg.voice(), "af_heart");
  assert_eq!(seg.language(), "en");
  assert!((seg.speed() - 1.5).abs() < 1e-6);
  assert!((seg.temperature() - 0.6).abs() < 1e-6);
  assert!((seg.top_p() - 0.85).abs() < 1e-6);
  assert_eq!(seg.top_k(), 50);
  assert_eq!(seg.repetition_penalty(), Some(1.2));
  assert_eq!(seg.max_tokens(), 900);
  assert!((seg.streaming_interval() - 2.5).abs() < 1e-6);
  assert_eq!(seg.segment_idx(), 4);
  assert!(seg.ref_audio().is_some());
  assert_eq!(seg.ref_text(), Some("ref transcript"));
}

/// A `TtsSegment` with no reference (`None`/`None`) — the non-cloning
/// shape the driver builds on the plain `tts_generate` path.
#[test]
fn tts_segment_without_reference() {
  let seg = TtsSegment::new(
    "x", "v", "en", 1.0, 0.7, 0.0, 0, None, 1200, 2.0, 0, None, None,
  );
  assert!(seg.ref_audio().is_none());
  assert!(seg.ref_text().is_none());
  assert_eq!(seg.repetition_penalty(), None);
  assert_eq!(seg.segment_idx(), 0);
}

// ───────────────── AudioChunk: construction / accessors / math ─────────────────

/// `AudioChunk::new` stores its envelope and every `&self` accessor reads
/// it back without materializing the audio (`audio_ref` is a no-eval
/// borrow; `len_samples` is a shape read).
#[test]
fn audio_chunk_new_accessors_no_eval() {
  let audio = Array::from_slice::<f32>(&[0.0_f32, 0.1, 0.2, 0.3], &[4]).unwrap();
  let chunk = AudioChunk::new(audio, 16_000, 2, true, false);
  assert_eq!(chunk.sample_rate(), 16_000);
  assert_eq!(chunk.segment_idx(), 2);
  assert!(chunk.is_streaming_chunk());
  assert!(!chunk.is_final_chunk());
  assert_eq!(chunk.len_samples(), 4);
  assert!(!chunk.is_empty());
  assert_eq!(chunk.audio_ref().shape(), vec![4], "no-eval shape read");
}

/// `duration_seconds` is `len_samples / sample_rate` in `f64`.
#[test]
fn audio_chunk_duration_seconds_is_samples_over_rate() {
  let audio = Array::from_slice::<f32>(&[0.0_f32; 24_000], &[24_000]).unwrap();
  let chunk = AudioChunk::new(audio, 24_000, 0, false, true);
  assert!((chunk.duration_seconds() - 1.0).abs() < 1e-12);
}

/// `duration_seconds` guards a zero sample rate: it returns `0.0`, not a
/// NaN / inf division.
#[test]
fn audio_chunk_duration_seconds_zero_rate_is_zero() {
  let audio = Array::from_slice::<f32>(&[0.0_f32, 0.1], &[2]).unwrap();
  let chunk = AudioChunk::new(audio, 0, 0, false, true);
  assert_eq!(chunk.duration_seconds(), 0.0);
  assert!(
    chunk.duration_seconds().is_finite(),
    "no NaN/inf for rate 0"
  );
}

/// A zero-length `[0]` waveform is a valid empty chunk: `is_empty()`,
/// `len_samples() == 0`, `duration_seconds() == 0.0`.
#[test]
fn audio_chunk_empty_waveform() {
  let audio = Array::from_slice::<f32>(&[], &[0]).unwrap();
  let chunk = AudioChunk::new(audio, 24_000, 0, false, true);
  assert!(chunk.is_empty());
  assert_eq!(chunk.len_samples(), 0);
  assert_eq!(chunk.duration_seconds(), 0.0);
}

/// `into_audio` moves the inner tensor out without an eval (the shape is
/// preserved); `samples` is the explicit `&mut` materialization step.
#[test]
fn audio_chunk_into_audio_and_samples() {
  let audio = Array::from_slice::<f32>(&[0.0_f32, 0.25, 0.5], &[3]).unwrap();
  let mut chunk = AudioChunk::new(audio, 24_000, 0, false, true);
  let pcm = chunk.samples().unwrap();
  assert_eq!(pcm, vec![0.0, 0.25, 0.5]);
  // `into_audio` hands back the [3] tensor.
  let moved = AudioChunk::new(
    Array::from_slice::<f32>(&[1.0_f32, 2.0], &[2]).unwrap(),
    24_000,
    0,
    false,
    true,
  )
  .into_audio();
  assert_eq!(moved.shape(), vec![2]);
}

// ───────────────── module constants ─────────────────

/// The documented mlx-audio default constants have the expected values.
#[test]
fn default_constants_match_mlx_audio() {
  assert_eq!(DEFAULT_VOICE, "af_heart");
  assert_eq!(DEFAULT_LANGUAGE, "en");
  assert!((DEFAULT_TEMPERATURE - 0.7).abs() < 1e-6);
  assert_eq!(DEFAULT_MAX_TOKENS, 1200);
  assert!((DEFAULT_STREAMING_INTERVAL - 2.0).abs() < 1e-6);
  assert_eq!(MAX_TEXT_BYTES, 1024 * 1024);
}

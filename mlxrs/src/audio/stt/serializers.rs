//! Transcript serializers — TXT / SRT / WebVTT / JSON writers ported 1:1 from
//! mlx-audio's [`stt/generate.py`][stt-gen] (`save_as_txt` / `save_as_srt` /
//! `save_as_vtt` / `save_as_json` plus the `format_timestamp` /
//! `format_vtt_timestamp` helpers, lines 104-225). The serializer surface is
//! the model-agnostic counterpart of the [`super::generate::stt_generate`]
//! decode loop: a concrete per-model decoder (whisper / parakeet / canary /
//! …) terminates by building a [`Transcript`] from its sampled tokens +
//! per-segment timing info, and the user then writes that [`Transcript`] to
//! disk in whichever format their downstream tool consumes.
//!
//! ## Reference shape
//!
//! mlx-audio's python serializers accept a duck-typed `segments` argument
//! that is one of two distinct shapes (`stt/generate.py:117-132` `_get_cues`):
//!
//! - Whisper-style: an object with `.text: str` + `.segments: list[dict]`
//!   where each dict carries `{start, end, text, [words], [speaker_id]}`.
//! - Parakeet-style: an object with `.text: str` + `.sentences:
//!   list[AlignedSentence]` ([`stt/models/parakeet/alignment.py:16`][palign])
//!   where each sentence carries `{text, start, end, duration,
//!   tokens: list[AlignedToken]}`.
//!
//! mlxrs ports both shapes into a single [`Transcript`] sum type with two
//! variants ([`Transcript::Segments`] / [`Transcript::Sentences`]); the
//! serializers branch on the variant exactly the way `_get_cues` and
//! `save_as_json` branch on `hasattr(segments, "sentences")`. The python
//! duck-type is preserved as a faithful sum type rather than a single
//! flattened struct because the JSON output schema is genuinely different
//! between the two shapes (Whisper: `{"text", "segments": [...]}`; Parakeet:
//! `{"text", "sentences": [...]}`) and merging them would silently change the
//! JSON shape consumers depend on.
//!
//! ## File-extension convention
//!
//! mlx-audio's writers take a path **without the extension** and append the
//! format-specific extension themselves (`open(f"{output_path}.txt", ...)`,
//! `stt/generate.py:137`). mlxrs preserves this exactly — the `path` argument
//! to each `save_as_*` is the **base path**, and the serializer appends the
//! format-specific extension (`.txt`, `.srt`, `.vtt`, `.json`) before
//! opening. This is the python signature, NOT a Rust-idiomatic
//! "caller-supplies-full-path" convention; the choice is faithful-port-over-
//! Rust-ergonomic because downstream tooling (CLI wrappers, batch scripts)
//! already passes the base path without extension.
//!
//! ## `wired_limit` / generation-stats parity audit on `stt/generate.rs`
//!
//! mlx-audio's `generate_transcription` (`stt/generate.py:272-413`) wraps the
//! per-model decoder call in a `wired_limit(model, [generation_stream])`
//! context manager (`stt/generate.py:232-269`) that calls
//! `mx.set_wired_limit(max_recommended_working_set_size)` for the duration of
//! the decode + restores the old limit on exit. mlxrs's
//! [`super::generate::stt_generate`] (issue [#176][a13]) does **not**
//! integrate `wired_limit` for two reasons:
//!
//! 1. **No mlxrs-safe wrapper for `mlx_set_wired_limit` exists yet.** The
//!    `mlxrs-sys` FFI exposes
//!    [`mlxrs_sys::mlx_set_wired_limit`][sys-swl] but
//!    [`crate::memory`] hasn't surfaced it as a safe `set_wired_limit` fn,
//!    and `mlx_device_info_get` (required to query
//!    `max_recommended_working_set_size`) also has no safe wrapper. Both are
//!    LM-side `wired_limit` integration prerequisites (`mlx-lm/generate.py`
//!    has the identical context manager) and live with the LM L6 follow-up,
//!    NOT this STT-serializers port. The STT loop will integrate
//!    `wired_limit` once the LM loop does — the same shared support surface,
//!    threaded behind whatever safe API the LM L6 lands.
//! 2. **mlxrs's STT loop is iterator-shaped, not a single-call wrapper.**
//!    [`super::generate::stt_generate`] returns an `Iterator<Item =
//!    Result<GenStep>>` (mirroring the LM
//!    [`crate::lm::generate::generate_step`] shape), so the analogue of
//!    python's `with wired_limit(model, [stream]): ... decode ...` is the
//!    caller wrapping their `.collect()` (or per-step `.next()` loop) in a
//!    `WiredLimitGuard` — the same caller-driven pattern
//!    `crate::lm::generate::stream_generate` will use once the LM L6 FFI
//!    surface is in place. A single per-call `wired_limit` integration
//!    inside `stt_generate` would lock the limit only while the constructor
//!    runs, NOT while the iterator is being driven — which would be worse
//!    than no integration (a misleading "we set the wired limit" claim that
//!    silently doesn't cover the decode work).
//!
//! Similarly, mlx-audio's `generate_transcription` reports per-run
//! [`GenerationStats`][gs]-shaped timing — `total_time`, `prompt_tokens`,
//! `generation_tokens`, `prompt_tps`, `generation_tps` — packed into the
//! per-model `STTOutput` dataclass. mlxrs's
//! [`super::generate::stt_generate`] returns the lower-level per-step
//! [`crate::lm::generate::GenStep`] iterator, mirroring
//! [`crate::lm::generate::generate_step`] (NOT the higher-level
//! [`crate::lm::generate::stream_generate`] that aggregates into
//! [`crate::lm::generate::GenerationResponse`] with the `prompt_tps` /
//! `generation_tps` fields). A `GenerationStats`-shaped aggregator wrapper
//! around `stt_generate` is the natural follow-up to ship alongside the
//! LM-L6 `wired_limit` integration (the same wrapping idiom both loops will
//! share); it doesn't belong in the per-step iterator itself.
//!
//! [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
//! [palign]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/parakeet/alignment.py
//! [a13]: https://github.com/uqio/mlxrs/issues/176
//! [sys-swl]: https://docs.rs/mlxrs-sys/latest/mlxrs_sys/fn.mlx_set_wired_limit.html
//! [gs]: crate::lm::generate::GenerationStats

use std::{
  collections::BTreeMap,
  fs::File,
  io::{BufWriter, Write},
  path::Path,
};

use derive_more::{IsVariant, TryUnwrap, Unwrap};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// One word-level cue inside a Whisper-style [`Segment`] — mirrors the
/// duck-typed `s["words"]` list entry in
/// [`mlx_audio.stt.generate.save_as_json`][stt-gen-json]
/// (`{"start": ..., "end": ..., "word": ...}`). Optional extra fields
/// (e.g. `probability`) the python emits via dict pass-through are kept in
/// [`Word::extra`] so a faithful round-trip never drops fields the per-model
/// decoder attached.
///
/// [stt-gen-json]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
  /// Word start time in seconds (mlx-audio python `w["start"]`).
  start: f64,
  /// Word end time in seconds (mlx-audio python `w["end"]`).
  end: f64,
  /// Word text (mlx-audio python `w["word"]`).
  word: String,
  /// Extra per-word fields the per-model decoder may attach (e.g.
  /// `probability`). Preserved verbatim through the JSON round-trip;
  /// faithful to `save_as_json`'s `seg["words"] = s["words"]` pass-through.
  ///
  /// `BTreeMap` (not `HashMap`) for deterministic JSON-output key order —
  /// matches `serde_json` `preserve_order` semantics for the rest of the
  /// struct's fields.
  #[serde(flatten)]
  extra: BTreeMap<String, serde_json::Value>,
}

impl Word {
  /// Construct a [`Word`].
  pub fn new(
    start: f64,
    end: f64,
    word: impl Into<String>,
    extra: BTreeMap<String, serde_json::Value>,
  ) -> Self {
    Self {
      start,
      end,
      word: word.into(),
      extra,
    }
  }

  /// Word start time in seconds.
  #[inline(always)]
  pub fn start(&self) -> f64 {
    self.start
  }

  /// Word end time in seconds.
  #[inline(always)]
  pub fn end(&self) -> f64 {
    self.end
  }

  /// Word text.
  #[inline(always)]
  pub fn word(&self) -> &str {
    &self.word
  }

  /// Extra per-word JSON fields.
  #[inline(always)]
  pub fn extra(&self) -> &BTreeMap<String, serde_json::Value> {
    &self.extra
  }
}

/// One Whisper-style transcript segment — mirrors the duck-typed
/// `segments.segments` list entries in [`mlx_audio.stt.generate`][stt-gen]
/// (`{"start": ..., "end": ..., "text": ..., ["words": ...], ["speaker_id":
/// ...]}`).
///
/// Per §1 EMPTY=ABSENT: `words` is a `Vec<Word>` (empty = no word-level
/// alignment) and `speaker_id` is a `String` (empty = no diarization). The
/// serde wire form skips both fields when they are empty, so JSON consumers
/// see the field absent rather than `null` or `[]`/`""` — matching the
/// python null/missing convention these consumers depend on.
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
  /// Segment start time in seconds (mlx-audio python `s["start"]`).
  start: f64,
  /// Segment end time in seconds (mlx-audio python `s["end"]`).
  end: f64,
  /// Segment text (mlx-audio python `s["text"]`).
  text: String,
  /// Word-level cues (mlx-audio python `s["words"]`). Empty when the
  /// per-model decoder did not emit word-level alignment (EMPTY=ABSENT:
  /// field is omitted from JSON when empty, matching python null/missing).
  #[serde(skip_serializing_if = "Vec::is_empty", default)]
  words: Vec<Word>,
  /// Speaker-id label (mlx-audio python `s["speaker_id"]`). Empty string
  /// when speaker diarization wasn't run (EMPTY=ABSENT: field is omitted
  /// from JSON when empty, matching python null/missing).
  #[serde(skip_serializing_if = "String::is_empty", default)]
  speaker_id: String,
}

impl Segment {
  /// Construct a [`Segment`].
  #[inline(always)]
  pub fn new(
    start: f64,
    end: f64,
    text: impl Into<String>,
    words: Vec<Word>,
    speaker_id: impl Into<String>,
  ) -> Self {
    Self {
      start,
      end,
      text: text.into(),
      words,
      speaker_id: speaker_id.into(),
    }
  }

  /// Segment start time in seconds.
  #[inline(always)]
  pub fn start(&self) -> f64 {
    self.start
  }

  /// Segment end time in seconds.
  #[inline(always)]
  pub fn end(&self) -> f64 {
    self.end
  }

  /// Segment text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Word-level cues as a slice. Empty slice = no word-level alignment
  /// (EMPTY=ABSENT — absent from JSON when empty).
  #[inline(always)]
  pub fn words_slice(&self) -> &[Word] {
    &self.words
  }

  /// Speaker-id label. Empty string = no diarization
  /// (EMPTY=ABSENT — absent from JSON when empty).
  #[inline(always)]
  pub fn speaker_id(&self) -> &str {
    &self.speaker_id
  }
}

/// One Parakeet-style aligned token — mirrors `AlignedToken` from
/// [`mlx_audio.stt.models.parakeet.alignment`][palign] (the per-character /
/// per-subword timing record `AlignedSentence.tokens` carries).
///
/// [palign]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/parakeet/alignment.py
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceToken {
  /// Token text (mlx-audio python `t.text`).
  text: String,
  /// Token start time in seconds (mlx-audio python `t.start`).
  start: f64,
  /// Token end time in seconds (mlx-audio python `t.end`; `start + duration`
  /// per `AlignedToken.__post_init__`).
  end: f64,
  /// Token duration in seconds (mlx-audio python `t.duration`).
  duration: f64,
}

impl SentenceToken {
  /// Construct a [`SentenceToken`].
  pub fn new(text: impl Into<String>, start: f64, end: f64, duration: f64) -> Self {
    Self {
      text: text.into(),
      start,
      end,
      duration,
    }
  }

  /// Token text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Token start time in seconds.
  #[inline(always)]
  pub fn start(&self) -> f64 {
    self.start
  }

  /// Token end time in seconds.
  #[inline(always)]
  pub fn end(&self) -> f64 {
    self.end
  }

  /// Token duration in seconds.
  #[inline(always)]
  pub fn duration(&self) -> f64 {
    self.duration
  }
}

/// One Parakeet-style aligned sentence — mirrors `AlignedSentence` from
/// [`mlx_audio.stt.models.parakeet.alignment`][palign]. Carries per-token
/// alignment plus optional speaker-id (the only `hasattr(s, "speaker_id")`
/// branch in `save_as_json`).
///
/// Per §1 EMPTY=ABSENT: `speaker_id` is a `String` (empty = no diarization).
/// The serde wire form skips the field when empty, so JSON consumers see the
/// field absent rather than `null` or `""` — matching the python
/// `hasattr(s, "speaker_id")` null/missing convention.
///
/// [palign]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/parakeet/alignment.py
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
  /// Sentence text (mlx-audio python `s.text`).
  text: String,
  /// Sentence start time in seconds (mlx-audio python `s.start`).
  start: f64,
  /// Sentence end time in seconds (mlx-audio python `s.end`).
  end: f64,
  /// Sentence duration in seconds (mlx-audio python `s.duration`).
  duration: f64,
  /// Per-token alignment (mlx-audio python `s.tokens`).
  tokens: Vec<SentenceToken>,
  /// Speaker-id label (mlx-audio python `s.speaker_id`; surfaced only when
  /// `hasattr(s, "speaker_id")`). Empty string when diarization wasn't run
  /// (EMPTY=ABSENT — field is omitted from JSON when empty, matching python
  /// null/missing).
  #[serde(skip_serializing_if = "String::is_empty", default)]
  speaker_id: String,
}

impl Sentence {
  /// Construct a [`Sentence`].
  #[inline(always)]
  pub fn new(
    text: impl Into<String>,
    start: f64,
    end: f64,
    duration: f64,
    tokens: Vec<SentenceToken>,
    speaker_id: impl Into<String>,
  ) -> Self {
    Self {
      text: text.into(),
      start,
      end,
      duration,
      tokens,
      speaker_id: speaker_id.into(),
    }
  }

  /// Sentence text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Sentence start time in seconds.
  #[inline(always)]
  pub fn start(&self) -> f64 {
    self.start
  }

  /// Sentence end time in seconds.
  #[inline(always)]
  pub fn end(&self) -> f64 {
    self.end
  }

  /// Sentence duration in seconds.
  #[inline(always)]
  pub fn duration(&self) -> f64 {
    self.duration
  }

  /// Per-token alignment.
  #[inline(always)]
  pub fn tokens(&self) -> &[SentenceToken] {
    &self.tokens
  }

  /// Speaker-id label. Empty string = no diarization
  /// (EMPTY=ABSENT — absent from JSON when empty).
  #[inline(always)]
  pub fn speaker_id(&self) -> &str {
    &self.speaker_id
  }
}

/// Payload for [`Transcript::Segments`] — carries the assembled text and the
/// per-segment timing list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentsPayload {
  /// Assembled transcription text (mlx-audio python `segments.text`).
  text: String,
  /// Per-segment timing (mlx-audio python `segments.segments`).
  segments: Vec<Segment>,
}

impl SegmentsPayload {
  /// Construct a [`SegmentsPayload`].
  pub fn new(text: impl Into<String>, segments: Vec<Segment>) -> Self {
    Self {
      text: text.into(),
      segments,
    }
  }

  /// Assembled transcription text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Per-segment timing.
  #[inline(always)]
  pub fn segments(&self) -> &[Segment] {
    &self.segments
  }
}

/// Payload for [`Transcript::Sentences`] — carries the assembled text and the
/// per-sentence timing list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentencesPayload {
  /// Assembled transcription text (mlx-audio python `segments.text`).
  text: String,
  /// Per-sentence timing + tokens (mlx-audio python `segments.sentences`).
  sentences: Vec<Sentence>,
}

impl SentencesPayload {
  /// Construct a [`SentencesPayload`].
  pub fn new(text: impl Into<String>, sentences: Vec<Sentence>) -> Self {
    Self {
      text: text.into(),
      sentences,
    }
  }

  /// Assembled transcription text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Per-sentence timing + tokens.
  #[inline(always)]
  pub fn sentences(&self) -> &[Sentence] {
    &self.sentences
  }
}

/// A transcript — the input to the four `save_as_*` serializers, mirroring
/// the duck-typed `segments` argument
/// [`mlx_audio.stt.generate.save_as_txt`][stt-gen] /
/// `save_as_srt` / `save_as_vtt` / `save_as_json` accept.
///
/// The two variants mirror the two shapes the python `_get_cues` +
/// `save_as_json` `hasattr(segments, "sentences")` branch on:
///
/// - [`Transcript::Segments`] — Whisper-style (object with `.text` +
///   `.segments: list[dict]`, optionally per-segment `words` / `speaker_id`).
/// - [`Transcript::Sentences`] — Parakeet-style (object with `.text` +
///   `.sentences: list[AlignedSentence]`, each carrying per-token alignment).
///
/// `text` (the assembled transcription string) is carried in BOTH variants
/// because every python serializer accesses `segments.text` regardless of
/// shape (`save_as_txt:141`, `save_as_json:176/202`).
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, IsVariant, Unwrap, TryUnwrap)]
#[unwrap(ref, ref_mut)]
#[serde(untagged)]
pub enum Transcript {
  /// Whisper-style transcript: `text` + `segments` (per-segment timing,
  /// optional word-level alignment, optional speaker-id).
  Segments(SegmentsPayload),
  /// Parakeet-style transcript: `text` + `sentences` (per-sentence timing +
  /// per-token alignment, optional speaker-id).
  Sentences(SentencesPayload),
}

impl Transcript {
  /// Returns the assembled transcription text regardless of variant
  /// (mlx-audio python `segments.text` access).
  pub fn text(&self) -> &str {
    match self {
      Transcript::Segments(p) => p.text(),
      Transcript::Sentences(p) => p.text(),
    }
  }
}

/// One time-keyed cue extracted by [`get_cues`] — the unified `{start, end,
/// text}` triple `save_as_srt` / `save_as_vtt` iterate over (mlx-audio
/// python `_get_cues` return shape, [`stt/generate.py:117-132`][stt-gen]).
///
/// `&str` borrow back into the source [`Transcript`] (no allocation per
/// cue) — mirrors python's dict-of-references-into-segments behavior and
/// avoids a per-cue `String` clone when the caller is just writing the cues
/// to disk.
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
#[derive(Debug, Clone, Copy)]
pub struct Cue<'a> {
  /// Cue start time in seconds.
  start: f64,
  /// Cue end time in seconds.
  end: f64,
  /// Cue text (borrowed back into the [`Transcript`]).
  text: &'a str,
}

impl<'a> Cue<'a> {
  /// Construct a [`Cue`].
  #[inline(always)]
  pub const fn new(start: f64, end: f64, text: &'a str) -> Self {
    Self { start, end, text }
  }

  /// Cue start time in seconds.
  #[inline(always)]
  pub fn start(&self) -> f64 {
    self.start
  }

  /// Cue end time in seconds.
  #[inline(always)]
  pub fn end(&self) -> f64 {
    self.end
  }

  /// Cue text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    self.text
  }
}

/// Extract unified cues from a [`Transcript`] — the model-agnostic
/// per-segment / per-sentence iterator the SRT / VTT writers consume.
///
/// Mirrors `_get_cues` ([`stt/generate.py:117-132`][stt-gen]):
///
/// - [`Transcript::Sentences`] (Parakeet): one cue per sentence (`{start,
///   end, text}` per `s` in `segments.sentences`). Per-token cues are NOT
///   emitted here — the python only emits segment-level cues for the
///   sentence variant.
/// - [`Transcript::Segments`] (Whisper): one cue per segment, plus one cue
///   per word inside that segment when the segment carries `words`. This
///   matches python `_get_cues`'s loop:
///   `cues.append({segment-level}); for w in s["words"]: cues.append(...)`.
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn get_cues(t: &Transcript) -> Vec<Cue<'_>> {
  match t {
    Transcript::Sentences(p) => p
      .sentences()
      .iter()
      .map(|s| Cue::new(s.start(), s.end(), s.text()))
      .collect(),
    Transcript::Segments(p) => {
      let mut cues = Vec::with_capacity(p.segments().len());
      for s in p.segments() {
        cues.push(Cue::new(s.start(), s.end(), s.text()));
        for w in s.words_slice() {
          cues.push(Cue::new(w.start(), w.end(), w.word()));
        }
      }
      cues
    }
  }
}

/// Convert seconds to the SRT `HH:MM:SS,mmm` timestamp format.
///
/// 1:1 port of [`mlx_audio.stt.generate.format_timestamp`][stt-gen]
/// (`stt/generate.py:104-109`):
///
/// ```python
/// hours = int(seconds // 3600)
/// minutes = int((seconds % 3600) // 60)
/// seconds = seconds % 60
/// return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
/// ```
///
/// Uses `.floor()` (not truncate-toward-zero) to mirror python `//` floor
/// division, and `((s / 60).floor()) % 60` arithmetic for the
/// `(seconds % 3600) // 60` minute extraction without an intermediate
/// `% 3600` (the two are arithmetically equivalent; the direct mod-then-div
/// path avoids the `(s % 3600)` rounding-loss edge case at the
/// `3600 * 2^k` cusps).
///
/// The `,` separator (vs WebVTT's `.`) is the SRT spec; see
/// [`format_vtt_timestamp`] for the dot variant.
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn format_timestamp(seconds: f64) -> String {
  // Python `int(seconds // 3600)`: floor-div toward -inf, then cast to int.
  // For non-finite or negative inputs the python code would happily emit
  // a malformed timestamp (e.g. "0-1:-1:-0,001"); mlxrs preserves the
  // mathematical behavior faithfully — the caller is responsible for
  // passing non-negative finite seconds (per-model decoders never emit
  // negative timestamps). A future hardening pass that rejects
  // non-finite / negative inputs is tracked as an `audio::stt` follow-up
  // and would be a coordinated lm-side decision (the analogous lm-side
  // helper doesn't exist yet).
  let hours = (seconds / 3600.0).floor();
  // `(seconds % 3600) // 60` in python; rewritten as `(s / 60).floor() % 60`
  // to skip the intermediate `% 3600` (arithmetically equivalent for the
  // representable f64 range we care about — STT cues never exceed
  // 24 hours, well within f64's exact-integer range for the underlying
  // `seconds`).
  let minutes = (seconds / 60.0).floor() % 60.0;
  // `seconds % 60` in python; `%` on f64 in Rust uses the IEEE 754
  // remainder operation with the sign of the dividend — matches python
  // for non-negative dividends.
  let rem = seconds - (seconds / 60.0).floor() * 60.0;
  // `f"{seconds:06.3f}"` — min width 6, 3 decimal places (zero-padded).
  // Format with `.` first, then replace with `,` — mirrors python's
  // `.replace(".", ",")` exactly so a rare locale-dependent decimal
  // separator (Rust uses `.` universally) is normalized identically.
  let raw = format!("{:02}:{:02}:{:06.3}", hours as i64, minutes as i64, rem);
  raw.replace('.', ",")
}

/// Convert seconds to the WebVTT `HH:MM:SS.mmm` timestamp format.
///
/// 1:1 port of [`mlx_audio.stt.generate.format_vtt_timestamp`][stt-gen]
/// (`stt/generate.py:112-114`): same as [`format_timestamp`] but with `.`
/// (the WebVTT separator) instead of `,` (the SRT separator). The python
/// implementation literally calls `format_timestamp(seconds).replace(",",
/// ".")` — mlxrs mirrors that exactly.
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn format_vtt_timestamp(seconds: f64) -> String {
  format_timestamp(seconds).replace(',', ".")
}

/// Save the transcript as plain text to `<path>.txt`.
///
/// 1:1 port of [`mlx_audio.stt.generate.save_as_txt`][stt-gen]
/// (`stt/generate.py:135-141`): opens `f"{output_path}.txt"` UTF-8 and
/// writes `segments.text`. mlxrs preserves the **base-path** convention
/// (`.txt` extension is appended by the serializer, NOT supplied by the
/// caller) for parity with mlx-audio's writer signature; downstream CLI
/// wrappers that already pass the base path work unchanged.
///
/// ## Stdout passthrough (`path == "-"`)
///
/// Mirrors the python `output_path != "-"` branch in `save_as_txt`
/// (`stt/generate.py:136-140`): when `path` is exactly `"-"` mlxrs writes
/// to [`std::io::stdout`] WITHOUT appending the `.txt` extension (no
/// `-.txt` file is created on disk), matching python's
/// `contextlib.nullcontext(sys.stdout)` branch.
///
/// # Errors
///
/// Returns [`Error::Backend`] when the destination file cannot be created
/// (permission, missing directory, …) or any byte cannot be written
/// (`ENOSPC`, broken pipe, …). The destination directory is **not** auto-
/// created — that's the caller's responsibility (mlx-audio's
/// `generate_transcription` does an `os.makedirs(..., exist_ok=True)`
/// before calling the serializer, see `stt/generate.py:396`).
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn save_as_txt(transcript: &Transcript, path: &Path) -> Result<()> {
  if path == Path::new("-") {
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    return save_as_txt_stdout(transcript, &mut w);
  }
  let final_path = with_extension(path, "txt");
  let f = File::create(&final_path).map_err(|e| Error::Backend {
    message: format!("save_as_txt: create {} failed: {e}", final_path.display()),
  })?;
  let mut w = BufWriter::new(f);
  save_as_txt_to_writer(transcript, &mut w).map_err(|e| Error::Backend {
    message: format!("save_as_txt: write {} failed: {e}", final_path.display()),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_txt: flush {} failed: {e}", final_path.display()),
  })?;
  Ok(())
}

/// Write the TXT body of `transcript` to any [`Write`] sink — extracted from
/// [`save_as_txt`] so the python-shape rendering is shared by both the
/// on-disk file branch AND the `path == "-"` stdout-passthrough branch (and
/// is exercised directly by the unit-test stdout asserts without needing a
/// real fd-redirect).
fn save_as_txt_to_writer<W: Write>(transcript: &Transcript, w: &mut W) -> std::io::Result<()> {
  w.write_all(transcript.text().as_bytes())
}

/// Stdout-branch delegate for [`save_as_txt`] — writes the TXT body via
/// [`save_as_txt_to_writer`] AND explicitly flushes the writer, surfacing
/// either failure as [`Error::Backend`]. Factored out of [`save_as_txt`] so
/// the flush-error path is unit-testable without going through a real
/// stdout fd (the test substitutes a flush-failing writer).
///
/// `StdoutLock` is line-buffered when attached to a tty and fully-buffered
/// when redirected; the writer-helper writes raw `text` with no trailing
/// newline, so without an explicit flush the final bytes can sit in the
/// stdout buffer past `save_as_txt`'s return (especially on
/// redirect-to-file / redirect-to-pipe). Mirror the file branch's explicit
/// `BufWriter::flush()` and surface failures as `Error::Backend`
/// (broken-pipe, `ENOSPC` on the receiving end, ...).
fn save_as_txt_stdout<W: Write>(transcript: &Transcript, w: &mut W) -> Result<()> {
  save_as_txt_to_writer(transcript, w).map_err(|e| Error::Backend {
    message: format!("save_as_txt: write to stdout failed: {e}"),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_txt: stdout flush failed: {e}"),
  })?;
  Ok(())
}

/// Save the transcript as SubRip `.srt` to `<path>.srt`.
///
/// 1:1 port of [`mlx_audio.stt.generate.save_as_srt`][stt-gen]
/// (`stt/generate.py:144-155`). Per-cue format (one cue per segment, plus
/// one cue per word when the segment carries `words` — see [`get_cues`]):
///
/// ```text
/// {INDEX}
/// {HH:MM:SS,mmm} --> {HH:MM:SS,mmm}
/// {TEXT}
///
/// ```
///
/// Indexing is **1-based** (python `enumerate(_get_cues(segments), 1)`) and
/// the cue blocks are separated by a blank line (the `\n\n` after `{TEXT}`).
/// Timestamps use [`format_timestamp`] (`,` separator, SRT spec).
///
/// ## Stdout passthrough (`path == "-"`)
///
/// Mirrors the python `output_path != "-"` branch in `save_as_srt`
/// (`stt/generate.py:145-149`): when `path` is exactly `"-"` mlxrs writes
/// to [`std::io::stdout`] WITHOUT appending the `.srt` extension.
///
/// # Errors
///
/// Same conditions as [`save_as_txt`].
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn save_as_srt(transcript: &Transcript, path: &Path) -> Result<()> {
  if path == Path::new("-") {
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    return save_as_srt_stdout(transcript, &mut w);
  }
  let final_path = with_extension(path, "srt");
  let f = File::create(&final_path).map_err(|e| Error::Backend {
    message: format!("save_as_srt: create {} failed: {e}", final_path.display()),
  })?;
  let mut w = BufWriter::new(f);
  save_as_srt_to_writer(transcript, &mut w).map_err(|e| Error::Backend {
    message: format!("save_as_srt: write {} failed: {e}", final_path.display()),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_srt: flush {} failed: {e}", final_path.display()),
  })?;
  Ok(())
}

/// Write the SRT body of `transcript` to any [`Write`] sink — extracted from
/// [`save_as_srt`] so the python-shape rendering is shared by both the
/// on-disk file branch AND the `path == "-"` stdout-passthrough branch.
fn save_as_srt_to_writer<W: Write>(transcript: &Transcript, w: &mut W) -> std::io::Result<()> {
  for (i, cue) in get_cues(transcript).iter().enumerate() {
    // python `for i, cue in enumerate(..., 1)` → 1-based index.
    let idx = i + 1;
    let block = format!(
      "{}\n{} --> {}\n{}\n\n",
      idx,
      format_timestamp(cue.start()),
      format_timestamp(cue.end()),
      cue.text(),
    );
    w.write_all(block.as_bytes())?;
  }
  Ok(())
}

/// Stdout-branch delegate for [`save_as_srt`] — writes the SRT body via
/// [`save_as_srt_to_writer`] AND explicitly flushes the writer, surfacing
/// either failure as [`Error::Backend`]. See
/// [`save_as_txt_stdout`] for the buffered-stdout rationale; the SRT body
/// ends with `\n\n` after the last cue but the partial bytes can still sit
/// in the stdout buffer when redirected, so the explicit flush is required.
fn save_as_srt_stdout<W: Write>(transcript: &Transcript, w: &mut W) -> Result<()> {
  save_as_srt_to_writer(transcript, w).map_err(|e| Error::Backend {
    message: format!("save_as_srt: write to stdout failed: {e}"),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_srt: stdout flush failed: {e}"),
  })?;
  Ok(())
}

/// Save the transcript as WebVTT `.vtt` to `<path>.vtt`.
///
/// 1:1 port of [`mlx_audio.stt.generate.save_as_vtt`][stt-gen]
/// (`stt/generate.py:158-170`). Same per-cue shape as
/// [`save_as_srt`] but prefixed with the `WEBVTT\n\n` header and using
/// [`format_vtt_timestamp`] (`.` separator, WebVTT spec) instead of `,`:
///
/// ```text
/// WEBVTT
///
/// {INDEX}
/// {HH:MM:SS.mmm} --> {HH:MM:SS.mmm}
/// {TEXT}
///
/// ```
///
/// ## Stdout passthrough (`path == "-"`)
///
/// Mirrors the python `output_path != "-"` branch in `save_as_vtt`
/// (`stt/generate.py:159-163`): when `path` is exactly `"-"` mlxrs writes
/// to [`std::io::stdout`] WITHOUT appending the `.vtt` extension.
///
/// # Errors
///
/// Same conditions as [`save_as_txt`].
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn save_as_vtt(transcript: &Transcript, path: &Path) -> Result<()> {
  if path == Path::new("-") {
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    return save_as_vtt_stdout(transcript, &mut w);
  }
  let final_path = with_extension(path, "vtt");
  let f = File::create(&final_path).map_err(|e| Error::Backend {
    message: format!("save_as_vtt: create {} failed: {e}", final_path.display()),
  })?;
  let mut w = BufWriter::new(f);
  save_as_vtt_to_writer(transcript, &mut w).map_err(|e| Error::Backend {
    message: format!("save_as_vtt: write {} failed: {e}", final_path.display()),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_vtt: flush {} failed: {e}", final_path.display()),
  })?;
  Ok(())
}

/// Write the WebVTT body of `transcript` to any [`Write`] sink — extracted
/// from [`save_as_vtt`] so the python-shape rendering is shared by both the
/// on-disk file branch AND the `path == "-"` stdout-passthrough branch.
fn save_as_vtt_to_writer<W: Write>(transcript: &Transcript, w: &mut W) -> std::io::Result<()> {
  // `WEBVTT` magic header — required by every WebVTT parser, NOT optional.
  w.write_all(b"WEBVTT\n\n")?;
  for (i, cue) in get_cues(transcript).iter().enumerate() {
    let idx = i + 1;
    let block = format!(
      "{}\n{} --> {}\n{}\n\n",
      idx,
      format_vtt_timestamp(cue.start()),
      format_vtt_timestamp(cue.end()),
      cue.text(),
    );
    w.write_all(block.as_bytes())?;
  }
  Ok(())
}

/// Stdout-branch delegate for [`save_as_vtt`] — writes the VTT body via
/// [`save_as_vtt_to_writer`] AND explicitly flushes the writer, surfacing
/// either failure as [`Error::Backend`]. See
/// [`save_as_txt_stdout`] for the buffered-stdout rationale; the VTT body
/// (including the `WEBVTT\n\n` header + every cue block) is pushed past
/// the stdout buffer before [`save_as_vtt`] returns.
fn save_as_vtt_stdout<W: Write>(transcript: &Transcript, w: &mut W) -> Result<()> {
  save_as_vtt_to_writer(transcript, w).map_err(|e| Error::Backend {
    message: format!("save_as_vtt: write to stdout failed: {e}"),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_vtt: stdout flush failed: {e}"),
  })?;
  Ok(())
}

/// Save the transcript as `serde_json`-formatted JSON to `<path>.json`.
///
/// 1:1 port of [`mlx_audio.stt.generate.save_as_json`][stt-gen]
/// (`stt/generate.py:173-225`). The JSON shape mirrors python's
/// `json.dump(result, f, ensure_ascii=False, indent=2)`:
///
/// - [`Transcript::Sentences`] (Parakeet shape, python `hasattr(segments,
///   "sentences")` branch):
///   ```json
///   {
///     "text": "...",
///     "sentences": [
///       {"text": "...", "start": ..., "end": ..., "duration": ...,
///        "tokens": [{"text": "...", "start": ..., "end": ..., "duration": ...}, ...],
///        "speaker_id": "..."}  // optional, only when present
///     ]
///   }
///   ```
/// - [`Transcript::Segments`] (Whisper shape, python `else` branch):
///   ```json
///   {
///     "text": "...",
///     "segments": [
///       {"text": "...", "start": ..., "end": ..., "duration": ...,
///        "words": [...],         // optional, only when present
///        "speaker_id": "..."}    // optional, only when present
///     ]
///   }
///   ```
///
/// `duration` is computed as `end - start` for Whisper segments (mlx-audio
/// python `seg["duration"] = s["end"] - s["start"]`); for Parakeet
/// sentences the duration is carried verbatim from
/// [`Sentence::duration`]. `ensure_ascii=False` (python) → mlxrs writes
/// raw UTF-8 (no `\uXXXX` escaping) — `serde_json::to_writer_pretty`'s
/// default is also raw UTF-8 (it only escapes JSON-required code points),
/// matching the python behavior.
///
/// ## Stdout passthrough (`path == "-"`)
///
/// Mirrors the python `output_path != "-"` branch in `save_as_json`
/// (`stt/generate.py:220-224`): when `path` is exactly `"-"` mlxrs writes
/// to [`std::io::stdout`] WITHOUT appending the `.json` extension.
///
/// # Errors
///
/// Returns [`Error::Backend`] for any file-creation / write / flush /
/// serialization failure; the destination is left untouched on
/// pre-serialization errors and may be partially written on a mid-write
/// I/O failure (matching `save_as_txt` / `save_as_srt` / `save_as_vtt` —
/// the atomic-rename pattern `save_wav` uses is heavier than these
/// faithful-port serializers warrant).
///
/// [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
pub fn save_as_json(transcript: &Transcript, path: &Path) -> Result<()> {
  if path == Path::new("-") {
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    return save_as_json_stdout(transcript, &mut w);
  }
  let final_path = with_extension(path, "json");
  let f = File::create(&final_path).map_err(|e| Error::Backend {
    message: format!("save_as_json: create {} failed: {e}", final_path.display()),
  })?;
  let mut w = BufWriter::new(f);
  save_as_json_to_writer(transcript, &mut w).map_err(|e| Error::Backend {
    message: format!(
      "save_as_json: serialize {} failed: {e}",
      final_path.display()
    ),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_json: flush {} failed: {e}", final_path.display()),
  })?;
  Ok(())
}

/// Write the JSON body of `transcript` (python-shape pretty-printed,
/// 2-space-indent) to any [`Write`] sink — extracted from [`save_as_json`]
/// so the python-shape rendering is shared by both the on-disk file branch
/// AND the `path == "-"` stdout-passthrough branch.
///
/// `serde_json::Error` is folded into [`std::io::Error`] via
/// [`std::io::Error::other`] so the writer-facing signature mirrors the
/// txt/srt/vtt helpers; the outer [`save_as_json`] re-wraps into
/// [`Error::Backend`] uniformly.
fn save_as_json_to_writer<W: Write>(transcript: &Transcript, w: &mut W) -> std::io::Result<()> {
  // Build the python-shaped Value tree. We do NOT just `serde_json::to_value`
  // on `Transcript` directly because the python shape differs from
  // `Transcript`'s natural serde shape in two ways:
  //   1. python's Whisper branch adds a computed `"duration": end - start`
  //      per segment that `Segment` doesn't carry as a field.
  //   2. python's Parakeet branch reorders fields with `"text"` first vs
  //      `Sentence`'s declaration order; the json output should be byte-
  //      identical to the python output to support tooling that
  //      string-matches it.
  let value = transcript_to_python_shape(transcript);
  // python `json.dump(..., ensure_ascii=False, indent=2)` → 2-space indent +
  // no ASCII escape. `serde_json::to_writer_pretty` defaults to 2-space
  // indent (`PrettyFormatter::default()` uses `b"  "`) and never escapes
  // non-ASCII UTF-8 chars, so this matches the python output byte-for-byte
  // for the same key ordering.
  serde_json::to_writer_pretty(w, &value).map_err(std::io::Error::other)
}

/// Stdout-branch delegate for [`save_as_json`] — writes the JSON body via
/// [`save_as_json_to_writer`] AND explicitly flushes the writer, surfacing
/// either failure as [`Error::Backend`]. See
/// [`save_as_txt_stdout`] for the buffered-stdout rationale; the JSON body
/// ends with `\n}` (the trailing close-brace, NOT a final newline) which
/// is never a flush trigger on a line-buffered tty, and is held in the
/// buffer entirely on a redirected stdout — without an explicit flush the
/// final JSON bytes can sit past [`save_as_json`]'s return.
fn save_as_json_stdout<W: Write>(transcript: &Transcript, w: &mut W) -> Result<()> {
  save_as_json_to_writer(transcript, w).map_err(|e| Error::Backend {
    message: format!("save_as_json: serialize to stdout failed: {e}"),
  })?;
  w.flush().map_err(|e| Error::Backend {
    message: format!("save_as_json: stdout flush failed: {e}"),
  })?;
  Ok(())
}

/// Build a `serde_json::Value` matching the python `save_as_json` output
/// shape (`stt/generate.py:173-225`) — separated from [`save_as_json`] so
/// the python-shape transformation is unit-testable without touching the
/// filesystem.
fn transcript_to_python_shape(t: &Transcript) -> serde_json::Value {
  use serde_json::{Map, Value, json};
  match t {
    Transcript::Sentences(p) => {
      // python `result = {"text": ..., "sentences": [...]}`
      let mut sents_arr: Vec<Value> = Vec::with_capacity(p.sentences().len());
      for s in p.sentences() {
        // python builds dict in {text, start, end, duration, tokens} order,
        // THEN appends speaker_id post-hoc (lines 196-199). Mirror exactly.
        let mut obj = Map::new();
        obj.insert("text".into(), Value::String(s.text().to_owned()));
        obj.insert("start".into(), json!(s.start()));
        obj.insert("end".into(), json!(s.end()));
        obj.insert("duration".into(), json!(s.duration()));
        let tok_arr: Vec<Value> = s
          .tokens()
          .iter()
          .map(|tk| {
            let mut tobj = Map::new();
            tobj.insert("text".into(), Value::String(tk.text().to_owned()));
            tobj.insert("start".into(), json!(tk.start()));
            tobj.insert("end".into(), json!(tk.end()));
            tobj.insert("duration".into(), json!(tk.duration()));
            Value::Object(tobj)
          })
          .collect();
        obj.insert("tokens".into(), Value::Array(tok_arr));
        if !s.speaker_id().is_empty() {
          obj.insert(
            "speaker_id".into(),
            Value::String(s.speaker_id().to_owned()),
          );
        }
        sents_arr.push(Value::Object(obj));
      }
      let mut root = Map::new();
      root.insert("text".into(), Value::String(p.text().to_owned()));
      root.insert("sentences".into(), Value::Array(sents_arr));
      Value::Object(root)
    }
    Transcript::Segments(p) => {
      // python `result = {"text": ..., "segments": []}` then `result["segments"].append(seg)`
      // where each `seg = {text, start, end, duration}` plus optional words
      // / speaker_id (lines 206-218). Mirror the dict-insertion order exactly.
      let mut segs_arr: Vec<Value> = Vec::with_capacity(p.segments().len());
      for s in p.segments() {
        let mut obj = Map::new();
        obj.insert("text".into(), Value::String(s.text().to_owned()));
        obj.insert("start".into(), json!(s.start()));
        obj.insert("end".into(), json!(s.end()));
        // python `"duration": s["end"] - s["start"]` — computed, NOT carried
        // on `Segment`. Use the same subtract so a Segment with degenerate
        // (end < start) timing produces the python-equivalent negative
        // duration rather than a Rust-side clamp.
        obj.insert("duration".into(), json!(s.end() - s.start()));
        // python `if "words" in s and s["words"]` (stt/generate.py:213): the
        // `words` key is emitted ONLY when the per-segment word list is
        // truthy (non-empty). An empty Vec is python-falsy and MUST be
        // dropped from the JSON, matching the python branch — otherwise
        // downstream tooling that `"words" in seg` checks would diverge
        // between python + mlxrs.
        if !s.words_slice().is_empty() {
          let words_arr: Vec<Value> = s
            .words_slice()
            .iter()
            .map(|w| {
              // python `seg["words"] = s["words"]` is a pass-through of the
              // per-word dict. We reconstruct {start, end, word, **extra}
              // to mirror the most common per-word dict shape; the `extra`
              // map's BTreeMap ordering keeps the JSON deterministic.
              let mut wobj = Map::new();
              wobj.insert("start".into(), json!(w.start()));
              wobj.insert("end".into(), json!(w.end()));
              wobj.insert("word".into(), Value::String(w.word().to_owned()));
              for (k, v) in w.extra() {
                wobj.insert(k.clone(), v.clone());
              }
              Value::Object(wobj)
            })
            .collect();
          obj.insert("words".into(), Value::Array(words_arr));
        }
        if !s.speaker_id().is_empty() {
          obj.insert(
            "speaker_id".into(),
            Value::String(s.speaker_id().to_owned()),
          );
        }
        segs_arr.push(Value::Object(obj));
      }
      let mut root = Map::new();
      root.insert("text".into(), Value::String(p.text().to_owned()));
      root.insert("segments".into(), Value::Array(segs_arr));
      Value::Object(root)
    }
  }
}

/// Append `ext` to `path` as a new extension component — the python
/// `f"{output_path}.{ext}"` convention (NOT [`Path::with_extension`], which
/// REPLACES any existing extension).
///
/// `path = "out"`, `ext = "txt"` → `"out.txt"`.
/// `path = "out.draft"`, `ext = "txt"` → `"out.draft.txt"` (not "out.txt").
fn with_extension(path: &Path, ext: &str) -> std::path::PathBuf {
  let mut s = path.as_os_str().to_owned();
  s.push(".");
  s.push(ext);
  std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
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

  // ---------- Finding 2 regression: writer-helper round-trip ----------

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

  // ---------- Round-2 finding: dash-stdout flush coverage ----------

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
  ///   tests to assert the flush-error path surfaces as [`Error::Backend`]
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
      Error::Backend { message } => {
        assert!(
          message.contains("save_as_txt") && message.contains("stdout flush failed"),
          "Error::Backend message must mention save_as_txt + stdout flush failure (got: {message})"
        );
      }
      other => panic!("expected Error::Backend, got {other:?}"),
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
      Error::Backend { message } => {
        assert!(
          message.contains("save_as_srt") && message.contains("stdout flush failed"),
          "Error::Backend message must mention save_as_srt + stdout flush failure (got: {message})"
        );
      }
      other => panic!("expected Error::Backend, got {other:?}"),
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
      Error::Backend { message } => {
        assert!(
          message.contains("save_as_vtt") && message.contains("stdout flush failed"),
          "Error::Backend message must mention save_as_vtt + stdout flush failure (got: {message})"
        );
      }
      other => panic!("expected Error::Backend, got {other:?}"),
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
      Error::Backend { message } => {
        assert!(
          message.contains("save_as_json") && message.contains("stdout flush failed"),
          "Error::Backend message must mention save_as_json + stdout flush failure (got: {message})"
        );
      }
      other => panic!("expected Error::Backend, got {other:?}"),
    }
    assert_eq!(
      w.flush_calls, 1,
      "flush must have been attempted exactly once"
    );
  }
}

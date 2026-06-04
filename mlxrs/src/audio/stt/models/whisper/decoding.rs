//! Whisper decoding — the full [`DecodingTask`] (`decoding.py:445-723`) plus
//! the temperature-fallback + 30-second seek loop ([`transcribe`],
//! `whisper.py:787-1300` core path).
//!
//! Faithful port of `mlx_audio.stt.models.whisper.decoding` for the
//! **single-utterance greedy** decode mode mlxrs ships (the reference raises
//! `NotImplementedError` for beam search; `best_of` sampling needs the
//! multi-trajectory ranker, deferred). The decode of one 30-second mel
//! segment produces one [`DecodingResult`]; [`transcribe`] slides the segment
//! over a longer mel and runs the temperature fallback per segment.
//!
//! ## Shape vs the reference
//! The reference [`DecodingTask`] is batch-oriented (`(n_audio, n_group)`);
//! mlxrs decodes one utterance (`n_audio = 1`) greedily (`n_group = 1`), so
//! the per-step logits are a single `(n_vocab,)` row and the token history is
//! a `Vec<u32>`. The three `LogitFilter`s apply to that row exactly as the
//! reference's numpy masks apply to `logits[k]` for `k = 0`.
//!
//! ## Building blocks reused
//! - `WhisperModel::decode_tokens` — the `Inference.logits` analogue (decoder
//!   forward with a caller-owned KV cache);
//! - [`AutoregressiveStt::encode`] — the encoder forward;
//! - [`super::tokenizer::HFTokenizerWrapper`] — the special-token ids +
//!   `sot_sequence` + `encode` / `decode`;
//! - [`crate::lm::sample::categorical_sampling`] — the `temperature > 0`
//!   draw (`GreedyDecoder`'s `mx.random.categorical`).

use std::io::Write as _;

use flate2::{Compression, write::ZlibEncoder};
use smol_str::format_smolstr;

use crate::{
  Array, Dtype, Error, Result,
  audio::stt::model::AutoregressiveStt,
  error::{InvariantViolationPayload, OutOfRangePayload},
  ops,
};

use super::{
  audio::{CHUNK_LENGTH, FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, SAMPLE_RATE, pad_or_trim},
  model::WhisperModel,
  timing,
  tokenizer::HFTokenizerWrapper,
};

/// The gzip/zlib compression ratio of `text` — `compression_ratio`
/// (`decoding.py:15-17`): `len(utf8) / len(zlib.compress(utf8))`.
///
/// A high ratio (`> compression_ratio_threshold`, default `2.4`) flags a
/// degenerate, highly-repetitive decode for the temperature fallback. The
/// reference uses Python's `zlib.compress` (DEFLATE, default level 6); this
/// uses [`flate2`]'s zlib encoder at the matching level. An empty string has
/// no ratio (the reference would divide by a non-zero zlib header length);
/// mlxrs returns `0.0` so it never spuriously trips the `> 2.4` gate.
pub fn compression_ratio(text: &str) -> f64 {
  let bytes = text.as_bytes();
  if bytes.is_empty() {
    return 0.0;
  }
  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
  // Writing to an in-memory `Vec` is infallible; `finish` flushes the stream.
  if encoder.write_all(bytes).is_err() {
    return 0.0;
  }
  match encoder.finish() {
    Ok(compressed) if !compressed.is_empty() => bytes.len() as f64 / compressed.len() as f64,
    _ => 0.0,
  }
}

/// The decode task — `"transcribe"` (source language) or `"translate"`
/// (into English). The reference threads this as the `DecodingOptions.task`
/// string; this re-exports the tokenizer's [`Task`].
pub use super::tokenizer::Task;

/// Options controlling a single 30-second-segment decode — `DecodingOptions`
/// (`decoding.py:115-149`), restricted to the fields the shipped
/// single-utterance greedy decode honors.
///
/// Beam search (`beam_size` / `patience`) and best-of-N sampling (`best_of`)
/// are not shipped (the reference raises `NotImplementedError` for beam
/// search), so those fields are omitted rather than carried-and-ignored. The
/// `prompt` / `prefix` conditioning, the timestamp options, and the
/// suppress-token controls are all honored.
#[derive(Debug, Clone)]
pub struct DecodingOptions {
  /// `"transcribe"` (X→X) or `"translate"` (X→English).
  pub task: Task,
  /// Language code (e.g. `"en"`); `None` ⇒ detect it (multilingual models)
  /// or default to `"en"` (English-only models).
  pub language: Option<String>,
  /// Sampling temperature. `0.0` ⇒ greedy argmax; `> 0.0` ⇒ a categorical
  /// draw (`GreedyDecoder`).
  pub temperature: f32,
  /// Maximum tokens to sample; `None` ⇒ `n_text_ctx / 2`
  /// (`decoding.py:461`). `Some(0)` emits no sampled tokens (matching the stt
  /// driver's `max_new_tokens == 0` semantics).
  pub sample_len: Option<usize>,
  /// Previous-context token ids fed before the current window
  /// (`decoding.py:539-549`).
  pub prompt: Vec<u32>,
  /// Prefix token ids forced at the start of the current window
  /// (`decoding.py:528-537`).
  pub prefix: Vec<u32>,
  /// Token ids to suppress. `SuppressSpec::NonSpeech` reproduces the
  /// reference default `"-1"` (the tokenizer
  /// [`non_speech_tokens`](HFTokenizerWrapper::non_speech_tokens) set + the
  /// special task / sot / sot_prev / sot_lm / no_speech tokens);
  /// `SuppressSpec::Ids` is an explicit list.
  pub suppress_tokens: SuppressSpec,
  /// Suppress blank (space / eot) at the first sampled position
  /// (`decoding.py:142`).
  pub suppress_blank: bool,
  /// Decode text only, with `<|notimestamps|>` in the sot sequence
  /// (`decoding.py:145`).
  pub without_timestamps: bool,
  /// Max initial timestamp in seconds; `None` disables the cap
  /// (`decoding.py:146`). Default `1.0`.
  pub max_initial_timestamp: Option<f32>,
}

impl Default for DecodingOptions {
  /// The reference defaults (`decoding.py:115-149`): transcribe, detect
  /// language, greedy (`temperature = 0`), suppress non-speech + blank, emit
  /// timestamps, `max_initial_timestamp = 1.0`.
  fn default() -> Self {
    Self {
      task: Task::Transcribe,
      language: None,
      temperature: 0.0,
      sample_len: None,
      prompt: Vec::new(),
      prefix: Vec::new(),
      suppress_tokens: SuppressSpec::NonSpeech,
      suppress_blank: true,
      without_timestamps: false,
      max_initial_timestamp: Some(1.0),
    }
  }
}

impl DecodingOptions {
  /// Clone every field **except** [`prompt`](Self::prompt), which is returned
  /// empty.
  ///
  /// The seek loop overwrites the prompt every window with the running history
  /// tail (`whisper.py:1033`), so the caller's original (possibly unbounded)
  /// `prompt` is never used by the per-window decode. Cloning `self` and then
  /// clearing `prompt` would copy that whole vector just to drop it; this skips
  /// the copy entirely. Behavior is unchanged — `prompt` is the only field the
  /// loop sets, and it is set to the same value either way.
  fn clone_without_prompt(&self) -> Self {
    Self {
      task: self.task,
      language: self.language.clone(),
      temperature: self.temperature,
      sample_len: self.sample_len,
      prompt: Vec::new(),
      prefix: self.prefix.clone(),
      suppress_tokens: self.suppress_tokens.clone(),
      suppress_blank: self.suppress_blank,
      without_timestamps: self.without_timestamps,
      max_initial_timestamp: self.max_initial_timestamp,
    }
  }
}

/// The `suppress_tokens` option (`decoding.py:139-141`): either the
/// reference's `"-1"` non-speech default or an explicit id list.
#[derive(Debug, Clone)]
pub enum SuppressSpec {
  /// The reference `"-1"`: suppress the tokenizer
  /// [`non_speech_tokens`](HFTokenizerWrapper::non_speech_tokens) set plus the
  /// task / sot / sot_prev / sot_lm / no_speech specials
  /// ([`get_suppress_tokens`]).
  NonSpeech,
  /// Suppress exactly these ids (still augmented with the specials, per
  /// `get_suppress_tokens`'s unconditional `result.extend([...])`).
  Ids(Vec<u32>),
  /// Suppress nothing (the reference's falsy `suppress_tokens`, e.g. an empty
  /// string / list — the `SuppressTokens` filter is not installed).
  None,
}

/// The result of decoding one 30-second mel segment — `DecodingResult`
/// (`decoding.py:152-162`), single-utterance.
#[derive(Debug, Clone)]
pub struct DecodingResult {
  /// The detected (or supplied) language code.
  pub language: String,
  /// The sampled token ids (between the sot sequence and the first eot).
  pub tokens: Vec<u32>,
  /// The decoded text (timestamp tokens dropped, trimmed).
  pub text: String,
  /// Mean token log-probability (`sum_logprob / (len + 1)`,
  /// `decoding.py:694-696`).
  pub avg_logprob: f64,
  /// `P(<|nospeech|>)` at the sot position (`decoding.py:611-613`); `NaN` if
  /// the model has no no-speech token.
  pub no_speech_prob: f64,
  /// The temperature this result was decoded at.
  pub temperature: f32,
  /// The decoded text's [`compression_ratio`].
  pub compression_ratio: f64,
}

/// Build the suppress-token id set — `get_suppress_tokens`
/// (`decoding.py:80-112`).
///
/// For [`SuppressSpec::NonSpeech`] this is the tokenizer's
/// [`HFTokenizerWrapper::non_speech_tokens`] set (the reference `"-1"`
/// default) plus the always-suppressed task / sot / sot_prev / sot_lm tokens
/// (+ no_speech). For [`SuppressSpec::Ids`] the explicit ids (minus any `-1`
/// sentinel) are used; the unconditional specials are always added.
/// `transcribe` / `translate` / `sot` / `sot_prev` / `sot_lm` / `no_speech`
/// are always suppressed (`decoding.py:99-110`).
///
/// # Errors
/// Propagates [`HFTokenizerWrapper::non_speech_tokens`]'s encode error when
/// the non-speech set is requested.
pub fn get_suppress_tokens(
  tokenizer: &HFTokenizerWrapper<'_>,
  spec: &SuppressSpec,
) -> Result<Vec<u32>> {
  let mut result: Vec<u32> = match spec {
    SuppressSpec::None => return Ok(Vec::new()),
    // `-1` ⇒ extend with the tokenizer non-speech set (`decoding.py:95-97`).
    SuppressSpec::NonSpeech => tokenizer.non_speech_tokens()?,
    SuppressSpec::Ids(ids) => ids.iter().copied().filter(|&t| t != NEG_ONE_U32).collect(),
  };

  // Always-suppressed specials (`decoding.py:99-110`): the task tokens, sot,
  // sot_prev, sot_lm, and no_speech.
  result.push(tokenizer.transcribe());
  result.push(tokenizer.translate());
  result.push(tokenizer.sot());
  result.push(tokenizer.sot_prev());
  result.push(tokenizer.sot_lm());
  result.push(tokenizer.no_speech());

  // `sorted(set(result))`.
  result.sort_unstable();
  result.dedup();
  Ok(result)
}

/// The reference's `-1` sentinel meaning "non-speech set"; filtered out of an
/// explicit id list (it is not a real token id).
const NEG_ONE_U32: u32 = u32::MAX;

// ───────────────────────── logit filters ──────────────────────────────────

/// A logit filter — `LogitFilter` (`decoding.py:333-346`). Applies to the
/// single-row logits **on device** given the current token history, returning
/// the masked logits as a new [`Array`] (the reference's `return logits +
/// mask`, where the mask is `-inf` at suppressed positions).
///
/// `logits` is the `(n_vocab,)` row (kept on device — never copied to the
/// host). `tokens` is the full decoded history (sot sequence + sampled
/// tokens, already host-side integers the decode loop owns), `sample_begin`
/// the index of the first sampled (post-sot) token. Each filter precomputes
/// its constant mask(s) once at construction and adds them on device, so the
/// per-step cost is an `add` (and, for the timestamp rules, the on-device
/// probability-mass comparison) — no host round-trip of the `n_vocab` row.
trait LogitFilter {
  /// Apply the filter to the `(n_vocab,)` logits row, returning the masked row.
  ///
  /// # Errors
  /// Propagates the device add / mask-construction / reduction op errors.
  fn apply(&self, logits: &Array, tokens: &[u32]) -> Result<Array>;
}

/// Build a `(n_vocab,)` additive mask Array — `0.0` everywhere except `-inf`
/// at each id in `ids` (the device equivalent of the reference's numpy
/// `mask = np.zeros(n_vocab); mask[ids] = -inf`). Out-of-range ids are skipped
/// (matching the CPU path's `get_mut` guard).
///
/// The `-inf` entries mark which slots [`overwrite_masked`] forces to `-inf`;
/// the suppression is applied as a boolean OVERWRITE (`select`), not an add, so
/// it is bit-equivalent to the reference's in-place `logits[ids] = -inf` for any
/// prior logit value — a `+inf` or `NaN` at a suppressed slot still becomes
/// `-inf`, where an additive `+ (-inf)` would have produced `NaN`.
fn scatter_neg_inf_mask(n_vocab: usize, ids: &[u32]) -> Result<Array> {
  let mut buf = vec![0.0_f32; n_vocab];
  for &id in ids {
    if let Some(slot) = buf.get_mut(id as usize) {
      *slot = f32::NEG_INFINITY;
    }
  }
  let n = i32::try_from(n_vocab).map_err(|_| dim_overflow("n_vocab"))?;
  Array::from_slice::<f32>(&buf, &[n])
}

/// Apply an additive `0.0` / `-inf` suppression `mask` to `logits` as an
/// OVERWRITE rather than an add: every slot the mask marks `-inf` is forced to a
/// real `-inf` in the output, every other slot keeps its `logits` value.
///
/// This is the device equivalent of the reference's in-place assignment
/// `logits[ids] = -inf` (numpy / `mx` indexed store), and is bit-equivalent to
/// it for ALL inputs — including non-finite ones. An additive `logits + mask`
/// only matches the assignment for finite logits: at a suppressed slot a `+inf`
/// logit would give `(+inf) + (-inf) = NaN` and a `NaN` logit would give `NaN`,
/// either of which then poisons the downstream `argmax` / `logsumexp` / chosen
/// log-prob. The boolean overwrite sets the slot to `-inf` regardless of the
/// prior value, so the masked region is never `NaN`.
///
/// `mask` is `(n_vocab,)`; the masked-slot predicate is `isneginf(mask)` (its
/// only non-zero entries are `-inf`), and `select(cond, -inf, logits)` keeps the
/// whole `n_vocab` row on device.
///
/// # Errors
/// Propagates the predicate / `full` / `select` op errors.
fn overwrite_masked(logits: &Array, mask: &Array) -> Result<Array> {
  // `true` exactly at the slots the mask marks `-inf` (the suppressed ids).
  let suppressed = ops::comparison::isneginf(mask)?;
  // A rank-0 `-inf` that `select` broadcasts over the masked slots; the
  // unmasked slots keep their original `logits` value.
  let neg_inf = Array::full::<f32>(&[0i32; 0], f32::NEG_INFINITY)?;
  ops::logical::select(&suppressed, &neg_inf, logits)
}

/// Suppress blank outputs at the first sampled position — `SuppressBlank`
/// (`decoding.py:349-359`): at `tokens.len() == sample_begin`, force the space
/// token(s) and eot to `-inf` via a precomputed suppression mask.
struct SuppressBlank {
  sample_begin: usize,
  /// The precomputed `(n_vocab,)` suppression mask: `-inf` at `encode(" ") +
  /// [eot]`, `0` elsewhere — built once (`decoding.py:352-354`). Applied as a
  /// boolean OVERWRITE ([`overwrite_masked`]), not an add, so a non-finite logit
  /// at a suppressed slot still becomes `-inf` (matching the reference's
  /// `logits[ids] = -inf` for all inputs).
  mask: Array,
}

impl SuppressBlank {
  /// Build from the tokenizer: `mask[encode(" ") + [eot]] = -inf`
  /// (`decoding.py:353`), precomputed as a device Array.
  fn new(tokenizer: &HFTokenizerWrapper<'_>, sample_begin: usize, n_vocab: usize) -> Result<Self> {
    let mut blank_ids = tokenizer.encode(" ")?;
    blank_ids.push(tokenizer.eot());
    Ok(Self {
      sample_begin,
      mask: scatter_neg_inf_mask(n_vocab, &blank_ids)?,
    })
  }
}

impl LogitFilter for SuppressBlank {
  fn apply(&self, logits: &Array, tokens: &[u32]) -> Result<Array> {
    if tokens.len() == self.sample_begin {
      overwrite_masked(logits, &self.mask)
    } else {
      logits.try_clone()
    }
  }
}

/// Suppress a fixed id set at every step — `SuppressTokens`
/// (`decoding.py:362-369`): force each suppressed id to `-inf` unconditionally.
struct SuppressTokens {
  /// The precomputed `(n_vocab,)` suppression mask: `-inf` at each suppressed
  /// id. Applied as a boolean OVERWRITE ([`overwrite_masked`]), not an add, so a
  /// non-finite logit at a suppressed slot still becomes `-inf`.
  mask: Array,
}

impl SuppressTokens {
  /// Build the precomputed `(n_vocab,)` `-inf` mask over `ids`
  /// (`decoding.py:364-366`).
  fn new(ids: &[u32], n_vocab: usize) -> Result<Self> {
    Ok(Self {
      mask: scatter_neg_inf_mask(n_vocab, ids)?,
    })
  }
}

impl LogitFilter for SuppressTokens {
  fn apply(&self, logits: &Array, _tokens: &[u32]) -> Result<Array> {
    overwrite_masked(logits, &self.mask)
  }
}

/// The timestamp-pair rules — `ApplyTimestampRules` (`decoding.py:372-442`).
///
/// Enforces that timestamp tokens (`>= timestamp_begin`) appear in pairs
/// except directly before eot, that timestamps are non-decreasing, that the
/// first sampled token is a timestamp (when emitting timestamps), and the
/// `max_initial_timestamp` cap. Finally, if the summed probability over all
/// timestamp tokens exceeds the max single text-token probability, it forces a
/// timestamp by masking every non-timestamp token.
///
/// The token-history-driven part of the mask is built host-side from the
/// `tokens` slice (the reference's `mask = np.zeros(...); mask[...] = -inf`),
/// then the probability-mass rule is evaluated **on device** against the
/// logits (so the `n_vocab` row never leaves the device); both are added to
/// the logits.
struct ApplyTimestampRules {
  sample_begin: usize,
  timestamp_begin: u32,
  /// `no_timestamps` id, suppressed unconditionally (it is handled by the
  /// `without_timestamps` sot sequence, not sampled).
  no_timestamps: u32,
  /// eot id (the boundary between text tokens `< eot` and specials).
  eot: u32,
  /// `round(max_initial_timestamp / precision)`; `None` disables the cap.
  max_initial_timestamp_index: Option<usize>,
  /// `n_vocab` — the logits-row width the host mask buffer is sized to.
  n_vocab: usize,
}

impl ApplyTimestampRules {
  fn new(
    tokenizer: &HFTokenizerWrapper<'_>,
    sample_begin: usize,
    max_initial_timestamp_index: Option<usize>,
    n_vocab: usize,
  ) -> Self {
    Self {
      sample_begin,
      timestamp_begin: tokenizer.timestamp_begin(),
      no_timestamps: tokenizer.no_timestamps(),
      eot: tokenizer.eot(),
      max_initial_timestamp_index,
      n_vocab,
    }
  }

  /// Build the **deterministic** additive `(n_vocab,)` mask from the token
  /// history alone — every rule of `ApplyTimestampRules.apply`
  /// (`decoding.py:384-426`) except the final probability-mass rule (which
  /// needs the logit values). Returns the host buffer (`0.0` / `-inf` per
  /// position), matching the reference's `mask = np.zeros(...)` exactly.
  fn deterministic_mask(&self, tokens: &[u32]) -> Vec<f32> {
    let n_vocab = self.n_vocab;
    let ts_begin = self.timestamp_begin as usize;
    let eot = self.eot as usize;
    let mut mask = vec![0.0_f32; n_vocab];

    // suppress <|notimestamps|> (`decoding.py:386-387`).
    if let Some(slot) = mask.get_mut(self.no_timestamps as usize) {
      *slot = f32::NEG_INFINITY;
    }

    // The sampled tail (post-sot sequence) — `seq = tokens[sample_begin:]`.
    let seq: &[u32] = tokens.get(self.sample_begin..).unwrap_or(&[]);

    let last_was_timestamp = seq.last().is_some_and(|&t| t >= self.timestamp_begin);
    // `penultimate_was_timestamp`: True when fewer than 2 sampled tokens, or
    // the second-to-last is a timestamp (`decoding.py:396-398`).
    let penultimate_was_timestamp = seq.len() < 2
      || seq
        .get(seq.len() - 2)
        .is_some_and(|&t| t >= self.timestamp_begin);

    if last_was_timestamp {
      if penultimate_was_timestamp {
        // Must be a non-timestamp next: mask all timestamp tokens.
        mask_range(&mut mask, ts_begin, n_vocab);
      } else {
        // Cannot be a normal text token: mask `[0, eot)`.
        mask_range(&mut mask, 0, eot);
      }
    }

    // Timestamps shouldn't decrease, and each segment must have a nonzero
    // length (`decoding.py:406-415`). A timestamp is any token
    // `>= timestamp_begin` (so `<|0.00|>` itself counts), and the upper masked
    // bound is the last seen timestamp `+ 1` — forcing the next timestamp to
    // be strictly greater — except when the last token opens a fresh pair (a
    // single trailing timestamp after a non-timestamp), where the same value
    // may close it. Masking `[timestamp_begin, last + 1)` when the last seen
    // timestamp is `<|0.00|>` masks `<|0.00|>` itself, preventing a zero-length
    // closing segment.
    let mut last_timestamp: Option<usize> = None;
    for &v in seq {
      if (v as usize) >= ts_begin {
        last_timestamp = Some(v as usize);
      }
    }
    if let Some(last_ts) = last_timestamp {
      let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
        // A single trailing timestamp opening a pair: the closing timestamp may
        // equal it, so do not add 1.
        last_ts
      } else {
        last_ts + 1
      };
      mask_range(&mut mask, ts_begin, timestamp_last);
    }

    // First sampled position: force a timestamp and apply the
    // `max_initial_timestamp` cap (`decoding.py:417-426`).
    if tokens.len() == self.sample_begin {
      mask_range(&mut mask, 0, ts_begin);
      if let Some(idx) = self.max_initial_timestamp_index {
        let last_allowed = ts_begin + idx;
        mask_range(&mut mask, last_allowed + 1, n_vocab);
      }
    }

    mask
  }
}

impl LogitFilter for ApplyTimestampRules {
  fn apply(&self, logits: &Array, tokens: &[u32]) -> Result<Array> {
    let n_vocab = self.n_vocab;
    let ts_begin = self.timestamp_begin as usize;
    let n = i32::try_from(n_vocab).map_err(|_| dim_overflow("n_vocab"))?;

    // Apply the deterministic (token-history) mask FIRST, exactly as the CPU
    // path masks the logits in place before the probability-mass rule reads
    // them. The deterministic buffer is built host-side from the token slice
    // (`0.0` / `-inf` per position), uploaded once, and applied as a boolean
    // OVERWRITE (`select`), not an add — so a suppressed slot becomes a real
    // `-inf` for ANY prior logit (a `+inf` / `NaN` logit would become `NaN`
    // under an additive `+ (-inf)`, poisoning the probability-mass `logsumexp`
    // and `max` below). The timestamp region the pair / monotonicity rules
    // already forbade is therefore `-inf` when the probability-mass rule reads
    // it (its `logsumexp` over an all-`-inf` region is `-inf`, never re-opening
    // text tokens).
    let det = self.deterministic_mask(tokens);
    let det_mask = Array::from_slice::<f32>(&det, &[n])?;
    let masked = overwrite_masked(logits, &det_mask)?;

    // The probability-mass rule, ON DEVICE (`decoding.py:428-441`): if the
    // summed probability over timestamps exceeds the max single text-token
    // probability, force a timestamp by masking `[0, timestamp_begin)`. The
    // reference computes the full log-probability row FIRST and slices it
    // AFTER:
    //   logprobs            = logits - logsumexp(logits, axis=-1)
    //   timestamp_logprob   = logprobs[timestamp_begin:].logsumexp(axis=-1)
    //   max_text_token_logprob = logprobs[:timestamp_begin].max(axis=-1)
    //   force               = timestamp_logprob > max_text_token_logprob
    // This is mirrored LITERALLY here rather than via the algebraically
    // cancelled `logsumexp(masked[ts_begin:]) > max(masked[:ts_begin])`. The
    // two agree for FINITE rows (the shared `- logsumexp(logits)` normalizer
    // cancels), but NOT for non-finite ones: with a `+inf` timestamp logit the
    // full-row `logsumexp(logits)` is `+inf`, so its `logprobs` slot is
    // `+inf - +inf = NaN`; `timestamp_logprob` is then NaN and `NaN > max_text`
    // is FALSE (no force) — whereas the cancelled form sees `+inf > max_text`
    // and wrongly masks every text token. Computing `logprobs` literally
    // reproduces the reference for ALL inputs, including `+inf` and `NaN`.
    // The comparison reads the **already-deterministically-masked** logits
    // (matching the CPU in-place order) and yields a rank-0 bool that `where`
    // broadcasts over the `[0, ts_begin)` region — the whole `n_vocab` row
    // stays on device.
    //
    // Reference precision: this rule runs in FLOAT32. In the mlx-audio reference
    // (`decoding.py:430-436`) `logprobs` and its `logsumexp` / `max` are all in
    // the dtype of `logits` — and the per-step logits are float32 because
    // `Inference.logits` returns `logits.astype(mx.float32)` (`decoding.py:175`)
    // before the filter chain. Upstream openai-whisper likewise computes
    // `logprobs = F.log_softmax(logits.float(), dim=-1)` (float32) then its
    // `logsumexp` / `max`. The device F32 `logsumexp` / `max` / `greater` below
    // therefore match the reference precision exactly; they are NOT widened to
    // f64 (which would diverge from both references on a knife-edge row).
    let ts_b = ts_begin.min(n_vocab);
    let ts_b_i = i32::try_from(ts_b).map_err(|_| dim_overflow("timestamp_begin"))?;
    if ts_b == 0 || ts_b >= n_vocab {
      // No text region or no timestamp region — the rule is inert.
      return Ok(masked);
    }
    // logprobs = masked - logsumexp(masked) over the full vocab axis, in F32.
    // `masked` is rank-1 `(n_vocab,)`, so the rank-0 `logsumexp` broadcasts.
    let row_lse = ops::reduction::logsumexp(&masked, false)?; // rank-0
    let logprobs = ops::arithmetic::subtract(&masked, &row_lse)?;
    let ts_slice = ops::indexing::slice(&logprobs, &[ts_b_i], &[n], &[1])?;
    let text_slice = ops::indexing::slice(&logprobs, &[0], &[ts_b_i], &[1])?;
    let ts_lse = ops::reduction::logsumexp(&ts_slice, false)?; // rank-0
    let text_max = ops::reduction::max(&text_slice, false)?; // rank-0
    let force = ops::comparison::greater(&ts_lse, &text_max)?; // rank-0 bool

    // masked[:ts_begin] = where(force, -inf, masked[:ts_begin]).
    let head = ops::indexing::slice(&masked, &[0], &[ts_b_i], &[1])?;
    let neg_inf = Array::full::<f32>(&[ts_b_i], f32::NEG_INFINITY)?;
    let new_head = ops::logical::select(&force, &neg_inf, &head)?;
    ops::indexing::slice_update(&masked, &new_head, &[0], &[ts_b_i], &[1])
  }
}

/// Set `mask[lo..hi]` to `-inf` (a numpy `mask[lo:hi] = -inf` slice),
/// clamping `hi` to the buffer length.
fn mask_range(mask: &mut [f32], lo: usize, hi: usize) {
  let hi = hi.min(mask.len());
  if lo < hi {
    for slot in &mut mask[lo..hi] {
      *slot = f32::NEG_INFINITY;
    }
  }
}

/// `log(sum(exp(x)))` over a slice, in `f64` for stability (numpy's
/// `logsumexp`). Empty (or all `-inf`) ⇒ `-inf`. Used by the once-per-utterance
/// [`detect_language`] masking (not the per-step decode, which reduces on
/// device).
fn logsumexp_slice(xs: &[f32]) -> f64 {
  let mut max = f64::NEG_INFINITY;
  for &x in xs {
    let x = x as f64;
    if x > max {
      max = x;
    }
  }
  if !max.is_finite() {
    // All `-inf` (or empty) ⇒ sum of exp is 0 ⇒ `-inf`.
    return f64::NEG_INFINITY;
  }
  let mut sum = 0.0_f64;
  for &x in xs {
    sum += (x as f64 - max).exp();
  }
  max + sum.ln()
}

// ───────────────────────── greedy decoder ─────────────────────────────────

/// The greedy token decoder — `GreedyDecoder` (`decoding.py:302-330`),
/// single-sequence.
///
/// Selects the next token (argmax for `temperature == 0`, else a categorical
/// draw) **on device**, accumulates the chosen token's log-probability, and
/// reports completion on eot. State across steps is the running `sum_logprob`
/// and the per-call PRNG key (advanced by splitting, mirroring mlx's keyed
/// RNG). Only two scalars per step leave the device — the chosen token id and
/// its log-probability — never the `n_vocab` logits row.
struct GreedyDecoder {
  temperature: f32,
  eot: u32,
  /// Accumulated `sum_logprobs` (`decoding.py:316-318`).
  sum_logprob: f64,
  /// The PRNG key, advanced per categorical draw (`temperature > 0`).
  key: Array,
}

impl GreedyDecoder {
  fn new(temperature: f32, eot: u32, seed: u64) -> Result<Self> {
    Ok(Self {
      temperature,
      eot,
      sum_logprob: 0.0,
      key: ops::random::key(seed)?,
    })
  }

  /// Select the next token from the `(n_vocab,)` logits row and update
  /// `sum_logprob` — `GreedyDecoder.update` (`decoding.py:307-325`),
  /// single-sequence, computed on device.
  ///
  /// - `logits`: the (already logit-filtered) `(n_vocab,)` row on device.
  /// - `last_token`: the previously-emitted token (the reference's
  ///   `tokens[:, -1]` — used to gate the logprob accumulation + the eot
  ///   "stick" behavior).
  ///
  /// Returns `(next_token, completed)`.
  fn update(&mut self, logits: &Array, last_token: u32) -> Result<(u32, bool)> {
    // Next token: `argmax` (`temperature == 0`) or a categorical draw, both on
    // device. argmax along the last axis ties to the lowest index, matching the
    // CPU path and mlx.
    let next = if self.temperature == 0.0 {
      let mut idx = ops::misc::argmax(logits, Some(-1), false)?;
      idx.item::<u32>()?
    } else {
      self.categorical(logits)?
    };

    // `logprobs = logits - logsumexp(logits)`; accumulate the chosen token's
    // logprob unless the previous token was eot (`decoding.py:315-318`). The
    // logsumexp + the chosen-logit gather run on device; only the resulting
    // scalar logprob is read back.
    if last_token != self.eot {
      let chosen_logprob = self.chosen_logprob(logits, next)?;
      self.sum_logprob += chosen_logprob;
    }

    // Once eot has been emitted, it sticks (`decoding.py:320-321`).
    let next = if last_token == self.eot {
      self.eot
    } else {
      next
    };

    let completed = next == self.eot;
    Ok((next, completed))
  }

  /// `logprobs[next] = logits[next] - logsumexp(logits)` — the chosen token's
  /// log-probability, computed on device, read back as one scalar. Returns
  /// `-inf` if `next` is out of range (the CPU path's `get` fallback).
  fn chosen_logprob(&self, logits: &Array, next: u32) -> Result<f64> {
    let v = logits.shape().first().copied().unwrap_or(0);
    if (next as usize) >= v {
      return Ok(f64::NEG_INFINITY);
    }
    let next_i = i32::try_from(next).map_err(|_| dim_overflow("next token"))?;
    let chosen = ops::indexing::slice(logits, &[next_i], &[next_i + 1], &[1])?;
    let lse = ops::reduction::logsumexp(logits, true)?; // rank-1 (1,)
    let mut logprob = ops::arithmetic::subtract(&chosen, &lse)?;
    Ok(logprob.item::<f32>()? as f64)
  }

  /// A temperature-scaled categorical draw, advancing the PRNG key
  /// (`decoding.py:297-299`/`:312-313`: `random.categorical(logits / temp)`).
  /// `logits` is the `(n_vocab,)` row already on device.
  fn categorical(&mut self, logits: &Array) -> Result<u32> {
    // Advance the key (mlx's keyed RNG: split, use one half, keep the other).
    let (next_key, sub) = ops::random::split(&self.key)?;
    self.key = next_key;
    let mut sampled = crate::lm::sample::categorical_sampling(logits, self.temperature, &sub)?;
    let token: u32 = sampled.item::<u32>()?;
    Ok(token)
  }
}

// ───────────────────────── decoding task ──────────────────────────────────

/// One 30-second-segment decode — `DecodingTask` (`decoding.py:445-723`),
/// single-utterance greedy.
///
/// Holds the resolved sot/initial-token sequence, the logit filters, and the
/// greedy decoder; [`DecodingTask::run`] threads them against the model's
/// `decode_tokens` to produce a [`DecodingResult`].
pub struct DecodingTask<'a> {
  model: &'a WhisperModel,
  tokenizer: &'a HFTokenizerWrapper<'a>,
  options: DecodingOptions,
  /// `n_text_ctx` — the decode context ceiling.
  n_ctx: usize,
  /// Maximum tokens to sample (`sample_len`, `decoding.py:461`). `0` emits no
  /// sampled tokens (the prefill / no_speech_prob still run); the main loop
  /// honors that cap rather than the reference's unconditional first step.
  sample_len: usize,
  /// The full initial token prefix: `(prompt) + sot_sequence + (prefix)`
  /// (`decoding.py:525-551`).
  initial_tokens: Vec<u32>,
  /// `len(initial_tokens)` — the index of the first sampled token.
  sample_begin: usize,
  /// `initial_tokens.index(sot)` (`decoding.py:469`).
  sot_index: usize,
  /// The logit filters, applied in order to each step's logits row.
  logit_filters: Vec<Box<dyn LogitFilter + 'a>>,
}

impl<'a> DecodingTask<'a> {
  /// Build a decode task from the model, the (already language/task-resolved)
  /// tokenizer wrapper, and the options — `DecodingTask.__init__`
  /// (`decoding.py:451-508`).
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if the resolved sot sequence is absent
  ///   (`sot` not in `initial_tokens`), or `max_initial_timestamp` rounds out
  ///   of range;
  /// - propagates the `SuppressBlank` encode error.
  pub fn new(
    model: &'a WhisperModel,
    tokenizer: &'a HFTokenizerWrapper<'a>,
    options: DecodingOptions,
  ) -> Result<Self> {
    let dims = model.dims();
    let n_ctx = dims.n_text_ctx();
    let sample_len = options.sample_len.unwrap_or(n_ctx / 2);

    // sot sequence (+ no_timestamps when `without_timestamps`).
    let sot_sequence = if options.without_timestamps {
      tokenizer.sot_sequence_including_notimestamps()
    } else {
      tokenizer.sot_sequence()
    };

    let initial_tokens =
      build_initial_tokens(tokenizer, &sot_sequence, &options, n_ctx, sample_len);
    // Fail-fast on a caller-supplied `prompt` / `prefix` id `>= n_vocab` BEFORE
    // any forward: those ids flow through `initial_tokens` into the prefill
    // `decode_tokens` and then the decoder token-embedding gather, where an
    // out-of-range id would index out of bounds. `decode_tokens` re-checks the
    // same value range at the gather chokepoint; validating here surfaces the
    // typed error at task construction (before the encoder runs) and names the
    // initial-token boundary. Shares the model's `validate_token_ids` helper.
    model.validate_token_ids("DecodingTask: initial token", &initial_tokens)?;
    let sample_begin = initial_tokens.len();
    let sot_index = initial_tokens
      .iter()
      .position(|&t| t == tokenizer.sot())
      .ok_or_else(|| {
        Error::OutOfRange(OutOfRangePayload::new(
          "DecodingTask: sot token",
          "must appear in the initial token sequence",
          format_smolstr!("sot={}", tokenizer.sot()),
        ))
      })?;

    // Logit filters (`decoding.py:484-508`).
    let n_vocab = dims.n_vocab();
    let mut logit_filters: Vec<Box<dyn LogitFilter + 'a>> = Vec::new();
    if options.suppress_blank {
      logit_filters.push(Box::new(SuppressBlank::new(
        tokenizer,
        sample_begin,
        n_vocab,
      )?));
    }
    let suppress_ids = get_suppress_tokens(tokenizer, &options.suppress_tokens)?;
    if !suppress_ids.is_empty() {
      logit_filters.push(Box::new(SuppressTokens::new(&suppress_ids, n_vocab)?));
    }
    if !options.without_timestamps {
      // precision = CHUNK_LENGTH / n_audio_ctx (≈ 0.02 s); the max initial
      // timestamp is rounded to that grid (`decoding.py:497-503`).
      let max_initial_timestamp_index = match options.max_initial_timestamp {
        Some(ts) => {
          let precision = CHUNK_LENGTH as f64 / dims.n_audio_ctx() as f64;
          let idx = (ts as f64 / precision).round();
          if !(0.0..=usize::MAX as f64).contains(&idx) {
            return Err(Error::OutOfRange(OutOfRangePayload::new(
              "DecodingTask: max_initial_timestamp_index",
              "must round to a non-negative in-range index",
              format_smolstr!("{idx}"),
            )));
          }
          Some(idx as usize)
        }
        None => None,
      };
      logit_filters.push(Box::new(ApplyTimestampRules::new(
        tokenizer,
        sample_begin,
        max_initial_timestamp_index,
        n_vocab,
      )));
    }

    Ok(Self {
      model,
      tokenizer,
      options,
      n_ctx,
      sample_len,
      initial_tokens,
      sample_begin,
      sot_index,
      logit_filters,
    })
  }

  /// Decode one mel segment (`(N_FRAMES, n_mels)` or pre-encoded
  /// `(n_audio_ctx, n_audio_state)`) — `DecodingTask.run` + `_main_loop`
  /// (`decoding.py:588-723`), single-utterance greedy.
  ///
  /// # Errors
  /// Propagates the encoder / decoder / sampling op errors.
  pub fn run(&self, mel: &Array, language: &str) -> Result<DecodingResult> {
    // Encoder forward (or pass-through if already-encoded features).
    let audio_features = self.audio_features(mel)?;

    // Run the main greedy loop.
    let (tokens, sum_logprob, no_speech_prob) = self.main_loop(&audio_features)?;

    // `tokens[sample_begin:]`, truncated at the first eot (`decoding.py:679`,
    // `:686`).
    let sampled: Vec<u32> = {
      let tail = tokens.get(self.sample_begin..).unwrap_or(&[]);
      let end = tail
        .iter()
        .position(|&t| t == self.tokenizer.eot())
        .unwrap_or(tail.len());
      tail[..end].to_vec()
    };

    let text = self.tokenizer.decode(&sampled, false)?.trim().to_string();
    // `avg_logprob = sum_logprob / (len + 1)` (`decoding.py:694-696`).
    let avg_logprob = sum_logprob / (sampled.len() as f64 + 1.0);

    Ok(DecodingResult {
      language: language.to_string(),
      compression_ratio: compression_ratio(&text),
      tokens: sampled,
      text,
      avg_logprob,
      no_speech_prob,
      temperature: self.options.temperature,
    })
  }

  /// Encode the mel (or pass an already-encoded feature tensor straight
  /// through) — `_get_audio_features` (`decoding.py:553-571`). mlxrs decodes
  /// in the model dtype rather than forcing fp16, so the dtype cast is
  /// dropped; the encoder is skipped if `mel` is already the encoder-output
  /// shape `(n_audio_ctx, n_audio_state)`.
  fn audio_features(&self, mel: &Array) -> Result<Array> {
    encode_once(self.model, mel)
  }

  /// The autoregressive greedy loop — `_main_loop` (`decoding.py:588-632`),
  /// single-sequence. Returns `(all_tokens, sum_logprob, no_speech_prob)`.
  fn main_loop(&self, audio_features: &Array) -> Result<(Vec<u32>, f64, f64)> {
    let mut decoder = GreedyDecoder::new(
      self.options.temperature,
      self.tokenizer.eot(),
      // mlx's GreedyDecoder draws from the global RNG; mlxrs threads an
      // explicit key. A fixed seed keeps the temperature draw reproducible
      // across runs (matters only for `temperature > 0`).
      0,
    )?;

    let mut tokens: Vec<u32> = self.initial_tokens.clone();

    // First forward: the whole initial prefix (`decoding.py:608`). This runs
    // unconditionally because the no_speech probability is read from the sot
    // position of THIS forward (`decoding.py:611-613`) — it is a property of
    // the prefill, not of any sampled token. `audio_features` is constant across
    // this whole trajectory, so the cross-attention K/V the first forward caches
    // is the K/V of these features for every subsequent step.
    let (logits3d, new_cache) = self.model.decode_tokens(&tokens, audio_features, None)?;
    let mut cache = Some(new_cache);

    let no_speech_prob = self.no_speech_prob(&logits3d)?;

    // `sample_len == 0` caps the sampled-token count at zero: honor it by
    // returning after the prefill / no_speech_prob read, emitting no token (the
    // reference's `range(1, sample_len)` loop runs the first `_step`
    // unconditionally, so a literal port would still emit one — this matches the
    // crate's `max_new_tokens == 0 ⇒ no tokens` semantics in the stt driver).
    if self.sample_len == 0 {
      drop(cache);
      return Ok((tokens, decoder.sum_logprob, no_speech_prob));
    }

    // First sampled token (`sample_len >= 1`): the last-position logits row,
    // kept on device through the filters + the greedy update.
    let last_row = last_position_row(&logits3d)?;
    let last_row = self.apply_filters(&last_row, &tokens)?;
    let last_token = *tokens.last().unwrap_or(&self.tokenizer.sot());
    let (mut next, mut completed) = decoder.update(&last_row, last_token)?;
    tokens.push(next);

    // Subsequent single-token steps (`decoding.py:618-631`).
    for _ in 1..self.sample_len {
      if completed || tokens.len() > self.n_ctx {
        break;
      }
      let cache_ref = cache.as_ref();
      let (logits3d, new_cache) = self
        .model
        .decode_tokens(&[next], audio_features, cache_ref)?;
      cache = Some(new_cache);

      let row = last_position_row(&logits3d)?;
      let row = self.apply_filters(&row, &tokens)?;
      let last_token = *tokens.last().unwrap_or(&self.tokenizer.eot());
      let (n, c) = decoder.update(&row, last_token)?;
      next = n;
      completed = c;
      tokens.push(next);
    }

    // `cache` is dropped here — the decode trajectory is finished.
    drop(cache);
    Ok((tokens, decoder.sum_logprob, no_speech_prob))
  }

  /// Apply every logit filter, in order, to the `(n_vocab,)` row on device,
  /// returning the masked row.
  ///
  /// # Errors
  /// Propagates the per-filter device op errors.
  fn apply_filters(&self, row: &Array, tokens: &[u32]) -> Result<Array> {
    let mut row = row.try_clone()?;
    for filter in &self.logit_filters {
      row = filter.apply(&row, tokens)?;
    }
    Ok(row)
  }

  /// `P(<|nospeech|>)` at the sot position from the first forward's
  /// `(1, T, n_vocab)` logits (`decoding.py:611-613`): softmax over the
  /// `sot_index` row, read the `no_speech` column. `NaN` if the model has no
  /// no_speech token in range.
  fn no_speech_prob(&self, logits3d: &Array) -> Result<f64> {
    let shape = logits3d.shape();
    if shape.len() != 3 {
      return Ok(f64::NAN);
    }
    let (t, v) = (shape[1], shape[2]);
    if self.sot_index >= t || self.tokenizer.no_speech() as usize >= v {
      return Ok(f64::NAN);
    }
    let si = i32::try_from(self.sot_index).map_err(|_| dim_overflow("sot_index"))?;
    let vi = i32::try_from(v).map_err(|_| dim_overflow("n_vocab"))?;
    // logits3d[0, sot_index, :] → (n_vocab,).
    let row =
      ops::indexing::slice(logits3d, &[0, si, 0], &[1, si + 1, vi], &[1, 1, 1])?.reshape(&[vi])?;
    let probs = ops::misc::softmax_axis(&row, -1, true)?;
    let ns = i32::try_from(self.tokenizer.no_speech()).map_err(|_| dim_overflow("no_speech"))?;
    let mut cell = ops::indexing::slice(&probs, &[ns], &[ns + 1], &[1])?;
    let p = cell.item::<f32>()?;
    Ok(p as f64)
  }

  /// Run the AlignAtt-policy decode of one streaming chunk's already-encoded
  /// `enc` over the `content_frames` real encoder frames available — the
  /// attention-guided streaming step for [`super::streaming`].
  ///
  /// This reuses the task's resolved sot/prefix prompt, its logit filters, and
  /// its [`GreedyDecoder`] (exactly the machinery [`Self::run`] /
  /// [`Self::main_loop`] use), but drives the loop with the AlignAtt commit/wait
  /// policy (Papi, Negri, Turchi 2023; the Simul-Whisper reference
  /// `backspacetg/simul_whisper`, `transcriber/simul_whisper.py`) instead of
  /// decoding the whole 30-second window:
  ///
  /// 1. prefill the initial tokens (sot sequence + any committed-prefix prompt),
  ///    reading the no-speech probability off the sot position of that forward;
  /// 2. then, token by token: filter the last-position logits, sample the next
  ///    token, and compute the token's most-attended audio frame from the
  ///    decode's cross-attention restricted to the model's alignment heads
  ///    ([`timing::alignatt_frame_attention`]);
  /// 3. **commit** the token if its argmax frame is more than `frame_threshold`
  ///    frames from the end of the available audio (`content_frames -
  ///    most_attended_frame > frame_threshold`); otherwise the token's evidence
  ///    sits at the audio boundary and may change with more audio, so it is
  ///    **dropped** and the decode stops (the reference's `if content_mel_len -
  ///    most_attened_frame <= frame_threshold: current_tokens =
  ///    current_tokens[:, :-1]; break`).
  ///
  /// On the final chunk (`is_last`) the reference relaxes the threshold to a
  /// fixed value (`4` frames upstream) and runs the decode to eot; the caller
  /// passes the effective `frame_threshold` (the streaming layer substitutes the
  /// last-chunk value), and `is_last` lets an eot-completed decode finish the
  /// utterance (there is no more audio, so a token at the boundary is still
  /// committed rather than waited on).
  ///
  /// Returns the [`AlignedDecode`]: the committed sampled tokens (between the sot
  /// sequence and the AlignAtt stop / eot), the per-committed-token argmax
  /// frames, the no-speech probability, and whether the decode hit eot.
  ///
  /// `frame_threshold` is the AlignAtt `f` (in encoder frames; one frame ≈ 0.02
  /// s) ALREADY resolved for this chunk (the caller passes the last-chunk value
  /// when `is_last`). `content_frames` is the real encoder-frame count
  /// (`available_mel_frames / 2`).
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `content_frames` is `0`, the alignment heads are
  ///   empty / mis-indexed, or a dimension overflows `i32`;
  /// - propagates the encoder-states validation, decoder forward, filter, and
  ///   sampler op errors.
  pub(crate) fn decode_aligned(
    &self,
    enc: &Array,
    content_frames: usize,
    frame_threshold: usize,
    is_last: bool,
  ) -> Result<AlignedDecode> {
    use crate::audio::stt::models::whisper::model::WhisperDecodeCache;

    let mut decoder = GreedyDecoder::new(self.options.temperature, self.tokenizer.eot(), 0)?;
    let mut tokens: Vec<u32> = self.initial_tokens.clone();
    let mut cache = WhisperDecodeCache::new();

    // First forward: the whole initial prefix (the sot sequence plus any forced
    // continuation prefix the streaming layer set, e.g. the committed in-window
    // tokens). The no-speech probability is a property of this prefill's sot
    // position, matching `main_loop`.
    let (logits3d, _cross_qk) = self
      .model
      .decode_step_with_cross_qk(&mut cache, enc, &tokens)?;
    let no_speech_prob = self.no_speech_prob(&logits3d)?;

    // The committed argmax frame per sampled token (parallel to the sampled tail
    // `tokens[sample_begin..]`).
    let mut committed_frames: Vec<usize> = Vec::new();
    let mut completed = false;

    // Sampled steps: filter → sample → AlignAtt commit/wait. Bounded by the
    // decode context and the `sample_len` cap. The guard is `< n_ctx` (not
    // `<=`): each iteration provisionally pushes ONE token before forwarding it,
    // so entering at `len == n_ctx` would forward an `n_ctx + 1` prefix past the
    // decoder's positional table. Stopping at `len == n_ctx` keeps every
    // forwarded prefix `<= n_ctx` (a token at the context boundary is simply not
    // emitted — the streaming caller slides the window for more).
    let mut logits3d = logits3d;
    let mut step = 0usize;
    while !completed && step < self.sample_len && tokens.len() < self.n_ctx {
      // Filter + sample the next token off the last-position row.
      let row = last_position_row(&logits3d)?;
      let row = self.apply_filters(&row, &tokens)?;
      let last_token = *tokens.last().unwrap_or(&self.tokenizer.sot());
      let (next, done) = decoder.update(&row, last_token)?;

      if done {
        // eot: the decode is complete. The eot itself carries no audio frame, so
        // it is never committed (the reference truncates the eot token); whether
        // the upstream utterance is finished is reported via `completed`.
        completed = true;
        break;
      }

      // Provisionally extend, forward the new token to obtain ITS cross-
      // attention, then apply the AlignAtt policy to decide commit vs wait. The
      // `< n_ctx` loop guard keeps this pushed prefix `<= n_ctx`.
      tokens.push(next);
      let (next_logits, cross_qk) = self
        .model
        .decode_step_with_cross_qk(&mut cache, enc, &tokens)?;
      let frames = timing::alignatt_frame_attention(self.model, &cross_qk, content_frames)?;
      // The just-sampled token is the LAST decoded position; its argmax frame is
      // the last entry of the per-token frame vector.
      let most_attended = frames.last().copied().unwrap_or(0);

      // AlignAtt commit/wait: if the token's evidence is within `frame_threshold`
      // frames of the audio end, it may change with more audio → drop it and
      // wait. On the final chunk there is no more audio, so the token commits
      // regardless (the reference still emits every decoded token on the last
      // chunk).
      if !alignatt_should_commit(content_frames, most_attended, frame_threshold, is_last) {
        // Wait: revert the provisional token and stop committing for this chunk
        // (the reference's `current_tokens = current_tokens[:, :-1]; break`).
        tokens.pop();
        break;
      }

      // Commit the token and its frame; continue decoding.
      committed_frames.push(most_attended);
      logits3d = next_logits;
      step += 1;
    }

    // The committed sampled tokens: `tokens[sample_begin..]` truncated at the
    // first eot (defensive — a committed eot is never pushed above).
    let sampled: Vec<u32> = {
      let tail = tokens.get(self.sample_begin..).unwrap_or(&[]);
      let end = tail
        .iter()
        .position(|&t| t == self.tokenizer.eot())
        .unwrap_or(tail.len());
      tail[..end].to_vec()
    };
    // `committed_frames` is parallel to the committed sampled tokens; clamp to
    // the sampled length in case the loop committed past a defensive truncation.
    committed_frames.truncate(sampled.len());

    Ok(AlignedDecode {
      tokens: sampled,
      frames: committed_frames,
      no_speech_prob,
      completed,
    })
  }
}

/// The AlignAtt commit/wait decision for ONE just-decoded token — the policy's
/// core inequality, factored out as a pure function so it can be tested against
/// an independent oracle on synthetic frames.
///
/// Returns `true` (COMMIT) when the token's most-attended audio frame is safely
/// behind the live audio edge — `content_frames - most_attended_frame >
/// frame_threshold`. Returns `false` (WAIT) when the frame is within
/// `frame_threshold` of the end, mirroring the Simul-Whisper reference's `if
/// content_mel_len - most_attened_frame <= frame_threshold: ... break`
/// (`backspacetg/simul_whisper`, `transcriber/simul_whisper.py`).
///
/// On the FINAL chunk (`is_last`) there is no more audio to wait for, so the
/// token always commits regardless of the threshold (the reference still emits
/// every decoded token on the last chunk). `content_frames.saturating_sub` keeps
/// the subtraction non-negative if a frame somehow exceeds `content_frames` (a
/// degenerate-attention argmax past the content slice); such a frame is at/over
/// the edge, so it correctly resolves to WAIT on a non-final chunk.
#[inline]
pub(crate) fn alignatt_should_commit(
  content_frames: usize,
  most_attended_frame: usize,
  frame_threshold: usize,
  is_last: bool,
) -> bool {
  if is_last {
    return true;
  }
  content_frames.saturating_sub(most_attended_frame) > frame_threshold
}

/// The result of one AlignAtt streaming decode ([`DecodingTask::decode_aligned`]).
///
/// Carries the tokens the AlignAtt policy COMMITTED for this chunk (the
/// just-decoded ones whose most-attended audio frame is safely behind the audio
/// boundary), each token's argmax encoder frame (for timing), the prefill's
/// no-speech probability, and whether the decode reached eot.
#[derive(Debug, Clone)]
pub(crate) struct AlignedDecode {
  /// The committed sampled token ids (sot sequence + prompt stripped, truncated
  /// before any eot).
  pub tokens: Vec<u32>,
  /// The most-attended encoder frame per committed token (parallel to
  /// [`Self::tokens`]).
  pub frames: Vec<usize>,
  /// `P(<|nospeech|>)` read off the prefill's sot position (`NaN` if the model
  /// has no no-speech token).
  pub no_speech_prob: f64,
  /// Whether the decode reached eot (the chunk's utterance is finished).
  pub completed: bool,
}

/// Build the `(prompt) + sot_sequence + (prefix)` initial token prefix —
/// `_get_initial_tokens` (`decoding.py:525-551`).
fn build_initial_tokens(
  tokenizer: &HFTokenizerWrapper<'_>,
  sot_sequence: &[u32],
  options: &DecodingOptions,
  n_ctx: usize,
  sample_len: usize,
) -> Vec<u32> {
  let mut tokens: Vec<u32> = sot_sequence.to_vec();

  // prefix: forced after the sot sequence, tail-truncated to leave room
  // (`decoding.py:528-537`).
  if !options.prefix.is_empty() {
    let max_prefix_len = (n_ctx / 2).saturating_sub(sample_len);
    let prefix = &options.prefix;
    let start = prefix.len().saturating_sub(max_prefix_len);
    tokens.extend_from_slice(&prefix[start..]);
  }

  // prompt: prepended (with sot_prev) before the sot sequence, tail-truncated
  // (`decoding.py:539-549`).
  if !options.prompt.is_empty() {
    let keep = (n_ctx / 2).saturating_sub(1);
    let prompt = &options.prompt;
    let start = prompt.len().saturating_sub(keep);
    let mut prefixed = Vec::with_capacity(1 + (prompt.len() - start) + tokens.len());
    prefixed.push(tokenizer.sot_prev());
    prefixed.extend_from_slice(&prompt[start..]);
    prefixed.extend_from_slice(&tokens);
    tokens = prefixed;
  }

  tokens
}

/// Extract the last-position logits row `(n_vocab,)` from a `(1, T, n_vocab)`
/// decoder output as a device [`Array`] (the reference's `logits[:, -1]`).
///
/// The row stays on device — it is **not** copied to the host. The greedy
/// argmax / categorical draw and the logit filters all operate on it on
/// device, so the per-step `n_vocab` round-trip the old `to_vec` path incurred
/// is gone (only the final chosen token id + its log-probability are read
/// back, one scalar each). Cast to `F32` to match `Inference.logits`'s
/// `.astype(mx.float32)`.
fn last_position_row(logits3d: &Array) -> Result<Array> {
  let shape = logits3d.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(crate::error::RankMismatchPayload::new(
      "DecodingTask: decoder logits must be rank 3 (1, T, V)",
      shape.len() as u32,
      shape,
    )));
  }
  let (t, v) = (shape[1], shape[2]);
  let last = i32::try_from(t.saturating_sub(1)).map_err(|_| dim_overflow("T"))?;
  let vi = i32::try_from(v).map_err(|_| dim_overflow("n_vocab"))?;
  let row = ops::indexing::slice(logits3d, &[0, last, 0], &[1, last + 1, vi], &[1, 1, 1])?;
  // → (n_vocab,), F32, contiguous — kept on device.
  let row = row.reshape(&[vi])?;
  let row = ops::misc::astype(&row, Dtype::F32)?;
  ops::shape::contiguous(&row, false)
}

/// A dimension exceeding `i32::MAX`.
fn dim_overflow(which: &'static str) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "DecodingTask: dimension",
    "must fit in i32",
    format_smolstr!("{which} exceeds i32::MAX"),
  ))
}

// ───────────────────────── detect language ────────────────────────────────

/// Detect the spoken language — `detect_language` (`decoding.py:20-77`),
/// single-utterance.
///
/// Runs a single `sot`-token forward (fresh cache, no interference with the
/// main decode), masks every non-language logit, and returns the
/// `(best_code, probs)` over the checkpoint's language tokens. Requires a
/// multilingual checkpoint with language tokens in its sot sequence.
///
/// `audio_features` is the encoder output `(1, n_audio_ctx, n_audio_state)`
/// (or a mel the encoder is run on).
///
/// # Errors
/// - [`Error::InvariantViolation`] if the checkpoint has no language tokens, or
///   none of its language tokens are in the model's vocabulary;
/// - [`Error::OutOfRange`] if a language token id is `>= n_vocab` (a corrupt /
///   mismatched tokenizer-model pair — masking it would leave an all-`-inf` row
///   whose `logit - logsumexp` is `NaN`, silently selecting the first code);
/// - propagates the encoder / decoder / softmax op errors.
pub fn detect_language<'a>(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'a>,
  audio_features: &Array,
) -> Result<(&'a str, Vec<(&'a str, f64)>)> {
  // `(code, token_id)` pairs in ONE aligned pass — never the two separate
  // (filtered tokens / unfiltered codes) vectors a positional zip would drift.
  let candidates = tokenizer.all_language_candidates();
  if candidates.is_empty() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "detect_language",
      "checkpoint has no language tokens — cannot perform language id",
    )));
  }

  // Single `sot` forward, fresh cache.
  let (logits3d, _cache) = model.decode_tokens(&[tokenizer.sot()], audio_features, None)?;
  // logits[:, 0] → the only (first) position's row. Language detection runs
  // once (not per decode step), so reading this row to the host here is fine —
  // it is the masking + softmax over a small language-token set, not the hot
  // per-step path the device filters replace.
  let mut row = last_position_row(&logits3d)?.to_vec::<f32>()?;
  let n_vocab = row.len();

  // Reject any language token id outside the model's actual vocabulary BEFORE
  // building the mask. A mismatched / corrupt tokenizer-model pair can carry
  // language ids `>= n_vocab`; those ids index nothing in the logits row, so
  // masking would leave the row all-`-inf` and `logit - logsumexp` would be
  // `-inf - -inf = NaN`, silently selecting the first candidate. Fail loudly
  // (the stricter faithful choice) naming the offending id.
  if let Some(&(code, id)) = candidates.iter().find(|&&(_, id)| id as usize >= n_vocab) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "detect_language: language token id",
      "must be < n_vocab (the decoder logits width)",
      format_smolstr!("code=<|{code}|>, id={id}, n_vocab={n_vocab}"),
    )));
  }
  // Every candidate id is now `< n_vocab`.

  // Suppress every non-language token (`decoding.py:59-61`).
  let lang_set: std::collections::HashSet<u32> = candidates.iter().map(|(_, id)| *id).collect();
  for (i, slot) in row.iter_mut().enumerate() {
    if !lang_set.contains(&(i as u32)) {
      *slot = f32::NEG_INFINITY;
    }
  }

  // The masked row keeps at least one finite language logit (every candidate id
  // is in range and was left unmasked), so `logsumexp` is finite. Guard it
  // anyway: an all-`-inf` row (e.g. a forward producing `-inf` logits) would
  // make every `logit - lse` a `NaN`, which would pick a bogus best code.
  let lse = logsumexp_slice(&row);
  if !lse.is_finite() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "detect_language",
      "masked language logits are all non-finite — cannot form a probability \
       distribution (degenerate forward or empty language set)",
    )));
  }

  // Softmax over the masked row, then read each language token's probability —
  // each `(code, id)` already paired, so the reported code always matches the
  // token whose logit it carries. Every id is in range here, so the indexing
  // never falls back to `-inf`.
  let mut probs: Vec<(&'a str, f64)> = Vec::with_capacity(candidates.len());
  let mut best: Option<(&'a str, f64)> = None;
  for &(code, id) in &candidates {
    let p = (row[id as usize] as f64 - lse).exp();
    probs.push((code, p));
    if best.is_none_or(|(_, bp)| p > bp) {
      best = Some((code, p));
    }
  }
  // `best` is `Some` (candidates is non-empty and checked above); the `"en"`
  // fallback is unreachable but kept as a total-function default.
  let best_code = best.map(|(code, _)| code).unwrap_or("en");
  Ok((best_code, probs))
}

// ───────────────────────── decode + transcribe ────────────────────────────

/// Resolve the decode language for one mel — the reference's
/// `_detect_language` / `DecodingTask._detect_language` language handling
/// (`decoding.py:573-586`, `whisper.py:767-785`).
///
/// When `requested` is `Some`, that code is used verbatim. When it is `None`
/// the language is detected: an English-only checkpoint trivially yields
/// `"en"`; a multilingual one runs [`detect_language`] on `audio_features`
/// (the already-encoded `(1, n_audio_ctx, n_audio_state)` states, so the
/// encoder is not re-run) and returns the most probable language token's code.
///
/// # Errors
/// Propagates [`detect_language`]'s op errors.
fn resolve_language(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  audio_features: &Array,
  requested: Option<&str>,
) -> Result<String> {
  if let Some(lang) = requested {
    return Ok(lang.to_string());
  }
  if !tokenizer.is_multilingual() {
    return Ok("en".to_string());
  }
  let (code, _probs) = detect_language(model, tokenizer, audio_features)?;
  Ok(code.to_string())
}

/// Decode one 30-second mel segment — `decode` (`decoding.py:726-758`),
/// single-utterance.
///
/// Owns language resolution: the mel is encoded once, then if
/// `options.language` is `None` (and the checkpoint is multilingual) the
/// language is detected from those features via [`detect_language`] and the
/// tokenizer's start-of-transcript language token is rebuilt to match
/// ([`HFTokenizerWrapper::with_language`]). The decode then runs on the
/// already-encoded features (no second encoder pass), and the resolved /
/// detected language is reported on the [`DecodingResult`].
///
/// # Errors
/// Propagates the encoder forward, [`detect_language`], [`DecodingTask::new`],
/// and [`DecodingTask::run`] errors.
pub fn decode(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  mel: &Array,
  options: DecodingOptions,
) -> Result<DecodingResult> {
  // Encode once; reuse for both detection and the decode.
  let audio_features = encode_once(model, mel)?;
  let language = resolve_language(
    model,
    tokenizer,
    &audio_features,
    options.language.as_deref(),
  )?;
  // Rebuild the tokenizer SOT language token from the resolved language.
  let resolved = tokenizer.with_language(&language);
  decode_resolved(model, &resolved, &audio_features, options, &language)
}

/// Decode one mel segment with an already-resolved language + rebuilt
/// tokenizer — the inner half of [`decode`] (and the per-temperature body of
/// [`decode_with_fallback`]), skipping language detection.
///
/// `audio_features` is encoded features (or a raw mel — [`DecodingTask::run`]
/// passes encoded features straight through); `language` is the resolved code
/// reported on the result.
///
/// # Errors
/// Propagates [`DecodingTask::new`] / [`DecodingTask::run`].
fn decode_resolved(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  audio_features: &Array,
  options: DecodingOptions,
  language: &str,
) -> Result<DecodingResult> {
  let task = DecodingTask::new(model, tokenizer, options)?;
  task.run(audio_features, language)
}

/// Encode a mel to encoder features `(1, n_audio_ctx, n_audio_state)`, or pass
/// an already-encoded feature tensor straight through — the encode-once step
/// shared by [`decode`] and language detection (`_get_audio_features`,
/// `decoding.py:553-571`).
///
/// # Errors
/// Propagates the encoder forward op errors.
fn encode_once(model: &WhisperModel, mel: &Array) -> Result<Array> {
  let dims = model.dims();
  let shape = mel.shape();
  let is_encoded = matches!(
    shape.as_slice(),
    [c, s] if *c == dims.n_audio_ctx() && *s == dims.n_audio_state()
  ) || matches!(
    shape.as_slice(),
    [1, c, s] if *c == dims.n_audio_ctx() && *s == dims.n_audio_state()
  );
  if is_encoded {
    if shape.len() == 2 {
      let c = i32::try_from(dims.n_audio_ctx()).map_err(|_| dim_overflow("n_audio_ctx"))?;
      let s = i32::try_from(dims.n_audio_state()).map_err(|_| dim_overflow("n_audio_state"))?;
      mel.reshape(&[1, c, s])
    } else {
      mel.try_clone()
    }
  } else {
    model.encode(mel)
  }
}

/// The temperature-fallback + quality-threshold decision for one segment —
/// the per-temperature loop of `decode_with_fallback` (`whisper.py:938-976`).
///
/// Given a [`DecodingResult`] and the three thresholds, decides whether to
/// retry at the next (higher) temperature: a too-repetitive
/// (`compression_ratio > compression_ratio_threshold`) or too-uncertain
/// (`avg_logprob < logprob_threshold`) result needs a fallback, **unless** it
/// is silence (`no_speech_prob > no_speech_threshold`), which forces accept.
/// Returns `true` if the result is acceptable (stop the fallback loop).
pub fn result_is_acceptable(
  result: &DecodingResult,
  compression_ratio_threshold: Option<f64>,
  logprob_threshold: Option<f64>,
  no_speech_threshold: Option<f64>,
) -> bool {
  let mut needs_fallback = false;
  if let Some(crt) = compression_ratio_threshold
    && result.compression_ratio > crt
  {
    needs_fallback = true; // too repetitive
  }
  if let Some(lpt) = logprob_threshold
    && result.avg_logprob < lpt
  {
    needs_fallback = true; // average logprob too low
  }
  if let Some(nst) = no_speech_threshold
    && result.no_speech_prob > nst
  {
    needs_fallback = false; // silence — accept
  }
  !needs_fallback
}

/// The default temperature schedule for the fallback
/// (`whisper.py:797`): `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`.
pub const DEFAULT_TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// The default compression-ratio threshold (`whisper.py:798`).
pub const DEFAULT_COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;
/// The default average-logprob threshold (`whisper.py:799`).
pub const DEFAULT_LOGPROB_THRESHOLD: f64 = -1.0;
/// The default no-speech probability threshold (`whisper.py:800`).
pub const DEFAULT_NO_SPEECH_THRESHOLD: f64 = 0.6;

/// Decode one segment with the temperature-fallback loop —
/// `decode_with_fallback` (`whisper.py:938-976`).
///
/// Tries each temperature in `temperatures` in order, accepting the first
/// result that passes [`result_is_acceptable`]; if every temperature fails,
/// the last result is returned. `language` is the already-resolved code and
/// `tokenizer` the already-rebuilt wrapper (the [`transcribe`] seek loop
/// resolves the language once, before the loop), so no per-temperature
/// language detection runs. The mel is encoded once and reused across the
/// temperature retries.
///
/// # Errors
/// Propagates the encoder forward / [`DecodingTask`] errors.
#[allow(clippy::too_many_arguments)]
pub fn decode_with_fallback(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  mel: &Array,
  base_options: &DecodingOptions,
  language: &str,
  temperatures: &[f32],
  compression_ratio_threshold: Option<f64>,
  logprob_threshold: Option<f64>,
  no_speech_threshold: Option<f64>,
) -> Result<DecodingResult> {
  // Encode the window once; every temperature retry decodes the same features.
  let audio_features = encode_once(model, mel)?;
  let mut last: Option<DecodingResult> = None;
  for &t in temperatures {
    let mut options = base_options.clone();
    options.temperature = t;
    let result = decode_resolved(model, tokenizer, &audio_features, options, language)?;
    let acceptable = result_is_acceptable(
      &result,
      compression_ratio_threshold,
      logprob_threshold,
      no_speech_threshold,
    );
    last = Some(result);
    if acceptable {
      break;
    }
  }
  // `temperatures` is never empty in practice ([`DEFAULT_TEMPERATURES`]); if a
  // caller passes an empty schedule, fall back to a single greedy decode.
  match last {
    Some(r) => Ok(r),
    None => decode_resolved(
      model,
      tokenizer,
      &audio_features,
      base_options.clone(),
      language,
    ),
  }
}

/// One transcribed segment of a longer mel — the per-window result of the
/// A per-word timing attached to a [`Segment`] when word timestamps are enabled
/// — the reference's `segment["words"]` entry (`whisper.py:272-279`).
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
  /// The word text (including any leading space / merged punctuation).
  pub word: String,
  /// The word start time in seconds (absolute, including the segment offset).
  pub start: f64,
  /// The word end time in seconds (absolute).
  pub end: f64,
  /// The mean per-token probability of the word's tokens.
  pub probability: f64,
}

/// seek loop (`whisper.py:996-1011` `new_segment`), trimmed to the fields
/// mlxrs surfaces. [`Self::words`] is populated only when word timestamps are
/// requested ([`TranscribeOptions::word_timestamps`]); it is empty on the
/// default (no-word-timestamp) path.
#[derive(Debug, Clone)]
pub struct Segment {
  /// Segment start time in seconds.
  pub start: f64,
  /// Segment end time in seconds.
  pub end: f64,
  /// The decoded text for this segment.
  pub text: String,
  /// The sampled token ids for this segment.
  pub tokens: Vec<u32>,
  /// The temperature this segment was decoded at.
  pub temperature: f32,
  /// Mean token log-probability for this segment.
  pub avg_logprob: f64,
  /// No-speech probability for this segment.
  pub no_speech_prob: f64,
  /// The compression ratio of this segment's text.
  pub compression_ratio: f64,
  /// The per-word timings — empty unless word timestamps were requested
  /// ([`TranscribeOptions::word_timestamps`]).
  pub words: Vec<Word>,
}

/// The whole-utterance transcription — the `generate` return shape
/// (`whisper.py:1290-1300`), trimmed to text + language + segments.
#[derive(Debug, Clone)]
pub struct TranscribeResult {
  /// The full concatenated text.
  pub text: String,
  /// The (detected or supplied) language code.
  pub language: String,
  /// The per-30-second-window segments.
  pub segments: Vec<Segment>,
}

/// Thresholds + temperature schedule for [`transcribe`] — the
/// quality-control knobs of `generate` (`whisper.py:797-800`). [`Default`] is
/// the reference's defaults.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
  /// The decode options applied per segment (task / language / suppress /
  /// timestamp knobs). The per-segment temperature is overridden by the
  /// fallback schedule.
  pub decode: DecodingOptions,
  /// The temperature fallback schedule ([`DEFAULT_TEMPERATURES`]).
  pub temperatures: Vec<f32>,
  /// Compression-ratio fallback threshold (`None` disables).
  pub compression_ratio_threshold: Option<f64>,
  /// Average-logprob fallback threshold (`None` disables).
  pub logprob_threshold: Option<f64>,
  /// No-speech threshold (`None` disables the silence skip).
  pub no_speech_threshold: Option<f64>,
  /// Attach per-word timestamps to each segment via the cross-attention DTW
  /// (`whisper.py:804`). `false` (the default) leaves the decode path
  /// byte-identical and adds **zero** cost; `true` runs the extra
  /// [`super::timing::add_word_timestamps`] alignment pass per window.
  pub word_timestamps: bool,
  /// Punctuation merged onto the **next** word when word timestamps are on
  /// ([`super::timing::PREPEND_PUNCTUATIONS`], `whisper.py:805`).
  pub prepend_punctuations: String,
  /// Punctuation merged onto the **previous** word
  /// ([`super::timing::APPEND_PUNCTUATIONS`], `whisper.py:806`).
  pub append_punctuations: String,
  /// When `Some` (and `word_timestamps` is on), skip silent periods longer than
  /// this threshold (seconds) around a detected hallucination
  /// (`whisper.py:808`, `:1171-1237`). `None` disables the heuristic.
  pub hallucination_silence_threshold: Option<f64>,
  /// Feed the previous window's decoded tokens as the next window's prompt
  /// (`whisper.py:801`). `true` (the reference default) conditions each window
  /// on the running transcript; `false` resets the prompt every window
  /// (`whisper.py:1279-1281`), trading cross-window consistency for resistance
  /// to repetition / timestamp-drift failure loops. A window decoded at
  /// `temperature > 0.5` also resets the prompt regardless of this flag.
  pub condition_on_previous_text: bool,
  /// Optional text prompting the **first** window (`whisper.py:802`, `:990-994`)
  /// — e.g. a custom vocabulary or proper nouns to bias the decode. The text is
  /// encoded (with a leading space) and seeded into the running token history
  /// before the seek loop; it conditions the first window's decode but is never
  /// emitted as transcript (`whisper.py:1299` strips it).
  pub initial_prompt: Option<String>,
  /// Seconds timestamps of the clips to process — `clip_timestamps`
  /// (`whisper.py:807`, `:915-931`), the reference's `List[float]` form.
  ///
  /// The list pairs up as `(start, end, start, end, …)`; each pair restricts the
  /// seek loop to `[start, end)` (converted to frames). An odd-length list
  /// leaves the final clip open-ended (its end defaults to the end of the
  /// audio). Empty (the default) reproduces the reference default `"0"` →
  /// `[0.0]` → one clip spanning the whole audio, so the seek loop is
  /// byte-identical to the unclipped path.
  pub clip_timestamps: Vec<f64>,
}

impl Default for TranscribeOptions {
  fn default() -> Self {
    Self {
      decode: DecodingOptions::default(),
      temperatures: DEFAULT_TEMPERATURES.to_vec(),
      compression_ratio_threshold: Some(DEFAULT_COMPRESSION_RATIO_THRESHOLD),
      logprob_threshold: Some(DEFAULT_LOGPROB_THRESHOLD),
      no_speech_threshold: Some(DEFAULT_NO_SPEECH_THRESHOLD),
      word_timestamps: false,
      prepend_punctuations: super::timing::PREPEND_PUNCTUATIONS.to_string(),
      append_punctuations: super::timing::APPEND_PUNCTUATIONS.to_string(),
      hallucination_silence_threshold: None,
      condition_on_previous_text: true,
      initial_prompt: None,
      clip_timestamps: Vec::new(),
    }
  }
}

/// The running prompt history threaded across the seek loop's windows —
/// `whisper.py`'s `all_tokens` / `prompt_reset_since` (`:986-994`, `:1033`,
/// `:1271-1281`).
///
/// `all_tokens` accumulates the optional `initial_prompt` tokens (seeded once,
/// before the loop) followed by every window's decoded tokens;
/// `prompt_reset_since` is the index from which the next window's prompt is
/// sliced. Conditioning a window resets to a different slice; resetting (the
/// `condition_on_previous_text == false` or `temperature > 0.5` case) moves
/// `prompt_reset_since` to the current tail so no prior text leaks forward.
///
/// This mirrors the reference exactly: the prompt fed to window *N* is
/// `all_tokens[prompt_reset_since:]` (`:1033`), the window's tokens are appended
/// after decoding (`:1271-1277`), and the reset condition is
/// `not condition_on_previous_text or result.temperature > 0.5` (`:1279-1281`).
struct PromptHistory {
  /// The initial-prompt tokens + every decoded window's tokens, in order.
  all_tokens: Vec<u32>,
  /// The index into [`Self::all_tokens`] the next window's prompt starts at.
  prompt_reset_since: usize,
  /// Whether to keep conditioning on previous text
  /// ([`TranscribeOptions::condition_on_previous_text`]).
  condition_on_previous_text: bool,
}

impl PromptHistory {
  /// Seed the history with the first window's prompt context.
  ///
  /// Precedence (documented on [`transcribe`]): `initial_prompt` wins when set —
  /// the reference's documented knob (`whisper.py:990-994`), encoded with a
  /// single leading space and a stripped body, then prepended to `all_tokens`.
  /// When `initial_prompt` is `None`, the lower-level `decode_prompt`
  /// ([`DecodingOptions::prompt`]) is used as the seed instead, so a caller who
  /// only set `decode.prompt` still conditions the first window (the reference
  /// overwrites `decode_options["prompt"]` at `:1033`, but mlxrs honors the
  /// caller's value as the initial context rather than silently dropping it).
  /// The first window's prompt is therefore exactly these seed tokens, since
  /// [`Self::window_prompt`] returns `all_tokens[prompt_reset_since..]` and
  /// `prompt_reset_since` starts at 0.
  ///
  /// The seed is bounded to the decoder-visible tail — the **last**
  /// `n_text_ctx / 2 - 1` tokens of the chosen prompt — because that is the only
  /// part [`Self::window_prompt`] ever returns (it always yields the last
  /// `n_text_ctx / 2 - 1` of `all_tokens[prompt_reset_since..]`). This is
  /// byte-identical to seeding the full prompt: for every window the decoder sees
  /// `last_K(seed ++ decoded_so_far)`, and `last_K(tail ++ decoded) ==
  /// last_K(full ++ decoded)` for any decoded suffix (taking the last `K` of a
  /// sequence ignores everything before its final `K` elements, and the tail
  /// already *is* those final `K` of the full seed). Across a reset
  /// `prompt_reset_since` advances past the entire seed in both cases, so the
  /// seeded prefix no longer participates at all — again identical. Storing only
  /// the tail avoids copying an unbounded caller prompt whose leading tokens the
  /// decoder never sees; the running history still accumulates every decoded
  /// window whole.
  ///
  /// # Errors
  /// Propagates the tokenizer encode error for the initial prompt.
  fn seed(
    tokenizer: &HFTokenizerWrapper<'_>,
    initial_prompt: Option<&str>,
    decode_prompt: &[u32],
    condition_on_previous_text: bool,
    n_text_ctx: usize,
  ) -> Result<Self> {
    // `window_prompt` only ever exposes the last `n_text_ctx / 2 - 1` tokens
    // (`build_initial_tokens`'s `keep`), so retain only that tail of the seed —
    // the leading prefix is unreachable by any window. Byte-identical to seeding
    // the full prompt (see the doc comment's `last_K` equivalence).
    let keep = (n_text_ctx / 2).saturating_sub(1);
    let all_tokens = match initial_prompt {
      Some(text) => {
        let encoded = tokenizer.encode(&format!(" {}", text.trim()))?;
        let start = encoded.len().saturating_sub(keep);
        encoded[start..].to_vec()
      }
      None => {
        let start = decode_prompt.len().saturating_sub(keep);
        decode_prompt[start..].to_vec()
      }
    };
    Ok(Self {
      all_tokens,
      prompt_reset_since: 0,
      condition_on_previous_text,
    })
  }

  /// The prompt for the next window — the decoder-visible tail of
  /// `all_tokens[prompt_reset_since:]` (`whisper.py:1033`).
  ///
  /// The reference assigns the whole `all_tokens[prompt_reset_since:]` slice to
  /// `decode_options["prompt"]`, but [`build_initial_tokens`]
  /// (`decoding.py:539-549`) then keeps only its **last** `n_text_ctx / 2 - 1`
  /// tokens (`keep = (n_ctx / 2) - 1`, tail-truncated). Returning that same tail
  /// here is byte-identical to returning the full slice: taking the last `keep`
  /// tokens and then having `build_initial_tokens` keep the last `keep` of those
  /// yields the identical token sequence, while avoiding cloning the leading
  /// tokens the decoder never sees. `n_text_ctx` is the model's text context
  /// length ([`ModelDimensions::n_text_ctx`]); the bound matches
  /// `build_initial_tokens` exactly so the decode result is unchanged.
  ///
  /// `all_tokens` itself still accumulates every window's tokens whole (the
  /// reset-window logic reads `prompt_reset_since` into the full vector); only
  /// the returned slice is bounded to the decoder window.
  fn window_prompt(&self, n_text_ctx: usize) -> &[u32] {
    let active = &self.all_tokens[self.prompt_reset_since.min(self.all_tokens.len())..];
    // `build_initial_tokens` keeps `(n_ctx / 2) - 1` prompt tokens (the tail).
    let keep = (n_text_ctx / 2).saturating_sub(1);
    let start = active.len().saturating_sub(keep);
    &active[start..]
  }

  /// Record a decoded window's tokens then apply the reset rule
  /// (`whisper.py:1271-1281`): the window's tokens are appended to `all_tokens`,
  /// and the prompt window is reset to the new tail when
  /// `!condition_on_previous_text` **or** the window's `temperature > 0.5` (a
  /// high-temperature fallback result must not condition the next window).
  fn push_window<'a, I>(&mut self, window_tokens: I, temperature: f32)
  where
    I: IntoIterator<Item = &'a u32>,
  {
    self.all_tokens.extend(window_tokens.into_iter().copied());
    if !self.condition_on_previous_text || temperature > 0.5 {
      self.prompt_reset_since = self.all_tokens.len();
    }
  }
}

/// Convert the `clip_timestamps` seconds list into `(start, end)` frame seek
/// clips — `whisper.py:915-931`.
///
/// Each timestamp is rounded to a frame index (`round(ts * FRAMES_PER_SECOND)`).
/// An empty list yields one clip spanning `[0, content_frames)` (the reference's
/// `"0"` default after its `if len == 0: append(0)` then odd-length fill); an
/// odd-length list leaves the final clip open-ended (its end is
/// `content_frames`); an even-length list clamps the final end to
/// `content_frames` (`whisper.py:928`). The flat list pairs as `(points[0],
/// points[1]), (points[2], points[3]), …` — `zip(points[::2], points[1::2])`.
///
/// Every `(start, end)` is additionally saturated into `[0, content_frames]` —
/// not just the final end — so a malformed earlier pair (a start or end beyond
/// the real audio length, as is reachable through user `clip_timestamps`) cannot
/// drive the seek loop past the content frames. A degenerate clip (zero-length
/// or inverted, `start >= end` after clamping) is dropped: it contributes no
/// windows, matching the reference's effective frame math (`content_frames -
/// seek` / `seek_clip_end - seek` would yield a non-positive `segment_size` for
/// such a clip, so it produces nothing). An empty `clip_timestamps` stays
/// byte-identical to the full-audio path: it yields the single `[0,
/// content_frames)` clip unchanged.
///
/// A non-finite clip timestamp (`NaN` / `±inf`) is rejected with
/// [`Error::OutOfRange`] before rounding, matching CPython's `round()`, which
/// raises `ValueError` on `round(nan)` and `OverflowError` on `round(inf)`
/// (`whisper.py:921`). Rust's `f64 as usize` cast would otherwise silently
/// saturate `NaN → 0` and `inf → usize::MAX`, turning a bogus value into a
/// degenerate (or full-audio) clip instead of an error. Finite values —
/// including negatives, which [`round_to_frames`] clamps to `0` — are unchanged.
///
/// The frame product `ts * FRAMES_PER_SECOND` is *also* required to be finite:
/// a finite-but-huge timestamp (e.g. `1e307`) overflows the product to `±inf`,
/// which [`round_to_frames`]'s `f64 as usize` cast would again saturate
/// (`+inf → usize::MAX`, `-inf → 0`), silently turning the bogus clip into a
/// full-audio or empty one. CPython instead raises `OverflowError` from
/// `round(finite * FRAMES_PER_SECOND)` the moment the product is infinite, so
/// the product is checked here (before rounding and before any integer cast) and
/// a non-finite product is rejected with the same typed error.
///
/// # Errors
/// [`Error::OutOfRange`] if any `clip_timestamps` value — or its
/// `FRAMES_PER_SECOND` frame product — is not finite.
fn compute_seek_clips(
  clip_timestamps: &[f64],
  content_frames: usize,
) -> Result<Vec<(usize, usize)>> {
  let mut seek_points: Vec<usize> = Vec::with_capacity(clip_timestamps.len());
  for &ts in clip_timestamps {
    // CPython's `round()` raises on `nan`/`inf`; the `f64 as usize` cast below
    // (inside `round_to_frames`) would instead silently coerce them, so reject
    // non-finite values here to preserve the reference's failure semantics.
    if !ts.is_finite() {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clip_timestamps: timestamp (seconds)",
        "must be finite",
        format_smolstr!("{ts}"),
      )));
    }
    // A finite-but-huge timestamp overflows `ts * FRAMES_PER_SECOND` to `±inf`,
    // which the `f64 as usize` cast inside `round_to_frames` would saturate
    // (`+inf → usize::MAX`, `-inf → 0`) into a bogus full-audio / empty clip.
    // CPython's `round(finite * FRAMES_PER_SECOND)` raises `OverflowError` once
    // the product is infinite, so reject a non-finite product here too — before
    // `round_to_frames` rounds and casts. (`round_to_frames` recomputes the same
    // product internally; verifying it here keeps that call's cast on a finite
    // value.)
    let frame_product = ts * FRAMES_PER_SECOND as f64;
    if !frame_product.is_finite() {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clip_timestamps: timestamp (seconds) × FRAMES_PER_SECOND",
        "frame product must be finite",
        format_smolstr!("{frame_product}"),
      )));
    }
    seek_points.push(round_to_frames(ts));
  }
  if seek_points.is_empty() {
    seek_points.push(0);
  }
  if seek_points.len() % 2 == 1 {
    seek_points.push(content_frames);
  }
  // `zip(points[::2], points[1::2])` — even/odd interleave into (start, end).
  // Clamp BOTH endpoints of EVERY pair to `[0, content_frames]` (the reference
  // clamps only the final end at `:928`, but its in-loop `content_frames - seek`
  // / `seek_clip_end - seek` math effectively bounds the rest; here the clamp is
  // explicit to keep the seek loop's subtraction non-negative). Drop degenerate
  // clips (`start >= end` after clamping) so they yield no windows.
  Ok(
    seek_points
      .chunks_exact(2)
      .filter_map(|pair| {
        let start = pair[0].min(content_frames);
        let end = pair[1].min(content_frames);
        (start < end).then_some((start, end))
      })
      .collect(),
  )
}

/// Transcribe a (possibly long) mel by sliding a 30-second window over it —
/// the core seek loop of `generate` (`whisper.py:978-1300`).
///
/// When [`TranscribeOptions::word_timestamps`] is set, each window additionally
/// runs the cross-attention DTW alignment ([`super::timing::add_word_timestamps`])
/// to attach per-word timings and (with
/// [`TranscribeOptions::hallucination_silence_threshold`]) the
/// hallucination-silence skip; when it is unset the loop is the plain
/// segment-level seek with no added cost.
///
/// `mel` is the full `(num_frames, n_mels)` log-mel spectrogram (frames on
/// axis 0), which the caller typically padded by a trailing 30-second chunk so
/// the final real window has full context (`log_mel_spectrogram`'s default
/// `padding = N_SAMPLES`). `content_frames` is the number of **real** (non-
/// padding) frames — the reference's `mel.shape[-2] - N_FRAMES` for a padded
/// mel, or the full length for an unpadded one (`whisper.py:763`). The seek
/// loop is bounded by `content_frames` (clamped to the mel length), so the
/// trailing pad is never decoded as a standalone content window.
///
/// Language resolution is owned here (`whisper.py:906-913`): when
/// `options.decode.language` is `None` (and the checkpoint is multilingual)
/// the language is detected from the first window via [`detect_language`], the
/// tokenizer's start-of-transcript language token is rebuilt to match
/// ([`HFTokenizerWrapper::with_language`]), and the resolved / detected
/// language is reported on the [`TranscribeResult`].
///
/// The loop advances `seek` over the content frames, decodes each `N_FRAMES`
/// window with the temperature fallback, skips silent windows (`no_speech_prob`
/// gate), and either fast-forwards a window or advances to the last in-window
/// timestamp.
///
/// Each window is conditioned on the running transcript
/// ([`TranscribeOptions::condition_on_previous_text`], `whisper.py:1033`): the
/// previous windows' decoded tokens (seeded with the first-window prompt) are
/// fed as the window's prompt, reset when conditioning is disabled or a
/// high-temperature fallback was used (`whisper.py:1279-1281`).
///
/// **First-window prompt precedence.** The initial context is seeded from
/// [`TranscribeOptions::initial_prompt`] when set (the reference's documented
/// knob, `whisper.py:990-994`); otherwise it falls back to the lower-level
/// [`DecodingOptions::prompt`] on `options.decode`. `initial_prompt` wins if
/// both are set. A caller-supplied `decode.prompt` is therefore honored as the
/// first window's context rather than silently dropped — the per-window prompt
/// overwrite (`whisper.py:1033`) is correct because the seed is already part of
/// the running history for window 0. Whichever seed is used only conditions the
/// decode; it is never part of the emitted transcript (the accumulated text is
/// built from the per-window segment texts, which never contain the prompt — the
/// reference achieves the same end via its `all_tokens[len(initial_prompt):]`
/// strip at `whisper.py:1299`).
///
/// Decoding is restricted to [`TranscribeOptions::clip_timestamps`]
/// (`whisper.py:1018-1026`); an empty clip list spans the whole audio. Each clip
/// is saturated into the real content-frame range, so any user clip list
/// terminates with a bounded window count (no hang, panic, or underflow) — a
/// degenerate or out-of-range clip simply contributes no windows.
///
/// # Errors
/// - [`Error::RankMismatch`] if `mel` is not rank 2;
/// - [`Error::OutOfRange`] if any [`TranscribeOptions::clip_timestamps`] value
///   is not finite (`NaN` / `±inf`), or if its `FRAMES_PER_SECOND` frame product
///   overflows to a non-finite value (a finite-but-huge timestamp);
/// - propagates [`detect_language`], [`decode_with_fallback`], the initial-
///   prompt encode, and the mel-slice op errors.
pub fn transcribe(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  mel: &Array,
  content_frames: usize,
  options: &TranscribeOptions,
) -> Result<TranscribeResult> {
  let shape = mel.shape();
  if shape.len() != 2 {
    return Err(Error::RankMismatch(crate::error::RankMismatchPayload::new(
      "transcribe: mel must be rank 2 (num_frames, n_mels)",
      shape.len() as u32,
      shape,
    )));
  }
  // Exclude trailing feature padding: the seek loop must stop at the last real
  // frame, never decoding the 30-second context pad as a content window. Clamp
  // to the mel length so an over-large `content_frames` cannot slice past it.
  let content_frames = content_frames.min(shape[0]);

  // Resolve the decode language once and rebuild the tokenizer SOT language
  // token to match (`whisper.py:906-913`). Detection — which encodes the first
  // window — only runs when the language is unknown and the checkpoint is
  // multilingual; a supplied language (or an English-only checkpoint) needs no
  // encoder pass here.
  let language = match options.decode.language.as_deref() {
    Some(lang) => lang.to_string(),
    None if !tokenizer.is_multilingual() => "en".to_string(),
    None => {
      // `pad_or_trim(mel, N_FRAMES, axis=-2)` first window (`whisper.py:783`).
      let first_window = pad_or_trim(&slice_frames(mel, 0, N_FRAMES.min(shape[0]))?, N_FRAMES, 0)?;
      let first_features = encode_once(model, &first_window)?;
      let (code, _probs) = detect_language(model, tokenizer, &first_features)?;
      code.to_string()
    }
  };
  let tokenizer = &tokenizer.with_language(&language);

  let timestamp_begin = tokenizer.timestamp_begin();
  // The decoder's text context length, used to bound each window's prompt slice
  // to exactly the tail `build_initial_tokens` keeps (`n_text_ctx / 2 - 1`).
  let n_text_ctx = model.dims().n_text_ctx();
  // mel frames per output token (`input_stride = N_FRAMES / n_audio_ctx = 2`).
  let input_stride = N_FRAMES / model.dims().n_audio_ctx().max(1);
  // time per output token (`input_stride * HOP / SR ≈ 0.02 s`).
  let time_precision = input_stride as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;

  let mut segments: Vec<Segment> = Vec::new();
  let mut all_text = String::new();
  // `last_speech_timestamp` (`whisper.py:1017`) — the running end of the last
  // emitted word, used by the word-timestamp duration hacks and the
  // hallucination heuristic. Unused on the no-word-timestamp path.
  let mut last_speech_timestamp = 0.0f64;

  // Clips to process (`whisper.py:915-931`, `:1018`): an empty `clip_timestamps`
  // yields one clip spanning the whole audio, so the loop is byte-identical to
  // the unclipped path. Each clip seeks to its start and bounds windows by its
  // end.
  let seek_clips = compute_seek_clips(&options.clip_timestamps, content_frames)?;
  // The running prompt history (`all_tokens` / `prompt_reset_since`), seeded
  // with the optional initial prompt (`whisper.py:986-994`). The seed retains
  // only the decoder-visible tail (`n_text_ctx / 2 - 1`), which is byte-identical
  // to seeding the full prompt since `window_prompt` only ever returns that tail.
  let mut history = PromptHistory::seed(
    tokenizer,
    options.initial_prompt.as_deref(),
    &options.decode.prompt,
    options.condition_on_previous_text,
    n_text_ctx,
  )?;

  // Per-window decode template. Every window decodes with `options.decode`
  // except its `prompt`, which the reference overwrites each iteration with the
  // running history tail (`whisper.py:1033`). Build the template by cloning every
  // field EXCEPT the prompt (left empty), so the caller's original
  // `decode.prompt` (used solely to seed `history` above, and possibly large) is
  // never cloned by this path at all — not even once to immediately drop it. The
  // per-window clone below then copies the (empty) template plus the bounded
  // `window_prompt` tail. The decode result is byte-identical to cloning
  // `options.decode` and overwriting `prompt` in-loop, since `prompt` is the only
  // field the loop ever changes.
  let decode_template = options.decode.clone_without_prompt();

  for (seek_clip_start, seek_clip_end) in seek_clips {
    let mut seek = seek_clip_start;
    // `compute_seek_clips` saturates every clip into `[0, content_frames]` and
    // drops inverted/zero-length clips, so `seek_clip_end <= content_frames` and
    // `seek_clip_start < seek_clip_end` always hold here. The extra `seek <
    // content_frames` guard plus the saturating `segment_size` below are belt-
    // and-braces: even a degenerate clip would contribute no windows (a zero
    // `segment_size` exits the loop) rather than spin or underflow.
    while seek < seek_clip_end && seek < content_frames {
      let time_offset = seek as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
      // Bound the window by the clip end as well as the audio end
      // (`whisper.py:1024-1026`). Saturating subtraction keeps `segment_size`
      // non-negative for any clip: a window that has run out of content frames
      // yields `segment_size == 0`, ending the loop instead of underflowing.
      let segment_size = N_FRAMES
        .min(content_frames.saturating_sub(seek))
        .min(seek_clip_end.saturating_sub(seek));
      if segment_size == 0 {
        break;
      }

      // mel[seek : seek + segment_size], padded to N_FRAMES.
      let mel_segment = slice_frames(mel, seek, segment_size)?;
      let mel_segment = pad_or_trim(&mel_segment, N_FRAMES, 0)?;

      // Condition this window on the running transcript
      // (`whisper.py:1033`: `decode_options["prompt"] = all_tokens[reset:]`).
      // `window_prompt` already trims to the decoder-visible tail
      // (`build_initial_tokens` keeps only the last `n_text_ctx / 2 - 1`), so
      // this clones exactly the tokens the decode will use. `decode_template`
      // already has an empty `prompt`, so this clone copies no caller prompt.
      let mut decode_options = decode_template.clone();
      decode_options.prompt = history.window_prompt(n_text_ctx).to_vec();

      let result = decode_with_fallback(
        model,
        tokenizer,
        &mel_segment,
        &decode_options,
        &language,
        &options.temperatures,
        options.compression_ratio_threshold,
        options.logprob_threshold,
        options.no_speech_threshold,
      )?;

      // No-voice-activity skip (`whisper.py:1038-1050`): skip silent windows,
      // unless the logprob is high enough to override.
      if let Some(nst) = options.no_speech_threshold {
        let mut should_skip = result.no_speech_prob > nst;
        if let Some(lpt) = options.logprob_threshold
          && result.avg_logprob > lpt
        {
          should_skip = false;
        }
        if should_skip {
          seek += segment_size;
          continue;
        }
      }

      let previous_seek = seek;
      // Timestamp-aware segment cut (`whisper.py:1081-1149`): find consecutive
      // timestamp pairs to cut sub-segments, else one segment for the whole
      // window. Collect into a per-window `current_segments` (the reference's
      // local list) — the global `segments` / `all_text` are extended below, so
      // the word-timestamp / hallucination pass can adjust this window first.
      let tokens = &result.tokens;
      let mut current_segments: Vec<Segment> = Vec::new();
      let (advance, single_timestamp_ending) = advance_and_collect_segments(
        tokens,
        timestamp_begin,
        time_offset,
        time_precision,
        segment_size,
        input_stride,
        &result,
        tokenizer,
        &mut current_segments,
      )?;
      seek += advance;

      if options.word_timestamps {
        // Attach per-word timings + apply the duration hacks (`whisper.py:1151-
        // 1161`), then re-derive `seek` from the last word end and run the
        // hallucination-silence skip. `skip_window` signals the reference's
        // leading-silence `continue` (the window's segments are dropped).
        let outcome = apply_word_timestamps(
          model,
          tokenizer,
          &mel_segment,
          segment_size,
          time_offset,
          previous_seek,
          seek,
          single_timestamp_ending,
          &mut current_segments,
          &mut last_speech_timestamp,
          content_frames,
          options,
        )?;
        seek = outcome.seek;
        if outcome.skip_window {
          continue;
        }

        // Clear degenerate segments before accumulating (`whisper.py:1253-1261`):
        // a segment that is instantaneous (`start == end`) or whose text is
        // whitespace-only has its text, tokens, and words emptied. This is part of
        // the word-timestamp finalization, so it is gated to the word-timestamp
        // path: the default `word_timestamps == false` path stays byte-identical
        // to the pre-feature behavior (every segment emitted with its sampled
        // tokens / text intact), which callers reading `Segment.tokens` rely on.
        clear_degenerate_segments(&mut current_segments);
      }

      // Thread this window's decoded tokens into the prompt history, then apply
      // the reset rule (`whisper.py:1271-1281`). This runs only for non-skipped
      // windows (the silence / leading-silence `continue` paths above bypass it,
      // matching the reference's bottom-of-loop `all_tokens.extend`), so a skipped
      // window leaves the next window's prompt unchanged.
      history.push_window(
        current_segments.iter().flat_map(|s| s.tokens.iter()),
        result.temperature,
      );

      // Append this window's segments + their text to the running totals.
      for segment in &current_segments {
        all_text.push_str(&segment.text);
      }
      segments.append(&mut current_segments);
    }
  }

  Ok(TranscribeResult {
    text: all_text.trim().to_string(),
    language,
    segments,
  })
}

/// Clear instantaneous / whitespace-only segments in place — `whisper.py:1253-
/// 1261` (`if a segment is instantaneous or does not contain text, clear it`).
///
/// For each segment whose `start == end` (zero duration) **or** whose text is
/// whitespace-only (`text.strip() == ""`), the reference empties its `text`,
/// `tokens`, **and** `words`, so the emitted segment carries no public words /
/// tokens and adds no whitespace to the accumulated transcript. Non-degenerate
/// segments are left untouched.
fn clear_degenerate_segments(segments: &mut [Segment]) {
  for segment in segments.iter_mut() {
    if segment.start == segment.end || segment.text.trim().is_empty() {
      segment.text.clear();
      segment.tokens.clear();
      segment.words.clear();
    }
  }
}

/// The outcome of the word-timestamp + hallucination pass for one window: the
/// (possibly re-derived) `seek` and whether the window's segments are dropped
/// (the reference's leading-silence `continue`, `whisper.py:1196`).
struct WordTimestampOutcome {
  seek: usize,
  skip_window: bool,
}

/// Attach per-word timestamps to a window's `current_segments`, re-derive `seek`
/// from the last word end, and run the hallucination-silence skip — the
/// `word_timestamps` branch of the seek loop (`whisper.py:1151-1242`).
///
/// `time_offset` / `previous_seek` are the pre-advance window position; `seek`
/// the post-segment-cut advance. `content_frames` bounds the final seek.
/// Returns the updated seek + whether to drop this window (leading-silence
/// hallucination skip).
///
/// # Errors
/// Propagates [`timing::add_word_timestamps`].
#[allow(clippy::too_many_arguments)]
fn apply_word_timestamps(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  mel_segment: &Array,
  segment_size: usize,
  time_offset: f64,
  previous_seek: usize,
  seek: usize,
  single_timestamp_ending: bool,
  current_segments: &mut Vec<Segment>,
  last_speech_timestamp: &mut f64,
  content_frames: usize,
  options: &TranscribeOptions,
) -> Result<WordTimestampOutcome> {
  let mut seek = seek;
  let segment_duration = segment_size as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
  // `window_end_time` is the end of the full PADDED 30-second window
  // (`seek + N_FRAMES`), NOT `time_offset + segment_duration` — both the OpenAI
  // and mlx-audio references derive it from `N_FRAMES` even on a final partial
  // window (`whisper.py:1021-1023`). The hallucination `remaining` /
  // `silence_after` checks below are defined against this padded end, so it is
  // kept verbatim for parity. (`segment_duration` / `content_duration` carry the
  // real audio extent where the reference uses those instead.)
  let window_end_time = (previous_seek + N_FRAMES) as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
  let content_duration = content_frames as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;

  // `add_word_timestamps(...)` (`whisper.py:1152-1161`): attach the words +
  // duration hacks. The reference passes the PRIOR `last_speech_timestamp` in
  // (for the per-segment pause hack) but does not feed the function's local
  // mutation back to the caller — the cross-window value is only advanced after
  // the skip decision below (`whisper.py:1239-1241`). `prev_last_speech` keeps
  // that prior accepted-speech end so the hallucination `silence_before` check
  // measures the gap against it, not against this window's last word.
  let prev_last_speech = *last_speech_timestamp;
  timing::add_word_timestamps(
    model,
    tokenizer,
    current_segments,
    mel_segment,
    segment_size,
    previous_seek,
    &options.prepend_punctuations,
    &options.append_punctuations,
    prev_last_speech,
  )?;

  // Re-derive `seek` from the last word end (`whisper.py:1163-1169`), unless the
  // window ended on a lone trailing timestamp.
  if !single_timestamp_ending
    && let Some(last_word_end) = timing::get_end(current_segments)
    && last_word_end > time_offset
  {
    seek = round_to_frames(last_word_end);
  }

  // Hallucination-silence skip (`whisper.py:1171-1237`), measured against the
  // PRIOR `last_speech_timestamp` (`prev_last_speech`).
  if let Some(threshold) = options.hallucination_silence_threshold {
    let skip = hallucination_silence_skip(HallucinationSkipParams {
      current_segments,
      threshold,
      prev_last_speech,
      time_offset,
      previous_seek,
      segment_size,
      segment_duration,
      window_end_time,
      content_duration,
      content_frames,
      single_timestamp_ending,
      seek,
    });
    // The leading-silence drop (`whisper.py:1196` `continue`) returns WITHOUT
    // advancing the cross-window `last_speech_timestamp` — the next window's
    // pause heuristic must still see the prior accepted-speech end.
    if skip.skip_window {
      return Ok(WordTimestampOutcome {
        seek: skip.seek,
        skip_window: true,
      });
    }
    seek = skip.seek;
  }

  // Final `last_word_end` → `last_speech_timestamp` (`whisper.py:1239-1241`):
  // advance the cross-window value only now, for a window that is NOT dropped.
  if let Some(last_word_end) = timing::get_end(current_segments) {
    *last_speech_timestamp = last_word_end;
  }

  Ok(WordTimestampOutcome {
    seek,
    skip_window: false,
  })
}

/// Inputs to [`hallucination_silence_skip`] — the per-window timing context for
/// the hallucination-silence skip (`whisper.py:1171-1237`). Bundled to keep the
/// signature within the argument lint.
struct HallucinationSkipParams<'a> {
  /// This window's segments (already carrying [`Word`]s). Truncated in place at
  /// a hallucination surrounded by silence (`whisper.py:1235`).
  current_segments: &'a mut Vec<Segment>,
  /// The configured `hallucination_silence_threshold`.
  threshold: f64,
  /// The PRIOR accepted-speech end (the cross-window `last_speech_timestamp`
  /// from before this window) — `hal_last_end`'s seed (`whisper.py:1200`).
  prev_last_speech: f64,
  /// This window's start time in seconds.
  time_offset: f64,
  /// This window's pre-advance frame offset.
  previous_seek: usize,
  /// This window's real (non-pad) frame count.
  segment_size: usize,
  /// `segment_size`'s duration in seconds (`hal_next_start` fallback).
  segment_duration: f64,
  /// The padded 30-second window end in seconds.
  window_end_time: f64,
  /// The full content (non-pad) duration in seconds.
  content_duration: f64,
  /// The total content frame count (the seek upper bound).
  content_frames: usize,
  /// Whether the window ended on a lone trailing timestamp.
  single_timestamp_ending: bool,
  /// The seek already re-derived from the last word end.
  seek: usize,
}

/// The result of [`hallucination_silence_skip`]: the (possibly fast-forwarded)
/// seek and whether the window is dropped (the leading-silence `continue`).
struct SilenceSkip {
  seek: usize,
  skip_window: bool,
}

/// The hallucination-silence skip block (`whisper.py:1172-1237`), factored out
/// of [`apply_word_timestamps`] so it operates purely on populated segments —
/// it never touches the cross-window `last_speech_timestamp`, only reading the
/// PRIOR value (`params.prev_last_speech`) for the `silence_before` /
/// `hal_last_end` gap. The caller advances the cross-window value afterwards,
/// and only for a window that is not dropped.
fn hallucination_silence_skip(params: HallucinationSkipParams<'_>) -> SilenceSkip {
  let HallucinationSkipParams {
    current_segments,
    threshold,
    prev_last_speech,
    time_offset,
    previous_seek,
    segment_size,
    segment_duration,
    window_end_time,
    content_duration,
    content_frames,
    single_timestamp_ending,
    mut seek,
  } = params;

  if !single_timestamp_ending
    && let Some(last_word_end) = timing::get_end(current_segments)
    && last_word_end > time_offset
  {
    let remaining = window_end_time - last_word_end;
    seek = if remaining > threshold {
      round_to_frames(last_word_end)
    } else {
      previous_seek + segment_size
    };
  }

  // If the first segment with words is anomalous, skip the leading silence
  // (`whisper.py:1186-1196`) — dropping this window's segments.
  if let Some(first) = timing::next_words_segment(current_segments)
    && timing::is_segment_anomaly(Some(first))
  {
    let gap = first.start - time_offset;
    if gap > threshold {
      seek = previous_seek + round_to_frames(gap);
      return SilenceSkip {
        seek,
        skip_window: true,
      };
    }
  }

  // Skip silence before any hallucination surrounded by silence
  // (`whisper.py:1198-1237`). `hal_last_end` seeds from the PRIOR accepted
  // speech (`prev_last_speech`), then walks each segment's own end.
  let mut hal_last_end = prev_last_speech;
  for si in 0..current_segments.len() {
    if current_segments[si].words.is_empty() {
      continue;
    }
    if is_segment_anomaly_at(current_segments, si) {
      // `next_segment = next_words_segment(current_segments[si + 1:])`
      // (`whisper.py:1206-1208`): the NEXT segment WITH WORDS, used for both
      // `hal_next_start` and the `silence_after` anomaly check below — never the
      // immediate `si + 1` segment, which may be empty / timestamp-only.
      let next_segment = timing::next_words_segment(&current_segments[si + 1..]);
      let hal_next_start =
        next_segment.map_or(time_offset + segment_duration, |s| s.words[0].start);
      let seg_start = current_segments[si].start;
      let seg_end = current_segments[si].end;
      let silence_before = seg_start - hal_last_end > threshold
        || seg_start < threshold
        || seg_start - time_offset < 2.0;
      let silence_after = hal_next_start - seg_end > threshold
        || timing::is_segment_anomaly(next_segment)
        || window_end_time - seg_end < 2.0;
      if silence_before && silence_after {
        seek = round_to_frames((time_offset + 1.0).max(seg_start));
        if content_duration - seg_end < threshold {
          seek = content_frames;
        }
        current_segments.truncate(si);
        break;
      }
    }
    hal_last_end = current_segments[si].end;
  }

  SilenceSkip {
    seek,
    skip_window: false,
  }
}

/// `is_segment_anomaly(current_segments[idx])` with bounds — `None` for an
/// out-of-range index (the reference's `next_segment` may be `None`).
fn is_segment_anomaly_at(segments: &[Segment], idx: usize) -> bool {
  timing::is_segment_anomaly(segments.get(idx))
}

/// `round(seconds * FRAMES_PER_SECOND)` clamped non-negative — the seek
/// frame conversion (`whisper.py`'s `round(... * FRAMES_PER_SECOND)`).
///
/// Python's `round()` rounds halves to the nearest **even** integer (banker's
/// rounding: `round(12.5) == 12`, `round(13.5) == 14`), whereas Rust's
/// [`f64::round`] rounds halves **away from zero** (`12.5 → 13`). A clip
/// timestamp landing exactly on a half-frame would therefore map to a different
/// frame than the reference and shift a clip boundary by one frame, so this
/// uses [`f64::round_ties_even`] to match Python's `round()` exactly. Every
/// caller mirrors a reference `round(... * FRAMES_PER_SECOND)` (the
/// `clip_timestamps` seek points at `whisper.py:921` and the word-timestamp /
/// hallucination seek re-derivations at `:1169`, `:1182`, `:1193`, `:1226`);
/// the `last_timestamp_pos * input_stride` advance at `:1127` is integer
/// arithmetic with no rounding and is computed without this helper.
fn round_to_frames(seconds: f64) -> usize {
  let frames = (seconds * FRAMES_PER_SECOND as f64).round_ties_even();
  if frames <= 0.0 { 0 } else { frames as usize }
}

/// Slice `mel[start : start + len]` along the frame axis (axis 0).
fn slice_frames(mel: &Array, start: usize, len: usize) -> Result<Array> {
  let shape = mel.shape();
  let n_mels = i32::try_from(shape[1]).map_err(|_| dim_overflow("n_mels"))?;
  let s = i32::try_from(start).map_err(|_| dim_overflow("seek"))?;
  let e = i32::try_from(start + len).map_err(|_| dim_overflow("seek + segment"))?;
  ops::indexing::slice(mel, &[s, 0], &[e, n_mels], &[1, 1])
}

/// Cut a window's tokens into segments at consecutive-timestamp boundaries,
/// append them to `segments`, and return `(advance, single_timestamp_ending)` —
/// the segment-level core of `whisper.py:1081-1149`. `advance` is the frame
/// step for `seek`; `single_timestamp_ending` (`whisper.py:1082-1085`) is true
/// when the window ended on a lone trailing timestamp (consumed in full),
/// consumed by the word-timestamp seek re-derivation.
///
/// The advance is: if the window ended on a consecutive-timestamp pair, advance
/// by the last timestamp's position; otherwise advance the full `segment_size`.
#[allow(clippy::too_many_arguments)]
fn advance_and_collect_segments(
  tokens: &[u32],
  timestamp_begin: u32,
  time_offset: f64,
  time_precision: f64,
  segment_size: usize,
  input_stride: usize,
  result: &DecodingResult,
  tokenizer: &HFTokenizerWrapper<'_>,
  segments: &mut Vec<Segment>,
) -> Result<(usize, bool)> {
  // `timestamp_tokens = tokens >= timestamp_begin`.
  let is_ts: Vec<bool> = tokens.iter().map(|&t| t >= timestamp_begin).collect();
  // `single_timestamp_ending = is_ts[-2:] == [False, True]`.
  let single_timestamp_ending =
    tokens.len() >= 2 && !is_ts[tokens.len() - 2] && is_ts[tokens.len() - 1];

  // `consecutive = where(is_ts[:-1] & is_ts[1:]) + 1`.
  let mut consecutive: Vec<usize> = Vec::new();
  for i in 0..is_ts.len().saturating_sub(1) {
    if is_ts[i] && is_ts[i + 1] {
      consecutive.push(i + 1);
    }
  }

  if !consecutive.is_empty() {
    let mut slices = consecutive;
    if single_timestamp_ending {
      slices.push(tokens.len());
    }
    let mut last_slice = 0usize;
    for &current_slice in &slices {
      let sliced = &tokens[last_slice..current_slice];
      if let (Some(&first), Some(&last)) = (sliced.first(), sliced.last()) {
        let start_pos = (first.saturating_sub(timestamp_begin)) as f64;
        let end_pos = (last.saturating_sub(timestamp_begin)) as f64;
        push_segment(
          segments,
          tokenizer,
          time_offset + start_pos * time_precision,
          time_offset + end_pos * time_precision,
          sliced,
          result,
        )?;
      }
      last_slice = current_slice;
    }

    if single_timestamp_ending {
      // The whole window was consumed up to the final timestamp.
      Ok((segment_size, true))
    } else {
      // Advance to the last consumed timestamp (`whisper.py:1127`:
      // `seek += last_timestamp_pos * input_stride`). The `.max(1)` guarantees
      // forward progress (a degenerate `last_timestamp_pos == 0` would stall
      // the seek loop — the reference relies on the timestamp-monotonicity rule
      // to avoid it, but a hardened cap is cheap), and `.min(segment_size)`
      // keeps the advance within the window (for a valid 30 s window
      // `last_timestamp_pos * input_stride` is naturally <= segment_size).
      let last_ts_pos = tokens
        .get(last_slice.saturating_sub(1))
        .map_or(0, |&t| t.saturating_sub(timestamp_begin) as usize);
      Ok(((last_ts_pos * input_stride).max(1).min(segment_size), false))
    }
  } else {
    // No consecutive timestamps: one segment for the whole window
    // (`whisper.py:1131-1149`). Duration from the last timestamp, if any.
    let mut duration = segment_size as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
    let timestamps: Vec<u32> = tokens
      .iter()
      .copied()
      .filter(|&t| t >= timestamp_begin)
      .collect();
    if let Some(&last_ts) = timestamps.last()
      && last_ts != timestamp_begin
    {
      let last_pos = last_ts.saturating_sub(timestamp_begin) as f64;
      duration = last_pos * time_precision;
    }
    push_segment(
      segments,
      tokenizer,
      time_offset,
      time_offset + duration,
      tokens,
      result,
    )?;
    Ok((segment_size, false))
  }
}

/// Build a [`Segment`] from a token slice + timing and append it to
/// `segments`. Text is the non-special (`< eot`) tokens decoded
/// (`whisper.py:1000-1005`). The caller accumulates `all_text` from the
/// window's segments after any word-timestamp adjustment.
fn push_segment(
  segments: &mut Vec<Segment>,
  tokenizer: &HFTokenizerWrapper<'_>,
  start: f64,
  end: f64,
  tokens: &[u32],
  result: &DecodingResult,
) -> Result<()> {
  let eot = tokenizer.eot();
  let text_tokens: Vec<u32> = tokens.iter().copied().filter(|&t| t < eot).collect();
  let text = tokenizer.decode(&text_tokens, false)?;
  segments.push(Segment {
    start,
    end,
    text,
    tokens: tokens.to_vec(),
    temperature: result.temperature,
    avg_logprob: result.avg_logprob,
    no_speech_prob: result.no_speech_prob,
    compression_ratio: result.compression_ratio,
    words: Vec::new(),
  });
  Ok(())
}

#[cfg(test)]
mod tests;

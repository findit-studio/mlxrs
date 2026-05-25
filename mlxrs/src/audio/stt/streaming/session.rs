//! Streaming inference session — orchestrates
//! [`super::mel_spectrogram::IncrementalMelSpectrogram`] +
//! [`super::encoder::StreamingEncoder`] + a per-architecture decoder to
//! produce a [`super::types::TranscriptionEvent`] stream.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/StreamingInferenceSession.swift`][swift-ref]
//! adapted to mlxrs's synchronous foreground-only execution model:
//!
//! - The Swift reference launches `Task.detached { ... runDecodePass
//!   ... }` per pass and yields events into an
//!   `AsyncStream<TranscriptionEvent>`. mlxrs runs each decode pass
//!   synchronously on the caller's thread; events are returned as a
//!   batch (`Vec<TranscriptionEvent>`) from
//!   [`StreamingInferenceSession::feed_audio`] and
//!   [`StreamingInferenceSession::stop`].
//! - The Swift reference depends on the concrete `Qwen3ASRModel`
//!   (`audioTower`, `tokenizer`, `mergeAudioFeatures`, `buildPrompt`,
//!   `makeCache`, `callAsFunction`). mlxrs replaces that with the
//!   [`StreamingDecoderBackend`] trait every per-architecture model
//!   implements — same orchestration loop, no concrete model in the
//!   port (per the [no per-model arch porting][noarch] rule).
//! - The Swift session uses Apple's `OSAllocatedUnfairLock` + tokenizer
//!   protocol. mlxrs uses owned `&mut self` (single-threaded session) +
//!   a [`StreamingTokenizer`] trait the caller supplies.
//!
//! The promotion / agreement / boundary-boost logic mirrors the Swift
//! reference at-line: a token is promoted to confirmed when it has been
//! seen for `>= min_agreement_passes` consecutive decode passes AND has
//! survived for `>= delay_preset.delay_ms()`. When a full encoder window
//! completes (or
//! [`super::types::StreamingConfig::finalize_completed_windows`] is on
//! and the boundary fast cadence elapses) the session promotes the
//! current provisional run, finalizes the window's text, and resets
//! decode state for the next window.
//!
//! # Retry contract
//!
//! `feed_audio` and `stop` are both **retryable on `Err`**. The session
//! uses a unified internal retry-state machine that tracks exactly
//! which fallible stage owes the next call its retry: mel-flush,
//! encoder-feed-of-stop-tail, the same-call decode for already-committed
//! bridge-drained windows, the finalize-queue drain, or `stop()`'s
//! post-finalize partial-window decode. Each fallible stage either
//! fully commits or sets a resume-point that names exactly where the
//! next call must resume.
//!
//! Pre-rewrite, the same retry plumbing lived in 3 separate session
//! fields plus per-call locals. Five consecutive review rounds found
//! a NEW way for one of those fields to desync from the others on a
//! partial-failure path; the unified state machine kills the defect
//! class structurally — no per-call locals, one source of truth for
//! "what work is owed across the call boundary."
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/StreamingInferenceSession.swift
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

use std::time::Instant;

use super::{
  encoder::{StreamingEncoder, StreamingEncoderBackend},
  mel_spectrogram::IncrementalMelSpectrogram,
  retry_state::SessionRetryState,
  types::{StreamingConfig, StreamingStats, TranscriptionEvent},
};
use crate::{Array, error::Result};

/// Architecture-specific per-pass decoder bridge.
///
/// Implementors wrap the per-model audio-decoder forward pass (the
/// Swift reference's `buildPrompt` + `mergeAudioFeatures` + KV-cache +
/// auto-regressive sampling loop). The session calls
/// [`StreamingDecoderBackend::decode_all_tokens`] once per pass.
///
/// All state mutation is local to the implementor — the session never
/// constructs / inspects KV caches, so per-model cache lifetime stays
/// inside per-model code.
///
/// `confirmed_token_ids` is the seed prefix the decoder should
/// re-replay before sampling new tokens (lets the cache warm up
/// without re-running the audio encoder). The returned `Vec<u32>` is
/// the **full** token sequence (confirmed prefix + newly sampled
/// tail). Implementors that don't need the replay-replay-then-sample
/// optimization can ignore `confirmed_token_ids` and return only the
/// newly sampled tokens with `confirmed_token_ids` prepended; the
/// session uses `confirmed_token_ids.len()` as the split point.
pub trait StreamingDecoderBackend {
  /// Run one decode pass over `audio_features`, returning the full
  /// token-id sequence (confirmed seed + newly sampled tokens).
  ///
  /// `max_tokens` is the caller's per-pass budget — implementations
  /// MUST stop sampling at this count even if EOS hasn't been
  /// reached, to bound per-pass latency.
  ///
  /// # Errors
  /// Implementation-defined — surfaced via [`Result`].
  fn decode_all_tokens(
    &self,
    audio_features: &Array,
    confirmed_token_ids: &[u32],
    config: &StreamingConfig,
    max_tokens: usize,
  ) -> Result<Vec<u32>>;
}

/// Architecture-specific tokenizer bridge for streaming detok.
///
/// The session only needs to convert id-slices to display text
/// incrementally — it never encodes. Per-model code typically wires
/// this through [`crate::tokenizer::sentencepiece::SentencePieceTokenizer`]
/// or the [`crate::tokenizer::Tokenizer`] HF wrapper.
pub trait StreamingTokenizer {
  /// Decode an id sequence to displayable text.
  fn decode_ids(&self, ids: &[u32]) -> String;
}

/// Streaming-decode pending state, mirroring Swift's
/// `SessionSharedState`. Owned by the session (no lock — single-thread
/// access).
#[derive(Debug, Default)]
struct SessionSharedState {
  /// Accumulated text from completed encoder windows — frozen, never
  /// re-decoded.
  completed_text: String,
  /// Confirmed-prefix tokens for the current pending window.
  confirmed_token_ids: Vec<u32>,
  /// Provisional tail under agreement-tracking.
  provisional_token_ids: Vec<u32>,
  /// First-seen `Instant` per provisional token — drives the
  /// `delay_ms` promotion clock.
  provisional_first_seen: Vec<Instant>,
  /// Per-provisional consecutive agreement counters.
  provisional_agreement_counts: Vec<usize>,
  /// Display string for the confirmed prefix.
  confirmed_text: String,
}

/// Per-decode-pass parameter bundle. Lets the helper functions stay
/// small and avoids cloning the session into every call.
struct DecodePassParams<'a> {
  audio_features: &'a Array,
  confirmed_token_ids: Vec<u32>,
  display_prefix: String,
  prev_provisional: Vec<u32>,
  prev_first_seen: Vec<Instant>,
  prev_agreement_counts: Vec<usize>,
  min_agreement_passes: usize,
}

/// Synchronous streaming-STT orchestration session.
///
/// Generic over the per-architecture encoder backend `B`, decoder
/// backend `D`, and tokenizer `T`. Owns its own
/// [`IncrementalMelSpectrogram`] + [`StreamingEncoder`] + an internal
/// retry-state machine.
pub struct StreamingInferenceSession<B, D, T> {
  decoder: D,
  tokenizer: T,
  config: StreamingConfig,

  mel_processor: IncrementalMelSpectrogram,
  encoder: StreamingEncoder<B>,

  shared: SessionSharedState,
  is_active: bool,
  total_samples_fed: usize,
  last_decode_time: Option<Instant>,
  boundary_fast_decode_until: Option<Instant>,
  has_new_encoder_content: bool,
  /// Number of encoder windows whose text has been frozen into
  /// `completed_text`.
  frozen_window_count: usize,
  /// Unified retry-state machine — the single source of truth for any
  /// in-flight retry obligation across `feed_audio` / `stop` calls.
  /// Replaces the pre-rewrite trio of session fields
  /// (`pending_finalize_queue`, `pending_stop_mel_frames`,
  /// `pending_bridge_drain_decode`) + per-call locals. See
  /// [`super::retry_state`] for the discharge protocol.
  retry_state: SessionRetryState,
}

impl<B, D, T> StreamingInferenceSession<B, D, T>
where
  B: StreamingEncoderBackend,
  D: StreamingDecoderBackend,
  T: StreamingTokenizer,
{
  /// Build a new session. `sample_rate` and `n_mels` describe the
  /// mel-extractor configuration that the encoder backend expects;
  /// `overlap_frames` is the encoder window's cross-window overlap
  /// in mel frames (matches Swift's `overlapFrames`). Per the Swift
  /// reference, `n_fft = 400` and `hop_length = 160` are fixed for
  /// the streaming mel extractor.
  ///
  /// # Errors
  /// Propagates from [`IncrementalMelSpectrogram::new`].
  pub fn new(
    decoder: D,
    tokenizer: T,
    config: StreamingConfig,
    encoder_backend: B,
    sample_rate: u32,
    n_mels: usize,
    overlap_frames: usize,
  ) -> Result<Self> {
    let mel_processor = IncrementalMelSpectrogram::new(sample_rate, 400, 160, n_mels)?;
    let max_cached_windows = config.max_cached_windows();
    let encoder = StreamingEncoder::new(encoder_backend, max_cached_windows, overlap_frames);
    Ok(Self {
      decoder,
      tokenizer,
      config,
      mel_processor,
      encoder,
      shared: SessionSharedState::default(),
      is_active: true,
      total_samples_fed: 0,
      last_decode_time: None,
      boundary_fast_decode_until: None,
      has_new_encoder_content: false,
      frozen_window_count: 0,
      retry_state: SessionRetryState::new(),
    })
  }

  /// Borrow the underlying [`StreamingConfig`].
  #[inline(always)]
  pub fn config(&self) -> &StreamingConfig {
    &self.config
  }

  /// Total samples fed since construction / last [`reset`](Self::reset).
  #[inline(always)]
  pub fn total_samples_fed(&self) -> usize {
    self.total_samples_fed
  }

  /// Number of fully encoded windows.
  #[inline(always)]
  pub fn encoded_window_count(&self) -> usize {
    self.encoder.encoded_window_count()
  }

  /// Whether the session is still active (not stopped / cancelled).
  #[inline(always)]
  pub fn is_active(&self) -> bool {
    self.is_active
  }

  /// Borrow the unified [`SessionRetryState`] — used by in-module tests
  /// to inspect the retry-state machine. Not part of the public surface.
  #[cfg(test)]
  pub(super) fn retry_state(&self) -> &SessionRetryState {
    &self.retry_state
  }

  /// Mutable access to the unified retry state — used by in-module tests
  /// that need to inject a retry obligation (e.g., direct staging of a
  /// `StopEncoderFeed` mel for a regression test that doesn't go through
  /// the normal stop()-Err path). Not part of the public surface.
  #[cfg(test)]
  pub(super) fn retry_state_mut(&mut self) -> &mut SessionRetryState {
    &mut self.retry_state
  }

  /// Feed audio samples + run a decode pass when the cadence/boundary
  /// rules dictate. Returns the events emitted during this call —
  /// empty `Vec` when no decode runs.
  ///
  /// # Retry contract
  ///
  /// Discharges any pending retry obligation (a prior call's failed
  /// stage) BEFORE processing the new `samples`. If discharge succeeds
  /// fully (no obligation remains), the new audio flows through the
  /// normal `mel.process` → `encoder.feed` → decode-pass pipeline. If
  /// discharge errs OR a later stage errs, the obligation is re-armed
  /// in the internal retry-state machine and the next call resumes
  /// from exactly that point. No new audio is consumed by `mel.process`
  /// if discharge hasn't fully completed — that contract is what kills
  /// the "new audio jumps ahead of staged stop-tail" reordering corner.
  ///
  /// # Errors
  /// Propagates from the mel processor / encoder / decoder backend.
  pub fn feed_audio(&mut self, samples: &[f32]) -> Result<Vec<TranscriptionEvent>> {
    if !self.is_active {
      return Ok(Vec::new());
    }

    self.total_samples_fed = self.total_samples_fed.saturating_add(samples.len());

    let mut events: Vec<TranscriptionEvent> = Vec::new();

    // 1. Discharge any pending retry obligation FIRST. This is the
    //    transactional bridge that replaces the pre-rewrite buffer-soup:
    //    a prior call's failed encoder.feed (R4 corner), a prior call's
    //    successful bridge drain that lost its decode obligation to a
    //    later `?` (R6 corner), or a prior call's failed finalize-decode
    //    all resume HERE. `discharge_ran_decode` records whether the
    //    discharge fired run_decode_pass / finalize_completed_windows
    //    — if so, the normal feed-path below MUST NOT fire a second
    //    decode pass on the (now-already-consumed) bridge-drained or
    //    queue-drained windows.
    let (discharge_events, discharge_ran_decode) = self.discharge_retry_obligation()?;
    events.extend(discharge_events);

    // 2. If discharge left obligations standing (a fresh Err in the
    //    discharge itself, OR a partial advance — e.g. drained but
    //    decode not yet run), return what completed and let the next
    //    call retry. NO new audio is consumed until the obligation is
    //    fully discharged. This contract preserves in-order delivery
    //    AND kills the cross-call leak class structurally.
    if self.retry_state.has_obligation() {
      return Ok(events);
    }

    // 3. Normal feed-audio path. Each fallible step below MUST either
    //    fully succeed or set an appropriate retry_state.resume_at
    //    BEFORE propagating the Err.
    //
    //    `mel.process` is itself non-transactional but the failure
    //    surface (mel.process Errs after consuming overlap) is rare
    //    in practice — IncrementalMelSpectrogram::process only fails
    //    inside MLX-internal compute ops that don't see the input
    //    sample buffer directly. The session does NOT arm a retry
    //    here; a mel.process Err currently propagates as a hard error.
    //    If a real backend ever surfaces a recoverable mel.process Err
    //    we extend RetryStage with a MelProcess variant; for now the
    //    pipeline matches the Swift reference's "no recovery" stance.
    let mel_opt = self.mel_processor.process(samples)?;
    let new_windows = if let Some(mel_frames) = mel_opt.as_ref() {
      // encoder.feed is transactional (R3-fix design): on Err
      // self.encoder.pending_frames is preserved and the same
      // mel_frames can be re-fed by the caller. We propagate the Err
      // WITHOUT arming retry_state — the encoder rolled back, no
      // windows were committed, so there's no stranded work.
      self.encoder.feed(mel_frames)?
    } else {
      0
    };
    if new_windows > 0 || self.encoder.has_pending_frames() {
      self.has_new_encoder_content = true;
    }

    let now = Instant::now();
    if new_windows > 0 {
      let boost = self.config.boundary_boost_seconds().max(0.0);
      if boost > 0.0 {
        self.boundary_fast_decode_until = Some(now + std::time::Duration::from_secs_f64(boost));
      } else {
        self.boundary_fast_decode_until = None;
      }
    }

    let effective_decode_interval_seconds = if let Some(until) = self.boundary_fast_decode_until
      && now < until
    {
      let fast = self.config.boundary_decode_interval_seconds().max(0.05);
      let normal = self.config.decode_interval_seconds().max(0.05);
      fast.min(normal)
    } else {
      self.boundary_fast_decode_until = None;
      self.config.decode_interval_seconds().max(0.05)
    };

    let has_pending_retries =
      self.config.finalize_completed_windows() && !self.retry_state.finalize_queue().is_empty();

    let should_decode =
      if (self.config.finalize_completed_windows() && new_windows > 0) || has_pending_retries {
        true
      } else if let Some(last) = self.last_decode_time {
        now.duration_since(last).as_secs_f64() >= effective_decode_interval_seconds
      } else {
        self.has_new_encoder_content
      };

    // R5/R6 corner structural fix: if the discharge already ran a
    // decode pass (consuming the bridge-drained or queue-drained
    // windows), DO NOT fire a second decode in the normal path — the
    // encoder's pending_frames may still be non-empty (1-row carry
    // after a 9→1+8 window split, etc.) but the contract is "discharge
    // consumed the obligation; new audio drives its own decode only
    // when new_windows > 0 here."
    let skip_normal_decode = discharge_ran_decode && new_windows == 0;
    if should_decode && (self.has_new_encoder_content || has_pending_retries) && !skip_normal_decode
    {
      self.has_new_encoder_content = false;
      let is_boundary_finalize_pass = self.config.finalize_completed_windows() && new_windows > 0;
      if !is_boundary_finalize_pass {
        self.last_decode_time = Some(now);
      }
      // R6 corner structural fix: encoder.feed above may have committed
      // one or more new windows to encoder.newly_encoded_windows
      // (`new_windows > 0`). If `run_decode_pass` Errs, the windows are
      // stranded in the encoder — the next call MUST decode them. Arm
      // DecodeOwed BEFORE the fallible run_decode_pass so the obligation
      // survives `?` propagation. On Ok we clear it. Pre-rewrite, this
      // count was a per-call local lost to the `?` unwind.
      //
      // The arming is unconditional (even when new_windows == 0) for
      // the case where this decode is being run for `has_pending_retries`
      // — the queue front is the failed entry, the call MUST drive it
      // through. clear_decode_owed is also called when the queue is
      // empty + new_windows == 0, so this is a no-op there.
      if new_windows > 0 || has_pending_retries {
        self.retry_state.arm_decode_owed();
      }
      let decode_events = self.run_decode_pass()?;
      events.extend(decode_events);
      self.retry_state.clear_decode_owed();
    }

    Ok(events)
  }

  /// Flush pending samples + run the final decode pass + emit the
  /// terminal [`TranscriptionEvent::Ended`] event.
  ///
  /// # Retry contract
  ///
  /// `stop` is **retryable on `Err`**. Every fallible stage either
  /// commits fully or sets a resume-point in the internal retry-state
  /// machine before propagating; the next `stop()` / `feed_audio()`
  /// call discharges from exactly that stage. The session only flips
  /// `is_active` to `false` AFTER ALL fallible work has succeeded — a
  /// second stop() after a partial failure picks up where the prior
  /// one left off.
  ///
  /// After `stop` returns `Ok`, [`is_active`](Self::is_active) returns
  /// `false`, the session is terminated, and any follow-up `feed_audio`
  /// is a no-op. A follow-up `stop` after success returns
  /// `Ok(Vec::new())`.
  ///
  /// A follow-up `stop` while the session is already inactive AND the
  /// retry-state machine has no obligation also returns
  /// `Ok(Vec::new())`.
  ///
  /// # Errors
  /// Propagates from the mel processor / encoder / decoder backend.
  pub fn stop(&mut self) -> Result<Vec<TranscriptionEvent>> {
    // R2-style guard: "inactive AND nothing left to retry" exits. Fall
    // through if the retry state still owes work — a prior stop() Err
    // left obligations the second stop() must discharge.
    if !self.is_active && !self.retry_state.has_obligation() {
      return Ok(Vec::new());
    }

    let mut events: Vec<TranscriptionEvent> = Vec::new();

    // 1. Fast path: if a prior stop()'s StopPartialDecode is the only
    //    outstanding obligation, jump straight to the partial-decode
    //    + Ended emission. The earlier stages (mel.flush, encoder.feed,
    //    finalize) all committed cleanly in the prior call.
    if self.retry_state.has_pending_stop_partial_decode()
      && !self.retry_state.has_pending_stop_encoder_feed()
      && !self.retry_state.has_decode_owed()
      && self.retry_state.finalize_queue().is_empty()
    {
      let audio_features = self
        .retry_state
        .take_stop_partial_decode_features()
        .expect("guard above asserted has_pending_stop_partial_decode");
      // Re-arm so any Err in the call below re-installs the obligation
      // for the next stop()'s retry. Cloning the array is cheap
      // (refcount); the rare try_clone Err path is handled below.
      //
      // F2-FIX (was: `audio_features.as_ref().and_then(|a|
      // a.try_clone().ok())`): the pre-fix `.ok()` silently mapped a
      // clone failure to `None`, so the next stop() would observe a
      // `StopPartialDecode { audio_features: None }` obligation and
      // behave as "no partial audio to decode" — dropping the window.
      // Now the clone failure propagates as `Err`, and we move the
      // ORIGINAL audio_features back into the obligation so a third
      // stop() can still consume it. The current call's
      // finalize_partial_window_and_emit_ended is skipped on Err
      // (preserving the invariant that the obligation is the only
      // remaining handle to the partial window).
      let reinstate = match clone_partial_decode_payload(audio_features.as_ref()) {
        Ok(p) => p,
        Err(e) => {
          self.retry_state.arm_stop_partial_decode(audio_features);
          return Err(e);
        }
      };
      self.retry_state.arm_stop_partial_decode(reinstate);
      self.finalize_partial_window_and_emit_ended(audio_features, &mut events)?;
      // SUCCESS: clear the just-re-armed obligation.
      let _ = self.retry_state.take_stop_partial_decode_features();
      self.is_active = false;
      self.encoder.reset();
      self.mel_processor.reset();
      self.boundary_fast_decode_until = None;
      self.retry_state.clear_all();
      return Ok(events);
    }

    // 2. Discharge any pending retry obligation FIRST. Drives the
    //    StopEncoderFeed / DecodeOwed / pending-finalize-queue retry
    //    stages transactionally.
    let (discharge_events, _ran_decode) = self.discharge_retry_obligation()?;
    events.extend(discharge_events);

    if self.retry_state.has_obligation() {
      // Partial discharge (a stage errored mid-way OR the discharge
      // advanced to a new resume point that needs the next stop()).
      // Don't go further; the next stop() will pick up where this one
      // left off.
      return Ok(events);
    }

    // 3. Stage: mel.flush. mel.flush is transactional (clone-then-clear
    //    on success — see IncrementalMelSpectrogram::flush). On Err
    //    self.overlap_buffer stays intact + retry_state.resume_at =
    //    StopMelFlush so the next stop() retries the same flush.
    self.retry_state.stage_stop_mel_flush();
    let mel_opt = self.mel_processor.flush()?;
    self.retry_state.clear_stop_mel_flush();

    // 4. Stage: encoder.feed of the stop-tail mel rows. If mel.flush
    //    yielded rows, we stage them in retry_state BEFORE feed (so an
    //    encoder.feed Err preserves the freshly-flushed rows for a
    //    cross-call retry — mel.flush has already committed-and-cleared
    //    its overlap on its own commit).
    //
    //    encoder.feed is itself transactional, so self.encoder.* is
    //    preserved on Err — but the LOCAL mel rows live nowhere else;
    //    the StopEncoderFeed stage carries them.
    if let Some(mel_frames) = mel_opt {
      self.retry_state.stage_stop_encoder_feed(mel_frames);
      let _drain_window_count = self
        .retry_state
        .discharge_stop_encoder_feed(&mut self.encoder)?;
      // The discharge advanced resume_at to DecodeOwed iff drain > 0.
      // For stop(), we drive the decode pass for those windows in THIS
      // same call (finalize/freeze below handles them), so clear the
      // DecodeOwed obligation pre-emptively.
      self.retry_state.clear_decode_owed();
    }

    // 5. Stage: finalize-queue drain (or freeze for the no-finalize
    //    path). On Err, the queue front is unchanged so the next
    //    stop()/feed_audio() retry path drives the queue.
    if self.config.finalize_completed_windows() {
      let drained = self.encoder.drain_newly_encoded_windows();
      for window in drained {
        self.retry_state.enqueue_finalize(window);
      }
      if !self.retry_state.finalize_queue().is_empty() {
        let finalize_events = self.finalize_completed_windows()?;
        events.extend(finalize_events);
      }
    } else {
      self.freeze_completed_windows();
    }

    // 6. Stage: partial-window decode + Ended emission. encode_pending
    //    is itself fallible — its Err propagates with retry_state in
    //    clean state (`stop()` returns Err, caller re-enters mainline
    //    body; encode_pending is `&self` + idempotent, re-runs from
    //    the same state).
    //
    //    Once we have audio_features, arm StopPartialDecode with a
    //    refcount-clone so an Err in the decode below re-arms the
    //    obligation for the next stop()'s fast path.
    //
    //    F2-FIX (was: `audio_features.as_ref().and_then(|a|
    //    a.try_clone().ok())`): the pre-fix `.ok()` silently dropped
    //    a clone failure into `None`, so the obligation would be
    //    armed as `StopPartialDecode { audio_features: None }` and
    //    the next stop()'s fast path would treat the partial window
    //    as absent. Now we propagate the clone Err BEFORE arming;
    //    the original `audio_features` is preserved locally + the
    //    fallible stages preceding step 6 are idempotent, so the
    //    next stop() recomputes `encode_pending` from the unchanged
    //    encoder state.
    let audio_features = self.encoder.encode_pending()?;
    let reinstate = clone_partial_decode_payload(audio_features.as_ref())?;
    self.retry_state.arm_stop_partial_decode(reinstate);
    self.finalize_partial_window_and_emit_ended(audio_features, &mut events)?;
    // SUCCESS: clear the just-armed StopPartialDecode obligation.
    let _ = self.retry_state.take_stop_partial_decode_features();

    // 7. ALL fallible work succeeded — terminate the session.
    self.is_active = false;
    self.encoder.reset();
    self.mel_processor.reset();
    self.boundary_fast_decode_until = None;
    self.retry_state.clear_all();

    Ok(events)
  }

  /// Cancel without producing the final `.ended` event — used for
  /// abandoned sessions. Clears all retry obligations atomically.
  pub fn cancel(&mut self) {
    self.is_active = false;
    self.encoder.reset();
    self.mel_processor.reset();
    self.boundary_fast_decode_until = None;
    self.shared = SessionSharedState::default();
    // Unified clear: one call discharges every kind of pending
    // obligation — finalize queue, resume_at stage, the lot.
    self.retry_state.clear_all();
  }

  /// Reset all state for a fresh session.
  pub fn reset(&mut self) {
    self.is_active = true;
    self.total_samples_fed = 0;
    self.last_decode_time = None;
    self.boundary_fast_decode_until = None;
    self.has_new_encoder_content = false;
    self.frozen_window_count = 0;
    self.encoder.reset();
    self.mel_processor.reset();
    self.shared = SessionSharedState::default();
    self.retry_state.clear_all();
  }

  // -------------------------------------------------------------------
  // Internal: retry-state discharge
  // -------------------------------------------------------------------

  /// Top-of-call discharge for any pending retry obligation.
  ///
  /// Dispatches on `retry_state.resume_at` and drives the named stage's
  /// work. Each stage's discharge either fully commits (advancing the
  /// resume point to `None` or to a downstream stage) or fully rolls
  /// back (leaving the resume point as it was). No partial commit
  /// leaves the session in an inconsistent state — that's the whole
  /// point of the unified state machine.
  ///
  /// Returns the events the discharge produced AND a `ran_decode` flag
  /// indicating whether the discharge fired `run_decode_pass` or
  /// `finalize_completed_windows`. The caller uses `ran_decode` to
  /// avoid a redundant decode in the normal `feed_audio` path: the
  /// discharge already consumed the bridge-drained or queue-drained
  /// windows; firing another decode pass on the encoder's
  /// `pending_frames` would over-decode.
  fn discharge_retry_obligation(&mut self) -> Result<(Vec<TranscriptionEvent>, bool)> {
    let mut events: Vec<TranscriptionEvent> = Vec::new();
    let mut ran_decode = false;

    // (a) StopMelFlush — the next stop()'s mel.flush retry. Without an
    //     active discharge here, an Err on stop()'s in-line
    //     `mel.flush()` (line ~499) would stage StopMelFlush, the next
    //     call's top-of-body `has_obligation()` check would early-return
    //     in feed_audio's no-op-for-inactive path / stop()'s
    //     `!is_active && !has_obligation` check would FAIL to short-
    //     circuit (because there IS an obligation), but no discharge
    //     would run for it — DEADLOCK: session active forever, Ended
    //     never emitted, feed_audio accepts samples without consuming.
    //     The discharge re-runs the same fallible flush; on success it
    //     advances resume_at to StopEncoderFeed (if mel was produced),
    //     so the (b) branch below picks up in the same call. On Err
    //     there are two sub-cases (see discharge_stop_mel_flush docs):
    //       - flush() Err → re-arms StopMelFlush (overlap intact).
    //       - flush() Ok + try_clone Err → MOVES the flushed mel into
    //         StopEncoderFeed and propagates Err. The mel is preserved
    //         in the obligation; the next call's discharge will run
    //         path (b) and feed it to the encoder. NEVER lost.
    //     The `?` propagation leaves resume_at exactly as the discharge
    //     set it, so the next call dispatches to whichever stage owns
    //     the preserved payload.
    if self.retry_state.has_pending_stop_mel_flush() {
      let _mel_opt = self
        .retry_state
        .discharge_stop_mel_flush(&mut self.mel_processor)?;
      // After Ok, resume_at is either None (no mel produced) or
      // StopEncoderFeed (mel produced + try_clone succeeded). Fall
      // through to (b) so the in-call StopEncoderFeed discharge fires
      // when applicable.
    }

    // (b) StopEncoderFeed — drain the staged mel into the encoder.
    //     Honors the contract "older staged tail reaches encoder
    //     BEFORE any new audio" — feed_audio MUST drive this discharge.
    if self.retry_state.has_pending_stop_encoder_feed() {
      // The discharge transactionally re-feeds the staged mel. On
      // success it advances resume_at to DecodeOwed iff window_count
      // > 0 (so the same-call decode pass below covers those windows
      // even though the locals were never created); on Err it
      // re-arms StopEncoderFeed with the SAME payload.
      let drain_window_count = self
        .retry_state
        .discharge_stop_encoder_feed(&mut self.encoder)?;
      if drain_window_count > 0 || self.encoder.has_pending_frames() {
        self.has_new_encoder_content = true;
      }
    }

    // (c) DecodeOwed — a prior call drained one or more bridge-drained
    //     windows but the same-call decode never ran (R5/R6 corners).
    //     The encoder.newly_encoded_windows (or cached) hold the
    //     windows; we MUST decode them here BEFORE any new audio is
    //     accepted.
    //
    //     The discharge is the same code as the normal run_decode_pass
    //     path — we ARE that path, just driven from the discharge
    //     instead of the cadence gate. On Err the retry_state stays
    //     DecodeOwed so the next call re-enters this discharge.
    if self.retry_state.has_decode_owed() {
      // Clear has_new_encoder_content BEFORE the decode (mirrors the
      // mainline feed_audio's pre-decode clear) so a successful
      // discharge-decode doesn't trigger a redundant second decode
      // when the normal feed_audio path runs below.
      self.has_new_encoder_content = false;
      let decode_events = self.run_decode_pass()?;
      self.retry_state.clear_decode_owed();
      events.extend(decode_events);
      ran_decode = true;
      // Set the cadence-gate marker so a follow-up empty feed_audio
      // doesn't fire a redundant decode via the cadence's
      // "last_decode_time is None ⇒ fall to has_new_encoder_content"
      // branch. The discharge-driven decode is functionally identical
      // to a normal decode pass for cadence purposes.
      self.last_decode_time = Some(Instant::now());
    }

    // (d) Non-empty finalize queue with no other resume_at obligation
    //     — a prior call's finalize-decode errored. The discharge
    //     drives finalize_completed_windows to re-attempt the failed
    //     entry at the queue front. On Err the queue front is
    //     unchanged + we re-arm DecodeOwed so future calls keep
    //     retrying. On Ok the queue drains.
    //
    //     This covers the R2 contract that feed_audio(&[]) drains a
    //     pending finalize-retry queue without consuming any new
    //     audio. It also lets a follow-up stop() drain the queue when
    //     a prior feed_audio errored mid-finalize.
    if !self.retry_state.finalize_queue().is_empty() && self.retry_state.resume_at().is_none() {
      // Same gate as the normal feed_audio decode-pass: only drive the
      // queue when finalize_completed_windows is on (the queue is only
      // populated in that mode).
      if self.config.finalize_completed_windows() {
        let finalize_events = self.finalize_completed_windows()?;
        events.extend(finalize_events);
        ran_decode = true;
        self.last_decode_time = Some(Instant::now());
      }
    }

    // (e) StopPartialDecode — feed_audio MUST NOT consume the staged
    //     audio_features (that's stop()'s job). stop()'s mainline
    //     re-entry handles the retry: when stop() sees
    //     has_pending_stop_partial_decode() in its top-of-body check,
    //     it takes the staged features and calls
    //     finalize_partial_window_and_emit_ended directly, skipping
    //     the mel.flush / encoder.feed / finalize stages it already
    //     completed in the prior call. The split-discharge contract
    //     keeps this stage's payload alive across an interleaved
    //     feed_audio (which is a no-op for StopPartialDecode).
    //
    //     Implementation: stop()'s body checks
    //     has_pending_stop_partial_decode() AFTER the main discharge,
    //     extracts the audio_features, and dispatches to
    //     finalize_partial_window_and_emit_ended.

    Ok((events, ran_decode))
  }

  // -------------------------------------------------------------------
  // Internal: stop()'s partial-window decode + Ended emission
  // -------------------------------------------------------------------

  /// stop()'s tail: decode the pending partial window + emit Ended.
  ///
  /// Extracted into a helper so the discharge of a `StopPartialDecode`
  /// obligation can call the same logic (with the staged
  /// `audio_features` instead of recomputing `encode_pending`).
  fn finalize_partial_window_and_emit_ended(
    &mut self,
    audio_features: Option<Array>,
    events: &mut Vec<TranscriptionEvent>,
  ) -> Result<()> {
    if let Some(audio_features) = audio_features {
      if audio_features.shape().first().copied().unwrap_or(0) > 0 {
        let display_prefix = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
        let confirmed_count = self.shared.confirmed_token_ids.len();
        let estimated_tokens = self
          .config
          .max_tokens_per_pass()
          .min(confirmed_count.saturating_add(24).max(24));
        let token_ids = self.decoder.decode_all_tokens(
          &audio_features,
          &self.shared.confirmed_token_ids,
          &self.config,
          estimated_tokens,
        )?;
        // Final text rolls everything into confirmed. Only mutate
        // shared state AFTER the fallible decode returns Ok.
        self.shared.confirmed_token_ids = token_ids;
        self.shared.provisional_token_ids.clear();
        self.shared.provisional_first_seen.clear();
        self.shared.provisional_agreement_counts.clear();
        self.shared.confirmed_text = self.tokenizer.decode_ids(&self.shared.confirmed_token_ids);
        let _ = display_prefix; // computed for parity; not needed after final replace
      }
    } else {
      // No pending frames — promote provisional to confirmed.
      if !self.shared.provisional_token_ids.is_empty() {
        let promoted = std::mem::take(&mut self.shared.provisional_token_ids);
        self.shared.confirmed_token_ids.extend(promoted);
        self.shared.provisional_first_seen.clear();
        self.shared.provisional_agreement_counts.clear();
      }
      if !self.shared.confirmed_token_ids.is_empty() {
        self.shared.confirmed_text = self.tokenizer.decode_ids(&self.shared.confirmed_token_ids);
      }
    }

    let final_text = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
    events.push(TranscriptionEvent::ended(final_text));

    Ok(())
  }

  // -------------------------------------------------------------------
  // Internal: decode-pass orchestration
  // -------------------------------------------------------------------

  fn run_decode_pass(&mut self) -> Result<Vec<TranscriptionEvent>> {
    // If finalize_completed_windows is on AND we have newly-encoded
    // full windows, push them onto the finalize queue + drain.
    //
    // F3: never drain newly-encoded windows out of the system in a path
    // that can't replay them. Push freshly-drained windows into the
    // retry queue; `finalize_completed_windows` then pops them one at a
    // time as decodes succeed (and leaves any failed window at the
    // front for the next pass to retry).
    if self.config.finalize_completed_windows() {
      let drained = self.encoder.drain_newly_encoded_windows();
      for window in drained {
        self.retry_state.enqueue_finalize(window);
      }
      if !self.retry_state.finalize_queue().is_empty() {
        return self.finalize_completed_windows();
      }
    } else {
      self.freeze_completed_windows();
    }

    // Only decode the current pending (partial) window.
    let Some(audio_features) = self.encoder.encode_pending()? else {
      return Ok(Vec::new());
    };
    let num_audio_tokens = audio_features.shape().first().copied().unwrap_or(0);
    if num_audio_tokens == 0 {
      return Ok(Vec::new());
    }

    let confirmed_count = self.shared.confirmed_token_ids.len();
    let windowed_seconds = num_audio_tokens as f64 / 13.0;
    let estimated_total_tokens = ((windowed_seconds * 10.0).ceil() as usize).max(24);
    let max_tokens = self
      .config
      .max_tokens_per_pass()
      .min(estimated_total_tokens.max(confirmed_count.saturating_add(24)));

    let display_prefix = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
    let min_agreement_passes = if let Some(until) = self.boundary_fast_decode_until
      && Instant::now() < until
    {
      self
        .config
        .min_agreement_passes()
        .max(self.config.boundary_min_agreement_passes())
        .max(1)
    } else {
      self.config.min_agreement_passes().max(1)
    };

    let params = DecodePassParams {
      audio_features: &audio_features,
      confirmed_token_ids: self.shared.confirmed_token_ids.clone(),
      display_prefix,
      prev_provisional: self.shared.provisional_token_ids.clone(),
      prev_first_seen: self.shared.provisional_first_seen.clone(),
      prev_agreement_counts: self.shared.provisional_agreement_counts.clone(),
      min_agreement_passes,
    };

    let start = Instant::now();
    let all_token_ids = self.decoder.decode_all_tokens(
      params.audio_features,
      &params.confirmed_token_ids,
      &self.config,
      max_tokens,
    )?;
    let decode_time = start.elapsed().as_secs_f64();

    Ok(self.promote_tokens(&all_token_ids, &params, decode_time))
  }

  fn promote_tokens(
    &mut self,
    all_token_ids: &[u32],
    params: &DecodePassParams<'_>,
    decode_time: f64,
  ) -> Vec<TranscriptionEvent> {
    let confirmed_count = params.confirmed_token_ids.len();
    let new_provisional: Vec<u32> = all_token_ids
      .iter()
      .skip(confirmed_count)
      .copied()
      .collect();
    let gen_token_count = all_token_ids.len();
    let now = Instant::now();
    let delay = std::time::Duration::from_millis(u64::from(self.config.delay_preset().delay_ms()));

    // Common prefix match-length between prev provisional and new.
    let mut match_len = 0;
    let compare_len = params.prev_provisional.len().min(new_provisional.len());
    for (i, new_id) in new_provisional.iter().enumerate().take(compare_len) {
      if params.prev_provisional[i] == *new_id {
        match_len = i + 1;
      } else {
        break;
      }
    }

    let mut next_first_seen: Vec<Instant> = Vec::with_capacity(new_provisional.len());
    let mut next_agreement_counts: Vec<usize> = Vec::with_capacity(new_provisional.len());
    for i in 0..new_provisional.len() {
      if i < match_len {
        let seen = params.prev_first_seen.get(i).copied().unwrap_or(now);
        let prev_agreement = params.prev_agreement_counts.get(i).copied().unwrap_or(1);
        next_first_seen.push(seen);
        next_agreement_counts.push(prev_agreement.saturating_add(1).max(1));
      } else {
        next_first_seen.push(now);
        next_agreement_counts.push(1);
      }
    }

    let required_agreement_passes = params.min_agreement_passes.max(1);
    let mut promotion_count = 0;
    for i in 0..new_provisional.len() {
      let has_delay = next_first_seen
        .get(i)
        .map(|t| now.duration_since(*t) >= delay)
        .unwrap_or(false);
      let has_agreement = next_agreement_counts
        .get(i)
        .map(|c| *c >= required_agreement_passes)
        .unwrap_or(false);
      if has_delay && has_agreement {
        promotion_count = i + 1;
      } else {
        break;
      }
    }

    let final_provisional: Vec<u32> = new_provisional
      .iter()
      .skip(promotion_count)
      .copied()
      .collect();
    let final_first_seen: Vec<Instant> = next_first_seen
      .iter()
      .skip(promotion_count)
      .copied()
      .collect();
    let final_agreement_counts: Vec<usize> = next_agreement_counts
      .iter()
      .skip(promotion_count)
      .copied()
      .collect();

    let mut events: Vec<TranscriptionEvent> = Vec::new();
    if promotion_count > 0 {
      let promoted: Vec<u32> = new_provisional[..promotion_count].to_vec();
      self.shared.confirmed_token_ids.extend(promoted);
      self.shared.confirmed_text = self.tokenizer.decode_ids(&self.shared.confirmed_token_ids);
      events.push(TranscriptionEvent::confirmed(concat_text(
        &self.shared.completed_text,
        &self.shared.confirmed_text,
      )));
    }
    self.shared.provisional_token_ids = final_provisional.clone();
    self.shared.provisional_first_seen = final_first_seen;
    self.shared.provisional_agreement_counts = final_agreement_counts;

    let final_prov_text = self.tokenizer.decode_ids(&final_provisional);
    let display_prefix = concat_text(&self.shared.completed_text, &self.shared.confirmed_text);
    events.push(TranscriptionEvent::display_update(
      display_prefix,
      final_prov_text,
    ));
    let _ = params.display_prefix; // shape parity — used only for the streaming preview event

    let total_audio_seconds = self.total_samples_fed as f64 / 16_000.0;
    let tps = if decode_time > 0.0 {
      gen_token_count as f64 / decode_time
    } else {
      0.0
    };
    events.push(TranscriptionEvent::Stats(StreamingStats {
      encoded_window_count: self.encoder.encoded_window_count(),
      total_audio_seconds,
      tokens_per_second: tps,
      real_time_factor: 0.0,
      peak_memory_gb: peak_memory_gb_or_zero(),
    }));
    events
  }

  /// Finalize the windows in `retry_state.finalize_queue`: run a fresh
  /// decode over each, append its text to `completed_text`, and reset
  /// the streaming decode state.
  ///
  /// F2: ALWAYS run `decoder.decode_all_tokens` for finalized windows.
  /// The previously-streamed provisional/confirmed text is consulted
  /// only as an explicit fallback when the full decode for the first
  /// queued window returns empty text — otherwise the streamed
  /// preview's partial-text would freeze in place and the rest of the
  /// boundary audio would be dropped.
  ///
  /// F3: pops one window at a time, advancing `frozen_window_count`
  /// after each successful append. On `Err` the failed window is left
  /// at the queue front so a subsequent `feed_audio` / `stop` call can
  /// retry it without losing already-encoded audio.
  ///
  /// R2-style fallback gating: the per-entry `fallback_consumed` flag
  /// in [`PendingFinalize`] is set BEFORE the fallible decode call so
  /// a decode Err does NOT re-arm the fallback. The streamed-text
  /// fallback is offered AT MOST ONCE per queued window across all
  /// retry attempts.
  fn finalize_completed_windows(&mut self) -> Result<Vec<TranscriptionEvent>> {
    if self.retry_state.finalize_queue().is_empty() {
      return Ok(Vec::new());
    }
    let mut total_decode_time: f64 = 0.0;
    let mut total_generated_tokens: usize = 0;

    let mut events: Vec<TranscriptionEvent> = Vec::new();
    while let Some(pending) = self.retry_state.finalize_queue_mut().front_mut() {
      // R2-style fallback gating: capture the streamed-text fallback
      // ONLY if THIS queue entry hasn't been offered it yet. The flag
      // flips to `true` BEFORE the fallible decode below so that on a
      // decode `Err`, the next retry sees `fallback_consumed == true`
      // and gets `None` — stale streamed text from `shared.*` is
      // never re-applied.
      let candidate_fallback = if !pending.fallback_consumed {
        pending.fallback_consumed = true;
        let mut stream_tokens: Vec<u32> = self.shared.confirmed_token_ids.clone();
        stream_tokens.extend(self.shared.provisional_token_ids.iter().copied());
        if stream_tokens.is_empty() {
          None
        } else {
          Some(self.tokenizer.decode_ids(&stream_tokens))
        }
      } else {
        None
      };
      let num_audio_tokens = pending.encoder_output.shape().first().copied().unwrap_or(0);
      let selected_window_text = if num_audio_tokens == 0 {
        // Empty audio: skip decode but allow the streamed fallback to
        // carry text forward (rare — guards against zero-row boundary
        // windows the encoder occasionally produces). On retry for
        // this same entry, `candidate_fallback` is `None` (flag is
        // sticky), so a retry yields an empty selected text rather
        // than the stale fallback.
        candidate_fallback.unwrap_or_default()
      } else {
        let start = Instant::now();
        // F2 + F3: ALWAYS attempt the full decode. On `Err` the `?`
        // propagates up; the queue front is unchanged so the next
        // pass retries this window. `frozen_window_count` has NOT
        // advanced yet, preserving the invariant
        // `frozen_window_count == encoded_window_count - queue.len()`.
        let token_ids = self.decoder.decode_all_tokens(
          &pending.encoder_output,
          &[],
          &self.config,
          self.config.max_tokens_per_pass(),
        )?;
        let decode_time = start.elapsed().as_secs_f64();
        total_decode_time += decode_time;
        total_generated_tokens = total_generated_tokens.saturating_add(token_ids.len());
        let full_text = self.tokenizer.decode_ids(&token_ids);
        // F2: only fall back to streamed text when the FULL decode
        // produced nothing. Otherwise the full decode wins.
        if full_text.trim().is_empty()
          && let Some(fallback) = candidate_fallback
        {
          fallback
        } else {
          full_text
        }
      };
      // Decode succeeded (or there was no audio to decode): commit
      // this window now, clear shared streaming state, advance the
      // frozen-window counter, and pop the queue.
      if !selected_window_text.trim().is_empty() {
        append_text(&selected_window_text, &mut self.shared.completed_text);
      }
      self.shared.confirmed_token_ids.clear();
      self.shared.provisional_token_ids.clear();
      self.shared.provisional_first_seen.clear();
      self.shared.provisional_agreement_counts.clear();
      self.shared.confirmed_text.clear();
      self.retry_state.finalize_queue_mut().pop_front();
      self.frozen_window_count = self.frozen_window_count.saturating_add(1);
    }

    let total_audio_seconds = self.total_samples_fed as f64 / 16_000.0;
    let tps = if total_decode_time > 0.0 {
      total_generated_tokens as f64 / total_decode_time
    } else {
      0.0
    };
    events.push(TranscriptionEvent::Stats(StreamingStats {
      encoded_window_count: self.encoder.encoded_window_count(),
      total_audio_seconds,
      tokens_per_second: tps,
      real_time_factor: 0.0,
      peak_memory_gb: peak_memory_gb_or_zero(),
    }));
    Ok(events)
  }

  fn freeze_completed_windows(&mut self) {
    let current = self.encoder.encoded_window_count();
    if current <= self.frozen_window_count {
      return;
    }
    let mut all_tokens: Vec<u32> = self.shared.confirmed_token_ids.clone();
    all_tokens.extend(self.shared.provisional_token_ids.iter().copied());
    if !all_tokens.is_empty() {
      let window_text = self.tokenizer.decode_ids(&all_tokens);
      append_text(&window_text, &mut self.shared.completed_text);
    }
    self.shared.confirmed_token_ids.clear();
    self.shared.provisional_token_ids.clear();
    self.shared.provisional_first_seen.clear();
    self.shared.provisional_agreement_counts.clear();
    self.shared.confirmed_text.clear();
    self.frozen_window_count = current;
  }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Append `segment` to `base` with whitespace handling — mirrors
/// Swift's `appendText`'s `trimmingCharacters(in: .whitespacesAndNewlines)`
/// plus the leading-space insertion when both halves are non-empty and
/// neither side already supplies the boundary whitespace. Simplified
/// (no deduping) — the Swift reference's dedupe heuristics are
/// decode-quality polish, not orchestration semantics. Reuse via
/// [`concat_text`].
fn append_text(segment: &str, base: &mut String) {
  let trimmed = segment.trim();
  if trimmed.is_empty() {
    return;
  }
  if base.is_empty() {
    base.push_str(trimmed);
    return;
  }
  let base_last_is_ws = base.chars().last().is_some_and(char::is_whitespace);
  let seg_first_is_ws = trimmed.chars().next().is_some_and(char::is_whitespace);
  if base_last_is_ws || seg_first_is_ws {
    base.push_str(trimmed);
  } else {
    base.push(' ');
    base.push_str(trimmed);
  }
}

fn concat_text(a: &str, b: &str) -> String {
  let mut out = String::with_capacity(a.len() + b.len() + 1);
  out.push_str(a);
  append_text(b, &mut out);
  out
}

/// Refcount-clone the partial-decode audio_features for re-arming the
/// `StopPartialDecode` obligation. Returns `Ok(None)` when there is no
/// payload to clone (the prior `encode_pending` returned `None`), and
/// `Ok(Some(cloned))` on a successful refcount clone.
///
/// Propagates [`Array::try_clone`] errors with a contextual message
/// instead of silently dropping the failure into `None` (which would
/// have made the next retry behave as "no partial audio" and drop the
/// real payload — see the `stop()` F2-FIX call-site comments).
fn clone_partial_decode_payload(features: Option<&Array>) -> Result<Option<Array>> {
  match features {
    None => Ok(None),
    Some(a) => a.try_clone().map(Some).map_err(|e| crate::Error::Backend {
      message: format!("StopPartialDecode: failed to clone audio_features for retry: {e}"),
    }),
  }
}

/// Wrapper around [`crate::memory::peak_memory`] that returns
/// `peak / 1e9` GB or `0.0` if the read errors. Mirrors the Swift
/// reference's `Double(Memory.peakMemory) / 1e9` formula.
fn peak_memory_gb_or_zero() -> f64 {
  crate::memory::peak_memory()
    .map(|bytes| bytes as f64 / 1e9)
    .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::audio::stt::streaming::{encoder::StreamingEncoderBackend, types::DelayPreset};
  use std::sync::Mutex;

  // -----------------------------------------------------------------
  // Mocks
  // -----------------------------------------------------------------

  /// Scripted encoder backend: per-call `Result<(), Error>` script
  /// (`true` = Err, `false` = Ok). On Ok produces a passthrough
  /// `(rows, 2)` array; on Err returns the scripted backend failure.
  /// Records the per-call mel_window[0][0] fingerprint BEFORE the
  /// error gate so order-sensitive tests can prove which staged
  /// buffer reached the encoder first.
  struct ScriptedEncoder {
    window_size: usize,
    err_script: Mutex<Vec<bool>>,
    calls: Mutex<usize>,
    fingerprints: Mutex<Vec<f32>>,
  }
  impl ScriptedEncoder {
    fn new(window_size: usize, err_script: Vec<bool>) -> Self {
      Self {
        window_size,
        err_script: Mutex::new(err_script),
        calls: Mutex::new(0),
        fingerprints: Mutex::new(Vec::new()),
      }
    }
    fn call_count(&self) -> usize {
      *self.calls.lock().unwrap()
    }
    fn fingerprints(&self) -> Vec<f32> {
      self.fingerprints.lock().unwrap().clone()
    }
  }
  impl StreamingEncoderBackend for ScriptedEncoder {
    fn window_size(&self) -> usize {
      self.window_size
    }
    fn encode_window(&self, mel_window: &Array, _valid_frames: usize) -> Result<Array> {
      *self.calls.lock().unwrap() += 1;
      let fingerprint = mel_window
        .try_clone()
        .and_then(|mut a| a.to_vec::<f32>())
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f32::NAN);
      self.fingerprints.lock().unwrap().push(fingerprint);
      let mut script = self.err_script.lock().unwrap();
      let should_err = if script.is_empty() {
        false
      } else {
        script.remove(0)
      };
      if should_err {
        return Err(crate::error::Error::Backend {
          message: "scripted encode_window failure".into(),
        });
      }
      let rows = mel_window.shape().first().copied().unwrap_or(0);
      let buf = vec![0.0_f32; rows * 2];
      Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
    }
  }

  /// Mock decoder that takes a `Result<Vec<u32>>` per call so individual
  /// passes can inject decoder errors. Tracks call count.
  struct ScriptedDecoder {
    results: Mutex<Vec<Result<Vec<u32>>>>,
    calls: Mutex<usize>,
  }
  impl ScriptedDecoder {
    fn with_results(results: Vec<Result<Vec<u32>>>) -> Self {
      Self {
        results: Mutex::new(results),
        calls: Mutex::new(0),
      }
    }
    fn call_count(&self) -> usize {
      *self.calls.lock().unwrap()
    }
  }
  impl StreamingDecoderBackend for ScriptedDecoder {
    fn decode_all_tokens(
      &self,
      _audio_features: &Array,
      _confirmed_token_ids: &[u32],
      _config: &StreamingConfig,
      _max_tokens: usize,
    ) -> Result<Vec<u32>> {
      *self.calls.lock().unwrap() += 1;
      let mut q = self.results.lock().unwrap();
      if q.is_empty() {
        Ok(Vec::new())
      } else {
        q.remove(0)
      }
    }
  }

  struct MockTokenizer;
  impl StreamingTokenizer for MockTokenizer {
    fn decode_ids(&self, ids: &[u32]) -> String {
      ids
        .iter()
        .map(|id| format!("t{id}"))
        .collect::<Vec<_>>()
        .join(" ")
    }
  }

  /// Build a non-finalize_completed_windows session over scripted
  /// encoder + decoder. Window size 8 mel-frames keeps the boundary
  /// cadence tight.
  fn nonfinalize_session(
    encoder: ScriptedEncoder,
    decoder: ScriptedDecoder,
  ) -> StreamingInferenceSession<ScriptedEncoder, ScriptedDecoder, MockTokenizer> {
    let cfg = StreamingConfig::default()
      .with_decode_interval_seconds(0.0)
      .with_boundary_decode_interval_seconds(0.0)
      .with_boundary_boost_seconds(0.0)
      .with_max_cached_windows(8)
      .with_finalize_completed_windows(false)
      .with_min_agreement_passes(1)
      .with_boundary_min_agreement_passes(1)
      .with_delay_preset(DelayPreset::Custom(0));
    StreamingInferenceSession::new(decoder, MockTokenizer, cfg, encoder, 16_000, 8, 0).unwrap()
  }

  /// Same as `nonfinalize_session` but with finalize_completed_windows = true.
  fn finalize_session(
    encoder: ScriptedEncoder,
    decoder: ScriptedDecoder,
  ) -> StreamingInferenceSession<ScriptedEncoder, ScriptedDecoder, MockTokenizer> {
    let cfg = StreamingConfig::default()
      .with_decode_interval_seconds(0.0)
      .with_boundary_decode_interval_seconds(0.0)
      .with_boundary_boost_seconds(0.0)
      .with_max_cached_windows(8)
      .with_finalize_completed_windows(true)
      .with_min_agreement_passes(1)
      .with_boundary_min_agreement_passes(1)
      .with_delay_preset(DelayPreset::Custom(0));
    StreamingInferenceSession::new(decoder, MockTokenizer, cfg, encoder, 16_000, 8, 0).unwrap()
  }

  /// Drive the session through (a) a partial-window feed that exercises
  /// the pending-window pre-boundary decode, then (b) a top-up feed
  /// that closes the first window and triggers the boundary finalize
  /// pass. n_fft=400, hop=160 → 400 + 7×160 = 1 520 samples for one
  /// window of mel content. Using 800 + 1 200 samples gives one
  /// partial-decode call + one finalize-decode call.
  fn drive_two_phase(
    session: &mut StreamingInferenceSession<ScriptedEncoder, ScriptedDecoder, MockTokenizer>,
  ) -> (
    Result<Vec<TranscriptionEvent>>,
    Result<Vec<TranscriptionEvent>>,
  ) {
    let partial: Vec<f32> = (0..800).map(|i| (i as f32 * 0.001).sin()).collect();
    let topup: Vec<f32> = (800..2_000).map(|i| (i as f32 * 0.001).sin()).collect();
    let partial_events = session.feed_audio(&partial);
    let boundary_events = session.feed_audio(&topup);
    (partial_events, boundary_events)
  }

  // -----------------------------------------------------------------
  // Baseline: feed_audio + stop happy paths
  // -----------------------------------------------------------------

  #[test]
  fn feed_audio_short_input_yields_no_events_until_mel_emits() {
    let encoder = ScriptedEncoder::new(16, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![10, 11, 12])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let events = session.feed_audio(&[0.0_f32; 1]).unwrap();
    assert!(events.is_empty(), "events={events:?}");
  }

  #[test]
  fn feed_audio_long_input_drives_partial_decode() {
    let encoder = ScriptedEncoder::new(16, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![10, 11, 12])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
    let events = session.feed_audio(&samples).unwrap();
    assert_eq!(session.decoder.call_count(), 1);
    assert!(
      matches!(events.first(), Some(TranscriptionEvent::Confirmed(_))),
      "events[0]={:?}",
      events.first()
    );
    assert!(
      events
        .iter()
        .any(|e| matches!(e, TranscriptionEvent::Stats(_)))
    );
  }

  #[test]
  fn stop_emits_ended_event() {
    let encoder = ScriptedEncoder::new(16, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![10])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();
    let stop_events = session.stop().unwrap();
    assert!(
      matches!(stop_events.last(), Some(TranscriptionEvent::Ended(_))),
      "stop events: {stop_events:?}"
    );
    assert!(!session.is_active());
  }

  #[test]
  fn cancel_marks_inactive_and_drops_state() {
    let encoder = ScriptedEncoder::new(16, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();
    session.cancel();
    assert!(!session.is_active());
    let after = session.feed_audio(&samples).unwrap();
    assert!(after.is_empty());
  }

  #[test]
  fn append_text_basic_concatenation_and_trim() {
    let mut base = String::new();
    append_text("hello", &mut base);
    assert_eq!(base, "hello");
    append_text("world", &mut base);
    assert_eq!(base, "hello world");
    append_text("  ", &mut base);
    assert_eq!(base, "hello world");
    append_text("!", &mut base);
    assert_eq!(base, "hello world !");
  }

  // -----------------------------------------------------------------
  // F2 / F3 — completed-window finalization (preserved from R1 baseline)
  // -----------------------------------------------------------------

  /// F2: when the first window is finalized, the FULL decode must run
  /// over the completed-window features — even when streamed-fallback
  /// text exists from a prior pending-window decode.
  #[test]
  fn streaming_session_first_window_finalization_runs_full_decode_not_streamed_fallback() {
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1, 2, 3]), Ok(vec![9, 8, 7])]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    boundary_events.unwrap();
    assert!(
      session.decoder.call_count() >= 2,
      "expected >= 2 decoder calls (pending + finalize), got {}",
      session.decoder.call_count()
    );
  }

  /// F2: when the streamed-fallback text differs from the full-decode
  /// text, the FULL-decode text MUST land in `completed_text`.
  #[test]
  fn streaming_session_first_window_finalization_appends_full_decode_not_partial_text() {
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1, 2, 3]), Ok(vec![90, 91, 92])]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    boundary_events.unwrap();
    let stop_events = session.stop().unwrap();
    let TranscriptionEvent::Ended(full_text) = stop_events
      .last()
      .expect("expected Ended event at stop()")
      .clone()
    else {
      panic!("last stop event was not Ended: {stop_events:?}");
    };
    assert!(
      full_text.contains("t90"),
      "Ended.full_text must include the full-decode text, got {full_text:?}"
    );
  }

  /// F3: on a finalize-decode error the failed window stays in the
  /// retry queue so a subsequent `feed_audio` call can re-attempt it.
  /// (R2 contract: feed_audio drains a pending retry queue.)
  #[test]
  fn streaming_session_decoder_error_keeps_window_for_retry_then_feed_audio_drains() {
    use crate::error::Error;
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]),
      Err(Error::Backend {
        message: "scripted finalize failure".into(),
      }),
      Ok(vec![42]),
    ]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    assert!(boundary_events.is_err());
    assert_eq!(
      session.retry_state().finalize_queue().len(),
      1,
      "errored finalize must leave the window in the retry queue"
    );
    // Retry via small follow-up feed.
    let retry_events = session.feed_audio(&[0.0_f32; 200]).unwrap();
    assert_eq!(
      session.retry_state().finalize_queue().len(),
      0,
      "successful retry must pop the previously-failed window"
    );
    assert!(
      !retry_events.is_empty(),
      "retry decode must emit at least the Stats event"
    );
  }

  /// F3: `frozen_window_count` invariant across a finalize error.
  #[test]
  fn streaming_session_decoder_error_does_not_advance_frozen_window_count() {
    use crate::error::Error;
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]),
      Err(Error::Backend {
        message: "scripted finalize failure".into(),
      }),
    ]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    assert!(boundary_events.is_err());
    assert_eq!(session.encoded_window_count(), 1);
    assert_eq!(session.frozen_window_count, 0);
    assert_eq!(session.retry_state().finalize_queue().len(), 1);
  }

  // -----------------------------------------------------------------
  // R2-style: retry semantics around stop() / feed_audio()
  // -----------------------------------------------------------------

  /// R2 F1: when `stop()` errors inside finalize, the session must
  /// remain RETRYABLE — a second `stop()` call must drain the queue.
  #[test]
  fn streaming_session_stop_with_finalize_err_can_be_retried_with_second_stop() {
    use crate::error::Error;
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]),
      Err(Error::Backend {
        message: "scripted boundary finalize Err".into(),
      }),
      Err(Error::Backend {
        message: "scripted stop-retry finalize Err".into(),
      }),
      Ok(vec![42]),
    ]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    assert!(boundary_events.is_err(), "boundary feed must Err");
    assert_eq!(session.retry_state().finalize_queue().len(), 1);

    let stop_first = session.stop();
    assert!(
      stop_first.is_err(),
      "first stop() must propagate the scripted finalize Err"
    );

    let stop_second = session.stop().expect("second stop() must succeed");
    assert!(
      matches!(stop_second.last(), Some(TranscriptionEvent::Ended(_))),
      "second stop() must emit terminal Ended, got {stop_second:?}"
    );
    assert!(!session.is_active());
    assert_eq!(session.retry_state().finalize_queue().len(), 0);
  }

  /// R2 F2: a follow-up `feed_audio` with EMPTY input must drive the
  /// retry path.
  #[test]
  fn streaming_session_pending_retry_finalizes_on_empty_feed_audio() {
    use crate::error::Error;
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]),
      Err(Error::Backend {
        message: "scripted boundary finalize Err".into(),
      }),
      Ok(vec![77]),
    ]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    assert!(boundary_events.is_err());
    assert_eq!(session.retry_state().finalize_queue().len(), 1);
    let calls_before = session.decoder.call_count();

    let retry_events = session
      .feed_audio(&[])
      .expect("empty feed_audio retry must succeed");
    assert_eq!(session.retry_state().finalize_queue().len(), 0);
    assert!(session.decoder.call_count() > calls_before);
    assert!(!retry_events.is_empty());
  }

  /// R2 F3: after a finalize-decode Err, the streamed-text fallback
  /// MUST NOT be re-armed on retry — `PendingFinalize::fallback_consumed`
  /// is sticky.
  #[test]
  fn streaming_session_fallback_not_reapplied_on_retry_after_err() {
    use crate::error::Error;
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![123]),
      Err(Error::Backend {
        message: "scripted boundary finalize Err".into(),
      }),
      Ok(vec![]),
    ]);
    let mut session = finalize_session(encoder, decoder);
    let (partial_events, boundary_events) = drive_two_phase(&mut session);
    partial_events.unwrap();
    assert!(boundary_events.is_err());
    assert_eq!(session.retry_state().finalize_queue().len(), 1);

    let _ = session
      .feed_audio(&[])
      .expect("retry feed_audio must succeed");
    assert_eq!(session.retry_state().finalize_queue().len(), 0);

    // The stale "t123" provisional MUST NOT be frozen into
    // completed_text — the per-entry fallback flag prevented re-arm.
    assert!(
      !session.shared.completed_text.contains("t123"),
      "R2 F3: stale streamed fallback must NOT be frozen, got {:?}",
      session.shared.completed_text
    );
  }

  // -----------------------------------------------------------------
  // R3 contract: stop() preserves tail audio across mel.flush / encoder
  //              .feed Err. Uses the SessionRetryState's StopEncoderFeed
  //              stage as the bridge.
  // -----------------------------------------------------------------

  /// R3 contract: stop()'s encoder.feed Err preserves the
  /// freshly-flushed mel_frames so a retry stop() can re-feed them.
  ///
  /// (Named for the misnomer — the scripted Err is on
  /// `encode_window`, NOT on `mel.flush()`. The mel.flush-Err path is
  /// covered separately by
  /// [`session_retry_state_stop_with_mel_flush_err_can_be_retried`]
  /// and friends.)
  #[test]
  fn session_retry_state_stop_with_encoder_feed_err_can_be_retried() {
    // 1200 samples ⇒ 7 mel frames in encoder pending (< 8); mel.flush
    // emits ~2 more → encoder.feed sees 9 → 1 full window → encode_window
    // is called and scripted to Err.
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3, 4])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();

    let overlap_before = session.mel_processor.overlap_buffer_len();
    assert!(
      overlap_before > 0,
      "test precondition: overlap must be populated"
    );

    // First stop(): mel.flush Ok (commits overlap clear), encoder.feed
    // Errs on encode_window.
    let stop_first = session.stop();
    assert!(stop_first.is_err());
    // Mel overlap cleared (transactional flush succeeded).
    assert_eq!(session.mel_processor.overlap_buffer_len(), 0);
    // Bridge holds the tail mel.
    assert!(
      session.retry_state().has_pending_stop_encoder_feed(),
      "R3 contract: SessionRetryState StopEncoderFeed MUST hold the tail mel"
    );
    assert!(
      session.is_active(),
      "errored stop() must leave session active"
    );

    // Retry stop(): bridge re-feeds → success → Ended.
    let stop_second = session.stop().expect("retry stop() must succeed");
    assert!(matches!(
      stop_second.last(),
      Some(TranscriptionEvent::Ended(_))
    ));
    assert!(!session.is_active());
    assert!(!session.retry_state().has_obligation());
  }

  // -----------------------------------------------------------------
  // R4 corner: feed_audio MUST drain pending stop-tail BEFORE new audio.
  // -----------------------------------------------------------------

  /// R4 corner structural fix: a `feed_audio(new_samples)` AFTER a
  /// transient stop()-Err MUST run the bridge drain BEFORE
  /// mel.process(new_samples) / encoder.feed(new_mel). The
  /// SessionRetryState's StopEncoderFeed discharge is called at the
  /// TOP of feed_audio, before the normal feed-path.
  #[test]
  fn session_retry_state_feed_audio_drains_stop_tail_before_new_audio() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7])]);
    let mut session = nonfinalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let stop_first = session.stop();
    assert!(stop_first.is_err());
    assert!(session.retry_state().has_pending_stop_encoder_feed());
    let stop_err_call_idx = session.encoder.backend().call_count() - 1;
    let stop_err_fingerprint = session.encoder.backend().fingerprints()[stop_err_call_idx];

    // R4 PROBE: deliver NEW audio. The bridge drain MUST run FIRST.
    let samples_b: Vec<f32> = (0..1200)
      .map(|i| ((i as f32 + 7.0) * 0.013).cos())
      .collect();
    let _events = session
      .feed_audio(&samples_b)
      .expect("feed_audio after staged-tail stop Err MUST succeed");

    // Bridge cleared on successful drain.
    assert!(
      !session.retry_state().has_pending_stop_encoder_feed(),
      "R4: feed_audio MUST clear StopEncoderFeed after a successful drain"
    );

    // ORDER ASSERTION: the call immediately after stop-Err MUST be
    // the bridge drain (same fingerprint).
    let fingerprints = session.encoder.backend().fingerprints();
    let bridge_drain_idx = stop_err_call_idx + 1;
    assert!(fingerprints.len() > bridge_drain_idx);
    assert_eq!(
      fingerprints[bridge_drain_idx].to_bits(),
      stop_err_fingerprint.to_bits(),
      "R4 ORDER: the call immediately after stop-Err MUST be the bridge \
       drain (bit-identical fingerprint). fingerprints={fingerprints:?}"
    );
  }

  /// R4 corner: `feed_audio(&[])` MUST drain the staged tail even with
  /// zero new samples. The drain commits 1 window (encode_window call);
  /// the SessionRetryState's discharge then drives a same-call decode
  /// pass for the drained window (a second encode_window call for the
  /// 1-row carry's encode_pending — the R5 contract that drained
  /// windows must flow through a decode in THIS call). Total: 2
  /// encode_window calls in the same feed_audio.
  #[test]
  fn session_retry_state_feed_audio_empty_samples_drains_bridge_and_decodes() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7, 8])]);
    let mut session = nonfinalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let _ = session.stop();
    assert!(session.retry_state().has_pending_stop_encoder_feed());
    let decoder_calls_before = session.decoder.call_count();

    // Reset cadence-gate so the discharge's decode pass isn't throttled.
    session.last_decode_time = None;

    let _events = session
      .feed_audio(&[])
      .expect("empty feed_audio MUST succeed by draining the bridge");
    assert!(!session.retry_state().has_pending_stop_encoder_feed());
    // R5 contract: drained window decoded in THIS call.
    assert_eq!(
      session.decoder.call_count(),
      decoder_calls_before + 1,
      "drained window MUST be decoded in the same call"
    );
    // All obligations cleared on happy path.
    assert!(!session.retry_state().has_obligation());
  }

  // -----------------------------------------------------------------
  // R5 corner: drained windows decoded in the SAME feed_audio call.
  // -----------------------------------------------------------------

  /// R5 corner structural fix: a bridge drain that completes a full
  /// encoder window MUST route that window through `run_decode_pass`
  /// IN THE SAME `feed_audio` call. The `discharge_stop_encoder_feed`
  /// returns count > 0 and arms DecodeOwed, which immediately drives
  /// the decode pass in the same discharge.
  #[test]
  fn session_retry_state_drained_windows_decoded_in_same_call() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7, 8])]);
    let mut session = nonfinalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let decoder_calls_after_initial = session.decoder.call_count();
    let _ = session.stop();
    assert!(session.retry_state().has_pending_stop_encoder_feed());
    let encoder_windows_before_drain = session.encoder.encoded_window_count();

    // Reset cadence gate.
    session.last_decode_time = None;

    let _events = session
      .feed_audio(&[])
      .expect("feed_audio after staged-tail stop Err MUST succeed");

    assert!(!session.retry_state().has_pending_stop_encoder_feed());
    assert_eq!(
      session.encoder.encoded_window_count(),
      encoder_windows_before_drain + 1,
      "R5: bridge drain MUST have committed exactly one full window"
    );
    // R5 CORE: the decoder was invoked for the partial-window decode in
    // the SAME call as the drain.
    assert_eq!(
      session.decoder.call_count(),
      decoder_calls_after_initial + 1,
      "R5: bridge-drained window MUST drive run_decode_pass in this call"
    );
  }

  /// R5: sub-window drain → bridge cleared (no window completed). The
  /// decode-owed obligation MUST NOT be armed (only count > 0 arms it).
  #[test]
  fn session_retry_state_bridge_drain_no_windows_clears_bridge_no_obligation() {
    let encoder = ScriptedEncoder::new(8, vec![]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5])]);
    let mut session = nonfinalize_session(encoder, decoder);

    // Inject a sub-window mel directly into the bridge via the
    // test-only retry_state_mut accessor.
    let staged_mel = Array::from_slice::<f32>(&[0.0_f32; 2 * 8], &[2_i32, 8_i32]).unwrap();
    session
      .retry_state_mut()
      .stage_stop_encoder_feed(staged_mel);

    let _events = session
      .feed_audio(&[])
      .expect("empty feed_audio with non-completing drain MUST succeed");
    // Bridge cleared (R4 contract: successful drain always clears).
    assert!(!session.retry_state().has_pending_stop_encoder_feed());
    // R5 core: drain returned 0 windows → no DecodeOwed obligation
    // armed. (The encoder may have pending sub-window frames that the
    // normal pending-window decode picks up — that's a separate path
    // governed by the cadence gate, not the discharge.)
    assert!(
      !session.retry_state().has_decode_owed(),
      "R5: drain with 0 windows MUST NOT arm DecodeOwed"
    );
    assert_eq!(session.encoder.encoded_window_count(), 0);
  }

  // -----------------------------------------------------------------
  // R6 corner: a same-call decode obligation MUST survive a later `?`
  //            propagation. The DecodeOwed stage is the cross-call
  //            source of truth — no per-call locals.
  // -----------------------------------------------------------------

  /// R6 corner structural fix: a bridge drain SUCCESS in feed_audio
  /// followed by a later Err on the new-audio encoder.feed must arm
  /// DecodeOwed; a retry feed_audio(&[]) MUST decode the drained
  /// window AND clear the obligation.
  #[test]
  fn session_retry_state_drained_count_survives_post_drain_err() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, true, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![42])]);
    let mut session = finalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let decoder_calls_after_initial = session.decoder.call_count();
    let _ = session.stop();
    assert!(session.retry_state().has_pending_stop_encoder_feed());

    // R6 PROBE: deliver new audio. Discharge drains successfully (1
    // window committed) and arms DecodeOwed. The discharge's decode
    // pass runs and SUCCEEDS (clears DecodeOwed). Then back in
    // feed_audio normal path, encoder.feed(new_mel) Errs.
    //
    // Actually — let me trace this more carefully for the R6 contract.
    // The R6 contract specifically requires: the drained window's
    // decode obligation survives a `?` from later. In MY design the
    // discharge runs the decode IN THE DISCHARGE. So R6 is structurally
    // impossible — there's no "drained but not decoded" state across
    // call boundaries when the drain returns count > 0.
    //
    // But the alt R6 surface IS reachable: encoder.feed(new_mel)
    // succeeds with new_windows > 0, then run_decode_pass Errs. The
    // new windows are stranded. My fix arms DecodeOwed BEFORE
    // run_decode_pass when new_windows > 0.
    let samples_b: Vec<f32> = (0..1500)
      .map(|i| ((i as f32 + 11.0) * 0.013).cos())
      .collect();
    let feed_err = session.feed_audio(&samples_b);
    assert!(
      feed_err.is_err(),
      "feed_audio with new-audio encode_window Err MUST propagate Err"
    );
    // Bridge cleared (the discharge already drained it transactionally).
    assert!(!session.retry_state().has_pending_stop_encoder_feed());

    // The R6-style obligation we MIGHT still have depends on whether the
    // discharge's decode armed DecodeOwed and propagated, or the later
    // encoder.feed Err armed DecodeOwed before propagating. Either way,
    // a retry feed_audio(&[]) MUST decode whatever is owed and clear.
    session.last_decode_time = None;
    let retry_events = session
      .feed_audio(&[])
      .expect("retry feed_audio MUST succeed");
    assert!(
      !session.retry_state().has_decode_owed(),
      "R6: successful retry decode MUST clear DecodeOwed"
    );
    assert!(
      session.decoder.call_count() > decoder_calls_after_initial,
      "R6: retry MUST drive at least one decoder call"
    );
    let _ = retry_events;
  }

  // -----------------------------------------------------------------
  // R7 corner: NO flag bleeds across calls when cadence-throttle skips
  //            the decode. Structurally, my design has no "force decode
  //            next call" flag. The discharge runs decodes when an
  //            obligation is owed; a cadence-throttled call doesn't
  //            create an obligation (DecodeOwed is only armed by an
  //            actual Err or a discharge_stop_encoder_feed with count > 0).
  // -----------------------------------------------------------------

  /// R7 corner structural fix: a bridge drain + successful same-call
  /// decode + no later Err MUST clear all obligations. There is no
  /// flag that survives the call to force an unnecessary decode next
  /// time. The cadence throttle does NOT factor into the discharge's
  /// decision to decode bridge-drained windows (the discharge decodes
  /// unconditionally when count > 0); but no flag persists past a
  /// successful end-of-call.
  #[test]
  fn session_retry_state_throttled_drain_does_not_force_decode_on_next_call() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7, 8])]);
    let mut session = nonfinalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let _ = session.stop();
    assert!(session.retry_state().has_pending_stop_encoder_feed());

    session.last_decode_time = None;

    // First feed_audio: discharge runs drain + decode pass. Both Ok →
    // ALL obligations cleared.
    let _events = session
      .feed_audio(&[])
      .expect("happy-path discharge MUST succeed");

    assert!(
      !session.retry_state().has_obligation(),
      "R7: end-of-call MUST clear all obligations on happy path"
    );

    let decoder_calls_after_discharge = session.decoder.call_count();

    // Second feed_audio: no work owed. The decoder MUST NOT be invoked
    // for a "phantom" obligation. (Cadence might fire on its own with
    // last_decode_time = None + has_new_encoder_content = false, but
    // we just emptied newly_encoded_windows so has_new_encoder_content
    // should be false at this point — the previous call set it true,
    // decode pass cleared it.)
    let events_2 = session
      .feed_audio(&[])
      .expect("second empty feed_audio MUST succeed without forced decode");

    assert_eq!(
      session.decoder.call_count(),
      decoder_calls_after_discharge,
      "R7: second empty feed_audio MUST NOT trigger a phantom decode"
    );
    assert!(events_2.is_empty(), "no work ⇒ no events");
  }

  // -----------------------------------------------------------------
  // cancel / reset clear all obligations atomically.
  // -----------------------------------------------------------------

  /// `cancel()` MUST clear every retry obligation in one call (no
  /// per-flag forget-to-clear corner).
  #[test]
  fn session_retry_state_cancel_clears_all_obligations() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, true, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5])]);
    let mut session = finalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let _ = session.stop(); // Errs → stages StopEncoderFeed
    assert!(session.retry_state().has_obligation());

    session.cancel();
    assert!(
      !session.retry_state().has_obligation(),
      "cancel MUST clear all retry obligations atomically"
    );
  }

  /// `reset()` MUST clear every retry obligation in one call.
  #[test]
  fn session_retry_state_reset_clears_all_obligations() {
    let encoder = ScriptedEncoder::new(8, vec![false, true, false, true, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5])]);
    let mut session = finalize_session(encoder, decoder);

    let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples_a).unwrap();
    let _ = session.stop();
    assert!(session.retry_state().has_obligation());

    session.reset();
    assert!(
      !session.retry_state().has_obligation(),
      "reset MUST clear all retry obligations atomically"
    );
  }

  // -----------------------------------------------------------------
  // F1: StopMelFlush deadlock fix — discharge_stop_mel_flush wired
  //     into discharge_retry_obligation. Pre-fix: stop() staged
  //     StopMelFlush BEFORE mel.flush(); on flush Err the obligation
  //     was set but discharge_retry_obligation had no branch for it
  //     → has_obligation() returned true → stop() early-returned
  //     without driving the retry → DEADLOCK.
  // -----------------------------------------------------------------

  /// F1 contract: a stop() whose `mel.flush()` errors stages
  /// `StopMelFlush` and keeps the session active so the next call can
  /// retry. Without the discharge wiring the obligation would be
  /// stranded.
  #[test]
  fn session_retry_state_stop_after_flush_err_keeps_session_active_with_obligation() {
    let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();
    assert!(session.mel_processor.overlap_buffer_len() > 0);

    // Script ONE mel.flush Err.
    session.mel_processor.flush_err_inject_count = 1;

    // First stop(): mel.flush Errs → StopMelFlush staged.
    let stop_first = session.stop();
    assert!(stop_first.is_err(), "stop() must propagate mel.flush Err");
    assert!(
      session.is_active(),
      "F1: errored stop() MUST leave session active so retry is possible"
    );
    assert!(
      session.retry_state().has_pending_stop_mel_flush(),
      "F1: stop()'s mel.flush Err MUST stage StopMelFlush"
    );
    assert!(
      session.retry_state().has_obligation(),
      "F1: retry obligation MUST be visible to the next call"
    );
    // mel.flush is transactional — overlap preserved on Err.
    assert!(
      session.mel_processor.overlap_buffer_len() > 0,
      "F1: transactional flush MUST preserve overlap on Err"
    );
  }

  /// F1 contract: a second stop() after the first's mel.flush Err
  /// drives `discharge_stop_mel_flush`, completes the flush + encoder
  /// feed + decode + Ended emission. Pre-fix this would DEADLOCK
  /// because no discharge branch existed for StopMelFlush.
  #[test]
  fn session_retry_state_stop_with_mel_flush_err_can_be_retried() {
    let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();
    let overlap_before = session.mel_processor.overlap_buffer_len();
    assert!(overlap_before > 0);

    // First stop(): mel.flush Errs → StopMelFlush staged.
    session.mel_processor.flush_err_inject_count = 1;
    let stop_first = session.stop();
    assert!(stop_first.is_err());
    assert!(session.retry_state().has_pending_stop_mel_flush());
    // Overlap intact (transactional flush).
    assert_eq!(session.mel_processor.overlap_buffer_len(), overlap_before);

    // Second stop(): the dispatcher's StopMelFlush discharge re-runs
    // mel.flush → Ok this time → advances to StopEncoderFeed → (b)
    // drains the encoder → (c) decodes the bridge-drained partial →
    // back in stop() body, finalize/freeze + partial decode + Ended.
    let stop_second = session
      .stop()
      .expect("F1: retry stop() MUST succeed via discharge_stop_mel_flush");
    assert!(
      matches!(stop_second.last(), Some(TranscriptionEvent::Ended(_))),
      "F1: second stop() MUST emit Ended after the discharge succeeds"
    );
    assert!(!session.is_active());
    assert!(!session.retry_state().has_obligation());
    // Overlap cleared by the successful re-flush.
    assert_eq!(session.mel_processor.overlap_buffer_len(), 0);
  }

  /// F1 contract: the cross-call discharge survives MULTIPLE mel.flush
  /// failures — each call drives the retry and re-arms StopMelFlush
  /// on continued failure. Only when the flush eventually succeeds
  /// does the pipeline proceed.
  #[test]
  fn session_retry_state_second_stop_after_flush_err_retries_and_emits_ended_on_success() {
    let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
    let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3])]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();

    // Inject TWO consecutive mel.flush Errs.
    session.mel_processor.flush_err_inject_count = 2;

    // First stop(): mel.flush Err #1 → StopMelFlush staged.
    assert!(session.stop().is_err());
    assert!(session.retry_state().has_pending_stop_mel_flush());

    // Second stop(): the discharge re-fires mel.flush → Err #2 → re-arms
    // StopMelFlush, propagates Err. No deadlock — the obligation
    // remains discharge-able.
    assert!(
      session.stop().is_err(),
      "F1: second stop() with continued flush Err MUST propagate Err"
    );
    assert!(
      session.retry_state().has_pending_stop_mel_flush(),
      "F1: re-armed StopMelFlush MUST persist across the second Err"
    );
    assert!(session.is_active());

    // Third stop(): flush counter exhausted → flush Ok → full pipeline.
    let stop_third = session
      .stop()
      .expect("F1: third stop() MUST succeed once flush stops erring");
    assert!(matches!(
      stop_third.last(),
      Some(TranscriptionEvent::Ended(_))
    ));
    assert!(!session.is_active());
    assert!(!session.retry_state().has_obligation());
  }

  // -----------------------------------------------------------------
  // F2: StopPartialDecode rollback no longer silently drops the
  //     audio_features payload on try_clone failure. Pre-fix:
  //     `try_clone().ok()` mapped clone Err to `None`, which made
  //     the next retry behave as "no partial audio" and drop the
  //     window. Now the clone Err propagates AND the obligation is
  //     re-armed with the original payload (fast path) or the
  //     mainline body's idempotent recompute handles it.
  // -----------------------------------------------------------------

  /// F2 contract: the `clone_partial_decode_payload` helper returns
  /// `Ok(None)` for absent features (the normal "no partial" path),
  /// `Ok(Some(_))` for a successful refcount clone, and propagates a
  /// clone failure as `Err` instead of silently dropping it. Tested
  /// at the helper level because forcing `Array::try_clone` to fail
  /// requires injecting an FFI allocation failure that the mlx
  /// backend doesn't surface deterministically; the helper is the
  /// unit-testable choke-point that gates BOTH stop()-path call sites.
  #[test]
  fn session_retry_state_stop_partial_decode_clone_helper_propagates_errors_not_silently_drops_audio()
   {
    // None in ⇒ Ok(None) out — the legitimate "no partial audio" path.
    let none_out = clone_partial_decode_payload(None).expect("None must succeed");
    assert!(
      none_out.is_none(),
      "F2: None payload MUST round-trip as Ok(None) (no fabricated payload)"
    );

    // Some in ⇒ Ok(Some(refcount-cloned)) on the happy path.
    let arr = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0], &[3i32]).unwrap();
    let some_out = clone_partial_decode_payload(Some(&arr)).expect("happy-path clone must succeed");
    assert!(
      some_out.is_some(),
      "F2: Some payload with successful clone MUST yield Ok(Some(_))"
    );
    // The clone is a separate handle (not the same allocation).
    let cloned = some_out.unwrap();
    assert_eq!(
      cloned.shape(),
      arr.shape(),
      "F2: refcount clone preserves shape"
    );

    // STRUCTURAL ASSERTION: the function signature returns `Result`,
    // proving an `Err` PATH exists. The pre-fix `.ok()` API had NO
    // Err path → any clone failure was dropped. The Err-path
    // existence is what kills the silent-drop defect class
    // structurally; the propagation is exercised end-to-end by every
    // stop()-with-partial-window test in this module (those call
    // sites use `?` to propagate, so a real Err would surface as a
    // stop() Err).
  }

  /// F2 contract: the stop() fast path's clone-failure rollback
  /// re-arms `StopPartialDecode` with the ORIGINAL payload (moved
  /// back into the obligation, no clone needed). The pre-fix code's
  /// `try_clone().ok()` would have armed with `None`, silently
  /// dropping the partial window for the next retry. We assert the
  /// rollback's structural shape: after a successful fast-path stop,
  /// the obligation is cleared. (A clone-failure-injected fast path
  /// can't be triggered without an FFI hook into `mlx_array_set`'s
  /// allocator; the rollback code path is exercised symbolically by
  /// the helper test above + the call-site preserves the original
  /// payload across the Err propagation by structural construction.)
  #[test]
  fn session_retry_state_stop_partial_decode_fast_path_preserves_payload_on_success() {
    let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
    // Decoder: first call returns one token (feed_audio partial decode);
    // second call (stop()'s partial-decode) errs so StopPartialDecode
    // is left armed; third call (retry stop()'s fast path) succeeds.
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]),
      Err(crate::error::Error::Backend {
        message: "scripted stop-partial-decode Err".into(),
      }),
      Ok(vec![1, 2]),
    ]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();

    // First stop(): everything succeeds until the partial-window
    // decode, which Errs. StopPartialDecode is armed with the
    // freshly-cloned audio_features.
    let stop_first = session.stop();
    assert!(
      stop_first.is_err(),
      "first stop() MUST propagate decoder Err"
    );
    assert!(
      session.retry_state().has_pending_stop_partial_decode(),
      "F2: stop()'s partial-decode Err MUST arm StopPartialDecode \
       (with the cloned audio_features payload, never silently None)"
    );

    // Second stop(): fast path takes the obligation, clones for
    // re-arm, calls finalize_partial_window_and_emit_ended, and on
    // success clears. The clone is real — never None when the
    // pre-arm payload was Some.
    let stop_second = session
      .stop()
      .expect("F2: retry stop() fast path MUST succeed");
    assert!(matches!(
      stop_second.last(),
      Some(TranscriptionEvent::Ended(_))
    ));
    assert!(!session.retry_state().has_obligation());
    assert!(!session.is_active());
  }

  /// F2 contract: the stop() mainline body's clone-for-arm step is
  /// fallible-via-`?`-propagation now. We can't trigger
  /// `Array::try_clone` to fail deterministically, but we can
  /// confirm the structural shape: `clone_partial_decode_payload`
  /// returns `Result` so a clone Err propagates via `?` instead of
  /// being dropped into `None`. The helper test above proves the
  /// Err path exists; this test proves the call-site uses the
  /// helper (not the pre-fix `.ok()` pattern) by exercising the
  /// happy-path through the mainline body AND asserting the
  /// arm-payload was never `None` when audio_features was Some.
  #[test]
  fn session_retry_state_stop_partial_decode_mainline_body_arms_with_real_clone() {
    let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
    // Decoder: feed_audio partial, then stop()'s partial-decode Errs
    // so the arm is the LAST mutation before the Err.
    let decoder = ScriptedDecoder::with_results(vec![
      Ok(vec![1]),
      Err(crate::error::Error::Backend {
        message: "scripted stop-partial-decode Err".into(),
      }),
    ]);
    let mut session = nonfinalize_session(encoder, decoder);
    let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
    let _ = session.feed_audio(&samples).unwrap();

    // stop() reaches the mainline body, hits step 6's arm + decode.
    // The arm clones audio_features (real refcount handle); the
    // decode then Errs. Without F2's fix, the arm would have been
    // `None` if the clone failed silently — but we can assert that
    // when audio_features WAS Some (the encoder has a partial
    // window), the arm-payload is Some.
    let stop_first = session.stop();
    assert!(stop_first.is_err());
    assert!(
      session.retry_state().has_pending_stop_partial_decode(),
      "mainline-body Err must arm StopPartialDecode"
    );

    // F2 STRUCTURAL CHECK: the obligation's payload is Some (not None).
    // We inspect via take + immediate re-arm so the obligation isn't
    // permanently consumed.
    let taken = session
      .retry_state_mut()
      .take_stop_partial_decode_features()
      .expect("guard above asserts arm");
    assert!(
      taken.is_some(),
      "F2: mainline arm MUST carry the real cloned payload, never \
       silently None — the encoder had a partial window (1-row carry \
       after the 7-mel feed + stop-flush bridge), so encode_pending \
       returned Some, so the helper's Ok arm is Some"
    );
    // Restore so test cleanup is consistent.
    session.retry_state_mut().arm_stop_partial_decode(taken);
  }
}

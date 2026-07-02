//! The Silero VAD model graph — port of
//! [`vad/models/silero_vad/silero_vad.py`][silero] (`SileroVADBranch`,
//! `Model`, and the standalone `_reflect_pad_right` / `_probs_to_timestamps`
//! helpers).
//!
//! ## Architecture (`SileroVADBranch.__call__`, `silero_vad.py:61-90`)
//!
//! Each sample-rate branch is, in order:
//!
//! 1. `_reflect_pad_right(x, pad)` — reflect one window's worth of samples off
//!    the right edge (`silero_vad.py:28-39`);
//! 2. `stft_conv` — a single strided 1-D conv (`1 → 2*cutoff` channels, kernel
//!    `filter_length`, stride `hop_length`, no bias) that *is* the learned STFT
//!    filterbank; its `2*cutoff` output channels are the stacked real / imag
//!    parts, recombined to a magnitude spectrum
//!    `sqrt(real² + imag²)` (`silero_vad.py:71-75`);
//! 3. four `relu(convk(·))` blocks (`128/64/64/128` channels, two stride-2)
//!    over the channels-last magnitude features (`silero_vad.py:77-80`);
//! 4. a single-layer LSTM (`128 → 128`) whose last-step hidden / cell are
//!    stacked into the returned recurrent state (`silero_vad.py:82-85`);
//! 5. the speech-prob head: `relu(hidden_seq)` → `sigmoid(final_conv)` (a
//!    `128 → 1` 1×1 conv) → squeeze the channel axis → mean over time, giving
//!    one probability per window (`silero_vad.py:87-89`).
//!
//! ## Chunking + streaming (`Model._predict_proba_array`, `silero_vad.py:268-321`)
//!
//! A whole waveform is processed as fixed `chunk_size`-sample frames (512 at
//! 16 kHz, 256 at 8 kHz), each prepended with `context_size` trailing samples
//! of left context carried from the previous frame, threaded through the LSTM
//! recurrent state. The streaming [`SileroVadModel::feed`]
//! (`silero_vad.py:162-196`) exposes the same single-frame step for realtime
//! use.
//!
//! ## Speech-segment extraction (`Model._probs_to_timestamps`, `silero_vad.py:360-427`)
//!
//! A per-frame probability sequence is collapsed to start/end sample-index
//! segments by a hysteresis state machine (a high `threshold` opens a segment,
//! a `neg_threshold = max(threshold-0.15, 0.01)` sustains it, `min_silence` of
//! sub-threshold audio closes it, segments below `min_speech` are dropped),
//! then padded by `speech_pad` and merged. Ported verbatim in
//! [`probs_to_timestamps`].
//!
//! [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  audio::vad::{
    models::silero_vad::config::{BranchConfig, ModelConfig},
    output::SpeechSegment,
  },
  dtype::Dtype,
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result},
  ops,
};

/// Chunk cadence at which `predict_proba` issues a non-blocking `async_eval`
/// over the per-chunk output + recurrent state to bound lazy-graph retention on
/// long audio — the reference `eval_every` default (`silero_vad.py:272/312-316`).
const EVAL_EVERY: usize = 16;

/// One Silero VAD sample-rate branch — port of `SileroVADBranch`
/// ([silero_vad.py:42-102][silero]).
///
/// Holds the learned STFT-conv filterbank, the four convolutional blocks, the
/// recurrent LSTM, and the `128 → 1` speech-prob head, plus the branch
/// [`BranchConfig`] driving the reflect-pad / real-imag split.
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L42-L102
pub struct SileroVadBranch {
  config: BranchConfig,
  /// `stft_conv` weight `(2*cutoff, filter_length, 1)`, channels-last.
  stft_conv_weight: Array,
  conv1: ConvBlock,
  conv2: ConvBlock,
  conv3: ConvBlock,
  conv4: ConvBlock,
  lstm: Lstm,
  /// `final_conv` (`128 → 1`, kernel 1) weight `(1, 1, 128)` + bias `(1,)`.
  final_conv_weight: Array,
  final_conv_bias: Array,
}

/// A `relu(conv1d(·) + bias)` block — `nn.Conv1d` with bias followed by
/// `nn.relu` (`silero_vad.py:54-57,77-80`). The weight is channels-last
/// `(C_out, K, C_in)` exactly as `mlx.nn.Conv1d` stores it, so no transpose is
/// needed (Silero ships native MLX `Conv1d` weights — unlike the HF-layout
/// `wav2vec2` checkpoint).
struct ConvBlock {
  weight: Array,
  bias: Array,
  stride: i32,
  padding: i32,
}

impl ConvBlock {
  /// `relu(conv1d(x, stride, padding) + bias)` over channels-last
  /// `(B, L, C_in)` → `(B, L', C_out)`.
  fn forward(&self, x: &Array) -> Result<Array> {
    let h = ops::conv::conv1d(x, &self.weight, self.stride, self.padding, 1, 1)?;
    let h = h.add(&self.bias)?;
    relu(&h)
  }
}

/// A 0-D `f32` scalar array (the `&[0i32; 0]`-shape rank-0 idiom).
fn scalar_f32(value: f32) -> Result<Array> {
  Array::from_slice::<f32>(&[value], &[0i32; 0])
}

/// A 0-D `i32` scalar index array — used as a single-element `take_axis` index
/// that *drops* the gathered axis (numpy `a[idx]` semantics), matching the
/// reference's `state[0]` / `seq[:, -1, :]` / `x[..., idx, :]` axis-dropping
/// indexing.
fn idx0(value: i32) -> Result<Array> {
  Array::from_slice::<i32>(&[value], &[0i32; 0])
}

/// `mx.maximum(x, 0)` in `x`'s dtype — the `nn.relu` of the reference. The
/// zero literal is built in `x`'s dtype so a half-precision activation is not
/// promoted to f32 (the dtype-preservation discipline).
fn relu(x: &Array) -> Result<Array> {
  let zero = scalar_f32(0.0)?.astype(x.dtype()?)?;
  ops::arithmetic::maximum(x, &zero)
}

/// Reflect-pad `x` on the right by `pad` samples — port of
/// `_reflect_pad_right` ([silero_vad.py:28-39][silero]).
///
/// Reproduces the reference gather exactly: append `x[..., L-2 : L-pad-2 : -1]`
/// (the reversed run of `pad` samples ending two before the last sample) to
/// `x`. A `pad <= 0` is the identity; a `pad >= L` is rejected (mlx-audio
/// raises `ValueError` — `silero_vad.py:31-35`).
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L28-L39
fn reflect_pad_right(x: &Array, pad: i32) -> Result<Array> {
  if pad <= 0 {
    return x.try_clone();
  }
  let shape = x.shape();
  let last = *shape.last().ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      "reflect_pad_right: input",
      shape.len() as u32,
      shape.clone(),
    ))
  })?;
  let len = i32::try_from(last).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "reflect_pad_right: input length",
      "must fit in i32",
      format_smolstr!("{last}"),
    ))
  })?;
  if len <= pad {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "reflect_pad_right: pad vs samples",
      "reflect padding requires more samples than the pad width",
      format_smolstr!("pad={pad}, samples={len}"),
    )));
  }
  // mx.arange(L-2, L-pad-2, -1): the reversed indices [L-2, L-3, …, L-pad-1].
  let indices = Array::arange::<i32>(len - 2, len - pad - 2, -1)?;
  let reflected = ops::indexing::take_axis(x, &indices, -1)?;
  ops::shape::concatenate(&[x, &reflected], -1)
}

impl SileroVadBranch {
  /// The branch sample-rate config.
  #[inline(always)]
  pub const fn config(&self) -> &BranchConfig {
    &self.config
  }

  /// Forward one already-context-prefixed window through the branch — port of
  /// `SileroVADBranch.__call__` ([silero_vad.py:61-90][silero]).
  ///
  /// `x` is `(B, T)` (or `(T,)`, promoted to `(1, T)`). `state`, if present,
  /// is the `(2, B, 128)` stacked hidden / cell from a prior step. Returns the
  /// `(B, 1)` mean speech probability for the window plus the new `(2, B, 128)`
  /// recurrent state.
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L61-L90
  pub fn forward(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
    // x.ndim == 1 → x[None, :]. A rank-2 input is used by reference (the
    // reflect-pad below produces a fresh array regardless), so it is not cloned.
    let promoted;
    let x = if x.ndim() == 1 {
      promoted = ops::shape::expand_dims_axes(x, &[0])?;
      &promoted
    } else {
      x
    };

    let (hidden, cell) = split_state(state)?;

    // _reflect_pad_right(x, pad) then stft_conv(x[..., None]).
    let x = reflect_pad_right(x, self.config.pad())?;
    // x[..., None]: (B, T) → (B, T, 1) channels-last for the stride-`hop` conv.
    let x = ops::shape::expand_dims_axes(&x, &[-1])?;
    let x = ops::conv::conv1d(
      &x,
      &self.stft_conv_weight,
      self.config.hop_length(),
      0,
      1,
      1,
    )?;

    // real = x[..., :cutoff]; imag = x[..., cutoff:]; x = sqrt(real² + imag²).
    let cutoff = self.config.cutoff();
    let real = slice_last_axis(&x, 0, cutoff)?;
    let imag = slice_last_axis(&x, cutoff, 2 * cutoff)?;
    let mag = real.multiply(&real)?.add(&imag.multiply(&imag)?)?;
    let x = ops::arithmetic::sqrt(&mag)?;

    // relu(conv1..conv4(x)).
    let x = self.conv1.forward(&x)?;
    let x = self.conv2.forward(&x)?;
    let x = self.conv3.forward(&x)?;
    let x = self.conv4.forward(&x)?;

    // hidden_seq, cell_seq = lstm(x, hidden, cell).
    let (hidden_seq, cell_seq) = self.lstm.forward(&x, hidden.as_ref(), cell.as_ref())?;
    // hidden = hidden_seq[:, -1, :]; cell = cell_seq[:, -1, :].
    let last_hidden = last_timestep(&hidden_seq)?;
    let last_cell = last_timestep(&cell_seq)?;
    let new_state = ops::shape::stack_axis(&[&last_hidden, &last_cell], 0)?;

    // x = relu(hidden_seq); x = sigmoid(final_conv(x)).
    let x = relu(&hidden_seq)?;
    let x = ops::conv::conv1d(&x, &self.final_conv_weight, 1, 0, 1, 1)?;
    let x = x.add(&self.final_conv_bias)?;
    let x = ops::arithmetic::sigmoid(&x)?;
    // x = mean(squeeze(x, -1), axis=1, keepdims=True): squeeze the (B, L, 1)
    // channel axis to (B, L), then mean over time → (B, 1).
    let x = ops::shape::squeeze_axes(&x, &[-1])?;
    let x = ops::reduction::mean_axes(&x, &[1], true)?;
    Ok((x, new_state))
  }
}

/// Split a `(2, B, 128)` stacked recurrent state into `(hidden, cell)` — port
/// of `SileroVADBranch._split_state` ([silero_vad.py:92-102][silero]). `None`
/// → `(None, None)` (the first step). A non-`(2, …)` state is rejected exactly
/// as the reference's `ValueError`.
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L92-L102
fn split_state(state: Option<&Array>) -> Result<(Option<Array>, Option<Array>)> {
  let Some(state) = state else {
    return Ok((None, None));
  };
  let shape = state.shape();
  if shape.len() != 3 || shape[0] != 2 {
    let rank = shape.len() as u32;
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "silero_vad state: expected (2, batch, 128)",
      rank,
      shape,
    )));
  }
  let hidden = state.take_axis(&idx0(0)?, 0)?;
  let cell = state.take_axis(&idx0(1)?, 0)?;
  Ok((Some(hidden), Some(cell)))
}

/// `seq[:, -1, :]` — the last time-step of a `(B, L, H)` sequence, as `(B, H)`.
fn last_timestep(seq: &Array) -> Result<Array> {
  let l = i32::try_from(seq.shape()[1]).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "silero_vad lstm seq length",
      "must fit in i32",
      format_smolstr!("{}", seq.shape()[1]),
    ))
  })?;
  let idx = idx0(l - 1)?;
  seq.take_axis(&idx, 1)
}

/// `x[..., lo:hi]` on the last axis of a rank-`n` array (full-rank slice with
/// every other axis full-range).
fn slice_last_axis(x: &Array, lo: i32, hi: i32) -> Result<Array> {
  let shape = x.shape();
  let n = shape.len();
  let mut start = vec![0_i32; n];
  let mut stop: Vec<i32> = shape
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "slice_last_axis: dim",
          "must fit in i32",
          format_smolstr!("{d}"),
        ))
      })
    })
    .collect::<Result<_>>()?;
  let strides = vec![1_i32; n];
  start[n - 1] = lo;
  stop[n - 1] = hi;
  ops::indexing::slice(x, &start, &stop, &strides)
}

/// A single-layer LSTM — faithful port of `mlx.nn.LSTM.__call__`
/// ([`recurrent.py`][lstm], the layer Silero's `SileroVADBranch` instantiates
/// as `nn.LSTM(128, 128)`).
///
/// `mlx.nn.LSTM` stores `Wx` as `(4H, D)`, `Wh` as `(4H, H)`, `bias` as
/// `(4H,)`, and per call forms `x @ Wx.T` / `hidden @ Wh.T`. Both transposes
/// are loop-invariant, so they are taken once at construction and the
/// `(D, 4H)` / `(H, 4H)` transposed views are stored here — the `addmm` inputs
/// are byte-identical to taking `Wx.T` per call, only the (lazy, zero-copy)
/// transpose node is shared instead of rebuilt every frame. The per-step gate
/// order is `i, f, g, o` (the split of the `4H` pre-activation), with
/// `c = f*c + i*g`, `h = o*tanh(c)`. The `x @ Wxᵀ (+ bias)` pre-activation is
/// computed once for the whole sequence, then the recurrence runs step-by-step
/// over the time axis.
///
/// [lstm]: https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/recurrent.py
struct Lstm {
  /// Transposed input-to-hidden weight `Wx.T` = `(D, 4H)`.
  wx_t: Array,
  /// Transposed hidden-to-hidden weight `Wh.T` = `(H, 4H)`.
  wh_t: Array,
  /// Gate bias `(4H,)`.
  bias: Array,
  /// Hidden size `H`.
  hidden_size: i32,
}

impl Lstm {
  /// Run the LSTM over a `(B, L, D)` (or `(L, D)`) sequence, optionally seeded
  /// with a prior `hidden` / `cell` (`(B, H)`). Returns the per-step hidden and
  /// cell sequences, both `(B, L, H)` — `mlx.nn.LSTM.__call__`.
  fn forward(
    &self,
    x: &Array,
    hidden: Option<&Array>,
    cell: Option<&Array>,
  ) -> Result<(Array, Array)> {
    // x = addmm(bias, x, Wx.T)  →  (…, L, 4H). The bias broadcasts over the
    // leading axes; addmm with alpha=beta=1 is `bias + x @ Wxᵀ`. `wx_t` / `wh_t`
    // are the loop-invariant transposed weights, taken once at construction.
    let pre = x.addmm(&self.bias, &self.wx_t, 1.0, 1.0)?;

    let time_axis = pre.ndim() - 2;
    let seq_len = i32::try_from(pre.shape()[time_axis]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad lstm: sequence length",
        "must fit in i32",
        format_smolstr!("{}", pre.shape()[time_axis]),
      ))
    })?;
    let time_axis_i = time_axis as i32;

    let h = self.hidden_size;
    let split_points = [h, 2 * h, 3 * h];

    // Owned running state (Array is a cheap refcounted handle; `try_clone`
    // bumps the rc, it does not copy the buffer).
    let mut hidden: Option<Array> = match hidden {
      Some(h) => Some(h.try_clone()?),
      None => None,
    };
    let mut cell: Option<Array> = match cell {
      Some(c) => Some(c.try_clone()?),
      None => None,
    };
    let mut all_hidden: Vec<Array> = Vec::with_capacity(seq_len as usize);
    let mut all_cell: Vec<Array> = Vec::with_capacity(seq_len as usize);

    for idx in 0..seq_len {
      // ifgo = pre[..., idx, :]  (drop the time axis).
      let mut ifgo = pre.take_axis(&idx0(idx)?, time_axis_i)?;
      // if hidden is not None: ifgo = mx.addmm(ifgo, hidden, wh.T) = ifgo +
      // hidden @ wh.T. In mlxrs `a.addmm(c, b)` = a @ b + c, so `a = hidden`,
      // `c = ifgo`, `b = wh_t`.
      if let Some(prev_h) = &hidden {
        ifgo = prev_h.addmm(&ifgo, &self.wh_t, 1.0, 1.0)?;
      }
      let parts = ops::shape::split_sections(&ifgo, &split_points, -1)?;
      let i = ops::arithmetic::sigmoid(&parts[0])?;
      let f = ops::arithmetic::sigmoid(&parts[1])?;
      let g = ops::arithmetic::tanh(&parts[2])?;
      let o = ops::arithmetic::sigmoid(&parts[3])?;

      // cell = f*cell + i*g  (or i*g on the first step).
      let new_cell = match &cell {
        Some(prev_c) => f.multiply(prev_c)?.add(&i.multiply(&g)?)?,
        None => i.multiply(&g)?,
      };
      // hidden = o * tanh(cell).
      let new_hidden = o.multiply(&ops::arithmetic::tanh(&new_cell)?)?;

      cell = Some(new_cell.try_clone()?);
      hidden = Some(new_hidden.try_clone()?);
      all_cell.push(new_cell);
      all_hidden.push(new_hidden);
    }

    let hidden_refs: Vec<&Array> = all_hidden.iter().collect();
    let cell_refs: Vec<&Array> = all_cell.iter().collect();
    // stack(all_hidden, axis=-2) → (B, L, H).
    let hidden_seq = ops::shape::stack_axis(&hidden_refs, -2)?;
    let cell_seq = ops::shape::stack_axis(&cell_refs, -2)?;
    Ok((hidden_seq, cell_seq))
  }
}

/// The Silero voice activity detector — port of `Model`
/// ([silero_vad.py:105-431][silero]). Holds both per-rate [`SileroVadBranch`]es
/// plus the [`ModelConfig`]; its [`VadModel::generate`](crate::audio::vad::VadModel::generate)
/// impl is the [`crate::audio::vad::VadModel`] entry point.
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L105-L431
pub struct SileroVadModel {
  config: ModelConfig,
  vad_16k: SileroVadBranch,
  vad_8k: SileroVadBranch,
}

/// The streaming recurrent state — port of `SileroVADState`
/// ([silero_vad.py:14-18][silero]): the optional `(2, B, 128)` LSTM state, the
/// `(B, context_size)` carried left context, and the branch sample rate.
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L14-L18
#[derive(Debug)]
pub struct SileroVadState {
  state: Option<Array>,
  context: Array,
  sample_rate: u32,
}

impl SileroVadState {
  /// The carried LSTM recurrent state (`None` before the first frame).
  #[inline(always)]
  pub fn state(&self) -> Option<&Array> {
    self.state.as_ref()
  }

  /// The carried left context `(B, context_size)`.
  #[inline(always)]
  pub fn context(&self) -> &Array {
    &self.context
  }

  /// The branch sample rate this state belongs to.
  #[inline(always)]
  pub const fn sample_rate(&self) -> u32 {
    self.sample_rate
  }
}

impl SileroVadModel {
  /// Construct a [`SileroVadModel`] from a parsed config and its two built
  /// branches. The loader ([`super::loader`]) builds the branches from the
  /// weight map and calls this.
  pub fn new(config: ModelConfig, vad_16k: SileroVadBranch, vad_8k: SileroVadBranch) -> Self {
    Self {
      config,
      vad_16k,
      vad_8k,
    }
  }

  /// The model config.
  #[inline(always)]
  pub const fn config(&self) -> &ModelConfig {
    &self.config
  }

  /// The model activation dtype.
  #[inline(always)]
  pub const fn dtype(&self) -> Dtype {
    self.config.dtype()
  }

  /// Select the branch for `sample_rate` — port of `Model._branch`
  /// ([silero_vad.py:353-358][silero]). Only 16 kHz and 8 kHz are supported.
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L353-L358
  pub fn branch(&self, sample_rate: u32) -> Result<&SileroVadBranch> {
    match sample_rate {
      16_000 => Ok(&self.vad_16k),
      8_000 => Ok(&self.vad_8k),
      other => Err(Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad: sample_rate",
        "Silero VAD supports 8000 Hz and 16000 Hz audio",
        format_smolstr!("{other}"),
      ))),
    }
  }

  /// Forward one already-context-prefixed window — port of `Model.__call__`
  /// ([silero_vad.py:121-148][silero]). Casts `x` (and any non-tuple `state`)
  /// to the model dtype, then runs the selected branch.
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L121-L148
  pub fn forward(
    &self,
    x: &Array,
    state: Option<&Array>,
    sample_rate: u32,
  ) -> Result<(Array, Array)> {
    let branch = self.branch(sample_rate)?;
    let x = x.astype(self.dtype())?;
    let state = match state {
      Some(s) => Some(s.astype(self.dtype())?),
      None => None,
    };
    branch.forward(&x, state.as_ref())
  }

  /// The initial streaming state for `batch_size` streams at `sample_rate` —
  /// port of `Model.initial_state` ([silero_vad.py:150-155][silero]): no LSTM
  /// state and a zeroed `(batch_size, context_size)` context.
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L150-L155
  pub fn initial_state(&self, batch_size: i32, sample_rate: u32) -> Result<SileroVadState> {
    let branch = self.branch(sample_rate)?;
    let context =
      Array::zeros::<f32>(&[batch_size, branch.config().context_size()])?.astype(self.dtype())?;
    Ok(SileroVadState {
      state: None,
      context,
      sample_rate,
    })
  }

  /// Feed one `chunk_size`-sample streaming frame — port of `Model.feed`
  /// ([silero_vad.py:162-196][silero]).
  ///
  /// `chunk` is `(chunk_size,)` or `(B, chunk_size)`. The new window is
  /// `concat([state.context, chunk])`, the probability + next LSTM state come
  /// from [`Self::forward`], and the next context is the last `context_size`
  /// samples of `chunk`. A wrong chunk width or a sample-rate mismatch with the
  /// state is rejected exactly as the reference's `ValueError`.
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L162-L196
  pub fn feed(
    &self,
    chunk: &Array,
    state: Option<SileroVadState>,
    sample_rate: u32,
  ) -> Result<(Array, SileroVadState)> {
    // Mirror the reference's mono `(T,)` / batched `(B, T)` contract (and its
    // `raise` on other ranks) — reject rank-0 / rank-3+ with a typed error so a
    // bad caller chunk cannot mis-thread or panic the streaming path.
    let ndim = chunk.ndim();
    if ndim != 1 && ndim != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "silero_vad feed: chunk must be rank-1 (T,) or rank-2 (B, T)",
        ndim as u32,
        chunk.shape().to_vec(),
      )));
    }
    let branch = self.branch(sample_rate)?;
    let chunk = chunk.astype(self.dtype())?;
    let chunk = if chunk.ndim() == 1 {
      ops::shape::expand_dims_axes(&chunk, &[0])?
    } else {
      chunk
    };
    let last_dim = *chunk.shape().last().unwrap_or(&0);
    let chunk_width = i32::try_from(last_dim).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad feed: chunk width",
        "must fit in i32",
        format_smolstr!("{last_dim}"),
      ))
    })?;
    if chunk_width != branch.config().chunk_size() {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad feed: chunk width",
        "must equal the branch chunk_size for the sample rate",
        format_smolstr!(
          "expected={}, got={chunk_width}",
          branch.config().chunk_size()
        ),
      )));
    }

    let batch = i32::try_from(chunk.shape()[0]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad feed: batch size",
        "must fit in i32",
        format_smolstr!("{}", chunk.shape()[0]),
      ))
    })?;
    let state = match state {
      Some(s) => s,
      None => self.initial_state(batch, sample_rate)?,
    };
    if state.sample_rate != sample_rate {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad feed: state sample rate",
        "streaming state sample rate must match the call",
        format_smolstr!("state={}, call={sample_rate}", state.sample_rate),
      )));
    }

    let window = ops::shape::concatenate(&[&state.context, &chunk], -1)?;
    let (probability, lstm_state) = self.forward(&window, state.state.as_ref(), sample_rate)?;
    // new_context = chunk[:, -context_size:].
    let ctx = branch.config().context_size();
    let new_context = slice_last_axis(&chunk, chunk_width - ctx, chunk_width)?;
    Ok((
      probability,
      SileroVadState {
        state: Some(lstm_state),
        context: new_context,
        sample_rate,
      },
    ))
  }

  /// Run VAD over a whole waveform, returning the per-frame probabilities —
  /// port of `Model._predict_proba_array` ([silero_vad.py:268-321][silero]).
  ///
  /// The waveform is right-padded to a whole number of `chunk_size` frames,
  /// prefixed with a zeroed `context_size` left context, then stepped frame by
  /// frame through the recurrent branch. A `(1, …)` input (the unbatched case)
  /// returns a 1-D `(n_frames,)` probability vector; a batched `(B, …)` input
  /// returns `(B, n_frames)`. An empty waveform returns an empty probability
  /// array (the reference's early return — `silero_vad.py:290-295`).
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L268-L321
  pub fn predict_proba(&self, audio: &Array, sample_rate: u32) -> Result<Array> {
    // The reference accepts only mono `(T,)` or batched `(B, T)` audio and
    // `raise`s otherwise (`silero_vad.py:31/100/176`). Reject any other rank
    // with a typed error BEFORE the shape indexing below — a rank-0 (scalar)
    // input would otherwise fall through the empty-input branch and panic on
    // `audio.shape()[0]` of an empty shape vector.
    let ndim = audio.ndim();
    if ndim != 1 && ndim != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "silero_vad predict_proba: audio must be rank-1 (T,) or rank-2 (B, T)",
        ndim as u32,
        audio.shape().to_vec(),
      )));
    }
    let branch = self.branch(sample_rate)?;
    let chunk_size = branch.config().chunk_size();
    let context_size = branch.config().context_size();
    let audio = audio.astype(self.dtype())?;
    let original_ndim = audio.ndim();

    let audio = if original_ndim == 1 {
      ops::shape::expand_dims_axes(&audio, &[0])?
    } else {
      audio
    };

    let total = *audio.shape().last().unwrap_or(&0);
    if total == 0 {
      // Empty input → empty probabilities (1-D or (B, 0)).
      let batch = i32::try_from(audio.shape()[0]).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "silero_vad predict_proba: batch size",
          "must fit in i32",
          format_smolstr!("{}", audio.shape()[0]),
        ))
      })?;
      return if original_ndim == 1 {
        Array::zeros::<f32>(&[0])?.astype(self.dtype())
      } else {
        Array::zeros::<f32>(&[batch, 0])?.astype(self.dtype())
      };
    }

    let total_i = i32::try_from(total).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad predict_proba: audio length",
        "must fit in i32",
        format_smolstr!("{total}"),
      ))
    })?;
    let batch = i32::try_from(audio.shape()[0]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad predict_proba: batch size",
        "must fit in i32",
        format_smolstr!("{}", audio.shape()[0]),
      ))
    })?;

    // A zero-ROW batch has the frame-count contract of the reference loop —
    // `(0, ceil(total / chunk))` — but running the chunk loop over a
    // data-less input would be width-proportional WORK (a `(0, huge)` shape
    // would iterate millions of empty windows). Short-circuit to the final
    // shape directly; every value axis is empty so the result is exact.
    if batch == 0 {
      // Ceil division in i64 (signed div_ceil is unstable; the +chunk-1 form
      // could overflow i32 at extreme widths). The result never exceeds
      // total_i, so the i32 narrowing is exact.
      let frames =
        ((i64::from(total_i) + i64::from(chunk_size) - 1) / i64::from(chunk_size)) as i32;
      return Array::zeros::<f32>(&[0, frames])?.astype(self.dtype());
    }

    // pad = (chunk_size - L % chunk_size) % chunk_size; right-pad with zeros.
    let pad = (chunk_size - total_i % chunk_size) % chunk_size;
    let audio = if pad > 0 {
      let zero = scalar_f32(0.0)?.astype(self.dtype())?;
      ops::shape::pad(&audio, &[1], &[0], &[pad], &zero, c"constant")?
    } else {
      audio
    };

    // Prepend the zeroed left context.
    let context = Array::zeros::<f32>(&[batch, context_size])?.astype(self.dtype())?;
    let audio = ops::shape::concatenate(&[&context, &audio], -1)?;
    let padded_len = i32::try_from(*audio.shape().last().unwrap_or(&0)).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad predict_proba: padded length",
        "must fit in i32",
        "overflow",
      ))
    })?;

    let mut outputs: Vec<Array> = Vec::new();
    let mut state: Option<Array> = None;
    // for pos in range(context_size, padded_len, chunk_size):
    //   window = audio[:, pos-context_size : pos+chunk_size]
    let mut pos = context_size;
    let mut step = 0usize;
    while pos < padded_len {
      let window = slice_last_axis(&audio, pos - context_size, pos + chunk_size)?;
      let (out, new_state) = self.forward(&window, state.as_ref(), sample_rate)?;
      // Mirror the reference (`silero_vad.py:306-313`): `async_eval(out, state)`
      // every `EVAL_EVERY` (16) steps to BOUND lazy-graph retention on long
      // audio. Without this a long recording (the VAD's actual use case)
      // accumulates the entire recurrent graph until the single final eval,
      // risking large memory growth / OOM. The reference's `step` is 1-based
      // (`enumerate(range(...), start=1)`), so `step` is incremented BEFORE the
      // check: this both matches the reference phase AND, critically, fires the
      // eval on the LAST chunk when the chunk count is an exact multiple of
      // EVAL_EVERY (a 0-based `step` tops out at count-1, never reaching the
      // multiple, and the tail guard below also skips the exact-multiple case —
      // so the final state would never be bounded). `async_eval` is non-blocking
      // and does not change the result, only when materialization happens.
      step += 1;
      if step.is_multiple_of(EVAL_EVERY) {
        crate::transforms::async_eval(&[&out, &new_state])?;
      }
      outputs.push(out);
      state = Some(new_state);
      pos += chunk_size;
    }
    // Reference tail (`silero_vad.py:315-316`): async_eval the last output +
    // state when the chunk count is not a multiple of `EVAL_EVERY` (when it IS,
    // the final in-loop eval above already bounded it).
    if !outputs.len().is_multiple_of(EVAL_EVERY)
      && let (Some(last), Some(st)) = (outputs.last(), state.as_ref())
    {
      crate::transforms::async_eval(&[last, st])?;
    }

    let out_refs: Vec<&Array> = outputs.iter().collect();
    // probabilities = concatenate(outputs, axis=1) → (B, n_frames).
    let mut probabilities = ops::shape::concatenate(&out_refs, 1)?;
    if original_ndim == 1 {
      // probabilities[0] → (n_frames,).
      probabilities = probabilities.take_axis(&idx0(0)?, 0)?;
    }
    Ok(probabilities)
  }

  /// Preprocess raw input audio for inference — port of
  /// `Model._prepare_audio_array` (silero_vad.py:323-351).
  ///
  /// Applies the reference's two corrections, both ABSENT from the lower-level
  /// [`Self::predict_proba`] (which assumes already-prepared mono / batched
  /// audio at a supported rate):
  ///
  /// 1. **stereo / multichannel downmix** — a rank-2 `(T, C)` waveform with a
  ///    small trailing channel count (`C <= 8 < T`) is averaged over the channel
  ///    axis to mono `(T,)`, while a `(B, T)` batch (trailing `T > 8`) is left
  ///    untouched, exactly as the reference's `ndim == 2 and shape[-1] <= 8 <
  ///    shape[0]` heuristic discriminates the two. Without this a `(T, 2)` stereo
  ///    clip is silently misread as a `T`-stream / 2-sample batch;
  /// 2. **resample** — a sample rate that is neither 8 kHz nor 16 kHz is linearly
  ///    resampled to 16 kHz (`target_sr = sr if sr in (8000, 16000) else 16000`),
  ///    so a 44.1 kHz / 48 kHz waveform is handled rather than rejected by
  ///    [`Self::branch`].
  ///
  /// Returns the prepared waveform and the resolved sample rate. A rank outside
  /// `(T,)` / `(B, T)` is rejected with a typed [`Error::RankMismatch`].
  pub fn prepare_audio(&self, audio: &Array, sample_rate: u32) -> Result<(Array, u32)> {
    // A zero sample rate can never resolve a timeline (the resample ratio and
    // the empty-batch width arithmetic both divide by it) — reject it with
    // the same typed class resample_linear uses, BEFORE any arithmetic, so
    // no path can reach an integer divide-by-zero.
    if sample_rate == 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad prepare_audio: sample_rate",
        "must be > 0",
        "0",
      )));
    }
    let ndim = audio.ndim();
    if ndim != 1 && ndim != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "silero_vad prepare_audio: audio must be rank-1 (T,) or rank-2 (B, T) / (T, C)",
        ndim as u32,
        audio.shape(),
      )));
    }
    // Empty input of ANY layout — (0,), (0, T), (B, 0) — has nothing to
    // downmix or resample; handle it before the downmix can reduce over a
    // zero-length axis (a `(2, 0)` input would otherwise satisfy the stereo
    // heuristic and `mean_axes` a zero-length axis) and before the resample
    // row loop. The resolved rate follows the target-rate net, and the TIME
    // AXIS follows it too: a zero-ROW batch with a nonzero width at an
    // unsupported rate is returned with the width a real row would have
    // after resampling (computed through the same resample_linear the
    // non-empty path uses), so the downstream chunking's frame count
    // reflects the resolved-rate timeline, not the original one.
    if audio.shape().contains(&0) {
      let target_sr = if sample_rate == 8_000 || sample_rate == 16_000 {
        sample_rate
      } else {
        16_000
      };
      let shape = audio.shape();
      let width = *shape.last().unwrap_or(&0);
      if sample_rate != target_sr && shape.len() == 2 && width > 0 {
        // The width a real row would resample to — the same arithmetic
        // resample_linear derives its output length from (`in_len * to_rate
        // / from_rate`, checked u64), computed WITHOUT materializing a dummy
        // row: a data-less `(0, huge)` shape costs only shape arithmetic (a
        // typed error on overflow), never a width-proportional allocation.
        let resampled_width = usize::try_from(
          (width as u64)
            .checked_mul(u64::from(target_sr))
            .ok_or_else(|| {
              Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::with_operands(
                "silero_vad prepare_audio: width * target_rate",
                "u64",
                [
                  ("width", width as u64),
                  ("target_rate", u64::from(target_sr)),
                ],
              ))
            })?
            / u64::from(sample_rate),
        )
        .unwrap_or(usize::MAX);
        let w = i32::try_from(resampled_width).map_err(|_| {
          Error::OutOfRange(OutOfRangePayload::new(
            "silero_vad prepare_audio: resampled width",
            "must fit in i32",
            format_smolstr!("{resampled_width}"),
          ))
        })?;
        return Ok((
          Array::zeros::<f32>(&[0, w])?.astype(audio.dtype()?)?,
          target_sr,
        ));
      }
      return Ok((audio.try_clone()?, target_sr));
    }
    // Stereo / multichannel downmix: (T, C) with C <= 8 < T → mean over C —
    // the EXACT reference predicate (`shape[-1] <= 8 < shape[0]`,
    // silero_vad.py:338): the row count must exceed 8, not merely the column
    // count, so a small batch such as (3, 2) stays a batch.
    let downmixed = if ndim == 2 {
      let shape = audio.shape();
      let rows = shape[0];
      let cols = shape[1];
      if cols <= 8 && 8 < rows {
        ops::reduction::mean_axes(audio, &[-1], false)?
      } else {
        audio.try_clone()?
      }
    } else {
      audio.try_clone()?
    };
    // Resample to a supported rate when the input is neither 8 kHz nor 16 kHz.
    let target_sr = if sample_rate == 8_000 || sample_rate == 16_000 {
      sample_rate
    } else {
      16_000
    };
    if sample_rate == target_sr {
      return Ok((downmixed, sample_rate));
    }
    Ok((
      resample_audio_array(&downmixed, sample_rate, target_sr)?,
      target_sr,
    ))
  }

  /// Per-frame speech probabilities for raw audio — port of `Model.predict` /
  /// the public `Model.predict_proba` (silero_vad.py:198-207).
  ///
  /// Prepares the audio ([`Self::prepare_audio`]: downmix + resample) then runs
  /// the lower-level [`Self::predict_proba`] chunking core. (mlxrs's
  /// [`Self::predict_proba`] is the reference's `_predict_proba_array`, which
  /// assumes already-prepared audio; this is the public entry that downmixes /
  /// resamples first.)
  pub fn predict(&self, audio: &Array, sample_rate: u32) -> Result<Array> {
    let (audio, sr) = self.prepare_audio(audio, sample_rate)?;
    self.predict_proba(&audio, sr)
  }

  /// Reset to the initial streaming state — alias for [`Self::initial_state`]
  /// (the reference's `Model.reset_state`, silero_vad.py:157-160).
  pub fn reset_state(&self, batch_size: i32, sample_rate: u32) -> Result<SileroVadState> {
    self.initial_state(batch_size, sample_rate)
  }

  /// Detect speech segments in raw audio — port of `Model.get_speech_timestamps`
  /// (silero_vad.py:209-241).
  ///
  /// Prepares the audio ([`Self::prepare_audio`]), runs [`Self::predict_proba`],
  /// then collapses the per-frame probabilities to padded speech segments via
  /// [`probs_to_timestamps`]. Each non-`None` [`SpeechTimestampOptions`] field
  /// overrides the corresponding model-config default for this call only (the
  /// reference's per-call keyword overrides); a default (all-`None`) options uses
  /// the config values, identical to the timestamps
  /// [`VadModel::generate`](crate::audio::vad::VadModel::generate) emits.
  ///
  /// Sample indices are returned as [`SpeechSegment`]s;
  /// [`SpeechSegment::start_seconds`] / [`SpeechSegment::end_seconds`] give the
  /// reference's `return_seconds=True` view.
  pub fn get_speech_timestamps(
    &self,
    audio: &Array,
    sample_rate: u32,
    options: SpeechTimestampOptions,
  ) -> Result<Vec<SpeechSegment>> {
    let (audio, sr) = self.prepare_audio(audio, sample_rate)?;
    let audio_len = *audio.shape().last().unwrap_or(&0) as i64;
    let mut probabilities = self.predict_proba(&audio, sr)?;
    probabilities.eval()?;
    // probs[0] when batched (the reference's `probs = probabilities[0] if 2-D`)
    // — guarded for a ZERO-ROW batch (a `(0, …)` input), whose rank-2
    // probabilities have no row 0 to take: an empty batch has no frames, so
    // the segment extraction runs over an empty probability vector and
    // returns no timestamps (consistent with the empty-waveform contract).
    let probs_vec: Vec<f32> = if probabilities.ndim() == 2 {
      if probabilities.shape()[0] == 0 {
        Vec::new()
      } else {
        let mut row = probabilities.take_axis(&idx0(0)?, 0)?;
        row.eval()?;
        row.astype(Dtype::F32)?.to_vec::<f32>()?
      }
    } else {
      probabilities.astype(Dtype::F32)?.to_vec::<f32>()?
    };
    let cfg = &self.config;
    let threshold = options.threshold.unwrap_or(cfg.threshold());
    let min_speech_duration_ms = options
      .min_speech_duration_ms
      .unwrap_or(cfg.min_speech_duration_ms());
    let min_silence_duration_ms = options
      .min_silence_duration_ms
      .unwrap_or(cfg.min_silence_duration_ms());
    let speech_pad_ms = options.speech_pad_ms.unwrap_or(cfg.speech_pad_ms());
    // Validate the per-call overrides exactly as ModelConfig::validate does at
    // load — the override path would otherwise bypass it. A negative pad widens
    // start and shrinks end, which for a one-frame run yields a SpeechSegment
    // with start > end, violating the output invariant.
    if !threshold.is_finite() {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad get_speech_timestamps: threshold",
        "must be finite",
        format_smolstr!("{threshold}"),
      )));
    }
    for (field, v) in [
      ("min_speech_duration_ms", min_speech_duration_ms),
      ("min_silence_duration_ms", min_silence_duration_ms),
      ("speech_pad_ms", speech_pad_ms),
    ] {
      if v < 0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "silero_vad get_speech_timestamps: negative duration/padding override",
          "must be >= 0",
          format_smolstr!("{field}={v}"),
        )));
      }
    }
    Ok(probs_to_timestamps(
      &probs_vec,
      audio_len,
      sr,
      threshold,
      min_speech_duration_ms,
      min_silence_duration_ms,
      speech_pad_ms,
    ))
  }
}

/// Per-call overrides for [`SileroVadModel::get_speech_timestamps`] — the
/// keyword arguments of `Model.get_speech_timestamps` (silero_vad.py:209-241).
///
/// Each `None` field falls back to the model-config value, exactly like the
/// reference's `self.config.X if x is None else x`. The [`Default`] (all-`None`)
/// reproduces calling the reference with no keyword overrides.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeechTimestampOptions {
  /// Speech-probability decision threshold (config default when `None`).
  pub threshold: Option<f64>,
  /// Minimum speech-segment duration in ms (config default when `None`).
  pub min_speech_duration_ms: Option<i32>,
  /// Minimum silence duration in ms to close a segment (config default when `None`).
  pub min_silence_duration_ms: Option<i32>,
  /// Padding in ms added to each side of a segment (config default when `None`).
  pub speech_pad_ms: Option<i32>,
}

/// Linearly resample a waveform [`Array`] from `from` to `to` Hz via the
/// [`crate::audio::io::resample_linear`] host primitive — a rank-1 `(T,)`
/// waveform, or each row of a rank-2 `(B, T)` batch. Used by
/// [`SileroVadModel::prepare_audio`] for the reference's resample-to-16-kHz net.
fn resample_audio_array(audio: &Array, from: u32, to: u32) -> Result<Array> {
  let shape = audio.shape();
  // Defensive: an input with any zero dim has nothing to resample; return it
  // unchanged rather than concatenating an empty row set (the caller already
  // guards this, but keep the helper sound on its own).
  if shape.contains(&0) {
    return audio.try_clone();
  }
  let resample_1d = |samples: &[f32], leading_batch: bool| -> Result<Array> {
    let out = crate::audio::io::resample_linear(samples, from, to)?;
    let n = i32::try_from(out.len()).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad prepare_audio: resampled length",
        "must fit in i32",
        format_smolstr!("{}", out.len()),
      ))
    })?;
    if leading_batch {
      Array::from_slice::<f32>(&out, &[1, n])
    } else {
      Array::from_slice::<f32>(&out, &[n])
    }
  };
  match shape.len() {
    1 => {
      let samples = audio.astype(Dtype::F32)?.to_vec::<f32>()?;
      resample_1d(&samples, false)
    }
    2 => {
      let rows = shape[0];
      let f32_audio = audio.astype(Dtype::F32)?;
      let mut resampled_rows: Vec<Array> = Vec::with_capacity(rows);
      for r in 0..i32::try_from(rows).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "silero_vad prepare_audio: batch size",
          "must fit in i32",
          format_smolstr!("{rows}"),
        ))
      })? {
        let samples = f32_audio.take_axis(&idx0(r)?, 0)?.to_vec::<f32>()?;
        resampled_rows.push(resample_1d(&samples, true)?);
      }
      let refs: Vec<&Array> = resampled_rows.iter().collect();
      ops::shape::concatenate(&refs, 0)
    }
    _ => Err(Error::RankMismatch(RankMismatchPayload::new(
      "silero_vad prepare_audio: resample rank",
      shape.len() as u32,
      shape,
    ))),
  }
}

/// One detected speech run in raw sample indices before padding — the
/// `{"start": …, "end": …}` dict the reference accumulates
/// (`silero_vad.py:378-408`).
#[derive(Debug, Clone, Copy)]
struct Speech {
  start: i64,
  end: i64,
}

/// Collapse a per-frame probability sequence to padded speech segments — port
/// of `Model._probs_to_timestamps` ([silero_vad.py:360-427][silero]).
///
/// `probs` are the per-frame probabilities (as `f32`s drawn from the model
/// output via `probs.tolist()`). The hysteresis is verbatim: a `prob >=
/// threshold` opens a segment, a `prob < neg_threshold` (with `neg_threshold =
/// max(threshold-0.15, 0.01)`) starts the silence timer, and once
/// `chunk_start - temp_end >= min_silence_samples` the segment closes (kept
/// only if at least `min_speech_samples` long). A still-open segment at the end
/// is closed at `min(audio_len, n_frames * chunk_size)`. Segments are then
/// padded by `speech_pad_samples` on each side and merged.
///
/// `chunk_size` is fixed by sample rate (512 at 16 kHz, 256 at 8 kHz —
/// `silero_vad.py:372`), independent of the branch STFT hop.
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L360-L427
#[allow(clippy::too_many_arguments)]
pub fn probs_to_timestamps(
  probs: &[f32],
  audio_len: i64,
  sample_rate: u32,
  threshold: f64,
  min_speech_duration_ms: i32,
  min_silence_duration_ms: i32,
  speech_pad_ms: i32,
) -> Vec<SpeechSegment> {
  let sr = sample_rate as f64;
  let chunk_size: i64 = if sample_rate == 16_000 { 512 } else { 256 };
  // The reference keeps these as floats (no truncation) for the comparisons.
  let min_speech_samples = sr * f64::from(min_speech_duration_ms) / 1000.0;
  let min_silence_samples = sr * f64::from(min_silence_duration_ms) / 1000.0;
  let speech_pad_samples = (sr * f64::from(speech_pad_ms) / 1000.0) as i64;
  let neg_threshold = (threshold - 0.15).max(0.01);

  let mut speeches: Vec<Speech> = Vec::new();
  let mut triggered = false;
  let mut current_start: i64 = 0;
  let mut temp_end: i64 = 0;

  for (idx, &prob) in probs.iter().enumerate() {
    let prob = f64::from(prob);
    let chunk_start = idx as i64 * chunk_size;

    if prob >= threshold && !triggered {
      triggered = true;
      current_start = chunk_start;
      temp_end = 0;
      continue;
    }
    if triggered && prob >= threshold {
      temp_end = 0;
      continue;
    }
    if triggered && prob < neg_threshold {
      if temp_end == 0 {
        temp_end = chunk_start;
      }
      if (chunk_start - temp_end) as f64 >= min_silence_samples {
        if (temp_end - current_start) as f64 >= min_speech_samples {
          speeches.push(Speech {
            start: current_start,
            end: temp_end,
          });
        }
        triggered = false;
        temp_end = 0;
      }
    }
  }

  if triggered {
    let end = audio_len.min(probs.len() as i64 * chunk_size);
    if (end - current_start) as f64 >= min_speech_samples {
      speeches.push(Speech {
        start: current_start,
        end,
      });
    }
  }

  // Pad each segment by `speech_pad` on each side and COALESCE when a padded
  // start overlaps the previous padded end — a faithful port of mlx-audio's
  // `silero_vad.py:410-417` (`if padded and start <= padded[-1]["end"]:
  // padded[-1]["end"] = max(padded[-1]["end"], end)`). This intentionally
  // matches mlx-audio's merge-on-overlap, NOT the upstream PyTorch snakers4
  // Silero `get_speech_timestamps`, which instead splits short inter-segment
  // silence between neighbors — our directive is 1:1 mlx-audio parity.
  let mut padded: Vec<Speech> = Vec::new();
  for speech in &speeches {
    let start = (speech.start - speech_pad_samples).max(0);
    let end = audio_len.min(speech.end + speech_pad_samples);
    if let Some(last) = padded.last_mut()
      && start <= last.end
    {
      last.end = last.end.max(end);
    } else {
      padded.push(Speech { start, end });
    }
  }

  padded
    .into_iter()
    .map(|s| SpeechSegment::new(s.start.max(0) as u64, s.end.max(0) as u64))
    .collect()
}

/// Drop the non-model `val_*` keys from a checkpoint weight map — port of
/// `Model.sanitize` ([silero_vad.py:429-431][silero]). The reference filters
/// `{k: v for k, v in weights.items() if not k.startswith("val_")}`.
///
/// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L429-L431
pub fn sanitize(weights: HashMap<String, Array>) -> HashMap<String, Array> {
  weights
    .into_iter()
    .filter(|(k, _)| !k.starts_with("val_"))
    .collect()
}

/// Internal helper for the loader: assemble a [`SileroVadBranch`] from its
/// already-extracted weight tensors. Kept here (next to [`SileroVadBranch`]'s
/// private fields) so the struct stays fully encapsulated.
///
/// The two LSTM weights are transposed once here (`Wx.T` / `Wh.T`) so the
/// per-frame forward reuses the loop-invariant transposed views instead of
/// rebuilding them every chunk.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_branch(
  config: BranchConfig,
  stft_conv_weight: Array,
  conv1: (Array, Array),
  conv2: (Array, Array),
  conv3: (Array, Array),
  conv4: (Array, Array),
  lstm_wx: &Array,
  lstm_wh: &Array,
  lstm_bias: Array,
  final_conv_weight: Array,
  final_conv_bias: Array,
) -> Result<SileroVadBranch> {
  // mlx.nn.LSTM stores Wx as (4H, D), Wh as (4H, H), bias as (4H,). Infer the
  // hidden size H from Wx's leading dim instead of hardcoding 128, and validate
  // the three LSTM tensors are mutually consistent so a wrong-shaped checkpoint
  // fails closed here with a clear error rather than mis-splitting the gates deep
  // in the recurrence.
  let wx_shape = lstm_wx.shape();
  let wh_shape = lstm_wh.shape();
  let bias_shape = lstm_bias.shape();
  if wx_shape.len() != 2 || wh_shape.len() != 2 || bias_shape.len() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "silero_vad lstm weights: expected Wx (4H, D), Wh (4H, H), bias (4H,)",
      wx_shape.len() as u32,
      wx_shape,
    )));
  }
  let four_h = wx_shape[0];
  if four_h == 0 || !four_h.is_multiple_of(4) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "silero_vad lstm Wx leading dim",
      "must be a positive multiple of 4 (the 4*hidden gate stack)",
      format_smolstr!("{four_h}"),
    )));
  }
  let hidden = four_h / 4;
  if wh_shape[0] != four_h || wh_shape[1] != hidden || bias_shape[0] != four_h {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "silero_vad lstm weights: Wh / bias inconsistent with Wx",
      "Wh must be (4H, H) and bias (4H,) for H = Wx.shape[0] / 4",
      format_smolstr!("Wx={wx_shape:?}, Wh={wh_shape:?}, bias={bias_shape:?}"),
    )));
  }
  let hidden_size = i32::try_from(hidden).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "silero_vad lstm hidden size",
      "must fit in i32",
      format_smolstr!("{hidden}"),
    ))
  })?;
  let wx_t = ops::shape::swapaxes(lstm_wx, -2, -1)?;
  let wh_t = ops::shape::swapaxes(lstm_wh, -2, -1)?;
  Ok(SileroVadBranch {
    config,
    stft_conv_weight,
    conv1: ConvBlock {
      weight: conv1.0,
      bias: conv1.1,
      stride: 1,
      padding: 1,
    },
    conv2: ConvBlock {
      weight: conv2.0,
      bias: conv2.1,
      stride: 2,
      padding: 1,
    },
    conv3: ConvBlock {
      weight: conv3.0,
      bias: conv3.1,
      stride: 2,
      padding: 1,
    },
    conv4: ConvBlock {
      weight: conv4.0,
      bias: conv4.1,
      stride: 1,
      padding: 1,
    },
    lstm: Lstm {
      wx_t,
      wh_t,
      bias: lstm_bias,
      hidden_size,
    },
    final_conv_weight,
    final_conv_bias,
  })
}

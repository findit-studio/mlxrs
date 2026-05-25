//! VLM multimodal input assembly — faithful 1:1 port of
//! `mlx-vlm/mlx_vlm/utils.py::prepare_inputs` and its audio/video glue
//! (`read_audio` / `load_audio` / `normalize_audio_features` /
//! `load_video`).
//!
//! ## Scope vs the python ref
//!
//! The python `prepare_inputs` (lines 1173–1449) is heavily coupled to a
//! HuggingFace `processor` object (it dispatches between
//! `processor.image_processor.preprocess` / `processor.tokenizer` /
//! `processor.feature_extractor` / per-class arg surface inspection).
//! mlxrs deliberately does NOT depend on HuggingFace's runtime — per the
//! `project_no_per_model_arch_porting` rule we port the model-agnostic
//! **algorithmic primitives**:
//!
//! 1. **Branch dispatch on content kind** — given pre-tokenized
//!    `text_token_batches: &[&[u32]]` and the optional image/audio/video
//!    payloads, decide which branches activate.
//! 2. **Padding-side handling** — pad varying-length token batches with
//!    `pad_token_id` to the max length, left- or right-padded per
//!    [`PaddingSide`].
//! 3. **`input_ids` + `attention_mask` array assembly** — stack into
//!    `[B, T]` `i32` / `bool` arrays (the python branch at lines
//!    1380–1392 does the same once the marker splice is resolved).
//! 4. **VLM-side audio/video glue** — [`read_audio`],
//!    [`load_audio_vlm`], [`normalize_audio_features`], [`load_video`].
//!    The audio entries wrap [`crate::audio::io::load_audio`] +
//!    [`crate::audio::io::resample_linear`] + the lossy-audio mean/std
//!    normalization at python line 1032–1034. [`load_video`] wraps
//!    [`crate::vlm::video::process_frames`] — frame decoding from a
//!    container is intentionally NOT ported (matches the existing
//!    `vlm/video.rs` policy: container decoding needs a codec
//!    dependency).
//!
//! Per-model `processor` calls (which the python ref delegates to) are
//! **out of scope** — those map 1:1 onto per-model arch impls (per the
//! `project_no_per_model_arch_porting` rule); a caller building per-model
//! support layers on top of mlxrs composes the per-model `processor`
//! with this primitive.
//!
//! ## Audio/video glue feature gating
//!
//! The audio glue wrappers ([`read_audio`], [`load_audio_vlm`],
//! [`normalize_audio_features`]) are gated on both the `vlm` AND `audio`
//! features (they bridge the two subsystems). [`load_video`] needs only
//! `vlm` (it wraps `vlm::video` primitives).

use crate::{
  array::Array,
  error::{Error, Result, try_with_capacity},
};

/// Padding side for varying-length `input_ids` batches — mirrors the
/// `padding_side: str = "left" | "right"` argument at
/// `mlx-vlm/mlx_vlm/utils.py:1183`.
///
/// `"left"` (the python default) is the right choice for autoregressive
/// generation: padding tokens before the prompt mean every batched
/// sequence's actual content lines up at the END (the position where
/// generation starts), so the same `position_ids[max_length-1]` queries
/// the correct token. `"right"` pads after the prompt, used by some
/// training/finetuning paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingSide {
  /// Pad on the LEFT (before the content). The python default.
  #[default]
  Left,
  /// Pad on the RIGHT (after the content).
  Right,
}

/// Output of [`prepare_inputs`] — the typed equivalent of the python
/// `model_inputs` dict (lines 1360–1449).
///
/// Carries the assembled `input_ids` + `attention_mask` arrays plus
/// optional multimodal payload arrays (pixel values, audio features,
/// video frames). The python ref builds a dict where each modality's
/// presence is implicit by key; the Rust port makes it explicit via
/// `Option<Array>` fields so a caller can `match`/`if let` on what was
/// actually built.
#[derive(Debug)]
pub struct PreparedInputs {
  /// `[B, T]` `i32` token ids, padded with `pad_token_id` per
  /// [`PaddingSide`]. Mirrors `model_inputs["input_ids"]` in the python
  /// ref (line 1387, line 1210).
  pub input_ids: Array,
  /// `[B, T]` `bool` mask — `true` at non-pad positions, `false` at
  /// padded positions. Mirrors `model_inputs["attention_mask"]` (lines
  /// 1390–1392, 1417–1419).
  pub attention_mask: Array,
  /// `[B, C, H, W]` (or whatever the per-model image processor emits)
  /// `f32` pixel values. `None` when no images were passed. Mirrors
  /// `model_inputs["pixel_values"]` (line 1389, line 1414).
  pub pixel_values: Option<Array>,
  /// Audio features `Array` — `None` when no audio was passed. Mirrors
  /// `model_inputs["input_features"]` (lines 1432, 1441–1447).
  pub input_features: Option<Array>,
  /// Per-frame video pixels `Array` — `None` when no videos were
  /// passed. Mirrors `model_inputs["pixel_values_videos"]` /
  /// equivalent for the video branch (the python ref names it
  /// per-processor).
  pub pixel_values_videos: Option<Array>,
}

/// Options for [`prepare_inputs`] — captures the python kwargs at
/// `mlx-vlm/mlx_vlm/utils.py:1173–1187` that affect input ASSEMBLY (the
/// per-processor kwargs that drive the HF dispatch are excluded by
/// scope — see the module doc).
#[derive(Debug, Clone)]
pub struct PrepareInputsOpts {
  /// `pad_token_id` — the token id used to pad varying-length
  /// `input_ids` batches. Mirrors `processor.pad_token_id` (line 1384,
  /// 1391). Required: there is no implicit default — the python ref
  /// reads it from the processor; the Rust port forces the caller to
  /// supply one because no in-process tokenizer is mandatory.
  pub pad_token_id: u32,
  /// `padding` — whether to pad varying-length sequences. Mirrors
  /// `padding: bool = True` (line 1182). `Default::default()` resolves
  /// to `true` via [`PrepareInputsOpts::default`] (see the impl
  /// override below).
  pub padding: bool,
  /// `padding_side` — see [`PaddingSide`]. Mirrors line 1183.
  pub padding_side: PaddingSide,
  /// Optional caller-supplied per-batch `attention_mask`. **When
  /// `Some(masks)`**: `masks` must have shape `[B][T_b]` with
  /// `masks.len() == text_token_batches.len()` AND
  /// `masks[i].len() == text_token_batches[i].len()` for each `i` —
  /// the supplied mask is treated as already authoritative and is
  /// padded with `false` (per [`PaddingSide`]) to match the padded
  /// `input_ids` shape (so any `false` positions inside the caller's
  /// mask survive into the output).
  ///
  /// **When `None` (default)**: the mask is computed from the internal
  /// padding step (positions filled with `pad_token_id` are marked
  /// `false`, all caller-supplied tokens are marked `true`).
  ///
  /// Mirrors HuggingFace's `tokenizer(..., return_tensors="pt",
  /// padding=True)` pattern where the tokenizer can EITHER autopad
  /// (the default branch) OR accept a pre-padded batch + an explicit
  /// `attention_mask` from the caller. Without this knob, a caller who
  /// pre-padded their batches upstream (uniform lengths) would have
  /// every position — including the pre-pads — marked `true` by the
  /// internal step, which silently corrupts model semantics (the
  /// padded positions get attended to). The explicit-mask path closes
  /// that hole.
  pub attention_mask: Option<Vec<Vec<bool>>>,
}

impl Default for PrepareInputsOpts {
  /// Mirrors the python `prepare_inputs` kwargs defaults:
  /// `pad_token_id=0`, `padding=true` (line 1182),
  /// `padding_side="left"` (line 1183), `attention_mask=None`
  /// (compute from the internal padding step).
  fn default() -> Self {
    Self {
      pad_token_id: 0,
      padding: true,
      padding_side: PaddingSide::Left,
      attention_mask: None,
    }
  }
}

/// Branch-dispatch + padding-side multimodal assembler — faithful 1:1
/// port of the model-agnostic core of
/// `mlx-vlm/mlx_vlm/utils.py::prepare_inputs` (lines 1173–1449).
///
/// `text_token_batches` is a `[B][T_b]` slice of pre-tokenized prompts
/// (one per batch entry; varying length is permitted and handled by the
/// padding branch). `pixel_values` / `input_features` /
/// `pixel_values_videos` carry the optional multimodal payloads —
/// caller-supplied because the per-model image processor / feature
/// extractor / video decoder are out of scope per the
/// `project_no_per_model_arch_porting` rule.
///
/// ## Behavior
///
/// - **Text-only** (`text_token_batches` only) → mirrors the python
///   text-only branch at lines 1196–1223: pad batches per
///   [`PrepareInputsOpts::padding_side`] and return only `input_ids` +
///   `attention_mask`.
/// - **Multimodal** (any of `pixel_values` / `input_features` /
///   `pixel_values_videos` supplied) → same padded `input_ids` +
///   `attention_mask` assembly, plus the corresponding optional
///   payload arrays (passed through verbatim — preprocessing has
///   already been done by the caller).
///
/// ## Padding semantics
///
/// - `padding=true` (default): pad to `max(T_b)` with `pad_token_id`,
///   left- or right-side per [`PrepareInputsOpts::padding_side`].
/// - `padding=false`: the batches must all be the same length already
///   (otherwise `Error::ShapeMismatch`).
///
/// ## Attention-mask semantics
///
/// - `opts.attention_mask = None` (default): mask is computed from the
///   internal padding step — `false` at padded positions (filled with
///   `opts.pad_token_id`), `true` at every caller-supplied token.
/// - `opts.attention_mask = Some(masks)`: caller-supplied per-batch
///   per-token mask is used directly. Required to have
///   `masks.len() == text_token_batches.len()` and
///   `masks[i].len() == text_token_batches[i].len()` for every `i`
///   (otherwise `Error::ShapeMismatch`); the supplied mask is then
///   padded with `false` (per [`PaddingSide`]) to match the padded
///   `input_ids` shape. Use this path when you pre-padded the batches
///   yourself upstream and you need padding positions to be marked
///   `false` (the internal step has no way to detect pre-padding).
///
/// # Errors
///
/// - `Error::ShapeMismatch` if `text_token_batches` is empty.
/// - `Error::ShapeMismatch` if `padding=false` and the batches have
///   varying lengths.
/// - `Error::ShapeMismatch` if any per-batch `T_b > i32::MAX` (mlx
///   dimension limit).
/// - `Error::ShapeMismatch` if `opts.attention_mask` is `Some(masks)`
///   and `masks` has the wrong outer or inner dimensions.
/// - `Error::OutOfMemory` if the row buffers cannot be allocated.
pub fn prepare_inputs(
  text_token_batches: &[&[u32]],
  pixel_values: Option<Array>,
  input_features: Option<Array>,
  pixel_values_videos: Option<Array>,
  opts: &PrepareInputsOpts,
) -> Result<PreparedInputs> {
  if text_token_batches.is_empty() {
    return Err(Error::ShapeMismatch {
      message: "prepare_inputs: text_token_batches is empty (need >= 1 batch entry)".into(),
    });
  }

  // Validate caller-supplied attention_mask shape if present. We do
  // this BEFORE any allocation so a dimension-mismatch path is cheap.
  if let Some(masks) = &opts.attention_mask {
    if masks.len() != text_token_batches.len() {
      return Err(Error::ShapeMismatch {
        message: format!(
          "prepare_inputs: opts.attention_mask outer length {} != text_token_batches.len() {}",
          masks.len(),
          text_token_batches.len()
        ),
      });
    }
    for (i, (m, b)) in masks.iter().zip(text_token_batches.iter()).enumerate() {
      if m.len() != b.len() {
        return Err(Error::ShapeMismatch {
          message: format!(
            "prepare_inputs: opts.attention_mask[{i}] length {} != text_token_batches[{i}] \
             length {}",
            m.len(),
            b.len()
          ),
        });
      }
    }
  }

  // Determine target T. With padding, this is max(T_b); without, it's
  // the common T_b (else error).
  let batch_size = text_token_batches.len();
  let lens: Vec<usize> = text_token_batches.iter().map(|b| b.len()).collect();
  let target_t = if opts.padding {
    *lens.iter().max().unwrap_or(&0)
  } else {
    let first = lens[0];
    if !lens.iter().all(|&l| l == first) {
      return Err(Error::ShapeMismatch {
        message: format!(
          "prepare_inputs: padding=false but batches have varying lengths ({lens:?}); \
           enable padding or pre-pad upstream"
        ),
      });
    }
    first
  };

  // mlx dimension limit: signed 32-bit. Reject early before the host
  // allocation.
  if target_t > i32::MAX as usize || batch_size > i32::MAX as usize {
    return Err(Error::ShapeMismatch {
      message: format!(
        "prepare_inputs: batch_size ({batch_size}) or target_t ({target_t}) exceeds i32::MAX \
         (mlx dimension limit)"
      ),
    });
  }

  // Total cell count with overflow guard (request-scaled allocation).
  let total = batch_size
    .checked_mul(target_t)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "prepare_inputs: batch_size * target_t overflows usize \
         (batch_size={batch_size}, target_t={target_t})"
      ),
    })?;

  // Row-major fill — `i32` ids (mlx convention: token ids are i32),
  // padded per opts.padding_side. The mask is either the caller's
  // per-token mask (padded with false in the pad positions) or the
  // internal "everything-true-except-pads" mask.
  let mut ids_buf: Vec<i32> = try_with_capacity(total)?;
  let mut mask_buf: Vec<bool> = try_with_capacity(total)?;
  let caller_mask = opts.attention_mask.as_deref();
  for (b, batch) in text_token_batches.iter().enumerate() {
    let pad_count = target_t - lens[b];
    // Resolve the mask source for this row: caller's per-token slice
    // (length already validated == batch.len()) OR `None` → fill
    // `true` at every caller-token position.
    let row_mask: Option<&[bool]> = caller_mask.map(|m| m[b].as_slice());
    match opts.padding_side {
      PaddingSide::Left => {
        for _ in 0..pad_count {
          ids_buf.push(opts.pad_token_id as i32);
          mask_buf.push(false);
        }
        for (i, &t) in batch.iter().enumerate() {
          ids_buf.push(t as i32);
          mask_buf.push(row_mask.is_none_or(|m| m[i]));
        }
      }
      PaddingSide::Right => {
        for (i, &t) in batch.iter().enumerate() {
          ids_buf.push(t as i32);
          mask_buf.push(row_mask.is_none_or(|m| m[i]));
        }
        for _ in 0..pad_count {
          ids_buf.push(opts.pad_token_id as i32);
          mask_buf.push(false);
        }
      }
    }
  }

  let input_ids = Array::from_slice::<i32>(&ids_buf, &(batch_size, target_t))?;
  let attention_mask = Array::from_slice::<bool>(&mask_buf, &(batch_size, target_t))?;

  Ok(PreparedInputs {
    input_ids,
    attention_mask,
    pixel_values,
    input_features,
    pixel_values_videos,
  })
}

// ==========================================================================
// VLM-side audio/video glue
// ==========================================================================

/// Read an audio file as `(samples_mono_f32, sample_rate)` — faithful
/// 1:1 port of `mlx-vlm/mlx_vlm/utils.py::read_audio` (lines 852–994),
/// minus the ffmpeg subprocess path (m4a/aac/ogg/opus that the python
/// ref shells out to ffmpeg for are not supported here — see
/// [`crate::audio::io::load_audio`]).
///
/// The python ref returns `(samples_2d, sample_rate)` where samples is
/// shape `(N, channels)`; this port returns the mono [`Vec<f32>`] from
/// [`crate::audio::io::load_audio`] (per that primitive's contract; see
/// the module doc on its single-channel restriction).
///
/// # Errors
///
/// Propagates from [`crate::audio::io::load_audio`] — file IO, codec,
/// non-mono, decoded-count overflow.
#[cfg(all(feature = "vlm", feature = "audio"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "vlm", feature = "audio"))))]
pub fn read_audio(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
  crate::audio::io::load_audio(path)
}

/// `load_audio(file, sr)` — faithful 1:1 port of
/// `mlx-vlm/mlx_vlm/utils.py::load_audio` (lines 997–1029).
///
/// Reads an audio file as mono `Vec<f32>` (via [`read_audio`]) and
/// resamples to `sr` Hz if needed (via
/// [`crate::audio::io::resample_linear`]). The python ref's URL branch
/// (`http://` / `https://`) is intentionally NOT ported — per project
/// policy mlxrs does not download from the network (see
/// `feedback_no_hf_hub_download`).
///
/// Returns the mono float32 sample buffer. Mirrors the python
/// `audio.mean(axis=1) if audio.ndim > 1 else audio` (line 1029)
/// downmix; mlxrs's [`crate::audio::io::load_audio`] already enforces
/// single-channel input, so the downmix is a no-op here (the python ref
/// supports stereo on the read side via miniaudio).
///
/// The function is named `load_audio_vlm` (not `load_audio`) because
/// [`crate::audio::io::load_audio`] already occupies the natural name;
/// this is the VLM-side wrapper that adds the SR-matching resample.
///
/// # Errors
///
/// Propagates from [`read_audio`] and
/// [`crate::audio::io::resample_linear`].
#[cfg(all(feature = "vlm", feature = "audio"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "vlm", feature = "audio"))))]
pub fn load_audio_vlm(path: &std::path::Path, sr: u32) -> Result<Vec<f32>> {
  let (samples, sample_rate) = read_audio(path)?;
  if sample_rate != sr {
    crate::audio::io::resample_linear(&samples, sample_rate, sr)
  } else {
    Ok(samples)
  }
}

/// Normalize mel-spectrogram features `(features - mean) / (std + 1e-6)`
/// — faithful 1:1 port of
/// `mlx-vlm/mlx_vlm/utils.py::normalize_audio_features` (lines 1032–1034).
///
/// The python ref applies a SINGLE-SCALAR `mx.mean` /
/// `mx.std` over the entire features tensor (NOT per-row / per-column),
/// then broadcasts the scalar offset/scale back. This is the lossy-audio
/// (MP3/AAC) feature normalization at python line 1295.
///
/// # Errors
///
/// Propagates from [`Array::mean`] / [`Array::std`] /
/// arithmetic ops on the input.
#[cfg(all(feature = "vlm", feature = "audio"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "vlm", feature = "audio"))))]
pub fn normalize_audio_features(features: &Array) -> Result<Array> {
  // python: (features - mx.mean(features)) / (mx.std(features) + 1e-6)
  // — single-scalar mean/std over the entire features tensor.
  // `keepdims=false` yields a 0-D scalar Array; ddof=0 to match
  // python's `mx.std` default (mlx-core's `std` defaults to ddof=0).
  let mean = features.mean(false)?;
  let std = features.std(false, 0)?;
  // `centered = features - mean` (scalar broadcast).
  let centered = crate::ops::arithmetic::subtract(features, &mean)?;
  // `denom = std + 1e-6` (scalar add via the 1-elem Array trick).
  let eps = Array::full::<f32>(&(1usize,), 1e-6_f32)?;
  let denom = crate::ops::arithmetic::add(&std, &eps)?;
  crate::ops::arithmetic::divide(&centered, &denom)
}

/// `load_video(video_path, fps=2.0, nframes=None, min_frames=4,
/// max_frames=768, frame_factor=2)` — faithful 1:1 port of
/// `mlx-vlm/mlx_vlm/utils.py::load_video` (lines 1037–1099) **arithmetic
/// only**: chooses an `nframes`, picks the frame indices, then delegates
/// the per-frame preprocessing to [`crate::vlm::video::process_frames`].
///
/// Container decoding (`mp4` → frames) is **intentionally NOT ported**
/// — it needs a codec dependency (the python ref uses `cv2.VideoCapture`).
/// This entry point exposes the model-agnostic sampling math and the
/// preprocessing composition; the caller is expected to provide a
/// decoded frame slice. See [`crate::vlm::video`]'s module doc for the
/// same codec-deferral policy.
///
/// `frames` is the caller-decoded `[T, H, W, 3]` RGB frame slice (one
/// `DynamicImage` per frame). `cfg` is the per-frame image processor
/// configuration. Returns the stacked `[T, H, W, 3]` `Array` from
/// [`crate::vlm::video::process_frames`].
///
/// `fps` / `nframes` / `min_frames` / `max_frames` / `frame_factor` are
/// accepted as a [`crate::vlm::video::FrameSampling`] config — the
/// caller chooses how many frames to sample via
/// [`crate::vlm::video::smart_nframes`] and selects indices via
/// [`crate::vlm::video::sample_frame_indices`] before passing the
/// sampled frames here. This split mirrors `vlm/video.rs`'s existing
/// API decomposition (sampling math + per-frame preprocessing).
///
/// **`cfg.layout` constraint — only
/// [`crate::vlm::image::Layout::Hwc`] is currently supported.**
/// `load_video` delegates to
/// [`crate::vlm::video::process_frames`], whose `[T, H, W, 3]` stack
/// contract is broken if a planar [`crate::vlm::image::Layout`] is
/// applied per frame. Video-tensor layout semantics are not yet pinned
/// to a per-model contract; callers needing planar output should
/// post-process the returned `[T, H, W, 3]` themselves until a future
/// PR adds first-class video-layout semantics. See
/// [`crate::vlm::video::process_frames`] for the full rationale.
///
/// # Errors
///
/// Propagates from [`crate::vlm::video::process_frames`] (including
/// [`crate::Error::Backend`] when `cfg.layout != Layout::Hwc`).
#[cfg(feature = "vlm")]
pub fn load_video(
  frames: &[::image::DynamicImage],
  cfg: &crate::vlm::image::ImageProcessorConfig,
) -> Result<Array> {
  crate::vlm::video::process_frames(frames, cfg)
}

//! The Silero VAD file-loading factory — `sanitize`, the weight-map → branch
//! assembly, the safetensors shard walk, and the [`load`] →
//! `Box<dyn VadModel>` factory the VAD registry dispatches `model_type ==
//! "silero_vad"` to.
//!
//! Faithful to the reference loading path: `Model.sanitize`
//! ([silero_vad.py:429-431][silero], the `val_*` drop, in [`super::model`]) and
//! the shared mlx-audio `base_load_model` pipeline (`get_model_path` →
//! `load_config` → `apply_quantization` → the per-architecture weight load),
//! reusing the same shared helpers the whisper / wav2vec2 / sensevoice loaders
//! do.
//!
//! Silero ships native MLX `nn.Conv1d` / `nn.LSTM` weights, so the per-branch
//! tensors are loaded verbatim — `<branch>.<layer>.weight` / `.bias` for the
//! convs, `<branch>.lstm.{Wx,Wh,bias}` for the LSTM — with no transpose
//! (unlike the HF-layout wav2vec2 checkpoint, which `swapaxes` its conv
//! weights). The `VADOutput` the model returns wires through the existing
//! [`crate::audio::vad::output::VadOutput`].
//!
//! [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

use smol_str::format_smolstr;

use crate::{
  array::Array,
  audio::vad::{
    load::VadModel,
    models::silero_vad::{
      config::{BranchConfig, ModelConfig},
      model::{SileroVadBranch, SileroVadModel, build_branch, sanitize},
    },
    output::{SpeechSegment, VadOutput},
  },
  error::{
    Error, FileIoPayload, FileOp, LayerKeyedPayload, MissingKeyPayload, OutOfRangePayload, Result,
  },
};

impl VadModel for SileroVadModel {
  /// Run VAD inference — port of `Model.generate`
  /// ([silero_vad.py:243-266][silero]).
  ///
  /// Runs [`SileroVadModel::predict_proba`] for the per-frame probabilities,
  /// evaluates them, then collapses them to padded speech-segment timestamps
  /// via [`crate::audio::vad::models::silero_vad::model::probs_to_timestamps`]
  /// using the model config's thresholds. The audio length used for the final
  /// segment clamp + padding is the input waveform length (the `(T,)` last
  /// axis).
  ///
  /// [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L243-L266
  fn generate(&self, audio: &Array, sample_rate: u32) -> Result<VadOutput> {
    // Preprocess (stereo downmix + resample-to-supported-rate) before inference,
    // exactly as the reference's `Model.generate` runs `_prepare_audio_array`
    // first (silero_vad.py:244). `sample_rate` below is the RESOLVED rate.
    let (audio, sample_rate) = self.prepare_audio(audio, sample_rate)?;
    let audio_len = *audio.shape().last().unwrap_or(&0) as i64;
    let mut probabilities = self.predict_proba(&audio, sample_rate)?;
    probabilities.eval()?;

    // probs.tolist(): the reference indexes probabilities[0] when 2-D; here
    // predict_proba already returns the 1-D vector for a 1-D input. For a
    // batched input we take the first row (matching `probs = probabilities[0]
    // if probabilities.ndim == 2`) — guarded for a ZERO-ROW batch (a `(0, …)`
    // input), whose rank-2 probabilities have no row 0 to take: an empty
    // batch has no frames, so the segment extraction runs over an empty
    // vector and the output carries no timestamps (the empty-waveform
    // contract).
    let probs_vec: Vec<f32> = if probabilities.ndim() == 2 {
      if probabilities.shape()[0] == 0 {
        Vec::new()
      } else {
        probabilities
          .take_axis(&Array::from_slice::<i32>(&[0], &[0i32; 0])?, 0)?
          .astype(crate::dtype::Dtype::F32)?
          .to_vec::<f32>()?
      }
    } else {
      probabilities
        .try_clone()?
        .astype(crate::dtype::Dtype::F32)?
        .to_vec::<f32>()?
    };

    let cfg = self.config();
    let timestamps: Vec<SpeechSegment> =
      crate::audio::vad::models::silero_vad::model::probs_to_timestamps(
        &probs_vec,
        audio_len,
        sample_rate,
        cfg.threshold(),
        cfg.min_speech_duration_ms(),
        cfg.min_silence_duration_ms(),
        cfg.speech_pad_ms(),
      );

    Ok(VadOutput {
      timestamps,
      probabilities,
      sample_rate,
    })
  }
}

impl SileroVadModel {
  /// Build a [`SileroVadModel`] from a parsed [`ModelConfig`] and a
  /// **sanitized** weight map (the `val_*`-stripped tensors).
  ///
  /// Each branch's tensors are looked up by the `mlx.nn` attribute-path keys
  /// the reference modules produce — `<branch>.stft_conv.weight`,
  /// `<branch>.conv{1..4}.{weight,bias}`, `<branch>.lstm.{Wx,Wh,bias}`,
  /// `<branch>.final_conv.{weight,bias}` — and loaded verbatim (Silero ships
  /// native MLX layouts).
  ///
  /// # Errors
  /// - [`Error::MissingKey`] naming the first absent tensor key.
  pub fn from_weights(config: ModelConfig, mut weights: HashMap<String, Array>) -> Result<Self> {
    // Silero ships dense checkpoints only (the reference never quantizes its tiny
    // model). A checkpoint carrying `<prefix>.scales` siblings is quantized;
    // loading it dense would silently misinterpret the packed weights, so fail
    // closed (the `.scales`-presence discriminator the other audio loaders share —
    // here a rejection, since Silero has no quantized path).
    if has_relevant_scales(&weights) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "silero_vad: quantized checkpoint",
        "Silero VAD is dense-only; a quantized checkpoint (with .scales tensors) is unsupported",
        "quantized",
      )));
    }
    let vad_16k = build_branch_from_weights(*config.branch_16k(), &mut weights, "vad_16k")?;
    let vad_8k = build_branch_from_weights(*config.branch_8k(), &mut weights, "vad_8k")?;
    Ok(Self::new(config, vad_16k, vad_8k))
  }

  /// Load a [`SileroVadModel`] from a local on-disk model directory — the
  /// convenience entry point mirroring mlx-audio's `vad.load` for this
  /// architecture.
  ///
  /// Pipeline (mirroring the shared loader):
  /// 1. [`crate::audio::load::get_model_path`] — resolve `path` to a local
  ///    directory (a Hub id is rejected per the no-network policy);
  /// 2. [`crate::audio::load::load_config`] — read + bound `config.json`,
  ///    parse the [`ModelConfig`];
  /// 3. `load_all_safetensors` — walk + merge the `*.safetensors` shards;
  /// 4. [`sanitize`] — drop the `val_*` keys;
  /// 5. [`SileroVadModel::from_weights`] — build the model.
  ///
  /// # Errors
  /// The errors of every pipeline step above (a missing directory / config, a
  /// malformed config, a missing or duplicated weight).
  pub fn load(path: &str) -> Result<Self> {
    let dir = crate::audio::load::get_model_path(path)?;
    let config_json = crate::audio::load::load_config(&dir)?;
    let config = ModelConfig::from_json(&config_json)?;

    let raw = load_all_safetensors(&dir)?;
    let weights = sanitize(raw);
    Self::from_weights(config, weights)
  }
}

/// Assemble one branch from the merged weight map under the `<branch>.` prefix,
/// moving each tensor out of the map (no clone). The map is consumed across the
/// two branch builds in [`SileroVadModel::from_weights`].
fn build_branch_from_weights(
  config: BranchConfig,
  weights: &mut HashMap<String, Array>,
  branch: &str,
) -> Result<SileroVadBranch> {
  let stft = take_weight(weights, branch, "stft_conv.weight")?;
  let conv1 = (
    take_weight(weights, branch, "conv1.weight")?,
    take_weight(weights, branch, "conv1.bias")?,
  );
  let conv2 = (
    take_weight(weights, branch, "conv2.weight")?,
    take_weight(weights, branch, "conv2.bias")?,
  );
  let conv3 = (
    take_weight(weights, branch, "conv3.weight")?,
    take_weight(weights, branch, "conv3.bias")?,
  );
  let conv4 = (
    take_weight(weights, branch, "conv4.weight")?,
    take_weight(weights, branch, "conv4.bias")?,
  );
  let lstm_wx = take_weight(weights, branch, "lstm.Wx")?;
  let lstm_wh = take_weight(weights, branch, "lstm.Wh")?;
  let lstm_bias = take_weight(weights, branch, "lstm.bias")?;
  let final_w = take_weight(weights, branch, "final_conv.weight")?;
  let final_b = take_weight(weights, branch, "final_conv.bias")?;

  build_branch(
    config, stft, conv1, conv2, conv3, conv4, &lstm_wx, &lstm_wh, lstm_bias, final_w, final_b,
  )
}

/// Remove the `<branch>.<suffix>` tensor from the weight map, or return a typed
/// [`Error::MissingKey`] naming it.
fn take_weight(weights: &mut HashMap<String, Array>, branch: &str, suffix: &str) -> Result<Array> {
  let key = format!("{branch}.{suffix}");
  weights
    .remove(&key)
    .ok_or_else(|| Error::MissingKey(MissingKeyPayload::new("silero_vad: missing weight", key)))
}

/// Detect a quantized Silero checkpoint: `true` if ANY tensor key ends in
/// `.scales`. Silero is dense-only, so any `.scales` sibling marks a quantized
/// (or mixed) checkpoint that [`SileroVadModel::from_weights`] rejects.
///
/// This is a COMPLETE `.scales`-presence check, not the `<prefix>.weight` +
/// `<prefix>.scales` sibling form quantize-aware loaders use: Silero consumes
/// its LSTM weights as `lstm.Wx` / `lstm.Wh` (not `*.weight`), so a
/// `lstm.Wx.scales` sibling would slip through a `.weight`-anchored check.
pub fn has_relevant_scales(weights: &HashMap<String, Array>) -> bool {
  weights.keys().any(|k| k.ends_with(".scales"))
}

/// Read and merge every `*.safetensors` shard under `dir` into one weight map —
/// the same shard walk the sensevoice / whisper loaders use (no hand-rolled
/// glob): sort the `.safetensors` entries by name and merge with
/// [`crate::model_validation::insert_unique`] so a cross-shard duplicate key
/// fails closed rather than silently overwriting an earlier tensor.
///
/// # Errors
/// - [`Error::FileIo`] if `dir` cannot be read (an entry fails mid-walk);
/// - [`Error::MissingKey`] if `dir` holds no `*.safetensors`;
/// - [`Error::LayerKeyed`] (the offending shard file name) wrapping an
///   [`Error::KeyCollision`] (the duplicated tensor key) if two shards define
///   the same key;
/// - propagates [`crate::io::load_safetensors`] read errors.
fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, Array>> {
  let entries = std::fs::read_dir(dir).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "silero_vad load: read model directory",
      FileOp::Read,
      dir.to_path_buf(),
      e,
    ))
  })?;
  let mut files: Vec<PathBuf> = entries
    .map(|entry| {
      entry.map(|e| e.path()).map_err(|e| {
        Error::FileIo(FileIoPayload::new(
          "silero_vad load: read model directory entry",
          FileOp::Read,
          dir.to_path_buf(),
          e,
        ))
      })
    })
    .collect::<Result<Vec<_>>>()?;
  files.retain(|p| p.extension().is_some_and(|ext| ext == "safetensors"));
  files.sort();
  if files.is_empty() {
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "silero_vad load: no *.safetensors in model directory",
      format_smolstr!("{}", dir.display()),
    )));
  }

  let mut all = HashMap::new();
  for f in &files {
    let shard = crate::io::load_safetensors(f)?;
    for (key, value) in shard {
      crate::model_validation::insert_unique(
        &mut all,
        key,
        value,
        "silero_vad load: duplicate tensor key across shards",
      )
      .map_err(|e| match e {
        Error::KeyCollision(_) => {
          Error::LayerKeyed(LayerKeyedPayload::new(f.to_string_lossy().into_owned(), e))
        }
        other => other,
      })?;
    }
  }
  Ok(all)
}

/// Load a Silero VAD model from a local on-disk directory, returning it as a
/// [`Box<dyn VadModel>`] — the [`crate::audio::vad::load()`] /
/// [`crate::audio::vad::load_model`] constructor closure target for
/// `model_type == "silero_vad"`.
///
/// Thin wrapper over [`SileroVadModel::load`] that boxes the concrete model
/// behind the [`VadModel`] trait object the shared VAD entry points return.
///
/// # Errors
/// Propagates [`SileroVadModel::load`]'s errors.
pub fn load(path: &str) -> Result<Box<dyn VadModel>> {
  Ok(Box::new(SileroVadModel::load(path)?))
}

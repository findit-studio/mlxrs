//! STS per-domain load entry points, ported from
//! [`mlx_audio.sts.utils`][sts-utils].
//!
//! Faithful 1:1 of mlx-audio's two-tier shape:
//!
//! - [`load_model`] ([sts-utils.py:112-163][sts-utils-loadmodel]) — the
//!   inner factory call mlx-audio funnels (post `_resolve_model_type`)
//!   into `base_load_model(model_path=…, category="sts",
//!   model_remapping=MODEL_REMAPPING, model_type=model_type, …)` plus
//!   the Moshi `from_pretrained(...)` early-return.
//! - [`load`] ([sts-utils.py:166-173][sts-utils-load]) — the alias over
//!   [`load_model`] (`return load_model(...)`). Mirrored for parity
//!   with `from mlx_audio.sts import load`.
//!
//! mlx-audio's `MODEL_REMAPPING` table ([sts-utils.py:13-26][sts-utils-remap])
//! routes STS architecture aliases (`deepfilter`/`deepfilternet`/`deepfilternet3`
//! → `deepfilternet`; `lfm_audio`/`lfm2_audio`/`lfm2.5` → `lfm_audio`;
//! `moshi`/`moshiko` → `moshi`; `mossformer2`/`mossformer2_se` →
//! `mossformer2_se`; `sam_audio`/`samaudio` → `sam_audio`) into
//! `mlx_audio.sts.models.<arch>`; per the no-per-model-arch rule mlxrs
//! returns a [`Model`] trait object the per-architecture loader's
//! caller constructs, so the table is exposed in
//! [`MODEL_REMAPPING`] for reference but no arch crate is imported
//! here.
//!
//! [sts-utils]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py
//! [sts-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py#L112-L163
//! [sts-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py#L166-L173
//! [sts-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py#L13-L26

use crate::{
  array::Array,
  audio::load::{LoadedAudioModel, base_load_model},
  error::Result,
};

/// The mlx-audio `MODEL_REMAPPING` table for STS architectures
/// ([sts-utils.py:13-26][sts-utils-remap]) — `(alias,
/// canonical_module_name)` pairs.
///
/// **Reference-only**: per the [no per-model arch porting][noarch]
/// rule, mlxrs does NOT import per-architecture crates from this table.
///
/// [sts-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py#L13-L26
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub const MODEL_REMAPPING: &[(&str, &str)] = &[
  ("deepfilter", "deepfilternet"),
  ("deepfilternet", "deepfilternet"),
  ("deepfilternet3", "deepfilternet"),
  ("lfm_audio", "lfm_audio"),
  ("lfm2_audio", "lfm_audio"),
  ("lfm2.5", "lfm_audio"),
  ("moshi", "moshi"),
  ("moshiko", "moshi"),
  ("mossformer2", "mossformer2_se"),
  ("mossformer2_se", "mossformer2_se"),
  ("sam_audio", "sam_audio"),
  ("samaudio", "sam_audio"),
];

/// The trait every per-architecture STS model implements — the analogue
/// of mlx-audio's per-arch `Model.process(audio) -> Array` /
/// `SAMAudio.separate(audio) -> SeparationResult` /
/// DeepFilterNet's `enhance(audio) -> Array` shape.
///
/// Per the [no per-model arch porting][noarch] rule, mlxrs ships no
/// concrete STS models; this trait is the *shape* per-architecture
/// crates (lfm_audio / sam_audio / deepfilternet / mossformer2_se /
/// moshi) implement so a caller can hand-off any STS architecture as a
/// `Box<dyn Model>` from [`load`] / [`load_model`].
///
/// `process` is the unified entry point (mlx-audio uses different
/// method names per architecture — `enhance` for noise suppression,
/// `separate` for source separation, generic forward for end-to-end
/// speech models; mlxrs unifies on `process` since the trait's contract
/// is "speech in, speech out" regardless of the per-model term).
///
/// `&self` because weights are immutable after load.
///
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub trait Model {
  /// Run STS inference on `audio`, returning the processed waveform.
  ///
  /// `audio` is the input waveform (typically a 1-D `(T,)` float
  /// [`Array`] in `[-1, 1]` at the model's expected sample rate). The
  /// returned [`Array`] is the architecture-specific output (e.g. an
  /// enhanced/denoised waveform for DeepFilterNet, separated sources
  /// for SAM-audio, an end-to-end response for LFM-audio).
  ///
  /// Per-architecture code wires the actual model forward.
  fn process(&self, audio: &Array) -> Result<Array>;

  /// The model's expected input/output PCM sample rate in Hz.
  ///
  /// Each mlx-audio STS architecture declares its own (`16_000` for
  /// most speech-enhancement / VAD-attached models; `48_000` for SAM-
  /// audio at 48k; etc.). Trait has no default — every concrete model
  /// must declare its rate.
  fn sample_rate(&self) -> u32;
}

/// Construct a [`Model`] from a local on-disk model directory —
/// faithful 1:1 of `mlx_audio.sts.utils.load_model`
/// ([sts-utils.py:112-163][sts-utils-loadmodel]).
///
/// Routes through the shared
/// [`crate::audio::load::base_load_model`] factory. The returned bundle
/// is handed to the caller-supplied `constructor` closure — per the
/// [no per-model arch porting][noarch] rule mlxrs does not bundle a
/// built-in registry / Moshi-special-case branch (the caller's
/// closure dispatches on `bundle.config_json` if it wants per-model
/// behavior).
///
/// `path` is the local on-disk path (a `hf://…` / `org/name` repo id
/// is rejected by [`crate::audio::load::get_model_path`] with a clear
/// no-network message).
///
/// Failures (missing dir / missing config / malformed JSON / constructor
/// error) are recoverable [`Error::Backend`].
///
/// [sts-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py#L112-L163
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
/// [`Error::Backend`]: crate::Error::Backend
pub fn load_model<F>(path: &str, constructor: F) -> Result<Box<dyn Model>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn Model>>,
{
  let bundle = base_load_model(path)?;
  constructor(bundle)
}

/// The **main entry point** for loading an STS model — faithful 1:1 of
/// `mlx_audio.sts.utils.load` ([sts-utils.py:166-173][sts-utils-load]).
/// Thin alias over [`load_model`].
///
/// [sts-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py#L166-L173
pub fn load<F>(path: &str, constructor: F) -> Result<Box<dyn Model>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn Model>>,
{
  load_model(path, constructor)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::{fs, path::PathBuf};

  struct FakeSts;

  impl Model for FakeSts {
    fn process(&self, audio: &Array) -> Result<Array> {
      let t = audio.size();
      Array::from_slice::<f32>(&vec![0.0; t], &(t,))
    }
    fn sample_rate(&self) -> u32 {
      16_000
    }
  }

  fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_audio_sts_load_{}_{}",
      std::process::id(),
      name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Factory + smoke test: constructor receives the resolved bundle;
  /// trait object's process + sample_rate are functional.
  #[test]
  fn load_sts_constructs_via_factory() {
    let dir = temp_dir("constructs_via_factory");
    let body = r#"{ "model_type": "deepfilternet", "sample_rate": 48000 }"#;
    fs::write(dir.join("config.json"), body).unwrap();

    let captured: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    let model: Box<dyn Model> = load(&dir.to_string_lossy(), |bundle| {
      *captured.borrow_mut() = Some(bundle.model_path().to_path_buf());
      Ok(Box::new(FakeSts))
    })
    .expect("load constructs via the supplied factory");

    assert_eq!(captured.into_inner().unwrap(), dir);
    assert_eq!(model.sample_rate(), 16_000);

    let probe = Array::from_slice::<f32>(&[0.0_f32; 8], &(8,)).unwrap();
    let out = model.process(&probe).unwrap();
    assert_eq!(out.shape(), vec![8]);
  }
}

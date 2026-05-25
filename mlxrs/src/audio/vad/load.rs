//! VAD per-domain load entry points, ported from
//! [`mlx_audio.vad.utils`][vad-utils].
//!
//! Faithful 1:1 of mlx-audio's two-tier shape:
//!
//! - [`load_model`] ([vad-utils.py:14-36][vad-utils-loadmodel]) — the
//!   inner factory call mlx-audio funnels into
//!   `base_load_model(model_path=…, category="vad",
//!   model_remapping=MODEL_REMAPPING, …)`.
//! - [`load`] ([vad-utils.py:39-64][vad-utils-load]) — the **main entry
//!   point**; thin alias over [`load_model`] (`return load_model(...)`).
//!   Carried as a separate function because mlx-audio's public surface
//!   carries both — the documented call site is `from mlx_audio.vad
//!   import load`.
//!
//! mlx-audio's `MODEL_REMAPPING` table ([vad-utils.py:8-11][vad-utils-remap])
//! routes per-architecture aliases (`silero` → `silero_vad`, `silero-vad`
//! → `silero_vad`) into the `mlx_audio.vad.models.<arch>` namespace; per
//! the no-per-model-arch rule mlxrs returns a [`VadModel`] trait object
//! (`Box<dyn VadModel>`) the per-architecture loader's caller constructs
//! itself, so the remap table is the **caller's** responsibility — this
//! module documents the mlx-audio names in [`MODEL_REMAPPING`] for
//! parity but never imports an arch crate.
//!
//! [vad-utils]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py
//! [vad-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py#L14-L36
//! [vad-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py#L39-L64
//! [vad-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py#L8-L11

use crate::{
  array::Array,
  audio::{
    load::{LoadedAudioModel, base_load_model},
    vad::output::VadOutput,
  },
  error::Result,
};

/// The mlx-audio `MODEL_REMAPPING` table for VAD architectures
/// ([vad-utils.py:8-11][vad-utils-remap]) — a static array of
/// `(alias, canonical_module_name)` pairs.
///
/// **Reference-only**: per the [no per-model arch porting][noarch] rule,
/// mlxrs does NOT import per-architecture crates from this table — it
/// exists purely so an end-of-pipeline caller (who DOES wire in concrete
/// architectures) can mirror mlx-audio's alias resolution. Add new
/// aliases here when the upstream table changes.
///
/// [vad-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py#L8-L11
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub const MODEL_REMAPPING: &[(&str, &str)] =
  &[("silero", "silero_vad"), ("silero-vad", "silero_vad")];

/// The trait every per-architecture VAD model implements — the analogue
/// of mlx-audio's per-arch `Model` class's
/// [`generate(audio, sample_rate=…) -> VADOutput`][vad-generate] method.
///
/// Per the [no per-model arch porting][noarch] rule, mlxrs ships no
/// concrete VAD models; this trait is the *shape* per-architecture
/// crates (silero_vad / sortformer / smart_turn / …) implement so a
/// caller can hand-off any VAD architecture as a `Box<dyn VadModel>`
/// from the [`load`] / [`load_model`] entry points.
///
/// `&self` because weights are immutable after load (matching every
/// other audio-domain trait in mlxrs — [`crate::audio::tts::model::TtsModel`]
/// / [`crate::audio::stt::model::Model`] — and mlx-audio's `nn.Module`
/// for inference).
///
/// [vad-generate]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L243-L266
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub trait VadModel {
  /// Run VAD inference on `audio` at `sample_rate` Hz, returning the
  /// per-frame probabilities + extracted speech timestamps — mirror of
  /// mlx-audio's [`Model.generate(audio, sample_rate=…)`][vad-generate].
  ///
  /// `audio` is the input waveform (typically a 1-D `(T,)` float
  /// [`Array`] in `[-1, 1]` at the model's expected sample rate).
  ///
  /// Per-architecture code wires the actual STFT / CNN / LSTM /
  /// sortformer body here.
  ///
  /// [vad-generate]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L243-L266
  fn generate(&self, audio: &Array, sample_rate: u32) -> Result<VadOutput>;
}

/// Construct a [`VadModel`] from a local on-disk model directory —
/// faithful 1:1 of `mlx_audio.vad.utils.load_model`
/// ([vad-utils.py:14-36][vad-utils-loadmodel]).
///
/// Routes through the shared
/// [`crate::audio::load::base_load_model`] factory (the analog of
/// mlx-audio's `return base_load_model(model_path=…, category="vad",
/// model_remapping=MODEL_REMAPPING, …)`). The returned
/// [`LoadedAudioModel`] bundle (model dir + verbatim config.json +
/// optional [`crate::lm::quant::PerLayerQuantization`]) is handed to
/// `constructor` — a caller-supplied closure that builds the concrete
/// per-architecture [`VadModel`] (the analog of mlx-audio's
/// `model_class.Model(model_config) + load_weights(...)`); per the
/// no-per-model-arch rule mlxrs does not bundle a built-in registry.
///
/// `path` is the local on-disk path (a `hf://…` / `org/name` repo id is
/// rejected by [`crate::audio::load::get_model_path`] with a clear
/// no-network message; see that function's docs).
///
/// Failures (missing dir / missing config / malformed JSON / constructor
/// error) are recoverable [`Error::Backend`].
///
/// [vad-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py#L14-L36
/// [`Error::Backend`]: crate::Error::Backend
pub fn load_model<F>(path: &str, constructor: F) -> Result<Box<dyn VadModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn VadModel>>,
{
  let bundle = base_load_model(path)?;
  constructor(bundle)
}

/// The **main entry point** for loading a VAD model — faithful 1:1 of
/// `mlx_audio.vad.utils.load` ([vad-utils.py:39-64][vad-utils-load]),
/// the thin alias over [`load_model`] mlx-audio's `from mlx_audio.vad
/// import load` call site uses.
///
/// Carried as a separate function because mlx-audio exposes it as
/// part of its documented public API
/// ([`mlx_audio.vad.__init__`][vad-init]: `__all__ = ["load",
/// "load_model"]`); mlxrs mirrors the two-name surface for parity.
///
/// [vad-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py#L39-L64
/// [vad-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/__init__.py
pub fn load<F>(path: &str, constructor: F) -> Result<Box<dyn VadModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn VadModel>>,
{
  load_model(path, constructor)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::audio::vad::output::SpeechSegment;
  use std::{fs, path::PathBuf};

  /// A fake VAD model the test uses as the constructor's output.
  struct FakeVad {
    sample_rate: u32,
  }

  impl VadModel for FakeVad {
    fn generate(&self, _audio: &Array, _sample_rate: u32) -> Result<VadOutput> {
      Ok(VadOutput {
        timestamps: vec![SpeechSegment::new(0, 1600)],
        probabilities: Array::from_slice::<f32>(&[0.9], &(1,)).unwrap(),
        sample_rate: self.sample_rate,
      })
    }
  }

  fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_audio_vad_load_{}_{}",
      std::process::id(),
      name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// The factory resolves the local dir, reads `config.json`, parses
  /// the optional quantization, then hands the bundle to the caller's
  /// constructor (the per-architecture `Model.from_config + load_weights`
  /// analog). Asserts the bundle the constructor receives has the
  /// resolved local-dir path + the verbatim JSON body — the contract
  /// the per-architecture loader downstream relies on.
  #[test]
  fn load_vad_constructs_via_factory() {
    let dir = temp_dir("constructs_via_factory");
    let body = r#"{ "model_type": "silero_vad" }"#;
    fs::write(dir.join("config.json"), body).unwrap();

    let captured: std::cell::RefCell<Option<(PathBuf, String)>> = std::cell::RefCell::new(None);
    let model = load(&dir.to_string_lossy(), |bundle| {
      assert!(
        bundle.quantization().is_none(),
        "dense config → no quantization"
      );
      *captured.borrow_mut() = Some((
        bundle.model_path().to_path_buf(),
        bundle.config_json().to_owned(),
      ));
      Ok(Box::new(FakeVad {
        sample_rate: 16_000,
      }))
    })
    .expect("load constructs via the supplied factory");

    // The factory fed the constructor the resolved local dir + the
    // verbatim JSON body.
    let (path, json) = captured.into_inner().expect("constructor was called");
    assert_eq!(path, dir);
    assert_eq!(json, body);

    // The trait object the factory returns is functional.
    let probe = Array::from_slice::<f32>(&[0.0], &(1,)).unwrap();
    let out = model.generate(&probe, 16_000).unwrap();
    assert_eq!(out.sample_rate, 16_000);
    assert_eq!(out.timestamps.len(), 1);
  }
}

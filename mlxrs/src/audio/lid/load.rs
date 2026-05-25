//! LID per-domain load entry points, ported from
//! [`mlx_audio.lid.utils`][lid-utils].
//!
//! Faithful 1:1 of mlx-audio's two-tier shape:
//!
//! - [`load_model`] ([lid-utils.py:16-38][lid-utils-loadmodel]) — the
//!   inner factory call mlx-audio funnels into
//!   `base_load_model(model_path=…, category="lid",
//!   model_remapping=MODEL_REMAPPING, …)`.
//! - [`load`] ([lid-utils.py:41-66][lid-utils-load]) — the **main entry
//!   point**; thin alias over [`load_model`] (`return load_model(...)`).
//!   Carried as a separate function because mlx-audio's public surface
//!   carries both (`from mlx_audio.lid import load`).
//!
//! mlx-audio's `MODEL_REMAPPING` table ([lid-utils.py:10-13][lid-utils-remap])
//! routes the two ecapa-tdnn aliases (`ecapa-tdnn` / `ecapa_tdnn`) into
//! `mlx_audio.lid.models.ecapa_tdnn`; per the no-per-model-arch rule
//! mlxrs returns a [`LidModel`] trait object the per-architecture
//! loader's caller constructs, so the table is exposed in
//! [`MODEL_REMAPPING`] for reference but no arch crate is imported
//! here.
//!
//! [lid-utils]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py
//! [lid-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L16-L38
//! [lid-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L41-L66
//! [lid-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L10-L13

use crate::{
  array::Array,
  audio::{
    lid::output::{LidOutput, LidPrediction},
    load::{LoadedAudioModel, base_load_model},
  },
  error::Result,
};

/// mlx-audio's documented sample rate for every LID architecture —
/// [lid-utils.py:8][lid-utils-sr] (`SAMPLE_RATE = 16000`).
///
/// Exposed here so a caller wiring a per-architecture [`LidModel`] can
/// resample input audio to the model's expected rate without restating
/// the constant.
///
/// [lid-utils-sr]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L8
pub const LID_SAMPLE_RATE: u32 = 16_000;

/// The mlx-audio `MODEL_REMAPPING` table for LID architectures
/// ([lid-utils.py:10-13][lid-utils-remap]) — `(alias,
/// canonical_module_name)` pairs.
///
/// **Reference-only**: per the [no per-model arch porting][noarch] rule,
/// mlxrs does NOT import per-architecture crates from this table — it
/// exists purely so an end-of-pipeline caller can mirror mlx-audio's
/// alias resolution.
///
/// [lid-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L10-L13
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub const MODEL_REMAPPING: &[(&str, &str)] =
  &[("ecapa-tdnn", "ecapa_tdnn"), ("ecapa_tdnn", "ecapa_tdnn")];

/// The trait every per-architecture LID model implements — the analogue
/// of mlx-audio's `Model.predict(audio, top_k=…) -> List[Tuple[str,
/// float]]` ([wav2vec_lid.py:101-148][lid-predict-wav2vec2],
/// [ecapa_tdnn.py:135-163][lid-predict-ecapa]).
///
/// Per the [no per-model arch porting][noarch] rule, mlxrs ships no
/// concrete LID models; this trait is the *shape* per-architecture
/// crates (wav2vec2 / ecapa_tdnn) implement so a caller can hand-off
/// any LID architecture as a `Box<dyn LidModel>` from the [`load`] /
/// [`load_model`] entry points.
///
/// `&self` because weights are immutable after load.
///
/// [lid-predict-wav2vec2]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/models/wav2vec2/wav2vec_lid.py#L101-L148
/// [lid-predict-ecapa]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/models/ecapa_tdnn/ecapa_tdnn.py#L135-L163
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub trait LidModel {
  /// Run LID inference on `audio` (typically a 16 kHz mono float
  /// waveform — see [`LID_SAMPLE_RATE`]), returning the top-`top_k`
  /// predictions sorted by probability descending — mirror of mlx-audio's
  /// `Model.predict(audio, top_k=…)`.
  ///
  /// `top_k` mirrors mlx-audio's `top_k=5` default; the caller passes
  /// the desired cap (the per-architecture implementation may also
  /// impose an internal cap based on the model's `id2label` size).
  fn predict(&self, audio: &Array, top_k: usize) -> Result<LidOutput>;
}

/// Helper: build an [`LidOutput`] from the raw `(language_code,
/// probability)` pairs an architecture's `predict` implementation
/// computes. Threading parity with mlx-audio's `return [(label, prob)
/// for …]` list-comprehension output.
///
/// `pairs` is **assumed sorted by probability descending** — the
/// per-architecture code's responsibility, matching mlx-audio's `sorted
/// (…, reverse=True)` precondition.
pub fn lid_output_from_pairs<I>(pairs: I) -> LidOutput
where
  I: IntoIterator<Item = (String, f32)>,
{
  LidOutput::new(
    pairs
      .into_iter()
      .map(|(language_code, probability)| LidPrediction::new(language_code, probability))
      .collect(),
  )
}

/// Construct a [`LidModel`] from a local on-disk model directory —
/// faithful 1:1 of `mlx_audio.lid.utils.load_model`
/// ([lid-utils.py:16-38][lid-utils-loadmodel]).
///
/// Routes through the shared
/// [`crate::audio::load::base_load_model`] factory (the analog of
/// mlx-audio's `return base_load_model(model_path=…, category="lid",
/// model_remapping=MODEL_REMAPPING, …)`). The returned bundle is
/// handed to the caller-supplied `constructor` closure (per the
/// no-per-model-arch rule mlxrs does not bundle a built-in registry).
///
/// Failures (missing dir / missing config / malformed JSON / constructor
/// error) are recoverable [`Error::Backend`].
///
/// [lid-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L16-L38
/// [`Error::Backend`]: crate::Error::Backend
pub fn load_model<F>(path: &str, constructor: F) -> Result<Box<dyn LidModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn LidModel>>,
{
  let bundle = base_load_model(path)?;
  constructor(bundle)
}

/// The **main entry point** for loading an LID model — faithful 1:1 of
/// `mlx_audio.lid.utils.load` ([lid-utils.py:41-66][lid-utils-load]).
/// Thin alias over [`load_model`].
///
/// [lid-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py#L41-L66
pub fn load<F>(path: &str, constructor: F) -> Result<Box<dyn LidModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn LidModel>>,
{
  load_model(path, constructor)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::{fs, path::PathBuf};

  struct FakeLid;

  impl LidModel for FakeLid {
    fn predict(&self, _audio: &Array, top_k: usize) -> Result<LidOutput> {
      let mut predictions = vec![
        LidPrediction::new("eng", 0.95),
        LidPrediction::new("fra", 0.03),
      ];
      predictions.truncate(top_k);
      Ok(LidOutput::new(predictions))
    }
  }

  fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_audio_lid_load_{}_{}",
      std::process::id(),
      name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Factory threads the resolved local-dir path + verbatim JSON body
  /// through the constructor; the returned trait object is functional.
  #[test]
  fn load_lid_constructs_via_factory() {
    let dir = temp_dir("constructs_via_factory");
    let body = r#"{ "model_type": "ecapa_tdnn", "num_labels": 107 }"#;
    fs::write(dir.join("config.json"), body).unwrap();

    let captured: std::cell::RefCell<Option<(PathBuf, String)>> = std::cell::RefCell::new(None);
    let model = load(&dir.to_string_lossy(), |bundle| {
      *captured.borrow_mut() = Some((
        bundle.model_path().to_path_buf(),
        bundle.config_json().to_owned(),
      ));
      Ok(Box::new(FakeLid))
    })
    .expect("load constructs via the supplied factory");

    let (path, json) = captured.into_inner().expect("constructor was called");
    assert_eq!(path, dir);
    assert_eq!(json, body);

    let probe = Array::from_slice::<f32>(&[0.0_f32; 16_000], &(16_000,)).unwrap();
    let out = model.predict(&probe, 2).unwrap();
    assert_eq!(out.predictions_slice().len(), 2);
    assert_eq!(out.predictions_slice()[0].language_code(), "eng");
  }

  /// Helper roundtrip: `lid_output_from_pairs` builds a sorted-input
  /// list into the typed envelope.
  #[test]
  fn lid_output_from_pairs_preserves_order() {
    let out = lid_output_from_pairs([
      ("eng".to_string(), 0.7),
      ("deu".to_string(), 0.2),
      ("fra".to_string(), 0.1),
    ]);
    assert_eq!(out.predictions_slice().len(), 3);
    assert_eq!(out.predictions_slice()[0].language_code(), "eng");
    assert_eq!(out.predictions_slice()[2].language_code(), "fra");
  }
}

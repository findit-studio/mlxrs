//! STT per-domain load entry points, ported from
//! [`mlx_audio.stt.utils`][stt-utils].
//!
//! Faithful 1:1 of mlx-audio's two-tier shape:
//!
//! - [`load_model`] ([stt-utils.py:64-89][stt-utils-loadmodel]) — the
//!   inner factory call mlx-audio funnels into
//!   `base_load_model(model_path=…, category="stt",
//!   model_remapping=MODEL_REMAPPING, …)`.
//! - [`load`] ([stt-utils.py:92-115][stt-utils-load]) — the **main
//!   entry point**; thin alias over [`load_model`]. Carried for parity
//!   with `from mlx_audio.stt import load`.
//!
//! mlx-audio's `MODEL_REMAPPING` table ([stt-utils.py:12-26][stt-utils-remap])
//! routes per-architecture aliases (`cohere_asr`, `fireredasr2`, `glm`
//! → `glmasr`, `sensevoice`, `voxtral`, `voxtral_realtime`, `vibevoice`
//! → `vibevoice_asr`, `qwen3_asr`, `canary`, `moonshine`, `mms`,
//! `granite_speech`, `qwen2_audio`) into
//! `mlx_audio.stt.models.<arch>`; per the no-per-model-arch rule
//! mlxrs returns a [`crate::audio::stt::model::Transcribe`] trait object the
//! per-architecture loader's caller constructs.
//!
//! [stt-utils]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py
//! [stt-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L64-L89
//! [stt-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L92-L115
//! [stt-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L12-L26

use crate::{
  audio::{
    load::{LoadedAudioModel, base_load_model},
    stt::model::Transcribe,
  },
  error::Result,
};

/// mlx-audio's documented sample rate for every STT architecture —
/// [stt-utils.py:10][stt-utils-sr] (`SAMPLE_RATE = 16000`).
///
/// Exposed here so a caller wiring a per-architecture STT model can
/// resample input audio to the model's expected rate without restating
/// the constant.
///
/// [stt-utils-sr]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L10
pub const STT_SAMPLE_RATE: u32 = 16_000;

/// The mlx-audio `MODEL_REMAPPING` table for STT architectures
/// ([stt-utils.py:12-26][stt-utils-remap]) — `(alias,
/// canonical_module_name)` pairs.
///
/// **Reference-only**: per the no per-model arch porting
/// rule, mlxrs does NOT import per-architecture crates from this table.
///
/// [stt-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L12-L26
pub const MODEL_REMAPPING: &[(&str, &str)] = &[
  ("cohere_asr", "cohere_asr"),
  ("fireredasr2", "fireredasr2"),
  ("glm", "glmasr"),
  ("sensevoice", "sensevoice"),
  ("voxtral", "voxtral"),
  ("voxtral_realtime", "voxtral_realtime"),
  ("vibevoice", "vibevoice_asr"),
  ("qwen3_asr", "qwen3_asr"),
  ("canary", "canary"),
  ("moonshine", "moonshine"),
  ("mms", "mms"),
  ("granite_speech", "granite_speech"),
  ("qwen2_audio", "qwen2_audio"),
];

/// Construct a [`Transcribe`] STT model from a local on-disk model directory —
/// faithful 1:1 of `mlx_audio.stt.utils.load_model`
/// ([stt-utils.py:64-89][stt-utils-loadmodel]).
///
/// Routes through the shared
/// [`crate::audio::load::base_load_model`] factory. The returned bundle
/// is handed to the caller-supplied `constructor` closure (per the
/// no per-model arch porting rule mlxrs does not bundle a
/// built-in registry). The constructor returns the universal
/// [`Transcribe`] contract; whether the concrete model is CTC or
/// autoregressive is its own detail.
///
/// `path` is the local on-disk path (a `hf://…` / `org/name` repo id is
/// rejected by [`crate::audio::load::get_model_path`] with a clear
/// no-network message).
///
/// Failures are typed: missing dir →
/// [`Error::MissingKey`](crate::error::Error::MissingKey), hub path →
/// [`Error::OutOfRange`](crate::error::Error::OutOfRange), malformed JSON →
/// [`Error::Parse`](crate::error::Error::Parse), constructor
/// error → caller-defined.
///
/// [stt-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L64-L89
pub fn load_model<F>(path: &str, constructor: F) -> Result<Box<dyn Transcribe>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn Transcribe>>,
{
  let bundle = base_load_model(path)?;
  constructor(bundle)
}

/// The **main entry point** for loading an STT model — faithful 1:1 of
/// `mlx_audio.stt.utils.load` ([stt-utils.py:92-115][stt-utils-load]).
/// Thin alias over [`load_model`].
///
/// [stt-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/utils.py#L92-L115
pub fn load<F>(path: &str, constructor: F) -> Result<Box<dyn Transcribe>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn Transcribe>>,
{
  load_model(path, constructor)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    array::Array,
    audio::stt::model::{TranscribeOptions, Transcription},
  };
  use std::{fs, path::PathBuf};

  /// A fake STT model implementing the universal [`Transcribe`] contract
  /// directly (the family-trait wiring is exercised in `generate`'s tests);
  /// here it returns a fixed transcription so the factory path can be tested
  /// through a trait object.
  struct FakeStt;

  impl Transcribe for FakeStt {
    fn transcribe(&self, _audio: &Array, _opts: &TranscribeOptions) -> Result<Transcription> {
      Ok(Transcription::new("fake transcript", None, Vec::new()))
    }
  }

  fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_audio_stt_load_{}_{}",
      std::process::id(),
      name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Factory + smoke test: the constructor receives the resolved bundle and
  /// the returned [`Transcribe`] trait object is functional (the per-model
  /// decode branches stay per-arch in user code).
  #[test]
  fn load_stt_constructs_via_factory() {
    let dir = temp_dir("constructs_via_factory");
    let body = r#"{ "model_type": "whisper", "n_mels": 80 }"#;
    fs::write(dir.join("config.json"), body).unwrap();

    let captured: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    let model = load(&dir.to_string_lossy(), |bundle| {
      *captured.borrow_mut() = Some(bundle.model_path().to_path_buf());
      Ok(Box::new(FakeStt))
    })
    .expect("load constructs via the supplied factory");

    assert_eq!(captured.into_inner().unwrap(), dir);
    let audio = Array::from_slice::<f32>(&[0.0_f32; 4], &[4]).unwrap();
    let out = model
      .transcribe(&audio, &TranscribeOptions::new())
      .expect("trait object transcribe");
    assert_eq!(out.text(), "fake transcript");
  }
}

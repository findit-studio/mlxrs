//! TTS per-domain load entry points, ported from
//! [`mlx_audio.tts.utils`][tts-utils].
//!
//! Faithful 1:1 of mlx-audio's two-tier shape:
//!
//! - [`load_model`] ([tts-utils.py:98-127][tts-utils-loadmodel]) — the
//!   inner factory call mlx-audio funnels into
//!   `base_load_model(model_path=…, category="tts",
//!   model_remapping=MODEL_REMAPPING, …)`.
//! - [`load`] ([tts-utils.py:130-153][tts-utils-load]) — the **main
//!   entry point**; thin alias over [`load_model`]. Mirrored for parity
//!   with `from mlx_audio.tts import load`.
//!
//! mlx-audio's `MODEL_REMAPPING` table ([tts-utils.py:19-45][tts-utils-remap])
//! is the largest in mlx-audio (29+ entries covering qwen3_tts, outetts,
//! spark, sesame/csm/marvis, voxcpm, vibevoice, chatterbox_turbo,
//! soprano, bailingmm, kitten_tts, echo_tts, fish_qwen3_omni,
//! irodori_tts, voxtral_tts, kugelaudio, longcat_audiodit, omnivoice,
//! melotts, moss_tts*); per the no-per-model-arch rule mlxrs returns a
//! [`crate::audio::tts::model::TtsModel`] trait object the
//! per-architecture loader's caller constructs.
//!
//! [tts-utils]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py
//! [tts-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py#L98-L127
//! [tts-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py#L130-L153
//! [tts-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py#L19-L45

use crate::{
  audio::{
    load::{LoadedAudioModel, base_load_model},
    tts::model::TtsModel,
  },
  error::Result,
};

/// The mlx-audio `MODEL_REMAPPING` table for TTS architectures
/// ([tts-utils.py:19-45][tts-utils-remap]) — `(alias,
/// canonical_module_name)` pairs.
///
/// **Reference-only**: per the no per-model arch porting
/// rule, mlxrs does NOT import per-architecture crates from this table.
///
/// [tts-utils-remap]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py#L19-L45
pub const MODEL_REMAPPING: &[(&str, &str)] = &[
  ("qwen3_tts", "qwen3_tts"),
  ("outetts", "outetts"),
  ("spark", "spark"),
  ("marvis", "sesame"),
  ("csm", "sesame"),
  ("voxcpm", "voxcpm"),
  ("voxcpm1.5", "voxcpm"),
  ("voxcpm2", "voxcpm2"),
  ("vibevoice_streaming", "vibevoice"),
  ("chatterbox_turbo", "chatterbox_turbo"),
  ("soprano", "soprano"),
  ("bailingmm", "bailingmm"),
  ("kitten", "kitten_tts"),
  ("echo_tts", "echo_tts"),
  ("fish_qwen3_omni", "fish_qwen3_omni"),
  ("irodori_tts", "irodori_tts"),
  ("voxtral_tts", "voxtral_tts"),
  ("kugelaudio", "kugelaudio"),
  ("audiodit", "longcat_audiodit"),
  ("longcat", "longcat_audiodit"),
  ("omnivoice", "omnivoice"),
  ("melotts", "melotts"),
  ("moss_tts_nano", "moss_tts_nano"),
  ("moss_tts_delay", "moss_tts"),
  ("moss_tts_local", "moss_tts"),
];

/// Construct a [`TtsModel`] from a local on-disk model directory —
/// faithful 1:1 of `mlx_audio.tts.utils.load_model`
/// ([tts-utils.py:98-127][tts-utils-loadmodel]).
///
/// Routes through the shared
/// [`crate::audio::load::base_load_model`] factory. The returned bundle
/// is handed to the caller-supplied `constructor` closure (per the
/// no per-model arch porting rule mlxrs does not bundle a
/// built-in registry).
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
/// [tts-utils-loadmodel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py#L98-L127
pub fn load_model<F>(path: &str, constructor: F) -> Result<Box<dyn TtsModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn TtsModel>>,
{
  let bundle = base_load_model(path)?;
  constructor(bundle)
}

/// The **main entry point** for loading a TTS model — faithful 1:1 of
/// `mlx_audio.tts.utils.load` ([tts-utils.py:130-153][tts-utils-load]).
/// Thin alias over [`load_model`].
///
/// [tts-utils-load]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/utils.py#L130-L153
pub fn load<F>(path: &str, constructor: F) -> Result<Box<dyn TtsModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn TtsModel>>,
{
  load_model(path, constructor)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::{fs, path::PathBuf};

  struct FakeTts;

  impl TtsModel for FakeTts {
    fn sample_rate(&self) -> u32 {
      24_000
    }
  }

  fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_audio_tts_load_{}_{}",
      std::process::id(),
      name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Factory + smoke test: constructor receives the resolved bundle;
  /// trait object's metadata is functional.
  #[test]
  fn load_tts_constructs_via_factory() {
    let dir = temp_dir("constructs_via_factory");
    let body = r#"{ "model_type": "kokoro" }"#;
    fs::write(dir.join("config.json"), body).unwrap();

    let captured: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    let model = load(&dir.to_string_lossy(), |bundle| {
      *captured.borrow_mut() = Some(bundle.model_path().to_path_buf());
      Ok(Box::new(FakeTts))
    })
    .expect("load constructs via the supplied factory");

    assert_eq!(captured.into_inner().unwrap(), dir);
    assert_eq!(model.sample_rate(), 24_000);
  }
}

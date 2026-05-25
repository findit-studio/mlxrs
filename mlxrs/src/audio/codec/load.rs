//! Codec per-domain load entry points.
//!
//! mlx-audio's [`codec/__init__.py`][codec-init] is a bare re-export
//! list (no shared `load` helper). mlxrs adds a [`load`] /
//! [`load_model`] entry point for parity with the other audio domains
//! (VAD / LID / STS / STT / TTS): a per-domain factory that routes
//! through the shared [`crate::audio::load::base_load_model`] and
//! returns a [`CodecModel`] trait object the per-architecture loader's
//! caller constructs (per the [no per-model arch porting][noarch] rule
//! mlxrs does not bundle a built-in codec registry).
//!
//! The per-architecture codec classes (DAC, Encodec, Mimi,
//! MossAudioTokenizer, Vocos, …) typically expose their own
//! `from_pretrained(...)` shim in mlx-audio; the closure passed to
//! [`load`] is the rust analog (the caller wires the
//! `Model.from_config(...) + load_weights(...)` body inside its
//! constructor).
//!
//! [codec-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/codec/__init__.py
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

use crate::{
  array::Array,
  audio::load::{LoadedAudioModel, base_load_model},
  error::{Error, Result},
};

/// Uniform `MODEL_REMAPPING` table for codec architectures — empty for
/// parity with the other audio domains' load surfaces
/// ([`crate::audio::tts::load::MODEL_REMAPPING`] et al.).
///
/// mlx-audio's [`codec/__init__.py`][codec-init] ships no remapping
/// (per-codec classes are imported by name with no alias table), so this
/// constant is intentionally empty — exposed only so generic caller code
/// `audio::<domain>::load::MODEL_REMAPPING` can iterate every domain's
/// remap table uniformly without a per-domain branch.
///
/// [codec-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/codec/__init__.py
pub const MODEL_REMAPPING: &[(&str, &str)] = &[];

/// The trait every per-architecture neural-audio codec implements — the
/// encode/decode pair every codec mlx-audio ships
/// ([`codec/__init__.py`][codec-init]'s `DAC`, `Encodec`, `Mimi`,
/// `MossAudioTokenizer`, `Vocos`, `StepAudio2Token2Wav`,
/// `EcapaTdnnBackbone`) implements.
///
/// Per the [no per-model arch porting][noarch] rule, mlxrs ships no
/// concrete codecs; this trait is the *shape* per-architecture crates
/// implement so a caller can hand-off any codec as a
/// `Box<dyn CodecModel>` from [`load`] / [`load_model`].
///
/// `&self` because weights are immutable after load.
///
/// [codec-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/codec/__init__.py
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
pub trait CodecModel {
  /// Encode a waveform into discrete (or continuous) codec codes.
  ///
  /// `audio` is the input waveform (typically a 1-D or 2-D float
  /// [`Array`] at the codec's expected sample rate — see
  /// [`CodecModel::sample_rate`]). The returned [`Array`] is the codec's
  /// code representation: typically `(B, K, T)` for RVQ codecs (Encodec /
  /// Mimi / DAC) or `(B, T, D)` continuous for VAE-style codecs (Vocos).
  ///
  /// Per-architecture code wires the actual encoder forward.
  ///
  /// Default impl is unsupported (`Err(Error::Backend)`) — every
  /// concrete codec MUST override it, matching mlx-audio's
  /// `Encodec.encode` / `Mimi.encode` per-class implementations.
  fn encode(&self, audio: &Array) -> Result<Array> {
    let _ = audio;
    Err(Error::Backend {
      message: "codec needs `encode` override (per-architecture)".into(),
    })
  }

  /// Decode codec codes back into a waveform.
  ///
  /// `codes` is the codec's code representation (the
  /// [`CodecModel::encode`] output shape). The returned [`Array`] is the
  /// reconstructed waveform at [`CodecModel::sample_rate`].
  ///
  /// Default impl is unsupported (`Err(Error::Backend)`) — every
  /// concrete codec MUST override it.
  fn decode(&self, codes: &Array) -> Result<Array> {
    let _ = codes;
    Err(Error::Backend {
      message: "codec needs `decode` override (per-architecture)".into(),
    })
  }

  /// The codec's output PCM sample rate in Hz.
  ///
  /// Each mlx-audio codec declares its own (`24_000` for Encodec /
  /// Mimi / DAC most variants; `44_100` for the high-fidelity DAC
  /// variant; `16_000` for some speech-codec variants). Trait has no
  /// default — every concrete codec must declare its rate.
  fn sample_rate(&self) -> u32;
}

/// Construct a [`CodecModel`] from a local on-disk model directory.
///
/// Routes through the shared
/// [`crate::audio::load::base_load_model`] factory. The returned
/// bundle (model dir + verbatim config.json + optional
/// [`crate::lm::quant::PerLayerQuantization`]) is handed to the
/// caller-supplied `constructor` closure (the analog of mlx-audio's
/// per-codec `from_pretrained(...)` shim).
///
/// `path` is the local on-disk path (a `hf://…` / `org/name` repo id
/// is rejected by [`crate::audio::load::get_model_path`] with a clear
/// no-network message).
///
/// Failures (missing dir / missing config / malformed JSON / constructor
/// error) are recoverable [`Error::Backend`].
///
/// [`Error::Backend`]: crate::Error::Backend
pub fn load_model<F>(path: &str, constructor: F) -> Result<Box<dyn CodecModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn CodecModel>>,
{
  let bundle = base_load_model(path)?;
  constructor(bundle)
}

/// The **main entry point** for loading a codec model — alias over
/// [`load_model`], following the two-name surface pattern the other
/// audio domains expose (`from mlx_audio.{vad,lid,stt,tts,sts} import
/// load, load_model`).
pub fn load<F>(path: &str, constructor: F) -> Result<Box<dyn CodecModel>>
where
  F: FnOnce(LoadedAudioModel) -> Result<Box<dyn CodecModel>>,
{
  load_model(path, constructor)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::{fs, path::PathBuf};

  struct FakeCodec;

  impl CodecModel for FakeCodec {
    fn encode(&self, audio: &Array) -> Result<Array> {
      // Mock encoder: identity downcast to (1, 1, T).
      let t = audio.size();
      Array::from_slice::<f32>(&vec![0.0; t], &(1, 1, t))
    }
    fn decode(&self, codes: &Array) -> Result<Array> {
      let t = codes.shape().iter().product::<usize>();
      Array::from_slice::<f32>(&vec![0.0; t], &(t,))
    }
    fn sample_rate(&self) -> u32 {
      24_000
    }
  }

  fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_audio_codec_load_{}_{}",
      std::process::id(),
      name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Factory + smoke test: the constructor receives the resolved bundle;
  /// the trait object's metadata + roundtrip is functional.
  #[test]
  fn load_codec_constructs_via_factory() {
    let dir = temp_dir("constructs_via_factory");
    let body = r#"{ "model_type": "encodec", "sample_rate": 24000 }"#;
    fs::write(dir.join("config.json"), body).unwrap();

    let captured: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    let model = load(&dir.to_string_lossy(), |bundle| {
      *captured.borrow_mut() = Some(bundle.model_path().to_path_buf());
      Ok(Box::new(FakeCodec))
    })
    .expect("load constructs via the supplied factory");

    assert_eq!(captured.into_inner().unwrap(), dir);
    assert_eq!(model.sample_rate(), 24_000);

    let probe = Array::from_slice::<f32>(&[0.0_f32; 8], &(8,)).unwrap();
    let codes = model.encode(&probe).unwrap();
    let back = model.decode(&codes).unwrap();
    assert_eq!(back.shape().iter().product::<usize>(), 8);
  }
}

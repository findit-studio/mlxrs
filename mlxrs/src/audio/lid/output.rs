//! The shared LID inference-result struct, ported from
//! [`mlx_audio.lid.models.{wav2vec2,ecapa_tdnn}.predict`][lid-predict-wav2vec2]
//! [(ecapa)][lid-predict-ecapa].
//!
//! Both LID architectures mlx-audio ships expose their
//! `Model.predict(audio, top_k=…)` result as the same shape: a
//! `List[Tuple[str, float]]` of `(language_code, probability)` pairs,
//! sorted by probability descending. mlxrs spells the same shape as a
//! typed [`LidPrediction`] +  [`LidOutput`] so a per-architecture LID
//! model can return one [`LidOutput`] downstream consumers can read
//! uniformly.
//!
//! [lid-predict-wav2vec2]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/models/wav2vec2/wav2vec_lid.py#L101-L148
//! [lid-predict-ecapa]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/models/ecapa_tdnn/ecapa_tdnn.py#L135-L163

/// One `(language_code, probability)` prediction in a [`LidOutput`] —
/// port of the `Tuple[str, float]` mlx-audio's
/// `predict(…) -> List[Tuple[str, float]]` returns
/// ([wav2vec_lid.py:101-148][lid-predict-wav2vec2],
/// [ecapa_tdnn.py:135-163][lid-predict-ecapa]).
///
/// `language_code` is the model's `id2label[idx]` lookup result (e.g.
/// `"eng"`, `"fra"`) or the `"LABEL_<idx>"` fallback when the model
/// config does not carry an `id2label` map (mirror of mlx-audio's
/// `id2label.get(str(idx), f"LABEL_{idx}")`). `probability` is a softmax
/// score in `[0, 1]`.
///
/// [lid-predict-wav2vec2]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/models/wav2vec2/wav2vec_lid.py#L101-L148
/// [lid-predict-ecapa]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/models/ecapa_tdnn/ecapa_tdnn.py#L135-L163
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LidPrediction {
  /// The predicted language code (e.g. `"eng"`, `"fra"`, …) or the
  /// `"LABEL_<idx>"` fallback when no `id2label` map is configured.
  language_code: String,
  /// The softmax probability for this language (in `[0, 1]`).
  probability: f32,
}

impl LidPrediction {
  /// Construct a [`LidPrediction`] from a language code and probability.
  pub fn new(language_code: impl Into<String>, probability: f32) -> Self {
    Self {
      language_code: language_code.into(),
      probability,
    }
  }

  /// The predicted language code (e.g. `"eng"`, `"fra"`, …).
  #[inline(always)]
  pub fn language_code(&self) -> &str {
    &self.language_code
  }

  /// The softmax probability for this language (in `[0, 1]`).
  #[inline(always)]
  pub fn probability(&self) -> f32 {
    self.probability
  }
}

/// The result of one LID inference pass — port of mlx-audio's
/// `Model.predict(audio, top_k=…)` return shape.
///
/// mlx-audio returns a raw `List[Tuple[str, float]]`; mlxrs wraps the
/// list in a struct so a future consumer can compose a richer envelope
/// (e.g. the input sample rate, model id, …) without breaking the call
/// sites. The list is **already sorted by probability descending** —
/// matching mlx-audio's
/// `sorted(enumerate(probs_list), key=lambda x: x[1], reverse=True)` —
/// so the top-1 prediction is at `predictions[0]`.
///
/// The list length mirrors mlx-audio's `top_k=5` default but is fully
/// governed by the caller; mlxrs does not impose a top-k cap here — the
/// per-architecture loader decides.
///
/// Both the typed [`LidPrediction`] entries and the wrapping
/// [`LidOutput`] derive full serde so a result can be persisted to disk
/// (the common "save the top-k as a JSON sidecar" consumer).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LidOutput {
  /// The top-k `(language_code, probability)` predictions sorted by
  /// probability descending — port of mlx-audio's
  /// `[(id2label[idx], prob) for idx, prob in indexed[:top_k]]`.
  predictions: Vec<LidPrediction>,
}

impl LidOutput {
  /// Construct a [`LidOutput`] from a pre-built predictions list.
  pub fn new(predictions: Vec<LidPrediction>) -> Self {
    Self { predictions }
  }

  /// The top-k predictions as a slice (sorted by probability descending).
  #[inline(always)]
  pub fn predictions_slice(&self) -> &[LidPrediction] {
    &self.predictions
  }

  /// The top-1 prediction, if any. Convenience accessor for the common
  /// "I just want the language code" caller.
  pub fn top(&self) -> Option<&LidPrediction> {
    self.predictions.first()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Construct a minimal [`LidOutput`], assert field shapes match
  /// mlx-audio's sorted-descending order, and serde round-trip the
  /// whole envelope (every field is serde-derivable — no [`Array`]
  /// handle in [`LidOutput`]).
  #[test]
  fn lid_output_struct_round_trips() {
    let predictions = vec![
      LidPrediction::new("eng", 0.92),
      LidPrediction::new("fra", 0.05),
      LidPrediction::new("deu", 0.03),
    ];
    let out = LidOutput::new(predictions);

    // Top-1 is the highest-probability entry.
    let top = out.top().expect("non-empty predictions has a top");
    assert_eq!(top.language_code(), "eng");
    assert!((top.probability() - 0.92).abs() < 1e-6);

    // Full serde round-trip (the whole struct is serde-derivable).
    let s = serde_json::to_string(&out).unwrap();
    let back: LidOutput = serde_json::from_str(&s).unwrap();
    assert_eq!(back, out);
  }

  /// An empty `predictions` vector yields `top() == None` — the empty
  /// case the model returns when `top_k == 0` or the input is degenerate.
  #[test]
  fn lid_output_empty_top_is_none() {
    let out = LidOutput::new(Vec::new());
    assert!(out.top().is_none());
  }
}

//! Wav2Vec2 CTC speech recognizer (`facebook/wav2vec2-base-960h`).
//!
//! Port of mlx-audio's `Wav2Vec2ForCTC` — the backbone in
//! [`stt/models/wav2vec/wav2vec.py`][wav2vec] (feature encoder + feature
//! projection + transformer encoder) composed with the CTC head + greedy
//! decode + waveform normalization in [`stt/models/mms/mms.py`][mms]. MMS
//! *is* Wav2Vec2ForCTC; `base-960h` is the same architecture minus the MMS
//! language-adapter logic (which is dropped here).
//!
//! The model is **not** autoregressive, so it does not implement the
//! [`crate::audio::stt::model::Model`] trait (encoder + per-token
//! cross-attention `decode_step` + KV cache). Inference is a single forward
//! over the raw 16 kHz mono waveform producing per-frame logits `(B, T', V)`,
//! followed by a greedy CTC collapse over a character vocabulary. The public
//! surface is therefore inherent:
//!
//! - [`Wav2Vec2Ctc::forward`] — waveform `(B, T)` → logits `(B, T', V)`.
//! - [`Wav2Vec2Ctc::transcribe`] — waveform → decoded `String`
//!   (normalize → forward → greedy CTC collapse → vocabulary map).
//!
//! ## Architecture (`base-960h`)
//!
//! `hidden=768, layers=12, heads=12, intermediate=3072, vocab=32`,
//! conv stack `dim=(512,)*7, stride=(5,2,2,2,2,2,2), kernel=(10,3,3,3,3,2,2),
//! bias=False`, positional conv `kernel=128, groups=16`,
//! `feat_extract_norm="group"`, `do_stable_layer_norm=False`,
//! `hidden_act="gelu"` (exact). 1 s of audio → ~49 frames.
//!
//! [wav2vec]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py
//! [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  error::{
    Error, InvariantViolationPayload, LengthMismatchPayload, MissingKeyPayload, OutOfRangePayload,
    ParsePayload, RankMismatchPayload, Result,
  },
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::{GroupNorm, LayerNorm},
  },
  ops,
};

// ───────────────────────────── config ─────────────────────────────

/// Wav2Vec2 model configuration — the typed subset of HF `config.json`
/// mlx-audio's `ModelConfig` ([wav2vec.py][cfg]) reads, restricted to the
/// fields the inference forward pass actually consumes.
///
/// Defaults match `facebook/wav2vec2-base-960h`. Like mlx-audio's
/// `BaseModelArgs.from_dict` (and the rest of mlxrs's `#[serde(default)]`
/// configs), unmodeled keys parse cleanly and absent keys fall back to the
/// default — a forward-compatible read, not `deny_unknown_fields`.
///
/// [cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L27-L74
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Wav2Vec2Config {
  /// Architecture id (`config.json` `model_type`, e.g. `"wav2vec2"`).
  #[serde(default = "default_model_type")]
  model_type: String,
  /// Vocabulary size — the CTC head's output width / logits last axis.
  #[serde(default = "default_vocab_size")]
  pub vocab_size: i32,
  /// Transformer hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Number of transformer encoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// Number of attention heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Feed-forward intermediate dimension.
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// `eps` shared by every `LayerNorm` and the L0 `GroupNorm`.
  #[serde(default = "default_layer_norm_eps")]
  pub layer_norm_eps: f32,
  /// Feature-encoder normalization scheme. `base-960h` is `"group"` (the
  /// only scheme this port supports — see [`Wav2Vec2Config::is_group_norm`]).
  #[serde(default = "default_feat_extract_norm")]
  pub feat_extract_norm: String,
  /// Per-conv-layer output channel widths (length `num_feat_extract_layers`).
  #[serde(default = "default_conv_dim")]
  pub conv_dim: Vec<i32>,
  /// Per-conv-layer stride.
  #[serde(default = "default_conv_stride")]
  pub conv_stride: Vec<i32>,
  /// Per-conv-layer kernel size.
  #[serde(default = "default_conv_kernel")]
  pub conv_kernel: Vec<i32>,
  /// Whether the feature-encoder convolutions carry a bias (`base-960h`:
  /// `false`).
  #[serde(default)]
  pub conv_bias: bool,
  /// Positional-conv-embedding kernel size (even → one trailing frame is
  /// cropped by the SamePad step).
  #[serde(default = "default_num_conv_pos_embeddings")]
  pub num_conv_pos_embeddings: i32,
  /// Positional-conv-embedding group count (grouped depthwise-ish conv).
  #[serde(default = "default_num_conv_pos_embedding_groups")]
  pub num_conv_pos_embedding_groups: i32,
  /// Number of feature-extractor conv layers.
  #[serde(default = "default_num_feat_extract_layers")]
  pub num_feat_extract_layers: i32,
  /// Whether the encoder uses the stable-layer-norm (pre-norm) arm.
  /// `base-960h`: `false` (post-norm). The pre-norm arm is **not** ported
  /// (see module docs).
  #[serde(default)]
  pub do_stable_layer_norm: bool,
}

#[cfg(feature = "wav2vec2")]
fn default_model_type() -> String {
  "wav2vec2".to_string()
}
#[cfg(feature = "wav2vec2")]
fn default_vocab_size() -> i32 {
  32
}
#[cfg(feature = "wav2vec2")]
fn default_hidden_size() -> i32 {
  768
}
#[cfg(feature = "wav2vec2")]
fn default_num_hidden_layers() -> i32 {
  12
}
#[cfg(feature = "wav2vec2")]
fn default_num_attention_heads() -> i32 {
  12
}
#[cfg(feature = "wav2vec2")]
fn default_intermediate_size() -> i32 {
  3072
}
#[cfg(feature = "wav2vec2")]
fn default_layer_norm_eps() -> f32 {
  1e-5
}
#[cfg(feature = "wav2vec2")]
fn default_feat_extract_norm() -> String {
  "group".to_string()
}
#[cfg(feature = "wav2vec2")]
fn default_conv_dim() -> Vec<i32> {
  vec![512, 512, 512, 512, 512, 512, 512]
}
#[cfg(feature = "wav2vec2")]
fn default_conv_stride() -> Vec<i32> {
  vec![5, 2, 2, 2, 2, 2, 2]
}
#[cfg(feature = "wav2vec2")]
fn default_conv_kernel() -> Vec<i32> {
  vec![10, 3, 3, 3, 3, 2, 2]
}
#[cfg(feature = "wav2vec2")]
fn default_num_conv_pos_embeddings() -> i32 {
  128
}
#[cfg(feature = "wav2vec2")]
fn default_num_conv_pos_embedding_groups() -> i32 {
  16
}
#[cfg(feature = "wav2vec2")]
fn default_num_feat_extract_layers() -> i32 {
  7
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Wav2Vec2Config {
  /// Parse a [`Wav2Vec2Config`] from an in-memory `config.json` string.
  ///
  /// Mirrors mlx-audio's `ModelConfig.from_dict` (a `json.load` restricted to
  /// the known keys). A malformed-JSON failure maps to [`Error::Parse`]; an
  /// unmodeled key is ignored (forward-compatible) and an absent key takes
  /// the `base-960h` default.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Wav2Vec2Config::from_json",
        "config JSON",
        e,
      ))
    })
  }

  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// `true` when `feat_extract_norm == "group"` — the only feature-encoder
  /// normalization scheme this port supports (`base-960h`). The `"layer"`
  /// (stable-layer-norm) variant used by `large-960h-lv60` is intentionally
  /// not ported.
  #[inline(always)]
  pub fn is_group_norm(&self) -> bool {
    self.feat_extract_norm == "group"
  }

  /// Per-head dimension `hidden_size / num_attention_heads`.
  fn head_dim(&self) -> Result<i32> {
    if self.num_attention_heads <= 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Wav2Vec2Config: num_attention_heads",
        "must be positive (> 0)",
        format_smolstr!("{}", self.num_attention_heads),
      )));
    }
    if self.hidden_size % self.num_attention_heads != 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Wav2Vec2Config: hidden_size / num_attention_heads",
        "hidden_size must be divisible by num_attention_heads",
      )));
    }
    Ok(self.hidden_size / self.num_attention_heads)
  }
}

// ───────────────────────────── sanitize ─────────────────────────────

/// Rewrite an HF Wav2Vec2ForCTC checkpoint into the layout this port loads —
/// the Rust analogue of mlx-audio's `Model.sanitize` ([mms.py:107-128][san],
/// plus the backbone's `wav2vec2.`-prefix strip from
/// [wav2vec.py:723][san-bb]). Pure key/axis bookkeeping, no MLX evaluation
/// beyond the lazy `swapaxes`.
///
/// Rules (applied per `(key, value)`):
/// 1. Strip a leading `wav2vec2.` backbone prefix (HF nests the encoder under
///    it while `lm_head` stays top-level).
/// 2. `*.conv.weight` and `*.conv.weight_v` / `*.conv.weight_g`:
///    `swapaxes(1, 2)` (HF conv weight `(out, in, k)` → MLX channels-last
///    `(out, k, in)`).
/// 3. `*.parametrizations.weight.original0` → `*.weight_g`,
///    `*.parametrizations.weight.original1` → `*.weight_v` (the PyTorch
///    weight-norm reparametrization rename), each also `swapaxes(1, 2)`.
/// 4. **Drop** training-only keys: `quantizer.*`, `project_*`,
///    `masked_spec_embed`.
/// 5. **Keep** `lm_head.*` — the CTC head (the backbone's own `sanitize`
///    drops it; this composed model needs it).
///
/// [san]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L107-L128
/// [san-bb]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L723
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  let mut out = HashMap::with_capacity(weights.len());
  for (mut k, mut v) in weights {
    // 1. Backbone prefix: HF Wav2Vec2ForCTC nests the encoder under
    //    `wav2vec2.` while `lm_head` is top-level. `replacen(.., 1)` only
    //    touches the leading occurrence.
    if let Some(stripped) = k.strip_prefix("wav2vec2.") {
      k = stripped.to_string();
    }

    // 2 + 3. Conv weight axis swaps and weight-norm parametrization renames.
    if k.ends_with(".conv.weight") || k.ends_with(".conv.weight_v") || k.ends_with(".conv.weight_g")
    {
      v = ops::shape::swapaxes(&v, 1, 2)?;
    } else if let Some(base) = k.strip_suffix(".parametrizations.weight.original0") {
      k = format!("{base}.weight_g");
      v = ops::shape::swapaxes(&v, 1, 2)?;
    } else if let Some(base) = k.strip_suffix(".parametrizations.weight.original1") {
      k = format!("{base}.weight_v");
      v = ops::shape::swapaxes(&v, 1, 2)?;
    }

    // 4. Drop training-only keys. `lm_head.*` is deliberately NOT dropped
    //    here (unlike the backbone's own sanitize) — it is the CTC head.
    if k.starts_with("quantizer.") || k.starts_with("project_") || k == "masked_spec_embed" {
      continue;
    }

    out.insert(k, v);
  }
  Ok(out)
}

// ───────────────────────────── CTC decode ─────────────────────────────

/// CTC blank token id (`pad_token_id = 0` for `base-960h`). Greedy decoding
/// drops this token and collapses runs.
#[cfg(feature = "wav2vec2")]
const CTC_BLANK: u32 = 0;

/// Greedy CTC collapse of a single per-frame argmax sequence — pure Rust over
/// `&[u32]`, the inner loop of mlx-audio's `Model._ctc_decode`
/// ([mms.py:33-45][dec]).
///
/// Emits a token only when it differs from the immediately preceding frame's
/// token **and** is not the [blank](CTC_BLANK): `token != prev && token != 0`.
/// `prev` is updated on **every** frame (including blanks), so a blank breaks
/// a run — `[5, 5, 0, 5]` decodes to `[5, 5]` (the blank separates the two
/// runs of `5`), while `[5, 5, 5]` collapses to `[5]`.
///
/// [dec]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L33-L45
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub fn ctc_greedy_collapse(predictions: &[u32]) -> Vec<u32> {
  let mut tokens = Vec::new();
  // `prev` starts at a sentinel that cannot equal any vocab id, matching the
  // reference's `prev = -1` (so the first non-blank frame always emits).
  let mut prev: Option<u32> = None;
  for &token in predictions {
    if Some(token) != prev && token != CTC_BLANK {
      tokens.push(token);
    }
    prev = Some(token);
  }
  tokens
}

// ───────────────────────────── vocabulary ─────────────────────────────

/// The character-level CTC vocabulary — the inverse of HF `vocab.json`'s
/// `{token_string: id}` map, used to render decoded token ids back to text.
///
/// mlx-audio loads `vocab.json` and inverts it to `{id: token}`
/// ([mms.py:145-155][voc]); the `base-960h` vocabulary is 32 single-character
/// tokens (`<pad>=0, <s>, </s>, <unk>, |, A-Z, '`) — **not** BPE, so no
/// tokenizer crate is needed. `_tokens_to_text` joins the per-id strings and
/// maps the word-delimiter `|` to a space ([mms.py:47-52][voc-text]).
///
/// [voc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L145-L155
/// [voc-text]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L47-L52
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
#[derive(Debug, Clone, Default)]
pub struct Vocab {
  /// `id → token string`. A `Vec` indexed by id keeps the common path
  /// (`tokens_to_text`) allocation-free of a hash lookup; ids past the end
  /// (or with a `None` slot, if `vocab.json` were sparse) render as the empty
  /// string, matching the reference's `self._vocab.get(t, "")`.
  id_to_token: Vec<Option<String>>,
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Vocab {
  /// Parse a `vocab.json` body — the HF `{token: id}` object — and invert it
  /// to `id → token`.
  ///
  /// Mirrors mlx-audio's `model._vocab = {v: k for k, v in vocab.items()}`
  /// ([mms.py:155][voc]). The nested-`{lang: {...}}` MMS multilingual form is
  /// **not** handled (`base-960h` is monolingual); a nested object is a parse
  /// error. A negative id is rejected.
  ///
  /// [voc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L155
  pub fn from_json(json: &str) -> Result<Self> {
    let map: HashMap<String, i64> = serde_json::from_str(json)
      .map_err(|e| Error::Parse(ParsePayload::new("Vocab::from_json", "vocab.json", e)))?;
    let max_id = map.values().copied().max().unwrap_or(-1);
    if max_id < 0 {
      // Empty vocab (or all-negative, rejected next) → no slots.
      return Ok(Self {
        id_to_token: Vec::new(),
      });
    }
    let len = usize::try_from(max_id)
      .map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "Vocab::from_json: token id",
          "must be a non-negative i64 that fits usize",
          format_smolstr!("{max_id}"),
        ))
      })?
      .saturating_add(1);
    let mut id_to_token: Vec<Option<String>> = vec![None; len];
    for (token, id) in map {
      let idx = usize::try_from(id).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "Vocab::from_json: token id",
          "must be non-negative",
          format_smolstr!("{id}"),
        ))
      })?;
      id_to_token[idx] = Some(token);
    }
    Ok(Self { id_to_token })
  }

  /// Number of id slots (one past the maximum id).
  #[inline(always)]
  pub fn len(&self) -> usize {
    self.id_to_token.len()
  }

  /// Whether the vocabulary is empty.
  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.id_to_token.is_empty()
  }

  /// The token string for `id`, or `None` if out of range / absent.
  #[inline]
  pub fn token(&self, id: u32) -> Option<&str> {
    self
      .id_to_token
      .get(id as usize)
      .and_then(|slot| slot.as_deref())
  }

  /// Render a decoded token-id sequence to text — the reference's
  /// `"".join(self._vocab.get(t, "")).replace("|", " ")`
  /// ([mms.py:52][voc-text]). Unknown ids contribute nothing; the
  /// word-delimiter `|` becomes a space.
  ///
  /// [voc-text]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L52
  pub fn tokens_to_text(&self, tokens: &[u32]) -> String {
    let mut text = String::new();
    for &t in tokens {
      if let Some(tok) = self.token(t) {
        text.push_str(tok);
      }
    }
    text.replace('|', " ")
  }
}

// ───────────────────────────── linear helper ─────────────────────────────

/// `y = x @ wᵀ (+ bias)` — a private `nn.Linear` forward.
///
/// mlxrs ships no public `nn::Linear` (GAP by design); each transformer
/// projection composes this. HF stores a `Linear` weight as `(out, in)`, so
/// the matmul is `x @ wᵀ`. With a bias this is the fused
/// `addmm(bias, x, wᵀ, 1, 1)`; without, a plain `matmul(x, wᵀ)`.
#[cfg(feature = "wav2vec2")]
fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
  let wt = ops::shape::swapaxes(weight, 0, 1)?;
  match bias {
    Some(b) => ops::linalg_basic::addmm(b, x, &wt, 1.0, 1.0),
    None => ops::linalg_basic::matmul(x, &wt),
  }
}

// ───────────────────────── weight-fetch helper ─────────────────────────

/// Pull a weight by exact key from the (sanitized) checkpoint map, erroring
/// with the key if absent.
#[cfg(feature = "wav2vec2")]
fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights
    .remove(key)
    .ok_or_else(|| Error::MissingKey(MissingKeyPayload::new("Wav2Vec2Ctc::from_weights", key)))
}

// ───────────────────────── feature encoder ─────────────────────────

/// A single feature-encoder convolution layer. Conv weight is channels-last
/// `(out, k, in)` (post-sanitize); `base-960h` convs are bias-free. L0
/// additionally carries an affine [`GroupNorm`] (`num_groups == dims`,
/// pytorch-compatible); L1-6 are conv → GELU only.
#[cfg(feature = "wav2vec2")]
struct ConvLayer {
  weight: Array,
  stride: i32,
  /// `Some` for L0 (`Wav2Vec2GroupNormConvLayer`), `None` for L1-6
  /// (`Wav2Vec2NoLayerNormConvLayer`).
  group_norm: Option<GroupNorm>,
}

#[cfg(feature = "wav2vec2")]
impl ConvLayer {
  /// `conv(x.swapaxes(-2,-1)) → [group_norm] → swapaxes(-2,-1) → gelu`.
  ///
  /// Input/output are `(B, C, L)` (channels-second). MLX conv1d is
  /// channels-last, so the layer transposes to `(B, L, C)` around the conv
  /// (and around the GroupNorm, which normalizes over the last/feature axis),
  /// exactly as the reference's `hidden_states.swapaxes(-2, -1)` bracketing.
  fn forward(&self, x: &Array) -> Result<Array> {
    // (B, C_in, L) → (B, L, C_in)
    let xt = ops::shape::swapaxes(x, -2, -1)?;
    // channels-last conv → (B, L', C_out)
    let mut h = ops::conv::conv1d(&xt, &self.weight, self.stride, 0, 1, 1)?;
    // GroupNorm (L0 only) runs in channels-last layout (feature = last axis).
    if let Some(gn) = &self.group_norm {
      h = gn.forward(&h)?;
    }
    // back to (B, C_out, L')
    let h = ops::shape::swapaxes(&h, -2, -1)?;
    crate::lm::nn::activations::gelu(&h)
  }
}

/// The feature encoder: `input[:, None]` → 7 conv layers → `(B, 512, T')`.
/// Ports `Wav2Vec2FeatureEncoder` ([wav2vec.py:250-276][fe]).
///
/// [fe]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L250-L276
#[cfg(feature = "wav2vec2")]
struct FeatureEncoder {
  conv_layers: Vec<ConvLayer>,
}

#[cfg(feature = "wav2vec2")]
impl FeatureEncoder {
  fn forward(&self, input_values: &Array) -> Result<Array> {
    // input_values (B, T) → (B, 1, T): insert a channel axis at position 1.
    let mut h = ops::shape::expand_dims_axes(input_values, &[1])?;
    for layer in &self.conv_layers {
      h = layer.forward(&h)?;
    }
    Ok(h)
  }
}

// ───────────────────────── feature projection ─────────────────────────

/// `LayerNorm(512) → Linear(512 → 768)` over `(B, T', 512)`.
/// Ports `Wav2Vec2FeatureProjection` ([wav2vec.py:279-290][fp]).
///
/// [fp]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L279-L290
#[cfg(feature = "wav2vec2")]
struct FeatureProjection {
  layer_norm: LayerNorm,
  proj_weight: Array,
  proj_bias: Array,
}

#[cfg(feature = "wav2vec2")]
impl FeatureProjection {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let normed = self.layer_norm.forward(hidden_states)?;
    linear(&normed, &self.proj_weight, Some(&self.proj_bias))
  }
}

// ───────────────────────── positional conv embedding ─────────────────────────

/// The weight-normalized positional conv embedding
/// (`Wav2Vec2PositionalConvEmbedding`, [wav2vec.py:218-247][pos]).
///
/// HF stores the weight-normalized conv as a magnitude `weight_g` and a
/// direction `weight_v`; the effective kernel is
/// `weight = weight_g * weight_v / ‖weight_v‖`, the norm reduced over every
/// axis except the kernel axis (axis 1 in MLX's channels-last
/// `(out, k, in)` layout, because the reference's `swapaxes(1, 2)` made
/// `except_dim=1` the kernel axis — see [`reconstruct_wn_weight`]). The fused
/// kernel is **reconstructed once at load**, so the forward pass is a plain
/// grouped conv. SamePad then crops the single trailing frame produced by the
/// even (128) kernel, and GELU follows.
///
/// [pos]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L218-L247
#[cfg(feature = "wav2vec2")]
struct PositionalConvEmbedding {
  /// The reconstructed fused conv kernel `(out, k, in/groups)`.
  weight: Array,
  bias: Array,
  groups: i32,
  padding: i32,
  /// Frames to crop from the end (SamePad): `1` for an even kernel, else `0`.
  num_pad_remove: i32,
}

#[cfg(feature = "wav2vec2")]
impl PositionalConvEmbedding {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    // hidden_states is (B, T, C) (channels-last already); plain grouped conv.
    let mut h = ops::conv::conv1d(hidden_states, &self.weight, 1, self.padding, 1, self.groups)?;
    h = h.add(&self.bias)?;
    // SamePad: drop the trailing `num_pad_remove` frames along the time axis
    // (axis 1 in channels-last (B, T, C)). The reference slices
    // `hidden_states[:, :-num_pad_remove, :]`.
    if self.num_pad_remove > 0 {
      let shape = h.shape();
      let rank = shape.len();
      let mut start = vec![0i32; rank];
      let mut stop: Vec<i32> = shape
        .iter()
        .map(|&d| {
          i32::try_from(d).map_err(|_| {
            Error::OutOfRange(OutOfRangePayload::new(
              "PositionalConvEmbedding: dim",
              "exceeds i32::MAX",
              format_smolstr!("{d}"),
            ))
          })
        })
        .collect::<Result<Vec<_>>>()?;
      let strides = vec![1i32; rank];
      // Crop the time axis (axis 1).
      start[1] = 0;
      stop[1] -= self.num_pad_remove;
      h = ops::indexing::slice(&h, &start, &stop, &strides)?;
    }
    crate::lm::nn::activations::gelu(&h)
  }
}

/// Reconstruct a weight-normalized conv kernel from its `(weight_g, weight_v)`
/// reparametrization: `weight = weight_g * weight_v / ‖weight_v‖`, the L2 norm
/// of `weight_v` taken over every axis **except** the kernel axis (axis 1),
/// with `keepdims` so it broadcasts.
///
/// This matches mlx-audio's `WNConv1d.__call__`
/// (`weight_g * weight_v / normalize_weight(weight_v, except_dim=1)`,
/// [wav2vec.py:206-211][wn]) where `normalize_weight(x, except_dim=1)` reduces
/// `sqrt(sum(x**2))` over `axes = (0, 2)` (every axis but `except_dim`). In
/// MLX's channels-last `(out, k, in)` layout that `except_dim=1` is the kernel
/// axis (the reference applied `swapaxes(1, 2)` to the stored `(out, in, k)`
/// tensor at load, turning the original PyTorch `dim=0` output-channel
/// reduction into this `except_dim=1` form). Reconstructed **once** at load —
/// numerically identical to recomputing it per forward.
///
/// [wn]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L206-L211
#[cfg(feature = "wav2vec2")]
fn reconstruct_wn_weight(weight_g: &Array, weight_v: &Array) -> Result<Array> {
  if weight_v.ndim() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "reconstruct_wn_weight: weight_v must be rank-3 (out, k, in)",
      weight_v.ndim() as u32,
      weight_v.shape(),
    )));
  }
  // ‖weight_v‖ over axes (0, 2), keepdims → shape (1, k, 1).
  let sq = weight_v.square()?;
  let sum_sq = ops::reduction::sum_axes(&sq, &[0, 2], true)?;
  let norm_v = sum_sq.sqrt()?;
  // weight_g * weight_v / norm_v.
  let scaled = weight_g.multiply(weight_v)?;
  scaled.divide(&norm_v)
}

// ───────────────────────── attention ─────────────────────────

/// Multi-head self-attention (`Wav2Vec2Attention`, [wav2vec.py:293-393][att]).
///
/// `q/k/v/out` are biased `Linear(768, 768)`. The query is **pre-scaled** by
/// `head_dim**-0.5` and SDPA is then called with `scale = 1.0` (matching the
/// reference's `query_states = self.q_proj(h) * self.scaling` followed by
/// `scaled_dot_product_attention(..., scale=1.0)`). `base-960h` runs with no
/// attention mask.
///
/// [att]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L293-L393
#[cfg(feature = "wav2vec2")]
struct Attention {
  q_weight: Array,
  q_bias: Array,
  k_weight: Array,
  k_bias: Array,
  v_weight: Array,
  v_bias: Array,
  out_weight: Array,
  out_bias: Array,
  num_heads: i32,
  head_dim: i32,
  /// `head_dim**-0.5`, pre-multiplied into the query.
  scaling: f32,
}

#[cfg(feature = "wav2vec2")]
impl Attention {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let shape = hidden_states.shape();
    // (B, T, C)
    let bsz = dim_i32(&shape, 0, "Attention: batch")?;
    let tgt_len = dim_i32(&shape, 1, "Attention: seq")?;

    // q pre-scaled by head_dim**-0.5; SDPA scale is then 1.0.
    let q = linear(hidden_states, &self.q_weight, Some(&self.q_bias))?;
    let scale = scalar_f32(self.scaling)?;
    let q = q.multiply(&scale)?;
    let k = linear(hidden_states, &self.k_weight, Some(&self.k_bias))?;
    let v = linear(hidden_states, &self.v_weight, Some(&self.v_bias))?;

    // (B, T, C) → (B, n_heads, T, head_dim): reshape then transpose(0,2,1,3).
    let q = self.split_heads(&q, bsz, tgt_len)?;
    let k = self.split_heads(&k, bsz, tgt_len)?;
    let v = self.split_heads(&v, bsz, tgt_len)?;

    let attn = scaled_dot_product_attention(&q, &k, &v, 1.0, Mask::None)?;

    // (B, n_heads, T, head_dim) → (B, T, C).
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = self.num_heads.checked_mul(self.head_dim).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Attention: num_heads * head_dim",
        "overflows i32",
        "overflow",
      ))
    })?;
    let attn = ops::shape::reshape(&attn, &[bsz, tgt_len, embed_dim])?;
    linear(&attn, &self.out_weight, Some(&self.out_bias))
  }

  /// `(B, T, C) → (B, n_heads, T, head_dim)`.
  fn split_heads(&self, x: &Array, bsz: i32, seq: i32) -> Result<Array> {
    let reshaped = ops::shape::reshape(x, &[bsz, seq, self.num_heads, self.head_dim])?;
    ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
  }
}

// ───────────────────────── feed-forward ─────────────────────────

/// `Linear(768 → 3072) → GELU → Linear(3072 → 768)`
/// (`Wav2Vec2FeedForward`, [wav2vec.py:396-417][ff]).
///
/// [ff]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L396-L417
#[cfg(feature = "wav2vec2")]
struct FeedForward {
  intermediate_weight: Array,
  intermediate_bias: Array,
  output_weight: Array,
  output_bias: Array,
}

#[cfg(feature = "wav2vec2")]
impl FeedForward {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let h = linear(
      hidden_states,
      &self.intermediate_weight,
      Some(&self.intermediate_bias),
    )?;
    let h = crate::lm::nn::activations::gelu(&h)?;
    linear(&h, &self.output_weight, Some(&self.output_bias))
  }
}

// ───────────────────────── encoder layer ─────────────────────────

/// A post-norm transformer encoder layer (`Wav2Vec2EncoderLayer`,
/// [wav2vec.py:436-465][el]):
/// `h = h + attn(h); h = layer_norm(h); h = h + ff(h); h = final_layer_norm(h)`.
///
/// [el]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L436-L465
#[cfg(feature = "wav2vec2")]
struct EncoderLayer {
  attention: Attention,
  layer_norm: LayerNorm,
  feed_forward: FeedForward,
  final_layer_norm: LayerNorm,
}

#[cfg(feature = "wav2vec2")]
impl EncoderLayer {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let attn = self.attention.forward(hidden_states)?;
    let h = hidden_states.add(&attn)?;
    let h = self.layer_norm.forward(&h)?;
    let ff = self.feed_forward.forward(&h)?;
    let h = h.add(&ff)?;
    self.final_layer_norm.forward(&h)
  }
}

// ───────────────────────── encoder ─────────────────────────

/// The transformer encoder (`Wav2Vec2Encoder`, non-stable arm,
/// [wav2vec.py:511-574][enc]): positional conv embedding added to the hidden
/// states, a `LayerNorm`, then the stack of [`EncoderLayer`]s.
///
/// [enc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L511-L574
#[cfg(feature = "wav2vec2")]
struct Encoder {
  pos_conv_embed: PositionalConvEmbedding,
  layer_norm: LayerNorm,
  layers: Vec<EncoderLayer>,
}

#[cfg(feature = "wav2vec2")]
impl Encoder {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let pos = self.pos_conv_embed.forward(hidden_states)?;
    let mut h = hidden_states.add(&pos)?;
    h = self.layer_norm.forward(&h)?;
    for layer in &self.layers {
      h = layer.forward(&h)?;
    }
    Ok(h)
  }
}

// ───────────────────────── model ─────────────────────────

/// Wav2Vec2 CTC speech recognizer (`facebook/wav2vec2-base-960h`).
///
/// See the [module docs](self) for the architecture and public API.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub struct Wav2Vec2Ctc {
  config: Wav2Vec2Config,
  feature_encoder: FeatureEncoder,
  feature_projection: FeatureProjection,
  encoder: Encoder,
  /// CTC head `Linear(hidden → vocab)`.
  lm_head_weight: Array,
  lm_head_bias: Array,
  vocab: Vocab,
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Wav2Vec2Ctc {
  /// The fixed input sample rate (16 kHz mono) — mlx-audio's
  /// `Model.sample_rate`.
  pub const SAMPLE_RATE: u32 = 16_000;

  /// Build a model from a parsed [`Wav2Vec2Config`], the **sanitized** weight
  /// map (run [`sanitize`] first), and an optional [`Vocab`] (for
  /// [`Wav2Vec2Ctc::transcribe`]).
  ///
  /// Only the `feat_extract_norm == "group"` / `do_stable_layer_norm == false`
  /// arm (i.e. `base-960h`) is supported; any other config is rejected.
  /// Every weight key the architecture needs must be present, else
  /// [`Error::MissingKey`].
  pub fn from_weights(
    config: Wav2Vec2Config,
    mut weights: HashMap<String, Array>,
    vocab: Vocab,
  ) -> Result<Self> {
    if !config.is_group_norm() {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Wav2Vec2Ctc: feat_extract_norm",
        "only \"group\" is supported (base-960h)",
        format_smolstr!("{}", config.feat_extract_norm),
      )));
    }
    if config.do_stable_layer_norm {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Wav2Vec2Ctc: do_stable_layer_norm",
        "only false (post-norm) is supported",
        "true",
      )));
    }

    let feature_encoder = build_feature_encoder(&config, &mut weights)?;
    let feature_projection = build_feature_projection(&config, &mut weights)?;
    let encoder = build_encoder(&config, &mut weights)?;
    let lm_head_weight = take(&mut weights, "lm_head.weight")?;
    let lm_head_bias = take(&mut weights, "lm_head.bias")?;

    Ok(Self {
      config,
      feature_encoder,
      feature_projection,
      encoder,
      lm_head_weight,
      lm_head_bias,
      vocab,
    })
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &Wav2Vec2Config {
    &self.config
  }

  /// The decoding vocabulary.
  #[inline(always)]
  pub fn vocab(&self) -> &Vocab {
    &self.vocab
  }

  /// Run the full forward pass: raw waveform `(B, T)` (or `(T,)` — promoted to
  /// `(1, T)`) → per-frame CTC logits `(B, T', vocab)`.
  ///
  /// This does **not** normalize the waveform (the reference's
  /// zero-mean-unit-variance step happens in `generate`); call
  /// [`Wav2Vec2Ctc::transcribe`] for the full normalize → forward → decode
  /// path. Returns a lazy [`Array`] (no implicit eval).
  pub fn forward(&self, waveform: &Array) -> Result<Array> {
    let input_values = match waveform.ndim() {
      1 => ops::shape::expand_dims_axes(waveform, &[0])?,
      _ => waveform.try_clone()?,
    };
    // (B, 512, T')
    let extract_features = self.feature_encoder.forward(&input_values)?;
    // (B, T', 512)
    let extract_features = ops::shape::transpose_axes(&extract_features, &[0, 2, 1])?;
    // (B, T', 768)
    let hidden_states = self.feature_projection.forward(&extract_features)?;
    let hidden_states = self.encoder.forward(&hidden_states)?;
    // (B, T', vocab)
    linear(
      &hidden_states,
      &self.lm_head_weight,
      Some(&self.lm_head_bias),
    )
  }

  /// Transcribe a raw 16 kHz mono waveform to text.
  ///
  /// Mirrors mlx-audio's `Model.generate` for `base-960h`: zero-mean /
  /// unit-variance normalize the waveform (`(x - mean) / sqrt(var + 1e-7)`,
  /// HF's `zero_mean_unit_var_norm`), forward to logits, greedy CTC collapse,
  /// then map the token ids through the vocabulary. Requires the model to have
  /// been built with a non-empty [`Vocab`]; otherwise returns an error.
  ///
  /// `waveform` is `(T,)` or `(B, T)`; only the first batch element's
  /// transcript is returned (matching the reference's `decoded[0]`).
  pub fn transcribe(&self, waveform: &Array) -> Result<String> {
    if self.vocab.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Wav2Vec2Ctc::transcribe",
        "model was built without a vocabulary (use forward + ctc_greedy_collapse)",
      )));
    }
    let normed = normalize_waveform(waveform)?;
    let logits = self.forward(&normed)?;
    // argmax over the vocab axis → (B, T') u32 ids.
    let mut predictions = ops::misc::argmax(&logits, Some(-1), false)?;
    let shape = predictions.shape();
    let seq_len = match shape.as_slice() {
      [_, t] => *t,
      [t] => *t,
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "Wav2Vec2Ctc::transcribe: predictions must be rank-1 or rank-2",
          shape.len() as u32,
          shape.clone(),
        )));
      }
    };
    let all_ids = predictions.to_vec::<u32>()?;
    // First batch row (the reference decodes `decoded[0]`).
    let first_row = &all_ids[..seq_len.min(all_ids.len())];
    let tokens = ctc_greedy_collapse(first_row);
    Ok(self.vocab.tokens_to_text(&tokens).trim().to_string())
  }

  /// Load a model from a local on-disk directory — the convenience entry
  /// point mirroring mlx-audio's `stt.load` for this architecture.
  ///
  /// Resolves `path` via [`crate::audio::load::get_model_path`] (local-only;
  /// a Hub id is rejected per the project's no-network policy), reads and
  /// parses `config.json`, loads + [`sanitize`]s the single un-sharded
  /// `model.safetensors`, and reads the character `vocab.json` (so
  /// [`Wav2Vec2Ctc::transcribe`] works). `vocab.json` is optional — if absent
  /// the model still loads and [`Wav2Vec2Ctc::forward`] works, but
  /// `transcribe` then errors.
  ///
  /// Only the single-file `model.safetensors` layout (the `base-960h`
  /// checkpoint) is handled here; sharded checkpoints are not (a missing file
  /// is a clear [`Error::MissingKey`]).
  pub fn load(path: &str) -> Result<Self> {
    let dir = crate::audio::load::get_model_path(path)?;
    let config_json = crate::audio::load::load_config(&dir)?;
    let config = Wav2Vec2Config::from_json(&config_json)?;

    let weights_path = dir.join("model.safetensors");
    if !weights_path.is_file() {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "Wav2Vec2Ctc::load: model.safetensors not found (sharded checkpoints unsupported)",
        format_smolstr!("{}", weights_path.display()),
      )));
    }
    let raw = crate::io::load_safetensors(&weights_path)?;
    let weights = sanitize(raw)?;

    // vocab.json is optional; an absent file leaves an empty Vocab (forward
    // still works, transcribe then errors with a clear message). Reuse the
    // shared bounded reader so a hostile directory can't OOM the loader.
    let vocab_path = dir.join("vocab.json");
    let vocab = match crate::lm::load::read_bounded_config_file(&vocab_path, "wav2vec2 vocab.json")?
    {
      Some(body) => Vocab::from_json(&body)?,
      None => Vocab::default(),
    };

    Self::from_weights(config, weights, vocab)
  }
}

/// Zero-mean / unit-variance waveform normalization, HF's
/// `zero_mean_unit_var_norm`: `(x - mean(x, -1)) / sqrt(var(x, -1) + 1e-7)`
/// (the `sqrt(var + eps)` form, not mms.py's `std + eps`, per the `base-960h`
/// feature extractor). Operates per row over the last axis. Promotes a `(T,)`
/// input to `(1, T)` first.
#[cfg(feature = "wav2vec2")]
fn normalize_waveform(waveform: &Array) -> Result<Array> {
  let x = match waveform.ndim() {
    1 => ops::shape::expand_dims_axes(waveform, &[0])?,
    _ => waveform.try_clone()?,
  };
  let mean = ops::reduction::mean_axes(&x, &[-1], true)?;
  let var = ops::reduction::var_axes(&x, &[-1], true, 0)?;
  let eps = scalar_f32(1e-7)?;
  let denom = var.add(&eps)?.sqrt()?;
  let centered = x.subtract(&mean)?;
  centered.divide(&denom)
}

// ───────────────────────── builders ─────────────────────────

/// Read `feature_extractor.conv_layers.{i}.*` into the 7-layer
/// [`FeatureEncoder`].
#[cfg(feature = "wav2vec2")]
fn build_feature_encoder(
  config: &Wav2Vec2Config,
  weights: &mut HashMap<String, Array>,
) -> Result<FeatureEncoder> {
  let n = config.num_feat_extract_layers;
  if n <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Wav2Vec2Config: num_feat_extract_layers",
      "must be positive (> 0)",
      format_smolstr!("{n}"),
    )));
  }
  let n_usize = n as usize;
  if config.conv_dim.len() < n_usize
    || config.conv_stride.len() < n_usize
    || config.conv_kernel.len() < n_usize
  {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "Wav2Vec2Config: conv_dim/stride/kernel length",
      n_usize,
      config
        .conv_dim
        .len()
        .min(config.conv_stride.len())
        .min(config.conv_kernel.len()),
    )));
  }
  let mut conv_layers = Vec::with_capacity(n_usize);
  for i in 0..n_usize {
    let prefix = format!("feature_extractor.conv_layers.{i}");
    let weight = take(weights, &format!("{prefix}.conv.weight"))?;
    let stride = config.conv_stride[i];
    // L0 carries an affine pytorch-compatible GroupNorm with
    // num_groups == dims == conv_dim[0]; L1-6 have no norm.
    let group_norm = if i == 0 {
      let dims = config.conv_dim[0];
      let gn_weight = take(weights, &format!("{prefix}.layer_norm.weight"))?;
      let gn_bias = take(weights, &format!("{prefix}.layer_norm.bias"))?;
      Some(GroupNorm::with_affine(
        dims,
        dims,
        config.layer_norm_eps,
        Some((gn_weight, gn_bias)),
        true,
      )?)
    } else {
      None
    };
    conv_layers.push(ConvLayer {
      weight,
      stride,
      group_norm,
    });
  }
  Ok(FeatureEncoder { conv_layers })
}

/// Read `feature_projection.*` into the [`FeatureProjection`].
#[cfg(feature = "wav2vec2")]
fn build_feature_projection(
  config: &Wav2Vec2Config,
  weights: &mut HashMap<String, Array>,
) -> Result<FeatureProjection> {
  let ln_weight = take(weights, "feature_projection.layer_norm.weight")?;
  let ln_bias = take(weights, "feature_projection.layer_norm.bias")?;
  let layer_norm = LayerNorm::new(Some(ln_weight), Some(ln_bias), config.layer_norm_eps);
  let proj_weight = take(weights, "feature_projection.projection.weight")?;
  let proj_bias = take(weights, "feature_projection.projection.bias")?;
  Ok(FeatureProjection {
    layer_norm,
    proj_weight,
    proj_bias,
  })
}

/// Read `encoder.*` into the [`Encoder`].
#[cfg(feature = "wav2vec2")]
fn build_encoder(config: &Wav2Vec2Config, weights: &mut HashMap<String, Array>) -> Result<Encoder> {
  // Positional conv embedding: reconstruct the fused weight-normalized kernel
  // once, then a plain grouped conv at forward time.
  let weight_g = take(weights, "encoder.pos_conv_embed.conv.weight_g")?;
  let weight_v = take(weights, "encoder.pos_conv_embed.conv.weight_v")?;
  let pos_weight = reconstruct_wn_weight(&weight_g, &weight_v)?;
  let pos_bias = take(weights, "encoder.pos_conv_embed.conv.bias")?;
  let kernel = config.num_conv_pos_embeddings;
  let pos_conv_embed = PositionalConvEmbedding {
    weight: pos_weight,
    bias: pos_bias,
    groups: config.num_conv_pos_embedding_groups,
    padding: kernel / 2,
    num_pad_remove: if kernel % 2 == 0 { 1 } else { 0 },
  };

  let ln_weight = take(weights, "encoder.layer_norm.weight")?;
  let ln_bias = take(weights, "encoder.layer_norm.bias")?;
  let layer_norm = LayerNorm::new(Some(ln_weight), Some(ln_bias), config.layer_norm_eps);

  let num_layers = config.num_hidden_layers;
  if num_layers < 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Wav2Vec2Config: num_hidden_layers",
      "must be non-negative",
      format_smolstr!("{num_layers}"),
    )));
  }
  let head_dim = config.head_dim()?;
  let scaling = (head_dim as f32).powf(-0.5);
  let mut layers = Vec::with_capacity(num_layers as usize);
  for i in 0..num_layers {
    let prefix = format!("encoder.layers.{i}");
    let attention = Attention {
      q_weight: take(weights, &format!("{prefix}.attention.q_proj.weight"))?,
      q_bias: take(weights, &format!("{prefix}.attention.q_proj.bias"))?,
      k_weight: take(weights, &format!("{prefix}.attention.k_proj.weight"))?,
      k_bias: take(weights, &format!("{prefix}.attention.k_proj.bias"))?,
      v_weight: take(weights, &format!("{prefix}.attention.v_proj.weight"))?,
      v_bias: take(weights, &format!("{prefix}.attention.v_proj.bias"))?,
      out_weight: take(weights, &format!("{prefix}.attention.out_proj.weight"))?,
      out_bias: take(weights, &format!("{prefix}.attention.out_proj.bias"))?,
      num_heads: config.num_attention_heads,
      head_dim,
      scaling,
    };
    let el_ln_weight = take(weights, &format!("{prefix}.layer_norm.weight"))?;
    let el_ln_bias = take(weights, &format!("{prefix}.layer_norm.bias"))?;
    let layer_norm_l = LayerNorm::new(Some(el_ln_weight), Some(el_ln_bias), config.layer_norm_eps);
    let feed_forward = FeedForward {
      intermediate_weight: take(
        weights,
        &format!("{prefix}.feed_forward.intermediate_dense.weight"),
      )?,
      intermediate_bias: take(
        weights,
        &format!("{prefix}.feed_forward.intermediate_dense.bias"),
      )?,
      output_weight: take(
        weights,
        &format!("{prefix}.feed_forward.output_dense.weight"),
      )?,
      output_bias: take(weights, &format!("{prefix}.feed_forward.output_dense.bias"))?,
    };
    let fln_weight = take(weights, &format!("{prefix}.final_layer_norm.weight"))?;
    let fln_bias = take(weights, &format!("{prefix}.final_layer_norm.bias"))?;
    let final_layer_norm = LayerNorm::new(Some(fln_weight), Some(fln_bias), config.layer_norm_eps);
    layers.push(EncoderLayer {
      attention,
      layer_norm: layer_norm_l,
      feed_forward,
      final_layer_norm,
    });
  }

  Ok(Encoder {
    pos_conv_embed,
    layer_norm,
    layers,
  })
}

// ───────────────────────── small helpers ─────────────────────────

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
#[cfg(feature = "wav2vec2")]
fn dim_i32(shape: &[usize], axis: usize, context: &'static str) -> Result<i32> {
  let d = *shape.get(axis).ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      context,
      "dim exceeds i32::MAX",
      format_smolstr!("{d}"),
    ))
  })
}

/// Build a rank-0 `f32` scalar Array of `value` (for broadcasting against a
/// lazy `Array`). Rank-0 so it NumPy-broadcasts against any rank without
/// lifting the operand.
#[cfg(feature = "wav2vec2")]
fn scalar_f32(value: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], value)
}

#[cfg(all(test, feature = "wav2vec2"))]
mod tests;

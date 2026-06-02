//! EmbeddingGemma configuration.
//!
//! Ports the `ModelArgs` dataclass of
//! [`mlx-lm`'s `models/gemma3_text.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py)
//! — the Gemma3 text backbone EmbeddingGemma reuses
//! (`mlx-embeddings`'s `models/gemma3_text.py` imports `ModelArgs` /
//! `TransformerBlock` / `RMSNorm` from there) — with defaults pinned to the
//! `google/embeddinggemma-300m` checkpoint.
//!
//! As elsewhere in the crate, parsing is forward-compatible: an unmodeled key
//! parses cleanly and an absent key falls back to its default
//! (`#[serde(default)]`, not `deny_unknown_fields`) — matching
//! `ModelArgs.from_dict`, which filters to the known signature parameters.
//!
//! [`Gemma3Config::validate`] pins every architecture-defining field (and
//! bounds every count / dimension) onto the shared [`crate::model_validation`]
//! toolkit **before** any tensor is allocated, so a corrupt / hostile /
//! wrong-architecture `config.json` fails fast with a typed [`crate::Error`]
//! instead of building the wrong graph or driving an oversized allocation —
//! the discipline of the merged SigLIP2 / Wav2Vec2 / LFM2 config validators.
//! A **quantized** checkpoint is supported: a `quantization` (or HF-style
//! `quantization_config`) block declares the per-layer scheme parameters, and
//! the loader builds each `nn.Linear` / `nn.Embedding` quantized via the shared
//! [`crate::nn::MaybeQuantizedLinear`] when the checkpoint carries the layer's
//! `.scales` sibling (the same auto-detect Whisper uses); the block is parsed
//! into a [`crate::lm::quant::PerLayerQuantization`] by
//! [`Gemma3Config::quantization`]. The bounds are presence / positivity /
//! structural / overflow (`checked_mul`) checks that surface a clear typed error
//! on a broken model file; there is no magnitude DoS ceiling (a library
//! faithfully loads its checkpoint — the consuming application owns input
//! bounding).

#[cfg(feature = "embeddinggemma")]
use crate::{
  error::{Error, ParsePayload, Result},
  model_validation::{
    require_cardinality, require_divisible, require_in_range, require_positive_finite_f32,
  },
};

/// Upper bound on a layer / head cardinality. A Gemma3 backbone has 24-48
/// layers; `4096` is far above any legitimate depth while still bounding the
/// per-layer allocation loop a hostile `num_hidden_layers` could otherwise
/// drive. Matches the SigLIP2 / LFM2 config `MAX_CARDINALITY` intent.
#[cfg(feature = "embeddinggemma")]
pub(crate) const MAX_CARDINALITY: u64 = 4096;

/// Inclusive upper bound on every *width*-like config field — `vocab_size`
/// (a few hundred K for Gemma's 262 144 token vocab), `hidden_size`,
/// `intermediate_size`, `head_dim`. Unlike a cardinality (which sizes an eager
/// per-layer `Vec`), a width names a matmul axis / embedding-table column, so
/// the `4096` cardinality cap is too tight. `1 << 20` (`1048576`) bounds every
/// width — the real Gemma vocab is 262 144 and the hidden a few thousand, all
/// far below — while keeping a malformed width a recoverable
/// [`Error::OutOfRange`] instead of an oversized allocation. Mirrors the LFM2 /
/// SigLIP2 config `MAX_CONFIG_DIM` width-cap discipline.
#[cfg(feature = "embeddinggemma")]
pub(crate) const MAX_CONFIG_DIM: i32 = 1 << 20;

/// EmbeddingGemma's Gemma3 text-backbone configuration. Defaults match
/// `google/embeddinggemma-300m`'s `config.json` (a Gemma3 270M-class text
/// transformer driven as a bidirectional encoder).
///
/// Ports `gemma3_text.py`'s `ModelArgs` (the subset EmbeddingGemma's forward
/// reads). The generative-only fields (`rope_scaling`,
/// `max_position_embeddings` as a decode bound) are parsed for parity but the
/// encoder does not use a KV cache.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3Config {
  /// Architecture id (`"gemma3_text"`).
  #[serde(default = "default_model_type")]
  model_type: String,
  /// Token-embedding table size / the vocabulary (`262144`).
  #[serde(default = "default_vocab_size")]
  pub vocab_size: i32,
  /// Transformer hidden / embedding dimension (`768`).
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Number of transformer layers (`24`).
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// Feed-forward intermediate dimension (`1152`).
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// Number of attention (query) heads (`3`).
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Per-head dimension (`256`). Gemma3 sets `head_dim` independently of
  /// `hidden_size / num_attention_heads` (here `768 / 3 = 256`, but the field
  /// is authoritative).
  #[serde(default = "default_head_dim")]
  pub head_dim: i32,
  /// `eps` shared by every `RMSNorm` (`1e-6`).
  #[serde(default = "default_rms_norm_eps")]
  pub rms_norm_eps: f64,
  /// Number of key/value heads for grouped-query attention (`1`).
  #[serde(default = "default_num_key_value_heads")]
  pub num_key_value_heads: i32,
  /// RoPE base (theta) for the **global** (full-attention) layers
  /// (`1_000_000`).
  #[serde(default = "default_rope_theta")]
  pub rope_theta: f64,
  /// RoPE base for the **sliding-window** (local) layers (`10_000`).
  #[serde(default = "default_rope_local_base_freq")]
  pub rope_local_base_freq: f64,
  /// The query pre-attention scalar — the SDPA scale is `scalar ** -0.5`
  /// (`256`, so `scale = 1/16`). Gemma3 decouples this from `head_dim`.
  #[serde(default = "default_query_pre_attn_scalar")]
  pub query_pre_attn_scalar: f64,
  /// Sliding-window size for the local-attention layers (`512`). Retained for
  /// parity; EmbeddingGemma runs the backbone as a **bidirectional** encoder
  /// (full attention every layer, with the padding mask), so the window is not
  /// applied (see the module docs).
  #[serde(default = "default_sliding_window")]
  pub sliding_window: i32,
  /// Layers `i` with `(i + 1) % sliding_window_pattern == 0` are **global**
  /// (full-attention, `rope_theta`); the rest are sliding-window (local,
  /// `rope_local_base_freq`). The pattern only selects which RoPE base a layer
  /// uses here — the attention is bidirectional throughout (`6`).
  #[serde(default = "default_sliding_window_pattern")]
  pub sliding_window_pattern: i32,
  /// Maximum position embeddings (`2048`) — the trained context window.
  /// Retained for parity; the encoder imposes no fixed sequence length.
  #[serde(default = "default_max_position_embeddings")]
  pub max_position_embeddings: i32,
  /// The weight-quantization block (`config.json` `quantization`, or its
  /// HF-style `quantization_config` alias), if the checkpoint declares one.
  /// A quantized EmbeddingGemma bundle (e.g.
  /// `mlx-community/embeddinggemma-300m-8bit`) carries a `quantization` block
  /// and packed `.weight` / `.scales` / `.biases` weight triples; the loader
  /// resolves the per-layer scheme parameters from this block (via
  /// [`quantization`](Self::quantization)) and builds each `nn.Linear` /
  /// `nn.Embedding` quantized through the shared [`crate::nn::MaybeQuantizedLinear`]
  /// when the checkpoint carries the layer's `.scales` sibling. The raw value
  /// is retained verbatim (not interpreted here) so the loader can deserialize
  /// it into a [`crate::lm::quant::PerLayerQuantization`].
  #[serde(default, alias = "quantization_config")]
  quantization: Option<serde_json::Value>,
}

// ── defaults (single source of truth; the `*_constants_match_defaults`
//    test pins these against the named architecture constants) ──

#[cfg(feature = "embeddinggemma")]
fn default_model_type() -> String {
  "gemma3_text".to_string()
}
#[cfg(feature = "embeddinggemma")]
fn default_vocab_size() -> i32 {
  262144
}
#[cfg(feature = "embeddinggemma")]
fn default_hidden_size() -> i32 {
  768
}
#[cfg(feature = "embeddinggemma")]
fn default_num_hidden_layers() -> i32 {
  24
}
#[cfg(feature = "embeddinggemma")]
fn default_intermediate_size() -> i32 {
  1152
}
#[cfg(feature = "embeddinggemma")]
fn default_num_attention_heads() -> i32 {
  3
}
#[cfg(feature = "embeddinggemma")]
fn default_head_dim() -> i32 {
  256
}
#[cfg(feature = "embeddinggemma")]
fn default_rms_norm_eps() -> f64 {
  1e-6
}
#[cfg(feature = "embeddinggemma")]
fn default_num_key_value_heads() -> i32 {
  1
}
#[cfg(feature = "embeddinggemma")]
fn default_rope_theta() -> f64 {
  1_000_000.0
}
#[cfg(feature = "embeddinggemma")]
fn default_rope_local_base_freq() -> f64 {
  10_000.0
}
#[cfg(feature = "embeddinggemma")]
fn default_query_pre_attn_scalar() -> f64 {
  256.0
}
#[cfg(feature = "embeddinggemma")]
fn default_sliding_window() -> i32 {
  512
}
#[cfg(feature = "embeddinggemma")]
fn default_sliding_window_pattern() -> i32 {
  6
}
#[cfg(feature = "embeddinggemma")]
fn default_max_position_embeddings() -> i32 {
  2048
}

#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
impl Gemma3Config {
  /// Parse a [`Gemma3Config`] from an in-memory `config.json` string. A
  /// malformed-JSON failure maps to [`Error::Parse`]; absent keys take their
  /// checkpoint defaults; unmodeled keys are ignored.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Gemma3Config::from_json",
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

  /// The parsed per-layer quantization config, or `None` for a dense checkpoint
  /// (no — or a `null` — `quantization` block).
  ///
  /// Deserializes the retained `quantization` (or `quantization_config`) block
  /// into a [`crate::lm::quant::PerLayerQuantization`] — the swift-faithful
  /// schema the loader keys on to resolve each layer's `(group_size, bits,
  /// mode)`. The loader threads the result to the backbone + Dense head, which
  /// build a quantized layer (via [`crate::nn::MaybeQuantizedLinear`]) whenever
  /// the checkpoint carries that layer's `.scales` sibling (the same per-layer
  /// auto-detect Whisper uses).
  ///
  /// # Errors
  /// [`Error::Parse`] if a present, non-null block is not a valid
  /// `PerLayerQuantization` (e.g. it is missing the required `group_size` /
  /// `bits`, or a per-layer value is neither `false` nor a quantization object).
  pub fn quantization(&self) -> Result<Option<crate::lm::quant::PerLayerQuantization>> {
    match self.quantization.as_ref() {
      // A present-but-`null` block carries no quantization (the dense path).
      None => Ok(None),
      Some(v) if v.is_null() => Ok(None),
      Some(v) => {
        let plq = serde_json::from_value::<crate::lm::quant::PerLayerQuantization>(v.clone())
          .map_err(|e| {
            Error::Parse(ParsePayload::new(
              "Gemma3Config::quantization",
              "`quantization` block",
              e,
            ))
          })?;
        Ok(Some(plq))
      }
    }
  }

  /// Whether layer `i` is a **global** (full-attention) layer — the Gemma3
  /// `is_global = i % sliding_window_pattern == sliding_window_pattern - 1`
  /// (equivalently `(i + 1) % pattern == 0`). A global layer uses
  /// [`rope_theta`](Self::rope_theta); a local layer uses
  /// [`rope_local_base_freq`](Self::rope_local_base_freq). `pattern` is
  /// validated `>= 1` so the modulo is well-defined.
  #[inline]
  pub fn is_global_layer(&self, i: i32) -> bool {
    i % self.sliding_window_pattern == self.sliding_window_pattern - 1
  }

  /// Reject a structurally invalid config with a typed error before any tensor
  /// is built.
  ///
  /// Pins `model_type` to `"gemma3_text"`; requires every dimension / count
  /// positive; bounds the layer + head counts and `sliding_window_pattern` by
  /// `MAX_CARDINALITY`; bounds every width-like field (`vocab_size`,
  /// `hidden_size`, `intermediate_size`, `head_dim`) by `MAX_CONFIG_DIM` so a
  /// hostile width cannot drive an oversized embedding-table / matmul-axis
  /// allocation; requires `num_attention_heads` divisible by
  /// `num_key_value_heads` (the grouped-query split); and requires the numeric
  /// `query_pre_attn_scalar`, `rms_norm_eps`, `rope_theta`, and
  /// `rope_local_base_freq` — each parsed as `f64` but **narrowed to `f32`** by
  /// the backbone (the SDPA scale, the RMSNorm eps, the RoPE bases) — to be
  /// finite and strictly positive *after that narrowing*, so a corrupt float (or
  /// one that overflows / underflows `f32`) cannot install an infinite / zero
  /// scale or produce `NaN` / `Inf` frequencies at inference.
  ///
  /// A `quantization` (or `quantization_config`) block is **not** rejected:
  /// quantized EmbeddingGemma checkpoints are supported — the loader resolves
  /// the per-layer scheme parameters from the block (via
  /// [`quantization`](Self::quantization)) and builds each layer quantized when
  /// its `.scales` sibling is present.
  pub fn validate(&self) -> Result<()> {
    crate::model_validation::pin_str(
      "Gemma3Config: model_type",
      self.model_type.as_str(),
      &["gemma3_text"],
    )?;
    // Width-like fields: positive AND within the width cap. `require_in_range(_,
    // 1, MAX_CONFIG_DIM)` rejects both a non-positive and an oversized value as
    // one [`Error::OutOfRange`].
    for (name, value) in [
      ("Gemma3Config: vocab_size", self.vocab_size),
      ("Gemma3Config: hidden_size", self.hidden_size),
      ("Gemma3Config: intermediate_size", self.intermediate_size),
      ("Gemma3Config: head_dim", self.head_dim),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    require_cardinality(
      "Gemma3Config: num_attention_heads",
      i64::from(self.num_attention_heads),
      MAX_CARDINALITY,
    )?;
    require_cardinality(
      "Gemma3Config: num_key_value_heads",
      i64::from(self.num_key_value_heads),
      MAX_CARDINALITY,
    )?;
    require_cardinality(
      "Gemma3Config: num_hidden_layers",
      i64::from(self.num_hidden_layers),
      MAX_CARDINALITY,
    )?;
    // `sliding_window_pattern` is the modulo divisor in `is_global_layer`; a
    // non-positive value would make `i % pattern` ill-defined (and a `0` would
    // divide-by-zero). Require it `>= 1` and bounded.
    require_cardinality(
      "Gemma3Config: sliding_window_pattern",
      i64::from(self.sliding_window_pattern),
      MAX_CARDINALITY,
    )?;
    // Grouped-query attention requires the query-head count divisible by the
    // kv-head count (the SDPA kernel repeats each kv head `n_heads / n_kv_heads`
    // times).
    require_divisible(
      "Gemma3Config: num_attention_heads",
      self.num_attention_heads,
      "num_key_value_heads",
      self.num_key_value_heads,
    )?;
    // `query_pre_attn_scalar` (the SDPA scale base, `scalar ** -0.5`), the
    // RMSNorm `eps`, and the two RoPE bases each parse as `f64` but the backbone
    // **narrows them to `f32`** (the SDPA scale, the RMSNorm eps, the RoPE
    // bases). A value that is finite-and-positive in `f64` can still be invalid
    // once narrowed — a huge magnitude overflows to `f32::INFINITY`, a tiny
    // positive underflows to `0.0` — which would install an infinite / zero
    // scale or an invalid RoPE base. Validate the EXACT `f32` value that will be
    // used, so a corrupt config fails fast with a typed error (a non-finite
    // narrowing, incl. `NaN`, is rejected too) instead of poisoning inference.
    require_positive_finite_f32(
      "Gemma3Config: query_pre_attn_scalar",
      self.query_pre_attn_scalar,
    )?;
    require_positive_finite_f32("Gemma3Config: rms_norm_eps", self.rms_norm_eps)?;
    require_positive_finite_f32("Gemma3Config: rope_theta", self.rope_theta)?;
    require_positive_finite_f32(
      "Gemma3Config: rope_local_base_freq",
      self.rope_local_base_freq,
    )?;
    Ok(())
  }
}

/// The `2_Dense` / `3_Dense` projection-head configuration EmbeddingGemma
/// applies **after** mean pooling (the SentenceTransformers `Dense` modules in
/// the checkpoint's `2_Dense` / `3_Dense` folders).
///
/// `mlx-embeddings`'s `Model.__init__` hard-codes the pair
/// `[Linear(hidden, hidden*4), Linear(hidden*4, hidden)]` (bias-free), so the
/// projection is a fixed `768 → 3072 → 768`. Matryoshka output truncation is a
/// *downstream* slice of the final 768-d vector, not a separate head, and is
/// surfaced through the baked [`crate::embeddings::PoolingConfig::dimension`]
/// (the `1_Pooling` / ST `word_embedding_dimension`), not here.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseConfig {
  /// The pooled-embedding (and final-output) width — `hidden_size` (`768`).
  pub hidden_size: i32,
  /// The intermediate width of the two-layer projection — `hidden_size * 4`
  /// (`3072`).
  pub intermediate_size: i32,
}

#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
impl DenseConfig {
  /// Derive the Dense-head dims from the backbone `hidden_size`, mirroring
  /// `mlx-embeddings`'s hard-coded `[Linear(hidden, hidden*4), Linear(hidden*4,
  /// hidden)]`. The `hidden * 4` product is overflow-checked so a hostile
  /// (already width-capped) `hidden_size` cannot wrap.
  pub fn from_hidden(hidden_size: i32) -> Result<Self> {
    let intermediate_size = crate::model_validation::checked_mul(
      "DenseConfig: hidden_size * 4",
      "hidden_size",
      hidden_size,
      "four",
      4,
    )?;
    Ok(Self {
      hidden_size,
      intermediate_size,
    })
  }
}

#[cfg(all(test, feature = "embeddinggemma"))]
mod tests;

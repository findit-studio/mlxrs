//! EmbeddingGemma — a Gemma3-backbone sentence-encoder
//! (`google/embeddinggemma-300m`).
//!
//! Ported from
//! [`mlx-embeddings`'s `models/gemma3_text.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/models/gemma3_text.py),
//! which reuses the Gemma3 text backbone (`ModelArgs` / `RMSNorm` /
//! `TransformerBlock`) from
//! [`mlx-lm`'s `models/gemma3_text.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py).
//!
//! This is a self-contained sentence-encoder under [`crate::embeddings`]. Unlike
//! the dual-tower SigLIP2 (a [`Contrastive`](crate::embeddings::Contrastive)
//! model with a fixed sticky-EOS text tower), EmbeddingGemma is a single-vector
//! text encoder: it produces per-token hidden states, **mean-pools** them over
//! the attention mask, applies a learned two-layer **Dense** projection, and
//! L2-normalizes.
//!
//! ## Architecture
//!
//! - **Backbone** (`backbone::Gemma3Backbone`): the Gemma3 text transformer
//!   driven as a **bidirectional encoder** — every layer attends fully over the
//!   real (non-padding) tokens via an additive padding mask, *not* causally.
//!   Grouped-query attention with per-head **query/key RMSNorm**, RoPE (the base
//!   alternates per the sliding-window pattern: global layers use `rope_theta`,
//!   local layers use `rope_local_base_freq`), the Gemma gated `gelu_approx`
//!   MLP, and the four-`RMSNorm` sandwich block. The token embedding is scaled
//!   by `sqrt(hidden_size)`. (The sliding window itself is *not* applied — the
//!   pattern only selects the RoPE base; the attention is full bidirectional
//!   throughout, matching the mlx-embeddings encoder.)
//! - **Pooling** ([`mean_pooling`](crate::embeddings::mean_pooling())): mask-aware
//!   mean over the sequence.
//! - **Dense head** (`dense::DenseHead`): two bias-free linear layers
//!   `[hidden → hidden*4, hidden*4 → hidden]` (the `2_Dense` / `3_Dense`
//!   SentenceTransformers modules), identity activation between them, applied to
//!   the pooled vector *before* normalization.
//! - **Normalize**: L2 over the final axis (python `normalize_embeddings`).
//!
//! ## Prompt format
//!
//! EmbeddingGemma is a task-prompted encoder: callers prepend a task prompt
//! (e.g. `"task: search result | query: "` for retrieval, `"title: none | text:
//! "` for documents) to each input *before* encoding. That prompting is the
//! consuming application's concern — this port faithfully tokenizes whatever
//! text it is handed (with BOS, no truncation cap by default) and pools/encodes
//! it; it imposes no prompt template and no input-size bound (the library
//! contract).
//!
//! ## Tokenization + padding ([`TextEmbedder`])
//!
//! The model owns its [`text_encoding`](TextEmbedder::text_encoding): the Gemma
//! tokenizer (`tokenizer.json`) with special tokens, right-padded to the batch
//! maximum with the `<pad>` id and a `0/1` attention mask
//! ([`Padding::DynamicRightPad`]) — the sentence-encoder default, faithful to
//! EmbeddingGemma's `padding_side="right"` + mask-aware mean pooling. No fixed
//! sequence length, no per-text truncation cap (the consuming application bounds
//! oversized prompts).
//!
//! ## Public surface ([`EmbeddingGemmaModel`])
//!
//! Built via [`EmbeddingGemmaModel::from_weights`] (sanitize + shape-pinned
//! load). It implements the golden [`TextEmbedder`] (owning its tokenization +
//! its forward → pool → dense → normalize) and answers the load factory's
//! [`EmbeddingModel`] umbrella (text accessor), so it registers into the
//! [`EmbeddingModelTypeRegistry`] via [`register`].

#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
pub mod backbone;

#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
pub mod config;

#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
pub mod dense;

#[cfg(feature = "embeddinggemma")]
mod shared;

#[cfg(feature = "embeddinggemma")]
use std::collections::HashMap;

#[cfg(feature = "embeddinggemma")]
use crate::{
  array::Array,
  embeddings::{
    Embedding, EmbeddingModel, EmbeddingModelConstructor, EmbeddingModelTypeRegistry,
    LoadedEmbeddingModel, Padding, PoolingConfig, PoolingStrategy, StPoolingConfig, TextEmbedder,
    TextEncoding,
    embeddinggemma::{
      backbone::{Gemma3Backbone, build_additive_mask},
      config::{DenseConfig, Gemma3Config},
      dense::DenseHead,
    },
    mean_pooling, pool_post,
  },
  error::Result,
  model_validation::reserve_or_error,
};

/// The top-level architecture id this model registers under (the `config.json`
/// `model_type` of a `google/embeddinggemma-300m` checkpoint).
#[cfg(feature = "embeddinggemma")]
pub const MODEL_TYPE: &str = "gemma3_text";

/// EmbeddingGemma sentence-encoder (`google/embeddinggemma-300m`).
///
/// See the [module docs](self) for the architecture and public API. Built via
/// [`from_weights`](Self::from_weights); encode a batch through the generic
/// [`crate::embeddings::encode()`] pipeline (which reads this model's
/// [`TextEmbedder`]) or call [`embed_text`](TextEmbedder::embed_text) directly.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
#[derive(Debug)]
pub struct EmbeddingGemmaModel {
  config: Gemma3Config,
  backbone: Gemma3Backbone,
  dense: DenseHead,
  /// The baked deployment pooling configuration (from `1_Pooling/config.json`):
  /// the matryoshka output dimension and the normalize flag the encode path
  /// applies after the Dense projection. The strategy is always
  /// [`PoolingStrategy::Mean`] for EmbeddingGemma (the reference mean-pools); a
  /// non-mean ST config is rejected at construction.
  pooling: PoolingConfig,
  /// The `<pad>` token id the dynamic right-padding writes into pad cells (the
  /// mask is `0` there, so the value never reaches the embedding). Gemma's
  /// `<pad>` is id `0`.
  pad_token_id: u32,
}

/// Gemma's `<pad>` token id — the id the HF `tokenizer.json` assigns to `<pad>`
/// (`0`). Written into the right-pad cells; masked out by the attention mask, so
/// it never contributes to the pooled embedding.
#[cfg(feature = "embeddinggemma")]
const GEMMA_PAD_TOKEN_ID: u32 = 0;

#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
impl EmbeddingGemmaModel {
  /// Build a model from a parsed [`Gemma3Config`], the **sanitized** weight map
  /// (run [`sanitize`] first), and the optional parsed `1_Pooling` config.
  ///
  /// The Dense-head dims are derived from `config.hidden_size` (the reference's
  /// hard-coded `hidden → hidden*4 → hidden`). The pooling config is baked from
  /// `pooling` (mean strategy, the matryoshka `dimension`, normalize) — a
  /// non-mean ST `pooling_mode` is rejected (EmbeddingGemma mean-pools). When
  /// `pooling` is absent the deployment default (mean + normalize + full
  /// dimension) is used.
  ///
  /// A **quantized** checkpoint loads through the same path: each `nn.Linear` /
  /// the token embedding is built quantized via the shared
  /// [`crate::nn::MaybeQuantizedLinear`] when the checkpoint carries that layer's
  /// `.scales` sibling (the same per-layer auto-detect Whisper uses), with the
  /// per-layer scheme parameters resolved from the config's `quantization` block
  /// (via [`Gemma3Config::quantization`]). The scheme is resolved ONLY when a
  /// relevant `.scales` is present (a one-pass pre-scan over the weight keys, the
  /// load-time half of that same discriminator), so a **dense** checkpoint (no
  /// `.scales`) loads dense regardless of any stale / foreign / partial
  /// `quantization` block the config may still carry — the scheme is only needed
  /// to interpret a scale that is actually present. The non-quant
  /// [`Gemma3Config::validate`] always runs first.
  ///
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions (typed [`crate::Error::ShapePairMismatch`] wrapped in
  /// [`crate::Error::LayerKeyed`]), exactly like the merged SigLIP2 / Wav2Vec2 /
  /// LFM2 ports; the quantized path pins the packed triple's logical shape
  /// identically.
  pub fn from_weights(
    config: Gemma3Config,
    mut weights: HashMap<String, Array>,
    pooling: Option<&StPoolingConfig>,
  ) -> Result<Self> {
    config.validate()?;
    // Resolve the per-layer quantization scheme ONLY when the checkpoint actually
    // carries a `.scales` sibling for some layer the model loads (the
    // `.scales`-presence discriminator the per-layer `Linear` / `Embedding`
    // loaders use, hoisted to the whole map). When no consumed layer is
    // quantized the scheme is irrelevant, so a DENSE checkpoint loads through the
    // unchanged dense path regardless of any stale / foreign / partial
    // `quantization` (or `quantization_config`) block the config may still
    // carry — only a present `.scales` makes an unresolvable scheme fatal (the
    // per-layer typed `InvariantViolation`). The non-quant `config.validate()`
    // above always runs. The result is threaded to the backbone + Dense head,
    // which pick quantized-vs-dense per layer by the same `.scales` sibling.
    let quant = if has_relevant_scales(&config, &weights) {
      config.quantization()?
    } else {
      None
    };
    let backbone = Gemma3Backbone::from_weights(&config, &mut weights, quant.as_ref())?;
    let dense_config = DenseConfig::from_hidden(config.hidden_size)?;
    let dense = DenseHead::from_weights(&dense_config, &mut weights, quant.as_ref())?;
    let pooling = resolve_pooling(pooling)?;

    Ok(Self {
      config,
      backbone,
      dense,
      pooling,
      pad_token_id: GEMMA_PAD_TOKEN_ID,
    })
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &Gemma3Config {
    &self.config
  }

  /// The baked deployment [`PoolingConfig`] (strategy + normalize + matryoshka
  /// dimension).
  #[inline(always)]
  pub fn pooling(&self) -> PoolingConfig {
    self.pooling
  }

  /// `true` if the token embedding loaded from a quantized checkpoint
  /// (test-only introspection for the quantized-load test).
  #[cfg(test)]
  pub(crate) fn embedding_is_quantized(&self) -> bool {
    self.backbone.embedding_is_quantized()
  }

  /// `true` if every backbone attention + MLP projection loaded quantized
  /// (test-only introspection).
  #[cfg(test)]
  pub(crate) fn all_projections_quantized(&self) -> bool {
    self.backbone.all_projections_quantized()
  }

  /// `true` if both Dense-head layers loaded quantized (test-only).
  #[cfg(test)]
  pub(crate) fn dense_head_is_quantized(&self) -> bool {
    self.dense.is_quantized()
  }

  /// Encode a `(batch, seq_len)` i32 token-id batch (with its `(batch, seq_len)`
  /// `{0,1}` attention mask) into the L2-normalized `(batch, output_dim)` text
  /// embedding.
  ///
  /// Mirrors `gemma3_text.py`'s (mlx-embeddings) `Model.__call__` tail: run the
  /// bidirectional backbone, **mean-pool** over the mask, apply the two-layer
  /// **Dense** projection, then (matryoshka-truncate and) L2-normalize. Pooling
  /// happens *before* the Dense projection (the SentenceTransformers pipeline
  /// order the reference follows: "Pool first, then dense").
  pub fn encode_text(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
    // Build the additive padding mask in the backbone's embedding dtype so SDPA
    // sees a matching-dtype mask.
    let dtype = self.backbone_dtype()?;
    let mask = build_additive_mask(attention_mask, dtype)?;

    // Backbone → (batch, seq_len, hidden).
    let hidden = self.backbone.forward(input_ids, &mask)?;

    // Mean-pool over the mask → (batch, hidden). The pooling reads the original
    // `{0,1}` mask (not the additive one), exactly as the reference passes the
    // raw `attention_mask` to `mean_pooling`.
    let pooled = mean_pooling(&hidden, attention_mask)?;

    // Dense projection → (batch, hidden).
    let projected = self.dense.forward(&pooled)?;

    // Matryoshka truncation + (optional) L2-normalize via the shared post-pool
    // tail. The baked `PoolingConfig` carries the deployment `dimension` /
    // `normalize`; `layer_norm` / `rms_norm` are off (EmbeddingGemma applies
    // neither to the pooled vector). This mirrors the reference's
    // `normalize_embeddings(text_embeds)` (with the ST `word_embedding_dimension`
    // truncation when present).
    pool_post(
      projected,
      self.pooling.normalize,
      self.pooling.dimension,
      self.pooling.layer_norm,
      self.pooling.rms_norm,
    )
  }

  /// The backbone's token-embedding dtype — the dtype the additive attention
  /// mask is cast to. Read from the embedding table (a cheap handle query, no
  /// eval).
  fn backbone_dtype(&self) -> Result<crate::dtype::Dtype> {
    self.backbone.embed_dtype()
  }
}

/// Resolve the baked [`PoolingConfig`] from the optional parsed `1_Pooling`
/// config. EmbeddingGemma is a **mean**-pooling sentence-encoder, so:
///
/// - the strategy is forced to [`PoolingStrategy::Mean`]; a parsed ST config
///   that declares a *different* mode is rejected with a typed error (loading a
///   non-mean pooling against EmbeddingGemma would silently produce the wrong
///   embedding);
/// - the matryoshka `dimension` is taken from the ST config (`None` keeps the
///   full width);
/// - `normalize` follows the ST config (always `true` for ST configs — the
///   `mlx-embeddings` / SentenceTransformers convention);
/// - the fused post-pool `layer_norm` / `rms_norm` are off (EmbeddingGemma
///   applies neither to the pooled vector).
///
/// When `pooling` is `None` (no `1_Pooling/config.json`), the deployment default
/// is mean + normalize + full dimension.
#[cfg(feature = "embeddinggemma")]
fn resolve_pooling(pooling: Option<&StPoolingConfig>) -> Result<PoolingConfig> {
  match pooling {
    Some(st) => {
      let strategy = st.strategy();
      if !matches!(strategy, PoolingStrategy::Mean) {
        return Err(crate::error::Error::InvariantViolation(
          crate::error::InvariantViolationPayload::new(
            "EmbeddingGemma: 1_Pooling pooling_mode",
            "must be mean — EmbeddingGemma is a mean-pooling sentence-encoder (a non-mean ST pooling config would produce a wrong embedding)",
          ),
        ));
      }
      Ok(PoolingConfig::new(
        PoolingStrategy::Mean,
        st.normalize(),
        st.dimension(),
        false,
        false,
      ))
    }
    None => Ok(PoolingConfig::new(
      PoolingStrategy::Mean,
      true,
      None,
      false,
      false,
    )),
  }
}

/// `true` if `weights` carries a `<prefix>.scales` sibling for some layer the
/// EmbeddingGemma loaders **actually consume** under this exact `config` — i.e.
/// the checkpoint is (at least partly) a quantized one.
///
/// This is the load-time half of the `.scales`-presence discriminator the
/// per-layer [`shared::Linear::from_weights`] / [`shared::Embedding::from_weights`]
/// use: those gate on the exact `<prefix>.scales` key for one layer; this
/// pre-scans the whole map once for the same signal across every layer the
/// loaders consume, so [`EmbeddingGemmaModel::from_weights`] resolves the
/// quantization scheme ([`Gemma3Config::quantization`]) ONLY when a scale
/// actually needs interpreting. A dense checkpoint (no relevant `.scales`) then
/// loads through the unchanged dense path regardless of any stale / foreign /
/// partial `quantization` (or `quantization_config`) block the config may still
/// carry — the scheme is irrelevant when no consumed layer is quantized.
///
/// The match is **exact and config-aware** (it runs after [`Gemma3Config::validate`]):
/// the relevant keys are precisely the `<prefix>.scales` siblings the
/// [`shared::Linear`] / [`shared::Embedding`] loaders build a `scales_key` for,
/// with the SAME `<prefix>` strings and the shared [`shared::SCALES_SUFFIX`] —
///
/// - `model.embed_tokens.scales` (the token embedding, [`backbone::Gemma3Backbone::from_weights`]);
/// - for each `i` in `0..config.num_hidden_layers` (the actual loaded layer
///   count): the attention `model.layers.{i}.self_attn.{q,k,v,o}_proj.scales`
///   and the MLP `model.layers.{i}.mlp.{gate,up,down}_proj.scales`;
/// - the Dense head's `dense.0.scales` / `dense.1.scales`
///   ([`dense::DenseHead::from_weights`]).
///
/// Building the loaders' exact `<prefix>.scales` strings and probing
/// `weights.contains_key` (not a suffix / `ends_with` match) means a foreign key
/// (`foreign.q_proj.scales`), an out-of-range layer index
/// (`model.layers.{N}.…` for `N >= num_hidden_layers`), or a never-quantized
/// tensor's stale `.scales` (`model.norm.scales`) is correctly IGNORED — exactly
/// the keys no loader ever consults. Reads only the map's keys (cheap string
/// lookups); no [`Array`] is touched.
#[cfg(feature = "embeddinggemma")]
fn has_relevant_scales(config: &Gemma3Config, weights: &HashMap<String, Array>) -> bool {
  use crate::embeddings::embeddinggemma::shared::SCALES_SUFFIX;

  // Probe the exact `<prefix>.scales` key the matching loader would build, with
  // the SAME `<prefix>` format and shared suffix the `Linear` / `Embedding`
  // loaders use.
  let has_scales = |prefix: &str| weights.contains_key(&format!("{prefix}{SCALES_SUFFIX}"));

  // The token embedding (backbone).
  if has_scales("model.embed_tokens") {
    return true;
  }
  // The Dense head's two bias-free projections (the `2_Dense` / `3_Dense`
  // modules, sanitized to `dense.0` / `dense.1`).
  if has_scales("dense.0") || has_scales("dense.1") {
    return true;
  }
  // Every per-layer projection, for the ACTUAL loaded layer count — an
  // out-of-range index `N >= num_hidden_layers` is never built, so its `.scales`
  // is irrelevant. `num_hidden_layers` is a `validate`d positive `i32`.
  (0..config.num_hidden_layers).any(|i| {
    let q = format!("model.layers.{i}.self_attn");
    let m = format!("model.layers.{i}.mlp");
    has_scales(&format!("{q}.q_proj"))
      || has_scales(&format!("{q}.k_proj"))
      || has_scales(&format!("{q}.v_proj"))
      || has_scales(&format!("{q}.o_proj"))
      || has_scales(&format!("{m}.gate_proj"))
      || has_scales(&format!("{m}.up_proj"))
      || has_scales(&format!("{m}.down_proj"))
  })
}

/// Rewrite a raw `google/embeddinggemma-300m` checkpoint into the layout
/// [`EmbeddingGemmaModel::from_weights`] loads — the Rust analogue of
/// `gemma3_text.py`'s (mlx-embeddings) `Model.sanitize`.
///
/// Three-way per-key branch:
/// 1. **Backbone key** (`"linear" not in k and "dense" not in k`): namespace it
///    under `model.` (HF stores the Gemma3 transformer's keys without the
///    `model.` nesting `mlx-lm` uses) unless it already starts with `model`.
/// 2. **Un-renamed ST Dense module** (`"linear" in k and "dense" not in k`):
///    rewrite the `{n}_Dense.linear` prefix to `dense.{0,1}`, classifying the
///    module by its **source module-prefix number** `{n}` — not by tensor shape.
///    The distinct `{n}` values across the Dense keys are mapped to dense ids in
///    ascending numeric order: the lowest `{n}` (the earlier SentenceTransformers
///    module) is `dense.0`, the next is `dense.1`. EmbeddingGemma has exactly
///    `2_Dense` and `3_Dense`, so this yields `2 → dense.0`, `3 → dense.1` — the
///    `2_Dense` expansion (`hidden → hidden*4`) and the `3_Dense` contraction
///    (`hidden*4 → hidden`), the same routing the reference's width test produces
///    on a dense checkpoint, but unambiguous for a **quantized** module whose
///    `.weight` is bit-packed and whose `.scales` / `.biases` are
///    `(out, in/group_size)` (a width test would misroute those siblings). All
///    three siblings (`.weight`, `.scales`, `.biases`) of one module route
///    together by `{n}`. When any raw ST Dense keys are present the distinct
///    module-number set must be **exactly** `{2, 3}`; any other set (a lone
///    `{2}`, a `{1, 2}`, a `{2, 4}`, or three-plus distinct numbers) is an
///    unexpected ST Dense layout, rejected with [`crate::Error::OutOfRange`]
///    rather than guessed (mapping an unexpected pair by ascending order would
///    silently bind the wrong ST modules to `dense.0` / `dense.1`).
/// 3. **Already-renamed Dense key** (`"dense" in k`): keep verbatim
///    (idempotency).
///
/// A duplicate destination key is rejected with [`crate::Error::KeyCollision`]
/// (via [`crate::model_validation::insert_unique`]) rather than letting an
/// arbitrary survivor silently overwrite the other.
///
/// The rewritten keys are built through the fallible String path (typed
/// [`crate::Error::AllocFailure`]) so a hostile checkpoint with enormous keys
/// cannot abort on a per-key rewrite allocation; the destination map reserve and
/// each `insert_unique` slot are fallible too.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  // Pass 1: classify the ST Dense modules by their source module-prefix number.
  // Every `{n}_Dense.linear.<suffix>` key contributes its `{n}`; the distinct
  // numbers map to dense ids in ascending numeric order (lowest → "0"). This
  // routes all three siblings (`.weight`, `.scales`, `.biases`) of one module
  // together regardless of packing — a quantized module's bit-packed `.weight`
  // and `(out, in/group_size)` `.scales`/`.biases` cannot be classified by shape.
  let mut dense_modules = DenseModuleMap::new();
  for k in weights.keys() {
    if k.contains("linear")
      && !k.contains("dense")
      && let Some((_, number, _)) = split_dense_linear_key(k)
      && let Ok(n) = number.parse::<u32>()
    {
      dense_modules.insert(n)?;
    }
  }
  // When the checkpoint carries raw ST Dense keys (`len > 0`), the distinct
  // module-number set must be exactly `{2, 3}` — any other set (`{1, 2}`,
  // `{2, 4}`, a lone `{2}`) is an unexpected ST Dense layout, rejected rather
  // than mapped by ascending order onto `dense.0` / `dense.1`. A checkpoint
  // already in the sanitized `dense.{0,1}` layout records nothing here
  // (`len == 0`) and is untouched.
  if dense_modules.len > 0 {
    dense_modules.validate_layout()?;
  }

  // Pass 2: rewrite each key into the loaded layout.
  let mut out: HashMap<String, Array> = HashMap::new();
  reserve_or_error(
    &mut out,
    "EmbeddingGemma sanitize: destination map",
    weights.len(),
  )?;
  for (k, v) in weights {
    let has_linear = k.contains("linear");
    let has_dense = k.contains("dense");
    let new_key = if !has_linear && !has_dense {
      // 1. Backbone key → namespace under `model.` (unless already there).
      if k.starts_with("model") {
        fallible_clone_str("sanitize: model key", &k)?
      } else {
        fallible_concat("sanitize: model namespace", &["model.", &k])?
      }
    } else if has_linear && !has_dense {
      // 2. Un-renamed ST Dense module: rewrite `{n}_Dense.linear.<suffix>` →
      //    `dense.{id}.<suffix>`, `id` from the precomputed module-number map.
      rewrite_dense_linear_key(&k, &dense_modules)?
    } else {
      // 3. Already-renamed Dense key: keep verbatim.
      fallible_clone_str("sanitize: dense key", &k)?
    };
    crate::model_validation::insert_unique(&mut out, new_key, v, "EmbeddingGemma sanitize")?;
  }
  Ok(out)
}

/// Split an un-renamed ST Dense key `{head}{n}_Dense.linear{suffix}` into its
/// three pieces: the `head` before the module number, the `{n}` digit run, and
/// the trailing `suffix` (`.weight`, `.scales`, or `.biases`).
///
/// The HF Dense keys are exactly `{n}_Dense.linear.<suffix>`, so this finds the
/// `_Dense.linear` literal, walks back over the leading digits, and returns the
/// borrowed slices. Returns `None` when the `_Dense.linear` marker is absent or
/// is not preceded by any digit (so callers can treat an unparseable Dense key
/// as a malformed layout rather than guess a module id). Shared by the
/// classification pass (which reads `{n}`) and [`rewrite_dense_linear_key`]
/// (which re-splices), so the marker-find / digit-walk lives in one place.
#[cfg(feature = "embeddinggemma")]
fn split_dense_linear_key(key: &str) -> Option<(&str, &str, &str)> {
  const MARKER: &str = "_Dense.linear";
  let marker_at = key.find(MARKER)?;
  // Walk back over the digits of `{n}_Dense.linear`; `None` if there are none.
  let digits_start = key[..marker_at]
    .char_indices()
    .rev()
    .take_while(|(_, c)| c.is_ascii_digit())
    .last()
    .map(|(i, _)| i)?;
  let head = &key[..digits_start]; // anything before the `{n}_Dense.linear`
  let number = &key[digits_start..marker_at]; // the `{n}` digit run
  let suffix = &key[marker_at + MARKER.len()..]; // `.weight` / `.scales` / `.biases`
  Some((head, number, suffix))
}

/// Rewrite an un-renamed ST Dense key `{head}{n}_Dense.linear{suffix}` to
/// `{head}dense.{id}{suffix}` (the reference's `re.sub(r"\d+_Dense\.linear",
/// f"dense.{key_id}", k)`), where `id` is the dense id `modules` assigned to the
/// source module number `{n}`. The full trailing `suffix` (`.weight`, `.scales`,
/// or `.biases`) is preserved, so all three siblings of one quantized module
/// route to the same `dense.{id}`.
///
/// A key whose `{n}_Dense.linear` marker is absent or has no parseable module
/// number is an unexpected ST Dense layout (never produced by a real checkpoint,
/// but possible from a hostile map): rejected with a typed
/// [`crate::Error::InvariantViolation`] rather than guessing a slot. Built
/// through the fallible String path (typed [`crate::Error::AllocFailure`]).
#[cfg(feature = "embeddinggemma")]
fn rewrite_dense_linear_key(key: &str, modules: &DenseModuleMap) -> Result<String> {
  if let Some((head, number, suffix)) = split_dense_linear_key(key)
    && let Some(id) = number.parse::<u32>().ok().and_then(|n| modules.id_of(n))
  {
    return fallible_concat(
      "sanitize: Dense linear rename",
      &[head, "dense.", id, suffix],
    );
  }
  Err(crate::error::Error::InvariantViolation(
    crate::error::InvariantViolationPayload::new(
      "EmbeddingGemma sanitize: ST Dense key",
      "must contain a parseable {n}_Dense.linear module marker",
    ),
  ))
}

/// The distinct `{n}_Dense.linear` module numbers a sanitize pass found, mapped
/// to dense ids in ascending numeric order: the lowest module number is
/// `dense.0`, the next is `dense.1`. EmbeddingGemma's SentenceTransformers
/// layout is **exactly** `2_Dense` and `3_Dense`, so when any raw ST Dense keys
/// are present the distinct module-number set must be exactly `{2, 3}`. Any
/// other set — a single `{2}`, a `{1, 2}`, a `{2, 4}`, or three-plus distinct
/// numbers — is an unexpected ST Dense layout, rejected with
/// [`crate::Error::OutOfRange`] (mapping an unexpected pair by ascending order
/// would silently bind the wrong ST modules to `dense.0` / `dense.1`).
///
/// Heap-free: the at-most-two distinct numbers live in a fixed stack array kept
/// sorted ascending, so the dense id of a number is its index.
#[cfg(feature = "embeddinggemma")]
struct DenseModuleMap {
  /// The distinct module numbers, kept sorted ascending (`len` entries valid).
  nums: [u32; 2],
  len: usize,
}

#[cfg(feature = "embeddinggemma")]
impl DenseModuleMap {
  /// The dense ids, indexed by ascending module-number position.
  const IDS: [&'static str; 2] = ["0", "1"];
  /// The one ST Dense module-number set a real EmbeddingGemma checkpoint
  /// carries (`2_Dense`, `3_Dense`); see [`Self::validate_layout`].
  const EXPECTED: [u32; 2] = [2, 3];

  fn new() -> Self {
    Self {
      nums: [0; 2],
      len: 0,
    }
  }

  /// Record a module number, keeping the set sorted-ascending and deduplicated.
  /// A third **distinct** number is already an unexpected ST Dense layout →
  /// typed [`crate::Error::OutOfRange`] (do not guess which two are the real
  /// head); the full exact-`{2, 3}` requirement is enforced once the pass
  /// completes by [`Self::validate_layout`].
  fn insert(&mut self, n: u32) -> Result<()> {
    // Already present (a sibling `.weight`/`.scales`/`.biases` of the same
    // module): nothing to do.
    if self.nums[..self.len].contains(&n) {
      return Ok(());
    }
    if self.len == self.nums.len() {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "EmbeddingGemma sanitize: distinct ST Dense module count",
          "must be at most 2 (EmbeddingGemma has 2_Dense and 3_Dense)",
          smol_str::format_smolstr!("{}", self.len + 1),
        ),
      ));
    }
    // Insertion-sort the new number into ascending position.
    let mut i = self.len;
    while i > 0 && self.nums[i - 1] > n {
      self.nums[i] = self.nums[i - 1];
      i -= 1;
    }
    self.nums[i] = n;
    self.len += 1;
    Ok(())
  }

  /// Assert the recorded distinct module-number set is **exactly** `{2, 3}`
  /// (EmbeddingGemma's `2_Dense` / `3_Dense`). Called after pass 1 *only when
  /// raw ST Dense keys were present* (`len > 0`); the already-sanitized
  /// `dense.0` / `dense.1` passthrough records nothing (`len == 0`) and so is
  /// never reached.
  ///
  /// The earlier per-insert guard already rejects three-plus distinct numbers;
  /// this generalizes the gate to a full exact-set check, so a two-module set
  /// that is *not* `{2, 3}` — `{1, 2}`, `{2, 4}`, … — or a lone `{2}` / `{3}`
  /// is rejected rather than mapped by ascending order onto the wrong slots.
  /// The offending set is reported as the [`crate::Error::OutOfRange`] value.
  fn validate_layout(&self) -> Result<()> {
    if self.nums[..self.len] != Self::EXPECTED {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "EmbeddingGemma sanitize: distinct ST Dense module-number set",
          "must be exactly {2, 3} (EmbeddingGemma's 2_Dense and 3_Dense)",
          smol_str::format_smolstr!("{:?}", &self.nums[..self.len]),
        ),
      ));
    }
    Ok(())
  }

  /// The dense id (`"0"` / `"1"`) for module number `n`, by its ascending
  /// position; `None` if `n` was never recorded.
  fn id_of(&self, n: u32) -> Option<&'static str> {
    self.nums[..self.len]
      .iter()
      .position(|&m| m == n)
      .map(|pos| Self::IDS[pos])
  }
}

/// Fallibly build the concatenation of `parts` into a freshly-reserved
/// [`String`], turning an allocator failure into a typed
/// [`crate::Error::AllocFailure`] instead of the abort `format!` would raise.
/// Mirrors the SigLIP2 `fallible_concat`.
#[cfg(feature = "embeddinggemma")]
fn fallible_concat(context: &'static str, parts: &[&str]) -> Result<String> {
  let total = parts
    .iter()
    .fold(0usize, |acc, p| acc.saturating_add(p.len()));
  let mut out = String::new();
  out.try_reserve_exact(total).map_err(|e| {
    crate::error::Error::AllocFailure(crate::error::AllocFailurePayload::new(
      "EmbeddingGemma key rewrite",
      context,
      total as u64,
      e,
    ))
  })?;
  for p in parts {
    out.push_str(p);
  }
  Ok(out)
}

/// Fallibly clone `s` into an owned [`String`] (the single-part
/// [`fallible_concat`]).
#[cfg(feature = "embeddinggemma")]
fn fallible_clone_str(context: &'static str, s: &str) -> Result<String> {
  fallible_concat(context, &[s])
}

/// The [`EmbeddingModelConstructor`] that builds an [`EmbeddingGemmaModel`] from
/// a loaded model directory, for registration into an
/// [`EmbeddingModelTypeRegistry`].
///
/// Parses the raw `config.json`, [`sanitize`]s a cheap clone of the loaded
/// weight map (mlx [`Array`] is a refcounted handle — the clone shares device
/// buffers, no data copy), and builds the model, baking the parsed
/// `1_Pooling/config.json` (`pooling`, the matryoshka dimension + mean strategy)
/// into the model's [`PoolingConfig`]. The map reserve and each cloned key are
/// built through the fallible path (typed [`crate::Error::AllocFailure`]).
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
pub fn constructor() -> EmbeddingModelConstructor {
  Box::new(
    |loaded: &LoadedEmbeddingModel,
     pooling: Option<&StPoolingConfig>|
     -> Result<Box<dyn EmbeddingModel>> {
      let config = Gemma3Config::from_json(loaded.config_json())?;
      let mut raw: HashMap<String, Array> = HashMap::new();
      reserve_or_error(
        &mut raw,
        "EmbeddingGemma constructor: sanitizable weight-map clone",
        loaded.weights_ref().len(),
      )?;
      for (k, v) in loaded.weights_ref() {
        let key = fallible_clone_str("constructor: weight-map key clone", k)?;
        raw.insert(key, v.try_clone()?);
      }
      let weights = sanitize(raw)?;
      let model = EmbeddingGemmaModel::from_weights(config, weights, pooling)?;
      Ok(Box::new(model))
    },
  )
}

/// Register [`EmbeddingGemmaModel`] under [`MODEL_TYPE`] (`"gemma3_text"`) into
/// `registry`, returning any constructor it displaced.
///
/// The registry is the documented architecture extension point; call this to
/// enable loading an EmbeddingGemma checkpoint through
/// [`crate::embeddings::load`].
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
pub fn register(registry: &mut EmbeddingModelTypeRegistry) -> Option<EmbeddingModelConstructor> {
  registry.register(MODEL_TYPE, constructor())
}

/// EmbeddingGemma as the universal text seam ([`TextEmbedder`]). It owns both
/// text stages a generic pipeline cannot standardize:
///
/// - [`text_encoding`](TextEmbedder::text_encoding) declares the Gemma
///   tokenizer's **dynamic right-pad** scheme ([`Padding::DynamicRightPad`]):
///   tokenize with special tokens, right-pad each row to the batch maximum with
///   the `<pad>` id, mask `0` over pad cells. No fixed length, no per-text
///   truncation cap (the consuming application bounds oversized prompts).
/// - [`embed_text`](TextEmbedder::embed_text) runs the bidirectional backbone →
///   mean-pool → Dense projection → L2-normalize via
///   [`encode_text`](EmbeddingGemmaModel::encode_text).
#[cfg(feature = "embeddinggemma")]
impl TextEmbedder for EmbeddingGemmaModel {
  fn text_encoding(&self) -> TextEncoding {
    TextEncoding::new(
      // The Gemma tokenizer encodes with special tokens (BOS).
      true,
      // No explicit per-text truncation cap — the consuming application bounds
      // oversized prompts (the library contract). The DynamicRightPad scheme has
      // no intrinsic cap either, so each text is tokenized at full length.
      None,
      Padding::DynamicRightPad {
        pad_token_id: self.pad_token_id,
      },
    )
  }

  fn embed_text(&self, input_ids: &Array, attention_mask: &Array) -> Result<Embedding> {
    Ok(Embedding::new(self.encode_text(input_ids, attention_mask)?))
  }
}

/// EmbeddingGemma answers the load factory umbrella's universal text capability;
/// it has no contrastive / late-interaction capability (a single-vector
/// sentence-encoder). The concrete model is reachable by downcast via
/// [`as_any`](EmbeddingModel::as_any).
#[cfg(feature = "embeddinggemma")]
impl EmbeddingModel for EmbeddingGemmaModel {
  fn as_text_embedder(&self) -> Option<&dyn TextEmbedder> {
    Some(self)
  }
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}

#[cfg(all(test, feature = "embeddinggemma"))]
mod tests;

//! Whisper neural-network building blocks: [`Linear`], [`Embedding`], and the
//! Whisper-variant [`MultiHeadAttention`].
//!
//! Faithful port of the corresponding `nn.Module`s in
//! `mlx_audio.stt.models.whisper.whisper` (`whisper.py:328-375` for
//! `MultiHeadAttention`; the `nn.Linear` / `nn.Embedding` it composes).
//!
//! These are kept **private to the Whisper model**: the Whisper attention is
//! non-standard (the `head_dim ** -0.25` scale on BOTH q and k, manual
//! `qkv_attention`, cross-attention K/V caching) and cannot reuse the generic
//! [`crate::lm::nn::attention`] fast-SDPA path, so it ports `qkv_attention`
//! by hand for exact parity.

use crate::{
  Array, Result,
  lm::nn::{activations::gelu, norm::LayerNorm},
  ops::{self, shape::concatenate},
};

/// A dense linear projection `y = x @ Wᵀ (+ b)`.
///
/// Mirrors `mlx.nn.Linear`: `weight` is stored `(out_features, in_features)`
/// and the forward transposes it, so `y = x @ weight.T + bias`. `bias` is
/// optional — Whisper's `key` projection is constructed `bias=False`
/// (`whisper.py:333`), every other projection carries a bias.
#[derive(Debug)]
pub(crate) struct Linear {
  /// `(out_features, in_features)` weight (the `mlx.nn.Linear` layout).
  weight: Array,
  /// Optional `(out_features,)` bias. `None` for Whisper's `key` projection.
  bias: Option<Array>,
}

impl Linear {
  /// Construct from a `(out_features, in_features)` `weight` and an optional
  /// `(out_features,)` `bias`.
  pub(crate) fn new(weight: Array, bias: Option<Array>) -> Self {
    Self { weight, bias }
  }

  /// `y = x @ weightᵀ (+ bias)`. `x` is `(..., in_features)`; the result is
  /// `(..., out_features)`.
  ///
  /// # Errors
  /// Propagates the transpose / matmul / add op errors.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    let wt = self.weight.transpose()?;
    let y = x.matmul(&wt)?;
    match &self.bias {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }

  /// The `(out_features, in_features)` weight (test / introspection).
  #[cfg(test)]
  pub(crate) fn weight_ref(&self) -> &Array {
    &self.weight
  }
}

/// A token embedding table `weight` of shape `(n_vocab, n_state)`, with a
/// weight-tied [`Embedding::as_linear`] projection.
///
/// Mirrors `mlx.nn.Embedding`: [`Embedding::forward`] gathers rows by integer
/// id; [`Embedding::as_linear`] reuses the SAME `weight` as a linear
/// projection `x @ weightᵀ` (the Whisper decoder's weight-tied logit head —
/// `whisper.py` `TextDecoder.__call__` ends with
/// `self.token_embedding.as_linear(x)`).
#[derive(Debug)]
pub(crate) struct Embedding {
  /// `(n_vocab, n_state)` embedding table.
  weight: Array,
}

impl Embedding {
  /// Construct from a `(n_vocab, n_state)` embedding table.
  pub(crate) fn new(weight: Array) -> Self {
    Self { weight }
  }

  /// Gather embedding rows: `weight[ids]` — gather along axis 0 (the vocab
  /// axis), mirroring `mlx.nn.Embedding.__call__`'s `self.weight[x]`. `ids`
  /// is an integer [`Array`] of any shape `S`; the result is `S ++
  /// (n_state,)`. (Plain `take` would flatten the table — `take_axis(.., 0)`
  /// is the row-gather.)
  ///
  /// # Errors
  /// Propagates the gather (`take_axis`) op error.
  pub(crate) fn forward(&self, ids: &Array) -> Result<Array> {
    self.weight.take_axis(ids, 0)
  }

  /// Weight-tied linear projection `x @ weightᵀ` (the decoder logit head).
  /// `x` is `(..., n_state)`; the result is `(..., n_vocab)`.
  ///
  /// # Errors
  /// Propagates the transpose / matmul op errors.
  pub(crate) fn as_linear(&self, x: &Array) -> Result<Array> {
    let wt = self.weight.transpose()?;
    x.matmul(&wt)
  }

  /// The `(n_vocab, n_state)` embedding table (test / introspection).
  #[cfg(test)]
  pub(crate) fn weight_ref(&self) -> &Array {
    &self.weight
  }
}

/// Whisper multi-head attention (`whisper.py:328-375`).
///
/// **Non-standard scaling**: `scale = (n_state / n_head) ** -0.25` is applied
/// to BOTH `q` and `k` before `q @ kᵀ` (so the effective scale is
/// `head_dim ** -0.5`, but split across the two operands — bit-for-bit with
/// the reference, which does NOT fold it into one factor). The softmax is
/// `precise=True`. The causal mask is **additive** and **offset-aware** —
/// `qk + mask[offset : offset + T_q, 0 : offset + T_q]`, where `offset` is the
/// warm-cache key count, so a multi-token warm-cache step masks each new query
/// against the keys at or before its absolute position. This ports
/// `qkv_attention` by hand rather than using the fast SDPA so the scaling split
/// and the mask slice match exactly.
///
/// Projections: `query` / `value` / `out` carry a bias; `key` does **not**
/// (`whisper.py:333`).
#[derive(Debug)]
pub(crate) struct MultiHeadAttention {
  n_head: usize,
  query: Linear,
  key: Linear,
  value: Linear,
  out: Linear,
}

/// The `(key, value)` pair produced by an attention step — the KV cache
/// payload. For self-attention these are the FULL (post-concatenation)
/// projected key/value along the time axis; for cross-attention they are the
/// projected encoder key/value (computed once, reused across decode steps).
/// Mirrors the reference's `(k, v)` tuple returned from `__call__`.
pub(crate) type KvPair = (Array, Array);

impl MultiHeadAttention {
  /// Construct from the four already-built [`Linear`] projections and the head
  /// count.
  pub(crate) fn new(n_head: usize, query: Linear, key: Linear, value: Linear, out: Linear) -> Self {
    Self {
      n_head,
      query,
      key,
      value,
      out,
    }
  }

  /// Run attention. Faithful port of `MultiHeadAttention.__call__`
  /// (`whisper.py:337-359`).
  ///
  /// - `x`: the query input `(B, T_q, n_state)`.
  /// - `xa`: the cross-attention key/value source `(B, T_kv, n_state)`.
  ///   `None` ⇒ self-attention (key/value projected from `x`).
  /// - `mask`: an optional additive causal mask `(>= offset + T_q, >= offset +
  ///   T_q)` (sliced offset-aware to `[offset : offset + T_q, 0 : offset +
  ///   T_q]`, where `offset` is the warm-cache key count) — the decoder's causal
  ///   mask.
  /// - `kv_cache`: an optional incoming `(k, v)` pair.
  ///   - self-attention (`xa is None`): the cached `(k, v)` are concatenated
  ///     with the freshly-projected `(k, v)` along the time axis (axis 1).
  ///   - cross-attention (`xa is Some`): when `kv_cache` is `Some`, the cached
  ///     `(k, v)` are reused verbatim (the encoder K/V never change across
  ///     decode steps); when `None`, they are projected from `xa`.
  ///
  /// Returns `(output, (k, v))` — the projected output `(B, T_q, n_state)` and
  /// the (full) `(k, v)` to store back in the cache. The reference's third
  /// `qk` return (the attention weights, for word-timing DTW) is dropped on
  /// this path; [`Self::forward_with_qk`] is the variant that surfaces it.
  ///
  /// This is the path normal decode reaches (through
  /// [`ResidualAttentionBlock::forward`]); it runs the no-`qk` attention core
  /// ([`Self::qkv_attention_no_qk`]), which drops the pre-softmax score tensor
  /// the instant the softmax weights are formed, so no `(B, H, T_q, T_kv)`
  /// score buffer is constructed-and-returned only to be discarded. The
  /// `qk`-returning core ([`Self::qkv_attention`]) is reserved for
  /// [`Self::forward_with_qk`] / the cross-`qk` collection path.
  ///
  /// # Errors
  /// Propagates the projection / reshape / transpose / matmul / softmax errors.
  pub(crate) fn forward(
    &self,
    x: &Array,
    xa: Option<&Array>,
    mask: Option<&Array>,
    kv_cache: Option<&KvPair>,
  ) -> Result<(Array, KvPair)> {
    let q = self.query.forward(x)?;
    let (k, v) = self.project_kv(x, xa, kv_cache)?;
    let wv = self.qkv_attention_no_qk(&q, &k, &v, mask)?;
    let out = self.out.forward(&wv)?;
    Ok((out, (k, v)))
  }

  /// Run attention and also surface the pre-`v` attention weights `qk` — the
  /// full three-tuple return of `MultiHeadAttention.__call__`
  /// (`whisper.py:337-359`), where `qk = softmax(q @ kᵀ * scale)` before the
  /// `@ v` multiply.
  ///
  /// Identical to [`Self::forward`] except it returns the third `qk` tensor
  /// `(B, H, T_q, T_kv)` (the per-head attention weights over the keys),
  /// mirroring the reference's `return self.out(wv), (k, v), qk`. The cross-
  /// attention `qk` is the input the later word-timestamp DTW alignment
  /// consumes; this method only extracts and exposes it (the alignment itself
  /// is a later phase).
  ///
  /// # Errors
  /// Propagates the projection / reshape / transpose / matmul / softmax errors.
  pub(crate) fn forward_with_qk(
    &self,
    x: &Array,
    xa: Option<&Array>,
    mask: Option<&Array>,
    kv_cache: Option<&KvPair>,
  ) -> Result<(Array, KvPair, Array)> {
    let q = self.query.forward(x)?;
    let (k, v) = self.project_kv(x, xa, kv_cache)?;
    let (wv, qk) = self.qkv_attention(&q, &k, &v, mask)?;
    let out = self.out.forward(&wv)?;
    Ok((out, (k, v), qk))
  }

  /// Project (and cache-merge) the attention key/value pair from the query
  /// input `x` / cross-attention source `xa` / incoming cache — the shared
  /// `(k, v)` derivation used by BOTH [`Self::forward`] and
  /// [`Self::forward_with_qk`].
  ///
  /// - self-attention (`xa is None`): project `(k, v)` from `x`, concatenating
  ///   any cached `(k, v)` along the time axis (axis 1);
  /// - cross-attention, first step (`xa is Some`, no cache): project `(k, v)`
  ///   from the encoder states `xa`;
  /// - cross-attention, cached (`xa is Some`, cache present): reuse the cached
  ///   encoder `(k, v)` verbatim, ignoring this step's `xa` — matching the
  ///   reference decoder, which projects the cross-attention `(k, v)` from the
  ///   first step's encoder states and reuses it for the rest of the decode.
  ///   The library trusts that a warm cache is threaded with the same
  ///   utterance's `xa`; reusing one cache across different utterances is
  ///   caller misuse the library does not detect (it would decode against the
  ///   first utterance's features). See `WhisperModel`'s threat-model note.
  ///
  /// # Errors
  /// Propagates the projection / concatenate / clone errors.
  fn project_kv(&self, x: &Array, xa: Option<&Array>, kv_cache: Option<&KvPair>) -> Result<KvPair> {
    match (xa, kv_cache) {
      (None, cache) => {
        let mut k = self.key.forward(x)?;
        let mut v = self.value.forward(x)?;
        if let Some((ck, cv)) = cache {
          k = concatenate(&[ck, &k], 1)?;
          v = concatenate(&[cv, &v], 1)?;
        }
        Ok((k, v))
      }
      (Some(xa), None) => {
        let k = self.key.forward(xa)?;
        let v = self.value.forward(xa)?;
        Ok((k, v))
      }
      (Some(_), Some((ck, cv))) => Ok((ck.try_clone()?, cv.try_clone()?)),
    }
  }

  /// Head-split `q` / `k` / `v` and form the scaled, masked PRE-softmax scores
  /// `qk` `(B, H, T_q, T_kv)` — the shared front half of the manual
  /// scaled-dot-product attention (`whisper.py:361-375`), used by both the
  /// no-`qk` and `qk`-returning cores.
  ///
  /// `q` is `(B, T_q, n_state)`, `k` / `v` are `(B, T_kv, n_state)`. Applies
  /// the `head_dim ** -0.25` scale to BOTH q and k and adds the offset-aware
  /// causal `mask` (if any). Returns `(qk, v_heads, n_batch, n_ctx, n_state)`:
  /// the pre-softmax scores, the head-split `v` `(B, H, T_kv, D)`, and the
  /// dims needed to recombine the heads back to `(B, T_q, n_state)`.
  ///
  /// # Errors
  /// Propagates the reshape / transpose / matmul / add errors.
  fn attention_scores(
    &self,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
  ) -> Result<(Array, Array, i32, i32, usize)> {
    let q_shape = q.shape();
    let n_batch = q_shape[0] as i32;
    let n_ctx = q_shape[1] as i32;
    let n_state = q_shape[2];
    let n_head = self.n_head as i32;
    let head_dim = (n_state / self.n_head) as i32;
    // scale = (n_state // n_head) ** -0.25, applied to BOTH q and k.
    let scale = (n_state as f64 / self.n_head as f64).powf(-0.25) as f32;
    let scale_arr = Array::full::<f32>(&[0i32; 0], scale)?;

    // q: (B, T_q, n_state) -> (B, T_q, H, D) -> (B, H, T_q, D), then * scale.
    let k_ctx = k.shape()[1] as i32;
    let q = q
      .reshape(&[n_batch, n_ctx, n_head, head_dim])?
      .transpose_axes(&[0, 2, 1, 3])?
      .multiply(&scale_arr)?;
    // k: (B, T_kv, n_state) -> (B, T_kv, H, D) -> (B, H, D, T_kv) (transposed
    // for q @ k), then * scale.
    let k = k
      .reshape(&[n_batch, k_ctx, n_head, head_dim])?
      .transpose_axes(&[0, 2, 3, 1])?
      .multiply(&scale_arr)?;
    // v: (B, T_kv, n_state) -> (B, T_kv, H, D) -> (B, H, T_kv, D).
    let v = v
      .reshape(&[n_batch, k_ctx, n_head, head_dim])?
      .transpose_axes(&[0, 2, 1, 3])?;

    // qk = q @ k -> (B, H, T_q, T_kv).
    let mut qk = q.matmul(&k)?;
    if let Some(m) = mask {
      // Offset-aware additive causal mask. With a warm self-attention cache the
      // key axis is `k_ctx = offset + T_q` (the cached positions plus the new
      // tokens), so the new queries sit at absolute positions `offset ..
      // offset + T_q`. Slice the precomputed `(n_text_ctx, n_text_ctx)` causal
      // mask to the ROWS of those query positions (`offset .. offset + T_q`) and
      // the COLUMNS of all keys (`0 .. k_ctx`), giving a `(T_q, k_ctx)` block
      // that broadcasts against `qk`'s `(B, H, T_q, k_ctx)` and masks each new
      // token against exactly the keys at or before its absolute position. The
      // offset is `k_ctx - T_q` (`k_ctx >= T_q`, the cache only grows); for a
      // fresh cache `offset == 0`, recovering the cold-start `mask[:T_q, :T_q]`.
      let offset = k_ctx - n_ctx;
      let m_slice = ops::indexing::slice(m, &[offset, 0], &[offset + n_ctx, k_ctx], &[1, 1])?;
      qk = qk.add(&m_slice)?;
    }
    Ok((qk, v, n_batch, n_ctx, n_state))
  }

  /// Recombine `softmax(qk) @ v` back to `(B, T_q, n_state)` — the shared back
  /// half of the manual scaled-dot-product attention. `w` is the post-softmax
  /// weights `(B, H, T_q, T_kv)`, `v` the head-split values `(B, H, T_kv, D)`.
  ///
  /// # Errors
  /// Propagates the matmul / transpose / reshape errors.
  fn combine_heads(
    w: &Array,
    v: &Array,
    n_batch: i32,
    n_ctx: i32,
    n_state: usize,
  ) -> Result<Array> {
    // out = (w @ v).transpose(0, 2, 1, 3).reshape(B, T_q, n_state).
    w.matmul(v)?
      .transpose_axes(&[0, 2, 1, 3])?
      .reshape(&[n_batch, n_ctx, n_state as i32])
  }

  /// The manual scaled-dot-product attention WITHOUT surfacing `qk`
  /// (`whisper.py:361-375`) — the core normal decode reaches through
  /// [`Self::forward`].
  ///
  /// Splits the heads, applies the `head_dim ** -0.25` scale to BOTH q and k,
  /// computes `softmax(q @ kᵀ + mask)`, recombines the heads, and returns ONLY
  /// the output `(B, T_q, n_state)`. The pre-softmax score tensor `qk`
  /// `(B, H, T_q, T_kv)` is dropped the instant the softmax weights are formed
  /// (it is neither returned nor retained past the softmax), so this path never
  /// constructs-and-returns a score buffer only to discard it. The
  /// `qk`-returning [`Self::qkv_attention`] is used solely for the cross-`qk`
  /// collection path.
  ///
  /// # Errors
  /// Propagates the reshape / transpose / matmul / softmax / add errors.
  fn qkv_attention_no_qk(
    &self,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
  ) -> Result<Array> {
    let (qk, v, n_batch, n_ctx, n_state) = self.attention_scores(q, k, v, mask)?;
    // w = softmax(qk, axis=-1, precise=True); the pre-softmax scores are no
    // longer needed once `w` is formed, so drop `qk` here — it does not escape.
    let w = ops::misc::softmax_axis(&qk, -1, true)?;
    drop(qk);
    Self::combine_heads(&w, &v, n_batch, n_ctx, n_state)
  }

  /// The manual scaled-dot-product attention RETURNING `qk`
  /// (`whisper.py:361-375`) — used only by [`Self::forward_with_qk`] / the
  /// cross-`qk` collection path.
  ///
  /// Identical core to [`Self::qkv_attention_no_qk`], but also returns the
  /// per-head PRE-softmax scores `qk` `(B, H, T_q, T_kv)` (`q @ kᵀ * scale`,
  /// masked — the reference's second `qkv_attention` return). The scores are
  /// the alignment signal the later word-timestamp DTW consumes.
  ///
  /// # Errors
  /// Propagates the reshape / transpose / matmul / softmax / add errors.
  fn qkv_attention(
    &self,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
  ) -> Result<(Array, Array)> {
    let (qk, v, n_batch, n_ctx, n_state) = self.attention_scores(q, k, v, mask)?;
    // w = softmax(qk, axis=-1, precise=True).
    let w = ops::misc::softmax_axis(&qk, -1, true)?;
    let out = Self::combine_heads(&w, &v, n_batch, n_ctx, n_state)?;
    // Return the PRE-softmax `qk` scores `(B, H, T_q, T_kv)` alongside the
    // output, exactly as the reference `qkv_attention` returns `out, qk` (the
    // scaled-and-masked scores, not the post-softmax `w`).
    Ok((out, qk))
  }
}

/// The per-block KV cache: `(self_attn_kv, cross_attn_kv)`.
///
/// Mirrors the reference's `kv_cache = (kv, cross_kv)` tuple threaded through
/// [`ResidualAttentionBlock`] (`whisper.py:395-406`):
/// - `.0` — the self-attention `(k, v)` (grows by one step each decode call);
/// - `.1` — the cross-attention `(k, v)` (the encoder K/V, computed once on the
///   first decode step and reused verbatim thereafter).
///
/// Both arms are `None` on the first call (the encoder's self-attention-only
/// blocks never use cross-attention, so `.1` stays `None` there).
pub(crate) type BlockKvCache = (Option<KvPair>, Option<KvPair>);

/// A Whisper transformer block — `ResidualAttentionBlock` (`whisper.py:378-406`).
///
/// Pre-norm residual structure:
/// 1. `x = x + attn(attn_ln(x), mask, self_kv_cache)` — masked self-attention;
/// 2. (decoder only) `x = x + cross_attn(cross_attn_ln(x), xa, cross_kv_cache)`
///    — cross-attention over the encoder states `xa`;
/// 3. `x = x + mlp2(gelu(mlp1(mlp_ln(x))))` — the position-wise MLP, hidden
///    width `4 * n_state`, exact (`approx="none"`) GELU.
///
/// `cross_attn` / `cross_attn_ln` are `Some` only for decoder blocks
/// (`cross_attention=True`); the encoder builds the block without them.
#[derive(Debug)]
pub(crate) struct ResidualAttentionBlock {
  attn: MultiHeadAttention,
  attn_ln: LayerNorm,
  /// Cross-attention over the encoder states — `Some` for decoder blocks only.
  cross_attn: Option<MultiHeadAttention>,
  /// LayerNorm before cross-attention — present iff [`Self::cross_attn`] is.
  cross_attn_ln: Option<LayerNorm>,
  /// First MLP projection `n_state -> 4 * n_state`.
  mlp1: Linear,
  /// Second MLP projection `4 * n_state -> n_state`.
  mlp2: Linear,
  /// LayerNorm before the MLP.
  mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
  /// Construct from the already-built sub-modules. `cross` is
  /// `Some((cross_attn, cross_attn_ln))` for a decoder block
  /// (`cross_attention=True`), `None` for an encoder block.
  pub(crate) fn new(
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross: Option<(MultiHeadAttention, LayerNorm)>,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
  ) -> Self {
    let (cross_attn, cross_attn_ln) = match cross {
      Some((ca, cln)) => (Some(ca), Some(cln)),
      None => (None, None),
    };
    Self {
      attn,
      attn_ln,
      cross_attn,
      cross_attn_ln,
      mlp1,
      mlp2,
      mlp_ln,
    }
  }

  /// `true` for a decoder block (carries cross-attention), `false` for an
  /// encoder block.
  #[cfg(test)]
  pub(crate) fn has_cross_attention(&self) -> bool {
    self.cross_attn.is_some()
  }

  /// Run the block. Faithful port of `ResidualAttentionBlock.__call__`
  /// (`whisper.py:395-406`).
  ///
  /// - `x`: the input `(B, T, n_state)`.
  /// - `xa`: the encoder states `(B, T_kv, n_state)` for cross-attention; `None`
  ///   for an encoder block (or a decoder block whose cross-attn is absent).
  /// - `mask`: the additive causal mask for self-attention (`None` in the
  ///   encoder).
  /// - `kv_cache`: the incoming `(self_kv, cross_kv)` cache; `None` on the first
  ///   step.
  ///
  /// Returns `(x, (self_kv, cross_kv))` — the block output and the updated
  /// cache to thread into the next decode step. The reference's `cross_qk`
  /// (the cross-attention weights, for word-timing DTW) is dropped on this
  /// path; [`Self::forward_with_cross_qk`] is the variant that surfaces it.
  ///
  /// This is a true no-`qk` path: the cross-attention runs through the plain
  /// [`MultiHeadAttention::forward`], which drops the `(B, H, T_q, T_kv)`
  /// attention-score tensor INSIDE the attention (before the residual / MLP),
  /// rather than materializing it and holding it live across the MLP the way
  /// [`Self::forward_with_cross_qk`] must. So an ordinary decode (which never
  /// reads the weights) does not raise peak memory by the score tensor.
  ///
  /// # Errors
  /// Propagates the LayerNorm / attention / MLP op errors.
  pub(crate) fn forward(
    &self,
    x: &Array,
    xa: Option<&Array>,
    mask: Option<&Array>,
    kv_cache: Option<&BlockKvCache>,
  ) -> Result<(Array, BlockKvCache)> {
    let (self_kv_in, cross_kv_in) = match kv_cache {
      Some((s, c)) => (s.as_ref(), c.as_ref()),
      None => (None, None),
    };

    // 1. Self-attention: `x = x + attn(attn_ln(x), mask=mask, kv_cache=kv)`.
    let normed = self.attn_ln.forward(x)?;
    let (attn_out, self_kv) = self.attn.forward(&normed, None, mask, self_kv_in)?;
    let mut x = x.add(&attn_out)?;

    // 2. Cross-attention (decoder blocks only) — the plain `forward`, so the
    //    score tensor is freed inside the attention, never held across the MLP.
    let cross_kv = match (&self.cross_attn, &self.cross_attn_ln) {
      (Some(cross_attn), Some(cross_attn_ln)) => {
        let normed = cross_attn_ln.forward(&x)?;
        let (cross_out, cross_kv) = cross_attn.forward(&normed, xa, None, cross_kv_in)?;
        x = x.add(&cross_out)?;
        Some(cross_kv)
      }
      // No cross-attention (encoder block): preserve any incoming cross cache
      // untouched (in practice always `None` for the encoder).
      _ => match cross_kv_in {
        Some((ck, cv)) => Some((ck.try_clone()?, cv.try_clone()?)),
        None => None,
      },
    };

    // 3. MLP: `x = x + mlp2(gelu(mlp1(mlp_ln(x))))`.
    let mlp_in = self.mlp_ln.forward(&x)?;
    let hidden = gelu(&self.mlp1.forward(&mlp_in)?)?;
    let mlp_out = self.mlp2.forward(&hidden)?;
    let x = x.add(&mlp_out)?;

    Ok((x, (Some(self_kv), cross_kv)))
  }

  /// Run the block and also surface the cross-attention weights `cross_qk` —
  /// the full three-tuple return of `ResidualAttentionBlock.__call__`
  /// (`whisper.py:395-406`), where the self-attention `qk` is dropped (`y, kv,
  /// _ = self.attn(...)`) and the cross-attention `qk` is returned.
  ///
  /// Returns `(x, (self_kv, cross_kv), cross_qk)`. `cross_qk` is `Some(qk)`
  /// `(B, H, T_q, T_kv)` for a decoder block (one carrying cross-attention),
  /// and `None` for an encoder block (no cross-attention) — mirroring the
  /// reference's `cross_qk = None` default that stays `None` unless the block
  /// has a `cross_attn`. Only the cross-attention `qk` is surfaced because the
  /// word-timestamp DTW aligns text positions to audio frames via the cross-
  /// attention pattern (the self-attention weights are not part of that).
  ///
  /// # Errors
  /// Propagates the LayerNorm / attention / MLP op errors.
  pub(crate) fn forward_with_cross_qk(
    &self,
    x: &Array,
    xa: Option<&Array>,
    mask: Option<&Array>,
    kv_cache: Option<&BlockKvCache>,
  ) -> Result<(Array, BlockKvCache, Option<Array>)> {
    let (self_kv_in, cross_kv_in) = match kv_cache {
      Some((s, c)) => (s.as_ref(), c.as_ref()),
      None => (None, None),
    };

    // 1. Self-attention: `x = x + attn(attn_ln(x), mask=mask, kv_cache=kv)`.
    //    The self-attention `qk` is dropped (`_` in the reference).
    let normed = self.attn_ln.forward(x)?;
    let (attn_out, self_kv) = self.attn.forward(&normed, None, mask, self_kv_in)?;
    let mut x = x.add(&attn_out)?;

    // 2. Cross-attention (decoder blocks only): `x = x + cross_attn(
    //    cross_attn_ln(x), xa, kv_cache=cross_kv)`. Surface its `qk` weights.
    let (cross_kv, cross_qk) = match (&self.cross_attn, &self.cross_attn_ln) {
      (Some(cross_attn), Some(cross_attn_ln)) => {
        let normed = cross_attn_ln.forward(&x)?;
        let (cross_out, cross_kv, cross_qk) =
          cross_attn.forward_with_qk(&normed, xa, None, cross_kv_in)?;
        x = x.add(&cross_out)?;
        (Some(cross_kv), Some(cross_qk))
      }
      // No cross-attention (encoder block): preserve any incoming cross cache
      // untouched (in practice always `None` for the encoder), and report no
      // cross-attention weights.
      _ => {
        let cross_kv = match cross_kv_in {
          Some((ck, cv)) => Some((ck.try_clone()?, cv.try_clone()?)),
          None => None,
        };
        (cross_kv, None)
      }
    };

    // 3. MLP: `x = x + mlp2(gelu(mlp1(mlp_ln(x))))`.
    let mlp_in = self.mlp_ln.forward(&x)?;
    let hidden = gelu(&self.mlp1.forward(&mlp_in)?)?;
    let mlp_out = self.mlp2.forward(&hidden)?;
    let x = x.add(&mlp_out)?;

    Ok((x, (Some(self_kv), cross_kv), cross_qk))
  }
}

/// Whisper sinusoidal positional embedding (`whisper.py:319-325`).
///
/// `inv_timescales = exp(-log(max_timescale) / (channels/2 - 1) *
/// arange(channels/2))`; `scaled_time = arange(length)[:, None] *
/// inv_timescales[None, :]`; returns `concat([sin(scaled_time),
/// cos(scaled_time)], axis=1)` of shape `(length, channels)`.
///
/// `channels` must be even (the reference asserts `channels % 2 == 0`).
/// `max_timescale` is `10000` in Whisper.
///
/// # Errors
/// - [`crate::Error::InvariantViolation`] if `channels` is odd or `< 2`, or
///   `length == 0`;
/// - propagates the arange / exp / matmul / concat op errors.
pub(crate) fn sinusoids(length: usize, channels: usize, max_timescale: f64) -> Result<Array> {
  use crate::error::{Error, InvariantViolationPayload};
  if channels < 2 || !channels.is_multiple_of(2) {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "sinusoids: channels",
      "must be even and >= 2",
    )));
  }
  if length == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "sinusoids: length",
      "must be > 0",
    )));
  }
  let half = channels / 2;
  let length_i32 = length as i32;
  let half_i32 = half as i32;

  // log_timescale_increment = log(max_timescale) / (half - 1).
  let log_inc = max_timescale.ln() / (half as f64 - 1.0);
  // inv_timescales = exp(-log_inc * arange(half)) — shape (half,).
  let ar = Array::arange::<f32>(0.0, half as f64, 1.0)?;
  let neg_inc = Array::full::<f32>(&[0i32; 0], (-log_inc) as f32)?;
  let inv_timescales = ar.multiply(&neg_inc)?.exp()?;

  // scaled_time = arange(length)[:, None] * inv_timescales[None, :] — outer
  // product (length, half). Build via reshape + broadcast:
  // (length, 1) * (1, half).
  let t = Array::arange::<f32>(0.0, length as f64, 1.0)?.reshape(&[length_i32, 1])?;
  let inv_row = inv_timescales.reshape(&[1, half_i32])?;
  let scaled_time = t.multiply(&inv_row)?;

  // concat([sin, cos], axis=1) -> (length, channels).
  let s = scaled_time.sin()?;
  let c = scaled_time.cos()?;
  concatenate(&[&s, &c], 1)
}

#[cfg(test)]
mod tests;

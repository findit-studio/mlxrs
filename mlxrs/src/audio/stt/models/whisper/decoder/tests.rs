use super::*;
use crate::{
  Dtype,
  audio::stt::models::whisper::layers::{Linear, MultiHeadAttention},
};

fn to_vec(a: &Array) -> Vec<f32> {
  a.try_clone().unwrap().to_vec::<f32>().unwrap()
}

fn arr(buf: &[f32], shape: &[i32]) -> Array {
  Array::from_slice::<f32>(buf, &shape.to_vec()).unwrap()
}

/// LayerNorm with the `nn.LayerNorm` default affine (`ones`/`zeros`), eps 1e-5.
fn layer_norm(n: usize) -> LayerNorm {
  LayerNorm::new(
    Some(Array::ones::<f32>(&(n,)).unwrap()),
    Some(Array::zeros::<f32>(&(n,)).unwrap()),
    1e-5,
  )
}

/// Identity `Linear` (`weight = I`, no bias).
fn identity_linear(n: usize) -> Linear {
  let mut w = vec![0.0_f32; n * n];
  for i in 0..n {
    w[i * n + i] = 1.0;
  }
  Linear::new(arr(&w, &[n as i32, n as i32]), None)
}

/// Zero `Linear` of shape `(out, in)` (forward yields zeros).
fn zero_linear(out: usize, inp: usize) -> Linear {
  Linear::new(
    Array::zeros::<f32>(&[out as i32, inp as i32]).unwrap(),
    None,
  )
}

/// A `cross_attention=True` decoder block whose attention AND MLP contributions
/// are all zero (the `out` projections and `mlp2` are zero), so the block is a
/// pure passthrough `x -> x`. Isolates the embedding / positional / final-norm
/// / logit wiring around the blocks.
fn transparent_decoder_block(n_state: usize, n_head: usize) -> ResidualAttentionBlock {
  let self_attn = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    zero_linear(n_state, n_state), // out = 0 ⇒ self-attn residual adds 0
  );
  let cross_attn = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    zero_linear(n_state, n_state), // out = 0 ⇒ cross-attn residual adds 0
  );
  ResidualAttentionBlock::new(
    self_attn,
    layer_norm(n_state),
    Some((cross_attn, layer_norm(n_state))),
    zero_linear(4 * n_state, n_state), // mlp1
    zero_linear(n_state, 4 * n_state), // mlp2 = 0 ⇒ MLP residual adds 0
    layer_norm(n_state),
  )
}

/// A real (non-zero) cross-attention decoder block with identity projections,
/// for exercising the cross-attention path end-to-end.
fn identity_decoder_block(n_state: usize, n_head: usize) -> ResidualAttentionBlock {
  let self_attn = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  let cross_attn = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  // Zero MLP (mlp2 = 0) to keep the test reference tractable; the attention
  // paths are the focus.
  ResidualAttentionBlock::new(
    self_attn,
    layer_norm(n_state),
    Some((cross_attn, layer_norm(n_state))),
    zero_linear(4 * n_state, n_state),
    zero_linear(n_state, 4 * n_state),
    layer_norm(n_state),
  )
}

/// Build a decoder. `n_vocab` rows of token embedding, `n_ctx` rows of learned
/// positional embedding (deterministic values), `blocks`, identity final norm.
fn build_decoder(
  n_vocab: usize,
  n_state: usize,
  n_ctx: usize,
  blocks: Vec<ResidualAttentionBlock>,
) -> (TextDecoder, Array, Array) {
  let tok_buf: Vec<f32> = (0..n_vocab * n_state).map(|i| 0.1 * i as f32).collect();
  let tok_w = arr(&tok_buf, &[n_vocab as i32, n_state as i32]);
  let pe_buf: Vec<f32> = (0..n_ctx * n_state)
    .map(|i| -0.05 * i as f32 + 0.2)
    .collect();
  let pe = arr(&pe_buf, &[n_ctx as i32, n_state as i32]);
  let dec = TextDecoder::new(
    Embedding::new(tok_w.try_clone().unwrap()),
    pe.try_clone().unwrap(),
    blocks,
    layer_norm(n_state),
    n_ctx,
    n_vocab,
    Dtype::F32,
  )
  .unwrap();
  (dec, tok_w, pe)
}

/// Oracle for the embedding + positional-slice + final-norm + weight-tied logit
/// wiring (offset 0, transparent blocks): the logits equal
/// `as_linear(ln(token_embedding(tokens) + positional_embedding[0:T]))`
/// recomputed independently from the public ops.
#[test]
fn decoder_embedding_positional_logits_oracle() {
  let n_vocab = 5usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 8usize;
  let blocks = vec![transparent_decoder_block(n_state, n_head)];
  let (dec, tok_w, pe) = build_decoder(n_vocab, n_state, n_ctx, blocks);

  // tokens (B=1, T=3) — passed as a `&[u32]` slice; the decoder builds the
  // `(1, T)` array internally.
  let tokens = [2u32, 0, 4];
  // encoder states (B=1, T_kv=2, n_state).
  let xa = arr(
    &[0.3, -0.1, 0.5, 0.2, 0.7, -0.4, 0.1, 0.9],
    &[1, 2, n_state as i32],
  );

  let (logits, cache) = dec.forward(&tokens, &xa, None).unwrap();
  assert_eq!(
    logits.shape(),
    vec![1, 3, n_vocab],
    "weight-tied logits shape"
  );
  assert_eq!(cache.len(), 1, "one cache entry per block");

  // Independent reference: token_embedding(tokens) + positional_embedding[0:3],
  // through a transparent block (x unchanged), final ln, weight-tied logits.
  // The reference embedding gathers from the `(1, T)` token array built here.
  let tokens_arr = Array::from_slice::<u32>(&tokens, &[1, 3]).unwrap();
  let emb = Embedding::new(tok_w);
  let token_emb = emb.forward(&tokens_arr).unwrap();
  let pe_slice = slice(&pe, &[0, 0], &[3, n_state as i32], &[1, 1]).unwrap();
  let summed = token_emb.add(&pe_slice).unwrap();
  let normed = layer_norm(n_state).forward(&summed).unwrap();
  let want = to_vec(&emb.as_linear(&normed).unwrap());

  let got = to_vec(&logits);
  assert_eq!(got.len(), want.len());
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < 1e-5, "logit[{i}] = {g} (ref {w})");
  }
}

/// The positional embedding is sliced by the KV-cache OFFSET. Build a cache
/// whose self-attention key time dim is `offset`, decode a single token, and
/// assert the resulting hidden uses `positional_embedding[offset:offset+1]`
/// (oracle via the transparent-block reference at the shifted slice).
#[test]
fn decoder_positional_slice_uses_cache_offset() {
  let n_vocab = 6usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 16usize;
  let blocks = vec![transparent_decoder_block(n_state, n_head)];
  let (dec, tok_w, pe) = build_decoder(n_vocab, n_state, n_ctx, blocks);

  // Fabricate a cache with a self-attention K/V of time length `offset = 5`.
  let offset = 5usize;
  let ck = Array::zeros::<f32>(&[1, offset as i32, n_state as i32]).unwrap();
  let cv = Array::zeros::<f32>(&[1, offset as i32, n_state as i32]).unwrap();
  // Cross cache present (the encoder K/V) — also zeros, transparent block.
  let cache: DecoderKvCache = vec![(Some((ck, cv)), None)];

  // One new token at absolute position `offset`.
  let tokens = [3u32];
  let xa = arr(&[0.1, 0.2, 0.3, 0.4], &[1, 1, n_state as i32]);

  let (logits, new_cache) = dec.forward(&tokens, &xa, Some(&cache)).unwrap();
  assert_eq!(logits.shape(), vec![1, 1, n_vocab]);
  // The self-attention cache grew by one step (offset + 1 = 6).
  let (self_kv, _) = &new_cache[0];
  let (k, _) = self_kv.as_ref().unwrap();
  assert_eq!(
    k.shape(),
    vec![1, offset + 1, n_state],
    "self-attn cache grew"
  );

  // Oracle: hidden = ln(token_embedding(token) + positional_embedding[5:6]).
  let tokens_arr = Array::from_slice::<u32>(&tokens, &[1, 1]).unwrap();
  let emb = Embedding::new(tok_w);
  let token_emb = emb.forward(&tokens_arr).unwrap();
  let pe_slice = slice(
    &pe,
    &[offset as i32, 0],
    &[(offset + 1) as i32, n_state as i32],
    &[1, 1],
  )
  .unwrap();
  let summed = token_emb.add(&pe_slice).unwrap();
  let normed = layer_norm(n_state).forward(&summed).unwrap();
  let want = to_vec(&emb.as_linear(&normed).unwrap());
  let got = to_vec(&logits);
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < 1e-5, "offset logit[{i}] = {g} (ref {w})");
  }
}

/// The cross-attention path is actually exercised: a decoder built with a real
/// (identity-projection) cross-attention block produces a DIFFERENT result for
/// two different encoder states `xa` (so the decoder genuinely attends to the
/// audio features). With a transparent block the output would be independent of
/// `xa`; this guards that cross-attention is wired in.
#[test]
fn decoder_cross_attention_depends_on_encoder_states() {
  let n_vocab = 5usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 8usize;
  let blocks = vec![identity_decoder_block(n_state, n_head)];
  let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, blocks);

  let tokens = [1u32, 2];
  let xa1 = arr(
    &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    &[1, 2, n_state as i32],
  );
  let xa2 = arr(
    &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    &[1, 2, n_state as i32],
  );

  let (logits1, _) = dec.forward(&tokens, &xa1, None).unwrap();
  let (logits2, _) = dec.forward(&tokens, &xa2, None).unwrap();
  let v1 = to_vec(&logits1);
  let v2 = to_vec(&logits2);
  let max_diff = v1
    .iter()
    .zip(v2.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max);
  assert!(
    max_diff > 1e-4,
    "cross-attention must make the logits depend on the encoder states (max diff {max_diff})"
  );
}

/// `cache_offset` is 0 for a fresh (`None`) cache, and reads the first block's
/// self-attention key time dim otherwise. The number of blocks equals the
/// number of cache entries returned.
#[test]
fn decoder_block_count_and_cache_shape() {
  let n_vocab = 4usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 8usize;
  let blocks = vec![
    transparent_decoder_block(n_state, n_head),
    transparent_decoder_block(n_state, n_head),
    transparent_decoder_block(n_state, n_head),
  ];
  let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, blocks);
  assert_eq!(dec.num_blocks(), 3);

  let tokens = [0u32, 1];
  let xa = arr(&[0.1, 0.2, 0.3, 0.4], &[1, 1, n_state as i32]);
  let (_, cache) = dec.forward(&tokens, &xa, None).unwrap();
  assert_eq!(cache.len(), 3, "one cache entry per block");
  // Each entry's self-attention cache has time dim 2 (the prompt length).
  for (self_kv, cross_kv) in &cache {
    let (k, _) = self_kv.as_ref().unwrap();
    assert_eq!(k.shape(), vec![1, 2, n_state]);
    // Cross cache populated on the first call (the encoder K/V).
    assert!(cross_kv.is_some(), "cross-attn cache populated");
  }
}

/// The precomputed additive causal mask is `(n_ctx, n_ctx)`, with `0` on/below
/// the diagonal and `-inf` strictly above it.
#[test]
fn decoder_causal_mask_is_upper_triangular_neg_inf() {
  let n_vocab = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize;
  let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, Vec::new());
  let mask = dec.mask_ref();
  assert_eq!(mask.shape(), vec![n_ctx, n_ctx]);
  let m = to_vec(mask);
  for i in 0..n_ctx {
    for j in 0..n_ctx {
      let v = m[i * n_ctx + j];
      if j > i {
        assert!(v == f32::NEG_INFINITY, "mask[{i}][{j}] = {v} (want -inf)");
      } else {
        assert!(v == 0.0, "mask[{i}][{j}] = {v} (want 0)");
      }
    }
  }
}

/// The positional slice rejects an `offset + T` beyond `n_text_ctx`.
#[test]
fn decoder_rejects_position_past_n_ctx() {
  let n_vocab = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize;
  let blocks = vec![transparent_decoder_block(n_state, 2)];
  let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, blocks);
  // A 5-token prompt exceeds n_ctx = 4 at offset 0 (all ids < n_vocab, so the
  // context bound — not the value-range guard — is the rejection).
  let tokens = [0u32, 1, 2, 3, 0];
  let xa = arr(&[0.1, 0.2, 0.3, 0.4], &[1, 1, n_state as i32]);
  let err = dec
    .forward(&tokens, &xa, None)
    .expect_err("prompt longer than n_text_ctx must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

/// The decode-context bound is checked BEFORE the token-embedding gather: a
/// prefix whose `offset + T` exceeds `n_text_ctx` (here at a non-zero cache
/// offset) is rejected with a typed `OutOfRange`, so the `(1, T, n_state)`
/// embedding the decoder would otherwise materialize never allocates. With
/// `n_ctx = 4` and a fabricated cache of length 3, even a single new token
/// (absolute position 3 → end 4) is at the bound (accepted), while a 2-token
/// window (end 5) is rejected.
#[test]
fn decoder_context_checked_before_embedding_at_offset() {
  let n_vocab = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize;
  let blocks = vec![transparent_decoder_block(n_state, 2)];
  let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, blocks);
  let xa = arr(&[0.1, 0.2, 0.3, 0.4], &[1, 1, n_state as i32]);

  // Fabricate a self-attention cache of time length 3 (offset = 3).
  let ck = Array::zeros::<f32>(&[1, 3, n_state as i32]).unwrap();
  let cv = Array::zeros::<f32>(&[1, 3, n_state as i32]).unwrap();
  let cache: DecoderKvCache = vec![(Some((ck, cv)), None)];

  // A 2-token window at offset 3 → end 5 > n_ctx 4: rejected before the gather.
  let two = [0u32, 1];
  let err = dec
    .forward(&two, &xa, Some(&cache))
    .expect_err("offset + T past n_text_ctx must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");

  // A single token at offset 3 → end 4 == n_ctx: accepted.
  let one = [0u32];
  assert!(dec.forward(&one, &xa, Some(&cache)).is_ok());
}

/// Cross-attention weight extraction (`forward_with_cross_qk`): a multi-block
/// decoder returns one cross-attention `qk` per block, each shaped
/// `(B, n_head, T, T_kv)` (the per-head attention scores over the encoder
/// frames), and the logits are byte-identical to the plain `forward` (the
/// extraction is a pure add-on, the normal decode path is unchanged).
#[test]
fn decoder_forward_with_cross_qk_returns_per_layer_weights() {
  let n_vocab = 5usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 8usize;
  // Two real cross-attention blocks so the per-layer list has length 2.
  let blocks = vec![
    identity_decoder_block(n_state, n_head),
    identity_decoder_block(n_state, n_head),
  ];
  let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, blocks);

  // tokens (B=1, T=3); encoder states (B=1, T_kv=2, n_state).
  let t = 3usize;
  let t_kv = 2usize;
  let tokens = [2u32, 0, 4];
  let xa = arr(
    &[0.3, -0.1, 0.5, 0.2, 0.7, -0.4, 0.1, 0.9],
    &[1, t_kv as i32, n_state as i32],
  );

  let (logits, cache, cross_qk) = dec.forward_with_cross_qk(&tokens, &xa, None).unwrap();
  assert_eq!(logits.shape(), vec![1, t, n_vocab], "logits shape");
  assert_eq!(cache.len(), 2, "one cache entry per block");

  // One cross-qk per decoder block, each `Some` (every block has cross-attn),
  // shaped (B=1, n_head, T, T_kv).
  assert_eq!(cross_qk.len(), 2, "one cross-qk per decoder block");
  for (i, qk) in cross_qk.iter().enumerate() {
    let qk = qk
      .as_ref()
      .unwrap_or_else(|| panic!("block {i} cross-qk must be Some"));
    assert_eq!(
      qk.shape(),
      vec![1, n_head, t, t_kv],
      "block {i} cross-qk shape (B, H, T, T_kv)"
    );
  }

  // The logits match the plain `forward` exactly — extraction changes nothing
  // on the normal path.
  let (logits_plain, _) = dec.forward(&tokens, &xa, None).unwrap();
  let a = to_vec(&logits);
  let b = to_vec(&logits_plain);
  assert_eq!(
    a, b,
    "forward_with_cross_qk logits must equal forward logits"
  );
}

/// The decoder is the single lowest crate-visible gather chokepoint: calling
/// `forward` / `forward_with_cross_qk` DIRECTLY (bypassing `WhisperModel`) with
/// a token id `== n_vocab` or `> n_vocab` is rejected with a typed `OutOfRange`
/// — NOT a panic or an out-of-bounds row gather in the `(n_vocab, n_text_state)`
/// token-embedding table — while a fully in-range slice still forwards (logits
/// shape on both methods, cross-qk shape on the cross-qk variant).
///
/// Because the entry now takes `&[u32]`, the signed / negative / float and the
/// non-`(1, T)` rank/shape classes are COMPILE-TIME impossible (a `&[u32]` slice
/// cannot carry a negative, fractional, or wrong-rank token), so only the
/// value-range (`id < n_vocab`) class needs a runtime test — it is the sole
/// remaining gather hazard the structural fix closes at the root.
#[test]
fn decoder_forward_rejects_out_of_range_token_id() {
  let n_vocab = 5usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 8usize;
  let make = || {
    let blocks = vec![identity_decoder_block(n_state, n_head)];
    let (dec, _, _) = build_decoder(n_vocab, n_state, n_ctx, blocks);
    dec
  };
  let xa = arr(
    &[0.3, -0.1, 0.5, 0.2, 0.7, -0.4, 0.1, 0.9],
    &[1, 2, n_state as i32],
  );

  // `id == n_vocab` (the first out-of-range row): rejected on both entries.
  let dec = make();
  let at_bound = [0u32, n_vocab as u32, 1];
  let err = dec
    .forward(&at_bound, &xa, None)
    .expect_err("id == n_vocab must be rejected, not gathered out of bounds");
  assert!(matches!(err, Error::OutOfRange(_)), "forward: got {err:?}");
  let err = dec
    .forward_with_cross_qk(&at_bound, &xa, None)
    .expect_err("id == n_vocab must be rejected on the cross-qk entry too");
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "forward_with_cross_qk: got {err:?}"
  );

  // `id > n_vocab` (far past the table): same typed rejection on both entries.
  let past = [1u32, (n_vocab + 7) as u32];
  let err = dec
    .forward(&past, &xa, None)
    .expect_err("id > n_vocab must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "forward: got {err:?}");
  let err = dec
    .forward_with_cross_qk(&past, &xa, None)
    .expect_err("id > n_vocab must be rejected on the cross-qk entry too");
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "forward_with_cross_qk: got {err:?}"
  );

  // A fully in-range slice (every id `< n_vocab`) still forwards: the logits are
  // `(1, T, n_vocab)` on both methods, and the cross-qk variant additionally
  // returns one `(1, n_head, T, T_kv)` weight tensor per block.
  let valid = [4u32, 0, 2]; // T = 3, all ids in 0..n_vocab
  let t = valid.len();
  let t_kv = 2usize;
  let (logits, _cache) = dec
    .forward(&valid, &xa, None)
    .expect("an in-range token slice must forward");
  assert_eq!(logits.shape(), vec![1, t, n_vocab], "forward logits shape");
  let (logits, _cache, cross_qk) = dec
    .forward_with_cross_qk(&valid, &xa, None)
    .expect("an in-range token slice must forward on the cross-qk entry");
  assert_eq!(
    logits.shape(),
    vec![1, t, n_vocab],
    "forward_with_cross_qk logits shape"
  );
  assert_eq!(cross_qk.len(), 1, "one cross-qk per decoder block");
  let qk = cross_qk[0].as_ref().expect("block 0 cross-qk must be Some");
  assert_eq!(
    qk.shape(),
    vec![1, n_head, t, t_kv],
    "cross-qk shape (B, H, T, T_kv)"
  );
}

// ---- fp16 / bf16 activation-dtype preservation ----------------------
//
// The decoder embeds tokens + the learned positional table (both checkpoint
// tensors, cast to the model dtype at load) and adds the precomputed causal mask
// inside attention. The reference builds the mask cast to the model dtype
// (`self._mask = create_additive_causal_mask(n_ctx).astype(dtype)`,
// `whisper.py:460-462`); the port mirrors this (`create_additive_causal_mask(n,
// dtype)`). With the attention-scale fix (the scalar scale is cast to the
// activation dtype), a full f16/bf16 decoder forward stays in that dtype through
// the position add, the self-/cross-attention, and the weight-tied logits. The
// decoder's `forward` returns logits in the activation dtype — the f32 logit
// cast lives at the model layer (`Inference.logits`'s `.astype(mx.float32)`), not
// in the decoder — so this is the right level to pin activation-dtype
// preservation. These tests assert the mask dtype and a dtype-preserving forward.

/// An identity `(n, n)` projection (no bias) in `dtype`.
fn identity_linear_dtype(n: usize, dtype: Dtype) -> Linear {
  let mut w = vec![0.0_f32; n * n];
  for i in 0..n {
    w[i * n + i] = 1.0;
  }
  Linear::new(arr(&w, &[n as i32, n as i32]).astype(dtype).unwrap(), None)
}

/// A zero `(out, in)` projection in `dtype`.
fn zero_linear_dtype(out: usize, inp: usize, dtype: Dtype) -> Linear {
  Linear::new(
    Array::zeros::<f32>(&[out as i32, inp as i32])
      .unwrap()
      .astype(dtype)
      .unwrap(),
    None,
  )
}

/// A unit LayerNorm (`ones`/`zeros` affine) in `dtype`.
fn layer_norm_dtype(n: usize, dtype: Dtype) -> LayerNorm {
  LayerNorm::new(
    Some(Array::ones::<f32>(&(n,)).unwrap().astype(dtype).unwrap()),
    Some(Array::zeros::<f32>(&(n,)).unwrap().astype(dtype).unwrap()),
    1e-5,
  )
}

/// A cross-attention decoder block (identity attention, zero MLP) in `dtype`.
fn identity_decoder_block_dtype(
  n_state: usize,
  n_head: usize,
  dtype: Dtype,
) -> ResidualAttentionBlock {
  let self_attn = MultiHeadAttention::new(
    n_head,
    identity_linear_dtype(n_state, dtype),
    identity_linear_dtype(n_state, dtype),
    identity_linear_dtype(n_state, dtype),
    identity_linear_dtype(n_state, dtype),
  );
  let cross_attn = MultiHeadAttention::new(
    n_head,
    identity_linear_dtype(n_state, dtype),
    identity_linear_dtype(n_state, dtype),
    identity_linear_dtype(n_state, dtype),
    identity_linear_dtype(n_state, dtype),
  );
  ResidualAttentionBlock::new(
    self_attn,
    layer_norm_dtype(n_state, dtype),
    Some((cross_attn, layer_norm_dtype(n_state, dtype))),
    zero_linear_dtype(4 * n_state, n_state, dtype),
    zero_linear_dtype(n_state, 4 * n_state, dtype),
    layer_norm_dtype(n_state, dtype),
  )
}

/// Build a decoder whose every weight is in `dtype` (mirroring an f16/bf16
/// checkpoint, whose token embedding, positional table, and block weights are
/// all cast to the model dtype at load), constructing the causal mask in `dtype`.
fn build_decoder_dtype(
  n_vocab: usize,
  n_state: usize,
  n_ctx: usize,
  blocks: Vec<ResidualAttentionBlock>,
  dtype: Dtype,
) -> TextDecoder {
  let tok_buf: Vec<f32> = (0..n_vocab * n_state).map(|i| 0.1 * i as f32).collect();
  let tok_w = arr(&tok_buf, &[n_vocab as i32, n_state as i32])
    .astype(dtype)
    .unwrap();
  let pe_buf: Vec<f32> = (0..n_ctx * n_state)
    .map(|i| -0.05 * i as f32 + 0.2)
    .collect();
  let pe = arr(&pe_buf, &[n_ctx as i32, n_state as i32])
    .astype(dtype)
    .unwrap();
  TextDecoder::new(
    Embedding::new(tok_w),
    pe,
    blocks,
    layer_norm_dtype(n_state, dtype),
    n_ctx,
    n_vocab,
    dtype,
  )
  .unwrap()
}

/// The causal mask is built in the model dtype (f16), so the `qk + mask` add
/// inside attention is dtype-consistent and does not promote to f32.
#[test]
fn decoder_mask_is_model_dtype_f16() {
  decoder_mask_is_model_dtype(Dtype::F16);
}

/// Same for bf16.
#[test]
fn decoder_mask_is_model_dtype_bf16() {
  decoder_mask_is_model_dtype(Dtype::BF16);
}

fn decoder_mask_is_model_dtype(dtype: Dtype) {
  let dec = build_decoder_dtype(4, 4, 4, Vec::new(), dtype);
  assert_eq!(
    dec.mask_ref().dtype().unwrap(),
    dtype,
    "the causal mask must be built in the model dtype \
     (whisper.py:460-462 `create_additive_causal_mask(n_ctx).astype(dtype)`)"
  );
}

/// A full f16 decoder forward (token+positional embed → self/cross attention →
/// final norm → weight-tied logits) preserves the activation dtype: f16 in → f16
/// logits, no hidden promotion to f32 by the position add, the attention scale,
/// or the masked scores. (The model layer casts logits to f32 afterwards; the
/// decoder itself stays in the activation dtype.)
#[test]
fn decoder_forward_preserves_f16() {
  decoder_forward_preserves_dtype(Dtype::F16);
}

/// Same for bf16.
#[test]
fn decoder_forward_preserves_bf16() {
  decoder_forward_preserves_dtype(Dtype::BF16);
}

fn decoder_forward_preserves_dtype(dtype: Dtype) {
  let n_vocab = 6usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let n_ctx = 8usize;
  let blocks = vec![identity_decoder_block_dtype(n_state, n_head, dtype)];
  let dec = build_decoder_dtype(n_vocab, n_state, n_ctx, blocks, dtype);
  let tokens: Vec<u32> = vec![1, 3, 0];
  // Encoder states `(1, T_kv, n_state)` in the activation dtype.
  let xa = arr(&[0.5, -0.5, 1.0, -1.0, 0.2, 0.4, -0.2, -0.4], &[1, 2, 4])
    .astype(dtype)
    .unwrap();
  let (logits, _cache) = dec.forward(&tokens, &xa, None).unwrap();
  assert_eq!(logits.shape(), vec![1, tokens.len(), n_vocab]);
  assert_eq!(
    logits.dtype().unwrap(),
    dtype,
    "decoder logits must stay in the activation dtype before the model-layer \
     f32 cast (no promotion through the position add / attention scale / mask)"
  );
}

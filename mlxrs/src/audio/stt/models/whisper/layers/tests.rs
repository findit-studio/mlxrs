use super::*;
use crate::{Dtype, Error};

fn to_vec(a: &Array) -> Vec<f32> {
  a.try_clone().unwrap().to_vec::<f32>().unwrap()
}

fn arr(buf: &[f32], shape: &[i32]) -> Array {
  // `&[i32]` (the slice) impls `IntoShape`, but a re-borrow `&&[i32]` does
  // not; collect to a `Vec<i32>` (which impls `IntoShape`) so the helper can
  // take an unsized `&[i32]` parameter.
  Array::from_slice::<f32>(buf, &shape.to_vec()).unwrap()
}

// ---- Linear ---------------------------------------------------------

/// `Linear::forward` is `x @ Wᵀ + b`. Hand-computed:
/// W = [[1,2,3],[4,5,6]] (out=2, in=3), b = [10, 20], x = [1,1,1] (1×3).
/// y = [1*1+1*2+1*3 + 10, 1*4+1*5+1*6 + 20] = [6+10, 15+20] = [16, 35].
#[test]
fn linear_forward_with_bias_closed_form() {
  let w = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
  let b = arr(&[10.0, 20.0], &[2]);
  let lin = Linear::new(w, Some(b));
  let x = arr(&[1.0, 1.0, 1.0], &[1, 3]);
  let y = to_vec(&lin.forward(&x).unwrap());
  assert_eq!(y, vec![16.0, 35.0]);
}

/// `Linear::forward` with `bias=None` (Whisper's `key` projection) is `x @ Wᵀ`.
/// W = [[1,0],[0,2]] (out=2,in=2), x = [3,5] → y = [3*1+5*0, 3*0+5*2] = [3,10].
#[test]
fn linear_forward_no_bias() {
  let w = arr(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
  let lin = Linear::new(w, None);
  let x = arr(&[3.0, 5.0], &[1, 2]);
  let y = to_vec(&lin.forward(&x).unwrap());
  assert_eq!(y, vec![3.0, 10.0]);
}

// ---- Embedding ------------------------------------------------------

/// `Embedding::forward` gathers rows by id. weight = [[1,2],[3,4],[5,6]]
/// (n_vocab=3, n_state=2); ids = [2, 0] → rows [[5,6],[1,2]].
#[test]
fn embedding_forward_gathers_rows() {
  let w = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
  let emb = Embedding::new(w);
  let ids = Array::from_slice::<i32>(&[2, 0], &[2]).unwrap();
  let out = emb.forward(&ids).unwrap();
  assert_eq!(out.shape(), vec![2, 2]);
  assert_eq!(to_vec(&out), vec![5.0, 6.0, 1.0, 2.0]);
}

/// `Embedding::as_linear` is the weight-tied projection `x @ weightᵀ`.
/// weight = [[1,2],[3,4],[5,6]] (3×2); x = [1,1] (1×2) →
/// y = [1*1+1*2, 1*3+1*4, 1*5+1*6] = [3, 7, 11] (1×3, i.e. n_vocab logits).
#[test]
fn embedding_as_linear_weight_tied() {
  let w = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
  let emb = Embedding::new(w);
  let x = arr(&[1.0, 1.0], &[1, 2]);
  let y = to_vec(&emb.as_linear(&x).unwrap());
  assert_eq!(y, vec![3.0, 7.0, 11.0]);
}

// ---- QuantizedEmbedding construction validation ---------------------

/// The affine group size the quantized-embedding fixtures use — a valid mlx
/// group size (`mlx/ops.cpp:4740`) that divides the fixture's `n_state`.
const EMB_QGROUP: i32 = 64;
/// 8-bit affine — the `whisper-large-v3-turbo-8bit` embedding scheme.
const EMB_QBITS: i32 = 8;

/// A real affine `(packed uint32 weight, scales, biases)` embedding triple of
/// shape `(n_vocab, n_state)`, produced by the actual `ops::quantized::quantize`
/// — the exact on-disk layout an mlx-community quantized checkpoint ships for a
/// quantized `nn.Embedding`. `n_state` must be a whole multiple of `EMB_QGROUP`.
fn affine_embedding_triple(n_vocab: usize, n_state: usize) -> (Array, Array, Array) {
  let mut data = Vec::with_capacity(n_vocab * n_state);
  for v in 0..n_vocab {
    for s in 0..n_state {
      data.push(((v * 5 + s) as f32) * 0.001);
    }
  }
  let dense = Array::from_slice::<f32>(&data, &(n_vocab, n_state)).unwrap();
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, EMB_QGROUP, EMB_QBITS, "affine", None).unwrap();
  (
    w_q,
    scales,
    biases.expect("affine produces per-group biases"),
  )
}

/// A well-formed affine quantized embedding constructs (the positive boundary
/// for the new validation: a correct triple is NOT rejected) and runs both the
/// gather forward and the weight-tied `as_linear` projection to finite values.
#[test]
fn quantized_embedding_accepts_valid_affine_triple() {
  let (w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let emb = Embedding::quantized(w_q, scales, Some(biases), EMB_QGROUP, EMB_QBITS, "affine")
    .expect("a well-formed affine quantized embedding must construct");
  assert!(emb.is_quantized());
  // forward (dequantize-gather) and as_linear (quantized_matmul) both run.
  let ids = Array::from_slice::<i32>(&[0, 3], &[2]).unwrap();
  let rows = emb.forward(&ids).unwrap();
  assert_eq!(rows.shape(), vec![2, EMB_QGROUP as usize]);
  for v in to_vec(&rows) {
    assert!(v.is_finite(), "dequantized embedding row non-finite: {v}");
  }
  let x = Array::ones::<f32>(&(1usize, EMB_QGROUP as usize)).unwrap();
  let logits = emb.as_linear(&x).unwrap();
  assert_eq!(logits.shape(), vec![1, 8]);
  for v in to_vec(&logits) {
    assert!(v.is_finite(), "quantized logit non-finite: {v}");
  }
}

/// MISSING `.biases` for an `affine` quantized embedding is rejected at
/// construction (affine REQUIRES the per-group biases — `mlx.quantize` always
/// writes them) — a typed `InvariantViolation`, not a deferred mlx-c
/// `dequantize` failure on the first gather.
#[test]
fn quantized_embedding_rejects_affine_without_biases() {
  let (w_q, scales, _biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let err = Embedding::quantized(w_q, scales, None, EMB_QGROUP, EMB_QBITS, "affine").unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for affine embedding with missing .biases, got {err:?}"
  );
}

/// STALE `.biases` on a scale-only (`mxfp4` / fp) quantized embedding is
/// rejected — the fp modes are scale-only (`fp_quantize` writes no biases), so a
/// lingering `.biases` from a prior affine quantization is a malformed
/// checkpoint, caught at load rather than confusing the mlx-c `dequantize`.
#[test]
fn quantized_embedding_rejects_scale_only_with_stale_biases() {
  // The arrays' metadata is all that is read; reuse a real affine packed weight
  // / scales, and pass a stale `biases` under the scale-only `mxfp4` mode.
  let (w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let err =
    Embedding::quantized(w_q, scales, Some(biases), EMB_QGROUP, EMB_QBITS, "mxfp4").unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for scale-only embedding with stale .biases, got {err:?}"
  );
}

/// A `.biases` whose SHAPE differs from `.scales` is rejected — `affine_quantize`
/// writes `biases` and `scales` with the identical `(n_vocab, n_groups)` shape,
/// so a divergent `biases` shape is a malformed checkpoint (it would otherwise
/// reach the per-row `dequantize` with a mismatched gather).
#[test]
fn quantized_embedding_rejects_wrong_shaped_biases() {
  let (w_q, scales, _biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  // Biases of a DIFFERENT vocab (4 rows instead of 8): a real affine triple's
  // biases of the wrong leading dim, so the shape disagrees with `scales`.
  let (_w2, _s2, wrong_biases) = affine_embedding_triple(4, EMB_QGROUP as usize);
  let err = Embedding::quantized(
    w_q,
    scales,
    Some(wrong_biases),
    EMB_QGROUP,
    EMB_QBITS,
    "affine",
  )
  .unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_) | Error::RankMismatch(_)),
    "expected a shape/rank mismatch for a wrong-shaped .biases, got {err:?}"
  );
}

/// A `.scales` with a CORRECT rank (2) and leading dim (n_vocab) but a wrong
/// TRAILING (per-group) dim is rejected — the packed weight recovers `n_state =
/// packed * 32 / bits` while the scales recover `n_state = scales.shape(-1) *
/// group_size`, and the two must agree (mlx's invariant). Pairing the
/// `EMB_QGROUP`-grouped packed weight (scales `(n_vocab, 1)`) with group_size-32
/// scales (`(n_vocab, 2)`) under the declared `EMB_QGROUP` makes the scales
/// recover a different width than the weight — the constructor catches it at
/// load rather than deferring to a deep `dequantize` / `quantized_matmul`
/// failure on the first gather / logit projection.
#[test]
fn quantized_embedding_rejects_wrong_scales_trailing_dim() {
  let (w_q, _scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  // Re-quantize the SAME `(8, EMB_QGROUP)` dense at group_size 32 to get scales
  // with a different trailing per-group count (`(8, 2)`) but the matching
  // leading dim (8).
  let mut data = Vec::with_capacity(8 * EMB_QGROUP as usize);
  for v in 0..8usize {
    for s in 0..EMB_QGROUP as usize {
      data.push(((v * 5 + s) as f32) * 0.001);
    }
  }
  let dense = Array::from_slice::<f32>(&data, &(8usize, EMB_QGROUP as usize)).unwrap();
  let (_w32, wrong_scales, _b32) =
    crate::ops::quantized::quantize(&dense, 32, EMB_QBITS, "affine", None).unwrap();
  assert_eq!(
    wrong_scales.shape(),
    vec![8, 2],
    "fixture: group_size-32 scales"
  );
  let err = Embedding::quantized(
    w_q,
    wrong_scales,
    Some(biases),
    EMB_QGROUP,
    EMB_QBITS,
    "affine",
  )
  .unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for a wrong scales trailing dim, got {err:?}"
  );
}

/// A non-`uint32` packed weight (a dense `f32` table) is rejected — the
/// quantized embedding's `dequantize` / `quantized_matmul` require the
/// `uint32`-packed layout.
#[test]
fn quantized_embedding_rejects_non_u32_weight() {
  let dense = arr(&vec![0.001_f32; 8 * EMB_QGROUP as usize], &[8, EMB_QGROUP]);
  let (_w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let err = Embedding::quantized(
    dense, // f32, not uint32
    scales,
    Some(biases),
    EMB_QGROUP,
    EMB_QBITS,
    "affine",
  )
  .unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for a non-uint32 quantized embedding weight, got {err:?}"
  );
}

/// An unknown `mode` tag is rejected (a typo must not reach mlx-c).
#[test]
fn quantized_embedding_rejects_unknown_mode() {
  let (w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let err =
    Embedding::quantized(w_q, scales, Some(biases), EMB_QGROUP, EMB_QBITS, "garbage").unwrap_err();
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "expected UnknownEnumValue for an unrecognized embedding mode, got {err:?}"
  );
}

/// A non-positive `group_size` is rejected before it could divide / reach mlx-c.
#[test]
fn quantized_embedding_rejects_zero_group_size() {
  let (w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let err = Embedding::quantized(w_q, scales, Some(biases), 0, EMB_QBITS, "affine").unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for a zero group_size, got {err:?}"
  );
}

/// The affine scale/bias dtype rule (mlx's
/// `issubdtype(result_type(scales, biases), floating)`) is deferred to mlx-c at
/// op-time, so construction does NOT validate it: an affine embedding triple
/// with INTEGER `scales` and FLOATING `biases` constructs OK. (mlx ITSELF
/// accepts this triple — the pair promotes to floating.) The cast changes only
/// the scales dtype.
#[test]
fn quantized_embedding_accepts_affine_integer_scales_floating_biases() {
  let (w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let int_scales = scales.astype(Dtype::I32).unwrap();
  let got = Embedding::quantized(
    w_q,
    int_scales,
    Some(biases),
    EMB_QGROUP,
    EMB_QBITS,
    "affine",
  );
  assert!(
    got.is_ok(),
    "expected integer-scales + floating-biases affine embedding to construct, got {got:?}"
  );
}

/// The mirror case: FLOATING `scales` with INTEGER `biases`. Construction does
/// not validate the affine scale/bias dtype (it is deferred to mlx-c at
/// op-time), so this triple constructs OK. (mlx ITSELF accepts it — the pair
/// promotes to floating.)
#[test]
fn quantized_embedding_accepts_affine_floating_scales_integer_biases() {
  let (w_q, scales, biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  let int_biases = biases.astype(Dtype::I32).unwrap();
  let got = Embedding::quantized(
    w_q,
    scales,
    Some(int_biases),
    EMB_QGROUP,
    EMB_QBITS,
    "affine",
  );
  assert!(
    got.is_ok(),
    "expected floating-scales + integer-biases affine embedding to construct, got {got:?}"
  );
}

/// The `fp` modes require `scales.dtype() == uint8`. Reuse a real affine packed
/// weight + scales (gs=`EMB_QGROUP`, bits=`EMB_QBITS`) under `mxfp4` with NO
/// biases: the width identity still holds, so the triple is shape-correct, but
/// the affine scales are floating — the fp-mode uint8 rule must reject them at
/// construction rather than deferring to the first gather / logit projection.
#[test]
fn quantized_embedding_rejects_fp_mode_non_uint8_scales() {
  let (w_q, scales, _biases) = affine_embedding_triple(8, EMB_QGROUP as usize);
  assert!(
    scales.dtype().unwrap() != Dtype::U8,
    "fixture: affine scales are floating, not uint8"
  );
  let err = Embedding::quantized(w_q, scales, None, EMB_QGROUP, EMB_QBITS, "mxfp4").unwrap_err();
  assert!(
    matches!(err, Error::UnsupportedDtype(_)),
    "expected UnsupportedDtype for non-uint8 fp-mode embedding scales, got {err:?}"
  );
}

// ---- sinusoids ------------------------------------------------------

/// `sinusoids(length=4, channels=8)` closed form. Row 0 is
/// `[sin(0)*4, cos(0)*4] = [0,0,0,0, 1,1,1,1]`. Other rows from
/// `inv_timescales = [1, 0.046416, 0.002154, 0.0001]` (oracle, computed
/// independently from `exp(-ln(10000)/3 * j)`).
#[test]
fn sinusoids_closed_form_4x8() {
  let s = sinusoids(4, 8, 10000.0).unwrap();
  assert_eq!(s.shape(), vec![4, 8]);
  let v = to_vec(&s);
  // Row 0: sin part 0, cos part 1.
  for j in 0..4 {
    assert!(v[j].abs() < 1e-6, "sin row0[{j}] = {}", v[j]);
    assert!(
      (v[4 + j] - 1.0).abs() < 1e-6,
      "cos row0[{j}] = {}",
      v[4 + j]
    );
  }
  // Row 1, col 0: sin(1 * 1.0) = sin(1) ≈ 0.841471; cos col 0 ≈ 0.540302.
  let row1 = &v[8..16];
  assert!((row1[0] - 0.841471).abs() < 1e-5, "sin(1) = {}", row1[0]);
  assert!((row1[4] - 0.540302).abs() < 1e-5, "cos(1) = {}", row1[4]);
  // Row 1, col 1: sin(1 * 0.046416) ≈ 0.046399.
  assert!(
    (row1[1] - 0.046399).abs() < 1e-5,
    "sin(0.046416) = {}",
    row1[1]
  );
  // Row 2, col 0: sin(2) ≈ 0.909297; cos(2) ≈ -0.416147.
  let row2 = &v[16..24];
  assert!((row2[0] - 0.909297).abs() < 1e-5, "sin(2) = {}", row2[0]);
  assert!((row2[4] + 0.416147).abs() < 1e-5, "cos(2) = {}", row2[4]);
}

/// `sinusoids` rejects an odd channel count (the reference asserts even).
#[test]
fn sinusoids_rejects_odd_channels() {
  assert!(sinusoids(4, 7, 10000.0).is_err());
}

// ---- MultiHeadAttention ---------------------------------------------

/// Build an identity `Linear` (weight = I, no bias) of size `n×n`.
fn identity_linear(n: usize) -> Linear {
  let mut w = vec![0.0_f32; n * n];
  for i in 0..n {
    w[i * n + i] = 1.0;
  }
  Linear::new(arr(&w, &[n as i32, n as i32]), None)
}

/// Identity-projection MHA (`query=key=value=out=I`, no bias) on a SINGLE
/// query/key token (`T=1`) collapses to identity: softmax over a length-1
/// key axis is `1.0`, so `out = v = x` regardless of the head scaling or
/// mask. Closed-form: out == x exactly. n_state=4, n_head=2.
#[test]
fn mha_single_token_is_identity() {
  let n_state = 4usize;
  let mha = MultiHeadAttention::new(
    2,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  let x = arr(&[0.3, -0.7, 1.1, 2.0], &[1, 1, 4]); // (B=1, T=1, n_state=4)
  let (out, (k, v)) = mha.forward(&x, None, None, None).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 4]);
  // out == x (softmax over 1 element is identity; identity projections).
  let got = to_vec(&out);
  let want = to_vec(&x);
  for (g, w) in got.iter().zip(want.iter()) {
    assert!((g - w).abs() < 1e-6, "single-token MHA out {g} != x {w}");
  }
  // Returned (k, v) are the projected (here identity) k/v == x.
  assert_eq!(to_vec(&k), want);
  assert_eq!(to_vec(&v), want);
}

/// MHA output must match an INDEPENDENT recomputation of `qkv_attention`
/// (the reference equation `whisper.py:361-375`) built from primitive ops in
/// the test — pinning the **both-q-and-k `head_dim**-0.25` scale split** and
/// the precise softmax. Uses identity projections so q=k=v=x and the test
/// reference is purely the attention core. n_state=4, n_head=2, T=3.
#[test]
fn mha_matches_reference_qkv_equation() {
  let n_state = 4usize;
  let n_head = 2usize;
  let mha = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  // (B=1, T=3, n_state=4).
  let xb = [
    0.1, 0.2, 0.3, 0.4, // t0
    1.0, -1.0, 0.5, -0.5, // t1
    -0.3, 0.7, -0.2, 0.9, // t2
  ];
  let x = arr(&xb, &[1, 3, 4]);
  let (out, _) = mha.forward(&x, None, None, None).unwrap();

  // Independent reference: q=k=v=x; scale = (n_state/n_head)**-0.25 on q AND k.
  let nb = 1i32;
  let nc = 3i32;
  let nh = n_head as i32;
  let hd = (n_state / n_head) as i32;
  let scale = (n_state as f64 / n_head as f64).powf(-0.25) as f32;
  let sa = Array::full::<f32>(&[0i32; 0], scale).unwrap();
  let q = x
    .reshape(&[nb, nc, nh, hd])
    .unwrap()
    .transpose_axes(&[0, 2, 1, 3])
    .unwrap()
    .multiply(&sa)
    .unwrap();
  let k = x
    .reshape(&[nb, nc, nh, hd])
    .unwrap()
    .transpose_axes(&[0, 2, 3, 1])
    .unwrap()
    .multiply(&sa)
    .unwrap();
  let v = x
    .reshape(&[nb, nc, nh, hd])
    .unwrap()
    .transpose_axes(&[0, 2, 1, 3])
    .unwrap();
  let qk = q.matmul(&k).unwrap();
  let w = crate::ops::misc::softmax_axis(&qk, -1, true).unwrap();
  let ref_out = to_vec(
    &w.matmul(&v)
      .unwrap()
      .transpose_axes(&[0, 2, 1, 3])
      .unwrap()
      .reshape(&[nb, nc, n_state as i32])
      .unwrap(),
  );
  let got = to_vec(&out);
  assert_eq!(got.len(), ref_out.len());
  for (i, (g, e)) in got.iter().zip(ref_out.iter()).enumerate() {
    assert!((g - e).abs() < 1e-5, "mha[{i}] = {g} (ref {e})");
  }
}

/// Self-attention KV cache: passing an incoming `(k, v)` concatenates it with
/// the freshly-projected k/v along the time axis (axis 1), so the returned
/// kv time dimension grows by the cached length. With identity projections,
/// the new step's k == the step input, and the returned k == concat(cache_k,
/// step_k). n_state=4, n_head=2; cache T=2, step T=1 → returned kv T=3.
#[test]
fn mha_self_attention_kv_cache_concatenates() {
  let n_state = 4usize;
  let mha = MultiHeadAttention::new(
    2,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  // Prior cache: 2 tokens.
  let cache_k = arr(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, 2, 4]);
  let cache_v = arr(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0], &[1, 2, 4]);
  // New step: 1 token.
  let step = arr(&[0.0, 0.0, 1.0, 1.0], &[1, 1, 4]);
  let (out, (k, v)) = mha
    .forward(
      &step,
      None,
      None,
      Some(&(cache_k.try_clone().unwrap(), cache_v.try_clone().unwrap())),
    )
    .unwrap();
  // Output is for the single new query token.
  assert_eq!(out.shape(), vec![1, 1, 4]);
  // Returned k/v have time dim 3 (2 cached + 1 new).
  assert_eq!(k.shape(), vec![1, 3, 4]);
  assert_eq!(v.shape(), vec![1, 3, 4]);
  // Returned k = concat(cache_k, step) (identity key projection).
  let kk = to_vec(&k);
  assert_eq!(
    &kk[0..8],
    &to_vec(&cache_k)[..],
    "cached k prefix preserved"
  );
  assert_eq!(&kk[8..12], &[0.0, 0.0, 1.0, 1.0], "new k appended");
}

/// Cross-attention caching: the FIRST call (`xa = Some`, no cache) projects
/// k/v from the encoder states; a SECOND call passing the returned `(k, v)`
/// as the cache reuses them verbatim and produces the IDENTICAL output (the
/// encoder K/V never change across decode steps). n_state=4, n_head=2.
#[test]
fn mha_cross_attention_cache_reuse_is_identical() {
  let n_state = 4usize;
  let mha = MultiHeadAttention::new(
    2,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  // Decoder query (B=1, T=1, 4); encoder states (B=1, T=2, 4).
  let x = arr(&[0.5, -0.5, 0.2, 0.8], &[1, 1, 4]);
  let xa = arr(&[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4], &[1, 2, 4]);
  let (out1, kv) = mha.forward(&x, Some(&xa), None, None).unwrap();
  // Second step: reuse cached encoder k/v.
  let (out2, _) = mha.forward(&x, Some(&xa), None, Some(&kv)).unwrap();
  let v1 = to_vec(&out1);
  let v2 = to_vec(&out2);
  assert_eq!(v1.len(), v2.len());
  for (a, b) in v1.iter().zip(v2.iter()) {
    assert!(
      (a - b).abs() < 1e-6,
      "cross-attn cached output must match: {a} vs {b}"
    );
  }
}

/// The additive causal mask blocks future positions. With a 2-token sequence
/// and a mask `[[0, -inf], [0, 0]]`, the FIRST query attends only to key 0,
/// so (identity projections) `out[0] == v[0] == x[0]`. Pins that the mask is
/// applied additively to `qk` before softmax.
#[test]
fn mha_additive_causal_mask_blocks_future() {
  let n_state = 4usize;
  let mha = MultiHeadAttention::new(
    2,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  let x = arr(&[1.0, 2.0, 3.0, 4.0, -9.0, -9.0, -9.0, -9.0], &[1, 2, 4]);
  // Causal mask (2×2): row 0 cannot see col 1.
  let neg_inf = f32::NEG_INFINITY;
  let mask = arr(&[0.0, neg_inf, 0.0, 0.0], &[2, 2]);
  let (out, _) = mha.forward(&x, None, Some(&mask), None).unwrap();
  let got = to_vec(&out);
  // out[0] must equal x[0] = [1,2,3,4] (attends only to key 0).
  assert!(
    (got[0] - 1.0).abs() < 1e-5
      && (got[1] - 2.0).abs() < 1e-5
      && (got[2] - 3.0).abs() < 1e-5
      && (got[3] - 4.0).abs() < 1e-5,
    "masked out[0] = {:?} (want [1,2,3,4])",
    &got[0..4]
  );
}

/// Build a full additive causal mask `(n, n)`: `0` on/below the diagonal,
/// `-inf` strictly above it — the precomputed decoder mask shape the attention
/// slices offset-aware.
fn full_causal_mask(n: usize) -> Array {
  let mut m = vec![0.0_f32; n * n];
  for i in 0..n {
    for j in 0..n {
      if j > i {
        m[i * n + j] = f32::NEG_INFINITY;
      }
    }
  }
  arr(&m, &[n as i32, n as i32])
}

/// A WARM-CACHE MULTI-TOKEN self-attention step masks correctly: forwarding two
/// new tokens against a 2-token cache (offset 2, T_q 2, k_ctx 4) must produce the
/// SAME output as positions 2 and 3 of a single cold 4-token decode. This pins
/// the offset-aware mask slice `mask[offset : offset + T_q, 0 : offset + T_q]`:
/// with the old `mask[:T_q, :T_q]` slice the broadcast against `qk`'s
/// `(B, H, T_q, k_ctx)` would either fail (T_q != k_ctx) or mask as if the new
/// tokens started at absolute position 0. Identity projections (q=k=v=x) isolate
/// the attention core. n_state=4, n_head=2.
#[test]
fn mha_warm_cache_multi_token_masks_like_cold_decode() {
  let n_state = 4usize;
  let n_head = 2usize;
  let mha = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  // Four distinct token rows (B=1, T=4, n_state=4).
  let x_all = arr(
    &[
      0.10, 0.20, 0.30, 0.40, // pos 0
      1.00, -1.00, 0.50, -0.50, // pos 1
      -0.30, 0.70, -0.20, 0.90, // pos 2
      0.60, 0.10, -0.80, 0.25, // pos 3
    ],
    &[1, 4, n_state as i32],
  );
  let mask = full_causal_mask(4);

  // COLD: a single 4-token decode (offset 0). out[i] attends to keys 0..=i.
  let (cold_out, _) = mha.forward(&x_all, None, Some(&mask), None).unwrap();
  assert_eq!(cold_out.shape(), vec![1, 4, n_state]);
  let cold = to_vec(&cold_out);

  // WARM: first decode the 2-token prefix (offset 0) to populate the cache,
  // then forward the 2 NEW tokens (offset 2, T_q 2) against that cache.
  let x_prefix = arr(
    &[0.10, 0.20, 0.30, 0.40, 1.00, -1.00, 0.50, -0.50],
    &[1, 2, n_state as i32],
  );
  let (_, prefix_kv) = mha.forward(&x_prefix, None, Some(&mask), None).unwrap();
  let x_new = arr(
    &[-0.30, 0.70, -0.20, 0.90, 0.60, 0.10, -0.80, 0.25],
    &[1, 2, n_state as i32],
  );
  let (warm_out, (warm_k, _)) = mha
    .forward(&x_new, None, Some(&mask), Some(&prefix_kv))
    .unwrap();
  // The warm step queries 2 tokens but its key axis is the full 4 (2 cached + 2
  // new) — the offset-aware mask is what makes this broadcast and mask validly.
  assert_eq!(warm_out.shape(), vec![1, 2, n_state]);
  assert_eq!(warm_k.shape(), vec![1, 4, n_state], "cache grew to 4 keys");
  let warm = to_vec(&warm_out);

  // The warm 2-token step must equal positions 2 and 3 of the cold 4-token
  // decode (same absolute positions, same causal visibility).
  let cold_tail = &cold[2 * n_state..4 * n_state];
  assert_eq!(warm.len(), cold_tail.len());
  for (i, (w, c)) in warm.iter().zip(cold_tail.iter()).enumerate() {
    assert!(w.is_finite(), "warm[{i}] is non-finite ({w})");
    assert!(
      (w - c).abs() < 1e-5,
      "warm multi-token out[{i}] = {w} != cold tail {c}"
    );
  }

  // The mask is load-bearing: with NO mask, the cold decode's position 2 would
  // attend to ALL four keys (including the future position 3), so its output
  // must DIFFER from the causally-masked position 2 (= the warm step's first
  // token, asserted equal above). If the mask were a silent no-op these would
  // coincide and the test could not catch it.
  let (unmasked_out, _) = mha.forward(&x_all, None, None, None).unwrap();
  let unmasked = to_vec(&unmasked_out);
  let masked_pos2 = &cold[2 * n_state..3 * n_state];
  let unmasked_pos2 = &unmasked[2 * n_state..3 * n_state];
  let max_diff = masked_pos2
    .iter()
    .zip(unmasked_pos2.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max);
  assert!(
    max_diff > 1e-4,
    "causal mask must change position 2 (future-token visibility) — max diff {max_diff}"
  );
}

/// A warm-cache SINGLE-token step (the reference's hot path: offset N, T_q 1)
/// still masks correctly — the single new token sees ALL cached keys plus
/// itself. With the offset-aware slice `mask[offset : offset + 1, 0 : offset +
/// 1]` this is an all-zero row (every key at/before the new position is
/// visible), matching the old single-column broadcast. Pins the fix is a strict
/// superset of the previous (single-token-only) behavior. n_state=4, n_head=2.
#[test]
fn mha_warm_cache_single_token_unchanged() {
  let n_state = 4usize;
  let mha = MultiHeadAttention::new(
    2,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  let mask = full_causal_mask(8);
  // A 2-token cache, one new token (offset 2, T_q 1, k_ctx 3).
  let cache_k = arr(
    &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    &[1, 2, n_state as i32],
  );
  let cache_v = arr(
    &[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
    &[1, 2, n_state as i32],
  );
  let step = arr(&[0.0, 0.0, 1.0, 1.0], &[1, 1, n_state as i32]);
  let (with_mask, _) = mha
    .forward(
      &step,
      None,
      Some(&mask),
      Some(&(cache_k.try_clone().unwrap(), cache_v.try_clone().unwrap())),
    )
    .unwrap();
  // A single new token at the cache end is causally allowed to see every key, so
  // the mask contributes nothing: the masked result equals the UNMASKED one.
  let (no_mask, _) = mha
    .forward(&step, None, None, Some(&(cache_k, cache_v)))
    .unwrap();
  let a = to_vec(&with_mask);
  let b = to_vec(&no_mask);
  for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
    assert!(
      (x - y).abs() < 1e-6,
      "single-token warm-cache mask must be a no-op: [{i}] {x} vs {y}"
    );
  }
}

/// The normal-decode path ([`MultiHeadAttention::forward`]) runs the no-`qk`
/// attention core ([`MultiHeadAttention::qkv_attention_no_qk`]) — which never
/// returns the pre-softmax score tensor — yet produces output BYTE-IDENTICAL
/// to the `qk`-returning core reached through
/// [`MultiHeadAttention::forward_with_qk`]. Asserts the split is purely a
/// score-buffer-lifetime change with zero behavioral drift, on BOTH the
/// masked self-attention path (what `ResidualAttentionBlock::forward` drives)
/// and the cross-attention path. NON-identity projections so the recombined
/// output and the softmax mixing are nontrivial. n_state=4, n_head=2.
#[test]
fn mha_forward_no_qk_core_matches_qk_returning_core() {
  let n_state = 4usize;
  // Distinct, non-identity q/k/v/out projections (so `out` is a real mix, not
  // a pass-through that would mask a divergence between the two cores).
  let wq = arr(
    &[
      0.5, 0.1, 0.0, -0.2, 0.3, 0.4, 0.2, 0.0, 0.1, -0.3, 0.6, 0.2, 0.0, 0.2, -0.1, 0.7,
    ],
    &[4, 4],
  );
  let wk = arr(
    &[
      0.2, -0.1, 0.3, 0.0, 0.1, 0.5, -0.2, 0.4, 0.0, 0.3, 0.2, -0.1, 0.4, 0.0, 0.1, 0.3,
    ],
    &[4, 4],
  );
  let wv = arr(
    &[
      0.3, 0.0, -0.2, 0.1, 0.0, 0.4, 0.1, -0.3, 0.2, -0.1, 0.5, 0.0, -0.1, 0.2, 0.0, 0.6,
    ],
    &[4, 4],
  );
  let wo = arr(
    &[
      0.4, 0.1, -0.1, 0.0, 0.2, 0.3, 0.0, 0.1, -0.2, 0.0, 0.5, 0.1, 0.1, -0.2, 0.2, 0.4,
    ],
    &[4, 4],
  );
  let mha = MultiHeadAttention::new(
    2,
    Linear::new(wq, None),
    Linear::new(wk, None),
    Linear::new(wv, None),
    Linear::new(wo, None),
  );

  // Self-attention with a causal mask (the path normal decode reaches).
  let x = arr(
    &[
      0.1, -0.2, 0.3, 0.4, 1.0, -1.0, 0.5, -0.5, -0.3, 0.7, -0.2, 0.9,
    ],
    &[1, 3, n_state as i32],
  );
  let mask = full_causal_mask(3);
  let (out_plain, (k_plain, v_plain)) = mha.forward(&x, None, Some(&mask), None).unwrap();
  let (out_qk, (k_qk, v_qk), qk) = mha.forward_with_qk(&x, None, Some(&mask), None).unwrap();
  // The plain path's output is identical to the qk-returning path's output.
  assert_eq!(
    to_vec(&out_plain),
    to_vec(&out_qk),
    "no-qk self-attn output must equal the qk-returning output byte-for-byte"
  );
  // The returned (k, v) are identical too (same projection path).
  assert_eq!(to_vec(&k_plain), to_vec(&k_qk), "self-attn k identical");
  assert_eq!(to_vec(&v_plain), to_vec(&v_qk), "self-attn v identical");
  // The qk-returning path surfaces the (B, H, T, T_kv) score tensor; the plain
  // path's signature cannot return it (so it never escapes that path).
  assert_eq!(
    qk.shape(),
    vec![1, 2, 3, 3],
    "cross-qk score shape (B,H,T,T_kv)"
  );

  // Cross-attention path: same equivalence.
  let xa = arr(
    &[
      0.2, 0.4, -0.1, 0.3, 0.5, -0.2, 0.1, 0.0, -0.3, 0.6, 0.2, 0.1,
    ],
    &[1, 3, n_state as i32],
  );
  let xq = arr(&[0.7, -0.4, 0.2, 0.1], &[1, 1, n_state as i32]);
  let (cross_plain, _) = mha.forward(&xq, Some(&xa), None, None).unwrap();
  let (cross_qk, _, _) = mha.forward_with_qk(&xq, Some(&xa), None, None).unwrap();
  assert_eq!(
    to_vec(&cross_plain),
    to_vec(&cross_qk),
    "no-qk cross-attn output must equal the qk-returning output byte-for-byte"
  );
}

/// Introspection accessors compile and return the stored weights.
#[test]
fn linear_embedding_weight_accessors() {
  let lin = Linear::new(arr(&[1.0, 2.0], &[1, 2]), None);
  assert!(!lin.is_quantized());
  assert_eq!(
    lin
      .weight_ref()
      .expect("dense Linear has a weight")
      .dtype()
      .unwrap(),
    Dtype::F32
  );
  let emb = Embedding::new(arr(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
  assert!(!emb.is_quantized());
  assert_eq!(
    emb
      .weight_ref()
      .expect("dense Embedding has a weight")
      .shape(),
    vec![2, 2]
  );
}

// ---- ResidualAttentionBlock -----------------------------------------

/// A unit-affine LayerNorm of width `n` (weight=ones, bias=zeros, default eps).
fn unit_layer_norm(n: usize) -> LayerNorm {
  LayerNorm::new(
    Some(Array::ones::<f32>(&(n,)).unwrap()),
    Some(Array::zeros::<f32>(&(n,)).unwrap()),
    1e-5,
  )
}

/// Build a residual block of width `n` (all identity attention projections,
/// unit LayerNorms, identity-ish MLP). `cross` toggles the cross-attention.
fn block(n: usize, n_head: usize, cross: bool) -> ResidualAttentionBlock {
  let mha = || {
    MultiHeadAttention::new(
      n_head,
      identity_linear(n),
      identity_linear(n),
      identity_linear(n),
      identity_linear(n),
    )
  };
  let cross_pair = if cross {
    Some((mha(), unit_layer_norm(n)))
  } else {
    None
  };
  // mlp1: n -> 4n (zeros so the MLP contributes nothing), mlp2: 4n -> n.
  let mlp1 = Linear::new(Array::zeros::<f32>(&(4 * n, n)).unwrap(), None);
  let mlp2 = Linear::new(Array::zeros::<f32>(&(n, 4 * n)).unwrap(), None);
  ResidualAttentionBlock::new(
    mha(),
    unit_layer_norm(n),
    cross_pair,
    mlp1,
    mlp2,
    unit_layer_norm(n),
  )
}

#[test]
fn block_has_cross_attention_only_for_decoder() {
  // Encoder block: no cross-attention.
  assert!(!block(4, 2, false).has_cross_attention());
  // Decoder block: cross-attention present.
  assert!(block(4, 2, true).has_cross_attention());
}

#[test]
fn encoder_block_runs_self_attention_only() {
  // An encoder block (no cross) on a single token with identity attention +
  // zero MLP returns the same shape and threads no cross cache.
  let blk = block(4, 2, false);
  let x = arr(&[0.5, -0.5, 1.5, -1.5], &[1, 1, 4]);
  let (out, (self_kv, cross_kv)) = blk.forward(&x, None, None, None).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 4]);
  assert!(self_kv.is_some());
  assert!(
    cross_kv.is_none(),
    "encoder block must not produce cross KV"
  );
}

#[test]
fn decoder_block_runs_cross_attention() {
  // A decoder block (cross=true) attends over `xa`; the returned cache carries
  // both the self and cross KV pairs.
  let blk = block(4, 2, true);
  let x = arr(&[0.5, -0.5, 1.5, -1.5], &[1, 1, 4]);
  let xa = arr(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], &[1, 2, 4]);
  let (out, (self_kv, cross_kv)) = blk.forward(&x, Some(&xa), None, None).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 4]);
  assert!(self_kv.is_some());
  assert!(cross_kv.is_some(), "decoder block must produce cross KV");
}

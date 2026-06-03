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

/// A LayerNorm with `weight = ones(n)`, `bias = zeros(n)` (the `nn.LayerNorm`
/// default affine), eps = 1e-5 (the `nn.LayerNorm` default).
fn layer_norm(n: usize) -> LayerNorm {
  LayerNorm::new(
    Some(Array::ones::<f32>(&(n,)).unwrap()),
    Some(Array::zeros::<f32>(&(n,)).unwrap()),
    1e-5,
  )
}

/// Identity `Linear` (`weight = I`, no bias) of size `n`.
fn identity_linear(n: usize) -> Linear {
  let mut w = vec![0.0_f32; n * n];
  for i in 0..n {
    w[i * n + i] = 1.0;
  }
  Linear::new(arr(&w, &[n as i32, n as i32]), None)
}

/// A self-attention-only `ResidualAttentionBlock` (encoder variant) with
/// identity projections and identity (ones/zeros) LayerNorms.
fn identity_encoder_block(n_state: usize, n_head: usize) -> ResidualAttentionBlock {
  let mha = MultiHeadAttention::new(
    n_head,
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
    identity_linear(n_state),
  );
  // MLP: identity mlp1 then identity-ish mlp2 — but mlp1 must map n_state ->
  // 4*n_state. Use a zero mlp2 so the MLP contributes nothing (isolates the
  // attention + residual wiring). mlp1: (4n, n) zeros; mlp2: (n, 4n) zeros.
  let mlp1 = Linear::new(
    Array::zeros::<f32>(&[(4 * n_state) as i32, n_state as i32]).unwrap(),
    None,
  );
  let mlp2 = Linear::new(
    Array::zeros::<f32>(&[n_state as i32, (4 * n_state) as i32]).unwrap(),
    None,
  );
  ResidualAttentionBlock::new(
    mha,
    layer_norm(n_state),
    None,
    mlp1,
    mlp2,
    layer_norm(n_state),
  )
}

/// Build conv weights/biases for a `Conv1dLayer` that, with kernel 3 and
/// pad 1, computes a per-position passthrough-ish projection. We use a weight
/// that picks out the CENTER tap only (so conv == a per-position linear map),
/// making the conv output independently reproducible. Weight layout is
/// `(C_out, K=3, C_in)`; the center tap (k=1) is an identity `C_out×C_in`
/// block, the k=0 / k=2 taps are zero.
fn center_tap_identity_conv(c_out: usize, c_in: usize) -> (Array, Array) {
  let mut w = vec![0.0_f32; c_out * 3 * c_in];
  let min_c = c_out.min(c_in);
  // Center tap is k = 1 of the 3-tap kernel; its flat offset within a `(3,
  // C_in)` output-channel block is `1 * c_in == c_in`.
  let center_tap = c_in;
  for c in 0..min_c {
    // index (c_out=c, k=1, c_in=c): row-major over (C_out, 3, C_in).
    w[c * 3 * c_in + center_tap + c] = 1.0;
  }
  let weight = arr(&w, &[c_out as i32, 3, c_in as i32]);
  let bias = Array::zeros::<f32>(&(c_out,)).unwrap();
  (weight, bias)
}

/// Encoder downsample contract: a `(3000, n_mels)` mel yields a
/// `(1, 1500, n_state)` output — `conv2`'s stride 2 halves `3000 -> 1500`.
#[test]
fn encoder_downsamples_3000_to_1500() {
  let n_mels = 2usize;
  let n_state = 2usize;
  let n_ctx = 1500usize;
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(), // no blocks
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  // (3000, n_mels) mel.
  let mel = Array::zeros::<f32>(&[3000i32, n_mels as i32]).unwrap();
  let out = enc.forward(&mel).unwrap();
  assert_eq!(out.shape(), vec![1, n_ctx, n_state]);
}

/// The precomputed positional embedding equals `sinusoids(n_ctx, n_state)` and
/// has the `(n_ctx, n_state)` shape.
#[test]
fn encoder_positional_embedding_is_sinusoids() {
  let n_ctx = 6usize;
  let n_state = 4usize;
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_state);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  let pe = enc.positional_embedding_ref();
  assert_eq!(pe.shape(), vec![n_ctx, n_state]);
  let want = to_vec(&sinusoids(n_ctx, n_state, MAX_TIMESCALE).unwrap());
  let got = to_vec(pe);
  for (g, w) in got.iter().zip(want.iter()) {
    assert!((g - w).abs() < 1e-6, "PE {g} != sinusoids {w}");
  }
}

/// Oracle: with 0 blocks the encoder output equals an INDEPENDENT recompute
/// `ln_post(gelu(conv2(gelu(conv1(x)))) + sinusoids)` built from the public
/// conv / gelu / LayerNorm / sinusoids ops — pinning the exact wiring
/// (conv1 → gelu → conv2 → gelu → + PE → ln_post).
#[test]
fn encoder_zero_block_matches_reference_wiring() {
  // n_state >= 4 so `sinusoids` (`half = n_state/2`, denominator `half - 1`) is
  // well-defined — n_state = 2 gives `half - 1 = 0` (a degenerate division the
  // reference shares; real Whisper n_state is >= 384).
  let n_mels = 4usize;
  let n_state = 4usize;
  // Choose an input frame count F so conv2 (stride 2, pad 1, k 3) yields
  // n_ctx = F/2. F = 8 → post-conv frames = (8 + 2 - 2 - 1)/2 + 1 = 4.
  let frames = 8usize;
  let n_ctx = 4usize;

  // Non-trivial conv weights (not the center-tap identity) so the test
  // exercises the real conv arithmetic. Random-ish deterministic values.
  let c1_w: Vec<f32> = (0..n_state * 3 * n_mels)
    .map(|i| 0.1 * (i as f32) - 0.3)
    .collect();
  let c1_b: Vec<f32> = (0..n_state).map(|i| 0.05 * i as f32).collect();
  let c2_w: Vec<f32> = (0..n_state * 3 * n_state)
    .map(|i| -0.07 * (i as f32) + 0.2)
    .collect();
  let c2_b: Vec<f32> = (0..n_state).map(|i| -0.02 * i as f32).collect();

  let c1_weight = arr(&c1_w, &[n_state as i32, 3, n_mels as i32]);
  let c1_bias = arr(&c1_b, &[n_state as i32]);
  let c2_weight = arr(&c2_w, &[n_state as i32, 3, n_state as i32]);
  let c2_bias = arr(&c2_b, &[n_state as i32]);

  let ln = layer_norm(n_state);
  let enc = AudioEncoder::new(
    c1_weight.try_clone().unwrap(),
    c1_bias.try_clone().unwrap(),
    c2_weight.try_clone().unwrap(),
    c2_bias.try_clone().unwrap(),
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();

  // Input mel (frames, n_mels), deterministic.
  let mel_buf: Vec<f32> = (0..frames * n_mels)
    .map(|i| 0.3 * (i as f32).sin())
    .collect();
  let mel = arr(&mel_buf, &[frames as i32, n_mels as i32]);

  let got = to_vec(&enc.forward(&mel).unwrap());

  // Independent reference build via public ops.
  let x = mel.reshape(&[1, frames as i32, n_mels as i32]).unwrap();
  let y1 = gelu(
    &conv1d(&x, &c1_weight, 1, 1, 1, 1)
      .unwrap()
      .add(&c1_bias)
      .unwrap(),
  )
  .unwrap();
  let y2 = gelu(
    &conv1d(&y1, &c2_weight, 2, 1, 1, 1)
      .unwrap()
      .add(&c2_bias)
      .unwrap(),
  )
  .unwrap();
  assert_eq!(y2.shape(), vec![1, n_ctx, n_state], "post-conv shape");
  let pe = sinusoids(n_ctx, n_state, MAX_TIMESCALE).unwrap();
  let summed = y2.add(&pe).unwrap();
  let want = to_vec(&ln.forward(&summed).unwrap());

  assert_eq!(got.len(), want.len());
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    // Finiteness guard so a NaN == NaN can never masquerade as a pass.
    assert!(g.is_finite(), "encoder[{i}] is non-finite ({g})");
    assert!((g - w).abs() < 1e-5, "encoder[{i}] = {g} (ref {w})");
  }
}

/// The pre-conv frame guard: a mel whose frame count does not equal the
/// encoder's expected pre-downsample width (`conv2.stride * n_ctx`) is rejected
/// with a typed error BEFORE `conv1` runs (so an oversized input cannot drive
/// the conv1 activation to an out-of-memory abort ahead of the post-conv shape
/// check).
#[test]
fn encoder_rejects_wrong_frame_count() {
  let n_mels = 2usize;
  let n_state = 2usize;
  // Configure n_ctx = 1500 (expected pre-conv width = 2 * 1500 = 3000) but feed
  // only 100 frames → mismatch.
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    1500,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  let mel = Array::zeros::<f32>(&[100i32, n_mels as i32]).unwrap();
  let err = enc
    .forward(&mel)
    .expect_err("wrong frame count must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

/// An OVER-LARGE mel (far more frames than the encoder's expected pre-downsample
/// width) is rejected before `conv1` allocates — the resource guard that keeps a
/// hostile frame count from materializing an oversized conv activation. With
/// `n_ctx = 4` (expected width 8), a 4096-frame mel is rejected up front.
#[test]
fn encoder_rejects_oversized_frame_count_before_conv() {
  let n_mels = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize; // expected pre-conv width = conv2.stride(2) * 4 = 8
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  let mel = Array::zeros::<f32>(&[4096i32, n_mels as i32]).unwrap();
  let err = enc
    .forward(&mel)
    .expect_err("oversized frame count must be rejected before conv1");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

/// The batch guard: an already-3-D mel whose leading dimension is not `1` is
/// rejected BEFORE `conv1` allocates (Whisper encodes one 30 s segment at a
/// time), so an oversized batch cannot drive the conv1 activation to
/// `B * frames * n_state` past the model config's `N_FRAMES * n_audio_state`
/// cap. With `n_ctx = 4` (expected pre-conv width 8), a `(4, 8, n_mels)` mel —
/// a correct frame count but batch 4 — is rejected.
#[test]
fn encoder_rejects_oversized_batch_before_conv() {
  let n_mels = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize; // expected pre-conv width = conv2.stride(2) * 4 = 8
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  // (B=4, frames=8, n_mels) — valid frame axis, oversized batch.
  let mel = Array::zeros::<f32>(&[4i32, 8, n_mels as i32]).unwrap();
  let err = enc
    .forward(&mel)
    .expect_err("oversized batch must be rejected before conv1");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
  // The batch-1 form with the same frame count is accepted.
  let ok_mel = Array::zeros::<f32>(&[1i32, 8, n_mels as i32]).unwrap();
  assert!(enc.forward(&ok_mel).is_ok());
}

/// The mel-channel guard: a mel whose channel (last) axis differs from the
/// configured `n_mels` (the `conv1` input-channel dimension) is rejected with a
/// typed error BEFORE `conv1` contracts that axis — so a public caller cannot
/// drive the convolution with an unbounded, caller-controlled `C_in`. With
/// `n_mels = 4` (conv1 input channels 4) and `n_ctx = 4` (expected frame width
/// 8), a `(8, 7)` mel — correct frame count but 7 channels — is rejected, while
/// the matching `(8, 4)` mel is accepted.
#[test]
fn encoder_rejects_wrong_mel_channel_width() {
  let n_mels = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize; // expected pre-conv frame width = conv2.stride(2) * 4 = 8
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  // (frames = 8, mels = 7): valid frame axis, WRONG channel width (expected 4).
  let wrong = Array::zeros::<f32>(&[8i32, 7]).unwrap();
  let err = enc
    .forward(&wrong)
    .expect_err("wrong mel channel width must be rejected before conv1");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
  // The matching channel width (n_mels = 4) with the same frame count is
  // accepted.
  let ok_mel = Array::zeros::<f32>(&[8i32, n_mels as i32]).unwrap();
  assert!(enc.forward(&ok_mel).is_ok());
}

/// The mel-channel guard also fires for an already-3-D `(1, frames, C)` input
/// whose channel axis is wrong: a `(1, 8, 9)` mel (batch 1, correct frames,
/// 9 channels vs the configured 4) is rejected before `conv1`.
#[test]
fn encoder_rejects_wrong_mel_channel_width_batched() {
  let n_mels = 4usize;
  let n_state = 4usize;
  let n_ctx = 4usize;
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  let wrong = Array::zeros::<f32>(&[1i32, 8, 9]).unwrap();
  let err = enc
    .forward(&wrong)
    .expect_err("wrong batched mel channel width must be rejected before conv1");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

/// With a non-empty stack of self-attention blocks the output keeps the
/// `(1, n_ctx, n_state)` shape (the blocks preserve the sequence shape).
#[test]
fn encoder_with_blocks_preserves_shape() {
  let n_mels = 4usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let frames = 8usize;
  let n_ctx = 4usize;
  let (c1w, c1b) = center_tap_identity_conv(n_state, n_mels);
  let (c2w, c2b) = center_tap_identity_conv(n_state, n_state);
  let blocks = vec![
    identity_encoder_block(n_state, n_head),
    identity_encoder_block(n_state, n_head),
  ];
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    blocks,
    layer_norm(n_state),
    Dtype::F32,
  )
  .unwrap();
  let mel_buf: Vec<f32> = (0..frames * n_mels).map(|i| 0.1 * i as f32).collect();
  let mel = arr(&mel_buf, &[frames as i32, n_mels as i32]);
  let out = enc.forward(&mel).unwrap();
  assert_eq!(out.shape(), vec![1, n_ctx, n_state]);
}

// ---- fp16 / bf16 activation-dtype preservation ----------------------
//
// The encoder adds the precomputed `sinusoids(n_ctx, n_state)` positional
// embedding to the post-conv activations (`x + positional_embedding`,
// `whisper.py:431`). The reference stores that table cast to the model dtype
// (`self._positional_embedding = sinusoids(...).astype(dtype)`,
// `whisper.py:422`). The port mirrors this — `AudioEncoder::new` casts the
// (f32-built) sinusoid to the `dtype` argument — so an f16/bf16 checkpoint's
// activations are not promoted to f32 by the positional add. These tests pin
// that the stored table is in the model dtype, and that a full f16/bf16 forward
// (conv front-end → sinusoid add → attention → ln_post) yields the same dtype.

/// A center-tap-identity conv (the [`center_tap_identity_conv`] layout) whose
/// weight + bias are in `dtype`, so an `x` of that dtype convolves without
/// promotion (a real checkpoint casts every conv weight to the model dtype).
fn center_tap_identity_conv_dtype(c_out: usize, c_in: usize, dtype: Dtype) -> (Array, Array) {
  let (w, b) = center_tap_identity_conv(c_out, c_in);
  (w.astype(dtype).unwrap(), b.astype(dtype).unwrap())
}

/// A self-attention-only encoder block (the [`identity_encoder_block`] wiring)
/// whose every weight is in `dtype`.
fn identity_encoder_block_dtype(
  n_state: usize,
  n_head: usize,
  dtype: Dtype,
) -> ResidualAttentionBlock {
  let id = || {
    let mut w = vec![0.0_f32; n_state * n_state];
    for i in 0..n_state {
      w[i * n_state + i] = 1.0;
    }
    Linear::new(
      arr(&w, &[n_state as i32, n_state as i32])
        .astype(dtype)
        .unwrap(),
      None,
    )
  };
  let mha = MultiHeadAttention::new(n_head, id(), id(), id(), id());
  let ln = || {
    LayerNorm::new(
      Some(
        Array::ones::<f32>(&(n_state,))
          .unwrap()
          .astype(dtype)
          .unwrap(),
      ),
      Some(
        Array::zeros::<f32>(&(n_state,))
          .unwrap()
          .astype(dtype)
          .unwrap(),
      ),
      1e-5,
    )
  };
  let mlp1 = Linear::new(
    Array::zeros::<f32>(&[(4 * n_state) as i32, n_state as i32])
      .unwrap()
      .astype(dtype)
      .unwrap(),
    None,
  );
  let mlp2 = Linear::new(
    Array::zeros::<f32>(&[n_state as i32, (4 * n_state) as i32])
      .unwrap()
      .astype(dtype)
      .unwrap(),
    None,
  );
  ResidualAttentionBlock::new(mha, ln(), None, mlp1, mlp2, ln())
}

/// The stored positional embedding is cast to the model dtype (f16), so the
/// `x + positional_embedding` add cannot promote an f16 activation to f32.
#[test]
fn encoder_positional_embedding_is_model_dtype_f16() {
  encoder_positional_embedding_is_model_dtype(Dtype::F16);
}

/// Same for bf16.
#[test]
fn encoder_positional_embedding_is_model_dtype_bf16() {
  encoder_positional_embedding_is_model_dtype(Dtype::BF16);
}

fn encoder_positional_embedding_is_model_dtype(dtype: Dtype) {
  let n_ctx = 6usize;
  let n_state = 4usize;
  let (c1w, c1b) = center_tap_identity_conv_dtype(n_state, n_state, dtype);
  let (c2w, c2b) = center_tap_identity_conv_dtype(n_state, n_state, dtype);
  let enc = AudioEncoder::new(
    c1w,
    c1b,
    c2w,
    c2b,
    n_ctx,
    n_state,
    Vec::new(),
    layer_norm(n_state),
    dtype,
  )
  .unwrap();
  assert_eq!(
    enc.positional_embedding_ref().dtype().unwrap(),
    dtype,
    "the sinusoid positional embedding must be stored in the model dtype \
     (whisper.py:422 `sinusoids(...).astype(dtype)`)"
  );
}

/// A full f16 encoder forward (conv → gelu → conv → gelu → + sinusoid → block →
/// ln_post) preserves the activation dtype end-to-end: f16 mel → f16 output, no
/// hidden promotion to f32 by the positional add or the attention scale.
#[test]
fn encoder_forward_preserves_f16() {
  encoder_forward_preserves_dtype(Dtype::F16);
}

/// Same for bf16.
#[test]
fn encoder_forward_preserves_bf16() {
  encoder_forward_preserves_dtype(Dtype::BF16);
}

fn encoder_forward_preserves_dtype(dtype: Dtype) {
  let n_mels = 4usize;
  let n_state = 4usize;
  let n_head = 2usize;
  let frames = 8usize;
  let n_ctx = 4usize;
  let (c1w, c1b) = center_tap_identity_conv_dtype(n_state, n_mels, dtype);
  let (c2w, c2b) = center_tap_identity_conv_dtype(n_state, n_state, dtype);
  let blocks = vec![identity_encoder_block_dtype(n_state, n_head, dtype)];
  let ln = LayerNorm::new(
    Some(
      Array::ones::<f32>(&(n_state,))
        .unwrap()
        .astype(dtype)
        .unwrap(),
    ),
    Some(
      Array::zeros::<f32>(&(n_state,))
        .unwrap()
        .astype(dtype)
        .unwrap(),
    ),
    1e-5,
  );
  let enc = AudioEncoder::new(c1w, c1b, c2w, c2b, n_ctx, n_state, blocks, ln, dtype).unwrap();
  let mel_buf: Vec<f32> = (0..frames * n_mels)
    .map(|i| 0.1 * (i as f32).sin())
    .collect();
  let mel = arr(&mel_buf, &[frames as i32, n_mels as i32])
    .astype(dtype)
    .unwrap();
  let out = enc.forward(&mel).unwrap();
  assert_eq!(out.shape(), vec![1, n_ctx, n_state]);
  assert_eq!(
    out.dtype().unwrap(),
    dtype,
    "encoder output must stay in the activation dtype (no promotion to f32)"
  );
}

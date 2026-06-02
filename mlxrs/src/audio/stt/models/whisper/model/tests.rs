use super::*;

// ───────────────────────── sanitize ───────────────────────────────────────

/// A rank-3 conv weight `(out, in, k)` (HF layout) with sequential values, so
/// the transpose to `(out, k, in)` is observable.
fn conv_hf(out: usize, in_c: usize, k: usize) -> Array {
  let data: Vec<f32> = (0..(out * in_c * k)).map(|i| i as f32).collect();
  Array::from_slice::<f32>(&data, &(out, in_c, k)).unwrap()
}

/// Run [`sanitize`] (key-remap only) and return just the renamed map, dropping
/// the `is_hf_format` flag — for the tests that only assert key remapping.
fn sanitize_map(w: HashMap<String, Array>) -> HashMap<String, Array> {
  sanitize(w).unwrap().0
}

/// Read an array's data without an `&mut` borrow. `to_vec` takes `&mut self`
/// and requires a row-major-contiguous buffer; `sanitize`'s conv transpose
/// yields a strided view, so force contiguity first.
fn to_vec(a: &Array) -> Vec<f32> {
  crate::ops::shape::contiguous(a, false)
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
}

#[test]
fn sanitize_hf_strips_model_prefix_and_remaps_keys() {
  let mut w = HashMap::new();
  // `model.` prefix marks the HF format; a self-attn q_proj must remap to
  // `encoder.blocks.0.attn.query.weight`.
  w.insert(
    "model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
    Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "model.decoder.layers.0.encoder_attn.k_proj.weight".to_string(),
    Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "model.decoder.layer_norm.weight".to_string(),
    Array::zeros::<f32>(&(2usize,)).unwrap(),
  );
  w.insert(
    "model.decoder.embed_tokens.weight".to_string(),
    Array::zeros::<f32>(&(4usize, 2usize)).unwrap(),
  );

  let out = sanitize_map(w);
  assert!(out.contains_key("encoder.blocks.0.attn.query.weight"));
  assert!(out.contains_key("decoder.blocks.0.cross_attn.key.weight"));
  assert!(out.contains_key("decoder.ln.weight"));
  assert!(out.contains_key("decoder.token_embedding.weight"));
}

#[test]
fn sanitize_hf_drops_encoder_embed_positions() {
  let mut w = HashMap::new();
  w.insert(
    "model.encoder.embed_positions.weight".to_string(),
    Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
  );
  // a keeper, so the map is non-trivially HF
  w.insert("model.encoder.conv1.weight".to_string(), conv_hf(2, 3, 3));
  let out = sanitize_map(w);
  // encoder.embed_positions is recomputed via sinusoids → dropped.
  assert!(!out.keys().any(|k| k.contains("embed_positions")));
}

#[test]
fn sanitize_hf_remaps_decoder_embed_positions_to_positional_embedding() {
  let mut w = HashMap::new();
  w.insert(
    "model.decoder.embed_positions.weight".to_string(),
    Array::zeros::<f32>(&(4usize, 2usize)).unwrap(),
  );
  let out = sanitize_map(w);
  assert!(out.contains_key("decoder.positional_embedding"));
  assert!(!out.keys().any(|k| k.contains("embed_positions")));
}

#[test]
fn sanitize_does_not_transpose_or_cast_conv_weight() {
  // sanitize is key-remap ONLY: a HF conv weight keeps its RAW `(out, in, k)`
  // layout and its checkpoint dtype — the `(out, in, k) -> (out, k, in)`
  // transpose and the dtype cast are deferred to the shape-validated builder, so
  // an oversized conv tensor is never transposed/cast before its shape is
  // checked. The `is_hf_format` flag is reported so the builder knows to
  // transpose.
  let mut w = HashMap::new();
  // HF (out=2, in=3, k=3). Sequential values so a transpose would be observable.
  w.insert("model.encoder.conv1.weight".to_string(), conv_hf(2, 3, 3));
  let (out, is_hf) = sanitize(w).unwrap();
  assert!(is_hf, "a `model.`-prefixed checkpoint is HF format");
  let conv = &out["encoder.conv1.weight"];
  // RAW (out, in, k) layout, untransposed — element order unchanged.
  assert_eq!(conv.shape(), vec![2, 3, 3]);
  let v = to_vec(conv);
  let expected: Vec<f32> = (0..18).map(|i| i as f32).collect();
  assert_eq!(v, expected, "sanitize must not transpose the conv weight");
  // dtype is left at the checkpoint dtype (no cast in sanitize).
  assert_eq!(conv.dtype().unwrap(), Dtype::F32);
}

#[test]
fn from_weights_transposes_hf_conv_weight_after_validation() {
  // Through `from_weights`, an HF conv weight is shape-validated against the
  // config-derived RAW HF layout `(n_state, n_mels, K)` and THEN transposed to
  // the MLX `(n_state, K, n_mels)` layout the conv path consumes. Build a tiny HF
  // checkpoint and assert the materialized encoder conv1 weight is transposed.
  let n = TINY.n_state; // 4
  let mels = TINY.n_mels; // 4
  // HF conv1 raw `(out=n, in=mels, k=3)` with sequential values.
  let raw: Vec<f32> = (0..(n * mels * 3)).map(|i| i as f32).collect();
  let conv1_hf = Array::from_slice::<f32>(&raw, &(n, mels, 3usize)).unwrap();

  // A full HF-format tiny checkpoint: take the MLX tiny weights and prefix every
  // key with `model.`, then swap in the HF-layout conv weights (which the MLX
  // builder stores `(out, k, in)` but HF ships `(out, in, k)`).
  let mlx = tiny_weights();
  let mut w: HashMap<String, Array> = HashMap::new();
  for (k, v) in mlx {
    // conv weights are replaced with the HF-layout versions below.
    if k == "encoder.conv1.weight" || k == "encoder.conv2.weight" {
      continue;
    }
    w.insert(format!("model.{k}"), v);
  }
  w.insert("model.encoder.conv1.weight".to_string(), conv1_hf);
  // conv2 HF raw `(out=n, in=n, k=3)`.
  w.insert(
    "model.encoder.conv2.weight".to_string(),
    Array::ones::<f32>(&(n, n, 3usize)).unwrap(),
  );

  let model = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap();
  let conv1 = model.encoder.conv1_weight_ref();
  // Materialized MLX layout `(out, k, in)` = (4, 3, 4).
  assert_eq!(conv1.shape(), vec![n, 3, mels]);
  // Element [o=0, k=1, in=0] of the transposed tensor equals element
  // [o=0, in=0, k=1] of the raw HF `(out, in, k)` = linear index 0*12+0*3+1 = 1.
  let v = to_vec(conv1);
  // transposed layout (out, k, in): linear index for (0,1,0) = (0*3+1)*mels+0 = 4.
  assert_eq!(v[4], 1.0);
}

#[test]
fn from_weights_does_not_transpose_mlx_conv_weight() {
  // An MLX-format checkpoint (no `model.` prefix) ships conv weights already in
  // the `(out, k, in)` layout; the builder must NOT transpose them (the HF-only
  // guard). The materialized conv1 equals the input conv1 verbatim.
  let n = TINY.n_state;
  let mels = TINY.n_mels;
  let mut w = tiny_weights();
  // A distinctive conv1 with sequential values in the MLX `(out, k, in)` layout.
  let seq: Vec<f32> = (0..(n * 3 * mels)).map(|i| i as f32).collect();
  let conv1_mlx = Array::from_slice::<f32>(&seq, &(n, 3usize, mels)).unwrap();
  let before = to_vec(&conv1_mlx);
  w.insert("encoder.conv1.weight".to_string(), conv1_mlx);
  let model = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap();
  let conv1 = model.encoder.conv1_weight_ref();
  assert_eq!(conv1.shape(), vec![n, 3, mels]);
  assert_eq!(to_vec(conv1), before, "MLX conv must not be transposed");
}

#[test]
fn sanitize_mlx_keeps_keys_verbatim() {
  // MLX keys pass through unchanged (no remap when not HF format).
  let mut w = HashMap::new();
  w.insert(
    "encoder.blocks.0.attn.query.weight".to_string(),
    Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
  );
  let (out, is_hf) = sanitize(w).unwrap();
  assert!(!is_hf, "no `model.` prefix → MLX format");
  assert!(out.contains_key("encoder.blocks.0.attn.query.weight"));
}

#[test]
fn from_weights_casts_consumed_weights_to_dtype() {
  // The dtype cast is deferred from sanitize to the builder; through
  // `from_weights` every CONSUMED tensor is cast to the model dtype (after its
  // shape is validated). Build a tiny model at f16 and assert the materialized
  // encoder conv1 weight carries the model dtype.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F16).unwrap();
  assert_eq!(model.dtype(), Dtype::F16);
  assert_eq!(
    model.encoder.conv1_weight_ref().dtype().unwrap(),
    Dtype::F16,
    "consumed conv1 weight is cast to the model dtype during the build"
  );
}

#[test]
fn sanitize_rejects_colliding_remapped_keys() {
  // Two distinct HF source keys collapse onto one sanitized key: `fc1` remaps
  // to `mlp1`, so `...layers.0.fc1.weight` and an already-`mlp1` sibling both
  // land on `decoder.blocks.0.mlp1.weight`. The collision is rejected rather
  // than letting a nondeterministic survivor win.
  let mut w = HashMap::new();
  w.insert(
    "model.decoder.layers.0.fc1.weight".to_string(),
    Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "model.decoder.layers.0.mlp1.weight".to_string(),
    Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
  );
  let err = sanitize(w).unwrap_err();
  assert!(
    matches!(&err, Error::KeyCollision(p) if p.context() == "WhisperModel::sanitize"),
    "expected KeyCollision from sanitize, got {err:?}"
  );
}

// ───────────────────────── full tiny-model build ──────────────────────────

use super::super::audio::N_FRAMES;

/// Tiny test dimensions: 1 encoder + 1 decoder layer, head_dim divisible.
///
/// `n_audio_ctx` is the architecturally fixed `N_FRAMES / 2` (`1500`) — pinned
/// at construction, so it cannot be shrunk for the test. Only the *width* dims
/// (`n_state`, `n_head`, `n_vocab`, `n_text_ctx`) are tiny; the conv / attention
/// weights do not scale with `n_audio_ctx`, so the encoder still runs cheaply on
/// the fixed `N_FRAMES`-frame mel.
struct TinyDims {
  n_mels: usize,
  n_audio_ctx: usize,
  n_state: usize,
  n_head: usize,
  n_vocab: usize,
  n_text_ctx: usize,
}

const TINY: TinyDims = TinyDims {
  n_mels: 4,
  // Fixed: conv2 stride-2 halves the `N_FRAMES` (3000) padded mel to 1500.
  n_audio_ctx: N_FRAMES / 2,
  n_state: 4,
  n_head: 2,
  n_vocab: 8,
  n_text_ctx: 6,
};

/// A `(N_FRAMES, n_mels)` mel — the fixed padded frame count every Whisper
/// segment carries into the encoder. After conv2's stride 2 → `n_audio_ctx`
/// (1500) frames, matching the encoder positional embedding.
fn tiny_mel() -> Array {
  Array::ones::<f32>(&(N_FRAMES, TINY.n_mels)).unwrap()
}

fn dims() -> ModelDimensions {
  ModelDimensions::new(
    TINY.n_mels,
    TINY.n_audio_ctx,
    TINY.n_state,
    TINY.n_head,
    1, // n_audio_layer
    TINY.n_vocab,
    TINY.n_text_ctx,
    TINY.n_state,
    TINY.n_head,
    1, // n_text_layer
  )
  .unwrap()
}

fn ones2(r: usize, c: usize) -> Array {
  Array::ones::<f32>(&(r, c)).unwrap()
}
fn zeros1(n: usize) -> Array {
  Array::zeros::<f32>(&(n,)).unwrap()
}
fn ones1(n: usize) -> Array {
  Array::ones::<f32>(&(n,)).unwrap()
}

/// Insert the weights for one attention sub-module under `prefix`.
fn put_attn(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  for p in ["query", "value", "out"] {
    w.insert(format!("{prefix}.{p}.weight"), ones2(n, n));
    w.insert(format!("{prefix}.{p}.bias"), zeros1(n));
  }
  // key: weight only (no bias).
  w.insert(format!("{prefix}.key.weight"), ones2(n, n));
}

/// Insert a full LayerNorm (weight=ones, bias=zeros).
fn put_ln(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  w.insert(format!("{prefix}.weight"), ones1(n));
  w.insert(format!("{prefix}.bias"), zeros1(n));
}

/// Insert one residual block (decoder = with cross-attn).
fn put_block(w: &mut HashMap<String, Array>, prefix: &str, n: usize, cross: bool) {
  put_attn(w, &format!("{prefix}.attn"), n);
  put_ln(w, &format!("{prefix}.attn_ln"), n);
  if cross {
    put_attn(w, &format!("{prefix}.cross_attn"), n);
    put_ln(w, &format!("{prefix}.cross_attn_ln"), n);
  }
  // mlp1: (4n, n), mlp2: (n, 4n).
  w.insert(format!("{prefix}.mlp1.weight"), ones2(4 * n, n));
  w.insert(format!("{prefix}.mlp1.bias"), zeros1(4 * n));
  w.insert(format!("{prefix}.mlp2.weight"), ones2(n, 4 * n));
  w.insert(format!("{prefix}.mlp2.bias"), zeros1(n));
  put_ln(w, &format!("{prefix}.mlp_ln"), n);
}

/// Build a complete MLX-format tiny Whisper checkpoint.
fn tiny_weights() -> HashMap<String, Array> {
  let n = TINY.n_state;
  let mut w = HashMap::new();

  // conv1 (n_state, k=3, n_mels), conv2 (n_state, k=3, n_state) — MLX layout.
  w.insert(
    "encoder.conv1.weight".to_string(),
    Array::ones::<f32>(&(n, 3usize, TINY.n_mels)).unwrap(),
  );
  w.insert("encoder.conv1.bias".to_string(), zeros1(n));
  w.insert(
    "encoder.conv2.weight".to_string(),
    Array::ones::<f32>(&(n, 3usize, n)).unwrap(),
  );
  w.insert("encoder.conv2.bias".to_string(), zeros1(n));
  put_block(&mut w, "encoder.blocks.0", n, false);
  put_ln(&mut w, "encoder.ln_post", n);

  // decoder.
  w.insert(
    "decoder.token_embedding.weight".to_string(),
    ones2(TINY.n_vocab, n),
  );
  w.insert(
    "decoder.positional_embedding".to_string(),
    ones2(TINY.n_text_ctx, n),
  );
  put_block(&mut w, "decoder.blocks.0", n, true);
  put_ln(&mut w, "decoder.ln", n);

  w
}

#[test]
fn log_mel_override_equals_log_mel_spectrogram_whisper() {
  // `WhisperModel::log_mel` overrides the STT trait default with the Whisper
  // front-end (`log_mel_spectrogram_whisper`: Slaney bank + log10 + dynamic
  // clamp + (T, n_mels) layout). It must equal calling that front-end
  // directly with the checkpoint's `n_mels` and no extra padding.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();

  // A deterministic two-tone waveform; 1600 samples → 10 frames after the
  // STFT drop-last, enough for a non-trivial mel.
  let buf: Vec<f32> = (0..1600)
    .map(|i| {
      let t = i as f32;
      0.6 * (2.0 * std::f32::consts::PI * 440.0 * t / 16_000.0).sin()
        + 0.3 * (2.0 * std::f32::consts::PI * 1200.0 * t / 16_000.0).sin()
    })
    .collect();
  let audio = Array::from_slice::<f32>(&buf, &[buf.len() as i32]).unwrap();

  let got = AutoregressiveStt::log_mel(&model, &audio).unwrap();
  let expected = super::super::audio::log_mel_spectrogram_whisper(&audio, TINY.n_mels, 0).unwrap();

  // (num_frames, n_mels) layout: n_mels on the LAST axis (the Whisper
  // override is distinct from the generic (n_mels, T) default).
  assert_eq!(got.shape(), expected.shape(), "whisper log_mel layout");
  assert_eq!(
    *got.shape().last().unwrap(),
    TINY.n_mels,
    "(num_frames, n_mels): n_mels on the last axis"
  );
  let g = to_vec(&got);
  let e = to_vec(&expected);
  assert_eq!(g.len(), e.len());
  for (gi, ei) in g.iter().zip(e.iter()) {
    assert!(
      (gi - ei).abs() <= 1e-5,
      "WhisperModel::log_mel must equal log_mel_spectrogram_whisper: got {gi}, expected {ei}"
    );
  }
}

#[test]
fn from_weights_builds_and_encodes() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();

  // Mel input: (N_FRAMES=3000, n_mels=4). After conv2 stride-2 → 1500 frames =
  // n_audio_ctx, matching the positional embedding.
  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();
  // (1, n_audio_ctx, n_state).
  assert_eq!(enc.shape(), vec![1, TINY.n_audio_ctx, TINY.n_state]);
}

#[test]
fn multi_layer_build_reserves_and_fills_every_block() {
  // The encoder / decoder block vectors and the decoder KV cache are reserved
  // through the fallible `reserve_or_error` path (not `Vec::with_capacity`), then
  // every layer is pushed into the reserved capacity. Build a 3-encoder /
  // 3-decoder-layer model and assert every block is present — the reserve+push
  // loop neither drops nor truncates a layer, and a forward mints a KV cache with
  // one entry per decoder block.
  let n = TINY.n_state;
  let layers = 3usize;
  let mut w = HashMap::new();
  w.insert(
    "encoder.conv1.weight".to_string(),
    Array::ones::<f32>(&(n, 3usize, TINY.n_mels)).unwrap(),
  );
  w.insert("encoder.conv1.bias".to_string(), zeros1(n));
  w.insert(
    "encoder.conv2.weight".to_string(),
    Array::ones::<f32>(&(n, 3usize, n)).unwrap(),
  );
  w.insert("encoder.conv2.bias".to_string(), zeros1(n));
  for i in 0..layers {
    put_block(&mut w, &format!("encoder.blocks.{i}"), n, false);
  }
  put_ln(&mut w, "encoder.ln_post", n);
  w.insert(
    "decoder.token_embedding.weight".to_string(),
    ones2(TINY.n_vocab, n),
  );
  w.insert(
    "decoder.positional_embedding".to_string(),
    ones2(TINY.n_text_ctx, n),
  );
  for i in 0..layers {
    put_block(&mut w, &format!("decoder.blocks.{i}"), n, true);
  }
  put_ln(&mut w, "decoder.ln", n);

  let dims = ModelDimensions::new(
    TINY.n_mels,
    TINY.n_audio_ctx,
    n,
    TINY.n_head,
    layers, // n_audio_layer
    TINY.n_vocab,
    TINY.n_text_ctx,
    n,
    TINY.n_head,
    layers, // n_text_layer
  )
  .unwrap();
  let model = WhisperModel::from_weights(dims, w, Dtype::F32).unwrap();
  // Every decoder block was reserved-then-pushed (no truncation).
  assert_eq!(model.decoder.num_blocks(), layers);

  // A forward mints a fresh KV cache with exactly one entry per decoder block.
  let enc = model.encode(&tiny_mel()).unwrap();
  let mut cache = model.new_cache();
  model.decode_step(&mut cache, &enc, &[0u32]).unwrap();
  let inner = cache.inner.as_ref().expect("cache populated after a step");
  assert_eq!(inner.len(), layers, "one KV cache entry per decoder block");
}

#[test]
fn encode_rejects_non_n_frames_mel() {
  // The config pins `n_audio_ctx` to N_FRAMES / 2, so a config-built encoder
  // expects exactly N_FRAMES (3000) input frames. A mel with any other frame
  // count is rejected with a typed ShapePairMismatch before conv1 runs — a
  // hostile (e.g. enormous) frame count cannot drive the conv activation toward
  // an out-of-memory abort.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let wrong = Array::ones::<f32>(&(N_FRAMES + 8, TINY.n_mels)).unwrap();
  let err = model.encode(&wrong).unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for a non-N_FRAMES mel, got {err:?}"
  );
  // The exact N_FRAMES count is accepted.
  assert!(model.encode(&tiny_mel()).is_ok());
}

#[test]
fn decode_step_returns_row_logits_and_advances_cache() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  // A fresh caller-owned cache (no model-stored decode state).
  let mut cache = model.new_cache();
  assert!(cache.is_empty(), "fresh cache holds no decoded positions");

  // Step 1: prefill a single-token prompt prefix → rank-1 (vocab,) logits.
  let mut tokens = vec![0u32];
  let logits = model.decode_step(&mut cache, &enc, &tokens).unwrap();
  assert_eq!(logits.shape(), vec![TINY.n_vocab]);
  assert_eq!(
    cache.len(),
    1,
    "cache advanced to one position after prefill"
  );

  // Step 2: the cache now holds 1 position, so only the new last token is
  // forwarded (the positional slice advances); still returns (vocab,).
  tokens.push(1);
  let logits2 = model.decode_step(&mut cache, &enc, &tokens).unwrap();
  assert_eq!(logits2.shape(), vec![TINY.n_vocab]);
  assert_eq!(cache.len(), 2, "cache advanced to two positions");

  // A second fresh cache restarts at position 0 — the model holds no decode
  // state between generations.
  let mut cache2 = model.new_cache();
  let logits3 = model.decode_step(&mut cache2, &enc, &[2]).unwrap();
  assert_eq!(logits3.shape(), vec![TINY.n_vocab]);
  assert_eq!(cache2.len(), 1);
}

#[test]
fn decode_step_with_cross_qk_exposes_per_layer_weights() {
  // The public cross-attention extraction (`decode_step_with_cross_qk`): the
  // (1, T, V) logits are byte-identical to the normal `decode_step` last-row
  // slice, and a per-decoder-layer cross-qk list is returned, each `Some` and
  // shaped (1, n_head, T, n_audio_ctx) — the attention pattern over the audio
  // frames the word-timestamp DTW will consume. Only extracted + exposed here.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let enc = model.encode(&tiny_mel()).unwrap();

  // Prefill a 3-token prefix on a fresh cache.
  let tokens = [0u32, 1, 2];
  let mut cache = model.new_cache();
  let (logits, cross_qk) = model
    .decode_step_with_cross_qk(&mut cache, &enc, &tokens)
    .unwrap();

  // Full (1, T, V) logits (not the last-row slice).
  assert_eq!(logits.shape(), vec![1, tokens.len(), TINY.n_vocab]);
  assert_eq!(
    cache.len(),
    tokens.len(),
    "cache advanced to the prefix length"
  );

  // One cross-qk per decoder layer (n_text_layer = 1), `Some`, shaped
  // (B=1, n_head, T, n_audio_ctx).
  assert_eq!(cross_qk.len(), 1, "one cross-qk per decoder layer");
  let qk = cross_qk[0]
    .as_ref()
    .expect("decoder block cross-qk must be Some");
  assert_eq!(
    qk.shape(),
    vec![1, TINY.n_head, tokens.len(), TINY.n_audio_ctx],
    "cross-qk shape (B, H, T, n_audio_ctx)"
  );

  // The last-position row matches `decode_step` exactly (the extraction does
  // not perturb the normal forward).
  let mut cache_plain = model.new_cache();
  let row = model.decode_step(&mut cache_plain, &enc, &tokens).unwrap();
  let (_, t, v) = (1i32, tokens.len() as i32, TINY.n_vocab as i32);
  let last_row = crate::ops::indexing::slice(&logits, &[0, t - 1, 0], &[1, t, v], &[1, 1, 1])
    .unwrap()
    .reshape(&[v])
    .unwrap();
  let want = row.try_clone().unwrap().to_vec::<f32>().unwrap();
  let got = last_row.try_clone().unwrap().to_vec::<f32>().unwrap();
  assert_eq!(
    got, want,
    "cross-qk path last-row logits == decode_step row"
  );
}

#[test]
fn decode_step_rejects_tokens_shorter_than_cache() {
  // The driver always extends `tokens`; a prefix shorter than the cache (no
  // new tokens) is a misuse, rejected with a typed EmptyInput rather than a
  // panic or a silent no-op.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();
  let mut cache = model.new_cache();
  model.decode_step(&mut cache, &enc, &[0, 1]).unwrap();
  // cache now holds 2 positions; passing a 2-token prefix yields no new tokens.
  let err = model.decode_step(&mut cache, &enc, &[0, 1]).unwrap_err();
  assert!(
    matches!(err, Error::EmptyInput(_)),
    "expected EmptyInput for a non-extending prefix, got {err:?}"
  );
}

/// A checkpoint that passes config validation but ships an oversized
/// `encoder.conv1.weight` (a huge output-channel count while `n_audio_state`
/// stays small) is rejected at build with a typed shape error — BEFORE the
/// encoder is built and BEFORE any forward materializes
/// `N_FRAMES * actual_out_channels`. This is the core resource guard: the
/// validated config caps now provably equal the actual tensor extents.
#[test]
fn from_weights_rejects_oversized_conv1_out_channels() {
  let n = TINY.n_state;
  let mut w = tiny_weights();
  // conv1 expected (n_state=4, k=3, n_mels=4); ship (4096, 3, 4) instead — a
  // huge out-channel count the config's small n_audio_state does not authorize.
  w.insert(
    "encoder.conv1.weight".to_string(),
    Array::ones::<f32>(&(4096usize, 3usize, TINY.n_mels)).unwrap(),
  );
  let err = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "encoder.conv1.weight");
      assert!(
        matches!(p.inner(), Error::ShapePairMismatch(_)),
        "inner must be ShapePairMismatch, got {:?}",
        p.inner()
      );
    }
    other => panic!("expected LayerKeyed(ShapePairMismatch), got {other:?}"),
  }
  // The correctly-shaped conv1 (a `_ = n`-width tensor) still builds.
  let _ = n;
  assert!(WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).is_ok());
}

/// An oversized HF-format conv weight is rejected against its RAW
/// `(out, in, k)` layout BEFORE the `(out, in, k) -> (out, k, in)` transpose (and
/// the dtype cast) run — so the oversized transpose/cast allocation never
/// happens. The keyed `ShapePairMismatch` reports the RAW HF expected layout
/// `(n_state, n_mels, K)` (= `[4, 4, 3]`), proving the validation ran on the
/// pre-transpose tensor; a post-transpose check would instead compare the MLX
/// `(n_state, K, n_mels)` layout.
#[test]
fn from_weights_rejects_oversized_hf_conv_before_transpose() {
  let n = TINY.n_state; // 4
  let mels = TINY.n_mels; // 4
  // A full HF checkpoint: prefix the MLX tiny weights with `model.`, drop the
  // conv weights, then ship an OVERSIZED HF-layout conv1 `(out=4096, in=mels,
  // k=3)` and a valid HF conv2.
  let mlx = tiny_weights();
  let mut w: HashMap<String, Array> = HashMap::new();
  for (k, v) in mlx {
    if k == "encoder.conv1.weight" || k == "encoder.conv2.weight" {
      continue;
    }
    w.insert(format!("model.{k}"), v);
  }
  // Oversized HF conv1: raw `(out, in, k)` with a huge out-channel count.
  w.insert(
    "model.encoder.conv1.weight".to_string(),
    Array::ones::<f32>(&(4096usize, mels, 3usize)).unwrap(),
  );
  w.insert(
    "model.encoder.conv2.weight".to_string(),
    Array::ones::<f32>(&(n, n, 3usize)).unwrap(),
  );

  let err = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "encoder.conv1.weight");
      match p.inner() {
        Error::ShapePairMismatch(sp) => {
          // The expected shape is the RAW HF `(out, in, k)` = (n_state, n_mels, K)
          // = [4, 4, 3] — NOT the post-transpose MLX [4, 3, 4]. This proves the
          // shape was validated BEFORE the transpose.
          assert_eq!(
            sp.expected(),
            [n, mels, 3].as_slice(),
            "expected shape must be the RAW HF (out, in, k) layout (pre-transpose)"
          );
          assert_eq!(
            sp.actual(),
            [4096, mels, 3].as_slice(),
            "actual shape is the oversized raw HF tensor"
          );
        }
        other => panic!("inner must be ShapePairMismatch, got {other:?}"),
      }
    }
    other => panic!("expected LayerKeyed(ShapePairMismatch), got {other:?}"),
  }
}

/// An oversized `decoder.token_embedding.weight` (a huge vocab row count while
/// `n_vocab` stays small) is rejected at build — the weight-tied logit head
/// would otherwise materialize `(1, T, huge_vocab)` logits and the embedding
/// table itself would exceed the `n_vocab * n_text_state` cap.
#[test]
fn from_weights_rejects_oversized_token_embedding() {
  let n = TINY.n_state;
  let mut w = tiny_weights();
  // token embedding expected (n_vocab=8, n_state=4); ship (1_000_000, 4).
  w.insert(
    "decoder.token_embedding.weight".to_string(),
    ones2(1_000_000, n),
  );
  let err = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "decoder.token_embedding.weight");
      assert!(matches!(p.inner(), Error::ShapePairMismatch(_)));
    }
    other => panic!("expected LayerKeyed(ShapePairMismatch), got {other:?}"),
  }
}

/// A wrong-shaped attention projection weight (here the decoder self-attn
/// `query`, a non-square `(out, in)`) is rejected at build with the keyed shape
/// error — every consumed linear/bias/layer-norm tensor is gated, not just the
/// convs and embeddings.
#[test]
fn from_weights_rejects_wrong_attention_weight_shape() {
  let n = TINY.n_state;
  let mut w = tiny_weights();
  // decoder block-0 self-attn query expected (n_state, n_state) = (4, 4); ship
  // a non-square (5, 4).
  w.insert(
    "decoder.blocks.0.attn.query.weight".to_string(),
    ones2(5, n),
  );
  let err = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "decoder.blocks.0.attn.query.weight");
      assert!(matches!(p.inner(), Error::ShapePairMismatch(_)));
    }
    other => panic!("expected LayerKeyed(ShapePairMismatch), got {other:?}"),
  }
}

/// A wrong-length LayerNorm weight (the encoder `ln_post`) is rejected — a
/// rank/length deviation in any consumed norm tensor fails fast.
#[test]
fn from_weights_rejects_wrong_layer_norm_shape() {
  let mut w = tiny_weights();
  // encoder ln_post weight expected (n_state,) = (4,); ship (8,).
  w.insert("encoder.ln_post.weight".to_string(), ones1(8));
  let err = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "encoder.ln_post.weight");
      assert!(matches!(p.inner(), Error::ShapePairMismatch(_)));
    }
    other => panic!("expected LayerKeyed(ShapePairMismatch), got {other:?}"),
  }
}

/// `decode_step` rejects a token prefix longer than `max_context`
/// (`n_text_ctx`) BEFORE building the `(1, T)` token array — so an oversized
/// prefix (or an oversized `max_new_tokens` accumulated by a driver) cannot
/// drive the `(1, T, n_text_state)` decoder embedding past the config context
/// cap. The error is a typed `OutOfRange`.
#[test]
fn decode_step_rejects_prefix_longer_than_max_context() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let enc = model.encode(&tiny_mel()).unwrap();
  let mut cache = model.new_cache();
  // max_context = n_text_ctx = 6; a 7-token prefix exceeds it.
  let too_long: Vec<u32> = (0..(TINY.n_text_ctx as u32 + 1)).collect();
  let err = model.decode_step(&mut cache, &enc, &too_long).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for an over-context prefix, got {err:?}"
  );
  // A prefix exactly at the context bound is accepted.
  let at_bound: Vec<u32> = (0..TINY.n_text_ctx as u32).collect();
  let mut cache2 = model.new_cache();
  assert!(model.decode_step(&mut cache2, &enc, &at_bound).is_ok());
}

/// `decode_step` validates the encoder-states extent BEFORE building the
/// token array or entering the decoder. An `enc` whose audio context exceeds
/// `n_audio_ctx` is rejected with a typed `ShapePairMismatch` — so the
/// cross-attention cannot form scores from an oversized `enc.shape()[1]`.
#[test]
fn decode_step_rejects_oversized_enc() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  // (1, n_audio_ctx + 8, n_audio_state) — a longer-than-config audio segment.
  let oversized = Array::ones::<f32>(&(1usize, TINY.n_audio_ctx + 8, TINY.n_state)).unwrap();
  let mut cache = model.new_cache();
  let err = model
    .decode_step(&mut cache, &oversized, &[0u32])
    .unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for an oversized enc, got {err:?}"
  );
  // The rejected step mutated nothing: the cache is still empty (no token array
  // or decoder forward ran).
  assert!(
    cache.is_empty(),
    "a rejected enc leaves the cache untouched"
  );
}

/// `decode_step` rejects a batched `enc` (leading dimension != 1) before
/// allocating — Whisper decodes one segment at a time, so a `[B>1, ...]` encoder
/// tensor cannot drive the cross-attention KV / score buffers past the
/// single-segment config caps.
#[test]
fn decode_step_rejects_batched_enc() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  // (B=4, n_audio_ctx, n_audio_state) — a valid per-segment shape but batched.
  let batched = Array::ones::<f32>(&(4usize, TINY.n_audio_ctx, TINY.n_state)).unwrap();
  let mut cache = model.new_cache();
  let err = model
    .decode_step(&mut cache, &batched, &[0u32])
    .unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for a batched enc, got {err:?}"
  );
  assert!(
    cache.is_empty(),
    "a rejected enc leaves the cache untouched"
  );
}

/// Same-utterance multi-step decoding (the same `enc` threaded on every step):
/// every step succeeds and advances the cache by one position — the canonical
/// single-utterance decode loop, with the cross-attention K/V projected on the
/// first step and reused thereafter.
#[test]
fn decode_step_same_encoder_states_reuses_cache_across_steps() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let enc = model.encode(&tiny_mel()).unwrap();

  // Drive several warm steps with the SAME enc each time. Every step must succeed
  // and advance the cache.
  let mut cache = model.new_cache();
  let mut tokens: Vec<u32> = Vec::new();
  for step in 0..5u32 {
    tokens.push(step);
    let logits = model
      .decode_step(&mut cache, &enc, &tokens)
      .unwrap_or_else(|e| panic!("warm step {step} must succeed: {e:?}"));
    assert_eq!(logits.shape(), vec![TINY.n_vocab]);
    assert_eq!(
      cache.len(),
      tokens.len(),
      "the cache advanced one position per step"
    );
  }
}

/// The same guard holds at the crate-private decoder entry
/// (`decode_tokens`), so the bound is enforced regardless of caller — this is
/// the chokepoint the public `detect_language` and the decoding task's main loop
/// both funnel through. An oversized or batched `encoder_states` is rejected
/// before the decoder forward.
#[test]
fn decode_tokens_rejects_oversized_and_batched_enc() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let tok = Array::from_slice::<u32>(&[0u32], &[1, 1]).unwrap();

  // Oversized audio context.
  let oversized = Array::ones::<f32>(&(1usize, TINY.n_audio_ctx + 8, TINY.n_state)).unwrap();
  let err = model.decode_tokens(&tok, &oversized, None).unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for an oversized enc, got {err:?}"
  );

  // Batched encoder states.
  let batched = Array::ones::<f32>(&(2usize, TINY.n_audio_ctx, TINY.n_state)).unwrap();
  let err = model.decode_tokens(&tok, &batched, None).unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for a batched enc, got {err:?}"
  );

  // A correctly-shaped single segment is accepted.
  let good = model.encode(&tiny_mel()).unwrap();
  assert!(model.decode_tokens(&tok, &good, None).is_ok());
}

/// The encoder batch guard surfaces through the model's `encode`: an already-3-D
/// mel with a leading batch dimension > 1 is rejected before conv1 (the
/// single-segment Whisper contract), with a typed `ShapePairMismatch`.
#[test]
fn encode_rejects_batched_mel() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  // (B=4, N_FRAMES, n_mels) — a valid frame count but an oversized batch.
  let batched = Array::ones::<f32>(&(4usize, N_FRAMES, TINY.n_mels)).unwrap();
  let err = model.encode(&batched).unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for a batched mel, got {err:?}"
  );
}

#[test]
fn from_weights_missing_weight_errors() {
  let mut w = tiny_weights();
  w.remove("decoder.ln.weight");
  let err = WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap_err();
  assert!(
    matches!(&err, Error::MissingKey(p) if p.key().contains("decoder.ln.weight")),
    "expected MissingKey(decoder.ln.weight), got {err:?}"
  );
}

#[test]
fn from_weights_empty_map_fails_on_missing_not_panic() {
  // Valid dims + an empty weight map: validate passes (eager, defense-in-depth
  // re-check), then the builder fails with a typed MissingKey on the first
  // absent weight rather than panicking.
  let err = WhisperModel::from_weights(dims(), HashMap::new(), Dtype::F32).unwrap_err();
  assert!(matches!(err, Error::MissingKey(_)));
}

#[test]
fn metadata_accessors() {
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  assert_eq!(model.eot(), 50257);
  assert_eq!(model.max_context(), TINY.n_text_ctx);
  assert_eq!(model.mel_config().n_mels(), TINY.n_mels);
  assert_eq!(model.mel_config().sample_rate(), 16_000);
  assert_eq!(model.dims().n_vocab(), TINY.n_vocab);
  assert_eq!(model.dtype(), Dtype::F32);
}

#[test]
fn eot_defaults_to_canonical_id() {
  // Without an attached tokenizer, `eot` falls back to the canonical
  // `<|endoftext|>` id.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  assert_eq!(model.eot(), 50257);
}

#[test]
fn with_eot_token_overrides_eot() {
  // `with_eot_token` records the loaded tokenizer's resolved eot id, which
  // `eot()` then reflects instead of the canonical default.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32)
    .unwrap()
    .with_eot_token(2);
  assert_eq!(model.eot(), 2);
}

#[test]
fn initial_tokens_without_tokenizer_is_typed_error() {
  // `initial_tokens` needs the tokenizer to build the start-of-transcript
  // language/task sequence; without one it is a typed InvariantViolation
  // rather than a panic or a bogus single-token seed.
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let err = model
    .initial_tokens(&crate::audio::stt::model::TranscribeOptions::new())
    .unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation without a tokenizer, got {err:?}"
  );
}

#[test]
fn transcribe_without_tokenizer_is_typed_error() {
  // The high-level Transcribe contract likewise requires an attached
  // tokenizer; without one it points the caller at `with_tokenizer` / the
  // lower-level decoding entry point via a typed error.
  use crate::audio::stt::model::{Transcribe as _, TranscribeOptions};
  let model = WhisperModel::from_weights(dims(), tiny_weights(), Dtype::F32).unwrap();
  let audio = Array::zeros::<f32>(&[16_000]).unwrap();
  let err = model
    .transcribe(&audio, &TranscribeOptions::new())
    .unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation without a tokenizer, got {err:?}"
  );
}

// ─────────────────── sharded safetensors merge ────────────────────────────

/// A fresh, writable per-test temp directory — the crate's no-`tempfile`-crate
/// convention (`temp_dir()` + pid + a process-unique counter), mirroring
/// `io::tests::fresh_dir` and `lm::load::save_tests`.
fn fresh_dir(tag: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs-whisper-{tag}-{}-{n}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// Write a single `{name -> array}` map to `<dir>/<file>.safetensors`.
fn write_shard(dir: &std::path::Path, file: &str, arrays: &HashMap<String, Array>) {
  let path = dir.join(format!("{file}.safetensors"));
  crate::io::save_safetensors(&path, arrays).unwrap();
}

#[test]
fn load_all_safetensors_rejects_duplicate_key_across_shards() {
  // Two shard files define the SAME tensor key. A last-wins `HashMap::extend`
  // would silently keep the later-sorted shard's tensor and decode with shadowed
  // parameters; the merge must instead fail closed with a typed error that names
  // both the duplicated key and its source shard file.
  let dir = fresh_dir("dup-shard");

  let mut a: HashMap<String, Array> = HashMap::new();
  a.insert(
    "encoder.conv1.weight".to_string(),
    Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap(),
  );
  // `b.safetensors` sorts after `a.safetensors`, so a last-wins merge would let
  // this shadowing tensor win; the SAME key collides.
  let mut b: HashMap<String, Array> = HashMap::new();
  b.insert(
    "encoder.conv1.weight".to_string(),
    Array::from_slice::<f32>(&[9.0_f32, 9.0], &(2usize,)).unwrap(),
  );
  write_shard(&dir, "a", &a);
  write_shard(&dir, "b", &b);

  let err = load_all_safetensors(&dir).unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      // The shard file name is the LayerKeyed context …
      assert!(
        p.layer().ends_with("b.safetensors"),
        "LayerKeyed layer must name the offending shard file, got {:?}",
        p.layer()
      );
      // … wrapping a KeyCollision that names the duplicated tensor key.
      match p.inner() {
        Error::KeyCollision(kp) => {
          assert_eq!(kp.key(), "encoder.conv1.weight");
          assert_eq!(
            kp.context(),
            "WhisperModel::load: duplicate tensor key across shards"
          );
        }
        other => panic!("inner must be KeyCollision, got {other:?}"),
      }
    }
    other => panic!("expected LayerKeyed(KeyCollision) for a cross-shard duplicate, got {other:?}"),
  }

  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_all_safetensors_merges_disjoint_shards() {
  // The normal multi-shard case: two shards with DISJOINT keys merge into one
  // map carrying every tensor (no spurious collision, no dropped tensor).
  let dir = fresh_dir("disjoint-shard");

  let mut a: HashMap<String, Array> = HashMap::new();
  a.insert(
    "encoder.conv1.weight".to_string(),
    Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap(),
  );
  let mut b: HashMap<String, Array> = HashMap::new();
  b.insert(
    "decoder.ln.weight".to_string(),
    Array::from_slice::<f32>(&[3.0_f32, 4.0, 5.0], &(3usize,)).unwrap(),
  );
  write_shard(&dir, "a", &a);
  write_shard(&dir, "b", &b);

  let mut merged = load_all_safetensors(&dir).unwrap();
  assert_eq!(merged.len(), 2, "both disjoint shards merge");
  let conv = merged.get_mut("encoder.conv1.weight").unwrap();
  assert_eq!(conv.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
  let ln = merged.get_mut("decoder.ln.weight").unwrap();
  assert_eq!(ln.to_vec::<f32>().unwrap(), vec![3.0, 4.0, 5.0]);

  let _ = std::fs::remove_dir_all(&dir);
}

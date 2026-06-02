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
fn sanitize_hf_carries_scales_biases_siblings_through_remap() {
  // A quantized HF checkpoint stores `<layer>.weight` (packed) alongside
  // `<layer>.scales` / `<layer>.biases`. The HF→MLX key-remap keys on the
  // `<module>.` prefix (not the `.weight` leaf), so the `.scales` / `.biases`
  // siblings must land on the SAME remapped layer as their `.weight` — else a
  // quantized checkpoint would lose its scales/biases and fail to load.
  let mut w = HashMap::new();
  for leaf in ["weight", "scales", "biases"] {
    w.insert(
      format!("model.encoder.layers.0.self_attn.q_proj.{leaf}"),
      Array::zeros::<f32>(&(2usize, 2usize)).unwrap(),
    );
    w.insert(
      format!("model.decoder.embed_tokens.{leaf}"),
      Array::zeros::<f32>(&(4usize, 2usize)).unwrap(),
    );
  }
  let out = sanitize_map(w);
  // q_proj → attn.query: all three siblings remapped together.
  assert!(out.contains_key("encoder.blocks.0.attn.query.weight"));
  assert!(out.contains_key("encoder.blocks.0.attn.query.scales"));
  assert!(out.contains_key("encoder.blocks.0.attn.query.biases"));
  // embed_tokens → token_embedding: the (tied) quantized embedding triple too.
  assert!(out.contains_key("decoder.token_embedding.weight"));
  assert!(out.contains_key("decoder.token_embedding.scales"));
  assert!(out.contains_key("decoder.token_embedding.biases"));
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

// ───────────────────── quantized-checkpoint loading ─────────────────────
//
// No local 8-bit whisper checkpoint is available
// (`MLXRS_WHISPER_MODEL_DIR`/`models/` carry only a dense `whisper-large-v3`),
// so the quantized load path is covered by a SYNTHETIC quantized checkpoint:
// a tiny config whose `n_state` is divisible by the affine `group_size` (mlx
// requires `group_size ∈ {32, 64, 128}`, `mlx/ops.cpp:4740`), with every
// attention/MLP `Linear` weight and the token-embedding weight replaced by the
// real `ops::quantized::quantize` `(weight, scales, biases)` triple — the exact
// on-disk layout an mlx-community 8-bit checkpoint ships. The model must then
// construct (building `QuantizedLinear` / quantized `Embedding` layers) and run
// a full encode + decode forward to finite logits.

/// The affine group size the synthetic quantized fixtures use (a valid
/// mlx group size — `mlx/ops.cpp:4740`).
const QGROUP: i32 = 64;
/// 8-bit affine — the `whisper-large-v3-turbo-8bit` scheme.
const QBITS: i32 = 8;

/// Tiny dims whose width (`n_state = QGROUP = 64`) is divisible by the affine
/// `group_size`, so every quantized weight's last axis (`n_state`,
/// `4*n_state`) is a whole number of groups. `n_audio_ctx` stays the fixed
/// `N_FRAMES / 2`.
fn quant_dims() -> ModelDimensions {
  let n_state = QGROUP as usize; // 64
  ModelDimensions::new(
    TINY.n_mels,
    TINY.n_audio_ctx,
    n_state,
    TINY.n_head,
    1, // n_audio_layer
    TINY.n_vocab,
    TINY.n_text_ctx,
    n_state,
    TINY.n_head,
    1, // n_text_layer
  )
  .unwrap()
}

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` affine triple (`<prefix>.weight` packed +
/// `<prefix>.scales` + `<prefix>.biases`), mirroring how an mlx-community
/// quantized checkpoint stores a quantized `Linear` / `Embedding`. The
/// `<prefix>.bias` (dense output bias), if any, is left untouched.
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .expect("dense weight present");
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// Build a synthetic MLX-format quantized Whisper checkpoint: the dense
/// `quant_dims()` weights, then every attention/MLP `Linear` weight and the
/// token-embedding weight quantized to the 8-bit affine triple. Conv weights,
/// LayerNorms, and the positional embedding stay dense (the `class_predicate`
/// in `whisper.py:674-676` only quantizes `nn.Linear` / `nn.Embedding`).
fn quant_weights() -> HashMap<String, Array> {
  let n = QGROUP as usize; // n_state = 64
  let mut w = HashMap::new();

  // conv front-end + encoder ln_post (DENSE).
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

  // decoder embeddings + block + final ln (DENSE first).
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

  // Now quantize every Linear projection and the token embedding.
  for stack in ["encoder.blocks.0", "decoder.blocks.0"] {
    // self-attention projections.
    for p in ["query", "key", "value", "out"] {
      quantize_weight_in_place(&mut w, &format!("{stack}.attn.{p}"));
    }
    // MLP projections.
    quantize_weight_in_place(&mut w, &format!("{stack}.mlp1"));
    quantize_weight_in_place(&mut w, &format!("{stack}.mlp2"));
  }
  // decoder cross-attention projections.
  for p in ["query", "key", "value", "out"] {
    quantize_weight_in_place(&mut w, &format!("decoder.blocks.0.cross_attn.{p}"));
  }
  // the weight-tied token embedding (an `nn.Embedding`, also quantized).
  quantize_weight_in_place(&mut w, "decoder.token_embedding");

  w
}

/// The parsed global 8-bit affine quantization config for the synthetic
/// checkpoint (the analogue of the `config.json` `quantization` block).
fn quant_config() -> crate::lm::quant::PerLayerQuantization {
  crate::lm::quant::PerLayerQuantization::from_global(crate::lm::quant::Quantization::affine(
    QGROUP, QBITS,
  ))
}

#[test]
fn from_weights_quantized_builds_quantized_layers() {
  // With a quantization config and a checkpoint whose Linear/Embedding weights
  // carry `.scales`/`.biases`, the model builds quantized layers (and still
  // builds — a packed `uint32` weight of a DIFFERENT shape than the dense
  // `(out, in)` would otherwise be rejected by the dense shape gate).
  let model = WhisperModel::from_weights_quantized(
    quant_dims(),
    quant_weights(),
    Dtype::F32,
    Some(&quant_config()),
  )
  .unwrap();
  assert_eq!(model.dims().n_text_state(), QGROUP as usize);
}

#[test]
fn from_weights_quantized_runs_forward_to_finite_logits() {
  // The real GOAL contract on a synthetic stand-in: an 8-bit checkpoint loads
  // AND runs a full encode + decode forward to FINITE logits (the quantized
  // attention/MLP `quantized_matmul` and the quantized weight-tied logit head
  // all execute through mlx-c).
  let model = WhisperModel::from_weights_quantized(
    quant_dims(),
    quant_weights(),
    Dtype::F32,
    Some(&quant_config()),
  )
  .unwrap();

  // The mel front-end is dense; the encoder's attention/MLP are quantized.
  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();
  assert_eq!(enc.shape(), vec![1, TINY.n_audio_ctx, QGROUP as usize]);

  // Decode a couple of steps through the quantized decoder + quantized
  // weight-tied logit head.
  let mut cache = model.new_cache();
  let mut logits = model.decode_step(&mut cache, &enc, &[0u32]).unwrap();
  assert_eq!(logits.shape(), vec![TINY.n_vocab]);
  for v in logits.to_vec::<f32>().unwrap() {
    assert!(
      v.is_finite(),
      "quantized decode produced a non-finite logit: {v}"
    );
  }

  let mut logits2 = model.decode_step(&mut cache, &enc, &[0u32, 1u32]).unwrap();
  assert_eq!(logits2.shape(), vec![TINY.n_vocab]);
  for v in logits2.to_vec::<f32>().unwrap() {
    assert!(v.is_finite(), "quantized warm-step logit non-finite: {v}");
  }
}

#[test]
fn from_weights_quantized_dense_checkpoint_unchanged() {
  // A NON-quantized checkpoint loads identically whether or not a quantization
  // config is threaded (the `.scales` sibling is the load-bearing signal; a
  // dense checkpoint has none, so the dense path runs regardless).
  let with_cfg =
    WhisperModel::from_weights_quantized(dims(), tiny_weights(), Dtype::F32, Some(&quant_config()))
      .unwrap();
  // Produces logits exactly like the plain `from_weights` path.
  let mel = tiny_mel();
  let enc = with_cfg.encode(&mel).unwrap();
  let mut cache = with_cfg.new_cache();
  let logits = with_cfg.decode_step(&mut cache, &enc, &[0u32]).unwrap();
  assert_eq!(logits.shape(), vec![TINY.n_vocab]);
}

#[test]
fn from_weights_quantized_scales_without_config_errors() {
  // Weights say quantized (`.scales` present) but no quantization config
  // resolved scheme params → a typed InvariantViolation, not a silent wrong
  // load. (Passing `None` for `quantization` means the dense path is taken,
  // which then hits the packed `uint32` weight with the dense `(out, in)`
  // shape gate — also a typed error. To exercise the explicit
  // scales-but-no-params guard, thread an empty per-layer config with no
  // global default so `quantization_for` returns None for the quantized
  // layer.)
  let empty_cfg =
    crate::lm::quant::PerLayerQuantization::new(None, std::collections::HashMap::new());
  let err = WhisperModel::from_weights_quantized(
    quant_dims(),
    quant_weights(),
    Dtype::F32,
    Some(&empty_cfg),
  )
  .unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for `.scales` present but no resolved params, got {err:?}"
  );
}

/// Replace `<prefix>.{weight,scales,biases}` in `w` with the 8-bit affine
/// triple of an `(out, in)` dense ramp — for splicing a quantized layer of a
/// DELIBERATELY WRONG logical shape into an otherwise-valid checkpoint. `in`
/// must be a whole multiple of `QGROUP` so the affine quantize is well-formed.
fn splice_quantized(w: &mut HashMap<String, Array>, prefix: &str, out: usize, in_features: usize) {
  let mut data = Vec::with_capacity(out * in_features);
  for o in 0..out {
    for i in 0..in_features {
      data.push(((o * 7 + i) as f32) * 0.001);
    }
  }
  let dense = Array::from_slice::<f32>(&data, &(out, in_features)).unwrap();
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_output_width_linear() {
  // A quantized attention `query` whose packed weight unpacks to the WRONG
  // logical output dim must be rejected at load time (the quantized path now
  // reaches the same config-shape gate the dense `take_shaped` enforces),
  // before any forward sizes the projection from the checkpoint tensor.
  let mut w = quant_weights();
  // Config expects (n_state, n_state) = (64, 64); splice a (96, 64) instead.
  splice_quantized(&mut w, "encoder.blocks.0.attn.query", 96, QGROUP as usize);
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed shape error for a wrong-output-width quantized Linear, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_input_width_linear() {
  // A quantized projection whose packed weight unpacks to the WRONG logical
  // INPUT width (the `quantized_matmul` contraction dim) is rejected too — a
  // valid multiple of `group_size` (128) that disagrees with the config's 64.
  let mut w = quant_weights();
  splice_quantized(
    &mut w,
    "encoder.blocks.0.attn.value",
    QGROUP as usize,
    2 * QGROUP as usize,
  );
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed shape error for a wrong-input-width quantized Linear, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_vocab_width_embedding() {
  // A quantized token embedding whose packed table unpacks to the WRONG vocab
  // dim (leading axis) is rejected — the weight-tied logit head would otherwise
  // emit `n_vocab` sized by the tensor, not the validated config.
  let mut w = quant_weights();
  // Config expects (n_vocab, n_state) = (8, 64); splice (16, 64).
  splice_quantized(&mut w, "decoder.token_embedding", 16, QGROUP as usize);
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed shape error for a wrong-vocab quantized embedding, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_state_width_embedding() {
  // A quantized token embedding whose packed table unpacks to the WRONG state
  // (input) width is rejected — (n_vocab=8, 128) vs the config's (8, 64).
  let mut w = quant_weights();
  splice_quantized(
    &mut w,
    "decoder.token_embedding",
    TINY.n_vocab,
    2 * QGROUP as usize,
  );
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed shape error for a wrong-state quantized embedding, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_affine_embedding_without_biases() {
  // A quantized token embedding under an `affine` config that is MISSING its
  // `.biases` sibling must be rejected at load time (affine requires per-group
  // biases): the builder's embedding path now runs the `Embedding::quantized`
  // structural gate, so the malformed triple is a typed error before any forward
  // reaches the mlx-c `dequantize` / weight-tied `quantized_matmul`.
  let mut w = quant_weights();
  // The token embedding is affine-quantized by `quant_weights`; drop its
  // `.biases` to model a malformed (or scale-only-with-affine-config) checkpoint.
  w.remove("decoder.token_embedding.biases")
    .expect("affine token-embedding biases present");
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for an affine quantized embedding missing .biases, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_mis_shaped_embedding_biases() {
  // A quantized token embedding whose `.biases` shape disagrees with `.scales`
  // is rejected at load — `affine_quantize` writes both with the identical
  // `(n_vocab, n_groups)` shape, so a divergent `.biases` is a malformed
  // checkpoint the builder's `Embedding::quantized` gate catches before the
  // first per-row `dequantize`.
  let mut w = quant_weights();
  // Replace the token-embedding `.biases` with a wrong-shaped one (a (1,) vector
  // in place of the `(n_vocab, n_groups)` per-group biases).
  w.insert(
    "decoder.token_embedding.biases".to_string(),
    Array::from_slice::<f32>(&[0.5], &(1usize,)).unwrap(),
  );
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_) | Error::RankMismatch(_)),
    "expected a shape/rank mismatch for a mis-shaped quantized embedding .biases, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_ignores_stale_bias_on_keyless_projection() {
  // The Whisper attention `key` projection is built `bias = false`. The dense
  // path leaves any stray `<prefix>.bias` unused; the quantized path must be
  // consistent and NOT adopt it (a wrongly-adopted bias of the wrong length
  // would otherwise be rejected by `QuantizedLinear::from_parts`' length gate,
  // failing the load). A deliberately wrong-length stray `key.bias` therefore
  // distinguishes the two: it must STILL load (the bias is ignored).
  let mut w = quant_weights();
  // `key` out dim is n_state = 64; splice a stray length-1 bias the dense path
  // would ignore. If the quantized path adopted it, the length check would fire.
  w.insert(
    "encoder.blocks.0.attn.key.bias".to_string(),
    Array::from_slice::<f32>(&[3.5], &(1usize,)).unwrap(),
  );
  let model =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .expect("a stray bias on the bias=false key projection must be ignored, not adopted");
  // And the model still runs a forward to finite logits (the key projection
  // carries no bias, exactly as the dense path).
  let enc = model.encode(&tiny_mel()).unwrap();
  let mut cache = model.new_cache();
  let mut logits = model.decode_step(&mut cache, &enc, &[0u32]).unwrap();
  for v in logits.to_vec::<f32>().unwrap() {
    assert!(
      v.is_finite(),
      "stale-bias-ignored decode logit non-finite: {v}"
    );
  }
}

#[test]
fn from_weights_quantized_rejects_missing_query_bias() {
  // A quantized attention `query` projection is built `bias = true`. mlx's
  // `QuantizedLinear.from_linear` preserves the source `Linear.bias`, so a
  // faithful quantized checkpoint carries `query.bias`; one missing it is
  // malformed and must FAIL fast with the SAME typed `MissingKey` the dense
  // path returns — not load a silently biasless projection that corrupts logits.
  let mut w = quant_weights();
  w.remove("encoder.blocks.0.attn.query.bias")
    .expect("query.bias present in the faithful quantized checkpoint");
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(&err, Error::MissingKey(p) if p.key().contains("encoder.blocks.0.attn.query.bias")),
    "expected MissingKey(query.bias) for a quantized biasful projection missing its bias, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_missing_mlp1_bias() {
  // Same arity contract on the MLP: the `mlp1` linear is `bias = true`, so a
  // quantized checkpoint missing `mlp1.bias` is malformed and fails fast with
  // the typed `MissingKey`, identical to the dense path.
  let mut w = quant_weights();
  w.remove("encoder.blocks.0.mlp1.bias")
    .expect("mlp1.bias present in the faithful quantized checkpoint");
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  assert!(
    matches!(&err, Error::MissingKey(p) if p.key().contains("encoder.blocks.0.mlp1.bias")),
    "expected MissingKey(mlp1.bias) for a quantized biasful MLP linear missing its bias, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_shape_bias() {
  // A quantized biasful projection whose dense `<prefix>.bias` has the wrong
  // shape is rejected at load with the SAME keyed `ShapePairMismatch` the dense
  // path returns — the bias is validated `(out,)` before it reaches the
  // constructor, never broadcast silently across every output channel.
  let mut w = quant_weights();
  // `value` out dim is n_state = 64; ship a wrong-length (1,) bias.
  w.insert(
    "encoder.blocks.0.attn.value.bias".to_string(),
    Array::from_slice::<f32>(&[0.25], &(1usize,)).unwrap(),
  );
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&quant_config()))
      .unwrap_err();
  match &err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "encoder.blocks.0.attn.value.bias");
      assert!(matches!(p.inner(), Error::ShapePairMismatch(_)));
    }
    other => panic!(
      "expected LayerKeyed(ShapePairMismatch) for a wrong-shape quantized bias, got {other:?}"
    ),
  }
}

// ───────────── HF-format per-layer quantization override ─────────────────
//
// An HF-format quantized checkpoint ships RAW HF weight keys
// (`model.encoder.layers.0.self_attn.q_proj.weight`) AND a quantization config
// whose per-layer override keys are RAW HF paths. `sanitize` remaps the weight
// keys into MLX-style prefixes (`encoder.blocks.0.attn.query`), and the builder
// resolves the per-layer scheme against those SANITIZED prefixes — so the
// per-layer config keys must be normalized through the same HF→MLX transform
// (`normalize_quant_keys`) or a per-layer override silently misses and the
// builder falls back to the GLOBAL scheme. These tests build HF checkpoints
// where one layer is quantized with params that DIFFER from the global and
// assert the override is honored after sanitize (the layer loads under its own
// `group_size`/`bits`, which its packed `(weight, scales)` shapes require).

/// The override `group_size` one Linear is quantized at (≠ the global
/// [`QGROUP`] = 64). A valid mlx affine group size that still divides the tiny
/// `n_state` = 64 width (`mlx/ops.cpp:4740`).
const QGROUP_OVERRIDE: i32 = 32;
/// The override `bits` the token embedding is quantized at (≠ the global
/// [`QBITS`] = 8).
const QBITS_OVERRIDE: i32 = 4;

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` affine triple at EXPLICIT `(group_size, bits)` —
/// the parameterized form of [`quantize_weight_in_place`], for splicing a layer
/// quantized under a per-layer override that differs from the global scheme.
fn quantize_weight_in_place_with(
  w: &mut HashMap<String, Array>,
  prefix: &str,
  group_size: i32,
  bits: i32,
) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .expect("dense weight present");
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, group_size, bits, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// Insert one HF-named attention sub-module (`q_proj`/`k_proj`/`v_proj`/
/// `out_proj`) under the HF `<prefix>` (e.g.
/// `model.encoder.layers.0.self_attn`). `k_proj` carries no bias (Whisper's
/// `bias = false` key projection), matching [`put_attn`]'s MLX layout.
fn put_attn_hf(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  for p in ["q_proj", "v_proj", "out_proj"] {
    w.insert(format!("{prefix}.{p}.weight"), ones2(n, n));
    w.insert(format!("{prefix}.{p}.bias"), zeros1(n));
  }
  w.insert(format!("{prefix}.k_proj.weight"), ones2(n, n));
}

/// Insert one HF-named residual block under the HF `<prefix>` (e.g.
/// `model.decoder.layers.0`), the HF-leaf-named twin of [`put_block`]: the
/// `self_attn` (+ `self_attn_layer_norm`), the `fc1`/`fc2` MLP (+
/// `final_layer_norm`), and — for the decoder — the `encoder_attn` cross
/// attention (+ `encoder_attn_layer_norm`).
fn put_block_hf(w: &mut HashMap<String, Array>, prefix: &str, n: usize, cross: bool) {
  put_attn_hf(w, &format!("{prefix}.self_attn"), n);
  put_ln(w, &format!("{prefix}.self_attn_layer_norm"), n);
  if cross {
    put_attn_hf(w, &format!("{prefix}.encoder_attn"), n);
    put_ln(w, &format!("{prefix}.encoder_attn_layer_norm"), n);
  }
  w.insert(format!("{prefix}.fc1.weight"), ones2(4 * n, n));
  w.insert(format!("{prefix}.fc1.bias"), zeros1(4 * n));
  w.insert(format!("{prefix}.fc2.weight"), ones2(n, 4 * n));
  w.insert(format!("{prefix}.fc2.bias"), zeros1(n));
  put_ln(w, &format!("{prefix}.final_layer_norm"), n);
}

/// The HF-format quantizable layer prefixes (the post-sanitize MLX twins are in
/// [`quant_weights`]): every `Linear` projection plus the token embedding.
fn hf_quant_prefixes() -> Vec<String> {
  let mut prefixes = Vec::new();
  for (stack, cross) in [
    ("model.encoder.layers.0", false),
    ("model.decoder.layers.0", true),
  ] {
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      prefixes.push(format!("{stack}.self_attn.{p}"));
    }
    prefixes.push(format!("{stack}.fc1"));
    prefixes.push(format!("{stack}.fc2"));
    if cross {
      for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        prefixes.push(format!("{stack}.encoder_attn.{p}"));
      }
    }
  }
  prefixes.push("model.decoder.embed_tokens".to_string());
  prefixes
}

/// Build a synthetic **HF-format** quantized Whisper checkpoint (raw HF weight
/// keys), structurally the HF twin of [`quant_weights`]: the conv front-end is
/// in HF `(out, in, kernel)` layout, and every quantizable layer is quantized
/// to the global [`QGROUP`]/[`QBITS`] affine triple — EXCEPT the layer at
/// `override_hf_prefix`, which is quantized at `(override_gs, override_bits)`
/// (the per-layer override). The returned map is what an HF-named mixed-quant
/// checkpoint ships before `sanitize`.
fn hf_quant_weights(
  override_hf_prefix: &str,
  override_gs: i32,
  override_bits: i32,
) -> HashMap<String, Array> {
  let n = QGROUP as usize; // n_state = 64
  let mut w = HashMap::new();

  // conv front-end (HF `(out, in, kernel)` layout) + encoder ln (HF name).
  w.insert(
    "model.encoder.conv1.weight".to_string(),
    Array::ones::<f32>(&(n, TINY.n_mels, 3usize)).unwrap(),
  );
  w.insert("model.encoder.conv1.bias".to_string(), zeros1(n));
  w.insert(
    "model.encoder.conv2.weight".to_string(),
    Array::ones::<f32>(&(n, n, 3usize)).unwrap(),
  );
  w.insert("model.encoder.conv2.bias".to_string(), zeros1(n));
  put_block_hf(&mut w, "model.encoder.layers.0", n, false);
  put_ln(&mut w, "model.encoder.layer_norm", n);

  // decoder embeddings (HF names) + block + final ln. The HF encoder positional
  // embedding is present and DROPPED by sanitize (recomputed via sinusoids).
  w.insert(
    "model.decoder.embed_tokens.weight".to_string(),
    ones2(TINY.n_vocab, n),
  );
  w.insert(
    "model.encoder.embed_positions.weight".to_string(),
    ones2(TINY.n_audio_ctx, n),
  );
  w.insert(
    "model.decoder.embed_positions.weight".to_string(),
    ones2(TINY.n_text_ctx, n),
  );
  put_block_hf(&mut w, "model.decoder.layers.0", n, true);
  put_ln(&mut w, "model.decoder.layer_norm", n);

  // Quantize every Linear + the token embedding. The override layer takes its
  // per-layer params; all others take the global scheme.
  for prefix in hf_quant_prefixes() {
    if prefix == override_hf_prefix {
      quantize_weight_in_place_with(&mut w, &prefix, override_gs, override_bits);
    } else {
      quantize_weight_in_place_with(&mut w, &prefix, QGROUP, QBITS);
    }
  }
  w
}

/// A global-affine [`PerLayerQuantization`] (global [`QGROUP`]/[`QBITS`]) plus a
/// single per-layer override keyed by the RAW HF path `hf_layer` →
/// `(override_gs, override_bits)` — the analogue of an HF `config.json`
/// `quantization` block carrying a per-layer entry.
fn hf_quant_config_with_override(
  hf_layer: &str,
  override_gs: i32,
  override_bits: i32,
) -> crate::lm::quant::PerLayerQuantization {
  let mut per_layer = HashMap::new();
  per_layer.insert(
    hf_layer.to_string(),
    crate::lm::quant::QuantizationOption::Quantize(crate::lm::quant::Quantization::affine(
      override_gs,
      override_bits,
    )),
  );
  crate::lm::quant::PerLayerQuantization::new(
    Some(crate::lm::quant::Quantization::affine(QGROUP, QBITS)),
    per_layer,
  )
}

#[test]
fn normalize_quant_keys_remaps_hf_per_layer_paths_to_sanitized() {
  // The per-layer override map's RAW HF keys are normalized into the SANITIZED
  // MLX namespace the builder resolves against — a Linear path and the token
  // embedding path, the two cases the regression tests below exercise end to
  // end. An already-MLX config is returned unchanged (idempotent).
  let cfg = hf_quant_config_with_override(
    "model.encoder.layers.0.self_attn.q_proj",
    QGROUP_OVERRIDE,
    QBITS,
  );
  let normalized = normalize_quant_keys(&cfg).unwrap();
  // The HF Linear path now resolves under its sanitized MLX prefix.
  assert_eq!(
    normalized.quantization_for("encoder.blocks.0.attn.query"),
    Some(crate::lm::quant::Quantization::affine(
      QGROUP_OVERRIDE,
      QBITS
    )),
    "HF q_proj override must resolve against the sanitized attn.query prefix"
  );
  // The raw HF key no longer resolves to the override (it was remapped, not
  // duplicated) — it falls back to the global default.
  assert_eq!(
    normalized.quantization_for("model.encoder.layers.0.self_attn.q_proj"),
    Some(crate::lm::quant::Quantization::affine(QGROUP, QBITS)),
    "the raw HF key must no longer carry the override after normalization"
  );

  // The token-embedding HF path maps onto the sanitized token_embedding prefix.
  let emb_cfg = hf_quant_config_with_override("model.decoder.embed_tokens", QGROUP, QBITS_OVERRIDE);
  let emb_norm = normalize_quant_keys(&emb_cfg).unwrap();
  assert_eq!(
    emb_norm.quantization_for("decoder.token_embedding"),
    Some(crate::lm::quant::Quantization::affine(
      QGROUP,
      QBITS_OVERRIDE
    )),
    "HF embed_tokens override must resolve against the sanitized token_embedding prefix"
  );

  // An already-MLX config passes through unchanged: normalization is
  // unconditional, and `remap_hf_key` is idempotent on MLX-native keys, so the
  // map is preserved.
  let mlx_cfg = quant_config();
  let mlx_norm = normalize_quant_keys(&mlx_cfg).unwrap();
  assert_eq!(
    mlx_norm, mlx_cfg,
    "an MLX-format config must pass through unchanged"
  );
}

#[test]
fn from_weights_quantized_honors_hf_per_layer_linear_override() {
  // (a) ONE Linear (attention q_proj) carries a per-layer override whose
  // `group_size` (32) differs from the global (64). The q_proj weight is
  // quantized at group_size=32, so its `.scales` has `n_state / 32 = 2` groups;
  // if the override were LOST and the builder fell back to the global
  // group_size=64, `check_quantized_shape` would recover `2 * 64 = 128 ≠ 64`
  // and reject the (valid) checkpoint. A successful build therefore proves the
  // HF-keyed override is honored against the sanitized `attn.query` prefix.
  let hf_layer = "model.encoder.layers.0.self_attn.q_proj";
  let w = hf_quant_weights(hf_layer, QGROUP_OVERRIDE, QBITS);
  let cfg = hf_quant_config_with_override(hf_layer, QGROUP_OVERRIDE, QBITS);
  let model = WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg))
    .expect("HF per-layer Linear override (group_size 32) must be honored after sanitize");
  assert_eq!(model.dims().n_text_state(), QGROUP as usize);

  // Control: WITHOUT the per-layer entry (global-only config), the q_proj — still
  // quantized at group_size=32 on disk — is resolved at the global group_size=64
  // and rejected. This pins that the override is genuinely load-bearing (the
  // layer cannot load under the global scheme), so the positive build above is
  // not a false pass from the global scheme coincidentally fitting.
  let w_again = hf_quant_weights(hf_layer, QGROUP_OVERRIDE, QBITS);
  let global_only = quant_config();
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w_again, Dtype::F32, Some(&global_only))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "a group_size-32 q_proj resolved under the global group_size-64 must be rejected, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_honors_hf_per_layer_embedding_override() {
  // (b) The token embedding (`decoder.embed_tokens`) carries a per-layer
  // override whose `bits` (4) differ from the global (8). The embedding weight
  // is packed at bits=4, so its `uint32` last dim is `n_state * 4 / 32 = 8`; if
  // the override were lost and the builder fell back to bits=8,
  // `check_quantized_shape` would recover `8 * 32 / 8 = 32 ≠ 64` and reject the
  // valid weight-tied table. A successful build (and finite logits through the
  // quantized weight-tied logit head) proves the HF-keyed embedding override is
  // honored against the sanitized `decoder.token_embedding` prefix.
  let hf_layer = "model.decoder.embed_tokens";
  let w = hf_quant_weights(hf_layer, QGROUP, QBITS_OVERRIDE);
  let cfg = hf_quant_config_with_override(hf_layer, QGROUP, QBITS_OVERRIDE);
  let model = WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg))
    .expect("HF per-layer embedding override (bits 4) must be honored after sanitize");

  // The quantized weight-tied logit head (the bits-4 embedding) runs to finite
  // logits, confirming the override scheme drives the real forward, not just the
  // load-time shape gate.
  let enc = model.encode(&tiny_mel()).unwrap();
  let mut cache = model.new_cache();
  let mut logits = model.decode_step(&mut cache, &enc, &[0u32]).unwrap();
  assert_eq!(logits.shape(), vec![TINY.n_vocab]);
  for v in logits.to_vec::<f32>().unwrap() {
    assert!(
      v.is_finite(),
      "bits-4 embedding-override logit non-finite: {v}"
    );
  }

  // Control: the global-only config (bits=8) cannot load the bits-4 table.
  let w_again = hf_quant_weights(hf_layer, QGROUP, QBITS_OVERRIDE);
  let global_only = quant_config();
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w_again, Dtype::F32, Some(&global_only))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "a bits-4 token embedding resolved under the global bits-8 must be rejected, got {err:?}"
  );
}

#[test]
fn sanitize_hf_quant_weights_reproduces_mlx_quant_layer_set() {
  // Independent oracle for the HF builder: `sanitize`-ing the HF-format
  // checkpoint must yield exactly the same KEY SET as the MLX-format
  // `quant_weights` fixture (the dropped encoder positional embedding aside),
  // proving the HF leaf names + the override splice land on the sanitized MLX
  // prefixes the builder consumes — independent of the quant resolution under
  // test. (Both fixtures quantize the q_proj at the global scheme here, so the
  // packed shapes match; only the KEY SET is asserted, not the tensor values.)
  let hf = hf_quant_weights("model.encoder.layers.0.self_attn.q_proj", QGROUP, QBITS);
  let (sanitized, is_hf) = sanitize(hf).unwrap();
  assert!(is_hf, "a `model.`-prefixed checkpoint is HF format");

  let mlx_keys: std::collections::BTreeSet<String> = quant_weights().into_keys().collect();
  let sanitized_keys: std::collections::BTreeSet<String> = sanitized.into_keys().collect();
  assert_eq!(
    sanitized_keys, mlx_keys,
    "sanitized HF quant checkpoint must match the MLX quant-checkpoint key set"
  );
}

/// Build a [`PerLayerQuantization`] whose per-layer map carries BOTH the raw HF
/// path and the already-sanitized MLX alias for the SAME layer — the raw key
/// remaps onto the sanitized key, so the two collide post-normalization. Each
/// alias gets its own override `(group_size, bits)`; passing equal params for
/// both yields an identical (harmless) collision, unequal params a conflict.
fn hf_quant_config_with_aliased_collision(
  hf_layer: &str,
  mlx_layer: &str,
  hf_gs: i32,
  hf_bits: i32,
  mlx_gs: i32,
  mlx_bits: i32,
) -> crate::lm::quant::PerLayerQuantization {
  let mut per_layer = HashMap::new();
  per_layer.insert(
    hf_layer.to_string(),
    crate::lm::quant::QuantizationOption::Quantize(crate::lm::quant::Quantization::affine(
      hf_gs, hf_bits,
    )),
  );
  per_layer.insert(
    mlx_layer.to_string(),
    crate::lm::quant::QuantizationOption::Quantize(crate::lm::quant::Quantization::affine(
      mlx_gs, mlx_bits,
    )),
  );
  crate::lm::quant::PerLayerQuantization::new(
    Some(crate::lm::quant::Quantization::affine(QGROUP, QBITS)),
    per_layer,
  )
}

#[test]
fn normalize_quant_keys_rejects_conflicting_aliased_per_layer_override() {
  // A mixed config names the SAME layer twice — once by its raw HF path
  // (`model.encoder.layers.0.self_attn.q_proj`, which `normalize_quant_keys`
  // remaps onto `encoder.blocks.0.attn.query`) and once by that sanitized MLX
  // alias directly — carrying DIFFERENT `group_size`s (64 vs 32). The two
  // overrides for one layer disagree, so the per-layer scheme is ambiguous and
  // normalization must fail closed with a typed `KeyCollision` naming the
  // sanitized layer key — never silently pick an arbitrary `HashMap`-order
  // survivor that could load the checkpoint with the wrong scheme.
  let cfg = hf_quant_config_with_aliased_collision(
    "model.encoder.layers.0.self_attn.q_proj",
    "encoder.blocks.0.attn.query",
    QGROUP,
    QBITS,
    QGROUP_OVERRIDE,
    QBITS,
  );
  let err = normalize_quant_keys(&cfg).unwrap_err();
  match &err {
    Error::KeyCollision(p) => {
      assert_eq!(
        p.key(),
        "encoder.blocks.0.attn.query",
        "the conflict error must name the colliding sanitized layer key"
      );
    }
    other => panic!("expected Error::KeyCollision for a conflicting alias, got {other:?}"),
  }

  // The verdict is order-independent: the source map is iterated in an arbitrary
  // order, but a conflicting collision ALWAYS errors. Re-running converges on the
  // same typed error regardless of which alias the iterator visits first.
  for _ in 0..8 {
    let cfg = hf_quant_config_with_aliased_collision(
      "model.encoder.layers.0.self_attn.q_proj",
      "encoder.blocks.0.attn.query",
      QGROUP,
      QBITS,
      QGROUP_OVERRIDE,
      QBITS,
    );
    assert!(
      matches!(normalize_quant_keys(&cfg), Err(Error::KeyCollision(_))),
      "a conflicting aliased override must error on every iteration order"
    );
  }
}

#[test]
fn normalize_quant_keys_accepts_identical_aliased_per_layer_override() {
  // The same two aliases for one layer, but carrying IDENTICAL params (both
  // group_size=32, bits=8) — a harmless duplicate. Normalization must converge
  // on a SINGLE entry under the sanitized key with that scheme, with no error,
  // independent of which alias the source iterator visits first.
  for _ in 0..8 {
    let cfg = hf_quant_config_with_aliased_collision(
      "model.encoder.layers.0.self_attn.q_proj",
      "encoder.blocks.0.attn.query",
      QGROUP_OVERRIDE,
      QBITS,
      QGROUP_OVERRIDE,
      QBITS,
    );
    let normalized = normalize_quant_keys(&cfg).unwrap();
    // Exactly one surviving entry, keyed by the sanitized path.
    assert_eq!(
      normalized.per_layer_ref().len(),
      1,
      "identical aliases must collapse to one normalized entry"
    );
    assert_eq!(
      normalized.quantization_for("encoder.blocks.0.attn.query"),
      Some(crate::lm::quant::Quantization::affine(
        QGROUP_OVERRIDE,
        QBITS
      )),
      "the surviving entry must carry the (identical) override scheme"
    );
    // The raw HF key was remapped (not duplicated), so it falls back to global.
    assert_eq!(
      normalized.quantization_for("model.encoder.layers.0.self_attn.q_proj"),
      Some(crate::lm::quant::Quantization::affine(QGROUP, QBITS)),
      "the raw HF alias must not survive as a separate entry"
    );
  }
}

#[test]
fn from_weights_quantized_loads_identical_aliased_per_layer_override() {
  // End to end: a checkpoint whose config names one q_proj layer by BOTH its raw
  // HF path and its sanitized MLX alias with IDENTICAL override params (32/8)
  // loads cleanly — the harmless duplicate collapses to the single override and
  // the layer loads under that scheme (its group_size-32 `.scales` would be
  // rejected under the global group_size-64, so a successful build proves the
  // override survived).
  let hf_layer = "model.encoder.layers.0.self_attn.q_proj";
  let w = hf_quant_weights(hf_layer, QGROUP_OVERRIDE, QBITS);
  let cfg = hf_quant_config_with_aliased_collision(
    hf_layer,
    "encoder.blocks.0.attn.query",
    QGROUP_OVERRIDE,
    QBITS,
    QGROUP_OVERRIDE,
    QBITS,
  );
  let model = WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg))
    .expect("identical aliased override must collapse and load under its scheme");
  assert_eq!(model.dims().n_text_state(), QGROUP as usize);
}

#[test]
fn from_weights_quantized_rejects_conflicting_aliased_per_layer_override() {
  // End to end: the same aliased layer with CONFLICTING override params
  // (HF=64/8 vs MLX-alias=32/8) is ambiguous; the quantized builder must surface
  // the typed `KeyCollision` from normalization rather than load with an
  // arbitrary scheme.
  let hf_layer = "model.encoder.layers.0.self_attn.q_proj";
  let w = hf_quant_weights(hf_layer, QGROUP_OVERRIDE, QBITS);
  let cfg = hf_quant_config_with_aliased_collision(
    hf_layer,
    "encoder.blocks.0.attn.query",
    QGROUP,
    QBITS,
    QGROUP_OVERRIDE,
    QBITS,
  );
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg)).unwrap_err();
  assert!(
    matches!(err, Error::KeyCollision(_)),
    "a conflicting aliased per-layer override must fail the quantized build, got {err:?}"
  );
}

// ── config-key normalization is independent of the weight-map format ───────
//
// The config carries HF-named or MLX-named per-layer keys independently of how
// the WEIGHTS are named: an MLX-native weight map (`is_hf_format == false`) can
// still ship a `config.json` `quantization` block whose per-layer override keys
// are raw HF paths. Because normalization runs unconditionally, the override
// resolves to its sanitized layer and the collision check runs for EVERY
// config, not only the HF-weight path — so a raw-HF override against MLX weights
// is honored, a conflicting alias errors, and an identical alias converges.

/// Build a synthetic **MLX-format** quantized Whisper checkpoint (already-MLX
/// weight keys, no `model.` prefix → `is_hf_format == false`), structurally the
/// MLX-named twin of [`hf_quant_weights`]: every quantizable layer is quantized
/// to the global [`QGROUP`]/[`QBITS`] affine triple EXCEPT the layer at
/// `override_mlx_prefix`, which is quantized at `(override_gs, override_bits)`.
/// Identical to [`quant_weights`] but for the one parameterized override layer,
/// so a per-layer override against MLX weights is genuinely load-bearing (its
/// packed `(weight, scales)` shapes are rejected under the global scheme).
fn mlx_quant_weights(
  override_mlx_prefix: &str,
  override_gs: i32,
  override_bits: i32,
) -> HashMap<String, Array> {
  let mut w = quant_weights();
  if override_gs != QGROUP || override_bits != QBITS {
    // `quant_weights` already packed this layer at the global scheme; re-pack
    // its DENSE matrix at the override scheme. Rebuild the dense `(out, in)`
    // ones matrix the fixture started from (every quantizable Linear is square
    // `n_state x n_state` except the MLP and the `n_vocab x n_state` embedding),
    // dropping the global triple first so the override triple replaces it.
    let n = QGROUP as usize;
    let (out, in_features) = if override_mlx_prefix == "decoder.token_embedding" {
      (TINY.n_vocab, n)
    } else if override_mlx_prefix.ends_with(".mlp1") {
      (4 * n, n)
    } else if override_mlx_prefix.ends_with(".mlp2") {
      (n, 4 * n)
    } else {
      (n, n)
    };
    w.remove(&format!("{override_mlx_prefix}.weight"));
    w.remove(&format!("{override_mlx_prefix}.scales"));
    w.remove(&format!("{override_mlx_prefix}.biases"));
    w.insert(
      format!("{override_mlx_prefix}.weight"),
      ones2(out, in_features),
    );
    quantize_weight_in_place_with(&mut w, override_mlx_prefix, override_gs, override_bits);
  }
  w
}

/// A global-affine [`PerLayerQuantization`] plus a single per-layer override
/// keyed by the RAW HF path `hf_layer` → `(override_gs, override_bits)`, but
/// paired with an MLX-format weight map — exercising a config whose key
/// namespace differs from the (MLX-named) weights.
fn raw_hf_override_on_mlx_config(
  hf_layer: &str,
  override_gs: i32,
  override_bits: i32,
) -> crate::lm::quant::PerLayerQuantization {
  hf_quant_config_with_override(hf_layer, override_gs, override_bits)
}

#[test]
fn from_weights_quantized_honors_raw_hf_override_against_mlx_weights() {
  // (1) MLX-format weights (`is_hf_format == false`) paired with a config whose
  // per-layer override is keyed by the RAW HF path. Normalization runs
  // regardless of the weight-map format, so the raw HF key
  // (`model.encoder.layers.0.self_attn.q_proj`) is remapped onto the sanitized
  // `encoder.blocks.0.attn.query` the builder resolves against — the q_proj
  // loads under its group_size-32 override (its `.scales` has `n_state/32 = 2`
  // groups, which the global group_size-64 would reject). A successful build
  // proves the raw-HF override is honored through the non-HF-weight path.
  let mlx_layer = "encoder.blocks.0.attn.query";
  let hf_layer = "model.encoder.layers.0.self_attn.q_proj";
  let w = mlx_quant_weights(mlx_layer, QGROUP_OVERRIDE, QBITS);
  let cfg = raw_hf_override_on_mlx_config(hf_layer, QGROUP_OVERRIDE, QBITS);
  let model = WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg))
    .expect("a raw-HF per-layer override against MLX weights must resolve to the sanitized layer");
  assert_eq!(model.dims().n_text_state(), QGROUP as usize);

  // Control: the global-only config (group_size 64) cannot load the group_size-32
  // q_proj — proving the override is load-bearing, not a coincidental global fit.
  let w_again = mlx_quant_weights(mlx_layer, QGROUP_OVERRIDE, QBITS);
  let global_only = quant_config();
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w_again, Dtype::F32, Some(&global_only))
      .unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "a group_size-32 q_proj resolved under the global group_size-64 must be rejected, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_conflicting_alias_against_mlx_weights() {
  // (2) MLX-format weights paired with a config naming the SAME layer twice —
  // by its raw HF path AND its sanitized MLX alias — with CONFLICTING params
  // (64 vs 32). Even though the weights are MLX-native, the unconditional
  // normalization remaps the raw HF key onto the MLX alias, the collision check
  // fires, and the ambiguous per-layer scheme is rejected with a typed
  // `KeyCollision` rather than loading under an arbitrary survivor.
  let mlx_layer = "encoder.blocks.0.attn.query";
  let hf_layer = "model.encoder.layers.0.self_attn.q_proj";
  let w = mlx_quant_weights(mlx_layer, QGROUP_OVERRIDE, QBITS);
  let cfg = hf_quant_config_with_aliased_collision(
    hf_layer,
    mlx_layer,
    QGROUP,
    QBITS,
    QGROUP_OVERRIDE,
    QBITS,
  );
  let err =
    WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg)).unwrap_err();
  assert!(
    matches!(err, Error::KeyCollision(_)),
    "a conflicting alias against MLX weights must fail the quantized build, got {err:?}"
  );

  // Order-independent at the normalization layer: a conflicting alias errors on
  // every source-map iteration order (MLX weights do not change the verdict).
  for _ in 0..8 {
    let cfg = hf_quant_config_with_aliased_collision(
      hf_layer,
      mlx_layer,
      QGROUP,
      QBITS,
      QGROUP_OVERRIDE,
      QBITS,
    );
    assert!(
      matches!(normalize_quant_keys(&cfg), Err(Error::KeyCollision(_))),
      "a conflicting aliased override must error on every iteration order"
    );
  }
}

#[test]
fn from_weights_quantized_converges_identical_alias_against_mlx_weights() {
  // (3) MLX-format weights paired with a config naming the same layer by its raw
  // HF path AND its sanitized MLX alias with IDENTICAL params (32/8). The
  // harmless duplicate collapses to one normalized entry and the layer loads
  // under that single override scheme (its group_size-32 `.scales` would be
  // rejected under the global group_size-64, so a successful build proves the
  // override survived the collision merge).
  let mlx_layer = "encoder.blocks.0.attn.query";
  let hf_layer = "model.encoder.layers.0.self_attn.q_proj";
  let w = mlx_quant_weights(mlx_layer, QGROUP_OVERRIDE, QBITS);
  let cfg = hf_quant_config_with_aliased_collision(
    hf_layer,
    mlx_layer,
    QGROUP_OVERRIDE,
    QBITS,
    QGROUP_OVERRIDE,
    QBITS,
  );
  // The normalization collapses the alias pair to a single entry.
  let normalized = normalize_quant_keys(&cfg).unwrap();
  assert_eq!(
    normalized.per_layer_ref().len(),
    1,
    "identical aliases must collapse to one normalized entry"
  );
  let model = WhisperModel::from_weights_quantized(quant_dims(), w, Dtype::F32, Some(&cfg))
    .expect("an identical alias against MLX weights must collapse and load under its scheme");
  assert_eq!(model.dims().n_text_state(), QGROUP as usize);
}

#[test]
fn remap_hf_key_is_idempotent_on_mlx_keys() {
  // `normalize_quant_keys` normalizes UNCONDITIONALLY, so it must never corrupt
  // an already-sanitized MLX config key: `remap_hf_key` applied to an MLX-native
  // key (and to a `<key>.weight`, the form the config normalization feeds it)
  // must yield that same key — no `model.` prefix to strip, no HF-form `KEY_MAP`
  // left-hand side to match, and no MLX target that re-triggers another pattern.
  let mlx_keys = [
    "encoder.blocks.0.attn.query.weight",
    "encoder.blocks.0.attn.key.weight",
    "encoder.blocks.0.attn.value.weight",
    "encoder.blocks.0.attn.out.weight",
    "encoder.blocks.0.mlp1.weight",
    "encoder.blocks.0.mlp2.weight",
    "encoder.blocks.0.attn_ln.weight",
    "encoder.blocks.0.mlp_ln.weight",
    "encoder.ln_post.weight",
    "decoder.blocks.0.cross_attn.query.weight",
    "decoder.blocks.0.cross_attn_ln.weight",
    "decoder.token_embedding.weight",
    "decoder.positional_embedding",
    "decoder.ln.weight",
  ];
  for key in mlx_keys {
    assert_eq!(
      remap_hf_key(key.to_string()).as_deref(),
      Some(key),
      "remap_hf_key must be a no-op on the already-sanitized MLX key {key}"
    );
    // Idempotent under a second application, so unconditional normalization of a
    // config that already carries MLX paths never double-remaps.
    let once = remap_hf_key(key.to_string()).unwrap();
    assert_eq!(
      remap_hf_key(once.clone()).as_deref(),
      Some(once.as_str()),
      "remap_hf_key must converge (idempotent) on the MLX key {key}"
    );
  }
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

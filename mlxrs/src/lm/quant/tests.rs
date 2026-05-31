use super::*;

fn arr_f32(data: &[f32], shape: &[usize]) -> Array {
  Array::from_slice::<f32>(data, &shape).expect("from_slice")
}

/// Construct a packed-quantized `.weight` array (dtype `uint32`,
/// the layout mlx's `quantize` writes). The new
/// [`TripleClass`]-based already-quantized detector validates that
/// `.weight` is `uint32` before passing a triple through, so the
/// "already quantized" test fixtures need to use this — a dense
/// `f32` `.weight` next to a `.scales` is now classified as an
/// orphan, not a valid triple.
fn arr_u32(data: &[u32], shape: &[usize]) -> Array {
  Array::from_slice::<u32>(data, &shape).expect("from_slice")
}

// ──────────────── Quantization parse (schema) ────────────────

#[test]
fn quantization_parses_minimal_block() {
  // The simplest mlx-lm form: just `{ group_size, bits }`, no `mode`.
  let cfg_json = r#"{ "quantization": { "group_size": 64, "bits": 4 } }"#;
  let plq = parse_quantization(cfg_json).unwrap().unwrap();
  let q = plq.quantization.expect("global quant present");
  assert_eq!(q.group_size, 64);
  assert_eq!(q.bits, 4);
  assert_eq!(q.mode, QuantMode::Affine);
  assert!(plq.per_layer.is_empty());
}

#[test]
fn quantization_parses_mode_explicit() {
  let cfg_json = r#"{ "quantization": { "group_size": 32, "bits": 4, "mode": "mxfp4" } }"#;
  let q = parse_quantization(cfg_json)
    .unwrap()
    .unwrap()
    .quantization
    .unwrap();
  assert_eq!(q.mode, QuantMode::Mxfp4);
}

#[test]
fn quantization_parses_per_layer_overrides() {
  // Mirrors the mlx-swift `BaseConfiguration.swift:103-118` doc example.
  let cfg_json = r#"{
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "model.embed_tokens": { "group_size": 32, "bits": 4 },
        "model.layers.0.self_attn.q_norm": false
      }
    }"#;
  let plq = parse_quantization(cfg_json).unwrap().unwrap();
  let q = plq.quantization.unwrap();
  assert_eq!(q.group_size, 64);
  assert_eq!(q.bits, 4);
  assert_eq!(plq.per_layer.len(), 2);
  match plq.per_layer.get("model.embed_tokens") {
    Some(QuantizationOption::Quantize(q2)) => {
      assert_eq!(q2.group_size, 32);
      assert_eq!(q2.bits, 4);
    }
    other => panic!("expected Quantize override, got {other:?}"),
  }
  assert_eq!(
    plq
      .per_layer
      .get("model.layers.0.self_attn.q_norm")
      .copied(),
    Some(QuantizationOption::Skip)
  );
  // `quantization_for` resolves correctly for each case.
  assert_eq!(
    plq.quantization_for("model.embed_tokens"),
    Some(Quantization {
      group_size: 32,
      bits: 4,
      mode: QuantMode::Affine,
    })
  );
  assert_eq!(
    plq.quantization_for("model.layers.0.self_attn.q_norm"),
    None
  );
  // An unlisted layer falls back to the global default.
  assert_eq!(
    plq.quantization_for("model.layers.5.mlp.gate_proj"),
    Some(q)
  );
}

#[test]
fn quantization_ignores_legacy_hf_keys() {
  // mlx-swift strips `quant_method` / `linear_class` / `quantization_mode`
  // before the per-layer scan (`BaseConfiguration.swift:152-154`).
  let cfg_json = r#"{
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "quant_method": "awq",
        "linear_class": "QuantizedLinear",
        "quantization_mode": "affine"
      }
    }"#;
  let plq = parse_quantization(cfg_json).unwrap().unwrap();
  assert!(plq.per_layer.is_empty());
  assert_eq!(plq.quantization.unwrap().group_size, 64);
}

#[test]
fn quantization_absent_returns_none() {
  // A valid config.json with no `quantization` key.
  let cfg_json = r#"{ "model_type": "qwen3", "hidden_size": 1024 }"#;
  let plq = parse_quantization(cfg_json).unwrap();
  assert!(plq.is_none());
}

#[test]
fn quantization_invalid_json_errors() {
  let plq = parse_quantization("{ not json");
  assert!(plq.is_err());
}

// ──────────────── quantize_weights ────────────────

/// Tiny canned weight map: two `*.weight` keys eligible for quantization,
/// one already-quantized triple, one 1-D bias, one weight whose last
/// axis is not a multiple of `group_size`. Confirms the predicate
/// (rank / last-axis / `.scales`-sibling-presence) selects exactly the
/// two eligible weights.
#[test]
fn quantize_weights_applies_to_eligible_and_skips_rest() {
  let group_size = 64_usize;
  let n_rows = 3_usize;
  // Two eligible weights: [3, 64].
  let w1 = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
  let w2 = arr_f32(&vec![-0.25_f32; n_rows * group_size], &[n_rows, group_size]);
  // Already-quantized layer: a STRUCTURALLY-VALID affine triple
  // (`<path>.weight` uint32 + `<path>.scales` (+ `<path>.biases`)
  // f32 of matching leading dims). Classified as
  // [`TripleClass::Valid`] → skipped + passed through verbatim (per
  // mlx-lm `utils.py:349-355`, sharpened to the actual mlx layout
  // — `mlx/ops.cpp:4789-4798`).
  // Packed shape: bits=4 packs 8 elements per uint32 → last axis is
  // `group_size / 8 = 8` for group_size=64.
  let already_w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let already_scales = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let already_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  // A bias (1-D) — not quantizable (rank < 2).
  let bias = arr_f32(&[1.0_f32, 2.0, 3.0], &[3]);
  // A weight whose last axis (63) is not a multiple of group_size 64.
  let odd_last = arr_f32(&vec![0.0_f32; 3 * 63], &[3, 63]);
  // A non-`.weight` key — should pass through verbatim.
  let other = arr_f32(&[42.0_f32], &[1]);

  let mut weights: Weights = HashMap::new();
  weights.insert("model.layers.0.q_proj.weight".to_string(), w1);
  weights.insert("model.layers.0.k_proj.weight".to_string(), w2);
  weights.insert("model.layers.1.v_proj.weight".to_string(), already_w);
  weights.insert("model.layers.1.v_proj.scales".to_string(), already_scales);
  weights.insert("model.layers.1.v_proj.biases".to_string(), already_biases);
  weights.insert("model.layers.0.q_proj.bias".to_string(), bias);
  weights.insert("model.layers.2.bad.weight".to_string(), odd_last);
  weights.insert("model.norm.gamma".to_string(), other);
  let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));

  let out = quantize_weights(weights, &cfg, &default_eligible).expect("quantize");

  // Eligible weights: replaced with quantized triples (.weight + .scales
  // + .biases for affine).
  for path in ["model.layers.0.q_proj", "model.layers.0.k_proj"] {
    let w_q = out.get(&format!("{path}.weight")).expect(".weight");
    let scales = out.get(&format!("{path}.scales")).expect(".scales");
    let biases = out
      .get(&format!("{path}.biases"))
      .expect(".biases (affine)");
    // mlx `affine_quantize` packs `bits=4` elements 8-per-uint32 along
    // the last axis, so the packed shape is `[N, dim / (32/bits)]` =
    // `[3, 8]` for `[3, 64]` at 4 bits. `scales` / `biases` shape is
    // `[N, dim / group_size]` = `[3, 1]` for one group per row.
    assert_eq!(w_q.shape(), vec![3, 8]);
    assert_eq!(w_q.dtype().unwrap(), crate::dtype::Dtype::U32);
    assert_eq!(scales.shape(), vec![3, 1]);
    assert_eq!(scales.dtype().unwrap(), crate::dtype::Dtype::F32);
    assert_eq!(biases.shape(), vec![3, 1]);
    assert_eq!(biases.dtype().unwrap(), crate::dtype::Dtype::F32);
  }

  // Skipped: already-quantized layer's triple passes through unchanged
  // (uint32 packed `.weight`, f32 `.scales` / `.biases` of matching
  // leading dims — exactly the layout mlx's `affine_quantize` writes).
  let pre_q_w = out.get("model.layers.1.v_proj.weight").expect("already-w");
  assert_eq!(pre_q_w.shape(), vec![n_rows, 8]);
  assert_eq!(pre_q_w.dtype().unwrap(), crate::dtype::Dtype::U32);
  assert!(out.contains_key("model.layers.1.v_proj.scales"));
  assert!(out.contains_key("model.layers.1.v_proj.biases"));

  // Skipped: 1-D bias and ragged-last-axis weight pass through.
  assert_eq!(
    out.get("model.layers.0.q_proj.bias").unwrap().shape(),
    vec![3]
  );
  assert_eq!(
    out.get("model.layers.2.bad.weight").unwrap().shape(),
    vec![3, 63]
  );

  // Skipped: non-`.weight` keys pass through verbatim.
  assert_eq!(out.get("model.norm.gamma").unwrap().shape(), vec![1]);

  // Skipped layers do NOT acquire a stray `.scales`/`.biases`.
  assert!(!out.contains_key("model.layers.0.q_proj.scales.scales"));
  assert!(!out.contains_key("model.layers.2.bad.scales"));
  assert!(!out.contains_key("model.layers.2.bad.biases"));
}

#[test]
fn quantize_then_dequantize_roundtrips_within_tolerance() {
  let group_size = 64_usize;
  let n_rows = 4_usize;
  // Modestly-varying f32 weights so the quantization grid actually
  // covers a useful range (a constant tensor quantizes / dequantizes
  // exactly with zero error, so this catches the lossy path).
  let data: Vec<f32> = (0..n_rows * group_size)
    .map(|i| (i as f32 / 128.0) - 1.0)
    .collect();
  let w = arr_f32(&data, &[n_rows, group_size]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.linear.weight".to_string(), w);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));

  let quantized = quantize_weights(weights, &cfg, &default_eligible).unwrap();
  let dequantized = dequantize_weights(quantized, &cfg).unwrap();

  let mut deq = dequantized
    .get("model.linear.weight")
    .expect("round-tripped .weight")
    .try_clone()
    .unwrap();
  assert_eq!(deq.shape(), vec![n_rows, group_size]);
  let deq_vec: Vec<f32> = deq.to_vec().unwrap();
  // `affine` at 4 bits is lossy; mlx's grouped affine over 64 elements
  // with a [-1, 1) range typically reconstructs within ~ a few %. Use a
  // generous tolerance — the test is for the round-trip plumbing
  // (predicate, triple writeback, dequantize_weights inverse), not the
  // quantizer's exact accuracy (which is mlx-c's job and is tested
  // elsewhere).
  let max_abs_err = data
    .iter()
    .zip(deq_vec.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max);
  assert!(
    max_abs_err < 0.05,
    "round-trip max abs err = {max_abs_err}; expected < 0.05 for 4-bit affine"
  );
}

#[test]
fn quantize_weights_per_layer_skip_passes_through() {
  let group_size = 64_usize;
  let n_rows = 2_usize;
  let w = arr_f32(&vec![0.1_f32; n_rows * group_size], &[n_rows, group_size]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.embed_tokens.weight".to_string(), w);

  // Global default would quantize, but a per-layer Skip turns it off.
  let mut per_layer = HashMap::new();
  per_layer.insert("model.embed_tokens".to_string(), QuantizationOption::Skip);
  let cfg = PerLayerQuantization {
    quantization: Some(Quantization::affine(group_size as i32, 4)),
    per_layer,
  };

  let out = quantize_weights(weights, &cfg, &default_eligible).unwrap();
  let pass = out.get("model.embed_tokens.weight").expect(".weight");
  assert_eq!(pass.shape(), vec![n_rows, group_size]);
  assert_eq!(pass.dtype().unwrap(), crate::dtype::Dtype::F32);
  assert!(!out.contains_key("model.embed_tokens.scales"));
  assert!(!out.contains_key("model.embed_tokens.biases"));
}

#[test]
fn quantize_weights_per_layer_override_uses_override_params() {
  let n_rows = 2_usize;
  // Eligible only at group_size 32 (last axis 32; the global default
  // would be group_size 64, which fails the `% group_size == 0` gate —
  // but the per-layer override at 32 makes it eligible).
  let last = 32_usize;
  let w = arr_f32(&vec![0.1_f32; n_rows * last], &[n_rows, last]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.embed_tokens.weight".to_string(), w);

  let mut per_layer = HashMap::new();
  per_layer.insert(
    "model.embed_tokens".to_string(),
    QuantizationOption::Quantize(Quantization::affine(32, 4)),
  );
  let cfg = PerLayerQuantization {
    quantization: Some(Quantization::affine(64, 4)),
    per_layer,
  };

  let out = quantize_weights(weights, &cfg, &default_eligible).unwrap();
  // Quantized at group_size 32: scales / biases have one group per row
  // (last / group_size = 32 / 32 = 1).
  let scales = out.get("model.embed_tokens.scales").expect(".scales");
  assert_eq!(scales.shape(), vec![n_rows, 1]);
  let w_q = out.get("model.embed_tokens.weight").expect(".weight");
  // bits=4 packs 8 elements per uint32 → last axis is 32 / 8 = 4.
  assert_eq!(w_q.shape(), vec![n_rows, 4]);
}

// ──────────────── triple-classification fixtures ────────────────

/// A weight whose key ends in `.weight` AND meets every
/// structural guard (rank ≥ 2, last-axis divisible by group_size) but
/// the caller-supplied eligibility predicate rejects → passes through
/// unchanged (no `.scales` / `.biases` emitted). Mirrors mlx-lm's
/// `wrapped_predicate` returning `False` for a non-Linear /
/// Embedding / SwitchLinear module (`utils.py:824`).
#[test]
fn quantize_weights_predicate_rejected_passes_through() {
  let group_size = 64_usize;
  let n_rows = 2_usize;
  let w = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.some_future_module.weight".to_string(), w);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
  // Predicate that rejects this specific architecture's "future" module.
  let reject_all: &Eligible<'_> = &|_path: &str, _arr: &Array| false;

  let out = quantize_weights(weights, &cfg, reject_all).unwrap();
  let pass = out.get("model.some_future_module.weight").expect(".weight");
  assert_eq!(pass.shape(), vec![n_rows, group_size]);
  assert_eq!(pass.dtype().unwrap(), crate::dtype::Dtype::F32);
  assert!(!out.contains_key("model.some_future_module.scales"));
  assert!(!out.contains_key("model.some_future_module.biases"));
}

/// A predicate that selects a SPECIFIC path AND every other
/// structural guard passes → that path IS quantized (.weight replaced,
/// .scales / .biases emitted), while a sibling path the predicate
/// rejects passes through unchanged. Confirms the predicate is the
/// PRIMARY filter and the structural guards run after.
#[test]
fn quantize_weights_predicate_approved_quantizes() {
  let group_size = 64_usize;
  let n_rows = 2_usize;
  let w_yes = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
  let w_no = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.linear_class.weight".to_string(), w_yes);
  weights.insert("model.other_class.weight".to_string(), w_no);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
  let only_linear: &Eligible<'_> = &|path: &str, _arr: &Array| path == "model.linear_class";

  let out = quantize_weights(weights, &cfg, only_linear).unwrap();
  // Selected: quantized triple.
  assert_eq!(
    out
      .get("model.linear_class.scales")
      .expect("scales for approved layer")
      .shape(),
    vec![n_rows, 1]
  );
  // Rejected: pass-through (no .scales emitted).
  assert_eq!(
    out
      .get("model.other_class.weight")
      .expect("rejected layer .weight kept")
      .shape(),
    vec![n_rows, group_size]
  );
  assert!(!out.contains_key("model.other_class.scales"));
  assert!(!out.contains_key("model.other_class.biases"));
}

// Schema-required keys.

#[test]
fn quantization_missing_bits_errors() {
  let cfg_json = r#"{ "quantization": { "group_size": 64 } }"#;
  let err = parse_quantization(cfg_json).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("bits"),
    "error should mention the missing `bits` key, got: {msg}"
  );
}

#[test]
fn quantization_missing_group_size_errors() {
  let cfg_json = r#"{ "quantization": { "bits": 4 } }"#;
  let err = parse_quantization(cfg_json).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("group_size"),
    "error should mention the missing `group_size` key, got: {msg}"
  );
}

#[test]
fn quantization_both_present_ok() {
  let cfg_json = r#"{ "quantization": { "group_size": 32, "bits": 4 } }"#;
  let plq = parse_quantization(cfg_json).unwrap().unwrap();
  let q = plq.quantization.expect("global quant present");
  assert_eq!(q.group_size, 32);
  assert_eq!(q.bits, 4);
}

// Stale sibling collision.

#[test]
fn quantize_weights_orphan_biases_collision_errors() {
  let group_size = 64_usize;
  let n_rows = 2_usize;
  let w = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
  // Orphan biases — NO matching `.scales`, so not a valid
  // already-quantized triple (mlx `affine_quantize` always writes
  // `.scales` alongside `.biases`, `mlx/ops.cpp:4793-4798`). The
  // `classify_triple` check runs BEFORE the eligibility predicate, so
  // this fires unconditionally for every `.weight` key with an orphan
  // `.biases` sibling.
  let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.foo.weight".to_string(), w);
  weights.insert("model.foo.biases".to_string(), stale_biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.foo"),
        "LayerKeyed must name the colliding layer, got layer={:?}",
        payload.layer()
      );
      assert!(
        matches!(payload.inner(), Error::MissingKey(_)),
        "inner must be MissingKey for stale `.biases` without `.scales`, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// A VALID already-quantized triple (`.weight` uint32 packed +
/// `.scales` (+ `.biases`) of matching leading dims, the exact layout
/// mlx's `affine_quantize` writes — `mlx/ops.cpp:4789-4798`) STILL
/// passes through unchanged. The new [`TripleClass`] validation must
/// not regress the already-quantized skip.
#[test]
fn quantize_weights_valid_existing_triple_still_skipped() {
  let n_rows = 2_usize;
  // Packed `.weight`: bits=4 packs 8 elements per uint32 → last axis
  // is `group_size / 8 = 8` for group_size=64.
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.already.weight".to_string(), w);
  weights.insert("model.already.scales".to_string(), scales);
  weights.insert("model.already.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let out = quantize_weights(weights, &cfg, &default_eligible).expect("valid triple passes");
  // `.weight` is the packed [N, 8] uint32 we inserted — not re-quantized.
  let w_out = out.get("model.already.weight").unwrap();
  assert_eq!(w_out.shape(), vec![n_rows, 8]);
  assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
  assert!(out.contains_key("model.already.scales"));
  assert!(out.contains_key("model.already.biases"));
}

/// A dense `.weight` (float dtype) next to a stale
/// `.scales` orphan (no valid quantized layout) → [`TripleClass::Invalid`]
/// → `Err(Backend)` naming the layer and the offending `.scales`. This is
/// the case where the old presence-only
/// `is_already_quantized` check would have classified this as "already
/// quantized" and silently passed through, leaving a dense `.weight` next
/// to a corrupt `.scales` for `dequantize_weights` to choke on.
///
/// `.biases` is included so the triple advances past the affine-arity
/// check and reaches the `.weight` dtype check (the regression this
/// fixture is asserting); a separate fixture covers the missing-`.biases`
/// arity case under `affine`.
#[test]
fn quantize_weights_orphan_scales_with_dense_weight_errors() {
  let group_size = 64_usize;
  let n_rows = 2_usize;
  // Dense f32 `.weight` (NOT a quantized uint32 packed matrix).
  let w = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
  // Stale orphan `.scales` + matching `.biases` next to it.
  let stale_scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.foo.weight".to_string(), w);
  weights.insert("model.foo.scales".to_string(), stale_scales);
  weights.insert("model.foo.biases".to_string(), stale_biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.foo"),
        "LayerKeyed must name the colliding layer, got layer={:?}",
        payload.layer()
      );
      // The orphan `.scales` with a dense `.weight` is classified as a dtype
      // mismatch in `classify_triple` (uint32 expected, F32 observed).
      assert!(
        matches!(payload.inner(), Error::UnsupportedDtype(_)),
        "inner must be UnsupportedDtype for dense-weight orphan, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// A `.weight` + `.scales` with MISMATCHED leading
/// dims (the `.weight` claims to be uint32 packed, but its rank or
/// leading shape doesn't match `.scales` as mlx's `quantize` would
/// produce). Classified as [`TripleClass::Invalid`] → `Err(Backend)`.
#[test]
fn quantize_weights_mismatched_scales_shape_errors() {
  let n_rows = 2_usize;
  // Packed `.weight` (uint32) at shape [N=2, 8].
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  // `.scales` with a different leading dim ([3, 1] vs `.weight`
  // leading dim of [2]).
  let bad_scales = arr_f32(&[1.0_f32; 3], &[3, 1]);
  // `.biases` matching `.scales` so the triple advances past the
  // affine-arity check and reaches the leading-dim mismatch check.
  let biases = arr_f32(&[0.0_f32; 3], &[3, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.foo.weight".to_string(), w);
  weights.insert("model.foo.scales".to_string(), bad_scales);
  weights.insert("model.foo.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.foo"),
        "LayerKeyed must name the colliding layer, got layer={:?}",
        payload.layer()
      );
      // Mismatched scales shape: classify_triple emits ShapePairMismatch
      // (leading-dim mismatch) inside the LayerKeyed wrapper.
      assert!(
        matches!(payload.inner(), Error::ShapePairMismatch(_)),
        "inner must be ShapePairMismatch for leading-dim mismatch, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

// A `quantize_weights_mismatched_biases_dtype_errors`
// test once asserted a `.biases`/`.scales` dtype-equality check we are
// intentionally removing to match the mlx-lm / mlx-swift reference loader
// paths (which trust mlx-c to validate scale dtypes at the
// `quantize` / `dequantize` call site — `mlx/mlx/ops.cpp:75-115`). The
// dtype-mismatched triple is now passed through to mlx-c, which surfaces
// a precise `[dequantize] ...` error. See the module-level "Validation
// contract" section.

// ──────────────── Structural shape sanity ────────────────

/// A uint32 rank-1 `.weight` next to a uint32 rank-1 `.scales`
/// (rank-equal, even leading-dim-equal trivially since both have only
/// a last axis). On dtype `uint32` + ranks equal alone this would look
/// like a [`TripleClass::Valid`] triple, but `classify_triple` rejects
/// it because mlx `quantize` requires rank ≥ 2 inputs
/// (`mlx/ops.cpp:4925-4929`).
#[test]
fn quantize_weights_rank1_uint32_triple_errors() {
  // Both `.weight` and `.scales` are rank-1 uint32 — would slip past
  // the dtype + rank-equality check, but mlx never emits a rank-1
  // quantized triple.
  let w = arr_u32(&[0_u32, 0, 0, 0], &[4]);
  let scales = arr_u32(&[1_u32], &[1]);
  // `.biases` matching `.scales` shape/dtype so the triple advances past
  // the affine-arity check and reaches the rank-≥-2 check.
  let biases = arr_u32(&[0_u32], &[1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.bad.weight".to_string(), w);
  weights.insert("model.bad.scales".to_string(), scales);
  weights.insert("model.bad.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.bad"),
        "LayerKeyed must name the malformed layer, got layer={:?}",
        payload.layer()
      );
      // rank-1 uint32 triple: classify_triple emits RankMismatch.
      assert!(
        matches!(payload.inner(), Error::RankMismatch(_)),
        "inner must be RankMismatch for rank-1 `.weight`, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// A `.weight` + `.scales` triple whose
/// `.scales` last-axis does NOT match the mlx invariant
/// `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`
/// (`mlx/ops.cpp:107`) now passes `classify_triple` (which only
/// checks dtype/rank/leading-dims, see the module-level "Validation
/// contract" section). The mismatch is caught downstream by mlx-c
/// at the `dequantize` call — the loader path no longer rejects it
/// upfront, mirroring mlx-lm's `quantize_module_predicate`
/// (`utils.py:823-835`) and mlx-swift's `QuantizationContainer.decode`
/// (`BaseConfiguration.swift:139-171`), which both trust mlx-c.
///
/// This test asserts the new pass-through behavior: an
/// already-quantized triple with structurally-sound dtype/rank/leading
/// dims is preserved verbatim regardless of the per-mode bits /
/// group_size pairing (mlx-c will validate when the user later
/// invokes `dequantize_weights` or any quantized matmul).
#[test]
fn quantize_weights_pre_quantized_triple_passes_through_to_mlxc() {
  // Packed `.weight` `[2, 8]` u32 + `.scales` `[2, 2]` f32 (+ `.biases`
  // matching). A last-axis invariant check would have
  // rejected this (expected scales-last = 8 * 32 / 4 / 64 = 1, not
  // 2). Instead, this passes through — mlx-c is the validator.
  let n_rows = 2_usize;
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows * 2], &[n_rows, 2]);
  let biases = arr_f32(&vec![0.0_f32; n_rows * 2], &[n_rows, 2]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.foo.weight".to_string(), w);
  weights.insert("model.foo.scales".to_string(), scales);
  weights.insert("model.foo.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let out = quantize_weights(weights, &cfg, &default_eligible)
    .expect("triple now passes through; mlx-c validates per-mode params at call time");
  // Triple preserved verbatim.
  let w_out = out.get("model.foo.weight").expect(".weight");
  assert_eq!(w_out.shape(), vec![n_rows, 8]);
  assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
  let s_out = out.get("model.foo.scales").expect(".scales");
  assert_eq!(s_out.shape(), vec![n_rows, 2]);
  assert!(out.contains_key("model.foo.biases"));
}

/// An affine triple with `bits=3` (mlx-supported,
/// `mlx/ops.cpp:4745-4750`: bits ∈ {2,3,4,5,6,8}) passes through.
/// A `32 % bits == 0` guard would incorrectly reject `bits ∈
/// {3, 5, 6}`; per the validation contract, per-mode bits
/// validation is delegated to mlx-c.
#[test]
fn quantize_weights_pre_quantized_bits3_triple_passes_through() {
  // A structurally-sound triple with `bits=3` per the per-layer
  // override. `classify_triple` only checks `.weight` is u32, rank
  // ≥ 2, leading-dims match — none of which depend on the bit width.
  // (The exact packed last-axis would depend on mlx's bits=3 packing,
  // but the loader path does not compute it.)
  let n_rows = 2_usize;
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.foo.weight".to_string(), w);
  weights.insert("model.foo.scales".to_string(), scales);
  weights.insert("model.foo.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 3));
  let out = quantize_weights(weights, &cfg, &default_eligible)
    .expect("bits=3 triple passes through; mlx supports bits ∈ {2,3,4,5,6,8}");
  let w_out = out.get("model.foo.weight").expect(".weight");
  assert_eq!(w_out.shape(), vec![n_rows, 8]);
  assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
}

/// Structural-shape regression: a CORRECT `.weight` `[2, 8]`
/// packed at `bits=4, group_size=64` with `.scales` `[2, 1]` (+
/// `.biases` matching `.scales` shape — affine-arity holds). Still
/// passes through (the basic shape-sanity checks all hold).
#[test]
fn quantize_weights_valid_triple_skipped() {
  let n_rows = 2_usize;
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.foo.weight".to_string(), w);
  weights.insert("model.foo.scales".to_string(), scales);
  weights.insert("model.foo.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let out =
    quantize_weights(weights, &cfg, &default_eligible).expect("valid triple passes through");
  let w_out = out.get("model.foo.weight").expect(".weight");
  assert_eq!(w_out.shape(), vec![n_rows, 8]);
  assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
  let s_out = out.get("model.foo.scales").expect(".scales");
  assert_eq!(s_out.shape(), vec![n_rows, 1]);
  assert!(out.contains_key("model.foo.biases"));
}

/// A triple at a path that the per-layer config marks as
/// `Skip`. The layer was intentionally not quantized — a pre-existing
/// triple at that path is a stale collision. Classified as
/// [`TripleClass::Invalid`] (the doc-level "Precondition" branch).
#[test]
fn quantize_weights_triple_on_skip_path_errors() {
  let n_rows = 2_usize;
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.embed_tokens.weight".to_string(), w);
  weights.insert("model.embed_tokens.scales".to_string(), scales);

  let mut per_layer = HashMap::new();
  per_layer.insert("model.embed_tokens".to_string(), QuantizationOption::Skip);
  let cfg = PerLayerQuantization {
    quantization: Some(Quantization::affine(64, 4)),
    per_layer,
  };

  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.embed_tokens"),
        "LayerKeyed must name the Skip layer, got layer={:?}",
        payload.layer()
      );
      // Skip override → KeyCollision (stale `.scales` next to a Skip layer).
      assert!(
        matches!(payload.inner(), Error::KeyCollision(_)),
        "inner must be KeyCollision for Skip-with-stale-scales, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

// ──────────────── per-mode bias arity ────────────────

/// An `affine` triple with NO `.biases` (only `.weight` + `.scales`)
/// is structurally incomplete. mlx `affine_quantize` emits
/// `{w_q, scales, biases}` unconditionally (`mlx/ops.cpp:4793-4798`); a
/// matching shape/dtype on `.scales` is not enough — the resolved mode
/// dictates the bias arity. Classified as [`TripleClass::Invalid`].
#[test]
fn quantize_weights_affine_triple_missing_biases_errors() {
  let n_rows = 2_usize;
  // Packed `.weight` `[2, 8]` u32 + `.scales` `[2, 1]` f32 — a layout
  // that matches the affine weight/scales invariant except for the
  // missing `.biases`.
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.affine_missing.weight".to_string(), w);
  weights.insert("model.affine_missing.scales".to_string(), scales);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.affine_missing"),
        "LayerKeyed must name the incomplete layer, got layer={:?}",
        payload.layer()
      );
      // Affine triple missing biases → inner MissingKey (the `.biases` key).
      let Error::MissingKey(inner) = payload.inner() else {
        panic!(
          "inner must be MissingKey for affine missing biases, got: {:?}",
          payload.inner()
        );
      };
      assert!(
        inner.key().contains(".biases"),
        "MissingKey must name the missing `.biases` sibling, got key={:?}",
        inner.key()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// An `mxfp4` triple with `.biases` present is a stale sibling
/// from a different mode. mlx `fp_quantize` emits `{w_q, scales}`
/// only — never `.biases` (`mlx/ops.cpp:4890,4898-4904`). Even if
/// shape/dtype happen to align with `.scales`, the bias slot MUST be
/// absent. Classified as [`TripleClass::Invalid`].
#[test]
fn quantize_weights_mxfp4_triple_with_stale_biases_errors() {
  let n_rows = 2_usize;
  // `mxfp4` requires `group_size=32`, `bits=4` (`mlx/ops.cpp:4808-4823`).
  // Unpacked last = packed_last * 32 / bits = 4 * 8 = 32 = group_size,
  // so scales last-axis = 32 / 32 = 1 — a structurally well-formed
  // `mxfp4` `.weight`/`.scales` pair.
  let w = arr_u32(&vec![0_u32; n_rows * 4], &[n_rows, 4]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  // Stale `.biases` from a different (affine) mode — same shape/dtype
  // as `.scales` so it looks valid to a shape-only check.
  let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.mxfp4_stale.weight".to_string(), w);
  weights.insert("model.mxfp4_stale.scales".to_string(), scales);
  weights.insert("model.mxfp4_stale.biases".to_string(), stale_biases);

  let cfg = PerLayerQuantization::from_global(Quantization {
    group_size: 32,
    bits: 4,
    mode: QuantMode::Mxfp4,
  });
  let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.mxfp4_stale"),
        "LayerKeyed must name the offending layer, got layer={:?}",
        payload.layer()
      );
      let Error::KeyCollision(inner) = payload.inner() else {
        panic!(
          "inner must be KeyCollision for mxfp4-with-stale-biases, got: {:?}",
          payload.inner()
        );
      };
      assert!(
        inner.key().contains(".biases"),
        "KeyCollision must name the stale `.biases` sibling, got key={:?}",
        inner.key()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// Regression: a structurally valid `mxfp4` triple
/// (`.weight` u32 + `.scales` matching, NO `.biases`) — the scale-only
/// layout `fp_quantize` actually writes (`mlx/ops.cpp:4890,4898-4904`).
/// Must pass through unchanged: the new arity check accepts the
/// `(Mxfp4 | Mxfp8 | Nvfp4, None)` arm.
#[test]
fn quantize_weights_valid_mxfp4_scales_only_triple_passes() {
  let n_rows = 2_usize;
  // `mxfp4` invariants: group_size=32, bits=4. Packed `.weight` `[2, 4]`
  // u32 → unpacks to `[2, 32]` (1 group per row) → `.scales` `[2, 1]`.
  let w = arr_u32(&vec![0_u32; n_rows * 4], &[n_rows, 4]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.mxfp4_ok.weight".to_string(), w);
  weights.insert("model.mxfp4_ok.scales".to_string(), scales);

  let cfg = PerLayerQuantization::from_global(Quantization {
    group_size: 32,
    bits: 4,
    mode: QuantMode::Mxfp4,
  });
  let out = quantize_weights(weights, &cfg, &default_eligible)
    .expect("scale-only mxfp4 triple passes through");
  // `.weight` and `.scales` preserved verbatim; `.biases` is NOT
  // synthesized (scale-only mode).
  let w_out = out.get("model.mxfp4_ok.weight").expect(".weight");
  assert_eq!(w_out.shape(), vec![n_rows, 4]);
  assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
  let s_out = out.get("model.mxfp4_ok.scales").expect(".scales");
  assert_eq!(s_out.shape(), vec![n_rows, 1]);
  assert!(!out.contains_key("model.mxfp4_ok.biases"));
}

// ──────────────── dequantize_weights mode-arity symmetry ────────────────

/// `dequantize_weights` is symmetric with
/// `quantize_weights`'s mode-arity check (the `affine`-requires-biases
/// / `mxfp*|nvfp4`-forbids-biases contract). Forwarding an affine
/// triple WITHOUT `.biases` to mlx-c's `dequantize` would silently
/// reconstruct without the zero-point. The arity check catches this
/// upfront and returns a clear error naming the layer and the resolved
/// `affine` mode.
#[test]
fn dequantize_weights_affine_missing_biases_errors() {
  let n_rows = 2_usize;
  // Structurally-valid affine `.weight` + `.scales` pair, but no
  // `.biases` — incomplete affine triple.
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.affine_no_bias.weight".to_string(), w);
  weights.insert("model.affine_no_bias.scales".to_string(), scales);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let err = dequantize_weights(weights, &cfg).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.affine_no_bias"),
        "LayerKeyed must name the layer, got layer={:?}",
        payload.layer()
      );
      let Error::MissingKey(inner) = payload.inner() else {
        panic!(
          "inner must be MissingKey for affine triple missing biases, got: {:?}",
          payload.inner()
        );
      };
      assert!(
        inner.key().contains(".biases"),
        "MissingKey must name the missing `.biases` sibling, got key={:?}",
        inner.key()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// An `mxfp4` triple WITH a stale `.biases` would be
/// forwarded to mlx-c, which silently dequantizes (ignoring the
/// biases). The arity check now catches this upfront and returns a
/// clear error naming the layer and the offending `mxfp4` mode.
#[test]
fn dequantize_weights_mxfp4_with_stale_biases_errors() {
  let n_rows = 2_usize;
  let w = arr_u32(&vec![0_u32; n_rows * 4], &[n_rows, 4]);
  let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
  let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.mxfp4_stale.weight".to_string(), w);
  weights.insert("model.mxfp4_stale.scales".to_string(), scales);
  weights.insert("model.mxfp4_stale.biases".to_string(), stale_biases);

  let cfg = PerLayerQuantization::from_global(Quantization {
    group_size: 32,
    bits: 4,
    mode: QuantMode::Mxfp4,
  });
  let err = dequantize_weights(weights, &cfg).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.mxfp4_stale"),
        "LayerKeyed must name the layer, got layer={:?}",
        payload.layer()
      );
      let Error::KeyCollision(inner) = payload.inner() else {
        panic!(
          "inner must be KeyCollision for mxfp4-with-stale-biases on dequantize, got: {:?}",
          payload.inner()
        );
      };
      assert!(
        inner.key().contains(".biases"),
        "KeyCollision must name the stale `.biases` sibling, got key={:?}",
        inner.key()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// `dequantize_weights` is symmetric with
/// `classify_triple`'s orphan-`.biases` guard. A map carrying
/// `.weight` (`uint32` packed) + `.biases` but NO `.scales` is never
/// a valid mlx-produced triple (mlx `affine_quantize` always writes
/// `.scales` alongside `.biases`, `mlx/ops.cpp:4793-4798`). Without a
/// dedicated guard the orphan falls through the discovery walk (which
/// only indexes `.scales` keys) and the `uint32` packed `.weight`
/// passes through to the dequantized output as-is. The orphan-bias
/// guard catches this upfront with the same exit point + message
/// style as the dequantize arity check.
#[test]
fn dequantize_weights_orphan_biases_with_packed_weight_errors() {
  let n_rows = 2_usize;
  // `uint32`-packed `.weight` shaped [2, 8] + `.biases` [2, 1], NO `.scales`.
  let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
  let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.orphan_bias.weight".to_string(), w);
  weights.insert("model.orphan_bias.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let err = dequantize_weights(weights, &cfg).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.orphan_bias"),
        "LayerKeyed must name the layer, got layer={:?}",
        payload.layer()
      );
      let Error::MissingKey(inner) = payload.inner() else {
        panic!(
          "inner must be MissingKey for orphan biases without scales, got: {:?}",
          payload.inner()
        );
      };
      assert!(
        inner.key().contains(".scales"),
        "MissingKey must name the missing `.scales` sibling, got key={:?}",
        inner.key()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// A too-broad orphan-bias guard would over-reject a normal dense
/// Linear layer carrying `P.weight` (F32) + `P.biases` (F32) with no
/// `P.scales` — that combination is a standard dense+bias layer, not a
/// malformed quantized triple. The narrowed guard only fires when
/// `P.weight` is `uint32` (the mlx-quantization signal,
/// `mlx/ops.cpp:4795,4900`); a dense (non-`uint32`) `.weight` passes
/// through verbatim, both keys preserved.
#[test]
fn dequantize_weights_dense_weight_with_biases_passes_through() {
  let n_rows = 2_usize;
  let n_cols = 8_usize;
  // Dense F32 `.weight` shaped [2, 8] + F32 `.biases` [8], NO `.scales`.
  let w = arr_f32(
    &(0..n_rows * n_cols).map(|i| i as f32).collect::<Vec<_>>(),
    &[n_rows, n_cols],
  );
  let biases = arr_f32(&vec![0.5_f32; n_cols], &[n_cols]);
  let mut weights: Weights = HashMap::new();
  weights.insert("model.dense.weight".to_string(), w);
  weights.insert("model.dense.biases".to_string(), biases);

  let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let out = dequantize_weights(weights, &cfg)
    .expect("dense `.weight` (F32) + `.biases` (F32) with no `.scales` must pass through");

  // Both keys preserved verbatim, dtypes unchanged.
  let mut w_out = out
    .get("model.dense.weight")
    .expect("passed-through .weight")
    .try_clone()
    .unwrap();
  let mut b_out = out
    .get("model.dense.biases")
    .expect("passed-through .biases")
    .try_clone()
    .unwrap();
  assert_eq!(w_out.dtype().unwrap(), Dtype::F32);
  assert_eq!(b_out.dtype().unwrap(), Dtype::F32);
  assert_eq!(w_out.shape(), vec![n_rows, n_cols]);
  assert_eq!(b_out.shape(), vec![n_cols]);
  let w_vec: Vec<f32> = w_out.to_vec().unwrap();
  let b_vec: Vec<f32> = b_out.to_vec().unwrap();
  assert_eq!(
    w_vec,
    (0..n_rows * n_cols).map(|i| i as f32).collect::<Vec<_>>(),
    "dense `.weight` data must be passed through verbatim"
  );
  assert_eq!(
    b_vec,
    vec![0.5_f32; n_cols],
    "`.biases` data must be passed through verbatim"
  );
}

// ──────────────── AutoAWQ on-load conversion ────────────────

/// `0xFFFF` packed at the AWQ bit positions for `[0xF, 0, 0xF, 0, 0xF, 0, 0xF, 0]`.
/// See [`AWQ_SHIFTS`] for the bit-layout algebra. Verifying this exact pattern
/// pins the inverse-permutation step and catches a swap to `[0..8] * bits`
/// (the swift `unpackAndReorder` form without the `take` step).
#[test]
fn unpack_awq_weights_single_int32_gives_8_nibbles() {
  // `0xFFFF` = `0xF | (0xF << 4) | (0xF << 8) | (0xF << 12)` — four 0xF
  // nibbles at AWQ shift positions [0, 4, 8, 12] = logical positions
  // [0, 2, 4, 6] (even). The shift table places them at output positions
  // 0, 2, 4, 6; the zero nibbles at AWQ positions [16, 20, 24, 28] land
  // at output positions 1, 3, 5, 7.
  let packed = Array::from_slice::<u32>(&[0xFFFF_u32], &(1usize, 1)).unwrap();
  let mut unpacked = unpack_awq_weights(&packed).unwrap();
  assert_eq!(unpacked.shape(), vec![1, 8]);
  assert_eq!(unpacked.dtype().unwrap(), Dtype::U32);
  assert_eq!(
    unpacked.to_vec::<u32>().unwrap(),
    vec![0xF, 0, 0xF, 0, 0xF, 0, 0xF, 0]
  );
}

/// Verify the inverse permutation: packing nibbles `[0, 1, 2, 3, 4, 5, 6, 7]`
/// at AWQ bit positions produces an int32 that unpacks to that natural order.
/// This is the load-bearing assertion — if the shift table were sequential
/// (`[0..8] * bits`) the output would be `[0, 2, 4, 6, 1, 3, 5, 7]` (the
/// AWQ-native scrambled order).
#[test]
fn unpack_awq_weights_reverses_awq_scramble() {
  // logical-pos → bit-pos: [0→0, 1→16, 2→4, 3→20, 4→8, 5→24, 6→12, 7→28].
  // The 0-nibble at bit 0 contributes nothing — drop the explicit `0_u32 |`
  // to avoid clippy's `identity_op` lint.
  let packed_val: u32 =
    (1_u32 << 16) | (2 << 4) | (3 << 20) | (4 << 8) | (5 << 24) | (6 << 12) | (7 << 28);
  assert_eq!(packed_val, 0x7531_6420);
  let packed = Array::from_slice::<u32>(&[packed_val], &(1usize, 1)).unwrap();
  let mut unpacked = unpack_awq_weights(&packed).unwrap();
  assert_eq!(unpacked.shape(), vec![1, 8]);
  assert_eq!(
    unpacked.to_vec::<u32>().unwrap(),
    vec![0, 1, 2, 3, 4, 5, 6, 7]
  );
}

/// 2-D `[rows, packed_cols]` qweight → `[rows, packed_cols * 8]`. Mirrors
/// the python ref's strict 2-D contract (`utils.py:75` `out_features,
/// packed_in = qweight.shape`).
#[test]
fn unpack_awq_weights_preserves_row_count_expands_cols_8x() {
  // 3 rows × 2 packed_cols = 6 int32. Use all zeros (the only shape we're
  // checking here).
  let packed = Array::from_slice::<u32>(&[0u32; 6], &(3usize, 2)).unwrap();
  let mut unpacked = unpack_awq_weights(&packed).unwrap();
  assert_eq!(unpacked.shape(), vec![3, 16]);
  assert_eq!(unpacked.to_vec::<u32>().unwrap(), vec![0u32; 48]);
}

/// All-zero packed input → all-zero unpacked output of correct shape.
#[test]
fn unpack_awq_weights_handles_zero_input() {
  let packed = Array::from_slice::<u32>(&[0u32, 0, 0, 0], &(2usize, 2)).unwrap();
  let mut unpacked = unpack_awq_weights(&packed).unwrap();
  assert_eq!(unpacked.shape(), vec![2, 16]);
  assert_eq!(unpacked.to_vec::<u32>().unwrap(), vec![0u32; 32]);
}

/// 1-D / 3-D / 0-D inputs are rejected with a clear shape error. Mirrors the
/// python ref's strict 2-D contract (`utils.py:75`).
#[test]
fn unpack_awq_weights_rejects_non_2d() {
  let r1 = Array::from_slice::<u32>(&[0u32; 4], &(4usize,)).unwrap();
  let err = unpack_awq_weights(&r1).unwrap_err();
  assert!(
    matches!(err, Error::RankMismatch(_)),
    "1-D should be RankMismatch, got {err:?}"
  );
  let r3 = Array::from_slice::<u32>(&[0u32; 8], &(2usize, 2, 2)).unwrap();
  assert!(matches!(
    unpack_awq_weights(&r3).unwrap_err(),
    Error::RankMismatch(_)
  ));
}

/// Non-32-bit-int dtype is rejected. AutoAWQ allocates `qweight` /
/// `qzeros` as `torch.int32` (signed) — we accept both `u32` AND `i32`
/// (see [`unpack_awq_weights_accepts_i32_input`]), but anything else
/// (floats, narrower ints, etc.) is a layout mismatch the caller should
/// fix upstream.
#[test]
fn unpack_awq_weights_rejects_non_32bit_int_dtype() {
  // f32 is the canonical "wrong" dtype to test against — narrow ints,
  // bool, and floats all hit the same `UnsupportedDtype` arm.
  let r = Array::from_slice::<f32>(&[0.0_f32; 4], &(2usize, 2)).unwrap();
  let err = unpack_awq_weights(&r).unwrap_err();
  assert!(
    matches!(err, Error::UnsupportedDtype(_)),
    "f32 dtype should be UnsupportedDtype, got {err:?}"
  );
}

/// i32 input is accepted (AutoAWQ's `WQLinear_GEMM`
/// allocates packed buffers as `torch.int32`, so standard on-disk
/// checkpoints carry the signed dtype). Output matches what the equivalent
/// u32 input would produce — verifying the bit-preserving reinterpret.
#[test]
fn unpack_awq_weights_accepts_i32_input() {
  // Pick a packed value whose high bit is SET — this is the case the
  // bug would corrupt: a value-preserving cast would clamp the negative
  // i32 to 0 (or saturate), losing the high nibble. The bit-preserving
  // view keeps `0xF` in the MSB nibble.
  let raw: u32 = 0xF0FF_FFFF;
  let signed: i32 = raw as i32;
  assert!(
    signed < 0,
    "fixture must be negative to exercise the sign bit"
  );
  let i32_packed = Array::from_slice::<i32>(&[signed], &(1usize, 1)).unwrap();
  let u32_packed = Array::from_slice::<u32>(&[raw], &(1usize, 1)).unwrap();

  let mut from_i32 = unpack_awq_weights(&i32_packed).expect("i32 input should be accepted");
  let mut from_u32 = unpack_awq_weights(&u32_packed).expect("u32 input still accepted");
  assert_eq!(from_i32.shape(), vec![1, 8]);
  assert_eq!(from_u32.shape(), vec![1, 8]);
  assert_eq!(from_i32.dtype().unwrap(), Dtype::U32);
  let i32_nibbles = from_i32.to_vec::<u32>().unwrap();
  let u32_nibbles = from_u32.to_vec::<u32>().unwrap();
  assert_eq!(
    i32_nibbles, u32_nibbles,
    "i32 input must produce the SAME nibbles as the equivalent u32 input (bit-preserving)"
  );
}

/// Existing u32 inputs continue to work (regression guard for the
/// `Cow::Borrowed(qweight)` short-circuit path).
#[test]
fn unpack_awq_weights_accepts_u32_input() {
  let raw: u32 = 0xF0FF_FFFF;
  let packed = Array::from_slice::<u32>(&[raw], &(1usize, 1)).unwrap();
  let out = unpack_awq_weights(&packed).expect("u32 input accepted");
  assert_eq!(out.shape(), vec![1, 8]);
  assert_eq!(out.dtype().unwrap(), Dtype::U32);
}

// ──────────────── transform_awq_weights ────────────────

/// Build a 1-element AWQ qweight (`[1, 1]` u32) whose 8 nibbles, in logical
/// order, equal `nibbles`.
fn awq_pack_one_row(nibbles: [u32; 8]) -> u32 {
  let mut packed = 0u32;
  for (k, &n) in nibbles.iter().enumerate() {
    packed |= (n & 0xF) << AWQ_SHIFTS[k];
  }
  packed
}

/// Compute the AutoAWQ-dequantize value for a single nibble:
/// `(nibble - zero) * scale` (`utils.py:144-147` comment).
fn awq_dequant(nibble: u32, zero: u32, scale: f32) -> f32 {
  (nibble as i32 - zero as i32) as f32 * scale
}

/// Compute the MLX-affine-dequantize value for a single nibble:
/// `nibble * scale + bias` (`mlx/ops.cpp` affine_dequantize convention).
fn mlx_dequant(nibble: u32, scale: f32, bias: f32) -> f32 {
  nibble as f32 * scale + bias
}

/// End-to-end round-trip: pick known AWQ qweight/qzeros/scales, run
/// `transform_awq_weights`, then verify that re-dequantizing the MLX-format
/// output (via the literal `nibble * scale + bias`) matches the original
/// AWQ-format dequant (`(nibble - zero) * scale`) at every output position.
/// This is the load-bearing semantic guarantee of the converter.
#[test]
fn transform_awq_weights_round_trips_known_fixture() {
  // in_features = 8, out_features = 8, group_size = 4, bits = 4.
  // → packed_out = 1, packed_in = 2, n_groups = 2.
  // qweight shape: [in_features, packed_out] = [8, 1]
  // scales  shape: [n_groups,    out_features] = [2, 8]
  // qzeros  shape: [n_groups,    packed_out] = [2, 1]
  let in_features = 8usize;
  let out_features = 8usize;
  let group_size = 4u32;
  let n_groups = 2usize;

  // Choose distinct nibbles per (in, out) so we can verify the transpose.
  // unpacked_awq[in, out] = ((in + 1) * 3 + out) % 16
  let awq_unpacked: Vec<Vec<u32>> = (0..in_features)
    .map(|i| {
      (0..out_features)
        .map(|o| (((i + 1) * 3 + o) % 16) as u32)
        .collect()
    })
    .collect();
  // Pack each row's 8 nibbles into one u32 → flat [in_features] u32 buffer.
  let qweight_data: Vec<u32> = (0..in_features)
    .map(|i| {
      let row: [u32; 8] = awq_unpacked[i].to_vec().try_into().unwrap();
      awq_pack_one_row(row)
    })
    .collect();
  let qweight = Array::from_slice::<u32>(&qweight_data, &(in_features, 1)).unwrap();

  // qzeros: per (group, out). Choose nibble = (group + out) % 16.
  let qzero_unpacked: Vec<Vec<u32>> = (0..n_groups)
    .map(|g| (0..out_features).map(|o| ((g + o) % 16) as u32).collect())
    .collect();
  let qzeros_data: Vec<u32> = (0..n_groups)
    .map(|g| {
      let row: [u32; 8] = qzero_unpacked[g].to_vec().try_into().unwrap();
      awq_pack_one_row(row)
    })
    .collect();
  let qzeros = Array::from_slice::<u32>(&qzeros_data, &(n_groups, 1)).unwrap();

  // scales: per (group, out). Distinct positive floats.
  let scales_data: Vec<f32> = (0..n_groups * out_features)
    .map(|i| 0.1_f32 * (i as f32 + 1.0))
    .collect();
  let scales = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();

  let mut weights: Weights = HashMap::new();
  weights.insert("layer.qweight".to_string(), qweight);
  weights.insert("layer.qzeros".to_string(), qzeros);
  weights.insert("layer.scales".to_string(), scales);

  let config = AwqLoadConfig {
    bits: 4,
    group_size,
    zero_point: true,
    version: "gemm".into(),
  };
  let (out, plq) = transform_awq_weights(weights, &config).expect("transform");

  // PerLayerQuantization carries the resolved (group_size=4, bits=4, affine).
  let g = plq.quantization.expect("global quant");
  assert_eq!(g.group_size, group_size as i32);
  assert_eq!(g.bits, 4);
  assert_eq!(g.mode, QuantMode::Affine);

  // Output keys: `layer.weight` (u32 [out, packed_in]),
  //              `layer.scales` (f32 [out, n_groups]),
  //              `layer.biases` (f32 [out, n_groups]).
  let mut weight_arr = out
    .get("layer.weight")
    .expect("layer.weight")
    .try_clone()
    .unwrap();
  let mut scales_arr = out
    .get("layer.scales")
    .expect("layer.scales")
    .try_clone()
    .unwrap();
  let mut biases_arr = out
    .get("layer.biases")
    .expect("layer.biases")
    .try_clone()
    .unwrap();
  assert!(
    !out.contains_key("layer.qweight"),
    "qweight key must be replaced by .weight"
  );
  assert!(
    !out.contains_key("layer.qzeros"),
    "qzeros key must be replaced by .biases"
  );
  assert_eq!(weight_arr.dtype().unwrap(), Dtype::U32);
  assert_eq!(weight_arr.shape(), vec![out_features, in_features / 8]);
  assert_eq!(scales_arr.shape(), vec![out_features, n_groups]);
  assert_eq!(biases_arr.shape(), vec![out_features, n_groups]);
  assert_eq!(scales_arr.dtype().unwrap(), Dtype::F32);
  assert_eq!(biases_arr.dtype().unwrap(), Dtype::F32);

  // Unpack the MLX-format weight back to natural nibbles for the assertion.
  // MLX-packed shifts: arange(8) * 4 = [0, 4, 8, 12, 16, 20, 24, 28].
  let weight_packed: Vec<u32> = weight_arr.to_vec().unwrap();
  let mut mlx_nibbles = vec![vec![0u32; in_features]; out_features];
  for o in 0..out_features {
    for pi in 0..(in_features / 8) {
      let word = weight_packed[o * (in_features / 8) + pi];
      for k in 0..8 {
        mlx_nibbles[o][pi * 8 + k] = (word >> (k as u32 * AWQ_BITS)) & AWQ_NIBBLE_MASK;
      }
    }
  }

  // Verify: mlx_nibbles[o][i] == awq_unpacked[i][o] (the transpose).
  for o in 0..out_features {
    for i in 0..in_features {
      assert_eq!(
        mlx_nibbles[o][i], awq_unpacked[i][o],
        "MLX-format nibble at (o={o}, i={i}) must equal AWQ-format nibble at (i={i}, o={o})"
      );
    }
  }

  // Verify MLX-dequant matches AWQ-dequant at every (i, o, group).
  let scales_flat: Vec<f32> = scales_arr.to_vec().unwrap();
  let biases_flat: Vec<f32> = biases_arr.to_vec().unwrap();
  for o in 0..out_features {
    for g in 0..n_groups {
      // Per-group scale + bias (MLX layout: [o, g]).
      let mlx_scale = scales_flat[o * n_groups + g];
      let mlx_bias = biases_flat[o * n_groups + g];
      // AWQ scale + zero for this (group, out) — AWQ scales/zeros are
      // per group (n_groups, out_features).
      let awq_scale = scales_data[g * out_features + o];
      let awq_zero = qzero_unpacked[g][o];
      // Every nibble in this group must dequantize identically.
      for i_in in 0..(group_size as usize) {
        let i = g * (group_size as usize) + i_in;
        let nibble = awq_unpacked[i][o];
        let awq_dq = awq_dequant(nibble, awq_zero, awq_scale);
        let mlx_dq = mlx_dequant(nibble, mlx_scale, mlx_bias);
        assert!(
          (awq_dq - mlx_dq).abs() < 1e-4,
          "AWQ dequant {awq_dq} != MLX dequant {mlx_dq} at (o={o}, g={g}, i={i}, nibble={nibble})"
        );
      }
    }
  }
}

/// Multiple AWQ-formatted layers in one input map: all transform correctly,
/// the PerLayerQuantization is a single global entry, and the non-AWQ keys
/// pass through verbatim.
#[test]
fn transform_awq_weights_handles_multiple_layers() {
  let group_size = 4u32;
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;
  // Two layers with all-zero qweight/qzeros + nonzero scales — verify both
  // exist + the pass-through key is preserved.
  let make_weights = |prefix: &str| -> Vec<(String, Array)> {
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
    let scales_data: Vec<f32> = (0..n_groups * out_features)
      .map(|i| 0.1_f32 * (i as f32 + 1.0))
      .collect();
    let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
    vec![
      (format!("{prefix}.qweight"), qw),
      (format!("{prefix}.qzeros"), qz),
      (format!("{prefix}.scales"), sc),
    ]
  };
  let mut weights: Weights = HashMap::new();
  for (k, v) in make_weights("layer0.q") {
    weights.insert(k, v);
  }
  for (k, v) in make_weights("layer1.q") {
    weights.insert(k, v);
  }
  // Pass-through key (e.g. `embed_tokens.weight`).
  let passthrough = Array::from_slice::<f32>(&[1.0_f32; 16], &(2usize, 8)).unwrap();
  weights.insert("embed_tokens.weight".to_string(), passthrough);

  let config = AwqLoadConfig {
    bits: 4,
    group_size,
    zero_point: true,
    version: String::new(),
  };
  let (out, plq) = transform_awq_weights(weights, &config).expect("transform");

  // Both layers transformed.
  assert!(out.contains_key("layer0.q.weight"));
  assert!(out.contains_key("layer0.q.scales"));
  assert!(out.contains_key("layer0.q.biases"));
  assert!(out.contains_key("layer1.q.weight"));
  assert!(out.contains_key("layer1.q.scales"));
  assert!(out.contains_key("layer1.q.biases"));
  // Originals gone.
  assert!(!out.contains_key("layer0.q.qweight"));
  assert!(!out.contains_key("layer1.q.qzeros"));
  // Pass-through preserved.
  let mut pt = out
    .get("embed_tokens.weight")
    .expect("pass-through")
    .try_clone()
    .unwrap();
  assert_eq!(pt.shape(), vec![2, 8]);
  assert_eq!(pt.to_vec::<f32>().unwrap(), vec![1.0_f32; 16]);
  // PerLayerQuantization global is set, per-layer empty.
  let g = plq.quantization.unwrap();
  assert_eq!(g.group_size, group_size as i32);
  assert_eq!(g.bits, 4);
  assert!(plq.per_layer.is_empty());
}

/// A `.qweight` with no `.scales` companion is rejected with a typed
/// [`Error::MissingKey`] naming the missing `.scales` key (mirrors mlx-lm's
/// implicit `KeyError`, `utils.py:109`).
#[test]
fn transform_awq_weights_rejects_missing_scales() {
  let in_features = 8usize;
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("layer.qweight".to_string(), qw);
  // No scales, no qzeros.
  let config = AwqLoadConfig::default();
  let err = transform_awq_weights(weights, &config).unwrap_err();
  let Error::MissingKey(p) = &err else {
    panic!("expected Error::MissingKey, got {err:?}");
  };
  assert_eq!(p.key(), "layer.scales");
  assert!(
    p.context()
      .contains("AWQ `.qweight` missing its `.scales` companion"),
    "context names the rule: {}",
    p.context()
  );
}

/// qweight/scales shape mismatch is rejected with a clear ShapePairMismatch.
#[test]
fn transform_awq_weights_rejects_mismatched_shapes() {
  let qw = Array::from_slice::<u32>(&[0u32; 8], &(8usize, 1)).unwrap();
  // Mismatched scales: should be [n_groups=2, out_features=8] but we give [4, 8]
  let sc = Array::from_slice::<f32>(&[0.1_f32; 32], &(4usize, 8)).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("layer.qweight".to_string(), qw);
  weights.insert("layer.scales".to_string(), sc);
  let config = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: String::new(),
  };
  let err = transform_awq_weights(weights, &config).unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(ref p) if matches!(p.inner(), Error::ShapePairMismatch(_))),
    "expected LayerKeyed(ShapePairMismatch), got {err:?}"
  );
}

/// A `.g_idx` key (GPTQ non-contiguous-group reorder) is rejected upfront.
#[test]
fn transform_awq_weights_rejects_g_idx() {
  let qw = Array::from_slice::<u32>(&[0u32; 8], &(8usize, 1)).unwrap();
  let sc = Array::from_slice::<f32>(&[0.1_f32; 16], &(2usize, 8)).unwrap();
  let gidx = Array::from_slice::<i32>(&[0i32, 1, 0, 1, 0, 1, 0, 1], &(8usize,)).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("layer.qweight".to_string(), qw);
  weights.insert("layer.scales".to_string(), sc);
  weights.insert("layer.g_idx".to_string(), gidx);
  let config = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: String::new(),
  };
  let err = transform_awq_weights(weights, &config).unwrap_err();
  let msg = format!("{err:?}");
  assert!(
    msg.contains("g_idx"),
    "error must mention g_idx, got: {msg}"
  );
}

/// Non-4 bits is rejected with a typed [`Error::OutOfRange`] that names the
/// `must be 4` requirement and the offending value.
#[test]
fn transform_awq_weights_rejects_non_4_bits() {
  let weights: Weights = HashMap::new();
  let config = AwqLoadConfig {
    bits: 8,
    ..AwqLoadConfig::default()
  };
  let err = transform_awq_weights(weights, &config).unwrap_err();
  let Error::OutOfRange(p) = &err else {
    panic!("expected Error::OutOfRange, got {err:?}");
  };
  assert!(
    p.context().contains("AWQ bits"),
    "context names the AWQ bits rule: {}",
    p.context()
  );
  assert!(
    p.requirement().contains("must be 4"),
    "requirement names the constraint: {}",
    p.requirement()
  );
  assert_eq!(p.value(), "8");
}

/// Symmetric quantization (`zero_point: false`): biases are computed from
/// the implicit `2^(bits-1) = 8` zero point, NOT from any qzeros that
/// might happen to be present.
#[test]
fn transform_awq_weights_symmetric_uses_implicit_zero() {
  let in_features = 8usize;
  let out_features = 8usize;
  let group_size = 4u32;
  let n_groups = 2usize;
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  // Scale = 1.0 everywhere so the bias check is trivial: bias = -8 * 1 = -8.
  let scales_data: Vec<f32> = vec![1.0_f32; n_groups * out_features];
  let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("layer.qweight".to_string(), qw);
  weights.insert("layer.scales".to_string(), sc);

  let config = AwqLoadConfig {
    bits: 4,
    group_size,
    zero_point: false,
    version: String::new(),
  };
  let (out, _) = transform_awq_weights(weights, &config).expect("transform");
  let mut biases_arr = out
    .get("layer.biases")
    .expect("layer.biases")
    .try_clone()
    .unwrap();
  let biases: Vec<f32> = biases_arr.to_vec().unwrap();
  // Every entry must be exactly -8.0.
  for &b in &biases {
    assert!(
      (b + 8.0_f32).abs() < 1e-5,
      "symmetric bias must be -2^(bits-1) * scale = -8.0, got {b}"
    );
  }
}

/// Empty input (no `.qweight` keys) is a no-op: pass-through verbatim plus
/// a `PerLayerQuantization` with the requested global params.
#[test]
fn transform_awq_weights_empty_input_is_noop() {
  let pt = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0], &(3usize,)).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("layer.weight".to_string(), pt);

  let config = AwqLoadConfig::default();
  let (out, plq) = transform_awq_weights(weights, &config).expect("transform");
  // Pass-through preserved.
  let mut got = out
    .get("layer.weight")
    .expect("pass-through")
    .try_clone()
    .unwrap();
  assert_eq!(got.to_vec::<f32>().unwrap(), vec![1.0_f32, 2.0, 3.0]);
  // Global quant set from config defaults.
  let g = plq.quantization.unwrap();
  assert_eq!(g.bits, 4);
  assert_eq!(g.group_size, 128);
  assert_eq!(g.mode, QuantMode::Affine);
}

// ──────────────── version validation ────────────────

/// Helper: build a minimal valid GEMM-shaped weights map (in=8, out=8,
/// gs=4, ng=2). Lets the version tests focus on the version field without
/// re-deriving the shape arithmetic each time.
fn awq_gemm_fixture_weights() -> Weights {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
  let scales_data: Vec<f32> = vec![1.0_f32; n_groups * out_features];
  let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
  let mut w: Weights = HashMap::new();
  w.insert("layer.qweight".to_string(), qw);
  w.insert("layer.qzeros".to_string(), qz);
  w.insert("layer.scales".to_string(), sc);
  w
}

/// `version = "gemv"` is REJECTED at the top of transform_awq_weights
/// (before any conversion work). The error message must name the offending
/// version and call out "not yet supported" — the spec-required signal.
#[test]
fn transform_awq_weights_rejects_gemv_version() {
  let config = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemv".into(),
  };
  let err = transform_awq_weights(awq_gemm_fixture_weights(), &config).unwrap_err();
  match err {
    Error::UnknownEnumValue(ref payload) => {
      assert_eq!(
        payload.value(),
        "gemv",
        "UnknownEnumValue must name the offending 'gemv' version"
      );
    }
    other => panic!("expected Error::UnknownEnumValue, got: {other:?}"),
  }
}

/// An unknown version string is REJECTED with the version named
/// in the message.
#[test]
fn transform_awq_weights_rejects_unknown_version() {
  let config = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "unsupported".into(),
  };
  let err = transform_awq_weights(awq_gemm_fixture_weights(), &config).unwrap_err();
  match err {
    Error::UnknownEnumValue(ref payload) => {
      assert_eq!(
        payload.value(),
        "unsupported",
        "UnknownEnumValue must name the offending version"
      );
    }
    other => panic!("expected Error::UnknownEnumValue, got: {other:?}"),
  }
}

/// Empty version (the serde default) is ACCEPTED — older AutoAWQ
/// checkpoints + mlxrs-internal construction both leave it empty.
#[test]
fn transform_awq_weights_accepts_empty_version() {
  let config = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: String::new(),
  };
  transform_awq_weights(awq_gemm_fixture_weights(), &config)
    .expect("empty version (serde default) must be accepted");
}

/// Explicit `"gemm"` is ACCEPTED.
#[test]
fn transform_awq_weights_accepts_gemm_version() {
  let config = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  transform_awq_weights(awq_gemm_fixture_weights(), &config)
    .expect("explicit 'gemm' version must be accepted");
}

// ──────────────── I32 qweight/qzeros acceptance ────────────────

/// A full I32 fixture (both qweight + qzeros allocated as `torch.int32`,
/// as AutoAWQ's `WQLinear_GEMM` does) round-trips through `transform_awq_weights`.
/// Includes a qweight value with the high bit SET — the bit-pattern that
/// would corrupt under a value-preserving `astype`.
#[test]
fn transform_awq_weights_accepts_i32_qweight_and_qzeros() {
  // Same shapes as the round-trip fixture: in=8, out=8, gs=4, ng=2.
  let in_features = 8usize;
  let out_features = 8usize;
  let group_size = 4u32;
  let n_groups = 2usize;
  // Pack a row with the high nibble set so the resulting u32 word's MSB
  // is `0xF` — when allocated as i32 this is a negative number.
  let qweight_data_u32: Vec<u32> = (0..in_features)
    .map(|i| {
      let nibbles = [
        (i % 16) as u32,
        ((i + 1) % 16) as u32,
        ((i + 2) % 16) as u32,
        ((i + 3) % 16) as u32,
        ((i + 4) % 16) as u32,
        ((i + 5) % 16) as u32,
        ((i + 6) % 16) as u32,
        0xF_u32, // high nibble = 0xF → MSB set when packed at AWQ_SHIFTS[7]=28
      ];
      awq_pack_one_row(nibbles)
    })
    .collect();
  let qweight_data_i32: Vec<i32> = qweight_data_u32.iter().map(|&u| u as i32).collect();
  assert!(
    qweight_data_i32.iter().any(|&v| v < 0),
    "fixture must contain a negative i32 to exercise the high-bit case"
  );

  // qzeros: also int32, with same fixture as the round-trip test.
  let qzero_unpacked: Vec<Vec<u32>> = (0..n_groups)
    .map(|g| (0..out_features).map(|o| ((g + o) % 16) as u32).collect())
    .collect();
  let qzeros_data_u32: Vec<u32> = (0..n_groups)
    .map(|g| {
      let row: [u32; 8] = qzero_unpacked[g].to_vec().try_into().unwrap();
      awq_pack_one_row(row)
    })
    .collect();
  let qzeros_data_i32: Vec<i32> = qzeros_data_u32.iter().map(|&u| u as i32).collect();

  let scales_data: Vec<f32> = (0..n_groups * out_features)
    .map(|i| 0.1_f32 * (i as f32 + 1.0))
    .collect();

  // Build the I32 weights map.
  let qw_i32 = Array::from_slice::<i32>(&qweight_data_i32, &(in_features, 1)).unwrap();
  let qz_i32 = Array::from_slice::<i32>(&qzeros_data_i32, &(n_groups, 1)).unwrap();
  let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
  let mut weights_i32: Weights = HashMap::new();
  weights_i32.insert("layer.qweight".to_string(), qw_i32);
  weights_i32.insert("layer.qzeros".to_string(), qz_i32);
  weights_i32.insert("layer.scales".to_string(), sc);

  let config = AwqLoadConfig {
    bits: 4,
    group_size,
    zero_point: true,
    version: "gemm".into(),
  };
  let (out, plq) =
    transform_awq_weights(weights_i32, &config).expect("i32 qweight + qzeros accepted");

  // The transformed `.weight` must be u32 (the MLX quantized output dtype).
  let weight_arr = out.get("layer.weight").expect("layer.weight");
  assert_eq!(weight_arr.dtype().unwrap(), Dtype::U32);
  // PLQ unchanged.
  let g = plq.quantization.expect("global quant");
  assert_eq!(g.bits, 4);
  assert_eq!(g.group_size, group_size as i32);
}

/// Pack a known-negative i32 fixture and verify the resulting
/// MLX-format output bit-pattern matches what the equivalent U32 input
/// produces — confirming the i32 path is bit-preserving end-to-end
/// (NOT value-preserving via `astype`, which would clamp negatives to 0).
#[test]
fn transform_awq_weights_preserves_bit_pattern_on_i32_input() {
  let in_features = 8usize;
  let out_features = 8usize;
  let group_size = 4u32;
  let n_groups = 2usize;

  // Build identical fixtures, one allocated as u32, the other as the
  // bitwise-equal i32 — feed both through and compare the .weight output.
  let qweight_data_u32: Vec<u32> = (0..in_features)
    .map(|i| {
      // Same scrambled nibbles with high bit set in MSB slot.
      let nibbles = [
        (i % 16) as u32,
        ((i + 7) % 16) as u32,
        ((i + 3) % 16) as u32,
        ((i + 5) % 16) as u32,
        ((i + 2) % 16) as u32,
        ((i + 6) % 16) as u32,
        ((i + 1) % 16) as u32,
        0xF_u32,
      ];
      awq_pack_one_row(nibbles)
    })
    .collect();
  let qweight_data_i32: Vec<i32> = qweight_data_u32.iter().map(|&u| u as i32).collect();

  let qzeros_data_u32: Vec<u32> = vec![0_u32; n_groups];
  let qzeros_data_i32: Vec<i32> = vec![0_i32; n_groups];
  let scales_data: Vec<f32> = (0..n_groups * out_features)
    .map(|i| 0.5_f32 + (i as f32) * 0.01)
    .collect();

  let build = |qw_dtype_i32: bool| -> Weights {
    let mut w: Weights = HashMap::new();
    if qw_dtype_i32 {
      w.insert(
        "layer.qweight".to_string(),
        Array::from_slice::<i32>(&qweight_data_i32, &(in_features, 1)).unwrap(),
      );
      w.insert(
        "layer.qzeros".to_string(),
        Array::from_slice::<i32>(&qzeros_data_i32, &(n_groups, 1)).unwrap(),
      );
    } else {
      w.insert(
        "layer.qweight".to_string(),
        Array::from_slice::<u32>(&qweight_data_u32, &(in_features, 1)).unwrap(),
      );
      w.insert(
        "layer.qzeros".to_string(),
        Array::from_slice::<u32>(&qzeros_data_u32, &(n_groups, 1)).unwrap(),
      );
    }
    w.insert(
      "layer.scales".to_string(),
      Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap(),
    );
    w
  };

  let cfg = AwqLoadConfig {
    bits: 4,
    group_size,
    zero_point: true,
    version: "gemm".into(),
  };
  let (out_u32, _) = transform_awq_weights(build(false), &cfg).expect("u32 path");
  let (out_i32, _) = transform_awq_weights(build(true), &cfg).expect("i32 path");

  let mut w_u32 = out_u32.get("layer.weight").unwrap().try_clone().unwrap();
  let mut w_i32 = out_i32.get("layer.weight").unwrap().try_clone().unwrap();
  let u32_buf: Vec<u32> = w_u32.to_vec().unwrap();
  let i32_buf: Vec<u32> = w_i32.to_vec().unwrap();
  assert_eq!(
    u32_buf, i32_buf,
    "i32 qweight must produce the SAME .weight bit-pattern as the equivalent u32 input"
  );
}

// ──────────────── .scales dtype validation ────────────────

/// Integer `.scales` (`i32`) is REJECTED — a hostile/malformed
/// checkpoint with integer scales would silently CAST every model float
/// to that integer through the dtype-unification loop. The validator
/// fires first and names the offending layer + the rejection reason.
#[test]
fn transform_awq_weights_rejects_integer_scales_dtype() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
  // INTEGER `.scales` — the bug class.
  let sc_int = Array::from_slice::<i32>(
    &vec![1_i32; n_groups * out_features],
    &(n_groups, out_features),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("model.layer0.qweight".to_string(), qw);
  weights.insert("model.layer0.qzeros".to_string(), qz);
  weights.insert("model.layer0.scales".to_string(), sc_int);

  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  let err = transform_awq_weights(weights, &cfg).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.layer0.scales"),
        "LayerKeyed must name the offending layer's `.scales` key, got layer={:?}",
        payload.layer()
      );
      assert!(
        matches!(payload.inner(), Error::UnsupportedDtype(_)),
        "inner must be UnsupportedDtype for non-floating scales, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// `u8` (unsigned narrow int) `.scales` is REJECTED with the same
/// error shape. Confirms the gate fires for narrow ints too — not just
/// the canonical `i32` case.
#[test]
fn transform_awq_weights_rejects_uint_scales_dtype() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
  let sc_u8 = Array::from_slice::<u8>(
    &vec![1_u8; n_groups * out_features],
    &(n_groups, out_features),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert("model.layer0.qweight".to_string(), qw);
  weights.insert("model.layer0.qzeros".to_string(), qz);
  weights.insert("model.layer0.scales".to_string(), sc_u8);

  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  let err = transform_awq_weights(weights, &cfg).unwrap_err();
  match err {
    Error::LayerKeyed(ref payload) => {
      assert!(
        payload.layer().contains("model.layer0.scales"),
        "LayerKeyed must name the offending layer's `.scales`, got layer={:?}",
        payload.layer()
      );
      assert!(
        matches!(payload.inner(), Error::UnsupportedDtype(_)),
        "inner must be UnsupportedDtype for non-floating scales, got: {:?}",
        payload.inner()
      );
    }
    other => panic!("expected Error::LayerKeyed, got: {other:?}"),
  }
}

/// HIERARCHICAL heterogeneous-precision `.scales` (mixing
/// dtypes where one IS a true superset of the others) must resolve to
/// the higher-precision target. This covers F32+F16 → F32 and F64+BF16
/// → F64. The F16+BF16 case is carved out (no superset relation,
/// see `..._escalates_f16_plus_bf16_to_f32`) — this test guards the
/// remaining cases where the simple "highest rank wins" answer IS still
/// correct (and lossless).
#[test]
fn resolve_awq_model_dtype_uses_highest_when_hierarchical() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  // Case 1: F32 + F16 → F32 (F32 is a strict superset of F16).
  let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
    .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
    .collect();
  let f32_scales_data: Vec<f32> = (0..n_groups * out_features)
    .map(|i| 0.5 + 0.01 * (i as f32))
    .collect();

  let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_b = Array::from_slice::<f32>(&f32_scales_data, &(n_groups, out_features)).unwrap();

  let weights: Weights = HashMap::from([
    ("layer_a.qweight".to_string(), qw_a),
    ("layer_a.scales".to_string(), sc_a),
    ("layer_b.qweight".to_string(), qw_b),
    ("layer_b.scales".to_string(), sc_b),
  ]);
  let mut prefixes: Vec<String> = vec!["layer_a".to_string(), "layer_b".to_string()];
  prefixes.sort();

  validate_awq_scales_are_floating(&weights, &prefixes).expect("both floating, must pass");
  let resolved = resolve_awq_model_dtype(&weights, &prefixes)
    .unwrap()
    .expect("some dtype");
  assert_eq!(
    resolved,
    Dtype::F32,
    "F32+F16 hierarchical must resolve to F32 (superset), got {resolved:?}"
  );

  // Case 2: F64 + BF16 → F64 (F64 is a strict superset of BF16:
  // more mantissa bits AND more exponent bits).
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();
  let f64_scales_data: Vec<f64> = (0..n_groups * out_features)
    .map(|i| 0.5 + 0.001 * (i as f64))
    .collect();

  let qw_c = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_c = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_d = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_d = Array::from_slice::<f64>(&f64_scales_data, &(n_groups, out_features)).unwrap();

  let weights2: Weights = HashMap::from([
    ("layer_c.qweight".to_string(), qw_c),
    ("layer_c.scales".to_string(), sc_c),
    ("layer_d.qweight".to_string(), qw_d),
    ("layer_d.scales".to_string(), sc_d),
  ]);
  let mut prefixes2: Vec<String> = vec!["layer_c".to_string(), "layer_d".to_string()];
  prefixes2.sort();

  validate_awq_scales_are_floating(&weights2, &prefixes2).expect("both floating, must pass");
  let resolved2 = resolve_awq_model_dtype(&weights2, &prefixes2)
    .unwrap()
    .expect("some dtype");
  assert_eq!(
    resolved2,
    Dtype::F64,
    "F64+BF16 hierarchical must resolve to F64 (superset), got {resolved2:?}"
  );
}

/// F16 and BF16 mixed alone (no F32/F64 present) must
/// escalate to F32. Neither half-float is a superset of the other —
/// F16 has more mantissa bits, BF16 has more exponent bits — so any
/// pick within the halves would be lossy for one side. The escalation
/// to F32 is order-independent (HashMap iteration may visit them in
/// either order via `prefixes`).
#[test]
fn resolve_awq_model_dtype_escalates_f16_plus_bf16_to_f32() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
    .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
    .collect();
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();

  let build = || {
    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
    ]);
    weights
  };

  // Forward order: [layer_a (F16), layer_b (BF16)].
  let weights = build();
  let prefixes: Vec<String> = vec!["layer_a".to_string(), "layer_b".to_string()];
  validate_awq_scales_are_floating(&weights, &prefixes).expect("both floating, must pass");
  let resolved = resolve_awq_model_dtype(&weights, &prefixes)
    .unwrap()
    .expect("some dtype");
  assert_eq!(
    resolved,
    Dtype::F32,
    "F16+BF16 must escalate to F32 (no half is a superset), got {resolved:?}"
  );

  // Reverse order: [layer_b (BF16), layer_a (F16)]. Result must be
  // identical — escalation does not depend on iteration order.
  let weights_r = build();
  let prefixes_r: Vec<String> = vec!["layer_b".to_string(), "layer_a".to_string()];
  validate_awq_scales_are_floating(&weights_r, &prefixes_r).expect("both floating, must pass");
  let resolved_r = resolve_awq_model_dtype(&weights_r, &prefixes_r)
    .unwrap()
    .expect("some dtype");
  assert_eq!(
    resolved_r,
    Dtype::F32,
    "F16+BF16 reversed order must still escalate to F32, got {resolved_r:?}"
  );
}

/// When F32 is already present alongside F16+BF16, it short-circuits
/// the escalation — F32 wins on rank and is already a superset of both
/// halves, no need to "escalate" further.
#[test]
fn resolve_awq_model_dtype_escalates_f16_plus_bf16_plus_f32_stays_at_f32() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
    .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
    .collect();
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();
  let f32_scales_data: Vec<f32> = (0..n_groups * out_features)
    .map(|i| 0.25 + 0.001 * (i as f32))
    .collect();

  let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_b = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_c = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_c = Array::from_slice::<f32>(&f32_scales_data, &(n_groups, out_features)).unwrap();

  let weights: Weights = HashMap::from([
    ("layer_a.qweight".to_string(), qw_a),
    ("layer_a.scales".to_string(), sc_a),
    ("layer_b.qweight".to_string(), qw_b),
    ("layer_b.scales".to_string(), sc_b),
    ("layer_c.qweight".to_string(), qw_c),
    ("layer_c.scales".to_string(), sc_c),
  ]);
  let prefixes: Vec<String> = vec![
    "layer_a".to_string(),
    "layer_b".to_string(),
    "layer_c".to_string(),
  ];
  validate_awq_scales_are_floating(&weights, &prefixes).expect("all floating, must pass");
  let resolved = resolve_awq_model_dtype(&weights, &prefixes)
    .unwrap()
    .expect("some dtype");
  assert_eq!(
    resolved,
    Dtype::F32,
    "F16+BF16+F32 must stay at F32 (F32 already > BF16 rank, no escalation), got {resolved:?}"
  );
}

/// When F64 is already present alongside F16+BF16, it stays at F64
/// (F64 outranks F32; F64 is also a superset of both halves so no
/// escalation is needed).
#[test]
fn resolve_awq_model_dtype_escalates_f16_plus_bf16_with_f64_stays_at_f64() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
    .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
    .collect();
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();
  let f64_scales_data: Vec<f64> = (0..n_groups * out_features)
    .map(|i| 0.25 + 0.001 * (i as f64))
    .collect();

  let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_b = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_c = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_c = Array::from_slice::<f64>(&f64_scales_data, &(n_groups, out_features)).unwrap();

  let weights: Weights = HashMap::from([
    ("layer_a.qweight".to_string(), qw_a),
    ("layer_a.scales".to_string(), sc_a),
    ("layer_b.qweight".to_string(), qw_b),
    ("layer_b.scales".to_string(), sc_b),
    ("layer_c.qweight".to_string(), qw_c),
    ("layer_c.scales".to_string(), sc_c),
  ]);
  let prefixes: Vec<String> = vec![
    "layer_a".to_string(),
    "layer_b".to_string(),
    "layer_c".to_string(),
  ];
  validate_awq_scales_are_floating(&weights, &prefixes).expect("all floating, must pass");
  let resolved = resolve_awq_model_dtype(&weights, &prefixes)
    .unwrap()
    .expect("some dtype");
  assert_eq!(
    resolved,
    Dtype::F64,
    "F16+BF16+F64 must stay at F64 (F64 already > BF16 rank, no escalation), got {resolved:?}"
  );
}

/// END-TO-END value preservation: a checkpoint with F16
/// `.scales` carrying the value `1.0009765625` (= 1 + 2⁻¹⁰, exactly
/// representable in F16 but NOT in BF16 — BF16's smallest delta near
/// 1 is 2⁻⁷ ≈ 0.0078) and a sibling BF16 `.scales` layer must round-
/// trip through `transform_awq_weights` with that F16 value PRESERVED.
///
/// Under a rank-only policy, the resolver returns BF16, the unification
/// loop casts F16 → BF16, and `1.0009765625` collapses to `1.0`
/// (silently corrupting every F16 scale value). With the escalation the
/// resolver picks F32, the cast is F16 → F32 (lossless), and the original
/// value survives.
#[test]
fn transform_awq_weights_preserves_f16_precision_when_mixed_with_bf16() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  // F16 layer: every scale = 1.0009765625 (= 1 + 2⁻¹⁰), exactly
  // representable in F16 (bits 0x3C01). BF16 has 7 mantissa bits so
  // its smallest delta near 1.0 is 2⁻⁷ ≈ 0.0078125 — the value
  // would round to 1.0 if cast to BF16.
  let f16_value = half::f16::from_bits(0x3C01);
  assert_eq!(
    f16_value.to_f32(),
    1.0 + (2.0_f32).powi(-10),
    "F16 fixture value must be exactly 1 + 2^-10"
  );
  // Sanity-check the BF16 truncation the test catches:
  let bf_round = half::bf16::from_f32(f16_value.to_f32());
  assert_eq!(
    bf_round.to_f32(),
    1.0,
    "pre-condition: casting F16 1.0009765625 → BF16 must truncate to 1.0 \
       (this is the lossy behavior the F16+BF16→F32 escalation prevents)"
  );

  let f16_scales_data: Vec<half::f16> = vec![f16_value; n_groups * out_features];
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();

  let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_b = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

  let weights: Weights = HashMap::from([
    ("layer_a.qweight".to_string(), qw_a),
    ("layer_a.scales".to_string(), sc_a),
    ("layer_b.qweight".to_string(), qw_b),
    ("layer_b.scales".to_string(), sc_b),
  ]);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false, // symmetric — no qzeros required.
    version: "gemm".into(),
  };

  let (out, _) = transform_awq_weights(weights, &cfg).expect("transform must succeed");

  // The resolved/unified dtype must be F32 (escalation kicked in).
  let mut sc_a_out = out
    .get("layer_a.scales")
    .expect("converted layer_a.scales present")
    .try_clone()
    .unwrap();
  assert_eq!(
    sc_a_out.dtype().unwrap(),
    Dtype::F32,
    "unified dtype must be F32 under the F16+BF16→F32 escalation"
  );

  // Read back as F32 and verify EVERY element still holds 1.0009765625.
  let vals: Vec<f32> = sc_a_out.to_vec().expect("read back as F32");
  for (i, &v) in vals.iter().enumerate() {
    assert_eq!(
      v,
      1.0 + (2.0_f32).powi(-10),
      "layer_a.scales[{i}] = {v} (bits 0x{:08X}) — F16 1.0009765625 must NOT have \
         been truncated through BF16 (would land at 1.0 == 0x3F800000)",
      v.to_bits()
    );
  }

  // layer_b (originally BF16) was also unified to F32 (lossless from
  // BF16 → F32 — BF16 mantissa fits in F32's 23 bits trivially).
  let sc_b_out = out
    .get("layer_b.scales")
    .expect("converted layer_b.scales present");
  assert_eq!(
    sc_b_out.dtype().unwrap(),
    Dtype::F32,
    "layer_b.scales must also be unified to F32"
  );
}

/// END-TO-END order-independence: same as the preservation
/// test above, but with prefix names swapped lexicographically (BF16
/// layer named to sort BEFORE the F16 layer). Guards against any
/// regression that would reintroduce the lex-last-wins behavior — the
/// resolver must still escalate to F32 and the F16 value must still
/// survive regardless of which prefix iterates last.
#[test]
fn transform_awq_weights_preserves_f16_precision_with_bf16_in_reversed_prefix_order() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  // Same F16 fixture value as the forward-order test (= 1 + 2⁻¹⁰).
  let f16_value = half::f16::from_bits(0x3C01);

  let f16_scales_data: Vec<half::f16> = vec![f16_value; n_groups * out_features];
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();

  // Naming: "alpha" (BF16) sorts BEFORE "zeta" (F16). Under a
  // lex-last policy this would have picked F16 (zeta last); under
  // a rank-only policy it would pick BF16. With the escalation it MUST be F32.
  let qw_alpha = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_alpha =
    Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_zeta = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_zeta =
    Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();

  let weights: Weights = HashMap::from([
    ("alpha.qweight".to_string(), qw_alpha),
    ("alpha.scales".to_string(), sc_alpha),
    ("zeta.qweight".to_string(), qw_zeta),
    ("zeta.scales".to_string(), sc_zeta),
  ]);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: "gemm".into(),
  };

  let (out, _) = transform_awq_weights(weights, &cfg).expect("transform must succeed");

  let mut sc_zeta_out = out
    .get("zeta.scales")
    .expect("converted zeta.scales present")
    .try_clone()
    .unwrap();
  assert_eq!(
    sc_zeta_out.dtype().unwrap(),
    Dtype::F32,
    "unified dtype must be F32 regardless of prefix order"
  );
  let vals: Vec<f32> = sc_zeta_out.to_vec().expect("read back as F32");
  for (i, &v) in vals.iter().enumerate() {
    assert_eq!(
      v,
      1.0 + (2.0_f32).powi(-10),
      "zeta.scales[{i}] = {v} — F16 precision must be preserved in reversed-order layout"
    );
  }
}

// ──────────────── collision with stale `.weight`/`.biases` ────────────────

/// Input carries `<prefix>.qweight + .scales + .qzeros + .weight` —
/// a stale dense `.weight` next to a valid AWQ triple. The converter would
/// emit `<prefix>.weight` from the AWQ conversion, then the remainder pass
/// would OVERWRITE it with the stale input. Preflight collision check
/// must REJECT this with a clear message naming the prefix + "collision".
#[test]
fn transform_awq_weights_rejects_collision_with_stale_weight() {
  let mut weights = awq_gemm_fixture_weights();
  // Add a stale dense `.weight` next to the AWQ triple. The exact shape
  // doesn't matter — the collision check fires at preflight, before any
  // shape validation.
  let stale = Array::from_slice::<f32>(&[0.0_f32; 16], &(2usize, 8)).unwrap();
  weights.insert("layer.weight".to_string(), stale);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  let err = transform_awq_weights(weights, &cfg).unwrap_err();
  let Error::KeyCollision(p) = &err else {
    panic!("expected Error::KeyCollision, got: {err:?}");
  };
  assert_eq!(p.key(), "layer.weight");
  assert!(
    p.context().contains(".qweight") && p.context().contains(".weight"),
    "context must name both the qweight and weight, got: {}",
    p.context()
  );
}

/// Same collision, but with `<prefix>.biases` instead of `.weight`.
#[test]
fn transform_awq_weights_rejects_collision_with_stale_biases() {
  let mut weights = awq_gemm_fixture_weights();
  let stale = Array::from_slice::<f32>(&[0.0_f32; 8], &(8usize,)).unwrap();
  weights.insert("layer.biases".to_string(), stale);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  let err = transform_awq_weights(weights, &cfg).unwrap_err();
  let Error::KeyCollision(p) = &err else {
    panic!("expected Error::KeyCollision, got: {err:?}");
  };
  assert_eq!(p.key(), "layer.biases");
  assert!(
    p.context().contains(".qweight") && p.context().contains(".biases"),
    "context must name both the qweight and biases, got: {}",
    p.context()
  );
}

/// An UNRELATED `.weight` key (different prefix) must NOT trigger
/// the collision check — the conversion proceeds and the unrelated dense
/// key passes through verbatim.
#[test]
fn transform_awq_weights_accepts_unrelated_weight_keys() {
  let mut weights = awq_gemm_fixture_weights();
  // Distinct prefix — embed_tokens.weight is the canonical pass-through.
  let pt = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap();
  weights.insert("embed_tokens.weight".to_string(), pt);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  let (out, _) = transform_awq_weights(weights, &cfg).expect("unrelated .weight must pass");
  // AWQ output present + pass-through preserved.
  assert!(
    out.contains_key("layer.weight"),
    "AWQ-converted .weight must be present"
  );
  assert!(
    out.contains_key("embed_tokens.weight"),
    "unrelated .weight must be preserved"
  );
}

/// BOTH stale `.weight` + `.biases` present → still errors (the
/// first detected one is fine; this confirms the second one wouldn't
/// somehow be quiet either, by removing the first and re-running).
#[test]
fn transform_awq_weights_rejects_collision_with_both_stale_keys() {
  let mut weights = awq_gemm_fixture_weights();
  let stale_w = Array::from_slice::<f32>(&[0.0_f32; 16], &(2usize, 8)).unwrap();
  let stale_b = Array::from_slice::<f32>(&[0.0_f32; 8], &(8usize,)).unwrap();
  weights.insert("layer.weight".to_string(), stale_w);
  weights.insert("layer.biases".to_string(), stale_b);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: true,
    version: "gemm".into(),
  };
  let err = transform_awq_weights(weights, &cfg).unwrap_err();
  assert!(
    matches!(err, Error::KeyCollision(_)),
    "must reject with KeyCollision, got: {err:?}"
  );

  // Now drop the .weight collision; the .biases collision alone must
  // still fire. (Confirms the gate is per-sibling, not "first-only".)
  let mut weights2 = awq_gemm_fixture_weights();
  let stale_b2 = Array::from_slice::<f32>(&[0.0_f32; 8], &(8usize,)).unwrap();
  weights2.insert("layer.biases".to_string(), stale_b2);
  let err2 = transform_awq_weights(weights2, &cfg).unwrap_err();
  match err2 {
    Error::KeyCollision(ref p) => {
      assert!(
        p.key().contains("layer.biases"),
        "must name the .biases collision when .weight is absent, got key={:?}",
        p.key()
      );
    }
    other => panic!("expected Error::KeyCollision, got: {other:?}"),
  }
}

// ──────────────── scoped unification ────────────────

/// A BF16 pass-through tensor (e.g. `embed_tokens.weight`) sitting
/// next to a single AWQ-quantized layer with BF16 `.scales` must keep its
/// ORIGINAL dtype (BF16) after `transform_awq_weights`. The unification
/// cast applies only to the AWQ-generated `.scales` / `.biases`, not to
/// pass-through floating tensors. (Walking every floating key in the
/// output map would be a no-op for a checkpoint whose pass-through
/// tensors are already at the resolved dtype — but the
/// *bytes-equivalence* contract is what we want: the pass-through value
/// is not touched at all.)
#[test]
fn transform_awq_weights_does_not_widen_passthrough_bf16_tensor() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  // BF16 scales — resolves to BF16 (single dtype, no escalation).
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

  // Large-ish BF16 pass-through tensor (`embed_tokens.weight`). The
  // exact value pattern doesn't matter — we just need it READABLE so the
  // post-transform comparison can confirm byte-equivalence.
  let pt_shape = (100usize, 100usize);
  let pt_data: Vec<half::bf16> = (0..pt_shape.0 * pt_shape.1)
    .map(|i| half::bf16::from_f32(0.001 * (i as f32 % 1000.0)))
    .collect();
  let pt = Array::from_slice::<half::bf16>(&pt_data, &pt_shape).unwrap();

  let weights: Weights = HashMap::from([
    ("layer.qweight".to_string(), qw),
    ("layer.scales".to_string(), sc),
    ("embed_tokens.weight".to_string(), pt),
  ]);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: "gemm".into(),
  };

  let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

  // Generated `.scales` is BF16 (resolved model_dtype, no escalation).
  let sc_out = out.get("layer.scales").expect("layer.scales generated");
  assert_eq!(
    sc_out.dtype().unwrap(),
    Dtype::BF16,
    "BF16-only AWQ scales must resolve to BF16, got {:?}",
    sc_out.dtype().unwrap()
  );

  // Pass-through embed_tokens.weight keeps BF16 dtype + shape.
  let mut pt_out = out
    .get("embed_tokens.weight")
    .expect("pass-through embed_tokens.weight preserved")
    .try_clone()
    .unwrap();
  assert_eq!(
    pt_out.dtype().unwrap(),
    Dtype::BF16,
    "pass-through BF16 tensor must NOT be widened by unification"
  );
  assert_eq!(
    pt_out.shape(),
    vec![pt_shape.0, pt_shape.1],
    "pass-through shape preserved"
  );
  // Byte-equivalence: every value identical to the source.
  let pt_back: Vec<half::bf16> = pt_out.to_vec().expect("read pass-through as BF16");
  assert_eq!(
    pt_back.len(),
    pt_data.len(),
    "pass-through element count preserved"
  );
  for (i, (&got, &want)) in pt_back.iter().zip(pt_data.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "pass-through value at index {i} must be byte-identical (got 0x{:04X}, want 0x{:04X})",
      got.to_bits(),
      want.to_bits()
    );
  }
}

/// A F16 pass-through `lm_head.weight` next to TWO AWQ layers
/// (one F16 scales, one BF16 scales — triggers the F32
/// escalation per `resolve_awq_model_dtype`) must STILL be F16 after
/// `transform_awq_weights`. The escalation only applies to the
/// AWQ-generated `.scales` / `.biases`, NOT to pass-through tensors.
/// An unscoped cast would widen `lm_head.weight` from F16 to F32,
/// doubling its resident size + adding a full-size cast allocation.
#[test]
fn transform_awq_weights_does_not_widen_passthrough_f16_tensor_when_mixed_with_bf16_awq_scales() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
    .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
    .collect();
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();

  let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_b = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

  // F16 pass-through `lm_head.weight`.
  let lm_head_shape = (32usize, 16usize);
  let lm_head_data: Vec<half::f16> = (0..lm_head_shape.0 * lm_head_shape.1)
    .map(|i| half::f16::from_f32(0.01 * (i as f32 % 100.0)))
    .collect();
  let lm_head = Array::from_slice::<half::f16>(&lm_head_data, &lm_head_shape).unwrap();

  let weights: Weights = HashMap::from([
    ("layer_a.qweight".to_string(), qw_a),
    ("layer_a.scales".to_string(), sc_a),
    ("layer_b.qweight".to_string(), qw_b),
    ("layer_b.scales".to_string(), sc_b),
    ("lm_head.weight".to_string(), lm_head),
  ]);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: "gemm".into(),
  };

  let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

  // AWQ-generated outputs ESCALATED to F32.
  let sc_a_out = out.get("layer_a.scales").expect("layer_a.scales");
  assert_eq!(
    sc_a_out.dtype().unwrap(),
    Dtype::F32,
    "AWQ-generated layer_a.scales must be cast to F32 under mixed-half escalation"
  );
  let sc_b_out = out.get("layer_b.scales").expect("layer_b.scales");
  assert_eq!(
    sc_b_out.dtype().unwrap(),
    Dtype::F32,
    "AWQ-generated layer_b.scales must be cast to F32 under mixed-half escalation"
  );
  let bi_a_out = out.get("layer_a.biases").expect("layer_a.biases");
  assert_eq!(
    bi_a_out.dtype().unwrap(),
    Dtype::F32,
    "AWQ-generated layer_a.biases must be cast to F32 under mixed-half escalation"
  );
  let bi_b_out = out.get("layer_b.biases").expect("layer_b.biases");
  assert_eq!(
    bi_b_out.dtype().unwrap(),
    Dtype::F32,
    "AWQ-generated layer_b.biases must be cast to F32 under mixed-half escalation"
  );

  // BUT pass-through lm_head.weight is STILL F16 (NOT cast to F32).
  let mut lm_head_out = out
    .get("lm_head.weight")
    .expect("pass-through lm_head.weight preserved")
    .try_clone()
    .unwrap();
  assert_eq!(
    lm_head_out.dtype().unwrap(),
    Dtype::F16,
    "pass-through F16 tensor must NOT be widened to F32 by the AWQ \
       mixed-half escalation — only the AWQ-generated .scales/.biases get widened"
  );
  assert_eq!(
    lm_head_out.shape(),
    vec![lm_head_shape.0, lm_head_shape.1],
    "pass-through lm_head shape preserved"
  );
  // Byte-equivalence: every F16 value identical to the source.
  let lm_back: Vec<half::f16> = lm_head_out.to_vec().expect("read lm_head as F16");
  for (i, (&got, &want)) in lm_back.iter().zip(lm_head_data.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "lm_head.weight[{i}] must be byte-identical (got 0x{:04X}, want 0x{:04X})",
      got.to_bits(),
      want.to_bits()
    );
  }
}

/// Explicit contrast — with 1 AWQ layer (BF16 scales, no
/// escalation: resolves to BF16) + 1 F16 pass-through key, the
/// AWQ-generated `.scales` IS cast (to the resolved BF16) but the
/// pass-through F16 tensor is left at F16. Confirms the cast is
/// **scoped** to AWQ-generated keys and not blanket-applied to every
/// floating output.
#[test]
fn transform_awq_weights_widens_only_generated_scales_and_biases() {
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();
  let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

  // F16 pass-through.
  let pt_shape = (16usize, 8usize);
  let pt_data: Vec<half::f16> = (0..pt_shape.0 * pt_shape.1)
    .map(|i| half::f16::from_f32(0.01 * (i as f32)))
    .collect();
  let pt = Array::from_slice::<half::f16>(&pt_data, &pt_shape).unwrap();

  let weights: Weights = HashMap::from([
    ("layer.qweight".to_string(), qw),
    ("layer.scales".to_string(), sc),
    ("model.norm.weight".to_string(), pt),
  ]);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: "gemm".into(),
  };

  let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

  // AWQ-generated .scales: BF16 (resolved model_dtype, no escalation).
  let sc_out = out.get("layer.scales").expect("layer.scales");
  assert_eq!(
    sc_out.dtype().unwrap(),
    Dtype::BF16,
    "AWQ-generated .scales is at the resolved BF16 model_dtype"
  );
  let bi_out = out.get("layer.biases").expect("layer.biases");
  assert_eq!(
    bi_out.dtype().unwrap(),
    Dtype::BF16,
    "AWQ-generated .biases is at the resolved BF16 model_dtype"
  );

  // Pass-through F16: still F16, NOT widened to BF16.
  let mut pt_out = out
    .get("model.norm.weight")
    .expect("pass-through model.norm.weight")
    .try_clone()
    .unwrap();
  assert_eq!(
    pt_out.dtype().unwrap(),
    Dtype::F16,
    "pass-through F16 tensor must NOT be cast to the resolved BF16 — \
       unification is scoped to AWQ-generated outputs only"
  );
  // Byte-equivalence.
  let pt_back: Vec<half::f16> = pt_out.to_vec().expect("read pass-through as F16");
  for (i, (&got, &want)) in pt_back.iter().zip(pt_data.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "pass-through value at index {i} must be byte-identical"
    );
  }
}

/// Resident-size proxy — a large-ish pass-through tensor next to
/// a single AWQ layer triggering F32 escalation (via a mixed F16+BF16
/// pair). The pass-through `Array::size()` × `dtype_size()` must be
/// IDENTICAL pre- vs post-transform (same shape, same dtype → identical
/// resident bytes). An unscoped cast would widen the pass-through from
/// BF16 → F32, doubling its resident size.
#[test]
fn transform_awq_weights_preserves_resident_size_for_passthrough() {
  fn dtype_size(d: Dtype) -> usize {
    match d {
      Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
      Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
      Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
      Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
    }
  }
  let in_features = 8usize;
  let out_features = 8usize;
  let n_groups = 2usize;

  // Mixed F16+BF16 AWQ pair → resolver escalates to F32.
  let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
    .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
    .collect();
  let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
    .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
    .collect();
  let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
  let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
  let sc_b = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

  // Large-ish BF16 pass-through. Record pre-transform resident size.
  let pt_shape = (256usize, 256usize);
  let pt_data: Vec<half::bf16> = (0..pt_shape.0 * pt_shape.1)
    .map(|i| half::bf16::from_f32((i as f32) * 1e-4))
    .collect();
  let pt = Array::from_slice::<half::bf16>(&pt_data, &pt_shape).unwrap();
  let pt_size_pre = pt.size() * dtype_size(pt.dtype().unwrap());
  assert_eq!(
    pt_size_pre,
    pt_shape.0 * pt_shape.1 * 2,
    "pre-transform BF16 pass-through resident size sanity"
  );

  let weights: Weights = HashMap::from([
    ("layer_a.qweight".to_string(), qw_a),
    ("layer_a.scales".to_string(), sc_a),
    ("layer_b.qweight".to_string(), qw_b),
    ("layer_b.scales".to_string(), sc_b),
    ("embed_tokens.weight".to_string(), pt),
  ]);
  let cfg = AwqLoadConfig {
    bits: 4,
    group_size: 4,
    zero_point: false,
    version: "gemm".into(),
  };

  let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

  let pt_out = out
    .get("embed_tokens.weight")
    .expect("pass-through preserved");
  // Same dtype + same shape ⇒ same resident size. (BF16 = 2 bytes;
  // had it been cast to F32 the size would have doubled to 4 bytes/elem.)
  assert_eq!(
    pt_out.dtype().unwrap(),
    Dtype::BF16,
    "pass-through must remain BF16 (not widened to F32 by the mixed-half escalation)"
  );
  assert_eq!(
    pt_out.shape(),
    vec![pt_shape.0, pt_shape.1],
    "pass-through shape preserved"
  );
  let pt_size_post = pt_out.size() * dtype_size(pt_out.dtype().unwrap());
  assert_eq!(
    pt_size_post,
    pt_size_pre,
    "pass-through resident size must be IDENTICAL post-transform \
       (an unscoped cast would double it from {pt_size_pre} to {} bytes)",
    pt_size_pre * 2
  );
}

// ──────────────── AwqLoadConfig ────────────────

/// AwqLoadConfig round-trips through serde from a typical AutoAWQ
/// `quantization_config` JSON block.
#[test]
fn awq_load_config_parses_quantization_json() {
  let json = r#"{
      "bits": 4,
      "group_size": 128,
      "zero_point": true,
      "version": "gemm"
    }"#;
  let cfg: AwqLoadConfig = serde_json::from_str(json).expect("parse");
  assert_eq!(cfg.bits, 4);
  assert_eq!(cfg.group_size, 128);
  assert!(cfg.zero_point);
  assert_eq!(cfg.version, "gemm");
}

/// Defaults populate when keys are absent (AutoAWQ omitted-field convention).
#[test]
fn awq_load_config_defaults_when_keys_absent() {
  let cfg: AwqLoadConfig = serde_json::from_str("{}").expect("parse");
  assert_eq!(cfg.bits, 4);
  assert_eq!(cfg.group_size, 128);
  assert!(cfg.zero_point);
  assert_eq!(cfg.version, "");
}

/// Default impl matches the JSON-deserialized defaults (audit cross-check).
#[test]
fn awq_load_config_default_matches_serde_default() {
  let from_default = AwqLoadConfig::default();
  let from_serde: AwqLoadConfig = serde_json::from_str("{}").unwrap();
  assert_eq!(from_default, from_serde);
}

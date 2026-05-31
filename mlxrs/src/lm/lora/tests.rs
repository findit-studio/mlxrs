use super::*;
use crate::Dtype;

// ───────────────────── hand-traced fixtures ─────────────────────

/// Base weight `W` of shape [output_dims=2, input_dims=3]:
/// ```text
/// [[1, 0, 0],
///  [0, 1, 0]]
/// ```
/// so `x @ Wᵀ` projects `x=[x0,x1,x2]` to `[x0, x1]`.
fn base_weight() -> Array {
  Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap()
}

/// `lora_a` of shape [input_dims=3, r=2]:
/// ```text
/// [[1, 0],
///  [0, 1],
///  [0, 0]]
/// ```
fn lora_a() -> Array {
  Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap()
}

/// `lora_b` of shape [r=2, output_dims=2]:
/// ```text
/// [[1, 0],
///  [0, 1]]
/// ```
fn lora_b() -> Array {
  Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap()
}

fn plain_params() -> AdapterParams {
  AdapterParams {
    lora_a: lora_a(),
    lora_b: lora_b(),
    magnitude: None,
  }
}

/// Build an mlx-lm-native [`LoraConfig`] (LoRA, the given `num_layers`
/// trailing-block window and `lora_parameters`).
fn mlxlm_config(num_layers: i32, lora_parameters: LoraParameters) -> LoraConfig {
  LoraConfig {
    fine_tune_type: FineTuneType::Lora,
    lora_parameters,
    use_dora: false,
    selection: AdapterSelection::MlxLm { num_layers },
  }
}

/// The mlx-lm trailing-block window count of a config — asserts the config
/// is mlx-lm-native (NOT PEFT, which has no `num_layers`). Test-only.
fn mlxlm_num_layers(cfg: &LoraConfig) -> i32 {
  match &cfg.selection {
    AdapterSelection::MlxLm { num_layers } => *num_layers,
    AdapterSelection::Peft(_) => panic!("expected an mlx-lm-native config, got PEFT"),
  }
}

/// The `keys`-allowlisted rank-2 `LoraParameters` the layer-selection tests
/// reuse (`scale = 2.0`, the given `keys`; empty = auto-discovery).
fn keyed_params(keys: Vec<String>) -> LoraParameters {
  LoraParameters {
    rank: 2,
    scale: Some(2.0),
    alpha: None,
    keys,
    dropout: None,
  }
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
  assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
  for (x, y) in a.iter().zip(b.iter()) {
    assert!((x - y).abs() <= tol, "‖{x} - {y}‖ > {tol} ({a:?} vs {b:?})");
  }
}

// ───────────────────── LoRALinear forward ─────────────────────

#[test]
fn lora_linear_forward_hand_traced() {
  // x = [1, 2, 3]; scale = 2.0.
  // base(x)  = x @ Wᵀ = [1, 2]
  // x @ a    = [1, 2]  (a picks first two coords)
  // (x@a)@b  = [1, 2]
  // out      = base + scale*z = [1 + 2*1, 2 + 2*2] = [3, 6]
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.forward(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
}

#[test]
fn lora_linear_forward_with_bias() {
  // bias = [10, 20]; out = [3, 6] + [10, 20] = [13, 26].
  let bias = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap();
  let base = BaseLinear::dense(base_weight(), Some(bias)).unwrap();
  let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.forward(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[13.0, 26.0], 1e-5);
}

#[test]
fn lora_linear_zero_b_is_identity() {
  // lora_b all zeros ⇒ the low-rank term vanishes ⇒ out == base(x).
  // (This is the just-loaded-before-training state; an inference adapter has
  // a trained, non-zero lora_b, but the math must reduce correctly.)
  let zero_b = Array::zeros::<f32>(&(2, 2)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: zero_b,
    magnitude: None,
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = LoRALinear::new(base, params, 20.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.forward(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[1.0, 2.0], 1e-5);
}

// ───────────────────── fuse == forward ─────────────────────

#[test]
fn lora_fuse_matches_forward() {
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();

  let mut via_forward = layer.forward(&x).unwrap();
  // Fuse, then run the fused base's plain forward — must match.
  let fused = layer.fuse(false).unwrap();
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    1e-5,
  );
}

#[test]
fn lora_fuse_with_bias_matches_forward() {
  let bias = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap();
  let base = BaseLinear::dense(base_weight(), Some(bias)).unwrap();
  let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut via_forward = layer.forward(&x).unwrap();
  let fused = layer.fuse(false).unwrap();
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    1e-5,
  );
}

// ───────────────────── DoRA forward ─────────────────────

#[test]
fn dora_linear_forward_hand_traced() {
  // DoRA with m chosen to equal ‖adapted‖₂ so the renorm is the identity,
  // making the expected output the same [3, 6] as the LoRA case — this
  // isolates the renorm wiring (m/denom == 1 row-wise).
  //
  // adapted = W + scale*(lora_bᵀ @ lora_aᵀ); with scale=2,
  //   lora_bᵀ = [[1,0],[0,1]], lora_aᵀ = [[1,0,0],[0,1,0]]
  //   lora_bᵀ @ lora_aᵀ = [[1,0,0],[0,1,0]]
  //   adapted = [[1,0,0],[0,1,0]] + 2*[[1,0,0],[0,1,0]] = [[3,0,0],[0,3,0]]
  //   ‖adapted‖₂ row-wise = [3, 3]
  // Set m = [3, 3] ⇒ m/denom = [1, 1] ⇒ out == LoRA out == [3, 6].
  let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: lora_b(),
    magnitude: Some(m),
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = DoRALinear::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.forward(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
}

#[test]
fn dora_linear_forward_renorm_halves() {
  // Same adapted norm [3, 3], but m = [1.5, 1.5] ⇒ m/denom = [0.5, 0.5] ⇒
  // out = 0.5 * [3, 6] = [1.5, 3.0].
  let m = Array::from_slice::<f32>(&[1.5, 1.5], &(2usize,)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: lora_b(),
    magnitude: Some(m),
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = DoRALinear::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.forward(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[1.5, 3.0], 1e-5);
}

#[test]
fn dora_fuse_matches_forward() {
  let m = Array::from_slice::<f32>(&[1.5, 2.5], &(2usize,)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: lora_b(),
    magnitude: Some(m),
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = DoRALinear::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut via_forward = layer.forward(&x).unwrap();
  let fused = layer.fuse(false).unwrap();
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    1e-4,
  );
}

#[test]
fn dora_requires_magnitude() {
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let err = DoRALinear::new(base, plain_params(), 2.0).unwrap_err();
  assert!(matches!(err, Error::MissingField(_)));
}

// ───────────────────── QLoRA (quantized base) ─────────────────────

#[test]
fn qlora_forward_matches_dense_within_quant_error() {
  // Quantize a dense base, wrap with LoRA, and assert the QLoRA forward is
  // close to the dense LoRA forward (within affine-quant error). Use a
  // group_size that divides input_dims and a wide-ish weight so the quant
  // error stays small.
  //
  // input_dims must be divisible by group_size; use input_dims=64,
  // output_dims=2, group_size=32, bits=8 (low error).
  let input_dims = 64usize;
  let output_dims = 2usize;
  // Dense weight: row 0 = 1.0s, row 1 = 0.5s (well-represented at 8 bits).
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

  // lora_a [input_dims, r=2] small constant; lora_b [r=2, output_dims].
  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let params = AdapterParams {
    lora_a: la,
    lora_b: lb,
    magnitude: None,
  };

  let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

  // Dense LoRA forward.
  let dense_base = BaseLinear::dense(dense_w.try_clone().unwrap(), None).unwrap();
  let dense_layer = LoRALinear::new(dense_base, params.try_clone().unwrap(), 2.0).unwrap();
  let mut dense_out = dense_layer.forward(&x).unwrap();

  // Quantized base (affine, group_size=32, bits=8).
  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
  let q_layer = LoRALinear::new(q_base, params, 2.0).unwrap();
  let mut q_out = q_layer.forward(&x).unwrap();

  // Within affine-quant error (8-bit, uniform weights → small).
  approx_eq(
    &q_out.to_vec::<f32>().unwrap(),
    &dense_out.to_vec::<f32>().unwrap(),
    1e-2,
  );
}

#[test]
fn qlora_fuse_dequantize_matches_forward() {
  // fuse(dequantize=true) on a quantized base yields a dense fused linear
  // whose forward matches the QLoRA forward within quant error.
  let input_dims = 64usize;
  let output_dims = 2usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let params = AdapterParams {
    lora_a: la,
    lora_b: lb,
    magnitude: None,
  };
  let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
  let q_layer = LoRALinear::new(q_base, params, 2.0).unwrap();
  let mut via_forward = q_layer.forward(&x).unwrap();

  let fused = q_layer.fuse(true).unwrap();
  assert!(matches!(fused, BaseLinear::Dense { .. }));
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    1e-2,
  );
}

// ───────────────────── config parsing ─────────────────────

#[test]
fn config_parse_lora_basic() {
  let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 4,
      "lora_parameters": { "rank": 16, "scale": 20.0 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
  assert_eq!(mlxlm_num_layers(&cfg), 4);
  assert_eq!(cfg.rank(), 16);
  assert_eq!(cfg.scale(), 20.0);
  assert!(!cfg.is_dora());
}

#[test]
fn config_parse_peft_flat_shape() {
  // A REAL PEFT adapter_config.json: NO `lora_parameters` nesting — `r`,
  // `lora_alpha`, `target_modules`, `lora_dropout`, `peft_type` all flat at
  // the top level. The dual-shape `Deserialize` must detect the PEFT shape
  // and map the flat fields, so a PEFT-trained adapter does NOT silently
  // fall back to the default rank/scale.
  let json = r#"{
      "peft_type": "LORA",
      "r": 16,
      "lora_alpha": 32.0,
      "target_modules": ["q_proj", "v_proj"],
      "lora_dropout": 0.05,
      "bias": "none"
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.rank(), 16, "PEFT top-level `r` must populate `rank`");
  // alpha/rank = 32/16 = 2.0 — `lora_alpha`/`r` resolves the scale.
  assert_eq!(cfg.scale(), 2.0);
  // `target_modules` lands in the PEFT selection (NOT `keys` — PEFT
  // selection is the richer `PeftSelection`).
  assert!(cfg.lora_parameters.keys_slice().is_empty());
  let peft = cfg.peft().expect("PEFT config must carry a PeftSelection");
  match &peft.target_modules {
    Some(ModuleMatcher::List(names)) => {
      assert_eq!(names, &["q_proj".to_string(), "v_proj".to_string()]);
    }
    other => panic!("expected a target_modules List, got {other:?}"),
  }
  // `lora_dropout` maps to `dropout` (carried, ignored at inference).
  assert_eq!(cfg.lora_parameters.dropout, Some(0.05));
  assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
  // A PEFT config must NOT inherit mlx-lm's `num_layers` window
  // — PEFT adapts EVERY matching block. The selection is `Peft`, never
  // `MlxLm { num_layers }`.
  assert!(
    matches!(cfg.selection, AdapterSelection::Peft(_)),
    "a PEFT config must select via PeftSelection, never the mlx-lm num_layers window"
  );
  assert!(!cfg.is_dora());
}

#[test]
fn config_parse_peft_use_dora() {
  // A PEFT config with `use_dora: true` selects DoRA (PEFT carries the DoRA
  // signal in `use_dora`, not a `fine_tune_type`).
  let json = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "use_dora": true
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert!(cfg.is_dora(), "PEFT `use_dora` must select DoRA");
  assert_eq!(cfg.scale(), 2.0);
}

#[test]
fn config_parse_peft_no_peft_type_still_detected() {
  // A flat config with top-level `r` / `lora_alpha` / `target_modules` but
  // NO `peft_type` is still recognized as the PEFT shape (some exporters
  // omit `peft_type`) — the absence of a `lora_parameters` nesting plus the
  // flat PEFT keys is the signal.
  let json = r#"{
      "r": 4,
      "lora_alpha": 8.0,
      "target_modules": ["o_proj"]
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.rank(), 4);
  assert_eq!(cfg.scale(), 2.0);
  let peft = cfg
    .peft()
    .expect("PEFT shape detected ⇒ PeftSelection present");
  assert!(matches!(
    &peft.target_modules,
    Some(ModuleMatcher::List(n)) if n == &["o_proj".to_string()]
  ));
}

#[test]
fn config_parse_peft_default_rank_when_r_absent() {
  // PEFT `r` defaults to 8 in `peft` itself — a PEFT config without `r` but
  // with another PEFT marker must still parse and use DEFAULT_LORA_RANK.
  let json = r#"{ "peft_type": "LORA", "lora_alpha": 16.0, "target_modules": ["q_proj"] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.rank(), DEFAULT_LORA_RANK);
}

#[test]
fn config_parse_peft_non_lora_peft_type_is_err() {
  // A non-LoRA PEFT method (LOHA / LOKR / IA3 / prompt-tuning / …) is a
  // different adapter kind — this loader handles LoRA/DoRA only, so a
  // `peft_type` other than "LORA" is a recoverable parse error.
  for kind in ["LOHA", "LOKR", "IA3", "PROMPT_TUNING"] {
    let json = format!(r#"{{ "peft_type": "{kind}", "r": 8, "target_modules": ["q_proj"] }}"#);
    assert!(
      LoraConfig::from_json(&json).is_err(),
      "peft_type {kind:?} must be rejected"
    );
  }
}

#[test]
fn config_parse_peft_type_case_insensitive() {
  // PEFT writes `peft_type` upper-case ("LORA"); accept any case.
  let json = r#"{ "peft_type": "Lora", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.rank(), 8);
}

#[test]
fn config_parse_peft_target_modules_regex() {
  // PEFT `target_modules` may be a single regex string — modeled faithfully
  // via the `regex` crate (`re.fullmatch` semantics), NOT rejected.
  let json = r#"{ "peft_type": "LORA", "r": 8, "target_modules": ".*\\.(q|v)_proj" }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let peft = cfg.peft().unwrap();
  let target = match &peft.target_modules {
    Some(ModuleMatcher::Regex(re)) => re,
    other => panic!("expected a target_modules Regex, got {other:?}"),
  };
  // `re.fullmatch` — the whole module key must match.
  assert!(target.is_match("model.layers.0.self_attn.q_proj"));
  assert!(target.is_match("model.layers.7.self_attn.v_proj"));
  assert!(!target.is_match("model.layers.0.self_attn.k_proj"));
}

#[test]
fn config_parse_peft_invalid_regex_target_modules_is_err() {
  // A `target_modules` regex string that fails to compile is a recoverable
  // parse error (a malformed regex must not silently match nothing).
  let json = r#"{ "peft_type": "LORA", "r": 8, "target_modules": "(unclosed" }"#;
  assert!(
    LoraConfig::from_json(json).is_err(),
    "an uncompilable `target_modules` regex must be rejected"
  );
}

#[test]
fn config_lora_parameters_nesting_wins_over_flat_keys() {
  // A `lora_parameters` object is the unambiguous mlx-lm-native marker: when
  // it is present the flat PEFT keys are NOT consulted (a real config never
  // mixes the two shapes — this just pins the detection precedence).
  let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 3,
      "lora_parameters": { "rank": 64, "scale": 8.0 },
      "r": 1, "lora_alpha": 999.0
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.rank(), 64, "nested `lora_parameters.rank` wins");
  assert_eq!(
    cfg.scale(),
    8.0,
    "nested literal `scale` wins, flat keys ignored"
  );
  assert_eq!(mlxlm_num_layers(&cfg), 3);
}

#[test]
fn config_parse_dora_and_alpha_scale() {
  // mlx-lm-native nested shape — its alpha key is `alpha`. alpha/rank scale:
  // alpha=32, rank=8 ⇒ scale=4.0. fine_tune_type dora.
  let json = r#"{
      "fine_tune_type": "dora",
      "num_layers": 2,
      "lora_parameters": { "rank": 8, "alpha": 32.0 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert!(cfg.is_dora());
  assert_eq!(cfg.scale(), 4.0);
}

#[test]
fn config_use_dora_flag() {
  let json = r#"{
      "fine_tune_type": "lora",
      "use_dora": true,
      "lora_parameters": { "rank": 8, "scale": 10.0 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert!(cfg.is_dora());
}

#[test]
fn config_defaults_and_unknown_keys_ignored() {
  // Minimal config + extra training-only keys → parses, defaults applied.
  let json = r#"{ "optimizer": "adam", "learning_rate": 1e-4 }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
  assert_eq!(mlxlm_num_layers(&cfg), DEFAULT_NUM_LAYERS);
  assert_eq!(cfg.rank(), DEFAULT_LORA_RANK);
  assert_eq!(cfg.scale(), DEFAULT_LORA_SCALE);
}

#[test]
fn config_unknown_fine_tune_type_is_err() {
  let json = r#"{ "fine_tune_type": "bogus" }"#;
  assert!(LoraConfig::from_json(json).is_err());
}

// ───────────────────── path/key helpers ─────────────────────

#[test]
fn path_key_matching() {
  assert!(path_matches_key(
    "model.layers.27.self_attn.q_proj",
    "self_attn.q_proj"
  ));
  assert!(path_matches_key("self_attn.q_proj", "self_attn.q_proj"));
  assert!(!path_matches_key(
    "model.layers.27.self_attn.k_proj",
    "q_proj"
  ));
  // Must match on a segment boundary, not a substring.
  assert!(!path_matches_key("model.xq_proj", "q_proj"));
}

#[test]
fn block_index_parsing() {
  assert_eq!(
    parse_block_index("model.layers.27.self_attn.q_proj"),
    Some(27)
  );
  assert_eq!(parse_block_index("model.layers.0.mlp.down_proj"), Some(0));
  assert_eq!(parse_block_index("model.embed_tokens"), None);
  assert_eq!(parse_block_index("lm_head"), None);
}

// ───────────────────── linear_to_lora_layers ─────────────────────

/// Build a tiny weight map with 4 decoder blocks, each carrying a single
/// `self_attn.q_proj.weight` (and one block also a `k_proj`), plus a
/// top-level `lm_head.weight`.
fn toy_weights() -> Weights {
  let mut w = Weights::new();
  for b in 0..4 {
    w.insert(
      format!("model.layers.{b}.self_attn.q_proj.weight"),
      base_weight(),
    );
  }
  w.insert(
    "model.layers.0.self_attn.k_proj.weight".to_string(),
    base_weight(),
  );
  w.insert("lm_head.weight".to_string(), base_weight());
  w
}

/// Adapter params for every q_proj path in the toy map (4 blocks).
fn toy_adapter_params() -> HashMap<String, AdapterParams> {
  toy_adapter_params_for(&[0, 1, 2, 3])
}

/// Adapter params for the q_proj paths of the given block indices only.
/// Used to keep an adapter's factor set aligned with the `num_layers` window
/// under test — the completeness postcondition rejects factors for a path
/// outside the selection, so a windowed test must supply only in-window
/// factors.
fn toy_adapter_params_for(blocks: &[i32]) -> HashMap<String, AdapterParams> {
  let mut m = HashMap::new();
  for &b in blocks {
    m.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  m
}

#[test]
fn lora_layers_keys_and_num_layers_window() {
  // keys=["self_attn.q_proj"], num_layers=2 ⇒ only blocks 2,3's q_proj wrap.
  // The adapter supplies factors for exactly those two blocks (an adapter
  // that also carried block-0/1 factors would now be a config mismatch — see
  // `lora_layers_extra_factors_outside_window_is_err`).
  let weights = toy_weights();
  let params = toy_adapter_params_for(&[2, 3]);
  let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  // Only blocks 2 and 3 are inside the trailing-2 window.
  assert!(layers.contains_key("model.layers.2.self_attn.q_proj"));
  assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.0.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.1.self_attn.q_proj"));
  // k_proj never matches the key.
  assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
  // lm_head is a non-block path and not in keys → untouched.
  assert!(!layers.contains_key("lm_head"));
  assert_eq!(layers.len(), 2);
}

#[test]
fn lora_layers_covers_all_blocks_when_num_layers_large() {
  let weights = toy_weights();
  let params = toy_adapter_params();
  // num_layers 16 > 4 blocks ⇒ all q_proj blocks wrap.
  let cfg = mlxlm_config(16, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
}

// ───────────────────── load_adapters end-to-end ─────────────────────

/// Write a mock adapter dir: adapter_config.json + adapters.safetensors with
/// factors for two q_proj paths.
fn write_mock_adapter(dir: &Path, fine_tune_type: &str, with_m: bool) {
  let config = format!(
    r#"{{
        "fine_tune_type": "{fine_tune_type}",
        "num_layers": 16,
        "lora_parameters": {{ "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }}
      }}"#
  );
  std::fs::write(dir.join("adapter_config.json"), config).unwrap();

  let mut arrays: HashMap<String, Array> = HashMap::new();
  for b in 0..4 {
    let path = format!("model.layers.{b}.self_attn.q_proj");
    arrays.insert(format!("{path}.lora_a"), lora_a());
    arrays.insert(format!("{path}.lora_b"), lora_b());
    if with_m {
      // m = ‖adapted‖₂ (so renorm is identity) → [3, 3] for these factors.
      arrays.insert(
        format!("{path}.m"),
        Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap(),
      );
    }
  }
  crate::io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
}

#[test]
fn load_adapters_lora_end_to_end() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_lora_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  write_mock_adapter(&tmp, "lora", false);

  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  // 4 q_proj blocks adapted.
  assert_eq!(layers.len(), 4);
  assert!(matches!(
    layers.get("model.layers.0.self_attn.q_proj"),
    Some(LoraLayer::Lora(_))
  ));

  // Forward through an adapted layer matches the hand-traced LoRA result.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layers
    .get("model.layers.0.self_attn.q_proj")
    .unwrap()
    .forward(&x)
    .unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);

  std::fs::remove_dir_all(&tmp).ok();
}

/// Write a mock adapter dir whose `adapter_config.json` is `config_json`
/// (caller-supplied, so a test can vary `rank`/`r`/`alpha`) and whose
/// `adapters.safetensors` carries rank-`r` factors for the 4 q_proj paths
/// over the toy `[2, 3]` base: `lora_a` is `[3, r]`, `lora_b` is `[r, 2]`.
fn write_mock_adapter_rank(dir: &Path, config_json: &str, r: usize) {
  std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();
  let la = Array::full::<f32>(&(3usize, r), 0.01).unwrap();
  let lb = Array::full::<f32>(&(r, 2usize), 0.01).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  for b in 0..4 {
    let path = format!("model.layers.{b}.self_attn.q_proj");
    arrays.insert(format!("{path}.lora_a"), la.try_clone().unwrap());
    arrays.insert(format!("{path}.lora_b"), lb.try_clone().unwrap());
  }
  crate::io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
}

/// Write a mock **PEFT** adapter dir: a caller-supplied
/// `adapter_config.json` plus a PEFT-keyed `adapter_model.safetensors`.
/// `paths` are the base-module paths (without `.weight`) to ship factors
/// for; for each, the PEFT tensors `base_model.model.<path>.lora_A.weight`
/// (`[r, in=3]`) and `.lora_B.weight` (`[out=2, r]`) are written — the PEFT
/// orientation (transposed vs the mlxrs scheme). When `with_dora`, a
/// `.lora_magnitude_vector` (`[out=2]`) is added per path. The PEFT factor
/// values are `value` (so the post-translation `lora_a`/`lora_b` are
/// constant — handy for hand-traced math).
fn write_mock_peft_adapter(
  dir: &Path,
  config_json: &str,
  paths: &[&str],
  r: usize,
  with_dora: bool,
  value: f32,
) {
  std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();
  // PEFT `lora_A.weight` is `[r, in_features]`; `lora_B.weight` is
  // `[out_features, r]` (the transpose of the mlxrs `lora_a` / `lora_b`).
  let lora_a_peft = Array::full::<f32>(&(r, 3usize), value).unwrap();
  let lora_b_peft = Array::full::<f32>(&(2usize, r), value).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  for path in paths {
    arrays.insert(
      format!("base_model.model.{path}.lora_A.weight"),
      lora_a_peft.try_clone().unwrap(),
    );
    arrays.insert(
      format!("base_model.model.{path}.lora_B.weight"),
      lora_b_peft.try_clone().unwrap(),
    );
    if with_dora {
      // DoRA magnitude — [out_features=2], no transpose in either scheme.
      arrays.insert(
        format!("base_model.model.{path}.lora_magnitude_vector"),
        Array::from_slice::<f32>(&[1.0, 1.0], &(2usize,)).unwrap(),
      );
    }
  }
  crate::io::save_safetensors(&dir.join("adapter_model.safetensors"), &arrays).unwrap();
}

#[test]
fn load_adapters_peft_flat_shape_rank16_end_to_end() {
  // A REAL PEFT-shaped adapter_config.json — flat top-level `peft_type` /
  // `r` / `lora_alpha` / `target_modules` / `lora_dropout`, NO
  // `lora_parameters` nesting — plus rank-16 factor tensors. The dual-shape
  // `Deserialize` must read `r:16` (so the rank-16 factors pass the
  // config-rank cross-check), resolve the scale to `lora_alpha/r = 32/16 =
  // 2.0`, and use `target_modules` to drive layer selection.
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_flat16_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA",
      "r": 16,
      "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"],
      "lora_dropout": 0.0,
      "bias": "none"
    }"#;
  let q_paths: Vec<String> = (0..4)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 16, false, 0.01);
  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  // `target_modules: ["self_attn.q_proj"]` selects the 4 q_proj paths (and
  // NOT the lone k_proj) — i.e. PEFT's `target_modules` drove the selection.
  assert_eq!(layers.len(), 4);
  assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
  // The resolved scale is lora_alpha/r = 32/16 = 2.0.
  if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
    assert_eq!(l.scale(), 2.0, "PEFT scale must be lora_alpha/r");
  } else {
    panic!("expected a LoRA layer");
  }
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_peft_flat_shape_rank8_scale_not_default() {
  // A PEFT config with `r:8` + rank-8 factors must load and resolve the
  // scale to `lora_alpha/8`, NOT the literal-`scale` default of 20.0 (PEFT
  // has no literal-`scale` key — the prior nested-only alias would have left
  // a real flat PEFT config defaulting both rank and scale).
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_flat8_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"]
    }"#;
  let q_paths: Vec<String> = (0..4)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 8, false, 0.01);
  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
    // lora_alpha/r = 32/8 = 4.0 — explicitly NOT DEFAULT_LORA_SCALE (20.0).
    assert_eq!(l.scale(), 4.0);
    assert_ne!(l.scale(), DEFAULT_LORA_SCALE);
  } else {
    panic!("expected a LoRA layer");
  }
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_rank_drift_is_shape_mismatch() {
  // mlx-lm-native config declares rank 8 with `alpha` present, but the
  // factor tensors are rank 16 (a stale `adapter_config.json` drift).
  // Without the config-vs-tensor rank cross-check this silently builds
  // rank-16 factors and scales by alpha/8 instead of alpha/16 — wrong
  // strength. It must fail loudly at load with a LengthMismatch.
  let tmp = std::env::temp_dir().join(format!("mlxrs_rankdrift_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 8, "alpha": 32.0, "keys": ["self_attn.q_proj"] }
    }"#;
  write_mock_adapter_rank(&tmp, cfg, 16);
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::LengthMismatch(_)),
    "rank drift must be a LengthMismatch, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_peft_rank_drift_is_shape_mismatch() {
  // The rank-drift guard must also catch a PEFT-flat config: `r:8`
  // declared but rank-16 factors shipped. The dual-shape `Deserialize`
  // reads `r` correctly, then `validate_config_rank` rejects the drift.
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_rankdrift_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"]
    }"#;
  // `r:8` declared, but rank-16 factors shipped.
  let q_paths: Vec<String> = (0..4)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 16, false, 0.01);
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::LengthMismatch(_)),
    "PEFT rank drift must be a LengthMismatch, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_dora_end_to_end() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_dora_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  write_mock_adapter(&tmp, "dora", true);

  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  assert!(matches!(
    layers.get("model.layers.0.self_attn.q_proj"),
    Some(LoraLayer::Dora(_))
  ));
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_dora_missing_magnitude_is_err() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_dora_nom_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  // fine_tune_type dora but no `.m` arrays → recoverable Err.
  write_mock_adapter(&tmp, "dora", false);
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::MissingKey(_)),
    "missing DoRA magnitude must be MissingKey, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_full_is_unsupported_err() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_full_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  write_mock_adapter(&tmp, "full", false);
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "fine_tune_type=full rejection must be UnknownEnumValue, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_unknown_fine_tune_type_is_err() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_bogus_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  write_mock_adapter(&tmp, "bogus", false);
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "unknown fine_tune_type must be a serde Parse error, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_missing_config_is_err() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_nocfg_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  // Only write the safetensors, no config.
  let arrays: HashMap<String, Array> = HashMap::new();
  crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::FileIo(_)),
    "missing adapter_config.json must be a FileIo error, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_missing_dir_is_err() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_nodir_test_{}", std::process::id()));
  // Do NOT create the dir.
  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::FileIo(_)),
    "missing adapter dir must be a FileIo error, got {err:?}"
  );
}

// ───────────────────── factor-shape validation ─────────────────────

#[test]
fn lora_rejects_mismatched_output_dims() {
  // lora_b last axis (3) != base output_dims (2).
  let bad_b = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: bad_b,
    magnitude: None,
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let err = LoRALinear::new(base, params, 2.0).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)));
}

#[test]
fn lora_rejects_rank_mismatch() {
  // lora_a [3, 2] but lora_b [3, 2] (leading 3 != a's r=2).
  let bad_b = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: bad_b,
    magnitude: None,
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let err = LoRALinear::new(base, params, 2.0).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)));
}

// ───────── lora_a input-dim cross-check ─────────

#[test]
fn lora_rejects_wrong_lora_a_input_dim_dense() {
  // Dense base W is [output_dims=2, input_dims=3]; a lora_a with leading axis
  // 2 (≠ input_dims 3) must be rejected at construction, not deferred to a
  // mlx-c matmul failure on the first forward.
  let bad_a = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let params = AdapterParams {
    lora_a: bad_a,
    lora_b: lora_b(),
    magnitude: None,
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let err = LoRALinear::new(base, params, 2.0).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)));
}

#[test]
fn lora_rejects_wrong_lora_a_input_dim_quantized() {
  // Quantized base: dense [2, 64] affine-quantized at 8 bits ⇒ packed [2, 16];
  // base_input_dims recovers 16 * 32 / 8 = 64. A lora_a with leading axis 32
  // (≠ 64) must be rejected at construction.
  let input_dims = 64usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(2, input_dims)).unwrap();
  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();

  // input_dims should be 64 — supply a wrong-width lora_a [32, 2].
  let bad_a = Array::full::<f32>(&(32usize, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let params = AdapterParams {
    lora_a: bad_a,
    lora_b: lb,
    magnitude: None,
  };
  let err = LoRALinear::new(q_base, params, 2.0).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)));
}

#[test]
fn lora_a_correct_input_dim_quantized_ok() {
  // The positive companion: a correctly-sized lora_a [64, 2] over the same
  // quantized base constructs cleanly (base_input_dims == 64 == lora_a[0]).
  let input_dims = 64usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(2, input_dims)).unwrap();
  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let params = AdapterParams {
    lora_a: la,
    lora_b: lb,
    magnitude: None,
  };
  assert!(LoRALinear::new(q_base, params, 2.0).is_ok());
}

// ───────── scale precedence (alpha wins) ─────────

#[test]
fn resolved_scale_alpha_only() {
  // alpha present, no scale ⇒ alpha / rank.
  let p = LoraParameters {
    rank: 8,
    scale: None,
    alpha: Some(32.0),
    keys: Vec::new(),
    dropout: None,
  };
  assert_eq!(p.resolved_scale(), 4.0);
}

#[test]
fn resolved_scale_scale_only() {
  // scale present, no alpha ⇒ the literal scale.
  let p = LoraParameters {
    rank: 8,
    scale: Some(7.5),
    alpha: None,
    keys: Vec::new(),
    dropout: None,
  };
  assert_eq!(p.resolved_scale(), 7.5);
}

#[test]
fn resolved_scale_alpha_wins_over_scale() {
  // BOTH present ⇒ alpha / rank WINS over the literal scale (PEFT precedence).
  // alpha=64, rank=16 ⇒ 4.0, NOT the literal 99.0.
  let p = LoraParameters {
    rank: 16,
    scale: Some(99.0),
    alpha: Some(64.0),
    keys: Vec::new(),
    dropout: None,
  };
  assert_eq!(p.resolved_scale(), 4.0);
}

#[test]
fn resolved_scale_neither_is_default() {
  // Neither present ⇒ DEFAULT_LORA_SCALE.
  let p = LoraParameters {
    rank: 8,
    scale: None,
    alpha: None,
    keys: Vec::new(),
    dropout: None,
  };
  assert_eq!(p.resolved_scale(), DEFAULT_LORA_SCALE);
}

#[test]
fn resolved_scale_alpha_with_nonpositive_rank_falls_back() {
  // Defensive floor: alpha present but rank <= 0 ⇒ `alpha / rank` is
  // undefined ⇒ fall through to the literal scale, then the default.
  let p = LoraParameters {
    rank: 0,
    scale: Some(5.0),
    alpha: Some(32.0),
    keys: Vec::new(),
    dropout: None,
  };
  assert_eq!(p.resolved_scale(), 5.0);
  let p_no_scale = LoraParameters {
    rank: -1,
    scale: None,
    alpha: Some(32.0),
    keys: Vec::new(),
    dropout: None,
  };
  assert_eq!(p_no_scale.resolved_scale(), DEFAULT_LORA_SCALE);
}

#[test]
fn config_both_scale_and_alpha_alpha_wins() {
  // mlx-lm-native config carrying BOTH a literal `scale` and `alpha` ⇒
  // alpha/rank wins over the literal scale.
  let json = r#"{
      "fine_tune_type": "lora",
      "lora_parameters": { "rank": 8, "scale": 50.0, "alpha": 16.0 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.scale(), 2.0); // 16 / 8, not the literal 50.0
}

// ───────── num_layers <= 0 selects ALL blocks ─────────

#[test]
fn lora_layers_num_layers_negative_one_selects_all_blocks() {
  // mlx-lm `model.layers[-max(-1,0):]` == `layers[-0:]` == `layers[0:]` ⇒
  // num_layers: -1 adapts EVERY decoder block, not none.
  let weights = toy_weights();
  let params = toy_adapter_params(); // factors for all 4 q_proj blocks
  let cfg = mlxlm_config(-1, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  assert_eq!(layers.len(), 4, "num_layers=-1 must adapt all 4 blocks");
  for b in 0..4 {
    assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
  }
}

#[test]
fn lora_layers_num_layers_zero_selects_all_blocks() {
  // num_layers: 0 ⇒ `max(0,0)=0` ⇒ `layers[-0:]` == all blocks too.
  let weights = toy_weights();
  let params = toy_adapter_params();
  let cfg = mlxlm_config(0, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  assert_eq!(layers.len(), 4, "num_layers=0 must adapt all 4 blocks");
}

// ───────── adapter-completeness postcondition ─────────

#[test]
fn lora_layers_explicit_key_missing_factors_is_err() {
  // keys=["self_attn.q_proj"], num_layers covers all 4 blocks, but the
  // adapter only supplies factors for blocks 0,1 ⇒ blocks 2,3 are selected
  // targets with no factors ⇒ typed `Error::MissingKey` (case a) keyed on
  // the FIRST (sorted) missing target.
  let weights = toy_weights();
  let params = toy_adapter_params_for(&[0, 1]);
  let cfg = mlxlm_config(16, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
  match err {
    Error::MissingKey(p) => {
      assert!(
        p.context().contains("explicitly-selected adapter target"),
        "context names the explicit-selection rule: {}",
        p.context()
      );
      assert_eq!(p.key(), "model.layers.2.self_attn.q_proj");
    }
    other => panic!("expected Error::MissingKey, got {other:?}"),
  }
}

#[test]
fn lora_layers_unused_adapter_factor_is_err() {
  // The adapter carries a factor group for a path that exists in NO base
  // weight (a path-prefix mismatch / config drift) ⇒ Err (case b): typed
  // `Error::LayerKeyed` keyed on the unused path wrapping a typed
  // `Error::InvariantViolation` calling out the "must match a base layer"
  // rule.
  let weights = toy_weights();
  let mut params = toy_adapter_params(); // all 4 q_proj blocks (all match)
  params.insert(
    "model.layers.99.self_attn.q_proj".to_string(),
    plain_params(),
  );
  let cfg = mlxlm_config(16, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
  match err {
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "model.layers.99.self_attn.q_proj");
      let Error::InvariantViolation(iv) = p.inner() else {
        panic!(
          "expected inner Error::InvariantViolation, got {:?}",
          p.inner()
        );
      };
      assert!(
        iv.context().contains("adapter factor group")
          && iv.requirement().contains("must match a base layer"),
        "inner violation should call out base-layer matching: {iv:?}"
      );
    }
    other => panic!("expected Error::LayerKeyed, got {other:?}"),
  }
}

#[test]
fn lora_layers_empty_result_is_err() {
  // keys names a projection that exists in NO base weight, and there are no
  // factors ⇒ nothing adapted ⇒ typed `Error::InvariantViolation` (case c).
  let weights = toy_weights();
  let params: HashMap<String, AdapterParams> = HashMap::new();
  let cfg = mlxlm_config(
    16,
    keyed_params(vec!["self_attn.nonexistent_proj".to_string()]),
  );
  let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
  match err {
    Error::InvariantViolation(p) => {
      assert_eq!(p.context(), "load_adapters: adapted-layer count");
      assert!(p.requirement().contains("must be >= 1"));
    }
    other => panic!("expected Error::InvariantViolation, got {other:?}"),
  }
}

#[test]
fn lora_layers_autodiscovery_partial_factors_is_ok() {
  // keys: None (auto-discovery) ⇒ a base linear without factors is EXPECTED
  // (the adapter trains only a subset); only the unused-factor (b) and
  // empty-result (c) checks apply. Factors for 2 of the 4 q_proj blocks ⇒ Ok.
  let weights = toy_weights();
  let params = toy_adapter_params_for(&[2, 3]);
  let cfg = mlxlm_config(16, keyed_params(Vec::new()));
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  assert_eq!(layers.len(), 2);
}

#[test]
fn load_adapters_unused_factor_end_to_end_is_err() {
  // End-to-end: an adapters.safetensors carrying a factor group for a path
  // absent from the base model ⇒ load_adapters rejects it.
  let tmp = std::env::temp_dir().join(format!("mlxrs_unused_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
  std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  for b in 0..4 {
    let path = format!("model.layers.{b}.self_attn.q_proj");
    arrays.insert(format!("{path}.lora_a"), lora_a());
    arrays.insert(format!("{path}.lora_b"), lora_b());
  }
  // A factor group for a path that is NOT in toy_weights().
  arrays.insert(
    "model.layers.42.self_attn.q_proj.lora_a".to_string(),
    lora_a(),
  );
  arrays.insert(
    "model.layers.42.self_attn.q_proj.lora_b".to_string(),
    lora_b(),
  );
  crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();

  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(ref p) if matches!(p.inner(), Error::InvariantViolation(_))),
    "unused factor group must be LayerKeyed(InvariantViolation), got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_empty_safetensors_is_err() {
  // An empty adapters.safetensors (no factor groups at all) ⇒ nothing adapted
  // ⇒ Err (case c), instead of a silently-unadapted Ok.
  let tmp = std::env::temp_dir().join(format!("mlxrs_emptyst_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
  std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
  let arrays: HashMap<String, Array> = HashMap::new();
  crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();

  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  assert!(
    matches!(err, Error::MissingKey(_)),
    "explicit-selection w/o factors must be MissingKey, got {err:?}"
  );
  std::fs::remove_dir_all(&tmp).ok();
}

// ───────── QDoRA forward via quantized_matmul ─────────

#[test]
fn qdora_forward_matches_dense_within_quant_error() {
  // QDoRA (DoRA over a quantized base) + bias: the forward must match the
  // dense DoRA forward within affine-quant error. By construction the
  // quantized base output runs through quantized_matmul (base_output_no_bias),
  // never a full dense-weight matmul — the dequantized weight is materialized
  // only for the adapted-weight L2-norm.
  let input_dims = 64usize;
  let output_dims = 2usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  // m = ‖adapted‖₂ row-wise of the DENSE adapted weight (so dense + quantized
  // share the same magnitude vector — the renorm is identical).
  let bias = Array::from_slice::<f32>(&[3.0, -1.0], &(output_dims,)).unwrap();

  let dense_params = AdapterParams {
    lora_a: la.try_clone().unwrap(),
    lora_b: lb.try_clone().unwrap(),
    magnitude: None,
  };
  // Build a DoRALinear over the dense base to read back its computed adapted
  // norm via fuse? Simpler: pick m = norm of (dense_w + scale*delta).
  let scale = 2.0f32;
  let delta = lora_delta(&dense_params, scale).unwrap();
  let adapted = dense_w.add(&delta).unwrap();
  let m = ops::linalg_full::norm(&adapted, 2.0, &[1], false).unwrap();

  let dense_base = BaseLinear::dense(
    dense_w.try_clone().unwrap(),
    Some(bias.try_clone().unwrap()),
  )
  .unwrap();
  let dense_layer = DoRALinear::new(
    dense_base,
    AdapterParams {
      lora_a: la.try_clone().unwrap(),
      lora_b: lb.try_clone().unwrap(),
      magnitude: Some(m.try_clone().unwrap()),
    },
    scale,
  )
  .unwrap();
  let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();
  let mut dense_out = dense_layer.forward(&x).unwrap();

  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base = BaseLinear::quantized(
    w_q,
    scales,
    biases,
    Some(bias.try_clone().unwrap()),
    32,
    8,
    "affine".to_string(),
  )
  .unwrap();
  let q_layer = DoRALinear::new(
    q_base,
    AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: Some(m),
    },
    scale,
  )
  .unwrap();
  let mut q_out = q_layer.forward(&x).unwrap();

  approx_eq(
    &q_out.to_vec::<f32>().unwrap(),
    &dense_out.to_vec::<f32>().unwrap(),
    2e-2,
  );
}

#[test]
fn qdora_forward_matches_fuse() {
  // QDoRA forward must equal its own fuse path within quant error — exercises
  // the quantized_matmul base output against the fused (renormalized) weight.
  let input_dims = 64usize;
  let output_dims = 2usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let m = Array::from_slice::<f32>(&[1.5, 2.5], &(output_dims,)).unwrap();
  let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
  let q_layer = DoRALinear::new(
    q_base,
    AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: Some(m),
    },
    2.0,
  )
  .unwrap();
  let mut via_forward = q_layer.forward(&x).unwrap();
  let fused = q_layer.fuse(true).unwrap();
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    2e-2,
  );
}

// ───────── adapters.safetensors hardening ─────────

#[test]
fn load_adapters_non_regular_safetensors_is_err() {
  // A directory planted where adapters.safetensors should be is not a regular
  // file ⇒ `adapter_candidate_present` classifies the probe outcome as
  // `CandidateProbe::NonRegular` and surfaces a typed `Error::FileIo` with
  // `ErrorKind::InvalidInput` from the `Stat` op. The structural fix makes
  // a non-regular candidate fail-fast (never falls through to the fallback),
  // so a directory at the preferred slot can NOT be silently masked by an
  // adjacent valid `adapter_model.safetensors` — misconfigurations of the
  // user's adapter directory surface immediately rather than being papered
  // over by reading a different file.
  let tmp = std::env::temp_dir().join(format!("mlxrs_nonreg_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
  std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
  // adapters.safetensors is a DIRECTORY, not a file.
  std::fs::create_dir_all(tmp.join("adapters.safetensors")).unwrap();

  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  match err {
    Error::FileIo(p) => {
      // Fail-fast: the error names the non-regular preferred path with
      // `FileOp::Stat` (the probe), NOT the fallback (which is absent).
      assert_eq!(p.path(), tmp.join("adapters.safetensors").as_path());
      assert_eq!(p.op(), FileOp::Stat);
      assert_eq!(p.inner().kind(), std::io::ErrorKind::InvalidInput);
    }
    other => panic!("expected Error::FileIo(InvalidInput, Stat), got {other:?}"),
  }
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn load_adapters_oversized_safetensors_is_err() {
  // A sparse file reporting a length beyond MAX_ADAPTER_SAFETENSORS_BYTES is
  // rejected on the stat, before any mmap. set_len makes a sparse file on
  // APFS/most filesystems — the on-disk footprint stays ~0.
  let tmp = std::env::temp_dir().join(format!("mlxrs_oversize_test_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
  std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
  let f = std::fs::File::create(tmp.join("adapters.safetensors")).unwrap();
  f.set_len(MAX_ADAPTER_SAFETENSORS_BYTES + 1).unwrap();
  drop(f);

  let weights = toy_weights();
  let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
  match err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap_name(), "MAX_ADAPTER_SAFETENSORS_BYTES");
      assert_eq!(p.cap(), MAX_ADAPTER_SAFETENSORS_BYTES);
      assert_eq!(p.observed(), MAX_ADAPTER_SAFETENSORS_BYTES + 1);
    }
    other => panic!("expected Error::CapExceeded, got {other:?}"),
  }
  std::fs::remove_dir_all(&tmp).ok();
}

// ═════════════════ HuggingFace PEFT — full surface ═════════════════

/// A weight map with `n` decoder blocks, each carrying a `self_attn.q_proj`
/// and a `self_attn.v_proj`, plus a top-level `lm_head.weight`.
fn peft_toy_weights(n: usize) -> Weights {
  let mut w = Weights::new();
  for b in 0..n {
    w.insert(
      format!("model.layers.{b}.self_attn.q_proj.weight"),
      base_weight(),
    );
    w.insert(
      format!("model.layers.{b}.self_attn.v_proj.weight"),
      base_weight(),
    );
  }
  w.insert("lm_head.weight".to_string(), base_weight());
  w
}

// ───────────── PEFT config: fields + defaults ─────────────

#[test]
fn peft_config_lora_alpha_defaults_to_8() {
  // PEFT `LoraConfig.lora_alpha` defaults to 8 (NOT the mlx-lm 20.0 literal).
  // A PEFT config omitting `lora_alpha` with `r:16` ⇒ scale 8/16 = 0.5.
  let json = r#"{ "peft_type": "LORA", "r": 16, "target_modules": ["q_proj"] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.lora_parameters.alpha, Some(DEFAULT_PEFT_LORA_ALPHA));
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 0.5);
}

#[test]
fn peft_config_accepts_and_ignores_training_only_fields() {
  // A real PEFT `LoraConfig` carries training-only / metadata fields with no
  // inference effect on already-saved factors — these BENIGN fields must parse
  // cleanly (accept-and-ignore), not error, even when set to real values.
  // (`layer_replication` / `trainable_token_indices` / `target_parameters` —
  // formerly in this list — are forward/structure-switching and are now
  // rejected by the reject-unknown-active backstop; see
  // `peft_config_structural_reject_examples_*`.)
  let json = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "init_lora_weights": "gaussian",
      "loftq_config": {},
      "eva_config": null,
      "corda_config": null,
      "task_type": "CAUSAL_LM",
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "revision": null,
      "base_model_name_or_path": "meta-llama/Llama-3-8B"
    }"#;
  let cfg = LoraConfig::from_json(json).expect("training-only fields must not error");
  assert_eq!(cfg.rank(), 8);
  assert_eq!(cfg.scale_for("q_proj"), 2.0);
}

#[test]
fn peft_config_lora_bias_true_is_err() {
  // `lora_bias: true` puts a bias on lora_B that PEFT adds in the forward —
  // mlxrs's LoRALinear has no such term, so a silent drop would be wrong
  // inference. It must be a recoverable parse error.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "lora_bias": true
    }"#;
  assert!(
    LoraConfig::from_json(json).is_err(),
    "`lora_bias: true` must be rejected (no lora_B-bias term in LoRALinear)"
  );
  // `lora_bias: false` (the default) is accepted.
  let ok = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "lora_bias": false }"#;
  assert!(LoraConfig::from_json(ok).is_ok());
}

#[test]
fn peft_config_bias_all_or_lora_only_is_err() {
  // PEFT `bias: "all"` / `"lora_only"` trains+saves `.bias` tensors that PEFT
  // adds in the forward (`utils/save_and_load.py` keeps `"bias" in k`);
  // mlxrs's LoRALinear has no adapted-bias slot, so a non-`"none"` value must
  // be a recoverable parse error (a silent drop would be wrong inference).
  for bias in ["all", "lora_only"] {
    let json = format!(
      r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "bias": {bias:?} }}"#
    );
    let err =
      LoraConfig::from_json(&json).expect_err(&format!("PEFT `bias: {bias:?}` must be rejected"));
    assert!(
      matches!(err, Error::Parse(_)),
      "expected Error::Parse for `bias: {bias:?}`, got {err:?}"
    );
  }
  // `bias: "none"` (the default) — and no `bias` key at all — are fine.
  let none = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "bias": "none" }"#;
  assert!(LoraConfig::from_json(none).is_ok());
  let absent = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"] }"#;
  assert!(LoraConfig::from_json(absent).is_ok());
}

#[test]
fn peft_config_nonempty_modules_to_save_is_err() {
  // PEFT `modules_to_save` trains+saves full modules alongside the low-rank
  // factors; mlxrs's low-rank loader has no saved-full-module slot, so a
  // non-empty list must be rejected (a silent drop of the full module weights
  // would be wrong inference).
  let json = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "modules_to_save": ["embed_tokens", "lm_head"] }"#;
  let err = LoraConfig::from_json(json).expect_err("non-empty `modules_to_save` must be rejected");
  assert!(
    matches!(err, Error::Parse(_)),
    "expected Error::Parse, got {err:?}"
  );
  // An empty `modules_to_save` (or absent) is fine — it ships no full modules.
  let empty = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "modules_to_save": [] }"#;
  assert!(LoraConfig::from_json(empty).is_ok());
}

#[test]
fn peft_key_translation_rejects_sidecar_bias_and_modules_to_save_tensors() {
  // A PEFT-prefixed tensor whose suffix is NOT a low-rank factor is a `.bias`
  // (PEFT `bias != "none"`) or a `modules_to_save` full-module weight — both
  // affect inference, so `translate_peft_keys` must REJECT (naming the key),
  // never silently drop. (Defense-in-depth at the weights file, mirroring the
  // config-level `bias` / `modules_to_save` rejection.)

  // (a) a `.bias` tensor adjacent to a LoRA path (PEFT `bias: "all"` /
  // `"lora_only"` saves `base_model.model.<path>.bias`).
  let bias_key = "base_model.model.model.layers.0.self_attn.q_proj.bias";
  let mut with_bias: HashMap<String, Array> = HashMap::new();
  with_bias.insert(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
    Array::zeros::<f32>(&(2, 3)).unwrap(),
  );
  with_bias.insert(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
    Array::zeros::<f32>(&(4, 2)).unwrap(),
  );
  with_bias.insert(
    bias_key.to_string(),
    Array::zeros::<f32>(&(4usize,)).unwrap(),
  );
  let err = translate_peft_keys(with_bias)
    .expect_err("a PEFT-prefixed `.bias` tensor must be rejected, not silently dropped");
  match err {
    Error::LayerKeyed(ref payload) => {
      assert_eq!(
        payload.layer(),
        bias_key,
        "the rejection must name the dropped key"
      );
      assert!(matches!(payload.inner(), Error::InvariantViolation(_)));
    }
    other => panic!("expected Error::LayerKeyed, got {other:?}"),
  }

  // (b) a `modules_to_save` full-module weight (the `modules_to_save.<adapter>.`
  // prefix is stripped on save → `base_model.model.<module>.weight`).
  let saved_key = "base_model.model.lm_head.weight";
  let mut with_saved: HashMap<String, Array> = HashMap::new();
  with_saved.insert(saved_key.to_string(), base_weight());
  let err = translate_peft_keys(with_saved)
    .expect_err("a PEFT-prefixed `modules_to_save` weight must be rejected");
  let Error::LayerKeyed(payload) = err else {
    panic!("expected LayerKeyed");
  };
  assert_eq!(payload.layer(), saved_key);
  assert!(matches!(payload.inner(), Error::InvariantViolation(_)));
}

#[test]
fn peft_config_exotic_variants_are_rejected() {
  // PEFT's exotic LoRA variants each CHANGE the inference forward — loading
  // such an adapter as plain LoRA would run it at the wrong behavior, so the
  // `Deserialize` must REJECT them loudly (not silently drop, as it does the
  // training-only fields). One non-default exotic field per config, on a
  // normal PEFT-flat config.
  let base = r#""peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"]"#;
  for (field, value) in [
    ("use_qalora", "true"),
    ("alora_invocation_tokens", "[1, 2, 3]"),
    ("velora_config", r#"{"rank": 4}"#),
    ("monteclora_config", r#"{"num_samples": 8}"#),
  ] {
    let json = format!("{{ {base}, {field:?}: {value} }}");
    let err = LoraConfig::from_json(&json).expect_err(&format!(
      "a PEFT adapter setting `{field}` must be rejected (it changes inference)"
    ));
    // The error is a typed `Error::Parse` whose inner serde error's `Display`
    // names the offending field (the `E::custom(...)` rejection string).
    let Error::Parse(p) = &err else {
      panic!("expected Error::Parse for `{field}`, got {err:?}");
    };
    assert_eq!(p.context(), "LoraConfig::from_json");
    let msg = p.inner().to_string();
    assert!(
      msg.contains(field),
      "the rejection error for `{field}` should name the field; got: {msg}"
    );
  }
}

#[test]
fn peft_config_exotic_variant_rejection_is_shape_independent() {
  // The exotic-variant rejection MUST run before the shape-detection branches
  // — it cannot be gated behind the PEFT-shape markers (`peft_type` / `r` /
  // `lora_alpha` / `target_modules`) or the `lora_parameters` early return.
  // An adapter that carries an exotic field but NO PEFT marker, or that uses
  // the mlx-lm-native `lora_parameters` nesting, must still be rejected —
  // otherwise it silently loads as plain/mlx-lm LoRA at the wrong behavior.
  for (label, json) in [
    // (a) exotic field, NO PEFT markers at all (would otherwise fall through
    // to the bare-config default mlx-lm path).
    ("no-marker use_qalora", r#"{ "use_qalora": true }"#),
    (
      "no-marker alora",
      r#"{ "alora_invocation_tokens": [7, 8] }"#,
    ),
    ("no-marker velora", r#"{ "velora_config": {"rank": 2} }"#),
    (
      "no-marker monteclora",
      r#"{ "monteclora_config": {"k": 1} }"#,
    ),
    // (b) exotic field alongside the mlx-lm-native `lora_parameters` nesting
    // (would otherwise hit the early `lora_parameters` return).
    (
      "mlx-lm-shape use_qalora",
      r#"{ "lora_parameters": { "rank": 8 }, "use_qalora": true }"#,
    ),
    (
      "mlx-lm-shape velora",
      r#"{ "fine_tune_type": "lora", "num_layers": 4,
            "lora_parameters": { "rank": 8 }, "velora_config": {"x": 1} }"#,
    ),
  ] {
    assert!(
      LoraConfig::from_json(json).is_err(),
      "exotic-field config {label:?} must be rejected regardless of on-disk shape"
    );
  }
}

#[test]
fn peft_config_exotic_variant_defaults_are_accepted() {
  // The exotic fields at their PEFT DEFAULTS are NOT a signal — a config that
  // carries `use_qalora: false` (the default) or sets the others to `null`
  // parses cleanly as a normal LoRA adapter. `qalora_group_size` (no longer a
  // modeled field — only meaningful with `use_qalora: true`) is left to
  // parse-and-drop, so a stray default `16` is likewise harmless.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "use_qalora": false, "qalora_group_size": 16,
      "alora_invocation_tokens": null, "velora_config": null, "monteclora_config": null
    }"#;
  let cfg = LoraConfig::from_json(json)
    .expect("exotic fields at their defaults must not trip the rejection");
  assert_eq!(cfg.rank(), 8);
  assert!(!cfg.is_dora());

  // A bare mlx-lm-native config carrying only the exotic *defaults* is also
  // fine — the shape-independent guard must not false-positive on `null`s.
  let mlx = r#"{ "lora_parameters": { "rank": 4 },
      "use_qalora": false, "velora_config": null, "monteclora_config": null }"#;
  assert!(
    LoraConfig::from_json(mlx).is_ok(),
    "exotic defaults on an mlx-lm-shaped config must not trip the guard"
  );
}

// ───────── reject-unknown-active: the structural backstop (PEFT-flat) ─────────

#[test]
fn peft_config_arrow_config_is_err() {
  // `arrow_config` switches the forward (PEFT `resolve_lora_variant` returns
  // an `ArrowLinearVariant`); it is NOT a modeled field, so the structural
  // backstop must reject it when set to an object — BEFORE any tensor
  // translation. (Caught generically, no per-field code.)
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "arrow_config": { "top_k": 3 }
    }"#;
  let err =
    LoraConfig::from_json(json).expect_err("`arrow_config` set must be rejected (forward variant)");
  let Error::Parse(p) = &err else {
    panic!("expected Error::Parse, got {err:?}");
  };
  let msg = p.inner().to_string();
  assert!(
    msg.contains("arrow_config"),
    "the rejection should name `arrow_config`; got: {msg}"
  );
}

#[test]
fn peft_config_use_bdlora_is_err() {
  // `use_bdlora` switches the forward (PEFT `resolve_lora_variant` returns a
  // `BdLoraLinearVariant`); un-modeled, so the structural backstop rejects an
  // object value.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "use_bdlora": { "nblocks": 2 }
    }"#;
  let err = LoraConfig::from_json(json).expect_err("`use_bdlora` set must be rejected");
  let Error::Parse(p) = &err else {
    panic!("expected Error::Parse, got {err:?}");
  };
  let msg = p.inner().to_string();
  assert!(
    msg.contains("use_bdlora"),
    "the rejection should name `use_bdlora`; got: {msg}"
  );
}

#[test]
fn peft_config_invented_unknown_active_field_is_err() {
  // The whole point of the structural posture: a field that does not exist in
  // *today's* PEFT, set to an active value, must be rejected by name with NO
  // code change. Proves the backstop catches genuinely NEW fields (object and
  // scalar forms both).
  for (field, value) in [
    ("some_future_variant", r#"{ "k": 1 }"#),
    ("another_future_knob", "7"),
    ("yet_another_variant", "true"),
    ("a_future_string_variant", r#""enabled""#),
  ] {
    let json = format!(
      r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], {field:?}: {value} }}"#
    );
    let err = LoraConfig::from_json(&json).expect_err(&format!(
      "an active unknown field `{field}` must be rejected by the structural backstop"
    ));
    let Error::Parse(p) = &err else {
      panic!("expected Error::Parse for `{field}`, got {err:?}");
    };
    let msg = p.inner().to_string();
    assert!(
      msg.contains(field),
      "the rejection for `{field}` should name the field; got: {msg}"
    );
  }
}

#[test]
fn peft_config_unknown_field_inactive_value_is_accepted() {
  // PEFT's variant-gating fields default to None (→ JSON null) or False when
  // off. An unknown field set to `null` or `false` is provably the inactive
  // default, so it must be IGNORED (loads fine) — otherwise merely carrying a
  // defaulted future field would spuriously fail.
  for value in ["null", "false"] {
    let json = format!(
      r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "some_future_variant": {value} }}"#
    );
    let cfg = LoraConfig::from_json(&json).unwrap_or_else(|e| {
      panic!("an inactive (`{value}`) unknown field must be ignored, got: {e:?}")
    });
    assert_eq!(cfg.rank(), 8);
  }
  // Several inactive unknowns at once — still fine.
  let json = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "future_a": null, "future_b": false, "future_c": null }"#;
  assert!(
    LoraConfig::from_json(json).is_ok(),
    "multiple inactive unknown fields must all be ignored"
  );
}

#[test]
fn peft_config_benign_fields_with_real_values_are_accepted() {
  // BENIGN-IGNORE fields carry metadata / training-only info with no effect on
  // already-saved factors at inference. Set to real (active) values they must
  // still load — they are on the explicit allowlist, not unknown.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "task_type": "CAUSAL_LM",
      "revision": "main",
      "base_model_name_or_path": "meta-llama/Llama-3-8B",
      "auto_mapping": { "base_model_class": "LlamaForCausalLM" },
      "inference_mode": true,
      "peft_version": "0.19.2.dev0",
      "megatron_core": "megatron.core",
      "megatron_config": { "tensor_model_parallel_size": 1 },
      "runtime_config": { "ephemeral_gpu_offload": true },
      "eva_config": { "rho": 2.0 },
      "corda_config": { "corda_method": "ipm" },
      "lora_ga_config": { "scale": "stable" },
      "loftq_config": { "loftq_bits": 4 },
      "qalora_group_size": 16,
      "ensure_weight_tying": true
    }"#;
  let cfg = LoraConfig::from_json(json)
    .expect("benign metadata / training-only fields must load even when set");
  assert_eq!(cfg.rank(), 8);
  assert_eq!(cfg.lora_parameters.alpha, Some(16.0));
}

#[test]
fn peft_config_init_lora_weights_allowlist_rejects_non_factor_modes() {
  // `init_lora_weights` is an ALLOWLIST: only the pure factor seeds
  // (gaussian/eva/orthogonal + booleans) load; every other string rejects.
  // The base-weight-MUTATING modes subtract a low-rank residual from
  // `base_layer.weight` at init (peft `lora/layer.py`:
  // olora_init/pissa_init/corda_init/loftq_init/lora_ga_init), so a RAW
  // checkpoint saved with one pairs its factors with a modified base —
  // applying them to the unmodified base is silently wrong. `pissa_niter_<N>`
  // and prefixed `corda*` (PEFT dispatches BOTH via `startswith`, layer.py
  // :225/:228) must reject, AND an unknown/future mode must reject by default
  // — the allowlist's whole point, since a reject-list missed `corda_v1`. The
  // message names the offending mode (actionable); matching is
  // case-insensitive.
  for mode in [
    "pissa",
    "pissa_niter_4",
    "PISSA_NITER_16",
    "olora",
    "corda",
    "corda_v1",
    "lora_ga",
    "loftq",
    "some_future_init_mode",
  ] {
    let json = format!(
      r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "init_lora_weights": "{mode}" }}"#
    );
    match LoraConfig::from_json(&json) {
      Err(Error::Parse(p)) => {
        let msg = p.inner().to_string();
        assert!(
          msg.contains(mode),
          "the rejection should name the mode `{mode}`; got: {msg}"
        );
      }
      Ok(_) => {
        panic!("`init_lora_weights: \"{mode}\"` must be rejected (mutates base weight at init)")
      }
      Err(other) => panic!("expected Error::Parse for `{mode}`, got {other:?}"),
    }
  }
  // The pure factor SEEDS only seed the LoRA factors (or are overwritten at
  // load) and leave the base untouched, so they must still load. PEFT's
  // conversion path also rewrites converted adapters to `true`.
  for init in ["\"gaussian\"", "\"eva\"", "\"orthogonal\"", "true", "false"] {
    let json = format!(
      r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "init_lora_weights": {init} }}"#
    );
    assert!(
      LoraConfig::from_json(&json).is_ok(),
      "`init_lora_weights: {init}` is a pure factor seed and must load"
    );
  }
}

#[test]
fn peft_config_structural_reject_examples_layer_replication_and_token_indices() {
  // These forward/structure-switching fields are deliberately NOT on the
  // benign allowlist, so the structural backstop rejects them when active —
  // even though there is no per-field check for them.
  let cases = [
    (
      "layer_replication",
      r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
          "target_modules": ["q_proj"], "layer_replication": [[0, 4], [2, 5]] }"#,
    ),
    (
      "trainable_token_indices",
      r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
          "target_modules": ["q_proj"], "trainable_token_indices": [0, 1, 2] }"#,
    ),
    (
      "target_parameters",
      r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
          "target_modules": [], "target_parameters": ["feed_forward.experts.gate_up_proj"] }"#,
    ),
  ];
  for (field, json) in cases {
    match LoraConfig::from_json(json) {
      Ok(_) => panic!("`{field}` set must be rejected by the structural backstop"),
      Err(Error::Parse(p)) => {
        let msg = p.inner().to_string();
        assert!(
          msg.contains(field),
          "the rejection should name `{field}`; got: {msg}"
        );
      }
      Err(other) => panic!("expected Error::Parse for `{field}`, got {other:?}"),
    }
  }
}

#[test]
fn peft_config_valid_flat_fixture_still_loads() {
  // Regression: a realistic, fully-populated PEFT-flat config — exactly the
  // shape `LoraConfig.save_pretrained` writes, where EVERY field is serialized
  // including the forward-switching ones at their inactive (`null` / `false`)
  // defaults — must still load after the structural rule. The reject-if-active
  // fields below (`layer_replication`, `trainable_token_indices`,
  // `target_parameters`, `use_bdlora`, `arrow_config`, …) are present but
  // inactive, so the backstop must ignore them; the rule must not regress this
  // common case.
  let json = r#"{
      "peft_type": "LORA",
      "task_type": "CAUSAL_LM",
      "auto_mapping": null,
      "peft_version": "0.19.2.dev0",
      "base_model_name_or_path": "meta-llama/Llama-3-8B",
      "revision": null,
      "inference_mode": true,
      "r": 16,
      "lora_alpha": 32.0,
      "lora_dropout": 0.05,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
      "exclude_modules": null,
      "bias": "none",
      "use_rslora": false,
      "use_dora": false,
      "fan_in_fan_out": false,
      "lora_bias": false,
      "modules_to_save": null,
      "init_lora_weights": true,
      "layers_to_transform": null,
      "layers_pattern": null,
      "rank_pattern": {},
      "alpha_pattern": {},
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "use_qalora": false,
      "qalora_group_size": 16,
      "alora_invocation_tokens": null,
      "loftq_config": {},
      "eva_config": null,
      "corda_config": null,
      "lora_ga_config": null,
      "velora_config": null,
      "monteclora_config": null,
      "layer_replication": null,
      "trainable_token_indices": null,
      "target_parameters": null,
      "use_bdlora": null,
      "arrow_config": null,
      "ensure_weight_tying": false,
      "runtime_config": {"ephemeral_gpu_offload": false}
    }"#;
  let cfg = LoraConfig::from_json(json).expect("a realistic PEFT-flat config must still load");
  assert_eq!(cfg.rank(), 16);
  assert_eq!(cfg.lora_parameters.alpha, Some(32.0));
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 2.0); // 32/16
  let peft = cfg.peft().expect("PEFT selection");
  assert!(matches!(&peft.target_modules, Some(ModuleMatcher::List(_))));
}

#[test]
fn mlx_lm_native_fixture_still_loads_with_unknown_keys() {
  // Regression + scope: the mlx-lm-NATIVE nested shape keeps its existing
  // accept-and-ignore behavior — the reject-unknown-active rule applies to the
  // PEFT-flat branch ONLY. An mlx-lm-native config (the `lora_parameters`
  // early return) with an extra unknown key still loads.
  let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 8,
      "lora_parameters": { "rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["q_proj"] },
      "some_native_extra_key": { "whatever": 1 }
    }"#;
  let cfg = LoraConfig::from_json(json)
    .expect("mlx-lm-native shape must keep accept-and-ignore for unknown keys");
  assert_eq!(cfg.rank(), 8);
  assert!(matches!(
    cfg.selection,
    AdapterSelection::MlxLm { num_layers: 8 }
  ));
}

#[test]
fn peft_key_translation_embedding_lora_precise_reject() {
  // PEFT embedding-LoRA saves `lora_embedding_A` / `lora_embedding_B` factors
  // (`adapter_layer_names`, `lora/layer.py:105`). These ARE legitimate
  // low-rank factors, NOT a bias / modules_to_save tensor — so the translation
  // must reject them with a PRECISE "embedding" message, not the generic
  // bias/modules_to_save one (which would misclassify them). Embedding-LoRA
  // application is deferred, so reject (don't load) — but correctly named.
  for suffix in [
    ".lora_embedding_A",
    ".lora_embedding_B",
    ".lora_embedding_A.weight",
    ".lora_embedding_B.weight",
  ] {
    let key = format!("base_model.model.model.embed_tokens{suffix}");
    let mut arrays: HashMap<String, Array> = HashMap::new();
    arrays.insert(key.clone(), Array::zeros::<f32>(&(2, 3)).unwrap());
    match translate_peft_keys(arrays) {
      Ok(_) => panic!("embedding-LoRA key {key:?} must be rejected, not accepted"),
      Err(Error::LayerKeyed(p)) => {
        assert_eq!(p.layer(), key, "LayerKeyed must name the offending key");
        let Error::InvariantViolation(iv) = p.inner() else {
          panic!(
            "expected inner Error::InvariantViolation, got {:?}",
            p.inner()
          );
        };
        assert!(
          iv.requirement().to_lowercase().contains("embedding"),
          "the rejection requirement must mention embedding; got: {}",
          iv.requirement()
        );
        assert!(
          !iv.requirement().contains("bias") && !iv.requirement().contains("modules_to_save"),
          "embedding-LoRA must NOT be misclassified as bias/modules_to_save; got: {}",
          iv.requirement()
        );
      }
      Err(other) => panic!("expected Error::LayerKeyed, got {other:?}"),
    }
  }
}

#[test]
fn peft_config_all_selection_fields_parse() {
  // Every inference-affecting PEFT selection field on one config.
  let json = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "exclude_modules": ["lm_head"],
      "use_rslora": true,
      "use_dora": false,
      "fan_in_fan_out": true,
      "layers_to_transform": [0, 2, 4],
      "layers_pattern": "layers",
      "rank_pattern": { "q_proj": 16 },
      "alpha_pattern": { "v_proj": 64 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let peft = cfg.peft().unwrap();
  assert!(matches!(&peft.target_modules, Some(ModuleMatcher::List(_))));
  assert!(matches!(
    &peft.exclude_modules,
    Some(ModuleMatcher::List(_))
  ));
  assert!(peft.use_rslora);
  assert!(peft.fan_in_fan_out);
  assert_eq!(peft.layers_to_transform.as_deref(), Some(&[0, 2, 4][..]));
  assert_eq!(peft.layers_pattern, vec!["layers".to_string()]);
  assert!(cfg.fan_in_fan_out());
}

// ───────────── PEFT scale: rsLoRA + rank/alpha patterns ─────────────

#[test]
fn peft_rslora_scale_is_alpha_over_sqrt_r() {
  // use_rslora=true ⇒ scale = lora_alpha / sqrt(r). r=16, alpha=32 ⇒
  // 32/sqrt(16) = 32/4 = 8.0. Non-rsLoRA would be 32/16 = 2.0.
  let json = r#"{
      "peft_type": "LORA", "r": 16, "lora_alpha": 32.0,
      "target_modules": ["q_proj"], "use_rslora": true
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 8.0);
}

#[test]
fn peft_non_rslora_scale_is_alpha_over_r() {
  // use_rslora absent ⇒ scale = lora_alpha / r = 32/16 = 2.0.
  let json = r#"{
      "peft_type": "LORA", "r": 16, "lora_alpha": 32.0,
      "target_modules": ["q_proj"]
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert!(!cfg.peft().unwrap().use_rslora);
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 2.0);
}

#[test]
fn peft_rank_pattern_overrides_rank_per_module() {
  // `rank_pattern: {"q_proj": 32}` ⇒ a q_proj module resolves rank 32; a
  // v_proj module (no pattern) keeps the config-wide `r:8`.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "rank_pattern": { "q_proj": 32 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.rank_for("model.layers.0.self_attn.q_proj"), 32);
  assert_eq!(cfg.rank_for("model.layers.0.self_attn.v_proj"), 8);
  // The scale follows the overridden rank: q_proj is 16/32 = 0.5; v_proj
  // is the config-wide 16/8 = 2.0.
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 0.5);
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.v_proj"), 2.0);
}

#[test]
fn peft_alpha_pattern_overrides_alpha_per_module() {
  // `alpha_pattern: {"q_proj": 64}` ⇒ a q_proj module scales by 64/r; a
  // v_proj module keeps the config-wide `lora_alpha:16`.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "alpha_pattern": { "q_proj": 64 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  // q_proj: alpha 64 / r 8 = 8.0; v_proj: alpha 16 / r 8 = 2.0.
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 8.0);
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.v_proj"), 2.0);
}

#[test]
fn peft_rank_and_alpha_pattern_with_rslora() {
  // rank_pattern + alpha_pattern + rsLoRA compose: q_proj resolves rank 16,
  // alpha 64 ⇒ rsLoRA scale 64/sqrt(16) = 64/4 = 16.0.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "use_rslora": true,
      "rank_pattern": { "q_proj": 16 }, "alpha_pattern": { "q_proj": 64 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 16.0);
}

#[test]
fn peft_pattern_lookup_anchors_at_segment_boundary() {
  // PEFT `get_pattern_key` is `re.match(rf"(.*\.)?({key})$", module)` — the
  // pattern key matches a dotted suffix, NOT a mid-string substring.
  let patterns = vec![("q_proj".to_string(), 99i32)];
  assert_eq!(
    pattern_lookup(&patterns, "model.layers.0.self_attn.q_proj"),
    Some(99)
  );
  assert_eq!(pattern_lookup(&patterns, "q_proj"), Some(99));
  // a substring `xq_proj` must NOT match (the `(.*\.)?` needs a dot).
  assert_eq!(pattern_lookup(&patterns, "model.xq_proj"), None);
  // no match ⇒ None (caller falls back to the default).
  assert_eq!(pattern_lookup(&patterns, "model.layers.0.mlp.down"), None);
}

#[test]
fn peft_pattern_lookup_regex_key() {
  // PEFT pattern keys are themselves regex fragments — a `layers.0.*q_proj`
  // pattern keys block 0 only.
  let patterns = vec![("layers\\.0\\..*q_proj".to_string(), 64i32)];
  assert_eq!(
    pattern_lookup(&patterns, "model.layers.0.self_attn.q_proj"),
    Some(64)
  );
  assert_eq!(
    pattern_lookup(&patterns, "model.layers.1.self_attn.q_proj"),
    None
  );
}

#[test]
fn peft_rank_pattern_resolves_in_json_insertion_order_not_sorted() {
  // PEFT `get_pattern_key` returns the FIRST dict key (in insertion order)
  // whose `re.match(rf"(.*\.)?({key})$", module)` matches. For OVERLAPPING
  // pattern keys this tie-break is the JSON order — NOT a lexicographic sort.
  // Both keys below match `…self_attn.q_proj`, but `".*\.q_proj"` sorts BEFORE
  // `"self_attn.q_proj"` lexicographically ('.' 0x2E < 's' 0x73). With the
  // keys written `self_attn.q_proj` FIRST, insertion order must win (rank 11),
  // proving the resolver preserves JSON order rather than sorting (which would
  // wrongly pick 22).
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "rank_pattern": { "self_attn.q_proj": 11, ".*\\.q_proj": 22 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert_eq!(
    cfg.rank_for("model.layers.0.self_attn.q_proj"),
    11,
    "first-in-JSON-order key must win (a lexicographic sort would pick 22)"
  );

  // Reversing the JSON order flips the winner — confirming order, not value
  // or specificity, is the tie-break.
  let reversed = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "rank_pattern": { ".*\\.q_proj": 22, "self_attn.q_proj": 11 }
    }"#;
  let cfg2 = LoraConfig::from_json(reversed).unwrap();
  assert_eq!(
    cfg2.rank_for("model.layers.0.self_attn.q_proj"),
    22,
    "with the order reversed the other key wins — pure insertion-order tie-break"
  );
}

#[test]
fn peft_alpha_pattern_resolves_in_json_insertion_order_not_sorted() {
  // Same insertion-order tie-break for `alpha_pattern`. Two overlapping keys;
  // `".*\.q_proj"` sorts first lexicographically, but `q_proj` is written
  // first, so its alpha (40) wins over the other (80) — a sort would pick 80.
  let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "alpha_pattern": { "q_proj": 40, ".*\\.q_proj": 80 }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  // scale = alpha / r = 40 / 8 = 5.0 (NOT 80/8 = 10.0).
  assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 5.0);
}

// ───────────── PEFT selection: target / exclude / layers ─────────────

#[test]
fn peft_select_target_modules_list() {
  // PEFT `target_modules` list: every block's q_proj wraps (NO num_layers
  // window — the historical bug), v_proj does not.
  let weights = peft_toy_weights(4);
  let mut params = HashMap::new();
  for b in 0..4 {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.0.self_attn.v_proj"));
}

#[test]
fn peft_select_target_modules_regex() {
  // PEFT `target_modules` as a regex string — `re.fullmatch` over the whole
  // module path. `.*self_attn\.q_proj` matches only the q_proj paths.
  let weights = peft_toy_weights(3);
  let mut params = HashMap::new();
  for b in 0..3 {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*self_attn\\.q_proj" }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 3).unwrap();
  assert_eq!(layers.len(), 3);
  assert!(layers.contains_key("model.layers.2.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.0.self_attn.v_proj"));
}

#[test]
fn peft_select_exclude_modules_list() {
  // PEFT `exclude_modules` removes a target match. target=regex matching
  // both q and v proj; exclude=["v_proj"] ⇒ only q_proj wraps.
  let weights = peft_toy_weights(2);
  let mut params = HashMap::new();
  for b in 0..2 {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*_proj", "exclude_modules": ["v_proj"] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 2).unwrap();
  assert_eq!(layers.len(), 2);
  assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.0.self_attn.v_proj"));
}

#[test]
fn peft_select_exclude_modules_regex() {
  // `exclude_modules` as a regex (`re.fullmatch`): exclude every v_proj.
  let weights = peft_toy_weights(2);
  let mut params = HashMap::new();
  for b in 0..2 {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*_proj", "exclude_modules": ".*\\.v_proj" }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 2).unwrap();
  assert_eq!(layers.len(), 2);
  assert!(!layers.contains_key("model.layers.1.self_attn.v_proj"));
}

#[test]
fn peft_select_layers_to_transform_int() {
  // `layers_to_transform: 1` (a bare int) ⇒ only block 1's q_proj wraps.
  let weights = peft_toy_weights(4);
  let mut params = HashMap::new();
  params.insert(
    "model.layers.1.self_attn.q_proj".to_string(),
    plain_params(),
  );
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"], "layers_to_transform": 1 }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  assert_eq!(layers.len(), 1);
  assert!(layers.contains_key("model.layers.1.self_attn.q_proj"));
}

#[test]
fn peft_select_layers_to_transform_list() {
  // `layers_to_transform: [0, 3]` ⇒ only blocks 0 and 3 wrap.
  let weights = peft_toy_weights(5);
  let mut params = HashMap::new();
  for b in [0, 3] {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"], "layers_to_transform": [0, 3] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 5).unwrap();
  assert_eq!(layers.len(), 2);
  assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
  assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
  assert!(!layers.contains_key("model.layers.1.self_attn.q_proj"));
}

#[test]
fn peft_select_layers_pattern_custom_attr() {
  // `layers_pattern: "h"` extracts the block index after a `.h.` attribute
  // (GPT-2-style `transformer.h.0.…`) instead of `.layers.`.
  let mut weights = Weights::new();
  for b in 0..3 {
    weights.insert(
      format!("transformer.h.{b}.attn.c_attn.weight"),
      base_weight(),
    );
  }
  let mut params = HashMap::new();
  params.insert("transformer.h.2.attn.c_attn".to_string(), plain_params());
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["c_attn"], "layers_to_transform": [2],
      "layers_pattern": "h" }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 3).unwrap();
  assert_eq!(layers.len(), 1);
  assert!(layers.contains_key("transformer.h.2.attn.c_attn"));
}

#[test]
fn peft_select_no_restriction_adapts_all_blocks_over_16() {
  // A PEFT config with no `layers_to_transform` must adapt
  // EVERY matching block — including blocks 16..19 on a 20-block model.
  // Applying mlx-lm's `num_layers=16` trailing window here would be
  // wrong: it would drop blocks 0..3.
  let weights = peft_toy_weights(20);
  let mut params = HashMap::new();
  for b in 0..20 {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
  }
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"] }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 20).unwrap();
  assert_eq!(layers.len(), 20, "PEFT must adapt ALL 20 blocks, no window");
  // Block 0 (which a trailing-16 window would drop) IS adapted.
  assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
  assert!(layers.contains_key("model.layers.19.self_attn.q_proj"));
}

#[test]
fn peft_target_modules_all_linear_string_is_sentinel_not_regex() {
  // PEFT's `"all-linear"` string is a SENTINEL (expand to all linears minus
  // the output head), NOT a regex. The literal string compiles as a regex
  // that full-matches only "all-linear" (i.e. nothing), so a regex read would
  // select nothing — `all-linear` must instead select all rank-2 linears.
  let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": "all-linear" }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  let peft = match &cfg.selection {
    AdapterSelection::Peft(p) => p,
    other => panic!("expected a PEFT selection, got {other:?}"),
  };
  assert!(
    matches!(peft.target_modules, Some(ModuleMatcher::AllLinear)),
    "the `all-linear` string must parse to the AllLinear sentinel, not a regex"
  );

  // 3 blocks of q_proj + v_proj (all rank-2) plus a top-level `lm_head` (also
  // rank-2). `all-linear` selects every rank-2 linear EXCEPT the output head;
  // the `lm_head` weight is in the map but must NOT be adapted. (Factors are
  // shipped only for the q/v linears — `all-linear` is auto-discovery, so a
  // discovered-but-untrained linear is simply skipped, and a non-selected
  // `lm_head` with no factors is correctly never touched.)
  let weights = peft_toy_weights(3);
  let mut params = HashMap::new();
  for b in 0..3 {
    params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    params.insert(format!("model.layers.{b}.self_attn.v_proj"), plain_params());
  }

  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 3).unwrap();
  // 3 q_proj + 3 v_proj = 6; lm_head excluded by the head filter.
  assert_eq!(
    layers.len(),
    6,
    "all-linear adapts every linear minus the head"
  );
  for b in 0..3 {
    assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
    assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.v_proj")));
  }
  assert!(
    !layers.contains_key("lm_head"),
    "all-linear must EXCLUDE the output head (lm_head)"
  );
  // The unit-level selector check (with lm_head factors present) lives in
  // `peft_target_modules_all_linear_excludes_head_and_non_rank2` — it proves
  // the head is excluded by the *selector*, not merely by missing factors.
}

#[test]
fn peft_target_modules_all_linear_is_case_insensitive() {
  // PEFT lowercases `target_modules` before the sentinel compare
  // (`target_modules.lower() == "all-linear"`).
  for s in ["All-Linear", "ALL-LINEAR"] {
    let json = format!(r#"{{ "peft_type": "LORA", "r": 2, "target_modules": {s:?} }}"#);
    let cfg = LoraConfig::from_json(&json).unwrap();
    assert!(
      matches!(
        &cfg.selection,
        AdapterSelection::Peft(p) if matches!(p.target_modules, Some(ModuleMatcher::AllLinear))
      ),
      "`{s}` must be recognized as the all-linear sentinel (case-insensitive)"
    );
  }
}

#[test]
fn peft_target_modules_all_linear_excludes_head_and_non_rank2() {
  // The AllLinear selector applies BOTH halves of the predicate: rank-2
  // ("is a linear") AND not-the-output-head. A rank-1 weight (e.g. a norm
  // gain) and the `lm_head` are both excluded; a normal rank-2 linear is in.
  let q_w = base_weight(); // rank-2
  let norm_w = Array::zeros::<f32>(&(8usize,)).unwrap(); // rank-1
  let head_w = base_weight(); // rank-2 but it IS the head
  let peft = PeftSelection {
    target_modules: Some(ModuleMatcher::AllLinear),
    exclude_modules: None,
    layers_to_transform: None,
    layers_pattern: Vec::new(),
    rank_pattern: Vec::new(),
    alpha_pattern: Vec::new(),
    use_rslora: false,
    fan_in_fan_out: false,
  };
  assert!(peft_module_is_selected(
    "model.layers.0.self_attn.q_proj",
    &q_w,
    &peft
  ));
  assert!(
    !peft_module_is_selected("model.layers.0.input_layernorm", &norm_w, &peft),
    "a rank-1 weight is not a linear — all-linear must skip it"
  );
  assert!(
    !peft_module_is_selected("lm_head", &head_w, &peft),
    "the output head is excluded by all-linear even though it is rank-2"
  );
  // A nested `lm_head` (e.g. `model.lm_head`) is also the head.
  assert!(!peft_module_is_selected("model.lm_head", &head_w, &peft));
}

// ───────────── ModuleMatcher / peft_layer_index units ─────────────

#[test]
fn module_matcher_list_is_exact_or_dotted_suffix() {
  let m = ModuleMatcher::List(vec!["q_proj".to_string()]);
  assert!(m.matches("model.layers.0.self_attn.q_proj"));
  assert!(m.matches("q_proj"));
  // a substring without a dot boundary must NOT match.
  assert!(!m.matches("model.xq_proj"));
  assert!(!m.matches("q_proj_extra"));
}

#[test]
fn module_matcher_regex_is_full_match() {
  let m = ModuleMatcher::Regex(Box::new(Regex::new(r".*\.q_proj").unwrap()));
  assert!(m.matches("model.layers.0.self_attn.q_proj"));
  // `re.fullmatch` — a trailing extra segment must NOT match (the `.*\.q_proj`
  // pattern cannot consume the trailing `.bias`).
  assert!(!m.matches("model.layers.0.self_attn.q_proj.bias"));
  // A regex anchored to a specific suffix only — `re.fullmatch` requires the
  // WHOLE key to match, so a key with extra leading content is rejected (a
  // `search`-style match would wrongly accept it).
  let suffix = ModuleMatcher::Regex(Box::new(Regex::new(r"q_proj").unwrap()));
  assert!(suffix.matches("q_proj"));
  assert!(!suffix.matches("model.layers.0.self_attn.q_proj"));
}

#[test]
fn peft_layer_index_default_and_custom_pattern() {
  // default pattern: digits between dots after a prefix.
  assert_eq!(
    peft_layer_index("model.layers.7.self_attn.q_proj", &[]),
    Some(7)
  );
  // custom attribute name.
  assert_eq!(
    peft_layer_index("transformer.h.3.attn.c_attn", &["h".to_string()]),
    Some(3)
  );
  // no extractable index ⇒ None.
  assert_eq!(peft_layer_index("lm_head", &[]), None);
}

// ───────────── PEFT weight-key translation ─────────────

#[test]
fn peft_key_translation_strips_prefix_maps_suffix_transposes() {
  // `base_model.model.<path>.lora_A.weight` → `<path>.lora_a`, transposed
  // ([r,in] → [in,r]); `lora_B.weight` → `.lora_b` ([out,r] → [r,out]);
  // `lora_magnitude_vector` → `.m` (no transpose).
  let mut raw: HashMap<String, Array> = HashMap::new();
  // PEFT lora_A: [r=2, in=3]; lora_B: [out=4, r=2]; magnitude: [out=4].
  raw.insert(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
    Array::zeros::<f32>(&(2, 3)).unwrap(),
  );
  raw.insert(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
    Array::zeros::<f32>(&(4, 2)).unwrap(),
  );
  raw.insert(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector".to_string(),
    Array::zeros::<f32>(&(4usize,)).unwrap(),
  );
  // A non-PEFT key (no `base_model.model.` prefix) is dropped.
  raw.insert("some.stray.weight".to_string(), base_weight());

  let out = translate_peft_keys(raw).unwrap();
  assert_eq!(out.len(), 3, "3 LoRA tensors, the stray key dropped");
  let path = "model.layers.0.self_attn.q_proj";
  // lora_a: PEFT [2,3] transposed → [3,2].
  assert_eq!(out[&format!("{path}.lora_a")].shape(), &[3, 2]);
  // lora_b: PEFT [4,2] transposed → [2,4].
  assert_eq!(out[&format!("{path}.lora_b")].shape(), &[2, 4]);
  // m: PEFT [4] unchanged.
  assert_eq!(out[&format!("{path}.m")].shape(), &[4]);
}

#[test]
fn peft_key_translation_magnitude_vector_dot_weight_variant() {
  // PEFT may store the DoRA magnitude as `lora_magnitude_vector.weight`
  // (the in-memory `ModuleDict` form) — both spellings map to `.m`.
  let mut raw: HashMap<String, Array> = HashMap::new();
  raw.insert(
    "base_model.model.q_proj.lora_magnitude_vector.weight".to_string(),
    Array::zeros::<f32>(&(2usize,)).unwrap(),
  );
  let out = translate_peft_keys(raw).unwrap();
  assert!(out.contains_key("q_proj.m"));
}

// ───────────── PEFT end-to-end ─────────────

#[test]
fn peft_end_to_end_rslora_scale_and_all_blocks() {
  // A real PEFT adapter dir (config + adapter_model.safetensors): rsLoRA
  // scale, all matching blocks adapted.
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_e2e_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA", "r": 16, "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"], "use_rslora": true
    }"#;
  let q_paths: Vec<String> = (0..4)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 16, false, 0.01);
  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  // rsLoRA scale = lora_alpha / sqrt(r) = 32 / 4 = 8.0.
  if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
    assert_eq!(l.scale(), 8.0, "rsLoRA scale must be alpha/sqrt(r)");
  } else {
    panic!("expected a LoRA layer");
  }
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn peft_end_to_end_dora_with_magnitude_vector() {
  // A PEFT DoRA adapter: `use_dora: true` + a `lora_magnitude_vector` tensor
  // per module. The DoRA layer must build (the magnitude is loaded from the
  // PEFT-keyed safetensors).
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_dora_e2e_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["self_attn.q_proj"], "use_dora": true
    }"#;
  let q_paths: Vec<String> = (0..4)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 2, true, 0.01);
  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  assert!(matches!(
    layers.get("model.layers.0.self_attn.q_proj"),
    Some(LoraLayer::Dora(_))
  ));
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn peft_end_to_end_rank_pattern_per_module_scale() {
  // A PEFT adapter where `rank_pattern` overrides one block's rank. Block 0
  // gets rank 4 (via the pattern, factors shipped at rank 4); blocks 1..3
  // get the config-wide rank 2. Each module's scale follows its rank.
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_rankpat_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA", "r": 2, "lora_alpha": 8.0,
      "target_modules": ["self_attn.q_proj"],
      "rank_pattern": { "layers\\.0\\..*q_proj": 4 }
    }"#;
  std::fs::write(tmp.join("adapter_config.json"), cfg).unwrap();
  // Block 0: rank-4 PEFT factors; blocks 1..3: rank-2.
  let mut arrays: HashMap<String, Array> = HashMap::new();
  for b in 0..4 {
    let r = if b == 0 { 4 } else { 2 };
    let path = format!("model.layers.{b}.self_attn.q_proj");
    arrays.insert(
      format!("base_model.model.{path}.lora_A.weight"),
      Array::full::<f32>(&(r, 3usize), 0.01).unwrap(),
    );
    arrays.insert(
      format!("base_model.model.{path}.lora_B.weight"),
      Array::full::<f32>(&(2usize, r), 0.01).unwrap(),
    );
  }
  crate::io::save_safetensors(&tmp.join("adapter_model.safetensors"), &arrays).unwrap();

  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  // Block 0: alpha 8 / rank 4 = 2.0 (the rank_pattern override).
  if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
    assert_eq!(l.scale(), 2.0, "rank_pattern block-0 scale = alpha/4");
  } else {
    panic!("expected a LoRA layer at block 0");
  }
  // Block 1: alpha 8 / rank 2 = 4.0 (the config-wide rank).
  if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.1.self_attn.q_proj") {
    assert_eq!(l.scale(), 4.0, "default-rank block-1 scale = alpha/2");
  } else {
    panic!("expected a LoRA layer at block 1");
  }
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn peft_end_to_end_exclude_modules() {
  // A PEFT adapter targeting `.*_proj` but excluding v_proj — the v_proj
  // base layers must NOT be adapted (and the adapter ships no v_proj
  // factors, so a wrong selection would also trip the completeness check).
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_excl_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{
      "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*_proj", "exclude_modules": ".*\\.v_proj"
    }"#;
  let weights = peft_toy_weights(3);
  let q_paths: Vec<String> = (0..3)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 2, false, 0.01);
  let layers = load_adapters(&weights, &tmp, None, 3).unwrap();
  assert_eq!(layers.len(), 3);
  for b in 0..3 {
    assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
    assert!(!layers.contains_key(&format!("model.layers.{b}.self_attn.v_proj")));
  }
  std::fs::remove_dir_all(&tmp).ok();
}

// ───────────── fan_in_fan_out ─────────────

#[test]
fn peft_fan_in_fan_out_transposes_base_weight() {
  // With `fan_in_fan_out: true` the base weight is stored `[in, out]`.
  // `build_base_linear` transposes it back to `[out, in]` so the LoRA
  // forward matches the same adapter applied to a standard `[out, in]` base.
  //
  // Standard base: W = [[1,0,0],[0,1,0]] ([out=2, in=3]).
  // fan_in_fan_out base: Wᵀ = [[1,0],[0,1],[0,0]] ([in=3, out=2]).
  let standard_w = base_weight();
  let fifo_w = standard_w.transpose().unwrap(); // [3, 2] — the [in, out] layout
  let mut std_weights = Weights::new();
  std_weights.insert(
    "model.layers.0.self_attn.q_proj.weight".to_string(),
    standard_w,
  );
  let mut fifo_weights = Weights::new();
  fifo_weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), fifo_w);

  let mut params = HashMap::new();
  params.insert(
    "model.layers.0.self_attn.q_proj".to_string(),
    plain_params(),
  );

  let std_cfg = LoraConfig::from_json(
    r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
        "target_modules": ["q_proj"], "fan_in_fan_out": false }"#,
  )
  .unwrap();
  let fifo_cfg = LoraConfig::from_json(
    r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
        "target_modules": ["q_proj"], "fan_in_fan_out": true }"#,
  )
  .unwrap();

  let std_layers = linear_to_lora_layers(&std_weights, &std_cfg, &params, None, 1).unwrap();
  let fifo_layers = linear_to_lora_layers(&fifo_weights, &fifo_cfg, &params, None, 1).unwrap();

  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut std_out = std_layers["model.layers.0.self_attn.q_proj"]
    .forward(&x)
    .unwrap();
  let mut fifo_out = fifo_layers["model.layers.0.self_attn.q_proj"]
    .forward(&x)
    .unwrap();
  // The fan_in_fan_out base, after the transpose, must give the SAME forward
  // as the standard base.
  approx_eq(
    &fifo_out.to_vec::<f32>().unwrap(),
    &std_out.to_vec::<f32>().unwrap(),
    1e-5,
  );
}

#[test]
fn peft_fan_in_fan_out_quantized_is_err() {
  // `fan_in_fan_out` over a quantized base is rejected — transposing a
  // packed quantized weight would corrupt the bit-packing.
  let weight = Array::zeros::<u32>(&(8, 4)).unwrap();
  let scales = Array::zeros::<f32>(&(8, 4)).unwrap();
  let qbiases = Array::zeros::<f32>(&(8, 4)).unwrap();
  let mut weights = Weights::new();
  weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), weight);
  weights.insert("model.layers.0.self_attn.q_proj.scales".to_string(), scales);
  weights.insert(
    "model.layers.0.self_attn.q_proj.biases".to_string(),
    qbiases,
  );

  let quant = crate::lm::quant::PerLayerQuantization::from_global(crate::lm::quant::Quantization {
    group_size: 32,
    bits: 4,
    mode: crate::lm::quant::QuantMode::Affine,
  });
  let err = build_base_linear(
    &weights,
    "model.layers.0.self_attn.q_proj",
    &weights["model.layers.0.self_attn.q_proj.weight"],
    Some(&quant),
    true, // fan_in_fan_out
  )
  .unwrap_err();
  assert!(matches!(
    err,
    Error::LayerKeyed(ref payload)
      if matches!(payload.inner(), Error::InvariantViolation(_))
  ));
}

// ───────────── safetensors filename + neither-shape ─────────────

#[test]
fn peft_load_uses_adapter_model_safetensors_filename() {
  // A PEFT config pairs with `adapter_model.safetensors` (not mlx-lm's
  // `adapters.safetensors`). `load_adapters` picks the file by config shape.
  let tmp = std::env::temp_dir().join(format!("mlxrs_peft_fname_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  let cfg = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["self_attn.q_proj"] }"#;
  let q_paths: Vec<String> = (0..4)
    .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
    .collect();
  let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
  // write_mock_peft_adapter writes `adapter_model.safetensors` — NOT
  // `adapters.safetensors`. Confirm load still succeeds.
  write_mock_peft_adapter(&tmp, cfg, &q_refs, 2, false, 0.01);
  assert!(!tmp.join("adapters.safetensors").exists());
  assert!(tmp.join("adapter_model.safetensors").exists());
  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn mlxlm_native_path_unchanged_by_peft_work() {
  // A faithful mlx-lm-native config still parses to the MlxLm selection and
  // loads via `adapters.safetensors` — the PEFT additions did not regress
  // the native path.
  let json = r#"{
      "fine_tune_type": "lora", "num_layers": 8,
      "lora_parameters": { "rank": 4, "scale": 16.0, "keys": ["q_proj"] }
    }"#;
  let cfg = LoraConfig::from_json(json).unwrap();
  assert!(matches!(
    cfg.selection,
    AdapterSelection::MlxLm { num_layers: 8 }
  ));
  assert!(cfg.peft().is_none());
  assert_eq!(cfg.scale_for("anything"), 16.0);
  assert_eq!(cfg.rank_for("anything"), 4);
  assert!(!cfg.fan_in_fan_out());
}

// ═════════════════════════════ DoRA — spec-named tests ═════════════════════════════
//
// Tests with the names called out by the DoRA spec (#161). Some of these are
// (renamed) duplicates of pre-existing hand-traced tests; keeping both
// preserves the existing coverage *and* surfaces the spec-named tests in the
// test report (the spec asked for these exact names).

/// `dora_linear_forward_matches_python_reference` — assert the
/// [`DoRALinear::forward`] output matches a hand-traced scalar reference
/// derived from mlx-lm `tuner/dora.py::DoRALinear.__call__`
/// (`tuner/dora.py:111-128`).
///
/// Setup: base `W = I_{[2,3]}` truncated, `lora_a = I_{[3,2]}` truncated,
/// `lora_b = I_2`, `scale = 2.0`, `x = [1, 2, 3]`. Picks `m = [3, 3]` so the
/// `m / ‖adapted‖₂` renorm is the identity, isolating the DoRA wiring against
/// the LoRA arithmetic; expected `out = [3, 6]`.
#[test]
fn dora_linear_forward_matches_python_reference() {
  // adapted = W + scale·(lora_bᵀ @ lora_aᵀ) = [[3,0,0],[0,3,0]], ‖·‖₂ = [3,3].
  // m = [3, 3] ⇒ renorm = identity. base(x) = [1, 2], scale·z = [2, 4] ⇒ [3, 6].
  let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: lora_b(),
    magnitude: Some(m),
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = DoRALinear::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.forward(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
}

/// `dora_linear_fuse_into_base_round_trip` — fuse the DoRA adapter into the
/// base, run the fused base's plain forward, assert it matches the un-fused
/// DoRA forward within fp tolerance (mlx-lm `tuner/dora.py:32-56` /
/// `DoRA+Layers.swift::fuse`).
#[test]
fn dora_linear_fuse_into_base_round_trip() {
  let m = Array::from_slice::<f32>(&[1.5, 2.5], &(2usize,)).unwrap();
  let params = AdapterParams {
    lora_a: lora_a(),
    lora_b: lora_b(),
    magnitude: Some(m),
  };
  let base = BaseLinear::dense(base_weight(), None).unwrap();
  let layer = DoRALinear::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut via_forward = layer.forward(&x).unwrap();
  // Fuse, then run the fused base's plain forward — must match.
  let fused = layer.fuse(false).unwrap();
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    1e-4,
  );
}

/// `dora_embedding_forward_matches_python_reference` — assert
/// [`DoRAEmbedding::forward`] matches a hand-traced scalar reference for
/// mlx-lm `tuner/dora.py::DoRAEmbedding.__call__` (`tuner/dora.py:198-210`).
///
/// Setup: `weight = I_{[3, 3]}` (3 token rows, 3 dims), so for `x = [0, 2]`:
/// `y[0] = [1,0,0]`, `y[1] = [0,0,1]`. `lora_a = zeros([3, 2])` and
/// `lora_b = zeros([2, 3])` ⇒ `z = 0` ⇒ `adapted == y` ⇒ `denom = ‖y‖₂ = [1, 1]`.
/// Setting `m = [1, 1, 1]` gives `m[x] / denom = [1, 1]` ⇒ `out == y`,
/// which validates the gather + per-token renorm wiring against a known
/// fixed point of the DoRA computation.
#[test]
fn dora_embedding_forward_matches_python_reference() {
  let num_embeddings = 3usize;
  let dims = 3usize;
  let r = 2usize;
  // weight = I_3 (one-hot rows).
  #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();

  let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
  let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, 2.0).unwrap();

  // Gather rows 0 and 2 ⇒ [[1,0,0], [0,0,1]].
  let ids = Array::from_slice::<i32>(&[0, 2], &(2usize,)).unwrap();
  let mut out = layer.forward(&ids).unwrap();
  approx_eq(
    &out.to_vec::<f32>().unwrap(),
    &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    1e-5,
  );
}

/// Companion to [`dora_embedding_forward_matches_python_reference`] for a
/// **non-identity** DoRA renorm — set `m` to *half* the per-token adapted
/// norm so the per-token renorm halves the output. With `lora_*` zero,
/// `adapted = y` and `‖y‖₂ = [1, 1]`; `m = [0.5, 0.5, 0.5]` ⇒ `m[x] / denom
/// = [0.5, 0.5]` ⇒ `out = 0.5 · y`. Validates the per-token renorm wiring
/// distinguishes from the global `as_linear` renorm path.
#[test]
fn dora_embedding_forward_per_token_renorm_halves() {
  let num_embeddings = 3usize;
  let dims = 3usize;
  let r = 2usize;
  #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
  let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&[0.5, 0.5, 0.5], &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, 1.0).unwrap();
  let ids = Array::from_slice::<i32>(&[1], &(1usize,)).unwrap();
  let mut out = layer.forward(&ids).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[0.0, 0.5, 0.0], 1e-5);
}

/// DoRAEmbedding's `as_linear` is the tied-weight LM-head path
/// (`tuner/dora.py:212-224`) — for a one-hot embedding table with zero
/// adapter, `as_linear(x) == x @ Iᵀ = x` modulo the global renorm
/// `(m / ‖weight‖₂)` which is `[1, 1, 1]` here ⇒ identity output.
#[test]
fn dora_embedding_as_linear_one_hot_identity() {
  let num_embeddings = 3usize;
  let dims = 3usize;
  let r = 2usize;
  #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
  let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
  // m = ‖weight‖₂ row-wise = [1, 1, 1] ⇒ renorm = identity globally.
  let m = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, 2.0).unwrap();
  // x = [[1, 2, 3]] ⇒ x @ Iᵀ = [1, 2, 3] ⇒ renormed = [1, 2, 3].
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let mut out = layer.as_linear(&x).unwrap();
  approx_eq(&out.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0], 1e-5);
}

/// [`DoRAEmbedding::fuse`] round-trip — fuse the adapter into a fresh dense
/// embedding and assert the fused weight's `as_linear` matches the un-fused
/// `as_linear` within fp tolerance (mlx-lm `tuner/dora.py:153-166`). The
/// `forward` path is per-token-renormed and intentionally distinct from
/// `fuse`; `as_linear` is the global-renorm path that fuse mirrors.
#[test]
fn dora_embedding_fuse_round_trip() {
  let num_embeddings = 3usize;
  let dims = 3usize;
  let r = 2usize;
  #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.5, 0.0,
        0.0, 1.0, 0.5,
        0.5, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  let lora_a =
    Array::from_slice::<f32>(&[0.1, 0.0, 0.0, 0.1, 0.1, 0.1], &(num_embeddings, r)).unwrap();
  let lora_b = Array::from_slice::<f32>(&[0.2, 0.0, 0.1, 0.0, 0.1, 0.2], &(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&[1.5, 2.0, 1.2], &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 0.5], &(1, dims)).unwrap();
  let mut via_aslinear = layer.as_linear(&x).unwrap();
  let fused = layer.fuse().unwrap();
  let mut via_fused_aslinear = fused.as_linear(&x).unwrap();
  approx_eq(
    &via_fused_aslinear.to_vec::<f32>().unwrap(),
    &via_aslinear.to_vec::<f32>().unwrap(),
    1e-4,
  );
}

/// DoRAEmbedding rejects a magnitude-less `AdapterParams` (LoRA-flavored
/// factors) at construction — same contract as [`DoRALinear`].
#[test]
fn dora_embedding_requires_magnitude() {
  let num_embeddings = 3usize;
  let dims = 3usize;
  let r = 2usize;
  let weight = Array::zeros::<f32>(&(num_embeddings, dims)).unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
  let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: None,
  };
  let err = DoRAEmbedding::new(base, params, 1.0).unwrap_err();
  assert!(
    matches!(&err, Error::MissingField(p)
        if p.type_name() == "DoRAEmbedding::new" && p.field().contains("magnitude")),
    "expected Error::MissingField naming `magnitude`, got {err:?}"
  );
}

/// DoRAEmbedding rejects a `lora_a` whose leading axis is not
/// `num_embeddings` (the embedding-orientation factor cross-check).
#[test]
fn dora_embedding_rejects_wrong_factor_shape() {
  let num_embeddings = 3usize;
  let dims = 3usize;
  let r = 2usize;
  let weight = Array::zeros::<f32>(&(num_embeddings, dims)).unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  // bad: lora_a is [2, r] instead of [num_embeddings=3, r].
  let bad_a = Array::zeros::<f32>(&(2usize, r)).unwrap();
  let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
  let m = Array::zeros::<f32>(&(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a: bad_a,
    lora_b,
    magnitude: Some(m),
  };
  let err = DoRAEmbedding::new(base, params, 1.0).unwrap_err();
  // `validate_embedding_factor_shapes` hits the leading-axis cross-check
  // (`a_leading_axis != num_embeddings`) and returns the typed
  // `Error::LengthMismatch` (expected = num_embeddings, actual = 2).
  assert!(
    matches!(&err, Error::LengthMismatch(p)
        if p.expected() == num_embeddings && p.actual() == 2
          && p.context().contains("lora_a")),
    "expected Error::LengthMismatch for wrong leading axis, got {err:?}"
  );
}

/// `qdora_linear_forward_matches_python_reference` — assert the QDoRA
/// forward (DoRA over a quantized base) matches the dense DoRA forward
/// within affine-quantization error, exercising the `quantized_matmul` base
/// path against the dense baseline.
#[test]
fn qdora_linear_forward_matches_python_reference() {
  let input_dims = 64usize;
  let output_dims = 2usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();

  // m derived from the *dense* adapted weight so dense and quantized share an
  // identical magnitude vector — any difference is then quantization error
  // alone (not a magnitude mismatch).
  let dense_params_no_m = AdapterParams {
    lora_a: la.try_clone().unwrap(),
    lora_b: lb.try_clone().unwrap(),
    magnitude: None,
  };
  let scale = 2.0f32;
  let delta = lora_delta(&dense_params_no_m, scale).unwrap();
  let adapted = dense_w.add(&delta).unwrap();
  let m = ops::linalg_full::norm(&adapted, 2.0, &[1], false).unwrap();

  let dense_base = BaseLinear::dense(dense_w.try_clone().unwrap(), None).unwrap();
  let dense_layer = DoRALinear::new(
    dense_base,
    AdapterParams {
      lora_a: la.try_clone().unwrap(),
      lora_b: lb.try_clone().unwrap(),
      magnitude: Some(m.try_clone().unwrap()),
    },
    scale,
  )
  .unwrap();
  let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();
  let mut dense_out = dense_layer.forward(&x).unwrap();

  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
  let q_layer = DoRALinear::new(
    q_base,
    AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: Some(m),
    },
    scale,
  )
  .unwrap();
  let mut q_out = q_layer.forward(&x).unwrap();

  approx_eq(
    &q_out.to_vec::<f32>().unwrap(),
    &dense_out.to_vec::<f32>().unwrap(),
    2e-2,
  );
}

/// `qdora_linear_fuse_round_trip` — fuse a QDoRA layer (`dequantize=true`)
/// into a dense base, assert the fused base's plain forward matches the
/// un-fused QDoRA forward within quantization error.
#[test]
fn qdora_linear_fuse_round_trip() {
  let input_dims = 64usize;
  let output_dims = 2usize;
  let mut wdata = vec![1.0f32; input_dims];
  wdata.extend(vec![0.5f32; input_dims]);
  let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
  let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
  let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let m = Array::from_slice::<f32>(&[1.5, 2.5], &(output_dims,)).unwrap();
  let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

  let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
  let q_base =
    BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
  let q_layer = DoRALinear::new(
    q_base,
    AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: Some(m),
    },
    2.0,
  )
  .unwrap();
  let mut via_forward = q_layer.forward(&x).unwrap();
  let fused = q_layer.fuse(true).unwrap();
  assert!(matches!(fused, BaseLinear::Dense { .. }));
  let mut via_fused = fused.base_output(&x).unwrap();
  approx_eq(
    &via_fused.to_vec::<f32>().unwrap(),
    &via_forward.to_vec::<f32>().unwrap(),
    2e-2,
  );
}

/// `load_dora_adapter_from_safetensors` — write a small adapter directory
/// (`adapter_config.json` with `fine_tune_type: "dora"`, plus
/// `adapters.safetensors` carrying `lora_a` / `lora_b` / `m` for each
/// targeted path), load via the existing [`load_adapters`] entry, and
/// verify the resulting layers are [`LoraLayer::Dora`] with the right
/// magnitude shape.
#[test]
fn load_dora_adapter_from_safetensors() {
  let tmp = std::env::temp_dir().join(format!("mlxrs_a2_dora_load_{}", std::process::id()));
  std::fs::create_dir_all(&tmp).unwrap();
  write_mock_adapter(&tmp, "dora", true);

  let weights = toy_weights();
  let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
  assert_eq!(layers.len(), 4);
  for b in 0..4 {
    let key = format!("model.layers.{b}.self_attn.q_proj");
    match layers.get(&key) {
      Some(LoraLayer::Dora(d)) => {
        // magnitude must be shape [output_dims=2] per `write_mock_adapter`'s
        // `m = [3, 3]` fixture.
        assert_eq!(d.magnitude().shape(), &[2]);
      }
      other => panic!("expected DoRA layer at {key}, got {other:?}"),
    }
  }
  std::fs::remove_dir_all(&tmp).ok();
}

/// `linear_to_dora_layers_grafts_correctly` — graft DoRA adapters into the
/// targeted linear paths of a synthetic model and verify only the targeted
/// layers are wrapped (and as the `Dora` variant), others are untouched.
/// Uses [`linear_to_lora_layers`] with a `fine_tune_type: "dora"` config —
/// the existing entrypoint is the "sibling" referenced in the DoRA spec
/// (dispatches to [`DoRALinear`] via `LoraConfig::is_dora()`).
#[test]
fn linear_to_dora_layers_grafts_correctly() {
  let weights = toy_weights();
  // mlx-lm-native DoRA config: keys=["self_attn.q_proj"], rank=2.
  let cfg = LoraConfig {
    fine_tune_type: FineTuneType::Dora,
    lora_parameters: LoraParameters {
      rank: 2,
      scale: Some(2.0),
      alpha: None,
      keys: vec!["self_attn.q_proj".to_string()],
      dropout: None,
    },
    use_dora: false,
    selection: AdapterSelection::MlxLm { num_layers: 16 },
  };

  // DoRA AdapterParams for each q_proj path — m chosen so the renorm is
  // identity (‖adapted‖₂ = [3, 3] for these factors at scale 2.0).
  let mut params = HashMap::new();
  for b in 0..4 {
    let path = format!("model.layers.{b}.self_attn.q_proj");
    params.insert(
      path,
      AdapterParams {
        lora_a: lora_a(),
        lora_b: lora_b(),
        magnitude: Some(Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap()),
      },
    );
  }

  let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
  // Exactly 4 q_proj paths wrapped (one per block); k_proj and lm_head left
  // untouched.
  assert_eq!(layers.len(), 4);
  for b in 0..4 {
    let key = format!("model.layers.{b}.self_attn.q_proj");
    assert!(
      matches!(layers.get(&key), Some(LoraLayer::Dora(_))),
      "expected DoRA layer at {key}"
    );
  }
  assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
  assert!(!layers.contains_key("lm_head"));
}

// ════════════════ DoRAEmbedding mixed-precision dtype ════════════════
//
// Regression coverage for the dtype flow in
// `DoRAEmbedding::forward` and `DoRAEmbedding::as_linear`: the low-rank
// product (`z` / `delta`) and the renorm scale (`m[x]/denom`,
// `m/denom`) must stay uncast through the L2 norm and the final
// multiply, matching mlx-lm `tuner/dora.py:198-224` (which keeps them
// uncast for the norm-and-scale compute and only casts the `out`
// accumulator). With f16 base and f32 adapter, casting them to the base
// / input dtype upfront silently drops ~16 bits of precision through the
// renorm divisor (and ~7 bits with bf16 base, where rounding is much
// coarser).
//
// Strategy: an `y ≈ -z` cancellation fixture so the f16/bf16 rounding of `z`
// perturbs ‖y + z‖ by a *relative* amount well above the fp16/bf16 tolerance
// floor — the uncast pipeline matches the f64 scalar reference; an
// upfront-cast pipeline would not. The companion regression-oracle
// test asserts this directly by computing both reference paths and
// confirming the real output is closer to the uncast one.

/// f16 round-trip on an f32 fixture — `f64(f16::from_f32(x))`. Models the
/// `astype(F16)` rounding mlx applies when an f32 source is cast to f16,
/// so the scalar reference operates on the SAME bit patterns the kernel
/// does.
fn f16_rt(x: f32) -> f64 {
  half::f16::from_f32(x).to_f64()
}

/// bf16 round-trip on an f32 fixture — `f64(bf16::from_f32(x))`.
fn bf16_rt(x: f32) -> f64 {
  half::bf16::from_f32(x).to_f64()
}

/// Cancellation fixture inputs reused by the four mixed-precision tests.
/// `y ≈ -z` per token (with a small `eps` perturbation so `denom > 0`) so
/// the per-token L2 norm of `adapted = y + z` is small and any rounding
/// error in `z` to fp16/bf16 shows up as a large *relative* change in the
/// renorm divisor. Coupled with order-magnitude `m`, the resulting
/// amplified `m/denom` multiplier makes the upfront-cast bug visible at
/// the fp16 tolerance floor.
///
/// All weight values are chosen to be near (but NOT all exactly) on the f16
/// representable grid — picking values like 1.0 (exact) for `y` and 0.99 …
/// fractions for the cancelling `z` means the f32-precision `z` carries
/// mantissa bits that f16 rounds away, exactly the scenario an
/// upfront-cast divergence magnifies through `‖adapted‖₂`.
#[allow(clippy::type_complexity)] // 5-tuple of nested Vec<Vec<f32>> is just
// the fixture's "5 input tensors" shape; aliasing each would obscure more
// than it'd clarify in a test fixture.
fn mp_fixture() -> (
  Vec<Vec<f32>>, // weight_f32 [4][4]
  Vec<Vec<f32>>, // lora_a_f32 [4][2]
  Vec<Vec<f32>>, // lora_b_f32 [2][4]
  Vec<f32>,      // m_f32 [4]
  f32,           // scale
) {
  // y = weight[tid] — chosen to be exactly representable in f16 so the
  // round-trip is the identity on y (isolating the divergence to z's
  // rounding).
  // Row 0: [1,1,1,1] — uniform, simplest cancellation.
  // Row 1: [0.5, 0.5, 0.5, 0.5] — half-scale, exact in f16/bf16.
  // Row 2: [0.25, 0.25, 0.25, 0.25] — quarter-scale.
  // Row 3: [-0.75, -0.75, -0.75, -0.75] — negative direction.
  let weight_f32 = vec![
    vec![1.0, 1.0, 1.0, 1.0],
    vec![0.5, 0.5, 0.5, 0.5],
    vec![0.25, 0.25, 0.25, 0.25],
    vec![-0.75, -0.75, -0.75, -0.75],
  ];
  // lora_a[tid] picks ONE adversarial column of lora_b per token (so z is
  // dominated by a single per-row contribution, easier to reason about).
  let lora_a_f32 = vec![
    vec![1.0, 0.0],
    vec![0.5, 0.0],
    vec![0.25, 0.0],
    vec![-0.75, 0.0],
  ];
  // lora_b row 0 carries the cancelling magnitude: `-0.99853` (≈ -1) for
  // all dims. Combined with lora_a (which picks the matching scalar), z
  // for each token is `-0.99853 * weight_scale`, so adapted = y + z ≈
  // tiny. The exact value -0.99853 is OFF the f16 grid (f16 ULP near 1
  // is ~9.77e-4), so f16-rounding z shifts it by an absolute amount
  // comparable to ‖adapted‖ itself.
  let lora_b_f32 = vec![
    vec![-0.99853, -0.99853, -0.99853, -0.99853],
    vec![0.0, 0.0, 0.0, 0.0],
  ];
  // m chosen so the renorm scale `m/denom` is large enough (~100s) that
  // the per-token cast-vs-uncast divergence in `denom` (relative ~25-30%
  // under this cancellation fixture, since z's f16-rounding error is a
  // sizeable fraction of ‖adapted‖) MULTIPLIES `out_pre` into a final-out
  // delta well above the fp16 tolerance floor (5e-3). m roughly tracks
  // each token's |y| so the final output stays in fp16 range (~order 1).
  let m_f32 = vec![1.0, 0.5, 0.25, 0.75];
  let scale = 1.0f32;
  (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale)
}

/// Build the scalar f64 reference for `DoRAEmbedding::forward` matching the
/// mlx-lm pipeline (`tuner/dora.py:198-210`) with optional `cast_z_upfront`
/// to model the divergent (cast-upfront) computation. The reference
/// operates on `rt(x)` — a
/// pre-rounded version of the f32 source (f16 or bf16 round-trip) so it
/// reflects the exact bits the kernel sees.
///
/// Returns the kernel-equivalent promoted-dtype outputs for each token in
/// `ids`, flattened to a `Vec<f32>` for direct comparison against the
/// kernel output (which we extract via `astype(F32)`). The final value is
/// NOT round-tripped to the narrow dtype — `forward` now returns the
/// promoted dtype directly (mlx-lm `tuner/dora.py:208` returns
/// `(self.m[x] / denom)[..., None] * out` with no astype; the port
/// mirrors that exactly).
#[allow(clippy::too_many_arguments)]
fn forward_scalar_reference(
  weight_f32: &[Vec<f32>],
  lora_a_f32: &[Vec<f32>],
  lora_b_f32: &[Vec<f32>],
  m_f32: &[f32],
  scale: f32,
  ids: &[usize],
  rt: fn(f32) -> f64,
  cast_z_upfront: bool,
) -> Vec<f32> {
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let scale_f64 = scale as f64;
  let mut out = Vec::with_capacity(ids.len() * dims);
  for &tid in ids {
    // y = round(weight[tid]) — what the kernel sees after the f16/bf16 cast.
    let y_rt: Vec<f64> = weight_f32[tid].iter().map(|&w| rt(w)).collect();
    // z_uncast[d] = scale * sum_r lora_a[tid][r] * lora_b[r][d] — f32→f64
    // (no rounding; the f32 source bits are exactly representable in f64).
    let mut z_uncast = vec![0.0f64; dims];
    for d in 0..dims {
      let mut acc = 0.0f64;
      for k in 0..r {
        acc += (lora_a_f32[tid][k] as f64) * (lora_b_f32[k][d] as f64);
      }
      z_uncast[d] = scale_f64 * acc;
    }
    // z_cast = round(z_uncast) — what `astype(y.dtype)` produces.
    let z_cast: Vec<f64> = z_uncast.iter().map(|&v| rt(v as f32)).collect();
    // out_pre = round(y + z_cast) — what `out = y + dropout(z).astype(y.dtype)`
    // produces (the add itself runs at y.dtype because both operands are now
    // f16/bf16).
    let out_pre: Vec<f64> = (0..dims)
      .map(|d| rt((y_rt[d] + z_cast[d]) as f32))
      .collect();
    // adapted = y + z_for_norm. The divergent path = `cast_z_upfront`
    // true; the correct path = false (uncast). mlx promotes
    // y(f16/bf16) + z(f32) to f32; we work at f64 to give the reference
    // bounded round-off well below the fp16/bf16 tolerance.
    let z_for_norm = if cast_z_upfront { &z_cast } else { &z_uncast };
    let adapted: Vec<f64> = (0..dims).map(|d| y_rt[d] + z_for_norm[d]).collect();
    let denom = adapted.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_scale = (m_f32[tid] as f64) / denom;
    // scaled_out = norm_scale * out_pre — at f64; mlx runs at f32
    // (promotion from f16/bf16 * f32). NO final cast to y.dtype — `forward`
    // returns the promoted dtype directly (mlx-lm `tuner/dora.py:208`),
    // so the reference is returned at the promoted dtype too (f32 for the
    // mixed-precision fixture). f64 → f32 narrowing is fine here: the f64
    // reference's round-off well below the fp16/bf16 tolerance floor used
    // by the assertions.
    for &op in &out_pre {
      let scaled = norm_scale * op;
      out.push(scaled as f32);
    }
  }
  out
}

/// Build the scalar f64 reference for `DoRAEmbedding::as_linear` matching
/// mlx-lm `tuner/dora.py:212-224`. With `cast_delta_upfront`, models the
/// divergent path (delta cast to weight.dtype before the row-norm);
/// without it, the uncast path. Output is the kernel-equivalent
/// `[batch, num_embeddings]` flattened to `Vec<f32>` for direct comparison
/// (the kernel returns the promoted dtype — for f16 base × f32 adapter →
/// f32 — so we DON'T round-trip the final value to fp16, matching the new
/// code's "no final astype" choice).
#[allow(clippy::too_many_arguments)]
fn as_linear_scalar_reference(
  weight_f32: &[Vec<f32>],
  lora_a_f32: &[Vec<f32>],
  lora_b_f32: &[Vec<f32>],
  m_f32: &[f32],
  scale: f32,
  x_f32: &[Vec<f32>],
  rt: fn(f32) -> f64,
  cast_delta_upfront: bool,
) -> Vec<f32> {
  let num_embeddings = weight_f32.len();
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let batch = x_f32.len();
  let scale_f64 = scale as f64;
  // delta_uncast[e][d] = scale * sum_r lora_a[e][r] * lora_b[r][d] (f64).
  let mut delta_uncast = vec![vec![0.0f64; dims]; num_embeddings];
  for e in 0..num_embeddings {
    for d in 0..dims {
      let mut acc = 0.0f64;
      for k in 0..r {
        acc += (lora_a_f32[e][k] as f64) * (lora_b_f32[k][d] as f64);
      }
      delta_uncast[e][d] = scale_f64 * acc;
    }
  }
  // delta_cast[e][d] = round(delta_uncast[e][d]) — what astype(weight.dtype)
  // produces; only used by the buggy-cast reference.
  let delta_cast: Vec<Vec<f64>> = delta_uncast
    .iter()
    .map(|row| row.iter().map(|&v| rt(v as f32)).collect())
    .collect();
  let delta_for_norm = if cast_delta_upfront {
    &delta_cast
  } else {
    &delta_uncast
  };
  // adapted[e][d] = weight_rt[e][d] + delta_for_norm[e][d] (f64; mlx promotes
  // to f32 — f64 reference is precise enough for the fp tolerance).
  let mut adapted = vec![vec![0.0f64; dims]; num_embeddings];
  for e in 0..num_embeddings {
    for d in 0..dims {
      adapted[e][d] = rt(weight_f32[e][d]) + delta_for_norm[e][d];
    }
  }
  // denom[e] = ‖adapted[e]‖₂, axis=1 (`tuner/dora.py:219`).
  let denom: Vec<f64> = adapted
    .iter()
    .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
    .collect();
  // norm_scale[e] = m[e] / denom[e] (UNCAST, `tuner/dora.py:222`).
  let norm_scale: Vec<f64> = (0..num_embeddings)
    .map(|e| (m_f32[e] as f64) / denom[e])
    .collect();
  // y[b][e] = sum_d x_rt[b][d] * weight_rt[e][d] — x@weightᵀ at the base
  // dtype. f16+f32 promote to f32; f64 reference is more than enough.
  let mut out = Vec::with_capacity(batch * num_embeddings);
  for x_row in x_f32 {
    let x_rt: Vec<f64> = x_row.iter().map(|&v| rt(v)).collect();
    for e in 0..num_embeddings {
      let mut y_be = 0.0f64;
      for d in 0..dims {
        y_be += x_rt[d] * rt(weight_f32[e][d]);
      }
      // scaled_z_be = scale * (x_rt @ lora_b.T @ lora_a.T)[e]
      //            = sum_d x_rt[d] * delta_uncast[e][d] / something...
      // Cleanest: z_be = sum_k (x @ lora_b.T)[k] * lora_a[e][k]
      //                = sum_k (sum_d x_rt[d] * lora_b[k][d]) * lora_a[e][k]
      let xb: Vec<f64> = (0..r)
        .map(|k| {
          (0..dims)
            .map(|d| x_rt[d] * (lora_b_f32[k][d] as f64))
            .sum::<f64>()
        })
        .collect();
      let z_be: f64 = (0..r).map(|k| xb[k] * (lora_a_f32[e][k] as f64)).sum();
      let scaled_z_be = scale_f64 * z_be;
      // out_pre = y + round(scaled_z, x.dtype). Cast scaled_z to x.dtype
      // first (mirrors mlx-lm `(self.scale * z).astype(x.dtype)`).
      let scaled_z_cast = rt(scaled_z_be as f32);
      let out_pre = y_be + scaled_z_cast;
      // Final: norm_scale[e] * out_pre. mlx promotes f32*f16 → f32; f64
      // reference returned as f32 for direct compare against the kernel
      // output extracted via `astype(F32)`. No final astype to base dtype
      // — mlx-lm doesn't cast here and the port doesn't either.
      out.push((norm_scale[e] * out_pre) as f32);
    }
  }
  out
}

/// `dora_embedding_forward_mixed_precision_matches_reference_f16_base_f32_adapter`
/// — exercise the dtype fix: with an f16 embedding weight and f32
/// adapter factors + magnitude, the renorm divisor must be computed at the
/// UNCAST dtype (`forward`'s `adapted = y + z` uses uncast `z`,
/// mirroring mlx-lm `tuner/dora.py:204`). Adversarial `y ≈ -z` fixture so
/// the f16 rounding of `z` perturbs ‖adapted‖ by a relative amount above
/// the fp16 tolerance — an upfront-cast computation would mismatch the
/// scalar reference by orders of magnitude.
///
/// Also asserts the output dtype is **f32** — `forward` carries no
/// trailing `astype(y.dtype)`, so it returns mlx's promoted dtype
/// directly (mlx-lm `tuner/dora.py:208` returns `(m[x]/denom)[..., None] *
/// out` with no astype; f16 base × f32 adapter promotes to f32).
#[test]
fn dora_embedding_forward_mixed_precision_matches_reference_f16_base_f32_adapter() {
  let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
  let num_embeddings = weight_f32.len();
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
  let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
  let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
  let weight_f16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let base = BaseEmbedding::dense(weight_f16).unwrap();
  let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
  let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, scale).unwrap();
  // Stress all four tokens — the adversarial cancellation is per-token.
  let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
  let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
  let out = layer.forward(&ids).unwrap();
  // Final dtype must be f32 — mlx promotes f16 × f32 → f32 on the final
  // `(m[x]/denom)[..., None] * out` multiply, and there is no narrowing
  // astype pinning the return to y.dtype.
  assert_eq!(
    out.dtype().unwrap(),
    Dtype::F32,
    "forward must return the promoted dtype = f32 (f16 base × f32 adapter)"
  );
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  let ids_usize: Vec<usize> = (0..num_embeddings).collect();
  let want = forward_scalar_reference(
    &weight_f32,
    &lora_a_f32,
    &lora_b_f32,
    &m_f32,
    scale,
    &ids_usize,
    f16_rt,
    false, // uncast-z pipeline for the renorm.
  );
  // Promoted-dtype output. Tolerance still ~5e-3: the cancellation-fixture
  // f16 rounding of `y` (which enters `adapted = y + z`) dominates the
  // residual error; the dropped final-narrowing cast does not buy a
  // tighter fit because the f16 round-off was already absorbed upstream.
  // Keeping the original tolerance preserves the test's defect-detection
  // power against the upfront-cast bug.
  approx_eq(&got, &want, 5e-3);
}

/// `dora_embedding_forward_mixed_precision_matches_reference_bf16_base_f32_adapter`
/// — bf16 sibling of the f16 test. bf16 has only ~7 mantissa bits, so the
/// upfront-cast bug's per-element error is ~16× the f16 case; tolerance is
/// loosened accordingly (`5e-2` per-element).
#[test]
fn dora_embedding_forward_mixed_precision_matches_reference_bf16_base_f32_adapter() {
  let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
  let num_embeddings = weight_f32.len();
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
  let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
  let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
  let weight_bf16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
    .unwrap()
    .astype(Dtype::BF16)
    .unwrap();
  let base = BaseEmbedding::dense(weight_bf16).unwrap();
  let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
  let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, scale).unwrap();
  let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
  let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
  let out = layer.forward(&ids).unwrap();
  // Promoted dtype = f32 (bf16 × f32 → f32 under mlx promotion); the
  // narrowing astype was removed.
  assert_eq!(
    out.dtype().unwrap(),
    Dtype::F32,
    "forward must return the promoted dtype = f32 (bf16 base × f32 adapter)"
  );
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  let ids_usize: Vec<usize> = (0..num_embeddings).collect();
  let want = forward_scalar_reference(
    &weight_f32,
    &lora_a_f32,
    &lora_b_f32,
    &m_f32,
    scale,
    &ids_usize,
    bf16_rt,
    false,
  );
  // bf16 tolerance: looser, matching its narrower mantissa (the bf16
  // round-off on y dominates; same reasoning as the f16 sibling).
  approx_eq(&got, &want, 5e-2);
}

/// `dora_embedding_as_linear_mixed_precision_matches_reference_f16_base_f32_adapter`
/// — analogous mixed-precision test for `as_linear`: with f16 weight + f32
/// adapter, the global adapted-row norm must be computed at the UNCAST
/// delta (mlx-lm `tuner/dora.py:218`'s `weight + (scale·lora_a) @ lora_b`).
/// Casting delta to weight.dtype before the row-norm would diverge; the
/// uncast path doesn't. Returned dtype is f32 (mlx promotes f32·f16 — no
/// final astype, mlx-lm doesn't cast either).
#[test]
fn dora_embedding_as_linear_mixed_precision_matches_reference_f16_base_f32_adapter() {
  let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
  let num_embeddings = weight_f32.len();
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
  let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
  let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
  let weight_f16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let base = BaseEmbedding::dense(weight_f16).unwrap();
  let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
  let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, scale).unwrap();
  // x rows tuned so x @ weightᵀ varies across the batch; passed as f16 to
  // match the embedding base dtype (typical LM-head call site).
  let x_f32 = vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.5, -0.25, 0.75, -0.125]];
  let flat_x: Vec<f32> = x_f32.iter().flatten().copied().collect();
  let x_arr = Array::from_slice::<f32>(&flat_x, &(x_f32.len(), dims))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let out = layer.as_linear(&x_arr).unwrap();
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  let want = as_linear_scalar_reference(
    &weight_f32,
    &lora_a_f32,
    &lora_b_f32,
    &m_f32,
    scale,
    &x_f32,
    f16_rt,
    false,
  );
  approx_eq(&got, &want, 5e-3);
}

/// `dora_embedding_as_linear_mixed_precision_matches_reference_bf16_base_f32_adapter`
/// — bf16 sibling of `as_linear`'s mixed-precision test.
#[test]
fn dora_embedding_as_linear_mixed_precision_matches_reference_bf16_base_f32_adapter() {
  let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
  let num_embeddings = weight_f32.len();
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
  let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
  let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
  let weight_bf16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
    .unwrap()
    .astype(Dtype::BF16)
    .unwrap();
  let base = BaseEmbedding::dense(weight_bf16).unwrap();
  let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
  let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, scale).unwrap();
  let x_f32 = vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.5, -0.25, 0.75, -0.125]];
  let flat_x: Vec<f32> = x_f32.iter().flatten().copied().collect();
  let x_arr = Array::from_slice::<f32>(&flat_x, &(x_f32.len(), dims))
    .unwrap()
    .astype(Dtype::BF16)
    .unwrap();
  let out = layer.as_linear(&x_arr).unwrap();
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  let want = as_linear_scalar_reference(
    &weight_f32,
    &lora_a_f32,
    &lora_b_f32,
    &m_f32,
    scale,
    &x_f32,
    bf16_rt,
    false,
  );
  approx_eq(&got, &want, 5e-2);
}

/// `dora_embedding_forward_loses_precision_with_upfront_cast_regression_oracle`
/// — assert the (uncast-z, uncast-norm-scale, no-final-astype)
/// pipeline matches the f64 scalar reference WAY MORE TIGHTLY than an
/// upfront-cast pipeline would. Cancellation fixture: with f16 base +
/// f32 adapter and `y ≈ -z`, the f16 rounding of `z` perturbs ‖adapted‖
/// by a relative amount that flows through `m/denom` and ends up well
/// above the fp16 tolerance floor on the final output — so the uncast
/// code matches the scalar reference at ≤ `5e-3`, while comparing against
/// the upfront-cast reference mismatches by ≥ `1e-2` on at least one
/// element. Also asserts the promoted return dtype (f32) — direct guard
/// against both the upfront-cast and final-narrowing-astype regressions.
#[test]
fn dora_embedding_forward_loses_precision_with_upfront_cast_regression_oracle() {
  let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
  let num_embeddings = weight_f32.len();
  let dims = weight_f32[0].len();
  let r = lora_a_f32[0].len();
  let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
  let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
  let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
  let weight_f16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let base = BaseEmbedding::dense(weight_f16).unwrap();
  let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
  let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
  let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, scale).unwrap();
  let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
  let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
  let out = layer.forward(&ids).unwrap();
  // Guard: forward returns the promoted dtype (f32), not the
  // base's f16. Re-introducing the final `astype(y.dtype)` would flip this.
  assert_eq!(
    out.dtype().unwrap(),
    Dtype::F32,
    "regression-oracle: forward must return the promoted dtype = f32"
  );
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  let ids_usize: Vec<usize> = (0..num_embeddings).collect();
  let want_new = forward_scalar_reference(
    &weight_f32,
    &lora_a_f32,
    &lora_b_f32,
    &m_f32,
    scale,
    &ids_usize,
    f16_rt,
    false, // uncast pipeline reference
  );
  let want_old = forward_scalar_reference(
    &weight_f32,
    &lora_a_f32,
    &lora_b_f32,
    &m_f32,
    scale,
    &ids_usize,
    f16_rt,
    true, // upfront-cast pipeline reference
  );
  let new_max_err = got
    .iter()
    .zip(want_new.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
  let old_max_err = got
    .iter()
    .zip(want_old.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
  assert!(
    new_max_err <= 5e-3,
    "uncast pipeline must match scalar reference at fp16 tol; got max err {new_max_err}",
  );
  assert!(
    old_max_err >= 1e-2,
    "upfront-cast pipeline must mismatch the scalar reference noticeably; got max err {old_max_err} (cancellation fixture may need re-tuning)",
  );
  // Sanity gap: the uncast pipeline matches the reference at least 5×
  // tighter than the upfront-cast one — the dtype flow is the difference.
  assert!(
    new_max_err * 5.0 <= old_max_err,
    "regression-oracle expected ≥5× tighter uncast-vs-upfront-cast fit; got uncast={new_max_err}, upfront-cast={old_max_err}",
  );
}

/// `dora_embedding_forward_returns_promoted_dtype_for_mixed_precision` —
/// explicit, focused dtype guard for the promoted-return-dtype fix. Asserts
/// that `forward` returns the mlx-promoted dtype (f32) for both `f16 base × f32
/// adapter` and `bf16 base × f32 adapter`, NOT the embedding's narrow
/// dtype. mlx-lm `tuner/dora.py:208` returns `(self.m[x] / denom)[...,
/// None] * out` directly — no astype — and the port now mirrors that.
/// Re-introducing a final `astype(y.dtype)` would flip these assertions.
///
/// This test does NOT exercise value parity (the
/// `*_matches_reference_*_base_f32_adapter` tests do that); it is a pure
/// dtype contract test.
#[test]
fn dora_embedding_forward_returns_promoted_dtype_for_mixed_precision() {
  for (narrow, label) in [(Dtype::F16, "f16"), (Dtype::BF16, "bf16")] {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    let weight_narrow = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(narrow)
      .unwrap();
    let base = BaseEmbedding::dense(weight_narrow).unwrap();
    // Adapter factors + magnitude stay f32 — mlx will promote on the
    // final multiply.
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
    let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
    let out = layer.forward(&ids).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      Dtype::F32,
      "forward must return promoted dtype f32 for {label} base × f32 adapter (no final narrowing astype)",
    );
  }
}

/// `dora_embedding_forward_preserves_base_dtype_for_uniform_precision` —
/// sanity sibling of the mixed-precision dtype test: when base AND adapter
/// share a dtype, `forward` returns THAT dtype because no operand triggers
/// mlx's promotion-on-mix rule. Direct guard against a defensive "just to
/// be safe" re-introduction of the final astype — a forward that always
/// did `.astype(y.dtype)` would also pass this test (it's the no-op case),
/// but combined with `*_returns_promoted_dtype_for_mixed_precision`'s
/// "must be f32 for f16/bf16 base × f32 adapter" assertion, the pair
/// triangulates: a re-introduced final astype would pass THIS test and
/// fail THAT one, pinpointing the regression.
///
/// Covers `(f32, f32)`, `(f16, f16)`, and `(bf16, bf16)`. The half-precision
/// cases exercise [`scaled`]'s coercion: the scalar `self.scale` is
/// coerced to `arr.dtype()` (mirroring mlx-lm `to_array(v, a.dtype())`) so
/// `z = scale · (lora_a[x] @ lora_b)` stays in the adapter's narrow dtype
/// instead of promoting to f32. An f32 mlx scalar would silently upcast
/// uniform-half adapters to f32 — this triple-test pins the helper's
/// behavior across all three uniform precisions.
#[test]
fn dora_embedding_forward_preserves_base_dtype_for_uniform_precision() {
  for (uniform, label) in [
    (Dtype::F32, "f32"),
    (Dtype::F16, "f16"),
    (Dtype::BF16, "bf16"),
  ] {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    // All operands at the same dtype — no promotion at any step; `forward`
    // returns `uniform`.
    let weight = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
    let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
    let out = layer.forward(&ids).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      uniform,
      "forward must return {label} when base AND adapter are uniform {label} (no promotion)",
    );
  }
}

// ───────────────────── scaled() coercion ─────────────────────

/// `scaled_helper_coerces_scalar_to_array_dtype` — unit test on the
/// [`scaled`] helper: the scalar `scale` operand is cast to `arr`'s dtype
/// BEFORE the multiply, mirroring mlx-lm's `to_array(v, a.dtype())`
/// scalar-coercion (mlx-lm `lora.py:97`, `dora.py:200`).
///
/// If the helper created an f32 mlx scalar, `scaled(f16_arr, …)` would
/// silently return an f32 array (mlx promotes f16 × f32 → f32) —
/// silently diverging from mlx-lm for uniform-half adapters. This test
/// triangulates the coercion across all three float dtypes the helper
/// is expected to round-trip preserving precision.
#[test]
fn scaled_helper_coerces_scalar_to_array_dtype() {
  for (dt, label) in [
    (Dtype::F16, "f16"),
    (Dtype::BF16, "bf16"),
    (Dtype::F32, "f32"),
  ] {
    let arr = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,))
      .unwrap()
      .astype(dt)
      .unwrap();
    let out = scaled(&arr, 0.5).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      dt,
      "scaled must coerce the scalar to the array's dtype and keep the {label} result in {label}",
    );
  }
}

/// `dora_embedding_forward_uniform_f16_adapter_returns_f16` — the
/// `scaled` coercion: with a uniform-f16 base + adapter, `DoRAEmbedding::forward`
/// must return f16 (mlx-lm `to_array(scale, a.dtype())` keeps the scalar
/// at f16, so `z = scale · lora_a[x] @ lora_b` stays at f16 and no
/// downstream op promotes). If `scaled` minted an f32 scalar, the
/// final `out` would be silently f32 — divergent from mlx-lm.
///
/// Hand-constructed deterministic fixture (all values exact in f16/bf16):
/// num_embeddings=2, dims=2, r=1; weight=[[1,0],[0,1]], lora_a=[[1],[0]],
/// lora_b=[[1,0]], m=[1,1], scale=1.0. Per-token math:
/// - x=0: y=[1,0], z=1·[1,0]=[1,0], adapted=[2,0], ‖·‖=2, m/denom=0.5,
///   out_pre=[2,0], out=0.5·[2,0]=[1,0].
/// - x=1: y=[0,1], z=1·[0,0]=[0,0], adapted=[0,1], ‖·‖=1, m/denom=1,
///   out_pre=[0,1], out=1·[0,1]=[0,1].
///
/// Expected for ids=[0,1] is [[1,0],[0,1]] — exact in f16/bf16.
#[test]
fn dora_embedding_forward_uniform_f16_adapter_returns_f16() {
  dora_embedding_forward_uniform_dtype_case(Dtype::F16, "f16");
}

/// `dora_embedding_forward_uniform_bf16_adapter_returns_bf16` — bf16
/// sibling of the f16 uniform-dtype contract test. Same fixture (all
/// values exact in bf16); asserts dtype = bf16 and value parity.
#[test]
fn dora_embedding_forward_uniform_bf16_adapter_returns_bf16() {
  dora_embedding_forward_uniform_dtype_case(Dtype::BF16, "bf16");
}

/// Shared driver for the uniform-dtype `forward` dtype + value contract.
/// See [`dora_embedding_forward_uniform_f16_adapter_returns_f16`]'s docstring
/// for the hand-traced fixture math.
fn dora_embedding_forward_uniform_dtype_case(uniform: Dtype, label: &str) {
  let num_embeddings = 2usize;
  let dims = 2usize;
  let r = 1usize;
  let weight = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(num_embeddings, dims))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  let lora_a = Array::from_slice::<f32>(&[1.0, 0.0], &(num_embeddings, r))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let lora_b = Array::from_slice::<f32>(&[1.0, 0.0], &(r, dims))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let m = Array::from_slice::<f32>(&[1.0, 1.0], &(num_embeddings,))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, 1.0f32).unwrap();
  let ids = Array::from_slice::<i32>(&[0, 1], &(2usize,)).unwrap();
  let out = layer.forward(&ids).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    uniform,
    "forward must return {label} for uniform-{label} base + adapter (scaled() coerces scalar to arr.dtype)",
  );
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  // [[1, 0], [0, 1]] — exact in f16/bf16; zero tolerance would also pass,
  // but a tight 1e-3 leaves headroom against any future kernel-order shift.
  approx_eq(&got, &[1.0, 0.0, 0.0, 1.0], 1e-3);
}

/// `dora_embedding_as_linear_uniform_f16_adapter_returns_f16` — the
/// `scaled` coercion for `as_linear`: with uniform-f16 base + adapter, the tied-weight
/// LM-head forward must also return f16. The same `scaled` helper is on the
/// hot path (the scale·lora_a delta), so the coercion applies symmetrically.
///
/// Hand-constructed fixture (all values exact in f16/bf16) with x=[[1, 1]]:
/// - y = x @ weightᵀ = [1, 1]
/// - z = (x @ lora_bᵀ) @ lora_aᵀ = [1, 0]
/// - adapted = weight + scale · lora_a @ lora_b = [[2,0],[0,1]]
/// - denom (axis=1) = [2, 1], norm_scale = [0.5, 1]
/// - out_pre = y + scale·z = [2, 1], out = norm_scale · out_pre = [1, 1].
///
/// Expected = [[1, 1]] — exact in f16/bf16.
#[test]
fn dora_embedding_as_linear_uniform_f16_adapter_returns_f16() {
  dora_embedding_as_linear_uniform_dtype_case(Dtype::F16, "f16");
}

/// `dora_embedding_as_linear_uniform_bf16_adapter_returns_bf16` — bf16
/// sibling of the `as_linear` uniform-dtype contract test.
#[test]
fn dora_embedding_as_linear_uniform_bf16_adapter_returns_bf16() {
  dora_embedding_as_linear_uniform_dtype_case(Dtype::BF16, "bf16");
}

/// Shared driver for the uniform-dtype `as_linear` dtype + value contract.
/// See [`dora_embedding_as_linear_uniform_f16_adapter_returns_f16`]'s
/// docstring for the hand-traced fixture math.
fn dora_embedding_as_linear_uniform_dtype_case(uniform: Dtype, label: &str) {
  let num_embeddings = 2usize;
  let dims = 2usize;
  let r = 1usize;
  let weight = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(num_embeddings, dims))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let base = BaseEmbedding::dense(weight).unwrap();
  let lora_a = Array::from_slice::<f32>(&[1.0, 0.0], &(num_embeddings, r))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let lora_b = Array::from_slice::<f32>(&[1.0, 0.0], &(r, dims))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let m = Array::from_slice::<f32>(&[1.0, 1.0], &(num_embeddings,))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let params = AdapterParams {
    lora_a,
    lora_b,
    magnitude: Some(m),
  };
  let layer = DoRAEmbedding::new(base, params, 1.0f32).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 1.0], &(1usize, dims))
    .unwrap()
    .astype(uniform)
    .unwrap();
  let out = layer.as_linear(&x).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    uniform,
    "as_linear must return {label} for uniform-{label} base + adapter (scaled() coerces scalar to arr.dtype)",
  );
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  let got = out_f32.to_vec::<f32>().unwrap();
  approx_eq(&got, &[1.0, 1.0], 1e-3);
}

/// `dora_linear_forward_uniform_f16_adapter_returns_f16` — sibling
/// for [`DoRALinear`]: the same [`scaled`] helper is on its hot path, so
/// the coercion propagates. DoRALinear's `forward` has an explicit trailing
/// `astype(x.dtype)` on the low-rank term (mlx-lm `tuner/lora.py:97` casts
/// `(scale * z).astype(x.dtype)`), so the dtype contract here is "out
/// matches x.dtype" — the scaled() coercion doesn't change THAT contract for
/// DoRALinear (the trailing astype already enforces it), but the test
/// pins the contract as a regression oracle against a future refactor
/// that elides the trailing astype.
///
/// Hand-traced fixture (all values exact in f16): input_dims=3,
/// output_dims=2, r=2; reuses [`base_weight`], [`lora_a`], [`lora_b`]
/// (the LoRA `[3, 6]` hand-trace) with m chosen so renorm = identity
/// (m = ‖adapted‖₂ row-wise = [3, 3], same as
/// [`dora_linear_forward_hand_traced`]) — expected out = [3, 6].
#[test]
fn dora_linear_forward_uniform_f16_adapter_returns_f16() {
  let weight = base_weight().astype(Dtype::F16).unwrap();
  let la = lora_a().astype(Dtype::F16).unwrap();
  let lb = lora_b().astype(Dtype::F16).unwrap();
  let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let params = AdapterParams {
    lora_a: la,
    lora_b: lb,
    magnitude: Some(m),
  };
  let base = BaseLinear::dense(weight, None).unwrap();
  let layer = DoRALinear::new(base, params, 2.0).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let out = layer.forward(&x).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    Dtype::F16,
    "DoRALinear::forward must return f16 for uniform-f16 base + adapter (trailing astype + scaled() coercion both contribute)",
  );
  let mut out_f32 = out.astype(Dtype::F32).unwrap();
  approx_eq(&out_f32.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-3);
}

// ──── locate_adapter_safetensors symlink-following regression ────
//
// `adapter_candidate_present` must use `metadata()` (NOT
// `symlink_metadata()`) so that a broken preferred symlink falls through
// to the fallback candidate, and a symlink loop surfaces as a typed
// `Error::FileIo` rather than short-circuiting on the link object itself.
// Unix-gated because they use `std::os::unix::fs::symlink` directly.

#[cfg(unix)]
#[test]
fn locate_adapter_safetensors_falls_back_when_preferred_is_broken_symlink() {
  use std::os::unix::fs::symlink;
  let tmp = std::env::temp_dir().join(format!(
    "mlxrs_lora_broken_symlink_{}_{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos())
      .unwrap_or(0)
  ));
  let _ = std::fs::remove_dir_all(&tmp);
  std::fs::create_dir_all(&tmp).unwrap();

  // Broken preferred symlink: adapters.safetensors -> does_not_exist
  symlink(tmp.join("does_not_exist"), tmp.join(MLX_LM_ADAPTER_FILE)).unwrap();
  // Valid fallback regular file (contents irrelevant — locate only stats).
  std::fs::write(tmp.join(PEFT_ADAPTER_FILE), b"valid bytes").unwrap();

  // mlx-lm-native config => preferred = adapters.safetensors (broken
  // symlink), fallback = adapter_model.safetensors (valid). Locate must
  // return the fallback.
  let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let found = locate_adapter_safetensors(&tmp, &cfg)
    .expect("expected fallback to be located when preferred is a broken symlink");
  assert_eq!(found, tmp.join(PEFT_ADAPTER_FILE));

  let _ = std::fs::remove_dir_all(&tmp);
}

// ──── locate_adapter_safetensors non-regular path fail-fast ────
//
// Two structural-classification tests pin the **NonRegular → fail-fast**
// contract of [`adapter_candidate_present`] / [`probe_candidate`]: a
// directory (or FIFO / socket / …) sitting at either the preferred or
// fallback adapter weights path must surface as a typed `Error::FileIo`
// with `ErrorKind::InvalidInput` rather than silently being treated as
// "absent" and falling through. Combined with the broken-symlink and
// symlink-loop regressions above, the suite now exhaustively pins all four
// outcomes of the `CandidateProbe` classification (Absent / Present /
// NonRegular / IoError) at both preferred and fallback positions — any
// future change that re-introduces a silent collapse will be caught.

#[cfg(unix)]
#[test]
fn locate_adapter_safetensors_rejects_non_regular_preferred_path_even_with_valid_fallback() {
  let tmp = std::env::temp_dir().join(format!(
    "mlxrs_lora_nonreg_preferred_{}_{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos())
      .unwrap_or(0)
  ));
  let _ = std::fs::remove_dir_all(&tmp);
  std::fs::create_dir_all(&tmp).unwrap();

  // Preferred slot (mlx-lm-native ⇒ `adapters.safetensors`) is a
  // DIRECTORY — non-regular path the user clearly wanted as a file.
  std::fs::create_dir(tmp.join(MLX_LM_ADAPTER_FILE)).unwrap();
  // Valid fallback present: must NOT be silently used.
  std::fs::write(tmp.join(PEFT_ADAPTER_FILE), b"valid bytes").unwrap();

  let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let err = locate_adapter_safetensors(&tmp, &cfg)
    .expect_err("expected fail-fast Error::FileIo for non-regular preferred path");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.path(),
        tmp.join(MLX_LM_ADAPTER_FILE).as_path(),
        "path round-trips through FileIoPayload"
      );
      assert_eq!(
        p.op(),
        FileOp::Stat,
        "non-regular surfaces from the stat probe"
      );
      assert_eq!(
        p.inner().kind(),
        std::io::ErrorKind::InvalidInput,
        "non-regular candidates surface with InvalidInput",
      );
    }
    other => panic!("expected Error::FileIo for non-regular preferred path, got {other:?}"),
  }

  let _ = std::fs::remove_dir_all(&tmp);
}

#[cfg(unix)]
#[test]
fn locate_adapter_safetensors_rejects_non_regular_fallback_path() {
  let tmp = std::env::temp_dir().join(format!(
    "mlxrs_lora_nonreg_fallback_{}_{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos())
      .unwrap_or(0)
  ));
  let _ = std::fs::remove_dir_all(&tmp);
  std::fs::create_dir_all(&tmp).unwrap();

  // Preferred (mlx-lm-native ⇒ `adapters.safetensors`) is genuinely
  // absent. Fallback (`adapter_model.safetensors`) is a DIRECTORY.
  std::fs::create_dir(tmp.join(PEFT_ADAPTER_FILE)).unwrap();

  let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let err = locate_adapter_safetensors(&tmp, &cfg)
    .expect_err("expected fail-fast Error::FileIo for non-regular fallback path");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.path(),
        tmp.join(PEFT_ADAPTER_FILE).as_path(),
        "path round-trips through FileIoPayload"
      );
      assert_eq!(
        p.op(),
        FileOp::Stat,
        "non-regular surfaces from the stat probe"
      );
      assert_eq!(
        p.inner().kind(),
        std::io::ErrorKind::InvalidInput,
        "non-regular candidates surface with InvalidInput",
      );
    }
    other => panic!("expected Error::FileIo for non-regular fallback path, got {other:?}"),
  }

  let _ = std::fs::remove_dir_all(&tmp);
}

#[cfg(unix)]
#[test]
fn locate_adapter_safetensors_surfaces_symlink_loop_as_typed_file_io() {
  use std::os::unix::fs::symlink;
  let tmp = std::env::temp_dir().join(format!(
    "mlxrs_lora_symlink_loop_{}_{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos())
      .unwrap_or(0)
  ));
  let _ = std::fs::remove_dir_all(&tmp);
  std::fs::create_dir_all(&tmp).unwrap();

  // Self-referential loop: adapters.safetensors -> adapters.safetensors.
  // `metadata()` on this resolves via the symlink, hits ELOOP, and returns
  // an `io::Error` whose kind is `FilesystemLoop` (Linux) or similar
  // (macOS surfaces `Uncategorized` for ELOOP on some toolchain versions).
  // The contract we assert is: the helper returns `Err(Error::FileIo(...))`
  // (NOT `Ok(true)` / `Ok(false)`) with the candidate path + FileOp::Stat.
  let preferred = tmp.join(MLX_LM_ADAPTER_FILE);
  symlink(&preferred, &preferred).unwrap();

  let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
  let err = locate_adapter_safetensors(&tmp, &cfg)
    .expect_err("expected typed FileIo error for symlink loop");
  match err {
    Error::FileIo(p) => {
      assert_eq!(
        p.path(),
        preferred.as_path(),
        "path round-trips through FileIoPayload"
      );
      assert_eq!(
        p.op(),
        FileOp::Stat,
        "loop surfaces from the stat probe, not open"
      );
    }
    other => panic!("expected Error::FileIo for symlink loop, got {other:?}"),
  }

  let _ = std::fs::remove_dir_all(&tmp);
}

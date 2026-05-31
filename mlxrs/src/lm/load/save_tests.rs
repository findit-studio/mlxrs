//! Model save / shard / introspect, in isolation. Shard boundaries
//! are hand-computed for a chosen cap; `save` then `load_weights` round-
//! trips (weights byte-equal, index.json correct); introspection helpers
//! are checked against hand-verified counts. No `peak_memory()` assert.

use super::*;
use crate::lm::quant::{PerLayerQuantization, QuantMode, Quantization};

/// A fresh, writable per-test temp directory (the crate's
/// no-`tempfile`-crate convention — `temp_dir()` + pid + a process-unique
/// counter, mirroring `lm::factory`'s `fresh_dir`).
fn fresh_dir(tag: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs-lm-save-{tag}-{}-{n}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// An `f32` weight of `n` elements, shape `[n]` — `n * 4` bytes.
fn f32_weight(n: usize) -> Array {
  Array::from_slice::<f32>(&vec![0.0_f32; n], &(n,)).unwrap()
}

// ─────────────────────── array_nbytes ───────────────────────

#[test]
fn array_nbytes_is_count_times_dtype_size() {
  // f32 → 4 bytes/elem; 10 elems → 40 bytes.
  assert_eq!(array_nbytes(&f32_weight(10)).unwrap(), 40);
  // u8 → 1 byte/elem.
  let u8s = Array::from_slice::<u8>(&[1u8, 2, 3], &(3usize,)).unwrap();
  assert_eq!(array_nbytes(&u8s).unwrap(), 3);
  // u32 → 4 bytes/elem; a `[2, 8]` packed matrix → 16 elems → 64 bytes.
  let u32s = Array::from_slice::<u32>(&[0u32; 16], &(2usize, 8)).unwrap();
  assert_eq!(array_nbytes(&u32s).unwrap(), 64);
}

// ─────────────────────── make_shards ───────────────────────

/// All-fits case: four 100-byte weights (`a`..`d`, each 25 `f32` elems =
/// 100 bytes = 400 bytes total) under the default 5-GiB cap land on a
/// single shard — the cap is never reached, so the loop never flushes.
#[test]
fn make_shards_all_fits_one_shard() {
  let mut w: Weights = HashMap::new();
  for name in ["a", "b", "c", "d"] {
    w.insert(name.to_string(), f32_weight(25)); // 100 bytes each
  }
  let one = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
  assert_eq!(one.len(), 1, "4×100 bytes fits in one 5-GiB shard");
  assert_eq!(one[0].len(), 4);
}

/// Hand-traced zero-cap split, matching mlx-lm `make_shards`
/// (`utils.py:598-619`) EXACTLY — including the empty leading shard the
/// reference's guard-free `shard_size + v.nbytes > cap` produces. With a
/// `0`-GiB cap, `0 + nbytes > 0` holds for every non-empty weight, so the
/// split fires every iteration — *including the first*, while `shard` is
/// still empty. Hand-trace over sorted weights `a, b, c, d` (100 bytes
/// each), exactly as `utils.py`: at `a`, `0 + 100 > 0` pushes the empty
/// `{}` and resets, then `shard = {a}`; at `b`, `100 + 100 > 0` pushes
/// `{a}`, then `shard = {b}`; `c` pushes `{b}`, `shard = {c}`; `d` pushes
/// `{c}`, `shard = {d}`; after the loop the trailing `{d}` is pushed —
/// giving `[{}, {a}, {b}, {c}, {d}]`, 5 shards with an empty leading one.
/// (Run `mlx_lm.utils.make_shards({"a":…,"b":…,"c":…,"d":…}, 0)` to
/// confirm.)
#[test]
fn make_shards_zero_cap_empty_leading_then_one_weight_per_shard() {
  let mut w: Weights = HashMap::new();
  for name in ["a", "b", "c", "d"] {
    w.insert(name.to_string(), f32_weight(25));
  }
  let shards = make_shards(&w, 0).unwrap();
  // 5 shards: an empty leading shard + one per weight (mlx-lm parity).
  assert_eq!(shards.len(), 5);
  assert!(
    shards[0].is_empty(),
    "guard-free split flushes an empty leading shard"
  );
  // Sorted-key order in the trailing single-weight shards.
  assert!(shards[1].contains_key("a"));
  assert!(shards[2].contains_key("b"));
  assert!(shards[3].contains_key("c"));
  assert!(shards[4].contains_key("d"));
  assert!(shards[1..].iter().all(|s| s.len() == 1));
}

/// An over-cap **first** sorted tensor. mlx-lm `make_shards` has no
/// empty-shard guard, so when the first tensor already exceeds the cap
/// the split fires immediately, flushing the still-empty initial shard.
/// Hand-trace `make_shards({"big": 400-byte, "small": 4-byte}, cap=0)`
/// from `utils.py:611-618`: at `big`, `0 + 400 > 0` pushes the empty `{}`
/// and resets, then `shard = {big}`; at `small`, `400 + 4 > 0` pushes
/// `{big}` and resets, then `shard = {small}`; after the loop the
/// trailing `{small}` is pushed — giving `[{}, {big}, {small}]`: an empty
/// leading shard, then the over-cap tensor on its own shard, then the
/// remainder. This port must produce the identical sequence (same shard
/// filenames + index data).
#[test]
fn make_shards_over_cap_first_tensor_empty_leading_shard() {
  let mut w: Weights = HashMap::new();
  w.insert("big".to_string(), f32_weight(100)); // 400 bytes — over a 0 cap
  w.insert("small".to_string(), f32_weight(1)); // 4 bytes
  let shards = make_shards(&w, 0).unwrap();
  assert_eq!(
    shards.len(),
    3,
    "empty leading + over-cap tensor + remainder"
  );
  assert!(
    shards[0].is_empty(),
    "over-cap first tensor flushes an empty leading shard"
  );
  // Sorted key order: `big` < `small`.
  assert_eq!(shards[1].len(), 1);
  assert!(shards[1].contains_key("big"));
  assert_eq!(shards[2].len(), 1);
  assert!(shards[2].contains_key("small"));
}

/// An empty weight map still yields exactly one (empty) shard — mlx-lm's
/// `shards.append(shard)` after the loop (`utils.py:618`) always runs.
#[test]
fn make_shards_empty_map_yields_one_empty_shard() {
  let w: Weights = HashMap::new();
  let shards = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
  assert_eq!(shards.len(), 1);
  assert!(shards[0].is_empty());
}

/// A single weight under an ample cap — one shard holding it. (For the
/// `0`-cap / over-cap edge case, where a lone weight yields an empty
/// leading shard `[{}, {solo}]`, see
/// [`make_shards_over_cap_first_tensor_empty_leading_shard`].)
#[test]
fn make_shards_single_weight_one_shard() {
  let mut w: Weights = HashMap::new();
  w.insert("solo".to_string(), f32_weight(7));
  let shards = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
  assert_eq!(shards.len(), 1);
  assert_eq!(shards[0].len(), 1);
  assert!(shards[0].contains_key("solo"));
}

// ─────────────────────── get_total_parameters ───────────────────────

/// Dense model: every array contributes its plain element count
/// (`sum(v.size …)`). Two weights of 25 + 7 elems → 32 parameters.
#[test]
fn get_total_parameters_dense_sums_sizes() {
  let mut w: Weights = HashMap::new();
  w.insert("model.embed.weight".to_string(), f32_weight(25));
  w.insert("model.norm.weight".to_string(), f32_weight(7));
  let total = get_total_parameters(&w, &PerLayerQuantization::default()).unwrap();
  assert_eq!(total, 32);
}

/// Quantized affine layer: a `<path>.weight` (`uint32` packed) with a
/// `<path>.scales` sibling counts as `weight.size * 32 / bits` logical
/// params. Both affine-quantization metadata buffers — `<path>.scales`
/// AND `<path>.biases` (the zero-point array, NOT a real module bias) —
/// are NOT counted, matching mlx-lm `get_total_parameters`'s quantized
/// branch (`m.weight.size * 32 // m.bits` plus only a genuine `m.bias`,
/// `utils.py:203-204`). Hand-trace: packed `.weight` = 16 `u32` elems,
/// `bits = 4` → `16 * 32 / 4 = 128` logical weights; `.scales` (2 elems)
/// → +0; `.biases` (2 elems) → +0 (quantization metadata, skipped). Plus
/// a dense `model.norm.weight` of 7 → +7. Total = 128 + 7 = 135.
#[test]
fn get_total_parameters_quantized_unpacks_weight_skips_scales_and_biases() {
  let mut w: Weights = HashMap::new();
  let packed = Array::from_slice::<u32>(&[0u32; 16], &(2usize, 8)).unwrap();
  w.insert("model.layers.0.q_proj.weight".to_string(), packed);
  w.insert(
    "model.layers.0.q_proj.scales".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  w.insert(
    "model.layers.0.q_proj.biases".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  w.insert("model.norm.weight".to_string(), f32_weight(7));

  let quant = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let total = get_total_parameters(&w, &quant).unwrap();
  assert_eq!(total, 128 + 7);
}

/// A genuine module bias (`.bias`, singular, with NO `.scales` sibling)
/// is a real model parameter and IS counted — only an affine
/// quantization `.biases` (plural, sibling to a `.weight` + `.scales`
/// triple) is skipped as metadata. Hand-trace: dense `model.fc.weight`
/// of 5 → +5; `model.fc.bias` of 3 → +3 (no `model.fc.scales`, so it is
/// a plain parameter). Total = 8.
#[test]
fn get_total_parameters_counts_genuine_module_bias() {
  let mut w: Weights = HashMap::new();
  w.insert("model.fc.weight".to_string(), f32_weight(5));
  w.insert("model.fc.bias".to_string(), f32_weight(3));
  let total = get_total_parameters(&w, &PerLayerQuantization::default()).unwrap();
  assert_eq!(total, 8);
}

/// An orphan `.biases` (plural) with a `.weight` sibling but NO `.scales`
/// sibling is not a valid affine triple, so it must NOT be skipped — it
/// falls through to the dense count. (mlx-lm never produces this shape;
/// the skip is gated on BOTH a `.weight` and a `.scales` sibling so a
/// stray `.biases` is still accounted for.) Hand-trace: `model.x.weight`
/// of 4 → +4; `model.x.biases` of 2, no `model.x.scales` → +2. Total = 6.
#[test]
fn get_total_parameters_orphan_biases_without_scales_is_counted() {
  let mut w: Weights = HashMap::new();
  w.insert("model.x.weight".to_string(), f32_weight(4));
  w.insert("model.x.biases".to_string(), f32_weight(2));
  let total = get_total_parameters(&w, &PerLayerQuantization::default()).unwrap();
  assert_eq!(total, 6);
}

/// A quantized triple (`.scales` present) with no resolvable
/// [`Quantization`] for its layer is a configuration error.
#[test]
fn get_total_parameters_quantized_without_params_errors() {
  let mut w: Weights = HashMap::new();
  w.insert(
    "model.q.weight".to_string(),
    Array::from_slice::<u32>(&[0u32; 8], &(2usize, 4)).unwrap(),
  );
  w.insert(
    "model.q.scales".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  // No global default, no per-layer override → unresolvable. The error is a
  // typed [`Error::LayerKeyed`] wrapping an [`Error::InvariantViolation`]
  // naming the offending layer.
  let err = get_total_parameters(&w, &PerLayerQuantization::default());
  let Err(Error::LayerKeyed(p)) = err else {
    panic!("expected Error::LayerKeyed, got {err:?}");
  };
  assert_eq!(p.layer(), "model.q");
  assert!(
    matches!(p.inner(), Error::InvariantViolation(iv)
        if iv.context().contains("quantized layer") && iv.requirement().contains("resolvable")),
    "expected inner InvariantViolation about resolvable quantization params, got {:?}",
    p.inner()
  );
}

// ─────────────────────── compute_bits_per_weight ───────────────────────

/// `model_bytes * 8 / model_params`. A single dense `f32` weight of 10
/// elems: `model_bytes = 40`, `model_params = 10` → `40 * 8 / 10 = 32.0`
/// bits per weight (exactly f32's 32 bits — a dense float model).
#[test]
fn compute_bits_per_weight_dense_f32_is_32() {
  let mut w: Weights = HashMap::new();
  w.insert("model.w.weight".to_string(), f32_weight(10));
  let bpw = compute_bits_per_weight(&w, &PerLayerQuantization::default()).unwrap();
  assert!((bpw - 32.0).abs() < 1e-9, "expected 32.0, got {bpw}");
}

/// Quantized: `model_bytes` sums EVERY array (`scales`/`biases` too —
/// the reference's `tree_reduce` over `model`, `utils.py:211-213`), but
/// `model_params` is the *unpacked* count with the affine `scales` AND
/// `biases` excluded as metadata. Hand-trace: packed `.weight` 16 `u32`
/// = 64 bytes; `.scales` 2 `f32` = 8 bytes; `.biases` 2 `f32` = 8 bytes
/// → `model_bytes = 80`. `model_params = packed_weight.size * 32 / bits`
/// `= 16 * 32 / 4 = 128` (the affine `.scales` AND `.biases` are NOT in
/// the denominator — they are quantization metadata). `bpw = 80 * 8 /
/// 128 = 5.0`.
#[test]
fn compute_bits_per_weight_quantized_includes_scale_overhead() {
  let mut w: Weights = HashMap::new();
  w.insert(
    "model.q.weight".to_string(),
    Array::from_slice::<u32>(&[0u32; 16], &(2usize, 8)).unwrap(),
  );
  w.insert(
    "model.q.scales".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  w.insert(
    "model.q.biases".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  let quant = PerLayerQuantization::from_global(Quantization::affine(64, 4));
  let bpw = compute_bits_per_weight(&w, &quant).unwrap();
  // model_bytes * 8 / (packed_weight.size * 32 / bits)  — `.biases` is
  // no longer in the denominator.
  let expected = 80.0 * 8.0 / 128.0;
  assert!(
    (bpw - expected).abs() < 1e-9,
    "expected {expected}, got {bpw}"
  );
}

/// An empty weight map has zero parameters → a clean error, not a
/// divide-by-zero NaN.
#[test]
fn compute_bits_per_weight_zero_params_errors() {
  let w: Weights = HashMap::new();
  let err = compute_bits_per_weight(&w, &PerLayerQuantization::default());
  assert!(
    matches!(&err, Err(Error::EmptyInput(p))
        if p.context() == "compute_bits_per_weight: model parameters"),
    "expected Error::EmptyInput naming `model parameters`, got {err:?}"
  );
}

// ─────────────────── does_model_support_input_embeddings ───────────────

#[test]
fn does_model_support_input_embeddings_false_for_text_model() {
  // The text-only `MockModel` inherits the `false` default.
  let model = crate::lm::model::MockModel::new(4);
  assert!(!does_model_support_input_embeddings(&model));
}

// ─────────────────────── shard_file_name ───────────────────────

/// The generation-tagged basename: `model-gen-{gen_id}-{idx:05}-of-{N:05}
/// .safetensors`. Uniform across single- and multi-shard sets so the
/// publish path has one code path. The exact `gen_id` value is not load-
/// critical (the loader follows the index, not the basename), it is
/// just a uniqueness handle so new shards never collide with a prior
/// checkpoint's shard names.
#[test]
fn shard_file_name_generation_tagged() {
  let gen_id = "1234567890123-deadbeef-00000000cafef00d";
  assert_eq!(
    shard_file_name(gen_id, 1, 1),
    format!("model-gen-{gen_id}-00001-of-00001.safetensors")
  );
  assert_eq!(
    shard_file_name(gen_id, 1, 3),
    format!("model-gen-{gen_id}-00001-of-00003.safetensors")
  );
  assert_eq!(
    shard_file_name(gen_id, 3, 3),
    format!("model-gen-{gen_id}-00003-of-00003.safetensors")
  );
  // Two distinct generation ids produce distinct basenames — the
  // property that lets new-checkpoint shards never overwrite old-
  // checkpoint shards on disk.
  assert_ne!(
    shard_file_name("first-gen-id", 1, 1),
    shard_file_name("second-gen-id", 1, 1),
    "different generation ids must produce different shard names"
  );
}

/// [`new_gen_id`] returns the expected `{ts_us:013}-{pid:08x}-{ctr:016x}`
/// shape (the `:013` is a MINIMUM width pad — a 2026-and-later µs
/// timestamp is naturally 16 digits and is left unpadded by the
/// format spec), the counter advances each call (so two saves from
/// the same process can never share a `gen_id`), and the PID + ctr
/// widths stay constant.
#[test]
fn new_gen_id_shape_and_counter_advance() {
  let a = new_gen_id();
  let b = new_gen_id();
  // Two calls produce two distinct ids (the counter component differs).
  assert_ne!(a, b, "successive new_gen_id() calls must differ");
  // Shape: three `-`-separated components.
  for id in [&a, &b] {
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(
      parts.len(),
      3,
      "gen_id has 3 dash-separated components: {id}"
    );
    // ts_us is decimal digits, minimum 13 wide (the format-spec pad;
    // a real 2026-and-later µs-since-epoch is 16 digits naturally).
    assert!(
      parts[0].len() >= 13,
      "ts_us is at least 13 chars wide (the format-spec pad): {}",
      parts[0]
    );
    assert!(
      parts[0].chars().all(|c| c.is_ascii_digit()),
      "ts_us is decimal: {}",
      parts[0]
    );
    assert_eq!(parts[1].len(), 8, "pid is 8 hex chars: {}", parts[1]);
    assert!(
      parts[1].chars().all(|c| c.is_ascii_hexdigit()),
      "pid is hex: {}",
      parts[1]
    );
    assert_eq!(parts[2].len(), 16, "ctr is 16 hex chars: {}", parts[2]);
    assert!(
      parts[2].chars().all(|c| c.is_ascii_hexdigit()),
      "ctr is hex: {}",
      parts[2]
    );
  }
  // The pid component is identical across two calls in the same
  // process — only the counter (and possibly the timestamp) advances.
  let a_parts: Vec<&str> = a.split('-').collect();
  let b_parts: Vec<&str> = b.split('-').collect();
  assert_eq!(
    a_parts[1], b_parts[1],
    "PID stable across calls in the same process"
  );
}

// ─────────────────────── save_model round-trip ───────────────────────

/// `save_model` writes a single generation-tagged shard (the 3 small
/// weights fit one 5-GiB shard) plus a `model.safetensors.index.json`;
/// [`load_weights`] reads the weights back byte-equal, and the index JSON
/// has the expected `metadata` + sorted `weight_map`.
#[test]
fn save_model_single_shard_round_trips() {
  let dir = fresh_dir("save-model-single");
  let mut w: Weights = HashMap::new();
  // Distinct values so byte-equality is meaningful.
  w.insert(
    "model.b.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
  );
  w.insert(
    "model.a.weight".to_string(),
    Array::from_slice::<f32>(&[4.0, 5.0], &(2usize,)).unwrap(),
  );

  save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();

  // Exactly one shard file, named with the generation-tagged
  // `…-00001-of-00001` form (uniform single- + multi-shard naming).
  let shards = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with("-00001-of-00001.safetensors")
  })
  .unwrap();
  assert_eq!(
    shards.len(),
    1,
    "exactly one generation-tagged single shard file"
  );
  assert!(dir.join("model.safetensors.index.json").is_file());

  // Weights round-trip byte-equal via the index.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2);
  assert_eq!(
    loaded
      .get_mut("model.a.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![4.0, 5.0]
  );
  assert_eq!(
    loaded
      .get_mut("model.b.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0]
  );

  // index.json: metadata + sorted weight_map.
  let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
  let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
  // total_size = (2 + 3) elems × 4 bytes = 20.
  assert_eq!(index["metadata"]["total_size"], 20);
  // dense → total_parameters = 2 + 3 = 5.
  assert_eq!(index["metadata"]["total_parameters"], 5);
  let wm = index["weight_map"].as_object().unwrap();
  assert_eq!(wm.len(), 2);
  // Both weights are in the same single shard. The shard basename in the
  // index matches the on-disk file.
  let shard_basename = shards[0]
    .file_name()
    .unwrap()
    .to_string_lossy()
    .into_owned();
  assert_eq!(wm["model.a.weight"], shard_basename);
  assert_eq!(wm["model.b.weight"], shard_basename);
  // weight_map keys are sorted (a before b).
  let keys: Vec<&String> = wm.keys().collect();
  assert_eq!(keys, vec!["model.a.weight", "model.b.weight"]);

  // 4-space indent — Python `json.dump(indent=4)` parity.
  assert!(index_text.contains("\n    \"metadata\""));
  assert!(
    !index_text.ends_with('\n'),
    "json.dump writes no trailing newline"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// `make_shards` borrows, never clones: each [`Shard`] entry points at the
/// very same `Array` object in the input `weights` map. Verified by
/// pointer identity (the shard's `&Array` is the input map's `&Array`).
#[test]
fn make_shards_borrows_without_cloning() {
  let mut w: Weights = HashMap::new();
  w.insert("x".to_string(), f32_weight(3));
  let shards = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
  assert_eq!(shards.len(), 1);
  let shard_ref: &Array = shards[0]["x"];
  let map_ref: &Array = w.get("x").unwrap();
  assert!(
    std::ptr::eq(shard_ref, map_ref),
    "make_shards must borrow the input array, not clone it"
  );
}

/// Multi-shard path: a `0`-GiB cap can't be passed to `save_model`
/// (it hard-codes [`MAX_FILE_SIZE_GB`]), so this exercises the multi-shard
/// *file naming + index* through `shard_file_name` +
/// [`crate::io::save_safetensors_view`] directly, then confirms a
/// hand-built 2-shard layout — published with its `weight_map` index —
/// reloads via [`load_weights`] (index-honoring path). Asserts the
/// generation-tagged naming scheme (`model-gen-{ts}-{idx:05}-of-{N:05}
/// .safetensors`) at the basename level + that the on-disk files exactly
/// match the index's `weight_map` values.
#[test]
fn save_model_multi_shard_naming_and_index_reload() {
  let dir = fresh_dir("save-model-multi");
  // Two weights; write them as a 2-shard layout by hand using the same
  // primitives `save_model` uses, to exercise the multi-shard names.
  let w0 = Array::from_slice::<f32>(&[10.0], &(1usize,)).unwrap();
  let w1 = Array::from_slice::<f32>(&[20.0, 21.0], &(2usize,)).unwrap();
  let shards: Vec<Shard<'_>> = vec![BTreeMap::from([("w0", &w0)]), BTreeMap::from([("w1", &w1)])];
  let count = shards.len();
  // Single generation id for the whole save — exactly what
  // `save_model` does internally (here a hand-crafted fixed value so
  // the asserted basenames are deterministic; production uses
  // `new_gen_id()`).
  let gen_id = "1234567890123-deadbeef-00000000cafef00d";
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
  let mut written_basenames: Vec<String> = Vec::new();
  for (i, s) in shards.iter().enumerate() {
    let name = shard_file_name(gen_id, i + 1, count);
    // Generation-tagged scheme + zero-padded indices.
    assert_eq!(
      name,
      format!(
        "model-gen-{gen_id}-{:05}-of-{:05}.safetensors",
        i + 1,
        count
      )
    );
    crate::io::save_safetensors_view(&dir.join(&name), s.iter().map(|(&k, &v)| (k, v)), &meta)
      .unwrap();
    for &k in s.keys() {
      weight_map.insert(k.to_string(), name.clone());
    }
    written_basenames.push(name);
  }
  // The index makes the shard set discoverable by the index-honoring
  // [`load_weights`] path (without it, an absent `model.safetensors` /
  // `weights.safetensors` / `*.gguf` would error — and the bare-glob
  // resurrection of pre-index code is intentionally gone).
  write_json_pretty_to_path(
    &dir.join("model.safetensors.index.json"),
    &serde_json::json!({
      "metadata": { "total_size": 12, "total_parameters": 3 },
      "weight_map": weight_map,
    }),
    "test: 2-shard index",
  )
  .unwrap();

  // Indices listed in the JSON exactly match the on-disk shard files
  // (no orphan shards on disk, no dangling index references).
  let on_disk: std::collections::BTreeSet<String> = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap()
  .into_iter()
  .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
  .collect();
  let indexed: std::collections::BTreeSet<String> = weight_map.values().cloned().collect();
  assert_eq!(
    on_disk, indexed,
    "index `weight_map` values must exactly match the on-disk shard set"
  );
  let expected: std::collections::BTreeSet<String> = written_basenames.into_iter().collect();
  assert_eq!(
    indexed, expected,
    "index lists every generation-tagged shard we wrote, no more, no less"
  );

  // Both shard files reload + merge via the index-honoring `load_weights`.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2);
  assert_eq!(
    loaded.get_mut("w0").unwrap().to_vec::<f32>().unwrap(),
    vec![10.0]
  );
  assert_eq!(
    loaded.get_mut("w1").unwrap().to_vec::<f32>().unwrap(),
    vec![20.0, 21.0]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── save_config ───────────────────────

/// `save_config` drops `_name_or_path` / `vision_config`, mirrors
/// `quantization` into `quantization_config`, sorts the keys, and writes
/// 4-space-indented JSON with no trailing newline.
#[test]
fn save_config_cleans_mirrors_and_sorts() {
  let dir = fresh_dir("save-config");
  let path = dir.join("config.json");
  let src = r#"{
      "model_type": "qwen3",
      "_name_or_path": "/tmp/should-be-dropped",
      "vision_config": {"drop": "me"},
      "hidden_size": 64,
      "quantization": {"group_size": 64, "bits": 4}
    }"#;
  save_config(src, &path).unwrap();

  let text = std::fs::read_to_string(&path).unwrap();
  let v: serde_json::Value = serde_json::from_str(&text).unwrap();
  let obj = v.as_object().unwrap();

  // Dropped keys.
  assert!(!obj.contains_key("_name_or_path"));
  assert!(!obj.contains_key("vision_config"));
  // `quantization` preserved AND mirrored to `quantization_config`.
  assert_eq!(obj["quantization"]["bits"], 4);
  assert_eq!(obj["quantization_config"]["bits"], 4);
  assert_eq!(obj["quantization_config"]["group_size"], 64);
  // Surviving content keys.
  assert_eq!(obj["model_type"], "qwen3");
  assert_eq!(obj["hidden_size"], 64);
  // Keys sorted ascending.
  let keys: Vec<&String> = obj.keys().collect();
  let mut sorted = keys.clone();
  sorted.sort();
  assert_eq!(keys, sorted, "config.json keys must be sorted");

  // 4-space indent, no trailing newline.
  assert!(text.contains("\n    \""));
  assert!(!text.ends_with('\n'));
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_config_rejects_non_object_json() {
  let dir = fresh_dir("save-config-bad");
  // A valid-but-non-object JSON (array) → `InvariantViolation` naming the
  // "must be an object" requirement; never reaches the JSON parser as an
  // error.
  let err = save_config("[1, 2, 3]", &dir.join("config.json"));
  assert!(
    matches!(&err, Err(Error::InvariantViolation(iv))
        if iv.context() == "save_config: config JSON" && iv.requirement() == "must be an object"),
    "expected Error::InvariantViolation for non-object JSON, got {err:?}"
  );
  // A non-JSON body → `Parse` from the serde_json failure.
  let err2 = save_config("not json at all", &dir.join("config.json"));
  assert!(
    matches!(&err2, Err(Error::Parse(p))
        if p.context() == "save_config: config" && p.input_kind() == "JSON"),
    "expected Error::Parse for non-JSON body, got {err2:?}"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── save (driver) ───────────────────────

/// The `save` driver writes both the sharded weights+index and the
/// cleaned `config.json`; the weights reload byte-equal and the config
/// is the cleaned/sorted form.
#[test]
fn save_driver_writes_weights_and_config() {
  let dir = fresh_dir("save-driver");
  let mut w: Weights = HashMap::new();
  w.insert(
    "model.embed_tokens.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4usize,)).unwrap(),
  );
  let config = r#"{"model_type": "qwen3", "_name_or_path": "drop", "hidden_size": 8}"#;

  save(&dir, &w, config, &PerLayerQuantization::default()).unwrap();

  // Weights side: a single generation-tagged shard plus the index.
  let shards = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap();
  assert_eq!(
    shards.len(),
    1,
    "the save driver produced exactly one generation-tagged shard"
  );
  assert!(dir.join("model.safetensors.index.json").is_file());
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(
    loaded
      .get_mut("model.embed_tokens.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0, 4.0]
  );

  // Config side: `_name_or_path` dropped, keys sorted.
  let cfg_text = std::fs::read_to_string(dir.join("config.json")).unwrap();
  let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
  assert!(!cfg.as_object().unwrap().contains_key("_name_or_path"));
  assert_eq!(cfg["model_type"], "qwen3");
  assert_eq!(cfg["hidden_size"], 8);

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── save_model overwrite semantics ───────────────────────

/// Overwriting a pre-existing checkpoint with the structurally-different
/// generation-tagged naming: the loader follows the NEW index, so only
/// the new weights are visible. Stale-shard files from the OLD
/// checkpoint may remain on disk as orphans — they are deliberately
/// invisible to load (the index is the authoritative manifest) and
/// they are NOT inline-cleaned by `save_model` (the inline cleanup was
/// removed because it raced concurrent readers; see
/// `save_model` rustdoc). This test asserts the *load* contract — only
/// the new keys appear — while letting orphan shards exist on disk;
/// `save_model_no_overwrite_of_old_shards` covers the on-disk side.
#[test]
fn save_model_overwrite_loads_only_new_weights() {
  let dir = fresh_dir("save-model-overwrite-loads-new");

  // Stale 3-shard checkpoint, hand-written with the OLD reference-
  // style multi-shard names (the form an earlier build, or any
  // hand-crafted checkpoint, could leave behind).
  let stale_vals = [
    ("stale.a.weight", vec![100.0_f32]),
    ("stale.b.weight", vec![200.0_f32, 201.0]),
    ("stale.c.weight", vec![300.0_f32, 301.0, 302.0]),
  ];
  let stale_arrays: Vec<(&str, Array)> = stale_vals
    .iter()
    .map(|(k, v)| (*k, Array::from_slice::<f32>(v, &(v.len(),)).unwrap()))
    .collect();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  let stale_count = stale_arrays.len();
  let mut stale_map: BTreeMap<String, String> = BTreeMap::new();
  for (i, (k, arr)) in stale_arrays.iter().enumerate() {
    let name = format!("model-{:05}-of-{:05}.safetensors", i + 1, stale_count);
    crate::io::save_safetensors_view(&dir.join(&name), std::iter::once((*k, arr)), &meta).unwrap();
    stale_map.insert((*k).to_string(), name);
  }
  // A stale index too — the new save's index rename overwrites it.
  write_json_pretty_to_path(
    &dir.join("model.safetensors.index.json"),
    &serde_json::json!({
      "metadata": { "total_size": 24, "total_parameters": 6 },
      "weight_map": stale_map,
    }),
    "test: stale index",
  )
  .unwrap();

  // Overwrite with a smaller single-shard checkpoint.
  let mut new_w: Weights = HashMap::new();
  new_w.insert(
    "fresh.x.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap(),
  );
  new_w.insert(
    "fresh.y.weight".to_string(),
    Array::from_slice::<f32>(&[3.0], &(1usize,)).unwrap(),
  );
  save_model(&dir, &new_w, &PerLayerQuantization::default()).unwrap();

  // `load_weights` sees ONLY the new checkpoint's keys — the stale
  // shards on disk are invisible because the new index does not list
  // them.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2, "only the two new weights load back");
  assert!(loaded.contains_key("fresh.x.weight"));
  assert!(loaded.contains_key("fresh.y.weight"));
  assert!(!loaded.contains_key("stale.a.weight"));
  assert!(!loaded.contains_key("stale.b.weight"));
  assert!(!loaded.contains_key("stale.c.weight"));
  assert_eq!(
    loaded
      .get_mut("fresh.x.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0]
  );

  // The index `weight_map` lists only the new keys; their values
  // reference exactly one generation-tagged shard.
  let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
  let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
  let wm = index["weight_map"].as_object().unwrap();
  assert_eq!(wm.len(), 2);
  let shard_x = wm["fresh.x.weight"].as_str().unwrap();
  let shard_y = wm["fresh.y.weight"].as_str().unwrap();
  assert_eq!(
    shard_x, shard_y,
    "both new weights land in the same single shard"
  );
  assert!(
    shard_x.starts_with("model-gen-") && shard_x.ends_with("-00001-of-00001.safetensors"),
    "new shard is generation-tagged: got {shard_x}"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// Re-saving the *same* checkpoint to a directory is a structurally
/// safe operation: each save publishes its own generation-tagged
/// shard, and the loader follows the latest index. The test asserts
/// the load contract is stable across two consecutive saves.
#[test]
fn save_model_resave_same_checkpoint_is_stable() {
  let dir = fresh_dir("save-model-resave");
  let mut w: Weights = HashMap::new();
  w.insert("m.w.weight".to_string(), f32_weight(4));

  save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();
  save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();

  // Each save writes its own generation-tagged shard; the loader sees
  // exactly the latest one (one entry in the index, one weight loaded).
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert!(loaded.contains_key("m.w.weight"));
  assert_eq!(
    loaded
      .get_mut("m.w.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![0.0, 0.0, 0.0, 0.0]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Generation-unique shard names mean a NEW save can never overwrite an
/// OLD save's shard files on disk: after two consecutive saves to the
/// same directory, BOTH saves' shard files coexist on disk, but only
/// the SECOND save's shards are listed in the current index, and the
/// loader returns exactly the second save's weights. This is the load-
/// time guarantee the inline-cleanup removal trades for: prior-
/// generation shards leak disk space but never corrupt the
/// previously-valid checkpoint.
#[test]
fn save_model_no_overwrite_of_old_shards() {
  let dir = fresh_dir("save-no-overwrite");

  // FIRST save: a single weight whose value is byte-distinct from the
  // second save's, so a confused load would surface obviously.
  let mut first: Weights = HashMap::new();
  first.insert(
    "w.first.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
  );
  save_model(&dir, &first, &PerLayerQuantization::default()).unwrap();
  let first_shards: Vec<String> = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap()
  .into_iter()
  .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
  .collect();
  assert_eq!(first_shards.len(), 1, "first save writes one shard");

  // Sleep so the millisecond timestamps of the two saves cannot
  // coincide (a 1-ms tick is enough; we add a small margin for
  // coarser-clock CI).
  std::thread::sleep(std::time::Duration::from_millis(5));

  // SECOND save: a different weight name + value.
  let mut second: Weights = HashMap::new();
  second.insert(
    "w.second.weight".to_string(),
    Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap(),
  );
  save_model(&dir, &second, &PerLayerQuantization::default()).unwrap();
  let all_shards: Vec<String> = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap()
  .into_iter()
  .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
  .collect();

  // (1) Both saves' shard files coexist on disk — the second save did
  // NOT inline-clean the first save's shard (no overwrite was possible
  // because the basenames carry different generation timestamps).
  assert_eq!(
    all_shards.len(),
    2,
    "both saves' shard files coexist on disk (no inline cleanup); got {all_shards:?}"
  );
  for s in &first_shards {
    assert!(
      all_shards.contains(s),
      "the first save's shard {s} must survive the second save"
    );
  }

  // (2) Only the SECOND save's shards are listed in the current
  // index — the orphan first-save shards are invisible to load.
  let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
  let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
  let wm = index["weight_map"].as_object().unwrap();
  assert_eq!(wm.len(), 1, "second save's index lists one weight");
  let indexed: std::collections::BTreeSet<String> = wm
    .values()
    .filter_map(|v| v.as_str().map(|s| s.to_string()))
    .collect();
  assert_eq!(
    indexed.len(),
    1,
    "all keys in the new index reference exactly one shard"
  );
  let indexed_shard = indexed.iter().next().unwrap().clone();
  assert!(
    !first_shards.contains(&indexed_shard),
    "the second save's index must not reference the first save's shard"
  );

  // (3) The loader returns exactly the SECOND save's weights via the
  // new index — no resurrected first-save tensors.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 1, "load sees only the new checkpoint");
  assert!(
    loaded.contains_key("w.second.weight"),
    "the second save's weight loads"
  );
  assert!(
    !loaded.contains_key("w.first.weight"),
    "the first save's weight is invisible to load (orphan on disk only)"
  );
  assert_eq!(
    loaded
      .get_mut("w.second.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![10.0, 20.0]
  );

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────── save_model failure-atomicity ───────────────────

/// Failure-atomic save: when a `save_model` overwrite
/// FAILS partway, the previously-valid checkpoint in the directory is
/// left **fully intact and loadable**, and no partial `.tmp.safetensors`
/// remains. A direct (non-atomic) per-shard write would clobber a
/// still-valid shard *before* the new checkpoint is durable, leaving the
/// directory neither the old checkpoint nor the new one.
///
/// Failure is injected by making the checkpoint directory read-only so
/// the next save's shard-tempfile `create_new` fails (mirrors
/// `cache_prompt`'s read-only-dir injection). POSIX-only (`unix`): the
/// permission bits are the failure lever.
#[cfg(unix)]
#[test]
fn save_model_failed_save_keeps_previous_checkpoint_intact() {
  use std::os::unix::fs::PermissionsExt;

  let dir = fresh_dir("save-model-failed-intact");

  // 1. Write a good single-shard checkpoint.
  let mut orig: Weights = HashMap::new();
  orig.insert(
    "orig.a.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
  );
  orig.insert(
    "orig.b.weight".to_string(),
    Array::from_slice::<f32>(&[4.0, 5.0], &(2usize,)).unwrap(),
  );
  save_model(&dir, &orig, &PerLayerQuantization::default()).unwrap();
  // The original generation-tagged shard set (snapshotted before the
  // failed save so we can assert it survives byte-identical).
  let orig_shards: Vec<std::path::PathBuf> = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap();
  assert!(
    !orig_shards.is_empty(),
    "the original save produced at least one generation-tagged shard"
  );
  let orig_shard_bytes: BTreeMap<std::path::PathBuf, Vec<u8>> = orig_shards
    .iter()
    .map(|p| (p.clone(), std::fs::read(p).unwrap()))
    .collect();
  let orig_index = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();

  // 2. Make the directory read-only so the next save's tempfile
  //    `create_new` fails. (Root could bypass this; CI/dev users are not.)
  let mut perms = std::fs::metadata(&dir).unwrap().permissions();
  let orig_mode = perms.mode();
  perms.set_mode(0o500); // r-x------ : no write ⇒ create_new fails
  std::fs::set_permissions(&dir, perms).unwrap();

  // 3. Attempt to overwrite with a different checkpoint — must fail.
  let mut replacement: Weights = HashMap::new();
  replacement.insert("SHOULD.NOT.WIN.weight".to_string(), f32_weight(7));
  let r = save_model(&dir, &replacement, &PerLayerQuantization::default());

  // Restore write perms BEFORE asserting so cleanup + reads work even if
  // an assert fails.
  let mut restore = std::fs::metadata(&dir).unwrap().permissions();
  restore.set_mode(orig_mode);
  std::fs::set_permissions(&dir, restore).unwrap();

  assert!(r.is_err(), "a save into a read-only dir must fail");

  // 4. The original checkpoint is untouched: same shard set + index, it
  //    still `load_weights`-loads byte-equal, and no leftover tempfile.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2, "only the original two weights load back");
  assert!(loaded.contains_key("orig.a.weight"));
  assert!(loaded.contains_key("orig.b.weight"));
  assert!(
    !loaded.contains_key("SHOULD.NOT.WIN.weight"),
    "the failed save's weight must not have leaked in"
  );
  assert_eq!(
    loaded
      .get_mut("orig.a.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0]
  );
  assert_eq!(
    loaded
      .get_mut("orig.b.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![4.0, 5.0]
  );
  assert_eq!(
    std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap(),
    orig_index,
    "the original index.json must survive the failed save unchanged"
  );
  // Every original generation-tagged shard file is still on disk and
  // byte-identical to its pre-failed-save state.
  for (path, bytes) in &orig_shard_bytes {
    assert!(
      path.is_file(),
      "original shard {} must survive the failed save",
      path.display()
    );
    assert_eq!(
      &std::fs::read(path).unwrap(),
      bytes,
      "original shard {} must be byte-identical after the failed save",
      path.display()
    );
  }
  let leftover_tmp = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(
    !leftover_tmp,
    "no partial tempfile may remain after a failed save"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// Failure-atomic save, rename-failure branch: when the final atomic
/// `rename` of the **index** fails (here the index path pre-exists as a
/// **directory**, which `fs::rename(file -> dir)` rejects), every staged
/// `.tmp.safetensors` is cleaned up — no leftover tempfile. Note that
/// the shard renames *do* succeed (their basenames are generation-
/// tagged and never collide with any pre-existing file), so this test
/// exercises specifically the index-rename failure path; the renamed
/// shards become orphan files (deliberately not inline-cleaned).
#[test]
fn save_model_failed_save_rename_failure_cleans_up_tempfiles() {
  let dir = fresh_dir("save-model-failed-rename");
  // Pre-create the INDEX path as a directory so the final
  // `rename(file -> dir)` of the staged index fails.
  std::fs::create_dir_all(dir.join("model.safetensors.index.json")).unwrap();

  let mut w: Weights = HashMap::new();
  w.insert("m.w.weight".to_string(), f32_weight(4));
  let r = save_model(&dir, &w, &PerLayerQuantization::default());
  assert!(
    r.is_err(),
    "rename of the index onto an existing directory must fail"
  );

  // The colliding directory at the index path is untouched.
  assert!(
    dir.join("model.safetensors.index.json").is_dir(),
    "the colliding directory at the index path must be left untouched"
  );
  // No `.tmp.safetensors` leftover — every staged tempfile was removed
  // on the rename-failure cleanup path.
  let leftover_tmp = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(
    !leftover_tmp,
    "every staged tempfile must be removed when a rename fails"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// `save_config` is failure-atomic too: a FAILED config write leaves a
/// previously-valid `config.json` fully intact and removes the tempfile.
/// Failure is injected with a read-only directory (POSIX-only).
#[cfg(unix)]
#[test]
fn save_config_failed_write_keeps_previous_config_intact() {
  use std::os::unix::fs::PermissionsExt;

  let dir = fresh_dir("save-config-failed-intact");
  let config_path = dir.join("config.json");

  // 1. Write a good config.
  save_config(r#"{"model_type": "good", "hidden_size": 8}"#, &config_path).unwrap();
  let orig = std::fs::read_to_string(&config_path).unwrap();

  // 2. Make the directory read-only so the next write's tempfile
  //    `create_new` fails.
  let mut perms = std::fs::metadata(&dir).unwrap().permissions();
  let orig_mode = perms.mode();
  perms.set_mode(0o500);
  std::fs::set_permissions(&dir, perms).unwrap();

  // 3. Attempt to overwrite — must fail.
  let r = save_config(r#"{"model_type": "SHOULD-NOT-WIN"}"#, &config_path);

  let mut restore = std::fs::metadata(&dir).unwrap().permissions();
  restore.set_mode(orig_mode);
  std::fs::set_permissions(&dir, restore).unwrap();

  assert!(r.is_err(), "a config write into a read-only dir must fail");

  // 4. The original config is byte-identical, no leftover tempfile.
  assert_eq!(
    std::fs::read_to_string(&config_path).unwrap(),
    orig,
    "the original config.json must survive the failed write unchanged"
  );
  let leftover_tmp = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(
    !leftover_tmp,
    "no partial tempfile may remain after a failed config write"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

// ────────── load_weights: index-honoring + fallback resolution ──────────

/// `load_weights` only loads shards listed in
/// `model.safetensors.index.json` — a stale `model-*.safetensors` left
/// on disk that is NOT in the index is invisible (the structural fix
/// that makes the [`save_model`] index-rename single-commit-point safe).
/// Hand-wires a single `model.safetensors` published via the
/// `weight_map`, plus a stale `model-00099-of-00099.safetensors` carrying
/// an extra weight; `load_weights` must return ONLY the indexed weight.
#[test]
fn load_weights_ignores_stale_shards_not_in_index() {
  let dir = fresh_dir("load-ignores-stale");
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());

  // The "real" indexed shard — a single `model.safetensors`.
  let real = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap();
  crate::io::save_safetensors_view(
    &dir.join("model.safetensors"),
    std::iter::once(("real.weight", &real)),
    &meta,
  )
  .unwrap();
  // The stale shard — present on disk, but NOT in the index. The
  // pre-structural-fix `load_weights` (which globbed
  // `model*.safetensors`) would have resurrected this tensor; the
  // index-honoring `load_weights` must NOT.
  let stale = Array::from_slice::<f32>(&[99.0], &(1usize,)).unwrap();
  crate::io::save_safetensors_view(
    &dir.join("model-00099-of-00099.safetensors"),
    std::iter::once(("stale.weight", &stale)),
    &meta,
  )
  .unwrap();
  // An index that names ONLY the real shard.
  let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
  weight_map.insert("real.weight".to_string(), "model.safetensors".to_string());
  write_json_pretty_to_path(
    &dir.join("model.safetensors.index.json"),
    &serde_json::json!({
      "metadata": { "total_size": 12, "total_parameters": 3 },
      "weight_map": weight_map,
    }),
    "test: index ignores stale",
  )
  .unwrap();

  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(
    loaded.len(),
    1,
    "only the indexed weight loads; the stale shard is invisible"
  );
  assert!(loaded.contains_key("real.weight"));
  assert!(
    !loaded.contains_key("stale.weight"),
    "an out-of-index shard must NOT resurrect tensors on load"
  );
  assert_eq!(
    loaded
      .get_mut("real.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0]
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// An un-sharded checkpoint that has only `model.safetensors` (no index
/// file at all — the simple HF single-file convention) still loads via
/// the second-tier fallback. Back-compat for fresh-from-`huggingface_hub`
/// directories that don't carry an index.
#[test]
fn load_weights_no_index_single_model_safetensors_loads() {
  let dir = fresh_dir("load-single-no-index");
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());

  let w = Array::from_slice::<f32>(&[7.0, 8.0], &(2usize,)).unwrap();
  crate::io::save_safetensors_view(
    &dir.join("model.safetensors"),
    std::iter::once(("only.weight", &w)),
    &meta,
  )
  .unwrap();
  // No `model.safetensors.index.json`.
  assert!(!dir.join("model.safetensors.index.json").exists());

  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(
    loaded
      .get_mut("only.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![7.0, 8.0]
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// Legacy `weights.safetensors`-only directory (pre-HF naming) still
/// loads via the third-tier fallback. No index, no `model.safetensors`,
/// just `weights.safetensors`. Back-compat for older hand-rolled or
/// pre-HF-convention checkpoints.
#[test]
fn load_weights_legacy_weights_safetensors_fallback_loads() {
  let dir = fresh_dir("load-legacy-weights");
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());

  let w = Array::from_slice::<f32>(&[42.0], &(1usize,)).unwrap();
  crate::io::save_safetensors_view(
    &dir.join("weights.safetensors"),
    std::iter::once(("legacy.weight", &w)),
    &meta,
  )
  .unwrap();
  assert!(!dir.join("model.safetensors").exists());
  assert!(!dir.join("model.safetensors.index.json").exists());

  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(
    loaded
      .get_mut("legacy.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![42.0]
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// `load_weights` errors when the index lists a shard that does NOT exist
/// on disk (the load-side counterpart to a torn-publish scenario where
/// only some shards were renamed). The message names the missing shard.
#[test]
fn load_weights_index_lists_missing_shard_errors() {
  let dir = fresh_dir("load-index-missing-shard");
  // Index references `model-00001-of-00002.safetensors` +
  // `model-00002-of-00002.safetensors`, but only the first is on disk.
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  let w = Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap();
  crate::io::save_safetensors_view(
    &dir.join("model-00001-of-00002.safetensors"),
    std::iter::once(("a.weight", &w)),
    &meta,
  )
  .unwrap();

  let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
  weight_map.insert(
    "a.weight".to_string(),
    "model-00001-of-00002.safetensors".to_string(),
  );
  weight_map.insert(
    "b.weight".to_string(),
    "model-00002-of-00002.safetensors".to_string(),
  );
  write_json_pretty_to_path(
    &dir.join("model.safetensors.index.json"),
    &serde_json::json!({
      "metadata": { "total_size": 8, "total_parameters": 2 },
      "weight_map": weight_map,
    }),
    "test: missing-shard index",
  )
  .unwrap();

  let r = load_weights(&dir);
  // The missing shard surfaces a typed `Error::FileIo(NotFound)` naming the
  // missing path with op = `Stat`.
  let Err(Error::FileIo(p)) = r else {
    panic!("a missing indexed shard must be an Error::FileIo, got {r:?}");
  };
  assert_eq!(p.op(), FileOp::Stat);
  assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
  assert_eq!(
    p.path(),
    dir.join("model-00002-of-00002.safetensors").as_path(),
    "path must name the missing shard"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// `load_weights` rejects an index whose `weight_map` value carries a
/// path component (an absolute or `..`-traversing shard name would
/// escape `dir`; HF convention is bare basenames in the same directory).
#[test]
fn load_weights_index_with_path_traversal_errors() {
  let dir = fresh_dir("load-index-path-traversal");
  let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
  weight_map.insert("evil.weight".to_string(), "../../etc/passwd".to_string());
  write_json_pretty_to_path(
    &dir.join("model.safetensors.index.json"),
    &serde_json::json!({
      "metadata": { "total_size": 0, "total_parameters": 0 },
      "weight_map": weight_map,
    }),
    "test: path-traversal index",
  )
  .unwrap();

  let r = load_weights(&dir);
  // The path-traversal shard name surfaces a typed `Error::LayerKeyed`
  // naming the offending `weight_map[<key>] -> <value>` and an inner
  // `Error::InvariantViolation` calling out the basename rule.
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("a path-traversal shard name must be an Error::LayerKeyed, got {r:?}");
  };
  assert!(
    p.layer().contains("weight_map[evil.weight]") && p.layer().contains("../../etc/passwd"),
    "layer should name the offending mapping, got `{}`",
    p.layer()
  );
  assert!(
    matches!(p.inner(), Error::InvariantViolation(iv)
        if iv.context().contains("weight_map shard name")
          && iv.requirement().contains("bare basename")),
    "expected inner InvariantViolation about bare basename, got {:?}",
    p.inner()
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// `load_weights` rejects a malformed (non-JSON) index file rather than
/// silently falling through to the next tier — an unparseable index is a
/// genuine corruption signal.
#[test]
fn load_weights_malformed_index_errors() {
  let dir = fresh_dir("load-index-malformed");
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    b"this is not valid JSON {{{",
  )
  .unwrap();
  let r = load_weights(&dir);
  // The malformed index is wrapped in `Error::LayerKeyed` naming the index
  // path and an inner `Error::Parse` from the JSON parser.
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("expected Error::LayerKeyed for a malformed index, got {r:?}");
  };
  assert!(
    p.layer().contains("model.safetensors.index.json"),
    "layer should name the index path, got `{}`",
    p.layer()
  );
  assert!(
    matches!(p.inner(), Error::Parse(pp)
        if pp.context() == "load_via_index: model weight index" && pp.input_kind() == "JSON"),
    "expected inner Error::Parse for malformed JSON, got {:?}",
    p.inner()
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// An empty model directory (no index, no safetensors, no GGUF) is the
/// final fall-through to an error. The `FileIo` payload's static `context()`
/// label lists every layout the resolver considered.
#[test]
fn load_weights_empty_dir_errors_listing_layouts() {
  let dir = fresh_dir("load-empty");
  let r = load_weights(&dir);
  let Err(Error::FileIo(p)) = r else {
    panic!("an empty dir must be an Error::FileIo, got {r:?}");
  };
  // The path is the model directory; the `op` is `Open` (the final
  // open-attempt that failed); the inner io::Error is `NotFound`.
  assert_eq!(p.path(), dir.as_path());
  assert_eq!(p.op(), FileOp::Open);
  assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
  // The static `context()` lists each resolver tier.
  let ctx = p.context();
  assert!(
    ctx.contains("model.safetensors.index.json")
      && ctx.contains("model.safetensors")
      && ctx.contains("weights.safetensors"),
    "the context must list each resolver tier, got: {ctx}"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

// ────────── save_model: index-rename single commit point ──────────

/// **The structural atomicity test.** A save that fails AFTER the shards
/// rename but BEFORE the index rename must leave the OLD checkpoint
/// loadable EXACTLY (every weight key + value byte-identical), with the
/// OLD `model.safetensors.index.json` untouched. The failure is injected
/// by pre-creating `model.safetensors.index.json` as a *directory* so the
/// final atomic `rename(file -> dir)` fails after every shard has been
/// renamed into place. Because new shards are generation-tagged
/// (`model-gen-{ts}-…`), the renames never collide with the OLD
/// `model-00001-of-00002.safetensors` and
/// `model-00002-of-00002.safetensors` files — the OLD shards are
/// untouched by construction.
#[test]
fn save_model_torn_publish_before_index_rename_keeps_old_checkpoint() {
  let dir = fresh_dir("torn-publish-before-index-rename");
  // 1. Write an OLD 2-shard checkpoint with NON-colliding names.
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  let old_a = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
  let old_b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0], &(3usize,)).unwrap();
  crate::io::save_safetensors_view(
    &dir.join("model-00001-of-00002.safetensors"),
    std::iter::once(("old.a.weight", &old_a)),
    &meta,
  )
  .unwrap();
  crate::io::save_safetensors_view(
    &dir.join("model-00002-of-00002.safetensors"),
    std::iter::once(("old.b.weight", &old_b)),
    &meta,
  )
  .unwrap();
  // The OLD index file — to be left untouched after the failed save.
  let mut old_wm: BTreeMap<String, String> = BTreeMap::new();
  old_wm.insert(
    "old.a.weight".to_string(),
    "model-00001-of-00002.safetensors".to_string(),
  );
  old_wm.insert(
    "old.b.weight".to_string(),
    "model-00002-of-00002.safetensors".to_string(),
  );
  let old_index_text = serde_json::to_string(&serde_json::json!({
    "metadata": { "total_size": 20, "total_parameters": 5 },
    "weight_map": old_wm,
  }))
  .unwrap();
  // Sanity: confirm the OLD shards are loadable when we put the OLD
  // index in place.
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    old_index_text.as_bytes(),
  )
  .unwrap();
  let mut sanity = load_weights(&dir).unwrap();
  assert_eq!(sanity.len(), 2);
  assert_eq!(
    sanity
      .get_mut("old.a.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0]
  );
  drop(sanity);
  // 2. Remove the OLD index file and plant a directory in its place so
  //    the final atomic `rename(file -> dir)` of the NEW index fails
  //    AFTER every NEW shard has been renamed into place. We will assert
  //    that after the failed save, restoring the OLD index file lets
  //    load follow it to the still-intact OLD shards.
  std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
  std::fs::create_dir_all(dir.join("model.safetensors.index.json")).unwrap();

  // 3. Attempt the new save — a smaller single-shard checkpoint.
  let mut new_w: Weights = HashMap::new();
  new_w.insert(
    "new.x.weight".to_string(),
    Array::from_slice::<f32>(&[100.0], &(1usize,)).unwrap(),
  );
  let r = save_model(&dir, &new_w, &PerLayerQuantization::default());
  assert!(
    r.is_err(),
    "the index rename onto an existing directory must fail"
  );

  // 4. The OLD shards must be untouched, and byte-identical.
  let old_a_path = dir.join("model-00001-of-00002.safetensors");
  let old_b_path = dir.join("model-00002-of-00002.safetensors");
  assert!(
    old_a_path.is_file(),
    "OLD shard 1 must survive the failed save"
  );
  assert!(
    old_b_path.is_file(),
    "OLD shard 2 must survive the failed save"
  );
  // The NEW shard was renamed into place before the failed index
  // rename; load ignores it as long as it isn't indexed. The OLD index
  // doesn't list it, so it's invisible to a load via the OLD index —
  // exactly the design's promise.
  let new_shards_on_disk: Vec<std::path::PathBuf> = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap();
  assert_eq!(
    new_shards_on_disk.len(),
    1,
    "the NEW shard rename SHOULD have succeeded (it's the index rename that fails); \
       this asserts the torn-publish scenario the test is targeting"
  );
  // No staged tempfile remains.
  let leftover_tmp = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(
    !leftover_tmp,
    "every staged tempfile must be removed when the index rename fails"
  );

  // 5. Restore the OLD index file (replacing the directory we used as
  //    the failure lever) and confirm load follows it to the still-
  //    intact OLD shards. The NEW shard is on disk but is invisible —
  //    load only sees the OLD-indexed shards.
  std::fs::remove_dir_all(dir.join("model.safetensors.index.json")).unwrap();
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    old_index_text.as_bytes(),
  )
  .unwrap();
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(
    loaded.len(),
    2,
    "the OLD checkpoint loads EXACTLY (both weights)"
  );
  assert_eq!(
    loaded
      .get_mut("old.a.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0],
    "old.a is byte-identical"
  );
  assert_eq!(
    loaded
      .get_mut("old.b.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![3.0, 4.0, 5.0],
    "old.b is byte-identical"
  );
  assert!(
    !loaded.contains_key("new.x.weight"),
    "the NEW shard is on disk but the OLD index ignores it"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// The new torn-publish guarantee, end-to-end via the public
/// [`save_model`] API: stage a save over an EXISTING checkpoint,
/// complete shard staging + shard renames, then fail the index rename
/// (by planting a directory at the index destination path). Because
/// the NEW shard basenames are generation-tagged, they cannot
/// overwrite the OLD shards — and the OLD index is left intact, so
/// the loader still returns the OLD checkpoint EXACTLY.
///
/// Distinct from `save_model_torn_publish_before_index_rename_keeps_old_checkpoint`:
/// that test hand-builds the OLD layout with the reference-style names
/// to assert the structural intent; this one round-trips through
/// `save_model` for both saves to prove the end-to-end guarantee
/// holds against the production code path.
#[test]
fn save_model_torn_after_shard_rename_before_index_rename_keeps_old_checkpoint() {
  let dir = fresh_dir("torn-after-shard-before-index");

  // 1. FIRST save: produce a legitimate checkpoint via `save_model`.
  let mut first: Weights = HashMap::new();
  first.insert(
    "first.alpha.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
  );
  first.insert(
    "first.beta.weight".to_string(),
    Array::from_slice::<f32>(&[4.0, 5.0], &(2usize,)).unwrap(),
  );
  save_model(&dir, &first, &PerLayerQuantization::default()).unwrap();

  // Snapshot the OLD checkpoint: every shard's bytes + the OLD index
  // body, all for a post-failure byte-equality check.
  let old_shard_paths: Vec<std::path::PathBuf> = collect_sorted(&dir, |n| {
    n.starts_with("model-gen-") && n.ends_with(".safetensors")
  })
  .unwrap();
  assert!(
    !old_shard_paths.is_empty(),
    "first save produced at least one shard"
  );
  let old_shard_bytes: BTreeMap<std::path::PathBuf, Vec<u8>> = old_shard_paths
    .iter()
    .map(|p| (p.clone(), std::fs::read(p).unwrap()))
    .collect();
  let old_index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();

  // 2. Plant a directory at the index path AFTER removing the OLD
  //    index file (so the OLD shards still sit on disk untouched, but
  //    the next `save_model`'s index rename will fail).
  std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
  std::fs::create_dir_all(dir.join("model.safetensors.index.json")).unwrap();

  // Sleep so the generation timestamp of the second save is guaranteed
  // distinct from the first save's, even on coarser-clock platforms.
  std::thread::sleep(std::time::Duration::from_millis(5));

  // 3. SECOND save: must fail at the index rename, after the new
  //    shard(s) have been renamed into place.
  let mut second: Weights = HashMap::new();
  second.insert(
    "second.gamma.weight".to_string(),
    Array::from_slice::<f32>(&[100.0, 200.0], &(2usize,)).unwrap(),
  );
  let r = save_model(&dir, &second, &PerLayerQuantization::default());
  assert!(
    r.is_err(),
    "the index rename onto an existing directory must fail"
  );

  // 4. Every OLD shard is still on disk + byte-identical to its
  //    pre-failed-save state (the unique generation-tagged basenames
  //    of the SECOND save guaranteed they could not overwrite anything).
  for (path, bytes) in &old_shard_bytes {
    assert!(
      path.is_file(),
      "OLD shard {} must survive the failed save",
      path.display()
    );
    assert_eq!(
      &std::fs::read(path).unwrap(),
      bytes,
      "OLD shard {} must be byte-identical after the failed save",
      path.display()
    );
  }

  // 5. Restore the OLD index file (replacing the failure-lever
  //    directory) and confirm the loader returns the OLD checkpoint
  //    EXACTLY — no resurrected second-save weights.
  std::fs::remove_dir_all(dir.join("model.safetensors.index.json")).unwrap();
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    old_index_text.as_bytes(),
  )
  .unwrap();
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded.contains_key("first.alpha.weight"));
  assert!(loaded.contains_key("first.beta.weight"));
  assert!(
    !loaded.contains_key("second.gamma.weight"),
    "the SECOND save's shard is on disk but the OLD index ignores it"
  );
  assert_eq!(
    loaded
      .get_mut("first.alpha.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0]
  );
  assert_eq!(
    loaded
      .get_mut("first.beta.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![4.0, 5.0]
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// Smoke test the `fsync_dir` helper: open + fsync + close on a
/// writable tmpdir works without error. The function returns
/// `io::Result<()>` rather than `Result<()>` so the call sites in
/// `save_model` / `commit_staged_config` wrap with their own error
/// context.
#[test]
fn fsync_dir_helper_basic() {
  let dir = fresh_dir("fsync-dir-helper");
  // Sanity: the helper signature is `fsync_dir(&Path) -> io::Result<()>`.
  let r: std::io::Result<()> = fsync_dir(&dir);
  r.expect("fsync_dir must succeed on a writable tmpdir");
  let _ = std::fs::remove_dir_all(&dir);
}

// ────────── save: config-staging cheap fix ──────────

/// `save` validates + stages the config BEFORE [`save_model`] touches any
/// weight. An invalid config (malformed JSON) over an existing checkpoint
/// leaves the checkpoint **byte-identical** to its pre-save state — every
/// weight, the index, and the `config.json` are untouched.
#[test]
fn save_invalid_config_keeps_existing_checkpoint_byte_identical() {
  let dir = fresh_dir("save-invalid-config-intact");

  // 1. Write a valid initial checkpoint via the public `save` driver.
  let mut w: Weights = HashMap::new();
  w.insert(
    "orig.a.weight".to_string(),
    Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap(),
  );
  w.insert(
    "orig.b.weight".to_string(),
    Array::from_slice::<f32>(&[30.0], &(1usize,)).unwrap(),
  );
  let good_config = r#"{"model_type": "qwen3", "hidden_size": 64}"#;
  save(&dir, &w, good_config, &PerLayerQuantization::default()).unwrap();

  // Capture the pre-failed-save byte snapshot of every file.
  let snapshot = |dir: &Path| -> BTreeMap<String, Vec<u8>> {
    let mut m: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
      if e.file_type().unwrap().is_file() {
        let name = e.file_name().to_string_lossy().into_owned();
        let bytes = std::fs::read(e.path()).unwrap();
        m.insert(name, bytes);
      }
    }
    m
  };
  let before = snapshot(&dir);
  assert!(
    before
      .keys()
      .any(|k| k.starts_with("model-gen-") && k.ends_with(".safetensors")),
    "the initial save produced a generation-tagged shard"
  );
  assert!(before.contains_key("model.safetensors.index.json"));
  assert!(before.contains_key("config.json"));

  // 2. Attempt a second save with an INVALID config — must fail.
  let bad_config = "this is not valid JSON at all";
  let other_weights: Weights = {
    let mut m: Weights = HashMap::new();
    m.insert(
      "SHOULD.NOT.WIN.weight".to_string(),
      Array::from_slice::<f32>(&[999.0], &(1usize,)).unwrap(),
    );
    m
  };
  let r = save(
    &dir,
    &other_weights,
    bad_config,
    &PerLayerQuantization::default(),
  );
  assert!(r.is_err(), "an invalid config must abort the save");

  // 3. EVERY file is byte-identical to the pre-failed-save state — the
  //    cheap config-staging fix's promise.
  let after = snapshot(&dir);
  // Filter out any tempfile (there should be none, but if any leaks
  // we want to assert separately and not have it pollute the byte-equal
  // comparison).
  let strip_tmp = |m: BTreeMap<String, Vec<u8>>| -> BTreeMap<String, Vec<u8>> {
    m.into_iter()
      .filter(|(k, _)| !k.ends_with(".tmp.safetensors"))
      .collect()
  };
  let leftover_tmp = after.keys().any(|k| k.ends_with(".tmp.safetensors"));
  assert_eq!(
    strip_tmp(before),
    strip_tmp(after),
    "every file under {} must be byte-identical after an invalid-config save",
    dir.display()
  );
  assert!(
    !leftover_tmp,
    "no staged config tempfile may remain after an invalid-config save"
  );

  // 4. The original checkpoint still loads cleanly.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2);
  assert_eq!(
    loaded
      .get_mut("orig.a.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![10.0, 20.0]
  );
  assert!(!loaded.contains_key("SHOULD.NOT.WIN.weight"));

  let _ = std::fs::remove_dir_all(&dir);
}

// ───────────── get_total_parameters: scale-only `.biases` ─────────────

/// A `.biases` tensor present under a **scale-only** quant mode
/// (`mxfp4` / `mxfp8` / `nvfp4`) is structurally invalid — those layouts
/// have no zero-point buffer and reject one. `get_total_parameters` must
/// flag it as an [`Error::Backend`], NOT silently skip it as it does for
/// the affine zero-point. Checked for all three scale-only modes.
#[test]
fn get_total_parameters_scale_only_biases_is_error() {
  for mode in [QuantMode::Mxfp4, QuantMode::Mxfp8, QuantMode::Nvfp4] {
    let mut w: Weights = HashMap::new();
    w.insert(
      "model.layers.0.q_proj.weight".to_string(),
      Array::from_slice::<u32>(&[0u32; 8], &(2usize, 4)).unwrap(),
    );
    w.insert(
      "model.layers.0.q_proj.scales".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    // A stale `.biases` sibling — invalid under a scale-only layout.
    w.insert(
      "model.layers.0.q_proj.biases".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    let quant = PerLayerQuantization::from_global(Quantization {
      group_size: 32,
      bits: 4,
      mode,
    });
    let err = get_total_parameters(&w, &quant);
    // The error is a typed `Error::LayerKeyed` wrapping a typed
    // `Error::KeyCollision` naming the offending `.biases` key and listing
    // the scale-only mode set in the static context.
    let Err(Error::LayerKeyed(p)) = &err else {
      panic!(
        "a `.biases` under scale-only `{}` must be an Error::LayerKeyed, got {err:?}",
        mode.as_str()
      );
    };
    assert!(
      p.layer().contains("q_proj") && p.layer().ends_with(".biases"),
      "layer should name the offending `.biases` key, got `{}`",
      p.layer()
    );
    let Error::KeyCollision(kp) = p.inner() else {
      panic!("expected inner Error::KeyCollision, got {:?}", p.inner());
    };
    // The static context names the scale-only mode set explicitly.
    assert!(
      kp.context().contains("mxfp4")
        && kp.context().contains("mxfp8")
        && kp.context().contains("nvfp4"),
      "context should list the scale-only modes, got: {}",
      kp.context()
    );
    assert_eq!(kp.key(), p.layer());
  }
}

/// The affine counterpart: under `QuantMode::Affine` the `.biases`
/// zero-point buffer is still correctly skipped as metadata (not
/// counted, no error). Hand-trace: packed `.weight` 8 `u32`, `bits = 4`
/// → `8 * 32 / 4 = 64` logical weights; `.scales` + `.biases` → +0.
/// Total = 64.
#[test]
fn get_total_parameters_affine_biases_still_skipped() {
  let mut w: Weights = HashMap::new();
  w.insert(
    "model.layers.0.q_proj.weight".to_string(),
    Array::from_slice::<u32>(&[0u32; 8], &(2usize, 4)).unwrap(),
  );
  w.insert(
    "model.layers.0.q_proj.scales".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  w.insert(
    "model.layers.0.q_proj.biases".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  let quant = PerLayerQuantization::from_global(Quantization::affine(32, 4));
  let total = get_total_parameters(&w, &quant).unwrap();
  assert_eq!(
    total, 64,
    "affine `.biases` skipped, only unpacked weight counts"
  );
}

// ────────── collision-resistant gen_id + fail-closed rename ──────────

/// Two consecutive `save_model` calls from the same process — even in a
/// tight loop where the µs timestamp may not advance between calls —
/// produce on-disk shards with distinct basenames, because the process-
/// global counter component of [`new_gen_id`] always advances. Without
/// the counter a timestamp-only tag would have collided
/// whenever two saves landed in the same ms / µs tick (and the second
/// save would have overwritten the first save's shard via
/// `fs::rename`); the counter closes that hole.
#[test]
fn gen_id_is_collision_resistant_across_same_ms_saves() {
  let dir_a = fresh_dir("gen-id-collision-a");
  let dir_b = fresh_dir("gen-id-collision-b");
  let mut w: Weights = HashMap::new();
  w.insert("w.weight".to_string(), f32_weight(2));
  // Back-to-back saves to two distinct dirs to keep the assertion
  // about basenames, not about a single-dir overwrite (that's covered
  // by `save_model_no_overwrite_of_old_shards`).
  save_model(&dir_a, &w, &PerLayerQuantization::default()).unwrap();
  save_model(&dir_b, &w, &PerLayerQuantization::default()).unwrap();

  let basenames = |dir: &Path| -> Vec<String> {
    collect_sorted(dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap()
    .into_iter()
    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
    .collect()
  };
  let a = basenames(&dir_a);
  let b = basenames(&dir_b);
  assert_eq!(a.len(), 1);
  assert_eq!(b.len(), 1);
  // Even if the µs timestamp tick did not advance between the two
  // saves (e.g. on a coarser-clock host), the counter advances, so
  // the basenames differ.
  assert_ne!(
    a[0], b[0],
    "two same-process saves must produce distinct gen_id-tagged basenames; \
       got {a:?} == {b:?}"
  );

  let _ = std::fs::remove_dir_all(&dir_a);
  let _ = std::fs::remove_dir_all(&dir_b);
}

/// Defense-in-depth: if a pre-existing file occupies one of the
/// predicted final shard paths (the collision-resistant `gen_id`
/// makes this statistically unreachable, so the test plants the file
/// by hand after forcing the gen_id via the test-only
/// `force_next_gen_id` helper) `save_model`'s atomic no-replace
/// `std::fs::hard_link` MUST fail with `ErrorKind::AlreadyExists`
/// and the save MUST surface that as
/// [`crate::Error::ShardPathCollision`] naming the offending path —
/// the planted file is byte-identical (the no-replace primitive
/// cannot overwrite, unlike `rename(2)`) and no staged tempfiles
/// leak.
#[test]
fn save_model_refuses_to_overwrite_existing_shard_basename() {
  let dir = fresh_dir("save-refuses-overwrite");

  // 1. Pick a known gen_id and plant a decoy file at the shard-1-of-1
  //    path that gen_id will predict.
  let forced_gen_id = "9999999999999-cafebabe-0000000000000042";
  let collision_path = dir.join(shard_file_name(forced_gen_id, 1, 1));
  let decoy_bytes = b"pre-existing decoy bytes that must NOT be overwritten";
  std::fs::write(&collision_path, decoy_bytes).unwrap();

  // 2. Force `save_model`'s next gen_id to match the planted path.
  force_next_gen_id(forced_gen_id);

  let mut w: Weights = HashMap::new();
  w.insert("w.weight".to_string(), f32_weight(2));
  let r = save_model(&dir, &w, &PerLayerQuantization::default());

  // 3. The save aborts with `Error::ShardPathCollision` naming the
  //    offending path — the atomic no-replace `hard_link` mapped
  //    `ErrorKind::AlreadyExists` to this variant.
  match r {
    Err(Error::ShardPathCollision(path)) => {
      assert_eq!(
        path, collision_path,
        "the collision error names the planted path"
      );
    }
    other => panic!("expected Err(ShardPathCollision), got {other:?}"),
  }

  // 4. The decoy file is byte-identical — `hard_link`'s no-replace
  //    semantics guarantee no overwrite.
  assert!(
    collision_path.is_file(),
    "the planted decoy at {} must still be a file",
    collision_path.display()
  );
  assert_eq!(
    std::fs::read(&collision_path).unwrap(),
    decoy_bytes,
    "the planted decoy must be byte-identical (hard_link refused to replace)"
  );

  // 5. No staged `.tmp.safetensors` leaks (every staged tempfile was
  //    cleaned up on the collision-cleanup path).
  let leftover_tmp = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(
    !leftover_tmp,
    "no staged tempfile may remain after a ShardPathCollision abort"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// The shard publish primitive must be **atomic no-replace**,
/// not a `symlink_metadata` + `rename` pre-check. A check-then-act
/// has a TOCTOU window: the stat returns `NotFound`, a concurrent
/// writer creates the final path, then `rename(2)` SILENTLY replaces
/// the racing peer's bytes. With `std::fs::hard_link` the race is
/// closed at the syscall boundary — the call either creates the new
/// directory entry or fails `AlreadyExists`, never overwriting.
///
/// The simplest faithful simulation of the race is to plant the
/// colliding file BEFORE calling `save_model`: from `hard_link`'s
/// perspective the final path already exists when the syscall runs,
/// which is exactly the state a TOCTOU race would leave the
/// filesystem in. (A `symlink_metadata` + `rename` implementation
/// would also catch this specific pre-plant via the pre-check, but
/// the contract under test is "the primitive is atomic no-replace",
/// not "the pre-check happens to catch a pre-plant". Together with
/// the original-test pre-plant case both arms are exercised: this
/// test ALSO asserts no-tempfile-leak + no-NEW-index-commit, which
/// the original does not.) Contract:
///
/// 1. `save_model` returns `Err(Error::ShardPathCollision(path))`
///    naming the planted path.
/// 2. The planted file is byte-identical — `hard_link` cannot
///    overwrite, so no bytes are clobbered.
/// 3. No `.tmp.safetensors` leaks — the collision-cleanup path
///    removed every staged tempfile.
/// 4. No NEW `model.safetensors.index.json` exists in the directory
///    — the index rename is the observable commit point and the
///    save aborted BEFORE it; the directory has no index file at
///    all (we started from a `fresh_dir`).
#[test]
fn save_model_concurrent_create_at_final_path_returns_collision_error_not_silent_overwrite() {
  let dir = fresh_dir("save-toctou-no-silent-overwrite");

  // Predict the final shard path from a forced gen_id and plant a
  // file there — equivalent to a concurrent peer winning the race
  // against a `symlink_metadata` + `rename` pre-check (from
  // `hard_link`'s perspective the path is already there when the
  // syscall runs).
  let forced_gen_id = "7777777777777-feedface-00000000beefcafe";
  let final_shard = dir.join(shard_file_name(forced_gen_id, 1, 1));
  let racer_bytes = b"racer-bytes: a concurrent writer's payload that MUST survive";
  std::fs::write(&final_shard, racer_bytes).unwrap();

  force_next_gen_id(forced_gen_id);

  let mut w: Weights = HashMap::new();
  w.insert("z.weight".to_string(), f32_weight(3));
  let r = save_model(&dir, &w, &PerLayerQuantization::default());

  // (1) `Err(ShardPathCollision { path: final_shard })`.
  match r {
    Err(Error::ShardPathCollision(path)) => {
      assert_eq!(
        path, final_shard,
        "collision error names the planted (racer) path"
      );
    }
    other => {
      panic!("expected Err(ShardPathCollision) from atomic no-replace hard_link, got {other:?}")
    }
  }

  // (2) Planted bytes survive byte-identical — `hard_link` is no-
  // replace; a silent overwrite would have clobbered these.
  assert!(
    final_shard.is_file(),
    "the racer file at {} must still be a regular file",
    final_shard.display()
  );
  assert_eq!(
    std::fs::read(&final_shard).unwrap(),
    racer_bytes,
    "racer bytes must be byte-identical — atomic no-replace forbids silent overwrite"
  );

  // (3) No leftover tempfile in the dir.
  let leftover_tmp = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(
    !leftover_tmp,
    "no staged .tmp.safetensors may remain after a ShardPathCollision"
  );

  // (4) No NEW index — the save aborted before the index rename
  // (the observable commit point). Directory was fresh, so the
  // index file must not exist.
  let index_path = dir.join("model.safetensors.index.json");
  assert!(
    !index_path.exists(),
    "no index commit may occur when shard publish fails: {} exists",
    index_path.display()
  );

  let _ = std::fs::remove_dir_all(&dir);
}

// ────────── post-index-commit durability warning ──────────

/// `save_model` returns `Ok(CommitOutcome::CommittedWithDurabilityWarning)`
/// — NOT `Err` — when the post-index-rename `fsync_dir` fails. The
/// visible checkpoint loads correctly; only the parent-directory
/// fsync hiccupped. This is the hole that returning `Err`
/// here would open: it would propagate through [`save`] and drop the staged
/// [`StagedConfig`], deleting its tempfile and leaving NEW
/// weights+index against the OLD config.
///
/// Driven via the test-only `arm_fsync_dir_fault(skip)`: `skip=1`
/// makes the shard-fsync succeed and the INDEX-fsync (the
/// observable-commit-point fsync) fail. The contract:
///
/// 1. `save_model` returns `Ok(CommittedWithDurabilityWarning(_))`.
/// 2. The on-disk checkpoint loads correctly (`load_weights` sees the
///    new weights).
#[test]
fn save_model_post_index_fsync_failure_keeps_visible_checkpoint() {
  let dir = fresh_dir("post-index-fsync-failure");

  let mut w: Weights = HashMap::new();
  w.insert(
    "v.alpha.weight".to_string(),
    Array::from_slice::<f32>(&[7.0, 8.0, 9.0], &(3usize,)).unwrap(),
  );
  w.insert(
    "v.beta.weight".to_string(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );

  // Arm: skip the FIRST fsync_dir call (after shard renames) then
  // fail the second (after the index rename — the durability fsync
  // that follows the observable commit point).
  let _guard = arm_fsync_dir_fault(1);
  let outcome = save_model(&dir, &w, &PerLayerQuantization::default())
    .expect("post-index fsync failure must NOT propagate as Err — it is a durability warning");
  drop(_guard);

  // (1) The returned outcome is the warning variant carrying the
  // injected error.
  let underlying = match outcome {
    CommitOutcome::CommittedWithDurabilityWarning(e) => e,
    CommitOutcome::Committed => {
      panic!("expected CommittedWithDurabilityWarning, got Committed")
    }
  };
  let underlying_msg = underlying.to_string();
  assert!(
    underlying_msg.contains("injected fsync_dir failure"),
    "the durability warning carries the underlying io::Error: got {underlying_msg}"
  );

  // (2) The visible checkpoint loads correctly — the index rename
  // succeeded, so `load_weights` sees the NEW weights.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded.contains_key("v.alpha.weight"));
  assert!(loaded.contains_key("v.beta.weight"));
  assert_eq!(
    loaded
      .get_mut("v.alpha.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![7.0, 8.0, 9.0]
  );
  assert_eq!(
    loaded
      .get_mut("v.beta.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0]
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// `save` proceeds to commit the staged config even when `save_model`
/// returned `CommittedWithDurabilityWarning` — the NEW `config.json`
/// MUST be byte-equal to the staged (cleaned/sorted) content, the OLD
/// `config.json` is gone, and `save`'s final return is
/// [`Error::DurabilityWarning`] with `committed: true`. This is the
/// end-to-end closure for the config-staging durability contract.
#[test]
fn save_post_commit_durability_warning_still_commits_config() {
  let dir = fresh_dir("save-post-commit-warning-commits-config");

  // 1. Initial save with a "before" config so we can prove the OLD
  //    config.json is gone after the second save.
  let mut w0: Weights = HashMap::new();
  w0.insert("w.weight".to_string(), f32_weight(2));
  let before_config = r#"{"model_type": "OLD", "hidden_size": 4}"#;
  save(&dir, &w0, before_config, &PerLayerQuantization::default()).unwrap();
  let old_cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
  assert!(
    old_cfg.contains("\"OLD\""),
    "the OLD config.json was written"
  );

  // 2. Second save with a "after" config + a fsync injection that
  //    fires AFTER the index rename inside save_model (skip=1 — the
  //    shard fsync passes, the index fsync fails). The save MUST
  //    still commit the config (otherwise the staged-config Drop
  //    would delete its tempfile and we'd be left with NEW
  //    weights+index against the OLD config).
  let mut w1: Weights = HashMap::new();
  w1.insert(
    "w.weight".to_string(),
    Array::from_slice::<f32>(&[5.0, 6.0], &(2usize,)).unwrap(),
  );
  let after_config = r#"{"model_type": "NEW", "hidden_size": 8}"#;

  let _guard = arm_fsync_dir_fault(1);
  let r = save(&dir, &w1, after_config, &PerLayerQuantization::default());
  drop(_guard);

  // (1) save's final return is `Err(DurabilityWarning{committed:true})`.
  match r {
    Err(Error::DurabilityWarning(p)) => {
      assert!(
        p.committed(),
        "save's DurabilityWarning must carry committed=true"
      );
      assert!(
        p.source()
          .to_string()
          .contains("injected fsync_dir failure"),
        "the underlying io::Error must be preserved: got {}",
        p.source()
      );
    }
    other => panic!("expected Err(DurabilityWarning), got {other:?}"),
  }

  // (2) The NEW config.json is on disk and byte-equal to the staged
  //    (cleaned/sorted) form of `after_config`.
  let new_cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
  assert!(
    new_cfg.contains("\"NEW\""),
    "the NEW config.json must be on disk: got {new_cfg}"
  );
  assert!(
    !new_cfg.contains("\"OLD\""),
    "the OLD config.json content must be gone: got {new_cfg}"
  );
  // The cleaned-and-sorted form of `after_config` (4-space indented,
  // sorted keys, no trailing newline).
  let expected_cfg = {
    let v: serde_json::Value = serde_json::from_str(after_config).unwrap();
    let obj = v.as_object().unwrap().clone();
    let sorted: BTreeMap<String, serde_json::Value> = obj.into_iter().collect();
    let mut buf = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
    serde::Serialize::serialize(&sorted, &mut ser).unwrap();
    String::from_utf8(buf).unwrap()
  };
  assert_eq!(
    new_cfg, expected_cfg,
    "the NEW config.json must be byte-equal to the staged (cleaned/sorted) form"
  );

  // (3) The visible weights are the NEW ones — `load_weights` loads
  //    via the NEW index that the (warned-on) save did commit.
  let mut loaded = load_weights(&dir).unwrap();
  assert_eq!(
    loaded.get_mut("w.weight").unwrap().to_vec::<f32>().unwrap(),
    vec![5.0, 6.0]
  );

  // (4) No staged tempfile leaks behind.
  let leftover = std::fs::read_dir(&dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
  assert!(!leftover, "no staged tempfile may leak");

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── LOAD-1 (#145): fd-bound atomic-save writers ───────────────────────

/// [`open_excl_temp_shard`] returns BOTH the open [`File`] and the path
/// so callers can write through the original-open fd (no reopen-by-name
/// TOCTOU window). The pre-LOAD-1 signature returned only the path; this
/// test asserts the post-fix shape, verifies the file was actually
/// created on disk, that we can write through the fd, and that the bytes
/// land on the inode the path points at.
#[test]
fn open_excl_temp_shard_returns_file_and_path() {
  use std::io::Write as _;
  let dir = fresh_dir("load1-open-excl-shape");
  let final_path = dir.join("model-00001-of-00001.safetensors");
  let (mut f, tmp) = open_excl_temp_shard(&final_path).unwrap();
  // The path is a same-directory `.tmp.safetensors` sibling of the
  // final path (no cross-directory tempfile).
  assert_eq!(tmp.parent().unwrap(), final_path.parent().unwrap());
  assert!(
    tmp
      .file_name()
      .unwrap()
      .to_string_lossy()
      .ends_with(".tmp.safetensors"),
    "tempfile must keep the .tmp.safetensors suffix, got {}",
    tmp.display()
  );
  // It exists on disk.
  assert!(tmp.exists(), "open_excl_temp_shard must create the file");
  // Writing through the returned `File` is observable at the path —
  // proves the `File` is bound to the same on-disk object as `tmp`.
  let payload = b"LOAD-1: fd-bound shard tempfile";
  f.write_all(payload).unwrap();
  drop(f);
  let on_disk = std::fs::read(&tmp).unwrap();
  assert_eq!(
    on_disk, payload,
    "bytes written through the returned File must land at the returned path"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// **The LOAD-1 TOCTOU regression test for the safetensors writer.**
/// Replace the staging tempfile with a symlink pointing at a "decoy"
/// file AFTER opening the staging fd, then call the fd-bound writer.
/// The writes must land in the ORIGINAL fd's inode (now anonymous —
/// the path resolves to the decoy), not the decoy. Inode comparison
/// catches the case where a reopen-by-name would have followed the
/// symlink.
#[test]
fn save_safetensors_to_file_writes_via_fd_not_reopen_by_path() {
  use std::os::unix::fs::MetadataExt;
  let dir = fresh_dir("load1-safetensors-fd-not-reopen");
  let staging = dir.join("staging.tmp.safetensors");
  let decoy = dir.join("decoy.target");
  // Plant the decoy with known bytes.
  std::fs::write(&decoy, b"DECOY: must not be overwritten").unwrap();
  let decoy_meta_before = std::fs::metadata(&decoy).unwrap();
  let decoy_inode_before = decoy_meta_before.ino();
  // Open the staging fd via the same primitive `save_model` uses (an
  // `O_EXCL` create).
  let (mut staging_file, staging_path) = open_excl_temp_shard(&staging).unwrap();
  let staging_inode = std::fs::metadata(&staging_path).unwrap().ino();
  assert_ne!(
    staging_inode, decoy_inode_before,
    "test sanity: staging tempfile + decoy must be distinct inodes"
  );
  // Simulate the attack: unlink the staging path + symlink it to the
  // decoy. A reopen-by-name from this point on would follow the symlink
  // and write into the decoy. The staging fd we just opened, however,
  // is still pinned to the original (now-anonymous) inode.
  std::fs::remove_file(&staging_path).unwrap();
  std::os::unix::fs::symlink(&decoy, &staging_path).unwrap();
  // Drive the fd-bound writer with a small array.
  let arr = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  crate::io::save_safetensors_to_file(&mut staging_file, std::iter::once(("w", &arr)), &meta)
    .unwrap();
  drop(staging_file);
  // Assert the decoy is byte-for-byte unchanged + still the same inode.
  let decoy_after = std::fs::read(&decoy).unwrap();
  assert_eq!(
    decoy_after, b"DECOY: must not be overwritten",
    "decoy must not be touched by the fd-bound writer"
  );
  let decoy_meta_after = std::fs::metadata(&decoy).unwrap();
  assert_eq!(
    decoy_meta_after.ino(),
    decoy_inode_before,
    "decoy inode must not have changed"
  );
  // Also: the symlink itself still resolves to the decoy (the staging
  // path entry is the symlink, not a new file).
  let lmeta = std::fs::symlink_metadata(&staging_path).unwrap();
  assert!(lmeta.file_type().is_symlink());
  let _ = std::fs::remove_dir_all(&dir);
}

/// **The LOAD-1 TOCTOU regression test for the JSON writer.** Same
/// shape as the safetensors test: replace the staging path with a
/// symlink to a decoy AFTER opening the staging fd, call the fd-bound
/// `write_json_pretty`, and assert the decoy is untouched. The
/// pre-LOAD-1 `write_json_pretty(&Path,...)` would `fs::write` the
/// symlinked decoy.
#[test]
fn write_json_pretty_writes_via_fd_not_reopen_by_path() {
  use std::os::unix::fs::MetadataExt;
  let dir = fresh_dir("load1-json-fd-not-reopen");
  let staging = dir.join("staging.tmp.safetensors");
  let decoy = dir.join("decoy.json");
  std::fs::write(&decoy, b"{\"decoy\": true}").unwrap();
  let decoy_inode_before = std::fs::metadata(&decoy).unwrap().ino();
  let (mut staging_file, staging_path) = open_excl_temp_shard(&staging).unwrap();
  std::fs::remove_file(&staging_path).unwrap();
  std::os::unix::fs::symlink(&decoy, &staging_path).unwrap();
  // Drive the fd-bound JSON writer.
  let value = serde_json::json!({
    "metadata": { "total_size": 0, "total_parameters": 0 },
    "weight_map": {},
  });
  write_json_pretty(
    &mut staging_file,
    &staging_path,
    &value,
    "LOAD-1: json fd-bound",
  )
  .unwrap();
  drop(staging_file);
  let decoy_after = std::fs::read(&decoy).unwrap();
  assert_eq!(
    decoy_after, b"{\"decoy\": true}",
    "decoy JSON must be untouched by the fd-bound writer"
  );
  assert_eq!(
    std::fs::metadata(&decoy).unwrap().ino(),
    decoy_inode_before,
    "decoy inode must not have changed"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Functional round-trip for the fd-bound safetensors writer: an array
/// written through `save_safetensors_to_file` reloads byte-for-byte via
/// [`crate::io::load_safetensors`]. Confirms the custom `mlx_io_writer`
/// (which delegates `tell`/`seek`/`write` to the supplied `&mut File`)
/// drives mlx-c through a correct safetensors layout — JSON header,
/// per-tensor `data_offsets`, then the contiguous tensor-data section
/// — equivalent in semantics to the path-based writer. The on-disk
/// byte sequence cannot be asserted equal to a path-based write because
/// mlx-c serializes the entry map (an `std::unordered_map`) in a
/// non-deterministic order — the safetensors LAYOUT is invariant, the
/// per-tensor offsets are not.
#[test]
fn save_safetensors_to_file_round_trips_via_path_load() {
  let dir = fresh_dir("load1-fd-round-trip");
  let arr_a = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(4usize,)).unwrap();
  let arr_b = Array::from_slice::<f32>(&[10.0_f32, 20.0], &(2usize,)).unwrap();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());

  // Fd-based write into a freshly-created `File`.
  let path = dir.join("via_fd.safetensors");
  let mut f = std::fs::File::create(&path).unwrap();
  crate::io::save_safetensors_to_file(&mut f, [("a", &arr_a), ("b", &arr_b)], &meta).unwrap();
  f.sync_all().unwrap();
  drop(f);

  // Reload through the path-based loader — proves the on-disk
  // safetensors layout is valid (parseable header, correct offsets,
  // correct dtype + shape encoding).
  let mut loaded = crate::io::load_safetensors(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  let a_read = loaded.get_mut("a").unwrap().to_vec::<f32>().unwrap();
  let b_read = loaded.get_mut("b").unwrap().to_vec::<f32>().unwrap();
  assert_eq!(a_read, vec![1.0, 2.0, 3.0, 4.0]);
  assert_eq!(b_read, vec![10.0, 20.0]);

  let _ = std::fs::remove_dir_all(&dir);
}

/// **Happy-path: prefilled file at non-zero cursor → clean
/// safetensors.** Without the internal rewind in
/// `save_safetensors_to_file`, a caller-supplied `File` at a non-zero
/// cursor would receive a safetensors header at the current cursor +
/// stale prefilled bytes as the prefix — producing a corrupt file
/// that `load_safetensors` could not parse, while the writer returned
/// `Ok(())`. This test pre-fills the file with 100 bytes, seeks to
/// byte 50, drives the writer with a small array, and asserts the
/// reload succeeds + the on-disk size equals exactly the new
/// safetensors payload size (no leading 50 bytes of garbage, no
/// trailing 50 bytes of garbage). Documents the destructive truncate
/// is part of the happy-path contract — see the "Destructive
/// mutation" section of `save_safetensors_to_file`'s doc comment.
#[test]
fn save_safetensors_to_file_truncates_prefilled_file_at_nonzero_offset() {
  use std::io::{Seek, SeekFrom, Write as _};
  let dir = fresh_dir("load1-fd-prefilled-nonzero");
  let path = dir.join("prefilled_nonzero.safetensors");
  // Pre-fill the file with 100 bytes of obviously-not-safetensors data,
  // then seek to byte 50. The writer must reset to byte 0 + truncate
  // before writing — otherwise the on-disk bytes would start with the
  // first 50 prefill bytes and the safetensors payload would follow at
  // offset 50, yielding an unparseable file.
  let mut f = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(true)
    .open(&path)
    .unwrap();
  f.write_all(&[0xAB_u8; 100]).unwrap();
  f.seek(SeekFrom::Start(50)).unwrap();
  let arr = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(4usize,)).unwrap();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  crate::io::save_safetensors_to_file(&mut f, std::iter::once(("w", &arr)), &meta).unwrap();
  f.sync_all().unwrap();
  drop(f);
  // The file must now parse as a clean safetensors with exactly the
  // one array we wrote — no leading garbage from the prefill prefix.
  let mut loaded = crate::io::load_safetensors(&path).unwrap();
  assert_eq!(loaded.len(), 1, "expected exactly one tensor in the file");
  let w = loaded.get_mut("w").unwrap().to_vec::<f32>().unwrap();
  assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
  // And the on-disk size must equal exactly the fresh safetensors
  // payload size — written-via-`save_safetensors_view` to a control
  // path with the same array + metadata. A retained prefill prefix
  // or suffix would push the size past the control. We can't hard-
  // code the byte count because mlx-c's JSON-header layout (key
  // order, whitespace) is an implementation detail, but parity with
  // the path-based writer is the contract this test establishes.
  let control_path = dir.join("control.safetensors");
  let mut control_arrays: HashMap<String, &Array> = HashMap::new();
  control_arrays.insert("w".to_string(), &arr);
  crate::io::save_safetensors_view(
    &control_path,
    control_arrays.iter().map(|(k, &v)| (k.as_str(), v)),
    &meta,
  )
  .unwrap();
  let on_disk = std::fs::metadata(&path).unwrap().len();
  let control_size = std::fs::metadata(&control_path).unwrap().len();
  assert_eq!(
    on_disk, control_size,
    "fd-bound writer on a prefilled-at-offset-50 file must produce the same \
       byte count as the path-based writer on a fresh file (proves rewind+truncate \
       wiped the 100-byte prefill); fd={on_disk}, control={control_size}"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// **Happy-path: prefilled file longer than new payload → trailing
/// bytes truncated.** Without the internal `set_len(0)` truncate, a
/// caller-supplied `File` that already held a much larger blob would
/// retain the trailing tail bytes after the new (shorter)
/// safetensors — the resulting file's prefix would parse but its
/// overall byte length would lie about the payload size, and
/// downstream tooling that mmaps / hashes / verifies the whole file
/// would see garbage past the safetensors EOF. Pre-fills 10000 bytes,
/// rewinds to 0, writes a small payload, asserts the final file size
/// matches a fresh small payload (well under 10000) and reloads
/// correctly. Documents the destructive truncate is part of the
/// happy-path contract — see the "Destructive mutation" section of
/// `save_safetensors_to_file`'s doc comment.
#[test]
fn save_safetensors_to_file_truncates_prefilled_file_longer_than_new_payload() {
  use std::io::{Seek, SeekFrom, Write as _};
  let dir = fresh_dir("load1-fd-prefilled-longer");
  let path = dir.join("prefilled_longer.safetensors");
  let mut f = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(true)
    .open(&path)
    .unwrap();
  // 10000 bytes of obviously-not-safetensors data, then rewind to 0.
  f.write_all(&[0xCD_u8; 10000]).unwrap();
  f.seek(SeekFrom::Start(0)).unwrap();
  let arr = Array::from_slice::<f32>(&[7.0_f32, 8.0, 9.0], &(3usize,)).unwrap();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  crate::io::save_safetensors_to_file(&mut f, std::iter::once(("w", &arr)), &meta).unwrap();
  f.sync_all().unwrap();
  drop(f);
  let mut loaded = crate::io::load_safetensors(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  let w = loaded.get_mut("w").unwrap().to_vec::<f32>().unwrap();
  assert_eq!(w, vec![7.0, 8.0, 9.0]);
  // Final file size must equal exactly a fresh control write — any
  // retained trailing prefill (the bytes past the new shorter
  // payload) would push it past the control. The control is
  // `save_safetensors_view` on a fresh path with the same single
  // array + metadata.
  let control_path = dir.join("control.safetensors");
  let mut control_arrays: HashMap<String, &Array> = HashMap::new();
  control_arrays.insert("w".to_string(), &arr);
  crate::io::save_safetensors_view(
    &control_path,
    control_arrays.iter().map(|(k, &v)| (k.as_str(), v)),
    &meta,
  )
  .unwrap();
  let on_disk = std::fs::metadata(&path).unwrap().len();
  let control_size = std::fs::metadata(&control_path).unwrap().len();
  assert_eq!(
    on_disk, control_size,
    "fd-bound writer on a 10000-byte-prefilled file must produce the same byte \
       count as the path-based writer on a fresh file (proves set_len(0) truncated \
       trailing prefill); fd={on_disk}, control={control_size}"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// **Defense-in-depth: interior-NUL in metadata key leaves file
/// untouched.** Verifies the structural ordering inside
/// `save_safetensors_to_file`: input-validation `Err` from
/// `build_string_map` (interior-NUL in a metadata key) returns BEFORE
/// the destructive `seek(0)` + `set_len(0)`, so a caller-owned
/// prefilled file is byte-identical to its pre-call state on this
/// error path.
///
/// NOT a contract — see the "Destructive mutation" section of
/// `save_safetensors_to_file`'s doc comment. Callers MUST NOT rely on
/// byte preservation across save failures; use the fd-bound
/// tempfile-staging pattern (open a same-directory `O_EXCL` `File`,
/// pass it to `save_safetensors_to_file`, `sync_all`, then `rename` /
/// `hard_link` to the final path — the open/write/fsync/drop fd-bound
/// steps are exemplified by `save_model` above at lines 1359-1372) to
/// preserve the fd-bound write-redirection mitigation through the
/// staging write. The fd-bound mitigation covers the WRITE PATH only;
/// the publication step (`rename` / `hard_link` by `temp_path`) is
/// pathname-based and still subject to directory-entry substitution
/// any time after the `O_EXCL` create and before publication (not
/// just after fsync). See the "Scope of this guarantee" caveat in
/// `save_safetensors_to_file`'s doc comment (its `# Destructive
/// mutation` doc section) for the publication-race options. This test
/// guards the defense-in-depth ordering does not regress, not a
/// behavioral contract callers can depend on.
#[test]
fn save_safetensors_to_file_preserves_existing_file_on_interior_nul_metadata() {
  let dir = fresh_dir("load1-fd-r2-nul-meta");
  let path = dir.join("preexisting_meta.safetensors");
  // Pre-fill the file with known content (NOT a valid safetensors —
  // we are testing that the file is left UNCHANGED on Err, not that
  // it's still loadable). The byte sequence is arbitrary; the
  // assertion is byte-equality to `original_bytes`.
  let original_bytes: &[u8] = b"existing valid safetensors payload here";
  std::fs::write(&path, original_bytes).unwrap();
  let original_len = original_bytes.len() as u64;
  // Sanity: file is exactly the prefill before the call.
  assert_eq!(
    std::fs::metadata(&path).unwrap().len(),
    original_len,
    "pre-call: file size must equal prefill length"
  );

  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .open(&path)
    .unwrap();
  let array = Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap();
  let mut bad_metadata: HashMap<String, String> = HashMap::new();
  // Interior NUL in the key — `CString::new` rejects this, so
  // `build_string_map` returns `Err` before any FFI call.
  bad_metadata.insert("key\0with-nul".to_string(), "value".to_string());

  let result = crate::io::save_safetensors_to_file(
    &mut file,
    std::iter::once(("name", &array)),
    &bad_metadata,
  );

  // The call must surface an interior-NUL `Error::Backend`.
  assert!(
    result.is_err(),
    "expected Err from interior-NUL in metadata key, got Ok"
  );
  let err_msg = format!("{}", result.unwrap_err());
  assert!(
    err_msg.contains("NUL") || err_msg.contains("nul"),
    "expected an interior-NUL error message, got: {err_msg}"
  );

  // Defense-in-depth: the file must be byte-identical to the prefill
  // because `build_string_map` rejected the interior-NUL key BEFORE
  // the destructive truncate ran. A regression that re-ordered the
  // truncate ahead of the validation step would zero this file.
  drop(file);
  let bytes_after = std::fs::read(&path).unwrap();
  assert_eq!(
    bytes_after, original_bytes,
    "DEFENSE-IN-DEPTH REGRESSION: input-validation Err from build_string_map must \
       return before the destructive seek+set_len so a caller-owned prefilled file is \
       byte-identical to its pre-call state on this error path. NOT a contract — see \
       save_safetensors_to_file's Destructive mutation doc section."
  );
  assert_eq!(
    bytes_after.len() as u64,
    original_len,
    "post-call: file size must still equal prefill length"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// **Defense-in-depth: interior-NUL in array name leaves file
/// untouched.** Symmetric to the metadata-key case above, exercising
/// the OTHER fallible map-build site (`build_array_map`). Verifies
/// the structural ordering: input-validation `Err` from
/// `build_array_map` returns BEFORE the destructive truncate.
///
/// NOT a contract — see the "Destructive mutation" section of
/// `save_safetensors_to_file`'s doc comment. Callers MUST NOT rely on
/// byte preservation across save failures.
#[test]
fn save_safetensors_to_file_preserves_existing_file_on_interior_nul_array_name() {
  let dir = fresh_dir("load1-fd-r2-nul-name");
  let path = dir.join("preexisting_name.safetensors");
  let original_bytes: &[u8] = b"another distinct prefilled payload, array-name path";
  std::fs::write(&path, original_bytes).unwrap();
  let original_len = original_bytes.len() as u64;
  assert_eq!(
    std::fs::metadata(&path).unwrap().len(),
    original_len,
    "pre-call: file size must equal prefill length"
  );

  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .open(&path)
    .unwrap();
  let array = Array::from_slice::<f32>(&[3.0_f32, 4.0, 5.0], &(3usize,)).unwrap();
  let good_metadata: HashMap<String, String> = HashMap::new();

  // Interior NUL in the array name — `build_array_map`'s
  // `CString::new(k)` rejects this and returns `Err` BEFORE any FFI
  // call.
  let bad_name = "arr\0with-nul";
  let result = crate::io::save_safetensors_to_file(
    &mut file,
    std::iter::once((bad_name, &array)),
    &good_metadata,
  );

  assert!(
    result.is_err(),
    "expected Err from interior-NUL in array name, got Ok"
  );
  let err_msg = format!("{}", result.unwrap_err());
  assert!(
    err_msg.contains("NUL") || err_msg.contains("nul"),
    "expected an interior-NUL error message, got: {err_msg}"
  );

  drop(file);
  let bytes_after = std::fs::read(&path).unwrap();
  assert_eq!(
    bytes_after, original_bytes,
    "DEFENSE-IN-DEPTH REGRESSION (array-name path): input-validation Err from \
       build_array_map must return before the destructive seek+set_len so a \
       caller-owned prefilled file is byte-identical to its pre-call state on this \
       error path. NOT a contract — see save_safetensors_to_file's Destructive \
       mutation doc section."
  );
  assert_eq!(
    bytes_after.len() as u64,
    original_len,
    "post-call: file size must still equal prefill length"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// **Defense-in-depth: empty-metadata save succeeds through the
/// NULL-sentinel guard.** Verifies the `ctx.is_null()` check in
/// `build_string_map` (installed to surface a hypothetical
/// `mlx_map_string_to_string_new()` allocation-failure sentinel) does
/// not reject valid handles: with empty metadata the insert loop runs
/// zero times, so the structural NULL guard is the only filter
/// between the `_new()` and the caller. A bug that inverted the
/// predicate or compared the wrong field would surface here as an
/// `Err` on the most common save shape. The structural test below
/// verifies the source carries the explicit check.
///
/// NOT a contract — verifies the defense-in-depth ordering does not
/// regress on the success path. See `save_safetensors_to_file`'s
/// Destructive mutation doc section.
#[test]
fn save_safetensors_to_file_empty_metadata_succeeds_with_null_check() {
  let dir = fresh_dir("load1-fd-r3-empty-meta-ok");
  let path = dir.join("empty_meta_ok.safetensors");
  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .create_new(true)
    .open(&path)
    .unwrap();
  let arr = Array::from_slice::<f32>(&[1.5_f32, 2.5, 3.5], &(3usize,)).unwrap();
  // Empty `HashMap<String, String>` metadata is the shape that
  // bypasses every `_insert` call in `build_string_map`, so the
  // structural `is_null()` guard is the only thing between a
  // hypothetical NULL-ctx sentinel from `_new()` and the caller. A
  // valid (non-NULL) handle must pass through unchanged.
  let empty_metadata: HashMap<String, String> = HashMap::new();
  crate::io::save_safetensors_to_file(&mut file, std::iter::once(("w", &arr)), &empty_metadata)
    .expect(
      "DEFENSE-IN-DEPTH REGRESSION: empty-metadata save_safetensors_to_file must \
         succeed — the NULL-sentinel guard in build_string_map must not reject valid \
         handles. See save_safetensors_to_file's Destructive mutation doc section.",
    );
  file.sync_all().unwrap();
  drop(file);

  let mut loaded = crate::io::load_safetensors(&path).unwrap();
  assert_eq!(loaded.len(), 1, "round-trip must yield exactly one array");
  let w = loaded.get_mut("w").unwrap().to_vec::<f32>().unwrap();
  assert_eq!(
    w,
    vec![1.5, 2.5, 3.5],
    "round-trip values must match the pre-save array"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// **Defense-in-depth structural: map-helper NULL-sentinel guards.**
/// Real allocation failure inside `mlx_map_string_to_array_new()` /
/// `mlx_map_string_to_string_new()` cannot be deterministically
/// injected from a unit test (no allocator-fault hook is plumbed
/// through to the C++ vendored map ctor), and the empty-input shape
/// makes EVERY post-construction defensive `_insert` call a no-op so
/// behavioral coverage cannot trip the NULL path on a real machine.
/// This test reads `mlxrs/src/io.rs` and asserts both `build_array_map`
/// and `build_string_map` carry an explicit `ctx.is_null()` check
/// immediately after the corresponding `_new()` constructor, and
/// drain `crate::error::LAST` rather than peek. A regression that
/// removes either check (e.g. a refactor that drops the guard or
/// reorders the call past the file mutation) will fail this test.
///
/// Guards the defense-in-depth ordering, not a byte-preservation
/// contract — see `save_safetensors_to_file`'s Destructive mutation
/// doc section.
#[test]
fn build_map_helpers_carry_null_sentinel_check() {
  // Read the SOURCE we shipped (not the compiled binary) so a future
  // edit that deletes the guard fails this test deterministically,
  // independent of inlining / optimization. The path is relative to
  // the cargo manifest dir of the `mlxrs` crate.
  let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/io.rs"))
    .expect("must be able to read mlxrs/src/io.rs to verify NULL-sentinel guards");

  // Locate `fn build_array_map` and assert the body contains an
  // explicit `is_null()` predicate AND a `mlx_map_string_to_array_new`
  // call within the same logical region. We slice the source at the
  // function header and check the next ~3 KiB — comfortably larger
  // than either helper body but small enough that any NULL check
  // found belongs to the function it follows.
  let array_fn = src
    .find("fn build_array_map")
    .expect("build_array_map must exist in io.rs");
  // Walk back to the nearest char boundary so a 3 KiB window that lands
  // inside a multi-byte rune (e.g. an `─` in a doc-comment frame) does not
  // panic. Cap matters more than exact-3000 — body fits comfortably.
  let array_end = {
    let target = (array_fn + 3000).min(src.len());
    let mut end = target;
    while end > array_fn && !src.is_char_boundary(end) {
      end -= 1;
    }
    end
  };
  let array_window = &src[array_fn..array_end];
  assert!(
    array_window.contains("mlx_map_string_to_array_new"),
    "DEFENSE-IN-DEPTH STRUCTURAL: build_array_map must still call \
       mlx_map_string_to_array_new"
  );
  assert!(
    array_window.contains("ctx.is_null()"),
    "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: build_array_map must check \
       `guard.0.ctx.is_null()` immediately after `mlx_map_string_to_array_new()` to \
       surface allocation-failure sentinels; the check appears to have been removed."
  );

  let string_fn = src
    .find("fn build_string_map")
    .expect("build_string_map must exist in io.rs");
  let string_end = {
    let target = (string_fn + 3000).min(src.len());
    let mut end = target;
    while end > string_fn && !src.is_char_boundary(end) {
      end -= 1;
    }
    end
  };
  let string_window = &src[string_fn..string_end];
  assert!(
    string_window.contains("mlx_map_string_to_string_new"),
    "DEFENSE-IN-DEPTH STRUCTURAL: build_string_map must still call \
       mlx_map_string_to_string_new"
  );
  assert!(
    string_window.contains("ctx.is_null()"),
    "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: build_string_map must check \
       `guard.0.ctx.is_null()` immediately after `mlx_map_string_to_string_new()` — \
       without this guard, an allocation failure on the empty-metadata save path \
       returns a NULL-ctx sentinel through `Ok(NULL)` to the caller. The check \
       appears to have been removed."
  );

  // Both windows must also drain LAST (not peek). The drain is the
  // crate's `crate::error::take_last()` / `LAST.with(...).take()`
  // idiom; either spelling is acceptable — but SOMETHING must
  // consume the TLS so a stale Err does not poison the next call.
  let drains_last = |window: &str| {
    window.contains("take_last()")
      || window.contains("LAST.with")
      || window.contains("crate::error::take_last")
  };
  assert!(
    drains_last(array_window),
    "DEFENSE-IN-DEPTH STRUCTURAL: build_array_map's NULL branch must DRAIN \
       crate::error::LAST (via take_last() or LAST.with(..).take()), not peek \
       — leaving a stale Err in the TLS pollutes later mlx-c calls on this thread."
  );
  assert!(
    drains_last(string_window),
    "DEFENSE-IN-DEPTH STRUCTURAL: build_string_map's NULL branch must DRAIN \
       crate::error::LAST (via take_last() or LAST.with(..).take()), not peek \
       — leaving a stale Err in the TLS pollutes later mlx-c calls on this thread."
  );
}

/// **Defense-in-depth structural: writer-new precedes truncate.**
/// `mlx_io_writer_new` allocates a `cwriter_holder` +
/// `std::shared_ptr<CWriter>` inside its `try`/`catch` (vendored
/// `mlx-c/mlx/c/private/io.h:126-129` +
/// `mlx-c/mlx/c/io_types.cpp:48-54`) and converts a `std::bad_alloc`
/// (or any other exception) into a `mlx_io_writer({nullptr})`
/// sentinel. Real allocation failure inside that ctor cannot be
/// deterministically injected from a unit test (no allocator-fault
/// hook is plumbed through to the vendored C++ ctor), so this test
/// guards the structural ordering: it reads `mlxrs/src/io.rs` and
/// asserts the lexical ordering — (1) `mlx_io_writer_new` is called
/// BEFORE `seek(SeekFrom::Start(0))`, (2) an explicit
/// `ctx.is_null()` check appears within ~10 lines of
/// `mlx_io_writer_new`, (3) a `take_last()` (or
/// `crate::error::take_last`) drain appears within ~20 lines of
/// `mlx_io_writer_new`, (4) `set_len(0)` appears AFTER the
/// `is_null()` check.
///
/// Guards the defense-in-depth ordering, not a byte-preservation
/// contract — see `save_safetensors_to_file`'s Destructive mutation
/// doc section.
#[test]
fn save_safetensors_to_file_writer_new_precedes_truncate() {
  // Read the SOURCE we shipped (not the compiled binary) so a future
  // edit that re-orders the writer ctor past the truncate fails this
  // test deterministically, independent of inlining / optimization.
  let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/io.rs"))
    .expect("must be able to read mlxrs/src/io.rs to verify writer-precedes-truncate ordering");

  // Restrict the search to the `save_safetensors_to_file` function
  // body so unrelated occurrences (the `WriterGuard` impl, the
  // vtable factory, the `cb_seek` callback) cannot satisfy the
  // assertions by accident.
  let fn_start = src
    .find("pub fn save_safetensors_to_file")
    .expect("save_safetensors_to_file must exist in io.rs");
  // The next sibling item header in this file is the
  // `// ─────── mlx_io_writer backed by &mut File ──────` divider
  // immediately followed by `struct WriterState`. Slice up to that
  // point to capture only the function body.
  let fn_tail = src[fn_start..]
    .find("struct WriterState")
    .expect("WriterState declaration must follow save_safetensors_to_file in io.rs");
  let body = &src[fn_start..fn_start + fn_tail];

  // Locate each landmark by byte offset within the function body so
  // we can compare lexical ordering directly.
  let writer_new_off = body.find("mlx_io_writer_new(").expect(
    "DEFENSE-IN-DEPTH STRUCTURAL: save_safetensors_to_file must construct the \
       mlx_io_writer via `mlx_io_writer_new(...)`; the writer-new call appears to \
       have been removed or renamed.",
  );
  let seek_off = body.find("seek(SeekFrom::Start(0))").expect(
    "DEFENSE-IN-DEPTH STRUCTURAL: save_safetensors_to_file must rewind via \
       `seek(SeekFrom::Start(0))`; the rewind appears to have been removed or renamed.",
  );
  let set_len_off = body.find("set_len(0)").expect(
    "DEFENSE-IN-DEPTH STRUCTURAL: save_safetensors_to_file must truncate via \
       `set_len(0)`; the truncate appears to have been removed or renamed.",
  );

  // Invariant 1: writer-new BEFORE seek.
  assert!(
    writer_new_off < seek_off,
    "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: `mlx_io_writer_new(...)` must appear \
       BEFORE `seek(SeekFrom::Start(0))` inside save_safetensors_to_file so an \
       allocation failure in the writer ctor surfaces as Err before the destructive \
       truncate. Current ordering has writer-new at byte {writer_new_off} and \
       seek at byte {seek_off}.",
  );

  // Invariant 2: explicit `is_null()` check within 10 lines after writer-new.
  let post_writer_new = &body[writer_new_off..];
  let next_lines: Vec<&str> = post_writer_new.lines().take(11).collect();
  let null_check_window = next_lines.join("\n");
  assert!(
    null_check_window.contains(".ctx.is_null()") || null_check_window.contains("ctx.is_null()"),
    "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: within 10 lines after \
       `mlx_io_writer_new(...)` there must be an explicit `.ctx.is_null()` check \
       that drains the NULL-ctx sentinel before any destructive file mutation. \
       The check appears to have been removed.",
  );

  // Invariant 3: `take_last()` (or `crate::error::take_last`) drain
  // within 20 lines after writer-new.
  let drain_lines: Vec<&str> = post_writer_new.lines().take(21).collect();
  let drain_window = drain_lines.join("\n");
  assert!(
    drain_window.contains("take_last()") || drain_window.contains("crate::error::take_last"),
    "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: within 20 lines after \
       `mlx_io_writer_new(...)` there must be a `take_last()` (or \
       `crate::error::take_last`) DRAIN of the TLS error slot — peeking would \
       leave a stale Err and poison the next unrelated mlx-c call on this thread. \
       The drain appears to have been removed or replaced with a peek.",
  );

  // Invariant 4: `set_len(0)` AFTER the `is_null()` check.
  let null_check_local_off = null_check_window
    .find("ctx.is_null()")
    .expect("invariant-2 guard above asserted this exists; cannot fail here");
  let null_check_abs_off = writer_new_off + null_check_local_off;
  assert!(
    null_check_abs_off < set_len_off,
    "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: `set_len(0)` must appear AFTER the \
       `ctx.is_null()` check so a NULL-ctx writer sentinel cannot bypass the guard \
       and trigger the destructive truncate. Current ordering has the null check at \
       byte {null_check_abs_off} and set_len at byte {set_len_off}.",
  );
}

/// **Defense-in-depth behavioral: writer-construction reached on
/// happy path.** Pre-fills a file with 50 bytes, then calls
/// `save_safetensors_to_file` with valid empty metadata and one tiny
/// array. Asserts the call returns `Ok(())` AND that the file ends up
/// bearing the safetensors header bytes (a valid `load_safetensors`
/// round-trip), proving the writer-construction is still reached and
/// the write still happens after the structural reorder. The
/// structural test above is the primary guard; this behavioral one
/// documents the happy path so a future refactor that BREAKS the
/// write itself fails fast.
#[test]
fn save_safetensors_to_file_writer_construction_precedes_truncate() {
  let dir = fresh_dir("load1-fd-r4-writer-precedes-truncate");
  let path = dir.join("prefilled_50_bytes.safetensors");
  {
    let mut prefill = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .create_new(true)
      .open(&path)
      .unwrap();
    std::io::Write::write_all(&mut prefill, &[0xA5_u8; 50]).unwrap();
    prefill.sync_all().unwrap();
  }
  // Reopen at the start so the existing 50-byte prefix would corrupt
  // the safetensors header without the rewind+truncate. On the happy
  // path the structural ordering still produces a valid round-trippable
  // file because writer-new succeeded and the truncate ran before the
  // write.
  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .open(&path)
    .unwrap();
  let arr = Array::from_slice::<f32>(&[7.0_f32, 8.0, 9.0], &(3usize,)).unwrap();
  let empty_metadata: HashMap<String, String> = HashMap::new();
  crate::io::save_safetensors_to_file(&mut file, std::iter::once(("v", &arr)), &empty_metadata)
    .expect(
      "DEFENSE-IN-DEPTH REGRESSION: happy-path save_safetensors_to_file with \
         empty metadata must succeed (writer construction reached + write completed)",
    );
  file.sync_all().unwrap();
  drop(file);

  // Round-trip the file: a valid safetensors header (which
  // `load_safetensors` parses) is the proof that the truncate +
  // write both happened after the writer-new succeeded.
  let mut loaded = crate::io::load_safetensors(&path).unwrap();
  assert_eq!(
    loaded.len(),
    1,
    "DEFENSE-IN-DEPTH REGRESSION: round-trip must yield exactly one array \
       (saved one)"
  );
  let v = loaded.get_mut("v").unwrap().to_vec::<f32>().unwrap();
  assert_eq!(
    v,
    vec![7.0, 8.0, 9.0],
    "DEFENSE-IN-DEPTH REGRESSION: round-trip values must match the pre-save \
       array — a mismatch would indicate the write did not run or wrote stale \
       prefix bytes"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// **Documents the destructive contract for MLX-internal errors.**
/// Pre-fills a file with 50 bytes, then calls
/// `save_safetensors_to_file` with a zero-element `Array` that mlx-c
/// rejects inside `mlx_save_safetensors_writer`. Asserts the call
/// returns `Err` AND that the file is truncated to 0 bytes (NOT
/// preserved).
///
/// This is the EXPECTED behavior per the "Destructive mutation"
/// section of `save_safetensors_to_file`'s doc comment — once the
/// defense-in-depth ordering has cleared (Rust map builds, FFI map
/// ctors, FFI writer ctor all `Ok`), the function commits to the
/// destructive `seek(0)` + `set_len(0)` before invoking
/// `mlx_save_safetensors_writer`. Any error from the writer (eval
/// failure, MLX-internal array-set rejection, header-build failure,
/// write-callback failure) leaves the file in a truncated /
/// partially-mutated state. Callers that need atomic-replace
/// semantics must use the fd-bound tempfile-staging pattern (open a
/// same-directory `O_EXCL` `File`, pass it to
/// `save_safetensors_to_file`, `sync_all`, then `rename` /
/// `hard_link` to the final path — the open/write/fsync/drop
/// fd-bound steps are exemplified by `save_model` above at lines
/// 1359-1372). The fd-bound mitigation protects the WRITE PATH from
/// `unlink + symlink` redirection; the publication step (`rename` /
/// `hard_link` by `temp_path`) is pathname-based and still subject
/// to directory-entry substitution any time after the `O_EXCL`
/// create and before publication (not just after fsync). See the
/// "Scope of this guarantee" caveat in
/// `save_safetensors_to_file`'s doc comment (its `# Destructive
/// mutation` doc section) for the publication-race options. Do NOT
/// use the path-taking `save_safetensors_view` for atomic
/// replacement: it reopens by name and reintroduces the write-path
/// TOCTOU window LOAD-1 closed.
///
/// A regression that "fixed" this by preserving the prefill on a
/// writer-error would silently change the function's contract; this
/// test catches such a regression by asserting the file IS
/// destructively truncated on the writer-error path.
#[test]
fn save_safetensors_to_file_truncates_on_mlx_internal_error_zero_element_array() {
  let dir = fresh_dir("load1-fd-destructive-zero-elem");
  let path = dir.join("destructive_contract.safetensors");
  let original_bytes: &[u8] = &[0xC3_u8; 50];
  std::fs::write(&path, original_bytes).unwrap();
  let original_len = original_bytes.len() as u64;
  assert_eq!(
    std::fs::metadata(&path).unwrap().len(),
    original_len,
    "pre-call: file size must equal prefill length"
  );

  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .open(&path)
    .unwrap();
  // A zero-element array constructs successfully in Rust (see e.g.
  // `embeddings::colvision` tests), so all the up-front validation +
  // FFI ctor steps succeed (input maps build, writer-new returns
  // non-NULL). mlx-c's safetensors writer then rejects the
  // zero-element shape inside `mlx_save_safetensors_writer` — AFTER
  // the destructive `seek(0)` + `set_len(0)` have already run. This
  // exercises the "Partially mutated or zero-length" branch of the
  // documented Destructive mutation contract.
  let zero_arr = Array::from_slice::<f32>(&[], &(0usize,)).unwrap();
  let empty_metadata: HashMap<String, String> = HashMap::new();

  let result = crate::io::save_safetensors_to_file(
    &mut file,
    std::iter::once(("zero", &zero_arr)),
    &empty_metadata,
  );

  assert!(
    result.is_err(),
    "expected Err from save_safetensors_to_file on a zero-element array — mlx-c's \
       safetensors writer rejects this shape. If the writer started accepting \
       zero-element arrays, pick another MLX-internal-rejection trigger to keep \
       coverage of the destructive-contract path."
  );

  drop(file);
  let post_len = std::fs::metadata(&path).unwrap().len();
  // The destructive `seek(0)` + `set_len(0)` run BEFORE
  // `mlx_save_safetensors_writer` is invoked, so the file is
  // truncated to 0 bytes (or written as a partial safetensors
  // header if mlx-c emitted some bytes before rejecting the
  // zero-element shape). The strict assertion the documented
  // contract makes is "not byte-identical to the prefill" — the
  // file is partially mutated or zero-length. Asserting
  // `post_len < original_len` covers both cases (early reject ⇒
  // 0 bytes; mid-stream reject ⇒ some bytes of partial header).
  // The original prefill (50 bytes of 0xC3) is not a valid
  // safetensors prefix, so a `post_len == original_len` with
  // byte-identical contents is the regression we are guarding
  // against.
  assert!(
    post_len < original_len,
    "DESTRUCTIVE CONTRACT: save_safetensors_to_file MUST destructively mutate the \
       file on an MLX-internal writer error (the destructive seek+set_len runs \
       before mlx_save_safetensors_writer is invoked). The file size went from \
       {original_len} bytes to {post_len} bytes — if this assertion fires because \
       the file is BYTE-IDENTICAL to the prefill, the function silently regained a \
       byte-preservation contract it explicitly disclaims. See \
       save_safetensors_to_file's Destructive mutation doc section."
  );

  let _ = std::fs::remove_dir_all(&dir);
}

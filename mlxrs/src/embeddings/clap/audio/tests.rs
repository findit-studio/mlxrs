//! Oracle / shape tests for the CLAP HTSAT Swin audio tower.
//!
//! No checkpoint is available, so these pin the tower-level assembly against
//! closed-form expectations and exercise a full-config-sized tower built from
//! synthetic weights. The Swin sub-blocks (window partition/reverse, the
//! relative-position-bias gather, the shifted-window mask, patch-merging) are
//! already pinned by `shared/tests.rs`; these focus on the tower seams:
//!
//! - **`reshape_mel2img`** — the output shape (`(1,1,1001,64) → (1,1,256,256)`)
//!   AND a hand-verifiable small-grid **fold** (a no-interp case where the
//!   reshape/permute element placement is checked against the HF arithmetic —
//!   the critical oracle, since the mel→image fold has no textclap cross-check);
//! - **the per-stage downsampling** — each stage halves the resolution and
//!   doubles the channel width;
//! - **the whole tower** at the real `laion/clap-htsat-unfused` config size:
//!   the `(1,1,1001,64) → (1,768)` pooled feature, the rank/shape input guards,
//!   the quantized-checkpoint load + forward, and **f16 / bf16** dtype
//!   preservation.

use std::collections::HashMap;

use super::*;
use crate::dtype::Dtype;

// ───────────────────────── small Array helpers ─────────────────────────

/// Cast `a` to f32, eval, and read it back as a flat `Vec<f32>`.
fn read_f32(a: &Array) -> Vec<f32> {
  let mut a = ops::misc::astype(a, Dtype::F32).unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// A `(rows, cols)` f32 matrix with small deterministic entries.
fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A `(n,)` f32 vector with small deterministic entries.
fn vec1(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize).map(|i| ((i % 5) as f32) * 0.01).collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

/// A `(rank0, rank1, rank2, rank3)` f32 tensor of `0, 1, 2, …` (row-major) — the
/// deterministic ramp the fold oracle places into the mel image.
fn arange4(d0: i32, d1: i32, d2: i32, d3: i32) -> Array {
  let n = (d0 * d1 * d2 * d3) as usize;
  let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
  Array::from_slice::<f32>(&data, &(d0 as usize, d1 as usize, d2 as usize, d3 as usize)).unwrap()
}

/// Insert a dense biased/unbiased `nn.Linear` (`{prefix}.weight` `(out, in)` +
/// optional `{prefix}.bias` `(out,)`).
fn put_linear(weights: &mut HashMap<String, Array>, prefix: &str, out: i32, in_f: i32, bias: bool) {
  weights.insert(format!("{prefix}.weight"), mat(out, in_f));
  if bias {
    weights.insert(format!("{prefix}.bias"), vec1(out));
  }
}

/// Insert a `LayerNorm` (`{prefix}.weight` + `{prefix}.bias`, both `(hidden,)`).
fn put_layer_norm(weights: &mut HashMap<String, Array>, prefix: &str, hidden: i32) {
  weights.insert(format!("{prefix}.weight"), vec1(hidden));
  weights.insert(format!("{prefix}.bias"), vec1(hidden));
}

/// The `((2·window − 1)², num_heads)` relative-position-bias table (constant
/// fill — the tower shape tests don't need a varied table).
fn bias_table(window: i32, num_heads: i32) -> Array {
  let span = 2 * window - 1;
  Array::full::<f32>(&((span * span) as usize, num_heads as usize), 0.02).unwrap()
}

// ────────────────────── synthetic full-tower weights ──────────────────────

const PATCH_HIDDEN: i32 = 96;
const WINDOW: i32 = 8;
const PATCH: i32 = 4;
const NUM_MELS: i32 = 64;
const EPS: f32 = 1e-5;
const DEPTHS: [i32; 4] = [2, 2, 6, 2];
const HEADS: [i32; 4] = [4, 8, 16, 32];

/// A real-size [`ClapConfig`] (dense). Every audio dim is pinned by `validate`,
/// so the tower is the genuine `laion/clap-htsat-unfused` HTSAT shape.
fn clap_config() -> ClapConfig {
  let cfg = ClapConfig::from_json("{}").unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Insert one Swin block's weights under `layers.{stage}.blocks.{i}`: the two
/// LayerNorms, the window attention (q/k/v/output + the bias table), and the
/// Swin MLP (`intermediate.dense` + `output.dense`, hidden = mlp_ratio·dim).
fn put_swin_block(w: &mut HashMap<String, Array>, stage: i32, i: i32, dim: i32, heads: i32) {
  let p = format!("layers.{stage}.blocks.{i}");
  put_layer_norm(w, &format!("{p}.layernorm_before"), dim);
  put_linear(w, &format!("{p}.attention.self.query"), dim, dim, true);
  put_linear(w, &format!("{p}.attention.self.key"), dim, dim, true);
  put_linear(w, &format!("{p}.attention.self.value"), dim, dim, true);
  w.insert(
    format!("{p}.attention.self.relative_position_bias_table"),
    bias_table(WINDOW, heads),
  );
  put_linear(w, &format!("{p}.attention.output.dense"), dim, dim, true);
  put_layer_norm(w, &format!("{p}.layernorm_after"), dim);
  let mlp_hidden = 4 * dim;
  put_linear(w, &format!("{p}.intermediate.dense"), mlp_hidden, dim, true);
  put_linear(w, &format!("{p}.output.dense"), dim, mlp_hidden, true);
}

/// Insert one stage's patch-merge under `layers.{stage}.downsample`:
/// `norm` `(4·dim,)` + `reduction` `(2·dim, 4·dim)` (bias-free).
fn put_patch_merge(w: &mut HashMap<String, Array>, stage: i32, dim: i32) {
  put_layer_norm(w, &format!("layers.{stage}.downsample.norm"), 4 * dim);
  put_linear(
    w,
    &format!("layers.{stage}.downsample.reduction"),
    2 * dim,
    4 * dim,
    false,
  );
}

/// Build a full synthetic dense weight map for the HTSAT audio tower at the real
/// config size: batch-norm buffers, the NHWC patch-embed conv + its norm, the
/// four stages (with downsamples on stages 0..2), and the final norm.
fn htsat_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();

  // batch_norm.{weight,bias,running_mean,running_var} — (num_mel_bins,).
  w.insert("batch_norm.weight".to_string(), vec1(NUM_MELS));
  w.insert("batch_norm.bias".to_string(), vec1(NUM_MELS));
  w.insert(
    "batch_norm.running_mean".to_string(),
    Array::full::<f32>(&(NUM_MELS as usize,), 0.0).unwrap(),
  );
  w.insert(
    "batch_norm.running_var".to_string(),
    Array::full::<f32>(&(NUM_MELS as usize,), 1.0).unwrap(),
  );

  // patch_embed.proj — NHWC conv weight (hidden, KH, KW, C_in=1) + bias (hidden,).
  let conv_n = (PATCH_HIDDEN * PATCH * PATCH) as usize;
  let conv_data: Vec<f32> = (0..conv_n)
    .map(|n| ((n % 11) as f32) * 0.003 + 0.001)
    .collect();
  w.insert(
    "patch_embed.proj.weight".to_string(),
    Array::from_slice::<f32>(
      &conv_data,
      &(
        PATCH_HIDDEN as usize,
        PATCH as usize,
        PATCH as usize,
        1usize,
      ),
    )
    .unwrap(),
  );
  w.insert("patch_embed.proj.bias".to_string(), vec1(PATCH_HIDDEN));
  put_layer_norm(&mut w, "patch_embed.norm", PATCH_HIDDEN);

  // The four stages.
  for stage in 0..4i32 {
    let dim = PATCH_HIDDEN << stage;
    for i in 0..DEPTHS[stage as usize] {
      put_swin_block(&mut w, stage, i, dim, HEADS[stage as usize]);
    }
    if stage < 3 {
      put_patch_merge(&mut w, stage, dim);
    }
  }

  // Final norm over num_features = 96 << 3 = 768.
  put_layer_norm(&mut w, "norm", PATCH_HIDDEN << 3);
  w
}

/// A `(1, 1, time, freq)` synthetic mel with small deterministic entries.
fn synthetic_mel(time: i32, freq: i32) -> Array {
  let n = (time * freq) as usize;
  let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.01 - 0.05).collect();
  Array::from_slice::<f32>(&data, &(1usize, 1usize, time as usize, freq as usize)).unwrap()
}

// ════════════════════════ reshape_mel2img ════════════════════════════

#[test]
fn reshape_mel2img_full_size_shape() {
  // The real CLAP fold: (1, 1, 1001, 64), spec_size 256, freq_ratio 4 →
  // (1, 1, 256, 256). time 1001 < spec_width 1024 (bicubic up), freq 64 ==
  // spec_height 64 (no interp).
  let mel = synthetic_mel(1001, NUM_MELS);
  let img = reshape_mel2img(&mel, 256, 4).unwrap();
  assert_eq!(
    img.shape(),
    vec![1, 1, 256, 256],
    "reshape_mel2img folds (1,1,1001,64) into the (1,1,256,256) image"
  );
}

#[test]
fn reshape_mel2img_known_fold_no_interpolation() {
  // THE critical fold oracle. Choose a tiny config where time == spec_width and
  // freq == spec_height so NO interpolation runs and the reshape/permute element
  // placement is exactly hand-verifiable against the HF arithmetic.
  //
  //   spec_size = 4, freq_ratio = 2  → spec_width = 8, spec_height = 2.
  //   input (1, 1, time=8, freq=2)   → no interp (8 == 8, 2 == 2).
  //   HF fold:
  //     reshape (1, channels*freq_ratio = 2, time//freq_ratio = 4, freq = 2)
  //     permute (0, 1, 3, 2)                         → (1, 2, 2, 4)
  //     reshape (1, channels = 1, freq*freq_ratio = 4, time//freq_ratio = 4)
  //                                                  → (1, 1, 4, 4)
  //
  // Input is the row-major ramp 0..16 in (1,1,8,2). Compute the expected output
  // independently by walking the same reshape/permute on the flat index space.
  let mel = arange4(1, 1, 8, 2);
  let img = reshape_mel2img(&mel, 4, 2).unwrap();
  assert_eq!(img.shape(), vec![1, 1, 4, 4], "folded image shape");

  // Independent reference: flat[t*2 + f] = t*2 + f (the ramp). Walk HF's two
  // reshapes + the permute over the index space.
  // Step A reshape to (cf=2, td=4, f=2): element (a, b, c) ← flat[a*(4*2) + b*2 + c].
  // Step B permute(0,1,3,2) over the (cf, td, f) trailing axes: (a, c, b).
  // Step C reshape to (4, 4): row-major flatten of the permuted (cf=2, f=2, td=4).
  let mut expected = vec![0f32; 16];
  let mut out_idx = 0usize;
  for a in 0..2 {
    // channels*freq_ratio
    for c in 0..2 {
      // freq (now the 3rd axis after permute)
      for b in 0..4 {
        // time//freq_ratio (now the 4th axis after permute)
        // permuted element (a, c, b) came from reshape-A element (a, b, c).
        let src = a * (4 * 2) + b * 2 + c;
        expected[out_idx] = src as f32;
        out_idx += 1;
      }
    }
  }
  assert_eq!(
    read_f32(&img),
    expected,
    "the reshape/permute fold places each element exactly per the HF arithmetic"
  );
}

#[test]
fn reshape_mel2img_upsamples_short_time_axis() {
  // When time < spec_width the time axis is bicubically upsampled to spec_width
  // before the fold; the folded image is still (1, 1, spec_size, spec_size).
  // spec_size = 8, freq_ratio = 2 → spec_width 16, spec_height 4. Input freq 4
  // (== spec_height, no freq interp), time 10 < 16 → time interp.
  let mel = synthetic_mel(10, 4);
  let img = reshape_mel2img(&mel, 8, 2).unwrap();
  assert_eq!(
    img.shape(),
    vec![1, 1, 8, 8],
    "(spec_size, spec_size) image"
  );
}

#[test]
fn reshape_mel2img_rejects_oversized_input() {
  // HF raises if time > spec_width or freq > spec_height.
  let mel = synthetic_mel(2000, NUM_MELS); // time 2000 > spec_width 1024
  let err = reshape_mel2img(&mel, 256, 4).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn reshape_mel2img_rejects_non_rank4() {
  let bad = Array::from_slice::<f32>(&[1.0f32; 8], &(2usize, 4usize)).unwrap();
  let err = reshape_mel2img(&bad, 4, 2).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

// ════════════════════════ patch-embed ════════════════════════════

#[test]
fn patch_embed_produces_grid_tokens() {
  // (1, 1, 8, 8) image, patch 4 → (1, 2*2 = 4, hidden) tokens, grid (2, 2).
  let mut w = HashMap::new();
  let hidden = 6;
  let conv_n = (hidden * PATCH * PATCH) as usize;
  let conv_data: Vec<f32> = (0..conv_n).map(|n| (n as f32) * 0.001).collect();
  w.insert(
    "patch_embed.proj.weight".to_string(),
    Array::from_slice::<f32>(
      &conv_data,
      &(hidden as usize, PATCH as usize, PATCH as usize, 1usize),
    )
    .unwrap(),
  );
  w.insert("patch_embed.proj.bias".to_string(), vec1(hidden));
  put_layer_norm(&mut w, "patch_embed.norm", hidden);

  let pe = PatchEmbed::from_weights(&mut w, 1, hidden, PATCH, EPS).unwrap();
  let image = arange4(1, 1, 8, 8);
  let (tokens, h_grid, w_grid) = pe.forward(&image).unwrap();
  assert_eq!(
    (h_grid, w_grid),
    (2, 2),
    "8 / patch_size 4 = 2 patches per side"
  );
  assert_eq!(
    tokens.shape(),
    vec![1, 4, hidden as usize],
    "(B, grid·grid, hidden)"
  );
}

// ════════════════════════ per-stage downsampling ════════════════════════════

/// Build a single `AudioStage` from a fresh synthetic weight map at a tiny dim.
fn build_stage(
  stage: i32,
  dim: i32,
  heads: i32,
  depth: i32,
  res: i32,
  has_downsample: bool,
) -> AudioStage {
  let mut w = HashMap::new();
  for i in 0..depth {
    put_swin_block(&mut w, stage, i, dim, heads);
  }
  if has_downsample {
    put_patch_merge(&mut w, stage, dim);
  }
  AudioStage::from_weights(
    &mut w,
    stage,
    dim,
    heads,
    depth,
    res,
    res,
    WINDOW,
    4.0,
    EPS,
    has_downsample,
    None,
  )
  .unwrap()
}

#[test]
fn stage_downsample_halves_resolution_and_doubles_channels() {
  // Stage with a 2×2 merge: (1, 16·16, 32) at res 16 → (1, 8·8, 64) at res 8.
  let dim = 32;
  let stage = build_stage(0, dim, 4, 2, 16, true);
  let tokens = 16 * 16;
  let x = Array::full::<f32>(&(1usize, tokens as usize, dim as usize), 0.01).unwrap();
  let (out, h, w) = stage.forward(&x).unwrap();
  assert_eq!((h, w), (8, 8), "patch-merge halves the resolution");
  assert_eq!(
    out.shape(),
    vec![1, (8 * 8) as usize, (2 * dim) as usize],
    "(B, (H/2)(W/2), 2·dim) after the merge"
  );
}

#[test]
fn stage_without_downsample_preserves_shape() {
  // The deepest stage (no merge): (1, 8·8, 64) at res 8 → unchanged shape.
  let dim = 64;
  let stage = build_stage(3, dim, 8, 2, 8, false);
  let tokens = 8 * 8;
  let x = Array::full::<f32>(&(1usize, tokens as usize, dim as usize), 0.01).unwrap();
  let (out, h, w) = stage.forward(&x).unwrap();
  assert_eq!((h, w), (8, 8), "no merge keeps the resolution");
  assert_eq!(out.shape(), vec![1, (8 * 8) as usize, dim as usize]);
}

#[test]
fn deepest_stage_resolution_equals_window_runs_unshifted() {
  // At res 8 == window 8, HF set_shift_and_window_size zeroes the shift for BOTH
  // blocks (min(res) <= window). The stage must build + run (a shift==window/2
  // here would be wrong); it returns a finite output.
  let dim = 64;
  let stage = build_stage(3, dim, 8, 2, 8, false);
  let x = Array::full::<f32>(&(1usize, 64usize, dim as usize), 0.02).unwrap();
  let (out, _, _) = stage.forward(&x).unwrap();
  assert!(
    read_f32(&out).iter().all(|v| v.is_finite()),
    "unshifted deepest stage is finite"
  );
}

// ════════════════════════ whole tower ════════════════════════════

#[test]
fn whole_tower_full_size_shape() {
  // (1, 1, 1001, 64) → (1, 768) pooled feature on the real config.
  let cfg = clap_config();
  let mut w = htsat_weights();
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  let mel = synthetic_mel(1001, NUM_MELS);
  let out = tower.forward(&mel).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, 768],
    "tower pooled feature (B, hidden)"
  );
  assert!(
    read_f32(&out).iter().all(|v| v.is_finite()),
    "pooled feature finite"
  );
}

#[test]
fn tower_rejects_non_rank4_input() {
  let cfg = clap_config();
  let mut w = htsat_weights();
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  let bad = Array::from_slice::<f32>(&[0.0f32; 64], &(1usize, 64usize)).unwrap();
  let err = tower.forward(&bad).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn tower_rejects_wrong_freq_axis() {
  let cfg = clap_config();
  let mut w = htsat_weights();
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  // freq 32 != num_mel_bins 64.
  let bad = synthetic_mel(1001, 32);
  let err = tower.forward(&bad).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn tower_rejects_wrong_channel_axis() {
  let cfg = clap_config();
  let mut w = htsat_weights();
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  // channels 2 != 1.
  let bad = Array::full::<f32>(&(1usize, 2usize, 1001usize, NUM_MELS as usize), 0.0).unwrap();
  let err = tower.forward(&bad).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

// ════════════════════════ dtype preservation ════════════════════════════

fn assert_tower_preserves_dtype(dtype: Dtype) {
  let cfg = clap_config();
  let mut w = htsat_weights();
  for v in w.values_mut() {
    *v = v.astype(dtype).unwrap();
  }
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  let mel = synthetic_mel(1001, NUM_MELS).astype(dtype).unwrap();
  let out = tower.forward(&mel).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    dtype,
    "tower output must stay {dtype:?} (no silent f32 promotion)"
  );
  assert_eq!(out.shape(), vec![1, 768]);
}

#[test]
fn tower_preserves_f16() {
  assert_tower_preserves_dtype(Dtype::F16);
}

#[test]
fn tower_preserves_bf16() {
  assert_tower_preserves_dtype(Dtype::BF16);
}

// ════════════════════════ quantized load + forward ════════════════════════════

/// Affine group size for the synthetic quantized checkpoint (divides every Swin
/// Linear's `in` axis: 96, 192, 384, 768 and the 4·dim MLP widths are all
/// multiples of 32).
const QGROUP: i32 = 32;
/// Bit depth for the synthetic quantized checkpoint.
const QBITS: i32 = 8;

/// Replace the dense `<prefix>.weight` with the real affine quantize triple.
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .unwrap_or_else(|| panic!("dense weight {prefix}.weight present"));
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// A `ClapConfig` JSON carrying a `quantization` block, so the loader resolves
/// the per-layer scheme for the `.scales`-bearing weights.
fn quant_config() -> ClapConfig {
  let json = format!(r#"{{ "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS} }} }}"#);
  let cfg = ClapConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Parse the quantization block from the config JSON the same way the loader
/// would (so the `.scales`-bearing layers resolve their scheme).
fn quant_from_json() -> PerLayerQuantization {
  let json = format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#);
  serde_json::from_str::<PerLayerQuantization>(&json).unwrap()
}

#[test]
fn tower_loads_and_forwards_quantized_checkpoint() {
  let cfg = quant_config();
  let mut w = htsat_weights();

  // Quantize every Swin nn.Linear: each block's q/k/v + output dense + the two
  // MLP dense layers, and each stage's patch-merge reduction. The batch-norm,
  // the patch-embed conv, the relative-position-bias tables, and the LayerNorms
  // stay dense (a conv / norm / table is not a quantized Linear).
  for stage in 0..4i32 {
    for i in 0..DEPTHS[stage as usize] {
      let p = format!("layers.{stage}.blocks.{i}");
      for proj in ["query", "key", "value"] {
        quantize_weight_in_place(&mut w, &format!("{p}.attention.self.{proj}"));
      }
      quantize_weight_in_place(&mut w, &format!("{p}.attention.output.dense"));
      quantize_weight_in_place(&mut w, &format!("{p}.intermediate.dense"));
      quantize_weight_in_place(&mut w, &format!("{p}.output.dense"));
    }
    if stage < 3 {
      quantize_weight_in_place(&mut w, &format!("layers.{stage}.downsample.reduction"));
    }
  }

  let quant = quant_from_json();
  let tower = HtsatAudioTower::from_weights_quantized(&cfg, &mut w, Some(&quant)).unwrap();
  assert!(
    tower.all_swin_linears_quantized(),
    "every Swin Linear (and patch-merge reduction) must have loaded quantized"
  );

  let mel = synthetic_mel(1001, NUM_MELS);
  let out = tower.forward(&mel).unwrap();
  assert_eq!(out.shape(), vec![1, 768]);
  assert!(
    read_f32(&out).iter().all(|v| v.is_finite()),
    "quantized tower forward is finite"
  );
}

// ════════════════════════ SW-MSA mask-cache perf bench ════════════════════════

/// `#[ignore]` timing bench for the SW-MSA shifted-window-mask cache (#365): the
/// real-config HTSAT tower has 5 shifted (SW-MSA) Swin blocks per forward (stages
/// 0/1: 1 each, stage 2: 3, stage 3: 0 — its `8 == window` resolution zeroes the
/// shift), so the un-cached path rebuilds `shifted_window_mask` 5× per forward.
/// This times the whole audio-tower forward at the real `(1,1,1001,64)` shape,
/// best-of-N (min, to cut GPU scheduling noise). The mask path is
/// weight-independent, so the synthetic-weight tower is representative.
///
/// Run with:
/// `cargo test -p mlxrs --features clap --lib -- --ignored --nocapture
/// embeddings::clap::audio::tests::bench_audio_tower_forward`
#[test]
#[ignore = "perf bench — run with --ignored --nocapture (SW-MSA mask-cache, #365)"]
fn bench_audio_tower_forward() {
  use std::time::Instant;
  const WARMUP: usize = 8;
  const ITERS: usize = 60;

  let cfg = clap_config();
  let mut w = htsat_weights();
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  let mel = synthetic_mel(1001, NUM_MELS);

  for _ in 0..WARMUP {
    let mut out = tower.forward(&mel).unwrap();
    out.eval().unwrap();
  }
  let mut times = Vec::with_capacity(ITERS);
  for _ in 0..ITERS {
    let t0 = Instant::now();
    let mut out = tower.forward(&mel).unwrap();
    out.eval().unwrap();
    times.push(t0.elapsed().as_secs_f64() * 1e3);
  }
  times.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let min = times[0];
  let median = times[times.len() / 2];
  println!(
    "\nMLXRS clap HTSAT audio-tower forward (1x1x1001x64, 5 SW-MSA blocks): \
     min={min:.3}ms median={median:.3}ms  (best of {ITERS})"
  );
}

/// Numerics witness for the SW-MSA mask cache (#365): the real-config tower
/// forward (which exercises all 5 shifted SW-MSA blocks) must be **bit-identical**
/// whether the mask is rebuilt each forward or precomputed once at construction.
/// These reference values were captured on the pre-cache forward; the post-cache
/// forward must reproduce them exactly (the cache changes *when* the same mask is
/// built, never its contents).
#[test]
fn audio_tower_forward_numerics_witness() {
  let cfg = clap_config();
  let mut w = htsat_weights();
  let tower = HtsatAudioTower::from_weights(&cfg, &mut w).unwrap();
  let mel = synthetic_mel(1001, NUM_MELS);
  let v = read_f32(&tower.forward(&mel).unwrap());
  assert_eq!(v.len(), 768);
  let sum: f64 = v.iter().map(|&x| x as f64).sum();
  // Captured on the pre-cache (mask-rebuilt-per-forward) implementation.
  assert!(
    (sum - 15.525_256_737_6).abs() < 1e-6,
    "pooled-feature sum drifted: {sum}"
  );
  assert!((v[0]).abs() < 1e-9, "v[0] = {}", v[0]);
  assert!((v[1] - (-0.005_061_477_4)).abs() < 1e-7, "v[1] = {}", v[1]);
  assert!((v[767] - 0.001_040_554).abs() < 1e-7, "v[767] = {}", v[767]);
}

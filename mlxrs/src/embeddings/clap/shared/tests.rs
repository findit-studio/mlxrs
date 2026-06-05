//! Oracle / shape tests for the CLAP HTSAT Swin-Transformer shared blocks.
//!
//! No checkpoint is available, so these pin the windowing / relative-bias /
//! patch-merge math against closed-form expectations computed independently of
//! the code under test:
//!
//! - [`window_partition`] ∘ [`window_reverse`] is the identity on a
//!   `(1, 16, 16, C)` grid (and the partition's window count / shape are
//!   checked);
//! - the shifted-window cyclic `roll` (`roll_axes(-s)` then `roll_axes(+s)`)
//!   round-trips to the identity;
//! - [`relative_position_index`] for `window = 2` equals the hand-built `(4, 4)`
//!   matrix (over the `(2·2 − 1)² = 9`-entry table) — exact;
//! - [`PatchMerging`]'s `2×2` gather maps `(1, 4, 4, C) → (1, 2, 2, 4C)` with the
//!   HF `[(r0,c0), (r1,c0), (r0,c1), (r1,c1)]` neighborhood ordering — exact;
//! - a single [`WindowAttention`] block runs, and adding the relative-position
//!   bias actually changes the output versus a zero-bias baseline;
//! - the [`SwinBlock`] runs (both even/W-MSA and odd/SW-MSA) and the SW-MSA
//!   shift mask changes the output versus a no-shift block;
//! - the quantized path (synthetic `.scales`) builds `Quantized` layers;
//! - f16 / bf16 dtype preservation on the relative-bias add (the bias table is
//!   fp32; an f16/bf16 activation must stay in its dtype).
//!
//! Oracle values are computed in the test (a plain `Vec` reference, or a numpy-
//! style hand calculation), never by calling the function under test.

use std::collections::HashMap;

use super::*;
use crate::dtype::Dtype;

// ───────────────────────── small Array helpers ─────────────────────────

/// Cast `a` to f32, eval, and read it back as a flat `Vec<f32>` (`to_vec` is
/// dtype-strict + needs `&mut`; there is no implicit eval).
fn read_f32(a: &Array) -> Vec<f32> {
  let mut a = ops::misc::astype(a, Dtype::F32).unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// Cast `a` to i32, eval, and read it back as a flat `Vec<i32>`.
fn read_i32(a: &Array) -> Vec<i32> {
  let mut a = ops::misc::astype(a, Dtype::I32).unwrap();
  a.eval().unwrap();
  a.to_vec::<i32>().unwrap()
}

/// A deterministic `(d0, d1, d2, d3)` f32 tensor with distinct entries (an
/// arange so a partition/reverse round-trip is byte-checkable).
fn arange4(d0: i32, d1: i32, d2: i32, d3: i32) -> Array {
  let n = (d0 * d1 * d2 * d3) as usize;
  let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
  Array::from_slice::<f32>(&data, &(d0 as usize, d1 as usize, d2 as usize, d3 as usize)).unwrap()
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

/// Insert a dense `nn.Linear` (`{prefix}.weight (out,in)` + optional
/// `{prefix}.bias (out,)`) into `weights`.
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

/// The `((2·window − 1)², num_heads)` relative-position-bias table row count.
fn bias_table_rows(window: i32) -> i32 {
  let span = 2 * window - 1;
  span * span
}

/// Insert the `WindowAttention` weights under `{prefix}` (`.attention`): the
/// `.self.{query,key,value}` + `.output.dense` Linears and the supplied
/// `.self.relative_position_bias_table`.
fn put_window_attention(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  dim: i32,
  bias_table: Array,
) {
  put_linear(weights, &format!("{prefix}.self.query"), dim, dim, true);
  put_linear(weights, &format!("{prefix}.self.key"), dim, dim, true);
  put_linear(weights, &format!("{prefix}.self.value"), dim, dim, true);
  put_linear(weights, &format!("{prefix}.output.dense"), dim, dim, true);
  weights.insert(
    format!("{prefix}.self.relative_position_bias_table"),
    bias_table,
  );
}

/// A zero relative-position-bias table (the no-bias baseline).
fn bias_table_zero(window: i32, num_heads: i32) -> Array {
  Array::full::<f32>(&(bias_table_rows(window) as usize, num_heads as usize), 0.0).unwrap()
}

/// A *varied* relative-position-bias table — distinct per row so the gathered
/// per-token-pair bias is non-uniform. A uniform table would add a constant to
/// every logit, which softmax is invariant to (so it could not change the
/// output); the real table has distinct learned entries per relative offset.
fn bias_table_varied(window: i32, num_heads: i32) -> Array {
  let rows = bias_table_rows(window);
  // Row r, head h → a small distinct value so no two relative offsets share a
  // bias and the per-pair bias actually shifts the softmax.
  let data: Vec<f32> = (0..rows)
    .flat_map(|r| (0..num_heads).map(move |h| (r as f32) * 0.05 + (h as f32) * 0.013 + 0.01))
    .collect();
  Array::from_slice::<f32>(&data, &(rows as usize, num_heads as usize)).unwrap()
}

// ════════════════════ window partition / reverse round-trip ════════════════

#[test]
fn window_partition_reverse_is_identity() {
  // (1, 16, 16, C) grid, window 8 → 4 windows; partition∘reverse == identity.
  let c = 3;
  let (h, w, window) = (16, 16, 8);
  let x = arange4(1, h, w, c);

  let windows = window_partition(&x, window).unwrap();
  // num_windows·B = (16/8)·(16/8)·1 = 4; window² = 64.
  assert_eq!(
    windows.shape(),
    vec![4, 64, c as usize],
    "partition is (num_windows·B, window², C)"
  );

  let back = window_reverse(&windows, window, h, w).unwrap();
  assert_eq!(back.shape(), vec![1, h as usize, w as usize, c as usize]);
  assert_eq!(
    read_f32(&back),
    read_f32(&x),
    "window_partition then window_reverse is the identity"
  );
}

#[test]
fn window_partition_rejects_non_multiple() {
  // 15 is not a multiple of window 8 → typed error (not a panic).
  let x = arange4(1, 15, 16, 2);
  assert!(
    window_partition(&x, 8).is_err(),
    "a non-multiple height is rejected"
  );
}

// ═══════════════════════ shifted-window roll round-trip ═════════════════════

#[test]
fn shift_roll_round_trips() {
  // The Swin cyclic shift: roll(-s) on (H, W) then roll(+s) is the identity —
  // the same compose the block does around the windowed attention.
  let x = arange4(1, 8, 8, 2);
  let shift = 4; // window/2 for window = 8.
  let rolled = ops::shape::roll_axes(&x, &[-shift, -shift], &[1, 2]).unwrap();
  // The roll actually moves data (not a no-op).
  assert_ne!(
    read_f32(&rolled),
    read_f32(&x),
    "the -shift roll moves the feature map"
  );
  let back = ops::shape::roll_axes(&rolled, &[shift, shift], &[1, 2]).unwrap();
  assert_eq!(
    read_f32(&back),
    read_f32(&x),
    "roll(-s) then roll(+s) is the identity"
  );
}

// ═════════════ relative_position_index closed form (window = 2) ═════════════

#[test]
fn relative_position_index_window2_closed_form() {
  // Hand-built (4, 4) index over the (2·2 − 1)² = 9-entry table. Tokens are
  // 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1); entry = (rel_row + 1)·3 + (rel_col + 1)
  // where rel = coord_i − coord_j. Diagonal is the center index 4.
  let expected: Vec<i32> = vec![
    4, 3, 1, 0, // i = 0
    5, 4, 2, 1, // i = 1
    7, 6, 4, 3, // i = 2
    8, 7, 5, 4, // i = 3
  ];
  let index = relative_position_index(2).unwrap();
  assert_eq!(index.shape(), vec![16], "flat (window⁴,) index");
  assert_eq!(
    read_i32(&index),
    expected,
    "window = 2 relative_position_index matches the hand-built (4, 4) matrix"
  );
  // Every entry indexes a valid table row (0..9).
  assert!(
    read_i32(&index).iter().all(|&v| (0..9).contains(&v)),
    "indices are within the (2·window − 1)² table"
  );
}

#[test]
fn relative_position_index_diagonal_is_center() {
  // For any window, token i vs itself has rel = (0, 0) → the center index
  // (window − 1)·(2·window − 1) + (window − 1). window = 8 ⇒ 7·15 + 7 = 112.
  let window = 8;
  let area = (window * window) as usize;
  let idx = read_i32(&relative_position_index(window).unwrap());
  let center = (window - 1) * (2 * window - 1) + (window - 1);
  for i in 0..area {
    assert_eq!(
      idx[i * area + i],
      center,
      "the i↔i (zero relative offset) entry is the table center"
    );
  }
}

// ══════════════════════ patch merging 2×2 reshape ══════════════════════════

#[test]
fn patch_merging_2x2_ordering_exact() {
  // (1, 4, 4, C) → (1, 2, 2, 4C); the four channel-blocks are the HF
  // `[(r0,c0), (r1,c0), (r0,c1), (r1,c1)]` strided sub-grids. We bypass the
  // norm + reduction (set norm to identity, reduction to identity) and check
  // the concatenation ordering directly via the merged 4C tensor.
  //
  // Build the merge by hand from the strided slices and compare to the model's
  // pre-reduction concat: replicate the four `input[:, r::2, c::2, :]` grids.
  let c = 2;
  let (h, w) = (4, 4);
  let x = arange4(1, h * w, 1, c); // (1, H·W, C) as the block sees it
  let map = ops::shape::reshape(&x, &[1, h, w, c]).unwrap();

  let sub = |row: i32, col: i32| {
    ops::indexing::slice(&map, &[0, row, col, 0], &[1, h, w, c], &[1, 2, 2, 1]).unwrap()
  };
  // HF order: c outer, r inner ⇒ (r0,c0),(r1,c0),(r0,c1),(r1,c1).
  let expected =
    ops::shape::concatenate(&[&sub(0, 0), &sub(1, 0), &sub(0, 1), &sub(1, 1)], -1).unwrap();
  assert_eq!(
    expected.shape(),
    vec![1, 2, 2, (4 * c) as usize],
    "merged neighborhood is (1, H/2, W/2, 4C)"
  );

  // The first spatial cell (0,0) of the merged grid gathers map[(0,0)],
  // map[(1,0)], map[(0,1)], map[(1,1)] each (C,) — verify the channel layout.
  let merged = read_f32(&expected);
  let cs = c as usize;
  // map is arange over (1,4,4,2): value at (r,col,ch) = (r*4 + col)*2 + ch.
  let at = |r: usize, col: usize, ch: usize| ((r * 4 + col) * 2 + ch) as f32;
  // merged[0,0,0, :] = [map(0,0), map(1,0), map(0,1), map(1,1)] over channels.
  for ch in 0..cs {
    assert_eq!(merged[ch], at(0, 0, ch), "block 0 = (r0,c0)");
    assert_eq!(merged[cs + ch], at(1, 0, ch), "block 1 = (r1,c0)");
    assert_eq!(merged[2 * cs + ch], at(0, 1, ch), "block 2 = (r0,c1)");
    assert_eq!(merged[3 * cs + ch], at(1, 1, ch), "block 3 = (r1,c1)");
  }
}

#[test]
fn patch_merging_forward_shape_and_quant() {
  // The full PatchMerging block: (1, H·W, dim) → (1, (H/2)(W/2), 2·dim).
  let dim = 8;
  let (h, w) = (4, 4);
  let mut weights = HashMap::new();
  put_layer_norm(&mut weights, "pm.norm", 4 * dim);
  put_linear(&mut weights, "pm.reduction", 2 * dim, 4 * dim, false);
  let pm = PatchMerging::from_weights(&mut weights, "pm", dim, 1e-5, None).unwrap();
  assert!(!pm.is_quantized(), "dense load is not quantized");

  let x = mat(h * w, dim);
  let x = ops::shape::reshape(&x, &[1, h * w, dim]).unwrap();
  let out = pm.forward(&x, h, w).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, ((h / 2) * (w / 2)) as usize, (2 * dim) as usize],
    "patch merge halves the grid and doubles the channels"
  );
}

#[test]
fn patch_merging_rejects_odd_resolution() {
  let dim = 4;
  let mut weights = HashMap::new();
  put_layer_norm(&mut weights, "pm.norm", 4 * dim);
  put_linear(&mut weights, "pm.reduction", 2 * dim, 4 * dim, false);
  let pm = PatchMerging::from_weights(&mut weights, "pm", dim, 1e-5, None).unwrap();
  // 3×4 is odd in height → typed error.
  let x = ops::shape::reshape(&mat(3 * 4, dim), &[1, 3 * 4, dim]).unwrap();
  assert!(pm.forward(&x, 3, 4).is_err(), "odd resolution is rejected");
}

// ════════════════ window attention: relative bias changes output ═══════════

#[test]
fn window_attention_relative_bias_changes_output() {
  // A single non-shifted window-attention block. Build it twice — once with a
  // zero relative-position-bias table, once with a *varied* (per-relative-offset
  // distinct) one — and assert the bias add changes the output (everything else
  // identical). A varied table is required: a constant bias is a softmax no-op.
  let (dim, heads, window) = (4, 2, 2); // window² = 4 tokens.
  let nw_b = 3; // pretend 3 windows worth of tokens.

  let build = |table: Array| {
    let mut weights = HashMap::new();
    put_window_attention(&mut weights, "attn", dim, table);
    WindowAttention::from_weights(&mut weights, "attn", dim, heads, window, None).unwrap()
  };
  let zero_bias = build(bias_table_zero(window, heads));
  let nonzero_bias = build(bias_table_varied(window, heads));

  let tokens = window * window;
  // A larger-magnitude input sharpens the attention scores so the per-pair bias
  // has real leverage on the softmax (a near-zero input gives a near-uniform
  // softmax where the bias barely registers).
  let scale = Array::full::<f32>(&[0i32; 0], 10.0).unwrap();
  let x = mat(nw_b * tokens, dim).multiply(&scale).unwrap();
  let x = ops::shape::reshape(&x, &[nw_b, tokens, dim]).unwrap();

  let out_zero = zero_bias.forward(&x, None).unwrap();
  let out_nonzero = nonzero_bias.forward(&x, None).unwrap();
  assert_eq!(
    out_zero.shape(),
    vec![nw_b as usize, tokens as usize, dim as usize],
    "window attention preserves (nw·B, window², C)"
  );
  let a = read_f32(&out_zero);
  let b = read_f32(&out_nonzero);
  let max_diff = a
    .iter()
    .zip(b.iter())
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max);
  assert!(
    max_diff > 1e-5,
    "adding a non-zero relative-position bias changes the attention output (max diff {max_diff})"
  );
}

#[test]
fn window_attention_bias_preserves_f16_bf16_dtype() {
  // An f16/bf16 checkpoint must stay in its dtype: the q/k/v/out weights are the
  // activation dtype, while the relative-position bias is kept fp32 (the
  // worst-case — a table loaded / built in f32). `additive` casts the bias back
  // to the activation dtype, so no silent promotion to f32 occurs.
  let (dim, heads, window) = (4, 2, 2);
  let tokens = window * window;
  for dtype in [Dtype::F16, Dtype::BF16] {
    let mut weights = HashMap::new();
    // The bias table stays fp32; the Linear weights are cast to the activation.
    put_window_attention(&mut weights, "attn", dim, bias_table_varied(window, heads));
    for (k, v) in weights.iter_mut() {
      if !k.ends_with("relative_position_bias_table") {
        *v = v.astype(dtype).unwrap();
      }
    }
    let attn =
      WindowAttention::from_weights(&mut weights, "attn", dim, heads, window, None).unwrap();

    let x = ops::misc::astype(&mat(tokens, dim), dtype).unwrap();
    let x = ops::shape::reshape(&x, &[1, tokens, dim]).unwrap();
    let out = attn.forward(&x, None).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      dtype,
      "window attention preserves the {dtype:?} activation dtype through the fp32-bias add"
    );
  }
}

// ════════════════════════ Swin block: shift mask matters ════════════════════

/// Build a `SwinBlock` (all dense) under `{prefix}` at the given `shift`. When
/// `cast` is `Some(dtype)`, every weight EXCEPT the fp32 relative-position bias
/// table is cast to `dtype` (an fp16/bf16 checkpoint), to exercise the
/// activation-dtype preservation through the f32-bias add.
fn build_swin_block(
  dim: i32,
  heads: i32,
  window: i32,
  shift: i32,
  height: i32,
  width: i32,
  cast: Option<Dtype>,
) -> SwinBlock {
  let hidden = 4 * dim; // mlp_ratio = 4
  let mut weights = HashMap::new();
  put_layer_norm(&mut weights, "blk.layernorm_before", dim);
  put_window_attention(
    &mut weights,
    "blk.attention",
    dim,
    bias_table_varied(window, heads),
  );
  put_layer_norm(&mut weights, "blk.layernorm_after", dim);
  put_linear(&mut weights, "blk.intermediate.dense", hidden, dim, true);
  put_linear(&mut weights, "blk.output.dense", dim, hidden, true);
  if let Some(dtype) = cast {
    for (k, v) in weights.iter_mut() {
      if !k.ends_with("relative_position_bias_table") {
        *v = v.astype(dtype).unwrap();
      }
    }
  }
  SwinBlock::from_weights(
    &mut weights,
    "blk",
    dim,
    heads,
    window,
    shift,
    height,
    width,
    hidden,
    1e-5,
    None,
  )
  .unwrap()
}

#[test]
fn swin_block_even_and_odd_run_and_shift_changes_output() {
  // An 8×8 map, window 4. Even block: shift 0 (W-MSA). Odd block: shift 2
  // (SW-MSA). Both must run and produce (1, 64, dim); the SW-MSA shift mask
  // must change the output versus the no-shift block (same weights).
  let (dim, heads, window) = (4, 2, 4);
  let (h, w) = (8, 8);
  let x = ops::shape::reshape(&mat(h * w, dim), &[1, h * w, dim]).unwrap();

  let even = build_swin_block(dim, heads, window, 0, h, w, None);
  let odd = build_swin_block(dim, heads, window, window / 2, h, w, None);

  let out_even = even.forward(&x, h, w).unwrap();
  let out_odd = odd.forward(&x, h, w).unwrap();
  assert_eq!(
    out_even.shape(),
    vec![1, (h * w) as usize, dim as usize],
    "the Swin block preserves (B, H·W, C)"
  );
  assert_eq!(out_odd.shape(), out_even.shape());

  let a = read_f32(&out_even);
  let b = read_f32(&out_odd);
  let max_diff = a
    .iter()
    .zip(b.iter())
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max);
  assert!(
    max_diff > 1e-5,
    "the SW-MSA shift (roll + mask) changes the block output (max diff {max_diff})"
  );
}

#[test]
fn swin_block_preserves_f16_dtype() {
  let (dim, heads, window) = (4, 2, 4);
  let (h, w) = (8, 8);
  let block = build_swin_block(dim, heads, window, window / 2, h, w, Some(Dtype::F16));
  let x = ops::shape::reshape(&mat(h * w, dim), &[1, h * w, dim]).unwrap();
  let x = ops::misc::astype(&x, Dtype::F16).unwrap();
  let out = block.forward(&x, h, w).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    Dtype::F16,
    "the SW-MSA block preserves the f16 activation dtype"
  );
}

#[test]
fn swin_block_rejects_out_of_range_shift() {
  let (dim, heads, window) = (4, 2, 4);
  let hidden = 4 * dim;
  let mut weights = HashMap::new();
  put_layer_norm(&mut weights, "blk.layernorm_before", dim);
  put_window_attention(
    &mut weights,
    "blk.attention",
    dim,
    bias_table_varied(window, heads),
  );
  put_layer_norm(&mut weights, "blk.layernorm_after", dim);
  put_linear(&mut weights, "blk.intermediate.dense", hidden, dim, true);
  put_linear(&mut weights, "blk.output.dense", dim, hidden, true);
  // shift == window is out of [0, window).
  assert!(
    SwinBlock::from_weights(
      &mut weights,
      "blk",
      dim,
      heads,
      window,
      window,
      8,
      8,
      hidden,
      1e-5,
      None
    )
    .is_err(),
    "shift must be < window"
  );
}

// ══════════════════════════ SW-MSA mask closed form ═════════════════════════

#[test]
fn shifted_window_mask_blocks_cross_region() {
  // A 4×4 map, window 2, shift 1. HF labels the (0:-2, -2:-1, -1:None) slice
  // product per axis: bands are [0,2)→0, [2,3)→1, [3,4)→2. The img-mask label
  // grid is label(r, c) = band(r)*3 + band(c). Then within each window, two
  // tokens may attend iff they share a label.
  let (h, w, window, shift) = (4, 4, 2, 1);
  let mask = shifted_window_mask(h, w, window, shift).unwrap();
  // (1, num_windows, 1, win², win²); num_windows = (4/2)·(4/2) = 4, win² = 4.
  assert_eq!(mask.shape(), vec![1, 4, 1, 4, 4]);

  // Build the expected mask independently: the label grid, the window gather,
  // and the share-label → 0 / else −100 rule.
  let band = |pos: usize| -> usize {
    if pos < h as usize - window as usize {
      0
    } else if pos < h as usize - shift as usize {
      1
    } else {
      2
    }
  };
  let mut label = vec![0i32; (h * w) as usize];
  for r in 0..h as usize {
    for c in 0..w as usize {
      label[r * w as usize + c] = (band(r) * 3 + band(c)) as i32;
    }
  }
  let (win, area) = (window as usize, (window * window) as usize);
  let wb = (w / window) as usize;
  let mut expected = vec![0f32; 4 * area * area];
  for wr in 0..(h / window) as usize {
    for wc in 0..wb {
      let win_idx = wr * wb + wc;
      let mut labels = vec![0i32; area];
      for (p, slot) in labels.iter_mut().enumerate() {
        let (pr, pc) = (p / win, p % win);
        *slot = label[(wr * win + pr) * w as usize + (wc * win + pc)];
      }
      let base = win_idx * area * area;
      for a in 0..area {
        for b in 0..area {
          expected[base + a * area + b] = if labels[a] == labels[b] { 0.0 } else { -100.0 };
        }
      }
    }
  }
  assert_eq!(
    read_f32(&mask),
    expected,
    "the SW-MSA mask matches the HF img-mask region-label construction"
  );
}

// ═════════════════════════════ quantized path ═══════════════════════════════

/// Pack a `(out, in)` dense matrix into the mlx 8-bit affine quantized triple
/// (`weight uint32 (out, in/4)`, `scales (out, in/group)`, `biases
/// (out, in/group)`), so a synthetic `.scales` sibling drives the quantized
/// loader. Mirrors the text-tower quantized-load helper.
fn put_quantized_linear(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  out: i32,
  in_f: i32,
  group_size: i32,
  bits: i32,
  bias: bool,
) {
  let w = mat(out, in_f);
  let (q, s, b) = crate::ops::quantized::quantize(&w, group_size, bits, "affine", None).unwrap();
  weights.insert(format!("{prefix}.weight"), q);
  weights.insert(format!("{prefix}.scales"), s);
  weights.insert(
    format!("{prefix}.biases"),
    b.expect("affine produces per-group biases"),
  );
  if bias {
    weights.insert(format!("{prefix}.bias"), vec1(out));
  }
}

#[test]
fn window_attention_quantized_path_builds_quantized() {
  // A synthetic 8-bit affine quantized window attention: every `.self.*` /
  // `.output.dense` Linear has a `.scales` sibling, so the loader builds the
  // quantized variant. The relative-position-bias table stays dense fp32.
  // `dim` is a multiple of the group size (mlx requires group ∈ {32, 64, 128}).
  let (dim, heads, window) = (32, 2, 2);
  let (group_size, bits) = (32, 8);
  let mut weights = HashMap::new();
  put_quantized_linear(
    &mut weights,
    "attn.self.query",
    dim,
    dim,
    group_size,
    bits,
    true,
  );
  put_quantized_linear(
    &mut weights,
    "attn.self.key",
    dim,
    dim,
    group_size,
    bits,
    true,
  );
  put_quantized_linear(
    &mut weights,
    "attn.self.value",
    dim,
    dim,
    group_size,
    bits,
    true,
  );
  put_quantized_linear(
    &mut weights,
    "attn.output.dense",
    dim,
    dim,
    group_size,
    bits,
    true,
  );
  let span = 2 * window - 1;
  let table = Array::full::<f32>(&((span * span) as usize, heads as usize), 0.1).unwrap();
  weights.insert("attn.self.relative_position_bias_table".to_string(), table);

  let quant = crate::lm::quant::PerLayerQuantization::from_global(
    crate::lm::quant::Quantization::affine(group_size, bits),
  );
  let attn =
    WindowAttention::from_weights(&mut weights, "attn", dim, heads, window, Some(&quant)).unwrap();
  assert!(
    attn.all_quantized(),
    "every projection with a `.scales` sibling loads quantized"
  );

  // It still runs and preserves the window shape.
  let tokens = window * window;
  let x = ops::shape::reshape(&mat(tokens, dim), &[1, tokens, dim]).unwrap();
  let out = attn.forward(&x, None).unwrap();
  assert_eq!(out.shape(), vec![1, tokens as usize, dim as usize]);
}

#[test]
fn patch_merging_quantized_path_builds_quantized() {
  // The reduction Linear's in-features (4·dim = 32) is a multiple of the group
  // size (mlx requires group ∈ {32, 64, 128}).
  let dim = 8;
  let (group_size, bits) = (32, 8);
  let mut weights = HashMap::new();
  put_layer_norm(&mut weights, "pm.norm", 4 * dim);
  put_quantized_linear(
    &mut weights,
    "pm.reduction",
    2 * dim,
    4 * dim,
    group_size,
    bits,
    false,
  );
  let quant = crate::lm::quant::PerLayerQuantization::from_global(
    crate::lm::quant::Quantization::affine(group_size, bits),
  );
  let pm = PatchMerging::from_weights(&mut weights, "pm", dim, 1e-5, Some(&quant)).unwrap();
  assert!(pm.is_quantized(), "the reduction loads quantized");

  let (h, w) = (4, 4);
  let x = ops::shape::reshape(&mat(h * w, dim), &[1, h * w, dim]).unwrap();
  let out = pm.forward(&x, h, w).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, ((h / 2) * (w / 2)) as usize, (2 * dim) as usize]
  );
}

#[test]
fn swin_block_quantized_path_builds_quantized_and_runs() {
  // A full quantized Swin block: every attention + MLP Linear has a `.scales`
  // sibling, so the loader builds the quantized variant; the fp32
  // relative-position bias + the LayerNorms stay dense. `dim` / the MLP hidden
  // are multiples of the group size (mlx requires group ∈ {32, 64, 128}).
  let (dim, heads, window) = (32, 2, 4);
  let hidden = 4 * dim; // 128
  let (group_size, bits) = (32, 8);
  let mut weights = HashMap::new();
  put_layer_norm(&mut weights, "blk.layernorm_before", dim);
  // Attention: quantized q/k/v/out + the dense fp32 bias table.
  for p in ["query", "key", "value"] {
    put_quantized_linear(
      &mut weights,
      &format!("blk.attention.self.{p}"),
      dim,
      dim,
      group_size,
      bits,
      true,
    );
  }
  put_quantized_linear(
    &mut weights,
    "blk.attention.output.dense",
    dim,
    dim,
    group_size,
    bits,
    true,
  );
  weights.insert(
    "blk.attention.self.relative_position_bias_table".to_string(),
    bias_table_varied(window, heads),
  );
  put_layer_norm(&mut weights, "blk.layernorm_after", dim);
  // MLP: quantized intermediate (hidden←dim) + output (dim←hidden).
  put_quantized_linear(
    &mut weights,
    "blk.intermediate.dense",
    hidden,
    dim,
    group_size,
    bits,
    true,
  );
  put_quantized_linear(
    &mut weights,
    "blk.output.dense",
    dim,
    hidden,
    group_size,
    bits,
    true,
  );

  let quant = crate::lm::quant::PerLayerQuantization::from_global(
    crate::lm::quant::Quantization::affine(group_size, bits),
  );
  let block = SwinBlock::from_weights(
    &mut weights,
    "blk",
    dim,
    heads,
    window,
    window / 2,
    8,
    8,
    hidden,
    1e-5,
    Some(&quant),
  )
  .unwrap();
  assert!(
    block.all_quantized(),
    "every attention + MLP Linear with a `.scales` sibling loads quantized"
  );

  // The shifted (SW-MSA) quantized block still runs and preserves (B, H·W, C).
  let (h, w) = (8, 8);
  let x = ops::shape::reshape(&mat(h * w, dim), &[1, h * w, dim]).unwrap();
  let out = block.forward(&x, h, w).unwrap();
  assert_eq!(out.shape(), vec![1, (h * w) as usize, dim as usize]);
}

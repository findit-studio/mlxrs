//! M4 VLM prompt-assembly primitives — splice + locate + multimodal mask
//! integration tests.
//!
//! Reference basis:
//! - python `mlx-vlm/mlx_vlm/prompt_utils.py` (model-agnostic primitives),
//! - python `mlx-vlm/mlx_vlm/utils.py::prepare_inputs` (chunks splice
//!   pattern at lines ~1370–1387),
//! - python
//!   `mlx-vlm/mlx_vlm/models/falcon_ocr/language.py::create_falcon_ocr_mask`
//!   (lines ~120–149) for the causal + bidirectional-within-image mask
//!   formula.
#![cfg(feature = "vlm")]

use mlxrs::{
  Array, Dtype,
  vlm::prompt::{
    MarkerPolicy, MultimodalPrompt, assemble_multimodal_prompt, build_multimodal_mask,
    insert_image_tokens, locate_image_tokens,
  },
};

// ──────────────────────── locate_image_tokens ────────────────────────

#[test]
fn locate_image_tokens_single_run() {
  // One image span of 3 placeholders, surrounded by text.
  let tokens = [10_u32, 99, 99, 99, 20];
  let spans = locate_image_tokens(&tokens, 99);
  assert_eq!(spans, vec![(1, 4)]);
}

#[test]
fn locate_image_tokens_multiple_runs() {
  // Two distinct (non-adjacent) image spans of 3 placeholders each.
  let tokens = [10_u32, 99, 99, 99, 20, 30, 99, 99, 99, 40];
  let spans = locate_image_tokens(&tokens, 99);
  assert_eq!(spans, vec![(1, 4), (6, 9)]);
}

#[test]
fn locate_image_tokens_no_runs() {
  // No image tokens at all.
  let tokens = [10_u32, 20, 30];
  let spans = locate_image_tokens(&tokens, 99);
  assert!(spans.is_empty());
}

#[test]
fn locate_image_tokens_empty_input() {
  // Empty input yields empty Vec (no panics, no underflow).
  let spans = locate_image_tokens(&[], 99);
  assert!(spans.is_empty());
}

#[test]
fn locate_image_tokens_run_at_start_and_end() {
  // Runs flush against both boundaries.
  let tokens = [99_u32, 99, 10, 20, 99];
  let spans = locate_image_tokens(&tokens, 99);
  assert_eq!(spans, vec![(0, 2), (4, 5)]);
}

// ──────────────────────── insert_image_tokens ────────────────────────

#[test]
fn insert_image_tokens_at_marker() {
  // Marker present at index 2; replace with 1 image * 3 tokens.
  let text = [1_u32, 2, 7, 3, 4]; // marker=7, image_token=99
  let out = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::Required).unwrap();
  assert_eq!(out, vec![1, 2, 99, 99, 99, 3, 4]);
}

#[test]
fn insert_image_tokens_rejects_non_contiguous_extra_marker() {
  // Two markers with text between them → rejected: python's `prepare_inputs`
  // splice only supports a single contiguous marker position, and silently
  // leaving the second marker in the buffer would (when
  // image_marker_id == image_token_id) inflate the placeholder count and
  // corrupt vision-feature alignment.
  let text = [1_u32, 7, 2, 7, 3];
  let err = insert_image_tokens(&text, 1, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("non-contiguous"), "unexpected error: {msg}");
}

#[test]
fn insert_image_tokens_contiguous_run_of_markers_replaced_as_one_unit() {
  // The `"<image>" * num_images` chat-template prefix tokenizes to a
  // contiguous run of marker tokens. The splice consumes the entire run
  // as ONE unit and emits `image_count * num_tokens_per_image`
  // placeholders. Here: 2 markers in a contiguous run, image_count=2,
  // num_tokens_per_image=3 → 6 placeholders, original run removed.
  let text = [1_u32, 7, 7, 2];
  let out = insert_image_tokens(&text, 2, 7, 99, 3, MarkerPolicy::Required).unwrap();
  assert_eq!(out, vec![1, 99, 99, 99, 99, 99, 99, 2]);
}

#[test]
fn insert_image_tokens_marker_equals_image_token_id_no_corruption() {
  // When the chat template uses the SAME token as both marker and
  // placeholder (the common mlx-vlm pattern where `<image>` tokenizes to
  // the model's image_token_index), a contiguous run of N markers is
  // replaced by `image_count * num_tokens_per_image` placeholders. The
  // result must NOT contain residual marker tokens that would inflate the
  // image placeholder count.
  let text = [1_u32, 7, 7, 7, 2]; // 3-marker run, marker == image_token_id == 7
  let out = insert_image_tokens(&text, 3, 7, 7, 2, MarkerPolicy::Required).unwrap();
  // Expect exactly 3 * 2 = 6 placeholder tokens, no residual markers.
  assert_eq!(out, vec![1, 7, 7, 7, 7, 7, 7, 2]);
  // Sanity: residual-marker count would have been visible as 6+2=8 sevens;
  // we got 6 sevens (between the 1 and 2).
  let sevens = out.iter().filter(|&&t| t == 7).count();
  assert_eq!(sevens, 6, "residual marker leak detected");
}

#[test]
fn insert_image_tokens_rejects_run_len_too_few() {
  // Marker run has 1 marker but image_count=3 → mismatch: caller/template
  // bug, must fail closed (not silently fan out one marker into 3 images).
  let text = [1_u32, 7, 2];
  let err = insert_image_tokens(&text, 3, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("run length") && msg.contains("does not match image_count"),
    "unexpected error: {msg}"
  );
}

#[test]
fn insert_image_tokens_rejects_run_len_too_many() {
  // Marker run has 3 markers but image_count=1 → mismatch: chat-template
  // emitted extra markers; silently deleting them would hide the producer
  // bug and could corrupt vision-feature alignment.
  let text = [1_u32, 7, 7, 7, 2];
  let err = insert_image_tokens(&text, 1, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("run length") && msg.contains("does not match image_count"),
    "unexpected error: {msg}"
  );
}

#[test]
fn insert_image_tokens_prepended_when_no_marker() {
  // No marker in text + PrependIfAbsent → PROMPT_WITH_IMAGE_TOKEN-style prepend.
  let text = [1_u32, 2, 3];
  let out = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::PrependIfAbsent).unwrap();
  assert_eq!(out, vec![99, 99, 99, 1, 2, 3]);
}

#[test]
fn insert_image_tokens_required_rejects_missing_marker() {
  // No marker in text + Required + image_count>0 → ShapeMismatch.
  // Fails closed against chat-template / tokenizer-version drift that
  // would silently rewrite prompt order under a marker-required template.
  let text = [1_u32, 2, 3];
  let err = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("no image_marker_id") && msg.contains("Required"),
    "unexpected error: {msg}"
  );
}

#[test]
fn insert_image_tokens_required_zero_images_passes_through_even_without_marker() {
  // Required + image_count=0 → passthrough (zero-image short-circuit
  // happens before the marker check). Mirrors the python no-image path.
  let text = [1_u32, 2, 3];
  let out = insert_image_tokens(&text, 0, 7, 99, 3, MarkerPolicy::Required).unwrap();
  assert_eq!(out, vec![1, 2, 3]);
}

#[test]
fn insert_image_tokens_zero_images_passthrough() {
  // image_count=0 → returns text unchanged (marker NOT consumed).
  let text = [1_u32, 7, 2];
  let out = insert_image_tokens(&text, 0, 7, 99, 3, MarkerPolicy::Required).unwrap();
  assert_eq!(out, vec![1, 7, 2]);
}

#[test]
fn insert_image_tokens_multi_image_at_marker() {
  // 2 images → producer emits 2 adjacent markers (`token * num_images`),
  // run consumed as one unit, 3 tokens/image → 6 placeholders total.
  let text = [1_u32, 7, 7, 2];
  let out = insert_image_tokens(&text, 2, 7, 99, 3, MarkerPolicy::Required).unwrap();
  assert_eq!(out, vec![1, 99, 99, 99, 99, 99, 99, 2]);
}

#[test]
fn insert_image_tokens_rejects_zero_tokens_per_image_with_images() {
  // Degenerate config: num_tokens_per_image=0 with image_count>0 → reject.
  // Silently emitting a text-only prompt under this config would drop the
  // caller's images on the floor (downstream attention has no image
  // placeholders to bind to vision features). Fail closed.
  let text = [1_u32, 7, 2];
  let err = insert_image_tokens(&text, 1, 7, 99, 0, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("num_tokens_per_image=0") && msg.contains("degenerate"),
    "unexpected error: {msg}"
  );
}

#[test]
fn insert_image_tokens_zero_tokens_per_image_with_zero_images_passes() {
  // image_count=0 AND num_tokens_per_image=0 → passthrough (the zero-image
  // short-circuit happens before the zero-width guard, and there are no
  // images to drop).
  let text = [1_u32, 7, 2];
  let out = insert_image_tokens(&text, 0, 7, 99, 0, MarkerPolicy::Required).unwrap();
  assert_eq!(out, vec![1, 7, 2]);
}

#[test]
fn insert_image_tokens_rejects_overflow_image_count() {
  // image_count * num_tokens_per_image overflows usize → ShapeMismatch,
  // no panic, no OOM-abort. Guards public primitives forwarding caller-
  // supplied request/config values.
  let text = [1_u32, 7, 2];
  let err = insert_image_tokens(&text, usize::MAX, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("overflows usize"), "unexpected error: {msg}");
}

#[test]
fn insert_image_tokens_rejects_overflow_tokens_per_image() {
  // Same overflow guard, symmetric in the other multiplicand.
  let text = [1_u32, 7, 2];
  let err = insert_image_tokens(&text, 2, 7, 99, usize::MAX, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("overflows usize"), "unexpected error: {msg}");
}

// ──────────────────────── build_multimodal_mask ────────────────────────

/// Expand `[1, 1, T, T]` bool mask into a (T, T) row-major Vec<bool> for assertions.
fn flatten_mask(mask: &mut Array, t: usize) -> Vec<bool> {
  assert_eq!(mask.shape(), vec![1, 1, t, t]);
  assert_eq!(mask.dtype().unwrap(), Dtype::Bool);
  mask.to_vec::<bool>().unwrap()
}

#[test]
fn build_multimodal_mask_causal_text_only() {
  // No image spans → pure lower-triangular (k <= q).
  let mut mask = build_multimodal_mask(4, &[]).unwrap();
  let v = flatten_mask(&mut mask, 4);
  for q in 0..4 {
    for k in 0..4 {
      assert_eq!(v[q * 4 + k], k <= q, "q={q} k={k}");
    }
  }
}

#[test]
fn build_multimodal_mask_bidirectional_within_image_span() {
  // T=6, one image span at [2,5) (3 placeholders).
  // Inside the span, every (q,k) attends regardless of order;
  // text<->image positions follow causal.
  let mut mask = build_multimodal_mask(6, &[(2, 5)]).unwrap();
  let v = flatten_mask(&mut mask, 6);
  for q in 0..6 {
    let q_in = (2..5).contains(&q);
    for k in 0..6 {
      let k_in = (2..5).contains(&k);
      let causal = k <= q;
      let same_image = q_in && k_in;
      let expect = causal || same_image;
      assert_eq!(v[q * 6 + k], expect, "q={q} k={k}");
    }
  }
}

#[test]
fn build_multimodal_mask_causal_across_image_spans() {
  // T=8, two image spans [1,3) and [5,7).
  // Within each span: bidirectional. Across spans: causal.
  let spans: &[(usize, usize)] = &[(1, 3), (5, 7)];
  let mut mask = build_multimodal_mask(8, spans).unwrap();
  let v = flatten_mask(&mut mask, 8);

  fn span_id(p: usize, spans: &[(usize, usize)]) -> Option<usize> {
    spans.iter().enumerate().find_map(
      |(i, &(s, e))| {
        if (s..e).contains(&p) { Some(i) } else { None }
      },
    )
  }
  for q in 0..8 {
    for k in 0..8 {
      let causal = k <= q;
      let same_image = match (span_id(q, spans), span_id(k, spans)) {
        (Some(qi), Some(ki)) => qi == ki,
        _ => false,
      };
      assert_eq!(v[q * 8 + k], causal || same_image, "q={q} k={k}");
    }
  }
}

#[test]
fn build_multimodal_mask_empty_seq_yields_zero_zero_mask() {
  // seq_len=0 → shape [1, 1, 0, 0] (no panics, no allocation of T*T buffer;
  // matches upstream create_falcon_ocr_mask rank-4 contract).
  let mut mask = build_multimodal_mask(0, &[]).unwrap();
  assert_eq!(mask.shape(), vec![1, 1, 0, 0]);
  assert_eq!(mask.to_vec::<bool>().unwrap(), Vec::<bool>::new());
}

#[test]
fn build_multimodal_mask_rejects_non_empty_spans_with_zero_seq_len() {
  // seq_len=0 + non-empty image_spans → inconsistent state, must error
  // (fail-closed; the documented validation contract says out-of-bounds /
  // non-empty spans must error, including under the zero-length fast path).
  let err = build_multimodal_mask(0, &[(0, 1)]).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("seq_len=0") && msg.contains("image_spans"),
    "unexpected error: {msg}"
  );
}

#[test]
fn build_multimodal_mask_rejects_empty_span_with_zero_seq_len() {
  // seq_len=0 + a degenerate (0,0) span → same fail-closed treatment as the
  // non-empty-spans case (the seq_len=0 path doesn't get to silently accept
  // any span entry).
  let err = build_multimodal_mask(0, &[(0, 0)]).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("seq_len=0") && msg.contains("image_spans"),
    "unexpected error: {msg}"
  );
}

#[test]
fn build_multimodal_mask_rejects_empty_span() {
  // (3, 3) is empty (start>=end) → ShapeMismatch, no panic.
  let err = build_multimodal_mask(5, &[(3, 3)]).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("empty"), "unexpected error: {msg}");
}

#[test]
fn build_multimodal_mask_rejects_out_of_bounds_span() {
  // span end exceeds seq_len.
  let err = build_multimodal_mask(4, &[(2, 5)]).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("exceeds seq_len"), "unexpected error: {msg}");
}

#[test]
fn build_multimodal_mask_rejects_overlapping_spans() {
  // Spans (1,4) and (3,5) overlap at position 3.
  let err = build_multimodal_mask(6, &[(1, 4), (3, 5)]).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("overlap"), "unexpected error: {msg}");
}

// ──────────────────────── assemble_multimodal_prompt ────────────────────────

#[test]
fn assemble_multimodal_prompt_roundtrip() {
  // 2 images, 3 placeholders each → the chat-template producer emits 2
  // adjacent markers (`token * num_images`); the splice consumes the run
  // and emits 6 placeholders, with per-image span boundaries `[(2,5), (5,8)]`
  // (NOT a single collapsed (2,8)). That lets the mask treat the two
  // images as distinct blocks: bidirectional within each image, causal
  // across images.
  let text = [1_u32, 2, 7, 7, 3];
  let p: MultimodalPrompt =
    assemble_multimodal_prompt(&text, 2, 7, 99, 3, MarkerPolicy::Required).unwrap();

  assert_eq!(p.tokens, vec![1, 2, 99, 99, 99, 99, 99, 99, 3]);
  assert_eq!(p.image_spans, vec![(2, 5), (5, 8)]);

  let t = p.tokens.len();
  let mut mask = p.attention_mask;
  assert_eq!(mask.shape(), vec![1, 1, t, t]);
  let v = mask.to_vec::<bool>().unwrap();
  // text@0 → only attends to itself (causal).
  assert!(v[0] && !v[1]);
  // image1 q=3 → attends to image1 k=2,3,4 (bidirectional within image1)
  // AND text@0,1 (causal); but NOT to image2 k=5,6,7 (future, different
  // image — the causal-across-images guarantee).
  for k in 0..=4 {
    assert!(
      v[3 * t + k],
      "image1 q=3 should attend to k={k} (causal+same-image)"
    );
  }
  for k in 5..=7 {
    assert!(
      !v[3 * t + k],
      "image1 q=3 must NOT attend to image2 k={k} (causal-across-images leak)"
    );
  }
  // text@8 → attends to everything prior (last token, causal allows all).
  for k in 0..t {
    assert!(v[8 * t + k], "last token q=8 should attend to all k={k}");
  }
}

#[test]
fn assemble_multimodal_prompt_causal_across_images_no_leak() {
  // Regression: with 3 adjacent images (chat-template emits 3 adjacent
  // markers), image_k must not see image_{k+1..}. Without per-image span
  // preservation the splice would collapse all placeholders into one big
  // bidirectional block and image_0's queries would leak forward to
  // image_2's keys.
  let text = [1_u32, 7, 7, 7, 2];
  let p = assemble_multimodal_prompt(&text, 3, 7, 99, 2, MarkerPolicy::Required).unwrap();
  // Splice: [1, 99,99, 99,99, 99,99, 2] → length 8.
  assert_eq!(p.tokens, vec![1, 99, 99, 99, 99, 99, 99, 2]);
  assert_eq!(p.image_spans, vec![(1, 3), (3, 5), (5, 7)]);
  let t = p.tokens.len();
  let mut mask = p.attention_mask;
  let v = mask.to_vec::<bool>().unwrap();

  // image1 q=1 → attends to image1 k=1,2 (same image) and text k=0
  // (causal). MUST NOT attend to image2 (k=3,4) or image3 (k=5,6).
  for k in 0..=2 {
    assert!(v[t + k], "image1 q=1 should attend to k={k}");
  }
  for k in 3..=6 {
    assert!(
      !v[t + k],
      "image1 q=1 must NOT attend to k={k} (across-image leak)"
    );
  }
  // image2 q=3 → attends to text k=0, image1 k=1,2 (causal), image2 k=3,4
  // (same image). MUST NOT attend to image3 (k=5,6).
  for k in 0..=4 {
    assert!(v[3 * t + k], "image2 q=3 should attend to k={k}");
  }
  for k in 5..=6 {
    assert!(
      !v[3 * t + k],
      "image2 q=3 must NOT attend to image3 k={k} (across-image leak)"
    );
  }
  // image3 q=5 → attends to all prior + image3's own (5,6); since q=5 is
  // the first in image3 and all earlier tokens are causally visible.
  for k in 0..=6 {
    assert!(v[5 * t + k], "image3 q=5 should attend to k={k}");
  }
}

#[test]
fn assemble_multimodal_prompt_text_only_passthrough() {
  // image_count=0 → tokens unchanged, no spans, pure causal mask.
  let text = [1_u32, 2, 3];
  let p = assemble_multimodal_prompt(&text, 0, 7, 99, 3, MarkerPolicy::Required).unwrap();
  assert_eq!(p.tokens, vec![1, 2, 3]);
  assert!(p.image_spans.is_empty());
  let mut mask = p.attention_mask;
  assert_eq!(mask.shape(), vec![1, 1, 3, 3]);
  let v = mask.to_vec::<bool>().unwrap();
  for q in 0..3 {
    for k in 0..3 {
      assert_eq!(v[q * 3 + k], k <= q);
    }
  }
}

#[test]
fn assemble_multimodal_prompt_rejects_final_len_above_i32_max_early() {
  // EARLY-rejection regression: with image_count just above i32::MAX and
  // num_tokens_per_image=1, `image_count * num_tokens_per_image` does NOT
  // overflow usize on 64-bit platforms (and on 32-bit platforms this case
  // can't be expressed), but the resulting final_len would exceed
  // `i32::MAX` — mlx's dimension limit. The assembler must reject this
  // BEFORE `insert_image_tokens` allocates the host buffer; otherwise we
  // OOM-abort on a request the mask would have rejected anyway. Skipped
  // on non-64-bit targets where the precondition cannot hold.
  if usize::BITS < 64 {
    return;
  }
  let text = [1_u32, 7, 2];
  let huge = (i32::MAX as usize) + 1;
  let err = assemble_multimodal_prompt(&text, huge, 7, 99, 1, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("exceeds i32::MAX"), "unexpected error: {msg}");
}

#[test]
fn assemble_multimodal_prompt_no_marker_prepends_then_spans_at_start() {
  // No marker + PrependIfAbsent → prepend; image span starts at index 0.
  let text = [1_u32, 2, 3];
  let p = assemble_multimodal_prompt(&text, 1, 7, 99, 2, MarkerPolicy::PrependIfAbsent).unwrap();
  assert_eq!(p.tokens, vec![99, 99, 1, 2, 3]);
  assert_eq!(p.image_spans, vec![(0, 2)]);
  let mut mask = p.attention_mask;
  assert_eq!(mask.shape(), vec![1, 1, 5, 5]);
  let v = mask.to_vec::<bool>().unwrap();
  // image@0 attends to image@1 (bidirectional in same span) even though k>q.
  assert!(v[1], "image@0 should attend to image@1 (bidirectional)");
  // text@2 attends to image@0,1 (causal).
  assert!(v[2 * 5] && v[2 * 5 + 1]);
}

#[test]
fn assemble_multimodal_prompt_required_rejects_missing_marker() {
  // No marker + Required + image_count>0 → ShapeMismatch (fails closed
  // against chat-template drift).
  let text = [1_u32, 2, 3];
  let err = assemble_multimodal_prompt(&text, 1, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("no image_marker_id") && msg.contains("Required"),
    "unexpected error: {msg}"
  );
}

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
    build_multimodal_mask_with_past, insert_image_tokens, locate_image_tokens,
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
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for non-contiguous marker, got {err:?}"
  );
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
  match &err {
    Error::LengthMismatch(p) => {
      assert_eq!(p.expected(), 3, "expected = image_count");
      assert_eq!(p.actual(), 1, "actual = run_len");
    }
    _ => panic!("expected LengthMismatch, got {err:?}"),
  }
}

#[test]
fn insert_image_tokens_rejects_run_len_too_many() {
  // Marker run has 3 markers but image_count=1 → mismatch: chat-template
  // emitted extra markers; silently deleting them would hide the producer
  // bug and could corrupt vision-feature alignment.
  let text = [1_u32, 7, 7, 7, 2];
  let err = insert_image_tokens(&text, 1, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  match &err {
    Error::LengthMismatch(p) => {
      assert_eq!(p.expected(), 1, "expected = image_count");
      assert_eq!(p.actual(), 3, "actual = run_len");
    }
    _ => panic!("expected LengthMismatch, got {err:?}"),
  }
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
  // No marker in text + Required + image_count>0 → MissingField.
  // Fails closed against chat-template / tokenizer-version drift that
  // would silently rewrite prompt order under a marker-required template.
  let text = [1_u32, 2, 3];
  let err = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
  assert!(
    matches!(err, Error::MissingField(_)),
    "expected MissingField under MarkerPolicy::Required, got {err:?}"
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
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for degenerate config, got {err:?}"
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
  // image_count * num_tokens_per_image overflows usize → ArithmeticOverflow,
  // no panic, no OOM-abort. Guards public primitives forwarding caller-
  // supplied request/config values.
  let text = [1_u32, 7, 2];
  let err = insert_image_tokens(&text, usize::MAX, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  assert!(
    matches!(err, Error::ArithmeticOverflow(_)),
    "expected ArithmeticOverflow, got {err:?}"
  );
}

#[test]
fn insert_image_tokens_rejects_overflow_tokens_per_image() {
  // Same overflow guard, symmetric in the other multiplicand.
  let text = [1_u32, 7, 2];
  let err = insert_image_tokens(&text, 2, 7, 99, usize::MAX, MarkerPolicy::Required).unwrap_err();
  assert!(
    matches!(err, Error::ArithmeticOverflow(_)),
    "expected ArithmeticOverflow, got {err:?}"
  );
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
fn build_multimodal_mask_with_past_zero_offset_equals_base() {
  // `build_multimodal_mask_with_past(seq, 0, spans)` is the
  // delegation target of `build_multimodal_mask(seq, spans)` — byte
  // identical at past_len=0.
  let mut base = build_multimodal_mask(6, &[(2, 5)]).unwrap();
  let mut with_past = build_multimodal_mask_with_past(6, 0, &[(2, 5)]).unwrap();
  assert_eq!(with_past.shape(), vec![1, 1, 6, 6]);
  assert_eq!(flatten_mask(&mut base, 6), flatten_mask(&mut with_past, 6));
}

#[test]
fn build_multimodal_mask_with_past_chunk_after_prefix() {
  // A chunk of seq_len=3 at cache_offset=4 (4 cached past tokens)
  // with a chunk-local image span (0,3) — the WHOLE chunk is one image.
  // Mask shape is [1, 1, 3, 7] (3 queries × (4 past + 3 current) keys).
  //   - past keys (k < 4): ALWAYS attend (causal — past precedes chunk).
  //   - current keys (k >= 4): chunk-local k'=k-4; attend iff k'<=q
  //     (causal) OR same image span (here the whole chunk is one span,
  //     so all 3×3 current cells attend bidirectionally).
  let past = 4usize;
  let seq = 3usize;
  let mut mask = build_multimodal_mask_with_past(seq, past, &[(0, 3)]).unwrap();
  assert_eq!(mask.shape(), vec![1, 1, seq, past + seq]);
  let v: Vec<bool> = mask.to_vec().unwrap();
  let total_keys = past + seq;
  for q in 0..seq {
    for k in 0..total_keys {
      let expect = if k < past {
        true // past always attended
      } else {
        // whole chunk is one image span ⇒ bidirectional within chunk
        true
      };
      assert_eq!(v[q * total_keys + k], expect, "q={q} k={k} past={past}");
    }
  }
}

#[test]
fn build_multimodal_mask_with_past_text_chunk_after_prefix() {
  // A pure-text chunk (no spans) of seq_len=3 at cache_offset=2.
  // Past keys always attend; current keys are causal (k'<=q). No
  // bidirectional block.
  let past = 2usize;
  let seq = 3usize;
  let mut mask = build_multimodal_mask_with_past(seq, past, &[]).unwrap();
  assert_eq!(mask.shape(), vec![1, 1, seq, past + seq]);
  let v: Vec<bool> = mask.to_vec().unwrap();
  let total_keys = past + seq;
  for q in 0..seq {
    for k in 0..total_keys {
      let expect = if k < past { true } else { (k - past) <= q };
      assert_eq!(v[q * total_keys + k], expect, "q={q} k={k}");
    }
  }
}

#[test]
fn build_multimodal_mask_with_past_rejects_chunk_local_span_out_of_bounds() {
  // A chunk-local span whose end exceeds seq_len is rejected (the
  // caller's chunk-local shift must keep spans inside [0, seq_len)).
  let err = build_multimodal_mask_with_past(3, 4, &[(1, 5)]).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for chunk-local span out of bounds, got: {err:?}"
  );
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
  // (3, 3) is empty (start>=end) → InvariantViolation, no panic.
  let err = build_multimodal_mask(5, &[(3, 3)]).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for empty span, got {err:?}"
  );
}

#[test]
fn build_multimodal_mask_rejects_out_of_bounds_span() {
  // span end exceeds seq_len → OutOfRange.
  let err = build_multimodal_mask(4, &[(2, 5)]).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for span end > seq_len, got {err:?}"
  );
}

#[test]
fn build_multimodal_mask_rejects_overlapping_spans() {
  // Spans (1,4) and (3,5) overlap at position 3 → InvariantViolation.
  let err = build_multimodal_mask(6, &[(1, 4), (3, 5)]).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for overlapping spans, got {err:?}"
  );
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
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange (i32::MAX dim limit), got {err:?}"
  );
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
  // No marker + Required + image_count>0 → MissingField (fails closed
  // against chat-template drift).
  let text = [1_u32, 2, 3];
  let err = assemble_multimodal_prompt(&text, 1, 7, 99, 2, MarkerPolicy::Required).unwrap_err();
  assert!(
    matches!(err, Error::MissingField(_)),
    "expected MissingField under MarkerPolicy::Required, got {err:?}"
  );
}

// =================================================================
// MessageFormat / MessageFormatter / get_message_json tests
// =================================================================
//
// Reference: `mlx-vlm/mlx_vlm/prompt_utils.py` lines 6–23 (enum), 27–89
// (`MODEL_CONFIG`), 92–100 (`SINGLE_IMAGE_ONLY_MODELS`), 151–189
// (`MessageBuilder`), 192–441 (`MessageFormatter`), 444–480
// (`get_message_json`).

use mlxrs::{
  Error,
  vlm::prompt::{
    ContentItem, FormatOpts, FormattedMessage, MAX_MESSAGE_FORMAT_ITEMS, MESSAGE_FORMAT_VARIANTS,
    MODEL_CONFIG, MessageBuilder, MessageContent, MessageFormat, MessageFormatter,
    SINGLE_IMAGE_ONLY_MODELS, get_message_json,
  },
};

// ──────────────────────── MessageFormat 18-variant table ────────────────────

#[test]
fn message_format_18_variants_table() {
  // Counting `prompt_utils.py` lines 9–23 yields exactly 15 enum variants
  // (the test name's "18" is an earlier over-count). Assert the count
  // matches the python ref AND each declared variant string maps to the
  // right Rust variant.
  assert_eq!(
    MESSAGE_FORMAT_VARIANTS.len(),
    15,
    "MessageFormat variant count does not match python ref (15)"
  );

  // Spot-check each variant has a unique Debug name (no aliases).
  use std::collections::HashSet;
  let names: HashSet<String> = MESSAGE_FORMAT_VARIANTS
    .iter()
    .map(|v| format!("{v:?}"))
    .collect();
  assert_eq!(names.len(), 15, "duplicate MessageFormat variants");
}

#[test]
fn model_config_size_matches_python_ref() {
  // The python `MODEL_CONFIG` dict has 58 entries
  // (verified via `grep -c 'MessageFormat\.' prompt_utils.py` over
  // lines 27–89). Assert the Rust table matches.
  assert_eq!(
    MODEL_CONFIG.len(),
    58,
    "MODEL_CONFIG size does not match python ref"
  );
}

#[test]
fn model_config_is_sorted_lexicographically() {
  // MessageFormatter::for_model uses binary search; assert the table
  // is sorted bytewise.
  for w in MODEL_CONFIG.windows(2) {
    assert!(
      w[0].0 < w[1].0,
      "MODEL_CONFIG not sorted: {:?} >= {:?}",
      w[0].0,
      w[1].0
    );
  }
}

#[test]
fn single_image_only_models_set() {
  // Faithful port of python lines 92–100 (7 entries).
  assert_eq!(SINGLE_IMAGE_ONLY_MODELS.len(), 7);
  // Spot-check key entries.
  assert!(SINGLE_IMAGE_ONLY_MODELS.contains(&"paligemma"));
  assert!(SINGLE_IMAGE_ONLY_MODELS.contains(&"mllama"));
  assert!(SINGLE_IMAGE_ONLY_MODELS.contains(&"falcon_ocr"));
}

// ──────────────────────── MessageFormatter::for_model ───────────────────────

#[test]
fn message_formatter_for_model_picks_right_variant() {
  // Python lines 27–89:
  //   "qwen2_vl"   -> LIST_WITH_IMAGE
  //   "qwen2_5_vl" -> LIST_WITH_IMAGE_FIRST
  //   "paligemma"  -> PROMPT_WITH_IMAGE_TOKEN
  //   "gemma3"     -> START_IMAGE_TOKEN
  //   "phi3_v"     -> NUMBERED_IMAGE_TOKENS
  //   "florence2"  -> PROMPT_ONLY
  let cases = [
    ("qwen2_vl", MessageFormat::ListWithImage),
    ("qwen2_5_vl", MessageFormat::ListWithImageFirst),
    ("paligemma", MessageFormat::PromptWithImageToken),
    ("gemma3", MessageFormat::StartImageToken),
    ("phi3_v", MessageFormat::NumberedImageTokens),
    ("florence2", MessageFormat::PromptOnly),
    ("ernie4_5_moe_vl", MessageFormat::ListWithImageUrlFirst),
    ("jvlm", MessageFormat::ImageTokenPipe),
    ("internvl_chat", MessageFormat::ListWithImageType),
    ("minicpmo", MessageFormat::ImageToken),
    ("bunny-llama", MessageFormat::ImageTokenNewline),
  ];
  for (model, expected) in cases {
    let f = MessageFormatter::for_model(model).unwrap_or_else(|e| {
      panic!("MessageFormatter::for_model({model}) failed: {e}");
    });
    assert_eq!(f.format_type, expected, "for_model({model})");
    assert_eq!(f.model_name, model.to_lowercase());
  }
}

#[test]
fn message_formatter_for_model_lowercases_input() {
  // Python line 196: `model_name.lower()`.
  let f = MessageFormatter::for_model("Qwen2_VL").unwrap();
  assert_eq!(f.model_name, "qwen2_vl");
  assert_eq!(f.format_type, MessageFormat::ListWithImage);
}

#[test]
fn message_formatter_for_model_unsupported_errors() {
  // Python lines 198–199: `ValueError(f"Unsupported model: {model_name}")`.
  let err = MessageFormatter::for_model("not_a_real_model").unwrap_err();
  match &err {
    Error::MissingKey(p) => {
      assert_eq!(p.key(), "not_a_real_model");
    }
    _ => panic!("expected MissingKey for unsupported model, got {err:?}"),
  }
}

// ──────────────────────── get_message_json shapes ────────────────────────

#[test]
fn get_message_json_text_only_shape() {
  // PROMPT_ONLY: content is the bare prompt string. Florence2 uses
  // this format.
  let out = get_message_json(
    "florence2",
    "describe this",
    Some(&FormatOpts {
      num_images: 0,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::String(s) => assert_eq!(s, "describe this"),
    other => panic!("expected String branch, got: {other:?}"),
  }
}

#[test]
fn get_message_json_image_content_embedded() {
  // qwen2_vl uses LIST_WITH_IMAGE: content = [text, image] (image
  // last, because image_first=False is the default for this variant).
  let out = get_message_json(
    "qwen2_vl",
    "describe this",
    Some(&FormatOpts {
      num_images: 1,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::Message(m) => {
      assert_eq!(m.role, "user");
      let items = match m.content {
        MessageContent::Items(v) => v,
        other => panic!("expected Items branch, got: {other:?}"),
      };
      assert_eq!(items.len(), 2);
      // First is text (LIST_WITH_IMAGE puts text first), second is image.
      match &items[0] {
        ContentItem::Text { text } => assert_eq!(text, "describe this"),
        other => panic!("expected Text at [0], got: {other:?}"),
      }
      assert_eq!(items[1], ContentItem::Image);
    }
    other => panic!("expected Message branch, got: {other:?}"),
  }
}

#[test]
fn get_message_json_image_first_shape() {
  // qwen2_5_vl uses LIST_WITH_IMAGE_FIRST: content = [image, text].
  let out = get_message_json(
    "qwen2_5_vl",
    "describe this",
    Some(&FormatOpts {
      num_images: 1,
      ..Default::default()
    }),
  )
  .unwrap();
  let items = match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Items(v) => v,
      _ => panic!(),
    },
    _ => panic!(),
  };
  assert_eq!(items.len(), 2);
  assert_eq!(items[0], ContentItem::Image);
  match &items[1] {
    ContentItem::Text { text } => assert_eq!(text, "describe this"),
    other => panic!("expected Text at [1], got: {other:?}"),
  }
}

#[test]
fn get_message_json_image_url_first_shape() {
  // ernie4_5_moe_vl uses LIST_WITH_IMAGE_URL_FIRST: content = [image_url, text].
  let out = get_message_json(
    "ernie4_5_moe_vl",
    "describe",
    Some(&FormatOpts {
      num_images: 1,
      ..Default::default()
    }),
  )
  .unwrap();
  let items = match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Items(v) => v,
      _ => panic!(),
    },
    _ => panic!(),
  };
  assert_eq!(items[0], ContentItem::ImageUrl);
}

#[test]
fn get_message_json_image_token_inline() {
  // minicpmo uses IMAGE_TOKEN: content is a flat string with
  // `<image>` prefixed (one per image). num_audios=0 to suppress the
  // python default-1 audio prefix (the python ref's default is also 1
  // — see line 208).
  let out = get_message_json(
    "minicpmo",
    "what is this",
    Some(&FormatOpts {
      num_images: 2,
      num_audios: 0,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Text(s) => assert_eq!(s, "<image><image>what is this"),
      other => panic!("expected Text branch, got: {other:?}"),
    },
    _ => panic!(),
  }
}

#[test]
fn get_message_json_start_image_token_suffixed() {
  // gemma3 uses START_IMAGE_TOKEN: token is APPENDED, not prepended
  // (python `image_first=False` at line 258).
  let out = get_message_json(
    "gemma3",
    "what is this",
    Some(&FormatOpts {
      num_images: 1,
      num_audios: 0,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Text(s) => assert_eq!(s, "what is this<start_of_image>"),
      _ => panic!(),
    },
    _ => panic!(),
  }
}

#[test]
fn get_message_json_numbered_image_tokens_phi() {
  // phi3_v uses NUMBERED_IMAGE_TOKENS: `<|image_1|><|image_2|>...prompt`.
  // Adding 2 audios should also prepend `<|audio_1|><|audio_2|>`.
  let out = get_message_json(
    "phi3_v",
    "describe",
    Some(&FormatOpts {
      num_images: 2,
      num_audios: 1,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::Message(m) => match m.content {
      // Order: images then audio, then prompt (python lines 391–403).
      MessageContent::Text(s) => assert_eq!(s, "<|image_1|><|image_2|><|audio_1|>describe"),
      _ => panic!(),
    },
    _ => panic!(),
  }
}

#[test]
fn get_message_json_prompt_with_image_token_paligemma() {
  // paligemma uses PROMPT_WITH_IMAGE_TOKEN: returns a bare String,
  // `"<image>"*N + prompt`. paligemma is in SINGLE_IMAGE_ONLY_MODELS
  // so we use num_images=1.
  let out = get_message_json(
    "paligemma",
    "describe",
    Some(&FormatOpts {
      num_images: 1,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::String(s) => assert_eq!(s, "<image>describe"),
    other => panic!("expected String branch, got: {other:?}"),
  }
}

#[test]
fn get_message_json_single_image_only_rejects_multi() {
  // paligemma is in SINGLE_IMAGE_ONLY_MODELS; num_images=2 should error.
  let err = get_message_json(
    "paligemma",
    "describe",
    Some(&FormatOpts {
      num_images: 2,
      ..Default::default()
    }),
  )
  .unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for SINGLE_IMAGE_ONLY model, got {err:?}"
  );
}

#[test]
fn get_message_json_video_message_qwen2_vl() {
  // qwen2_vl normally uses LIST_WITH_IMAGE, but the video special-case
  // (python lines 221–231) routes to _format_video_message when
  // opts.video is non-empty.
  let out = get_message_json(
    "qwen2_vl",
    "what's in this video",
    Some(&FormatOpts {
      num_images: 0,
      video: vec!["video.mp4".to_string()],
      max_pixels: 224 * 224,
      fps: vec![],
      ..Default::default()
    }),
  )
  .unwrap();
  let items = match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Items(v) => v,
      _ => panic!(),
    },
    _ => panic!(),
  };
  assert_eq!(items.len(), 2);
  // First: video item.
  match &items[0] {
    ContentItem::Video {
      video,
      max_pixels,
      fps,
    } => {
      assert_eq!(video, "video.mp4");
      assert_eq!(*max_pixels, 224 * 224);
      assert_eq!(*fps, 1);
    }
    other => panic!("expected Video at [0], got: {other:?}"),
  }
  // Second: text item.
  match &items[1] {
    ContentItem::Text { text } => assert_eq!(text, "what's in this video"),
    other => panic!("expected Text at [1], got: {other:?}"),
  }
}

#[test]
fn message_formatter_assistant_collapses_to_string() {
  // _format_list_with_image_type: assistant role collapses content to a
  // flat string (python lines 343–346). Uses internvl_chat
  // (LIST_WITH_IMAGE_TYPE).
  let f = MessageFormatter::for_model("internvl_chat").unwrap();
  let out = f
    .format_message(
      "the answer is 42",
      &FormatOpts {
        role: "assistant".to_string(),
        num_images: 0,
        ..Default::default()
      },
    )
    .unwrap();
  match out {
    FormattedMessage::Message(m) => {
      assert_eq!(m.role, "assistant");
      match m.content {
        MessageContent::Text(s) => assert_eq!(s, "the answer is 42"),
        other => panic!("expected Text branch, got: {other:?}"),
      }
    }
    _ => panic!(),
  }
}

#[test]
fn message_formatter_skip_image_token_suppresses_images() {
  // skip_image_token=true should suppress the image entries — used by
  // apply_chat_template for non-target turns in a multi-message list.
  let f = MessageFormatter::for_model("qwen2_vl").unwrap();
  let out = f
    .format_message(
      "describe",
      &FormatOpts {
        num_images: 2,
        skip_image_token: true,
        ..Default::default()
      },
    )
    .unwrap();
  let items = match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Items(v) => v,
      _ => panic!(),
    },
    _ => panic!(),
  };
  // No images, just the text.
  assert_eq!(items.len(), 1);
  match &items[0] {
    ContentItem::Text { .. } => (),
    other => panic!("expected Text only, got: {other:?}"),
  }
}

#[test]
fn message_builder_audio_and_video_messages() {
  // Spot-check the static MessageBuilder constructors.
  assert_eq!(MessageBuilder::image_message(), ContentItem::Image);
  assert_eq!(MessageBuilder::image_url_message(), ContentItem::ImageUrl);
  assert_eq!(MessageBuilder::audio_message(), ContentItem::Audio);
  match MessageBuilder::text_message("hello") {
    ContentItem::Text { text } => assert_eq!(text, "hello"),
    other => panic!("got {other:?}"),
  }
  match MessageBuilder::content_message("world") {
    ContentItem::ContentText { text } => assert_eq!(text, "world"),
    other => panic!("got {other:?}"),
  }
  match MessageBuilder::video_message("v.mp4", 1234, 5) {
    ContentItem::Video {
      video,
      max_pixels,
      fps,
    } => {
      assert_eq!(video, "v.mp4");
      assert_eq!(max_pixels, 1234);
      assert_eq!(fps, 5);
    }
    other => panic!("got {other:?}"),
  }
}

#[test]
fn message_formatter_list_with_image_type_text_image_last_variant() {
  // LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST is declared in the python
  // enum (line 14) but no model in MODEL_CONFIG selects it; the Rust
  // port exposes it via the dispatcher so a caller can use it
  // explicitly. Construct via direct method call (no model lookup).
  let f = MessageFormatter {
    model_name: "synthetic".to_string(),
    format_type: MessageFormat::ListWithImageTypeTextImageLast,
  };
  let out = f
    .format_message(
      "what is this",
      &FormatOpts {
        num_images: 1,
        num_audios: 0,
        ..Default::default()
      },
    )
    .unwrap();
  let items = match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Items(v) => v,
      _ => panic!(),
    },
    _ => panic!(),
  };
  // image_first=false → text first, then image (LIST_WITH_IMAGE_TYPE
  // path uses text_message because message_type=Text).
  assert_eq!(items.len(), 2);
  match &items[0] {
    ContentItem::Text { text } => assert_eq!(text, "what is this"),
    other => panic!("expected Text at [0], got: {other:?}"),
  }
  assert_eq!(items[1], ContentItem::Image);
}

// =================================================================
// Regressions: FormatOpts split, allocation hardening,
// attention_mask threading.
// =================================================================

// ──────────────────────── Defaults split ───────────────────────

#[test]
fn get_message_json_text_only_with_qwen2_vl_default_emits_no_image() {
  // FormatOpts::get_message_default (the public-API defaults at python
  // prompt_utils.py:444–480, num_images=0 num_audios=0) MUST be used
  // by `get_message_json(_, _, None)`, so a text-only call to a model
  // whose normal format would inject an image entry (qwen2_vl uses
  // LIST_WITH_IMAGE) emits NO image-content entry. Forwarding
  // `&FormatOpts::default()` (num_images=1, the formatter-internal
  // default) would spuriously inject an image entry into a text-only
  // call; the public-API defaults avoid that.
  let out = get_message_json("qwen2_vl", "describe this", None).unwrap();
  match out {
    FormattedMessage::Message(m) => {
      assert_eq!(m.role, "user");
      let items = match m.content {
        MessageContent::Items(v) => v,
        other => panic!("expected Items branch, got: {other:?}"),
      };
      // Text-only call → exactly one Text item, NO Image entries.
      assert_eq!(items.len(), 1, "expected only text item, got: {items:?}");
      assert!(
        matches!(items[0], ContentItem::Text { .. }),
        "expected Text at [0], got: {:?}",
        items[0]
      );
      assert!(
        !items
          .iter()
          .any(|c| matches!(c, ContentItem::Image | ContentItem::ImageUrl)),
        "no Image/ImageUrl entries expected for text-only call, got: {items:?}"
      );
    }
    other => panic!("expected Message branch, got: {other:?}"),
  }
}

#[test]
fn get_message_json_text_only_with_internvl_chat_default_emits_no_audio() {
  // internvl_chat uses LIST_WITH_IMAGE_TYPE which is the formatter
  // path that appends `num_audios` ContentItem::Audio entries. With
  // the public-API defaults (num_audios=0), a text-only call must NOT
  // emit any audio entries.
  let out = get_message_json("internvl_chat", "describe this", None).unwrap();
  match out {
    FormattedMessage::Message(m) => {
      assert_eq!(m.role, "user");
      let items = match m.content {
        MessageContent::Items(v) => v,
        other => panic!("expected Items branch, got: {other:?}"),
      };
      // Text-only call → exactly one ContentText item, NO Audio entries.
      assert_eq!(items.len(), 1, "expected only text item, got: {items:?}");
      assert!(
        !items.iter().any(|c| matches!(c, ContentItem::Audio)),
        "no Audio entries expected for text-only call, got: {items:?}"
      );
    }
    other => panic!("expected Message branch, got: {other:?}"),
  }
}

#[test]
fn format_message_internal_default_keeps_one_image() {
  // FormatOpts::formatter_default() preserves the python in-class
  // kwarg defaults (num_images=1, num_audios=1) at line 207-208 so a
  // direct `MessageFormatter::format_message` call with
  // formatter_default() still emits the one-image structure. This
  // pins the per-API split: changing the public-API default to
  // num_images=0 must NOT change the formatter-internal default.
  let f = MessageFormatter::for_model("qwen2_vl").unwrap();
  let out = f
    .format_message("describe this", &FormatOpts::formatter_default())
    .unwrap();
  match out {
    FormattedMessage::Message(m) => {
      let items = match m.content {
        MessageContent::Items(v) => v,
        other => panic!("expected Items branch, got: {other:?}"),
      };
      // qwen2_vl LIST_WITH_IMAGE puts text first then images; with
      // num_images=1 we expect exactly one image entry.
      assert_eq!(items.len(), 2, "expected text + 1 image, got: {items:?}");
      assert_eq!(
        items
          .iter()
          .filter(|c| matches!(c, ContentItem::Image))
          .count(),
        1,
        "expected exactly one Image entry under formatter_default, got: {items:?}"
      );
    }
    other => panic!("expected Message branch, got: {other:?}"),
  }
}

// ──────────────────────── Allocation hardening ─────────────────

#[test]
fn get_message_json_extreme_num_images_returns_oom_error() {
  // num_images = usize::MAX / 2 → must return Err (cap-exceeded or
  // OOM, NOT a panic). An unguarded `format_list_with_image`'s
  // `Vec::with_capacity(1 + opts.num_images)` would attempt a
  // catastrophic allocation; the count is cap-checked against
  // MAX_MESSAGE_FORMAT_ITEMS first → Error::Backend.
  let err = get_message_json(
    "qwen2_vl",
    "describe",
    Some(&FormatOpts {
      num_images: usize::MAX / 2,
      ..Default::default()
    }),
  )
  .unwrap_err();
  // Must be either CapExceeded (cap-exceeded path) or OutOfMemory —
  // both are recoverable, neither panics.
  assert!(
    matches!(err, Error::CapExceeded(_) | Error::OutOfMemory),
    "expected CapExceeded/OutOfMemory, got: {err:?}"
  );
  // The cap-exceeded path is the expected one (the value is way above
  // MAX_MESSAGE_FORMAT_ITEMS = 1024); the payload context names the
  // count and cap_name names the cap for triage.
  if let Error::CapExceeded(p) = &err {
    assert_eq!(p.cap_name(), "MAX_MESSAGE_FORMAT_ITEMS");
    assert_eq!(p.context(), "num_images");
  }
}

#[test]
fn get_message_json_with_skip_image_token_does_not_allocate_for_num_images() {
  // skip_image_token=true with a pathological num_images MUST NOT
  // touch the count-scaled reserve — the gate is BEFORE the cap
  // check. qwen2_vl uses LIST_WITH_IMAGE (Vec<ContentItem>); if the
  // skip-gate ran after the reserve, `Vec::with_capacity(1 + num_images)`
  // would panic on 1_000_000 before reaching it.
  //
  // The skip-gate sets `effective_n = 0`, bypassing the cap check
  // entirely. Test asserts the call succeeds cheaply with a single text
  // item.
  let out = get_message_json(
    "qwen2_vl",
    "describe",
    Some(&FormatOpts {
      num_images: 1_000_000, // would blow up the unhardened reserve
      skip_image_token: true,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::Message(m) => {
      let items = match m.content {
        MessageContent::Items(v) => v,
        other => panic!("expected Items branch, got: {other:?}"),
      };
      // skip_image_token=true → NO image entries, just text.
      assert_eq!(items.len(), 1, "skip_image_token must suppress images");
      assert!(
        matches!(items[0], ContentItem::Text { .. }),
        "expected Text only, got: {items:?}"
      );
    }
    other => panic!("expected Message branch, got: {other:?}"),
  }
}

#[test]
fn format_with_token_skip_image_does_not_allocate_for_num_images() {
  // Same skip-gate behavior on the IMAGE_TOKEN family
  // (`<image>` * num_images + prompt) — minicpmo. An unguarded
  // `String::with_capacity(token.len() * num_images)` would blow up
  // on 1_000_000; the skip gate sets effective_n=0 first.
  let out = get_message_json(
    "minicpmo",
    "describe",
    Some(&FormatOpts {
      num_images: 1_000_000,
      num_audios: 0,
      skip_image_token: true,
      ..Default::default()
    }),
  )
  .unwrap();
  match out {
    FormattedMessage::Message(m) => match m.content {
      MessageContent::Text(s) => {
        // No `<image>` prefix because skip_image_token=true.
        assert_eq!(s, "describe");
      }
      other => panic!("expected Text branch, got: {other:?}"),
    },
    other => panic!("expected Message branch, got: {other:?}"),
  }
}

#[test]
fn format_message_extreme_num_audios_returns_cap_error() {
  // num_audios above cap → Error::Backend. Same shape as the
  // num_images case but exercises the audio gate on LIST_WITH_IMAGE_TYPE
  // (internvl_chat) which is the formatter path that allocates per
  // num_audios.
  let f = MessageFormatter::for_model("internvl_chat").unwrap();
  let err = f
    .format_message(
      "describe",
      &FormatOpts {
        num_audios: MAX_MESSAGE_FORMAT_ITEMS + 1,
        ..Default::default()
      },
    )
    .unwrap_err();
  match &err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap_name(), "MAX_MESSAGE_FORMAT_ITEMS");
      assert_eq!(p.context(), "num_audios");
    }
    _ => panic!("expected CapExceeded (cap-exceeded), got: {err:?}"),
  }
}

#[test]
fn format_message_video_count_above_cap_returns_error() {
  // opts.video.len() above MAX_MESSAGE_FORMAT_ITEMS → cap error.
  // Use the format_video_message path on qwen2_vl with a video list
  // longer than the cap.
  let videos: Vec<String> = (0..MAX_MESSAGE_FORMAT_ITEMS + 1)
    .map(|i| format!("v{i}.mp4"))
    .collect();
  let err = get_message_json(
    "qwen2_vl",
    "describe",
    Some(&FormatOpts {
      num_images: 0,
      video: videos,
      ..Default::default()
    }),
  )
  .unwrap_err();
  match &err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap_name(), "MAX_MESSAGE_FORMAT_ITEMS");
      assert_eq!(p.context(), "video.len()");
    }
    _ => panic!("expected CapExceeded (cap-exceeded), got: {err:?}"),
  }
}

// ──────────────── PROMPT_WITH_IMAGE_TOKEN / PROMPT_WITH_START_IMAGE_TOKEN ────────────────
//
// Regression coverage for role/skip-image-token gating: the `<image>`
// marker must NOT be suppressed when `role != "user"` or
// `skip_image_token=true`. Python ref
// (`mlx-vlm/mlx_vlm/prompt_utils.py:265-269`) emits the marker
// **unconditionally** for both PROMPT_WITH_IMAGE_TOKEN and
// PROMPT_WITH_START_IMAGE_TOKEN: the lambda only closes over
// `num_images` and `prompt`. These tests pin the unconditional
// behavior and assert the allocation cap is still in place.

#[test]
fn format_message_paligemma_with_skip_image_token_still_emits_image_marker() {
  // paligemma → PROMPT_WITH_IMAGE_TOKEN. Even with skip_image_token=true
  // the python lambda emits `<image>` * num_images + prompt.
  let f = MessageFormatter::for_model("paligemma").unwrap();
  let out = f
    .format_message(
      "describe",
      &FormatOpts {
        role: "user".to_string(),
        skip_image_token: true,
        num_images: 1,
        ..Default::default()
      },
    )
    .unwrap();
  match out {
    FormattedMessage::String(s) => assert_eq!(s, "<image>describe"),
    other => panic!("expected String branch, got: {other:?}"),
  }
}

#[test]
fn format_message_paligemma_assistant_role_still_emits_image_marker() {
  // paligemma → PROMPT_WITH_IMAGE_TOKEN. role="assistant" must NOT
  // suppress the marker — python lambda is unconditional.
  let f = MessageFormatter::for_model("paligemma").unwrap();
  let out = f
    .format_message(
      "describe",
      &FormatOpts {
        role: "assistant".to_string(),
        skip_image_token: false,
        num_images: 1,
        ..Default::default()
      },
    )
    .unwrap();
  match out {
    FormattedMessage::String(s) => assert_eq!(s, "<image>describe"),
    other => panic!("expected String branch, got: {other:?}"),
  }
}

#[test]
fn format_message_paligemma_extreme_num_images_returns_cap_error() {
  // paligemma → PROMPT_WITH_IMAGE_TOKEN. num_images=1_000_000 must hit
  // the SINGLE_IMAGE_ONLY_MODELS guard FIRST (paligemma is single-
  // image-only, so num_images > 1 short-circuits before the helper's
  // own cap). Either way the result is `Err(OutOfRange)` for the
  // single-image guard, or `Err(CapExceeded)` had the model not been in
  // SINGLE_IMAGE_ONLY_MODELS. We test the helper cap directly via
  // a non-single-image-only synthetic formatter below.
  let f = MessageFormatter::for_model("paligemma").unwrap();
  let err = f
    .format_message(
      "describe",
      &FormatOpts {
        num_images: 1_000_000,
        ..Default::default()
      },
    )
    .unwrap_err();
  // paligemma is in SINGLE_IMAGE_ONLY_MODELS, so the
  // num_images > 1 guard fires first as OutOfRange.
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange (single-image guard), got: {err:?}"
  );

  // Now exercise the helper's own cap by synthesizing a formatter
  // whose model_name is NOT in SINGLE_IMAGE_ONLY_MODELS so the cap
  // check (1024 < 1_000_000) is reached.
  let synth = MessageFormatter {
    model_name: "synthetic_pwit".to_string(),
    format_type: MessageFormat::PromptWithImageToken,
  };
  let err = synth
    .format_message(
      "describe",
      &FormatOpts {
        num_images: 1_000_000,
        ..Default::default()
      },
    )
    .unwrap_err();
  match &err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap_name(), "MAX_MESSAGE_FORMAT_ITEMS");
      assert_eq!(p.context(), "num_images");
    }
    _ => panic!("expected CapExceeded on PromptWithImageToken cap, got: {err:?}"),
  }
}

#[test]
fn format_message_paligemma_num_images_overflow_returns_error() {
  // PROMPT_WITH_IMAGE_TOKEN: num_images = usize::MAX / 2 must NOT
  // panic — either the cap fires (Backend) or, in a hypothetical
  // future where the cap is raised past usize::MAX / 2, the
  // checked_mul(7).checked_add(prompt.len()) guard fires
  // (Backend overflow). With the current cap (1024) this fires as
  // Backend cap-exceeded. Test via a synthetic non-single-image-only
  // formatter (paligemma's single-image guard would short-circuit).
  let synth = MessageFormatter {
    model_name: "synthetic_pwit".to_string(),
    format_type: MessageFormat::PromptWithImageToken,
  };
  let err = synth
    .format_message(
      "describe",
      &FormatOpts {
        num_images: usize::MAX / 2,
        ..Default::default()
      },
    )
    .unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_) | Error::OutOfMemory),
    "expected CapExceeded or OutOfMemory (no panic), got: {err:?}"
  );
}

// PROMPT_WITH_START_IMAGE_TOKEN — no model in MODEL_CONFIG currently
// maps to this format, so the tests synthesize a MessageFormatter
// directly via the public struct fields. The helper is still kept
// faithful for completeness with the python ref.

#[test]
fn format_message_prompt_with_start_image_token_with_skip_still_emits_marker() {
  // PROMPT_WITH_START_IMAGE_TOKEN. skip_image_token=true must NOT
  // suppress the suffix — python ref `prompt_utils.py:268-269` is
  // unconditional.
  let f = MessageFormatter {
    model_name: "synthetic_pwsit".to_string(),
    format_type: MessageFormat::PromptWithStartImageToken,
  };
  let out = f
    .format_message(
      "describe",
      &FormatOpts {
        role: "user".to_string(),
        skip_image_token: true,
        num_images: 1,
        ..Default::default()
      },
    )
    .unwrap();
  match out {
    FormattedMessage::String(s) => assert_eq!(s, "describe<start_of_image>"),
    other => panic!("expected String branch, got: {other:?}"),
  }
}

#[test]
fn format_message_prompt_with_start_image_token_assistant_role_still_emits_marker() {
  // PROMPT_WITH_START_IMAGE_TOKEN. role="assistant" must NOT suppress
  // the suffix.
  let f = MessageFormatter {
    model_name: "synthetic_pwsit".to_string(),
    format_type: MessageFormat::PromptWithStartImageToken,
  };
  let out = f
    .format_message(
      "describe",
      &FormatOpts {
        role: "assistant".to_string(),
        skip_image_token: false,
        num_images: 1,
        ..Default::default()
      },
    )
    .unwrap();
  match out {
    FormattedMessage::String(s) => assert_eq!(s, "describe<start_of_image>"),
    other => panic!("expected String branch, got: {other:?}"),
  }
}

#[test]
fn format_message_prompt_with_start_image_token_extreme_num_images_returns_cap_error() {
  // PROMPT_WITH_START_IMAGE_TOKEN. num_images=1_000_000 → cap error
  // (Backend). Synthetic model is NOT in SINGLE_IMAGE_ONLY_MODELS, so
  // the helper's own cap check fires.
  let f = MessageFormatter {
    model_name: "synthetic_pwsit".to_string(),
    format_type: MessageFormat::PromptWithStartImageToken,
  };
  let err = f
    .format_message(
      "describe",
      &FormatOpts {
        num_images: 1_000_000,
        ..Default::default()
      },
    )
    .unwrap_err();
  match &err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap_name(), "MAX_MESSAGE_FORMAT_ITEMS");
      assert_eq!(p.context(), "num_images");
    }
    _ => panic!("expected CapExceeded (cap-exceeded), got: {err:?}"),
  }
}

#[test]
fn format_message_prompt_with_start_image_token_num_images_overflow_returns_error() {
  // PROMPT_WITH_START_IMAGE_TOKEN. num_images = usize::MAX / 2 must
  // NOT panic — cap (Backend) or checked_mul overflow guard
  // (Backend / OutOfMemory) catches it.
  let f = MessageFormatter {
    model_name: "synthetic_pwsit".to_string(),
    format_type: MessageFormat::PromptWithStartImageToken,
  };
  let err = f
    .format_message(
      "describe",
      &FormatOpts {
        num_images: usize::MAX / 2,
        ..Default::default()
      },
    )
    .unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_) | Error::OutOfMemory),
    "expected CapExceeded or OutOfMemory (no panic), got: {err:?}"
  );
}

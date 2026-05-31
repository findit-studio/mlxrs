//! Unit tests for [`ChunkedKvCache`] internals that the public-API
//! integration suite (`tests/lm_cache_chunked.rs`) cannot reach: the
//! `checked_*` overflow/underflow closures (a faithful `set_state` derives
//! `offset` from `keys.shape[2]` and `set_meta_state` parses a `usize`, so a
//! `usize::MAX`-class hostile value can only be injected via an in-module
//! struct literal), the private [`ChunkedKvCache::set_seq`] write-bounds
//! helper, and the empty/`None` match arms of `state`/`materialize`/`copy`.
//!
//! Oracle discipline: every retained-token assertion feeds DISTINCT marker
//! K/V streams (keys 10/20/30…, values 100/200/300…) and checks the exact
//! resulting buffer element-by-element against a hand-traced
//! `cache.py`-faithful expectation — never against the function under test.
//! Validation branches are matched by typed `Error` variant + payload.
use super::*;

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// so each row's marker value reads straight out of `to_vec`. `S ==
/// vals.len()`.
fn kv(vals: &[f32]) -> Array {
  Array::from_slice::<f32>(vals, &(1usize, 1, vals.len(), 1)).unwrap()
}

/// Row-major host read of a (possibly strided) 4-D KV array — route every
/// returned slice through `contiguous` first (a `seq_slice` view may be
/// strided), mirroring the sibling `batch_rotating` tests.
fn rows(a: &Array) -> Vec<f32> {
  ops::shape::contiguous(a, false)
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
}

// ── maybe_trim_front: the `keys is None` no-op arm (line 120) ────────────

/// `maybe_trim_front` with a real `chunk_size` but an EMPTY cache (keys ==
/// None) takes the `(_, _) => return Ok(())` arm (line 120) — a no-op that
/// leaves `offset`/`start_position` untouched. Distinct from the
/// `chunk_size == None` short-circuit (line 115), which the integration
/// suite covers.
#[test]
fn maybe_trim_front_empty_cache_is_noop() {
  let mut c = ChunkedKvCache::new(Some(4));
  assert!(c.is_empty());
  c.maybe_trim_front().unwrap();
  assert!(c.is_empty(), "no-op must not populate the buffer");
  assert_eq!(c.offset(), 0);
  assert_eq!(c.meta_state(), vec!["4", "0"], "start_position untouched");
}

// ── maybe_trim_front: the `start_position + added` overflow (142-147) ────

/// A hostile restored `start_position` near `usize::MAX` makes the
/// `start_position += buf_len - chunk_size` bump overflow; the `checked_add`
/// closure (lines 141-150) surfaces it as a recoverable `ArithmeticOverflow`
/// carrying both operands, with NO partial mutation (keys/values/offset
/// untouched). Built via a struct literal because a faithful restore can
/// never inject `start_position == usize::MAX`.
#[test]
fn maybe_trim_front_start_position_overflow_is_rejected() {
  // buf_len 8 >= chunk_size 4 -> added = 4; usize::MAX + 4 overflows.
  let buf = kv(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]); // [1,1,8,1]
  let vbuf = kv(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]);
  let mut c = ChunkedKvCache {
    keys: Some(buf.try_clone().unwrap()),
    values: Some(vbuf.try_clone().unwrap()),
    offset: 8,
    chunk_size: Some(4),
    start_position: usize::MAX,
  };
  let err = c.maybe_trim_front().unwrap_err();
  match err {
    Error::ArithmeticOverflow(p) => {
      assert!(
        p.context().contains("maybe_trim_front") && p.context().contains("start_position"),
        "context must name the maybe_trim_front start_position add, got: {}",
        p.context()
      );
      assert!(
        p.operands().iter().any(|(n, v)| *n == "added" && *v == 4),
        "operands must carry added=4, got: {:?}",
        p.operands()
      );
      assert!(
        p.operands()
          .iter()
          .any(|(n, v)| *n == "start_position" && *v == usize::MAX as u64),
        "operands must carry start_position=usize::MAX, got: {:?}",
        p.operands()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  // No partial mutation: start_position is still MAX, buffer length intact.
  assert_eq!(
    c.meta_state(),
    vec!["4".to_string(), usize::MAX.to_string()],
    "start_position must be unchanged on the Err path"
  );
  let st = c.state().unwrap();
  assert_eq!(st[0].shape(), vec![1, 1, 8, 1], "keys buffer untrimmed");
}

// ── set_seq: the `start + S` overflow context arms (208-219) ─────────────

/// `ChunkedKvCache::set_seq` with `a == usize::MAX` overflows `a + s`; the
/// `checked_add` closure picks the per-target context string ("keys" /
/// "values") and returns `ArithmeticOverflow` carrying `start`/`S` — never a
/// panic. The private helper is reachable only from an in-module test.
#[test]
fn set_seq_write_start_plus_s_overflow_keys_and_values() {
  let buf = kv(&[10.0]); // [1,1,1,1]
  let new = kv(&[20.0]);
  for (name, want_ctx) in [
    ("keys", "keys write start"),
    ("values", "values write start"),
  ] {
    let err = ChunkedKvCache::set_seq(name, &buf, usize::MAX, 1, &new).unwrap_err();
    match err {
      Error::ArithmeticOverflow(p) => {
        assert!(
          p.context().contains(want_ctx),
          "{name}: context must be the per-target write-start arm, got: {}",
          p.context()
        );
        assert!(
          p.operands()
            .iter()
            .any(|(n, v)| *n == "start" && *v == usize::MAX as u64),
          "{name}: operands must carry start=usize::MAX, got: {:?}",
          p.operands()
        );
        assert!(
          p.operands().iter().any(|(n, v)| *n == "S" && *v == 1),
          "{name}: operands must carry S=1, got: {:?}",
          p.operands()
        );
      }
      other => panic!("{name}: expected ArithmeticOverflow, got {other:?}"),
    }
  }
}

// ── set_seq: the `end > l` OutOfRange arms incl. the generic `_` (220-230) ─

/// `set_seq` with a write window `[a, end)` extending past the buffer length
/// `l` returns `OutOfRange` (no silent truncation). All three context arms
/// are exercised: the two named targets (221-223) and the generic `_ =>`
/// fallback (line 224) via a non-"keys"/"values" name.
#[test]
fn set_seq_window_past_buffer_is_out_of_range() {
  let buf = kv(&[10.0]); // length 1 on the seq axis
  let new = kv(&[20.0, 21.0, 22.0, 23.0, 24.0]); // S = 5 -> end 5 > l 1
  for name in ["keys", "values", "other"] {
    let err = ChunkedKvCache::set_seq(name, &buf, 0, 5, &new).unwrap_err();
    match err {
      Error::OutOfRange(p) => {
        assert!(
          p.context().contains("write window end"),
          "{name}: context must name the out-of-bounds write window, got: {}",
          p.context()
        );
        // The generic fallback arm omits the per-target prefix; the named
        // arms include it. Both name the window-end violation above.
        if name == "keys" || name == "values" {
          assert!(
            p.context().contains(name),
            "{name}: named arm must include the target buffer name, got: {}",
            p.context()
          );
        }
      }
      other => panic!("{name}: expected OutOfRange, got {other:?}"),
    }
  }
}

/// `set_seq` SUCCESS: a fully in-bounds partial-window splice overwrites
/// exactly `[a, a+s)` and keeps the surrounding rows — closed-form oracle on
/// distinct markers (buffer rows 10,11,12,13; write 99 at index 1).
#[test]
fn set_seq_partial_window_splices_in_place() {
  let buf = kv(&[10.0, 11.0, 12.0, 13.0]);
  let new = kv(&[99.0]);
  let spliced = ChunkedKvCache::set_seq("keys", &buf, 1, 1, &new).unwrap();
  assert_eq!(spliced.shape(), vec![1, 1, 4, 1], "buffer length preserved");
  assert_eq!(
    rows(&spliced),
    vec![10.0, 99.0, 12.0, 13.0],
    "only row 1 overwritten; rows 0/2/3 retained"
  );
}

// ── update: prev = offset - start_position underflow (310-319) ───────────

/// A hostile restored `start_position > offset` makes `prev = offset -
/// start_position` underflow; the `checked_sub` closure (lines 310-319)
/// surfaces a recoverable `ArithmeticOverflow` with NO mutation. Injected
/// via struct literal (a faithful trace keeps `start_position <= offset`).
#[test]
fn update_prev_underflow_when_start_exceeds_offset() {
  let mut c = ChunkedKvCache {
    keys: None,
    values: None,
    offset: 3,
    chunk_size: Some(4),
    start_position: 5, // > offset -> prev underflows
  };
  let t = kv(&[10.0]);
  let err = c.update(&t, &t).unwrap_err();
  match err {
    Error::ArithmeticOverflow(p) => {
      assert!(
        p.context().contains("offset - start_position"),
        "context must name the prev underflow, got: {}",
        p.context()
      );
      assert!(
        p.operands().iter().any(|(n, v)| *n == "offset" && *v == 3)
          && p
            .operands()
            .iter()
            .any(|(n, v)| *n == "start_position" && *v == 5),
        "operands must carry offset=3, start_position=5, got: {:?}",
        p.operands()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  assert_eq!(c.offset(), 3, "offset unchanged on the Err path");
  assert!(c.is_empty(), "buffer unchanged (still None)");
}

// ── update: prev + S overflow (326-332) ──────────────────────────────────

/// `offset == usize::MAX`, `start_position == 0` -> `prev == usize::MAX`, so
/// `prev + S` overflows (lines 326-332). Recoverable `ArithmeticOverflow`,
/// no mutation.
#[test]
fn update_prev_plus_s_overflow_is_rejected() {
  let mut c = ChunkedKvCache {
    keys: None,
    values: None,
    offset: usize::MAX,
    chunk_size: None,
    start_position: 0,
  };
  let t = kv(&[10.0]);
  let err = c.update(&t, &t).unwrap_err();
  match err {
    Error::ArithmeticOverflow(p) => {
      assert!(
        p.context().contains("prev + S"),
        "context must name prev + S, got: {}",
        p.context()
      );
      assert!(
        p.operands()
          .iter()
          .any(|(n, v)| *n == "prev" && *v == usize::MAX as u64),
        "operands must carry prev=usize::MAX, got: {:?}",
        p.operands()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  assert_eq!(c.offset(), usize::MAX, "offset unchanged on the Err path");
}

// ── update: offset + S overflow (428-434) ────────────────────────────────

/// `offset == start_position == usize::MAX` -> `prev == 0` (sub ok), `prev +
/// S == 1` (add ok), the empty-cache realloc builds the 256-row zero block
/// (allocatable), and only then does `offset + S == usize::MAX + 1` overflow
/// (lines 428-434). Recoverable `ArithmeticOverflow`, no mutation. This
/// isolates the `offset + S` closure downstream of the realloc.
#[test]
fn update_offset_plus_s_overflow_after_realloc() {
  let mut c = ChunkedKvCache {
    keys: None,
    values: None,
    offset: usize::MAX,
    chunk_size: None,
    start_position: usize::MAX, // prev = MAX - MAX = 0
  };
  let t = kv(&[10.0]); // [1,1,1,1] -> realloc zeros [1,1,256,1] is fine
  let err = c.update(&t, &t).unwrap_err();
  match err {
    Error::ArithmeticOverflow(p) => {
      assert!(
        p.context().contains("offset + S"),
        "context must name offset + S, got: {}",
        p.context()
      );
      assert!(
        p.operands()
          .iter()
          .any(|(n, v)| *n == "offset" && *v == usize::MAX as u64),
        "operands must carry offset=usize::MAX, got: {:?}",
        p.operands()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  assert_eq!(c.offset(), usize::MAX, "offset unchanged on the Err path");
  assert!(c.is_empty(), "buffer not committed on the Err path");
}

// ── update: realloc with prev % step == 0 onto an EXISTING buffer (395) ──

/// The `prev % step != 0` false branch's `else (None, None)` arm (line 395):
/// a realloc whose write cursor `prev` is an exact multiple of `step`
/// concatenates the fresh zero block onto the EXISTING buffer directly (no
/// partial-tail drop). Reached by an S==256 prefill (buffer grows to exactly
/// 256, `offset == 256`) followed by an S==1 update (`prev == 256`, `256 %
/// 256 == 0`, and `prev + 1 == 257 > 256` forces the realloc). Distinct K/V
/// streams; closed-form oracle on the retained rows.
#[test]
fn update_realloc_prev_multiple_of_step_keeps_existing_buffer() {
  let mut c = ChunkedKvCache::new(None);
  // Prefill S=256: n_steps = (256 + 256 - 1) // 256 = 1, total = 256; the
  // empty branch allocates [1,1,256,1] and splices [0,256) -> buffer len
  // 256, offset 256. Distinct K (1000+i) / V (2000+i) markers per row.
  let kpre: Vec<f32> = (0..256).map(|i| 1000.0 + i as f32).collect();
  let vpre: Vec<f32> = (0..256).map(|i| 2000.0 + i as f32).collect();
  let (_pk, _pv) = c.update(&kv(&kpre), &kv(&vpre)).unwrap();
  assert_eq!(c.offset(), 256);

  // S=1 update: prev = 256 - 0 = 256; prev % 256 == 0 -> the (None, None)
  // else arm keeps `pk`/`pv` whole and concatenates the new 256-row zero
  // block (buffer -> 512). offset += 1 -> 257; end = 257; the splice writes
  // the new row at [256, 257). Return keys[..., :257, :].
  let (rk, rv) = c.update(&kv(&[314.0]), &kv(&[628.0])).unwrap();
  assert_eq!(c.offset(), 257);
  assert_eq!(
    rk.shape(),
    vec![1, 1, 257, 1],
    "logical length 257 returned"
  );
  let rk_rows = rows(&rk);
  let rv_rows = rows(&rv);
  assert_eq!(rk_rows.len(), 257);
  // The 256 prefill rows are retained verbatim (the existing buffer was NOT
  // partial-tail-dropped), and the new row is appended at index 256.
  assert_eq!(
    &rk_rows[..256],
    kpre.as_slice(),
    "prefill keys retained whole by the prev%step==0 realloc arm"
  );
  assert_eq!(rk_rows[256], 314.0, "new key row appended at index 256");
  assert_eq!(
    &rv_rows[..256],
    vpre.as_slice(),
    "prefill values retained whole (own stream)"
  );
  assert_eq!(rv_rows[256], 628.0, "new value row appended at index 256");
}

// ── state: the empty `_ => Ok(Vec::new())` arm (line 508) ────────────────

/// `state()` on a fresh (empty) cache returns `[]` via the `_ =>
/// Ok(Vec::new())` arm (line 508) — never a panic on a `None` buffer.
#[test]
fn state_empty_cache_is_empty_vec() {
  let c = ChunkedKvCache::new(Some(4));
  assert!(c.state().unwrap().is_empty());
  // `None` chunk_size empty cache too (still the same arm).
  let c2 = ChunkedKvCache::new(None);
  assert!(c2.state().unwrap().is_empty());
}

// ── materialize: the Some(keys)/Some(values) eval arms + empty no-op ─────

/// `materialize()` force-evals the stored `keys` (523-525) and `values`
/// (526-528) buffers in place; the observable state is unchanged afterward.
/// An empty cache hits both `None` branches (a pure no-op).
#[test]
fn materialize_evals_buffers_and_empty_is_noop() {
  let mut c = ChunkedKvCache::new(Some(4));
  // Distinct K/V so the post-materialize readback is load-bearing.
  let (_k, _v) = c.update(&kv(&[10.0, 20.0]), &kv(&[100.0, 200.0])).unwrap();
  c.materialize().unwrap();
  // Pure memory barrier: offset and the logical buffer contents are intact.
  assert_eq!(c.offset(), 2);
  let st = c.state().unwrap();
  assert_eq!(
    rows(&st[0]),
    vec![10.0, 20.0],
    "keys unchanged by materialize"
  );
  assert_eq!(
    rows(&st[1]),
    vec![100.0, 200.0],
    "values unchanged by materialize (own stream)"
  );

  // Empty cache: keys/values are None -> both `if let Some` guards are
  // false; materialize is a no-op.
  let mut empty = ChunkedKvCache::new(None);
  empty.materialize().unwrap();
  assert!(empty.is_empty());
}

// ── set_meta_state: the chunk_size parse-error closure (615-621) ─────────

/// A non-numeric, non-"None" `chunk_size` token makes the `parse::<usize>()`
/// fail; the closure (lines 615-621) wraps it as a recoverable
/// `Error::Parse` naming `chunk_size`, leaving the cache unmutated (the
/// parse runs before any field assignment).
#[test]
fn set_meta_state_chunk_size_parse_error_leaves_cache_unmutated() {
  let mut c = ChunkedKvCache::new(Some(7));
  let err = c
    .set_meta_state(&["not_a_number".to_string(), "0".to_string()])
    .unwrap_err();
  match err {
    Error::Parse(p) => {
      assert!(
        p.context().contains("chunk_size"),
        "context must name chunk_size, got: {}",
        p.context()
      );
      assert_eq!(p.input_kind(), "usize");
    }
    other => panic!("expected Parse, got {other:?}"),
  }
  // Unmutated: chunk_size still 7, start_position still 0.
  assert_eq!(c.meta_state(), vec!["7", "0"]);
}

// ── trim: the offset - start_position underflow closure (656-665) ────────

/// A hostile restored `start_position > offset` makes `trim`'s `offset -
/// start_position` span underflow; the `checked_sub` closure (lines 656-665)
/// surfaces a recoverable `ArithmeticOverflow` with NO mutation. Struct
/// literal injection (a faithful trace keeps `start_position <= offset`).
#[test]
fn trim_start_exceeds_offset_underflow_is_rejected() {
  let mut c = ChunkedKvCache {
    keys: None,
    values: None,
    offset: 2,
    chunk_size: Some(4),
    start_position: 5, // > offset
  };
  let err = c.trim(1).unwrap_err();
  match err {
    Error::ArithmeticOverflow(p) => {
      assert!(
        p.context().contains("trim") && p.context().contains("offset - start_position"),
        "context must name the trim span underflow, got: {}",
        p.context()
      );
      assert!(
        p.operands().iter().any(|(n, v)| *n == "offset" && *v == 2)
          && p
            .operands()
            .iter()
            .any(|(n, v)| *n == "start_position" && *v == 5),
        "operands must carry offset=2, start_position=5, got: {:?}",
        p.operands()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  assert_eq!(c.offset(), 2, "offset unchanged on the Err path");
}

// ── copy: the Some(keys)/Some(values) clone arms (737-744) + None arms ────

/// `copy()` of a POPULATED cache exercises the `Some(a) => Some(a.try_clone)`
/// arms for both `keys` (line 738) and `values` (line 742); the copy is an
/// independent, equal cache (distinct K/V streams read back element-by-
/// element). A copy of an EMPTY cache hits the `None` arms (lines 739/743).
#[test]
fn copy_clones_both_buffers_and_empty_takes_none_arms() {
  let mut c = ChunkedKvCache::new(Some(4));
  // Distinct K/V markers so the copied buffers are load-bearing.
  c.update(&kv(&[10.0, 20.0, 30.0]), &kv(&[100.0, 200.0, 300.0]))
    .unwrap();
  let cp = c.copy().unwrap();
  assert_eq!(cp.offset(), 3, "copied scalar offset matches");
  assert_eq!(cp.reference_class_name(), "ChunkedKVCache");
  let st = cp.state().unwrap();
  assert_eq!(st.len(), 2);
  assert_eq!(
    rows(&st[0]),
    vec![10.0, 20.0, 30.0],
    "copied keys are an exact independent duplicate"
  );
  assert_eq!(
    rows(&st[1]),
    vec![100.0, 200.0, 300.0],
    "copied values track their own stream"
  );
  // Independence: advancing the ORIGINAL must not perturb the copy
  // (try_clone shares refcounts, but the cache only ever reassigns its
  // arrays — copy and original evolve independently).
  c.update(&kv(&[40.0]), &kv(&[400.0])).unwrap();
  assert_eq!(c.offset(), 4, "original advanced");
  assert_eq!(cp.offset(), 3, "copy untouched by the original's update");

  // Empty copy: keys/values None -> the `None` clone arms.
  let empty = ChunkedKvCache::new(None);
  let ecp = empty.copy().unwrap();
  assert!(ecp.is_empty());
  assert_eq!(ecp.offset(), 0);
  assert!(ecp.state().unwrap().is_empty());
}

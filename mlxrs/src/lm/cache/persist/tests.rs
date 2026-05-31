//! Unit tests for the prompt-cache persistence layer's PRIVATE helpers
//! (`dense_len` / `unflatten_arrays` / `unflatten_side`) and the
//! file-level typed-`Err` gates, focused on what the integration suites
//! (`tests/lm_cache_persist.rs`, `tests/lm_cache_prompt.rs`) cannot reach
//! or only assert with a bare `is_err()`:
//!
//!  * `dense_len` directly: dense-OK, the three `what` context strings,
//!    the `usize::MAX` overflow arm (`ArithmeticOverflow`), and the
//!    non-dense arm (`LengthMismatch`) — none reachable from outside the
//!    module;
//!  * `unflatten_arrays` / `unflatten_side` directly: sub-index ordering,
//!    empty input, swift-parity skip of non-`i.j` / non-numeric keys, the
//!    scalar-`"0.{i}"` empty-vs-truthy meta forms, list-wins-over-scalar
//!    collision, dotted `"1.key"` remainder, unknown-tag skip;
//!  * the file-level gates with their EXACT typed payloads: the
//!    non-regular-file (`FileIo`/`FileOp::Open`) and missing-file gates,
//!    the size cap (`CapExceeded` + `MAX_PROMPT_CACHE_BYTES` closed-form),
//!    the 4-D rank gate (`LayerKeyed(RankMismatch)`), and the no-meta
//!    emptiness gate (`LayerKeyed(InvariantViolation)`);
//!  * the empty-cache (`&[]`) save→load round-trip (0 caches, 0 meta) and
//!    the closed-form scalar-`"0.{i}"=""` side-table emission for a
//!    no-meta cache.
//!
//! Oracles are ROUND-TRIP (save→load, recovered == original) or CLOSED-
//! FORM (the on-disk key layout / `dense_len` arithmetic computed from the
//! format spec, never by calling the writer) or EXACT typed-error-variant
//! matching. Truncated/garbage *payloads* go through mlx-c's safetensors
//! parser, whose error variant is not part of persist.rs's contract, so
//! those are asserted `is_err()` (no panic) without pinning the variant.

use super::*;
use crate::{
  array::Array,
  error::{Error, FileOp},
  lm::cache::{ArraysCache, RotatingKvCache, StandardKvCache},
};

// ── fixtures (mirror tests/lm_cache_persist.rs idioms) ──

/// Unique temp path per test name, PID-scoped so parallel test bins do
/// not collide. Mirrors `tests/lm_cache_persist.rs::temp_path`.
fn temp_path(name: &str) -> std::path::PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!(
    "mlxrs_persist_inline_{}_{}",
    std::process::id(),
    name
  ));
  p
}

/// A `[1, 1, S, 1]` 4-D KV tensor whose sequence values are the given ids
/// — the canonical KV fixture shape (`KV_NDIM == 4`). Identical to the
/// ramp used across the cache module's tests.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

// ─────────────────────── dense_len (private) ───────────────────────────

#[test]
fn dense_len_dense_ok_returns_max_plus_one() {
  // A faithful flattened list is dense: indices 0..len, so max_idx+1 ==
  // present. CLOSED-FORM: for indices {0,1,2} -> present=3, max=2 -> 3.
  assert_eq!(dense_len(2, 3, "array sub").unwrap(), 3);
  assert_eq!(dense_len(0, 1, "meta_state").unwrap(), 1);
  assert_eq!(dense_len(4, 5, "class").unwrap(), 5);
}

#[test]
fn dense_len_non_dense_is_length_mismatch_with_per_what_context() {
  // A gap (present < max+1) is corrupt/adversarial -> LengthMismatch with
  // expected=present, actual=max+1, and the `what`-specific static
  // context. Pin all three call-site contexts.
  for (what, ctx) in [
    (
      "array sub",
      "prompt cache: non-dense array sub indices (corrupt or incompatible file)",
    ),
    (
      "meta_state",
      "prompt cache: non-dense meta_state indices (corrupt or incompatible file)",
    ),
    (
      "class",
      "prompt cache: non-dense class indices (corrupt or incompatible file)",
    ),
  ] {
    // indices {0,2}: present=2, max=2 -> n=3 != 2.
    match dense_len(2, 2, what) {
      Err(Error::LengthMismatch(p)) => {
        assert_eq!(p.context(), ctx, "context for what={what:?}");
        assert_eq!(p.expected(), 2, "expected == present (distinct keys)");
        assert_eq!(p.actual(), 3, "actual == max_idx + 1");
      }
      other => panic!("non-dense ({what}) must be LengthMismatch, got {other:?}"),
    }
  }
}

#[test]
fn dense_len_unknown_what_falls_back_to_generic_context() {
  // The `_ =>` arms of both match blocks: an out-of-vocabulary `what`
  // yields the generic (non-`what`-tagged) context, not a panic.
  match dense_len(2, 2, "bogus") {
    Err(Error::LengthMismatch(p)) => {
      assert_eq!(
        p.context(),
        "prompt cache: non-dense indices (corrupt or incompatible file)"
      );
    }
    other => panic!("expected generic-context LengthMismatch, got {other:?}"),
  }
}

#[test]
fn dense_len_overflow_is_arithmetic_overflow_with_operand() {
  // max_idx == usize::MAX makes `max_idx + 1` overflow -> the
  // ArithmeticOverflow arm, carrying the offending `max_idx` operand and
  // the `what`-specific context. This arm is UNREACHABLE from a file-level
  // test (no usize::MAX key fits the size cap); only a direct unit hits it.
  match dense_len(usize::MAX, 0, "array sub") {
    Err(Error::ArithmeticOverflow(p)) => {
      assert_eq!(p.context(), "prompt cache: array sub index overflows usize");
      assert_eq!(p.op_type(), "usize");
      assert!(
        p.operands()
          .iter()
          .any(|(n, v)| *n == "max_idx" && *v == usize::MAX as u64),
        "operands must carry max_idx=usize::MAX, got {:?}",
        p.operands()
      );
    }
    other => panic!("overflow must be ArithmeticOverflow, got {other:?}"),
  }
  // The meta_state / class contexts on the same overflow arm.
  match dense_len(usize::MAX, 0, "meta_state") {
    Err(Error::ArithmeticOverflow(p)) => {
      assert_eq!(
        p.context(),
        "prompt cache: meta_state index overflows usize"
      )
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  match dense_len(usize::MAX, 0, "class") {
    Err(Error::ArithmeticOverflow(p)) => {
      assert_eq!(p.context(), "prompt cache: class index overflows usize")
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

// ─────────────────────── unflatten_arrays (private) ────────────────────

#[test]
fn unflatten_arrays_empty_input_is_empty_map() {
  let out = unflatten_arrays(HashMap::new()).unwrap();
  assert!(out.is_empty(), "no array keys -> no cache groups");
}

#[test]
fn unflatten_arrays_orders_by_sub_index_not_insertion_order() {
  // mlx-c map iteration order is unspecified; the parser must order the
  // per-cache arrays by parsed sub-index `j`. Insert j=1 before j=0.
  let mut flat: HashMap<String, Array> = HashMap::new();
  flat.insert("0.1".to_string(), kv(&[40.0]));
  flat.insert("0.0".to_string(), kv(&[10.0]));
  let mut out = unflatten_arrays(flat).unwrap();
  let mut v = out.remove(&0).expect("cache 0 present");
  assert_eq!(v.len(), 2);
  // Ordered by j: slot 0 is the [10] tensor, slot 1 the [40] tensor.
  assert_eq!(v[0].to_vec::<f32>().unwrap(), vec![10.0]);
  assert_eq!(v[1].to_vec::<f32>().unwrap(), vec![40.0]);
}

#[test]
fn unflatten_arrays_skips_non_ij_and_non_numeric_keys() {
  // swift parity: a key with no `.` (`"5"`), or a non-base-10 i/j
  // (`"x.0"`, `"0.y"`), is silently ignored — NOT an error.
  let mut flat: HashMap<String, Array> = HashMap::new();
  flat.insert("0.0".to_string(), kv(&[1.0]));
  flat.insert("5".to_string(), kv(&[2.0])); // no dot
  flat.insert("x.0".to_string(), kv(&[3.0])); // non-numeric i
  flat.insert("0.y".to_string(), kv(&[4.0])); // non-numeric j
  let out = unflatten_arrays(flat).unwrap();
  assert_eq!(out.len(), 1, "only the valid `0.0` key forms a group");
  assert_eq!(out[&0].len(), 1);
}

#[test]
fn unflatten_arrays_non_dense_sub_indices_is_err() {
  // A per-cache gap ({0,2}, no 1) flows through dense_len -> LengthMismatch
  // (this is the direct-unit twin of the file-level
  // `non_dense_array_sub_indices_is_err` integration test, exercised on
  // the helper boundary itself).
  let mut flat: HashMap<String, Array> = HashMap::new();
  flat.insert("0.0".to_string(), kv(&[1.0]));
  flat.insert("0.2".to_string(), kv(&[2.0]));
  match unflatten_arrays(flat) {
    Err(Error::LengthMismatch(p)) => {
      assert_eq!(
        p.context(),
        "prompt cache: non-dense array sub indices (corrupt or incompatible file)"
      );
      assert_eq!(p.expected(), 2);
      assert_eq!(p.actual(), 3);
    }
    other => panic!("non-dense sub-index must be LengthMismatch, got {other:?}"),
  }
}

// ─────────────────────── unflatten_side (private) ──────────────────────

#[test]
fn unflatten_side_scalar_empty_meta_is_empty_list() {
  // `"0.{i}"=""` is mlx-lm's tree_flatten of the scalar `_BaseCache
  // .meta_state` (the no-meta form) -> cache_info[i] == [] (empty).
  let mut side: HashMap<String, String> = HashMap::new();
  side.insert("0.0".to_string(), String::new());
  side.insert("2.0".to_string(), "KVCache".to_string());
  let (info, user, classes) = unflatten_side(side).unwrap();
  assert_eq!(info.get(&0).map(Vec::as_slice), Some(&[][..]));
  assert!(user.is_empty());
  assert_eq!(classes, vec!["KVCache".to_string()]);
}

#[test]
fn unflatten_side_truthy_scalar_meta_is_preserved_as_one_element() {
  // A *truthy* scalar `"0.{i}"="garbage"` is NOT silently dropped: it is
  // preserved as a 1-element list `["garbage"]` so the per-kind emptiness
  // gate in load_prompt_cache can reject it (mlx-lm's setter `raise`s on a
  // truthy value, cache.py:142-145).
  let mut side: HashMap<String, String> = HashMap::new();
  side.insert("0.0".to_string(), "garbage".to_string());
  side.insert("2.0".to_string(), "KVCache".to_string());
  let (info, _user, _classes) = unflatten_side(side).unwrap();
  assert_eq!(
    info.get(&0).map(Vec::as_slice),
    Some(&["garbage".to_string()][..]),
    "a truthy scalar meta_state survives as a 1-element list"
  );
}

#[test]
fn unflatten_side_list_meta_and_dotted_user_key_and_dense_classes() {
  // Cover the three tags together:
  //   "0.0.{j}"  -> a LIST meta_state (RotatingKVCache's 4-tuple shape),
  //   "1.{key}"  -> user metadata, key is the verbatim dotted remainder,
  //   "2.{i}"    -> a DENSE class list.
  let mut side: HashMap<String, String> = HashMap::new();
  side.insert("0.0.0".to_string(), "4".to_string());
  side.insert("0.0.1".to_string(), "8".to_string());
  side.insert("0.0.2".to_string(), "2".to_string());
  side.insert("0.0.3".to_string(), "2".to_string());
  side.insert("1.a.b.c".to_string(), "nested".to_string());
  side.insert("2.0".to_string(), "RotatingKVCache".to_string());
  side.insert("2.1".to_string(), "KVCache".to_string());
  let (info, user, classes) = unflatten_side(side).unwrap();
  assert_eq!(
    info.get(&0).map(Vec::as_slice),
    Some(
      &[
        "4".to_string(),
        "8".to_string(),
        "2".to_string(),
        "2".to_string()
      ][..]
    ),
    "the 4-element list meta_state reconstructs in sub-index order"
  );
  assert!(
    !info.contains_key(&1),
    "cache 1 has no meta keys -> absent (sparse)"
  );
  assert_eq!(
    user.get("a.b.c").map(String::as_str),
    Some("nested"),
    "dotted user-metadata key survives as the verbatim remainder"
  );
  assert_eq!(
    classes,
    vec!["RotatingKVCache".to_string(), "KVCache".to_string()]
  );
}

#[test]
fn unflatten_side_list_meta_wins_over_scalar_for_same_index() {
  // A (corrupt) file carrying BOTH "0.{i}" and "0.{i}.{j}" for the same i:
  // the list form (inserted first, non-empty) wins; the `or_insert_with`
  // for the scalar does not clobber it.
  let mut side: HashMap<String, String> = HashMap::new();
  side.insert("0.0.0".to_string(), "list-val".to_string());
  side.insert("0.0".to_string(), "scalar-val".to_string());
  side.insert("2.0".to_string(), "RotatingKVCache".to_string());
  let (info, _user, _classes) = unflatten_side(side).unwrap();
  assert_eq!(
    info.get(&0).map(Vec::as_slice),
    Some(&["list-val".to_string()][..]),
    "the more-specific list meta wins over the scalar form"
  );
}

#[test]
fn unflatten_side_skips_unknown_tag_and_non_numeric_indices() {
  // An unknown leading tag ("9.0") is ignored; a non-numeric class index
  // ("2.x") / scalar index ("0.x") / list index ("0.x.0") is skipped
  // (swift parity), never an error.
  let mut side: HashMap<String, String> = HashMap::new();
  side.insert("9.0".to_string(), "ignored".to_string());
  side.insert("2.x".to_string(), "KVCache".to_string());
  side.insert("0.x".to_string(), String::new());
  side.insert("0.x.0".to_string(), "v".to_string());
  let (info, user, classes) = unflatten_side(side).unwrap();
  assert!(info.is_empty(), "no parseable meta indices");
  assert!(user.is_empty(), "no `1.` keys");
  assert!(classes.is_empty(), "no parseable `2.{{i}}` class index");
}

#[test]
fn unflatten_side_non_dense_classes_is_err() {
  // Classes at {0,2}, gap at 1 -> dense_len("class") -> LengthMismatch
  // (direct-unit twin of the file-level `non_dense_class_indices_is_err`).
  let mut side: HashMap<String, String> = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("2.2".to_string(), "KVCache".to_string());
  match unflatten_side(side) {
    Err(Error::LengthMismatch(p)) => {
      assert_eq!(
        p.context(),
        "prompt cache: non-dense class indices (corrupt or incompatible file)"
      );
      assert_eq!(p.expected(), 2);
      assert_eq!(p.actual(), 3);
    }
    other => panic!("non-dense classes must be LengthMismatch, got {other:?}"),
  }
}

// ─────────────────── save_prompt_cache closed-form layout ──────────────

#[test]
fn save_no_meta_cache_emits_scalar_zero_i_empty_string() {
  // CLOSED-FORM: a no-meta KVCache must emit the mlx-lm scalar `"0.{i}"=""`
  // (mandatory for mlx-lm cross-load), the class under `"2.{i}"`, and NO
  // `"0.{i}.{j}"` list key. An EMPTY cache emits no `"{i}.{j}"` array keys.
  let path = temp_path("scalar_meta.safetensors");
  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(StandardKvCache::new())];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  let (arrays, side) = crate::io::load_safetensors_with_metadata(&path).unwrap();
  assert!(arrays.is_empty(), "empty-state cache writes no array keys");
  assert_eq!(side.get("2.0").map(String::as_str), Some("KVCache"));
  assert_eq!(
    side.get("0.0").map(String::as_str),
    Some(""),
    "no-meta cache emits the scalar empty-string `0.0`"
  );
  assert!(!side.contains_key("0.0.0"), "scalar form, not a list");
  let _ = std::fs::remove_file(&path);
}

#[test]
fn empty_cache_list_round_trips_to_zero_caches() {
  // Saving an EMPTY `&[]` slice writes a file with no class/array/meta
  // keys; loading it yields zero caches and empty metadata (Python's
  // `all([])`/`zip([])` faithful empty case). User metadata still survives.
  let path = temp_path("empty_slice.safetensors");
  let cache: Vec<Box<dyn KvCache>> = Vec::new();
  let mut meta = HashMap::new();
  meta.insert("model".to_string(), "demo".to_string());
  save_prompt_cache(&path, &cache, &meta).unwrap();

  let (loaded, loaded_meta) = load_prompt_cache(&path).unwrap();
  assert!(loaded.is_empty(), "no classes -> zero caches");
  assert_eq!(loaded_meta.get("model").map(String::as_str), Some("demo"));
  let _ = std::fs::remove_file(&path);
}

#[test]
fn save_then_load_rotating_round_trips_meta_and_offset() {
  // ROUND-TRIP through the in-module fns for a RotatingKVCache (4-element
  // list meta_state path): recovered offset + meta_state == original.
  let path = temp_path("rotating_rt.safetensors");
  let mut c = RotatingKvCache::new(8, 4);
  c.update(&kv(&[1.0, 2.0, 3.0]), &kv(&[4.0, 5.0, 6.0]))
    .unwrap();
  let want_offset = c.offset();
  let want_meta = c.meta_state();
  assert_eq!(
    want_meta.len(),
    4,
    "rotating meta is (keep,max_size,offset,idx)"
  );

  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(c)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();
  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(loaded[0].reference_class_name(), "RotatingKVCache");
  assert_eq!(loaded[0].offset(), want_offset);
  assert_eq!(loaded[0].meta_state(), want_meta);
  let _ = std::fs::remove_file(&path);
}

// ─────────────── load_prompt_cache path-gate typed errors ──────────────

#[test]
fn load_missing_file_is_fileio_open_err() {
  // A path that does not exist fails the pre-`crate::io` open gate with a
  // FileIo(FileOp::Open) carrying the path — never a panic.
  let path = temp_path("does_not_exist.safetensors");
  let _ = std::fs::remove_file(&path); // ensure absent
  match load_prompt_cache(&path) {
    Err(Error::FileIo(p)) => {
      assert_eq!(p.op(), FileOp::Open);
      assert_eq!(p.context(), "cannot open prompt cache");
      assert_eq!(p.path(), path.as_path());
    }
    Err(e) => panic!("missing file must be FileIo(Open), got Err({e:?})"),
    Ok(_) => panic!("missing file must be FileIo(Open), got Ok"),
  }
}

#[test]
fn load_directory_is_not_regular_file_err() {
  // A directory has `metadata().len() == 0` yet is non-regular; the
  // post-open `is_file()` fstat gate rejects it as FileIo(FileOp::Open)
  // BEFORE the path reaches mlx-c — never a panic / unbounded read.
  let dir = temp_path("a_directory");
  std::fs::create_dir_all(&dir).unwrap();
  match load_prompt_cache(&dir) {
    Err(Error::FileIo(p)) => {
      assert_eq!(p.op(), FileOp::Open);
      assert_eq!(
        p.context(),
        "prompt cache target is not a regular file; refusing to read"
      );
    }
    Err(e) => {
      let _ = std::fs::remove_dir_all(&dir);
      panic!("a directory must be FileIo(Open, not-regular), got Err({e:?})");
    }
    Ok(_) => {
      let _ = std::fs::remove_dir_all(&dir);
      panic!("a directory must be FileIo(Open, not-regular), got Ok");
    }
  }
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_oversized_file_is_cap_exceeded_err() {
  // The size gate rejects a regular file whose authoritative fstat len()
  // exceeds MAX_PROMPT_CACHE_BYTES with CapExceeded, BEFORE reading a
  // single byte. Use a SPARSE file (`set_len` past the cap) so the test
  // costs no real disk — only the file's *reported* length matters to the
  // O(1) gate. CLOSED-FORM cap == MAX_PROMPT_CACHE_BYTES.
  let path = temp_path("oversized.safetensors");
  let f = std::fs::File::create(&path).unwrap();
  let huge = MAX_PROMPT_CACHE_BYTES + 1;
  f.set_len(huge).unwrap();
  drop(f);
  match load_prompt_cache(&path) {
    Err(Error::CapExceeded(p)) => {
      assert_eq!(p.cap_name(), "MAX_PROMPT_CACHE_BYTES");
      assert_eq!(p.cap(), MAX_PROMPT_CACHE_BYTES);
      assert_eq!(p.observed(), huge);
      assert_eq!(
        p.context(),
        "load_prompt_cache: file size; refusing to read"
      );
    }
    Err(e) => {
      let _ = std::fs::remove_file(&path);
      panic!("oversized file must be CapExceeded, got Err({e:?})");
    }
    Ok(_) => {
      let _ = std::fs::remove_file(&path);
      panic!("oversized file must be CapExceeded, got Ok");
    }
  }
  let _ = std::fs::remove_file(&path);
}

#[test]
fn load_truncated_garbage_payload_is_err_not_panic() {
  // A regular, in-bounds file whose BYTES are not valid safetensors fails
  // inside mlx-c's parser. The exact variant is mlx-c's (not persist.rs's
  // contract), so assert only `is_err()` + no panic.
  let path = temp_path("garbage.safetensors");
  std::fs::write(&path, b"not a safetensors file at all").unwrap();
  assert!(
    load_prompt_cache(&path).is_err(),
    "a garbage payload must be a recoverable Err, never a panic"
  );
  let _ = std::fs::remove_file(&path);
}

// ─────────────── load_prompt_cache reconstruction-gate payloads ────────

#[test]
fn load_wrong_rank_kv_state_is_layerkeyed_rankmismatch() {
  // A KVCache (in KV_RANK_KINDS) with a present but RANK-1 state array is
  // rejected by the 4-D rank gate as LayerKeyed(RankMismatch) BEFORE
  // from_state — pins the exact nested payload + observed shape (the
  // integration suite only asserts `is_err()` for the rotating variant).
  let path = temp_path("wrong_rank_kv.safetensors");
  let mut arrays = HashMap::new();
  arrays.insert(
    "0.0".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
  );
  arrays.insert(
    "0.1".to_string(),
    Array::from_slice::<f32>(&[4.0, 5.0, 6.0], &(3usize,)).unwrap(),
  );
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("0.0".to_string(), String::new());
  crate::io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  match load_prompt_cache(&path) {
    Err(Error::LayerKeyed(p)) => {
      assert!(
        p.layer().contains("cache 0") && p.layer().contains("KVCache"),
        "layer key must name cache 0 + the kind, got: {}",
        p.layer()
      );
      match p.inner() {
        Error::RankMismatch(r) => {
          assert_eq!(r.actual(), 1, "observed rank of the corrupt array");
          assert_eq!(r.actual_shape(), &[3]);
        }
        other => panic!("inner must be RankMismatch, got {other:?}"),
      }
    }
    Err(e) => panic!("wrong-rank KV state must be LayerKeyed(RankMismatch), got Err({e:?})"),
    Ok(_) => panic!("wrong-rank KV state must be LayerKeyed(RankMismatch), got Ok"),
  }
  let _ = std::fs::remove_file(&path);
}

#[test]
fn load_no_meta_kind_with_truthy_meta_is_layerkeyed_invariant() {
  // A KVCache (NO_META_KINDS) carrying a non-empty meta_state is rejected
  // by the emptiness gate as LayerKeyed(InvariantViolation) — pin the
  // exact nested payload (integration suite asserts only `is_err()`).
  // Use a 4-D state so the rank gate passes and the META gate is what
  // fires.
  let path = temp_path("truthy_meta_kv.safetensors");
  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0, 2.0]));
  arrays.insert("0.1".to_string(), kv(&[3.0, 4.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("0.0".to_string(), "garbage".to_string()); // truthy scalar
  crate::io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  match load_prompt_cache(&path) {
    Err(Error::LayerKeyed(p)) => {
      assert!(
        p.layer().contains("cache 0") && p.layer().contains("KVCache"),
        "layer key must name cache 0 + the kind, got: {}",
        p.layer()
      );
      match p.inner() {
        Error::InvariantViolation(iv) => {
          assert_eq!(iv.requirement(), "must have empty meta_state");
        }
        other => panic!("inner must be InvariantViolation, got {other:?}"),
      }
    }
    Err(e) => {
      panic!("no-meta kind + truthy meta must be LayerKeyed(InvariantViolation), got Err({e:?})")
    }
    Ok(_) => {
      panic!("no-meta kind + truthy meta must be LayerKeyed(InvariantViolation), got Ok")
    }
  }
  let _ = std::fs::remove_file(&path);
}

// ─────────────────── can_trim / trim_prompt_cache ──────────────────────

#[test]
fn can_trim_empty_is_vacuously_true_and_trim_returns_zero() {
  // Python `all([]) == True`; trim on empty returns 0 (the `len == 0`
  // short-circuit), never an error.
  let mut empty: Vec<Box<dyn KvCache>> = Vec::new();
  assert!(can_trim_prompt_cache(&empty), "all([]) is vacuously true");
  assert_eq!(trim_prompt_cache(&mut empty, 5).unwrap(), 0);
}

#[test]
fn can_trim_false_when_any_cache_not_trimmable_and_trim_is_zero() {
  // ArraysCache (e.g. a Mamba/SSM state cache) does not override
  // is_trimmable, so it inherits the trait default `false` and is
  // genuinely non-trimmable in every state — unlike RotatingKvCache,
  // whose is_trimmable is `offset < max_size` (a fresh one is trimmable).
  // So a mixed list with it is not trimmable and trim returns 0 without
  // mutating anything.
  let mut std_c = StandardKvCache::new();
  std_c.update(&kv(&[1.0, 2.0]), &kv(&[3.0, 4.0])).unwrap();
  let arrays_c = ArraysCache::mamba();
  let mut cache: Vec<Box<dyn KvCache>> = vec![Box::new(std_c), Box::new(arrays_c)];
  assert!(
    !can_trim_prompt_cache(&cache),
    "a non-trimmable member makes the whole list non-trimmable"
  );
  assert_eq!(
    trim_prompt_cache(&mut cache, 1).unwrap(),
    0,
    "not-trimmable -> 0 trimmed, nothing mutated"
  );
  assert_eq!(cache[0].offset(), 2, "trimmable member left untouched");
}

#[test]
fn trim_all_trimmable_returns_first_cache_count() {
  // All-StandardKvCache (trimmable): every cache is trimmed; the returned
  // count is cache[0]'s (mlx-lm `[...][0]`). Each layer's KV is the same
  // length, so all trims agree.
  let mut c0 = StandardKvCache::new();
  c0.update(&kv(&[1.0, 2.0, 3.0]), &kv(&[4.0, 5.0, 6.0]))
    .unwrap();
  let mut c1 = StandardKvCache::new();
  c1.update(&kv(&[7.0, 8.0, 9.0]), &kv(&[1.0, 1.0, 1.0]))
    .unwrap();
  let mut cache: Vec<Box<dyn KvCache>> = vec![Box::new(c0), Box::new(c1)];
  let trimmed = trim_prompt_cache(&mut cache, 2).unwrap();
  assert_eq!(
    trimmed, 2,
    "min(offset=3, n=2) == 2, reported from cache[0]"
  );
  assert_eq!(cache[0].offset(), 1, "cache 0 trimmed to offset 1");
  assert_eq!(
    cache[1].offset(),
    1,
    "cache 1 ALSO trimmed (list-comp trims all)"
  );
}

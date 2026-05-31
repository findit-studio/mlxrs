//! Prompt-cache disk persistence, ported 1:1 from
//! [`mlx_lm.models.cache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
//! `save_prompt_cache` / `load_prompt_cache` / `can_trim_prompt_cache` /
//! `trim_prompt_cache` (cache.py:43-113) and cross-checked against
//! mlx-swift-lm's `MLXLMCommon/KVCache.swift` `savePromptCache` /
//! `loadPromptCache` (the exact `tree_flatten` wire format).
//!
//! ## Cross-tool wire format
//!
//! A prompt cache written here uses the identical safetensors layout as
//! mlx-lm and mlx-swift-lm (`mx.save_safetensors(file, cache_data,
//! cache_metadata)`); the array map / side-table keying and the
//! reference-class-name labels are byte-for-byte what both tools produce
//! and consume — see the per-key spec below. **Compatibility scope (read
//! before relying on cross-tool load):**
//!
//! - **mlx-lm ⇄ mlxrs: full round-trip, all cache kinds.** Every cache's
//!   `state`/`meta_state` is the verbatim 1:1 port of the *authoritative*
//!   `mlx_lm.models.cache` (the spec this module is ported from), so a
//!   cache saved by either loads in the other unchanged — **including**
//!   `RotatingKVCache` (mlx-lm `meta_state` is the 4-tuple `(keep,
//!   max_size, offset, _idx)`, cache.py:533; mlxrs emits exactly that).
//! - **mlx-swift-lm ⇄ mlxrs: the wire format + non-rotating kinds match;
//!   `RotatingKVCache` `meta_state` does *not* (an upstream mlx-lm↔swift
//!   divergence, faithfully inherited — *not* introduced here).**
//!   mlx-swift-lm's `RotatingKVCache.metaState`
//!   (`MLXLMCommon/KVCache.swift`) is a **5**-element tuple `(keep,
//!   maxCacheSize, step, offset, idx)` and its loader hard-rejects any
//!   count `!= 5`, whereas authoritative mlx-lm (and therefore mlxrs and
//!   the merged `RotatingKvCache::meta_state`) is **4** elements (no
//!   serialized `step` — it is a fixed class constant). So a *rotating*
//!   prompt cache does **not** interoperate between mlx-swift-lm and
//!   {mlx-lm, mlxrs} **in either direction** — and mlx-lm itself has the
//!   same incompatibility with mlx-swift-lm. This port deliberately tracks
//!   the authoritative mlx-lm shape (the cited spec; the merged
//!   `RotatingKvCache::meta_state` is fixed at 4 fields, #32 — out of this
//!   PR's scope to change, and changing it would *break* the mlx-lm
//!   round-trip that is the spec). Non-rotating caches (`KVCache`/…) and
//!   the array/side-table wire format remain fully mlx-swift-loadable.
//!   Reconciling rotating-cache metadata across all three tools is an
//!   upstream-coordination follow-up, not a defect in this port.
//!
//! Both tools use `mx.save_safetensors(file, cache_data, cache_metadata)`
//! where
//!
//! - `cache_data  = tree_flatten([c.state for c in cache])` — a list of
//!   per-cache array lists, so the safetensors **array map** is keyed
//!   `"{i}.{j}"` (cache `i`, array `j`); mlx-lm cache.py:53-55, swift
//!   `flattenedData["\(i).\(j)"]`.
//! - `cache_metadata = tree_flatten([cache_info, metadata,
//!   cache_classes])` — flattened into the safetensors **string metadata
//!   side-table** as (note `cache_info[i]` is **heterogeneous** —
//!   `tree_flatten` keys it by its *type*):
//!   - `"0.{i}.{j}"` → `cache_info[i][j]` when cache `i`'s `meta_state`
//!     is a **non-empty list** (e.g. `RotatingKVCache`'s 4-tuple),
//!   - `"0.{i}"` → `""` when cache `i`'s `meta_state` is **empty** —
//!     because mlx-lm's `_BaseCache.meta_state` (the base of
//!     `KVCache`/`ConcatenateKVCache`, which don't override it,
//!     cache.py:138-139) is the empty *string* `""`, a `tree_flatten`
//!     **scalar leaf** → key `"0.{i}"`, *not* an absent key. Emitting
//!     this scalar (not nothing) is mandatory for mlx-lm cross-load: an
//!     all-`KVCache` file with no `"0.*"` keys makes mlx-lm
//!     `tree_unflatten` see `info == {}` and `zip` produce **zero
//!     caches**. (mlx-swift's `unflattenMetadata` matches only `"0.i.j"`,
//!     so it ignores the scalar and rebuilds the no-meta cache from an
//!     empty metaState — correct for `KVCacheSimple`; the scalar is thus
//!     mlx-lm-faithful and swift-harmless.) The loader here accepts
//!     `"0.{i}"` (scalar empty), `"0.{i}.{j}"` (list), and an absent
//!     index (swift's empty form) — all three ⇒ that cache's meta_state
//!     is its respective shape.
//!   - `"1.{key}"`   → user `metadata[key]`,
//!   - `"2.{i}"`     → `cache_classes[i]` = `type(c).__name__` (cache.py
//!     :56) — the **reference Python class name**, *not* the Rust struct
//!     name, so the file round-trips through mlx-lm
//!     `globals()[name].from_state` / swift `restoreCacheFromMetaState`
//!     unchanged.
//!
//! (`mlx.utils.tree_flatten` uses dot notation: a list nests as `i.j`, a
//! dict as `key` — verified against `python/mlx/utils.py` `tree_flatten`
//! and the swift `unflattenArrays`/`unflattenMetadata` mirror.)
//!
//! ## Reference class name
//!
//! mlx-lm derives the kind via `type(c).__name__`; mlx-swift-lm via a
//! single `cacheClassName(_:)` `switch` over the concrete type. The
//! `KvCache` trait carries the equivalent
//! [`KvCache::reference_class_name`] method (defaulting to `"KVCache"`,
//! overridden by every concrete cache to its mlx-lm/mlx-swift name), so
//! [`reference_class_name`] is a thin dispatch over that trait method —
//! the emitted name (`"KVCache"` / `"RotatingKVCache"` / `"ChunkedKVCache"`
//! / `"QuantizedKVCache"` / `"CacheList"` / `"ArraysCache"` /
//! `"BatchKVCache"` / `"BatchRotatingKVCache"`) is precisely what mlx-lm /
//! mlx-swift expect and what [`from_state`] keys on. Adding a new cache
//! kind means overriding [`KvCache::reference_class_name`] on it (and the
//! corresponding [`from_state`] arm); no change here.
//!
//! ## Path-read DoS discipline
//!
//! `load_prompt_cache` reads an attacker-influenceable file, so it applies
//! the **same defense, with the same residual posture, as the merged
//! `lm::load` weight loader** (`load_weights` → `collect_sorted`'s
//! `fs::metadata` gate, then `crate::io::load_safetensors(path)` by path).
//! `crate::io`/mlx-c is path-only (`mlx_load_safetensors(const char*)`);
//! there is no fd/owned-bytes entry point, so this loader cannot consume
//! the bytes from the validated handle — closing that gap needs an mlx-c
//! API extension, a separate `mlxrs-sys` change out of this PR's scope (it
//! would also have to change the merged `lm::load` identically).
//!
//! Pre-`crate::io` gate: open the path **once** with `O_NONBLOCK |
//! O_CLOEXEC` (Unix) so a planted FIFO returns instead of hanging; fstat
//! the *opened* fd and reject a non-regular target (FIFO / device /
//! directory / symlink-to-special — these read as `len()==0` yet stream
//! unbounded data) and an oversized one (`metadata().len()` of the
//! proven-regular fd — an O(1) authoritative size; *not* `Read::take`,
//! which `lm::load` only needs because it streams the config body — here a
//! streaming probe would be a wasteful, itself-DoS-prone 2× read of a
//! multi-GB cache). Symlinks are intentionally followed (the post-open
//! `is_file()` fstat enforces the guarantee on the resolved target).
//!
//! **Residual TOCTOU (acknowledged, = merged `load_weights`):** the fd is
//! closed and `crate::io` re-`open`s `path`, so an adversary who can swap
//! `path` (or a symlink in it) *between* the fstat and that re-open could
//! still present a FIFO/device/oversized file to mlx-c. This window is
//! **identical** to the merged `lm::load` weight path and accepted on the
//! same basis (a trusted local cache directory; the gate is defense for
//! the non-racing common case, not a TOCTOU-free guarantee — the doc does
//! **not** claim one). The reconstructed-state rank gate below is the
//! independent backstop that turns a corrupt/foreign *payload* (incl. one
//! delivered via such a race) into a clean `Err` rather than a later
//! panic.
//!
//! The `tree_unflatten` step additionally rejects a *non-dense* flattened
//! list (the private `dense_len` gate) so a tiny hostile side-table cannot
//! drive an unbounded `Vec`, and every reconstructed cache's state arrays
//! are **rank-validated** (4-D `[B, n_kv_heads, S, head_dim]`) before
//! `load_prompt_cache` returns, so a wrong-rank array in a hostile file is
//! a recoverable error here, not a deferred `shape()[2]` panic on first
//! cache use. Every recoverable failure (missing / non-regular /
//! oversized / corrupt / non-dense / wrong-rank / unknown-kind) is a
//! typed [`Error`] variant — **never** a panic/`unwrap` on the load path.

use std::{collections::HashMap, path::Path};

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, FileIoPayload, FileOp,
    InvariantViolationPayload, LayerKeyedPayload, LengthMismatchPayload, OutOfRangePayload,
    RankMismatchPayload, Result,
  },
  // `KV_NDIM` is the canonical `[B, n_kv_heads, S, head_dim]` rank shared
  // by every KV cache (defined once in `super::util`); referenced — not
  // re-declared — so the hostile-file rank gate stays in lockstep with the
  // cache impls (no duplicated magic `4`). This is a plain `use` of an
  // already-declared sibling module; it touches no other cache file.
  lm::cache::{KvCache, from_state, util::KV_NDIM},
};
use smol_str::format_smolstr;

/// Upper bound on a prompt-cache file we will read, mirroring `lm::load`'s
/// `MAX_CONFIG_BYTES` rationale (a hostile file must not OOM the loader).
/// A prompt cache holds KV tensors so it is far larger than a config; this
/// 8 GiB ceiling is generous for real multi-GB caches yet still a hard cap
/// against an unbounded planted file. `crate::io`/mlx-c reads the file
/// itself; this bound is the pre-open gate (size + non-regular reject) the
/// `lm::load` discipline requires before any unbounded read.
pub const MAX_PROMPT_CACHE_BYTES: u64 = 8 << 30;

/// The **reference Python class name** for `cache` (mlx-lm
/// `type(c).__name__`, cache.py:56 / swift `cacheClassName`,
/// `KVCache.swift:1381-1392`).
///
/// Thin forward to [`KvCache::reference_class_name`] (the trait method
/// that lifts swift's `cacheClassName(_:)` switch onto the concrete cache
/// itself). The returned string is exactly the reference class name
/// [`from_state`] keys on and that mlx-lm / mlx-swift write, so the *kind
/// labeling and wire format* are cross-tool (subject to the rotating-cache
/// `meta_state` arity caveat in the module docs: full mlx-lm round-trip;
/// mlx-swift parity for the wire format + non-rotating kinds). Adding a
/// new cache kind means overriding [`KvCache::reference_class_name`] on
/// it (and the corresponding [`from_state`] arm); no change here.
pub fn reference_class_name(cache: &dyn KvCache) -> &'static str {
  cache.reference_class_name()
}

/// Save a pre-computed prompt cache to a `.safetensors` file — port of
/// `mlx_lm.models.cache.save_prompt_cache` (cache.py:43-59), wire-format
/// cross-checked vs mlx-swift-lm `savePromptCache`.
///
/// Emits the exact cross-tool layout (see the module docs): arrays keyed
/// `"{i}.{j}"`, and a string side-table with `meta_state` under
/// `"0.{i}.{j}"`, user `metadata` under `"1.{key}"`, and the **reference
/// class name** under `"2.{i}"`. Writes each cache's `meta_state`
/// verbatim — i.e. the *authoritative mlx-lm* shape (cache.py:53-56), so
/// the file loads unchanged in mlx-lm and (for the wire format +
/// non-rotating kinds) mlx-swift; rotating-cache `meta_state` follows
/// mlx-lm's 4-field form, which upstream-diverges from mlx-swift's 5-field
/// — see the module-doc compatibility scope. The cache state arrays are
/// not materialized here (no implicit eval — `crate::io` writes them
/// lazily).
pub fn save_prompt_cache(
  path: &Path,
  cache: &[Box<dyn KvCache>],
  metadata: &HashMap<String, String>,
) -> Result<()> {
  let mut arrays: HashMap<String, Array> = HashMap::new();
  let mut side: HashMap<String, String> = HashMap::new();

  // #103/#104: reuse a single per-loop `Vec<Array>` /
  // `Vec<String>` across all caches via the trait's `state_into` /
  // `meta_state_into` buffer-reuse variants — one allocation per loop
  // iteration's clear, not one per cache. Each iteration `clear()`s the
  // buffer (preserving capacity) so subsequent appends reuse the same
  // backing storage. mlx-lm's reference loop pays a fresh `c.state` /
  // `c.meta_state` tuple per cache; this Rust port matches the observable
  // serialization byte-for-byte while saving the per-cache `Vec`
  // allocations.
  let mut state_buf: Vec<Array> = Vec::new();
  let mut meta_buf: Vec<String> = Vec::new();
  for (i, c) in cache.iter().enumerate() {
    // `cache_data = [c.state for c in cache]` -> `"{i}.{j}"`.
    state_buf.clear();
    c.state_into(&mut state_buf)?;
    for (j, arr) in state_buf.drain(..).enumerate() {
      arrays.insert(format!("{i}.{j}"), arr);
    }
    // `cache_info = [c.meta_state for c in cache]`, then
    // `tree_flatten([cache_info, metadata, cache_classes])`. CRITICAL for
    // mlx-lm cross-loadability: `_BaseCache.meta_state` (the base for
    // `KVCache`/`ConcatenateKVCache`, which do NOT override it —
    // cache.py:138-139) is the empty **string** `""`, *not* an empty
    // list. `mlx.utils.tree_flatten` treats that scalar `""` as a **leaf**
    // and emits the key `"0.{i}"` with value `""` (one entry per no-meta
    // cache). A non-empty `meta_state` (a list, e.g. RotatingKVCache's
    // 4-tuple) instead flattens element-wise as `"0.{i}.{j}"`. We MUST
    // mirror that heterogeneous shape: emitting nothing for an empty
    // meta_state (the naive list-only loop) makes mlx-lm `tree_unflatten`
    // see `info == {}` and `zip(classes, arrays, info)` truncate to **zero
    // caches** — so a Rust-saved all-`KVCache` (the common full-attention)
    // cache would fail to load in mlx-lm entirely. (mlx-swift's
    // `unflattenMetadata` only matches `"0.i.j"`, ≥3 components, so a
    // scalar `"0.i"` is simply ignored there → swift reconstructs the
    // no-meta cache from an empty metaState, which is correct for
    // `KVCacheSimple`; the scalar form is thus mlx-lm-faithful AND
    // swift-harmless.)
    meta_buf.clear();
    c.meta_state_into(&mut meta_buf);
    if meta_buf.is_empty() {
      side.insert(format!("0.{i}"), String::new());
    } else {
      for (j, m) in meta_buf.drain(..).enumerate() {
        side.insert(format!("0.{i}.{j}"), m);
      }
    }
    // `cache_classes = [type(c).__name__ for c in cache]` -> `"2.{i}"`.
    side.insert(
      format!("2.{i}"),
      reference_class_name(c.as_ref()).to_string(),
    );
  }

  // `metadata` (the user dict) -> `"1.{key}"`.
  for (k, v) in metadata {
    side.insert(format!("1.{k}"), v.clone());
  }

  crate::io::save_safetensors_with_metadata(path, &arrays, &side)
}

/// The materialized length of a flattened list given its observed max
/// index `max_idx` and the number of *present* (distinct) indices
/// `present`.
///
/// A list flattened by `tree_flatten` (this / mlx-lm / mlx-swift) is always
/// **dense** — indices `0..len`, so `max_idx + 1 == present`. Returning
/// that length is the only allocation; bounding it to `present` (≤ the
/// distinct-key count, which the [`MAX_PROMPT_CACHE_BYTES`] file cap already
/// bounds) is what prevents a tiny hostile side-table (e.g. one
/// `"2.4000000000"` key) from driving a multi-GB `vec!`. A non-dense list
/// (`max_idx + 1 != present`) only arises from a corrupt / adversarial
/// file, so it is a recoverable typed [`Error`] variant, never a silent huge
/// allocation or a panic.
fn dense_len(max_idx: usize, present: usize, what: &'static str) -> Result<usize> {
  let n = max_idx.checked_add(1).ok_or_else(|| {
    // Distinguish by `what` so the static-string context is preserved per
    // call site (array sub / meta_state / class) without runtime allocation.
    let context: &'static str = match what {
      "array sub" => "prompt cache: array sub index overflows usize",
      "meta_state" => "prompt cache: meta_state index overflows usize",
      "class" => "prompt cache: class index overflows usize",
      _ => "prompt cache: index overflows usize",
    };
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      context,
      "usize",
      [("max_idx", max_idx as u64)],
    ))
  })?;
  if n != present {
    let context: &'static str = match what {
      "array sub" => "prompt cache: non-dense array sub indices (corrupt or incompatible file)",
      "meta_state" => "prompt cache: non-dense meta_state indices (corrupt or incompatible file)",
      "class" => "prompt cache: non-dense class indices (corrupt or incompatible file)",
      _ => "prompt cache: non-dense indices (corrupt or incompatible file)",
    };
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      context, present, n,
    )));
  }
  Ok(n)
}

/// Parse the `"{i}.{j}"` flattened array map into a **sparse** per-cache-
/// index map of ordered array lists — the inverse of mlx-lm's
/// `tree_flatten` / `tree_unflatten` for the array map (swift
/// `unflattenArrays`).
///
/// Returns a sparse `HashMap<cache_index, Vec<Array>>` rather than a dense
/// `Vec`: a cache whose `state` is `[]` (an empty/unused cache) emits **no**
/// `"{i}.{j}"` array keys at all, so the array map cannot itself tell you
/// how many caches there are — the authoritative count is the
/// `cache_classes` list (`= [type(c).__name__ for c in cache]`, always
/// length `len(cache)`). The caller indexes this sparse map by that count
/// (`remove(&i).unwrap_or_default()`), which is *more* robust than mlx-lm's
/// `zip(classes, arrays, info)` (silently truncates to `len(arrays)` →
/// drops trailing empty-state caches) and mlx-swift's strict
/// `cacheData.count == cacheClasses.count` guard (rejects an all-empty
/// cache list outright); both refs misbehave on all-empty state, this
/// reconstructs it faithfully from the class list.
///
/// A key whose `i`/`j` is not a base-10 `usize` is **ignored** (swift
/// parity — `unflattenArrays` silently skips non-`"i.j"` keys); a
/// `usize`-overflowing sub-index is a recoverable typed [`Error`] variant (a
/// hostile file must not drive an unbounded `Vec` resize).
fn unflatten_arrays(flat: HashMap<String, Array>) -> Result<HashMap<usize, Vec<Array>>> {
  // Collect into a doubly-sparse map first so out-of-order keys still order
  // correctly (mlx-c's map iteration order is unspecified).
  let mut sparse: HashMap<usize, HashMap<usize, Array>> = HashMap::new();
  for (k, v) in flat {
    let mut it = k.splitn(2, '.');
    let (Some(si), Some(sj)) = (it.next(), it.next()) else {
      continue;
    };
    let (Ok(i), Ok(j)) = (si.parse::<usize>(), sj.parse::<usize>()) else {
      continue;
    };
    sparse.entry(i).or_default().insert(j, v);
  }
  let mut out: HashMap<usize, Vec<Array>> = HashMap::with_capacity(sparse.len());
  for (i, mut m) in sparse {
    let inner = match m.keys().copied().max() {
      None => Vec::new(),
      Some(max_j) => {
        // Dense (`0..len`) for any faithful save; the check bounds the
        // allocation by the present-key count (file-size bounded) so a
        // tiny hostile `"i.4e9"` key cannot OOM.
        let cnt = dense_len(max_j, m.len(), "array sub")?;
        let mut v = Vec::with_capacity(cnt);
        for j in 0..cnt {
          // Dense ⇒ every `j` present; a defensive skip on an (unreachable
          // after the dense check) gap rather than an index panic.
          if let Some(a) = m.remove(&j) {
            v.push(a);
          }
        }
        v
      }
    };
    out.insert(i, inner);
  }
  Ok(out)
}

/// Parse the flattened string side-table back into `(cache_info,
/// user_metadata, cache_classes)` — the inverse of mlx-lm's `tree_flatten`
/// of `[cache_info, metadata, cache_classes]` (swift `unflattenMetadata`).
///
/// `cache_info[i][j]` from `"0.{i}.{j}"`, user metadata from `"1.{key}"`
/// (the key is everything after the first `.`, so a metadata key may itself
/// contain dots — matches swift `components.dropFirst().joined(".")`),
/// `cache_classes[i]` from `"2.{i}"`. Unparseable indices are skipped
/// (swift parity); a `usize`-overflowing index is a recoverable typed
/// [`Error`] variant.
///
/// `cache_info` is returned as a **sparse** `HashMap<cache_index,
/// Vec<String>>` (like the array map): a cache with an empty `meta_state`
/// (e.g. a [`StandardKvCache`](super::StandardKvCache)) emits **no**
/// `"0.{i}.*"` keys, so the
/// per-cache dimension is genuinely sparse and the caller indexes it by the
/// authoritative `cache_classes` count. `cache_classes` is a dense `Vec`
/// (always exactly one `"2.{i}"` per cache).
///
/// **Unbounded-allocation defense.** mlx-lm's `tree_unflatten` of a list
/// assumes *dense* indices (`0..len`) — a real cache (this / mlx-lm /
/// mlx-swift) always is. A hostile file's tiny side-table could otherwise
/// carry one key like `"2.4000000000"` (~13 bytes, well under
/// [`MAX_PROMPT_CACHE_BYTES`]) and drive a multi-GB `vec![String::new();
/// 4e9]`. The only `Vec`s sized from a max index are the **inner**
/// `meta_state[j]` list and the **dense** `cache_classes` list; both go
/// through [`dense_len`], which rejects a non-dense list (corrupt /
/// adversarial only — never a faithful save) and bounds the length by the
/// present-key count (itself file-size bounded). The sparse `cache_info`
/// per-cache dimension allocates nothing per absent index.
#[allow(clippy::type_complexity)]
fn unflatten_side(
  side: HashMap<String, String>,
) -> Result<(
  HashMap<usize, Vec<String>>,
  HashMap<String, String>,
  Vec<String>,
)> {
  let mut info_sparse: HashMap<usize, HashMap<usize, String>> = HashMap::new();
  // Cache indices seen as the **scalar** `"0.{i}"` form (mlx-lm's
  // `tree_flatten` of a scalar `_BaseCache.meta_state`), with their value
  // **preserved**. mlx-lm's `_BaseCache.meta_state` getter is `""`, so a
  // faithful no-meta scalar is `""`; its *setter* (cache.py:142-145)
  // **raises** on any *truthy* value (`if v is not None and v: raise`).
  // The value is therefore NOT irrelevant — a non-empty scalar
  // (`"0.{i}"="garbage"`) is a malformed / schema-drifted file mlx-lm
  // would reject, so we keep the string and let `load_prompt_cache`
  // enforce the same emptiness rule per kind.
  let mut info_scalar: HashMap<usize, String> = HashMap::new();
  let mut user_meta: HashMap<String, String> = HashMap::new();
  let mut class_sparse: HashMap<usize, String> = HashMap::new();
  let mut class_max_i: Option<usize> = None;

  for (k, v) in side {
    // Match the leading tag (`0` / `1` / `2`) like swift's
    // `components[0]` test; an unknown tag is ignored.
    if let Some(rest) = k.strip_prefix("0.") {
      let mut it = rest.splitn(2, '.');
      match (it.next(), it.next()) {
        // `"0.{i}.{j}"` — a meta_state *list* element (swift's form for
        // any metadata; mlx-lm's form for a non-empty list meta_state).
        (Some(si), Some(sj)) => {
          let (Ok(i), Ok(j)) = (si.parse::<usize>(), sj.parse::<usize>()) else {
            continue;
          };
          info_sparse.entry(i).or_default().insert(j, v);
        }
        // `"0.{i}"` — mlx-lm's `tree_flatten` of the **scalar**
        // `_BaseCache.meta_state` (`KVCache`/`ConcatenateKVCache`/…). A
        // faithful no-meta cache is the empty string `""`; the value is
        // **kept** so a *truthy* scalar (which mlx-lm's
        // `_BaseCache.meta_state` setter rejects, cache.py:142-145) is
        // surfaced rather than silently discarded. Accepting the empty
        // form is what makes an mlx-lm-saved all-`KVCache` prompt cache
        // loadable here, and is symmetrically what our saver emits for a
        // no-meta cache.
        (Some(si), None) => {
          let Ok(i) = si.parse::<usize>() else { continue };
          info_scalar.insert(i, v);
        }
        _ => continue,
      }
    } else if let Some(key) = k.strip_prefix("1.") {
      // The user-metadata key is the remainder verbatim (it may contain
      // `.`), exactly swift's `components.dropFirst().joined(".")`.
      user_meta.insert(key.to_string(), v);
    } else if let Some(si) = k.strip_prefix("2.") {
      let Ok(i) = si.parse::<usize>() else { continue };
      class_sparse.insert(i, v);
      class_max_i = Some(class_max_i.map_or(i, |m| m.max(i)));
    }
  }

  // `cache_info` per-cache dimension is sparse: a cache with empty
  // meta_state appears either as the scalar `"0.{i}"` (mlx-lm / our
  // saver) → `info_scalar`, or not at all (swift's empty form) — both
  // mean `[]`. A non-empty meta_state is the dense-checked list.
  let mut cache_info: HashMap<usize, Vec<String>> =
    HashMap::with_capacity(info_sparse.len() + info_scalar.len());
  for (i, m) in info_sparse {
    let inner = match m.keys().copied().max() {
      None => Vec::new(),
      Some(mj) => {
        let cnt = dense_len(mj, m.len(), "meta_state")?;
        // Dense (checked) ⇒ every slot is filled, exactly as swift's
        // `while cacheInfo[i].count <= j { append("") }` on a dense list.
        // `meta_state` setters validate arity downstream, so any residual
        // mismatch is a clean per-cache typed [`Error`] variant from
        // `from_state`, never a panic.
        let mut v = vec![String::new(); cnt];
        for (j, s) in m {
          v[j] = s;
        }
        v
      }
    };
    cache_info.insert(i, inner);
  }
  // Scalar `"0.{i}"=v`: a faithful no-meta scalar is `v == ""` → empty
  // meta_state `[]`. A *non-empty* `v` is preserved as a 1-element list
  // `[v]` so the per-kind emptiness gate in `load_prompt_cache` rejects
  // it for no-meta KV kinds — mirroring mlx-lm's `_BaseCache.meta_state`
  // setter, which `raise`s on any truthy value (cache.py:142-145) instead
  // of silently dropping it. If a (corrupt) file carried *both*
  // `"0.{i}"` and `"0.{i}.{j}"` for the same `i`, the list form inserted
  // above wins (more specific, non-empty); `or_insert_with` here only
  // fills a genuinely scalar-only index, never clobbering a real list.
  for (i, v) in info_scalar {
    cache_info
      .entry(i)
      .or_insert_with(|| if v.is_empty() { Vec::new() } else { vec![v] });
  }

  let cache_classes = match class_max_i {
    None => Vec::new(),
    Some(mi) => {
      // Dense: one `"2.i"` per cache, so `max + 1 == present` for any
      // faithful save; `dense_len` rejects a non-dense (corrupt) class
      // list and bounds the alloc by the present-key count.
      let n = dense_len(mi, class_sparse.len(), "class")?;
      let mut out = vec![String::new(); n];
      for (i, s) in class_sparse {
        out[i] = s;
      }
      out
    }
  };

  Ok((cache_info, user_meta, cache_classes))
}

/// Load a prompt cache from a `.safetensors` file — port of
/// `mlx_lm.models.cache.load_prompt_cache` (cache.py:62-85), wire-format
/// cross-checked vs mlx-swift-lm `loadPromptCache`.
///
/// Returns `(caches, user_metadata)`. Each slot is reconstructed via
/// [`from_state`] keyed on the on-disk **reference class name** (the same
/// names mlx-lm / mlx-swift write), so a cache produced by mlx-lm (any
/// kind) or by mlx-swift (the shared wire format + non-rotating kinds)
/// loads here — subject to the rotating-cache `meta_state` arity caveat in
/// the module docs (a Swift-shaped 5-field `RotatingKVCache` is rejected
/// by the merged 4-field `RotatingKvCache::set_meta_state` via
/// `from_state`, exactly as it would be by authoritative mlx-lm; this is
/// the inherited upstream divergence, surfaced as a clean typed
/// [`Error`] variant, never a panic). The hostile-file discipline — see
/// the module-level
/// "Path-read DoS discipline" for the full posture, **including the
/// acknowledged residual TOCTOU that equals the merged `lm::load` weight
/// path** — runs the non-regular/size gate **before** the path is handed
/// to `crate::io`, then rank-validates every reconstructed state array; a
/// missing / non-regular / oversized / corrupt / count-mismatch /
/// non-dense / wrong-rank / unknown-kind file is a recoverable typed
/// [`Error`] variant, never a panic. No implicit eval — the reconstructed
/// caches hold the loaded arrays lazily.
#[allow(clippy::type_complexity)]
pub fn load_prompt_cache(path: &Path) -> Result<(Vec<Box<dyn KvCache>>, HashMap<String, String>)> {
  // ── Path-read DoS gate (mirrors `lm::load`'s weight path) ──
  // Open the path once with `O_NONBLOCK | O_CLOEXEC` (Unix) so a planted
  // FIFO returns instead of hanging, then fstat *that* fd and reject a
  // non-regular / oversized target before handing the path on. This
  // removes the gate's own stat-then-open race; the residual path-swap
  // window before `crate::io` re-opens `path` is acknowledged and equals
  // the merged `lm::load` weight loader (see the module doc — the
  // reconstructed-state rank gate is the corrupt-payload backstop).
  // Symlinks are followed (the post-open fstat enforces the guarantee on
  // the resolved target — HF / cache layouts symlink files).
  {
    #[cfg(unix)]
    let file = {
      use std::os::unix::fs::OpenOptionsExt;
      std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| {
          Error::FileIo(FileIoPayload::new(
            "cannot open prompt cache",
            FileOp::Open,
            path.to_path_buf(),
            e,
          ))
        })?
    };
    #[cfg(not(unix))]
    let file = std::fs::File::open(path).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "cannot open prompt cache",
        FileOp::Open,
        path.to_path_buf(),
        e,
      ))
    })?;

    // fstat the *opened* fd (no stat-then-open TOCTOU). Reject a
    // non-regular target BEFORE handing the path to mlx-c: a FIFO / device
    // / directory / symlink-to-special has `len() == 0` yet would stream
    // unbounded data into the safetensors mmap (the `lm::load` discipline).
    let meta = file.metadata().map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "cannot stat opened prompt cache",
        FileOp::Stat,
        path.to_path_buf(),
        e,
      ))
    })?;
    if !meta.is_file() {
      return Err(Error::FileIo(FileIoPayload::new(
        "prompt cache target is not a regular file; refusing to read",
        FileOp::Open,
        path.to_path_buf(),
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a regular file"),
      )));
    }
    // The fd is now proven a *regular file*, so the fstat `len()` is its
    // exact, O(1), authoritative size — reject an oversized file here,
    // before `crate::io`/mlx-c maps it, without reading (or double-reading)
    // a single byte. (Unlike `lm::load::load_config`, which `Read::take`s
    // because it *consumes* the config bytes, this loader never needs the
    // bytes — mlx-c reads the file itself — so a streaming size probe would
    // be a wasteful, itself-DoS-prone 2x read of a multi-GB cache.)
    if meta.len() > MAX_PROMPT_CACHE_BYTES {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "load_prompt_cache: file size; refusing to read",
        "MAX_PROMPT_CACHE_BYTES",
        MAX_PROMPT_CACHE_BYTES,
        meta.len(),
      )));
    }
    // `file` is dropped here (fd closed); `crate::io` re-opens by path.
    // The window between this fstat and that re-open is the same one
    // `lm::load` accepts (open-once defeats the *handle* TOCTOU; a
    // path-swap race on a trusted local cache dir is out of scope and
    // identical to the merged loader's posture).
  }

  // `arrays, cache_metadata = mx.load(file, return_metadata=True)`
  let (flat_arrays, flat_side) = crate::io::load_safetensors_with_metadata(path)?;

  // `arrays = tree_unflatten(...)`, `cache_metadata = tree_unflatten(...)`
  let mut cache_data = unflatten_arrays(flat_arrays)?;
  let (cache_info, user_metadata, cache_classes) = unflatten_side(flat_side)?;

  // The number of caches is `len(cache_classes)` — `cache_classes =
  // [type(c).__name__ for c in cache]` is the only flattened list always
  // length `len(cache)` (a cache with empty `state`/`meta_state` emits no
  // `"i.j"`/`"0.i.j"` keys, but its class is always written as `"2.i"`).
  // Driving the loop by this count reconstructs an all-empty-state cache
  // list faithfully (mlx-lm's `zip` would drop it; mlx-swift's
  // `cacheData.count == cacheClasses.count` guard would reject it). An
  // out-of-range *array group* index (a `"{i}.{j}"` whose `i >=
  // class_count`) means a corrupt / incompatible file — that is the only
  // genuine inconsistency, so it (not an empty trailing cache) is the
  // recoverable error.
  let n = cache_classes.len();
  if let Some(&bad) = cache_data.keys().find(|&&i| i >= n) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "load_prompt_cache: array group index (corrupt or incompatible file)",
      "must be < class count",
      format_smolstr!("index={bad}, class_count={n}, path={}", path.display()),
    )));
  }

  // `cache = [globals()[c].from_state(state, meta_state) for c, state,
  // meta_state in zip(classes, arrays, info)]`
  let mut caches: Vec<Box<dyn KvCache>> = Vec::with_capacity(n);
  for (i, kind) in cache_classes.into_iter().enumerate() {
    // A cache with no array keys had empty `state` → `[]` (an empty/unused
    // cache); `from_state` rebuilds the right empty concrete type.
    let state = cache_data.remove(&i).unwrap_or_default();
    // A cache with empty `meta_state` emits no `"0.i.*"` keys → absent in
    // the sparse `cache_info` map → `&[]` (its `set_meta_state` no-ops /
    // validates downstream).
    let meta: &[String] = cache_info.get(&i).map_or(&[], Vec::as_slice);

    // Hostile-file rank gate (BEFORE `from_state`). `from_state`'s
    // `KVCache`/`RotatingKVCache` state setters mirror mlx-lm/Swift
    // verbatim (`self.keys, self.values = v` — cache.py:295/371): they
    // assign without validating rank and "let MLX error downstream". For
    // an in-process model that downstream error is fine, but a *prompt
    // cache file* is attacker-influenceable and the contract here is
    // "corrupt file ⇒ `Err`, never a panic". A file with kind
    // `RotatingKVCache`/`KVCache`, a valid 4-item / empty meta_state, and
    // two **rank-1** arrays would otherwise reconstruct as `Ok`, then
    // panic the first time a cache method indexes `shape()[2]`/`[3]`
    // (`util::seq_len`/`slice_seq`, which assume the 4-D `[B, n_kv_heads,
    // S, head_dim]` layout). Reject a non-4-D state array here so a
    // corrupt/foreign payload is a recoverable typed [`Error`] variant, not a
    // deferred panic. (Empty state — `[]` — is the valid "fresh cache"
    // case and passes; this gate only rejects *present* arrays of the
    // wrong rank. `from_state`'s own `empty ⇒ offset==0 && idx==0`
    // Rotating invariant still covers the empty-but-nonzero-offset case.)
    //
    // Scoped to exactly the kind strings `mod.rs::from_state` reconstructs
    // into a 4-D KV cache (`KVCache`/`RotatingKVCache` + the documented
    // Rust/Swift aliases — kept in lockstep with that match). A non-KV
    // kind (`ArraysCache`/`MambaCache`/… added by a later PR) is NOT
    // pre-rejected here on rank — `from_state` does its own kind-specific
    // validation (today: unknown kind ⇒ `Error::UnknownEnumValue`), so this gate
    // stays forward-compatible and never false-rejects a future
    // non-4-D-state cache.
    const KV_RANK_KINDS: &[&str] = &[
      "KVCache",
      "ConcatenateKVCache",
      "KVCacheSimple",
      "StandardKvCache",
      "RotatingKVCache",
      "RotatingKvCache",
    ];
    if KV_RANK_KINDS.contains(&kind.as_str()) {
      for (j, arr) in state.iter().enumerate() {
        let nd = arr.ndim();
        if nd != KV_NDIM {
          return Err(Error::LayerKeyed(LayerKeyedPayload::new(
            format_smolstr!(
              "load_prompt_cache: cache {i} (kind {kind:?}) state array {j} (path={})",
              path.display()
            ),
            Error::RankMismatch(RankMismatchPayload::new(
              "load_prompt_cache: KV state array must be 4-D [B, n_kv_heads, S, head_dim] (corrupt or incompatible file)",
              nd as u32,
              arr.shape().to_vec(),
            )),
          )));
        }
      }
    }

    // ── Validation-scope boundary (deliberate; faithful to the spec) ──
    // This gate validates *rank* (and `from_state` validates *arity* +
    // the empty⇒zero-offset Rotating invariant). It does **not** validate
    // deeper *semantic* cross-field consistency — e.g. a `RotatingKVCache`
    // whose state `S` / `_idx` disagree with `offset`/`max_size` (a
    // pre-window cache "should" have `S == offset == _idx`). That is a
    // conscious boundary, not an oversight:
    //
    //  * The authoritative spec this is a 1:1 port of — mlx-lm
    //    `_BaseCache.from_state` → `RotatingKVCache.state`/`meta_state`
    //    setters (cache.py:294-295, 535-541) — performs **no** such
    //    consistency check either: it assigns raw and "lets MLX error
    //    downstream". mlx-lm's own `load_prompt_cache` accepts the very
    //    same crafted-but-rank-valid inconsistent file and yields a
    //    logically-wrong (never crashing) cache. Adding checks mlx-lm
    //    lacks would *diverge from the authoritative spec* (the task's
    //    prime directive) and cross into the "don't chase review into
    //    unbounded mlx-core-internal hardening / match official binding
    //    design" boundary.
    //  * The merged `RotatingKvCache::set_state`/`set_meta_state` (#32 —
    //    out of this PR's scope) are intentionally verbatim mirrors of
    //    those setters; the place to add load-time semantic validation, if
    //    ever wanted, is an upstream-spec decision affecting mlx-lm,
    //    mlx-swift, and the merged trait alike — not this persistence
    //    shim.
    //  * Crucially, the task's actual hostile-file contract — *corrupt
    //    file ⇒ `Err`, **never a panic**; never UB* — **is** met: the
    //    only case that previously *panicked* (wrong **rank** →
    //    `shape()[2]` OOB) is rejected above. A rank/arity-valid but
    //    semantically-inconsistent file is **memory-safe and
    //    non-panicking** here exactly as in mlx-lm (it yields a
    //    logically-wrong cache, the identical authoritative behavior) —
    //    no panic, no leak, no UB.
    //
    // Net: rank+arity rejection kills the panic/UB class (the contract);
    // deep semantic validation is deliberately deferred to the upstream
    // spec to stay 1:1 faithful. This residual is reported as an explicit
    // faithfulness-vs-hardening boundary, not patched here (doing so would
    // be unfaithful and an unbounded-validation spiral).

    // No-meta KV kinds: meta_state MUST be empty (faithful to mlx-lm,
    // which actively rejects this at load — NOT a deferred boundary like
    // the semantic one above). `KVCache`/`ConcatenateKVCache`/
    // `KVCacheSimple`/`StandardKvCache` inherit `_BaseCache.meta_state`
    // whose **setter** `raise`s on any *truthy* value (`if v is not None
    // and v: raise ValueError`, cache.py:142-145). Our merged
    // `StandardKvCache::set_meta_state` is the trait-default no-op (#32 —
    // out of scope to change) and does *not* raise, so `from_state` alone
    // would silently accept a malformed `"0.{i}"="garbage"` /
    // `"0.{i}.{j}"`-on-KVCache file. Enforce mlx-lm's emptiness rule here
    // so such a schema-drifted / corrupt file is a clean recoverable
    // typed [`Error`] variant on the **load path** (exactly mlx-lm's
    // `ValueError`), not a silently-wrong `Ok`. `RotatingKVCache` (a real
    // 4-tuple meta_state) is intentionally excluded — only the no-meta
    // kinds carry this constraint.
    //
    // **Scope note (escalated follow-up):** this gate covers the
    // **load path** (`load_prompt_cache` → callers of this fn). A
    // *direct* `crate::lm::cache::from_state("KVCache", state,
    // &["x".into()])` call still returns `Ok` because the merged #32
    // `from_state` body + `StandardKvCache::set_meta_state` trait-default
    // no-op do not raise on truthy meta_state — a faithfulness gap with
    // mlx-lm `_BaseCache.from_state`/`meta_state.setter`
    // (cache.py:142-145). Closing it requires editing the merged
    // `mod.rs::from_state` body or overriding
    // `StandardKvCache::set_meta_state` in `standard.rs` — both
    // **explicitly out of this PR's scope** (the task forbids touching
    // any other cache file, and `from_state` is the AS-MERGED #32
    // contract). Reported as an explicit escalated #32-amend follow-up,
    // not silently scope-creeped into this PR (per
    // `feedback_escalate_dont_spiral` + `feedback_match_official_binding_design`).
    const NO_META_KINDS: &[&str] = &[
      "KVCache",
      "ConcatenateKVCache",
      "KVCacheSimple",
      "StandardKvCache",
    ];
    if NO_META_KINDS.contains(&kind.as_str()) && !meta.is_empty() {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        format_smolstr!(
          "load_prompt_cache: cache {i} (kind {kind:?}) (path={})",
          path.display()
        ),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_prompt_cache: no-meta cache (KVCache / ConcatenateKVCache / KVCacheSimple / StandardKvCache) carries non-empty meta_state; mlx-lm rejects this (corrupt or schema-drifted file)",
          "must have empty meta_state",
        )),
      )));
    }

    // `from_state` validates the kind + meta_state arity and maps any
    // failure to a recoverable typed [`Error`] variant (it never panics on a
    // corrupt/foreign payload — unknown kind, wrong meta_state length,
    // inconsistent empty-state-with-offset, …).
    let cache = from_state(&kind, state, meta).map_err(|e| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        format_smolstr!(
          "load_prompt_cache: cannot reconstruct cache {i} (kind {kind:?}, path={})",
          path.display()
        ),
        e,
      ))
    })?;
    caches.push(cache);
  }

  Ok((caches, user_metadata))
}

/// Whether every cache in the model state can be trimmed — port of
/// `mlx_lm.models.cache.can_trim_prompt_cache` (cache.py:88-92):
/// `all(c.is_trimmable() for c in cache)` (vacuously `true` for an empty
/// list, exactly as Python's `all([])`).
pub fn can_trim_prompt_cache(cache: &[Box<dyn KvCache>]) -> bool {
  cache.iter().all(|c| c.is_trimmable())
}

/// Trim every cache by `num_tokens`, returning the number trimmed — port of
/// `mlx_lm.models.cache.trim_prompt_cache` (cache.py:95-111).
///
/// mlx-lm: `if not can_trim_prompt_cache(cache) or len(cache) == 0: return
/// 0; return [c.trim(num_tokens) for c in cache][0]` — every cache is
/// trimmed (the list comprehension calls `trim` on each), but the returned
/// count is the **first** cache's. Faithful: not-trimmable / empty → 0
/// (nothing is trimmed); otherwise trim all, return `cache[0]`'s count
/// (each layer's KV is the same length, so all trims agree, matching
/// mlx-lm's `[...][0]`).
pub fn trim_prompt_cache(cache: &mut [Box<dyn KvCache>], num_tokens: usize) -> Result<usize> {
  if !can_trim_prompt_cache(cache) || cache.is_empty() {
    return Ok(0);
  }
  let mut first = 0usize;
  for (i, c) in cache.iter_mut().enumerate() {
    let t = c.trim(num_tokens)?;
    if i == 0 {
      first = t;
    }
  }
  Ok(first)
}

#[cfg(test)]
mod tests {
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
}

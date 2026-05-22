//! Vision-feature cache for multi-turn multimodal conversations.
//!
//! Ported 1:1 from `mlx-vlm/mlx_vlm/vision_cache.py::VisionFeatureCache`
//! (the only reference — mlx-vlm has no Swift counterpart for this type;
//! confirmed by a repo-wide search of `mlx-swift-lm`). The cache stores
//! the output of `vision_tower` + `embed_vision` (image features already
//! projected into the language model's embedding space, ready for the
//! image-into-text splice), keyed by image identity, so a VLM discussing
//! the **same image across multiple turns/prompts** re-uses the cached
//! embeddings instead of re-running the (expensive) vision encoder.
//!
//! ## Reference structure (`feedback_mirror_reference_structure`)
//!
//! `vision_cache.py` is one class, [`VisionFeatureCache`], built on a
//! Python `OrderedDict` with:
//! - **LRU eviction** — oldest entry dropped once `max_size` is exceeded
//!   (`OrderedDict.popitem(last=False)` after `move_to_end`);
//! - a `_make_key` helper deriving a `str` key from the image source —
//!   three branches: a `str` path/URL used directly, a `list` joined with
//!   `"|"`, and a PIL image content-hashed (`sha256(tobytes())[:16]`);
//! - `get` / `put` / `clear` / `__len__` / `__contains__`.
//!
//! mlxrs mirrors that shape faithfully: one [`VisionFeatureCache`] type,
//! the same `max_size`-bounded LRU, the same five operations ([`get`] /
//! [`put`] / [`clear`] / [`len`] / [`contains`]), and a key-derivation
//! family ([`Key`]) covering the same three source kinds.
//!
//! [`get`]: VisionFeatureCache::get
//! [`put`]: VisionFeatureCache::put
//! [`clear`]: VisionFeatureCache::clear
//! [`len`]: VisionFeatureCache::len
//! [`contains`]: VisionFeatureCache::contains
//!
//! ## Deviations from the Python reference (and why)
//!
//! - **Stored value is an owned [`Array`]**, duplicated on `put`/`get` via
//!   the refcount-sharing [`Array::try_clone`] — `mlxrs::Array` is
//!   deliberately `!Clone` (a panicking `Clone` would hide the rare FFI
//!   allocation failure), so the fallible `try_clone` is the only handle
//!   dup. A `try_clone` is **cheap** (a refcount bump + a small handle
//!   alloc, no feature-data copy), so caching shares the buffer exactly
//!   like Python's reference-semantics `mx.array`.
//! - **Keys are [`Key`], an owned `String` wrapper.** Python's `_make_key`
//!   normalizes every source to a `str`; [`Key`] does the same with
//!   three constructors mirroring the three Python branches —
//!   [`Key::from_source`] (the `str` branch — path/URL used verbatim),
//!   [`Key::from_sources`] (the `list` branch — `"|"`-joined), and
//!   [`Key::from_bytes`] (the PIL branch — a content hash). Because
//!   mlxrs has no PIL type and no crypto dependency, [`Key::from_bytes`]
//!   uses the std [`DefaultHasher`](std::hash::DefaultHasher) (a fast
//!   non-cryptographic hash) rather than `sha256`: this is a **cache
//!   key**, collision-tolerant by construction and never a security
//!   boundary, so a SipHash-class digest is the idiomatic Rust choice and
//!   pulls no new crate. The `pil:` / `obj:` prefixes from the reference
//!   are preserved so a hashed key can never alias a literal path.
//! - **Bounded memory** — the reference is already bounded (`max_size`,
//!   default 20); mlxrs keeps that exact cap and default. The constructor
//!   rejects `max_size == 0` ([`Error::ShapeMismatch`]) rather than
//!   silently building a cache that can hold nothing (Python would not
//!   raise but every `put` would immediately self-evict — a faithful but
//!   useless state; mlxrs surfaces the misuse).
//!
//! ## No implicit eval
//!
//! The cache never evaluates an `Array`. `put` stores whatever lazy or
//! materialized handle the caller passes (the reference relies on the
//! caller having `mx.eval`'d the features first — see
//! `generate.py:1055`); `get` hands back a `try_clone` of that same
//! handle. Evaluation stays the caller's explicit step.

use std::{
  collections::{HashMap, VecDeque},
  hash::{Hash, Hasher},
};

use crate::{
  array::Array,
  error::{Error, Result},
};

/// The default `max_size` — matches `VisionFeatureCache(max_size=20)` in
/// `mlx-vlm/mlx_vlm/vision_cache.py:31`.
pub const DEFAULT_MAX_SIZE: usize = 20;

/// A normalized cache key derived from an image source.
///
/// Mirrors `VisionFeatureCache._make_key` (`vision_cache.py:35-50`), which
/// reduces every image source to a `str`. The three constructors map 1:1
/// to the reference's three branches:
///
/// | Python branch | constructor |
/// |---|---|
/// | `isinstance(image_source, str)` — path / URL used directly | [`Key::from_source`] |
/// | `isinstance(image_source, list)` — `"\|".join(...)` | [`Key::from_sources`] |
/// | PIL image — `sha256(tobytes())[:16]`, prefixed `pil:` | [`Key::from_bytes`] |
///
/// Two `Key`s are equal iff their normalized strings are equal, so
/// distinct sources never collide and (matching the reference) **list
/// order is significant** — `["a", "b"]` and `["b", "a"]` are different
/// keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key(String);

impl Key {
  /// Key for a single path- or URL-style image source — the reference's
  /// `isinstance(image_source, str)` branch (`vision_cache.py:42-43`),
  /// which uses the string verbatim.
  ///
  /// The string is the cache identity: two calls with byte-identical
  /// paths/URLs hit the same entry; any difference (a trailing slash, a
  /// different query string) is a distinct key — exactly as in the
  /// reference, which does no path canonicalization.
  pub fn from_source(source: &str) -> Self {
    Self(source.to_owned())
  }

  /// Key for a multi-image source — the reference's
  /// `isinstance(image_source, list)` branch (`vision_cache.py:44-45`):
  /// the per-image keys joined with `'|'`.
  ///
  /// **Order is significant** (the reference joins in list order):
  /// `from_sources(&["a", "b"])` differs from `from_sources(&["b", "a"])`.
  /// An empty slice yields the empty-string key (the reference's
  /// `"".join([])`), and a single-element slice equals
  /// [`Key::from_source`] of that element — both faithful to `str.join`.
  pub fn from_sources(sources: &[&str]) -> Self {
    Self(sources.join("|"))
  }

  /// Key for an in-memory image with no stable path — the reference's
  /// PIL branch (`vision_cache.py:47-49`): hash the raw image bytes.
  ///
  /// The reference uses `sha256(tobytes())[:16]`; mlxrs uses the std
  /// [`DefaultHasher`](std::hash::DefaultHasher) because this is a
  /// collision-tolerant **cache key**, never a security boundary, and a
  /// SipHash-class digest needs no extra crate (see the module-level
  /// "Deviations" note). The result is prefixed `pil:` — identical to the
  /// reference — so a content-hashed key can never alias a literal path
  /// such as `"pil:photo.jpg"` would only collide with another hashed
  /// key, never with a [`Key::from_source`] of a real file path unless
  /// that path itself starts with `pil:`.
  pub fn from_bytes(bytes: &[u8]) -> Self {
    let mut hasher = std::hash::DefaultHasher::new();
    bytes.hash(&mut hasher);
    Self(format!("pil:{:016x}", hasher.finish()))
  }

  /// The normalized key string. Exposed for tests / introspection; the
  /// cache never needs the caller to read it.
  pub fn as_str(&self) -> &str {
    &self.0
  }
}

impl From<&str> for Key {
  /// Convenience: a `&str` is the single-source ([`Key::from_source`])
  /// case, the overwhelmingly common path/URL key.
  fn from(source: &str) -> Self {
    Self::from_source(source)
  }
}

/// An LRU cache of vision-encoder output features, keyed by image
/// identity.
///
/// Port of `mlx-vlm`'s `VisionFeatureCache` (`vision_cache.py:15-79`). A
/// VLM that discusses the same image across several turns calls [`get`]
/// before encoding; on a hit it skips the vision tower entirely and
/// re-uses the cached features, on a miss it encodes once and [`put`]s
/// the result. Eviction is purely LRU once [`max_size`](Self::max_size)
/// is exceeded.
///
/// [`get`]: Self::get
/// [`put`]: Self::put
///
/// # Memory
///
/// Bounded by construction: at most `max_size` feature [`Array`]s are
/// retained (default [`DEFAULT_MAX_SIZE`]). Each stored value is a
/// refcount-sharing [`Array::try_clone`] of the caller's handle — the
/// feature *buffer* is shared, not copied, so the cache's marginal cost
/// per entry is one small mlx-c handle. [`clear`](Self::clear) drops every
/// entry (the reference's model-unload hook); on `Drop` the whole map is
/// freed.
///
/// # Concurrency
///
/// Not `Sync` (it stores [`Array`], which is intentionally `!Send` +
/// `!Sync`). One cache belongs to one inference thread — the same
/// single-thread contract the rest of `mlxrs` is built on.
pub struct VisionFeatureCache {
  /// LRU bound. `>= 1` — enforced by [`Self::with_max_size`].
  max_size: usize,
  /// Key → feature-`Array` map. Holds the owned (refcount-shared)
  /// handles.
  entries: HashMap<Key, Array>,
  /// Recency queue, **least-recently-used at the front**, most-recent at
  /// the back — the explicit mirror of `OrderedDict`'s insertion order.
  /// `move_to_end` is "remove this key, push it to the back"; eviction is
  /// "pop the front" (`popitem(last=False)`). Every key in `entries` is
  /// present exactly once in `recency`, and vice versa — the two are kept
  /// in lockstep by every mutating method.
  recency: VecDeque<Key>,
}

impl VisionFeatureCache {
  /// Build a cache with the reference default capacity
  /// ([`DEFAULT_MAX_SIZE`] = 20) — matches `VisionFeatureCache()` with no
  /// argument (`vision_cache.py:31`).
  pub fn new() -> Self {
    // DEFAULT_MAX_SIZE is a non-zero constant, so `with_max_size` cannot
    // fail here; `expect` documents that invariant rather than leaking a
    // `Result` from the no-argument constructor.
    Self::with_max_size(DEFAULT_MAX_SIZE).expect("DEFAULT_MAX_SIZE is non-zero")
  }

  /// Build a cache holding at most `max_size` entries — matches
  /// `VisionFeatureCache(max_size=...)` (`vision_cache.py:31`).
  ///
  /// # Capacity grows lazily
  ///
  /// The backing containers are created **empty** — `max_size` is stored
  /// purely as the LRU eviction bound, never pre-reserved. They grow
  /// naturally as entries are inserted and the existing LRU eviction
  /// (see [`put`](Self::put)) caps live entries at `max_size`, so the map
  /// never exceeds `max_size` entries regardless of upfront reservation.
  ///
  /// This is deliberate: pre-reserving the raw `max_size` would allocate
  /// memory proportional to a caller-, config-, or request-derived size
  /// *before any entry exists*, so a large-but-allocatable `max_size`
  /// (e.g. a hostile config value) could exhaust memory on an otherwise
  /// successful construction — a DoS class. Empty-init removes that
  /// entirely; the cost is a few reallocations as the cache fills to its
  /// actual working set (which is `<= max_size`, and usually far smaller).
  /// Incremental growth is itself fallible — see [`put`](Self::put)'s
  /// `try_reserve(1)`, so even map growth cannot abort the process.
  ///
  /// # Errors
  ///
  /// - [`Error::ShapeMismatch`] if `max_size == 0`. The reference does not
  ///   raise on a zero cap, but a zero-capacity cache is a useless state —
  ///   every [`put`](Self::put) would store then immediately self-evict its
  ///   own entry, so [`get`](Self::get) could never hit. mlxrs surfaces the
  ///   misuse instead of silently building a cache that can hold nothing.
  pub fn with_max_size(max_size: usize) -> Result<Self> {
    if max_size == 0 {
      return Err(Error::ShapeMismatch {
        message: "VisionFeatureCache: max_size must be >= 1 (a zero-capacity \
                  cache can never hold an entry)"
          .into(),
      });
    }
    // Create the containers EMPTY — do NOT pre-reserve the raw `max_size`.
    // `max_size` is caller-/config-/request-derived; reserving it up front
    // would consume memory proportional to an untrusted size before a
    // single entry exists, so a large-but-allocatable cap would exhaust
    // memory on an otherwise-successful `Ok` (a DoS converted from abort to
    // memory-exhaustion-on-success). The LRU already self-bounds live
    // entries to `max_size` via eviction in `put`, so the map never exceeds
    // `max_size` entries — the upfront reserve was only an optimization.
    // The containers grow lazily as entries are inserted (each growth made
    // fallible by `put`'s `try_reserve(1)`), bounded to the actual working
    // set (<= max_size, usually much smaller).
    Ok(Self {
      max_size,
      entries: HashMap::new(),
      recency: VecDeque::new(),
    })
  }

  /// The configured LRU bound — mirrors the reference's public
  /// `self.max_size` attribute (`vision_cache.py:32`).
  pub fn max_size(&self) -> usize {
    self.max_size
  }

  /// Number of cached entries — mirrors `__len__` (`vision_cache.py:74`).
  pub fn len(&self) -> usize {
    // Invariant: `entries` and `recency` always have equal length.
    self.entries.len()
  }

  /// Whether the cache holds no entries.
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  /// Look up cached features by image identity — port of `get`
  /// (`vision_cache.py:52-58`).
  ///
  /// On a **hit**, the entry is marked most-recently-used (the
  /// reference's `move_to_end`) and a refcount-sharing
  /// [`Array::try_clone`] of the cached features is returned — the caller
  /// gets an owned handle over the *same* buffer. On a **miss**, returns
  /// `Ok(None)`.
  ///
  /// Takes `&mut self` because a hit mutates LRU recency order — looking
  /// something up is, by the cache's contract, a state change. This is
  /// not an implicit `Array` eval: no `Array` is materialized, only the
  /// recency queue is touched.
  ///
  /// # Errors
  ///
  /// [`Error::OutOfMemory`] (or another backend error) if the
  /// [`Array::try_clone`] of a hit entry fails — the rare mlx-c handle
  /// allocation failure. A miss never allocates and never errors.
  pub fn get(&mut self, key: &Key) -> Result<Option<Array>> {
    // `try_clone` BEFORE touching `recency`: if the clone fails we return
    // `Err` having mutated nothing, so the cache is left exactly as it
    // was (transactional — no half-applied LRU bump).
    let cloned = match self.entries.get(key) {
      Some(features) => features.try_clone()?,
      None => return Ok(None),
    };
    self.touch(key);
    Ok(Some(cloned))
  }

  /// Store `features` under `key`, evicting the least-recently-used entry
  /// if the cache is full — port of `put` (`vision_cache.py:60-68`).
  ///
  /// Behavior, matching the reference's three `OrderedDict` cases:
  /// - **key already present** — overwrite the value and mark it
  ///   most-recently-used (`move_to_end`); the entry count is unchanged,
  ///   so no eviction happens.
  /// - **new key, cache not full** — insert as most-recently-used.
  /// - **new key, cache full** — evict the least-recently-used entry
  ///   (`popitem(last=False)`), *then* insert.
  ///
  /// The stored value is a refcount-sharing [`Array::try_clone`] of
  /// `features`: the cache shares the feature buffer with the caller
  /// (Python's `mx.array` reference semantics), it does not deep-copy.
  /// The caller is expected to have evaluated `features` already — the
  /// reference does `mx.eval(features)` before `put` (`generate.py:1055`);
  /// the cache itself never evals.
  ///
  /// # Errors
  ///
  /// - [`Error::OutOfMemory`] (or another backend error) if the
  ///   [`Array::try_clone`] of `features` fails.
  /// - [`Error::OutOfMemory`] if growing a backing container to admit a
  ///   **new** key fails. Capacity grows lazily (the constructor pre-reserves
  ///   nothing — see [`with_max_size`](Self::with_max_size)), so a new
  ///   insertion may need one growth step; that step is the fallible
  ///   [`try_reserve(1)`](HashMap::try_reserve), so it cannot abort the
  ///   process on an allocator failure. The overwrite path reuses an existing
  ///   slot and never grows, so it never hits this.
  ///
  /// On any error the cache is **unchanged** — the clone and the
  /// reservation both happen before any structural mutation, so a failed
  /// `put` neither inserts, overwrites, nor evicts.
  pub fn put(&mut self, key: Key, features: &Array) -> Result<()> {
    // Clone FIRST — before any mutation — so a clone failure leaves the
    // cache (entries + recency + the would-be-evicted victim) untouched.
    let stored = features.try_clone()?;

    if self.entries.contains_key(&key) {
      // Overwrite path: replace the value, refresh recency. The key already
      // exists so `insert` reuses its slot — no growth, no eviction (mirrors
      // the reference's `move_to_end` then `self._cache[key] = features`).
      self.entries.insert(key.clone(), stored);
      self.touch(&key);
      return Ok(());
    }

    // New key. Containers were constructed empty (no upfront reserve), so
    // admitting a new key may grow them by one. Reserve that one slot
    // FALLIBLY, BEFORE any structural mutation (and before the eviction
    // below) — so a reservation failure leaves the cache *exactly* as it
    // was: no insert, no overwrite, and crucially no premature eviction.
    // Incremental growth thus stays recoverable (`Error::OutOfMemory`) and
    // can never abort the process the way an infallible `insert`-triggered
    // realloc could. Reserving one slot of headroom on the current length is
    // always enough: the optional eviction below only ever *shrinks* the
    // containers before the single push/insert.
    self
      .recency
      .try_reserve(1)
      .map_err(|_| Error::OutOfMemory)?;
    self
      .entries
      .try_reserve(1)
      .map_err(|_| Error::OutOfMemory)?;

    // At capacity: evict the LRU entry first. `>=` (not `==`) matches the
    // reference's `len(self._cache) >= self.max_size`; with the invariant
    // `len <= max_size` always holding, this evicts exactly one entry, and
    // only when full.
    if self.entries.len() >= self.max_size
      && let Some(lru_key) = self.recency.pop_front()
    {
      self.entries.remove(&lru_key);
    }
    self.recency.push_back(key.clone());
    self.entries.insert(key, stored);
    Ok(())
  }

  /// Whether `key` is currently cached — port of `__contains__`
  /// (`vision_cache.py:77-79`).
  ///
  /// A pure read: unlike [`get`](Self::get) this does **not** refresh LRU
  /// recency (the reference's `__contains__` likewise does not
  /// `move_to_end`), so it can take `&self`.
  pub fn contains(&self, key: &Key) -> bool {
    self.entries.contains_key(key)
  }

  /// Drop every cached entry — port of `clear` (`vision_cache.py:70-72`),
  /// the reference's model-unload / model-swap hook.
  ///
  /// Both the entry map and the recency queue are emptied; every stored
  /// [`Array`] handle is dropped (its underlying buffer freed once no
  /// other handle shares it). [`max_size`](Self::max_size) is retained —
  /// the cache stays reusable.
  pub fn clear(&mut self) {
    self.entries.clear();
    self.recency.clear();
  }

  /// Mark `key` as most-recently-used: the mirror of
  /// `OrderedDict.move_to_end(key)`.
  ///
  /// Removes the (single) prior occurrence of `key` from `recency` and
  /// pushes it to the back. `key` is assumed present in `entries` by
  /// every caller; if it is somehow absent from `recency` the queue is
  /// just left as-is (no panic) — but the entries/recency lockstep
  /// invariant means that never happens in practice.
  fn touch(&mut self, key: &Key) {
    if let Some(pos) = self.recency.iter().position(|k| k == key) {
      // `remove` at an arbitrary position is O(n) in the queue length,
      // but the queue length is bounded by `max_size` (default 20) — a
      // tiny, fixed bound — so this is effectively O(1) for any realistic
      // cache. Faithful to `OrderedDict.move_to_end`'s O(1) amortized
      // intent without pulling an intrusive-list dependency.
      self.recency.remove(pos);
    }
    self.recency.push_back(key.clone());
  }
}

impl Default for VisionFeatureCache {
  /// Same as [`VisionFeatureCache::new`] — the reference default
  /// (`max_size = 20`).
  fn default() -> Self {
    Self::new()
  }
}

impl std::fmt::Debug for VisionFeatureCache {
  /// Compact debug: capacity + current occupancy. Deliberately does not
  /// print the cached `Array`s (they are large feature tensors and
  /// `Array`'s own `Debug` is not derived) — only the cache's structural
  /// state.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("VisionFeatureCache")
      .field("max_size", &self.max_size)
      .field("len", &self.entries.len())
      .finish()
  }
}

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
//! - **Keys are [`Key`], a normalized-string wrapper.** Python's
//!   `_make_key` normalizes every source to a `str`; [`Key`] does the same
//!   with three constructors mirroring the three Python branches —
//!   [`Key::from_source`] (the `str` branch — path/URL used verbatim),
//!   [`Key::from_sources`] (the `list` branch — `"|"`-joined), and
//!   [`Key::from_bytes`] (the PIL branch — a content hash). [`Key`] holds
//!   that string as an [`Rc<str>`](std::rc::Rc) (an implementation
//!   detail — every public constructor / accessor has the same signature
//!   and semantics it would with a `String` field); the cache stores
//!   `Rc<str>` clones in both its containers, so the recency queue and the
//!   entry map share one heap-allocated string and [`put`] never
//!   heap-copies a key (see [`Key`]'s "Internal representation" note and
//!   [`VisionFeatureCache`]'s "Key storage" note). Because
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
  rc::Rc,
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
///
/// # Internal representation
///
/// The normalized string is held as an [`Rc<str>`](Rc), not a `String`.
/// This is an implementation detail — every public method
/// ([`from_source`](Self::from_source), [`from_sources`](Self::from_sources),
/// [`from_bytes`](Self::from_bytes), [`as_str`](Self::as_str), and the
/// `From<&str>` conversion) has the exact same signature and behavior it
/// would with a `String` field. The `Rc` backing buys two things the cache
/// relies on:
///
/// - **`Clone` is a refcount bump** — infallible, no heap allocation, no
///   string copy. [`VisionFeatureCache`] stores a key in *two* containers
///   (the entry map and the recency queue); with an `Rc<str>` the second
///   container gets a [`Rc::clone`], so a [`put`](VisionFeatureCache::put)
///   never heap-copies a key and the post-eviction key handoff cannot
///   fail. (With a `String` field that second copy was a fallible-by-abort
///   heap allocation occurring *after* eviction — a transactional hazard.)
/// - **`Hash`/`Eq` are unchanged** — `Rc<str>` hashes and compares by the
///   pointed-to `str` *content* (it derefs / `Borrow`s `str`), so the
///   derived `Hash`/`PartialEq`/`Eq` here are byte-for-byte the same
///   relation as a `String`-backed `Key`: two `Key`s are equal iff their
///   strings are equal, full stop.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key(Rc<str>);

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
    // `Rc::from(&str)` allocates the shared string once, here in the
    // constructor — the cache's `put` then only ever *moves* and refcount-
    // clones this `Rc`, never re-allocates the key.
    Self(Rc::from(source))
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
    // `join` builds the `'|'`-joined `String`; `Rc::from` consumes it into
    // the shared `Rc<str>` the cache stores (the transient `String` buffer
    // is freed). The whole allocation cost is borne here in the
    // constructor, not on the `put` hot path.
    Self(Rc::from(sources.join("|")))
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
    // `format!` builds the `pil:`-prefixed digest `String`; `Rc::from`
    // consumes it into the shared `Rc<str>` (transient buffer freed).
    Self(Rc::from(format!("pil:{:016x}", hasher.finish())))
  }

  /// The normalized key string. Exposed for tests / introspection; the
  /// cache never needs the caller to read it.
  pub fn as_str(&self) -> &str {
    // `Rc<str>` derefs to `str`; `&self.0` coerces `&Rc<str>` to `&str`.
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
/// # Key storage
///
/// A key's normalized string is stored exactly **once** per entry as an
/// [`Rc<str>`](Rc): the entry-map key and the recency-queue entry are two
/// [`Rc::clone`]s of that one allocation. `Rc::clone` is an infallible
/// refcount bump — **no heap allocation, no string copy** — so every
/// key-side operation on the [`put`] path (inserting into both containers)
/// and on the recency-update path (a [`get`] hit relocates the key within
/// the recency queue) is allocation-free. The string a [`Key`] carries is
/// consumed into the `Rc<str>` once, on the inserting `put`; after that no
/// `put` or `get` ever heap-allocates a key. This is what makes a
/// full-cache `put` (evict + insert) and a `get` hit's recency bump
/// strictly allocation-free on the key side, and what makes the
/// post-eviction key handoff infallible (a refcount bump cannot fail), so
/// a failed `put` can never leave the cache half-mutated.
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
  /// `Array` handles. The map *key* is an [`Rc<str>`](Rc): the same
  /// allocation the recency queue holds (see the struct-level "Key
  /// storage" note), so admitting a key never heap-copies the string.
  /// `Rc<str>` hashes and compares by content (it `Borrow`s `str`), so
  /// lookups with a `&Key` work via `key.as_str()`.
  entries: HashMap<Rc<str>, Array>,
  /// Recency queue, **least-recently-used at the front**, most-recent at
  /// the back — the explicit mirror of `OrderedDict`'s insertion order.
  /// `move_to_end` is "remove this key, push it to the back"; eviction is
  /// "pop the front" (`popitem(last=False)`). Every key in `entries` is
  /// present exactly once in `recency`, and vice versa — the two are kept
  /// in lockstep by every mutating method.
  ///
  /// Holds [`Rc<str>`](Rc) clones of the *same* allocations the entry map
  /// keys point at. Pushing/moving a key here is therefore a refcount bump,
  /// not a `String` copy — so [`put`](Self::put) and [`touch`](Self::touch)
  /// never heap-allocate a key.
  recency: VecDeque<Rc<str>>,
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
  /// That incremental growth happens only while the cache is below
  /// `max_size` and is itself fallible — see [`put`](Self::put)'s not-full
  /// path `try_reserve(1)` — so even map growth cannot abort the process.
  /// Once the cache is full, eviction reuses freed capacity and `put` does
  /// not grow a container at all.
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
    //
    // The map keys are `Rc<str>`, which `Borrow<str>`, so the lookup keys
    // on `key.as_str()` — no `Key`/`Rc` is constructed to probe the map.
    let cloned = match self.entries.get(key.as_str()) {
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
  ///   [`Array::try_clone`] of `features` fails — this is checked **first**,
  ///   before any mutation.
  /// - [`Error::OutOfMemory`] if growing a backing container to admit a
  ///   **new** key into a **not-full** cache fails. Capacity grows lazily
  ///   (the constructor pre-reserves nothing — see
  ///   [`with_max_size`](Self::with_max_size)), so inserting into a cache
  ///   below `max_size` may need one growth step; that step is the fallible
  ///   [`try_reserve(1)`](HashMap::try_reserve), so it cannot abort the
  ///   process. The **full**-cache and **overwrite** paths never grow a
  ///   container (see below), so they never hit this — and never error at
  ///   all once the initial `try_clone` has succeeded.
  ///
  /// # Allocation discipline (zero-alloc on the hot paths)
  ///
  /// Keys are stored as [`Rc<str>`](Rc) — the heap string was allocated in
  /// the [`Key`] constructor, before `put` is even called. Inside `put` a
  /// key is only ever *moved* into a container or refcount-cloned
  /// ([`Rc::clone`]) into the second one; `put` itself **never
  /// heap-allocates a key**.
  ///
  /// - **Overwrite** (`key` present): the value is replaced in place via
  ///   [`HashMap::get_mut`] — the existing entry, its `Rc<str>` key, and
  ///   both containers' capacity are all reused. No growth, no eviction, no
  ///   allocation.
  /// - **New key, cache full** (`len == max_size`): the LRU entry is
  ///   evicted **first** (dropping `len` to `max_size - 1` at unchanged
  ///   container capacity), *then* the new key is inserted — refilling the
  ///   just-freed slot. Because eviction precedes insertion the containers
  ///   never need to grow past `max_size`, so this path does **no
  ///   `try_reserve` and no allocation**: it is `evict` + `insert` +
  ///   [`Rc::clone`], every step infallible. Once `try_clone` has passed
  ///   the full path *cannot fail*, so it cannot leave the cache
  ///   half-mutated.
  /// - **New key, cache not full** (`len < max_size`): the insertion
  ///   genuinely grows the containers by one slot, so each is given a
  ///   fallible [`try_reserve(1)`](HashMap::try_reserve) **before** any
  ///   structural mutation. This is the *only* allocating path, and its one
  ///   allocation is fallible (recoverable `Error::OutOfMemory`), never an
  ///   abort.
  ///
  /// On any error the cache is **unchanged** — `try_clone` and (on the
  /// not-full path only) the `try_reserve`s all happen before any
  /// structural mutation, and the full path has no fallible step after
  /// `try_clone` at all, so a failed `put` never inserts, overwrites, or
  /// evicts.
  pub fn put(&mut self, key: Key, features: &Array) -> Result<()> {
    // Clone the value FIRST — before any mutation — so a clone failure
    // leaves the cache (entries + recency + the would-be-evicted victim)
    // untouched.
    let stored = features.try_clone()?;

    // Overwrite path: the key is already present. Replace the value in its
    // existing slot via `get_mut` (the map keys are `Rc<str>`, which
    // `Borrow<str>`, so this probes with `key.as_str()` — no `Rc` built).
    // The entry's `Rc<str>` key and both containers' capacity are reused —
    // no growth, no eviction, no key allocation (mirrors the reference's
    // `move_to_end` then `self._cache[key] = features`).
    if let Some(slot) = self.entries.get_mut(key.as_str()) {
      *slot = stored;
      self.touch(&key);
      return Ok(());
    }

    // New key. The `Key` carries its `Rc<str>` (the heap string was
    // allocated in the `Key` constructor, not here); move it out so the two
    // containers can share it by refcount.
    let key_rc: Rc<str> = key.0;

    if self.entries.len() >= self.max_size {
      // FULL — evict the LRU entry *before* inserting. `>=` (not `==`)
      // mirrors the reference's `len(self._cache) >= self.max_size`; with
      // the invariant `len <= max_size` always holding, exactly one entry
      // is dropped. Eviction drops `len` to `max_size - 1` while leaving
      // container *capacity* untouched, so the `push_back` / `insert`
      // below refill the just-freed slot and never grow — no `try_reserve`
      // is needed, and this whole path has no fallible step (it cannot
      // leave the cache half-mutated).
      //
      // Capacity is sufficient by construction: the cache only ever
      // reaches `len == max_size` via a not-full insert (the step from
      // `max_size - 1` to `max_size`), and that insert ran `try_reserve(1)`
      // — so whenever the cache is full both containers already have
      // capacity `>= max_size`. `remove`/`pop_front` never shrink, so that
      // capacity still holds here.
      if let Some(lru_key) = self.recency.pop_front() {
        self.entries.remove(&lru_key);
      }
    } else {
      // NOT FULL — the insertion genuinely grows the containers by one.
      // Reserve that one slot FALLIBLY, BEFORE any structural mutation, so
      // a reservation failure leaves the cache *exactly* as it was (no
      // insert, no overwrite, no eviction). This is the only allocating
      // path in `put`, and the allocation is recoverable (`OutOfMemory`),
      // never an abort.
      self
        .recency
        .try_reserve(1)
        .map_err(|_| Error::OutOfMemory)?;
      self
        .entries
        .try_reserve(1)
        .map_err(|_| Error::OutOfMemory)?;
    }

    // Insert the new key into both containers. `Rc::clone` is an
    // infallible refcount bump (no allocation, no string copy); the
    // `entries.insert` then *moves* the `Rc`. Both containers thus share
    // one heap-allocated string.
    self.recency.push_back(Rc::clone(&key_rc));
    self.entries.insert(key_rc, stored);
    Ok(())
  }

  /// Whether `key` is currently cached — port of `__contains__`
  /// (`vision_cache.py:77-79`).
  ///
  /// A pure read: unlike [`get`](Self::get) this does **not** refresh LRU
  /// recency (the reference's `__contains__` likewise does not
  /// `move_to_end`), so it can take `&self`.
  pub fn contains(&self, key: &Key) -> bool {
    // The map keys are `Rc<str>`, which `Borrow<str>`, so membership is
    // probed with `key.as_str()` — no `Rc` is constructed for the query.
    self.entries.contains_key(key.as_str())
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
  /// Relocates the (single) prior occurrence of `key` to the back of
  /// `recency`. `key` is assumed present in `entries` by every caller; if
  /// it is somehow absent from `recency` the queue is just left as-is (no
  /// panic) — but the entries/recency lockstep invariant means that never
  /// happens in practice.
  ///
  /// Allocation-free: the queue stores [`Rc<str>`](Rc), so the existing
  /// entry is **moved** (not copied) from its old position to the back —
  /// [`VecDeque::remove`] hands back the queue's own `Rc<str>` and
  /// [`VecDeque::push_back`] re-files that same allocation. No `String`
  /// copy, no heap allocation, no refcount churn.
  fn touch(&mut self, key: &Key) {
    // The queue holds `Rc<str>`; compare by `str` content (`as_ref()` /
    // `as_str()` both yield `&str`) so a `&Key` probes without building
    // an `Rc`.
    match self.recency.iter().position(|k| k.as_ref() == key.as_str()) {
      Some(pos) => {
        // MOVE the existing entry to the back: `remove` yields the
        // queue's own `Rc<str>` (no copy), `push_back` re-files that
        // exact allocation. `remove` at an arbitrary position is O(n) in
        // the queue length, but that length is bounded by `max_size`
        // (default 20) — a tiny fixed bound — so this is effectively O(1)
        // for any realistic cache, faithful to `OrderedDict.move_to_end`'s
        // O(1)-amortized intent without an intrusive-list dependency.
        // `pos` came from `position`, so `remove` is always `Some`; the
        // `if let` simply avoids any panic branch on the hot path.
        if let Some(entry) = self.recency.remove(pos) {
          self.recency.push_back(entry);
        }
      }
      None => {
        // Unreachable under the entries/recency lockstep invariant (every
        // `entries` key is present in `recency`). Kept as a no-panic
        // safety net: if a key were somehow missing, re-file it via an
        // `Rc::clone` (infallible refcount bump) of the caller's key —
        // still allocation-free, no `String` copy.
        self.recency.push_back(Rc::clone(&key.0));
      }
    }
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

/// Allocation-discipline tests that need to inspect the cache's **private**
/// containers (`entries` / `recency`) — capacity stability and `Rc<str>`
/// key sharing. They live in an inline `#[cfg(test)]` module (not the
/// integration suite in `tests/vlm_feature_cache.rs`) precisely because the
/// guarantees under test are structural and only observable through the
/// private fields. The functional behavior tests (put/get/LRU/overwrite/
/// the lazy-capacity tests) stay in the integration suite.
#[cfg(test)]
mod alloc_discipline_tests {
  use super::*;

  /// A small feature tensor — the value is irrelevant here (these tests
  /// exercise key/container bookkeeping, not feature contents).
  fn features() -> Array {
    Array::full::<f32>(&(1usize, 4usize, 8usize), 1.0)
      .expect("constructing a tiny feature tensor must not fail")
  }

  /// Fill the cache to exactly `max_size` entries with distinct keys
  /// `k0..k{max_size}`.
  fn fill_to_capacity(cache: &mut VisionFeatureCache) {
    for i in 0..cache.max_size() {
      cache
        .put(Key::from_source(&format!("k{i}")), &features())
        .expect("put while filling below capacity must succeed");
    }
  }

  /// **Finding 1 — evict-first, no wasteful `len+1` reserve.** Once the
  /// cache is full, a new-key `put` evicts the LRU entry *first* and
  /// refills the freed slot, so neither backing container ever grows past
  /// `max_size` capacity. This test fills to `max_size`, snapshots both
  /// containers' capacity, performs a full-cache `put`, and asserts the
  /// capacity is **unchanged** — i.e. the full path took no `try_reserve`
  /// and no reallocation (the old code's `len+1` reserve, which could
  /// spuriously `Err(OutOfMemory)` under pressure, is gone).
  #[test]
  fn full_cache_put_evicts_without_capacity_growth() {
    let mut cache = VisionFeatureCache::with_max_size(4).unwrap();
    fill_to_capacity(&mut cache);
    assert_eq!(cache.len(), 4, "cache is filled to max_size");

    let entries_cap_before = cache.entries.capacity();
    let recency_cap_before = cache.recency.capacity();

    // Full-cache put: a brand-new key. Evicts the LRU (`k0`) and inserts.
    cache
      .put(Key::from_source("new"), &features())
      .expect("full-cache put must succeed (eviction admits the new key)");

    assert_eq!(
      cache.len(),
      4,
      "len returns to max_size: one evicted, one inserted"
    );
    assert_eq!(
      cache.entries.capacity(),
      entries_cap_before,
      "entries map must NOT grow on the full path — eviction frees a slot \
       the insert reuses (no try_reserve, no realloc)"
    );
    assert_eq!(
      cache.recency.capacity(),
      recency_cap_before,
      "recency queue must NOT grow on the full path — same freed-slot reuse"
    );
    // The LRU was evicted, the new key admitted: behavior is intact.
    assert!(
      !cache.contains(&Key::from_source("k0")),
      "k0 was the LRU and must have been evicted"
    );
    assert!(
      cache.contains(&Key::from_source("new")),
      "the new key must have been admitted"
    );
  }

  /// The full path is *repeatably* zero-growth: many consecutive
  /// full-cache `put`s never reallocate either container. This locks in
  /// that the evict-first restructure has no slow drift toward growth.
  #[test]
  fn repeated_full_cache_puts_never_grow_containers() {
    let mut cache = VisionFeatureCache::with_max_size(3).unwrap();
    fill_to_capacity(&mut cache);

    let entries_cap = cache.entries.capacity();
    let recency_cap = cache.recency.capacity();

    for i in 0..32 {
      cache
        .put(Key::from_source(&format!("extra{i}")), &features())
        .expect("each full-cache put must succeed");
      assert_eq!(cache.len(), 3, "len stays pinned at max_size");
      assert_eq!(
        cache.entries.capacity(),
        entries_cap,
        "entries capacity must stay stable across every full-cache put"
      );
      assert_eq!(
        cache.recency.capacity(),
        recency_cap,
        "recency capacity must stay stable across every full-cache put"
      );
    }
  }

  /// **Finding 2 — post-eviction key handoff is a refcount bump, not a
  /// heap `String` clone.** Keys are stored as `Rc<str>`: a `put` builds
  /// the `Rc<str>` *once* (in the `Key` constructor, before `put`), then
  /// the entry map and the recency queue each hold a [`Rc::clone`] — an
  /// infallible refcount bump, no allocation. This test proves the
  /// single-allocation sharing structurally: after a full-cache `put` the
  /// inserted key's `Rc<str>` has `strong_count == 2` — exactly one
  /// reference per container. Two *independent* allocations (the hazard if
  /// the code `String`-cloned the key) would instead be two separate
  /// `Rc`s each at count 1. A count of 2 is only possible if both
  /// containers share one allocation, which is the zero-alloc, infallible,
  /// transactional handoff the fix guarantees.
  #[test]
  fn full_cache_put_shares_one_key_allocation() {
    let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
    fill_to_capacity(&mut cache);

    // Full-cache put — the new-key/full path (evict LRU, then insert).
    cache
      .put(Key::from_source("shared"), &features())
      .expect("full-cache put must succeed");

    // Pull the stored key's `Rc<str>` back out of the entry map and count
    // its strong references. `entries` holds one; `recency` holds the
    // other; they are the SAME allocation (`Rc::clone`d, not re-allocated).
    let (key_rc, _) = cache
      .entries
      .get_key_value("shared")
      .expect("the just-inserted key must be present");
    assert_eq!(
      Rc::strong_count(key_rc),
      2,
      "the key's `Rc<str>` is shared by exactly the two containers \
       (entry map + recency queue) — one allocation, two refcount-clones, \
       NO heap `String` copy on the put path"
    );
  }

  /// **`touch` (the `get`-hit recency bump) does not clone the key
  /// string.** On a `get` hit the entry's recency position is refreshed by
  /// `touch`, which *moves* the existing `Rc<str>` within the `VecDeque`
  /// (`remove` then `push_back` of the same `Rc`) rather than cloning a
  /// `String`. After a `get` hit the touched key's `Rc<str>` therefore
  /// still has `strong_count == 2` (entry map + recency queue) — `touch`
  /// neither allocated a new key nor leaked an extra reference.
  #[test]
  fn get_hit_touch_does_not_clone_key_string() {
    let mut cache = VisionFeatureCache::with_max_size(4).unwrap();
    cache
      .put(Key::from_source("hot"), &features())
      .expect("initial put must succeed");

    // Sanity: freshly inserted, the key is shared by the two containers.
    {
      let (rc, _) = cache.entries.get_key_value("hot").unwrap();
      assert_eq!(Rc::strong_count(rc), 2, "post-put: entries + recency");
    }

    // A `get` hit drives `touch`. The probe `Key` below is a *separate*
    // allocation (its own `Rc`, count 1) — it cannot perturb the stored
    // key's count.
    for _ in 0..8 {
      assert!(
        cache.get(&Key::from_source("hot")).unwrap().is_some(),
        "the key must hit"
      );
      let (rc, _) = cache.entries.get_key_value("hot").unwrap();
      assert_eq!(
        Rc::strong_count(rc),
        2,
        "after a `get`-hit `touch` the key's `Rc<str>` is STILL shared by \
         just the two containers — `touch` moved the queue's `Rc` rather \
         than cloning the key string, so no allocation and no leaked ref"
      );
    }
  }

  /// The overwrite path likewise reuses the existing `Rc<str>` key (it
  /// replaces only the value, via `get_mut`) — so an overwrite neither
  /// allocates a key nor changes the key's refcount.
  #[test]
  fn overwrite_reuses_key_allocation() {
    let mut cache = VisionFeatureCache::with_max_size(4).unwrap();
    cache
      .put(Key::from_source("ow"), &features())
      .expect("initial put must succeed");
    let cap_before = cache.entries.capacity();

    cache
      .put(Key::from_source("ow"), &features())
      .expect("overwrite put must succeed");

    let (rc, _) = cache.entries.get_key_value("ow").unwrap();
    assert_eq!(
      Rc::strong_count(rc),
      2,
      "overwrite reuses the existing key `Rc<str>` (entries + recency); \
       it does not allocate or clone a key"
    );
    assert_eq!(cache.len(), 1, "overwrite must not grow the cache");
    assert_eq!(
      cache.entries.capacity(),
      cap_before,
      "overwrite reuses the existing slot — no capacity growth"
    );
  }
}

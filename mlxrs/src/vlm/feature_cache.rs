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
//!   [`Key::from_source`] (the `str` branch — path/URL), [`Key::from_sources`]
//!   (the `list` branch — a multi-image source), and [`Key::from_bytes`]
//!   (the PIL branch — a content hash). Each constructor prefixes a distinct
//!   variant tag (`s:` / `l:` / `b:`), and the list variant length-prefixes
//!   its components. This is a deliberate deviation — the reference's encoding
//!   *aliases* distinct image identities (a `'|'`-joined list collides with a
//!   literal `'|'`-bearing path; a `pil:`-hashed key collides with a literal
//!   `pil:…` path), which would silently feed one image's cached embeddings to
//!   a different image. The variant tag makes **cross-variant** collision
//!   impossible by construction, and the length-prefix makes the
//!   [`from_sources`](Key::from_sources) list encoding **injective** (see
//!   [`Key`]'s "Internal representation" note). The per-variant key contract
//!   is:
//!   - [`from_source`](Key::from_source) / [`from_sources`](Key::from_sources)
//!     carry the **full source string(s)** verbatim (tagged, and the list
//!     length-prefixed), so they are **injective** — distinct sources always
//!     produce distinct keys, so a cache hit can never return a different
//!     image's features.
//!   - [`from_bytes`](Key::from_bytes) **digests** arbitrary image bytes to a
//!     fixed-width value, so it is a digest, *not* an injection — it is
//!     **collision-resistant**, not collision-free. A digest maps an unbounded
//!     byte space onto fixed-width output, so a collision is possible in
//!     principle; the 128-bit width (below) makes it astronomically unlikely
//!     (see that constructor's note).
//!
//!   [`Key`] holds the encoded string as an [`Arc<str>`](std::sync::Arc) (an
//!   implementation detail — every public constructor / accessor has the same
//!   signature and semantics it would with a `String` field); the cache stores
//!   `Arc<str>` clones in both its containers, so the recency queue and the
//!   entry map share one heap-allocated string and [`put`] never heap-copies a
//!   key (see [`Key`]'s "Internal representation" note and
//!   [`VisionFeatureCache`]'s "Key storage" note). `Arc` (not `Rc`) keeps the
//!   public [`Key`] type `Send + Sync` — see [`Key`]'s "Internal
//!   representation" note. Because mlxrs has no PIL type and adds no crypto
//!   dependency, [`Key::from_bytes`] does not use the reference's `sha256`; it
//!   builds a **128-bit** digest from two domain-separated
//!   [`DefaultHasher`](std::hash::DefaultHasher) (SipHash) passes. 128 bits
//!   lifts the birthday bound to ≈2⁶⁴ distinct images before a collision is
//!   *expected* — practically unreachable for a cache — so a collision is
//!   negligible without pulling any new crate (a content hash here is a cache
//!   key, never a security boundary; cryptographic strength is not required,
//!   only practical collision-resistance).
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
  sync::Arc,
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
/// | Python branch | constructor | encoded form | contract |
/// |---|---|---|---|
/// | `isinstance(image_source, str)` — path / URL | [`Key::from_source`] | `s:<source>` | injective |
/// | `isinstance(image_source, list)` — multi-image | [`Key::from_sources`] | `l:` + length-prefixed components | injective |
/// | PIL image — `sha256(tobytes())[:16]` | [`Key::from_bytes`] | `b:<128-bit hexdigest>` | collision-resistant |
///
/// Two `Key`s are equal iff their encoded strings are equal. **List order
/// is significant** (matching the reference) — `["a", "b"]` and `["b", "a"]`
/// are different keys.
///
/// # Per-variant key contract
///
/// The three constructors do **not** share one guarantee — the "contract"
/// column above is exact:
///
/// - [`from_source`](Self::from_source) and [`from_sources`](Self::from_sources)
///   are **injective**: they carry the full source string(s) verbatim (tagged,
///   and the list length-prefixed), so distinct image identities *always*
///   produce distinct keys. A cache hit on one of these keys can never return
///   a different image's features.
/// - [`from_bytes`](Self::from_bytes) is **collision-resistant**, not
///   injective: it *digests* arbitrary image bytes onto a fixed-width 128-bit
///   value, so by the pigeonhole principle two different byte slices *can* in
///   principle map to the same key. The 128-bit digest makes that
///   astronomically unlikely (birthday bound ≈2⁶⁴ images), so for any
///   practical workload a `from_bytes` collision never occurs — but the
///   guarantee is collision-*resistance*, not the injectivity the
///   string-carrying variants give.
///
/// The **variant tag** (`s:` / `l:` / `b:`) is a separate, unconditional
/// guarantee that holds for *all three*: two keys from *different*
/// constructors can never be equal, so `from_bytes`'s digest can never alias a
/// literal path/list source (and vice versa) regardless of the digest's value.
///
/// # Internal representation — unambiguous encoding
///
/// The key is **not** the reference's bare normalized string. The reference
/// derives a `str` that *aliases* distinct image identities, and a cache
/// hit on an aliased key returns the wrong stored features — silently
/// feeding one image's embeddings to a different image/prompt. Two concrete
/// aliasing bugs in the reference's scheme, and how mlxrs's encoding closes
/// each:
///
/// - **Cross-variant aliasing.** The reference joins a list with `'|'` and
///   hashes PIL bytes with a `pil:` prefix, but a single-source `str` is
///   used verbatim — so a literal path `"a|b"` collides with the list
///   `["a", "b"]`, and a literal path `"pil:deadbeef"` collides with a
///   `from_bytes` digest. mlxrs prefixes each constructor with a **distinct
///   variant tag**: `s:` for [`from_source`](Self::from_source), `l:` for
///   [`from_sources`](Self::from_sources), `b:` for [`from_bytes`](Self::from_bytes).
///   The tag is the first two bytes of every key, so two keys from
///   *different* constructors can never be equal — regardless of what the
///   user's source string contains. A source string of literally `"l:x"`
///   encodes to `s:l:x` (an `s:` key); it cannot equal any `l:` key,
///   because the tag is prepended to — never spoofable from within — the
///   user's bytes.
/// - **Within-list aliasing.** A bare `'|'`-join is not injective: a list
///   *component* may itself contain `'|'`, so `["a|b"]` and `["a", "b"]`
///   both join to `"a|b"`. [`from_sources`](Self::from_sources) instead
///   **length-prefixes** every component — `<byte-len>:<component>` — so the
///   decode boundaries are unambiguous whatever characters a component
///   holds: `["a|b"]` encodes `l:3:a|b`, `["a", "b"]` encodes `l:1:a1:b`,
///   and the two differ. The list encoding is injective.
///
/// Together the variant tag (kills cross-variant aliasing) and the
/// length-prefixed list components (kill within-list aliasing) make the two
/// **string-carrying** variants — [`from_source`](Self::from_source) and
/// [`from_sources`](Self::from_sources) — **injective**: distinct path/URL/list
/// identities always produce distinct keys, so a cache hit on those can never
/// return a different image's features. The third variant,
/// [`from_bytes`](Self::from_bytes), is a fixed-width *digest* of the raw
/// bytes, so it is **collision-resistant** rather than injective (a digest
/// cannot be injective over an unbounded byte space — see the "Per-variant key
/// contract" section above and that constructor's note); its variant tag still
/// unconditionally prevents cross-variant aliasing with the two injective
/// variants. The encoded form is an internal cache key —
/// [`as_str`](Self::as_str) exposes it for tests/introspection, but no caller
/// parses it back into a source.
///
/// The encoded string is held as an [`Arc<str>`](Arc), not a `String`.
/// This is an implementation detail — every public method
/// ([`from_source`](Self::from_source), [`from_sources`](Self::from_sources),
/// [`from_bytes`](Self::from_bytes), [`as_str`](Self::as_str), and the
/// `From<&str>` conversion) has the exact same signature and behavior it
/// would with a `String` field. The `Arc` backing buys three things:
///
/// - **`Clone` is a refcount bump** — infallible, no heap allocation, no
///   string copy. [`VisionFeatureCache`] stores a key in *two* containers
///   (the entry map and the recency queue); with an `Arc<str>` the second
///   container gets an [`Arc::clone`], so a [`put`](VisionFeatureCache::put)
///   never heap-copies a key and the post-eviction key handoff cannot
///   fail. (With a `String` field that second copy was a fallible-by-abort
///   heap allocation occurring *after* eviction — a transactional hazard.)
///   The bump is a single atomic increment — negligible for this cache's
///   use, and the same allocation-free handoff a non-atomic `Rc` gave.
/// - **`Hash`/`Eq` are unchanged** — `Arc<str>` hashes and compares by the
///   pointed-to `str` *content* (it derefs / `Borrow`s `str`), so the
///   derived `Hash`/`PartialEq`/`Eq` here are byte-for-byte the same
///   relation as a `String`-backed `Key`: two `Key`s are equal iff their
///   strings are equal, full stop.
/// - **`Send + Sync` are preserved** — `Arc<str>` is `Send + Sync` (a
///   non-atomic `Rc<str>` is neither), so the public `Key` keeps the
///   `Send`/`Sync` auto-traits a `String`-backed `Key` had. Downstream code
///   may precompute, queue, or move `Key`s across thread/task boundaries.
///   (This is `Key` alone — [`VisionFeatureCache`] stores [`Array`], which
///   is intentionally `!Send`/`!Sync`; only the *key* is thread-portable.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key(Arc<str>);

// Compile-time guard: the public `Key` must stay `Send + Sync`. The prior
// `Rc<str>` backing silently dropped both auto-traits; `Arc<str>` restores
// them. A regression back to `Rc` (or any other `!Send`/`!Sync` field)
// fails this assertion at compile time. `VisionFeatureCache` itself is
// deliberately NOT asserted here — it stores `Array`, which is intentionally
// `!Send`/`!Sync` (one cache belongs to one inference thread); only the
// thread-portable `Key` carries the contract.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Key>();
};

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
    // The encoded key is `"s:" + raw`. The `s:` variant tag namespaces this
    // constructor: a `from_source` key always starts `s:`, a `from_sources`
    // key `l:`, a `from_bytes` key `b:` — so no two constructor variants can
    // ever produce equal keys, *even* if the user's `source` string is
    // literally `l:...` or `b:pil:...` (that just becomes `s:l:...` /
    // `s:b:pil:...`, still uniquely an `s:` key). See the `Key` type's
    // "Internal representation — unambiguous encoding" note.
    //
    // `Arc::from(String)` allocates the shared string once, here in the
    // constructor — the cache's `put` then only ever *moves* and refcount-
    // clones this `Arc`, never re-allocates the key. `with_capacity` sizes
    // the buffer exactly (`"s:"` + `source`) so the `push_str`es never
    // reallocate.
    let mut encoded = String::with_capacity(2 + source.len());
    encoded.push_str("s:");
    encoded.push_str(source);
    Self(Arc::from(encoded))
  }

  /// Key for a multi-image source — the reference's
  /// `isinstance(image_source, list)` branch (`vision_cache.py:44-45`),
  /// which `'|'`-joins the per-image source strings.
  ///
  /// **Order is significant**: `from_sources(&["a", "b"])` differs from
  /// `from_sources(&["b", "a"])`.
  ///
  /// # Encoding — length-prefixed, not delimiter-joined
  ///
  /// mlxrs does **not** use the reference's bare `'|'`-join. A plain join is
  /// not injective: a component may itself contain the `'|'` delimiter, so
  /// `["a|b"]` and `["a", "b"]` would both join to `"a|b"` and alias to the
  /// same key — silently feeding one image-list's cached embeddings to a
  /// *different* image list. Instead each component is **length-prefixed**:
  /// the encoding is `"l:"` followed, per component, by `<byte-len> + ":" +
  /// <component>`. The byte length is the component's UTF-8 length, so the
  /// decoder boundary is unambiguous regardless of which characters
  /// (including `'|'` or `':'`) the component contains — the list encoding
  /// is injective. `["a|b"]` encodes `l:3:a|b`; `["a", "b"]` encodes
  /// `l:1:a1:b`; they differ.
  ///
  /// The `l:` variant tag also namespaces this constructor against
  /// [`from_source`](Self::from_source) (`s:`) and [`from_bytes`](Self::from_bytes)
  /// (`b:`) — see the `Key` type's "Internal representation" note.
  ///
  /// An empty slice yields the bare tag `"l:"` (still distinct from every
  /// other key); a single-element slice `["x"]` encodes `l:1:x`, which —
  /// unlike the reference's `"|".join(["x"]) == "x"` — does **not** equal
  /// [`from_source`](Self::from_source) of `"x"` (that is `s:x`). The
  /// non-aliasing guarantee is strictly stronger than the reference here,
  /// which is the intended fix.
  pub fn from_sources(sources: &[&str]) -> Self {
    use std::fmt::Write as _;

    // Pre-size the buffer exactly: the `l:` tag, then per component its
    // decimal byte-length, a `':'` separator, and the component bytes. This
    // exact `with_capacity` means the writes below never reallocate.
    let mut cap = 2; // "l:"
    for s in sources {
      let len = s.len();
      // Decimal digit count of `len`: `ilog10() + 1` for `len >= 1`; the
      // `len == 0` component is one digit (`"0"`). `0.ilog10()` would panic,
      // so the `== 0` arm is taken explicitly.
      let digits = if len == 0 {
        1
      } else {
        len.ilog10() as usize + 1
      };
      cap += digits + 1 + len; // <digits> + ':' + <component>
    }
    let mut encoded = String::with_capacity(cap);
    encoded.push_str("l:");
    for s in sources {
      // Length-prefix each component: `<byte-len>:<component>`. `s.len()` is
      // the UTF-8 byte length, so the next component begins exactly `len`
      // bytes after the `':'` separator — the boundary is unambiguous even
      // if `s` itself contains `'|'`, `':'`, or digits. That is what makes
      // the list encoding injective: the reference's bare `'|'`-join was
      // not (`["a|b"]` and `["a", "b"]` both joined to `"a|b"`).
      //
      // `write!` formats the `usize` length straight into `encoded` — no
      // intermediate `String`. Writing to a `String` is infallible, so the
      // `Result` is discarded (`let _`); the only `fmt::Error` source is a
      // failing `Write` impl and `String`'s never fails.
      let _ = write!(encoded, "{}:{}", s.len(), s);
    }
    // `Arc::from` consumes the buffer into the shared `Arc<str>` the cache
    // stores; the whole allocation cost is borne here in the constructor,
    // not on the `put` hot path.
    Self(Arc::from(encoded))
  }

  /// Key for an in-memory image with no stable path — the reference's
  /// PIL branch (`vision_cache.py:47-49`): hash the raw image bytes.
  ///
  /// # Contract: collision-resistant, *not* injective
  ///
  /// Unlike [`from_source`](Self::from_source) /
  /// [`from_sources`](Self::from_sources) (which carry the full source string
  /// and are therefore **injective** — distinct sources always yield distinct
  /// keys), this constructor **digests** arbitrary image bytes onto a
  /// fixed-width value. A digest cannot be injective over an unbounded byte
  /// space (pigeonhole), so two *different* byte slices *can* in principle map
  /// to the same key. The digest below is **128-bit**, which makes that
  /// astronomically unlikely — the birthday bound is ≈2⁶⁴ distinct images
  /// before a collision is even *expected*, far beyond any cache's lifetime —
  /// so for any practical workload a `from_bytes` collision never happens. But
  /// the guarantee is **collision-resistance**, not the injectivity the
  /// string-carrying variants give: callers must not assume two distinct
  /// images can *never* share a `from_bytes` key, only that it is
  /// negligibly unlikely.
  ///
  /// The reference uses `sha256(tobytes())[:16]`. mlxrs builds the 128-bit
  /// digest from two domain-separated [`DefaultHasher`](std::hash::DefaultHasher)
  /// (SipHash) passes instead of pulling a crypto crate: a content hash here
  /// is a **cache key**, never a security boundary, so cryptographic strength
  /// is not required — only practical collision-resistance, which 128 bits of
  /// SipHash provides. This adds no new dependency (see the module-level
  /// "Deviations" note).
  ///
  /// ## Why two passes (and why domain-separated)
  ///
  /// A single [`DefaultHasher`](std::hash::DefaultHasher) yields only 64 bits
  /// (birthday bound ≈2³² images — uncomfortably reachable). Two passes give
  /// 128 bits, but a naïve "hash the same bytes twice" would produce two
  /// *identical* 64-bit halves, because `DefaultHasher::new()` is always seeded
  /// with the same fixed SipHash keys — so the second half would carry no new
  /// information. Each pass is therefore **domain-separated** by feeding a
  /// distinct one-byte tag (`0` then `1`) into the hasher *before* the image
  /// bytes; the two halves then depend on the input through two different
  /// SipHash inputs and are effectively independent, giving the full 128-bit
  /// width.
  ///
  /// # Encoding
  ///
  /// The result is `"b:"` followed by the fixed-width **32-hex-char** (128-bit)
  /// digest. The `b:` variant tag namespaces this constructor against
  /// [`from_source`](Self::from_source) (`s:`) and
  /// [`from_sources`](Self::from_sources) (`l:`). This is what makes
  /// cross-variant aliasing **impossible by construction** (independent of the
  /// digest's collision-resistance): a `from_bytes` key always starts `b:`, a
  /// `from_source` key always `s:`, so a literal path source — even one named
  /// exactly like a digest, or the reference's old `pil:`-prefixed shape —
  /// encodes to `s:...` and can never equal a real `from_bytes` key. (The tag
  /// is on the *outside* and the user's bytes never reach it.)
  pub fn from_bytes(bytes: &[u8]) -> Self {
    // 128-bit digest = two domain-separated 64-bit SipHash passes. Each pass
    // is seeded with the same fixed `DefaultHasher` keys, so they are
    // distinguished by a leading one-byte domain tag (`0` / `1`) hashed
    // *before* `bytes`; without that the two halves would be identical and
    // the key would carry only 64 bits of entropy.
    let half = |domain: u8| -> u64 {
      let mut hasher = std::hash::DefaultHasher::new();
      hasher.write_u8(domain);
      bytes.hash(&mut hasher);
      hasher.finish()
    };
    let hi = half(0);
    let lo = half(1);
    // `format!` builds the `b:`-prefixed fixed-width hex digest `String`;
    // `Arc::from` consumes it into the shared `Arc<str>` (transient buffer
    // freed). Each half is 16 hex chars, so the encoded key is always exactly
    // `"b:" + 32` chars.
    Self(Arc::from(format!("b:{hi:016x}{lo:016x}")))
  }

  /// The normalized key string — the internal, namespaced cache-key
  /// encoding (a `s:` / `l:` / `b:` variant tag plus the variant's payload;
  /// see the type-level "Internal representation" note). Exposed for tests
  /// / introspection only; the cache never needs the caller to read it, and
  /// no caller parses it — it is an opaque cache key, not the raw source.
  pub fn as_str(&self) -> &str {
    // `Arc<str>` derefs to `str`; `&self.0` coerces `&Arc<str>` to `&str`.
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
/// [`Arc<str>`](Arc): the entry-map key and the recency-queue entry are two
/// [`Arc::clone`]s of that one allocation. `Arc::clone` is an infallible
/// refcount bump (a single atomic increment) — **no heap allocation, no
/// string copy** — so every key-side operation on the [`put`] path
/// (inserting into both containers) and on the recency-update path (a
/// [`get`] hit relocates the key within the recency queue) is
/// allocation-free. The string a [`Key`] carries is consumed into the
/// `Arc<str>` once, on the inserting `put`; after that no `put` or `get`
/// ever heap-allocates a key. This is what makes a full-cache `put`
/// (evict + insert) and a `get` hit's recency bump strictly
/// allocation-free on the key side, and what makes the post-eviction key
/// handoff infallible (a refcount bump cannot fail), so a failed `put` can
/// never leave the cache half-mutated.
///
/// # Concurrency
///
/// Neither `Send` nor `Sync` — it stores [`Array`], which is intentionally
/// `!Send` + `!Sync`. One cache belongs to one inference thread, the same
/// single-thread contract the rest of `mlxrs` is built on. Note this is the
/// *cache* type only: its [`Key`] type *is* `Send + Sync` (it is backed by
/// an `Arc<str>`), so keys may be precomputed or moved across threads even
/// though the cache they index must not.
pub struct VisionFeatureCache {
  /// LRU bound. `>= 1` — enforced by [`Self::with_max_size`].
  max_size: usize,
  /// Key → feature-`Array` map. Holds the owned (refcount-shared)
  /// `Array` handles. The map *key* is an [`Arc<str>`](Arc): the same
  /// allocation the recency queue holds (see the struct-level "Key
  /// storage" note), so admitting a key never heap-copies the string.
  /// `Arc<str>` hashes and compares by content (it `Borrow`s `str`), so
  /// lookups with a `&Key` work via `key.as_str()`.
  entries: HashMap<Arc<str>, Array>,
  /// Recency queue, **least-recently-used at the front**, most-recent at
  /// the back — the explicit mirror of `OrderedDict`'s insertion order.
  /// `move_to_end` is "remove this key, push it to the back"; eviction is
  /// "pop the front" (`popitem(last=False)`). Every key in `entries` is
  /// present exactly once in `recency`, and vice versa — the two are kept
  /// in lockstep by every mutating method.
  ///
  /// Holds [`Arc<str>`](Arc) clones of the *same* allocations the entry map
  /// keys point at. Pushing/moving a key here is therefore a refcount bump,
  /// not a `String` copy — so [`put`](Self::put) and [`touch`](Self::touch)
  /// never heap-allocate a key.
  recency: VecDeque<Arc<str>>,
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
    // The map keys are `Arc<str>`, which `Borrow<str>`, so the lookup keys
    // on `key.as_str()` — no `Key`/`Arc` is constructed to probe the map.
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
  /// Keys are stored as [`Arc<str>`](Arc) — the heap string was allocated
  /// in the [`Key`] constructor, before `put` is even called. Inside `put`
  /// a key is only ever *moved* into a container or refcount-cloned
  /// ([`Arc::clone`]) into the second one; `put` itself **never
  /// heap-allocates a key**.
  ///
  /// - **Overwrite** (`key` present): the value is replaced in place via
  ///   [`HashMap::get_mut`] — the existing entry, its `Arc<str>` key, and
  ///   both containers' capacity are all reused. No growth, no eviction, no
  ///   allocation.
  /// - **New key, cache full** (`len == max_size`): the LRU entry is
  ///   evicted **first** (dropping `len` to `max_size - 1` at unchanged
  ///   container capacity), *then* the new key is inserted — refilling the
  ///   just-freed slot. Because eviction precedes insertion the containers
  ///   never need to grow past `max_size`, so this path does **no
  ///   `try_reserve` and no allocation**: it is `evict` + `insert` +
  ///   [`Arc::clone`], every step infallible. Once `try_clone` has passed
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
    // existing slot via `get_mut` (the map keys are `Arc<str>`, which
    // `Borrow<str>`, so this probes with `key.as_str()` — no `Arc` built).
    // The entry's `Arc<str>` key and both containers' capacity are reused —
    // no growth, no eviction, no key allocation (mirrors the reference's
    // `move_to_end` then `self._cache[key] = features`).
    if let Some(slot) = self.entries.get_mut(key.as_str()) {
      *slot = stored;
      self.touch(&key);
      return Ok(());
    }

    // New key. The `Key` carries its `Arc<str>` (the heap string was
    // allocated in the `Key` constructor, not here); move it out so the two
    // containers can share it by refcount.
    let key_arc: Arc<str> = key.0;

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

    // Insert the new key into both containers. `Arc::clone` is an
    // infallible refcount bump (no allocation, no string copy); the
    // `entries.insert` then *moves* the `Arc`. Both containers thus share
    // one heap-allocated string.
    self.recency.push_back(Arc::clone(&key_arc));
    self.entries.insert(key_arc, stored);
    Ok(())
  }

  /// Whether `key` is currently cached — port of `__contains__`
  /// (`vision_cache.py:77-79`).
  ///
  /// A pure read: unlike [`get`](Self::get) this does **not** refresh LRU
  /// recency (the reference's `__contains__` likewise does not
  /// `move_to_end`), so it can take `&self`.
  pub fn contains(&self, key: &Key) -> bool {
    // The map keys are `Arc<str>`, which `Borrow<str>`, so membership is
    // probed with `key.as_str()` — no `Arc` is constructed for the query.
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
  /// Allocation-free: the queue stores [`Arc<str>`](Arc), so the existing
  /// entry is **moved** (not copied) from its old position to the back —
  /// [`VecDeque::remove`] hands back the queue's own `Arc<str>` and
  /// [`VecDeque::push_back`] re-files that same allocation. No `String`
  /// copy, no heap allocation, no refcount churn.
  fn touch(&mut self, key: &Key) {
    // The queue holds `Arc<str>`; compare by `str` content (`as_ref()` /
    // `as_str()` both yield `&str`) so a `&Key` probes without building
    // an `Arc`.
    match self.recency.iter().position(|k| k.as_ref() == key.as_str()) {
      Some(pos) => {
        // MOVE the existing entry to the back: `remove` yields the
        // queue's own `Arc<str>` (no copy), `push_back` re-files that
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
        // `Arc::clone` (infallible refcount bump) of the caller's key —
        // still allocation-free, no `String` copy.
        self.recency.push_back(Arc::clone(&key.0));
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
/// containers (`entries` / `recency`) — capacity stability and `Arc<str>`
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
  /// heap `String` clone.** Keys are stored as `Arc<str>`: a `put` builds
  /// the `Arc<str>` *once* (in the `Key` constructor, before `put`), then
  /// the entry map and the recency queue each hold an [`Arc::clone`] — an
  /// infallible refcount bump, no allocation. This test proves the
  /// single-allocation sharing structurally: after a full-cache `put` the
  /// inserted key's `Arc<str>` has `strong_count == 2` — exactly one
  /// reference per container. Two *independent* allocations (the hazard if
  /// the code `String`-cloned the key) would instead be two separate
  /// `Arc`s each at count 1. A count of 2 is only possible if both
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

    // Pull the stored key's `Arc<str>` back out of the entry map and count
    // its strong references. `entries` holds one; `recency` holds the
    // other; they are the SAME allocation (`Arc::clone`d, not re-allocated).
    // The map key is the *encoded* form, so probe with `Key::as_str()`
    // (`from_source("shared")` encodes to `s:shared`), not the raw literal.
    let (key_arc, _) = cache
      .entries
      .get_key_value(Key::from_source("shared").as_str())
      .expect("the just-inserted key must be present");
    assert_eq!(
      Arc::strong_count(key_arc),
      2,
      "the key's `Arc<str>` is shared by exactly the two containers \
       (entry map + recency queue) — one allocation, two refcount-clones, \
       NO heap `String` copy on the put path"
    );
  }

  /// **`touch` (the `get`-hit recency bump) does not clone the key
  /// string.** On a `get` hit the entry's recency position is refreshed by
  /// `touch`, which *moves* the existing `Arc<str>` within the `VecDeque`
  /// (`remove` then `push_back` of the same `Arc`) rather than cloning a
  /// `String`. After a `get` hit the touched key's `Arc<str>` therefore
  /// still has `strong_count == 2` (entry map + recency queue) — `touch`
  /// neither allocated a new key nor leaked an extra reference.
  #[test]
  fn get_hit_touch_does_not_clone_key_string() {
    let mut cache = VisionFeatureCache::with_max_size(4).unwrap();
    cache
      .put(Key::from_source("hot"), &features())
      .expect("initial put must succeed");

    // Sanity: freshly inserted, the key is shared by the two containers.
    // The map key is the *encoded* form — probe with `Key::as_str()`.
    let hot = Key::from_source("hot");
    {
      let (arc, _) = cache.entries.get_key_value(hot.as_str()).unwrap();
      assert_eq!(Arc::strong_count(arc), 2, "post-put: entries + recency");
    }

    // A `get` hit drives `touch`. The probe `Key` below is a *separate*
    // allocation (its own `Arc`, count 1) — it cannot perturb the stored
    // key's count.
    for _ in 0..8 {
      assert!(
        cache.get(&Key::from_source("hot")).unwrap().is_some(),
        "the key must hit"
      );
      let (arc, _) = cache.entries.get_key_value(hot.as_str()).unwrap();
      assert_eq!(
        Arc::strong_count(arc),
        2,
        "after a `get`-hit `touch` the key's `Arc<str>` is STILL shared by \
         just the two containers — `touch` moved the queue's `Arc` rather \
         than cloning the key string, so no allocation and no leaked ref"
      );
    }
  }

  /// The overwrite path likewise reuses the existing `Arc<str>` key (it
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

    // The map key is the *encoded* form — probe with `Key::as_str()`.
    let (arc, _) = cache
      .entries
      .get_key_value(Key::from_source("ow").as_str())
      .unwrap();
    assert_eq!(
      Arc::strong_count(arc),
      2,
      "overwrite reuses the existing key `Arc<str>` (entries + recency); \
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

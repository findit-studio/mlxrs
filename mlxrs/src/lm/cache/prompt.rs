//! Prompt-reuse data structures, ported 1:1 from
//! [`mlx_lm.models.cache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
//! (`TokenBuffer` :1487-1521, `PromptTrieResult` :1523-1530, `PromptTrie`
//! :1532-1620, `LRUPromptCache` :1623-1763) and cross-checked against
//! mlx-swift-lm's `MLXLMCommon` KV-cache prompt-reuse surface.
//!
//! This is a **prompt-reuse + disk** feature, *not* a per-layer cache: none
//! of these types implement [`KvCache`]. They cache whole
//! per-prompt `Vec<Box<dyn KvCache>>` states keyed by `(model, token
//! prefix)` so a new request that shares a prefix with a prior one can skip
//! recomputing that prefix.
//!
//! **Model key.** mlx-lm keys the trie on the *Python model object's
//! identity* (`model not in self._trie` is dict hashing on the object). Rust
//! has no object identity, so the generic `M` is the model key the caller
//! chooses (a model id / path `String`, an `usize` slot, …); it only needs
//! [`Eq`] + [`Hash`] + [`Clone`], the exact contract Python's dict-key use
//! imposes. This is the same Rust-idiomatic substitution `from_state` makes
//! for the reference's `globals()[name]` (a `&str` match) — behavior is 1:1,
//! the *spelling* of "which model" is Rust-native.
//!
//! **`TokenBuffer`.** mlx-lm's `TokenBuffer` is an over-allocated
//! `mx.array(int32)` grown in `step`-sized (256) chunks, exactly mirroring
//! `KVCache`'s step buffer. As with [`StandardKvCache`](super::StandardKvCache)
//! vs mlx-lm's `KVCache`, the step buffer is a pure allocation optimization
//! with **no** observable effect: `update_and_fetch` returns `_buffer[:end]`
//! and `.tokens` is `_buffer[:_size]` — the over-allocated tail zeros are
//! always sliced off before any observer. So the *observable* contract
//! (push, get the logical prefix) is reproduced directly over a `Vec<i32>`;
//! the type carries no `mlxrs::Array` and needs no eval, matching the
//! no-implicit-eval rule.
//!
//! No implicit eval: nothing here materializes an array (the cached
//! `Box<dyn KvCache>` states are moved/cloned via their own
//! [`KvCache::copy`], never `eval`ed).

use std::{
  collections::{HashMap, VecDeque},
  hash::Hash,
};

use crate::{
  error::{Error, MissingKeyPayload, Result},
  lm::cache::{KvCache, can_trim_prompt_cache, trim_prompt_cache},
};

// ───────────────────────── TokenBuffer ─────────────────────────

/// mlx-lm `TokenBuffer.step` — the over-allocation granularity. Kept for
/// 1:1 documentation parity; the observable contract does not depend on it
/// (the step buffer is sliced off before any observer, exactly as
/// [`StandardKvCache`](super::StandardKvCache) elides mlx-lm's `KVCache`
/// step buffer).
pub const TOKEN_BUFFER_STEP: usize = 256;

/// A simple, efficiently-appendable token buffer — port of
/// `mlx_lm.models.cache.TokenBuffer` (cache.py:1487-1521).
///
/// mlx-lm holds an over-allocated `mx.array(int32)`; the observable surface
/// is only the logical prefix (`update_and_fetch -> _buffer[:end]`, `tokens
/// -> _buffer[:_size]`), so this stores the logical tokens directly in a
/// `Vec<i32>` (the step buffer is a no-observable-effect allocation
/// optimization — same reasoning as the `StandardKvCache` port). `i32`
/// mirrors mlx-lm's `dtype=mx.int32`.
#[derive(Debug, Clone, Default)]
pub struct TokenBuffer {
  buffer: Vec<i32>,
}

impl TokenBuffer {
  /// A new buffer seeded with `tokens` (mlx-lm `TokenBuffer(tokens=[])`).
  pub fn new(tokens: &[i32]) -> Self {
    Self {
      buffer: tokens.to_vec(),
    }
  }

  /// Append `tokens` and return the full logical token slice — mlx-lm
  /// `TokenBuffer.update_and_fetch` (`self._buffer[:end]`, cache.py
  /// :1500-1512). The over-allocation in mlx-lm only affects `_buffer.size`,
  /// never the returned `[:end]`, so appending and returning the logical
  /// prefix is byte-identical. **Deviation (faithful, lint-driven):**
  /// mlx-lm's signature is "fallible" only because it allocates an
  /// `mx.array`; a `Vec<i32>` push cannot fail, so this is infallible
  /// (returning `Result` would trip the workspace's
  /// `clippy::unnecessary_wraps = deny`) and returns a borrowed `&[i32]`
  /// (no per-call allocation — the allocation-discipline rule; mlx-lm's
  /// return is itself a non-owning array view, so a borrow is the faithful
  /// shape).
  pub fn update_and_fetch(&mut self, tokens: &[i32]) -> &[i32] {
    self.buffer.extend_from_slice(tokens);
    &self.buffer
  }

  /// The logical token slice — mlx-lm `TokenBuffer.tokens`
  /// (`self._buffer[:self._size]`, cache.py:1518-1520). Borrowed (mlx-lm's
  /// `_buffer[:_size]` is itself a view; no allocation).
  pub fn tokens(&self) -> &[i32] {
    &self.buffer
  }

  /// Logical token count (mlx-lm `self._size`).
  pub fn len(&self) -> usize {
    self.buffer.len()
  }

  /// Whether the buffer is empty.
  pub fn is_empty(&self) -> bool {
    self.buffer.is_empty()
  }
}

// ───────────────────────── PromptTrie ─────────────────────────

/// Insertion-ordered child-edge map for a trie node.
///
/// mlx-lm's trie node is a Python `dict` whose key iteration is **insertion
/// order** (`for tok in current` in `PromptTrie.search`, cache.py:1617).
/// The `search` `longer` DFS pushes those children onto a **LIFO** stack
/// (cache.py:1610-1618), so equal-length value-bearing extensions resolve
/// deterministically to the *last-inserted* sibling's path (`best` is only
/// replaced on a *strictly* shorter `extra`, so among equal lengths the
/// first one popped — i.e. the last inserted — wins). A `HashMap` would
/// make that tie hash-order-dependent and could pick a different `longer`
/// than mlx-lm (changing `LruPromptCache` reuse when siblings differ in
/// trimmability). So children are kept in a `Vec` in **insertion order**
/// (the crate forbids new deps, so no `indexmap`; per-node token fanout is
/// tiny — the distinct next-token set — so linear find is appropriate and
/// matches the small logical dict mlx-lm uses). All accessors below mirror
/// the `dict` operations `add`/`get`/`pop`/`pop_prefixes`/`search` use.
struct ChildMap<V> {
  /// `(token, child)` in first-insertion order (Python dict order).
  entries: Vec<(i32, TrieNode<V>)>,
}

impl<V> ChildMap<V> {
  fn new() -> Self {
    Self {
      entries: Vec::new(),
    }
  }

  fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  fn get(&self, tok: i32) -> Option<&TrieNode<V>> {
    self
      .entries
      .iter()
      .find_map(|(k, v)| (*k == tok).then_some(v))
  }

  fn get_mut(&mut self, tok: i32) -> Option<&mut TrieNode<V>> {
    self
      .entries
      .iter_mut()
      .find_map(|(k, v)| (*k == tok).then_some(v))
  }

  /// Get the child for `tok`, inserting a fresh node **appended in
  /// insertion order** if absent (mlx-lm `if tok not in current:
  /// current[tok] = {}`, cache.py:1542-1543 — a new dict key goes last).
  /// Returns `&mut` to the (possibly just-created) child.
  fn get_or_insert(&mut self, tok: i32) -> &mut TrieNode<V> {
    // Two-step (find index, then index) to satisfy the borrow checker
    // without a second lookup map; fanout is tiny so this is cheap.
    if let Some(i) = self.entries.iter().position(|(k, _)| *k == tok) {
      &mut self.entries[i].1
    } else {
      self.entries.push((tok, TrieNode::new()));
      &mut self.entries.last_mut().expect("just pushed").1
    }
  }

  /// Remove the child edge for `tok` (mlx-lm `del parent[tok]`,
  /// cache.py:1566). Insertion order of the remaining edges is preserved
  /// (`Vec::remove` shifts; a deleted-then-reinserted token would re-append
  /// last, exactly as a Python dict `del` then re-add does).
  fn remove(&mut self, tok: i32) {
    if let Some(i) = self.entries.iter().position(|(k, _)| *k == tok) {
      self.entries.remove(i);
    }
  }

  /// Children in **insertion order** (mlx-lm `for tok in current`,
  /// cache.py:1617). The `search` DFS pushes these onto its LIFO stack in
  /// this order so `stack.pop()` yields them last-inserted-first — byte-for
  /// byte mlx-lm's traversal order.
  fn iter_insertion_order(&self) -> impl Iterator<Item = (i32, &TrieNode<V>)> {
    self.entries.iter().map(|(k, v)| (*k, v))
  }
}

/// One trie node: child edges keyed by token id, plus an optional value at
/// this node (mlx-lm stores the value under the `"__value__"` dict key; a
/// dedicated `Option` field is the faithful Rust shape — `"__value__"`
/// cannot collide with a token-id edge).
struct TrieNode<V> {
  children: ChildMap<V>,
  value: Option<V>,
}

impl<V> TrieNode<V> {
  fn new() -> Self {
    Self {
      children: ChildMap::new(),
      value: None,
    }
  }

  /// Node "size" the way mlx-lm tests `len(node) > 0` in `pop`
  /// (cache.py:1564): the dict holds child edges **and** possibly
  /// `"__value__"`, so emptiness ⇔ no children **and** no value.
  fn is_empty(&self) -> bool {
    self.children.is_empty() && self.value.is_none()
  }
}

/// The result of [`PromptTrie::search`] — port of
/// `mlx_lm.models.cache.PromptTrieResult` (cache.py:1523-1530).
///
/// `model` is dropped from this struct (Python carries it only to thread it
/// straight back into `self._trie.get(result.model, ...)`; the Rust caller
/// already has the `&M` it passed to [`search`](PromptTrie::search), so
/// re-returning it would be redundant — every field below is exactly the
/// reference's, identically computed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTrieResult {
  /// An exact match was found for the full token sequence (mlx-lm
  /// `exact`). `Some([])` is the empty-tokens-with-root-value case.
  pub exact: Option<Vec<i32>>,
  /// The longest stored prefix that carries a value (mlx-lm `shorter`).
  pub shorter: Option<Vec<i32>>,
  /// The shortest stored sequence that extends beyond `tokens` (mlx-lm
  /// `longer`).
  pub longer: Option<Vec<i32>>,
  /// Length of the common prefix with any stored path (mlx-lm
  /// `common_prefix`).
  pub common_prefix: usize,
}

/// A token-prefix trie mapping `(model, tokens)` to a value — port of
/// `mlx_lm.models.cache.PromptTrie` (cache.py:1532-1620).
///
/// `M` is the model key (see the module docs); `V` is the stored value
/// (for [`LruPromptCache`] this is a cache entry).
pub struct PromptTrie<M, V = ()> {
  trie: HashMap<M, TrieNode<V>>,
}

impl<M: Eq + Hash + Clone, V> Default for PromptTrie<M, V> {
  fn default() -> Self {
    Self::new()
  }
}

impl<M: Eq + Hash + Clone, V> PromptTrie<M, V> {
  /// An empty trie (mlx-lm `PromptTrie.__init__`).
  pub fn new() -> Self {
    Self {
      trie: HashMap::new(),
    }
  }

  /// Insert `value` at `(model, tokens)`, returning the previous value if
  /// any — mlx-lm `PromptTrie.add` (cache.py:1536-1547).
  pub fn add(&mut self, model: &M, tokens: &[i32], value: V) -> Option<V> {
    let root = self.trie.entry(model.clone()).or_insert_with(TrieNode::new);
    let mut current = root;
    for &tok in tokens {
      current = current.children.get_or_insert(tok);
    }
    current.value.replace(value)
  }

  /// The value at `(model, tokens)`, or `None` if the path / value is
  /// absent — mlx-lm `PromptTrie.get` (cache.py:1549-1553). mlx-lm raises a
  /// `KeyError` on a missing path; the recoverable Rust shape is `None`
  /// (callers in this module only `get` paths `search` already proved
  /// present, exactly as mlx-lm does).
  pub fn get(&self, model: &M, tokens: &[i32]) -> Option<&V> {
    let mut current = self.trie.get(model)?;
    for &tok in tokens {
      current = current.children.get(tok)?;
    }
    current.value.as_ref()
  }

  /// Remove and return the value at `(model, tokens)`, pruning now-empty
  /// interior nodes — mlx-lm `PromptTrie.pop` (cache.py:1555-1567).
  ///
  /// Returns `None` if the path / value is absent (mlx-lm `path[-1].pop`
  /// would `KeyError`; the recoverable shape is `None` — callers only pop
  /// paths they inserted).
  pub fn pop(&mut self, model: &M, tokens: &[i32]) -> Option<V> {
    // Take the value first; if absent, nothing to pop (mirrors Python's
    // `path[-1].pop("__value__")`).
    {
      let mut current = self.trie.get(model)?;
      for &tok in tokens {
        current = current.children.get(tok)?;
      }
      current.value.as_ref()?;
    }
    let root = self.trie.get_mut(model)?;
    // Re-walk to the terminal node taking the value, then prune empty
    // interiors bottom-up. Python keeps a `path` of node refs; Rust's
    // borrow checker forbids holding the whole ancestor chain mutably, so
    // the prune is a recursive descent that, on the way back up, deletes a
    // child edge iff that child became empty (`len(node) == 0`) — exactly
    // mlx-lm's `for i in range(len(tokens), 0, -1): if len(node) > 0:
    // break; del parent[tok]` (it stops pruning at the first non-empty
    // ancestor; the recursion's early-return-on-non-empty does the same).
    fn take_and_prune<V>(node: &mut TrieNode<V>, toks: &[i32]) -> Option<V> {
      match toks.split_first() {
        None => node.value.take(),
        Some((&tok, rest)) => {
          let child = node.children.get_mut(tok)?;
          let v = take_and_prune(child, rest)?;
          if child.is_empty() {
            node.children.remove(tok);
          }
          Some(v)
        }
      }
    }
    take_and_prune(root, tokens)
  }

  /// Pop every value strictly along the path *before* the full
  /// `tokens` — mlx-lm `PromptTrie.pop_prefixes` (cache.py:1569-1576).
  ///
  /// Returns `(prefix_len, value)` pairs in walk order. `prefix_len` is the
  /// number of tokens consumed before that value's node (mlx-lm's `i`, the
  /// loop index at which the value was found). The value at the *terminal*
  /// `tokens` node is **not** popped (the loop pops `current` *before*
  /// descending into `tokens[i]`, so it never visits the final node).
  pub fn pop_prefixes(&mut self, model: &M, tokens: &[i32]) -> Vec<(usize, V)> {
    let mut values = Vec::new();
    let Some(mut current) = self.trie.get_mut(model) else {
      return values;
    };
    for (i, &tok) in tokens.iter().enumerate() {
      if let Some(v) = current.value.take() {
        values.push((i, v));
      }
      // mlx-lm `current = current[tok]` would KeyError on a missing edge;
      // the recoverable shape stops the walk (callers pass a path proven
      // present by `search`, so this never short-circuits in practice).
      match current.children.get_mut(tok) {
        Some(next) => current = next,
        None => break,
      }
    }
    values
  }

  /// Longest-prefix / shortest-extension search — port of
  /// `mlx_lm.models.cache.PromptTrie.search` (cache.py:1578-1620), traced
  /// branch-for-branch.
  pub fn search(&self, model: &M, tokens: &[i32]) -> PromptTrieResult {
    // `if model not in self._trie:` -> all-None, common_prefix 0.
    let Some(root) = self.trie.get(model) else {
      return PromptTrieResult {
        exact: None,
        shorter: None,
        longer: None,
        common_prefix: 0,
      };
    };

    let mut current = root;

    // `if not tokens and "__value__" in current:` -> exact == [].
    if tokens.is_empty() && current.value.is_some() {
      return PromptTrieResult {
        exact: Some(Vec::new()),
        shorter: None,
        longer: None,
        common_prefix: 0,
      };
    }

    // Walk the tokens as far as we can. `last_index` tracks the last index
    // at which a value was seen (mlx-lm's `last_index`, initialized to -1
    // and only set when `"__value__" in current`); modeled as `Option`.
    let mut last_index: Option<usize> = None;
    let mut index: usize = 0;
    while index < tokens.len() {
      match current.children.get(tokens[index]) {
        Some(next) => {
          current = next;
          if current.value.is_some() {
            last_index = Some(index);
          }
          index += 1;
        }
        None => break,
      }
    }

    // `if last_index == len(tokens) - 1 >= 0:` -> exact == tokens.
    // (`len(tokens) - 1` would be -1 for empty tokens; the `>= 0` guards
    // it. Modeled directly: tokens non-empty AND last_index == len-1.)
    if !tokens.is_empty()
      && let Some(li) = last_index
      && li == tokens.len() - 1
    {
      return PromptTrieResult {
        exact: Some(tokens.to_vec()),
        shorter: None,
        longer: None,
        common_prefix: 0,
      };
    }

    // `shorter = tokens[: last_index + 1]` iff `last_index > 0`. Python's
    // `last_index > 0` is *strict* and on the int (initial -1): so a value
    // only at index 0 (`last_index == 0`) yields NO `shorter` — ported
    // faithfully as `li > 0` (a single-token prefix value is intentionally
    // not surfaced as `shorter`).
    let shorter = match last_index {
      Some(li) if li > 0 => Some(tokens[..li + 1].to_vec()),
      _ => None,
    };

    // `common_prefix = index`. `longer` only when `index > 0`: DFS from the
    // node the walk stopped at for the shortest value-bearing extension
    // (`best` = the shortest `extra`), then `tokens[:index] + best`.
    let common_prefix = index;
    let mut longer = None;
    if index > 0 {
      let mut best: Option<Vec<i32>> = None;
      // mlx-lm `stack = [(current, [])]`; pop, and if the node has a value
      // record `extra` when it is the shortest so far, else (only if it
      // could still beat `best`) push each child with `extra + [tok]`.
      let mut stack: Vec<(&TrieNode<V>, Vec<i32>)> = vec![(current, Vec::new())];
      while let Some((node, extra)) = stack.pop() {
        if node.value.is_some() {
          if best.as_ref().is_none_or(|b| extra.len() < b.len()) {
            best = Some(extra);
          }
        } else if best.as_ref().is_none_or(|b| extra.len() < b.len()) {
          // mlx-lm `for tok in current: stack.append(...)` — iterate
          // children in **insertion order** (Python dict order) and push
          // onto the LIFO `stack`, so the subsequent `stack.pop()`s yield
          // them *last-inserted-first*, exactly mlx-lm's traversal. With
          // `best` only replaced on a strictly shorter `extra`, an
          // equal-length `longer` tie deterministically resolves to the
          // last-inserted sibling's path — byte-for-byte mlx-lm.
          for (tok, child) in node.children.iter_insertion_order() {
            let mut e = extra.clone();
            e.push(tok);
            stack.push((child, e));
          }
        }
      }
      // mlx-lm unconditionally does `longer = tokens[:index] + best`. The
      // walk stopped at a node reachable from the root, and the loop above
      // explores its entire subtree; a node with no value-bearing
      // descendant is unreachable here (a leaf always carries a value in
      // this trie's usage — every inserted path terminates in a value, and
      // `pop`/`pop_prefixes` prune value-less leaves), so `best` is always
      // `Some` whenever `index > 0`, exactly matching mlx-lm (which would
      // `TypeError` on `tokens[:index] + None`). Guarded as `if let` so a
      // degenerate value-less subtree is a no-`longer` result, never a
      // panic.
      if let Some(best) = best {
        let mut l = tokens[..index].to_vec();
        l.extend(best);
        longer = Some(l);
      }
    }

    PromptTrieResult {
      exact: None,
      shorter,
      longer,
      common_prefix,
    }
  }
}

// ───────────────────────── LruPromptCache ─────────────────────────

/// One cached prompt state — port of `LRUPromptCache.CacheEntry`
/// (cache.py:1624-1628): the per-prompt `Vec<Box<dyn KvCache>>`, its total
/// byte size, and which conversational bucket it belongs to. (Exactly
/// mlx-lm's three fields — the `(model, tokens)` key is *not* stored on the
/// entry there either; eviction recovers it from the [`CacheOrder`] deque,
/// whose elements are the `(M, Vec<i32>)` keys.)
struct CacheEntry {
  prompt_cache: Vec<Box<dyn KvCache>>,
  nbytes: usize,
  cache_type: String,
}

/// The bucketed LRU ordering — port of `LRUPromptCache.CacheOrder`
/// (cache.py:1630-1657). One FIFO deque per conversational type; `pop`
/// implements mlx-lm's exact "balance the buckets" eviction policy.
struct CacheOrder<M> {
  ordering: Vec<String>,
  lrus: HashMap<String, VecDeque<(M, Vec<i32>)>>,
}

impl<M: Eq + Clone> CacheOrder<M> {
  /// Default ordering `["assistant", "user", "system"]` (mlx-lm
  /// `CacheOrder.__init__`).
  fn new() -> Self {
    let ordering: Vec<String> = ["assistant", "user", "system"]
      .iter()
      .map(|s| s.to_string())
      .collect();
    let lrus = ordering
      .iter()
      .map(|k| (k.clone(), VecDeque::new()))
      .collect();
    Self { ordering, lrus }
  }

  /// Total entries across all buckets (mlx-lm `__len__`).
  fn len(&self) -> usize {
    self.lrus.values().map(VecDeque::len).sum()
  }

  /// Append `(model, tokens)` to `cache_type`'s deque (mlx-lm `push`).
  fn push(&mut self, model: &M, tokens: &[i32], cache_type: &str) {
    if let Some(d) = self.lrus.get_mut(cache_type) {
      d.push_back((model.clone(), tokens.to_vec()));
    }
  }

  /// Remove the first matching `(model, tokens)` from whichever bucket has
  /// it, scanning buckets in `ordering` (mlx-lm `remove` — `break`s after
  /// the first successful `deque.remove`).
  fn remove(&mut self, model: &M, tokens: &[i32]) {
    for ct in &self.ordering {
      if let Some(d) = self.lrus.get_mut(ct)
        && let Some(pos) = d
          .iter()
          .position(|(m, t)| m == model && t.as_slice() == tokens)
      {
        d.remove(pos);
        break;
      }
    }
  }

  /// Evict and return the next `(model, tokens)` per mlx-lm's bucket-
  /// balancing policy (cache.py:1649-1657):
  ///
  /// ```text
  /// i = 0
  /// while i + 1 < len(ordering):
  ///     a, b = lru[ordering[i]], lru[ordering[i+1]]
  ///     if a and len(a) >= len(b): return a.popleft()
  ///     i += 1
  /// return b.popleft()
  /// ```
  ///
  /// Returns `None` only if every bucket is empty (mlx-lm assumes a
  /// non-empty cache — `popleft` on an empty deque would `IndexError`; the
  /// recoverable shape is `None`, and every caller guards `len() > limit`
  /// first so the cache is non-empty exactly as in mlx-lm).
  fn pop(&mut self) -> Option<(M, Vec<i32>)> {
    let mut i = 0;
    while i + 1 < self.ordering.len() {
      let len_a = self.lrus.get(&self.ordering[i]).map_or(0, VecDeque::len);
      let len_b = self
        .lrus
        .get(&self.ordering[i + 1])
        .map_or(0, VecDeque::len);
      if len_a > 0 && len_a >= len_b {
        return self
          .lrus
          .get_mut(&self.ordering[i])
          .and_then(VecDeque::pop_front);
      }
      i += 1;
    }
    // `return b.popleft()` — the last bucket (`ordering[i]` where now `i +
    // 1 == len`, i.e. the final element). For the default 3-bucket order
    // this is `ordering[2]` ("system"), exactly mlx-lm's terminal
    // `lru_b.popleft()`.
    self
      .ordering
      .last()
      .and_then(|k| self.lrus.get_mut(k))
      .and_then(VecDeque::pop_front)
  }

  /// Entry count in one bucket (mlx-lm `len(self._lrus[cache_type])`,
  /// used by `stats_by_type`).
  fn type_len(&self, cache_type: &str) -> usize {
    self.lrus.get(cache_type).map_or(0, VecDeque::len)
  }
}

/// Per-bucket statistics — port of one row of `LRUPromptCache.stats_by_type`
/// (cache.py:1756-1763).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheTypeStats {
  /// Number of cached sequences in this bucket.
  pub n_sequences: usize,
  /// Total bytes held by this bucket.
  pub n_bytes: usize,
}

/// mlx-lm's `1 << 63` "effectively unbounded" sentinel for the byte / count
/// caps (cache.py:1659, :1742-1743). On a 64-bit target `usize` is 64-bit so
/// `1 << 63` is representable; this is the documented faithful default.
pub const LRU_UNBOUNDED: usize = 1 << 63;

/// An LRU cache of whole prompt states keyed by `(model, token prefix)` —
/// port of `mlx_lm.models.cache.LRUPromptCache` (cache.py:1623-1763).
///
/// It combines a [`PromptTrie`] (longest-prefix lookup) with a bucketed
/// `CacheOrder` (eviction). `fetch_nearest_cache` returns a *copy* of the
/// best reusable state plus the suffix tokens still to be processed;
/// `insert_cache` stores a state, dropping now-redundant prefixes and
/// enforcing the size / byte caps.
pub struct LruPromptCache<M: Eq + Hash + Clone> {
  max_size: usize,
  max_bytes: usize,
  trie: PromptTrie<M, CacheEntry>,
  lru: CacheOrder<M>,
  n_bytes: usize,
  n_bytes_by_type: HashMap<String, usize>,
}

impl<M: Eq + Hash + Clone> LruPromptCache<M> {
  /// A new cache with at most `max_size` sequences and `max_bytes` bytes —
  /// mlx-lm `LRUPromptCache(max_size=10, max_bytes=1 << 63)`. Pass
  /// [`LRU_UNBOUNDED`] for "effectively unbounded".
  pub fn new(max_size: usize, max_bytes: usize) -> Self {
    let lru = CacheOrder::new();
    let n_bytes_by_type = lru.ordering.iter().map(|k| (k.clone(), 0usize)).collect();
    Self {
      max_size,
      max_bytes,
      trie: PromptTrie::new(),
      lru,
      n_bytes: 0,
      n_bytes_by_type,
    }
  }

  /// Number of cached sequences (mlx-lm `__len__`).
  pub fn len(&self) -> usize {
    self.lru.len()
  }

  /// Whether the cache holds no sequences.
  pub fn is_empty(&self) -> bool {
    self.lru.len() == 0
  }

  /// Total bytes held (mlx-lm `nbytes`).
  pub fn nbytes(&self) -> usize {
    self.n_bytes
  }

  /// Find the most reusable cached state for `(model, tokens)` — port of
  /// `LRUPromptCache.fetch_nearest_cache` (cache.py:1674-1694).
  ///
  /// Returns `(Some(copy_of_state), remaining_tokens)` when a usable
  /// prefix/exact/longer match exists (the state is deep-copied via
  /// [`KvCache::copy`] so the cached entry is never mutated; for the
  /// "longer" branch it is trimmed back to the shared prefix), or `(None,
  /// tokens)` when nothing reusable was found.
  #[allow(clippy::type_complexity)]
  pub fn fetch_nearest_cache(
    &self,
    model: &M,
    tokens: &[i32],
  ) -> Result<(Option<Vec<Box<dyn KvCache>>>, Vec<i32>)> {
    let result = self.trie.search(model, tokens);

    // `if result.exact is not None:` -> deep copy, no remaining tokens.
    // mlx-lm dereferences `self._trie.get(result.model, result.exact)`
    // directly (a `KeyError` if the invariant were broken). The invariant
    // (`search` only returns a path that exists *in this same, unmutated
    // trie*) genuinely holds, but a missing entry is treated as "no usable
    // match" rather than a panic — a strictly safer no-op on the (dead in
    // practice) invariant-violation path, never panicking on a fetch.
    if let Some(exact) = &result.exact
      && let Some(entry) = self.trie.get(model, exact)
    {
      let copy = copy_prompt_cache(&entry.prompt_cache)?;
      return Ok((Some(copy), Vec::new()));
    }

    let short_length = result.shorter.as_ref().map_or(0, Vec::len);

    // `if result.longer is not None and result.common_prefix >
    // short_length:`
    if let Some(longer) = &result.longer
      && result.common_prefix > short_length
      && let Some(entry) = self.trie.get(model, longer)
    {
      // `if can_trim_prompt_cache(cache_entry.prompt_cache):`
      if can_trim_prompt_cache(&entry.prompt_cache) {
        let mut cache = copy_prompt_cache(&entry.prompt_cache)?;
        // `prefix = min(len(tokens) - 1, result.common_prefix)`. mlx-lm's
        // `len(tokens) - 1`: `tokens` is non-empty whenever `longer` is
        // `Some` (an empty `tokens` only ever yields the `exact == []`
        // early return or all-None), so the `- 1` does not underflow;
        // `saturating_sub` keeps the degenerate empty case (unreachable
        // here) a 0, never a panic.
        let prefix = tokens.len().saturating_sub(1).min(result.common_prefix);
        // `num_to_trim = len(result.longer) - prefix`. `longer` extends
        // `tokens[:index]` and `prefix <= common_prefix == index <=
        // len(longer)`, so this is non-negative exactly as in mlx-lm;
        // `saturating_sub` is a panic guard only.
        let num_to_trim = longer.len().saturating_sub(prefix);
        trim_prompt_cache(&mut cache, num_to_trim)?;
        return Ok((Some(cache), tokens[prefix..].to_vec()));
      }
    }

    // `if short_length > 0:` (`short_length > 0` ⇒ `shorter` is `Some`).
    if short_length > 0
      && let Some(shorter) = &result.shorter
      && let Some(entry) = self.trie.get(model, shorter)
    {
      let copy = copy_prompt_cache(&entry.prompt_cache)?;
      return Ok((Some(copy), tokens[short_length..].to_vec()));
    }

    // `return None, tokens`
    Ok((None, tokens.to_vec()))
  }

  /// Store `prompt_cache` for `(model, tokens)` under `cache_type` — port
  /// of `LRUPromptCache.insert_cache` (cache.py:1696-1737).
  ///
  /// Updates the byte counters, replaces any prior entry at the same key,
  /// drops now-redundant *prefix* entries when the cache is trimmable, then
  /// evicts to satisfy the size / byte caps. `cache_type` defaults (mlx-lm
  /// keyword default) to `"assistant"` — call [`insert_cache_assistant`](
  /// LruPromptCache::insert_cache_assistant) for that.
  ///
  /// **Unknown `cache_type` ⇒ `Err`, no mutation (faithful to mlx-lm).**
  /// Authoritative mlx-lm indexes fixed per-type dicts —
  /// `self._n_bytes_by_type[cache_type] += ...` (cache.py:1711) and
  /// `self._lrus[cache_type].append(...)` (cache.py:1639) — so a
  /// `cache_type` outside the `CacheOrder` ordering raises `KeyError`
  /// *before* the entry is durably inserted (the trie `add` at
  /// cache.py:1712 is never reached). The Rust-idiomatic mirror of that
  /// fail-fast is an early `Err(Error::MissingKey)` validated **before any
  /// state mutation** — never silently dropping the bucket (which would
  /// leave a fetchable, untracked, un-evictable entry that
  /// `len`/`stats_by_type`/`trim_to`/the `max_bytes` loop cannot see and
  /// that bypasses `max_size`/`max_bytes`). This is fallible where mlx-lm
  /// is (it raises); a valid `cache_type` is infallible-equivalent.
  pub fn insert_cache(
    &mut self,
    model: &M,
    tokens: &[i32],
    prompt_cache: Vec<Box<dyn KvCache>>,
    cache_type: &str,
  ) -> Result<()> {
    // Validate `cache_type` BEFORE touching the trie / byte counters / lru
    // (mlx-lm raises `KeyError` at the first `self._n_bytes_by_type[...]`
    // / `self._lrus[...]` index, before `self._trie.add`). A bucket is
    // valid iff it is one of `CacheOrder`'s fixed ordering keys.
    if !self.lru.ordering.iter().any(|k| k == cache_type) {
      // The supported set is RUNTIME-derived (`self.lru.ordering`, not a
      // `&'static` list), so this uses `MissingKey` (runtime-keyed lookup
      // miss) rather than `UnknownEnumValue` (which requires a static
      // `supported` list).
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "LruPromptCache::add: cache_type (must be one of the configured CacheOrder buckets)",
        cache_type,
      )));
    }

    // `entry = CacheEntry(prompt_cache, sum(c.nbytes for c in
    // prompt_cache), cache_type)`
    let trimmable = can_trim_prompt_cache(&prompt_cache);
    let entry_nbytes: usize = prompt_cache.iter().map(|c| c.nbytes()).sum();
    let entry = CacheEntry {
      prompt_cache,
      nbytes: entry_nbytes,
      cache_type: cache_type.to_string(),
    };

    // `self._n_bytes += entry.nbytes; self._n_bytes_by_type[cache_type] +=
    // entry.nbytes; prev = self._trie.add(...)` (cache_type validated
    // above, so the byte-by-type counter is guaranteed present — no
    // silent `or_insert` that would create an untracked bucket).
    self.n_bytes += entry_nbytes;
    if let Some(b) = self.n_bytes_by_type.get_mut(cache_type) {
      *b += entry_nbytes;
    }
    let prev = self.trie.add(model, tokens, entry);

    // `if prev is not None:` subtract its bytes, remove its lru slot.
    if let Some(prev) = prev {
      self.n_bytes -= prev.nbytes;
      if let Some(b) = self.n_bytes_by_type.get_mut(&prev.cache_type) {
        *b -= prev.nbytes;
      }
      self.lru.remove(model, tokens);
    }
    self.lru.push(model, tokens, cache_type);

    // `if can_trim_prompt_cache(prompt_cache):` pop every shorter prefix
    // (they "just take space"). mlx-lm's loop body has a known upstream
    // quirk: it does `self._lru.remove(model, tokens[:prefix_len])` but
    // subtracts `entry.nbytes` where `entry` is the *prefix* value the
    // `for` rebound (`for prefix_len, entry in
    // self._trie.pop_prefixes(...)`) — so the byte math is correct (it uses
    // the popped prefix entry's bytes) and `tokens[:prefix_len]` is the
    // prefix key. Ported verbatim.
    if trimmable {
      for (prefix_len, prefix_entry) in self.trie.pop_prefixes(model, tokens) {
        self.n_bytes -= prefix_entry.nbytes;
        if let Some(b) = self.n_bytes_by_type.get_mut(&prefix_entry.cache_type) {
          *b -= prefix_entry.nbytes;
        }
        self.lru.remove(model, &tokens[..prefix_len]);
      }
    }

    // `if len(self._lru) > self.max_size:` evict ONE (mlx-lm uses an `if`,
    // not a `while`, here — a single insert adds one, so one eviction
    // restores the bound).
    if self.lru.len() > self.max_size
      && let Some((m, t)) = self.lru.pop()
      && let Some(e) = self.trie.pop(&m, &t)
    {
      self.n_bytes -= e.nbytes;
      if let Some(b) = self.n_bytes_by_type.get_mut(&e.cache_type) {
        *b -= e.nbytes;
      }
    }
    // `while self._n_bytes > self.max_bytes:` evict until under the byte
    // cap.
    while self.n_bytes > self.max_bytes {
      let Some((m, t)) = self.lru.pop() else { break };
      let Some(e) = self.trie.pop(&m, &t) else {
        break;
      };
      self.n_bytes -= e.nbytes;
      if let Some(b) = self.n_bytes_by_type.get_mut(&e.cache_type) {
        *b -= e.nbytes;
      }
    }
    Ok(())
  }

  /// [`insert_cache`](LruPromptCache::insert_cache) with mlx-lm's default
  /// `cache_type="assistant"` (always a valid bucket, so this never
  /// `Err`s on the type — it forwards the `Result` only for signature
  /// uniformity with [`insert_cache`](LruPromptCache::insert_cache)).
  pub fn insert_cache_assistant(
    &mut self,
    model: &M,
    tokens: &[i32],
    prompt_cache: Vec<Box<dyn KvCache>>,
  ) -> Result<()> {
    self.insert_cache(model, tokens, prompt_cache, "assistant")
  }

  /// Evict until at most `n_sequences` sequences **and** at most `n_bytes`
  /// bytes remain — port of `LRUPromptCache.trim_to` (cache.py:1739-1754).
  /// `None` means "no limit" (mlx-lm's `1 << 63`); a negative request is
  /// clamped to 0 (`max(0, ...)`, here just `usize`).
  pub fn trim_to(&mut self, n_sequences: Option<usize>, n_bytes: Option<usize>) {
    let n_sequences = n_sequences.unwrap_or(LRU_UNBOUNDED);
    let n_bytes = n_bytes.unwrap_or(LRU_UNBOUNDED);

    while self.lru.len() > n_sequences {
      let Some((m, t)) = self.lru.pop() else { break };
      let Some(e) = self.trie.pop(&m, &t) else {
        break;
      };
      self.n_bytes -= e.nbytes;
      if let Some(b) = self.n_bytes_by_type.get_mut(&e.cache_type) {
        *b -= e.nbytes;
      }
    }
    while self.n_bytes > n_bytes {
      let Some((m, t)) = self.lru.pop() else { break };
      let Some(e) = self.trie.pop(&m, &t) else {
        break;
      };
      self.n_bytes -= e.nbytes;
      if let Some(b) = self.n_bytes_by_type.get_mut(&e.cache_type) {
        *b -= e.nbytes;
      }
    }
  }

  /// Per-bucket statistics — port of `LRUPromptCache.stats_by_type`
  /// (cache.py:1756-1763), keyed by conversational type.
  pub fn stats_by_type(&self) -> HashMap<String, CacheTypeStats> {
    let mut result = HashMap::new();
    for ct in &self.lru.ordering {
      result.insert(
        ct.clone(),
        CacheTypeStats {
          n_sequences: self.lru.type_len(ct),
          n_bytes: self.n_bytes_by_type.get(ct).copied().unwrap_or(0),
        },
      );
    }
    result
  }
}

/// Deep-copy a `Vec<Box<dyn KvCache>>` (mlx-lm's `copy.deepcopy` of a prompt
/// cache, cache.py:1678/1684/1692). Each cache's own
/// [`KvCache::copy`] is the faithful deep copy; it is
/// fallible (the underlying [`Array::try_clone`](crate::Array::try_clone) is
/// fallible per #33), so a clone failure is propagated as an [`Error`], never
/// swallowed into a half-built cache.
///
/// [`Error`]: crate::Error
fn copy_prompt_cache(cache: &[Box<dyn KvCache>]) -> Result<Vec<Box<dyn KvCache>>> {
  cache.iter().map(|c| c.copy()).collect()
}

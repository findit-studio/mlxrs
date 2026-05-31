//! In-isolation, hand-traced [`ChatSession`] tests built on the crate's
//! deterministic [`crate::lm::model::MockModel`] fixture and the committed
//! `WordLevel` chat-template fixture tokenizer (`mlxrs/tests/fixtures`).
//!
//! `MockModel`'s argmax is its last vocab index, and `forward` advances
//! every cache layer by the token-window length — so a turn is fully
//! predictable (the reply is a run of the last-index token) and the cache
//! `offset()` is an exact, observable witness of cross-turn reuse.

use super::*;
use crate::{lm::model::MockModel, tokenizer::StreamingDetokenizer};

/// Load the committed fixture tokenizer (`<s> <unk> </s> hello world the
/// quick brown fox <think> </think>`, with a `<|role|>content` chat
/// template). Reachable from the in-crate `#[cfg(test)]` build.
fn fixture_tokenizer() -> Tokenizer {
  let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("fixtures");
  Tokenizer::from_path(&dir, None).expect("load fixture tokenizer")
}

/// A small, non-sliding cache config (matches `MockModel`'s flat KV state).
fn cache_config() -> CacheConfig {
  CacheConfig {
    num_hidden_layers: 2,
    sliding_window: None,
  }
}

/// Build a fresh session: `MockModel` (vocab 11, the fixture vocab size)
/// and the fixture tokenizer, with a small `max_tokens` for a quick
/// deterministic run.
fn session(max_tokens: usize) -> ChatSession {
  let cfg = GenConfig {
    max_tokens,
    ..Default::default()
  };
  ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .generate_params(cfg)
  .build()
  .expect("build")
}

#[test]
fn fresh_session_has_no_cache_until_first_turn() {
  // The Swift `currentCache()` observable: `nil` before generation.
  let mut s = session(4);
  assert!(!s.has_cache(), "fresh session: cache unrealised");
  assert!(s.current_cache().is_none());
  assert!(s.history().is_empty());

  let reply = s.respond("hello").expect("respond");
  assert!(!reply.is_empty(), "MockModel produces a non-empty reply");
  assert!(s.has_cache(), "cache realised after the first turn");
}

#[test]
fn multi_turn_reuses_cache_and_accumulates_history() {
  // The core contract: turn 2 reuses turn 1's cache (the cache offset
  // grows monotonically — turn 2 only prefills the *new* tokens, never
  // re-prefills turn 1), and both turns land in the history.
  let mut s = session(3);

  let _ = s.respond("hello world").expect("turn 1");
  // history: [user "hello world", assistant <reply>]
  assert_eq!(s.history().len(), 2);
  assert_eq!(s.history()[0].role, Role::User);
  assert_eq!(s.history()[1].role, Role::Assistant);

  let offset_after_turn_1 = s
    .current_cache()
    .expect("cache realised")
    .first()
    .expect(">=1 layer")
    .offset();
  assert!(offset_after_turn_1 > 0, "turn 1 advanced the cache");

  let _ = s.respond("the quick fox").expect("turn 2");
  // history: 4 messages now (two full turns).
  assert_eq!(s.history().len(), 4);
  assert_eq!(s.history()[2].role, Role::User);
  assert_eq!(s.history()[2].content(), "the quick fox");
  assert_eq!(s.history()[3].role, Role::Assistant);

  let offset_after_turn_2 = s
    .current_cache()
    .expect("cache realised")
    .first()
    .expect("layer")
    .offset();
  // Monotonic growth witnesses reuse: turn 2 extended turn 1's cache
  // rather than starting from a fresh (offset-0) cache.
  assert!(
    offset_after_turn_2 > offset_after_turn_1,
    "turn 2 reused + extended the cache (offset {offset_after_turn_1} -> {offset_after_turn_2})"
  );
}

#[test]
fn turn_two_prefills_only_the_new_suffix_not_the_whole_history() {
  // The core incremental-prefill contract. Turn 2's cache
  // offset must grow by exactly the *new* rendered suffix (the tokens not
  // already in the cache) plus the tokens generated this turn — NOT by the
  // whole turn-2 render (which re-includes turn 1). The bug being guarded:
  // feeding the full render onto the already-advanced cache, re-appending
  // the entire prior conversation's KV.
  let max_tokens = 3;
  let mut s = session(max_tokens);

  // Turn 1.
  let _ = s.respond("hello world").expect("turn 1");
  let off1 = s.current_cache().expect("cache realised")[0].offset();
  assert!(off1 > 0, "turn 1 advanced the cache");

  // Render turn 2's full prompt the same way `stream_respond_as` will —
  // the running history now holds turn 1, so this render re-includes it.
  let (prompt2, _) = s
    .build_turn_prompt("the quick fox", Role::User)
    .expect("render turn 2");
  let full_render2 = prompt2.len();

  // The suffix actually fed to `generate_step` is `prompt2[off1..]`: the
  // tokens beyond what the cache already encodes.
  assert!(
    full_render2 > off1,
    "turn-2 render ({full_render2}) extends past the cached prefix ({off1})"
  );
  let suffix_len = full_render2 - off1;

  // Turn 2.
  let _ = s.respond("the quick fox").expect("turn 2");
  let off2 = s.current_cache().expect("cache realised")[0].offset();

  // `generate_step` over a P-token prompt advances the cache by `P - 1`
  // (prefill) + `1` (first decode step) + `max_tokens - 1` (later decode
  // steps) = `P - 1 + max_tokens`. With incremental prefill `P` is the
  // SUFFIX length, so the growth is exactly this:
  let expected_growth = suffix_len - 1 + max_tokens;
  assert_eq!(
    off2 - off1,
    expected_growth,
    "turn 2 grew the cache by the new suffix ({suffix_len}) + generated \
       ({max_tokens}) only, not the whole render"
  );

  // The decisive anti-bug assertion: the growth is strictly LESS than a
  // full re-prefill of the turn-2 render would cost. If the session were
  // re-feeding the whole conversation, `off2 - off1` would be
  // `full_render2 - 1 + max_tokens` instead.
  assert!(
    off2 - off1 < full_render2,
    "turn 2 did NOT re-prefill the whole conversation (grew {}, full \
       render is {full_render2})",
    off2 - off1
  );
}

#[test]
fn instructions_change_forces_a_cache_rebuild_not_wrong_output() {
  // The divergence fallback. Changing `instructions` between
  // turns makes turn 2's render NOT a prefix-extension of turn 1's cached
  // tokens (a new leading system message shifts everything). The session
  // must detect this and rebuild the cache from scratch — feeding the full
  // turn-2 render — rather than feed a bogus "suffix" onto a stale cache.
  let max_tokens = 3;
  let mut s = session(max_tokens);

  let _ = s.respond("hello").expect("turn 1");
  let off1 = s.current_cache().expect("realised")[0].offset();

  // Change the system instructions — turn 2's render gains a leading
  // `<|system|>...` block, so it no longer extends turn 1's cached prefix.
  s.set_instructions(Some("hello world the quick".to_string()));
  let (prompt2, _) = s
    .build_turn_prompt("world", Role::User)
    .expect("render turn 2");
  let full_render2 = prompt2.len();

  let _ = s.respond("world").expect("turn 2");
  let off2 = s.current_cache().expect("realised")[0].offset();

  // A rebuild re-prefills the WHOLE turn-2 render: growth from offset 0 is
  // `full_render2 - 1 + max_tokens`. (Incremental prefill is impossible
  // here — the divergence is real — so the larger cost is correct.)
  assert_eq!(
    off2,
    full_render2 - 1 + max_tokens,
    "instructions change rebuilt the cache from scratch (offset reset, \
       full render re-prefilled)"
  );
  // The rebuilt cache offset bears no relation to turn 1's — it did not
  // continue the stale cache.
  assert!(
    off2 != off1 + (full_render2 - 1 + max_tokens),
    "the stale cache was discarded, not extended"
  );
}

#[test]
fn every_cache_layer_advances_in_lockstep() {
  // `make_prompt_cache` builds one cache per layer; a turn must advance
  // all of them equally (the model drives every layer each `forward`).
  let mut s = session(3);
  let _ = s.respond("hello").expect("turn");
  let cache = s.current_cache().expect("realised");
  assert_eq!(cache.len(), 2, "one cache per decoder layer");
  let off0 = cache[0].offset();
  assert!(off0 > 0);
  assert!(
    cache.iter().all(|c| c.offset() == off0),
    "all layers advance in lockstep"
  );
}

#[test]
fn streaming_and_non_streaming_respond_are_consistent() {
  // `respond` is documented as "drain `stream_respond`, concatenate the
  // text" — two sessions given the identical turn must agree.
  let mut a = session(5);
  let non_streaming = a.respond("hello world").expect("non-streaming");

  let mut b = session(5);
  let mut streamed = String::new();
  {
    let stream = b.stream_respond("hello world").expect("stream");
    for resp in stream {
      streamed.push_str(&resp.expect("stream step").text);
    }
  }
  assert_eq!(
    non_streaming, streamed,
    "streaming and non-streaming respond produce the same text"
  );
  // Both sessions recorded the same history shape.
  assert_eq!(a.history().len(), b.history().len());
  assert_eq!(a.history()[1].content(), b.history()[1].content);
}

#[test]
fn streaming_reply_matches_recorded_history() {
  // The text yielded by the stream must equal the assistant message the
  // `Drop` write-back appends to the history.
  let mut s = session(4);
  let mut streamed = String::new();
  {
    let stream = s.stream_respond("hello").expect("stream");
    for resp in stream {
      streamed.push_str(&resp.expect("step").text);
    }
  }
  assert_eq!(s.history().len(), 2);
  assert_eq!(
    s.history()[1].content(),
    streamed,
    "the recorded assistant turn equals the streamed text"
  );
}

#[test]
fn finish_reason_is_length_when_max_tokens_reached() {
  // `MockModel`'s argmax (last vocab index, 10 = `</think>`) is never an
  // eos id (eos = `</s>` = 2), so generation always runs to `max_tokens`
  // — the final response must report `finish_reason = "length"`.
  let mut s = session(3);
  let mut reasons = Vec::new();
  {
    let stream = s.stream_respond("hello").expect("stream");
    for resp in stream {
      reasons.push(resp.expect("step").finish_reason);
    }
  }
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Length));
  // Exactly one terminal reason; the rest are `None`.
  assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
  assert_eq!(reasons.len(), 3, "max_tokens responses produced");
}

#[test]
fn clear_drops_cache_and_history_keeps_instructions() {
  // The Swift `clear()`: cache + history reset, instructions preserved.
  let mut s = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .instructions("be terse")
  .generate_params(GenConfig {
    max_tokens: 3,
    ..Default::default()
  })
  .build()
  .expect("build");

  let _ = s.respond("hello").expect("turn");
  assert!(s.has_cache());
  assert!(!s.history().is_empty());

  s.clear();
  assert!(!s.has_cache(), "clear() drops the cache");
  assert!(s.history().is_empty(), "clear() drops the history");
  assert_eq!(
    s.instructions(),
    Some("be terse"),
    "clear() preserves instructions"
  );

  // A turn after clear starts from a fresh cache (offset resets).
  let _ = s.respond("world").expect("post-clear turn");
  assert!(s.has_cache());
  assert_eq!(s.history().len(), 2, "history restarts after clear");
}

#[test]
fn early_drop_of_stream_still_records_partial_turn() {
  // The Swift interrupt semantics (`testChatSessionAsyncInterrupt`): an
  // abandoned stream still commits its (partial) reply + advanced cache.
  let mut s = session(10);
  let mut streamed = String::new();
  {
    let mut stream = s.stream_respond("hello").expect("stream");
    // consume exactly one token, then drop the stream
    let first = stream.next().expect("first token").expect("ok");
    streamed.push_str(&first.text);
    assert!(first.finish_reason.is_none() || first.finish_reason.is_some());
  }
  // Drop committed: cache realised + a (partial) assistant turn recorded.
  assert!(s.has_cache(), "interrupted turn still realised the cache");
  assert_eq!(s.history().len(), 2, "interrupted turn still recorded");
  assert_eq!(s.history()[1].role, Role::Assistant);

  // `commit()` finalizes the detokenizer on an interrupted
  // standard stream, so the recorded reply is token-complete — it includes
  // every streamed segment (and would include any tail a BPE/SPM detok
  // withholds until `finalize()`). The recorded text must therefore start
  // with everything the stream yielded.
  assert!(
    s.history()[1].content().starts_with(&streamed),
    "recorded reply ({:?}) includes the streamed text ({streamed:?})",
    s.history()[1].content()
  );

  // The session is still usable for a follow-up turn (cache reused).
  let off_before = s.current_cache().unwrap()[0].offset();
  let _ = s.respond("world").expect("follow-up turn");
  let off_after = s.current_cache().unwrap()[0].offset();
  assert!(off_after > off_before, "follow-up reused the cache");
}

#[test]
fn early_drop_then_followup_does_incremental_prefill() {
  // After an early-dropped standard
  // stream, the next turn must still do *incremental* prefill — feeding
  // only the new suffix. That only works if the partial reply recorded by
  // `commit()` (after `finalize()`) round-trips to exactly the tokens the
  // interrupted turn left in the cache; a token-incomplete reply would
  // make turn 2's render diverge and force a (slower, but still correct)
  // full rebuild. Asserting incremental prefill here transitively proves
  // the committed history matches the committed cache state.
  let max_tokens = 8;
  let mut s = session(max_tokens);
  {
    let mut stream = s.stream_respond("hello world").expect("stream");
    // Consume a few tokens, then abandon the stream mid-generation.
    let _ = stream.next().expect("token 1").expect("ok");
    let _ = stream.next().expect("token 2").expect("ok");
  }
  let off1 = s.current_cache().expect("realised")[0].offset();
  assert!(off1 > 0, "interrupted turn advanced the cache");

  // Render turn 2 the way `stream_respond_as` will.
  let (prompt2, _) = s
    .build_turn_prompt("the quick fox", Role::User)
    .expect("render turn 2");
  let full_render2 = prompt2.len();
  assert!(
    full_render2 > off1,
    "turn-2 render extends past the interrupted cache"
  );
  let suffix_len = full_render2 - off1;

  let _ = s.respond("the quick fox").expect("turn 2");
  let off2 = s.current_cache().expect("realised")[0].offset();

  // Incremental prefill: growth == new suffix + generated, NOT the whole
  // render. If the committed partial reply did not match the cache, turn 2
  // would diverge and rebuild (growth `full_render2 - 1 + max_tokens`).
  assert_eq!(
    off2 - off1,
    suffix_len - 1 + max_tokens,
    "follow-up after an early drop still prefilled only the new suffix"
  );
  assert!(
    off2 - off1 < full_render2,
    "follow-up did not re-prefill the whole conversation"
  );
}

#[test]
fn history_seeded_session_replays_then_realises_cache() {
  // The Swift `history:` initializer: cache `nil` until the first turn,
  // then the restored messages are folded into the live history.
  let seeded = vec![ChatMessage::user("hello"), ChatMessage::assistant("world")];
  let mut s = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .history(seeded)
  .generate_params(GenConfig {
    max_tokens: 3,
    ..Default::default()
  })
  .build()
  .expect("build");

  // `.history` state behaves like `.empty`: no cache before generation.
  assert!(!s.has_cache(), "history-seeded: cache unrealised pre-turn");
  assert!(s.history().is_empty(), "live history empty pre-turn");

  let _ = s.respond("the fox").expect("first turn");
  assert!(s.has_cache(), "cache realised after the first turn");
  // live history: 2 replayed + (user "the fox" + assistant reply) = 4.
  assert_eq!(s.history().len(), 4, "replayed history folded in");
  assert_eq!(s.history()[0].content(), "hello");
  assert_eq!(s.history()[1].content(), "world");
  assert_eq!(s.history()[2].content(), "the fox");
  assert_eq!(s.history()[3].role, Role::Assistant);
}

#[test]
fn save_cache_errors_before_any_generation() {
  // The Swift `ChatSessionError.noCacheAvailable`.
  let s = session(3);
  let path = std::env::temp_dir().join("mlxrs-l11-chat-session-nocache.safetensors");
  let err = s.save_cache(&path).expect_err("no cache yet");
  // Surfaced as a Backend error carrying the ChatSessionError message.
  assert!(
    format!("{err}").contains("no KV cache"),
    "noCacheAvailable surfaced: {err}"
  );
}

#[test]
fn instructions_are_rendered_into_the_prompt() {
  // A session with instructions must prepend a system message: the
  // rendered prompt differs from an instruction-free session's, and the
  // cache offset (== prompt length on turn 1) is therefore larger.
  let with = {
    let mut s = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .instructions("hello world the quick brown fox")
    .generate_params(GenConfig {
      max_tokens: 1,
      ..Default::default()
    })
    .build()
    .expect("build");
    let _ = s.respond("hello").expect("turn");
    s.current_cache().unwrap()[0].offset()
  };
  let without = {
    let mut s = session(1);
    let _ = s.respond("hello").expect("turn");
    s.current_cache().unwrap()[0].offset()
  };
  assert!(
    with > without,
    "the system instructions lengthened the prompt ({without} -> {with})"
  );
}

#[test]
fn set_instructions_and_generate_params_accessors() {
  // The Swift public `instructions` / `generateParameters` getters+setters.
  let mut s = session(3);
  assert!(s.instructions().is_none());
  s.set_instructions(Some("be brief".to_string()));
  assert_eq!(s.instructions(), Some("be brief"));
  s.set_instructions(None);
  assert!(s.instructions().is_none());

  assert_eq!(s.generate_params().max_tokens, 3);
  s.generate_params_mut().max_tokens = 7;
  assert_eq!(s.generate_params().max_tokens, 7);
}

#[test]
fn speculative_session_runs_multi_turn_and_accumulates_history() {
  // The optional speculative-decoding path (the Swift
  // `SpeculativeDecodingConfig`). A `MockModel` self-draft (the same
  // deterministic model as target and draft) accepts every proposed
  // token, so the turn completes; the session must accumulate the history
  // and stay usable for a second turn.
  let mut s = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .speculative(SpeculativeDecodingConfig::new(
    Rc::new(MockModel::new(11)),
    cache_config(),
  ))
  .generate_params(GenConfig {
    max_tokens: 4,
    ..Default::default()
  })
  .build()
  .expect("build");

  assert!(
    !s.has_cache(),
    "speculative session: cache unrealised pre-turn"
  );
  let reply1 = s.respond("hello").expect("speculative turn 1");
  assert!(!reply1.is_empty(), "speculative decoding produced a reply");
  assert_eq!(s.history().len(), 2);

  // A second turn still works (the speculative path rebuilds its cache
  // each turn — a documented divergence — but the history is correct).
  let reply2 = s.respond("world").expect("speculative turn 2");
  assert!(!reply2.is_empty());
  assert_eq!(s.history().len(), 4);
  assert_eq!(s.history()[2].content(), "world");
  assert_eq!(s.history()[3].role, Role::Assistant);
}

#[test]
fn speculative_session_does_not_expose_a_saveable_cache() {
  // `speculative_stream_generate` consumes its KV caches and
  // does not return them. A speculative turn must NOT present a freshly
  // rebuilt (offset-0) cache as the current/saveable cache — that cache
  // does not encode the conversation, so `save_cache` would persist a
  // cache that cannot restore the session. `has_cache()` stays `false`
  // even after a turn, and `save_cache` returns a speculative-specific
  // error (NOT `noCacheAvailable` — the session HAS run).
  let mut s = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .speculative(SpeculativeDecodingConfig::new(
    Rc::new(MockModel::new(11)),
    cache_config(),
  ))
  .generate_params(GenConfig {
    max_tokens: 3,
    ..Default::default()
  })
  .build()
  .expect("build");

  let _ = s.respond("hello").expect("speculative turn");
  // History accumulated, the turn ran — but no realised cache.
  assert_eq!(s.history().len(), 2, "the turn was still recorded");
  assert!(
    !s.has_cache(),
    "a speculative session never exposes a realised cache"
  );
  assert!(
    s.current_cache().is_none(),
    "no current cache for a speculative session"
  );

  // save_cache fails with the speculative-specific error, distinct from
  // the pre-generation `noCacheAvailable`.
  let path = std::env::temp_dir().join("mlxrs-l11-chat-session-spec-nocache.safetensors");
  let err = s
    .save_cache(&path)
    .expect_err("speculative cache not saveable");
  let msg = format!("{err}");
  assert!(
    msg.contains("speculative"),
    "speculative-specific save error surfaced: {msg}"
  );
  assert!(
    !msg.contains("call respond"),
    "not the pre-generation noCacheAvailable error: {msg}"
  );
  assert!(
    !path.exists(),
    "no cache file was written for the speculative session"
  );
}

/// Build a per-layer KV cache pre-advanced to exactly `n_tokens` — the
/// stand-in for a `ChatSessionBuilder::cache`-restored prefix whose token
/// ids the builder was never given (`MockModel::forward` advances every
/// layer by the token-window length). `n_tokens` must be `> 0`.
fn prefilled_opaque_cache(n_tokens: usize) -> Vec<Box<dyn KvCache>> {
  assert!(n_tokens > 0, "an opaque cache must have a non-empty prefix");
  let model = MockModel::new(11);
  let mut cache = make_prompt_cache(&cache_config());
  let window: Vec<i32> = (0..n_tokens as i32).map(|i| i % 11).collect();
  let arr = crate::array::Array::from_slice::<i32>(&window, &(1usize, n_tokens))
    .expect("opaque-prefill token window");
  let _ = model.forward(&arr, &mut cache).expect("opaque prefill");
  assert_eq!(cache[0].offset(), n_tokens, "opaque cache pre-advanced");
  cache
}

/// Build a `ChatSession` restored from a `prefilled_opaque_cache`.
fn cache_restored_session(opaque_len: usize, max_tokens: usize) -> ChatSession {
  ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .generate_params(GenConfig {
    max_tokens,
    ..Default::default()
  })
  .cache(prefilled_opaque_cache(opaque_len))
  .build()
  .expect("build")
}

#[test]
fn cache_restore_short_prompt_reuses_opaque_prefix_not_rebuilds() {
  // The documented `ChatSessionBuilder::cache` prefix-caching
  // path. The restored cache holds an
  // OPAQUE prefix of `opaque_len` tokens whose ids the session never knew;
  // a turn's render NEVER re-renders that prefix. A *short* first prompt
  // (rendered shorter than `opaque_len`) must still REUSE the restored
  // cache — feeding the WHOLE new render as the suffix that continues the
  // opaque prefix — NOT rebuild from an empty cache (which would silently
  // discard the restored context).
  let max_tokens = 3;
  // `opaque_len` chosen strictly between the short and long renders below,
  // so the short render is shorter than the opaque prefix (the exact case
  // the buggy `prompt_ids.len() > opaque_len` guard rebuilt-from-empty).
  let opaque_len = 10;

  // Measure the short render the way `stream_respond_as` will (a fresh,
  // history-free session renders identically — `.cache()` adds no history).
  let p_short = session(max_tokens)
    .build_turn_prompt("hello", Role::User)
    .expect("render short prompt")
    .0
    .len();
  assert!(
    p_short < opaque_len,
    "the short render ({p_short}) must be shorter than the opaque prefix \
       ({opaque_len}) to exercise the rebuilt-from-empty bug"
  );

  let mut s = cache_restored_session(opaque_len, max_tokens);
  // A `cache:`-restored session reports a realised cache immediately.
  assert!(s.has_cache(), "cache-restored session: cache realised");

  let _ = s.respond("hello").expect("first turn over restored cache");
  let off = s.current_cache().expect("cache realised")[0].offset();

  // The restored cache continued from `opaque_len`: the WHOLE short render
  // was fed as the suffix, then `max_tokens` decode steps ran. Offset =
  // opaque + (P - 1 prefill) + max_tokens. The `-1` is the final sampled
  // token (sampled, never fed back).
  assert_eq!(
    off,
    opaque_len + p_short - 1 + max_tokens,
    "the full new render ({p_short}) was fed onto the opaque prefix \
       ({opaque_len}); offset = opaque + P - 1 + generated"
  );
  // Decisive anti-bug assertion: a rebuild-from-empty would give
  // `p_short - 1 + max_tokens` with NO `opaque_len` term — strictly less
  // than `opaque_len`. The offset exceeding `opaque_len` witnesses that
  // the restored prefix was kept, not discarded.
  assert!(
    off > opaque_len + max_tokens,
    "the restored opaque prefix was REUSED, not rebuilt-from-empty \
       (offset {off} retains the opaque {opaque_len} tokens)"
  );

  // A second turn must do incremental prefill against the now-`known`
  // region (turn-1 render + reply), proving `commit()` recorded `known`
  // correctly relative to `offset - opaque_len`.
  let (prompt2, _) = s
    .build_turn_prompt("world", Role::User)
    .expect("render turn 2");
  let full_render2 = prompt2.len();
  let _ = s.respond("world").expect("second turn");
  let off2 = s.current_cache().expect("realised")[0].offset();
  // Reuse (not rebuild): grew by only the new suffix, strictly less than a
  // full re-prefill of the turn-2 render.
  assert!(
    off2 > off && off2 - off < full_render2,
    "turn 2 reused the cache (grew {} < full render {full_render2})",
    off2 - off
  );
}

#[test]
fn cache_restore_long_prompt_feeds_full_new_prompt_no_dropped_tokens() {
  // The other half of the `cache:` path. When the
  // first render is LONGER than the opaque prefix, feeding
  // `prompt_ids[opaque_len..]` would drop the first `opaque_len` tokens of
  // the ACTUAL new prompt. Instead, feed the ENTIRE new render as the
  // suffix continuing the opaque prefix.
  let max_tokens = 4;
  let opaque_len = 10;

  let p_long = session(max_tokens)
    .build_turn_prompt("hello world the quick brown fox", Role::User)
    .expect("render long prompt")
    .0
    .len();
  assert!(
    p_long > opaque_len,
    "the long render ({p_long}) must exceed the opaque prefix ({opaque_len}) \
       to exercise the dropped-first-tokens bug"
  );

  let mut s = cache_restored_session(opaque_len, max_tokens);
  let _ = s
    .respond("hello world the quick brown fox")
    .expect("first turn over restored cache");
  let off = s.current_cache().expect("cache realised")[0].offset();

  // The FULL `p_long`-token render was fed onto the opaque prefix — NOT
  // `prompt_ids[opaque_len..]`. Offset = opaque + (p_long - 1) + max_tokens.
  assert_eq!(
    off,
    opaque_len + p_long - 1 + max_tokens,
    "the FULL new render ({p_long} tokens) was fed; no first-{opaque_len} \
       tokens dropped"
  );
  // The dropped-tokens bug would have fed only `p_long - opaque_len` tokens
  // ⇒ offset `opaque_len + (p_long - opaque_len) - 1 + max_tokens` =
  // `p_long - 1 + max_tokens`. The real offset is larger by exactly
  // `opaque_len` — the witness that no leading tokens were dropped.
  assert_eq!(
    off - (p_long - 1 + max_tokens),
    opaque_len,
    "offset retains the full opaque prefix; the dropped-tokens bug would \
       lose exactly {opaque_len} tokens"
  );
}

/// Build a temp-dir tokenizer whose `decoder` is `ByteLevel` (so
/// `Tokenizer::detokenizer()` yields a real `BpeStreamingDetokenizer`) and
/// whose vocab includes the token `"â"` — a single GPT-2 byte-encoded char
/// for raw byte `0xE2`, a UTF-8 *lead* byte. A run of `"â"` tokens decodes
/// to an incomplete UTF-8 sequence (`"\u{fffd}…"`), which the BPE
/// detokenizer **withholds in its `unflushed` buffer** — released only by
/// `finalize()`. Returns `(tokenizer, â_id, vocab_size)`.
fn bpe_withholding_tokenizer() -> (Tokenizer, u32, usize) {
  // The committed WordLevel fixture vocab (ids 0..=10) plus `"â"` at 11.
  let a_id = 11u32;
  let tokenizer_json = json!({
    "version": "1.0",
    "truncation": Value::Null,
    "padding": Value::Null,
    "added_tokens": [
      { "id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": true },
      { "id": 1, "content": "<s>", "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": true },
      { "id": 2, "content": "</s>", "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": true }
    ],
    "normalizer": Value::Null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": Value::Null,
    // The ByteLevel decoder is what `infer_detokenizer_class` keys on to
    // pick the BPE streaming detokenizer.
    "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                 "use_regex": true },
    "model": {
      "type": "WordLevel",
      "vocab": {
        "<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "the": 5,
        "quick": 6, "brown": 7, "fox": 8, "<think>": 9, "</think>": 10,
        "â": a_id
      },
      "unk_token": "<unk>"
    }
  });
  let config_json = json!({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "clean_up_tokenization_spaces": false,
    "chat_template":
      "{{ bos_token }}{% for m in messages %}{{ '<|' + m['role'] + '|>' }}\
       {{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
  });

  // Multiple `#[test]` fns call this fixture concurrently (cargo runs the
  // lib-test binary multi-threaded by default); a shared `(pid)`-only dir
  // races on `remove_dir_all` + `write` between parallel callers. Append a
  // per-call atomic counter so each caller gets a unique dir and the race
  // is impossible — pre-existing flake surfaced by `cargo hack test
  // --each-feature` parallelism (see PR #256 cleanup follow-up).
  use std::sync::atomic::{AtomicU64, Ordering};
  static SEQ: AtomicU64 = AtomicU64::new(0);
  let seq = SEQ.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-l11-bpe-withhold-{}-{}",
    std::process::id(),
    seq
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).expect("temp tokenizer dir");
  std::fs::write(
    dir.join("tokenizer.json"),
    serde_json::to_string(&tokenizer_json).expect("serialize tokenizer.json"),
  )
  .expect("write tokenizer.json");
  std::fs::write(
    dir.join("tokenizer_config.json"),
    serde_json::to_string(&config_json).expect("serialize tokenizer_config.json"),
  )
  .expect("write tokenizer_config.json");

  let tok = Tokenizer::from_path(&dir, None).expect("load BPE-decoder tokenizer");
  // The vocab is 0..=11 (12 ids).
  (tok, a_id, 12)
}

#[test]
fn speculative_interrupted_stream_flushes_detokenizer_tail() {
  // An interrupted speculative turn must record token-complete
  // text. The speculative driver's streaming detokenizer is finalized only
  // on eos / `max_tokens`; a `ChatResponseStream` dropped mid-stream must
  // flush that detokenizer's withheld tail in `commit()`, or the recorded
  // assistant message loses the tail of the last produced token and the
  // next speculative turn rebuilds from a truncated history.
  let (tok, a_id, vocab) = bpe_withholding_tokenizer();
  // A `MockModel` whose argmax is `â` (id 11): every produced token decodes
  // to byte 0xE2, so the BPE detok withholds the whole reply in `unflushed`
  // until `finalize()` (`last_segment()` sees an empty `text` meanwhile).
  let mut canned = vec![0.0_f32; vocab];
  canned[a_id as usize] = 10.0;
  let target = MockModel {
    canned: canned.clone(),
    n_kv_heads: 1,
    head_dim: 2,
  };
  let draft = MockModel {
    canned,
    n_kv_heads: 1,
    head_dim: 2,
  };

  let mut s = ChatSession::builder(Box::new(target), tok, cache_config())
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(draft),
      cache_config(),
    ))
    .generate_params(GenConfig {
      max_tokens: 12,
      ..Default::default()
    })
    .build()
    .expect("build");

  // Drain only a few tokens, then drop the stream mid-generation.
  let mut produced_tokens: Vec<u32> = Vec::new();
  let mut streamed = String::new();
  {
    let mut stream = s.stream_respond("hello").expect("speculative stream");
    for _ in 0..3 {
      let r = stream.next().expect("token").expect("ok");
      produced_tokens.push(r.token);
      streamed.push_str(&r.text);
    }
    // `stream` dropped here mid-generation (finish_reason never seen).
  }
  assert_eq!(produced_tokens.len(), 3, "drained 3 tokens before drop");
  assert!(
    produced_tokens.iter().all(|&t| t == a_id),
    "the mock samples the withheld `â` token every step"
  );

  // The committed assistant turn.
  assert_eq!(s.history().len(), 2, "interrupted turn still recorded");
  let recorded = s.history()[1].content();

  // Token-complete oracle: feed the SAME produced tokens into an
  // independent BPE detokenizer and `finalize()` — the committed history
  // text must equal that full detokenization. Without flushing the
  // detokenizer tail in `commit()` the recorded text would be MISSING the
  // withheld tail (the BPE detok buffers every `â` in `unflushed` until
  // `finalize()`), so `recorded` would be the empty mid-stream `text`
  // instead.
  let reference = {
    let mut d = crate::tokenizer::BpeStreamingDetokenizer::new(
      // `(token_string, id)` vocab pairs — only `â` is produced, but a
      // faithful detok is built over the full vocab.
      vec![("â".to_string(), a_id)],
      false,
    );
    for &t in &produced_tokens {
      d.add_token(t);
    }
    d.finalize();
    d.last_segment()
  };
  assert_eq!(
    *recorded, reference,
    "the interrupted speculative turn recorded token-complete text \
       (detokenizer tail flushed): recorded {recorded:?} == finalized {reference:?}"
  );
  // The withheld tail is non-empty here, so the fix is load-bearing: the
  // recorded text is strictly longer than what the stream yielded
  // mid-flight (every mid-stream `last_segment()` returned "").
  assert!(
    !reference.is_empty() && recorded.len() > streamed.len(),
    "the BPE detok genuinely withheld a tail that `commit()` flushed \
       (streamed {streamed:?}, recorded {recorded:?})"
  );
}

/// A model that delegates every call to `inner` for the first `ok_calls`
/// `forward` invocations and then returns a [`Error::InvariantViolation`] on every
/// later one. Used to drive a [`ChatSession`] stream into the `Err` branch
/// of [`Iterator::next`] partway through generation so the
/// detokenizer-not-finalized-on-an-error-terminated-stream case is exercised.
///
/// `forward` is `&self`, so the call counter sits behind a [`Cell`]
/// ([`MockModel`] is single-thread). The optional `forward_embeddings` hook
/// just delegates — the LM path only uses `forward`.
struct ErringAfterModel {
  inner: MockModel,
  ok_calls: std::cell::Cell<usize>,
}

impl ErringAfterModel {
  fn new(inner: MockModel, ok_calls: usize) -> Self {
    Self {
      inner,
      ok_calls: std::cell::Cell::new(ok_calls),
    }
  }
}

impl Model for ErringAfterModel {
  fn forward(
    &self,
    tokens: &crate::array::Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<crate::array::Array> {
    let remaining = self.ok_calls.get();
    if remaining == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "ErringAfterModel::forward",
        "budget exhausted (test fixture)",
      )));
    }
    self.ok_calls.set(remaining - 1);
    self.inner.forward(tokens, cache)
  }

  fn forward_embeddings(
    &self,
    embeddings: &crate::array::Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<crate::array::Array> {
    self.inner.forward_embeddings(embeddings, cache)
  }
}

#[test]
fn build_rejects_cache_plus_speculative_combination() {
  // The `.cache(restored).speculative(..)` builder
  // combination MUST be rejected at `build()` time, with no session
  // constructed. The opaque cache cannot be re-rendered from history (its
  // token ids are unknown to the session), and `speculative_stream_generate`
  // consumes its KV caches and does not return them — so the first
  // speculative turn would use the restored prefix and the next turn would
  // silently fall back to a fresh offset-0 cache, losing the prefix without
  // warning. We reject up-front with an actionable message pointing at
  // either workaround (drop one of `.cache(..)` / `.speculative(..)`).
  let restored = prefilled_opaque_cache(8);
  let res = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .cache(restored)
  .speculative(SpeculativeDecodingConfig::new(
    Rc::new(MockModel::new(11)),
    cache_config(),
  ))
  .build();

  let err = match res {
    Ok(_) => panic!("cache + speculative must be rejected at build()"),
    Err(e) => e,
  };
  let msg = format!("{err}");
  // Decisive: the error names the unsupported combination AND points at
  // BOTH alternatives, so the caller can act on it without reading source.
  assert!(
    msg.contains("speculative") && msg.contains("cache"),
    "error names the rejected combination: {msg}"
  );
  assert!(
    msg.contains(".cache") && msg.contains(".speculative"),
    "error points at both workarounds: {msg}"
  );

  // Reverse builder order: `.speculative(..)` first then `.cache(..)` — the
  // rejection is order-independent (both fields are set at `build()` time).
  let restored2 = prefilled_opaque_cache(8);
  let res2 = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .speculative(SpeculativeDecodingConfig::new(
    Rc::new(MockModel::new(11)),
    cache_config(),
  ))
  .cache(restored2)
  .build();
  assert!(
    res2.is_err(),
    "rejection is independent of builder-method order"
  );

  // Sanity-check the same combination minus `.cache(..)` builds fine — the
  // rejection is precisely scoped to the unsupported pair, not a regression
  // of the lone `.speculative(..)` path.
  let ok = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .speculative(SpeculativeDecodingConfig::new(
    Rc::new(MockModel::new(11)),
    cache_config(),
  ))
  .build();
  assert!(
    ok.is_ok(),
    ".speculative(..) alone still builds (only the cache+speculative combo is unsupported)"
  );
}

#[test]
fn standard_error_terminated_stream_flushes_detokenizer_tail() {
  // The error-terminated case on the STANDARD path. An
  // error-terminated stream (the `Generator` yields `Err`) would skip
  // `commit()`'s finalize if the gate were `!self.finished`, because
  // `next()` sets `finished = true` on BOTH natural termination AND `Err`.
  // Splitting natural completion (which already finalized) from
  // error/early-drop termination (which has NOT) via `detok_finalized` makes
  // an `Err`-terminated BPE/SPM stream flush the withheld tail and
  // record token-complete history.
  let (tok, a_id, vocab) = bpe_withholding_tokenizer();
  // Argmax samples `â` (id 11) every step — every produced token decodes to
  // byte 0xE2, so the BPE detok withholds the whole reply in `unflushed`
  // until `finalize()` (mid-stream `last_segment()` returns "").
  let mut canned = vec![0.0_f32; vocab];
  canned[a_id as usize] = 10.0;
  let inner = MockModel {
    canned,
    n_kv_heads: 1,
    head_dim: 2,
  };
  // Allow `ok_steps` successful `forward` calls (1 prefill + decode steps),
  // then return Err on every later one — so the stream's standard generator
  // yields a few `Ok(GenStep)`s and then an `Err`, fusing with
  // `finished = true` but `detok_finalized = false`. If `commit()` skipped
  // the flush, the assistant message would be empty (every mid-stream
  // segment was "").
  let ok_steps: usize = 4;
  let model = ErringAfterModel::new(inner, ok_steps);

  let mut s = ChatSession::builder(Box::new(model), tok, cache_config())
    .generate_params(GenConfig {
      // `max_tokens` set high so the error fires before the natural-finish
      // branch can set `detok_finalized = true`.
      max_tokens: 32,
      ..Default::default()
    })
    .build()
    .expect("build");

  // Drain the stream — collect every token actually produced (the run ends
  // when the wrapped model returns `Err`).
  let mut produced_tokens: Vec<u32> = Vec::new();
  let mut streamed = String::new();
  let mut saw_err = false;
  {
    let stream = s.stream_respond("hello").expect("stream");
    for resp in stream {
      match resp {
        Ok(r) => {
          produced_tokens.push(r.token);
          streamed.push_str(&r.text);
        }
        Err(_) => {
          saw_err = true;
          break;
        }
      }
      // `break` on Err — the explicit drain ends, the stream is then
      // dropped (its `commit()` must flush the tail).
    }
  }
  assert!(saw_err, "the stream MUST yield an Err mid-generation");
  assert!(
    !produced_tokens.is_empty(),
    "at least one token must have streamed before the Err"
  );
  assert!(
    produced_tokens.iter().all(|&t| t == a_id),
    "every sampled token is the withheld `â`"
  );

  // The committed assistant turn.
  assert_eq!(s.history().len(), 2, "error-terminated turn still recorded");
  let recorded = s.history()[1].content();

  // Token-complete oracle: feed the SAME produced tokens into an
  // independent BPE detokenizer and `finalize()` — the committed history
  // text must equal that full detokenization. Gating on `!self.finished`
  // would skip the flush on the Err path: `recorded` would equal
  // `streamed`, which is empty (the BPE detok buffered every `â` in
  // `unflushed`).
  let reference = {
    let mut d =
      crate::tokenizer::BpeStreamingDetokenizer::new(vec![("â".to_string(), a_id)], false);
    for &t in &produced_tokens {
      d.add_token(t);
    }
    d.finalize();
    d.last_segment()
  };
  assert_eq!(
    *recorded, reference,
    "the error-terminated standard turn recorded token-complete text \
       (detok tail flushed in commit): recorded {recorded:?} == finalized {reference:?}"
  );
  // Load-bearing: the BPE detok genuinely withheld a tail — without the
  // flush the recorded text would have been empty (`streamed.is_empty()`).
  assert!(
    !reference.is_empty() && recorded.len() > streamed.len(),
    "the BPE detok genuinely withheld a tail commit() flushed \
       (streamed {streamed:?}, recorded {recorded:?})"
  );
}

#[test]
fn speculative_error_terminated_stream_flushes_detokenizer_tail() {
  // The same error-terminated case on the SPECULATIVE
  // path. An `Err` from the speculative iterator sets `finished = true`
  // without finalizing the inner driver's detok; keying `commit()`
  // on `!self.finished` would skip the flush. Instead, `commit()`
  // gates on `detok_finalized`, which the Err arm intentionally leaves
  // `false`, so `finalize_tail()` runs and flushes the withheld tail.
  let (tok, a_id, vocab) = bpe_withholding_tokenizer();
  let mut canned = vec![0.0_f32; vocab];
  canned[a_id as usize] = 10.0;

  // Wrap BOTH target and draft in `ErringAfterModel`: the speculative
  // driver calls both each cycle, so erring either eventually surfaces as
  // an `Err` step from the iterator. The driver's call shape is
  // `1 + n_draft_tokens` draft forwards per cycle and `1 + 1` target
  // forwards per cycle (1 prefill + 1 verify), so a `draft.ok_calls` budget
  // sized to allow ~1 full cycle (1 prefill + 5 propose) plus a partial
  // second-cycle draft propose lets the iterator emit up to a full burst of
  // tokens (~6) and then `Err` in the next cycle's draft phase. The target
  // budget is set high so the error is sourced from the draft (easier to
  // reason about: target stays consistent for stream bookkeeping).
  let target_inner = MockModel {
    canned: canned.clone(),
    n_kv_heads: 1,
    head_dim: 2,
  };
  let draft_inner = MockModel {
    canned,
    n_kv_heads: 1,
    head_dim: 2,
  };
  // target survives the whole run — the Err comes from draft.
  let target = ErringAfterModel::new(target_inner, 64);
  // draft: 1 prefill + 5 propose = 6 (one full cycle), then a partial 2nd
  // cycle: 1 propose succeeds, next propose Errs → ≥1 burst of tokens
  // already streamed when the Err arrives.
  let draft = ErringAfterModel::new(draft_inner, 7);

  let mut s = ChatSession::builder(Box::new(target), tok, cache_config())
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(draft),
      cache_config(),
    ))
    .generate_params(GenConfig {
      max_tokens: 32,
      ..Default::default()
    })
    .build()
    .expect("build");

  // Drain the stream until an `Err` is yielded; then drop.
  let mut produced_tokens: Vec<u32> = Vec::new();
  let mut streamed = String::new();
  let mut saw_err = false;
  {
    let stream = s.stream_respond("hello").expect("speculative stream");
    for resp in stream {
      match resp {
        Ok(r) => {
          produced_tokens.push(r.token);
          streamed.push_str(&r.text);
        }
        Err(_) => {
          saw_err = true;
          break;
        }
      }
    }
  }
  assert!(
    saw_err,
    "the speculative stream MUST yield an Err mid-generation"
  );
  assert!(
    !produced_tokens.is_empty(),
    "at least one token must have streamed before the speculative Err"
  );
  assert!(
    produced_tokens.iter().all(|&t| t == a_id),
    "every sampled token is the withheld `â`"
  );

  // Committed assistant turn must equal the independent finalized detok of
  // the tokens actually produced. Without the flush `recorded` would equal
  // the empty `streamed` — every mid-stream `last_segment()` returned ""
  // because the BPE detok buffered every `â`.
  assert_eq!(s.history().len(), 2, "error-terminated turn still recorded");
  let recorded = s.history()[1].content();

  let reference = {
    let mut d =
      crate::tokenizer::BpeStreamingDetokenizer::new(vec![("â".to_string(), a_id)], false);
    for &t in &produced_tokens {
      d.add_token(t);
    }
    d.finalize();
    d.last_segment()
  };
  assert_eq!(
    *recorded, reference,
    "the error-terminated speculative turn recorded token-complete text: \
       recorded {recorded:?} == finalized {reference:?}"
  );
  assert!(
    !reference.is_empty() && recorded.len() > streamed.len(),
    "speculative BPE detok genuinely withheld a tail commit() flushed \
       (streamed {streamed:?}, recorded {recorded:?})"
  );
}

// ---------------------------------------------------------------------------
// Small closed-form unit coverage for the value types + private helpers that
// the full-generation tests above don't exercise directly.
// ---------------------------------------------------------------------------

#[test]
fn role_as_str_round_trips_every_variant() {
  // The lowercase template keys `messages[i]["role"]` — covering the `Tool`
  // arm the full-generation tests never tag a turn with.
  assert_eq!(Role::System.as_str(), "system");
  assert_eq!(Role::User.as_str(), "user");
  assert_eq!(Role::Assistant.as_str(), "assistant");
  assert_eq!(Role::Tool.as_str(), "tool");
  // `Display` is defined as `self.as_str()`, so it agrees on every variant.
  assert_eq!(format!("{}", Role::Tool), "tool");
  // `derive_more::IsVariant` predicates line up with the named variant.
  assert!(Role::Tool.is_tool());
  assert!(!Role::User.is_tool());
}

#[test]
fn chat_message_constructors_set_role_and_content() {
  // Each role-specific constructor tags the right `Role` and keeps the text
  // verbatim — including `tool`, which the generation tests never build.
  let t = ChatMessage::tool("tool result payload");
  assert_eq!(t.role, Role::Tool);
  assert_eq!(t.content(), "tool result payload");

  let s = ChatMessage::system("sys");
  assert_eq!(s.role, Role::System);
  assert_eq!(s.content(), "sys");

  let u = ChatMessage::user("u");
  assert_eq!(u.role, Role::User);
  let a = ChatMessage::assistant("a");
  assert_eq!(a.role, Role::Assistant);

  // `ChatMessage::new` is the common path the role constructors delegate to.
  let n = ChatMessage::new(Role::Tool, String::from("owned"));
  assert_eq!(n, ChatMessage::tool("owned"));
}

#[test]
fn chat_session_error_display_messages_are_distinct_and_actionable() {
  // `ChatSessionError`'s OWN `Display` (distinct from the
  // `From<ChatSessionError> for Error` conversion the `save_cache` tests
  // observe — that surfaces an `InvariantViolation` message instead).
  let no_cache = format!("{}", ChatSessionError::NoCacheAvailable);
  assert!(
    no_cache.contains("no KV cache") && no_cache.contains("save_cache"),
    "NoCacheAvailable Display points at the missing-generation cause: {no_cache}"
  );

  let spec_save = format!("{}", ChatSessionError::SpeculativeCacheUnsupported);
  assert!(
    spec_save.contains("speculative") && spec_save.contains("consumes its KV caches"),
    "SpeculativeCacheUnsupported Display explains the consumed-cache reason: {spec_save}"
  );

  let spec_restore = format!("{}", ChatSessionError::SpeculativeCacheRestoreUnsupported);
  assert!(
    spec_restore.contains(".cache(") && spec_restore.contains(".speculative("),
    "SpeculativeCacheRestoreUnsupported Display points at both workarounds: {spec_restore}"
  );

  // The three messages are pairwise distinct (no copy/paste collision).
  assert_ne!(no_cache, spec_save);
  assert_ne!(no_cache, spec_restore);
  assert_ne!(spec_save, spec_restore);

  // `std::error::Error` is implemented (no `source`, like the Swift enum).
  let e: &dyn std::error::Error = &ChatSessionError::NoCacheAvailable;
  assert!(e.source().is_none());
}

#[test]
fn save_cache_writes_a_safetensors_file_after_a_real_turn() {
  // The `CacheSlot::Realised` arm of `save_cache`: after a turn realises the
  // cache, persisting it succeeds and writes the file. (The error arms are
  // covered by `save_cache_errors_before_any_generation` +
  // `speculative_session_does_not_expose_a_saveable_cache`.)
  let mut s = session(3);
  let _ = s.respond("hello").expect("turn");
  assert!(s.has_cache(), "the turn realised the cache");

  let path = std::env::temp_dir().join(format!(
    "mlxrs-l11-chat-session-savecache-{}.safetensors",
    std::process::id()
  ));
  let _ = std::fs::remove_file(&path);
  s.save_cache(&path).expect("save the realised cache");
  assert!(path.exists(), "save_cache wrote the safetensors file");
  // It is round-trip-loadable (the standard cache-restore path).
  let (restored, _meta) =
    crate::lm::cache::load_prompt_cache(&path).expect("reload the saved cache");
  assert_eq!(
    restored.len(),
    cache_config().num_hidden_layers,
    "the saved cache has one entry per decoder layer"
  );
  let _ = std::fs::remove_file(&path);
}

/// A temp-dir tokenizer with a valid `tokenizer.json` but **no chat template**
/// (no `tokenizer_config.json` at all), so `Tokenizer::from_path` succeeds yet
/// `apply_chat_template` returns an `Err` — the input needed to drive
/// `build_turn_prompt`'s template-error closure.
fn templateless_tokenizer() -> Tokenizer {
  let tokenizer_json = json!({
    "version": "1.0",
    "truncation": Value::Null,
    "padding": Value::Null,
    "added_tokens": [],
    "normalizer": Value::Null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": Value::Null,
    "decoder": Value::Null,
    "model": {
      "type": "WordLevel",
      "vocab": { "<unk>": 0, "hello": 1, "world": 2 },
      "unk_token": "<unk>"
    }
  });
  use std::sync::atomic::{AtomicU64, Ordering};
  static SEQ: AtomicU64 = AtomicU64::new(0);
  let seq = SEQ.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-l11-no-template-{}-{}",
    std::process::id(),
    seq
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).expect("temp tokenizer dir");
  // No `tokenizer_config.json` ⇒ `chat_template` is `None`.
  std::fs::write(
    dir.join("tokenizer.json"),
    serde_json::to_string(&tokenizer_json).expect("serialize tokenizer.json"),
  )
  .expect("write tokenizer.json");
  Tokenizer::from_path(&dir, Some(&[2u32])).expect("load templateless tokenizer")
}

#[test]
fn build_turn_prompt_maps_a_template_failure_to_a_parse_error() {
  // A tokenizer with no chat template makes `apply_chat_template_ids` fail;
  // `build_turn_prompt` must wrap that into an `Error::Parse` (its
  // `map_err` closure), not panic or pass the raw tokenizer error through.
  let s = ChatSession::builder(
    Box::new(MockModel::new(3)),
    templateless_tokenizer(),
    cache_config(),
  )
  .generate_params(GenConfig {
    max_tokens: 1,
    ..Default::default()
  })
  .build()
  .expect("build");

  let err = s
    .build_turn_prompt("hello", Role::User)
    .expect_err("templateless render must fail");
  assert!(
    matches!(err, Error::Parse(_)),
    "the template failure is surfaced as Error::Parse, got {err:?}"
  );
  // The wrapped message identifies the failing stage + carries the cause.
  let msg = format!("{err}");
  assert!(
    msg.contains("chat template"),
    "the parse error names the chat-template stage: {msg}"
  );

  // The whole turn surfaces the same error through `respond` (the render is
  // step 1 of `stream_respond_as`, before any model call).
  let mut s2 = ChatSession::builder(
    Box::new(MockModel::new(3)),
    templateless_tokenizer(),
    cache_config(),
  )
  .build()
  .expect("build");
  let resp_err = s2.respond("hello").expect_err("respond must propagate it");
  assert!(matches!(resp_err, Error::Parse(_)));
}

#[test]
fn respond_propagates_a_generation_error_after_recording_the_turn() {
  // `respond_as`'s `Err` arm: a model that errors on its first `forward`
  // makes the first stream poll yield `Err`; `respond_as` breaks the drain
  // loop, then returns that error. The turn is still recorded (the stream's
  // `Drop` commits the user prompt + an empty assistant reply).
  let inner = MockModel::new(11);
  // `ok_calls = 0` ⇒ the very first `forward` (prefill / first decode) errors.
  let model = ErringAfterModel::new(inner, 0);
  let mut s = ChatSession::builder(Box::new(model), fixture_tokenizer(), cache_config())
    .generate_params(GenConfig {
      max_tokens: 8,
      ..Default::default()
    })
    .build()
    .expect("build");

  let err = s
    .respond("hello")
    .expect_err("generation error must propagate");
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "the model's error is propagated verbatim, got {err:?}"
  );
  // The interrupted turn was still recorded (user + assistant messages).
  assert_eq!(s.history().len(), 2, "the failed turn is still recorded");
  assert_eq!(s.history()[0].role, Role::User);
  assert_eq!(s.history()[0].content(), "hello");
  assert_eq!(s.history()[1].role, Role::Assistant);
}

#[test]
fn standard_zero_max_tokens_yields_nothing_and_realises_the_cache() {
  // `max_tokens = 0`: the inner `Generator` returns `None` on its first poll
  // (`produced(0) >= max_tokens(0)` before any model call), so the stream's
  // standard `next()` hits the unexpected-`None` arm — fuses, yields no
  // token. `commit()` then finalizes the (empty) detok and realises the
  // offset-0 cache.
  let mut s = session(0);
  let mut count = 0usize;
  {
    let stream = s.stream_respond("hello").expect("stream");
    for resp in stream {
      let _ = resp.expect("no error on the empty path");
      count += 1;
    }
  }
  assert_eq!(count, 0, "max_tokens = 0 yields no tokens");
  // The turn is recorded with an empty assistant reply, and the cache is
  // realised (offset 0 — prefill never ran).
  assert!(s.has_cache(), "the empty turn still realised the cache");
  assert_eq!(s.history().len(), 2);
  assert_eq!(s.history()[1].role, Role::Assistant);
  assert_eq!(s.history()[1].content(), "", "no reply was produced");
  assert_eq!(
    s.current_cache().expect("realised")[0].offset(),
    0,
    "prefill never ran on the zero-token path"
  );
}

#[test]
fn speculative_zero_max_tokens_yields_nothing_and_marks_spent() {
  // `max_tokens = 0` on the speculative path: the inner speculative iterator
  // returns `None` on its first poll, so the stream's speculative `next()`
  // hits its `None` arm (sets `finished` + `detok_finalized`). The slot
  // becomes `SpeculativeSpent` and an empty assistant turn is recorded.
  let mut s = ChatSession::builder(
    Box::new(MockModel::new(11)),
    fixture_tokenizer(),
    cache_config(),
  )
  .speculative(SpeculativeDecodingConfig::new(
    Rc::new(MockModel::new(11)),
    cache_config(),
  ))
  .generate_params(GenConfig {
    max_tokens: 0,
    ..Default::default()
  })
  .build()
  .expect("build");

  let mut count = 0usize;
  {
    let stream = s.stream_respond("hello").expect("speculative stream");
    for resp in stream {
      let _ = resp.expect("no error on the empty speculative path");
      count += 1;
    }
  }
  assert_eq!(count, 0, "max_tokens = 0 yields no speculative tokens");
  assert_eq!(s.history().len(), 2, "the empty turn is recorded");
  assert_eq!(s.history()[1].content(), "");
  // Speculative sessions never expose a realised cache (the slot is spent).
  assert!(!s.has_cache());
  assert!(s.current_cache().is_none());
}

#[test]
fn standard_turn_finishes_on_eos_with_a_stop_reason() {
  // The standard `next()` eos arm: a `MockModel` whose argmax is the
  // tokenizer's eos id (`</s>` = 2) ends generation on the very first decode
  // step with `finish_reason = Eos` — distinct from the `Length` stop the
  // last-vocab-index `MockModel::new` always hits.
  let mut canned = vec![0.0_f32; 11];
  canned[2] = 10.0; // argmax == eos id 2
  let model = MockModel {
    canned,
    n_kv_heads: 1,
    head_dim: 2,
  };
  let mut s = ChatSession::builder(Box::new(model), fixture_tokenizer(), cache_config())
    .generate_params(GenConfig {
      // High cap so termination is the eos token, not `max_tokens`.
      max_tokens: 32,
      ..Default::default()
    })
    .build()
    .expect("build");

  let mut reasons = Vec::new();
  let mut tokens = Vec::new();
  {
    let stream = s.stream_respond("hello").expect("stream");
    for resp in stream {
      let r = resp.expect("step");
      reasons.push(r.finish_reason);
      tokens.push(r.token);
    }
  }
  // Exactly one yielded response — the eos token itself — carrying `Eos`.
  assert_eq!(reasons, vec![Some(FinishReason::Eos)]);
  assert_eq!(
    tokens,
    vec![2],
    "the eos token (id 2) was the yielded token"
  );
  // The turn was recorded and the cache realised (eos finalized the detok).
  assert!(s.has_cache());
  assert_eq!(s.history().len(), 2);
}

#[test]
fn take_cache_allocates_the_draft_cache_for_a_realised_speculative_slot() {
  // The `(Some(spec), None)` arm of `take_cache`: a speculative session whose
  // `Realised` slot carries no draft cache yet gets one allocated, once. This
  // state isn't reachable through the public builder (`.cache(..)` +
  // `.speculative(..)` is rejected at `build()`), so the session is assembled
  // directly from its parts (the test module is a child of `session`).
  let mut s = ChatSession {
    model: Box::new(MockModel::new(11)),
    tokenizer: fixture_tokenizer(),
    cache_config: cache_config(),
    instructions: None,
    generate_params: GenConfig::default(),
    speculative: Some(SpeculativeDecodingConfig::new(
      Rc::new(MockModel::new(11)),
      cache_config(),
    )),
    cache: CacheSlot::Realised {
      cache: make_prompt_cache(&cache_config()),
      draft_cache: None,
      cached: CachedTokens::empty(),
    },
    history: Vec::new(),
  };

  let (main, draft, cached) = s.take_cache();
  assert_eq!(
    main.len(),
    cache_config().num_hidden_layers,
    "the main cache is returned unchanged"
  );
  let draft = draft.expect("a speculative session's draft cache is allocated");
  assert_eq!(
    draft.len(),
    cache_config().num_hidden_layers,
    "the freshly-built draft cache has one entry per draft-model layer"
  );
  assert_eq!(cached.opaque_len, 0, "an empty cached-token record carried");
  assert!(cached.known.is_empty());
}

#[test]
fn rc_model_forwards_and_forward_embeddings_delegate_to_the_inner_model() {
  // `RcModel` (the `Rc`-shared draft-model adaptor `DraftConfig` is fed each
  // speculative turn) forwards both entry points to the inner `Rc<dyn Model>`.
  let rc: Rc<dyn Model> = Rc::new(MockModel::new(5));
  let m = RcModel(Rc::clone(&rc));

  // `forward` delegates: a `[1, 3]` window advances the cache by 3 and
  // returns `[1, 3, 5]` logits (exactly `MockModel::forward`).
  let mut cache = make_prompt_cache(&cache_config());
  let tokens = crate::array::Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).expect("tokens");
  let logits = m.forward(&tokens, &mut cache).expect("forward delegates");
  assert_eq!(logits.shape(), vec![1, 3, 5]);
  assert!(cache.iter().all(|c| c.offset() == 3));

  // `forward_embeddings` delegates to `MockModel`'s default (the
  // unimplemented VLM seam ⇒ `Err`), proving the override forwards rather
  // than re-implementing.
  let mut cache2: Vec<Box<dyn KvCache>> = Vec::new();
  let emb = crate::array::Array::from_slice::<f32>(&[0.0, 1.0], &(1usize, 1, 2)).expect("emb");
  assert!(
    m.forward_embeddings(&emb, &mut cache2).is_err(),
    "forward_embeddings forwards to the inner model's erroring default seam"
  );
}

#[test]
fn commit_falls_back_to_opaque_when_the_cache_outruns_the_named_tokens() {
  // `commit()`'s defensive fallback (the `else` arm): if the realised cache's
  // `offset()` exceeds everything the stream can name
  // (`opaque_len + prompt_ids + generated`), the whole cache is recorded as
  // an opaque prefix rather than truncating a too-short `known` region. The
  // standard `Generator` never produces this (offset grows in lockstep with
  // the fed tokens), so the stream is assembled directly with a generator
  // whose cache is pre-advanced far past its (deliberately tiny) `prompt_ids`.
  let model = MockModel::new(11);
  let m: &dyn Model = &model;
  // A cache already advanced to 20 tokens, but a `prompt_ids` of length 1 and
  // no generated tokens ⇒ `known_len (20) > logical.len() (1)`.
  let advanced = prefilled_opaque_cache(20);
  assert_eq!(advanced[0].offset(), 20);

  let generator = build_generator(m, &[3u32], advanced, GenConfig::default());
  let driver: Driver<'_> = Driver::Standard(Box::new(StandardTurn {
    generator,
    draft_cache: None,
  }));

  let mut slot = CacheSlot::Empty;
  let mut history: Vec<ChatMessage> = Vec::new();
  {
    let mut stream = ChatResponseStream {
      cache_slot: &mut slot,
      history: &mut history,
      driver: Some(driver),
      detok: fixture_tokenizer().detokenizer(),
      eos: vec![2],
      max_tokens: 4,
      prompt_tokens: 1,
      produced: 0,
      reply: String::new(),
      // Deliberately far shorter than the cache's offset.
      prompt_ids: vec![3],
      opaque_len: 0,
      generated: Vec::new(),
      finished: false,
      detok_finalized: false,
      committed: false,
    };
    // Commit WITHOUT polling the generator (so `into_cache()` returns the
    // pre-advanced offset-20 cache verbatim).
    stream.commit();
  }

  match slot {
    CacheSlot::Realised { cache, cached, .. } => {
      assert_eq!(cache[0].offset(), 20, "the advanced cache was stored");
      // The fallback recorded the WHOLE cache as opaque (no nameable known
      // region), so the next turn rebuilds rather than feeding a bogus
      // suffix.
      assert_eq!(
        cached.opaque_len, 20,
        "the cache that outran the named tokens is treated as fully opaque"
      );
      assert!(
        cached.known.is_empty(),
        "no known region recorded for the opaque fallback"
      );
    }
    _ => panic!("commit() must realise the cache"),
  }
  // The assistant turn is still appended (empty reply — nothing streamed).
  assert_eq!(history.len(), 1);
  assert_eq!(history[0].role, Role::Assistant);
}

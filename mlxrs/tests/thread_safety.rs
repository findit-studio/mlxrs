// `Array` + `Stream` `!Send`/`!Sync` enforcement lives at compile time via
// `static_assertions::assert_not_impl_any!` in `src/array/mod.rs` and
// `src/stream.rs` — those produce typecheck errors at crate-compile time, no
// runtime + no nested-cargo overhead. The trybuild equivalents (formerly in
// `tests/ui-tests/{no_send,no_sync,stream_no_send,stream_no_sync}.rs`) were
// redundant defense-in-depth and each paid ~60s of cargo-startup overhead
// per CI run (3 `TestCases::new()` invocations contending on
// `target/.cargo-lock`). Dropped — the inline static asserts are the
// enforced contract.
//
// `IntoShape` sealing is **structural** (`pub trait IntoShape: private::Sealed`
// where `Sealed` is crate-private) — a compile-time assert is not expressible
// from inside the crate (the trait IS reachable from inside; we need to prove
// it isn't reachable from *outside*). The trybuild fixture stays.

#[test]
fn into_shape_is_sealed() {
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/ui-tests/sealed_into_shape.rs");
}

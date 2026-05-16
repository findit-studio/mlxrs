#[test]
fn array_is_neither_send_nor_sync() {
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/ui-tests/no_send.rs");
  t.compile_fail("tests/ui-tests/no_sync.rs");
}

#[test]
fn into_shape_is_sealed() {
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/ui-tests/sealed_into_shape.rs");
}

#[test]
fn stream_is_neither_send_nor_sync() {
  // Stream is an index into mlx-c++ per-thread CommandEncoder state; using
  // a GPU stream off its creating thread fails at the FFI boundary. Same
  // class of constraint as Array (Codex PR #13 review).
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/ui-tests/stream_no_send.rs");
  t.compile_fail("tests/ui-tests/stream_no_sync.rs");
}

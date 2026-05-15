#[test]
fn array_is_neither_send_nor_sync() {
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/ui-tests/no_send.rs");
  t.compile_fail("tests/ui-tests/no_sync.rs");
}

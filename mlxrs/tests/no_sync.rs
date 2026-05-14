#[test]
fn array_is_not_sync() {
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/ui-tests/no_sync.rs");
}

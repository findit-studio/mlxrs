//! End-to-end smoke tests reachable from outside the crate.

#[test]
fn version_returns_non_empty_string() {
  let v = mlxrs::version();
  assert!(!v.is_empty(), "mlxrs::version() returned empty string");
  assert!(
    v.chars().next().unwrap().is_ascii_digit(),
    "version doesn't start with a digit: {v:?}"
  );
}

#[test]
fn version_is_cached() {
  // Same `&'static str` should be returned by repeat calls.
  let a = mlxrs::version();
  let b = mlxrs::version();
  assert_eq!(a.as_ptr(), b.as_ptr(), "version() is not cached");
}

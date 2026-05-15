// Asserts that mlxrs::IntoShape is sealed — a downstream type cannot
// implement it. If this ever compiles, downstream code can supply a
// shape callback slice that bypasses the FFI-boundary validation in
// from_slice / ones / zeros / full / reshape.

struct EvilShape;

impl mlxrs::IntoShape for EvilShape {
  fn with_shape<R>(
    &self,
    f: impl FnOnce(&[std::ffi::c_int]) -> mlxrs::Result<R>,
  ) -> mlxrs::Result<R> {
    f(&[-1, 0])
  }
}

fn main() {}

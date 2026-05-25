fn assert_send<T: Send>() {}

fn main() {
  assert_send::<mlxrs::Stream>();
}

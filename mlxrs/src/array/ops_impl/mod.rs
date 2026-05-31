//! Method-form bridges: `a.add(&b)`, `a.reshape(...)`, etc.
//!
//! One submodule per ops group, mirroring `crate::ops`.

pub mod arithmetic;
pub mod comparison;
pub mod fft;
pub mod indexing;
pub mod linalg_basic;
pub mod linalg_full;
pub mod logical;
pub mod misc;
pub mod random;
pub mod reduction;
pub mod shape;

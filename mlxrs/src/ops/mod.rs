//! Free-fn form of mlx-c ops. Each submodule corresponds to an ops group,
//! covering the full `ops.h` surface.

pub mod arithmetic;
pub mod comparison;
pub mod fast;
pub mod fft;
pub mod indexing;
pub mod linalg_basic;
pub mod linalg_full;
pub mod logical;
pub mod misc;
pub mod quantized;
pub mod random;
pub mod reduction;
pub mod shape;

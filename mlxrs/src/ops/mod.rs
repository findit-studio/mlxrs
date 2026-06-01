//! Free-fn form of mlx-c ops. Each submodule corresponds to an ops group,
//! covering the full `ops.h` surface, plus a few composed primitives MLX
//! has no single op for (e.g. [`interpolation::bicubic_interpolate`]).

pub mod arithmetic;
pub mod comparison;
pub mod conv;
pub mod fast;
pub mod fft;
pub mod indexing;
pub mod interpolation;
pub mod linalg_basic;
pub mod linalg_full;
pub mod logical;
pub mod misc;
pub mod quantized;
pub mod random;
pub mod reduction;
pub mod shape;

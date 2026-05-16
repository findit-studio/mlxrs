//! Free-fn form of mlx-c ops. Each submodule corresponds to an ops group.
//!
//! Phase 3.5 ships the 7 archetype templates (one per pattern). Phase 4 fans
//! the rest of `ops.h` out across 4 parallel branches, each owning 2 groups.

pub mod arithmetic;
pub mod comparison;
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

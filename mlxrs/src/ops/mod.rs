//! Free-fn form of mlx-c ops. Each submodule corresponds to an ops group.
//!
//! Phase 3 ships only `arithmetic::add` as the canonical template. Phase 3.5
//! adds the other 6 archetypes (sum, reshape, slice, concatenate, addmm, argmax).
//! Phase 4 fans out the rest of `ops.h` across 4 parallel branches.

pub mod arithmetic;

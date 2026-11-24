//! Compiled kernels of EvalSuite.
//!
//! Every function here is a direct port of a reference implementation in the `evalsuite` Python package and is
//! tested to return exactly the same result: same counts, same tie-breaks and, where floating point is involved,
//! the same operations in the same order. Words are passed as integer ids (the caller maps each distinct token
//! to an id), characters as Unicode code points.

pub mod coco;
pub mod hash;
pub mod meteor;
pub mod resample;
pub mod ter;
pub mod text;

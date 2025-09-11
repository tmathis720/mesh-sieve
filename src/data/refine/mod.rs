//! Data refinement utilities for mesh sections.
//!
//! This module provides traits and helpers for restricting and refining data
//! along sieves, including delta transformations, slice extraction, and sieved arrays.
//! Infallible helpers and the legacy [`Map`] adapter are available only when
//! the `map-adapter` feature is enabled.

pub mod delta;
pub mod helpers;
pub mod sieved_array;

// re-export the main pieces at the top level:
pub use delta::{Delta, SliceDelta};
pub use helpers::{
    try_restrict_closure, try_restrict_closure_vec, try_restrict_star, try_restrict_star_vec,
};
#[cfg(feature = "map-adapter")]
pub use helpers::{restrict_closure, restrict_closure_vec, restrict_star, restrict_star_vec};
#[cfg(feature = "rayon")]
pub use helpers::{try_restrict_closure_vec_parallel, try_restrict_star_vec_parallel};
pub use sieved_array::SievedArray;

/// A sifter is a vector of (PointId, Polarity) pairs for refinement.
pub type Sifter = Vec<(
    crate::topology::point::PointId,
    crate::topology::arrow::Polarity,
)>;

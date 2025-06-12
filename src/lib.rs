//! Public prelude for sieve-rs: mesh/data-management library for PDE codes
//!
//! All Sieve implementations provide `points()`, `base_points()`, and `cap_points()` iterators for global point set access.

pub mod algs;
pub mod data;
pub mod overlap;
pub mod topology;

#[cfg(feature = "metis-support")]
pub use algs::metis_partition;

#[cfg(feature = "mpi-support")]
pub mod partitioning;

pub use topology::sieve::{Sieve, InMemorySieve};

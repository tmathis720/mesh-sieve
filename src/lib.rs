//! Public prelude for sieve-rs: mesh/data-management library for PDE codes

pub mod algs;
pub mod data;
pub mod overlap;
pub mod topology;

#[cfg(feature = "metis-support")]
pub use algs::metis_partition;

#[cfg(feature = "partitioning")]
pub mod partitioning;

pub use topology::sieve::{Sieve, InMemorySieve, LatticeOps};

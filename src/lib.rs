//! Public prelude for sieve-rs: mesh/data-management library for PDE codes

pub mod topology;
pub mod data;
pub mod overlap;
pub mod algs;

#[cfg(feature = "metis-support")]
pub use algs::metis_partition;

#[cfg(feature = "partitioning")]
pub mod partitioning;
//! Public prelude for sieve-rs: mesh/data-management library for PDE codes

pub mod topology;
pub mod data;
pub mod overlap;
pub mod algs;

#[cfg(feature = "metis-support")]
pub mod metis_partition;

#[cfg(feature = "partitioning")]
pub mod partitioning;
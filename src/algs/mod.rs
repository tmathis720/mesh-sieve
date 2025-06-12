//! Algorithm module: re-exports public algorithms for mesh and partitioning operations.
//!
//! This module provides access to mesh completion, distribution, dual graph construction,
//! partitioning, traversal, and lattice utilities.

pub mod communicator;
pub mod completion;
pub mod distribute;
pub mod dual_graph;
pub mod lattice;
pub mod metis_partition;
pub mod partition;
pub mod traversal;

pub use completion::complete_section;
pub use lattice::adjacent;
pub use crate::algs::distribute::distribute_mesh;

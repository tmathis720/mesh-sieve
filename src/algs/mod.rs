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
pub mod rcm;
pub mod reduction;
pub mod traversal_core;
pub mod traversal;
pub mod traversal_ref;
pub mod wire;

pub use crate::algs::distribute::distribute_mesh;
pub use completion::complete_section;
pub use lattice::adjacent;

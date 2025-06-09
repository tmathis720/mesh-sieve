//! Re-export public algorithms.

pub mod traversal;
pub mod lattice;
pub mod communicator;
pub mod completion;
pub mod dual_graph;
pub mod metis_partition;
pub mod partition;

pub use lattice::adjacent;
pub use completion::complete_section;

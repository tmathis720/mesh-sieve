//! Re-export public algorithms.

pub mod communicator;
pub mod completion;
pub mod dual_graph;
pub mod lattice;
pub mod metis_partition;
pub mod partition;
pub mod traversal;

pub use completion::complete_section;
pub use lattice::adjacent;

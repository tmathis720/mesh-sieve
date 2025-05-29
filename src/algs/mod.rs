//! Re-export public algorithms.

pub mod traversal;
pub mod lattice;
pub mod communicator;
pub mod completion;
pub mod dual_graph;
pub mod partition;

pub use lattice::{meet, join, adjacent};
pub use completion::complete_section;

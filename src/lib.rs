//! Public prelude for sieve-rs: mesh/data-management library for PDE codes
//!
//! All Sieve implementations provide `points()`, `base_points()`, and `cap_points()` iterators for global point set access.

// Re-export our major subsystems:
pub mod topology;
pub mod data;
pub mod overlap;
pub mod algs;
#[cfg(feature = "mpi-support")]
pub mod partitioning;

/// A convenient prelude to import the most-used traits & types:
pub mod prelude {
    pub use crate::topology::sieve::Sieve;
    pub use crate::topology::sieve::InMemorySieve;
    pub use crate::topology::stack::{Stack, InMemoryStack};
    pub use crate::topology::point::PointId;
    pub use crate::data::atlas::Atlas;
    pub use crate::data::section::{Section, Map};
    pub use crate::overlap::delta::{Delta, CopyDelta, AddDelta};
    pub use crate::overlap::overlap::Overlap;
    pub use crate::algs::communicator::Communicator;
    #[cfg(feature="mpi-support")]
    pub use crate::algs::communicator::MpiComm;
    #[cfg(feature="rayon")]
    pub use crate::algs::communicator::RayonComm;
    pub use crate::algs::completion::{complete_sieve, complete_section, complete_stack};
}

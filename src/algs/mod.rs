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
pub mod traversal;
pub mod traversal_core;
pub mod traversal_ref;
pub mod wire;
pub mod submesh;

pub use crate::algs::distribute::{
    distribute_mesh, distribute_with_overlap, CellPartitioner, CustomPartitioner,
    DistributionConfig, DistributedMeshData, ProvidedPartition,
};
#[cfg(feature = "metis-support")]
pub use crate::algs::distribute::MetisPartitioner;
pub use completion::{
    complete_section, complete_section_with_ownership, complete_section_with_tags,
    complete_section_with_tags_and_ownership,
};
pub use lattice::adjacent;
pub use submesh::{extract_by_label, SubmeshMaps};

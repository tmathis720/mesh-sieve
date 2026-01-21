//! Algorithm module: re-exports public algorithms for mesh and partitioning operations.
//!
//! This module provides access to mesh completion, distribution, dual graph construction,
//! partitioning, traversal, and lattice utilities.

pub mod communicator;
pub mod completion;
pub mod adjacency_graph;
pub mod boundary;
pub mod distribute;
pub mod dual_graph;
pub mod extrude;
pub mod field_transfer;
pub mod interpolate;
pub mod lattice;
pub mod metis_partition;
pub mod partition;
pub mod rcm;
pub mod reduction;
pub mod renumber;
pub mod submesh;
pub mod traversal;
pub mod traversal_core;
pub mod traversal_ref;
pub mod wire;

#[cfg(feature = "metis-support")]
pub use crate::algs::distribute::MetisPartitioner;
pub use crate::algs::distribute::{
    CellPartitioner, CustomPartitioner, DistributedMeshData, DistributionConfig, ProvidedPartition,
    distribute_mesh, distribute_with_overlap,
};
pub use completion::{
    complete_section, complete_section_with_ownership, complete_section_with_tags,
    complete_section_with_tags_and_ownership,
};
pub use field_transfer::{
    transfer_section_by_nearest_cell_centroid, transfer_section_by_nearest_point,
    transfer_section_by_refinement_map, transfer_section_by_shared_labels,
};
pub use lattice::adjacent;
pub use boundary::{
    BoundaryClassification, BoundaryLabelValues, DEFAULT_BOUNDARY_LABEL, DEFAULT_INTERIOR_LABEL,
    classify_boundary_points, label_boundary_points, label_boundary_points_with,
};
pub use renumber::{
    StratifiedOrdering, renumber_coordinate_dm, renumber_points, renumber_points_stratified,
    stratified_permutation,
};
pub use submesh::{SubmeshMaps, extract_by_label};

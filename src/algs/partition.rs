//! Partitioning wrappers for ParMETIS / Zoltan
//! and native Onizuka-inspired partitioning (see partitioning/).
//!
//! This module provides a unified interface for graph partitioning, including
//! native algorithms and wrappers for external libraries. It also exposes
//! partitioning metrics such as edge cut and replication factor.

#[cfg(feature = "mpi-support")]
pub use crate::partitioning::{
    PartitionMap, PartitionerConfig, PartitionerError,
    metrics::{edge_cut, replication_factor},
    partition,
};

#[cfg(feature = "mpi-support")]
use crate::partitioning::graph_traits::PartitionableGraph;

/// Partition a graph using the Onizuka et al. inspired native partitioner.
///
/// # Arguments
/// * `graph` - The input graph implementing `PartitionableGraph`.
/// * `cfg` - Partitioning configuration (number of parts, balance, etc).
///
/// # Returns
/// * `Ok(PartitionMap)` on success, mapping each vertex to a part.
/// * `Err(PartitionerError)` on failure.
#[cfg(feature = "mpi-support")]
pub fn native_partition<G>(
    graph: &G,
    cfg: &PartitionerConfig,
) -> Result<PartitionMap<G::VertexId>, PartitionerError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    partition(graph, cfg)
}

/// Compute the edge cut of a partitioning.
///
/// Returns the number of edges crossing between parts.
#[cfg(feature = "mpi-support")]
pub fn partition_edge_cut<G>(graph: &G, pm: &PartitionMap<G::VertexId>) -> usize
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd + Eq + std::hash::Hash + Copy + 'static,
{
    edge_cut(graph, pm)
}

/// Compute the replication factor of a partitioning.
///
/// Returns the average number of parts each vertex is present in.
#[cfg(feature = "mpi-support")]
pub fn partition_replication_factor<G>(graph: &G, pm: &PartitionMap<G::VertexId>) -> f64
where
    G: PartitionableGraph,
    G::VertexId: Eq + std::hash::Hash + Copy,
{
    replication_factor(graph, pm)
}

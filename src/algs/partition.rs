//! Partitioning wrappers for ParMETIS/Zoltan and native partitioning.
//!
//! NOTE: All edge-consuming routines MUST use `PartitionableGraph::edges()`,
//! which yields each undirected edge exactly once (u < v). Backends may
//! override `edges()` with a more efficient implementation.

#[cfg(feature = "mpi-support")]
pub use crate::partitioning::{
    PartitionMap, PartitionerConfig, PartitionerError,
    metrics::{edge_cut, replication_factor},
    partition,
};

#[cfg(feature = "mpi-support")]
pub use crate::partitioning::graph_traits::PartitionableGraph; // re-export for users

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
/// Uses `g.edges()` (undirected edges, yielded once).
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
/// Internally uses `g.edges()`; no nested parallelism or mutex storms.
#[cfg(feature = "mpi-support")]
pub fn partition_replication_factor<G>(graph: &G, pm: &PartitionMap<G::VertexId>) -> f64
where
    G: PartitionableGraph,
    G::VertexId: Eq + std::hash::Hash + Copy + Sync + 'static,
{
    replication_factor(graph, pm)
}

/// Convenience: compute both edge cut and replication factor in one call.
///
/// Implemented with a single pass over `g.edges()` to avoid double work.
#[cfg(feature = "mpi-support")]
pub fn partition_metrics<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> (usize, f64)
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd + Eq + std::hash::Hash + Copy + Sync + 'static,
{
    use std::collections::HashMap;
    use std::collections::hash_map::Entry;

    let mut cut = 0usize;
    // Bitmask of incident parts per vertex (fast RF for k<=64).
    let mut mask: HashMap<G::VertexId, u64> = HashMap::new();

    use rayon::iter::ParallelIterator;
    for (u, v) in g.edges().collect::<Vec<_>>() {
        let pu = pm.part_of(u);
        let pv = pm.part_of(v);
        if pu != pv {
            cut += 1;
        }
        let mu = 1u64 << (pu % 64);
        let mv = 1u64 << (pv % 64);
        match mask.entry(u) {
            Entry::Occupied(mut e) => *e.get_mut() |= mu,
            Entry::Vacant(e) => {
                e.insert(mu);
            }
        }
        match mask.entry(v) {
            Entry::Occupied(mut e) => *e.get_mut() |= mv,
            Entry::Vacant(e) => {
                e.insert(mv);
            }
        }
    }

    let rf = if mask.is_empty() {
        0.0
    } else {
        let total: u64 = mask.values().map(|m| m.count_ones() as u64).sum();
        total as f64 / mask.len() as f64
    };
    (cut, rf)
}

/// Debug checker for the `edges()` contract:
/// - undirected, each edge exactly once (u < v)
/// - no self-loops
#[cfg(all(
    feature = "mpi-support",
    any(debug_assertions, feature = "check-graph-edges")
))]
pub fn debug_check_edges_contract<G>(g: &G)
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd + Eq + std::hash::Hash + Copy + std::fmt::Debug + 'static,
{
    use rayon::iter::ParallelIterator;
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for (u, v) in g.edges().collect::<Vec<_>>() {
        debug_assert!(u < v, "edges() must yield u < v; got ({:?},{:?})", u, v);
        debug_assert!(u != v, "edges() must not yield self-loops");
        let key = (u, v);
        debug_assert!(
            seen.insert(key),
            "duplicate edge from edges(): ({:?},{:?})",
            u,
            v
        );
    }
}

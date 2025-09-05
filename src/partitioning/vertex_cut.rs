//! Vertex cut construction for distributed graph partitioning.
//!
//! This module provides [`build_vertex_cuts`], a parallel algorithm for assigning primary owners
//! and replica lists for vertices in a partitioned graph, supporting distributed ghosting and communication.

use crate::partitioning::error::PartitionError;
use crate::partitioning::PartitionMap;
use crate::partitioning::graph_traits::PartitionableGraph;
use ahash::AHasher;
use parking_lot::Mutex;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*; // brings IntoParallelIterator, ParallelIterator, etc.
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Build primary owner and replica lists for each vertex in a partitioned graph.
///
/// For each edge `(u, v)` in the graph where `pm.part_of(u) != pm.part_of(v)`,
/// chooses a “primary owner” of that edge (by hashing `(u,v)` with a salt),
/// then records the other endpoint as a “replica.”
///
/// Returns `(Vec<PartId>, Vec<Vec<(VertexId, PartId)>>)`:
/// - `primary_owner[v]` is the primary owner’s part ID for vertex `v`.
/// - `replicas[v]` is a `Vec` of `(neighbor_vertex, neighbor_part)` that need ghosting.
///
/// # Arguments
/// - `graph`: The partitioned graph.
/// - `pm`: The partition map indicating part assignments for each vertex.
/// - `salt`: A salt value for deterministic hashing.
///
/// # Returns
/// A tuple containing:
/// - `primary_owner`: Vector mapping each vertex to its primary owner's part ID.
/// - `replicas`: Vector of vectors, where each inner vector lists the replicas for a vertex.
///
/// # Parallelism
/// This function uses parallel iteration for efficient processing of large graphs.
///
/// # Features
/// Only available with the `mpi-support` feature enabled.
#[cfg(feature = "mpi-support")]
pub fn build_vertex_cuts<G>(
    graph: &G,
    pm: &PartitionMap<G::VertexId>,
    salt: u64,
) -> Result<(Vec<usize>, Vec<Vec<(G::VertexId, usize)>>), PartitionError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    // 1. Determine number of vertices
    let verts: Vec<G::VertexId> = graph.vertices().collect();
    let n = verts.len();
    // Map vertex ID to index
    let vert_idx: std::collections::HashMap<G::VertexId, usize> = verts.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();

    // Determine how many parts we have (max part ID + 1)
    let num_parts = {
        let mut max = 0;
        let mut any = false;
        for (_v, &p) in pm.iter() {
            any = true;
            if p > max { max = p; }
        }
        if !any {
            return Err(PartitionError::NoParts);
        }
        max + 1
    };
    // Create an atomic counter for each part’s replica-load
    let part_replica_count: Vec<AtomicUsize> =
        (0..num_parts).map(|_| AtomicUsize::new(0)).collect();

    // 2. Allocate primary_part array, default = usize::MAX
    let mut primary_part: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();

    // 3. Allocate one Mutex<Vec<...>> per vertex for its replicas
    let replica_lists: Vec<Mutex<Vec<(G::VertexId, usize)>>> =
        (0..n).map(|_| Mutex::new(Vec::new())).collect();

    // 4. Parallel edge sweep: for each u, for each neighbor v where u < v
    verts.par_iter().for_each(|&u| {
        graph.neighbors(u).into_par_iter().for_each(|v| {
            if u < v {
                let pu = pm.get(&u).copied().ok_or(PartitionError::MissingPartition(u)).unwrap();
                let pv = pm.get(&v).copied().ok_or(PartitionError::MissingPartition(v)).unwrap();
                if pu != pv {
                    let count_u = part_replica_count[pu].load(Ordering::Relaxed);
                    let count_v = part_replica_count[pv].load(Ordering::Relaxed);
                    let (owner, other, owner_part) = if count_u < count_v {
                        (u, v, pu)
                    } else if count_v < count_u {
                        (v, u, pv)
                    } else {
                        let mut h = AHasher::default();
                        h.write_u64(salt);
                        h.write_u64(u as u64);
                        h.write_u64(v as u64);
                        let hashval = h.finish();
                        if (hashval & 1) == 0 {
                            (u, v, pu)
                        } else {
                            (v, u, pv)
                        }
                    };
                    let owner_idx = *vert_idx.get(&owner)
                        .ok_or(PartitionError::VertexNotFound(owner)).unwrap();
                    let other_idx = *vert_idx.get(&other)
                        .ok_or(PartitionError::VertexNotFound(other)).unwrap();
                    part_replica_count[owner_part].fetch_add(1, Ordering::Relaxed);
                    primary_part[owner_idx].store(owner_part, Ordering::Relaxed);
                    primary_part[other_idx].store(owner_part, Ordering::Relaxed);
                    {
                        let mut guard = replica_lists[owner_idx].lock();
                        guard.push((other, owner_part));
                    }
                    {
                        let mut guard = replica_lists[other_idx].lock();
                        guard.push((owner, owner_part));
                    }
                }
            }
        });
    });

    // 5. Deduplicate & sort each vertex’s replica list
    for mutex in &replica_lists {
        let mut vec = mutex.lock();
        vec.sort_unstable();
        vec.dedup();
    }

    // 6. For any vertex never set, assign its own part
    for (v, atomic_part) in primary_part.iter_mut().enumerate() {
        if atomic_part.load(Ordering::Relaxed) == usize::MAX {
            let vert = verts[v];
            atomic_part.store(pm.get(&vert).copied().ok_or(PartitionError::MissingPartition(vert))?, Ordering::Relaxed);
        }
    }

    // 7. Collect final primary_owner Vec<usize>
    let primary_owner: Vec<usize> = primary_part
        .into_iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    // 8. Collect final replicas Vec<Vec<(VertexId, usize)>>
    let replicas: Vec<Vec<(G::VertexId, usize)>> =
        replica_lists.into_iter().map(|m| m.into_inner()).collect();

    Ok((primary_owner, replicas))
}

#[cfg(test)]
#[cfg(feature = "mpi-support")]
mod tests {
    use super::*;
    use crate::partitioning::PartitionMap;
    use crate::partitioning::graph_traits::PartitionableGraph;
    use rayon::iter::IntoParallelIterator;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    /// Simple graph implementation for testing
    struct TestGraph {
        edges: Vec<(usize, usize)>,
        n: usize,
    }
    impl TestGraph {
        fn new(edges: Vec<(usize,usize)>, n: usize) -> Self {
            TestGraph { edges, n }
        }
    }

    impl PartitionableGraph for TestGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize> where Self: 'a;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize> where Self: 'a;
        type NeighIter<'a> = std::vec::IntoIter<usize> where Self: 'a;
        type EdgeParIter<'a> = rayon::vec::IntoIter<(usize, usize)> where Self: 'a;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.neighbors_seq(v).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
            let ns = self.edges.iter()
                .filter_map(|&(a,b)| if a==v { Some(b) } else if b==v { Some(a) } else { None })
                .collect::<Vec<_>>();
            ns.into_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }
        fn edges(&self) -> Self::EdgeParIter<'_> {
            self.edges.clone().into_par_iter()
        }
    }

    #[test]
    fn vertex_cut_cycle_nondeterministic() {
        // 4-cycle: 0-1-2-3-0 with two parts: {0,1}=0 and {2,3}=1
        let g = TestGraph::new(vec![(0,1),(1,2),(2,3),(3,0)], 4);
        let mut pm = PartitionMap::with_capacity(4);
        pm.insert(0, 0); pm.insert(1, 0);
        pm.insert(2, 1); pm.insert(3, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 42).unwrap();

        // Edges (1-2) and (3-0) cross partitions; owner must be same for both endpoints
        for &(u,v) in &[(1,2),(3,0)] {
            assert_eq!(primary[u], primary[v], "Primary owner mismatch on edge {}-{}", u, v);
            let p = primary[u];
            // Check that each lists the other as a replica
            let has_uv = replicas[u].iter().any(|&(nbr, part)| nbr==v && part==p);
            let has_vu = replicas[v].iter().any(|&(nbr, part)| nbr==u && part==p);
            assert!(has_uv && has_vu, "Missing replica entries between {} and {}", u, v);
        }
    }

    #[test]
    fn vertex_cut_isolated_vertex() {
        // Single isolated vertex should own itself, no replicas
        let g = TestGraph::new(vec![], 1);
        let mut pm = PartitionMap::with_capacity(1);
        pm.insert(0, 0);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 99).unwrap();
        assert_eq!(primary, vec![0]);
        assert!(replicas[0].is_empty(), "Isolated vertex should have no replicas");
    }

    #[test]
    fn vertex_cut_two_node_owner_balancing() {
        // 2-node graph: 0-1, different parts => primary must agree, and each replicates the other
        let g = TestGraph::new(vec![(0,1)], 2);
        let mut pm = PartitionMap::with_capacity(2);
        pm.insert(0, 0); pm.insert(1, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 0).unwrap();

        // Both endpoints share the same owner
        assert_eq!(primary[0], primary[1], "Both endpoints must share a primary owner");
        // Each should list the other in its replica list
        assert!(replicas[0].iter().any(|&(nbr, _)| nbr==1), "Vertex 0 missing replica for 1");
        assert!(replicas[1].iter().any(|&(nbr, _)| nbr==0), "Vertex 1 missing replica for 0");
    }

    #[test]
    fn vertex_cut_triangle_distinct_parts() {
        // Triangle: 0-1-2-0, all in distinct parts; ensure at least one replica per vertex
        let g = TestGraph::new(vec![(0,1),(1,2),(2,0)], 3);
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0); pm.insert(1, 1); pm.insert(2, 2);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 123).unwrap();

        for v in 0..3 {
            // primary owner must be some valid part
            assert!(primary[v] < 3, "Invalid owner for vertex {}", v);
            // and should have at least one replica since every edge crosses
            assert!(!replicas[v].is_empty(), "Vertex {} has no replicas", v);
        }
    }
}

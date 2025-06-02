use crate::partitioning::graph_traits::PartitionableGraph;
use crate::partitioning::PartitionMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::collections::HashSet;
use ahash::AHasher;
use rayon::iter::ParallelIterator;

/// For each edge `(u, v)` in the graph where `pm.part_of(u) != pm.part_of(v)`,
/// choose a “primary owner” of that edge (e.g., by hashing `(u,v)` with a salt),
/// then record the other endpoint as a “replica.”
/// Returns a `(Vec<PartId>, Vec<Vec<(VertexId, PartId)>>)`:
/// - `primary_owner[v] == some_part` for each `v`.
/// - `replicas[v]` is a `Vec` of `(neighbor_vertex, neighbor_part)` that need to be
///    ghosted into `primary_owner[v]`’s part.
pub fn build_vertex_cuts<G>(
    graph: &G,
    pm: &PartitionMap<G::VertexId>,
    salt: u64,
) -> (Vec<usize>, Vec<Vec<(G::VertexId, usize)>>)
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    let n = graph.vertices().count();
    let mut primary_part: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
    let replica_lists: Vec<Mutex<Vec<(G::VertexId, usize)>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();
    use std::hash::Hasher;
    graph.vertices().par_iter().for_each(|&u| {
        graph.neighbors(u).for_each(|&v| {
            if u < v {
                let pu = pm.part_of(u);
                let pv = pm.part_of(v);
                if pu != pv {
                    let mut h = AHasher::default();
                    h.write_u64(salt as u64);
                    h.write_u64(u as u64);
                    h.write_u64(v as u64);
                    let hashval = h.finish();
                    let (owner, other, owner_part, _other_part) =
                        if (hashval & 1) == 0 {
                            (u, v, pu, pv)
                        } else {
                            (v, u, pv, pu)
                        };
                    primary_part[owner].store(owner_part, Ordering::Relaxed);
                    primary_part[other].store(owner_part, Ordering::Relaxed);
                    {
                        let mut guard = replica_lists[owner].lock();
                        guard.push((other, owner_part));
                    }
                    {
                        let mut guard = replica_lists[other].lock();
                        guard.push((owner, owner_part));
                    }
                }
            }
        });
    });
    // Deduplicate and sort each replica list
    for mutex in &replica_lists {
        let mut vec = mutex.lock();
        vec.sort_unstable();
        vec.dedup();
    }
    // For any vertex never set, set primary_part[v] = pm.part_of(v)
    for (v, part) in primary_part.iter_mut().enumerate() {
        if part.load(Ordering::Relaxed) == usize::MAX {
            part.store(pm.part_of(v), Ordering::Relaxed);
        }
    }
    let primary_owner: Vec<usize> = primary_part.into_iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let replicas: Vec<Vec<(G::VertexId, usize)>> = replica_lists.into_iter().map(|m| m.into_inner()).collect();
    (primary_owner, replicas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;
    struct TestGraph {
        edges: Vec<(usize, usize)>,
        n: usize,
    }
    impl PartitionableGraph for TestGraph {
        type VertexId = usize;
        fn vertices(&self) -> Vec<Self::VertexId> { (0..self.n).collect() }
        fn neighbors(&self, v: usize) -> Vec<Self::VertexId> {
            self.edges.iter().filter_map(|&(a, b)| {
                if a == v { Some(b) } else if b == v { Some(a) } else { None }
            }).collect()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors(v).len()
        }
    }
    #[test]
    fn vertex_cut_cycle() {
        // 4-cycle: 0-1-2-3-0, (0,1)->0, (2,3)->1
        let g = TestGraph { edges: vec![(0,1),(1,2),(2,3),(3,0)], n: 4 };
        let mut pm = PartitionMap::with_capacity(4);
        pm.insert(0, 0); pm.insert(1, 0); pm.insert(2, 1); pm.insert(3, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 42);
        // Check that each vertex has a primary part and correct replica structure
        for v in 0..4 {
            assert!(primary[v] == 0 || primary[v] == 1);
        }
        // Edges (1-2) and (3-0) cross parts, so replicas should be non-empty for those
        assert!(replicas[1].iter().any(|&(n, p)| n == 2 && p == 1));
        assert!(replicas[2].iter().any(|&(n, p)| n == 1 && p == 0));
        assert!(replicas[3].iter().any(|&(n, p)| n == 0 && p == 0));
        assert!(replicas[0].iter().any(|&(n, p)| n == 3 && p == 1));
    }
    #[test]
    fn vertex_cut_triangle_distinct() {
        // Triangle: 0-1-2-0, all in different parts
        let g = TestGraph { edges: vec![(0,1),(1,2),(2,0)], n: 3 };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0); pm.insert(1, 1); pm.insert(2, 2);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 123);
        for v in 0..3 {
            assert!(primary[v] < 3);
            assert!(!replicas[v].is_empty());
        }
    }
    #[test]
    fn vertex_cut_isolated() {
        // Isolated vertex 0
        let g = TestGraph { edges: vec![], n: 1 };
        let mut pm = PartitionMap::with_capacity(1);
        pm.insert(0, 0);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 99);
        assert_eq!(primary[0], 0);
        assert!(replicas[0].is_empty());
    }
    #[test]
    fn vertex_cut_two_node_owner_rule() {
        // 2-node graph: 0-1, different parts
        let g = TestGraph { edges: vec![(0,1)], n: 2 };
        let mut pm = PartitionMap::with_capacity(2);
        pm.insert(0, 0); pm.insert(1, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 0);
        // Both should have the same primary (owner's part)
        assert!(primary[0] == primary[1]);
        // Each should have the other as a replica
        assert!(replicas[0].iter().any(|&(n, _)| n == 1));
        assert!(replicas[1].iter().any(|&(n, _)| n == 0));
    }

    #[test]
    fn vertex_cut_path_internal_and_boundary() {
        // 3-node path: 0-1-2, pm(0)=0, pm(1)=0, pm(2)=1
        let g = TestGraph { edges: vec![(0,1),(1,2)], n: 3 };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0); pm.insert(1, 0); pm.insert(2, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 42);
        // Vertex 0 is internal, should have its own part
        assert_eq!(primary[0], 0);
        // Vertices 1 and 2 should have the same primary (owner's part)
        assert_eq!(primary[1], primary[2]);
        // Replicas: 1 and 2 should reference each other
        assert!(replicas[1].iter().any(|&(n, _)| n == 2));
        assert!(replicas[2].iter().any(|&(n, _)| n == 1));
    }
}

use crate::partitioning::PartitionMap;
use crate::partitioning::graph_traits::PartitionableGraph;
use ahash::AHasher;
use parking_lot::Mutex;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*; // brings IntoParallelIterator, ParallelIterator, etc.
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};

/// For each edge `(u, v)` in the graph where `pm.part_of(u) != pm.part_of(v)`,
/// choose a “primary owner” of that edge (by hashing `(u,v)` with a salt),
/// then record the other endpoint as a “replica.”  
/// Returns `(Vec<PartId>, Vec<Vec<(VertexId, PartId)>>)`:
/// - `primary_owner[v]` is the primary owner’s part ID for vertex v.
/// - `replicas[v]` is a Vec of `(neighbor_vertex, neighbor_part)` that need ghosting.
pub fn build_vertex_cuts<G>(
    graph: &G,
    pm: &PartitionMap<G::VertexId>,
    salt: u64,
) -> (Vec<usize>, Vec<Vec<(G::VertexId, usize)>>)
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    // 1. Determine number of vertices
    let n = graph.vertices().count();

    // 2. Allocate primary_part array, default = usize::MAX
    let mut primary_part: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();

    // 3. Allocate one Mutex<Vec<...>> per vertex for its replicas
    let replica_lists: Vec<Mutex<Vec<(G::VertexId, usize)>>> =
        (0..n).map(|_| Mutex::new(Vec::new())).collect();

    // 4. Parallel edge sweep: for each u, for each neighbor v where u < v
    graph
        .vertices() // yields a parallel iterator over VertexId
        .into_par_iter()
        .for_each(|u| {
            // For each neighbor v of u (owned Vec<usize>), iterate in parallel
            graph.neighbors(u).into_par_iter().for_each(|v| {
                if u < v {
                    let pu = pm.part_of(u);
                    let pv = pm.part_of(v);
                    if pu != pv {
                        // Decide owner via hash(salt,u,v)
                        let mut h = AHasher::default();
                        h.write_u64(salt);
                        h.write_u64(u as u64);
                        h.write_u64(v as u64);
                        let hashval = h.finish();
                        let (owner, other, owner_part) = if (hashval & 1) == 0 {
                            (u, v, pu)
                        } else {
                            (v, u, pv)
                        };

                        // 4.a Set primary_part for both vertices to the owner’s part
                        primary_part[owner].store(owner_part, Ordering::Relaxed);
                        primary_part[other].store(owner_part, Ordering::Relaxed);

                        // 4.b Push replica entries under mutex
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

    // 5. Deduplicate & sort each vertex’s replica list
    for mutex in &replica_lists {
        let mut vec = mutex.lock();
        vec.sort_unstable();
        vec.dedup();
    }

    // 6. For any vertex never set, assign its own part
    for (v, atomic_part) in primary_part.iter_mut().enumerate() {
        if atomic_part.load(Ordering::Relaxed) == usize::MAX {
            atomic_part.store(pm.part_of(v), Ordering::Relaxed);
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
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.edges
                .iter()
                .filter_map(|&(a, b)| {
                    if a == v {
                        Some(b)
                    } else if b == v {
                        Some(a)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .into_par_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors(v).count()
        }
    }

    #[test]
    fn vertex_cut_cycle() {
        // 4‐cycle: 0–1–2–3–0, (0,1)->0, (2,3)->1
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            n: 4,
        };
        let mut pm = PartitionMap::with_capacity(4);
        pm.insert(0, 0);
        pm.insert(1, 0);
        pm.insert(2, 1);
        pm.insert(3, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 42);

        // Each vertex should have a valid primary (0 or 1)
        for v in 0..4 {
            assert!(primary[v] == 0 || primary[v] == 1);
        }
        // Edges (1–2) and (3–0) cross parts
        // Accept either direction of ghosting due to hash-based owner selection
        assert!(
            replicas[1]
                .iter()
                .any(|&(n, p)| n == 2 && (p == 0 || p == 1))
                || replicas[2]
                    .iter()
                    .any(|&(n, p)| n == 1 && (p == 0 || p == 1)),
            "Expected a ghost edge between 1 and 2"
        );
        assert!(
            replicas[3]
                .iter()
                .any(|&(n, p)| n == 0 && (p == 0 || p == 1))
                || replicas[0]
                    .iter()
                    .any(|&(n, p)| n == 3 && (p == 0 || p == 1)),
            "Expected a ghost edge between 3 and 0"
        );
    }

    #[test]
    fn vertex_cut_triangle_distinct() {
        // Triangle: 0–1–2–0, all in different parts
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2), (2, 0)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0);
        pm.insert(1, 1);
        pm.insert(2, 2);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 123);

        for v in 0..3 {
            assert!(primary[v] < 3);
            assert!(!replicas[v].is_empty());
        }
    }

    #[test]
    fn vertex_cut_isolated() {
        // Isolated vertex 0
        let g = TestGraph {
            edges: vec![],
            n: 1,
        };
        let mut pm = PartitionMap::with_capacity(1);
        pm.insert(0, 0);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 99);
        assert_eq!(primary[0], 0);
        assert!(replicas[0].is_empty());
    }

    #[test]
    fn vertex_cut_two_node_owner_rule() {
        // 2‐node graph: 0–1, different parts
        let g = TestGraph {
            edges: vec![(0, 1)],
            n: 2,
        };
        let mut pm = PartitionMap::with_capacity(2);
        pm.insert(0, 0);
        pm.insert(1, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 0);

        // Both should have the same primary (owner’s part)
        assert_eq!(primary[0], primary[1]);
        // Each should list the other as a replica
        assert!(replicas[0].iter().any(|&(n, _)| n == 1));
        assert!(replicas[1].iter().any(|&(n, _)| n == 0));
    }

    #[test]
    fn vertex_cut_path_internal_and_boundary() {
        // 3‐node path: 0–1–2, pm(0)=0, pm(1)=0, pm(2)=1
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0);
        pm.insert(1, 0);
        pm.insert(2, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 42);

        // Vertex 0 is internal → should keep its own part
        assert_eq!(primary[0], 0);
        // Vertices 1 and 2 share a crossing edge → same primary
        assert_eq!(primary[1], primary[2]);
        // Each lists the other as a replica
        assert!(replicas[1].iter().any(|&(n, _)| n == 2));
        assert!(replicas[2].iter().any(|&(n, _)| n == 1));
    }
}

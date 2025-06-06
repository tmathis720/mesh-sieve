// Metrics skeleton for partitioning
#![cfg(feature = "partitioning")]

use super::{PartitionableGraph, PartitionMap};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

/// Returns the part ID for a given vertex.
impl<V: Eq + Hash + Copy> PartitionMap<V> {
    pub fn part_of(&self, v: V) -> usize {
        *self.get(&v).expect("vertex not found in PartitionMap")
    }
}

/// Computes the edge cut of a partitioning (O(E)).
/// Intended for debug/CI use.
pub fn edge_cut<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> usize
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd + Eq + Hash + Copy,
{
    // We iterate over all (u,v) with u < v and count how many cross‐parts.
    // Then divide by 1 if vertices() yields each undirected edge exactly once.
    // If neighbors() is symmetric (u->v and v->u), we must divide by 2 at the end.

    // Build a Vec of all undirected edges (u < v) in parallel:
    let cut_count: usize = g
        .vertices()
        .flat_map(|u| {
            // NeighParIter is a ParallelIterator, not Iterator, so use .filter_map directly
            g.neighbors(u)
                .filter_map(move |v| {
                    if u < v {
                        if pm.part_of(u) != pm.part_of(v) {
                            Some(1)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
        })
        .sum();

    cut_count
}

/// Computes the replication factor of a partitioning (O(E)).
/// Intended for debug/CI use.
pub fn replication_factor<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> f64
where
    G: PartitionableGraph,
    G::VertexId: Eq + Hash + Copy,
{
    use rayon::prelude::*;
    // 1. Gather all vertices into a Vec so we can index them [0..n)
    let verts: Vec<G::VertexId> = g.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return 0.0;
    }

    // 2. Build a map from VertexId -> position index
    let idx_map: HashMap<G::VertexId, usize> = verts.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();

    // 3. Create a Vec<HashSet<usize>> per vertex to accumulate all owning parts (thread-safe)
    let owners: Vec<std::sync::Mutex<HashSet<usize>>> = (0..n).map(|_| std::sync::Mutex::new(HashSet::new())).collect();

    // 4. For each vertex u, mark its own part, and also its part for each neighbor v (in parallel)
    verts.par_iter().for_each(|&u| {
        let pu = pm.part_of(u);
        let u_idx = idx_map[&u];
        owners[u_idx].lock().unwrap().insert(pu);
        g.neighbors(u).for_each(|v| {
            let v_idx = idx_map[&v];
            owners[v_idx].lock().unwrap().insert(pu);
        });
    });

    // 5. Sum the size of each owner‐set
    let total_owned: usize = owners.iter().map(|s| s.lock().unwrap().len()).sum();

    // 6. Return average = total / n
    total_owned as f64 / n as f64
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
            self.edges.iter().filter_map(|&(a, b)| {
                if a == v {
                    Some(b)
                } else if b == v {
                    Some(a)
                } else {
                    None
                }
            }).collect::<Vec<_>>().into_par_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors(v).count()
        }
    }

    #[test]
    fn edge_cut_cycle() {
        // 4-cycle: 0-1-2-3-0
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            n: 4,
        };
        let mut pm = PartitionMap::with_capacity(4);
        // All in same part
        for v in 0..4 {
            pm.insert(v, 0);
        }
        assert_eq!(edge_cut(&g, &pm), 0);

        // (0,1) in part 0, (2,3) in part 1
        let mut pm2 = PartitionMap::with_capacity(4);
        pm2.insert(0, 0);
        pm2.insert(1, 0);
        pm2.insert(2, 1);
        pm2.insert(3, 1);
        assert_eq!(edge_cut(&g, &pm2), 2);
    }

    #[test]
    fn edge_cut_path() {
        // Path: 0-1-2-3
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2), (2, 3)],
            n: 4,
        };
        let mut pm = PartitionMap::with_capacity(4);
        for v in 0..3 {
            pm.insert(v, 0);
        }
        pm.insert(3, 1);
        assert_eq!(edge_cut(&g, &pm), 1);
    }

    #[test]
    fn replication_factor_trivial() {
        // Path: 0-1-2 with all in same part → RF == 1.0
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        for v in 0..3 {
            pm.insert(v, 0);
        }
        let rf = replication_factor(&g, &pm);
        assert!((rf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn replication_factor_path() {
        // Path: 0-1-2, (0,1)->0, (2)->1 → expected RF ≈ (1 + 2 + 1)/3 = 4/3
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0);
        pm.insert(1, 0);
        pm.insert(2, 1);
        let rf = replication_factor(&g, &pm);
        assert!(
            (rf - 1.3333).abs() < 2e-3,
            "replication_factor was {} (expected ~1.3333)",
            rf
        );
    }

    #[test]
    fn replication_factor_distinct() {
        // 0-1-2, all in different parts → RF ≈ (2 + 3 + 2)/3 = 7/3 ≈ 2.3333
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0);
        pm.insert(1, 1);
        pm.insert(2, 2);
        let rf = replication_factor(&g, &pm);
        assert!((rf - 2.3333).abs() < 1e-3);
    }
}

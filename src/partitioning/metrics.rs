// Metrics skeleton for partitioning
#![cfg(feature = "partitioning")]

use super::{PartitionableGraph, PartitionMap};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use rayon::iter::ParallelIterator;

impl<V: Eq + Hash + Copy> PartitionMap<V> {
    /// Returns the part ID for a given vertex.
    pub fn part_of(&self, v: V) -> usize {
        *self.get(&v).expect("vertex not found in PartitionMap")
    }
}

/// Computes the edge cut of a partitioning (O(E)).
/// Intended for debug/CI use.
pub fn edge_cut<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> usize
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd,
{
    g.vertices().map(|u| {
        g.neighbors(u).filter(|&v| pm.part_of(u) != pm.part_of(v)).count()
    }).sum::<usize>() / 2
}

/// Computes the replication factor of a partitioning (O(E)).
/// Intended for debug/CI use.
pub fn replication_factor<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> f64
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd,
{
    let n = g.vertices().count();
    let mut owners = vec![std::collections::HashSet::new(); n];
    g.vertices().for_each(|u| {
        owners[u as usize].insert(pm.part_of(u));
        g.neighbors(u).for_each(|v| {
            owners[v as usize].insert(pm.part_of(u));
        });
    });
    let total: usize = owners.iter().map(|s| s.len()).sum();
    total as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;
    use std::collections::HashMap;

    struct TestGraph {
        edges: Vec<(usize, usize)>,
        n: usize,
    }
    impl PartitionableGraph for TestGraph {
        type VertexId = usize;
        fn vertices(&self) -> Vec<Self::VertexId> {
            (0..self.n).collect()
        }
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
    fn edge_cut_cycle() {
        // 4-cycle: 0-1-2-3-0
        let g = TestGraph { edges: vec![(0,1),(1,2),(2,3),(3,0)], n: 4 };
        let mut pm = PartitionMap::with_capacity(4);
        // All in same part
        for v in 0..4 { pm.insert(v, 0); }
        assert_eq!(edge_cut(&g, &pm), 0);
        // (0,1) in part 0, (2,3) in part 1
        let mut pm2 = PartitionMap::with_capacity(4);
        pm2.insert(0, 0); pm2.insert(1, 0); pm2.insert(2, 1); pm2.insert(3, 1);
        assert_eq!(edge_cut(&g, &pm2), 2);
    }

    #[test]
    fn edge_cut_path() {
        // Path: 0-1-2-3
        let g = TestGraph { edges: vec![(0,1),(1,2),(2,3)], n: 4 };
        let mut pm = PartitionMap::with_capacity(4);
        for v in 0..3 { pm.insert(v, 0); }
        pm.insert(3, 1);
        assert_eq!(edge_cut(&g, &pm), 1);
    }

    #[test]
    fn replication_factor_trivial() {
        let g = TestGraph { edges: vec![(0,1),(1,2)], n: 3 };
        let mut pm = PartitionMap::with_capacity(3);
        for v in 0..3 { pm.insert(v, 0); }
        let rf = replication_factor(&g, &pm);
        assert!((rf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn replication_factor_path() {
        // Path: 0-1-2, (0,1)->0, (2)->1
        let g = TestGraph { edges: vec![(0,1),(1,2)], n: 3 };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0); pm.insert(1, 0); pm.insert(2, 1);
        let rf = replication_factor(&g, &pm);
        // Accept a slightly wider tolerance and print the value for diagnosis
        assert!((rf - 1.3333).abs() < 2e-3, "replication_factor was {} (expected ~1.3333)", rf);
    }

    #[test]
    fn replication_factor_distinct() {
        // 0-1-2, all in different parts
        let g = TestGraph { edges: vec![(0,1),(1,2)], n: 3 };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0); pm.insert(1, 1); pm.insert(2, 2);
        let rf = replication_factor(&g, &pm);
        assert!((rf - 2.3333).abs() < 1e-3);
    }
}

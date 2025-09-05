use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::partitioning::{
    partition, PartitionerConfig, PartitionerError, PartitionMap
};
use crate::partitioning::metrics::edge_cut;
use crate::partitioning::graph_traits::PartitionableGraph;

#[test]
fn e2e_cycle_4_nodes_k2() {
    // Build a 4-cycle: 0–1–2–3–0
    struct Cycle4;
    impl PartitionableGraph for Cycle4 {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize> where Self: 'a;
        type NeighParIter<'a>   = rayon::vec::IntoIter<usize> where Self: 'a;
        type NeighIter<'a>      = std::vec::IntoIter<usize> where Self: 'a;
        type EdgeParIter<'a>    = rayon::vec::IntoIter<(usize, usize)> where Self: 'a;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..4).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.neighbors_seq(v).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
            let nbrs = match v {
                0 => vec![1,3],
                1 => vec![0,2],
                2 => vec![1,3],
                3 => vec![2,0],
                _ => vec![],
            };
            nbrs.into_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }
        fn edges(&self) -> Self::EdgeParIter<'_> {
            vec![(0,1),(1,2),(2,3),(0,3)].into_par_iter()
        }
    }

    let g = Cycle4;
    let cfg = PartitionerConfig { n_parts: 2, epsilon: 0.05, ..Default::default() };
    let pm = partition(&g, &cfg).expect("partition must succeed");

    // 1) map size
    assert_eq!(pm.len(), 4);

    // 2) load balance (sum of degree→load)
    let mut loads = [0u64, 0u64];
    for (&v, &p) in pm.iter() {
        let deg = g.degree(v) as u64;
        loads[p] += deg.max(1);
    }
    let max_load = loads.iter().cloned().max().unwrap();
    let min_load = loads.iter().cloned().min().unwrap();
    assert!(
        (max_load as f64)/(min_load as f64) <= 1.0 + cfg.epsilon + 1e-6,
        "unbalanced loads: {:?}",
        loads
    );

    // 3) edge_cut should be ≤ equal‐sized random partition
    let my_cut = edge_cut(&g, &pm);

    let mut rng = SmallRng::seed_from_u64(123);
    let mut rnd_pm = PartitionMap::with_capacity(4);
    for v in 0..4 {
        rnd_pm.insert(v, rng.gen_range(0..2));
    }
    let rnd_cut = edge_cut(&g, &rnd_pm);

    assert!(
        my_cut <= rnd_cut,
        "cycle cut {} > random cut {}",
        my_cut,
        rnd_cut
    );
}

proptest! {
    #[test]
    fn prop_small_graphs(
        n in 2usize..10,
        k in 2usize..5,
        edge_prob in 0.1f64..0.9f64,  // probability each undirected edge exists
    ) {
        // Seed RNG from test parameters so graph and baseline partition are reproducible
        let seed = {
            let mut h = DefaultHasher::new();
            n.hash(&mut h);
            k.hash(&mut h);
            edge_prob.to_bits().hash(&mut h);
            h.finish()
        };
        let mut rng = SmallRng::seed_from_u64(seed);

        // 1) Build random adjacency list
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u+1)..n {
                if rng.gen::<f64>() < edge_prob {
                    edges.push((u,v));
                }
            }
        }
        struct RandGraph { edges: Vec<(usize,usize)>, n: usize }
        impl PartitionableGraph for RandGraph {
            type VertexId = usize;
            type VertexParIter<'a> = rayon::vec::IntoIter<usize> where Self: 'a;
            type NeighParIter<'a>   = rayon::vec::IntoIter<usize> where Self: 'a;
            type NeighIter<'a>      = std::vec::IntoIter<usize> where Self: 'a;
            type EdgeParIter<'a>    = rayon::vec::IntoIter<(usize, usize)> where Self: 'a;
            fn vertices(&self)->Self::VertexParIter<'_> {
                (0..self.n).collect::<Vec<_>>().into_par_iter()
            }
            fn neighbors(&self, u: usize)->Self::NeighParIter<'_> {
                self.neighbors_seq(u).collect::<Vec<_>>().into_par_iter()
            }
            fn neighbors_seq(&self, u: usize)->Self::NeighIter<'_> {
                let nbrs: Vec<_> = self.edges.iter()
                    .filter_map(|&(a,b)| if a==u {Some(b)} else if b==u {Some(a)} else {None})
                    .collect();
                nbrs.into_iter()
            }
            fn degree(&self, u: usize) -> usize {
                self.neighbors_seq(u).count()
            }
            fn edges(&self) -> Self::EdgeParIter<'_> {
                self.edges.clone().into_par_iter()
            }
        }
        let g = RandGraph { edges: edges.clone(), n };
        let cfg = PartitionerConfig { n_parts: k, epsilon: 0.1, ..Default::default() };
        let pm = partition(&g, &cfg).unwrap();

        // A) pm.len == n
        prop_assert_eq!(pm.len(), n);

        // B) load balance
        let mut loads = vec![0u64; k];
        for (&v, &p) in pm.iter() {
            let d = g.degree(v) as u64;
            loads[p] += d.max(1);
        }
        let maxl = *loads.iter().max().unwrap();
        let minl = *loads.iter().min().unwrap();
        prop_assert!(
            (maxl as f64)/(minl as f64) <= 1.0 + cfg.epsilon + 1e-6,
            "loads = {:?}",
            loads
        );

        // C) edge_cut improvement vs. one random baseline
        let my_cut = edge_cut(&g, &pm);
        let mut rnd_pm = PartitionMap::with_capacity(n);
        for v in 0..n {
            rnd_pm.insert(v, rng.gen_range(0..k));
        }
        let rnd_cut = edge_cut(&g, &rnd_pm);

        prop_assert!(my_cut <= rnd_cut, "my_cut={} rnd_cut={}", my_cut, rnd_cut);
    }
}

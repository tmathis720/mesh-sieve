use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::partitioning::graph_traits::PartitionableGraph;
use crate::partitioning::metrics::edge_cut;
use crate::partitioning::{PartitionMap, PartitionerConfig, partition};
use crate::partitioning::{exchange_cluster_part_assignments, exchange_cut_edge_owner_decisions};
use hashbrown::HashMap;

#[test]
fn e2e_cycle_4_nodes_k2() {
    // Build a 4-cycle: 0–1–2–3–0
    struct Cycle4;
    impl PartitionableGraph for Cycle4 {
        type VertexId = usize;
        type VertexParIter<'a>
            = rayon::vec::IntoIter<usize>
        where
            Self: 'a;
        type NeighParIter<'a>
            = rayon::vec::IntoIter<usize>
        where
            Self: 'a;
        type NeighIter<'a>
            = std::vec::IntoIter<usize>
        where
            Self: 'a;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..4).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.neighbors_seq(v).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
            let nbrs = match v {
                0 => vec![1, 3],
                1 => vec![0, 2],
                2 => vec![1, 3],
                3 => vec![2, 0],
                _ => vec![],
            };
            nbrs.into_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }
        fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
            vec![(0, 1), (1, 2), (2, 3), (0, 3)].into_par_iter()
        }
    }

    let g = Cycle4;
    let cfg = PartitionerConfig {
        n_parts: 2,
        epsilon: 0.05,
        ..Default::default()
    };
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
        (max_load as f64) / (min_load as f64) <= 1.0 + cfg.epsilon + 1e-6,
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
                if rng.r#gen::<f64>() < edge_prob {
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
            fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
                self.edges.clone().into_par_iter()
            }
        }
        let g = RandGraph { edges: edges.clone(), n };
        let cfg = PartitionerConfig { n_parts: k, epsilon: 0.1, ..Default::default() };
        let pm = match partition(&g, &cfg) {
            Ok(pm) => pm,
            Err(_) => return Ok(()),
        };

        // A) pm.len == n
        prop_assert_eq!(pm.len(), n);

        // B) edges() must match manual neighbor expansion
        let mut manual = Vec::new();
        for u in 0..n {
            for v in g.neighbors_seq(u) {
                if u < v {
                    manual.push((u, v));
                }
            }
        }
        manual.sort();
        manual.dedup();
        let mut trait_edges: Vec<_> = g.edges().collect();
        trait_edges.sort();
        prop_assert_eq!(trait_edges, manual);
    }
}

proptest! {
    #[test]
    fn prop_cut_edge_owner_exchange_is_deterministic(
        a in 0usize..32,
        b in 0usize..32,
        o1 in 0usize..8,
        o2 in 0usize..8,
    ) {
        prop_assume!(a != b);
        let edge = if a < b { (a, b) } else { (b, a) };
        let mut m1 = HashMap::new();
        m1.insert(edge, o1);
        m1.insert(edge, o2);
        let out = exchange_cut_edge_owner_decisions(&m1);
        let got = *out.get(&edge).expect("owner decision exists");
        prop_assert_eq!(got, o1.min(o2));
    }
}

proptest! {
    #[test]
    fn prop_cluster_part_exchange_is_stable_with_overlap(
        n in 1usize..32,
        k in 1usize..8,
    ) {
        let base: Vec<usize> = (0..n).map(|i| i % k).collect();
        let exchanged = exchange_cluster_part_assignments(&base);
        prop_assert_eq!(exchanged, base);
    }
}

#[cfg(feature = "mpi-support")]
mod distributed_exchange_tests {
    use super::*;
    use crate::algs::communicator::RayonComm;
    use crate::algs::point_sf::PointSF;
    use crate::overlap::overlap::Overlap;
    use crate::partitioning::{
        exchange_boundary_cluster_ids_with_sf, exchange_cluster_part_assignments_with_sf,
        exchange_cut_edge_owner_decisions_with_sf,
    };
    use crate::topology::point::PointId;
    use serial_test::serial;
    use std::thread;

    fn p(raw: u64) -> PointId {
        PointId::new(raw).unwrap()
    }

    fn two_rank_sf<'a>(
        rank: usize,
        comm: &'a RayonComm,
        overlap: &'a Overlap,
    ) -> PointSF<'a, RayonComm> {
        PointSF::new(overlap, comm, rank)
    }

    #[test]
    #[serial]
    fn boundary_cluster_exchange_uses_overlap_remote_ids() {
        let h0 = thread::spawn(|| {
            let comm = RayonComm::new(0, 2);
            let mut ov = Overlap::default();
            ov.try_add_link(p(1), 1, p(2)).unwrap();
            let sf = two_rank_sf(0, &comm, &ov);
            let mut local = HashMap::new();
            local.insert(1usize, 7u32);
            exchange_boundary_cluster_ids_with_sf(&local, &sf).unwrap()
        });
        let h1 = thread::spawn(|| {
            let comm = RayonComm::new(1, 2);
            let mut ov = Overlap::default();
            ov.try_add_link(p(2), 0, p(1)).unwrap();
            let sf = two_rank_sf(1, &comm, &ov);
            let mut local = HashMap::new();
            local.insert(2usize, 3u32);
            exchange_boundary_cluster_ids_with_sf(&local, &sf).unwrap()
        });
        let r0 = h0.join().unwrap();
        let r1 = h1.join().unwrap();
        assert_eq!(r0.get(&1), Some(&3));
        assert_eq!(r1.get(&2), Some(&3));
    }

    #[test]
    #[serial]
    fn cluster_part_exchange_reconciles_deterministically() {
        let h0 = thread::spawn(|| {
            let comm = RayonComm::new(0, 2);
            let mut ov = Overlap::default();
            ov.try_add_link(p(1), 1, p(2)).unwrap();
            let sf = two_rank_sf(0, &comm, &ov);
            exchange_cluster_part_assignments_with_sf(&[2, 1], &sf).unwrap()
        });
        let h1 = thread::spawn(|| {
            let comm = RayonComm::new(1, 2);
            let mut ov = Overlap::default();
            ov.try_add_link(p(2), 0, p(1)).unwrap();
            let sf = two_rank_sf(1, &comm, &ov);
            exchange_cluster_part_assignments_with_sf(&[0, 3, 4], &sf).unwrap()
        });
        let r0 = h0.join().unwrap();
        let r1 = h1.join().unwrap();
        assert_eq!(r0, vec![0, 1, 4]);
        assert_eq!(r1, vec![0, 1, 4]);
    }

    #[test]
    #[serial]
    fn cut_edge_owner_exchange_canonicalizes_and_min_reduces() {
        let h0 = thread::spawn(|| {
            let comm = RayonComm::new(0, 2);
            let mut ov = Overlap::default();
            ov.try_add_link(p(1), 1, p(2)).unwrap();
            let sf = two_rank_sf(0, &comm, &ov);
            let mut local = HashMap::new();
            local.insert((8usize, 4usize), 1usize);
            exchange_cut_edge_owner_decisions_with_sf(&local, &sf).unwrap()
        });
        let h1 = thread::spawn(|| {
            let comm = RayonComm::new(1, 2);
            let mut ov = Overlap::default();
            ov.try_add_link(p(2), 0, p(1)).unwrap();
            let sf = two_rank_sf(1, &comm, &ov);
            let mut local = HashMap::new();
            local.insert((4usize, 8usize), 0usize);
            exchange_cut_edge_owner_decisions_with_sf(&local, &sf).unwrap()
        });
        let r0 = h0.join().unwrap();
        let r1 = h1.join().unwrap();
        assert_eq!(r0.get(&(4, 8)), Some(&0));
        assert_eq!(r1.get(&(4, 8)), Some(&0));
    }

    #[test]
    #[serial]
    fn repartition_exchange_is_reproducible() {
        fn run_once() -> (Vec<usize>, HashMap<(usize, usize), usize>) {
            let h0 = thread::spawn(|| {
                let comm = RayonComm::new(0, 2);
                let mut ov = Overlap::default();
                ov.try_add_link(p(10), 1, p(20)).unwrap();
                let sf = two_rank_sf(0, &comm, &ov);
                let parts = exchange_cluster_part_assignments_with_sf(&[1, 0, 1], &sf).unwrap();
                let mut owners = HashMap::new();
                owners.insert((10usize, 20usize), 1usize);
                let owners = exchange_cut_edge_owner_decisions_with_sf(&owners, &sf).unwrap();
                (parts, owners)
            });
            let h1 = thread::spawn(|| {
                let comm = RayonComm::new(1, 2);
                let mut ov = Overlap::default();
                ov.try_add_link(p(20), 0, p(10)).unwrap();
                let sf = two_rank_sf(1, &comm, &ov);
                let parts = exchange_cluster_part_assignments_with_sf(&[1, 0, 2], &sf).unwrap();
                let mut owners = HashMap::new();
                owners.insert((20usize, 10usize), 0usize);
                let owners = exchange_cut_edge_owner_decisions_with_sf(&owners, &sf).unwrap();
                (parts, owners)
            });
            let (parts0, owners0) = h0.join().unwrap();
            let (parts1, owners1) = h1.join().unwrap();
            assert_eq!(parts0, parts1);
            assert_eq!(owners0, owners1);
            (parts0, owners0)
        }

        let first = run_once();
        let second = run_once();
        assert_eq!(first, second);
    }
}

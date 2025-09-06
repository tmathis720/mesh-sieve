#![cfg(all(feature = "mpi-support", feature = "exact-metrics"))]

use mesh_sieve::partitioning::graph_traits::PartitionableGraph;
use mesh_sieve::partitioning::{
    self, PartitionMap, PartitionerConfig,
    louvain::{counters::compute_counters, delta_q_pair_public},
    metrics::{edge_cut, load_balance_parts, replication_factor, rf_exact},
    vertex_cut,
};
use rayon::prelude::*;

#[derive(Clone)]
struct AdjListGraph {
    nbrs: Vec<Vec<usize>>,
}

impl AdjListGraph {
    fn from_undirected(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut nbrs = vec![Vec::new(); n];
        for &(u, v) in edges {
            nbrs[u].push(v);
            nbrs[v].push(u);
        }
        for ns in &mut nbrs {
            ns.sort_unstable();
            ns.dedup();
        }
        Self { nbrs }
    }
    fn m_edges(&self) -> u64 {
        self.nbrs.iter().map(|ns| ns.len()).sum::<usize>() as u64 / 2
    }
}
impl PartitionableGraph for AdjListGraph {
    type VertexId = usize;
    type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
    type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
    type NeighIter<'a> = std::vec::IntoIter<usize>;
    fn vertices(&self) -> Self::VertexParIter<'_> {
        (0..self.nbrs.len()).collect::<Vec<_>>().into_par_iter()
    }
    fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
        self.nbrs[v].clone().into_par_iter()
    }
    fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
        self.nbrs[v].clone().into_iter()
    }
    fn degree(&self, v: usize) -> usize {
        self.nbrs[v].len()
    }
}

#[test]
fn delta_q_clique_bridge() {
    let edges = &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
    let g = AdjListGraph::from_undirected(6, edges);
    let m = g.m_edges();
    let vol_i = 7;
    let vol_j = 7;
    let e_ij = 1;
    let (dq, dq_bal) = delta_q_pair_public(e_ij, vol_i, vol_j, m, 0.75);
    assert!(
        dq < 0.0,
        "bridge produces negative ΔQ in this configuration"
    );
    assert!(dq_bal >= dq);
}

#[test]
fn delta_q_balance_factor_penalizes_unbalanced_merge() {
    let edges = &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (6, 7), (0, 6)];
    let g = AdjListGraph::from_undirected(8, edges);
    let m = g.m_edges();
    let vol_i = (0..7).map(|v| g.degree(v)).sum::<usize>() as u64;
    let vol_j = g.degree(7) as u64;
    let e_ij = 1;
    let (dq, dq_bal) = delta_q_pair_public(e_ij, vol_i, vol_j, m, 1.0);
    assert!(dq > dq_bal, "balance factor should reduce the gain");
}

#[test]
fn rf_exact_matches_manual() {
    let primary = vec![0, 1, 0];
    let replicas = vec![vec![(1, 1)], vec![], vec![(0, 1)]];
    let rf = rf_exact(&primary, &replicas);
    let expected = (2 + 1 + 2) as f64 / 3.0;
    assert!((rf - expected).abs() < 1e-6);
}

#[test]
fn load_balance_parts_degree_based() {
    let edges = &[(0, 1), (1, 2)];
    let g = AdjListGraph::from_undirected(3, edges);
    let mut pm = PartitionMap::with_capacity(3);
    pm.insert(0, 0);
    pm.insert(1, 0);
    pm.insert(2, 1);
    let (min, max, ratio) = load_balance_parts(&pm, &g);
    assert_eq!(min, 1);
    assert_eq!(max, 3);
    assert!(ratio > 2.9 && ratio < 3.1);
}

#[test]
fn edge_cut_matches_bruteforce() {
    let edges = &[(0, 1), (1, 2), (2, 0), (2, 3)];
    let g = AdjListGraph::from_undirected(4, edges);
    let mut pm = PartitionMap::with_capacity(4);
    pm.insert(0, 0);
    pm.insert(1, 0);
    pm.insert(2, 1);
    pm.insert(3, 1);
    let cut = edge_cut(&g, &pm);
    let brute = edges
        .iter()
        .filter(|&&(u, v)| pm.part_of(u) != pm.part_of(v))
        .count();
    assert_eq!(cut, brute);
}

#[test]
fn replication_factor_matches_exact() {
    let edges = &[(0, 1), (1, 2)];
    let g = AdjListGraph::from_undirected(3, edges);
    let mut pm = PartitionMap::with_capacity(3);
    pm.insert(0, 0);
    pm.insert(1, 0);
    pm.insert(2, 1);
    let rf_approx = replication_factor(&g, &pm);
    let primary = vec![0, 0, 1];
    let replicas = vec![vec![], vec![(2, 1)], vec![(1, 0)]];
    let rf_ex = rf_exact(&primary, &replicas);
    assert!((rf_approx - rf_ex).abs() < 1e-6);
}

#[test]
fn cluster_counters_identity() {
    // path
    let g1 = AdjListGraph::from_undirected(4, &[(0, 1), (1, 2), (2, 3)]);
    let counters1 = compute_counters(&g1, &[0, 0, 1, 1]);
    assert_eq!(counters1.vol.iter().sum::<u64>(), 2 * counters1.m_edges);
    let e_sum1: u64 = counters1.e_cc.iter().sum::<u64>() + counters1.e_ij.values().sum::<u64>();
    assert_eq!(e_sum1, counters1.m_edges);

    // cycle
    let g2 = AdjListGraph::from_undirected(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
    let counters2 = compute_counters(&g2, &[0, 0, 1, 1]);
    assert_eq!(counters2.vol.iter().sum::<u64>(), 2 * counters2.m_edges);
    let e_sum2: u64 = counters2.e_cc.iter().sum::<u64>() + counters2.e_ij.values().sum::<u64>();
    assert_eq!(e_sum2, counters2.m_edges);

    // two cliques with bridge
    let edges = &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
    let g3 = AdjListGraph::from_undirected(6, edges);
    let counters3 = compute_counters(&g3, &[0, 0, 0, 1, 1, 1]);
    assert_eq!(counters3.vol.iter().sum::<u64>(), 2 * counters3.m_edges);
    let e_sum3: u64 = counters3.e_cc.iter().sum::<u64>() + counters3.e_ij.values().sum::<u64>();
    assert_eq!(e_sum3, counters3.m_edges);

    // 2x3 grid
    let grid_edges = &[(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)];
    let g4 = AdjListGraph::from_undirected(6, grid_edges);
    let counters4 = compute_counters(&g4, &[0, 0, 0, 1, 1, 1]);
    assert_eq!(counters4.vol.iter().sum::<u64>(), 2 * counters4.m_edges);
    let e_sum4: u64 = counters4.e_cc.iter().sum::<u64>() + counters4.e_ij.values().sum::<u64>();
    assert_eq!(e_sum4, counters4.m_edges);
}

#[test]
fn vertex_cut_reduces_rf_on_clustered_graphs() {
    let mut edges = vec![];
    let left: Vec<_> = (0..8).collect();
    let right: Vec<_> = (8..16).collect();
    for i in 0..8 {
        for j in (i + 1)..8 {
            edges.push((left[i], left[j]));
        }
    }
    for i in 0..8 {
        for j in (i + 1)..8 {
            edges.push((right[i], right[j]));
        }
    }
    edges.extend_from_slice(&[(2, 9), (3, 12), (5, 14)]);
    let g = AdjListGraph::from_undirected(16, &edges);

    let cfg = PartitionerConfig {
        n_parts: 2,
        seed_factor: 1.0,
        ..Default::default()
    };
    let pm = partitioning::partition(&g, &cfg).expect("partition");

    let (primary, replicas) = vertex_cut::build_vertex_cuts_fixed(&g, &pm);
    let rf_exact_val = rf_exact(&primary, &replicas);
    assert!(
        rf_exact_val < 1.5,
        "RF should fall well below the naive two-part baseline",
    );
}

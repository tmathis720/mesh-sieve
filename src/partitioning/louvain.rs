//! Louvain-style clustering for distributed graph partitioning.
//!
//! ## Objective (Newman modularity with balance factor)
//!
//! We maximize the *balanced modularity gain* when merging clusters `i` and `j`:
//!
//! Let
//! - `m` = number of *undirected* edges in the graph (unit: edges).
//! - `vol_c` = sum of degrees of vertices in cluster `c` (unit: stubs; `∑_c vol_c = 2m`).
//! - `e_rs` = number of undirected edges with one endpoint in cluster `r` and the other in cluster `s` (unit: edges).
//!
//! Define the *fractional* quantities:
//! - `E_rs = e_rs / (2m)`,
//! - `a_c = vol_c / (2m)`.
//!
//! Classical modularity can be written as `Q = ∑_c (E_cc - a_c^2)`.
//! The modularity *gain* from merging clusters `i` and `j` is:
//!
//!     ΔQ_ij = 2 * (E_ij - a_i * a_j)
//!
//! In terms of raw counts, this is equivalently:
//!
//!     ΔQ_ij = (e_ij / m) - (vol_i * vol_j) / (2 m^2)
//!
//! We then apply a *size-balance factor* `f_ij = min(vol_i, vol_j) / max(vol_i, vol_j)`
//! and maximize `ΔQ'_ij = α * ΔQ_ij * f_ij`, where `α ∈ [0,1]` is a configuration parameter
//! controlling the strength of balance vs. modularity improvement.
//!
//! This module provides a simple Louvain-style community detection algorithm for use in
//! parallel and distributed partitioning. The clustering is controlled by [`PartitionerConfig`]
//! and is intended for use with graphs implementing [`PartitionableGraph`].
//!
//! The main entry point is [`louvain_cluster`], which returns a vector of cluster IDs for each vertex.

use crate::partitioning::PartitionerConfig;
use crate::partitioning::graph_traits::PartitionableGraph;
use hashbrown::HashMap as FastMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub mod counters;

/// Cluster statistics maintained during clustering.
#[derive(Debug, Clone)]
struct ClusterInfo {
    /// Sum of degrees of vertices in this cluster.
    volume: u64,
    /// Number of vertices in this cluster.
    size: u32,
    /// Number of internal edges (for logging/metrics).
    inner_edges: u64,
}

#[cfg(feature = "exact-metrics")]
#[inline]
pub fn delta_q_pair(e_ij: u64, vol_i: u64, vol_j: u64, m_edges: u64, alpha: f64) -> (f64, f64) {
    if m_edges == 0 {
        return (0.0, 0.0);
    }
    let m = m_edges as f64;
    let e_term = (e_ij as f64) / m;
    let a_term = (vol_i as f64) * (vol_j as f64) / (2.0 * m * m);
    let dq = e_term - a_term;
    let f = (vol_i.min(vol_j) as f64) / (vol_i.max(vol_j).max(1) as f64);
    let dq_bal = alpha * dq * f;
    (dq, dq_bal)
}

#[cfg(not(feature = "exact-metrics"))]
#[inline]
pub(crate) fn delta_q_pair(
    e_ij: u64,
    vol_i: u64,
    vol_j: u64,
    m_edges: u64,
    alpha: f64,
) -> (f64, f64) {
    if m_edges == 0 {
        return (0.0, 0.0);
    }
    let m = m_edges as f64;
    let e_term = (e_ij as f64) / m;
    let a_term = (vol_i as f64) * (vol_j as f64) / (2.0 * m * m);
    let dq = e_term - a_term;
    let f = (vol_i.min(vol_j) as f64) / (vol_i.max(vol_j).max(1) as f64);
    let dq_bal = alpha * dq * f;
    (dq, dq_bal)
}

#[cfg(feature = "exact-metrics")]
pub use delta_q_pair as delta_q_pair_public;

/// Perform Louvain-style clustering on a partitionable graph.
///
/// The gain in modularity for merging clusters `i` and `j` is computed as
///
/// `ΔQ(i,j) = (E_ij / m) - (vol_i * vol_j) / (2 m^2)`
///
/// where `m` is the number of undirected edges returned by [`PartitionableGraph::edges`]
/// and `vol_*` is the sum of degrees in each cluster.  A balanced variant is then applied:
///
/// `ΔQ_bal = α * ΔQ(i,j) * min{vol_i/vol_j, vol_j/vol_i}`
///
/// Only merges with positive `ΔQ_bal` are accepted.  The returned vector contains the
/// final cluster IDs for each vertex (indexed by vertex order in `graph.vertices()`),
/// remapped to a dense range.
///
/// # Arguments
/// - `graph`: The input graph.
/// - `cfg`: Configuration parameters for clustering.
///
/// # Parallelism
/// Each iteration performs a parallel fold over `graph.edges()`, making the algorithm
/// suitable for moderately sized graphs.
///
/// # Features
/// Only available with the `mpi-support` feature enabled.
pub fn louvain_cluster<G>(graph: &G, cfg: &PartitionerConfig) -> Vec<u32>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    // Collect vertices and degrees
    let verts: Vec<usize> = graph.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return Vec::new();
    }
    let idx_map: HashMap<usize, usize> = verts
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    let degrees: Vec<u64> = verts.iter().map(|&u| graph.degree(u) as u64).collect();

    // Number of undirected edges
    let m_edges: u64 = graph.edges().count() as u64;
    if m_edges == 0 {
        return (0..n).map(|i| i as u32).collect();
    }

    // Initial cluster assignments
    let cluster_ids: Vec<AtomicU32> = (0..n).map(|i| AtomicU32::new(i as u32)).collect();
    let mut clusters: FastMap<u32, ClusterInfo> = FastMap::with_capacity(n);
    let mut members: FastMap<u32, Vec<usize>> = FastMap::with_capacity(n);
    for i in 0..n {
        let vid = i as u32;
        clusters.insert(
            vid,
            ClusterInfo {
                volume: degrees[i],
                size: 1,
                inner_edges: 0,
            },
        );
        members.insert(vid, vec![i]);
    }

    fn make_key(a: u32, b: u32) -> (u32, u32) {
        if a < b { (a, b) } else { (b, a) }
    }

    for _iter in 0..cfg.max_iters {
        // Build inter-cluster edge counts in parallel
        let cluster_adj: FastMap<(u32, u32), u64> = graph
            .edges()
            .fold(
                || FastMap::<(u32, u32), u64>::new(),
                |mut local, (u, v)| {
                    let iu = *idx_map.get(&u).unwrap();
                    let iv = *idx_map.get(&v).unwrap();
                    let cu = cluster_ids[iu].load(Ordering::Relaxed);
                    let cv = cluster_ids[iv].load(Ordering::Relaxed);
                    if cu != cv {
                        *local.entry(make_key(cu, cv)).or_insert(0) += 1;
                    }
                    local
                },
            )
            .reduce(
                || FastMap::<(u32, u32), u64>::new(),
                |mut a, b| {
                    for (k, w) in b {
                        *a.entry(k).or_insert(0) += w;
                    }
                    a
                },
            );

        // Find best positive merge
        let mut best: Option<(u32, u32, f64, u64)> = None;
        for (&(ci, cj), &e_ij) in cluster_adj.iter() {
            let (Some(info_i), Some(info_j)) = (clusters.get(&ci), clusters.get(&cj)) else {
                continue;
            };
            let (_, dq_bal) = delta_q_pair(e_ij, info_i.volume, info_j.volume, m_edges, cfg.alpha);
            if dq_bal > 0.0 {
                match best {
                    Some((_, _, best_dq, _)) if dq_bal <= best_dq => {}
                    _ => best = Some((ci, cj, dq_bal, e_ij)),
                }
            }
        }

        let Some((ci, cj, _dq, e_ij)) = best else {
            break;
        };

        // Merge cj into ci
        if let Some(mut vec_j) = members.remove(&cj) {
            if let Some(vec_i) = members.get_mut(&ci) {
                for &v in &vec_j {
                    cluster_ids[v].store(ci, Ordering::Relaxed);
                }
                vec_i.append(&mut vec_j);
            }
        }

        if let Some(info_j) = clusters.remove(&cj) {
            if let Some(info_i) = clusters.get_mut(&ci) {
                info_i.volume = info_i.volume.saturating_add(info_j.volume);
                info_i.size = info_i.size.saturating_add(info_j.size);
                info_i.inner_edges = info_i
                    .inner_edges
                    .saturating_add(info_j.inner_edges)
                    .saturating_add(e_ij);
            }
        }

        if clusters.len() as u32 <= (cfg.seed_factor * cfg.n_parts as f64).ceil() as u32 {
            break;
        }
    }

    // Remap cluster IDs to dense range
    let mut unique: Vec<u32> = clusters.keys().copied().collect();
    unique.sort_unstable();
    let mut remap: FastMap<u32, u32> = FastMap::with_capacity(unique.len());
    for (new_id, old_id) in unique.iter().enumerate() {
        remap.insert(*old_id, new_id as u32);
    }
    let final_clusters: Vec<u32> = (0..n)
        .map(|i| {
            let old = cluster_ids[i].load(Ordering::Relaxed);
            *remap.get(&old).expect("stale cluster id")
        })
        .collect();
    final_clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;
    use rayon::iter::IntoParallelIterator;
    struct PathGraph {
        n: usize,
    }
    impl PathGraph {
        fn new(n: usize) -> Self {
            PathGraph { n }
        }
    }
    impl PartitionableGraph for PathGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighIter<'a> = std::vec::IntoIter<usize>;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.neighbors_seq(v).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
            let mut neigh = Vec::new();
            if v > 0 {
                neigh.push(v - 1);
            }
            if v + 1 < self.n {
                neigh.push(v + 1);
            }
            neigh.into_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }
        fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
            (0..self.n.saturating_sub(1))
                .map(|i| (i, i + 1))
                .collect::<Vec<_>>()
                .into_par_iter()
        }
    }
    #[test]
    fn test_louvain_path_graph_small() {
        let pg = PathGraph::new(10);
        let cfg = PartitionerConfig {
            n_parts: 2,
            alpha: 0.5,
            seed_factor: 2.0,
            rng_seed: 123,
            max_iters: 10,
            epsilon: 0.05,
            enable_phase1: true,
            enable_phase2: true,
            enable_phase3: true,
        };
        let clustering = louvain_cluster(&pg, &cfg);
        let unique_clusters: Vec<u32> = {
            let mut tmp = clustering.clone();
            tmp.sort_unstable();
            tmp.dedup();
            tmp
        };
        let allowed = ((cfg.seed_factor * cfg.n_parts as f64).ceil() as u32).max(1);
        assert!(
            (unique_clusters.len() as u32 <= pg.n as u32)
                && (unique_clusters.len() as u32 >= allowed),
            "Unexpected number of clusters: {} (expected between {} and {}, clustering = {:?})",
            unique_clusters.len(),
            allowed,
            pg.n,
            clustering
        );
    }
    #[test]
    fn test_no_merge_if_alpha_zero() {
        let pg = PathGraph::new(5);
        let cfg = PartitionerConfig {
            n_parts: 1,
            alpha: 0.0,
            seed_factor: 1.0,
            rng_seed: 42,
            max_iters: 5,
            epsilon: 0.05,
            enable_phase1: true,
            enable_phase2: true,
            enable_phase3: true,
        };
        let clustering = louvain_cluster(&pg, &cfg);
        assert_eq!(clustering.len(), 5);
        let unique = {
            let mut tmp = clustering.clone();
            tmp.sort_unstable();
            tmp.dedup();
            tmp
        };
        assert_eq!(unique.len(), 5);
    }

    struct EmptyGraph {
        n: usize,
    }

    impl PartitionableGraph for EmptyGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighIter<'a> = std::vec::IntoIter<usize>;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, _v: usize) -> Self::NeighParIter<'_> {
            Vec::new().into_par_iter()
        }
        fn neighbors_seq(&self, _v: usize) -> Self::NeighIter<'_> {
            Vec::new().into_iter()
        }
        fn degree(&self, _v: usize) -> usize {
            0
        }
        fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
            Vec::new().into_par_iter()
        }
    }

    #[test]
    fn test_edgeless_graph_identity() {
        let g = EmptyGraph { n: 4 };
        let cfg = PartitionerConfig {
            n_parts: 2,
            alpha: 0.5,
            seed_factor: 1.0,
            rng_seed: 0,
            max_iters: 5,
            epsilon: 0.05,
            enable_phase1: true,
            enable_phase2: true,
            enable_phase3: true,
        };
        let clustering = louvain_cluster(&g, &cfg);
        let expected: Vec<u32> = (0..g.n).map(|i| i as u32).collect();
        assert_eq!(clustering, expected);
    }
}

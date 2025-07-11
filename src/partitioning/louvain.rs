//! Louvain-style clustering for distributed graph partitioning.
//!
//! This module provides a simple Louvain-style community detection algorithm for use in
//! parallel and distributed partitioning. The clustering is controlled by [`PartitionerConfig`]
//! and is intended for use with graphs implementing [`PartitionableGraph`].
//!
//! The main entry point is [`louvain_cluster`], which returns a vector of cluster IDs for each vertex.

#![cfg(feature = "mpi-support")]


use crate::partitioning::PartitionerConfig;
use crate::partitioning::graph_traits::PartitionableGraph;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Internal cluster representation for Louvain clustering.
#[derive(Debug, Clone)]
struct Cluster {
    /// Cluster ID.
    id: u32,
    /// Total volume (sum of degrees) of the cluster.
    volume: u64,
    /// Number of internal edges (not currently used).
    internal_edges: u64,
}

impl Cluster {
    /// Create a new cluster with the given ID and initial volume.
    fn new(v: u32, deg: u64) -> Self {
        Cluster {
            id: v,
            volume: deg,
            internal_edges: 0,
        }
    }
}

/// Perform Louvain-style clustering on a partitionable graph.
///
/// Returns a vector of cluster IDs (one per vertex), with IDs remapped to a contiguous range.
///
/// # Arguments
/// - `graph`: The input graph.
/// - `cfg`: Configuration parameters for clustering.
///
/// # Returns
/// A vector of cluster IDs for each vertex (indexed by vertex order in `graph.vertices()`).
///
/// # Parallelism
/// This implementation is not fully parallel, but is suitable for moderate-sized graphs.
///
/// # Features
/// Only available with the `mpi-support` feature enabled.
pub fn louvain_cluster<G>(graph: &G, cfg: &PartitionerConfig) -> Vec<u32>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    // Collect all vertex IDs and build an index map
    let verts: Vec<usize> = graph.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return Vec::new();
    }
    let idx_map: HashMap<usize, usize> = verts.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    let degrees: Vec<u64> = verts.iter().map(|&u| graph.degree(u) as u64).collect();
    // Reconstruct edges from neighbors
    let mut all_edges = Vec::new();
    for &u in &verts {
        for v in graph.neighbors(u).collect::<Vec<_>>() {
            if u < v {
                all_edges.push((u, v));
            }
        }
    }
    let m_f64: f64 = (all_edges.len() as u64 / 2) as f64;
    let cluster_ids: Vec<AtomicU32> = (0..n).map(|u| AtomicU32::new(u as u32)).collect();
    let mut clusters: HashMap<u32, Cluster> = HashMap::with_capacity(n);
    for u in 0..n {
        let vid = u as u32;
        clusters.insert(vid, Cluster::new(vid, degrees[u]));
    }
    for _iter in 0..cfg.max_iters {
        let mut intercluster_edges: HashMap<(u32, u32), u64> = HashMap::new();
        for &(u, v) in &all_edges {
            let iu = idx_map[&u];
            let iv = idx_map[&v];
            if iu < iv {
                let cu = cluster_ids[iu].load(Ordering::Relaxed);
                let cv = cluster_ids[iv].load(Ordering::Relaxed);
                if cu != cv {
                    let key = if cu < cv { (cu, cv) } else { (cv, cu) };
                    *intercluster_edges.entry(key).or_insert(0) += 1;
                }
            }
        }
        let mut best_pair: Option<(u32, u32, f64)> = None;
        for (&(ci, cj), &e_ij) in intercluster_edges.iter() {
            let cl_i = clusters.get(&ci).unwrap();
            let cl_j = clusters.get(&cj).unwrap();
            let vol_i = cl_i.volume as f64;
            let vol_j = cl_j.volume as f64;
            let delta_mod = (e_ij as f64 / m_f64) - (vol_i * vol_j) / (2.0 * m_f64 * m_f64);
            let f_factor = vol_i.min(vol_j) / vol_i.max(vol_j);
            let delta_bal = cfg.alpha * delta_mod * f_factor;
            if delta_bal > 0.0 {
                match best_pair {
                    Some((_, _, best_val)) if delta_bal <= best_val => {}
                    _ => {
                        best_pair = Some((ci, cj, delta_bal));
                    }
                }
            }
        }
        let (merge_i, merge_j) = if let Some((ci, cj, _)) = best_pair {
            (ci, cj)
        } else {
            break;
        };
        for u in 0..n {
            if cluster_ids[u].load(Ordering::Relaxed) == merge_j {
                cluster_ids[u].store(merge_i, Ordering::Relaxed);
            }
        }
        clusters.clear();
        for u in 0..n {
            let cid = cluster_ids[u].load(Ordering::Relaxed);
            let entry = clusters.entry(cid).or_insert_with(|| Cluster {
                id: cid,
                volume: 0,
                internal_edges: 0,
            });
            entry.volume += degrees[u];
        }
        if clusters.len() as u32 <= ((cfg.seed_factor * cfg.n_parts as f64).ceil() as u32).max(1) {
            break;
        }
    }
    let unique_ids: Vec<u32> = {
        let mut tmp: Vec<u32> = (0..n)
            .map(|u| cluster_ids[u].load(Ordering::Relaxed))
            .collect();
        tmp.sort_unstable();
        tmp.dedup();
        tmp
    };
    let mut remap: HashMap<u32, u32> = HashMap::with_capacity(unique_ids.len());
    for (new_id, &old_id) in unique_ids.iter().enumerate() {
        remap.insert(old_id, new_id as u32);
    }
    let final_clusters: Vec<u32> = (0..n)
        .map(|u| {
            let old = cluster_ids[u].load(Ordering::Relaxed);
            *remap.get(&old).unwrap()
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
        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            let mut neigh = Vec::new();
            if v > 0 {
                neigh.push(v - 1);
            }
            if v + 1 < self.n {
                neigh.push(v + 1);
            }
            neigh.into_par_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors(v).count()
        }
    }
    impl PathGraph {
        fn edges_serial(&self) -> Vec<(usize, usize)> {
            let mut edges = Vec::with_capacity((self.n - 1) * 2);
            for i in 0..(self.n - 1) {
                edges.push((i, i + 1));
                edges.push((i + 1, i));
            }
            edges
        }
        fn edges(&self) -> impl ParallelIterator<Item = (usize, usize)> {
            let e = self.edges_serial();
            e.into_par_iter()
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
}

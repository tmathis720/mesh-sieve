#![cfg(feature = "partitioning")]

use crate::partitioning::PartitionerConfig;
use crate::partitioning::graph_traits::PartitionableGraph;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug, Clone)]
struct Cluster {
    id: u32,
    volume: u64,
    internal_edges: u64,
}

impl Cluster {
    fn new(v: u32, deg: u64) -> Self {
        Cluster {
            id: v,
            volume: deg,
            internal_edges: 0,
        }
    }
}

pub fn louvain_cluster<G>(graph: &G, cfg: &PartitionerConfig) -> Vec<u32>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    let n: usize = graph.vertices().count();
    if n == 0 {
        return Vec::new();
    }
    let degrees: Vec<u64> = graph.vertices().map(|u| graph.degree(u) as u64).collect();
    // NOTE: PartitionableGraph does not require an edges() method, so we reconstruct edges from neighbors.
    let all_edges: Vec<(usize, usize)> = graph
        .vertices()
        .flat_map(|u| {
            graph
                .neighbors(u)
                .filter_map(move |v| if u < v { Some((u, v)) } else { None })
        })
        .collect();
    let m_f64: f64 = (all_edges.len() as u64 / 2) as f64;
    let cluster_ids: Vec<AtomicU32> = (0..n).map(|u| AtomicU32::new(u as u32)).collect();
    let mut clusters: HashMap<u32, Cluster> = HashMap::with_capacity(n);
    for u in 0..n {
        let vid = u as u32;
        clusters.insert(vid, Cluster::new(vid, degrees[u]));
    }
    let mut current_cluster_count: u32 = n as u32;
    let seed_limit: u32 = ((cfg.seed_factor * cfg.n_parts as f64).ceil() as u32).max(1);
    for _iter in 0..cfg.max_iters {
        let mut intercluster_edges: HashMap<(u32, u32), u64> = HashMap::new();
        for &(u, v) in &all_edges {
            if u < v {
                let cu = cluster_ids[u].load(Ordering::Relaxed);
                let cv = cluster_ids[v].load(Ordering::Relaxed);
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
        current_cluster_count = clusters.len() as u32;
        if current_cluster_count <= seed_limit {
            break;
        }
    }
    let mut unique_ids: Vec<u32> = {
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
        };
        let clustering = louvain_cluster(&pg, &cfg);
        let mut unique_clusters: Vec<u32> = {
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
        };
        let clustering = louvain_cluster(&pg, &cfg);
        assert_eq!(clustering.len(), 5);
        let mut unique = clustering.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 5);
    }
}

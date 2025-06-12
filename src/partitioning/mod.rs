//! Entry-point for native graph partitioning.
#![cfg_attr(not(feature = "partitioning"), allow(dead_code, unused_imports))]

#[cfg(feature = "partitioning")]
pub mod binpack;
#[cfg(feature = "partitioning")]
pub mod graph_traits;
#[cfg(feature = "partitioning")]
pub mod louvain;
#[cfg(feature = "partitioning")]
pub mod metrics;
#[cfg(feature = "partitioning")]
pub mod parallel;
#[cfg(feature = "partitioning")]
pub mod seed_select;
#[cfg(feature = "partitioning")]
pub mod vertex_cut;

#[cfg(feature = "partitioning")]
pub use self::metrics::*;

#[cfg(feature = "partitioning")]
use hashbrown::HashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::hash::Hash;

#[cfg(feature = "partitioning")]
pub type PartitionId = usize;

#[cfg(feature = "partitioning")]
#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    pub n_parts: usize,
    pub alpha: f64,
    pub seed_factor: f64,
    pub rng_seed: u64,
    pub max_iters: usize,
}

#[cfg(feature = "partitioning")]
impl Default for PartitionerConfig {
    fn default() -> Self {
        Self {
            n_parts: 2,
            alpha: 0.75,
            seed_factor: 4.0,
            rng_seed: 42,
            max_iters: 20,
        }
    }
}

#[cfg(feature = "partitioning")]
#[derive(Debug, Clone)]
pub struct PartitionMap<V: Eq + Hash + Copy>(HashMap<V, PartitionId>);

#[cfg(feature = "partitioning")]
impl<V: Eq + Hash + Copy> PartitionMap<V> {
    pub fn with_capacity(cap: usize) -> Self {
        Self(HashMap::with_capacity(cap))
    }
    pub fn insert(&mut self, v: V, p: PartitionId) {
        self.0.insert(v, p);
    }
    pub fn get(&self, v: &V) -> Option<&PartitionId> {
        self.0.get(v)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&V, &PartitionId)> {
        self.0.iter()
    }
}

#[cfg(feature = "partitioning")]
use crate::partitioning::graph_traits::PartitionableGraph;

#[cfg(feature = "partitioning")]
#[derive(Debug)]
pub enum PartitionerError {
    MaxIter,
    // Add more as needed
}

#[cfg(feature = "partitioning")]
pub fn partition<G>(
    graph: &G,
    cfg: &PartitionerConfig,
) -> Result<PartitionMap<G::VertexId>, PartitionerError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    use crate::partitioning::{
        binpack::Item,
        binpack::merge_clusters_into_parts,
        binpack::partition_clusters,
        louvain::louvain_cluster,
        metrics::{edge_cut, replication_factor},
        vertex_cut::build_vertex_cuts,
    };

    let verts: Vec<_> = graph.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return Ok(PartitionMap::with_capacity(0));
    }

    // 1. Louvain clustering
    let clusters = louvain_cluster(graph, cfg);
    let n_clusters = clusters.iter().copied().max().unwrap_or(0) as usize + 1;

    // ————————————————————————————————————————————————
    // 1. Gather all undirected edges u<v
    // ————————————————————————————————————————————————
    let all_edges: Vec<(usize, usize)> = graph
        .vertices()
        .flat_map(|u| {
            graph
                .neighbors(u)
                .filter_map(move |v| if u < v { Some((u, v)) } else { None })
        })
        .collect();

    // ————————————————————————————————————————————————
    // 2. Build a map (cid_a, cid_b) → number of edges between them
    // ————————————————————————————————————————————————
    let mut cluster_adj: HashMap<(u32, u32), u64> =
        HashMap::with_capacity(all_edges.len());
    for (u, v) in all_edges {
        let cu = clusters[u];
        let cv = clusters[v];
        if cu != cv {
            // keep key ordered so (2,5) and (5,2) fold together
            let key = if cu < cv { (cu, cv) } else { (cv, cu) };
            *cluster_adj.entry(key).or_insert(0) += 1;
        }
    }

    // 3. Build cluster -> vertices map and cluster loads
    let mut cluster_to_verts: HashMap<u32, Vec<usize>> = HashMap::with_capacity(n_clusters);
    let mut cluster_loads: HashMap<u32, u64> = HashMap::with_capacity(n_clusters);
    for (v, &cid) in verts.iter().zip(clusters.iter()) {
        cluster_to_verts.entry(cid).or_default().push(*v);
        let deg = graph.degree(*v) as u64;
        *cluster_loads.entry(cid).or_insert(0) += deg.max(1); // avoid zero-load clusters
    }

    // 4. Binpack clusters into parts, with adjacency
    let items: Vec<Item> = cluster_to_verts
        .iter()
        .map(|(&cid, _)| {
            let load = *cluster_loads.get(&cid).unwrap_or(&1);

            // build adjacency list for this cluster id
            let mut adj = Vec::new();
            for (&(a, b), &count) in &cluster_adj {
                if a == cid {
                    adj.push((b as usize, count));
                } else if b == cid {
                    adj.push((a as usize, count));
                }
            }

            Item {
                cid: cid as usize,
                load,
                adj,
            }
        })
        .collect();

    let cluster_part = merge_clusters_into_parts(&items, cfg.n_parts);

    // 5. Assign each vertex to its cluster's part
    let mut pm = PartitionMap::with_capacity(n);
    for (i, (_cid, verts)) in cluster_to_verts.iter().enumerate() {
        let part = cluster_part[i];
        for &v in verts {
            pm.insert(v, part);
        }
    }

    // 6. Vertex cut construction (primary owners, replicas)
    let _vertex_cut = build_vertex_cuts(graph, &pm, cfg.rng_seed);

    // 7. Metrics (for debug/logging)
    let _cut = edge_cut(graph, &pm);
    let _rep = replication_factor(graph, &pm);
    // Optionally: log or print metrics here

    Ok(pm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::iter::IntoParallelIterator;
    #[derive(Debug)]
    struct DummyGraph;
    impl PartitionableGraph for DummyGraph {
        type VertexId = usize;
        type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
        type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
        fn vertices(&self) -> Self::VertexParIter<'_> {
            vec![0, 1, 2, 3].into_par_iter()
        }
        fn neighbors(&self, _v: Self::VertexId) -> Self::NeighParIter<'_> {
            Vec::new().into_par_iter()
        }
        fn degree(&self, _v: Self::VertexId) -> usize {
            0
        }
    }
    #[test]
    fn trivial_partition_compiles() {
        let g = DummyGraph;
        let cfg = PartitionerConfig {
            n_parts: 2,
            ..Default::default()
        };
        let pm = partition(&g, &cfg).expect("partition should succeed");
        assert_eq!(pm.len(), 4);
    }
}

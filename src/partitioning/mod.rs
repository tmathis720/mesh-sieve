//! Entry-point for native graph partitioning.
#![cfg_attr(not(feature = "partitioning"), allow(dead_code, unused_imports))]

pub mod graph_traits;
pub mod metrics;
pub mod seed_select;
pub mod louvain;
pub mod binpack;
pub mod vertex_cut;
pub mod parallel;

pub use self::metrics::*;

#[cfg(feature = "partitioning")]
use hashbrown::HashMap;
use std::hash::Hash;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

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
pub fn partition<G>(graph: &G, cfg: &PartitionerConfig) -> Result<PartitionMap<G::VertexId>, PartitionerError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    use crate::partitioning::{louvain::louvain_cluster, binpack::partition_clusters, binpack::Item, vertex_cut::build_vertex_cuts, metrics::{edge_cut, replication_factor}};
    use std::collections::HashMap;

    let verts: Vec<_> = graph.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return Ok(PartitionMap::with_capacity(0));
    }

    // 1. Louvain clustering
    let clusters = louvain_cluster(graph, cfg);
    let n_clusters = clusters.iter().copied().max().unwrap_or(0) as usize + 1;

    // 2. Build cluster -> vertices map and cluster loads
    let mut cluster_to_verts: HashMap<u32, Vec<usize>> = HashMap::with_capacity(n_clusters);
    let mut cluster_loads: HashMap<u32, u64> = HashMap::with_capacity(n_clusters);
    for (v, &cid) in verts.iter().zip(clusters.iter()) {
        cluster_to_verts.entry(cid).or_default().push(*v);
        let deg = graph.degree(*v) as u64;
        *cluster_loads.entry(cid).or_insert(0) += deg.max(1); // avoid zero-load clusters
    }

    // 3. Binpack clusters into parts
    let mut items: Vec<Item> = cluster_to_verts.iter().map(|(&cid, verts)| {
        let load = *cluster_loads.get(&cid).unwrap_or(&1);
        Item { cid: cid as usize, load, adj: vec![] }
    }).collect();
    let cluster_part = partition_clusters(&items, cfg.n_parts, 0.05);

    // 4. Assign each vertex to its cluster's part
    let mut pm = PartitionMap::with_capacity(n);
    for (i, (&cid, verts)) in cluster_to_verts.iter().enumerate() {
        let part = cluster_part[i];
        for &v in verts {
            pm.insert(v, part);
        }
    }

    // 5. Vertex cut construction (primary owners, replicas)
    let _vertex_cut = build_vertex_cuts(graph, &pm, cfg.rng_seed);

    // 6. Metrics (for debug/logging)
    let _cut = edge_cut(graph, &pm);
    let _rep = replication_factor(graph, &pm);
    // Optionally: log or print metrics here

    Ok(pm)
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let cfg = PartitionerConfig { n_parts: 2, ..Default::default() };
        let pm = partition(&g, &cfg).expect("partition should succeed");
        assert_eq!(pm.len(), 4);
    }
}

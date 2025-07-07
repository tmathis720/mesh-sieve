//! # Native Graph Partitioning (Onizuka et al. 2017)
//!
//! This module provides a three-phase, native Rust implementation of the
//! balanced graph‐partitioning strategy described in Onizuka _et al._ [2017].
//!
//! ## Phase 1: Balanced Modularity Clustering
//!
//! We first run a **Louvain-style** clustering with a Wakita–Tsurumi adjustment
//! to promote similarly sized clusters.  This is implemented in:
//! - [`louvain::louvain_cluster`]  
//!
//! ## Phase 2: Adjacency-Aware Cluster Merge
//!
//! Next, we treat each cluster as an “item” with both load and inter-cluster
//! edge weights, and perform an adjacency‐guided merge into _k_ parts via:
//! - [`binpack::merge_clusters_into_parts`]  
//!
//! ## Phase 3: Load-Aware Graph Conversion
//!
//! Finally, we convert the edge-cut partition into a vertex-cut assignment,
//! choosing owners to minimize per-part replica load.  See:
//! - [`vertex_cut::build_vertex_cuts`]
//!
//! [2017]: https://doi.org/10.1145/3126908.3126929
#![cfg_attr(not(feature = "mpi-support"), allow(dead_code, unused_imports))]

#[cfg(feature = "mpi-support")]
pub mod binpack;
#[cfg(feature = "mpi-support")]
pub mod error;
#[cfg(feature = "mpi-support")]
pub mod graph_traits;
#[cfg(feature = "mpi-support")]
pub mod louvain;
#[cfg(feature = "mpi-support")]
pub mod metrics;
#[cfg(feature = "mpi-support")]
pub mod parallel;
#[cfg(feature = "mpi-support")]
pub mod seed_select;
#[cfg(feature = "mpi-support")]
pub mod vertex_cut;

#[cfg(feature = "mpi-support")]
pub use self::metrics::*;

#[cfg(feature = "mpi-support")]
use hashbrown::HashMap;
use log::debug;
use std::hash::Hash;
use rayon::prelude::*;

#[cfg(feature = "mpi-support")]
pub type PartitionId = usize;

#[cfg(feature = "mpi-support")]
#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    pub n_parts: usize,
    pub alpha: f64,
    pub seed_factor: f64,
    pub rng_seed: u64,
    pub max_iters: usize,
    /// allowed imbalance: max_load/min_load ≤ 1 + epsilon
    pub epsilon: f64,
    /// Enable/disable each phase
    pub enable_phase1: bool,
    pub enable_phase2: bool,
    pub enable_phase3: bool,
}

impl Default for PartitionerConfig {
    fn default() -> Self {
        Self {
            n_parts: 2,
            alpha: 0.75,
            seed_factor: 4.0,
            rng_seed: 42,
            max_iters: 20,
            epsilon: 0.05,
            enable_phase1: true,
            enable_phase2: true,
            enable_phase3: true,
        }
    }
}

#[cfg(feature = "mpi-support")]
#[derive(Debug, Clone)]
pub struct PartitionMap<V: Eq + Hash + Copy>(HashMap<V, PartitionId>);

#[cfg(feature = "mpi-support")]
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

#[cfg(feature = "mpi-support")]
use crate::partitioning::graph_traits::PartitionableGraph;

#[cfg(feature = "mpi-support")]
#[derive(Debug)]
pub enum PartitionerError {
    /// Louvain hit max iterations without converging
    MaxIter,
    /// The cluster‐merge phase never found a positive adjacency merge
    NoPositiveMerge,
    /// Final part loads are unbalanced: max/min = ratio > tolerance
    Unbalanced {
        max_load: u64,
        min_load: u64,
        ratio:    f64,
        tolerance: f64,
    },
    /// Error during vertex cut construction
    VertexCut(crate::partitioning::error::PartitionError),
}

impl From<crate::partitioning::error::PartitionError> for PartitionerError {
    fn from(e: crate::partitioning::error::PartitionError) -> Self {
        PartitionerError::VertexCut(e)
    }
}

#[cfg(feature = "mpi-support")]
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
    let clusters = if cfg.enable_phase1 {
        louvain_cluster(graph, cfg)
    } else {
        graph.vertices().map(|u| u as u32).collect()
    };
    let n_clusters = clusters.iter().copied().max().unwrap_or(0) as usize + 1;
    // Phase 1 metrics
    let mut pm1 = PartitionMap::with_capacity(n);
    for (v, &cid) in verts.iter().zip(clusters.iter()) {
        pm1.insert(*v, cid as usize);
    }
    let cut1 = edge_cut(graph, &pm1);
    let rep1 = replication_factor(graph, &pm1);
    debug!(
        "Phase 1 (Louvain): clusters={}  edge_cut={}  replication_factor={:.3}",
        n_clusters, cut1, rep1
    );

    // Build a mapping from vertex ID to index in verts
    let vert_idx: HashMap<usize, usize> = verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();

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
        let cu = clusters[*vert_idx.get(&u).expect("vertex not found in vert_idx")] ;
        let cv = clusters[*vert_idx.get(&v).expect("vertex not found in vert_idx")] ;
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
    let cluster_part = if cfg.enable_phase2 {
        merge_clusters_into_parts(&items, cfg.n_parts, cfg.epsilon)?
    } else {
        items.iter().map(|it| it.cid % cfg.n_parts).collect()
    };
    // 5. Assign each vertex to its cluster's part
    let mut pm = PartitionMap::with_capacity(n);
    for (i, (_cid, verts)) in cluster_to_verts.iter().enumerate() {
        let part = cluster_part[i];
        for &v in verts {
            pm.insert(v, part);
        }
    }
    let cut2 = edge_cut(graph, &pm);
    let rep2 = replication_factor(graph, &pm);
    debug!(
        "Phase 2 (Merge): parts={}  edge_cut={}  replication_factor={:.3}",
        cfg.n_parts, cut2, rep2
    );
    // 6. Vertex cut construction (primary owners, replicas)
    let (primary, replicas) = if cfg.enable_phase3 {
        build_vertex_cuts(graph, &pm, cfg.rng_seed)?
    } else {
        Ok::<_, PartitionerError>((pm.iter().map(|(_v,&p)| p).collect(), vec![Vec::new(); n]))?
    };
    let total_replicas: usize = replicas.iter().map(|r| r.len()).sum();
    debug!(
        "Phase 3 (VertexCut): primary_count={}  total_replicas={}",
        primary.len(),
        total_replicas
    );
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
        let cfg = PartitionerConfig {
            n_parts: 2,
            ..Default::default()
        };
        match partition(&g, &cfg) {
            Ok(pm) => assert_eq!(pm.len(), 4),
            Err(PartitionerError::NoPositiveMerge) => {}, // Acceptable for edgeless graph
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}

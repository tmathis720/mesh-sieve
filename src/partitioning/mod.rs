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
use rayon::prelude::*;
use std::hash::Hash;

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
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
        ratio: f64,
        tolerance: f64,
    },
    /// Error during vertex cut construction
    VertexCut(crate::partitioning::error::PartitionError),
    /// The `degrees` slice had the wrong length.
    DegreeLengthMismatch { expected: usize, got: usize },
    /// Other errors
    Other(String),
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
        binpack::merge_clusters_into_parts,
        binpack::Item,
        louvain::louvain_cluster,
        metrics::{edge_cut, replication_factor},
        vertex_cut::build_vertex_cuts,
    };

    // ------------- Phase 0: vertices & trivial case -------------
    let verts: Vec<_> = graph.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return Ok(PartitionMap::with_capacity(0));
    }

    // Map vertex ID -> dense index [0..n)
    // (Assume edges() only yields vertices present in vertices(); we debug_assert below.)
    let vert_idx: HashMap<usize, usize> = verts
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    // ------------- Phase 1: Louvain (fixed) -------------
    let clusters: Vec<u32> = if cfg.enable_phase1 {
        louvain_cluster(graph, cfg)
    } else {
        // Dense, contiguous cluster IDs when Phase 1 disabled
        verts
            .iter()
            .enumerate()
            .map(|(i, _)| (i as u32 % cfg.n_parts as u32))
            .collect()
    };

    // Defensive checks
    if clusters.len() != n {
        return Err(PartitionerError::Other(
            "louvain_cluster returned wrong length".into(),
        ));
    }

    // Dense #clusters (assumes louvain remapped to 0..C-1)
    let n_clusters: usize = clusters.iter().copied().max().unwrap_or(0) as usize + 1;

    // Phase 1 metrics (same as before)
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

    // ------------- Phase 2 prework: loads & inter-cluster adjacency -------------

    // A) per-cluster load: sum of degrees (≥1) — simple, sequential is fine (O(V))
    //    If you prefer parallel, switch to Vec<AtomicU64> and par_iter over verts.
    let mut cluster_loads: Vec<u64> = vec![0; n_clusters];
    for &v in &verts {
        // Safe: degree() is read-only
        let deg = graph.degree(v) as u64;
        let vi = *vert_idx
            .get(&v)
            .expect("vertex from vertices() not in index map");
        let cid = clusters[vi] as usize;
        // Avoid zero-load clusters (as you did)
        cluster_loads[cid] += deg.max(1);
    }

    // B) cluster-to-vertex lists: O(V) and small memory; sequential is robust.
    let mut cluster_to_verts: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for &v in &verts {
        let vi = vert_idx[&v];
        let cid = clusters[vi] as usize;
        cluster_to_verts[cid].push(v);
    }

    // C) Inter-cluster adjacency via parallel fold/reduce over edges()
    //    No all_edges allocation; memory proportional to #inter-cluster pairs.
    //
    //    Key: (min(cu, cv), max(cu, cv)) → edge count
    //
    //    We use `hashbrown::HashMap` if available (imported at top under cfg),
    //    which is faster and merges cheaply; std::collections::HashMap also works.
    let cluster_adj: HashMap<(u32, u32), u64> = graph
        .edges()
        .fold(
            || HashMap::<(u32, u32), u64>::new(),
            |mut local, (u, v)| {
                // Convert u,v to cluster IDs
                let iu = match vert_idx.get(&u) {
                    Some(&i) => i,
                    None => {
                        debug_assert!(false, "edges() yielded u not in vertices()");
                        return local;
                    }
                };
                let iv = match vert_idx.get(&v) {
                    Some(&i) => i,
                    None => {
                        debug_assert!(false, "edges() yielded v not in vertices()");
                        return local;
                    }
                };

                let cu = clusters[iu];
                let cv = clusters[iv];
                if cu != cv {
                    let key = if cu < cv { (cu, cv) } else { (cv, cu) };
                    *local.entry(key).or_insert(0) += 1;
                }
                local
            },
        )
        .reduce(
            || HashMap::<(u32, u32), u64>::new(),
            |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            },
        );

    // D) Build symmetric adjacency lists per cluster (deterministic order)
    let mut adj_lists: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n_clusters];
    for (&(a, b), &w) in cluster_adj.iter() {
        let ai = a as usize;
        let bi = b as usize;
        adj_lists[ai].push((bi, w));
        adj_lists[bi].push((ai, w));
    }
    for lst in &mut adj_lists {
        lst.sort_unstable_by_key(|&(cid, _)| cid); // determinism
    }

    // E) Build Items in dense cid order (cid == index). Zero-adj clusters get empty vec.
    let items: Vec<Item> = (0..n_clusters)
        .map(|cid| Item {
            cid,
            load: cluster_loads[cid],
            adj: adj_lists[cid].clone(),
        })
        .collect();

    // ------------- Phase 2: merge clusters into parts -------------
    let cluster_part = if cfg.enable_phase2 {
        merge_clusters_into_parts(&items, cfg.n_parts, cfg.epsilon)?
    } else {
        // Deterministic fallback
        (0..n_clusters).map(|cid| cid % cfg.n_parts).collect()
    };

    // Assign each vertex to its cluster's part
    let mut pm = PartitionMap::with_capacity(n);
    for cid in 0..n_clusters {
        let part = cluster_part[cid];
        for &v in &cluster_to_verts[cid] {
            pm.insert(v, part);
        }
    }

    // Phase 2 metrics
    let cut2 = edge_cut(graph, &pm);
    let rep2 = replication_factor(graph, &pm);
    debug!(
        "Phase 2 (Merge): parts={}  edge_cut={}  replication_factor={:.3}",
        cfg.n_parts, cut2, rep2
    );

    // ------------- Phase 3: vertex-cut -------------
    let (_primary, replicas) = if cfg.enable_phase3 {
        build_vertex_cuts(graph, &pm, cfg.rng_seed)?
    } else {
        // keep structure but do nothing
        (vec![0; n], vec![Vec::new(); n])
    };
    let total_replicas: usize = replicas.iter().map(|r| r.len()).sum();
    debug!(
        "Phase 3 (VertexCut): primary_count={}  total_replicas={}",
        n, total_replicas
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
        type NeighIter<'a> = std::vec::IntoIter<usize>;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            vec![0, 1, 2, 3].into_par_iter()
        }
        fn neighbors(&self, _v: Self::VertexId) -> Self::NeighParIter<'_> {
            Vec::new().into_par_iter()
        }
        fn neighbors_seq(&self, _v: Self::VertexId) -> Self::NeighIter<'_> {
            Vec::new().into_iter()
        }
        fn degree(&self, _v: Self::VertexId) -> usize {
            0
        }
        fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
            Vec::new().into_par_iter()
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
            Err(PartitionerError::NoPositiveMerge) => {} // Acceptable for edgeless graph
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}

//! # Native Graph Partitioning (Onizuka et al. 2017)
//!
//! This module provides a three-phase, native Rust implementation of the
//! balanced graph‐partitioning strategy described in Onizuka _et al._ [2017].
//!
//! We avoid iterative, color-based many-to-many collectives; communication is
//! batched per coarse phase. Each phase exposes a single boundary where an
//! all-to-all (or sparse all-to-all) could be inserted in a distributed
//! setting.
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
//! ## Future Work: Distributed Execution
//!
//! On coarse levels, the active rank set should shrink so each rank maintains
//! `O(n/p)` vertices and `O(m/p)` edges. After Phase 1, exchange cluster IDs for
//! boundary vertices once, then continue locally. After Phase 2, exchange only
//! cluster→part assignments. Phase 3 requires exchanging owner decisions for cut
//! edges. These boundaries act as fold points for future graph redistribution.
//!
//! [2017]: https://doi.org/10.1145/3126908.3126929
#![cfg_attr(not(feature = "mpi-support"), allow(dead_code, unused_imports))]

#[cfg(feature = "mpi-support")]
pub mod binpack;
#[cfg(feature = "mpi-support")]
pub mod error;
#[cfg(feature = "mpi-support")]
pub mod graph_traits;
pub mod louvain;
pub mod metrics;
#[cfg(feature = "mpi-support")]
pub mod parallel;
#[cfg(feature = "mpi-support")]
pub mod seed_select;
#[cfg(feature = "mpi-support")]
pub mod vertex_cut;

pub use self::metrics::*;

#[cfg(feature = "mpi-support")]
use hashbrown::HashMap;
use log::debug;
use rayon::prelude::*;
#[cfg(feature = "mpi-support")]
use std::collections::{BTreeMap, BTreeSet};
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
use crate::algs::communicator::{Communicator, Wait};
#[cfg(feature = "mpi-support")]
use crate::algs::point_sf::PointSF;
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

/// Wire format for sparse adjacency summaries (future distributed use).
#[cfg(feature = "mpi-support")]
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ClusterSummary {
    pub cid: u32,
    pub load: u64,
    /// neighbors stored as parallel arrays
    pub nbr_cids: Vec<u32>,
    pub nbr_wts: Vec<u64>,
}

#[cfg(feature = "mpi-support")]
const PARTITION_EXCHANGE_BOUNDARY_CLUSTERS: u16 = 10_001;
#[cfg(feature = "mpi-support")]
const PARTITION_EXCHANGE_CLUSTER_PARTS: u16 = 10_011;
#[cfg(feature = "mpi-support")]
const PARTITION_EXCHANGE_CUT_OWNERS: u16 = 10_021;

#[cfg(feature = "mpi-support")]
fn partition_comm_error(e: crate::mesh_error::MeshSieveError) -> PartitionerError {
    PartitionerError::Other(format!("distributed partition exchange failed: {e}"))
}

#[cfg(feature = "mpi-support")]
fn sorted_sf_neighbors<C>(sf: &PointSF<'_, C>) -> Vec<usize>
where
    C: Communicator + Sync,
{
    let mut neighbors: BTreeSet<usize> = sf.leaves().map(|leaf| leaf.remote.rank).collect();
    neighbors.remove(&sf.rank());
    neighbors.into_iter().collect()
}

#[cfg(feature = "mpi-support")]
fn exchange_variable_bytes<C>(
    comm: &C,
    neighbors: &[usize],
    payloads: &BTreeMap<usize, Vec<u8>>,
    base_tag: u16,
) -> Result<BTreeMap<usize, Vec<u8>>, PartitionerError>
where
    C: Communicator + Sync,
{
    if comm.size() <= 1 || neighbors.is_empty() {
        return Ok(BTreeMap::new());
    }

    let size_tag = base_tag;
    let data_tag = base_tag.wrapping_add(1);

    let mut recv_sizes = BTreeMap::new();
    for &nbr in neighbors {
        let mut buf = [0u8; 8];
        let handle = comm
            .irecv_result(nbr, size_tag, &mut buf)
            .map_err(partition_comm_error)?;
        recv_sizes.insert(nbr, handle);
    }

    let mut size_send_handles = Vec::with_capacity(neighbors.len());
    let mut size_send_buffers = Vec::with_capacity(neighbors.len());
    for &nbr in neighbors {
        let len = payloads.get(&nbr).map_or(0u64, |bytes| bytes.len() as u64);
        let buf = len.to_le_bytes();
        size_send_handles.push(
            comm.isend_result(nbr, size_tag, &buf)
                .map_err(partition_comm_error)?,
        );
        size_send_buffers.push(buf);
    }

    let mut lengths = BTreeMap::new();
    let mut first_err = None;
    for (nbr, handle) in recv_sizes {
        match handle.wait() {
            Some(bytes) if bytes.len() == 8 => {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(&bytes);
                lengths.insert(nbr, u64::from_le_bytes(raw) as usize);
            }
            Some(bytes) if first_err.is_none() => {
                first_err = Some(PartitionerError::Other(format!(
                    "rank {} expected 8 size bytes from rank {nbr}, got {}",
                    comm.rank(),
                    bytes.len()
                )));
            }
            None if first_err.is_none() => {
                first_err = Some(PartitionerError::Other(format!(
                    "rank {} failed receiving exchange size from rank {nbr}",
                    comm.rank()
                )));
            }
            _ => {}
        }
    }
    for handle in size_send_handles {
        let _ = handle.wait();
    }
    drop(size_send_buffers);
    if let Some(err) = first_err {
        return Err(err);
    }

    let mut recv_data = BTreeMap::new();
    for (&nbr, &len) in &lengths {
        let mut buf = vec![0u8; len];
        let handle = comm
            .irecv_result(nbr, data_tag, &mut buf)
            .map_err(partition_comm_error)?;
        recv_data.insert(nbr, (handle, len));
    }

    let mut data_send_handles = Vec::with_capacity(neighbors.len());
    for &nbr in neighbors {
        let bytes = payloads.get(&nbr).map_or(&[][..], |v| &v[..]);
        data_send_handles.push(
            comm.isend_result(nbr, data_tag, bytes)
                .map_err(partition_comm_error)?,
        );
    }

    let mut incoming = BTreeMap::new();
    let mut first_err = None;
    for (nbr, (handle, expected_len)) in recv_data {
        match handle.wait() {
            Some(bytes) if bytes.len() == expected_len => {
                incoming.insert(nbr, bytes);
            }
            Some(bytes) if first_err.is_none() => {
                first_err = Some(PartitionerError::Other(format!(
                    "rank {} expected {expected_len} data bytes from rank {nbr}, got {}",
                    comm.rank(),
                    bytes.len()
                )));
            }
            None if first_err.is_none() => {
                first_err = Some(PartitionerError::Other(format!(
                    "rank {} failed receiving exchange payload from rank {nbr}",
                    comm.rank()
                )));
            }
            _ => {}
        }
    }
    for handle in data_send_handles {
        let _ = handle.wait();
    }
    if let Some(err) = first_err {
        Err(err)
    } else {
        Ok(incoming)
    }
}

#[cfg(feature = "mpi-support")]
fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "mpi-support")]
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, PartitionerError> {
    if bytes.len().saturating_sub(*cursor) < 8 {
        return Err(PartitionerError::Other(
            "truncated partition exchange payload".into(),
        ));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(raw))
}

#[cfg(feature = "mpi-support")]
fn encode_pairs_u64(entries: &[(u64, u64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + entries.len() * 16);
    push_u64(&mut out, entries.len() as u64);
    for &(a, b) in entries {
        push_u64(&mut out, a);
        push_u64(&mut out, b);
    }
    out
}

#[cfg(feature = "mpi-support")]
fn decode_pairs_u64(bytes: &[u8]) -> Result<Vec<(u64, u64)>, PartitionerError> {
    let mut cursor = 0usize;
    let n = read_u64(bytes, &mut cursor)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push((read_u64(bytes, &mut cursor)?, read_u64(bytes, &mut cursor)?));
    }
    if cursor != bytes.len() {
        return Err(PartitionerError::Other(
            "partition exchange payload has trailing bytes".into(),
        ));
    }
    Ok(out)
}

#[cfg(feature = "mpi-support")]
fn encode_triples_u64(entries: &[(u64, u64, u64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + entries.len() * 24);
    push_u64(&mut out, entries.len() as u64);
    for &(a, b, c) in entries {
        push_u64(&mut out, a);
        push_u64(&mut out, b);
        push_u64(&mut out, c);
    }
    out
}

#[cfg(feature = "mpi-support")]
fn decode_triples_u64(bytes: &[u8]) -> Result<Vec<(u64, u64, u64)>, PartitionerError> {
    let mut cursor = 0usize;
    let n = read_u64(bytes, &mut cursor)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push((
            read_u64(bytes, &mut cursor)?,
            read_u64(bytes, &mut cursor)?,
            read_u64(bytes, &mut cursor)?,
        ));
    }
    if cursor != bytes.len() {
        return Err(PartitionerError::Other(
            "partition exchange payload has trailing bytes".into(),
        ));
    }
    Ok(out)
}

#[cfg(feature = "mpi-support")]
pub(crate) fn exchange_boundary_cluster_ids(local: &HashMap<usize, u32>) -> HashMap<usize, u32> {
    // Single-rank compatibility wrapper for callers that have no PointSF.
    local.clone()
}

/// Exchange cluster IDs for shared boundary vertices using the supplied PointSF.
///
/// Each rank sends `(remote_vertex, local_cluster)` along SF leaves and receives
/// entries keyed by its local vertex IDs.  Reconciliation is deterministic: the
/// lowest cluster ID wins when more than one neighbor reports the same boundary
/// vertex.
#[cfg(feature = "mpi-support")]
pub(crate) fn exchange_boundary_cluster_ids_with_sf<C>(
    local: &HashMap<usize, u32>,
    sf: &PointSF<'_, C>,
) -> Result<HashMap<usize, u32>, PartitionerError>
where
    C: Communicator + Sync,
{
    let Some(comm) = sf.comm() else {
        return Ok(exchange_boundary_cluster_ids(local));
    };
    let neighbors = sorted_sf_neighbors(sf);
    if comm.size() <= 1 || neighbors.is_empty() {
        return Ok(exchange_boundary_cluster_ids(local));
    }

    let mut per_neighbor: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
    for leaf in sf.leaves() {
        let nbr = leaf.remote.rank;
        if nbr == sf.rank() {
            continue;
        }
        let local_vertex = leaf.local.get() as usize;
        if let Some(&cid) = local.get(&local_vertex) {
            per_neighbor
                .entry(nbr)
                .or_default()
                .push((leaf.remote.point.get(), cid as u64));
        }
    }
    for entries in per_neighbor.values_mut() {
        entries.sort_unstable();
        entries.dedup();
    }
    let payloads = neighbors
        .iter()
        .copied()
        .map(|nbr| {
            let bytes = encode_pairs_u64(per_neighbor.get(&nbr).map_or(&[][..], |v| &v[..]));
            (nbr, bytes)
        })
        .collect();

    let incoming = exchange_variable_bytes(
        comm,
        &neighbors,
        &payloads,
        PARTITION_EXCHANGE_BOUNDARY_CLUSTERS,
    )?;
    let mut out = local.clone();
    for (_nbr, bytes) in incoming {
        for (vertex, cid) in decode_pairs_u64(&bytes)? {
            out.entry(vertex as usize)
                .and_modify(|cur| *cur = (*cur).min(cid as u32))
                .or_insert(cid as u32);
        }
    }
    Ok(out)
}

#[cfg(feature = "mpi-support")]
pub(crate) fn exchange_cluster_part_assignments(local: &[usize]) -> Vec<usize> {
    // Single-rank compatibility wrapper for callers that have no PointSF.
    local.to_vec()
}

/// Exchange cluster-to-part assignments visible in the overlap neighborhood.
///
/// The dense result is extended as needed for remote cluster IDs.  Conflicting
/// assignments are reconciled by choosing the lowest part ID so repeated runs
/// and different receive orders produce identical maps.
#[cfg(feature = "mpi-support")]
pub(crate) fn exchange_cluster_part_assignments_with_sf<C>(
    local: &[usize],
    sf: &PointSF<'_, C>,
) -> Result<Vec<usize>, PartitionerError>
where
    C: Communicator + Sync,
{
    let Some(comm) = sf.comm() else {
        return Ok(exchange_cluster_part_assignments(local));
    };
    let neighbors = sorted_sf_neighbors(sf);
    if comm.size() <= 1 || neighbors.is_empty() {
        return Ok(exchange_cluster_part_assignments(local));
    }

    let mut entries: Vec<_> = local
        .iter()
        .copied()
        .enumerate()
        .map(|(cid, part)| (cid as u64, part as u64))
        .collect();
    entries.sort_unstable();
    let bytes = encode_pairs_u64(&entries);
    let payloads = neighbors
        .iter()
        .copied()
        .map(|nbr| (nbr, bytes.clone()))
        .collect();
    let incoming = exchange_variable_bytes(
        comm,
        &neighbors,
        &payloads,
        PARTITION_EXCHANGE_CLUSTER_PARTS,
    )?;

    let mut out = local.to_vec();
    for (_nbr, bytes) in incoming {
        for (cid, part) in decode_pairs_u64(&bytes)? {
            let cid = cid as usize;
            let part = part as usize;
            if cid >= out.len() {
                out.resize(cid + 1, part);
            }
            out[cid] = out[cid].min(part);
        }
    }
    Ok(out)
}

#[cfg(feature = "mpi-support")]
pub(crate) fn exchange_cut_edge_owner_decisions(
    local: &HashMap<(usize, usize), usize>,
) -> HashMap<(usize, usize), usize> {
    // Single-rank compatibility wrapper.  The distributed variant below applies
    // the same deterministic minimum-owner reduction to incoming candidates.
    let mut grouped: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (&edge, &owner) in local {
        let edge = canonical_edge(edge);
        grouped
            .entry(edge)
            .and_modify(|cur| *cur = (*cur).min(owner))
            .or_insert(owner);
    }
    grouped.into_iter().collect()
}

#[cfg(feature = "mpi-support")]
fn canonical_edge(edge: (usize, usize)) -> (usize, usize) {
    if edge.0 <= edge.1 {
        edge
    } else {
        (edge.1, edge.0)
    }
}

/// Exchange cut-edge owner candidates with overlapped ranks and reduce them to
/// deterministic winners.  The owner for each undirected edge is the lowest
/// proposed owner rank across all received and local candidates.
#[cfg(feature = "mpi-support")]
pub(crate) fn exchange_cut_edge_owner_decisions_with_sf<C>(
    local: &HashMap<(usize, usize), usize>,
    sf: &PointSF<'_, C>,
) -> Result<HashMap<(usize, usize), usize>, PartitionerError>
where
    C: Communicator + Sync,
{
    let Some(comm) = sf.comm() else {
        return Ok(exchange_cut_edge_owner_decisions(local));
    };
    let neighbors = sorted_sf_neighbors(sf);
    if comm.size() <= 1 || neighbors.is_empty() {
        return Ok(exchange_cut_edge_owner_decisions(local));
    }

    let mut entries: Vec<_> = local
        .iter()
        .map(|(&(u, v), &owner)| {
            let (a, b) = canonical_edge((u, v));
            (a as u64, b as u64, owner as u64)
        })
        .collect();
    entries.sort_unstable();
    entries.dedup();
    let bytes = encode_triples_u64(&entries);
    let payloads = neighbors
        .iter()
        .copied()
        .map(|nbr| (nbr, bytes.clone()))
        .collect();
    let incoming =
        exchange_variable_bytes(comm, &neighbors, &payloads, PARTITION_EXCHANGE_CUT_OWNERS)?;

    let mut out = exchange_cut_edge_owner_decisions(local);
    for (_nbr, bytes) in incoming {
        for (u, v, owner) in decode_triples_u64(&bytes)? {
            let edge = canonical_edge((u as usize, v as usize));
            let owner = owner as usize;
            out.entry(edge)
                .and_modify(|cur| *cur = (*cur).min(owner))
                .or_insert(owner);
        }
    }
    Ok(out)
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
            .map(|(i, _)| i as u32 % cfg.n_parts as u32)
            .collect()
    };
    let _boundary_clusters = exchange_boundary_cluster_ids(&HashMap::new());

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
    let cluster_part = exchange_cluster_part_assignments(&cluster_part);

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
    let _cut_edge_owners = exchange_cut_edge_owner_decisions(&HashMap::new());
    let total_replicas: usize = replicas.iter().map(|r| r.len()).sum();
    debug!(
        "Phase 3 (VertexCut): primary_count={}  total_replicas={}",
        n, total_replicas
    );

    Ok(pm)
}

#[cfg(test)]
mod tests;

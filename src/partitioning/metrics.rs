//! ## Replication Factor (RF) and Load Balance
//!
//! - RF: average number of parts each vertex appears in. If `P(v)` is the set of parts
//!   containing vertex `v` (primary or replica), then
//!
//!       RF = (1 / |V|) * ∑_{v∈V} |P(v)|
//!
//!   In unit tests we compute RF exactly from `(primary_owner, replicas)` returned by
//!   `vertex_cut::build_vertex_cuts`. The debug/CI `RF(g, pm)` helper approximates RF
//!   by marking neighbor‐induced presence (adequate for sanity checks).
//!
//! - Load balance factor (LB): if `L_p` is the load of part `p` (default: sum of degrees
//!   of vertices owned by `p`), then
//!
//!       LB = max_p L_p / (min_p L_p + ε)
//!
//!   We enforce `LB ≤ 1 + epsilon` in bin-packing; see `PartitionerConfig::epsilon`.
//!
//! This module provides functions for evaluating the quality of graph
//! partitionings, including edge cut and replication factor. These are
//! intended for debugging, testing and CI validation of partitioning algorithms.
//!
//! With the `mem-snapshot` feature enabled, [`memory_snapshot_bytes`] returns the
//! current resident set size (RSS) for detecting memory regressions in benches.

use super::{PartitionMap, PartitionableGraph};
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Returns the part ID for a given vertex.
///
/// # Panics
/// Panics if the vertex is not present in the partition map.
impl<V: Eq + Hash + Copy> PartitionMap<V> {
    pub fn part_of(&self, v: V) -> usize {
        *self.get(&v).expect("vertex not found in PartitionMap")
    }
}

// -----------------------------------------------------------------------------
// Utility: count number of parts in a partition map

#[inline]
fn num_parts_in_pm<V: Eq + Hash + Copy>(pm: &PartitionMap<V>) -> Option<usize> {
    let mut maxp: Option<usize> = None;
    for (_, &p) in pm.iter() {
        maxp = Some(maxp.map_or(p, |m| m.max(p)));
    }
    maxp.map(|m| m + 1)
}

const WORD_BITS: usize = usize::BITS as usize;

/// Lock-free dynamic bitset used for replication-factor computation.
#[derive(Debug)]
struct AtomicBitset {
    words: Box<[AtomicUsize]>,
}

impl AtomicBitset {
    #[inline]
    fn new(num_parts: usize) -> Self {
        let n_words = (num_parts + WORD_BITS - 1) / WORD_BITS;
        let mut v = Vec::with_capacity(n_words);
        for _ in 0..n_words {
            v.push(AtomicUsize::new(0));
        }
        Self {
            words: v.into_boxed_slice(),
        }
    }

    #[inline]
    fn or_bit(&self, part: usize) {
        let wi = part / WORD_BITS;
        let bi = part % WORD_BITS;
        let mask = 1usize << bi;
        // Relaxed is sufficient: reads occur only after parallel joins.
        self.words[wi].fetch_or(mask, Ordering::Relaxed);
    }

    #[inline]
    fn popcount(&self) -> usize {
        // Safe to read with Relaxed after all writers join.
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }
}

// -----------------------------------------------------------------------------
// Metrics

/// Counts undirected cross-part edges via [`PartitionableGraph::edges`].
///
/// Each undirected edge is considered once.  Complexity O(E).
pub fn edge_cut<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> usize
where
    G: PartitionableGraph,
    G::VertexId: PartialOrd + Eq + Hash + Copy + Sync + 'static,
{
    g.edges()
        .map(|(u, v)| (pm.part_of(u) != pm.part_of(v)) as usize)
        .sum()
}

/// Average number of parts a vertex appears in (its own part plus parts of
/// neighbouring vertices).
///
/// Lock-free implementation using atomic bitsets.  Complexity
/// O(E + V * W) where `W = ceil(P / WORD_BITS)` and `P` is the number of parts.
pub fn replication_factor<G>(g: &G, pm: &PartitionMap<G::VertexId>) -> f64
where
    G: PartitionableGraph,
    G::VertexId: Eq + Hash + Copy + Sync + 'static,
{
    use rayon::prelude::*;

    // 1) Collect vertices to index them.
    let verts: Vec<G::VertexId> = g.vertices().collect();
    let n = verts.len();
    if n == 0 {
        return 0.0;
    }

    // 2) Map vertex -> dense index.
    let idx_map: HashMap<G::VertexId, usize> = verts
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    // 3) Number of parts from partition map.
    let num_parts = match num_parts_in_pm(pm) {
        Some(np) if np > 0 => np,
        _ => return 0.0,
    };

    // 4) Per-vertex atomic bitset.
    let masks: Vec<AtomicBitset> = (0..n).map(|_| AtomicBitset::new(num_parts)).collect();

    // 5) Edge sweep: for each edge (u,v), mark parts of u in v and vice versa.
    g.edges().for_each(|(u, v)| {
        let pu = pm.part_of(u);
        let pv = pm.part_of(v);
        let ui = *idx_map.get(&u).expect("vertex not found");
        let vi = *idx_map.get(&v).expect("vertex not found");
        masks[vi].or_bit(pu);
        masks[ui].or_bit(pv);
    });

    // 6) Ensure every vertex counts its own part (isolated vertices).
    verts.par_iter().for_each(|&u| {
        let ui = idx_map[&u];
        let p = pm.part_of(u);
        masks[ui].or_bit(p);
    });

    // 7) Popcount and average.
    let total_owned: usize = masks.par_iter().map(|m| m.popcount()).sum();
    total_owned as f64 / n as f64
}

/// Compute the exact replication factor from explicit primary owners and replica lists.
#[allow(dead_code)]
pub fn rf_exact<V>(primary: &[usize], replicas: &[Vec<(V, usize)>]) -> f64 {
    use std::collections::HashSet;
    let n = primary.len();
    assert_eq!(n, replicas.len(), "length mismatch");
    if n == 0 {
        return 0.0;
    }
    let mut total = 0usize;
    for (p, reps) in primary.iter().zip(replicas.iter()) {
        let mut parts: HashSet<usize> = HashSet::new();
        parts.insert(*p);
        for &(_, part) in reps {
            parts.insert(part);
        }
        total += parts.len();
    }
    total as f64 / n as f64
}

/// Compute min/max part load (sum of vertex degrees) and their ratio.
#[allow(dead_code)]
pub fn load_balance_parts<G>(pm: &PartitionMap<G::VertexId>, g: &G) -> (u64, u64, f64)
where
    G: PartitionableGraph,
    G::VertexId: Eq + Hash + Copy + Sync,
{
    let num_parts = match num_parts_in_pm(pm) {
        Some(n) if n > 0 => n,
        _ => return (0, 0, 1.0),
    };
    let mut loads = vec![0u64; num_parts];
    let verts: Vec<G::VertexId> = g.vertices().collect();
    for v in verts {
        let part = pm.part_of(v);
        loads[part] += g.degree(v) as u64;
    }
    let min = *loads.iter().min().unwrap_or(&0);
    let max = *loads.iter().max().unwrap_or(&0);
    let ratio = if min == 0 {
        f64::INFINITY
    } else {
        max as f64 / (min as f64 + f64::EPSILON)
    };
    (min, max, ratio)
}

/// Snapshot the current resident set size in bytes (Linux only).
#[cfg(feature = "mem-snapshot")]
pub fn memory_snapshot_bytes() -> Option<u64> {
    use std::fs::File;
    use std::io::Read;
    let mut buf = String::new();
    if File::open("/proc/self/statm")
        .and_then(|mut f| f.read_to_string(&mut buf))
        .is_ok()
    {
        if let Some(rss_pages) = buf.split_whitespace().nth(1) {
            if let Ok(pages) = rss_pages.parse::<u64>() {
                // Assume 4 KiB pages
                return Some(pages * 4096);
            }
        }
    }
    None
}

/// Stub when `mem-snapshot` feature is disabled.
#[cfg(not(feature = "mem-snapshot"))]
pub fn memory_snapshot_bytes() -> Option<u64> {
    None
}

// -----------------------------------------------------------------------------
// Tests

#[cfg(test)]
#[cfg(feature = "mpi-support")]
mod tests {
    use super::*;
    use crate::partitioning::graph_traits::PartitionableGraph;
    use rayon::iter::IntoParallelIterator;

    /// Simple test graph using an explicit edge list.
    struct TestGraph {
        edges: Vec<(usize, usize)>,
        n: usize,
    }

    impl PartitionableGraph for TestGraph {
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
            let ns: Vec<_> = self
                .edges
                .iter()
                .filter_map(|&(a, b)| {
                    if a == v {
                        Some(b)
                    } else if b == v {
                        Some(a)
                    } else {
                        None
                    }
                })
                .collect();
            ns.into_iter()
        }

        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }

        fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
            self.edges.clone().into_par_iter()
        }
    }

    #[test]
    fn triangle_same_part() {
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2), (0, 2)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        for v in 0..3 {
            pm.insert(v, 0);
        }
        assert_eq!(edge_cut(&g, &pm), 0);
        let rf = replication_factor(&g, &pm);
        assert!((rf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn two_vertices_different_parts() {
        let g = TestGraph {
            edges: vec![(0, 1)],
            n: 2,
        };
        let mut pm = PartitionMap::with_capacity(2);
        pm.insert(0, 0);
        pm.insert(1, 1);
        assert_eq!(edge_cut(&g, &pm), 1);
        let rf = replication_factor(&g, &pm);
        assert!((rf - 2.0).abs() < 1e-6);
    }

    #[test]
    fn star_graph_replication() {
        // Star: center 0 in part 0, leaves 1..4 in part 1.
        let g = TestGraph {
            edges: vec![(0, 1), (0, 2), (0, 3), (0, 4)],
            n: 5,
        };
        let mut pm = PartitionMap::with_capacity(5);
        pm.insert(0, 0);
        for v in 1..5 {
            pm.insert(v, 1);
        }
        assert_eq!(edge_cut(&g, &pm), 4);
        let rf = replication_factor(&g, &pm);
        assert!((rf - 2.0).abs() < 1e-6);
    }

    #[test]
    fn isolated_vertex() {
        let g = TestGraph {
            edges: vec![],
            n: 1,
        };
        let mut pm = PartitionMap::with_capacity(1);
        pm.insert(0, 0);
        assert_eq!(edge_cut(&g, &pm), 0);
        let rf = replication_factor(&g, &pm);
        assert!((rf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn many_parts_bitset() {
        // Center vertex 0 connected to 199 leaves each with a unique part.
        let mut edges = Vec::new();
        for v in 1..200 {
            edges.push((0, v));
        }
        let g = TestGraph { edges, n: 200 };
        let mut pm = PartitionMap::with_capacity(200);
        for v in 0..200 {
            pm.insert(v, v);
        }
        let rf = replication_factor(&g, &pm);
        // Expected RF ≈ (200 + 199*2) / 200 = 2.99
        assert!((rf - 2.99).abs() < 1e-2);
    }

    #[test]
    fn replication_factor_deterministic() {
        let g = TestGraph {
            edges: vec![(0, 1), (1, 2)],
            n: 3,
        };
        let mut pm = PartitionMap::with_capacity(3);
        pm.insert(0, 0);
        pm.insert(1, 1);
        pm.insert(2, 2);
        let rf1 = replication_factor(&g, &pm);
        let rf2 = replication_factor(&g, &pm);
        assert!((rf1 - rf2).abs() < 1e-12);
    }
}

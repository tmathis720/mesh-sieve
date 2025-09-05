//! Vertex cut construction for distributed graph partitioning.
//!
//! Builds a single primary owner for each vertex and replica lists for
//! cross-part edges. The algorithm proceeds in three phases:
//!   1. Accumulate per-vertex histograms of incident parts.
//!   2. Choose a primary owner for every vertex using load, locality and a
//!      salted hash for deterministic tie breaking.
//!   3. Sweep the edges again to produce replica lists for vertices that
//!      touch different primary parts.

use crate::partitioning::error::PartitionError;
use crate::partitioning::graph_traits::PartitionableGraph;
use crate::partitioning::PartitionMap;
use ahash::AHasher;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};

#[inline]
fn salted_key(salt: u64, v: usize, p: usize) -> u64 {
    let mut h = AHasher::default();
    h.write_u64(salt);
    h.write_u64(v as u64);
    h.write_u64(p as u64);
    h.finish()
}

#[cfg(feature = "mpi-support")]
pub fn build_vertex_cuts<G>(
    graph: &G,
    pm: &PartitionMap<G::VertexId>,
    salt: u64,
) -> Result<(Vec<usize>, Vec<Vec<(G::VertexId, usize)>>), PartitionError>
where
    G: PartitionableGraph<VertexId = usize> + Sync,
{
    // Collect vertices and build index map
    let verts: Vec<G::VertexId> = graph.vertices().collect();
    let n = verts.len();
    let vert_idx: HashMap<G::VertexId, usize> = verts
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    // Determine number of parts
    let num_parts = {
        let mut max = None;
        for (_v, &p) in pm.iter() {
            max = Some(max.map_or(p, |m: usize| m.max(p)));
        }
        match max {
            Some(m) => m + 1,
            None => return Err(PartitionError::NoParts),
        }
    };

    type PartId = usize;
    type VIdx = usize;

    #[inline]
    fn bump(map: &mut HashMap<PartId, u32>, p: PartId) {
        *map.entry(p).or_insert(0) = map.get(&p).copied().unwrap_or(0).saturating_add(1);
    }

    struct LocalHist {
        map: HashMap<VIdx, HashMap<PartId, u32>>,
    }

    // First pass: accumulate incident-part histograms
    let hist = graph
        .edges()
        .fold(
            || LocalHist {
                map: HashMap::new(),
            },
            |mut acc, (u, v)| {
                let pu = *pm
                    .get(&u)
                    .ok_or(PartitionError::MissingPartition(u))
                    .unwrap();
                let pv = *pm
                    .get(&v)
                    .ok_or(PartitionError::MissingPartition(v))
                    .unwrap();
                let ui = *vert_idx
                    .get(&u)
                    .ok_or(PartitionError::VertexNotFound(u))
                    .unwrap();
                let vi = *vert_idx
                    .get(&v)
                    .ok_or(PartitionError::VertexNotFound(v))
                    .unwrap();

                bump(acc.map.entry(ui).or_insert_with(HashMap::new), pv);
                bump(acc.map.entry(vi).or_insert_with(HashMap::new), pu);
                acc
            },
        )
        .reduce(
            || LocalHist {
                map: HashMap::new(),
            },
            |mut a, b| {
                for (vx, hb) in b.map {
                    let ha = a.map.entry(vx).or_insert_with(HashMap::new);
                    for (p, c) in hb {
                        *ha.entry(p).or_insert(0) =
                            ha.get(&p).copied().unwrap_or(0).saturating_add(c);
                    }
                }
                a
            },
        )
        .map;

    // Second pass: choose primary owner for each vertex
    let part_owner_load: Vec<AtomicUsize> = (0..num_parts).map(|_| AtomicUsize::new(0)).collect();
    let mut primary: Vec<usize> = vec![usize::MAX; n];

    #[cfg(not(feature = "deterministic-owners"))]
    {
        primary.par_iter_mut().enumerate().for_each(|(ui, slot)| {
            let v = verts[ui];
            let mut cand = hist.get(&ui).cloned().unwrap_or_default();
            let own = *pm.get(&v).expect("partition missing for vertex");
            cand.entry(own).or_insert(0);

            let mut best: Option<(usize, usize, u64, PartId)> = None;
            for (&p, &deg) in &cand {
                let load = part_owner_load[p].load(Ordering::Relaxed);
                let key = salted_key(salt, v, p);
                let t = (load, usize::MAX - deg as usize, key, p);
                if best.map_or(true, |b| t < (b.0, b.1, b.2, b.3)) {
                    best = Some(t);
                }
            }
            let chosen = best.unwrap().3;
            part_owner_load[chosen].fetch_add(1, Ordering::Relaxed);
            *slot = chosen;
        });
    }

    #[cfg(feature = "deterministic-owners")]
    {
        let mut loads = vec![0usize; num_parts];
        for ui in 0..n {
            let v = verts[ui];
            let mut cand = hist.get(&ui).cloned().unwrap_or_default();
            let own = *pm.get(&v).ok_or(PartitionError::MissingPartition(v))?;
            cand.entry(own).or_insert(0);
            let mut best: Option<(usize, usize, u64, PartId)> = None;
            for (&p, &deg) in &cand {
                let key = salted_key(salt, v, p);
                let t = (loads[p], usize::MAX - deg as usize, key, p);
                if best.map_or(true, |b| t < (b.0, b.1, b.2, b.3)) {
                    best = Some(t);
                }
            }
            let chosen = best.unwrap().3;
            loads[chosen] += 1;
            primary[ui] = chosen;
        }
    }

    // Third pass: construct replica lists
    struct LocalRep<G: PartitionableGraph> {
        map: HashMap<VIdx, Vec<(G::VertexId, usize)>>,
    }

    let rep_map = graph
        .edges()
        .fold(
            || LocalRep::<G> {
                map: HashMap::new(),
            },
            |mut acc, (u, v)| {
                let ui = *vert_idx
                    .get(&u)
                    .ok_or(PartitionError::VertexNotFound(u))
                    .unwrap();
                let vi = *vert_idx
                    .get(&v)
                    .ok_or(PartitionError::VertexNotFound(v))
                    .unwrap();
                let pu = primary[ui];
                let pv = primary[vi];
                if pu != pv {
                    acc.map.entry(ui).or_default().push((v, pv));
                    acc.map.entry(vi).or_default().push((u, pu));
                }
                acc
            },
        )
        .reduce(
            || LocalRep::<G> {
                map: HashMap::new(),
            },
            |mut a, b| {
                for (vx, mut list) in b.map {
                    a.map.entry(vx).or_default().extend(list.drain(..));
                }
                a
            },
        )
        .map;

    let mut replicas: Vec<Vec<(G::VertexId, usize)>> = vec![Vec::new(); n];
    for (ui, mut list) in rep_map {
        list.sort_unstable();
        list.dedup();
        replicas[ui] = list;
    }

    Ok((primary, replicas))
}

#[cfg(test)]
#[cfg(feature = "mpi-support")]
mod tests {
    use super::*;
    use rayon::iter::IntoParallelIterator;

    /// Simple graph implementation for testing
    struct TestGraph {
        edges: Vec<(usize, usize)>,
        n: usize,
    }

    impl TestGraph {
        fn new(edges: Vec<(usize, usize)>, n: usize) -> Self {
            TestGraph { edges, n }
        }
    }

    impl PartitionableGraph for TestGraph {
        type VertexId = usize;
        type VertexParIter<'a>
            = rayon::vec::IntoIter<usize>
        where
            Self: 'a;
        type NeighParIter<'a>
            = rayon::vec::IntoIter<usize>
        where
            Self: 'a;
        type NeighIter<'a>
            = std::vec::IntoIter<usize>
        where
            Self: 'a;
        type EdgeParIter<'a>
            = rayon::vec::IntoIter<(usize, usize)>
        where
            Self: 'a;

        fn vertices(&self) -> Self::VertexParIter<'_> {
            (0..self.n).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors(&self, v: usize) -> Self::NeighParIter<'_> {
            self.neighbors_seq(v).collect::<Vec<_>>().into_par_iter()
        }
        fn neighbors_seq(&self, v: usize) -> Self::NeighIter<'_> {
            let ns = self
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
                .collect::<Vec<_>>();
            ns.into_iter()
        }
        fn degree(&self, v: usize) -> usize {
            self.neighbors_seq(v).count()
        }
        fn edges(&self) -> Self::EdgeParIter<'_> {
            self.edges.clone().into_par_iter()
        }
    }

    #[test]
    fn vertex_cut_cycle_replicas() {
        // 4-cycle: 0-1-2-3-0 with two parts: {0,1}=0 and {2,3}=1
        let g = TestGraph::new(vec![(0, 1), (1, 2), (2, 3), (3, 0)], 4);
        let mut pm = PartitionMap::with_capacity(4);
        pm.insert(0, 0);
        pm.insert(1, 0);
        pm.insert(2, 1);
        pm.insert(3, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 42).unwrap();

        // Cross edges (1,2) and (3,0) should create replicas on both endpoints
        for &(u, v) in &[(1, 2), (3, 0)] {
            let pu = primary[u];
            let pv = primary[v];
            assert!(
                replicas[u].contains(&(v, pv)),
                "missing replica for {u}->{v}"
            );
            assert!(
                replicas[v].contains(&(u, pu)),
                "missing replica for {v}->{u}"
            );
        }
    }

    #[test]
    fn vertex_cut_isolated_vertex() {
        let g = TestGraph::new(vec![], 1);
        let mut pm = PartitionMap::with_capacity(1);
        pm.insert(0, 0);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 99).unwrap();
        assert_eq!(primary, vec![0]);
        assert!(replicas[0].is_empty());
    }

    #[test]
    fn vertex_cut_two_node_replicas() {
        let g = TestGraph::new(vec![(0, 1)], 2);
        let mut pm = PartitionMap::with_capacity(2);
        pm.insert(0, 0);
        pm.insert(1, 1);
        let (primary, replicas) = build_vertex_cuts(&g, &pm, 0).unwrap();

        assert!(replicas[0].contains(&(1, primary[1])));
        assert!(replicas[1].contains(&(0, primary[0])));
    }
}

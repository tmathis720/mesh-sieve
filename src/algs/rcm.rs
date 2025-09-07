//! Distributed Reverse Cuthill-McKee (RCM) reordering for mesh-sieve.
//!
//! Implements distributed RCM as described in Azad et al., using the Sieve and Communicator abstractions.
//!
//! See: Algorithm 2 (pseudo-peripheral vertex finder) and Algorithm 3 (RCM proper) in the reference.

use crate::algs::communicator::Communicator;
use crate::prelude::Sieve;
use std::collections::HashMap;

/// Which neighbor relation to use for RCM.
/// `Undirected` = union(cone, support), unique and sorted (deterministic).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RcmAdjacency {
    /// Only outgoing arrows (cone).
    Down,
    /// Only incoming arrows (support).
    Up,
    /// Union of outgoing and incoming arrows.
    Undirected,
}

/// Compute a distributed‐RCM ordering of the point‐graph in `sieve`.
/// Returns a Vec<Point> of length = number of points, giving the new 0‐based labels.
///
/// See: Algorithm 2 (pseudo-peripheral vertex finder) and Algorithm 3 (RCM proper) in the reference.
///
/// # Arguments
/// * `sieve` - The mesh Sieve structure implementing the point-graph.
/// * `comm` - The communicator (NoComm, RayonComm, or MpiComm) for distributed execution.
///
/// # Returns
/// A vector of points in RCM order (new 0-based labels).
///
/// # Example
/// ```text
/// // See examples/distributed_rcm.rs for usage.
/// ```
/// Back-compatible default: undirected adjacency (cone ∪ support).
pub fn distributed_rcm<S, C>(sieve: &S, comm: &C) -> Vec<<S as Sieve>::Point>
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    distributed_rcm_with(sieve, comm, RcmAdjacency::Undirected)
}

/// Explicit-adjacency variant of RCM.
pub fn distributed_rcm_with<S, C>(
    sieve: &S,
    comm: &C,
    adj: RcmAdjacency,
) -> Vec<<S as Sieve>::Point>
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    let prims = RcmPrims::new_with(sieve, comm, adj);
    run_rcm(prims)
}

fn run_rcm<S, C>(prims: RcmPrims<S, C>) -> Vec<<S as Sieve>::Point>
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    let n = prims.idx_to_point.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Pseudo-peripheral root
    let root_idx = find_pseudo_peripheral_root(&prims);

    // 2. RCM BFS
    let mut labels = vec![-1isize; n];
    let mut visited = vec![false; n];
    let mut frontier = vec![root_idx];
    visited[root_idx] = true;
    labels[root_idx] = 0;
    let mut next_label: isize = 1;

    while !frontier.is_empty() {
        let mut next_pairs = Vec::new();
        for &u in &frontier {
            let parent_label = labels[u];
            for &v in &prims.adj[u] {
                if !visited[v] {
                    next_pairs.push((v, parent_label));
                }
            }
        }

        // Reduce to smallest parent label per vertex
        next_pairs.sort_unstable_by_key(|&(v, lbl)| (v, lbl));
        next_pairs.dedup_by_key(|pair| pair.0);

        // Order by (parent_label, degree, vertex)
        let mut triples: Vec<(isize, usize, usize)> = next_pairs
            .into_iter()
            .map(|(v, lbl)| (lbl, prims.degree[v], v))
            .collect();
        triples.sort_unstable();
        let ordered_vs: Vec<usize> = triples.into_iter().map(|t| t.2).collect();

        for &v in &ordered_vs {
            visited[v] = true;
            labels[v] = next_label;
            next_label += 1;
        }

        frontier = ordered_vs;
    }

    debug_validate_labels_contiguous(&labels);

    // Reverse by label
    let mut label_point: Vec<(isize, <S as Sieve>::Point)> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, prims.idx_to_point[i]))
        .collect();
    label_point.sort_by_key(|&(l, _)| -l);
    let perm: Vec<_> = label_point.into_iter().map(|(_, p)| p).collect();

    debug_validate_permutation(&prims.idx_to_point, &perm);

    perm
}

#[inline]
fn debug_validate_labels_contiguous(labels: &[isize]) {
    #[cfg(any(debug_assertions, feature = "check-rcm"))]
    {
        let n = labels.len();
        let mut seen = vec![false; n];
        for &l in labels {
            assert!(l >= 0 && (l as usize) < n, "RCM: non-contiguous or out-of-range label {l}");
            let i = l as usize;
            assert!(!seen[i], "RCM: duplicate label {i}");
            seen[i] = true;
        }
        assert!(seen.into_iter().all(|b| b), "RCM: not all labels set");
    }
}

#[inline]
fn debug_validate_permutation<P: Eq + Copy + std::hash::Hash + std::fmt::Debug>(
    universe: &[P],
    perm: &[P],
) {
    #[cfg(any(debug_assertions, feature = "check-rcm"))]
    {
        use std::collections::HashSet;
        assert_eq!(perm.len(), universe.len(), "RCM: perm length mismatch");
        let a: HashSet<_> = universe.iter().copied().collect();
        let b: HashSet<_> = perm.iter().copied().collect();
        assert_eq!(a, b, "RCM: perm is not a reordering of base points");
    }
}

/// Primitives and mappings for distributed RCM computation.
pub struct RcmPrims<'a, S: Sieve, C: Communicator> {
    pub sieve: &'a S,
    pub comm: &'a C,
    /// Map from Point to local vertex index [0, n)
    pub point_to_idx: HashMap<<S as Sieve>::Point, usize>,
    /// Map from local vertex index to Point
    pub idx_to_point: Vec<<S as Sieve>::Point>,
    /// Degree vector: degree[i] = adj[i].len()
    pub degree: Vec<usize>,
    /// Adjacency list by local index, sorted and deduplicated
    pub adj: Vec<Vec<usize>>,
    /// Mode used to build adjacency
    pub mode: RcmAdjacency,
}

impl<'a, S, C> RcmPrims<'a, S, C>
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    /// Build RcmPrims from a Sieve and Communicator
    pub fn new(sieve: &'a S, comm: &'a C) -> Self {
        Self::new_with(sieve, comm, RcmAdjacency::Undirected)
    }

    pub fn new_with(sieve: &'a S, comm: &'a C, mode: RcmAdjacency) -> Self {
        // 1) collect base points and maps
        let all_points: Vec<<S as Sieve>::Point> = sieve.base_points().collect();
        let n = all_points.len();

        let mut point_to_idx = HashMap::with_capacity(n);
        let mut idx_to_point = Vec::with_capacity(n);
        for (i, p) in all_points.iter().enumerate() {
            point_to_idx.insert(*p, i);
            idx_to_point.push(*p);
        }

        // 2) build adjacency according to mode
        let mut adj = vec![Vec::<usize>::new(); n];
        for (i, &p) in idx_to_point.iter().enumerate() {
            let mut nbr_pts: Vec<<S as Sieve>::Point> = match mode {
                RcmAdjacency::Down => sieve.cone_points(p).collect(),
                RcmAdjacency::Up => sieve.support_points(p).collect(),
                RcmAdjacency::Undirected => {
                    let mut v: Vec<_> = sieve.cone_points(p).collect();
                    v.extend(sieve.support_points(p));
                    v
                }
            };

            nbr_pts.retain(|&q| q != p);
            nbr_pts.sort_unstable();
            nbr_pts.dedup();

            let mut nbr_idx = Vec::with_capacity(nbr_pts.len());
            for q in nbr_pts {
                if let Some(&j) = point_to_idx.get(&q) {
                    nbr_idx.push(j);
                }
            }
            nbr_idx.sort_unstable();
            adj[i] = nbr_idx;
        }

        let degree = adj.iter().map(|v| v.len()).collect();

        Self { sieve, comm, point_to_idx, idx_to_point, degree, adj, mode }
    }

    #[inline]
    pub fn neighbors_idx(&self, u: usize) -> &[usize] {
        &self.adj[u]
    }

}

/// Find a pseudo-peripheral root vertex using Algorithm 2 (Azad et al.).
pub fn find_pseudo_peripheral_root<S, C>(prims: &RcmPrims<S, C>) -> usize
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    let n = prims.idx_to_point.len();
    if n == 0 { return 0; }
    let mut r = 0usize;
    let mut last_lvl = 0usize;
    loop {
        let mut level: Vec<Vec<usize>> = vec![vec![r]];
        let mut seen = vec![false; n];
        seen[r] = true;

        while let Some(curr) = level.last() {
            if curr.is_empty() { break; }
            let mut next = Vec::new();
            for &u in curr {
                for &v in &prims.adj[u] {
                    if !seen[v] {
                        seen[v] = true;
                        next.push(v);
                    }
                }
            }
            if next.is_empty() { break; }
            next.sort_unstable();
            next.dedup();
            level.push(next);
        }

        let lvl = level.len();
        let last_layer = &level[lvl - 1];
        let &r_prime = last_layer.iter().min_by_key(|&&v| prims.degree[v]).unwrap_or(&r);
        if lvl <= last_lvl { break; }
        last_lvl = lvl;
        r = r_prime;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::topology::cache::InvalidateCache;

    #[derive(Debug, Default)]
    struct MockSieve {
        cone_adj: Vec<HashSet<usize>>,
        support_adj: Vec<HashSet<usize>>,
    }

    impl InvalidateCache for MockSieve {
        fn invalidate_cache(&mut self) {}
    }

    impl Sieve for MockSieve {
        type Point = usize;
        type Payload = ();
        type ConeIter<'a> = Box<dyn Iterator<Item = (Self::Point, Self::Payload)> + 'a>;
        type SupportIter<'a> = Box<dyn Iterator<Item = (Self::Point, Self::Payload)> + 'a>;

        fn base_points(&self) -> Box<dyn Iterator<Item = Self::Point> + '_> {
            Box::new(0..self.cone_adj.len())
        }
        fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a> {
            Box::new(self.cone_adj[p].iter().map(move |&q| (q, ())))
        }
        fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a> {
            Box::new(self.support_adj[p].iter().map(move |&q| (q, ())))
        }
        fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, _payload: Self::Payload) {
            self.cone_adj[src].insert(dst);
            self.support_adj[dst].insert(src);
        }
        fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload> {
            let removed = self.cone_adj[src].remove(&dst);
            self.support_adj[dst].remove(&src);
            if removed { Some(()) } else { None }
        }
        fn cap_points(&self) -> Box<dyn Iterator<Item = Self::Point> + '_> {
            Box::new(0..self.support_adj.len())
        }
    }

    // Minimal mock Communicator
    struct NoComm;

    trait Wait {
        fn wait(self);
    }

    impl Wait for () {
        fn wait(self) {}
    }

    impl Communicator for NoComm {
        type SendHandle = ();
        type RecvHandle = ();

        fn isend(&self, _peer: usize, _tag: u16, _buf: &[u8]) -> Self::SendHandle { () }
        fn irecv(&self, _peer: usize, _tag: u16, _buf: &mut [u8]) -> Self::RecvHandle { () }
        fn rank(&self) -> usize { 0 }
        fn size(&self) -> usize { 1 }
    }

    fn make_line_graph(n: usize) -> MockSieve {
        let mut s = MockSieve {
            cone_adj: vec![HashSet::new(); n],
            support_adj: vec![HashSet::new(); n],
        };
        for i in 0..n - 1 {
            s.add_arrow(i, i + 1, ());
            s.add_arrow(i + 1, i, ());
        }
        s
    }

    #[test]
    fn test_rcm_line_graph() {
        let sieve = make_line_graph(5);
        let comm = NoComm;
        let order = distributed_rcm(&sieve, &comm);
        // Should be a permutation of 0..5
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        // RCM order is not unique; do not assert a specific order
    }

    #[test]
    fn test_rcm_empty_graph() {
        let sieve = MockSieve { cone_adj: vec![], support_adj: vec![] };
        let comm = NoComm;
        let order = distributed_rcm(&sieve, &comm);
        assert!(order.is_empty());
    }

    #[test]
    fn test_find_pseudo_peripheral_root() {
        let sieve = make_line_graph(4);
        let comm = NoComm;
        let prims = RcmPrims::new(&sieve, &comm);
        let root = find_pseudo_peripheral_root(&prims);
        // On a line, root should be one of the endpoints (0 or n-1)
        assert!(root == 0 || root == 3);
    }

    #[test]
    fn test_rcm_star_graph() {
        // Star: 0 connected to 1,2,3,4
        let mut s = MockSieve { cone_adj: vec![HashSet::new(); 5], support_adj: vec![HashSet::new(); 5] };
        for i in 1..5 {
            s.add_arrow(0, i, ());
            s.add_arrow(i, 0, ());
        }
        let sieve = s;
        let comm = NoComm;
        let order = distributed_rcm(&sieve, &comm);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn rcm_default_is_undirected() {
        // Directed edges 0->1, 2->1 to exercise support edges
        let mut s = MockSieve { cone_adj: vec![HashSet::new(); 3], support_adj: vec![HashSet::new(); 3] };
        s.add_arrow(0, 1, ());
        s.add_arrow(2, 1, ());
        let comm = NoComm;
        let def = distributed_rcm(&s, &comm);
        let und = distributed_rcm_with(&s, &comm, RcmAdjacency::Undirected);
        assert_eq!(def, und);
    }

    #[test]
    fn rcm_labels_contiguous() {
        let sieve = make_line_graph(4);
        let comm = NoComm;
        let perm = distributed_rcm(&sieve, &comm);
        let mut set = HashSet::new();
        for p in &perm {
            set.insert(*p);
        }
        assert_eq!(set.len(), sieve.base_points().count());
    }

    #[test]
    fn rcm_explicit_modes_match_expectations() {
        let sieve = make_line_graph(5);
        let comm = NoComm;
        let und = distributed_rcm_with(&sieve, &comm, RcmAdjacency::Undirected);
        let dn = distributed_rcm_with(&sieve, &comm, RcmAdjacency::Down);
        let up = distributed_rcm_with(&sieve, &comm, RcmAdjacency::Up);
        assert_eq!(und.len(), dn.len());
        assert_eq!(und.len(), up.len());
    }
}

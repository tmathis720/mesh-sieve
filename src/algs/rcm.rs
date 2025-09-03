//! Distributed Reverse Cuthill-McKee (RCM) reordering for mesh-sieve.
//!
//! Implements distributed RCM as described in Azad et al., using the Sieve and Communicator abstractions.
//!
//! See: Algorithm 2 (pseudo-peripheral vertex finder) and Algorithm 3 (RCM proper) in the reference.

use crate::algs::communicator::Communicator;
use crate::prelude::Sieve;
use std::collections::HashMap;

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
pub fn distributed_rcm<S, C>(
    sieve: &S,
    comm: &C,
) -> Vec<<S as Sieve>::Point>
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    let prims = RcmPrims::new(sieve, comm);
    let n = prims.idx_to_point.len();
    if n == 0 {
        return Vec::new();
    }
    // 1. Find pseudo-peripheral root
    let root_idx = find_pseudo_peripheral_root(&prims);
    // 2. RCM proper (Algorithm 3)
    let mut labels = vec![-1isize; n];
    labels[root_idx] = 0;
    let mut nv = 1isize;
    let mut frontier = vec![root_idx];
    let mut visited = vec![false; n];
    visited[root_idx] = true;
    while !frontier.is_empty() {
        // a. SET current frontier labels
        for &v in &frontier {
            labels[v] = nv;
        }
        nv += frontier.len() as isize;
        // b. SPMSPV: get (v, parent_label) for all unvisited neighbors
        let mut next_pairs = prims.spmspv(&frontier, &labels, &visited);
        // c. Remove already visited
        next_pairs.retain(|&(v, _)| labels[v] == -1);
        // d. REDUCE: min label for each v
        let reduced = RcmPrims::<S, C>::reduce_min_label(&next_pairs);
        // e. SORTPERM: sort by (parent_label, degree, v)
        let triples: Vec<(isize, usize, usize)> = reduced
            .iter()
            .map(|(&v, &label)| (label, prims.degree[v], v))
            .collect();
        let ordered_vs = RcmPrims::<S, C>::sortperm(&triples);
        // f. Mark as visited and set labels
        for &v in &ordered_vs {
            visited[v] = true;
        }
        frontier = ordered_vs;
    }
    // 3. Return points in reverse label order
    let mut label_point: Vec<(isize, <S as Sieve>::Point)> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, prims.idx_to_point[i]))
        .collect();
    label_point.sort_by_key(|&(l, _)| -l);
    label_point.iter().map(|&(_, p)| p).collect()
}

/// Primitives and mappings for distributed RCM computation.
pub struct RcmPrims<'a, S: Sieve, C: Communicator> {
    pub sieve: &'a S,
    pub comm: &'a C,
    /// Map from Point to local vertex index [0, n)
    pub point_to_idx: HashMap<<S as Sieve>::Point, usize>,
    /// Map from local vertex index to Point
    pub idx_to_point: Vec<<S as Sieve>::Point>,
    /// Degree vector: degree[i] = out-degree of vertex i
    pub degree: Vec<usize>,
}

impl<'a, S, C> RcmPrims<'a, S, C>
where
    S: Sieve,
    <S as Sieve>::Point: Ord,
    C: Communicator,
{
    /// Build RcmPrims from a Sieve and Communicator
    pub fn new(sieve: &'a S, comm: &'a C) -> Self {
        // Collect all base points (vertices)
        let all_points: Vec<<S as Sieve>::Point> = sieve.base_points().collect();
        let n = all_points.len();
        // Map Point <-> idx
        let mut point_to_idx = HashMap::with_capacity(n);
        let mut idx_to_point = Vec::with_capacity(n);
        for (i, p) in all_points.iter().enumerate() {
            point_to_idx.insert(*p, i);
            idx_to_point.push(*p);
        }
        // Build degree vector
        let mut degree = vec![0; n];
        for (i, p) in all_points.iter().enumerate() {
            degree[i] = sieve.cone(*p).count();
        }
        Self {
            sieve,
            comm,
            point_to_idx,
            idx_to_point,
            degree,
        }
    }

    /// IND: Identity mapping (frontier is Vec<usize> of vertex indices)
    pub fn ind(frontier: &[usize]) -> Vec<usize> {
        frontier.to_vec()
    }

    /// SELECT: Filter a frontier by a predicate and visited set
    pub fn select(f: &[usize], visited: &[bool], cond: impl Fn(usize) -> bool) -> Vec<usize> {
        f.iter().copied().filter(|&v| !visited[v] && cond(v)).collect()
    }

    /// SET: Set labels for a dense label vector
    pub fn set_labels(labels: &mut [isize], indices: &[usize], value: isize) {
        for &v in indices {
            labels[v] = value;
        }
    }

    /// SPMSPV: Given a frontier, find all unvisited neighbors and their parent labels
    /// (local part; distributed part to be handled in full algorithm)
    pub fn spmspv(&self, frontier: &[usize], labels: &[isize], visited: &[bool]) -> Vec<(usize, isize)> {
        let mut out = Vec::new();
        for &u in frontier {
            let point = self.idx_to_point[u];
            for (v_point, _) in self.sieve.cone(point) {
                if let Some(&v) = self.point_to_idx.get(&v_point) {
                    if !visited[v] {
                        out.push((v, labels[u]));
                    }
                }
            }
        }
        // In distributed: AllGather/AlltoAll and min-reduce by v
        out
    }

    /// REDUCE: Reduce (v, label) pairs by v -> min(label)
    pub fn reduce_min_label(pairs: &[(usize, isize)]) -> HashMap<usize, isize> {
        let mut map: HashMap<usize, isize> = HashMap::new();
        for &(v, label) in pairs {
            map.entry(v).and_modify(|l: &mut isize| *l = (*l).min(label)).or_insert(label);
        }
        map
    }

    /// SORTPERM: Sort (label, degree, v) triples and return ordered vertex indices
    pub fn sortperm(triples: &[(isize, usize, usize)]) -> Vec<usize> {
        let mut sorted = triples.to_vec();
        sorted.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        sorted.into_iter().map(|t| t.2).collect()
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
    let mut visited = vec![false; n];
    // Pick an arbitrary start vertex (first base point)
    let mut r = 0;
    let mut last_lvl = 0;
    loop {
        // Level structure via BFS
        let mut level = vec![vec![r]];
        visited.fill(false);
        visited[r] = true;
        while !level.last().unwrap().is_empty() {
            let mut next = Vec::new();
            for &u in level.last().unwrap() {
                for (v_point, _) in prims.sieve.cone(prims.idx_to_point[u]) {
                    if let Some(&v) = prims.point_to_idx.get(&v_point) {
                        if !visited[v] {
                            visited[v] = true;
                            next.push(v);
                        }
                    }
                }
            }
            if next.is_empty() { break; }
            level.push(next);
        }
        let lvl = level.len();
        // From last level, pick vertex of minimum degree
        let last_level = &level[lvl - 1];
        let &r_prime = last_level.iter().min_by_key(|&&v| prims.degree[v]).unwrap_or(&r);
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

    // Minimal mock Sieve for testing: undirected graph as adjacency list
    use crate::topology::cache::InvalidateCache;

    #[derive(Debug, Default)]
    struct MockSieve {
        adj: Vec<HashSet<usize>>,
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
            Box::new(0..self.adj.len())
        }
        fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a> {
            Box::new(self.adj[p].iter().map(move |&q| (q, ())))
        }
        fn support<'a>(&'a self, _p: Self::Point) -> Self::SupportIter<'a> {
            // For undirected mock, support is the same as cone
            Box::new(std::iter::empty())
        }
        fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, _payload: Self::Payload) {
            self.adj[src].insert(dst);
        }
        fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload> {
            if self.adj[src].remove(&dst) {
                Some(())
            } else {
                None
            }
        }
        fn cap_points(&self) -> Box<dyn Iterator<Item = Self::Point> + '_> {
            Box::new(0..self.adj.len())
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
        let mut adj = vec![HashSet::new(); n];
        for i in 0..n {
            if i > 0 {
                adj[i].insert(i - 1);
            }
            if i + 1 < n {
                adj[i].insert(i + 1);
            }
        }
        MockSieve { adj }
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
        let sieve = MockSieve { adj: vec![] };
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
        let mut adj = vec![HashSet::new(); 5];
        for i in 1..5 {
            adj[0].insert(i);
            adj[i].insert(0);
        }
        let sieve = MockSieve { adj };
        let comm = NoComm;
        let order = distributed_rcm(&sieve, &comm);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}

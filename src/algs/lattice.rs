//! Set-lattice helpers: meet, join, adjacency and helpers.
//! All output vectors are **sorted & deduplicated** for deterministic behaviour.

use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

type P = PointId;

/// Controls how to form the "boundary" of `p` that defines adjacency:
/// - If `max_down_depth = Some(1)`, neighbors are those sharing a **face** (FV style).
/// - If `max_down_depth = Some(2)`, also through vertices (typical 2D FE).
/// - If `max_down_depth = None`, full transitive closure (all lower strata).
#[derive(Clone, Copy, Debug)]
pub struct AdjacencyOpts {
    pub max_down_depth: Option<u32>,
}

impl Default for AdjacencyOpts {
    fn default() -> Self { Self { max_down_depth: Some(1) } }
}

fn boundary_points<S>(sieve: &S, p: P, max_down_depth: Option<u32>) -> Vec<P>
where S: Sieve<Point = P>
{
    use std::collections::{HashSet, VecDeque};
    match max_down_depth {
        None => {
            let mut v: Vec<_> = sieve.cone_points(p).collect();
            let mut seen: HashSet<_> = v.iter().copied().collect();
            let mut q: VecDeque<_> = v.iter().copied().map(|x| (x, 1)).collect();
            while let Some((r, _d)) = q.pop_front() {
                for s in sieve.cone_points(r) {
                    if seen.insert(s) { v.push(s); q.push_back((s, 0)); }
                }
            }
            v.sort_unstable(); v.dedup(); v
        }
        Some(k) => {
            if k == 0 { return vec![]; }
            let mut out = Vec::new();
            let mut seen = HashSet::new();
            let mut q = VecDeque::from_iter(sieve.cone_points(p).map(|x| (x, 1)));
            while let Some((r, d)) = q.pop_front() {
                if seen.insert(r) { out.push(r); }
                if d < k {
                    for s in sieve.cone_points(r) { q.push_back((s, d+1)); }
                }
            }
            out.sort_unstable(); out.dedup(); out
        }
    }
}

/// Cells adjacent to `p` according to the policy.
/// Adjacent = share any boundary point in the chosen boundary set.
pub fn adjacent_with<S>(sieve: &S, p: P, opts: AdjacencyOpts) -> Vec<P>
where S: Sieve<Point = P>
{
    use std::collections::HashSet;
    let boundary = boundary_points(sieve, p, opts.max_down_depth);
    let mut neigh = HashSet::new();
    for b in boundary {
        for (cell, _) in sieve.support(b) {
            if cell != p { neigh.insert(cell); }
        }
    }
    let mut out: Vec<P> = neigh.into_iter().collect();
    out.sort_unstable();
    out
}

/// Backward-compatible default: neighbors across **faces** only (FV style).
pub fn adjacent<S>(sieve: &S, p: P) -> Vec<P>
where S: Sieve<Point = P>
{
    adjacent_with(sieve, p, AdjacencyOpts::default())
}

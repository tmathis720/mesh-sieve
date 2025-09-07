//! Set-lattice helpers: meet, join, adjacency and helpers.
//!
//! *Adjacency semantics*
//! - Two `cells` (points at the same height) are adjacent if they share any
//!   boundary point found by walking down from a cell by `max_down_depth`
//!   levels.
//! - `Some(1)` &rarr; faces only (finite-volume style, and the default).
//! - `Some(2)` &rarr; faces and vertices (typical 2D finite-element).
//! - `Some(0)` &rarr; empty boundary (no adjacency).
//! - `None` &rarr; full transitive down-closure (potentially expensive).
//!
//! *Determinism*
//! - All returned vectors are sorted ascending and deduplicated.
//!
//! *Cycle safety*
//! - A visited set is always used during the down-walk to prevent infinite
//!   loops on cyclic inputs.
//!
//! *Same-stratum adjacency*
//! - `support(b)` may return parents at different heights. For mesh
//!   cell-to-cell adjacency we filter neighbors to the same height as the seed
//!   cell by default. Disable this via [`AdjacencyOpts::same_stratum_only`]
//!   when cross-stratum relations are desired.

use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::strata::compute_strata;

type P = PointId;

/// Controls how to form the "boundary" of `p` that defines adjacency:
/// - If `max_down_depth = Some(1)`, neighbors are those sharing a **face**.
/// - If `max_down_depth = Some(2)`, also through vertices.
/// - If `max_down_depth = None`, full transitive closure (all lower strata).
#[derive(Clone, Copy, Debug)]
pub struct AdjacencyOpts {
    pub max_down_depth: Option<u32>,
    /// When `true`, only return neighbors with the same height as `p`.
    /// This is correct for cell-to-cell adjacency in meshes.
    pub same_stratum_only: bool,
}

impl Default for AdjacencyOpts {
    fn default() -> Self {
        Self { max_down_depth: Some(1), same_stratum_only: true }
    }
}

#[inline]
fn boundary_points<S>(sieve: &S, p: P, max_down_depth: Option<u32>) -> Vec<P>
where
    S: Sieve<Point = P>,
{
    use std::collections::{HashSet, VecDeque};

    match max_down_depth {
        // No boundary
        Some(0) => Vec::new(),

        // Faces only (most common)
        Some(1) => {
            let mut v: Vec<P> = sieve.cone_points(p).collect();
            v.sort_unstable();
            v.dedup();
            v
        }

        // Faces + vertices (common in 2D FE)
        Some(2) => {
            let mut out = Vec::new();
            let mut seen: HashSet<P> = HashSet::with_capacity(16);
            let mut q: VecDeque<(P, u32)> = VecDeque::with_capacity(16);
            q.extend(sieve.cone_points(p).map(|x| (x, 1)));

            while let Some((r, d)) = q.pop_front() {
                if seen.insert(r) {
                    out.push(r);
                    if d < 2 {
                        for s in sieve.cone_points(r) {
                            if !seen.contains(&s) {
                                q.push_back((s, d + 1));
                            }
                        }
                    }
                }
            }
            out.sort_unstable();
            out.dedup();
            out
        }

        // Full down-closure (or arbitrary depth > 2)
        None | Some(_) => {
            let limit = max_down_depth.unwrap_or(u32::MAX);
            let mut out = Vec::new();
            let mut seen: HashSet<P> = HashSet::with_capacity(64);
            let mut q: VecDeque<(P, u32)> = VecDeque::with_capacity(64);
            q.extend(sieve.cone_points(p).map(|x| (x, 1)));

            while let Some((r, d)) = q.pop_front() {
                if seen.insert(r) {
                    out.push(r);
                    if d < limit {
                        for s in sieve.cone_points(r) {
                            if !seen.contains(&s) {
                                q.push_back((s, d + 1));
                            }
                        }
                    }
                }
            }
            out.sort_unstable();
            out.dedup();
            out
        }
    }
}

/// Cells adjacent to `p` according to the policy.
/// Adjacent = share any boundary point in the chosen boundary set.
/// Respects [`AdjacencyOpts::same_stratum_only`].
#[inline]
pub fn adjacent_with<S>(sieve: &S, p: P, opts: AdjacencyOpts) -> Vec<P>
where
    S: Sieve<Point = P>,
{
    use std::collections::HashSet;

    let height_map = if opts.same_stratum_only {
        compute_strata(sieve).ok().map(|cache| cache.height)
    } else {
        None
    };
    let seed_height = height_map.as_ref().and_then(|hm| hm.get(&p).copied());

    let boundary = boundary_points(sieve, p, opts.max_down_depth);
    let mut neigh = HashSet::new();
    for b in boundary {
        for (cell, _) in sieve.support(b) {
            if cell == p {
                continue;
            }
            if let (Some(hp), Some(hm)) = (seed_height, height_map.as_ref())
                && hm.get(&cell).copied() != Some(hp) {
                    continue;
                }
            neigh.insert(cell);
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

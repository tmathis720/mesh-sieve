//! Clone-free DFS/BFS helpers for Sieves that implement `SieveRef`.

use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::SieveRef;
use std::collections::{HashSet, VecDeque};

pub type Point = PointId;

/// Generic DFS over point-only neighbors.
fn dfs_points<S, I, F>(sieve: &S, seeds: I, mut nbrs: F) -> Vec<Point>
where
    S: Sieve<Point = Point> + SieveRef,
    I: IntoIterator<Item = Point>,
    F: for<'a> FnMut(&'a S, Point) -> Box<dyn Iterator<Item = Point> + 'a>,
{
    let mut stack: Vec<Point> = seeds.into_iter().collect();
    let mut seen: HashSet<Point> = stack.iter().copied().collect();

    while let Some(p) = stack.pop() {
        for q in nbrs(sieve, p) {
            if seen.insert(q) {
                stack.push(q);
            }
        }
    }
    let mut out: Vec<Point> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

/// Transitive closure using `cone_points()` (no payload clones).
#[inline]
pub fn closure_ref<S, I>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point> + SieveRef,
    I: IntoIterator<Item = Point>,
{
    dfs_points(sieve, seeds, |s, p| Box::new(SieveRef::cone_points(s, p)))
}

/// Transitive star using `support_points()` (no payload clones).
#[inline]
pub fn star_ref<S, I>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point> + SieveRef,
    I: IntoIterator<Item = Point>,
{
    dfs_points(sieve, seeds, |s, p| {
        Box::new(SieveRef::support_points(s, p))
    })
}

/// BFS depth map using `cone_points()` (no payload clones).
pub fn depth_map_ref<S>(sieve: &S, seed: Point) -> Vec<(Point, u32)>
where
    S: Sieve<Point = Point> + SieveRef,
{
    let mut depths = Vec::new();
    let mut seen = HashSet::new();
    let mut q = VecDeque::from([(seed, 0)]);

    while let Some((p, d)) = q.pop_front() {
        if seen.insert(p) {
            depths.push((p, d));
            for q_pt in SieveRef::cone_points(sieve, p) {
                q.push_back((q_pt, d + 1));
            }
        }
    }
    depths.sort_by_key(|&(p, _)| p);
    depths
}

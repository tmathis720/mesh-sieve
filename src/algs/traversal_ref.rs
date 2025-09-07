//! Traversal helpers over Sieve topologies (clone-free for `SieveRef`).
//!
//! ## Determinism
//! - The unordered helpers (`*_ref`) return a **sorted** set of visited points,
//!   so the **output** is deterministic given the Sieve contents, even if internal visitation
//!   depends on hash iteration order.
//! - The ordered builder (`OrderedTraversalBuilder`) emits points in **chart order**
//!   (height-major then point order via `compute_strata`), so both **process** and **output**
//!   are deterministic.
//!
//! ## Complexity (unordered)
//! - DFS/BFS visit each point at most once: O(V + E) time, O(V) memory.
//!
//! ## Complexity (ordered)
//! - One `compute_strata` precomputation (cached in the Sieve) plus O(V + E).

use super::traversal::{Dir, Strategy};
use crate::mesh_error::MeshSieveError;
use crate::topology::bounds::PointLike;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::SieveRef;
use crate::topology::sieve::strata::compute_strata;
use std::collections::{HashSet, VecDeque};

pub type Point = PointId;

/// Generic DFS over point-only neighbors.
fn dfs_points<S, I, F>(sieve: &S, seeds: I, mut nbrs: F) -> Vec<Point>
where
    S: Sieve<Point = Point> + SieveRef,
    I: IntoIterator<Item = Point>,
    F: for<'a> FnMut(&'a S, Point) -> Box<dyn Iterator<Item = Point> + 'a>,
{
    let seed_vec: Vec<Point> = seeds.into_iter().collect();
    let mut seen: HashSet<Point> = HashSet::with_capacity(seed_vec.len().saturating_mul(2));
    seen.extend(seed_vec.iter().copied());
    let mut stack: Vec<Point> = seed_vec;

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
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** O(V + E) time, O(V) memory.
/// - **Determinism:** output sorted ascending.
#[inline]
pub fn closure_ref<S, I>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point> + SieveRef,
    I: IntoIterator<Item = Point>,
{
    dfs_points(sieve, seeds, |s, p| Box::new(SieveRef::cone_points(s, p)))
}

/// Transitive star using `support_points()` (no payload clones).
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** O(V + E) time, O(V) memory.
/// - **Determinism:** output sorted ascending.
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
///
/// - **Complexity:** O(V + E) time, O(V) memory.
/// - **Determinism:** output sorted by point id.
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

/// Builder for deterministic traversals in chart order without cloning payloads.
///
/// - **Determinism:** process and output follow chart order.
/// - **Complexity:** one `compute_strata` plus O(V + E).
pub struct OrderedTraversalBuilder<'a, S: Sieve + SieveRef> {
    sieve: &'a mut S,
    seeds: Vec<S::Point>,
    dir: Dir,
    strat: Strategy,
}

impl<'a, S> OrderedTraversalBuilder<'a, S>
where
    S: Sieve + SieveRef,
    S::Point: Copy + Ord,
{
    /// Create a new ordered builder over `sieve`.
    pub fn new(sieve: &'a mut S) -> Self {
        Self {
            sieve,
            seeds: Vec::new(),
            dir: Dir::Down,
            strat: Strategy::DFS,
        }
    }
    /// Seed starting points.
    pub fn seeds<I: IntoIterator<Item = S::Point>>(mut self, it: I) -> Self {
        self.seeds = it.into_iter().collect();
        self
    }
    /// Traversal direction.
    pub fn dir(mut self, d: Dir) -> Self {
        self.dir = d;
        self
    }
    /// Depth-first search.
    pub fn dfs(mut self) -> Self {
        self.strat = Strategy::DFS;
        self
    }
    /// Breadth-first search.
    pub fn bfs(mut self) -> Self {
        self.strat = Strategy::BFS;
        self
    }

    /// Run traversal emitting points in chart order.
    pub fn run(self) -> Result<Vec<S::Point>, MeshSieveError> {
        match self.strat {
            Strategy::DFS => self.run_dfs_ordered(),
            Strategy::BFS => self.run_bfs_ordered(),
        }
    }

    fn run_dfs_ordered(self) -> Result<Vec<S::Point>, MeshSieveError> {
        let Self {
            sieve, seeds, dir, ..
        } = self;
        let strata = compute_strata(&*sieve)?;
        let chart = strata.chart_points;
        let index = strata.chart_index;
        let n = chart.len();
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        stack.reserve(seeds.len().saturating_mul(2));
        for p in seeds {
            if let Some(i) = index.get(&p).copied() {
                if !seen[i] {
                    seen[i] = true;
                    stack.push(i);
                }
            }
        }

        while let Some(i) = stack.pop() {
            let p = chart[i];
            let mut nbrs: Vec<usize> = match dir {
                Dir::Down => SieveRef::cone_points(&*sieve, p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Up => SieveRef::support_points(&*sieve, p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Both => SieveRef::cone_points(&*sieve, p)
                    .chain(SieveRef::support_points(&*sieve, p))
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
            };
            nbrs.sort_unstable();
            nbrs.dedup();
            for j in nbrs.into_iter().rev() {
                if !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }

        let mut out = Vec::with_capacity(seen.iter().filter(|&&b| b).count());
        for (i, &flag) in seen.iter().enumerate() {
            if flag {
                out.push(chart[i]);
            }
        }
        Ok(out)
    }

    fn run_bfs_ordered(self) -> Result<Vec<S::Point>, MeshSieveError> {
        let Self {
            sieve, seeds, dir, ..
        } = self;
        let strata = compute_strata(&*sieve)?;
        let chart = strata.chart_points;
        let index = strata.chart_index;
        let n = chart.len();
        let mut seen = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        q.reserve(seeds.len().saturating_mul(2));
        for p in seeds {
            if let Some(i) = index.get(&p).copied() {
                if !seen[i] {
                    seen[i] = true;
                    q.push_back(i);
                }
            }
        }

        while let Some(i) = q.pop_front() {
            let p = chart[i];
            let mut nbrs: Vec<usize> = match dir {
                Dir::Down => SieveRef::cone_points(&*sieve, p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Up => SieveRef::support_points(&*sieve, p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Both => SieveRef::cone_points(&*sieve, p)
                    .chain(SieveRef::support_points(&*sieve, p))
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
            };
            nbrs.sort_unstable();
            nbrs.dedup();
            for j in nbrs {
                if !seen[j] {
                    seen[j] = true;
                    q.push_back(j);
                }
            }
        }

        let mut out = Vec::with_capacity(seen.iter().filter(|&&b| b).count());
        for (i, &flag) in seen.iter().enumerate() {
            if flag {
                out.push(chart[i]);
            }
        }
        Ok(out)
    }
}

/// Deterministic transitive closure without payload clones.
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** one `compute_strata` plus O(V + E).
/// - **Determinism:** process and output follow chart order.
pub fn closure_ordered_ref<I, S>(sieve: &mut S, seeds: I) -> Result<Vec<S::Point>, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: PointLike,
    I: IntoIterator<Item = S::Point>,
{
    OrderedTraversalBuilder::new(sieve)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds)
        .run()
}

/// Deterministic transitive star without payload clones.
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** one `compute_strata` plus O(V + E).
/// - **Determinism:** process and output follow chart order.
pub fn star_ordered_ref<I, S>(sieve: &mut S, seeds: I) -> Result<Vec<S::Point>, MeshSieveError>
where
    S: Sieve + SieveRef,
    S::Point: PointLike,
    I: IntoIterator<Item = S::Point>,
{
    OrderedTraversalBuilder::new(sieve)
        .dir(Dir::Up)
        .dfs()
        .seeds(seeds)
        .run()
}

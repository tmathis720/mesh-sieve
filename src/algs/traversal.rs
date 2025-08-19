//! DFS/BFS traversal helpers for Sieve topologies.

use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::strata::compute_strata;
use std::collections::{HashSet, VecDeque};

pub type Point = PointId;

#[derive(Clone, Copy, Debug)]
pub enum Dir {
    Down,
    Up,
    Both,
}

#[derive(Clone, Copy, Debug)]
pub enum Strategy {
    DFS,
    BFS,
}

pub struct TraversalBuilder<'a, S: Sieve> {
    sieve: &'a S,
    seeds: Vec<S::Point>,
    dir: Dir,
    strat: Strategy,
    max_depth: Option<u32>,
    /// If returns true on a discovered point, traversal stops early.
    early_stop: Option<&'a dyn Fn(S::Point) -> bool>,
}

impl<'a, S: Sieve> TraversalBuilder<'a, S>
where
    S::Point: Ord + Copy,
{
    pub fn new(sieve: &'a S) -> Self {
        Self {
            sieve,
            seeds: Vec::new(),
            dir: Dir::Down,
            strat: Strategy::DFS,
            max_depth: None,
            early_stop: None,
        }
    }
    pub fn seeds<I: IntoIterator<Item = S::Point>>(mut self, it: I) -> Self {
        self.seeds = it.into_iter().collect();
        self
    }
    pub fn dir(mut self, d: Dir) -> Self {
        self.dir = d;
        self
    }
    pub fn dfs(mut self) -> Self {
        self.strat = Strategy::DFS;
        self
    }
    pub fn bfs(mut self) -> Self {
        self.strat = Strategy::BFS;
        self
    }
    pub fn max_depth(mut self, d: Option<u32>) -> Self {
        self.max_depth = d;
        self
    }
    pub fn early_stop(mut self, f: &'a dyn Fn(S::Point) -> bool) -> Self {
        self.early_stop = Some(f);
        self
    }

    pub fn run(self) -> Vec<S::Point> {
        match self.strat {
            Strategy::DFS => self.run_dfs(),
            Strategy::BFS => self.run_bfs(),
        }
    }

    fn run_dfs(self) -> Vec<S::Point> {
        let TraversalBuilder {
            sieve,
            seeds,
            dir,
            strat: _,
            max_depth,
            early_stop,
        } = self;
        let mut seen: HashSet<S::Point> = seeds.iter().copied().collect();
        let mut stack: Vec<(S::Point, u32)> = seeds.into_iter().map(|p| (p, 0)).collect();

        while let Some((p, d)) = stack.pop() {
            if let Some(f) = early_stop {
                if f(p) {
                    break;
                }
            }
            if max_depth.map_or(false, |md| d >= md) {
                continue;
            }
            for q in step_neighbors(sieve, dir, p) {
                if seen.insert(q) {
                    stack.push((q, d + 1));
                }
            }
        }
        let mut out: Vec<_> = seen.into_iter().collect();
        out.sort_unstable();
        out
    }

    fn run_bfs(self) -> Vec<S::Point> {
        let TraversalBuilder {
            sieve,
            seeds,
            dir,
            strat: _,
            max_depth,
            early_stop,
        } = self;
        let mut seen: HashSet<S::Point> = seeds.iter().copied().collect();
        let mut q: VecDeque<(S::Point, u32)> = seeds.into_iter().map(|p| (p, 0)).collect();

        while let Some((p, d)) = q.pop_front() {
            if let Some(f) = early_stop {
                if f(p) {
                    break;
                }
            }
            if max_depth.map_or(false, |md| d >= md) {
                continue;
            }
            for qn in step_neighbors(sieve, dir, p) {
                if seen.insert(qn) {
                    q.push_back((qn, d + 1));
                }
            }
        }
        let mut out: Vec<_> = seen.into_iter().collect();
        out.sort_unstable();
        out
    }
}

fn step_neighbors<'a, S: Sieve>(
    sieve: &'a S,
    dir: Dir,
    p: S::Point,
) -> Box<dyn Iterator<Item = S::Point> + 'a> {
    match dir {
        Dir::Down => Box::new(sieve.cone_points(p)),
        Dir::Up => Box::new(sieve.support_points(p)),
        Dir::Both => Box::new(sieve.cone_points(p).chain(sieve.support_points(p))),
    }
}

// --- old helpers, now using the builder (kept for compatibility) ---

/// Complete transitive closure following `cone` arrows.
pub fn closure<I, S>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point>,
    I: IntoIterator<Item = Point>,
{
    TraversalBuilder::new(sieve)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds)
        .run()
}

/// Complete transitive star following `support` arrows.
pub fn star<I, S>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point>,
    I: IntoIterator<Item = Point>,
{
    TraversalBuilder::new(sieve)
        .dir(Dir::Up)
        .dfs()
        .seeds(seeds)
        .run()
}

/// Computes the link of a point (definition unchanged).
pub fn link<S: Sieve<Point = Point>>(sieve: &S, p: Point) -> Vec<Point> {
    let mut cl = closure(sieve, [p]);
    let mut st = star(sieve, [p]);
    cl.sort_unstable();
    st.sort_unstable();
    use std::collections::HashSet;
    let cone: HashSet<_> = sieve.cone_points(p).collect();
    let sup: HashSet<_> = sieve.support_points(p).collect();
    cl.retain(|x| st.binary_search(x).is_ok() && *x != p && !cone.contains(x) && !sup.contains(x));
    cl
}

/// Optional BFS distance map – used by coarsening / agglomeration.
pub fn depth_map<S: Sieve<Point = Point>>(sieve: &S, seed: Point) -> Vec<(Point, u32)> {
    let out = TraversalBuilder::new(sieve)
        .dir(Dir::Down)
        .bfs()
        .seeds([seed])
        .run();
    // reconstruct depths with a second BFS (cheap) to keep signature unchanged
    use std::collections::{HashMap, VecDeque};
    let mut depth = HashMap::new();
    let mut q = VecDeque::from([(seed, 0)]);
    while let Some((p, d)) = q.pop_front() {
        if depth.insert(p, d).is_none() {
            for qn in sieve.cone_points(p) {
                q.push_back((qn, d + 1));
            }
        }
    }
    let mut v: Vec<_> = out
        .into_iter()
        .map(|p| (p, *depth.get(&p).unwrap_or(&0)))
        .collect();
    v.sort_by_key(|&(p, _)| p);
    v
}

// --- ordered traversals using strata chart ---

/// Deterministic transitive closure following `cone` arrows, emitting points in
/// chart order (height-major, then point order).
pub fn closure_ordered<I, S>(sieve: &mut S, seeds: I) -> Result<Vec<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    I: IntoIterator<Item = S::Point>,
{
    let cache = compute_strata(sieve)?;
    let chart = cache.chart_points;
    let index_map = cache.chart_index;
    let n = chart.len();
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();

    for p in seeds {
        if let Some(i) = index_map.get(&p).copied() {
            if !seen[i] {
                seen[i] = true;
                stack.push(i);
            }
        }
    }

    while let Some(i) = stack.pop() {
        let p = chart[i];
        let mut nbrs: Vec<usize> = Vec::new();
        for q in sieve.cone_points(p) {
            if let Some(j) = index_map.get(&q).copied() {
                nbrs.push(j);
            }
        }
        nbrs.sort_unstable();
        nbrs.dedup();
        for j in nbrs.into_iter().rev() {
            if !seen[j] {
                seen[j] = true;
                stack.push(j);
            }
        }
    }

    let mut out = Vec::new();
    for (i, flag) in seen.iter().enumerate() {
        if *flag {
            out.push(chart[i]);
        }
    }
    Ok(out)
}

/// Deterministic transitive star following `support` arrows, emitting points in
/// chart order (height-major, then point order).
pub fn star_ordered<I, S>(sieve: &mut S, seeds: I) -> Result<Vec<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    I: IntoIterator<Item = S::Point>,
{
    let cache = compute_strata(sieve)?;
    let chart = cache.chart_points;
    let index_map = cache.chart_index;
    let n = chart.len();
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();

    for p in seeds {
        if let Some(i) = index_map.get(&p).copied() {
            if !seen[i] {
                seen[i] = true;
                stack.push(i);
            }
        }
    }

    while let Some(i) = stack.pop() {
        let p = chart[i];
        let mut nbrs: Vec<usize> = Vec::new();
        for q in sieve.support_points(p) {
            if let Some(j) = index_map.get(&q).copied() {
                nbrs.push(j);
            }
        }
        nbrs.sort_unstable();
        nbrs.dedup();
        for j in nbrs.into_iter().rev() {
            if !seen[j] {
                seen[j] = true;
                stack.push(j);
            }
        }
    }

    let mut out = Vec::new();
    for (i, flag) in seen.iter().enumerate() {
        if *flag {
            out.push(chart[i]);
        }
    }
    Ok(out)
}

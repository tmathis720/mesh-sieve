//! Traversal helpers over Sieve topologies.
//!
//! ## Determinism
//! - The unordered builder (`TraversalBuilder`) returns a **sorted** set of visited points,
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

use crate::algs::traversal_core::{self as core, Dir as CoreDir, Neigh, Strategy as CoreStrategy};
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{Overlap, local};
use crate::topology::bounds::PointLike;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::strata::compute_strata;
use std::collections::{HashMap, HashSet, VecDeque};

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

struct SieveNeigh<'a, S: Sieve> {
    sieve: &'a S,
}

impl<'a, S: Sieve> Neigh<'a, S::Point> for SieveNeigh<'a, S>
where
    S::Point: Copy,
{
    type Iter = Box<dyn Iterator<Item = S::Point> + 'a>;
    fn down(&'a self, p: S::Point) -> Self::Iter {
        Box::new(self.sieve.cone_points(p))
    }
    fn up(&'a self, p: S::Point) -> Self::Iter {
        Box::new(self.sieve.support_points(p))
    }
}

fn map_dir(d: Dir) -> CoreDir {
    match d {
        Dir::Down => CoreDir::Down,
        Dir::Up => CoreDir::Up,
        Dir::Both => CoreDir::Both,
    }
}

/// Builder for unordered traversals over a [`Sieve`].
///
/// - **Determinism:** returns the visited set sorted ascending.
/// - **Complexity:** O(V + E) time and O(V) memory.
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
    S::Point: Ord + Copy + std::hash::Hash + Eq,
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

    /// Execute the traversal.
    ///
    /// - **Determinism:** output sorted ascending.
    /// - **Complexity:** O(V + E) time / O(V) memory.
    pub fn run(self) -> Vec<S::Point> {
        match self.strat {
            Strategy::DFS => self.run_dfs(),
            Strategy::BFS => self.run_bfs(),
        }
    }

    fn run_dfs(self) -> Vec<S::Point> {
        let opts = core::Opts {
            dir: map_dir(self.dir),
            strat: CoreStrategy::DFS,
            max_depth: self.max_depth,
            deterministic: true,
            early_stop: self.early_stop,
        };
        core::traverse(&SieveNeigh { sieve: self.sieve }, self.seeds, opts)
    }

    fn run_bfs(self) -> Vec<S::Point> {
        let opts = core::Opts {
            dir: map_dir(self.dir),
            strat: CoreStrategy::BFS,
            max_depth: self.max_depth,
            deterministic: true,
            early_stop: self.early_stop,
        };
        core::traverse(&SieveNeigh { sieve: self.sieve }, self.seeds, opts)
    }
}

/// Builder for deterministic traversals in chart order.
///
/// Borrowing `&mut S` allows reusing the cached strata (`compute_strata`) between
/// traversals.
pub struct OrderedTraversalBuilder<'a, S: Sieve> {
    sieve: &'a mut S,
    seeds: Vec<S::Point>,
    dir: Dir,
    strat: Strategy,
}

impl<'a, S> OrderedTraversalBuilder<'a, S>
where
    S: Sieve,
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
    /// Seed starting points for traversal.
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
            if let Some(i) = index.get(&p).copied()
                && !seen[i]
            {
                seen[i] = true;
                stack.push(i);
            }
        }

        while let Some(i) = stack.pop() {
            let p = chart[i];
            let mut nbrs: Vec<usize> = match dir {
                Dir::Down => sieve
                    .cone_points(p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Up => sieve
                    .support_points(p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Both => sieve
                    .cone_points(p)
                    .chain(sieve.support_points(p))
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
            if let Some(i) = index.get(&p).copied()
                && !seen[i]
            {
                seen[i] = true;
                q.push_back(i);
            }
        }

        while let Some(i) = q.pop_front() {
            let p = chart[i];
            let mut nbrs: Vec<usize> = match dir {
                Dir::Down => sieve
                    .cone_points(p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Up => sieve
                    .support_points(p)
                    .filter_map(|q| index.get(&q).copied())
                    .collect(),
                Dir::Both => sieve
                    .cone_points(p)
                    .chain(sieve.support_points(p))
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

// --- old helpers, now using the builder (kept for compatibility) ---

/// Complete transitive closure following `cone` arrows.
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** O(V + E) time, O(V) memory.
/// - **Determinism:** output sorted ascending.
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

/// Alias for local closure without communication.
pub fn closure_local<I, S>(sieve: &S, seeds: I) -> Vec<Point>
where
    S: Sieve<Point = Point>,
    I: IntoIterator<Item = Point>,
{
    closure(sieve, seeds)
}

/// Complete transitive star following `support` arrows.
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** O(V + E) time, O(V) memory.
/// - **Determinism:** output sorted ascending.
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

/// Link of `p`.
///
/// `link(p) = (closure(p) ∩ star(p)) \ {p} \ cone(p) \ support(p)`
///
/// - Excludes immediate `cone(p)` and `support(p)` to match the standard topological link.
/// - **Complexity:** O(V + E) due to closure/star; output sorted for determinism.
pub fn link<S: Sieve<Point = Point>>(sieve: &S, p: Point) -> Vec<Point> {
    let mut cl = closure(sieve, [p]);
    let mut st = star(sieve, [p]);
    cl.sort_unstable();
    st.sort_unstable();
    use std::collections::HashSet;
    let cone: HashSet<_> = sieve.cone_points(p).collect();
    let sup: HashSet<_> = sieve.support_points(p).collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < cl.len() && j < st.len() {
        match cl[i].cmp(&st[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let x = cl[i];
                if x != p && !cone.contains(&x) && !sup.contains(&x) {
                    out.push(x);
                }
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Optional BFS distance map – used by coarsening / agglomeration.
///
/// - **Complexity:** O(V + E) time, O(V) memory.
/// - **Determinism:** output sorted by point id.
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

/// Completion type for `closure_completed`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CompletionKind {
    Cone,
    Support,
    Both,
}

/// When to perform completion.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CompletionTiming {
    Pre,
    OnDemand,
}

/// Policy controlling distributed closure completion.
#[derive(Copy, Clone, Debug)]
pub struct CompletionPolicy {
    pub kind: CompletionKind,
    pub timing: CompletionTiming,
    pub batch: usize,
    pub depth_limit: Option<u32>,
}

impl CompletionPolicy {
    pub fn cone_ondemand() -> Self {
        Self {
            kind: CompletionKind::Cone,
            timing: CompletionTiming::OnDemand,
            batch: 128,
            depth_limit: None,
        }
    }
    pub fn cone_pre(depth: Option<u32>) -> Self {
        Self {
            kind: CompletionKind::Cone,
            timing: CompletionTiming::Pre,
            batch: 0,
            depth_limit: depth,
        }
    }
}

/// Completed transitive closure on a partitioned mesh. Fetches missing
/// cones/supports for remote frontier points according to `policy`.
pub fn closure_completed<S, C, I>(
    sieve: &mut S,
    seeds: I,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
    policy: CompletionPolicy,
) -> Vec<Point>
where
    S: Sieve<Point = Point> + InvalidateCache,
    S::Payload: Default + Clone,
    I: IntoIterator<Item = Point>,
    C: crate::algs::communicator::Communicator + Sync,
{
    use crate::algs::completion::closure_fetch::{ReqKind, fetch_adjacency};

    const TAG: u16 = 0xA100;

    // collect the incoming iterator so we can iterate it multiple times
    let seed_vec: Vec<Point> = seeds.into_iter().collect();

    let fuse = |s: &mut S, adj: &HashMap<Point, Vec<Point>>| {
        for (&src, dsts) in adj {
            for &dst in dsts {
                s.add_arrow(src, dst, Default::default());
            }
        }
        s.invalidate_cache();
    };

    if matches!(policy.timing, CompletionTiming::Pre)
        && matches!(policy.kind, CompletionKind::Cone | CompletionKind::Both)
    {
        let mut q: VecDeque<(Point, u32)> = seed_vec.iter().copied().map(|p| (p, 0)).collect();
        let mut seen = HashSet::new();
        let mut by_owner: HashMap<usize, Vec<Point>> = HashMap::new();
        while let Some((p, d)) = q.pop_front() {
            if !seen.insert(p) {
                continue;
            }
            if policy.depth_limit.is_none_or(|limit| d < limit) {
                for (qpt, _) in sieve.cone(p) {
                    q.push_back((qpt, d + 1));
                }
            }
            if let Some(owner) = overlap.cone(local(p)).find_map(|(_, r)| Some(r.rank))
                && owner != my_rank
            {
                by_owner.entry(owner).or_default().push(p);
            }
        }
        if !by_owner.is_empty()
            && let Ok(adj) = fetch_adjacency(&by_owner, ReqKind::Cone, comm, TAG)
        {
            fuse(sieve, &adj);
        }
    }

    let mut stack: Vec<Point> = seed_vec.clone();
    let mut seen: HashSet<Point> = stack.iter().copied().collect();
    let mut batch: HashMap<usize, Vec<Point>> = HashMap::new();

    while let Some(p) = stack.pop() {
        let mut advanced = false;
        for (qpt, _) in sieve.cone(p) {
            if seen.insert(qpt) {
                stack.push(qpt);
                advanced = true;
            }
        }

        if !advanced
            && matches!(policy.kind, CompletionKind::Cone | CompletionKind::Both)
            && let Some(owner) = overlap.cone(local(p)).find_map(|(_, r)| Some(r.rank))
            && owner != my_rank
        {
            let e = batch.entry(owner).or_default();
            if !e.contains(&p) {
                e.push(p);
            }
        }

        if policy.batch > 0 {
            let total: usize = batch.values().map(|v| v.len()).sum();
            if total >= policy.batch {
                if let Ok(adj) = fetch_adjacency(&batch, ReqKind::Cone, comm, TAG) {
                    fuse(sieve, &adj);
                }
                batch.clear();
            }
        }
    }

    if !batch.is_empty()
        && let Ok(adj) = fetch_adjacency(&batch, ReqKind::Cone, comm, TAG)
    {
        fuse(sieve, &adj);
    }

    closure_local(sieve, seen)
}

// --- ordered traversals using strata chart ---

/// Deterministic transitive closure following `cone` arrows.
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** one `compute_strata` plus O(V + E).
/// - **Determinism:** process and output follow chart order.
pub fn closure_ordered<I, S>(sieve: &mut S, seeds: I) -> Result<Vec<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: PointLike,
    I: IntoIterator<Item = S::Point>,
{
    OrderedTraversalBuilder::new(sieve)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds)
        .run()
}

/// Deterministic transitive star following `support` arrows.
///
/// - **Preconditions:** `seeds` exist in `sieve`.
/// - **Complexity:** one `compute_strata` plus O(V + E).
/// - **Determinism:** process and output follow chart order.
pub fn star_ordered<I, S>(sieve: &mut S, seeds: I) -> Result<Vec<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: PointLike,
    I: IntoIterator<Item = S::Point>,
{
    OrderedTraversalBuilder::new(sieve)
        .dir(Dir::Up)
        .dfs()
        .seeds(seeds)
        .run()
}

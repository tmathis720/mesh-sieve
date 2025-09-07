//! Build a CSR (compressed-sparse-row) **dual graph** of a mesh.
//!
//! Each input cell becomes a vertex. Two cells are connected if they share
//! a boundary entity. The boundary policy is controlled via
//! [`AdjacencyOpts`](crate::algs::lattice::AdjacencyOpts) and defaults to
//! **faces only** (finite-volume style).
//!
//! Non-manifold boundaries are handled correctly: all cells incident to the
//! same boundary entity form a full clique. The output is deterministic:
//! CSR vertex `i` corresponds to the `i`‑th cell in the input order and every
//! neighbour list is sorted and deduplicated.
//!
//! ```rust
//! use mesh_sieve::algs::dual_graph::{build_dual, build_dual_with_opts, DualGraphOpts};
//! use mesh_sieve::algs::lattice::AdjacencyOpts;
//! use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
//! use mesh_sieve::topology::point::PointId;
//! use mesh_sieve::topology::sieve::Sieve;
//!
//! let mut s = InMemorySieve::<PointId, ()>::new();
//! let v = |i| PointId::new(i).unwrap();
//! let (c0, c1, f) = (v(1), v(2), v(10));
//! s.add_arrow(c0, f, ());
//! s.add_arrow(c1, f, ());
//!
//! // Default: faces only, unit weights
//! let _dg = build_dual(&s, vec![c0, c1]);
//!
//! // Custom boundary depth and weights
//! fn w(p: PointId) -> i32 { p.get() as i32 }
//! let opts = DualGraphOpts { boundary: AdjacencyOpts { max_down_depth: Some(2), same_stratum_only: true }, symmetrize: true };
//! let _dg2 = build_dual_with_opts(&s, vec![c0, c1], opts, Some(w));
//! ```
//!
//! The graph is symmetrised by default and contains no self-loops.

use std::collections::HashMap;

use crate::algs::lattice::AdjacencyOpts;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

/// CSR triple representing the dual graph.
#[derive(Debug, Clone)]
pub struct DualGraph {
    pub xadj: Vec<usize>,
    pub adjncy: Vec<usize>,
    pub vwgt: Vec<i32>, // ParMETIS expects i32
}

/// Options controlling dual-graph construction.
#[derive(Clone, Copy, Debug)]
pub struct DualGraphOpts {
    /// Boundary policy used to connect cells. Default = faces only.
    pub boundary: AdjacencyOpts,
    /// Ensure `i<->j` is present in both adjacency lists (default: true).
    pub symmetrize: bool,
}

impl Default for DualGraphOpts {
    fn default() -> Self {
        Self { boundary: AdjacencyOpts::default(), symmetrize: true }
    }
}

/// Optional weight callback. If `None`, all weights are set to `1`.
pub type WeightFn = Option<fn(PointId) -> i32>;

/// Build a dual graph with explicit options and optional weights.
pub fn build_dual_with_opts<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
    opts: DualGraphOpts,
    w: WeightFn,
) -> DualGraph
where
    S: Sieve<Point = PointId>,
{
    build_dual_inner(sieve, cells, opts, w).0
}

/// Same as [`build_dual_with_opts`] but also returns the cell order used for CSR.
pub fn build_dual_with_order_and_opts<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
    opts: DualGraphOpts,
    w: WeightFn,
) -> (DualGraph, Vec<PointId>)
where
    S: Sieve<Point = PointId>,
{
    build_dual_inner(sieve, cells, opts, w)
}

/// Backward-compatible default: faces-only boundary and unit weights.
pub fn build_dual<S>(sieve: &S, cells: impl IntoIterator<Item = PointId>) -> DualGraph
where
    S: Sieve<Point = PointId>,
{
    build_dual_with_opts(sieve, cells, DualGraphOpts::default(), None)
}

/// Backward-compatible default returning the cell ordering.
pub fn build_dual_with_order<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
) -> (DualGraph, Vec<PointId>)
where
    S: Sieve<Point = PointId>,
{
    build_dual_with_order_and_opts(sieve, cells, DualGraphOpts::default(), None)
}

// ---------------------------------------------------------------------------
// Internal routine
// ---------------------------------------------------------------------------
fn build_dual_inner<S>(
    sieve: &S,
    cells_iter: impl IntoIterator<Item = PointId>,
    opts: DualGraphOpts,
    w: WeightFn,
) -> (DualGraph, Vec<PointId>)
where
    S: Sieve<Point = PointId>,
{
    // 0) Assign CSR indices deterministically from insertion order
    let cells: Vec<PointId> = cells_iter.into_iter().collect();
    let n = cells.len();

    if n == 0 {
        return (
            DualGraph { xadj: vec![0], adjncy: Vec::new(), vwgt: Vec::new() },
            cells,
        );
    }

    // 1) boundary entity -> incident cell indices
    use std::collections::hash_map::Entry;

    fn boundary_points<S2: Sieve<Point = PointId>>(
        s: &S2,
        p: PointId,
        max_down_depth: Option<u32>,
    ) -> Vec<PointId> {
        use std::collections::{HashSet, VecDeque};
        match max_down_depth {
            None => {
                let mut out: Vec<PointId> = s.cone_points(p).collect();
                let mut seen: HashSet<PointId> = out.iter().copied().collect();
                let mut q: VecDeque<(PointId, u32)> = out.iter().copied().map(|x| (x, 1)).collect();
                while let Some((r, _d)) = q.pop_front() {
                    for qn in s.cone_points(r) {
                        if seen.insert(qn) {
                            out.push(qn);
                            q.push_back((qn, 0));
                        }
                    }
                }
                out.sort_unstable();
                out.dedup();
                out
            }
            Some(k) if k == 0 => Vec::new(),
            Some(k) => {
                let mut out = Vec::new();
                let mut seen = HashSet::new();
                let mut q = VecDeque::from_iter(s.cone_points(p).map(|x| (x, 1)));
                while let Some((r, d)) = q.pop_front() {
                    if seen.insert(r) {
                        out.push(r);
                    }
                    if d < k {
                        for qn in s.cone_points(r) {
                            q.push_back((qn, d + 1));
                        }
                    }
                }
                out.sort_unstable();
                out.dedup();
                out
            }
        }
    }

    let mut incident: HashMap<PointId, Vec<usize>> = HashMap::new();
    incident.reserve(n * 4);
    for (ci, &cell) in cells.iter().enumerate() {
        let bnd = boundary_points(sieve, cell, opts.boundary.max_down_depth);
        for b in bnd {
            match incident.entry(b) {
                Entry::Occupied(mut e) => e.get_mut().push(ci),
                Entry::Vacant(e) => {
                    e.insert(vec![ci]);
                }
            }
        }
    }

    // 2) Clique incident cells on each boundary entity
    let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (_b, mut cells_on_b) in incident {
        if cells_on_b.len() < 2 {
            continue;
        }
        cells_on_b.sort_unstable();
        cells_on_b.dedup();
        for i in 0..cells_on_b.len() {
            let ci = cells_on_b[i];
            for &cj in &cells_on_b[(i + 1)..] {
                neigh[ci].push(cj);
                if opts.symmetrize {
                    neigh[cj].push(ci);
                }
            }
        }
    }

    // 3) Sort & dedup neighbor lists
    let mut total_edges = 0usize;
    for (i, list) in neigh.iter_mut().enumerate() {
        list.sort_unstable();
        list.dedup();
        if let Ok(pos) = list.binary_search(&i) {
            list.remove(pos);
        }
        total_edges += list.len();
    }

    // 4) Convert to CSR
    let mut xadj = Vec::with_capacity(n + 1);
    let mut adjncy = Vec::with_capacity(total_edges);
    xadj.push(0);
    for list in &neigh {
        adjncy.extend(list.iter().copied());
        xadj.push(adjncy.len());
    }

    // 5) Weights
    let vwgt = if let Some(f) = w {
        cells.iter().map(|&c| f(c)).collect()
    } else {
        vec![1; n]
    };

    (DualGraph { xadj, adjncy, vwgt }, cells)
}


//! Build adjacency graphs for mesh strata.
//!
//! Provides utilities for cell-to-cell adjacency (via shared faces/edges) and
//! vertex-to-vertex adjacency (via shared support entities such as edges).
//!
//! Determinism:
//! - You can choose the ordering of input points; neighbor lists are always
//!   sorted and deduplicated.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::Sieve;

use crate::algs::lattice::AdjacencyOpts;

/// CSR-style adjacency graph for a chosen point ordering.
#[derive(Debug, Clone)]
pub struct AdjacencyGraph {
    /// CSR offsets into `adjncy` for each vertex.
    pub xadj: Vec<usize>,
    /// CSR adjacency list (indices into `order`).
    pub adjncy: Vec<usize>,
    /// Point ordering that defines vertex indices.
    pub order: Vec<PointId>,
}

impl AdjacencyGraph {
    /// Return the neighbor index slice for vertex `i`.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.adjncy[self.xadj[i]..self.xadj[i + 1]]
    }
}

/// Ordering options for point lists.
#[derive(Clone, Copy, Debug)]
pub enum AdjacencyOrdering {
    /// Preserve the provided input order (deduplicated in a stable manner).
    Input,
    /// Sort points ascending by `PointId`.
    Sorted,
}

/// Options for building cell-to-cell adjacency graphs.
#[derive(Clone, Copy, Debug)]
pub struct CellAdjacencyOpts {
    /// Boundary policy (faces-only by default).
    pub boundary: AdjacencyOpts,
    /// Ordering of the cell list used in the graph.
    pub ordering: AdjacencyOrdering,
    /// Add symmetric neighbors in both directions.
    pub symmetrize: bool,
}

impl Default for CellAdjacencyOpts {
    fn default() -> Self {
        Self {
            boundary: AdjacencyOpts::default(),
            ordering: AdjacencyOrdering::Sorted,
            symmetrize: true,
        }
    }
}

/// Options for building vertex-to-vertex adjacency graphs.
#[derive(Clone, Copy, Debug)]
pub struct VertexAdjacencyOpts {
    /// Maximum depth of the upward (support) walk.
    /// - `Some(1)` =&gt; edges only (default)
    /// - `Some(2)` =&gt; edges+faces (2D), etc.
    /// - `Some(0)` =&gt; empty (no adjacency)
    /// - `None` =&gt; full upward closure
    pub max_up_depth: Option<u32>,
    /// Ordering of the vertex list used in the graph.
    pub ordering: AdjacencyOrdering,
    /// Add symmetric neighbors in both directions.
    pub symmetrize: bool,
}

impl Default for VertexAdjacencyOpts {
    fn default() -> Self {
        Self {
            max_up_depth: Some(1),
            ordering: AdjacencyOrdering::Sorted,
            symmetrize: true,
        }
    }
}

/// Build a cell-to-cell adjacency graph over all height-0 cells in the sieve.
pub fn build_cell_adjacency_graph<S>(
    sieve: &S,
    opts: CellAdjacencyOpts,
) -> Result<AdjacencyGraph, MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let strata = compute_strata(sieve)?;
    let cells = strata.strata.get(0).cloned().unwrap_or_default();
    Ok(build_cell_adjacency_graph_with_cells(sieve, cells, opts))
}

/// Build a cell-to-cell adjacency graph for a provided cell list.
pub fn build_cell_adjacency_graph_with_cells<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
    opts: CellAdjacencyOpts,
) -> AdjacencyGraph
where
    S: Sieve<Point = PointId>,
{
    let cells = order_points(cells, opts.ordering);
    build_shared_boundary_graph(
        &cells,
        |p| downward_boundary_points(sieve, p, opts.boundary.max_down_depth),
        opts.symmetrize,
    )
}

/// Build a vertex-to-vertex adjacency graph over all depth-0 vertices in the sieve.
pub fn build_vertex_adjacency_graph<S>(
    sieve: &S,
    opts: VertexAdjacencyOpts,
) -> Result<AdjacencyGraph, MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let strata = compute_strata(sieve)?;
    let vertices: Vec<PointId> = strata
        .chart_points
        .iter()
        .copied()
        .filter(|p| strata.depth.get(p).copied() == Some(0))
        .collect();
    let vertices = order_points(vertices, opts.ordering);
    Ok(build_vertex_adjacency_graph_with_vertices(
        sieve, vertices, opts,
    ))
}

/// Build a vertex-to-vertex adjacency graph for a provided vertex list.
pub fn build_vertex_adjacency_graph_with_vertices<S>(
    sieve: &S,
    vertices: impl IntoIterator<Item = PointId>,
    opts: VertexAdjacencyOpts,
) -> AdjacencyGraph
where
    S: Sieve<Point = PointId>,
{
    let vertices = order_points(vertices, opts.ordering);
    build_shared_boundary_graph(
        &vertices,
        |p| upward_boundary_points(sieve, p, opts.max_up_depth),
        opts.symmetrize,
    )
}

fn order_points<I>(points: I, ordering: AdjacencyOrdering) -> Vec<PointId>
where
    I: IntoIterator<Item = PointId>,
{
    let mut out: Vec<PointId> = points.into_iter().collect();
    match ordering {
        AdjacencyOrdering::Input => {
            let mut seen = HashSet::with_capacity(out.len());
            out.retain(|p| seen.insert(*p));
        }
        AdjacencyOrdering::Sorted => {
            out.sort_unstable();
            out.dedup();
        }
    }
    out
}

fn downward_boundary_points<S>(
    sieve: &S,
    p: PointId,
    max_down_depth: Option<u32>,
) -> Vec<PointId>
where
    S: Sieve<Point = PointId>,
{
    match max_down_depth {
        Some(0) => Vec::new(),
        Some(1) => {
            let mut out: Vec<PointId> = sieve.cone_points(p).collect();
            out.sort_unstable();
            out.dedup();
            out
        }
        None | Some(_) => {
            let limit = max_down_depth.unwrap_or(u32::MAX);
            let mut out = Vec::new();
            let mut seen: HashSet<PointId> = HashSet::new();
            let mut q: VecDeque<(PointId, u32)> = VecDeque::new();
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

fn upward_boundary_points<S>(
    sieve: &S,
    p: PointId,
    max_up_depth: Option<u32>,
) -> Vec<PointId>
where
    S: Sieve<Point = PointId>,
{
    match max_up_depth {
        Some(0) => Vec::new(),
        Some(1) => {
            let mut out: Vec<PointId> = sieve.support_points(p).collect();
            out.sort_unstable();
            out.dedup();
            out
        }
        None | Some(_) => {
            let limit = max_up_depth.unwrap_or(u32::MAX);
            let mut out = Vec::new();
            let mut seen: HashSet<PointId> = HashSet::new();
            let mut q: VecDeque<(PointId, u32)> = VecDeque::new();
            q.extend(sieve.support_points(p).map(|x| (x, 1)));
            while let Some((r, d)) = q.pop_front() {
                if seen.insert(r) {
                    out.push(r);
                    if d < limit {
                        for s in sieve.support_points(r) {
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

fn build_shared_boundary_graph(
    points: &[PointId],
    boundary: impl Fn(PointId) -> Vec<PointId>,
    symmetrize: bool,
) -> AdjacencyGraph {
    let n = points.len();
    if n == 0 {
        return AdjacencyGraph {
            xadj: vec![0],
            adjncy: Vec::new(),
            order: Vec::new(),
        };
    }

    let mut incident: HashMap<PointId, Vec<usize>> = HashMap::new();
    incident.reserve(n * 4);
    for (i, &p) in points.iter().enumerate() {
        for b in boundary(p) {
            incident.entry(b).or_default().push(i);
        }
    }

    let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (_b, mut verts_on_b) in incident {
        if verts_on_b.len() < 2 {
            continue;
        }
        verts_on_b.sort_unstable();
        verts_on_b.dedup();
        for i in 0..verts_on_b.len() {
            let vi = verts_on_b[i];
            for &vj in &verts_on_b[(i + 1)..] {
                neigh[vi].push(vj);
                if symmetrize {
                    neigh[vj].push(vi);
                }
            }
        }
    }

    let mut total_edges = 0usize;
    for (i, list) in neigh.iter_mut().enumerate() {
        list.sort_unstable();
        list.dedup();
        if let Ok(pos) = list.binary_search(&i) {
            list.remove(pos);
        }
        total_edges += list.len();
    }

    let mut xadj = Vec::with_capacity(n + 1);
    let mut adjncy = Vec::with_capacity(total_edges);
    xadj.push(0);
    for list in &neigh {
        adjncy.extend(list.iter().copied());
        xadj.push(adjncy.len());
    }

    AdjacencyGraph {
        xadj,
        adjncy,
        order: points.to_vec(),
    }
}

//! Mesh graph exports for adjacency (cell/vertex) in CSR or edge-list form.
//!
//! This module provides lightweight wrappers around the adjacency builders so
//! callers can obtain CSR adjacency with optional shared-boundary weights, or
//! undirected edge lists for cell adjacency.

use std::collections::HashMap;

use crate::algs::adjacency_graph::{
    AdjacencyOrdering, CellAdjacencyEdges, CellAdjacencyOpts, VertexAdjacencyOpts,
    build_cell_adjacency_edges, build_cell_adjacency_graph_with_cells,
    build_vertex_adjacency_graph_with_vertices,
};
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::topology::sieve::strata::compute_strata;

/// Weighting mode for adjacency graphs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdjacencyWeighting {
    /// No weights (unweighted adjacency).
    None,
    /// Weight edges by the number of shared boundary entities.
    SharedBoundaryCount,
}

/// CSR-style adjacency graph with optional weights.
#[derive(Debug, Clone)]
pub struct MeshGraph {
    /// CSR offsets into `adjncy` for each vertex.
    pub xadj: Vec<usize>,
    /// CSR adjacency list (indices into `order`).
    pub adjncy: Vec<usize>,
    /// Point ordering that defines vertex indices.
    pub order: Vec<PointId>,
    /// Optional per-edge weights aligned with `adjncy`.
    pub weights: Option<Vec<u32>>,
}

impl MeshGraph {
    /// Return the neighbor index slice for vertex `i`.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.adjncy[self.xadj[i]..self.xadj[i + 1]]
    }

    /// Return the neighbor weight slice for vertex `i`, if present.
    #[inline]
    pub fn neighbor_weights(&self, i: usize) -> Option<&[u32]> {
        self.weights
            .as_ref()
            .map(|w| &w[self.xadj[i]..self.xadj[i + 1]])
    }
}

/// Build a cell-to-cell adjacency graph over all height-0 cells.
pub fn cell_adjacency_graph<S>(
    sieve: &S,
    opts: CellAdjacencyOpts,
    weighting: AdjacencyWeighting,
) -> Result<MeshGraph, MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
    let strata = compute_strata(sieve)?;
    let cells = strata.strata.get(0).cloned().unwrap_or_default();
    Ok(cell_adjacency_graph_with_cells(
        sieve, cells, opts, weighting,
    ))
}

/// Build a cell-to-cell adjacency graph for a provided cell list.
pub fn cell_adjacency_graph_with_cells<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
    opts: CellAdjacencyOpts,
    weighting: AdjacencyWeighting,
) -> MeshGraph
where
    S: Sieve<Point = PointId>,
{
    let cells: Vec<PointId> = cells.into_iter().collect();
    if matches!(weighting, AdjacencyWeighting::None) {
        let graph = build_cell_adjacency_graph_with_cells(sieve, cells, opts);
        return MeshGraph {
            xadj: graph.xadj,
            adjncy: graph.adjncy,
            order: graph.order,
            weights: None,
        };
    }

    let cells = order_points(cells, opts.ordering);
    build_weighted_shared_boundary_graph(
        &cells,
        |p| downward_boundary_points(sieve, p, opts.boundary.max_down_depth),
        opts.symmetrize,
    )
}

/// Build an undirected edge list for a provided cell list.
pub fn cell_adjacency_edges_for_cells<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
    cell_dimension: u32,
    by: crate::algs::adjacency_graph::CellAdjacencyBy,
    ordering: AdjacencyOrdering,
) -> CellAdjacencyEdges
where
    S: Sieve<Point = PointId>,
{
    build_cell_adjacency_edges(sieve, cells, cell_dimension, by, ordering)
}

/// Build a vertex-to-vertex adjacency graph over all depth-0 vertices.
pub fn vertex_adjacency_graph<S>(
    sieve: &S,
    opts: VertexAdjacencyOpts,
    weighting: AdjacencyWeighting,
) -> Result<MeshGraph, MeshSieveError>
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
    Ok(vertex_adjacency_graph_with_vertices(
        sieve, vertices, opts, weighting,
    ))
}

/// Build a vertex-to-vertex adjacency graph for a provided vertex list.
pub fn vertex_adjacency_graph_with_vertices<S>(
    sieve: &S,
    vertices: impl IntoIterator<Item = PointId>,
    opts: VertexAdjacencyOpts,
    weighting: AdjacencyWeighting,
) -> MeshGraph
where
    S: Sieve<Point = PointId>,
{
    let vertices: Vec<PointId> = vertices.into_iter().collect();
    if matches!(weighting, AdjacencyWeighting::None) {
        let graph = build_vertex_adjacency_graph_with_vertices(sieve, vertices, opts);
        return MeshGraph {
            xadj: graph.xadj,
            adjncy: graph.adjncy,
            order: graph.order,
            weights: None,
        };
    }

    let vertices = order_points(vertices, opts.ordering);
    build_weighted_shared_boundary_graph(
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
            let mut seen = std::collections::HashSet::with_capacity(out.len());
            out.retain(|p| seen.insert(*p));
        }
        AdjacencyOrdering::Sorted => {
            out.sort_unstable();
            out.dedup();
        }
    }
    out
}

fn downward_boundary_points<S>(sieve: &S, p: PointId, max_down_depth: Option<u32>) -> Vec<PointId>
where
    S: Sieve<Point = PointId>,
{
    use std::collections::{HashSet, VecDeque};

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

fn upward_boundary_points<S>(sieve: &S, p: PointId, max_up_depth: Option<u32>) -> Vec<PointId>
where
    S: Sieve<Point = PointId>,
{
    use std::collections::{HashSet, VecDeque};

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

fn build_weighted_shared_boundary_graph(
    points: &[PointId],
    boundary: impl Fn(PointId) -> Vec<PointId>,
    symmetrize: bool,
) -> MeshGraph {
    let n = points.len();
    if n == 0 {
        return MeshGraph {
            xadj: vec![0],
            adjncy: Vec::new(),
            order: Vec::new(),
            weights: Some(Vec::new()),
        };
    }

    let mut incident: HashMap<PointId, Vec<usize>> = HashMap::new();
    incident.reserve(n * 4);
    for (i, &p) in points.iter().enumerate() {
        for b in boundary(p) {
            incident.entry(b).or_default().push(i);
        }
    }

    let mut neigh: Vec<HashMap<usize, u32>> = vec![HashMap::new(); n];
    for (_b, mut verts_on_b) in incident {
        if verts_on_b.len() < 2 {
            continue;
        }
        verts_on_b.sort_unstable();
        verts_on_b.dedup();
        for i in 0..verts_on_b.len() {
            let vi = verts_on_b[i];
            for &vj in &verts_on_b[(i + 1)..] {
                if vi == vj {
                    continue;
                }
                *neigh[vi].entry(vj).or_insert(0) += 1;
                if symmetrize {
                    *neigh[vj].entry(vi).or_insert(0) += 1;
                }
            }
        }
    }

    let mut xadj = Vec::with_capacity(n + 1);
    let mut adjncy = Vec::new();
    let mut weights = Vec::new();
    xadj.push(0);
    for (i, map) in neigh.into_iter().enumerate() {
        let mut neighbors: Vec<(usize, u32)> = map.into_iter().collect();
        neighbors.sort_by_key(|(idx, _)| *idx);
        if let Some(pos) = neighbors.iter().position(|(idx, _)| *idx == i) {
            neighbors.remove(pos);
        }
        adjncy.extend(neighbors.iter().map(|(idx, _)| *idx));
        weights.extend(neighbors.iter().map(|(_, w)| *w));
        xadj.push(adjncy.len());
    }

    MeshGraph {
        xadj,
        adjncy,
        order: points.to_vec(),
        weights: Some(weights),
    }
}

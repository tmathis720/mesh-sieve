//! Build a CSR (compressed-sparse-row) *dual graph* of a mesh.
//
// Each *cell* is a vertex; an undirected edge is added between any two
// cells that share at least one lower-dimensional entity (face / edge / vertex),
// exactly following Knepley & Karpeev, 2009 §3.
//
// Returned in ParMETIS-ready CSR triples:
//
// * `xadj[i] .. xadj[i+1]`   = neighbour list of cell *i*
// * `adjncy`                 = concatenated neighbour vertices
// * `vwgt[i]`                = (optional) vertex weight, default = 1
//
// The dual graph is **symmetrised** (i↔j appear in both lists) and
// **self-free** (no loops).

use std::collections::{HashMap, HashSet};

use crate::algs::traversal::closure;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

/// CSR triple
#[derive(Debug, Clone)]
pub struct DualGraph {
    pub xadj: Vec<usize>,
    pub adjncy: Vec<usize>,
    pub vwgt: Vec<i32>, // ParMETIS expects i32
}

/// Build dual graph. Cell indices are assigned by *insertion order*
/// of the `cells` iterator passed in.
///
/// If you want a specific ordering (e.g. global id array), call
/// `build_dual_with_order` variant below.
pub fn build_dual<S>(sieve: &S, cells: impl IntoIterator<Item = PointId>) -> DualGraph
where
    S: Sieve<Point = PointId>,
{
    build_dual_inner(sieve, cells).0
}

/// Same as `build_dual` but also returns `Vec<PointId>` mapping CSR vertex → cell id.
pub fn build_dual_with_order<S>(
    sieve: &S,
    cells: impl IntoIterator<Item = PointId>,
) -> (DualGraph, Vec<PointId>)
where
    S: Sieve<Point = PointId>,
{
    build_dual_inner(sieve, cells)
}

// ---------------------------------------------------------------------------
// internal routine
// ---------------------------------------------------------------------------
fn build_dual_inner<S>(
    sieve: &S,
    cells_iter: impl IntoIterator<Item = PointId>,
) -> (DualGraph, Vec<PointId>)
where
    S: Sieve<Point = PointId>,
{
    // 0. assign CSR vertex ids
    let cells: Vec<PointId> = cells_iter.into_iter().collect();
    let n = cells.len();
    let mut idx_of: HashMap<PointId, usize> = HashMap::with_capacity(n);
    for (i, &c) in cells.iter().enumerate() {
        idx_of.insert(c, i);
    }

    // 1. first-seen map: lower-dim “face” → cell-index
    let mut first_face_owner: HashMap<PointId, usize> = HashMap::new();

    // 2. adjacency list being built
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for (cell_idx, &cell) in cells.iter().enumerate() {
        // collect faces/edges/verts once
        let cone_faces: HashSet<PointId> = closure(sieve, [cell]).into_iter().collect();

        for face in cone_faces {
            if let Some(&other_cell_idx) = first_face_owner.get(&face) {
                // second time we see this face –> add undirected edge
                adj[cell_idx].insert(other_cell_idx);
                adj[other_cell_idx].insert(cell_idx);
            } else {
                // first owner
                first_face_owner.insert(face, cell_idx);
            }
        }
    }

    // 3. Convert HashSet adjacency → CSR vectors
    let mut xadj = Vec::with_capacity(n + 1);
    let mut adjncy = Vec::new();
    xadj.push(0);
    for nbrs in &adj {
        adjncy.extend(nbrs.iter().copied());
        xadj.push(adjncy.len());
    }

    // 4. Simple unit vertex weights
    let vwgt = vec![1; n];

    (DualGraph { xadj, adjncy, vwgt }, cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::sieve::InMemorySieve;

    // helper to build two triangles sharing an edge
    fn tiny_mesh() -> (InMemorySieve<PointId>, Vec<PointId>) {
        let v = |i| PointId::new(i);
        let t0 = v(10);
        let t1 = v(11);
        // vertices
        let (a, b, c, d) = (v(1), v(2), v(3), v(4));
        let mut s = InMemorySieve::<PointId, ()>::default();
        for p in [a, b, c] {
            s.add_arrow(t0, p, ());
            s.add_arrow(p, t0, ());
        }
        for p in [b, c, d] {
            s.add_arrow(t1, p, ());
            s.add_arrow(p, t1, ());
        }
        (s, vec![t0, t1])
    }

    #[test]
    fn dual_graph_two_cells() {
        let (mesh, cells) = tiny_mesh();
        let (dg, order) = build_dual_with_order(&mesh, cells.clone());

        // should be 2 vertices with a single undirected edge
        assert_eq!(dg.xadj, vec![0, 1, 2]);
        assert_eq!(dg.adjncy.len(), 2);
        // each vertex's neighbour list contains the other
        assert_eq!(dg.adjncy[dg.xadj[0]], 1);
        assert_eq!(dg.adjncy[dg.xadj[1]], 0);
        // CSR order mapping is identity here
        assert_eq!(order, cells);
    }
}

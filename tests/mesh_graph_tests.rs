use mesh_sieve::algs::adjacency_graph::{
    AdjacencyOrdering, CellAdjacencyBy, CellAdjacencyOpts, VertexAdjacencyOpts,
};
use mesh_sieve::mesh_graph::{
    AdjacencyWeighting, cell_adjacency_edges_for_cells, cell_adjacency_graph_with_cells,
    vertex_adjacency_graph_with_vertices,
};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn v(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn two_cell_two_face_mesh() -> (InMemorySieve<PointId, ()>, Vec<PointId>) {
    let (c0, c1) = (v(10), v(11));
    let (f0, f1) = (v(20), v(21));
    let mut s = InMemorySieve::<PointId, ()>::new();
    for &f in &[f0, f1] {
        s.add_arrow(c0, f, ());
        s.add_arrow(c1, f, ());
    }
    (s, vec![c0, c1])
}

#[test]
fn cell_adjacency_exports_csr_and_edges() {
    let (s, cells) = two_cell_two_face_mesh();
    let opts = CellAdjacencyOpts::default();
    let graph = cell_adjacency_graph_with_cells(&s, cells.clone(), opts, AdjacencyWeighting::None);
    assert_eq!(graph.order, vec![v(10), v(11)]);
    assert_eq!(graph.xadj, vec![0, 1, 2]);
    assert_eq!(graph.adjncy, vec![1, 0]);
    assert!(graph.weights.is_none());

    let edges = cell_adjacency_edges_for_cells(
        &s,
        cells,
        2,
        CellAdjacencyBy::Faces,
        AdjacencyOrdering::Sorted,
    );
    assert_eq!(edges.edges, vec![(v(10), v(11))]);
}

#[test]
fn cell_adjacency_exports_weighted_csr() {
    let (s, cells) = two_cell_two_face_mesh();
    let opts = CellAdjacencyOpts::default();
    let graph =
        cell_adjacency_graph_with_cells(&s, cells, opts, AdjacencyWeighting::SharedBoundaryCount);
    assert_eq!(graph.xadj, vec![0, 1, 2]);
    assert_eq!(graph.adjncy, vec![1, 0]);
    assert_eq!(graph.weights, Some(vec![2, 2]));
}

#[test]
fn vertex_adjacency_exports_weighted_csr() {
    let (v1, v2, v3) = (v(1), v(2), v(3));
    let (e1, e2) = (v(10), v(11));
    let mut s = InMemorySieve::<PointId, ()>::new();
    s.add_arrow(e1, v1, ());
    s.add_arrow(e1, v2, ());
    s.add_arrow(e2, v2, ());
    s.add_arrow(e2, v3, ());

    let opts = VertexAdjacencyOpts::default();
    let graph = vertex_adjacency_graph_with_vertices(
        &s,
        vec![v1, v2, v3],
        opts,
        AdjacencyWeighting::SharedBoundaryCount,
    );
    assert_eq!(graph.order, vec![v1, v2, v3]);
    assert_eq!(graph.xadj, vec![0, 1, 3, 4]);
    assert_eq!(graph.adjncy, vec![1, 0, 2, 1]);
    assert_eq!(graph.weights, Some(vec![1, 1, 1, 1]));
}

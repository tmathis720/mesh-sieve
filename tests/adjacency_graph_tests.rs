use mesh_sieve::algs::adjacency_graph::{
    AdjacencyOrdering, CellAdjacencyOpts, VertexAdjacencyOpts,
    build_cell_adjacency_graph_with_cells, build_vertex_adjacency_graph_with_vertices,
};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::Sieve;

fn v(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

fn mesh_face_vertex() -> (InMemorySieve<PointId, ()>, Vec<PointId>) {
    let (c0, c1, c2) = (v(10), v(11), v(12));
    let f_shared = v(20);
    let f0 = v(21);
    let f1 = v(22);
    let f1v = v(23);
    let f2v = v(24);
    let f2 = v(25);
    let v_shared = v(30);

    let mut s = InMemorySieve::<PointId, ()>::new();
    for &f in &[f_shared, f0] {
        s.add_arrow(c0, f, ());
    }
    for &f in &[f_shared, f1, f1v] {
        s.add_arrow(c1, f, ());
    }
    for &f in &[f2, f2v] {
        s.add_arrow(c2, f, ());
    }
    s.add_arrow(f1v, v_shared, ());
    s.add_arrow(f2v, v_shared, ());

    (s, vec![c0, c1, c2])
}

#[test]
fn cell_face_adjacency_matches_shared_faces() {
    let (s, cells) = mesh_face_vertex();
    let opts = CellAdjacencyOpts::default();
    let graph = build_cell_adjacency_graph_with_cells(&s, cells, opts);
    assert_eq!(graph.order, vec![v(10), v(11), v(12)]);
    assert_eq!(graph.xadj, vec![0, 1, 2, 2]);
    assert_eq!(graph.adjncy, vec![1, 0]);
}

#[test]
fn vertex_adjacency_from_shared_edges() {
    let (v1, v2, v3) = (v(1), v(2), v(3));
    let (e1, e2) = (v(10), v(11));
    let mut s = InMemorySieve::<PointId, ()>::new();
    s.add_arrow(e1, v1, ());
    s.add_arrow(e1, v2, ());
    s.add_arrow(e2, v2, ());
    s.add_arrow(e2, v3, ());

    let opts = VertexAdjacencyOpts::default();
    let graph = build_vertex_adjacency_graph_with_vertices(&s, vec![v1, v2, v3], opts);
    assert_eq!(graph.order, vec![v1, v2, v3]);
    assert_eq!(graph.xadj, vec![0, 1, 3, 4]);
    assert_eq!(graph.adjncy, vec![1, 0, 2, 1]);
}

#[test]
fn ordering_controls_vertex_indices() {
    let (s, cells) = mesh_face_vertex();
    let opts = CellAdjacencyOpts {
        ordering: AdjacencyOrdering::Input,
        ..CellAdjacencyOpts::default()
    };
    let custom_order = vec![v(12), v(10), v(11), v(12)];
    let graph = build_cell_adjacency_graph_with_cells(&s, custom_order, opts);
    assert_eq!(graph.order, vec![v(12), v(10), v(11)]);
    assert_eq!(graph.xadj, vec![0, 0, 1, 2]);
    assert_eq!(graph.adjncy, vec![2, 1]);
    assert_eq!(cells.len(), 3);
}

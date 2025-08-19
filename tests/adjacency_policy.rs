use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::algs::lattice::{adjacent_with, AdjacencyOpts};

#[test]
fn face_neighbors_only() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    let v = |i| PointId::new(i).unwrap();
    // two triangles sharing an edge: 1->(3,4,5), 2->(5,6,7); edges->verts
    s.add_arrow(v(1), v(3), ()); s.add_arrow(v(1), v(4), ()); s.add_arrow(v(1), v(5), ());
    s.add_arrow(v(2), v(5), ()); s.add_arrow(v(2), v(6), ()); s.add_arrow(v(2), v(7), ());
    s.add_arrow(v(5), v(8), ()); s.add_arrow(v(5), v(9), ());
    let neigh = adjacent_with(&s, v(1), AdjacencyOpts { max_down_depth: Some(1) });
    assert_eq!(neigh, vec![v(2)]); // share edge 5
}

#[test]
fn through_vertices_too() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    let v = |i| PointId::new(i).unwrap();
    // two cells sharing only a vertex
    s.add_arrow(v(1), v(3), ()); s.add_arrow(v(1), v(4), ());
    s.add_arrow(v(2), v(4), ()); s.add_arrow(v(2), v(5), ());
    let neigh = adjacent_with(&s, v(1), AdjacencyOpts { max_down_depth: Some(2) });
    assert_eq!(neigh, vec![v(2)]);
}

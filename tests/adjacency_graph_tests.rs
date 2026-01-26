use mesh_sieve::algs::adjacency_graph::{
    AdjacencyOrdering, CellAdjacencyBy, CellAdjacencyOpts, CellAdjacencyStratum,
    VertexAdjacencyOpts, build_cell_adjacency_edges_in_stratum,
    build_cell_adjacency_graph_with_cells, build_cell_adjacency_lists_in_stratum,
    build_vertex_adjacency_graph_with_vertices,
};
use mesh_sieve::algs::interpolate::interpolate_edges_faces;
use mesh_sieve::algs::meshgen::{MeshGenOptions, StructuredCellType, structured_box_2d, structured_box_3d};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use mesh_sieve::topology::sieve::{MutableSieve, Sieve};

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

fn assert_two_cell_directed_adjacency(order: &[PointId], neighbors: &[Vec<PointId>]) {
    assert_eq!(order.len(), 2);
    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0], vec![order[1]]);
    assert_eq!(neighbors[1], vec![order[0]]);
}

fn assert_two_cell_undirected_edge(order: &[PointId], edges: &[(PointId, PointId)]) {
    assert_eq!(edges, &vec![(order[0].min(order[1]), order[0].max(order[1]))]);
}

fn tet_pair_mesh() -> (InMemorySieve<PointId, ()>, Section<CellType, VecStorage<CellType>>) {
    let (c0, c1) = (v(100), v(101));
    let (v0, v1, v2, v3, v4) = (v(1), v(2), v(3), v(4), v(5));
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    for p in [c0, c1, v0, v1, v2, v3, v4] {
        MutableSieve::add_point(&mut sieve, p);
    }
    for p in [v0, v1, v2, v3] {
        sieve.add_arrow(c0, p, ());
    }
    for p in [v0, v1, v2, v4] {
        sieve.add_arrow(c1, p, ());
    }

    let mut atlas = Atlas::default();
    for p in [c0, c1, v0, v1, v2, v3, v4] {
        atlas.try_insert(p, 1).unwrap();
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(atlas);
    cell_types.try_set(c0, &[CellType::Tetrahedron]).unwrap();
    cell_types.try_set(c1, &[CellType::Tetrahedron]).unwrap();
    for p in [v0, v1, v2, v3, v4] {
        cell_types.try_set(p, &[CellType::Vertex]).unwrap();
    }

    (sieve, cell_types)
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

#[test]
fn adjacency_triangles_and_quads() {
    let mut tri_mesh = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )
    .unwrap();
    let mut tri_sieve = tri_mesh.sieve;
    let mut tri_cell_types = tri_mesh.cell_types.take().unwrap();
    interpolate_edges_faces(&mut tri_sieve, &mut tri_cell_types).unwrap();

    let tri_lists = build_cell_adjacency_lists_in_stratum(
        &tri_sieve,
        CellAdjacencyStratum::Height(0),
        2,
        CellAdjacencyBy::Faces,
        AdjacencyOrdering::Sorted,
    )
    .unwrap();
    assert_two_cell_directed_adjacency(&tri_lists.order, &tri_lists.neighbors);
    let tri_edges = build_cell_adjacency_edges_in_stratum(
        &tri_sieve,
        CellAdjacencyStratum::Height(0),
        2,
        CellAdjacencyBy::Vertices,
        AdjacencyOrdering::Sorted,
    )
    .unwrap();
    assert_two_cell_undirected_edge(&tri_edges.order, &tri_edges.edges);

    let mut quad_mesh = structured_box_2d(
        2,
        1,
        [0.0, 0.0],
        [2.0, 1.0],
        StructuredCellType::Quadrilateral,
        MeshGenOptions::default(),
    )
    .unwrap();
    let mut quad_sieve = quad_mesh.sieve;
    let mut quad_cell_types = quad_mesh.cell_types.take().unwrap();
    interpolate_edges_faces(&mut quad_sieve, &mut quad_cell_types).unwrap();

    let quad_lists = build_cell_adjacency_lists_in_stratum(
        &quad_sieve,
        CellAdjacencyStratum::Height(0),
        2,
        CellAdjacencyBy::Faces,
        AdjacencyOrdering::Sorted,
    )
    .unwrap();
    assert_two_cell_directed_adjacency(&quad_lists.order, &quad_lists.neighbors);
}

#[test]
fn adjacency_tets_and_hexes() {
    let (mut tet_sieve, mut tet_cell_types) = tet_pair_mesh();
    interpolate_edges_faces(&mut tet_sieve, &mut tet_cell_types).unwrap();
    let tet_lists = build_cell_adjacency_lists_in_stratum(
        &tet_sieve,
        CellAdjacencyStratum::Height(0),
        3,
        CellAdjacencyBy::Faces,
        AdjacencyOrdering::Sorted,
    )
    .unwrap();
    assert_two_cell_directed_adjacency(&tet_lists.order, &tet_lists.neighbors);
    let tet_edges = build_cell_adjacency_edges_in_stratum(
        &tet_sieve,
        CellAdjacencyStratum::Height(0),
        3,
        CellAdjacencyBy::Vertices,
        AdjacencyOrdering::Sorted,
    )
    .unwrap();
    assert_two_cell_undirected_edge(&tet_edges.order, &tet_edges.edges);

    let mut hex_mesh = structured_box_3d(
        2,
        1,
        1,
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        StructuredCellType::Hexahedron,
        MeshGenOptions::default(),
    )
    .unwrap();
    let mut hex_sieve = hex_mesh.sieve;
    let mut hex_cell_types = hex_mesh.cell_types.take().unwrap();
    interpolate_edges_faces(&mut hex_sieve, &mut hex_cell_types).unwrap();
    let hex_lists = build_cell_adjacency_lists_in_stratum(
        &hex_sieve,
        CellAdjacencyStratum::Height(0),
        3,
        CellAdjacencyBy::Faces,
        AdjacencyOrdering::Sorted,
    )
    .unwrap();
    assert_two_cell_directed_adjacency(&hex_lists.order, &hex_lists.neighbors);
}

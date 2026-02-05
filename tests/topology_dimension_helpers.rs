use mesh_sieve::algs::interpolate::interpolate_edges_faces;
use mesh_sieve::algs::meshgen::{
    MeshGenOptions, StructuredCellType, structured_box_2d, structured_box_3d,
};
use mesh_sieve::topology::utils::points_of_dim_with_cell_types;

fn assert_sorted(points: &[mesh_sieve::topology::point::PointId]) {
    for window in points.windows(2) {
        assert!(window[0] < window[1], "points are not strictly ordered");
    }
}

#[test]
fn counts_for_2d_quad_mesh() {
    let mut mesh = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Quadrilateral,
        MeshGenOptions::default(),
    )
    .expect("mesh build");

    let mut cell_types = mesh.cell_types.take().expect("cell types");
    interpolate_edges_faces(&mut mesh.sieve, &mut cell_types).expect("interpolate");
    mesh.cell_types = Some(cell_types);

    let cells =
        points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 2).expect("cells");
    let edges =
        points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 1).expect("edges");
    let vertices = points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 0)
        .expect("vertices");

    assert_eq!(cells.len(), 1);
    assert_eq!(edges.len(), 4);
    assert_eq!(vertices.len(), 4);

    assert_sorted(&cells);
    assert_sorted(&edges);
    assert_sorted(&vertices);
}

#[test]
fn counts_for_3d_hex_mesh() {
    let mut mesh = structured_box_3d(
        1,
        1,
        1,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        StructuredCellType::Hexahedron,
        MeshGenOptions::default(),
    )
    .expect("mesh build");

    let mut cell_types = mesh.cell_types.take().expect("cell types");
    interpolate_edges_faces(&mut mesh.sieve, &mut cell_types).expect("interpolate");
    mesh.cell_types = Some(cell_types);

    let cells =
        points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 3).expect("cells");
    let faces =
        points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 2).expect("faces");
    let edges =
        points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 1).expect("edges");
    let vertices = points_of_dim_with_cell_types(&mut mesh.sieve, mesh.cell_types.as_ref(), 0)
        .expect("vertices");

    assert_eq!(cells.len(), 1);
    assert_eq!(faces.len(), 6);
    assert_eq!(edges.len(), 12);
    assert_eq!(vertices.len(), 8);

    assert_sorted(&cells);
    assert_sorted(&faces);
    assert_sorted(&edges);
    assert_sorted(&vertices);
}

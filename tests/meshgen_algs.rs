use mesh_sieve::algs::meshgen::{
    MeshGenOptions, StructuredCellType, reference_cell, structured_box_1d, structured_box_2d,
    structured_box_3d,
};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::sieve::Sieve;

#[test]
fn structured_generators_emit_expected_point_counts() {
    let m1 = structured_box_1d(2, 0.0, 1.0, MeshGenOptions::default()).unwrap();
    assert_eq!(m1.sieve.points().count(), 5);

    let m2 = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )
    .unwrap();
    assert_eq!(m2.sieve.points().count(), 6);

    let m3 = structured_box_3d(
        1,
        1,
        1,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        StructuredCellType::Hexahedron,
        MeshGenOptions::default(),
    )
    .unwrap();
    assert_eq!(m3.sieve.points().count(), 9);
}

#[test]
fn reference_cells_exist() {
    for ct in [
        CellType::Segment,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Hexahedron,
    ] {
        let mesh = reference_cell(ct, MeshGenOptions::default()).unwrap();
        assert!(mesh.sieve.points().count() > 0);
    }
}

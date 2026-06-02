use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::fluent::FluentReader;
use mesh_sieve::io::ply::PlyReader;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

#[test]
fn reads_ascii_ply_triangle_and_quad_surface() {
    let ply = b"ply
format ascii 1.0
element vertex 5
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
2 0 0
3 0 1 2
4 1 4 2 3
";
    let mesh = PlyReader::default().read(&ply[..]).unwrap();
    let coords = mesh.coordinates.as_ref().unwrap();
    assert_eq!(
        coords
            .section()
            .try_restrict(PointId::new(1).unwrap())
            .unwrap(),
        &[0.0, 0.0, 0.0]
    );
    let cell_types = mesh.cell_types.as_ref().unwrap();
    assert_eq!(
        cell_types.try_restrict(PointId::new(6).unwrap()).unwrap(),
        &[CellType::Triangle]
    );
    assert_eq!(
        cell_types.try_restrict(PointId::new(7).unwrap()).unwrap(),
        &[CellType::Quadrilateral]
    );
    assert_eq!(mesh.sieve.cone_points(PointId::new(6).unwrap()).count(), 3);
}

#[test]
fn reads_compact_fluent_fixture() {
    let fluent = b"vertices 4
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
cells 1
cell 1 2 3 4
";
    let mesh = FluentReader::default().read(&fluent[..]).unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();
    assert_eq!(
        cell_types.try_restrict(PointId::new(5).unwrap()).unwrap(),
        &[CellType::Quadrilateral]
    );
    assert_eq!(mesh.sieve.cone_points(PointId::new(5).unwrap()).count(), 4);
}

#[test]
fn reads_ply_fixture_file() {
    let ply = include_bytes!("fixtures/io/surface.ply");
    let mesh = PlyReader::default().read(&ply[..]).unwrap();
    assert!(mesh.coordinates.is_some());
    assert_eq!(mesh.sieve.cone_points(PointId::new(6).unwrap()).count(), 3);
}

#[test]
fn reads_fluent_fixture_file_with_labels() {
    let fluent = include_bytes!("fixtures/io/fluent_quad.msh");
    let mesh = FluentReader::default().read(&fluent[..]).unwrap();
    assert_eq!(mesh.sieve.cone_points(PointId::new(5).unwrap()).count(), 4);
    assert_eq!(
        mesh.labels
            .as_ref()
            .unwrap()
            .get_label(PointId::new(1).unwrap(), "fluent:bc"),
        Some(7)
    );
}

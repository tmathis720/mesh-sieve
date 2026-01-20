use mesh_sieve::io::gmsh::{GmshReader, GmshWriter};
use mesh_sieve::io::{SieveSectionReader, SieveSectionWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

const MESH: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0 0 0
2 1 0 0
3 0 1 0
4 0.5 0 0
5 0.5 0.5 0
6 0 0.5 0
$EndNodes
$Elements
1
10 9 2 7 42 1 2 3 4 5 6
$EndElements
"#;

#[test]
fn gmsh_round_trip_preserves_topology_labels_and_cell_types() {
    let reader = GmshReader::default();
    let mesh = reader.read(MESH.as_bytes()).expect("read gmsh");

    let coords = mesh.coordinates.as_ref().expect("coords");
    assert_eq!(coords.dimension(), 2, "dimension should be preserved");

    let mut output = Vec::new();
    let writer = GmshWriter::default();
    writer.write(&mut output, &mesh).expect("write gmsh");

    let round_tripped = reader.read(output.as_slice()).expect("read back");
    let coords_rt = round_tripped.coordinates.as_ref().expect("coords");
    assert_eq!(coords_rt.dimension(), 2, "dimension round-trip");

    let elem = PointId::new(10).expect("element id");
    let nodes: Vec<_> = round_tripped.sieve.cone_points(elem).collect();
    let expected_nodes = vec![
        PointId::new(1).unwrap(),
        PointId::new(2).unwrap(),
        PointId::new(3).unwrap(),
        PointId::new(4).unwrap(),
        PointId::new(5).unwrap(),
        PointId::new(6).unwrap(),
    ];
    assert_eq!(nodes, expected_nodes, "connectivity round-trip");

    let cell_types = round_tripped.cell_types.as_ref().expect("cell types");
    let cell_type = cell_types.try_restrict(elem).expect("cell type")[0];
    assert_eq!(cell_type, CellType::Triangle);

    let labels = round_tripped.labels.as_ref().expect("labels");
    assert_eq!(labels.get_label(elem, "gmsh:physical"), Some(7));
    assert_eq!(labels.get_label(elem, "gmsh:entity"), Some(42));
}

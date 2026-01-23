use mesh_sieve::io::exodus::{ExodusReader, ExodusWriter};
use mesh_sieve::io::{SieveSectionReader, SieveSectionWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

const MESH: &str = r#"EXODUS
DIM 2
NODES 3
1 0 0
2 1 0
3 0 1
ELEMENTS 1
10 Triangle 3 1 2 3
LABELS 2
10 exodus:material 7
10 exodus:region 42
END
"#;

#[test]
fn exodus_round_trip_preserves_labels_cell_types_and_coords() {
    let reader = ExodusReader::default();
    let mesh = reader.read(MESH.as_bytes()).expect("read exodus");

    let coords = mesh.coordinates.as_ref().expect("coords");
    assert_eq!(coords.dimension(), 2, "dimension preserved");

    let mut output = Vec::new();
    let writer = ExodusWriter::default();
    writer.write(&mut output, &mesh).expect("write exodus");

    let round_tripped = reader.read(output.as_slice()).expect("read back");
    let coords_rt = round_tripped.coordinates.as_ref().expect("coords");
    assert_eq!(coords_rt.dimension(), 2, "dimension round-trip");
    assert_eq!(
        coords_rt.try_restrict(PointId::new(2).unwrap()).unwrap(),
        &[1.0, 0.0]
    );

    let elem = PointId::new(10).expect("element id");
    let nodes: Vec<_> = round_tripped.sieve.cone_points(elem).collect();
    let expected_nodes = vec![
        PointId::new(1).unwrap(),
        PointId::new(2).unwrap(),
        PointId::new(3).unwrap(),
    ];
    assert_eq!(nodes, expected_nodes, "connectivity round-trip");

    let cell_types = round_tripped.cell_types.as_ref().expect("cell types");
    let cell_type = cell_types.try_restrict(elem).expect("cell type")[0];
    assert_eq!(cell_type, CellType::Triangle);

    let labels = round_tripped.labels.as_ref().expect("labels");
    assert_eq!(labels.get_label(elem, "exodus:material"), Some(7));
    assert_eq!(labels.get_label(elem, "exodus:region"), Some(42));
}

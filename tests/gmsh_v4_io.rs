use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::gmsh::{GmshReader, GmshWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

const V4_ASCII: &str = r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 7 "fluid"
$EndPhysicalNames
$Entities
0 0 1 0
1 0 0 0 1 1 0 1 7 0
$EndEntities
$Nodes
1 6 1 6
2 1 0 6
1
2
3
4
5
6
0 0 0
1 0 0
0 1 0
0.5 0 0
0.5 0.5 0
0 0.5 0
$EndNodes
$Elements
1 1 20 20
2 1 9 1
20 1 2 3 4 5 6
$EndElements
"#;

#[test]
fn reads_gmsh_v4_ascii_entities_high_order_and_metadata() {
    let mesh = GmshReader::default()
        .read(V4_ASCII.as_bytes())
        .expect("read v4 ascii");
    let elem = PointId::new(20).unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();
    assert_eq!(
        cell_types.try_restrict(elem).unwrap()[0],
        CellType::Triangle
    );
    let cone: Vec<_> = mesh.sieve.cone_points(elem).collect();
    assert_eq!(cone.len(), 6);
    let labels = mesh.labels.as_ref().unwrap();
    assert_eq!(labels.get_label(elem, "gmsh:physical"), Some(7));
    assert_eq!(labels.get_label(elem, "gmsh:entity"), Some(1));
    assert_eq!(labels.get_label(elem, "gmsh:entity_dim"), Some(2));
    assert_eq!(labels.get_label(elem, "gmsh:element_type"), Some(9));
    let ho = mesh.coordinates.as_ref().unwrap().high_order().unwrap();
    assert_eq!(ho.section().try_restrict(elem).unwrap().len(), 6);
}

fn push_line(out: &mut Vec<u8>, line: &str) {
    out.extend_from_slice(line.as_bytes());
    out.push(b'\n');
}
fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn v4_binary_triangle() -> Vec<u8> {
    let mut out = Vec::new();
    push_line(&mut out, "$MeshFormat");
    push_line(&mut out, "4.1 1 8");
    push_i32(&mut out, 1);
    push_line(&mut out, "");
    push_line(&mut out, "$EndMeshFormat");
    push_line(&mut out, "$Entities");
    push_u64(&mut out, 0);
    push_u64(&mut out, 0);
    push_u64(&mut out, 1);
    push_u64(&mut out, 0);
    push_i32(&mut out, 1);
    for _ in 0..6 {
        push_f64(&mut out, 0.0);
    }
    push_u64(&mut out, 1);
    push_i32(&mut out, 7);
    push_u64(&mut out, 0);
    push_line(&mut out, "");
    push_line(&mut out, "$EndEntities");
    push_line(&mut out, "$Nodes");
    push_u64(&mut out, 1);
    push_u64(&mut out, 3);
    push_u64(&mut out, 1);
    push_u64(&mut out, 3);
    push_i32(&mut out, 2);
    push_i32(&mut out, 1);
    push_i32(&mut out, 0);
    push_u64(&mut out, 3);
    for id in 1..=3 {
        push_u64(&mut out, id);
    }
    for (x, y, z) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] {
        push_f64(&mut out, x);
        push_f64(&mut out, y);
        push_f64(&mut out, z);
    }
    push_line(&mut out, "");
    push_line(&mut out, "$EndNodes");
    push_line(&mut out, "$Elements");
    push_u64(&mut out, 1);
    push_u64(&mut out, 1);
    push_u64(&mut out, 10);
    push_u64(&mut out, 10);
    push_i32(&mut out, 2);
    push_i32(&mut out, 1);
    push_i32(&mut out, 2);
    push_u64(&mut out, 1);
    push_u64(&mut out, 10);
    for id in 1..=3 {
        push_u64(&mut out, id);
    }
    push_line(&mut out, "");
    push_line(&mut out, "$EndElements");
    out
}

#[test]
fn reads_gmsh_v4_binary_triangle() {
    let mesh = GmshReader::default()
        .read(v4_binary_triangle().as_slice())
        .expect("read v4 binary");
    let elem = PointId::new(10).unwrap();
    assert_eq!(
        mesh.cell_types
            .as_ref()
            .unwrap()
            .try_restrict(elem)
            .unwrap()[0],
        CellType::Triangle
    );
    assert_eq!(
        mesh.labels
            .as_ref()
            .unwrap()
            .get_label(elem, "gmsh:physical"),
        Some(7)
    );
}

#[test]
fn writes_and_reads_gmsh_v4_ascii_round_trip() {
    let reader = GmshReader::default();
    let mesh = reader.read(V4_ASCII.as_bytes()).expect("read seed");
    let mut out = Vec::new();
    GmshWriter::default()
        .write_v4_ascii(&mut out, &mesh)
        .expect("write v4");
    let round_trip = reader.read(out.as_slice()).expect("read v4 output");
    let elem = PointId::new(20).unwrap();
    assert_eq!(
        round_trip
            .cell_types
            .as_ref()
            .unwrap()
            .try_restrict(elem)
            .unwrap()[0],
        CellType::Triangle
    );
}

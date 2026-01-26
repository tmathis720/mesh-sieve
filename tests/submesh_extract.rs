use mesh_sieve::algs::submesh::{SubmeshSelection, extract_by_label};
use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::gmsh::GmshReader;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use std::io::Cursor;

fn read_gmsh(
    contents: &str,
) -> mesh_sieve::io::MeshData<
    mesh_sieve::topology::sieve::InMemorySieve<PointId, ()>,
    f64,
    mesh_sieve::data::storage::VecStorage<f64>,
    mesh_sieve::data::storage::VecStorage<CellType>,
> {
    let reader = GmshReader::default();
    reader.read(Cursor::new(contents)).unwrap()
}

fn gmsh_triangle_mesh() -> &'static str {
    "$MeshFormat\n\
2.2 0 8\n\
$EndMeshFormat\n\
$Nodes\n\
3\n\
1 0 0 0\n\
2 1 0 0\n\
3 0 1 0\n\
$EndNodes\n\
$Elements\n\
4\n\
10 1 2 10 1 1 2\n\
11 1 2 20 1 2 3\n\
12 1 2 20 1 3 1\n\
13 2 2 99 1 1 2 3\n\
$EndElements\n"
}

#[test]
fn extract_boundary_submesh_single_label() {
    let mesh = read_gmsh(gmsh_triangle_mesh());
    let labels = mesh.labels.as_ref().unwrap();

    let (submesh, maps) = extract_by_label(
        &mesh,
        labels,
        "gmsh:physical",
        10,
        SubmeshSelection::FullClosure,
    )
    .unwrap();
    let points: Vec<_> = submesh.sieve.points().collect();
    assert_eq!(points.len(), 3);

    let parent_line = PointId::new(10).unwrap();
    let sub_line = maps.parent_to_sub[&parent_line];
    let cone: Vec<_> = submesh.sieve.cone_points(sub_line).collect();
    assert_eq!(cone.len(), 2);

    let labels_out = submesh.labels.as_ref().unwrap();
    assert_eq!(labels_out.get_label(sub_line, "gmsh:physical"), Some(10));

    let coords = submesh.coordinates.as_ref().unwrap();
    for parent_node in [1u64, 2u64] {
        let node = PointId::new(parent_node).unwrap();
        let sub_node = maps.parent_to_sub[&node];
        assert!(coords.section().atlas().contains(sub_node));
    }

    let cell_types = submesh.cell_types.as_ref().unwrap();
    assert_eq!(
        cell_types.try_restrict(sub_line).unwrap(),
        &[CellType::Segment]
    );
}

#[test]
fn extract_boundary_submesh_multiple_lines() {
    let mesh = read_gmsh(gmsh_triangle_mesh());
    let labels = mesh.labels.as_ref().unwrap();

    let (submesh, maps) = extract_by_label(
        &mesh,
        labels,
        "gmsh:physical",
        20,
        SubmeshSelection::FullClosure,
    )
    .unwrap();
    let points: Vec<_> = submesh.sieve.points().collect();
    assert_eq!(points.len(), 5);

    for parent_line in [11u64, 12u64] {
        let parent_line = PointId::new(parent_line).unwrap();
        let sub_line = maps.parent_to_sub[&parent_line];
        let cone: Vec<_> = submesh.sieve.cone_points(sub_line).collect();
        assert_eq!(cone.len(), 2);
        assert_eq!(
            submesh
                .labels
                .as_ref()
                .unwrap()
                .get_label(sub_line, "gmsh:physical"),
            Some(20)
        );
    }
}

#[test]
fn extract_boundary_faces_depth_limited() {
    let mesh = read_gmsh(gmsh_triangle_mesh());
    let labels = mesh.labels.as_ref().unwrap();

    let (submesh, maps) = extract_by_label(
        &mesh,
        labels,
        "gmsh:physical",
        20,
        SubmeshSelection::ClosureDepth(0),
    )
    .unwrap();
    let points: Vec<_> = submesh.sieve.points().collect();
    assert_eq!(points.len(), 2);

    for parent_line in [11u64, 12u64] {
        let parent_line = PointId::new(parent_line).unwrap();
        let sub_line = maps.parent_to_sub[&parent_line];
        let cone: Vec<_> = submesh.sieve.cone_points(sub_line).collect();
        assert!(cone.is_empty());
        assert_eq!(
            submesh
                .labels
                .as_ref()
                .unwrap()
                .get_label(sub_line, "gmsh:physical"),
            Some(20)
        );
    }
}

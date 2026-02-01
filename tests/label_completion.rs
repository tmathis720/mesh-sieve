use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::gmsh::GmshReader;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::{complete_label_value, extract_submesh_from_label};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use std::collections::HashSet;
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
fn label_completion_adds_closure_points() {
    let mesh = read_gmsh(gmsh_triangle_mesh());
    let labels = mesh.labels.as_ref().unwrap();

    let completed = complete_label_value(&mesh.sieve, labels, "gmsh:physical", 10);
    let node1 = PointId::new(1).unwrap();
    let node2 = PointId::new(2).unwrap();
    let node3 = PointId::new(3).unwrap();

    assert_eq!(completed.get_label(node1, "gmsh:physical"), Some(10));
    assert_eq!(completed.get_label(node2, "gmsh:physical"), Some(10));
    assert_eq!(completed.get_label(node3, "gmsh:physical"), None);
}

#[test]
fn extract_submesh_from_label_preserves_topology_and_coordinates() {
    let mesh = read_gmsh(gmsh_triangle_mesh());
    let labels = mesh.labels.as_ref().unwrap();

    let (submesh, maps) = extract_submesh_from_label(&mesh, labels, "gmsh:physical", 10).unwrap();

    let parent_line = PointId::new(10).unwrap();
    let sub_line = maps.parent_to_sub[&parent_line];
    let cone: HashSet<_> = submesh.sieve.cone_points(sub_line).collect();
    let sub_node1 = maps.parent_to_sub[&PointId::new(1).unwrap()];
    let sub_node2 = maps.parent_to_sub[&PointId::new(2).unwrap()];
    assert_eq!(cone.len(), 2);
    assert!(cone.contains(&sub_node1));
    assert!(cone.contains(&sub_node2));

    let parent_coords = mesh.coordinates.as_ref().unwrap();
    let sub_coords = submesh.coordinates.as_ref().unwrap();
    for parent_node in [1u64, 2u64] {
        let parent_node = PointId::new(parent_node).unwrap();
        let sub_node = maps.parent_to_sub[&parent_node];
        assert_eq!(
            parent_coords.section().try_restrict(parent_node).unwrap(),
            sub_coords.section().try_restrict(sub_node).unwrap()
        );
    }

    let labels_out = submesh.labels.as_ref().unwrap();
    assert_eq!(labels_out.get_label(sub_node1, "gmsh:physical"), Some(10));
    assert_eq!(labels_out.get_label(sub_node2, "gmsh:physical"), Some(10));
}

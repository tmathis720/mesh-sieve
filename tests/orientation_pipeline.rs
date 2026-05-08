use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::algs::distribute_mesh;
use mesh_sieve::algs::interpolate::interpolate_edges_faces;
use mesh_sieve::algs::submesh::{SubmeshSelection, extract_by_label};
use mesh_sieve::io::gmsh::{GmshReadOptions, GmshReader};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh_full_topology;
use mesh_sieve::topology::sieve::{OrientedSieve, Sieve};

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn mixed_import_interpolate_distribute_submesh_refine_preserves_orientation_closure() {
    let msh = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 1 1 0
$EndNodes
$Elements
2
10 2 2 7 1 1 2 3
20 4 2 9 2 1 2 3 4
$EndElements
"#;

    let reader = GmshReader::default();
    let mut mesh = reader
        .read_with_options(
            msh.as_bytes(),
            GmshReadOptions {
                validate_topology: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut cell_types = mesh.cell_types.take().unwrap();

    let interpolation = interpolate_edges_faces(&mut mesh.sieve, &mut cell_types).unwrap();
    assert!(!interpolation.edge_points.is_empty());
    assert!(
        mesh.sieve
            .cone_o(pid(10))
            .any(|(_, orientation)| orientation == -1)
    );

    let triangle_closure = mesh.sieve.closure_o([pid(10)]);
    assert!(triangle_closure.iter().any(|(point, _)| *point == pid(1)));
    assert!(
        triangle_closure
            .iter()
            .any(|(point, orientation)| *point == pid(3) && *orientation != 0)
    );

    let max_id = mesh.sieve.points().map(|p| p.get()).max().unwrap() as usize;
    let parts = vec![0; max_id];
    let (distributed, _overlap) = distribute_mesh(&mesh.sieve, &parts, &NoComm).unwrap();
    assert_eq!(
        distributed.cone_o(pid(10)).collect::<Vec<_>>(),
        mesh.sieve.cone_o(pid(10)).collect::<Vec<_>>()
    );

    mesh.cell_types = Some(cell_types.clone());
    let labels = mesh.labels.as_ref().unwrap();
    let (submesh, maps) = extract_by_label(
        &mesh,
        labels,
        "gmsh:physical",
        7,
        SubmeshSelection::FullClosure,
    )
    .unwrap();
    let sub_triangle = maps.parent_to_sub[&pid(10)];
    assert!(
        submesh
            .sieve
            .cone_o(sub_triangle)
            .any(|(_, orientation)| orientation == -1)
    );

    let mut refine_input = mesh.sieve.clone();
    let refined = refine_mesh_full_topology(&mut refine_input, &cell_types).unwrap();
    assert!(
        refined
            .sieve
            .closure_o([refined.cell_refinement[0].1[0].0])
            .len()
            > 1
    );
}

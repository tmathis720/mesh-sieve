use mesh_sieve::algs::extrude::extrude_surface_layers;
use mesh_sieve::algs::interpolate::interpolate_edges_faces;
use mesh_sieve::algs::traversal::{TraversalOrder, closure_to_depth};
use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::gmsh::{GmshReadOptions, GmshReader};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh;

const MIXED_ELEMENT_MESH: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 2 0 0
$EndNodes
$Elements
2
10 2 0 1 2 3
11 3 0 2 3 4 5
$EndElements
"#;

#[test]
fn gmsh_mixed_element_mesh_ingests_and_traverses() {
    let reader = GmshReader::default();
    let mesh = reader
        .read_with_options(
            MIXED_ELEMENT_MESH.as_bytes(),
            GmshReadOptions {
                validate_mixed_dimensions: true,
                ..Default::default()
            },
        )
        .expect("read mixed-element mesh");

    let cell_types = mesh.cell_types.as_ref().expect("cell types");
    let tri = PointId::new(10).unwrap();
    let quad = PointId::new(11).unwrap();
    assert_eq!(cell_types.try_restrict(tri).unwrap()[0], CellType::Triangle);
    assert_eq!(
        cell_types.try_restrict(quad).unwrap()[0],
        CellType::Quadrilateral
    );

    let closure = closure_to_depth(&mesh.sieve, [tri, quad], 0, Some(TraversalOrder::Sorted))
        .expect("closure traversal");
    let mut closure_sorted = closure.clone();
    closure_sorted.sort_unstable();
    let expected: Vec<PointId> = (1..=5).map(|id| PointId::new(id).unwrap()).collect();
    assert_eq!(closure_sorted, expected);
}

#[test]
fn mixed_element_pipelines_support_surface_meshes() {
    let reader = GmshReader::default();
    let mesh = reader
        .read(MIXED_ELEMENT_MESH.as_bytes())
        .expect("read mixed-element mesh");

    let coords = mesh.coordinates.as_ref().expect("coordinates");
    let cell_types = mesh.cell_types.as_ref().expect("cell types");

    let mut sieve_interp = mesh.sieve.clone();
    let mut cell_types_interp = cell_types.clone();
    let interpolation = interpolate_edges_faces(&mut sieve_interp, &mut cell_types_interp)
        .expect("interpolate mixed elements");
    assert!(!interpolation.edge_points.is_empty());

    let mut sieve_refine = mesh.sieve.clone();
    let refined = refine_mesh(&mut sieve_refine, cell_types).expect("refine mixed elements");
    assert_eq!(refined.cell_refinement.len(), 2);

    let extruded = extrude_surface_layers(&mesh.sieve, cell_types, coords, &[0.0, 1.0]).unwrap();
    let cell_types_out = extruded.cell_types.as_ref().expect("extruded types");
    assert!(
        cell_types_out
            .iter()
            .any(|(_, ty)| ty[0] == CellType::Prism)
    );
    assert!(
        cell_types_out
            .iter()
            .any(|(_, ty)| ty[0] == CellType::Hexahedron)
    );
}

use mesh_sieve::io::gmsh::{GmshReadOptions, GmshReader};
use mesh_sieve::mesh_error::MeshSieveError;

const DUPLICATE_ARROW_MESH: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
1
10 2 0 1 2 2
$EndElements
"#;

#[test]
fn gmsh_duplicate_arrows_rejected_when_validation_enabled() {
    let reader = GmshReader::default();
    let err = reader
        .read_with_options(
            DUPLICATE_ARROW_MESH.as_bytes(),
            GmshReadOptions {
                check_geometry: false,
                validate_topology: true,
            },
        )
        .expect_err("expected duplicate arrow validation error");
    assert!(
        matches!(err, MeshSieveError::DuplicateArrow { .. }),
        "unexpected error: {err:?}"
    );
}

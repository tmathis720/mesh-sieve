use mesh_sieve::io::exodus::{ExodusReadOptions, ExodusReader};
use mesh_sieve::mesh_error::MeshSieveError;

const CONE_SIZE_MISMATCH: &str = r#"EXODUS
DIM 2
NODES 3
1 0 0
2 1 0
3 0 1
ELEMENTS 1
10 Triangle 2 1 2
LABELS 0
END
"#;

const CLOSURE_MISMATCH: &str = r#"EXODUS
DIM 2
NODES 5
1 0 0
2 1 0
3 0 1
4 1 1
5 0 1
ELEMENTS 2
10 Triangle 3 1 2 3
2 Segment 2 4 5
LABELS 0
END
"#;

#[test]
fn exodus_cone_size_mismatch_rejected_with_validation() {
    let reader = ExodusReader::default();
    let err = reader
        .read_with_options(
            CONE_SIZE_MISMATCH.as_bytes(),
            ExodusReadOptions {
                validate_topology: true,
            },
        )
        .expect_err("expected cone size mismatch error");
    assert!(
        matches!(err, MeshSieveError::ConeSizeMismatch { .. }),
        "unexpected error: {err:?}"
    );
}

#[test]
fn exodus_closure_mismatch_rejected_with_validation() {
    let reader = ExodusReader::default();
    let err = reader
        .read_with_options(
            CLOSURE_MISMATCH.as_bytes(),
            ExodusReadOptions {
                validate_topology: true,
            },
        )
        .expect_err("expected closure mismatch error");
    assert!(
        matches!(err, MeshSieveError::ClosureVertexCountMismatch { .. }),
        "unexpected error: {err:?}"
    );
}

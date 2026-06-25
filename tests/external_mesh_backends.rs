use std::fs;
use std::path::PathBuf;

use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;

#[cfg(unix)]
fn fake_executable(name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let mut path = std::env::temp_dir();
    path.push(format!(
        "mesh-sieve-{name}-{}-{}.sh",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    // Publish the executable only after its writable file handle has been
    // closed. This avoids transient ETXTBSY failures when tests are run by a
    // highly parallel runner or on filesystems with coarse timestamp reuse.
    let staging = path.with_extension("staging");
    fs::write(&staging, body).unwrap();
    let mut perms = fs::metadata(&staging).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&staging, perms).unwrap();
    fs::rename(staging, &path).unwrap();
    path
}

#[cfg(feature = "triangle-support")]
#[test]
fn triangle_backend_preserves_region_and_boundary_markers() {
    use mesh_sieve::algs::meshgen::{
        TriangleInput, TriangleOptions, TriangleRegion, generate_with_triangle,
    };

    let exe = fake_executable(
        "triangle",
        r#"#!/usr/bin/env bash
set -eu
poly="${@: -1}"
prefix="${poly%.poly}"
cat > "${prefix}.1.node" <<'EOF'
3 2 0 1
1 0 0 11
2 1 0 12
3 0 1 13
EOF
cat > "${prefix}.1.ele" <<'EOF'
1 3 1
1 1 2 3 99
EOF
"#,
    );
    let input = TriangleInput {
        vertices: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        vertex_markers: vec![11, 12, 13],
        segments: vec![[0, 1], [1, 2], [2, 0]],
        segment_markers: vec![21, 22, 23],
        regions: vec![TriangleRegion {
            point: [0.25, 0.25],
            attribute: 99,
            max_area: Some(0.1),
        }],
        ..TriangleInput::default()
    };
    let mesh = generate_with_triangle(
        &input,
        &TriangleOptions {
            executable: exe.display().to_string(),
            ..TriangleOptions::default()
        },
    )
    .expect("triangle fake backend");

    let labels = mesh.labels.as_ref().expect("labels");
    assert_eq!(
        labels.get_label(PointId::new(1).unwrap(), "triangle:vertex_marker"),
        Some(11)
    );
    assert_eq!(
        labels.get_label(PointId::new(2).unwrap(), "boundary"),
        Some(12)
    );
    assert_eq!(
        labels.get_label(PointId::new(4).unwrap(), "triangle:region"),
        Some(99)
    );
    assert_eq!(
        mesh.cell_types
            .as_ref()
            .unwrap()
            .try_restrict(PointId::new(4).unwrap())
            .unwrap()[0],
        CellType::Triangle
    );
}

#[cfg(feature = "triangle-support")]
#[test]
fn triangle_backend_reports_malformed_output() {
    use mesh_sieve::algs::meshgen::{TriangleInput, TriangleOptions, generate_with_triangle};

    let exe = fake_executable(
        "triangle-bad",
        r#"#!/usr/bin/env bash
set -eu
poly="${@: -1}"
prefix="${poly%.poly}"
cat > "${prefix}.1.node" <<'EOF'
3 2 0 0
1 0 0
2 1 0
3 0 1
EOF
cat > "${prefix}.1.ele" <<'EOF'
1 6 0
1 1 2 3 4 5 6
EOF
"#,
    );
    let err = generate_with_triangle(
        &TriangleInput {
            vertices: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            segments: vec![[0, 1], [1, 2], [2, 0]],
            ..TriangleInput::default()
        },
        &TriangleOptions {
            executable: exe.display().to_string(),
            ..TriangleOptions::default()
        },
    )
    .expect_err("unsupported element order should fail")
    .to_string();
    assert!(err.contains("unsupported element order"), "{err}");
}

#[cfg(feature = "tetgen-support")]
#[test]
fn tetgen_backend_preserves_region_and_boundary_markers() {
    use mesh_sieve::algs::meshgen::{
        TetGenInput, TetGenOptions, TetGenRegion, generate_with_tetgen,
    };

    let exe = fake_executable(
        "tetgen",
        r#"#!/usr/bin/env bash
set -eu
poly="${@: -1}"
prefix="${poly%.poly}"
cat > "${prefix}.1.node" <<'EOF'
4 3 0 1
1 0 0 0 31
2 1 0 0 32
3 0 1 0 33
4 0 0 1 34
EOF
cat > "${prefix}.1.ele" <<'EOF'
1 4 1
1 1 2 3 4 77
EOF
"#,
    );
    let input = TetGenInput {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vertex_markers: vec![31, 32, 33, 34],
        facets: vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![2, 0, 3]],
        facet_markers: vec![41, 42, 43, 44],
        regions: vec![TetGenRegion {
            point: [0.1, 0.1, 0.1],
            attribute: 77,
            max_volume: Some(0.2),
        }],
        ..TetGenInput::default()
    };
    let mesh = generate_with_tetgen(
        &input,
        &TetGenOptions {
            executable: exe.display().to_string(),
            ..TetGenOptions::default()
        },
    )
    .expect("tetgen fake backend");

    let labels = mesh.labels.as_ref().expect("labels");
    assert_eq!(
        labels.get_label(PointId::new(1).unwrap(), "tetgen:vertex_marker"),
        Some(31)
    );
    assert_eq!(
        labels.get_label(PointId::new(5).unwrap(), "tetgen:region"),
        Some(77)
    );
    assert_eq!(
        mesh.cell_types
            .as_ref()
            .unwrap()
            .try_restrict(PointId::new(5).unwrap())
            .unwrap()[0],
        CellType::Tetrahedron
    );
}

#[cfg(feature = "gmsh-support")]
#[test]
fn gmsh_backend_generates_and_remeshes_with_physical_labels() {
    use mesh_sieve::algs::meshgen::{
        GmshInput, GmshOptions, MeshGenOptions, StructuredCellType, generate_with_gmsh,
        remesh_with_gmsh, structured_box_2d,
    };
    use mesh_sieve::topology::labels::LabelSet;

    let exe = fake_executable(
        "gmsh",
        r#"#!/usr/bin/env bash
set -eu
out=""
in=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) shift; out="$1" ;;
    *.msh) in="$1" ;;
  esac
  shift || true
done
if [ -n "$in" ] && [ -f "$in" ]; then
  cp "$in" "$out"
else
cat > "$out" <<'EOF'
$MeshFormat
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
10 2 2 7 42 1 2 3
$EndElements
EOF
fi
"#,
    );
    let opts = GmshOptions {
        executable: exe.display().to_string(),
        format: "msh2".into(),
        ..GmshOptions::default()
    };
    let mesh = generate_with_gmsh(
        &GmshInput {
            geo: "Point(1) = {0,0,0};".into(),
            dimension: 2,
        },
        &opts,
    )
    .expect("gmsh fake backend");
    let labels = mesh.labels.as_ref().expect("labels");
    assert_eq!(
        labels.get_label(PointId::new(10).unwrap(), "gmsh:physical"),
        Some(7)
    );
    assert_eq!(
        labels.get_label(PointId::new(10).unwrap(), "gmsh:entity"),
        Some(42)
    );

    let mut seed = structured_box_2d(
        1,
        1,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )
    .unwrap();
    let mut labels = LabelSet::new();
    labels.set_label(PointId::new(5).unwrap(), "gmsh:physical", 9);
    labels.set_label(PointId::new(5).unwrap(), "gmsh:entity", 5);
    seed.labels = Some(labels);
    let remeshed = remesh_with_gmsh(&seed, 2, &opts).expect("gmsh fake remesh");
    assert_eq!(
        remeshed
            .labels
            .as_ref()
            .unwrap()
            .get_label(PointId::new(5).unwrap(), "gmsh:physical"),
        Some(9)
    );
}

#[cfg(any(
    feature = "triangle-support",
    feature = "tetgen-support",
    feature = "gmsh-support"
))]
#[test]
fn missing_executable_diagnostic_names_backend() {
    #[cfg(feature = "triangle-support")]
    {
        use mesh_sieve::algs::meshgen::{TriangleInput, TriangleOptions, generate_with_triangle};
        let err = generate_with_triangle(
            &TriangleInput {
                vertices: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                segments: vec![[0, 1], [1, 2], [2, 0]],
                ..TriangleInput::default()
            },
            &TriangleOptions {
                executable: "mesh-sieve-definitely-missing-triangle".into(),
                ..TriangleOptions::default()
            },
        )
        .expect_err("missing executable")
        .to_string();
        assert!(err.contains("failed to execute Triangle"), "{err}");
    }
}

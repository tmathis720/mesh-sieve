//! DMPLEX-like external mesh creation/refinement/remeshing workflows.
//!
//! Run with one or more backends enabled, for example:
//! `cargo run --example external_mesh_workflows --features triangle-support,gmsh-support`.

use mesh_sieve::algs::meshgen::{
    GmshInput, GmshOptions, MeshGenOptions, StructuredCellType, TetGenInput, TetGenOptions,
    TriangleInput, TriangleOptions, generate_with_gmsh, generate_with_tetgen,
    generate_with_triangle, remesh_with_gmsh, structured_box_2d,
};
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn count_cells(
    mesh: &MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
) -> usize {
    let Some(cell_types) = mesh.cell_types.as_ref() else {
        return 0;
    };
    cell_types
        .atlas()
        .points()
        .filter(|point| {
            cell_types
                .try_restrict(*point)
                .is_ok_and(|values| values[0] != CellType::Vertex)
        })
        .count()
}

fn report(name: &str, result: mesh_sieve::algs::meshgen::MeshGenResult) {
    match result {
        Ok(mesh) => println!(
            "{name}: {} cells, {} points",
            count_cells(&mesh),
            mesh.sieve.points().count()
        ),
        Err(err) => println!("{name}: unavailable ({err})"),
    }
}

fn main() {
    // Create: Triangle constrained triangulation of a square with a square hole.
    let triangle_domain = TriangleInput {
        vertices: vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.4, 0.4],
            [0.6, 0.4],
            [0.6, 0.6],
            [0.4, 0.6],
        ],
        segments: vec![
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
        ],
        holes: vec![[0.5, 0.5]],
    };
    let triangle_options = TriangleOptions {
        max_area: Some(0.02),
        ..TriangleOptions::default()
    };
    report(
        "triangle create",
        generate_with_triangle(&triangle_domain, &triangle_options),
    );

    // Refine: Gmsh creates a refined unit square from a simple geometry script.
    let gmsh_geo = r#"
        SetFactory("OpenCASCADE");
        Rectangle(1) = {0, 0, 0, 1, 1, 0};
        Mesh.CharacteristicLengthMin = 0.05;
        Mesh.CharacteristicLengthMax = 0.10;
    "#;
    let gmsh_input = GmshInput {
        geo: gmsh_geo.into(),
        dimension: 2,
    };
    report(
        "gmsh create/refine",
        generate_with_gmsh(&gmsh_input, &GmshOptions::default()),
    );

    // Remesh: start from mesh-sieve's structured triangles, then ask Gmsh to reprocess it.
    let coarse = structured_box_2d(
        2,
        2,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )
    .expect("structured seed mesh");
    report(
        "gmsh remesh",
        remesh_with_gmsh(&coarse, 2, &GmshOptions::default()),
    );

    // Create: TetGen tetrahedralizes a closed tetrahedral PLC.
    let tet_domain = TetGenInput {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        facets: vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![2, 0, 3]],
        holes: Vec::new(),
    };
    report(
        "tetgen create",
        generate_with_tetgen(&tet_domain, &TetGenOptions::default()),
    );
}

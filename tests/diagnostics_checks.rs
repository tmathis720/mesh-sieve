use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::diagnostics::{MeshCheckOptions, mesh_dot_graph, mesh_json_debug_dump, run_mesh_checks};
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, Sieve};

fn pid(id: u64) -> PointId { PointId::new(id).unwrap() }

fn tiny_mesh() -> MeshSieve {
    let mut mesh = MeshSieve::default();
    mesh.add_arrow(pid(3), pid(1), ());
    mesh.add_arrow(pid(3), pid(2), ());
    mesh
}

#[test]
fn check_all_detects_missing_ownership_precisely() {
    let mut mesh = tiny_mesh();
    let mut ownership = PointOwnership::default();
    ownership.set(pid(1), 0, false).unwrap();

    let err = run_mesh_checks::<f64, VecStorage<f64>, VecStorage<mesh_sieve::topology::cell_type::CellType>>(
        &mut mesh,
        None,
        None,
        Some(&ownership),
        None,
        std::iter::empty(),
        MeshCheckOptions::all(),
    )
    .unwrap_err();

    assert!(matches!(err, MeshSieveError::TopologyPointMissingOwnership { .. }));
}

#[test]
fn viewers_emit_expected_formats() {
    let mesh = tiny_mesh();
    let dot = mesh_dot_graph(&mesh);
    assert!(dot.starts_with("digraph MeshSieve"));
    let json = mesh_json_debug_dump(&mesh);
    assert!(json.contains("\"edges\""));

    let mut atlas = Atlas::default();
    atlas.try_insert(pid(1), 1).unwrap();
    let _section: Section<f64, VecStorage<f64>> = Section::new(atlas);
}

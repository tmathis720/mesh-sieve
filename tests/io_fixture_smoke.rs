use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::exodus::ExodusReader;
use mesh_sieve::io::gmsh::GmshReader;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;

#[test]
fn reads_gmsh_fixture_file() {
    let fixture = include_bytes!("fixtures/io/gmsh_v2_triangle.msh");
    let mesh = GmshReader::default().read(&fixture[..]).unwrap();
    assert_eq!(
        mesh.cell_types
            .as_ref()
            .unwrap()
            .try_restrict(PointId::new(10).unwrap())
            .unwrap(),
        &[CellType::Triangle]
    );
}

#[test]
fn reads_exodus_ascii_fixture_file() {
    let fixture = include_bytes!("fixtures/io/exodus_ascii_triangle.exo");
    let mesh = ExodusReader::default().read(&fixture[..]).unwrap();
    assert_eq!(
        mesh.labels
            .as_ref()
            .unwrap()
            .get_label(PointId::new(10).unwrap(), "exodus:material"),
        Some(7)
    );
}

#[test]
fn hdf5_family_fixture_manifests_are_present() {
    assert!(include_str!("fixtures/io/cgns_hdf5_fixture.json").contains("CGNS/HDF5"));
    assert!(include_str!("fixtures/io/exodus_ii_hdf5_fixture.json").contains("Exodus II"));
    assert!(include_str!("fixtures/io/petsc_dmplex_hdf5_fixture.json").contains("PETSc DMPlex"));
}

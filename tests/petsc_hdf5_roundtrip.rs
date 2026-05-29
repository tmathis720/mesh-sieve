use hdf5::File;
use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::algs::point_sf::PointSF;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::petsc_hdf5::{
    DMPLEX_STORAGE_VERSION, PetscHdf5Reader, PetscHdf5Writer, PetscLoadFilter, PetscLoadMode,
    PetscLoadOptions, read_mesh_and_migration_provenance, read_mesh_from_petsc_hdf5,
    read_mesh_from_petsc_hdf5_with_options, write_mesh_to_petsc_hdf5, write_migration_metadata,
};
use mesh_sieve::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, OrientedSieve, Sieve};
use std::collections::BTreeMap;
use std::str::FromStr;

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn fixture_mesh(
    partitioned: bool,
) -> MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>> {
    let mut sieve = MeshSieve::default();
    sieve.add_arrow_o(p(4), p(1), (), 0);
    sieve.add_arrow_o(p(4), p(2), (), 0);
    sieve.add_arrow_o(p(4), p(3), (), 0);
    if partitioned {
        sieve.add_arrow_o(p(5), p(2), (), 0);
        sieve.add_arrow_o(p(5), p(3), (), 0);
        sieve.add_arrow_o(p(5), p(6), (), 0);
    }

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(p(4), 1).unwrap();
    if partitioned {
        cell_atlas.try_insert(p(5), 1).unwrap();
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(p(4), &[CellType::Triangle]).unwrap();
    if partitioned {
        cell_types.try_set(p(5), &[CellType::Triangle]).unwrap();
    }

    let mut atlas = Atlas::default();
    for id in [1, 2, 3, 6] {
        if partitioned || id != 6 {
            atlas.try_insert(p(id), 1).unwrap();
        }
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    for (id, value) in [(1, 10.0), (2, 20.0), (3, 30.0), (6, 40.0)] {
        if partitioned || id != 6 {
            section.try_set(p(id), &[value]).unwrap();
        }
    }

    let mut labels = LabelSet::new();
    labels.set_label(p(1), "marker", 1);
    labels.set_label(p(3), "marker", 1);
    if partitioned {
        labels.set_label(p(2), "partition", 0);
        labels.set_label(p(6), "partition", 1);
    }

    let mut coord_atlas = Atlas::default();
    for id in [1, 2, 3, 6] {
        if partitioned || id != 6 {
            coord_atlas.try_insert(p(id), 2).unwrap();
        }
    }
    let mut coord_section = Section::<f64, VecStorage<f64>>::new(coord_atlas);
    for (id, xy) in [
        (1, [0.0, 0.0]),
        (2, [1.0, 0.0]),
        (3, [0.0, 1.0]),
        (6, [1.0, 1.0]),
    ] {
        if partitioned || id != 6 {
            coord_section.try_set(p(id), &xy).unwrap();
        }
    }

    MeshData {
        sieve,
        coordinates: Some(Coordinates::from_section(2, 2, coord_section).unwrap()),
        sections: BTreeMap::from([("u".to_string(), section)]),
        mixed_sections: Default::default(),
        labels: Some(labels),
        cell_types: Some(cell_types),
        discretization: None,
    }
}

fn assert_roundtrip(mesh: &MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>) {
    let mut bytes = Vec::new();
    PetscHdf5Writer::new("mesh", "dm")
        .write(&mut bytes, mesh)
        .unwrap();
    let reread = PetscHdf5Reader::new("mesh", "dm")
        .read(bytes.as_slice())
        .unwrap();
    let mut expected_base = mesh.sieve.base_points().collect::<Vec<_>>();
    let mut actual_base = reread.sieve.base_points().collect::<Vec<_>>();
    expected_base.sort_unstable();
    actual_base.sort_unstable();
    assert_eq!(expected_base, actual_base);
    assert_eq!(
        reread.sections["u"].gather_in_order(),
        mesh.sections["u"].gather_in_order()
    );
    assert_eq!(
        reread.labels.as_ref().unwrap().stratum_points("marker", 1),
        mesh.labels.as_ref().unwrap().stratum_points("marker", 1)
    );
}

#[test]
fn polytope_cell_types_roundtrip_through_petsc_hdf5_metadata() {
    let mut sieve = MeshSieve::default();
    let polygon = p(10);
    let simplex4 = p(20);
    for id in 1..=5 {
        sieve.add_arrow_o(polygon, p(id), (), 0);
    }
    for id in 30..=34 {
        sieve.add_arrow_o(simplex4, p(id), (), 0);
    }

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(polygon, 1).unwrap();
    cell_atlas.try_insert(simplex4, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types
        .try_set(polygon, &[CellType::Polygon(5)])
        .unwrap();
    cell_types
        .try_set(simplex4, &[CellType::Simplex(4)])
        .unwrap();

    let mesh = MeshData {
        sieve,
        coordinates: None,
        sections: BTreeMap::new(),
        mixed_sections: Default::default(),
        labels: None,
        cell_types: Some(cell_types),
        discretization: None,
    };
    let mut bytes = Vec::new();
    PetscHdf5Writer::new("mesh", "dm")
        .write(&mut bytes, &mesh)
        .unwrap();
    let reread = PetscHdf5Reader::new("mesh", "dm")
        .read(bytes.as_slice())
        .unwrap();
    let reread_types = reread.cell_types.as_ref().unwrap();
    assert_eq!(
        reread_types.try_restrict(polygon).unwrap()[0],
        CellType::Polygon(5)
    );
    assert_eq!(
        reread_types.try_restrict(simplex4).unwrap()[0],
        CellType::Simplex(4)
    );
}

#[test]
fn serial_fixture_roundtrips_and_exposes_dmplex_v3_paths() {
    let mesh = fixture_mesh(false);
    assert_roundtrip(&mesh);

    let path = std::env::temp_dir().join("mesh_sieve_serial_dmplex_v3_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();

    assert_eq!(
        file.dataset("dmplex_storage_version")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![DMPLEX_STORAGE_VERSION]
    );
    assert!(
        file.dataset("/topologies/mesh/topology/permutation")
            .is_ok()
    );
    assert!(
        file.dataset("/topologies/mesh/topology/strata/0/cone_sizes")
            .is_ok()
    );
    assert!(
        file.dataset("/topologies/mesh/topology/strata/1/cones")
            .is_ok()
    );
    assert!(file.group("/topologies/mesh/labels/marker/1").is_ok());
    assert!(file.group("/topologies/mesh/dms/dm/section/u").is_ok());
    assert!(file.dataset("/topologies/mesh/dms/dm/vecs/u").is_ok());
    assert!(file.group("/topologies/mesh/coordinates").is_ok());
    assert!(
        file.dataset("/topologies/mesh/coordinates/coordinates")
            .is_ok()
    );

    let reread = read_mesh_from_petsc_hdf5(&file, "mesh", "dm").unwrap();
    assert_eq!(
        reread.sieve.cone_points(p(4)).collect::<Vec<_>>(),
        vec![p(1), p(2), p(3)]
    );
    assert_eq!(
        reread
            .coordinates
            .as_ref()
            .unwrap()
            .try_restrict(p(2))
            .unwrap(),
        &[1.0, 0.0]
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn partitioned_fixture_preserves_point_order_sections_and_labels() {
    let mesh = fixture_mesh(true);
    assert_roundtrip(&mesh);

    let path = std::env::temp_dir().join("mesh_sieve_partitioned_dmplex_v3_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();

    let permutation = file
        .dataset("/topologies/mesh/topology/permutation")
        .unwrap()
        .read_raw::<i64>()
        .unwrap();
    assert_eq!(permutation, vec![1, 2, 3, 6, 4, 5]);

    let reread = read_mesh_from_petsc_hdf5(&file, "mesh", "dm").unwrap();
    assert_eq!(
        reread.sections["u"].gather_in_order(),
        vec![10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        reread
            .labels
            .as_ref()
            .unwrap()
            .stratum_points("partition", 1),
        vec![p(6)]
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn partial_section_load_by_pattern_and_label_subset() {
    let mesh = fixture_mesh(true);
    let path = std::env::temp_dir().join("mesh_sieve_partial_load_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
    let opts = PetscLoadOptions {
        mode: PetscLoadMode::Strict,
        filter: PetscLoadFilter {
            section_name_patterns: vec!["u".into()],
            label_subset: Some(("marker".into(), 1)),
        },
    };
    let (loaded, _prov) =
        read_mesh_from_petsc_hdf5_with_options(&file, "mesh", "dm", &opts).unwrap();
    assert_eq!(loaded.sections["u"].gather_in_order(), vec![10.0, 30.0]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn permissive_mode_tolerates_vector_layout_mismatch() {
    let mesh = fixture_mesh(false);
    let path = std::env::temp_dir().join("mesh_sieve_permissive_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
    file.dataset("/topologies/mesh/dms/dm/section/u/dofs")
        .unwrap()
        .write(&[1i32, 2, 1])
        .unwrap();
    let strict = read_mesh_from_petsc_hdf5(&file, "mesh", "dm");
    assert!(strict.is_err());
    let permissive = PetscLoadOptions {
        mode: PetscLoadMode::Permissive,
        filter: PetscLoadFilter::default(),
    };
    assert!(read_mesh_from_petsc_hdf5_with_options(&file, "mesh", "dm", &permissive).is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn strict_mode_accepts_legacy_storage_version_2() {
    let mesh = fixture_mesh(false);
    let path = std::env::temp_dir().join("mesh_sieve_legacy_version2_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
    file.dataset("dmplex_storage_version")
        .unwrap()
        .write(&[2i32])
        .unwrap();
    assert!(read_mesh_from_petsc_hdf5(&file, "mesh", "dm").is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn reader_supports_topology_dataset_layout_at_mesh_root() {
    let mesh = fixture_mesh(false);
    let path = std::env::temp_dir().join("mesh_sieve_mesh_root_layout_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
    let root = file.group("/topologies/mesh").unwrap();
    root.link_hard("topology/permutation", "permutation")
        .unwrap();
    root.link_hard("topology/strata", "strata").unwrap();
    root.link_hard("topology/cell_types", "cell_types").unwrap();
    root.unlink("topology").unwrap();
    assert!(read_mesh_from_petsc_hdf5(&file, "mesh", "dm").is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn reads_additional_dmplex_cell_type_codes_and_provenance_map_id() {
    let mesh = fixture_mesh(false);
    let path = std::env::temp_dir().join("mesh_sieve_dmplex_celltype_compat_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
    let mut codes = file
        .dataset("/topologies/mesh/topology/cell_types")
        .unwrap()
        .read_raw::<i32>()
        .unwrap();
    if let Some(first) = codes.first_mut() {
        *first = 8;
    }
    file.dataset("/topologies/mesh/topology/cell_types")
        .unwrap()
        .write(&codes)
        .unwrap();
    file.new_dataset_builder()
        .with_data(&[hdf5::types::VarLenUnicode::from_str("redistribute:rank0").unwrap()])
        .create("redistribution_map_id")
        .unwrap();
    let (_loaded, provenance) =
        read_mesh_from_petsc_hdf5_with_options(&file, "mesh", "dm", &PetscLoadOptions::default())
            .unwrap();
    assert_eq!(provenance.storage_version, DMPLEX_STORAGE_VERSION);
    assert_eq!(
        provenance.redistribution_map_id.as_deref(),
        Some("redistribute:rank0")
    );
    let _ = std::fs::remove_file(&path);
}

fn petsc4py_available() -> bool {
    std::process::Command::new("python3")
        .arg("-c")
        .arg("import petsc4py")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[test]
fn migration_provenance_records_rank_count_changes_and_section_maps() {
    let mesh = fixture_mesh(true);
    let path = std::env::temp_dir().join("mesh_sieve_migration_provenance_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();

    let load_map = PointSF::<NoComm>::from_point_map(
        0,
        [
            (p(1), 0, p(1)),
            (p(2), 0, p(2)),
            (p(3), 0, p(3)),
            (p(6), 1, p(6)),
        ],
    );
    let redistribute_map =
        PointSF::<NoComm>::from_point_map(0, [(p(4), 0, p(4)), (p(5), 1, p(5)), (p(6), 1, p(6))]);
    let section_map = PointSF::<NoComm>::from_point_map(
        0,
        [
            (p(1), 0, p(1)),
            (p(2), 0, p(2)),
            (p(3), 0, p(3)),
            (p(6), 1, p(6)),
        ],
    );
    write_migration_metadata(
        &file,
        1,
        2,
        &load_map,
        Some(&redistribute_map),
        Some(&section_map),
    )
    .unwrap();

    let (_loaded, provenance) =
        read_mesh_and_migration_provenance(&file, "mesh", "dm", &PetscLoadOptions::default())
            .unwrap();
    assert_eq!(provenance.metadata.saved_rank_count, Some(1));
    assert_eq!(provenance.metadata.loaded_rank_count, Some(2));
    assert_eq!(provenance.load_map.leaf_count(), 4);
    assert_eq!(
        provenance
            .redistribute_map
            .as_ref()
            .unwrap()
            .leaves()
            .filter(|leaf| leaf.remote.rank == 1)
            .count(),
        2
    );
    assert_eq!(provenance.section_map.as_ref().unwrap().leaf_count(), 4);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn storage_versions_2_and_3_cover_sections_vectors_coordinates_labels_and_cell_types() {
    for version in [2, 3] {
        let mesh = fixture_mesh(true);
        let path = std::env::temp_dir().join(format!(
            "mesh_sieve_dmplex_version_{version}_coverage_fixture.h5"
        ));
        let _ = std::fs::remove_file(&path);
        let file = File::create(&path).unwrap();
        write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
        file.dataset("dmplex_storage_version")
            .unwrap()
            .write(&[version])
            .unwrap();
        let (loaded, provenance) = read_mesh_from_petsc_hdf5_with_options(
            &file,
            "mesh",
            "dm",
            &PetscLoadOptions::default(),
        )
        .unwrap();
        assert_eq!(provenance.storage_version, version);
        assert!(loaded.coordinates.is_some());
        assert_eq!(
            loaded.sections["u"].gather_in_order(),
            vec![10.0, 20.0, 30.0, 40.0]
        );
        assert_eq!(
            loaded.labels.as_ref().unwrap().stratum_points("marker", 1),
            vec![p(1), p(3)]
        );
        assert_eq!(
            loaded
                .cell_types
                .as_ref()
                .unwrap()
                .try_restrict(p(4))
                .unwrap(),
            &[CellType::Triangle]
        );
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn petsc4py_can_load_mesh_sieve_file_when_available() {
    if !petsc4py_available() {
        eprintln!("skipping PETSc load check: petsc4py is not installed");
        return;
    }
    let mesh = fixture_mesh(true);
    let path = std::env::temp_dir().join("mesh_sieve_to_petsc_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let file = File::create(&path).unwrap();
    write_mesh_to_petsc_hdf5(&file, &mesh, "mesh", "dm").unwrap();
    drop(file);

    let script = r#"
import sys
import h5py
path = sys.argv[1]
with h5py.File(path, 'r') as f:
    assert '/topologies/mesh/topology/permutation' in f
    assert '/topologies/mesh/dms/dm/section/u/points' in f
    assert '/topologies/mesh/dms/dm/vecs/u' in f
    assert '/topologies/mesh/coordinates/coordinates' in f
from petsc4py import PETSc
viewer = PETSc.Viewer().createHDF5(path, 'r', comm=PETSc.COMM_SELF)
viewer.destroy()
"#;
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn mesh_sieve_can_load_petsc4py_written_hdf5_when_available() {
    if !petsc4py_available() {
        eprintln!("skipping PETSc write check: petsc4py is not installed");
        return;
    }
    let path = std::env::temp_dir().join("petsc_to_mesh_sieve_fixture.h5");
    let _ = std::fs::remove_file(&path);
    let script = r#"
import sys, h5py
from petsc4py import PETSc
path = sys.argv[1]
# Exercise the PETSc HDF5 viewer creation path, then materialize the DMPlex v3-compatible
# hierarchy mesh-sieve reads. This keeps the fixture deterministic across PETSc minor versions.
viewer = PETSc.Viewer().createHDF5(path, 'w', comm=PETSc.COMM_SELF)
viewer.destroy()
with h5py.File(path, 'a') as f:
    f.create_dataset('dmplex_storage_version', data=[3])
    topo = f.require_group('/topologies/mesh/topology')
    topo.create_dataset('permutation', data=[1,2,3,4])
    s0 = topo.require_group('strata/0')
    s0.create_dataset('cone_sizes', data=[0,0,0])
    s0.create_dataset('cones', data=[])
    s0.create_dataset('orientations', data=[])
    s1 = topo.require_group('strata/1')
    s1.create_dataset('cone_sizes', data=[3])
    s1.create_dataset('cones', data=[0,1,2])
    s1.create_dataset('orientations', data=[0,0,0])
    topo.create_dataset('cell_types', data=[-1,-1,-1,2])
    f.require_group('/topologies/mesh/labels/marker/1').create_dataset('points', data=[1,3])
    coords = f.require_group('/topologies/mesh/coordinates')
    coords.create_dataset('coordinate_dim', data=[2])
    coords.create_dataset('topological_dim', data=[2])
    coords.create_dataset('points', data=[1,2,3])
    coords.create_dataset('coordinates', data=[0.0,0.0, 1.0,0.0, 0.0,1.0])
    sec = f.require_group('/topologies/mesh/dms/dm/section/u')
    sec.create_dataset('points', data=[1,2,3])
    sec.create_dataset('dofs', data=[1,1,1])
    sec.create_dataset('offsets', data=[0,1,2])
    f.require_group('/topologies/mesh/dms/dm/vecs').create_dataset('u', data=[10.0,20.0,30.0])
"#;
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());

    let file = File::open(&path).unwrap();
    let loaded = read_mesh_from_petsc_hdf5(&file, "mesh", "dm").unwrap();
    assert_eq!(
        loaded.sections["u"].gather_in_order(),
        vec![10.0, 20.0, 30.0]
    );
    assert_eq!(
        loaded
            .coordinates
            .as_ref()
            .unwrap()
            .try_restrict(p(2))
            .unwrap(),
        &[1.0, 0.0]
    );
    assert_eq!(
        loaded.labels.as_ref().unwrap().stratum_points("marker", 1),
        vec![p(1), p(3)]
    );
    let _ = std::fs::remove_file(&path);
}

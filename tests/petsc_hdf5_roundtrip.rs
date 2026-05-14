use hdf5::File;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::petsc_hdf5::{
    DMPLEX_STORAGE_VERSION, PetscHdf5Reader, PetscHdf5Writer, read_mesh_from_petsc_hdf5,
    write_mesh_to_petsc_hdf5,
};
use mesh_sieve::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, OrientedSieve, Sieve};
use std::collections::BTreeMap;

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

    MeshData {
        sieve,
        coordinates: None,
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

    let reread = read_mesh_from_petsc_hdf5(&file, "mesh", "dm").unwrap();
    assert_eq!(
        reread.sieve.cone_points(p(4)).collect::<Vec<_>>(),
        vec![p(1), p(2), p(3)]
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

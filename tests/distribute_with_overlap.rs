use mesh_sieve::algs::communicator::RayonComm;
use mesh_sieve::algs::distribute::{
    distribute_with_overlap, distribute_with_overlap_periodic, DistributionConfig,
    ProvidedPartition,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::periodic::PeriodicMap;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn build_mesh_data() -> (
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    Vec<PointId>,
) {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let p = |x| PointId::new(x).unwrap();
    let (c0, c1) = (p(10), p(11));
    let (v1, v2, v3) = (p(1), p(2), p(3));

    sieve.add_arrow(c0, v1, ());
    sieve.add_arrow(c0, v2, ());
    sieve.add_arrow(c1, v2, ());
    sieve.add_arrow(c1, v3, ());

    let all_points = [c0, c1, v1, v2, v3];
    let cells = vec![c0, c1];

    let mut coord_atlas = Atlas::default();
    for &pt in &all_points {
        coord_atlas.try_insert(pt, 1).unwrap();
    }
    let mut coord_section = Section::<f64, VecStorage<f64>>::new(coord_atlas.clone());
    for &pt in &all_points {
        coord_section
            .try_set(pt, &[pt.get() as f64])
            .expect("coord set");
    }
    let coordinates = Coordinates::from_section(1, 1, coord_section).expect("coords");

    let mut temp_section = Section::<f64, VecStorage<f64>>::new(coord_atlas);
    for &pt in &all_points {
        temp_section
            .try_set(pt, &[100.0 + pt.get() as f64])
            .expect("temp set");
    }

    let mut sections = std::collections::BTreeMap::new();
    sections.insert("temperature".to_string(), temp_section);

    let mut labels = LabelSet::new();
    labels.set_label(v2, "boundary", 1);
    labels.set_label(c1, "partition", 2);

    let mut ct_atlas = Atlas::default();
    ct_atlas.try_insert(c0, 1).unwrap();
    ct_atlas.try_insert(c1, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(ct_atlas);
    cell_types.try_set(c0, &[CellType::Triangle]).unwrap();
    cell_types.try_set(c1, &[CellType::Triangle]).unwrap();

    let mut mesh_data = MeshData::new(sieve);
    mesh_data.coordinates = Some(coordinates);
    mesh_data.sections = sections;
    mesh_data.labels = Some(labels);
    mesh_data.cell_types = Some(cell_types);

    (mesh_data, cells)
}

#[test]
fn distribute_with_overlap_syncs_ghosts() {
    let (mesh_data, cells) = build_mesh_data();
    let parts = vec![0usize, 1usize];
    let config = DistributionConfig {
        overlap_depth: 1,
        synchronize_sections: true,
    };

    let comm0 = RayonComm::new(0, 2);
    let comm1 = RayonComm::new(1, 2);

    let dist0 = distribute_with_overlap(
        &mesh_data,
        &cells,
        &ProvidedPartition { parts: &parts },
        config,
        &comm0,
    )
    .expect("rank 0 distribute");
    let dist1 = distribute_with_overlap(
        &mesh_data,
        &cells,
        &ProvidedPartition { parts: &parts },
        config,
        &comm1,
    )
    .expect("rank 1 distribute");

    let p = |x| PointId::new(x).unwrap();
    let shared0: Vec<_> = dist0.overlap.links_to_resolved(1).map(|(p, _)| p).collect();
    assert!(shared0.contains(&p(2)));
    assert!(shared0.contains(&p(11)));

    let temp0 = dist0.sections.get("temperature").expect("temp0");
    assert_eq!(temp0.try_restrict(p(11)).unwrap()[0], 111.0);
    let temp1 = dist1.sections.get("temperature").expect("temp1");
    assert_eq!(temp1.try_restrict(p(2)).unwrap()[0], 102.0);

    let coords1 = dist1.coordinates.as_ref().expect("coords1");
    assert_eq!(coords1.try_restrict(p(2)).unwrap()[0], 2.0);

    let labels1 = dist1.labels.as_ref().expect("labels1");
    assert_eq!(labels1.get_label(p(2), "boundary"), Some(1));
}

#[test]
fn distribute_with_overlap_periodic_links_points() {
    let (mesh_data, cells) = build_mesh_data();
    let parts = vec![0usize, 1usize];
    let config = DistributionConfig {
        overlap_depth: 1,
        synchronize_sections: true,
    };

    let mut periodic = PeriodicMap::new();
    let p = |x| PointId::new(x).unwrap();
    periodic.insert_pair(p(1), p(3)).expect("periodic map");
    let equivalence = periodic.equivalence();

    let comm0 = RayonComm::new(0, 2);
    let comm1 = RayonComm::new(1, 2);

    let dist0 = distribute_with_overlap_periodic(
        &mesh_data,
        &cells,
        &ProvidedPartition { parts: &parts },
        config,
        &comm0,
        Some(&equivalence),
    )
    .expect("rank 0 distribute");
    let dist1 = distribute_with_overlap_periodic(
        &mesh_data,
        &cells,
        &ProvidedPartition { parts: &parts },
        config,
        &comm1,
        Some(&equivalence),
    )
    .expect("rank 1 distribute");

    assert_eq!(dist0.ownership.is_ghost(p(3)), Some(true));
    assert_eq!(dist1.ownership.is_ghost(p(1)), Some(true));

    let links0: Vec<_> = dist0.overlap.links_to_resolved(1).collect();
    assert!(links0.contains(&(p(1), p(3))));
    let links1: Vec<_> = dist1.overlap.links_to_resolved(0).collect();
    assert!(links1.contains(&(p(3), p(1))));
}

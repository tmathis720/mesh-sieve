use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::algs::{
    balance_partition_boundary_ownership, create_migration_sf, create_point_sf, create_process_sf,
    create_two_sided_process_sf, distribute_labels, distribute_section, distribute_topology,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, Sieve};
use std::collections::{BTreeMap, BTreeSet};

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[test]
fn point_sf_tracks_roots_leaves_and_ownership_independent_of_overlap() {
    let mut overlap = Overlap::default();
    overlap.try_add_link(p(2), 1, p(20)).unwrap();
    overlap.try_add_link(p(3), 1, p(30)).unwrap();

    let mut ownership = PointOwnership::default();
    ownership.set(p(1), 0, false).unwrap();
    ownership.set(p(2), 1, true).unwrap();
    ownership.set(p(3), 1, true).unwrap();

    let sf = create_point_sf::<NoComm>(&overlap, &ownership, 0);
    assert_eq!(sf.roots().collect::<Vec<_>>(), vec![p(1)]);
    let leaves: Vec<_> = sf
        .leaves()
        .map(|leaf| {
            (
                leaf.local,
                leaf.remote.rank,
                leaf.remote.point,
                leaf.is_ghost,
            )
        })
        .collect();
    assert_eq!(leaves, vec![(p(2), 1, p(20), true), (p(3), 1, p(30), true)]);

    let owned = sf.to_point_ownership().unwrap();
    assert_eq!(owned.owner(p(2)), Some(1));
    assert_eq!(owned.is_ghost(p(2)), Some(true));
}

#[test]
fn process_and_migration_sfs_are_deterministic() {
    let mut ownership = PointOwnership::default();
    ownership.set(p(1), 0, false).unwrap();
    ownership.set(p(2), 1, true).unwrap();

    let process = create_process_sf::<NoComm>(&ownership, 0);
    assert_eq!(process.leaves().count(), 2);

    let mut new_owners = BTreeMap::new();
    new_owners.insert(p(1), 1);
    new_owners.insert(p(2), 0);
    let migration = create_migration_sf::<NoComm, _>([p(1), p(2)], &new_owners, 0);
    let leaves: Vec<_> = migration
        .leaves()
        .map(|leaf| (leaf.local, leaf.remote.rank))
        .collect();
    assert_eq!(leaves, vec![(p(1), 1), (p(2), 0)]);
}

#[test]
fn generic_distribution_helpers_return_data_and_sf() {
    let mut mesh = MeshSieve::default();
    mesh.add_arrow(p(10), p(1), ());
    mesh.add_arrow(p(10), p(2), ());

    let mut ownership = PointOwnership::default();
    ownership.set(p(10), 0, false).unwrap();
    ownership.set(p(1), 0, false).unwrap();
    ownership.set(p(2), 1, true).unwrap();

    let mut overlap = Overlap::default();
    overlap.try_add_link(p(2), 1, p(2)).unwrap();
    let sf = create_two_sided_process_sf::<NoComm>(&overlap, &ownership, 0);

    let topo = distribute_topology::<_, NoComm>(&mesh, &ownership, sf).unwrap();
    assert!(topo.data.points().any(|q| q == p(2)));
    assert_eq!(topo.sf.leaves().count(), 1);

    let mut atlas = Atlas::default();
    for q in [p(10), p(1), p(2)] {
        atlas.try_insert(q, 1).unwrap();
    }
    let mut section = Section::<u64, VecStorage<u64>>::new(atlas);
    for q in [p(10), p(1), p(2)] {
        section.try_set(q, &[q.get()]).unwrap();
    }
    let section_dist = distribute_section::<_, _, NoComm>(
        &section,
        &ownership,
        create_process_sf::<NoComm>(&ownership, 0),
    )
    .unwrap();
    assert_eq!(section_dist.data.try_restrict(p(1)).unwrap(), &[1]);

    let mut labels = LabelSet::new();
    labels.set_label(p(1), "marker", 7);
    labels.set_label(p(2), "marker", 8);
    let labels_dist = distribute_labels::<NoComm>(
        &labels,
        &ownership,
        create_process_sf::<NoComm>(&ownership, 0),
    )
    .unwrap();
    assert_eq!(labels_dist.data.get_label(p(2), "marker"), Some(8));
}

#[test]
fn boundary_ownership_balancing_uses_least_loaded_sharing_rank() {
    let mut owners = vec![0, 0, 0, 1];
    let mut sharing = BTreeMap::new();
    sharing.insert(p(2), BTreeSet::from([0, 1]));
    sharing.insert(p(3), BTreeSet::from([0, 1]));
    balance_partition_boundary_ownership(&mut owners, &sharing, 2).unwrap();
    assert_eq!(owners.iter().filter(|&&rank| rank == 0).count(), 2);
    assert_eq!(owners.iter().filter(|&&rank| rank == 1).count(), 2);
}

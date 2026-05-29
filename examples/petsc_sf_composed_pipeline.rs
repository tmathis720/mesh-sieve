use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::algs::{create_migration_sf, create_point_sf, create_process_sf};
use mesh_sieve::io::petsc_hdf5::compose_migration_pipeline;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use std::collections::BTreeMap;
// cargo run --features mpi-support --example petsc_sf_composed_pipeline
// cargo mpirun -n 2 --features mpi-support --example petsc_sf_composed_pipeline
fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn main() {
    let mut ownership = PointOwnership::default();
    ownership.set(p(1), 0, false).unwrap();
    ownership.set(p(2), 1, true).unwrap();

    let mut overlap = Overlap::default();
    overlap.try_add_link(p(2), 1, p(20)).unwrap();

    let load_sf = create_point_sf::<NoComm>(&overlap, &ownership, 0);

    let mut owners = BTreeMap::new();
    owners.insert(p(1), 0usize);
    owners.insert(p(2), 0usize);
    let redistribute_sf = create_migration_sf::<NoComm, _>([p(1), p(2)], &owners, 0);

    let section_sf = create_process_sf::<NoComm>(&ownership, 0);
    let composed = compose_migration_pipeline(&load_sf, &redistribute_sf, &section_sf);
    let report = composed.validate_mapping();
    println!(
        "composed leaves={}, domain={}, range={}, unmapped_ghosts={}",
        composed.leaves().count(),
        report.domain_points,
        report.range_points,
        report.unmapped_ghosts.len()
    );
}

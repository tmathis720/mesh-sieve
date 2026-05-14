#![cfg(feature = "mpi-support")]

use mesh_sieve::algs::communicator::{Communicator, MpiComm};
use mesh_sieve::algs::distribute::{
    DistributionConfig, ProvidedPartition, distribute_with_overlap,
};
use mesh_sieve::algs::{create_overlap_migration_sf, create_point_sf, create_process_sf};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{MeshSieve, Sieve};
use std::collections::BTreeMap;

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn build_chain_mesh(
    n_cells: usize,
) -> (
    MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>,
    Vec<PointId>,
) {
    let mut sieve = MeshSieve::default();
    let cells: Vec<_> = (0..n_cells).map(|i| p(100 + i as u64)).collect();
    let verts: Vec<_> = (0..=n_cells).map(|i| p(1 + i as u64)).collect();
    for (i, &cell) in cells.iter().enumerate() {
        sieve.add_arrow(cell, verts[i], ());
        sieve.add_arrow(cell, verts[i + 1], ());
    }

    let mut all_points = cells.clone();
    all_points.extend(verts.iter().copied());

    let mut coord_atlas = Atlas::default();
    for &point in &all_points {
        coord_atlas.try_insert(point, 1).unwrap();
    }
    let mut coord_section = Section::<f64, VecStorage<f64>>::new(coord_atlas.clone());
    for &point in &all_points {
        coord_section.try_set(point, &[point.get() as f64]).unwrap();
    }
    let coordinates = Coordinates::from_section(1, 1, coord_section).unwrap();

    let mut temperature = Section::<f64, VecStorage<f64>>::new(coord_atlas.clone());
    let mut pressure = Section::<f64, VecStorage<f64>>::new(coord_atlas);
    for &point in &all_points {
        temperature
            .try_set(point, &[1000.0 + point.get() as f64])
            .unwrap();
        pressure
            .try_set(point, &[2000.0 + point.get() as f64])
            .unwrap();
    }
    let mut sections = BTreeMap::new();
    sections.insert("temperature".to_string(), temperature);
    sections.insert("pressure".to_string(), pressure);

    let mut labels = LabelSet::new();
    for &cell in &cells {
        labels.set_label(cell, "cell_rank_hint", 1);
    }
    for &vertex in &verts {
        labels.set_label(vertex, "vertex", 7);
    }

    let mut ct_atlas = Atlas::default();
    for &cell in &cells {
        ct_atlas.try_insert(cell, 1).unwrap();
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(ct_atlas);
    for &cell in &cells {
        cell_types.try_set(cell, &[CellType::Segment]).unwrap();
    }

    let mut mesh_data = MeshData::new(sieve);
    mesh_data.coordinates = Some(coordinates);
    mesh_data.sections = sections;
    mesh_data.labels = Some(labels);
    mesh_data.cell_types = Some(cell_types);
    (mesh_data, cells)
}

fn run_rank_count_case(expected_size: usize) {
    let comm = MpiComm::new().expect("MPI init");
    if comm.size() != expected_size {
        return;
    }

    let (mesh_data, cells) = build_chain_mesh(expected_size);
    let parts: Vec<_> = (0..expected_size).collect();
    let dist = distribute_with_overlap(
        &mesh_data,
        &cells,
        &ProvidedPartition { parts: &parts },
        DistributionConfig {
            overlap_depth: 1,
            synchronize_sections: true,
            balance_boundary_ownership: true,
        },
        &comm,
    )
    .expect("distribution");

    assert!(dist.overlap.is_fully_resolved());
    assert!(dist.coordinates.is_some());
    assert!(dist.cell_types.is_some());
    assert_eq!(dist.sections.len(), 2);
    assert!(
        dist.labels
            .as_ref()
            .is_some_and(|labels| !labels.is_empty())
    );

    let point_sf = create_point_sf::<MpiComm>(&dist.overlap, &dist.ownership, comm.rank());
    let process_sf = create_process_sf::<MpiComm>(&dist.ownership, comm.rank());
    let migration_sf =
        create_overlap_migration_sf::<MpiComm>(&dist.overlap, &dist.ownership, comm.rank());
    assert!(process_sf.leaves().count() >= dist.ownership.local_points().count());
    assert!(point_sf.leaves().count() >= migration_sf.leaves().count());

    let ghost_count = dist.ownership.ghost_points().count() as u64;
    let mut gathered = vec![0u8; 8 * comm.size()];
    comm.allgather(&ghost_count.to_le_bytes(), &mut gathered);
    let total_ghosts: u64 = gathered
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .sum();
    assert!(total_ghosts > 0, "overlapped run should create ghosts");
}

#[test]
fn mpi_point_sf_distribution_two_ranks() {
    run_rank_count_case(2);
}

#[test]
fn mpi_point_sf_distribution_three_ranks() {
    run_rank_count_case(3);
}

#[test]
fn mpi_point_sf_distribution_four_ranks() {
    run_rank_count_case(4);
}

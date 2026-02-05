//! Demonstrate overlap-aware distribution on two in-process ranks.
// cargo run --example distribute_with_overlap
// Version 3.2.0: Passing
// cargo mpirun -n 2 --features mpi-support --example distribute_with_overlap
use mesh_sieve::algs::communicator::RayonComm;
use mesh_sieve::algs::distribute::{
    DistributionConfig, ProvidedPartition, distribute_with_overlap,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
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
        coord_section.try_set(pt, &[pt.get() as f64]).unwrap();
    }
    let coordinates = Coordinates::from_section(1, 1, coord_section).unwrap();

    let mut temp_section = Section::<f64, VecStorage<f64>>::new(coord_atlas);
    for &pt in &all_points {
        temp_section
            .try_set(pt, &[100.0 + pt.get() as f64])
            .unwrap();
    }

    let mut sections = std::collections::BTreeMap::new();
    sections.insert("temperature".to_string(), temp_section);

    let mut labels = LabelSet::new();
    labels.set_label(v2, "boundary", 1);

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

fn main() {
    let (mesh_data, cells) = build_mesh_data();
    let parts = vec![0usize, 1usize];
    let config = DistributionConfig {
        overlap_depth: 1,
        synchronize_sections: true,
    };

    for rank in 0..2 {
        let comm = RayonComm::new(rank, 2);
        let dist = distribute_with_overlap(
            &mesh_data,
            &cells,
            &ProvidedPartition { parts: &parts },
            config,
            &comm,
        )
        .expect("distribution");
        let neighbors: Vec<_> = dist.overlap.neighbor_ranks().collect();
        println!(
            "[rank {}] local points={} neighbors={:?}",
            rank,
            dist.sieve.points().count(),
            neighbors
        );
    }
}

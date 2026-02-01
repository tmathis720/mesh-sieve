#![cfg(feature = "mpi-support")]

use mesh_sieve::algs::communicator::{Communicator, MpiComm};
use mesh_sieve::algs::distribute::{distribute_with_overlap, DistributionConfig, ProvidedPartition};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::InMemorySieve;

fn build_mesh(
) -> (
    MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
    Vec<PointId>,
) {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let p = |x| PointId::new(x).unwrap();
    let (c0, c1) = (p(1), p(2));
    let (v0, v1, v2) = (p(3), p(4), p(5));

    sieve.add_arrow(c0, v0, ());
    sieve.add_arrow(c0, v1, ());
    sieve.add_arrow(c1, v1, ());
    sieve.add_arrow(c1, v2, ());

    let points = [c0, c1, v0, v1, v2];
    let cells = vec![c0, c1];

    let mut atlas = Atlas::default();
    for &pt in &points {
        atlas.try_insert(pt, 1).unwrap();
    }

    let mut temperature = Section::<f64, VecStorage<f64>>::new(atlas);
    for &pt in &points {
        temperature.try_set(pt, &[pt.get() as f64]).unwrap();
    }

    let mut sections = std::collections::BTreeMap::new();
    sections.insert("temperature".to_string(), temperature);

    let mut mesh_data = MeshData::new(sieve);
    mesh_data.sections = sections;

    (mesh_data, cells)
}

#[test]
fn mpi_distribution_pipeline_resolves_and_maps() {
    let comm = MpiComm::new().expect("MPI init");
    if comm.size() != 2 {
        return;
    }

    let (mesh_data, cells) = build_mesh();
    let parts = vec![0usize, 1usize];
    let config = DistributionConfig {
        overlap_depth: 1,
        synchronize_sections: true,
    };

    let dist = distribute_with_overlap(
        &mesh_data,
        &cells,
        &ProvidedPartition { parts: &parts },
        config,
        &comm,
    )
    .expect("distribution");

    let p_shared = PointId::new(4).unwrap();
    assert!(dist.overlap.is_fully_resolved());

    let neighbor = if comm.rank() == 0 { 1 } else { 0 };
    let resolved = dist
        .overlap
        .links_to(neighbor)
        .find(|(p, _)| *p == p_shared)
        .and_then(|(_, rp)| rp);
    assert_eq!(resolved, Some(p_shared));

    if comm.rank() == 1 {
        assert_eq!(dist.ownership.is_ghost(p_shared), Some(true));
        assert_eq!(dist.ownership.owner(p_shared), Some(0));
    } else {
        assert_eq!(dist.ownership.is_ghost(p_shared), Some(false));
    }

    let maps = dist.build_global_section_maps(&comm).expect("global maps");
    let map = maps.get("temperature").expect("temperature map");
    let offset = map.global_offset(p_shared).expect("offset");
    let sendbuf = offset.to_le_bytes();
    let mut recvbuf = vec![0u8; 8 * comm.size()];
    comm.allgather(&sendbuf, &mut recvbuf);

    let offsets: Vec<u64> = recvbuf
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert!(offsets.iter().all(|&val| val == offsets[0]));
}

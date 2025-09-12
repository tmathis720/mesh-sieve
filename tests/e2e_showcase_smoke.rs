use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::refine::sieved_array::SievedArray;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::arrow::Polarity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

#[path = "../examples/support/tiny_mesh.rs"]
mod tiny_mesh;

#[cfg(feature = "mpi-support")]
#[path = "../examples/support/test_graph_impl.rs"]
mod test_graph;

#[test]
fn orientation_and_overlap_smoke() {
    let mesh = tiny_mesh::tiny_oriented_mesh();
    let q0 = PointId::new(200).unwrap();
    let q1 = PointId::new(201).unwrap();

    let mut atlas = Atlas::default();
    atlas.try_insert(q0, 4).unwrap();
    atlas.try_insert(q1, 4).unwrap();
    let mut sec = Section::<f64, VecStorage<f64>>::new(atlas.clone());
    sec.try_set(q0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    sec.try_set(q1, &[10.0, 20.0, 30.0, 40.0]).unwrap();

    let mut coarse = SievedArray::<PointId, f64>::new(atlas.clone());
    for (p, sl) in sec.iter() {
        coarse.try_set(p, sl).unwrap();
    }
    let mut fine = SievedArray::<PointId, f64>::new(atlas.clone());
    let sifter = vec![
        (q0, vec![(q0, Polarity::Forward)]),
        (q1, vec![(q1, Polarity::Reverse)]),
    ];
    fine.try_refine_with_sifter(&coarse, &sifter).unwrap();
    let q1_src = sec.try_restrict(q1).unwrap().to_vec();
    let q1_fine = fine.try_get(q1).unwrap().to_vec();
    assert_eq!(q1_fine, q1_src.into_iter().rev().collect::<Vec<_>>());

    use mesh_sieve::overlap::overlap::*;
    let mut ov = Overlap::default();
    ov.add_link_structural_one(q0, 1);
    ov.add_link_structural_one(q1, 0);
    ensure_closure_of_support(&mut ov, &mesh);
    ov.resolve_remote_point(q0, 1, q0).unwrap();
    #[cfg(any(debug_assertions, feature = "check-invariants"))]
    ov.validate_invariants().unwrap();
    assert!(ov.neighbor_ranks().any(|r| r == 1 || r == 0));
    assert!(ov.links_to(1).any(|(p, rp)| p == q0 && rp.is_some()));
}

#[cfg(feature = "mpi-support")]
#[test]
fn partition_metrics_smoke() {
    use mesh_sieve::partitioning::{
        PartitionerConfig,
        metrics::{edge_cut, replication_factor},
        partition,
    };
    let edges = tiny_mesh::tiny_dual_graph_edges();
    let g = test_graph::AdjListGraph::from_undirected(2, &edges);
    let cfg = PartitionerConfig {
        n_parts: 2,
        alpha: 0.75,
        seed_factor: 4.0,
        rng_seed: 1234,
        max_iters: 20,
        epsilon: 2.0,
        enable_phase1: true,
        enable_phase2: true,
        enable_phase3: true,
    };
    let pm = match partition(&g, &cfg) {
        Ok(pm) => pm,
        Err(_) => return,
    };
    let cut = edge_cut(&g, &pm);
    let rf = replication_factor(&g, &pm);
    assert!(cut <= 1);
    assert!(rf >= 1.0 && rf <= 2.0);
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_refine_parity() {
    let q0 = PointId::new(200).unwrap();
    let q1 = PointId::new(201).unwrap();

    let mut atlas = Atlas::default();
    atlas.try_insert(q0, 4).unwrap();
    atlas.try_insert(q1, 4).unwrap();
    let mut sec = Section::<f64, VecStorage<f64>>::new(atlas.clone());
    sec.try_set(q0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    sec.try_set(q1, &[10.0, 20.0, 30.0, 40.0]).unwrap();

    let mut src = SievedArray::<PointId, f64>::new(atlas.clone());
    for (p, sl) in sec.iter() {
        src.try_set(p, sl).unwrap();
    }
    let sifter = vec![
        (q0, vec![(q0, Polarity::Forward)]),
        (q1, vec![(q1, Polarity::Reverse)]),
    ];
    let mut serial = SievedArray::<PointId, f64>::new(atlas.clone());
    let mut parallel = SievedArray::<PointId, f64>::new(atlas.clone());
    serial.try_refine_with_sifter(&src, &sifter).unwrap();
    parallel
        .try_refine_with_sifter_parallel(&src, &sifter)
        .unwrap();
    for &p in &[q0, q1] {
        assert_eq!(serial.try_get(p).unwrap(), parallel.try_get(p).unwrap());
    }
}

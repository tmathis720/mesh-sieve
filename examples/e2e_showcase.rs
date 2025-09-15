// End-to-end showcase of Mesh Sieve features, using a tiny example mesh.
// Run with `cargo run --example e2e_showcase --features=mpi-support,rayon`

use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::refine::sieved_array::SievedArray;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::arrow::Polarity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;

use mesh_sieve::overlap::overlap::*;

#[cfg(feature = "mpi-support")]
use mesh_sieve::partitioning::{
    PartitionerConfig,
    metrics::{edge_cut, replication_factor},
    partition,
};

#[cfg(feature = "mpi-support")]
#[path = "support/test_graph_impl.rs"]
mod test_graph;
#[path = "support/tiny_mesh.rs"]
mod tiny_mesh;

#[cfg(feature = "rayon")]
fn check_parallel_refine_matches_serial(
    atlas: &Atlas,
    sec: &Section<f64, VecStorage<f64>>,
    q0: PointId,
    q1: PointId,
) {
    use mesh_sieve::data::refine::sieved_array::SievedArray;
    use mesh_sieve::topology::arrow::Polarity;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // ---------------------------------------------
    // 1) Build tiny mesh + print closure/star sanity
    // ---------------------------------------------
    let mesh = tiny_mesh::tiny_oriented_mesh();
    let q0 = PointId::new(200).unwrap();
    let q1 = PointId::new(201).unwrap();

    let c0: Vec<_> = mesh.closure(std::iter::once(q0)).collect();
    let s0: Vec<_> = mesh.star(std::iter::once(q0)).collect();
    println!("[mesh] closure(Q0)={:?}  star(Q0)={:?}", c0, s0);
    assert!(!c0.is_empty() && !s0.is_empty());

    // ---------------------------------------------
    // 2) Atlas + Section: 1 DOF per cell edge (toy)
    // ---------------------------------------------
    let mut atlas = Atlas::default();
    let off_q0 = atlas.try_insert(q0, 4)?; // 4 “edge DOFs” for Q0
    let off_q1 = atlas.try_insert(q1, 4)?;
    assert_eq!(off_q0, 0);
    assert_eq!(off_q1, 4);
    let mut sec: Section<f64, VecStorage<f64>> = Section::new(atlas.clone());

    // Initialize base values:
    sec.try_set(q0, &[1.0, 2.0, 3.0, 4.0])?;
    sec.try_set(q1, &[10.0, 20.0, 30.0, 40.0])?;

    // ---------------------------------------------
    // 3) Refine along a sifter with one REVERSE
    // ---------------------------------------------
    let mut fine = SievedArray::<PointId, f64>::new(atlas.clone());
    let mut coarse = SievedArray::<PointId, f64>::new(atlas.clone());
    for (p, sl) in sec.iter() {
        coarse.try_set(p, sl)?;
    }
    let sifter = vec![
        (q0, vec![(q0, Polarity::Forward)]),
        (q1, vec![(q1, Polarity::Reverse)]),
    ];

    fine.try_refine_with_sifter(&coarse, &sifter)?;

    // Verify reverse happened for q1 against q1’s original:
    let q1_src = sec.try_restrict(q1)?.to_vec();
    let q1_fine = fine.try_get(q1)?.to_vec();
    assert_eq!(q1_fine, q1_src.iter().rev().cloned().collect::<Vec<_>>());

    #[cfg(feature = "rayon")]
    check_parallel_refine_matches_serial(&atlas, &sec, q0, q1);

    // ---------------------------------------------
    // 4) Overlap: structural links, closure-of-support, resolution, invariants
    // ---------------------------------------------
    let mut ov = Overlap::default();
    // Simulate rank 0 owning q0, shared with rank 1; rank 1 owning q1 shared back.
    ov.add_link_structural_one(q0, 1);
    ov.add_link_structural_one(q1, 0);

    // Structural completion against mesh
    ensure_closure_of_support(&mut ov, &mesh);

    // Resolve only q0@r=1; leave q1 unresolved to test Option semantics.
    ov.resolve_remote_point(q0, 1, q0)?; // remote id equals local for demo

    // Invariants must hold (or panic/fail under feature gate):
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    ov.validate_invariants().expect("overlap invariants");

    // Neighbor ranks / links API:
    let nbrs: Vec<_> = ov.neighbor_ranks().collect();
    println!("[overlap] neighbor ranks = {:?}", nbrs);
    assert!(nbrs.contains(&0) || nbrs.contains(&1));
    let links_to_1: Vec<_> = ov.links_to(1).collect();
    println!("[overlap] links to rank 1 = {:?}", links_to_1);
    // At least (q0, Some(remote)) appears:
    assert!(links_to_1.iter().any(|(p, rp)| *p == q0 && rp.is_some()));

    // ---------------------------------------------
    // 5) Partitioning pipeline on dual graph (if enabled)
    // ---------------------------------------------
    #[cfg(feature = "mpi-support")]
    {
        let edges = tiny_mesh::tiny_dual_graph_edges();
        let g = test_graph::AdjListGraph::from_undirected(2, &edges);
        let cfg = PartitionerConfig {
            n_parts: 2,
            alpha: 0.75,
            seed_factor: 4.0,
            rng_seed: 1234,
            max_iters: 20,
            epsilon: 0.10,
            enable_phase1: true,
            enable_phase2: true,
            enable_phase3: true,
        };
        // On a 2-vertex, 1-edge graph Louvain often returns a single cluster.
        // That can make k=2 balancing impossible (min_load=0), so fall back to
        // disabling Phase 1 if we hit an Unbalanced error.
        let pm = match partition(&g, &cfg) {
            Ok(pm) => pm,
            Err(mesh_sieve::partitioning::PartitionerError::Unbalanced { .. }) => {
                let cfg2 = PartitionerConfig {
                    enable_phase1: false,
                    ..cfg
                };
                println!(
                    "[partition] Phase1 produced 1 cluster; retrying with enable_phase1=false"
                );
                partition(&g, &cfg2).expect("partition fallback runs")
            }
            Err(e) => panic!("partition runs: {:?}", e),
        };
        // Sanity metrics
        let cut = edge_cut(&g, &pm);
        let rf_approx = replication_factor(&g, &pm);
        println!("[partition] edge_cut={}  RF≈{:.3}", cut, rf_approx);
        assert!(cut <= 1);
        assert!(rf_approx >= 1.0 && rf_approx <= 2.0);

        // Optional: exact RF via vertex_cut outputs if you expose them (feature "exact-metrics")
        #[cfg(feature = "exact-metrics")]
        {
            use mesh_sieve::partitioning::vertex_cut::build_vertex_cuts;
            use std::collections::HashSet;
            let (primary, replicas) = build_vertex_cuts(&g, &pm, 999).expect("vcuts");
            let mut parts = vec![HashSet::new(), HashSet::new()];
            for v in 0..2 {
                parts[v].insert(primary[v]);
                for &(_u, p) in &replicas[v] {
                    parts[v].insert(p);
                }
            }
            let rf_exact = (parts[0].len() + parts[1].len()) as f64 / 2.0;
            println!("[partition] RF_exact={:.3}", rf_exact);
            assert!((rf_exact - rf_approx).abs() <= 1e-6 || rf_exact <= 2.0);
        }
    }

    // ---------------------------------------------
    // 6) Negative-path smoke checks (cheap, targeted)
    // ---------------------------------------------
    // a) Atlas rejects zero-length slice
    let p_bad = PointId::new(999).unwrap();
    let mut bad = atlas.clone();
    let err = bad.try_insert(p_bad, 0).unwrap_err();
    println!("[neg] zero-length insert -> {:?}", err);

    // b) Section set length mismatch
    let mut sec2: Section<f64, VecStorage<f64>> = Section::new(atlas.clone());
    let err = sec2.try_set(q0, &[1.0, 2.0, 3.0]).unwrap_err();
    println!("[neg] slice-length mismatch -> {:?}", err);

    // c) Overlap: wrong rank in payload (guarded by validate_invariants)
    #[cfg(any(feature = "strict-invariants", feature = "check-invariants"))]
    {
        use mesh_sieve::overlap::overlap::{OvlId, Remote};
        let src = OvlId::Local(q0);
        let dst = OvlId::Part(7);
        let mut ov_bad = ov.clone();
        Sieve::add_arrow(
            &mut ov_bad,
            src,
            dst,
            Remote {
                rank: 5,
                remote_point: None,
            },
        );
        assert!(ov_bad.validate_invariants().is_err());
    }

    println!("OK: e2e showcase finished");
    Ok(())
}

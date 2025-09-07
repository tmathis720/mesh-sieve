use mesh_sieve::algs::communicator::NoComm;
use mesh_sieve::algs::distribute::distribute_mesh;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

#[test]
fn structural_overlap_only() {
    let mut g = InMemorySieve::<PointId, ()>::default();
    let p = |x| PointId::new(x).unwrap();
    g.add_arrow(p(1), p(2), ());
    g.add_arrow(p(2), p(3), ());
    let parts = vec![0, 1, 1];

    let comm = NoComm;
    let (local0, ov0) = distribute_mesh(&g, &parts, &comm).unwrap();
    assert_eq!(local0.points().count(), 0);

    let ranks: Vec<_> = ov0.neighbor_ranks().collect();
    assert_eq!(ranks, vec![1]);
    let links: Vec<_> = ov0.links_to(1).collect();
    assert!(
        links
            .iter()
            .any(|(p, rp)| *p == PointId::new(3).unwrap() && rp.is_none())
    );
}

#[test]
fn invariants_hold() {
    let mut g = InMemorySieve::<PointId, ()>::default();
    let p = |x| PointId::new(x).unwrap();
    g.add_arrow(p(1), p(2), ());
    g.add_arrow(p(2), p(3), ());
    let parts = vec![0, 1, 1];
    let comm = NoComm;
    let (_local, ov) = distribute_mesh(&g, &parts, &comm).unwrap();

    #[cfg(any(debug_assertions, feature = "check-invariants"))]
    ov.validate_invariants().unwrap();
}

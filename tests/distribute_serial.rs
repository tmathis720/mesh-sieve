#[test]
fn distribute_mesh_serial() {
    use sieve_rs::algs::communicator::NoComm;
    use sieve_rs::topology::sieve::{InMemorySieve, Sieve};
    use sieve_rs::topology::point::PointId;
    use sieve_rs::algs::distribute_mesh;
    // Build a simple mesh: 1->2, 2->3
    let mut global = InMemorySieve::<PointId,()>::default();
    global.add_arrow(PointId::new(1), PointId::new(2), ());
    global.add_arrow(PointId::new(2), PointId::new(3), ());
    // Partition map: 1→0, 2→1, 3→1
    let parts = vec![0,1,1];
    let comm = NoComm;
    let (local, overlap) = distribute_mesh(&global, &parts, &comm);
    // Assert local ownership
    assert_eq!(local.cone(PointId::new(1)).count(), 0);
    assert_eq!(local.cone(PointId::new(2)).count(), 1);
    // Overlap should contain remote links for ghost points
    let ghosts: Vec<_> = overlap.support(PointId::new(2)).collect();
    assert!(ghosts.iter().any(|&(src,rem)| src==PointId::new(2) && rem.rank==0));
}

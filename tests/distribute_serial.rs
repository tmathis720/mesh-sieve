#[test]
fn distribute_mesh_serial() {
    use mesh_sieve::algs::communicator::NoComm;
    use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::algs::distribute_mesh;
    // Build a simple mesh: 1->2, 2->3
    let mut global = InMemorySieve::<PointId,()>::default();
    global.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
    global.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
    // Partition map: 1→0, 2→1, 3→1
    let parts = vec![0,1,1];
    let comm = NoComm;
    let (local, overlap) = distribute_mesh(&global, &parts, &comm).unwrap();
    // Assert local ownership: only point 1 is owned by rank 0, so no local arrows
    assert_eq!(local.cone(PointId::new(1).unwrap()).count(), 0);
    assert_eq!(local.cone(PointId::new(2).unwrap()).count(), 0);
    assert_eq!(local.cone(PointId::new(3).unwrap()).count(), 0);
    // Overlap should contain remote links for ghost points
    let ghosts: Vec<_> = overlap.links_to(1).collect();
    println!("ghosts for rank 1: {:?}", ghosts);
    // In this partition, point 3 is a ghost for rank 1, so overlap should contain (3, 3)
    assert!(ghosts.contains(&(PointId::new(3).unwrap(), PointId::new(3).unwrap())));
}

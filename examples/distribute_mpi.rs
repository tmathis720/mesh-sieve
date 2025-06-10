// This example demonstrates how to distribute a simple mesh across two MPI ranks
// using the `sieve_rs` library. It creates a mesh with three points and two arrows,
// partitions it such that rank 0 owns point 1 and rank 1 owns points 2 and 3, and
// then distributes the mesh using the `distribute_mesh` function. Each rank prints
// its local mesh and overlap for manual inspection.
fn main() {
    use sieve_rs::algs::communicator::{Communicator, MpiComm};
    use sieve_rs::topology::sieve::{InMemorySieve, Sieve};
    use sieve_rs::topology::point::PointId;
    use sieve_rs::algs::distribute_mesh;
    let comm = MpiComm::new();
    if comm.size() != 2 {
        eprintln!("This test requires 2 MPI ranks");
        return;
    }
    // Build a simple mesh: 1->2, 2->3
    let mut global = InMemorySieve::<PointId,()>::default();
    global.add_arrow(PointId::new(1), PointId::new(2), ());
    global.add_arrow(PointId::new(2), PointId::new(3), ());
    // Partition map: 1→0, 2→1, 3→1
    let parts = vec![0,1,1];
    let (local, overlap) = distribute_mesh(&global, &parts, &comm);
    // Synchronize processes if needed; uncomment the following if using the `mpi` crate:
    // use mpi::traits::Communicator as _;
    // comm.barrier();
    // Each rank prints its local mesh and overlap for manual inspection
    println!("Rank {} local: {:?}", comm.rank(), local);
    println!("Rank {} overlap: {:?}", comm.rank(), overlap);
    // Optionally, add assertions for expected ownership/ghosts per rank
}

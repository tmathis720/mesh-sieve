// This example demonstrates how to distribute a simple mesh across two MPI ranks
// using the `mesh_sieve` library. It creates a mesh with three points and two arrows,
// partitions it such that rank 0 owns point 1 and rank 1 owns points 2 and 3, and
// then distributes the mesh using the `distribute_mesh` function. Each rank prints
// its local mesh and overlap for manual inspection.
fn main() {
    use mpi::traits::*;
    use mesh_sieve::algs::communicator::{Communicator, MpiComm};
    use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::algs::distribute_mesh;
    // 1. Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = MpiComm::new();
    if comm.size() != 2 {
        eprintln!("This test requires 2 MPI ranks");
        return;
    }
    // Build a simple mesh: 1->2, 2->3
    let mut global = InMemorySieve::<PointId,()>::default();
    global.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
    global.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
    // Partition map: 1→0, 2→1, 3→1
    let parts = vec![0,1,1];
    let (local, overlap) = distribute_mesh(&global, &parts, &comm);
    // Each rank prints its local mesh and overlap for manual inspection
    println!("Rank {} local: {:?}", comm.rank(), local);
    println!("Rank {} overlap: {:?}", comm.rank(), overlap);
    // Synchronize before exit
    world.barrier();
}

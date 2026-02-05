// cargo mpirun -n 2 --features mpi-support --example distribute_mpi
// Version 3.2.0: Passing
// This example demonstrates how to distribute a simple mesh across two MPI ranks
// using the `mesh_sieve` library. It creates a mesh with three points and two arrows,
// partitions it such that rank 0 owns point 1 and rank 1 owns points 2 and 3, and
// then distributes the mesh using the `distribute_mesh` function. Each rank prints
// its local mesh and overlap for manual inspection.
#[cfg(feature = "mpi-support")]
fn main() {
    use mesh_sieve::algs::communicator::{Communicator, MpiComm};
    use mesh_sieve::algs::distribute_mesh;
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
    // 1. Initialize MPI
    let comm = MpiComm::new().expect("MPI initialization failed");
    if comm.size() != 2 {
        eprintln!("This test requires 2 MPI ranks");
        return;
    }
    // Build a simple mesh: 1->2, 2->3
    let mut global = InMemorySieve::<PointId, ()>::default();
    global.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
    global.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
    // Partition map: 1→0, 2→1, 3→1
    let parts = vec![0, 1, 1];
    let (local, overlap) = match distribute_mesh(&global, &parts, &comm) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Error distributing mesh: {:?}", e);
            return;
        }
    };
    // Each rank prints its local mesh and overlap for manual inspection
    println!("Rank {} local: {:?}", comm.rank(), local);
    println!("Rank {} overlap: {:?}", comm.rank(), overlap);
    // Synchronize before exit
    comm.barrier();
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}

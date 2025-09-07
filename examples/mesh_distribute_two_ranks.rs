// examples/mesh_distribute_two_ranks.rs
// cargo mpirun -n 2 --features mpi-support --example mesh_distribute_two_ranks
// Version 1.2.0: Passing
// This example demonstrates how to distribute a simple mesh across two MPI ranks
// using the `mesh_sieve` library. It creates a mesh with two arrows, partitions it
// such that rank 0 owns the first arrow and rank 1 owns the second arrow, and
// then distributes the mesh using the `distribute_mesh` function. Each rank prints
// its local mesh and overlap for manual inspection.

#[cfg(feature = "mpi-support")]
fn main() {
    use mesh_sieve::algs::communicator::{Communicator, MpiComm};
    use mesh_sieve::algs::distribute::distribute_mesh;
    use mesh_sieve::overlap::overlap::{Overlap as OvlGraph, OvlId};
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
    let comm = MpiComm::default();
    let size = Communicator::size(&comm);
    let rank = Communicator::rank(&comm);

    if size != 2 {
        if rank == 0 {
            eprintln!("Run with exactly 2 MPI ranks");
        }
        return;
    }

    // 1) Build a toy global mesh: two arrows 1→2 and 3→4
    let mut global = InMemorySieve::<PointId, ()>::default();
    global.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
    global.add_arrow(PointId::new(3).unwrap(), PointId::new(4).unwrap(), ());

    // 2) Partition map: rank 0 owns {1,2}, rank 1 owns {3,4}
    let parts = vec![0, 0, 1, 1];

    // 3) Call distribute
    let (local, overlap) = match distribute_mesh(&global, &parts, &comm) {
        Ok((local, overlap)) => (local, overlap),
        Err(e) => {
            eprintln!("[rank {}] distribute_mesh failed: {}", rank, e);
            return;
        }
    };

    // 4) Each rank should see its submesh plus one overlap arrow
    let my_pts: Vec<_> = local.points().collect();
    println!("[rank {}] local points: {:?}", rank, my_pts);
    let partition_pt = OvlGraph::partition_node_id(rank);
    let ovl_pts: Vec<_> = overlap
        .support(partition_pt)
        .filter_map(|(p, _)| match p {
            OvlId::Local(q) => Some(q),
            _ => None,
        })
        .collect();
    println!("[rank {}] overlap points: {:?}", rank, ovl_pts);

    // Assert: rank 0 sees only PointId(1)→PointId(2) locally,
    //         plus an overlap link from 3→partition(1)
    if rank == 0 {
        assert!(local.points().any(|p| p == PointId::new(1).unwrap()));
        assert!(ovl_pts.is_empty());
    } else {
        assert!(local.points().any(|p| p == PointId::new(3).unwrap()));
        assert!(ovl_pts.contains(&PointId::new(1).unwrap()));
    }

    println!("[rank {}] distribute_mesh test passed", rank);
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}

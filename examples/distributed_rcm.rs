//! Example: Distributed RCM on a 2D grid mesh using MPI.
//! cargo mpirun -n 4 --features mpi-support --example distributed_rcm
//! Version 1.3.0: Passing
//! This example demonstrates how to use the `mesh_sieve` library to perform a distributed reverse Cuthill-McKee (RCM) ordering on a simple 2D grid mesh.

/// Example partitioning: assign each row of the grid to a rank
#[cfg(feature = "mpi-support")]
fn partition_vertices(nx: usize, ny: usize, size: usize) -> Vec<usize> {
    let mut parts = vec![0; nx * ny];
    let rows_per_rank = ny / size + if ny % size > 0 { 1 } else { 0 };
    for y in 0..ny {
        let rank = y / rows_per_rank;
        for x in 0..nx {
            let idx = y * nx + x;
            parts[idx] = rank.min(size - 1);
        }
    }
    parts
}

#[cfg(feature = "mpi-support")]
fn main() {
    use mesh_sieve::algs::communicator::{Communicator, MpiComm};
    use mesh_sieve::algs::rcm::distributed_rcm;
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
    let comm = MpiComm::new().expect("MPI initialization failed");
    let rank = comm.rank();
    let size = comm.size();

    // Build a 4x4 grid graph as a Sieve
    let nx = 4;
    let ny = 4;
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    for y in 0..ny {
        for x in 0..nx {
            let v = PointId::new((y * nx + x + 1) as u64).unwrap();
            // Right neighbor
            if x + 1 < nx {
                let v_right = PointId::new((y * nx + (x + 1) + 1) as u64).unwrap();
                sieve.add_arrow(v, v_right, ());
                sieve.add_arrow(v_right, v, ());
            }
            // Down neighbor
            if y + 1 < ny {
                let v_down = PointId::new(((y + 1) * nx + x + 1) as u64).unwrap();
                sieve.add_arrow(v, v_down, ());
                sieve.add_arrow(v_down, v, ());
            }
        }
    }

    // Partition: assign each vertex to a rank
    let parts = partition_vertices(nx, ny, size);
    let local_vertices: Vec<_> = (0..nx * ny)
        .filter(|&i| parts[i] == rank)
        .map(|i| PointId::new((i + 1) as u64).unwrap())
        .collect();
    let mut local_sieve = InMemorySieve::<PointId, ()>::default();
    // Build local sieve: add all local vertices and their local edges
    for &v in &local_vertices {
        for (tgt, _) in sieve.cone(v) {
            if local_vertices.contains(&tgt) {
                local_sieve.add_arrow(v, tgt, ());
            }
        }
    }

    println!("Rank {}: local vertices = {:?}", rank, local_vertices);
    // Count local edges
    let local_edge_count: usize = local_vertices.iter().map(|&v| sieve.cone(v).count()).sum();
    println!("Rank {}: local edges = {}", rank, local_edge_count);

    // Run distributed RCM
    let rcm_order = distributed_rcm(&local_sieve, &comm);
    // Sort and compare to check for valid permutation
    let mut rcm_sorted = rcm_order.clone();
    rcm_sorted.sort_by_key(|pid| pid.get());
    let mut expected = local_vertices.clone();
    expected.sort_by_key(|pid| pid.get());
    let valid = rcm_sorted == expected;
    assert!(
        valid,
        "RCM order is not a valid permutation for rank {}",
        rank
    );
    println!("Rank {}: RCM order {:?}", rank, rcm_order);

    comm.barrier();
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}

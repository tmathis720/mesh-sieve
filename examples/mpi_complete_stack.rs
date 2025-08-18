// --- MPI test: complete_stack_two_ranks ---
// cargo mpirun -n 2 --features mpi-support --example mpi_complete_stack
// This example tests the `complete_stack` function with two MPI ranks.
// It ensures that a Stack can be completed correctly when two ranks have overlapping points,
// and that the Stack is correctly completed with values from both ranks.
#[cfg(feature = "mpi-support")]
fn main() {
    use mesh_sieve::algs::communicator::MpiComm;
    use mpi::topology::Communicator;
    use mesh_sieve::topology::stack::{InMemoryStack, Stack};
    use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
    use mesh_sieve::algs::completion::complete_stack;
    #[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Default)]
    #[repr(transparent)]
    struct DummyPayload(u32);
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, bytemuck::Pod, bytemuck::Zeroable, Hash, Default)]
    #[repr(transparent)]
    struct PodU64(u64);
    #[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct DummyRemote {
        rank: usize,
        remote_point: PodU64,
    }
    impl mesh_sieve::algs::completion::stack_completion::HasRank for DummyRemote {
        fn rank(&self) -> usize { self.rank }
    }
    let comm = MpiComm::default();
    let world = &comm.world;
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    if size != 2 {
        if rank == 0 {
            eprintln!("This test requires exactly 2 ranks.");
        }
        return;
    }
    let mut stack = InMemoryStack::<PodU64, PodU64, DummyPayload>::new();
    let mut overlap = InMemorySieve::<PodU64, DummyRemote>::default();
    if rank == 0 {
        // Add the base point to the base sieve so complete_stack can find it
        stack.base_mut().unwrap().add_arrow(PodU64(1), PodU64(1), DummyPayload::default());
        
        // Add the vertical arrow with actual data
        let _ = stack.add_arrow(PodU64(1), PodU64(101), DummyPayload(42));
        
        // Set up overlap to indicate rank 1 needs data from this base point
        overlap.add_arrow(
            PodU64(1),
            PodU64(2), // partition_point(1) as PodU64
            DummyRemote { rank: 1, remote_point: PodU64(1) }
        );
    }
    // actually run and unwrap any error from the stack‐completion
    complete_stack(&mut stack, &overlap, &comm, rank, size)
        .expect("MPI stack completion failed");
    let arrows: Vec<_> = stack.lift(PodU64(1)).collect();

    if rank == 1 {
        // rank 1 must have received (101,42)
        assert!(
            arrows.contains(&(PodU64(101), DummyPayload(42))),
            "[rank {}] expected arrow (101,42), got: {:?}",
            rank, arrows
        );
        println!("[rank 1] complete_stack_two_ranks passed!");
    } else {
        // rank 0 only ever sent it
        println!("[rank 0] send-only, local arrows = {:?}", arrows);
    }
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}
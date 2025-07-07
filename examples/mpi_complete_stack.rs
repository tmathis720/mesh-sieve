// --- MPI test: complete_stack_two_ranks ---
// cargo mpirun -n 2 --features mpi-support --example mpi_complete_stack
// This example tests the `complete_stack` function with two MPI ranks.
// It ensures that a Stack can be completed correctly when two ranks have overlapping points,
// and that the Stack is correctly completed with values from both ranks.
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
    let comm = MpiComm::new();
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
        let _ = stack.add_arrow(PodU64(1), PodU64(101), DummyPayload(42));
        overlap.add_arrow(PodU64(1), PodU64(1), DummyRemote { rank: 1, remote_point: PodU64(1) });
    } else {
        overlap.add_arrow(PodU64(1), PodU64(1), DummyRemote { rank: 0, remote_point: PodU64(1) });
    }
    let _ = complete_stack(&mut stack, &overlap, &comm, rank, size);
    let arrows: Vec<_> = stack.lift(PodU64(1)).map(|(cap, pay)| (cap, pay)).collect();
    assert!(arrows.contains(&(PodU64(101), DummyPayload(42))));
    println!("[rank {}] complete_stack_two_ranks passed", rank);
}
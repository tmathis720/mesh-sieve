use mpi::topology::Communicator;


// --- MPI test: complete_section_no_overlap ---
// Run with: mpirun -n 3 cargo run --example mpi_complete
fn main() {
    use sieve_rs::algs::communicator::MpiComm;
    use sieve_rs::topology::point::PointId;
    use sieve_rs::overlap::overlap::Overlap;
    use sieve_rs::data::atlas::Atlas;
    use sieve_rs::data::section::Section;
    use sieve_rs::overlap::delta::CopyDelta;
    use sieve_rs::algs::completion::complete_section;
    let comm = MpiComm::new();
    let world = &comm.world;
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    if size != 3 {
        if rank == 0 {
            eprintln!("This test requires exactly 3 ranks.");
        }
        return;
    }
    let mut atlas = Atlas::default();
    if rank == 2 {
        atlas.insert(PointId::new(2), 1);
    }
    let mut sec = Section::<u32>::new(atlas);
    if rank == 2 {
        sec.set(PointId::new(2), &[42]);
    }
    let ovlp = Overlap::default();
    let delta = CopyDelta;
    complete_section(&mut sec, &ovlp, &comm, &delta, rank, size);
    if rank == 2 {
        assert_eq!(sec.restrict(PointId::new(2))[0], 42);
        println!("[rank 2] complete_section_no_overlap passed");
    }
}
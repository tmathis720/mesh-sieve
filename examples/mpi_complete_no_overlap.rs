use mpi::topology::Communicator;


// --- MPI test: complete_section_no_overlap ---
// Run with: mpirun -n 3 cargo run --example mpi_complete
fn main() {
    use mesh_sieve::algs::communicator::MpiComm;
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::overlap::overlap::Overlap;
    use mesh_sieve::data::atlas::Atlas;
    use mesh_sieve::data::section::Section;
    use mesh_sieve::overlap::delta::CopyDelta;
    use mesh_sieve::algs::completion::complete_section;
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
        atlas.try_insert(PointId::new(2).unwrap(), 1).expect("Failed to insert into atlas");
    }
    let mut sec = Section::<u32>::new(atlas);
    if rank == 2 {
        sec.set(PointId::new(2).unwrap(), &[42]);
    }
    let mut ovlp = Overlap::default();
    let delta = CopyDelta;
    complete_section(&mut sec, &mut ovlp, &comm, &delta, rank, size);
    if rank == 2 {
        assert_eq!(sec.restrict(PointId::new(2).unwrap())[0], 42);
        println!("[rank 2] complete_section_no_overlap passed");
    }
}
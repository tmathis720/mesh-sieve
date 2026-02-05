// cargo mpirun -n 3 --features mpi-support --example mpi_complete_no_overlap
// Version 3.2.0: Passing
// This example demonstrates how to complete a Section in MPI without overlap.
// It uses the `complete_section` function to ensure that a Section can be completed
// correctly when no overlap exists, and that the Section is correctly completed
// with values from the ranks that own the points.

#[cfg(feature = "mpi-support")]
fn main() {
    use mesh_sieve::algs::communicator::MpiComm;
    use mesh_sieve::algs::completion::complete_section;
    use mesh_sieve::data::atlas::Atlas;
    use mesh_sieve::data::section::Section;
    use mesh_sieve::data::storage::VecStorage;
    use mesh_sieve::overlap::delta::CopyDelta;
    use mesh_sieve::overlap::overlap::Overlap;
    use mesh_sieve::topology::point::PointId;
    use mpi::topology::Communicator;
    let comm = MpiComm::new().expect("MPI initialization failed");
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
        atlas
            .try_insert(PointId::new(2).unwrap(), 1)
            .expect("Failed to insert into atlas");
    }
    let mut sec = Section::<u32, VecStorage<u32>>::new(atlas);
    if rank == 2 {
        sec.try_set(PointId::new(2).unwrap(), &[42])
            .expect("Failed to set section value");
    }
    let mut ovlp = Overlap::default();
    complete_section::<u32, VecStorage<u32>, CopyDelta, MpiComm>(&mut sec, &ovlp, &comm, rank)
        .expect("section completion failed");
    if rank == 2 {
        match sec.try_restrict(PointId::new(2).unwrap()) {
            Ok(values) => assert_eq!(values[0], 42),
            Err(e) => panic!("Failed to restrict section: {:?}", e),
        }
        println!("[rank 2] complete_section_no_overlap passed");
    }
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}

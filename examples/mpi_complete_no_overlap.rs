// cargo mpirun -n 3 --features mpi-support --example mpi_complete_no_overlap
// This example demonstrates how to complete a Section in MPI without overlap.
// It uses the `complete_section` function to ensure that a Section can be completed
// correctly when no overlap exists, and that the Section is correctly completed
// with values from the ranks that own the points.
use mpi::topology::Communicator;

fn main() {
    use mesh_sieve::algs::communicator::MpiComm;
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::overlap::overlap::Overlap;
    use mesh_sieve::data::atlas::Atlas;
    use mesh_sieve::data::section::Section;
    use mesh_sieve::overlap::delta::CopyDelta;
    use mesh_sieve::algs::completion::complete_section;
    let comm = MpiComm::default();
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
        sec.try_set(PointId::new(2).unwrap(), &[42]).expect("Failed to set section value");
    }
    let mut ovlp = Overlap::default();
    let delta = CopyDelta;
    complete_section(&mut sec, &mut ovlp, &comm, &delta, rank, size);
    if rank == 2 {
        match sec.try_restrict(PointId::new(2).unwrap()) {
            Ok(values) => assert_eq!(values[0], 42),
            Err(e) => panic!("Failed to restrict section: {:?}", e),
        }
        println!("[rank 2] complete_section_no_overlap passed");
    }
}
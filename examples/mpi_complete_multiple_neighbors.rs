// --- MPI test: complete_section_multiple_neighbors ---
pub fn main() {
    use mesh_sieve::algs::communicator::MpiComm;
    use mpi::topology::Communicator;
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
    let mut ovlp = Overlap::default();
    // Insert all points into atlas before constructing Section
    match rank {
        0 => {
            atlas.insert(PointId::new(1), 1);
            atlas.insert(PointId::new(2), 1);
            ovlp.add_link(PointId::new(1), 1, PointId::new(101));
            ovlp.add_link(PointId::new(2), 2, PointId::new(201));
        }
        1 => {
            atlas.insert(PointId::new(101), 1); // local
            atlas.insert(PointId::new(1), 1);   // remote (from rank 0)
            ovlp.add_link(PointId::new(101), 0, PointId::new(1));
        }
        2 => {
            atlas.insert(PointId::new(201), 1); // local
            atlas.insert(PointId::new(2), 1);   // remote (from rank 0)
            ovlp.add_link(PointId::new(201), 0, PointId::new(2));
        }
        _ => {}
    }
    // Now construct Section after all atlas.insert
    let mut sec = Section::<u32>::new(atlas);
    if rank == 0 {
        sec.set(PointId::new(1), &[1]);
        sec.set(PointId::new(2), &[2]);
    }
    let delta = CopyDelta;
    complete_section(&mut sec, &mut ovlp, &comm, &delta, rank, size);
    match rank {
        0 => {
            assert_eq!(sec.restrict(PointId::new(1))[0], 1);
            assert_eq!(sec.restrict(PointId::new(2))[0], 2);
            println!("[rank 0] complete_section_multiple_neighbors passed");
        }
        1 => {
            assert_eq!(sec.restrict(PointId::new(101))[0], 1);
            println!("[rank 1] complete_section_multiple_neighbors passed");
        }
        2 => {
            assert_eq!(sec.restrict(PointId::new(201))[0], 2);
            println!("[rank 2] complete_section_multiple_neighbors passed");
        }
        _ => {}
    }
}
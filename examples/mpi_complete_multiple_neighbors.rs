// --- MPI test: complete_section_multiple_neighbors ---
// cargo mpirun -n 3 --features mpi-support --example mpi_complete_multiple_neighbors
// Version 1.3.0: passing
//! This example tests the `complete_section` function with multiple neighbors.
//! It ensures that a Section can be completed correctly when multiple ranks have neighbors
//! that share the same point, and that the Section is correctly completed
//! with values from all ranks.
//!
#[cfg(feature = "mpi-support")]
pub fn main() {
    use mesh_sieve::algs::communicator::MpiComm;
    use mesh_sieve::algs::completion::complete_section;
    use mesh_sieve::data::atlas::Atlas;
    use mesh_sieve::data::section::Section;
    use mesh_sieve::data::storage::VecStorage;
    use mesh_sieve::overlap::delta::CopyDelta;
    use mesh_sieve::overlap::overlap::{Overlap, OvlId, Remote};
    use mesh_sieve::topology::point::PointId;
    use mesh_sieve::topology::sieve::sieve_trait::Sieve;
    use mpi::topology::Communicator;
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
    let mut ovlp = Overlap::default();
    // Insert all points into atlas before constructing Section
    match rank {
        0 => {
            atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
            atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
            // also allocate ghost slots for neighbors so symmetric exchange can fuse
            atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap();
            atlas.try_insert(PointId::new(201).unwrap(), 1).unwrap();
            // Use add_arrow to avoid closure enforcement issues
            // partition_point(1) = PointId(2), partition_point(2) = PointId(3)
            Sieve::add_arrow(
                &mut ovlp,
                OvlId::Local(PointId::new(1).unwrap()),
                OvlId::Part(1),
                Remote {
                    rank: 1,
                    remote_point: Some(PointId::new(101).unwrap()),
                },
            );
            Sieve::add_arrow(
                &mut ovlp,
                OvlId::Local(PointId::new(2).unwrap()),
                OvlId::Part(2),
                Remote {
                    rank: 2,
                    remote_point: Some(PointId::new(201).unwrap()),
                },
            );
        }
        1 => {
            atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap(); // local
            atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap(); // remote (from rank 0)
            Sieve::add_arrow(
                &mut ovlp,
                OvlId::Local(PointId::new(101).unwrap()),
                OvlId::Part(0),
                Remote {
                    rank: 0,
                    remote_point: Some(PointId::new(1).unwrap()),
                },
            );
        }
        2 => {
            atlas.try_insert(PointId::new(201).unwrap(), 1).unwrap(); // local
            atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap(); // remote (from rank 0)
            Sieve::add_arrow(
                &mut ovlp,
                OvlId::Local(PointId::new(201).unwrap()),
                OvlId::Part(0),
                Remote {
                    rank: 0,
                    remote_point: Some(PointId::new(2).unwrap()),
                },
            );
        }
        _ => {}
    }
    // Now construct Section after all atlas.insert
    let mut sec = Section::<u32, VecStorage<u32>>::new(atlas);
    if rank == 0 {
        sec.try_set(PointId::new(1).unwrap(), &[1])
            .expect("Failed to set value for PointId 1");
        sec.try_set(PointId::new(2).unwrap(), &[2])
            .expect("Failed to set value for PointId 2");
    }
    complete_section::<u32, VecStorage<u32>, CopyDelta, MpiComm>(&mut sec, &ovlp, &comm, rank)
        .expect("section completion failed");
    match rank {
        0 => {
            assert_eq!(
                sec.try_restrict(PointId::new(1).unwrap())
                    .expect("Point 1 missing")[0],
                1
            );
            assert_eq!(
                sec.try_restrict(PointId::new(2).unwrap())
                    .expect("Point 2 missing")[0],
                2
            );
            println!("[rank 0] complete_section_multiple_neighbors passed");
        }
        1 => {
            assert_eq!(
                sec.try_restrict(PointId::new(101).unwrap())
                    .expect("Point 101 missing")[0],
                1
            );
            println!("[rank 1] complete_section_multiple_neighbors passed");
        }
        2 => {
            assert_eq!(
                sec.try_restrict(PointId::new(201).unwrap())
                    .expect("Point 201 missing")[0],
                2
            );
            println!("[rank 2] complete_section_multiple_neighbors passed");
        }
        _ => {}
    }
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}

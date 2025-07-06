//! Run with: mpirun -n 2 cargo run --example mpi_complete
use mesh_sieve::algs::communicator::MpiComm;
use mpi::topology::Communicator;
use std::process;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::algs::completion::complete_section;

fn main() {
    // 1) Init MPI
    let comm = MpiComm::new();
    let world = &comm.world;
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    if size != 2 {
        eprintln!("This example requires exactly 2 ranks.");
        process::exit(1);
    }

    // 2) Define two mesh points: p0 for rank 0, p1 for rank 1
    let p0 = PointId::new(1).expect("Failed to create PointId p0");
    let p1 = PointId::new(2).expect("Failed to create PointId p1");

    // 3) Build the Overlap so each rank will send its owned point to the other
    let mut ovlp = Overlap::default();
    if rank == 0 {
        // rank0 owns p0, ghost slot is p1 on rank1
        ovlp.add_link(p0, 1, p1);
    } else {
        // rank1 owns p1, ghost slot is p0 on rank0
        ovlp.add_link(p1, 0, p0);
    }

    // 4) Build a Section that contains both the owned slot and the ghost slot
    // so that after complete_section() each rank can see both values.
    let mut atlas = Atlas::default();
    if rank == 0 {
        atlas.insert(p0, 1);  // my owned DOF
        atlas.insert(p1, 1);  // ghost DOF
    } else {
        atlas.insert(p1, 1);
        atlas.insert(p0, 1);
    }
    let mut sec = Section::<u32>::new(atlas);

    // 5) Seed the owned DOF with a distinct value
    if rank == 0 {
        sec.set(p0, &[100]);
    } else {
        sec.set(p1, &[200]);
    }

    // Debug: print neighbour links before exchange
    let links = mesh_sieve::algs::completion::neighbour_links::neighbour_links(&sec, &mut ovlp, rank);
    println!("[rank {}] neighbour_links: {:?}", rank, links);

    // 6) Perform the two-phase exchange
    let delta = CopyDelta;
    complete_section(&mut sec, &mut ovlp, &comm, &delta, rank, size);

    // 7) Check the result
    if rank == 0 {
        // rank0 should have received rank1’s 200 into its p1 slot
        let got = sec.restrict(p1)[0];
        println!("rank0: received ghost p1 = {}", got);
        assert_eq!(got, 200);
    } else {
        // rank1 should have received rank0’s 100 into its p0 slot
        let got = sec.restrict(p0)[0];
        println!("rank1: received ghost p0 = {}", got);
        assert_eq!(got, 100);
    }
    if rank == 0 {
        println!("MPI two-rank example succeeded!");
    }
}

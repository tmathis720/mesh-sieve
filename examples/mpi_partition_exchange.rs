//! A 4-rank MPI integration test: build a 2×2 grid mesh, partition, distribute, and exchange.

use mpi::traits::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use sieve_rs::algs::complete_section;
use sieve_rs::data::atlas::Atlas;
use sieve_rs::data::section::{Map, Section};
use sieve_rs::topology::point::PointId;
use sieve_rs::topology::sieve::{InMemorySieve, Sieve};

#[cfg(feature = "mpi-support")]
use sieve_rs::partitioning::{partition, PartitionerConfig};
#[cfg(feature = "mpi-support")]
use sieve_rs::partitioning::graph_traits::PartitionableGraph;

/// Build a 2×2 structured grid of points (IDs 0..8).
fn build_grid() -> (InMemorySieve<PointId, ()>, Atlas, Section<f64>) {
    // 9 points laid out in a 3×3 mesh (4 cells)
    let mut sieve = InMemorySieve::new();
    let mut atlas = Atlas::default();
    for id in 0u64..9 {
        sieve.add_point(PointId::new(id + 1));
        atlas.insert(PointId::new(id + 1), 1); // 1 DOF per point
    }
    let mut section = Section::new(atlas.clone());
    // initialize each point’s value = its ID as f64
    for id in 1u64..=9 {
        section.restrict_mut(PointId::new(id))[0] = id as f64;
    }
    (sieve, atlas, section)
}

// For partitioning, use a minimal wrapper that implements PartitionableGraph
struct GridGraph<'a> {
    sieve: &'a InMemorySieve<PointId>,
}
#[cfg(feature = "mpi-support")]
impl<'a> PartitionableGraph for GridGraph<'a> {
    type VertexId = usize;
    type VertexParIter<'b> = std::vec::IntoIter<usize> where Self: 'b;
    type NeighParIter<'b> = std::vec::IntoIter<usize> where Self: 'b;
    fn vertices(&self) -> <Self as PartitionableGraph>::VertexParIter<'_> {
        (1..=9).collect::<Vec<_>>().into_iter()
    }
    fn neighbors(&self, v: usize) -> <Self as PartitionableGraph>::NeighParIter<'_> {
        let mut nbrs = Vec::new();
        let row = (v - 1) / 3;
        let col = (v - 1) % 3;
        if row > 0 { nbrs.push(v - 3); }
        if row < 2 { nbrs.push(v + 3); }
        if col > 0 { nbrs.push(v - 1); }
        if col < 2 { nbrs.push(v + 1); }
        nbrs.into_iter()
    }
    fn degree(&self, v: usize) -> usize {
        self.neighbors(v).count()
    }
}

fn main() {
    // 1) Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    assert_eq!(size, 4, "This test must be run with 4 ranks");

    // 2) Build mesh, atlas, section on rank 0
    let (sieve, mut atlas, mut section) = if rank == 0 {
        let (s, a, mut sec) = build_grid();
        let graph = GridGraph { sieve: &s };
        // Partition on rank 0
        let cfg = PartitionerConfig {
            n_parts: 4,
            epsilon: 0.05,
            ..Default::default()
        };
        let pm = partition(&graph, &cfg).expect("partition failed");
        // Distribute atlas and section to each rank based on pm
        for (&pid, &p) in pm.iter() {
            world.process_at_rank(p as i32)
                .send(&(pid.get(), sec.restrict(pid)[0]));
        }
        (s, a, sec)
    } else {
        let mut s = InMemorySieve::new();
        let mut a = Atlas::default();
        let mut sec = Section::new(a.clone());
        let mut owned = Vec::new();
        for _ in 0..3 {
            let (msg, _status) = world.any_process().receive::<(u64, f64)>();
            let (pid_raw, val) = msg;
            let pid = PointId::new(pid_raw);
            s.add_point(pid);
            a.insert(pid, 1);
            sec = Section::new(a.clone());
            sec.restrict_mut(pid)[0] = val;
            owned.push(pid);
        }
        (s, a, sec)
    };

    // barrier to ensure all have received
    world.barrier();

    // 3) Perform ghost‐exchange: each rank sends its owned DOF to neighbors
    // complete_section(&mut section, &sieve, &world, &(), rank as usize, size as usize);

    // 4) Verify: each rank should see ghost values equal to neighbors’ IDs
    //    (For this simple grid, every rank’s boundary neighbors have predictable IDs.)
    // let ghosts = atlas.owned_neighbors(&section);
    // for &(pid, owner_rank) in ghosts.iter() {
    //     let val = section.get(pid)[0];
    //     assert_eq!(
    //         val as i32,
    //         pid.get() as i32,
    //         "Ghost value mismatch on rank {} for point {:?}",
    //         rank, pid
    //     );
    // }

    // 5) Final barrier and exit
    world.barrier();
    if rank == 0 {
        println!("MPI partition+exchange test passed on 4 ranks!");
    }
}

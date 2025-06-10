// src/algs/distribute.rs

use crate::algs::communicator::Communicator;
use mpi::topology::Communicator as MpiCommunicator;
use crate::topology::point::PointId;
use crate::topology::sieve::{Sieve, InMemorySieve};
use crate::overlap::overlap::Remote;
use crate::algs::completion::{sieve_completion, section_completion};
use std::collections::HashMap;

/// Distribute a global mesh across MPI ranks.
///
/// # Arguments
/// - `mesh`: the full global mesh (arrows of type `Payload=()`)  
/// - `parts`: a slice of length `mesh.points().count()`, mapping each `PointId.get() as usize` to a rank  
/// - `comm`: your communicator (MPI or Rayon)
///
/// # Returns
/// `(local_mesh, overlap)` where:
/// - `local_mesh`: only those arrows owned by this rank  
/// - `overlap`: the overlap Sieve (arrows `PointId→partition_pt(rank)`) for ghost‐exchange  
pub fn distribute_mesh<M, C>(
    mesh: &M,
    parts: &[usize],
    comm: &C,
) -> (InMemorySieve<PointId,()>, InMemorySieve<PointId,Remote>)
where
    M: Sieve<Point = PointId, Payload = ()>,
    C: Communicator + Sync,
{
    let my_rank = comm.rank();      // assume your Communicator exposes `rank()`
    let n_ranks = comm.size();      // and `size()`
    // 1) Build the “overlap” sieve
    let mut overlap = InMemorySieve::<PointId,Remote>::default();
    for p in mesh.points() {
        let owner = parts[p.get() as usize];
        let part_pt = PointId::new((owner as u64)+1);
        if p != part_pt {
            overlap.add_arrow(p, part_pt, Remote { rank: owner, remote_point: p });
        }
    }

    // 2) Extract local submesh: only arrows whose src→dst are both owned here
    let mut local = InMemorySieve::<PointId,()>::default();
    for p in mesh.base_points() {
        if parts[p.get() as usize] == my_rank {
            for (dst, _) in mesh.cone(p) {
                if parts[dst.get() as usize] == my_rank {
                    local.add_arrow(p, dst, ());
                }
            }
        }
    }

    // 3) Complete the overlap graph of arrows across ranks
    let overlap_clone = overlap.clone();
    sieve_completion::complete_sieve(&mut overlap, &overlap_clone, comm, my_rank);

    // 4) (Optional: exchange data if needed, but for mesh topology with () payload, this is not required)

    (local, overlap)
}

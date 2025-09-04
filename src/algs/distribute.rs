// src/algs/distribute.rs

use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::overlap::overlap::Overlap;
use crate::algs::communicator::Communicator;



/// Distribute a global mesh across ranks, returning the local submesh and overlap graph.
///
/// Implements Sec. 3 mesh distribution: builds the overlap Sieve and extracts the local submesh.
///
/// # Arguments
/// - `mesh`: the full global mesh (arrows of type `Payload=()`)
/// - `parts`: a slice mapping each `PointId.get() as usize` to a rank
/// - `comm`: your communicator (MPI or Rayon)
///
/// # Returns
/// `(local_mesh, overlap)` where:
/// - `local_mesh`: only those arrows owned by this rank
/// - `overlap`: the overlap Sieve (arrows `PointId→partition_pt(rank)`) for ghost‐exchange
///
/// # Example (serial)
/// ```rust
/// use mesh_sieve::algs::communicator::NoComm;
/// use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
/// use mesh_sieve::topology::point::PointId;
/// use mesh_sieve::algs::distribute_mesh;
/// let mut global = InMemorySieve::<PointId,()>::default();
/// global.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
/// global.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
/// let parts = vec![0,1,1];
/// let comm = NoComm;
/// let (local, overlap) = distribute_mesh(&global, &parts, &comm).unwrap();
/// assert_eq!(local.cone(PointId::new(1).unwrap()).count(), 0);
/// assert_eq!(local.cone(PointId::new(2).unwrap()).count(), 0);
/// assert_eq!(local.cone(PointId::new(3).unwrap()).count(), 0);
/// let ghosts: Vec<_> = overlap.links_to_resolved(1).collect();
/// assert!(ghosts.contains(&(PointId::new(3).unwrap(), PointId::new(3).unwrap())));
/// ```
/// # Example (MPI)
/// ```ignore
/// #![cfg(feature="mpi-support")]
/// use mesh_sieve::algs::communicator::MpiComm;
/// // ... same as above, but use MpiComm::new() and run with mpirun -n 2 ...
/// ```
pub fn distribute_mesh<M, C>(
    mesh: &M,
    parts: &[usize],
    comm: &C,
) -> Result<(InMemorySieve<PointId, ()>, Overlap), crate::mesh_error::MeshSieveError>
where
    M: Sieve<Point=PointId, Payload=()>,
    C: Communicator + Sync,
{
    let my_rank = comm.rank();
    let _n_ranks = comm.size();
    // 1) Build the “overlap” sieve
    let mut overlap = Overlap::new();
    for p in mesh.points() {
        let idx = p.get().checked_sub(1)
            .ok_or(crate::mesh_error::MeshSieveError::PartitionIndexOutOfBounds(p.get() as usize))? as usize;
        let owner = *parts.get(idx)
            .ok_or(crate::mesh_error::MeshSieveError::PartitionIndexOutOfBounds(p.get() as usize))?;
        if owner != my_rank {
            overlap.add_link(p, owner, p);
        }
    }
    // 2) Extract local submesh: only arrows whose src→dst are both owned here
    let mut local = InMemorySieve::<PointId,()>::default();
    for base in mesh.base_points() {
        let base_idx = base.get().checked_sub(1)
            .ok_or(crate::mesh_error::MeshSieveError::PartitionIndexOutOfBounds(base.get() as usize))? as usize;
        if *parts.get(base_idx).ok_or(crate::mesh_error::MeshSieveError::PartitionIndexOutOfBounds(base.get() as usize))? == my_rank {
            for (dst, _) in mesh.cone(base) {
                let dst_idx = dst.get().checked_sub(1)
                    .ok_or(crate::mesh_error::MeshSieveError::PartitionIndexOutOfBounds(dst.get() as usize))? as usize;
                if *parts.get(dst_idx).ok_or(crate::mesh_error::MeshSieveError::PartitionIndexOutOfBounds(dst.get() as usize))? == my_rank {
                    local.add_arrow(base, dst, ());
                }
            }
        }
    }
    // 3) (Optional) completion of overlap graph could be performed here
    // 4) (Optional: exchange data if needed, but for mesh topology with () payload, this is not required)
    Ok((local, overlap))
}

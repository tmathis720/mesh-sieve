// src/algs/distribute.rs

use crate::algs::communicator::Communicator;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{Overlap, OvlId};
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};

/// Distribute a global mesh across ranks, returning the local submesh and overlap graph.
///
/// Phase A of distribution: extract the local topology and create structural overlap
/// links (`Local(p) -> Part(r)`), leaving `remote_point` unresolved for later phases.
///
/// # Arguments
/// - `mesh`: the full global mesh (arrows of type `Payload = ()`)
/// - `parts`: mapping each `PointId` (1-based) to an owning rank
/// - `comm`: communicator providing `rank()` and `size()`
///
/// # Returns
/// `(local_mesh, overlap)` where:
/// - `local_mesh`: only arrows whose endpoints are both owned by this rank
/// - `overlap`: bipartite `Local(p) -> Part(r)` links for every foreign point
///
/// ## Phases
/// - **Phase A (here):** extract local topology and build structural overlap.
/// - **Phase B (later):** expand overlap via mesh closure rules.
/// - **Phase C (later):** resolve remote IDs via exchange/service.
/// - **Phase D (later):** complete section/stack data using the overlap.
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
/// let parts = vec![0, 1, 1];
/// let comm = NoComm;
/// let (_local, overlap) = distribute_mesh(&global, &parts, &comm).unwrap();
/// let ranks: Vec<_> = overlap.neighbor_ranks().collect();
/// assert!(ranks.contains(&1));
/// let links: Vec<_> = overlap.links_to(1).collect();
/// assert!(links
///     .iter()
///     .any(|(p, rp)| *p == PointId::new(3).unwrap() && rp.is_none()));
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
) -> Result<(InMemorySieve<PointId, ()>, Overlap), MeshSieveError>
where
    M: Sieve<Point = PointId, Payload = ()>,
    C: Communicator + Sync,
{
    let my_rank = comm.rank();

    // ---------- Pass 0: collect points and validate `parts` ----------
    let mut max_id = 0u64;
    let pts: Vec<PointId> = mesh
        .points()
        .inspect(|p| max_id = max_id.max(p.get()))
        .collect();
    if parts.len() < max_id as usize {
        return Err(MeshSieveError::PartitionIndexOutOfBounds(parts.len()));
    }

    // ---------- Pass 1: collect foreign points ----------
    let mut foreign_pts: Vec<(PointId, usize)> = Vec::new();
    foreign_pts.reserve(pts.len() / 2);

    for p in &pts {
        let owner = owner_of(parts, *p)?;
        if owner != my_rank {
            foreign_pts.push((*p, owner));
        }
    }

    // ---------- Build Overlap ----------
    let mut overlap = Overlap::default();
    overlap.add_links_structural_bulk(foreign_pts);

    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    overlap.validate_invariants()?;

    // ---------- Build local submesh ----------
    let mut local = InMemorySieve::<PointId, ()>::default();
    for src in &pts {
        if owner_of(parts, *src)? == my_rank {
            for (dst, _) in mesh.cone(*src) {
                if owner_of(parts, dst)? == my_rank {
                    local.add_arrow(*src, dst, ());
                }
            }
        }
    }

    Ok((local, overlap))
}

/// Only for single-process demos/tests: set `remote_point = Some(local_p)` for all links.
pub fn resolve_overlap_identity(overlap: &mut Overlap) {
    let mut to_resolve = Vec::new();
    for src in overlap.base_points() {
        if let OvlId::Local(p) = src {
            for (dst, rem) in overlap.cone(src) {
                if let OvlId::Part(r) = dst {
                    debug_assert_eq!(rem.rank, r);
                    to_resolve.push((p, r));
                }
            }
        }
    }
    for (p, r) in to_resolve {
        overlap
            .resolve_remote_point(p, r, p)
            .expect("resolve_remote_point failed");
    }
}

#[inline]
fn owner_of(parts: &[usize], p: PointId) -> Result<usize, MeshSieveError> {
    let idx = p
        .get()
        .checked_sub(1)
        .ok_or(MeshSieveError::PartitionIndexOutOfBounds(p.get() as usize))? as usize;
    parts
        .get(idx)
        .copied()
        .ok_or(MeshSieveError::PartitionIndexOutOfBounds(idx))
}

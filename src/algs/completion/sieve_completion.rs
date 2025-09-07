//! Complete missing sieve arrows across ranks using minimal wire arrows.
//!
//! The protocol performs a symmetric two-phase exchange:
//! 1. Each rank sends the number of arrows it will send to every neighbor.
//! 2. Ranks exchange the actual `(src,dst)` arrow pairs, already translated into
//!    the receiver's local `PointId` space via the [`Overlap`] mapping.
//!
//! The wire payload is reduced to two `u64` integers (`WireArrow`), removing
//! dependencies on remote ranks or mesh payloads. Inserted edges carry the
//! `Default` payload of the mesh's `Sieve` implementation.

use std::collections::{BTreeSet, HashMap, HashSet};

use bytemuck::{cast_slice, cast_slice_mut, Zeroable};

use crate::algs::communicator::{CommTag, Communicator, SieveCommTags, Wait};
use crate::algs::completion::size_exchange::exchange_sizes_symmetric;
use crate::algs::wire::WireArrow;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::point::PointId;
use crate::topology::sieve::sieve_trait::Sieve;

/// Translate a local `PointId` into the neighbor's ID space using the
/// [`Overlap`] mapping.
fn remote_id_for(overlap: &Overlap, nbr: usize, local: PointId) -> Result<PointId, MeshSieveError> {
    overlap
        .links_to(nbr)
        .find(|(p, _)| *p == local)
        .and_then(|(_, rp)| rp)
        .ok_or_else(|| MeshSieveError::MissingOverlap {
            source: format!(
                "Unresolved mapping for local {} to neighbor {}",
                local.get(),
                nbr
            )
            .into(),
        })
}

/// Build per-neighbor buffers of [`WireArrow`] records, expressed in the
/// receiver's ID space.
fn build_wires<S: Sieve<Point = PointId>>(
    mesh: &S,
    overlap: &Overlap,
    neighbors: &[usize],
) -> Result<HashMap<usize, Vec<WireArrow>>, MeshSieveError> {
    let mut srcs_per_nbr: HashMap<usize, Vec<PointId>> = HashMap::new();
    for &nbr in neighbors {
        let mut srcs: Vec<PointId> = overlap.links_to(nbr).map(|(p, _)| p).collect();
        srcs.sort_unstable();
        srcs.dedup();
        srcs_per_nbr.insert(nbr, srcs);
    }

    let mut est_cap: HashMap<usize, usize> = HashMap::new();
    for (&nbr, srcs) in &srcs_per_nbr {
        let mut sum = 0usize;
        for &s in srcs {
            sum += mesh.cone_points(s).count();
        }
        est_cap.insert(nbr, sum);
    }

    let mut wires: HashMap<usize, Vec<WireArrow>> = HashMap::new();
    for (&nbr, srcs) in &srcs_per_nbr {
        let mut buf = Vec::with_capacity(*est_cap.get(&nbr).unwrap_or(&0));
        for &src_local in srcs {
            let src_remote = remote_id_for(overlap, nbr, src_local)?;
            let mut dsts: Vec<PointId> = mesh.cone_points(src_local).collect();
            dsts.sort_unstable();
            dsts.dedup();
            for dst_local in dsts {
                let dst_remote = remote_id_for(overlap, nbr, dst_local)?;
                buf.push(WireArrow::new(src_remote.get(), dst_remote.get()));
            }
        }
        buf.sort_unstable_by_key(|w| (w.src(), w.dst()));
        buf.dedup_by_key(|w| (w.src(), w.dst()));
        wires.insert(nbr, buf);
    }
    Ok(wires)
}

/// Complete missing sieve arrows using explicit communication tags.
pub fn complete_sieve_with_tags<S, C>(
    mesh: &mut S,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
    tags: SieveCommTags,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
    S::Payload: Default + Clone + Send + 'static,
    C: Communicator + Sync,
{
    #[cfg(any(debug_assertions, feature = "check-invariants"))]
    overlap.validate_invariants()?;

    // Determine deterministic neighbor set (exclude self)
    let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
    nb.remove(&my_rank);
    let neighbors: Vec<usize> = nb.iter().copied().collect();

    // Build wire buffers (checks for unresolved mappings)
    let wires = build_wires(mesh, overlap, &neighbors)?;

    if comm.is_no_comm() || comm.size() <= 1 || neighbors.is_empty() {
        mesh.invalidate_cache();
        return Ok(());
    }

    let all_neighbors: HashSet<usize> = neighbors.iter().copied().collect();

    // Phase 1: exchange counts
    let counts = exchange_sizes_symmetric(&wires, comm, tags.sizes.as_u16(), &all_neighbors)?;

    // Phase 2: exchange payloads
    let mut recv_payloads = Vec::new();
    for &nbr in &neighbors {
        let n_items = counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![WireArrow::zeroed(); n_items];
        let h = comm.irecv(nbr, tags.data.as_u16(), cast_slice_mut(&mut buf));
        recv_payloads.push((nbr, h, buf));
    }

    let mut pending_sends = Vec::new();
    for &nbr in &neighbors {
        let out = wires.get(&nbr).map_or(&[][..], |v| &v[..]);
        pending_sends.push(comm.isend(nbr, tags.data.as_u16(), cast_slice(out)));
    }

    let mut maybe_err: Option<MeshSieveError> = None;
    for (nbr, h, mut buf) in recv_payloads {
        match h.wait() {
            Some(raw) if raw.len() == buf.len() * core::mem::size_of::<WireArrow>() => {
                cast_slice_mut(&mut buf).copy_from_slice(&raw);
                for w in &buf {
                    let src = PointId::new(w.src())
                        .map_err(|e| MeshSieveError::MeshError(Box::new(e)))?;
                    let dst = PointId::new(w.dst())
                        .map_err(|e| MeshSieveError::MeshError(Box::new(e)))?;
                    mesh.add_arrow(src, dst, S::Payload::default());
                }
            }
            Some(raw) if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!(
                        "payload size mismatch: expected {}B, got {}B",
                        buf.len() * core::mem::size_of::<WireArrow>(),
                        raw.len()
                    )
                    .into(),
                });
            }
            None if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: "recv returned None".into(),
                });
            }
            _ => {}
        }
    }

    for s in pending_sends {
        let _ = s.wait();
    }

    mesh.invalidate_cache();

    if let Some(e) = maybe_err {
        Err(e)
    } else {
        Ok(())
    }
}

/// Convenience wrapper using a legacy default tag (0xC0DE).
pub fn complete_sieve<S, C>(
    mesh: &mut S,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
    S::Payload: Default + Clone + Send + 'static,
    C: Communicator + Sync,
{
    let tags = SieveCommTags::from_base(CommTag::new(0xC0DE));
    complete_sieve_with_tags(mesh, overlap, comm, my_rank, tags)
}

/// Iteratively complete the sieve until no new points or arrows are added.
pub fn complete_sieve_until_converged<S, C>(
    mesh: &mut S,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId>,
    S::Payload: Default + Clone + Send + 'static,
    C: Communicator + Sync,
{
    let mut prev = std::collections::HashSet::new();
    loop {
        let before: std::collections::HashSet<_> = mesh.points().collect();
        complete_sieve(mesh, overlap, comm, my_rank)?;
        let after: std::collections::HashSet<_> = mesh.points().collect();
        if after == before || after == prev {
            break;
        }
        prev = after.clone();
        mesh.invalidate_cache();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algs::communicator::NoComm;

    #[test]
    fn unresolved_mapping_errors() {
        let mut mesh = crate::topology::sieve::InMemorySieve::<PointId, ()>::default();
        mesh.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
        let mut ovlp = Overlap::new();
        ovlp.add_link_structural_one(PointId::new(1).unwrap(), 1);
        ovlp.add_link_structural_one(PointId::new(2).unwrap(), 1);
        let comm = NoComm;
        let tags = SieveCommTags::from_base(CommTag::new(0x4200));
        let res = complete_sieve_with_tags(&mut mesh, &ovlp, &comm, 0, tags);
        assert!(matches!(res, Err(MeshSieveError::MissingOverlap { .. })));
    }
}

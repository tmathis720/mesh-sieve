//! Complete missing sieve arrows across ranks using minimal wire arrows.
//!
//! This module synchronizes sieve structure between distributed ranks by
//! exchanging only `(src,dst)` pairs already translated into the receiver's
//! local `PointId` space.  The protocol mirrors the section completion: a
//! symmetric two-phase exchange (sizes then data) tagged via [`SieveCommTags`].

use std::collections::{BTreeSet, HashMap, HashSet};

use bytemuck::{Zeroable, cast_slice, cast_slice_mut};

use crate::algs::communicator::{CommTag, Communicator, SieveCommTags, Wait};
use crate::algs::completion::size_exchange::exchange_sizes_symmetric;
use crate::algs::wire::WireArrow;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use crate::topology::sieve::sieve_trait::Sieve;

/// Translate a local `PointId` to the neighbor's `PointId` via `Overlap`.
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

/// Build per-neighbor wire buffers in the receiver's ID space.
fn build_wires<S>(
    mesh: &S,
    overlap: &Overlap,
    neighbors: &[usize],
) -> Result<HashMap<usize, Vec<WireArrow>>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
{
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
    S: Sieve<Point = PointId> + InvalidateCache,
    S::Payload: Default + Clone + Send + 'static,
    C: Communicator + Sync,
{
    #[cfg(any(debug_assertions, feature = "check-invariants"))]
    overlap.validate_invariants()?;
    if comm.is_no_comm() || comm.size() <= 1 {
        mesh.invalidate_cache();
        return Ok(());
    }

    // Deterministic neighbor set (excluding self)
    let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
    nb.remove(&my_rank);
    let neighbors: Vec<usize> = nb.into_iter().collect();
    if neighbors.is_empty() {
        mesh.invalidate_cache();
        return Ok(());
    }
    let all_neighbors: HashSet<usize> = neighbors.iter().copied().collect();

    // Build wire buffers per neighbor
    let wires = build_wires(mesh, overlap, &neighbors)?;

    // Phase 1: symmetric exchange of counts
    let counts = exchange_sizes_symmetric(&wires, comm, tags.sizes, &all_neighbors)?;

    // Phase 2: payload exchange
    let mut recvs = Vec::new();
    for &nbr in &neighbors {
        let n = counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![WireArrow::zeroed(); n];
        let h = comm.irecv(nbr, tags.data.as_u16(), cast_slice_mut(&mut buf));
        recvs.push((nbr, h, buf));
    }

    let mut sends = Vec::new();
    for &nbr in &neighbors {
        let out = wires.get(&nbr).map_or(&[][..], |v| &v[..]);
        sends.push(comm.isend(nbr, tags.data.as_u16(), cast_slice(out)));
    }

    let mut maybe_err: Option<MeshSieveError> = None;
    for (nbr, h, mut buf) in recvs {
        match h.wait() {
            Some(raw) if raw.len() == buf.len() * std::mem::size_of::<WireArrow>() => {
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
                let exp = buf.len() * std::mem::size_of::<WireArrow>();
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("payload size mismatch: expected {exp}B, got {}B", raw.len())
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

    for h in sends {
        let _ = h.wait();
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
    S: Sieve<Point = PointId> + InvalidateCache,
    S::Payload: Default + Clone + Send + 'static,
    C: Communicator + Sync,
{
    let tags = SieveCommTags::from_base(CommTag::new(0xC0DE));
    complete_sieve_with_tags(mesh, overlap, comm, my_rank, tags)
}

/// Iteratively completes the sieve until no new points/arrows are added.
pub fn complete_sieve_until_converged<S, C>(
    sieve: &mut S,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    S: Sieve<Point = PointId> + InvalidateCache,
    S::Payload: Default + Clone + Send + 'static,
    C: Communicator + Sync,
{
    let mut prev = std::collections::HashSet::new();
    loop {
        let before: std::collections::HashSet<_> = sieve.points().collect();
        complete_sieve(sieve, overlap, comm, my_rank)?;
        let after: std::collections::HashSet<_> = sieve.points().collect();
        if after == before || after == prev {
            break;
        }
        prev = after.clone();
        sieve.invalidate_cache();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algs::communicator::Communicator;
    use crate::overlap::overlap::Overlap;
    use crate::topology::sieve::InMemorySieve;

    #[test]
    fn unresolved_mapping_errors() {
        // Dummy communicator that claims two ranks but performs no I/O.
        struct DummyComm;
        impl Communicator for DummyComm {
            type SendHandle = ();
            type RecvHandle = ();
            fn isend(&self, _peer: usize, _tag: u16, _buf: &[u8]) -> Self::SendHandle {}
            fn irecv(&self, _peer: usize, _tag: u16, _buf: &mut [u8]) -> Self::RecvHandle {}
            fn rank(&self) -> usize {
                0
            }
            fn size(&self) -> usize {
                2
            }
        }

        let mut sieve: InMemorySieve<PointId, ()> = InMemorySieve::default();
        let mut ovlp = Overlap::new();
        ovlp.add_link_structural_one(PointId::new(1).unwrap(), 1); // unresolved
        let comm = DummyComm;
        let tags = SieveCommTags::from_base(CommTag::new(0x5100));
        let res = complete_sieve_with_tags(&mut sieve, &ovlp, &comm, 0, tags);
        assert!(matches!(res, Err(MeshSieveError::MissingOverlap { .. })));
    }
}

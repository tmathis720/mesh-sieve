//! Complete missing sieve arrows across ranks using fixed wire triples.
//!
//! This module provides routines for synchronizing and completing sieve arrows
//! across distributed ranks, using packed wire triples for efficient communication.
//! It supports iterative completion until convergence and ensures DAG invariants.

use std::collections::HashMap;

use bytemuck::Zeroable;

use crate::algs::communicator::Wait;
use crate::algs::wire::{WireArrowTriple, WireCount};
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{local, OvlId, Remote};
use crate::prelude::{Communicator, Overlap};
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use crate::topology::sieve::InMemorySieve;
use crate::topology::sieve::sieve_trait::Sieve;

/// Complete missing sieve arrows across ranks.
///
/// Exchanges arrow‐counts and arrow‐payloads in two symmetric phases,
/// always waiting on every nonblocking send/recv before returning.
pub fn complete_sieve<C: Communicator>(
    sieve: &mut InMemorySieve<PointId, Remote>,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError> {
    const BASE_TAG: u16 = 0xC0DE;

    // 0) nothing to do for serial / single‐rank
    if comm.is_no_comm() || comm.size() <= 1 {
        sieve.strata.take();
        return Ok(());
    }

    // 1) Who needs which arrows?
    let mut nb_links: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    for (&p, outs) in &sieve.adjacency_out {
        for (_d, _) in outs {
            for (_d2, rem) in overlap.cone(local(p)) {
                if rem.rank != my_rank {
                    nb_links
                        .entry(rem.rank)
                        .or_default()
                        .push((p, rem.remote_point.expect("overlap unresolved")));
                }
            }
        }
    }
    if nb_links.is_empty() {
        let me_pt = Overlap::partition_node_id(my_rank);
        for (src, rem) in overlap.support(me_pt) {
            if rem.rank != my_rank {
                if let OvlId::Local(src_pt) = src {
                    nb_links
                        .entry(rem.rank)
                        .or_default()
                        .push((rem.remote_point.expect("overlap unresolved"), src_pt));
                }
            }
        }
    }

    // Peers to talk to
    let peers: Vec<usize> = (0..comm.size()).filter(|&r| r != my_rank).collect();

    // We'll accumulate all send‐handles here (phase1 + phase2)
    let mut pending_sends: Vec<C::SendHandle> = Vec::new();
    // And collect any error we see, but still drain all handles before returning
    let mut maybe_err: Option<MeshSieveError> = None;

    // --- Phase 1: exchange counts ---------------------------------------
    // 1a) post all receives for counts
    let mut size_recvs: Vec<(usize, C::RecvHandle, WireCount)> =
        Vec::with_capacity(peers.len());
    for &peer in &peers {
        let mut cnt = WireCount::new(0);
        let h = comm.irecv(peer, BASE_TAG, bytemuck::cast_slice_mut(std::slice::from_mut(&mut cnt)));
        size_recvs.push((peer, h, cnt));
    }
    // 1b) post all sends for counts
    for &peer in &peers {
        let cnt = WireCount::new(nb_links.get(&peer).map(|v| v.len()).unwrap_or(0));
        pending_sends.push(comm.isend(peer, BASE_TAG, bytemuck::cast_slice(std::slice::from_ref(&cnt))));
    }
    // 1c) wait for all count‐recvs
    let mut sizes_in: HashMap<usize, usize> = HashMap::new();
    for (peer, h, mut cnt) in size_recvs {
        match h.wait() {
            Some(data) if data.len() == std::mem::size_of::<WireCount>() => {
                let bytes = bytemuck::cast_slice_mut(std::slice::from_mut(&mut cnt));
                bytes.copy_from_slice(&data);
                sizes_in.insert(peer, cnt.get());
            }
            Some(data) => {
                maybe_err.get_or_insert_with(|| MeshSieveError::CommError {
                    neighbor: peer,
                    source: Box::new(crate::mesh_error::CommError(format!(
                        "expected {} bytes for size from {}, got {}",
                        std::mem::size_of::<WireCount>(),
                        peer,
                        data.len()
                    ))),
                });
            }
            None => {
                maybe_err.get_or_insert_with(|| MeshSieveError::CommError {
                    neighbor: peer,
                    source: Box::new(crate::mesh_error::CommError(format!(
                        "failed to recv size from {}",
                        peer
                    ))),
                });
            }
        }
    }

    // --- Phase 2: exchange actual WireArrowTriple payloads -------------------
    // 2a) post all receives for triples
    let mut data_recvs: Vec<(usize, C::RecvHandle, Vec<WireArrowTriple>)> =
        Vec::with_capacity(peers.len());
    for &peer in &peers {
        let n = *sizes_in.get(&peer).unwrap_or(&0);
        let mut buffer = vec![WireArrowTriple::zeroed(); n];
        let bytes = bytemuck::cast_slice_mut(buffer.as_mut_slice());
        let h = comm.irecv(peer, BASE_TAG + 1, bytes);
        data_recvs.push((peer, h, buffer));
    }
    // 2b) post all sends of our triples
    for &peer in &peers {
        let mut triples = Vec::new();
        if let Some(links) = nb_links.get(&peer) {
            for &(src, _) in links {
                if let Some(outs) = sieve.adjacency_out.get(&src) {
                    for (d, payload) in outs {
                        triples.push(WireArrowTriple::new(
                            src.get(),
                            d.get(),
                            payload.remote_point.expect("overlap unresolved").get(),
                            payload.rank as u32,
                        ));
                    }
                }
            }
        }
        let bytes = bytemuck::cast_slice(&triples);
        // always post a send, even if empty
        pending_sends.push(comm.isend(peer, BASE_TAG + 1, bytes));
    }

    // 3) wait + integrate all triple‐recvs
    let mut inserted = std::collections::HashSet::new();
    for (peer, h, mut buffer) in data_recvs {
        match h.wait() {
            Some(raw)
                if raw.len() == buffer.len() * std::mem::size_of::<WireArrowTriple>() =>
            {
                let view = bytemuck::cast_slice_mut(buffer.as_mut_slice());
                view.copy_from_slice(&raw);
                for t in &buffer {
                    let (src, dst, remote_point, rank) = t.decode();
                    match (
                        PointId::new(src),
                        PointId::new(dst),
                        PointId::new(remote_point),
                    ) {
                        (Ok(src_pt), Ok(dst_pt), Ok(rem_pt)) => {
                            let payload = Remote {
                                rank: rank as usize,
                                remote_point: Some(rem_pt),
                            };
                            if inserted.insert((src_pt, dst_pt)) {
                                sieve
                                    .adjacency_out
                                    .entry(src_pt)
                                    .or_default()
                                    .push((dst_pt, payload));
                                sieve
                                    .adjacency_in
                                    .entry(dst_pt)
                                    .or_default()
                                    .push((src_pt, payload));
                            }
                        }
                        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                            maybe_err.get_or_insert_with(|| MeshSieveError::MeshError(Box::new(e)));
                        }
                    }
                }
            }
            Some(raw) => {
                maybe_err.get_or_insert_with(|| MeshSieveError::CommError {
                    neighbor: peer,
                    source: Box::new(crate::mesh_error::CommError(format!(
                        "expected {} bytes for triples from {}, got {}",
                        buffer.len() * std::mem::size_of::<WireArrowTriple>(),
                        peer,
                        raw.len()
                    ))),
                });
            }
            None => {
                maybe_err.get_or_insert_with(|| MeshSieveError::CommError {
                    neighbor: peer,
                    source: Box::new(crate::mesh_error::CommError(format!(
                        "failed to recv triples from {}",
                        peer
                    ))),
                });
            }
        }
    }

    // Invalidate cached strata
    sieve.strata.take();

    // 4) always drain all sends
    for send in pending_sends {
        let _ = send.wait();
    }

    // 5) finally, propagate error or success
    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(())
    }
}

/// Iteratively completes the sieve until no new points/arrows are added.
pub fn complete_sieve_until_converged(
    sieve: &mut InMemorySieve<PointId, Remote>,
    overlap: &Overlap,
    comm: &impl Communicator,
    my_rank: usize,
) -> Result<(), MeshSieveError> {
    let mut prev = std::collections::HashSet::new();
    loop {
        let before: std::collections::HashSet<_> = sieve.points().collect();
        complete_sieve(sieve, overlap, comm, my_rank)?;
        let after: std::collections::HashSet<_> = sieve.points().collect();
        if after == before || after == prev {
            break;
        }
        prev = after.clone();
        InvalidateCache::invalidate_cache(sieve);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // TODO: add tests for complete_sieve with a mock Communicator
}

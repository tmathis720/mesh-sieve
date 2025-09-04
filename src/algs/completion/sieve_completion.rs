//! Complete missing sieve arrows across ranks by packing WireTriple.
//!
//! This module provides routines for synchronizing and completing sieve arrows
//! across distributed ranks, using packed wire triples for efficient communication.
//! It supports iterative completion until convergence and ensures DAG invariants.

use std::collections::HashMap;

use bytemuck::Zeroable;

use crate::algs::communicator::Wait;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{local, OvlId, Remote};
use crate::prelude::{Communicator, Overlap};
use crate::topology::point::PointId;
use crate::topology::sieve::sieve_trait::Sieve;
use crate::topology::sieve::InMemorySieve;
use crate::topology::cache::InvalidateCache;

/// Packed arrow for network transport.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireTriple {
    src: u64,
    dst: u64,
    remote_point: u64,
    rank: usize,
}

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
                        .push((p, rem.remote_point));
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
                        .push((rem.remote_point, src_pt));
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
    let mut size_recvs: Vec<(usize, C::RecvHandle, [u8; 4])> = Vec::with_capacity(peers.len());
    for &peer in &peers {
        let buf = [0u8; 4];
        let h = comm.irecv(peer, BASE_TAG, &mut buf.clone());
        size_recvs.push((peer, h, buf));
    }
    // 1b) post all sends for counts
    for &peer in &peers {
        let cnt = nb_links.get(&peer).map(|v| v.len()).unwrap_or(0) as u32;
        let buf = cnt.to_le_bytes();
        pending_sends.push(comm.isend(peer, BASE_TAG, &buf));
    }
    // 1c) wait for all count‐recvs
    let mut sizes_in: HashMap<usize, usize> = HashMap::new();
    for (peer, h, mut buf) in size_recvs {
        match h.wait() {
            Some(data) if data.len() == buf.len() => {
                buf.copy_from_slice(&data);
                sizes_in.insert(peer, u32::from_le_bytes(buf) as usize);
            }
            Some(data) => {
                maybe_err.get_or_insert_with(|| MeshSieveError::CommError {
                    neighbor: peer,
                    source: Box::new(crate::mesh_error::CommError(format!(
                        "expected 4 bytes for size from {}, got {}",
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

    // --- Phase 2: exchange actual WireTriple payloads -------------------
    // 2a) post all receives for triples
    let mut data_recvs: Vec<(usize, C::RecvHandle, Vec<WireTriple>)> =
        Vec::with_capacity(peers.len());
    for &peer in &peers {
        let n = *sizes_in.get(&peer).unwrap_or(&0);
        let mut buffer = vec![WireTriple::zeroed(); n];
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
                        triples.push(WireTriple {
                            src: src.get(),
                            dst: d.get(),
                            remote_point: payload.remote_point.get(),
                            rank: payload.rank,
                        });
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
            Some(raw) if raw.len() == buffer.len() * std::mem::size_of::<WireTriple>() => {
                let view = bytemuck::cast_slice_mut(buffer.as_mut_slice());
                view.copy_from_slice(&raw);
                for &WireTriple {
                    src,
                    dst,
                    remote_point,
                    rank,
                } in &buffer
                {
                    match (
                        PointId::new(src),
                        PointId::new(dst),
                        PointId::new(remote_point),
                    ) {
                        (Ok(src_pt), Ok(dst_pt), Ok(rem_pt)) => {
                            let payload = Remote {
                                rank,
                                remote_point: rem_pt,
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
                        buffer.len() * std::mem::size_of::<WireTriple>(),
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

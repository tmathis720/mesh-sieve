//! Complete missing sieve arrows across ranks by packing WireTriple.
//!
//! This module provides routines for synchronizing and completing sieve arrows
//! across distributed ranks, using packed wire triples for efficient communication.
//! It supports iterative completion until convergence and ensures DAG invariants.

use std::collections::HashMap;

use crate::algs::communicator::Wait;
use crate::algs::completion::partition_point;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Remote;
use crate::prelude::{Communicator, Overlap};
use crate::topology::point::PointId;
use crate::topology::sieve::{Sieve, InMemorySieve};
use crate::topology::stratum::InvalidateCache;
use bytemuck::Zeroable;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireTriple {
    /// Source point (as u64).
    src: u64,
    /// Destination point (as u64).
    dst: u64,
    /// Remote rank.
    rank: usize,
}

/// Complete missing sieve arrows across ranks.
///
/// This function exchanges and integrates arrows between ranks using the provided communicator,
/// updating the sieve with any missing arrows discovered via overlap links.
///
/// # Arguments
/// - `sieve`: The in-memory sieve to complete.
/// - `overlap`: The overlap structure describing remote relationships.
/// - `comm`: The communicator for message passing.
/// - `my_rank`: The current rank.
///
/// # Side Effects
/// Modifies the sieve in place and ensures DAG invariants.
pub fn complete_sieve<C: Communicator>(
    sieve: &mut InMemorySieve<PointId, Remote>,
    overlap: &Overlap,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError> {
    const BASE_TAG: u16 = 0xC0DE;

    // --- EARLY RETURN FOR SERIAL / NOCOMM -------------------------------
    // In pure‐serial tests, or if we only have one rank, there's nothing
    // to exchange.
    if comm.is_no_comm() || comm.size() <= 1 {
        sieve.strata.take();
        return Ok(());
    }

    // 1. Build the "who–needs–what" table: nb_links[peer] = Vec<(src, dst)>
    // ---------------------------------------------------------------------
    let mut nb_links: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();

    // Any arrows we *own* that somebody else references
    for (&p, outs) in &sieve.adjacency_out {
        for (_dst, _payload) in outs {
            for (_dst2, rem) in overlap.cone(p) {
                if rem.rank != my_rank {
                    nb_links.entry(rem.rank)
                            .or_default()
                            .push((p, rem.remote_point));
                }
            }
        }
    }

    // If we have no owned arrows, we may still have remote points that
    // other ranks own but point *into* us -------------------------------⇣
    if nb_links.is_empty() {
        let me_pt = partition_point(my_rank);
        for (src, rem) in overlap.support(me_pt) {
            if rem.rank != my_rank {
                nb_links.entry(rem.rank)
                        .or_default()
                        .push((rem.remote_point, src));
            }
        }
    }

    // ---------------------------------------------------------------------
    // 2. Symmetric two-phase exchange with *every* other rank
    // ---------------------------------------------------------------------
    let peers: Vec<usize> = (0..comm.size()).filter(|&r| r != my_rank).collect();

    // ---- Phase 1: exchange the counts ------------------------------------
    let mut size_recvs = HashMap::new();
    for &peer in &peers {
        // always post the receive
        let mut buf = [0u8; 4];
        let h = comm.irecv(peer, BASE_TAG, &mut buf);
        size_recvs.insert(peer, (h, buf));
    }

    // non-blocking sends (store handles so we can wait later)
    let mut pending_sends: Vec<C::SendHandle> = Vec::new();
    for &peer in &peers {
        let cnt = nb_links.get(&peer).map(|v| v.len()).unwrap_or(0) as u32;
        pending_sends.push(comm.isend(peer, BASE_TAG, &cnt.to_le_bytes()));
    }

    // collect the sizes we just received
    let mut sizes_in = HashMap::new();
    for (peer, (h, mut buf)) in size_recvs {
        let data = h
            .wait()
            .ok_or_else(|| MeshSieveError::CommError {
                neighbor: peer,
                source: Box::new(crate::mesh_error::CommError(format!("failed to recv size from {peer}"))),
            })?;
        buf.copy_from_slice(&data);
        sizes_in.insert(peer, u32::from_le_bytes(buf) as usize);
    }

    // ---- Phase 2: exchange the actual payloads ---------------------------
    // We need an *aligned* buffer of WireTriple
    let mut data_recvs = HashMap::new();
    for &peer in &peers {
        let n = sizes_in[&peer];
        // Allocate n WireTriples (zeroed for Pod safety)
        let mut buffer: Vec<WireTriple> = vec![WireTriple::zeroed(); n];
        // View as &mut [u8] for MPI recv
        let bytes = bytemuck::cast_slice_mut(buffer.as_mut_slice());
        let h = comm.irecv(peer, BASE_TAG + 1, bytes);
        data_recvs.insert(peer, (h, buffer));
    }

    for (&peer, links) in &nb_links {
        // build the triples we owe this peer
        let mut triples = Vec::with_capacity(links.len());
        for &(src, _dst) in links {
            if let Some(outs) = sieve.adjacency_out.get(&src) {
                for (d, payload) in outs {
                    triples.push(WireTriple {
                        src: src.get(),
                        dst: d.get(),
                        rank: payload.rank,
                    });
                }
            }
        }
        let bytes = bytemuck::cast_slice(&triples);
        pending_sends.push(comm.isend(peer, BASE_TAG + 1, bytes));
    }
    // peers with no links still need the empty message:
    for &peer in &peers {
        if !nb_links.contains_key(&peer) {
            pending_sends.push(comm.isend(peer, BASE_TAG + 1, &[]));
        }
    }

    // ---------------------------------------------------------------------
    // 3. Integrate everything we have received
    // ---------------------------------------------------------------------
    let mut inserted = std::collections::HashSet::new();
    for (peer, (h, mut buffer)) in data_recvs {
        let raw = h.wait().ok_or_else(|| {
            MeshSieveError::CommError {
                neighbor: peer,
                source: Box::new(crate::mesh_error::CommError(format!("failed to recv triples from {peer}"))),
            }
        })?;
        // raw is &[u8]; copy into our aligned [WireTriple]
        let bytes = bytemuck::cast_slice_mut(buffer.as_mut_slice());
        bytes.copy_from_slice(&raw);

        // Now buffer is properly aligned so we can view it as WireTriple
        for &WireTriple { src, dst, rank } in &buffer {
            let src_pt = PointId::new(src)?;
            let dst_pt = PointId::new(dst)?;
            let payload = Remote {
                rank,
                remote_point: dst_pt,
            };
            if inserted.insert((src_pt, dst_pt)) {
                sieve.adjacency_out.entry(src_pt).or_default().push((dst_pt, payload));
                sieve.adjacency_in.entry(dst_pt).or_default().push((src_pt, payload));
            }
        }
    }

    sieve.strata.take(); // invalidate cached strata

    // ---------------------------------------------------------------------
    // 4. Make *sure* every non-blocking send is complete
    // ---------------------------------------------------------------------
    for h in pending_sends {
        h.wait();                   // () for NoComm / Rayon; real wait for MPI
    }
    Ok(())
}


/// Iteratively completes the sieve until no new points/arrows are added.
///
/// This function repeatedly calls [`complete_sieve`] until convergence,
/// ensuring the sieve is fully synchronized across all ranks.
///
/// # Arguments
/// - `sieve`: The in-memory sieve to complete.
/// - `overlap`: The overlap structure.
/// - `comm`: The communicator.
/// - `my_rank`: The current rank.
pub fn complete_sieve_until_converged(
    sieve: &mut crate::topology::sieve::InMemorySieve<
        crate::topology::point::PointId,
        crate::overlap::overlap::Remote,
    >,
    overlap: &crate::overlap::overlap::Overlap,
    comm: &impl crate::algs::communicator::Communicator,
    my_rank: usize,
) -> Result<(), crate::mesh_error::MeshSieveError> {
    let mut prev_points = std::collections::HashSet::new();
    loop {
        let before: std::collections::HashSet<_> = sieve.points().collect();
        complete_sieve(sieve, overlap, comm, my_rank)?;
        let after: std::collections::HashSet<_> = sieve.points().collect();
        if after == before || after == prev_points {
            break;
        }
        prev_points = after.clone();
        InvalidateCache::invalidate_cache(sieve);
    }
    Ok(())
}

// Optionally, add #[cfg(test)] mod tests for sieve completion

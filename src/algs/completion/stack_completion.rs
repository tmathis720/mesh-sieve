//! Complete the vertical‐stack arrows (mirror of section completion).
//!
//! This module provides routines for completing stack arrows in a distributed mesh,
//! mirroring section completion logic. It handles symmetric communication of stack
//! data between ranks using the [`Communicator`] trait.

use std::collections::HashMap;
use crate::algs::communicator::Wait;
use crate::algs::wire::{WireCount, WireStackTriple, WIRE_PAYLOAD_MAX};
use crate::topology::sieve::sieve_trait::Sieve;

/// Trait for extracting rank from overlap payloads.
pub trait HasRank {
    /// Returns the MPI rank associated with this payload.
    fn rank(&self) -> usize;
}

impl HasRank for crate::overlap::overlap::Remote {
    fn rank(&self) -> usize { self.rank }
}

/// Complete the stack by exchanging arrows with all neighbor ranks.
///
/// This function performs symmetric communication of stack data, exchanging
/// arrow counts and payloads with all ranks except `my_rank`. It uses the
/// provided communicator for non-blocking send/receive operations.
///
/// # Type Parameters
/// - `P`: Base point type (must be POD, hashable, etc.).
/// - `Q`: Cap point type (must be POD, hashable, etc.).
/// - `Pay`: Payload type (must be POD).
/// - `C`: Communicator type.
/// - `S`: Stack type.
/// - `O`: Overlap sieve type.
/// - `R`: Overlap payload type (must implement [`HasRank`]).
///
/// # Arguments
/// - `stack`: The stack to complete.
/// - `overlap`: The overlap sieve describing sharing relationships.
/// - `comm`: The communicator for message passing.
/// - `my_rank`: The current MPI rank.
/// - `n_ranks`: Total number of ranks.
pub fn complete_stack<P, Q, Pay, C, S, O, R>(
    stack: &mut S,
    overlap: &O,
    comm: &C,
    my_rank: usize,
    n_ranks: usize,
) -> Result<(), crate::mesh_error::MeshSieveError>
where
    P: Copy + bytemuck::Pod + bytemuck::Zeroable + Default + Eq + std::hash::Hash + Send + 'static,
    Q: Copy + bytemuck::Pod + bytemuck::Zeroable + Default + Eq + std::hash::Hash + Send + 'static,
    Pay: Copy + bytemuck::Pod + bytemuck::Zeroable + Default + PartialEq + Send + 'static,
    C: crate::algs::communicator::Communicator + Sync,
    S: crate::topology::stack::Stack<Point = P, CapPt = Q, Payload = Pay>,
    O: crate::topology::sieve::Sieve<Point = P, Payload = R> + Sync,
    R: HasRank + Copy + Send + 'static,
{
    const BASE_TAG: u16 = 0xC0DE;
    assert!(std::mem::size_of::<P>() == 8);
    assert!(std::mem::size_of::<Q>() == 8);
    assert!(std::mem::size_of::<Pay>() <= WIRE_PAYLOAD_MAX);
    // 1. Find all neighbors (ranks) to communicate with
    let mut nb_links: HashMap<usize, Vec<(P, Q, Pay)>> = HashMap::new();
    // Only treat as owned if the base point has at least one non-default payload
    for base in stack.base().base_points() {
        let mut has_owned = false;
        let mut owned_caps = Vec::new();
        for (cap, pay) in stack.lift(base) {
            if pay != Pay::default() {
                has_owned = true;
                owned_caps.push((cap, pay));
            }
        }
        if !has_owned {
            continue;
        }
        for (cap, pay) in owned_caps {
            for (_dst, rem) in overlap.cone(base) {
                if rem.rank() != my_rank {
                    nb_links.entry(rem.rank())
                        .or_default()
                        .push((base, cap, pay));
                }
            }
        }
    }
    // --- DEADLOCK FIX: ensure symmetric communication ---
    // Use all ranks except my_rank as neighbors
    let all_neighbors: std::collections::HashSet<usize> = (0..n_ranks).filter(|&r| r != my_rank).collect();
    // 2. Exchange sizes (always post send/recv for all neighbors)
    let mut recv_size = HashMap::new();
    for &nbr in &all_neighbors {
        let mut cnt = WireCount::new(0);
        let h = comm.irecv(nbr, BASE_TAG, bytemuck::cast_slice_mut(std::slice::from_mut(&mut cnt)));
        recv_size.insert(nbr, (h, cnt));
    }
    let mut pending_size_sends = Vec::with_capacity(all_neighbors.len());
    let mut size_send_bufs   = Vec::with_capacity(all_neighbors.len());
    for &nbr in &all_neighbors {
        let count = WireCount::new(nb_links.get(&nbr).map_or(0, |v| v.len()));
        let h     = comm.isend(nbr, BASE_TAG, bytemuck::cast_slice(std::slice::from_ref(&count)));
        pending_size_sends.push(h);
        size_send_bufs.push(count);
    }
    let mut sizes_in = HashMap::new();
    for (nbr, (h, mut cnt)) in recv_size {
        let data = h.wait().ok_or_else(|| crate::mesh_error::CommError(format!("failed to receive size from rank {nbr}")))?;
        let bytes = bytemuck::cast_slice_mut(std::slice::from_mut(&mut cnt));
        bytes.copy_from_slice(&data);
        sizes_in.insert(nbr, cnt.get());
    }
    for send_h in pending_size_sends {
        let _ = send_h.wait();
    }
    // 3. Exchange data (always post send/recv for all neighbors)
    use bytemuck::{bytes_of, bytes_of_mut, cast_slice, cast_slice_mut, Zeroable};
    let mut recv_data = HashMap::new();
    for &nbr in &all_neighbors {
        let n_items = sizes_in.get(&nbr).copied().unwrap_or(0);
        let mut buf = vec![WireStackTriple::zeroed(); n_items];
        let h = comm.irecv(nbr, BASE_TAG + 1, cast_slice_mut(buf.as_mut_slice()));
        recv_data.insert(nbr, (h, buf));
    }
    let mut pending_data_sends = Vec::with_capacity(all_neighbors.len());
    let mut data_send_bufs     = Vec::with_capacity(all_neighbors.len());
    for &nbr in &all_neighbors {
        let triples = nb_links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let wire: Vec<WireStackTriple> = triples
            .iter()
            .map(|&(b, c, p)| {
                let base_u64: u64 = bytemuck::cast(b);
                let cap_u64: u64 = bytemuck::cast(c);
                WireStackTriple::new(base_u64, cap_u64, bytes_of(&p))
            })
            .collect();
        let bytes = cast_slice(&wire);
        let h = comm.isend(nbr, BASE_TAG + 1, bytes);
        pending_data_sends.push(h);
        data_send_bufs.push(wire);
    }
    for (_nbr, (h, mut buf)) in recv_data {
        let raw = h.wait().ok_or_else(|| crate::mesh_error::CommError("failed to receive stack data".to_string()))?;
        let buf_bytes = cast_slice_mut(buf.as_mut_slice());
        buf_bytes.copy_from_slice(&raw);
        for w in &buf {
            let base: P = bytemuck::cast(w.base());
            let cap: Q = bytemuck::cast(w.cap());
            let mut pay = Pay::zeroed();
            let pay_slice = bytes_of_mut(&mut pay);
            pay_slice.copy_from_slice(&w.pay[..pay_slice.len()]);
            let _ = stack.add_arrow(base, cap, pay);
        }
    }
    for send_h in pending_data_sends {
        let _ = send_h.wait();
    }
    Ok(())
}


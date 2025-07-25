//! Complete the vertical‐stack arrows (mirror of section completion).
//!
//! This module provides routines for completing stack arrows in a distributed mesh,
//! mirroring section completion logic. It handles symmetric communication of stack
//! data between ranks using the [`Communicator`] trait.

use std::collections::HashMap;
use crate::algs::communicator::Wait;
use crate::topology::sieve::sieve_trait::Sieve;

/// A tightly-packed triple of (base, cap, payload).
///
/// Used for efficient serialization of stack arrows.
#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireTriple<P, Q, Pay>
where
    P: Copy + bytemuck::Pod + bytemuck::Zeroable,
    Q: Copy + bytemuck::Pod + bytemuck::Zeroable,
    Pay: Copy + bytemuck::Pod + bytemuck::Zeroable,
{
    base: P,
    cap:  Q,
    pay:  Pay,
}

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
        let mut buf = [0u8; 4];
        let h = comm.irecv(nbr, BASE_TAG, &mut buf);
        recv_size.insert(nbr, (h, buf));
    }
    // --- keep each `to_le_bytes()` alive until we wait on its Request ---
    let mut pending_size_sends = Vec::with_capacity(all_neighbors.len());
    let mut size_send_bufs   = Vec::with_capacity(all_neighbors.len());
    for &nbr in &all_neighbors {
        let count = nb_links.get(&nbr).map_or(0, |v| v.len()) as u32;
        let buf   = count.to_le_bytes();              // stash
        let h     = comm.isend(nbr, BASE_TAG, &buf);  // non‐blocking
        pending_size_sends.push(h);
        size_send_bufs.push(buf);                      // keep it alive
    }
    let mut sizes_in = HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        let data = h.wait().ok_or_else(|| crate::mesh_error::CommError(format!("failed to receive size from rank {}", nbr)))?;
        buf.copy_from_slice(&data);
        sizes_in.insert(nbr, u32::from_le_bytes(buf) as usize);
    }
    // now that everyone’s recv’d their count, we can finish our non-blocking sends
    for send_h in pending_size_sends {
        let _ = send_h.wait();
    }
    // (and now `size_send_bufs` can go out of scope and free its buffers)
    // 3. Exchange data (always post send/recv for all neighbors)
    use bytemuck::cast_slice;
    use bytemuck::cast_slice_mut;
    let mut recv_data = HashMap::new();
    for &nbr in &all_neighbors {
        let n_items = sizes_in.get(&nbr).copied().unwrap_or(0);
        let mut buf = vec![WireTriple::<P, Q, Pay> { base: P::default(), cap: Q::default(), pay: Pay::default() }; n_items];
        let h = comm.irecv(nbr, BASE_TAG + 1, cast_slice_mut(&mut buf));
        recv_data.insert(nbr, (h, buf));
    }
    // --- keep each `wire` Vec alive until its Request completes ---
    let mut pending_data_sends = Vec::with_capacity(all_neighbors.len());
    let mut data_send_bufs     = Vec::with_capacity(all_neighbors.len());
    for &nbr in &all_neighbors {
        let triples = nb_links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let wire: Vec<WireTriple<P, Q, Pay>> = triples.iter()
            .map(|&(b, c, p)| WireTriple { base: b, cap: c, pay: p })
            .collect();
        let bytes = cast_slice(&wire);
        let h = comm.isend(nbr, BASE_TAG + 1, bytes);
        pending_data_sends.push(h);
        data_send_bufs.push(wire);  // stash the Vec so it doesn't drop early
    }
    for (_nbr, (h, mut buf)) in recv_data {
        let raw = h.wait().ok_or_else(|| crate::mesh_error::CommError("failed to receive stack data".to_string()))?;
        let buf_bytes = cast_slice_mut(&mut buf);
        buf_bytes.copy_from_slice(&raw);
        let incoming: &[WireTriple<P,Q,Pay>] = &buf;
        for &WireTriple { base, cap, pay } in incoming {
            let _ = stack.add_arrow(base, cap, pay);
        }
    }
    // now drain those data‐phase sends
    for send_h in pending_data_sends {
        let _ = send_h.wait();
    }
    // (and now `data_send_bufs` can drop safely)
    Ok(())
}


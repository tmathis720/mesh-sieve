//! Complete the vertical‐stack arrows (mirror of section completion).
//!
//! This module provides routines for completing stack arrows in a distributed mesh,
//! mirroring section completion logic. Communication uses explicit [`CommTag`]s and
//! drains all send/receive handles before returning. Neighbor ranks are derived from
//! the provided overlap graph, ensuring only true neighbors participate in the exchange.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::algs::wire::{WirePoint, cast_slice, cast_slice_mut};
use bytemuck::{Pod, Zeroable};

use crate::algs::communicator::{StackCommTags, Wait};
use crate::algs::completion::size_exchange::exchange_sizes_symmetric;
use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::sieve_trait::Sieve;

/// Trait for extracting rank from overlap payloads.
pub trait HasRank {
    /// Returns the MPI rank associated with this payload as a 32‑bit value.
    fn rank_u32(&self) -> u32;
}

impl HasRank for crate::overlap::overlap::Remote {
    #[inline]
    fn rank_u32(&self) -> u32 {
        u32::try_from(self.rank)
            .expect("rank does not fit in u32; increase wire width or cap n_ranks")
    }
}

/// Fixed width `(base, cap, payload)` triple used on the wire.
#[repr(C)]
#[derive(Copy, Clone, Zeroable)]
struct WireTriple64<Pay>
where
    Pay: Copy + Pod + Zeroable,
{
    base_le: u64,
    cap_le: u64,
    pay: Pay,
}

impl<Pay: Copy + Pod + Zeroable> WireTriple64<Pay> {
    fn new(base: u64, cap: u64, pay: Pay) -> Self {
        Self {
            base_le: base.to_le(),
            cap_le: cap.to_le(),
            pay,
        }
    }
}

unsafe impl<Pay: Copy + Pod + Zeroable> Pod for WireTriple64<Pay> {}

/// Complete the stack by exchanging arrows with all true neighbor ranks.
pub fn complete_stack_with_tags<P, Q, Pay, C, S, O, R>(
    stack: &mut S,
    overlap: &O,
    comm: &C,
    my_rank: usize,
    n_ranks: usize,
    tags: StackCommTags,
) -> Result<(), MeshSieveError>
where
    P: WirePoint + Default + Eq + std::hash::Hash + Copy + Send + 'static,
    Q: WirePoint + Default + Eq + std::hash::Hash + Copy + Send + 'static,
    Pay: Copy + Pod + Zeroable + Default + PartialEq + Send + 'static,
    C: crate::algs::communicator::Communicator + Sync,
    S: crate::topology::stack::Stack<Point = P, CapPt = Q, VerticalPayload = Pay>,
    O: Sieve<Point = P, Payload = R> + Sync,
    R: HasRank + Copy + Send + 'static,
{
    if n_ranks == 0 {
        return Err(MeshSieveError::CommError {
            neighbor: my_rank,
            source: "n_ranks must be > 0".into(),
        });
    }

    // 1. Build owned links per neighbor
    let mut nb_links: HashMap<usize, Vec<(P, Q, Pay)>> = HashMap::new();
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
                let r = rem.rank_u32() as usize;
                if r != my_rank {
                    nb_links.entry(r).or_default().push((base, cap, pay));
                }
            }
        }
    }

    // 2. Determine neighbor set and validate ranks.
    //
    // We still inspect the provided overlap to validate any referenced ranks,
    // but for symmetric exchange we communicate with all ranks (except self).
    // This ensures progress even if only one side seeded overlap structure.
    let mut nb_seen: BTreeSet<usize> = BTreeSet::new();
    for p in overlap.base_points() {
        for (_dst, rem) in overlap.cone(p) {
            nb_seen.insert(rem.rank_u32() as usize);
        }
    }
    for &r in &nb_seen {
        if r >= n_ranks {
            return Err(MeshSieveError::CommError {
                neighbor: r,
                source: format!("rank {r} ≥ n_ranks {n_ranks}").into(),
            });
        }
    }
    // Symmetric neighbor set: all ranks except self.
    let neighbors: Vec<usize> = (0..n_ranks).filter(|&r| r != my_rank).collect();
    let all_neighbors: HashSet<usize> = neighbors.iter().copied().collect();

    // 3. Build wire buffers per neighbor
    let mut wires: HashMap<usize, Vec<WireTriple64<Pay>>> = HashMap::new();
    for (&nbr, triples) in nb_links.iter() {
        let mut buf = Vec::with_capacity(triples.len());
        for &(b, c, p) in triples {
            buf.push(WireTriple64::new(b.to_wire(), c.to_wire(), p));
        }
        wires.insert(nbr, buf);
    }

    // 4. Symmetric exchange of counts
    let counts = exchange_sizes_symmetric(&wires, comm, tags.sizes, &all_neighbors)?;

    // 5. Exchange payloads
    let mut recv_data = Vec::new();
    for &nbr in &neighbors {
        let n = counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![WireTriple64::<Pay>::zeroed(); n];
        let h = comm.irecv_result(nbr, tags.data.as_u16(), cast_slice_mut(&mut buf))?;
        recv_data.push((nbr, h, buf));
    }

    let mut pending_sends = Vec::new();
    for &nbr in &neighbors {
        let out = wires.get(&nbr).map_or(&[][..], |v| &v[..]);
        pending_sends.push(comm.isend_result(nbr, tags.data.as_u16(), cast_slice(out))?);
    }

    let mut maybe_err: Option<MeshSieveError> = None;
    for (nbr, h, mut buf) in recv_data {
        match h.wait() {
            Some(raw) if raw.len() == buf.len() * std::mem::size_of::<WireTriple64<Pay>>() => {
                if maybe_err.is_none() {
                    cast_slice_mut(&mut buf).copy_from_slice(&raw);
                    for w in &buf {
                        let b = P::from_wire(u64::from_le(w.base_le));
                        let c = Q::from_wire(u64::from_le(w.cap_le));
                        let _ = stack.add_arrow(b, c, w.pay);
                    }
                }
            }
            Some(raw) if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!(
                        "payload size mismatch: expected {}B, got {}B",
                        buf.len() * std::mem::size_of::<WireTriple64<Pay>>(),
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

    if let Some(e) = maybe_err {
        Err(e)
    } else {
        Ok(())
    }
}

/// Convenience wrapper using a legacy default base tag (0xC0DE).
pub fn complete_stack<P, Q, Pay, C, S, O, R>(
    stack: &mut S,
    overlap: &O,
    comm: &C,
    my_rank: usize,
    n_ranks: usize,
) -> Result<(), MeshSieveError>
where
    P: WirePoint + Default + Eq + std::hash::Hash + Copy + Send + 'static,
    Q: WirePoint + Default + Eq + std::hash::Hash + Copy + Send + 'static,
    Pay: Copy + Pod + Zeroable + Default + PartialEq + Send + 'static,
    C: crate::algs::communicator::Communicator + Sync,
    S: crate::topology::stack::Stack<Point = P, CapPt = Q, VerticalPayload = Pay>,
    O: Sieve<Point = P, Payload = R> + Sync,
    R: HasRank + Copy + Send + 'static,
{
    let base = comm.reserve_tag_range(2)?;
    let tags = StackCommTags::from_base(base);
    complete_stack_with_tags::<P, Q, Pay, C, S, O, R>(stack, overlap, comm, my_rank, n_ranks, tags)
}

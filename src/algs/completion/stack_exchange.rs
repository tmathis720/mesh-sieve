//! Complete the vertical-stack arrows (mirror of section completion).
//!
//! This module provides routines for completing stack arrows in a distributed mesh.
//! It performs symmetric handshakes (sizes then data) with each neighbor and drains
//! all outstanding communication handles before returning.

use crate::algs::communicator::{StackCommTags, Wait};
use crate::algs::completion::size_exchange::exchange_sizes_symmetric;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{local, Overlap};
use crate::topology::sieve::sieve_trait::Sieve;
use std::collections::{BTreeSet, HashMap, HashSet};

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};

/// Bridge trait for converting point identifiers to/from the u64 wire format.
pub trait WirePoint: Copy {
    fn to_wire(self) -> u64;
    fn from_wire(w: u64) -> Self;
}

impl WirePoint for crate::topology::point::PointId {
    #[inline]
    fn to_wire(self) -> u64 {
        self.get()
    }
    #[inline]
    fn from_wire(w: u64) -> Self {
        crate::topology::point::PointId::new(w)
            .expect("invalid PointId on wire")
    }
}

/// Extract an MPI rank in u32 form from overlap payloads.
pub trait HasRank {
    fn rank_u32(&self) -> u32;
}

impl HasRank for crate::overlap::overlap::Remote {
    #[inline]
    fn rank_u32(&self) -> u32 {
        u32::try_from(self.rank)
            .expect("rank does not fit in u32; increase wire width or cap n_ranks")
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct WireTriple64<Pay> {
    base: u64,
    cap: u64,
    pay: Pay,
}

unsafe impl<Pay: Pod> Pod for WireTriple64<Pay> {}
unsafe impl<Pay: Zeroable> Zeroable for WireTriple64<Pay> {}

/// Complete the stack by exchanging arrows with neighbor ranks.
///
/// The communication is fully symmetric and safe for MPI: send buffers are kept
/// alive until their corresponding handles complete, and all receive/send handles
/// are drained even if an error occurs.
#[allow(clippy::too_many_arguments)]
pub fn complete_stack<P, Q, Pay, C, S>(
    stack: &mut S,
    overlap: &Overlap,
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
    S: crate::topology::stack::Stack<Point = P, CapPt = Q, Payload = Pay>,
{
    if n_ranks == 0 || my_rank >= n_ranks {
        return Err(MeshSieveError::CommError {
            neighbor: my_rank,
            source: "invalid rank configuration".into(),
        });
    }

    // 1. Gather send lists per neighbor
    let mut nb_links: HashMap<usize, Vec<(P, Q, Pay)>> = HashMap::new();
    for base in stack.base().base_points() {
        let mut owned_caps = Vec::new();
        for (cap, pay) in stack.lift(base) {
            if pay != Pay::default() {
                owned_caps.push((cap, pay));
            }
        }
        if owned_caps.is_empty() {
            continue;
        }
        for (cap, pay) in owned_caps {
            let base_pid = <crate::topology::point::PointId as WirePoint>::from_wire(base.to_wire());
            for (_dst, rem) in overlap.cone(local(base_pid)) {
                let r = rem.rank_u32() as usize;
                if r != my_rank {
                    nb_links.entry(r).or_default().push((base, cap, pay));
                }
            }
        }
    }

    // 2. Determine deterministic neighbor set from overlap
    let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
    nb.remove(&my_rank);
    for &r in &nb {
        if r >= n_ranks {
            return Err(MeshSieveError::CommError {
                neighbor: r,
                source: format!("rank {} ≥ n_ranks {}", r, n_ranks).into(),
            });
        }
    }
    let neighbors: Vec<usize> = nb.iter().copied().collect();
    let all_neighbors: HashSet<usize> = neighbors.iter().copied().collect();

    // 3. Build wire payloads per neighbor
    let mut wires: HashMap<usize, Vec<WireTriple64<Pay>>> = HashMap::new();
    for (&nbr, triples) in &nb_links {
        let mut buf = Vec::with_capacity(triples.len());
        for &(b, c, p) in triples {
            buf.push(WireTriple64 {
                base: b.to_wire(),
                cap: c.to_wire(),
                pay: p,
            });
        }
        buf.sort_unstable_by_key(|w| (w.base, w.cap));
        buf.dedup_by_key(|w| (w.base, w.cap));
        wires.insert(nbr, buf);
    }

    // 4. Symmetric size exchange
    let counts = exchange_sizes_symmetric(&wires, comm, tags.sizes, &all_neighbors)?;

    // 5. Post all receives for the data phase
    let mut recv_data = Vec::new();
    for &nbr in &neighbors {
        let n = counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![WireTriple64::<Pay>::zeroed(); n];
        let h = comm.irecv(nbr, tags.data.as_u16(), cast_slice_mut(&mut buf));
        recv_data.push((nbr, h, buf));
    }

    // 6. Post all sends and keep buffers alive
    let mut pending_sends = Vec::new();
    let mut keep_alive = Vec::new();
    for &nbr in &neighbors {
        let out = wires.remove(&nbr).unwrap_or_default();
        pending_sends.push(comm.isend(nbr, tags.data.as_u16(), cast_slice(&out)));
        keep_alive.push(out); // keep vectors alive until wait completes
    }

    // 7. Wait for all receives (collect errors but do not early-return)
    let mut maybe_err = None;
    for (nbr, h, mut buf) in recv_data {
        match h.wait() {
            Some(raw)
                if raw.len() == buf.len() * std::mem::size_of::<WireTriple64<Pay>>() =>
            {
                cast_slice_mut(&mut buf).copy_from_slice(&raw);
                for w in &buf {
                    let b = P::from_wire(w.base);
                    let c = Q::from_wire(w.cap);
                    let _ = stack.add_arrow(b, c, w.pay);
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

    // 8. Always drain sends
    for s in pending_sends {
        let _ = s.wait();
    }

    if let Some(e) = maybe_err {
        Err(e)
    } else {
        Ok(())
    }
}


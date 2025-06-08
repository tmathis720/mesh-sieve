//! Section completion (Algorithm 4 in Knepley & Karpeev 2009).
//!
//! Two-phase “sizes then data” exchange so every rank receives ghost copies
//! from owners.
//!
//! 1. Count bytes per neighbour, send/recv sizes (tag = BASE_TAG).
//! 2. Pack values (Delta::restrict), send/recv bulk buffers (tag = BASE_TAG+1).
//! 3. On receive, Delta::fuse() into local Section.

use std::collections::HashMap;
use bytemuck::{Pod, cast_slice};

use crate::data::section::Section;
use crate::algs::communicator::{Communicator, Wait};
use crate::overlap::overlap::Overlap;
use crate::overlap::delta::Delta;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use crate::overlap::overlap::Remote;

/// Marker trait for “bytes can be shipped as-is”.
/// Implemented automatically for Delta::Part when it is `bytemuck::Pod`.
pub trait Wire: Pod + Send + Sync + 'static {}
impl<T: Pod + Send + Sync + 'static> Wire for T {}

/// Map a zero-based MPI rank into a non-zero PointId.
pub fn partition_point(rank: usize) -> PointId {
    PointId::new((rank as u64) + 1)
}

/// Build a map `peer_rank -> Vec<(my_pt, their_pt)>` so that:
/// - On owner ranks, `(my_pt, their_pt)` means "I own `my_pt`, please send it to `their_pt` on peer."
/// - On ghost ranks, `(my_pt, their_pt)` means "I own no data, but I need to receive from peer's `their_pt` into my `my_pt`."
fn neighbour_links<V: Clone + Default>(
    section: &Section<V>,
    ovlp: &Overlap,
    my_rank: usize,
) -> HashMap<usize, Vec<(PointId, PointId)>> {
    let mut out: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    let me_pt = partition_point(my_rank);
    let mut has_owned = false;

    // 1) OWNER-side: for every local point I actually own,
    //    send to every ghost rank that I linked
    for (p, _) in section.iter() {
        has_owned = true;
        // for every outgoing arrow from my mesh-point -> partition_point(peer)
        for (_dst, rem) in ovlp.cone(p) {
            if rem.rank != my_rank {
                out.entry(rem.rank)
                   .or_default()
                   .push((p, rem.remote_point));
            }
        }
    }

    // 2) GHOST-side: for every arrow original owners added into my partition point,
    //    receive from that owner’s `src` into my `rem.remote_point`
    // Always run this for ghost ranks (those with no owned points)
    if !has_owned {
        for (src, rem) in ovlp.support(me_pt) {
            if rem.rank != my_rank {
                out.entry(rem.rank)
                   .or_default()
                   .push((rem.remote_point, src));
            }
        }
    }

    out
}

pub fn complete_section<V, D, C>(
    section: &mut Section<V>,
    overlap: &Overlap,
    comm: &C,
    delta: &D,
    my_rank: usize,
) where
    V: Clone + Default + Send + 'static,
    D: Delta<V> + Send + Sync + 'static,
    D::Part: Pod,
    C: Communicator + Sync,
{
    println!("[rank {}] Entered complete_section", my_rank);
    const BASE_TAG: u16 = 0xBEEF;
    let nb_links = neighbour_links(section, overlap, my_rank);

    // --- Stage 1: exchange sizes ---
    let mut recv_size = HashMap::new();
    for (&nbr, _) in &nb_links {
        println!("[rank {}] Posting irecv for size from {}", my_rank, nbr);
        let buf = [0u8;4];
        let h = comm.irecv(nbr, BASE_TAG, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }
    for (&nbr, links) in &nb_links {
        let count = links.len() as u32;
        println!("[rank {}] Posting isend for size to {} ({} items)", my_rank, nbr, count);
        comm.isend(nbr, BASE_TAG, &count.to_le_bytes());
    }
    let mut sizes_in = HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        println!("[rank {}] Waiting for size from {}", my_rank, nbr);
        let data = h.wait().expect("size receive");
        buf.copy_from_slice(&data);
        println!("[rank {}] Received size from {}: {:?}", my_rank, nbr, buf);
        sizes_in.insert(nbr, u32::from_le_bytes(buf));
    }

    // --- Stage 2: exchange data ---
    let mut recv_data = HashMap::new();
    for (&nbr, _links) in &nb_links {
        let n_items = sizes_in[&nbr] as usize;
        let mut buffer = vec![0u8; n_items * std::mem::size_of::<D::Part>()];
        println!("[rank {}] Posting irecv for data from {} ({} items)", my_rank, nbr, n_items);
        let h = comm.irecv(nbr, BASE_TAG+1, &mut buffer);
        recv_data.insert(nbr, (h, buffer));
    }
    for (&nbr, links) in &nb_links {
        let mut scratch = Vec::with_capacity(links.len());
        for &(loc, _) in links {
            let slice = section.restrict(loc);
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        println!("[rank {}] Posting isend for data to {} ({} items)", my_rank, nbr, links.len());
        comm.isend(nbr, BASE_TAG+1, bytes);
    }
    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().expect("data receive");
        buffer.copy_from_slice(&raw);
        let parts: &[D::Part] = cast_slice(&buffer);
        let links = &nb_links[&nbr];
        for ((_, dst), part) in links.iter().zip(parts) {
            let mut_slice = section.restrict_mut(*dst);
            D::fuse(&mut mut_slice[0], *part);
        }
    }

    // (suppress unused variable warning for delta, which is used via D::restrict/fuse)
    let _ = delta;
    // Before posting receives
    println!("[rank {}] Posting receives", my_rank);
    // After posting receives, before sends
    println!("[rank {}] Posting sends", my_rank);
    // After all sends
    println!("[rank {}] All sends posted", my_rank);
    // Before waiting for receives
    println!("[rank {}] Waiting for receives", my_rank);
    // After all receives
    println!("[rank {}] All receives complete", my_rank);
    println!("[rank {}] Leaving complete_section", my_rank);
}

/// Complete the vertical stack structure (exchange stack arrows across ranks).
/// Each triple (base, cap, payload) is sent to ghost ranks so they can reconstruct the stack.
pub fn complete_stack<P, Q, Pay, C, S, O, R>(
    _stack: &mut S,
    _overlap: &O,
    _comm: &C,
    _my_rank: usize,
) where
    P: Copy + bytemuck::Pod + Eq + std::hash::Hash + Send + 'static,
    Q: Copy + bytemuck::Pod + Eq + std::hash::Hash + Send + 'static,
    Pay: Copy + bytemuck::Pod + Send + 'static,
    C: crate::algs::communicator::Communicator + Sync,
    S: crate::topology::stack::Stack<Point = P, CapPt = Q, Payload = Pay>,
    O: crate::topology::sieve::Sieve<Point = P, Payload = R> + Sync,
    R: Copy + Send + 'static,
{
    use crate::topology::stack::Stack;
    // use crate::topology::point::PointId;
    // use crate::topology::arrow::Orientation;
    // use crate::topology::stack::InMemoryStack;
    // use crate::overlap::overlap::{Overlap, Remote};
    // use crate::algs::communicator::NoComm;
    // use bytemuck::{Pod, Zeroable};
    // use bytemuck::derive::{Pod as DerivePod, Zeroable as DeriveZeroable};

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::topology::stack::{InMemoryStack, Stack};
        use crate::algs::communicator::NoComm;
        use bytemuck::{Pod, Zeroable};
        // Use the derive macros from bytemuck directly
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
        #[repr(C)]
        pub struct DummyPayload(pub u8);

        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
        #[repr(transparent)]
        pub struct PodPointId(pub u64);
        impl PodPointId {
            pub fn new(raw: u64) -> Self { Self(raw) }
        }

        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
        #[repr(C)]
        pub struct TestRemote {
            pub rank: usize,
            pub remote_point: PodPointId,
        }

        type TestOverlap = crate::topology::sieve::InMemorySieve<PodPointId, TestRemote>;

        #[test]
        fn complete_stack_sends_and_receives_triples() {
            let mut stack0 = InMemoryStack::<PodPointId, PodPointId, DummyPayload>::new();
            let mut stack1 = InMemoryStack::<PodPointId, PodPointId, DummyPayload>::new();
            stack0.add_arrow(PodPointId::new(1), PodPointId::new(101), DummyPayload(42));
            stack1.add_arrow(PodPointId::new(2), PodPointId::new(102), DummyPayload(99));
            let mut overlap = TestOverlap::default();
            overlap.add_arrow(PodPointId::new(1), PodPointId::new(0), TestRemote { rank: 1, remote_point: PodPointId::new(1) });
            overlap.add_arrow(PodPointId::new(2), PodPointId::new(1), TestRemote { rank: 0, remote_point: PodPointId::new(2) });
            let comm = NoComm;
            super::complete_stack::<PodPointId, PodPointId, DummyPayload, _, _, _, TestRemote>(&mut stack0, &overlap, &comm, 0);
            super::complete_stack::<PodPointId, PodPointId, DummyPayload, _, _, _, TestRemote>(&mut stack1, &overlap, &comm, 1);
            let arrows0: Vec<_> = stack0.lift(PodPointId::new(1)).collect();
            let arrows0b: Vec<_> = stack0.lift(PodPointId::new(2)).collect();
            let arrows1: Vec<_> = stack1.lift(PodPointId::new(2)).collect();
            let arrows1b: Vec<_> = stack1.lift(PodPointId::new(1)).collect();
            assert!(arrows0.iter().any(|&(cap, pay)| cap == PodPointId::new(101) && *pay == DummyPayload(42)));
            assert!(arrows0b.iter().any(|&(cap, pay)| cap == PodPointId::new(102) && *pay == DummyPayload(99)));
            assert!(arrows1.iter().any(|&(cap, pay)| cap == PodPointId::new(102) && *pay == DummyPayload(99)));
            assert!(arrows1b.iter().any(|&(cap, pay)| cap == PodPointId::new(101) && *pay == DummyPayload(42)));
        }

        #[test]
        fn complete_stack_local_sim() {
            let mut stack0 = InMemoryStack::<PodPointId, PodPointId, DummyPayload>::new();
            let mut stack1 = InMemoryStack::<PodPointId, PodPointId, DummyPayload>::new();
            Stack::add_arrow(&mut stack0, PodPointId::new(1), PodPointId::new(101), DummyPayload(42));
            Stack::add_arrow(&mut stack1, PodPointId::new(2), PodPointId::new(102), DummyPayload(99));
            let mut overlap = TestOverlap::default();
            overlap.add_arrow(PodPointId::new(1), PodPointId::new(0), TestRemote { rank: 1, remote_point: PodPointId::new(1) });
            overlap.add_arrow(PodPointId::new(2), PodPointId::new(1), TestRemote { rank: 0, remote_point: PodPointId::new(2) });
            let comm = NoComm;
            super::complete_stack::<PodPointId, PodPointId, DummyPayload, _, _, _, TestRemote>(&mut stack0, &overlap, &comm, 0);
            super::complete_stack::<PodPointId, PodPointId, DummyPayload, _, _, _, TestRemote>(&mut stack1, &overlap, &comm, 1);
            let mut found0 = false;
            for (cap, pay) in stack0.lift(PodPointId::new(2)) {
                if cap == PodPointId::new(102) && *pay == DummyPayload(99) {
                    found0 = true;
                }
            }
            let mut found1 = false;
            for (cap, pay) in stack1.lift(PodPointId::new(1)) {
                if cap == PodPointId::new(101) && *pay == DummyPayload(42) {
                    found1 = true;
                }
            }
            assert!(found0 && found1);
        }
    }
}

use crate::topology::sieve::InMemorySieve;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireTriple {
    src: u64,
    dst: u64,
    rank: usize,
}

/// Complete missing topology arrows across MPI ranks.
pub fn complete_sieve(
    sieve: &mut InMemorySieve<PointId, Remote>,
    overlap: &crate::overlap::overlap::Overlap,
    comm: &impl crate::algs::communicator::Communicator,
    my_rank: usize,
) {
    const BASE_TAG: u16 = 0xC0DE;
    let mut nb_links: std::collections::HashMap<usize, Vec<(PointId, PointId)>> = std::collections::HashMap::new();
    let me_pt = partition_point(my_rank);
    let mut has_owned = false;
    for (&p, outs) in &sieve.adjacency_out {
        has_owned = true;
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
    if !has_owned {
        for (src, rem) in overlap.support(me_pt) {
            if rem.rank != my_rank {
                nb_links.entry(rem.rank)
                    .or_default()
                    .push((rem.remote_point, src));
            }
        }
    }
    let mut recv_size = std::collections::HashMap::new();
    for (&nbr, _) in &nb_links {
        let buf = [0u8; 4];
        let h = comm.irecv(nbr, BASE_TAG, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }
    for (&nbr, links) in &nb_links {
        let count = links.len() as u32;
        comm.isend(nbr, BASE_TAG, &count.to_le_bytes());
    }
    let mut sizes_in = std::collections::HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        let data = h.wait().expect("size receive");
        buf.copy_from_slice(&data);
        sizes_in.insert(nbr, u32::from_le_bytes(buf) as usize);
    }
    let mut recv_data = std::collections::HashMap::new();
    for (&nbr, _) in &nb_links {
        let n_items = sizes_in[&nbr];
        let mut buffer = vec![0u8; n_items * std::mem::size_of::<WireTriple>()];
        let h = comm.irecv(nbr, BASE_TAG + 1, &mut buffer);
        recv_data.insert(nbr, (h, buffer));
    }
    for (&nbr, links) in &nb_links {
        let mut triples = Vec::with_capacity(links.len());
        for &(src, dst) in links {
            if let Some(outs) = sieve.adjacency_out.get(&src) {
                for (d, payload) in outs {
                    if *d == dst {
                        triples.push(WireTriple { src: src.get(), dst: dst.get(), rank: payload.rank });
                    }
                }
            }
        }
        let bytes = bytemuck::cast_slice(&triples);
        comm.isend(nbr, BASE_TAG + 1, bytes);
    }
    let mut inserted = std::collections::HashSet::new();
    for (_nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().expect("data receive");
        buffer.copy_from_slice(&raw);
        let triples: &[WireTriple] = bytemuck::cast_slice(&buffer);
        for WireTriple { src, dst, rank } in triples {
            let src_pt = PointId::new(*src);
            let dst_pt = PointId::new(*dst);
            let payload = Remote { rank: *rank, remote_point: dst_pt };
            if inserted.insert((src_pt, dst_pt)) {
                let already = sieve.adjacency_out.get(&src_pt)
                    .map_or(false, |v| v.iter().any(|(d, _)| *d == dst_pt));
                if !already {
                    sieve.adjacency_out.entry(src_pt).or_default().push((dst_pt, payload));
                    sieve.adjacency_in.entry(dst_pt).or_default().push((src_pt, payload));
                }
            }
        }
    }
    sieve.strata.take();
    crate::topology::utils::assert_dag(sieve);
}

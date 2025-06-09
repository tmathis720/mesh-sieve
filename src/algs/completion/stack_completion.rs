//! Complete the vertical‐stack arrows (mirror of section completion).

use crate::algs::communicator::Wait;
use std::collections::HashMap;

/// A tightly-packed triple of (base, cap, payload).
#[repr(C)]
#[derive(Copy, Clone)]
struct WireTriple<P, Q, Pay>
where
    P: Copy,
    Q: Copy,
    Pay: Copy,
{
    base: P,
    cap: Q,
    pay: Pay,
}

/// Trait for extracting rank from overlap payloads
pub trait HasRank {
    fn rank(&self) -> usize;
}

impl HasRank for crate::overlap::overlap::Remote {
    fn rank(&self) -> usize {
        self.rank
    }
}

pub fn complete_stack<P, Q, Pay, C, S, O, R>(
    stack: &mut S,
    overlap: &O,
    comm: &C,
    my_rank: usize,
    n_ranks: usize,
) where
    P: Copy + bytemuck::Pod + Eq + std::hash::Hash + Send + 'static,
    Q: Copy + bytemuck::Pod + Eq + std::hash::Hash + Send + 'static,
    Pay: Copy + bytemuck::Pod + Send + 'static,
    C: crate::algs::communicator::Communicator + Sync,
    S: crate::topology::stack::Stack<Point = P, CapPt = Q, Payload = Pay>,
    O: crate::topology::sieve::Sieve<Point = P, Payload = R> + Sync,
    R: HasRank + Copy + Send + 'static,
{
    const BASE_TAG: u16 = 0xC0DE;
    // 1. Find all neighbors (ranks) to communicate with
    let mut nb_links: HashMap<usize, Vec<(P, Q, Pay)>> = HashMap::new();
    // Iterate over all base points in the stack's base sieve
    for base in stack.base_points() {
        for (cap, pay) in stack.lift(base) {
            for (_dst, rem) in overlap.cone(base) {
                if rem.rank() != my_rank {
                    nb_links
                        .entry(rem.rank())
                        .or_default()
                        .push((base, cap, *pay));
                }
            }
        }
    }
    // --- DEADLOCK FIX: ensure symmetric communication ---
    // Use all ranks except my_rank as neighbors
    let all_neighbors: std::collections::HashSet<usize> =
        (0..n_ranks).filter(|&r| r != my_rank).collect();
    // 2. Exchange sizes (always post send/recv for all neighbors)
    let mut recv_size = HashMap::new();
    for &nbr in &all_neighbors {
        let buf = [0u8; 4];
        let h = comm.irecv(nbr, BASE_TAG, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }
    for &nbr in &all_neighbors {
        let count = nb_links.get(&nbr).map_or(0, |v| v.len()) as u32;
        comm.isend(nbr, BASE_TAG, &count.to_le_bytes());
    }
    let mut sizes_in = HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        let data = h.wait().expect("size receive");
        buf.copy_from_slice(&data);
        sizes_in.insert(nbr, u32::from_le_bytes(buf) as usize);
    }
    // 3. Exchange data (always post send/recv for all neighbors)

    let item_size = std::mem::size_of::<WireTriple<P, Q, Pay>>();
    let mut recv_data = HashMap::new();
    for &nbr in &all_neighbors {
        let n_items = sizes_in.get(&nbr).copied().unwrap_or(0);
        let mut buf = vec![0u8; n_items * item_size];
        let h = comm.irecv(nbr, BASE_TAG + 1, &mut buf);
        recv_data.insert(nbr, (h, buf));
    }
    for &nbr in &all_neighbors {
        let triples = nb_links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let wire: Vec<WireTriple<P, Q, Pay>> = triples
            .iter()
            .map(|&(b, c, p)| WireTriple {
                base: b,
                cap: c,
                pay: p,
            })
            .collect();
        // SAFETY: WireTriple<P,Q,Pay> is repr(C) and all fields are Pod
        let bytes = if wire.is_empty() {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    wire.as_ptr() as *const u8,
                    wire.len() * std::mem::size_of::<WireTriple<P, Q, Pay>>(),
                )
            }
        };
        comm.isend(nbr, BASE_TAG + 1, bytes);
    }
    for (_nbr, (h, mut buf)) in recv_data {
        let raw = h.wait().expect("data receive");
        buf.copy_from_slice(&raw);
        // SAFETY: buf is a byte buffer of WireTriple<P,Q,Pay>
        let incoming: &[WireTriple<P, Q, Pay>] = if buf.is_empty() {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    buf.as_ptr() as *const WireTriple<P, Q, Pay>,
                    buf.len() / std::mem::size_of::<WireTriple<P, Q, Pay>>(),
                )
            }
        };
        for &WireTriple { base, cap, pay } in incoming {
            stack.add_arrow(base, cap, pay);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algs::communicator::RayonComm;
    use crate::topology::sieve::{InMemorySieve, Sieve};
    use crate::topology::stack::{InMemoryStack, Stack};
    use std::thread;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(transparent)]
    struct DummyPayload(pub u32);

    #[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Hash)]
    #[repr(transparent)]
    struct PodU64(pub u64);

    #[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct DummyRemote {
        rank: usize,
        remote_point: PodU64,
    }
    impl HasRank for DummyRemote {
        fn rank(&self) -> usize {
            self.rank
        }
    }

    #[test]
    fn test_complete_stack_two_ranks() {
        // Simulate two ranks in parallel threads
        let t0 = thread::spawn(|| {
            let mut stack = InMemoryStack::<PodU64, PodU64, DummyPayload>::new();
            stack.add_arrow(PodU64(1), PodU64(101), DummyPayload(42));
            let mut overlap = InMemorySieve::<PodU64, DummyRemote>::default();
            overlap.add_arrow(
                PodU64(1),
                PodU64(1),
                DummyRemote {
                    rank: 1,
                    remote_point: PodU64(1),
                },
            );
            let comm = RayonComm::new(0);
            complete_stack(&mut stack, &overlap, &comm, 0, 2);
            stack
                .lift(PodU64(1))
                .map(|(cap, pay)| (cap, *pay))
                .collect::<Vec<_>>()
        });
        let t1 = thread::spawn(|| {
            let mut stack = InMemoryStack::<PodU64, PodU64, DummyPayload>::new();
            let mut overlap = InMemorySieve::<PodU64, DummyRemote>::default();
            overlap.add_arrow(
                PodU64(1),
                PodU64(1),
                DummyRemote {
                    rank: 0,
                    remote_point: PodU64(1),
                },
            );
            let comm = RayonComm::new(1);
            complete_stack(&mut stack, &overlap, &comm, 1, 2);
            stack
                .lift(PodU64(1))
                .map(|(cap, pay)| (cap, *pay))
                .collect::<Vec<_>>()
        });
        let res0 = t0.join().unwrap();
        let res1 = t1.join().unwrap();
        // Both ranks should have the same arrow after completion
        assert!(res0.contains(&(PodU64(101), DummyPayload(42))));
        assert!(res1.contains(&(PodU64(101), DummyPayload(42))));
    }

    // #[test]
    // fn test_complete_stack_no_overlap() {
    //     let mut stack = InMemoryStack::<PodU64, PodU64, DummyPayload>::new();
    //     stack.add_arrow(PodU64(2), PodU64(202), DummyPayload(99));
    //     let overlap = InMemorySieve::<PodU64, DummyRemote>::default();
    //     let comm = RayonComm::new(0);
    //     complete_stack(&mut stack, &overlap, &comm, 0, 2);
    //     let arrows: Vec<_> = stack
    //         .lift(PodU64(2))
    //         .map(|(cap, pay)| (cap, *pay))
    //         .collect();
    //     assert!(arrows.contains(&(PodU64(202), DummyPayload(99))));
    // }
}

//! Complete the vertical‐stack arrows (mirror of section completion).

use std::collections::HashMap;
use crate::algs::communicator::Wait;

/// A tightly-packed triple of (base, cap, payload).
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

/// Trait for extracting rank from overlap payloads
pub trait HasRank {
    fn rank(&self) -> usize;
}

impl HasRank for crate::overlap::overlap::Remote {
    fn rank(&self) -> usize { self.rank }
}

pub fn complete_stack<P, Q, Pay, C, S, O, R>(
    stack: &mut S,
    overlap: &O,
    comm: &C,
    my_rank: usize,
    n_ranks: usize,
) where
    P: Copy + bytemuck::Pod + bytemuck::Zeroable + Default + Eq + std::hash::Hash + Send + 'static,
    Q: Copy + bytemuck::Pod + bytemuck::Zeroable + Default + Eq + std::hash::Hash + Send + 'static,
    Pay: Copy + bytemuck::Pod + bytemuck::Zeroable + Default + Send + 'static,
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
                    nb_links.entry(rem.rank())
                        .or_default()
                        .push((base, cap, *pay));
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
    use bytemuck::cast_slice;
    use bytemuck::cast_slice_mut;
    let mut recv_data = HashMap::new();
    for &nbr in &all_neighbors {
        let n_items = sizes_in.get(&nbr).copied().unwrap_or(0);
        let mut buf = vec![WireTriple::<P, Q, Pay> { base: P::default(), cap: Q::default(), pay: Pay::default() }; n_items];
        let h = comm.irecv(nbr, BASE_TAG + 1, cast_slice_mut(&mut buf));
        recv_data.insert(nbr, (h, buf));
    }
    for &nbr in &all_neighbors {
        let triples = nb_links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let wire: Vec<WireTriple<P,Q,Pay>> =
            triples.iter()
                   .map(|&(b,c,p)| WireTriple { base:b, cap:c, pay:p })
                   .collect();
        let bytes = cast_slice(&wire);
        comm.isend(nbr, BASE_TAG + 1, bytes);
    }
    for (_nbr, (h, mut buf)) in recv_data {
        let raw = h.wait().expect("data receive");
        let buf_bytes = cast_slice_mut(&mut buf);
        buf_bytes.copy_from_slice(&raw);
        let incoming: &[WireTriple<P,Q,Pay>] = &buf;
        for &WireTriple { base, cap, pay } in incoming {
            stack.add_arrow(base, cap, pay);
        }
    }
}


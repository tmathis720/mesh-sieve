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
use crate::overlap::overlap::{Overlap};
use crate::overlap::delta::Delta;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;

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

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
fn partition_point(rank: usize) -> PointId {
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

    // 1) OWNER-side: for every local point I actually own,
    //    send to every ghost rank that I linked
    for (p, _) in section.iter() {
        // for every outgoing arrow from my mesh-point -> partition_point(peer)
        for (dst, rem) in ovlp.cone(p) {
            if rem.rank != my_rank {
                out.entry(rem.rank)
                   .or_default()
                   .push((p, rem.remote_point));
            }
        }
    }

    // 2) GHOST-side: for every arrow original owners added into my partition point,
    //    receive from that owner’s `src` into my `rem.remote_point`
    for (src, rem) in ovlp.support(me_pt) {
        if rem.rank != my_rank {
            // rem.remote_point here is my local ghost‐slot ID
            out.entry(rem.rank)
               .or_default()
               .push((rem.remote_point, src));
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
    V: Clone + Send + Default + 'static,
    D: Delta<V> + Sync,
    D::Part: Wire,
    C: Communicator + Sync,
{
    const BASE_TAG: u16 = 0xBEEF;
    let nb_links = neighbour_links(section, overlap, my_rank);

    // --- Stage 1: exchange sizes ---
    let mut recv_size = HashMap::new();
    for (&nbr, _) in &nb_links {
        let buf = [0u8;4];
        let h = comm.irecv(nbr, BASE_TAG, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }
    for (&nbr, links) in &nb_links {
        let count = links.len() as u32;
        comm.isend(nbr, BASE_TAG, &count.to_le_bytes());
    }
    let mut sizes_in = HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        let data = h.wait().expect("size receive");
        buf.copy_from_slice(&data);
        sizes_in.insert(nbr, u32::from_le_bytes(buf));
    }

    // --- Stage 2: exchange data ---
    let mut recv_data = HashMap::new();
    for (&nbr, links) in &nb_links {
        let n_items = sizes_in[&nbr] as usize;
        let mut buffer = vec![0u8; n_items * std::mem::size_of::<D::Part>()];
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
        comm.isend(nbr, BASE_TAG+1, bytes);
    }
    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().expect("data receive");
        buffer.copy_from_slice(&raw);
        let parts: &[D::Part] = cast_slice(&buffer);
        let links = &nb_links[&nbr];
        for ((loc, _), part) in links.iter().zip(parts) {
            let mut_slice = section.restrict_mut(*loc);
            D::fuse(&mut mut_slice[0], part.clone());
        }
    }

    // (suppress unused variable warning for delta, which is used via D::restrict/fuse)
    let _ = delta;
}

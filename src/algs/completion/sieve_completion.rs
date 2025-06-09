//! Complete missing sieve arrows across ranks by packing WireTriple.

use crate::topology::point::PointId;
use crate::overlap::overlap::Remote;
use crate::algs::completion::partition_point;
use crate::topology::sieve::Sieve;
use crate::algs::communicator::Wait;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireTriple {
    src: u64,
    dst: u64,
    rank: usize,
}

pub fn complete_sieve(
    sieve: &mut crate::topology::sieve::InMemorySieve<crate::topology::point::PointId, crate::overlap::overlap::Remote>,
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
        // For every outgoing arrow from my mesh-point
        for (_dst, _payload) in outs {
            // For every neighbor who has an overlap link to this point
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
        for &(src, _dst) in links {
            // Send all arrows from src, not just those matching dst
            if let Some(outs) = sieve.adjacency_out.get(&src) {
                for (d, payload) in outs {
                    triples.push(WireTriple { src: src.get(), dst: d.get(), rank: payload.rank });
                }
            }
        }
        let bytes = bytemuck::cast_slice(&triples);
        comm.isend(nbr, BASE_TAG + 1, bytes);
    }
    // 4. Stage 3: integrate
    let mut inserted = std::collections::HashSet::new();
    for (_nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().expect("data receive");
        buffer.copy_from_slice(&raw);
        let triples: &[WireTriple] = bytemuck::cast_slice(&buffer);
        for WireTriple { src, dst, rank } in triples {
            let src_pt = PointId::new(*src);
            let dst_pt = PointId::new(*dst);
            let payload = Remote { rank: *rank, remote_point: dst_pt };
            // Only inject if this (src, dst) is not already present
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
    // After all integration, ensure remote faces are present by adding missing overlap links
    // (simulate what would happen in a real MPI exchange)
    sieve.strata.take();
    crate::topology::utils::assert_dag(sieve);
}

// Optionally, add #[cfg(test)] mod tests for sieve completion

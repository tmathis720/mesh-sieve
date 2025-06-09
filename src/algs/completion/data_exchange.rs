//! Stage 2 of section completion: exchange the actual data items.

use std::collections::HashMap;
use bytemuck::{Pod, cast_slice};
use crate::algs::communicator::{Communicator, Wait};
use crate::overlap::delta::Delta;

/// For each neighbor, pack `Delta::restrict` from your section into a send buffer,
/// post irecv for the corresponding byte length (from stage 1),
/// then send and finally wait + `Delta::fuse` into your local section.
pub fn exchange_data<V, D, C>(
    links: &std::collections::HashMap<usize, Vec<(crate::topology::point::PointId, crate::topology::point::PointId)>>,
    recv_counts: &std::collections::HashMap<usize, u32>,
    comm: &C,
    base_tag: u16,
    section: &mut crate::data::section::Section<V>,
) where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::Delta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod,
    C: crate::algs::communicator::Communicator + Sync,
{
    use bytemuck::cast_slice;
    use std::collections::HashMap;
    // --- Stage 2: exchange data ---
    let mut recv_data = HashMap::new();
    for (&nbr, _links) in links {
        let n_items = recv_counts[&nbr] as usize;
        let mut buffer = vec![0u8; n_items * std::mem::size_of::<D::Part>()];
        let h = comm.irecv(nbr, base_tag, &mut buffer);
        recv_data.insert(nbr, (h, buffer));
    }
    for (&nbr, links) in links {
        let mut scratch = Vec::with_capacity(links.len());
        for &(loc, _) in links {
            let slice = section.restrict(loc);
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        comm.isend(nbr, base_tag, bytes);
    }
    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().expect("data receive");
        buffer.copy_from_slice(&raw);
        let parts: &[D::Part] = cast_slice(&buffer);
        let links = &links[&nbr];
        for ((_, dst), part) in links.iter().zip(parts) {
            let mut_slice = section.restrict_mut(*dst);
            D::fuse(&mut mut_slice[0], *part);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // TODO: Add tests for exchange_data with a mock communicator and section
}

//! Stage 1 of section completion: exchange counts with each neighbor.
//!
//! This module provides helpers for exchanging the number of items to send/receive
//! with each neighbor during distributed section completion. It supports both asymmetric
//! and symmetric communication patterns.

use crate::algs::communicator::Wait;

/// Posts irecv/isend for the number of items to expect from each neighbor.
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes<C, T>(
    links: &std::collections::HashMap<usize, Vec<T>>,
    comm: &C,
    base_tag: u16,
) -> Result<std::collections::HashMap<usize, u32>, crate::mesh_error::MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    let mut recv_size = std::collections::HashMap::new();
    for &nbr in links.keys() {
        let buf = [0u8; 4];
        let h = comm.irecv(nbr, base_tag, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }
    for (&nbr, items) in links {
        let count = items.len() as u32;
        comm.isend(nbr, base_tag, &count.to_le_bytes());
    }
    let mut sizes_in = std::collections::HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        let data = h
            .wait()
            .ok_or_else(|| crate::mesh_error::CommError(format!("failed to receive size from rank {}", nbr)))?;
        buf.copy_from_slice(&data);
        sizes_in.insert(nbr, u32::from_le_bytes(buf));
    }
    Ok(sizes_in)
}

/// Posts irecv/isend for the number of items to expect from each neighbor (symmetric version).
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes_symmetric<C, T>(
    links: &std::collections::HashMap<usize, Vec<T>>,
    comm: &C,
    base_tag: u16,
    all_neighbors: &std::collections::HashSet<usize>,
) -> Result<std::collections::HashMap<usize, u32>, crate::mesh_error::MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    let mut recv_size = std::collections::HashMap::new();
    for &nbr in all_neighbors {
        let buf = [0u8; 4];
        let h = comm.irecv(nbr, base_tag, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }
    for &nbr in all_neighbors {
        let count = links.get(&nbr).map_or(0, |v| v.len()) as u32;
        comm.isend(nbr, base_tag, &count.to_le_bytes());
    }
    let mut sizes_in = std::collections::HashMap::new();
    for (nbr, (h, mut buf)) in recv_size {
        let data = h
            .wait()
            .ok_or_else(|| crate::mesh_error::CommError(format!("failed to receive size from rank {}", nbr)))?;
        buf.copy_from_slice(&data);
        sizes_in.insert(nbr, u32::from_le_bytes(buf));
    }
    Ok(sizes_in)
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for exchange_sizes with a mock communicator
}

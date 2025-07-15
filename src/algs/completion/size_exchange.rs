//! Stage 1 of section completion: exchange counts with each neighbor.
//!
//! This module provides helpers for exchanging the number of items to send/receive
//! with each neighbor during distributed section completion. It supports both asymmetric
//! and symmetric communication patterns.

use crate::algs::communicator::Wait;
use crate::mesh_error::MeshSieveError;
use std::collections::HashMap;

/// Posts irecv/isend for the number of items to expect from each neighbor.
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes<C, T>(
    links: &HashMap<usize, Vec<T>>,
    comm: &C,
    base_tag: u16,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    // 1) post all receives
    let mut recv_size: HashMap<usize, (C::RecvHandle, [u8; 4])> = HashMap::new();
    for &nbr in links.keys() {
        let buf = [0u8; 4];
        let h = comm.irecv(nbr, base_tag, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }

    // 2) post all sends and stash their handles
    let mut pending_sends = Vec::with_capacity(links.len());
    for (&nbr, items) in links {
        let count = items.len() as u32;
        let buf = count.to_le_bytes();
        let handle = comm.isend(nbr, base_tag, &buf);
        pending_sends.push(handle);
    }

    // 3) wait for all recvs, collect counts (but do not early-return!)
    let mut sizes_in: HashMap<usize, u32> = HashMap::new();
    let mut maybe_err: Option<MeshSieveError> = None;

    for (nbr, (h, mut buf)) in recv_size {
        if maybe_err.is_some() {
            break;
        }
        match h.wait() {
            Some(data) => {
                if data.len() != buf.len() {
                    maybe_err = Some(MeshSieveError::CommError {
                        neighbor: nbr,
                        source: format!(
                            "expected 4 bytes for size header, got {}",
                            data.len()
                        ).into(),
                    });
                    break;
                }
                buf.copy_from_slice(&data);
                sizes_in.insert(nbr, u32::from_le_bytes(buf));
            }
            None => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("failed to receive size from rank {}", nbr).into(),
                });
                break;
            }
        }
    }

    // 4) always drain all send handles before returning
    for send in pending_sends {
        let _ = send.wait();
    }

    // 5) now return error or success
    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(sizes_in)
    }
}

/// Posts irecv/isend for the number of items to expect from each neighbor (symmetric version).
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes_symmetric<C, T>(
    links: &HashMap<usize, Vec<T>>,
    comm: &C,
    base_tag: u16,
    all_neighbors: &std::collections::HashSet<usize>,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    // 1) post all receives
    let mut recv_size: HashMap<usize, (C::RecvHandle, [u8; 4])> = HashMap::new();
    for &nbr in all_neighbors {
        let buf = [0u8; 4];
        let h = comm.irecv(nbr, base_tag, &mut buf.clone());
        recv_size.insert(nbr, (h, buf));
    }

    // 2) post all sends and stash handles + buffers
    let mut pending_sends = Vec::with_capacity(all_neighbors.len());
    let mut send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let count = links.get(&nbr).map_or(0, |v| v.len()) as u32;
        let buf = count.to_le_bytes();
        let handle = comm.isend(nbr, base_tag, &buf);
        pending_sends.push(handle);
        send_bufs.push(buf); // keep alive until after wait
    }

    // 3) wait for all recvs, collect counts (but do not early-return!)
    let mut sizes_in: HashMap<usize, u32> = HashMap::new();
    let mut maybe_err: Option<MeshSieveError> = None;

    for (nbr, (h, mut buf)) in recv_size {
        if maybe_err.is_some() {
            break;
        }
        match h.wait() {
            Some(data) => {
                if data.len() != buf.len() {
                    maybe_err = Some(MeshSieveError::CommError {
                        neighbor: nbr,
                        source: format!(
                            "expected 4 bytes for size header, got {}",
                            data.len()
                        ).into(),
                    });
                    break;
                }
                buf.copy_from_slice(&data);
                sizes_in.insert(nbr, u32::from_le_bytes(buf));
            }
            None => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("failed to receive size from rank {}", nbr).into(),
                });
                break;
            }
        }
    }

    // 4) always drain all send handles before returning
    for send in pending_sends {
        let _ = send.wait();
    }

    // 5) now return error or success
    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(sizes_in)
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for exchange_sizes and exchange_sizes_symmetric
}

//! Stage 1 of section completion: exchange counts with each neighbor.
//!
//! This module provides helpers for exchanging the number of items to send/receive
//! with each neighbor during distributed section completion. It supports both asymmetric
//! and symmetric communication patterns.

use crate::algs::communicator::Wait;
use crate::mesh_error::MeshSieveError;
use std::collections::{HashMap, HashSet};

/// Posts irecv/isend for the number of items to expect from each neighbor (asymmetric).
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes<C, T>(
    links: &HashMap<usize, Vec<T>>,
    comm: &C,
    base_tag: u16,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    // 1) post all receives (storing each buffer in our map)
    let mut recv_size: HashMap<usize, (C::RecvHandle, [u8; 4])> = HashMap::new();
    for &nbr in links.keys() {
        let mut buf = [0u8; 4];
        let h = comm.irecv(nbr, base_tag, &mut buf);
        recv_size.insert(nbr, (h, buf));
    }

    // 2) post all sends
    let mut pending_sends = Vec::with_capacity(links.len());
    for (&nbr, items) in links.iter() {
        let count = items.len() as u32;
        let buf = count.to_le_bytes();
        pending_sends.push(comm.isend(nbr, base_tag, &buf));
    }

    // 3) wait for all recvs, collect counts (but do not early–return)
    let mut sizes_in = HashMap::new();
    let mut maybe_err = None;
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
                            "expected {} bytes for size header, got {}",
                            buf.len(),
                            data.len()
                        )
                        .into(),
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

    // 5) return error or success
    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(sizes_in)
    }
}

/// Posts irecv/isend for the number of items to expect from each neighbor (symmetric).
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes_symmetric<C, T>(
    links: &HashMap<usize, Vec<T>>,
    comm: &C,
    base_tag: u16,
    all_neighbors: &HashSet<usize>,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    // 1) post all receives
    let mut recv_size: HashMap<usize, (C::RecvHandle, [u8; 4])> = HashMap::new();
    for &nbr in all_neighbors {
        let mut buf = [0u8; 4];
        let h = comm.irecv(nbr, base_tag, &mut buf);
        recv_size.insert(nbr, (h, buf));
    }

    // 2) post all sends and stash buffers so they're alive until sends complete
    let mut pending_sends = Vec::with_capacity(all_neighbors.len());
    let mut send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let count = links.get(&nbr).map_or(0, |v| v.len()) as u32;
        let buf = count.to_le_bytes();
        pending_sends.push(comm.isend(nbr, base_tag, &buf));
        send_bufs.push(buf);
    }

    // 3) wait for all recvs, collect counts (but do not early–return)
    let mut sizes_in = HashMap::new();
    let mut maybe_err = None;
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
                            "expected {} bytes for size header, got {}",
                            buf.len(),
                            data.len()
                        )
                        .into(),
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

    // 5) return error or success
    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(sizes_in)
    }
}

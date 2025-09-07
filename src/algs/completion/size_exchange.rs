//! Stage 1 of section completion: exchange counts with each neighbor.
//!
//! This module provides helpers for exchanging the number of items to send/receive
//! with each neighbor during distributed section completion. It supports both asymmetric
//! and symmetric communication patterns. All functions take a typed [`CommTag`] and
//! guarantee that every send/receive handle is drained before returning, even if an
//! error occurs.

use crate::algs::communicator::{CommTag, Wait};
use crate::algs::wire::{WireCount, cast_slice, cast_slice_mut};
use crate::mesh_error::MeshSieveError;
use std::collections::{HashMap, HashSet};

/// Posts irecv/isend for the number of items to expect from each neighbor (asymmetric).
/// Returns a map `nbr → u32` once all receives have completed.
pub fn exchange_sizes<C, T>(
    links: &HashMap<usize, Vec<T>>,
    comm: &C,
    tag: CommTag,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    // 1) post all receives (storing each buffer in our map)
    let mut recv_size: HashMap<usize, (C::RecvHandle, WireCount)> = HashMap::new();
    for &nbr in links.keys() {
        let mut cnt = WireCount::new(0);
        let h = comm.irecv(
            nbr,
            tag.as_u16(),
            cast_slice_mut(std::slice::from_mut(&mut cnt)),
        );
        recv_size.insert(nbr, (h, cnt));
    }

    // 2) post all sends and keep buffers alive until completion
    let mut pending_sends = Vec::with_capacity(links.len());
    let mut send_bufs = Vec::with_capacity(links.len());
    for (&nbr, items) in links.iter() {
        let count = WireCount::new(items.len());
        pending_sends.push(comm.isend(
            nbr,
            tag.as_u16(),
            cast_slice(std::slice::from_ref(&count)),
        ));
        send_bufs.push(count);
    }

    // 3) wait for all recvs, collect counts (but do not early–return)
    let mut sizes_in = HashMap::new();
    let mut maybe_err = None;
    for (nbr, (h, mut cnt)) in recv_size {
        match h.wait() {
            Some(data) if data.len() == std::mem::size_of::<WireCount>() => {
                if maybe_err.is_none() {
                    let bytes = cast_slice_mut(std::slice::from_mut(&mut cnt));
                    bytes.copy_from_slice(&data);
                    sizes_in.insert(nbr, cnt.get() as u32);
                }
            }
            Some(data) if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!(
                        "expected {} bytes for size header, got {}",
                        std::mem::size_of::<WireCount>(),
                        data.len()
                    )
                    .into(),
                });
            }
            None if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("failed to receive size from rank {nbr}").into(),
                });
            }
            _ => {} // already have an error; just drain
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
    tag: CommTag,
    all_neighbors: &HashSet<usize>,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: crate::algs::communicator::Communicator + Sync,
{
    // 1) post all receives
    let mut recv_size: HashMap<usize, (C::RecvHandle, WireCount)> = HashMap::new();
    for &nbr in all_neighbors {
        let mut cnt = WireCount::new(0);
        let h = comm.irecv(
            nbr,
            tag.as_u16(),
            cast_slice_mut(std::slice::from_mut(&mut cnt)),
        );
        recv_size.insert(nbr, (h, cnt));
    }

    // 2) post all sends and stash buffers so they're alive until sends complete
    let mut pending_sends = Vec::with_capacity(all_neighbors.len());
    let mut send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let count = WireCount::new(links.get(&nbr).map_or(0, |v| v.len()));
        pending_sends.push(comm.isend(
            nbr,
            tag.as_u16(),
            cast_slice(std::slice::from_ref(&count)),
        ));
        send_bufs.push(count);
    }

    // 3) wait for all recvs, collect counts (but do not early–return)
    let mut sizes_in = HashMap::new();
    let mut maybe_err = None;
    for (nbr, (h, mut cnt)) in recv_size {
        match h.wait() {
            Some(data) if data.len() == std::mem::size_of::<WireCount>() => {
                if maybe_err.is_none() {
                    let bytes = cast_slice_mut(std::slice::from_mut(&mut cnt));
                    bytes.copy_from_slice(&data);
                    sizes_in.insert(nbr, cnt.get() as u32);
                }
            }
            Some(data) if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!(
                        "expected {} bytes for size header, got {}",
                        std::mem::size_of::<WireCount>(),
                        data.len()
                    )
                    .into(),
                });
            }
            None if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("failed to receive size from rank {nbr}").into(),
                });
            }
            _ => {}
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

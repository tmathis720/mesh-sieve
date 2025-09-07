// src/algs/completion/closure_fetch.rs
//! Request/response for batched cone/support fetches for selected points.
//! Used by `closure_completed(...)`.

use crate::algs::communicator::{Communicator, Wait};
use crate::algs::wire::{WireAdj, WireCount, WireHdr, WirePoint, WIRE_VERSION};
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

#[repr(u8)]
#[derive(Copy, Clone)]
pub enum ReqKind {
    Cone = 1,
    Support = 2,
}

/// Send to each owner rank the list of points to fetch; receive adjacency lists back.
/// Returns map: src_point -> Vec<dst_point>.
pub fn fetch_adjacency<C: Communicator>(
    requests: &HashMap<usize, Vec<PointId>>,
    kind: ReqKind,
    comm: &C,
    base_tag: u16,
) -> Result<HashMap<PointId, Vec<PointId>>, MeshSieveError> {
    use bytemuck::{cast_slice, cast_slice_mut, Zeroable};

    // 1) Post receives for replies (counts then payload)
    let mut recv_counts = Vec::new();
    for (&rank, _) in requests {
        let mut cnt = WireCount::new(0);
        let h = comm.irecv(
            rank,
            base_tag + 1,
            cast_slice_mut(std::slice::from_mut(&mut cnt)),
        );
        recv_counts.push((rank, h, cnt));
    }

    // 2) Send requests (header + point list)
    let mut pending_sends = Vec::new();
    let mut _keep_alive: Vec<Vec<WirePoint>> = Vec::new();
    let mut _keep_hdrs: Vec<WireHdr> = Vec::new();
    let mut _keep_counts: Vec<WireCount> = Vec::new();
    for (&rank, pts) in requests {
        let hdr = WireHdr::new(kind as u16);
        let body: Vec<WirePoint> = pts.iter().map(|p| WirePoint::of(p.get())).collect();
        let cnt = WireCount::new(body.len());
        pending_sends.push(comm.isend(
            rank,
            base_tag,
            cast_slice(std::slice::from_ref(&hdr)),
        ));
        pending_sends.push(comm.isend(
            rank,
            base_tag + 1,
            cast_slice(std::slice::from_ref(&cnt)),
        ));
        pending_sends.push(comm.isend(rank, base_tag, cast_slice(&body)));
        _keep_hdrs.push(hdr);
        _keep_counts.push(cnt);
        _keep_alive.push(body);
    }

    // 3) Gather reply sizes (number of WireAdj records)
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for (rank, h, mut cnt) in recv_counts {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: rank,
            source: Box::new(crate::mesh_error::CommError("missing count reply".into())),
        })?;
        let bytes = cast_slice_mut(std::slice::from_mut(&mut cnt));
        if raw.len() != bytes.len() {
            return Err(MeshSieveError::CommError {
                neighbor: rank,
                source: Box::new(crate::mesh_error::CommError(format!(
                    "expected {}B count, got {}",
                    bytes.len(),
                    raw.len()
                ))),
            });
        }
        bytes.copy_from_slice(&raw);
        counts.insert(rank, cnt.get());
    }

    // 4) Post receives for adjacency payloads
    let mut recv_payloads = Vec::new();
    for (&rank, &nrec) in &counts {
        let mut buf = vec![WireAdj::zeroed(); nrec];
        let h = comm.irecv(rank, base_tag + 2, cast_slice_mut(buf.as_mut_slice()));
        recv_payloads.push((rank, h, buf));
    }

    // 5) Wait for sends to finish
    for s in pending_sends {
        let _ = s.wait();
    }

    // 6) Unpack replies
    let mut out: HashMap<PointId, Vec<PointId>> = HashMap::new();
    for (rank, h, mut buf) in recv_payloads {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: rank,
            source: Box::new(crate::mesh_error::CommError("missing payload reply".into())),
        })?;
        let bytes = cast_slice_mut(buf.as_mut_slice());
        if bytes.len() != raw.len() {
            return Err(MeshSieveError::CommError {
                neighbor: rank,
                source: Box::new(crate::mesh_error::CommError(format!(
                    "adj payload: expected {}B, got {}",
                    bytes.len(),
                    raw.len()
                ))),
            });
        }
        bytes.copy_from_slice(&raw);
        for a in &buf {
            let sp = PointId::new(a.src()).map_err(|e| MeshSieveError::MeshError(Box::new(e)))?;
            let dp = PointId::new(a.dst()).map_err(|e| MeshSieveError::MeshError(Box::new(e)))?;
            out.entry(sp).or_default().push(dp);
        }
    }

    Ok(out)
}

/// Service-side handler (owner rank): handle one fetch (if present).
/// Call this periodically on each rank with a handle to your local **mesh** Sieve (not Overlap).
pub fn service_once_mesh_fetch<S: Sieve<Point = PointId>>(
    comm: &impl Communicator,
    local_mesh: &S,
    base_tag: u16,
) -> bool {
    use crate::algs::communicator::Wait;
    use bytemuck::{cast_slice, cast_slice_mut, Zeroable};

    let me = comm.rank();
    let mut handled = false;
    for peer in 0..comm.size() {
        if peer == me {
            continue;
        }
        let mut hdr = WireHdr::new(0);
        let maybe = comm.irecv(peer, base_tag, cast_slice_mut(std::slice::from_mut(&mut hdr)));
        if let Some(raw) = maybe.wait() {
            if raw.len() != std::mem::size_of::<WireHdr>() {
                continue;
            }
            if hdr.version() != WIRE_VERSION {
                continue;
            }

            let mut cnt = WireCount::new(0);
            let hcnt = comm.irecv(
                peer,
                base_tag + 1,
                cast_slice_mut(std::slice::from_mut(&mut cnt)),
            );
            if let Some(rc) = hcnt.wait() {
                if rc.len() != std::mem::size_of::<WireCount>() {
                    continue;
                }

                let npts = cnt.get();

                let mut pts = vec![WirePoint::zeroed(); npts];
                let hpts = comm.irecv(peer, base_tag, cast_slice_mut(pts.as_mut_slice()));
                if let Some(raw_pts) = hpts.wait() {
                    if raw_pts.len() != npts * std::mem::size_of::<WirePoint>() {
                        continue;
                    }
                    cast_slice_mut(pts.as_mut_slice()).copy_from_slice(&raw_pts);

                    let mut wires = Vec::<WireAdj>::new();
                    wires.reserve(npts * 4);
                    match hdr.kind() {
                        x if x == ReqKind::Cone as u16 => {
                            for p in &pts {
                                if let Ok(src) = PointId::new(p.get()) {
                                    for (q, _) in local_mesh.cone(src) {
                                        wires.push(WireAdj::new(p.get(), q.get()));
                                    }
                                }
                            }
                        }
                        x if x == ReqKind::Support as u16 => {
                            for p in &pts {
                                if let Ok(src) = PointId::new(p.get()) {
                                    for (q, _) in local_mesh.support(src) {
                                        wires.push(WireAdj::new(p.get(), q.get()));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    let cnt = WireCount::new(wires.len());
                    let _ = comm
                        .isend(peer, base_tag + 1, cast_slice(std::slice::from_ref(&cnt)))
                        .wait();
                    let _ = comm.isend(peer, base_tag + 2, cast_slice(&wires)).wait();
                    handled = true;
                }
            }
        }
    }
    handled
}

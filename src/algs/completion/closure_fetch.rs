// src/algs/completion/closure_fetch.rs
//! Request/response for batched cone/support fetches for selected points.
//! Used by `closure_completed(...)`.

use crate::algs::communicator::{Communicator, Wait};
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

#[repr(u8)]
#[derive(Copy, Clone)]
pub(crate) enum ReqKind {
    Cone = 1,
    Support = 2,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireHdr {
    kind: u8,
    _pad: [u8; 7],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WirePoint {
    id: u64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireAdj {
    src: u64,
    dst: u64,
}

/// Send to each owner rank the list of points to fetch; receive adjacency lists back.
/// Returns map: src_point -> Vec<dst_point>.
pub fn fetch_adjacency<C: Communicator>(
    requests: &HashMap<usize, Vec<PointId>>,
    kind: ReqKind,
    comm: &C,
    base_tag: u16,
) -> Result<HashMap<PointId, Vec<PointId>>, MeshSieveError> {
    use bytemuck::{cast_slice, cast_slice_mut};

    // 1) Post receives for replies (counts then payload)
    let mut recv_counts = Vec::new();
    for (&rank, _) in requests {
        let mut buf = [0u8; 4];
        let h = comm.irecv(rank, base_tag + 1, &mut buf);
        recv_counts.push((rank, h, buf));
    }

    // 2) Send requests (header + point list)
    let hdr = WireHdr {
        kind: kind as u8,
        _pad: [0; 7],
    };
    let mut pending_sends = Vec::new();
    let mut _keep_alive: Vec<Vec<WirePoint>> = Vec::new();
    for (&rank, pts) in requests {
        let mut body: Vec<WirePoint> = pts.iter().map(|p| WirePoint { id: p.get() }).collect();
        let bytes_hdr = cast_slice(&[hdr]);
        let bytes_pts = cast_slice(&body);
        pending_sends.push(comm.isend(rank, base_tag, bytes_hdr));
        let cnt = (body.len() as u32).to_le_bytes();
        pending_sends.push(comm.isend(rank, base_tag + 3, &cnt));
        pending_sends.push(comm.isend(rank, base_tag, bytes_pts));
        _keep_alive.push(body);
    }

    // 3) Gather reply sizes (number of WireAdj records)
    let mut counts: HashMap<usize, u32> = HashMap::new();
    for (rank, h, mut buf) in recv_counts {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: rank,
            source: Box::new(crate::mesh_error::CommError("missing count reply".into())),
        })?;
        if raw.len() != buf.len() {
            return Err(MeshSieveError::CommError {
                neighbor: rank,
                source: Box::new(crate::mesh_error::CommError(format!(
                    "expected {}B count, got {}",
                    buf.len(),
                    raw.len()
                ))),
            });
        }
        buf.copy_from_slice(&raw);
        counts.insert(rank, u32::from_le_bytes(buf));
    }

    // 4) Post receives for adjacency payloads
    let mut recv_payloads = Vec::new();
    for (&rank, &nrec) in &counts {
        let mut buf = vec![WireAdj { src: 0, dst: 0 }; nrec as usize];
        let h = comm.irecv(rank, base_tag + 2, cast_slice_mut(&mut buf));
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
        let bytes = cast_slice_mut(&mut buf);
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
        for &WireAdj { src, dst } in &buf {
            let sp = PointId::new(src).map_err(|e| MeshSieveError::MeshError(Box::new(e)))?;
            let dp = PointId::new(dst).map_err(|e| MeshSieveError::MeshError(Box::new(e)))?;
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
    use bytemuck::{cast_slice, cast_slice_mut};

    let me = comm.rank();
    let mut handled = false;
    for peer in 0..comm.size() {
        if peer == me {
            continue;
        }
        let mut hdr_buf = [0u8; std::mem::size_of::<WireHdr>()];
        let maybe = comm.irecv(peer, base_tag, &mut hdr_buf);
        if let Some(raw) = maybe.wait() {
            if raw.len() != hdr_buf.len() {
                continue;
            }
            hdr_buf.copy_from_slice(&raw);
            let hdr: WireHdr = bytemuck::pod_read_unaligned(&hdr_buf);

            let mut cnt_buf = [0u8; 4];
            let hcnt = comm.irecv(peer, base_tag + 3, &mut cnt_buf);
            if let Some(rc) = hcnt.wait() {
                if rc.len() != 4 {
                    continue;
                }
                cnt_buf.copy_from_slice(&rc);
                let npts = u32::from_le_bytes(cnt_buf) as usize;

                let mut pts = vec![WirePoint { id: 0 }; npts];
                let hpts = comm.irecv(peer, base_tag, cast_slice_mut(&mut pts));
                if let Some(raw_pts) = hpts.wait() {
                    if raw_pts.len() != npts * std::mem::size_of::<WirePoint>() {
                        continue;
                    }
                    cast_slice_mut(&mut pts).copy_from_slice(&raw_pts);

                    let mut wires = Vec::<WireAdj>::new();
                    wires.reserve(npts * 4);
                    match hdr.kind {
                        x if x == ReqKind::Cone as u8 => {
                            for &WirePoint { id } in &pts {
                                if let Ok(p) = PointId::new(id) {
                                    for (q, _) in local_mesh.cone(p) {
                                        wires.push(WireAdj {
                                            src: id,
                                            dst: q.get(),
                                        });
                                    }
                                }
                            }
                        }
                        x if x == ReqKind::Support as u8 => {
                            for &WirePoint { id } in &pts {
                                if let Ok(p) = PointId::new(id) {
                                    for (q, _) in local_mesh.support(p) {
                                        wires.push(WireAdj {
                                            src: id,
                                            dst: q.get(),
                                        });
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    let cnt = (wires.len() as u32).to_le_bytes();
                    let _ = comm.isend(peer, base_tag + 1, &cnt).wait();
                    let _ = comm.isend(peer, base_tag + 2, cast_slice(&wires)).wait();
                    handled = true;
                }
            }
        }
    }
    handled
}

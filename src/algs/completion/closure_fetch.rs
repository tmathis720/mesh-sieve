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

    // 0) Post receives for reply header + counts
    let mut recv_counts = Vec::new();
    for (&rank, _) in requests {
        let mut hdr = WireHdr::new(0);
        let mut cnt = WireCount::new(0);
        let h_hdr = comm.irecv(
            rank,
            base_tag,
            cast_slice_mut(std::slice::from_mut(&mut hdr)),
        );
        let h_cnt = comm.irecv(
            rank,
            base_tag + 1,
            cast_slice_mut(std::slice::from_mut(&mut cnt)),
        );
        recv_counts.push((rank, hdr, cnt, h_hdr, h_cnt));
    }

    // 1) Send requests (header + point list)
    let mut pending_sends = Vec::new();
    let mut _keep_pts: Vec<Vec<WirePoint>> = Vec::new();
    let mut _keep_hdrs: Vec<WireHdr> = Vec::new();
    let mut _keep_cnts: Vec<WireCount> = Vec::new();
    for (&rank, pts) in requests {
        let body: Vec<WirePoint> = pts.iter().map(|p| WirePoint::of(p.get())).collect();
        let hdr = WireHdr::new(kind as u16);
        let cnt = WireCount::new(body.len());
        pending_sends.push(comm.isend(rank, base_tag, cast_slice(std::slice::from_ref(&hdr))));
        pending_sends.push(comm.isend(rank, base_tag + 1, cast_slice(std::slice::from_ref(&cnt))));
        pending_sends.push(comm.isend(rank, base_tag, cast_slice(&body)));
        _keep_pts.push(body);
        _keep_hdrs.push(hdr);
        _keep_cnts.push(cnt);
    }

    // 2) Gather reply sizes and validate headers
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for (rank, mut hdr, mut cnt, h_hdr, h_cnt) in recv_counts {
        let raw_hdr = h_hdr.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: rank,
            source: Box::new(crate::mesh_error::CommError("missing hdr".into())),
        })?;
        let bytes_hdr = cast_slice_mut(std::slice::from_mut(&mut hdr));
        if raw_hdr.len() != bytes_hdr.len() {
            return Err(MeshSieveError::CommError {
                neighbor: rank,
                source: Box::new(crate::mesh_error::CommError(format!(
                    "expected {}B hdr, got {}",
                    bytes_hdr.len(),
                    raw_hdr.len()
                ))),
            });
        }
        bytes_hdr.copy_from_slice(&raw_hdr);
        if hdr.version() != WIRE_VERSION {
            return Err(MeshSieveError::CommError {
                neighbor: rank,
                source: Box::new(crate::mesh_error::CommError(format!(
                    "wire version mismatch: {}",
                    hdr.version()
                ))),
            });
        }

        let raw_cnt = h_cnt.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: rank,
            source: Box::new(crate::mesh_error::CommError("missing count".into())),
        })?;
        let bytes_cnt = cast_slice_mut(std::slice::from_mut(&mut cnt));
        if raw_cnt.len() != bytes_cnt.len() {
            return Err(MeshSieveError::CommError {
                neighbor: rank,
                source: Box::new(crate::mesh_error::CommError(format!(
                    "expected {}B count, got {}",
                    bytes_cnt.len(),
                    raw_cnt.len()
                ))),
            });
        }
        bytes_cnt.copy_from_slice(&raw_cnt);
        counts.insert(rank, cnt.get());
    }

    // 3) Post receives for adjacency payloads
    let mut recv_payloads = Vec::new();
    for (&rank, &nrec) in &counts {
        let mut buf = vec![WireAdj::zeroed(); nrec];
        let h = comm.irecv(rank, base_tag + 2, cast_slice_mut(buf.as_mut_slice()));
        recv_payloads.push((rank, h, buf));
    }

    // 4) Wait for sends to finish
    for s in pending_sends {
        let _ = s.wait();
    }

    // 5) Unpack replies
    let mut out: HashMap<PointId, Vec<PointId>> = HashMap::new();
    for (rank, h, mut buf) in recv_payloads {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: rank,
            source: Box::new(crate::mesh_error::CommError("missing payload".into())),
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

use crate::algs::communicator::PollWait;
use bytemuck::{cast_slice, cast_slice_mut, Zeroable};

enum Stage<C: Communicator> {
    WaitingHdr {
        h: C::RecvHandle,
        buf: WireHdr,
    },
    WaitingCnt {
        h: C::RecvHandle,
        hdr: WireHdr,
        cnt: WireCount,
    },
    WaitingPts {
        h: C::RecvHandle,
        hdr: WireHdr,
        cnt: WireCount,
        pts: Vec<WirePoint>,
    },
}

struct PeerState<C: Communicator> {
    peer: usize,
    stage: Stage<C>,
}

struct MeshFetchServer<C: Communicator, S: Sieve<Point = PointId> + Clone> {
    states: Vec<PeerState<C>>,
    comm: C,
    mesh: S,
    base_tag: u16,
}

impl<C, S> MeshFetchServer<C, S>
where
    C: Communicator,
    C::RecvHandle: PollWait,
    S: Sieve<Point = PointId> + Clone,
{
    fn new(comm: C, mesh: S, base_tag: u16) -> Self {
        let me = comm.rank();
        let mut states = Vec::new();
        for peer in 0..comm.size() {
            if peer == me {
                continue;
            }
            let mut hdr = WireHdr::new(0);
            let h = comm.irecv(
                peer,
                base_tag,
                cast_slice_mut(std::slice::from_mut(&mut hdr)),
            );
            states.push(PeerState {
                peer,
                stage: Stage::WaitingHdr { h, buf: hdr },
            });
        }
        Self {
            states,
            comm,
            mesh,
            base_tag,
        }
    }

    fn poll(&mut self) -> usize {
        let mut completed = 0;
        for ps in &mut self.states {
            match &mut ps.stage {
                Stage::WaitingHdr { h, buf } => {
                    if let Some(raw) = h.try_wait() {
                        let bytes = cast_slice_mut(std::slice::from_mut(buf));
                        if raw.len() == bytes.len() {
                            bytes.copy_from_slice(&raw);
                            if buf.version() == WIRE_VERSION {
                                let mut cnt = WireCount::new(0);
                                let hcnt = self.comm.irecv(
                                    ps.peer,
                                    self.base_tag + 1,
                                    cast_slice_mut(std::slice::from_mut(&mut cnt)),
                                );
                                ps.stage = Stage::WaitingCnt {
                                    h: hcnt,
                                    hdr: *buf,
                                    cnt,
                                };
                                continue;
                            }
                        }
                        let mut hdr2 = WireHdr::new(0);
                        let h2 = self.comm.irecv(
                            ps.peer,
                            self.base_tag,
                            cast_slice_mut(std::slice::from_mut(&mut hdr2)),
                        );
                        ps.stage = Stage::WaitingHdr { h: h2, buf: hdr2 };
                    }
                }
                Stage::WaitingCnt { h, hdr, cnt } => {
                    if let Some(raw) = h.try_wait() {
                        let bytes = cast_slice_mut(std::slice::from_mut(cnt));
                        if raw.len() == bytes.len() {
                            bytes.copy_from_slice(&raw);
                            let n = cnt.get();
                            let mut pts = vec![WirePoint::zeroed(); n];
                            let hp = self.comm.irecv(
                                ps.peer,
                                self.base_tag,
                                cast_slice_mut(pts.as_mut_slice()),
                            );
                            ps.stage = Stage::WaitingPts {
                                h: hp,
                                hdr: *hdr,
                                cnt: *cnt,
                                pts,
                            };
                            continue;
                        }
                        let mut hdr2 = WireHdr::new(0);
                        let h2 = self.comm.irecv(
                            ps.peer,
                            self.base_tag,
                            cast_slice_mut(std::slice::from_mut(&mut hdr2)),
                        );
                        ps.stage = Stage::WaitingHdr { h: h2, buf: hdr2 };
                    }
                }
                Stage::WaitingPts { h, hdr, cnt, pts } => {
                    if let Some(raw) = h.try_wait() {
                        if raw.len() == cnt.get() * std::mem::size_of::<WirePoint>() {
                            cast_slice_mut(pts.as_mut_slice()).copy_from_slice(&raw);
                            let mut wires = Vec::<WireAdj>::new();
                            wires.reserve(cnt.get() * 4);
                            match hdr.kind() {
                                x if x == ReqKind::Cone as u16 => {
                                    for p in pts.iter() {
                                        if let Ok(src) = PointId::new(p.get()) {
                                            for (q, _) in self.mesh.cone(src) {
                                                wires.push(WireAdj::new(p.get(), q.get()));
                                            }
                                        }
                                    }
                                }
                                x if x == ReqKind::Support as u16 => {
                                    for p in pts.iter() {
                                        if let Ok(src) = PointId::new(p.get()) {
                                            for (q, _) in self.mesh.support(src) {
                                                wires.push(WireAdj::new(p.get(), q.get()));
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                            let hdr_out = WireHdr::new(hdr.kind());
                            let cnt_out = WireCount::new(wires.len());
                            let _ = self.comm.isend(
                                ps.peer,
                                self.base_tag,
                                cast_slice(std::slice::from_ref(&hdr_out)),
                            );
                            let _ = self.comm.isend(
                                ps.peer,
                                self.base_tag + 1,
                                cast_slice(std::slice::from_ref(&cnt_out)),
                            );
                            let _ = self
                                .comm
                                .isend(ps.peer, self.base_tag + 2, cast_slice(&wires));
                            let mut hdr2 = WireHdr::new(0);
                            let h2 = self.comm.irecv(
                                ps.peer,
                                self.base_tag,
                                cast_slice_mut(std::slice::from_mut(&mut hdr2)),
                            );
                            ps.stage = Stage::WaitingHdr { h: h2, buf: hdr2 };
                            completed += 1;
                        } else {
                            let mut hdr2 = WireHdr::new(0);
                            let h2 = self.comm.irecv(
                                ps.peer,
                                self.base_tag,
                                cast_slice_mut(std::slice::from_mut(&mut hdr2)),
                            );
                            ps.stage = Stage::WaitingHdr { h: h2, buf: hdr2 };
                        }
                    }
                }
            }
        }
        completed
    }
}

/// ### Progress model
/// Call repeatedly to make non-blocking progress on pending fetch requests.
/// Each invocation consumes at most one ready message per peer and never
/// blocks. Returns `true` if any request was fully handled.
pub fn service_once_mesh_fetch<C, S>(comm: &C, local_mesh: &S, base_tag: u16) -> bool
where
    C: Communicator + Clone + Send + 'static,
    C::RecvHandle: PollWait + Send,
    S: Sieve<Point = PointId> + Clone + Send + 'static,
{
    use once_cell::sync::OnceCell;
    use std::any::Any;
    use std::sync::Mutex;

    static SERVER: OnceCell<Mutex<Box<dyn Any + Send>>> = OnceCell::new();
    let cell = SERVER.get_or_init(|| {
        let server = MeshFetchServer::new(comm.clone(), local_mesh.clone(), base_tag);
        Mutex::new(Box::new(server) as Box<dyn Any + Send>)
    });
    let mut g = cell.lock().expect("server mutex poisoned");
    let server = g
        .downcast_mut::<MeshFetchServer<C, S>>()
        .expect("type mismatch");
    server.poll() > 0
}

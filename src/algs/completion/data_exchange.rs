//! Stage 2 of section completion: exchange the actual data items.

use crate::algs::communicator::Wait;
use crate::mesh_error::MeshSieveError;

/// For each neighbor, pack `Delta::restrict` from your section into a send buffer,
/// post irecv for the corresponding byte length (from stage 1),
/// then send and finally wait + `Delta::fuse` into your local section.
pub fn exchange_data<V, D, C>(
    links: &std::collections::HashMap<
        usize,
        Vec<(crate::topology::point::PointId, crate::topology::point::PointId)>,
    >,
    recv_counts: &std::collections::HashMap<usize, u32>,
    comm: &C,
    base_tag: u16,
    section: &mut crate::data::section::Section<V>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    use bytemuck::cast_slice;
    use std::collections::HashMap;

    // --- Stage 2: exchange data (asymmetric) ---
    // 1) post all receives
    let mut recv_data: HashMap<usize, (C::RecvHandle, Vec<u8>)> = HashMap::new();
    for &nbr in links.keys() {
        let n_items = *recv_counts
            .get(&nbr)
            .ok_or(MeshSieveError::MissingRecvCount { neighbor: nbr })?
            as usize;
        let mut buffer = vec![0u8; n_items * std::mem::size_of::<D::Part>()];
        let h = comm.irecv(nbr, base_tag, &mut buffer);
        recv_data.insert(nbr, (h, buffer));
    }

    // 2) post all sends
    let mut pending_sends = Vec::with_capacity(recv_counts.len());
    for &nbr in recv_counts.keys() {
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut scratch = Vec::with_capacity(link_vec.len());
        for &(send_loc, _) in link_vec {
            let slice = section
                .try_restrict(send_loc)
                .map_err(|e| MeshSieveError::SectionAccess {
                    point: send_loc,
                    source: Box::new(e),
                })?;
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        pending_sends.push(comm.isend(nbr, base_tag, bytes));
    }

    // 3) wait for all recvs and fuse
    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: nbr,
            source: "No data received (wait returned None)".into(),
        })?;
        if raw.len() != buffer.len() {
            return Err(MeshSieveError::BufferSizeMismatch {
                neighbor: nbr,
                expected: buffer.len(),
                got: raw.len(),
            });
        }
        buffer.copy_from_slice(&raw);
        let parts: &[D::Part] = cast_slice(&buffer);
        let link_vec = &links[&nbr];
        if parts.len() != link_vec.len() {
            return Err(MeshSieveError::PartCountMismatch {
                neighbor: nbr,
                expected: link_vec.len(),
                got: parts.len(),
            });
        }
        for ((_, recv_loc), part) in link_vec.iter().zip(parts) {
            let mut_slice = section
                .try_restrict_mut(*recv_loc)
                .map_err(|e| MeshSieveError::SectionAccess {
                    point: *recv_loc,
                    source: Box::new(e),
                })?;
            D::fuse(&mut mut_slice[0], *part);
        }
    }

    // 4) then wait for every send to complete
    for send in pending_sends {
        let _ = send.wait();
    }

    Ok(())
}

/// Symmetric version: post _both_ send+recv to _every_ neighbor (even if count == 0),
/// so that no rank ever blocks waiting for a peer that never sends.
pub fn exchange_data_symmetric<V, D, C>(
    links: &std::collections::HashMap<
        usize,
        Vec<(crate::topology::point::PointId, crate::topology::point::PointId)>,
    >,
    recv_counts: &std::collections::HashMap<usize, u32>,
    comm: &C,
    base_tag: u16,
    section: &mut crate::data::section::Section<V>,
    all_neighbors: &std::collections::HashSet<usize>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    use bytemuck::{cast_slice, cast_slice_mut};
    use std::collections::HashMap;

    // 1) post all recvs (possibly zero‐length)
    let mut recv_data: HashMap<usize, (C::RecvHandle, Vec<D::Part>)> = HashMap::new();
    for &nbr in all_neighbors {
        let n_items = recv_counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buffer = vec![D::Part::default(); n_items];
        let h = comm.irecv(nbr, base_tag, cast_slice_mut(&mut buffer));
        recv_data.insert(nbr, (h, buffer));
    }

    // 2) pack & post all sends
    let mut pending_sends = Vec::with_capacity(all_neighbors.len());
    let mut _keep_alive = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut scratch = Vec::with_capacity(link_vec.len());
        for &(send_loc, _) in link_vec {
            let slice = section
                .try_restrict(send_loc)
                .map_err(|e| MeshSieveError::SectionAccess {
                    point: send_loc,
                    source: Box::new(e),
                })?;
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        pending_sends.push(if bytes.is_empty() {
            comm.isend(nbr, base_tag, &[])
        } else {
            comm.isend(nbr, base_tag, bytes)
        });
        _keep_alive.push(scratch); // ensure data outlives the nonblocking send
    }

    // 3) wait+fuse all recvs, capturing only the first error
    let mut maybe_err: Option<MeshSieveError> = None;
    let mut recvs: Vec<_> = recv_data.into_iter().collect();
    for (nbr, (h, mut buffer)) in recvs.drain(..) {
        match h.wait() {
            Some(raw) if maybe_err.is_none() => {
                let buf_bytes = cast_slice_mut(&mut buffer);
                if raw.len() != buf_bytes.len() {
                    maybe_err = Some(MeshSieveError::BufferSizeMismatch {
                        neighbor: nbr,
                        expected: buf_bytes.len(),
                        got: raw.len(),
                    });
                } else {
                    buf_bytes.copy_from_slice(&raw);
                    let parts: &[D::Part] = &buffer;
                    let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
                    if parts.len() != link_vec.len() {
                        maybe_err = Some(MeshSieveError::PartCountMismatch {
                            neighbor: nbr,
                            expected: link_vec.len(),
                            got: parts.len(),
                        });
                    } else {
                        for ((_, recv_loc), part) in link_vec.iter().zip(parts) {
                            match section.try_restrict_mut(*recv_loc) {
                                Ok(mut_slice) => D::fuse(&mut mut_slice[0], *part),
                                Err(e) => maybe_err = Some(MeshSieveError::SectionAccess {
                                    point: *recv_loc,
                                    source: Box::new(e),
                                }),
                            }
                        }
                    }
                }
            }
            Some(_) | None if maybe_err.is_none() => {
                // record first communication error
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: "No data received (wait returned None)".into(),
                });
            }
            _ => {}
        }
    }

    // 4) always wait for every send
    for send in pending_sends {
        let _ = send.wait();
    }

    // 5) return error if any
    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    struct DummyValue(pub i32);

    // Dummy Delta implementation
    struct DummyDelta;
    impl crate::overlap::delta::ValueDelta<DummyValue> for DummyDelta {
        type Part = i32;
        fn restrict(v: &DummyValue) -> i32 {
            v.0
        }
        fn fuse(v: &mut DummyValue, part: i32) {
            v.0 += part;
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    #[ignore]
    fn test_exchange_data_rayon_comm() {
        use crate::topology::point::PointId;
        use crate::data::atlas::Atlas;
        use crate::data::section::Section;
        use std::collections::HashMap;
        use crate::algs::completion::data_exchange::exchange_data;
        // Setup: two ranks (0 and 1), each with one value to send/receive
        let base_tag = 42;
        // Use RayonComm for realistic intra-process comm
        let comm0 = crate::algs::communicator::RayonComm::new(0, 2);
        let comm1 = crate::algs::communicator::RayonComm::new(1, 2);

        // Build Atlas and Section for each rank
        let mut atlas0 = Atlas::default();
        atlas0.try_insert(PointId::new(10).unwrap(), 1);
        atlas0.try_insert(PointId::new(20).unwrap(), 1);
        let mut section0 = Section::new(atlas0);
        section0.try_set(PointId::new(10).unwrap(), &[DummyValue(5)]);
        section0.try_set(PointId::new(20).unwrap(), &[DummyValue(0)]);
        let mut atlas1 = Atlas::default();
        atlas1.try_insert(PointId::new(10).unwrap(), 1);
        atlas1.try_insert(PointId::new(20).unwrap(), 1);
        let mut section1 = Section::new(atlas1);
        section1.try_set(PointId::new(10).unwrap(), &[DummyValue(0)]);
        section1.try_set(PointId::new(20).unwrap(), &[DummyValue(7)]);

        // Links and recv_counts for each rank
        let mut links0 = HashMap::new();
        links0.insert(1, vec![(PointId::new(10).unwrap(), PointId::new(20).unwrap())]);
        let mut recv_counts0 = HashMap::new();
        recv_counts0.insert(1, 1);
        let mut links1 = HashMap::new();
        links1.insert(0, vec![(PointId::new(20).unwrap(), PointId::new(10).unwrap())]);
        let mut recv_counts1 = HashMap::new();
        recv_counts1.insert(0, 1);

        // Spawn threads for each rank
        let t0 = std::thread::spawn(move || {
            exchange_data::<DummyValue, DummyDelta, crate::algs::communicator::RayonComm>(
                &links0,
                &recv_counts0,
                &comm0,
                base_tag,
                &mut section0,
            ).unwrap();
            (
                section0.try_restrict(PointId::new(10).unwrap()).unwrap()[0],
                section0.try_restrict(PointId::new(20).unwrap()).unwrap()[0],
            )
        });
        let t1 = std::thread::spawn(move || {
            exchange_data::<DummyValue, DummyDelta, crate::algs::communicator::RayonComm>(
                &links1,
                &recv_counts1,
                &comm1,
                base_tag,
                &mut section1,
            ).unwrap();
            (
                section1.try_restrict(PointId::new(10).unwrap()).unwrap()[0],
                section1.try_restrict(PointId::new(20).unwrap()).unwrap()[0],
            )
        });
        let t0_res = t0.join();
        let t1_res = t1.join();
        match (t0_res, t1_res) {
            (Ok((s0_10, s0_20)), Ok((s1_10, s1_20))) => {
                // Rank 0 should have received 7 into 20
                assert_eq!(s0_10, DummyValue(5));
                assert_eq!(s0_20, DummyValue(7));
                // Rank 1 should have received 5 into 10
                assert_eq!(s1_10, DummyValue(5));
                assert_eq!(s1_20, DummyValue(7));
            }
            (Err(e0), Err(e1)) => {
                panic!("test_exchange_data_rayon_comm: both threads panicked: {:?} | {:?}", e0, e1);
            }
            (Err(e0), _) => {
                panic!("test_exchange_data_rayon_comm: thread 0 panicked: {:?}", e0);
            }
            (_, Err(e1)) => {
                panic!("test_exchange_data_rayon_comm: thread 1 panicked: {:?}", e1);
            }
            _ => panic!("test_exchange_data_rayon_comm: thread join failed or deadlocked"),
            // unreachable: all cases are covered above
        }
    }
}

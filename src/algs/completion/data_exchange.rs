//! Stage 2 of section completion: exchange the actual data items.

use crate::algs::communicator::Wait;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;

/// For each neighbor, pack `Delta::restrict` from your section into a send buffer,
/// post irecv for the corresponding byte length (from stage 1),
/// then send and finally wait + `Delta::fuse` into your local section.
///
/// This function assumes each point slice has **exactly one degree of freedom**.
/// For vector DOFs use [`exchange_data_symmetric`], which supports arbitrary
/// slice lengths via a length-prefixed protocol.
pub fn exchange_data<V, S, D, C>(
    links: &std::collections::HashMap<
        usize,
        Vec<(
            crate::topology::point::PointId,
            crate::topology::point::PointId,
        )>,
    >,
    recv_counts: &std::collections::HashMap<usize, u32>,
    comm: &C,
    base_tag: u16,
    section: &mut crate::data::section::Section<V, S>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + 'static,
    S: Storage<V>,
    D: crate::overlap::delta::ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    use crate::algs::wire::cast_slice;
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
        let h = comm.irecv_result(nbr, base_tag, &mut buffer)?;
        recv_data.insert(nbr, (h, buffer));
    }

    // 2) post all sends
    let mut pending_sends = Vec::with_capacity(recv_counts.len());
    for &nbr in recv_counts.keys() {
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut scratch = Vec::with_capacity(link_vec.len());
        for &(send_loc, _) in link_vec {
            let slice =
                section
                    .try_restrict(send_loc)
                    .map_err(|e| MeshSieveError::SectionAccess {
                        point: send_loc,
                        source: Box::new(e),
                    })?;
            debug_assert_eq!(slice.len(), 1, "exchange_data assumes scalar DOFs");
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        pending_sends.push(comm.isend_result(nbr, base_tag, bytes)?);
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
        let parts: &[D::Part] = crate::algs::wire::cast_slice_from(&buffer);
        let link_vec = &links[&nbr];
        if parts.len() != link_vec.len() {
            return Err(MeshSieveError::PartCountMismatch {
                neighbor: nbr,
                expected: link_vec.len(),
                got: parts.len(),
            });
        }
        for ((_, recv_loc), part) in link_vec.iter().zip(parts) {
            let mut_slice =
                section
                    .try_restrict_mut(*recv_loc)
                    .map_err(|e| MeshSieveError::SectionAccess {
                        point: *recv_loc,
                        source: Box::new(e),
                    })?;
            debug_assert_eq!(mut_slice.len(), 1, "exchange_data assumes scalar DOFs");
            D::fuse(&mut mut_slice[0], *part);
        }
    }

    // 4) then wait for every send to complete
    for send in pending_sends {
        let _ = send.wait();
    }

    Ok(())
}

/// Symmetric version that supports vector DOFs per point.
///
/// For each neighbor we exchange:
/// 1. point counts (from `exchange_sizes_symmetric`),
/// 2. a `u32` length per point, and
/// 3. a flat payload of `D::Part` values.
pub fn exchange_data_symmetric<V, S, D, C>(
    links: &std::collections::HashMap<
        usize,
        Vec<(
            crate::topology::point::PointId,
            crate::topology::point::PointId,
        )>,
    >,
    recv_counts: &std::collections::HashMap<usize, u32>,
    comm: &C,
    base_tag: u16,
    section: &mut crate::data::section::Section<V, S>,
    all_neighbors: &std::collections::HashSet<usize>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + 'static,
    S: Storage<V>,
    D: crate::overlap::delta::ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default + Copy,
    C: crate::algs::communicator::Communicator + Sync,
{
    use crate::algs::wire::{cast_slice, cast_slice_mut};
    use std::collections::HashMap;

    let tag_len = base_tag;
    let tag_data = base_tag + 1;

    // lengths phase
    let mut recv_lens: HashMap<usize, (C::RecvHandle, Vec<u32>)> = HashMap::new();
    for &nbr in all_neighbors {
        let n_points = recv_counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![0u32; n_points];
        let h = comm.irecv_result(nbr, tag_len, cast_slice_mut(&mut buf))?;
        recv_lens.insert(nbr, (h, buf));
    }

    let mut pending_len_sends = Vec::with_capacity(all_neighbors.len());
    let mut keep_len_send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut lens_out = Vec::<u32>::with_capacity(link_vec.len());
        for &(send_loc, _) in link_vec {
            let slice =
                section
                    .try_restrict(send_loc)
                    .map_err(|e| MeshSieveError::SectionAccess {
                        point: send_loc,
                        source: Box::new(e),
                    })?;
            lens_out.push(u32::try_from(slice.len()).unwrap_or(u32::MAX));
        }
        let bytes = cast_slice(&lens_out);
        pending_len_sends.push(comm.isend_result(nbr, tag_len, bytes)?);
        keep_len_send_bufs.push(lens_out);
    }

    let mut lens_in: HashMap<usize, Vec<u32>> = HashMap::with_capacity(all_neighbors.len());
    let mut first_err: Option<MeshSieveError> = None;
    for (nbr, (h, mut buf)) in recv_lens {
        let raw = match h.wait() {
            Some(r) => r,
            None => {
                if first_err.is_none() {
                    first_err = Some(MeshSieveError::CommError {
                        neighbor: nbr,
                        source: "lengths recv returned None".into(),
                    });
                }
                continue;
            }
        };
        let expect_bytes = buf.len() * std::mem::size_of::<u32>();
        if raw.len() != expect_bytes {
            if first_err.is_none() {
                first_err = Some(MeshSieveError::BufferSizeMismatch {
                    neighbor: nbr,
                    expected: expect_bytes,
                    got: raw.len(),
                });
            }
            continue;
        }
        cast_slice_mut(&mut buf).copy_from_slice(&raw);
        if buf.len() != recv_counts.get(&nbr).copied().unwrap_or(0) as usize {
            if first_err.is_none() {
                first_err = Some(MeshSieveError::LengthsCountMismatch {
                    neighbor: nbr,
                    expected: recv_counts.get(&nbr).copied().unwrap_or(0) as usize,
                    got: buf.len(),
                });
            }
            continue;
        }
        lens_in.insert(nbr, buf);
    }
    for s in pending_len_sends {
        let _ = s.wait();
    }
    drop(keep_len_send_bufs);
    if let Some(e) = first_err {
        return Err(e);
    }

    // payload phase
    let mut recv_parts: HashMap<usize, (C::RecvHandle, Vec<D::Part>)> = HashMap::new();
    for (&nbr, lens) in &lens_in {
        let total_in: usize = lens.iter().map(|&x| x as usize).sum();
        let mut buf = vec![D::Part::default(); total_in];
        let h = comm.irecv_result(nbr, tag_data, cast_slice_mut(&mut buf))?;
        recv_parts.insert(nbr, (h, buf));
    }

    let mut pending_data_sends = Vec::with_capacity(all_neighbors.len());
    let mut keep_data_send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut flat = Vec::<D::Part>::new();
        for &(send_loc, _) in link_vec {
            let slice =
                section
                    .try_restrict(send_loc)
                    .map_err(|e| MeshSieveError::SectionAccess {
                        point: send_loc,
                        source: Box::new(e),
                    })?;
            for v in slice {
                flat.push(D::restrict(v));
            }
        }
        let bytes = cast_slice(&flat);
        pending_data_sends.push(comm.isend_result(nbr, tag_data, bytes)?);
        keep_data_send_bufs.push(flat);
    }

    let mut first_err: Option<MeshSieveError> = None;
    for (nbr, (h, mut buf)) in recv_parts {
        let raw = match h.wait() {
            Some(r) => r,
            None => {
                if first_err.is_none() {
                    first_err = Some(MeshSieveError::CommError {
                        neighbor: nbr,
                        source: "payload recv returned None".into(),
                    });
                }
                continue;
            }
        };
        let expect_bytes = buf.len() * std::mem::size_of::<D::Part>();
        if raw.len() != expect_bytes {
            if first_err.is_none() {
                first_err = Some(MeshSieveError::BufferSizeMismatch {
                    neighbor: nbr,
                    expected: expect_bytes,
                    got: raw.len(),
                });
            }
            continue;
        }
        cast_slice_mut(&mut buf).copy_from_slice(&raw);

        let lens = &lens_in[&nbr];
        let pairs = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        if lens.len() != pairs.len() {
            if first_err.is_none() {
                first_err = Some(MeshSieveError::LengthsCountMismatch {
                    neighbor: nbr,
                    expected: pairs.len(),
                    got: lens.len(),
                });
            }
            continue;
        }

        let mut cursor = 0usize;
        let mut neighbor_failed = false;
        for ((_, recv_loc), &m_u32) in pairs.iter().zip(lens) {
            let m = m_u32 as usize;
            let dst = match section.try_restrict_mut(*recv_loc) {
                Ok(d) => d,
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(MeshSieveError::SectionAccess {
                            point: *recv_loc,
                            source: Box::new(e),
                        });
                    }
                    neighbor_failed = true;
                    break;
                }
            };
            if dst.len() != m {
                if first_err.is_none() {
                    first_err = Some(MeshSieveError::PayloadCountMismatch {
                        neighbor: nbr,
                        expected: m,
                        got: dst.len(),
                    });
                }
                neighbor_failed = true;
                break;
            }
            let chunk = &buf[cursor..cursor + m];
            for (d, &part) in dst.iter_mut().zip(chunk.iter()) {
                D::fuse(d, part);
            }
            cursor += m;
        }
        if neighbor_failed {
            continue;
        }
        debug_assert_eq!(cursor, buf.len(), "payload cursor mismatch");
    }

    for s in pending_data_sends {
        let _ = s.wait();
    }
    drop(keep_data_send_bufs);

    if let Some(e) = first_err {
        return Err(e);
    }

    Ok(())
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
        use crate::algs::completion::data_exchange::exchange_data;
        use crate::data::atlas::Atlas;
        use crate::data::section::Section;
        use crate::data::storage::VecStorage;
        use crate::topology::point::PointId;
        use std::collections::HashMap;
        // Setup: two ranks (0 and 1), each with one value to send/receive
        let base_tag = 42;
        // Use RayonComm for realistic intra-process comm
        let comm0 = crate::algs::communicator::RayonComm::new(0, 2);
        let comm1 = crate::algs::communicator::RayonComm::new(1, 2);

        // Build Atlas and Section for each rank
        let mut atlas0 = Atlas::default();
        atlas0.try_insert(PointId::new(10).unwrap(), 1);
        atlas0.try_insert(PointId::new(20).unwrap(), 1);
        let mut section0 = Section::<DummyValue, VecStorage<DummyValue>>::new(atlas0);
        section0.try_set(PointId::new(10).unwrap(), &[DummyValue(5)]);
        section0.try_set(PointId::new(20).unwrap(), &[DummyValue(0)]);
        let mut atlas1 = Atlas::default();
        atlas1.try_insert(PointId::new(10).unwrap(), 1);
        atlas1.try_insert(PointId::new(20).unwrap(), 1);
        let mut section1 = Section::<DummyValue, VecStorage<DummyValue>>::new(atlas1);
        section1.try_set(PointId::new(10).unwrap(), &[DummyValue(0)]);
        section1.try_set(PointId::new(20).unwrap(), &[DummyValue(7)]);

        // Links and recv_counts for each rank
        let mut links0 = HashMap::new();
        links0.insert(
            1,
            vec![(PointId::new(10).unwrap(), PointId::new(20).unwrap())],
        );
        let mut recv_counts0 = HashMap::new();
        recv_counts0.insert(1, 1);
        let mut links1 = HashMap::new();
        links1.insert(
            0,
            vec![(PointId::new(20).unwrap(), PointId::new(10).unwrap())],
        );
        let mut recv_counts1 = HashMap::new();
        recv_counts1.insert(0, 1);

        // Spawn threads for each rank
        let t0 = std::thread::spawn(move || {
            exchange_data::<
                DummyValue,
                VecStorage<DummyValue>,
                DummyDelta,
                crate::algs::communicator::RayonComm,
            >(&links0, &recv_counts0, &comm0, base_tag, &mut section0)
            .unwrap();
            (
                section0.try_restrict(PointId::new(10).unwrap()).unwrap()[0],
                section0.try_restrict(PointId::new(20).unwrap()).unwrap()[0],
            )
        });
        let t1 = std::thread::spawn(move || {
            exchange_data::<
                DummyValue,
                VecStorage<DummyValue>,
                DummyDelta,
                crate::algs::communicator::RayonComm,
            >(&links1, &recv_counts1, &comm1, base_tag, &mut section1)
            .unwrap();
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
                panic!(
                    "test_exchange_data_rayon_comm: both threads panicked: {:?} | {:?}",
                    e0, e1
                );
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

    #[cfg(feature = "rayon")]
    #[test]
    fn test_exchange_data_symmetric_vector_dofs() {
        use crate::algs::completion::data_exchange::exchange_data_symmetric;
        use crate::data::atlas::Atlas;
        use crate::data::section::Section;
        use crate::topology::point::PointId;
        use std::collections::{HashMap, HashSet};

        let base_tag = 100u16;
        let comm0 = crate::algs::communicator::RayonComm::new(0, 2);
        let comm1 = crate::algs::communicator::RayonComm::new(1, 2);

        let mut atlas0 = Atlas::default();
        atlas0.try_insert(PointId::new(10).unwrap(), 3);
        atlas0.try_insert(PointId::new(20).unwrap(), 2);
        let mut section0 = Section::new(atlas0);
        section0
            .try_set(
                PointId::new(10).unwrap(),
                &[DummyValue(1), DummyValue(2), DummyValue(3)],
            )
            .unwrap();
        section0
            .try_set(PointId::new(20).unwrap(), &[DummyValue(0), DummyValue(0)])
            .unwrap();

        let mut atlas1 = Atlas::default();
        atlas1.try_insert(PointId::new(10).unwrap(), 3);
        atlas1.try_insert(PointId::new(20).unwrap(), 2);
        let mut section1 = Section::new(atlas1);
        section1
            .try_set(
                PointId::new(10).unwrap(),
                &[DummyValue(0), DummyValue(0), DummyValue(0)],
            )
            .unwrap();
        section1
            .try_set(PointId::new(20).unwrap(), &[DummyValue(4), DummyValue(5)])
            .unwrap();

        let mut links0 = HashMap::new();
        links0.insert(
            1,
            vec![(PointId::new(10).unwrap(), PointId::new(20).unwrap())],
        );
        let mut links1 = HashMap::new();
        links1.insert(
            0,
            vec![(PointId::new(20).unwrap(), PointId::new(10).unwrap())],
        );

        let mut recv_counts0 = HashMap::new();
        recv_counts0.insert(1, 1u32);
        let mut recv_counts1 = HashMap::new();
        recv_counts1.insert(0, 1u32);

        let all0: HashSet<usize> = [1usize].into_iter().collect();
        let all1: HashSet<usize> = [0usize].into_iter().collect();

        let t0 = std::thread::spawn(move || {
            exchange_data_symmetric::<
                DummyValue,
                crate::data::storage::VecStorage<DummyValue>,
                DummyDelta,
                crate::algs::communicator::RayonComm,
            >(
                &links0,
                &recv_counts0,
                &comm0,
                base_tag,
                &mut section0,
                &all0,
            )
            .unwrap();
            section0
                .try_restrict(PointId::new(20).unwrap())
                .unwrap()
                .to_vec()
        });
        let t1 = std::thread::spawn(move || {
            exchange_data_symmetric::<
                DummyValue,
                crate::data::storage::VecStorage<DummyValue>,
                DummyDelta,
                crate::algs::communicator::RayonComm,
            >(
                &links1,
                &recv_counts1,
                &comm1,
                base_tag,
                &mut section1,
                &all1,
            )
            .unwrap();
            section1
                .try_restrict(PointId::new(10).unwrap())
                .unwrap()
                .to_vec()
        });

        let recv0 = t0.join().unwrap();
        let recv1 = t1.join().unwrap();

        assert_eq!(recv0, vec![DummyValue(4), DummyValue(5)]);
        assert_eq!(recv1, vec![DummyValue(1), DummyValue(2), DummyValue(3)]);
    }
}

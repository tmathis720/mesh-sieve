//! Stage 2 of section completion: exchange the actual data items.

use crate::algs::communicator::Wait;
use crate::mesh_error::MeshSieveError;

/// For each neighbor, pack `Delta::restrict` from your section into a send buffer,
/// post irecv for the corresponding byte length (from stage 1),
/// then send and finally wait + `Delta::fuse` into your local section.
pub fn exchange_data<V, D, C>(
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
    section: &mut crate::data::section::Section<V>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::Delta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    use bytemuck::cast_slice;
    use std::collections::HashMap;
    // --- Stage 2: exchange data ---
    let mut recv_data = HashMap::new();
    for &nbr in links.keys() {
        let n_items = *recv_counts.get(&nbr).ok_or(MeshSieveError::MissingRecvCount { neighbor: nbr })? as usize;
        let mut buffer = vec![0u8; n_items * std::mem::size_of::<D::Part>()];
        let h = comm.irecv(nbr, base_tag, &mut buffer);
        recv_data.insert(nbr, (h, buffer));
    }
    for (&nbr, links) in links {
        let mut scratch = Vec::with_capacity(links.len());
        for &(loc, _) in links {
            let slice = section.try_restrict(loc).map_err(|e| MeshSieveError::SectionAccess { point: loc, source: Box::new(e) })?;
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        if !bytes.is_empty() {
            comm.isend(nbr, base_tag, bytes);
        }
    }
    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: nbr,
            source: "No data received (wait returned None)".into(),
        })?;
        if raw.len() != buffer.len() {
            return Err(MeshSieveError::BufferSizeMismatch { neighbor: nbr, expected: buffer.len(), got: raw.len() });
        }
        buffer.copy_from_slice(&raw);
        let parts: &[D::Part] = cast_slice(&buffer);
        let links = &links[&nbr];
        if parts.len() != links.len() {
            return Err(MeshSieveError::PartCountMismatch { neighbor: nbr, expected: links.len(), got: parts.len() });
        }
        for ((_, dst), part) in links.iter().zip(parts) {
            let mut_slice = section.try_restrict_mut(*dst).map_err(|e| MeshSieveError::SectionAccess { point: *dst, source: Box::new(e) })?;
            D::fuse(&mut mut_slice[0], *part);
        }
    }
    Ok(())
}

/// For each neighbor, pack `Delta::restrict` from your section into a send buffer,
/// post irecv for the corresponding byte length (from stage 1),
/// then send and finally wait + `Delta::fuse` into your local section.
/// This version always posts send/recv for all neighbors, even if count is zero,
/// to prevent deadlocks in section completion.
pub fn exchange_data_symmetric<V, D, C>(
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
    section: &mut crate::data::section::Section<V>,
    all_neighbors: &std::collections::HashSet<usize>,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + 'static,
    D: crate::overlap::delta::Delta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default,
    C: crate::algs::communicator::Communicator + Sync,
{
    use bytemuck::cast_slice;
    use bytemuck::cast_slice_mut;
    use std::collections::HashMap;
    let mut recv_data = HashMap::new();
    let mut pending_sends = Vec::new();
    for &nbr in all_neighbors {
        let n_items = recv_counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buffer = vec![D::Part::default(); n_items];
        let h = comm.irecv(nbr, base_tag, cast_slice_mut(&mut buffer));
        recv_data.insert(nbr, (h, buffer));
    }
    for &nbr in all_neighbors {
        let links_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut scratch = Vec::with_capacity(links_vec.len());
        for &(loc, _) in links_vec {
            let slice = section.try_restrict(loc).map_err(|e| MeshSieveError::SectionAccess { point: loc, source: Box::new(e) })?;
            scratch.push(D::restrict(&slice[0]));
        }
        let bytes = cast_slice(&scratch);
        if !bytes.is_empty() {
            pending_sends.push(comm.isend(nbr, base_tag, bytes));
        } else {
            // Always post a send, even if empty, to match the symmetric protocol
            pending_sends.push(comm.isend(nbr, base_tag, &[]));
        }
    }
    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: nbr,
            source: "No data received (wait returned None)".into(),
        })?;
        let buf_bytes = cast_slice_mut(&mut buffer);
        if raw.len() != buf_bytes.len() {
            return Err(MeshSieveError::BufferSizeMismatch { neighbor: nbr, expected: buf_bytes.len(), got: raw.len() });
        }
        buf_bytes.copy_from_slice(&raw);
        let parts: &[D::Part] = &buffer;
        let links_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        if parts.len() != links_vec.len() {
            return Err(MeshSieveError::PartCountMismatch { neighbor: nbr, expected: links_vec.len(), got: parts.len() });
        }
        for ((_, dst), part) in links_vec.iter().zip(parts) {
            let mut_slice = section.try_restrict_mut(*dst).map_err(|e| MeshSieveError::SectionAccess { point: *dst, source: Box::new(e) })?;
            D::fuse(&mut mut_slice[0], *part);
        }
    }
    // Wait for all sends to complete (important for MPI correctness)
    for h in pending_sends {
        let _ = h.wait();
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    struct DummyValue(pub i32);

    // Dummy Delta implementation
    struct DummyDelta;
    impl crate::overlap::delta::Delta<DummyValue> for DummyDelta {
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
        let comm0 = crate::algs::communicator::RayonComm::new(0);
        let comm1 = crate::algs::communicator::RayonComm::new(1);

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

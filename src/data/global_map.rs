//! Global DOF numbering for a local [`Section`].
//!
//! `LocalToGlobalMap` computes a deterministic, ownership-aware global index
//! for each local point/DOF pair. Owned points are numbered first by rank
//! (ascending) and then by point ID within each rank. Ghost points receive
//! their global offsets from their owning rank via the overlap graph.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::algs::communicator::{CommTag, Communicator, SectionCommTags, Wait};
use crate::algs::completion::neighbour_links::neighbour_links_with_ownership;
use crate::algs::wire::{WireCount, cast_slice, cast_slice_mut};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;

/// Mapping from local point/DOF pairs to unique global indices.
#[derive(Clone, Debug, Default)]
pub struct LocalToGlobalMap {
    offsets: Vec<Option<u64>>,
    dof_lengths: Vec<Option<usize>>,
    total_dofs: u64,
}

impl LocalToGlobalMap {
    /// Build a global numbering map using explicit communication tags and ownership data.
    pub fn from_section_with_tags_and_ownership<V, S, C>(
        section: &Section<V, S>,
        overlap: &Overlap,
        ownership: &PointOwnership,
        comm: &C,
        my_rank: usize,
        tags: SectionCommTags,
    ) -> Result<Self, MeshSieveError>
    where
        S: Storage<V>,
        C: Communicator + Sync,
    {
        #[cfg(any(
            debug_assertions,
            feature = "strict-invariants",
            feature = "check-invariants"
        ))]
        overlap.validate_invariants()?;

        let max_id = section.atlas().points().map(|p| p.get()).max().unwrap_or(0) as usize;
        let mut map = LocalToGlobalMap {
            offsets: vec![None; max_id],
            dof_lengths: vec![None; max_id],
            total_dofs: 0,
        };
        map.populate_dof_lengths(section)?;
        map.assign_owned_offsets(section, ownership, comm, my_rank)?;

        let mut has_ghosts = false;
        for p in section.atlas().points() {
            let owner = ownership.owner_or_err(p)?;
            if owner != my_rank {
                has_ghosts = true;
            }
        }

        let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
        nb.remove(&my_rank);
        if nb.is_empty() {
            if has_ghosts {
                return Err(MeshSieveError::MissingOverlap {
                    source: format!("rank {my_rank} has ghost points but no overlap").into(),
                });
            }
            return Ok(map);
        }

        let mut links = neighbour_links_with_ownership(section, overlap, ownership, my_rank)?;
        for link_vec in links.values_mut() {
            link_vec.sort_unstable_by_key(|(send_loc, _)| send_loc.get());
        }

        let mut all_neighbors: HashSet<usize> = overlap.neighbor_ranks().collect();
        all_neighbors.extend(links.keys().copied());
        all_neighbors.remove(&my_rank);

        let send_counts = build_send_counts(&links, section, ownership, my_rank)?;
        let recv_counts =
            exchange_sizes_with_counts(&send_counts, comm, tags.sizes, &all_neighbors)?;
        exchange_offsets(
            &links,
            &recv_counts,
            comm,
            tags.data,
            section,
            ownership,
            &mut map,
            &all_neighbors,
        )?;

        map.ensure_complete(section)?;
        Ok(map)
    }

    /// Build a global numbering map using ownership metadata and a legacy default tag (0xBEEF).
    pub fn from_section_with_ownership<V, S, C>(
        section: &Section<V, S>,
        overlap: &Overlap,
        ownership: &PointOwnership,
        comm: &C,
        my_rank: usize,
    ) -> Result<Self, MeshSieveError>
    where
        S: Storage<V>,
        C: Communicator + Sync,
    {
        let tags = SectionCommTags::from_base(CommTag::new(0xBEEF));
        Self::from_section_with_tags_and_ownership(section, overlap, ownership, comm, my_rank, tags)
    }

    /// Return the global offset (start index) for a point.
    pub fn global_offset(&self, point: PointId) -> Result<u64, MeshSieveError> {
        let idx = point_index(point)?;
        self.offsets
            .get(idx)
            .and_then(|val| *val)
            .ok_or(MeshSieveError::PointNotInAtlas(point))
    }

    /// Return the global index for a local point/DOF pair.
    pub fn global_index(&self, point: PointId, dof: usize) -> Result<u64, MeshSieveError> {
        let idx = point_index(point)?;
        let len = self
            .dof_lengths
            .get(idx)
            .and_then(|val| *val)
            .ok_or(MeshSieveError::PointNotInAtlas(point))?;
        if dof >= len {
            return Err(MeshSieveError::ConstraintIndexOutOfBounds {
                point,
                index: dof,
                len,
            });
        }
        Ok(self.global_offset(point)? + dof as u64)
    }

    /// Return the global DOF range `[start, end)` for a point.
    pub fn global_range(&self, point: PointId) -> Result<std::ops::Range<u64>, MeshSieveError> {
        let idx = point_index(point)?;
        let len = self
            .dof_lengths
            .get(idx)
            .and_then(|val| *val)
            .ok_or(MeshSieveError::PointNotInAtlas(point))? as u64;
        let start = self.global_offset(point)?;
        Ok(start..start + len)
    }

    /// Total number of globally owned DOFs across all ranks.
    pub fn total_dofs(&self) -> u64 {
        self.total_dofs
    }

    fn populate_dof_lengths<V, S>(&mut self, section: &Section<V, S>) -> Result<(), MeshSieveError>
    where
        S: Storage<V>,
    {
        for p in section.atlas().points() {
            let (.., len) = section
                .atlas()
                .get(p)
                .ok_or(MeshSieveError::PointNotInAtlas(p))?;
            let idx = point_index(p)?;
            if idx >= self.dof_lengths.len() {
                self.dof_lengths.resize(idx + 1, None);
                self.offsets.resize(idx + 1, None);
            }
            self.dof_lengths[idx] = Some(len);
        }
        Ok(())
    }

    fn assign_owned_offsets<V, S, C>(
        &mut self,
        section: &Section<V, S>,
        ownership: &PointOwnership,
        comm: &C,
        my_rank: usize,
    ) -> Result<(), MeshSieveError>
    where
        S: Storage<V>,
        C: Communicator + Sync,
    {
        let mut owned_points: Vec<PointId> = section
            .atlas()
            .points()
            .filter(|&p| ownership.is_owned_by(p, my_rank))
            .collect();
        owned_points.sort_unstable();

        let mut local_total = 0u64;
        for p in &owned_points {
            let idx = point_index(*p)?;
            let len = self
                .dof_lengths
                .get(idx)
                .and_then(|val| *val)
                .ok_or(MeshSieveError::PointNotInAtlas(*p))? as u64;
            self.offsets[idx] = Some(local_total);
            local_total = local_total.saturating_add(len);
        }

        let n_ranks = comm.size().max(1);
        let mut recvbuf = vec![0u8; n_ranks * std::mem::size_of::<u64>()];
        comm.allgather(&local_total.to_le_bytes(), &mut recvbuf);

        let mut totals = vec![0u64; n_ranks];
        for (idx, chunk) in recvbuf.chunks_exact(8).enumerate() {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(chunk);
            totals[idx] = u64::from_le_bytes(raw);
        }
        let base: u64 = totals.iter().take(my_rank).copied().sum();
        self.total_dofs = totals.iter().copied().sum();

        for p in &owned_points {
            let idx = point_index(*p)?;
            if let Some(offset) = self.offsets.get_mut(idx).and_then(|val| val.as_mut()) {
                *offset = offset.saturating_add(base);
            }
        }

        Ok(())
    }

    fn ensure_complete<V, S>(&self, section: &Section<V, S>) -> Result<(), MeshSieveError>
    where
        S: Storage<V>,
    {
        for p in section.atlas().points() {
            let idx = point_index(p)?;
            if self.offsets.get(idx).and_then(|val| *val).is_none() {
                return Err(MeshSieveError::MissingOverlap {
                    source: format!("missing global offset for point {p:?}").into(),
                });
            }
        }
        Ok(())
    }
}

fn point_index(point: PointId) -> Result<usize, MeshSieveError> {
    point
        .get()
        .checked_sub(1)
        .ok_or(MeshSieveError::InvalidPointId)
        .map(|idx| idx as usize)
}

fn build_send_counts<V, S>(
    links: &HashMap<usize, Vec<(PointId, PointId)>>,
    section: &Section<V, S>,
    ownership: &PointOwnership,
    my_rank: usize,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    S: Storage<V>,
{
    let mut counts = HashMap::with_capacity(links.len());
    for (nbr, link_vec) in links {
        let mut count = 0usize;
        for &(send_loc, _) in link_vec {
            if section.atlas().contains(send_loc) && ownership.owner_or_err(send_loc)? == my_rank {
                count += 1;
            }
        }
        counts.insert(*nbr, u32::try_from(count).unwrap_or(u32::MAX));
    }
    Ok(counts)
}

fn exchange_sizes_with_counts<C>(
    send_counts: &HashMap<usize, u32>,
    comm: &C,
    tag: CommTag,
    all_neighbors: &HashSet<usize>,
) -> Result<HashMap<usize, u32>, MeshSieveError>
where
    C: Communicator + Sync,
{
    let mut recv_size: HashMap<usize, (C::RecvHandle, WireCount)> = HashMap::new();
    for &nbr in all_neighbors {
        let mut cnt = WireCount::new(0);
        let h = comm.irecv_result(
            nbr,
            tag.as_u16(),
            cast_slice_mut(std::slice::from_mut(&mut cnt)),
        )?;
        recv_size.insert(nbr, (h, cnt));
    }

    let mut pending_sends = Vec::with_capacity(all_neighbors.len());
    let mut send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let count = send_counts.get(&nbr).copied().unwrap_or(0);
        let wire = WireCount::new(count as usize);
        pending_sends.push(comm.isend_result(
            nbr,
            tag.as_u16(),
            cast_slice(std::slice::from_ref(&wire)),
        )?);
        send_bufs.push(wire);
    }

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

    for send in pending_sends {
        let _ = send.wait();
    }
    drop(send_bufs);

    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(sizes_in)
    }
}

fn exchange_offsets<V, S, C>(
    links: &HashMap<usize, Vec<(PointId, PointId)>>,
    recv_counts: &HashMap<usize, u32>,
    comm: &C,
    tag: CommTag,
    section: &Section<V, S>,
    ownership: &PointOwnership,
    map: &mut LocalToGlobalMap,
    all_neighbors: &HashSet<usize>,
) -> Result<(), MeshSieveError>
where
    S: Storage<V>,
    C: Communicator + Sync,
{
    let mut recv_data: HashMap<usize, (C::RecvHandle, Vec<u64>)> = HashMap::new();
    for &nbr in all_neighbors {
        let n_items = recv_counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buffer = vec![0u64; n_items];
        let h = comm.irecv_result(nbr, tag.as_u16(), cast_slice_mut(&mut buffer))?;
        recv_data.insert(nbr, (h, buffer));
    }

    let mut pending_sends = Vec::with_capacity(all_neighbors.len());
    let mut send_bufs = Vec::with_capacity(all_neighbors.len());
    for &nbr in all_neighbors {
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut scratch = Vec::new();
        for &(send_loc, _) in link_vec {
            if section.atlas().contains(send_loc)
                && ownership.owner_or_err(send_loc)? == comm.rank()
            {
                let idx = point_index(send_loc)?;
                if let Some(offset) = map.offsets.get(idx).and_then(|val| *val) {
                    scratch.push(offset);
                }
            }
        }
        let bytes = cast_slice(&scratch);
        pending_sends.push(comm.isend_result(nbr, tag.as_u16(), bytes)?);
        send_bufs.push(scratch);
    }

    for (nbr, (h, mut buffer)) in recv_data {
        let raw = h.wait().ok_or_else(|| MeshSieveError::CommError {
            neighbor: nbr,
            source: "No data received (wait returned None)".into(),
        })?;
        if raw.len() != buffer.len() * std::mem::size_of::<u64>() {
            return Err(MeshSieveError::BufferSizeMismatch {
                neighbor: nbr,
                expected: buffer.len() * std::mem::size_of::<u64>(),
                got: raw.len(),
            });
        }
        cast_slice_mut(&mut buffer).copy_from_slice(&raw);
        let parts: &[u64] = &buffer;
        let link_vec = links.get(&nbr).map_or(&[][..], |v| &v[..]);
        let mut recv_pairs = Vec::new();
        for &(send_loc, recv_loc) in link_vec {
            if !section.atlas().contains(recv_loc) {
                continue;
            }
            if ownership.owner_or_err(recv_loc)? == nbr {
                recv_pairs.push((send_loc, recv_loc));
            }
        }
        recv_pairs.sort_unstable_by_key(|(send_loc, _)| send_loc.get());
        if parts.len() != recv_pairs.len() {
            return Err(MeshSieveError::PartCountMismatch {
                neighbor: nbr,
                expected: recv_pairs.len(),
                got: parts.len(),
            });
        }
        for ((_, recv_loc), offset) in recv_pairs.iter().zip(parts) {
            let idx = point_index(*recv_loc)?;
            if idx >= map.offsets.len() {
                map.offsets.resize(idx + 1, None);
            }
            map.offsets[idx] = Some(*offset);
        }
    }

    for send in pending_sends {
        let _ = send.wait();
    }
    drop(send_bufs);

    Ok(())
}

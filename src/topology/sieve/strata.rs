//! Strata computation utilities for sieves.
//!
//! This module provides [`StrataCache`], a structure for storing precomputed height, depth, and strata
//! information for points in a sieve, as well as [`compute_strata`], a function to compute these values
//! for any sieve instance.
//!
//! # Errors
//! * [`MeshSieveError::MissingPointInCone`]: an arrow references a point **not present in
//!   `base_points ∪ cap_points`**. The error aggregates all such points (examples provided) and is
//!   raised before any topological pass.
//! * [`MeshSieveError::CycleDetected`]: the topology contains a cycle.

use crate::algs::communicator::{CommTag, Communicator, SectionCommTags, Wait};
use crate::algs::wire::{WireCount, cast_slice, cast_slice_mut};
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{Overlap, local};
use crate::topology::bounds::PointLike;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

/// Precomputed stratum information for a sieve.
///
/// Stores the height, depth, and strata (points at each height) for efficient queries,
/// as well as the diameter (maximum height) of the structure.
#[derive(Clone, Debug)]
pub struct StrataCache<P> {
    /// Map from point to its height (distance from any zero-in-degree source).
    pub height: HashMap<P, u32>,
    /// Map from point to its depth (distance down to any zero-out-degree sink).
    pub depth: HashMap<P, u32>,
    /// Vectors of points at each height: `strata[h] = points at height h`.
    pub strata: Vec<Vec<P>>,
    /// Maximum height (diameter) of the sieve.
    pub diameter: u32,

    /// Deterministic global ordering of points (height-major, then point order).
    pub chart_points: Vec<P>, // index -> point
    /// Reverse lookup from point to chart index.
    pub chart_index: HashMap<P, usize>, // point -> index
}

impl<P: PointLike> StrataCache<P> {
    /// Create a new, empty `StrataCache`.
    pub fn new() -> Self {
        Self {
            height: HashMap::new(),
            depth: HashMap::new(),
            strata: Vec::new(),
            diameter: 0,
            chart_points: Vec::new(),
            chart_index: HashMap::new(),
        }
    }

    /// Index of `p` in the chart, if present.
    #[inline]
    pub fn index_of(&self, p: P) -> Option<usize> {
        self.chart_index.get(&p).copied()
    }

    /// Point at chart index `i`.
    #[inline]
    pub fn point_at(&self, i: usize) -> P {
        self.chart_points[i]
    }

    /// Total number of points in the chart.
    #[inline]
    pub fn len(&self) -> usize {
        self.chart_points.len()
    }
}

impl<P: PointLike> Default for StrataCache<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Stratum selector for numbering utilities.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StratumAxis {
    /// Height strata (cells first).
    Height,
    /// Depth strata (vertices first).
    Depth,
}

/// Contiguous point numbering within each stratum.
#[derive(Clone, Debug)]
pub struct StratumNumbering<P> {
    /// Axis used to build strata.
    pub axis: StratumAxis,
    /// Stratum index for each point.
    pub stratum_index: HashMap<P, u32>,
    /// Contiguous ID within each stratum.
    pub id_in_stratum: HashMap<P, u64>,
    /// Total number of points in each stratum.
    pub stratum_sizes: Vec<u64>,
}

impl<P: PointLike> StratumNumbering<P> {
    /// Return the stratum for a point.
    pub fn stratum_of(&self, p: P) -> Option<u32> {
        self.stratum_index.get(&p).copied()
    }

    /// Return the contiguous ID within the point's stratum.
    pub fn id_of(&self, p: P) -> Option<u64> {
        self.id_in_stratum.get(&p).copied()
    }

    /// Return the number of points in stratum `s`.
    pub fn stratum_size(&self, s: u32) -> Option<u64> {
        self.stratum_sizes.get(s as usize).copied()
    }
}

/// Compute strata information on-the-fly (no cache).
///
/// Returns a [`StrataCache`] containing height, depth, strata, and diameter information for all points.
///
/// ## Complexity
/// - Time: **O(|V| + |E|)** (Kahn topological sort + forward/backward passes)
/// - Space: **O(|V| + |E|)** for intermediate degree maps and result vectors.
///
/// # Errors
/// * [`MeshSieveError::MissingPointInCone`]: any arrow `(src → dst)` references a point not present
///   in `base_points ∪ cap_points`. The error aggregates all such points and is raised before any
///   topological pass.
/// * [`MeshSieveError::CycleDetected`]: the topology contains a cycle.
pub fn compute_strata<S>(s: &S) -> Result<StrataCache<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: PointLike,
{
    use std::collections::{HashMap, HashSet};

    // 0) Authoritative vertex set: V = base ∪ cap
    let mut in_deg: HashMap<S::Point, u32> = HashMap::new();
    for p in s.base_points() {
        in_deg.entry(p).or_insert(0);
    }
    for p in s.cap_points() {
        in_deg.entry(p).or_insert(0);
    }

    // 1) Validate edges against V and accumulate in-degrees
    //    We check both directions to catch backends that under-report one role.
    let mut missing: HashSet<S::Point> = HashSet::new();

    // 1a) cone: sources must be in V by construction; ensure all dst ∈ V
    let sources: Vec<_> = in_deg.keys().copied().collect();
    for p in sources {
        for (q, _) in s.cone(p) {
            if let Some(d) = in_deg.get_mut(&q) {
                *d += 1;
            } else {
                missing.insert(q);
            }
        }
    }

    // 1b) support: destinations must be in V; ensure all src ∈ V
    let caps: Vec<_> = in_deg.keys().copied().collect();
    for q in caps {
        for (src, _) in s.support(q) {
            if !in_deg.contains_key(&src) {
                missing.insert(src);
            }
        }
    }

    if !missing.is_empty() {
        let mut examples: Vec<_> = missing.iter().copied().collect();
        examples.sort_unstable();
        examples.truncate(8);
        return Err(MeshSieveError::MissingPointInCone(format!(
            "Topology references points not declared in base_points∪cap_points; examples: {examples:?} ({} missing total)",
            missing.len()
        )));
    }

    // 2) Kahn’s topological sort on V using the validated in-degrees
    let mut stack: Vec<_> = in_deg
        .iter()
        .filter_map(|(&p, &d)| (d == 0).then_some(p))
        .collect();

    let mut topo = Vec::with_capacity(in_deg.len());
    while let Some(p) = stack.pop() {
        topo.push(p);
        for (q, _) in s.cone(p) {
            // safe: all q ∈ V by validation above
            let d = in_deg.get_mut(&q).unwrap();
            *d -= 1;
            if *d == 0 {
                stack.push(q);
            }
        }
    }

    if topo.len() != in_deg.len() {
        return Err(MeshSieveError::CycleDetected);
    }

    // 3) Heights (same as before)
    let mut height = HashMap::new();
    for &p in &topo {
        let h = s
            .support(p)
            .map(|(pred, _)| height.get(&pred).copied().unwrap_or(0))
            .max()
            .map_or(0, |m| m + 1);
        height.insert(p, h);
    }
    let max_h = *height.values().max().unwrap_or(&0);
    let mut strata = vec![Vec::new(); (max_h + 1) as usize];
    for (&p, &h) in &height {
        strata[h as usize].push(p);
    }

    // 4) Depths (same as before)
    let mut depth = HashMap::new();
    for &p in topo.iter().rev() {
        let d = s
            .cone(p)
            .map(|(succ, _)| depth.get(&succ).copied().unwrap_or(0))
            .max()
            .map_or(0, |m| m + 1);
        depth.insert(p, d);
    }

    // 5) Deterministic chart (same as before)
    for lev in &mut strata {
        lev.sort_unstable();
    }
    let mut chart_points = Vec::with_capacity(height.len());
    for lev in &strata {
        chart_points.extend(lev.iter().copied());
    }
    let mut chart_index = HashMap::with_capacity(chart_points.len());
    for (i, p) in chart_points.iter().copied().enumerate() {
        chart_index.insert(p, i);
    }

    Ok(StrataCache {
        height,
        depth,
        strata,
        diameter: max_h,
        chart_points,
        chart_index,
    })
}

/// Build contiguous stratum numbering for a local sieve.
pub fn stratum_numbering_local<S>(
    sieve: &S,
    axis: StratumAxis,
) -> Result<StratumNumbering<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: PointLike,
{
    let cache = compute_strata(sieve)?;
    let (strata, stratum_index) = build_strata(&cache, axis);
    Ok(number_from_strata(axis, strata, stratum_index))
}

/// Build contiguous stratum numbering across ranks using ownership/overlap metadata.
pub fn stratum_numbering_global<S, C>(
    sieve: &S,
    axis: StratumAxis,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
) -> Result<StratumNumbering<PointId>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    C: Communicator + Sync,
{
    let tags = SectionCommTags::from_base(CommTag::new(0xC0DE));
    stratum_numbering_global_with_tags(sieve, axis, overlap, ownership, comm, my_rank, tags)
}

/// Build contiguous stratum numbering across ranks with explicit communication tags.
pub fn stratum_numbering_global_with_tags<S, C>(
    sieve: &S,
    axis: StratumAxis,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
    tags: SectionCommTags,
) -> Result<StratumNumbering<PointId>, MeshSieveError>
where
    S: Sieve<Point = PointId>,
    C: Communicator + Sync,
{
    let cache = compute_strata(sieve)?;
    let (strata, stratum_index) = build_strata(&cache, axis);

    let local_n = strata.len() as u64;
    let mut n_recvbuf = vec![0u8; comm.size().max(1) * std::mem::size_of::<u64>()];
    comm.allgather(&local_n.to_le_bytes(), &mut n_recvbuf);
    let mut n_vals = Vec::with_capacity(comm.size().max(1));
    for chunk in n_recvbuf.chunks_exact(8) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        n_vals.push(u64::from_le_bytes(raw));
    }
    let global_n = n_vals.iter().copied().max().unwrap_or(local_n) as usize;

    let mut owned_by_stratum = vec![Vec::new(); strata.len()];
    for (s, lev) in strata.iter().enumerate() {
        for &p in lev {
            if ownership.is_owned_by(p, my_rank) {
                owned_by_stratum[s].push(p);
            }
        }
    }

    let mut local_counts = vec![0u64; global_n];
    for (idx, owned) in owned_by_stratum.iter().enumerate() {
        local_counts[idx] = owned.len() as u64;
    }

    let mut counts_buf = vec![0u8; global_n * std::mem::size_of::<u64>()];
    for (slot, value) in counts_buf.chunks_exact_mut(8).zip(local_counts.iter()) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    let mut counts_recv = vec![0u8; comm.size().max(1) * counts_buf.len()];
    if !counts_buf.is_empty() {
        comm.allgather(&counts_buf, &mut counts_recv);
    }

    let mut stratum_sizes = vec![0u64; global_n];
    let mut offsets = vec![0u64; global_n];
    if !counts_recv.is_empty() {
        for (rank, chunk) in counts_recv.chunks_exact(counts_buf.len()).enumerate() {
            for (idx, slot) in chunk.chunks_exact(8).enumerate() {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(slot);
                let val = u64::from_le_bytes(raw);
                stratum_sizes[idx] = stratum_sizes[idx].saturating_add(val);
                if rank < my_rank {
                    offsets[idx] = offsets[idx].saturating_add(val);
                }
            }
        }
    }

    let mut id_in_stratum = HashMap::new();
    for (s, owned) in owned_by_stratum.iter().enumerate() {
        for (idx, &p) in owned.iter().enumerate() {
            id_in_stratum.insert(p, offsets[s].saturating_add(idx as u64));
        }
    }

    let mut has_ghosts = false;
    for &p in stratum_index.keys() {
        if ownership.owner_or_err(p)? != my_rank {
            has_ghosts = true;
            break;
        }
    }

    let mut neighbors: std::collections::BTreeSet<usize> = overlap.neighbor_ranks().collect();
    neighbors.remove(&my_rank);
    if neighbors.is_empty() {
        if has_ghosts {
            return Err(MeshSieveError::MissingOverlap {
                source: format!("rank {my_rank} has ghost points but no overlap").into(),
            });
        }
        return Ok(StratumNumbering {
            axis,
            stratum_index,
            id_in_stratum,
            stratum_sizes,
        });
    }

    let mut links = overlap_links(stratum_index.keys().copied(), overlap, ownership, my_rank)?;
    for link_vec in links.values_mut() {
        link_vec.sort_unstable_by_key(|(send_loc, _)| send_loc.get());
    }

    let send_counts = build_send_counts(&links, ownership, my_rank)?;
    let all_neighbors: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
    let recv_counts = exchange_sizes_with_counts(&send_counts, comm, tags.sizes, &all_neighbors)?;
    exchange_ids(
        &links,
        &recv_counts,
        comm,
        tags.data,
        ownership,
        &mut id_in_stratum,
        &all_neighbors,
    )?;

    for &p in stratum_index.keys() {
        if id_in_stratum.get(&p).is_none() {
            return Err(MeshSieveError::MissingOverlap {
                source: format!("missing stratum id for point {p:?}").into(),
            });
        }
    }

    Ok(StratumNumbering {
        axis,
        stratum_index,
        id_in_stratum,
        stratum_sizes,
    })
}

fn build_strata<P: PointLike>(
    cache: &StrataCache<P>,
    axis: StratumAxis,
) -> (Vec<Vec<P>>, HashMap<P, u32>) {
    match axis {
        StratumAxis::Height => (cache.strata.clone(), cache.height.clone()),
        StratumAxis::Depth => {
            let max_depth = cache.depth.values().copied().max().unwrap_or(0) as usize;
            let mut strata = vec![Vec::new(); max_depth + 1];
            for (&p, &d) in &cache.depth {
                strata[d as usize].push(p);
            }
            for lev in &mut strata {
                lev.sort_unstable();
            }
            (strata, cache.depth.clone())
        }
    }
}

fn number_from_strata<P: PointLike>(
    axis: StratumAxis,
    strata: Vec<Vec<P>>,
    stratum_index: HashMap<P, u32>,
) -> StratumNumbering<P> {
    let mut id_in_stratum = HashMap::new();
    let mut stratum_sizes = Vec::with_capacity(strata.len());
    for lev in &strata {
        stratum_sizes.push(lev.len() as u64);
        for (idx, &p) in lev.iter().enumerate() {
            id_in_stratum.insert(p, idx as u64);
        }
    }
    StratumNumbering {
        axis,
        stratum_index,
        id_in_stratum,
        stratum_sizes,
    }
}

fn overlap_links<I>(
    points: I,
    overlap: &Overlap,
    ownership: &PointOwnership,
    my_rank: usize,
) -> Result<HashMap<usize, Vec<(PointId, PointId)>>, MeshSieveError>
where
    I: IntoIterator<Item = PointId>,
{
    let mut out: HashMap<usize, Vec<(PointId, PointId)>> = HashMap::new();
    for p in points {
        let owner = ownership.owner_or_err(p)?;
        if owner == my_rank {
            for (_dst, rem) in overlap.cone(local(p)) {
                if rem.rank != my_rank {
                    let remote_pt = rem
                        .remote_point
                        .ok_or(MeshSieveError::OverlapLinkMissing(p, rem.rank))?;
                    out.entry(rem.rank).or_default().push((p, remote_pt));
                }
            }
        } else {
            let mut remote_point = None;
            for (_dst, rem) in overlap.cone(local(p)) {
                if rem.rank == owner {
                    remote_point = rem.remote_point;
                    break;
                }
            }
            let remote_pt = remote_point.ok_or(MeshSieveError::OverlapLinkMissing(p, owner))?;
            out.entry(owner).or_default().push((remote_pt, p));
        }
    }
    if out.is_empty() {
        return Err(MeshSieveError::MissingOverlap {
            source: format!("rank {my_rank} has no neighbour links").into(),
        });
    }
    Ok(out)
}

fn build_send_counts(
    links: &HashMap<usize, Vec<(PointId, PointId)>>,
    ownership: &PointOwnership,
    my_rank: usize,
) -> Result<HashMap<usize, u32>, MeshSieveError> {
    let mut counts = HashMap::with_capacity(links.len());
    for (nbr, link_vec) in links {
        let mut count = 0usize;
        for &(send_loc, _) in link_vec {
            if ownership
                .owner(send_loc)
                .is_some_and(|owner| owner == my_rank)
            {
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
    all_neighbors: &std::collections::HashSet<usize>,
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

fn exchange_ids<C>(
    links: &HashMap<usize, Vec<(PointId, PointId)>>,
    recv_counts: &HashMap<usize, u32>,
    comm: &C,
    tag: CommTag,
    ownership: &PointOwnership,
    ids: &mut HashMap<PointId, u64>,
    all_neighbors: &std::collections::HashSet<usize>,
) -> Result<(), MeshSieveError>
where
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
            if ownership
                .owner(send_loc)
                .is_some_and(|owner| owner == comm.rank())
            {
                if let Some(id) = ids.get(&send_loc).copied() {
                    scratch.push(id);
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
            if ownership.owner(recv_loc).is_some_and(|owner| owner == nbr) {
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
        for ((_, recv_loc), id) in recv_pairs.iter().zip(parts) {
            ids.insert(*recv_loc, *id);
        }
    }

    for send in pending_sends {
        let _ = send.wait();
    }
    drop(send_bufs);

    Ok(())
}

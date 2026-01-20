// src/algs/distribute.rs

use crate::algs::communicator::Communicator;
use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::mixed_section::{MixedSectionStore, TaggedSection};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::{Overlap, OvlId};
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use std::collections::{BTreeMap, BTreeSet};

/// Distribute a global mesh across ranks, returning the local submesh and overlap graph.
///
/// Phase A of distribution: extract the local topology and create structural overlap
/// links (`Local(p) -> Part(r)`), leaving `remote_point` unresolved for later phases.
///
/// # Arguments
/// - `mesh`: the full global mesh (arrows of type `Payload = ()`)
/// - `parts`: mapping each `PointId` (1-based) to an owning rank
/// - `comm`: communicator providing `rank()` and `size()`
///
/// # Returns
/// `(local_mesh, overlap)` where:
/// - `local_mesh`: only arrows whose endpoints are both owned by this rank
/// - `overlap`: bipartite `Local(p) -> Part(r)` links for every foreign point
///
/// ## Phases
/// - **Phase A (here):** extract local topology and build structural overlap.
/// - **Phase B (later):** expand overlap via mesh closure rules.
/// - **Phase C (later):** resolve remote IDs via exchange/service.
/// - **Phase D (later):** complete section/stack data using the overlap.
///
/// # Example (serial)
/// ```rust
/// use mesh_sieve::algs::communicator::NoComm;
/// use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
/// use mesh_sieve::topology::point::PointId;
/// use mesh_sieve::algs::distribute_mesh;
/// let mut global = InMemorySieve::<PointId,()>::default();
/// global.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
/// global.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
/// let parts = vec![0, 1, 1];
/// let comm = NoComm;
/// let (_local, overlap) = distribute_mesh(&global, &parts, &comm).unwrap();
/// let ranks: Vec<_> = overlap.neighbor_ranks().collect();
/// assert!(ranks.contains(&1));
/// let links: Vec<_> = overlap.links_to(1).collect();
/// assert!(links
///     .iter()
///     .any(|(p, rp)| *p == PointId::new(3).unwrap() && rp.is_none()));
/// ```
/// # Example (MPI)
/// ```ignore
/// #![cfg(feature="mpi-support")]
/// use mesh_sieve::algs::communicator::MpiComm;
/// // ... same as above, but use MpiComm::new() and run with mpirun -n 2 ...
/// ```
pub fn distribute_mesh<M, C>(
    mesh: &M,
    parts: &[usize],
    comm: &C,
) -> Result<(InMemorySieve<PointId, ()>, Overlap), MeshSieveError>
where
    M: Sieve<Point = PointId, Payload = ()>,
    C: Communicator + Sync,
{
    let my_rank = comm.rank();

    // ---------- Pass 0: collect points and validate `parts` ----------
    let mut max_id = 0u64;
    let pts: Vec<PointId> = mesh
        .points()
        .inspect(|p| max_id = max_id.max(p.get()))
        .collect();
    if parts.len() < max_id as usize {
        return Err(MeshSieveError::PartitionIndexOutOfBounds(parts.len()));
    }

    // ---------- Pass 1: collect foreign points ----------
    let mut foreign_pts: Vec<(PointId, usize)> = Vec::new();
    foreign_pts.reserve(pts.len() / 2);

    for p in &pts {
        let owner = owner_of(parts, *p)?;
        if owner != my_rank {
            foreign_pts.push((*p, owner));
        }
    }

    // ---------- Build Overlap ----------
    let mut overlap = Overlap::default();
    overlap.add_links_structural_bulk(foreign_pts);

    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    overlap.validate_invariants()?;

    // ---------- Build local submesh ----------
    let mut local = InMemorySieve::<PointId, ()>::default();
    for src in &pts {
        if owner_of(parts, *src)? == my_rank {
            for (dst, _) in mesh.cone(*src) {
                if owner_of(parts, dst)? == my_rank {
                    local.add_arrow(*src, dst, ());
                }
            }
        }
    }

    Ok((local, overlap))
}

/// Only for single-process demos/tests: set `remote_point = Some(local_p)` for all links.
pub fn resolve_overlap_identity(overlap: &mut Overlap) {
    let mut to_resolve = Vec::new();
    for src in overlap.base_points() {
        if let OvlId::Local(p) = src {
            for (dst, rem) in overlap.cone(src) {
                if let OvlId::Part(r) = dst {
                    debug_assert_eq!(rem.rank, r);
                    to_resolve.push((p, r));
                }
            }
        }
    }
    for (p, r) in to_resolve {
        overlap
            .resolve_remote_point(p, r, p)
            .expect("resolve_remote_point failed");
    }
}

/// Configuration for overlap-aware distribution.
#[derive(Clone, Copy, Debug)]
pub struct DistributionConfig {
    /// Number of ghost layers to include around the partition boundary.
    pub overlap_depth: usize,
    /// Whether to copy global section-like data onto ghost points.
    pub synchronize_sections: bool,
}

impl Default for DistributionConfig {
    fn default() -> Self {
        Self {
            overlap_depth: 1,
            synchronize_sections: true,
        }
    }
}

/// Result of distributing a mesh plus associated data.
#[derive(Debug)]
pub struct DistributedMeshData<V, St, CtSt>
where
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Local topology with ghost layers included.
    pub sieve: InMemorySieve<PointId, ()>,
    /// Overlap graph with resolved remote IDs.
    pub overlap: Overlap,
    /// Point owners derived from the cell partition.
    pub point_owners: Vec<usize>,
    /// Ownership metadata for local points, including ghost status.
    pub ownership: PointOwnership,
    /// Cell partition assignment used for distribution.
    pub cell_parts: Vec<usize>,
    /// Optional coordinate section for local points.
    pub coordinates: Option<Coordinates<V, St>>,
    /// Named local sections for distributed points.
    pub sections: BTreeMap<String, Section<V, St>>,
    /// Tagged local sections with mixed scalar types.
    pub mixed_sections: MixedSectionStore,
    /// Point labels filtered to the local point set.
    pub labels: Option<LabelSet>,
    /// Optional cell-type section for local points.
    pub cell_types: Option<Section<CellType, CtSt>>,
}

/// Partition hook for distributing cell-based meshes.
pub trait CellPartitioner<M>
where
    M: Sieve<Point = PointId, Payload = ()>,
{
    /// Return a partition index for each input cell.
    fn partition_cells(
        &self,
        mesh: &M,
        cells: &[PointId],
        n_parts: usize,
    ) -> Result<Vec<usize>, MeshSieveError>;
}

/// Use a precomputed cell partition.
pub struct ProvidedPartition<'a> {
    pub parts: &'a [usize],
}

impl<M> CellPartitioner<M> for ProvidedPartition<'_>
where
    M: Sieve<Point = PointId, Payload = ()>,
{
    fn partition_cells(
        &self,
        _mesh: &M,
        cells: &[PointId],
        _n_parts: usize,
    ) -> Result<Vec<usize>, MeshSieveError> {
        if self.parts.len() != cells.len() {
            return Err(MeshSieveError::PartitionIndexOutOfBounds(self.parts.len()));
        }
        Ok(self.parts.to_vec())
    }
}

/// Use a custom partitioner callback.
pub struct CustomPartitioner<F>(pub F);

impl<M, F> CellPartitioner<M> for CustomPartitioner<F>
where
    M: Sieve<Point = PointId, Payload = ()>,
    F: Fn(&M, &[PointId], usize) -> Result<Vec<usize>, MeshSieveError>,
{
    fn partition_cells(
        &self,
        mesh: &M,
        cells: &[PointId],
        n_parts: usize,
    ) -> Result<Vec<usize>, MeshSieveError> {
        (self.0)(mesh, cells, n_parts)
    }
}

/// Partition cells using METIS (requires the `metis-support` feature).
#[cfg(feature = "metis-support")]
pub struct MetisPartitioner;

#[cfg(feature = "metis-support")]
impl<M> CellPartitioner<M> for MetisPartitioner
where
    M: Sieve<Point = PointId, Payload = ()>,
{
    fn partition_cells(
        &self,
        mesh: &M,
        cells: &[PointId],
        n_parts: usize,
    ) -> Result<Vec<usize>, MeshSieveError> {
        let dual = crate::algs::dual_graph::build_dual(mesh, cells.to_vec());
        let partition = dual.metis_partition(
            n_parts
                .try_into()
                .map_err(|_| MeshSieveError::PartitionIndexOutOfBounds(n_parts))?,
        );
        Ok(partition.part.into_iter().map(|p| p as usize).collect())
    }
}

/// High-level distribution with overlap expansion and data synchronization.
///
/// This orchestrates:
/// 1. Cell partitioning (METIS or custom hook),
/// 2. Point-owner assignment,
/// 3. Ghost-layer construction to the requested depth,
/// 4. Overlap resolution (remote IDs),
/// 5. Optional synchronization of sections/coordinates/cell types by copying
///    global data onto local ghost points.
///
/// Labels are filtered directly to local points so they survive redistribution
/// even when no synchronization is required.
///
/// # Assumptions
/// This helper assumes all ranks share a consistent global `PointId` space, so
/// remote IDs are resolved using the identity mapping.
pub fn distribute_with_overlap<M, V, St, CtSt, C, P>(
    mesh_data: &MeshData<M, V, St, CtSt>,
    cells: &[PointId],
    partitioner: &P,
    config: DistributionConfig,
    comm: &C,
) -> Result<DistributedMeshData<V, St, CtSt>, MeshSieveError>
where
    M: Sieve<Point = PointId, Payload = ()>,
    V: Clone + Default + Send + PartialEq + 'static,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
    C: Communicator + Sync,
    P: CellPartitioner<M>,
{
    let my_rank = comm.rank();
    let n_ranks = comm.size().max(1);
    let cell_parts = partitioner.partition_cells(&mesh_data.sieve, cells, n_ranks)?;
    if cell_parts.len() != cells.len() {
        return Err(MeshSieveError::PartitionIndexOutOfBounds(cell_parts.len()));
    }
    if cell_parts.iter().any(|&p| p >= n_ranks) {
        let bad = cell_parts.iter().copied().max().unwrap_or(0);
        return Err(MeshSieveError::PartitionIndexOutOfBounds(bad));
    }

    let points: Vec<PointId> = mesh_data.sieve.points().collect();
    let max_id = points.iter().map(|p| p.get()).max().unwrap_or(0) as usize;
    let point_owners = assign_point_owners(mesh_data, cells, &cell_parts, max_id)?;

    let adjacency = build_adjacency(&mesh_data.sieve, max_id);
    let local_sets = build_local_sets(&points, &point_owners, &adjacency, n_ranks, config);
    let point_ranks = build_point_ranks(&local_sets, max_id);
    let local_set = local_sets
        .get(my_rank)
        .ok_or(MeshSieveError::PartitionIndexOutOfBounds(my_rank))?;

    let overlap = build_overlap_for_rank(my_rank, local_set, &point_ranks)?;
    let ownership =
        PointOwnership::from_local_set(local_set.iter().copied(), &point_owners, my_rank)?;

    let local_sieve = build_local_sieve(mesh_data, local_set);
    let labels = mesh_data
        .labels
        .as_ref()
        .map(|l| l.filtered_to_points(local_set.iter().copied()));

    let coordinates = match &mesh_data.coordinates {
        Some(coords) => {
            let section =
                build_local_section(coords.section(), local_set, &point_owners, my_rank, config)?;
            Some(Coordinates::from_section(coords.dimension(), section)?)
        }
        None => None,
    };

    let mut sections = BTreeMap::new();
    for (name, section) in &mesh_data.sections {
        let local_section =
            build_local_section(section, local_set, &point_owners, my_rank, config)?;
        sections.insert(name.clone(), local_section);
    }

    let mut mixed_sections = MixedSectionStore::default();
    for (name, section) in mesh_data.mixed_sections.iter() {
        let local_section =
            build_local_tagged_section(section, local_set, &point_owners, my_rank, config)?;
        mixed_sections.insert_tagged(name.clone(), local_section);
    }

    let cell_types = match &mesh_data.cell_types {
        Some(section) => Some(build_local_section(
            section,
            local_set,
            &point_owners,
            my_rank,
            config,
        )?),
        None => None,
    };

    Ok(DistributedMeshData {
        sieve: local_sieve,
        overlap,
        point_owners,
        ownership,
        cell_parts,
        coordinates,
        sections,
        mixed_sections,
        labels,
        cell_types,
    })
}

fn assign_point_owners<M, V, St, CtSt>(
    mesh_data: &MeshData<M, V, St, CtSt>,
    cells: &[PointId],
    cell_parts: &[usize],
    max_id: usize,
) -> Result<Vec<usize>, MeshSieveError>
where
    M: Sieve<Point = PointId, Payload = ()>,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    let mut owners = vec![0usize; max_id];
    let mut seen = vec![false; max_id];
    for (cell, &part) in cells.iter().zip(cell_parts.iter()) {
        for p in mesh_data.sieve.closure_iter(std::iter::once(*cell)) {
            let idx = p
                .get()
                .checked_sub(1)
                .ok_or(MeshSieveError::PartitionIndexOutOfBounds(p.get() as usize))?
                as usize;
            if idx >= owners.len() {
                return Err(MeshSieveError::PartitionIndexOutOfBounds(idx));
            }
            if !seen[idx] || part < owners[idx] {
                owners[idx] = part;
                seen[idx] = true;
            }
        }
    }
    Ok(owners)
}

fn build_adjacency<M>(mesh: &M, max_id: usize) -> Vec<BTreeSet<PointId>>
where
    M: Sieve<Point = PointId, Payload = ()>,
{
    let mut adjacency = vec![BTreeSet::new(); max_id];
    for src in mesh.points() {
        let src_idx = (src.get() - 1) as usize;
        for (dst, _) in mesh.cone(src) {
            let dst_idx = (dst.get() - 1) as usize;
            if src_idx < max_id {
                adjacency[src_idx].insert(dst);
            }
            if dst_idx < max_id {
                adjacency[dst_idx].insert(src);
            }
        }
    }
    adjacency
}

fn build_local_sets(
    points: &[PointId],
    owners: &[usize],
    adjacency: &[BTreeSet<PointId>],
    n_ranks: usize,
    config: DistributionConfig,
) -> Vec<BTreeSet<PointId>> {
    let mut owned_sets = vec![BTreeSet::new(); n_ranks];
    for &p in points {
        let idx = (p.get() - 1) as usize;
        if let Some(&owner) = owners.get(idx) {
            if owner < n_ranks {
                owned_sets[owner].insert(p);
            }
        }
    }

    let mut boundary = vec![BTreeSet::new(); n_ranks];
    for &p in points {
        let p_idx = (p.get() - 1) as usize;
        let owner_p = owners.get(p_idx).copied().unwrap_or(0);
        if owner_p >= n_ranks {
            continue;
        }
        for q in adjacency.get(p_idx).into_iter().flat_map(|s| s.iter()) {
            let q_idx = (q.get() - 1) as usize;
            let owner_q = owners.get(q_idx).copied().unwrap_or(0);
            if owner_p != owner_q && owner_p < n_ranks {
                boundary[owner_p].insert(p);
            }
        }
    }

    let mut local_sets = Vec::with_capacity(n_ranks);
    for rank in 0..n_ranks {
        let mut local = owned_sets[rank].clone();
        if config.overlap_depth == 0 {
            local_sets.push(local);
            continue;
        }

        let mut ghost = BTreeSet::new();
        let mut frontier = BTreeSet::new();
        for p in &boundary[rank] {
            let p_idx = (p.get() - 1) as usize;
            for q in adjacency.get(p_idx).into_iter().flat_map(|s| s.iter()) {
                let q_idx = (q.get() - 1) as usize;
                if owners.get(q_idx).copied().unwrap_or(0) != rank && ghost.insert(*q) {
                    frontier.insert(*q);
                }
            }
        }

        for _ in 1..config.overlap_depth {
            if frontier.is_empty() {
                break;
            }
            let mut next_frontier = BTreeSet::new();
            for p in &frontier {
                let p_idx = (p.get() - 1) as usize;
                for q in adjacency.get(p_idx).into_iter().flat_map(|s| s.iter()) {
                    let q_idx = (q.get() - 1) as usize;
                    if owners.get(q_idx).copied().unwrap_or(0) != rank && ghost.insert(*q) {
                        next_frontier.insert(*q);
                    }
                }
            }
            frontier = next_frontier;
        }

        local.extend(ghost);
        local_sets.push(local);
    }

    local_sets
}

fn build_point_ranks(local_sets: &[BTreeSet<PointId>], max_id: usize) -> Vec<Vec<usize>> {
    let mut point_ranks = vec![Vec::new(); max_id];
    for (rank, points) in local_sets.iter().enumerate() {
        for &p in points {
            let idx = (p.get() - 1) as usize;
            if idx < max_id {
                point_ranks[idx].push(rank);
            }
        }
    }
    for ranks in &mut point_ranks {
        ranks.sort_unstable();
        ranks.dedup();
    }
    point_ranks
}

fn build_overlap_for_rank(
    rank: usize,
    local_set: &BTreeSet<PointId>,
    point_ranks: &[Vec<usize>],
) -> Result<Overlap, MeshSieveError> {
    let mut edges = Vec::new();
    for &p in local_set {
        let idx = (p.get() - 1) as usize;
        if let Some(ranks) = point_ranks.get(idx) {
            for &nbr in ranks {
                if nbr != rank {
                    edges.push((p, nbr));
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mut overlap = Overlap::default();
    overlap.try_add_links_structural_bulk(edges.iter().copied())?;
    overlap.resolve_remote_points(edges.into_iter().map(|(p, r)| (p, r, p)))?;
    Ok(overlap)
}

fn build_local_sieve<M, V, St, CtSt>(
    mesh_data: &MeshData<M, V, St, CtSt>,
    local_set: &BTreeSet<PointId>,
) -> InMemorySieve<PointId, ()>
where
    M: Sieve<Point = PointId, Payload = ()>,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    let mut local = InMemorySieve::<PointId, ()>::default();
    let points: Vec<PointId> = mesh_data.sieve.points().collect();
    for src in &points {
        if !local_set.contains(src) {
            continue;
        }
        for (dst, _) in mesh_data.sieve.cone(*src) {
            if local_set.contains(&dst) {
                local.add_arrow(*src, dst, ());
            }
        }
    }
    local
}

fn build_local_section<V, St>(
    section: &Section<V, St>,
    local_set: &BTreeSet<PointId>,
    owners: &[usize],
    my_rank: usize,
    config: DistributionConfig,
) -> Result<Section<V, St>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for p in local_set {
        if let Some((_off, len)) = section.atlas().get(*p) {
            atlas.try_insert(*p, len)?;
        }
    }
    let mut local_section = Section::<V, St>::new(atlas);
    let points: Vec<PointId> = local_section.atlas().points().collect();
    for p in points {
        let idx = (p.get() - 1) as usize;
        let is_owner = owners.get(idx).copied().unwrap_or(0) == my_rank;
        if config.synchronize_sections || is_owner {
            let data = section.try_restrict(p)?;
            local_section.try_set(p, data)?;
        }
    }
    Ok(local_section)
}

fn build_local_tagged_section(
    section: &TaggedSection,
    local_set: &BTreeSet<PointId>,
    owners: &[usize],
    my_rank: usize,
    config: DistributionConfig,
) -> Result<TaggedSection, MeshSieveError> {
    Ok(match section {
        TaggedSection::F64(sec) => TaggedSection::F64(build_local_section(
            sec, local_set, owners, my_rank, config,
        )?),
        TaggedSection::F32(sec) => TaggedSection::F32(build_local_section(
            sec, local_set, owners, my_rank, config,
        )?),
        TaggedSection::I32(sec) => TaggedSection::I32(build_local_section(
            sec, local_set, owners, my_rank, config,
        )?),
        TaggedSection::I64(sec) => TaggedSection::I64(build_local_section(
            sec, local_set, owners, my_rank, config,
        )?),
        TaggedSection::U32(sec) => TaggedSection::U32(build_local_section(
            sec, local_set, owners, my_rank, config,
        )?),
        TaggedSection::U64(sec) => TaggedSection::U64(build_local_section(
            sec, local_set, owners, my_rank, config,
        )?),
    })
}

#[inline]
fn owner_of(parts: &[usize], p: PointId) -> Result<usize, MeshSieveError> {
    let idx = p
        .get()
        .checked_sub(1)
        .ok_or(MeshSieveError::PartitionIndexOutOfBounds(p.get() as usize))? as usize;
    parts
        .get(idx)
        .copied()
        .ok_or(MeshSieveError::PartitionIndexOutOfBounds(idx))
}

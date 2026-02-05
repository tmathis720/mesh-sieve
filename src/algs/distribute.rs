// src/algs/distribute.rs

use crate::algs::communicator::{CommTag, Communicator, Wait};
use crate::algs::completion::{complete_section_with_ownership, complete_sieve};
use crate::algs::point_sf::PointSF;
use crate::algs::wire::{WirePointRepr, cast_slice, cast_slice_mut};
use crate::data::atlas::Atlas;
use crate::data::coordinates::{Coordinates, HighOrderCoordinates};
use crate::data::discretization::Discretization;
use crate::data::global_map::LocalToGlobalMap;
use crate::data::mixed_section::{MixedSectionStore, TaggedSection};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::overlap::delta::{CellTypeDelta, CopyDelta};
use crate::overlap::overlap::{Overlap, OvlId};
use crate::overlap::overlap::{ensure_closure_of_support, expand_one_layer_mesh};
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::ownership::PointOwnership;
use crate::topology::periodic::PointEquivalence;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};
use crate::topology::validation::debug_validate_overlap_ownership_topology;
use bytemuck::Zeroable;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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
    /// Optional discretization metadata keyed by regions.
    pub discretization: Option<Discretization>,
}

impl<V, St, CtSt> DistributedMeshData<V, St, CtSt>
where
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Build a global DOF map for a specific local section using ownership metadata.
    pub fn build_global_map_for_section<C>(
        &self,
        section: &Section<V, St>,
        comm: &C,
    ) -> Result<LocalToGlobalMap, MeshSieveError>
    where
        C: Communicator + Sync,
    {
        LocalToGlobalMap::from_section_with_ownership(
            section,
            &self.overlap,
            &self.ownership,
            comm,
            comm.rank(),
        )
    }

    /// Build global DOF maps for all named sections in this distributed mesh.
    pub fn build_global_section_maps<C>(
        &self,
        comm: &C,
    ) -> Result<BTreeMap<String, LocalToGlobalMap>, MeshSieveError>
    where
        C: Communicator + Sync,
    {
        let mut maps = BTreeMap::new();
        for (name, section) in &self.sections {
            let map = self.build_global_map_for_section(section, comm)?;
            maps.insert(name.clone(), map);
        }
        Ok(maps)
    }

    /// Update ghost values for all registered sections and labels.
    pub fn distribute_fields<C>(&mut self, comm: &C) -> Result<(), MeshSieveError>
    where
        C: Communicator + Sync,
        V: Clone + Default + Send + PartialEq + bytemuck::Pod + 'static,
    {
        let sf = PointSF::with_ownership(&self.overlap, &self.ownership, comm, comm.rank());
        sf.validate()?;

        if let Some(coords) = &mut self.coordinates {
            sf.complete_section(coords.section_mut())?;
            if let Some(high_order) = coords.high_order_mut() {
                sf.complete_section(high_order.section_mut())?;
            }
        }

        for section in self.sections.values_mut() {
            sf.complete_section(section)?;
        }

        for (_name, section) in self.mixed_sections.iter_mut() {
            complete_tagged_section_with_ownership(
                section,
                &self.overlap,
                &self.ownership,
                comm,
                comm.rank(),
            )?;
        }

        if let Some(cell_types) = &mut self.cell_types {
            complete_section_with_ownership::<CellType, _, CellTypeDelta, C>(
                cell_types,
                &self.overlap,
                &self.ownership,
                comm,
                comm.rank(),
            )?;
        }

        if let Some(labels) = &mut self.labels {
            complete_labels_with_ownership(
                labels,
                &self.overlap,
                &self.ownership,
                comm,
                comm.rank(),
            )?;
        }

        Ok(())
    }
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
    V: Clone + Default + Send + PartialEq + bytemuck::Pod + 'static,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
    C: Communicator + Sync,
    P: CellPartitioner<M>,
{
    distribute_with_overlap_periodic(mesh_data, cells, partitioner, config, comm, None)
}

/// High-level distribution with overlap expansion and periodic equivalence support.
///
/// When `periodic` is supplied, overlap construction and ghost expansion treat
/// periodic point equivalences as adjacency, and resolved overlap links map to
/// the periodic counterpart on neighboring ranks.
pub fn distribute_with_overlap_periodic<M, V, St, CtSt, C, P>(
    mesh_data: &MeshData<M, V, St, CtSt>,
    cells: &[PointId],
    partitioner: &P,
    config: DistributionConfig,
    comm: &C,
    periodic: Option<&PointEquivalence>,
) -> Result<DistributedMeshData<V, St, CtSt>, MeshSieveError>
where
    M: Sieve<Point = PointId, Payload = ()>,
    V: Clone + Default + Send + PartialEq + bytemuck::Pod + 'static,
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

    let periodic_classes = periodic
        .map(|eq| build_periodic_classes(&points, eq))
        .unwrap_or_default();
    let periodic_remote_map =
        build_periodic_remote_map(&periodic_classes, &point_owners, point_owners.len());

    let adjacency = build_adjacency(&mesh_data.sieve, max_id, &periodic_classes);
    let use_comm_completion = should_use_comm_completion(comm);

    let mut owned_set = BTreeSet::new();
    for &p in &points {
        let idx = (p.get() - 1) as usize;
        if point_owners.get(idx).copied().unwrap_or(0) == my_rank {
            owned_set.insert(p);
        }
    }

    let mut overlap = Overlap::default();
    let mut frontier: BTreeMap<usize, BTreeSet<PointId>> = BTreeMap::new();
    if config.overlap_depth > 0 {
        for &p in &owned_set {
            let p_idx = (p.get() - 1) as usize;
            for q in adjacency.get(p_idx).into_iter().flat_map(|s| s.iter()) {
                let q_idx = (q.get() - 1) as usize;
                let owner_q = point_owners.get(q_idx).copied().unwrap_or(0);
                if owner_q != my_rank {
                    overlap.try_add_link_structural_one(p, owner_q)?;
                    frontier.entry(owner_q).or_default().insert(p);
                }
            }
        }

        ensure_closure_of_support(&mut overlap, &mesh_data.sieve);

        for _layer in 0..config.overlap_depth {
            if frontier.is_empty() {
                break;
            }
            let mut next_frontier: BTreeMap<usize, BTreeSet<PointId>> = BTreeMap::new();
            for (&nbr, seeds) in &frontier {
                if seeds.is_empty() {
                    continue;
                }
                let before: BTreeSet<_> = overlap.links_to(nbr).map(|(p, _)| p).collect();
                expand_one_layer_mesh(&mut overlap, &mesh_data.sieve, seeds.iter().copied(), nbr);
                let after: BTreeSet<_> = overlap.links_to(nbr).map(|(p, _)| p).collect();
                let added: BTreeSet<_> = after.difference(&before).copied().collect();
                if !added.is_empty() {
                    next_frontier.insert(nbr, added);
                }
            }
            frontier = next_frontier;
        }
    }

    resolve_overlap_via_exchange(
        &mut overlap,
        comm,
        my_rank,
        Some(&periodic_remote_map),
        use_comm_completion,
    )?;
    ensure_periodic_remote_links(&mut overlap)?;

    let mut local_set = owned_set.clone();
    for nbr in overlap.neighbor_ranks() {
        for (p, remote) in overlap.links_to(nbr) {
            local_set.insert(p);
            if let Some(remote_point) = remote {
                local_set.insert(remote_point);
            }
        }
    }

    let local_sieve = if use_comm_completion {
        let mut local = build_local_sieve(mesh_data, &local_set);
        complete_sieve(&mut local, &overlap, comm, my_rank)?;
        local
    } else {
        build_local_sieve(mesh_data, &local_set)
    };

    let local_points: BTreeSet<_> = local_sieve.points().collect();
    let owned_points: BTreeSet<_> = local_points
        .iter()
        .copied()
        .filter(|p| {
            let idx = (p.get() - 1) as usize;
            point_owners.get(idx).copied().unwrap_or(0) == my_rank
        })
        .collect();

    let ownership =
        PointOwnership::from_local_set(local_points.iter().copied(), &point_owners, my_rank)?;

    debug_validate_overlap_ownership_topology(&local_sieve, &ownership, Some(&overlap), my_rank)?;

    let section_points = if config.synchronize_sections {
        &local_points
    } else {
        &owned_points
    };
    let copy_all_sections = config.synchronize_sections && !use_comm_completion;

    let labels = mesh_data
        .labels
        .as_ref()
        .map(|l| l.filtered_to_points(section_points.iter().copied()));

    let coordinates = match &mesh_data.coordinates {
        Some(coords) => {
            let mut section = if copy_all_sections {
                build_local_section_full(coords.section(), section_points)?
            } else {
                build_local_section_owned(coords.section(), section_points, &owned_points)?
            };
            if config.synchronize_sections && use_comm_completion {
                complete_section_with_ownership::<V, St, CopyDelta, C>(
                    &mut section,
                    &overlap,
                    &ownership,
                    comm,
                    my_rank,
                )?;
            }
            let mut out = Coordinates::from_section(
                coords.topological_dimension(),
                coords.embedding_dimension(),
                section,
            )?;
            if let Some(high_order) = coords.high_order() {
                let mut ho_section = if copy_all_sections {
                    build_local_section_full(high_order.section(), section_points)?
                } else {
                    build_local_section_owned(high_order.section(), section_points, &owned_points)?
                };
                if config.synchronize_sections && use_comm_completion {
                    complete_section_with_ownership::<V, St, CopyDelta, C>(
                        &mut ho_section,
                        &overlap,
                        &ownership,
                        comm,
                        my_rank,
                    )?;
                }
                let ho = HighOrderCoordinates::from_section(high_order.dimension(), ho_section)?;
                out.set_high_order(ho)?;
            }
            Some(out)
        }
        None => None,
    };

    let mut sections = BTreeMap::new();
    for (name, section) in &mesh_data.sections {
        let mut local_section = if copy_all_sections {
            build_local_section_full(section, section_points)?
        } else {
            build_local_section_owned(section, section_points, &owned_points)?
        };
        if config.synchronize_sections && use_comm_completion {
            complete_section_with_ownership::<V, St, CopyDelta, C>(
                &mut local_section,
                &overlap,
                &ownership,
                comm,
                my_rank,
            )?;
        }
        sections.insert(name.clone(), local_section);
    }

    let mut mixed_sections = MixedSectionStore::default();
    for (name, section) in mesh_data.mixed_sections.iter() {
        let mut local_section = if copy_all_sections {
            build_local_tagged_section(section, section_points, section_points)?
        } else {
            build_local_tagged_section(section, section_points, &owned_points)?
        };
        if config.synchronize_sections && use_comm_completion {
            complete_tagged_section_with_ownership(
                &mut local_section,
                &overlap,
                &ownership,
                comm,
                my_rank,
            )?;
        }
        mixed_sections.insert_tagged(name.clone(), local_section);
    }

    let cell_types = match &mesh_data.cell_types {
        Some(section) => {
            let local_section = if config.synchronize_sections {
                build_local_section_full(section, section_points)?
            } else {
                build_local_section_owned(section, section_points, &owned_points)?
            };
            Some(local_section)
        }
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
        discretization: mesh_data.discretization.clone(),
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

fn build_adjacency<M>(
    mesh: &M,
    max_id: usize,
    periodic_classes: &BTreeMap<PointId, Vec<PointId>>,
) -> Vec<BTreeSet<PointId>>
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
    apply_periodic_adjacency(&mut adjacency, periodic_classes, max_id);
    adjacency
}

fn build_periodic_classes(
    points: &[PointId],
    periodic: &PointEquivalence,
) -> BTreeMap<PointId, Vec<PointId>> {
    let mut eq = periodic.clone();
    eq.classes(points.iter().copied())
}

fn apply_periodic_adjacency(
    adjacency: &mut [BTreeSet<PointId>],
    periodic_classes: &BTreeMap<PointId, Vec<PointId>>,
    max_id: usize,
) {
    for class in periodic_classes.values() {
        if class.len() < 2 {
            continue;
        }
        for i in 0..class.len() {
            for j in (i + 1)..class.len() {
                let a = class[i];
                let b = class[j];
                let a_idx = (a.get().saturating_sub(1)) as usize;
                let b_idx = (b.get().saturating_sub(1)) as usize;
                if a_idx < max_id {
                    adjacency[a_idx].insert(b);
                }
                if b_idx < max_id {
                    adjacency[b_idx].insert(a);
                }
            }
        }
    }
}

fn build_periodic_remote_map(
    periodic_classes: &BTreeMap<PointId, Vec<PointId>>,
    owners: &[usize],
    max_id: usize,
) -> BTreeMap<(PointId, usize), PointId> {
    let mut remote_map = BTreeMap::new();
    for class in periodic_classes.values() {
        if class.len() < 2 {
            continue;
        }
        let mut rank_to_point: BTreeMap<usize, PointId> = BTreeMap::new();
        for &p in class {
            let idx = (p.get().saturating_sub(1)) as usize;
            if idx >= max_id {
                continue;
            }
            let owner = owners.get(idx).copied().unwrap_or(0);
            rank_to_point
                .entry(owner)
                .and_modify(|existing| {
                    if p < *existing {
                        *existing = p;
                    }
                })
                .or_insert(p);
        }
        for &p in class {
            let idx = (p.get().saturating_sub(1)) as usize;
            if idx >= max_id {
                continue;
            }
            let owner = owners.get(idx).copied().unwrap_or(0);
            for (&rank, &remote_point) in &rank_to_point {
                if rank != owner {
                    remote_map.insert((p, rank), remote_point);
                }
            }
        }
    }
    remote_map
}

fn ensure_periodic_remote_links(overlap: &mut Overlap) -> Result<(), MeshSieveError> {
    let mut extras: Vec<(PointId, usize)> = Vec::new();
    for nbr in overlap.neighbor_ranks() {
        for (local, remote) in overlap.links_to(nbr) {
            if let Some(remote_point) = remote {
                if remote_point != local {
                    extras.push((remote_point, nbr));
                }
            }
        }
    }
    extras.sort_unstable();
    extras.dedup();
    for (local, nbr) in extras {
        overlap.try_add_link_structural_one(local, nbr)?;
        overlap.resolve_remote_point(local, nbr, local)?;
    }
    Ok(())
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
    for &point in local_set {
        local.add_point(point);
    }
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

fn build_local_section_owned<V, St>(
    section: &Section<V, St>,
    local_points: &BTreeSet<PointId>,
    owned_points: &BTreeSet<PointId>,
) -> Result<Section<V, St>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for p in local_points {
        if let Some((_off, len)) = section.atlas().get(*p) {
            atlas.try_insert(*p, len)?;
        }
    }
    let mut local_section = Section::<V, St>::new(atlas);
    for p in owned_points {
        if local_section.atlas().get(*p).is_none() {
            continue;
        }
        let data = section.try_restrict(*p)?;
        local_section.try_set(*p, data)?;
    }
    Ok(local_section)
}

fn build_local_section_full<V, St>(
    section: &Section<V, St>,
    local_points: &BTreeSet<PointId>,
) -> Result<Section<V, St>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
{
    build_local_section_owned(section, local_points, local_points)
}

fn build_local_tagged_section(
    section: &TaggedSection,
    local_points: &BTreeSet<PointId>,
    owned_points: &BTreeSet<PointId>,
) -> Result<TaggedSection, MeshSieveError> {
    Ok(match section {
        TaggedSection::F64(sec) => {
            TaggedSection::F64(build_local_section_owned(sec, local_points, owned_points)?)
        }
        TaggedSection::F32(sec) => {
            TaggedSection::F32(build_local_section_owned(sec, local_points, owned_points)?)
        }
        TaggedSection::I32(sec) => {
            TaggedSection::I32(build_local_section_owned(sec, local_points, owned_points)?)
        }
        TaggedSection::I64(sec) => {
            TaggedSection::I64(build_local_section_owned(sec, local_points, owned_points)?)
        }
        TaggedSection::U32(sec) => {
            TaggedSection::U32(build_local_section_owned(sec, local_points, owned_points)?)
        }
        TaggedSection::U64(sec) => {
            TaggedSection::U64(build_local_section_owned(sec, local_points, owned_points)?)
        }
    })
}

fn complete_tagged_section_with_ownership<C>(
    section: &mut TaggedSection,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    C: Communicator + Sync,
{
    match section {
        TaggedSection::F64(sec) => complete_section_with_ownership::<f64, _, CopyDelta, C>(
            sec, overlap, ownership, comm, my_rank,
        ),
        TaggedSection::F32(sec) => complete_section_with_ownership::<f32, _, CopyDelta, C>(
            sec, overlap, ownership, comm, my_rank,
        ),
        TaggedSection::I32(sec) => complete_section_with_ownership::<i32, _, CopyDelta, C>(
            sec, overlap, ownership, comm, my_rank,
        ),
        TaggedSection::I64(sec) => complete_section_with_ownership::<i64, _, CopyDelta, C>(
            sec, overlap, ownership, comm, my_rank,
        ),
        TaggedSection::U32(sec) => complete_section_with_ownership::<u32, _, CopyDelta, C>(
            sec, overlap, ownership, comm, my_rank,
        ),
        TaggedSection::U64(sec) => complete_section_with_ownership::<u64, _, CopyDelta, C>(
            sec, overlap, ownership, comm, my_rank,
        ),
    }
}

fn complete_labels_with_ownership<C>(
    labels: &mut LabelSet,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
) -> Result<(), MeshSieveError>
where
    C: Communicator + Sync,
{
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    overlap.validate_invariants()?;

    let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
    nb.remove(&my_rank);
    if nb.is_empty() {
        return Ok(());
    }

    labels.clear_points(ownership.ghost_points());

    let mut send_payloads: HashMap<usize, Vec<u8>> = HashMap::new();
    for (name, point, value) in labels.iter() {
        let owner = ownership.owner_or_err(point)?;
        if owner != my_rank {
            continue;
        }
        let name_len: u16 = name
            .len()
            .try_into()
            .map_err(|_| MeshSieveError::CommError {
                neighbor: my_rank,
                source: format!("label name too long: {name}").into(),
            })?;
        for (_dst, rem) in overlap.cone(crate::overlap::overlap::local(point)) {
            if rem.rank == my_rank {
                continue;
            }
            let remote_pt = rem
                .remote_point
                .ok_or(MeshSieveError::OverlapLinkMissing(point, rem.rank))?;
            let payload = send_payloads.entry(rem.rank).or_default();
            payload.extend_from_slice(&remote_pt.get().to_le_bytes());
            payload.extend_from_slice(&value.to_le_bytes());
            payload.extend_from_slice(&name_len.to_le_bytes());
            payload.extend_from_slice(name.as_bytes());
        }
    }

    let neighbors: HashSet<usize> = nb.iter().copied().collect();
    let tag = CommTag::new(0xD1A5);
    let counts = crate::algs::completion::size_exchange::exchange_sizes_symmetric::<_, u8>(
        &send_payloads,
        comm,
        tag,
        &neighbors,
    )?;

    let mut recvs = Vec::new();
    for &nbr in &nb {
        let n = counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![0u8; n];
        let h = comm.irecv_result(nbr, tag.offset(1).as_u16(), &mut buf)?;
        recvs.push((nbr, h, buf));
    }

    let mut sends = Vec::new();
    for &nbr in &nb {
        let out = send_payloads.get(&nbr).map_or(&[][..], |v| &v[..]);
        sends.push(comm.isend_result(nbr, tag.offset(1).as_u16(), out)?);
    }

    let mut maybe_err: Option<MeshSieveError> = None;
    for (nbr, h, mut buf) in recvs {
        match h.wait() {
            Some(raw) if raw.len() == buf.len() => {
                buf.copy_from_slice(&raw);
                if let Err(err) = decode_label_payload(&buf, nbr).and_then(|entries| {
                    for (point, name, value) in entries {
                        labels.set_label(point, &name, value);
                    }
                    Ok(())
                }) {
                    if maybe_err.is_none() {
                        maybe_err = Some(err);
                    }
                }
            }
            Some(raw) if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!(
                        "label payload size mismatch: expected {}B, got {}B",
                        buf.len(),
                        raw.len()
                    )
                    .into(),
                });
            }
            None if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("failed to receive label payload from rank {nbr}").into(),
                });
            }
            _ => {}
        }
    }

    for send in sends {
        let _ = send.wait();
    }

    if let Some(err) = maybe_err {
        Err(err)
    } else {
        Ok(())
    }
}

fn decode_label_payload(
    buf: &[u8],
    neighbor: usize,
) -> Result<Vec<(PointId, String, i32)>, MeshSieveError> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < buf.len() {
        let remaining = buf.len() - idx;
        if remaining < (8 + 4 + 2) {
            return Err(MeshSieveError::CommError {
                neighbor,
                source: "label payload truncated header".into(),
            });
        }
        let mut raw_id = [0u8; 8];
        raw_id.copy_from_slice(&buf[idx..idx + 8]);
        idx += 8;
        let mut raw_val = [0u8; 4];
        raw_val.copy_from_slice(&buf[idx..idx + 4]);
        idx += 4;
        let mut raw_len = [0u8; 2];
        raw_len.copy_from_slice(&buf[idx..idx + 2]);
        idx += 2;
        let name_len = u16::from_le_bytes(raw_len) as usize;
        if idx + name_len > buf.len() {
            return Err(MeshSieveError::CommError {
                neighbor,
                source: "label payload truncated name".into(),
            });
        }
        let name_bytes = &buf[idx..idx + name_len];
        idx += name_len;
        let name = std::str::from_utf8(name_bytes).map_err(|e| MeshSieveError::CommError {
            neighbor,
            source: format!("label name invalid utf8: {e}").into(),
        })?;
        let point = PointId::new(u64::from_le_bytes(raw_id))?;
        let value = i32::from_le_bytes(raw_val);
        out.push((point, name.to_string(), value));
    }
    Ok(out)
}

fn resolve_overlap_via_exchange<C>(
    overlap: &mut Overlap,
    comm: &C,
    my_rank: usize,
    periodic_remote_map: Option<&BTreeMap<(PointId, usize), PointId>>,
    use_comm_completion: bool,
) -> Result<(), MeshSieveError>
where
    C: Communicator + Sync,
{
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants"
    ))]
    overlap.validate_invariants()?;

    let mut nb: BTreeSet<usize> = overlap.neighbor_ranks().collect();
    nb.remove(&my_rank);
    if nb.is_empty() {
        return Ok(());
    }

    let neighbors: Vec<usize> = nb.into_iter().collect();
    if !use_comm_completion || comm.is_no_comm() || comm.size() <= 1 {
        for &nbr in &neighbors {
            let points: Vec<PointId> = overlap.links_to(nbr).map(|(p, _)| p).collect();
            for p in points {
                let remote = periodic_remote_map
                    .and_then(|map| map.get(&(p, nbr)).copied())
                    .unwrap_or(p);
                overlap.resolve_remote_point(p, nbr, remote)?;
            }
        }
        return Ok(());
    }

    let mut send_points: HashMap<usize, Vec<WirePointRepr>> = HashMap::new();
    let mut local_points: HashMap<usize, Vec<PointId>> = HashMap::new();
    for &nbr in &neighbors {
        let mut pts: Vec<PointId> = overlap.links_to(nbr).map(|(p, _)| p).collect();
        pts.sort_unstable();
        pts.dedup();
        let send: Vec<WirePointRepr> = pts.iter().map(|p| WirePointRepr::of(p.get())).collect();
        send_points.insert(nbr, send);
        local_points.insert(nbr, pts);
    }

    let all_neighbors: HashSet<usize> = neighbors.iter().copied().collect();
    let tag = CommTag::new(0xD00D);
    let counts = crate::algs::completion::size_exchange::exchange_sizes_symmetric(
        &send_points,
        comm,
        tag,
        &all_neighbors,
    )?;

    let mut recvs = Vec::new();
    for &nbr in &neighbors {
        let n = counts.get(&nbr).copied().unwrap_or(0) as usize;
        let mut buf = vec![WirePointRepr::zeroed(); n];
        let h = comm.irecv_result(nbr, tag.offset(1).as_u16(), cast_slice_mut(&mut buf))?;
        recvs.push((nbr, h, buf));
    }

    let mut sends = Vec::new();
    for &nbr in &neighbors {
        let out = send_points.get(&nbr).map_or(&[][..], |v| &v[..]);
        sends.push(comm.isend_result(nbr, tag.offset(1).as_u16(), cast_slice(out))?);
    }

    let mut recv_sets: HashMap<usize, HashSet<u64>> = HashMap::new();
    let mut maybe_err: Option<MeshSieveError> = None;
    for (nbr, h, mut buf) in recvs {
        match h.wait() {
            Some(raw) if raw.len() == buf.len() * std::mem::size_of::<WirePointRepr>() => {
                cast_slice_mut(&mut buf).copy_from_slice(&raw);
                let set: HashSet<u64> = buf.iter().map(|p| p.get()).collect();
                recv_sets.insert(nbr, set);
            }
            Some(raw) if maybe_err.is_none() => {
                let exp = buf.len() * std::mem::size_of::<WirePointRepr>();
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: format!("payload size mismatch: expected {exp}B, got {}B", raw.len())
                        .into(),
                });
            }
            None if maybe_err.is_none() => {
                maybe_err = Some(MeshSieveError::CommError {
                    neighbor: nbr,
                    source: "recv returned None".into(),
                });
            }
            _ => {}
        }
    }

    for h in sends {
        let _ = h.wait();
    }

    if let Some(err) = maybe_err {
        return Err(err);
    }

    for (&nbr, points) in &local_points {
        let recv_set = recv_sets.get(&nbr).ok_or(MeshSieveError::MissingOverlap {
            source: format!("missing neighbor list from rank {nbr}").into(),
        })?;
        for &p in points {
            let remote = periodic_remote_map
                .and_then(|map| map.get(&(p, nbr)).copied())
                .unwrap_or(p);
            if !recv_set.contains(&remote.get()) {
                return Err(MeshSieveError::MissingOverlap {
                    source: format!("neighbor {nbr} missing shared point {}", remote.get()).into(),
                });
            }
            overlap.resolve_remote_point(p, nbr, remote)?;
        }
    }

    Ok(())
}

fn should_use_comm_completion<C: Communicator + Sync + 'static>(comm: &C) -> bool {
    !comm.is_no_comm()
        && comm.size() > 1
        && std::any::TypeId::of::<C>()
            != std::any::TypeId::of::<crate::algs::communicator::RayonComm>()
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

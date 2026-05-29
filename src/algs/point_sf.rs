//! DMPlex-style PointSF and migration helpers.
//!
//! A PETSc `PetscSF` is a star forest whose leaves name local points and whose
//! roots name remote `(rank, point)` owners.  This module keeps that mapping as
//! first-class data instead of treating it as only a completion wrapper around
//! [`Overlap`].  The existing overlap-based completion entry points remain for
//! backwards compatibility, while the owned root/leaf tables can be used to
//! construct process SFs, migration SFs, overlap SFs, and data-distribution
//! results in a DMPlex-like pipeline.

use crate::algs::communicator::Communicator;
use crate::algs::completion::{complete_section, complete_section_with_ownership};
use crate::data::atlas::Atlas;
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::delta::CopyDelta;
use crate::overlap::overlap::Overlap;
use crate::topology::labels::LabelSet;
use crate::topology::ownership::{OwnershipEntry, PointOwnership};
use crate::topology::point::PointId;
use crate::topology::sieve::{MeshSieve, OrientedSieve, Sieve};
use std::collections::{BTreeMap, BTreeSet};

/// Diagnostics for SF map validation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SfValidationReport {
    pub domain_points: usize,
    pub range_points: usize,
    pub duplicate_leaves: Vec<PointId>,
    pub unmapped_ghosts: Vec<PointId>,
}

/// A remote root in a star forest.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RemotePoint {
    /// MPI rank that owns or otherwise roots the point.
    pub rank: usize,
    /// Point identifier on `rank`.
    pub point: PointId,
}

/// One leaf edge in a point star forest.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PointSfLeaf {
    /// Local leaf point.
    pub local: PointId,
    /// Remote root reached by this leaf.
    pub remote: RemotePoint,
    /// Ownership rank used for deciding ghost/owned semantics.
    pub owner_rank: usize,
    /// True when `local` is a ghost on this rank.
    pub is_ghost: bool,
}

/// Owned result of a generic distribution operation.
#[derive(Clone, Debug)]
pub struct SfDistribution<T, C: 'static> {
    /// Migrated/local data.
    pub data: T,
    /// Star forest used to migrate or complete the data.
    pub sf: PointSF<'static, C>,
}

/// Point star forest with optional overlap/communicator adapters.
#[derive(Clone, Debug)]
pub struct PointSF<'a, C> {
    leaves: Vec<PointSfLeaf>,
    roots: BTreeSet<PointId>,
    owner: BTreeMap<PointId, OwnershipEntry>,
    overlap: Option<&'a Overlap>,
    comm: Option<&'a C>,
    ownership_ref: Option<&'a PointOwnership>,
    my_rank: usize,
}

impl<'a, C> PointSF<'a, C>
where
    C: Communicator + Sync,
{
    /// Create a PointSF from owned root/leaf mapping tables.
    pub fn from_leaves<I>(my_rank: usize, roots: I, leaves: Vec<PointSfLeaf>) -> Self
    where
        I: IntoIterator<Item = PointId>,
    {
        let roots: BTreeSet<_> = roots.into_iter().collect();
        let mut owner = BTreeMap::new();
        for root in &roots {
            owner.insert(
                *root,
                OwnershipEntry {
                    owner: my_rank,
                    is_ghost: false,
                },
            );
        }
        for leaf in &leaves {
            owner.insert(
                leaf.local,
                OwnershipEntry {
                    owner: leaf.owner_rank,
                    is_ghost: leaf.is_ghost,
                },
            );
        }
        Self {
            leaves,
            roots,
            owner,
            overlap: None,
            comm: None,
            ownership_ref: None,
            my_rank,
        }
    }

    /// Create an identity SF over points owned by this rank.
    pub fn identity<I>(my_rank: usize, points: I) -> Self
    where
        I: IntoIterator<Item = PointId>,
    {
        let points: Vec<_> = points.into_iter().collect();
        let leaves = points
            .iter()
            .copied()
            .map(|point| PointSfLeaf {
                local: point,
                remote: RemotePoint {
                    rank: my_rank,
                    point,
                },
                owner_rank: my_rank,
                is_ghost: false,
            })
            .collect();
        Self::from_leaves(my_rank, points, leaves)
    }

    /// Create an SF from explicit local-to-remote point/rank migration edges.
    pub fn from_point_map<I>(my_rank: usize, edges: I) -> Self
    where
        I: IntoIterator<Item = (PointId, usize, PointId)>,
    {
        let mut remote_roots = BTreeSet::new();
        let leaves = edges
            .into_iter()
            .map(|(local, remote_rank, remote_point)| {
                if remote_rank == my_rank {
                    remote_roots.insert(remote_point);
                }
                PointSfLeaf {
                    local,
                    remote: RemotePoint {
                        rank: remote_rank,
                        point: remote_point,
                    },
                    owner_rank: remote_rank,
                    is_ghost: remote_rank != my_rank,
                }
            })
            .collect();
        Self::from_leaves(my_rank, remote_roots, leaves)
    }

    /// Number of local leaf edges represented by this SF.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Create a PointSF without ownership metadata from an overlap graph.
    pub fn new(overlap: &'a Overlap, comm: &'a C, my_rank: usize) -> Self {
        Self::from_overlap(overlap, None, Some(comm), my_rank)
    }

    /// Create a PointSF with ownership metadata from an overlap graph.
    pub fn with_ownership(
        overlap: &'a Overlap,
        ownership: &'a PointOwnership,
        comm: &'a C,
        my_rank: usize,
    ) -> Self {
        Self::from_overlap(overlap, Some(ownership), Some(comm), my_rank)
    }

    /// Build a first-class PointSF from an overlap and optional ownership map.
    pub fn from_overlap(
        overlap: &'a Overlap,
        ownership: Option<&'a PointOwnership>,
        comm: Option<&'a C>,
        my_rank: usize,
    ) -> Self {
        let mut leaves = Vec::new();
        let mut roots = BTreeSet::new();
        let mut owner = BTreeMap::new();
        if let Some(ownership) = ownership {
            for p in ownership.local_points() {
                if let Some(entry) = ownership.entry(p) {
                    owner.insert(p, entry);
                    if !entry.is_ghost {
                        roots.insert(p);
                    }
                }
            }
        }
        for rank in overlap.neighbor_ranks() {
            for (local, remote) in overlap.links_to(rank) {
                let remote_point = remote.unwrap_or(local);
                let entry = ownership
                    .and_then(|o| o.entry(local))
                    .unwrap_or(OwnershipEntry {
                        owner: rank,
                        is_ghost: rank != my_rank,
                    });
                leaves.push(PointSfLeaf {
                    local,
                    remote: RemotePoint {
                        rank,
                        point: remote_point,
                    },
                    owner_rank: entry.owner,
                    is_ghost: entry.is_ghost,
                });
                owner.insert(local, entry);
            }
        }
        leaves.sort_unstable();
        Self {
            leaves,
            roots,
            owner,
            overlap: Some(overlap),
            comm,
            ownership_ref: ownership,
            my_rank,
        }
    }

    /// Borrow the underlying overlap graph, if this SF was made from one.
    pub fn overlap(&self) -> Option<&'a Overlap> {
        self.overlap
    }

    /// Borrow the communicator, if available.
    pub fn comm(&self) -> Option<&'a C> {
        self.comm
    }

    /// Borrow optional ownership metadata supplied at construction.
    pub fn ownership(&self) -> Option<&'a PointOwnership> {
        self.ownership_ref
    }

    /// Rank for this PointSF.
    pub fn rank(&self) -> usize {
        self.my_rank
    }

    /// Local root points in deterministic order.
    pub fn roots(&self) -> impl Iterator<Item = PointId> + '_ {
        self.roots.iter().copied()
    }

    /// Leaf edges in deterministic order.
    pub fn leaves(&self) -> impl Iterator<Item = &PointSfLeaf> + '_ {
        self.leaves.iter()
    }

    /// Local ownership entry recorded in this SF.
    pub fn ownership_entry(&self, point: PointId) -> Option<OwnershipEntry> {
        self.owner.get(&point).copied()
    }

    /// Convert the SF root/leaf ownership metadata into a [`PointOwnership`] map.
    pub fn to_point_ownership(&self) -> Result<PointOwnership, MeshSieveError> {
        let mut out = PointOwnership::default();
        for (&point, entry) in &self.owner {
            out.set(point, entry.owner, entry.is_ghost)?;
        }
        Ok(out)
    }

    /// Validate overlap/ownership/SF consistency in debug builds.
    pub fn validate(&self) -> Result<(), MeshSieveError> {
        #[cfg(any(
            debug_assertions,
            feature = "strict-invariants",
            feature = "check-invariants"
        ))]
        {
            if let Some(overlap) = self.overlap {
                overlap.validate_invariants()?;
                if let Some(ownership) = self.ownership_ref {
                    for src in overlap.base_points() {
                        if let Some(point) = src.as_local()
                            && ownership.entry(point).is_none()
                        {
                            return Err(MeshSieveError::OverlapPointMissingOwnership { point });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compose `self` with `next`, i.e. `next ∘ self`.
    pub fn compose(&self, next: &PointSF<'_, C>) -> PointSF<'static, C> {
        let mut composed = Vec::new();
        let next_by_local: BTreeMap<PointId, RemotePoint> = next
            .leaves
            .iter()
            .map(|leaf| (leaf.local, leaf.remote))
            .collect();
        for leaf in &self.leaves {
            let remote = next_by_local
                .get(&leaf.remote.point)
                .copied()
                .unwrap_or(leaf.remote);
            composed.push(PointSfLeaf {
                local: leaf.local,
                remote,
                owner_rank: remote.rank,
                is_ghost: remote.rank != self.my_rank,
            });
        }
        let roots: Vec<_> = composed
            .iter()
            .filter_map(|leaf| (leaf.remote.rank == self.my_rank).then_some(leaf.remote.point))
            .collect();
        PointSF::from_leaves(self.my_rank, roots, composed)
    }

    /// Invert this SF if leaves map uniquely by remote point.
    pub fn invert(&self) -> Result<PointSF<'static, C>, MeshSieveError> {
        let mut seen = BTreeSet::new();
        let mut inv = Vec::with_capacity(self.leaves.len());
        for leaf in &self.leaves {
            if !seen.insert(leaf.remote.point) {
                return Err(MeshSieveError::MeshIoParse(
                    "cannot invert SF with duplicate remote points".to_string(),
                ));
            }
            inv.push(PointSfLeaf {
                local: leaf.remote.point,
                remote: RemotePoint {
                    rank: self.my_rank,
                    point: leaf.local,
                },
                owner_rank: self.my_rank,
                is_ghost: false,
            });
        }
        let roots = self.leaves.iter().map(|l| l.local);
        Ok(PointSF::from_leaves(self.my_rank, roots, inv))
    }

    /// Restrict this SF to points in a label value.
    pub fn restrict_to_label(
        &self,
        labels: &LabelSet,
        name: &str,
        value: i32,
    ) -> PointSF<'static, C> {
        let leaves: Vec<_> = self
            .leaves
            .iter()
            .copied()
            .filter(|leaf| labels.get_label(leaf.local, name) == Some(value))
            .collect();
        let roots: Vec<_> = leaves.iter().map(|leaf| leaf.remote.point).collect();
        PointSF::from_leaves(self.my_rank, roots, leaves)
    }

    /// Validate cardinality and mapping coherence, reporting duplicates and unmapped ghosts.
    pub fn validate_mapping(&self) -> SfValidationReport {
        let mut report = SfValidationReport::default();
        let mut domain = BTreeSet::new();
        let mut range = BTreeSet::new();
        let mut counts: BTreeMap<PointId, usize> = BTreeMap::new();
        for leaf in &self.leaves {
            domain.insert(leaf.local);
            range.insert(leaf.remote.point);
            *counts.entry(leaf.local).or_insert(0) += 1;
        }
        report.domain_points = domain.len();
        report.range_points = range.len();
        report.duplicate_leaves = counts
            .into_iter()
            .filter_map(|(p, n)| (n > 1).then_some(p))
            .collect();
        report.unmapped_ghosts = self
            .owner
            .iter()
            .filter_map(|(&p, entry)| (entry.is_ghost && !domain.contains(&p)).then_some(p))
            .collect();
        report
    }

    /// Complete a section using CopyDelta and optional ownership metadata.
    pub fn complete_section<V, S>(&self, section: &mut Section<V, S>) -> Result<(), MeshSieveError>
    where
        V: Clone + Default + Send + PartialEq + bytemuck::Pod + 'static,
        S: Storage<V>,
    {
        self.validate()?;
        let overlap = self.overlap.ok_or(MeshSieveError::MissingOverlap {
            source: "PointSF has no overlap adapter for section completion".into(),
        })?;
        let comm = self.comm.ok_or(MeshSieveError::CommError {
            neighbor: self.my_rank,
            source: "PointSF has no communicator for section completion".into(),
        })?;
        if let Some(ownership) = self.ownership_ref {
            complete_section_with_ownership::<V, S, CopyDelta, C>(
                section,
                overlap,
                ownership,
                comm,
                self.my_rank,
            )
        } else {
            complete_section::<V, S, CopyDelta, C>(section, overlap, comm, self.my_rank)
        }
    }
}

/// Create the point SF describing current local roots and ghost leaves.
pub fn create_point_sf<C>(
    overlap: &Overlap,
    ownership: &PointOwnership,
    my_rank: usize,
) -> PointSF<'static, C>
where
    C: Communicator + Sync,
{
    let mut leaves = Vec::new();
    for rank in overlap.neighbor_ranks() {
        for (local, remote) in overlap.links_to(rank) {
            let entry = ownership.entry(local).unwrap_or(OwnershipEntry {
                owner: rank,
                is_ghost: rank != my_rank,
            });
            leaves.push(PointSfLeaf {
                local,
                remote: RemotePoint {
                    rank,
                    point: remote.unwrap_or(local),
                },
                owner_rank: entry.owner,
                is_ghost: entry.is_ghost,
            });
        }
    }
    leaves.sort_unstable();
    PointSF::from_leaves(my_rank, ownership.owned_points(), leaves)
}

/// Create a process SF: every local point is rooted at its owning rank.
pub fn create_process_sf<C>(ownership: &PointOwnership, my_rank: usize) -> PointSF<'static, C>
where
    C: Communicator + Sync,
{
    let leaves = ownership
        .local_points()
        .filter_map(|p| ownership.entry(p).map(|entry| (p, entry)))
        .map(|(local, entry)| PointSfLeaf {
            local,
            remote: RemotePoint {
                rank: entry.owner,
                point: local,
            },
            owner_rank: entry.owner,
            is_ghost: entry.owner != my_rank,
        })
        .collect();
    PointSF::from_leaves(my_rank, ownership.owned_points(), leaves)
}

/// Create a migration SF from old local points to new owner ranks.
pub fn create_migration_sf<C, I>(
    points: I,
    new_owners: &BTreeMap<PointId, usize>,
    my_rank: usize,
) -> PointSF<'static, C>
where
    C: Communicator + Sync,
    I: IntoIterator<Item = PointId>,
{
    let leaves = points
        .into_iter()
        .filter_map(|p| new_owners.get(&p).copied().map(|owner| (p, owner)))
        .map(|(local, owner)| PointSfLeaf {
            local,
            remote: RemotePoint {
                rank: owner,
                point: local,
            },
            owner_rank: owner,
            is_ghost: owner != my_rank,
        })
        .collect();
    let roots = new_owners
        .iter()
        .filter_map(|(&p, &owner)| (owner == my_rank).then_some(p));
    PointSF::from_leaves(my_rank, roots, leaves)
}

/// Create a two-sided process SF by retaining only ranks that mutually share points.
pub fn create_two_sided_process_sf<C>(
    overlap: &Overlap,
    ownership: &PointOwnership,
    my_rank: usize,
) -> PointSF<'static, C>
where
    C: Communicator + Sync,
{
    let mut sf = create_point_sf::<C>(overlap, ownership, my_rank);
    sf.leaves
        .retain(|leaf| leaf.remote.rank != my_rank && leaf.remote.point.get() > 0);
    sf
}

/// Create the migration SF induced by an overlap graph.
pub fn create_overlap_migration_sf<C>(
    overlap: &Overlap,
    ownership: &PointOwnership,
    my_rank: usize,
) -> PointSF<'static, C>
where
    C: Communicator + Sync,
{
    create_two_sided_process_sf::<C>(overlap, ownership, my_rank)
}

/// Distribute topology by filtering to owned roots plus SF leaves.
pub fn distribute_topology<M, C>(
    mesh: &M,
    ownership: &PointOwnership,
    sf: PointSF<'static, C>,
) -> Result<SfDistribution<MeshSieve, C>, MeshSieveError>
where
    M: OrientedSieve<Point = PointId, Payload = (), Orient = i32>,
    C: Communicator + Sync,
{
    let mut keep: BTreeSet<PointId> = ownership.local_points().collect();
    keep.extend(sf.leaves().map(|leaf| leaf.remote.point));
    let mut out = MeshSieve::default();
    for &p in &keep {
        out.add_point(p);
    }
    for src in mesh.points() {
        if !keep.contains(&src) {
            continue;
        }
        for (dst, orient) in mesh.cone_o(src) {
            if keep.contains(&dst) {
                out.add_arrow_o(src, dst, (), orient);
            }
        }
    }
    Ok(SfDistribution { data: out, sf })
}

/// Distribute a section and return the SF used.
pub fn distribute_section<V, St, C>(
    section: &Section<V, St>,
    ownership: &PointOwnership,
    sf: PointSF<'static, C>,
) -> Result<SfDistribution<Section<V, St>, C>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
    C: Communicator + Sync,
{
    let points: BTreeSet<_> = ownership.local_points().collect();
    let mut atlas = Atlas::default();
    for &p in &points {
        if let Some((_off, len)) = section.atlas().get(p) {
            atlas.try_insert(p, len)?;
        }
    }
    let mut out = Section::<V, St>::new(atlas);
    for p in ownership.owned_points() {
        if out.atlas().get(p).is_some() {
            out.try_set(p, section.try_restrict(p)?)?;
        }
    }
    Ok(SfDistribution { data: out, sf })
}

/// Distribute field data (alias of [`distribute_section`]).
pub fn distribute_field<V, St, C>(
    field: &Section<V, St>,
    ownership: &PointOwnership,
    sf: PointSF<'static, C>,
) -> Result<SfDistribution<Section<V, St>, C>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
    C: Communicator + Sync,
{
    distribute_section(field, ownership, sf)
}

/// Distribute labels and return the SF used.
pub fn distribute_labels<C>(
    labels: &LabelSet,
    ownership: &PointOwnership,
    sf: PointSF<'static, C>,
) -> Result<SfDistribution<LabelSet, C>, MeshSieveError>
where
    C: Communicator + Sync,
{
    Ok(SfDistribution {
        data: labels.filtered_to_points(ownership.local_points()),
        sf,
    })
}

/// Balance ownership of partition-boundary points using deterministic least-load assignment.
pub fn balance_partition_boundary_ownership(
    point_owners: &mut [usize],
    point_sharing: &BTreeMap<PointId, BTreeSet<usize>>,
    n_ranks: usize,
) -> Result<(), MeshSieveError> {
    let mut load = vec![0usize; n_ranks.max(1)];
    for &owner in point_owners.iter() {
        if owner >= load.len() {
            return Err(MeshSieveError::PartitionIndexOutOfBounds(owner));
        }
        load[owner] += 1;
    }
    for (&point, ranks) in point_sharing {
        if ranks.len() < 2 {
            continue;
        }
        let idx = point
            .get()
            .checked_sub(1)
            .ok_or(MeshSieveError::InvalidPointId)? as usize;
        if idx >= point_owners.len() {
            return Err(MeshSieveError::PartitionIndexOutOfBounds(idx));
        }
        let current = point_owners[idx];
        let best = ranks
            .iter()
            .copied()
            .filter(|&rank| rank < load.len())
            .min_by_key(|&rank| (load[rank], rank))
            .unwrap_or(current);
        if best != current {
            load[current] = load[current].saturating_sub(1);
            load[best] += 1;
            point_owners[idx] = best;
        }
    }
    Ok(())
}

//! Ownership metadata for mesh points.
//!
//! The [`PointOwnership`] map records the owning rank and whether a point is a ghost
//! on the current rank, enabling consistent partition-aware queries.

use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use std::collections::BTreeSet;

#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OwnershipEntry {
    pub owner: usize,
    pub is_ghost: bool,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PointOwnership {
    entries: Vec<Option<OwnershipEntry>>,
}

impl PointOwnership {
    /// Create an empty ownership map sized for `max_id` points.
    pub fn with_capacity(max_id: usize) -> Self {
        Self {
            entries: vec![None; max_id],
        }
    }

    /// Returns the number of tracked point slots (including empty entries).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if no entries exist in the map.
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|entry| entry.is_none())
    }

    /// Insert or update an ownership entry for `point`.
    pub fn set(
        &mut self,
        point: PointId,
        owner: usize,
        is_ghost: bool,
    ) -> Result<(), MeshSieveError> {
        let idx = point
            .get()
            .checked_sub(1)
            .ok_or(MeshSieveError::InvalidPointId)? as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, None);
        }
        self.entries[idx] = Some(OwnershipEntry { owner, is_ghost });
        Ok(())
    }

    /// Insert or update an ownership entry using `my_rank` to determine ghostness.
    pub fn set_from_owner(
        &mut self,
        point: PointId,
        owner: usize,
        my_rank: usize,
    ) -> Result<(), MeshSieveError> {
        let is_ghost = owner != my_rank;
        self.set(point, owner, is_ghost)
    }

    /// Insert or update an ownership entry, keeping the smallest owner when repeated.
    pub fn set_owner_min(
        &mut self,
        point: PointId,
        owner: usize,
        my_rank: usize,
    ) -> Result<(), MeshSieveError> {
        let idx = point
            .get()
            .checked_sub(1)
            .ok_or(MeshSieveError::InvalidPointId)? as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, None);
        }
        let updated_owner = match self.entries[idx] {
            Some(existing) => existing.owner.min(owner),
            None => owner,
        };
        self.entries[idx] = Some(OwnershipEntry {
            owner: updated_owner,
            is_ghost: updated_owner != my_rank,
        });
        Ok(())
    }

    /// Retrieve the ownership entry for a point.
    pub fn entry(&self, point: PointId) -> Option<OwnershipEntry> {
        let idx = point.get().checked_sub(1)? as usize;
        self.entries.get(idx).copied().flatten()
    }

    /// Retrieve the owning rank for a point.
    pub fn owner(&self, point: PointId) -> Option<usize> {
        self.entry(point).map(|entry| entry.owner)
    }

    /// Retrieve the owning rank or return an error if missing.
    pub fn owner_or_err(&self, point: PointId) -> Result<usize, MeshSieveError> {
        self.owner(point)
            .ok_or(MeshSieveError::MissingOwnership(point))
    }

    /// Returns whether the point is marked as a ghost on this rank.
    pub fn is_ghost(&self, point: PointId) -> Option<bool> {
        self.entry(point).map(|entry| entry.is_ghost)
    }

    /// Returns true if the point is owned by `rank`.
    pub fn is_owned_by(&self, point: PointId, rank: usize) -> bool {
        self.owner(point).is_some_and(|owner| owner == rank)
    }

    /// Iterate over all local points tracked by this map.
    pub fn local_points(&self) -> impl Iterator<Item = PointId> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| entry.map(|_| PointId::new((idx + 1) as u64).ok()).flatten())
    }

    /// Iterate over owned points (non-ghosts).
    pub fn owned_points(&self) -> impl Iterator<Item = PointId> + '_ {
        self.entries.iter().enumerate().filter_map(|(idx, entry)| {
            entry.and_then(|entry| {
                if entry.is_ghost {
                    None
                } else {
                    PointId::new((idx + 1) as u64).ok()
                }
            })
        })
    }

    /// Iterate over ghost points.
    pub fn ghost_points(&self) -> impl Iterator<Item = PointId> + '_ {
        self.entries.iter().enumerate().filter_map(|(idx, entry)| {
            entry.and_then(|entry| {
                if entry.is_ghost {
                    PointId::new((idx + 1) as u64).ok()
                } else {
                    None
                }
            })
        })
    }

    /// Collect all local points into a sorted set.
    pub fn local_set(&self) -> BTreeSet<PointId> {
        self.local_points().collect()
    }

    /// Collect all ghost points into a sorted set.
    pub fn ghost_set(&self) -> BTreeSet<PointId> {
        self.ghost_points().collect()
    }

    /// Collect all owned points into a sorted set.
    pub fn owned_set(&self) -> BTreeSet<PointId> {
        self.owned_points().collect()
    }

    /// Return a new ownership map containing only the provided points.
    pub fn filtered_to_points<I>(&self, points: I) -> Result<Self, MeshSieveError>
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut filtered = Self::default();
        for point in points {
            let entry = self
                .entry(point)
                .ok_or(MeshSieveError::MissingOwnership(point))?;
            filtered.set(point, entry.owner, entry.is_ghost)?;
        }
        Ok(filtered)
    }

    /// Build a point-ownership map from a local set and per-point owners.
    pub fn from_local_set<I>(
        local_set: I,
        owners: &[usize],
        my_rank: usize,
    ) -> Result<Self, MeshSieveError>
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut ownership = Self::with_capacity(owners.len());
        for point in local_set {
            let idx = point
                .get()
                .checked_sub(1)
                .ok_or(MeshSieveError::InvalidPointId)? as usize;
            let owner = owners
                .get(idx)
                .copied()
                .ok_or(MeshSieveError::PartitionIndexOutOfBounds(idx))?;
            ownership.set_from_owner(point, owner, my_rank)?;
        }
        Ok(ownership)
    }
}

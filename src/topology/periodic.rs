//! Periodic point identification utilities.
//!
//! This module provides two complementary layers for periodic identification:
//! - [`PointEquivalence`], a union-find structure that tracks equivalence classes.
//! - [`PeriodicMap`], a master/slave mapping for assembly or constraint workflows.
//!
//! The utilities here can be used to collapse point IDs into a quotient topology or
//! to generate mapping layers for periodic assembly.

use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::{BTreeMap, BTreeSet};

/// Union-find structure tracking equivalence classes of points.
#[derive(Debug, Default, Clone)]
pub struct PointEquivalence {
    parent: BTreeMap<PointId, PointId>,
    rank: BTreeMap<PointId, u32>,
}

impl PointEquivalence {
    /// Create an empty equivalence relation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize an equivalence structure with the provided points.
    pub fn with_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut eq = Self::default();
        for p in points {
            eq.add_point(p);
        }
        eq
    }

    /// Insert a point into the equivalence structure (as its own class).
    pub fn add_point(&mut self, point: PointId) {
        self.parent.entry(point).or_insert(point);
        self.rank.entry(point).or_insert(0);
    }

    /// Iterate over all points currently tracked.
    pub fn points(&self) -> impl Iterator<Item = PointId> + '_ {
        self.parent.keys().copied()
    }

    fn find_root(&mut self, point: PointId) -> PointId {
        if !self.parent.contains_key(&point) {
            self.add_point(point);
            return point;
        }
        let parent = self.parent[&point];
        if parent == point {
            return point;
        }
        let root = self.find_root(parent);
        self.parent.insert(point, root);
        root
    }

    /// Return the canonical representative for a point (with path compression).
    pub fn representative(&mut self, point: PointId) -> PointId {
        self.find_root(point)
    }

    /// Union two points and return the representative.
    pub fn union(&mut self, a: PointId, b: PointId) -> PointId {
        let ra = self.find_root(a);
        let rb = self.find_root(b);
        if ra == rb {
            return ra;
        }
        let rank_a = self.rank.get(&ra).copied().unwrap_or(0);
        let rank_b = self.rank.get(&rb).copied().unwrap_or(0);
        if rank_a < rank_b {
            self.parent.insert(ra, rb);
            rb
        } else {
            self.parent.insert(rb, ra);
            if rank_a == rank_b {
                self.rank.insert(ra, rank_a + 1);
            }
            ra
        }
    }

    /// Record that two points are equivalent.
    pub fn add_equivalence(&mut self, a: PointId, b: PointId) {
        self.union(a, b);
    }

    /// Check whether two points are in the same equivalence class.
    pub fn are_equivalent(&mut self, a: PointId, b: PointId) -> bool {
        self.find_root(a) == self.find_root(b)
    }

    /// Build a canonical map for a set of points.
    pub fn canonical_map<I>(&mut self, points: I) -> BTreeMap<PointId, PointId>
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut map = BTreeMap::new();
        for p in points {
            map.insert(p, self.find_root(p));
        }
        map
    }

    /// Partition points into equivalence classes.
    pub fn classes<I>(&mut self, points: I) -> BTreeMap<PointId, Vec<PointId>>
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut classes: BTreeMap<PointId, Vec<PointId>> = BTreeMap::new();
        for p in points {
            let rep = self.find_root(p);
            classes.entry(rep).or_default().push(p);
        }
        classes
    }

    /// Return all current representatives in sorted order.
    pub fn representatives(&mut self) -> BTreeSet<PointId> {
        let points: Vec<_> = self.points().collect();
        points.into_iter().map(|p| self.find_root(p)).collect()
    }
}

/// Master/slave periodic map for assembly and constraint workflows.
#[derive(Debug, Default, Clone)]
pub struct PeriodicMap {
    master_for: BTreeMap<PointId, PointId>,
}

impl PeriodicMap {
    /// Create an empty periodic map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a master/slave pair.
    ///
    /// Returns an error if the slave already maps to a different master.
    pub fn insert_pair(&mut self, master: PointId, slave: PointId) -> Result<(), MeshSieveError> {
        if let Some(existing) = self.master_for.get(&slave) {
            if *existing != master {
                return Err(MeshSieveError::PeriodicMappingConflict {
                    slave,
                    existing: *existing,
                    new: master,
                });
            }
        }
        self.master_for.insert(slave, master);
        Ok(())
    }

    /// Retrieve the master for a slave point.
    pub fn master_of(&self, slave: PointId) -> Option<PointId> {
        self.master_for.get(&slave).copied()
    }

    /// Iterate over master/slave pairs.
    pub fn pairs(&self) -> impl Iterator<Item = (PointId, PointId)> + '_ {
        self.master_for
            .iter()
            .map(|(slave, master)| (*master, *slave))
    }

    /// Convert the periodic map into an equivalence relation.
    pub fn equivalence(&self) -> PointEquivalence {
        let mut eq = PointEquivalence::new();
        for (master, slave) in self.pairs() {
            eq.add_equivalence(master, slave);
        }
        eq
    }
}

/// Collapse points into their canonical representatives.
pub fn collapse_points<I>(
    points: I,
    equivalence: &mut PointEquivalence,
) -> BTreeMap<PointId, PointId>
where
    I: IntoIterator<Item = PointId>,
{
    equivalence.canonical_map(points)
}

/// Build a quotient sieve by collapsing equivalent points.
///
/// Any arrow that maps to a self-loop is skipped.
pub fn quotient_sieve<S>(sieve: &S, equivalence: &mut PointEquivalence) -> S
where
    S: Sieve + Default,
    S::Payload: Clone,
{
    let mut quotient = S::default();
    for src in sieve.base_points() {
        let rep_src = equivalence.representative(src);
        for (dst, payload) in sieve.cone(src) {
            let rep_dst = equivalence.representative(dst);
            if rep_src == rep_dst {
                continue;
            }
            quotient.add_arrow(rep_src, rep_dst, payload.clone());
        }
    }
    quotient
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equivalence_groups_points() {
        let a = PointId::new(1).unwrap();
        let b = PointId::new(2).unwrap();
        let c = PointId::new(3).unwrap();
        let mut eq = PointEquivalence::new();
        eq.add_equivalence(a, b);
        assert!(eq.are_equivalent(a, b));
        assert!(!eq.are_equivalent(a, c));
        eq.add_equivalence(b, c);
        assert!(eq.are_equivalent(a, c));
    }

    #[test]
    fn periodic_map_rejects_conflicts() {
        let master_a = PointId::new(1).unwrap();
        let master_b = PointId::new(2).unwrap();
        let slave = PointId::new(3).unwrap();
        let mut map = PeriodicMap::new();
        map.insert_pair(master_a, slave).unwrap();
        let err = map.insert_pair(master_b, slave).unwrap_err();
        match err {
            MeshSieveError::PeriodicMappingConflict { slave: s, .. } => {
                assert_eq!(s, slave);
            }
            _ => panic!("unexpected error"),
        }
    }
}

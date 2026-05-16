//! Topological anchors for nonconforming/adapted meshes.
//!
//! Anchors record the coarse parent point(s) that define an adapted point.  They
//! are intentionally topology-level metadata (not field data) so adjacency,
//! closure, ownership, and constraint generation can all refer to the same
//! parentage after local refinement or coarsening.

use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Classification for an anchored point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnchorKind {
    /// A regular child produced by a conforming refinement/coarsening template.
    Refined,
    /// A constrained point, typically a midpoint on an unsplit neighbour edge.
    Hanging,
    /// A point that is constrained but should be retained as an explicit anchor.
    Constrained,
}

/// Parentage for one adapted point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PointAnchor {
    /// Parents defining this point; two parents denote an edge midpoint, more
    /// parents denote face/cell interpolation, and one parent denotes identity.
    pub parents: Vec<PointId>,
    /// Whether the point is regular, hanging, or otherwise constrained.
    pub kind: AnchorKind,
}

impl PointAnchor {
    /// Construct a new point anchor with sorted, duplicate-free parents.
    pub fn new<I>(parents: I, kind: AnchorKind) -> Self
    where
        I: IntoIterator<Item = PointId>,
    {
        let mut parents: Vec<_> = parents.into_iter().collect();
        parents.sort_unstable();
        parents.dedup();
        Self { parents, kind }
    }

    /// Returns true when this point requires a constraint equation.
    pub fn is_constrained(&self) -> bool {
        matches!(self.kind, AnchorKind::Hanging | AnchorKind::Constrained)
    }
}

/// DMPlex-style anchor map for adapted/nonconforming topology.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TopologicalAnchors {
    anchors: BTreeMap<PointId, PointAnchor>,
}

impl TopologicalAnchors {
    /// Insert or replace parentage for a point.
    pub fn insert<I>(&mut self, point: PointId, parents: I, kind: AnchorKind)
    where
        I: IntoIterator<Item = PointId>,
    {
        self.anchors.insert(point, PointAnchor::new(parents, kind));
    }

    /// Borrow an anchor entry.
    pub fn get(&self, point: PointId) -> Option<&PointAnchor> {
        self.anchors.get(&point)
    }

    /// Iterate over all anchored points in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &PointAnchor)> + '_ {
        self.anchors.iter().map(|(point, anchor)| (*point, anchor))
    }

    /// Return true when no anchors are recorded.
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Mark an existing anchored point as constrained/hanging.
    pub fn set_kind(&mut self, point: PointId, kind: AnchorKind) {
        if let Some(anchor) = self.anchors.get_mut(&point) {
            anchor.kind = kind;
        }
    }

    /// Parent points for a point, including the point itself when unanchored.
    pub fn parent_points(&self, point: PointId) -> Vec<PointId> {
        self.anchors
            .get(&point)
            .map(|anchor| anchor.parents.clone())
            .unwrap_or_else(|| vec![point])
    }

    /// Expand a point to itself plus its anchor parents.
    pub fn anchored_points(&self, point: PointId) -> BTreeSet<PointId> {
        let mut out = BTreeSet::new();
        out.insert(point);
        if let Some(anchor) = self.anchors.get(&point) {
            out.extend(anchor.parents.iter().copied());
        }
        out
    }

    /// Downward closure augmented with anchor parents for each visited point.
    pub fn anchor_aware_closure<S>(
        &self,
        sieve: &S,
        seeds: impl IntoIterator<Item = PointId>,
    ) -> BTreeSet<PointId>
    where
        S: Sieve<Point = PointId>,
    {
        let mut visited = BTreeSet::new();
        let mut queue: VecDeque<PointId> = seeds.into_iter().collect();
        while let Some(point) = queue.pop_front() {
            if !visited.insert(point) {
                continue;
            }
            for child in sieve.cone_points(point) {
                queue.push_back(child);
            }
            if let Some(anchor) = self.anchors.get(&point) {
                for parent in &anchor.parents {
                    queue.push_back(*parent);
                }
            }
        }
        visited
    }

    /// Upward star augmented by points anchored to any visited point.
    pub fn anchor_aware_star<S>(
        &self,
        sieve: &S,
        seeds: impl IntoIterator<Item = PointId>,
    ) -> BTreeSet<PointId>
    where
        S: Sieve<Point = PointId>,
    {
        let mut visited = BTreeSet::new();
        let mut queue: VecDeque<PointId> = seeds.into_iter().collect();
        while let Some(point) = queue.pop_front() {
            if !visited.insert(point) {
                continue;
            }
            for parent in sieve.support_points(point) {
                queue.push_back(parent);
            }
            for (child, anchor) in &self.anchors {
                if anchor.parents.contains(&point) {
                    queue.push_back(*child);
                }
            }
        }
        visited
    }
}

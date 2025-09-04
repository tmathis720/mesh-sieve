//! # Overlap graph (bipartite)
//!
//! The overlap is a **bipartite** directed graph over points [`OvlId`]:
//! `Local(PointId)` (mesh entities on this rank) and `Part(usize)` (partition nodes).
//!
//! **Invariants:**
//! - Edges are **only** `Local(_) -> Part(r)`.
//! - No `Local->Local` and no `Part->*` edges.
//! - For any edge `Local(p) -> Part(r)` with payload [`Remote`], `Remote.rank == r`.
//!
//! The [`Overlap`] newtype wraps an [`InMemorySieve`] and enforces these
//! invariants for all mutation routes (`add_arrow`, `set_cone`, etc.).
//! This design avoids ID collisions, keeps algorithms branch-light, and makes
//! later phases (closure completion, remote resolution) straightforward.

use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};

/// Overlap vertex identifier: either a local mesh point, or a partition node.
#[derive(
    Copy,
    Clone,
    Eq,
    PartialEq,
    Hash,
    Ord,
    PartialOrd,
    Debug,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum OvlId {
    /// Real mesh entity on this rank
    Local(PointId),
    /// Partition node representing MPI rank `r`
    Part(usize),
}

impl OvlId {
    #[inline]
    pub fn is_local(self) -> bool {
        matches!(self, OvlId::Local(_))
    }
    #[inline]
    pub fn is_part(self) -> bool {
        matches!(self, OvlId::Part(_))
    }

    #[inline]
    pub fn expect_local(self) -> PointId {
        match self {
            OvlId::Local(p) => p,
            _ => panic!("expected Local(_)")
        }
    }
    #[inline]
    pub fn expect_part(self) -> usize {
        match self {
            OvlId::Part(r) => r,
            _ => panic!("expected Part(_)")
        }
    }
}

#[inline]
pub fn local(p: PointId) -> OvlId {
    OvlId::Local(p)
}
#[inline]
pub fn part(r: usize) -> OvlId {
    OvlId::Part(r)
}

/// Remote: (rank, remote_point) is exactly SF leaf→root as in PETSc SF.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Remote {
    pub rank: usize,
    pub remote_point: PointId,
}

#[derive(Clone, Debug, Default)]
pub struct Overlap {
    inner: InMemorySieve<OvlId, Remote>,
}

impl Overlap {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Expose read-only view when absolutely needed (debug, tests).
    #[inline]
    pub fn as_inner(&self) -> &InMemorySieve<OvlId, Remote> {
        &self.inner
    }

    /// Partition vertex for rank `r`.
    #[inline]
    pub fn partition_node_id(rank: usize) -> OvlId {
        part(rank)
    }

    /// Add a typed link: Local(p) -> Part(r) with payload {rank:r, remote_point}.
    ///
    /// This is the primary constructor for edges and enforces invariants.
    pub fn add_link(&mut self, p: PointId, r: usize, remote: PointId) {
        self.add_arrow(local(p), part(r), Remote { rank: r, remote_point: remote });
    }

    /// Debug/feature-gated invariant checker.
    pub fn validate_invariants(&self) -> Result<(), MeshSieveError> {
        use MeshSieveError as E;
        for src in self.inner.base_points() {
            if !matches!(src, OvlId::Local(_)) {
                return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                    "Overlap invariant: base_points must be Local(_)"))) );
            }
            for (dst, rem) in self.inner.cone(src) {
                match (src, dst) {
                    (OvlId::Local(_), OvlId::Part(r)) => {
                        if rem.rank != r {
                            return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                                "Overlap invariant: payload.rank != Part(r)"))));
                        }
                    }
                    _ => {
                        return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                            "Overlap invariant: only Local->Part edges allowed"))));
                    }
                }
            }
        }
        for q in self.inner.cap_points() {
            if !matches!(q, OvlId::Part(_)) {
                return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                    "Overlap invariant: cap_points must be Part(_)"))));
            }
        }
        Ok(())
    }

    /// Iterator of distinct neighbor ranks present as partition nodes.
    pub fn neighbor_ranks(&self) -> impl Iterator<Item = usize> + '_ {
        use std::collections::BTreeSet;
        let ranks: BTreeSet<usize> = self
            .cap_points()
            .filter_map(|q| match q {
                OvlId::Part(r) => Some(r),
                _ => None,
            })
            .collect();
        ranks.into_iter()
    }

    /// Typed links to a neighbor rank (local -> remote_point).
    pub fn links_to(&self, nbr: usize) -> impl Iterator<Item = (PointId, PointId)> + '_ {
        self.support(part(nbr)).filter_map(move |(src, rem)| match src {
            OvlId::Local(p) if rem.rank == nbr => Some((p, rem.remote_point)),
            _ => None,
        })
    }
}

impl InvalidateCache for Overlap {
    #[inline]
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache();
    }
}

impl Sieve for Overlap {
    type Point = OvlId;
    type Payload = Remote;

    type ConeIter<'a> = <InMemorySieve<OvlId, Remote> as Sieve>::ConeIter<'a>
    where
        Self: 'a;
    type SupportIter<'a> = <InMemorySieve<OvlId, Remote> as Sieve>::SupportIter<'a>
    where
        Self: 'a;

    #[inline]
    fn cone<'a>(&'a self, p: OvlId) -> Self::ConeIter<'a> {
        self.inner.cone(p)
    }

    #[inline]
    fn support<'a>(&'a self, p: OvlId) -> Self::SupportIter<'a> {
        self.inner.support(p)
    }

    fn add_arrow(&mut self, src: OvlId, dst: OvlId, payload: Remote) {
        match (src, dst) {
            (OvlId::Local(_), OvlId::Part(r)) => {
                debug_assert_eq!(payload.rank, r, "payload.rank must match Part(r)");
                if payload.rank != r {
                    return;
                }
            }
            _ => {
                return;
            }
        }
        if self.inner.cone(src).any(|(q, _)| q == dst) {
            return;
        }
        self.inner.reserve_cone(src, 1);
        self.inner.reserve_support(dst, 1);
        self.inner.add_arrow(src, dst, payload);
    }

    #[inline]
    fn remove_arrow(&mut self, src: OvlId, dst: OvlId) -> Option<Remote> {
        self.inner.remove_arrow(src, dst)
    }

    #[inline]
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = OvlId> + 'a> {
        self.inner.base_points()
    }

    #[inline]
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = OvlId> + 'a> {
        self.inner.cap_points()
    }

    // default impls of set_cone/add_support/etc. will call our add_arrow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariants_hold_for_basic_links() {
        let mut ov = Overlap::new();
        let p1 = PointId::new(1).unwrap();
        let r1 = 3usize;
        ov.add_link(p1, r1, PointId::new(101).unwrap());

        for src in ov.base_points() {
            assert!(matches!(src, OvlId::Local(_)));
        }
        for dst in ov.cap_points() {
            assert!(matches!(dst, OvlId::Part(_)));
        }

        ov.validate_invariants().unwrap();
    }

    #[test]
    fn neighbor_ranks_and_links_to() {
        let mut ov = Overlap::new();
        ov.add_link(PointId::new(1).unwrap(), 1, PointId::new(101).unwrap());
        ov.add_link(PointId::new(2).unwrap(), 2, PointId::new(201).unwrap());
        let ranks: std::collections::BTreeSet<_> = ov.neighbor_ranks().collect();
        assert_eq!(ranks, [1usize, 2usize].into_iter().collect());

        let pairs: Vec<_> = ov.links_to(1).collect();
        assert!(
            pairs.contains(&(PointId::new(1).unwrap(), PointId::new(101).unwrap()))
        );
    }
}


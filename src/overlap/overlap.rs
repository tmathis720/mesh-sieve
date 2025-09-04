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
//!
//! ## Typical lifecycle
//! 1. Seed structure: for each shared local mesh point `p`, call
//!    [`Overlap::add_link_structural`].
//! 2. Complete structure: call [`ensure_closure_of_support`] with the mesh.
//! 3. Discover mapping via neighbor exchange (e.g., global IDs).
//! 4. Resolve: call [`Overlap::resolve_remote_point`] (or the batch variant)
//!    for each `(local, neighbor_rank, remote)` triple.
//! 5. Assert [`Overlap::is_fully_resolved`] before constructing on-wire buffers.

use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, Sieve};

/// Overlap vertex identifier: either a local mesh point, or a partition node.
#[derive(
    Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug, serde::Serialize, serde::Deserialize,
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
            _ => panic!("expected Local(_)"),
        }
    }
    #[inline]
    pub fn expect_part(self) -> usize {
        match self {
            OvlId::Part(r) => r,
            _ => panic!("expected Part(_)"),
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

/// Remote link: `rank` identifies the neighbor; `remote_point` is filled once
/// the mapping is known.  This mirrors PETSc’s SF: build structure first, then
/// resolve remote IDs after an exchange.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Remote {
    pub rank: usize,
    pub remote_point: Option<PointId>,
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

    /// Add a typed link: `Local(p) -> Part(r)` with known `remote` ID.
    ///
    /// This is the primary constructor for edges when the remote mapping is
    /// already known.  For structure-only insertion, use
    /// [`add_link_structural`].
    pub fn add_link(&mut self, p: PointId, r: usize, remote: PointId) {
        self.add_arrow(
            local(p),
            part(r),
            Remote {
                rank: r,
                remote_point: Some(remote),
            },
        );
    }

    /// Insert `Local(p) -> Part(r)` if missing; leaves `remote_point = None`.
    pub fn add_link_structural(&mut self, p: PointId, r: usize) {
        let src = local(p);
        let dst = part(r);
        if self.inner.cone(src).any(|(q, _)| q == dst) {
            return;
        }
        self.inner.reserve_cone(src, 1);
        self.inner.reserve_support(dst, 1);
        self.add_arrow(
            src,
            dst,
            Remote {
                rank: r,
                remote_point: None,
            },
        );
        // add_arrow already enforces Local->Part and rank equality
    }

    /// Resolve the remote ID for `(Local(local) -> Part(rank))`.
    ///
    /// Idempotent: resolving to the same ID twice is a no-op.  Conflicting
    /// resolutions return an error.
    pub fn resolve_remote_point(
        &mut self,
        local_pt: PointId,
        rank: usize,
        remote: PointId,
    ) -> Result<(), MeshSieveError> {
        use MeshSieveError::*;
        let src = local(local_pt);
        let dst = part(rank);

        if let Some(vec) = self.inner.adjacency_out.get_mut(&src) {
            if let Some((_, rem)) = vec.iter_mut().find(|(d, _)| *d == dst) {
                if rem.rank != rank {
                    return Err(OverlapRankMismatch {
                        expected: rank,
                        found: rem.rank,
                    });
                }
                match rem.remote_point {
                    None => {
                        rem.remote_point = Some(remote);
                        if let Some(vec_in) = self.inner.adjacency_in.get_mut(&dst) {
                            if let Some((_, rem_in)) = vec_in.iter_mut().find(|(s, _)| *s == src) {
                                rem_in.remote_point = Some(remote);
                            }
                        }
                        self.invalidate_cache();
                        return Ok(());
                    }
                    Some(existing) if existing == remote => return Ok(()),
                    Some(existing) => {
                        return Err(OverlapResolutionConflict {
                            local: local_pt,
                            rank,
                            existing: Some(existing),
                            new: remote,
                        });
                    }
                }
            }
        }
        Err(OverlapLinkMissing(local_pt, rank))
    }

    /// Resolve many `(local, rank, remote)` triples; stops at first error.
    pub fn resolve_remote_points<I>(&mut self, triples: I) -> Result<(), MeshSieveError>
    where
        I: IntoIterator<Item = (PointId, usize, PointId)>,
    {
        for (p, r, rp) in triples {
            self.resolve_remote_point(p, r, rp)?;
        }
        Ok(())
    }

    /// Debug/feature-gated invariant checker.
    pub fn validate_invariants(&self) -> Result<(), MeshSieveError> {
        use MeshSieveError as E;
        for src in self.inner.base_points() {
            if !matches!(src, OvlId::Local(_)) {
                return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                    "Overlap invariant: base_points must be Local(_)",
                ))));
            }
            for (dst, rem) in self.inner.cone(src) {
                match (src, dst) {
                    (OvlId::Local(_), OvlId::Part(r)) => {
                        if rem.rank != r {
                            return Err(MeshSieveError::OverlapRankMismatch {
                                expected: r,
                                found: rem.rank,
                            });
                        }
                        if let Some(rp) = rem.remote_point {
                            let _ = rp; // hook for external validators
                        }
                    }
                    _ => {
                        return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                            "Overlap invariant: only Local->Part edges allowed",
                        ))));
                    }
                }
            }
        }
        for q in self.inner.cap_points() {
            if !matches!(q, OvlId::Part(_)) {
                return Err(E::MeshError(Box::new(E::UnsupportedStackOperation(
                    "Overlap invariant: cap_points must be Part(_)",
                ))));
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

    /// Pairs to neighbor `r` (`local`, `Option<remote>`). Includes unresolved links.
    pub fn links_to_maybe(
        &self,
        r: usize,
    ) -> impl Iterator<Item = (PointId, Option<PointId>)> + '_ {
        self.support(part(r))
            .filter_map(move |(src, rem)| match src {
                OvlId::Local(p) if rem.rank == r => Some((p, rem.remote_point)),
                _ => None,
            })
    }

    /// Only resolved links (local, remote).
    pub fn links_to_resolved(&self, r: usize) -> impl Iterator<Item = (PointId, PointId)> + '_ {
        self.links_to_maybe(r)
            .filter_map(|(p, opt)| opt.map(|rp| (p, rp)))
    }

    /// Backward-compatible helper returning only resolved links.
    pub fn links_to(&self, r: usize) -> impl Iterator<Item = (PointId, PointId)> + '_ {
        self.links_to_resolved(r)
    }

    /// Count edges whose remote_point is still unresolved.
    pub fn unresolved_count(&self) -> usize {
        self.cap_points()
            .filter_map(|q| match q {
                OvlId::Part(r) => Some(r),
                _ => None,
            })
            .map(move |r| self.support(part(r)))
            .flatten()
            .filter(|(_, rem)| rem.remote_point.is_none())
            .count()
    }

    /// Returns true if all links have resolved remote IDs.
    #[inline]
    pub fn is_fully_resolved(&self) -> bool {
        self.unresolved_count() == 0
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

    type ConeIter<'a>
        = <InMemorySieve<OvlId, Remote> as Sieve>::ConeIter<'a>
    where
        Self: 'a;
    type SupportIter<'a>
        = <InMemorySieve<OvlId, Remote> as Sieve>::SupportIter<'a>
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

/// Ensure closure-of-support: for any `Local(p) -> Part(r)` edge already present,
/// also link every `q` in `mesh.closure({p})` to `Part(r)`. New links are
/// structural only (`remote_point = None`).
pub fn ensure_closure_of_support<M: Sieve<Point = PointId>>(ov: &mut Overlap, mesh: &M) {
    use std::collections::HashSet;

    // Gather seed pairs from existing edges
    let seeds: Vec<(PointId, usize)> = ov
        .base_points()
        .filter_map(|id| match id {
            OvlId::Local(p) => Some(p),
            _ => None,
        })
        .flat_map(|p| {
            ov.cone(local(p)).filter_map(move |(dst, rem)| match dst {
                OvlId::Part(r) => Some((p, r, rem.rank)),
                _ => None,
            })
        })
        .map(|(p, r, rr)| {
            debug_assert_eq!(r, rr);
            (p, r)
        })
        .collect();

    let mut seen: HashSet<(PointId, usize)> = seeds.iter().copied().collect();
    for (p, r) in seeds {
        for q in mesh.closure(std::iter::once(p)) {
            if seen.insert((q, r)) {
                ov.add_link_structural(q, r);
            }
        }
    }
    ov.invalidate_cache();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_error::MeshSieveError;
    use crate::topology::sieve::InMemorySieve as Mesh;

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
        assert!(pairs.contains(&(PointId::new(1).unwrap(), PointId::new(101).unwrap())));
    }

    #[test]
    fn structural_then_resolve() {
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        ov.add_link_structural(p, 1);
        let v_unresolved: Vec<_> = ov.links_to_maybe(1).collect();
        assert_eq!(v_unresolved, vec![(p, None)]);
        assert!(ov.links_to_resolved(1).next().is_none());

        ov.resolve_remote_point(p, 1, PointId::new(10).unwrap())
            .unwrap();
        let v_resolved: Vec<_> = ov.links_to_resolved(1).collect();
        assert_eq!(v_resolved, vec![(p, PointId::new(10).unwrap())]);
    }

    #[test]
    fn resolve_idempotent_and_conflict() {
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        ov.add_link_structural(p, 1);
        ov.resolve_remote_point(p, 1, PointId::new(10).unwrap())
            .unwrap();
        // Idempotent
        ov.resolve_remote_point(p, 1, PointId::new(10).unwrap())
            .unwrap();
        // Conflict
        let err = ov
            .resolve_remote_point(p, 1, PointId::new(11).unwrap())
            .unwrap_err();
        assert!(matches!(
            err,
            MeshSieveError::OverlapResolutionConflict { .. }
        ));
    }

    #[test]
    fn resolve_missing_link() {
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        let err = ov
            .resolve_remote_point(p, 1, PointId::new(10).unwrap())
            .unwrap_err();
        assert!(matches!(err, MeshSieveError::OverlapLinkMissing(_, _)));
    }

    #[test]
    fn closure_adds_structural_links() {
        // mesh: 1 -> 2
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
        let mut ov = Overlap::new();
        ov.add_link_structural(PointId::new(1).unwrap(), 7);
        ensure_closure_of_support(&mut ov, &mesh);
        let v: Vec<_> = ov.links_to_maybe(7).collect();
        assert_eq!(v.len(), 2);
        assert!(v.contains(&(PointId::new(1).unwrap(), None)));
        assert!(v.contains(&(PointId::new(2).unwrap(), None)));
    }
}

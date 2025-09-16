//! # Overlap graph (bipartite)
//!
//! The overlap is a **bipartite** directed graph over points [`OvlId`]:
//! `Local(PointId)` (mesh entities on this rank) and `Part(usize)` (partition nodes).
//!
//! **Invariants:**
//! 1. Bipartite direction: every edge is `Local(_) -> Part(r)`.
//!    No `Local->Local` and no `Part->*` edges.
//! 2. Rank consistency: for any edge `Local(p) -> Part(r)` with payload [`Remote`],
//!    `Remote.rank == r`.
//! 3. Point-set partitioning:
//!    - All `base_points()` are `Local(_)`.
//!    - All `cap_points()` are `Part(_)`.
//! 4. No duplicate edges: per `(src, dst)` there is at most one edge.
//! 5. *(feature `check-empty-part`)* No dangling `Part` nodes: every `Part(r)`
//!    must have at least one incoming edge.
//!
//! The invariants checker runs in debug builds or when the feature
//! `strict-invariants` (or its alias `check-invariants`) is enabled. It performs
//! an `O(V + E)` walk over the graph without cloning payloads. Basic invariants
//! (bipartite edges, rank consistency, duplicate detection, etc.) are always
//! verified; with `strict-invariants` enabled it also checks that the
//! `adjacency_out` and `adjacency_in` maps mirror each other exactly
//! (payloads and counts), an additional `O(E)` cross-check guarding against
//! storage regressions.
//!
//! The [`Overlap`] newtype wraps an [`InMemorySieve`] and enforces these
//! invariants for all mutation routes (`add_arrow`, `set_cone`, etc.).
//! This design avoids ID collisions, keeps algorithms branch-light, and makes
//! later phases (closure completion, remote resolution) straightforward.
//!
//! ## Typical lifecycle
//! 1. Seed structure: for each shared local mesh point `p`, call
//!    [`Overlap::try_add_link_structural_one`].
//! 2. Complete structure: call [`ensure_closure_of_support`] with the mesh.
//! 3. Discover mapping via neighbor exchange (e.g., global IDs).
//! 4. Resolve: call [`Overlap::resolve_remote_point`] (or the batch variant)
//!    for each `(local, neighbor_rank, remote)` triple.
//! 5. Assert [`Overlap::is_fully_resolved`] before constructing on-wire buffers.

//! ## Mesh-driven overlap completion
//!
//! Overlap structure must mirror the mesh topology. If a local mesh point `p`
//! is shared with neighbor rank `r`, then every `q ∈ closure_mesh(p)` must also
//! be shared with `r`. The helper [`ensure_closure_of_support`] expands the
//! overlap using **mesh** incidence (cones/closure), keeping structural links
//! only (no fabricated remote IDs). Mappings can be filled later via
//! [`Overlap::resolve_remote_point`].

//! ## Neighbor and link queries
//! - [`neighbor_ranks`](Overlap::neighbor_ranks) returns a **sorted** set of
//!   partition nodes present in this overlap.
//! - [`links_to`](Overlap::links_to) yields `(local, Option<remote>)` for all
//!   local entities shared with rank `r`. The `remote` is `None` until resolved
//!   via [`resolve_remote_point`](Overlap::resolve_remote_point).
//! - Use [`links_to_resolved`](Overlap::links_to_resolved) or the `_sorted`
//!   variants when constructing on-wire layouts.
//! - No `my_rank` parameter is needed; neighbors are explicitly modeled as
//!   `Part(r)` vertices in the graph.
//!
//! ## Accessors
//! Prefer [`OvlId::try_local`] / [`OvlId::try_part`] when you need contextual
//! errors, or [`OvlId::as_local`] / [`OvlId::as_part`] for optional probing.
//! The legacy [`OvlId::expect_local`] / [`OvlId::expect_part`] helpers are
//! deprecated and will be removed in a future major release.

use crate::mesh_error::MeshSieveError;
use crate::overlap::perf::{FastMap, FastSet};
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

    /// Returns the wrapped [`PointId`] when `self` is [`OvlId::Local`].
    #[inline]
    pub fn as_local(self) -> Option<PointId> {
        match self {
            OvlId::Local(p) => Some(p),
            _ => None,
        }
    }

    /// Returns the wrapped rank when `self` is [`OvlId::Part`].
    #[inline]
    pub fn as_part(self) -> Option<usize> {
        match self {
            OvlId::Part(r) => Some(r),
            _ => None,
        }
    }

    /// Returns the wrapped [`PointId`] or a contextual error when `self` is not [`OvlId::Local`].
    #[inline]
    pub fn try_local(self) -> Result<PointId, MeshSieveError> {
        self.as_local()
            .ok_or(MeshSieveError::OverlapExpectedLocal { found: self })
    }

    /// Returns the wrapped rank or a contextual error when `self` is not [`OvlId::Part`].
    #[inline]
    pub fn try_part(self) -> Result<usize, MeshSieveError> {
        self.as_part()
            .ok_or(MeshSieveError::OverlapExpectedPart { found: self })
    }

    #[deprecated(
        since = "2.0.0",
        note = "Use OvlId::try_local() or OvlId::as_local() instead"
    )]
    #[inline]
    pub fn expect_local(self) -> PointId {
        self.try_local()
            .unwrap_or_else(|err| panic!("OvlId::expect_local failed: {err}"))
    }

    #[deprecated(
        since = "2.0.0",
        note = "Use OvlId::try_part() or OvlId::as_part() instead"
    )]
    #[inline]
    pub fn expect_part(self) -> usize {
        self.try_part()
            .unwrap_or_else(|err| panic!("OvlId::expect_part failed: {err}"))
    }
}

impl std::convert::TryFrom<OvlId> for PointId {
    type Error = MeshSieveError;

    #[inline]
    fn try_from(id: OvlId) -> Result<Self, Self::Error> {
        id.try_local()
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
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
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

    /// Try to add a typed link `Local(p) -> Part(r)` with known `remote` ID.
    ///
    /// Structure is inserted if missing, and the remote point is resolved.
    /// Idempotent for the same remote ID; conflicting resolutions error.
    pub fn try_add_link(
        &mut self,
        p: PointId,
        r: usize,
        remote: PointId,
    ) -> Result<(), MeshSieveError> {
        // Ensure structural edge exists
        let _ = self.try_add_structural_edge(p, r)?;
        // Resolve / confirm remote ID
        self.resolve_remote_point(p, r, remote)?;
        Ok(())
    }

    /// Back-compat infallible wrapper. Panics in debug/strict mode on error.
    #[deprecated(note = "Use try_add_link; will be removed in a future major")]
    pub fn add_link(&mut self, p: PointId, r: usize, remote: PointId) {
        if let Err(e) = self.try_add_link(p, r, remote) {
            #[cfg(any(
                debug_assertions,
                feature = "strict-invariants",
                feature = "check-invariants",
            ))]
            panic!("add_link({p:?},{r},{remote:?}) failed: {e}");
        }
        self.debug_validate();
    }

    #[inline]
    fn existing_pairs(&self) -> impl Iterator<Item = (PointId, usize)> + '_ {
        self.inner
            .adjacency_out
            .iter()
            .filter_map(|(src, outs)| match *src {
                OvlId::Local(p) => Some((p, outs)),
                _ => None,
            })
            .flat_map(|(p, outs)| {
                outs.iter().filter_map(move |(dst, rem)| match dst {
                    OvlId::Part(r) => {
                        debug_assert_eq!(rem.rank, *r);
                        Some((p, *r))
                    }
                    _ => None,
                })
            })
    }

    /// Try to insert `Local(p) -> Part(r)` structurally (`remote_point = None`).
    /// Returns `Ok(true)` if inserted, `Ok(false)` if it already existed.
    fn try_add_structural_edge(&mut self, p: PointId, r: usize) -> Result<bool, MeshSieveError> {
        let src = local(p);
        let dst = part(r);

        // Duplicate?
        let exists = self
            .inner
            .adjacency_out
            .get(&src)
            .is_some_and(|outs| outs.iter().any(|(q, _)| *q == dst));
        if exists {
            return Ok(false);
        }

        // Reserve and insert
        self.inner.reserve_cone(src, 1);
        self.inner.reserve_support(dst, 1);
        self.inner.add_arrow(
            src,
            dst,
            Remote {
                rank: r,
                remote_point: None,
            },
        );

        #[cfg(any(
            debug_assertions,
            feature = "strict-invariants",
            feature = "check-invariants",
        ))]
        self.debug_validate();

        Ok(true)
    }

    /// Fallible structural insert. Returns `Ok(true)` if inserted, `Ok(false)` if duplicate.
    pub fn try_add_link_structural_one(
        &mut self,
        p: PointId,
        r: usize,
    ) -> Result<bool, MeshSieveError> {
        self.try_add_structural_edge(p, r)
    }

    /// Existing infallible API (deprecated). Internally uses the fallible path.
    #[deprecated(note = "Use try_add_link_structural_one; will be removed in a future major")]
    pub fn add_link_structural_one(&mut self, p: PointId, r: usize) -> bool {
        match self.try_add_structural_edge(p, r) {
            Ok(inserted) => inserted,
            Err(e) => {
                #[cfg(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                ))]
                panic!("add_link_structural_one({p:?},{r}) failed: {e}");
                #[cfg(not(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                )))]
                {
                    false
                }
            }
        }
    }

    /// Fallible bulk structural insert; returns number of **new** edges inserted.
    ///
    /// Deterministic: edges are inserted in lexicographic order of `(PointId, rank)`
    /// regardless of `Hash{Map,Set}` or feature-driven iteration order.
    pub fn try_add_links_structural_bulk<I>(&mut self, edges: I) -> Result<usize, MeshSieveError>
    where
        I: IntoIterator<Item = (PointId, usize)>,
    {
        let mut to_add: Vec<(OvlId, OvlId)> = Vec::new();
        let mut need_cone: FastMap<OvlId, usize> = FastMap::default();
        let mut need_support: FastMap<OvlId, usize> = FastMap::default();
        let mut seen_batch: FastSet<(OvlId, OvlId)> = FastSet::default();

        for (p, r) in edges {
            let src = local(p);
            let dst = part(r);
            let key = (src, dst);

            if !seen_batch.insert(key) {
                continue;
            }
            let exists = self
                .inner
                .adjacency_out
                .get(&src)
                .is_some_and(|outs| outs.iter().any(|(q, _)| *q == dst));
            if exists {
                continue;
            }

            *need_cone.entry(src).or_insert(0) += 1;
            *need_support.entry(dst).or_insert(0) += 1;
            to_add.push(key);
        }

        if to_add.is_empty() {
            return Ok(0);
        }

        // Deterministic order
        to_add.sort_unstable_by_key(|(src, dst)| {
            let p = match *src {
                OvlId::Local(p) => p,
                _ => unreachable!(),
            };
            let r = match *dst {
                OvlId::Part(r) => r,
                _ => unreachable!(),
            };
            (p, r)
        });

        for (src, k) in need_cone {
            self.inner.reserve_cone(src, k);
        }
        for (dst, k) in need_support {
            self.inner.reserve_support(dst, k);
        }

        for (src, dst) in to_add.iter().copied() {
            let r = match dst {
                OvlId::Part(r) => r,
                _ => unreachable!(),
            };
            self.inner.add_arrow(
                src,
                dst,
                Remote {
                    rank: r,
                    remote_point: None,
                },
            );
        }

        self.invalidate_cache();
        self.debug_validate();

        Ok(to_add.len())
    }

    /// Deprecated infallible bulk wrapper.
    ///
    /// Deterministic: edges are inserted in lexicographic order of
    /// `(PointId, rank)`.
    #[deprecated(note = "Use try_add_links_structural_bulk")]
    pub fn add_links_structural_bulk<I>(&mut self, edges: I) -> usize
    where
        I: IntoIterator<Item = (PointId, usize)>,
    {
        match self.try_add_links_structural_bulk(edges) {
            Ok(n) => n,
            Err(e) => {
                #[cfg(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                ))]
                panic!("add_links_structural_bulk failed: {e}");
                #[cfg(not(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                )))]
                {
                    0
                }
            }
        }
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

        if let Some(vec) = self.inner.adjacency_out.get_mut(&src)
            && let Some((_, rem)) = vec.iter_mut().find(|(d, _)| *d == dst)
        {
            if rem.rank != rank {
                return Err(OverlapRankMismatch {
                    expected: rank,
                    found: rem.rank,
                });
            }
            match rem.remote_point {
                None => {
                    rem.remote_point = Some(remote);
                    if let Some(vec_in) = self.inner.adjacency_in.get_mut(&dst)
                        && let Some((_, rem_in)) = vec_in.iter_mut().find(|(s, _)| *s == src)
                    {
                        rem_in.remote_point = Some(remote);
                    }
                    self.invalidate_cache();
                    self.debug_validate();
                    return Ok(());
                }
                Some(existing) if existing == remote => {
                    self.debug_validate();
                    return Ok(());
                }
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
        self.debug_validate();
        Ok(())
    }

    /// Validate overlap invariants.
    ///
    /// Always checks bipartite structure, rank consistency, base/cap partitioning,
    /// and duplicate edges. With feature `strict-invariants` (alias
    /// `check-invariants`) enabled, it additionally cross-checks that every
    /// `adjacency_out` edge has a mirrored entry in `adjacency_in` with identical
    /// payload and that edge counts match. The strict check is `O(E)` and compiled
    /// out otherwise.
    pub fn validate_invariants(&self) -> Result<(), MeshSieveError> {
        use MeshSieveError as E;
        use std::collections::HashSet;

        for src in self.inner.adjacency_out.keys() {
            if let OvlId::Part(_) = src {
                return Err(E::OverlapPartInBasePoints);
            }
        }
        for dst in self.inner.adjacency_in.keys() {
            if let OvlId::Local(_) = dst {
                return Err(E::OverlapLocalInCapPoints);
            }
        }

        for (src, outs) in &self.inner.adjacency_out {
            let mut seen_dst: HashSet<OvlId> = HashSet::with_capacity(outs.len());
            for (dst, rem) in outs {
                match (src, dst) {
                    (OvlId::Local(_), OvlId::Part(r)) => {
                        if rem.rank != *r {
                            return Err(E::OverlapRankMismatch {
                                expected: *r,
                                found: rem.rank,
                            });
                        }
                    }
                    _ => {
                        return Err(E::OverlapNonBipartite {
                            src: *src,
                            dst: *dst,
                        });
                    }
                }
                if !seen_dst.insert(*dst) {
                    return Err(E::OverlapDuplicateEdge {
                        src: *src,
                        dst: *dst,
                    });
                }
            }
        }

        #[cfg(feature = "check-empty-part")]
        for dst in self.inner.adjacency_in.keys() {
            if let OvlId::Part(r) = dst {
                if self
                    .inner
                    .adjacency_in
                    .get(dst)
                    .is_none_or(|v| v.is_empty())
                {
                    return Err(E::OverlapEmptyPart { rank: *r });
                }
            }
        }

        #[cfg(any(feature = "strict-invariants", feature = "check-invariants"))]
        {
            let mut out_map: FastMap<(OvlId, OvlId), &Remote> = FastMap::default();
            let mut out_count: usize = 0;
            for (src, outs) in &self.inner.adjacency_out {
                for (dst, rem) in outs {
                    out_map.insert((*src, *dst), rem);
                    out_count += 1;
                }
            }

            let mut in_count: usize = 0;
            for (dst, ins) in &self.inner.adjacency_in {
                for (src, rem_in) in ins {
                    in_count += 1;
                    match out_map.get(&(*src, *dst)) {
                        None => {
                            return Err(E::OverlapInOutMirrorMissing {
                                src: *src,
                                dst: *dst,
                            });
                        }
                        Some(rem_out) => {
                            if **rem_out != *rem_in {
                                return Err(E::OverlapInOutPayloadMismatch {
                                    src: *src,
                                    dst: *dst,
                                    out: **rem_out,
                                    inn: *rem_in,
                                });
                            }
                        }
                    }
                }
            }

            if out_count != in_count {
                return Err(E::OverlapInOutEdgeCountMismatch {
                    out_edges: out_count,
                    in_edges: in_count,
                });
            }
        }

        Ok(())
    }

    /// Panics on invariant violations in debug or when feature `strict-invariants` (alias `check-invariants`) is enabled.
    #[inline]
    pub fn debug_validate(&self) {
        #[cfg(any(
            debug_assertions,
            feature = "strict-invariants",
            feature = "check-invariants"
        ))]
        if let Err(e) = self.validate_invariants() {
            panic!("Overlap invariant violated: {e}");
        }
    }

    /// Iterator of distinct neighbor ranks present as partition nodes.
    #[inline]
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
    #[inline]
    pub fn links_to(&self, r: usize) -> impl Iterator<Item = (PointId, Option<PointId>)> + '_ {
        self.support(part(r))
            .filter_map(move |(src, rem)| match src {
                OvlId::Local(p) if rem.rank == r => Some((p, rem.remote_point)),
                _ => None,
            })
    }

    /// Only resolved links (local, remote).
    #[inline]
    pub fn links_to_resolved(&self, r: usize) -> impl Iterator<Item = (PointId, PointId)> + '_ {
        self.links_to(r)
            .filter_map(|(p, opt)| opt.map(|rp| (p, rp)))
    }

    /// Deterministic (sorted by local `PointId`) list of links to `r` (may be unresolved).
    pub fn links_to_sorted(&self, r: usize) -> Vec<(PointId, Option<PointId>)> {
        let mut v: Vec<_> = self.links_to(r).collect();
        v.sort_unstable_by_key(|(p, _)| *p);
        v
    }

    /// Deterministic (sorted) list of resolved links to `r`.
    pub fn links_to_resolved_sorted(&self, r: usize) -> Vec<(PointId, PointId)> {
        let mut v: Vec<_> = self.links_to_resolved(r).collect();
        v.sort_unstable_by_key(|(p, _)| *p);
        v
    }

    /// Count edges whose remote_point is still unresolved.
    pub fn unresolved_count(&self) -> usize {
        self.neighbor_ranks()
            .map(|r| self.unresolved_count_to(r))
            .sum()
    }

    /// Count unresolved links to a specific rank `r`.
    pub fn unresolved_count_to(&self, r: usize) -> usize {
        self.links_to(r).filter(|(_, rp)| rp.is_none()).count()
    }

    /// Returns true if all links have resolved remote IDs.
    #[inline]
    pub fn is_fully_resolved(&self) -> bool {
        self.unresolved_count() == 0
    }

    /// Legacy name for [`links_to`].
    #[deprecated(note = "renamed to links_to")]
    pub fn links_to_maybe(
        &self,
        r: usize,
    ) -> impl Iterator<Item = (PointId, Option<PointId>)> + '_ {
        self.links_to(r)
    }

    /// Legacy wrapper that accepted a `my_rank` parameter.
    #[deprecated(note = "my_rank parameter is unused; call neighbor_ranks()")]
    pub fn neighbor_ranks_legacy(&self, _my_rank: usize) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_ranks()
    }

    /// Legacy wrapper that accepted a `my_rank` parameter.
    #[deprecated(note = "my_rank parameter is unused; call links_to(r)")]
    pub fn links_to_legacy(
        &self,
        nbr: usize,
        _my_rank: usize,
    ) -> impl Iterator<Item = (PointId, Option<PointId>)> + '_ {
        self.links_to(nbr)
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
                #[cfg(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                ))]
                assert_eq!(
                    payload.rank, r,
                    "Overlap::add_arrow payload.rank {} must equal Part({})",
                    payload.rank, r
                );
                if payload.rank != r {
                    return;
                }
            }
            _ => {
                #[cfg(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                ))]
                panic!("Overlap::add_arrow non-bipartite edge: {src:?} -> {dst:?}");
                #[cfg(not(any(
                    debug_assertions,
                    feature = "strict-invariants",
                    feature = "check-invariants",
                )))]
                {
                    return;
                }
            }
        }
        if self
            .inner
            .adjacency_out
            .get(&src)
            .is_some_and(|v| v.iter().any(|(q, _)| *q == dst))
        {
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

#[macro_export]
/// Validate overlap invariants in debug builds or when `strict-invariants`
/// (alias `check-invariants`) is enabled. In strict mode this also ensures the
/// in/out adjacency mirrors remain in sync.
macro_rules! ovl_debug_validate {
    ($ovl:expr) => {
        if cfg!(any(
            debug_assertions,
            feature = "strict-invariants",
            feature = "check-invariants"
        )) {
            if let Err(e) = ($ovl).validate_invariants() {
                panic!("Overlap invariant violated: {e}");
            }
        }
    };
}

/// One-hop upward support of `seeds`, deduped and returned in deterministic order.
fn support1<M: Sieve<Point = PointId>>(
    mesh: &M,
    seeds: impl IntoIterator<Item = PointId>,
) -> Vec<PointId> {
    use std::collections::BTreeSet;
    let mut set = BTreeSet::new();
    for p in seeds {
        for (q, _) in mesh.support(p) {
            set.insert(q);
        }
    }
    set.into_iter().collect()
}

/// Expand structurally by one-hop star → full closure toward `neighbor`.
///
/// For each seed `p`, walks one level up (`support`) to gather immediate
/// neighbors, then adds all points in the full `closure` of those neighbors to
/// the overlap as structural links (`remote_point = None`). Existing links are
/// left untouched. Returns the number of **new** edges inserted. Deterministic
/// order is preserved for reproducible behavior.
pub fn expand_one_layer_mesh<M: Sieve<Point = PointId>>(
    ov: &mut Overlap,
    mesh: &M,
    seeds: impl IntoIterator<Item = PointId>,
    neighbor: usize,
) -> usize {
    use std::collections::BTreeSet;

    let mut to_add: BTreeSet<PointId> = BTreeSet::new();
    for s in support1(mesh, seeds) {
        for q in mesh.closure(std::iter::once(s)) {
            to_add.insert(q);
        }
    }

    let res = ov.try_add_links_structural_bulk(to_add.into_iter().map(|q| (q, neighbor)));
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants",
    ))]
    let added = res.expect("expand_one_layer_mesh failed");
    #[cfg(not(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants",
    )))]
    let added = res.unwrap_or(0);
    ov.debug_validate();
    added
}

/// Ensure closure-of-support: for any `Local(p) -> Part(r)` edge already present,
/// also link every `q` in `mesh.closure({p})` to `Part(r)`. New links are
/// structural only (`remote_point = None`).
///
/// Collected edges are sorted before bulk insertion so the resulting
/// adjacency order is deterministic.
pub fn ensure_closure_of_support<M: Sieve<Point = PointId>>(ov: &mut Overlap, mesh: &M) {
    let mut already: FastSet<(PointId, usize)> = FastSet::default();
    for (p, r) in ov.existing_pairs() {
        already.insert((p, r));
    }

    let mut new_edges: FastSet<(PointId, usize)> = FastSet::default();
    for (p, r) in already.iter().copied() {
        for q in mesh.closure_iter(std::iter::once(p)) {
            if already.contains(&(q, r)) {
                continue;
            }
            new_edges.insert((q, r));
        }
    }

    let mut edges: Vec<(PointId, usize)> = new_edges.into_iter().collect();
    edges.sort_unstable();
    let res = ov.try_add_links_structural_bulk(edges);
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants",
    ))]
    res.expect("ensure_closure_of_support failed");
    ov.debug_validate();
}

/// Incremental variant: expand closure only from explicit `(point, rank)` seeds.
///
/// Useful when a subset of links was newly added and only those need
/// propagation. Idempotent and safe to call multiple times.
///
/// Collected edges are sorted before bulk insertion so the resulting
/// adjacency order is deterministic.
pub fn ensure_closure_of_support_from_seeds<M: Sieve<Point = PointId>, I>(
    ov: &mut Overlap,
    mesh: &M,
    seeds: I,
) where
    I: IntoIterator<Item = (PointId, usize)>,
{
    let mut already: FastSet<(PointId, usize)> = FastSet::default();
    for (p, r) in ov.existing_pairs() {
        already.insert((p, r));
    }

    let mut new_edges: FastSet<(PointId, usize)> = FastSet::default();
    for (p, r) in seeds {
        for q in mesh.closure_iter(std::iter::once(p)) {
            if already.insert((q, r)) {
                new_edges.insert((q, r));
            }
        }
    }

    let mut edges: Vec<(PointId, usize)> = new_edges.into_iter().collect();
    edges.sort_unstable();
    let res = ov.try_add_links_structural_bulk(edges);
    #[cfg(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants",
    ))]
    res.expect("ensure_closure_of_support_from_seeds failed");
    ov.debug_validate();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_error::MeshSieveError;
    use crate::topology::sieve::InMemorySieve as Mesh;
    use std::convert::TryFrom;

    fn insert_raw_edge(ov: &mut Overlap, src: OvlId, dst: OvlId, rem: Remote) {
        ov.inner
            .adjacency_out
            .entry(src)
            .or_default()
            .push((dst, rem));
        ov.inner
            .adjacency_in
            .entry(dst)
            .or_default()
            .push((src, rem));
    }

    #[test]
    fn ovlid_as_and_try_helpers() {
        let pid = PointId::new(7).unwrap();
        let rank = 3usize;
        let local_id = local(pid);
        let part_id = part(rank);

        assert_eq!(local_id.as_local(), Some(pid));
        assert_eq!(local_id.as_part(), None);
        assert_eq!(part_id.as_part(), Some(rank));
        assert_eq!(part_id.as_local(), None);

        assert_eq!(local_id.try_local().unwrap(), pid);
        assert!(matches!(
            local_id.try_part(),
            Err(MeshSieveError::OverlapExpectedPart { found }) if found == local_id
        ));
        assert_eq!(part_id.try_part().unwrap(), rank);
        assert!(matches!(
            part_id.try_local(),
            Err(MeshSieveError::OverlapExpectedLocal { found }) if found == part_id
        ));
    }

    #[test]
    fn ovlid_try_from_pointid() {
        let pid = PointId::new(1).unwrap();
        let local_id = local(pid);
        let part_id = part(0);

        assert_eq!(PointId::try_from(local_id).unwrap(), pid);
        assert!(PointId::try_from(part_id).is_err());
    }

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
        ov.add_link(PointId::new(1).unwrap(), 2, PointId::new(201).unwrap());
        ov.add_link(PointId::new(2).unwrap(), 1, PointId::new(101).unwrap());
        let ranks: Vec<_> = ov.neighbor_ranks().collect();
        assert_eq!(ranks, vec![1usize, 2usize]);

        let pairs: Vec<_> = ov.links_to_resolved(1).collect();
        assert!(pairs.contains(&(PointId::new(2).unwrap(), PointId::new(101).unwrap())));
    }

    #[test]
    fn try_structural_dedup_single() {
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        let r = 1usize;
        assert_eq!(ov.try_add_link_structural_one(p, r).unwrap(), true);
        assert_eq!(ov.try_add_link_structural_one(p, r).unwrap(), false);
        assert_eq!(ov.links_to(r).count(), 1);
        ov.validate_invariants().unwrap();
    }

    #[test]
    fn bulk_dedup() {
        let mut ov = Overlap::new();
        let p1 = PointId::new(1).unwrap();
        let p2 = PointId::new(2).unwrap();
        let r = 3usize;
        let added = ov
            .try_add_links_structural_bulk(vec![(p1, r), (p1, r), (p2, r)])
            .unwrap();
        assert_eq!(added, 2);
        let added2 = ov
            .try_add_links_structural_bulk(vec![(p1, r), (p2, r)])
            .unwrap();
        assert_eq!(added2, 0);
        let order: Vec<_> = ov.links_to_sorted(r).into_iter().map(|(p, _)| p).collect();
        assert_eq!(order, vec![p1, p2]);
        ov.validate_invariants().unwrap();
    }

    #[test]
    fn try_add_link_idempotent_and_conflict() {
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        let r = 1usize;
        assert_eq!(ov.try_add_link_structural_one(p, r).unwrap(), true);
        ov.try_add_link(p, r, PointId::new(10).unwrap()).unwrap();
        ov.try_add_link(p, r, PointId::new(10).unwrap()).unwrap();
        let err = ov
            .try_add_link(p, r, PointId::new(11).unwrap())
            .unwrap_err();
        assert!(matches!(
            err,
            MeshSieveError::OverlapResolutionConflict { .. }
        ));
    }

    #[test]
    fn structural_then_resolve() {
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        ov.try_add_link_structural_one(p, 1).unwrap();
        let v_unresolved: Vec<_> = ov.links_to(1).collect();
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
        ov.try_add_link_structural_one(p, 1).unwrap();
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
    fn basic_closure_completion() {
        // cell -> faces -> vertices
        let cell = PointId::new(1).unwrap();
        let f0 = PointId::new(10).unwrap();
        let f1 = PointId::new(11).unwrap();
        let v0 = PointId::new(100).unwrap();
        let v1 = PointId::new(101).unwrap();
        let v2 = PointId::new(110).unwrap();
        let v3 = PointId::new(111).unwrap();
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(cell, f0, ());
        mesh.add_arrow(cell, f1, ());
        mesh.add_arrow(f0, v0, ());
        mesh.add_arrow(f0, v1, ());
        mesh.add_arrow(f1, v2, ());
        mesh.add_arrow(f1, v3, ());

        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(cell, 7).unwrap();
        ensure_closure_of_support(&mut ov, &mesh);

        let links: Vec<_> = ov.links_to(7).collect();
        assert_eq!(links.len(), 7);
        for p in [cell, f0, f1, v0, v1, v2, v3] {
            assert!(links.contains(&(p, None)));
        }
    }

    #[test]
    fn closure_idempotent() {
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(PointId::new(1).unwrap(), 7)
            .unwrap();
        ensure_closure_of_support(&mut ov, &mesh);
        let count = ov.links_to(7).count();
        ensure_closure_of_support(&mut ov, &mesh);
        assert_eq!(ov.links_to(7).count(), count);
    }

    #[test]
    fn closure_multiple_ranks() {
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(PointId::new(1).unwrap(), 1)
            .unwrap();
        ov.try_add_link_structural_one(PointId::new(1).unwrap(), 2)
            .unwrap();
        ensure_closure_of_support(&mut ov, &mesh);

        let set1: std::collections::HashSet<_> = ov.links_to(1).collect();
        let set2: std::collections::HashSet<_> = ov.links_to(2).collect();
        let expected: std::collections::HashSet<_> =
            [PointId::new(1).unwrap(), PointId::new(2).unwrap()]
                .into_iter()
                .map(|p| (p, None))
                .collect();
        assert_eq!(set1, expected);
        assert_eq!(set2, expected);
    }

    #[test]
    fn closure_preserves_resolved_edges() {
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
        let mut ov = Overlap::new();
        let p = PointId::new(1).unwrap();
        let r = 5usize;
        ov.add_link(p, r, PointId::new(101).unwrap());
        ensure_closure_of_support(&mut ov, &mesh);
        let links: Vec<_> = ov.links_to(r).collect();
        assert!(links.contains(&(p, Some(PointId::new(101).unwrap()))));
        assert!(links.contains(&(PointId::new(2).unwrap(), None)));
    }

    #[test]
    fn closure_from_seeds() {
        // mesh: 1 -> 2, 3 -> 4
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
        mesh.add_arrow(PointId::new(3).unwrap(), PointId::new(4).unwrap(), ());
        let r = 9usize;

        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(PointId::new(1).unwrap(), r)
            .unwrap();
        ov.try_add_link_structural_one(PointId::new(3).unwrap(), r)
            .unwrap();

        ensure_closure_of_support_from_seeds(&mut ov, &mesh, [(PointId::new(1).unwrap(), r)]);
        let links: std::collections::HashSet<_> = ov.links_to(r).collect();
        assert!(links.contains(&(PointId::new(1).unwrap(), None)));
        assert!(links.contains(&(PointId::new(2).unwrap(), None)));
        assert!(links.contains(&(PointId::new(3).unwrap(), None)));
        assert!(!links.contains(&(PointId::new(4).unwrap(), None)));

        ensure_closure_of_support_from_seeds(&mut ov, &mesh, [(PointId::new(3).unwrap(), r)]);
        let links2: std::collections::HashSet<_> = ov.links_to(r).collect();
        assert!(links2.contains(&(PointId::new(4).unwrap(), None)));
    }

    #[test]
    fn expand_one_layer_one_hop() {
        // cell -> edges -> vertices
        let cell = PointId::new(1).unwrap();
        let e0 = PointId::new(10).unwrap();
        let e1 = PointId::new(11).unwrap();
        let v0 = PointId::new(100).unwrap();
        let v1 = PointId::new(101).unwrap();
        let v2 = PointId::new(102).unwrap();
        let mut mesh = Mesh::<PointId, ()>::default();
        mesh.add_arrow(cell, e0, ());
        mesh.add_arrow(cell, e1, ());
        mesh.add_arrow(e0, v0, ());
        mesh.add_arrow(e0, v1, ());
        mesh.add_arrow(e1, v1, ());
        mesh.add_arrow(e1, v2, ());

        let neighbor = 7usize;
        let seed = v1;
        let mut ov = Overlap::new();

        let added = expand_one_layer_mesh(&mut ov, &mesh, [seed], neighbor);
        assert_eq!(added, 5);

        let links: std::collections::BTreeSet<_> = ov.links_to(neighbor).collect();
        let expected: std::collections::BTreeSet<_> =
            [(e0, None), (e1, None), (v0, None), (v1, None), (v2, None)]
                .into_iter()
                .collect();
        assert_eq!(links, expected);
        assert!(!links.contains(&(cell, None))); // one-hop: cell not added

        // Only specified neighbor present
        let ranks: Vec<_> = ov.neighbor_ranks().collect();
        assert_eq!(ranks, vec![neighbor]);

        // All mappings unresolved
        assert!(ov.links_to(neighbor).all(|(_, rp)| rp.is_none()));

        // Idempotent
        let added2 = expand_one_layer_mesh(&mut ov, &mesh, [seed], neighbor);
        assert_eq!(added2, 0);
        assert_eq!(ov.links_to(neighbor).count(), expected.len());
    }

    #[test]
    fn links_sorted_deterministic() {
        let mut ov = Overlap::new();
        // insert out of order
        ov.try_add_link(PointId::new(3).unwrap(), 1, PointId::new(103).unwrap())
            .unwrap();
        ov.try_add_link(PointId::new(1).unwrap(), 1, PointId::new(101).unwrap())
            .unwrap();
        ov.try_add_link(PointId::new(2).unwrap(), 1, PointId::new(102).unwrap())
            .unwrap();

        let sorted = ov.links_to_sorted(1);
        let order: Vec<_> = sorted.iter().map(|(p, _)| *p).collect();
        assert_eq!(
            order,
            vec![
                PointId::new(1).unwrap(),
                PointId::new(2).unwrap(),
                PointId::new(3).unwrap(),
            ]
        );

        let sorted_res = ov.links_to_resolved_sorted(1);
        let order_res: Vec<_> = sorted_res.iter().map(|(p, _)| *p).collect();
        assert_eq!(order_res, order);
    }

    #[test]
    fn invariant_bipartite_violation() {
        let mut ov = Overlap::new();
        let src = local(PointId::new(1).unwrap());
        let dst = local(PointId::new(2).unwrap());
        ov.inner.adjacency_out.entry(src).or_default().push((
            dst,
            Remote {
                rank: 0,
                remote_point: None,
            },
        ));
        let err = ov.validate_invariants().unwrap_err();
        assert_eq!(err, MeshSieveError::OverlapNonBipartite { src, dst });
    }

    #[test]
    fn invariant_rank_mismatch() {
        let mut ov = Overlap::new();
        insert_raw_edge(
            &mut ov,
            local(PointId::new(1).unwrap()),
            part(5),
            Remote {
                rank: 4,
                remote_point: None,
            },
        );
        let err = ov.validate_invariants().unwrap_err();
        assert_eq!(
            err,
            MeshSieveError::OverlapRankMismatch {
                expected: 5,
                found: 4,
            }
        );
    }

    #[test]
    fn invariant_duplicate_edge() {
        let mut ov = Overlap::new();
        let src = local(PointId::new(1).unwrap());
        let dst = part(3);
        let rem = Remote {
            rank: 3,
            remote_point: None,
        };
        insert_raw_edge(&mut ov, src, dst, rem);
        insert_raw_edge(&mut ov, src, dst, rem);
        let err = ov.validate_invariants().unwrap_err();
        assert_eq!(err, MeshSieveError::OverlapDuplicateEdge { src, dst });
    }

    #[test]
    fn invariant_part_in_base_points() {
        let mut ov = Overlap::new();
        ov.inner.adjacency_out.insert(part(7), Vec::new());
        let err = ov.validate_invariants().unwrap_err();
        assert_eq!(err, MeshSieveError::OverlapPartInBasePoints);
    }

    #[test]
    fn invariant_local_in_cap_points() {
        let mut ov = Overlap::new();
        ov.inner
            .adjacency_in
            .insert(local(PointId::new(8).unwrap()), Vec::new());
        let err = ov.validate_invariants().unwrap_err();
        assert_eq!(err, MeshSieveError::OverlapLocalInCapPoints);
    }

    #[test]
    fn invariant_happy_path() {
        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(PointId::new(1).unwrap(), 2)
            .unwrap();
        assert!(ov.validate_invariants().is_ok());
    }

    #[test]
    #[should_panic]
    fn trait_add_arrow_non_bipartite_panics() {
        let mut ov = Overlap::new();
        let src = local(PointId::new(1).unwrap());
        let dst = local(PointId::new(2).unwrap());
        ov.add_arrow(
            src,
            dst,
            Remote {
                rank: 0,
                remote_point: None,
            },
        );
    }

    #[test]
    #[should_panic]
    fn trait_add_arrow_rank_mismatch_panics() {
        let mut ov = Overlap::new();
        let src = local(PointId::new(1).unwrap());
        let dst = part(1);
        ov.add_arrow(
            src,
            dst,
            Remote {
                rank: 2,
                remote_point: None,
            },
        );
    }

    #[cfg(not(any(
        debug_assertions,
        feature = "strict-invariants",
        feature = "check-invariants",
    )))]
    #[test]
    fn trait_add_arrow_invalid_noop_release() {
        let mut ov = Overlap::new();
        let src = local(PointId::new(1).unwrap());
        let dst = local(PointId::new(2).unwrap());
        ov.add_arrow(
            src,
            dst,
            Remote {
                rank: 0,
                remote_point: None,
            },
        );
        assert!(ov.inner.adjacency_out.is_empty());
    }
}

#[cfg(all(test, any(feature = "strict-invariants", feature = "check-invariants")))]
mod strict_invariant_tests {
    use super::*;
    use crate::mesh_error::MeshSieveError;
    use crate::topology::point::PointId;

    fn pid(n: u64) -> PointId {
        PointId::new(n).unwrap()
    }

    #[test]
    fn detects_missing_in_mirror() {
        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(pid(1), 7).unwrap();
        ov.debug_validate();

        let src = OvlId::Local(pid(1));
        let dst = OvlId::Part(7);
        if let Some(vec_out) = ov.inner.adjacency_out.get_mut(&src) {
            vec_out.retain(|(d, _)| *d != dst);
        }

        let err = ov.validate_invariants().unwrap_err();
        match err {
            MeshSieveError::OverlapInOutMirrorMissing { src: s, dst: d } => {
                assert_eq!(s, src);
                assert_eq!(d, dst);
            }
            e => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn detects_payload_mismatch() {
        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(pid(2), 3).unwrap();
        ov.resolve_remote_point(pid(2), 3, pid(42)).unwrap();
        ov.debug_validate();

        let src = OvlId::Local(pid(2));
        let dst = OvlId::Part(3);
        if let Some(vec_in) = ov.inner.adjacency_in.get_mut(&dst) {
            if let Some((_, rem)) = vec_in.iter_mut().find(|(s, _)| *s == src) {
                rem.remote_point = Some(pid(99));
            }
        }

        let err = ov.validate_invariants().unwrap_err();
        match err {
            MeshSieveError::OverlapInOutPayloadMismatch {
                src: s,
                dst: d,
                out,
                inn,
            } => {
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_ne!(out, inn);
            }
            e => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn detects_count_mismatch() {
        let mut ov = Overlap::new();
        ov.try_add_link_structural_one(pid(5), 1).unwrap();
        ov.try_add_link_structural_one(pid(6), 1).unwrap();
        ov.debug_validate();

        let dst = OvlId::Part(1);
        if let Some(vec_in) = ov.inner.adjacency_in.get_mut(&dst) {
            if let Some(edge) = vec_in.first().cloned() {
                vec_in.push(edge);
            }
        }

        let err = ov.validate_invariants().unwrap_err();
        match err {
            MeshSieveError::OverlapInOutEdgeCountMismatch {
                out_edges,
                in_edges,
            } => {
                assert_ne!(out_edges, in_edges);
            }
            e => panic!("unexpected error: {e}"),
        }
    }
}

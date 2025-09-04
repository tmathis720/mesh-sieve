//! Core trait for sieve data structures in mesh topology.
//!
//! This module defines the [`Sieve`] trait, the core read/write incidence API
//! for mesh topologies. The trait supports generic point and payload types and
//! includes methods for traversing, querying, and **arrow-level** mutation. For
//! point/role mutators, see [`MutableSieve`](super::mutable::MutableSieve).

use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::traversal_iter::{ClosureBothIter, ClosureIter, StarIter};

pub use crate::topology::cache::InvalidateCache;

/// Core bidirectional incidence API for mesh topology.
///
/// The `Sieve` trait abstracts the concept of a directed incidence structure (such as a
/// mesh or cell complex), supporting efficient traversal and mutation of arrows (edges)
/// between points (nodes).
///
/// Implementations must maintain a **simple directed graph** invariant: for any
/// `(src, dst)` pair there is at most one stored arrow. Calling [`add_arrow`] on an
/// existing pair must **replace** the payload in place rather than creating a parallel
/// edge. All mutation routines are expected to keep the outgoing (`cone`) and
/// incoming (`support`) adjacencies in perfect mirror, removing or updating both sides
/// of the structure.
///
/// # Associated Types
/// - `Point`: The type of points in the sieve (must be `Copy`, `Eq`, `Hash`, and `Ord`).
/// - `Payload`: The type of payloads associated with arrows.
/// - `ConeIter`: Iterator over outgoing arrows from a point.
/// - `SupportIter`: Iterator over incoming arrows to a point.
///
/// # Provided Methods
/// - Arrow insertion/removal
/// - Traversal (cone, support, closure, star, etc.)
/// - Lattice operations (meet, join)
/// - Strata helpers (height, depth, diameter)
pub trait Sieve: Default + InvalidateCache
where
    Self::Point: Ord + std::fmt::Debug,
{
    type Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug;
    type Payload;

    type ConeIter<'a>: Iterator<Item = (Self::Point, Self::Payload)>
    where
        Self: 'a;
    type SupportIter<'a>: Iterator<Item = (Self::Point, Self::Payload)>
    where
        Self: 'a;

    /// Outgoing arrows from `p`.
    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a>;
    /// Incoming arrows to `p`.
    fn support<'a>(&'a self, p: Self::Point) -> Self::SupportIter<'a>;

    /// Insert or replace the arrow `src → dst`.
    ///
    /// If an arrow already exists between the two points, its payload is replaced in
    /// place. Implementations must ensure no parallel edges are created and must keep
    /// the mirror entry in the support adjacency up to date.
    fn add_arrow(&mut self, src: Self::Point, dst: Self::Point, payload: Self::Payload);
    /// Remove arrow `src → dst`, returning its payload.
    fn remove_arrow(&mut self, src: Self::Point, dst: Self::Point) -> Option<Self::Payload>;

    /// All “base” points (with outgoing arrows).
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;
    /// All “cap” points (with incoming arrows).
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a>;

    /// Convenience iterator over the destination points in `p`'s cone,
    /// discarding payload values.
    #[inline]
    fn cone_points<'a>(&'a self, p: Self::Point) -> impl Iterator<Item = Self::Point> + 'a
    where
        Self: Sized,
    {
        self.cone(p).map(|(q, _)| q)
    }
    /// Convenience iterator over the source points in `p`'s support,
    /// discarding payload values.
    #[inline]
    fn support_points<'a>(&'a self, p: Self::Point) -> impl Iterator<Item = Self::Point> + 'a
    where
        Self: Sized,
    {
        self.support(p).map(|(q, _)| q)
    }

    /// Returns true if an arrow `src → dst` exists.
    #[inline]
    fn has_arrow(&self, src: Self::Point, dst: Self::Point) -> bool
    where
        Self: Sized,
    {
        self.cone_points(src).any(|q| q == dst)
    }

    /// Return an iterator over **all** points in this Sieve’s domain
    /// (points that appear as a source or a destination of any arrow).
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // collect anything with outgoing or incoming arrows
        for p in self.base_points() {
            set.insert(p);
        }
        for p in self.cap_points() {
            set.insert(p);
        }
        Box::new(set.into_iter())
    }

    // --- graph traversals ---
    /// Concrete iterator over the transitive closure (downward) from `seeds`.
    /// Prefer this over [`closure`] for zero-alloc traversal.
    fn closure_iter<'s, I>(&'s self, seeds: I) -> ClosureIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        ClosureIter::new(self, seeds)
    }

    /// Concrete iterator over the transitive star (upward) from `seeds`.
    fn star_iter<'s, I>(&'s self, seeds: I) -> StarIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        StarIter::new(self, seeds)
    }

    /// Concrete iterator over both directions (cone + support).
    fn closure_both_iter<'s, I>(&'s self, seeds: I) -> ClosureBothIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        ClosureBothIter::new(self, seeds)
    }

    fn closure<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        Box::new(self.closure_iter(seeds))
    }
    fn star<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        Box::new(self.star_iter(seeds))
    }
    fn closure_both<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        Box::new(self.closure_both_iter(seeds))
    }

    // --- lattice ops ---
    // --- Lattice operations: meet and join ---
    /// Computes the minimal separator (meet) of two points in the Sieve.
    ///
    /// Definition (Sieve): the smallest set whose removal makes `closure(a)` and
    /// `closure(b)` disjoint. We compute `C = closure(a) ∩ closure(b)` and keep
    /// only elements minimal w.r.t. the downward reachability (cone-transitive)
    /// order: drop `x` if some other candidate's closure contains `x`
    /// (equivalently, if `x` appears in another candidate's cone). See
    /// Knepley–Karpeev: Table 1/2 (meet).  // refs: SPR-2009 Table 1,2
    fn meet<'s>(
        &'s self,
        a: Self::Point,
        b: Self::Point,
    ) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        Self::Point: Ord,
    {
        use std::collections::HashSet;

        // Candidate set: intersection of closures
        let mut ca: Vec<_> = self.closure(std::iter::once(a)).collect();
        let mut cb: Vec<_> = self.closure(std::iter::once(b)).collect();
        ca.sort_unstable();
        cb.sort_unstable();
        let mut cand: Vec<_> = ca
            .into_iter()
            .filter(|x| cb.binary_search(x).is_ok())
            .collect();
        cand.sort_unstable();
        cand.dedup();

        // HashSet for O(1) membership test
        let cand_set: HashSet<_> = cand.iter().copied().collect();

        // Keep only minimal elements: x is minimal if no other candidate's closure contains x
        let mut out: Vec<_> = cand
            .into_iter()
            .filter(|&x| {
                let mut minimal = true;
                for y in self.star(std::iter::once(x)) {
                    if y != x && cand_set.contains(&y) {
                        minimal = false; // some candidate covers x -> not minimal
                        break;
                    }
                }
                minimal
            })
            .collect();

        out.sort_unstable();
        out.dedup();
        Box::new(out.into_iter())
    }

    /// Computes the minimal separator (join) of two points in the Sieve.
    ///
    /// Definition (Sieve): the smallest set whose removal makes `star(a)` and
    /// `star(b)` disjoint. We compute `C = star(a) ∩ star(b)` and keep only
    /// elements minimal w.r.t. the upward reachability (support-transitive)
    /// order: drop x if x reaches another candidate via star.  See
    /// Knepley–Karpeev: Table 1/2 (join).  // refs: SPR-2009 Table 1,2
    fn join<'s>(
        &'s self,
        a: Self::Point,
        b: Self::Point,
    ) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        Self::Point: Ord,
    {
        use std::collections::HashSet;

        // Candidate set: intersection of stars
        let mut sa: Vec<_> = self.star(std::iter::once(a)).collect();
        let mut sb: Vec<_> = self.star(std::iter::once(b)).collect();
        sa.sort_unstable();
        sb.sort_unstable();
        let mut cand: Vec<_> = sa
            .into_iter()
            .filter(|x| sb.binary_search(x).is_ok())
            .collect();
        cand.sort_unstable();
        cand.dedup();

        let cand_set: HashSet<_> = cand.iter().copied().collect();

        // Keep only minimal elements in the upward (support) order:
        // x is minimal if star(x) contains no other candidate.
        let mut out: Vec<_> = cand
            .into_iter()
            .filter(|&x| {
                let mut minimal = true;
                for y in self.star(std::iter::once(x)) {
                    if y != x && cand_set.contains(&y) {
                        minimal = false; // x reaches a higher candidate -> not minimal
                        break;
                    }
                }
                minimal
            })
            .collect();

        out.sort_unstable();
        out.dedup();
        Box::new(out.into_iter())
    }

    // --- strata helpers (default impl via compute_strata) ---
    /// Distance from any zero-in-degree “source” to `p`.
    fn height(&mut self, p: Self::Point) -> Result<u32, MeshSieveError>
    where
        Self::Point: Ord,
        Self: Sized,
    {
        let cache = compute_strata(&*self)?;
        Ok(cache.height.get(&p).copied().unwrap_or(0))
    }

    /// Distance from `p` down to any zero-out-degree “sink”.
    fn depth(&mut self, p: Self::Point) -> Result<u32, MeshSieveError>
    where
        Self::Point: Ord,
        Self: Sized,
    {
        let cache = compute_strata(&*self)?;
        Ok(cache.depth.get(&p).copied().unwrap_or(0))
    }

    /// Maximum height (diameter of the DAG).
    fn diameter(&mut self) -> Result<u32, MeshSieveError>
    where
        Self: Sized,
    {
        Ok(compute_strata(&*self)?.diameter)
    }

    /// Iterator over all points at height `k`.
    fn height_stratum<'a>(
        &'a mut self,
        k: u32,
    ) -> Result<Box<dyn Iterator<Item = Self::Point> + 'a>, MeshSieveError>
    where
        Self: Sized,
    {
        let cache = compute_strata(&*self)?;
        let items = cache.strata.get(k as usize).cloned().unwrap_or_default();
        Ok(Box::new(items.into_iter()))
    }

    /// Iterator over all points at depth `k`.
    fn depth_stratum<'a>(
        &'a mut self,
        k: u32,
    ) -> Result<Box<dyn Iterator<Item = Self::Point> + 'a>, MeshSieveError>
    where
        Self::Point: Ord,
        Self: Sized,
    {
        let cache = compute_strata(&*self)?;
        let pts: Vec<_> = cache
            .depth
            .iter()
            .filter_map(|(&p, &d)| if d == k { Some(p) } else { None })
            .collect();
        Ok(Box::new(pts.into_iter()))
    }

    /// Deterministic contiguous index for a point within the strata chart.
    fn chart_index(&mut self, p: Self::Point) -> Result<Option<usize>, MeshSieveError>
    where
        Self: Sized,
    {
        let cache = compute_strata(&*self)?;
        Ok(cache.index_of(p))
    }

    /// Full chart of points in deterministic order (index → point).
    fn chart_points(&mut self) -> Result<Vec<Self::Point>, MeshSieveError>
    where
        Self: Sized,
    {
        let cache = compute_strata(&*self)?;
        Ok(cache.chart_points.clone())
    }

    /// # Strata helpers example
    /// ```rust
    /// # use mesh_sieve::topology::sieve::Sieve;
    /// # use mesh_sieve::topology::sieve::InMemorySieve;
    /// # use mesh_sieve::topology::point::PointId;
    /// let mut s = InMemorySieve::<PointId,()>::default();
    /// // 1→2→3→4
    /// s.add_arrow(PointId::new(1).unwrap(), PointId::new(2).unwrap(), ());
    /// s.add_arrow(PointId::new(2).unwrap(), PointId::new(3).unwrap(), ());
    /// s.add_arrow(PointId::new(3).unwrap(), PointId::new(4).unwrap(), ());
    /// assert_eq!(s.height(PointId::new(4).unwrap())?, 3);
    /// assert_eq!(s.depth(PointId::new(1).unwrap())?, 3);
    /// assert_eq!(s.diameter()?, 3);
    /// let h2: Vec<_> = s.height_stratum(2)?.collect();
    /// let d1: Vec<_> = s.depth_stratum(1)?.collect();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// These methods now return `Err(MeshSieveError::CycleDetected)` on cycles (or other topology errors), instead of panicking.
    ///
    /// # Performance
    /// These helpers are relatively expensive, as they require analyzing the entire Sieve structure.
    /// For incremental or real-time applications, consider caching the results.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::{Sieve, InMemorySieve};
    /// let mut s = InMemorySieve::<u32>::default();
    /// s.add_arrow(1, 2, ());
    /// s.add_arrow(2, 3, ());
    /// s.add_arrow(3, 4, ());
    /// s.add_arrow(4, 5, ());
    /// s.add_arrow(2, 5, ());
    /// // 1→2→3→4→5
    /// //    ↘
    /// //      ↙
    /// assert_eq!(s.height(5)?, 4);
    /// assert_eq!(s.depth(1)?, 4);
    /// assert_eq!(s.diameter()?, 4);
    /// let h2: Vec<_> = s.height_stratum(2)?.collect();
    /// let d1: Vec<_> = s.depth_stratum(1)?.collect();
    /// Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    /// Insert a brand-new point `p` into the domain (no arrows yet).
    fn add_point(&mut self, p: Self::Point)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::add_point(self, p)
    }
    /// Remove point `p` and all its arrows.
    fn remove_point(&mut self, p: Self::Point)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::remove_point(self, p)
    }
    /// Ensure `p` appears in the base (outgoing) point set, even if no arrows yet.
    fn add_base_point(&mut self, p: Self::Point)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::add_base_point(self, p)
    }
    /// Ensure `p` appears in the cap (incoming) point set.
    fn add_cap_point(&mut self, p: Self::Point)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::add_cap_point(self, p)
    }
    /// Remove `p` from base_points (dropping its outgoing arrows).
    fn remove_base_point(&mut self, p: Self::Point)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::remove_base_point(self, p)
    }
    /// Remove `p` from cap_points (dropping its incoming arrows).
    fn remove_cap_point(&mut self, p: Self::Point)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::remove_cap_point(self, p)
    }
    /// Replace `p`’s entire cone with the given chain (dst↦payload).
    fn set_cone(
        &mut self,
        p: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self: super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        <Self as super::mutable::MutableSieve>::set_cone(self, p, chain)
    }
    /// Append the given chain to `p`’s cone.
    fn add_cone(
        &mut self,
        p: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self: super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        <Self as super::mutable::MutableSieve>::add_cone(self, p, chain)
    }
    /// Replace `q`’s entire support with the given chain (src↦payload).
    fn set_support(
        &mut self,
        q: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self: super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        <Self as super::mutable::MutableSieve>::set_support(self, q, chain)
    }
    /// Append the given chain to `q`’s support.
    fn add_support(
        &mut self,
        q: Self::Point,
        chain: impl IntoIterator<Item = (Self::Point, Self::Payload)>,
    )
    where
        Self: super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        <Self as super::mutable::MutableSieve>::add_support(self, q, chain)
    }

    /// Hint to preallocate additional space in the cone (outgoing) adjacency of `p`.
    fn reserve_cone(&mut self, p: Self::Point, additional: usize)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::reserve_cone(self, p, additional)
    }

    /// Hint to preallocate additional space in the support (incoming) adjacency of `q`.
    fn reserve_support(&mut self, q: Self::Point, additional: usize)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::reserve_support(self, q, additional)
    }

    /// Produce a new Sieve containing only the base points in `chain` (and their arrows).
    fn restrict_base(&self, chain: impl IntoIterator<Item = Self::Point>) -> Self
    where
        Self: Sized + Default,
        Self::Payload: Clone,
    {
        let mut out = Self::default();
        for p in chain {
            for (dst, pay) in self.cone(p) {
                out.add_arrow(p, dst, pay.clone());
            }
        }
        out
    }
    /// Produce a new Sieve containing only the cap points in `chain` (and their arrows).
    fn restrict_cap(&self, chain: impl IntoIterator<Item = Self::Point>) -> Self
    where
        Self: Sized + Default,
        Self::Payload: Clone,
    {
        let mut out = Self::default();
        for q in chain {
            for (src, pay) in self.support(q) {
                out.add_arrow(src, q, pay.clone());
            }
        }
        out
    }
}

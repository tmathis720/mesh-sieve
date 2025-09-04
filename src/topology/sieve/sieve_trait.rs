//! Core trait for sieve data structures in mesh topology.
//!
//! This module defines the [`Sieve`] trait, the core read/write incidence API
//! for mesh topologies. The trait supports generic point and payload types and
//! includes methods for traversing, querying, and **arrow-level** mutation. For
//! point/role mutators, see [`MutableSieve`](super::mutable::MutableSieve).

use crate::mesh_error::MeshSieveError;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::traversal_iter::{
    ClosureBothIter, ClosureBothIterRef, ClosureIter, ClosureIterRef, StarIter, StarIterRef,
};

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
/// ## Edge uniqueness (no multi-edges)
/// A Sieve stores at most one arrow for any `(src, dst)` pair. [`add_arrow`] behaves as
/// an upsert: inserting when missing and replacing payload (and orientation) when present.
/// Debug builds verify this invariant and panic if parallel edges or mirror mismatches
/// are detected.
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
    ///
    /// ## Complexity
    /// - Time: **O(degree(p))**
    /// - Space: **O(1)** (iterator adaptor only)
    /// - Backend notes:
    ///   - [`InMemorySieve`](crate::topology::sieve::InMemorySieve): iteration over a
    ///     `Vec`, average-case stable cost.
    ///   - CSR/frozen backends: iteration walks a contiguous slice (cache-friendly).
    fn cone<'a>(&'a self, p: Self::Point) -> Self::ConeIter<'a>;

    /// Incoming arrows to `p`.
    ///
    /// ## Complexity
    /// - Time: **O(degree_in(p))**
    /// - Space: **O(1)**
    /// - Backend notes: as for [`cone`](Self::cone).
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

    /// All points that appear as either a source (`base_points`) or destination
    /// (`cap_points`).
    ///
    /// ## Complexity
    /// - Time: **O(|V|)**
    /// - Space: **O(|V|)** (uses a `HashSet` to form the union)
    ///
    /// ## Notes
    /// - Backends may override for lower overhead (e.g., CSR can iterate a dense chart).
    /// - For deterministic global order, prefer [`chart_points`](Self::chart_points).
    fn points<'a>(&'a self) -> Box<dyn Iterator<Item = Self::Point> + 'a> {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for p in self.base_points() {
            set.insert(p);
        }
        for p in self.cap_points() {
            set.insert(p);
        }
        Box::new(set.into_iter())
    }

    /// Deterministic set of all points in total order (`Ord`).
    #[inline]
    fn points_sorted(&self) -> Vec<Self::Point>
    where
        Self: Sized,
    {
        let mut v: Vec<_> = self.points().collect();
        v.sort_unstable();
        v
    }

    /// Deterministic whole-graph order guided by strata (height-major then `Ord`).
    /// Uses the cached chart when available; computes strata otherwise.
    ///
    /// ## Complexity
    /// - First use after mutation: **O(|V| + |E|)** (strata cache build).
    /// - Subsequent calls: **O(|V|)** to clone the chart vector.
    ///
    /// Returns an error if the topology is cyclic.
    #[inline]
    fn points_chart_order(&mut self) -> Result<Vec<Self::Point>, MeshSieveError>
    where
        Self: Sized,
    {
        self.chart_points()
    }

    // --- graph traversals ---
    /// If you only require point IDs during traversal, implement
    /// [`SieveRef`](crate::topology::sieve::SieveRef) and use the `_ref`
    /// variants which borrow payloads and avoid cloning.

    /// Concrete iterator over the transitive closure (downward) from `seeds`.
    /// Prefer this over [`closure`] for zero-alloc traversal.
    ///
    /// ## Complexity
    /// - Worst case: **O(|V| + |E|)** time, **O(|V|)** space (visited set + stack),
    ///   but bounded to the subgraph reachable from `seeds`.
    /// - Determinism: unspecified neighbor order unless you use sorted variants
    ///   (see [`closure_iter_sorted`](Self::closure_iter_sorted)); deterministic for CSR
    ///   if neighbors are pre-sorted.
    fn closure_iter<'s, I>(&'s self, seeds: I) -> ClosureIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        ClosureIter::new(self, seeds)
    }

    /// Concrete iterator over the star (upward) from `seeds`.
    /// Complexity and determinism notes as in [`closure_iter`](Self::closure_iter).
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

    /// Deterministic closure iterator: seeds sorted by `Ord`.
    #[inline]
    fn closure_iter_sorted<'s, I>(&'s self, seeds: I) -> ClosureIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        ClosureIter::new_sorted(self, seeds)
    }

    /// Deterministic star iterator: seeds sorted by `Ord`.
    #[inline]
    fn star_iter_sorted<'s, I>(&'s self, seeds: I) -> StarIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        StarIter::new_sorted(self, seeds)
    }

    /// Deterministic bi-directional closure iterator: seeds sorted by `Ord`.
    #[inline]
    fn closure_both_iter_sorted<'s, I>(&'s self, seeds: I) -> ClosureBothIter<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: Sized,
    {
        ClosureBothIter::new_sorted(self, seeds)
    }

    /// Boxed deterministic closure iterator.
    fn closure_sorted<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        Box::new(self.closure_iter_sorted(seeds))
    }

    /// Boxed deterministic star iterator.
    fn star_sorted<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        Box::new(self.star_iter_sorted(seeds))
    }

    /// Boxed deterministic bi-directional closure iterator.
    fn closure_both_sorted<'s, I>(&'s self, seeds: I) -> Box<dyn Iterator<Item = Self::Point> + 's>
    where
        I: IntoIterator<Item = Self::Point>,
    {
        Box::new(self.closure_both_iter_sorted(seeds))
    }

    /// Borrow-based downward closure: requires [`SieveRef`] and clones no payloads.
    #[inline]
    fn closure_iter_ref<'s, I>(&'s self, seeds: I) -> ClosureIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        ClosureIterRef::new_ref(self, seeds)
    }

    /// Borrow-based upward star.
    #[inline]
    fn star_iter_ref<'s, I>(&'s self, seeds: I) -> StarIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        StarIterRef::new_ref(self, seeds)
    }

    /// Borrow-based both-direction closure.
    #[inline]
    fn closure_both_iter_ref<'s, I>(&'s self, seeds: I) -> ClosureBothIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        ClosureBothIterRef::new_ref(self, seeds)
    }

    /// Deterministic ref variant: seeds sorted/deduped.
    #[inline]
    fn closure_iter_ref_sorted<'s, I>(&'s self, seeds: I) -> ClosureIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        ClosureIterRef::new_ref_sorted(self, seeds)
    }

    /// Deterministic ref variant: seeds sorted/deduped.
    #[inline]
    fn star_iter_ref_sorted<'s, I>(&'s self, seeds: I) -> StarIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        StarIterRef::new_ref_sorted(self, seeds)
    }

    /// Deterministic ref variant: seeds sorted/deduped.
    #[inline]
    fn closure_both_iter_ref_sorted<'s, I>(&'s self, seeds: I) -> ClosureBothIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        ClosureBothIterRef::new_ref_sorted(self, seeds)
    }

    /// Strongly deterministic ref variant sorting neighbors on expansion.
    #[inline]
    fn closure_iter_ref_sorted_neighbors<'s, I>(&'s self, seeds: I) -> ClosureIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        ClosureIterRef::new_ref_sorted_neighbors(self, seeds)
    }

    /// Strongly deterministic ref variant sorting neighbors on expansion.
    #[inline]
    fn star_iter_ref_sorted_neighbors<'s, I>(&'s self, seeds: I) -> StarIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        StarIterRef::new_ref_sorted_neighbors(self, seeds)
    }

    /// Strongly deterministic ref variant sorting neighbors on expansion.
    #[inline]
    fn closure_both_iter_ref_sorted_neighbors<'s, I>(
        &'s self,
        seeds: I,
    ) -> ClosureBothIterRef<'s, Self>
    where
        I: IntoIterator<Item = Self::Point>,
        Self: crate::topology::sieve::SieveRef + Sized,
    {
        ClosureBothIterRef::new_ref_sorted_neighbors(self, seeds)
    }

    // --- lattice ops ---
    // --- Lattice operations: meet and join ---
    /// Computes the meet of two points in the Sieve.
    ///
    /// Definition (Sieve): the smallest set whose removal makes `closure(a)` and
    /// `closure(b)` disjoint. We compute `C = closure(a) ∩ closure(b)` and keep
    /// only elements **maximal with respect to downward reachability**: drop `x`
    /// if some *higher* candidate’s closure contains `x` (equivalently, if `x`
    /// appears in another candidate's cone). See Knepley–Karpeev: Table 1/2
    /// (meet).  // refs: SPR-2009 Table 1,2

    /// ## Complexity
    /// - Let `G_s` be the subgraph reachable from `{a}` and `{b}` in the downward
    ///   direction.
    /// - Time: **O(|V_s| + |E_s|)** for the two traversals + **O(k log k)** to
    ///   sort/dedup candidates (`k = |C|`) +
    ///   **O(\sum_{x∈C} degree_up(x))** for the dominance filter.
    /// - Space: **O(|V_s|)** for visited sets.

    /// ## Determinism
    /// Deterministic if neighbor order is deterministic (e.g., CSR with sorted
    /// neighbors, or if you use the “sorted neighbor” traversal variants).

    /// ```rust
    /// use mesh_sieve::topology::sieve::{Sieve, InMemorySieve};
    ///
    /// // DAG:
    /// //   c1 → f
    /// //   c2 → f
    /// //   f  → e1, e2
    /// //   e1 → v1, v2
    /// //   e2 → v2, v3
    ///
    /// let (c1,c2,f,e1,e2,v1,v2,v3) = (1,2,3,4,5,6,7,8);
    /// let mut s = InMemorySieve::<u32, ()>::default();
    /// s.add_arrow(c1, f, ());
    /// s.add_arrow(c2, f, ());
    /// s.add_arrow(f, e1, ());
    /// s.add_arrow(f, e2, ());
    /// s.add_arrow(e1, v1, ());
    /// s.add_arrow(e1, v2, ());
    /// s.add_arrow(e2, v2, ());
    /// s.add_arrow(e2, v3, ());
    ///
    /// // meet(c1, c2) keeps the highest elements in closure(c1)∩closure(c2),
    /// // i.e., the shared face `f`.
    /// let mut m: Vec<_> = s.meet(c1, c2).collect();
    /// m.sort_unstable();
    /// assert_eq!(m, vec![f]);
    /// ```
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

        // Keep only maximal elements: x is maximal if no other candidate's closure contains x
        let mut out: Vec<_> = cand
            .into_iter()
            .filter(|&x| {
                let mut maximal = true;
                for y in self.star(std::iter::once(x)) {
                    if y != x && cand_set.contains(&y) {
                        maximal = false; // some higher candidate covers x -> not maximal
                        break;
                    }
                }
                maximal
            })
            .collect();

        out.sort_unstable();
        out.dedup();
        Box::new(out.into_iter())
    }

    /// Computes the join of two points in the Sieve.
    ///
    /// Definition (Sieve): the smallest set whose removal makes `star(a)` and
    /// `star(b)` disjoint. We compute `C = star(a) ∩ star(b)` and keep only
    /// elements **maximal with respect to upward reachability**: drop `x` if
    /// `star(x)` contains another candidate above it. See Knepley–Karpeev:
    /// Table 1/2 (join).  // refs: SPR-2009 Table 1,2

    /// ## Complexity
    /// - Let `G_s` be the subgraph reachable from `{a}` and `{b}` in the upward
    ///   direction.
    /// - Time: **O(|V_s| + |E_s|)** for the two traversals + **O(k log k)** to
    ///   sort/dedup candidates (`k = |C|`) +
    ///   **O(\sum_{x∈C} degree_down(x))** for the dominance filter.
    /// - Space: **O(|V_s|)** for visited sets.

    /// ## Determinism
    /// Deterministic under deterministic neighbor order (e.g., CSR with sorted
    /// neighbors or using sorted traversal variants).

    /// ```rust
    /// use mesh_sieve::topology::sieve::{Sieve, InMemorySieve};
    ///
    /// // Reuse the graph from the `meet` example.
    /// let (c1,c2,f,e1,e2,v1,v2,v3) = (1,2,3,4,5,6,7,8);
    /// let mut s = InMemorySieve::<u32, ()>::default();
    /// s.add_arrow(c1, f, ());
    /// s.add_arrow(c2, f, ());
    /// s.add_arrow(f, e1, ());
    /// s.add_arrow(f, e2, ());
    /// s.add_arrow(e1, v1, ());
    /// s.add_arrow(e1, v2, ());
    /// s.add_arrow(e2, v2, ());
    /// s.add_arrow(e2, v3, ());
    ///
    /// // join(v1, v3) keeps the highest elements in star(v1)∩star(v3),
    /// // which are the cells c1 and c2.
    /// let mut j: Vec<_> = s.join(v1, v3).collect();
    /// j.sort_unstable();
    /// assert_eq!(j, vec![c1, c2]);
    /// ```
    ///
    /// ```rust
    /// use mesh_sieve::topology::sieve::{Sieve, InMemorySieve};
    ///
    /// // Y-shape:
    /// //    a     b
    /// //     \   /
    /// //      x
    /// //     / \
    /// //    y   z
    ///
    /// let (a,b,x,y,z) = (10,11,12,13,14);
    /// let mut s = InMemorySieve::<u32, ()>::default();
    /// s.add_arrow(a, x, ());
    /// s.add_arrow(b, x, ());
    /// s.add_arrow(x, y, ());
    /// s.add_arrow(x, z, ());
    ///
    /// let mut m: Vec<_> = s.meet(a,b).collect();
    /// m.sort_unstable();
    /// assert_eq!(m, vec![x]);
    ///
    /// let mut j: Vec<_> = s.join(y,z).collect();
    /// j.sort_unstable();
    /// assert_eq!(j, vec![a,b]);
    /// ```
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

        // Keep only maximal elements in the upward (support) order:
        // x is maximal if star(x) contains no other candidate above it.
        let mut out: Vec<_> = cand
            .into_iter()
            .filter(|&x| {
                let mut maximal = true;
                for y in self.star(std::iter::once(x)) {
                    if y != x && cand_set.contains(&y) {
                        maximal = false; // x reaches a higher candidate -> not maximal
                        break;
                    }
                }
                maximal
            })
            .collect();

        out.sort_unstable();
        out.dedup();
        Box::new(out.into_iter())
    }

    // --- strata helpers (default impl via compute_strata) ---
    /// Distance from any zero-in-degree “source” to `p`.
    ///
    /// ## Complexity
    /// - First call after a topology change triggers strata computation:
    ///   **O(|V| + |E|)** time, **O(|V| + |E|)** space (heights, depths, strata).
    /// - Subsequent calls are **O(1)** from cache (amortized via `OnceCell`).
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

    /// Iterator over points at height `k`.
    /// - With cache: collecting the layer is **O(|strata[k]|)** time,
    ///   **O(|strata[k]|)** space.
    /// - Without cache (if using [`compute_strata`](crate::topology::sieve::strata::compute_strata)):
    ///   part of the **O(|V| + |E|)** pass.
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

    /// Deterministic global chart (height-major, then `Ord`).
    ///
    /// ## Complexity
    /// - First use after mutation: **O(|V| + |E|)** (strata cache build).
    /// - Subsequent calls: **O(|V|)** to clone the `chart_points` vector.
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
    ) where
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
    ) where
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
    ) where
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
    ) where
        Self: super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        <Self as super::mutable::MutableSieve>::add_support(self, q, chain)
    }

    /// Hint to preallocate additional space in the cone (outgoing) adjacency of `p`.
    ///
    /// # Contract
    /// - Pure performance hint; **must not** change topology or payloads.
    /// - **Must not** invalidate derived caches; it has no logical effect on the mesh.
    /// - Implementations may over-allocate but must not shrink.
    /// - Safe to call redundantly.
    ///
    /// Use this before bulk [`add_arrow`](Sieve::add_arrow) / [`add_cone`](Sieve::add_cone)
    /// when the number of outgoing arrows to be appended for `p` is known or can be
    /// estimated (e.g. during mesh construction).
    fn reserve_cone(&mut self, p: Self::Point, additional: usize)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::reserve_cone(self, p, additional)
    }

    /// Hint to preallocate additional space in the support (incoming) adjacency of `q`.
    ///
    /// Mirrors [`reserve_cone`](Sieve::reserve_cone); use when you know/estimate how many
    /// incoming arrows will end at `q`.
    fn reserve_support(&mut self, q: Self::Point, additional: usize)
    where
        Self: super::mutable::MutableSieve,
    {
        <Self as super::mutable::MutableSieve>::reserve_support(self, q, additional)
    }

    /// Produce a new Sieve containing only the base points in `chain` (and their arrows).
    fn restrict_base(&self, chain: impl IntoIterator<Item = Self::Point>) -> Self
    where
        Self: Sized + Default + super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        use std::collections::HashMap;

        // First pass: count how many outgoing (and mirrored incoming) edges we will insert.
        // We also gather the seeds so we don't iterate them again.
        let mut seeds: Vec<Self::Point> = Vec::new();
        let mut out_counts: HashMap<Self::Point, usize> = HashMap::new();
        let mut in_counts: HashMap<Self::Point, usize> = HashMap::new();

        for p in chain {
            seeds.push(p);
            let mut deg = 0usize;
            for (q, _) in self.cone(p) {
                deg += 1;
                *in_counts.entry(q).or_default() += 1;
            }
            if deg > 0 {
                out_counts.insert(p, deg);
            }
        }

        // Second pass: build the restricted sieve, pre-reserving where possible.
        let mut out = Self::default();

        // Ensure base/cap point presence for seeds & targets before reserve.
        for &p in &seeds {
            Sieve::add_base_point(&mut out, p);
        }
        for (&q, _) in &in_counts {
            Sieve::add_cap_point(&mut out, q);
        }

        // Reserve based on counts (hint-only if backend ignores it).
        for (&p, &k) in &out_counts {
            Sieve::reserve_cone(&mut out, p, k);
        }
        for (&q, &k) in &in_counts {
            Sieve::reserve_support(&mut out, q, k);
        }

        // Final insertion
        for p in seeds {
            for (q, pay) in self.cone(p) {
                out.add_arrow(p, q, pay.clone());
            }
        }
        out
    }
    /// Produce a new Sieve containing only the cap points in `chain` (and their arrows).
    fn restrict_cap(&self, chain: impl IntoIterator<Item = Self::Point>) -> Self
    where
        Self: Sized + Default + super::mutable::MutableSieve,
        Self::Payload: Clone,
    {
        use std::collections::HashMap;

        let mut seeds: Vec<Self::Point> = Vec::new();
        let mut in_counts: HashMap<Self::Point, usize> = HashMap::new();
        let mut out_counts: HashMap<Self::Point, usize> = HashMap::new();

        for q in chain {
            seeds.push(q);
            let mut deg = 0usize;
            for (p, _) in self.support(q) {
                deg += 1;
                *out_counts.entry(p).or_default() += 1;
            }
            if deg > 0 {
                in_counts.insert(q, deg);
            }
        }

        let mut out = Self::default();

        for &q in &seeds {
            Sieve::add_cap_point(&mut out, q);
        }
        for (&p, _) in &out_counts {
            Sieve::add_base_point(&mut out, p);
        }

        for (&q, &k) in &in_counts {
            Sieve::reserve_support(&mut out, q, k);
        }
        for (&p, &k) in &out_counts {
            Sieve::reserve_cone(&mut out, p, k);
        }

        for q in seeds {
            for (p, pay) in self.support(q) {
                out.add_arrow(p, q, pay.clone());
            }
        }
        out
    }
}

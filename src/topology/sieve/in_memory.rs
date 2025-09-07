//! In-memory implementation of the [`Sieve`] trait.
//!
//! This module provides [`InMemorySieve`], a simple and efficient in-memory representation
//! of a sieve using hash maps for adjacency storage. Hash-map + `Vec` adjacency
//! yields average-case **O(1)** updates. Iteration order follows insertion order of
//! the adjacency vectors (not globally deterministic unless you pre-sort).
//! It supports generic point and payload types.

use super::build_ext::SieveBuildExt;
use super::mutable::MutableSieve;
use super::query_ext::SieveQueryExt;
use super::sieve_ref::SieveRef;
use super::sieve_trait::Sieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::_debug_invariants::debug_invariants;
use crate::topology::bounds::{PayloadLike, PointLike};
use crate::topology::cache::InvalidateCache;
use crate::topology::sieve::strata::{StrataCache, compute_strata};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;

/// An in-memory sieve implementation using hash maps for adjacency storage.
///
/// # Type Parameters
/// - `P`: The type of points in the sieve. Must implement `Ord`.
/// - `T`: The type of payloads associated with arrows. Defaults to `()`.
#[derive(Clone, Debug)]
pub struct InMemorySieve<P, T = ()>
where
    P: PointLike,
{
    /// Outgoing adjacency: maps each point to a vector of (destination, payload) pairs.
    pub adjacency_out: HashMap<P, Vec<(P, T)>>,
    /// Incoming adjacency: maps each point to a vector of (source, payload) pairs.
    pub adjacency_in: HashMap<P, Vec<(P, T)>>,
    /// Cached strata information for the sieve.
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P, T> InMemorySieve<P, Arc<T>>
where
    P: PointLike,
{
    /// Insert by value; wraps once into `Arc<T>`.
    #[inline]
    pub fn add_arrow_val(&mut self, src: P, dst: P, payload: T) {
        self.add_arrow(src, dst, Arc::new(payload));
    }

    /// Batch convenience that wraps each payload once.
    #[inline]
    pub fn add_cone_val(&mut self, p: P, chain: impl IntoIterator<Item = (P, T)>) {
        for (dst, pay) in chain {
            self.add_arrow(p, dst, Arc::new(pay));
        }
        self.invalidate_cache();
    }
}

impl<P: PointLike, T> Default for InMemorySieve<P, T> {
    fn default() -> Self {
        Self {
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            strata: OnceCell::new(),
        }
    }
}

impl<P: PointLike, T: PayloadLike> InMemorySieve<P, T> {
    /// Creates a new, empty `InMemorySieve`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Constructs an `InMemorySieve` from an iterator of arrows.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let arrows = vec![(1, 2, "a"), (1, 3, "b")];
    /// let sieve = InMemorySieve::from_arrows(arrows);
    /// assert_eq!(sieve.cone(1).count(), 2);
    /// ```
    pub fn from_arrows<I: IntoIterator<Item = (P, P, T)>>(arrows: I) -> Self
    where
        T: Clone,
    {
        let mut sieve = Self::default();
        for (src, dst, payload) in arrows {
            sieve.add_arrow(src, dst, payload);
        }
        sieve
    }

    #[inline]
    pub fn strata_cache(&self) -> Result<&StrataCache<P>, MeshSieveError> {
        self.strata.get_or_try_init(|| compute_strata(self))
    }

    #[inline]
    pub fn invalidate_strata(&mut self) {
        self.strata.take();
    }

    #[inline]
    fn scrub_outgoing_only(&mut self, src: P) {
        let old: Vec<(P, T)> = std::mem::take(self.adjacency_out.entry(src).or_default());
        for (dst, _) in old {
            if let Some(ins) = self.adjacency_in.get_mut(&dst) {
                ins.retain(|(s, _)| *s != src);
            }
        }
    }

    #[inline]
    fn scrub_incoming_only(&mut self, dst: P) {
        let old: Vec<(P, T)> = std::mem::take(self.adjacency_in.entry(dst).or_default());
        for (src, _) in old {
            if let Some(outs) = self.adjacency_out.get_mut(&src) {
                outs.retain(|(d, _)| *d != dst);
            }
        }
    }

    /// Sort adjacency lists in-place for deterministic neighbor order.
    /// Mirrors remain consistent as edges are untouched.
    pub fn sort_adjacency(&mut self) {
        for outs in self.adjacency_out.values_mut() {
            outs.sort_unstable_by_key(|(dst, _)| *dst);
        }
        for ins in self.adjacency_in.values_mut() {
            ins.sort_unstable_by_key(|(src, _)| *src);
        }
    }

    #[cfg(any(debug_assertions, feature = "strict-invariants"))]
    pub(crate) fn debug_assert_invariants(&self) {
        use crate::topology::_debug_invariants as dbg;

        let out_view: std::collections::HashMap<P, Vec<(P, ())>> = self
            .adjacency_out
            .iter()
            .map(|(&src, v)| (src, v.iter().map(|(dst, _)| (*dst, ())).collect()))
            .collect();
        dbg::assert_no_dups_per_src(&out_view);

        let out_total: usize = self.adjacency_out.values().map(|v| v.len()).sum();
        let in_total: usize = self.adjacency_in.values().map(|v| v.len()).sum();
        debug_assert_eq!(out_total, in_total, "total out != total in");

        let out_pairs = dbg::count_pairs(
            self.adjacency_out
                .iter()
                .flat_map(|(&src, vec)| vec.iter().map(move |(dst, _)| (src, *dst))),
        );
        let in_pairs = dbg::count_pairs(
            self.adjacency_in
                .iter()
                .flat_map(|(&dst, vec)| vec.iter().map(move |(src, _)| (*src, dst))),
        );
        dbg::counts_equal(&out_pairs, &in_pairs, "adjacency_out", "adjacency_in");

        // Optional self-loop check:
        // for (&src, vec) in &self.adjacency_out {
        //     for (dst, _) in vec {
        //         debug_assert_ne!(src, *dst, "self-loop forbidden: {:?}", src);
        //     }
        // }
    }

    #[inline]
    pub fn has_arrow(&self, src: P, dst: P) -> bool {
        self.adjacency_out
            .get(&src)
            .is_some_and(|v| v.iter().any(|(d, _)| *d == dst))
    }

    /// Pre-size cone/support capacities from `(src, dst, count)` tuples.
    pub fn reserve_from_edge_counts(&mut self, counts: impl IntoIterator<Item = (P, P, usize)>) {
        use std::collections::HashMap;
        let mut by_src: HashMap<P, usize> = HashMap::new();
        let mut by_dst: HashMap<P, usize> = HashMap::new();

        for (src, dst, k) in counts {
            *by_src.entry(src).or_default() += k;
            *by_dst.entry(dst).or_default() += k;
        }
        for (src, k) in by_src {
            MutableSieve::reserve_cone(self, src, k);
        }
        for (dst, k) in by_dst {
            MutableSieve::reserve_support(self, dst, k);
        }
    }

    /// Convenience helper to preallocate from a raw edge list.
    pub fn reserve_from_edges(&mut self, edges: impl IntoIterator<Item = (P, P)>) {
        use std::collections::HashMap;
        let mut by_src: HashMap<P, usize> = HashMap::new();
        let mut by_dst: HashMap<P, usize> = HashMap::new();
        for (s, d) in edges {
            *by_src.entry(s).or_default() += 1;
            *by_dst.entry(d).or_default() += 1;
        }
        for (s, k) in by_src {
            MutableSieve::reserve_cone(self, s, k);
        }
        for (d, k) in by_dst {
            MutableSieve::reserve_support(self, d, k);
        }
    }

    /// Optional: compact any excess capacity after bulk construction.
    pub fn shrink_to_fit(&mut self) {
        for v in self.adjacency_out.values_mut() {
            v.shrink_to_fit();
        }
        for v in self.adjacency_in.values_mut() {
            v.shrink_to_fit();
        }
    }
}

impl<P: PointLike, T: PayloadLike> InvalidateCache for InMemorySieve<P, T> {
    #[inline]
    fn invalidate_cache(&mut self) {
        self.strata.take();
    }
}

type ConeMapIter<'a, P, T> = std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, T)>;
type ConeRefMapIter<'a, P, T> =
    std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, &'a T)>;

impl<P: PointLike, T: PayloadLike> Sieve for InMemorySieve<P, T> {
    type Point = P;
    type Payload = T;
    type ConeIter<'a>
        = ConeMapIter<'a, P, T>
    where
        Self: 'a;
    type SupportIter<'a>
        = ConeMapIter<'a, P, T>
    where
        Self: 'a;

    /// Returns an iterator over the cone of a point.
    ///
    /// The cone of a point `p` is the set of all points that can be reached from `p`
    /// by following arrows, along with the payloads of the arrows.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2,());
    /// let mut cone: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
    /// cone.sort();
    /// assert_eq!(cone, vec![2]);
    /// ```
    fn cone<'a>(&'a self, p: P) -> Self::ConeIter<'a> {
        fn map_fn<P: Copy, T: Clone>((dst, pay): &(P, T)) -> (P, T) {
            (*dst, pay.clone())
        }
        let f: fn(&(P, T)) -> (P, T) = map_fn::<P, T>;
        self.adjacency_out
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }
    /// Returns an iterator over the support of a point.
    ///
    /// The support of a point `p` is the set of all points that can reach `p`
    /// by following arrows in the reverse direction, along with the payloads of the arrows.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2,());
    /// let mut support: Vec<_> = s.support(2).map(|(u, _)| u).collect();
    /// support.sort();
    /// assert_eq!(support, vec![1]);
    /// ```
    fn support<'a>(&'a self, p: P) -> Self::SupportIter<'a> {
        fn map_fn<P: Copy, T: Clone>((src, pay): &(P, T)) -> (P, T) {
            (*src, pay.clone())
        }
        let f: fn(&(P, T)) -> (P, T) = map_fn::<P, T>;
        self.adjacency_in
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }
    /// Adds a new arrow from `src` to `dst` with the given `payload`.
    ///
    /// ## Complexity (amortized)
    /// - Time: **O(1)** expected to append to both adjacency vectors (hash map
    ///   lookup + `Vec::push`), **O(degree(dst))** in worst case if a rehash or
    ///   vector reallocation occurs.
    /// - Space: amortized **O(1)** per insertion.
    ///
    /// This method updates the outgoing adjacency of `src` and the incoming
    /// adjacency of `dst`. It also invalidates the cache for strata information.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, char>::new();
    /// s.add_arrow(1, 2, 'a');
    /// assert_eq!(s.cone(1).count(), 1);
    /// ```
    fn add_arrow(&mut self, src: P, dst: P, payload: T) {
        // Upsert outgoing
        let outs = self.adjacency_out.entry(src).or_default();
        if let Some(slot) = outs.iter_mut().find(|(d, _)| *d == dst) {
            slot.1 = payload.clone();
        } else {
            outs.push((dst, payload.clone()));
        }

        // Upsert incoming mirror
        let ins = self.adjacency_in.entry(dst).or_default();
        if let Some(slot) = ins.iter_mut().find(|(s, _)| *s == src) {
            slot.1 = payload;
        } else {
            ins.push((src, payload));
        }

        self.invalidate_cache();
        debug_invariants!(self);
    }

    /// Removes the arrow from `src` to `dst`, returning the associated payload if it
    /// existed.
    ///
    /// ## Complexity
    /// - **O(degree(src) + degree(dst))** (linear scan to find & remove).
    ///
    /// This method updates both the outgoing adjacency of `src` and the incoming
    /// adjacency of `dst`. It also invalidates the cache for strata information.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2, ());
    /// assert_eq!(s.remove_arrow(1, 2), Some(()));
    /// assert_eq!(s.remove_arrow(1, 2), None);
    /// ```
    fn remove_arrow(&mut self, src: P, dst: P) -> Option<T> {
        let mut removed = None;
        if let Some(v) = self.adjacency_out.get_mut(&src)
            && let Some(pos) = v.iter().position(|(d, _)| *d == dst)
        {
            removed = Some(v.remove(pos).1);
        }
        if let Some(v) = self.adjacency_in.get_mut(&dst)
            && let Some(pos) = v.iter().position(|(s, _)| *s == src)
        {
            v.remove(pos);
        }
        self.invalidate_cache();
        debug_invariants!(self);
        removed
    }

    // strata helpers now provided by Sieve trait default impls
    /// Returns an iterator over all points in the sieve.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2, ());
    /// let mut all_points: Vec<_> = s.points().collect();
    /// all_points.sort();
    /// assert_eq!(all_points, vec![1, 2]);
    /// ```
    fn base_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(self.adjacency_out.keys().copied())
    }
    /// Returns an iterator over all cap points in the sieve.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2, ());
    /// let mut all_caps: Vec<_> = s.cap_points().collect();
    /// all_caps.sort();
    /// assert_eq!(all_caps, vec![2]);
    /// ```
    fn cap_points<'a>(&'a self) -> Box<dyn Iterator<Item = P> + 'a> {
        Box::new(self.adjacency_in.keys().copied())
    }

    fn height(&mut self, p: P) -> Result<u32, MeshSieveError> {
        let cache = self.strata_cache()?;
        Ok(cache.height.get(&p).copied().unwrap_or(0))
    }

    fn depth(&mut self, p: P) -> Result<u32, MeshSieveError> {
        let cache = self.strata_cache()?;
        Ok(cache.depth.get(&p).copied().unwrap_or(0))
    }

    fn diameter(&mut self) -> Result<u32, MeshSieveError> {
        Ok(self.strata_cache()?.diameter)
    }

    fn height_stratum<'a>(
        &'a mut self,
        k: u32,
    ) -> Result<Box<dyn Iterator<Item = P> + 'a>, MeshSieveError> {
        let items = self
            .strata_cache()?
            .strata
            .get(k as usize)
            .cloned()
            .unwrap_or_default();
        Ok(Box::new(items.into_iter()))
    }

    fn depth_stratum<'a>(
        &'a mut self,
        k: u32,
    ) -> Result<Box<dyn Iterator<Item = P> + 'a>, MeshSieveError> {
        let cache = self.strata_cache()?;
        let pts: Vec<_> = cache
            .depth
            .iter()
            .filter_map(|(&p, &d)| if d == k { Some(p) } else { None })
            .collect();
        Ok(Box::new(pts.into_iter()))
    }

    fn chart_index(&mut self, p: P) -> Result<Option<usize>, MeshSieveError> {
        Ok(self.strata_cache()?.index_of(p))
    }

    fn chart_points(&mut self) -> Result<Vec<P>, MeshSieveError> {
        Ok(self.strata_cache()?.chart_points.clone())
    }
}

impl<P: PointLike, T: PayloadLike> SieveQueryExt for InMemorySieve<P, T> {
    #[inline]
    fn out_degree(&self, p: P) -> usize {
        self.adjacency_out.get(&p).map_or(0, |v| v.len())
    }
    #[inline]
    fn in_degree(&self, p: P) -> usize {
        self.adjacency_in.get(&p).map_or(0, |v| v.len())
    }
}

impl<P: PointLike, T: PayloadLike> SieveBuildExt for InMemorySieve<P, T>
where
    T: Clone,
{
    fn add_arrows_from<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = (P, P, T)>,
    {
        use std::collections::HashMap;
        let mut by_src: HashMap<P, HashMap<P, T>> = HashMap::new();
        let mut by_dst: HashMap<P, usize> = HashMap::new();
        for (s, d, pay) in edges {
            by_src.entry(s).or_default().insert(d, pay);
        }
        for m in by_src.values() {
            for &d in m.keys() {
                *by_dst.entry(d).or_default() += 1;
            }
        }
        for (s, m) in &by_src {
            MutableSieve::reserve_cone(self, *s, m.len());
        }
        for (d, k) in by_dst {
            MutableSieve::reserve_support(self, d, k);
        }
        for (s, m) in by_src {
            let out = self.adjacency_out.entry(s).or_default();
            for (d, pay) in m {
                if let Some(pos) = out.iter().position(|(dd, _)| *dd == d) {
                    out[pos].1 = pay.clone();
                    if let Some(ins) = self.adjacency_in.get_mut(&d)
                        && let Some(pos2) = ins.iter().position(|(ss, _)| *ss == s)
                    {
                        ins[pos2].1 = pay;
                    }
                } else {
                    out.push((d, pay.clone()));
                    self.adjacency_in.entry(d).or_default().push((s, pay));
                }
            }
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_invariants();
    }

    fn add_arrows_dedup_from<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = (P, P, T)>,
    {
        use std::collections::{HashMap, HashSet};
        let mut by_src: HashMap<P, HashMap<P, T>> = HashMap::new();
        let mut by_dst: HashMap<P, usize> = HashMap::new();
        let mut seen: HashSet<(P, P)> = HashSet::new();
        for (s, d, pay) in edges {
            if seen.insert((s, d)) {
                by_src.entry(s).or_default().insert(d, pay);
                *by_dst.entry(d).or_default() += 1;
            } else {
                by_src.get_mut(&s).unwrap().insert(d, pay);
            }
        }
        for (s, m) in &by_src {
            MutableSieve::reserve_cone(self, *s, m.len());
        }
        for (d, k) in by_dst {
            MutableSieve::reserve_support(self, d, k);
        }
        for (s, m) in by_src {
            let out = self.adjacency_out.entry(s).or_default();
            for (d, pay) in m {
                if let Some(pos) = out.iter().position(|(dd, _)| *dd == d) {
                    out[pos].1 = pay.clone();
                    if let Some(ins) = self.adjacency_in.get_mut(&d)
                        && let Some(pos2) = ins.iter().position(|(ss, _)| *ss == s)
                    {
                        ins[pos2].1 = pay;
                    }
                } else {
                    out.push((d, pay.clone()));
                    self.adjacency_in.entry(d).or_default().push((s, pay));
                }
            }
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_invariants();
    }
}

impl<P, T> MutableSieve for InMemorySieve<P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    fn reserve_cone(&mut self, p: P, additional: usize) {
        self.adjacency_out.entry(p).or_default().reserve(additional);
    }

    fn reserve_support(&mut self, q: P, additional: usize) {
        self.adjacency_in.entry(q).or_default().reserve(additional);
    }

    fn add_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn remove_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
        }
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_base_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_cap_point(&mut self, p: P) {
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn remove_base_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
            self.adjacency_out.remove(&p);
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn remove_cap_point(&mut self, p: P) {
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
            self.adjacency_in.remove(&p);
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn set_cone(&mut self, p: P, chain: impl IntoIterator<Item = (P, T)>) {
        let mut new_cone: Vec<(P, T)> = chain.into_iter().collect();

        // Dedup by destination, last wins
        {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            let mut dedup = Vec::with_capacity(new_cone.len());
            for (dst, pay) in new_cone.into_iter().rev() {
                if seen.insert(dst) {
                    dedup.push((dst, pay));
                }
            }
            dedup.reverse();
            new_cone = dedup;
        }

        // Reserve outgoing for p and incoming per destination
        self.adjacency_out
            .entry(p)
            .or_default()
            .reserve(new_cone.len());
        {
            use std::collections::HashMap;
            let mut dst_counts: HashMap<P, usize> = HashMap::new();
            for (dst, _) in &new_cone {
                *dst_counts.entry(*dst).or_default() += 1;
            }
            for (dst, cnt) in dst_counts {
                MutableSieve::reserve_support(self, dst, cnt);
            }
        }

        // Remove mirrors of old cone
        let old_cone: Vec<(P, T)> = std::mem::take(self.adjacency_out.entry(p).or_default());
        for (dst, _) in &old_cone {
            if let Some(ins) = self.adjacency_in.get_mut(dst) {
                ins.retain(|(src, _)| *src != p);
            }
        }

        // Install new cone and mirrors
        let out = self.adjacency_out.entry(p).or_default();
        *out = new_cone.clone();
        for (dst, pay) in new_cone {
            self.adjacency_in.entry(dst).or_default().push((p, pay));
        }

        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_cone(&mut self, p: P, chain: impl IntoIterator<Item = (P, T)>) {
        let add: Vec<(P, T)> = chain.into_iter().collect();

        // Reserve outgoing for p and mirrored incoming per destination
        self.adjacency_out.entry(p).or_default().reserve(add.len());
        {
            use std::collections::HashMap;
            let mut dst_counts: HashMap<P, usize> = HashMap::new();
            for (dst, _) in &add {
                *dst_counts.entry(*dst).or_default() += 1;
            }
            for (dst, cnt) in dst_counts {
                MutableSieve::reserve_support(self, dst, cnt);
            }
        }

        let out = self.adjacency_out.entry(p).or_default();
        for (dst, pay) in add {
            if let Some(slot) = out.iter_mut().find(|(d, _)| *d == dst) {
                slot.1 = pay.clone();
            } else {
                out.push((dst, pay.clone()));
            }
            let ins = self.adjacency_in.entry(dst).or_default();
            if let Some(slot) = ins.iter_mut().find(|(s, _)| *s == p) {
                slot.1 = pay;
            } else {
                ins.push((p, pay));
            }
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn set_support(&mut self, q: P, chain: impl IntoIterator<Item = (P, T)>) {
        let mut new_sup: Vec<(P, T)> = chain.into_iter().collect();

        // Dedup by source, last wins
        {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            let mut dedup = Vec::with_capacity(new_sup.len());
            for (src, pay) in new_sup.into_iter().rev() {
                if seen.insert(src) {
                    dedup.push((src, pay));
                }
            }
            dedup.reverse();
            new_sup = dedup;
        }

        // Reserve incoming for q and outgoing per source
        self.adjacency_in
            .entry(q)
            .or_default()
            .reserve(new_sup.len());
        {
            use std::collections::HashMap;
            let mut src_counts: HashMap<P, usize> = HashMap::new();
            for (src, _) in &new_sup {
                *src_counts.entry(*src).or_default() += 1;
            }
            for (src, cnt) in src_counts {
                MutableSieve::reserve_cone(self, src, cnt);
            }
        }

        // Remove mirrors of old support
        let old_sup: Vec<(P, T)> = std::mem::take(self.adjacency_in.entry(q).or_default());
        for (src, _) in &old_sup {
            if let Some(outs) = self.adjacency_out.get_mut(src) {
                outs.retain(|(dst, _)| *dst != q);
            }
        }

        // Install new support and mirrors
        let ins = self.adjacency_in.entry(q).or_default();
        *ins = new_sup.clone();
        for (src, pay) in new_sup {
            self.adjacency_out.entry(src).or_default().push((q, pay));
        }

        self.invalidate_cache();
        debug_invariants!(self);
    }

    fn add_support(&mut self, q: P, chain: impl IntoIterator<Item = (P, T)>) {
        let add: Vec<(P, T)> = chain.into_iter().collect();

        // Reserve incoming for q and mirrored outgoing per source
        self.adjacency_in.entry(q).or_default().reserve(add.len());
        {
            use std::collections::HashMap;
            let mut src_counts: HashMap<P, usize> = HashMap::new();
            for (src, _) in &add {
                *src_counts.entry(*src).or_default() += 1;
            }
            for (src, cnt) in src_counts {
                MutableSieve::reserve_cone(self, src, cnt);
            }
        }

        let ins = self.adjacency_in.entry(q).or_default();
        for (src, pay) in add {
            if let Some(slot) = ins.iter_mut().find(|(s, _)| *s == src) {
                slot.1 = pay.clone();
            } else {
                ins.push((src, pay.clone()));
            }
            let outs = self.adjacency_out.entry(src).or_default();
            if let Some(slot) = outs.iter_mut().find(|(d, _)| *d == q) {
                slot.1 = pay;
            } else {
                outs.push((q, pay));
            }
        }
        self.invalidate_cache();
        debug_invariants!(self);
    }
}

impl<P, T> SieveRef for InMemorySieve<P, T>
where
    P: PointLike,
    T: PayloadLike,
{
    type ConeRefIter<'a>
        = ConeRefMapIter<'a, P, T>
    where
        Self: 'a;
    type SupportRefIter<'a>
        = ConeRefMapIter<'a, P, T>
    where
        Self: 'a;

    fn cone_ref<'a>(&'a self, p: P) -> Self::ConeRefIter<'a> {
        fn map_fn<P: Copy, T>((dst, pay): &(P, T)) -> (P, &T) {
            (*dst, pay)
        }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_out
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }

    fn support_ref<'a>(&'a self, p: P) -> Self::SupportRefIter<'a> {
        fn map_fn<P: Copy, T>((src, pay): &(P, T)) -> (P, &T) {
            (*src, pay)
        }
        let f: fn(&(P, T)) -> (P, &T) = map_fn::<P, T>;
        self.adjacency_in
            .get(&p)
            .map(|v| v.iter().map(f))
            .unwrap_or_else(|| [].iter().map(f))
    }
}

#[cfg(test)]
mod sieve_tests {
    use super::InMemorySieve;
    use crate::topology::sieve::Sieve;

    #[test]
    fn insertion_and_removal() {
        let mut s = InMemorySieve::<u32, ()>::new();
        assert_eq!(s.remove_arrow(1, 2), None);
        s.add_arrow(1, 2, ());
        assert_eq!(s.remove_arrow(1, 2), Some(()));
    }

    #[test]
    fn cone_and_support() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 1, ());
        let mut cone: Vec<_> = s.cone(1).map(|(d, _)| d).collect();
        cone.sort();
        assert_eq!(cone, vec![2]);
        let mut support: Vec<_> = s.support(1).map(|(u, _)| u).collect();
        support.sort();
        assert_eq!(support, vec![2]);
    }

    #[test]
    fn closure_and_star_and_closure_both() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 3, ());
        // closure(1) == [1,2,3]
        let mut closure: Vec<_> = Sieve::closure(&s, [1]).collect();
        closure.sort();
        assert_eq!(closure, vec![1, 2, 3]);
        // star(3) == [3,2,1]
        let mut star: Vec<_> = Sieve::star(&s, [3]).collect();
        star.sort();
        assert_eq!(star, vec![1, 2, 3]);
        // closure_both(2) == [2,1,3]
        let mut both: Vec<_> = Sieve::closure_both(&s, [2]).collect();
        both.sort();
        assert_eq!(both, vec![1, 2, 3]);
    }

    #[test]
    fn meet_and_join() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 3, ());
        let mut m: Vec<_> = s.meet(1, 2).collect();
        m.sort();
        let mut j: Vec<_> = s.join(2, 3).collect();
        j.sort();
        // Minimal separators: meet(1,2) = {2}, join(2,3) = {1}
        assert_eq!(m, vec![2]);
        assert_eq!(j, vec![1]);
    }

    #[test]
    fn points_base_points_cap_points() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        s.add_arrow(2, 3, ());
        let mut all: Vec<_> = s.points().collect();
        all.sort();
        assert_eq!(all, vec![1, 2, 3]);
        let mut base: Vec<_> = s.base_points().collect();
        base.sort();
        assert_eq!(base, vec![1, 2]);
        let mut cap: Vec<_> = s.cap_points().collect();
        cap.sort();
        assert_eq!(cap, vec![2, 3]);
    }
}

#[cfg(test)]
mod covering_api_tests {
    use super::InMemorySieve;
    use crate::topology::sieve::Sieve;

    #[test]
    fn add_and_remove_point() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_point(1);
        assert!(s.points().any(|p| p == 1));
        s.remove_point(1);
        assert!(s.points().any(|p| p == 1));
        assert!(s.cone(1).next().is_none());
        assert!(s.support(1).next().is_none());
    }

    #[test]
    fn add_and_remove_base_cap_point() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_base_point(2);
        assert!(s.base_points().any(|p| p == 2));
        s.remove_base_point(2);
        assert!(!s.base_points().any(|p| p == 2));
        s.add_cap_point(3);
        assert!(s.cap_points().any(|p| p == 3));
        s.remove_cap_point(3);
        assert!(!s.cap_points().any(|p| p == 3));
    }

    #[test]
    fn set_and_add_cone() {
        let mut s = InMemorySieve::<u32, i32>::new();
        s.set_cone(1, vec![(2, 10), (3, 20)]);
        let mut cone: Vec<_> = s.cone(1).collect();
        cone.sort_by_key(|(d, _)| *d);
        assert_eq!(cone, vec![(2, 10), (3, 20)]);
        s.add_cone(1, vec![(4, 30)]);
        let mut cone: Vec<_> = s.cone(1).collect();
        cone.sort_by_key(|(d, _)| *d);
        assert_eq!(cone, vec![(2, 10), (3, 20), (4, 30)]);
    }

    #[test]
    fn set_and_add_support() {
        let mut s = InMemorySieve::<u32, i32>::new();
        s.set_support(5, vec![(1, 100), (2, 200)]);
        let mut support: Vec<_> = s.support(5).collect();
        support.sort_by_key(|(src, _)| *src);
        assert_eq!(support, vec![(1, 100), (2, 200)]);
        s.add_support(5, vec![(3, 300)]);
        let mut support: Vec<_> = s.support(5).collect();
        support.sort_by_key(|(src, _)| *src);
        assert_eq!(support, vec![(1, 100), (2, 200), (3, 300)]);
    }

    #[test]
    fn restrict_base_and_cap() {
        let mut s = InMemorySieve::<u32, i32>::new();
        s.add_arrow(1, 2, 10);
        s.add_arrow(1, 3, 20);
        s.add_arrow(4, 5, 30);
        let base = s.restrict_base(vec![1]);
        let mut cone: Vec<_> = base.cone(1).collect();
        cone.sort_by_key(|(d, _)| *d);
        assert_eq!(cone, vec![(2, 10), (3, 20)]);
        assert!(base.cone(4).next().is_none());
        let cap = s.restrict_cap(vec![2, 3]);
        let mut support2: Vec<_> = cap.support(2).collect();
        support2.sort_by_key(|(src, _)| *src);
        assert_eq!(support2, vec![(1, 10)]);
        let mut support3: Vec<_> = cap.support(3).collect();
        support3.sort_by_key(|(src, _)| *src);
        assert_eq!(support3, vec![(1, 20)]);
        assert!(cap.support(5).next().is_none());
    }

    #[test]
    fn cache_invalidation_on_mutation() {
        let mut s = InMemorySieve::<u32, ()>::new();
        s.add_arrow(1, 2, ());
        let d0 = s.diameter().unwrap();
        s.add_point(3);
        let d1 = s.diameter().unwrap();
        assert!(d1 <= d0 + 1);
        s.remove_point(1);
        let _ = s.diameter(); // should not panic
    }
}

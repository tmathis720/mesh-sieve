//! In-memory implementation of the [`Sieve`] trait.
//!
//! This module provides [`InMemorySieve`], a simple and efficient in-memory representation
//! of a sieve using hash maps for adjacency storage. It supports generic point and payload types.

use super::mutable::MutableSieve;
use super::sieve_ref::SieveRef;
use super::sieve_trait::Sieve;
use crate::mesh_error::MeshSieveError;
use crate::topology::cache::InvalidateCache;
use crate::topology::sieve::strata::{compute_strata, StrataCache};
use once_cell::sync::OnceCell;
use std::collections::HashMap;

/// An in-memory sieve implementation using hash maps for adjacency storage.
///
/// # Type Parameters
/// - `P`: The type of points in the sieve. Must implement `Ord`.
/// - `T`: The type of payloads associated with arrows. Defaults to `()`.
#[derive(Clone, Debug)]
pub struct InMemorySieve<P, T = ()>
where
    P: Ord + std::fmt::Debug,
{
    /// Outgoing adjacency: maps each point to a vector of (destination, payload) pairs.
    pub adjacency_out: HashMap<P, Vec<(P, T)>>,
    /// Incoming adjacency: maps each point to a vector of (source, payload) pairs.
    pub adjacency_in: HashMap<P, Vec<(P, T)>>,
    /// Cached strata information for the sieve.
    pub strata: OnceCell<StrataCache<P>>,
}

impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T> Default for InMemorySieve<P, T> {
    fn default() -> Self {
        Self {
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            strata: OnceCell::new(),
        }
    }
}

impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T: Clone> InMemorySieve<P, T> {
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

    #[cfg(debug_assertions)]
    pub fn debug_assert_consistent(&self) {
        for (src, outs) in &self.adjacency_out {
            for (dst, _) in outs {
                let ok = self
                    .adjacency_in
                    .get(dst)
                    .map_or(false, |ins| ins.iter().any(|(s, _)| s == src));
                debug_assert!(
                    ok,
                    "Missing mirror in[{dst:?}] for out edge ({src:?} -> {dst:?})"
                );
            }
        }
        for (dst, ins) in &self.adjacency_in {
            for (src, _) in ins {
                let ok = self
                    .adjacency_out
                    .get(src)
                    .map_or(false, |outs| outs.iter().any(|(d, _)| d == dst));
                debug_assert!(
                    ok,
                    "Missing mirror out[{src:?}] for in edge ({src:?} -> {dst:?})"
                );
            }
        }
    }

    #[inline]
    pub fn has_arrow(&self, src: P, dst: P) -> bool {
        self.adjacency_out
            .get(&src)
            .map_or(false, |v| v.iter().any(|(d, _)| *d == dst))
    }

    #[cfg(debug_assertions)]
    pub fn debug_assert_no_parallel_edges_src(&self, src: P) {
        if let Some(v) = self.adjacency_out.get(&src) {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (dst, _) in v {
                assert!(
                    seen.insert(*dst),
                    "duplicate edges out of {:?} to {:?}",
                    src,
                    dst
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    pub fn debug_assert_no_parallel_edges_dst(&self, dst: P) {
        if let Some(v) = self.adjacency_in.get(&dst) {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (src, _) in v {
                assert!(
                    seen.insert(*src),
                    "duplicate edges into {:?} from {:?}",
                    dst,
                    src
                );
            }
        }
    }
}

impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T: Clone> InvalidateCache
    for InMemorySieve<P, T>
{
    #[inline]
    fn invalidate_cache(&mut self) {
        self.strata.take();
    }
}

type ConeMapIter<'a, P, T> = std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, T)>;
type ConeRefMapIter<'a, P, T> =
    std::iter::Map<std::slice::Iter<'a, (P, T)>, fn(&'a (P, T)) -> (P, &'a T)>;

impl<P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug, T: Clone> Sieve
    for InMemorySieve<P, T>
{
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
    /// This method updates the outgoing adjacency of `src` and the incoming adjacency of `dst`.
    /// It also invalidates the cache for strata information.
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

        #[cfg(debug_assertions)]
        {
            self.debug_assert_consistent();
            self.debug_assert_no_parallel_edges_src(src);
            self.debug_assert_no_parallel_edges_dst(dst);
        }
    }

    /// Removes the arrow from `src` to `dst`, returning the associated payload if it existed.
    ///
    /// This method updates both the outgoing adjacency of `src` and the incoming adjacency of `dst`.
    /// It also invalidates the cache for strata information.
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
        if let Some(v) = self.adjacency_out.get_mut(&src) {
            if let Some(pos) = v.iter().position(|(d, _)| *d == dst) {
                removed = Some(v.remove(pos).1);
            }
        }
        if let Some(v) = self.adjacency_in.get_mut(&dst) {
            if let Some(pos) = v.iter().position(|(s, _)| *s == src) {
                v.remove(pos);
            }
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
        removed
    }

    // strata helpers now provided by Sieve trait default impls
    /// Adds a point to the sieve, creating empty adjacencies for it.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_point(1);
    /// assert!(s.adjacency_out.contains_key(&1));
    /// assert!(s.adjacency_in.contains_key(&1));
    /// ```
    /// Adds a base point to the sieve, creating an empty outgoing adjacency for it.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_base_point(1);
    /// assert!(s.adjacency_out.contains_key(&1));
    /// ```
    /// Sets the cone for a point, replacing any existing cone.
    ///
    /// This method takes ownership of the provided iterator, consuming it.
    /// Restricts the sieve to only include arrows originating from the given chain of base points.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2, ());
    /// s.add_arrow(3, 4, ());
    /// let restricted = s.restrict_base(vec![1, 3]);
    /// assert_eq!(restricted.cone(1).count() + restricted.cone(3).count(), 2);
    /// ```
    fn restrict_base(&self, chain: impl IntoIterator<Item = P>) -> Self {
        let mut out = Self::default();
        for p in chain {
            if let Some(outs) = self.adjacency_out.get(&p) {
                for (d, pay) in outs {
                    out.add_arrow(p, *d, pay.clone());
                }
            }
        }
        out
    }
    /// Restricts the sieve to only include arrows ending at the given chain of cap points.
    ///
    /// # Example
    /// ```rust
    /// use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
    /// use mesh_sieve::topology::sieve::Sieve;
    /// let mut s = InMemorySieve::<u32, ()>::new();
    /// s.add_arrow(1, 2, ());
    /// s.add_arrow(3, 4, ());
    /// let restricted = s.restrict_cap(vec![2, 4]);
    /// assert_eq!(restricted.support(2).count() + restricted.support(4).count(), 2);
    /// ```
    fn restrict_cap(&self, chain: impl IntoIterator<Item = P>) -> Self {
        let mut out = Self::default();
        for q in chain {
            if let Some(ins) = self.adjacency_in.get(&q) {
                for (src, pay) in ins {
                    out.add_arrow(*src, q, pay.clone());
                }
            }
        }
        out
    }
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

impl<P, T> MutableSieve for InMemorySieve<P, T>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
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
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn remove_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
        }
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn add_base_point(&mut self, p: P) {
        self.adjacency_out.entry(p).or_default();
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn add_cap_point(&mut self, p: P) {
        self.adjacency_in.entry(p).or_default();
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn remove_base_point(&mut self, p: P) {
        if self.adjacency_out.contains_key(&p) {
            self.scrub_outgoing_only(p);
            self.adjacency_out.remove(&p);
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    fn remove_cap_point(&mut self, p: P) {
        if self.adjacency_in.contains_key(&p) {
            self.scrub_incoming_only(p);
            self.adjacency_in.remove(&p);
        }
        self.invalidate_cache();
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
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

        // Remove mirrors of old cone
        let old_cone: Vec<(P, T)> = std::mem::take(self.adjacency_out.entry(p).or_default());
        for (dst, _) in &old_cone {
            if let Some(ins) = self.adjacency_in.get_mut(dst) {
                ins.retain(|(src, _)| *src != p);
            }
        }

        // Install new cone and mirrors
        self.adjacency_out.insert(p, new_cone.clone());
        for (dst, pay) in new_cone {
            let ins = self.adjacency_in.entry(dst).or_default();
            if let Some(slot) = ins.iter_mut().find(|(s, _)| *s == p) {
                slot.1 = pay;
            } else {
                ins.push((p, pay));
            }
        }

        self.invalidate_cache();

        #[cfg(debug_assertions)]
        {
            self.debug_assert_consistent();
            self.debug_assert_no_parallel_edges_src(p);
        }
    }

    fn add_cone(&mut self, p: P, chain: impl IntoIterator<Item = (P, T)>) {
        for (dst, pay) in chain {
            self.add_arrow(p, dst, pay);
        }
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

        // Remove mirrors of old support
        let old_sup: Vec<(P, T)> = std::mem::take(self.adjacency_in.entry(q).or_default());
        for (src, _) in &old_sup {
            if let Some(outs) = self.adjacency_out.get_mut(src) {
                outs.retain(|(dst, _)| *dst != q);
            }
        }

        // Install new support and mirrors
        self.adjacency_in.insert(q, new_sup.clone());
        for (src, pay) in new_sup {
            let outs = self.adjacency_out.entry(src).or_default();
            if let Some(slot) = outs.iter_mut().find(|(d, _)| *d == q) {
                slot.1 = pay;
            } else {
                outs.push((q, pay));
            }
        }

        self.invalidate_cache();

        #[cfg(debug_assertions)]
        {
            self.debug_assert_consistent();
            self.debug_assert_no_parallel_edges_dst(q);
        }
    }

    fn add_support(&mut self, q: P, chain: impl IntoIterator<Item = (P, T)>) {
        for (src, pay) in chain {
            self.add_arrow(src, q, pay);
        }
    }
}

impl<P, T> SieveRef for InMemorySieve<P, T>
where
    P: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
    T: Clone,
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
